"""
screening_db.py — NephroSense (dual-backend)
=============================================
Storage for Home Screening + consent records.
Separate tables from the main assessments — database.py data is untouched.
Sensitive content is encrypted with the same Fernet key as database.py.

Backend is chosen automatically:
  DATABASE_URL set  → PostgreSQL (Supabase)
  DATABASE_URL unset → SQLite (local dev, same as before)

All helpers (_exec, _fetchall, _fetchone, _insert_returning_id, _ph, _ph1,
_USE_PG, _encrypt, _decrypt) are imported from database.py so there is a
single source of truth for dialect switching.
"""
import json
from datetime import datetime

from database import (
    _USE_PG,
    _exec,
    _fetchall,
    _fetchone,
    _insert_returning_id,
    _ph,
    _ph1,
    _encrypt,
    _decrypt,
)

CONSENT_VERSION = "1"


# ── schema ────────────────────────────────────────────────────────────────────

def create_tables():
    if _USE_PG:
        _create_tables_pg()
    else:
        _create_tables_sqlite()


def _create_tables_pg():
    _exec("""
        CREATE TABLE IF NOT EXISTS screenings (
            id           INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            user_id      INTEGER NOT NULL REFERENCES users(id),
            person_label TEXT,
            band         TEXT,
            payload      TEXT,
            created_at   TIMESTAMP DEFAULT NOW()
        )
    """)
    _exec("""
        CREATE TABLE IF NOT EXISTS consents (
            user_id      INTEGER PRIMARY KEY REFERENCES users(id),
            version      TEXT,
            accepted_at  TEXT
        )
    """)


def _create_tables_sqlite():
    from database import get_conn  # SQLite-only path — safe here
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS screenings (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            person_label TEXT,
            band         TEXT,
            payload      TEXT,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS consents (
            user_id     INTEGER PRIMARY KEY,
            version     TEXT,
            accepted_at TEXT
        )
    """)
    conn.commit()

    # migrate: add accepted_at if the table was created by an older version
    c.execute("PRAGMA table_info(consents)")
    if "accepted_at" not in {row[1] for row in c.fetchall()}:
        c.execute("ALTER TABLE consents ADD COLUMN accepted_at TEXT")
        conn.commit()

    conn.close()


# ── consent ───────────────────────────────────────────────────────────────────

def has_consent(user_id):
    ph = _ph1()
    row = _fetchone(f"SELECT version FROM consents WHERE user_id={ph}", (user_id,))
    return bool(row) and row["version"] == CONSENT_VERSION


def record_consent(user_id):
    """Insert or update the consent record for this user."""
    ph = _ph1()
    now = datetime.now().isoformat(timespec="seconds")

    # Check whether a row already exists
    existing = _fetchone(f"SELECT user_id FROM consents WHERE user_id={ph}", (user_id,))

    if existing:
        # UPDATE existing row
        _exec(
            f"UPDATE consents SET version={ph}, accepted_at={ph} WHERE user_id={ph}",
            (CONSENT_VERSION, now, user_id),
        )
    else:
        # INSERT new row
        # Note: SQLite would use INSERT OR REPLACE; PG uses a plain INSERT.
        # We pre-check existence above so a plain INSERT is safe on both backends.
        _exec(
            f"INSERT INTO consents (user_id, version, accepted_at) VALUES ({_ph(3)})",
            (user_id, CONSENT_VERSION, now),
        )


# ── screenings ────────────────────────────────────────────────────────────────

def save_screening(user_id, band, payload, person_label=""):
    """
    Save one screening record.
    person_label and payload are stored Fernet-encrypted.
    payload must be a JSON-serialisable dict.
    """
    _insert_returning_id(
        f"INSERT INTO screenings (user_id, person_label, band, payload) VALUES ({_ph(4)})",
        (user_id, _encrypt(person_label), band, _encrypt(json.dumps(payload))),
    )


def list_screenings(user_id, limit=10):
    """
    Return the most recent `limit` screening records for this user,
    with person_label and payload decrypted.
    created_at is normalised to a string (handles both TEXT and TIMESTAMP columns).
    """
    ph = _ph1()
    rows = _fetchall(
        f"SELECT * FROM screenings WHERE user_id={ph} ORDER BY id DESC LIMIT {int(limit)}",
        (user_id,),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["person_label"] = _decrypt(d.get("person_label"))
        # Normalise created_at: PostgreSQL returns a datetime object, SQLite returns a string
        ca = d.get("created_at")
        if hasattr(ca, "isoformat"):
            d["created_at"] = ca.isoformat(timespec="seconds")
        else:
            d["created_at"] = str(ca or "")
        try:
            d["payload"] = json.loads(_decrypt(d.get("payload"), default="{}") or "{}")
        except (ValueError, TypeError):
            d["payload"] = {}
        out.append(d)
    return out


def last_self_screening_date(user_id):
    """
    Date of the user's own most recent screening
    (health-worker screenings of other people are excluded — they have a non-empty person_label).
    """
    for r in list_screenings(user_id, limit=50):
        if not r["person_label"]:
            ca = r.get("created_at", "")
            try:
                return datetime.strptime(str(ca)[:10], "%Y-%m-%d").date()
            except ValueError:
                return None
    return None


def delete_screening_data(user_id):
    """Delete all screening records for this user (GDPR / data-deletion request)."""
    ph = _ph1()
    _exec(f"DELETE FROM screenings WHERE user_id={ph}", (user_id,))


# ── run schema creation on import ─────────────────────────────────────────────
create_tables()
