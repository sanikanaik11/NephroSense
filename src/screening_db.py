"""
screening_db.py — NephroSense
Storage for Home Screening + consent. Separate tables, so database.py and your
existing data are untouched. Sensitive content is encrypted with the same Fernet
key database.py already uses (database/enc.key).
"""
import json
from datetime import datetime

from database import get_conn, _encrypt, _decrypt

CONSENT_VERSION = "1"


def create_tables():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS screenings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        person_label TEXT,
        band TEXT,
        payload TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS consents (
        user_id INTEGER PRIMARY KEY,
        version TEXT,
        accepted_at TEXT)""")
    conn.commit()
    conn.close()


def has_consent(user_id):
    conn = get_conn()
    row = conn.execute("SELECT version FROM consents WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return bool(row) and row["version"] == CONSENT_VERSION


def record_consent(user_id):
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO consents (user_id, version, accepted_at) VALUES (?,?,?)",
                 (user_id, CONSENT_VERSION, datetime.now().isoformat(timespec="seconds")))
    conn.commit()
    conn.close()


def save_screening(user_id, band, payload, person_label=""):
    """payload (inputs, reasons, bp summary) and person_label are stored encrypted."""
    conn = get_conn()
    conn.execute("INSERT INTO screenings (user_id, person_label, band, payload) VALUES (?,?,?,?)",
                 (user_id, _encrypt(person_label), band, _encrypt(json.dumps(payload))))
    conn.commit()
    conn.close()


def list_screenings(user_id, limit=10):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM screenings WHERE user_id=? ORDER BY id DESC LIMIT ?",
                        (user_id, limit)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["person_label"] = _decrypt(d.get("person_label"))
        try:
            d["payload"] = json.loads(_decrypt(d.get("payload"), default="{}") or "{}")
        except (ValueError, TypeError):
            d["payload"] = {}
        out.append(d)
    return out


def last_self_screening_date(user_id):
    """Date of the user's own latest screening (health-worker screenings of others don't count)."""
    for r in list_screenings(user_id, limit=50):
        if not r["person_label"]:
            try:
                return datetime.strptime(r["created_at"][:10], "%Y-%m-%d").date()
            except ValueError:
                return None
    return None


def delete_screening_data(user_id):
    conn = get_conn()
    conn.execute("DELETE FROM screenings WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


create_tables()
