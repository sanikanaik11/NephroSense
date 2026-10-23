"""
screening_db.py — NephroSense (dual-backend)
============================================
Storage for Home Screening + consent records.

All tables (screenings, screening_consents) are created by database.py's
create_tables() so both SQLite and PostgreSQL schemas are handled there.
This module only performs DML using the shared _exec / _fetchall / _fetchone
/ _ph / _ph1 / _insert_id helpers — no SQLite-specific constructs here.
"""
import json
from datetime import datetime

from database import (
    _exec, _fetchall, _fetchone, _insert_id,
    _ph, _ph1,
    _encrypt, _decrypt,
)

CONSENT_VERSION = "1"


# ── consent ───────────────────────────────────────────────────────────────────

def has_consent(user_id):
    ph = _ph1()
    row = _fetchone(
        f"SELECT version FROM screening_consents WHERE user_id={ph}",
        (user_id,),
    )
    return bool(row) and row["version"] == CONSENT_VERSION


def record_consent(user_id):
    """Upsert: insert if no record exists, update if one already does."""
    ph = _ph1()
    now = datetime.now().isoformat(timespec="seconds")
    existing = _fetchone(
        f"SELECT user_id FROM screening_consents WHERE user_id={ph}",
        (user_id,),
    )
    if existing:
        _exec(
            f"UPDATE screening_consents SET version={ph}, accepted_at={ph} WHERE user_id={ph}",
            (CONSENT_VERSION, now, user_id),
        )
    else:
        _exec(
            f"INSERT INTO screening_consents (user_id, version, accepted_at) VALUES ({_ph(3)})",
            (user_id, CONSENT_VERSION, now),
        )


# ── screenings ────────────────────────────────────────────────────────────────

def save_screening(user_id, band, payload, person_label=""):
    """
    Save one screening record.
    person_label and payload are stored Fernet-encrypted.
    payload must be a JSON-serialisable dict.
    """
    sql = f"INSERT INTO screenings (user_id, person_label, band, payload) VALUES ({_ph(4)})"
    _insert_id(sql, (
        user_id,
        _encrypt(person_label),
        band,
        _encrypt(json.dumps(payload)),
    ))


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
        # normalise created_at: PostgreSQL returns datetime objects, SQLite returns strings
        ca = d.get("created_at")
        d["created_at"] = ca.isoformat(timespec="seconds") if hasattr(ca, "isoformat") else str(ca or "")
        try:
            d["payload"] = json.loads(_decrypt(d.get("payload"), default="{}") or "{}")
        except (ValueError, TypeError):
            d["payload"] = {}
        out.append(d)
    return out


def last_self_screening_date(user_id):
    """
    Date of the user's own most recent screening.
    Health-worker screenings of other people have a non-empty person_label and are excluded.
    """
    for r in list_screenings(user_id, limit=50):
        if not r["person_label"]:
            try:
                return datetime.strptime(str(r.get("created_at", ""))[:10], "%Y-%m-%d").date()
            except ValueError:
                return None
    return None


def delete_screening_data(user_id):
    """Delete all screening records for this user (GDPR / data deletion)."""
    ph = _ph1()
    _exec(f"DELETE FROM screenings WHERE user_id={ph}", (user_id,))
