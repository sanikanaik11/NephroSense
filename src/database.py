"""
database.py — NephroSense v3 (dual-backend: SQLite locally, PostgreSQL on Streamlit Cloud)
==========================================================================================

Backend selection
-----------------
If  st.secrets["DATABASE_URL"]  (or os.environ["DATABASE_URL"])  is set  →  PostgreSQL.
Otherwise  →  SQLite  (database/nephrosense.db, same as before, zero config change).

Encryption key
--------------
Cloud:  ENC_KEY from st.secrets / os.environ  (base64 Fernet key string).
Local:  database/enc.key  (auto-generated on first run, never committed to git).

All public function signatures are UNCHANGED — no callers need editing.

Needs: pip install bcrypt cryptography psycopg2-binary
"""
import sqlite3
import os
import json
from contextlib import contextmanager
from datetime import datetime

import bcrypt
from cryptography.fernet import Fernet, InvalidToken

from age_utils import calculate_age

# ── secret helper (safe before Streamlit is initialised) ─────────────────────
def _secret(key, default=None):
    try:
        import streamlit as st
        v = st.secrets.get(key)
        if v is not None:
            return str(v)
    except Exception:
        pass
    return os.environ.get(key, default)


# ── backend selection ─────────────────────────────────────────────────────────
DATABASE_URL = _secret("DATABASE_URL")
_USE_PG      = bool(DATABASE_URL)

# ── local paths (SQLite only) ─────────────────────────────────────────────────
_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(_ROOT, "database", "nephrosense.db")
KEY_PATH = os.path.join(_ROOT, "database", "enc.key")

# ── Fernet encryption ─────────────────────────────────────────────────────────
def _get_fernet():
    enc_key = _secret("ENC_KEY")
    if enc_key:
        raw = enc_key.encode() if isinstance(enc_key, str) else enc_key
        return Fernet(raw)
    # local: read or create key file
    os.makedirs(os.path.dirname(KEY_PATH), exist_ok=True)
    if not os.path.exists(KEY_PATH):
        with open(KEY_PATH, "wb") as f:
            f.write(Fernet.generate_key())
    with open(KEY_PATH, "rb") as f:
        return Fernet(f.read())


_fernet = _get_fernet()

ENCRYPTED_HEALTH_FIELDS = [
    "diabetes", "hypertension", "heart_disease", "kidney_history",
    "uti_history", "blood_group",
]


def _encrypt(value):
    if value is None or value == "":
        return ""
    return _fernet.encrypt(str(value).encode()).decode()


def _decrypt(value, default=""):
    if value is None or value == "":
        return default
    try:
        return _fernet.decrypt(str(value).encode()).decode()
    except (InvalidToken, ValueError):
        return value   # legacy plaintext row


# ── password helpers ──────────────────────────────────────────────────────────
def hash_pw(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def _is_bcrypt_hash(stored):
    return stored.startswith(("$2a$", "$2b$", "$2y$"))


def verify_pw(pw, stored_hash):
    if _is_bcrypt_hash(stored_hash):
        try:
            return bcrypt.checkpw(pw.encode(), stored_hash.encode())
        except ValueError:
            return False
    import hashlib
    return hashlib.sha256(pw.encode()).hexdigest() == stored_hash


# ── connection helpers ────────────────────────────────────────────────────────
def get_conn():
    """SQLite connection — kept for backward compatibility with export scripts."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _pg_conn():
    """Yield a psycopg2 connection with RealDictCursor; auto-commit / rollback."""
    import psycopg2
    import psycopg2.extras
    conn = psycopg2.connect(DATABASE_URL,
                             cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── SQL dialect helpers ───────────────────────────────────────────────────────
def _ph(n=1):
    """n comma-separated placeholders: '?,?' (SQLite) or '%s,%s' (PG)."""
    p = "%s" if _USE_PG else "?"
    return ", ".join([p] * n)


def _ph1():
    return "%s" if _USE_PG else "?"


# ── generic execute helpers ───────────────────────────────────────────────────
def _exec(sql, params=()):
    """Run a single DML/DDL statement, no return value."""
    if _USE_PG:
        with _pg_conn() as conn:
            conn.cursor().execute(sql, params)
    else:
        c = get_conn()
        c.execute(sql, params)
        c.commit()
        c.close()


def _fetchall(sql, params=()):
    if _USE_PG:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    else:
        conn = get_conn()
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]


def _fetchone(sql, params=()):
    if _USE_PG:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
    else:
        conn = get_conn()
        row = conn.execute(sql, params).fetchone()
        conn.close()
        return dict(row) if row else None


def _insert_id(sql, params=()):
    """INSERT and return the new row's id."""
    if _USE_PG:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql + " RETURNING id", params)
            return cur.fetchone()["id"]
    else:
        conn = get_conn()
        cur = conn.execute(sql, params)
        conn.commit()
        new_id = cur.lastrowid
        conn.close()
        return new_id


# ── schema creation ───────────────────────────────────────────────────────────
CONSENT_VERSION = "2026-10-v1"


def create_tables():
    if _USE_PG:
        _create_tables_pg()
    else:
        _create_tables_sqlite()


def _create_tables_pg():
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        id             SERIAL PRIMARY KEY,
        full_name      TEXT NOT NULL,
        email          TEXT UNIQUE NOT NULL,
        password       TEXT NOT NULL,
        gender         TEXT NOT NULL,
        age            INTEGER NOT NULL,
        role           TEXT DEFAULT 'patient',
        phone          TEXT,
        dob            TEXT,
        blood_group    TEXT,
        city           TEXT,
        diabetes       TEXT DEFAULT '',
        hypertension   TEXT DEFAULT '',
        heart_disease  TEXT DEFAULT '',
        kidney_history TEXT DEFAULT '',
        uti_history    TEXT DEFAULT '',
        smoking        TEXT,
        water_intake   TEXT,
        extra_data     TEXT,
        last_login     TEXT,
        created_at     TIMESTAMP DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS assessments (
        id                SERIAL PRIMARY KEY,
        user_id           INTEGER NOT NULL REFERENCES users(id),
        risk_score        REAL,
        ckd_stage         INTEGER,
        risk_level        TEXT,
        lifestyle_score   REAL,
        eye_score         REAL,
        voice_score       REAL,
        clinical_notes    TEXT,
        lifestyle_answers TEXT,
        egfr              REAL,
        egfr_stage        TEXT,
        created_at        TIMESTAMP DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS consents (
        id         SERIAL PRIMARY KEY,
        user_id    INTEGER NOT NULL,
        version    TEXT NOT NULL,
        language   TEXT,
        created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS bp_readings (
        id          SERIAL PRIMARY KEY,
        user_id     INTEGER NOT NULL,
        subject     TEXT NOT NULL,
        day_idx     INTEGER NOT NULL,
        period      TEXT NOT NULL,
        reading_enc TEXT NOT NULL,
        taken_at    TEXT
    );
    CREATE TABLE IF NOT EXISTS home_screens (
        id            SERIAL PRIMARY KEY,
        user_id       INTEGER NOT NULL,
        subject       TEXT NOT NULL,
        band          TEXT,
        payload_enc   TEXT,
        model_version TEXT,
        created_at    TEXT
    );
    CREATE TABLE IF NOT EXISTS screenings (
        id           SERIAL PRIMARY KEY,
        user_id      INTEGER NOT NULL,
        person_label TEXT,
        band         TEXT,
        payload      TEXT,
        created_at   TIMESTAMP DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS screening_consents (
        user_id     INTEGER PRIMARY KEY,
        version     TEXT,
        accepted_at TEXT
    );
    """
    with _pg_conn() as conn:
        conn.cursor().execute(ddl)


def _create_tables_sqlite():
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name      TEXT NOT NULL,
            email          TEXT UNIQUE NOT NULL,
            password       TEXT NOT NULL,
            gender         TEXT NOT NULL,
            age            INTEGER NOT NULL,
            role           TEXT DEFAULT 'patient',
            phone          TEXT,
            dob            TEXT,
            blood_group    TEXT,
            city           TEXT,
            diabetes       TEXT DEFAULT '',
            hypertension   TEXT DEFAULT '',
            heart_disease  TEXT DEFAULT '',
            kidney_history TEXT DEFAULT '',
            uti_history    TEXT DEFAULT '',
            smoking        TEXT,
            water_intake   TEXT,
            extra_data     TEXT,
            last_login     TEXT,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS assessments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            risk_score      REAL,
            ckd_stage       INTEGER,
            risk_level      TEXT,
            lifestyle_score REAL,
            eye_score       REAL,
            voice_score     REAL,
            clinical_notes  TEXT,
            lifestyle_answers TEXT,
            egfr            REAL,
            egfr_stage      TEXT,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS consents (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            version    TEXT NOT NULL,
            language   TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS bp_readings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            subject     TEXT NOT NULL,
            day_idx     INTEGER NOT NULL,
            period      TEXT NOT NULL,
            reading_enc TEXT NOT NULL,
            taken_at    TEXT
        );
        CREATE TABLE IF NOT EXISTS home_screens (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            subject       TEXT NOT NULL,
            band          TEXT,
            payload_enc   TEXT,
            model_version TEXT,
            created_at    TEXT
        );
        CREATE TABLE IF NOT EXISTS screenings (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            person_label TEXT,
            band         TEXT,
            payload      TEXT,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS screening_consents (
            user_id     INTEGER PRIMARY KEY,
            version     TEXT,
            accepted_at TEXT
        );
    """)

    # migrate: add any missing columns to existing tables
    c.execute("PRAGMA table_info(users)")
    existing = {row[1] for row in c.fetchall()}
    for col, defn in {
        "role": "TEXT DEFAULT 'patient'", "phone": "TEXT", "dob": "TEXT",
        "blood_group": "TEXT", "city": "TEXT", "diabetes": "TEXT DEFAULT ''",
        "hypertension": "TEXT DEFAULT ''", "heart_disease": "TEXT DEFAULT ''",
        "kidney_history": "TEXT DEFAULT ''", "uti_history": "TEXT DEFAULT ''",
        "smoking": "TEXT", "water_intake": "TEXT", "extra_data": "TEXT", "last_login": "TEXT",
    }.items():
        if col not in existing:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {defn}")

    c.execute("PRAGMA table_info(assessments)")
    existing = {row[1] for row in c.fetchall()}
    for col, defn in {
        "risk_score": "REAL", "ckd_stage": "INTEGER", "risk_level": "TEXT",
        "lifestyle_score": "REAL", "eye_score": "REAL", "voice_score": "REAL",
        "clinical_notes": "TEXT", "lifestyle_answers": "TEXT",
        "egfr": "REAL", "egfr_stage": "TEXT",
    }.items():
        if col not in existing:
            c.execute(f"ALTER TABLE assessments ADD COLUMN {col} {defn}")

    # migrate screening_consents: add accepted_at if missing
    c.execute("PRAGMA table_info(screening_consents)")
    existing = {row[1] for row in c.fetchall()}
    if "accepted_at" not in existing:
        c.execute("ALTER TABLE screening_consents ADD COLUMN accepted_at TEXT")

    conn.commit()
    conn.close()


# ── public API ────────────────────────────────────────────────────────────────
def register_user(
    full_name, email, password, gender, age,
    role="patient", phone="", dob="", blood_group="", city="",
    diabetes=False, hypertension=False, heart_disease=False,
    kidney_history=False, uti_history=False,
    smoking="Non-smoker", water_intake="1-2 litres", extra_data="",
):
    ph = _ph1()
    if _fetchone(f"SELECT id FROM users WHERE email={ph}", (email,)):
        return False, "This email is already registered. Please sign in."

    sql = f"""
        INSERT INTO users
        (full_name, email, password, gender, age, role, phone, dob, blood_group,
         city, diabetes, hypertension, heart_disease, kidney_history, uti_history,
         smoking, water_intake, extra_data)
        VALUES ({_ph(18)})
    """
    _insert_id(sql, (
        full_name, email, hash_pw(password), gender, age, role, phone, dob,
        _encrypt(blood_group), city,
        _encrypt(int(diabetes)), _encrypt(int(hypertension)),
        _encrypt(int(heart_disease)), _encrypt(int(kidney_history)),
        _encrypt(int(uti_history)),
        smoking, water_intake, extra_data,
    ))
    return True, f"Account created! Welcome, {full_name.split()[0]}."


def login_user(email, password):
    ph = _ph1()
    row = _fetchone(
        f"""SELECT id, full_name, email, password, gender, age, role, phone, city, dob,
               blood_group, diabetes, hypertension, heart_disease,
               kidney_history, extra_data, last_login
            FROM users WHERE email={ph}""",
        (email,),
    )
    if not row or not verify_pw(password, row["password"]):
        return False, "Incorrect email or password."

    user = dict(row)
    previous_last_login = user.get("last_login")

    # recalculate age from DOB so it never goes stale
    _age = calculate_age(user.get("dob"))
    if _age is not None and _age != user.get("age"):
        _exec(f"UPDATE users SET age={ph} WHERE id={ph}", (_age, row["id"]))
        user["age"] = _age

    # upgrade legacy SHA-256 hash to bcrypt silently
    if not _is_bcrypt_hash(row["password"]):
        _exec(f"UPDATE users SET password={ph} WHERE id={ph}", (hash_pw(password), row["id"]))

    now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    _exec(f"UPDATE users SET last_login={ph} WHERE id={ph}", (now_str, row["id"]))

    user.pop("password", None)
    user["blood_group"] = _decrypt(user.get("blood_group"))
    for f in ("diabetes", "hypertension", "heart_disease", "kidney_history"):
        user[f] = int(_decrypt(user.get(f), default="0") or 0)
    user["last_login"] = previous_last_login or now_str
    return True, user


def get_user_history(user_id):
    ph = _ph1()
    return _fetchall(
        f"SELECT * FROM assessments WHERE user_id={ph} ORDER BY created_at DESC, id DESC",
        (user_id,),
    )


def get_latest_component_scores(user_id):
    ph = _ph1()
    result = {}
    for col in ("lifestyle_score", "eye_score", "voice_score"):
        row = _fetchone(
            f"SELECT {col} FROM assessments "
            f"WHERE user_id={ph} AND {col} IS NOT NULL "
            f"ORDER BY created_at DESC, id DESC LIMIT 1",
            (user_id,),
        )
        result[col] = row[col] if row else None
    return result


def get_latest_clinical_notes(user_id):
    ph = _ph1()
    row = _fetchone(
        f"SELECT clinical_notes FROM assessments "
        f"WHERE user_id={ph} AND clinical_notes IS NOT NULL AND clinical_notes != '' "
        f"ORDER BY created_at DESC, id DESC LIMIT 1",
        (user_id,),
    )
    return row["clinical_notes"] if row else None


def get_latest_lifestyle_answers(user_id):
    ph = _ph1()
    row = _fetchone(
        f"SELECT lifestyle_answers FROM assessments "
        f"WHERE user_id={ph} AND lifestyle_answers IS NOT NULL AND lifestyle_answers != '' "
        f"ORDER BY created_at DESC, id DESC LIMIT 1",
        (user_id,),
    )
    return row["lifestyle_answers"] if row else None


def get_latest_egfr(user_id):
    ph = _ph1()
    row = _fetchone(
        f"SELECT egfr, egfr_stage FROM assessments "
        f"WHERE user_id={ph} AND egfr IS NOT NULL "
        f"ORDER BY created_at DESC, id DESC LIMIT 1",
        (user_id,),
    )
    return dict(row) if row else None


def save_assessment(
    user_id, risk_score, ckd_stage, risk_level,
    lifestyle_score=None, eye_score=None, voice_score=None,
    clinical_notes=None, egfr=None, egfr_stage=None, lifestyle_answers=None,
):
    sql = f"""
        INSERT INTO assessments
        (user_id, risk_score, ckd_stage, risk_level,
         lifestyle_score, eye_score, voice_score, clinical_notes,
         egfr, egfr_stage, lifestyle_answers)
        VALUES ({_ph(11)})
    """
    _insert_id(sql, (
        user_id, risk_score, ckd_stage, risk_level,
        lifestyle_score, eye_score, voice_score, clinical_notes,
        egfr, egfr_stage, lifestyle_answers,
    ))


# ── home screening helpers ────────────────────────────────────────────────────
def _now():
    return datetime.now().isoformat(timespec="seconds")


def save_consent(user_id, version=CONSENT_VERSION, language="en"):
    sql = f"INSERT INTO consents (user_id, version, language, created_at) VALUES ({_ph(4)})"
    _insert_id(sql, (user_id, version, language, _now()))


def has_consent(user_id, version=CONSENT_VERSION):
    ph = _ph1()
    row = _fetchone(
        f"SELECT 1 as ok FROM consents WHERE user_id={ph} AND version={ph} LIMIT 1",
        (user_id, version),
    )
    return row is not None


def add_bp_reading(user_id, subject, day_idx, period, systolic, diastolic):
    sql = f"""INSERT INTO bp_readings
              (user_id, subject, day_idx, period, reading_enc, taken_at)
              VALUES ({_ph(6)})"""
    _insert_id(sql, (
        user_id, subject, int(day_idx), period,
        _encrypt(f"{float(systolic)},{float(diastolic)}"), _now(),
    ))


def get_bp_readings(user_id, subject):
    ph = _ph1()
    rows = _fetchall(
        f"SELECT day_idx, period, reading_enc FROM bp_readings "
        f"WHERE user_id={ph} AND subject={ph} ORDER BY id",
        (user_id, subject),
    )
    out = []
    for r in rows:
        try:
            s, d = _decrypt(r["reading_enc"]).split(",")
            out.append((int(r["day_idx"]), r["period"], float(s), float(d)))
        except ValueError:
            continue
    return out


def clear_bp_readings(user_id, subject):
    ph = _ph1()
    _exec(f"DELETE FROM bp_readings WHERE user_id={ph} AND subject={ph}", (user_id, subject))


def save_home_screen(user_id, subject, band, inputs, model_version):
    sql = f"""INSERT INTO home_screens
              (user_id, subject, band, payload_enc, model_version, created_at)
              VALUES ({_ph(6)})"""
    _insert_id(sql, (
        user_id, subject, band,
        _encrypt(json.dumps(inputs)), model_version, _now(),
    ))


def get_latest_home_screen(user_id, subject):
    ph = _ph1()
    row = _fetchone(
        f"SELECT band, payload_enc, model_version, created_at FROM home_screens "
        f"WHERE user_id={ph} AND subject={ph} ORDER BY id DESC LIMIT 1",
        (user_id, subject),
    )
    if not row:
        return None
    try:
        payload = json.loads(_decrypt(row["payload_enc"]) or "{}")
    except ValueError:
        payload = {}
    return {
        "band": row["band"], "inputs": payload,
        "model_version": row["model_version"], "created_at": row["created_at"],
    }


def delete_home_data(user_id):
    ph = _ph1()
    _exec(f"DELETE FROM bp_readings WHERE user_id={ph}", (user_id,))
    _exec(f"DELETE FROM home_screens WHERE user_id={ph}", (user_id,))


# run at import time
create_tables()
