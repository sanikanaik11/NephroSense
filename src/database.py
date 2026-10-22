"""
database.py — NephroSense v3 (dual-backend)
============================================
* LOCAL dev  → SQLite  (database/nephrosense.db, same as before)
* Deployed   → PostgreSQL via DATABASE_URL in .streamlit/secrets.toml
               (Supabase free tier works perfectly)

How the backend is chosen
--------------------------
If  st.secrets["DATABASE_URL"]  (or os.environ["DATABASE_URL"]) is set,
PostgreSQL is used.  Otherwise SQLite is used.  No other code changes needed.

Encryption key
--------------
On SQLite the key is stored in database/enc.key (same as before).
On PostgreSQL the key must be stored as  st.secrets["ENC_KEY"]
(a base64 Fernet key string — generate once with Fernet.generate_key().decode()).

Setup for Streamlit Cloud (Supabase example)
--------------------------------------------
In your app's Secrets panel add:

    DATABASE_URL = "postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres"
    ENC_KEY      = "<your-fernet-key>"          # base64 string, no quotes around it
    GROQ_API_KEY = "..."
    GEMINI_API_KEY = "..."
"""
import os
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import bcrypt
from cryptography.fernet import Fernet, InvalidToken

# ── pick up secrets safely (works before st is initialised) ──────────────────
def _secret(key, default=None):
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)


# ── which backend? ────────────────────────────────────────────────────────────
DATABASE_URL = _secret("DATABASE_URL")          # None → use SQLite
_USE_PG = bool(DATABASE_URL)

# ── paths (SQLite only) ───────────────────────────────────────────────────────
_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH   = os.path.join(_ROOT, "database", "nephrosense.db")
KEY_PATH  = os.path.join(_ROOT, "database", "enc.key")

# ── Fernet encryption ─────────────────────────────────────────────────────────
def _get_fernet():
    # Cloud: key comes from secrets
    enc_key_str = _secret("ENC_KEY")
    if enc_key_str:
        return Fernet(enc_key_str.encode() if isinstance(enc_key_str, str) else enc_key_str)
    # Local: key stored in file (auto-generated if missing)
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
        return value  # legacy plaintext row


# ── connection helpers ────────────────────────────────────────────────────────
@contextmanager
def _get_pg_conn():
    """Yield a psycopg2 connection; always closes it."""
    import psycopg2
    import psycopg2.extras
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_conn():
    """SQLite only — kept for backward compat with export_dataset.py"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── SQL dialect helpers ───────────────────────────────────────────────────────
def _ph(n=1):
    """Return n placeholders: '?' for SQLite, '%s' for PG."""
    ph = "%s" if _USE_PG else "?"
    return ", ".join([ph] * n)

def _ph1():
    return "%s" if _USE_PG else "?"

def _serial():
    return "SERIAL" if _USE_PG else "INTEGER"

def _autoincrement():
    return "" if _USE_PG else "AUTOINCREMENT"

def _now_sql():
    return "NOW()" if _USE_PG else "CURRENT_TIMESTAMP"


# ── execute helpers ───────────────────────────────────────────────────────────
def _exec(sql, params=()):
    """Run a single DML/DDL statement."""
    if _USE_PG:
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
    else:
        conn = get_conn()
        conn.execute(sql, params)
        conn.commit()
        conn.close()

def _fetchall(sql, params=()):
    if _USE_PG:
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
    else:
        conn = get_conn()
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

def _fetchone(sql, params=()):
    if _USE_PG:
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return dict(row) if row else None
    else:
        conn = get_conn()
        row = conn.execute(sql, params).fetchone()
        conn.close()
        return dict(row) if row else None

def _insert_returning_id(sql, params=()):
    """INSERT and return the new row's id."""
    if _USE_PG:
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
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
def create_tables():
    if _USE_PG:
        _create_tables_pg()
    else:
        _create_tables_sqlite()


def _create_tables_pg():
    ddl_users = """
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
        )
    """
    ddl_assessments = """
        CREATE TABLE IF NOT EXISTS assessments (
            id                INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
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
        )
    """
    ddl_login_log = """
        CREATE TABLE IF NOT EXISTS login_log (
            id           INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            user_id      INTEGER NOT NULL,
            logged_in_at TIMESTAMP NOT NULL
        )
    """
    with _get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl_users)
            cur.execute(ddl_assessments)
            cur.execute(ddl_login_log)


def _create_tables_sqlite():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
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
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS login_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            logged_in_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id           INTEGER NOT NULL,
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
            created_at        TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()

    # migrate: add any missing columns
    c.execute("PRAGMA table_info(users)")
    existing_user_cols = {row[1] for row in c.fetchall()}
    for col, col_def in {
        "role": "TEXT DEFAULT 'patient'", "phone": "TEXT", "dob": "TEXT",
        "blood_group": "TEXT", "city": "TEXT", "diabetes": "TEXT DEFAULT ''",
        "hypertension": "TEXT DEFAULT ''", "heart_disease": "TEXT DEFAULT ''",
        "kidney_history": "TEXT DEFAULT ''", "uti_history": "TEXT DEFAULT ''",
        "smoking": "TEXT", "water_intake": "TEXT", "extra_data": "TEXT", "last_login": "TEXT",
    }.items():
        if col not in existing_user_cols:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {col_def}")

    c.execute("PRAGMA table_info(assessments)")
    existing_asm_cols = {row[1] for row in c.fetchall()}
    for col, col_def in {
        "risk_score": "REAL", "ckd_stage": "INTEGER", "risk_level": "TEXT",
        "lifestyle_score": "REAL", "eye_score": "REAL", "voice_score": "REAL",
        "clinical_notes": "TEXT", "lifestyle_answers": "TEXT",
        "egfr": "REAL", "egfr_stage": "TEXT",
    }.items():
        if col not in existing_asm_cols:
            c.execute(f"ALTER TABLE assessments ADD COLUMN {col} {col_def}")

    conn.commit()
    conn.close()


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


# ── age helper ────────────────────────────────────────────────────────────────
def age_from_dob(dob, today=None):
    from datetime import date
    today = today or date.today()
    txt = str(dob or "").strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y"):
        try:
            b = datetime.strptime(txt, fmt).date()
        except ValueError:
            continue
        age = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
        return age if 0 <= age <= 120 else None
    return None


# ── public API ────────────────────────────────────────────────────────────────
def register_user(
    full_name, email, password, gender, age,
    role="patient", phone="", dob="", blood_group="", city="",
    diabetes=False, hypertension=False, heart_disease=False,
    kidney_history=False, uti_history=False,
    smoking="Non-smoker", water_intake="1-2 litres", extra_data="",
):
    ph = _ph1()
    existing = _fetchone(f"SELECT id FROM users WHERE email={ph}", (email,))
    if existing:
        return False, "This email is already registered. Please sign in."

    sql = f"""
        INSERT INTO users
        (full_name, email, password, gender, age, role, phone, dob, blood_group,
         city, diabetes, hypertension, heart_disease, kidney_history, uti_history,
         smoking, water_intake, extra_data)
        VALUES ({_ph(18)})
    """
    _insert_returning_id(sql, (
        full_name, email, hash_pw(password), gender, age, role, phone, dob,
        _encrypt(blood_group),
        city,
        _encrypt(int(diabetes)), _encrypt(int(hypertension)),
        _encrypt(int(heart_disease)), _encrypt(int(kidney_history)),
        _encrypt(int(uti_history)),
        smoking, water_intake, extra_data,
    ))
    return True, f"Account created! Welcome, {full_name.split()[0]}."


def login_user(email, password):
    ph = _ph1()
    sql = f"""
        SELECT id, full_name, email, password, gender, age, role, phone, city,
               blood_group, diabetes, hypertension, heart_disease,
               kidney_history, extra_data, last_login, dob
        FROM users WHERE email={ph}
    """
    row = _fetchone(sql, (email,))
    if not row or not verify_pw(password, row["password"]):
        return False, "Incorrect email or password."

    user = dict(row)
    previous_last_login = user.get("last_login")

    # upgrade legacy sha256 hashes
    if not _is_bcrypt_hash(row["password"]):
        _exec(f"UPDATE users SET password={ph} WHERE id={ph}", (hash_pw(password), row["id"]))

    now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _exec(f"UPDATE users SET last_login={ph} WHERE id={ph}", (now_str, row["id"]))
    _exec(f"INSERT INTO login_log (user_id, logged_in_at) VALUES ({_ph(2)})", (row["id"], now_iso))

    user.pop("password", None)
    user["blood_group"] = _decrypt(user.get("blood_group"))
    for f in ("diabetes", "hypertension", "heart_disease", "kidney_history"):
        user[f] = int(_decrypt(user.get(f), default="0") or 0)
    user["last_login"] = previous_last_login or now_str
    _a = age_from_dob(user.get("dob"))
    if _a is not None:
        user["age"] = _a

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
    _insert_returning_id(sql, (
        user_id, risk_score, ckd_stage, risk_level,
        lifestyle_score, eye_score, voice_score, clinical_notes,
        egfr, egfr_stage, lifestyle_answers,
    ))


# run schema creation when the module is imported
create_tables()
