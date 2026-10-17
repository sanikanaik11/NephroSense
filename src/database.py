"""
database.py — NephroSense v2 (security-hardened)
Handles all user data with full patient profile.

Two security upgrades:
  1. Passwords: bcrypt with a per-user salt. Old accounts created under
     the previous unsalted SHA-256 version still log in correctly and
     are silently upgraded to bcrypt on their next successful login.
  2. Sensitive health fields (diabetes, hypertension, heart_disease,
     kidney_history, uti_history, blood_group) are encrypted at rest
     using Fernet symmetric encryption.

Plus: assessments now also store clinical_notes — a JSON blob of the
optional symptom/medical-history answers (urine colour, medications,
etc.) collected in the Health Assessment. These are NOT used in the
risk score; they're saved purely as documentation for a doctor visit.

Plus: users now also store last_login — updated on every successful
sign-in, shown in the sidebar so the patient/doctor can see when the
account was last accessed.

Needs: pip install bcrypt cryptography
"""
import sqlite3
import os
import json
from datetime import datetime

import bcrypt
from cryptography.fernet import Fernet, InvalidToken

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "database",
    "nephrosense.db",
)
KEY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "database",
    "enc.key",
)

ENCRYPTED_HEALTH_FIELDS = [
    "diabetes", "hypertension", "heart_disease", "kidney_history",
    "uti_history", "blood_group",
]


def _get_fernet():
    os.makedirs(os.path.dirname(KEY_PATH), exist_ok=True)
    if not os.path.exists(KEY_PATH):
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as f:
            f.write(key)
    with open(KEY_PATH, "rb") as f:
        key = f.read()
    return Fernet(key)


_fernet = _get_fernet()


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
        return value  # legacy plaintext row from before encryption was added


def hash_pw(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def _is_bcrypt_hash(stored):
    return stored.startswith("$2a$") or stored.startswith("$2b$") or stored.startswith("$2y$")


def verify_pw(pw, stored_hash):
    if _is_bcrypt_hash(stored_hash):
        try:
            return bcrypt.checkpw(pw.encode(), stored_hash.encode())
        except ValueError:
            return False
    else:
        import hashlib
        return hashlib.sha256(pw.encode()).hexdigest() == stored_hash


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables():
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
    conn.commit()

    c.execute("PRAGMA table_info(users)")
    existing_user_cols = {row[1] for row in c.fetchall()}
    needed_user_cols = {
        "role": "TEXT DEFAULT 'patient'",
        "phone": "TEXT",
        "dob": "TEXT",
        "blood_group": "TEXT",
        "city": "TEXT",
        "diabetes": "TEXT DEFAULT ''",
        "hypertension": "TEXT DEFAULT ''",
        "heart_disease": "TEXT DEFAULT ''",
        "kidney_history": "TEXT DEFAULT ''",
        "uti_history": "TEXT DEFAULT ''",
        "smoking": "TEXT",
        "water_intake": "TEXT",
        "extra_data": "TEXT",
        "last_login": "TEXT",
    }
    for col, col_def in needed_user_cols.items():
        if col not in existing_user_cols:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {col_def}")
    conn.commit()

    c.execute("""
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
        )
    """)
    conn.commit()

    c.execute("PRAGMA table_info(assessments)")
    existing_cols = {row[1] for row in c.fetchall()}
    needed_cols = {
        "risk_score": "REAL",
        "ckd_stage": "INTEGER",
        "risk_level": "TEXT",
        "lifestyle_score": "REAL",
        "eye_score": "REAL",
        "voice_score": "REAL",
        "clinical_notes": "TEXT",
        "lifestyle_answers": "TEXT",
        "egfr": "REAL",
        "egfr_stage": "TEXT",
    }
    for col, col_type in needed_cols.items():
        if col not in existing_cols:
            c.execute(f"ALTER TABLE assessments ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()


def register_user(
    full_name,
    email,
    password,
    gender,
    age,
    role="patient",
    phone="",
    dob="",
    blood_group="",
    city="",
    diabetes=False,
    hypertension=False,
    heart_disease=False,
    kidney_history=False,
    uti_history=False,
    smoking="Non-smoker",
    water_intake="1-2 litres",
    extra_data="",
):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email=?", (email,))
    if c.fetchone():
        conn.close()
        return False, "This email is already registered. Please sign in."

    c.execute(
        """
        INSERT INTO users
        (full_name, email, password, gender, age, role, phone, dob, blood_group,
         city, diabetes, hypertension, heart_disease, kidney_history, uti_history,
         smoking, water_intake, extra_data)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            full_name,
            email,
            hash_pw(password),
            gender,
            age,
            role,
            phone,
            dob,
            _encrypt(blood_group),
            city,
            _encrypt(int(diabetes)),
            _encrypt(int(hypertension)),
            _encrypt(int(heart_disease)),
            _encrypt(int(kidney_history)),
            _encrypt(int(uti_history)),
            smoking,
            water_intake,
            extra_data,
        ),
    )
    conn.commit()
    conn.close()
    return True, f"Account created! Welcome, {full_name.split()[0]}."


def login_user(email, password):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """
        SELECT id, full_name, email, password, gender, age, role, phone, city,
               blood_group, diabetes, hypertension, heart_disease,
               kidney_history, extra_data, last_login
        FROM users
        WHERE email=?
        """,
        (email,),
    )
    row = c.fetchone()

    if not row or not verify_pw(password, row["password"]):
        conn.close()
        return False, "Incorrect email or password."

    user = dict(row)
    previous_last_login = user.get("last_login")

    if not _is_bcrypt_hash(row["password"]):
        c.execute("UPDATE users SET password=? WHERE id=?", (hash_pw(password), row["id"]))
        conn.commit()

    # Record this login's timestamp for next time, but return the
    # PREVIOUS last_login to display ("last login: ..." should show
    # when they last logged in before this current session, not the
    # moment that just happened).
    now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    c.execute("UPDATE users SET last_login=? WHERE id=?", (now_str, row["id"]))
    conn.commit()

    conn.close()

    user.pop("password", None)
    user["blood_group"] = _decrypt(user.get("blood_group"))
    for f in ("diabetes", "hypertension", "heart_disease", "kidney_history"):
        user[f] = int(_decrypt(user.get(f), default="0") or 0)
    user["last_login"] = previous_last_login or now_str

    return True, user


def get_user_history(user_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM assessments WHERE user_id=? ORDER BY created_at DESC, id DESC",
        (user_id,),
    )
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_component_scores(user_id):
    conn = get_conn()
    c = conn.cursor()
    result = {}
    for col in ("lifestyle_score", "eye_score", "voice_score"):
        c.execute(
            f"SELECT {col} FROM assessments "
            f"WHERE user_id=? AND {col} IS NOT NULL "
            f"ORDER BY created_at DESC, id DESC LIMIT 1",
            (user_id,),
        )
        row = c.fetchone()
        result[col] = row[0] if row else None
    conn.close()
    return result


def get_latest_clinical_notes(user_id):
    """
    Returns the JSON string of the most recent Health Assessment's
    symptom/medical-history answers (see lifestyle_model.CLINICAL_QUESTIONS),
    or None if the user hasn't submitted any yet.
    """
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT clinical_notes FROM assessments "
        "WHERE user_id=? AND clinical_notes IS NOT NULL AND clinical_notes != '' "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (user_id,),
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def get_latest_lifestyle_answers(user_id):
    """
    Returns the JSON string of the most recent Health Assessment's raw
    QUESTIONS answers (age, bp, sg, etc.), so the form can be
    pre-filled instead of asking the user to type everything again
    on every visit.
    """
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT lifestyle_answers FROM assessments "
        "WHERE user_id=? AND lifestyle_answers IS NOT NULL AND lifestyle_answers != '' "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (user_id,),
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def get_latest_egfr(user_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT egfr, egfr_stage FROM assessments "
        "WHERE user_id=? AND egfr IS NOT NULL "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (user_id,),
    )
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def save_assessment(
    user_id,
    risk_score,
    ckd_stage,
    risk_level,
    lifestyle_score=None,
    eye_score=None,
    voice_score=None,
    clinical_notes=None,
    egfr=None,
    egfr_stage=None,
    lifestyle_answers=None,
):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO assessments
        (user_id, risk_score, ckd_stage, risk_level,
         lifestyle_score, eye_score, voice_score, clinical_notes, egfr, egfr_stage,
         lifestyle_answers)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            risk_score,
            ckd_stage,
            risk_level,
            lifestyle_score,
            eye_score,
            voice_score,
            clinical_notes,
            egfr,
            egfr_stage,
            lifestyle_answers,
        ),
    )
    conn.commit()
    conn.close()


create_tables()