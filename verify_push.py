"""
Pre-push verification script — NephroSense
Runs entirely read-only (except creating/deleting a temporary test user).
Does NOT push to GitHub. Does NOT modify any application files.
"""
import sys
import os
import sqlite3
import json
import re
import subprocess
import py_compile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

PASS = "PASS"
FAIL = "FAIL"
failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {name:<45} {PASS}")
    else:
        msg = f"  {name:<45} {FAIL}" + (f"  ({detail})" if detail else "")
        print(msg)
        failures.append(name)


def section(title):
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


# ── 1. Backend detection ──────────────────────────────────────────────────────
section("1 — Backend detection")
import database as db
check("_USE_PG is False (SQLite locally)", not db._USE_PG)

# ── 2. Syntax checks ──────────────────────────────────────────────────────────
section("2 — Syntax checks")
for f in ["src/database.py", "src/screening_db.py", "export_dataset.py"]:
    try:
        py_compile.compile(f, doraise=True)
        check(f, True)
    except py_compile.PyCompileError as e:
        check(f, False, str(e)[:60])

# ── 3. requirements.txt ───────────────────────────────────────────────────────
section("3 — requirements.txt packages")
reqs = open("requirements.txt").read()
for pkg in ["streamlit", "bcrypt", "cryptography", "psycopg2-binary",
            "openpyxl", "pandas", "numpy", "scikit-learn"]:
    check(pkg, pkg in reqs)

# ── 4. .gitignore ─────────────────────────────────────────────────────────────
section("4 — .gitignore excludes sensitive files")
gi = open(".gitignore").read()
for item in ["database/nephrosense.db", "database/enc.key",
             ".streamlit/secrets.toml", "exports/"]:
    check(item, item in gi)

# ── 5. Fernet encryption round-trip ──────────────────────────────────────────
section("5 — Fernet key loading and encryption")
check("_fernet is initialised", db._fernet is not None)
enc = db._encrypt("test_patient_data_123")
check("encrypt/decrypt round-trip", db._decrypt(enc) == "test_patient_data_123")
check("_decrypt(empty string) == ''", db._decrypt("") == "")
check("_decrypt(None) == ''",         db._decrypt(None) == "")
check("encrypted value != plaintext", enc != "test_patient_data_123")

# ── 6. Table existence ────────────────────────────────────────────────────────
section("6 — All tables exist (SQLite)")
import screening_db as sc
conn_raw = sqlite3.connect(db.DB_PATH)
tables = {r[0] for r in conn_raw.execute(
    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
conn_raw.close()
for t in ["users", "assessments", "login_log", "screenings", "consents"]:
    check(f"table '{t}' exists", t in tables)

# ── 7. consents column migration ─────────────────────────────────────────────
section("7 — consents table has accepted_at column")
conn_raw2 = sqlite3.connect(db.DB_PATH)
cols = {r[1] for r in conn_raw2.execute("PRAGMA table_info(consents)").fetchall()}
conn_raw2.close()
check("consents.accepted_at column present", "accepted_at" in cols)

# ── 8. Registration & login ───────────────────────────────────────────────────
section("8 — Registration and login")
TEST_EMAIL = "_verify_push_test@nephrosense.local"
# remove any leftover from a prior run
c = sqlite3.connect(db.DB_PATH)
c.execute("DELETE FROM users WHERE email=?", (TEST_EMAIL,))
c.commit(); c.close()

ok, msg = db.register_user(
    "Verify Test", TEST_EMAIL, "SecurePass!1", "Female", 28,
    city="Mumbai", diabetes=True, hypertension=False)
check("register_user (new account)", ok, msg)

ok2, msg2 = db.register_user("Dup", TEST_EMAIL, "pw", "Male", 30)
check("register_user (duplicate → rejected)", not ok2)

ok3, user = db.login_user(TEST_EMAIL, "SecurePass!1")
check("login_user (correct password)", ok3, str(user)[:60] if not ok3 else "")
uid = user["id"] if ok3 else None
check("login returns full_name", ok3 and user.get("full_name") == "Verify Test")
check("login returns city", ok3 and user.get("city") == "Mumbai")
check("password NOT in returned user dict", ok3 and "password" not in user)

ok4, _ = db.login_user(TEST_EMAIL, "WrongPassword")
check("login_user (wrong password → rejected)", not ok4)

# ── 9. Encrypted fields at rest ───────────────────────────────────────────────
section("9 — Encrypted health fields stored as ciphertext")
if uid:
    raw_row = sqlite3.connect(db.DB_PATH)
    raw_row.row_factory = sqlite3.Row
    r = raw_row.execute(
        "SELECT diabetes, blood_group FROM users WHERE id=?", (uid,)).fetchone()
    raw_row.close()
    stored = r["diabetes"]
    check("diabetes NOT stored as plaintext '1'", stored not in ("1", 1))
    check("diabetes NOT stored as plaintext '0'", stored not in ("0", 0))
    check("decrypted diabetes == '1'", db._decrypt(stored) == "1")
else:
    check("skipped (login failed)", False, "uid not available")

# ── 10. Assessments ───────────────────────────────────────────────────────────
section("10 — save_assessment and get_user_history")
if uid:
    db.save_assessment(uid, risk_score=42.5, ckd_stage=2, risk_level="MEDIUM",
                       lifestyle_score=40.0, egfr=78.3, egfr_stage="G2",
                       clinical_notes=json.dumps({"urine_color": "yellow"}))
    db.save_assessment(uid, risk_score=28.0, ckd_stage=1, risk_level="LOW",
                       lifestyle_score=25.0, egfr=90.1, egfr_stage="G1")
    hist = db.get_user_history(uid)
    check("get_user_history returns 2 records", len(hist) == 2, f"got {len(hist)}")
    check("most recent record first (LOW)", hist[0]["risk_level"] == "LOW")
    check("second record is MEDIUM", hist[1]["risk_level"] == "MEDIUM")

    scores = db.get_latest_component_scores(uid)
    check("get_latest_component_scores lifestyle_score=25.0",
          scores.get("lifestyle_score") == 25.0)

    egfr_row = db.get_latest_egfr(uid)
    check("get_latest_egfr == 90.1", egfr_row and egfr_row.get("egfr") == 90.1)

    notes = db.get_latest_clinical_notes(uid)
    check("get_latest_clinical_notes returns notes from first assessment",
          notes is not None and "urine_color" in notes)

# ── 11. Screening consent ─────────────────────────────────────────────────────
section("11 — screening_db consent")
if uid:
    check("has_consent before record == False", sc.has_consent(uid) == False)
    sc.record_consent(uid)
    check("has_consent after record == True",  sc.has_consent(uid) == True)
    sc.record_consent(uid)  # second call = UPDATE path
    check("has_consent after second call == True (upsert)", sc.has_consent(uid) == True)

# ── 12. Screening records ─────────────────────────────────────────────────────
section("12 — screening_db save/list/delete")
if uid:
    sc.save_screening(uid, "LOW",    {"bp": 118, "note": "self"}, person_label="")
    sc.save_screening(uid, "MEDIUM", {"bp": 145, "note": "mum"},  person_label="Mother")
    rows = sc.list_screenings(uid, limit=10)
    check("list_screenings returns 2 rows",           len(rows) == 2, f"got {len(rows)}")
    check("payload is dict (decrypted)",              isinstance(rows[0]["payload"], dict))
    check("created_at normalised to str",             isinstance(rows[0]["created_at"], str))
    check("most recent first (MEDIUM)",               rows[0]["band"] == "MEDIUM")
    check("person_label decrypted (Mother)",          rows[0]["person_label"] == "Mother")
    check("self-screening has empty person_label",    rows[1]["person_label"] == "")

    d = sc.last_self_screening_date(uid)
    check("last_self_screening_date is not None",     d is not None)

    sc.delete_screening_data(uid)
    check("delete_screening_data empties records",    sc.list_screenings(uid) == [])

# ── 13. Cleanup ───────────────────────────────────────────────────────────────
section("13 — Cleanup test data")
if uid:
    c2 = sqlite3.connect(db.DB_PATH)
    c2.execute("DELETE FROM consents    WHERE user_id=?", (uid,))
    c2.execute("DELETE FROM assessments WHERE user_id=?", (uid,))
    c2.execute("DELETE FROM login_log   WHERE user_id=?", (uid,))
    c2.execute("DELETE FROM users       WHERE id=?",      (uid,))
    c2.commit(); c2.close()
    print(f"  Test user (uid={uid}) and all related records removed  PASS")

# ── 14. Git diff secret scan ──────────────────────────────────────────────────
section("14 — Git diff: no real secrets in changed files")
diff_result = subprocess.run(
    ["git", "diff", "HEAD", "--",
     "src/database.py", "src/screening_db.py",
     "requirements.txt", ".gitignore", "export_dataset.py"],
    capture_output=True, text=True, cwd=".",
)
diff_lines = [l for l in diff_result.stdout.splitlines() if l.startswith("+")]

secret_patterns = [
    ("Real GROQ key",   r"gsk_[A-Za-z0-9]{20,}"),
    ("Real Gemini key", r"AQ\.[A-Za-z0-9_\-]{20,}"),
    ("Real PG URL",     r"postgresql://[^<\s\"]+:[^<\s\"]+@[^<\s\"]+\.supabase\.co"),
    ("Real ENC_KEY",    r'ENC_KEY\s*=\s*"[A-Za-z0-9+/=]{30,}"'),
    ("Real password",   r"password\s*=\s*\"[A-Za-z0-9!@#\$%]{8,}\""),
]
found_secrets = []
for line in diff_lines:
    for name, pattern in secret_patterns:
        if re.search(pattern, line):
            found_secrets.append((name, line[:80]))

check("No real GROQ/Gemini/PG/ENC keys in diff",
      not found_secrets,
      "; ".join(f"{n}" for n, _ in found_secrets) if found_secrets else "")

if found_secrets:
    for kind, line in found_secrets:
        print(f"    !! {kind}: {line}")

# ── 15. secrets.toml not tracked ─────────────────────────────────────────────
section("15 — .streamlit/secrets.toml not tracked by git")
tracked = subprocess.run(
    ["git", "ls-files", ".streamlit/secrets.toml"],
    capture_output=True, text=True, cwd="."
).stdout.strip()
check(".streamlit/secrets.toml not in git tracking", tracked == "")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=" * 62)
if failures:
    print(f"  RESULT: {len(failures)} FAILED")
    for f in failures:
        print(f"    ✗ {f}")
else:
    print("  RESULT: ALL CHECKS PASSED — safe to push to GitHub")
print("=" * 62)
sys.exit(1 if failures else 0)
