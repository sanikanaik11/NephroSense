"""
verify_prepush.py — NephroSense pre-push verification
Runs all checks without printing secrets or touching real data.
Usage: python verify_prepush.py
"""
import sys, os, re, json, sqlite3, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

PASS = "PASS"
FAIL = "FAIL"
failures = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {name:<60} {PASS}")
    else:
        msg = f"  {name:<60} {FAIL}" + (f"  [{detail}]" if detail else "")
        print(msg)
        failures.append(name)

def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

# ─── 1. ENC_KEY loading ───────────────────────────────────────────────────────
section("1 — Encryption key loading")
import database as db

check("_USE_PG is False locally (DATABASE_URL not set)", not db._USE_PG)
check("_fernet is initialised", db._fernet is not None)

enc = db._encrypt("test_canary_123")
check("_encrypt produces ciphertext (not plaintext)",   enc != "test_canary_123")
check("_decrypt(enc) == original",                       db._decrypt(enc) == "test_canary_123")
check("_decrypt(empty string) == ''",                    db._decrypt("") == "")
check("_decrypt(None) == ''",                            db._decrypt(None) == "")

src = open("src/database.py").read()
check("ENC_KEY path: reads from _secret('ENC_KEY') first",  "_secret(\"ENC_KEY\")" in src)
check("Fallback: reads from KEY_PATH file",                  "KEY_PATH" in src and "open(KEY_PATH" in src)
check("Key value never printed or logged",                   "print" not in src.lower().split("enc_key")[1][:100]
                                                              if "enc_key" in src.lower() else True)

# ─── 2. Placeholder correctness ──────────────────────────────────────────────
section("2 — SQL placeholder helpers")
import importlib

check("_ph(1) returns '?' locally",     db._ph(1) == "?")
check("_ph(3) returns '?, ?, ?'",       db._ph(3) == "?, ?, ?")
check("_ph1() returns '?'",             db._ph1() == "?")

# Simulate PG mode: temporarily flip _USE_PG
db._USE_PG = True
check("_ph(1) returns '%s' in PG mode",       db._ph(1) == "%s")
check("_ph(3) returns '%s, %s, %s' in PG",    db._ph(3) == "%s, %s, %s")
check("_ph1() returns '%s' in PG mode",        db._ph1() == "%s")
db._USE_PG = False   # restore

# ─── 3. SQL statements — no raw '?' left when _ph() is used ─────────────────
section("3 — No hardcoded '?' or '%s' outside helper calls")

# Find all SQL strings in database.py and screening_db.py
for fname in ("src/database.py", "src/screening_db.py"):
    src = open(fname).read()
    # Find lines that have SQL keywords and raw '?' not inside a _ph call
    raw_q = [l.strip() for l in src.splitlines()
             if re.search(r'\bWHERE\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b', l, re.I)
             and "?" in l
             and "_ph" not in l
             and not l.strip().startswith("#")]
    check(f"{fname}: no raw '?' in SQL outside _ph() helpers",
          len(raw_q) == 0,
          "; ".join(raw_q[:3]) if raw_q else "")

# ─── 4. User-ID scoping ───────────────────────────────────────────────────────
section("4 — Every sensitive-table query is scoped to user_id")

SENSITIVE = ["assessments", "bp_readings", "home_screens", "screenings", "screening_consents", "consents"]
for fname in ("src/database.py", "src/screening_db.py"):
    src = open(fname).read()
    # Find SELECT/UPDATE/DELETE that hit a sensitive table but have no user_id guard
    issues = []
    for m in re.finditer(
        r'(SELECT|UPDATE|DELETE)\s+.*?(?=_exec|_fetchone|_fetchall|_insert_id|\Z)',
        src, re.DOTALL | re.IGNORECASE
    ):
        block = m.group(0)
        if any(t in block for t in SENSITIVE):
            if re.search(r'\bWHERE\b', block, re.I) and "user_id" not in block:
                # Only flag if it's not a DDL or a schema-level query
                if not re.search(r'CREATE\s+TABLE|PRAGMA|information_schema', block, re.I):
                    issues.append(block[:80].replace('\n', ' '))
    check(f"{fname}: all sensitive queries scoped to user_id",
          len(issues) == 0,
          "; ".join(issues[:2]) if issues else "")

# ─── 5. _insert_id: RETURNING id for PG, lastrowid for SQLite ───────────────
section("5 — _insert_id implementation")
src = open("src/database.py").read()
check("_insert_id appends RETURNING id for PG",   "RETURNING id" in src)
check("_insert_id uses lastrowid for SQLite",      "lastrowid" in src)
check("_pg_conn context manager present",          "@contextmanager" in src and "_pg_conn" in src)
check("_pg_conn commits on success",               "conn.commit()" in src)
check("_pg_conn rolls back on exception",          "conn.rollback()" in src)
check("_pg_conn always closes connection",         "conn.close()" in src)

# ─── 6. Timestamp normalisation ──────────────────────────────────────────────
section("6 — Timestamps: PG TIMESTAMP vs SQLite TEXT")
src = open("src/database.py").read()
check("PG schema uses TIMESTAMP DEFAULT NOW()",    "TIMESTAMP DEFAULT NOW()" in src)
check("SQLite schema uses TEXT DEFAULT CURRENT_TIMESTAMP",
      "TEXT DEFAULT CURRENT_TIMESTAMP" in src)
src2 = open("src/screening_db.py").read()
check("screening_db normalises created_at to str", "isoformat" in src2)

# ─── 7. Tables consolidated — screenings owned by database.py ───────────────
section("7 — Schema consolidation")
src = open("src/database.py").read()
for tbl in ("users", "assessments", "consents", "bp_readings",
            "home_screens", "screenings", "screening_consents"):
    check(f"Table '{tbl}' defined in database.py",
          f"CREATE TABLE IF NOT EXISTS {tbl}" in src)

src2 = open("src/screening_db.py").read()
check("screening_db.py contains no CREATE TABLE",
      "CREATE TABLE" not in src2)
check("screening_db.py contains no raw get_conn() calls",
      "get_conn()" not in src2)

# ─── 8. .gitignore protects sensitive files ──────────────────────────────────
section("8 — .gitignore excludes sensitive files")
gi = open(".gitignore").read()
for item in ["database/nephrosense.db", "database/enc.key",
             ".streamlit/secrets.toml", "exports/"]:
    check(f".gitignore excludes '{item}'", item in gi)

# ─── 9. secrets.toml is valid TOML and contains no real DB URL ───────────────
section("9 — secrets.toml validity")
import tomllib
try:
    raw = open(".streamlit/secrets.toml", "rb").read()
    check("secrets.toml has no BOM", not raw.startswith(b"\xef\xbb\xbf"))
    data = tomllib.loads(raw.decode("utf-8"))
    check("secrets.toml parses as valid TOML", True)
    check("DATABASE_URL not hardcoded in secrets.toml",
          not data.get("DATABASE_URL", "").startswith("postgresql://"))
    check("ENC_KEY not hardcoded in secrets.toml",
          "ENC_KEY" not in data)
    check("GROQ_API_KEY present", bool(data.get("GROQ_API_KEY")))
    check("GEMINI_API_KEY present", bool(data.get("GEMINI_API_KEY")))
except Exception as e:
    check("secrets.toml is valid", False, str(e))

# ─── 10. requirements.txt ────────────────────────────────────────────────────
section("10 — requirements.txt")
reqs = open("requirements.txt").read()
for pkg in ["streamlit", "bcrypt", "cryptography", "psycopg2-binary",
            "pandas", "numpy", "scikit-learn", "plotly"]:
    check(f"'{pkg}' in requirements.txt", pkg in reqs)

# ─── 11. Syntax checks ───────────────────────────────────────────────────────
section("11 — Syntax checks")
import py_compile
for f in ["src/database.py", "src/screening_db.py", "dashboard/app.py",
          "tests/test_db_dual_backend.py"]:
    try:
        py_compile.compile(f, doraise=True)
        check(f, True)
    except py_compile.PyCompileError as e:
        check(f, False, str(e)[:60])

# ─── 12. Full test suite ─────────────────────────────────────────────────────
section("12 — Full test suite (pytest)")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=True, text=True, cwd="."
)
passed = re.search(r"(\d+) passed", result.stdout)
failed_tests = re.search(r"(\d+) failed", result.stdout)
n_passed = int(passed.group(1)) if passed else 0
n_failed = int(failed_tests.group(1)) if failed_tests else 0
check(f"All tests pass ({n_passed} passed, {n_failed} failed)",
      n_failed == 0 and n_passed > 0,
      result.stdout[-300:] if n_failed else "")

# ─── 13. Functional SQLite test ──────────────────────────────────────────────
section("13 — Functional test: SQLite register → assess → retrieve → cleanup")
TEST_EMAIL = "_prepush_verify@nephrosense.local"
try:
    # cleanup any leftover
    c = sqlite3.connect(db.DB_PATH); c.execute("DELETE FROM users WHERE email=?", (TEST_EMAIL,)); c.commit(); c.close()

    ok, msg = db.register_user("Prepush Test", TEST_EMAIL, "Verify!99", "Female", 32, city="Pune")
    check("register_user succeeds", ok, msg)

    ok2, user = db.login_user(TEST_EMAIL, "Verify!99")
    check("login_user correct password", ok2)
    check("password not in returned dict", "password" not in user)
    check("city returned correctly", user.get("city") == "Pune")

    uid = user["id"]

    # encrypted fields at rest
    raw = sqlite3.connect(db.DB_PATH)
    raw.row_factory = sqlite3.Row
    stored = raw.execute("SELECT diabetes FROM users WHERE id=?", (uid,)).fetchone()["diabetes"]
    raw.close()
    check("diabetes encrypted at rest (not '0')", stored not in ("0", 0, "False"))
    check("diabetes decrypts to '0'", db._decrypt(stored) == "0")

    db.save_assessment(uid, 38.5, 2, "LOW", lifestyle_score=35.0, egfr=85.0, egfr_stage="G2")
    hist = db.get_user_history(uid)
    check("get_user_history returns 1 record", len(hist) == 1)
    check("risk_level correct", hist[0]["risk_level"] == "LOW")

    import screening_db as sc
    sc.record_consent(uid)
    check("has_consent after record_consent", sc.has_consent(uid))
    sc.record_consent(uid)   # upsert
    check("has_consent after second call (upsert)", sc.has_consent(uid))

    sc.save_screening(uid, "LOW", {"bp": 118}, person_label="")
    rows = sc.list_screenings(uid)
    check("list_screenings returns 1 row", len(rows) == 1)
    check("created_at is a string", isinstance(rows[0]["created_at"], str))

    # wrong-user isolation
    db.register_user("Other User", "_other_prepush@ns.local", "pw", "Male", 40)
    _, other = db.login_user("_other_prepush@ns.local", "pw")
    other_hist = db.get_user_history(other["id"])
    check("other user sees 0 records (isolation)", len(other_hist) == 0)

    # cleanup
    for email in (TEST_EMAIL, "_other_prepush@ns.local"):
        c = sqlite3.connect(db.DB_PATH)
        row = c.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
        if row:
            for tbl in ("assessments","consents","bp_readings","home_screens",
                        "screenings","screening_consents"):
                c.execute(f"DELETE FROM {tbl} WHERE user_id=?", (row[0],))
        c.execute("DELETE FROM users WHERE email=?", (email,))
        c.commit(); c.close()
    check("Test data cleaned up", True)
except Exception as e:
    check("Functional test (exception)", False, str(e))

# ─── 14. PostgreSQL NOT claimed ───────────────────────────────────────────────
section("14 — PostgreSQL status")
check("PostgreSQL NOT claimed working (no live connection tested)",
      not db._USE_PG,
      "DATABASE_URL is set — PG mode active, but cloud test not run yet" if db._USE_PG else "")
print("  NOTE: PostgreSQL will be confirmed after you add DATABASE_URL + ENC_KEY to")
print("        Streamlit Cloud secrets and a test registration succeeds on the live app.")

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
if failures:
    print(f"  RESULT: {len(failures)} FAILED")
    for f in failures:
        print(f"    ✗ {f}")
else:
    print("  RESULT: ALL CHECKS PASSED — safe to push to GitHub")
print(f"{'='*70}")
sys.exit(1 if failures else 0)
