"""
test_db_dual_backend.py — NephroSense
Exercises every public database function on the LOCAL SQLite backend
(no live PostgreSQL required). A dummy test user is created, tested,
and fully removed — no real patient data touched.
"""
import os
import sys
import json
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import database as db
import screening_db as sc

TEST_EMAIL = "_pytest_db_dual@nephrosense.local"


def _cleanup(uid=None):
    """Remove every trace of the test user from all tables."""
    conn = db.get_conn()
    if uid:
        for tbl in ("assessments", "consents", "bp_readings",
                    "home_screens", "screenings", "screening_consents"):
            conn.execute(f"DELETE FROM {tbl} WHERE user_id=?", (uid,))
    conn.execute("DELETE FROM users WHERE email=?", (TEST_EMAIL,))
    conn.commit()
    conn.close()


def test_backend_is_sqlite():
    assert not db._USE_PG, "Expected SQLite locally (DATABASE_URL must not be set)"


def test_encryption_roundtrip():
    enc = db._encrypt("test_value_123")
    assert enc != "test_value_123"
    assert db._decrypt(enc) == "test_value_123"
    assert db._decrypt("") == ""
    assert db._decrypt(None) == ""


def test_all_tables_exist():
    conn = db.get_conn()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    required = {"users", "assessments", "consents", "bp_readings",
                "home_screens", "screenings", "screening_consents"}
    missing = required - tables
    assert not missing, f"Missing tables: {missing}"


def test_register_and_login():
    _cleanup()
    ok, msg = db.register_user(
        "DB Test User", TEST_EMAIL, "TestPass!1", "Female", 30,
        city="Mumbai", diabetes=True, hypertension=False,
    )
    assert ok, f"register_user failed: {msg}"

    # duplicate email rejected
    ok2, _ = db.register_user("Dup", TEST_EMAIL, "pw", "Male", 25)
    assert not ok2

    ok3, user = db.login_user(TEST_EMAIL, "TestPass!1")
    assert ok3, f"login_user failed: {user}"
    assert user["full_name"] == "DB Test User"
    assert user["city"] == "Mumbai"
    assert "password" not in user

    # wrong password rejected
    ok4, _ = db.login_user(TEST_EMAIL, "WrongPass")
    assert not ok4


def test_encrypted_fields_at_rest():
    _cleanup()
    db.register_user("Enc Test", TEST_EMAIL, "pw123", "Male", 40, diabetes=True)
    conn = db.get_conn()
    row = conn.execute("SELECT diabetes FROM users WHERE email=?", (TEST_EMAIL,)).fetchone()
    conn.close()
    stored = row["diabetes"]
    assert stored not in ("1", 1, "True", True), "diabetes stored as plaintext — encryption failed"
    assert db._decrypt(stored) == "1"


def test_save_and_retrieve_assessments():
    _cleanup()
    db.register_user("Asm Test", TEST_EMAIL, "pw", "Male", 50)
    _, user = db.login_user(TEST_EMAIL, "pw")
    uid = user["id"]

    db.save_assessment(uid, risk_score=42.5, ckd_stage=2, risk_level="MEDIUM",
                       lifestyle_score=40.0, egfr=78.3, egfr_stage="G2",
                       clinical_notes=json.dumps({"urine_color": "yellow"}))
    db.save_assessment(uid, risk_score=28.0, ckd_stage=1, risk_level="LOW",
                       lifestyle_score=25.0, egfr=91.0, egfr_stage="G1")

    hist = db.get_user_history(uid)
    assert len(hist) == 2
    assert hist[0]["risk_level"] == "LOW"    # most recent first
    assert hist[1]["risk_level"] == "MEDIUM"

    scores = db.get_latest_component_scores(uid)
    assert scores["lifestyle_score"] == 25.0

    egfr = db.get_latest_egfr(uid)
    assert egfr["egfr"] == 91.0

    notes = db.get_latest_clinical_notes(uid)
    assert notes is not None and "urine_color" in notes

    _cleanup(uid)


def test_consent_and_home_screening():
    _cleanup()
    db.register_user("Home Test", TEST_EMAIL, "pw", "Male", 45)
    _, user = db.login_user(TEST_EMAIL, "pw")
    uid = user["id"]

    assert not db.has_consent(uid)
    db.save_consent(uid)
    assert db.has_consent(uid)

    db.add_bp_reading(uid, "self", 1, "am", 120, 80)
    db.add_bp_reading(uid, "self", 1, "pm", 125, 82)
    readings = db.get_bp_readings(uid, "self")
    assert len(readings) == 2
    assert readings[0] == (1, "am", 120.0, 80.0)

    db.save_home_screen(uid, "self", "LOW", {"age": 45, "sbp": 120}, "v1")
    result = db.get_latest_home_screen(uid, "self")
    assert result is not None
    assert result["band"] == "LOW"
    assert result["inputs"]["sbp"] == 120

    db.clear_bp_readings(uid, "self")
    assert db.get_bp_readings(uid, "self") == []

    db.delete_home_data(uid)
    assert db.get_latest_home_screen(uid, "self") is None

    _cleanup(uid)


def test_screening_db_consent_and_records():
    _cleanup()
    db.register_user("Scr Test", TEST_EMAIL, "pw", "Female", 28)
    _, user = db.login_user(TEST_EMAIL, "pw")
    uid = user["id"]

    # consent
    assert not sc.has_consent(uid)
    sc.record_consent(uid)
    assert sc.has_consent(uid)
    sc.record_consent(uid)   # upsert path
    assert sc.has_consent(uid)

    # save screening records
    sc.save_screening(uid, "LOW",    {"bp": 118, "note": "self"},   person_label="")
    sc.save_screening(uid, "MEDIUM", {"bp": 145, "note": "parent"}, person_label="Mother")

    rows = sc.list_screenings(uid, limit=10)
    assert len(rows) == 2
    assert isinstance(rows[0]["payload"], dict)
    assert isinstance(rows[0]["created_at"], str)
    assert rows[0]["band"] == "MEDIUM"
    assert rows[0]["person_label"] == "Mother"
    assert rows[1]["person_label"] == ""

    d = sc.last_self_screening_date(uid)
    assert d is not None

    sc.delete_screening_data(uid)
    assert sc.list_screenings(uid) == []

    _cleanup(uid)


def test_user_scoping():
    """A user must not be able to read another user's assessments."""
    _cleanup()
    db.register_user("User A", TEST_EMAIL, "pw", "Male", 30)
    _, userA = db.login_user(TEST_EMAIL, "pw")
    uid_a = userA["id"]

    # use a different email for user B
    email_b = "_pytest_db_userB@nephrosense.local"
    conn = db.get_conn()
    conn.execute("DELETE FROM users WHERE email=?", (email_b,))
    conn.commit()
    conn.close()

    db.register_user("User B", email_b, "pw", "Female", 25)
    _, userB = db.login_user(email_b, "pw")
    uid_b = userB["id"]

    db.save_assessment(uid_a, 55.0, 2, "MEDIUM")
    db.save_assessment(uid_b, 20.0, 1, "LOW")

    hist_a = db.get_user_history(uid_a)
    hist_b = db.get_user_history(uid_b)

    assert all(r["user_id"] == uid_a for r in hist_a), "User A sees other user's records"
    assert all(r["user_id"] == uid_b for r in hist_b), "User B sees other user's records"
    assert len(hist_a) == 1
    assert len(hist_b) == 1

    # cleanup
    conn = db.get_conn()
    for uid in (uid_a, uid_b):
        conn.execute("DELETE FROM assessments WHERE user_id=?", (uid,))
        conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.execute("DELETE FROM users WHERE email=?", (email_b,))
    conn.commit()
    conn.close()
    _cleanup()


if __name__ == "__main__":
    tests = [
        test_backend_is_sqlite,
        test_encryption_roundtrip,
        test_all_tables_exist,
        test_register_and_login,
        test_encrypted_fields_at_rest,
        test_save_and_retrieve_assessments,
        test_consent_and_home_screening,
        test_screening_db_consent_and_records,
        test_user_scoping,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{'ALL PASSED' if not failed else f'{failed} FAILED'} ({passed}/{len(tests)})")
