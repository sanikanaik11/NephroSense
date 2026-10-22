"""
export_dataset.py — NephroSense
Exports patient data (name, phone, email, city, health answers, scores) as dataset files.
One row per assessment; questionnaire answers expanded into columns; rows coloured by risk level.
Passwords are NEVER exported. The files contain DECRYPTED health data — keep them private.

Usage
  python scripts\\export_dataset.py                       -> this project's database
  python scripts\\export_dataset.py <db or folder> ...    -> ALSO merge other copies of the app, e.g.
  python scripts\\export_dataset.py "D:\\nephrosense" "E:\\friend1\\nephrosense\\database\\nephrosense.db"

For each copy you need its  database\\nephrosense.db  AND  database\\enc.key  (same folder).
Output: exports\\nephrosense_dataset.xlsx / .csv  and  users.csv  (column "source" shows which copy a row came from)
"""
import json
import os
import sqlite3
import sys

from datetime import datetime, date

import pandas as pd
from cryptography.fernet import Fernet, InvalidToken

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(ROOT, "database", "nephrosense.db")
ENC = ["blood_group", "diabetes", "hypertension", "heart_disease", "kidney_history", "uti_history"]


LOCAL_TZ = datetime.now().astimezone().tzinfo


def _to_local(series):
    """SQLite CURRENT_TIMESTAMP is UTC -> show this computer's local time."""
    dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return dt.dt.strftime("%Y-%m-%d %H:%M:%S")


def _age_on(dob, when):
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y"):
        try:
            b = datetime.strptime(str(dob).strip(), fmt).date()
        except ValueError:
            continue
        a = when.year - b.year - ((when.month, when.day) < (b.month, b.day))
        return a if 0 <= a <= 120 else None
    return None


def _find_db(p):
    p = os.path.abspath(p.strip().strip('"'))
    if os.path.isfile(p):
        return p
    for cand in (os.path.join(p, "database", "nephrosense.db"), os.path.join(p, "nephrosense.db")):
        if os.path.isfile(cand):
            return cand
    return None


def _decryptor(db_path):
    key_path = os.path.join(os.path.dirname(db_path), "enc.key")
    f = Fernet(open(key_path, "rb").read()) if os.path.exists(key_path) else None
    if f is None:
        print(f"  ! no enc.key next to {db_path} — encrypted fields will stay blank")

    def dec(v):
        if v is None or v == "":
            return ""
        if f is None:
            return ""
        try:
            return f.decrypt(str(v).encode()).decode()
        except (InvalidToken, ValueError):
            return v   # older plain-text row
    return dec


def _expand(raw, prefix):
    try:
        d = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return {}
    out = {}
    for k, v in (d or {}).items():
        if k.startswith("_") and k not in ("_have_blood", "_have_urine", "_blood_date", "_urine_date"):
            continue
        out[f"{prefix}{k.lstrip('_')}"] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
    return out


def load_one(db_path):
    _folder = os.path.dirname(os.path.dirname(db_path))
    source = (os.path.splitdrive(_folder)[0] + " " + os.path.basename(_folder)).strip() or db_path
    dec = _decryptor(db_path)
    conn = sqlite3.connect(db_path)
    users = pd.read_sql_query("SELECT * FROM users", conn)
    try:
        asm = pd.read_sql_query("SELECT * FROM assessments ORDER BY user_id, created_at", conn)
    except Exception:
        asm = pd.DataFrame()
    try:
        lg = pd.read_sql_query("SELECT user_id, logged_in_at FROM login_log", conn)
    except Exception:
        lg = pd.DataFrame(columns=["user_id", "logged_in_at"])
    conn.close()

    users = users.drop(columns=["password"], errors="ignore")
    for c in ENC:
        if c in users:
            users[c] = users[c].apply(dec)
    users = users.rename(columns={"id": "user_id", "created_at": "registered_at"})
    users["registered_at"] = _to_local(users["registered_at"])
    if "created_at" in asm:
        asm["created_at"] = _to_local(asm["created_at"])
    if "phone" in users:   # keep mobile numbers as text (no 9.87E+09 in Excel)
        users["phone"] = users["phone"].astype("string").str.replace(r"\.0$", "", regex=True)
    users.insert(0, "source", source)

    rows = []
    for _, r in asm.iterrows():
        row = r.drop(labels=[c for c in ("clinical_notes", "lifestyle_answers") if c in r]).to_dict()
        row.update(_expand(r.get("lifestyle_answers"), "ans_"))
        row.update(_expand(r.get("clinical_notes"), "clin_"))
        rows.append(row)
    ds = pd.DataFrame(rows).rename(columns={"id": "assessment_id", "created_at": "assessed_at"})
    if len(ds):
        ds.insert(0, "source", source)
    # ---- login events: registration + every recorded sign-in (+ the last sign-in of older accounts) ----
    ev = []
    logged = set(lg["user_id"]) if len(lg) else set()
    for _, r in lg.iterrows():
        ev.append((r["user_id"], pd.to_datetime(r["logged_in_at"], errors="coerce"), "Login"))
    for _, u in users.iterrows():
        ev.append((u["user_id"], pd.to_datetime(u["registered_at"], errors="coerce"), "Registered"))
        if u["user_id"] not in logged and str(u.get("last_login") or "").strip():
            t = pd.to_datetime(str(u["last_login"]), format="%d %b %Y, %I:%M %p", errors="coerce")
            if pd.notna(t):
                ev.append((u["user_id"], t, "Login (last recorded)"))
    events = pd.DataFrame(ev, columns=["user_id", "login_time", "event"]).dropna(subset=["login_time"])
    events.insert(0, "source", source)
    print(f"  {source}: {len(users)} registered users, {len(ds)} assessments, {len(events)} login events  ({db_path})")
    return users, ds, events


NICE = {
    "user_id": "Patient ID", "full_name": "Full Name", "email": "Email", "phone": "Mobile", "city": "City",
    "gender": "Gender", "age": "Age", "dob": "Date of Birth", "role": "Account Type", "registered_at": "Registered On",
    "last_login": "Last Login", "assessment_id": "Assessment ID", "assessed_at": "Assessment Date",
    "risk_level": "Risk Level", "risk_score": "Risk Score (0-100)", "ckd_stage": "CKD Stage (estimate)",
    "lifestyle_score": "Questionnaire Score", "eye_score": "Eye Check (reference only)", "voice_score": "Voice Score",
    "egfr": "eGFR", "egfr_stage": "eGFR Stage", "blood_group": "Blood Group", "diabetes": "Diabetes",
    "hypertension": "Hypertension", "heart_disease": "Heart Disease", "kidney_history": "Kidney History",
    "uti_history": "UTI History", "smoking": "Smoking", "water_intake": "Water Intake", "source": "Source",
    "extra_data": "Extra Notes",
}
ANS = {"age": "Age (answered)", "bp": "Diastolic BP", "sg": "Urine Specific Gravity", "al": "Urine Albumin", "su": "Urine Sugar",
       "bgr": "Blood Glucose", "bu": "Blood Urea", "sc": "Serum Creatinine", "sod": "Sodium", "pot": "Potassium",
       "hemo": "Haemoglobin", "pcv": "Packed Cell Volume", "wbcc": "WBC Count", "rbcc": "RBC Count",
       "htn": "High BP (answered)", "dm": "Diabetes (answered)", "cad": "Heart Disease (answered)",
       "appet": "Appetite", "pe": "Pedal Oedema", "ane": "Anaemia"}
YESNO_USER = ["diabetes", "hypertension", "heart_disease", "kidney_history", "uti_history"]
ORDER = ["full_name", "email", "phone", "age", "gender", "city", "dob",
         "assessed_at", "risk_level", "risk_score", "ckd_stage", "lifestyle_score", "eye_score", "voice_score",
         "egfr", "egfr_stage", "blood_group", "diabetes", "hypertension", "heart_disease", "kidney_history",
         "uti_history", "smoking", "water_intake", "registered_at", "last_login", "user_id", "assessment_id"]
ALWAYS_FRONT = {"full_name", "email", "phone", "age", "gender", "city", "dob", "assessed_at", "risk_level", "risk_score"}


def _pretty(col):
    if col in NICE:
        return NICE[col]
    if col.startswith("ans_"):
        k = col[4:]
        return ANS.get(k, k.replace("_", " ").title())
    if col.startswith("clin_"):
        return col[5:].replace("_", " ").title()
    return col.replace("_", " ").title()


def _unique_map(cols, namer):
    """column -> display name, never two identical headers (e.g. profile 'Smoking' vs answered 'Smoking')."""
    seen, out = {}, {}
    for c in cols:
        n = namer(c)
        if n in seen:
            n = f"{n} (assessment)" if c.startswith(("ans_", "clin_")) else f"{n} (profile)"
        seen[n] = c
        out[c] = n
    return out


def main():
    targets = sys.argv[1:] or [DEFAULT_DB]
    all_users, all_ds, all_ev = [], [], []
    for t in targets:
        db = _find_db(t)
        if not db:
            print(f"  ! database not found for: {t}")
            continue
        u, d, e = load_one(db)
        all_users.append(u)
        all_ds.append(d)
        all_ev.append(e)
    if not all_users:
        raise SystemExit("No database found.")
    users = pd.concat(all_users, ignore_index=True)
    asm = pd.concat([d for d in all_ds if len(d)], ignore_index=True) if any(len(d) for d in all_ds) else pd.DataFrame()

    users = users.sort_values("registered_at").drop_duplicates(subset=["email"], keep="last").reset_index(drop=True)
    if len(asm):
        asm = asm.merge(users[["source", "user_id"]], on=["source", "user_id"], how="inner")   # only kept people
    users["age"] = [(_age_on(d, date.today()) if _age_on(d, date.today()) is not None else a)
                    for d, a in zip(users["dob"], users["age"])]
    for c in YESNO_USER:
        if c in users:
            users[c] = users[c].map(lambda v: {"1": "Yes", "0": "No"}.get(str(v).strip(), v))
    if len(asm):
        _key = [c for c in ("source", "user_id", "assessed_at", "risk_score", "lifestyle_score", "eye_score", "voice_score")
                if c in asm.columns]
        asm = asm.drop_duplicates(subset=_key).reset_index(drop=True)
    # EVERY registered person appears, even with no assessment yet
    base = users.copy()
    full = base.merge(asm, on=["source", "user_id"], how="left") if len(asm) else base.assign(assessment_id=pd.NA)
    sort_cols = [c for c in ("source", "registered_at", "user_id", "assessed_at") if c in full]
    full = full.sort_values(sort_cols).reset_index(drop=True)

    multi = full["source"].nunique() > 1
    extras = [c for c in full.columns if c not in ORDER and c not in ("source", "extra_data", "password", "role")
              and not c.startswith(("ans_", "clin_"))]
    ans_cols = sorted(c for c in full.columns if c.startswith("ans_"))
    clin_cols = sorted(c for c in full.columns if c.startswith("clin_"))
    def _filled(c):
        return int(full[c].replace("", pd.NA).notna().sum())
    ordered = [c for c in ORDER if c in full.columns] + extras
    answer_cols = sorted(ans_cols + clin_cols, key=lambda c: (-_filled(c), c))
    cols = ordered + answer_cols
    blank = [c for c in cols if _filled(c) == 0 and c not in ALWAYS_FRONT]
    cols = [c for c in cols if c not in blank] + blank + (["source"] if multi else [])
    dataset = full[cols].rename(columns=_unique_map(cols, _pretty))

    # Patients sheet: ONE ROW PER PERSON = profile + their LATEST answers (every question they filled)
    lf = full.sort_values([c for c in ("source", "user_id", "assessed_at") if c in full]) \
             .groupby(["source", "user_id"], as_index=False).tail(1).copy()
    counts = asm.groupby(["source", "user_id"]).size().rename("n_assess").reset_index() if len(asm) else None
    lf = lf.merge(counts, on=["source", "user_id"], how="left") if counts is not None else lf.assign(n_assess=0)
    lf["n_assess"] = lf["n_assess"].fillna(0).astype(int)
    front = ["full_name", "email", "phone", "age", "gender", "city", "dob", "registered_at", "last_login",
             "n_assess", "risk_level", "assessed_at", "risk_score", "ckd_stage", "lifestyle_score", "eye_score",
             "voice_score", "egfr", "egfr_stage", "blood_group", "diabetes", "hypertension", "heart_disease",
             "kidney_history", "uti_history", "smoking", "water_intake"]
    pcols = [c for c in front if c in lf.columns] + sorted(
        [c for c in lf.columns if c.startswith(("ans_", "clin_"))],
        key=lambda c: (-int(lf[c].replace("", pd.NA).notna().sum()), c))
    pblank = [c for c in pcols if int(lf[c].replace("", pd.NA).notna().sum()) == 0 and c not in ALWAYS_FRONT]
    pcols = [c for c in pcols if c not in pblank] + pblank + (["source"] if multi else [])
    pnames = {"n_assess": "No. of Assessments", "risk_level": "Latest Risk Level",
              "assessed_at": "Latest Assessment Date", "risk_score": "Latest Risk Score (0-100)"}
    patients = lf[pcols].rename(columns=_unique_map(pcols, lambda c: pnames.get(c, _pretty(c)))) \
                        .sort_values("Full Name").reset_index(drop=True)

    # ---- LOGIN HISTORY: one row per login, with that person's answers AS OF that login ----
    events = pd.concat(all_ev, ignore_index=True)
    events = events.merge(users[["source", "user_id", "email"]], on=["source", "user_id"], how="inner")
    events = events.drop_duplicates(subset=["email", "login_time", "event"]).sort_values("login_time")
    if len(asm):
        a = asm.copy()
        a["_t"] = pd.to_datetime(a["assessed_at"] if "assessed_at" in a else a["created_at"], errors="coerce")
        a = a.dropna(subset=["_t"]).sort_values("_t")
        a = a.rename(columns={"id": "assessment_id", "created_at": "assessed_at"}) if "id" in a else a
        logins = pd.merge_asof(events, a, left_on="login_time", right_on="_t", by=["source", "user_id"],
                               direction="backward").drop(columns=["_t"])
    else:
        logins = events.copy()
    ucols = [c for c in users.columns if c not in ("age", "registered_at", "last_login")]
    logins = logins.merge(users[ucols + ["age"]].drop(columns=[c for c in ucols if c in logins.columns
                                                              and c not in ("source", "user_id")]),
                          on=["source", "user_id"], how="left", suffixes=("", "_u"))
    logins["age"] = [(_age_on(d, t.date()) if _age_on(d, t.date()) is not None else a_)
                     for d, t, a_ in zip(logins["dob"], logins["login_time"], logins["age"])]
    logins["login_time"] = logins["login_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    lfront = ["login_time", "event", "full_name", "email", "phone", "age", "gender", "city", "dob",
              "assessed_at", "risk_level", "risk_score", "ckd_stage", "lifestyle_score", "eye_score", "voice_score",
              "egfr", "egfr_stage", "blood_group", "diabetes", "hypertension", "heart_disease", "kidney_history",
              "uti_history", "smoking", "water_intake", "user_id", "assessment_id"]
    lcols = [c for c in lfront if c in logins.columns]
    lans = sorted([c for c in logins.columns if c.startswith(("ans_", "clin_"))],
                  key=lambda c: (-int(logins[c].replace("", pd.NA).notna().sum()), c))
    lcols += lans
    lblank = [c for c in lcols if int(logins[c].replace("", pd.NA).notna().sum()) == 0
              and c not in ALWAYS_FRONT and c not in ("login_time", "event")]
    lcols = [c for c in lcols if c not in lblank] + lblank + (["source"] if multi else [])
    lnames = {"login_time": "Login Time", "event": "Event", "assessed_at": "Latest Assessment Date",
              "risk_level": "Risk Level", "risk_score": "Risk Score (0-100)"}
    login_sheet = logins[lcols].rename(columns=_unique_map(lcols, lambda c: lnames.get(c, _pretty(c))))

    out = os.path.join(ROOT, "exports")
    os.makedirs(out, exist_ok=True)
    dataset.to_csv(os.path.join(out, "nephrosense_dataset.csv"), index=False, encoding="utf-8-sig")
    patients.to_csv(os.path.join(out, "patients.csv"), index=False, encoding="utf-8-sig")
    login_sheet.to_csv(os.path.join(out, "login_history.csv"), index=False, encoding="utf-8-sig")
    xlsx_path = os.path.join(out, "nephrosense_dataset.xlsx")
    try:                                   # file open in Excel? then save under a new name instead of failing
        if os.path.exists(xlsx_path):
            open(xlsx_path, "ab").close()
    except PermissionError:
        from datetime import datetime
        xlsx_path = os.path.join(out, f"nephrosense_dataset_{datetime.now():%H%M%S}.xlsx")
        print(f"  ! nephrosense_dataset.xlsx is open in Excel — saving as {os.path.basename(xlsx_path)} instead")
    try:
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        colours = {"LOW": ("C6EFCE", "00B050"), "MEDIUM": ("FFEB9C", "FFC000"), "HIGH": ("FFC7CE", "FF0000")}
        thin = Side(style="thin", color="D9D9D9")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            login_sheet.to_excel(w, sheet_name="Login History", index=False)
            patients.to_excel(w, sheet_name="Patients", index=False)
            dataset.to_excel(w, sheet_name="Dataset", index=False)
            for ws in w.sheets.values():
                for c in ws[1]:
                    c.font = Font(bold=True, color="FFFFFF")
                    c.fill = PatternFill("solid", fgColor="1F3864")
                    c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
                ws.row_dimensions[1].height = 32
                ws.freeze_panes = "C2"
                ws.auto_filter.ref = ws.dimensions
                for i, col in enumerate(ws.columns, 1):
                    width = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:80])
                    ws.column_dimensions[get_column_letter(i)].width = min(max(width + 3, 11), 38)
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
                heads = [c.value for c in ws[1]]
                for name in ("Risk Level", "Latest Risk Level"):
                    if name in heads:
                        ci = heads.index(name)
                        for row in ws.iter_rows(min_row=2):
                            lvl = str(row[ci].value or "").strip().upper()
                            if lvl in colours:
                                light, strong = colours[lvl]
                                for cell in row:
                                    cell.fill = PatternFill("solid", fgColor=light)
                                row[ci].fill = PatternFill("solid", fgColor=strong)
                                row[ci].font = Font(bold=True, color="FFFFFF" if lvl != "MEDIUM" else "000000")
                                row[ci].alignment = Alignment(horizontal="center")
        print("Excel file saved:", xlsx_path)
    except ImportError as e:
        print("Excel not written — run: pip install openpyxl", e)
    except Exception as e:
        print("Excel not written:", e)
    print(f"TOTAL: {len(patients)} registered people, {len(asm)} assessments -> {out}")


if __name__ == "__main__":
    main()
