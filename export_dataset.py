"""
export_dataset.py — NephroSense
Exports all patient data as dataset files (CSV + Excel) you can open in Excel.
One row per assessment, with the questionnaire answers expanded into columns.
Passwords are NEVER exported. The file contains DECRYPTED health data — keep it private.

    python scripts\\export_dataset.py
Output: exports\\nephrosense_dataset.csv  (+ coloured .xlsx, users.csv)
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from database import get_conn, _decrypt, _USE_PG  # noqa: E402

ENC = ["blood_group", "diabetes", "hypertension", "heart_disease", "kidney_history", "uti_history"]


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


def main():
    if _USE_PG:
        raise SystemExit(
            "This root-level export script reads the local SQLite file directly.\n"
            "When DATABASE_URL is set (PostgreSQL backend), use scripts/export_dataset.py\n"
            "which connects to the correct database."
        )
    conn = get_conn()
    users = pd.read_sql_query("SELECT * FROM users", conn)
    asm = pd.read_sql_query("SELECT * FROM assessments ORDER BY user_id, created_at", conn)
    conn.close()

    users = users.drop(columns=["password"], errors="ignore")
    for c in ENC:
        if c in users:
            users[c] = users[c].apply(lambda v: _decrypt(v, default=""))
    users = users.rename(columns={"id": "user_id", "created_at": "registered_at"})

    rows = []
    for _, r in asm.iterrows():
        row = r.drop(labels=["clinical_notes", "lifestyle_answers"]).to_dict()
        row.update(_expand(r.get("lifestyle_answers"), "ans_"))
        row.update(_expand(r.get("clinical_notes"), "clin_"))
        rows.append(row)
    ds = pd.DataFrame(rows).rename(columns={"id": "assessment_id", "created_at": "assessed_at"})
    ds = ds.merge(users, on="user_id", how="left") if len(ds) else ds

    out = os.path.join(ROOT, "exports")
    os.makedirs(out, exist_ok=True)
    users.to_csv(os.path.join(out, "users.csv"), index=False, encoding="utf-8-sig")
    ds.to_csv(os.path.join(out, "nephrosense_dataset.csv"), index=False, encoding="utf-8-sig")
    try:
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter
        colours = {"LOW": ("C6EFCE", "63BE7B"), "MEDIUM": ("FFEB9C", "FFC000"), "HIGH": ("FFC7CE", "FF5A5A")}
        with pd.ExcelWriter(os.path.join(out, "nephrosense_dataset.xlsx"), engine="openpyxl") as w:
            ds.to_excel(w, sheet_name="assessments", index=False)
            users.to_excel(w, sheet_name="users", index=False)
            for name, ws in w.sheets.items():
                for c in ws[1]:
                    c.font = Font(bold=True, color="FFFFFF")
                    c.fill = PatternFill("solid", fgColor="1D4ED8")
                    c.alignment = Alignment(wrap_text=True, vertical="center")
                ws.freeze_panes = "A2"
                for i, col in enumerate(ws.columns, 1):
                    width = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:60])
                    ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 10), 40)
            ws = w.sheets["assessments"]
            heads = [c.value for c in ws[1]]
            if "risk_level" in heads:
                ci = heads.index("risk_level") + 1
                for row in ws.iter_rows(min_row=2):
                    lvl = str(row[ci - 1].value or "").strip().upper()
                    if lvl in colours:
                        light, strong = colours[lvl]
                        for cell in row:                      # whole row tinted
                            cell.fill = PatternFill("solid", fgColor=light)
                        row[ci - 1].fill = PatternFill("solid", fgColor=strong)   # risk cell stronger
                        row[ci - 1].font = Font(bold=True)
    except Exception as e:  # openpyxl missing
        print("Excel not written (pip install openpyxl):", e)
    print(f"{len(users)} users, {len(ds)} assessments -> {out}")


if __name__ == "__main__":
    main()
