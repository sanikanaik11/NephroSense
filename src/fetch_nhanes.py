"""
fetch_nhanes.py — NephroSense  (REPLACE existing)
Downloads NHANES 2017-2018 (CDC) and converts it to the SAME units/meaning
as the UCI + Bangladesh CKD data. The old version mixed units, which made
healthy people look high-risk:

  FIXED  bp   : now DIASTOLIC (UCI 'bp' is diastolic ~80; old code used systolic ~120)
  FIXED  wbcc : now cells/cumm (NHANES gives thousands -> x1000)
  FIXED  bu   : now UREA (NHANES gives BUN -> x2.14; UCI 'bu' is urea)
  FIXED  al   : from real urine ACR (ALB_CR_J) instead of blank
  FIXED  target: KDIGO definition  eGFR < 60  OR  ACR >= 30 mg/g
  Adults (18+) only. Columns NHANES doesn't have are filled with neutral/
  physiological values (sg=1.020, cad=0, appet=good, pe=0, su from glucose>180,
  ane from WHO haemoglobin cut-offs) — not random numbers.

Run (needs internet), then retrain:
    python src/fetch_nhanes.py
    python src/train_ckd_model.py
"""
import pandas as pd
import numpy as np
import requests
import io
import os

BASE = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/"
FILES = {
    "demo": "DEMO_J.XPT", "bp": "BPX_J.XPT", "biochem": "BIOPRO_J.XPT",
    "cbc": "CBC_J.XPT", "diab": "DIQ_J.XPT", "bpq": "BPQ_J.XPT", "acr": "ALB_CR_J.XPT",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nhanes_ckd.csv")


def fetch(name, fname):
    print(f"Downloading {name} ...")
    resp = requests.get(BASE + fname, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    if b"HEADER RECORD" not in resp.content[:300].upper():
        raise ValueError(f"{name}: not a real XPT file (network/firewall may be blocking wwwn.cdc.gov).")
    return pd.read_sas(io.BytesIO(resp.content), format="xport")


def egfr_ckdepi_2021(age, female, scr):
    if pd.isna(age) or pd.isna(female) or pd.isna(scr) or scr <= 0:
        return np.nan
    k, a, f = (0.7, -0.241, 1.012) if female else (0.9, -0.302, 1.0)
    x = scr / k
    return 142 * (min(x, 1) ** a) * (max(x, 1) ** -1.2) * (0.9938 ** age) * f


def main():
    try:
        demo = fetch("demo", FILES["demo"])[["SEQN", "RIDAGEYR", "RIAGENDR"]]

        bp = fetch("bp", FILES["bp"])
        di_cols = [c for c in bp.columns if c.startswith("BPXDI")]
        bp[di_cols] = bp[di_cols].replace(0, np.nan)          # 0 = failed reading
        bp["bp"] = bp[di_cols].mean(axis=1)                   # DIASTOLIC, matches UCI
        bp = bp[["SEQN", "bp"]]

        bio = fetch("biochem", FILES["biochem"])[["SEQN", "LBXSCR", "LBXSGL", "LBXSBU", "LBXSNASI", "LBXSKSI"]]
        bio.columns = ["SEQN", "sc", "bgr", "bu_bun", "sod", "pot"]

        cbc = fetch("cbc", FILES["cbc"])[["SEQN", "LBXHGB", "LBXHCT", "LBXWBCSI", "LBXRBCSI"]]
        cbc.columns = ["SEQN", "hemo", "pcv", "wbcc", "rbcc"]

        diab = fetch("diab", FILES["diab"])[["SEQN", "DIQ010"]]
        diab.columns = ["SEQN", "dm_raw"]
        bpq = fetch("bpq", FILES["bpq"])[["SEQN", "BPQ020"]]
        bpq.columns = ["SEQN", "htn_raw"]
    except Exception as e:
        print(f"\nFAILED to download NHANES: {e}")
        print("Skipping NHANES — training will continue with only UCI + Bangladesh.")
        return

    try:
        acr = fetch("acr", FILES["acr"])[["SEQN", "URDACT"]]
        acr.columns = ["SEQN", "acr"]
    except Exception as e:
        print(f"  (ACR file unavailable: {e}) — target will use eGFR only.")
        acr = pd.DataFrame({"SEQN": demo["SEQN"], "acr": np.nan})

    df = (demo.merge(bp, on="SEQN").merge(bio, on="SEQN").merge(cbc, on="SEQN")
              .merge(diab, on="SEQN", how="left").merge(bpq, on="SEQN", how="left")
              .merge(acr, on="SEQN", how="left"))
    df = df.rename(columns={"RIDAGEYR": "age"})
    df = df[df["age"] >= 18].copy()

    df["dm"] = df["dm_raw"].map({1: 1, 2: 0}).fillna(0)
    df["htn"] = df["htn_raw"].map({1: 1, 2: 0}).fillna(0)

    # ── unit alignment with UCI ──
    df["bu"] = df["bu_bun"] * 2.14          # BUN -> urea
    df["wbcc"] = df["wbcc"] * 1000.0        # 10^3/uL -> cells/cumm

    # ── target (KDIGO: eGFR<60 or albuminuria >=30 mg/g) ──
    female = (df["RIAGENDR"] == 2)
    df["egfr"] = [egfr_ckdepi_2021(a, f, s) for a, f, s in zip(df["age"], female, df["sc"])]
    df = df.dropna(subset=["egfr"])
    female = (df["RIAGENDR"] == 2)
    df["target"] = ((df["egfr"] < 60) | (df["acr"].fillna(0) >= 30)).astype(int)

    # ── urine albumin on the UCI 0-5 scale from real ACR ──
    a = df["acr"]
    df["al"] = np.select([a < 30, a < 100, a < 300, a < 1000], [0, 1, 2, 3], default=4)
    df.loc[a.isna(), "al"] = 0

    # ── neutral / physiological fills for fields NHANES lacks ──
    df["sg"] = 1.020
    df["su"] = (df["bgr"] > 180).astype(int)          # glycosuria above renal threshold
    df["cad"] = 0
    df["appet"] = 1
    df["pe"] = 0
    df["ane"] = (df["hemo"] < np.where(female, 12.0, 13.0)).astype(int)   # WHO cut-offs

    keep = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv",
            "wbcc", "rbcc", "htn", "dm", "cad", "appet", "pe", "ane", "target"]
    df = df[keep]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} real NHANES adult records -> {OUT_PATH}")
    print(f"CKD positive: {int(df['target'].sum())}, negative: {int((df['target'] == 0).sum())}")
    print(f"Sanity check — median diastolic BP {df['bp'].median():.0f}, urea {df['bu'].median():.0f}, WBC {df['wbcc'].median():.0f}")


if __name__ == "__main__":
    main()