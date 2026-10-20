"""
nhanes_ckd_model.py — NephroSense
Trains the TEST-FREE kidney-risk model on NHANES (free, CDC) and writes
models/testfree_model.joblib + models/MODEL_CARD.md. Every number in the card
is computed here from the data; nothing is typed in by hand.

Run (needs internet once; files are cached in data/nhanes/):
    pip install scikit-learn pandas numpy joblib
    python src/nhanes_ckd_model.py

Cycles: 2015-16 (suffix I) and 2017-18 (suffix J).
Inputs (all asked without any test): age, sex, systolic BP, BMI, waist,
diabetes, told-high-BP, heart disease (CHF/CHD/heart attack), kidney stones.
Target (single visit, measured in the NHANES lab): eGFR < 60 (CKD-EPI 2021,
serum creatinine) OR urine ACR >= 30 mg/g.
NOT in the model, because NHANES has no usable variable: family history of
kidney disease and NSAID use. They stay as separate checklist items.
"""
import os
import sys
import urllib.request
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from testfree_risk import FEATURES, egfr_ckd_epi_2021  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "nhanes")
MODEL_OUT = os.path.join(ROOT, "models", "testfree_model.joblib")
CARD_OUT = os.path.join(ROOT, "models", "MODEL_CARD.md")
CYCLES = {"2015-2016": "I", "2017-2018": "J"}
FILES = ["DEMO", "BPX", "BMX", "DIQ", "BPQ", "MCQ", "KIQ_U", "BIOPRO", "ALB_CR"]
SEED = 42


def _get(cycle, suffix, name):
    os.makedirs(CACHE, exist_ok=True)
    fn = f"{name}_{suffix}.XPT"
    path = os.path.join(CACHE, fn)
    if not os.path.exists(path):
        url = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{fn}"
        print("downloading", url)
        urllib.request.urlretrieve(url, path)
    return pd.read_sas(path, format="xport", encoding="utf-8")


def _yn(s):
    """1->1, 2->0, anything else (7/9/blank) -> NaN. Never guessed."""
    return s.map({1: 1.0, 2: 0.0})


def load_cycle(cycle, suffix):
    t = {n: _get(cycle, suffix, n).set_index("SEQN") for n in FILES}
    d = t["DEMO"][["RIAGENDR", "RIDAGEYR"]].copy()
    d["female"] = (d["RIAGENDR"] == 2).astype(float)
    d["age"] = d["RIDAGEYR"]
    bp = t["BPX"][[c for c in ("BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4") if c in t["BPX"]]]
    d["sbp"] = bp.mean(axis=1)                       # mean of the measured readings
    d["bmi"] = t["BMX"]["BMXBMI"]
    d["waist_cm"] = t["BMX"]["BMXWAIST"]
    d["diabetes"] = _yn(t["DIQ"]["DIQ010"])          # "borderline" (3) -> NaN -> row dropped
    d["hypertension"] = _yn(t["BPQ"]["BPQ020"])
    h = pd.concat([_yn(t["MCQ"][c]) for c in ("MCQ160B", "MCQ160C", "MCQ160E")], axis=1)
    d["heart_disease"] = np.where(h.max(axis=1) == 1, 1.0, np.where(h.notna().all(axis=1), 0.0, np.nan))
    d["kidney_stones"] = _yn(t["KIQ_U"]["KIQ026"])
    scr, acr = t["BIOPRO"]["LBXSCR"], t["ALB_CR"]["URDACT"]
    d = d.join(scr).join(acr, how="inner").dropna(subset=["LBXSCR", "URDACT"])
    d["egfr"] = [egfr_ckd_epi_2021(c, a, f == 1.0) for c, a, f in zip(d["LBXSCR"], d["age"], d["female"])]
    d["target"] = ((d["egfr"] < 60) | (d["URDACT"] >= 30)).astype(int)
    d["cycle"] = cycle
    return d


def build_dataset():
    df = pd.concat([load_cycle(c, s) for c, s in CYCLES.items()])
    n_raw = len(df)
    df = df[df["age"] >= 20]
    n_adult = len(df)
    df = df.dropna(subset=FEATURES + ["target"])
    return df, {"n_adults_with_labs": int(n_adult), "n_complete": int(len(df)), "n_raw": int(n_raw)}


def _pipe():
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])


def fit_bundle(df, n_splits=5, seed=SEED, sens_target=0.90, spec_target=0.90):
    """Trains + cross-validates. Returns the bundle saved to disk."""
    X, y = df[FEATURES].to_numpy(float), df["target"].to_numpy(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_auc = []
    for tr, te in skf.split(X, y):
        m = _pipe().fit(X[tr], y[tr])
        fold_auc.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    oof = cross_val_predict(_pipe(), X, y, cv=skf, method="predict_proba")[:, 1]

    fpr, tpr, thr = roc_curve(y, oof)
    t_low = float(thr[np.argmax(tpr >= sens_target)])             # Low band keeps >=90% of cases above it
    t_high = float(thr[np.where(fpr <= 1 - spec_target)[0][-1]])  # High band: >=90% specificity
    if not t_low < t_high:
        raise RuntimeError("Band cut-offs collapsed; the model is not separating the groups.")

    def metrics_at(t):
        pred = oof >= t
        return {"sensitivity": float((pred & (y == 1)).sum() / (y == 1).sum()),
                "specificity": float((~pred & (y == 0)).sum() / (y == 0).sum())}

    bands = {"LOW": oof < t_low, "POSSIBLE": (oof >= t_low) & (oof < t_high), "HIGH": oof >= t_high}
    band_info = {b: {"n": int(m.sum()), "share_of_people": float(m.mean()),
                     "kidney_marker_rate": float(y[m].mean()) if m.any() else None}
                 for b, m in bands.items()}

    final = _pipe().fit(X, y)
    return {
        "pipeline": final, "features": FEATURES, "t_low": t_low, "t_high": t_high,
        "band_info": band_info,
        "metrics": {
            "n": int(len(y)), "prevalence": float(y.mean()),
            "cv_auc_mean": float(np.mean(fold_auc)), "cv_auc_std": float(np.std(fold_auc)),
            "cv_auc_pooled": float(roc_auc_score(y, oof)),
            "at_low_cutoff (Possible or High)": metrics_at(t_low),
            "at_high_cutoff (High only)": metrics_at(t_high),
            "folds": n_splits,
        },
        "trained_on": date.today().isoformat(),
    }


def write_card(bundle, counts, path=CARD_OUT):
    m, bi = bundle["metrics"], bundle["band_info"]
    lo, hi = m["at_low_cutoff (Possible or High)"], m["at_high_cutoff (High only)"]
    coefs = bundle["pipeline"].named_steps["clf"].coef_[0]
    sc = bundle["pipeline"].named_steps["scaler"].scale_
    rows = "\n".join(f"| {f} | {c / s:+.4f} |" for f, c, s in zip(bundle["features"], coefs, sc))
    txt = f"""# NephroSense test-free risk model — model card
Generated {bundle['trained_on']} by src/nhanes_ckd_model.py. All figures computed from the data below.

## Purpose
Home pre-screen: who should get a kidney blood test (creatinine/eGFR) and urine ACR. **Not a diagnosis.**

## Data
NHANES 2015-16 and 2017-18 (US CDC, free). Adults >= 20 with lab results: {counts['n_adults_with_labs']}.
Complete cases used: **{m['n']}** (rows with any missing input are dropped, never filled).
Unweighted (NHANES survey weights not applied).

## Target
eGFR < 60 mL/min/1.73 m² (CKD-EPI 2021, serum creatinine) **or** urine ACR >= 30 mg/g, on one visit.
Prevalence in training data: {m['prevalence']:.1%}. A single-visit marker, not confirmed CKD (needs > 3 months).

## Inputs
{', '.join(bundle['features'])}. Not included (no usable NHANES variable): family history of kidney disease, NSAID use.

## Model
Logistic regression on standardised inputs (transparent, per-person reasons come from its coefficients).

| Input | Coefficient per unit (log-odds) |
|---|---|
{rows}

## Performance ({m['folds']}-fold stratified cross-validation, out-of-fold predictions)
- AUC: **{m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f}** across folds (pooled {m['cv_auc_pooled']:.3f})
- Flagging Possible-or-High: sensitivity **{lo['sensitivity']:.1%}**, specificity **{lo['specificity']:.1%}**
- Flagging High only: sensitivity **{hi['sensitivity']:.1%}**, specificity **{hi['specificity']:.1%}**

## Bands (cut-offs chosen from the data: Low keeps ~90% sensitivity, High keeps ~90% specificity)
| Band | People | Share | Had a kidney marker |
|---|---|---|---|
""" + "\n".join(
        f"| {b} | {bi[b]['n']} | {bi[b]['share_of_people']:.1%} | "
        f"{'n/a' if bi[b]['kidney_marker_rate'] is None else format(bi[b]['kidney_marker_rate'], '.1%')} |"
        for b in ("LOW", "POSSIBLE", "HIGH")) + """

## Limitations
- US population; **not yet validated on Indian adults.** Run scripts/validate_app.py on real local data before relying on it.
- NHANES BP is clinic-measured; the app uses a 7-day home average.
- Label is a one-time lab marker; transient albuminuria is included.
- A LOW band does not rule out kidney disease. Early CKD usually has no symptoms.
- Red-flag symptoms override every band (see src/triage.py).
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def main():
    df, counts = build_dataset()
    print(counts)
    bundle = fit_bundle(df)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(bundle, MODEL_OUT)
    write_card(bundle, counts)
    print(bundle["metrics"])
    print("saved", MODEL_OUT, "and", CARD_OUT)


if __name__ == "__main__":
    main()
