"""
validate_app.py — NephroSense
Checks the app's test-free score against REAL measured kidney labs.

CSV columns (one row per person, 200-500 people):
  age, sex, sbp, bmi, waist_cm, diabetes, hypertension, heart_disease, kidney_stones,
  creatinine_mg_dl, acr_mg_g           (optionally egfr; if absent it is computed, CKD-EPI 2021)
Reference standard: eGFR < 60 OR ACR >= 30 mg/g (same definition as training).

    python scripts/validate_app.py my_people.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from testfree_risk import score, load_bundle, egfr_ckd_epi_2021, BAND_ORDER  # noqa: E402

MIN_N = 200


def validate(df, bundle=None, seed=42, n_boot=1000):
    bundle = bundle or load_bundle()
    if bundle is None:
        raise SystemExit("Model not found — train first.")
    if len(df) < MIN_N:
        raise SystemExit(f"Need at least {MIN_N} people; got {len(df)}. Result would not be reliable.")
    df = df.copy()
    if "egfr" not in df or df["egfr"].isna().all():
        df["egfr"] = [egfr_ckd_epi_2021(c, a, str(s).lower().startswith("f"))
                      for c, a, s in zip(df["creatinine_mg_dl"], df["age"], df["sex"])]
    df["truth"] = ((df["egfr"] < 60) | (df["acr_mg_g"] >= 30)).astype(int)

    res = [score(r.to_dict(), bundle) for _, r in df.iterrows()]
    ok = np.array([r["scored"] for r in res])
    d = df[ok].copy()
    d["band"] = [r["band"] for r, k in zip(res, ok) if k]
    d["p"] = [r["raw_probability"] for r, k in zip(res, ok) if k]
    n_ref = int(len(df) - ok.sum())
    y = d["truth"].to_numpy()
    if y.min() == y.max():
        raise SystemExit("Only one outcome class in the scored people; cannot compute AUC.")

    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], d["p"].to_numpy()[i]))

    def sens_spec(mask):
        return {"sensitivity": float((mask & (y == 1)).sum() / (y == 1).sum()),
                "specificity": float((~mask & (y == 0)).sum() / (y == 0).sum())}

    rank = d["band"].map(BAND_ORDER).to_numpy()
    out = {
        "n_people": int(len(df)), "n_scored": int(ok.sum()), "n_refused_incomplete": n_ref,
        "reference_prevalence": float(y.mean()),
        "auc": float(roc_auc_score(y, d["p"])),
        "auc_95ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "possible_or_high": sens_spec(rank >= 1),
        "high_only": sens_spec(rank >= 2),
        "per_band": {b: {"n": int((d["band"] == b).sum()),
                         "marker_rate": (float(y[d["band"] == b].mean()) if (d["band"] == b).any() else None)}
                     for b in BAND_ORDER},
    }
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    result = validate(pd.read_csv(sys.argv[1]))
    print(json.dumps(result, indent=2))
    with open("validation_report.json", "w") as f:
        json.dump(result, f, indent=2)
