"""
testfree_risk.py — NephroSense
Test-free CKD risk pre-screen (no blood / urine / imaging inputs).

Model: logistic regression trained on NHANES by src/nhanes_ckd_model.py.
Target: a kidney-damage marker on a single visit (eGFR < 60 by CKD-EPI 2021
or urine ACR >= 30 mg/g). This is NOT a confirmed CKD diagnosis (KDIGO needs
the abnormality to persist > 3 months).

Rules enforced here:
  * Nothing is ever filled in silently. If a required input is missing or
    out of range the function REFUSES to score and says exactly why.
  * Output is a band (LOW / POSSIBLE / HIGH) with reasons — no exact
    percentage is shown. Band cut-offs come from the training data
    (stored in the model bundle), not from guesses.
"""
import math
import os

import joblib

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "testfree_model.joblib"
)

FEATURES = ["age", "female", "sbp", "bmi", "waist_cm",
            "diabetes", "hypertension", "heart_disease", "kidney_stones"]

# plain-language labels used for "reasons"
LABELS = {
    "age": "Age", "female": "Sex", "sbp": "Systolic blood pressure",
    "bmi": "Body-mass index", "waist_cm": "Waist size",
    "diabetes": "Diabetes", "hypertension": "High blood pressure diagnosis",
    "heart_disease": "Heart disease / heart failure / heart attack",
    "kidney_stones": "Past kidney stones",
}

RANGES = {"age": (20, 100), "sbp": (70, 260), "bmi": (12, 70), "waist_cm": (40, 200)}
BINARY = ["diabetes", "hypertension", "heart_disease", "kidney_stones"]

BAND_ORDER = {"LOW": 0, "POSSIBLE": 1, "HIGH": 2}
STRIP_POSITIVE = {"positive", "trace", "1+", "2+", "3+"}


def egfr_ckd_epi_2021(creatinine_mg_dl, age, female):
    """CKD-EPI 2021 (race-free) creatinine equation, mL/min/1.73 m²."""
    k = 0.7 if female else 0.9
    a = -0.241 if female else -0.302
    r = creatinine_mg_dl / k
    val = 142 * (min(r, 1) ** a) * (max(r, 1) ** -1.200) * (0.9938 ** age)
    return val * 1.012 if female else val


def load_bundle(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _refuse(missing=None, invalid=None):
    parts = []
    if missing:
        parts.append("missing: " + ", ".join(LABELS.get(m, m) for m in missing))
    if invalid:
        parts.append("not usable: " + "; ".join(invalid))
    return {"scored": False, "band": None, "reasons": [], "missing": missing or [],
            "invalid": invalid or [],
            "message": "Not enough reliable information to score — " + " | ".join(parts)
                       + ". Nothing has been assumed for you."}


def _clean(inputs):
    """Returns (values dict, missing list, invalid list). No defaults, no clipping."""
    vals, missing, invalid = {}, [], []
    sex = str(inputs.get("sex", "")).strip().lower()
    if sex.startswith("f"):
        vals["female"] = 1.0
    elif sex.startswith("m"):
        vals["female"] = 0.0
    else:
        missing.append("female")

    for k in ("age", "sbp", "bmi", "waist_cm"):
        v = inputs.get(k)
        if v is None or v == "":
            missing.append(k)
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            invalid.append(f"{LABELS[k]} is not a number")
            continue
        lo, hi = RANGES[k]
        if not (lo <= v <= hi) or math.isnan(v):
            invalid.append(f"{LABELS[k]} {v:g} is outside {lo}-{hi}")
            continue
        vals[k] = v

    for k in BINARY:
        v = inputs.get(k)
        if isinstance(v, str):
            v = {"yes": 1, "no": 0}.get(v.strip().lower())
        if v in (True, 1):
            vals[k] = 1.0
        elif v in (False, 0):
            vals[k] = 0.0
        else:
            missing.append(k)
    return vals, missing, invalid


def _reasons(bundle, vals):
    pipe = bundle["pipeline"]
    scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    out = []
    for i, f in enumerate(bundle["features"]):
        z = (vals[f] - scaler.mean_[i]) / scaler.scale_[i]
        c = float(clf.coef_[0][i] * z)          # log-odds vs. the average NHANES adult
        out.append((f, c))
    out.sort(key=lambda t: t[1], reverse=True)
    reasons = []
    for f, c in out:
        if c <= 0.10:
            continue
        if f in BINARY:
            if vals[f] != 1.0:
                continue
            txt, value = LABELS[f], None
        elif f == "female":
            txt, value = "Sex", None
        else:
            txt, value = f"{LABELS[f]} ({vals[f]:g})", vals[f]
        reasons.append({"key": f, "value": value, "factor": txt,
                        "direction": "raises", "strength": round(c, 2)})
    return reasons[:5]


def score(inputs, bundle=None, strip_result=None):
    """
    inputs: age, sex, sbp (home 7-day average), bmi, waist_cm, diabetes,
            hypertension, heart_disease, kidney_stones.
    strip_result: optional home urine-albumin strip ("negative"/"positive"/...).
    """
    bundle = bundle or load_bundle()
    if bundle is None:
        return {"scored": False, "band": None, "reasons": [], "missing": [], "invalid": [],
                "message": "Model file not found. Run: python src/nhanes_ckd_model.py"}

    vals, missing, invalid = _clean(inputs)
    if missing or invalid:
        return _refuse(missing, invalid)

    p = float(bundle["pipeline"].predict_proba([[vals[f] for f in bundle["features"]]])[0][1])
    if p < bundle["t_low"]:
        band = "LOW"
    elif p < bundle["t_high"]:
        band = "POSSIBLE"
    else:
        band = "HIGH"

    reasons = _reasons(bundle, vals)
    notes = []
    strip = str(strip_result).strip().lower() if strip_result else None
    if strip in STRIP_POSITIVE:
        reasons.insert(0, {"key": "strip", "value": None, "factor": "Home urine-albumin strip was positive",
                           "direction": "raises", "strength": None})
        notes.append("A positive home strip can only raise the band, never lower it. "
                     "Confirm with a lab urine ACR test.")
        if band == "LOW":
            band = "POSSIBLE"
    elif strip == "negative":
        notes.append("A negative home strip does not lower the band — strips miss early albuminuria.")

    info = bundle["band_info"][band]
    return {
        "scored": True, "band": band, "reasons": reasons, "missing": [], "invalid": [],
        "notes": notes,
        "band_stats": info,   # real NHANES numbers for people in this band (see model card)
        "raw_probability": p,  # internal use (validation only) — do not display
        "disclaimer": "Risk pre-screen only, not a diagnosis. A kidney blood test (creatinine/eGFR) "
                      "and urine ACR are needed to know for sure.",
    }
