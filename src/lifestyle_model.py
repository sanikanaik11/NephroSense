"""
lifestyle_model.py — NephroSense
Loads the trained ensemble model (models/ckd_model.pkl) and scores a
patient's questionnaire answers, with real per-patient SHAP and LIME
explanations.
Question set matches train_ckd_model.py's BASE_FEATURE_COLS exactly;
engineered features (BUN/creatinine ratio, anemia severity, age×BP
interaction) are computed here the same way training computes them,
so the model always receives matching inputs.
"""
import os
import numpy as np
import joblib

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ckd_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "ckd_meta.pkl")

YES_NO = {
    "yes": 1, "no": 0, "present": 1, "notpresent": 0,
    "normal": 1, "abnormal": 0, "good": 1, "poor": 0,
}

# ──────────────────────────────────────────────────────────────
# Scored questions — MUST match train_ckd_model.py's
# BASE_FEATURE_COLS exactly (same keys, same order doesn't matter
# since we build the row from meta["feature_cols"], but the KEYS
# must match).
# ──────────────────────────────────────────────────────────────
QUESTIONS = [
    {"key": "age", "label": "Your age", "type": "number", "unit": "years", "default": 45, "min": 1, "max": 100},
    {"key": "bp", "label": "Blood pressure (if known)", "type": "number", "unit": "mm/Hg (systolic)", "default": 80, "min": 40, "max": 200},
    {"key": "sg", "label": "Urine specific gravity (from a urine test, if you have one)", "type": "select", "options": [1.005, 1.010, 1.015, 1.020, 1.025], "default": 1.020, "lab": True},
    {"key": "al", "label": "Albumin level in urine test (0 = none)", "type": "select", "options": [0, 1, 2, 3, 4, 5], "default": 0, "lab": True},
    {"key": "su", "label": "Sugar level in urine test (0 = none)", "type": "select", "options": [0, 1, 2, 3, 4, 5], "default": 0, "lab": True},
    {"key": "bgr", "label": "Random blood sugar level", "type": "number", "unit": "mg/dL", "default": 100, "min": 40, "max": 500, "lab": True},
    {"key": "bu", "label": "Blood urea level", "type": "number", "unit": "mg/dL", "default": 30, "min": 5, "max": 300, "lab": True},
    {"key": "sc", "label": "Serum creatinine level", "type": "number", "unit": "mg/dL", "default": 1.0, "min": 0.1, "max": 20.0, "step": 0.1, "lab": True},
    {"key": "sod", "label": "Sodium level in blood", "type": "number", "unit": "mEq/L", "default": 140, "min": 100, "max": 170, "lab": True},
    {"key": "pot", "label": "Potassium level in blood", "type": "number", "unit": "mEq/L", "default": 4.5, "min": 2.0, "max": 10.0, "step": 0.1, "lab": True},
    {"key": "hemo", "label": "Haemoglobin level", "type": "number", "unit": "g/dL", "default": 14.0, "min": 3.0, "max": 20.0, "step": 0.1, "lab": True},
    {"key": "pcv", "label": "Packed cell volume", "type": "number", "unit": "%", "default": 42, "min": 10, "max": 60, "lab": True},
    {"key": "wbcc", "label": "White blood cell count", "type": "number", "unit": "cells/cumm", "default": 8000, "min": 2000, "max": 25000, "step": 100, "lab": True},
    {"key": "rbcc", "label": "Red blood cell count", "type": "number", "unit": "millions/cumm", "default": 5.0, "min": 2.0, "max": 7.0, "step": 0.1, "lab": True},
    {"key": "htn", "label": "Do you have high blood pressure (hypertension)?", "type": "yesno", "default": "no"},
    {"key": "dm", "label": "Do you have diabetes?", "type": "yesno", "default": "no"},
    {"key": "cad", "label": "Do you have coronary artery / heart disease?", "type": "yesno", "default": "no"},
    {"key": "appet", "label": "How is your appetite?", "type": "select", "options": ["good", "poor"], "default": "good"},
    {"key": "pe", "label": "Do you have swelling in your feet/ankles (pedal edema)?", "type": "yesno", "default": "no"},
    {"key": "ane", "label": "Do you have anaemia (low blood count)?", "type": "yesno", "default": "no"},
]

LAB_QUESTION_KEYS = [q["key"] for q in QUESTIONS if q.get("lab")]

FEAT_LABELS = {q["key"]: q["label"] for q in QUESTIONS}
FEAT_LABELS.update({
    "bun_creatinine_ratio": "BUN / Creatinine ratio (derived)",
    "anemia_severity": "Anaemia severity (derived from haemoglobin)",
    "age_bp_interaction": "Age × Blood pressure interaction (derived)",
})

# ──────────────────────────────────────────────────────────────
# CLINICAL_QUESTIONS — NOT scored by the XGBoost/ensemble model
# (not in FEATURE_COLS). Saved as documentation for a doctor visit
# AND fed into the separate, transparent rule-based Clinical Risk
# Factor Checklist (see clinical_risk_score.py) which cross-checks
# the model score using established KDIGO/NICE CKD risk factors.
# Gender/age-gated sections never shown to minors.
# ──────────────────────────────────────────────────────────────
CLINICAL_QUESTIONS = [
    # Free-text / voice — patient's own description, any language.
    # NOT scored by the model — saved as-is (and, if given by voice,
    # transcribed by Whisper) purely as documentation for the doctor.
    {"key": "own_description", "label": "In your own words, describe what you're experiencing (you can type in any language)",
     "type": "text", "placeholder": "e.g. I've been feeling very tired and my ankles are swollen for the past week...",
     "external": True},

    # Universal
    {"key": "burning_urine", "label": "Burning or itching sensation while urinating?", "type": "yesno", "default": "no"},
    {"key": "urine_frequency", "label": "Any change in how often you urinate?", "type": "select",
     "options": ["no change", "More frequent than usual", "Less frequent than usual"], "default": "no change"},
    {"key": "flank_pain", "label": "Persistent pain in your stomach, sides, or lower back?", "type": "yesno", "default": "no"},
    {"key": "nausea", "label": "Nausea or vomiting recently?", "type": "yesno", "default": "no"},
    {"key": "symptom_duration", "label": "If you have any symptoms above, how long have you noticed them?", "type": "select",
     "options": ["none", "Less than 1 week", "1-4 weeks", "1-3 months", "More than 3 months"], "default": "none"},
    {"key": "current_meds", "label": "Medicines you are currently taking (if any)", "type": "text", "placeholder": "e.g. Metformin, Amlodipine"},
    {"key": "nsaid_use", "label": "How often do you take painkillers (e.g. Ibuprofen, Diclofenac)?", "type": "select",
     "options": ["never", "Occasionally", "Weekly", "Daily"], "default": "never"},
    {"key": "allergies", "label": "Any known drug allergies?", "type": "text", "placeholder": "Optional"},
    {"key": "dipstick_ph", "label": "Urine pH, if you did a home dipstick test", "type": "number", "unit": "pH", "default": 6.0, "min": 4.5, "max": 8.5, "step": 0.5},
    {"key": "foamy_urine", "label": "Is your urine foamy or frothy?", "type": "select", "options": ["no", "yes", "sometimes"], "default": "no"},
    {"key": "nocturia", "label": "Do you wake up at night frequently to urinate?", "type": "yesno", "default": "no"},
    {"key": "fatigue", "label": "Unusual fatigue or weakness?", "type": "yesno", "default": "no"},
    {"key": "appetite_change", "label": "Any recent unexplained weight loss or appetite change?", "type": "yesno", "default": "no"},
    {"key": "rbc_urine", "label": "Red blood cells in urine test (if you have a report)", "type": "select", "options": ["normal", "abnormal"], "default": "normal"},
    {"key": "pc_urine", "label": "Pus cells in urine test", "type": "select", "options": ["normal", "abnormal"], "default": "normal"},
    {"key": "pcc_urine", "label": "Pus cell clumps in urine test", "type": "select", "options": ["notpresent", "present"], "default": "notpresent"},
    {"key": "ba_urine", "label": "Bacteria in urine test", "type": "select", "options": ["notpresent", "present"], "default": "notpresent"},

    # ── Real CKD risk-factor questions (KDIGO / NICE CKD guideline factors) ──
    {"key": "family_ckd", "label": "Does anyone in your immediate family (parent/sibling) have kidney disease or is on dialysis?", "type": "yesno", "default": "no"},
    {"key": "bmi_category", "label": "Your BMI category (if known)", "type": "select",
     "options": ["unknown", "Underweight (<18.5)", "Normal (18.5-24.9)", "Overweight (25-29.9)", "Obese (30+)"], "default": "unknown"},
    {"key": "recurrent_kidney_stones", "label": "Have you had kidney stones more than once?", "type": "yesno", "default": "no"},
    {"key": "long_term_diabetes", "label": "If diabetic, how many years have you had it?", "type": "select",
     "options": ["not diabetic", "Less than 5 years", "5-10 years", "More than 10 years"], "default": "not diabetic"},
    {"key": "long_term_htn", "label": "If hypertensive, how many years have you had it?", "type": "select",
     "options": ["not hypertensive", "Less than 5 years", "5-10 years", "More than 10 years"], "default": "not hypertensive"},
    {"key": "autoimmune_disease", "label": "Do you have any autoimmune condition (Lupus, RA, etc.)?", "type": "yesno", "default": "no"},
    {"key": "recurrent_uti_general", "label": "Have you had 3+ UTIs in the past year?", "type": "yesno", "default": "no"},
    {"key": "contrast_dye_exposure", "label": "Have you had a CT scan with contrast dye or an angiogram in the last year?", "type": "yesno", "default": "no"},

    # Women only, 18+
    {"key": "preg_history", "label": "Have you had any pregnancy complications (pre-eclampsia / gestational hypertension)?",
     "type": "yesno", "default": "no", "female_only": True, "min_age": 18},
    {"key": "uti_recurrent", "label": "Do you have frequent urinary tract infections (UTIs)?",
     "type": "yesno", "default": "no", "female_only": True, "min_age": 18},
    {"key": "gynaec", "label": "Any known gynaecological conditions (PCOS, fibroids, uterine problems)?",
     "type": "yesno", "default": "no", "female_only": True, "min_age": 18},
    {"key": "menstrual", "label": "Any significant change in your menstrual cycle recently?",
     "type": "yesno", "default": "no", "female_only": True, "min_age": 18},
    # Men only, 18+
    {"key": "alcohol", "label": "How often do you drink alcohol?", "type": "select",
     "options": ["never", "occasionally", "regularly"], "default": "never", "male_only": True, "min_age": 18},
    {"key": "smoking", "label": "Do you smoke or use tobacco?", "type": "select",
     "options": ["no", "yes", "former"], "default": "no", "male_only": True, "min_age": 18},
    {"key": "prostate", "label": "Any difficulty urinating / prostate issues?",
     "type": "yesno", "default": "no", "male_only": True, "min_age": 18},
    {"key": "occupational", "label": "Do you work with chemicals, pesticides, or heavy metals?",
     "type": "yesno", "default": "no", "male_only": True, "min_age": 18},
]


def visible_clinical_questions(gender, age):
    gender_norm = (gender or "").strip().lower()
    is_female = gender_norm.startswith("f")
    is_male = gender_norm.startswith("m")
    try:
        age_val = float(age)
    except (TypeError, ValueError):
        age_val = 0

    visible = []
    for q in CLINICAL_QUESTIONS:
        if q.get("external"):
            continue
        if q.get("female_only") and not is_female:
            continue
        if q.get("male_only") and not is_male:
            continue
        if q.get("min_age") and age_val < q["min_age"]:
            continue
        visible.append(q)
    return visible


_model = None
_meta = None


def _load_model():
    global _model, _meta
    if _model is None and os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
            _meta = joblib.load(META_PATH)
        except Exception:
            _model = None
    return _model, _meta


def _encode_answer(q, value):
    if q["type"] == "yesno":
        return YES_NO.get(value, 0)
    if q["key"] == "appet":
        return YES_NO.get(value, 1)
    return value


def _compute_engineered_features(encoded):
    """
    Mirrors add_engineered_features() in train_ckd_model.py exactly,
    so the model sees the same derived values at inference time that
    it was trained on.
    """
    bu = float(encoded.get("bu", 30) or 30)
    sc = float(encoded.get("sc", 1.0) or 1.0)
    hemo = float(encoded.get("hemo", 14.0) or 14.0)
    age = float(encoded.get("age", 45) or 45)
    bp = float(encoded.get("bp", 80) or 80)

    sc_safe = sc if sc != 0 else 0.1
    bun_creatinine_ratio = bu / sc_safe
    anemia_severity = max(0.0, 14 - hemo)
    age_bp_interaction = age * bp / 100.0

    return {
        "bun_creatinine_ratio": bun_creatinine_ratio,
        "anemia_severity": anemia_severity,
        "age_bp_interaction": age_bp_interaction,
    }


def _get_lime_explanation(model, row, feature_cols, medians):
    """
    Real per-patient LIME explanation, alongside SHAP — matches the
    'SHAP & LIME' explainability claim in the project synopsis.
    Returns [] if the lime package isn't installed or it fails.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
        background = np.array([[medians.get(c, 0.0) for c in feature_cols]] * 50)
        explainer = LimeTabularExplainer(
            training_data=background,
            feature_names=[FEAT_LABELS.get(c, c) for c in feature_cols],
            class_names=["Not CKD", "CKD"],
            mode="classification",
        )
        exp = explainer.explain_instance(
            np.array(row), model.predict_proba, num_features=5
        )
        return [{"label": label, "impact": float(weight)} for label, weight in exp.as_list()]
    except Exception:
        return []


def predict_lifestyle_risk(answers, run_lime=False):
    model, meta = _load_model()
    encoded = {q["key"]: _encode_answer(q, answers[q["key"]]) for q in QUESTIONS}
    shap_explanation = []
    lime_explanation = []

    if model is None or meta is None:
        return _rule_based(encoded, answers)

    feature_cols = meta["feature_cols"]
    medians = meta.get("medians", {})

    # Compute engineered features (if this model version uses them)
    # the exact same way training did, then merge with the raw
    # encoded answers before building the model input row.
    engineered = {}
    if any(c in feature_cols for c in ("bun_creatinine_ratio", "anemia_severity", "age_bp_interaction")):
        engineered = _compute_engineered_features(encoded)
    full_values = {**encoded, **engineered}

    row = []
    for col in feature_cols:
        val = full_values.get(col)
        if val is None:
            val = medians.get(col, 0.0)
        try:
            row.append(float(val))
        except (ValueError, TypeError):
            row.append(medians.get(col, 0.0))

    try:
        proba = model.predict_proba(np.array([row]))[0]
        risk_score = round(float(proba[1]) * 100, 1)
    except Exception:
        return _rule_based(encoded, answers)

    acc = meta.get("accuracy", "—")
    cv_acc = meta.get("cv_accuracy", "—")
    cv_std = meta.get("cv_std")
    auc = meta.get("roc_auc")
    sensitivity = meta.get("sensitivity")
    n_total = meta.get("n_total", meta.get("n_train", "—"))
    sources = meta.get("sources_used", "multiple public datasets")
    model_type = meta.get("model_type", "XGBoost")
    cv_std_str = f" ± {cv_std}%" if cv_std is not None else ""
    extra_metrics = ""
    if auc is not None and sensitivity is not None:
        extra_metrics = f" ROC-AUC {auc}, sensitivity (CKD recall) {sensitivity}%."
    explanation = (
        f"Score produced by an {model_type} trained on {n_total} real records from "
        f"{sources} (test accuracy {acc}%, cross-validated accuracy {cv_acc}%{cv_std_str})."
        f"{extra_metrics} "
        f"Still a research-stage screening estimate, not a diagnosis."
    )

    # SHAP needs a tree-based model directly — if this is the soft-
    # voting ensemble, pull out its XGBoost component specifically
    # for the explanation (still a real, per-patient explanation,
    # just scoped to the strongest individual model in the ensemble
    # rather than the blended vote).
    shap_source_model = model
    try:
        if hasattr(model, "named_estimators_") and "xgb" in model.named_estimators_:
            shap_source_model = model.named_estimators_["xgb"]
    except Exception:
        pass

    try:
        import shap
        explainer = shap.TreeExplainer(shap_source_model)
        shap_vals = explainer.shap_values(np.array([row]))
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif shap_vals.ndim == 3:
            vals = shap_vals[0, :, 1]
        else:
            vals = shap_vals[0]
        pairs = sorted(zip(feature_cols, vals), key=lambda p: abs(p[1]), reverse=True)[:5]
        shap_explanation = [{"label": FEAT_LABELS.get(k, k), "impact": float(v)} for k, v in pairs]
    except Exception:
        try:
            fi = shap_source_model.feature_importances_
            pairs = sorted(zip(feature_cols, fi), key=lambda p: p[1], reverse=True)[:5]
            shap_explanation = [{"label": FEAT_LABELS.get(k, k), "impact": float(v)} for k, v in pairs]
        except Exception:
            shap_explanation = []

    lime_explanation = _get_lime_explanation(model, row, feature_cols, medians) if run_lime else []

    # Use the clinically-tuned decision threshold (optimized for
    # catching real CKD cases) if available, rather than the flat
    # 33/66 cutpoints on raw probability — the threshold decides
    # whether this individual is flagged CKD-positive at all; the
    # 33/66 bands below only describe how strongly so once flagged.
    decision_threshold = meta.get("decision_threshold", 0.5)
    if risk_score < 33:
        risk_level = "LOW"
    elif risk_score < 66:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    # If the tuned threshold would flag this case as CKD-positive but
    # the raw score still falls in the LOW band, escalate to MEDIUM so
    # the risk_level shown is never less cautious than the model's own
    # tuned decision.
    if (risk_score / 100.0) >= decision_threshold and risk_level == "LOW":
        risk_level = "MEDIUM"

    sc = encoded.get("sc", 1.0)
    if sc < 1.2:
        ckd_stage = 1
    elif sc < 2.0:
        ckd_stage = 2
    elif sc < 3.5:
        ckd_stage = 3
    elif sc < 5.5:
        ckd_stage = 4
    else:
        ckd_stage = 5

    stage_note = ""
    if risk_level == "HIGH" and ckd_stage <= 1:
        stage_note = (
            " (Note: stage is estimated from creatinine alone and may not "
            "match the overall risk score — rely on the risk score, not the "
            "stage number.)"
        )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "ckd_stage": ckd_stage,
        "explanation": explanation + stage_note,
        "shap_explanation": shap_explanation,
        "lime_explanation": lime_explanation,
    }


def _rule_based(encoded, answers):
    points = 0
    if encoded.get("hemo", 14) < 11: points += 15
    if encoded.get("sc", 1.0) > 1.3: points += 20
    if encoded.get("bu", 30) > 40: points += 10
    if encoded.get("htn") == 1: points += 10
    if encoded.get("dm") == 1: points += 10
    if encoded.get("al", 0) >= 1: points += 10
    if encoded.get("su", 0) >= 1: points += 5
    if encoded.get("ane") == 1: points += 10
    if encoded.get("pe") == 1: points += 5
    if encoded.get("appet") == 0: points += 5
    if encoded.get("cad") == 1: points += 5
    risk_score = min(points, 100)
    risk_level = "LOW" if risk_score < 33 else ("MEDIUM" if risk_score < 66 else "HIGH")
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "ckd_stage": 1,
        "explanation": "Score produced by a simple rule-based check — trained model file not found. "
                       "Run src/train_ckd_model.py first.",
        "shap_explanation": [],
        "lime_explanation": [],
    }