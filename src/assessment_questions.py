"""
assessment_questions.py — NephroSense
The TEST-FREE Health Assessment question set. Nobody needs a blood or urine report to answer it.

  * Age is never asked — it is calculated from the date of birth (age_utils.py).
  * Questions depend on sex and age (men 45+: urine-flow questions; women of child-bearing age: pregnancy
    questions; everyone: the same kidney risk factors).
  * "Not sure" is a real answer. It is NOT turned into "no" silently: derive() lists every "not sure" so the
    result can say which items were not counted.
  * Required questions must be answered (no pre-selected answers). If one is missing the page refuses to score.

Pure Python (no Streamlit) so it can be unit-tested.
"""
from age_utils import calculate_age  # noqa: F401  (re-exported for the app)

YNU = ("yes", "no", "not sure")
ADULT_MIN_AGE = 18

# ── Questions ──────────────────────────────────────────────────────────────────────────────────
# kind: ynu | select | number | text.   required: must be answered.   sex: "f"/"m".   show_if: fn(answers)->bool
def _q(key, label, kind, **kw):
    d = {"key": key, "label": label, "kind": kind}
    d.update(kw)
    return d


YEARS = ["Less than 5 years", "5-10 years", "More than 10 years"]

SECTIONS = [
    ("🩸 Blood pressure", [
        _q("htn", "Has a doctor ever told you that you have high blood pressure (BP)?", "ynu", required=True),
        _q("bp_meds", "Do you take BP medicine every day?", "ynu", required=True, show_if=lambda a: a.get("htn") == "yes"),
        _q("htn_years", "How long have you had high BP?", "select", options=YEARS, required=True,
           show_if=lambda a: a.get("htn") == "yes"),
        _q("sbp", "Your latest BP reading you know — UPPER number (optional)", "number", min=70, max=250, step=1),
        _q("dbp", "Your latest BP reading — LOWER number (optional)", "number", min=40, max=150, step=1),
        _q("bp_when", "When was that BP reading taken?", "select",
           options=["This week", "In the last month", "1-12 months ago", "More than a year ago"], required=True,
           show_if=lambda a: a.get("sbp") is not None or a.get("dbp") is not None),
    ]),
    ("🍬 Sugar (diabetes)", [
        _q("dm", "Has a doctor ever told you that you have diabetes (sugar)?", "ynu", required=True),
        _q("dm_years", "How long have you had diabetes?", "select", options=YEARS, required=True,
           show_if=lambda a: a.get("dm") == "yes"),
    ]),
    ("❤️ Heart and blood vessels", [
        _q("cad", "Have you had a heart attack, angina (heart pain), bypass/stent or heart failure?", "ynu",
           required=True),
        _q("stroke", "Have you ever had a stroke (paralysis / speech loss from a blood-vessel problem)?", "ynu"),
    ]),
    ("⚖️ Body measurements", [
        _q("height", "Height (cm)", "number", required=True, min=100, max=230, step=1),
        _q("weight", "Weight (kg)", "number", required=True, min=20, max=250, step=0.5),
        _q("waist", "Waist around the navel (cm) — optional", "number", min=40, max=200, step=1),
    ]),
    ("👪 Family and past history", [
        _q("known_ckd", "Has a doctor ever told YOU that you have a kidney problem (kidney disease, protein in urine)?",
           "ynu", required=True),
        _q("family_ckd", "Has a parent, brother or sister had kidney disease or needed dialysis?", "ynu", required=True),
        _q("fam_dm", "Diabetes in a parent, brother or sister?", "ynu"),
        _q("fam_htn", "High BP in a parent, brother or sister?", "ynu"),
        _q("stones", "Have you had kidney stones?", "select", required=True,
           options=["Never", "Once", "More than once"]),
        _q("uti3", "Have you had 3 or more urine infections (UTI) in the last year?", "ynu"),
        _q("autoimmune", "Do you have lupus, rheumatoid arthritis or another autoimmune disease?", "ynu"),
        _q("ane", "Have you ever been told you have anaemia (low haemoglobin / low blood)?", "ynu"),
        _q("contrast", "In the last year, did you have a CT scan with contrast dye or an angiogram?", "ynu"),
    ]),
    ("💊 Medicines and habits", [
        _q("nsaid", "How often do you take painkillers such as ibuprofen, diclofenac or aceclofenac?", "select",
           required=True, options=["Never", "Occasionally", "Weekly", "Daily"]),
        _q("herbal", "Do you regularly take herbal, ayurvedic or home-remedy medicines?", "ynu"),
        _q("tobacco", "Tobacco (cigarette, bidi, gutkha, khaini)?", "select", options=["Never", "Former", "Current"]),
        _q("alcohol", "Alcohol?", "select", options=["Never", "Occasionally", "Regularly"]),
        _q("occupational", "Do you work in farming, or with pesticides, chemicals or heavy metals?", "ynu"),
        _q("current_meds", "Names of medicines you take daily (optional — read from the strip or prescription)", "text"),
    ]),
    ("🧍 Men — urine flow", [
        _q("prostate", "Weak urine stream, dribbling, difficulty starting, or feeling the bladder is not empty?",
           "ynu", sex="m", min_age=45),
    ]),
    ("🤰 Women — pregnancy", [
        _q("preg_now", "Are you pregnant now?", "ynu", sex="f", min_age=18, max_age=55, required=True),
        _q("preg_hx", "In an earlier pregnancy, did you have high BP, pre-eclampsia or swelling with high BP?", "ynu",
           sex="f", min_age=18, max_age=60),
    ]),
]

# Symptoms (tick boxes). `flag` = also a red-flag key for triage.py.
SYMPTOMS = [
    # key,          label,                                              flag
    ("little_urine", "Passing very little or no urine", "little_urine"),
    ("breathless", "Breathless at rest or while lying down", "breathless"),
    ("chest_pain", "Chest pain or pressure", "chest_pain"),
    ("confusion", "Confusion, fits or unusual drowsiness", "confusion"),
    ("swelling", "Swelling of feet, ankles or face", "swelling"),
    ("vomiting", "Repeated vomiting", "vomiting"),
    ("blood_urine", "Blood in urine, or tea / cola coloured urine", "blood_urine"),
    ("foamy", "Foamy or frothy urine", None),
    ("more_urine", "Passing urine more often than usual", None),
    ("night_urine", "Waking 2 or more times at night to pass urine", None),
    ("tired", "Unusual tiredness or weakness", None),
    ("poor_appetite", "Poor appetite or weight loss without trying", None),
    ("nausea", "Nausea", None),
    ("itch", "Itching of the skin all over", None),
    ("flank", "Pain in the sides or lower back", None),
    ("burning", "Burning while passing urine", None),
]
FLAG_KEYS = {k: f for k, _l, f in SYMPTOMS if f}
SYMPTOM_DURATIONS = ["Less than 1 week", "1-4 weeks", "1-3 months", "More than 3 months"]

REQUIRED_NOTE = "Please answer every question marked *."


# ── Visibility ─────────────────────────────────────────────────────────────────────────────────
def sex_code(gender):
    g = str(gender or "").strip().lower()
    return "f" if g.startswith("f") else ("m" if g.startswith("m") else None)


def applicable(q, sex, age, answers):
    if q.get("sex") and q["sex"] != sex:
        return False
    if q.get("min_age") and (age is None or age < q["min_age"]):
        return False
    if q.get("max_age") and (age is None or age > q["max_age"]):
        return False
    if q.get("show_if") and not q["show_if"](answers or {}):
        return False
    return True


def visible_sections(sex, age, answers=None):
    """[(title, [question, ...])] for this person; empty sections are dropped. `answers` drives follow-ups."""
    out = []
    for title, qs in SECTIONS:
        vis = [q for q in qs if applicable(q, sex, age, answers)]
        if vis:
            out.append((title, vis))
    return out


def _empty(v):
    return v is None or (isinstance(v, str) and not v.strip())


def missing_required(answers, sex, age):
    """Labels of required, applicable questions that are still unanswered."""
    out = []
    for _t, qs in SECTIONS:
        for q in qs:
            if q.get("required") and applicable(q, sex, age, answers) and _empty((answers or {}).get(q["key"])):
                out.append(q["label"])
    return out


def bmi_value(height_cm, weight_kg):
    try:
        h, w = float(height_cm), float(weight_kg)
    except (TypeError, ValueError):
        return None
    if not (100 <= h <= 230 and 20 <= w <= 250):
        return None
    return round(w / ((h / 100.0) ** 2), 1)


def bmi_category(bmi):
    if bmi is None:
        return "unknown"
    if bmi < 18.5:
        return "Underweight (<18.5)"
    if bmi < 25:
        return "Normal (18.5-24.9)"
    if bmi < 30:
        return "Overweight (25-29.9)"
    return "Obese (30+)"


def red_flags(symptoms):
    """Triage keys (see triage.py) for the ticked symptoms."""
    return [FLAG_KEYS[s] for s in (symptoms or []) if s in FLAG_KEYS]


def stop_reason(answers, sex, age):
    """'pregnant' / 'minor' / None — cases where this screening must not give a score."""
    if age is not None and age < ADULT_MIN_AGE:
        return "minor"
    if sex == "f" and (answers or {}).get("preg_now") == "yes":
        return "pregnant"
    return None


# ── Mapping to the keys the existing engines use ───────────────────────────────────────────────
def _yn(v):
    return "yes" if v == "yes" else "no"


def derive(raw, symptoms, symptom_duration, sex, age):
    """raw: {question key: answer or None}.  symptoms: list of ticked symptom keys.
    Returns dict(model_answers, clinical_answers, unsure, notes, bmi).
      model_answers    -> the keys lifestyle_model / risk_engine already read (htn, dm, cad, pe, ane, appet, bp, age)
      clinical_answers -> the keys clinical_risk_score / disease_predictor already read
      unsure           -> questions answered "not sure" (NOT counted as risk factors — shown to the user)
      notes            -> plain-language notes about data that was not used"""
    raw = raw or {}
    sym = set(symptoms or [])
    labels = {q["key"]: q["label"] for _t, qs in SECTIONS for q in qs}
    unsure = [labels[k] for k, v in raw.items() if v == "not sure" and k in labels]
    notes = []

    bmi = bmi_value(raw.get("height"), raw.get("weight"))
    sbp, dbp = raw.get("sbp"), raw.get("dbp")
    recent = raw.get("bp_when") in ("This week", "In the last month", "1-12 months ago")
    reading_ok = sbp is not None and dbp is not None and dbp < sbp and recent
    if (sbp is not None or dbp is not None) and not reading_ok:
        notes.append("Your BP reading was not used (it needs both numbers, the upper larger than the lower, "
                     "and taken in the last 12 months).")
    if raw.get("htn") == "yes" and not reading_ok:
        notes.append("No recent BP reading was given. A home BP reading (morning and evening for 7 days) "
                     "makes the result more reliable.")

    model_answers = {
        "age": age,
        "bp": float(dbp) if reading_ok else 80.0,         # only read when blood-test data is also used
        "htn": _yn(raw.get("htn")),
        "dm": _yn(raw.get("dm")),
        "cad": _yn(raw.get("cad")),
        "appet": "poor" if "poor_appetite" in sym else "good",
        "pe": "yes" if "swelling" in sym else "no",
        "ane": _yn(raw.get("ane")),
    }
    if reading_ok:
        model_answers["sbp"] = float(sbp)

    stones = raw.get("stones")
    c = {
        "burning_urine": "yes" if "burning" in sym else "no",
        "urine_frequency": ("Less frequent than usual" if "little_urine" in sym
                            else "More frequent than usual" if "more_urine" in sym else "no change"),
        "flank_pain": "yes" if "flank" in sym else "no",
        "nausea": "yes" if ("nausea" in sym or "vomiting" in sym) else "no",
        "symptom_duration": (symptom_duration if (sym and symptom_duration) else "none"),
        "current_meds": (raw.get("current_meds") or "").strip(),
        "nsaid_use": raw.get("nsaid") or "not answered",
        "foamy_urine": "yes" if "foamy" in sym else "no",
        "nocturia": "yes" if "night_urine" in sym else "no",
        "fatigue": "yes" if "tired" in sym else "no",
        "appetite_change": "yes" if "poor_appetite" in sym else "no",
        "family_ckd": _yn(raw.get("family_ckd")),
        "bmi_category": bmi_category(bmi),
        "recurrent_kidney_stones": "yes" if stones == "More than once" else "no",
        "long_term_diabetes": (raw.get("dm_years") or "not answered") if raw.get("dm") == "yes" else "not diabetic",
        "long_term_htn": (raw.get("htn_years") or "not answered") if raw.get("htn") == "yes" else "not hypertensive",
        "autoimmune_disease": _yn(raw.get("autoimmune")),
        "recurrent_uti_general": _yn(raw.get("uti3")),
        "contrast_dye_exposure": _yn(raw.get("contrast")),
        "alcohol": (raw.get("alcohol") or "not answered").lower(),
        "smoking": {"Never": "no", "Former": "former", "Current": "yes"}.get(raw.get("tobacco"), "not answered"),
        "occupational": _yn(raw.get("occupational")),
        "prostate": "yes" if (sex == "m" and raw.get("prostate") == "yes") else "no",
        "preg_history": "yes" if (sex == "f" and raw.get("preg_hx") == "yes") else "no",
        "uti_recurrent": "yes" if (sex == "f" and raw.get("uti3") == "yes") else "no",
        "gynaec": "no", "menstrual": "no",
        # facts from the patient's own answers that clinical_risk_score reads as "hx_" keys
        "hx_fam_diabetes": _yn(raw.get("fam_dm")),
        "hx_fam_hypertension": _yn(raw.get("fam_htn")),
        "hx_self_kidney_stones": "yes" if stones in ("Once", "More than once") else "no",
        # recorded for the doctor; not scored
        "stroke": _yn(raw.get("stroke")),
        "herbal_meds": _yn(raw.get("herbal")),
        "known_ckd": _yn(raw.get("known_ckd")),
        "itching": "yes" if "itch" in sym else "no",
        "waist_cm": raw.get("waist"),
        "bmi": bmi,
    }
    if unsure:
        notes.append("You answered \"not sure\" to: " + "; ".join(unsure) +
                     ". These were NOT counted as risk factors, so the result may be an under-estimate.")
    if raw.get("known_ckd") == "yes":
        notes.append("You said a doctor has already told you about a kidney problem. This pre-screen is not a "
                     "substitute for follow-up — please keep your regular nephrology / doctor visits.")
    return {"model_answers": model_answers, "clinical_answers": c, "unsure": unsure, "notes": notes, "bmi": bmi}