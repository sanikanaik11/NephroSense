"""
clinical_risk_score.py — NephroSense  (REPLACE existing)
A transparent, rule-based clinical risk factor checklist based on
established CKD risk factors (KDIGO / NICE CKD guidelines: diabetes
duration, hypertension duration, family history, recurrent UTIs,
NSAID/contrast exposure, autoimmune disease, obesity).

NEW: also uses points the patient wrote in their own notes and CONFIRMED
(keys starting with "hx_", produced by history_extractor.py), e.g. family
history of diabetes / high blood pressure / kidney stones.

This is NOT a machine-learning prediction — every point maps to a named factor.
"""

RISK_FACTORS = [
    ("family_ckd", "yes", 15, "Family history of kidney disease"),
    ("recurrent_kidney_stones", "yes", 8, "Recurrent kidney stones"),
    ("long_term_diabetes", "More than 10 years", 20, "Diabetes >10 years"),
    ("long_term_diabetes", "5-10 years", 12, "Diabetes 5-10 years"),
    ("long_term_htn", "More than 10 years", 15, "Hypertension >10 years"),
    ("long_term_htn", "5-10 years", 8, "Hypertension 5-10 years"),
    ("autoimmune_disease", "yes", 12, "Autoimmune condition present"),
    ("recurrent_uti_general", "yes", 8, "Recurrent UTIs (3+/year)"),
    ("contrast_dye_exposure", "yes", 5, "Recent contrast dye / CT exposure"),
    ("nsaid_use", "Daily", 12, "Daily painkiller (NSAID) use"),
    ("nsaid_use", "Weekly", 6, "Weekly painkiller (NSAID) use"),
    ("bmi_category", "Obese (30+)", 8, "Obesity (BMI 30+)"),
    # ── from the patient's own confirmed notes ──
    ("hx_fam_diabetes", "yes", 5, "Family history of diabetes (from your notes)"),
    ("hx_fam_hypertension", "yes", 4, "Family history of high blood pressure (from your notes)"),
    ("hx_fam_kidney_stones", "yes", 3, "Family history of kidney stones (from your notes)"),
    ("hx_self_kidney_stones", "yes", 4, "Past kidney stones (from your notes)"),
    ("hx_fam_heart", "yes", 2, "Family history of heart disease (from your notes)"),
    # ── from scan reports confirmed by the patient (ultrasound / MRI / CT) ──
    ("img_polycystic", "yes", 15, "Polycystic kidneys on scan"),
    ("img_small_kidney", "yes", 12, "Small / shrunken kidney(s) on scan"),
    ("img_cortical_thinning", "yes", 10, "Kidney cortical thinning / scarring on scan"),
    ("img_echogenic_kidney", "yes", 8, "Increased kidney echogenicity (medical renal disease) on scan"),
    ("img_hydronephrosis", "yes", 8, "Hydronephrosis on scan"),
    ("img_solitary_kidney", "yes", 10, "Only one kidney on scan"),
    ("img_kidney_stone", "yes", 5, "Kidney stone(s) on scan"),
    ("img_enlarged_prostate", "yes", 3, "Enlarged prostate on scan"),
]


def calculate_clinical_risk_score(clinical_answers):
    """
    clinical_answers: dict of {key: answer} from CLINICAL_QUESTIONS (+ hx_* confirmed notes).
    Returns dict with score (0-100, capped), risk_level, and the exact
    list of factors that contributed — fully explainable, no hidden logic.
    """
    total = 0
    triggered = []

    for key, match_value, points, label in RISK_FACTORS:
        answer = clinical_answers.get(key, "")
        if str(answer).strip() == match_value:
            total += points
            triggered.append({"factor": label, "points": points})

    total = min(total, 100)

    if total < 20:
        level = "LOW"
    elif total < 45:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "clinical_risk_score": total,
        "clinical_risk_level": level,
        "triggered_factors": triggered,
        "note": "This is a transparent, rule-based checklist of established CKD "
                "risk factors (KDIGO/NICE guidelines) — not a machine-learning "
                "prediction. Shown separately from the AI pattern model.",
    }