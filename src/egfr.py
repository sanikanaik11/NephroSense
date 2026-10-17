"""
egfr.py — NephroSense
Standard CKD-EPI 2021 (race-free) creatinine equation for estimated GFR.
"""

def calculate_egfr_ckdepi(age, sex, scr_mg_dl):
    try:
        age = float(age)
        scr = float(scr_mg_dl)
    except (TypeError, ValueError):
        return None

    if age <= 0 or scr <= 0:
        return None

    sex_norm = (sex or "").strip().lower()
    is_female = sex_norm.startswith("f")

    if is_female:
        k, a = 0.7, -0.241
    else:
        k, a = 0.9, -0.302

    scr_k = scr / k
    min_term = min(scr_k, 1) ** a
    max_term = max(scr_k, 1) ** -1.200
    age_term = 0.9938 ** age
    sex_multiplier = 1.012 if is_female else 1.0

    egfr = 142 * min_term * max_term * age_term * sex_multiplier
    egfr = round(egfr, 1)

    if egfr >= 90:
        stage, desc = 1, "Stage 1 — Normal or high (G1)"
    elif egfr >= 60:
        stage, desc = 2, "Stage 2 — Mildly decreased (G2)"
    elif egfr >= 45:
        stage, desc = "3a", "Stage 3a — Mild-moderate decrease (G3a)"
    elif egfr >= 30:
        stage, desc = "3b", "Stage 3b — Moderate-severe decrease (G3b)"
    elif egfr >= 15:
        stage, desc = 4, "Stage 4 — Severely decreased (G4)"
    else:
        stage, desc = 5, "Stage 5 — Kidney failure (G5)"

    return {
        "egfr": egfr,
        "stage": stage,
        "stage_label": desc,
    }