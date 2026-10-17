"""
risk_engine.py — NephroSense  (REPLACE existing)

Guideline-based CKD risk engine (KDIGO eGFR x albuminuria "heat map").
PRIMARY score. The ML model is only a minor (20%) second opinion.

REPORT AGE (new): every blood / urine report has a date. Old results must not reassure:
  age of report      NORMAL old report                         ABNORMAL old report
  0-90 days   fresh  used normally                             used normally
  91-180      aging  used, patient told to repeat if symptoms  used
  181-365     old    used, but score cannot drop below the     used, confidence reduced,
                     history/symptom score; confidence -1      "repeat now"
  > 365       expired NOT used to judge current risk           kept (can't assume it healed),
                     (history/symptoms only, Limited)          confidence Limited, "repeat now"
Abnormal results are never discarded just because they are old.

Confidence:  High = recent blood + urine | Moderate = recent blood only (or stale urine)
             Limited = no usable blood result -> score from history/symptoms only (max 60)
"""
from egfr import calculate_egfr_ckdepi
from clinical_risk_score import calculate_clinical_risk_score
try:
    from imaging_reader import FINDING_LABELS, MAJOR_KEYS
except Exception:                       # imaging_reader not installed yet
    FINDING_LABELS, MAJOR_KEYS = {}, set()

KDIGO_MATRIX = {
    "G1": (0, 1, 2), "G2": (0, 1, 2),
    "G3a": (1, 2, 3), "G3b": (2, 3, 3),
    "G4": (3, 3, 3), "G5": (3, 3, 3),
}
BASE_SCORE = (8, 40, 66, 88)
CATEGORY_NAME = ("Low risk", "Moderately increased risk", "High risk", "Very high risk")
STAGE_INT = {"G1": 1, "G2": 2, "G3a": 3, "G3b": 3, "G4": 4, "G5": 5}
A_NAMES = ("A1 — normal", "A2 — mildly raised", "A3 — markedly raised")
CONF = ["Limited", "Moderate", "High"]

def report_band(days):
    if days is None:
        return None
    if days <= 90:
        return "fresh"
    if days <= 180:
        return "aging"
    if days <= 365:
        return "old"
    return "expired"


def age_text(days):
    if days < 0:
        return "in the future"
    if days < 60:
        return f"{days} day{'s' if days != 1 else ''}"
    if days < 365:
        return f"{days // 30} months"
    years, rem = divmod(days, 365)
    months = rem // 30
    return f"{years} year{'s' if years != 1 else ''}" + (f" {months} months" if months else "")


def _g_stage(e):
    if e >= 90: return "G1"
    if e >= 60: return "G2"
    if e >= 45: return "G3a"
    if e >= 30: return "G3b"
    if e >= 15: return "G4"
    return "G5"


def _a_index(al):
    # dipstick scale: 0 nil, 1 trace, 2 = 1+, 3 = 2+, 4 = 3+, 5 = 4+
    # trace is common in healthy people (concentrated urine, exercise, fever) -> still A1
    al = float(al or 0)
    if al <= 1: return 0
    if al <= 3: return 1
    return 2


def _yes(a, k):
    return str(a.get(k, "")).strip().lower() == "yes"


def _num(a, k, d=0.0):
    try:
        return float(a.get(k, d))
    except (TypeError, ValueError):
        return d


def _lab_flags(a, gender, have_blood, have_urine):
    flags = []
    male = str(gender or "").strip().lower().startswith("m")
    if have_blood:
        hb_cut = 13.0 if male else 12.0
        if _num(a, "hemo", 14) < hb_cut:
            flags.append(f"Low haemoglobin ({_num(a, 'hemo'):.1f} g/dL)")
        if _num(a, "bu", 30) > 45:
            flags.append(f"High blood urea ({_num(a, 'bu'):.0f} mg/dL)")
        pot = _num(a, "pot", 4.5)
        if pot < 3.5 or pot > 5.2:
            flags.append(f"Potassium out of range ({pot:.1f} mEq/L)")
        sod = _num(a, "sod", 140)
        if sod < 135 or sod > 145:
            flags.append(f"Sodium out of range ({sod:.0f} mEq/L)")
        if _num(a, "bgr", 100) >= 200:
            flags.append(f"High random blood sugar ({_num(a, 'bgr'):.0f} mg/dL)")
    if have_urine and _num(a, "su", 0) >= 2:
        flags.append("Sugar present in urine")
    return flags


def _level(score):
    return "LOW" if score < 33 else ("MEDIUM" if score < 66 else "HIGH")


def _history_score(answers, clinical_answers, age, flags, clin_pts, urine_used, img_major=False, img_minor_pts=0):
    """Score from history + symptoms (+ urine if usable). Capped at 60 — no kidney-function result."""
    pts = 0.0
    pts += 10 if _yes(answers, "htn") else 0
    pts += 12 if _yes(answers, "dm") else 0
    pts += 6 if _yes(answers, "cad") else 0
    pts += 8 if _yes(answers, "pe") else 0
    pts += 6 if _yes(answers, "ane") else 0
    pts += 4 if str(answers.get("appet", "good")).lower() == "poor" else 0
    pts += 8 if age >= 60 else (4 if age >= 50 else 0)
    pts += 5 if str(clinical_answers.get("foamy_urine", "no")).lower() in ("yes", "sometimes") else 0
    pts += 2 if _yes(clinical_answers, "flank_pain") else 0
    pts += 2 if _yes(clinical_answers, "burning_urine") else 0
    pts += len(flags) * 3 + clin_pts * 0.5
    pts += (20 if img_major else 0) + img_minor_pts
    score = max(5.0, min(60.0, pts))
    if img_major:
        score = max(score, 35.0)
    if urine_used:
        al = _num(answers, "al", 0)
        if al >= 4:
            score = max(score, 60.0)
        elif al >= 2:
            score = max(score, 35.0)
    return score


def assess_risk(answers, clinical_answers, user, ml_result, have_blood, have_urine,
                blood_days=None, urine_days=None, imaging_days=None):
    gender = (user or {}).get("gender", "")
    age = _num(answers, "age", (user or {}).get("age", 40) or 40)
    clin_pts = calculate_clinical_risk_score(clinical_answers or {})["clinical_risk_score"]
    reasons, report_notes = [], []

    comorb = 0
    if _yes(answers, "dm"):
        comorb += 4; reasons.append("You reported diabetes — a major kidney risk factor")
    if _yes(answers, "htn"):
        comorb += 3; reasons.append("You reported high blood pressure — a major kidney risk factor")
    if _yes(answers, "cad"):
        comorb += 2; reasons.append("You reported heart disease")
    if clin_pts >= 20:
        reasons.append(f"Clinical risk-factor checklist: {clin_pts}/100 (family history, painkillers, etc.)")

    hx = clinical_answers or {}
    fam_bits = [n for k, n in (("hx_fam_diabetes", "diabetes"), ("hx_fam_hypertension", "high blood pressure"),
                               ("hx_fam_kidney_stones", "kidney stones")) if str(hx.get(k, "")).lower() == "yes"]
    if fam_bits:
        reasons.append("Family history from your notes (" + ", ".join(fam_bits) + ") — counted in the risk-factor checklist")
    if str(hx.get("hx_self_recent_fever", "")).lower() == "yes":
        reasons.append("You mentioned a recent fever — fever can temporarily change urine and blood results; "
                       "repeat any abnormal test after you recover")

    # ── scan reports (ultrasound / MRI / CT), confirmed by the patient ──
    img_major = [FINDING_LABELS[k] for k in sorted(MAJOR_KEYS) if _yes(hx, f"img_{k}")]
    img_minor_pts = (4 if _yes(hx, "img_kidney_stone") else 0) + (3 if _yes(hx, "img_enlarged_prostate") else 0) \
        + (2 if _yes(hx, "img_bladder_outlet") else 0) + (2 if _yes(hx, "img_renal_artery_stenosis") else 0)
    if img_major:
        reasons.append("Scan report shows structural kidney changes (" + "; ".join(img_major) +
                       ") — doctors count this as a sign of kidney damage even when blood tests look normal")
    if _yes(hx, "img_kidney_stone"):
        reasons.append("Scan report mentions kidney/ureteric stone(s)")
    if _yes(hx, "img_renal_mass"):
        reasons.append("Scan report mentions a mass/lesion — please show it to a doctor soon (not part of the score)")

    # ── report age ──────────────────────────────────────────
    bb = report_band(blood_days) if have_blood else None
    ub = report_band(urine_days) if have_urine else None

    egfr0 = calculate_egfr_ckdepi(age, gender, answers.get("sc")) if (have_blood and age >= 18) else None
    blood_abn = bool(have_blood) and (
        bool(_lab_flags(answers, gender, True, False)) or (egfr0 is not None and egfr0["egfr"] < 90))
    urine_abn = bool(have_urine) and (_num(answers, "al", 0) >= 2 or _num(answers, "su", 0) >= 2)

    blood_used = bool(have_blood) and not (bb == "expired" and not blood_abn)
    urine_used = bool(have_urine) and not (ub == "expired" and not urine_abn)

    def _note(kind, days, band, abnormal):
        a = age_text(days)
        if band == "aging":
            return (f"Your {kind} report is {a} old. It was used, but please repeat it if you have "
                    f"symptoms, diabetes or high blood pressure.")
        if band == "old":
            if abnormal:
                return (f"Your {kind} report is {a} old and showed abnormal values — please repeat it now "
                        f"to see whether things have changed.")
            return (f"Your {kind} report is {a} old. A normal result this old cannot confirm your kidneys "
                    f"are healthy today, so your history and symptoms are also counted. Please repeat it soon.")
        if band == "expired":
            if abnormal:
                return (f"Your {kind} report is {a} old but showed abnormal values, so it was kept. "
                        f"Please repeat it now.")
            return (f"Your {kind} report is {a} old, so it was NOT used to judge your current risk. "
                    f"Please repeat the test and update this assessment.")
        return None

    if have_blood and bb in ("aging", "old", "expired"):
        report_notes.append(_note("blood", blood_days, bb, blood_abn))
    if have_urine and ub in ("aging", "old", "expired"):
        report_notes.append(_note("urine", urine_days, ub, urine_abn))

    flags = _lab_flags(answers, gender, blood_used, urine_used)
    reasons.extend(flags)

    egfr = egfr0 if blood_used else None
    ml_used = bool(ml_result) and "trained on" in str(ml_result.get("explanation", ""))
    ml_score = ml_result.get("risk_score") if (ml_used and blood_used and bb in (None, "fresh", "aging")) else None

    hist_score = _history_score(answers, clinical_answers or {}, age, flags, clin_pts, urine_used,
                                img_major=bool(img_major), img_minor_pts=img_minor_pts)

    g = None
    cat = None
    if egfr:
        g = _g_stage(egfr["egfr"])
        a_idx = _a_index(answers.get("al")) if urine_used else 0
        idx = KDIGO_MATRIX[g][a_idx]
        idx_before_img = idx
        if img_major and idx < 1:
            idx = 1                      # structural damage on imaging = marker of kidney damage (KDIGO)
        cat = CATEGORY_NAME[idx]
        reasons.insert(0, f"Estimated eGFR {egfr['egfr']} mL/min/1.73m² (CKD-EPI 2021) → {egfr['stage_label']}")
        if urine_used:
            reasons.insert(1, f"Urine albumin category: {A_NAMES[a_idx]}")
        else:
            reasons.insert(1, "No usable urine test — albuminuria could not be checked")

        if str(gender or "").strip().lower()[:1] not in ("f", "m"):
            reasons.append("eGFR was calculated with the male-sex coefficient because gender is not Female/Male")
        if urine_used and _num(answers, "al", 0) == 1:
            reasons.append("Trace protein in urine — often harmless (concentrated urine, exercise, fever) "
                           "but worth repeating")
        extra = len(flags) * 3 + comorb + clin_pts * 0.25 + img_minor_pts + (5 if (img_major and idx_before_img >= 1) else 0)
        if g == "G2" and age < 40:
            extra += 5
        score = min(100.0, BASE_SCORE[idx] + min(15.0, extra))

        if ml_score is not None:
            blended = score + 0.2 * (float(ml_score) - score)
            if idx == 0 and not flags:
                blended = min(blended, 30.0)
            if idx >= 2:
                blended = max(blended, score)
            score = blended
            reasons.append(f"AI pattern model (20% weight): {ml_score}%")

        # old NORMAL results cannot lower the score below what history/symptoms indicate
        stale_normal = (bb == "old" and not blood_abn) or (ub == "old" and not urine_abn)
        if stale_normal and hist_score > score:
            score = hist_score
            reasons.append("Because the report is old, your history and symptoms set a minimum for the score")

        conf_i = 2 if urine_used else 1
        if bb in ("old", "expired"):
            conf_i -= 1
        if urine_used and ub in ("old", "expired"):
            conf_i -= 1
        confidence = CONF[max(0, conf_i)]
    else:
        score = hist_score
        if have_blood and not blood_used:
            reasons.insert(0, "Your blood report is too old to use — score is based on history and symptoms only")
        elif not have_blood:
            reasons.insert(0, "No blood test provided — kidney function (creatinine/eGFR) could not be measured; "
                              "this score uses your history and symptoms only")
        if have_blood and blood_used and egfr is None and age < 18:
            reasons.insert(0, "The eGFR formula is validated for adults (18+) only, so kidney function could not be "
                              "calculated — please ask a doctor to interpret your child's/your blood report")
        if urine_used and _num(answers, "al", 0) >= 2:
            reasons.insert(0, "Protein (albumin) found in urine")
        confidence = "Limited"

    # ── scan notes (the scan findings themselves are scored above via the confirmed img_* answers) ──
    if _yes(hx, "img_renal_mass"):
        report_notes.append("Your scan report mentions a mass/lesion. Please show the report to a doctor promptly.")
    if _yes(hx, "img_hydronephrosis") or _yes(hx, "img_kidney_stone"):
        report_notes.append("Your scan suggests a possible blockage or stone. Blocked urine flow can damage the kidneys — "
                            "please see a urologist or physician soon.")
    if _yes(hx, "img_normal_study") and not img_major:
        reasons.append("Scan report says the kidneys look normal (a normal scan cannot rule out early kidney disease, "
                       "so it does not lower the score)")
    if imaging_days is not None and imaging_days > 730 and any(_yes(hx, k) for k in hx if str(k).startswith("img_")):
        report_notes.append(f"Your scan report is {age_text(imaging_days)} old; kidney findings can change — consider repeating it.")

    score = round(max(0.0, min(100.0, score)), 1)
    level = _level(score)

    stale = any(b in ("old", "expired") for b in (bb, ub))
    if level == "HIGH":
        advice = "Please see a nephrologist or physician soon. Ask for serum creatinine/eGFR and a urine ACR test (repeat after 3 months to confirm)."
    elif level == "MEDIUM":
        advice = "Consult a doctor in the next few weeks and repeat creatinine + urine ACR tests to confirm."
    elif confidence == "Limited" or stale:
        advice = "Low based on the information available, but it is not a confirmed result. Please get a fresh blood creatinine and urine albumin test done."
    else:
        advice = "Your kidney markers look healthy. Repeat a kidney check yearly (every 6 months if you have diabetes or high BP)."

    if _yes(hx, "img_renal_mass"):
        advice = "Your scan mentions a mass/lesion — please show the report to a doctor promptly. " + advice
    if not reasons:
        reasons.append("No kidney risk factors were reported")

    return {
        "risk_score": score,
        "risk_level": level,
        "ckd_stage": STAGE_INT.get(g) if g else None,
        "stage_text": egfr["stage_label"] if egfr else None,
        "kdigo_category": cat,
        "confidence": confidence,
        "reasons": reasons,
        "advice": advice,
        "ml_score": ml_score,
        "blood_used": blood_used,
        "urine_used": urine_used,
        "blood_days": blood_days,
        "urine_days": urine_days,
        "report_notes": report_notes,
        "imaging_used": sorted(k[4:] for k in hx if str(k).startswith("img_") and _yes(hx, k)),
    }