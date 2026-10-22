"""
disease_predictor.py — NephroSense  (NEW FILE -> put in src/)

Suggests WHICH kidney condition(s) the patient's pattern is most consistent
with. Transparent evidence-point rules based on standard nephrology
associations (KDIGO / NICE / textbook). No free public dataset has
multi-disease kidney labels, so this is deliberately NOT a black-box ML
model: every suggestion lists the exact evidence behind it.
Only POSITIVE evidence adds points (default/blank answers add nothing).
>=4 points -> "Possible", >=8 -> "Higher likelihood". Not a diagnosis.
"""


class _Ctx:
    def __init__(self, answers, clinical, user, egfr, have_blood, have_urine):
        self.d = {**(answers or {}), **(clinical or {})}
        g = str((user or {}).get("gender", "")).strip().lower()
        self.female, self.male = g.startswith("f"), g.startswith("m")
        self.age = self.num("age", (user or {}).get("age", 0) or 0)
        self.egfr, self.blood, self.urine = egfr, have_blood, have_urine

    def yes(self, k):
        return str(self.d.get(k, "")).strip().lower() == "yes"

    def eq(self, k, v):
        return str(self.d.get(k, "")).strip().lower() == v.lower()

    def num(self, k, default=0.0):
        try:
            return float(self.d.get(k, default))
        except (TypeError, ValueError):
            return default

    @property
    def al(self):
        return self.num("al") if self.urine else 0.0

    @property
    def su(self):
        return self.num("su") if self.urine else 0.0


def predict_diseases(answers, clinical, user, egfr_value, have_blood, have_urine, max_items=4):
    c = _Ctx(answers, clinical, user, egfr_value, have_blood, have_urine)
    e = c.egfr
    egfr60 = e is not None and e < 60
    egfr90 = e is not None and e < 90
    hb_cut = 13.0 if c.male else 12.0
    low_hb = c.blood and c.num("hemo", 14) < hb_cut
    rbc_ab, pc_ab = c.eq("rbc_urine", "abnormal"), c.eq("pc_urine", "abnormal")
    ba, pcc = c.eq("ba_urine", "present"), c.eq("pcc_urine", "present")

    catalog = [
        ("Chronic kidney disease (cause to be determined by a doctor)", "Nephrologist",
         "Serum creatinine/eGFR and urine ACR repeated after 3 months, kidney ultrasound", [
            (egfr60, 5, f"Estimated eGFR is below 60 ({e})"),
            (c.al >= 2, 3, "Protein (albumin) detected in urine"),
            (egfr60 and low_hb, 1, "Low haemoglobin together with reduced eGFR"),
            (c.eq("symptom_duration", "More than 3 months"), 1, "Symptoms lasting more than 3 months"),
            (c.yes("img_small_kidney"), 4, "Scan shows small kidneys (from your scan report)"),
            (c.yes("img_echogenic_kidney"), 3, "Scan shows increased kidney echogenicity (from your scan report)"),
            (c.yes("img_cortical_thinning"), 2, "Scan shows kidney scarring"),
        ]),
        ("Diabetic kidney disease", "Nephrologist + Diabetologist",
         "Urine ACR, HbA1c, serum creatinine/eGFR, eye check for retinopathy", [
            (c.yes("dm"), 3, "You have diabetes"),
            (c.eq("long_term_diabetes", "More than 10 years"), 3, "Diabetes for more than 10 years"),
            (c.eq("long_term_diabetes", "5-10 years"), 2, "Diabetes for 5-10 years"),
            (c.al >= 2, 3, "Protein (albumin) detected in urine"),
            (egfr60, 2, "Reduced eGFR"),
            (c.su >= 2, 1, "Sugar in urine"),
            (c.yes("htn"), 1, "High blood pressure as well"),
            (c.yes("hx_fam_diabetes"), 1, "Diabetes in your family (from your notes)"),
        ]),
        ("Hypertensive kidney disease (nephrosclerosis)", "Nephrologist / Physician",
         "24-hour BP monitoring, urine ACR, creatinine/eGFR, kidney ultrasound", [
            (c.yes("htn"), 3, "You have high blood pressure"),
            (c.eq("long_term_htn", "More than 10 years"), 3, "High BP for more than 10 years"),
            (c.eq("long_term_htn", "5-10 years"), 2, "High BP for 5-10 years"),
            (c.num("bp") >= 90, 2, "Diastolic BP is 90 or higher"),
            (egfr60, 2, "Reduced eGFR"),
            (c.al >= 2, 1, "Protein in urine"),
            (c.age >= 45, 1, "Age 45 or older"),
            (c.yes("hx_fam_hypertension"), 1, "High blood pressure in your family (from your notes)"),
            (c.eq("bp_control", "Often high (140/90 or more)"), 2, "Blood pressure often high"),
        ]),
        ("Glomerulonephritis / Nephrotic syndrome", "Nephrologist",
         "Urine ACR/PCR, urine microscopy, serum albumin, lipid profile, possibly kidney biopsy", [
            (c.eq("foamy_urine", "yes"), 3, "Foamy/frothy urine"),
            (c.eq("foamy_urine", "sometimes"), 1, "Sometimes foamy urine"),
            (c.yes("pe"), 2, "Swelling of feet/ankles"),
            (c.yes("puffy_eyes"), 1, "Puffy face/eyes in the morning"),
            (c.al >= 3, 3, "Significant protein in urine"),
            (rbc_ab, 2, "Red blood cells in urine"),
            (c.age < 40, 1, "Younger age group (common for this condition)"),
            (c.yes("img_echogenic_kidney"), 1, "Scan shows increased kidney echogenicity"),
        ]),
        ("Urinary tract infection / Pyelonephritis", "Physician / Urologist",
         "Urine routine & culture, CBC, ultrasound if recurrent", [
            (c.yes("burning_urine"), 3, "Burning or itching while urinating"),
            (pc_ab, 2, "Pus cells in urine"),
            (ba, 2, "Bacteria in urine"),
            (pcc, 1, "Pus cell clumps in urine"),
            (c.yes("recurrent_uti_general") or c.yes("uti_recurrent"), 2, "Repeated UTIs"),
            (c.yes("flank_pain"), 1, "Pain in sides/back"),
            (c.eq("urine_frequency", "More frequent than usual"), 1, "Urinating more often"),
            (c.female, 1, "Women have UTIs more often"),
            (c.yes("hx_self_uti"), 2, "You mentioned a urine infection (from your notes)"),
            (c.yes("hx_self_recent_fever"), 1, "Recent fever (from your notes)"),
        ]),
        ("Kidney stones (nephrolithiasis)", "Urologist",
         "Ultrasound KUB or non-contrast CT, urine routine, serum calcium/uric acid", [
            (c.yes("recurrent_kidney_stones"), 4, "Kidney stones more than once"),
            (c.yes("flank_pain"), 2, "Pain in sides/back/lower abdomen"),
            (rbc_ab, 1, "Red blood cells in urine"),
            (c.yes("nausea"), 1, "Nausea or vomiting"),
            (c.yes("burning_urine"), 1, "Burning urination"),
            (c.yes("img_kidney_stone"), 4, "Your scan report mentions kidney/ureteric stone(s)"),
            (c.yes("hx_fam_kidney_stones"), 2, "Kidney stones in your family (from your notes)"),
            (c.yes("hx_self_kidney_stones"), 3, "You had kidney stones before (from your notes)"),
            (c.yes("hx_self_kidney_stones_recurrent"), 2, "Stones more than once (from your notes)"),
            (c.yes("img_normal_study"), -3, "Against this: your kidney scan was reported normal (no stones or blockage seen)"),
        ]),
        ("Urinary obstruction / Enlarged prostate (BPH)", "Urologist",
         "Ultrasound with post-void residual, PSA, creatinine/eGFR", [
            (c.yes("prostate"), 3, "Difficulty urinating / prostate issues"),
            (c.yes("nocturia"), 2, "Waking at night to urinate"),
            (c.male and c.age >= 50, 2, "Male aged 50+"),
            (c.eq("urine_frequency", "Less frequent than usual") or c.eq("urine_frequency", "More frequent than usual"), 1, "Change in urination frequency"),
            (c.yes("img_enlarged_prostate"), 3, "Scan shows an enlarged prostate"),
            (egfr60, 2, "Reduced eGFR"),
            (c.yes("img_hydronephrosis"), 5, "Scan shows dilated kidney drainage (hydronephrosis)"),
            (c.yes("img_bladder_outlet"), 2, "Scan shows bladder outflow changes"),
            (c.yes("img_normal_study"), -3, "Against this: your kidney scan was reported normal (no blockage seen)"),
        ]),
        ("Polycystic kidney disease / inherited kidney disease", "Nephrologist",
         "Kidney ultrasound, BP check, creatinine/eGFR, genetic counselling", [
            (c.yes("family_ckd"), 3, "Close family member with kidney disease/dialysis"),
            (c.yes("img_polycystic"), 8, "Your scan report describes polycystic kidneys"),
            (c.yes("flank_pain"), 1, "Pain in sides/back"),
            (c.yes("htn"), 1, "High blood pressure"),
            (rbc_ab, 1, "Red blood cells in urine"),
            (egfr60, 2, "Reduced eGFR"),
            (c.yes("img_normal_study"), -3, "Against this: your kidney scan was reported normal (no cysts seen)"),
        ]),
        ("Chronic pyelonephritis / reflux scarring", "Nephrologist / Urologist",
         "Urine culture, kidney ultrasound, DMSA scan, creatinine/eGFR", [
            (c.yes("img_cortical_thinning"), 4, "Scan shows kidney scarring"),
            (c.yes("recurrent_uti_general") or c.yes("uti_recurrent"), 2, "Repeated UTIs"),
            (c.yes("img_small_kidney"), 1, "Scan shows small kidneys"),
            (c.yes("img_normal_study"), -3, "Against this: your kidney scan was reported normal (no scarring seen)"),
        ]),
        ("Lupus / autoimmune kidney disease", "Nephrologist + Rheumatologist",
         "ANA/anti-dsDNA, complement levels, urine ACR & microscopy, creatinine", [
            (c.yes("autoimmune_disease"), 4, "You have an autoimmune condition"),
            (c.al >= 2, 2, "Protein in urine"),
            (c.eq("foamy_urine", "yes"), 1, "Foamy urine"),
            (c.yes("pe"), 1, "Swelling"),
            (rbc_ab, 1, "Red blood cells in urine"),
            (c.female and 15 <= c.age <= 45, 1, "Most common in women aged 15-45"),
        ]),
        ("Painkiller (NSAID) / drug-related kidney injury", "Physician / Nephrologist",
         "Stop-and-review of medicines with your doctor, creatinine/eGFR, urine tests", [
            (c.eq("nsaid_use", "Daily"), 4, "Daily painkiller use"),
            (c.eq("nsaid_use", "Weekly"), 2, "Weekly painkiller use"),
            (egfr60, 3, "Reduced eGFR"),
            (egfr90 and not egfr60, 2, "Mildly reduced eGFR"),
            (c.yes("contrast_dye_exposure"), 1, "Recent contrast dye exposure"),
        ]),
        ("Acute kidney injury (sudden kidney function drop)", "Physician urgently / Nephrologist",
         "Creatinine & urea now, urine output monitoring, electrolytes, ultrasound", [
            (c.yes("contrast_dye_exposure"), 3, "CT/angiogram contrast dye in the last year"),
            (c.eq("urine_frequency", "Less frequent than usual"), 2, "Passing much less urine"),
            (c.eq("symptom_duration", "Less than 1 week") or c.eq("symptom_duration", "1-4 weeks"), 2, "Symptoms started recently"),
            (egfr60, 2, "Reduced eGFR"),
            (c.yes("nausea"), 1, "Nausea or vomiting"),
            (c.yes("hx_self_recent_fever"), 1, "Recent fever / illness (from your notes)"),
        ]),
        ("Mass or lesion mentioned on scan (needs prompt review)", "Urologist / Nephrologist",
         "Doctor review of the scan; contrast CT or MRI as advised", [
            (c.yes("img_renal_mass"), 8, "A mass/lesion is mentioned in your scan report"),
        ]),
    ]

    out = []
    for name, spec, tests, rules in catalog:
        hit = [(p, t) for cond, p, t in rules if cond]
        pts = sum(p for p, _ in hit)
        if pts >= 4:
            out.append({
                "name": name,
                "likelihood": "Higher likelihood" if pts >= 8 else "Possible",
                "points": pts,
                "evidence": [t for _, t in hit],
                "tests": tests,
                "specialist": spec,
            })
    out.sort(key=lambda x: x["points"], reverse=True)
    return out[:max_items]