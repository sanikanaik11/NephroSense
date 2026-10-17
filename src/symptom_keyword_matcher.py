"""
symptom_keyword_matcher.py — NephroSense
Transparent, rule-based keyword matching — NOT AI, NOT a black box —
that scans the patient's own (typed or translated-from-voice) English
description and suggests which items in the doctor-facing Symptoms &
Medical History checklist might apply, so nothing the patient already
said gets missed if they forget to also tick the matching checkbox
further down the form.

IMPORTANT SAFETY DESIGN:
  - This ONLY suggests values for CLINICAL_QUESTIONS fields — the
    doctor-documentation fields that do NOT feed the ML risk score.
  - It NEVER touches the scored QUESTIONS fields (age, bp, htn, dm,
    pe, cad, ane, etc.) — so free-text descriptions can never
    silently change a patient's risk score, only help pre-fill
    optional documentation fields.
  - Every suggestion requires the patient's explicit confirmation
    (a checkbox they tick) before it's applied to the form — nothing
    is auto-filled without the patient seeing and agreeing to it.
"""

# Each entry: (list of keyword phrases to look for, target CLINICAL_QUESTIONS
# key, value to suggest, human-readable label shown to the patient).
KEYWORD_RULES = [
    (["burning", "burning sensation", "painful urination", "pain while urinating",
      "pain when urinating", "stinging"], "burning_urine", "yes",
     "Burning/painful urination"),

    (["urinating more", "urinate more often", "frequent urination", "peeing a lot",
      "going to toilet often"], "urine_frequency", "More frequent than usual",
     "More frequent urination"),

    (["urinating less", "less urine", "not urinating much", "hardly any urine"],
     "urine_frequency", "Less frequent than usual",
     "Less frequent urination"),

    (["back pain", "side pain", "stomach pain", "flank pain", "pain in my sides",
      "pain in my back", "abdominal pain"], "flank_pain", "yes",
     "Back/side/stomach pain"),

    (["nausea", "vomiting", "throwing up", "feel like vomiting", "vomited"],
     "nausea", "yes", "Nausea or vomiting"),

    (["foamy urine", "frothy urine", "bubbles in my urine", "bubbly urine"],
     "foamy_urine", "yes", "Foamy/frothy urine"),

    (["waking up at night to urinate", "urinate at night", "night urination",
      "peeing at night"], "nocturia", "yes", "Waking at night to urinate"),

    (["tired", "fatigue", "fatigued", "weakness", "feeling weak", "exhausted",
      "no energy"], "fatigue", "yes", "Unusual fatigue/weakness"),

    (["losing weight", "weight loss", "loss of appetite", "not eating",
      "no appetite", "don't feel like eating"], "appetite_change", "yes",
     "Weight loss or appetite change"),

    (["kidney stone", "kidney stones"], "recurrent_kidney_stones", "yes",
     "Kidney stones mentioned"),

    (["family history", "my father has kidney", "my mother has kidney",
      "runs in my family", "dialysis in my family"], "family_ckd", "yes",
     "Family history of kidney disease"),
]


def detect_symptom_suggestions(english_text):
    """
    english_text: the patient's description, already in English
                   (typed directly, or translated from a voice note).
    Returns a list of dicts: [{key, value, label, matched_phrase}, ...]
    — one per distinct CLINICAL_QUESTIONS field that seems to apply,
    based on simple substring matching. Always deduplicated by key
    (first matching rule wins per field).
    """
    if not english_text:
        return []

    text_lower = english_text.lower()
    suggestions = []
    seen_keys = set()

    for phrases, key, value, label in KEYWORD_RULES:
        if key in seen_keys:
            continue
        for phrase in phrases:
            if phrase in text_lower:
                suggestions.append({
                    "key": key,
                    "value": value,
                    "label": label,
                    "matched_phrase": phrase,
                })
                seen_keys.add(key)
                break

    return suggestions