"""
history_extractor.py — NephroSense  (NEW FILE -> put in src/)

Reads the patient's own notes ("Anything else your doctor should know?") and picks out
kidney-relevant history with simple, transparent keyword rules (no AI, no translation):

  FAMILY : kidney stones, kidney disease/dialysis, diabetes, high blood pressure, thyroid, heart disease
  SELF   : past kidney stones (and repeated stones), past infections (typhoid, malaria, dengue,
           jaundice/hepatitis, TB, COVID), urine infection, recent fever (days/weeks ago)

It tracks WHO a clause is about (family member vs the patient) so "my mother and grandmother
have diabetes" is a family history, while "I had typhoid" is the patient's own. Tolerates common
spelling mistakes (diabtic, presure). The patient confirms every point before it is used.
"""
import re

_REL = re.compile(
    r"\b(mother|mom|mum|mummy|father|dad|papa|grandmother|grandfather|grandma|grandpa|"
    r"sister|brother|parents?|aunt|uncle|maternal|paternal|family|relatives?)\b")
_SELF = re.compile(r"\b(i|my|me|myself)\b")

_STONE = re.compile(r"stone|calcul|mutkhada|pathri")
_KIDNEY_DIS = re.compile(r"dialysis|kidney (disease|failure|problem|transplant|damage)|renal|\bckd\b")
_DIAB = re.compile(r"diab|sugar")
_HTN = re.compile(r"pressure|presure|hypertens|\bbp\b|\bb\.p\b")
_THYROID = re.compile(r"thyroid")
_HEART = re.compile(r"heart (attack|disease|problem|failure)|cardiac|bypass|angioplasty")
_REPEAT = re.compile(r"\b(2|3|4|two|three|twice|thrice|multiple|again|repeated|recurrent|many)\b")

_INFECTIONS = [
    ("typhoid", re.compile(r"typhoid|enteric")),
    ("malaria", re.compile(r"malaria")),
    ("dengue", re.compile(r"dengue")),
    ("jaundice/hepatitis", re.compile(r"jaundice|hepatitis")),
    ("tuberculosis", re.compile(r"tuberculosis|\btb\b")),
    ("COVID-19", re.compile(r"covid|corona")),
]
_UTI = re.compile(r"\buti\b|urine infection|urinary infection|burning urine")
_FEVER = re.compile(r"fever|bukhar|high temperature")
_RECENT = re.compile(
    r"(\d+|a|one|two|three|four|five|few|couple of)\s*(day|days|week|weeks)\s*(ago|back)|"
    r"yesterday|today|tonight|this week|last week|currently|right now|these days|since (yesterday|morning|\d)")


def _clauses(text):
    t = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
    parts = re.split(
        r"[.;\n,!?]| also | but | then |\s(?=i\s+(?:had|have|am|was|got|suffered|take|took|feel|felt)\b)", t)
    return [p.strip() for p in parts if p and p.strip()]


def extract_history(text):
    """Returns a list of dicts: key, label, kind ('family'/'self'), snippet."""
    items, seen = [], set()
    subject, rels = "self", []

    def add(key, label, kind, snippet):
        if key in seen:
            return
        seen.add(key)
        items.append({"key": key, "label": label, "kind": kind, "snippet": snippet[:90]})

    for cl in _clauses(text):
        found_rel = _REL.findall(cl)
        if found_rel:
            subject = "family"
            for r in found_rel:
                if r not in rels:
                    rels.append(r)
        elif _SELF.search(cl) and not re.search(r"\b(her|his|their)\b", cl):
            subject = "self"

        if subject == "family":
            who = ", ".join(rels[:3]) if rels else "family"
            tag = f"Family history — {who}"
            if _STONE.search(cl):
                add("fam_kidney_stones", f"{tag}: kidney stones", "family", cl)
            if _KIDNEY_DIS.search(cl):
                add("fam_kidney_disease", f"{tag}: kidney disease / dialysis", "family", cl)
            if _DIAB.search(cl):
                add("fam_diabetes", f"{tag}: diabetes", "family", cl)
            if _HTN.search(cl):
                add("fam_hypertension", f"{tag}: high blood pressure", "family", cl)
            if _THYROID.search(cl):
                add("fam_thyroid", f"{tag}: thyroid problem", "family", cl)
            if _HEART.search(cl):
                add("fam_heart", f"{tag}: heart disease", "family", cl)
        else:
            if _STONE.search(cl):
                add("self_kidney_stones", "You: kidney stones in the past", "self", cl)
                if _REPEAT.search(cl):
                    add("self_kidney_stones_recurrent", "You: kidney stones more than once", "self", cl)
            if _UTI.search(cl):
                add("self_uti", "You: urine infection", "self", cl)
            for name, rx in _INFECTIONS:
                if rx.search(cl):
                    add(f"self_past_{name.split('/')[0].replace('-', '').lower()}",
                        f"You: {name} (past illness)", "self", cl)
            for m in _FEVER.finditer(cl):
                window = cl[max(0, m.start() - 20): m.end() + 30]
                if _RECENT.search(window):
                    add("self_recent_fever", "You: recent fever", "self", cl)
                    break
    return items