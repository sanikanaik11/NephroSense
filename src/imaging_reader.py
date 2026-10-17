"""
imaging_reader.py — NephroSense  (NEW FILE -> put in src/)

Reads the TEXT of an ultrasound (sonography/USG), MRI or CT report (already OCR'd by
report_reader.read_text_from_files) and picks out kidney-relevant findings with transparent
keyword rules. Handles negation ("no hydronephrosis", "calculus not seen") and prefers the
IMPRESSION / CONCLUSION section when the report has one.

ONLY the kidneys / ureters / bladder / prostate are considered. A "lesion" in the liver, a cyst on
the ovary, stones in the gall bladder etc. are ignored (a whole-abdomen ultrasound mentions many organs).

The patient always confirms / edits the findings in the app — nothing is applied silently.

Findings:
  MAJOR (structural signs of kidney damage — a KDIGO "marker of kidney damage"):
      small_kidney, cortical_thinning, echogenic_kidney, polycystic, hydronephrosis
  MINOR: kidney_stone, renal_cyst, enlarged_prostate, bladder_outlet, renal_artery_stenosis
  ATTENTION (needs a doctor, not scored): renal_mass
  normal_study (reported normal; never lowers a score)
"""
import re

FINDINGS = [
    ("small_kidney", "Small / shrunken kidney(s)", True,
     r"small[- ]sized kidney|small kidneys?|small (?:right |left |rt |lt )?kidney|shrunken|atrophic (?:kidney|changes)|"
     r"contracted kidney|(?:reduced|decreased) (?:in )?(?:renal |kidney )?size|kidneys? (?:is|are) reduced in size|"
     r"reduced renal (?:size|volume)"),
    ("cortical_thinning", "Cortical thinning / scarring of kidney", True,
     r"cortical thinning|thinned (?:renal )?cortex|thin (?:renal )?cortex|cortical scarring|renal scarring|scarred kidney|"
     r"parenchymal thinning|reduced (?:renal )?parenchymal thickness"),
    ("echogenic_kidney", "Increased echogenicity / medical renal disease", True,
     r"(?:increased|raised|bright|prominent) (?:renal |cortical |parenchymal )?(?:echogenicity|echotexture|echoes)|"
     r"echogenic kidneys?|medical renal disease|renal parenchymal disease|parenchymal (?:changes|disease)|"
     r"grade (?:i{1,3}|[1-3]) (?:medical )?(?:renal|rpd)|loss of corticomedullary differentiation|\bcmd (?:is )?(?:lost|reduced|poor)|"
     r"poor corticomedullary"),
    ("polycystic", "Polycystic kidney disease", True,
     r"polycystic|\bpkd\b|multiple (?:bilateral )?(?:renal )?cysts? (?:in|involving) both kidneys|bilateral multiple (?:renal )?cysts"),
    ("hydronephrosis", "Hydronephrosis / dilated urine passage", True,
     r"hydronephro\w*|hydroureter\w*|pelvi-?\s?calyceal (?:system )?(?:dilat\w*|ectasia|prominence)|pelvicaliectasis|"
     r"pcs (?:is )?dilat\w*|dilated (?:pelvicalyceal|renal pelvis|ureter)|renal pelvis (?:is )?dilat\w*"),
    ("solitary_kidney", "Only one kidney (absent / removed)", True,
     r"solitary kidney|single kidney|only one kidney|(?:right|left|rt|lt) kidney (?:is )?(?:not seen|absent|not visuali[sz]ed)|"
     r"renal agenesis|nephrectomy|post[- ]nephrectomy|nephrectomi[sz]ed"),
    ("kidney_stone", "Kidney / ureteric stone(s)", False,
     r"calcul(?:us|i)|\bstones?\b|nephrolithiasis|urolithiasis|renal concretions?|echogenic focus with (?:acoustic )?shadow"),
    ("renal_cyst", "Simple kidney cyst(s)", False,
     r"simple (?:renal |cortical )?cysts?|(?:renal|cortical|parapelvic) cysts?|cyst (?:is )?(?:seen|noted) in (?:the )?(?:right|left)"),
    ("enlarged_prostate", "Enlarged prostate", False,
     r"enlarged prostate|prostatomegaly|\bbph\b|benign prostatic|prostate (?:is )?(?:mildly |moderately |grossly )?enlarged|prostatic enlargement"),
    ("bladder_outlet", "Bladder outflow problem (thick bladder wall / residual urine)", False,
     r"bladder outlet obstruction|trabeculated|trabeculation|thick[- ]walled bladder|bladder wall (?:is )?thick\w*|"
     r"bladder diverticul\w*|significant (?:post[- ]?void )?residual"),
    ("renal_artery_stenosis", "Narrowing of kidney artery", False,
     r"renal artery stenosis|renal artery narrowing|\bras\b(?= of)"),
    ("renal_mass", "Mass / lesion in the kidney (needs a doctor)", False,
     r"\bmass\b|\blesion\b|tumou?r|neoplasm|angiomyolipoma|space[- ]occupying|\bsol\b"),
]
FINDING_LABELS = {k: lab for k, lab, _m, _p in FINDINGS}
FINDING_LABELS["normal_study"] = "Kidneys / urinary tract reported as normal"
MAJOR_KEYS = {k for k, _l, major, _p in FINDINGS if major}
_PATTERNS = [(k, lab, major, re.compile(p, re.I)) for k, lab, major, p in FINDINGS]

_NEG_BEFORE = re.compile(
    r"\b(?:no|not|without|absent|nil|negative|neither|nor|free of|rule[ds]? out|excluded)\b|no evidence of|unremarkable|"
    r"\bwnl\b", re.I)
_NEG_AFTER = re.compile(
    r"not (?:seen|visuali[sz]ed|noted|identified|demonstrated|appreciated|detected)|absent|ruled out|excluded|"
    r"is not|are not|nil", re.I)
_CONJ = re.compile(r"\b(?:and|but|however|with|except|while|whereas|also|though)\b", re.I)
_NORMAL = re.compile(
    r"both kidneys? (?:are |is )?(?:normal|unremarkable)|kidneys? (?:are |is )?normal in (?:size|shape|position)|"
    r"normal (?:study|scan|sonography|ultrasound|usg|mri|ct)|no (?:significant |obvious )?abnormalit\w*|"
    r"normal renal|unremarkable (?:study|scan)", re.I)


_URINARY = re.compile(
    r"kidneys?|renal|nephr\w*|ureter\w*|pelvi-?\s?calyceal|calyx|calyces|calyceal|\bpcs\b|\brpd\b|"
    r"(?<!gall )(?<!gall-)(?<!gb )urinary|(?<!gall )(?<!gall-)\bbladder|prostat\w*|\bbph\b|\blkub\b|\bkub\b|"
    r"\b(?:right|left|rt|lt)\s+(?:upper|mid|lower)\s+(?:pole|zone|calyx)", re.I)
_OTHER_ORGAN = re.compile(
    r"uterus|ovar(?:y|ies)|endometri\w*|liver|hepatic|spleen|pancrea\w*|appendix|portal|\bcbd\b|bile duct|aorta|"
    r"lymph|thyroid|breast|lung", re.I)
_GALL = re.compile(r"gall[ -]?bladder|\bgb\b|cholecyst\w*", re.I)
_NORMAL_WORDS = re.compile(r"\b(?:normal|unremarkable|maintained|preserved|no (?:abnormality|abnormal))\b", re.I)


def _is_urinary(clause):
    c = _GALL.sub(" ", clause)
    return bool(_URINARY.search(c))


def _modality(t):
    low = t.lower()
    if re.search(r"\bmri\b|magnetic resonance", low):
        return "MRI"
    if re.search(r"\bcect\b|\bnect\b|\bct\b|computed tomography|\bhrct\b", low):
        return "CT scan"
    if re.search(r"ultra ?sound|sonograph|\busg\b|\bsono\b|doppler", low):
        return "Ultrasound (sonography)"
    if re.search(r"x-?ray|\bkub\b", low):
        return "X-ray"
    return "Other"


def _impression(t):
    m = re.search(r"\b(?:impression|conclusion|opinion|diagnosis|summary)\b\s*[:\-]?", t, re.I)
    if not m:
        return ""
    return t[m.end(): m.end() + 1500].strip()


def _negated(clause, m):
    pre = clause[max(0, m.start() - 45): m.start()]
    pre = _CONJ.split(pre)[-1]
    if _NEG_BEFORE.search(pre):
        return True
    return bool(_NEG_AFTER.search(clause[m.end(): m.end() + 30]))


def extract_imaging(text):
    """Returns dict: modality, impression (the kidney/urinary part of the report), findings
    [{key,label,major,snippet}], normal (bool)."""
    t = re.sub(r"[ \t]+", " ", text or "")
    clauses = [c.strip() for c in re.split(r"(?<!\d)\.|\.(?!\d)|[;\n]| but | however ", t) if c.strip()]
    # keep a clause when it is about the urinary tract; a short follow-on clause inherits the subject
    urinary, prev_urinary = [], False
    for c in clauses:
        u = _is_urinary(c)
        if u or (prev_urinary and len(c) < 60 and not re.match(r"^[A-Za-z /&-]{2,25}:", c)
                 and not _OTHER_ORGAN.search(c)):
            urinary.append(c)
            prev_urinary = True
        else:
            prev_urinary = False
    findings, seen = [], set()
    scope_low = " ".join(urinary).lower()
    for c in urinary:
        for key, label, major, rx in _PATTERNS:
            if key in seen:
                continue
            for m in rx.finditer(c):
                if _negated(c, m):
                    continue
                if key == "renal_cyst" and "polycystic" in scope_low:
                    continue
                seen.add(key)
                findings.append({"key": key, "label": label, "major": major, "snippet": c[:120]})
                break
    normal = (not findings) and any(_NORMAL_WORDS.search(c) for c in urinary)
    kidney_text = ". ".join(urinary)[:500].strip()
    return {
        "modality": _modality(t),
        "impression": kidney_text,
        "findings": findings,
        "normal": normal,
    }