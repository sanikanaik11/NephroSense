"""
report_reader.py — NephroSense  (REPLACE existing)

Reads PRINTED lab reports (phone photos of paper, scanned pages, PDFs) and fills the
blood / urine values NephroSense needs. Built for typical Indian lab layouts
(Test | Result | Unit | Reference range).

  * Phone-photo handling: EXIF rotation, resize, contrast clean-up, adaptive threshold,
    automatic rotation retries — the best scan is kept.
  * Two OCR engines: Tesseract (needs the free Tesseract program, auto-found on Windows) OR
    RapidOCR (pip only). ocr_status() tells the app what is missing.
  * Scanned PDFs are read too (pip install pymupdf). Several pages/files are combined.
  * SAFE parsing: serum albumin (g/dL) is never mistaken for urine albumin; blood glucose never
    for urine sugar; urine-microscopy RBC never mixed with the CBC RBC count; units converted
    (creatinine umol/L, Hb g/L, BUN -> urea, glucose mmol/L, WBC x10^3); implausible numbers rejected.
    Anything not read confidently is left blank.
  * Urine ACR (mg/g) is converted to the dipstick-style grade used by the app.

Engines (first that works is used):
  1. Google Gemini vision (free API key in env GEMINI_API_KEY) — reads photos AND scanned PDFs very
     accurately; the report image is sent to Google. Skipped when no key is set.
  2. Tesseract / RapidOCR on this computer. If none is installed, the app installs RapidOCR + PyMuPDF
     automatically the first time a report is scanned (set NEPHROSENSE_AUTO_INSTALL=0 to disable).

Manual install (optional):
    pip install pytesseract pillow opencv-python pypdf pymupdf        (+ Tesseract program)
    pip install rapidocr-onnxruntime pillow opencv-python pypdf pymupdf   (no extra program)
"""
import hashlib
import io
import os
import re
import shutil
import threading

os.environ.setdefault("OMP_THREAD_LIMIT", "1")   # pages are read in parallel; avoid CPU oversubscription
_READ_CACHE = {}

REPORT_HINTS = {
    "sg":   "Urine Routine report → row \"Specific Gravity\"",
    "al":   "Urine Routine report → row \"Albumin\" / \"Protein\" (Nil, Trace, +, ++ …)",
    "su":   "Urine Routine report → row \"Sugar\" / \"Glucose\" (urine)",
    "bgr":  "Blood Sugar report → row \"Random Blood Sugar\" / \"Blood Glucose (Random)\"",
    "bu":   "Kidney Function Test (KFT) → row \"Blood Urea\"",
    "sc":   "Kidney Function Test (KFT) → row \"Serum Creatinine\"",
    "sod":  "Electrolytes panel → row \"Sodium (Na+)\"",
    "pot":  "Electrolytes panel → row \"Potassium (K+)\"",
    "hemo": "Complete Blood Count (CBC) → row \"Haemoglobin (Hb)\"",
    "pcv":  "Complete Blood Count (CBC) → row \"Packed Cell Volume (PCV) / Haematocrit\"",
    "wbcc": "Complete Blood Count (CBC) → row \"Total WBC Count\"",
    "rbcc": "Complete Blood Count (CBC) → row \"RBC Count\"",
}

_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"


def _to_float(s):
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", s):
        return float(s.replace(",", ""))
    if re.fullmatch(r"\d+,\d{1,2}", s):
        return float(s.replace(",", "."))
    return float(s)


def _lines(text):
    t = (text or "").lower().replace("|", " ").replace("_", " ").replace("µ", "u").replace("μ", "u")
    t = re.sub(r"\b([a-z])\.([a-z])\.(?:([a-z])\.?)?", lambda m: "".join(g for g in m.groups() if g), t)
    out = []
    for ln in t.splitlines():
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if ln:
            out.append(ln)
    return out


def _candidates(lines):
    """Single lines first, then pairs (label on one line, value on the next)."""
    for ln in lines:
        yield ln
    for i in range(len(lines) - 1):
        yield lines[i] + " " + lines[i + 1]


# Typical printed reference ranges (lower-bound window, upper-bound window). If the line carries a range that
# does not look like THIS test's range, the OCR paired the name with another row's value -> reject it.
_RANGE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:[.,]\d+)?)")
# printed reference range windows: (lowest start, highest start, lowest end, highest end)
_REF = {"hemo": (10.5, 13.6, 14.0, 18.0), "pcv": (32, 42, 44, 56), "wbcc": (3000, 4600, 9500, 12000),
        "rbcc": (3.3, 4.8, 4.7, 6.6), "sc": (0.3, 0.95, 1.0, 1.7), "sod": (128, 138, 142, 150),
        "pot": (3.0, 3.8, 4.7, 5.6), "bgr": (55, 82, 110, 210)}


def _ref_ok(key, lo, hi):
    w = _REF.get(key)
    if not w:
        return True
    if key == "wbcc" and hi < 100:
        lo, hi = lo * 1000, hi * 1000
    if key == "hemo" and lo > 30:
        lo, hi = lo / 10, hi / 10
    return w[0] <= lo <= w[1] and w[2] <= hi <= w[3]


def _find_number(lines, labels, lo, hi, skip_if=None, transform=None, key=None, suspects=None):
    """First plausible number after a test name. On lab tables the printed reference range on the SAME row is used as
    a check: a creatinine name sitting next to a 0.0-1.2 range belongs to another test (tilted photo), so the
    value from the neighbouring row whose range fits is taken instead; if nothing fits, the value is flagged."""
    def value_at(cand, m):
        rest = cand[m.end(): m.end() + 70]
        mn = re.search(_NUM, rest)
        if not mn:
            return None, None
        try:
            v = _to_float(mn.group(1))
        except ValueError:
            return None, None
        if transform:
            v = transform(v, cand)
        return v, _RANGE.search(rest[mn.end(): mn.end() + 60])

    rejected = []
    for label in labels:
        rx = re.compile(label)
        for i, cand in enumerate(lines):
            m = rx.search(cand)
            if not m or (skip_if and skip_if(cand)):
                continue
            v, mr = value_at(cand, m)
            if v is None or not (lo <= v <= hi):
                continue
            if key in _REF and mr:
                try:
                    if not _ref_ok(key, _to_float(mr.group(1)), _to_float(mr.group(2))):
                        rejected.append((i, v))
                        continue
                except ValueError:
                    pass
            return round(v, 3)
        for i in range(len(lines) - 1):                       # name at the end of one line, value on the next
            cand = lines[i] + " " + lines[i + 1]
            m = rx.search(cand)
            if not m or (skip_if and skip_if(cand)) or m.end() > len(lines[i]):
                continue
            mn = re.search(_NUM, cand[m.end(): m.end() + 70])
            if not mn or m.end() + mn.start() <= len(lines[i]):     # number must really be on the next line
                continue
            v, _mr = value_at(cand, m)
            if v is not None and lo <= v <= hi:
                return round(v, 3)
    if rejected and key in _REF:
        for i, v in rejected:
            for j in (i - 1, i + 1, i - 2, i + 2):
                if not (0 <= j < len(lines)):
                    continue
                for mr in _RANGE.finditer(lines[j]):
                    try:
                        if not _ref_ok(key, _to_float(mr.group(1)), _to_float(mr.group(2))):
                            continue
                    except ValueError:
                        continue
                    nums = re.findall(_NUM, lines[j][:mr.start()])
                    if not nums:
                        continue
                    try:
                        nv = _to_float(nums[-1])
                    except ValueError:
                        continue
                    if transform:
                        nv = transform(nv, lines[j])
                    if nv is not None and lo <= nv <= hi:
                        return round(nv, 3)
        if suspects is not None:
            suspects.append(key)
        return round(rejected[0][1], 3)
    return None


def _grade(rest):
    r = rest[:45]
    if re.search(r"\b(?:nil|negative|neg|absent|not detected|nad)\b", r):
        return 0
    if re.search(r"\btrace\b|\btr\b", r):
        return 1
    m = re.search(r"\b([1-4])\s*\+", r)
    if m:
        return int(m.group(1)) + 1
    m = re.search(r"(\+{1,4})", r)
    if m:
        return len(m.group(1)) + 1
    if re.search(r"\b(?:present|positive|detected)\b", r):
        return 2
    return None


def _find_grade(lines, label, skip_words):
    rx = re.compile(label)
    for cand in _candidates(lines):
        m = rx.search(cand)
        if not m or any(w in cand for w in skip_words):
            continue
        g = _grade(cand[m.end():])
        if g is not None:
            return g
    return None


def _micro_state(rest, normal_upper):
    r = rest[:45]
    if re.search(r"\b(?:nil|absent|occasional|not seen|none|nad)\b", r):
        return "normal"
    if re.search(r"\b(?:plenty|many|numerous|loaded|full field|present)\b|\+{1,4}", r):
        return "abnormal"
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", r)
    if m:
        return "normal" if int(m.group(2)) <= normal_upper else "abnormal"
    m = re.search(r"\b(\d+)\b", r)
    if m:
        return "normal" if int(m.group(1)) <= normal_upper else "abnormal"
    return None


def parse_report_text(text):
    """Returns {key: value} for everything that could be read with confidence."""
    L = _lines(text)
    found = {}
    suspects = []

    def put(k, v):
        if v is not None:
            found[k] = v

    # ── blood ──
    put("hemo", _find_number(
        L, [r"h[ae]+moglobin", r"\bhgb\b", r"\bhb\b"], 3, 20,
        skip_if=lambda c: "a1c" in c or "hba1c" in c,
        transform=lambda v, c: v / 10 if (v > 30 and "g/l" in c) else v, key="hemo", suspects=suspects))
    put("pcv", _find_number(L, [r"packed cell volume", r"\bpcv\b", r"h[ae]matocrit", r"\bhct\b"], 10, 60, key="pcv", suspects=suspects))
    put("wbcc", _find_number(
        L, [r"total (?:wbc|leu[ck]ocyte|w\.?b\.?c\.?)(?: count)?", r"total leu[ck]ocyte count", r"\btlc\b",
            r"\bwbc\b(?: count)?"],
        2000, 25000, transform=lambda v, c: v * 1000 if v < 100 else v, key="wbcc", suspects=suspects))
    put("rbcc", _find_number(
        L, [r"total rbc count", r"\brbc count", r"red blood cell count", r"red cell count"], 2, 7,
        skip_if=lambda c: "hpf" in c, key="rbcc", suspects=suspects))
    put("sc", _find_number(
        L, [r"creatin\w*"], 0.1, 20,
        skip_if=lambda c: any(w in c for w in ("urine", "clearance", "ratio", "acr", "24 hr", "24hr")),
        transform=lambda v, c: (v / 88.4) if ("umol" in c or (v > 20 and v <= 2000)) else v, key="sc", suspects=suspects))
    bun = _find_number(L, [r"blood urea nitrogen", r"\bbun\b", r"urea nitrogen"], 2, 140,
                       transform=lambda v, c: v * 2.14)
    urea = _find_number(L, [r"(?:blood |serum )?urea\b(?! nitrogen)"], 5, 300,
                        skip_if=lambda c: "nitrogen" in c or "urine" in c, key="bu", suspects=suspects)
    put("bu", urea if urea is not None else bun)
    put("sod", _find_number(L, [r"sodium", r"\bna\s?\+"], 100, 170, key="sod", suspects=suspects))
    put("pot", _find_number(L, [r"potassium", r"\bk\s?\+"], 2, 10, key="pot", suspects=suspects))
    put("bgr", _find_number(
        L, [r"random (?:blood )?s\w*", r"random (?:blood )?(?:sugar|glucose)", r"\brbs\b", r"glucose[ ,(]*random",
            r"blood (?:sugar|glucose)(?: \(?random\)?)?", r"plasma glucose", r"\bglucose\b"], 40, 700,
        skip_if=lambda c: any(w in c for w in ("urine", "hba1c", "a1c", "fasting", "postprandial", "post prandial",
                                               "ppbs", "fbs", "gtt", "tolerance", "pp ")),
        transform=lambda v, c: v * 18 if "mmol" in c else v, key="bgr", suspects=suspects))

    # ── urine ──
    put("sg", _find_number(L, [r"specific gravity", r"\bsp\.? ?gr", r"\bs\.g\.?\b"], 1.0, 1.04,
                           transform=lambda v, c: v / 1000 if 1000 <= v <= 1040 else v))
    al = _find_grade(L, r"(?:albumin|protein)", ("serum", "total protein", "globulin", "a/g", "g/dl", "g/l",
                                                  "micro", "creatinine", "acr", "24 hr"))
    acr = _find_number(L, [r"albumin[ /-]*creatinine ratio", r"\bacr\b", r"micro ?albumin(?:uria)?"], 0, 20000,
                       transform=lambda v, c: v * 8.84 if "mg/mmol" in c else v)
    if acr is not None:
        al = 0 if acr < 30 else 2 if acr < 100 else 3 if acr < 300 else 4 if acr < 1000 else 5
    put("al", al)
    put("su", _find_grade(L, r"(?:urine )?(?:sugar|glucose)", ("random", "fasting", "blood", "serum", "plasma",
                                                                "mg/dl", "mmol", "hba1c", "post")))

    # urine microscopy
    for cand in _candidates(L):
        if "pc_urine" not in found:
            m = re.search(r"pus cells?|\bpc\b|leu[ck]ocytes?", cand)
            if m and "clump" not in cand and "esterase" not in cand:
                st = _micro_state(cand[m.end():], 5)
                if st:
                    found["pc_urine"] = st
        if "rbc_urine" not in found:
            m = re.search(r"(?:\br\.?b\.?c\.?s?\b|red blood cells?|red cells?)", cand)
            if m and not any(w in cand for w in ("count", "million", "cumm", "mm3", "g/dl", "%", "volume",
                                                  "rdw", "mcv", "mch", "hemoglobin", "haemoglobin")):
                st = _micro_state(cand[m.end():], 2)
                if st:
                    found["rbc_urine"] = st
        if "pcc_urine" not in found:
            m = re.search(r"pus cell clumps?|\bpcc\b", cand)
            if m:
                r = cand[m.end(): m.end() + 40]
                if re.search(r"\b(?:nil|absent|not seen|none)\b", r):
                    found["pcc_urine"] = "notpresent"
                elif re.search(r"\b(?:present|seen|few|many)\b|\+", r):
                    found["pcc_urine"] = "present"
        if "ba_urine" not in found:
            m = re.search(r"bacteria|bacilli|cocci", cand)
            if m:
                r = cand[m.end(): m.end() + 40]
                if re.search(r"\b(?:nil|absent|not seen|none)\b", r):
                    found["ba_urine"] = "notpresent"
                elif re.search(r"\b(?:present|seen|few|many|plenty|moderate)\b|\+", r):
                    found["ba_urine"] = "present"
    if suspects:
        found["_check"] = sorted(set(suspects))      # values whose printed reference range did not fit -> ask the patient to check
    return found


# ── OCR engines ───────────────────────────────────────────────
_KW = ("kidney", "impression", "ultrasound", "sonography", "mri", "bladder", "renal", "prostate", "findings",
       "creatinine", "haemoglobin", "hemoglobin", "urea", "albumin", "specific gravity", "wbc", "urine")
_rapid_engine = {"obj": None}


def _score_text(t):
    low = (t or "").lower()
    r = parse_report_text(t)
    bad = len(r.get("_check", []))
    return 3 * (len(r) - (1 if bad else 0)) - 4 * bad + sum(1 for w in _KW if w in low)


def _has_suspect(t):
    return bool(parse_report_text(t).get("_check"))


def _setup_tesseract():
    try:
        import pytesseract
    except ImportError:
        return None
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd and os.path.exists(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
        return pytesseract
    if shutil.which("tesseract"):
        return pytesseract
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
              os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
              "/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"):
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return pytesseract
    return None


def _make_rapid(cls, kind):
    """Create the engine with a lighter text-detection size when this version accepts it (same accuracy for
    printed reports, noticeably faster). Falls back to the default settings."""
    tries = ([dict(det_limit_side_len=960, det_limit_type="max")] if kind == "old"
             else [dict(params={"Det.limit_side_len": 960, "Det.limit_type": "max"})])
    for kw in tries + [{}]:
        try:
            return cls(**kw)
        except Exception:
            continue
    return None


def _rapid():
    if _rapid_engine["obj"] is None:
        _rapid_engine["obj"] = False
        try:
            from rapidocr_onnxruntime import RapidOCR
            eng = _make_rapid(RapidOCR, "old")
            _rapid_engine["obj"] = ("old", eng) if eng else False
        except Exception:
            try:
                from rapidocr import RapidOCR
                eng = _make_rapid(RapidOCR, "new")
                _rapid_engine["obj"] = ("new", eng) if eng else False
            except Exception:
                _rapid_engine["obj"] = False
    return _rapid_engine["obj"] or None


_warm = {"started": False}


def warm_up():
    """Load the reader in the background when the app starts, so the first report doesn't pay the loading time."""
    if _warm["started"]:
        return
    _warm["started"] = True

    def _go():
        try:
            _setup_tesseract()
            _rapid()
        except Exception:
            pass

    threading.Thread(target=_go, daemon=True).start()


_install_state = {"tried": False}


def _auto_install():
    """One-time silent install of the free OCR pieces (RapidOCR + PyMuPDF) when none are present."""
    if _install_state["tried"] or os.environ.get("NEPHROSENSE_AUTO_INSTALL", "1") == "0":
        return
    _install_state["tried"] = True
    import importlib
    import subprocess
    import sys

    def pip(*pkgs):
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *pkgs],
                               capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                print("[report_reader] pip install failed:", pkgs, (r.stderr or "")[-300:])
            return r.returncode == 0
        except Exception as e:
            print("[report_reader] pip install error:", pkgs, repr(e))
            return False

    try:
        import fitz  # noqa: F401
    except ImportError:
        pip("pymupdf")
    if not (_setup_tesseract() or _rapid()):
        if not pip("rapidocr-onnxruntime"):
            pip("rapidocr", "onnxruntime")
    importlib.invalidate_caches()
    _rapid_engine["obj"] = None


def _gemini_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()


def uses_online_ai():
    return bool(_gemini_key())


_GEMINI_PROMPT = (
    "Transcribe this medical report EXACTLY as printed. Output plain text only: one table row per line, "
    "columns separated by ' | ' (Test | Result | Unit | Reference range). Keep every number, unit and word "
    "exactly as printed — do not correct, interpret, summarise or add anything. Include section headings, "
    "the sample-collected and reported dates, and for scan reports (ultrasound / MRI / CT) the full Findings "
    "and Impression text. Leave out the patient's name, address, phone number and doctor names."
)


def _gemini_transcribe(file_bytes, mime):
    key = _gemini_key()
    if not key:
        return ""
    import base64
    try:
        import requests
    except ImportError:
        return ""
    models = [m for m in (os.environ.get("GEMINI_MODEL"), "gemini-2.5-flash", "gemini-flash-latest",
                          "gemini-2.0-flash") if m]
    body = {
        "contents": [{"parts": [{"inline_data": {"mime_type": mime, "data": base64.b64encode(file_bytes).decode()}},
                                {"text": _GEMINI_PROMPT}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192},
    }
    for model in dict.fromkeys(models):
        if "2.5" in model:
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
        else:
            body["generationConfig"].pop("thinkingConfig", None)
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": key}, json=body, timeout=120)
            if r.status_code != 200:
                print(f"[report_reader] Gemini {model} -> HTTP {r.status_code}: {r.text[:200]}")
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "\n".join(p.get("text", "") for p in parts).strip()
            if len(text) >= 30:
                return text
        except Exception as e:
            print(f"[report_reader] Gemini {model} error: {e!r}")
    return ""


def ocr_status():
    """(ok, short message). Never shows install instructions to patients."""
    if _gemini_key():
        return True, "Report scanner ready (online AI)."
    engines = [n for n, ok in (("Tesseract", bool(_setup_tesseract())), ("RapidOCR", bool(_rapid()))) if ok]
    if engines:
        return True, "Report scanner ready (" + ", ".join(engines) + ")."
    return False, "The report scanner is getting ready — please try again in a minute, or type the values by hand."


def _prepare_image(pil):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(pil).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > 2600:
        s = 2600 / longest
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    elif longest < 1200:
        s = 1500 / longest
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img


def _skew_angle(gray):
    """Estimate page tilt (degrees) from how sharply text rows line up. Tilted phone photos make table
    rows slope, which pairs a test name with the wrong result — straightening fixes that."""
    import numpy as np
    from PIL import Image
    w, h = gray.size
    s = 700.0 / max(w, h)
    small = gray.resize((max(1, int(w * s)), max(1, int(h * s))))
    a = np.array(small).astype(float)
    mask = (a < a.mean() * 0.8).astype("uint8") * 255
    base = Image.fromarray(mask)

    def score(ang):
        rows = np.array(base.rotate(ang, resample=Image.NEAREST, fillcolor=0)).sum(axis=1).astype(float)
        return float(np.sum(np.diff(rows) ** 2))

    best = max((score(a_), a_) for a_ in np.arange(-10, 10.01, 0.5))[1]
    best = max((score(a_), a_) for a_ in np.arange(best - 0.5, best + 0.51, 0.1))[1]
    return float(best)


def _deskew(gray):
    from PIL import Image
    try:
        ang = _skew_angle(gray)
    except Exception:
        return gray
    if abs(ang) < 0.4:
        return gray
    return gray.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=255)


def _variants(img):
    """Lazy: the straightened page first; slower clean-ups only if earlier passes were poor."""
    from PIL import Image, ImageOps
    gray = ImageOps.autocontrast(img.convert("L"), cutoff=1)
    straight = _deskew(gray)
    yield straight
    if straight is not gray:
        yield gray
    try:
        import cv2
        import numpy as np
        g = cv2.fastNlMeansDenoising(np.array(straight), h=8)
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 12)
        yield Image.fromarray(th)
    except Exception:
        return


def _tess_text_conf(tess, img, psm):
    """Tesseract text plus its own average word confidence (0-100)."""
    d = tess.image_to_data(img, config=f"--oem 3 --psm {psm}", output_type=tess.Output.DICT)
    lines, confs = {}, []
    for i, txt in enumerate(d["text"]):
        if not str(txt).strip():
            continue
        try:
            c = float(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if c < 0:
            continue
        lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(str(txt))
        confs.append(c)
    text = "\n".join(" ".join(v) for _k, v in sorted(lines.items()))
    return text, (sum(confs) / len(confs) if confs else 0.0)


_rapid_lock = threading.Lock()
_last_stats = {"seconds": 0.0, "pages": 0, "engine": ""}


def last_stats():
    return dict(_last_stats)


def _rapid_read(pil):
    """RapidOCR -> (text, average_confidence 0-1, number_of_text_boxes). One engine, used one thread at a time."""
    eng = _rapid()
    if eng is None:
        return "", 0.0, 0
    import numpy as np
    w, h = pil.size
    m = max(w, h)
    if m > 1700:                                   # detection cost grows with pixels
        pil = pil.resize((int(w * 1700 / m), int(h * 1700 / m)))
    kind, engine = eng
    arr = np.array(pil.convert("RGB"))
    items, confs = [], []
    with _rapid_lock:
        if kind == "old":
            try:
                res, _ = engine(arr, use_cls=False)
            except TypeError:
                res, _ = engine(arr)
            for box, txt, sc in (res or []):
                ys = [q[1] for q in box]
                xs = [q[0] for q in box]
                items.append(((min(ys) + max(ys)) / 2.0, max(ys) - min(ys), min(xs), txt))
                try:
                    confs.append(float(sc))
                except (TypeError, ValueError):
                    pass
        else:
            try:
                out = engine(arr, use_cls=False)
            except TypeError:
                out = engine(arr)
            boxes, txts = getattr(out, "boxes", None), getattr(out, "txts", None)
            scs = getattr(out, "scores", None)
            if boxes is not None and txts is not None:
                for i, (box, txt) in enumerate(zip(boxes, txts)):
                    ys = [q[1] for q in box]
                    xs = [q[0] for q in box]
                    items.append(((min(ys) + max(ys)) / 2.0, max(ys) - min(ys), min(xs), txt))
                    if scs is not None:
                        try:
                            confs.append(float(scs[i]))
                        except (TypeError, ValueError, IndexError):
                            pass
    if not items:
        return "", 0.0, 0
    items.sort(key=lambda r: r[0])
    heights = sorted(i[1] for i in items)
    tol = 0.6 * heights[len(heights) // 2]
    lines, cur, cur_y = [], [], None
    for yc, hh, x, txt in items:
        if cur_y is None or abs(yc - cur_y) <= tol:
            cur.append((x, txt))
            cur_y = yc if cur_y is None else (cur_y + yc) / 2
        else:
            lines.append(cur)
            cur, cur_y = [(x, txt)], yc
    if cur:
        lines.append(cur)
    text = "\n".join("   ".join(t for _, t in sorted(ln)) for ln in lines)
    return text, (sum(confs) / len(confs) if confs else 0.8), len(items)


def _rapid_text(pil):
    return _rapid_read(pil)[0]


def ocr_image(pil):
    """Best OCR text for one page. Returns (text, error_or_None). A clear, upright page needs ONE pass."""
    img = _prepare_image(pil)
    tess = _setup_tesseract()
    rapid = _rapid()
    if not tess and not rapid:
        _auto_install()
        tess = _setup_tesseract()
        rapid = _rapid()
    if not tess and not rapid:
        return "", ocr_status()[1]

    best, best_score = "", -1
    readable = False

    def consider(t):
        nonlocal best, best_score
        if not t or not t.strip():
            return
        sc = _score_text(t)
        if sc > best_score:
            best, best_score = t, sc

    gen = _variants(img)
    first = next(gen)
    if tess:
        try:
            t, conf = _tess_text_conf(tess, first, 6)
            consider(t)
            if conf >= 70 and sum(ch.isalpha() for ch in t) >= 100 and not _has_suspect(t):
                return best, None                      # clear, upright, readable -> done
        except Exception:
            pass
    if rapid:
        try:
            t, conf, n = _rapid_read(first.convert("RGB"))
            consider(t)
            if n >= 8 and conf >= 0.75 and not _has_suspect(t):   # readable page (even if it has no lab values)
                return best, None
            readable = n >= 8 and conf >= 0.6
        except Exception:
            pass
    # unclear page -> a few more passes
    if tess and best_score < 14:
        try:
            consider(tess.image_to_string(first, config="--oem 3 --psm 4"))
        except Exception:
            pass
        if best_score < 14:
            for v in gen:
                try:
                    consider(tess.image_to_string(v, config="--oem 3 --psm 6"))
                except Exception:
                    continue
                if best_score >= 14:
                    break
    if best_score < 4 and not readable:   # maybe sideways / upside-down
        for angle in (90, 270, 180):
            if tess:
                try:
                    consider(tess.image_to_string(first.rotate(angle, expand=True), config="--oem 3 --psm 6"))
                except Exception:
                    pass
            if rapid and best_score < 4:
                try:
                    consider(_rapid_text(img.rotate(angle, expand=True)))
                except Exception:
                    pass
            if best_score >= 4:
                break
    if not best.strip():
        return "", "Couldn't read this photo clearly. Please retake it flat, in bright light, with the whole page in view."
    return best, None


def _pdf_pages(pdf_bytes):
    """("text", str) | ("images", [PIL pages]) | ("error", message).
    Scanned PDFs normally hold one photo per page: those are taken directly with pypdf, so NO extra
    package (and no pip install) is needed. PyMuPDF is only a fallback."""
    text, reader = "", None
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        text = ""
    if len(text.strip()) >= 80:
        return "text", text
    if reader is not None:
        try:
            from PIL import Image
            pages = []
            for pg in reader.pages:
                best = None
                for im in pg.images:
                    pil = Image.open(io.BytesIO(im.data))
                    pil.load()
                    if best is None or pil.size[0] * pil.size[1] > best.size[0] * best.size[1]:
                        best = pil
                if best is None:
                    raise ValueError("page without an embedded image")
                pages.append(best)
            if pages:
                return "images", pages
        except Exception as e:
            print("[report_reader] embedded-image extraction failed:", repr(e))
    try:
        import fitz  # PyMuPDF
    except ImportError:
        _auto_install()
        try:
            import fitz  # noqa: F401
        except ImportError:
            return "error", ("This PDF could not be opened right now. Please upload clear photos (JPG/PNG) "
                             "of the pages instead, or type the values by hand.")
    try:
        from PIL import Image
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [Image.open(io.BytesIO(pg.get_pixmap(dpi=170).tobytes("png"))) for pg in doc]
        return ("images", pages) if pages else ("error", "This PDF has no pages.")
    except Exception as e:
        return "error", f"Could not read that PDF ({type(e).__name__})."


def _pdf_text(pdf_bytes):
    kind, payload = _pdf_pages(pdf_bytes)
    if kind == "text":
        return payload, None
    if kind == "error":
        return "", payload
    results = [ocr_image(im) for im in payload]
    parts = [t for t, _e in results if t and t.strip()]
    if not parts:
        return "", next((e for _t, e in results if e), "Couldn't read this PDF.")
    return "\n".join(parts), None


_text_cache = {}


def reader_ready():
    """True when a reader is available right now (no one-time setup pending)."""
    return bool(_gemini_key() or _setup_tesseract() or _rapid())


def _mime(ext):
    return {"pdf": "application/pdf", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")


def read_text_from_file(file_bytes, filename):
    """(text, error) for one uploaded photo / scan / PDF. Results are cached per file content."""
    key = hashlib.sha1(file_bytes).hexdigest()
    if key in _READ_CACHE:
        return _READ_CACHE[key], None
    text, err = _read_text_uncached(file_bytes, filename)
    if text and text.strip():
        if len(_READ_CACHE) > 40:
            _READ_CACHE.clear()
        _READ_CACHE[key] = text
    return text, err


def _read_text_uncached(file_bytes, filename):
    ext = (filename or "").lower().rsplit(".", 1)[-1]
    if _gemini_key():
        try:
            data, mime = file_bytes, _mime(ext)
            if ext != "pdf":
                from PIL import Image
                img = _prepare_image(Image.open(io.BytesIO(file_bytes)))
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=88)
                data, mime = buf.getvalue(), "image/jpeg"
            t = _gemini_transcribe(data, mime)
            if t:
                return t, None
        except Exception as e:
            print("[report_reader] Gemini path failed:", repr(e))
    if ext == "pdf":
        return _pdf_text(file_bytes)
    try:
        from PIL import Image
        pil = Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        return "", f"Could not open that image ({type(e).__name__}). Try a JPG or PNG photo."
    return ocr_image(pil)


def read_text_from_files(files, progress=None):
    """files: [(bytes, filename)] -> (combined_text, error_or_None). Every PAGE is one task, so progress moves page
    by page. Pages run in parallel with Tesseract/online AI; RapidOCR shares one engine, so it runs one at a time."""
    import hashlib
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    files = list(files)
    if not files:
        return "", "No file was uploaded."
    ckey = hashlib.sha1(b"".join(hashlib.sha1(b).digest() for b, _n in files)).hexdigest()
    if ckey in _text_cache:
        return _text_cache[ckey], None

    t0 = time.time()
    online = bool(_gemini_key())
    tasks, first_err = [], None
    for b, n in files:
        ext = (n or "").lower().rsplit(".", 1)[-1]
        if online:
            tasks.append(("file", (b, n)))
        elif ext == "pdf":
            kind, payload = _pdf_pages(b)
            if kind == "text":
                tasks.append(("text", payload))
            elif kind == "images":
                tasks += [("img", im) for im in payload]
            else:
                first_err = first_err or payload
        else:
            try:
                from PIL import Image
                im = Image.open(io.BytesIO(b))
                im.load()
                tasks.append(("img", im))
            except Exception as e:
                first_err = first_err or f"Could not open that image ({type(e).__name__}). Try a JPG or PNG photo."
    if not tasks:
        return "", first_err or "Nothing could be read from the uploaded file(s)."

    def run(task):
        kind, payload = task
        if kind == "text":
            return payload, None
        if kind == "img":
            return ocr_image(payload)
        return _read_text_uncached(payload[0], payload[1])

    tess = _setup_tesseract()
    workers = min(4, len(tasks)) if (online or tess) else 1
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            try:
                results[futs[fut]] = fut.result()
            except Exception as e:
                results[futs[fut]] = ("", f"Could not read a page ({type(e).__name__}).")
            done += 1
            if progress:
                try:
                    progress(done, len(tasks))
                except Exception:
                    pass
    texts = [t for t, _e in results if t and t.strip()]
    _last_stats.update(seconds=time.time() - t0, pages=len(tasks),
                       engine="online AI" if online else ("Tesseract" if tess else ("RapidOCR" if _rapid() else "")))
    print(f"[report_reader] {len(tasks)} page(s) read in {_last_stats['seconds']:.1f}s via {_last_stats['engine']}")
    if not texts:
        return "", first_err or next((e for _t, e in results if e), "Nothing could be read from the uploaded file(s).")
    combined = "\n".join(texts)
    _text_cache[ckey] = combined
    return combined, None


def extract_values_from_files(files, progress=None):
    """Returns (found_values, raw_text_or_None, error_or_None) for one or many pages."""
    text, err = read_text_from_files(files, progress)
    if not text:
        return {}, None, err
    return parse_report_text(text), text, None


def extract_values_from_report(file_bytes, filename):
    return extract_values_from_files([(file_bytes, filename)]) 
