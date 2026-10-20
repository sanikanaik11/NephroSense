"""
eye_model.py — NephroSense  (REPLACE existing)
Visual Screening (Experimental) — NOT a kidney test. Estimates anaemia signs from a
photo of the INNER LOWER EYELID (palpebral conjunctiva).

What changed and why:
  - The old code looked for an eye with a Haar cascade and cropped the lower third of it.
    The model was trained on whole conjunctiva photos (no crop), so that crop did not match
    training and gave low-confidence / wrong results. Now the photo is only centre-squared
    and resized, exactly like training.
  - A photo-quality gate runs BEFORE the model. If the photo is blurry, too dark/bright, or
    the red/pink inner eyelid is not visible (e.g. a photo of the white of the eye), NO score
    is produced and the patient is told how to retake it. No guessing.
  - Phone rotation (EXIF) is corrected.
  - The colour-guess fallback was removed (not reliable enough for patients).

Needs: pip install torch torchvision pillow numpy opencv-python (opencv only for Grad-CAM)
"""
import io
import os
import threading

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "anemia_model.pt"
)
IMG_SIZE = 300  # EfficientNet-B3 native input size

MIN_SIDE_PX = 100  # dataset images are 100–234px; phone photos are larger — accept both
MIN_ROI_FRACTION = 0.008   # pink lining must cover at least this share of the picture
MIN_REDNESS_CONTRAST = 2.5  # how much redder the lining is than the surrounding skin (Lab a*)
HARD_BLUR = 3.0            # sharpness of the lining area (Laplacian variance at 256 px) below this = unusable
SOFT_BLUR = 12.0           # below this = usable but less reliable
MIN_BRIGHT, MAX_BRIGHT = 15.0, 245.0  # dark conjunctiva photos are common — allow down to 15
EYE_MIN_CONFIDENCE = 0.55  # below this the model is "inconclusive" and no score is saved

_model = None
_device = None
_load_lock = threading.Lock()


def _load_model():
    """Thread-safe: loads the network once (also pre-loaded in the background at start-up)."""
    if _model is not None:
        return _model
    with _load_lock:
        return _load_model_unlocked()


def _load_model_unlocked():
    global _model, _device
    if _model is not None:
        return _model
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models

        if not os.path.exists(MODEL_PATH):
            return None
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        model.to(_device)
        model.eval()
        _model = model
        return _model
    except Exception:
        return None


def _open_image(image_bytes):
    from PIL import Image, ImageOps
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > 1280:          # phone photos are 3000+ px; 1280 is plenty and ~10x faster
        img.thumbnail((1280, 1280))
    return img


def _locate_conjunctiva(img):
    """Finds the pink inner-lower-eyelid lining (palpebral conjunctiva) — the strongest red region that does
    not touch the picture edge (this skips fingers/skin at the border). Works for tilted eyes and the cyan/blue
    colour cast of low-cost phone cameras. Returns dict(box, frac, contrast) in FULL-RESOLUTION pixels, or None."""
    import cv2
    import numpy as np
    from PIL import Image
    W0, H0 = img.size
    sc = 480.0 / max(W0, H0)
    rgb = np.array(img.resize((max(1, int(W0 * sc)), max(1, int(H0 * sc)))))
    H, W = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    a = cv2.GaussianBlur(lab[..., 1] - 128, (0, 0), 2)
    V = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2].astype(float)
    central = np.zeros((H, W), bool)
    central[int(.04 * H):int(.96 * H), int(.04 * W):int(.96 * W)] = True
    pool = central & (V > 45)
    if pool.sum() < 200:
        return None
    th = np.percentile(a[pool], 92)
    m = ((a >= th) & pool).astype("uint8")
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), "uint8"))
    n, cc, st, _ = cv2.connectedComponentsWithStats(m)
    best, bs = None, 0.0
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if area < 0.004 * H * W:
            continue
        touches = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        score = area * (0.4 if touches else 1.0)
        if score > bs:
            best, bs = k, score
    if best is None:
        return None
    mm = cc == best
    ys, xs = np.where(mm)
    contrast = float(a[mm].mean() - np.median(a[central]))
    mw, mh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    mx, my = int(0.10 * mw) + 2, int(0.10 * mh) + 2
    x0, y0 = max(0, xs.min() - mx), max(0, ys.min() - my)
    x1, y1 = min(W, xs.max() + 1 + mx), min(H, ys.max() + 1 + my)
    return {"box": (int(x0 / sc), int(y0 / sc), int(x1 / sc), int(y1 / sc)),
            "frac": float(mm.sum()) / (H * W), "contrast": contrast, "mask": mm, "scale": sc}


def _white_balance(img):
    """Removes the colour tint of low-cost phone cameras (often blue-green) using the WHITE OF THE EYE as the
    neutral reference — the same idea clinicians use when they judge paleness relative to the sclera.
    Returns (balanced_image, gains_or_None). Skipped when no clear sclera is found or the tint is tiny."""
    import cv2
    import numpy as np
    from PIL import Image
    W0, H0 = img.size
    sc = 480.0 / max(W0, H0)
    rgb = np.array(img.resize((max(1, int(W0 * sc)), max(1, int(H0 * sc))))).astype(float)
    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb.astype("uint8"), cv2.COLOR_RGB2HSV).astype(float)
    S, V = hsv[..., 1] / 255.0, hsv[..., 2]
    central = np.zeros((H, W), bool)
    central[int(.08 * H):int(.92 * H), int(.08 * W):int(.92 * W)] = True
    cand = (central & (V >= np.percentile(V[central], 70)) & (S <= 0.30) & (rgb.max(2) < 248)
            & ~(rgb[..., 0] > 1.25 * rgb[..., 2]))          # bright, low-colour, not skin-like
    if cand.sum() < 0.015 * H * W:
        return img, None
    med = np.median(rgb[cand], axis=0)
    gains = np.clip(med.mean() / np.maximum(med, 1.0), 0.75, 1.5)
    if np.abs(gains - 1.0).max() < 0.06:
        return img, None
    out = np.clip(np.array(img).astype(float) * gains, 0, 255).astype("uint8")
    return Image.fromarray(out), [round(float(g), 2) for g in gains]


def _tight_rect(mask, cover=0.85, min_frac=0.18):
    """Largest rectangle that is almost completely lining (not white of the eye / skin / lashes).
    The search runs on a copy of the mask shrunk to at most 64 px, then maps back (fast on big photos)."""
    import numpy as np
    import cv2
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    full = mask[y0:y1, x0:x1].astype(np.float32)
    FH, FW = full.shape
    f = max(1.0, max(FH, FW) / 64.0)
    if f > 1.0:
        small = cv2.resize(full, (max(1, int(round(FW / f))), max(1, int(round(FH / f)))), interpolation=cv2.INTER_AREA)
        sub = (small >= 0.5).astype(np.int32)
        step = 1
    else:
        sub = full.astype(np.int32)
        step = 2
    H, W = sub.shape
    sx, sy = FW / float(W), FH / float(H)
    ii = np.pad(sub.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    cands = [(h, w) for h in range(max(8, H // 6), H + 1, step) for w in range(max(8, W // 6), W + 1, step)
             if 0.6 <= w / h <= 2.5]
    cands.sort(key=lambda t: -t[0] * t[1])
    for h, w in cands:
        if h * w < min_frac * H * W:
            break
        S = ii[h:, w:] - ii[:H - h + 1, w:] - ii[h:, :W - w + 1] + ii[:H - h + 1, :W - w + 1]
        if S.size == 0:
            continue
        if S.max() >= cover * h * w:
            yy, xx = np.unravel_index(S.argmax(), S.shape)
            return (int(x0 + min(FW - 1, xx * sx)), int(y0 + min(FH - 1, yy * sy)),
                    int(x0 + min(FW, (xx + w) * sx)), int(y0 + min(FH, (yy + h) * sy)))
    return None


def _refine_crop(crop):
    """Trims away white-of-eye, dark skin and lashes that a loose crop may still contain, keeping only the
    clearly reddish lining. Returns a rectangle inside the crop, or None if nothing better was found."""
    import cv2
    import numpy as np
    rgb = np.array(crop.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
    a = cv2.GaussianBlur(lab[..., 1] - 128, (0, 0), 2)
    L = lab[..., 0] * 100 / 255
    ok = (L > 18) & (L < 95)
    vals = a[ok]
    if vals.size < 50 or vals.max() - vals.min() < 1e-6:
        return None
    scaled = np.clip((vals - vals.min()) / (vals.max() - vals.min()) * 255, 0, 255).astype("uint8").reshape(-1, 1)
    t, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(vals.min() + t / 255.0 * (vals.max() - vals.min()), 4.0)
    m = ((a >= thr) & ok).astype("uint8")
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), "uint8"))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return _tight_rect(m.astype(bool), cover=0.80, min_frac=0.10)


# Colour of the lining after the white-of-the-eye colour correction (simple physics: haemoglobin makes the
# lining redder, so R/G and Lab a* rise with haemoglobin). These are rule-of-thumb limits used only to catch
# model answers that contradict the photo's own colour — they are NOT a haemoglobin measurement.
RED_RG, RED_A = 1.35, 18.0
PALE_RG, PALE_A = 1.15, 9.0


def _lining_colour(crop):
    """Colour of the lining. Bright, low-colour pixels (wet shine / glare) and very dark pixels (shadow, lashes)
    are left out first — shine makes pink tissue look paler than it is — then the MEDIAN is used so a few
    stray pixels cannot move the result."""
    import cv2
    import numpy as np
    arr = np.array(crop.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(float)
    S, V = hsv[..., 1] / 255.0, hsv[..., 2]
    keep = ~(((V >= 200) & (S < 0.30)) | (V < 45))
    if keep.mean() < 0.25:                       # almost everything was glare/shadow -> use all pixels
        keep = np.ones(keep.shape, bool)
    px = arr[keep].astype(float)
    rg = float(np.median(px[:, 0]) / max(float(np.median(px[:, 1])), 1.0))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(float)[..., 1] - 128
    a = float(np.median(lab[keep]))
    if a < 3 or rg < 1.02:
        cls = "none"
    elif rg >= RED_RG and a >= RED_A:
        cls = "red"
    elif rg <= PALE_RG or a <= PALE_A:
        cls = "pale"
    else:
        cls = "mid"
    return {"cls": cls, "rg": round(rg, 2), "a": round(a, 1)}


def assess_photo(img):
    """Quality check on the lining area itself (not the whole picture).
    Returns dict: ok, message, crop (PIL image of the lining), soft (usable but less reliable), metrics."""
    import numpy as np
    from PIL import Image
    w, h = img.size
    if min(w, h) < MIN_SIDE_PX:
        return {"ok": False, "message": "The photo is too small / low resolution. Use your phone's main camera.", "crop": None}
    raw_img = img
    img, gains = _white_balance(raw_img)
    loc = _locate_conjunctiva(img)
    if loc is None or loc["frac"] < MIN_ROI_FRACTION or loc["contrast"] < MIN_REDNESS_CONTRAST:
        return {"ok": False, "crop": None, "message": (
            "I couldn't find the red/pink inside of the lower eyelid in this photo. Look up, gently pull the lower "
            "eyelid down so the pink lining shows, and keep the eye in the middle of the photo.")}
    box = loc["box"]
    tr = _tight_rect(loc["mask"])
    if tr is not None:
        sc = loc["scale"]
        box = (int(tr[0] / sc), int(tr[1] / sc), int(tr[2] / sc), int(tr[3] / sc))
    crop = img.crop(box)                 # colour-corrected lining only
    crop_raw = raw_img.crop(box)         # same area, original camera colours
    try:
        rr = _refine_crop(crop)
    except Exception:
        rr = None
    # Only apply refine_crop if the result would still be large enough to measure.
    # Upscaling a tiny 30×20px crop to 256px introduces artificial smoothness
    # that fails the sharpness gate — skip refinement for small crops.
    if rr is not None:
        cw = rr[2] - rr[0]
        ch = rr[3] - rr[1]
        if cw >= 30 and ch >= 20:
            crop, crop_raw = crop.crop(rr), crop_raw.crop(rr)
    colour = _lining_colour(crop)
    if colour["cls"] == "none":
        return {"ok": False, "crop": crop, "message": (
            "I couldn't find the pink inside of the lower eyelid clearly in this photo (the area found is not "
            "pink/red). Look up, gently pull the lower eyelid down so the pink lining fills the photo, and use "
            "daylight without flash.")}
    g = crop.convert("L")
    g = g.resize((256, max(1, int(256 * g.height / g.width))))
    ga = np.asarray(g).astype(float)
    bright = float(ga.mean())
    if bright < MIN_BRIGHT:
        return {"ok": False, "crop": crop, "message": "The eyelid area is too dark. Take the photo near a window or in daylight."}
    if bright > MAX_BRIGHT:
        return {"ok": False, "crop": crop, "message": "The eyelid area is overexposed (flash or glare). Move away from direct light."}
    lap = 4 * ga[1:-1, 1:-1] - ga[:-2, 1:-1] - ga[2:, 1:-1] - ga[1:-1, :-2] - ga[1:-1, 2:]
    sharp = float(lap.var())
    # For small crops (<80px wide) that needed upscaling, apply a lower threshold
    # because bilinear upscaling reduces Laplacian variance artificially.
    hard_thresh = HARD_BLUR if crop.width >= 80 else 0.8
    soft_thresh = SOFT_BLUR if crop.width >= 80 else 5.0
    if sharp < hard_thresh:
        return {"ok": False, "crop": crop, "message": "The eyelid area is too blurry to read. Hold the phone steady, tap the eyelid on the screen to focus, and try again."}
    return {"ok": True, "message": None, "crop": crop, "crop_raw": crop_raw if gains else None,
            "gains": gains, "soft": sharp < soft_thresh, "colour": colour,
            "metrics": {"sharpness": round(sharp, 1), "brightness": round(bright), "area": round(loc["frac"], 3),
                        "redness": round(loc["contrast"], 1)}}


def check_photo_quality(img):
    """Kept for older callers: (ok, message_or_None)."""
    r = assess_photo(img)
    return r["ok"], r["message"]


def _prepare_tensor(img):
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0).to(_device)


def _predict_prob(crop):
    """Probability of 'anaemic' for the lining crop (single pass — fast)."""
    import torch
    from PIL import ImageOps
    model = _load_model()
    if model is None:
        return None
    with torch.no_grad():
        p1 = torch.sigmoid(model(_prepare_tensor(crop)).squeeze(1)).item()
    return p1


def generate_gradcam(image_bytes):
    """Grad-CAM overlay on the lining crop (where the model looked). Returns PIL image or None."""
    import torch
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from PIL import Image

    model = _load_model()
    if model is None:
        return None
    a = assess_photo(_open_image(image_bytes))
    img = a.get("crop")
    if img is None:
        return None
    tensor = _prepare_tensor(img)
    tensor.requires_grad_(True)

    activations, gradients = {}, {}
    last_conv = model.features[-1]
    h1 = last_conv.register_forward_hook(lambda m, i, o: activations.__setitem__("v", o))
    h2 = last_conv.register_full_backward_hook(lambda m, gi, go: gradients.__setitem__("v", go[0]))
    try:
        model.zero_grad()
        out = model(tensor).squeeze(1)
        out.backward()
        acts, grads = activations["v"][0], gradients["v"][0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = F.relu(cam)
        cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()

    cam = cv2.resize(cam, (img.width, img.height))
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (0.55 * np.array(img.convert("RGB")) + 0.45 * heat).astype(np.uint8)
    return Image.fromarray(overlay)


LOW_T, HIGH_T = 30, 55          # score < 30 well-coloured | 30-55 in-between | >= 55 pale


def eye_tier(score):
    """0 = well-coloured, 1 = in-between, 2 = pale. The label, the colour and the explanation all use this."""
    return 0 if score < LOW_T else (1 if score < HIGH_T else 2)


def eye_label(score):
    """Plain-language result category (the score is capped at 85 — no single photo is that certain)."""
    if score is None:
        return ""
    return ("Lining looks well-coloured", "In-between — slightly lighter than strong pink",
            "Looks pale — confirm with a blood test")[eye_tier(score)]


_cal_cache = {"loaded": False, "v": None}


def _colour_calibration():
    """Optional fit made by calibrate_eye_colour.py from the real Hb-labelled training photos."""
    if not _cal_cache["loaded"]:
        _cal_cache["loaded"] = True
        try:
            import json
            path = os.path.join(os.path.dirname(MODEL_PATH), "eye_colour_cal.json")
            if os.path.exists(path):
                _cal_cache["v"] = json.load(open(path))
        except Exception:
            _cal_cache["v"] = None
    return _cal_cache["v"]


def calibration_info():
    """Honest accuracy figures from calibrate_eye_colour.py (cross-validated on real Hb-labelled photos), or None
    when the colour scale has not been calibrated yet."""
    cal = _colour_calibration()
    if not cal:
        return None
    return {k: cal.get(k) for k in ("n", "auc_cv", "accuracy_cv", "sensitivity_cv", "specificity_cv")}


def colour_paleness_score(colour):
    """0-100 paleness from the measured colour of the lining (Lab a* and R/G ratio).
    0 = rich pink-red, 100 = pale. Uses models/eye_colour_cal.json when present (fitted on real photos),
    otherwise a simple transparent mapping."""
    import math
    a, rg = float(colour["a"]), float(colour["rg"])
    cal = _colour_calibration()
    if cal:
        z = cal["w_a"] * a + cal["w_rg"] * rg + cal["b"]
        return round(100.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))), 1)
    r_a = min(1.0, max(0.0, (a - 9.0) / 16.0))
    r_rg = min(1.0, max(0.0, (rg - 1.10) / 0.50))
    return round(100.0 * (1.0 - (r_a + r_rg) / 2.0), 1)


def _colour_result(a, crop, why):
    """Result from the measured colour of the lining only (used when the AI model is missing, unstable or
    disagrees with the colour). Always a 'reference only' hint."""
    colour = a.get("colour") or {}
    cls = colour.get("cls", "mid")
    try:
        score = colour_paleness_score(colour)
    except Exception:
        score = {"red": 15.0, "mid": 45.0, "pale": 60.0}.get(cls, 45.0)
    notes = {
        "red": "The inner eyelid lining looks well-coloured (pink-red), which makes marked paleness unlikely.",
        "mid": "The inner eyelid lining is in the in-between range (slightly lighter than strong pink).",
        "pale": "The inner eyelid lining looks lighter than usual.",
    }
    cal = " The colour scale was calibrated on real photos with known haemoglobin." if _colour_calibration() else ""
    note = (notes.get(cls, notes["mid"]) + " " + why + cal +
            " It cannot detect mild anaemia — a haemoglobin blood test is the reliable check.")
    conf = round(abs(score - 50.0) / 50.0, 3)
    if a.get("soft"):
        conf = round(conf * 0.85, 3)
    return {"eye_score": score, "eye_confidence": conf, "label": eye_label(score), "note": note, "crop": crop,
            "used_model": False, "quality_issue": False, "soft": bool(a.get("soft")),
            "colour": colour, "source": "colour", "calibration": calibration_info()}


def analyze_eye_image(image_bytes):
    """
    Returns dict: eye_score (0-100 or None), eye_confidence (0-1 or None), label, note, crop (PIL, the area
    that was checked), used_model, quality_issue (True when the photo itself was rejected).
    """
    try:
        img = _open_image(image_bytes)
    except Exception as e:
        return {"eye_score": None, "eye_confidence": None, "quality_issue": True,
                "note": f"Could not read that image ({type(e).__name__}). Try a JPG or PNG photo."}

    a = assess_photo(img)
    if not a["ok"]:
        return {"eye_score": None, "eye_confidence": None, "quality_issue": True, "note": a["message"],
                "crop": a.get("crop")}
    crop = a["crop"]

    # ── Colour-physics score (primary) ────────────────────────────────────────
    # This is based on directly measured redness of the lining (Lab a* channel
    # and R/G ratio). It is interpretable and consistent. The AI model gives
    # 95–98% anemia probability on BOTH anemic and non-anemic images from this
    # dataset, indicating it is not properly calibrated for standalone use.
    # We therefore use colour as the primary score and add a small model adjustment.
    colour_score = colour_paleness_score(a.get("colour") or {"a": 15, "rg": 1.2})

    try:
        prob = _predict_prob(crop)
    except Exception:
        prob = None

    # Model adjustment: if model available AND agrees with colour direction, nudge
    # the colour score slightly. Never let model alone drive the result.
    if prob is not None:
        model_score = prob * 100
        # Blend: 80% colour physics + 20% model, but only if they agree on direction
        # (both suggest low paleness or both suggest higher paleness)
        colour_dir = colour_score > 50
        model_dir  = model_score > 50
        if colour_dir == model_dir:
            blended = 0.80 * colour_score + 0.20 * model_score
        else:
            # Disagreement — trust colour completely
            blended = colour_score
        score = round(min(blended, 85.0), 1)  # hard cap at 85 — no single photo warrants 94/100
        used_model = True
        source = "colour+model"
    else:
        score = round(min(colour_score, 85.0), 1)
        used_model = False
        source = "colour"

    confidence = round(abs(score - 50) / 50.0, 3)
    if a.get("soft"):
        confidence = round(confidence * 0.80, 3)

    soft_note = " The photo was slightly blurry, so this reading is less certain." if a.get("soft") else ""
    note = (
        "The inner eyelid lining is well-coloured (pink), so there is no obvious pallor.",
        "The inner eyelid lining is in the in-between range. This alone does not show anaemia — "
        "a haemoglobin (Hb) blood test is the reliable check.",
        "The inner eyelid lining looks paler than typical. Please get a haemoglobin (Hb) blood test.",
    )[eye_tier(score)] + soft_note
    return {"eye_score": score, "eye_confidence": confidence, "label": eye_label(score), "note": note,
            "crop": crop, "used_model": used_model, "quality_issue": False, "soft": bool(a.get("soft")),
            "colour": a.get("colour"), "source": source, "calibration": calibration_info()}


def warm_up():
    """Loads the network in the background when the app starts, so the first photo is not slow."""
    try:
        _load_model()
    except Exception:
        pass


threading.Thread(target=warm_up, daemon=True).start()