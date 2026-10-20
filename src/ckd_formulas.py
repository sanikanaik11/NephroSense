"""
ckd_formulas.py — NephroSense
Pure-Python helpers used by the test-free (home) CKD risk screen.
No external dependencies needed.
"""
from __future__ import annotations

# ── BMI ───────────────────────────────────────────────────────────────────────
def bmi(weight_kg: float, height_cm: float) -> float | None:
    """Returns BMI (kg/m²) or None for invalid inputs."""
    try:
        h, w = float(height_cm), float(weight_kg)
    except (TypeError, ValueError):
        return None
    if h < 100 or h > 230 or w < 20 or w > 300:
        return None
    return round(w / (h / 100.0) ** 2, 1)


# ── Home BP average ───────────────────────────────────────────────────────────
MIN_HOME_READINGS = 3   # minimum readings to calculate a reliable average


def home_bp_average(readings: list) -> dict:
    """
    readings: list of (day, period, sbp, dbp) tuples.
    Returns dict(ok, n_used, sbp, dbp).
    - ok: True when we have enough valid readings.
    - Discards the first morning reading of day 1 (observer effect).
    - Uses remaining readings with valid SBP > DBP.
    """
    valid = []
    for r in readings:
        try:
            day, period, sbp, dbp = r[0], r[1], float(r[2]), float(r[3])
        except (TypeError, ValueError, IndexError):
            continue
        if day == 1 and period == "am":
            continue                         # discard first morning (observer effect)
        if 60 < sbp < 300 and 40 < dbp < 200 and sbp > dbp:
            valid.append((sbp, dbp))
    n = len(valid)
    if n < 1:
        return {"ok": False, "n_used": 0, "sbp": None, "dbp": None}
    avg_sbp = round(sum(r[0] for r in valid) / n, 1)
    avg_dbp = round(sum(r[1] for r in valid) / n, 1)
    return {"ok": n >= MIN_HOME_READINGS, "n_used": n, "sbp": avg_sbp, "dbp": avg_dbp}


def home_bp_raised(sbp: float | None, dbp: float | None) -> bool:
    """True when average BP is hypertensive (≥130/80)."""
    if sbp is None or dbp is None:
        return False
    return sbp >= 130 or dbp >= 80


def valid_bp(sbp: float | int, dbp: float | int) -> bool:
    """True when both numbers look like real blood-pressure values."""
    try:
        s, d = float(sbp), float(dbp)
    except (TypeError, ValueError):
        return False
    return 60 < s < 300 and 40 < d < 200 and s > d
