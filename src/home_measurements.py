"""
home_measurements.py — NephroSense
Guided 7-day home BP routine, BMI from weight/height, waist check.

BP protocol (ESH / AHA home-BP guidance): 2 readings, 1 min apart, morning and
evening for 7 days; discard day 1; need >= 12 valid readings on >= 3 days.
"""
MIN_READINGS = 12
MIN_DAYS = 3


def _valid(r):
    try:
        s, d = float(r["sys"]), float(r["dia"])
    except (KeyError, TypeError, ValueError):
        return False
    return 70 <= s <= 260 and 40 <= d <= 150 and s > d


valid_reading = _valid


def home_bp_average(readings):
    """readings: list of {"day": 1-7, "sys": .., "dia": ..}. Never guesses."""
    good = [r for r in (readings or []) if _valid(r) and r.get("day") is not None]
    days_all = {int(r["day"]) for r in good}
    if len(days_all) >= 4:                       # discard first day only if enough remain
        good = [r for r in good if int(r["day"]) != min(days_all)]
    days = {int(r["day"]) for r in good}
    if len(good) < MIN_READINGS or len(days) < MIN_DAYS:
        return {"ok": False, "n": len(good), "days": len(days),
                "message": f"Need at least {MIN_READINGS} valid readings on {MIN_DAYS}+ different "
                           f"days (have {len(good)} on {len(days)}). Keep going with the 7-day routine."}
    n = len(good)
    return {"ok": True, "n": n, "days": len(days),
            "sbp": round(sum(float(r["sys"]) for r in good) / n, 1),
            "dbp": round(sum(float(r["dia"]) for r in good) / n, 1)}


def bmi(weight_kg, height_cm):
    try:
        w, h = float(weight_kg), float(height_cm) / 100
    except (TypeError, ValueError):
        return None
    if not (20 <= w <= 300 and 1.2 <= h <= 2.2):
        return None
    return round(w / (h * h), 1)


def waist_ok(waist_cm):
    try:
        return 40 <= float(waist_cm) <= 200
    except (TypeError, ValueError):
        return False
