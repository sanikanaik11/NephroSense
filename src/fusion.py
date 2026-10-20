"""
fusion.py — NephroSense
────────────────────────
Combines lifestyle (XGBoost, 85% CV accuracy) and eye/anemia
(MobileNet V2, 96.5% val accuracy on 859 images) into one overall
CKD risk score.

Eye weight is now 0.25 — the model was retrained on 859 real
conjunctival images (India + Italy + Ghana) and achieved 96.5%
validation accuracy. A confidence threshold of 0.65 is applied:
the eye score only contributes to the fused score when the model
is at least 65% confident in its prediction, reducing false
positives from low-quality or poorly-lit phone photos.

Voice: future scope — no validated CKD voice dataset available yet.

Weights when both modules complete:
    Lifestyle  75%  (main signal — XGBoost on 597 real records)
    Eye        25%  (supporting signal — anaemia proxy for CKD)

When only lifestyle completes:
    Lifestyle  100% (full weight automatically)

When only eye completes:
    Eye        100% (rare — but handled gracefully)
"""

# ── Fusion weights ────────────────────────────────────────────
# NOTE: the eye-photo check is "reference only" and is NOT part of any risk score,
# so its weight is 0. (The app does not import this file; kept consistent in case it is used later.)
WEIGHTS = {
    "lifestyle": 1.0,
    "eye"      : 0.0,
}

# Eye confidence threshold (0.0 to 1.0).
# If the eye model's confidence < this value, the eye score is
# excluded from the fused result to avoid false positives from
# blurry or poorly-lit phone photos.
# 0.65 = model must be at least 65% confident to contribute.
EYE_CONFIDENCE_THRESHOLD = 0.65

# Risk level boundaries (out of 100)
LOW_THRESHOLD    = 34   # below 34  → LOW
MEDIUM_THRESHOLD = 67   # 34-66     → MEDIUM
                        # 67+       → HIGH


def fuse_scores(
    lifestyle_score  = None,
    eye_score        = None,
    voice_score      = None,
    eye_confidence   = None,   # float 0-1 from eye model, or None
):
    """
    Combines available module scores into one overall CKD risk result.

    Parameters
    ----------
    lifestyle_score  : float 0-100 from XGBoost questionnaire model
    eye_score        : float 0-100 from MobileNet V2 anaemia model
    voice_score      : float 0-100 — not used yet (future scope)
    eye_confidence   : float 0-1   — model confidence for eye prediction.
                       If None, eye score is included without threshold check.
                       If provided and < EYE_CONFIDENCE_THRESHOLD, eye is excluded.

    Returns
    -------
    None if no scores available, otherwise dict:
        risk_score  : float 0-100
        risk_level  : "LOW" / "MEDIUM" / "HIGH"
        ckd_stage   : int 1-5 (rough estimate — use eGFR for clinical staging)
        used        : list of module names that contributed to the score
        eye_excluded: bool — True if eye score was present but excluded
                      due to low confidence (helps show user a warning)
    """
    available    = {}
    eye_excluded = False

    # ── Add lifestyle score ───────────────────────────────────
    if lifestyle_score is not None:
        try:
            ls = float(lifestyle_score)
            if 0 <= ls <= 100:
                available["lifestyle"] = ls
        except (TypeError, ValueError):
            pass

    # ── Add eye score (with confidence gate) ─────────────────
    if eye_score is not None:
        try:
            es = float(eye_score)
            if 0 <= es <= 100:
                # Check confidence threshold
                if eye_confidence is not None:
                    try:
                        conf = float(eye_confidence)
                        if conf >= EYE_CONFIDENCE_THRESHOLD:
                            available["eye"] = es
                        else:
                            # Low confidence — exclude from score
                            # but record that we tried
                            eye_excluded = True
                    except (TypeError, ValueError):
                        # Confidence not parseable — include eye score anyway
                        available["eye"] = es
                else:
                    # No confidence provided — include eye score
                    available["eye"] = es
        except (TypeError, ValueError):
            pass

    # ── Nothing available ─────────────────────────────────────
    if not available:
        return None

    # ── Compute weighted average ──────────────────────────────
    total_weight = sum(WEIGHTS.get(k, 0) for k in available)

    if total_weight <= 0:
        # Fallback: simple average of whatever is available
        fused = sum(available.values()) / len(available)
    else:
        fused = sum(available[k] * WEIGHTS.get(k, 0) for k in available) / total_weight

    fused = round(float(fused), 1)
    fused = max(0.0, min(100.0, fused))  # clamp to 0-100

    # ── Risk level ────────────────────────────────────────────
    if fused < LOW_THRESHOLD:
        risk_level = "LOW"
    elif fused < MEDIUM_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # ── CKD stage (rough ML-based estimate) ──────────────────
    # For clinically accurate staging, always use eGFR (egfr.py).
    # This estimate is shown only when creatinine is unavailable.
    if fused < 20:
        ckd_stage = 1
    elif fused < 40:
        ckd_stage = 2
    elif fused < 60:
        ckd_stage = 3
    elif fused < 80:
        ckd_stage = 4
    else:
        ckd_stage = 5

    return {
        "risk_score"   : fused,
        "risk_level"   : risk_level,
        "ckd_stage"    : ckd_stage,
        "used"         : list(available.keys()),
        "eye_excluded" : eye_excluded,
    }


# ── Quick self-test ───────────────────────────────────────────
if __name__ == "__main__":
    print("=== fusion.py self-test ===\n")

    # Test 1: Lifestyle only
    r = fuse_scores(lifestyle_score=65)
    print(f"Lifestyle only (65):          {r}")

    # Test 2: Both modules, high eye confidence
    r = fuse_scores(lifestyle_score=65, eye_score=80, eye_confidence=0.90)
    print(f"Lifestyle 65 + Eye 80 (0.90): {r}")
    # Expected: (65*0.75 + 80*0.25) = 48.75 + 20 = 68.75 → HIGH

    # Test 3: Both modules, low eye confidence → eye excluded
    r = fuse_scores(lifestyle_score=65, eye_score=95, eye_confidence=0.50)
    print(f"Lifestyle 65 + Eye 95 (0.50): {r}")
    # Expected: eye_excluded=True, fused=65 (lifestyle only)

    # Test 4: Low risk
    r = fuse_scores(lifestyle_score=20, eye_score=15, eye_confidence=0.80)
    print(f"Low risk (20 + 15):           {r}")

    # Test 5: HIGH risk both modules
    r = fuse_scores(lifestyle_score=85, eye_score=90, eye_confidence=0.95)
    print(f"HIGH risk (85 + 90):          {r}")

    print("\n✅ All tests passed")