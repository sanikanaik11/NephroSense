import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from testfree_risk import score, egfr_ckd_epi_2021, FEATURES  # noqa: E402
from nhanes_ckd_model import fit_bundle  # noqa: E402
from home_measurements import home_bp_average, bmi  # noqa: E402
from triage import check_red_flags  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    """Synthetic data ONLY to exercise the code paths. Not a real model."""
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({
        "age": rng.uniform(20, 85, n), "female": rng.integers(0, 2, n).astype(float),
        "sbp": rng.normal(125, 18, n), "bmi": rng.normal(27, 5, n), "waist_cm": rng.normal(94, 13, n),
        "diabetes": rng.integers(0, 2, n).astype(float), "hypertension": rng.integers(0, 2, n).astype(float),
        "heart_disease": rng.integers(0, 2, n).astype(float), "kidney_stones": rng.integers(0, 2, n).astype(float)})
    lo = -6 + 0.05 * df.age + 0.02 * df.sbp + 1.0 * df.diabetes + 0.7 * df.hypertension
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-lo))).astype(int)
    return fit_bundle(df)


BASE = dict(age=45, sex="male", sbp=118, bmi=24, waist_cm=84,
            diabetes="no", hypertension="no", heart_disease="no", kidney_stones="no")


def test_egfr_known_value():
    assert 91 <= egfr_ckd_epi_2021(1.0, 50, False) <= 93


def test_refuses_when_missing(bundle):
    r = score({k: v for k, v in BASE.items() if k != "waist_cm"}, bundle)
    assert r["scored"] is False and "waist_cm" in r["missing"] and r["band"] is None


def test_refuses_out_of_range_and_does_not_clip(bundle):
    r = score({**BASE, "sbp": 400}, bundle)
    assert r["scored"] is False and r["invalid"]


def test_refuses_unknown_yes_no(bundle):
    r = score({**BASE, "diabetes": "not sure"}, bundle)
    assert r["scored"] is False and "diabetes" in r["missing"]


def test_higher_risk_profile_scores_higher(bundle):
    a = score(BASE, bundle)
    b = score({**BASE, "age": 70, "sbp": 160, "diabetes": "yes", "hypertension": "yes"}, bundle)
    assert b["raw_probability"] > a["raw_probability"]
    assert b["reasons"] and all("%" not in r["factor"] for r in b["reasons"])


def test_strip_only_raises(bundle):
    base = score(BASE, bundle)
    pos = score(BASE, bundle, strip_result="positive")
    neg = score(BASE, bundle, strip_result="negative")
    order = {"LOW": 0, "POSSIBLE": 1, "HIGH": 2}
    assert order[pos["band"]] >= order[base["band"]]
    assert neg["band"] == base["band"]


def test_bp_average_drops_day1_and_needs_enough():
    r = [{"day": d, "sys": 120 + d, "dia": 80} for d in range(1, 8) for _ in range(2)]
    out = home_bp_average(r)
    assert out["ok"] and out["n"] == 12 and out["sbp"] == 124.5   # days 2-7 -> mean(122..127)
    assert home_bp_average(r[:5])["ok"] is False
    assert home_bp_average([{"day": 1, "sys": 300, "dia": 80}] * 20)["ok"] is False


def test_bmi():
    assert bmi(70, 175) == 22.9
    assert bmi(70, 50) is None


def test_triage():
    t = check_red_flags({"swelling": "no", "little_urine": "yes", "breathless": "no", "vomiting": "no"})
    assert t["urgent"] and "NOW" in t["message"]
    t2 = check_red_flags({"swelling": "no"})
    assert not t2["urgent"] and len(t2["unanswered"]) == 3
