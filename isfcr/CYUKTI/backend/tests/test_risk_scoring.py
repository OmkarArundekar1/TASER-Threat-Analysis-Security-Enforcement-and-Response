"""
Regression tests for risk_scoring.py, extracted from dashboard_api.py to
fix a real bug found while auditing the real 33-campaign dataset:
ml/label_generator.py was applying 0-100-scale severity thresholds
directly to the raw, unbounded cumulative risk_score (which reached
8330 on real data), so nearly every real campaign trivially read as
"Critical". These tests pin the corrected, shared behavior.
"""

from risk_scoring import normalize_risk_score, risk_level_from_score, severity_from_tps


def test_normalize_clamps_at_100():
    assert normalize_risk_score(999999) == 100


def test_normalize_zero_is_zero():
    assert normalize_risk_score(0) == 0
    assert normalize_risk_score(None) == 0


def test_normalize_scales_linearly_below_ceiling():
    # TPS_CEILING default is 1500
    assert normalize_risk_score(750) == 50


def test_risk_level_thresholds():
    assert risk_level_from_score(80) == "CRITICAL"
    assert risk_level_from_score(60) == "HIGH"
    assert risk_level_from_score(35) == "MEDIUM"
    assert risk_level_from_score(34.9) == "LOW"


def test_severity_from_tps_real_outlier_is_still_critical():
    # CAMP_427A075C, risk_score=8330 (real data) — clamps to 100, still Critical
    assert severity_from_tps(8330) == "CRITICAL"


def test_severity_from_tps_mid_range_real_campaign_is_not_trivially_critical():
    # CAMP_D8605E81, risk_score=630 (real data) — the pre-fix bug compared
    # this raw value directly against an 80-threshold and called it
    # "Critical"; correctly normalized (630/1500*100=42) it's Medium.
    assert severity_from_tps(630) == "MEDIUM"


def test_severity_from_tps_low_real_campaign_is_low():
    # CAMP_A93237FD, risk_score=90 (real data) — pre-fix this also
    # trivially cleared the raw 80-threshold as "Critical"; normalized
    # (90/1500*100=6) it's Low.
    assert severity_from_tps(90) == "LOW"
