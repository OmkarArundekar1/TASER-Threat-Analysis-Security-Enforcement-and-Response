"""
risk_scoring.py
==================
Single source of truth for turning a raw cumulative TPS/risk_score
(unbounded — it's a running sum of per-alert TPS contributions, see
neo4j_client.create_attack_event) into a normalized 0-100 score and a
Critical/High/Medium/Low(/CRITICAL/HIGH/MEDIUM/LOW) label.

Previously this existed as a local, correct implementation inside
dashboard_api.py, and a SEPARATE, incorrect implementation inside
ml/label_generator.py that applied the same 0-100-scale thresholds
directly to the raw, unnormalized risk_score — which routinely exceeds
1000+ for sustained attacks (see docs/findings on CAMP_427A075C,
risk_score=8330), making nearly every real campaign read as "Critical"
regardless of actual severity. Consolidated here so there is exactly
one definition of what these thresholds mean.

TPS_CEILING is a genuine open calibration question, not just a
constant: its default (1500) was set against an earlier, smaller
dataset ("suitable for datasets with max TPS ~1285" per the original
comment). The real lab dataset now audited has a max of 8330 — nearly
6x that ceiling. Changing TPS_CEILING reshapes the entire severity
label distribution, which is a research-methodology decision, not a
routine engineering one — it is deliberately left as the historical
default here rather than silently recalibrated.
"""

from __future__ import annotations

import os

TPS_CEILING = int(os.environ.get("TPS_CEILING", "1500"))


def normalize_risk_score(raw_tps: float | int | None) -> float:
    raw = raw_tps or 0
    return min(100, round((raw / max(TPS_CEILING, 1)) * 100))


def risk_level_from_score(score: float) -> str:
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 35:
        return "MEDIUM"
    return "LOW"


def severity_from_tps(tps: float | int | None) -> str:
    return risk_level_from_score(normalize_risk_score(tps))
