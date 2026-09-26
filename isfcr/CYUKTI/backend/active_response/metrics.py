"""
active_response/metrics.py
=============================
Response-lifecycle latency and rate metrics, computed only from real
recorded timestamps/outcomes -- every function returns an explicit
NOT_MEASURED-shaped result (not a fabricated 0.0 or None-as-zero) when
there isn't enough real data yet. Percentiles reuse a small pure
function (no new dependency, consistent with evaluation_metrics.py's
existing house style); rate metrics reuse
evaluation_metrics.wilson_confidence_interval for a defensible interval
at small n.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


@dataclass
class LatencyStats:
    n: int
    p50_seconds: float | None
    p95_seconds: float | None
    p99_seconds: float | None
    mean_seconds: float | None
    status: str  # "MEASURED" | "NOT_MEASURED"
    reason: str = ""

    def to_dict(self) -> dict:
        return {"n": self.n, "p50_seconds": self.p50_seconds, "p95_seconds": self.p95_seconds,
                "p99_seconds": self.p99_seconds, "mean_seconds": self.mean_seconds,
                "status": self.status, "reason": self.reason}


def latency_stats(start_end_pairs: list[tuple[str, str]], min_n: int = 1) -> LatencyStats:
    """start_end_pairs: list of (start_iso, end_iso) real timestamp pairs."""
    if len(start_end_pairs) < min_n:
        return LatencyStats(len(start_end_pairs), None, None, None, None, "NOT_MEASURED",
                             f"Only {len(start_end_pairs)} observation(s); need >= {min_n}.")
    deltas = sorted((_parse(e) - _parse(s)).total_seconds() for s, e in start_end_pairs)
    return LatencyStats(
        n=len(deltas),
        p50_seconds=_percentile(deltas, 0.50),
        p95_seconds=_percentile(deltas, 0.95),
        p99_seconds=_percentile(deltas, 0.99) if len(deltas) >= 3 else None,
        mean_seconds=sum(deltas) / len(deltas),
        status="MEASURED",
    )


@dataclass
class RateStats:
    n: int
    successes: int
    rate: float | None
    confidence_interval_95: dict | None
    status: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {"n": self.n, "successes": self.successes, "rate": self.rate,
                "confidence_interval_95": self.confidence_interval_95,
                "status": self.status, "reason": self.reason}


def rate_stats(outcomes: list[bool]) -> RateStats:
    if not outcomes:
        return RateStats(0, 0, None, None, "NOT_MEASURED", "No observations recorded yet.")
    from evaluation_metrics import wilson_confidence_interval
    successes = sum(1 for o in outcomes if o)
    ci = wilson_confidence_interval(successes, len(outcomes))
    return RateStats(len(outcomes), successes, successes / len(outcomes), ci, "MEASURED")


@dataclass
class ResponseLifecycleTimestamps:
    correlation_id: str
    detected_at: str | None = None
    investigation_completed_at: str | None = None
    decided_at: str | None = None
    requested_at: str | None = None
    executed_at: str | None = None
    verified_at: str | None = None


def compute_lifecycle_latencies(records: list[ResponseLifecycleTimestamps]) -> dict:
    def pairs(attr_a: str, attr_b: str) -> list[tuple[str, str]]:
        return [
            (getattr(r, attr_a), getattr(r, attr_b))
            for r in records
            if getattr(r, attr_a) is not None and getattr(r, attr_b) is not None
        ]

    return {
        "detection_to_decision": latency_stats(pairs("detected_at", "decided_at")).to_dict(),
        "decision_to_containment_request": latency_stats(pairs("decided_at", "requested_at")).to_dict(),
        "containment_execution": latency_stats(pairs("requested_at", "executed_at")).to_dict(),
        "containment_verification": latency_stats(pairs("executed_at", "verified_at")).to_dict(),
        "total_detection_to_verified": latency_stats(pairs("detected_at", "verified_at")).to_dict(),
    }
