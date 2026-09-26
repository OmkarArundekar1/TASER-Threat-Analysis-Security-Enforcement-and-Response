"""
evaluators/base.py
=====================
Shared result type and the AUTO_PROPOSED-rejection rule every evaluator
must apply before reporting a metric as final.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from ground_truth.schema import ReviewStatus


class MetricStatus(str, Enum):
    MEASURED = "MEASURED"                              # ran on HUMAN_REVIEWED or LOCKED ground truth
    MEASURED_PRELIMINARY = "MEASURED_PRELIMINARY"        # ran on AUTO_PROPOSED only -- real number, not yet final
    NOT_MEASURED = "NOT_MEASURED"                        # evaluator exists, tested, but no dataset run yet
    UNMEASURABLE = "UNMEASURABLE"                        # ran, but zero valid opportunities existed
    BLOCKED_BY_ENVIRONMENT = "BLOCKED_BY_ENVIRONMENT"    # a stated infra/human-availability reason
    GROUND_TRUTH_REVIEW_REQUIRED = "GROUND_TRUTH_REVIEW_REQUIRED"


@dataclass
class EvaluationResult:
    task: str
    status: MetricStatus
    n: int = 0
    metrics: dict = field(default_factory=dict)
    ground_truth_review_status: str = ""
    dataset_name: str = ""
    method: str = ""
    limitations: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class GroundTruthReviewRequired(Exception):
    """Raised by strict-mode evaluators when only AUTO_PROPOSED ground
    truth is available and the caller asked for a final (non-
    preliminary) result."""


def classify_review_status(status: ReviewStatus, allow_preliminary: bool) -> MetricStatus:
    if status in (ReviewStatus.HUMAN_REVIEWED, ReviewStatus.LOCKED):
        return MetricStatus.MEASURED
    if allow_preliminary:
        return MetricStatus.MEASURED_PRELIMINARY
    raise GroundTruthReviewRequired(
        f"Only AUTO_PROPOSED ground truth is available. Per the evaluation framework's "
        f"anti-fabrication rule, this cannot be reported as a final MEASURED result. "
        f"Call with allow_preliminary=True to get a clearly-labeled MEASURED_PRELIMINARY "
        f"number, or run evaluation/review/export_review_queue.py and get it human-reviewed first."
    )
