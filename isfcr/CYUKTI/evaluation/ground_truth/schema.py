"""
ground_truth/schema.py
========================
The ground-truth record CYUKTI's outputs are evaluated against.

ANTI-CIRCULARITY BOUNDARY (enforced, not just documented):
A GroundTruthRecord's `expected_*` fields must never be populated from
CYUKTI's own resolver/engine output. `GroundTruthRecord.validate()`
raises if a `system_prediction_marker` is present on any expected_*
field (see the SYSTEM_OUTPUT_MARKERS check below) -- this is a
last-resort safety net, not the primary control (the primary control
is simply: nothing in ground_truth/builder.py ever imports
mitre_resolver, campaign_manager, threat_attribution_engine,
threat_qualification, or prediction_engine to *produce* an expected_*
value).

Three review states, promoted only forward, never silently:
    AUTO_PROPOSED   -- produced by GroundTruthBuilder from independent
                        sources (scenario definition, known lab
                        topology, raw alert text) with NO human review.
    HUMAN_REVIEWED  -- a human has looked at the raw evidence and
                        confirmed or corrected the label.
    LOCKED          -- a HUMAN_REVIEWED dataset has been hashed and
                        frozen (dataset_locking.py) and may be cited as
                        the paper's ground truth.

Evaluators must refuse to treat AUTO_PROPOSED-only data as final
(see evaluators/base.py's `require_reviewed()`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ReviewStatus(str, Enum):
    AUTO_PROPOSED = "AUTO_PROPOSED"
    HUMAN_REVIEWED = "HUMAN_REVIEWED"
    LOCKED = "LOCKED"


# Any expected_* field literally equal to one of these sentinel values
# is rejected -- a crude but real guard against someone later wiring a
# CYUKTI call into the builder by mistake and forgetting to relabel it.
_SYSTEM_OUTPUT_MARKERS = {"__FROM_MITRE_RESOLVER__", "__FROM_CAMPAIGN_MANAGER__",
                          "__FROM_ATTRIBUTION_ENGINE__", "__FROM_THREAT_QUALIFICATION__",
                          "__FROM_PREDICTION_ENGINE__"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GroundTruthRecord:
    sample_id: str
    source: str                       # e.g. "wazuh_alert", "neo4j_campaign", "synthetic_generator_config"
    timestamp: str                    # when the underlying event/campaign actually occurred
    scenario_id: str                  # foreign key into scenarios/scenarios.json
    raw_event_id: str                 # e.g. a Wazuh rule id + agent + line offset, or a real Campaign.campaign_id (see note)

    attacker_identity: str | None = None
    victim_identity: str | None = None
    expected_attack: str | None = None                # one-line human description
    expected_mitre_techniques: list[str] = field(default_factory=list)
    expected_campaign_id: str | None = None            # an INDEPENDENT grouping key, e.g. f"{scenario_id}:{attacker}:{session_window}" -- never CYUKTI's own Campaign.campaign_id
    expected_threat_status: str | None = None          # NOT_THREAT | SUSPICIOUS | QUALIFIED_THREAT
    expected_attribution: str | None = None            # the real, known attacker identity (e.g. "KALI-01")

    reviewer: str = "unreviewed"
    review_status: ReviewStatus = ReviewStatus.AUTO_PROPOSED
    evidence_reference: str = ""                       # pointer to the raw doc/file/query that justifies this label
    labeling_method: str = ""                          # how the label was derived, e.g. "scenario_definition + rule_text_cross_reference (independent of mitre_resolver.py)"

    created_at: str = field(default_factory=_now_iso)
    dataset_version: str = "v0-unlocked"

    def validate(self) -> None:
        for f in (self.expected_attack, self.expected_campaign_id, self.expected_threat_status,
                  self.expected_attribution, *self.expected_mitre_techniques):
            if f in _SYSTEM_OUTPUT_MARKERS:
                raise ValueError(
                    f"GroundTruthRecord {self.sample_id}: an expected_* field is a system-output "
                    f"marker ({f!r}) -- ground truth must never be derived from CYUKTI's own output."
                )
        if self.review_status != ReviewStatus.AUTO_PROPOSED and self.reviewer in ("", "unreviewed"):
            raise ValueError(
                f"GroundTruthRecord {self.sample_id}: review_status={self.review_status.value} "
                f"but no reviewer is named -- cannot promote past AUTO_PROPOSED without an identified reviewer."
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["review_status"] = self.review_status.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "GroundTruthRecord":
        d = dict(d)
        d["review_status"] = ReviewStatus(d["review_status"])
        return GroundTruthRecord(**d)


def record_hash(record: GroundTruthRecord) -> str:
    """Stable content hash of a record's ground-truth fields (excludes
    created_at/reviewer/dataset_version so re-saving with a new
    timestamp doesn't change the hash of the underlying label)."""
    payload = {
        "sample_id": record.sample_id,
        "scenario_id": record.scenario_id,
        "raw_event_id": record.raw_event_id,
        "attacker_identity": record.attacker_identity,
        "victim_identity": record.victim_identity,
        "expected_attack": record.expected_attack,
        "expected_mitre_techniques": sorted(record.expected_mitre_techniques),
        "expected_campaign_id": record.expected_campaign_id,
        "expected_threat_status": record.expected_threat_status,
        "expected_attribution": record.expected_attribution,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
