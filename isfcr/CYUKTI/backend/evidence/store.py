"""
evidence/store.py
====================
In-memory collection of Evidence for one investigation (one campaign or
operation). Responsible for dedup, lookup, and the two aggregate signals
the investigation loop (backend/investigation) needs:

    - weighted_confidence(): a relevance-weighted average of how much we
      trust the evidence gathered so far. This is NOT a substitute for a
      calibrated model probability — see backend/investigation/confidence.py
      for how this combines with model-based confidence.

    - detect_conflicts(): a narrow, explainable heuristic — evidence from
      two DIFFERENT sources about the same entity whose confidences
      diverge by more than CONFLICT_THRESHOLD. This deliberately does not
      claim to detect semantic contradictions (e.g. two CTI feeds
      disagreeing about an IP's reputation in incompatible ways); it only
      flags "these sources disagree about how much to trust this," which
      is the signal the investigation loop can act on (gather more
      evidence, prefer the higher-confidence source, or lower overall
      confidence).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from evidence.schema import Evidence

CONFLICT_THRESHOLD = 0.4


@dataclass
class EvidenceConflict:
    entity_id: str
    evidence_a: Evidence
    evidence_b: Evidence
    confidence_gap: float


class EvidenceStore:
    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}

    def add(self, evidence: Evidence) -> bool:
        """Add one evidence item. Returns False if it was already present (dedup)."""
        if evidence.evidence_id in self._items:
            return False
        self._items[evidence.evidence_id] = evidence
        return True

    def add_many(self, evidence_list) -> int:
        return sum(1 for e in evidence_list if self.add(e))

    def all(self) -> list[Evidence]:
        return list(self._items.values())

    def by_source(self, source) -> list[Evidence]:
        source = source.value if hasattr(source, "value") else source
        return [e for e in self._items.values() if e.source.value == source]

    def by_type(self, type_) -> list[Evidence]:
        type_ = type_.value if hasattr(type_, "value") else type_
        return [e for e in self._items.values() if e.type.value == type_]

    def __len__(self) -> int:
        return len(self._items)

    def weighted_confidence(self) -> float:
        """Relevance-weighted mean confidence across all evidence.

        Evidence with relevance == 0.0 contributes NOTHING to this
        average — it is excluded, not given full weight. An earlier
        version fell back to unweighted (full-weight) contribution here
        "so a freshly-collected store doesn't read as zero confidence",
        but that was backwards: it let a single authoritative-but-
        irrelevant fact (e.g. MITRE reference text about a technique,
        confidence=1.0, relevance=0.0 because nothing has scored it
        against the actual investigation yet) trivially satisfy the
        confidence threshold and stop the investigation after one step,
        confirmed by running the real investigation loop against real
        campaigns — every run stopped after exactly one MITRE lookup. A
        store with only irrelevant evidence SHOULD read as low
        confidence, because that correctly signals "keep investigating."
        """
        items = self.all()
        if not items:
            return 0.0

        weights = [e.relevance for e in items]
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(e.confidence * w for e, w in zip(items, weights))
        return round(weighted_sum / total_weight, 4)

    def detect_conflicts(self) -> list[EvidenceConflict]:
        by_entity: dict[str, list[Evidence]] = defaultdict(list)
        for e in self._items.values():
            for rel in (e.relationships or [e.source_id]):
                by_entity[rel].append(e)

        conflicts = []
        for entity_id, group in by_entity.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if a.source == b.source:
                        continue
                    gap = abs(a.confidence - b.confidence)
                    if gap >= CONFLICT_THRESHOLD:
                        conflicts.append(EvidenceConflict(entity_id, a, b, round(gap, 4)))

        return conflicts
