"""
campaign_selection.py
========================
BestCampaignSelector: takes multiple candidate historical campaigns
(e.g. GNN topology top-K retrieval) and selects the single best match
internally while keeping every candidate and every per-signal score
inspectable -- see CAMPAIGN_SELECTION.md.

Critical rule enforced here (rule 6 of the roadmap): GNN topology
similarity is never the sole basis for selection. It is one of five
explicit, separately-reported signals combined by documented, fixed
weights (WEIGHTS below) -- not a learned or opaque blend. A candidate
missing a signal (e.g. GNN disabled) has that signal excluded from its
composite score, with the remaining weights renormalized to sum to 1,
rather than being penalized with a fabricated zero.

Historical playbook success is deliberately NOT folded into the
composite similarity score: "is this the same/similar campaign" and
"did our past response to it work" are different questions (the first
is about identity/structure, the second is about outcome quality) --
both are reported, but blending them would hide which one is driving a
given selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

WEIGHTS = {
    "topology": 0.30,
    "technique": 0.25,
    "temporal": 0.20,
    "attacker": 0.15,
    "host": 0.10,
}

# Confidence categorization is a fixed, documented function of the
# score gap between the top two candidates -- not a fabricated
# probability. A wide gap means the winner is clearly ahead; a narrow
# gap means the choice is close and the analyst should look at the
# alternatives.
CONFIDENCE_HIGH_GAP = 0.20
CONFIDENCE_MEDIUM_GAP = 0.08


@dataclass
class CandidateSignals:
    topology_similarity: float | None = None
    technique_similarity: float | None = None
    temporal_similarity: float | None = None
    attacker_similarity: float | None = None
    host_similarity: float | None = None

    def to_dict(self) -> dict:
        return {
            "topology_similarity": self.topology_similarity,
            "technique_similarity": self.technique_similarity,
            "temporal_similarity": self.temporal_similarity,
            "attacker_similarity": self.attacker_similarity,
            "host_similarity": self.host_similarity,
        }


@dataclass
class CampaignCandidate:
    campaign_id: str
    signals: CandidateSignals
    composite_score: float
    historical_playbook_success_rate: float | None = None
    historical_playbook_executions: int = 0

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "signals": self.signals.to_dict(),
            "composite_score": self.composite_score,
            "historical_playbook_success_rate": self.historical_playbook_success_rate,
            "historical_playbook_executions": self.historical_playbook_executions,
        }


@dataclass
class SelectionResult:
    ranked_candidates: list[CampaignCandidate]
    selected: CampaignCandidate | None
    alternatives: list[CampaignCandidate]
    confidence: str  # "HIGH" | "MEDIUM" | "LOW" | "NONE" (no candidates)
    score_gap: float | None
    explanation: str

    def to_dict(self) -> dict:
        return {
            "ranked_candidates": [c.to_dict() for c in self.ranked_candidates],
            "selected": self.selected.to_dict() if self.selected else None,
            "alternatives": [c.to_dict() for c in self.alternatives],
            "confidence": self.confidence,
            "score_gap": self.score_gap,
            "explanation": self.explanation,
        }


def _composite_score(signals: CandidateSignals) -> float:
    available = {
        name: value for name, value in signals.to_dict().items()
        if value is not None and name.replace("_similarity", "") in WEIGHTS
    }
    if not available:
        return 0.0
    weight_key = lambda name: name.replace("_similarity", "")
    total_weight = sum(WEIGHTS[weight_key(name)] for name in available)
    if total_weight == 0:
        return 0.0
    return sum(WEIGHTS[weight_key(name)] * value for name, value in available.items()) / total_weight


def _ip_similarity(current_ip: str | None, candidate_ip: str | None) -> float | None:
    """1.0 exact match, 0.5 same /24 subnet, 0.0 otherwise. A simple,
    explainable graded identity metric -- not a learned similarity."""
    if not current_ip or not candidate_ip:
        return None
    if current_ip == candidate_ip:
        return 1.0
    current_octets = current_ip.split(".")
    candidate_octets = candidate_ip.split(".")
    if len(current_octets) == 4 and len(candidate_octets) == 4 and current_octets[:3] == candidate_octets[:3]:
        return 0.5
    return 0.0


def _temporal_similarity(current_time: datetime | None, candidate_time: datetime | None) -> float | None:
    """Linear decay over a 1-week scale: same time -> 1.0, a week or
    more apart -> 0.0. A fixed, documented formula, not learned."""
    if current_time is None or candidate_time is None:
        return None
    delta_hours = abs((current_time - candidate_time).total_seconds()) / 3600.0
    return max(0.0, 1.0 - delta_hours / 168.0)


def _explain(selected: CampaignCandidate, others: list[CampaignCandidate]) -> str:
    signal_values = {k: v for k, v in selected.signals.to_dict().items() if v is not None}
    if not signal_values:
        return f"{selected.campaign_id} selected: no comparable signals were available for any candidate."

    ranked_signals = sorted(signal_values.items(), key=lambda kv: kv[1], reverse=True)
    strongest = ranked_signals[0]
    weakest = ranked_signals[-1]

    label = lambda name: name.replace("_similarity", "").replace("_", " ")

    if len(ranked_signals) == 1 or strongest[1] == weakest[1]:
        return (
            f"{selected.campaign_id} ranked highest with a composite score of "
            f"{selected.composite_score:.2f}, driven primarily by {label(strongest[0])} "
            f"similarity ({strongest[1]:.0%})."
        )

    return (
        f"{selected.campaign_id} ranked highest (composite score {selected.composite_score:.2f}) "
        f"because its {label(strongest[0])} similarity ({strongest[1]:.0%}) matched the current "
        f"campaign most closely, despite comparatively weaker {label(weakest[0])} similarity ({weakest[1]:.0%})."
    )


class BestCampaignSelector:
    def select(self, candidates: list[CampaignCandidate]) -> SelectionResult:
        if not candidates:
            return SelectionResult(
                ranked_candidates=[], selected=None, alternatives=[],
                confidence="NONE", score_gap=None,
                explanation="No candidate campaigns were available to select from.",
            )

        ranked = sorted(candidates, key=lambda c: c.composite_score, reverse=True)
        selected = ranked[0]
        alternatives = ranked[1:]

        if len(ranked) > 1:
            gap = ranked[0].composite_score - ranked[1].composite_score
        else:
            gap = None

        if gap is None:
            confidence = "HIGH"  # only one candidate existed at all
        elif gap >= CONFIDENCE_HIGH_GAP:
            confidence = "HIGH"
        elif gap >= CONFIDENCE_MEDIUM_GAP:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        explanation = _explain(selected, alternatives)

        return SelectionResult(
            ranked_candidates=ranked, selected=selected, alternatives=alternatives,
            confidence=confidence, score_gap=gap, explanation=explanation,
        )

    def build_candidate(
        self,
        campaign_id: str,
        topology_similarity: float | None = None,
        technique_similarity: float | None = None,
        temporal_similarity: float | None = None,
        attacker_similarity: float | None = None,
        host_similarity: float | None = None,
        historical_playbook_success_rate: float | None = None,
        historical_playbook_executions: int = 0,
    ) -> CampaignCandidate:
        signals = CandidateSignals(
            topology_similarity=topology_similarity, technique_similarity=technique_similarity,
            temporal_similarity=temporal_similarity, attacker_similarity=attacker_similarity,
            host_similarity=host_similarity,
        )
        return CampaignCandidate(
            campaign_id=campaign_id, signals=signals, composite_score=_composite_score(signals),
            historical_playbook_success_rate=historical_playbook_success_rate,
            historical_playbook_executions=historical_playbook_executions,
        )


selector = BestCampaignSelector()


def build_candidates_from_campaigns(
    current_campaign, candidate_campaign_ids: list[str], neo4j_session,
    memory_store=None,
) -> list[CampaignCandidate]:
    """Real end-to-end candidate construction: loads each candidate's
    real CampaignContext (reusing dashboard_api._load_campaign_context,
    the same function every other evidence-aware endpoint uses -- no
    duplicate representation), computes each signal from real data, and
    never fabricates a signal it can't compute (GNN disabled -> that
    candidate's topology_similarity stays None, not zero).

    `memory_store`: optional soar.memory.PlaybookMemoryStore, to attach
    historical_playbook_success_rate when a playbook exists for that
    candidate campaign. None (default) -- historical playbook context
    isn't every caller's concern.
    """
    from dashboard_api import _load_campaign_context
    from soar.matcher import _technique_jaccard

    candidates = []
    for candidate_id in candidate_campaign_ids:
        if candidate_id == current_campaign.campaign_id:
            continue
        try:
            candidate_context = _load_campaign_context(neo4j_session, candidate_id)
        except Exception:
            logger.exception("campaign_selection: failed loading candidate %s", candidate_id)
            candidate_context = None
        if candidate_context is None:
            continue

        technique_similarity = _technique_jaccard(
            current_campaign.techniques or set(), candidate_context.techniques or set()
        )
        attacker_similarity = _ip_similarity(current_campaign.attacker_ip, candidate_context.attacker_ip)
        host_similarity = _ip_similarity(current_campaign.victim_ip, candidate_context.victim_ip)
        temporal_similarity = _temporal_similarity(
            getattr(current_campaign, "first_seen", None), getattr(candidate_context, "first_seen", None)
        )

        topology_similarity = None
        try:
            from ml.gnn.topology_similarity import gnn_topology_similarity_between_campaigns
            topology_similarity = gnn_topology_similarity_between_campaigns(
                current_campaign.campaign_id, candidate_id
            )
        except Exception:
            logger.exception("campaign_selection: GNN topology similarity failed for %s", candidate_id)

        success_rate, executions = None, 0
        if memory_store is not None:
            for playbook in memory_store.list_playbooks():
                if playbook.source_campaign_id == candidate_id:
                    executions, successes, failures, success_rate = memory_store.historical_match_stats(
                        playbook.playbook_id
                    )
                    break

        candidates.append(selector.build_candidate(
            campaign_id=candidate_id,
            topology_similarity=topology_similarity,
            technique_similarity=technique_similarity,
            temporal_similarity=temporal_similarity,
            attacker_similarity=attacker_similarity,
            host_similarity=host_similarity,
            historical_playbook_success_rate=success_rate,
            historical_playbook_executions=executions,
        ))

    return candidates
