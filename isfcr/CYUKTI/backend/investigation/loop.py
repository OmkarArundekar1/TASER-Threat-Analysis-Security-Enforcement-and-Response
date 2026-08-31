"""
investigation/loop.py
========================
The evidence-aware investigation loop:

    estimate confidence/uncertainty (evidence gathered + model verdict, if any)
        -> stop? -> yes: return investigation record
                 -> no: select next-best-evidence action (uncertainty-aware)
                        -> execute it:
                             - a fact-gathering action -> real engine call -> Evidence
                             - InvestigationAction.XGBOOST_PREDICTION -> real
                               model_predictor() -> class-probability dict,
                               stored as this investigation's model verdict
                        -> repeat

Two important, deliberate limitations, stated rather than hidden:
    1. The model verdict is computed AT MOST ONCE per investigation (when
       XGBOOST_PREDICTION is selected) — ml.runtime_predictor recomputes
       features fresh from Neo4j each call, and gathering more Evidence
       objects does not currently feed back into that feature vector.
       Re-running it mid-investigation would return the same answer, so
       the loop doesn't pretend otherwise by calling it repeatedly.
    2. Every fact-gathering action requires either live Neo4j (MITRE/CTI/
       detection/attribution/graph engines) or a trained model. Both are
       injected (`action_executor`, `model_predictor`) rather than
       hardcoded, so tests can exercise the loop's decisions without
       live infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from evidence.schema import Evidence
from evidence.store import EvidenceStore
from investigation.actions import ACTION_METADATA, InvestigationAction
from investigation.confidence import ConfidenceEstimate, estimate_confidence
from investigation.next_best_evidence import ActionValue, select_next_best_evidence
from investigation.stopping import StoppingDecision, check_stopping

ActionExecutor = Callable[[InvestigationAction], list[Evidence]]
ModelPredictor = Callable[[], "dict[str, float] | None"]


@dataclass
class InvestigationState:
    """A reconstructable snapshot of the investigation at one point in time."""

    step_number: int
    conclusion: str | None            # argmax(model_probabilities), if a model verdict exists
    model_probabilities: dict[str, float] | None
    model_confidence: float | None
    uncertainty: float
    evidence_reliability: float
    evidence_coverage: float
    candidate_evidence: list[InvestigationAction]  # actions still available to select from
    selected_evidence: InvestigationAction | None  # action chosen this step (None for the initial state)
    investigation_cost: float         # cumulative cost of actions taken so far


@dataclass
class InvestigationStep:
    step_index: int
    action_taken: InvestigationAction
    action_value: ActionValue
    evidence_added: int
    confidence_after: ConfidenceEstimate
    state_after: InvestigationState


@dataclass
class InvestigationRecord:
    steps: list[InvestigationStep] = field(default_factory=list)
    final_confidence: ConfidenceEstimate | None = None
    stopping_reason: str = ""
    evidence_store: EvidenceStore = field(default_factory=EvidenceStore)

    def to_dict(self) -> dict:
        return {
            "steps": [
                {
                    "step_index": s.step_index,
                    "action_taken": s.action_taken.value,
                    "action_value": s.action_value.value,
                    "evidence_added": s.evidence_added,
                    "model_probabilities": s.confidence_after.model_probabilities,
                    "model_confidence": s.confidence_after.model_confidence,
                    "model_uncertainty": s.confidence_after.model_uncertainty,
                    "evidence_reliability": s.confidence_after.evidence_reliability,
                    "evidence_coverage": s.confidence_after.evidence_coverage,
                    "investigation_confidence": s.confidence_after.investigation_confidence,
                    "uncertainty": s.confidence_after.uncertainty,
                }
                for s in self.steps
            ],
            "final_confidence": (
                self.final_confidence.investigation_confidence if self.final_confidence else None
            ),
            "final_model_probabilities": self.final_confidence.model_probabilities if self.final_confidence else None,
            "stopping_reason": self.stopping_reason,
            "total_evidence": len(self.evidence_store),
        }


def _cumulative_cost(taken_actions: list[InvestigationAction]) -> float:
    return round(sum(ACTION_METADATA[a].cost for a in taken_actions), 4)


def run_investigation(
    action_executor: ActionExecutor,
    model_predictor: ModelPredictor | None = None,
    initial_evidence: list[Evidence] | None = None,
    confidence_threshold: float = 0.75,
    max_uncertainty: float = 0.4,
    max_steps: int = 8,
) -> InvestigationRecord:
    store = EvidenceStore()
    if initial_evidence:
        store.add_many(initial_evidence)

    record = InvestigationRecord(evidence_store=store)
    remaining_actions = list(ACTION_METADATA.keys())
    taken_actions: list[InvestigationAction] = []
    model_probabilities: dict[str, float] | None = None
    step_index = 0

    while True:
        confidence = estimate_confidence(store, model_probabilities)

        best_action_value = select_next_best_evidence(store, remaining_actions, confidence.uncertainty)
        stop = check_stopping(
            confidence, step_index, best_action_value.value if best_action_value else None,
            confidence_threshold=confidence_threshold, max_uncertainty=max_uncertainty, max_steps=max_steps,
        )

        if stop.should_stop:
            record.final_confidence = confidence
            record.stopping_reason = stop.reason
            return record

        action = best_action_value.action

        if action == InvestigationAction.XGBOOST_PREDICTION:
            model_probabilities = model_predictor() if model_predictor else None
            new_evidence: list[Evidence] = []
        else:
            new_evidence = action_executor(action)

        added = store.add_many(new_evidence)
        taken_actions.append(action)
        step_index += 1

        confidence_after = estimate_confidence(store, model_probabilities)
        remaining_actions = [a for a in remaining_actions if a != action]

        record.steps.append(InvestigationStep(
            step_index=step_index,
            action_taken=action,
            action_value=best_action_value,
            evidence_added=added,
            confidence_after=confidence_after,
            state_after=InvestigationState(
                step_number=step_index,
                conclusion=(
                    max(model_probabilities, key=model_probabilities.get) if model_probabilities else None
                ),
                model_probabilities=confidence_after.model_probabilities,
                model_confidence=confidence_after.model_confidence,
                uncertainty=confidence_after.uncertainty,
                evidence_reliability=confidence_after.evidence_reliability,
                evidence_coverage=confidence_after.evidence_coverage,
                candidate_evidence=list(remaining_actions),
                selected_evidence=action,
                investigation_cost=_cumulative_cost(taken_actions),
            ),
        ))


def default_action_executor(campaign_context, current_attack_id: str, event_id: str | None) -> ActionExecutor:
    """Wires fact-gathering investigation actions to CYUKTI's real engines."""

    def execute(action: InvestigationAction) -> list[Evidence]:
        if action == InvestigationAction.MITRE_KNOWLEDGE:
            import mitre_feature_engine
            from evidence.collectors.mitre_collector import collect_mitre_evidence
            return collect_mitre_evidence(mitre_feature_engine.engine.extract_features(current_attack_id))

        if action == InvestigationAction.MITRE_SEMANTIC_SEARCH:
            from rag.mitre_retriever import mitre_retriever
            query_text = " ".join(campaign_context.techniques) or current_attack_id
            return mitre_retriever.query(query_text)

        if action == InvestigationAction.CTI_LOOKUP:
            import threat_intelligence_engine
            from evidence.collectors.cti_collector import collect_cti_evidence
            result = threat_intelligence_engine.engine.calculate(campaign_context.attacker_ip)
            return collect_cti_evidence(campaign_context.attacker_ip, result)

        if action == InvestigationAction.DETECTION_CHECK:
            if not event_id:
                return []
            import detection_confidence_engine
            from evidence.collectors.detection_collector import collect_detection_evidence
            result = detection_confidence_engine.engine.calculate(event_id)
            return collect_detection_evidence(event_id, result)

        if action == InvestigationAction.ATTRIBUTION_MATCH:
            import threat_attribution_engine
            from evidence.collectors.attribution_collector import collect_attribution_evidence
            result = threat_attribution_engine.engine.attribute(campaign_context)
            return collect_attribution_evidence(result)

        if action == InvestigationAction.CAMPAIGN_HISTORY:
            import attribution_context as attribution_context_module
            from evidence.collectors.campaign_history_collector import collect_campaign_history_evidence
            campaigns = attribution_context_module.context.load_historical_campaigns()
            return collect_campaign_history_evidence(campaign_context.techniques, campaigns)

        if action == InvestigationAction.GRAPH_STRUCTURE:
            import graph_feature_engine
            from evidence.collectors.graph_collector import collect_graph_evidence
            features = graph_feature_engine.graph_analytics.extract_features(
                campaign_context.campaign_id, force_reload=True
            )
            return collect_graph_evidence(campaign_context.campaign_id, features)

        if action == InvestigationAction.XGBOOST_PREDICTION:
            # Handled directly in run_investigation via model_predictor —
            # this action never reaches the fact-gathering executor.
            return []

        return []

    return execute


def default_model_predictor(campaign_context, current_attack_id: str, event_id: str) -> ModelPredictor | None:
    """Real XGBoost severity predictor for production use.

    Returns None (controlled fallback, not an exception) if no trained
    model exists at the expected path — estimate_confidence() already
    handles model_probabilities=None correctly by falling back to pure
    evidence-based confidence.
    """
    import os

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "models", "xgb_severity.json")
    model_path = os.path.normpath(model_path)

    if not os.path.exists(model_path):
        return None

    def predict() -> dict[str, float] | None:
        try:
            from ml.runtime_predictor import RuntimeCampaignPredictor
            predictor = RuntimeCampaignPredictor.from_model_path(model_path)
            result = predictor.predict_for_campaign(campaign_context, current_attack_id, event_id or "")
            return result["probabilities"]
        except Exception:
            return None

    return predict
