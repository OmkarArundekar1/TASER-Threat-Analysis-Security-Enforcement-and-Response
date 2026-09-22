from dataclasses import dataclass
from operation_manager import OperationManager
from operation_feature_engine import OperationFeatureEngine
from operation_decision_engine import OperationDecisionEngine
from neo4j_client import (
    get_active_operations,
    get_recent_inactive_operations,
    reopen_operation_db
)

@dataclass
class CorrelationResult:
    matched: bool
    operation_id: str | None
    confidence: float
    score: float
    breakdown: dict
    candidate_count: int
    candidates: list
    # Additive, informational only -- see GNN_PRODUCTION_INTEGRATION.md.
    # Max GNN topology-embedding cosine similarity between this campaign
    # and the campaigns already in `operation_id` (None if GNN is disabled,
    # unavailable, or there is no matched operation). NEVER participates in
    # `score`/`matched`/`decision` -- OperationDecisionEngine.weights still
    # has graph_similarity at 0.00 and is completely untouched by this
    # field; this is a separate, additive signal for downstream reasoning
    # to consult, not a silent redefinition of the existing decision.
    gnn_topology_similarity: float | None = None
    
class CampaignCorrelationEngine:

    def __init__(self):
        self.manager = OperationManager()
        self.feature_engine = OperationFeatureEngine()
        self.decision_engine = OperationDecisionEngine()

    def correlate(self, campaign_context):
        if campaign_context is None:
            return CorrelationResult(
                matched=False,
                operation_id=None,
                confidence=0,
                score=0,
                breakdown={},
                candidate_count=0,
                candidates=[]
            )

        operation_ids = get_active_operations()
        searching_inactive = False
        
        if not operation_ids:
            operation_ids = get_recent_inactive_operations()
            searching_inactive = True
        
        if not operation_ids:
            return CorrelationResult(
                matched=False,
                operation_id=None,
                confidence=0,
                score=0,
                breakdown={},
                candidate_count=0,
                candidates=[]
            )
        best_operation = None
        best_decision = None

        candidate_results = []
        for operation_id in operation_ids:
            operation_context = self.manager.build_operation_context(
                operation_id
            )

            if operation_context is None:
                continue

            features = self.feature_engine.extract_features(
                campaign_context,
                operation_context
            )
            print("\n===== CAMPAIGN CONTEXT =====")
            print(campaign_context.techniques)
            print(campaign_context.attack_chain)
            print(campaign_context.predicted_next)
            
            print("\n===== OPERATION CONTEXT =====")
            print(operation_context.techniques)
            print(operation_context.attack_chain)
            print(operation_context.prediction_profile)

            decision = self.decision_engine.evaluate(
                features
            )
            print("\n========== OPERATION FEATURE DEBUG ==========")
            print(f"Operation : {operation_context.operation_id}")
            print(f"Attacker  : {features.attacker_similarity}")
            print(f"Victim    : {features.victim_similarity}")
            print(f"Technique : {features.technique_similarity}")
            print(f"Temporal  : {features.temporal_similarity}")
            print(f"Chain     : {features.chain_similarity}")
            print(f"Prediction: {features.prediction_similarity}")
            print(f"Graph     : {features.graph_similarity}")
            print(f"Score     : {decision.score}")
            print("=============================================\n")
            candidate_results.append({
                "operation_id": operation_id,
                "score": decision.score,
                "decision": decision.decision
            })

            if (
                best_decision is None
                or
                decision.score > best_decision.score
            ):
                best_operation = operation_context
                best_decision = decision

        if best_decision is None:
            return CorrelationResult(
                matched=False,
                operation_id=None,
                confidence=0,
                score=0,
                breakdown={},
                candidate_count=len(operation_ids),
                candidates=[]
            )

        if (
            searching_inactive
            and
            best_decision.decision == "ATTACH_TO_OPERATION"
        ):
            print("\n[FAST OPERATION REOPEN]")
            print(best_operation.operation_id)
        
            reopen_operation_db(
                best_operation.operation_id
            )

        matched = best_decision.decision == "ATTACH_TO_OPERATION"

        return CorrelationResult(
            matched=matched,
            operation_id=(
                best_operation.operation_id
                if matched
                else None
            ),
            confidence=best_decision.confidence,
            score=best_decision.score,
            breakdown=best_decision.breakdown,
            candidate_count=len(operation_ids),
            candidates=candidate_results,
            gnn_topology_similarity=(
                self._gnn_topology_similarity_to_operation(campaign_context, best_operation)
                if matched else None
            ),
        )

    def _gnn_topology_similarity_to_operation(self, campaign_context, operation_context) -> float | None:
        """Additive, informational only (see CorrelationResult's own
        docstring note above). Max GNN topology similarity between the
        current campaign and any campaign already in this operation --
        fails safe to None for any reason (GNN disabled, no artifact,
        empty operation, malformed graphs), never raises, never affects
        the decision already made above."""
        try:
            from ml.gnn.topology_similarity import gnn_topology_similarity_between_campaigns

            campaign_ids = getattr(operation_context, "campaign_ids", None) or []
            similarities = [
                s for s in (
                    gnn_topology_similarity_between_campaigns(campaign_context.campaign_id, other_id)
                    for other_id in campaign_ids
                    if other_id != campaign_context.campaign_id
                )
                if s is not None
            ]
            return max(similarities) if similarities else None
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "GNN topology similarity computation failed for operation correlation "
                "(campaign %s) -- continuing without it.", campaign_context.campaign_id,
            )
            return None

engine = CampaignCorrelationEngine()