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

        return CorrelationResult(
            matched=(
                best_decision.decision
                == "ATTACH_TO_OPERATION"
            ),
            operation_id=(
                best_operation.operation_id
                if best_decision.decision == "ATTACH_TO_OPERATION"
                else None
            ),
            confidence=best_decision.confidence,
            score=best_decision.score,
            breakdown=best_decision.breakdown,
            candidate_count=len(operation_ids),
            candidates=candidate_results
        )

engine = CampaignCorrelationEngine()