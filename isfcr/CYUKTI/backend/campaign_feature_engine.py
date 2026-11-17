from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from neo4j_client import (
    driver,
    get_prediction_distribution
)
from graph_feature_engine import graph_analytics
from runtime_graph_feature_engine import engine as runtime_engine
from config import CAMPAIGN_TIMEOUT

@dataclass
class CampaignFeatures:
    transition_similarity: Optional[float]  # Merged prediction + chain
    temporal_similarity: Optional[float]
    attacker_similarity: Optional[float]
    duplicate_similarity: Optional[float]
    graph_similarity: Optional[float]
    runtime_similarity: Optional[float]

class CampaignFeatureEngine:
    def prediction_similarity(
        self,
        context,
        current_technique
    ):
        if context.last_technique is None:
            return None

        distribution = get_prediction_distribution(
            context.last_technique
        )

        if not distribution:
            return None

        probability = distribution.get(
            current_technique,
            0.0
        )

        return round(probability, 2)

    def chain_similarity(self, context, current_technique):
        """
        Compute chain similarity with unseen transition handling.

        Critical fix: Instead of returning 0.0 for unseen transitions,
        return a small default value (0.1) to prevent campaign fragmentation.
        "Never seen this transition" means "no information", not "evidence against".
        """
        if context.last_technique is None:
            print("[CHAIN] last_technique is None")
            return None

        # Default probability for unseen transitions - key fix for campaign splitting
        UNSEEN_TRANSITION_DEFAULT = 0.1

        with driver.session() as session:
            learned = session.run(
                """
                MATCH (:Technique {attack_id:$previous})
                      -[r:NEXT_TECHNIQUE]->()
                RETURN count(r) AS total
                """,
                previous=context.last_technique
            ).single()

            total = learned["total"]

            if total == 0:
                # No transitions learned from this technique yet
                # Return default instead of 0 to avoid splitting
                print("\n===== CHAIN SIMILARITY =====")
                print("Previous          :", context.last_technique)
                print("Current           :", current_technique)
                print("Total Transitions :", 0)
                print("Transition Count  :", 0)
                print("Using default     :", UNSEEN_TRANSITION_DEFAULT)
                print("============================")
                return UNSEEN_TRANSITION_DEFAULT

            result = session.run(
                """
                MATCH (:Technique {attack_id:$previous})
                      -[r:NEXT_TECHNIQUE]->
                      (:Technique {attack_id:$current})
                RETURN coalesce(r.count, 0) AS transition_count
                """,
                previous=context.last_technique,
                current=current_technique
            ).single()

            transition_count = result["transition_count"] if result else 0

            # If this specific transition was never seen, use default
            if transition_count == 0:
                print("\n===== CHAIN SIMILARITY =====")
                print("Previous          :", context.last_technique)
                print("Current           :", current_technique)
                print("Total Transitions :", total)
                print("Transition Count  :", 0)
                print("Using default     :", UNSEEN_TRANSITION_DEFAULT)
                print("============================")
                return UNSEEN_TRANSITION_DEFAULT

            print("\n===== CHAIN SIMILARITY =====")
            print("Previous          :", context.last_technique)
            print("Current           :", current_technique)
            print("Total Transitions :", total)
            print("Transition Count  :", transition_count)
            print("============================")

            return round(transition_count / total, 2)
    def temporal_similarity(
        self,
        context,
        reference_time=None
    ):
        """reference_time: event time of the alert being evaluated
        (event_time.py). Formula and thresholds are unchanged; only the
        reference moved from processing time to event time. None keeps the
        legacy processing-time behaviour for direct callers. A late alert
        (reference_time < last_seen) gives a negative gap, which falls in
        the "within timeout" band (1.0)."""
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        seconds = (
            reference_time
            -
            context.last_seen
        ).total_seconds()
        timeout = CAMPAIGN_TIMEOUT
        if seconds <= timeout:
            return 1.0
        similarity = max(
            0.2,
            1.0 - ((seconds - timeout) / (timeout * 4))
        )

        return round(similarity, 2)

    def attacker_similarity(
        self,
        context,
        attacker_ip
    ):
        if context.attacker_ip == attacker_ip:
            return 1.0

        return 0.0

    def duplicate_similarity(
        self,
        context
    ):
        runtime = runtime_engine.extract_features(
            context.campaign_id
        )
        return min(
            runtime.duplicate_frequency,
            1.0
        )

    def graph_similarity(
        self,
        context
    ):
        graph_analytics.refresh_campaign(
            context.campaign_id
        )
        graph = graph_analytics.extract_features(
            context.campaign_id
        )
        score = (
            graph.graph_density
            +
            min(
                graph.attack_chain_depth / 10,
                1.0
            )
            +
            min(
                graph.campaign_complexity / 100,
                1.0
            )
        ) / 3
        return round(score, 2)

    def runtime_similarity(
        self,
        context
    ):
        runtime = runtime_engine.extract_features(
            context.campaign_id
        )
        score = min(
            (
                runtime.attack_event_count
                +
                runtime.prediction_frequency
            ) / 2,
            1.0
        )
        return score

    def extract(
        self,
        context,
        current_technique,
        attacker_ip,
        reference_time=None
    ):
        """
        Extract features for campaign correlation.

        Critical fix: Merge prediction_similarity and chain_similarity into
        a single transition_similarity feature. Both measure "what comes next"
        from different angles (ML prediction vs historical transitions), so
        combining them prevents one missing pattern from dominating.
        """
        pred_sim = self.prediction_similarity(context, current_technique)
        chain_sim = self.chain_similarity(context, current_technique)

        # Merge prediction and chain into one transition feature
        # Use average when both available, or whichever is available
        if pred_sim is not None and chain_sim is not None:
            transition_sim = round((pred_sim + chain_sim) / 2, 2)
        elif pred_sim is not None:
            transition_sim = pred_sim
        elif chain_sim is not None:
            transition_sim = chain_sim
        else:
            transition_sim = None

        return CampaignFeatures(
            transition_similarity=transition_sim,
            temporal_similarity=self.temporal_similarity(
                context,
                reference_time
            ),
            attacker_similarity=self.attacker_similarity(
                context,
                attacker_ip
            ),
            duplicate_similarity=self.duplicate_similarity(
                context
            ),
            graph_similarity=self.graph_similarity(
                context
            ),
            runtime_similarity=self.runtime_similarity(
                context
            )
        )
engine = CampaignFeatureEngine()