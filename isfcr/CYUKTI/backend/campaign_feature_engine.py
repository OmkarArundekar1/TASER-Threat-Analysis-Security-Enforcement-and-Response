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
    prediction_similarity: Optional[float]
    chain_similarity: Optional[float]
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
        if context.last_technique is None:
            print("[CHAIN] last_technique is None")
            return None
    
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
                print("\n===== CHAIN SIMILARITY =====")
                print("Previous          :", context.last_technique)
                print("Current           :", current_technique)
                print("Total Transitions :", 0)
                print("Transition Count  :", 0)
                print("============================")
                return 0.0
    
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
    
            print("\n===== CHAIN SIMILARITY =====")
            print("Previous          :", context.last_technique)
            print("Current           :", current_technique)
            print("Total Transitions :", total)
            print("Transition Count  :", transition_count)
            print("============================")
    
            return round(transition_count / total, 2)
    def temporal_similarity(
        self,
        context
    ):
        seconds = (
            datetime.now(timezone.utc)
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
        attacker_ip
    ):
        return CampaignFeatures(
            prediction_similarity=self.prediction_similarity(
                context,
                current_technique
            ),
            chain_similarity=self.chain_similarity(
                context,
                current_technique
            ),
            temporal_similarity=self.temporal_similarity(
                context
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