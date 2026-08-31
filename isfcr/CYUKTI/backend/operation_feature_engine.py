from dataclasses import dataclass
from datetime import timezone
from datetime_utils import normalize_datetime
from neo4j_client import get_prediction_distribution

@dataclass
class OperationFeatures:
    attacker_similarity: float = 0.0
    victim_similarity: float = 0.0
    temporal_similarity: float = 0.0
    technique_similarity: float = 0.0
    chain_similarity: float = 0.0
    graph_similarity: float = 0.0
    prediction_similarity: float = 0.0

class OperationFeatureEngine:
    def extract_features(
        self,
        campaign_context,
        operation_context
    ):
        features = OperationFeatures()
        features.attacker_similarity = (
            self.extract_attacker_similarity(
                campaign_context,
                operation_context
            )
        )
        features.victim_similarity = (
            self.extract_victim_similarity(
                campaign_context,
                operation_context
            )
        )
        features.temporal_similarity = (
            self.extract_temporal_similarity(
                campaign_context,
                operation_context
            )
        )
        features.technique_similarity = (
            self.extract_technique_similarity(
                campaign_context,
                operation_context
            )
        )
        features.chain_similarity = (
            self.extract_chain_similarity(
                campaign_context,
                operation_context
            )
        )
        features.graph_similarity = (
            self.extract_graph_similarity(
                campaign_context,
                operation_context
            )
        )
        features.prediction_similarity = (
            self.extract_prediction_similarity(
                campaign_context,
                operation_context
            )
        )
        return features
    def extract_attacker_similarity(
        self,
        campaign_context,
        operation_context
    ):
        if not operation_context.primary_attacker:
            return 0.0
        return float(
            campaign_context.attacker_ip == operation_context.primary_attacker
        )

    def extract_victim_similarity(
        self,
        campaign_context,
        operation_context
    ):
        return float(
            campaign_context.victim_ip
            in operation_context.victims
        )

    def extract_technique_similarity(
        self,
        campaign_context,
        operation_context
    ):
        campaign = campaign_context.techniques
        operation = operation_context.techniques
        if not campaign and not operation:
            return 0.0
        intersection = len(campaign & operation)
        union = len(campaign | operation)
        return intersection / union

    def extract_temporal_similarity(
        self,
        campaign_context,
        operation_context
    ):
        campaign_time = campaign_context.first_seen
        operation_time = operation_context.last_seen
        if campaign_time is None or operation_time is None:
            return 0.0
        campaign_dt = normalize_datetime(campaign_time)
        operation_dt = normalize_datetime(operation_time)
        
        if campaign_dt is None or operation_dt is None:
            return 0.0
        
        if campaign_dt.tzinfo is None:
            campaign_dt = campaign_dt.replace(tzinfo=timezone.utc)
        
        if operation_dt.tzinfo is None:
            operation_dt = operation_dt.replace(tzinfo=timezone.utc)
        delta_hours = abs(
            (campaign_dt - operation_dt).total_seconds()
        ) / 3600
        if delta_hours <= 1:
            return 1.0
        if delta_hours <= 24:
            return 0.8
        if delta_hours <= 24 * 7:
            return 0.5
        return 0.0

    def extract_chain_similarity(
        self,
        campaign_context,
        operation_context
    ):
        campaign = campaign_context.attack_chain
        operation = operation_context.attack_chain
        if not campaign or not operation:
            return 0.0
        m = len(campaign)
        n = len(operation)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if campaign[i - 1] == operation[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(
                        dp[i - 1][j],
                        dp[i][j - 1]
                    )
        lcs_length = dp[m][n]
        return lcs_length / max(m, n)

    def extract_prediction_similarity(
        self,
        campaign_context,
        operation_context
    ):
        if campaign_context.last_technique is None:
            return 0.0
    
        distribution = get_prediction_distribution(
            campaign_context.last_technique
        )
    
        if not distribution:
            return 0.0
    
        if not operation_context.techniques:
            return 0.0
    
        score = 0.0
    
        for technique in operation_context.techniques:
            score = max(
                score,
                distribution.get(
                    technique,
                    0.0
                )
            )
    
        return round(score, 2)

    def extract_graph_similarity(
        self,
        campaign_context,
        operation_context
    ):
        return 0.0