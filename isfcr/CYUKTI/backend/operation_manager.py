from campaign_context import CampaignContext
from operation_context import OperationContext
from neo4j_client import (
    get_campaign_context_data,
    get_operation_context_data,
    expire_stale_operations_db
)
from config import OPERATION_TIMEOUT
from datetime_utils import normalize_datetime

class OperationManager:
    def build_campaign_context(self, campaign_id):
        data = get_campaign_context_data(campaign_id)
        if data is None:
            return None
        campaign = data["campaign"]
        return CampaignContext(
            campaign_id=campaign["campaign_id"],
            attacker_ip=campaign.get("attacker_ip", ""),
            victim_ip=campaign.get("victim_ip", ""),
            status=campaign.get("status", "ACTIVE"),
            first_seen=normalize_datetime(
                campaign.get("first_seen")
            ),
            last_seen=normalize_datetime(
                campaign.get("last_seen")
            ),
            last_technique=campaign.get("last_technique"),
            predicted_next=campaign.get("predicted_next"),
            prediction_confidence=campaign.get(
                "prediction_confidence", 0.0
            ),
            prediction_hits=campaign.get(
                "prediction_hits", 0
            ),
            prediction_misses=campaign.get(
                "prediction_misses", 0
            ),
            prediction_generated_at=normalize_datetime(
                campaign.get("prediction_generated_at")
            ),
            reopened_count=campaign.get(
                "reopened_count", 0
            ),
            risk_score=campaign.get(
                "risk_score", 0.0
            ),
            techniques=set(data["techniques"]),
            attack_chain=data["attack_chain"],
            observed_chain=data["attack_chain"]
        )

    def build_operation_context(self, operation_id):
        data = get_operation_context_data(operation_id)
        if data is None:
            return None
        operation = data["operation"]
        campaigns = data["campaigns"]
        return OperationContext(
            operation_id=operation["operation_id"],
            campaign_ids=[
                c["campaign_id"]
                for c in campaigns
            ],
            primary_attacker=operation.get("primary_attacker", ""),
            victims={
                c.get("victim_ip", "")
                for c in campaigns
                if c.get("victim_ip")
            },
            techniques=set(data["techniques"]),
            attack_chain=data["attack_chain"],
            first_seen=normalize_datetime(
                operation.get("created_at")
            ),
            
            last_seen=normalize_datetime(
                operation.get("last_seen")
            ),
        )

    def expire_active_operations(self):
        expired = expire_stale_operations_db(OPERATION_TIMEOUT)

        for operation_id in expired:
            print(f"[OPERATION EXPIRED] {operation_id}")

        return expired

operation_manager = OperationManager()