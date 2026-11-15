"""
shuffle_workflow_publisher.py
==============================
Enhanced Shuffle workflow publishing for CYUKTI campaigns and operations.

This module creates properly structured Shuffle workflow payloads from
CYUKTI's campaign intelligence for automated response workflows.

Addresses the requirement to "make it publish in both MISP and Shuffle"
by ensuring campaign data is properly formatted for Shuffle workflows.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict

from soar.shuffle_client import ShuffleClient, ShuffleTriggerOutcome
from neo4j_client import driver

logger = logging.getLogger(__name__)


@dataclass
class ShuffleCampaignPayload:
    """Campaign data formatted for Shuffle workflows."""
    campaign_id: str
    attacker_ip: str
    victim_ip: str
    techniques: List[str]
    attack_chain: List[str]
    first_seen: str
    last_seen: str
    status: str
    risk_score: float
    severity: str
    alert_count: int
    operation_id: Optional[str] = None
    operation_name: Optional[str] = None
    recommended_actions: List[str] = None
    
    def __post_init__(self):
        if self.recommended_actions is None:
            self.recommended_actions = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


class ShuffleWorkflowPublisher:
    """
    Publishes CYUKTI campaigns to Shuffle workflows for automated response.
    """
    
    def __init__(self, shuffle_client: ShuffleClient):
        """
        Initialize Shuffle publisher.
        
        Args:
            shuffle_client: Configured ShuffleClient instance
        """
        self.client = shuffle_client
        logger.info("Initialized ShuffleWorkflowPublisher")
    
    def get_campaign_data(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract campaign data from Neo4j.
        
        Args:
            campaign_id: Campaign ID to extract
        
        Returns:
            Campaign data dict or None if not found
        """
        try:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Campaign {campaign_id: $campaign_id})
                    OPTIONAL MATCH (c)-[:IN_OPERATION]->(o:Operation)
                    OPTIONAL MATCH (c)-[:OBSERVED_TECHNIQUE]->(t:Technique)
                    
                    RETURN 
                        c.campaign_id AS campaign_id,
                        c.attacker_ip AS attacker_ip,
                        c.victim_ip AS victim_ip,
                        c.first_seen AS first_seen,
                        c.last_seen AS last_seen,
                        c.status AS status,
                        c.risk_score AS risk_score,
                        c.severity AS severity,
                        c.alert_count AS alert_count,
                        o.operation_id AS operation_id,
                        o.name AS operation_name,
                        collect(DISTINCT t.attack_id) AS techniques
                    """,
                    campaign_id=campaign_id
                ).single()
                
                if not result:
                    logger.warning(f"Campaign {campaign_id} not found in database")
                    return None
                
                # Get attack chain in order
                chain_result = session.run(
                    """
                    MATCH (c:Campaign {campaign_id: $campaign_id})
                          -[:HAS_EVENT]->(e:AttackEvent)
                          -[:REPRESENTS_TECHNIQUE]->(t:Technique)
                    RETURN t.attack_id AS technique
                    ORDER BY e.timestamp
                    """,
                    campaign_id=campaign_id
                )
                
                attack_chain = [r["technique"] for r in chain_result if r["technique"]]
                
                # Get recommended actions
                actions_result = session.run(
                    """
                    MATCH (c:Campaign {campaign_id: $campaign_id})
                          -[:RECOMMENDED_ACTION]->(a:Action)
                    RETURN collect(a.name) AS actions
                    """,
                    campaign_id=campaign_id
                ).single()
                
                recommended_actions = actions_result["actions"] if actions_result else []
                
                return {
                    "campaign_id": result["campaign_id"],
                    "attacker_ip": result["attacker_ip"],
                    "victim_ip": result["victim_ip"],
                    "techniques": result["techniques"] or [],
                    "attack_chain": attack_chain,
                    "first_seen": result["first_seen"].isoformat() if isinstance(result["first_seen"], datetime) else str(result["first_seen"]),
                    "last_seen": result["last_seen"].isoformat() if isinstance(result["last_seen"], datetime) else str(result["last_seen"]),
                    "status": result["status"],
                    "risk_score": float(result["risk_score"] or 0.0),
                    "severity": result["severity"] or "UNKNOWN",
                    "alert_count": int(result["alert_count"] or 0),
                    "operation_id": result["operation_id"],
                    "operation_name": result["operation_name"],
                    "recommended_actions": recommended_actions
                }
                
        except Exception as e:
            logger.exception(f"Failed to extract campaign {campaign_id}: {e}")
            return None
    
    def build_workflow_payload(
        self,
        campaign_data: Dict[str, Any],
        workflow_type: str = "incident_response"
    ) -> Dict[str, Any]:
        """
        Build Shuffle workflow payload from campaign data.
        
        Args:
            campaign_data: Campaign data dict
            workflow_type: Type of workflow to trigger
        
        Returns:
            Shuffle workflow payload dict
        """
        # Build base payload
        payload = {
            "workflow_type": workflow_type,
            "source": "CYUKTI",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "campaign": campaign_data
        }
        
        # Add workflow-specific data based on severity
        severity = campaign_data.get("severity", "UNKNOWN")
        
        if severity in ("CRITICAL", "HIGH"):
            payload["priority"] = "high"
            payload["auto_response_enabled"] = False  # Require human approval
        else:
            payload["priority"] = "medium"
            payload["auto_response_enabled"] = False  # Always require approval by default
        
        # Add suggested actions based on techniques observed
        techniques = campaign_data.get("techniques", [])
        suggested_actions = self._get_suggested_actions(techniques, severity)
        payload["suggested_actions"] = suggested_actions
        
        # Add context for decision making
        payload["context"] = {
            "attack_stage": self._determine_attack_stage(campaign_data.get("attack_chain", [])),
            "risk_level": self._risk_level_from_score(campaign_data.get("risk_score", 0)),
            "requires_isolation": severity in ("CRITICAL", "HIGH"),
            "requires_forensics": True,
            "escalate_to_soc": severity in ("CRITICAL", "HIGH")
        }
        
        return payload
    
    def _get_suggested_actions(
        self,
        techniques: List[str],
        severity: str
    ) -> List[Dict[str, str]]:
        """
        Get suggested response actions based on observed techniques.
        
        Args:
            techniques: List of ATT&CK technique IDs
            severity: Campaign severity
        
        Returns:
            List of suggested action dicts
        """
        actions = []
        
        # Technique-specific actions
        if "T1110" in techniques or "T1110.001" in techniques:
            actions.append({
                "action": "block_source_ip",
                "description": "Block brute force source IP",
                "requires_approval": severity not in ("CRITICAL", "HIGH")
            })
        
        if "T1021.004" in techniques:  # SSH
            actions.append({
                "action": "disable_ssh_access",
                "description": "Temporarily disable SSH access for compromised account",
                "requires_approval": True
            })
        
        if "T1078" in techniques:  # Valid Accounts
            actions.append({
                "action": "reset_credentials",
                "description": "Force password reset for affected accounts",
                "requires_approval": True
            })
        
        if "T1059" in techniques:  # Command and Scripting Interpreter
            actions.append({
                "action": "isolate_host",
                "description": "Isolate victim host from network",
                "requires_approval": True
            })
        
        if "T1041" in techniques or "T1048" in techniques:  # Exfiltration
            actions.append({
                "action": "block_outbound_traffic",
                "description": "Block outbound traffic to suspicious destinations",
                "requires_approval": severity == "CRITICAL"
            })
        
        # Always add forensics collection
        actions.append({
            "action": "collect_forensics",
            "description": "Collect forensic data from affected systems",
            "requires_approval": False
        })
        
        return actions
    
    def _determine_attack_stage(self, attack_chain: List[str]) -> str:
        """
        Determine current attack stage from technique chain.
        
        Args:
            attack_chain: Ordered list of technique IDs
        
        Returns:
            Attack stage name
        """
        if not attack_chain:
            return "unknown"
        
        # Map techniques to stages (simplified)
        stage_map = {
            "T1595": "reconnaissance",
            "T1110": "initial_access",
            "T1078": "persistence",
            "T1548": "privilege_escalation",
            "T1021": "lateral_movement",
            "T1041": "exfiltration"
        }
        
        # Return stage of most recent technique
        last_technique = attack_chain[-1]
        return stage_map.get(last_technique, "unknown")
    
    def _risk_level_from_score(self, risk_score: float) -> str:
        """Convert numeric risk score to level."""
        if risk_score >= 75:
            return "critical"
        elif risk_score >= 50:
            return "high"
        elif risk_score >= 25:
            return "medium"
        else:
            return "low"
    
    def publish_campaign(
        self,
        campaign_id: str,
        workflow_type: str = "incident_response"
    ) -> Dict[str, Any]:
        """
        Publish a campaign to Shuffle workflow.
        
        Args:
            campaign_id: Campaign ID to publish
            workflow_type: Type of workflow to trigger
        
        Returns:
            Dict with:
            - success: bool
            - outcome: ShuffleTriggerOutcome
            - execution_id: Shuffle execution ID (if available)
            - error: error message (if failed)
        """
        try:
            # Get campaign data
            campaign_data = self.get_campaign_data(campaign_id)
            if not campaign_data:
                return {
                    "success": False,
                    "error": f"Campaign {campaign_id} not found"
                }
            
            # Build payload
            payload = self.build_workflow_payload(campaign_data, workflow_type)
            
            # Trigger workflow
            result = self.client.trigger(payload)
            
            success = result.outcome in (
                ShuffleTriggerOutcome.TRIGGERED,
                ShuffleTriggerOutcome.SYNCHRONOUS_RESULT
            )
            
            logger.info(
                f"Published campaign {campaign_id} to Shuffle: "
                f"outcome={result.outcome.value}"
            )
            
            return {
                "success": success,
                "outcome": result.outcome.value,
                "execution_id": result.shuffle_execution_id,
                "output": result.output,
                "error": result.error
            }
            
        except Exception as e:
            logger.exception(f"Failed to publish campaign {campaign_id} to Shuffle")
            return {
                "success": False,
                "error": str(e)
            }
    
    def publish_operation(
        self,
        operation_id: str,
        workflow_type: str = "incident_response"
    ) -> Dict[str, Any]:
        """
        Publish all campaigns in an operation to Shuffle.
        
        Args:
            operation_id: Operation ID to publish
            workflow_type: Type of workflow to trigger
        
        Returns:
            Dict with results per campaign
        """
        try:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (o:Operation {operation_id: $operation_id})
                          <-[:IN_OPERATION]-(c:Campaign)
                    RETURN collect(c.campaign_id) AS campaigns
                    """,
                    operation_id=operation_id
                ).single()
                
                if not result or not result["campaigns"]:
                    return {
                        "success": False,
                        "error": f"Operation {operation_id} not found or has no campaigns"
                    }
                
                campaigns = result["campaigns"]
                results = {}
                
                for campaign_id in campaigns:
                    results[campaign_id] = self.publish_campaign(
                        campaign_id,
                        workflow_type
                    )
                
                success_count = sum(1 for r in results.values() if r.get("success"))
                
                logger.info(
                    f"Published operation {operation_id} to Shuffle: "
                    f"{success_count}/{len(campaigns)} campaigns successful"
                )
                
                return {
                    "success": success_count == len(campaigns),
                    "operation_id": operation_id,
                    "campaigns": results,
                    "success_count": success_count,
                    "total_count": len(campaigns)
                }
                
        except Exception as e:
            logger.exception(f"Failed to publish operation {operation_id} to Shuffle")
            return {
                "success": False,
                "error": str(e)
            }
    
    def health_check(self) -> bool:
        """Check Shuffle connectivity."""
        return self.client.health_check()


def create_publisher_from_config() -> ShuffleWorkflowPublisher:
    """Create Shuffle publisher from config.py settings."""
    from soar.shuffle_client import build_default_client
    client = build_default_client()
    return ShuffleWorkflowPublisher(client)
