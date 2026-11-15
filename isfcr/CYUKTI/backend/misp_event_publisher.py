"""
misp_event_publisher.py
========================
Enhanced MISP event publishing for CYUKTI campaigns and operations.

This module creates properly structured MISP events from CYUKTI's
campaign intelligence, including:
- Campaign metadata and timeline
- ATT&CK techniques observed
- IOCs (IPs, domains, files)
- Severity and risk scores
- Operation correlation

Addresses the requirement to "make it publish in both MISP and Shuffle"
by ensuring MISP events are properly structured and published.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from cti_publisher import CTIPublisher
from neo4j_client import driver

logger = logging.getLogger(__name__)


@dataclass
class CampaignIntelligence:
    """Campaign data extracted from Neo4j for MISP publishing."""
    campaign_id: str
    attacker_ip: str
    victim_ip: str
    techniques: List[str]
    first_seen: datetime
    last_seen: datetime
    status: str
    risk_score: float
    severity: str
    alert_count: int
    operation_id: Optional[str] = None
    operation_name: Optional[str] = None
    attack_chain: List[str] = None
    
    def __post_init__(self):
        if self.attack_chain is None:
            self.attack_chain = []


class MISPEventPublisher:
    """
    Publishes CYUKTI campaigns to MISP as properly structured threat events.
    """
    
    def __init__(self, misp_url: str, api_key: str, verify_ssl: bool = False):
        """
        Initialize MISP publisher.
        
        Args:
            misp_url: MISP instance URL
            api_key: MISP API key
            verify_ssl: Whether to verify SSL certificates
        """
        self.publisher = CTIPublisher(misp_url, api_key, verify_ssl)
        logger.info(f"Initialized MISPEventPublisher for {misp_url}")
    
    def get_campaign_intelligence(self, campaign_id: str) -> Optional[CampaignIntelligence]:
        """
        Extract campaign intelligence from Neo4j.
        
        Args:
            campaign_id: Campaign ID to extract
        
        Returns:
            CampaignIntelligence object or None if not found
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
                
                # Get attack chain
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
                
                return CampaignIntelligence(
                    campaign_id=result["campaign_id"],
                    attacker_ip=result["attacker_ip"],
                    victim_ip=result["victim_ip"],
                    techniques=result["techniques"] or [],
                    first_seen=result["first_seen"],
                    last_seen=result["last_seen"],
                    status=result["status"],
                    risk_score=result["risk_score"] or 0.0,
                    severity=result["severity"] or "UNKNOWN",
                    alert_count=result["alert_count"] or 0,
                    operation_id=result["operation_id"],
                    operation_name=result["operation_name"],
                    attack_chain=attack_chain
                )
                
        except Exception as e:
            logger.exception(f"Failed to extract campaign {campaign_id}: {e}")
            return None
    
    def build_misp_event(self, intel: CampaignIntelligence) -> Dict[str, Any]:
        """
        Build MISP event structure from campaign intelligence.
        
        Args:
            intel: CampaignIntelligence object
        
        Returns:
            MISP event dict ready for publishing
        """
        # Determine threat level based on severity
        threat_level_map = {
            "CRITICAL": "1",  # High
            "HIGH": "2",      # Medium
            "MEDIUM": "3",    # Low
            "LOW": "4",       # Undefined
            "UNKNOWN": "4"
        }
        threat_level = threat_level_map.get(intel.severity, "4")
        
        # Build event info
        event_info = f"CYUKTI Campaign {intel.campaign_id}"
        if intel.operation_name:
            event_info += f" (Operation: {intel.operation_name})"
        
        # Build tags
        tags = [
            {"name": f"cyukti:campaign-id=\"{intel.campaign_id}\""},
            {"name": f"cyukti:severity=\"{intel.severity}\""},
            {"name": f"cyukti:status=\"{intel.status}\""},
        ]
        
        if intel.operation_id:
            tags.append({"name": f"cyukti:operation-id=\"{intel.operation_id}\""})
        
        # Add ATT&CK technique tags
        for technique in intel.techniques:
            tags.append({"name": f"misp-galaxy:mitre-attack-pattern=\"{technique}\""})
        
        # Build attributes (IOCs and metadata)
        attributes = []
        
        # Add IPs
        attributes.append({
            "type": "ip-src",
            "category": "Network activity",
            "value": intel.attacker_ip,
            "comment": "Attacker source IP",
            "to_ids": True
        })
        
        attributes.append({
            "type": "ip-dst",
            "category": "Network activity",
            "value": intel.victim_ip,
            "comment": "Target victim IP",
            "to_ids": False
        })
        
        # Add risk score as text attribute
        attributes.append({
            "type": "text",
            "category": "Other",
            "value": f"Risk Score: {intel.risk_score:.2f}",
            "comment": "CYUKTI computed risk score",
            "to_ids": False
        })
        
        # Add alert count
        attributes.append({
            "type": "counter",
            "category": "Other",
            "value": str(intel.alert_count),
            "comment": "Number of alerts in campaign",
            "to_ids": False
        })
        
        # Add attack chain as text
        if intel.attack_chain:
            chain_str = " -> ".join(intel.attack_chain)
            attributes.append({
                "type": "text",
                "category": "Other",
                "value": f"Attack Chain: {chain_str}",
                "comment": "Observed ATT&CK technique sequence",
                "to_ids": False
            })
        
        # Add timeline
        attributes.append({
            "type": "datetime",
            "category": "Other",
            "value": intel.first_seen.isoformat() if isinstance(intel.first_seen, datetime) else str(intel.first_seen),
            "comment": "Campaign first seen",
            "to_ids": False
        })
        
        attributes.append({
            "type": "datetime",
            "category": "Other",
            "value": intel.last_seen.isoformat() if isinstance(intel.last_seen, datetime) else str(intel.last_seen),
            "comment": "Campaign last seen",
            "to_ids": False
        })
        
        # Build complete event
        event = {
            "Event": {
                "info": event_info,
                "threat_level_id": threat_level,
                "analysis": "2" if intel.status == "ACTIVE" else "1",  # 2=Complete, 1=Ongoing
                "distribution": "3",  # All communities
                "published": False,
                "Attribute": attributes,
                "Tag": tags
            }
        }
        
        return event
    
    def publish_campaign(self, campaign_id: str, update: bool = False) -> Dict[str, Any]:
        """
        Publish a campaign to MISP.
        
        Args:
            campaign_id: Campaign ID to publish
            update: If True, update existing event instead of creating new
        
        Returns:
            Dict with:
            - success: bool
            - event_id: MISP event ID (if successful)
            - error: error message (if failed)
        """
        try:
            # Extract intelligence
            intel = self.get_campaign_intelligence(campaign_id)
            if not intel:
                return {
                    "success": False,
                    "error": f"Campaign {campaign_id} not found"
                }
            
            # Build MISP event
            event = self.build_misp_event(intel)
            
            # Check if event already exists
            existing_event_id = self.publisher.search_campaign(campaign_id)
            
            if existing_event_id and update:
                # Update existing event
                result = self.publisher.update_event(existing_event_id, event)
                result["event_id"] = existing_event_id
                logger.info(
                    f"Updated MISP event {existing_event_id} for campaign {campaign_id}"
                )
            elif existing_event_id and not update:
                # Event exists but update=False
                return {
                    "success": False,
                    "event_id": existing_event_id,
                    "error": "Event already exists (use update=True to update)"
                }
            else:
                # Create new event
                result = self.publisher.create_event(event)
                logger.info(
                    f"Created MISP event {result.get('event_id')} for campaign {campaign_id}"
                )
            
            return result
            
        except Exception as e:
            logger.exception(f"Failed to publish campaign {campaign_id}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def publish_operation(self, operation_id: str) -> Dict[str, Any]:
        """
        Publish all campaigns in an operation to MISP.
        
        Args:
            operation_id: Operation ID to publish
        
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
                    results[campaign_id] = self.publish_campaign(campaign_id, update=True)
                
                success_count = sum(1 for r in results.values() if r.get("success"))
                
                logger.info(
                    f"Published operation {operation_id}: "
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
            logger.exception(f"Failed to publish operation {operation_id}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def health_check(self) -> bool:
        """Check MISP connectivity."""
        return self.publisher.health_check()


def create_publisher_from_config() -> MISPEventPublisher:
    """Create MISP publisher from config.py settings."""
    import config
    return MISPEventPublisher(
        misp_url=config.MISP_URL,
        api_key=config.MISP_API_KEY,
        verify_ssl=config.VERIFY_MISP_SSL
    )
