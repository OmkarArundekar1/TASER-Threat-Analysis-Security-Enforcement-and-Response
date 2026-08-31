from neo4j import GraphDatabase
import uuid
from datetime import datetime, timezone, timedelta
from config import DEDUP_WINDOW, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from graph_schema import (
    ATTACKER_DEFAULTS,
    ATTACK_EVENT_DEFAULTS,
    CAMPAIGN_DEFAULTS
)
from operation_schema import OPERATION_DEFAULTS
URI = NEO4J_URI
USERNAME = NEO4J_USERNAME
PASSWORD = NEO4J_PASSWORD

from dataclasses import dataclass
from typing import List, Optional
from datetime_utils import normalize_datetime

@dataclass
class MitreFeatures:
    attack_id: str
    technique_name: str
    description: str
    platforms: List[str]
    platform_count: int
    domains: List[str]
    domain_count: int
    kill_chain_phases: List[str]
    kill_chain_count: int
    is_subtechnique: bool
    deprecated: bool
    revoked: bool
    object_version: str
    threat_actor_count: int
    malware_count: int
    tool_count: int
    mitigation_count: int
    subtechnique_count: int
    parent_technique: Optional[str]

@dataclass
class RuntimeGraphFeatures:
    campaign_count: int
    attack_event_count: int
    graph_degree: int
    incoming_chain_count: int
    outgoing_chain_count: int
    prediction_frequency: int
    duplicate_frequency: int

@dataclass
class ThreatIntelFeatures:
    ip_reputation: float
    threat_actor_reputation: float
    malware_confidence: float
    tool_confidence: float
    misp_confidence: float
    ioc_confidence: float

@dataclass
class DetectionConfidence:
    wazuh_level: int
    suricata_score: float
    zeek_score: float
    sigma_score: float
    yara_score: float
    detection_confidence: float

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)
def create_attack_event(
    campaign_id,
    attacker_ip,
    victim_ip,
    mitre_id,
    technique,
    stage,
    tps,
    fingerprint,
    rule_id,
    agent_id,
    rule_level=0,
    investigation_payload="{}",
    mitre_provenance="NATIVE_WAZUH",
    mitre_confidence="CONFIRMED",
    mitre_reason="Wazuh rule.mitre.id",
    mitre_resolver_version="",
    mitre_technique_ids=None,
):
    event_id = str(uuid.uuid4())
    with driver.session() as session:

        query = """
        MERGE (a:Attacker {
            ip:$attacker_ip
        })
        
        ON CREATE SET
            a.first_seen = datetime(),
        
            a.vt_reputation = $vt_reputation,
            a.threat_actor_reputation = $threat_actor_reputation,
        
            a.malware_confidence = $malware_confidence,
            a.tool_confidence = $tool_confidence,
            a.misp_confidence = $misp_confidence,
            a.ioc_confidence = $ioc_confidence
        
        SET
            a.last_seen = datetime()
        WITH a
        MERGE (h:Host {
            ip:$victim_ip
        })

        ON CREATE SET
            h.first_seen = datetime()

        SET
            h.last_seen = datetime()
        WITH a, h
        MATCH (c:Campaign {
            campaign_id:$campaign_id
        })
        
        SET
            c.last_seen = datetime(),
            c.last_technique = $mitre_id,
            c.occurrences = coalesce(c.occurrences,0)+1,
            c.total_tps = coalesce(c.total_tps,0)+$tps,
            c.risk_score = coalesce(c.risk_score,0)+$tps
        WITH a, h, c
        MERGE (t:Technique {
            attack_id:$mitre_id
        })

        ON CREATE SET
            t.name = $technique,
            t.stage = $stage,
            t.first_seen = datetime(),
            t.last_seen = datetime(),
            t.occurrences = 1,
            t.total_tps = $tps

        ON MATCH SET
            t.last_seen = datetime(),
            t.occurrences = coalesce(t.occurrences,0)+1,
            t.total_tps = coalesce(t.total_tps,0)+$tps
        WITH a, h, c, t
        MERGE (s:Stage {
            name:$stage
        })
        WITH a, h, c, t, s
        CREATE (e:AttackEvent {
            event_id:$event_id,
            campaign_id:$campaign_id,
            attack_id:$mitre_id,
            fingerprint:$fingerprint,
            rule_id:$rule_id,
            agent_id:$agent_id,
            technique:$technique,
            stage:$stage,
            attacker_ip:$attacker_ip,
            victim_ip:$victim_ip,
            first_seen:datetime(),
            last_seen:datetime(),
            occurrences:1,
            tps:$tps,
            rule_level:$rule_level,
            suricata_score:$suricata_score,
            zeek_score:$zeek_score,
            sigma_score:$sigma_score,
            yara_score:$yara_score,
            investigation_payload:$investigation_payload,
            prediction_generated_at:$prediction_generated_at,
            mitre_status:"RESOLVED",
            mitre_provenance:$mitre_provenance,
            mitre_confidence:$mitre_confidence,
            mitre_reason:$mitre_reason,
            mitre_resolver_version:$mitre_resolver_version,
            mitre_technique_ids:$mitre_technique_ids
        })
        WITH a, h, c, t, s, e
        MERGE (a)-[:LAUNCHED]->(c)

        MERGE (c)-[:TARGETS]->(h)

        MERGE (c)-[:HAS_EVENT]->(e)

        MERGE (e)-[:MATCHES]->(t)

        MERGE (t)-[:BELONGS_TO]->(s)

        RETURN
            e.event_id AS event_id
        """

        result = session.run(
            query,
            event_id=event_id,
            campaign_id=campaign_id,
            attacker_ip=attacker_ip,
            victim_ip=victim_ip,
            mitre_id=mitre_id,
            fingerprint=fingerprint,
            rule_id=rule_id,
            agent_id=agent_id,
            technique=technique,
            stage=stage,
            tps=tps,
            investigation_payload=investigation_payload,
            prediction_generated_at=CAMPAIGN_DEFAULTS["prediction_generated_at"],
            vt_reputation=ATTACKER_DEFAULTS["vt_reputation"],
            threat_actor_reputation=ATTACKER_DEFAULTS["threat_actor_reputation"],
            malware_confidence=ATTACKER_DEFAULTS["malware_confidence"],
            tool_confidence=ATTACKER_DEFAULTS["tool_confidence"],
            misp_confidence=ATTACKER_DEFAULTS["misp_confidence"],
            ioc_confidence=ATTACKER_DEFAULTS["ioc_confidence"],            
            rule_level=rule_level,
            suricata_score=ATTACK_EVENT_DEFAULTS["suricata_score"],
            zeek_score=ATTACK_EVENT_DEFAULTS["zeek_score"],
            sigma_score=ATTACK_EVENT_DEFAULTS["sigma_score"],
            yara_score=ATTACK_EVENT_DEFAULTS["yara_score"],
            mitre_provenance=mitre_provenance,
            mitre_confidence=mitre_confidence,
            mitre_reason=mitre_reason,
            mitre_resolver_version=mitre_resolver_version,
            mitre_technique_ids=list(mitre_technique_ids or [mitre_id]),
        )

        record = result.single()
        return record["event_id"]


def create_unattributed_attack_event(
    campaign_id,
    attacker_ip,
    victim_ip,
    fingerprint,
    rule_id,
    agent_id,
    rule_level=0,
    investigation_payload="{}",
    mitre_provenance="UNKNOWN",
    mitre_confidence="NONE",
    mitre_reason="",
    mitre_resolver_version="",
    mitre_technique_ids=None,
):
    """Phase 20: creates an AttackEvent for a Wazuh alert that
    mitre_resolver.resolve_mitre() could not defensibly attribute to an
    ATT&CK technique. Preserves the raw alert as evidence and attaches it
    to the campaign via HAS_EVENT, but deliberately:

      - never sets AttackEvent.attack_id (omitted, not a fake technique)
      - never creates or MERGEs a Technique node/MATCHES relationship
      - never touches Campaign.last_technique (only last_seen/occurrences)
      - never touches Campaign.total_tps/risk_score (tps contribution is
        zero for an unattributed event, so leaving them unchanged is
        already correct -- no arithmetic needed)

    This intentionally does NOT call campaign_manager.append_technique(),
    chain_updater.update_attack_chain(), or prediction_engine.predict_next()
    -- those all key on a real technique ID and must not run for an
    unattributed event (see realtime_socgraph.py's UNKNOWN branch).
    """
    event_id = str(uuid.uuid4())
    with driver.session() as session:
        query = """
        MERGE (a:Attacker {
            ip:$attacker_ip
        })

        ON CREATE SET
            a.first_seen = datetime()

        SET
            a.last_seen = datetime()
        WITH a
        MERGE (h:Host {
            ip:$victim_ip
        })

        ON CREATE SET
            h.first_seen = datetime()

        SET
            h.last_seen = datetime()
        WITH a, h
        MATCH (c:Campaign {
            campaign_id:$campaign_id
        })

        SET
            c.last_seen = datetime(),
            c.occurrences = coalesce(c.occurrences,0)+1
        WITH a, h, c
        CREATE (e:AttackEvent {
            event_id:$event_id,
            campaign_id:$campaign_id,
            fingerprint:$fingerprint,
            rule_id:$rule_id,
            agent_id:$agent_id,
            attacker_ip:$attacker_ip,
            victim_ip:$victim_ip,
            first_seen:datetime(),
            last_seen:datetime(),
            occurrences:1,
            tps:0,
            rule_level:$rule_level,
            investigation_payload:$investigation_payload,
            mitre_status:"UNKNOWN",
            mitre_provenance:$mitre_provenance,
            mitre_confidence:$mitre_confidence,
            mitre_reason:$mitre_reason,
            mitre_resolver_version:$mitre_resolver_version,
            mitre_technique_ids:$mitre_technique_ids
        })
        WITH a, h, c, e
        MERGE (a)-[:LAUNCHED]->(c)

        MERGE (c)-[:TARGETS]->(h)

        MERGE (c)-[:HAS_EVENT]->(e)

        RETURN
            e.event_id AS event_id
        """

        result = session.run(
            query,
            event_id=event_id,
            campaign_id=campaign_id,
            attacker_ip=attacker_ip,
            victim_ip=victim_ip,
            fingerprint=fingerprint,
            rule_id=rule_id,
            agent_id=agent_id,
            rule_level=rule_level,
            investigation_payload=investigation_payload,
            mitre_provenance=mitre_provenance,
            mitre_confidence=mitre_confidence,
            mitre_reason=mitre_reason,
            mitre_resolver_version=mitre_resolver_version,
            mitre_technique_ids=list(mitre_technique_ids or []),
        )

        record = result.single()
        return record["event_id"]


def update_duplicate_event(event_id, tps, investigation_payload="{}"):
    with driver.session() as session:
        session.run(
            """
            MATCH (e:AttackEvent {event_id:$event_id})

            SET
                e.last_seen = datetime(),
                e.occurrences = coalesce(e.occurrences,0) + 1,
                e.tps = coalesce(e.tps,0) + $tps,
                e.investigation_payload = $investigation_payload

            WITH e

            MATCH (e)-[:MATCHES]->(t:Technique)

            SET
                t.last_seen = datetime(),
                t.occurrences = coalesce(t.occurrences,0) + 1,
                t.total_tps = coalesce(t.total_tps,0) + $tps

            WITH e

            MATCH (c:Campaign {campaign_id:e.campaign_id})

            SET
                c.last_seen = datetime(),
                c.occurrences = coalesce(c.occurrences,0) + 1,
                c.total_tps = coalesce(c.total_tps,0) + $tps,
                c.risk_score = coalesce(c.risk_score,0) + $tps
            """,
            event_id=event_id,
            tps=tps,
            investigation_payload=investigation_payload
        )
def batch_update_duplicates(records):
    if not records:
        return
    with driver.session() as session:
        session.run(
            """
            UNWIND $records AS r

            MATCH (e:AttackEvent {event_id:r.event_id})

            SET
                e.last_seen = datetime(r.last_seen),
                e.occurrences = coalesce(e.occurrences,0) + r.occurrences,
                e.tps = coalesce(e.tps,0) + r.total_tps,
                e.investigation_payload = r.latest_payload

            WITH e, r

            MATCH (e)-[:MATCHES]->(t:Technique)

            SET
                t.last_seen = datetime(r.last_seen),
                t.occurrences = coalesce(t.occurrences,0) + r.occurrences,
                t.total_tps = coalesce(t.total_tps,0) + r.total_tps

            WITH e, r

            MATCH (c:Campaign {campaign_id:e.campaign_id})

            SET
                c.last_seen = datetime(r.last_seen),
                c.occurrences = coalesce(c.occurrences,0) + r.occurrences,
                c.total_tps = coalesce(c.total_tps,0) + r.total_tps,
                c.risk_score = coalesce(c.risk_score,0) + r.total_tps
            """,
            records=records
        )        
        
def store_likely_next(campaign_id, predicted_technique, confidence):
    store_prediction(
        campaign_id,
        predicted_technique,
        confidence
    )
    
def update_campaign_similarity(campaign_id):
    with driver.session() as session:
        query = """
        MATCH (c1:Campaign {campaign_id: $campaign_id})
        MATCH (c2:Campaign)
        WHERE c1 <> c2
        
        OPTIONAL MATCH (c1)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t1:Technique)
        WITH c1, c2, collect(DISTINCT t1) AS c1_techs
        OPTIONAL MATCH (c2)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t2:Technique)
        WITH c1, c2, c1_techs, collect(DISTINCT t2) AS c2_techs
        WITH c1, c2, c1_techs, c2_techs, [t IN c1_techs WHERE t IN c2_techs] AS shared_t
        WITH c1, c2, size(shared_t) AS shared_tech_count,
             size(c1_techs) + size(c2_techs) - size(shared_t) AS total_tech_count,
             [t IN shared_t | coalesce(t.name, t.attack_id)] AS shared_techniques,
             [t IN shared_t | coalesce(t.stage, 'Unknown')] AS shared_tactics_raw

        OPTIONAL MATCH (a1:Attacker)-[:LAUNCHED]->(c1)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, collect(DISTINCT a1.ip) AS c1_a
        OPTIONAL MATCH (a2:Attacker)-[:LAUNCHED]->(c2)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, c1_a, collect(DISTINCT a2.ip) AS c2_a
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, [a IN c1_a WHERE a IN c2_a] AS shared_a

        OPTIONAL MATCH (c1)-[:TARGETS]->(h1:Host)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, shared_a, collect(DISTINCT h1.ip) AS c1_h
        OPTIONAL MATCH (c2)-[:TARGETS]->(h2:Host)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, shared_a, c1_h, collect(DISTINCT h2.ip) AS c2_h
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, shared_a, [h IN c1_h WHERE h IN c2_h] AS shared_h

        WITH c1, c2, shared_techniques, shared_tactics_raw, shared_a, shared_h,
             (CASE WHEN total_tech_count = 0 THEN 0.0 ELSE toFloat(shared_tech_count) / total_tech_count END * 60.0) +
             (CASE WHEN size(shared_a) > 0 THEN 20.0 ELSE 0.0 END) +
             (CASE WHEN size(shared_h) > 0 THEN 20.0 ELSE 0.0 END) AS similarity_score
             
        WHERE similarity_score >= 75.0
        
        // Remove old relationships for this pair to avoid duplicates
        OPTIONAL MATCH (c1)-[old:SIMILAR_TO]->(c2)
        DELETE old
        
        WITH c1, c2, round(similarity_score) AS score, shared_techniques, shared_tactics_raw
        
        MERGE (c1)-[r:SIMILAR_TO]->(c2)
        SET r.score = score,
            r.shared_techniques = shared_techniques,
            r.shared_tactics = apoc.coll.toSet(shared_tactics_raw)
        """
        try:
            session.run(query, campaign_id=campaign_id)
        except Exception as e:
            if "apoc" in str(e).lower():
                fallback_query = query.replace("apoc.coll.toSet(shared_tactics_raw)", "shared_tactics_raw")
                session.run(fallback_query, campaign_id=campaign_id)
            else:
                raise e

def update_actor_attribution(campaign_id):
    with driver.session() as session:
        query = """
        MATCH (c:Campaign {campaign_id: $campaign_id})-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t:Technique)
        WITH c, collect(DISTINCT t) AS c_techs
        MATCH (ta:ThreatActor)-[:USES]->(ta_t:Technique)
        WITH c, c_techs, ta, collect(DISTINCT ta_t) AS ta_techs
        WITH c, ta, c_techs, ta_techs, [t IN c_techs WHERE t IN ta_techs] AS shared_t
        WITH c, ta, size(shared_t) AS shared_tech_count,
             size(c_techs) + size(ta_techs) - size(shared_t) AS total_tech_count,
             [t IN shared_t | coalesce(t.name, t.attack_id)] AS shared_techniques
        WHERE shared_tech_count > 0
        
        WITH c, ta, shared_tech_count, total_tech_count, shared_techniques,
        
             CASE
                 WHEN total_tech_count <= 0 THEN 0.0
                 ELSE (toFloat(shared_tech_count) / toFloat(total_tech_count)) * 100
             END AS base_confidence
             
        WITH c, ta,
             CASE WHEN base_confidence + (shared_tech_count * 2) > 99 THEN 99.0 
                  ELSE base_confidence + (shared_tech_count * 2) END AS confidence,
             shared_techniques
             
        WHERE confidence >= 50.0
        
        OPTIONAL MATCH (c)-[old:RESEMBLES]->(ta)
        DELETE old
        
        WITH c, ta, round(confidence) AS conf, shared_techniques
        
        MERGE (c)-[r:RESEMBLES]->(ta)
        SET r.confidence = conf,
            r.matched_techniques = shared_techniques
        """
        session.run(query, campaign_id=campaign_id)
def get_active_campaign_db(attacker_ip, victim_ip):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Campaign)
            WHERE c.attacker_ip = $attacker_ip
              AND c.victim_ip = $victim_ip
              AND c.status = 'ACTIVE'
            RETURN
                c.campaign_id AS campaign_id,
                c.first_seen AS first_seen,
                c.last_seen AS last_seen,
                c.last_technique AS last_technique,
                c.predicted_next AS predicted_next,
                c.prediction_confidence AS prediction_confidence,
                c.prediction_generated_at AS prediction_generated_at,
                c.reopened_count AS reopened_count,
                c.status AS status
            ORDER BY c.last_seen DESC
            LIMIT 1
            """,
            attacker_ip=attacker_ip,
            victim_ip=victim_ip
        )

        record = result.single()

        if record is None:
            return None
        return {
            "campaign_id": record["campaign_id"],
            "first_seen": record["first_seen"],
            "last_seen": record["last_seen"],
            "last_technique": record["last_technique"],
            "predicted_next": record["predicted_next"],
            "prediction_confidence": record["prediction_confidence"],
            "prediction_generated_at": record["prediction_generated_at"],
            "reopened_count": record["reopened_count"],
            "status": record["status"]
        }

def get_recent_inactive_campaign_db(
    attacker_ip,
    victim_ip,
    limit=5
):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Campaign)
            WHERE
                c.attacker_ip = $attacker_ip
                AND c.victim_ip = $victim_ip
                AND c.status IN ['INACTIVE','REOPENED']
            RETURN
                c.campaign_id AS campaign_id,
                c.first_seen AS first_seen,
                c.last_seen AS last_seen,
                c.last_technique AS last_technique,
                c.predicted_next AS predicted_next,
                c.prediction_confidence AS prediction_confidence,
                c.prediction_generated_at AS prediction_generated_at,
                c.reopened_count AS reopened_count,
                c.status AS status
            ORDER BY c.last_seen DESC
            LIMIT $limit
            """,
            attacker_ip=attacker_ip,
            victim_ip=victim_ip,
            limit=limit
        )
        campaigns = []
        for record in result:
            campaigns.append(dict(record))
        return campaigns     

def reopen_campaign_db(campaign_id):
    with driver.session() as session:
        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})
            SET
                c.status='ACTIVE',
                c.last_seen=datetime(),
                c.predicted_next=NULL,
                c.prediction_confidence=0,
                c.reopened_count=
                    coalesce(c.reopened_count,0)+1
            """,
            campaign_id=campaign_id
        )
        

def create_campaign_db(attacker_ip, victim_ip):
    campaign_id = f"CAMP_{uuid.uuid4().hex[:8].upper()}"
    print(">>>> create_campaign_db ENTER")
    with driver.session() as session:

        session.run(
            """
            CREATE (c:Campaign{
                campaign_id:$campaign_id,
                attacker_ip:$attacker_ip,
                victim_ip:$victim_ip,

                status:'ACTIVE',

                first_seen:datetime(),
                last_seen:datetime(),

                occurrences:0,
                total_tps:0,

                campaign_count:$campaign_count,
                prediction_frequency:$prediction_frequency,
                duplicate_frequency:$duplicate_frequency,

                risk_score:$risk_score,
                risk_confidence:$risk_confidence,
                risk_trend:$risk_trend,

                last_technique:"",
                predicted_next:"",
                prediction_confidence:0,
                prediction_hits:0,
                prediction_misses:0,

                reopened_count:0,

                dynamic_risk:0,
                risk_level:"LOW",
                risk_color:"GREEN",
                risk_updated_at:"",
                risk_breakdown:"{}"
            })
            """,
            campaign_id=campaign_id,
            attacker_ip=attacker_ip,
            victim_ip=victim_ip,

            campaign_count=CAMPAIGN_DEFAULTS["campaign_count"],
            prediction_frequency=CAMPAIGN_DEFAULTS["prediction_frequency"],
            duplicate_frequency=CAMPAIGN_DEFAULTS["duplicate_frequency"],
            risk_score=CAMPAIGN_DEFAULTS["risk_score"],
            risk_confidence=CAMPAIGN_DEFAULTS["risk_confidence"],
            risk_trend=CAMPAIGN_DEFAULTS["risk_trend"]
        )

    return campaign_id
    
def update_campaign_activity(campaign_id, technique=None):
    with driver.session() as session:
        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})

            SET
                c.last_seen = datetime(),
                c.status = 'ACTIVE',
                c.last_technique = $technique
            """,
            campaign_id=campaign_id,
            technique=technique
        )
        
def close_campaign_db(campaign_id):
    with driver.session() as session:

        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})

            SET
                c.status='INACTIVE',
                c.closed_at=datetime()
            """,
            campaign_id=campaign_id
        )

def expire_stale_campaigns_db(timeout_seconds):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Campaign)
            WHERE
                c.status = 'ACTIVE'
                AND
                c.last_seen < datetime() - duration({seconds:$timeout})
            SET
                c.status = 'INACTIVE',
                c.closed_at = datetime()
            RETURN
                c.campaign_id AS campaign_id
            """,
            timeout=timeout_seconds
        )
        return [
            record["campaign_id"]
            for record in result
        ]

def create_operation_db(campaign_context):
    operation_id = f"OP_{uuid.uuid4().hex[:8].upper()}"
    with driver.session() as session:
        session.run(
            """
            CREATE (o:Operation{
                operation_id:$operation_id,
                status:$status,
                created_at:$created_at,
                last_seen:$last_seen,
                campaign_count:0,
                confidence:$confidence,
                correlation_score:$correlation_score,
                primary_attacker:$primary_attacker,
                primary_target:$primary_target,
                predicted_goal:$predicted_goal,
                notes:$notes
            })
            """,
            operation_id=operation_id,
            status="ACTIVE",
            created_at=campaign_context.first_seen,
            last_seen=campaign_context.last_seen,
            confidence=0.0,
            correlation_score=0.0,
            primary_attacker=campaign_context.attacker_ip,
            primary_target=campaign_context.victim_ip,
            predicted_goal="",
            notes=""
        )
    return operation_id
    
def update_operation_activity(
    operation_id,
    campaign_context
):
    with driver.session() as session:
        session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
            SET
                o.last_seen=$last_seen,
                o.status='ACTIVE'
            """,
            operation_id=operation_id,
            last_seen=campaign_context.last_seen
        )
        
def close_operation_db(operation_id):
    with driver.session() as session:
        session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
            SET
                o.status='INACTIVE',
                o.closed_at=datetime()
            """,
            operation_id=operation_id
        )

def reopen_operation_db(operation_id):
    with driver.session() as session:
        session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})

            SET
                o.status='ACTIVE',
                o.last_seen=datetime()

            REMOVE o.closed_at
            """,
            operation_id=operation_id
        )

def attach_campaign_to_operation(operation_id, campaign_context):
    with driver.session() as session:
        session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
            MATCH (c:Campaign {campaign_id:$campaign_id})
            MERGE (o)-[r:HAS_CAMPAIGN]->(c)
            ON CREATE SET
                o.campaign_count = coalesce(o.campaign_count, 0) + 1
            SET
                o.last_seen = $last_seen,
                o.primary_attacker =
                    CASE
                        WHEN o.primary_attacker = ''
                        THEN $attacker
                        ELSE o.primary_attacker
                    END,
                o.primary_target =
                    CASE
                        WHEN o.primary_target = ''
                        THEN $victim
                        ELSE o.primary_target
                    END
            """,
            operation_id=operation_id, 
            campaign_id=campaign_context.campaign_id,
            last_seen=campaign_context.last_seen,
            attacker=campaign_context.attacker_ip,
            victim=campaign_context.victim_ip
        )

def expire_stale_operations_db(timeout_seconds):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (o:Operation)
            WHERE
                o.status = 'ACTIVE'
                AND o.last_seen < datetime() - duration({seconds:$operation_timeout})
            SET
                o.status = 'INACTIVE',
                o.closed_at = datetime()
            RETURN o.operation_id AS operation_id
            """,
            operation_timeout=timeout_seconds
        )
        return [record["operation_id"] for record in result]
        

def get_active_operations():
    with driver.session() as session:
        result = session.run(
            """
            MATCH (o:Operation)
            WHERE o.status = 'ACTIVE'
            RETURN o.operation_id AS operation_id
            """
        )
        return [
            record["operation_id"]
            for record in result
        ]

def get_recent_inactive_operations(limit=5):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (o:Operation)
            WHERE o.status='INACTIVE'
            RETURN o.operation_id AS operation_id
            ORDER BY o.last_seen DESC
            LIMIT $limit
            """,
            limit=limit
        )

        return [
            record["operation_id"]
            for record in result
        ]

def store_prediction(campaign_id, predicted, confidence):
    with driver.session() as session:
        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})

            OPTIONAL MATCH (c)-[old:LIKELY_NEXT]->()
            DELETE old
            WITH *
            OPTIONAL MATCH (t:Technique {attack_id:$predicted})
            
            FOREACH (_ IN CASE WHEN t IS NULL THEN [] ELSE [1] END |
            
                MERGE (c)-[r:LIKELY_NEXT]->(t)
            
                SET
                    r.confidence=$confidence,
                    r.generated_at=datetime()
            )
            
            SET
                c.predicted_next=$predicted,
                c.prediction_confidence=$confidence,
                c.prediction_generated_at=datetime()

            MERGE (c)-[r:LIKELY_NEXT]->(t)

            SET
                r.confidence=$confidence,
                r.generated_at=datetime(),

                c.predicted_next=$predicted,
                c.prediction_confidence=$confidence,
                c.prediction_generated_at=datetime()
            """,
            campaign_id=campaign_id,
            predicted=predicted,
            confidence=confidence
        )

def get_prediction_distribution(previous_technique):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (:Technique {attack_id:$previous})
                  -[r:NEXT_TECHNIQUE]->
                  (t:Technique)

            RETURN
                t.attack_id AS technique,
                r.count AS count
            """,
            previous=previous_technique
        )

        rows = list(result)

        if not rows:
            return {}

        total = sum(
            row["count"]
            for row in rows
        )

        if total == 0:
            return {}

        return {
            row["technique"]:
            row["count"] / total
            for row in rows
        }             

def archive_campaign_db(campaign_id):
    with driver.session() as session:

        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})

            SET
                c.status='ARCHIVED',
                c.last_technique=NULL,
                c.predicted_next=NULL,
                c.prediction_confidence=0,
                c.archived_at=datetime()
            """,
            campaign_id=campaign_id
        )

def get_campaign_chain(campaign_id):

    with driver.session() as session:

        result = session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})
                  -[:HAS_EVENT]->
                  (e:AttackEvent)
                  -[:MATCHES]->
                  (t:Technique)

            RETURN
                t.attack_id AS attack_id

            ORDER BY e.first_seen
            """,
            campaign_id=campaign_id
        )

        return [
            r["attack_id"]
            for r in result
        ]
def find_recent_duplicate(fingerprint):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:AttackEvent {fingerprint:$fingerprint})
            WHERE e.last_seen >= datetime() - duration({seconds:$window})
            RETURN
                e.event_id AS event_id,
                e.campaign_id AS campaign_id,
                e.last_seen AS last_seen
            ORDER BY e.last_seen DESC
            LIMIT 1
            """,
            fingerprint=fingerprint,
            window=DEDUP_WINDOW
        )

        record = result.single()
        if record is None:
            return None
        return {
            "event_id": record["event_id"],
            "campaign_id": record["campaign_id"],
            "last_seen": record["last_seen"]
        }
def get_mitre_features(attack_id):
    query = """
    MATCH (t:Technique {attack_id:$attack_id})
    OPTIONAL MATCH (ta:ThreatActor)-[:USES]->(t)
    WITH
        t,
        count(DISTINCT ta) AS threat_actor_count
    OPTIONAL MATCH (m:Malware)-[:USES]->(t)
    WITH
        t,
        threat_actor_count,
        count(DISTINCT m) AS malware_count

    OPTIONAL MATCH (tool:Tool)-[:USES]->(t)

    WITH
        t,
        threat_actor_count,
        malware_count,
        count(DISTINCT tool) AS tool_count
    OPTIONAL MATCH (coa:CourseOfAction)-[:MITIGATES]->(t)
    WITH
        t,
        threat_actor_count,
        malware_count,
        tool_count,
        count(DISTINCT coa) AS mitigation_count
    OPTIONAL MATCH (child:Technique)-[:SUBTECHNIQUE_OF]->(t)
    WITH
        t,
        threat_actor_count,
        malware_count,
        tool_count,
        mitigation_count,
        count(DISTINCT child) AS subtechnique_count

    OPTIONAL MATCH (t)-[:SUBTECHNIQUE_OF]->(parent:Technique)
    RETURN
        t.attack_id                AS attack_id,
        t.name                     AS technique_name,
        coalesce(t.description,"") AS description,
        coalesce(t.platforms,[])   AS platforms,
        size(coalesce(t.platforms,[]))
            AS platform_count,
        coalesce(t.domains,[])
            AS domains,
        size(coalesce(t.domains,[]))
            AS domain_count,
        coalesce(t.kill_chain_phases,[])
            AS kill_chain_phases,
        size(coalesce(t.kill_chain_phases,[]))
            AS kill_chain_count,
        coalesce(t.is_subtechnique,false)
            AS is_subtechnique,
        coalesce(t.deprecated,false)
            AS deprecated,
        coalesce(t.revoked,false)
            AS revoked,
        coalesce(t.object_version,"")
            AS object_version,
        threat_actor_count,
        malware_count,
        tool_count,
        mitigation_count,
        subtechnique_count,
        parent.attack_id
            AS parent_technique
    """
    with driver.session() as session:
        record = session.run(
            query,
            attack_id=attack_id
        ).single()
        if record is None:
            return None
        return MitreFeatures(
            attack_id=record["attack_id"],
            technique_name=record["technique_name"],
            description=record["description"],
            platforms=record["platforms"],
            platform_count=record["platform_count"],
            domains=record["domains"],
            domain_count=record["domain_count"],
            kill_chain_phases=record["kill_chain_phases"],
            kill_chain_count=record["kill_chain_count"],
            is_subtechnique=record["is_subtechnique"],
            deprecated=record["deprecated"],
            revoked=record["revoked"],
            object_version=record["object_version"],
            threat_actor_count=record["threat_actor_count"],
            malware_count=record["malware_count"],
            tool_count=record["tool_count"],
            mitigation_count=record["mitigation_count"],
            subtechnique_count=record["subtechnique_count"],
            parent_technique=record["parent_technique"]
        )

def get_runtime_graph_features(campaign_id):
    query = """
    MATCH (c:Campaign {campaign_id:$campaign_id})
    OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)
    WITH
        c,
        count(DISTINCT e) AS attack_event_count
    OPTIONAL MATCH (c)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t:Technique)
    WITH
        c,
        attack_event_count,
        collect(DISTINCT t) AS techniques

    UNWIND techniques AS tech

    OPTIONAL MATCH (tech)<-[in_rel:NEXT_TECHNIQUE]-()

    WITH
        c,
        attack_event_count,
        techniques,
        count(DISTINCT in_rel) AS incoming_chain_count
    UNWIND techniques AS tech2
    OPTIONAL MATCH (tech2)-[out_rel:NEXT_TECHNIQUE]->()
    WITH
        c,
        attack_event_count,
        techniques,
        incoming_chain_count,
        count(DISTINCT out_rel) AS outgoing_chain_count
    RETURN
        coalesce(c.campaign_count,1)
            AS campaign_count,
        attack_event_count,
        size(techniques)
            AS graph_degree,
        incoming_chain_count,
        outgoing_chain_count,
        coalesce(c.prediction_frequency,0)
            AS prediction_frequency,
        coalesce(c.duplicate_frequency,0)
            AS duplicate_frequency
    """
    with driver.session() as session:
        record = session.run(
            query,
            campaign_id=campaign_id
        ).single()
        if record is None:
            return RuntimeGraphFeatures(
                campaign_count=1,
                attack_event_count=0,
                graph_degree=0,
                incoming_chain_count=0,
                outgoing_chain_count=0,
                prediction_frequency=0,
                duplicate_frequency=0,
            )
        return RuntimeGraphFeatures(
            campaign_count=record["campaign_count"],
            attack_event_count=record["attack_event_count"],
            graph_degree=record["graph_degree"],
            incoming_chain_count=record["incoming_chain_count"],
            outgoing_chain_count=record["outgoing_chain_count"],
            prediction_frequency=record["prediction_frequency"],
            duplicate_frequency=record["duplicate_frequency"]
        )

def get_threat_intel_features(attacker_ip):
    query = """
    MATCH (a:Attacker {ip:$attacker_ip})
    RETURN
        coalesce(a.vt_reputation,0.0)
            AS ip_reputation,
        coalesce(a.threat_actor_reputation,0.0)
            AS threat_actor_reputation,
        coalesce(a.malware_confidence,0.0)
            AS malware_confidence,
        coalesce(a.tool_confidence,0.0)
            AS tool_confidence,
        coalesce(a.misp_confidence,0.0)
            AS misp_confidence,
        coalesce(a.ioc_confidence,0.0)
            AS ioc_confidence
    """
    with driver.session() as session:
        record = session.run(
            query,
            attacker_ip=attacker_ip
        ).single()
        if record is None:
            return ThreatIntelFeatures(
                0,0,0,0,0,0
            )
        return ThreatIntelFeatures(
            ip_reputation=record["ip_reputation"],
            threat_actor_reputation=record["threat_actor_reputation"],
            malware_confidence=record["malware_confidence"],
            tool_confidence=record["tool_confidence"],
            misp_confidence=record["misp_confidence"],
            ioc_confidence=record["ioc_confidence"]
        )

def get_detection_confidence(event_id):
    query = """
    MATCH (e:AttackEvent {event_id:$event_id})
    RETURN
        coalesce(e.rule_level,0)
            AS wazuh_level,
        coalesce(e.suricata_score,0.0)
            AS suricata_score,
        coalesce(e.zeek_score,0.0)
            AS zeek_score,
        coalesce(e.sigma_score,0.0)
            AS sigma_score,
        coalesce(e.yara_score,0.0)
            AS yara_score
    """
    with driver.session() as session:
        record = session.run(
            query,
            event_id=event_id
        ).single()
        if record is None:
            return DetectionConfidence(
                0,
                0,
                0,
                0,
                0,
                0
            )
        wazuh = record["wazuh_level"]
        confidence = (
            wazuh +
            record["suricata_score"] +
            record["zeek_score"] +
            record["sigma_score"] +
            record["yara_score"]
        )
        confidence = min(confidence, 10)
        return DetectionConfidence(
            wazuh_level=wazuh,
            suricata_score=record["suricata_score"],
            zeek_score=record["zeek_score"],
            sigma_score=record["sigma_score"],
            yara_score=record["yara_score"],
            detection_confidence=confidence
        )
    
def get_runtime_features(attack_id):
    query = """
    MATCH (t:Technique {attack_id:$attack_id})
    OPTIONAL MATCH
        (c:Campaign)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t)
    WITH
        t,
        count(DISTINCT c) AS campaign_count
    OPTIONAL MATCH
        (:AttackEvent)-[:MATCHES]->(t)
    WITH
        t,
        campaign_count,
        count(*) AS event_count
    OPTIONAL MATCH
        (prev:Technique)-[r1:NEXT_TECHNIQUE]->(t)
    WITH
        t,
        campaign_count,
        event_count,
        count(r1) AS incoming_chain_count
    OPTIONAL MATCH
        (t)-[r2:NEXT_TECHNIQUE]->(next:Technique)
    WITH
        t,
        campaign_count,
        event_count,
        incoming_chain_count,
        count(r2) AS outgoing_chain_count
    RETURN
        campaign_count,
        event_count,
        incoming_chain_count,
        outgoing_chain_count,
        (
            incoming_chain_count +
            outgoing_chain_count
        ) AS graph_degree,
        coalesce(t.occurrences,0) AS duplicate_frequency,
        coalesce(t.total_tps,0) AS cumulative_tps
    """
    with driver.session() as session:
        result = session.run(
            query,
            attack_id=attack_id
        ).single()

        if result is None:
            return {
                "campaign_count":0,
                "event_count":0,
                "incoming_chain_count":0,
                "outgoing_chain_count":0,
                "graph_degree":0,
                "duplicate_frequency":0,
                "cumulative_tps":0
            }
        return dict(result)
def store_dynamic_risk(campaign_id, dynamic_risk):
    with driver.session() as session:
        previous = session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})
            RETURN c.dynamic_risk AS risk
            """,
            campaign_id=campaign_id
        ).single()
        previous_score = 0
        if previous and previous["risk"] is not None:
            previous_score = previous["risk"]
        if previous_score == 0:
            trend = "NEW"
        elif dynamic_risk.risk_score > previous_score:
            trend = "INCREASING"
        elif dynamic_risk.risk_score < previous_score:
            trend = "DECREASING"
        else:
            trend = "STABLE"
        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})
            SET
            c.dynamic_risk=$risk,
            c.risk_level=$level,
            c.risk_color=$color,
            c.risk_trend=$trend,
            c.risk_confidence=$confidence,
            c.risk_breakdown=$breakdown,
            c.risk_updated_at=datetime()
            """,
            campaign_id=campaign_id,
            risk=dynamic_risk.risk_score,
            level=dynamic_risk.risk_level,
            color=dynamic_risk.risk_color,
            breakdown=str(dynamic_risk.breakdown),
            trend=trend,
            confidence=dynamic_risk.confidence
        )  

def store_cti_confidence(campaign_id, cti):
    with driver.session() as session:
        session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})

            SET
                c.cti_score = $score,
                c.cti_level = $level,
                c.cti_publish = $publish,
                c.cti_breakdown = $breakdown,
                c.cti_updated_at = datetime()
            """,
            campaign_id=campaign_id,
            score=cti.score,
            level=cti.level,
            publish=cti.publish,
            breakdown=str(cti.breakdown)
        )

def get_campaign_context_data(campaign_id):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})
            OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)
            OPTIONAL MATCH (e)-[:MATCHES]->(t:Technique)
            RETURN
                c,
                collect(DISTINCT t.attack_id) AS techniques,
                collect(
                    DISTINCT {
                        attack_id: e.attack_id,
                        first_seen: e.first_seen
                    }
                ) AS events
            """,
            campaign_id=campaign_id
        )
        record = result.single()
        print("\n===== RAW CAMPAIGN CONTEXT =====")
        print(record["techniques"])
        print(record["events"])
        print("===============================")
        if record is None:
            return None
        techniques = [
            t for t in record["techniques"]
            if t is not None
        ]
        events = [
            e for e in record["events"]
            if e["attack_id"] is not None
        ]
        events.sort(
            key=lambda e: normalize_datetime(e["first_seen"])
                         or datetime.min.replace(tzinfo=timezone.utc)
        )
        
        attack_chain = [
            e["attack_id"]
            for e in events
        ]
        return {
            "campaign": dict(record["c"]),
            "techniques": techniques,
            "attack_chain": attack_chain
        }

def get_operation_context_data(operation_id):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
            RETURN o
            """,
            operation_id=operation_id
        )
        record = result.single()

        if record is None:
            return None
        operation = dict(record["o"])
        result = session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
                  -[:HAS_CAMPAIGN]->
                  (c:Campaign)

            RETURN c
            ORDER BY c.first_seen
            """,
            operation_id=operation_id
        )
        campaigns = [
            dict(record["c"])
            for record in result
        ]
        result = session.run(
            """
            MATCH (o:Operation {operation_id:$operation_id})
                  -[:HAS_CAMPAIGN]->
                  (c:Campaign)
                  -[:HAS_EVENT]->
                  (e:AttackEvent)
                  -[:MATCHES]->
                  (t:Technique)
        
            RETURN
                collect(DISTINCT t.attack_id) AS techniques,
                collect(
                    DISTINCT {
                        attack_id: e.attack_id,
                        first_seen: e.first_seen
                    }
                ) AS events
            """,
            operation_id=operation_id
        )
        
        record = result.single()
        
        techniques = [
            t for t in record["techniques"]
            if t is not None
        ]
        
        events = [
            e for e in record["events"]
            if e["attack_id"] is not None
        ]
        
        events.sort(
            key=lambda e: normalize_datetime(e["first_seen"])
                         or datetime.min.replace(tzinfo=timezone.utc)
        )
        
        attack_chain = [
            e["attack_id"]
            for e in events
        ]
        return {
            "operation": operation,
            "campaigns": campaigns,
            "techniques": techniques,
            "attack_chain": attack_chain
        }

def get_dynamic_risk(campaign_id):
    with driver.session() as session:
        record = session.run(
            """
            MATCH (c:Campaign {campaign_id:$campaign_id})
            RETURN
                c.dynamic_risk AS risk,
                c.risk_level AS level,
                c.risk_color AS color,
                c.risk_trend AS trend,
                c.risk_confidence AS confidence,
                c.risk_updated_at AS updated_at
            """,
            campaign_id=campaign_id
        ).single()
        if record is None:
            return None
        return dict(record)
        
def close():
    driver.close()