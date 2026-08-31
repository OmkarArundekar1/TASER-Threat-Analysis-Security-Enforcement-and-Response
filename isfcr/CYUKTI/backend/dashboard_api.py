from flask import Flask, jsonify, request
from flask_cors import CORS
from neo4j_client import driver
import prediction_engine
import recommendation_engine
import datetime
import json
import os
import logging

from neo4j.exceptions import Neo4jError, ServiceUnavailable

from campaign_context import CampaignContext
from investigation.loop import default_action_executor, default_model_predictor, run_investigation
from rag.mitre_retriever import mitre_retriever
from risk_scoring import TPS_CEILING, normalize_risk_score, risk_level_from_score, severity_from_tps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


def neo4j_datetime_to_iso(val):
    if val is None:
        return None
    if hasattr(val, 'iso_format'):
        return val.iso_format()
    return str(val)


def format_campaign_label(campaign_id, index=None):
    if index is not None:
        return f"Campaign #{index}"
    if campaign_id and len(campaign_id) > 24:
        return f"Campaign …{campaign_id[-8:]}"
    return campaign_id or "Campaign"


def format_node_label(node_type, props, campaign_index=None):
    if node_type == "Campaign":
        return format_campaign_label(props.get("campaign_id"), campaign_index)
    if node_type == "Technique":
        return props.get("name") or props.get("attack_id") or "Technique"
    if node_type == "Attacker":
        return props.get("ip") or "Attacker"
    if node_type == "Host":
        return props.get("ip") or "Host"
    if node_type == "AttackEvent":
        return props.get("technique") or props.get("attack_id") or "Event"
    return props.get("name") or props.get("attack_id") or props.get("ip") or node_type

@app.route('/api/health')
def health():
    try:
        driver.verify_connectivity()
        return jsonify({"status": "healthy", "neo4j": "connected", "timestamp": datetime.datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"status": "degraded", "neo4j": "disconnected", "timestamp": datetime.datetime.now().isoformat()}), 500

def get_time_where_clause(time_range, entity_alias, timestamp_field="last_seen"):
    if not time_range or time_range == 'all':
        return ""
    
    ranges = {
        '15m': 'datetime() - duration({minutes: 15})',
        '1h': 'datetime() - duration({hours: 1})',
        '24h': 'datetime() - duration({hours: 24})',
        '7d': 'datetime() - duration({days: 7})'
    }
    
    if time_range in ranges:
        return f"{entity_alias}.{timestamp_field} >= {ranges[time_range]}"
    return ""

@app.route('/api/overview')
def overview():
    time_range = request.args.get('time_range', 'all')
    time_filter = get_time_where_clause(time_range, "e")
    where_clause = f"WHERE {time_filter}" if time_filter else ""
    
    with driver.session() as session:
        result = session.run(f"""
            MATCH (e:AttackEvent) {where_clause} WITH count(e) as total_events
            OPTIONAL MATCH (c:Campaign) WITH total_events, count(c) as active_campaigns
            OPTIONAL MATCH (a:Attacker) WITH total_events, active_campaigns, count(a) as unique_attackers
            OPTIONAL MATCH (t:Technique) WITH total_events, active_campaigns, unique_attackers, count(t) as techniques_detected
            OPTIONAL MATCH ()-[r:NEXT_TECHNIQUE]->() WITH total_events, active_campaigns, unique_attackers, techniques_detected, count(r) as learned_transitions
            OPTIONAL MATCH (h:Host) WITH total_events, active_campaigns, unique_attackers, techniques_detected, learned_transitions, count(h) as total_hosts
            RETURN total_events, active_campaigns, unique_attackers, techniques_detected, learned_transitions, total_hosts
        """).single()
        
        return jsonify({
            "total_events": result["total_events"] or 0,
            "active_campaigns": result["active_campaigns"] or 0,
            "unique_attackers": result["unique_attackers"] or 0,
            "techniques_detected": result["techniques_detected"] or 0,
            "learned_transitions": result["learned_transitions"] or 0,
            "total_hosts": result["total_hosts"] or 0,
            "timestamp": datetime.datetime.now().isoformat()
        })

@app.route('/api/events')
def events():
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 50))
    campaign = request.args.get('campaign')
    attacker = request.args.get('attacker')
    technique = request.args.get('technique')
    time_range = request.args.get('time_range', 'all')
    skip = (page - 1) * page_size
    
    match_clause = "MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique), (a:Attacker)-[:LAUNCHED]->(c), (c)-[:TARGETS]->(h:Host)"
    where_clauses = []
    params = {}
    
    time_filter = get_time_where_clause(time_range, "e")
    if time_filter:
        where_clauses.append(time_filter)
        
    if campaign:
        where_clauses.append("c.campaign_id = $campaign")
        params["campaign"] = campaign
    if attacker:
        where_clauses.append("a.ip = $attacker")
        params["attacker"] = attacker
    if technique:
        where_clauses.append("t.attack_id = $technique")
        params["technique"] = technique
        
    where_str = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    with driver.session() as session:
        count_res = session.run(f"{match_clause}{where_str} RETURN count(e) as total", **params).single()
        total = count_res["total"] if count_res else 0
        
        query = f"""
            {match_clause}{where_str}
            RETURN elementId(e) as id, e.last_seen as timestamp, e.first_seen as first_seen,
                   e.occurrences as occurrences, e.tps as tps,
                   a.ip as attacker_ip, h.ip as victim_ip,
                   t.attack_id as technique_id, t.name as technique_name, e.stage as stage, 
                   c.campaign_id as campaign_id
            ORDER BY e.last_seen DESC SKIP $skip LIMIT $limit
        """
        params["skip"] = skip
        params["limit"] = page_size
        
        res = session.run(query, **params)
        events_list = []
        for record in res:
            tps = record.get("tps") or 0
            events_list.append({
                "id": record["id"],
                "timestamp": neo4j_datetime_to_iso(record["timestamp"]),
                "attacker_ip": record["attacker_ip"],
                "victim_ip": record["victim_ip"],
                "technique_id": record["technique_id"],
                "technique_name": record["technique_name"],
                "stage": record["stage"],
                "severity": severity_from_tps(tps),
                "occurrences": record.get("occurrences") or 1,
                "campaign_id": record["campaign_id"]
            })
            
        return jsonify({
            "events": events_list,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 0
        })

@app.route('/api/events/<event_id>')
def event_detail(event_id):
    with driver.session() as session:
        query = """
            MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique),
                  (a:Attacker)-[:LAUNCHED]->(c), (c)-[:TARGETS]->(h:Host)
            WHERE elementId(e) = $event_id
            RETURN e.last_seen as timestamp, e.first_seen as first_seen, e.occurrences as occurrences, e.tps as tps,
                   a.ip as attacker_ip, h.ip as victim_ip,
                   t.attack_id as technique_id, t.name as technique_name, e.stage as stage, 
                   c.campaign_id as campaign_id, e.investigation_payload as investigation_payload
        """
        record = session.run(query, event_id=event_id).single()
        if not record:
            return jsonify({"error": "Not found"}), 404
        
        pred = None
        likely = session.run("""
            MATCH (c:Campaign {campaign_id: $campaign_id})-[r:LIKELY_NEXT]->(pt:Technique)
            RETURN pt.attack_id AS predicted, r.confidence AS confidence, r.generated_at AS generated_at
        """, campaign_id=record["campaign_id"]).single()
        if likely and likely["predicted"]:
            pred = {
                "predicted": likely["predicted"],
                "confidence": likely["confidence"],
                "generated_at": neo4j_datetime_to_iso(likely["generated_at"])
            }
        else:
            engine_pred = prediction_engine.predict_next(record["technique_id"])
            if engine_pred:
                pred = engine_pred

        recs = recommendation_engine.get_recommendations(record["technique_id"])

        payload = {}
        if record.get("investigation_payload"):
            try:
                payload = json.loads(record["investigation_payload"])
            except Exception:
                pass

        detail = {
            "id": event_id,
            "timestamp": neo4j_datetime_to_iso(record["timestamp"]),
            "first_seen": neo4j_datetime_to_iso(record["first_seen"]),
            "occurrences": record["occurrences"],
            "attacker_ip": record["attacker_ip"],
            "victim_ip": record["victim_ip"],
            "technique_id": record["technique_id"],
            "technique_name": record["technique_name"],
            "stage": record["stage"],
            "severity": severity_from_tps(record.get("tps")),
            "campaign_id": record["campaign_id"],
            "mitre_tactic": record["stage"],
            "predicted_next_attack": pred["predicted"] if pred else None,
            "prediction_confidence": pred["confidence"] if pred else None,
            "prediction_generated_at": pred.get("generated_at") if pred else None,
            "recommendations": recs,
            "investigation_payload": payload,
            "raw_wazuh_event": payload
        }
        return jsonify(detail)

@app.route('/api/campaigns')
def campaigns():
    time_range = request.args.get('time_range', 'all')
    attacker_filter = request.args.get('attacker')
    time_filter = get_time_where_clause(time_range, "c")

    where_parts = []
    params = {}
    if time_filter:
        where_parts.append(time_filter)
    if attacker_filter:
        where_parts.append("a.ip = $attacker")
        params["attacker"] = attacker_filter
    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    
    with driver.session() as session:
        res = session.run(f"""
            MATCH (a:Attacker)-[:LAUNCHED]->(c:Campaign)
            OPTIONAL MATCH (c)-[:TARGETS]->(h:Host)
            OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique)
            OPTIONAL MATCH (c)-[ln:LIKELY_NEXT]->(pt:Technique)
            {where_clause}
            WITH c, a, h, collect(DISTINCT t) as techniques, count(DISTINCT e) as event_count,
                 head(collect(DISTINCT pt.attack_id)) AS predicted_technique,
                 head(collect(DISTINCT ln.confidence)) AS prediction_confidence
            RETURN c.campaign_id as campaign_id, a.ip as attacker_ip, h.ip as victim_ip,
                   c.first_seen as first_seen, c.last_seen as last_seen, event_count,
                   c.tps as raw_tps, c.last_technique as latest_technique,
                   predicted_technique, prediction_confidence
            ORDER BY c.last_seen DESC
        """, **params)
        campaigns_list = []
        for idx, r in enumerate(res, start=1):
            raw_tps = r["raw_tps"] or 0
            risk_score = normalize_risk_score(raw_tps)
            campaigns_list.append({
                "campaign_id": r["campaign_id"],
                "campaign_label": format_campaign_label(r["campaign_id"], idx),
                "attacker_ip": r["attacker_ip"],
                "victim_ip": r["victim_ip"] or "Unknown",
                "first_seen": neo4j_datetime_to_iso(r["first_seen"]),
                "last_seen": neo4j_datetime_to_iso(r["last_seen"]),
                "event_count": r["event_count"],
                "risk_score": risk_score,
                "raw_tps": raw_tps,
                "risk_level": risk_level_from_score(risk_score),
                "latest_technique": r["latest_technique"],
                "predicted_technique": r["predicted_technique"],
                "prediction_confidence": r["prediction_confidence"]
            })
        return jsonify({"campaigns": campaigns_list, "count": len(campaigns_list)})

@app.route('/api/correlation/campaigns/<campaign_id>')
def correlation_campaigns(campaign_id):
    with driver.session() as session:
        query = """
        MATCH (c1:Campaign {campaign_id: $campaign_id})
        MATCH (c2:Campaign)
        WHERE c1 <> c2

        // Techniques
        OPTIONAL MATCH (c1)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t1:Technique)
        WITH c1, c2, collect(DISTINCT t1) AS c1_techs
        OPTIONAL MATCH (c2)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t2:Technique)
        WITH c1, c2, c1_techs, collect(DISTINCT t2) AS c2_techs
        WITH c1, c2, c1_techs, c2_techs, [t IN c1_techs WHERE t IN c2_techs] AS shared_t
        WITH c1, c2, size(shared_t) AS shared_tech_count,
             size(c1_techs) + size(c2_techs) - size(shared_t) AS total_tech_count,
             [t IN shared_t | coalesce(t.name, t.attack_id)] AS shared_techniques,
             [t IN shared_t | coalesce(t.stage, 'Unknown')] AS shared_tactics_raw

        // Attacker
        OPTIONAL MATCH (a1:Attacker)-[:LAUNCHED]->(c1)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, collect(DISTINCT a1.ip) AS c1_a
        OPTIONAL MATCH (a2:Attacker)-[:LAUNCHED]->(c2)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, c1_a, collect(DISTINCT a2.ip) AS c2_a
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, [a IN c1_a WHERE a IN c2_a] AS shared_a

        // Host
        OPTIONAL MATCH (c1)-[:TARGETS]->(h1:Host)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, shared_a, collect(DISTINCT h1.ip) AS c1_h
        OPTIONAL MATCH (c2)-[:TARGETS]->(h2:Host)
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, shared_a, c1_h, collect(DISTINCT h2.ip) AS c2_h
        WITH c1, c2, shared_tech_count, total_tech_count, shared_techniques, shared_tactics_raw, shared_a, [h IN c1_h WHERE h IN c2_h] AS shared_h

        // Calculation
        WITH c2, shared_techniques, shared_tactics_raw, shared_a, shared_h,
             (CASE WHEN total_tech_count = 0 THEN 0.0 ELSE toFloat(shared_tech_count) / total_tech_count END * 60.0) +
             (CASE WHEN size(shared_a) > 0 THEN 20.0 ELSE 0.0 END) +
             (CASE WHEN size(shared_h) > 0 THEN 20.0 ELSE 0.0 END) AS similarity_score
        WHERE similarity_score > 0
        RETURN c2.campaign_id AS campaign_id,
               c2.tps AS raw_tps,
               round(similarity_score) AS similarity_score,
               shared_techniques,
               shared_tactics_raw,
               shared_a AS shared_attackers,
               shared_h AS shared_hosts
        ORDER BY similarity_score DESC LIMIT 10
        """
        logger.info(f"Executing Cypher Query: Correlation for campaign {campaign_id}\n{query}")
        res = session.run(query, campaign_id=campaign_id)
        similar_campaigns = []
        for idx, r in enumerate(res, start=1):
            tactics_set = list(set(r["shared_tactics_raw"]))
            similar_campaigns.append({
                "campaign_id": r["campaign_id"],
                "campaign_label": format_campaign_label(r["campaign_id"], None),
                "similarity_score": r["similarity_score"],
                "shared_techniques": r["shared_techniques"],
                "shared_tactics": tactics_set,
                "shared_attackers": r["shared_attackers"],
                "shared_hosts": r["shared_hosts"]
            })
        return jsonify({"campaign_id": campaign_id, "similar_campaigns": similar_campaigns})

@app.route('/api/attribution/actors/<campaign_id>')
def attribution_actors(campaign_id):
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
        
        OPTIONAL MATCH (ta)-[:USES]->(m:Malware)
        WITH c, ta, shared_tech_count, total_tech_count, shared_techniques, collect(DISTINCT m.name) AS ta_malware
        
        OPTIONAL MATCH (ta)-[:USES]->(tool:Tool)
        WITH c, ta, shared_tech_count, total_tech_count, shared_techniques, ta_malware, collect(DISTINCT tool.name) AS ta_tools
        
        WITH ta.name AS actor_name, 
             ta.description AS description,
             (toFloat(shared_tech_count) / max(total_tech_count, 1)) * 100 AS base_confidence,
             shared_tech_count,
             shared_techniques,
             ta_malware,
             ta_tools
             
        // Boost confidence slightly if they match many techniques absolutely
        WITH actor_name, description, 
             CASE WHEN base_confidence + (shared_tech_count * 2) > 99 THEN 99.0 
                  ELSE base_confidence + (shared_tech_count * 2) END AS confidence,
             shared_techniques, ta_malware, ta_tools
             
        RETURN actor_name, round(confidence) AS confidence, description,
               shared_techniques, ta_malware, ta_tools
        ORDER BY confidence DESC LIMIT 5
        """
        logger.info(f"Executing Cypher Query: Attribution for campaign {campaign_id}\n{query}")
        res = session.run(query, campaign_id=campaign_id)
        actors = []
        for r in res:
            actors.append({
                "actor_name": r["actor_name"],
                "confidence": r["confidence"],
                "description": r["description"],
                "shared_techniques": r["shared_techniques"],
                "malware": r["ta_malware"][:5], # limit list size for UI
                "tools": r["ta_tools"][:5]
            })
        return jsonify({"campaign_id": campaign_id, "attribution": actors})

@app.route('/api/risk/propagation/<campaign_id>')
def risk_propagation(campaign_id):
    with driver.session() as session:
        query = """
        MATCH (c:Campaign {campaign_id: $campaign_id})
        OPTIONAL MATCH (a:Attacker)-[:LAUNCHED]->(c)
        OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique)
        OPTIONAL MATCH (c)-[:TARGETS]->(h:Host)
        
        WITH c, a, h, t, e.tps AS e_tps
        
        WITH c,
             a.ip AS attacker_ip, 
             h.ip AS host_ip,
             collect(DISTINCT {id: coalesce(t.attack_id, 'Unknown'), name: coalesce(t.name, 'Unknown'), score: coalesce(e_tps, 0)}) AS techniques
             
        RETURN c.campaign_id AS campaign_id,
               c.tps AS campaign_tps,
               attacker_ip,
               host_ip,
               techniques
        """
        logger.info(f"Executing Cypher Query: Risk Propagation for campaign {campaign_id}\n{query}")
        res = session.run(query, campaign_id=campaign_id).single()
        if not res:
            return jsonify({"error": "Campaign not found"}), 404
            
        c_tps = res["campaign_tps"] or 0
        c_risk = normalize_risk_score(c_tps)
        
        tech_contributions = {}
        total_tech_score = 0
        for tech in res["techniques"]:
            t_id = tech["id"]
            if t_id == 'Unknown': continue
            tech_contributions[t_id] = tech_contributions.get(t_id, 0) + tech["score"]
            total_tech_score += tech["score"]
            
        tech_list = []
        for t_id, score in tech_contributions.items():
            name = next((t["name"] for t in res["techniques"] if t["id"] == t_id), t_id)
            percent = round((score / max(total_tech_score, 1)) * 100)
            tech_list.append({
                "node_id": t_id,
                "node_name": name,
                "node_type": "Technique",
                "risk_score": normalize_risk_score(score),
                "contribution_percent": percent
            })
            
        tech_list.sort(key=lambda x: x["contribution_percent"], reverse=True)
        
        nodes = []
        # Attacker node
        if res["attacker_ip"]:
            nodes.append({
                "node_id": res["attacker_ip"],
                "node_name": res["attacker_ip"],
                "node_type": "Attacker",
                "risk_score": c_risk,
                "contribution_percent": 100
            })
        
        # Campaign node
        nodes.append({
            "node_id": res["campaign_id"],
            "node_name": format_campaign_label(res["campaign_id"], None),
            "node_type": "Campaign",
            "risk_score": c_risk,
            "contribution_percent": 100
        })
        
        # Add top 5 techniques
        nodes.extend(tech_list[:5])
        
        # Host node
        if res["host_ip"]:
            nodes.append({
                "node_id": res["host_ip"],
                "node_name": res["host_ip"],
                "node_type": "Host",
                "risk_score": c_risk, # In this simplified model, host inherits full campaign risk
                "contribution_percent": 100
            })
            
        return jsonify({"campaign_id": campaign_id, "propagation": nodes})

@app.route('/api/campaigns/<campaign_id>/timeline')
def campaign_timeline(campaign_id):
    with driver.session() as session:
        query = """
        MATCH (c:Campaign {campaign_id: $campaign_id})-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique)
        RETURN elementId(e) as id, e.first_seen as first_seen, e.last_seen as last_seen, e.occurrences as occurrences,
               t.attack_id as technique_id, t.name as technique_name, e.stage as stage
        ORDER BY e.first_seen ASC
        """
        res = session.run(query, campaign_id=campaign_id)
        events = []
        for r in res:
            events.append({
                "id": r["id"],
                "technique_id": r["technique_id"],
                "technique_name": r["technique_name"],
                "stage": r["stage"],
                "first_seen": r["first_seen"].iso_format() if hasattr(r["first_seen"], 'iso_format') else str(r["first_seen"]),
                "last_seen": r["last_seen"].iso_format() if hasattr(r["last_seen"], 'iso_format') else str(r["last_seen"]),
                "occurrences": r["occurrences"] or 1
            })
        
        return jsonify({"events": events, "campaign_id": campaign_id})

@app.route('/api/attackers')
def attackers():
    with driver.session() as session:
        res = session.run("""
            MATCH (a:Attacker)-[:LAUNCHED]->(c:Campaign)
            OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique)
            OPTIONAL MATCH (c)-[ln:LIKELY_NEXT]->(pt:Technique)
            WITH a, count(DISTINCT c) AS campaign_count,
                 count(DISTINCT e) AS event_count,
                 sum(coalesce(c.tps, 0)) AS raw_tps,
                 min(c.first_seen) AS first_seen,
                 max(c.last_seen) AS last_seen,
                 collect(DISTINCT t.attack_id) AS techniques,
                 head(collect(pt.attack_id)) AS predicted_technique
            RETURN a.ip AS attacker_ip, campaign_count, event_count, raw_tps,
                   first_seen, last_seen, techniques, predicted_technique
            ORDER BY raw_tps DESC, event_count DESC
        """)
        attackers_list = []
        for r in res:
            raw_tps = r["raw_tps"] or 0
            risk_score = normalize_risk_score(raw_tps)
            attackers_list.append({
                "attacker_ip": r["attacker_ip"],
                "campaign_count": r["campaign_count"],
                "event_count": r["event_count"],
                "risk_score": risk_score,
                "raw_tps": raw_tps,
                "risk_level": risk_level_from_score(risk_score),
                "first_seen": neo4j_datetime_to_iso(r["first_seen"]),
                "last_seen": neo4j_datetime_to_iso(r["last_seen"]),
                "techniques_observed": [t for t in (r["techniques"] or []) if t],
                "predicted_technique": r["predicted_technique"]
            })
        return jsonify({"attackers": attackers_list, "count": len(attackers_list)})

@app.route('/api/graph/expand')
def expand_graph():
    node_id = request.args.get('node_id')
    depth = int(request.args.get('depth', 1))
    
    if not node_id:
        return jsonify({"error": "Missing node_id parameter"}), 400
        
    with driver.session() as session:
        start = session.run("""
            MATCH (start) WHERE elementId(start) = $node_id
            RETURN labels(start)[0] AS node_type, start.attack_id AS attack_id
        """, node_id=node_id).single()

        if not start:
            return jsonify({"nodes": [], "links": []})

        if start["node_type"] == "Technique":
            query = f"""
            MATCH (t:Technique) WHERE elementId(t) = $node_id
            MATCH path = (t)-[*1..{depth}]-(m)
            WHERE any(label IN labels(m) WHERE label IN ['ThreatActor', 'Malware', 'Tool', 'Mitigation', 'Tactic', 'Technique'])
            UNWIND nodes(path) AS n
            UNWIND relationships(path) AS r
            WITH collect(DISTINCT n) AS all_nodes, collect(DISTINCT r) AS all_rels
            RETURN all_nodes, all_rels
            """
        else:
            query = f"""
            MATCH (start) WHERE elementId(start) = $node_id
            MATCH path = (start)-[*1..{depth}]-(m)
            WHERE any(label IN labels(m) WHERE label IN ['ThreatActor', 'Malware', 'Tool', 'Mitigation', 'Tactic', 'Campaign', 'Technique', 'Attacker', 'Host'])
            UNWIND nodes(path) AS n
            UNWIND relationships(path) AS r
            WITH collect(DISTINCT n) AS all_nodes, collect(DISTINCT r) AS all_rels
            RETURN all_nodes, all_rels
            """
        res = session.run(query, node_id=node_id).single()
        if not res or not res["all_nodes"]:
            return jsonify({"nodes": [], "links": []})
            
        nodes = []
        for n in res["all_nodes"]:
            n_id = n.element_id
            n_type = list(n.labels)[0] if n.labels else 'Unknown'
            n_props = dict(n)
            nodes.append({
                "id": n_id,
                "type": n_type,
                "label": format_node_label(n_type, n_props),
                "attack_id": n_props.get("attack_id"),
                "campaign_id": n_props.get("campaign_id"),
                "full_label": n_props.get("campaign_id") or n_props.get("name") or n_props.get("attack_id")
            })
            
        links = []
        for r in res["all_rels"]:
            links.append({
                "source": r.start_node.element_id,
                "target": r.end_node.element_id,
                "label": r.type
            })
                
        return jsonify({"nodes": nodes, "links": links})

@app.route('/api/graph/paths')
def graph_paths():
    campaign_id = request.args.get('campaign_id')
    timeframe = request.args.get('timeframe')
    
    with driver.session() as session:
        time_filter = ""
        if timeframe == '24h':
            time_filter = "AND e.first_seen >= datetime() - duration('PT24H')"
        elif timeframe == '7d':
            time_filter = "AND e.first_seen >= datetime() - duration('P7D')"
        elif timeframe == '30d':
            time_filter = "AND e.first_seen >= datetime() - duration('P30D')"
            
        campaign_filter = ""
        if campaign_id:
            campaign_filter = "AND c.campaign_id = $campaign_id"

        # Note: we need to handle LIKELY_NEXT relationships properly without failing if there are no AttackEvents
        query = f"""
        // Get all techniques involved
        OPTIONAL MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique)
        WHERE 1=1 {time_filter} {campaign_filter}
        WITH collect(DISTINCT t) AS event_techs
        
        OPTIONAL MATCH (c:Campaign)-[:LIKELY_NEXT]->(t:Technique)
        WHERE 1=1 {campaign_filter}
        WITH event_techs, collect(DISTINCT t) AS likely_techs
        
        WITH [x IN (event_techs + likely_techs) WHERE x IS NOT NULL] AS all_techs
        UNWIND all_techs AS t
        WITH DISTINCT t
        
        // Calculate global frequency and max TPS for node sizing
        OPTIONAL MATCH (c2:Campaign)-[:HAS_EVENT]->(e2:AttackEvent)-[:MATCHES]->(t)
        WHERE 1=1 {time_filter} {campaign_filter}
        WITH t, count(DISTINCT c2) AS campaign_count, sum(e2.occurrences) AS frequency, max(e2.tps) AS max_tps
        
        // Return nodes
        RETURN elementId(t) AS id, 'Technique' AS type, t.attack_id AS attack_id, coalesce(t.name, t.attack_id) AS name,
               t.description AS description, t.stage AS stage,
               frequency, campaign_count, max_tps
        """
        
        logger.info(f"Executing Cypher Query: Path Graph Nodes\n{query}")
        nodes_res = session.run(query, campaign_id=campaign_id)
        
        nodes = []
        node_ids = set()
        
        for r in nodes_res:
            n_id = r["id"]
            node_ids.add(n_id)
            nodes.append({
                "id": n_id,
                "type": "Technique",
                "label": r["name"],
                "attack_id": r["attack_id"],
                "description": r["description"],
                "stage": r["stage"],
                "frequency": r["frequency"] or 0,
                "campaign_count": r["campaign_count"] or 0,
                "max_tps": r["max_tps"] or 0
            })
            
        # Get Edges
        links_query = f"""
        // NEXT_TECHNIQUE edges
        OPTIONAL MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t1:Technique)
        WHERE 1=1 {time_filter} {campaign_filter}
        WITH collect(DISTINCT t1) AS event_techs
        
        OPTIONAL MATCH (c:Campaign)-[:LIKELY_NEXT]->(t2:Technique)
        WHERE 1=1 {campaign_filter}
        WITH event_techs, collect(DISTINCT t2) AS likely_techs
        
        WITH [x IN (event_techs + likely_techs) WHERE x IS NOT NULL] AS all_techs
        
        OPTIONAL MATCH (t1:Technique)-[r:NEXT_TECHNIQUE]->(t2:Technique)
        WHERE t1 IN all_techs AND t2 IN all_techs
        WITH all_techs, collect(DISTINCT {{source: elementId(t1), target: elementId(t2), label: 'NEXT_TECHNIQUE', count: coalesce(r.count, 1), confidence: null}}) AS next_links
        
        // LIKELY_NEXT edges
        OPTIONAL MATCH (c:Campaign)-[r:LIKELY_NEXT]->(t:Technique)
        WHERE 1=1 {campaign_filter}
        OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t_src:Technique)
        WITH next_links, collect(DISTINCT {{source: elementId(t_src), target: elementId(t), label: 'LIKELY_NEXT', count: null, confidence: coalesce(r.confidence, 0)}}) AS likely_links
        
        RETURN next_links, likely_links
        """
        
        logger.info(f"Executing Cypher Query: Path Graph Edges\n{links_query}")
        links_res = session.run(links_query, campaign_id=campaign_id).single()
        
        links = []
        if links_res:
            next_links = links_res["next_links"] or []
            likely_links = links_res["likely_links"] or []
            
            for l in next_links:
                if l["source"] and l["target"] and l["source"] in node_ids and l["target"] in node_ids:
                    links.append(l)
            
            # Since LIKELY_NEXT connects campaign to technique, but the path explorer is technique to technique
            # We connect the most recent technique of the campaign to the likely next technique!
            # The query above did exactly that: (c)->(e)->(t_src) and (c)->[LIKELY]->(t)
            for l in likely_links:
                if l["source"] and l["target"] and l["source"] in node_ids and l["target"] in node_ids:
                    # Filter out self loops for likely_next just in case
                    if l["source"] != l["target"]:
                        links.append(l)
                        
        # Get mitigations for nodes if APOC or similar isn't strictly required
        for node in nodes:
            mitig_query = """
            MATCH (t:Technique {attack_id: $attack_id})-[:MITIGATED_BY]->(m:Mitigation)
            RETURN m.name AS name, m.description AS description LIMIT 3
            """
            try:
                m_res = session.run(mitig_query, attack_id=node["attack_id"])
                node["mitigations"] = [{"name": mr["name"], "description": mr["description"]} for mr in m_res]
            except:
                node["mitigations"] = []
                
        return jsonify({"nodes": nodes, "links": links})

import re

@app.route('/api/query', methods=['POST'])
def execute_query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query parameter"}), 400
        
    query = data['query'].strip()
    
    # 1. Validation: Block forbidden mutations
    forbidden_keywords = [
        r'\bCREATE\b', r'\bMERGE\b', r'\bDELETE\b', r'\bDETACH\b',
        r'\bSET\b', r'\bREMOVE\b', r'\bDROP\b', r'\bLOAD CSV\b'
    ]
    
    upper_query = query.upper()
    for pattern in forbidden_keywords:
        if re.search(pattern, upper_query):
            return jsonify({
                "error": "Query validation failed: Mutating commands (CREATE, MERGE, DELETE, etc.) are strictly forbidden in the console."
            }), 403
            
    # 2. Execution via execute_read (read_transaction was removed in neo4j driver 5.x+)
    def _execute_read(tx, cypher):
        result = tx.run(cypher)
        # Parse records for table view
        records = []
        columns = result.keys()
        
        # Build graph data if paths or nodes are returned
        nodes = []
        node_ids = set()
        links = []
        
        def _process_node(n):
            if n.element_id not in node_ids:
                node_ids.add(n.element_id)
                props = dict(n)
                nodes.append({
                    "id": n.element_id,
                    "type": list(n.labels)[0] if n.labels else "Unknown",
                    "label": props.get("name") or props.get("attack_id") or props.get("ip") or props.get("campaign_id") or n.element_id,
                    "props": props
                })
                
        def _process_rel(r):
            links.append({
                "source": r.start_node.element_id,
                "target": r.end_node.element_id,
                "label": r.type,
                "props": dict(r)
            })

        for record in result:
            row_dict = {}
            for col in columns:
                val = record[col]
                row_dict[col] = str(val) if not isinstance(val, (int, float, bool, type(None))) else val
                
                # Check for Graph Objects (Nodes, Relationships, Paths)
                if hasattr(val, 'labels'): # Node
                    _process_node(val)
                elif hasattr(val, 'start_node'): # Relationship
                    _process_rel(val)
                    _process_node(val.start_node)
                    _process_node(val.end_node)
                elif hasattr(val, 'nodes') and hasattr(val, 'relationships'): # Path
                    for n in val.nodes: _process_node(n)
                    for r in val.relationships: _process_rel(r)
                    
            records.append(row_dict)
            
        return {
            "columns": columns,
            "records": records,
            "count": len(records),
            "graph": {
                "nodes": nodes,
                "links": links
            }
        }
        
    try:
        with driver.session() as session:
            payload = session.execute_read(_execute_read, query)
            return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def _build_graph_response(session, layer, campaign_filter=None, attacker_filter=None):
    campaign_clause = ""
    attacker_clause = ""
    params = {}
    if campaign_filter:
        campaign_clause = "AND c.campaign_id = $campaign_id"
        params["campaign_id"] = campaign_filter
    if attacker_filter:
        attacker_clause = "AND a.ip = $attacker_ip"
        params["attacker_ip"] = attacker_filter
    combined_filter = f"{campaign_clause} {attacker_clause}"

    campaign_index_map = {}
    idx_res = session.run("MATCH (c:Campaign) RETURN c.campaign_id AS id ORDER BY c.first_seen ASC")
    for i, row in enumerate(idx_res, start=1):
        campaign_index_map[row["id"]] = i

    node_list_query = ""
    links_queries = []

    if layer == 'investigation':
        node_list_query = f"""
        MATCH (c:Campaign)
        WHERE true {campaign_clause.replace("AND a.ip", "AND a.ip")}
        MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique)
        OPTIONAL MATCH (t)-[:NEXT_TECHNIQUE]->(n_t:Technique)
        OPTIONAL MATCH (c)-[:LIKELY_NEXT]->(l_t:Technique)
        WITH collect(DISTINCT c) + collect(DISTINCT e) + collect(DISTINCT t) + collect(DISTINCT n_t) + collect(DISTINCT l_t) AS node_list
        UNWIND node_list AS n WITH DISTINCT n WHERE n IS NOT NULL
        RETURN elementId(n) AS id, labels(n)[0] AS type, properties(n) AS props
        """
        links_queries.append(f"""
        MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique)
        WHERE true {campaign_clause}
        RETURN elementId(c) AS s, elementId(e) AS t, 'HAS_EVENT' AS label, null AS confidence, null AS count, null AS props
        UNION
        MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique)
        WHERE true {campaign_clause}
        RETURN elementId(e) AS s, elementId(t) AS t, 'MATCHES' AS label, null AS confidence, null AS count, null AS props
        """)
        # NEXT_TECHNIQUE and LIKELY_NEXT are added universally below if nodes exist
        
    elif layer == 'threat_intel':
        node_list_query = f"""
        MATCH (c:Campaign)
        WHERE true {campaign_clause}
        OPTIONAL MATCH (c)-[:SIMILAR_TO]->(c2:Campaign)
        OPTIONAL MATCH (c)-[:RESEMBLES]->(ta:ThreatActor)
        OPTIONAL MATCH (ta)-[:USES]->(t:Technique)
        WITH collect(DISTINCT c) + collect(DISTINCT c2) + collect(DISTINCT ta) + collect(DISTINCT t) AS node_list
        UNWIND node_list AS n WITH DISTINCT n WHERE n IS NOT NULL
        RETURN elementId(n) AS id, labels(n)[0] AS type, properties(n) AS props
        """
        links_queries.append(f"""
        MATCH (c:Campaign)-[r:SIMILAR_TO]->(c2:Campaign)
        WHERE true {campaign_clause}
        RETURN elementId(c) AS s, elementId(c2) AS t, 'SIMILAR_TO' AS label, r.score AS confidence, null AS count, r AS props
        UNION
        MATCH (c:Campaign)-[r:RESEMBLES]->(ta:ThreatActor)
        WHERE true {campaign_clause}
        RETURN elementId(c) AS s, elementId(ta) AS t, 'RESEMBLES' AS label, r.confidence AS confidence, null AS count, r AS props
        UNION
        MATCH (c:Campaign)-[:RESEMBLES]->(ta:ThreatActor)-[r:USES]->(t:Technique)
        WHERE true {campaign_clause}
        RETURN elementId(ta) AS s, elementId(t) AS t, 'USES' AS label, null AS confidence, null AS count, null AS props
        """)

    else: # default campaign layer
        node_list_query = f"""
        MATCH (a:Attacker)-[:LAUNCHED]->(c:Campaign)
        WHERE true {combined_filter}
        OPTIONAL MATCH (c)-[:TARGETS]->(h:Host)
        OPTIONAL MATCH (c)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t:Technique)
        WITH collect(DISTINCT a) + collect(DISTINCT c) + collect(DISTINCT h) + collect(DISTINCT t) AS node_list
        UNWIND node_list AS n WITH DISTINCT n WHERE n IS NOT NULL
        RETURN elementId(n) AS id, labels(n)[0] AS type, properties(n) AS props
        """
        links_queries.append(f"""
        MATCH (a:Attacker)-[:LAUNCHED]->(c:Campaign)
        WHERE true {combined_filter}
        RETURN elementId(a) AS s, elementId(c) AS t, 'LAUNCHED' AS label, null AS confidence, null AS count, null AS props
        UNION
        MATCH (a:Attacker)-[:LAUNCHED]->(c:Campaign)-[:TARGETS]->(h:Host)
        WHERE true {combined_filter}
        RETURN elementId(c) AS s, elementId(h) AS t, 'TARGETS' AS label, null AS confidence, null AS count, null AS props
        UNION
        MATCH (a:Attacker)-[:LAUNCHED]->(c:Campaign)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t:Technique)
        WHERE true {combined_filter}
        RETURN elementId(c) AS s, elementId(t) AS t, 'USES_TECHNIQUE' AS label, null AS confidence, null AS count, null AS props
        """)

    nodes = []
    node_ids = set()
    for r in session.run(node_list_query, **params):
        props = dict(r["props"] or {})
        n_type = r["type"]
        campaign_idx = campaign_index_map.get(props.get("campaign_id")) if n_type == "Campaign" else None
        nodes.append({
            "id": r["id"],
            "type": n_type,
            "label": format_node_label(n_type, props, campaign_idx),
            "full_label": props.get("campaign_id") or props.get("name") or props.get("attack_id") or props.get("ip"),
            "attack_id": props.get("attack_id"),
            "campaign_id": props.get("campaign_id"),
            "tps": props.get("tps"),
            "description": props.get("description")
        })
        node_ids.add(r["id"])

    links = []
    
    # Run primary layer queries
    for q in links_queries:
        for r in session.run(q, **params):
            if r["s"] in node_ids and r["t"] in node_ids:
                links.append({
                    "source": r["s"],
                    "target": r["t"],
                    "label": r["label"],
                    "confidence": r["confidence"],
                    "count": r["count"],
                    "edge_props": dict(r["props"] or {})
                })

    # Add NEXT_TECHNIQUE and LIKELY_NEXT for Investigation layer
    if layer == 'investigation':
        nt_query = """
        MATCH (t1:Technique)-[r:NEXT_TECHNIQUE]->(t2:Technique)
        RETURN elementId(t1) AS s, elementId(t2) AS t, 'NEXT_TECHNIQUE' AS label, null AS confidence, r.count AS count
        """
        for r in session.run(nt_query):
            if r["s"] in node_ids and r["t"] in node_ids:
                links.append({
                    "source": r["s"],
                    "target": r["t"],
                    "label": "NEXT_TECHNIQUE",
                    "confidence": r["confidence"],
                    "count": r["count"],
                    "edge_props": {}
                })
                
        likely_query = f"""
        MATCH (c:Campaign)-[r:LIKELY_NEXT]->(t:Technique)
        WHERE true {campaign_clause}
        RETURN elementId(c) AS s, elementId(t) AS t, 'LIKELY_NEXT' AS label, r.confidence AS confidence, null AS count, null AS props
        """
        for r in session.run(likely_query, **params):
            if r["s"] in node_ids and r["t"] in node_ids:
                links.append({
                    "source": r["s"],
                    "target": r["t"],
                    "label": r["label"],
                    "confidence": r["confidence"],
                    "count": r["count"],
                    "edge_props": dict(r["props"] or {})
                })

    return {"nodes": nodes, "links": links}

@app.route('/api/analytics/paths')
def analytics_paths():
    with driver.session() as session:
        # Get actual observed paths from campaigns
        query = """
        MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique)
        WITH c, e, t ORDER BY e.first_seen ASC
        WITH c, collect(t.attack_id) AS attack_path, collect(coalesce(t.name, t.attack_id)) as attack_path_names
        WHERE size(attack_path) > 1
        RETURN attack_path, 
               attack_path_names,
               count(c) AS frequency, 
               max(c.tps) AS max_raw_tps,
               collect(c.campaign_id)[0..3] AS example_campaigns
        ORDER BY frequency DESC, max_raw_tps DESC
        LIMIT 20
        """
        logger.info(f"Executing Cypher Query: Analytics Paths\n{query}")
        res = session.run(query)
        paths = []
        for r in res:
            raw_tps = r["max_raw_tps"] or 0
            risk_score = normalize_risk_score(raw_tps)
            paths.append({
                "path": r["attack_path"],
                "path_names": r["attack_path_names"],
                "frequency": r["frequency"],
                "risk_score": risk_score,
                "risk_level": risk_level_from_score(risk_score),
                "example_campaigns": r["example_campaigns"]
            })
            
        return jsonify({"paths": paths})

@app.route('/api/attack-chain')
def attack_chain():
    campaign = request.args.get('campaign')
    with driver.session() as session:
        if campaign:
            result = session.run("""
                MATCH (c:Campaign {campaign_id: $campaign_id})-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique)
                RETURN t.attack_id AS technique_id, t.name AS name, e.stage AS stage,
                       e.occurrences AS detection_count, e.tps AS tps, e.first_seen AS first_seen
                ORDER BY e.first_seen ASC
            """, campaign_id=campaign)
            chain = []
            for r in result:
                chain.append({
                    "technique_id": r["technique_id"],
                    "name": r["name"],
                    "stage": r["stage"],
                    "detection_count": r["detection_count"],
                    "tps": r["tps"],
                    "first_seen": neo4j_datetime_to_iso(r["first_seen"])
                })
            return jsonify({"campaign_id": campaign, "chain": chain})

        result = session.run("""
            MATCH (t1:Technique)-[r:NEXT_TECHNIQUE]->(t2:Technique)
            RETURN t1.attack_id AS from_id, t1.name AS from_name,
                   t2.attack_id AS to_id, t2.name AS to_name,
                   r.count AS count
            ORDER BY r.count DESC LIMIT 50
        """)
        transitions = []
        total_count = 0
        for r in result:
            c = r["count"] or 0
            total_count += c
            transitions.append({
                "from": r["from_id"], "from_name": r["from_name"], "from_stage": "Unknown",
                "to": r["to_id"], "to_name": r["to_name"], "to_stage": "Unknown",
                "count": c, "confidence": 0.0
            })
        for t in transitions:
            t["confidence"] = round(t["count"] / max(total_count, 1), 4)
        return jsonify({"transitions": transitions, "chain": []})

@app.route('/api/predictions')
def predictions():
    with driver.session() as session:
        likely_res = session.run("""
            MATCH (c:Campaign)-[r:LIKELY_NEXT]->(pt:Technique)
            OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(ct:Technique)
            WITH c, r, pt, collect(DISTINCT ct)[-1] AS current
            RETURN c.campaign_id AS campaign_id,
                   current.attack_id AS current_technique,
                   current.name AS current_name,
                   current.stage AS stage,
                   pt.attack_id AS predicted_technique,
                   pt.name AS predicted_name,
                   r.confidence AS confidence,
                   r.generated_at AS generated_at
            ORDER BY r.confidence DESC
        """)
        preds = []
        seen = set()
        for r in likely_res:
            conf = r["confidence"] or 0
            preds.append({
                "campaign_id": r["campaign_id"],
                "current_technique": r["current_technique"] or r["predicted_technique"],
                "predicted_technique": r["predicted_technique"],
                "confidence": conf,
                "risk_level": risk_level_from_score(conf),
                "stage": r["stage"],
                "generated_at": neo4j_datetime_to_iso(r["generated_at"]) or datetime.datetime.now().isoformat(),
                "source": "LIKELY_NEXT"
            })
            seen.add(r["campaign_id"])

        fallback = session.run("""
            MATCH (c:Campaign)-[:HAS_EVENT]->(e:AttackEvent)-[:MATCHES]->(t:Technique)
            WITH c, t ORDER BY e.last_seen ASC
            WITH c, collect(t)[-1] AS latest
            WHERE NOT c.campaign_id IN $seen
            RETURN c.campaign_id AS campaign_id, latest.attack_id AS technique_id, latest.stage AS stage
        """, seen=list(seen))
        for r in fallback:
            pred = prediction_engine.predict_next(r["technique_id"])
            if pred:
                conf = pred.get("confidence", 0)
                preds.append({
                    "campaign_id": r["campaign_id"],
                    "current_technique": r["technique_id"],
                    "predicted_technique": pred.get("predicted", "UNKNOWN"),
                    "confidence": conf,
                    "risk_level": risk_level_from_score(conf),
                    "stage": r["stage"],
                    "generated_at": datetime.datetime.now().isoformat(),
                    "source": "NEXT_TECHNIQUE"
                })

        return jsonify({"predictions": preds, "count": len(preds)})

@app.route('/api/recommendations')
def recommendations():
    tech = request.args.get('technique')
    if tech:
        recs = recommendation_engine.get_recommendations(tech)
        return jsonify({"technique": tech, "recommendations": recs})
    else:
        with driver.session() as session:
            res = session.run("MATCH (t:Technique) RETURN t.attack_id as t_id LIMIT 10")
            all_recs = []
            for r in res:
                mit = recommendation_engine.get_recommendations(r["t_id"])
                if mit:
                    all_recs.append({"technique_id": r["t_id"], "recommendations": mit})
            return jsonify({"recommendations": all_recs})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict the next technique given a current technique ID."""
    data = request.get_json()
    if not data or 'current_technique' not in data:
        return jsonify({"error": "Missing current_technique parameter"}), 400
    technique = data['current_technique']
    result = prediction_engine.predict_next(technique)
    if result:
        return jsonify(result)
    return jsonify({"current": technique, "predicted": None, "confidence": 0})

@app.route('/api/graph')
def graph():
    view_mode = request.args.get('view_mode', 'campaign')
    campaign_filter = request.args.get('campaign')
    attacker_filter = request.args.get('attacker')
    with driver.session() as session:
        return jsonify(_build_graph_response(session, view_mode, campaign_filter, attacker_filter))


def _load_campaign_context(session, campaign_id):
    """Build a CampaignContext from Neo4j for the evidence/investigation layer.

    Deliberately re-queries Neo4j directly rather than using
    campaign_manager.get_context_by_campaign_id(), which only checks that
    manager's in-memory cache — populated by the live ingestion pipeline,
    not by this API process, so it would be empty here in the normal
    deployment topology (dashboard API and listener as separate processes).
    """
    record = session.run(
        """
        MATCH (c:Campaign {campaign_id: $campaign_id})
        OPTIONAL MATCH (a:Attacker)-[:LAUNCHED]->(c)
        OPTIONAL MATCH (c)-[:TARGETS]->(h:Host)
        OPTIONAL MATCH (c)-[:HAS_EVENT]->(:AttackEvent)-[:MATCHES]->(t:Technique)
        OPTIONAL MATCH (c)-[pred:LIKELY_NEXT]->(pt:Technique)
        WITH c, a, h, collect(DISTINCT t.attack_id) AS techniques,
             pt.attack_id AS predicted_next, pred.confidence AS prediction_confidence
        RETURN c, a.ip AS attacker_ip, h.ip AS victim_ip, techniques,
               predicted_next, prediction_confidence
        LIMIT 1
        """,
        campaign_id=campaign_id,
    ).single()

    if record is None or record["c"] is None:
        return None

    c = record["c"]
    context = CampaignContext(
        campaign_id=campaign_id,
        attacker_ip=record["attacker_ip"] or "",
        victim_ip=record["victim_ip"] or "",
        risk_score=c.get("risk_score", 0.0) or 0.0,
        last_technique=c.get("last_technique"),
        first_seen=c.get("first_seen").to_native() if c.get("first_seen") else None,
        last_seen=c.get("last_seen").to_native() if c.get("last_seen") else None,
        predicted_next=record["predicted_next"],
        prediction_confidence=record["prediction_confidence"] or 0.0,
    )
    context.techniques = set(record["techniques"] or [])
    context.attack_chain = list(context.techniques)
    return context


def _try_load_campaign_context(campaign_id):
    """(context, error_response) — error_response is None on success.

    Isolates the one Neo4j call every new evidence-aware endpoint below
    needs, so a database outage surfaces as a clean 503 JSON error
    instead of an uncaught driver exception reaching Flask's default
    error handler.
    """
    try:
        with driver.session() as session:
            return _load_campaign_context(session, campaign_id), None
    except (ServiceUnavailable, Neo4jError) as e:
        logger.error("Neo4j unavailable while loading campaign %s: %s", campaign_id, e)
        return None, (jsonify({"error": f"Database unavailable: {e}"}), 503)


@app.route('/api/investigate/<campaign_id>', methods=['POST'])
def investigate_campaign(campaign_id):
    """Run the evidence-aware investigation loop for a campaign.

    Body (all optional): {"attack_id": str, "event_id": str,
    "confidence_threshold": float, "max_steps": int}. attack_id defaults
    to the campaign's last observed technique if omitted.
    """
    body = request.get_json(silent=True) or {}

    context, error_response = _try_load_campaign_context(campaign_id)
    if error_response:
        return error_response
    if context is None:
        return jsonify({"error": f"Campaign {campaign_id} not found"}), 404

    attack_id = body.get('attack_id') or context.last_technique
    if not attack_id:
        return jsonify({
            "error": "No attack_id provided and campaign has no last_technique recorded"
        }), 400

    event_id = body.get('event_id', '')
    executor = default_action_executor(context, attack_id, event_id)
    model_predictor = default_model_predictor(context, attack_id, event_id)

    try:
        record = run_investigation(
            executor,
            model_predictor=model_predictor,
            confidence_threshold=float(body.get('confidence_threshold', 0.75)),
            max_uncertainty=float(body.get('max_uncertainty', 0.4)),
            max_steps=int(body.get('max_steps', 8)),
        )
    except Exception as e:
        logger.exception("Investigation failed for campaign %s", campaign_id)
        return jsonify({"error": str(e)}), 500

    payload = record.to_dict()
    payload["campaign_id"] = campaign_id
    payload["evidence"] = [e.to_dict() for e in record.evidence_store.all()]
    return jsonify(payload)


@app.route('/api/rag/mitre/search', methods=['POST'])
def rag_mitre_search():
    """Semantic search over the real MITRE ATT&CK technique corpus."""
    body = request.get_json(silent=True) or {}
    query_text = body.get('query')
    if not query_text:
        return jsonify({"error": "Missing query parameter"}), 400

    top_k = int(body.get('top_k', 5))
    try:
        results = mitre_retriever.query(query_text, top_k=top_k)
    except FileNotFoundError as e:
        return jsonify({"error": f"MITRE corpus unavailable: {e}"}), 503

    return jsonify({"query": query_text, "results": [e.to_dict() for e in results]})


@app.route('/api/ml/predict/severity', methods=['POST'])
def ml_predict_severity():
    """Predict campaign severity via the trained XGBoost classifier.

    Returns 503 (not a crash) if no model has been trained on real
    campaign data yet — ml/train_xgboost.py refuses to train on
    synthetic fixtures being shipped as if they were real, so until
    enough real campaigns have resolved, there is honestly nothing to
    serve here.
    """
    body = request.get_json(silent=True) or {}
    campaign_id = body.get('campaign_id')
    if not campaign_id:
        return jsonify({"error": "Missing campaign_id parameter"}), 400

    model_path = os.path.join(os.path.dirname(__file__), "ml", "models", "xgb_severity.json")
    if not os.path.exists(model_path):
        return jsonify({
            "error": "No trained severity model available yet. The XGBoost classifier "
                     "trains on real resolved-campaign records (ml.dataset_builder) — "
                     "see ml/train_xgboost.py."
        }), 503

    context, error_response = _try_load_campaign_context(campaign_id)
    if error_response:
        return error_response
    if context is None:
        return jsonify({"error": f"Campaign {campaign_id} not found"}), 404

    attack_id = body.get('attack_id') or context.last_technique
    if not attack_id:
        return jsonify({"error": "No attack_id provided and campaign has no last_technique recorded"}), 400

    try:
        from ml.runtime_predictor import RuntimeCampaignPredictor
        predictor = RuntimeCampaignPredictor.from_model_path(model_path)
        result = predictor.predict_for_campaign(context, attack_id, body.get('event_id', ''))
    except Exception as e:
        logger.exception("Severity prediction failed for campaign %s", campaign_id)
        return jsonify({"error": str(e)}), 500

    return jsonify({"campaign_id": campaign_id, **result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
