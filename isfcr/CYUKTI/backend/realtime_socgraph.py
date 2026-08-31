import json
import os
import requests
from feature_orchestrator import engine as feature_engine
from severity_engine import engine as severity_engine
from dynamic_risk_engine import engine as dynamic_risk_engine
from prediction_engine import predict_next
from recommendation_engine import get_recommendations
from mitre_mapper import MITRE_TO_STAGE
from dynamic_tps import dynamic_tps
from chain_updater import update_attack_chain
from campaign_manager import campaign_manager
from neo4j_client import (
    create_attack_event,
    create_unattributed_attack_event,
    update_duplicate_event,
    update_campaign_similarity,
    update_actor_attribution,
    store_dynamic_risk,
    store_cti_confidence,
    attach_campaign_to_operation,
    create_operation_db,
    update_operation_activity,
)
from mitre_resolver import resolve_mitre, PROVENANCE_UNKNOWN, PROVENANCE_AMBIGUOUS

from dedup_engine import (
    dedup_engine,
    duplicate_buffer,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)
from config import (
    ENABLE_DUPLICATE_BUFFER,
    MISP_URL,
    MISP_API_KEY,
    VERIFY_MISP_SSL,
)
from campaign_correlation_engine import engine as operation_engine
from operation_manager import operation_manager
from detection_confidence_engine import engine as detection_engine
from threat_intelligence_engine import engine as threat_engine
from cti_confidence_engine import engine as cti_engine
from cti_publisher import CTIPublisher
from misp_sync import MISPSync
from misp_event_generator import IncidentContext
from threat_attribution_engine import engine as attribution_engine
publisher = CTIPublisher(
    MISP_URL,
    MISP_API_KEY,
    VERIFY_MISP_SSL,
)
sync = MISPSync(publisher)
SHUFFLE_WEBHOOK = ""

def extract_iocs(alert):
    data = alert.get("data", {})
    agent = alert.get("agent", {})

    attacker = (
        data.get("src_ip")
        or data.get("srcip")
    )

    victim = (
        data.get("dest_ip")
        or data.get("dstip")
    )
    if not victim:
        victim = (
            data.get("dest_ip")
            or data.get("dstip")
            or agent.get("name")
            or agent.get("id")
            or data.get("hostname")
            or alert.get("hostname")
            or alert.get("location")
        )
    if not attacker:
        attacker = victim
    return attacker, victim

IGNORED_AUDIT_COMMANDS = {
    "cat",
    "grep",
    "find",
    "env",
    "ls",
    "pwd",
    "echo",
    "sleep",
    "true",
    "false",
    "dirname",
    "basename",
    "which",
    "head",
    "tail",
    "sort",
    "uniq",
    "cut",
    "sed",
    "awk",
    "tr",
    "printf",
}


def extract_audit_command(alert):
    data = alert.get("data", {})
    audit = data.get("audit")

    if not isinstance(audit, dict):
        return None

    command = (
        audit.get("command")
        or audit.get("exe")
        or audit.get("comm")
    )

    if not command:
        return None

    command = str(command).strip()
    executable = command.split()[0]

    return os.path.basename(executable)


def should_ignore_audit_command(alert):
    command = extract_audit_command(alert)

    if command is None:
        return False

    if command in IGNORED_AUDIT_COMMANDS:
        logger.debug(
            "Ignored helper command: %s",
            command,
        )
        return True

    return False

def _process_unattributed_alert(alert, rule, rule_id, agent_id, resolution):
    """Phase 20 UNKNOWN/AMBIGUOUS path: preserves the alert as evidence
    without fabricating an ATT&CK technique. Deliberately does NOT call
    campaign_manager.append_technique(), chain_updater.update_attack_chain(),
    prediction_engine.predict_next(), or MISP publication -- all of those
    key on a real technique and must not run for an unattributed event.
    """
    attacker_ip, victim_ip = extract_iocs(alert)

    compact_payload = json.dumps({
        "rule_id": rule.get("id"),
        "rule_description": rule.get("description"),
        "rule_level": rule.get("level"),
        "rule_groups": rule.get("groups", []),
        "rule_firedtimes": rule.get("firedtimes"),
        "mitre": rule.get("mitre"),
        "network": alert.get("data", {}).get("network", {}),
        "http": alert.get("data", {}).get("http", {}),
        "suricata": alert.get("data", {}).get("suricata", {}),
        "agent": alert.get("agent", {}),
        "location": alert.get("location"),
        "timestamp": alert.get("timestamp"),
    })

    duplicate, fingerprint = dedup_engine.is_duplicate(
        attacker_ip=attacker_ip,
        victim_ip=victim_ip,
        technique_id=None,
        rule_id=rule_id,
        agent_id=agent_id,
    )

    if duplicate:
        metadata = dedup_engine.get_metadata(fingerprint)
        if metadata:
            update_duplicate_event(
                event_id=metadata["event_id"],
                tps=0,
                investigation_payload=compact_payload,
            )
        print("\n[UNATTRIBUTED DUPLICATE]")
        print(f"Rule ID : {rule_id}")
        print(f"Reason  : {resolution.reason}")
        return True

    rule_level = rule.get("level", 0)

    # Phase 20E: resolve_campaign_context(..., include_inactive=True) was
    # removed here -- campaign_manager.load_from_database's include_inactive
    # branch calls neo4j_client.get_recent_inactive_campaign_db(), which
    # returns list[dict], while load_from_database indexes the result as a
    # single dict (data["last_seen"]) -- a pre-existing type mismatch with
    # no prior caller anywhere in the codebase before this Phase 20 path,
    # so it has never been exercised until now. Fixing that mismatch is out
    # of scope for Phase 20 (campaign_manager.py/neo4j_client.py are not to
    # be touched here); instead the UNKNOWN path uses only the same
    # known-good active-campaign lookup the resolved path already relies on
    # (include_inactive defaults to False -> get_active_campaign_db, which
    # correctly returns a single dict or None). This means an UNKNOWN event
    # can attach to an existing ACTIVE campaign, but will not look up a
    # recently-closed INACTIVE one -- it creates a fresh campaign shell
    # instead in that case. See realtime_socgraph.py docstring.
    context = campaign_manager.resolve_campaign_context(
        attacker_ip, victim_ip
    )
    if context is None:
        # No active campaign for this attacker/victim pair -- create the
        # campaign shell via the existing (unmodified) campaign_manager
        # helper, with no technique to record. This does not corrupt any
        # prior last_technique, since there is no prior state for a
        # brand-new campaign.
        context = campaign_manager.create_campaign_context(
            attacker_ip, victim_ip, current_technique=None
        )
    campaign_id = context.campaign_id

    event_id = create_unattributed_attack_event(
        campaign_id=campaign_id,
        attacker_ip=attacker_ip,
        victim_ip=victim_ip,
        fingerprint=fingerprint,
        rule_id=rule_id,
        agent_id=agent_id,
        rule_level=rule_level,
        investigation_payload=compact_payload,
        mitre_provenance=resolution.provenance,
        mitre_confidence=resolution.confidence,
        mitre_reason=resolution.reason,
        mitre_resolver_version=resolution.resolver_version,
        mitre_technique_ids=list(resolution.technique_ids),
    )
    dedup_engine.register_event(
        fingerprint=fingerprint,
        campaign_id=campaign_id,
        event_id=event_id,
    )

    print("\n[UNATTRIBUTED ALERT]")
    print(f"Campaign : {campaign_id}")
    print(f"Rule ID  : {rule_id}")
    print(f"Reason   : {resolution.reason}")

    return True


def process_alert(alert):
    try:
        if not isinstance(alert, dict):
            return False
        
        rule = alert.get("rule")
        if not rule:
            logger.info("Skipping alert: missing rule")
            return False
        if should_ignore_audit_command(alert):
            return False
        rule_id = rule.get("id")
        
        agent_id = (
            alert.get("agent", {})
                 .get("id", "UNKNOWN")
        )
        resolution = resolve_mitre(alert)

        if resolution.provenance in (PROVENANCE_UNKNOWN, PROVENANCE_AMBIGUOUS):
            logger.info(
                "No defensible ATT&CK mapping (%s): %s",
                resolution.provenance,
                resolution.reason,
            )
            return _process_unattributed_alert(alert, rule, rule_id, agent_id, resolution)

        mitre = rule.get("mitre") or {}
        mitre_id = resolution.technique_ids[0]
        native_technique_names = mitre.get("technique") or []
        # native_technique_names[0] preserves the exact prior behavior
        # (techniques[0]) for NATIVE_WAZUH alerts, which always carry
        # both id and technique together in practice. REVIEWED_RULE_MAPPING
        # and DETERMINISTIC_INFERENCE results have no Wazuh-supplied name,
        # so the technique_id itself is used as a placeholder -- this is
        # not a fabricated name, just a stand-in when Wazuh didn't supply
        # one; the real name already exists on the imported STIX Technique
        # node and is unaffected by this placeholder.
        technique = native_technique_names[0] if native_technique_names else mitre_id
        COMMAND_WHITELIST = {
            "bash",
            "sh",
            "python",
            "python3",
            "curl",
            "wget",
            "ssh",
            "scp",
            "nmap",
            "hydra",
            "nc",
            "netcat",
            "socat",
            "msfconsole",
            "crontab",
            "systemctl",
            "iptables",
            "chmod",
            "chown",
            "sudo",
            "whoami",
            "hostname",
            "uname",
            "ifconfig",
            "ip",
            "netstat",
            "ss",
            "ps",
        }
        stage = MITRE_TO_STAGE.get(
            mitre_id,
            "Unknown"
        )
        attacker_ip, victim_ip = extract_iocs(alert)

        tps = dynamic_tps.get_tps(
            stage=stage,
            technique_id=mitre_id,
            campaign_id=None,     
            attacker_ip=attacker_ip,
        )
        rule_level = rule.get("level", 0)
        compact_payload = json.dumps({
        
            "rule_id": rule.get("id"),
            "rule_description": rule.get("description"),
            "rule_level": rule.get("level"),
            "rule_groups": rule.get("groups", []),
            "rule_firedtimes": rule.get("firedtimes"),
        
            "mitre": mitre,
        
            "network": alert.get("data", {}).get("network", {}),
            "http": alert.get("data", {}).get("http", {}),
            "suricata": alert.get("data", {}).get("suricata", {}),
        
            "agent": alert.get("agent", {}),
            "location": alert.get("location"),
        
            "timestamp": alert.get("timestamp")
        })
        print("\n========== DEDUP INPUT ==========")
        print(f"Attacker  : {attacker_ip}")
        print(f"Victim    : {victim_ip}")
        print(f"Technique : {mitre_id}")
        print(f"Rule ID   : {rule_id}")
        print(f"Agent ID  : {agent_id}")
        print("=================================")
        duplicate, fingerprint = dedup_engine.is_duplicate(
            attacker_ip=attacker_ip,
            victim_ip=victim_ip,
            technique_id=mitre_id,
            rule_id=rule_id,
            agent_id=agent_id
        )
        print(f"Fingerprint : {fingerprint}")
        print(f"Duplicate   : {duplicate}")
        if duplicate:
            metadata = dedup_engine.get_metadata(fingerprint)
            if metadata:        
                if ENABLE_DUPLICATE_BUFFER:
                    success = duplicate_buffer.add_duplicate(
                        fingerprint=fingerprint,
                        event_id=metadata["event_id"],
                        campaign_id=metadata["campaign_id"],
                        tps=tps,
                        payload=compact_payload
                    )
                    if success:
                        print("\n==============================================")
                        print("[DUPLICATE ALERT]")
                        print(f"Fingerprint : {fingerprint[:12]}...")
                        print(f"Event ID    : {metadata['event_id']}")
                        print("Buffered for batch update.")
                        print(f"Pending Buffer : {duplicate_buffer.size()}")
                        print("==============================================")
                    else:
                        print(
                            f"[WARNING] Duplicate buffer full "
                            f"({duplicate_buffer.size()} fingerprints). "
                            "Falling back to immediate Neo4j update."
                        )
        
                        update_duplicate_event(
                            event_id=metadata["event_id"],
                            tps=tps,
                            investigation_payload=compact_payload
                        )
        
                else:
        
                    update_duplicate_event(
                        event_id=metadata["event_id"],
                        tps=tps,
                        investigation_payload=compact_payload
                    )
        
                    print("\n==============================================")
                    print("[DUPLICATE ALERT]")
                    print(f"Fingerprint : {fingerprint[:12]}...")
                    print(f"Event ID    : {metadata['event_id']}")
                    print("Existing AttackEvent updated.")
                    print("==============================================")
        
            return True
            
        campaign_id = campaign_manager.get_or_create_campaign(
            attacker_ip,
            victim_ip,
            mitre_id
        )
        
        print("\n===================================================")
        print("[NEW ALERT]")
        print(f"Campaign : {campaign_id}")
        print(f"Attacker : {attacker_ip}")
        print(f"Victim   : {victim_ip}")
        print(f"Technique: {mitre_id}")
        print(f"Stage    : {stage}")
        print("===================================================")

        event_id = create_attack_event(       
            campaign_id=campaign_id,
            attacker_ip=attacker_ip,
            victim_ip=victim_ip,
            mitre_id=mitre_id,
            technique=technique,
            stage=stage,
            tps=tps,
            fingerprint=fingerprint,
            rule_id=rule_id,
            agent_id=agent_id,
            rule_level=rule_level,
            investigation_payload=compact_payload,
            mitre_provenance=resolution.provenance,
            mitre_confidence=resolution.confidence,
            mitre_reason=resolution.reason,
            mitre_resolver_version=resolution.resolver_version,
            mitre_technique_ids=list(resolution.technique_ids),
        )
        dedup_engine.register_event(
            fingerprint=fingerprint,
            campaign_id=campaign_id,
            event_id=event_id
        )
        
        print("[GRAPH] AttackEvent Created")
        print(f"[GRAPH] Event ID : {event_id}")
        features = feature_engine.extract_features(
            mitre_id,
            campaign_id,
            attacker_ip,
            event_id
        )      
        severity = severity_engine.calculate(
            features
        )
        dynamic_risk = dynamic_risk_engine.calculate(
            features,
            severity
        )
        store_dynamic_risk(
            campaign_id,
            dynamic_risk
        )
        print("\n[DYNAMIC RISK]")       
        print(
            f"Score      : {dynamic_risk.risk_score:.2f}"
        )
        print(
            f"Level      : {dynamic_risk.risk_level}"
        )
        print(
            f"Confidence : {dynamic_risk.confidence}%"
        )
        for key, value in dynamic_risk.breakdown.items():
            print(f"  {key:<25}: {value}")
        update_attack_chain(
            campaign_id,
            mitre_id
        )

        print("[CHAIN] Updated")
        try:

            update_campaign_similarity(
                campaign_id
            )

            update_actor_attribution(
                campaign_id
            )

            print("[INTELLIGENCE] Updated")

        except Exception as e:

            print(f"[INTELLIGENCE ERROR] {e}")

        prediction = predict_next(
            campaign_id,
            mitre_id
        )

        predicted = "NONE"
        confidence = 0
        recommendations = []
        if prediction:

            predicted = prediction["predicted"]
            confidence = prediction["confidence"]

            print("\n[PREDICTION]")
            print(
                f"{mitre_id}  -->  {predicted}"
            )

            print(
                f"Confidence : {confidence}%"
            )

            recommendations = get_recommendations(
                predicted
            )

            print("\n[RECOMMENDATIONS]")

            for rec in recommendations:

                text = (
                    rec["recommendation"]
                    if isinstance(rec, dict)
                    else rec
                )

                print(f" - {text}")

        else:

            print("\n[PREDICTION]")
            print("No learned transition.")

        context = operation_manager.build_campaign_context(
            campaign_id
        )
        print("\n===== FRESH CAMPAIGN CONTEXT =====")
        print("Techniques :", context.techniques)
        print("Chain      :", context.attack_chain)
        print("Predicted  :", context.predicted_next)
        print("=================================")
        result = operation_engine.correlate(context)
        
        print("\n========== OPERATION CORRELATION ==========")
        print(f"Matched     : {result.matched}")
        print(f"Score       : {result.score:.3f}")
        print(f"Confidence  : {result.confidence:.1f}%")
        print(f"Candidates  : {result.candidate_count}")
        print("===========================================\n")

        detection = detection_engine.calculate(event_id)
        
        threat = threat_engine.calculate(attacker_ip)
        
        cti = cti_engine.calculate(
            detection_confidence=detection.confidence,
            risk_score=dynamic_risk.risk_score,
            threat_confidence=threat.confidence,
            campaign_confidence=result.confidence,
            prediction_confidence=confidence,
        )
        store_cti_confidence(
            campaign_id,
            cti
        )
        print("\n========== CTI CONFIDENCE ==========")
        print(f"Score      : {cti.score}")
        print(f"Level      : {cti.level}")
        print(f"Publish    : {cti.publish}")
        
        for name, item in cti.breakdown.items():
            print(
                f"{name:<24}: "
                f"{item['value']:>6.2f} "
                f"(+{item['contribution']:>5.2f})"
            )
        
        print("====================================")
        incident = IncidentContext(
            campaign_id=campaign_id,
            operation_id=result.operation_id if result.matched else "PENDING",
        
            attacker_ip=attacker_ip,
            victim_ip=victim_ip,
        
            event_id=event_id,
        
            technique=mitre_id,
            stage=stage,
        
            prediction=predicted,
            prediction_confidence=confidence,
        
            detection=detection,
            threat=threat,
        
            dynamic_risk=dynamic_risk,
            cti=cti,
        
            recommendations=recommendations,
        
            investigation_payload=json.dumps(
                alert,
                indent=2,
                default=str,
            ),
        
            timestamp=str(alert.get("timestamp", "")),
        )
        # ---------- Threat Attribution ----------
        attribution = None
        try:    
            attribution = attribution_engine.attribute(
                context
            )
            incident.attribution = attribution
            print("\n========== THREAT ATTRIBUTION ==========")
            if attribution.actors:
                for actor in attribution.actors:
                    print(
                        f"{actor.actor:<20}"
                        f"{actor.total_score:>8.2f}%"
                    )
            else:
                print("No historical campaign match found.")
            print("========================================")
        
        except Exception:
            logger.exception(
                "Threat Attribution failed"
            )
        
        if result.matched:
            operation_id = result.operation_id
        
            attach_campaign_to_operation(
                operation_id,
                context
            )
        
            update_operation_activity(
                operation_id,
                context
            )
        
            print(
                f"[OPERATION] Attached campaign "
                f"{campaign_id} -> {operation_id}"
            )
        
        else:
        
            operation_id = create_operation_db(context)
        
            attach_campaign_to_operation(
                operation_id,
                context
            )
        
            print(f"[OPERATION] Created {operation_id}")
        incident.operation_id = operation_id
        result = sync.publish_campaign(incident)
        
        print("\n========== MISP ==========")
        
        print(f"Action      : {result.get('action', '-')}")
        print(f"Success     : {result['success']}")
        
        if "reason" in result:
            print(f"Reason      : {result['reason']}")
        
        if result["success"]:
        
            response = result.get("response", {})
        
            if isinstance(response, dict):
        
                event = response.get("Event", {})
        
                if event:
                    print(
                        f"MISP Event ID : "
                        f"{event.get('id','UNKNOWN')}"
                    )
        
        print("==========================")
        try:

            response = requests.post(

                SHUFFLE_WEBHOOK,

                json={
                    "campaign": campaign_id,
                    "attacker_ip": attacker_ip,
                    "victim_ip": victim_ip,
                    "current": mitre_id,
                    "predicted": predicted,
                    "confidence": confidence,
                    "event_id": event_id,
                    "stage": stage,
                    "risk_score": dynamic_risk.risk_score,
                    "risk_level": dynamic_risk.risk_level,
                    "recommendations": recommendations if prediction else []
                },

                timeout=10

            )

            print(
                f"[SHUFFLE] HTTP {response.status_code}"
            )

        except Exception as e:

            print(
                f"[SHUFFLE ERROR] {e}"
            )

        return True

    except Exception as e:

        print(f"[PROCESS ALERT ERROR] {e}")

        return False