"""Direct test of what a duplicate UNKNOWN alert does to Campaign.last_seen, using
each variant's own neo4j_client against the disposable database.
usage (cwd = a variant): test_dup_update.py <variant> <out.json>"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

assert os.environ["NEO4J_URI"] == "bolt://127.0.0.1:17690", "disposable instance only"
sys.path.insert(0, os.getcwd())
import neo4j_client as n  # noqa: E402

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
at = lambda s: T0 + timedelta(seconds=s)  # noqa: E731

cid = n.create_campaign_db("10.98.0.5", "10.98.0.9", event_time=at(0))
unknown = n.create_unattributed_attack_event(
    campaign_id=cid, attacker_ip="10.98.0.5", victim_ip="10.98.0.9", fingerprint="fp-u",
    rule_id="31101", agent_id="098", event_time=at(2), event_time_source="ALERT_TIMESTAMP",
)
resolved = n.create_attack_event(
    campaign_id=cid, attacker_ip="10.98.0.5", victim_ip="10.98.0.9", mitre_id="T1595", technique="T1595",
    stage="Reconnaissance", tps=0, fingerprint="fp-r", rule_id="900001", agent_id="098",
    event_time=at(3), event_time_source="ALERT_TIMESTAMP",
)


def read(event_id):
    records, _, _ = n.driver.execute_query(
        "MATCH (c:Campaign {campaign_id:$cid}) MATCH (e:AttackEvent {event_id:$eid}) "
        "RETURN toString(c.last_seen) AS campaign_last_seen, c.occurrences AS campaign_occurrences, "
        "toString(e.last_seen) AS event_last_seen, e.occurrences AS event_occurrences",
        cid=cid, eid=event_id)
    return records[0].data()


out = {"variant": sys.argv[1], "before": read(unknown)}
n.update_duplicate_event(unknown, tps=0, investigation_payload="{}", event_time=at(50))
out["after_unknown_duplicate_t50"] = read(unknown)
n.update_duplicate_event(resolved, tps=0, investigation_payload="{}", event_time=at(60))
out["after_resolved_duplicate_t60"] = read(resolved)
n.driver.execute_query("MATCH (n) DETACH DELETE n")
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out))
