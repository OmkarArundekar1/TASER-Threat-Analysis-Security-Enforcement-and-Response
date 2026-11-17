"""Read-only production fingerprint: counts and newest timestamps."""
import json, sys
sys.path.insert(0, ".")
from neo4j import READ_ACCESS
from neo4j_client import driver
Q = {
 "nodes": "MATCH (n) RETURN count(n) AS v", "rels": "MATCH ()-[r]->() RETURN count(r) AS v",
 "campaigns": "MATCH (c:Campaign) RETURN count(c) AS v", "operations": "MATCH (o:Operation) RETURN count(o) AS v",
 "events": "MATCH (e:AttackEvent) RETURN count(e) AS v",
 "next_technique": "MATCH ()-[r:NEXT_TECHNIQUE]->() RETURN sum(r.count) AS v",
 "newest_event": "MATCH (e:AttackEvent) RETURN toString(max(e.last_seen)) AS v",
 "newest_campaign": "MATCH (c:Campaign) RETURN toString(max(c.last_seen)) AS v",
 "active_campaigns": "MATCH (c:Campaign {status:'ACTIVE'}) RETURN count(c) AS v",
}
with driver.session(default_access_mode=READ_ACCESS) as s:
    print(json.dumps({k: s.execute_read(lambda t, q=q: t.run(q).single()["v"]) for k, q in Q.items()}, sort_keys=True))
