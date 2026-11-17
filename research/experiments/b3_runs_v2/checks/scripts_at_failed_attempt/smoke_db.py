"""Phase 4 connectivity check, run from inside WSL against the disposable
instance only. usage: smoke_db.py <out.json> [--wipe-only]"""
import json
import os
import socket
import sys

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError

URI = os.environ["NEO4J_URI"]
assert URI == "bolt://127.0.0.1:17690", f"disposable instance only, got {URI}"
AUTH = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])

if "--wipe-only" in sys.argv:
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.execute_query("MATCH (n) DETACH DELETE n")
    sys.exit(0)

out = {"uri": URI}

with socket.create_connection(("127.0.0.1", 17690), timeout=5) as sock:
    out["tcp_from_wsl"] = {"connected": True, "peer": list(sock.getpeername())[:2]}

try:
    with GraphDatabase.driver(URI, auth=(AUTH[0], "definitely-wrong")) as bad:
        bad.verify_connectivity()
    out["wrong_password_rejected"] = False
except AuthError:
    out["wrong_password_rejected"] = True

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    out["bolt_auth_ok"] = True
    q = lambda c, **k: [r.data() for r in driver.execute_query(c, **k).records]  # noqa: E731
    out["server"] = q("CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition")
    out["settings"] = q(
        "SHOW SETTINGS YIELD name, value WHERE name IN "
        "['server.memory.heap.initial_size','server.memory.heap.max_size','server.memory.pagecache.size'] "
        "RETURN name, value ORDER BY name")
    out["nodes_at_start"] = q("MATCH (n) RETURN count(n) AS n")[0]["n"]
    q("CREATE (:B3Smoke {id:$id, v:1})", id="smoke")
    out["read_back"] = q("MATCH (s:B3Smoke {id:$id}) RETURN s.v AS v", id="smoke")
    q("MATCH (s:B3Smoke) DELETE s")
    out["nodes_after_delete"] = q("MATCH (n) RETURN count(n) AS n")[0]["n"]

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
