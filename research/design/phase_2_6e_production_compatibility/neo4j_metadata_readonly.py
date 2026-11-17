"""Read-only metadata from the Neo4j the application is configured to use. One attempt, a fixed list of queries, READ access mode, and nothing is retried
on failure, including authentication. Reads credentials from the application's own .env, and prints none. Reads no alert, campaign or event data: labels,
schema and version only. Refuses to start if a query in the list holds a write clause.
usage: python -I -B neo4j_metadata_readonly.py BACKEND_DIR OUTPUT_JSON"""
import datetime as dt
import json
import re
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

BACKEND, OUTPUT = Path(sys.argv[1]), Path(sys.argv[2])
QUERIES = {
    "components": "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition",
    "databases": "SHOW DATABASES YIELD name, type, access, role, writer, requestedStatus, currentStatus, default, home RETURN name, type, access, role, writer, requestedStatus, currentStatus, default, home",
    "constraints": "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties RETURN name, type, entityType, labelsOrTypes, properties",
    "indexes": "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes AS labels, properties, state, owningConstraint RETURN name, type, entityType, labels, properties, state, owningConstraint",
    "labels": "CALL db.labels() YIELD label RETURN label ORDER BY label",
    "settings": "CALL dbms.listConfig() YIELD name, value WHERE name IN ['db.transaction.timeout', 'db.lock.acquisition.timeout', 'dbms.transaction.concurrent.maximum', 'db.tx_log.rotation.retention_policy', 'server.memory.heap.max_size', 'server.memory.pagecache.size', 'dbms.mode', 'initial.dbms.default_database'] RETURN name, value ORDER BY name",
}
WRITE_CLAUSE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|ALTER|LOAD|TERMINATE|FOREACH|GRANT|REVOKE|DENY|START|STOP)\b", re.I)
for name, query in QUERIES.items():
    if WRITE_CLAUSE.search(query):
        raise SystemExit(f"STOP: query {name} holds a write clause")


def read_env(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


env = read_env(BACKEND / ".env")
uri = env["NEO4J_URI"]
parsed = urlparse(uri)
if parsed.username or parsed.password:
    raise SystemExit("STOP: the configured URI holds credentials, and this script won't handle that form")
if parsed.scheme not in ("bolt",) or parsed.hostname not in ("localhost", "127.0.0.1") or parsed.port != 7687:
    raise SystemExit(f"STOP: the configured target isn't the expected local form: scheme {parsed.scheme}, port {parsed.port}")

# Only the configured target can be reached from here.
real_connect = socket.socket.connect


def guarded(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6) and (address[0] not in ("127.0.0.1", "::1", "localhost") or address[1] != parsed.port):
        raise SystemExit(f"STOP: attempted connection outside the configured target: {address}")
    return real_connect(sock, address)


socket.socket.connect = guarded
from neo4j import READ_ACCESS, GraphDatabase, unit_of_work  # noqa: E402
import neo4j  # noqa: E402

result = {"collected_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "configured_target": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}", "database_configured_by_the_application": None,
          "credential_source": "backend .env, names NEO4J_USERNAME and NEO4J_PASSWORD, values not recorded", "driver": neo4j.__version__, "access_mode": "READ", "queries": QUERIES, "results": {}, "query_types": {}, "errors": {}}
driver = GraphDatabase.driver(uri, auth=(env.get("NEO4J_USERNAME", ""), env.get("NEO4J_PASSWORD", "")), connection_timeout=5, max_transaction_retry_time=0.1)
try:
    for name, query in QUERIES.items():
        try:
            def read(tx):
                outcome = tx.run(query)
                rows = [dict(r) for r in outcome]
                return rows, outcome.consume().query_type
            database = "system" if name == "databases" else None
            with driver.session(default_access_mode=READ_ACCESS, database=database) as session:
                rows, query_type = session.execute_read(unit_of_work(timeout=10)(read))
            result["results"][name] = rows
            result["query_types"][name] = query_type
            if query_type not in ("r", "s"):
                raise SystemExit(f"STOP: {name} reported query type {query_type}")
        except SystemExit:
            raise
        except Exception as error:
            code = getattr(error, "code", None)
            result["errors"][name] = {"type": type(error).__name__, "code": code}
            if type(error).__name__ in ("AuthError", "ServiceUnavailable", "SessionExpired"):
                break  # no repeat attempts at a server that refused or can't be reached
finally:
    driver.close()
OUTPUT.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: result[k] for k in ("collected_utc", "configured_target", "driver", "errors", "query_types")}, default=str))
print(json.dumps(result["results"].get("components")))
