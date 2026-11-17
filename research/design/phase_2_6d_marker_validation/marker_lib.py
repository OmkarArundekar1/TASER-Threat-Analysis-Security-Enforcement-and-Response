"""Shared code for the D6 marker validation. Isolated: it imports nothing production as a module, so it can't load .env or
production settings, and it can't reach any Neo4j but the disposable one.

Isolation, checked before any write, in order:
  1. A socket guard refuses every connection except 127.0.0.1 on this experiment's two ports. Port 7687 can't be reached.
  2. confirm_environment() requires the expected Docker Desktop daemon, this experiment's label, its port mapping and its volume.
  3. verify_disposable() requires the random token stamped into the database, which must have been empty when stamped.

The real write helper is used without importing neo4j_client: its functions are lifted from the source by AST, with this
experiment's driver in place of the production one. So outcomes here are classified by the code a coordinator would use.
FaultProxy comes from ~/cyukti_diag/helper_revalidation_v1/txnlib.py, the version that sends FIN on a dropped connection.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import os
import socket
import struct
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2] / "isfcr" / "CYUKTI" / "backend"
STATE_FILE = Path(os.environ.get("CYUKTI_D6_STATE_DIR", Path.home() / "cyukti_diag" / "d6_marker_v1" / "state")) / "conn.json"
if not STATE_FILE.exists():
    raise SystemExit(f"STOP: no disposable instance. Run env/neo4j_up.sh first. Expected state at {STATE_FILE}")
CONN = json.loads(STATE_FILE.read_text())
EXPECTED_DAEMON = "7879fd3e-b372-45e5-942a-07cc7ca07c63"
EXPERIMENT_LABEL = "d6-marker-v1"
ALLOWED_PORTS = {CONN["upstream_port"], CONN["proxy_port"]}
if ALLOWED_PORTS & {7687, 7474, 7473}:
    raise SystemExit("STOP: the experiment's ports include a standard Neo4j port")

_real_connect = socket.socket.connect


class TargetRefused(BaseException):
    """A connection outside the experiment. A BaseException, so nothing can absorb it as an ordinary failure."""


def _guarded_connect(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        if address[0] not in ("127.0.0.1", "::1", "localhost") or address[1] not in ALLOWED_PORTS:
            raise TargetRefused(f"attempted connection outside the experiment: {address}")
    return _real_connect(sock, address)


socket.socket.connect = _guarded_connect

from neo4j import GraphDatabase, unit_of_work  # noqa: E402
from neo4j.exceptions import DriverError, IncompleteCommit, Neo4jError  # noqa: E402

logging.getLogger("neo4j").setLevel(logging.CRITICAL)
logging.getLogger("lifted_helper").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------- environment

def _docker(*args, timeout=60):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise SystemExit(f"STOP: docker {args[0]} failed")
    return result.stdout.strip()


def confirm_environment():
    """Positive identification of the disposable container. Returns what it found, with no credentials."""
    daemon = _docker("info", "--format", "{{.ID}}")
    platform = _docker("version", "--format", "{{.Server.Platform.Name}}")
    if daemon != EXPECTED_DAEMON or not platform.startswith("Docker Desktop"):
        raise SystemExit("STOP: this isn't the expected Docker Desktop daemon")
    info = json.loads(_docker("inspect", CONN["container"]))[0]
    if info["Config"]["Labels"].get("cyukti.experiment") != EXPERIMENT_LABEL:
        raise SystemExit("STOP: the container doesn't carry this experiment's label")
    binding = info["HostConfig"]["PortBindings"].get("7687/tcp")
    if binding != [{"HostIp": "127.0.0.1", "HostPort": str(CONN["upstream_port"])}]:
        raise SystemExit("STOP: the container's port mapping isn't this experiment's")
    mounts = [(m["Type"], m.get("Name"), m["Destination"]) for m in info["Mounts"]]
    if ("volume", CONN["volume"], "/data") not in mounts:
        raise SystemExit("STOP: the container isn't using this experiment's volume")
    return {"daemon_id": daemon, "platform": platform, "container": CONN["container"], "container_id": info["Id"][:16], "image": info["Config"]["Image"],
            "mounts": mounts, "target": CONN["uri"], "proxy_target": CONN["proxy_uri"], "memory": info["HostConfig"]["Memory"], "nano_cpus": info["HostConfig"]["NanoCpus"]}


def direct_driver(**kwargs):
    return GraphDatabase.driver(CONN["uri"], auth=(CONN["user"], CONN["password"]), connection_timeout=5, **kwargs)


def proxied_driver(**kwargs):
    settings = {"connection_timeout": 5, "max_transaction_retry_time": 3.0, "max_connection_pool_size": 5}
    settings.update(kwargs)
    return GraphDatabase.driver(CONN["proxy_uri"], auth=(CONN["user"], CONN["password"]), **settings)


def initialize(driver):
    """Stamps an EMPTY database with this run's token, and creates the marker constraint. Refuses a database holding anything else."""
    with driver.session() as session:
        total = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        stamped = session.run("MATCH (m:CyuktiDisposableTestDatabase {token:$t}) RETURN count(m) AS n", t=CONN["marker"]).single()["n"]
        if total and not stamped:
            raise SystemExit(f"STOP: the database holds {total} nodes and not this experiment's token")
        if not total:
            session.run("CREATE (:CyuktiDisposableTestDatabase {token:$t})", t=CONN["marker"]).consume()
        session.run("CREATE CONSTRAINT applied_write_key IF NOT EXISTS FOR (m:AppliedWrite) REQUIRE m.request_key IS UNIQUE").consume()
        session.run("CREATE CONSTRAINT other_unique_uid IF NOT EXISTS FOR (m:OtherUnique) REQUIRE m.uid IS UNIQUE").consume()
        session.run("CREATE INDEX attack_event_id IF NOT EXISTS FOR (e:AttackEvent) ON (e.event_id)").consume()
        session.run("CALL db.awaitIndexes(60)").consume()


def verify_disposable(driver):
    with driver.session() as session:
        if session.run("MATCH (m:CyuktiDisposableTestDatabase {token:$t}) RETURN count(m) AS n", t=CONN["marker"]).single()["n"] != 1:
            raise SystemExit("STOP: the database doesn't hold this experiment's token")


def server_identity(driver):
    with driver.session() as session:
        component = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition").single()
        return {"name": component["name"], "versions": component["versions"], "edition": component["edition"]}


def constraints(driver):
    with driver.session() as session:
        return [dict(r) for r in session.run("SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties RETURN name, type, labelsOrTypes, properties")]


# ---------------------------------------------------------------- the real write helper, lifted

_HELPER_NAMES = {"WriteOutcome", "_VERIFIED_COMMIT_REJECTIONS", "_SUMMARY_COUNTERS", "_WRITE_LIMIT_CEILING_SECONDS", "_WRITE_LIMITS_TOTAL_CEILING_SECONDS",
                 "_WRITE_ACQUISITION_CEILING_SECONDS", "WriteLimits", "DEFAULT_WRITE_LIMITS", "WriteResult", "_RetryAfterCommitAttempt", "_RetryBudgetExhausted",
                 "_classify_write_failure", "_check_parameter_value", "_checked_statements", "_execute_write", "_finish_write", "write_statements", "write_query"}
_tree = ast.parse((BACKEND / "neo4j_client.py").read_text(encoding="utf-8"))
_picked = [n for n in _tree.body if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _HELPER_NAMES)
           or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in _HELPER_NAMES)]
_found = {getattr(n, "name", None) or n.targets[0].id for n in _picked}
if _found != _HELPER_NAMES:
    raise SystemExit(f"STOP: the helper's source no longer has {_HELPER_NAMES - _found}")
_compiled = compile(ast.Module(_picked, []), "neo4j_client.py", "exec", dont_inherit=True)


def helper(driver):
    """A private copy of the real helper's functions, bound to this driver. Nothing else from neo4j_client is run."""
    namespace = {"unit_of_work": unit_of_work, "DriverError": DriverError, "IncompleteCommit": IncompleteCommit, "Neo4jError": Neo4jError,
                 "logging": logging, "math": math, "time": time, "Enum": Enum, "dataclass": dataclass, "field": field, "Optional": Optional,
                 "logger": logging.getLogger("lifted_helper"), "driver": driver, "__name__": "lifted_neo4j_client"}
    exec(_compiled, namespace)
    return types_namespace(namespace)


def types_namespace(namespace):
    class Helper:
        write_statements = staticmethod(namespace["write_statements"])
        write_query = staticmethod(namespace["write_query"])
        WriteLimits = namespace["WriteLimits"]
        WriteOutcome = namespace["WriteOutcome"]
    return Helper


# ---------------------------------------------------------------- the marker contract under test

# update_duplicate_event's statement (neo4j_client.py), with the marker inside its first MATCH, as D6_APPLIED_WRITE_MARKER_DESIGN.md proposes.
MARKED_DUPLICATE_UPDATE = """
WITH coalesce($event_time, datetime()) AS dt
MATCH (e:AttackEvent {event_id:$event_id})
CREATE (:AppliedWrite {request_key:$request_key, applied_at:datetime()})
SET
    e.last_seen = CASE WHEN e.last_seen IS NULL OR dt > e.last_seen THEN dt ELSE e.last_seen END,
    e.occurrences = coalesce(e.occurrences,0) + 1,
    e.tps = coalesce(e.tps,0) + $tps,
    e.investigation_payload = $investigation_payload
WITH e, dt
MATCH (e)-[:MATCHES]->(t:Technique)
SET
    t.last_seen = datetime(),
    t.occurrences = coalesce(t.occurrences,0) + 1,
    t.total_tps = coalesce(t.total_tps,0) + $tps
WITH e, dt
MATCH (c:Campaign {campaign_id:e.campaign_id})
SET
    c.last_seen = CASE WHEN c.last_seen IS NULL OR dt > c.last_seen THEN dt ELSE c.last_seen END,
    c.occurrences = coalesce(c.occurrences,0) + 1,
    c.total_tps = coalesce(c.total_tps,0) + $tps,
    c.risk_score = coalesce(c.risk_score,0) + $tps
"""
UNMARKED_DUPLICATE_UPDATE = MARKED_DUPLICATE_UPDATE.replace("CREATE (:AppliedWrite {request_key:$request_key, applied_at:datetime()})\n", "")
# The fuller marker E9 tests: operation and fingerprint are immutable properties of the marker.
FINGERPRINTED_DUPLICATE_UPDATE = MARKED_DUPLICATE_UPDATE.replace(
    "CREATE (:AppliedWrite {request_key:$request_key, applied_at:datetime()})",
    "CREATE (:AppliedWrite {request_key:$request_key, applied_at:datetime(), operation:$operation, fingerprint:$fingerprint})")
# A control with no constraint on its label, to show what the constraint is doing.
UNCONSTRAINED_DUPLICATE_UPDATE = MARKED_DUPLICATE_UPDATE.replace("(:AppliedWrite {", "(:AppliedWriteNoConstraint {")
# The trap D6 names: a marker as its own statement, not dependent on the mutation's MATCH.
SEPARATE_MARKER = "CREATE (:AppliedWrite {request_key:$request_key, applied_at:datetime()})"

LOOKUP = "MATCH (m:AppliedWrite {request_key:$request_key}) RETURN m.request_key AS request_key, toString(m.applied_at) AS applied_at, m.operation AS operation, m.fingerprint AS fingerprint"
CONSTRAINT_FAILED = "Neo.ClientError.Schema.ConstraintValidationFailed"
APPLIED, ABSENT, UNKNOWN, INCONSISTENT = "APPLIED", "ABSENT", "UNKNOWN", "INCONSISTENT"


def params(event_id, request_key, *, tps=0, payload='{"rule_id":"86601"}', event_time=None, **extra):
    return {"event_id": event_id, "request_key": request_key, "tps": tps, "investigation_payload": payload, "event_time": event_time, **extra}


@dataclass(frozen=True)
class Lookup:
    """APPLIED is positive evidence. ABSENT says only that a read that succeeded found none at that moment. UNKNOWN says nothing."""
    status: str
    marker: Optional[dict] = None
    error_type: Optional[str] = None
    error_code: Optional[str] = None


def lookup(driver, request_key, *, timeout=3.0, deadline=6.0, query=LOOKUP):
    """The read a recovery step would use. Any failure, including one past the client's own deadline, is UNKNOWN. Never ABSENT."""
    box = {}

    def work():
        try:
            with driver.session(max_transaction_retry_time=1.0, connection_acquisition_timeout=2.0) as session:
                box["rows"] = session.execute_read(unit_of_work(timeout=timeout)(lambda tx: tx.run(query, request_key=request_key).data()))
        except BaseException as error:  # the point: nothing here may read as absence
            box["error"] = error

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(deadline)
    if thread.is_alive():
        return Lookup(UNKNOWN, error_type="ClientDeadline")
    if "error" in box:
        return Lookup(UNKNOWN, error_type=type(box["error"]).__name__, error_code=getattr(box["error"], "code", None))
    rows = box["rows"]
    if len(rows) == 1:
        return Lookup(APPLIED, rows[0])
    return Lookup(ABSENT) if not rows else Lookup(INCONSISTENT, {"count": len(rows)})


# ---------------------------------------------------------------- the fault proxy

CLIENT = {0x01: "HELLO", 0x02: "GOODBYE", 0x0F: "RESET", 0x10: "RUN", 0x11: "BEGIN", 0x12: "COMMIT", 0x13: "ROLLBACK", 0x2F: "DISCARD", 0x3F: "PULL",
          0x54: "TELEMETRY", 0x66: "ROUTE", 0x6A: "LOGON", 0x6B: "LOGOFF"}
SERVER = {0x70: "SUCCESS", 0x71: "RECORD", 0x7E: "IGNORED", 0x7F: "FAILURE"}


class Timeline:
    def __init__(self):
        self.events, self.lock, self.start = [], threading.Lock(), time.monotonic()

    def add(self, source, what):
        with self.lock:
            self.events.append({"t_ms": round((time.monotonic() - self.start) * 1000, 1), "source": source, "event": what})

    def names(self):
        with self.lock:
            return [f'{e["source"]}:{e["event"]}' for e in self.events]


def _hang_up(*sockets):
    for sock in sockets:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()


def _read_exact(sock, n):
    data = b""
    while len(data) < n:
        part = sock.recv(n - len(data))
        if not part:
            raise ConnectionError("closed")
        data += part
    return data


def _read_message(sock):
    raw, payload = b"", b""
    while True:
        head = _read_exact(sock, 2)
        raw += head
        size = struct.unpack(">H", head)[0]
        if size == 0:
            if payload:
                return raw, payload
            continue
        chunk = _read_exact(sock, size)
        raw += chunk
        payload += chunk


def _forward_varint(src, dst):
    value, shift = 0, 0
    while True:
        byte = _read_exact(src, 1)
        dst.sendall(byte)
        value |= (byte[0] & 0x7F) << shift
        shift += 7
        if not byte[0] & 0x80:
            return value


class FaultProxy:
    """A Bolt-aware TCP proxy to the disposable server. Acts at exact protocol messages, only on connections that sent a RUN holding the armed id.
      drop_before  closes both sockets instead of forwarding a client message
      hold         delays a client message until released
      drop_reply   withholds the reply to a client message and closes the client's socket, while still reading from the server until every
                   pipelined request is answered, so the server finishes while the client never hears
      lose_after_run_reply  delivers RUN's reply, holds the pipelined PULL, then closes both sides"""

    def __init__(self):
        self.rule = None
        self.timeline = Timeline()
        self.run_id = None
        self.release = threading.Event()
        self.holding = threading.Event()
        self.fired = 0
        self.conn_seq = 0
        self.stop = False
        self.conns = {}
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", CONN["proxy_port"]))
        self.server.listen(64)
        threading.Thread(target=self._accept, daemon=True).start()

    def arm(self, run_id, action=None, msg=None, times=1):
        self.rule = {"action": action, "msg": msg, "times": times} if action else None
        self.run_id = run_id.encode()
        self.timeline = Timeline()
        self.release = threading.Event()
        self.holding = threading.Event()
        self.fired = 0

    def kill_all(self):
        for client, upstream in list(self.conns.values()):
            _hang_up(client, upstream)
        self.conns.clear()

    def close(self):
        self.stop = True
        _hang_up(self.server)
        self.kill_all()

    def _fire(self, action, msg, armed):
        rule = self.rule
        if armed and rule and rule["action"] == action and rule["msg"] == msg and rule["times"] > 0:
            rule["times"] -= 1
            self.fired += 1
            return True
        return False

    def _accept(self):
        while not self.stop:
            try:
                client, _ = self.server.accept()
            except OSError:
                return
            self.conn_seq += 1
            threading.Thread(target=self._handle, args=(client, self.conn_seq), daemon=True).start()

    def _handle(self, client, cid):
        upstream = socket.socket()
        try:
            upstream.connect(("127.0.0.1", CONN["upstream_port"]))
            upstream.sendall(_read_exact(client, 20))
            answer = _read_exact(upstream, 4)
            client.sendall(answer)
            if answer == b"\x00\x00\x01\xff":  # handshake manifest v1
                count = _forward_varint(upstream, client)
                for _ in range(count):
                    client.sendall(_read_exact(upstream, 4))
                _forward_varint(upstream, client)
                upstream.sendall(_read_exact(client, 4))
                _forward_varint(client, upstream)
        except (ConnectionError, OSError):
            client.close()
            upstream.close()
            return
        state = {"armed": False, "pending": [], "client_dead": False, "lock": threading.Lock(), "run_reply_sent": threading.Event()}
        self.conns[cid] = (client, upstream)
        threading.Thread(target=self._client_to_server, args=(client, upstream, state, cid), daemon=True).start()
        threading.Thread(target=self._server_to_client, args=(client, upstream, state, cid), daemon=True).start()

    def _client_to_server(self, client, upstream, state, cid):
        try:
            while True:
                raw, payload = _read_message(client)
                name = CLIENT.get(payload[1], hex(payload[1]))
                if name == "RUN" and self.run_id and self.run_id in payload:
                    state["armed"] = True
                if state["armed"]:
                    self.timeline.add("client>proxy", f"{name}#c{cid}")
                if self._fire("drop_before", name, state["armed"]):
                    self.timeline.add("proxy", f"DROPPED {name} before server, closed both sides")
                    _hang_up(client, upstream)
                    return
                if self._fire("lose_after_run_reply", name, state["armed"]):
                    state["run_reply_sent"].wait(5)
                    time.sleep(0.2)
                    self.timeline.add("proxy", f"DROPPED {name} before server, closed both sides")
                    _hang_up(client, upstream)
                    return
                if self._fire("hold", name, state["armed"]):
                    self.timeline.add("proxy", f"HOLDING {name}")
                    self.holding.set()
                    self.release.wait(30)
                    self.timeline.add("proxy", f"RELEASED {name}")
                with state["lock"]:
                    state["pending"].append(name)
                upstream.sendall(raw)
        except (ConnectionError, OSError):
            if not state["pending"]:
                try:
                    upstream.close()
                except OSError:
                    pass

    def _server_to_client(self, client, upstream, state, cid):
        try:
            while True:
                raw, payload = _read_message(upstream)
                name = SERVER.get(payload[1], hex(payload[1]))
                reply_to = None
                if name != "RECORD":
                    with state["lock"]:
                        reply_to = state["pending"].pop(0) if state["pending"] else "?"
                if state["armed"] and name != "RECORD":
                    note = f"{name} to {reply_to}"
                    if name == "FAILURE":
                        start = payload.find(b"Neo.")
                        note += " " + payload[start:start + 70].split(b"\x00")[0].decode("ascii", "replace").split("\x8d")[0] if start >= 0 else ""
                    if state["client_dead"]:
                        note += " (not delivered)"
                    self.timeline.add("server>proxy", note)
                if reply_to and self._fire("drop_reply", reply_to, state["armed"]):
                    self.timeline.add("proxy", f"WITHHELD reply to {reply_to}, closed client side")
                    state["client_dead"] = True
                    note = f"{name} to {reply_to} (not delivered)"
                    self.timeline.add("server>proxy", note)
                    try:
                        client.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    client.close()
                if state["client_dead"]:
                    with state["lock"]:
                        done = not state["pending"]
                    if done:
                        upstream.close()
                        return
                    continue
                client.sendall(raw)
                if reply_to == "RUN":
                    state["run_reply_sent"].set()
        except (ConnectionError, OSError):
            try:
                client.close()
            except OSError:
                pass


# ---------------------------------------------------------------- what's in the database, read directly

class Context:
    """One run's connections, with unique deterministic keys. Reads for assertions always go directly to the server, never through the proxy."""

    def __init__(self, label="run"):
        self.environment = confirm_environment()
        self.direct = direct_driver(max_connection_pool_size=64)
        initialize(self.direct)
        verify_disposable(self.direct)
        self.proxy = FaultProxy()
        self.help = helper(self.direct)
        self.prefix = f"{label}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.sequence = 0

    def key(self, scenario):
        self.sequence += 1
        return f"cyukti.v1|wazuh:d6-manager/{scenario}.{self.prefix}.{self.sequence}|duplicate_event.update"

    def event(self, attributed=False):
        self.sequence += 1
        event_id = f"E-{self.prefix}-{self.sequence}"
        with self.direct.session() as session:
            if attributed:
                session.run("CREATE (:Campaign {campaign_id:$c, occurrences:0, total_tps:0, risk_score:0}) "
                            "CREATE (e:AttackEvent {event_id:$e, campaign_id:$c, occurrences:0, tps:0})-[:MATCHES]->(:Technique {attack_id:$e, occurrences:0, total_tps:0})",
                            e=event_id, c=f"C-{event_id}").consume()
            else:
                session.run("CREATE (:Campaign {campaign_id:$c, occurrences:0, total_tps:0, risk_score:0}) CREATE (:AttackEvent {event_id:$e, campaign_id:$c, occurrences:0, tps:0})",
                            e=event_id, c=f"C-{event_id}").consume()
        return event_id

    def markers(self, request_key, label="AppliedWrite"):
        with self.direct.session() as session:
            return session.run(f"MATCH (m:{label} {{request_key:$k}}) RETURN count(m) AS n", k=request_key).single()["n"]

    def occurrences(self, event_id):
        with self.direct.session() as session:
            return session.run("MATCH (e:AttackEvent {event_id:$e}) RETURN e.occurrences AS n", e=event_id).single()["n"]

    def payload(self, event_id):
        with self.direct.session() as session:
            return session.run("MATCH (e:AttackEvent {event_id:$e}) RETURN e.investigation_payload AS p", e=event_id).single()["p"]

    def campaign_occurrences(self, event_id):
        with self.direct.session() as session:
            return session.run("MATCH (c:Campaign {campaign_id:$c}) RETURN c.occurrences AS n", c=f"C-{event_id}").single()["n"]

    def technique_occurrences(self, event_id):
        with self.direct.session() as session:
            return session.run("MATCH (t:Technique {attack_id:$e}) RETURN t.occurrences AS n", e=event_id).single()["n"]

    def settled_absent(self, request_key, event_id, seconds=2.5):
        """True only if neither appears over the whole period. A snapshot wouldn't show an uncommitted transaction ending later."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.markers(request_key) or self.occurrences(event_id):
                return False
            time.sleep(0.1)
        return True

    def close(self):
        self.proxy.close()
        self.direct.close()


def describe(result):
    return {"outcome": result.outcome.value, "attempts": result.attempts, "code": result.exception_code, "type": result.exception_type,
            "nodes_created": (result.counters or {}).get("nodes_created")}


def restart_server(direct, *, wait=120):
    """Kills the disposable server dead and starts it again on the same volume. Only ever this experiment's container."""
    confirm_environment()
    _docker("kill", CONN["container"])
    _docker("start", CONN["container"])
    end = time.monotonic() + wait
    while time.monotonic() < end:
        try:
            with direct.session() as session:
                session.run("RETURN 1").consume()
            return True
        except Exception:
            time.sleep(1)
    return False
