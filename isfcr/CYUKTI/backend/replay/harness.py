"""
replay/harness.py
====================
Deterministic replay of archived Wazuh alerts through the real
realtime_socgraph.process_alert() pipeline (research/STEP2_EVENT_TIME.md).

Guarantees
  * Event time only: alerts are processed under the STRICT event-time policy
    (event_time.py); an alert without a valid, timezone-aware timestamp is
    rejected and recorded, never given processing time.
  * Deterministic order: alerts are sorted by (event time, position in the
    input file). Identical timestamps keep file order.
  * Speed-independent: `speed` only inserts wall-clock sleeps between alerts
    (event gap / speed); every lifecycle decision uses event time, and the
    optional background maintenance thread (like the live listener's) expires
    against the event-time watermark. A final maintenance sweep at the last
    watermark makes the end state independent of when that thread ran.
  * No outward side effects: MISP publication and the Shuffle webhook are
    replaced by in-memory recorders for the duration of the run.
  * Fresh database only: refuses to run if the target Neo4j already contains
    Campaign nodes (protects the production graph).

Usage (from isfcr/CYUKTI/backend, NEO4J_URI/USERNAME/PASSWORD pointing at a
disposable instance):

    python -m replay.harness --alerts ../data/raw_logs/alerts.json --out <dir> [--speed 500] [--maintenance-interval 0.05]
    python -m replay.harness --compare <dirA> <dirB>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from event_time import EventTimeError, POLICY_STRICT, parse_timestamp, raw_timestamp  # noqa: E402

COMPARED_DECISIONS_EXCLUDED = ("CLOSE_EXPIRED", "MAINTENANCE_EXPIRE")


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

@dataclass
class LoadedAlerts:
    ordered: list            # [(position, event_time, alert)] sorted
    rejected: list           # [{"position", "reason", "raw"}]
    total_lines: int
    sha256: str


def load_alerts(path: str) -> LoadedAlerts:
    digest = hashlib.sha256()
    accepted, rejected = [], []
    total = 0
    with open(path, "rb") as fh:
        for position, raw_line in enumerate(fh):
            digest.update(raw_line)
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            total += 1
            try:
                alert = json.loads(line)
            except json.JSONDecodeError:
                rejected.append({"position": position, "reason": "INVALID_JSON", "raw": None})
                continue
            if not isinstance(alert, dict):
                rejected.append({"position": position, "reason": "NOT_AN_OBJECT", "raw": None})
                continue
            try:
                event_time = parse_timestamp(raw_timestamp(alert))
            except EventTimeError as err:
                rejected.append({"position": position, "reason": err.reason,
                                 "raw": None if err.raw is None else str(err.raw)})
                continue
            accepted.append((position, event_time, alert))
    accepted.sort(key=lambda item: (item[1], item[0]))
    return LoadedAlerts(accepted, rejected, total, digest.hexdigest())


# --------------------------------------------------------------------------
# Side-effect isolation
# --------------------------------------------------------------------------

class _RecordingSync:
    """Stands in for misp_sync.MISPSync: records what WOULD be published."""

    def __init__(self, real):
        self._real = real
        self.would_publish = []

    def publish_campaign(self, incident):
        decision = self._real.should_publish(incident)
        self.would_publish.append({"campaign_id": incident.campaign_id, "publish": bool(decision)})
        return {"success": False, "action": "replay-suppressed", "reason": "replay harness: no MISP side effects"}


class _NoNetwork:
    def __init__(self):
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("replay harness: outbound HTTP disabled")


# --------------------------------------------------------------------------
# Replay core (pure w.r.t. the injected pipeline -- unit-testable)
# --------------------------------------------------------------------------

@dataclass
class ReplayResult:
    traces: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    wall_seconds: float = 0.0


def replay(ordered, process, manager, *, speed=None, maintenance=None,
           maintenance_interval=None, clock=None, sleep=time.sleep) -> ReplayResult:
    """ordered: [(position, event_time, alert)] already sorted.
    process(alert, trace=dict, event_time_policy=...) -> bool.
    manager: CampaignManager whose decision_log is captured.
    maintenance(reference_time): background expiry; run on a wall-clock thread
    every maintenance_interval seconds with the event-time watermark, and once
    more at the end."""
    from event_time import event_clock as default_clock
    clock = clock or default_clock

    manager.decision_log = []
    result = ReplayResult()
    stop = threading.Event()

    def _maintenance_loop():
        while not stop.wait(maintenance_interval):
            watermark = clock.watermark
            if watermark is not None:
                maintenance(watermark)

    worker = None
    if maintenance is not None and maintenance_interval:
        worker = threading.Thread(target=_maintenance_loop, daemon=True)
        worker.start()

    started = time.monotonic()
    previous_time = None
    try:
        for index, (position, event_time, alert) in enumerate(ordered):
            if speed and previous_time is not None:
                gap = (event_time - previous_time).total_seconds()
                if gap > 0:
                    sleep(gap / speed)
            previous_time = event_time
            trace = {"index": index, "position": position}
            ok = process(alert, trace=trace, event_time_policy=POLICY_STRICT)
            trace["ok"] = bool(ok)
            result.traces.append(trace)
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=10)
    if maintenance is not None and clock.watermark is not None:
        maintenance(clock.watermark)   # deterministic end-of-replay sweep
    result.wall_seconds = round(time.monotonic() - started, 3)
    result.decisions = list(manager.decision_log)
    manager.decision_log = None
    return result


# --------------------------------------------------------------------------
# Canonicalisation (random CAMP_/OP_ ids -> order-of-first-appearance labels)
# --------------------------------------------------------------------------

def canonical_campaign_labels(traces) -> dict:
    labels = {}
    for trace in traces:
        cid = trace.get("campaign_id")
        if cid and cid not in labels:
            labels[cid] = f"C{len(labels) + 1:04d}"
    return labels


def comparable_view(traces, decisions, graph_snapshot) -> dict:
    """Everything that must be identical across runs of the same input."""
    labels = canonical_campaign_labels(traces)
    for c in graph_snapshot.get("campaigns", []):          # campaigns never named in a trace
        if c["campaign_id"] not in labels:
            labels[c["campaign_id"]] = f"C_UNTRACED_{c['campaign_id']}"
    op_labels = {}
    for op in sorted(graph_snapshot.get("operations", []),
                     key=lambda o: min((labels.get(c, "~") for c in o["campaign_ids"]), default="~")):
        op_labels[op["operation_id"]] = f"O{len(op_labels) + 1:04d}"

    assignments = [
        {"index": t["index"], "position": t["position"], "path": t.get("path"),
         "campaign": labels.get(t.get("campaign_id")), "event_time": t.get("event_time"),
         "late": t.get("late"), "event_time_error": t.get("event_time_error")}
        for t in traces
    ]
    lifecycle = [
        {k: (labels.get(v, v) if k == "campaign_id" else v) for k, v in d.items()}
        for d in decisions if d["decision"] not in COMPARED_DECISIONS_EXCLUDED
    ]
    campaigns = sorted(
        ({**{k: v for k, v in c.items() if k != "campaign_id"}, "campaign": labels[c["campaign_id"]]}
         for c in graph_snapshot.get("campaigns", [])),
        key=lambda c: c["campaign"],
    )
    operations = sorted(
        ({"operation": op_labels[o["operation_id"]],
          "campaigns": sorted(labels.get(c, c) for c in o["campaign_ids"]),
          "status": o.get("status"), "closed_at": o.get("closed_at")}
         for o in graph_snapshot.get("operations", [])),
        key=lambda o: o["operation"],
    )
    return {
        "assignments": assignments,
        "lifecycle_decisions": lifecycle,
        "campaigns": campaigns,
        "transitions": graph_snapshot.get("transitions", []),
        "operations": operations,
    }


def view_hash(view) -> str:
    return hashlib.sha256(json.dumps(view, sort_keys=True, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------
# Neo4j snapshot (read-only)
# --------------------------------------------------------------------------

def _iso(value):
    if value is None:
        return None
    if hasattr(value, "to_native"):
        value = value.to_native()
    return value.astimezone(timezone.utc).isoformat() if isinstance(value, datetime) else str(value)


def snapshot_graph(driver) -> dict:
    def _read(tx):
        campaigns = tx.run(
            "MATCH (c:Campaign) OPTIONAL MATCH (c)-[:HAS_EVENT]->(e:AttackEvent) "
            "WITH c, e ORDER BY e.first_seen, e.ingested_at, e.event_id "
            "RETURN c.campaign_id AS campaign_id, c.attacker_ip AS attacker_ip, c.victim_ip AS victim_ip, "
            "c.status AS status, c.first_seen AS first_seen, c.last_seen AS last_seen, "
            "c.closed_at AS closed_at, c.reopened_count AS reopened_count, "
            "collect(coalesce(e.attack_id, 'UNKNOWN')) AS sequence, count(e) AS events").data()
        transitions = tx.run(
            "MATCH (a:Technique)-[r:NEXT_TECHNIQUE]->(b:Technique) "
            "RETURN a.attack_id AS src, b.attack_id AS dst, r.count AS count ORDER BY src, dst").data()
        operations = tx.run(
            "MATCH (o:Operation) OPTIONAL MATCH (o)-[:HAS_CAMPAIGN]->(c:Campaign) "
            "RETURN o.operation_id AS operation_id, o.status AS status, o.closed_at AS closed_at, "
            "collect(c.campaign_id) AS campaign_ids").data()
        return campaigns, transitions, operations

    with driver.session() as session:
        campaigns, transitions, operations = session.execute_read(_read)
    for c in campaigns:
        for k in ("first_seen", "last_seen", "closed_at"):
            c[k] = _iso(c[k])
    for o in operations:
        o["closed_at"] = _iso(o["closed_at"])
    return {"campaigns": campaigns, "transitions": transitions, "operations": operations}


def _require_fresh_database(driver, allow_nonempty: bool):
    with driver.session() as session:
        n = session.execute_read(lambda tx: tx.run("MATCH (c:Campaign) RETURN count(c) AS n").single()["n"])
    if n and not allow_nonempty:
        raise SystemExit(
            f"Refusing to replay: target database already holds {n} Campaign nodes. "
            "Replay must run against a fresh, disposable database (never the production graph)."
        )


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run_cli(args):
    # Belt-and-braces before the pipeline is imported: no GNN, no reachable
    # MISP (publication is additionally replaced by _RecordingSync below).
    os.environ.setdefault("GNN_ENABLED", "false")
    os.environ["MISP_URL"] = os.environ.get("REPLAY_MISP_URL", "https://127.0.0.1:9")
    import realtime_socgraph as rsg
    from campaign_manager import campaign_manager
    from operation_manager import operation_manager
    from dedup_engine import dedup_engine, duplicate_buffer
    from event_time import event_clock
    from neo4j_client import driver
    import config

    _require_fresh_database(driver, args.allow_nonempty)
    loaded = load_alerts(args.alerts)
    if args.limit:
        loaded.ordered = loaded.ordered[: args.limit]

    campaign_manager.reset()
    dedup_engine.reset()
    event_clock.reset()
    duplicate_buffer.snapshot()

    recording_sync = _RecordingSync(rsg.sync)
    rsg.sync = recording_sync
    no_network = _NoNetwork()
    rsg.requests = no_network

    def maintenance(reference_time):
        campaign_manager.expire_active_campaigns(reference_time=reference_time)
        operation_manager.expire_active_operations(reference_time=reference_time)

    started_at = datetime.now(timezone.utc).isoformat()
    result = replay(
        loaded.ordered, rsg.process_alert, campaign_manager,
        speed=args.speed, maintenance=maintenance, maintenance_interval=args.maintenance_interval,
    )
    snapshot = snapshot_graph(driver)
    view = comparable_view(result.traces, result.decisions, snapshot)

    os.makedirs(args.out, exist_ok=True)
    paths = {name: os.path.join(args.out, name) for name in (
        "manifest.json", "view.json", "traces.jsonl", "decisions.jsonl", "graph_snapshot.json", "rejected.json")}
    manifest = {
        "harness": "replay/harness.py", "started_at": started_at,
        "git_commit": _git_commit(), "input": os.path.abspath(args.alerts), "input_sha256": loaded.sha256,
        "input_lines": loaded.total_lines, "accepted": len(loaded.ordered), "rejected": len(loaded.rejected),
        "limit": args.limit, "speed": args.speed, "maintenance_interval": args.maintenance_interval,
        "event_time_policy": POLICY_STRICT, "campaign_timeout": config.CAMPAIGN_TIMEOUT,
        "dedup_window": config.DEDUP_WINDOW, "operation_timeout": config.OPERATION_TIMEOUT,
        "wall_seconds": result.wall_seconds, "neo4j_uri": os.environ.get("NEO4J_URI"),
        "misp_publications_suppressed": len(recording_sync.would_publish),
        "misp_would_publish": sum(1 for p in recording_sync.would_publish if p["publish"]),
        "outbound_http_blocked": no_network.calls,
        "view_sha256": view_hash(view),
        "counts": {
            "campaigns": len(view["campaigns"]), "operations": len(view["operations"]),
            "transitions": len(view["transitions"]),
            "paths": _count(t.get("path") for t in result.traces),
        },
    }
    with open(paths["manifest.json"], "w") as f:
        json.dump(manifest, f, indent=1)
    with open(paths["view.json"], "w") as f:
        json.dump(view, f, indent=1, sort_keys=True, default=str)
    with open(paths["traces.jsonl"], "w") as f:
        for t in result.traces:
            f.write(json.dumps(t, default=str) + "\n")
    with open(paths["decisions.jsonl"], "w") as f:
        for d in result.decisions:
            f.write(json.dumps(d, default=str) + "\n")
    with open(paths["graph_snapshot.json"], "w") as f:
        json.dump(snapshot, f, indent=1, default=str)
    with open(paths["rejected.json"], "w") as f:
        json.dump(loaded.rejected, f, indent=1)
    print(json.dumps(manifest, indent=1))


def _count(values):
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def compare_runs(dir_a, dir_b) -> dict:
    views = []
    for d in (dir_a, dir_b):
        with open(os.path.join(d, "view.json")) as f:
            views.append(json.load(f))
    a, b = views
    report = {"identical": view_hash(a) == view_hash(b), "sections": {}}
    for section in a:
        same = json.dumps(a[section], sort_keys=True) == json.dumps(b[section], sort_keys=True)
        entry = {"identical": same, "len_a": len(a[section]), "len_b": len(b[section])}
        if not same and isinstance(a[section], list):
            for i, (x, y) in enumerate(zip(a[section], b[section])):
                if x != y:
                    entry["first_difference"] = {"at": i, "a": x, "b": y}
                    break
        report["sections"][section] = entry
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--alerts")
    parser.add_argument("--out")
    parser.add_argument("--speed", type=float, default=None, help="event-seconds per wall-second; default: no pacing")
    parser.add_argument("--maintenance-interval", type=float, default=None,
                        help="run background expiry every N wall-seconds (like the live listener)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-nonempty", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    args = parser.parse_args(argv)
    if args.compare:
        report = compare_runs(*args.compare)
        print(json.dumps(report, indent=1, default=str))
        return 0 if report["identical"] else 1
    if not (args.alerts and args.out):
        parser.error("--alerts and --out are required unless --compare is used")
    run_cli(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
