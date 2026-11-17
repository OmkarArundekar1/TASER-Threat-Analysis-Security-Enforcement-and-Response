"""Builds the B3 v2 study variants from the current repository backend.
Study-only: nothing here touches the repository's production code.

usage: build_variants_v2.py <repo_backend> <variants_root>

Every variant = repository backend (without tests, logs, datasets, .env,
caches) + the SAME study harness patch. A and B then add their activity policy
and nothing else. .env is excluded on purpose: a variant has no way to pick up
production credentials, and the harness refuses any URI but the disposable one.
"""
import re
import shutil
import sys

src, root = sys.argv[1:3]
IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "tests", "logs", ".env", "misp_cache.json", "datasets",
)


def patch(path, pairs):
    text = open(path, encoding="utf-8").read()
    for old, new in pairs:
        pattern = "\n".join(
            re.escape(line.rstrip()) + "[ \t]*" if line.strip() else "[ \t]*"
            for line in old.split("\n")
        )
        count = len(re.findall(pattern, text))
        assert count == 1, (path, count, old[:70])
        text = re.sub(pattern, lambda _: new, text)
    open(path, "w", encoding="utf-8", newline="").write(text)


DIAGNOSTICS = '''class Diagnostics:
    """Study-only diagnostics, written unbuffered to <run>/diagnostics.log.
    A faulthandler watchdog is re-armed before every alert: if one alert (or
    the restart step, or the end-of-run work) takes longer than the limit, the
    stacks of all threads are written to the file and the process exits with
    status 1, so a stuck run cannot hang a batch."""

    def __init__(self, path, watchdog_seconds):
        self.watchdog_seconds = watchdog_seconds
        self.file = open(path, "a", buffering=1)
        self.started = time.monotonic()

    def log(self, message):
        line = f"{datetime.now(timezone.utc).isoformat()} +{time.monotonic() - self.started:9.3f}s {message}"
        self.file.write(line + "\\n")
        self.file.flush()
        print(line, file=sys.stderr, flush=True)

    def arm(self):
        faulthandler.dump_traceback_later(self.watchdog_seconds, exit=True, file=self.file)

    def beat(self, index):
        self.arm()
        if index % 500 == 0:
            self.log(f"PROGRESS index={index}")

    def disarm(self):
        faulthandler.cancel_dump_traceback_later()


'''

HARNESS = [
    ("import json\nimport os\n", "import faulthandler\nimport json\nimport os\n"),
    ('''def replay(ordered, process, manager, *, speed=None, maintenance=None,
           maintenance_interval=None, clock=None, sleep=time.sleep) -> ReplayResult:''',
     '''def replay(ordered, process, manager, *, speed=None, maintenance=None,
           maintenance_interval=None, clock=None, sleep=time.sleep,
           restart_at=None, restart=None, heartbeat=None) -> ReplayResult:'''),
    ('''            previous_time = event_time
            trace = {"index": index, "position": position}''',
     '''            previous_time = event_time
            if heartbeat is not None:
                heartbeat(index)
            if restart_at is not None and index == restart_at:
                restart()
            trace = {"index": index, "position": position}'''),
    ('''def run_cli(args):
    # Belt-and-braces before the pipeline is imported: no GNN, no reachable''',
     DIAGNOSTICS + '''REQUIRED_NEO4J_URI = "bolt://127.0.0.1:17690"


def run_cli(args):
    # Study guard, checked before anything can connect: only the disposable
    # instance is accepted, whatever the rest of the environment says.
    uri = os.environ.get("NEO4J_URI")
    if uri != REQUIRED_NEO4J_URI:
        raise SystemExit(f"Refusing to replay: NEO4J_URI is {uri!r}, expected {REQUIRED_NEO4J_URI!r}.")
    if os.path.exists(os.path.join(args.out, "manifest.json")):
        raise SystemExit(f"Refusing to replay: {args.out} already holds a completed run.")
    os.makedirs(args.out, exist_ok=True)
    diag = Diagnostics(os.path.join(args.out, "diagnostics.log"), args.watchdog_seconds)
    diag.log(f"RUN_START variant={os.environ.get('B3_VARIANT')} restart_at={args.restart_at} "
             f"watchdog_seconds={args.watchdog_seconds} pid={os.getpid()}")
    # Belt-and-braces before the pipeline is imported: no GNN, no reachable'''),
    ('''    started_at = datetime.now(timezone.utc).isoformat()
    result = replay(
        loaded.ordered, rsg.process_alert, campaign_manager,
        speed=args.speed, maintenance=maintenance, maintenance_interval=args.maintenance_interval,
    )''',
     '''    from neo4j_client import batch_update_duplicates

    def flush_buffer():
        pending = duplicate_buffer.snapshot()
        if pending:
            batch_update_duplicates(pending)
        return len(pending)

    def restart():
        """Simulated graceful listener restart: flush the duplicate buffer
        (as the listener does on stop), then drop all in-memory state.
        Neo4j is kept."""
        diag.log(f"RESTART_BEGIN before_index={args.restart_at} pending_fingerprints={duplicate_buffer.size()}")
        flushed = flush_buffer()
        diag.log(f"RESTART_FLUSH_DONE records={flushed}")
        log = campaign_manager.decision_log
        campaign_manager.reset()
        campaign_manager.decision_log = log
        dedup_engine.reset()
        event_clock.reset()
        diag.log("RESTART_END state_reset=campaign_manager,dedup_engine,event_clock")

    started_at = datetime.now(timezone.utc).isoformat()
    diag.arm()
    result = replay(
        loaded.ordered, rsg.process_alert, campaign_manager,
        speed=args.speed, maintenance=maintenance, maintenance_interval=args.maintenance_interval,
        restart_at=args.restart_at, restart=restart, heartbeat=diag.beat,
    )
    diag.arm()
    diag.log(f"REPLAY_LOOP_DONE alerts={len(result.traces)}")
    diag.log(f"FINAL_FLUSH_DONE records={flush_buffer()}")'''),
    ('''        "limit": args.limit, "speed": args.speed, "maintenance_interval": args.maintenance_interval,''',
     '''        "limit": args.limit, "speed": args.speed, "maintenance_interval": args.maintenance_interval,
        "variant": os.environ.get("B3_VARIANT"), "restart_at": args.restart_at,
        "watchdog_seconds": args.watchdog_seconds,'''),
    ('''        json.dump(loaded.rejected, f, indent=1)
    print(json.dumps(manifest, indent=1))''',
     '''        json.dump(loaded.rejected, f, indent=1)
    diag.disarm()
    diag.log(f"RUN_END view_sha256={manifest['view_sha256']}")
    print(json.dumps(manifest, indent=1))'''),
    ('''    parser.add_argument("--allow-nonempty", action="store_true")''',
     '''    parser.add_argument("--allow-nonempty", action="store_true")
    parser.add_argument("--restart-at", type=int, default=None)
    parser.add_argument("--watchdog-seconds", type=float, default=300.0)'''),
]

for name in ("CUR", "A", "B"):
    destination = f"{root}/{name}"
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(src, destination, ignore=IGNORE)
    patch(f"{destination}/replay/harness.py", HARNESS)

# ---------- Policy A: every accepted alert extends activity, in memory AND Neo4j
patch(f"{root}/A/campaign_manager.py", [
    ('''    def reset(self):''', '''    def touch(self, campaign_id, event_time):
        """Policy A: any accepted alert attached to the campaign extends its
        in-memory activity exactly as the Neo4j writers extend c.last_seen."""
        if campaign_id is None:
            return
        context = self.get_context_by_campaign_id(campaign_id)
        if context is not None:
            context.last_seen = _later(context.last_seen, _reference_time(event_time))

    def reset(self):'''),
])
patch(f"{root}/A/realtime_socgraph.py", [
    ('''        if trace is not None:
            trace.update(path="DUPLICATE_UNATTRIBUTED", fingerprint=fingerprint,''',
     '''        campaign_manager.touch((metadata or {}).get("campaign_id"), event_time.value)
        if trace is not None:
            trace.update(path="DUPLICATE_UNATTRIBUTED", fingerprint=fingerprint,'''),
    ('''    dedup_engine.register_event(
        fingerprint=fingerprint,
        campaign_id=campaign_id,
        event_id=event_id,
        event_time=event_time.value,
    )''', '''    dedup_engine.register_event(
        fingerprint=fingerprint,
        campaign_id=campaign_id,
        event_id=event_id,
        event_time=event_time.value,
    )
    campaign_manager.touch(campaign_id, event_time.value)'''),
    ('''            if trace is not None:
                trace.update(path="DUPLICATE", fingerprint=fingerprint,''',
     '''            campaign_manager.touch((metadata or {}).get("campaign_id"), event_time.value)
            if trace is not None:
                trace.update(path="DUPLICATE", fingerprint=fingerprint,'''),
])

# Without this, A would not be "all accepted alerts" in Neo4j: update_duplicate_event
# reaches the Campaign only through MATCH (e)-[:MATCHES]->(t), and UNKNOWN events have
# no MATCHES edge, so an UNKNOWN duplicate never updates Campaign.last_seen (verified
# by scripts/test_dup_update.py). Only last_seen is added here: Campaign.occurrences,
# total_tps and risk_score keep their existing behaviour.
patch(f"{root}/A/neo4j_client.py", [
    ('''                e.investigation_payload = $investigation_payload

            WITH e, dt

            MATCH (e)-[:MATCHES]->(t:Technique)''',
     '''                e.investigation_payload = $investigation_payload

            WITH e, dt

            OPTIONAL MATCH (activity:Campaign {campaign_id:e.campaign_id})
            SET activity.last_seen = CASE WHEN activity.last_seen IS NULL OR dt > activity.last_seen THEN dt ELSE activity.last_seen END

            WITH e, dt

            MATCH (e)-[:MATCHES]->(t:Technique)'''),
])

# ---------- Policy B: only resolved, non-duplicate alerts extend activity, in memory AND Neo4j
patch(f"{root}/B/neo4j_client.py", [
    ('''            c.last_seen = CASE WHEN c.last_seen IS NULL OR t > c.last_seen THEN t ELSE c.last_seen END,
            c.occurrences = coalesce(c.occurrences,0)+1
        WITH a, h, c, t''', '''            c.occurrences = coalesce(c.occurrences,0)+1
        WITH a, h, c, t'''),
    ('''                c.last_seen = CASE WHEN c.last_seen IS NULL OR dt > c.last_seen THEN dt ELSE c.last_seen END,
                c.occurrences = coalesce(c.occurrences,0) + 1,''',
     '''                c.occurrences = coalesce(c.occurrences,0) + 1,'''),
    ('''                c.last_seen = CASE WHEN c.last_seen IS NULL OR datetime(r.last_seen) > c.last_seen THEN datetime(r.last_seen) ELSE c.last_seen END,
                c.occurrences = coalesce(c.occurrences,0) + r.occurrences,''',
     '''                c.occurrences = coalesce(c.occurrences,0) + r.occurrences,'''),
])
print("variants built under", root)
