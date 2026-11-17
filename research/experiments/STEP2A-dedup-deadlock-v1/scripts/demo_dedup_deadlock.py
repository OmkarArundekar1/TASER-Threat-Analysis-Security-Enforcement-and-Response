"""READ-ONLY isolation check: no replay, no database, no repo change.
Imports dedup_engine from <backend_dir>, stubs find_recent_duplicate to report a
recent event (what Neo4j returns right after a listener restart, when the
in-memory dedup cache is empty), and calls is_duplicate in a daemon thread.
usage: demo_dedup_deadlock.py <backend_dir>"""
import faulthandler, os, sys, threading

backend = sys.argv[1]
os.environ["NEO4J_URI"] = "bolt://127.0.0.1:9"      # closed port: production unreachable
sys.path.insert(0, backend)
import dedup_engine as d  # noqa: E402

d.find_recent_duplicate = lambda fingerprint, reference_time=None: {
    "event_id": "evt-1", "campaign_id": "CAMP_1", "last_seen": None}
engine = d.DeduplicationEngine()
result = {}


def call():
    result["value"] = engine.is_duplicate("1.2.3.4", "10.0.0.5", "T1110", "5716", "001")


t = threading.Thread(target=call, daemon=True)
t.start()
t.join(5)
print("lock type:", type(engine.lock).__name__)
print("returned within 5 s:", not t.is_alive(), "| result:", result.get("value"))
if t.is_alive():
    print("DEADLOCKED. Stack of the blocked thread:")
    faulthandler.dump_traceback(all_threads=True, file=sys.stdout)
os._exit(0)
