"""Watchdog check, no database. Runs the variant harness's real Diagnostics and
replay loop with a process() that calls a dedup engine.
usage (cwd = a variant): test_watchdog.py <dedup_module.py> <diagnostics.log>
The pre-fix module reproduces the real deadlock: the watchdog must dump the stack
and exit 1. The fixed module must finish and exit 0."""
import importlib.util
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

os.environ["NEO4J_URI"] = "bolt://127.0.0.1:9"
sys.path.insert(0, os.getcwd())
module_path, diag_path = sys.argv[1:3]

spec = importlib.util.spec_from_file_location("dedup_under_test", module_path)
dedup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dedup)
dedup.find_recent_duplicate = lambda fingerprint, reference_time=None: {"event_id": "e", "campaign_id": "c", "last_seen": None}
engine = dedup.DeduplicationEngine()

from replay.harness import Diagnostics, replay  # noqa: E402

diag = Diagnostics(diag_path, 2.0)
diag.log("WATCHDOG_TEST_START")


def process(alert, trace=None, event_time_policy=None):
    trace.update(path="TEST", result=str(engine.is_duplicate("a", "v", "T1", "r", "g")[0]))
    return True


T = datetime(2026, 1, 1, tzinfo=timezone.utc)
result = replay([(0, T, {})], process, SimpleNamespace(decision_log=None), heartbeat=diag.beat)
diag.disarm()
diag.log(f"WATCHDOG_TEST_END traces={result.traces}")
