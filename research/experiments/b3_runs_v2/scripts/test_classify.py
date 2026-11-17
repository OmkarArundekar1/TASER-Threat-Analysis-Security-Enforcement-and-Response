"""Unit tests for classify_run.py: every status, from crafted run directories."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_run import STATUSES, classify  # noqa: E402

GOOD_DIAG = "RUN_START\nRESTART_BEGIN\nRESTART_FLUSH_DONE\nRESTART_END\nRUN_END x\n"
OOM = "after_run: state=exited oom_killed=true exit_code=137 restarts=0\nstop_rc=0\nafter_stop: state=exited oom_killed=true exit_code=137 restarts=0"


def full_run(run, restart_at=4500, alerts=2, diag=GOOD_DIAG, ok=True):
    files = {
        "manifest.json": json.dumps({"accepted": alerts, "restart_at": restart_at, "view_sha256": "abc"}),
        "view.json": "{}", "graph_snapshot.json": "{}", "rejected.json": "[]",
        "traces.jsonl": "\n".join(json.dumps({"index": i, "ok": ok}) for i in range(alerts)),
        "decisions.jsonl": '{"decision":"CREATE"}', "diagnostics.log": diag,
        "neo4j_resources.txt": "x", "run.log": "fine",
    }
    for name, text in files.items():
        with open(os.path.join(run, name), "w") as handle:
            handle.write(text)


def write(run, name, text):
    with open(os.path.join(run, name), "w") as handle:
        handle.write(text)


failures = []
seen = set()


def case(label, expected, setup, exit_code, timed_out=False, mode="full"):
    with tempfile.TemporaryDirectory() as run:
        setup(run)
        status, detail, _, _ = classify(run, exit_code, timed_out, mode)
    seen.add(status)
    verdict = "ok" if status == expected else "FAIL"
    if status != expected:
        failures.append(label)
    print(f"{verdict:4} {label:48} -> {status} {detail[:60] if detail else ''}")


none = lambda run: None  # noqa: E731
case("timeout wins, even with oom recorded", "EXPERIMENT_TIMEOUT", lambda r: write(r, "container_status", OOM), 124, True)
case("wrapper precheck", "PRECHECK_FAILED", lambda r: write(r, "wrapper_detail", "wrong platform"), 20)
case("wrapper container start", "CONTAINER_START_FAILED", lambda r: write(r, "wrapper_detail", "no start"), 21)
case("oom beats harness failure", "CONTAINER_OOM", lambda r: write(r, "container_status", OOM), 1)
case("oom beats success exit", "CONTAINER_OOM", lambda r: (full_run(r), write(r, "container_status", OOM)), 0)
case("watchdog marker", "WATCHDOG_TIMEOUT", lambda r: write(r, "diagnostics.log", "RUN_START\nTimeout (0:05:00)!\nThread 0x1\n"), 1)
case("harness nonzero", "HARNESS_FAILED", none, 1)
case("rc 0 but no artifacts", "HARNESS_FAILED", none, 0)
case("rc 0, alert failures", "HARNESS_FAILED", lambda r: full_run(r, ok=False), 0)
case("rc 0, restart markers missing", "HARNESS_FAILED", lambda r: full_run(r, diag="RUN_START\nRUN_END x\n"), 0)
case("rc 0, no RUN_END", "HARNESS_FAILED", lambda r: full_run(r, restart_at=None, diag="RUN_START\n"), 0)
case("rc 0, complete with restart", "COMPLETED", full_run, 0)
case("rc 0, complete without restart", "COMPLETED", lambda r: full_run(r, restart_at=None, diag="RUN_END x\n"), 0)
case("dry mode skips artifact checks", "COMPLETED", none, 0, mode="dry")
print("statuses exercised:", sorted(seen), "| missing:", sorted(set(STATUSES) - seen))
sys.exit(1 if failures or set(STATUSES) - seen else 0)
