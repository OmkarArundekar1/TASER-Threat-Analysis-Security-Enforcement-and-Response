"""Classifies one finished experiment and writes its status files atomically.

usage: classify_run.py <run_dir> --exit-code N --mode full|dry|test
           [--timed-out] [--experiment-timeout S] [--started T] [--ended T]
           [--force-status STATUS --force-detail TEXT]

Writes:
  status       the classification, one word
  exit_status  the actual process exit code, separately
  status.json  both, plus stage, detail, container and cleanup state, checks

Priority: EXPERIMENT_TIMEOUT > PRECHECK_FAILED > CONTAINER_START_FAILED >
CONTAINER_OOM > WATCHDOG_TIMEOUT > HARNESS_FAILED > COMPLETED.
COMPLETED in full mode also requires complete, consistent artifacts.
"""
import argparse
import json
import os
import re

STATUSES = ("PRECHECK_FAILED", "CONTAINER_START_FAILED", "HARNESS_FAILED", "WATCHDOG_TIMEOUT",
            "EXPERIMENT_TIMEOUT", "CONTAINER_OOM", "COMPLETED")
REQUIRED = ("manifest.json", "view.json", "traces.jsonl", "decisions.jsonl", "graph_snapshot.json",
            "rejected.json", "diagnostics.log", "neo4j_resources.txt")
RESTART_MARKERS = ("RESTART_BEGIN", "RESTART_FLUSH_DONE", "RESTART_END")


def read(run, name):
    try:
        with open(os.path.join(run, name), errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def write_atomic(run, name, text):
    tmp = os.path.join(run, f".{name}.tmp.{os.getpid()}")
    with open(tmp, "w") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, os.path.join(run, name))


def artifact_problems(run):
    """Empty list means the run's artifacts are complete and self-consistent."""
    problems = [f"missing:{name}" for name in REQUIRED if not os.path.isfile(os.path.join(run, name))]
    if problems:
        return problems, {}
    checks = {}
    try:
        manifest = json.loads(read(run, "manifest.json"))
        for name in ("view.json", "graph_snapshot.json", "rejected.json"):
            json.loads(read(run, name))
        traces = [json.loads(line) for line in read(run, "traces.jsonl").splitlines()]
        decisions = read(run, "decisions.jsonl").splitlines()
        [json.loads(line) for line in decisions]
    except ValueError as err:
        return [f"invalid_json:{err}"], {}
    diagnostics = read(run, "diagnostics.log")
    log = read(run, "run.log")
    checks.update(
        accepted=manifest.get("accepted"), traces=len(traces), decisions=len(decisions),
        alerts_ok=sum(1 for t in traces if t.get("ok")),
        process_alert_errors=log.count("PROCESS ALERT ERROR"), tracebacks=log.count("Traceback (most recent"),
        run_end_marker="RUN_END" in diagnostics, view_sha256=manifest.get("view_sha256"),
    )
    if checks["traces"] != checks["accepted"]:
        problems.append(f"traces {checks['traces']} != accepted {checks['accepted']}")
    if checks["alerts_ok"] != checks["traces"]:
        problems.append(f"alerts not ok: {checks['traces'] - checks['alerts_ok']}")
    if checks["process_alert_errors"]:
        problems.append(f"process_alert errors: {checks['process_alert_errors']}")
    if not checks["run_end_marker"]:
        problems.append("no RUN_END marker")
    if manifest.get("restart_at") is not None:
        checks["restart_markers"] = {m: diagnostics.count(m) for m in RESTART_MARKERS}
        missing = [m for m, n in checks["restart_markers"].items() if n != 1]
        if missing:
            problems.append(f"restart markers not exactly once: {missing}")
    return problems, checks


def classify(run, exit_code, timed_out, mode, force_status=None, force_detail=""):
    container = read(run, "container_status")
    diagnostics = read(run, "diagnostics.log")
    detail = force_detail or read(run, "wrapper_detail")
    problems, checks = [], {}
    if force_status:
        status = force_status
    elif timed_out:
        status = "EXPERIMENT_TIMEOUT"
    elif exit_code == 20:
        status = "PRECHECK_FAILED"
    elif exit_code == 21:
        status = "CONTAINER_START_FAILED"
    elif re.search(r"oom_killed=true", container):
        status = "CONTAINER_OOM"
    elif re.search(r"^Timeout \(", diagnostics, re.M):
        status = "WATCHDOG_TIMEOUT"
    elif exit_code != 0:
        status = "HARNESS_FAILED"
    else:
        if mode == "full":
            problems, checks = artifact_problems(run)
        status = "HARNESS_FAILED" if problems else "COMPLETED"
        if problems:
            detail = "; ".join(problems)
    return status, detail, problems, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--mode", choices=("full", "dry", "test"), required=True)
    parser.add_argument("--timed-out", action="store_true")
    parser.add_argument("--experiment-timeout", type=int)
    parser.add_argument("--started")
    parser.add_argument("--ended")
    parser.add_argument("--force-status", choices=STATUSES)
    parser.add_argument("--force-detail", default="")
    args = parser.parse_args()

    status, detail, problems, checks = classify(
        args.run_dir, args.exit_code, args.timed_out, args.mode, args.force_status, args.force_detail)
    record = {
        "status": status, "exit_code": args.exit_code, "mode": args.mode,
        "stage_at_end": read(args.run_dir, "stage") or None, "detail": detail or None,
        "timed_out": args.timed_out, "experiment_timeout_seconds": args.experiment_timeout,
        "started": args.started, "ended": args.ended,
        "container_status": read(args.run_dir, "container_status") or None,
        "cleanup_status": read(args.run_dir, "cleanup_status") or None,
        "watchdog_fired": bool(re.search(r"^Timeout \(", read(args.run_dir, "diagnostics.log"), re.M)),
        "artifact_problems": problems, "checks": checks,
    }
    write_atomic(args.run_dir, "status.json", json.dumps(record, indent=1))
    write_atomic(args.run_dir, "exit_status", str(args.exit_code))
    write_atomic(args.run_dir, "status", status)
    print(status)


if __name__ == "__main__":
    main()
