"""Offline replay through the REAL listener source: AlertFileHandler.read_new_lines, accept_alert and queue_worker, from the
files in LISTENER_DIR, so the same script measures the code before and after the fix. Starts nothing live. start_listener()
is never called. Everything that could reach Neo4j, MISP or campaign logic is a stub, and process_alert only records what
reached it. ALERT_FILE, offset.dat and logs all point into a temporary directory, which is removed afterwards.

Prints counts only, never alert content. Run with stderr discarded: the real logging setup logs each alert.
usage (from isfcr/CYUKTI/backend): python -B replay_listener.py LISTENER_DIR ALERT_FILE"""
import ast
import collections
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import types

BACKEND = os.getcwd()
LISTENER_DIR, SAMPLE = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
assert not SAMPLE.startswith("/var/ossec")
sys.dont_write_bytecode = True
sys.path.insert(0, BACKEND)
signal.signal = lambda *args: None


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_listener(work):
    """A fresh copy of the listener, with empty caches, as a restarted process has."""
    for name in ("realtime_socgraph", "dedup_engine", "neo4j_client", "campaign_manager", "operation_manager", "event_time"):
        stub = types.ModuleType(name)
        stub.__dict__.update({"process_alert": lambda alert: True, "duplicate_flush_worker": object(), "duplicate_buffer": object(),
                              "batch_update_duplicates": None, "campaign_manager": object(), "operation_manager": object(), "event_clock": object()})
        sys.modules[name] = stub
    sys.modules["utils"] = load_file("listener_utils_replayed", os.path.join(LISTENER_DIR, "utils.py"))
    os.chdir(work)
    module = load_file("wazuh_listener_replayed", os.path.join(LISTENER_DIR, "wazuh_listener.py"))
    module.__file__ = os.path.join(work, "wazuh_listener.py")
    module.ALERT_FILE = os.path.join(work, "alerts.json")
    logging.getLogger().handlers[:] = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
    return module


class Run:
    """One listener process: its real worker thread drains its real queue into `processed`."""

    def __init__(self, work):
        self.listener = load_listener(work)
        self.processed = []
        self.counts = collections.Counter()
        self.refused = []
        real_accept = self.listener.accept_alert

        def counted(alert):
            try:
                accepted = real_accept(alert)
            except Exception as error:
                self.counts[f"failed:{type(error).__name__}"] += 1
                raise
            self.counts["accepted" if accepted else "duplicate"] += 1
            if not accepted:
                self.refused.append(alert)
            status = (alert.get("source_identity") or {}).get("status")
            if status:
                self.counts[f"status:{status}"] += 1
            return accepted
        self.listener.accept_alert = counted
        self.listener.process_alert = lambda alert: self.processed.append(alert) or True
        self.worker = threading.Thread(target=self.listener.queue_worker, daemon=True)
        self.worker.start()
        self.handler = self.listener.AlertFileHandler()

    def read_all(self):
        size = os.path.getsize(self.listener.ALERT_FILE)
        started = time.perf_counter()
        polls = 0
        while self.handler.offset < size:
            self.handler.read_new_lines()
            polls += 1
            if polls % 50 == 0:
                time.sleep(0.001)
        self.listener.alert_queue.join()
        return time.perf_counter() - started, polls

    def stop(self):
        self.listener.stop_event.set()
        self.worker.join(5)


def legacy_key(alert):
    """generate_alert_id as it was before Phase 2.6C, verbatim in effect, to name the alerts it collapsed."""
    data = alert.get("data", {})
    raw = (f"{alert.get('timestamp', '')}|{alert.get('rule', {}).get('id', '')}|{alert.get('agent', {}).get('id', '')}|"
           f"{data.get('srcip') or alert.get('srcip', '')}|{data.get('dstip') or alert.get('dstip', '')}|{alert.get('location', '')}")
    return hashlib.sha256(raw.encode()).hexdigest()


def lift_pipeline():
    tree = ast.parse(open(os.path.join(BACKEND, "realtime_socgraph.py"), encoding="utf-8").read())
    names = {"extract_iocs", "extract_audit_command", "should_ignore_audit_command", "IGNORED_AUDIT_COMMANDS"}
    picked = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names) or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in names)]
    namespace = {"os": os, "logger": type("L", (), {"debug": staticmethod(lambda *a: None)})}
    exec(compile(ast.Module(picked, []), "realtime_socgraph.py", "exec"), namespace)
    return namespace


PIPELINE = lift_pipeline()


def modeled_routes(stream):
    """A MODEL of later routing, not a production fact: the same fingerprint and 90 s window as dedup_engine, with no Neo4j."""
    from datetime import datetime
    counts, last = collections.Counter(), {}
    for alert in stream:
        rule = alert.get("rule")
        if not rule or PIPELINE["should_ignore_audit_command"](alert):
            counts["skipped"] += 1
            continue
        native = (rule.get("mitre") or {}).get("id") or []
        attacker, victim = PIPELINE["extract_iocs"](alert)
        fingerprint = f"{attacker}|{victim}|{native[0] if native else None}|{rule.get('id')}|{alert.get('agent', {}).get('id', 'UNKNOWN')}"
        now = datetime.strptime(alert["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
        seen = last.get(fingerprint)
        kind = "attributed" if native else "unattributed"
        if seen is not None and now - seen <= 90:
            counts[f"{kind} duplicate"] += 1
            last[fingerprint] = max(seen, now)
        else:
            counts[f"{kind} new event"] += 1
            last[fingerprint] = now
    return dict(sorted(counts.items()))


def scenario(lines, name="alerts.json"):
    work = tempfile.mkdtemp(prefix="cyukti-p26c-replay-")
    with open(os.path.join(work, name), "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    with open(os.path.join(work, "offset.dat"), "w") as handle:
        handle.write("0")
    return work


def report(label, run, took=None):
    counts = dict(sorted(run.counts.items()))
    print(f"{label}: {counts}" + (f"; {took:.3f} s" if took is not None else ""))


raw_lines = open(SAMPLE, encoding="utf-8", errors="replace").readlines()
alerts = [json.loads(line) for line in raw_lines if line.strip()]
print("input sha256", hashlib.sha256(open(SAMPLE, "rb").read()).hexdigest(), "alerts", len(alerts))
for name in ("utils.py", "wazuh_listener.py"):
    print(f"listener source {name} sha256", hashlib.sha256(open(os.path.join(LISTENER_DIR, name), "rb").read()).hexdigest())
groups = collections.defaultdict(list)
for alert in alerts:
    groups[legacy_key(alert)].append(alert)
previously_refused = [a for members in groups.values() if len(members) > 1 for a in members[1:]]
different_ioc = [a for members in groups.values() if len(members) > 1 for a in members[1:] if PIPELINE["extract_iocs"](a) != PIPELINE["extract_iocs"](members[0])]
print(f"the legacy key groups {len(previously_refused)} alerts under an earlier one; of those {len(different_ioc)} name a different attacker or victim")
print("modeled routes, all 9,015:", modeled_routes(alerts))

print("\n=== first pass and second pass, one process")
work = scenario(raw_lines)
run = Run(work)
took, polls = run.read_all()
report("first pass", run, took)
print(f"reached process_alert {len(run.processed)}; distinct Wazuh ids {len({a.get('id') for a in run.processed})}; polls {polls}; alerts per second {len(alerts) / took:.0f}")
ids = {a.get("id") for a in run.processed}
print(f"of the {len(previously_refused)} previously refused: preserved {sum(a['id'] in ids for a in previously_refused)}, distinct ids {len({a['id'] for a in previously_refused})}")
print(f"of the {len(different_ioc)} with a different attacker or victim: preserved {sum(a['id'] in ids for a in different_ioc)}")
print("modeled routes, what reached process_alert:", modeled_routes(run.processed))
first_counts = collections.Counter(run.counts)
run.counts.clear()
run.handler.offset = 0
took, polls = run.read_all()
report("second pass, same file, same process", run, took)
print(f"reached process_alert in total {len(run.processed)}; alerts per second {len(alerts) / took:.0f}")
saved = open(os.path.join(work, "offset.dat")).read()
print(f"offset saved equals file size: {int(saved) == os.path.getsize(os.path.join(work, 'alerts.json'))}")
run.stop()

print("\n=== restart, after a clean run: the saved offset is at the end")
restart = Run(work)
restart.read_all()
report("restart, offset at the end", restart)
print(f"reached process_alert {len(restart.processed)}")
restart.stop()

print("\n=== restart, as after a crash before the offset was saved: the offset says 0")
with open(os.path.join(work, "offset.dat"), "w") as handle:
    handle.write("0")
crash = Run(work)
crash.read_all()
report("restart, offset 0", crash)
print(f"reached process_alert {len(crash.processed)}; distinct Wazuh ids {len({a.get('id') for a in crash.processed})}. These are redelivered: the seen cache died with the process")
crash.stop()
shutil.rmtree(work)

print("\n=== the same alerts under a second manager name: one file holding both")
renamed = [json.dumps({**a, "manager": {"name": "second-manager"}}) + "\n" for a in alerts]
work = scenario(raw_lines + renamed)
run = Run(work)
run.read_all()
report("two manager names", run)
print(f"reached process_alert {len(run.processed)}")
run.stop()
shutil.rmtree(work)

for label, change in (("id removed", lambda a: {k: v for k, v in a.items() if k != "id"}), ("id empty", lambda a: {**a, "id": ""}),
                      ("id not a string", lambda a: {**a, "id": 1773199644.1234}), ("id malformed", lambda a: {**a, "id": "not-an-id"}),
                      ("manager removed", lambda a: {k: v for k, v in a.items() if k != "manager"})):
    print(f"\n=== every alert with {label}, read twice")
    lines = [json.dumps(change(a)) + "\n" for a in alerts]
    work = scenario(lines)
    run = Run(work)
    run.read_all()
    report("first pass", run)
    reached = len(run.processed)
    run.counts.clear()
    run.handler.offset = 0
    run.read_all()
    report("second pass", run)
    print(f"reached process_alert: first pass {reached}, after second pass {len(run.processed)}")
    run.stop()
    shutil.rmtree(work)
print("\n=== lines that can never be alerts, between real ones: one read")
for label, poison in (("truncated line ending in a newline", '{"id": "1.2", "trunc\n'), ("a JSON array", "[1, 2]\n"), ("a JSON number", "5\n"), ("malformed, ending in a brace", "{not json}\n")):
    good = [json.dumps(alerts[0]) + "\n", poison, json.dumps(alerts[1]) + "\n"]
    work = scenario(good)
    run = Run(work)
    for _ in range(3):
        run.handler.read_new_lines()
    run.listener.alert_queue.join()
    size = os.path.getsize(run.listener.ALERT_FILE)
    print(f"{label}: reached process_alert {len(run.processed)} of 2 real alerts; reader at the end of the file: {run.handler.offset == size}")
    run.stop()
    shutil.rmtree(work)
os.chdir(BACKEND)
print("\nno temporary directory left:", not [d for d in os.listdir(tempfile.gettempdir()) if d.startswith("cyukti-p26c-replay-")])
