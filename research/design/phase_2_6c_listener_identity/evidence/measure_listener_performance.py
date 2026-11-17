"""Benchmarks the real listener functions from LISTENER_DIR, offline, on copies of the sample in a temporary directory. process_alert
is a stub, so this measures the listener and not the pipeline behind it. Where a service time is given, it's a synthetic sleep.
Benchmarking at W07U's observed peak isn't a migration of W07U. Prints numbers only.
usage (from isfcr/CYUKTI/backend): python -B measure_listener_performance.py LISTENER_DIR ALERT_FILE [--paced]"""
import collections
import hashlib
import importlib.util
import json
import logging
import os
import platform
import shutil
import signal
import statistics
import sys
import tempfile
import threading
import time
import tracemalloc
import types

BACKEND = os.getcwd()
LISTENER_DIR, SAMPLE = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
PACED = "--paced" in sys.argv
assert not SAMPLE.startswith("/var/ossec")
sys.dont_write_bytecode = True
sys.path.insert(0, BACKEND)
signal.signal = lambda *args: None
PEAK = 377


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_listener(work, log=True):
    for name in ("realtime_socgraph", "dedup_engine", "neo4j_client", "campaign_manager", "operation_manager", "event_time"):
        stub = types.ModuleType(name)
        stub.__dict__.update({"process_alert": lambda alert: True, "duplicate_flush_worker": object(), "duplicate_buffer": object(),
                              "batch_update_duplicates": None, "campaign_manager": object(), "operation_manager": object(), "event_clock": object()})
        sys.modules[name] = stub
    sys.modules["utils"] = load_file("listener_utils_measured", os.path.join(LISTENER_DIR, "utils.py"))
    os.chdir(work)
    module = load_file("wazuh_listener_measured", os.path.join(LISTENER_DIR, "wazuh_listener.py"))
    module.__file__ = os.path.join(work, "wazuh_listener.py")
    module.ALERT_FILE = os.path.join(work, "alerts.json")
    root = logging.getLogger()
    root.handlers[:] = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    root.setLevel(logging.INFO if log else logging.CRITICAL)
    return module


def fresh(lines=()):
    work = tempfile.mkdtemp(prefix="cyukti-p26c-perf-")
    with open(os.path.join(work, "alerts.json"), "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    with open(os.path.join(work, "offset.dat"), "w") as handle:
        handle.write("0")
    return work


raw = open(SAMPLE, encoding="utf-8").readlines()
alerts = [json.loads(line) for line in raw if line.strip()]
print(f"python {platform.python_version()}, {platform.platform()}; {len(alerts)} alerts")
print("listener", hashlib.sha256(open(os.path.join(LISTENER_DIR, "wazuh_listener.py"), "rb").read()).hexdigest()[:16],
      "utils", hashlib.sha256(open(os.path.join(LISTENER_DIR, "utils.py"), "rb").read()).hexdigest()[:16])
utils = load_file("listener_utils_timing", os.path.join(LISTENER_DIR, "utils.py"))
assign = getattr(utils, "assign_source_identity", None)
generate = getattr(utils, "generate_alert_id", None)

print("\n--- identity generation, 5 runs over the sample")
copies = [dict(a) for a in alerts]
runs = []
for _ in range(5):
    started = time.perf_counter()
    for alert in copies:
        if assign:
            assign(alert)
        else:
            alert["alert_id"] = generate(alert)
    runs.append(time.perf_counter() - started)
best = statistics.median(runs)
print(f"median {best * 1000:.1f} ms for {len(alerts)} alerts: {len(alerts) / best:,.0f} alerts a second, {best / len(alerts) * 1e6:.2f} us each")

print("\n--- duplicate decision: accept_alert refusing a repeat, by cache size")
work = fresh()
listener = load_listener(work, log=False)
listener.alert_queue = __import__("queue").Queue(maxsize=0)
now = time.time()
for size in (100, 1000, 10000):
    listener.seen_alerts.clear()
    for number in range(size):
        listener.seen_alerts[f"wazuh:manager-01/1773187200.{number}"] = now
    repeat = {"alert_id": "wazuh:manager-01/1773187200.0"}
    samples = []
    for _ in range(300):
        started = time.perf_counter()
        assert listener.accept_alert(repeat) is False
        samples.append((time.perf_counter() - started) * 1e6)
    samples.sort()
    print(f"cache {size:>6}: median {statistics.median(samples):8.1f} us, p99 {samples[int(len(samples) * .99)]:8.1f} us")
shutil.rmtree(work)

print("\n--- memory of a full seen cache, by key form (10,000 entries)")
tracemalloc.start()
base = tracemalloc.take_snapshot()
cache = collections.OrderedDict()
for number in range(10000):
    cache[f"wazuh:manager-01/1773187200.{100000000 + number}"] = time.time()
first = sum(s.size_diff for s in tracemalloc.take_snapshot().compare_to(base, "filename"))
cache2 = collections.OrderedDict()
for number in range(10000):
    cache2[hashlib.sha256(str(number).encode()).hexdigest()] = time.time()
second = sum(s.size_diff for s in tracemalloc.take_snapshot().compare_to(base, "filename")) - first
tracemalloc.stop()
print(f"source key form: {first / 1024:.0f} KiB; earlier 64-character hash form: {second / 1024:.0f} KiB")
del cache, cache2


def passes(log):
    work = fresh(raw)
    listener = load_listener(work, log=log)
    listener.process_alert = lambda alert: True
    worker = threading.Thread(target=listener.queue_worker, daemon=True)
    worker.start()
    handler = listener.AlertFileHandler()
    size = os.path.getsize(listener.ALERT_FILE)
    timings = []
    for _ in range(2):
        handler.offset = 0
        started = time.perf_counter()
        while handler.offset < size:
            handler.read_new_lines()
        listener.alert_queue.join()
        timings.append(time.perf_counter() - started)
    listener.stop_event.set()
    worker.join(5)
    logging.shutdown
    os.chdir(BACKEND)
    shutil.rmtree(work)
    return timings


print("\n--- reader and worker, whole sample, no service time")
for label, log in (("with the listener's own per-alert logging", True), ("logging off", False)):
    first, second = passes(log)
    print(f"{label}: first pass {first:.3f} s, {len(alerts) / first:,.0f} alerts a second; second pass {second:.3f} s, {len(alerts) / second:,.0f} alerts a second")


def paced(service_ms, seconds=15):
    """A writer appends at 377 alerts a second. The real reader polls at the real interval, and the real worker takes service_ms each."""
    total = PEAK * seconds
    work = fresh()
    listener = load_listener(work, log=True)
    interval = 0.5
    done, refused = [], [0]
    delay = service_ms / 1000

    def service(alert):
        if delay:
            time.sleep(delay)
        done.append(time.perf_counter())
        return True
    listener.process_alert = service
    real = listener.accept_alert

    def counted(alert):
        try:
            return real(alert)
        except Exception:
            refused[0] += 1
            raise
    listener.accept_alert = counted
    worker = threading.Thread(target=listener.queue_worker, daemon=True)
    worker.start()
    handler = listener.AlertFileHandler()
    written = [0]

    def writer():
        started = time.perf_counter()
        with open(listener.ALERT_FILE, "a", encoding="utf-8") as handle:
            while written[0] < total:
                due = min(total, int((time.perf_counter() - started) * PEAK))
                if due > written[0]:
                    handle.writelines(raw[written[0]:due])
                    handle.flush()
                    written[0] = due
                time.sleep(0.01)
    thread = threading.Thread(target=writer, daemon=True)
    started = time.perf_counter()
    thread.start()
    deepest = 0
    backlog_at_end = None
    while True:
        handler.read_new_lines()
        deepest = max(deepest, listener.alert_queue.qsize())
        if not thread.is_alive() and backlog_at_end is None:
            backlog_at_end = total - len(done)
        if not thread.is_alive() and handler.offset >= os.path.getsize(listener.ALERT_FILE) and len(done) >= total:
            break
        if time.perf_counter() - started > 180:
            break
        time.sleep(interval)
    finished = time.perf_counter() - started
    listener.stop_event.set()
    worker.join(5)
    os.chdir(BACKEND)
    shutil.rmtree(work)
    print(f"service {service_ms:>4} ms: written {total}, processed {len(done)}, deepest queue {deepest}, queue-full refusals {refused[0]} (each retried), "
          f"behind when writing stopped {backlog_at_end}, finished {finished:.1f} s after the start")


if PACED:
    print(f"\n--- {PEAK} alerts a second for 15 s, real poll interval, synthetic service time per alert")
    for service in (0, 1, 2, 5, 9.2):
        paced(service)
print("\ntemporary directories left:", len([d for d in os.listdir(tempfile.gettempdir()) if d.startswith("cyukti-p26c-perf-")]))
