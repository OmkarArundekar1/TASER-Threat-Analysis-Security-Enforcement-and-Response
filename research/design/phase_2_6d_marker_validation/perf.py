"""Preliminary latency observations on the disposable instance, through the lifted write helper, on one machine. These aren't production throughput.
usage: python -B perf.py [REPETITIONS]"""
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import marker_lib as m

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
ctx = m.Context("perf")
results = {"repetitions": N, "environment": ctx.environment, "server": m.server_identity(ctx.direct), "measures": {}}


def summarize(name, samples):
    samples = sorted(samples)
    row = {"n": len(samples), "median_ms": round(statistics.median(samples), 3), "p95_ms": round(samples[int(len(samples) * 0.95) - 1], 3), "max_ms": round(samples[-1], 3)}
    results["measures"][name] = row
    print(f"{name:<62} n={row['n']:<5} median {row['median_ms']:8.3f}  p95 {row['p95_ms']:8.3f}  max {row['max_ms']:8.3f} ms", flush=True)


def timed(call, repeat=N):
    samples = []
    for number in range(repeat):
        started = time.perf_counter()
        call(number)
        samples.append((time.perf_counter() - started) * 1000)
    return samples


events = [ctx.event() for _ in range(N)]
for number in range(20):  # warm the connection and plans
    ctx.help.write_query("warm", m.UNMARKED_DUPLICATE_UPDATE, m.params(events[0], "warm"))
summarize("mutation alone, in the helper (baseline)", timed(lambda n: ctx.help.write_query("d6", m.UNMARKED_DUPLICATE_UPDATE, m.params(events[n], "none"))))
keys = [ctx.key("perf") for _ in range(N)]
summarize("marker plus mutation, in the helper", timed(lambda n: ctx.help.write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params(events[n], keys[n]))))
summarize("lookup, marker exists", timed(lambda n: m.lookup(ctx.direct, keys[n])))
summarize("lookup, marker absent", timed(lambda n: m.lookup(ctx.direct, keys[n] + "-absent")))


def duplicate(n):
    result = ctx.help.write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params(events[n], keys[n]))
    assert result.exception_code == m.CONSTRAINT_FAILED


summarize("duplicate refused by the constraint", timed(duplicate))


def duplicate_then_lookup(n):
    duplicate(n)
    assert m.lookup(ctx.direct, keys[n]).status == m.APPLIED


summarize("duplicate detected: refused, then the lookup", timed(duplicate_then_lookup))

for contenders in (2, 8, 16):
    samples, rounds = [], max(20, N // 25)
    for _ in range(rounds):
        event, key = ctx.event(), ctx.key("perf")
        barrier = threading.Barrier(contenders)

        def attempt():
            barrier.wait()
            started = time.perf_counter()
            ctx.help.write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params(event, key))
            samples.append((time.perf_counter() - started) * 1000)
        threads = [threading.Thread(target=attempt) for _ in range(contenders)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
    summarize(f"contention: {contenders} clients on one key, per client", samples)

print("\n--- does cost change as markers accumulate?", flush=True)
BATCH = 20000
with ctx.direct.session() as session:
    before = session.run("MATCH (m:AppliedWrite) RETURN count(m) AS n").single()["n"]


def clean_restart():
    """Community has no db.checkpoint(). A clean stop checkpoints, so sizes are compared after one. Only ever this experiment's container."""
    m.confirm_environment()
    m._docker("restart", m.CONN["container"], timeout=180)
    end = time.monotonic() + 180
    while time.monotonic() < end:
        try:
            with ctx.direct.session() as session:
                session.run("RETURN 1").consume()
            return
        except Exception:
            time.sleep(1)
    raise SystemExit("STOP: the disposable server didn't come back")


def data_bytes():
    out = subprocess.run(["docker", "exec", m.CONN["container"], "du", "-sb", "/data/databases/neo4j"], capture_output=True, text=True, timeout=60).stdout
    return int(out.split()[0])


clean_restart()
size_before = data_bytes()
for start in range(0, BATCH, 2000):
    with ctx.direct.session() as session:
        session.run("UNWIND range($a, $b) AS i CREATE (:AppliedWrite {request_key:'cyukti.v1|wazuh:d6-manager/1773187200.' + toString(i) + '.' + $p + '|duplicate_event.update', applied_at:datetime()})",
                    a=start, b=start + 1999, p=ctx.prefix).consume()
with ctx.direct.session() as session:
    after = session.run("MATCH (m:AppliedWrite) RETURN count(m) AS n").single()["n"]
clean_restart()
size_after = data_bytes()
results["storage"] = {"markers_added": after - before, "bytes_before": size_before, "bytes_after": size_after, "bytes_per_marker": round((size_after - size_before) / (after - before), 1), "note": "database directory after a clean restart, including transaction logs, with store allocation in blocks: a rough figure"}
print(json.dumps(results["storage"]), flush=True)
events2 = [ctx.event() for _ in range(N // 2)]
keys2 = [ctx.key("perf") for _ in range(N // 2)]
summarize(f"marker plus mutation, with {after} markers present", timed(lambda n: ctx.help.write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params(events2[n], keys2[n])), N // 2))
summarize(f"lookup, marker exists, with {after} markers present", timed(lambda n: m.lookup(ctx.direct, keys2[n]), N // 2))
ctx.close()
path = Path(__file__).resolve().parent / "results" / "performance.json"
path.write_text(json.dumps(results, indent=1, default=str))
