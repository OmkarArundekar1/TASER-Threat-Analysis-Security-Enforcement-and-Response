"""perf.py's last two measures came straight after a restart, so a cold server and a larger marker count were mixed. This repeats them warm, on the same
data, to separate the two. Run after perf.py. usage: python -B perf_warm.py"""
import statistics
import time

import marker_lib as m

ctx = m.Context("warm")
with ctx.direct.session() as session:
    present = session.run("MATCH (m:AppliedWrite) RETURN count(m) AS n").single()["n"]
print(f"markers present: {present}")


def measure(name, call, repeat=500):
    samples = []
    for number in range(repeat):
        started = time.perf_counter()
        call(number)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    print(f"{name:<52} n={repeat:<5} median {statistics.median(samples):8.3f}  p95 {samples[int(repeat * 0.95) - 1]:8.3f}  max {samples[-1]:8.3f} ms", flush=True)


events = [ctx.event() for _ in range(1000)]
keys = [ctx.key("warm") for _ in range(1000)]
measure("warm-up, marker plus mutation, discard these", lambda n: ctx.help.write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params(events[n], keys[n])), 500)
measure("marker plus mutation, warm", lambda n: ctx.help.write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params(events[n + 500], keys[n + 500])), 500)
measure("mutation alone, warm", lambda n: ctx.help.write_query("d6", m.UNMARKED_DUPLICATE_UPDATE, m.params(events[n], "none")), 500)
measure("lookup, marker exists, warm", lambda n: m.lookup(ctx.direct, keys[n]), 500)
measure("lookup, marker absent, warm", lambda n: m.lookup(ctx.direct, keys[n] + "-absent"), 500)
ctx.close()
