# Performance results

**[M]** Measured once each by `evidence/measure_listener_performance.py`, offline. Outputs: `evidence/performance_before_fix.txt`, `evidence/performance_after_fix.txt`. Python 3.12.3 on WSL2. **One machine, single runs,** so small differences are noise. `process_alert` is a stub, and service times are synthetic sleeps. **This is a benchmark, not a migration of W07U.**

## Identity and duplicate lookup

| Measure | Before | After |
|---|---|---|
| Identity generation, per alert | 2.53 us | 2.83 us |
| Identity generation, alerts a second | 395,187 | 352,769 |
| Refusing a repeat, cache of 100, median | 4.7 us | 5.4 us |
| Refusing a repeat, cache of 1,000, median | 44.9 us | 51.9 us |
| **Refusing a repeat, cache of 10,000, median** | **527.6 us** | **579.8 us** |
| Refusing a repeat, cache of 10,000, p99 | 905.1 us | 988.7 us |

**The lookup's cost is its expiry scan,** a pre-existing walk over the whole cache on every call, in `listener/wazuh_listener.py:accept_alert` L128-162. It grows with cache size, and it's unchanged by this phase. It dominates identity generation by about 200 times at a full cache.

## Reading the sample

| Case | Before | After |
|---|---|---|
| First pass, with per-alert logging | 1,955 a second | 1,784 a second |
| Second pass, with per-alert logging | 1,824 a second | 1,362 a second |
| First pass, logging off | 3,506 a second | 2,671 a second |
| Second pass, logging off | 1,967 a second | 1,431 a second |

**[M] The fixed reader looks slower here,** by roughly 9% to 27%, more than identity generation explains. I can't say from single runs how much is noise and how much is real, and I haven't isolated it. The second pass is slower in both versions, because every lookup scans a full cache.

## Memory

A full cache of 10,000 entries: **1,640 KiB** with source keys, 1,903 KiB with the earlier hash keys. The cache's limits are unchanged.

## At W07U's observed peak

A writer appended at **377 alerts a second for 15 s.** The real reader polled at its real 0.5 s interval, and the real worker took a synthetic time per alert.

| Service per alert | Capacity | Processed of 5,655 | Deepest queue | Queue-full refusals, each retried | Behind when writing stopped | Finished at |
|---|---|---|---|---|---|---|
| 0 ms | | 5,655 | 0 | 0 | 0 | 15.4 s |
| 1 ms | ~1,000/s | 5,655 | 195 | 0 | 155 | 15.6 s |
| 2 ms | ~500/s | 5,655 | 206 | 0 | 56 | 15.9 s |
| **5 ms** | ~200/s | 5,655 | 999 | 38 | **2,937** | **31.6 s** |
| **9.2 ms** | ~108/s | 5,655 | 1,000 | 80 | **4,133** | **55.7 s** |

**None was lost in these runs.** A full queue stops the reader, and the alert stays in the file. "Behind" counts alerts written but not processed.

## The bottleneck

**The listener isn't the limit at 377 a second.** Its reader manages over 1,300 a second even with a full cache.

**`process_alert` is the limit:** it has to average under about 2.65 ms per alert to keep up with this peak over a sustained period. Its real time, with Neo4j, **wasn't measured here.** Phase 2.6B measured the journal alone at 9.2 ms for one correlated write, the last row, so **a synchronous coordinator on W07U would fall behind in sustained bursts.**

Backlog sits in the file, and it's safe until rotation. A backlog over a rotation hits the rotation gaps in `OFFSET_AND_QUEUE_SAFETY.md`.

## Proposed separately, not done

| Finding | Bounded proposal |
|---|---|
| The expiry scan walks the whole cache per alert | Expire from the oldest end only, since entries are time-ordered. A small change, once its clock behaviour is tested |
| At the peak, 10,000 entries cover about 26 seconds, short of the 600 s TTL | Decide the real replay window needed, then size the cache |
| Per-alert INFO logging costs about a third of reader time | Lower to DEBUG. It changes what operators see, so it's your call |
| `process_alert` sets the limit | Measure it on a disposable server, as part of D13's benchmark |
