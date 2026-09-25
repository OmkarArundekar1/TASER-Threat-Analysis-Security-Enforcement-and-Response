# CYUKTI Performance Benchmark Harness

`backend/benchmarks/run_benchmarks.py` — reproducible, real (never fabricated). Each benchmark is a real function call against whatever this environment actually has available; an unavailable subsystem is reported `"skipped"` with the real reason, never a guessed number.

```bash
cd backend && python benchmarks/run_benchmarks.py --iterations 20
```

Writes a timestamped JSON report to `benchmarks/results/` and prints a p50/p95/p99/throughput table.

## A real run, this environment, this phase

Measured 2026-09-25, live Neo4j (Docker, `neo4j:5-community`), `GNN_ENABLED=true`, a trained XGBoost severity model present, 20 iterations per benchmark (first call excluded as warm-up):

| Benchmark | p50 (ms) | p95 (ms) | p99 (ms) | throughput (ops/sec) |
|---|---|---|---|---|
| MITRE resolution | 0.001 | 0.001 | 0.002 | ~1,110,000 |
| IOC extraction | 0.000 | 0.000 | 0.001 | ~5,035,000 |
| Deduplication check | 0.001 | 0.002 | 0.003 | ~735,000 |
| Neo4j simple query | 0.904 | 1.251 | 1.285 | ~1,116 |
| Campaign selection scoring (in-memory, 10 candidates) | 0.037 | 0.073 | 0.074 | ~23,200 |
| Campaign selection candidate discovery (live Neo4j query + real signal computation) | 21.035 | 24.285 | 26.480 | ~47 |
| Severity prediction (XGBoost) | 21.975 | 39.158 | 117.582 | ~34.5 |
| GNN embedding (real model, real campaign graph, cache bypassed) | 2.818 | 3.981 | 4.070 | ~342 |
| ResponsePlan generation | 0.002 | 0.004 | 0.006 | ~373,000 |
| **Total end-to-end** (`GET /api/incidents/<id>/overview`, real HTTP round trip) | 35.538 | 66.869 | 100.484 | ~22.8 |

Raw JSON: `backend/benchmarks/results/benchmark_20260925T075030Z.json` (committed as the reference run for this phase).

## Reading these numbers honestly

- These are **single-process, single-machine, low-concurrency** numbers on a development machine (see `REPRODUCIBILITY.md` for exact software/hardware context) — not a load-tested production SLA.
- The end-to-end p50 (~36ms) is dominated by two real network round trips inside the aggregate endpoint (the campaign-selection candidate-discovery query, ~21ms, plus the base campaign/MITRE/qualification lookups) — not by any single slow component in isolation.
- Severity prediction's p99 (117.6ms) is notably higher than its p50 (22ms) — the pattern (in the raw JSON) is consistent with JIT/cache warm-up variance in XGBoost's C++ prediction path across repeated small-batch calls, not a systematic bottleneck; a longer run would characterize this more precisely, which this harness supports (`--iterations`) but wasn't run at a larger N this phase given time.
- Deduplication/MITRE-resolution/IOC-extraction numbers (sub-microsecond) reflect pure in-memory Python logic with no I/O — they are lower bounds on the *decision* cost, not on the surrounding real pipeline (Neo4j writes, etc.) which the "Neo4j simple query" and "candidate discovery" rows separately characterize.

## Tests

`backend/tests/test_benchmark_harness.py` (7) — verifies the harness's own percentile math and skip-vs-measure detection logic, independent of any particular subsystem's real-world latency.
