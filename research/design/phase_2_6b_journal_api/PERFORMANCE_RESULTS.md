# Performance results

**[M]** Measured once by `evidence/measure_performance.py`, output in `evidence/performance_output.txt`, on synthetic disposable journals in a temporary directory on WSL2's Linux filesystem. Python 3.12.3, SQLite 3.45.1. The payload is W07U-shaped: 807 statement bytes, including the sample's median 384-byte `investigation_payload`.

**These are one machine's numbers, from a single run.** Trust comparisons between rows more than the absolute values. fsync cost varies a lot by disk.

## Correlation overhead

| Operation | Previous code (v1) | Uncorrelated (v2) | Correlated (v2) |
|---|---|---|---|
| `create_intent`, median | 3.101 ms | 2.966 ms | 2.869 ms |
| `create_intent`, p99 | 11.642 ms | 10.008 ms | 10.472 ms |
| Whole life, create to `COMMITTED`, median | 9.174 ms | 9.368 ms | 9.228 ms |
| Whole life, p99 | 35.607 ms | 39.081 ms | 36.223 ms |

**The differences are within noise.** The cost is the fsync, and the new work is small beside it.

## Preflight

| Case | 1,000 records | 10,000 records |
|---|---|---|
| New correlated key, median | 0.039 ms | 0.040 ms |
| Duplicate, median | 0.063 ms | 0.071 ms |
| Uncorrelated, median | 0.023 ms | 0.022 ms |

## Lookup

| Case | 1,000 records | 10,000 records |
|---|---|---|
| One epoch, median | 0.029 ms | 0.029 ms |
| Unknown key, median | 0.010 ms | 0.010 ms |

**No growth with the journal's size.** Cost grows with one request's epochs:

| Epochs for one request | `find_by_correlation_key`, median | Preflight of the next, median |
|---|---|---|
| 1 | 0.026 ms | 0.062 ms |
| 2 | 0.043 ms | 0.080 ms |
| 5 | 0.088 ms | 0.118 ms |
| 20 | 0.335 ms | 0.348 ms |
| 100 | 1.694 ms | 1.491 ms |

A hundred epochs would need a hundred human resolutions, so that's far beyond any real case.

## Integrity

| Case, 3,000 records | `check_integrity` |
|---|---|
| Uncorrelated | 178.1 ms |
| Correlated | 208.1 ms |
| After migrating, uncorrelated | 250.8 ms |

It's a full scan, as it always was, and isn't on a write path. The last figure includes a fresh connection's cold cache, so I wouldn't read it as a cost of migration.

## Growth

| Case, 3,000 records with 3 history entries each | File size | Per record |
|---|---|---|
| Uncorrelated | 9,457,664 bytes | 3,153 bytes |
| Correlated | 10,342,400 bytes | 3,447 bytes |

**Correlation adds about 294 bytes a record, or 9%,** mostly the key repeated on each history entry, plus the index. At W07U's sample rate of 11,197 requests a day, the journal alone would grow by **about 39 MB a day.** That number matters for D12.

Migrating: 9,281,536 bytes to 9,285,632, one 4,096-byte page.

## What this means for later work

**[M] The journal alone allows about 108 whole lives a second here,** at 9.2 ms each. That's three fsynced commits. **W07U's busiest second in the sample holds 377 requests.**

With the ledger's commits and a Neo4j transaction added, a synchronous coordinator on W07U **would fall behind in bursts** on this machine. A full queue delays rather than loses, so it's backlog, not loss, until log rotation is reached. This doesn't block this phase. It's the first real figure for D13's benchmark condition, and for D11.

**Unmeasured:** the ledger on the same path, and the full coordinator.

## Normal write paths

| Added to a write path | Cost |
|---|---|
| `create_intent`, correlated | One indexed read of that key's records, inside the transaction |
| `mark_in_flight`, `authorize_retry`, correlated | One indexed read of that key's records |
| Outcomes | None beyond the row already loaded |
| Uncorrelated records | None |

**No unbounded scan or full-history read was added to a write path.** `recover_unfinished` and `check_integrity` still scan, as before, and neither is on one.
