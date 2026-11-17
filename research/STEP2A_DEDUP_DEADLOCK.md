# Step 2A — Dedup restart deadlock fix and replay image preparation

**Date:** 2026-11-04 · **Base:** `3afe62c` + Steps 1–2 (uncommitted working tree)
**Evidence:** `research/experiments/STEP2A-dedup-deadlock-v1/`

## 1. Defect

`DeduplicationEngine.is_duplicate()` holds `self.lock` (a non-reentrant `threading.Lock`) across check-then-populate. On a cache miss with a recent event in Neo4j, it called `reload_cache()`, which acquired `self.lock` again. The thread blocked on itself forever: no exception, no log line, no CPU use.

- **Trigger:** the in-memory cache is empty (fresh process) and a repeat alert arrives while Neo4j still holds its fingerprint inside the dedup window.
- **Scope:** `reload_cache()` had exactly one caller, this one, so the path could never complete. The code is the same at `HEAD` (pre-existing, not introduced by Steps 1–2).
- **Production impact:** a listener restarted within the dedup window could hang silently on the first repeat alert, and its single queue worker would stop processing.
- **Link to the B3 stall:** the first alert after the simulated restart (index 4500) was a resolved duplicate whose fingerprint had last fired 0.034 s earlier. This is a strong inference; the hung process no longer exists to inspect. It replaces the earlier, incorrect explanation in the B3 status report and in `PRESERVATION_NOTE.md` (left unedited). The `wsl.exe` timeouts are a separate issue: order-7 kernel allocation failures from fragmented memory (journal, 19:15:19 onward).

## 2. Fix (`backend/dedup_engine.py` only)

- New private `_store_reloaded_entry()` performs the cache write and assumes the caller holds the lock, the same convention as `cleanup()`.
- `is_duplicate()` calls it directly, still under the lock.
- Public `reload_cache()` keeps its signature and takes the lock itself, then calls the helper.

**Why not `RLock`:** the lock is held across a multi-step operation, and nesting here was a bug. An `RLock` would also stop this hang, but it would hide any future nested acquisition, and nothing else in the class needs reentrancy. A test pins the lock type.

**Unchanged:** fingerprint generation, `DEDUP_WINDOW`, classification, cleanup, `register_event`, flush behaviour, the DB lookup and when it runs (still under the lock, with the same arguments).

## 3. Tests

New: `backend/tests/test_dedup_reload_deadlock.py` (13). Calls that could hang run in daemon threads with bounded joins.

| Group | Tests | Pre-fix | Post-fix |
|---|---|---|---|
| Restart path returns `(True, fingerprint)` | 1 | FAIL (deadlock) | pass |
| Cache repopulated from the DB event (exact entry) | 1 | FAIL | pass |
| Lock released afterwards | 1 | FAIL | pass |
| Next repeat served from cache, no DB call, occurrences = 2 | 1 | FAIL | pass |
| DB lookup still runs under the lock | 1 | FAIL | pass |
| 16 concurrent repeats: all complete, 1 DB lookup, occurrences = 16 | 1 | FAIL | pass |
| Reload path alongside `register_event` (40 threads) | 1 | FAIL | pass |
| Public `reload_cache` still waits for the lock | 1 | pass | pass |
| Lock is still non-reentrant | 1 | pass | pass |
| Fingerprint value pinned | 1 | pass | pass |
| No DB event: not a duplicate, no cache entry | 1 | pass | pass |
| DB lookup receives the alert's event time | 1 | pass | pass |
| Entry past the window falls through to the DB | 1 | pass | pass |

Pre-fix total: 7 failed, 6 passed. Post-fix: 13 passed, plus 5 further consecutive runs of 13/13.

**Full suite**, with `NEO4J_URI=bolt://127.0.0.1:9`, MISP at a closed port, Shuffle URLs empty: **929 passed, 29 skipped, 2 failed** (previously 916/29/2; +13 new). The 2 failures are the same as at baseline: `test_operation_attribution_integration.py::test_realistic_two_alert_scenario…` and `test_threat_attribution.py::test_attribution_evidence_derives…`. Both report `ServiceUnavailable` for `127.0.0.1:9`, confirming they were held away from production. Unrelated to this fix.

## 4. Before/after dedup comparison

`scripts/dedup_differential.py` drives the saved pre-fix module and the fixed module with identical inputs. The DB fallback is stubbed to "nothing recent", the only answer the pre-fix engine survives.

| Input | Result |
|---|---|
| 200 randomized sequences, 80,000 decisions (window edges, late alerts, ties, cleanup) | decisions and final cache identical in all 200 |
| Full archive, 9,015 alerts, harness order | identical; 8,753 duplicates |
| Same, against Step 2 run A's recorded per-alert paths | 0 differences |

**Only behaviour change:** the DB-hit path now returns `(True, fingerprint)` instead of hanging. It's the path's original intent, but it has **never completed in production**, so what follows it (a restart resuming duplicate handling from Neo4j) gets its first real exercise in the B3 restart runs.

## 5. Replay infrastructure

- Docker Desktop: `neo4j:5.26.27` pulled. Image id `sha256:b8b4c30662aa…`, digest `sha256:937fbd163e30…`, community edition, 590 MB. Same version as production (5.26.27).
- Daemons: the CLI (`default` context, `/var/run/docker.sock`) reaches `docker-desktop`. `neo4j-soc` runs on Ubuntu's native `dockerd`, which the CLI cannot currently see.
- Scratch `with_fresh_neo4j.sh`: image pinned; refuses to start unless the engine is `docker-desktop` (exit 4) or if the image is absent, so no implicit pull (exit 5). Memory limits unchanged. Archived copies of the script keep the old tag, because they record what actually ran.

## 6. Untouched

Production Neo4j fingerprint (nodes, relationships, campaigns, operations, events, transition total, newest timestamps) is byte-identical before and after; only read transactions were used. Same 11 running containers; `wazuh-manager` not restarted; no listener; no replay; no temporary containers; `misp_cache.json` unchanged (Sep 26); B3 run checksums verified.

## 7. Remaining risks

1. `is_duplicate` still holds the lock during the Neo4j lookup, so a slow database blocks other dedup callers. Pre-existing and harmless with one worker; left alone to keep the fix minimal.
2. `get_metadata` reads the cache without the lock. Pre-existing.
3. After a restart, the reloaded entry starts with `occurrences = 1`, and the event's existing count stays in Neo4j. This is the code's existing design, first exercised now.
4. Not yet verified: a Docker Desktop port published on `127.0.0.1:17690` being reachable from the Ubuntu distro (MISP's published port is, but it binds all interfaces).
5. Memory fragmentation can recur and block new `wsl.exe` sessions while services stay up.
