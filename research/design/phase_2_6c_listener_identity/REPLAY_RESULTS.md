# Replay results

**[M]** Measured offline by `evidence/replay_listener.py`, which runs the real `AlertFileHandler.read_new_lines`, `accept_alert` and `queue_worker` from the listener files it's given. So the same script tests both versions. `process_alert` is a stub that records what reached it. No service, Neo4j or live file is involved, and `start_listener` is never called.

**Input:** `isfcr/CYUKTI/data/raw_logs/alerts.json`, 9,015 alerts, sha256 `e958c1cc9211036b2e9feaf48990c36cf370d724c1c139183229cb59565b3098`.

**Outputs:** `evidence/replay_before_fix.txt`, `evidence/replay_after_fix.txt`.

## Required criteria

| Condition | Required | Result |
|---|---|---|
| First pass | All 9,015 accounted for | **9,015 accepted,** 9,015 distinct ids at `process_alert`, in file order |
| Previously refused 70 | All preserved | **70 of 70,** with 70 distinct ids |
| Second pass | Source-identical alerts recognized | **9,015 duplicates,** none reached `process_alert` again |
| Different source ids | Never rejected for a content collision | **0 rejected.** Content isn't consulted |
| Different manager namespaces | Remain distinct | **18,030 of 18,030 accepted,** when the sample appears under two names |
| Missing source id | Not silently discarded | **9,015 of 9,015 delivered** |
| Restart replay | No permanent loss from the offset | **9,015 of 9,015 redelivered** |

## Counts, reported separately

| Scenario | Accepted | Duplicate | Malformed id | Missing or empty id | No manager | Failed |
|---|---|---|---|---|---|---|
| **Before the fix, first pass** | 8,945 | **70** | not distinguished | | | 0 |
| **After, first pass** | **9,015** | 0 | 0 | 0 | 0 | 0 |
| After, second pass | 0 | 9,015 | 0 | 0 | 0 | 0 |
| After, restart at the saved offset | 0 | 0 | 0 | 0 | 0 | 0 |
| After, restart with the offset at 0 | 9,015 | 0 | 0 | 0 | 0 | 0 |
| After, `id` removed, first pass | 9,015 | 0 | 0 | 9,015 | 0 | 0 |
| After, `id` empty, first pass | 9,015 | 0 | 0 | 9,015 | 0 | 0 |
| After, `id` not a string, first pass | 9,015 | 0 | 9,015 | 0 | 0 | 0 |
| After, `id` malformed, first pass | 9,015 | 0 | 9,015 | 0 | 0 | 0 |
| After, manager removed, first pass | 9,015 | 0 | 0 | 0 | 9,015 | 0 |

**In the last five, the second pass delivers all 9,015 again.** That's the documented trade-off: no identity, no suppression, 18,030 at `process_alert` in all.

The same scenarios on the old code refuse 70 alerts every time, and a second manager name makes no difference: 9,085 refused of 18,030.

## Lines that can never be alerts

Two real alerts around one bad line, three reads:

| Bad line | Before: real alerts through, reader past it | After |
|---|---|---|
| Truncated, ending in a newline | **1 of 2, stuck** | 2 of 2, past it |
| JSON array | **1 of 2, stuck** | 2 of 2, past it |
| JSON number | **1 of 2, stuck** | 2 of 2, past it |
| Malformed, ending in `}` | 2 of 2, past it | 2 of 2, past it |

## The seven

[M] 7 of the 70 name a different attacker or victim from the alert they were merged into. All 7 now reach `process_alert`. [T] A test confirms each keeps its own values through `extract_iocs`.

## Conditions on these results

- **The second pass ran within seconds.** Suppression lasts up to `ALERT_CACHE_TTL` (600 s) and `MAX_ALERT_CACHE` (10,000). [T] Past either limit, a replayed alert is delivered again. **A file over 10,000 alerts wouldn't be fully recognized on a second pass.**
- **`process_alert` is a stub,** so these show arrival, not processing.
- **One sample,** one manager, no restart or rotation in it.
- **"Restart with the offset at 0"** stands in for a crash before the offset was saved. Real SIGKILL windows are in the tests.
