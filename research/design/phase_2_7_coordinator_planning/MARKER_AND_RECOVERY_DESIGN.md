# Marker and recovery design

Planning only. **No marker or constraint exists, and nothing here is implemented.** Backend paths are relative to `isfcr/CYUKTI/backend/`. **[F]** fact, **[R]** recommendation, **[H]** hypothesis, **[U]** unverified.

## The contract, as accepted

`(:AppliedWrite {request_key, applied_at, operation, fingerprint})`

| Rule | Implementation |
|---|---|
| Same statement and transaction as the mutation | Templates in the operation registry. **A writer can't omit it** |
| Inside the mutation's first `MATCH` | [F] E1's control: a separate statement commits false evidence |
| Unique `request_key` | The constraint below |
| Fingerprint excludes attempt and server time | `fingerprint_payload` over the unmarked statements. `applied_at` is the server's `datetime()`, not a parameter |
| Operation and fingerprint immutable | [F] Community can't enforce it. **Application rule:** nothing issues `SET` on the label, and a test scans for it |
| Same key, same fingerprint: already applied | `ALREADY_APPLIED` |
| Same key, different fingerprint: conflict | `MARKER_CONFLICT` |
| `ConstraintValidationFailed` alone proves nothing | [F] E5 |
| Absence never resolves anything | [F] 2.6D: a marker appeared after the client died |

## Two fingerprints, and why it's safe

| Fingerprint | Over | Stored in | Used for |
|---|---|---|---|
| **Mutation fingerprint** | The unmarked statements and their parameters | **The marker** | Detecting a reused key in Neo4j |
| **Request fingerprint** | The rendered statements, marker included | The ledger | Detecting a changed request, and `begin_execution`'s check |

**A marker can't hold a hash of a statement that contains that hash,** so one over the unmarked form is needed. The rendered form is a pure function of the mutation form, so the two can't drift for one version of the code. **[F] A changed template changes the ledger's fingerprint, and open requests conflict:** deliberate, so rollout must drain them first.

## Lookup

`read_applied_write(request_key)` in `neo4j_client.py`. It's additive, and reads only.

| Result | Meaning | Evidence |
|---|---|---|
| `APPLIED` | One marker. Returned with `operation` and `fingerprint` | Positive |
| `ABSENT` | A read that succeeded found none, on this store, at this moment | **Only what 2.6D's nine conditions allow** |
| `UNKNOWN` | **Any failure,** including a client deadline | None |
| `INCONSISTENT` | More than one, so the constraint is missing or was | Damage |

It has `execute_read`, a 3 s transaction timeout, and a 6 s client-side deadline from a joined thread: [F] a silent stalled read has no driver timeout. Failed reads don't retry.

## Startup checks: `verify_startup()`

Run in this order before marker mode:

| # | Check | If it fails |
|---|---|---|
| 1 | Connect, and `dbms.components`: record version and edition | `UNKNOWN`: marker mode refused, and the worker holds |
| 2 | Version and edition match what's been validated | Refused, naming the difference |
| 3 | `SHOW CONSTRAINTS`: `applied_write_key`, UNIQUENESS, NODE, `AppliedWrite`, exactly `request_key` | Refused |
| 4 | `SHOW INDEXES`: its owned index is `ONLINE` | Refused until online |
| 5 | `db.labels` | Next table |
| 6 | Sample up to 1,000 markers for missing `operation` or `fingerprint` | Alarm. Refused if any |

| Constraint | Label | Meaning | Action |
|---|---|---|---|
| Present | Present or absent | Normal | Proceed |
| **Absent** | Absent | Rollout hasn't happened | **Marker mode refused.** Migrated routes hold, or the switch is off |
| **Absent** | **Present** | **Dropped, or markers made without it** | **Refused, with an alarm.** Count duplicate keys read-only, and report them |
| Present with other properties | Any | Someone else's constraint | Refused |

[F] Check 3 re-runs on each health-gate recovery, because a dropped constraint silently allows double applies: E10.

## Rollout

Each step is **an operator act, approved separately, never by a code path.**

| # | Step | Verify | Rollback |
|---|---|---|---|
| 0 | **Rehearsal passed.** Below | Evidence archived | |
| 1 | Back up the graph | Backup restores on a disposable server | |
| 2 | No open requests: `inspect` shows nothing in `IN_PROGRESS`, `UNCERTAIN`, `NOT_COMPLETED`, `RETRY_AUTHORIZED`. **Today there are none, since nothing uses the coordinator** | Report archived | |
| 3 | `CREATE CONSTRAINT applied_write_key IF NOT EXISTS FOR (m:AppliedWrite) REQUIRE m.request_key IS UNIQUE` | `SHOW CONSTRAINTS`, `SHOW INDEXES`, `db.awaitIndexes` | `DROP CONSTRAINT applied_write_key`. Nothing else depends on it yet |
| 4 | Enable the coordinator in **shadow:** no writes, only `verify_startup` | Report | Disable |
| 5 | One writer, behind a switch, default off. **That's Phase 2.8** | The acceptance list | Switch off |

**[F] A schema change can't share a transaction with data,** and the empty label means it's quick. [U] Lock impact at production scale is an X4 question.

### Rollback, once markers exist

| Action | Safe? |
|---|---|
| Turn the writer's switch off | **Yes.** Old path resumes. Markers are inert: no read path traverses them |
| Keep markers | **Yes. Required** |
| Drop the constraint | **Not while marker mode could run.** And `verify_startup` then refuses |
| Delete markers | **No.** [F] E10: a retry applies again. Prohibited without a separately approved retention design |
| **Downgrade** | [F] The old `update_duplicate_event` doesn't look at markers, so a re-delivered alert could double-count during a rollback. Accepted, and a reason for the backlog's door check |

### Existing data

[F] `AppliedWrite` doesn't exist, so no existing node conflicts. **Existing `AttackEvent` nodes have no marker,** so a request from before the marker has no evidence, and absence says nothing: condition 8 in 2.6D.

### Read paths with markers present

| Path | Status |
|---|---|
| Typed queries | [F] Unaffected: no relationship, and an unused label |
| `graph_feature_engine.py`, untyped relationship matches | [F] Unaffected, because a marker has no relationships |
| `dashboard_api.py`, `labels(n)` on nodes already found | [F] 2.6A's scan: none can reach a marker |
| **Free-form read queries in the dashboard** | **Can show them,** if someone asks for all nodes |
| `research/` scripts | [U] Not scanned |
| **Backend read paths on a graph that has markers** | [U] **Never run. X4** |

### Rehearsal

**[F] Phase 2.6D's disposable-server results aren't production-scale validation.** Run on a disposable copy of production's data, or a rehearsal of equal shape, on the same version. **Getting a copy needs your authorization.**

| Step | Pass |
|---|---|
| Restore; `SHOW CONSTRAINTS` gives the same ten | Matches 2.6E |
| Constraint creation, timed | Completes, with no other writer harmed |
| E1 | Marker and mutation together. Campaign untouched on unattributed |
| E5 | One marker, one effect |
| E6 | 16 threads, 4 processes: 0 wrong rounds. Also with legacy writers running |
| Marker-present read paths | Same results, for every dashboard route and feature read |
| Performance | Marker overhead, at real size |
| Failure and recovery | Kill under load. 0 marker-effect disagreements |

## Inconsistent marker and ledger states

| Marker | Ledger | Likely cause | Action |
|---|---|---|---|
| `APPLIED`, matching | `COMPLETED` | Normal | None |
| `APPLIED`, matching | `IN_PROGRESS`, owner alive | Writing now | None |
| `APPLIED`, matching | `UNCERTAIN` | Lost reply | Evidence for `FOUND_COMMITTED` |
| `APPLIED`, matching | `NOT_COMPLETED` | **A person found "not committed" wrongly, or `ABSENT` was misread** | Report. A retry gets `ALREADY_APPLIED` |
| `APPLIED`, matching | Missing | **Ledger lost or restored** | Report. Refuse by marker: `ALREADY_APPLIED` |
| `APPLIED`, **differing** | Any | Reused key, or a code change without draining | `MARKER_CONFLICT` |
| `ABSENT` | `COMPLETED` | **Graph restored, or marker deleted** | **Alarm. Never re-run.** A person |
| `ABSENT` | `UNCERTAIN` | Unknown | Only the conditions in 2.6D. A person |
| `ABSENT` | `REGISTERED`, `NOT_COMPLETED` | Normal | None |
| `INCONSISTENT` | Any | Constraint was missing | **Marker mode stops** |
| `UNKNOWN` | Any | Lookup failed | No conclusion |

**[R] No code path resolves any of these on its own.** `check_consistency` reports them, reading a bounded sample of completed requests.

## Automatic resolution

[R] **None in Phase 2.7.** A matching marker is strong positive evidence, so `FOUND_COMMITTED` from it could later be automated under a named policy principal. That's **OD-7,** once real volume is known. **Absence never.**

## D13: re-delivery across operations

**The problem.** [F] An alert that created an event through W06 is re-read after a crash, with the dedup cache empty. `find_recent_duplicate` finds its own event, so it routes as a duplicate. Its W07U key differs, so **neither ledger nor marker catches it,** and `occurrences` gains 1.

**A marker can't solve this,** because the two operations are rightly separate requests.

### Treatment, in layers

| # | Layer | Stops | Status |
|---|---|---|---|
| 1 | **Backlog door.** `UNIQUE(source_key)`, so a re-read alert never reaches `process_alert` again | **Everything from the reader,** whichever route it took | [R] In the design. Testable offline |
| 2 | **No automatic re-run.** A crashed alert becomes `INTERRUPTED` | The crash case itself | [R] In the design. OD-3 |
| 3 | **Recorded route.** Once `process_alert` picks a route, the backlog keeps it. Re-processing compares, and a different route **holds** it, not runs it | A person's re-run taking another route | [R] Needs the route exposed before the first write: WP-T4 |
| 4 | **Source-prefix check,** optional: before a W07U write for this alert, a marker `STARTS WITH` its source key **under another operation** | Replays bypassing the backlog, such as a fresh backlog with an old graph | [U] **Needs W06 migrated, and X8.** Not for the pilot |

**What's left:**

| Case | Remaining risk |
|---|---|
| **Alert with no identity** | Can be re-delivered. [F] 0 of 681,259 |
| **Backlog lost or restored** | Layer 1 is gone. Layer 4 would then matter |
| **A person's authorized re-run** | Layer 3 reports it. The person decides |
| **Archive replay into the same graph** | Blocked by layer 1 while the backlog's keys remain. **So those keys must be retained as long as markers are:** OD-6 |

**This is a treatment, not a proof.** [H] Layers 1 and 2 close the mechanism in code. Tests will show it.

## Stale payload protection

**The problem.** [F] `update_duplicate_event` replaces `investigation_payload` with no check, while `last_seen` is monotonic. So a retry, or a late alert, overwrites newer evidence shown to analysts.

**[R] Compare-and-set on event time,** in the marked template:

```cypher
WITH coalesce($event_time, datetime()) AS dt
MATCH (e:AttackEvent {event_id:$event_id})
CREATE (:AppliedWrite {request_key:$request_key, applied_at:datetime(), operation:$operation, fingerprint:$fingerprint})
WITH e, dt, (e.payload_event_time IS NULL OR dt >= e.payload_event_time) AS newer
SET e.last_seen = CASE WHEN e.last_seen IS NULL OR dt > e.last_seen THEN dt ELSE e.last_seen END,
    e.occurrences = coalesce(e.occurrences,0) + 1,
    e.tps = coalesce(e.tps,0) + $tps,
    e.investigation_payload = CASE WHEN newer THEN $investigation_payload ELSE e.investigation_payload END,
    e.payload_event_time = CASE WHEN newer THEN dt ELSE e.payload_event_time END
// The Technique and Campaign clauses of update_duplicate_event follow, unchanged
```

[F] The current statement is at `neo4j_client.py:update_duplicate_event` L712-748. Only its first `SET` changes, plus the marker line. [U] The exact text is settled in WP-M2, and it has to keep `WITH e, dt` before the later clauses.

| Design point | Reason |
|---|---|
| **A new property,** `payload_event_time` | `last_seen` is moved by other writers, so it can't say whose payload is held |
| **Null accepted** | Existing events have none, and the first write sets it |
| `>=` on equal times | [F] Alerts share milliseconds. The later-processed wins, and both are that instant's evidence |
| **The count always applies** | An older alert still happened |
| **Surfacing** | Only `retry` can present older data, in file order. After a retry, a read of `payload_event_time` shows whether it was kept, reported as `RETRY_PAYLOAD_NOT_APPLIED`, a note and not a failure |

[U] **X2 must confirm the `WITH ... AS newer` value is computed before the `SET`,** on the disposable server.

**[F] It changes behaviour for late alerts, too,** which campaign decision CA-D8 leaves open, and it interacts with CA-D12-F5: **OD-8.** [F] 0 late alerts in the sample.

### Which writers need it

| Writer | Field overwritten | Needs |
|---|---|---|
| **W07** | `investigation_payload` | **The CAS above** |
| **W08** | `r.latest_payload` | The same, row by row, with the batch's latest time |
| W04 | Status, `last_technique` | [R] An event-time guard when migrated |
| W03, W02, W15 | Status, `closed_at`, reopen | [R] A guard. A delayed replay reverses a lifecycle change |
| W09, W13, W14 | Risk, prediction, CTI | [R] A guard, or refuse automatic retry |
| W17, W18 | `last_seen`, not monotonic | [R] Make monotonic when migrated |
| W01, W05, W06, W10, W16 | Create or add | Marker covers them. No overwritten field |
| W11, W12, W19, W20 | Convergent | Not needed |

**Until a writer has its guard, the registry marks it `retry_requires_review`,** and `retry` says so, rather than running it silently. Conflicts surface in `inspect`.

## Request identity cases

| Case | Handling | Layer |
|---|---|---|
| Same key, same payload | `DUPLICATE`, then by state | Ledger |
| Same key, different payload | `RequestConflict`, audited. Nothing runs | Ledger, then marker |
| Different keys, same payload | Both apply. [F] E7 | By design |
| One alert, several operations | Separate keys, separate requests. **Not duplicates** | Registry |
| Repeated delivery after a restart | Refused at the door | Backlog |
| Stale retry over a newer payload | CAS | Template |
| Ambiguous, may still commit | Held. [F] No retry until a person acts. Earliest review is dispatch + 8 s + margin. **The constraint catches a mistake** | Ledger, marker |
| Human-resolved | Path A or B, per 2.6B | Journal, ledger |
| Marked key with a mismatch in a later code version | Conflict, by design. Drain first | Marker |
| No identity | D14. No key, no coordinated write, counted | Registry |

## Transaction timeouts and lock contention

[F] Production: `db.transaction.timeout = 0s`, `db.lock.acquisition.timeout = 0s`. **No setting changes.**

| Item | Design | Status |
|---|---|---|
| **Transaction timeout** | **8 s, the helper's default,** per transaction | [F] `neo4j_client.py:_execute_write` L289-360 sends it |
| **Maximum duration, one call** | [F] About 22 to 23 s by the helper's own calculation with defaults, and it isn't a deadline. A silent stalled read is unbounded | Watchdog |
| **Lock acquisition** | **No per-transaction setting exists in the helper,** and the server's is 0. **The transaction timeout is the only limit** | [U] **X1 must prove a lock wait ends at the transaction timeout** |
| **Retry after a timeout** | Before COMMIT: `NOT_COMMITTED`, **held for authorization.** No automatic retry, per 2.5 and D5 | [F] Classification tested |
| **Marked writes** | Stay on the helper. A registry test forbids `session.run` for them | Your safeguard |

### When a legacy writer holds a lock

[F] Unmigrated writers use `session.run` with no timeout, so such a transaction can hold a lock for as long as it runs.

| Effect | Behaviour |
|---|---|
| The marked write waits | Up to 8 s, then `NOT_COMMITTED`, if X1 confirms |
| The one worker is blocked | **Up to that limit per alert.** Backlog grows, and nothing is lost |
| Repeated | [R] 3 such timeouts in a row trips the health gate, and `SHOW TRANSACTIONS` is logged as a diagnostic, read only |
| **[F] Same-worker writers** | Run one after another, so they can't block each other |
| **Flush and maintenance threads** | **Can.** These are the real contenders. [U] X1 includes them |

**[H] Deadlocks** come back as transient errors, which the driver retries inside the attempt, before COMMIT.

### Effect on admission

**A blocked worker doesn't slow the reader.** Admission continues to the backlog, up to its limits. That's the point of separating them.

## Retention

| Item | Rule |
|---|---|
| Markers | **Kept.** No deletion without a separately approved design |
| Ledger rows | Kept: D12 |
| Journal rows | Kept: D12 |
| Backlog keys | [R] **Kept as long as markers,** because layer 1 depends on them. Hash and key only: about 150 bytes a row |
| Backlog `raw_line` | Dropped on `DONE`, `SKIPPED`. Wazuh's archive still holds the alert |

[M] Growth at the archive's rate, about 76,000 alerts a month on average, is small. On a day like 20 April it's about 380,000 keys. **Real growth comes from B6, and OD-6 decides.**
