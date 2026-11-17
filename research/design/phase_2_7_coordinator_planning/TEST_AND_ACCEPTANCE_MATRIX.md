# Test and acceptance matrix

Planning only. **None of these tests exists yet, and none was run in this phase.** Package names are in `IMPLEMENTATION_WORK_PACKAGES.md`.

| Class | Meaning | Safe to run during implementation? |
|---|---|---|
| **U** Unit | Pure logic, real SQLite stores in temp directories | Yes |
| **F** Fake-driver | Real journal and ledger, fake helper and fake marker reader | Yes |
| **D** Disposable Neo4j | A throwaway 5.26.27 server, separate ports, removed after | Yes, with authorization to start one |
| **R** Production-scale rehearsal | A disposable copy, at production's shape and size | **Needs a copy you authorize** |
| **L** Live-production validation | The real system | **Not safe now. Needs explicit approval,** and only after everything else |

**●** primary, **○** secondary or confirming.

## Your required cases

| ID | Scenario | U | F | D | R | L | Expected result | WP |
|---|---|---|---|---|---|---|---|---|
| T-1 | Same key, same fingerprint | ● | ● | ○ | | | `DUPLICATE`, by state. No second run | C1 |
| T-2 | Same key, different fingerprint | ● | ● | ○ | | | `RequestConflict`, audited. Nothing runs. Marker: `MARKER_CONFLICT` | C1, M2 |
| T-3 | Different keys, identical payloads | ● | ● | ● | | | Both apply | C1, M2 |
| T-4 | Duplicate delivery after a restart | ● | ● | ○ | | | Refused at the backlog. Without the backlog, the ledger answers by state | T1, T2 |
| T-5 | Commit succeeds, client sees ambiguous | | ● | ● | ○ | | `AMBIGUOUS`, `UNCERTAIN`. Marker present. Nothing auto-resolves. A retry can't apply twice | M2, R1 |
| T-6 | Commit never reaches Neo4j | | ● | ● | | | `AMBIGUOUS` or `NOT_COMMITTED`, by classification. No marker, no effect | M2 |
| T-7 | Marker present after ambiguous | | ● | ● | | | Evidence supports `FOUND_COMMITTED`. **A person still decides** | R1 |
| T-8 | Marker absent while an earlier transaction is in flight | | ● | ● | | | Tool says "not proof", with the earliest review time. A later commit makes it `APPLIED` | R1 |
| T-9 | Lookup unavailable | | ● | ● | | | `UNKNOWN`. Nothing concluded. Dead-server and deadline cases | M1 |
| T-10 | Constraint missing | | ● | ● | ○ | | **Marker mode refused** | M1 |
| T-11 | Constraint violation, marker present | | ● | ● | ○ | | Fingerprint matches: `ALREADY_APPLIED`. Differs: `MARKER_CONFLICT` | M1, M2 |
| T-12 | Constraint violation, marker absent | | ● | ● | | | Another constraint: `EXECUTED_NOT_COMMITTED`. Never "already applied" | M1, M2 |
| T-13 | Marker and mutation atomicity | | | ● | ● | | Agree in every outcome, including a kill. Target missing: neither | M2 |
| T-14 | Concurrent same key | ● | ● | ● | ● | | One claim wins. Bypassing the ledger: one marker, one effect | C1, M2 |
| T-15 | Cross-process same key | ● | | ● | ● | | The same across 4 processes | C1, M2 |
| T-16 | Cross-operation re-delivery | ● | ● | ● | | | The W06 creator's alert re-read: refused at the door. **`occurrences` stays put** | T2, T4 |
| T-17 | Stale `investigation_payload` retry | | | ● | | | Count rises. Newer payload stays. Note reported | M2 |
| T-18 | Legacy writer lock contention | | | ● | ● | | **X1.** Ends at the transaction timeout, or BL-3's alternative is used. Classified correctly | M2 |
| T-19 | Neo4j unavailable | | ● | ● | | | Gate opens. Worker holds. Reader continues. Nothing discarded. Resumes in order | T4 |
| T-20 | Journal unavailable | ● | ● | | | | `NOT_DISPATCHED_UNAVAILABLE` before dispatch. After a commit: `COMMITTED_UNRECORDED`, never re-run | C1, T4 |
| T-21 | Ledger unavailable | ● | ● | | | | The same rule. Journal outcome preserved for recovery | C1, T4 |
| T-22 | Process crash at every transition | ● | ● | ● | ○ | | Every row in `COORDINATOR_STATE_MACHINE.md`. **Zero re-executions** | C1, R1, T4 |
| T-23 | Queue full | ● | | | | | **Reader pauses. Offset unchanged. Alerts stay in the file.** Resumes with hysteresis | T2, T4 |
| T-24 | Rotation while a backlog exists | ● | | | | ○ | Finishes the old file. Old file missing: alarm, recorded gap | T3 |
| T-25 | Durable backlog recovery after restart | ● | ● | ● | ○ | | `RECEIVED` resumes in order. `PROCESSING` becomes `INTERRUPTED` | T1, T4 |
| T-26 | Shutdown during an in-flight transaction | | ● | ● | | | Waits for the outcome. Past the limit: stays `IN_FLIGHT`, recovery makes it uncertain | T4 |

## Crash points, for T-22

Each is an `os._exit` on real SQLite files. Every one has an expected combined state from the state-machine document.

| # | Between | Expected after recovery |
|---|---|---|
| 1 | `register` and `create_intent` | `REGISTERED_NO_INTENT`, can't run. Resubmit creates it |
| 2 | `create_intent` and `begin_execution` | Intent reused |
| 3 | `begin_execution` and `mark_in_flight` | `UNCERTAIN`, `CLAIMED_NOT_DISPATCHED` |
| 4 | `mark_in_flight` and dispatch | Uncertain, no marker |
| 5 | During the transaction | Uncertain, marker either way |
| 6 | Commit and journal outcome | Uncertain, **marker present** |
| 7 | Journal outcome and ledger outcome | Ledger finished from the journal, at that attempt |
| 8 | Constraint refusal and marker evidence recorded | Journal outcome missing: uncertain, and the marker then shows it |
| 9 | Journal `NOT_COMMITTED` with evidence, before ledger `COMPLETED` | `record_completed` from the diagnostic |
| 10 | Resolve: journal and ledger | `RESOLUTION_INCOMPLETE` |
| 11 | Ledger authorize and epoch creation | Authorization survives |
| 12 | Epoch creation and claim | The epoch's reused, never a second |
| 13 | Backlog: admit batch and offset save | Repeats refused at the door |
| 14 | Backlog: offset save and next read | Nothing lost |
| 15 | Backlog: claim and first write | `INTERRUPTED` |
| 16 | Backlog: last write and `DONE` | `INTERRUPTED` |
| 17 | `DONE_BUFFERED` and flush | `INTERRUPTED`, showing CA-D7's loss |

## Additional cases

| ID | Scenario | Class | Expected |
|---|---|---|---|
| T-27 | `COMMITTED_NO_EFFECT` | F, D | `COMPLETED`, error logged, no marker |
| T-28 | Path A and Path B, end to end | F, D | Attempts rise. Record and epoch rules held |
| T-29 | A wrong `FOUND_NOT_COMMITTED`, retried | D | Constraint refuses. `ALREADY_APPLIED`. One effect |
| T-30 | Label exists, constraint doesn't | D | Refused, with an alarm |
| T-31 | Duplicate markers restored from a dump | D | `INCONSISTENT`. Marker mode stops |
| T-32 | Marker deleted, ledger `COMPLETED` | D | Alarm. Never re-run |
| T-33 | Template changed with an open request | F | Conflict. Drain required |
| T-34 | A canary value in payload and manager | U, F | Never in a log or report |
| T-35 | Cypher in alert fields | U, D | Stored as data |
| T-36 | Oversized, invalid UTF-8, deep or non-object lines | U | Rejected individually. Reader continues |
| T-37 | Stores from different deployments | U | Startup refuses |
| T-38 | Marker-present read paths | D, R | All dashboard and feature results identical |
| T-39 | `process_alert` raising, skipped, rejected | U | `FAILED_HELD`, `SKIPPED`, by `trace` |
| T-40 | Disk full during each store's write | U | Holds, or pauses. Nothing lost |
| T-41 | Second listener started | U | Refuses to start |
| T-42 | Back-to-back authorized retries | F | Each needs its own authorization |

## What can't be run safely now

| Item | Why | When |
|---|---|---|
| **Everything, in this phase** | This is planning only. Nothing's implemented | Each with its package |
| **All D tests** | They need a disposable server start, authorized per package | WP-M1 on |
| **All R tests** | **They need a production data copy** | WP-M3 |
| **All L checks** | They touch production | After WP-M3. **Each needs your explicit approval** |
| T-24 live | Needs a real rotation | X9, then Phase 2.8 |
| T-18 against real legacy writers | Disposable tests use legacy-style writers, not production's traffic mix | R, then observed in Phase 2.8 |
| X3, X6, X9 | Read production files | Each with your authorization |

## Live validation, for later

Listed so it's on record. **None of it is authorized.**

| Check | Needs |
|---|---|
| Fixed listener, reading the real file | Your decision on the rollout, and 2.6C's gaps |
| Constraint exists, verified | The rollout steps in `MARKER_AND_RECOVERY_DESIGN.md` |
| Arrival rate and backlog in real traffic | Monitoring in place |
| Real rotation | One real event, observed |

## Phase-level acceptance

| Gate | Criteria |
|---|---|
| **Before any writer touches it** | All U, F and D pass. Zero re-executions. Zero marker disagreements. X1, X2 answered |
| **Before a production constraint** | All R pass. Backup restored. Operator steps rehearsed |
| **Before ingestion uses the backlog** | T-16, T-19, T-22 to T-26 pass. OD-1 to OD-5 decided. Benchmark reported |
| **Before W07U** | "Phase 2.8 prerequisites" in the report |
