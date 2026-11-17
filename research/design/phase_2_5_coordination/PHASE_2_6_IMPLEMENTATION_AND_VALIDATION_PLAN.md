# Phase 2.6 implementation and validation plan

Phase 2.5, design only. **Nothing here is authorized or started.** Everything in this plan is **[P] proposed.** No test named below exists yet, and none was run for this phase. **[L]** marks a known limitation.

## Stages

Each stage needs its own authorization. Production writers are untouched until stage F.

| Stage | Work | Touches production? | Exit evidence |
|---|---|---|---|
| A | Your decisions on D1, D3, D4, D13 | No | Recorded decisions |
| B | Verify D1 on production-shaped data, read-only | No | Uniqueness across rotation |
| C | API changes AC-1, AC-2, AC-3, plus AC-4 and AC-5 if approved | No: journal and ledger only | Existing tests still pass; schema migration tested |
| D | Build the coordinator, standalone, with its registry | No | Test groups 1 to 7 pass |
| E | Real-server validation on a disposable Neo4j | No | Test group 8 passes |
| F | Migrate W07 on the unattributed path, behind a switch, default off | **Yes** | Test group 9, then your go or no-go |
| G | Further writers in tier order, one at a time | **Yes** | The same evidence for each |

**Not in Phase 2.6:** offset integration (D10) and W08, until D8 is decided.

## Test matrix

**Type:** F is fake-only, S is SQLite-backed, N is real Neo4j, C is process crash, O needs an explicit operator step in the test.

### Group 1: state transitions

| ID | Test | Type |
|---|---|---|
| T1.1 | New request runs once, to `COMPLETED` and `COMMITTED`, with attempt 1 | F, S |
| T1.2 | `NOT_COMMITTED` gives `NOT_COMPLETED` and `NOT_COMMITTED`, and nothing retries | F, S |
| T1.3 | `AMBIGUOUS` gives `UNCERTAIN` and `AMBIGUOUS`, and nothing retries | F, S |
| T1.4 | Every matrix row in `CROSS_COMPONENT_STATE_TRANSITIONS.md` has a test, named by its number | F, S |
| T1.5 | Every disposition is reachable, and each one's "ran or didn't" claim is checked against a recording fake | F, S |
| T1.6 | Invalid combinations give `CoordinationInconsistency`, and change nothing | S |
| T1.7 | Combined state is computed, and there's no column or file holding it | S |

### Group 2: journal and ledger interaction

| ID | Test | Type |
|---|---|---|
| T2.1 | Ledger write always precedes the journal's for registration and claim | F |
| T2.2 | Journal outcome always precedes the ledger's | F |
| T2.3 | A failure at each store call leaves the expected pair, for every call | S |
| T2.4 | The helper is never called unless claim and `IN_FLIGHT` both committed | F |
| T2.5 | Ledger attempt is tagged on each journal `IN_FLIGHT`, across epochs | S |
| T2.6 | Journal `check_statements` refusals never reach `register` | S |
| T2.7 | Journal's stored statements reproduce the ledger's fingerprint, for each supported type | S |
| T2.8 | Second epoch created only from `RESOLVED`, with identical statements | S |
| T2.9 | A failed journal write after commit retries only the journal, and the helper's call count stays 1 | F, S |

### Group 3: duplicates and conflicts

| ID | Test | Type |
|---|---|---|
| T3.1 | Duplicate in each ledger state returns the disposition in the table, and runs only where it says | S |
| T3.2 | Conflict raises, audits, doesn't touch the journal, and doesn't run | S |
| T3.3 | Different keys, same payload: both run | S |
| T3.4 | Registry refuses a missing source, and never generates | F |
| T3.5 | Two different duplicate alerts on one event: both count | S, N |
| T3.6 | Random ids in a payload produce a conflict (proves why AC-7 is needed) | S |

### Group 4: concurrency

| ID | Test | Type |
|---|---|---|
| T4.1 | Threads submitting one key: one runs | S |
| T4.2 | Separate processes submitting one key: one runs | S, C |
| T4.3 | Conflicting payloads across processes: one identity | S, C |
| T4.4 | No lock is held across the helper call: a slow fake doesn't block another key | F |
| T4.5 | Recovery during live work leaves live owners alone | S, C |
| T4.6 | Two recoveries at once convert each request once | S, C |
| T4.7 | Live worker and recovery racing in window 8 settle on the same state | S |

### Group 5: crash injection, one test per failure window

Each one kills a real process with `SIGKILL`, restarts, runs recovery, and checks the pair and a call count on the fake or real target.

| ID | Window | Expected result | Type |
|---|---|---|---|
| T5.1 | 1 Before registration | Nothing recorded | C |
| T5.2 | 2 Registration to intent | `REGISTERED_NO_INTENT`, can't run | C, S |
| T5.3 | 3 Intent to claim | Pending, unchanged | C, S |
| T5.4 | 4 Claim to `IN_FLIGHT` | `CLAIMED_NOT_DISPATCHED`, target not called | C, S |
| T5.5 | 5 `IN_FLIGHT` to helper | Uncertain, target not called | C, S |
| T5.6 | 6 During the transaction | Uncertain | C, N |
| T5.7 | 7 After commit, before journal | Uncertain, **data present, not re-sent** | C, N |
| T5.8 | 8 Journal to ledger outcome | Ledger finished automatically, nothing re-sent | C, S |
| T5.9 | 9 During recovery, at each step | The next run finishes | C, S |
| T5.10 | 10 Between resolution writes | `RESOLUTION_INCOMPLETE`, not runnable, a repeat finishes | C, S, O |
| T5.11 | 11 Authorization to dispatch | Authorization survives, unused | C, S, O |
| T5.12 | 12 New epoch to claim | The record is reused, and there's no second one | C, S, O |
| T5.13 | Random kills | Sound, and no request both `COMPLETED` and running | C, S |

### Group 6: stale outcomes

| ID | Test | Type |
|---|---|---|
| T6.1 | Attempt 1's outcome during attempt 2 is refused by both stores | S |
| T6.2 | Late outcome after recovery marked it uncertain is refused | S |
| T6.3 | A stale call changes neither store, checked by comparing both before and after | S |

### Group 7: manual resolution and retry

| ID | Test | Type |
|---|---|---|
| T7.1 | Each decision gives the mapped pair, with principal, reason and evidence in history | S, O |
| T7.2 | `FOUND_NOT_COMMITTED` doesn't allow running, until `authorize_retry` | S, O |
| T7.3 | Each authorization allows one attempt | S, O |
| T7.4 | `retry` takes no statements, and sends the journal's | S, O |
| T7.5 | A different decision on a half-resolved request raises | S, O |
| T7.6 | `authorize_retry` touches only the ledger | S, O |
| T7.7 | Empty principal, reason or evidence is refused | S |
| T7.8 | No code path reaches `UNCERTAIN` to running without both acts, by random walk | S |

### Group 8: real Neo4j, disposable only

| ID | Test | Type |
|---|---|---|
| T8.1 | Rerun the Phase 2.2.1R fault scenarios through the coordinator: each outcome matches the graph | N |
| T8.2 | COMMIT reply lost: data present, uncertain, not re-sent, and counters not doubled | N, C |
| T8.3 | COMMIT dropped: data absent, uncertain, not re-sent | N, C |
| T8.4 | Authorized retry after a proven non-commit applies once | N, O |
| T8.5 | The real W07 query with a real key, run twice by duplicate submit, counts once | N |
| T8.6 | If D6 is approved: marker present exactly when the write committed, across all fault scenarios | N |
| T8.7 | If D6(c) is approved: a wrongly authorized second run fails on the constraint, and nothing doubles | N, O |
| T8.8 | Verify [A]: W13's `MERGE` with a null target | N |
| T8.9 | Latency: time per request, with and without the coordinator | N |

### Group 9: migrating one writer

| ID | Test | Type |
|---|---|---|
| T9.1 | Switch off: behaviour identical to today, by existing tests | F |
| T9.2 | Switch on: graph result equals today's, on a replay of the same alerts, on a disposable database | N |
| T9.3 | A store failure follows D15, and the alert isn't acknowledged as written | S |
| T9.4 | Crash at each window, using the real writer | C, N |
| T9.5 | Existing dedup, listener and buffer tests unchanged and passing | F |

### Regression, at every stage

Write helper, journal (unit, crash, faults), ledger (unit, crash, faults, concurrency), child lifecycle, production safety, dedup and buffer, and the full suite: currently 2,140 collected, 2,111 passed, 29 skipped.

## Acceptance before any production writer migrates

**All of these, with evidence archived outside the repository:**

1. Groups 1 to 8 pass, with exact counts and logs.
2. Every failure window has a passing crash test using real process death.
3. Real-server runs show that no fault scenario doubled a counter or re-sent a write.
4. Mutation checks cover the coordinator, classified as they were in Phase 2.4, with every survivor explained.
5. D1 is verified on production-shaped data.
6. D15 is decided, and the operator workflow has been tried end to end.
7. The latency cost per alert is measured and accepted by you.
8. A rollback is tested: switch off, and the writer behaves as it does today.
9. Protected files and frozen evidence are verified unchanged.
10. **Your explicit go, for that one writer.**

## Limits of this plan

- [L] Tests can't prove power-loss durability. It needs a separate experiment, and it isn't planned here.
- [L] Tests on one machine can't cover real network partitions.
- [L] Process race tests show real contention in those runs. They can't force every interleaving.
- [L] The 27 unexplained blocked network attempts in older tests are still untraced.
