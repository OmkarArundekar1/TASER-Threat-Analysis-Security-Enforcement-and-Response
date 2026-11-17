# Cross-component state transitions

Phase 2.5, design only. **Nothing here is implemented.** This table is **[P] proposed** behaviour built only from transitions the journal and ledger already support ([V] `neo4j_write_journal.py` L115-123, `neo4j_idempotency_ledger.py` L142-152). Where something more is needed, it's marked **DESIGN CHANGE** with the `AC-n` from `COORDINATOR_INTERFACE_PROPOSAL.md`.

**Columns:** L is the ledger state, J is the journal state, Att is the ledger attempt, Run is whether Neo4j execution is allowed now, Auto is whether an automatic retry is allowed, and Manual is whether reconciliation by a person is required.

## Matrix

| # | Scenario | L | J | Att | Run | Auto | Manual | Evidence for the transition |
|---|---|---|---|---|---|---|---|---|
| 1 | New logical request | `REGISTERED` | `PENDING` | 0 | Yes, once claimed | n/a | No | `register` returned `NEW`, and `create_intent` committed |
| 2 | Existing request, identical payload | Unchanged | Unchanged | Unchanged | Depends on state: see below | No | Depends | `register` returned `DUPLICATE` |
| 3 | Existing request, conflicting payload | Unchanged, conflict row added | Unchanged | Unchanged | **No** | No | **Yes** | `IdempotencyConflictError` with differences |
| 4 | Registered, not yet executed | `REGISTERED` | `PENDING` or none | 0 | Only by an explicit submit | No | No, reported | Records exist; no claim |
| 5 | Journal intent created, ledger update fails | `REGISTERED` | `PENDING` | 0 | **No** for this call | No | No | `begin_execution` raised `LedgerWriteError`, which rolls back |
| 6 | Ledger registered, journal creation fails | `REGISTERED` | None | 0 | **No** | No | No, resubmit | `create_intent` raised. **DESIGN CHANGE:** AC-1 and AC-2 |
| 7 | Execution begins | `IN_PROGRESS` | `IN_FLIGHT` | n | Yes, this attempt only | No | No | Claim won, and the journal committed |
| 8 | Neo4j confirms commit | `COMPLETED` | `COMMITTED` | n | No | No | No | `WriteOutcome.COMMITTED` |
| 9 | Neo4j confirms non-commit | `NOT_COMPLETED` | `NOT_COMMITTED` | n | Not without authorization | No (D5) | Yes, to authorize | `WriteOutcome.NOT_COMMITTED` |
| 10 | Connection failure before execution | `NOT_COMPLETED` | `NOT_COMMITTED` | n | Not without authorization | No (D5) | Yes, to authorize | `NOT_COMMITTED` with 0 helper attempts |
| 11a | Connection failure during execution, before COMMIT | `NOT_COMPLETED` | `NOT_COMMITTED` | n | Not without authorization | Driver only, inside the attempt | Yes, to authorize | `NOT_COMMITTED`, COMMIT never sent |
| 11b | Connection failure with COMMIT sent | `UNCERTAIN` | `AMBIGUOUS` | n | **No** | **No** | **Yes** | `AMBIGUOUS`, `IncompleteCommit` |
| 12a | Client timeout waiting on a worker | `IN_PROGRESS` | `IN_FLIGHT` | n | **No** | **No** | Not yet | Nothing changes: a timeout concludes nothing |
| 12b | Server timeout at COMMIT | `UNCERTAIN` | `AMBIGUOUS` | n | **No** | **No** | **Yes** | `AMBIGUOUS`, `TransactionTimedOutClientConfiguration` |
| 13 | Neo4j commits, journal update fails | `IN_PROGRESS` | `IN_FLIGHT` | n | **No** | **No** | Yes, if the retry of the journal write also fails | `mark_committed` raised. Retrying the journal write isn't a Neo4j retry |
| 14 | Journal commit recorded, ledger completion fails | `IN_PROGRESS`, then `COMPLETED` by recovery | `COMMITTED` | n | **No** | No, it's reconciled | No | Journal `COMMITTED` with a matching attempt |
| 15a | Crash before Neo4j, no claim | `REGISTERED` | `PENDING` or none | 0 | Only by an explicit submit | No | No, reported | No claim recorded |
| 15b | Crash after claim, before journal `IN_FLIGHT` | `UNCERTAIN` | `PENDING` | n | **No** | **No** | **Yes** | Recovery. The helper can't have run, but that's left for a person (D5) |
| 15c | Crash after both, before the helper | `UNCERTAIN` | `AMBIGUOUS` | n | **No** | **No** | **Yes** | Owner's lock is free. Records can't show it never ran |
| 16 | Crash during execution | `UNCERTAIN` | `AMBIGUOUS` | n | **No** | **No** | **Yes** | Owner's lock is free |
| 17 | Crash right after Neo4j commit | `UNCERTAIN` | `AMBIGUOUS` | n | **No** | **No** | **Yes** | Owner's lock is free. The data is in Neo4j, and only a query can show it |
| 18 | Recovery finds unfinished records | See the algorithm in `FAILURE_WINDOW_AND_RECOVERY_ANALYSIS.md` | | | **No** | **No** | For what it reports | Ordered reconciliation |
| 19a | Operator resolves: found committed | `COMPLETED` | `RESOLVED`, `FOUND_COMMITTED` | n | No | No | Done | Principal, reason and evidence |
| 19b | Operator resolves: found not committed | `NOT_COMPLETED` | `RESOLVED`, `FOUND_NOT_COMMITTED` | n | Not until authorized | No | Done, then authorize | As above |
| 19c | Operator resolves: abandoned | `ABANDONED` | `RESOLVED`, `ABANDONED` | n | Never | No | Done | As above |
| 20a | Operator authorizes retry, journal `NOT_COMMITTED` | `RETRY_AUTHORIZED` | `NOT_COMMITTED` | n | Yes, once claimed. Then L `IN_PROGRESS` n+1, J `IN_FLIGHT` by `authorize_retry` | No | Done | Named principal and reason |
| 20b | Operator authorizes retry, journal `RESOLVED` | `RETRY_AUTHORIZED` | `RESOLVED`, then a new `PENDING` record | n | Yes, once claimed. Then L `IN_PROGRESS` n+1, new J `IN_FLIGHT` | No | Done | As above. **New journal epoch,** D4 |
| 21 | Stale attempt reports after a newer one exists | Unchanged | Unchanged | Newer | No | No | Logged | `StaleAttemptError`. **DESIGN CHANGE:** the journal is fenced by a check first, and atomically with AC-3 |

## Scenario 2: what a duplicate does

The combined state decides. [V] `DUPLICATE` itself grants nothing.

| Ledger state found | Journal | Coordinator's answer | Runs? |
|---|---|---|---|
| `REGISTERED` | `PENDING` | Proceed to claim | Yes |
| `REGISTERED` | None | Create the intent, then claim | Yes |
| `IN_PROGRESS` | Any | `IN_PROGRESS_ELSEWHERE` | No |
| `COMPLETED` | Any | `ALREADY_COMPLETED`. [L] It can't return rows from the original run | No |
| `NOT_COMPLETED` | `NOT_COMMITTED` or `RESOLVED` | `AWAITING_AUTHORIZATION` | No |
| `RETRY_AUTHORIZED` | Any | Proceed to claim | Yes |
| `UNCERTAIN` | Any | `BLOCKED_UNCERTAIN` | No |
| `ABANDONED` | Any | `BLOCKED_ABANDONED` | No |

## Helper outcome mapping

| `WriteOutcome` | Journal call | Ledger call |
|---|---|---|
| `COMMITTED` | `mark_committed` | `record_completed` |
| `NOT_COMMITTED` | `mark_not_committed` | `record_not_completed` |
| `AMBIGUOUS` | `mark_ambiguous` | `record_uncertain` |
| `BaseException` escaped the helper | `mark_ambiguous`, best effort | `record_uncertain`, best effort. Otherwise recovery |

[P] Error details passed along are only `exception_type`, `exception_code` and `diagnostic`. [V] Both stores refuse an exception object, because its message can quote values.

## Resolution mapping

| Operator's decision | Journal `resolve_ambiguous` | Ledger `resolve_uncertain` |
|---|---|---|
| Found committed | `FOUND_COMMITTED` | `FOUND_COMPLETED` |
| Found not committed | `FOUND_NOT_COMMITTED` | `FOUND_NOT_COMPLETED` |
| Abandoned | `ABANDONED` | `ABANDONED` |

## Combinations that shouldn't arise

If recovery finds one of these, it reports `INCONSISTENT`, changes nothing, and asks a person. **It never resolves one towards running.**

| Ledger | Journal | What it means |
|---|---|---|
| `COMPLETED` | `IN_FLIGHT` or `PENDING` | The ordering rule was broken, or a store was restored from backup |
| `COMPLETED` | `NOT_COMMITTED` or `AMBIGUOUS` | The same |
| `RETRY_AUTHORIZED` | `IN_FLIGHT` | The same |
| `REGISTERED` | `IN_FLIGHT`, `COMMITTED`, `NOT_COMMITTED`, `AMBIGUOUS` | The same |
| `ABANDONED` | `COMMITTED` | A person abandoned something the journal says committed |
| None | Any | A journal record with no ledger request: ledger restored, or damaged |
| Any | Journal attempt tag doesn't match the ledger attempt | Stale data, or a restored backup |
