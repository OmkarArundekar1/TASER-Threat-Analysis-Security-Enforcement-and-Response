# D4: retry lifecycle

Implemented in `neo4j_write_journal.py:NeoWriteJournal.create_intent` L1129-1170 and `neo4j_write_journal.py:NeoWriteJournal._admit_epoch` L1025-1044. Backend paths are relative to `isfcr/CYUKTI/backend/`. **[V]** verified by tests.

## The model

| Rule | Status |
|---|---|
| A resolved intent stays final | [V] Unchanged: `RESOLVED` has no outgoing transition |
| A later authorized retry creates a new intent | [V] Epoch n + 1 |
| It's linked by the same key | [V] |
| It carries the same operation and payload | [V] Compared by `operation` and `intent_sha256`, in the insert's transaction |
| Earlier intent and history stay immutable | [V] Compared before and after |
| The journal performs no retry, replay or execution | [V] A new intent stays `PENDING` until a caller moves it. No execute path exists |

## When a new intent is allowed

```python
journal.create_intent(previous.operation, previous.statements, return_rows=previous.return_rows,
                      correlation_key=key, epoch=previous.epoch + 1, authorized_by=..., reason=...)
```

| Previous epoch | Result | Code |
|---|---|---|
| None, and epoch 1 is asked for | Created | |
| Exists at the epoch asked for | Refused | `DUPLICATE_CORRELATION` |
| Epoch isn't highest + 1 | Refused | `EPOCH_OUT_OF_ORDER` |
| `PENDING`, `IN_FLIGHT`, `COMMITTED`, `NOT_COMMITTED` or `AMBIGUOUS` | Refused | `PREVIOUS_NOT_RESOLVED` |
| `RESOLVED`, `FOUND_COMMITTED` or `ABANDONED` | Refused | `PREVIOUS_NOT_RETRYABLE` |
| `RESOLVED`, `FOUND_NOT_COMMITTED`, different operation or intent | Refused | `INTENT_MISMATCH` |
| `RESOLVED`, `FOUND_NOT_COMMITTED`, same intent, authorization stated | **Created** | |

**What "identical" means is strict.** Tests refuse: another query, value, instant, or time zone on the same instant, an `int` becoming a `float`, a tuple becoming a list, a different parameter order, an extra statement, another operation, or a different `return_rows`.

[V] A record's stored statements rebuild its exact hash, for every value type in the journal's preservation table.

## Authorization

[V] `authorized_by` and `reason` are required for any later epoch, refused on a first epoch, and kept in the new record's first history entry. Integrity checks enforce both.

**Limit:** the journal can only require that authorization was stated. It can't confirm it, because that's in the ledger. **This is an addition to 2.6A's signature, for your confirmation.**

## Path A: a confirmed non-commit keeps its record

[V] Per Phase 2.5 and 2.6A, `NOT_COMMITTED` reuses the record through `authorize_retry`, with a higher attempt, and history keeps both. Creating a new intent from it is refused.

**Your brief lists "new intent after confirmed non-commit and explicit retry authorization" as a test.** I followed the contract, and tested that:

- A helper-confirmed `NOT_COMMITTED` retries on the same record (`test_a_confirmed_non_commit_retries_on_the_same_record_and_gets_no_new_intent`).
- A non-commit concluded by a person gets the new intent (`test_a_new_intent_after_a_resolved_non_commit_keeps_the_original_exactly`).

**If you meant the first case to create a new intent, that's a contract change.** Please say so.

## The seven crash windows

[V] A real child process runs the real journal and real ledger on separate files, stops at each durable step, and dies by SIGKILL. A fresh process runs both recoveries, checks the pair, and continues. All in `tests/test_neo4j_write_journal_lifecycle.py`.

| Window | J after recovery | L after recovery | Can claim? | How it continues |
|---|---|---|---|---|
| 1 Journal resolved, ledger not | `RESOLVED` | `UNCERTAIN` | No | Repeat the ledger's resolve. A different journal decision is refused |
| 2 Authorized, no new intent | `RESOLVED` | `RETRY_AUTHORIZED` | One attempt | Create e2, claim, run |
| 3 New intent, no claim | `RESOLVED`, `PENDING` | `RETRY_AUTHORIZED` | One attempt | `DuplicateCorrelationError` and preflight both name e2, which is reused. **No second intent** |
| 4 Claim, before `IN_FLIGHT` | `RESOLVED`, `PENDING` | `UNCERTAIN`, a2 | No | Resolve in the ledger alone. a3 starts the same e2. **A gap in attempts** |
| 5 Claim, before journal retry (Path A) | `NOT_COMMITTED`, a1 | `UNCERTAIN`, a2 | No | **Legal pair 1.** The lower attempt shows this claim didn't run. a3 retries the same record |
| 6 `IN_FLIGHT`, helper unknown | `RESOLVED`, `AMBIGUOUS` | `UNCERTAIN`, a2 | No | Resolve both, authorize, create e3 |
| 7 Outcome in journal, not ledger | `RESOLVED`, outcome | `IN_PROGRESS`, a2 | No | `record.ledger_attempt` equals the ledger's, so the outcome can be mirrored. Tested for all three outcomes |

**Legal pair 2,** `test_legal_pair_ledger_not_completed_with_journal_pending`: L `NOT_COMPLETED`, J `PENDING`, after a claim whose `IN_FLIGHT` failed. The next authorized claim starts the same record with `mark_in_flight`.

**What these show, and what they don't.** They show the journal supports each state and that none of them run anything. The stepping code is test code. **A coordinator doesn't exist,** and these aren't a distributed transaction: the two stores commit separately, which is what each window tests.

## Required cases

| Case from the brief | Test |
|---|---|
| New intent after confirmed non-commit and authorization | See "Path A" above |
| After human resolution of an ambiguous outcome | `test_a_new_intent_after_a_resolved_non_commit_keeps_the_original_exactly` |
| Without authorization evidence | `test_a_new_intent_without_stated_authorization_is_refused` |
| Payload mismatch across epochs | `test_a_new_epoch_cannot_change_the_intent_in_any_way`, `test_a_new_epoch_cannot_change_the_operation_or_return_rows` |
| Skipped or regressed epoch | `test_a_skipped_epoch_is_refused`, `test_a_regressed_or_repeated_epoch_is_refused` |
| Reuse of a previous intent identity | `test_an_intent_identity_can_never_be_supplied_or_reused` |
| Stale outcome after a newer intent | `test_a_stale_outcome_after_a_newer_intent_exists_changes_neither_record`, `test_a_stale_outcome_from_before_a_crash_cannot_reach_the_newer_epoch` |
| Crash between creation and the next ledger step | `test_window_3_new_intent_with_no_claim_is_reused_and_not_duplicated` |
| Concurrent creation of the next epoch | Threads, two instances, and `test_processes_creating_the_same_next_epoch_produce_one_record` |

## Ledger

**Unchanged, and no change was needed.** [V] Its hash is the same before and after. The future coordinator is responsible for sequencing the two stores.
