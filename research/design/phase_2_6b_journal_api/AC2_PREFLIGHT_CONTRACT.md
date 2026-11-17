# AC-2: preflight contract

Implemented as `neo4j_write_journal.py:NeoWriteJournal.preflight_intent` L1172-1231. Backend paths are relative to `isfcr/CYUKTI/backend/`. **[V]** verified by tests, **[M]** measured.

```python
preflight_intent(operation, statements, *, return_rows=False, correlation_key=None, epoch=None,
                 authorized_by=None, reason=None, ledger_attempt=None) -> PreflightResult
```

## What it answers

"What would `create_intent` do with this now, and could `ledger_attempt` start?" **It says nothing about the future.**

**It doesn't guarantee:** that creation will succeed, since state can change, the disk can fill, or a lock can be held. It doesn't guarantee that the write will ever run. `create_intent` revalidates inside its own transaction.

## Result

```python
PreflightResult(accepted: bool, code: PreflightCode, reason: str,
                intent_sha256: str | None, existing_record_id: str | None)
```

| Code | Meaning | Equivalent in `create_intent` |
|---|---|---|
| `ACCEPTED` | Would be created | Creates |
| `INVALID_ARGUMENT` | Bad operation, `return_rows` or statements, or a one-shot iterator | `ValueError`, `TypeError` |
| `UNSUPPORTED_PARAMETER` | A value can't be restored exactly | `UnsupportedParameterError` |
| `INVALID_CORRELATION` | Partial or malformed key, epoch or attempt | `TypeError`, `ValueError` |
| `AUTHORIZATION_REQUIRED` | Later epoch without `authorized_by` and `reason`, or a first epoch with them | `ValueError`, `TypeError` |
| `DUPLICATE_CORRELATION` | `(key, epoch)` exists. `existing_record_id` names it | `DuplicateCorrelationError` |
| `EPOCH_OUT_OF_ORDER` | Skipped epoch, or later epoch with no chain | `EpochChainError` |
| `PREVIOUS_NOT_RESOLVED` | Previous epoch isn't `RESOLVED` | `EpochChainError` |
| `PREVIOUS_NOT_RETRYABLE` | Previous resolved as `FOUND_COMMITTED` or `ABANDONED` | `EpochChainError` |
| `INTENT_MISMATCH` | Different operation or intent from the previous epoch | `EpochChainError` |
| `STALE_ATTEMPT` | `ledger_attempt` doesn't exceed one already seen | `StaleAttemptError`, at the start of execution |
| `JOURNAL_DAMAGED` | A record for this key fails its checksum | `JournalCorruptionError` |
| `READ_ONLY` | This instance can't create | `JournalReadOnlyError` |

**It still raises** for a closed journal or failing storage, because nothing can be said.

## Consistency with `create_intent`

**Both call the same functions:** `_check_operation`, `_checked_correlation`, `_checked_authorization`, `_serialize_intent`, and `neo4j_write_journal.py:NeoWriteJournal._admit_epoch` L1025-1044. There's no separate copy of the rules to drift.

[V] Verified three ways:

1. 28 specified cases, each asserting the code, then asserting `create_intent` agrees.
2. Payload refusals run through both.
3. **A random walk of 600 proposals** from whatever state it reaches: every acceptance must be created and every refusal refused again. It asserts that the walk reached both answers.

**One deliberate difference:** preflight refuses a one-shot iterator that `create_intent` accepts. Reading it would leave nothing for the real call, which would then be refused, so this is what keeps preflight's acceptance honest. The iterator isn't consumed.

## Side effects

**None.** The reads run in a `BEGIN DEFERRED` transaction, which is always rolled back.

[V] Tests compare before and after, for accepted and refused calls alike:

| Compared | How |
|---|---|
| Every row, including checksums | `SELECT *` on both tables |
| File bytes | SHA-256 of the database and its `-wal`. The `-shm` file is excluded, because it holds read locks rather than data |
| Integrity | `check_integrity` passes |
| No open transaction | `in_transaction` is false after each answer |
| No write lock | It answers while another connection holds `BEGIN IMMEDIATE` |

[V] **Concurrency:** 8 threads make 640 calls with correct answers, and the state is unchanged.

**Parameter values never reach a refusal's text.** Tested with a secret in a refused value.

## Cost

[M] Median 0.040 ms for a new correlated key, and 0.071 ms for a duplicate, at 10,000 records. It's the same at 1,000. See `PERFORMANCE_RESULTS.md`.
