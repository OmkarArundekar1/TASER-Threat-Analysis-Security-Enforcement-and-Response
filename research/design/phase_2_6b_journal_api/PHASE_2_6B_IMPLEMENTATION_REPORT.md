# Phase 2.6B implementation report

Journal API coordination and retry lifecycle. Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Labels:** **[V]** verified behaviour, **[M]** measured result, **[A]** assumption, **[Q]** outstanding question.

## Result

**AC-1, AC-2, AC-3 and D4's new-intent lifecycle are implemented in the journal, with a tested v1 to v2 migration.** Nothing production imports the journal, so **production writes are still not protected by it.** That needs the coordinator, which wasn't built.

| Acceptance criterion | Status | Evidence |
|---|---|---|
| AC-1, AC-2, AC-3 implemented | Met | `AC1_CORRELATION_METADATA.md`, `AC2_PREFLIGHT_CONTRACT.md`, `AC3_ATTEMPT_FENCING.md` |
| D4 new-intent lifecycle implemented | Met, **with one question for you** (below) | `D4_RETRY_LIFECYCLE.md` |
| Legacy uncorrelated behaviour compatible | Met | `COMPATIBILITY_AND_REGRESSION.md` |
| History and hash-chain integrity preserved | Met | `SCHEMA_MIGRATION_REPORT.md` |
| Migration from the previous schema tested | Met, using the previous implementation itself | `SCHEMA_MIGRATION_REPORT.md` |
| Stale outcomes can't modify newer attempts | Met | `AC3_ATTEMPT_FENCING.md` |
| A new epoch can't silently change the intent | Met | `D4_RETRY_LIFECYCLE.md` |
| Preflight demonstrably side-effect-free | Met | `AC2_PREFLIGHT_CONTRACT.md` |
| Focused, migration, fault and regression results recorded | Met | `TEST_AND_FAULT_RESULTS.md` |
| No production writer, listener, coordinator or ledger change | Met | `COMPATIBILITY_AND_REGRESSION.md` |
| Working tree and protected evidence accounted for | Met | `COMPATIBILITY_AND_REGRESSION.md` |

## Source files changed

| File | Change |
|---|---|
| `neo4j_write_journal.py` | The implementation: 464 lines added, 33 removed |
| `tests/test_neo4j_write_journal.py` | Two pins only: the schema version at L75, and the public method set at L799-801 |
| `tests/test_neo4j_write_journal_correlation.py` | New |
| `tests/test_neo4j_write_journal_migration.py` | New |
| `tests/test_neo4j_write_journal_lifecycle.py` | New |
| `tests/_neo4j_journal_lifecycle_child.py` | New: the process that dies |
| `tests/_neo4j_journal_v1_frozen.py` | New: a byte-identical copy of the previous journal, so old files come from the real old code |

**Nothing else in the backend changed,** including `neo4j_idempotency_ledger.py`.

## APIs

```python
NeoWriteJournal.create_intent(operation, statements, *, return_rows=False, correlation_key=None, epoch=None,
                              authorized_by=None, reason=None) -> JournalRecord
NeoWriteJournal.preflight_intent(operation, statements, *, return_rows=False, correlation_key=None, epoch=None,
                                 authorized_by=None, reason=None, ledger_attempt=None) -> PreflightResult
NeoWriteJournal.find_by_correlation_key(key) -> tuple[JournalRecord, ...]
NeoWriteJournal.mark_in_flight(record_id, *, worker_id=None, ledger_attempt=None) -> JournalRecord
NeoWriteJournal.mark_committed(record_id, *, expected_ledger_attempt=None) -> JournalRecord
NeoWriteJournal.mark_not_committed(record_id, *, error_type=None, error_code=None, diagnostic=None, expected_ledger_attempt=None) -> JournalRecord
NeoWriteJournal.mark_ambiguous(record_id, *, error_type=None, error_code=None, diagnostic=None, expected_ledger_attempt=None) -> JournalRecord
NeoWriteJournal.authorize_retry(record_id, *, authorized_by, reason, worker_id=None, ledger_attempt=None) -> JournalRecord
```

**New types:** `PreflightResult`, `PreflightCode`, `StaleAttemptError`, `DuplicateCorrelationError`, `EpochChainError`. **New fields:** `correlation_key`, `epoch` and `ledger_attempt` on `JournalRecord` and `JournalEvent`.

## Design conflicts, and how each was settled

The brief says to use Phase 2.6A as the source of truth and not to invent policy. These are the places where the documents disagreed or left a gap.

| # | Conflict | What I did | Needs you? |
|---|---|---|---|
| 1 | **Brief's test list:** "new intent after confirmed non-commit and explicit retry authorization". **Phase 2.6A Path A:** a confirmed `NOT_COMMITTED` reuses its record, and only `RESOLVED` leads to a new intent | Followed 2.6A. A new intent from a `NOT_COMMITTED` record is **refused**, and it's tested both ways. A non-commit concluded by a person (`RESOLVED`, `FOUND_NOT_COMMITTED`) gets the new intent | **Yes:** confirm this reading |
| 2 | **Brief:** test a new intent "without explicit authorization evidence". **2.6A's `create_intent`** has no such argument | Added `authorized_by` and `reason`, required for epochs after the first and kept in the first history entry. It follows the journal's own `authorize_retry` pattern. **The journal can only require that it was stated:** it can't see the ledger | **Yes:** this is an addition to the contract |
| 3 | **Brief AC-1:** correlated intents supply "all required correlation arguments together", including the attempt. **2.6A:** the intent is created before the claim, so the attempt doesn't exist yet | Key and epoch come together at creation. The attempt arrives at `mark_in_flight` or `authorize_retry`, and is required there | Confirm |
| 4 | **2.6A AC-2:** a pure module function, `check_statements`, no file access. **Brief AC-2:** must check journal state and conflicts, and return a structured result | Followed the brief: a read-only method, `preflight_intent`. It runs the same code as `create_intent` | No |
| 5 | **2.6A:** refuse v1 files rather than write a migrator. **Brief:** migration required and tested | Followed the brief | No |
| 6 | **2.6A:** a store can't enforce attempts across epochs, so the coordinator checks. **Brief AC-3:** the journal enforces | Followed the brief: an attempt must exceed every attempt on the key, on any epoch. Costs one indexed read per start | No |
| 7 | **2.6A:** a non-rising attempt raises `InvalidTransitionError` | Raises `StaleAttemptError`, a subclass, so callers catching the parent still work | No |

## State-transition rules

**[V] `_TRANSITIONS` is unchanged:** the same seven moves, and the existing test pinning it still passes. There's no second state machine. The rules added are:

| Rule | Where |
|---|---|
| A correlated record must always be given its attempt | `neo4j_write_journal.py:NeoWriteJournal._transition` L1061-1125 |
| An attempt must exceed all attempts seen for the key | `neo4j_write_journal.py:NeoWriteJournal._admit_attempt` L1046-1051 |
| An outcome must name the record's current attempt | `neo4j_write_journal.py:NeoWriteJournal._transition` L1061-1125 |
| No outcome on a correlated record before an attempt started. This removes `PENDING` to `NOT_COMMITTED` for correlated records only | Same |
| A later epoch needs the previous one `RESOLVED` as `FOUND_NOT_COMMITTED`, with the same operation and intent | `neo4j_write_journal.py:NeoWriteJournal._admit_epoch` L1025-1044 |

## Tests

[M] Final recorded runs, all in an empty network namespace, with no service started:

| Run | Result |
|---|---|
| New focused tests | **275 passed** |
| Existing journal, faults, crash and child lifecycle tests | **314 passed** |
| Ledger compatibility | **346 passed** |
| Full backend suite | **2,386 passed, 29 skipped, 0 failed** |
| Mutation check | **82 of 82 caught,** and the no-op control passes |

The full suite's previous baseline was 2,111 passed and 29 skipped. The difference of 275 is exactly the new tests.

## Performance, in one finding

[M] Correlation costs almost nothing over version 1. But the numbers answer an earlier open question: **a whole correlated life in the journal takes a median of 9.2 ms here,** because each of three steps is an fsynced commit. That's about 108 a second, for the journal alone, before the ledger or Neo4j. **W07U's sample peak is 377 a second,** so bursts would queue. This bears on D13's benchmark condition. See `PERFORMANCE_RESULTS.md`.

## Not done, as instructed

Coordinator, listener remediation, `AppliedWrite` markers or constraints, deterministic ids, W07U migration, and any production integration.

## Limits

- [V] **Power loss wasn't tested.** Tests use SIGKILL, which keeps the page cache.
- [V] **Process races show real contention in those runs, and can't force every interleaving.** Where it mattered, one race was also made deterministic.
- [V] **A forged journal can still pass.** There's no secret key, so someone who recomputes every checksum and keeps the contents coherent isn't detected. The content checks catch incoherent forgeries.
- [V] **The test steps stand in for a coordinator's order of calls.** Real coordination is unbuilt, so the seven windows show the journal supports that order, and nothing more.
- [Q] Resolution isn't fenced by attempt, per the contract. Resolving is a person's act on a record that's already `AMBIGUOUS`.

## Outstanding questions

1. Confirm conflict 1: a confirmed non-commit reuses its record.
2. Confirm conflict 2: `authorized_by` and `reason` on later epochs.
3. Confirm conflict 3: the attempt arrives at the start of execution, not at creation.
