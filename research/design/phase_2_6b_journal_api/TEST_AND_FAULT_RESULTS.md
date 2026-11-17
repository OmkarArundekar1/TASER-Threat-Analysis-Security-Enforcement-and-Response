# Test and fault results

All recorded runs used an empty network namespace and started no service. **[M]** measured.

## Totals

| Run | Result | Output |
|---|---|---|
| `tests/test_neo4j_write_journal_correlation.py` | 224 passed | `evidence/test_focused_by_file.txt` |
| `tests/test_neo4j_write_journal_migration.py` | 28 passed | Same |
| `tests/test_neo4j_write_journal_lifecycle.py` | 23 passed | Same |
| **All new, one run** | **275 passed** | `evidence/test_focused_output.txt` |
| **Existing journal, faults, crash, child lifecycle** | **314 passed** | `evidence/test_existing_journal_output.txt` |
| **Ledger unit, faults, crash** | **346 passed** | `evidence/test_ledger_output.txt` |
| **Full backend** | **2,386 passed, 29 skipped, 0 failed** | `evidence/test_full_suite_output.txt` |
| Migration and lifecycle tests, 3 more repeats | 51 passed each | `evidence/test_repeat_stability.txt` |

**Skips:** the same 29 count as the baseline before this phase. None comes from these tests.

## Mutation check

[M] 82 deliberate bugs, one at a time, on a copy in a temporary overlay. **82 caught, 0 survived,** and the no-op control passes. Script: `evidence/mutation_check.py`. Output: `evidence/mutation_output.txt`.

**It found real gaps, and I fixed them:**

| Run | Finding | Fix |
|---|---|---|
| 1 | The harness labelled the mutant named "control characters accepted" as its control | Compared by position instead |
| 1 | "event hash ignores correlation" survived: content checks masked it | Direct tests that each correlation field changes both checksums |
| 1 | "attempt rise in one record not checked" survived | A forged-record test |
| 2 | "no recheck under the lock" survived, after being caught in run 1: **that protection rested on a race** | A deterministic test |

Also found before the mutation runs: the original random-kill test ended in an assertion too weak to fail, and it renamed files between rounds. It now keeps one journal, requires over 8 records, and requires that an epoch 2 was reached.

**Limit:** the mutants were chosen by me, so they test what I thought of. They cover the new logic only.

## Fault and crash coverage

| Durable transition | SIGKILL test | Injected failure test |
|---|---|---|
| Create first intent | `dies_creating_first`, no record | Serialization failures and history failure |
| Create next epoch | `dies_creating_next_epoch`, no record | Preflight and create agreement |
| `mark_in_flight` | `dies_marking_in_flight`, stays `PENDING` | Existing Phase 2.3 tests apply |
| `mark_committed` | `dies_marking_committed`, then `AMBIGUOUS` | `test_a_failed_transition_on_a_correlated_record_changes_nothing` |
| `mark_not_committed` | `dies_marking_not_committed`, then `AMBIGUOUS` | Existing Phase 2.3 tests apply |
| `mark_ambiguous` | `dies_marking_ambiguous`, then `AMBIGUOUS` | Existing Phase 2.3 tests apply |
| `authorize_retry` | `dies_authorizing_retry`, stays `NOT_COMMITTED` | Existing Phase 2.3 tests apply |
| `resolve_ambiguous` | `dies_resolving`, stays `AMBIGUOUS` | Existing Phase 2.3 tests apply |
| Migration | `migrate_dying`, stays version 1 | SQLite and non-SQLite failure |
| Arbitrary moments | 8 kills at random times across a full two-epoch loop | |

After each kill, a fresh process runs both recoveries, and checks that the stores are sound, that **no record is `IN_FLIGHT` after recovery,** and that no second intent exists.

**Lock contention, full disk, kernel write refusal, foreign and damaged files:** the Phase 2.3 fault tests still pass, on the changed code.

## Concurrency

| Case | Threads | Two instances | Real processes |
|---|---|---|---|
| Next-epoch creation | 12 | 2 | 4 |
| Outcome fencing | 12 | 2 | 5 |
| Migration | | | 4, plus a deterministic version |
| Preflight | 8, 640 calls | | |

## Network attempts in the full run

[M] The test guard blocks and reports network attempts. The new tests add no network attempt: their names don't appear in the guard's report in `evidence/test_full_suite_output.txt`. The older unexpected attempts remain, untraced, as before this phase.

## Limitations

- **Power loss isn't tested.** SIGKILL keeps the page cache. Durability against power loss rests on SQLite's FULL setting.
- **A race test shows contention in that run,** and can't force every interleaving.
- **Crash tests use the real ledger, but test code steps through the calls.** There's no coordinator to test.
- **The windows use fixed kill points.** Only the arbitrary-moment test lands elsewhere.
- **Windows beyond those of D4,** such as Phase 2.5's windows 1 to 3, belong to coordinator tests.
