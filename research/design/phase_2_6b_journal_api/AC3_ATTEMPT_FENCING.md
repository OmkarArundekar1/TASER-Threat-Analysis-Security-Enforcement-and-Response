# AC-3: attempt fencing and epoch-chain enforcement

Implemented, mainly in `neo4j_write_journal.py:NeoWriteJournal._transition` L1061-1125, the only code that changes an existing record. Backend paths are relative to `isfcr/CYUKTI/backend/`. **[V]** verified by tests.

## Rules

| # | Rule | Result if broken |
|---|---|---|
| 1 | A correlated record's start must give `ledger_attempt` | `TypeError` |
| 2 | That attempt must exceed every attempt seen for the key, **on any epoch** | `StaleAttemptError` |
| 3 | A correlated record's outcome must give `expected_ledger_attempt` | `TypeError` |
| 4 | That attempt must equal the record's current one | `StaleAttemptError` |
| 5 | No outcome before any attempt started | `InvalidTransitionError` |
| 6 | An uncorrelated record takes neither argument | `TypeError` |
| 7 | The state table still applies | `InvalidTransitionError` |

Every refusal happens inside `BEGIN IMMEDIATE`, and changes neither row nor history. Tests compare all rows and the file bytes.

## How each brief requirement is met

| Requirement | Implementation | Test |
|---|---|---|
| Correlated records always carry attempts | Rules 1, 3 | `test_starting_a_correlated_record_needs_a_valid_attempt`, `test_a_wrong_attempt_changes_nothing` |
| Attempts only increase | Rule 2, and trigger `journal_records_attempt_rises` | `test_an_attempt_must_rise_across_epochs_too`, `test_a_retry_of_a_correlated_record_needs_a_higher_attempt` |
| Journal enforces the epoch chain | `neo4j_write_journal.py:NeoWriteJournal._admit_epoch` L1025-1044 | See `D4_RETRY_LIFECYCLE.md` |
| New epoch carries identical intent | Same | `test_a_new_epoch_cannot_change_the_intent_in_any_way` |
| Old outcome can't alter the current attempt | Rule 4 | `test_an_old_attempt_cannot_touch_the_retry_that_replaced_it` |
| Outcome names the intent and attempt | `record_id` plus `expected_ledger_attempt` | As above |
| Stale, duplicate, mismatched, out-of-order all refused unchanged | Rules 4, 5, 7 | `test_a_repeated_outcome_changes_nothing`, `test_no_outcome_can_be_reported_before_an_attempt_started`, `test_a_stale_outcome_after_a_newer_intent_exists_changes_neither_record` |
| Uncorrelated semantics retained | Rule 6 | `test_an_uncorrelated_record_is_exactly_what_it_always_was`, plus all existing tests |
| Concurrency can't bypass it | Checked in the transaction | Threads, two instances, and real processes |
| Integrity detects inconsistent history | `neo4j_write_journal.py:NeoWriteJournal._correlation_is_sound` L1455-1483, `neo4j_write_journal.py:NeoWriteJournal._chain_problems` L1486-1515 | Forged-record tests |

## Order of checks in a transition

1. Load and verify the row.
2. Presence of the attempt argument: a mistake in the caller, reported first.
3. For outcomes: the fence, **before** the state check, as 2.6A's contract sets out.
4. The state table.
5. For starts: the rise check.
6. Update row and history together.

## Interaction with `RESOLVED`

| Situation | Result |
|---|---|
| Wrong attempt, `RESOLVED` record | `StaleAttemptError`. Unchanged |
| Matching attempt, `RESOLVED` record | `InvalidTransitionError`. Unchanged |
| Either, on an old epoch after a new one exists | The same. Neither record changes |
| `resolve_ambiguous` | Not fenced, per the contract. It's a person's act on a record that's already `AMBIGUOUS`, and it can only happen once |

## Recovery

`neo4j_write_journal.py:NeoWriteJournal.recover_unfinished` L1321-1387 passes the attempt it read to the fenced transition. If the record moved on between its read and its write, the conversion is refused, instead of marking a newer live attempt `AMBIGUOUS`. The recovery metadata records `ledger_attempt`, as well as the existing `attempt`.

The state of an uncorrelated record during recovery is handled exactly as before.

## Concurrent submissions

| Test | Contenders | Result |
|---|---|---|
| `test_threads_reporting_different_attempts_let_only_the_current_one_win` | 12 threads, attempts 1 and 5 | One attempt-5 winner. All attempt-1 reports stale |
| `test_two_instances_reporting_at_once_cannot_bypass_the_fence` | 2 instances, 2 different outcomes | One wins |
| `test_processes_reporting_different_attempts_cannot_bypass_the_fence` | 5 processes | One attempt-2 winner. All attempt-1 reports `StaleAttemptError` |

[V] **These are real races, and they can't force every interleaving.** What protects the fence is that the check and write share `BEGIN IMMEDIATE`.

## Honest scope

[V] Fencing protects the journal's own records. It can't tell that a claim is valid in the ledger: the journal believes the attempt it's given, as long as it rises. Comparing the two stores is the coordinator's job.
