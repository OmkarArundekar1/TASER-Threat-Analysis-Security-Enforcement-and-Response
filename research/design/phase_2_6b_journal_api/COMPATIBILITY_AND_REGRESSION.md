# Compatibility and regression

Backend paths are relative to `isfcr/CYUKTI/backend/`. **[V]** verified, **[M]** measured.

## Legacy uncorrelated API

| Compatibility point | Result |
|---|---|
| Existing call forms | [V] All new arguments are keyword-only with default `None`. Existing callers change nothing |
| State table | [V] Unchanged, and the existing test pinning it passes |
| `PENDING` to `NOT_COMMITTED` | [V] Still allowed for uncorrelated records |
| Checksums | [V] Same formula as version 1, compared against the frozen previous code |
| History | [V] New columns stay empty on uncorrelated records |
| Existing files | [V] Migrate on first writable open |
| Errors | [V] Unchanged for uncorrelated use |

## Changes to existing tests

**Exactly two pins, in `tests/test_neo4j_write_journal.py`.** Both were predicted by Phase 2.6A, and neither weakens a check:

| Line | Before | After | Why |
|---|---|---|---|
| L75 | `SCHEMA_VERSION == 1` | `SCHEMA_VERSION == 2` | The version did change |
| L799-801 | Public method set | Adds `preflight_intent`, `find_by_correlation_key` | Those methods were added. The test still requires an exact set |

No other existing test was changed. `tests/test_neo4j_write_journal_faults.py`, which 2.6A listed as likely to change, needed no edit, because version 1's objects and hashes were kept compatible.

## Regression

[M]

| Suite | Before this phase | After |
|---|---|---|
| Full backend | 2,111 passed, 29 skipped, 0 failed | **2,386 passed, 29 skipped, 0 failed** |
| New tests | | 275 |
| Journal and child lifecycle | | 314 passed |
| Ledger | | 346 passed |

## Ledger

[V] `neo4j_idempotency_ledger.py` is byte-identical: see `evidence/source_hashes_before.txt` and `evidence/source_hashes_after.txt`. No incompatibility arose. Its key rule, attempts and resolution values fit the journal's contract, and both stores work together in the lifecycle tests.

## Production code

[V] `tests/test_neo4j_write_journal.py` still proves that no production module imports the journal, and that the helper doesn't mention it. So:

| Component | Changed? |
|---|---|
| Listener | No |
| Any Neo4j writer, `neo4j_client.py` | No |
| Coordinator | Doesn't exist |
| `AppliedWrite`, Neo4j constraints | Don't exist |
| Deterministic ids | No |

**Every backend production file other than `neo4j_write_journal.py` has an unchanged hash.** See `evidence/preservation_audit.txt`.

## Working tree and protected evidence

Recorded in `evidence/preservation_audit.txt`: HEAD, status before and after, protected files, the B3 freeze manifest, five evidence archives, and the Phase 2.5 and 2.6A packages.

No commit, push, reset, clean, stash, checkout or revert was run.
