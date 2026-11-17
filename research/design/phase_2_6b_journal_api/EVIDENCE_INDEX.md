# Evidence index

Phase 2.6B. Paths under `evidence/` are in this folder. Backend paths are relative to `isfcr/CYUKTI/backend/`. File hashes are in `MANIFEST.sha256`.

## Scripts and outputs

| Item | Purpose |
|---|---|
| `evidence/measure_performance.py`, `evidence/performance_output.txt` | Performance on disposable journals |
| `evidence/mutation_check.py`, `evidence/mutation_output.txt` | 82 mutants on overlay copies |
| `evidence/check_package.py`, `evidence/check_output.txt` | Citations and figures in these documents |
| `evidence/test_focused_output.txt`, `evidence/test_focused_by_file.txt` | New tests |
| `evidence/test_existing_journal_output.txt` | Phase 2.3 tests and child lifecycle |
| `evidence/test_ledger_output.txt` | Ledger compatibility |
| `evidence/test_full_suite_output.txt` | Full backend suite |
| `evidence/test_repeat_stability.txt` | Repeats of the process-based tests |
| `evidence/single_mutant_recheck.txt` | The deterministic migration test alone against "no recheck under the lock" |
| `evidence/source_hashes_before.txt`, `evidence/source_hashes_after.txt` | Journal, ledger, helper and test hashes around this phase |
| `evidence/preservation_audit.txt` | Git, production files, protected files and archives |

## Sources

| Claim | Source |
|---|---|
| Contract | `research/design/phase_2_6a_decisions/`, `research/design/phase_2_5_coordination/`, each unchanged and verified against its manifest |
| Implementation | `neo4j_write_journal.py`, cited inline in each document |
| Tests | `tests/test_neo4j_write_journal_correlation.py`, `tests/test_neo4j_write_journal_migration.py`, `tests/test_neo4j_write_journal_lifecycle.py`, `tests/_neo4j_journal_lifecycle_child.py` |
| Previous implementation | `tests/_neo4j_journal_v1_frozen.py`, sha256 prefix `c69834f8475dc742` |

## What wasn't run

No Neo4j, Wazuh, listener, MISP, Shuffle, B3 experiment or replay. The ledger was exercised only by its tests and these tests, on temporary files.
