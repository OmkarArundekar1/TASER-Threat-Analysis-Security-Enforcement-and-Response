# Evidence index

Phase 2.6C. Paths under `evidence/` are in this folder. Backend paths are relative to `isfcr/CYUKTI/backend/`. File hashes are in `MANIFEST.sha256`.

| Item | Purpose | Kind |
|---|---|---|
| `evidence/before_fix/utils.py`, `evidence/before_fix/wazuh_listener.py` | The listener before this phase, byte for byte | Source |
| `evidence/replay_listener.py` | Replay through the real listener functions | Script |
| `evidence/replay_before_fix.txt` | The defect reproduced | Measured offline |
| `evidence/replay_after_fix.txt` | The fix, same script | Measured offline |
| `evidence/measure_listener_performance.py` | Benchmarks | Script |
| `evidence/performance_before_fix.txt`, `evidence/performance_after_fix.txt` | Results | Measured offline |
| `evidence/mutation_check.py`, `evidence/mutation_output.txt` | 54 mutants on overlay copies | Measured offline |
| `evidence/test_new_output.txt`, `evidence/test_repeat_stability.txt` | New tests | Reproduced in tests |
| `evidence/test_existing_listener_output.txt` | Existing listener-related tests | Reproduced in tests |
| `evidence/test_journal_ledger_output.txt` | Journal and ledger, unaffected | Reproduced in tests |
| `evidence/test_full_suite_output.txt` | Full backend suite | Reproduced in tests |
| `evidence/check_package.py`, `evidence/check_output.txt` | Citations and figures in these documents | Script |
| `evidence/source_hashes_before.txt`, `evidence/source_hashes_after.txt` | Modified source and test hashes | Recorded |
| `evidence/preservation_audit.txt` | Git, production files, protected files, archives | Recorded |

## Input

`isfcr/CYUKTI/data/raw_logs/alerts.json`, 9,015 alerts, sha256 `e958c1cc9211036b2e9feaf48990c36cf370d724c1c139183229cb59565b3098`. Gitignored, with undocumented provenance. It's the only real alert sample in the repository.

## Earlier phases, referenced and unchanged

`research/design/phase_2_5_coordination/`, `research/design/phase_2_6a_decisions/`, `research/design/phase_2_6b_journal_api/`. Each verifies against its manifest.

## What wasn't run

No live listener, `/var/ossec` file, Wazuh service, Neo4j, MISP, Shuffle, B3 experiment, or production alert. The listener's log and `offset.dat` weren't opened.
