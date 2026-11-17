# Evidence index

Where each claim in these documents comes from. Backend paths are relative to `isfcr/CYUKTI/backend/`.

## Repository state

| Item | Value |
|---|---|
| HEAD | `3afe62c4216574a906feaffe87e35ddb084db941`, branch `main` |
| Working tree | 182 status entries at the start, all preserved. This phase only adds this folder |

## Source files read

| Area | Files |
|---|---|
| Write helper, writers | `neo4j_client.py` |
| Journal | `neo4j_write_journal.py` |
| Ledger | `neo4j_idempotency_ledger.py` |
| Alert flow | `realtime_socgraph.py`, `listener/wazuh_listener.py`, `listener/utils.py`, `dedup_engine.py` |
| Campaign and operation flow | `campaign_manager.py`, `campaign_correlation_engine.py`, `operation_manager.py`, `operation_repository.py`, `chain_updater.py`, `prediction_engine.py` |
| Maintenance and setup | `campaign_reconstruction.py`, `risk_recalculation.py`, `attack_chain_builder.py`, `mitre_import/` (by scan) |
| Reads confirmed by scan | `dashboard_api.py`, `soar/api.py`, `recommendation_engine.py`, `graph_feature_engine.py`, `attribution_context.py`, `replay/harness.py`, `benchmarks/run_benchmarks.py`, `_verify_neo4j.py`, `ml/gnn/` |
| Configuration | `config.py` |
| Tests, for behaviour | `tests/test_neo4j_write_journal*.py`, `tests/test_neo4j_idempotency_ledger*.py`, `tests/_neo4j_journal_child.py`, `tests/_neo4j_ledger_child.py`, `tests/test_neo4j_write_helper.py` |

## Generated evidence, in `evidence/`

All of it came from read-only scripts, and none needed a service.

| File | What it shows |
|---|---|
| `scan_cypher_and_driver_calls.py`, `cypher_scan.txt` | Every write-looking Cypher constant and driver call in the backend, by AST |
| `trace_callers.py`, `callers.txt` | Modules importing the driver, unusual or dynamic `.run` calls, callers of each writer, importers of candidates |
| `line_map.py`, `lines.json` | Line range of every function cited |
| `measure_alert_ids.py`, `alert_id_measurement.txt` | The `alert_id` collision figures. Counts only, no alert content |
| `check_documents.py` | The checks below |

## Document checks

`check_documents.py` confirms, against the source:

- Every citation in the form `` `file:Symbol` L a-b `` matches that symbol's real line range.
- Every line-only citation's text is what the document says it is, for the transition tables and `config.py`.
- All 20 writers appear in the inventory with all the required fields, and the tier and identity tables add up.
- Every function holding a Cypher write in the raw scan is accounted for as a writer or an exclusion, so none can be quietly missing.
- All 21 required scenarios are in the matrix.
- Every state named in the state diagram exists in the ledger, and every ledger transition is in the diagram, with nothing extra.
- Every required document exists, and carries the labels.
- Any mention of exactly-once execution in the documents is a denial.

## Earlier evidence relied on

Nothing here was changed.

| Claim | Evidence |
|---|---|
| Discarded results can lose a write; commit reply lost leaves data with no confirmation; a retried committed batch double counts | `~/cyukti_diag/txn_experiment_v1/` |
| Helper outcomes on a real server; timeout and acquisition behaviour; a killed connection's transaction gone within about 1 s | `~/cyukti_diag/helper_revalidation_v1/` |
| Journal behaviour under crash and fault | `~/cyukti_diag/journal_phase23_v1/` |
| Ledger behaviour under crash, fault and contention | `~/cyukti_diag/ledger_phase24_v1/` |
| Child lifecycle, and the 2,111-passing suite | `~/cyukti_diag/ledger_phase24_v1/closeout_20261105/refinement/` |

## Evidence quality

| Level | Applies to |
|---|---|
| **Direct, from source** | Writer inventory, API semantics, transition tables, callers |
| **Measured** | `alert_id` collisions, on one 9,015-alert repository sample |
| **Tested earlier** | Journal, ledger and helper behaviour |
| **From reading, not running** | W07 stopping before Campaign for unattributed events; the listener discarding collided alerts; W13's possible null failure |
| **Unverified assumption** | Wazuh `id` in production; SQLite's WAL `ATTACH` behaviour, from its documentation; how long a dead client's transaction stays open |
| **Not done** | Diagrams were checked by script for state names and transitions, but **they weren't rendered,** because no Mermaid renderer is installed. Nothing in this design has been run |
