# Evidence index

Phase 2.6A. Paths under `evidence/` are in this folder. Backend paths are relative to `isfcr/CYUKTI/backend/`. File hashes are in `MANIFEST.sha256`, and source hashes in `evidence/source_hashes.txt`.

## Scripts and their outputs

All of them are read-only, offline and started by hand. None opens a database or the network, or imports the listener. They print counts and field names, never alert content.

| Script | Output | Establishes |
|---|---|---|
| `evidence/d1_wazuh_id_analysis.py` | `evidence/d1_wazuh_id_analysis_output.txt` | `id` shape and order, `manager` and `cluster` presence, hash merging |
| `evidence/d2_replay_accept_alert.py` | `evidence/d2_replay_output.txt` | Refusals by the real functions, what differs, and modelled effect |
| `evidence/d3_d13_payload_analysis.py` | `evidence/d3_d13_payload_analysis_output.txt` | W07U payloads through the real serializer, fingerprint and key check |
| `evidence/d6_d13_static_scan.py` | `evidence/d6_d13_static_scan_output.txt` | Label-free queries, and who reads dedup metadata |
| `evidence/check_package.py` | `evidence/check_output.txt` | Citations and numbers in these documents |

## Inputs

| Input | sha256 | Note |
|---|---|---|
| `isfcr/CYUKTI/data/raw_logs/alerts.json` | `e958c1cc9211036b2e9feaf48990c36cf370d724c1c139183229cb59565b3098` | 10,002,071 bytes, 9,015 alerts. Gitignored by `.gitignore:10`. **Provenance undocumented** |
| `isfcr/CYUKTI/evaluation/ground_truth/raw/mitre_mapping_v0_alerts.json` | `78ad16bf073cfcd3350d1d637aedd94983b6a8f6b288cc210877f5e091f2775c` | One wrapper object, no alert-level `id` |
| `isfcr/CYUKTI/experiments/records/EXP-2026-09-25-nmap-scan-01.json` | `71661a5380acb22070cdef2728182cdfa12fec235f67590c60e4f0f9080b612e` | An experiment record, not an alert |
| `research/experiments/b3_runs_v2/smoke/synthetic_alerts.jsonl` | `7d38b3685ea9ad71a0dded040a9f1a5ca2708f0d6129e89d543d2b99665afbb0` | 8 synthetic alerts, no `id` or `manager` |

**The repository holds only one real alert sample.** Every measured statement about Wazuh rests on it.

## External sources

Read for this phase through a fetch agent, at tag `v4.9.0`:

- `https://raw.githubusercontent.com/wazuh/wazuh/v4.9.0/src/analysisd/format/to_json.c`: `id` and `manager.name` construction.
- `https://raw.githubusercontent.com/wazuh/wazuh/v4.9.0/src/analysisd/analysisd.c`: `__crt_ftell` assignments.

**[L] Limits:** one tag, read rather than run, and the deployed version is unknown. A search found no official documentation of `id` semantics. Its contents weren't copied in or hashed here.

## Prior evidence, referenced and unchanged

| Item | Path | Used for |
|---|---|---|
| Phase 2.5 package | `research/design/phase_2_5_coordination/`, `MANIFEST.sha256` sha256 `fc8f42370c8ccb7738486c3fd317f482d8b9857281a988dcc8b8e54e5a7afad9` | Decisions, writers, AC list, windows |
| `ConstraintValidationFailed` before commit gives `NOT_COMMITTED`, attempts 1 (#6). Late COMMIT rejected, server 0, in 7 of 7 (B7, B8). A killed connection's transaction gone in about 1 s (B9) | `~/cyukti_diag/helper_revalidation_v1/` | D6 |
| Transaction experiment | `~/cyukti_diag/txn_experiment_v1/TRANSACTION_EXPERIMENT_REPORT.md` | The reproduced double count |
| Journal and ledger validation | `~/cyukti_diag/journal_phase23_v1/`, `~/cyukti_diag/ledger_phase24_v1/` | Store behaviour |

## Claims and their sources

| Claim | Source | Kind |
|---|---|---|
| Refusal of 70 alerts | `evidence/d2_replay_output.txt` | Measured |
| Refusal mechanism | `listener/wazuh_listener.py:accept_alert` L125-148, `listener/utils.py:generate_alert_id` L3-27 | Read |
| Offset moves past a refused alert | `listener/wazuh_listener.py:AlertFileHandler.read_new_lines` L228-306 | Read |
| 9,015 distinct IDs, offset order | `evidence/d1_wazuh_id_analysis_output.txt` | Measured |
| `id` construction | External source above | Read externally |
| W07U determinism, key lengths, burst | `evidence/d3_d13_payload_analysis_output.txt` | Measured, `event_id` fixed |
| Modelled routing counts | `evidence/d2_replay_output.txt` | **Model,** without Neo4j |
| W07U stops before the Campaign | `neo4j_client.py:update_duplicate_event` L712-748, `neo4j_client.py:create_unattributed_attack_event` L588-709, `evidence/check_output.txt` | Read and scanned. **Not run** |
| Current journal and ledger behaviour | Cited inline in each document | Read |
| Marker behaviour | None measured for this label | **Proposed,** apart from the cited constraint error |

## What wasn't run

No test, service, database, listener, replay harness, B3 experiment, or MISP or Shuffle call. `/var/ossec`, `offset.dat`, listener logs, MISP caches, `playbook_memory.db` and `neo4j-soc` weren't touched.
