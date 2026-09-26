# CYUKTI — Results, Metrics, and Dataset Evidence

> Last verified: 2026-09-26. **New independent evaluation**: MITRE mapping accuracy, threat qualification accuracy, campaign correlation, and attribution accuracy are no longer purely `NOT MEASURED / NOT AVAILABLE` — see `review/evaluation_results.md` and `review/paper_metrics_source_of_truth.md` for real (preliminary, AI-assisted-not-human-reviewed) numbers from the new `evaluation/` framework. The original 2026-08-31 extraction date below is preserved for the sections that describe the frozen ML dataset and its Phase 17/18 analyses (those numbers are unchanged by design); the test-suite and live-Neo4j sections have been refreshed to 2026-09-25.

All values below were extracted from live queries, fresh test runs, or repository files on **2026-08-31**, except where marked as re-verified 2026-09-25. Every entry states its source explicitly. Entries with no repository evidence are marked `NOT MEASURED / NOT AVAILABLE`.

## Test suite

| Metric | Value | Source |
|---|---|---|
| Total tests (backend) | 689 | `pytest -q` fresh run, backend/, 2026-09-25 (was 150 on 2026-08-31) |
| Passed | 689 | same |
| Failed | 0 | same |
| Skipped | 0 | same |
| Test files (backend) | 65 | `tests/` directory listing, 2026-09-25 (was 21) |
| Total tests (frontend) | 109 | frontend test run, 2026-09-25 (new since the 2026-08-31 snapshot — no frontend suite existed then) |
| Test files (frontend) | 17 | frontend `tests/`/`__tests__` listing |
| **Combined total** | **798** | 689 backend + 109 frontend |
| SOAR/playbook tests (subset of backend total) | 83 | 9 files, `tests/test_soar_*.py` — new subsystem since 2026-08-31 |

Largest test files by count (2026-08-31 snapshot, not re-tallied at the current 689-test scale): `test_investigation.py` (25), `test_evidence.py` (13), `test_mitre_resolver.py` (12), `test_gnn.py` (11), `test_next_technique_pipeline.py` (9).

## Neo4j graph statistics (live, 2026-09-25; was 2026-08-31)

| Metric | Value | Source |
|---|---|---|
| Campaign nodes | 111 | `MATCH (c:Campaign) RETURN count(c)` (was 65 on 2026-08-31) |
| AttackEvent nodes | 212 | `MATCH (e:AttackEvent) RETURN count(e)` (was 124) |
| Orphaned AttackEvents | 0 (124/124 linked) as of 2026-09-12 | `MATCH (:Campaign)-[:HAS_EVENT]->(e) RETURN count(DISTINCT e)` — not re-run at the current 212-event scale. *Planned:* re-run this check at the current 212-event graph scale in the next verification session. |
| Technique nodes | 858 | `MATCH (t:Technique) RETURN count(t)` — unchanged, vendored corpus |
| NEXT_TECHNIQUE edges | 3 | `MATCH ()-[r:NEXT_TECHNIQUE]->() RETURN count(r)` — unchanged |
| Attacker nodes | 9 | live query (was 5) |
| Host nodes | 12 | live query (was 4) |
| Operation nodes | 50 | live query (was 43) |
| **Total nodes (all labels)** | **2,539** | live query, 2026-09-25 |
| **Total relationships (all types)** | **20,804** | live query, 2026-09-25 |
| AttackEvents with `mitre_status="UNKNOWN"` | 5 (as of the 2026-08-31/09-12 check) | live query (Phase 20 live-created events only) — current total unattributed count is higher at the new 212-event scale, not individually re-broken-out this pass. *Planned:* re-run this breakdown at the current 212-event scale in the next verification session. |
| AttackEvents with `mitre_status="RESOLVED"` | 0 (as of the same check) | live query — no native-mapped alert had been processed by the Phase 20 code path yet at that time; all pre-Phase-20 events lack this property entirely |

**Interpretation**: the 111/212/858 figures (2026-09-25) represent the *entire accumulated history* of the live lab, grown further since the 65/124/858 snapshot on 2026-08-31 (real campaigns from Phases 16-19, the 5 Phase-20-tagged UNKNOWN events, plus continued live ingestion since). This is not a fixed "dataset" — it is live, growing graph state. The frozen ML dataset (below) is a separate, deliberately-snapshotted 60-row CSV that does **not** track this growth at all — it was frozen at the Phase 17/18 snapshot and has not been rebuilt since.

## Frozen ML dataset (`ml/datasets/campaign_dataset.csv`)

| Metric | Value | Source |
|---|---|---|
| Rows (campaigns) | 60 | `pd.read_csv(...)`, 2026-08-31 |
| Columns | 66 | same (57 declared features + identifiers/labels) |
| Severity distribution | Low=53, Medium=3, Critical=4, **High=0** | same |
| `next_technique` populated | 20/60 | same |
| `prediction_correct` distribution | NaN(N/A)=48, 0=8, 1=4 | same |
| Evaluable next-technique predictions | 12 (4 correct, 8 incorrect) | derived: 20 non-null minus 8 single/no-prediction NA cases = 12 |
| Exact feature-vector duplicate rows | 11 (in 3 groups) | Phase 17 analysis (`scripts/phase17_dataset_validity_gate.py`) |
| Zero-variance features | 23 of 57 | Phase 17 analysis |
| Attacker identities | 3 | Phase 17 analysis |
| Victim identities | 2 | Phase 17 analysis |
| Attacker/victim pairs | 4 | Phase 17 analysis |
| Technique-set compositions | 13 (8 recurring archetypes) | Phase 17 analysis |
| Dataset validity gate verdict | `NOT_READY_FOR_CALIBRATION` | `scripts/phase17_dataset_validity_gate.py` output, Phase 17 |

*Planned:* the Phase 19 dataset-expansion spec directly targets every one of these gaps — more attackers/victims, deduplicated feature vectors, deconfounded severity labels, spread-out collection dates, and real High-severity examples.

**This dataset has not changed since Phase 17/18** — confirmed by matching row count, severity distribution, and `prediction_correct` distribution exactly against the values recorded during Phase 18. This is expected: rebuilding it was explicitly out of scope for every phase after Phase 18.

## ML model evaluation

**IMPORTANT CAVEAT, stated once and applying to every number in this subsection**: the only dataset available to `evaluate_model.py` is the same 60-row CSV the model was **trained on**. No held-out test/validation split exists anywhere in the repository. The numbers below are **in-sample (training-set) metrics**, not a generalization estimate. Given Phase 17's own finding that this dataset is `NOT_READY_FOR_CALIBRATION` (severe class imbalance, attacker/severity confounding, 11 duplicate rows), these numbers should **not** be presented as evidence of real-world model performance.
*Planned:* the Phase 19 dataset expansion is designed to reach the scale needed to support a proper held-out train/test evaluation.

Evaluation run 2026-08-31 (`ml.evaluate_model.evaluate()`, model=`ml/models/xgb_severity.json`, target=`severity`, n=60):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Critical | 1.000 | 0.250 | 0.400 | 4 |
| Low | 0.962 | 0.962 | 0.962 | 53 |
| Medium | 0.500 | 1.000 | 0.667 | 3 |
| **Accuracy** | — | — | **0.917** | 60 |
| Macro avg | 0.821 | 0.737 | 0.676 | 60 |
| Weighted avg | 0.942 | 0.917 | 0.910 | 60 |

Confusion matrix (rows=true, cols=predicted, order Critical/Low/Medium):
```
[[1, 2, 1],
 [0, 51, 2],
 [0, 0, 3]]
```
Mean model confidence: 0.7639.

**Correct interpretation**: the model fits the 60 training rows well overall (weighted F1=0.91) but performs poorly on the rare Critical class in-sample (recall=0.25, i.e., it misclassified 3 of 4 Critical campaigns even on data it was trained on) — a direct symptom of the severe class imbalance (4 Critical vs 53 Low) documented in Phase 17. No High-severity examples exist to evaluate at all. This is `IMPLEMENTED — NOT YET BENCHMARKED` in the proper sense (a real held-out benchmark).
*Planned:* the Phase 19 dataset expansion explicitly targets adding real High/Critical-severity examples, which the current dataset lacks entirely.

### GNN, SSL autoencoder, SSFT

| Metric | Value | Source |
|---|---|---|
| GNN real-campaign benchmark | NOT MEASURED / NOT AVAILABLE | No real-campaign evaluation script/output found; `test_gnn.py`'s 11 tests use synthetic graphs only. *Planned:* build a real-campaign benchmark once the Phase 19 dataset expansion provides enough real campaign graphs to make one meaningful. |
| SSL autoencoder training windows | 21,176 (reported in prior-session record; not re-extracted from a live artifact this session — `window_features.npy` is 52,855,424 bytes, consistent with a large real window set but exact row count not re-verified today) | `ml/artifacts/window_features.npy`, dated March 2026 |
| SSL autoencoder convergence | Confirmed by `test_train_ssl_autoencoder_produces_artifacts_and_converges` (pass) | `test_ssl_pipeline.py` |
| SSL anomaly-detection accuracy/AUC on real held-out attack traffic | NOT MEASURED / NOT AVAILABLE | No such evaluation artifact found. *Planned:* build this benchmark once labeled anomalous/benign real traffic is available. |

## MITRE resolution coverage (Phase 20) — see `05_phase20_results.md` for full detail and a documented discrepancy against previously-reported figures.

## Attribution / correlation

| Metric | Value | Source |
|---|---|---|
| Threat attribution accuracy (real ground truth) | NOT MEASURED / NOT AVAILABLE | No ground-truth attacker-identity dataset exists in the repo. *Planned:* build a ground-truth attacker-identity benchmark dataset so a real attribution-accuracy metric can be computed. |
| Operation/correlation matching accuracy | NOT MEASURED / NOT AVAILABLE | 50 Operation nodes exist as of 2026-09-25 (existence confirmed, was 43 on 2026-08-31) but no labeled "correct grouping" evaluation exists. *Planned:* evaluate matching quality once a labeled set of correctly/incorrectly grouped campaigns exists. |

## RAG

| Metric | Value | Source |
|---|---|---|
| Indexed real ATT&CK documents | >500 (test asserts `len(internal._documents) > 500`; code comment states 858 non-deprecated techniques) | `test_mitre_retriever_indexes_hundreds_of_real_techniques`, `tests/test_rag.py:64-67` |
| Retrieval precision/recall on a labeled query set | NOT MEASURED / NOT AVAILABLE | No labeled retrieval benchmark exists. *Planned:* construct a labeled query set to measure retrieval precision/recall directly. |

## Latency / throughput / infrastructure

**Update, 2026-09-25**: a real benchmark harness (`backend/benchmarks/run_benchmarks.py`) now exists and has been run — per-stage latency is no longer `NOT MEASURED`. See `BENCHMARKS.md` and `journal_ready_data.md` Section 23 for the full table; summary below.

| Metric | Value | Source |
|---|---|---|
| MITRE resolution (p50/p95/p99) | 0.001 / 0.001 / 0.002 ms | `run_benchmarks.py`, committed run `benchmark_20260925T075030Z.json` |
| Deduplication check (p50/p95/p99) | 0.001 / 0.002 / 0.003 ms | same |
| Neo4j simple query (p50/p95/p99) | 0.904 / 1.251 / 1.285 ms | same |
| Campaign selection candidate discovery (p50/p95/p99) | 21.035 / 24.285 / 26.480 ms | same |
| Severity prediction / XGBoost (p50/p95/p99) | 21.975 / 39.158 / 117.582 ms | same |
| GNN embedding (p50/p95/p99) | 2.818 / 3.981 / 4.070 ms | same |
| **End-to-end** `GET /api/incidents/<id>/overview` (p50/p95/p99) | **35.538 / 66.869 / 100.484 ms** | same — real HTTP round trip |
| Throughput (end-to-end) | ~22.8 ops/sec | same |
| Listener throughput (sustained ingestion rate) | NOT MEASURED / NOT AVAILABLE | Only qualitative "offset caught up, no backlog" observed live; not separately benchmarked |
| Memory usage | NOT MEASURED / NOT AVAILABLE | Not instrumented |

Caveat carried from `BENCHMARKS.md`: single-process, single-machine, low-concurrency numbers on a development machine — not a load-tested production SLA.

## Investigation confidence architecture — real XGBoost probabilities (historical claim, not re-verified this session)

The Phase 20 task brief for this session stated that real XGBoost probabilities were previously recorded for three named campaigns (CAMP_427A075C, CAMP_1429ADB4, CAMP_D8605E81) as part of validating the confidence architecture. **This session did not independently re-run or re-verify those specific numbers** — they are carried forward from prior-session project history, not fresh evidence gathered today. If citing them in a review, label them explicitly as "previously recorded, from project history" rather than as evidence generated during this audit.
