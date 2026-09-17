# CYUKTI — Results, Metrics, and Dataset Evidence

All values below were extracted from live queries, fresh test runs, or repository files on **2026-08-31**. Every entry states its source explicitly. Entries with no repository evidence are marked `NOT MEASURED / NOT AVAILABLE`.

## Test suite

| Metric | Value | Source |
|---|---|---|
| Total tests | 150 | `pytest -q` fresh run, backend/, 2026-08-31 |
| Passed | 150 | same |
| Failed | 0 | same |
| Skipped | 0 | same |
| Test files | 21 | `tests/` directory listing |
| Runtime | ~23.5s | same run |

Largest test files by count: `test_investigation.py` (25), `test_evidence.py` (13), `test_mitre_resolver.py` (12), `test_gnn.py` (11), `test_next_technique_pipeline.py` (9).

## Neo4j graph statistics (live, 2026-08-31)

| Metric | Value | Source |
|---|---|---|
| Campaign nodes | 65 | `MATCH (c:Campaign) RETURN count(c)` |
| AttackEvent nodes | 124 | `MATCH (e:AttackEvent) RETURN count(e)` |
| Orphaned AttackEvents | 0 (124/124 linked) | `MATCH (:Campaign)-[:HAS_EVENT]->(e) RETURN count(DISTINCT e)` |
| Technique nodes | 858 | `MATCH (t:Technique) RETURN count(t)` |
| NEXT_TECHNIQUE edges | 3 | `MATCH ()-[r:NEXT_TECHNIQUE]->() RETURN count(r)` |
| Attacker nodes | 5 | live query |
| Host nodes | 4 | live query |
| Operation nodes | 43 | live query |
| AttackEvents with `mitre_status="UNKNOWN"` | 5 | live query (Phase 20 live-created events only) |
| AttackEvents with `mitre_status="RESOLVED"` | 0 | live query — no native-mapped alert has been processed by the Phase 20 code path yet; all pre-Phase-20 events lack this property entirely |

**Interpretation**: the 65/124/858 figures represent the *entire accumulated history* of the live lab (real campaigns from Phases 16-19, plus 5 new Phase-20-tagged UNKNOWN events). This is not a fixed "dataset" — it is live, growing graph state. The frozen ML dataset (below) is a separate, deliberately-snapshotted 60-row CSV that does **not** include the 5 new UNKNOWN events or the 2 campaigns/4 events added since the Phase 17 snapshot.

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

**This dataset has not changed since Phase 17/18** — confirmed by matching row count, severity distribution, and `prediction_correct` distribution exactly against the values recorded during Phase 18. This is expected: rebuilding it was explicitly out of scope for every phase after Phase 18.

## ML model evaluation

**IMPORTANT CAVEAT, stated once and applying to every number in this subsection**: the only dataset available to `evaluate_model.py` is the same 60-row CSV the model was **trained on**. No held-out test/validation split exists anywhere in the repository. The numbers below are **in-sample (training-set) metrics**, not a generalization estimate. Given Phase 17's own finding that this dataset is `NOT_READY_FOR_CALIBRATION` (severe class imbalance, attacker/severity confounding, 11 duplicate rows), these numbers should **not** be presented as evidence of real-world model performance.

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

### GNN, SSL autoencoder, SSFT

| Metric | Value | Source |
|---|---|---|
| GNN real-campaign benchmark | NOT MEASURED / NOT AVAILABLE | No real-campaign evaluation script/output found; `test_gnn.py`'s 11 tests use synthetic graphs only |
| SSL autoencoder training windows | 21,176 (reported in prior-session record; not re-extracted from a live artifact this session — `window_features.npy` is 52,855,424 bytes, consistent with a large real window set but exact row count not re-verified today) | `ml/artifacts/window_features.npy`, dated March 2026 |
| SSL autoencoder convergence | Confirmed by `test_train_ssl_autoencoder_produces_artifacts_and_converges` (pass) | `test_ssl_pipeline.py` |
| SSL anomaly-detection accuracy/AUC on real held-out attack traffic | NOT MEASURED / NOT AVAILABLE | No such evaluation artifact found |

## MITRE resolution coverage (Phase 20) — see `05_phase20_results.md` for full detail and a documented discrepancy against previously-reported figures.

## Attribution / correlation

| Metric | Value | Source |
|---|---|---|
| Threat attribution accuracy (real ground truth) | NOT MEASURED / NOT AVAILABLE | No ground-truth attacker-identity dataset exists in the repo |
| Operation/correlation matching accuracy | NOT MEASURED / NOT AVAILABLE | 43 Operation nodes exist (existence confirmed) but no labeled "correct grouping" evaluation exists |

## RAG

| Metric | Value | Source |
|---|---|---|
| Indexed real ATT&CK documents | >500 (test asserts `len(internal._documents) > 500`; code comment states 858 non-deprecated techniques) | `test_mitre_retriever_indexes_hundreds_of_real_techniques`, `tests/test_rag.py:64-67` |
| Retrieval precision/recall on a labeled query set | NOT MEASURED / NOT AVAILABLE | No labeled retrieval benchmark exists |

## Latency / throughput / infrastructure

| Metric | Value | Source |
|---|---|---|
| Per-alert processing latency | NOT MEASURED / NOT AVAILABLE | No timing instrumentation found in `process_alert()` |
| Listener throughput | NOT MEASURED / NOT AVAILABLE | Not instrumented; only qualitative "offset caught up, no backlog" observed live |
| API response latency | NOT MEASURED / NOT AVAILABLE | Not instrumented |
| Memory usage | NOT MEASURED / NOT AVAILABLE | Not instrumented |

## Investigation confidence architecture — real XGBoost probabilities (historical claim, not re-verified this session)

The Phase 20 task brief for this session stated that real XGBoost probabilities were previously recorded for three named campaigns (CAMP_427A075C, CAMP_1429ADB4, CAMP_D8605E81) as part of validating the confidence architecture. **This session did not independently re-run or re-verify those specific numbers** — they are carried forward from prior-session project history, not fresh evidence gathered today. If citing them in a review, label them explicitly as "previously recorded, from project history" rather than as evidence generated during this audit.
