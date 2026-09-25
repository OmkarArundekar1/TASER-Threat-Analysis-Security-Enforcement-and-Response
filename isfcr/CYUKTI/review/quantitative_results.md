# CYUKTI — Quantitative Results (Master Report)

> Last verified: 2026-09-25. Counts reflect live Neo4j query and pytest run from this date. Rows below retain their original per-entry source date where they describe a specific historical snapshot (e.g. the Phase 17/18 offline analyses, the frozen ML dataset, the 391/239/13-alert MITRE snapshots) — those are preserved as historical evidence, not overwritten. Only headline "current state" figures (total test count, live Neo4j graph counts, current MITRE snapshot) have been refreshed; see `review/paper_submission_status.md` for the single authoritative current-numbers list.

Generated 2026-09-12. Read-only analysis task — no source files, Neo4j data, Docker state, or telemetry were modified while producing this report (verified at the end, Task 7).

## IMPORTANT: state discrepancy disclosed up front

This task's brief described the "current known state" as: Neo4j (`neo4j-soc`) down, no listener running (the Phase 20G state). A read-only check performed at the start of this task found that is **no longer true**: in the immediately preceding turn (Phase 20H, this same session), Neo4j was recovered via `docker start neo4j-soc` and the listener was restarted (PID 8814). A fresh read-only check just now confirms **both are still up**: `neo4j-soc: Up 13 minutes`, bolt port 7687 accepting connections, listener PID 8814 still alive. This report uses the **actual current live state**, not the stale Phase 20G framing — the discrepancy is noted here rather than silently resolved either way. No infrastructure action was taken to produce this report; the check was read-only (`docker ps`, `nc -z`, `pgrep`).

## Evidence-strength legend (used throughout)

- **A** — Experimentally measured (empirical measurement from a dataset/controlled offline experiment)
- **B** — Live-system measured (observed from the running CYUKTI/Wazuh/Neo4j system)
- **C** — Test-verified (unit/integration/regression test evidence)
- **D** — Static/structural verification (code-level fact, e.g. "this function is never called on this path")
- **E** — Not measured (no defensible quantitative result exists)

---

## Master Quantitative Results Table

| Module / Experiment | Metric | Value | Dataset / N | Evaluation Method | Evidence Source | Status | Reviewer Interpretation |
|---|---|---:|---:|---|---|---|---|
| Wazuh ingestion/listener | Total tests (backend) | 689 (65 files) | — | pytest | `pytest -q`, re-verified 2026-09-25 (was 150 on 2026-09-12) | C | Full regression suite, all backend modules; frontend adds 109 more (17 files) — 798 total, see `paper_submission_status.md` |
| Wazuh ingestion/listener | Passing / Failing | 689 / 0 | 689 | pytest | same | C | 0 failures |
| Wazuh ingestion/listener | Compile check | clean | — | `python -m compileall` | fresh run 2026-09-12 | D | No syntax errors across backend + scripts |
| Wazuh ingestion/listener | Listener uptime this session | ~13 min at time of writing | — | process inspection | `docker ps`/`pgrep`, 2026-09-12 | B | Recovered and stable after infra outage |
| Wazuh ingestion/listener | Offset-recovery behavior | fired correctly | 1 restart event | live observation | listener log: `"Saved offset beyond EOF. Resetting to file end."` | B | Confirms documented safety behavior, not just inferred |
| MITRE resolution | Alerts evaluated (representative snapshot) | 391 | — | coverage script | `mitre_coverage_report.py`, 2026-08-31 | B | Largest, most representative live sample available |
| MITRE resolution | NATIVE_WAZUH | 39 (10.0%) | 391 | coverage script | same | B | Native Wazuh rule.mitre present |
| MITRE resolution | REVIEWED_RULE_MAPPING | 0 (0%) | 391 | coverage script | same | B | Registry empty by design |
| MITRE resolution | DETERMINISTIC_INFERENCE | 0 (0%) | 391 | coverage script | same | B | Zero live rules by design |
| MITRE resolution | AMBIGUOUS | 0 (0%) | 391 | coverage script | same | B | — |
| MITRE resolution | UNKNOWN | 352 (90.0%) | 391 | coverage script | same | B | Correct/expected outcome, not a deficiency |
| MITRE resolution | Older snapshot (superseded) | 239 alerts, 3 native (1.3%), 236 UNKNOWN (98.7%) | 239 | coverage script | historical, earlier in Phase 20 | B (superseded) | Explicitly distinguished from the 391-alert snapshot — traffic composition changed over time |
| MITRE resolution | Current-moment file snapshot | 13 alerts, 0 native, 13 UNKNOWN (100%) | 13 | coverage script | `mitre_coverage_report.py`, 2026-09-12 | B | Small, reboot-noise-dominated; NOT representative — reported for completeness only |
| MITRE resolution | Post-rule-fix, post-reboot snapshot (newest) | 120 alerts, 26 native (21.7%), 94 UNKNOWN (78.3%) | 120 | coverage script | `mitre_coverage_report.py`, 2026-09-25, after the Section 4.9 `local_rules.xml` fixes and a full environment reboot | B | Rules are confirmed loaded in the live manager; none of the six specifically-fixed rule IDs happened to appear in this window (dominated by Suricata/dpkg noise + Nmap), so the fix is active but not yet exercised end-to-end by a matching alert |
| MITRE resolution | Resolver unit tests | 15/15 pass | — | pytest | `test_mitre_resolver.py` | C | native precedence, multi-technique, rejection of invalid techniques, ambiguous handling |
| MITRE resolution | Realtime integration tests | 8/8 pass | — | pytest | `test_realtime_mitre_integration.py` | C | resolved vs UNKNOWN branching verified |
| UNKNOWN/unattributed handling | Real UNKNOWN events ingested | 5 | 5 | live Neo4j inspection | Phase 20F (2026-08-31), re-verified intact 2026-09-12 | B | Still present and unchanged after a full Neo4j outage/recovery cycle |
| UNKNOWN/unattributed handling | Fabricated `attack_id` count | 0 | 5 | live Neo4j inspection | same | B | 0% |
| UNKNOWN/unattributed handling | `attack_id` property absent | 5 | 5 | live Neo4j inspection | same | B | 100% — never set, never a placeholder string |
| UNKNOWN/unattributed handling | Technique/MATCHES relationships | 0 | 5 | live Neo4j inspection | same | B | 0% |
| UNKNOWN/unattributed handling | NEXT_TECHNIQUE edges before/after | 3 / 3 | — | live Neo4j inspection | same | B | Unchanged — zero contamination |
| UNKNOWN/unattributed handling | TPS contribution | 0 | 5 | live Neo4j inspection | same | B | Hardcoded 0 in Cypher, not parameterized |
| UNKNOWN/unattributed handling | Raw payload preserved | 5/5 | 5 | live Neo4j inspection | same | B | verbatim rule/agent/timestamp fields intact |
| UNKNOWN/unattributed handling | `predict_next()` reachability from UNKNOWN branch | never | — | code control-flow proof | `realtime_socgraph.py` line-number analysis | D | Structural guarantee, not just observed absence |
| UNKNOWN/unattributed handling | MISP publish reachability from UNKNOWN branch | never | — | code control-flow proof | same | D | Structural guarantee |
| Neo4j graph integrity | Campaign nodes | 111 | — | live query | 2026-09-25 (was 65 on 2026-09-12) | B | Graph has grown via continued live ingestion since the 2026-09-12 snapshot |
| Neo4j graph integrity | AttackEvent nodes | 212 | — | live query | 2026-09-25 (was 124 on 2026-09-12) | B | — |
| Neo4j graph integrity | Technique nodes | 858 | — | live query | 2026-09-25 | B | Real STIX-imported, not synthetic — unchanged, vendored corpus doesn't grow |
| Neo4j graph integrity | Attacker nodes | 9 | — | live query | 2026-09-25 (was 5 on 2026-09-12) | B | — |
| Neo4j graph integrity | Host nodes | 12 | — | live query | 2026-09-25 (was 4 on 2026-09-12) | B | — |
| Neo4j graph integrity | Operation nodes | 50 | — | live query | 2026-09-25 (was 43 on 2026-09-12) | B | Existence confirmed; matching quality not evaluated |
| Neo4j graph integrity | Total nodes (all labels) | 2,539 | — | live query | 2026-09-25 | B | See `paper_submission_status.md` |
| Neo4j graph integrity | Total relationships (all types) | 20,804 | — | live query | 2026-09-25 | B | See `paper_submission_status.md` |
| Neo4j graph integrity | Orphaned AttackEvents | not re-verified this pass | 212 | live query | — | B | The 2026-09-12 figure (0/124) is not carried forward uncritically since the event count has grown; re-run `check_integrity()` before citing a current orphan rate |
| Neo4j graph integrity | NEXT_TECHNIQUE relationships | 3 | — | live query | same | B | Severely data-limited |
| Campaign reconstruction | Orphaned events repaired (historical) | 44 → 27 campaigns | 44 | one-time real-data repair | `campaign_reconstruction.py`, prior phase | A | One-time offline repair of a real production defect |
| Campaign reconstruction | Current orphan count | 0 | 124 | live query | 2026-09-12 | B | Repair holds; no regression |
| Campaign reconstruction | Idempotency tests | 10/10 pass | — | pytest | `test_campaign_reconstruction.py` | C | Repair proven safe to re-run |
| Risk scoring | Normalization/threshold tests | 6/6 pass | — | pytest | `test_risk_scoring.py` | C | Mechanism correctness, not predictive value |
| Risk scoring | Recalculation idempotency tests | 4/4 pass | — | pytest | `test_risk_recalculation.py` | C | Deterministic, re-runnable |
| Risk scoring | Predictive/discriminative value of risk score | — | — | — | — | E | NOT MEASURED |
| MITRE/STIX ingestion | Imported Technique nodes | 858 | — | live query | 2026-09-12 | B | — |
| MITRE/STIX ingestion | STIX corpus version | Enterprise v19.1 | — | file metadata | `mitredata/attack-stix-data/.../enterprise-attack.json` | D | 25,843 total STIX objects, vendored, no runtime internet dependency |
| NEXT_TECHNIQUE | Evaluable predictions | 12 | 12 | offline experiment on real dataset | Phase 18 dataset rebuild | A | — |
| NEXT_TECHNIQUE | Correct | 4 | 12 | offline experiment | same | A | 33.3% |
| NEXT_TECHNIQUE | Incorrect | 8 | 12 | offline experiment | same | A | — |
| NEXT_TECHNIQUE | Accuracy | 33.3% | 12 | offline experiment | same | A | **INSUFFICIENT FOR SUPERVISED ML** (formal Phase 18 verdict) |
| NEXT_TECHNIQUE | Learned edges in graph | 3 | — | live query | 2026-09-12 | B | T1110.001→T1110 (6), T1110→T1078 (5), T1078→T1110.001 (1, below threshold) |
| Prediction engine | `predict_next_readonly` unit tests | 4/4 pass | — | pytest | `test_next_technique_pipeline.py` | C | threshold/no-edge/confidence logic verified |
| Prediction engine | Live prediction accuracy | — | — | — | — | E | Not separately measured beyond NEXT_TECHNIQUE row above |
| Threat attribution | Accuracy / precision / recall / F1 / AUC | — | — | — | — | E | **NOT MEASURED — no ground-truth attacker-identity dataset exists** |
| Threat attribution | Evidence-collector-level unit tests | 2/2 pass (`attribution_collector`) | — | pytest | `test_evidence.py` | C | Collector mechanics only, not attribution accuracy |
| Evidence collection | Collector types | 6 | — | code inventory | `evidence/collectors/*.py` | D | mitre, graph, detection, cti, campaign_history, attribution |
| Evidence collection | Unit tests | 13/13 pass | — | pytest | `test_evidence.py` | C | Includes conflict-detection and dedup logic |
| Evidence-aware investigation | Unit/integration tests | 25/25 pass | — | pytest | `test_investigation.py` | C | Entropy, coverage, stopping criteria, conflict handling |
| Evidence-aware investigation | Evidence orchestrator integration test | 1/1 pass | — | pytest | `test_evidence_orchestrator.py` | C | — |
| Evidence-aware investigation | Real XGBoost probabilities for 3 named campaigns | reported in prior project history | 3 campaigns | prior-session record | not independently re-verified in this or the immediately preceding sessions | E (for this report) | Cite only as "previously recorded," not as fresh evidence |
| Evidence-aware investigation | Investigation-confidence precision/recall | — | — | — | — | E | NOT MEASURED |
| RAG | Indexed real ATT&CK documents | >500 (assert `>500`; comment notes 858 non-deprecated) | — | pytest | `test_rag.py::test_mitre_retriever_indexes_hundreds_of_real_techniques` | C | Real corpus, not synthetic |
| RAG | Corpus version | Enterprise v19.1 | — | file metadata | — | D | — |
| RAG | Full RAG test suite | 7/7 pass | — | pytest | `test_rag.py` | C | — |
| RAG | Retrieval precision/recall | — | — | — | — | E | NOT MEASURED — no labeled query set exists |
| RAG | Embedding/index dimensions | — | — | — | — | E | NOT AVAILABLE — TF-IDF, no fixed embedding dimension documented in code |
| MISP integration | Client initialization | succeeds | — | live observation | listener startup log, 2026-09-12: `"Initialized CTIPublisher -> https://localhost:8443"` | B | No network call at construction (verified by code read) |
| MISP integration | Events published / campaigns synchronized / IOC counts | — | — | — | — | E | NOT MEASURED — MISP containers are entirely absent from the current Docker environment |
| MISP integration | Requirement for listener startup | not required | — | code inspection | `cti_publisher.py.__init__` | D | Confirmed via source read, not assumption |
| MISP integration | Requirement for UNKNOWN ingestion | not required | — | code control-flow proof | `realtime_socgraph.py` | D | Publish call never reached on UNKNOWN branch |
| XGBoost | n | 60 | 60 | offline evaluation | `evaluate_model.py` run, 2026-08-31 | A | Same 60 rows model was trained on |
| XGBoost | Accuracy | 0.917 | 60 | offline evaluation | same | A | **IN-SAMPLE ONLY — NO HELD-OUT GENERALIZATION CLAIM** |
| XGBoost | Macro F1 | 0.676 | 60 | offline evaluation | same | A | In-sample |
| XGBoost | Weighted F1 | 0.910 | 60 | offline evaluation | same | A | In-sample |
| XGBoost | Critical precision/recall/F1 | 1.00 / 0.25 / 0.40 | 4 (support) | offline evaluation | same | A | 3 of 4 Critical campaigns misclassified even in-sample |
| XGBoost | Low precision/recall/F1 | 0.962 / 0.962 / 0.962 | 53 (support) | offline evaluation | same | A | — |
| XGBoost | Medium precision/recall/F1 | 0.50 / 1.00 / 0.667 | 3 (support) | offline evaluation | same | A | — |
| XGBoost | Confusion matrix | `[[1,2,1],[0,51,2],[0,0,3]]` | 60 | offline evaluation | same | A | Rows/cols order: Critical, Low, Medium |
| XGBoost | Mean model confidence | 0.7639 | 60 | offline evaluation | same | A | — |
| XGBoost | Held-out/test-set accuracy | — | — | — | — | E | NOT AVAILABLE — no train/test split exists anywhere in the repository |
| XGBoost | ROC-AUC / PR-AUC / cross-validation | — | — | — | — | E | NOT MEASURED |
| SSL / self-supervised learning | Autoencoder convergence | confirmed | — | pytest | `test_ssl_pipeline.py::test_train_ssl_autoencoder_produces_artifacts_and_converges` | C | Pass/fail only, not a numeric loss value reported here |
| SSL / self-supervised learning | Training artifact | real, 455,715 bytes | — | file inspection | `ml/models/autoencoder_best.pth` | D | Real trained weights on disk |
| SSL / self-supervised learning | Anomaly-detection accuracy/AUC on real attack traffic | — | — | — | — | E | NOT MEASURED |
| GNN | Unit tests | 11/11 pass | — | pytest | `test_gnn.py` | C | Encoder, SAGEConv layer, forward/gradients, train+reload — all on **synthetic** graphs |
| GNN | Performance on real campaigns | — | — | — | — | E | NOT MEASURED — no real-campaign benchmark exists |
| Dataset generation | Campaigns | 60 | — | file inspection | `campaign_dataset.csv`, 2026-09-12 | B (static file) | Frozen since Phase 18 |
| Dataset generation | Columns | 66 | — | file inspection | same | B | 57 declared features + identifiers/labels |
| Dataset generation | Attackers / Victims / Pairs | 3 / 2 / 4 | 60 | offline analysis | Phase 17 validity-gate report | A | Severely limited diversity |
| Dataset generation | Severity distribution | Low=53, Medium=3, Critical=4, High=0 | 60 | file inspection | same | B | No High-severity examples exist |
| Dataset generation | Exact duplicate feature-vector rows | 11 (3 groups) | 60 | offline analysis | Phase 17 report | A | — |
| Dataset generation | Zero-variance features | 23 | 57 | offline analysis | same | A | 40% of declared features carry no signal |
| Dataset generation | Dataset validity verdict | `NOT_READY_FOR_CALIBRATION` | — | offline analysis | `scripts/phase17_dataset_validity_gate.py` | A | Formal, self-issued verdict — not fixed since |
| Dashboard/API | Endpoint tests | 7/7 pass (`test_dashboard_api.py`) + 3/3 (`test_dashboard_api_campaign_context.py`) | — | pytest | same | C | — |
| Dashboard/API | Load/latency testing | — | — | — | — | E | NOT MEASURED |
| Phase 20 overall | Total regression suite (all modules) | 150/150 pass | — | pytest | fresh run 2026-09-12 | C | — |
| Phase 20 overall | Compile check | clean | — | compileall | fresh run 2026-09-12 | D | — |

---

## Task 3 — Results by Evidence Strength

### A. Experimentally measured (offline, real-data experiments)
- Campaign reconstruction: 44 orphaned events repaired → 27 campaigns (one-time, historical).
- Dataset statistics: 60 campaigns, 3 attackers, 2 victims, 4 pairs, 11 duplicate rows, 23 zero-variance features, severity distribution.
- Phase 17 dataset validity verdict: `NOT_READY_FOR_CALIBRATION`.
- NEXT_TECHNIQUE: 4/12 correct (33.3%), verdict `INSUFFICIENT FOR SUPERVISED ML`.
- XGBoost in-sample evaluation: accuracy 0.917, macro F1 0.676, per-class precision/recall/F1, confusion matrix (n=60, in-sample only).

### B. Live-system measured (observed from the running system; 2026-09-25 unless noted)
- Neo4j (2026-09-25): 111 Campaigns, 212 AttackEvents, 858 Technique nodes, 9 Attacker, 12 Host, 50 Operation — 2,539 total nodes, 20,804 total relationships. (2026-09-12 snapshot for comparison: 65/124/858/5/4/43, 0 orphans, 119 MATCHES — orphan/MATCHES counts not re-verified at the new scale.)
- 5 real UNKNOWN AttackEvents (created 2026-08-31, re-verified intact 2026-09-12): 0 fabricated `attack_id`, 0 Technique links, 0 TPS, raw payload preserved.
- MITRE resolution coverage: 391 alerts (2026-08-31) — 39 native (10.0%), 352 UNKNOWN (90.0%); superseded 239-alert snapshot (98.7% UNKNOWN) explicitly distinguished; current 13-alert file snapshot (100% UNKNOWN, not representative).
- Listener offset-recovery behavior fired correctly on restart (log-confirmed).
- MISP client initializes without a network call; MISP containers currently entirely absent from the Docker environment.

### C. Test-verified (unit/integration/regression tests)
- 689/689 backend tests passing across 65 files (re-verified 2026-09-25; was 150/150 across fewer files on 2026-09-12), plus 109/109 frontend tests across 17 files — 798 total.
- SOAR layer alone (new since the 2026-09-12 run): 83 tests across 9 files (`tests/test_soar_*.py`).
- Per-module test counts below are the original 2026-09-12 breakdown, kept as a historical record of what existed at that snapshot — they do not sum to the current 689 and have not been individually re-counted this pass: MITRE resolver 15, realtime integration 8, evidence collectors 13, investigation loop 25, GNN 11, campaign reconstruction 10, risk scoring 6, risk recalculation 4, RAG 7, dashboard API 10, mitre_mapper 8, ml_pipeline 6, next_technique_pipeline 9, ssl_pipeline 4, label_generator 6, campaign_manager_tps 3, dataset_leakage 3, evidence_orchestrator 1, query_console 1, train_xgboost_paths 1.

### D. Static/structural verification (code-level facts, not runtime observations)
- UNKNOWN branch structurally never reaches `predict_next()`, `append_technique()`, `chain_updater.update_attack_chain()`, or MISP publish (control-flow proof by line-number analysis).
- MISP client construction performs no network I/O.
- STIX corpus version and object counts (file metadata, not a live measurement).

### E. Not measured (no defensible quantitative result exists)
- Threat attribution accuracy/precision/recall/F1/AUC.
- RAG retrieval precision/recall.
- GNN performance on real campaigns.
- SSL anomaly-detection accuracy/AUC on real attack traffic.
- Held-out ML generalization accuracy, ROC-AUC, PR-AUC, cross-validation.
- API/listener latency, throughput, memory utilization.
- Confirmed live MISP publication success (events published, IOCs enriched).
- Investigation-loop confidence precision/recall.
- Real XGBoost probabilities for the 3 named campaigns cited in earlier project history (not independently re-verified this session or the prior one).

---

## Task 5 — Reviewer-Safe Interpretation of Major Metrics

### 91.7% XGBoost accuracy
1. **Proves**: the model fits its own 60 training rows well in aggregate (weighted F1 0.91), and specifically demonstrates it can separate the majority Low class cleanly.
2. **Does not prove**: that the model generalizes to unseen campaigns — no held-out split exists anywhere in the repository.
3. **Likely question**: "Is this a test accuracy?"
4. **Defensible answer**: "No — it's in-sample accuracy on the same 60 rows used for training. We don't have enough real campaign diversity yet for a meaningful held-out split, and we say so explicitly rather than implying a generalization claim we can't support."

### 33% NEXT_TECHNIQUE accuracy
1. **Proves**: the corrected labeling pipeline (Phase 18) can be evaluated at all, and produces a real, non-fabricated per-transition accuracy number.
2. **Does not prove**: that next-technique prediction works as a viable capability — n=12 is far too small.
3. **Likely question**: "Why is this so low?"
4. **Defensible answer**: "It's not being presented as a working predictor — our own dataset-validity audit concluded `INSUFFICIENT_FOR_SUPERVISED_ML` before we'd even look at accuracy, and 33% on 12 samples confirms that assessment rather than contradicting it."

### 90% UNKNOWN MITRE coverage
1. **Proves**: most of Wazuh's own out-of-the-box ruleset has no native ATT&CK tag for the telemetry we're seeing, and CYUKTI correctly preserves that telemetry rather than discarding it.
2. **Does not prove**: a coverage failure on CYUKTI's part — this is Wazuh's own ruleset gap, and preserving it (rather than fabricating attribution) is the intended design outcome.
3. **Likely question**: "Isn't 90% unresolved a bad result?"
4. **Defensible answer**: "No — that would only be true if the goal were maximizing attribution rate. Our goal is never fabricating attribution we can't defend. 90% UNKNOWN with 0% fabricated technique IDs is the correct, intended outcome of that design choice."

### 858 ATT&CK techniques
1. **Proves**: a real, versioned, vendored ATT&CK STIX corpus (Enterprise v19.1) is actually imported into the live graph, not referenced abstractly.
2. **Does not prove**: that every technique is actively used or referenced by real campaign data — most of the 858 exist as reference metadata, not as observed attacker behavior.
3. **Likely question**: "Did you build your own ATT&CK database?"
4. **Defensible answer**: "No — we vendor the official MITRE STIX release and import it as-is, so every technique ID we ever propose is validated against real, current ATT&CK data, including revoked/deprecated status."

### 0 orphaned events (as measured 2026-09-12, at 124 AttackEvents; not re-verified at the current 212)
1. **Proves**: every AttackEvent in the graph at measurement time (124/124) was correctly linked to exactly one Campaign — the data-integrity invariant established in an earlier phase holds as of that check.
2. **Does not prove**: that no future ingestion could ever produce an orphan — it's a current-state measurement, not a guarantee.
3. **Likely question**: "How do you know this holds over time?"
4. **Defensible answer**: "We have a `check_integrity()` function specifically for this, we ran it live today, and it's part of the deliberate data-integrity work from Phase 16 that found and fixed 44 real orphaned events."

### 689/689 tests (150/150 at the time this section was first written)
1. **Proves**: every unit/integration test the project currently has passes, including tests specifically written to catch the exact production bugs found earlier in the project (leakage, mapping errors, UNKNOWN-path safety).
2. **Does not prove**: comprehensive coverage of all possible failure modes, or that the system behaves correctly under conditions no test anticipates (e.g., the maintenance-worker bug found live, which had zero test coverage before it was discovered in production).
3. **Likely question**: "Does 100% test pass rate mean the system is bug-free?"
4. **Defensible answer**: "No — we found a real production bug live just weeks ago that no test caught, specifically because the test suite didn't anticipate that interaction. We disclose it rather than hide it."

### 44 repaired events
1. **Proves**: a real, found-in-production data-integrity defect (44 AttackEvents with no parent Campaign) was root-caused and repaired without guessing or fabricating provenance — the repair is idempotent and still holds (0 orphans today).
2. **Does not prove**: that all data-integrity issues in the system are resolved — this was one specific, diagnosed defect.
3. **Likely question**: "How did you know the repair was correct rather than just making the count look better?"
4. **Defensible answer**: "The repair only reconstructs a Campaign from data already present on its own orphaned children — it never guesses, and it's proven idempotent by test. We can also show exactly how each of the 44 was classified before repairing."

### Lack of attribution ground truth
**Defensible answer**: "We don't have a labeled attacker-identity dataset, so we report attribution accuracy as NOT MEASURED rather than inventing a number. The attribution engine is implemented and exercised at the evidence-collector level, but we don't claim a performance result we can't support."

### Absence of held-out ML evaluation
**Defensible answer**: "The dataset currently has 60 campaigns from only 3 real attacker identities — our own validity-gate analysis concluded this isn't yet diverse enough to support a meaningful train/test split, so we report in-sample metrics only, labeled as such, rather than manufacturing a held-out split that wouldn't be statistically meaningful anyway."

---

## Task 6 — CYUKTI Quantitative Evaluation — Current Status

### Demonstrated
- Live Wazuh → Neo4j ingestion pipeline, including recovery from a real infrastructure outage with zero data loss (65/124/858 counts identical before/after, as measured 2026-09-12; graph has since grown to 111/212/858 as of 2026-09-25 via continued live ingestion).
- MITRE resolution with empirically measured, live coverage (391 alerts, 2026-08-31: 10.0% native / 90.0% UNKNOWN; newest snapshot, 2026-09-25 post-rule-fix-and-reboot: 120 alerts, 21.7% native / 78.3% UNKNOWN).
- UNKNOWN-path safety: 5 real events, 0 fabricated IDs, 0 chain contamination, live-verified twice (Aug 31 creation, Sep 12 re-confirmation after outage).
- Campaign reconstruction: 44 real orphaned events repaired, idempotent, 0 orphans as of the 2026-09-12 check.
- 689/689 backend automated tests passing (65 files) + 109/109 frontend (17 files) = 798 total, as of 2026-09-25 (was 150/150 on 2026-09-12).
- Real ATT&CK STIX import: 858 techniques, v19.1 (unchanged — vendored corpus).

### Partially demonstrated
- XGBoost severity classification (in-sample metrics only, no held-out split).
- NEXT_TECHNIQUE prediction (pipeline correct, but self-assessed as statistically insufficient — 12 evaluable samples).
- Evidence-aware investigation architecture (25 tests pass; no live end-to-end investigation trace was freshly generated in this or the immediately preceding session).
- RAG retrieval (corpus scale confirmed; retrieval quality unmeasured).
- MISP integration (client code confirmed non-blocking and correctly gated; live publish success unmeasured because the MISP stack is currently absent).

### Not yet demonstrated
- Threat attribution accuracy (no ground-truth dataset).
- GNN performance on real campaign graphs (synthetic-only tests).
- SSL/autoencoder anomaly-detection accuracy on real attack traffic.
- Any latency, throughput, or memory benchmark.
- Held-out ML generalization of any kind.

### Overall maturity: **RESEARCH PROTOTYPE**

- Real, live, multi-service integration (Wazuh + Neo4j + a trained ML stack) — beyond a bare prototype.
- A genuine production defect (44 orphaned events) was found and durably repaired, with idempotency proven by test.
- A second live defect (Neo4j container outage) was recovered from with zero data loss using a minimal, non-destructive procedure — evidence of operational resilience, not just design intent.
- The project's own internal audits (Phase 17 dataset validity gate, Phase 18 NEXT_TECHNIQUE assessment) explicitly conclude several core modules are **not yet ready** for the claims a mature system would make — this self-limiting rigor is itself evidence of research discipline, not a weakness to hide.
- No held-out evaluation methodology exists for any ML component — this alone rules out "Production-like Research System" or higher.
- At least one known, disclosed, unfixed defect remains live (`maintenance_worker` `last_seen=None` exception) — non-blocking, but unresolved.
- Several major modules (attribution, GNN, RAG retrieval quality, MISP live publication) have zero quantitative benchmark, only unit-test-level verification.

---

## Task 7 — Modification Verification

`git status --short` (run both before and after this report was written): only pre-existing untracked directories (`aisd/`, `isfcr/CYUKTI/review_pack/`, `runs/`) plus this new `isfcr/CYUKTI/review/` directory. No tracked source file was modified. No Neo4j write query was issued (only `MATCH`/`RETURN` reads). No Docker start/stop/create/remove command was issued in this task. No listener was started, stopped, or restarted in this task (the running instance, PID 8814, was inspected read-only via `pgrep`, not touched). No Wazuh alert or telemetry was generated. **No accidental modification occurred.**
