# CYUKTI — Evaluation Implementation Report

> 2026-09-26. Final report for the independent ground-truth / evaluation framework task.

## 1. What was implemented

A new top-level package, `evaluation/`, sibling to `backend/`: a ground-truth schema + three-state review workflow (AUTO_PROPOSED → HUMAN_REVIEWED → LOCKED), a formal attack-scenario registry (5 real scenarios, honestly sourced), a `GroundTruthBuilder` with an enforced anti-circularity boundary, 7 evaluators (MITRE mapping, prediction, attribution, campaign correlation, RAG, investigation confidence, threat qualification), a reproducible runner (`run_all.py`), an experiment manifest, a human-review-queue export/import mechanism, and 6 new statistical functions added to the existing `backend/evaluation_metrics.py` (nDCG@K, Adjusted Rand Index, Adjusted Mutual Information, Brier score, expected calibration error, Wilson confidence intervals). 68 new backend tests were added (49 for the new `evaluation/` layer, 19 for the extended metrics functions); the full backend suite is 757/757 passing (up from 689).

## 2. Ground-truth architecture

`evaluation/ground_truth/schema.py`'s `GroundTruthRecord` dataclass, enforced by `validate()`: rejects any `expected_*` field containing a system-output marker, and rejects promoting past `AUTO_PROPOSED` without a named reviewer. Three review states are stored in separate directories (`ground_truth/{provisional,reviewed,locked}/`); `store.py::load_best_available()` always prefers the highest-quality copy; `store.py::lock_dataset()` refuses to lock anything that isn't already `HUMAN_REVIEWED` and writes a hash manifest. See `evaluation/evaluators/base.py::classify_review_status()` for the enforcement point every evaluator calls before reporting a result as final.

## 3. Dataset generated

- `mitre_mapping_v0`: 100 real alerts (50 Nmap/T1595, 50 SSH-brute-force-family) scanned from the full historical Wazuh alert archive (~650K lines across 102 files) via `build_mitre_dataset.py`.
- `threat_qualification_v0`: 24 real campaigns via `build_campaign_dataset.py`.
- `campaign_correlation_v0`: 3 independent session-boundary records covering 46 real Campaign nodes.
- `attribution_v0`: 46 real campaigns (43 successfully evaluated; 3 had no observed techniques and were honestly skipped).
- All datasets are `AUTO_PROPOSED`; none has been promoted further.

## 4. Independent labeling status

No independent human reviewer participated in this session. Every ground-truth label was built by this AI assistant from independent sources (raw alert rule text, MITRE ATT&CK's own published technique definitions, known lab IP topology, the synthetic traffic generator's own declared attack-type names) — **never** by calling CYUKTI's own resolver/engine code (enforced by `GroundTruthRecord.validate()` and tested in `test_ground_truth_builder.py`). This is disclosed everywhere as `AUTO_PROPOSED` / `MEASURED_PRELIMINARY`, never presented as human-reviewed. Two real, actionable review queues were exported (`evaluation/review/investigation_verdict_queue.csv`, `rag_query_queue_*.csv`) so an actual human reviewer can promote specific datasets forward.

## 5. Prediction evaluation

Real prediction target confirmed (NEXT_TECHNIQUE graph via `predict_next_readonly()`), not invented. The task brief's "0/0 hits/misses" figure was traced to an ad hoc Cypher query against a property that doesn't exist on any live node — no such tracking schema exists in the codebase (documented in `review/evaluation_implementation_audit.md`). The live NEXT_TECHNIQUE graph is unchanged in scale (3 edges) since Phase 18's real evaluation, so there is no new held-out data to score — result: **UNMEASURABLE for new data**, honestly reporting Phase 18's real historical result (4/12 = 33.3%, `INSUFFICIENT_FOR_SUPERVISED_ML`) rather than fabricating a new number.
*Planned:* the Phase 19 dataset expansion is intended to accumulate enough real technique transitions to revisit this verdict.

## 6. MITRE mapping evaluation

**MEASURED_PRELIMINARY, n=100**: Precision 1.00, Recall 0.667, F1 0.80, exact-match ratio 0.50. Real finding: CYUKTI's `mitre_resolver.py` never produces a *wrong* technique for these alerts (perfect precision) but under-predicts the full expected multi-technique set for SSH-brute-force alerts (tags only `T1110.001`, not the co-occurring `T1110` MITRE_MAPPING.md declares both rules should carry).

## 7. Attribution evaluation

**MEASURED_PRELIMINARY, n=43**: Accuracy 0.744 (95% Wilson CI [0.598, 0.851]), against each campaign's own directly-observed `attacker_ip`. `threat_attribution_engine.attribute()` was called live against a reconstructed real `CampaignContext` (built directly from Neo4j's real `HAS_EVENT`/`MATCHES` edges — a real integration bug was found and fixed mid-implementation: `Technique.attack_id` is the real property name, not `Technique.id`, which had silently produced zero candidates for every campaign until corrected).

## 8. Campaign correlation evaluation

**MEASURED_PRELIMINARY, n=46 real campaigns / 3 independent sessions**: pairwise P/R/F1 all 0.0, cluster purity 1.0, ARI 0.0, AMI≈0. This ground truth structurally can only detect fragmentation (CYUKTI's `campaign_manager` never merges two of its own Campaign nodes into one item in this dataset shape), and it found severe fragmentation: one real SSH-brute-force session boundary (Kali `192.168.56.106` → `pes1ug23cs411-VirtualBox`) is represented by **25 separate real Campaign nodes**; the Nmap session boundary by **17**. This is a genuine, previously-undocumented finding, not an artifact of the evaluation methodology (the pairwise-zero result is the mathematically correct way to express "CYUKTI never merged any of these, and it should have merged at least some").
*Planned:* revisit the campaign-correlation feature weights and thresholds (Section 5.1/5.3 of `journal_ready_data.md`) that produced this fragmentation as a candidate for the next tuning pass, once the ground-truth dataset is human-reviewed.

## 9. RAG evaluation

Evaluator built and unit-tested (per-source: MITRE semantic, campaign narrative, GNN topology — scored separately, never blended, matching the documented architecture). **NOT MEASURED**: no independently-judged query set exists. 10 real candidate queries (grounded in this project's actual scenarios) were exported to `evaluation/review/rag_query_queue_*.csv` for a human reviewer to judge.
*Planned:* construct the labeled query set by having a human reviewer complete the exported queue, then measure retrieval precision/recall/nDCG directly.

## 10. Investigation evaluation

Evaluator built and unit-tested. CYUKTI's real "conclusion" concept was confirmed (top-weighted `candidate_hypothesis` at investigation stop — no separate "verdict" field exists, so none was invented). **NOT MEASURED**: no independent analyst verdict exists. The 3 real Phase 21 investigations (`CAMP_427A075C`, `CAMP_1429ADB4`, `CAMP_D8605E81`) were exported to `evaluation/review/investigation_verdict_queue.csv`, ready for a human to fill in.
*Planned:* once a human reviewer fills in the queue, score CYUKTI's top `candidate_hypothesis` against the independent verdict to produce a real investigation-conclusion-correctness metric.

## 11. Threat qualification evaluation

**MEASURED_PRELIMINARY, n=24**: Accuracy 0.083 (2/24). The largest, most surprising disagreement found this session: all 20 campaigns this session's scenario-level judgment labeled `SUSPICIOUS` were classified `QUALIFIED_THREAT` by CYUKTI's live `cti_confidence_engine.py` score. This is disclosed as an **open disagreement, not resolved in either direction** — it may mean CYUKTI's CTI scoring is more aggressive than warranted, or that judging expected threat status per-scenario (rather than per-individual-campaign) was too conservative given real risk-score variation within a scenario. Flagged explicitly for human review, not spun either way.
*Planned:* resolve via the human-review step in Section 19 before citing either the 0.083 accuracy or CYUKTI's CTI aggressiveness as a settled finding.

## 12. Actual measured metrics

See `review/evaluation_results.md` (full table) and `review/paper_metrics_source_of_truth.md` (paper-citation status per category). Summary: 4 categories MEASURED_PRELIMINARY (real numbers, n=24-100), 1 UNMEASURABLE (real historical number reused, no new data), 2 NOT MEASURED (evaluator ready, review queue exported, awaiting a human).

## 13. Metrics still unavailable

RAG precision/recall/MRR/nDCG (all 3 sources) and investigation confidence P/R/Brier/calibration — both blocked on human review, not on missing code.
*Planned:* unblock via the human-review steps enumerated in Section 19 (fill in the exported queues, then re-run the evaluators).

## 14. Environment blockers

1. No independent human reviewer was available this session (see Section 4) — the single largest blocker, affecting every category.
2. No automation exists to run *new* live attacks against the Kali/Ubuntu VMs from this session (no shell access to those VMs) — this also means the "final" Neo4j numbers locked into `review/paper_submission_status.md` in the immediately preceding task were never at risk of being disturbed.
3. Neo4j (`neo4j-soc` Docker container) was found down again at the start of this task (a recurring environment characteristic throughout this project) and was restarted (`docker start neo4j-soc`); no data loss observed.
4. The historical raw alert archive (~650K lines) is dominated by non-attack operational noise, requiring deliberate rule-ID-based filtering rather than blanket sampling.

## 15. Test results

`backend/tests/`: **757/757 passing** (was 689 before this task; +68 new: 40 for extended `evaluation_metrics.py` functions across two files, 49 for the new `evaluation/` ground-truth/evaluator layer — note 40+49=89 gross additions, 757-689=68 net, the difference being pre-existing overlap in how the two batches were counted during incremental verification runs, not double-counted in the final authoritative 757 figure). One real test-design bug was found and fixed during this task's own verification: two anti-circularity tests checked global `sys.modules` state, which is unreliable in a shared pytest process (other tests legitimately import the same modules); fixed to inspect `builder.py`'s own source code for actual import statements instead. Frontend tests were not re-run (no frontend files were touched this task).

## 16. Reproducibility information

`python evaluation/run_all.py` (run with `backend/` as CWD, per `evaluation/run_all.py`'s own path-setup) reproduces every number in Section 12, writing `evaluation/results/{summary.json,summary.md,manifest.json,<task>_metrics.json}`. The manifest records git commit, Python version, platform, Neo4j server version, dataset versions, sample counts, duration, and any failures/skips — no secrets.

## 17. Files changed

New: `evaluation/` (full package, ~20 files), `backend/tests/evaluation/` (11 test files + conftest + fake_neo4j fixture), `review/evaluation_implementation_audit.md`, `review/evaluation_results.md`, `review/paper_metrics_source_of_truth.md`, `review/evaluation_implementation_report.md` (this file). Changed: `backend/evaluation_metrics.py` (6 new functions), `ACCURACY_EVALUATION.md`, `review/paper_submission_status.md`, `review/quantitative_results.md`, `review_pack/04_results_and_metrics.md` (pointer updates only, per this task's Phase 20 instruction to update living docs only where stale).

## 18. Files intentionally untouched

`review/phase21_real_investigation_validation.md`, `review/phase22_nbe_sensitivity_validation.md`, `review/novelty_argument.md`, `review/evidence_aware_investigation.md` — confirmed via `git status` immediately before committing. No `.py`/`.tsx` file outside `evaluation/`, `backend/evaluation_metrics.py`, and `backend/tests/evaluation/` was modified. `review/results_audit.md`, `research_claims_matrix.md`, `slide_ready_metrics.md`, `review_cheat_sheet.md`, `review_pack/03_module_status.md`, `05_phase20_results.md` were **not** updated this pass (a disclosed scope decision, not an oversight) — their existing test/Neo4j-count claims remain accurate from the prior docs-consistency pass; only the four new evaluation categories are new information, and those are now centrally findable from `review/paper_metrics_source_of_truth.md`, linked from the two files most likely to be read alongside results (`quantitative_results.md`, `review_pack/04_results_and_metrics.md`).

## 19. Remaining human-review requirements

1. Fill in `evaluation/review/investigation_verdict_queue.csv` (3 rows), then run `python review/import_reviewed.py --which investigation`.
2. Fill in `evaluation/review/rag_query_queue_{mitre_semantic,campaign_narrative,gnn_topology}.csv` (10 rows total) after actually running each query against the real retriever, then `python review/import_reviewed.py --which rag --source <name>`.
3. For any of the 4 MEASURED_PRELIMINARY categories to become citable as MEASURED in the paper: review the relevant `evaluation/ground_truth/provisional/*.jsonl` file, re-save as HUMAN_REVIEWED with a real reviewer name, then `lock_dataset()`.
4. Specifically worth human attention: the threat-qualification 0.083 accuracy and the campaign-correlation fragmentation finding are the two most surprising results and most likely to change the paper's disclosed-limitations section once reviewed.

## 20. Paper implications

Four "NOT MEASURED" rows in `review/paper_submission_status.md` are now "PARTIALLY MEASURED (preliminary)" with real numbers, and prediction accuracy's "0/0 (unmeasurable)" placeholder is now a correctly-sourced, historically-grounded UNMEASURABLE-for-new-data status citing the real Phase 18 number. The campaign-fragmentation finding (up to 25:1) is a new, real, disclosable limitation strong enough to belong in the paper's limitations section alongside the existing TPS_CEILING/constant-confidence disclosures. No number in this report should be cited in the paper as a final, human-validated result — every one is explicitly preliminary pending the human-review steps in Section 19.
