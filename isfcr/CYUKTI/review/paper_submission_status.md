# CYUKTI — Paper Submission Status

> Last updated: 2026-09-26

## Paper
- Title: CYUKTI: A Graph-Based Campaign Correlation and Evidence-Aware Investigation Framework for Real-Time Cyber Threat Detection
- Format: IEEEtran two-column LaTeX, 7 pages compiled
- Target: IEEE CONECCT 2027 (Bengaluru) or CSNT 2027 (March, India)
- Status: Draft complete, zero unfilled placeholders, ready for guide review

## Key numbers in the paper (source of truth)
- Neo4j: 2,539 nodes, 20,804 relationships
- Campaigns: 111 | Operations: 50 | AttackEvents: 212
- Dedup ratio: 93.3% (212 / 3,185 occurrences)
- MITRE: 21.7% NATIVE_WAZUH, 78.3% UNKNOWN (post-restart)
- XGBoost LOGO: 90.0% accuracy, macro F1 0.449
- GNN ablation: delta 0.000 on all metrics
- Latency: p50 35.5ms, p95 66.9ms end-to-end
- Tests: 757 backend + 109 frontend = 866 passing (+68 backend tests from the new independent evaluation framework, `evaluation/`)
- API: 46 routes (33 + 13 SOAR)
- Investigation actions: 10
- Prediction accuracy: UNMEASURABLE for new data (live NEXT_TECHNIQUE graph unchanged at 3 edges); historical Phase 18 result reused: 4/12 = 33.3%, `INSUFFICIENT_FOR_SUPERVISED_ML`
- Attribution accuracy: **PARTIALLY MEASURED** (preliminary) — 74.4% [59.8%, 85.1% 95% CI], n=43, AI-assisted AUTO_PROPOSED ground truth against real attacker_ip, not yet human-reviewed — see `review/paper_metrics_source_of_truth.md`
- MITRE mapping accuracy: **PARTIALLY MEASURED** (preliminary) — P=1.00/R=0.667/F1=0.80, n=100
- Threat qualification accuracy: **PARTIALLY MEASURED** (preliminary) — accuracy=0.083, n=24 (large real disagreement, disclosed as an open question — see `review/evaluation_results.md`)
- Campaign correlation (pairwise/ARI): **PARTIALLY MEASURED** (preliminary) — pairwise F1=0.0, ARI=0.0, n=46 real campaigns (severe fragmentation found: one real session split across up to 25 CYUKTI campaigns)
- RAG retrieval precision/recall: NOT MEASURED (evaluator built + tested; review queue exported, needs human relevance judgments)
- Investigation confidence P/R: NOT MEASURED (evaluator built + tested; review queue exported, needs human verdicts)

## What the paper discloses as limitations
- TPS_CEILING 1500 vs observed max 34,820
- Constant confidence floor 42.86% (3/7 hardcoded terms)
- Threat-intel fields permanently zero
- Investigation action ordering invariant across campaigns (Phase 21/22)
- Dataset NOT_READY_FOR_CALIBRATION (Phase 17 verdict)
- No adversarial robustness analysis
- No user study
- No commercial SIEM comparison
- MISP currently unreachable
- Shuffle never live-triggered
- Active containment/closed-loop response layer (backend/active_response/) is implemented and tested (876 backend tests) but has never executed against a real host — BLOCKED_BY_ENVIRONMENT, not live-verified; if the paper describes this capability at all, it must be framed as an implemented, unit/integration-tested architecture, not a demonstrated defensive capability. See `review/final_experimental_validation_report.md`.
- Campaign correlation fragments real repeated-session traffic heavily (one real brute-force session split across up to 25 separate Campaign nodes) — found via the new independent evaluation framework, not previously measured
- Independent evaluation ground truth (MITRE/threat-qualification/attribution/campaign-correlation) is AI-assisted-preliminary, not yet human-reviewed — see `review/paper_metrics_source_of_truth.md`
