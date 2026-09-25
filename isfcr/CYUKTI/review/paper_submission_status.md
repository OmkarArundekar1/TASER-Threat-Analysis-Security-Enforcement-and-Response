# CYUKTI — Paper Submission Status

> Last updated: 2026-09-25

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
- Tests: 689 backend + 109 frontend = 798 passing
- API: 46 routes (33 + 13 SOAR)
- Investigation actions: 10
- Prediction accuracy: 0/0 (unmeasurable)
- Attribution accuracy: NOT MEASURED

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
