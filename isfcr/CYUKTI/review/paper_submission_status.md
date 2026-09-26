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
  *Planned:* the same Phase 19 dataset expansion is intended to accumulate enough real technique transitions to revisit this verdict.
- Attribution accuracy: **PARTIALLY MEASURED** (preliminary) — 74.4% [59.8%, 85.1% 95% CI], n=43, AI-assisted AUTO_PROPOSED ground truth against real attacker_ip, not yet human-reviewed — see `review/paper_metrics_source_of_truth.md`
  *Planned:* complete human review of the AUTO_PROPOSED ground truth before this number is treated as final.
- MITRE mapping accuracy: **PARTIALLY MEASURED** (preliminary) — P=1.00/R=0.667/F1=0.80, n=100
  *Planned:* complete human review of the underlying labels before this number is treated as final.
- Threat qualification accuracy: **PARTIALLY MEASURED** (preliminary AUTO_PROPOSED: accuracy=0.083; re-examined MEASURED_ASSISTANT_ADJUDICATED, per-campaign: accuracy=0.292, but the real finding is 100% recall / 0% precision on QUALIFIED_THREAT — see `review/evaluation_threat_qualification_adjudication.md`). Neither number is human-reviewed.
  *Planned:* complete human review of both the AUTO_PROPOSED and assistant-adjudicated ground truth, and investigate the 0% precision finding before any accuracy figure is treated as final.
- Campaign correlation (pairwise/ARI): **PARTIALLY MEASURED** (preliminary) — pairwise F1=0.0, ARI=0.0, n=46 real campaigns (severe fragmentation found: one real session split across up to 25 CYUKTI campaigns)
  *Planned:* complete human review of the correlation ground truth and investigate campaign-merging heuristics to address the disclosed fragmentation.
- RAG retrieval precision/recall: NOT MEASURED (evaluator built + tested; review queue exported, needs human relevance judgments)
  *Planned:* construct a labeled query set (the exported review queue) to measure retrieval precision/recall directly.
- Investigation confidence P/R: NOT MEASURED (evaluator built + tested; review queue exported, needs human verdicts)
  *Planned:* collect human verdicts against the exported review queue to measure investigation-confidence precision/recall directly.

## What the paper discloses as limitations
- TPS_CEILING 1500 vs observed max 34,820
  *Planned:* re-derive `TPS_CEILING` from the observed peak once a sustained-load ingestion benchmark is available.
- Constant confidence floor 42.86% (3/7 hardcoded terms)
  *Planned:* replace the hardcoded floor terms with measured values once enough live investigation runs support calibrating them.
- Threat-intel fields permanently zero
  *Planned:* reconfirm once `MISP_API_KEY` is configured and the MISP service is running for a verification session, which is the source of these fields.
- Investigation action ordering invariant across campaigns (Phase 21/22)
  *Planned:* per Phase 22's own recommendation, revisit the NBE scoring formula so adaptive terms can encode evidence content, not just action-type sequence, once that redesign is scoped.
- Dataset NOT_READY_FOR_CALIBRATION (Phase 17 verdict)
  *Planned:* the Phase 19 dataset-expansion spec directly targets the gaps behind this verdict.
- No adversarial robustness analysis
  *Planned:* add an adversarial robustness evaluation once the Phase 19 dataset expansion provides enough data to support one.
- No user study
  *Planned:* scope a user study with SOC analysts once the tool is stable enough to evaluate with human participants.
- No commercial SIEM comparison
  *Planned:* add a comparative benchmark against a commercial SIEM once a suitable evaluation environment is available.
- MISP currently unreachable
  *Planned:* reconfirm once `MISP_API_KEY` is configured and the MISP service is running for a verification session.
- Shuffle never live-triggered
  *Planned:* live-trigger Shuffle against the configured webhook in a future verification session.
- Active containment/closed-loop response layer (backend/active_response/) is implemented and tested (876 backend tests) but has never executed against a real host — BLOCKED_BY_ENVIRONMENT, not live-verified; if the paper describes this capability at all, it must be framed as an implemented, unit/integration-tested architecture, not a demonstrated defensive capability. See `review/final_experimental_validation_report.md`.
  *Planned:* run one real `BLOCK_SOURCE_IP` action via `IptablesFirewallBackend` with a before/after connection test through `ContainmentVerifier` once operator-driven access to the Kali/Ubuntu VMs is available.
- Campaign correlation fragments real repeated-session traffic heavily (one real brute-force session split across up to 25 separate Campaign nodes) — found via the new independent evaluation framework, not previously measured
  *Planned:* investigate campaign-merging heuristics to reduce this fragmentation, tracked as a new finding from the independent evaluation framework.
- Independent evaluation ground truth (MITRE/threat-qualification/attribution/campaign-correlation) is AI-assisted-preliminary, not yet human-reviewed — see `review/paper_metrics_source_of_truth.md`
  *Planned:* complete human review of all four AUTO_PROPOSED ground-truth categories before any of these metrics are treated as final.
