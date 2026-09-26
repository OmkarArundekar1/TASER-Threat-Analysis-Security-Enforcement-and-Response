# CYUKTI Accuracy / Evaluation Framework

> **Update, 2026-09-26**: an independent evaluation framework (`evaluation/`) now exists and has produced real, preliminary (AI-assisted, not yet human-reviewed) measurements for MITRE mapping, threat qualification, campaign correlation, and attribution — see `review/paper_metrics_source_of_truth.md` for the authoritative current status of each row below and `review/evaluation_implementation_report.md` for full methodology. The (A)/(B)/(C) labels below are left as originally written; the new results are a **(B-preliminary)** category this task introduces, not a clean upgrade to (A), because the ground truth is independent of CYUKTI's code but not independent of a certified human reviewer.

## The honest starting point

CYUKTI has multiple distinct tasks, each requiring its own metric — and, critically, **independently-verified ground truth is not equally available for all of them in this environment.** Every task below is labeled:

- **(A) measured on real, independent ground truth**
- **(B) measured on proxy/diagnostic labels** (useful signal, but not independent ground truth)
- **(C) unavailable — no independent ground truth exists in this environment**

No task is ever scored against labels CYUKTI itself generated.

## Per-task status

| Task | Status | Detail |
|---|---|---|
| GNN historical campaign retrieval | **(A)** | `ml/gnn/retrieval_evaluation.py` — LOGO cross-validation grouped by real attacker IP (a raw identity fact, never a derived similarity score). See `GNN_RETRIEVAL_EVALUATION.md`. |
| Severity prediction (XGBoost, incl. GNN-embedding ablation) | **(A)** | `ml/gnn/xgboost_ablation.py` — LOGO-based. Frozen null result (no measurable benefit from GNN embeddings) — **not reopened or modified this phase**, per rule 17/18. |
| MITRE mapping accuracy | **(C)** | This lab's "attacks" come from a synthetic traffic generator (`LOGIN_ATTACK`, `BOT_ATTACK`, etc. — `MITRE_MAPPING.md`). Scoring CYUKTI's own resolved IDs against the generator's own labels, or against CYUKTI's own prior output, is circular — explicitly forbidden by this phase's rules. Wazuh's *native* rule mappings are vendor-provided and could arguably serve as a tiny independently-sourced reference, but that's a handful of rules, not a dataset. |
| Threat qualification (NOT_THREAT/SUSPICIOUS/QUALIFIED_THREAT) | **(C)** | No analyst has manually reviewed and labeled a sample of real incidents in this environment, independent of CYUKTI's own CTI-confidence scoring. |
| Campaign/operation correlation | **(C)** | No independently-labeled "these alerts truly belong to the same real-world campaign" dataset exists — the campaigns ARE CYUKTI's own correlation output. |
| Campaign selection (best-match choice) | **(C)** | Same reasoning — no independent record of which historical campaign was "actually" the right match for a given incident. |
| Playbook recommendation effectiveness | **(C)**, pending **(B)** | The SOAR layer tracks real `success`/`failed` outcomes once playbooks actually run (a legitimate, if narrow, proxy signal once it exists) — but zero real Shuffle executions have occurred in this environment yet (`SHUFFLE_INFRASTRUCTURE_BLOCKED`), so there is no data to compute a rate from, real or fabricated. |

**No numbers are reported for the (C) rows in this environment.** Reporting them would mean either fabricating labels or measuring CYUKTI against itself.

## The measurement infrastructure (`evaluation_metrics.py`)

Ground-truth-agnostic, unit-tested, ready to use the moment real labels exist. Maps directly to Task 6's requested metric families:

| Requested for | Functions |
|---|---|
| MITRE mapping (accuracy/precision/recall/macro F1/weighted F1/confusion matrix) | `precision_recall_f1`, `classification_report` (accuracy, macro/weighted/micro F1, balanced accuracy), `confusion_matrix` (full N×N, fixed label order), `multilabel_exact_match_ratio` (multi-technique alerts) |
| Threat qualification (precision/recall/F1/FPR/FNR) | `precision_recall_f1`, `false_positive_negative_rates` |
| Campaign correlation (pairwise P/R/F1, cluster purity) | `pairwise_precision_recall_f1`, `cluster_purity`, `campaign_fragmentation` (over-merging vs. over-fragmentation, reported separately, never blended) |
| Severity (accuracy/balanced accuracy/macro F1/weighted F1/per-class) | `classification_report` (task-agnostic — reused as-is, not re-implemented per task) |
| GNN retrieval (MRR/Recall@K) | `recall_at_k`, `mean_reciprocal_rank` |
| Playbook recommendation (Recall@K/Precision@K) | `recall_at_k`, `precision_at_k` |

21 tests (`test_evaluation_metrics.py`) verify each function against hand-computed values (perfect prediction, known error patterns, over-merging vs. over-fragmenting clusters, empty/no-match edge cases) — every function is correctness-tested even though no CYUKTI-specific (C)-row dataset is run through them in this environment.

## What would be needed to fill the (C) rows in

For each: a human (or another independent, non-CYUKTI-derived source) reviewing a sample of real alerts/campaigns and recording what they actually were, kept completely separate from anything CYUKTI itself computed. Given this lab's synthetic traffic generator, the most defensible near-term option is manual review of a sample of real Wazuh alerts once the corrected rule mappings (`MITRE_MAPPING.md`) have been live for a period, cross-checked by a second reviewer.

## Accuracy dashboard

Not built as a separate section: a dashboard showing metrics with no real data behind them would itself be a form of fabrication. Once any (C) row gets real labeled samples, `evaluation_metrics.py`'s output is already JSON-serializable dict data, trivially renderable following the same pattern as `SystemHealthPage.tsx`/`ThreatIntelligencePage.tsx`.
