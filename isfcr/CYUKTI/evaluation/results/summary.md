# CYUKTI Evaluation Run Summary

Run at 2026-09-26T05:47:53.661631+00:00 | commit `f8bdf07e10d0` | Neo4j 5.26.27 | duration 2.5s

| Task | Status | n | Headline metric |
|---|---|---|---|
| mitre_mapping | MEASURED_PRELIMINARY | 100 | 0.800 |
| prediction | UNMEASURABLE | 12 | - |
| attribution | MEASURED_PRELIMINARY | 43 | 0.744 |
| campaign_correlation | MEASURED_PRELIMINARY | 46 | 0.000 |
| threat_qualification | MEASURED_PRELIMINARY | 24 | 0.083 |
| rag_mitre_semantic | NOT_MEASURED | 0 | - |
| rag_campaign_narrative | NOT_MEASURED | 0 | - |
| rag_gnn_topology | NOT_MEASURED | 0 | - |
| investigation | NOT_MEASURED | 0 | - |

*Planned:* the `prediction` row's `UNMEASURABLE` status is expected to
change once the Phase 19 dataset expansion accumulates enough real
technique transitions to revisit the `INSUFFICIENT_FOR_SUPERVISED_ML`
verdict.

## Skipped / not measured

- rag_mitre_semantic: No independently-judged query set exists at evaluation/queries/rag_queries_mitre_semantic.json. Building one requires a human reviewer to judge relevant documents for a real incident context per query (see evaluation/review/export_review_queue.py for the mechanism to produce this) -- fabricating relevance judgments here would defeat the purpose of an independent evaluation, so this is honestly reported as not yet measured rather than guessed.
  *Planned:* construct that labeled query set via `evaluation/review/export_review_queue.py` to measure retrieval precision/recall directly.
- rag_campaign_narrative: No independently-judged query set exists at evaluation/queries/rag_queries_campaign_narrative.json. Building one requires a human reviewer to judge relevant documents for a real incident context per query (see evaluation/review/export_review_queue.py for the mechanism to produce this) -- fabricating relevance judgments here would defeat the purpose of an independent evaluation, so this is honestly reported as not yet measured rather than guessed.
  *Planned:* construct that labeled query set via `evaluation/review/export_review_queue.py` to measure retrieval precision/recall directly.
- rag_gnn_topology: No independently-judged query set exists at evaluation/queries/rag_queries_gnn_topology.json. Building one requires a human reviewer to judge relevant documents for a real incident context per query (see evaluation/review/export_review_queue.py for the mechanism to produce this) -- fabricating relevance judgments here would defeat the purpose of an independent evaluation, so this is honestly reported as not yet measured rather than guessed.
  *Planned:* construct that labeled query set via `evaluation/review/export_review_queue.py` to measure retrieval precision/recall directly.
- investigation: No independent analyst verdicts exist at evaluation/labels/investigation_verdicts.json. CYUKTI's investigation loop has been run against real campaigns (Phase 21/22, review/phase21_real_investigation_validation.md), but no human has independently judged whether the resulting top hypothesis was actually correct for any of those runs. Building this requires a reviewer to read the real evidence trail and judge it -- see evaluation/review/export_review_queue.py.
  *Planned:* have a reviewer produce those independent verdicts via `evaluation/review/export_review_queue.py` in a future evaluation session.
