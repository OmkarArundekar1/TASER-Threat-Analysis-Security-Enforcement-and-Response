# CYUKTI — Final Evaluation Report

> 2026-09-26. This report documents the outcome of the independent-ground-truth review lifecycle task. **Headline finding: no dataset reached LOCKED status this session, because no independent human reviewer was available.** Every number in this report remains `MEASURED_PRELIMINARY`, exactly as it was before this task started. What changed is the review infrastructure (now complete and tested for all 4 preliminary categories, not just 2) and a clean anti-circularity audit. This is reported plainly rather than dressed up as more progress than it is.

## 1. Evaluation architecture

Unchanged from `review/evaluation_implementation_report.md` — this task explicitly reused, not rebuilt, `evaluation/`. Added: `export_mitre_queue()`, `export_attribution_queue()`, `export_threat_qualification_queue()`, `export_campaign_correlation_queue()` in `evaluation/review/export_review_queue.py`, and matching `import_mitre_reviews()`, `import_attribution_reviews()`, `import_threat_qualification_reviews()`, `import_campaign_correlation_reviews()` in `import_reviewed.py`. `evaluation/review/__init__.py` was added (it was missing, which would have made the module unimportable as a package).

## 2. Ground-truth methodology

Unchanged: AUTO_PROPOSED labels built independently of CYUKTI's own resolver/engine code (raw alert text, MITRE ATT&CK definitions, known lab IP topology). No new ground-truth generation logic was added this task — only the human-facing review surface around the existing datasets.

## 3. Reviewer methodology

**No human reviewer participated in this session.** Per this task's own explicit rule ("You MUST NOT... fabricate a reviewer identity... claim a human reviewed records when nobody did"), this session did not fill in any review queue, did not assign itself as a reviewer, and did not promote any record to `HUMAN_REVIEWED`. Four real, complete, evidence-rich review queues now exist (`evaluation/review/{mitre,attribution,threat_qualification,campaign_correlation}_review_queue_*.csv`), each showing raw evidence, the AI-proposed label, and CYUKTI's own live prediction in a clearly separate column — ready for an actual reviewer, but untouched.

## 4. Dataset versions

`mitre_mapping_v0`, `attribution_v0`, `campaign_correlation_v0`, `threat_qualification_v0` — all still 100% `AUTO_PROPOSED`. `store.load_records(name, ReviewStatus.HUMAN_REVIEWED)` returns `[]` for all four (verified: `test_review_import_export.py`'s fixtures exercise exactly this path). No `LOCKED` dataset exists anywhere in the repository.

## 5. MITRE mapping results

**MEASURED_PRELIMINARY only** (unchanged from the prior task): P=1.00, R=0.667, F1=0.80, n=100. Review queue now exists (100 rows, ACCEPT/MODIFY/REJECT/UNCERTAIN + corrected-techniques column) — `mitre_review_queue_mitre_mapping_v0.csv`.

## 6. Attribution results

**MEASURED_PRELIMINARY only**: accuracy 0.744 [0.598, 0.851] 95% CI, n=43. Review queue now exists (46 rows — note: 3 more than the 43 scored, since 3 campaigns had no observed techniques and were skipped by the evaluator but still have a real attacker_ip to review).

## 7. Campaign correlation results

**MEASURED_PRELIMINARY only**: pairwise P/R/F1 = 0/0/0, purity 1.0, ARI 0.0, n=46 real campaigns / 3 sessions. The "25 Campaign nodes for one real session" finding **remains true** — nothing was changed to improve it, per this task's explicit instruction. A representative pairwise review queue was exported (18 sampled pairs: within-session candidates for "should have merged," cross-session candidates for "correctly separate," fixed random seed for reproducibility) rather than the full ~1,035 possible pairs across 46 items, which would not be a reasonable human review burden — this sampling tradeoff is disclosed here and in the CSV's own header comments, not presented as exhaustive.

## 8. Threat qualification results

**MEASURED_PRELIMINARY only**: accuracy 0.083, n=24. This remains the highest-priority item for actual human review — the review queue (`threat_qualification_review_queue_threat_qualification_v0.csv`) shows the raw `cti_score` and CYUKTI's derived classification in clearly-labeled reference columns, explicitly separate from the fillable ground-truth column, so a reviewer isn't anchored on either number while making an independent call.

## 9. Prediction results

Re-inspected per Phase 9's explicit instruction not to reuse an old number to look complete: the live NEXT_TECHNIQUE graph is still **3 edges** (`prediction_eval.py`'s live query, re-run this session) — unchanged from the prior task and from Phase 18. Status remains **UNMEASURABLE for new data**; the historical Phase 18 result (4/12=33.3%) is retained only as clearly-labeled historical context, never presented as a fresh measurement. No new prediction opportunities were manufactured.

## 10. RAG retrieval results

**NOT_MEASURED**, unchanged. Per Phase 7's explicit instruction ("do NOT regenerate them unnecessarily"), the existing `rag_query_queue_*.csv` files (10 candidate queries, exported in the prior task) were left as-is — still empty of any relevance judgment.

## 11. Investigation confidence results

**NOT_MEASURED**, unchanged. `investigation_verdict_queue.csv` (the 3 real Phase 21 investigations, exported in the prior task) left as-is per the same "don't regenerate unnecessarily" instruction — still empty of any verdict.

## 12. Preliminary vs. final comparison

| Category | Preliminary (this + prior task) | Final (locked, human-reviewed) |
|---|---|---|
| MITRE mapping | P=1.00/R=0.667/F1=0.80, n=100 | **Does not exist** — no LOCKED dataset |
| Attribution | 74.4% [59.8,85.1], n=43 | **Does not exist** |
| Campaign correlation | pairwise F1=0.0, ARI=0.0, n=46 | **Does not exist** |
| Threat qualification | 8.3%, n=24 | **Does not exist** |
| Prediction | UNMEASURABLE (historical 33.3% reused) | **Does not exist** |
| RAG (3 sources) | NOT_MEASURED | **Does not exist** |
| Investigation | NOT_MEASURED | **Does not exist** |

Every "Final" cell is genuinely empty, not populated with the preliminary number relabeled. This table exists specifically so nobody mistakes the preliminary column for the final one later.

## 13. Confidence intervals

Unchanged from the prior task: attribution's 95% Wilson CI [0.598, 0.851] is the only interval currently computed and reported (via `evaluation_metrics.wilson_confidence_interval`). MITRE mapping, campaign correlation, and threat qualification do not yet have reported intervals in the evaluator output — a real gap, noted here rather than silently left implicit. (Not fixed this task, since it touches evaluator code the task said not to rebuild; flagged as a candidate follow-up.)

## 14. Failure/disagreement analysis

The threat-qualification 8.3% disagreement was inspected this session (not just re-quoted): all 20 real campaigns from the SSH-brute-force session boundary that this project's AI-assisted scenario judgment labeled `SUSPICIOUS` were classified `QUALIFIED_THREAT` by CYUKTI's live `cti_score` (all scoring 65.81, above `PUBLISH_THRESHOLD=40`). Two campaigns from the mixed-exploitation session landed on each side. Per this task's explicit instruction not to invent unsupported explanations, only what the reviewed evidence actually supports is stated: **the disagreement is systematic (all 20, not scattered) and score-driven (cti_score is a fixed 65.81 for that entire session in the data inspected), not scattered noise** — consistent with either (a) CYUKTI's CTI scoring being more aggressive than this AI-assisted per-scenario judgment expected, or (b) the per-scenario ground truth being too conservative for a brute-force pattern that, once qualified as a real repeated-attempt session, may legitimately warrant `QUALIFIED_THREAT`. Distinguishing (a) from (b) requires an actual human security analyst's judgment on the raw evidence — that is exactly what `threat_qualification_review_queue_threat_qualification_v0.csv` is for, and this report does not guess further than the data supports.

## 15. Remaining NOT_MEASURED metrics

RAG retrieval (all 3 sources) and investigation confidence — both blocked purely on human review, not on missing code (Section 10/11).

## 16. Environment blockers

1. **No independent human reviewer was available this session** — the single blocker preventing every dataset from reaching LOCKED. This is not a technical limitation; the review artifacts are complete and ready.
2. Neo4j (`neo4j-soc`) was up and stable throughout this task (no restart needed this time).
3. No new live attack execution was performed or attempted (none was needed — this task only builds review tooling, per its own explicit scope).

## 17. Test results

**762/762 backend tests passing** (was 757 at the start of this task; +5 new tests for the review import/export machinery, `test_review_import_export.py`). Frontend tests not re-run (no frontend files touched). New edge cases covered: empty review CSV (promotes nothing), missing review CSV (raises, does not silently proceed), UNCERTAIN/REJECT rows (correctly skipped, not force-labeled), MODIFY rows (correctly override the technique list), pairwise UNCERTAIN labels (correctly excluded from the imported pairwise file).

## 18. Reproducibility

`python evaluation/run_all.py` was re-run this session and produced **byte-identical metric values** to the prior task's run (same n, same precision/recall/F1/accuracy/CI figures for every category) — confirming the pipeline is deterministic given unchanged ground truth and an unchanged live Neo4j state. Only the manifest's timestamp/duration/commit-hash fields differ between runs, as expected.

## 19. Paper implications

**None of the four preliminary numbers may be cited in the paper as final, human-validated results** — this was true before this task and remains true after it. What this task changes for the paper: the path from "preliminary AI number" to "citable result" is now fully built and tested for all four categories (not just two), and the single highest-priority item for the actual human reviewer (the guide, or the student) is unambiguous: `evaluation/review/threat_qualification_review_queue_threat_qualification_v0.csv`, because an 8.3% agreement rate is either a real, important finding about CYUKTI's threat-qualification aggressiveness or an artifact of overly-conservative scenario-level ground truth — and only a human reading the raw evidence can tell which.

## 20. Remaining human actions

1. Fill in `evaluation/review/mitre_review_queue_mitre_mapping_v0.csv` (100 rows) → `python evaluation/review/import_reviewed.py --which mitre` (via the CLI, not yet wired to `import_reviewed.py`'s `__main__` block — see note below) → `ground_truth.store.lock_dataset("mitre_mapping_v0", reviewer=..., source_commit=...)`.
2. Same pattern for `attribution_review_queue_attribution_v0.csv`, `threat_qualification_review_queue_threat_qualification_v0.csv`, `campaign_correlation_review_queue_campaign_correlation_v0.csv`.
3. Fill in the RAG and investigation queues (unchanged asks from the prior task).
4. `import_reviewed.py --which {mitre,attribution,threat_qualification,campaign_correlation}` was verified end-to-end via the actual CLI this session (not just via test mocks) against the real, still-unfilled queues — each correctly reports "0 promoted, all skipped" right now, and will promote real rows the moment a human fills them in. A real fix was needed and applied during this verification: the script was missing a `sys.path` entry for the `evaluation/`/`backend/` packages, which would have made every one of these four new import functions crash with `ModuleNotFoundError` on first real use — caught and fixed before being reported as done.
5. Once any dataset is `HUMAN_REVIEWED`, re-run `python evaluation/run_all.py` — the evaluators already prefer `LOCKED` > `HUMAN_REVIEWED` > `AUTO_PROPOSED` automatically (no code change needed) and will report `MEASURED` instead of `MEASURED_PRELIMINARY` the moment real reviewed data exists.
