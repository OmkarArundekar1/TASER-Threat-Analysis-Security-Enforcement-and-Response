# CYUKTI — Paper Metrics Source of Truth (Evaluation Framework)

> Last verified: 2026-09-26. This file is the authoritative status list for the 7 metric categories the evaluation audit identified as missing. It supersedes the "0/0" and "NOT MEASURED" placeholders in `review/paper_submission_status.md` for these specific rows — see that file for the rest of the paper's numbers, which this task did not touch. Categories: **MEASURED**, **MEASURED_ASSISTANT_ADJUDICATED** (an AI assistant's own independent re-examination of real evidence — more granular than AUTO_PROPOSED, but still explicitly not human-reviewed), **MEASURED_PRELIMINARY**, **NOT MEASURED**, **UNMEASURABLE**, **BLOCKED_BY_ENVIRONMENT**. Never silently converted between them.

| Category | Status | Value | n | Notes |
|---|---|---|---|---|
| MITRE mapping accuracy | **MEASURED_PRELIMINARY** | P=1.00, R=0.667, F1=0.80 | 100 | AUTO_PROPOSED ground truth (AI-assisted, not human-reviewed). Real disagreement found: CYUKTI under-tags the full multi-technique set for SSH brute-force alerts. |
| Threat qualification accuracy | **MEASURED_PRELIMINARY** (original), **MEASURED_ASSISTANT_ADJUDICATED** (re-examined) | Original: Accuracy=0.083 (n=24, per-scenario AUTO_PROPOSED label). Re-examined per-campaign: Accuracy=0.292 (7/24), but the load-bearing finding is 100% recall / 0% precision on QUALIFIED_THREAT (CYUKTI never misses an adjudicated real threat, but calls every weaker case QUALIFIED_THREAT too). | 24 | Full row-by-row re-examination with real Neo4j evidence + MITRE ATT&CK/Atomic Red Team reference material: `review/evaluation_threat_qualification_adjudication.md`. Still not human-reviewed — both numbers require an independent human analyst before either may be cited as final. |
| Campaign correlation (pairwise P/R/F1, ARI/AMI) | **MEASURED_PRELIMINARY** | Pairwise F1=0.0, ARI=0.0, purity=1.0 | 46 | AUTO_PROPOSED. Real finding: severe campaign fragmentation (one real session split across up to 25 CYUKTI Campaign nodes). |
| Attribution accuracy | **MEASURED_PRELIMINARY** | Accuracy=0.744 [0.598, 0.851] (95% Wilson CI) | 43 | AUTO_PROPOSED. Coarse IP-based proxy for identity, not a named-actor ground truth. |
| Prediction accuracy | **UNMEASURABLE** (for new data) | Historical: 4/12 = 33.3% (Phase 18, reused) | 12 (historical) | Live NEXT_TECHNIQUE graph unchanged in scale (3 edges) since Phase 18 — no new held-out opportunity exists to generate a fresh number from. |
| RAG retrieval precision/recall (3 sources) | **NOT MEASURED** | — | 0 | Evaluator built + unit-tested. No independently-judged query set exists. Review queue exported and ready (`evaluation/review/rag_query_queue_*.csv`, 10 candidate queries). |
| Investigation confidence P/R | **NOT MEASURED** | — | 0 | Evaluator built + unit-tested. No independent analyst verdict exists. Review queue exported and ready (`evaluation/review/investigation_verdict_queue.csv`, the 3 real Phase 21 investigations). |

## Why nothing here is `MEASURED` (only `MEASURED_PRELIMINARY`)

No independent human security analyst reviewed any ground-truth label in this pass — this session (an AI assistant) built labels from raw alert text, MITRE ATT&CK's own published definitions, and known lab IP topology, which is independent of CYUKTI's own code but not independent of this AI's judgment. `evaluation/ground_truth/store.py::lock_dataset()` refuses to promote a dataset to `LOCKED` unless it is first `HUMAN_REVIEWED`, and no dataset in this project has reached that state yet. **To move any row above from MEASURED_PRELIMINARY to MEASURED**, a human reviewer needs to:
1. Open the relevant dataset under `evaluation/ground_truth/provisional/*.jsonl` and either confirm or correct each label.
2. Re-save it with `review_status=HUMAN_REVIEWED` and a real `reviewer` name (`ground_truth/store.py::save_records`).
3. Run `evaluation/ground_truth/store.py::lock_dataset()` to freeze and hash it.
4. Re-run `python evaluation/run_all.py` — the evaluators automatically prefer `LOCKED` > `HUMAN_REVIEWED` > `AUTO_PROPOSED` data and will report `MEASURED` once reviewed data exists.

## What this changes in `ACCURACY_EVALUATION.md`

That document's (A)/(B)/(C) classification predates this task and correctly stated "(C) — no independent ground truth exists" for MITRE mapping, threat qualification, campaign/operation correlation, and campaign selection. This task adds a genuine, if preliminary, independent measurement for four of those — `ACCURACY_EVALUATION.md` has been updated with a pointer to this file rather than rewritten, since the underlying caveat (AI-assisted, not human-reviewed) means the honest classification is still not a clean "(A) real independent ground truth" — it is a new, disclosed middle category this task introduces: **(B-preliminary)**.
