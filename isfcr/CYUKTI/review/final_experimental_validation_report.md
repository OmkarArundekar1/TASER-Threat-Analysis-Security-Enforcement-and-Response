# CYUKTI — Final Experimental Validation Report

> 2026-09-26. **Headline finding, stated up front rather than buried: this phase's primary objective — human ground-truth review — could not be performed.** No human reviewer participated in this session. Every review queue was checked directly (`evaluation/review/*.csv`, 8 files, 201 total rows) and confirmed to have **zero rows with a non-empty reviewer field**. Per this project's own standing rule ("You MUST NOT... claim a human reviewed records when nobody did"), this report does not fabricate a reviewer, does not promote any dataset, and does not report any "final" metric. What follows is an honest account of what *was* legitimately re-verified this phase, and an explicit statement of what remains exactly where it was.

## A. Executive summary

Re-verified (not re-derived): the anti-circularity leakage audit (clean, unchanged), live prediction status (still `UNMEASURABLE`, NEXT_TECHNIQUE graph still 3 edges), active-response lab availability (confirmed `BLOCKED_BY_ENVIRONMENT` by direct ping test this phase, not assumed from memory), and the complete regression suite (backend 876/876, frontend 113/113, build clean — all unchanged, since no source code required changing). Produced one new artifact, `evaluation/results/final_paper_dataset.json`, packaging the existing real preliminary numbers with explicit provenance in one place — it changes no number and promotes nothing. **All four evaluation categories remain exactly `MEASURED_PRELIMINARY`. Prediction remains `UNMEASURABLE`. Active response remains `BLOCKED_BY_ENVIRONMENT`.**

## B. Human review status

**0 reviewers. 0 reviewed rows out of 201 total rows across 8 queue files**, confirmed by direct Python inspection of every CSV this phase (not assumed from prior context):

| Queue | Rows | Reviewed |
|---|---|---|
| `mitre_review_queue_mitre_mapping_v0.csv` | 100 | 0 |
| `attribution_review_queue_attribution_v0.csv` | 46 | 0 |
| `threat_qualification_review_queue_threat_qualification_v0.csv` | 24 | 0 |
| `campaign_correlation_review_queue_campaign_correlation_v0.csv` | 18 | 0 |
| `investigation_verdict_queue.csv` | 3 | 0 |
| `rag_query_queue_mitre_semantic.csv` | 5 | 0 |
| `rag_query_queue_campaign_narrative.csv` | 3 | 0 |
| `rag_query_queue_gnn_topology.csv` | 2 | 0 |

All 8 files' modification timestamps (05:17–05:41 on 2026-09-26) match exactly when this session generated them in the prior evaluation-review phase — nothing has changed since.

## C. Locked dataset status

**No dataset was locked. None was eligible.** `ground_truth.store.lock_dataset()` refuses (raises) unless a dataset is already `HUMAN_REVIEWED` — this precondition was not attempted to be worked around. Every dataset in `evaluation/ground_truth/provisional/` remains exactly `AUTO_PROPOSED`. Per Section 2's own instruction ("If review coverage is insufficient: HUMAN_REVIEW_REQUIRED"): **all four datasets are `HUMAN_REVIEW_REQUIRED`.**

## D. Leakage audit

Re-run this phase (not skipped): grepped `evaluation/ground_truth/builder.py` and `evaluation/build_campaign_dataset.py` for every `expected_*` field assignment — all nine trace to `scenario.*` (the hand-authored `AttackScenario` registry) or an independently-constructed session key, none to a CYUKTI engine output. Grepped `ground_truth/` and `scenarios/` for any import of `mitre_resolver`, `campaign_manager`, `threat_attribution_engine`, `threat_qualification`, or `prediction_engine` — zero matches. **Result: clean, unchanged from the prior audit.**

## E. MITRE results

**No change — still `MEASURED_PRELIMINARY`, n=100, P=1.00/R=0.667/F1=0.80.** Section 3's request ("using ONLY human-reviewed/locked ground truth") cannot be executed: zero human-reviewed MITRE records exist. No "PRELIMINARY vs HUMAN_REVIEWED" comparison exists because the HUMAN_REVIEWED side is empty, not because it was skipped.

## F. Attribution results

**No change — still `MEASURED_PRELIMINARY`, n=43, accuracy=0.744 [0.598, 0.851] 95% Wilson CI.** Same reason as Section E.

## G. Campaign-correlation results

**No change — still `MEASURED_PRELIMINARY`, n=46, pairwise F1=0.0, purity=1.0, ARI=0.0, AMI≈0.** The fragmentation finding (one real session split across up to 25 Campaign nodes) was re-inspected, not re-derived: still true, and **not manually merged or altered in any way to improve the metric**, per explicit instruction.

## H. Threat-qualification results

**No change — still `MEASURED_PRELIMINARY`, n=24, accuracy=0.083.** Section 6 explicitly asked this phase to determine, via human review, whether the 8.3% figure or the underlying AI-generated scenario labels are correct. **That determination cannot be made without a human reviewer**, and none was available. The SSH-brute-force disagreement (20/20 scenario-labeled-SUSPICIOUS campaigns scored QUALIFIED_THREAT by CYUKTI's live `cti_score`) remains exactly as documented in the prior evaluation report — preserved, not resolved, not spun toward either "CYUKTI is right" or "the ground truth is wrong."

## I. Prediction status

**Re-checked live this phase.** `prediction_eval.evaluate()` re-run against the current Neo4j graph: **still 3 `NEXT_TECHNIQUE` edges**, identical to Phase 18 and to the immediately preceding phase's check. Status: **`UNMEASURABLE`**. Per Section 7's explicit instruction ("Do not reuse an earlier number"), this report does **not** restate the historical Phase 18 figure (4/12, 33.3%) as any part of this phase's result — it is mentioned here only as the reason the graph's lack of growth is unsurprising, not as a metric this phase produced.

## J. Active-response validation

**Re-checked this phase with a real test, not an assumption**: `ping -c 1 192.168.56.106` and `192.168.56.105` — both 0/1 packets received (0% reachable) from this environment. No `VBoxManage`/`vboxmanage` tooling is present. **`ACTIVE_RESPONSE = BLOCKED_BY_ENVIRONMENT`, confirmed, not simulated.** No attack was run, no containment was executed, no verification was attempted against any real system.

## K. Response latency

**`NOT_MEASURED`** — zero real containment events exist (Section J), so `active_response.metrics`'s functions (unchanged, real, tested against synthetic timestamps only) have nothing real to compute from. No number is reported.

## L. Regression results

| Suite | Result | Change from Phase Z |
|---|---|---|
| Backend (`pytest`) | **876/876 passed** | 0 (unchanged — no backend source was modified this phase) |
| Frontend (`vitest`) | **113/113 passed** | 0 (unchanged) |
| Frontend production build (`npm run build`) | **Clean** | 0 (unchanged) |
| Evaluation tests (subset of backend total) | Passing, +5 new (`test_final_paper_dataset.py`) | +5 |
| Leakage audit | Clean | 0 (re-verified, Section D) |
| Security tests (`test_safety_audit.py`, `test_concurrency.py`, subset of backend total) | Passing | 0 (unchanged, re-run) |

**Total backend: 881/881** (876 + 5 new for the packaged artifact's own tests).

## M. Security findings

No new code was written this phase except the packaging artifact and its tests (both pure data/read-only, no new attack surface). Phase Z's findings (race condition found and fixed; replay-attack surface and missing RBAC disclosed, not fixed) stand unchanged and are not re-litigated here.

## N. Research-claim matrix

| Claim | Status |
|---|---|
| Real-time threat detection | SUPPORTED (live-verified, pre-existing) |
| Deduplication | SUPPORTED (live-verified, pre-existing) |
| Campaign correlation | MEASURED_PRELIMINARY (n=46, severe fragmentation found) |
| Graph-based threat intelligence | SUPPORTED (structural + live graph evidence, pre-existing) |
| GNN contribution | IMPLEMENTED_NOT_VALIDATED (frozen null-benefit result) |
| Multi-RAG retrieval | NOT_MEASURED |
| Evidence-aware investigation | IMPLEMENTED_NOT_VALIDATED (disclosed negative result on adaptivity, frozen) |
| Next-best-evidence | IMPLEMENTED_NOT_VALIDATED (same, frozen) |
| Threat qualification | MEASURED_PRELIMINARY (n=24, accuracy=0.083, unresolved disagreement) |
| Attribution | MEASURED_PRELIMINARY (n=43, accuracy=0.744) |
| Active containment | IMPLEMENTED_NOT_VALIDATED, BLOCKED_BY_ENVIRONMENT for live verification |
| Containment verification | IMPLEMENTED_NOT_VALIDATED (structurally evidence-gated; never live-tested against a real connection) |
| Response memory | IMPLEMENTED_NOT_VALIDATED |
| Closed-loop response | IMPLEMENTED_NOT_VALIDATED, BLOCKED_BY_ENVIRONMENT |

No claim above is marked `SUPPORTED` on preliminary or unvalidated evidence; no claim is upgraded from its Phase Z status.

## O. Final metrics table

| Metric | Definition | Dataset | N | Ground Truth | Value | 95% CI | Status | Evidence Source |
|---|---|---|---|---|---|---|---|---|
| MITRE mapping | Multi-label P/R/F1 | `mitre_mapping_v0` | 100 | AUTO_PROPOSED | P=1.00, R=0.667, F1=0.80 | — | MEASURED_PRELIMINARY | `evaluation/results/mitre_mapping_metrics.json` |
| Attribution | Top-candidate identity match | `attribution_v0` | 43 | AUTO_PROPOSED | 0.744 | [0.598, 0.851] | MEASURED_PRELIMINARY | `evaluation/results/attribution_metrics.json` |
| Campaign correlation | Pairwise + clustering agreement | `campaign_correlation_v0` | 46 | AUTO_PROPOSED | F1=0.0, ARI=0.0, purity=1.0 | — | MEASURED_PRELIMINARY | `evaluation/results/campaign_correlation_metrics.json` |
| Threat qualification | Classification agreement | `threat_qualification_v0` | 24 | AUTO_PROPOSED | 0.083 | — | MEASURED_PRELIMINARY | `evaluation/results/threat_qualification_metrics.json` |
| Prediction | Top-1 next-technique accuracy | live NEXT_TECHNIQUE graph | 0 new | — | — | — | UNMEASURABLE | `evaluation/results/prediction_metrics.json` |
| RAG (3 sources) | P@K/R@K/MRR/nDCG@K | 10 candidate queries | 0 | none recorded | — | — | NOT_MEASURED | `evaluation/results/rag_*_metrics.json` |
| Investigation confidence | Verdict agreement | 3 real investigations | 0 | none recorded | — | — | NOT_MEASURED | `evaluation/results/investigation_metrics.json` |
| Active response (all) | Containment/verification/latency | — | 0 | — | — | — | BLOCKED_BY_ENVIRONMENT | `review/phaseZ_final_integration_audit.md` |

Machine-readable version: `evaluation/results/final_paper_dataset.json` (this table, plus per-metric findings/limitations, in JSON).

## P. Known limitations

Unchanged from Phase X/Z, restated for completeness: no human reviewer available in this environment; no lab VM access; replay-attack surface and no RBAC in `active_response/`; correlation-ID not wired into live ingestion; Shuffle payload lacks correlation_id; no dedicated containment-effectiveness aggregate. **New this phase**: confirmed (not assumed) that the lab VMs are network-unreachable from this environment (0% ping success), closing any ambiguity about whether "no shell access" might still have allowed a network-level workaround.

## Q. Paper-readiness assessment

The paper (per `review/paper_submission_status.md`, updated this phase with one line on active-response status) already correctly labels the four evaluation categories `PARTIALLY MEASURED (preliminary)`, correctly states prediction as unmeasurable, and correctly discloses Shuffle/MISP as non-live. **No change was needed to any number** in that file this phase — it did not overclaim before this phase and does not need correction now. The one addition: an explicit note that active containment, if described in the paper at all, must be framed as implemented-and-tested, not demonstrated. **The paper is not ready to claim**: human-validated accuracy for any of the four preliminary categories, a measured prediction accuracy, live containment, or production-hardened security posture (replay/RBAC gaps).

## R. Exact remaining blockers

1. **A real, independent human reviewer** to fill in the 8 CSV queues in `evaluation/review/` — this is the single blocker preventing every "final" (as opposed to preliminary) evaluation number in this project.
2. **Real Kali/Ubuntu lab access** (VM console or SSH) — the single blocker preventing any live active-response validation.
3. Everything else audited this phase (leakage, regression, security posture, research claims, documentation) was found consistent and required no correction.

---

### Final deliverable (Section 17 of the task brief)

1. Human-reviewed sample counts: **0 / 201** across all 8 queues.
2. Locked dataset versions: **none exist**.
3–6. Final MITRE/attribution/campaign-correlation/threat-qualification metrics: **unchanged from `MEASURED_PRELIMINARY`** — see Section O.
7. Prediction status: **UNMEASURABLE** (3 live NEXT_TECHNIQUE edges, unchanged).
8. Active-response validation status: **BLOCKED_BY_ENVIRONMENT**, confirmed via a real, failed ping test this phase.
9. Test counts: **backend 881/881** (876 + 5 new), **frontend 113/113**, build clean.
10. Security findings: none new this phase (Phase Z's fix and disclosures stand).
11. Remaining environment blockers: human reviewer, lab VM access (Section R).
12. Research-claim status: no claim upgraded; see Section N.
13. Paper-readiness status: ready to submit **with its current, accurate preliminary/limitation framing** — not ready to claim final, human-validated, or live-demonstrated results anywhere.

**CYUKTI is not fully validated and is not production-ready.** It is a research prototype with a real, tested, internally-consistent detection-through-response architecture, four honestly-labeled preliminary evaluation results, and a clearly-scoped, clearly-blocked path to final validation that requires resources (a human reviewer, lab access) outside this session's reach — not more engineering.
