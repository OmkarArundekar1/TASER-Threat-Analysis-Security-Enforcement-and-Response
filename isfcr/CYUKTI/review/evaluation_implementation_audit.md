# CYUKTI — Evaluation Implementation Audit (Phase 0)

> Last verified: 2026-09-26. This is the mandatory pre-implementation audit for the independent ground-truth / evaluation framework task. Nothing below is claimed as "implemented" unless it already existed and was verified by reading the actual file; everything proposed is marked as proposed, not built.

## 1. Existing capabilities found in the repository

| Component | File(s) | What it actually does | Reusable for this task? |
|---|---|---|---|
| Generic metric functions | `backend/evaluation_metrics.py` (223 lines) | `precision_recall_f1`, `confusion_matrix`, `classification_report` (accuracy/macro-F1/micro-F1/weighted-F1/balanced accuracy), `false_positive_negative_rates`, `multilabel_exact_match_ratio`, `pairwise_precision_recall_f1`, `cluster_purity`, `campaign_fragmentation`, `recall_at_k`, `precision_at_k`, `mean_reciprocal_rank`. Explicitly ground-truth-agnostic — takes `y_true`/`y_pred` from the caller, never computes its own labels. | **Yes — this is the metrics layer the spec asks for.** It will be extended (nDCG@K, Brier score, calibration error, Adjusted Rand Index/AMI, Wilson confidence intervals), not duplicated. |
| Metric correctness tests | `backend/tests/test_evaluation_metrics.py` (171 lines, 21 tests) | Hand-computed-value tests for every function above. | Yes, unchanged; new evaluators get their own new test files. |
| Per-scenario experiment recorder | `experiments/schema.json`, `experiments/record_experiment.py`, `experiments/README.md`, `experiments/examples/`, `experiments/records/` | A JSON-Schema-validated record of one real (or explicitly-labeled synthetic) scenario run end-to-end through CYUKTI — real alert, real MITRE resolution, real campaign, real threat-qualification/campaign-selection output, real errors. One record = one scenario. Already has a real, committed example (`EXP-2026-09-25-nmap-scan-01.json`, a real live Kali→Ubuntu Nmap scan). 6 tests (`test_experiment_recording.py`). | **Yes, directly** — this is the natural home for "attack scenario → what CYUKTI actually did." It is NOT a ground-truth store (it records CYUKTI's own output, not an independent label), but the evaluation framework's `runs/` layer should reuse this schema/mechanism rather than inventing a second one. |
| Accuracy/evaluation framework doc | `ACCURACY_EVALUATION.md` | The authoritative (A)/(B)/(C) ground-truth-availability classification the task's brief quotes from. Already states plainly that MITRE mapping, threat qualification, campaign/operation correlation, campaign selection, and playbook effectiveness are all (C) — "no independent ground truth exists" — precisely to avoid scoring CYUKTI against itself. | Yes — this document's classification is the baseline this task is meant to advance from (C) toward (A)/(B) wherever real independent labels can be produced. |
| MITRE resolver (system under test, NOT ground truth) | `backend/mitre_resolver.py`, `backend/mitre_rule_registry.py`, `backend/mitre_mapper.py` | 4-tier resolution (NATIVE_WAZUH > REVIEWED_RULE_MAPPING > DETERMINISTIC_INFERENCE > UNKNOWN). | System-under-test only. Its output (`resolution.technique_ids`, `.provenance`) is the `cyukti_*` side of the MITRE evaluation, never the `ground_truth_*` side. |
| Campaign manager / correlation (system under test) | `backend/campaign_manager.py`, `backend/campaign_decision_engine.py`, `backend/campaign_correlation_engine.py`, `backend/operation_manager.py` | Generates `campaign_id`s (`CAMP_xxxxxxxx`) via its own 7-feature weighted similarity scoring. | System-under-test only. `Campaign.campaign_id` and `Operation.operation_id` must never be used as ground-truth cluster labels — confirmed by re-reading `campaign_decision_engine.py`: cluster assignment is entirely CYUKTI's own similarity output, there is no independent grouping signal stored anywhere in the graph. |
| Attribution engine (system under test) | `backend/threat_attribution_engine.py`, `backend/attribution_similarity.py`, `backend/attribution_models.py` | Ranks candidate historical campaigns by `attacker_ip` match + technique/chain similarity; the "attacker identity" it outputs is itself (an `attacker_ip` string), not an independently verified identity. | System-under-test only. |
| Prediction engine (system under test) | `backend/prediction_engine.py`, `backend/chain_updater.py` | `predict_next_readonly()` reads learned `NEXT_TECHNIQUE` edges (frequency-counted real technique transitions, `MIN_TRANSITION_OBSERVATIONS` gated). No stored `prediction_hits`/`prediction_misses` counter exists as a graph property anywhere — the "0/0" figure quoted in the task brief was an ad hoc Cypher query run earlier this session (`OPTIONAL MATCH` over a property that doesn't exist on any live node), not a persisted tracking schema. Confirmed by grep: no `prediction_hit`/`prediction_miss`/`prediction_correct` property write exists in any backend `.py` file outside the frozen, offline `ml/label_generator.py` pipeline (which computes `prediction_correct` only inside the frozen 60-row `campaign_dataset.csv`, a completely different artifact from live NEXT_TECHNIQUE tracking). | The real, existing prediction target is **NEXT_TECHNIQUE edge-based `predict_next_readonly()`**, evaluated exactly as Phase 18 already did (retrospective correct-vs-actual on real `AttackEvent.first_seen` ordering) — this task must not invent a different prediction target. |
| Threat qualification (system under test) | `backend/threat_qualification.py` | Real enum: `NOT_THREAT`, `SUSPICIOUS`, `QUALIFIED_THREAT` (confirmed, lines 33-35). | System-under-test only. |
| RAG retrievers (system under test) | `backend/rag/mitre_retriever.py`, `backend/rag/campaign_retriever.py`, `backend/rag/gnn_topology_retriever.py` (if present) | Three independently-scored sources, per `journal_ready_data.md` Section 21/10.2 — never blended. | System-under-test only; a query set with independently-judged relevance must be built per source. |
| Investigation loop (system under test) | `backend/investigation/*.py` | Produces `InvestigationState`/`ConfidenceEstimate`; there is no single stored field called a "verdict" — the closest analogue is the final `investigation_confidence` plus whichever hypothesis in `candidate_hypotheses` has the highest weight at stop time. | The evaluator must define its comparison target as "the top `candidate_hypothesis` at investigation stop" — not invent a new concept. |
| Known, independent lab topology | `INCIDENT_VIEW.md`, `MITRE_MAPPING.md`, `journal_ready_data.md` | Real, documented, infrastructure-level facts independent of CYUKTI's own inference: Kali attacker = `192.168.56.106`; Ubuntu targets = `192.168.56.105` and `pes1ug23cs411-VirtualBox`; specific real Wazuh rule IDs/descriptions for specific real attack types (Nmap → rule 100500, SSH brute force → rule family 5503/5760/5715/100510/100511 per Section 4.9 fixes). | **Yes — this is the one genuinely independent-of-CYUKTI signal available in this environment**: which physical machine sent traffic, and what a raw Wazuh/Suricata rule description says, independent of what CYUKTI's resolver/campaign engine subsequently does with it. |
| Raw historical alert archive | `/var/ossec/logs/alerts/alerts.json` + `/var/ossec/logs/alerts/2026/<Month>/ossec-alerts-DD.json[.gz]` (found in the previous phase; ~463,638 uncompressed lines, current file 120 lines) | The full raw Wazuh alert history this project has ever produced, most of it non-attack operational noise. | Yes, as the raw source pool to draw a labeled MITRE/threat-qualification sample from — but requires care: most lines are not attack-relevant at all (dpkg, disk, agent-connect noise), so a "label every alert" approach would produce a mostly-trivial sample; the sample must be drawn deliberately from lines that plausibly represent attack-adjacent rules. |
| Atomic Red Team / automated attack execution | — | **Does not exist.** Confirmed by repo-wide search: no `atomic-red-team`, no attack-automation scripts anywhere in `scripts/`, `backend/`, or elsewhere. All historical attacks in this project (Nmap, SSH brute force, DoS, web attacks) were run manually against the live Kali/Ubuntu VMs, and the record of exactly what was run lives only in prose (`INCIDENT_VIEW.md`, `MITRE_MAPPING.md`, `journal_ready_data.md`), not in a machine-readable scenario registry. | No existing automation to reuse; Phase 2's scenario registry must be built from scratch, and any *new* attack execution requires the operator (VM access, Kali shell) — this session has shell access to the Wazuh-manager host, but has not been given, and did not assume, unattended control of the separate Kali/Ubuntu attacker/victim VMs. |
| Test infrastructure | `backend/tests/` (65 files, 689 tests), `frontend/tests`-equivalent (17 files, 109 tests) | Real, current (2026-09-25 re-verification, see `paper_submission_status.md`). | Yes — new evaluator tests follow the same `backend/tests/test_*.py` + `pytest` convention, no new test runner needed. |

## 2. Missing capabilities (confirmed absent, not merely undocumented)

- No `evaluation/` directory or package of any kind.
- No ground-truth schema, builder, or storage of any kind (provisional/reviewed/locked states do not exist).
- No attack scenario registry (machine-readable).
- No independent-labeling / analyst-review workflow (CSV, JSON, or UI).
- No MITRE-mapping-accuracy evaluator (against independent labels — `evaluation_metrics.py`'s `classification_report` exists but nothing calls it with independent MITRE ground truth).
- No prediction-accuracy evaluator wired to real NEXT_TECHNIQUE opportunities beyond the one-off Phase 18 script (`scripts/phase18_next_technique_diagnosis.py`, itself operating on the frozen 60-row dataset, not live data).
- No attribution-accuracy evaluator.
- No campaign-correlation pairwise/ARI/AMI evaluator wired to independent cluster labels.
- No RAG query set or per-source retrieval evaluator.
- No investigation-conclusion-correctness evaluator.
- No threat-qualification-accuracy evaluator.
- No experiment manifest capturing environment/versions/seeds for a *batch* evaluation run (the existing `experiments/` mechanism is per-scenario, not per-evaluation-run).
- No dataset-locking/hash-manifest mechanism.
- `Adjusted Rand Index`, `Adjusted Mutual Information`, `nDCG@K`, `Brier score`, `calibration error`, and binomial confidence intervals do not exist anywhere in `evaluation_metrics.py`.

## 3. Reusable components (confirmed, will be extended not replaced)

1. `backend/evaluation_metrics.py` — extend with clustering-agreement and calibration functions.
2. `experiments/schema.json` + `record_experiment.py` — reused as the per-scenario "what CYUKTI actually did" record inside `evaluation/runs/`.
3. `ACCURACY_EVALUATION.md`'s (A)/(B)/(C) classification — the new `evaluation/` framework's job is to move specific rows from (C) toward (A)/(B), and this doc gets updated (not replaced) once real measurements exist.
4. Known lab topology (attacker/victim IPs, rule-ID-to-technique facts already documented in `MITRE_MAPPING.md`/`INCIDENT_VIEW.md`) — the seed for independent ground-truth labeling.
5. `backend/tests/` conventions — new evaluator tests slot in directly.

## 4. Proposed evaluation architecture

```
evaluation/
  __init__.py
  ground_truth/
    schema.py            # dataclasses / JSON-schema for a GroundTruthRecord + ReviewState enum
    builder.py            # GroundTruthBuilder (Phase 3)
    store.py              # load/save provisional|reviewed|locked JSONL files, hashing for locking
    raw/                  # frozen raw inputs pulled from real Wazuh alerts / real Neo4j campaigns (read-only snapshots)
    provisional/          # AUTO_PROPOSED records (see honesty note below)
    reviewed/              # HUMAN_REVIEWED records (empty until a human actually reviews)
    locked/                 # LOCKED, hash-manifested datasets ready to cite in the paper
  scenarios/
    registry.py            # AttackScenario dataclass + JSON registry file
    scenarios.json          # the actual registered scenarios (existing historical ones, reconstructed honestly from docs)
  evaluators/
    mitre_eval.py
    prediction_eval.py
    attribution_eval.py
    campaign_correlation_eval.py
    rag_eval.py
    investigation_eval.py
    threat_qualification_eval.py
  review/
    export_review_queue.py   # writes a CSV/JSON queue for a human to fill in ground-truth columns
    import_reviewed.py       # reads a filled-in queue back, promotes PROVISIONAL -> HUMAN_REVIEWED
  manifest.py               # ExperimentManifest: git commit, versions, dataset hash, timestamp, seed
  run_all.py                 # python -m evaluation.run_all
  results/                    # generated: summary.json/.md, <task>_metrics.json per category
tests/
  (added to backend/tests/, matching existing convention)
    test_ground_truth_schema.py
    test_ground_truth_builder.py
    test_mitre_eval.py
    test_prediction_eval.py
    test_attribution_eval.py
    test_campaign_correlation_eval.py
    test_rag_eval.py
    test_investigation_eval.py
    test_threat_qualification_eval.py
    test_evaluation_manifest.py
```

Deviation from the task's suggested layout: `evaluation/` is placed at the CYUKTI repo root (sibling to `backend/`, `frontend/`, `review/`), not inside `backend/`, because it evaluates both backend and (potentially) frontend-observable behavior and must import `backend/*` modules explicitly rather than being a subpackage of it — this mirrors how `experiments/` and `scripts/` already sit at the repo root as independent, cross-cutting tooling. Metric-computation stays in `backend/evaluation_metrics.py` (extended in place) rather than moving to `evaluation/metrics/`, since it is already imported by `backend/tests/test_evaluation_metrics.py` and moving it would break that existing, passing test file for no benefit.

## 5. Exact files to add or change

**New files** (all listed in Section 4's tree above, plus):
- `review/evaluation_implementation_audit.md` (this file)
- `review/evaluation_results.md`
- `review/paper_metrics_source_of_truth.md`
- `review/evaluation_implementation_report.md` (final report, written last)

**Changed files**:
- `backend/evaluation_metrics.py` — add `adjusted_rand_index`, `adjusted_mutual_information`, `ndcg_at_k`, `brier_score`, `expected_calibration_error`, `wilson_confidence_interval`.
- `ACCURACY_EVALUATION.md` — update only the specific (C) rows that this task actually moves to (A)/(B)/`PARTIALLY MEASURED`, with an explicit new date.
- Living review docs (`review/quantitative_results.md`, `results_audit.md`, `research_claims_matrix.md`, `slide_ready_metrics.md`, `review_cheat_sheet.md`, `review_pack/03_module_status.md`, `04_results_and_metrics.md`, `05_phase20_results.md`, `paper_submission_status.md`) — updated only after real evaluation numbers exist, per the task's own Phase 20/21 instructions.

**Untouched** (per the task's explicit instruction): `review/phase21_real_investigation_validation.md`, `review/phase22_nbe_sensitivity_validation.md`, `review/novelty_argument.md`, `review/evidence_aware_investigation.md`.

## 6. Dependencies

None new. `evaluation_metrics.py`'s existing functions use only the Python standard library (`collections`, `itertools`); the additions (ARI/AMI/nDCG/Brier/Wilson interval) are all implementable in pure Python without adding `scikit-learn` or `scipy` as a new dependency, consistent with this project's existing preference for hand-implemented, dependency-light statistics (see the hand-implemented GraphSAGE in `ml/gnn/`).

## 7. External blockers (disclosed up front, not discovered mid-task)

1. **No independent human analyst is available in this session.** Per the task's own Phase 3/11/23 instructions, this means every ground-truth label this session produces must be filed as `AUTO_PROPOSED`, never `HUMAN_REVIEWED` or `LOCKED`. This session (an AI assistant) independently cross-referencing raw Wazuh rule text against MITRE ATT&CK documentation is a *meaningfully more independent* signal than calling CYUKTI's own resolver, but it is **not** a substitute for a qualified human reviewer, and will not be represented as one. The review-queue export (Phase 11) is what actually unblocks this — it's a real, usable artifact the user (or another reviewer) can act on.
   *Planned:* have a qualified human reviewer work through the exported review queue to promote `AUTO_PROPOSED` records toward `HUMAN_REVIEWED`/`LOCKED`.
2. **No live attack-execution automation exists**, and this session has shell access only to the Wazuh-manager/backend host, not to the separate Kali attacker VM or Ubuntu victim VM's own consoles. Running genuinely *new* attack scenarios (Phase 16) to generate fresh sequential sessions for prediction-opportunity evaluation would require either (a) operator-driven manual execution on the Kali VM while this session observes the resulting Wazuh alerts, or (b) tooling this session does not currently have. **This is flagged as a decision for the user**, not silently skipped: running new attacks would also change the Neo4j graph counts (`Campaign`/`AttackEvent`/`Operation` totals) that were just locked into `review/paper_submission_status.md` as the paper's final numbers in the immediately preceding task. Proceeding with new live attacks without confirming this tradeoff would risk quietly invalidating work just marked "final."
   *Planned:* obtain explicit operator sign-off before running any new live attacks, so the already-locked Neo4j counts are never disturbed without a deliberate decision.
3. **The existing raw alert archive is dominated by non-attack noise** (dpkg, disk-space, agent-connect events across ~463K historical lines) — a MITRE-accuracy sample drawn from it must be deliberately stratified toward attack-adjacent rule IDs, or it will trivially saturate at "correctly UNKNOWN" on housekeeping noise and produce a misleadingly high agreement number.
4. **Neo4j must be reachable** for the campaign-correlation and attribution evaluators (they read real `Campaign`/`AttackEvent` nodes) — connectivity was last confirmed live earlier this session; re-verified at evaluator-run time, not assumed.

## 8. Expected outputs

Per category, one of exactly four states — never silently converted between them:

- **MEASURED** — a real evaluator ran against a real (even if small, even if AUTO_PROPOSED-labeled) independent dataset and produced a number.
- **NOT MEASURED** — the evaluator exists and is tested, but has not been run against real data this session (e.g. no query set built yet).
- **UNMEASURABLE** — ran, but the real data produced zero valid opportunities (e.g. zero prediction opportunities in the current NEXT_TECHNIQUE graph).
- **BLOCKED_BY_ENVIRONMENT** — cannot be measured in this session for a stated infrastructure/human-availability reason (e.g. no independent human reviewer to promote AUTO_PROPOSED → LOCKED).

Given the blockers in Section 7, the realistic expectation set going into implementation is: MITRE mapping, campaign correlation, and attribution can reach **MEASURED (on AUTO_PROPOSED, not LOCKED, ground truth — explicitly caveated as preliminary/AI-assisted-provisional)** using existing historical Neo4j data and independently-known lab topology. Prediction accuracy is expected to land at **UNMEASURABLE** or a very small-n MEASURED, since the live NEXT_TECHNIQUE graph has only 3 learned edges. RAG and investigation-confidence evaluation are expected to land at **NOT MEASURED** initially (evaluator built and tested, but the required query set / analyst-verdict set does not yet exist and each requires either substantial independent construction or a human reviewer this session does not have) unless time permits building an initial AI-assisted-provisional query set for at least the MITRE-RAG source. Threat qualification is expected to reach **MEASURED (AUTO_PROPOSED)** using the same independent-topology-plus-rule-text methodology as MITRE mapping.
