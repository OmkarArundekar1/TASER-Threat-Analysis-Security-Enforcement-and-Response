# CYUKTI — Baseline Audit Against Research Feedback

**Date:** 2026-11-03 · **Repository HEAD:** `3afe62c` · **Scope:** read-only audit. No production code was modified.
**Feedback source:** `feedback .pdf` (repo root, 3 pages; sections (a)–(e)).

## Status log (this document is the frozen baseline; progress is tracked here)

| Date | Finding | Status | Evidence |
|---|---|---|---|
| 2026-11-03 | A2b live chain learning disabled | **Fixed + verified** (unit regression and live replay on a disposable Neo4j) | `STEP1_CHAIN_LEARNING.md`, `experiments/STEP1-chain-learning-live-replay-v1/` |
| 2026-11-03 | A2c origin of 3 historical edges | Narrowed (seed script ruled out); **blocked on live Neo4j** | `STEP1_CHAIN_LEARNING.md` §5, `queries/readonly_live_graph_validation.py` |
| 2026-11-04 | A8 wall-clock time in campaign lifecycle | **Fixed + verified**: event time throughout; deterministic replay harness; 4 runs at different speeds identical | `STEP2_EVENT_TIME.md`, `experiments/STEP2-event-time-replay-v1/` |
| 2026-11-04 | B3 (part) memory vs Neo4j `last_seen` divergence | Made deterministic (sweep order); **semantic choice still open** | `STEP2_EVENT_TIME.md` §8, §9.1 |
| 2026-11-04 | Live-graph validation | Superseded by the next row: was blocked while `neo4j-soc` was down | — |
| 2026-11-04 | Live-graph validation (read-only) | **Done.** A2c narrowed further: all 3 edges created 2026-07-17 05:39–05:54, last updated 06:10:02, never since, despite events through 2026-10-13. C2-data updated: 165 campaigns, 58 operations, 102 without an operation (71 UNKNOWN-only, 31 with resolved events), 4 in two operations; 69 self-loops | `experiments/STEP2A-dedup-deadlock-v1/live_validation_20261104.json` |
| 2026-11-04 | New: dedup restart self-deadlock (pre-existing, not in the original audit) | **Fixed + verified** | `STEP2A_DEDUP_DEADLOCK.md` |
| 2026-11-04 | B3 correction: UNKNOWN duplicates do **not** extend `Campaign.last_seen` in Neo4j (no `MATCHES` edge stops `update_duplicate_event` before the campaign update). `STEP2_EVENT_TIME.md` §8/§9.1 and earlier reports said they do | Verified on a real Neo4j; repository unchanged; decision pending in the B3 study | `experiments/b3_runs_v2/README.md`, `experiments/b3_runs_v2/smoke/dup_update_CUR.json` |
| 2026-11-04 | B3 activity policy | **Decided: CUR retained as frozen reference, not declared correct; A and B not selected.** Activity semantics still open: next is instrumentation and an activity-ownership contract. Evidence frozen | `B3_DECISION.md`, `experiments/b3_runs_v2/comparison/REPORT.md`, `experiments/b3_runs_v2/FREEZE_20261104.sha256` |

## 0. How this audit was produced, and its limits

| Evidence source | Used for | Status |
|---|---|---|
| Source code at HEAD (`isfcr/CYUKTI/backend`, `evaluation/`, `scripts/`) | Execution-path tracing | Read directly |
| `evaluation/results/*.json` (run 2026-09-26, commit `f8bdf07`) | Historical metrics | Read directly |
| `backend/ml/datasets/campaign_dataset.csv` (60 rows, gitignored) | Severity/ML dataset | Re-analysed |
| `data/raw_logs/alerts.json` (9,015 Wazuh alerts, 2026-03-11, gitignored) | Rule-firing and UNKNOWN-rate measurement | Re-analysed |
| `evaluation/scenarios/scenarios.json` | Ground-truth definition | Read directly |
| Project docs deleted from the working tree earlier this session (`journal_ready_data.md`, `review/*`, `review_pack/*`, `OPERATIONS_VIEW.md`, …) | Historical claims | Read from `git show HEAD:<path>` (still recoverable) |
| Backend test suite | Baseline | Ran: **863 passed, 29 skipped, 2 failed**. Both failures are `ServiceUnavailable` (Neo4j not running), in `test_operation_attribution_integration.py` and `test_threat_attribution.py`. These tests aren't marked `requires_live_neo4j`. |
| **Live Neo4j graph** | Current graph state, edge counts, operation membership | **NOT AVAILABLE.** Nothing listening on 7687/7474. Every claim about live graph state below is quoted from committed docs/results and is labelled as such. |

Verdict labels used below:
- **VERIFIED-CODE:** proven by reading the code path at HEAD.
- **VERIFIED-DATA:** reproduced from data files in the repo.
- **REPORTED:** stated in committed results/docs; not re-checked because Neo4j is offline.
- **UNRESOLVED:** evidence unavailable; no guess made.

---

## 1. Traced execution paths (as-is)

### 1.1 Alert → campaign (the core loop)
`listener/wazuh_listener.py` → `realtime_socgraph.process_alert(alert)`:

1. `should_ignore_audit_command()` drops helper shell commands.
2. `mitre_resolver.resolve_mitre(alert)`: NATIVE_WAZUH → REVIEWED_RULE_MAPPING → DETERMINISTIC_INFERENCE → UNKNOWN.
3. **UNKNOWN/AMBIGUOUS** → `_process_unattributed_alert()` (§1.4), return.
4. `extract_iocs()`: attacker = `data.src_ip|srcip`, victim = `data.dest_ip|dstip|agent.name|…`. **If there is no source IP, attacker := victim** (`realtime_socgraph.py:101-102`).
5. `dedup_engine.is_duplicate(attacker, victim, technique, rule_id, agent_id)`, 90 s window on wall-clock `time.time()`.
6. `campaign_manager.get_or_create_campaign(attacker, victim, technique)`:
   - `resolve_campaign()` looks up the cached/DB campaign keyed **`(attacker_ip, victim_ip)`** (`campaign_manager.py:87-92`).
   - If the active campaign isn't expired (`CAMPAIGN_TIMEOUT = 120 s`, wall-clock), `campaign_feature_engine.extract()` → `campaign_decision_engine.evaluate()`; continue if `score ≥ 0.35`, else **close the campaign**.
   - Otherwise try inactive campaigns for the **same pair**: "fast reopen" if first technique == current and temporal ≥ 0.8, else reopen if `score ≥ 0.70`.
   - Otherwise `create_campaign_context()` sets `skip_next_chain_update = True`.
   - `activate_campaign()` → `update_campaign_activity()` **writes `c.last_technique = current`**.
7. `neo4j_client.create_attack_event()` also sets `c.last_technique = current` and builds Attacker/Host/Campaign/Technique/Stage/AttackEvent nodes.
8. Feature orchestrator → `severity_engine` → `dynamic_risk_engine` → `store_dynamic_risk`.
9. `chain_updater.update_attack_chain(campaign_id, technique)` reads `c.last_technique` as *previous*.
10. `update_campaign_similarity` (SIMILAR_TO), `update_actor_attribution` (RESEMBLES).
11. `prediction_engine.predict_next()` → `LIKELY_NEXT`.
12. `operation_manager.build_campaign_context` → `campaign_correlation_engine.correlate()` → attach to an existing operation, or `create_operation_db`.
13. Detection / threat / `cti_confidence_engine` → `misp_sync.publish_campaign()` (auto-publishes if CTI score ≥ 40).
14. POST to `SHUFFLE_WEBHOOK`. The module-level value is the hard-coded `""` (`realtime_socgraph.py:76`), so this always errors, and the error is caught.

### 1.2 Campaign continuation score (the split decision)
`config.CAMPAIGN_WEIGHTS`: prediction 0.35, chain 0.30, temporal 0.20, attacker 0.10, runtime 0.03, graph 0.01, duplicate 0.01. Missing (`None`) features are dropped and the remaining weights renormalised (`campaign_decision_engine.py:40-55`).

| Situation for `last_technique = A`, incoming `B` | prediction | chain | Resulting score | Outcome |
|---|---|---|---|---|
| A has **no** learned outgoing edges | `None` (dropped) | `0.0` | ≈ (0.20+0.10+0.03+ε)/0.65 ≈ **0.51–0.54** | continue |
| A has learned edges, but **not** A→B | `0.0` | `0.0` | 0.20+0.10+0.03+0.01·g+0.01·d ≤ **0.35** (normally 0.33–0.34) | **split** |
| A has learned edges and **B == A** (repeat of same technique) | `0.0` | `0.0` | same as above | **split** |
| A→B learned | p | c | ≥ 0.35 | continue |

The feedback's arithmetic (0.34 + 0.65·0) is reproduced exactly. Learning a successor for A turns every *other* successor of A from "no information" into "evidence against" (VERIFIED-CODE). That is the "learning makes it stricter" mechanism.

### 1.3 Transition learning / next-technique prediction
- Writer: `chain_updater.update_attack_chain()` only. `attack_chain_builder.py` is a manual seeding script with a hard-coded 5-step chain; nothing calls it.
- Reader: `neo4j_client.get_prediction_distribution`, `campaign_feature_engine.chain_similarity`, `prediction_engine.predict_next_readonly` (min 3 observations, ≥ 30% confidence).
- **Live graph at last evaluation:** 3 edges, `T1110.001→T1110 (6)`, `T1110→T1078 (5)`, `T1078→T1110.001 (1)`, each with confidence 1.0 (REPORTED, `prediction_metrics.json`).

### 1.4 UNKNOWN path
`_process_unattributed_alert()`: dedup → `resolve_campaign_context(attacker, victim)` (active only) → otherwise `create_campaign_context(..., current_technique=None)` → `create_unattributed_attack_event` (no Technique node, no `last_technique` change).

It skips all of the following: the decision engine, chain learning, prediction, risk, **operation correlation**, and MISP. It also never refreshes the in-memory `context.last_seen`, only the Neo4j copy.

### 1.5 Operation layer
`OperationDecisionEngine.weights`: attacker 0.20, victim 0.15, technique-Jaccard 0.30, temporal 0.20, chain-LCS 0.10, prediction 0.05, graph 0.00. Attach threshold **0.70**. Only *active* operations are searched if any exist; inactive ones are searched only when none are active.

### 1.6 Downstream consumers of campaigns
Severity/risk (`severity_engine`, `dynamic_risk_engine`, `risk_scoring`), XGBoost (`ml/train_xgboost.py`, `ml/runtime_predictor.py`), GNN (`ml/gnn/*`, `rag/gnn_topology_retriever.py`), historical retrieval (`rag/campaign_retriever.py`, `evidence/collectors/campaign_history_collector.py`, `attribution_*`), investigation (`investigation/loop.py`, `next_best_evidence.py`), MISP (`misp_sync.py`, `cti_confidence_engine.py`), SOAR (`soar/*`, `active_response/*`). Every one of them keys on `campaign_id` and/or the campaign's technique set.

---

## 2. Feedback-to-code traceability matrix

### (a) Campaign splitting: the core design issue

| ID | Feedback requirement | Actual current implementation | File / function | Verdict | Proposed correction | Dependencies | Tests required |
|---|---|---|---|---|---|---|---|
| A1 | Unseen transition must mean "no information", not 0 | `prediction_similarity` returns `distribution.get(B, 0.0)`; `chain_similarity` returns `count/total` (0 if unseen) once A has any outgoing edge | `campaign_feature_engine.py:23-91`; `campaign_decision_engine.evaluate` | **VERIFIED-CODE** (§1.2) | Return a prior for unseen transitions. Candidates are a Laplace/Dirichlet-smoothed estimate `(n_AB + α)/(n_A + α·K)` or the PDF's 0.1 floor. **Self-transitions (B == A) must score as continuation, not 0.** | A3 (merging the features changes the arithmetic); A4 (choosing α/floor) | Unit: unseen A→B with learned A→C does not split. Unit: A→A does not split. Regression: the §1.2 table rows. |
| A2 | Record the transition even when splitting, if same attacker and close in time | On every new/reopened campaign `skip_next_chain_update = True`, so the A→B that caused the split is never learned | `campaign_manager.create_campaign_context` L372, reopen paths L513/L575; `chain_updater.py:43-61` | **VERIFIED-CODE** | Record A→B whenever same attacker (or linked entity, see C1) and Δt ≤ gap threshold, independent of the split decision. Use **alert timestamps** for Δt (see A8). | A8, A9 | Unit: split with small Δt still increments A→B. Unit: large Δt does not. |
| A2b | *(not in feedback, found during audit)* Live chain learning appears **completely disabled** | `activate_campaign()` → `update_campaign_activity()` sets `c.last_technique = current` **before** `update_attack_chain()` reads `c.last_technique` as *previous*. `create_attack_event()` does the same. So previous == current, and the code takes the "Duplicate technique ignored" branch on every live event. | `campaign_manager.py:597-616`; `neo4j_client.update_campaign_activity` L677-690, `create_attack_event` L130; `chain_updater.py:22-68` | **VERIFIED-CODE** (ordering). Live effect UNRESOLVED (Neo4j offline). | Pass `previous_technique` explicitly from the in-memory context captured *before* activation, instead of re-reading the overwritten DB property. | None. **Must be fixed first; every other learning fix depends on it.** | Integration test with fake Neo4j: two sequential events A then B in one campaign create the edge A→B. No such test exists today: `test_realtime_mitre_integration.py` mocks `update_attack_chain`. |
| A2c | *(audit)* Provenance of the 3 learned edges | Given A2b, these edges can't have come from the HEAD live path. They don't match `attack_chain_builder.ATTACK_FLOW` either. | — | **UNRESOLVED.** Possibly an older code ordering (only one commit of history exists, `5c66e7c`), or scripts. | Inspect edge `created_at` timestamps once Neo4j is up; reset the transition graph before re-running experiments. | Neo4j access | — |
| A3 | Merge prediction + chain into one clue with a smaller weight | Both features read the same `NEXT_TECHNIQUE` edges out of A; together they carry 0.65 of the weight. `prediction` = `count(A→B) / Σ counts` (a probability). **`chain` = `count(A→B) / number_of_outgoing_edges`** (`RETURN count(r) AS total`, L52-60). That is not a probability and **can exceed 1**: live edge `T1110.001→T1110` has count 6 and is the only edge, so chain = 6.0 and contributes 1.8 to a score meant to lie in [0,1]. | `campaign_feature_engine.py:23-91`; `config.CAMPAIGN_WEIGHTS` | **VERIFIED-CODE.** The two are redundant (same evidence, both 0 for an unseen B), and `chain` is additionally mis-normalised. | Replace with one `transition_plausibility` feature (smoothed, A1) at a reduced weight. Add a tactic-progression prior from ATT&CK kill-chain order (data already imported as `Technique.kill_chain_phases`) as an independent, label-free signal. | A1, B (labels), A4 | Unit: the new feature equals the smoothed probability. Weight-sum invariant test. |
| A4 | Fit weights on real labelled data (e.g. AIT), not by hand | All weights and thresholds are hard-coded constants, and comments say so | `config.CAMPAIGN_WEIGHTS`, `campaign_manager.py:73-80`, `operation_decision_engine.py:13-22`, `cti_confidence_engine.py:1-12`, `next_best_evidence.py` W_* | **VERIFIED-CODE** | Build an offline replay + grid/Bayesian search harness. Objective: maximise the harmonic mean of purity and inverse purity (or pairwise F1 / ARI) on labelled training scenarios. Report on held-out scenarios. | A8 (replay), D1 (AIT dataset), A5 (metric) | Harness determinism test; no-leakage test (tuning set ≠ reporting set). |
| A5 | Report inverse purity alongside purity | Only `cluster_purity`, pairwise P/R/F1, ARI, AMI and `campaign_fragmentation`. **No inverse purity.** | `backend/evaluation_metrics.py:154-195`; `evaluation/evaluators/campaign_correlation_eval.py:67-93` | **VERIFIED-CODE** | Add `inverse_purity`, purity/inverse-purity F (Van Rijsbergen F-measure), `campaigns_per_true_attack`, B-cubed P/R. | none | Hand-computed unit tests: perfect, all-split, all-merged. |
| A6 | Target ~1–2 campaigns per real attack (now ~15) | 46 campaigns / 3 sessions = **15.3**. Fragmentation 4 / 17 / 25. Purity 1.0, pairwise F1 0.0, ARI 0.0. | `evaluation/results/campaign_correlation_metrics.json` | **REPORTED** (consistent with code) | Acceptance gate: campaigns-per-attack ≤ 2 **and** purity not degraded below an agreed floor (guards against over-merging). | A1–A3, A7, C1 | Evaluator asserts both metrics are emitted. |
| A7 | *(audit; underlies the (a) fragmentation)* Timeout and reopen | `CAMPAIGN_TIMEOUT = 120 s` wall-clock. After 2 min of silence the campaign goes INACTIVE. Reopen needs score ≥ 0.70, which (§1.2) is unreachable when the transition terms are 0, unless first technique == current and the gap is ≤ ~216 s. | `config.py:13`; `campaign_manager.py:74, 489-515, 567` | **VERIFIED-CODE** | Make the inactivity gap a tuned parameter (A4) measured in alert time. Lower the reopen bar and make it consistent with the continuation score. | A4, A8 | Unit: pause of N·timeout with plausible next step reopens. |
| A8 | *(audit; blocks (a)iv and (d))* Wall-clock time everywhere | `datetime.now()` / Neo4j `datetime()` / `time.time()` are used for last_seen, temporal similarity, dedup, expiry and operation timing. The alert's own `timestamp` is only stored inside the payload. | `campaign_manager.py` (L140, 188, 362, 389, 510, 572, 604), `campaign_feature_engine.temporal_similarity`, `dedup_engine.is_duplicate`, all `neo4j_client` writers | **VERIFIED-CODE** | Thread an event-time clock (`alert.timestamp`) through the pipeline, with wall-clock as fallback for live mode. **Without this, AIT or archive replay at speed produces meaningless temporal features, and repeated runs cannot be reproduced.** | none | Replay of the same alert file twice gives identical campaigns. Replay ×10 speed gives the same campaigns as ×1. |
| A9 | Rerun everything after fixing campaigns | All downstream datasets/metrics were built on the fragmented campaigns (see §3) | `scripts/build_real_dataset.py`, `ml/*`, `evaluation/run_all.py` | **VERIFIED-DATA** | Re-run the pipeline end-to-end after A1–A8; regenerate the dataset, models and evaluations; archive the old numbers as "pre-fix baseline". | everything above | Pipeline smoke test from replay → dataset → evaluation. |

### (b) Labelling starvation

| ID | Feedback requirement | Actual current implementation | File / function | Verdict | Proposed correction | Dependencies | Tests required |
|---|---|---|---|---|---|---|---|
| B1 | ~82% of alerts are UNKNOWN; only Wazuh labels | Tiers 2 and 3 are **empty by design**: `REVIEWED_RULE_MAPPINGS = {}` and `DETERMINISTIC_INFERENCE_RULES = []`. Effectively one tier. | `mitre_rule_registry.py:34-42`; `mitre_resolver.py:167` | **VERIFIED-CODE** | Populate the reviewed registry (B2). Optionally add deterministic structural rules, e.g. Suricata signature-ID families. | B2 | Resolver tests per new mapping; tier-precedence tests (exist). |
| B1-data | Size of the problem | Archive `data/raw_logs/alerts.json`: **9,015 alerts, 85.5% UNKNOWN**, 35 distinct rule IDs, **8 rules cover 95% of volume**. Repo snapshots: 98.7% → 90.0% → 78.3% UNKNOWN. | `scripts/mitre_coverage_report.py`; `git show HEAD:isfcr/CYUKTI/review_pack/05_phase20_results.md` | **VERIFIED-DATA.** The PDF's "82%" isn't stated verbatim anywhere, but falls inside the measured range. | Labelling effort is small: ~35 rule IDs. Example candidates (need human review, **not** decided here): 86601 "ET SCAN Possible Nmap User-Agent" (UNKNOWN today), 31101 web 400 burst, 533 new listening port, 510 rootcheck. | Two independent reviewers (E1) | — |
| B2 | Map rules that actually fire to ATT&CK | No mapping table or review workflow for rule→technique. The review-queue infrastructure exists for ground truth only. | `evaluation/review/*`; `mitre_rule_registry.py` | **VERIFIED-CODE** | `research/rule_mapping/` CSV: rule_id, description, candidate technique(s), rationale, reviewer A, reviewer B, adjudicated, κ. Load the adjudicated rows into `REVIEWED_RULE_MAPPINGS` with `reviewer` fields populated. | E1 | Every registry entry validated against the STIX corpus (exists: `_validate_technique_ids`). |
| B3 | UNKNOWN alerts are kept but the smart parts can't use them | UNKNOWN path skips the decision engine, transitions, prediction, risk, **operation correlation** and MISP. It doesn't refresh in-memory `last_seen`, so an active campaign can expire while UNKNOWN alerts are still arriving. It creates a campaign with `observed_chain=[None]`. | `realtime_socgraph._process_unattributed_alert` L168-270; `campaign_manager.create_campaign_context` | **VERIFIED-CODE** | (i) Refresh `last_seen` on UNKNOWN events. (ii) Run operation correlation for UNKNOWN-only campaigns (see C2). (iii) Use rule-group / rule-ID as a coarse non-ATT&CK token for temporal grouping without fabricating techniques. | A8, C2 | Unit: UNKNOWN event keeps the campaign alive. Unit: UNKNOWN-only campaign gets an operation. |
| B4 | Stage scoring can't see progression, so serious attacks look mild | `MITRE_TO_STAGE` has 13 techniques; `TPS_MAP` has 6 tactics. In the archive, natively-labelled techniques T1083, T1059.007, T1548.003, T1059, T1565.001, T1562.001 fall to stage "Unknown" → **TPS 0**. Today's `local_rules.xml` maps T1105, T1498.001 and T1499.002, also TPS 0. | `mitre_mapper.py:43-59`; `config.TPS_MAP` | **VERIFIED-CODE/DATA** | Derive stage from the imported STIX `kill_chain_phases` for all 14 tactics (code exists: `mitre_resolver._tactic_labels_from_kill_chain_phases`). Extend `TPS_MAP` to 14 tactics; that is a documented risk-model decision. | none | Every technique in the archive has a stage ≠ Unknown. |
| B5 | Evaluation hides the starvation | `mitre_mapping_v0` sample is **100% NATIVE_WAZUH** (`unknown_rate 0.0`), so it measures label *precision* on already-labelled alerts and nothing about coverage. | `evaluation/results/mitre_mapping_metrics.json` | **VERIFIED-DATA** | Add a coverage metric over **all** alerts (labelled share per provenance) plus accuracy over a stratified sample that includes UNKNOWNs. | B2 | Evaluator test with UNKNOWN rows. |
| B6 | *(audit)* Rule-ID instability | `local_rules.xml` still contains a **duplicate rule ID 100002** (level 0 and level 6). Rule 100200 meant "Zeek network activity" in the March archive but maps to T1110 today. Historical alerts can't be relabelled by rule ID alone. | `backend/wazuh_rules/local_rules.xml` | **VERIFIED-DATA** | Version the rule file and key mappings by (rule_id, ruleset_version) or by rule description hash. Fix the duplicate ID. | none | XML lint test: unique IDs. |

### (c) Cross-host / operation correlation

| ID | Feedback requirement | Actual current implementation | File / function | Verdict | Proposed correction | Dependencies | Tests required |
|---|---|---|---|---|---|---|---|
| C1 | Campaign must follow the attacker through the network (A compromised → A attacks B) | Campaign key is strictly `(attacker_ip, victim_ip)`. No host-to-host pivot edge exists in the graph schema (Attacker-LAUNCHED→Campaign-TARGETS→Host only). | `campaign_manager._key` L87-92; `neo4j_client.get_active_campaign_db` L527-566; `create_attack_event` | **VERIFIED-CODE** | Add entity resolution (IP ↔ agent name ↔ hostname) and a pivot rule: if host H was a *victim* in campaign K within window W and H now appears as *attacker*, link to K (a `PIVOTED_FROM` edge). | A8, C3 | Scenario test: .106→A then A→B yields one campaign/operation. |
| C2 | …or show the operation layer reliably rejoins fragments | **It structurally cannot for lateral movement.** New attacker + new victim gives max score 0.30+0.20+0.10+0.05 = **0.65 < 0.70**. Same attacker/victim fragments need technique Jaccard ≳ 0.5, but split fragments differ in technique by construction. UNKNOWN-only campaigns are never correlated (B3). Inactive operations are ignored while any operation is active. | `operation_decision_engine.py:13-37`; `operation_feature_engine.py`; `campaign_correlation_engine.correlate` L49-55 | **VERIFIED-CODE** | Fix C1 first, then re-evaluate operations with the same purity/inverse-purity metrics at operation level. | C1, A5 | Unit: max-score bound test. Fragment-rejoin scenario test. |
| C2-data | Evidence | A live query documented 2026-09-27: **154 campaigns, 57 operations, 88 campaigns (≈60%) with no operation**. The root cause was "not root-caused" in the doc. The code shows UNKNOWN-only campaigns never reach operation correlation; that is a probable but **unconfirmed** share of the 88. | `git show HEAD:isfcr/CYUKTI/backend/OPERATIONS_VIEW.md` | **REPORTED** + code-consistent | — | Neo4j access to confirm | — |
| C3 | *(audit)* Identity defects inflate fragments | (i) Host-only alerts set attacker := victim. In the ML dataset, **23 of 60 campaigns are self-loops** (`pes1ug23cs411-VirtualBox → itself`); 9.1% of archive alerts have no source IP. (ii) The same victim may appear both as an IP (`192.168.56.105`) and an agent name (`pes1ug23cs411-VirtualBox`). Whether these are the same machine is **UNRESOLVED** (scenario notes conflict: SESSION-BOUNDARY-001 is titled "Kali(.106)" but records attacker .105). | `realtime_socgraph.extract_iocs` L78-103 | **VERIFIED-DATA** (i); **UNRESOLVED** (ii) | Store the source as `UNKNOWN_SOURCE`, not the victim. Attach host-only alerts to the active campaign(s) targeting that host. Add an asset inventory (IP↔hostname). | Lab topology documentation | Unit: host-only alert attaches to the existing campaign on that victim. |
| C4 | *(audit)* CTI "campaign confidence" is a constant | `campaign_confidence = OperationDecision.confidence = len(IMPLEMENTED_FEATURES)/7·100 = 71.4` always, matched or not | `operation_decision_engine.py:42-44`; `realtime_socgraph.py:596-602` | **VERIFIED-CODE** | Feed the real correlation score (or the campaign continuation score). | A1–A4 | Unit: CTI breakdown varies with the correlation result. |

### (d) Lab and evidence base

| ID | Feedback requirement | Actual current implementation | File / function | Verdict | Proposed correction | Dependencies | Tests required |
|---|---|---|---|---|---|---|---|
| D1 | Add the public AIT dataset | **Absent.** Present only: CICIDS2017 flow CSVs (`isfcr/CYUKTI/datasets/*.csv`, network flows, not Wazuh alerts) and one March-2026 Wazuh/Suricata/Zeek archive (`data/raw_logs/`). No AIT loader/adapter. | — | **VERIFIED** | Download AIT-ADS / AIT-LDS (the AIT Alert Data Set contains Wazuh + AMiner alerts with attack-step labels). Write an adapter → CYUKTI alert schema → replay harness (A8). | A8; disk/network; AIT licence review | Adapter schema tests; label-join tests. |
| D2 | More varied attacks; repeat 5–10×; error bars | 3 sessions in `scenarios.json`; several have `provenance: synthetic_generator_declared_intent`, not live Kali. Single-run results; Wilson CIs only on proportions; no repeated-run harness. `benchmarks/run_benchmarks.py` measures latency, not accuracy. | `evaluation/scenarios/scenarios.json`; `evaluation/run_all.py` | **VERIFIED** | Scripted scenario library (recon, brute force, web exploit, privesc, persistence, lateral movement, exfil) with seeds. N=5–10 runs → mean ± sd / bootstrap CI. Record each run with `experiments/record_experiment.py` (exists). | A8, lab VMs (Kali + ≥2 Ubuntu victims for C1) | Harness aggregates N runs deterministically. |
| D3 | *(audit)* Ground truth can't test (c) | Ground-truth sessions are keyed `SESSION::<scenario>::<attacker>-><victim>`, the **same key as the system**. It can't detect cross-host merge failures or over-merging (stated in the evaluator's own limitations). | `evaluation/evaluators/campaign_correlation_eval.py`; `ground_truth/builder.py` | **VERIFIED-CODE** | Define ground truth by **attack episode ID** (from AIT labels or scenario run IDs), not by IP pair. | D1, D2 | — |

### (e) AI-decided labels

| ID | Feedback requirement | Actual current implementation | File / function | Verdict | Proposed correction | Dependencies | Tests required |
|---|---|---|---|---|---|---|---|
| E1 | Two people label independently; report agreement, or disclose AI use | All four ground-truth datasets are `AUTO_PROPOSED` (AI-assisted). `store.lock_dataset()` correctly refuses to lock without `HUMAN_REVIEWED`. **No inter-annotator agreement metric** exists (no Cohen's κ / Krippendorff's α in the codebase). Threat-qualification has an "assistant-adjudicated" layer, also AI. | `evaluation/ground_truth/store.py`; `evaluation/results/summary.json` (`dataset_versions`) | **VERIFIED** | Add a dual-annotator workflow (reviewer_a/reviewer_b columns in review CSVs), `cohens_kappa()` in `evaluation_metrics.py`, adjudication step, and an explicit AI-use disclosure field in the manifest. | Two human annotators (team) | κ unit tests vs hand-computed values. |

---

## 3. Historical claims checked against code and data

| Claim (feedback or repo) | Check | Result |
|---|---|---|
| "3 real attacks became 40+ campaigns" | `campaign_correlation_metrics.json` | **Confirmed:** 46 campaigns / 3 sessions |
| "~15 campaigns per real attack" | 46/3 | **Confirmed:** 15.3 |
| "50 of 60 campaigns are Low" | `campaign_dataset.csv` | **Worse than stated:** Low 53, Medium 3, Critical 4, High 0 |
| Severity dataset = pieces of the same few attacks | dataset pairs | **Confirmed:** only 4 attacker/victim pairs; 35/60 campaigns have one unique technique; 23/60 are attacker == victim self-loops |
| Severity label validity | `ml/label_generator.py` | Label is a deterministic function of summed stage TPS. `risk_score` is excluded as leakage (`dataset_utils.LEAKAGE_COLUMNS`), but technique-derived features remain correlated. Indirect leakage is **plausible, UNRESOLVED** (needs an ablation). |
| Production XGBoost metrics | `ml/train_xgboost.py:115-171` | Random `train_test_split` (not grouped by attack), so fragments of the same attack sit on both sides. **Calibration is fit on the validation split and then scored on that same split.** The LOGO ablation (`xgboost_ablation.py`) is grouped by attacker IP, i.e. ~3–4 groups. Reported pooled accuracy 0.900 / macro-F1 0.449 ≈ majority-class behaviour (Low recall 1.0, Critical recall 0.25). |
| "MRR 0.975 same-attacker retrieval" | `ml/gnn/retrieval_evaluation.py`; journal §11.3 | **Confirmed the number exists**, n=68. With only ~3 attacker groups, a random ranker already scores high MRR (the relevant set is ~⅓–½ of candidates). No random/trivial baseline is reported next to it. Feedback point stands, and the baseline issue makes it stronger. |
| "GNN added nothing" | journal §12.2 | **Confirmed by the repo itself:** severity delta 0.000 on every metric |
| "Only 3 attack patterns ever learned" | `prediction_metrics.json` | **Confirmed:** 3 edges. A2b explains why the live path can't add more. |
| Next-technique prediction accuracy | `prediction_metrics.json` | 4/12 = 33% (Wilson 14–61%), status UNMEASURABLE for new data |
| "Purity looks excellent" | metrics | **Confirmed:** purity 1.0 with pairwise F1 0.0 |
| "Operation layer repairs fragments" | §2 C2 | **Refuted by code bound** and by 57 operations for 3 sessions (REPORTED) |
| Threat qualification / MISP publishing on counts, not behaviour | `cti_confidence_engine.py`; `detection_confidence_engine.py`; `misp_sync.should_publish` | **Partly confirmed:** publish = CTI score ≥ 40. Detection = `min(100, rule_level·10)`·0.25 plus the constant 71.4·0.20 = 14.3 is already ≈ 39.3 for any rule level ≥ 10. The publish decision is effectively gated by Wazuh rule level. Matches the repo's own "100% recall / 0% precision on QUALIFIED_THREAT" finding. |
| Investigation (NBE) depends on campaigns | `investigation/next_best_evidence.py` | NBE ranking is **campaign-content-agnostic** (static per-action metadata plus novelty). `mitre_knowledge` is always step 1. Already disclosed in journal §10.x. Not a feedback item, but its evaluation is campaign-dependent. |

---

## 4. Feature classification

### 4.1 Already solves part of the feedback (keep, reuse)
- `mitre_resolver.py` tiered design with provenance + STIX validation. The framework for (b) exists; it only needs content.
- `evaluation_metrics.py`: pairwise P/R/F1, ARI, AMI, fragmentation, Wilson CI. Extend it; don't rewrite.
- `evaluation/` framework: scenario registry, ground-truth store with `HUMAN_REVIEWED` lock gate, review-queue export/import, manifest with git commit + versions. This is the right base for (d)/(e).
- `experiments/record_experiment.py` + `schema.json`: per-run records for repeated experiments.
- `prediction_engine.predict_next_readonly`: side-effect-free predictor suitable for offline evaluation.
- Safety rails for response: `active_response/policy.py` (AUTO_CONTAIN off by default, never-block list), SOAR `requires_approval` actions. These meet the feedback's "respond safely" praise.
- `mitre_import/` STIX corpus incl. `kill_chain_phases`, which is enough to derive stages for all tactics (B4) and a label-free tactic-order prior (A3).

### 4.2 Incomplete
- `REVIEWED_RULE_MAPPINGS` and `DETERMINISTIC_INFERENCE_RULES` (empty).
- `MITRE_TO_STAGE` / `TPS_MAP` (13 techniques / 6 tactics).
- Inverse purity, B-cubed, campaigns-per-attack, inter-annotator κ.
- Replay harness / event-time clock; AIT adapter; multi-run aggregation with error bars.
- Ground truth by attack episode (not IP pair); over-merge test scenarios.

### 4.3 Needs modification
- `campaign_feature_engine.py` + `campaign_decision_engine.py` + `config.CAMPAIGN_WEIGHTS`: smoothing, feature merge, self-transition handling, `chain_similarity` normalisation (currently > 1 possible), fitted weights (A1, A3, A4).
- `campaign_manager.py`: chain-update skip flag, timeout/reopen policy, keying (A2, A7, C1).
- `chain_updater.py` + `campaign_manager.activate_campaign` / `neo4j_client.update_campaign_activity`: previous-technique ordering bug (A2b). **Highest priority.**
- `realtime_socgraph.extract_iocs` + `_process_unattributed_alert` (C3, B3).
- `operation_feature_engine` / `operation_decision_engine` / `campaign_correlation_engine` (C1, C2, C4).
- `ml/train_xgboost.py`: grouped split by attack episode; separate calibration and test splits.
- `ml/gnn/retrieval_evaluation.py`: add random/trivial baselines; group by episode.
- `wazuh_rules/local_rules.xml`: duplicate ID 100002.

### 4.4 Should not be changed (protect during the rework)
- The **no-fabrication principle**: UNKNOWN stays a first-class outcome; new mappings go through the reviewed tier with named reviewers, never inferred silently.
- Active-response safety defaults and approval gating (`active_response/`, `soar/`).
- MITRE STIX import and technique validation.
- The ground-truth `lock_dataset()` human-review gate.
- Investigation evidence/provenance schema (`evidence/schema.py`, `evidence/store.py`).
- Frozen historical results: keep the `evaluation/results/*.json` files as the "pre-fix baseline" rather than overwriting them.

---

## 5. Missing datasets and infrastructure

| Item | Needed for | Status |
|---|---|---|
| AIT Alert Data Set (Wazuh/AMiner alerts with attack labels) | A4, D1, D3, B1 generalisation | **Missing**; must be downloaded (size, licence to confirm) |
| Live Neo4j (`neo4j-soc` container) | Confirm A2b live effect, A2c edge provenance, C2-data | **Not running** in this environment |
| Event-time replay harness | A4, A8, D2 | **Missing** (code work) |
| Second victim VM (and pivot path) | C1 lateral-movement scenarios | **UNRESOLVED**: lab topology not documented in repo; cannot tell if available |
| Two human annotators | B2, E1 | Team resource, not code |
| Asset inventory (IP ↔ hostname ↔ agent) | C3 | **Missing** |
| Wazuh manager access (`wazuh-analysisd -t`) | B6, confirming loaded rule set | Root access required (doc notes it was unavailable) |
| `aisd/`, `runs/` (untracked) | — | Unrelated to CYUKTI (pothole detection / YOLO runs); ignore |

---

## 6. Proposed order of work (for approval; nothing implemented)

1. **A2b** fix the chain-learning ordering bug, plus a test that fails today. *Small, isolated, unblocks all learning.*
2. **A8** event-time clock + **replay harness** over `data/raw_logs/alerts.json`. Lets every later change be measured reproducibly.
3. **A5** inverse purity / B-cubed / campaigns-per-attack. Re-score the current baseline so the "before" number is honest.
4. **A1 + A3 + A2 + A7** smoothed single transition feature, self-transition fix, record-on-split, timeout/reopen consistency.
5. **C3 → C1 → C2/C4** identity hygiene, pivot linking, operation layer re-evaluation.
6. **B2/B4/B5** dual-reviewed rule mapping (~35 rules), stage derivation from STIX, coverage metric. **E1** κ workflow alongside.
7. **D1/D2/D3** AIT adapter, episode-keyed ground truth, scenario library, N-run aggregation with error bars.
8. **A4** weight fitting on training scenarios, report on held-out. Then **A9**: rebuild datasets, retrain XGBoost with grouped splits and baselines, re-run GNN retrieval with trivial baselines, re-run all evaluations.

Decisions needed from the team before step 4: the smoothing form (PDF 0.1 floor vs Laplace/Dirichlet); the gap threshold for record-on-split; whether `TPS_MAP` is extended to all 14 tactics (risk-model policy).
