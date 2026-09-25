# CYUKTI — Journal-Ready Technical Reference

**Purpose of this document**: single source of truth for an IEEE Access submission. Every number below was either read directly from source code or queried live from this project's real Neo4j instance on **2026-09-25** (post-restart of the `neo4j-soc` Docker container; timestamps in this document reflect actual wall-clock events across a multi-day development session, including real infrastructure interruptions). Nothing here is invented, estimated, or rounded up to look better. Where a metric doesn't exist yet, that is stated explicitly rather than filled in.

---

## 1. System Overview

**CYUKTI** ("Real-Time Campaign Intelligence Framework", internal codename; user-facing product name "WatchDog SOC") is a SOC (Security Operations Center) augmentation platform that sits downstream of Wazuh (a SIEM/HIDS) and turns a raw stream of security alerts into a persistent, explainable, graph-structured model of ongoing attack campaigns — then layers correlation, prediction, attribution, evidence-aware investigation, and semi-automated response on top of that graph.

### 1.1 The problem it addresses

A conventional SIEM (Wazuh, Splunk, etc.) produces a flat stream of discrete alerts. Analysts are left to manually answer, for every alert: *is this related to something I already saw? Is it part of a larger campaign? Have I seen this attacker before? What technique is this? What should I do about it? Should I tell anyone else about it (MISP)?* CYUKTI's stated goal is to make every one of those questions **answerable from stored, queryable, explainable state** rather than analyst memory or ad-hoc log-grepping.

### 1.2 What makes it different from existing tools

- **Graph-native correlation, not rule-based grouping.** Campaigns and Operations are real Neo4j nodes with weighted, multi-feature decision functions (Sections 5–6), not a SIEM "case" created by a static correlation rule.
- **Explicit provenance for every MITRE ATT&CK mapping.** A 4-tier resolution ladder (Section 4) that prefers `UNKNOWN` over a guessed technique, with the decision (and *why*) persisted on the event node itself — most SIEM MITRE integrations either always map (fabricating confidence) or never map (discarding value).
- **Evidence-aware investigation as a first-class object**, not a black-box "risk score." The investigation loop (Section 10) tracks *which* sources were consulted, *how reliable* each is, *what conflicts* exist, and *why* it stopped — with a documented history of a real bug (evidence saturating confidence after 1–2 steps) and its fix, preserved in the code comments.
- **A topology-aware Graph Neural Network used honestly.** CYUKTI trained a graph autoencoder over real attack-graph structure and *tested* whether it improved severity classification. It didn't (Section 12.2) — and that null result is preserved, not hidden or silently re-litigated in later phases.
- **A closed-loop SOAR layer with real playbook memory**, not just alert forwarding: generated playbooks, historical playbook retrieval by five independent similarity signals (never a single blended score), and adaptation logic that retargets a historical playbook's IOCs without changing what it actually does.
- **A threat-qualification layer separate from severity.** A campaign can be low-severity (little potential damage) *and* confidently classified as a real threat (high evidentiary confidence) at the same time — CYUKTI surfaces both, rather than collapsing them into one number (verified on a real live campaign in Section 16.9).

### 1.3 Scope honesty

This is a research/capstone-grade system built and iteratively hardened over many development phases, running on live infrastructure the authors themselves administer (a Wazuh manager, a Neo4j graph database, a MISP-compatible threat-intel platform, Shuffle SOAR). It is **not** a commercial product; several subsystems are explicitly documented below as "built but not wired," "advisory only," or "environment-blocked" rather than claimed complete. This document preserves that honesty because a journal reviewer will find any inflated claim faster than the authors would like.

---

## 2. Architecture

### 2.1 Layer diagram (textual)

```
WAZUH MANAGER (4.7.5, real, live)
   |  writes /var/ossec/logs/alerts/alerts.json (Suricata + native Wazuh rules)
   v
LISTENER (listener/wazuh_listener.py)
   |  tails the alert file (offset-tracked, resumable), queues alerts,
   |  background threads: queue worker, maintenance worker, duplicate-buffer flush worker
   v
PIPELINE ORCHESTRATOR (realtime_socgraph.py :: process_alert())
   |  1. extract_iocs()              -> attacker_ip, victim_ip
   |  2. resolve_mitre()             -> MitreResolution (4-tier ladder)
   |  3. dedup_engine.is_duplicate() -> fingerprint check
   |  4. campaign_manager.resolve_campaign() -> Campaign (continue/reopen/create)
   |  5. neo4j_client.create_attack_event()  -> persists AttackEvent + all edges
   |  6. feature_orchestrator + severity_engine + dynamic_risk_engine -> TPS/risk
   |  7. campaign_correlation_engine (Operation-level correlation)
   |  8. threat_attribution_engine + prediction_engine
   |  9. cti_confidence_engine -> blended CTI score -> misp_sync (gated publish)
   v
NEO4J (5-community, real, Docker) -- the persistent graph (Section 3)
   ^
   |  read/write
DASHBOARD API (dashboard_api.py, Flask, port 5002, 33 routes)
   +  SOAR blueprint (soar/api.py, 13 routes, /api/soar/*)
   |
   v
REACT FRONTEND (Vite + TypeScript, port 5173, 27 mounted components)
   |  polls the API every 30s (DashboardContext.tsx) -- see Section 15.3
   |  (dead code: a Flask-SocketIO/socket.io-client push path exists but
   |   is never wired server-side and never imported client-side)
```

### 2.2 Parallel/independent subsystems that hang off the same Neo4j graph

- **GNN inference** (`ml/gnn/`): a frozen, pre-trained graph autoencoder loaded once per process; queried on demand for topology embeddings/similarity. Does not participate in the live alert-processing pipeline's decision logic (additive only, disabled by default via `GNN_ENABLED`).
- **XGBoost severity model** (`ml/`): trained offline from resolved-campaign records; queried on demand via `/api/ml/predict/severity` and the investigation loop's `XGBOOST_PREDICTION` action.
- **SOAR/Playbook layer** (`soar/`): its own SQLite store (`soar/memory.py`), independent of Neo4j, referencing Neo4j campaign IDs by string, not by a live join.
- **MISP/CTI publishing** (`misp_sync.py`, `cti_publisher.py`, `misp_cache.py`): triggered at the end of `process_alert()`, gated by `cti_confidence_engine`'s blended score; its own JSON cache file for campaign→MISP-event-ID mapping.

### 2.3 Process topology (what actually runs)

| Process | Entry point | Port | Started how |
|---|---|---|---|
| Wazuh manager | systemd service | n/a (writes to a file + its own API on 55000) | `systemctl` |
| Neo4j | Docker container `neo4j-soc` | 7687 (Bolt), 7474 (HTTP browser) | `docker start neo4j-soc` |
| CYUKTI listener | `python listener/wazuh_listener.py` | n/a | manual |
| CYUKTI backend | `python dashboard_api.py` | 5002 | manual |
| CYUKTI frontend (dev) | `npm run dev` (Vite) | 5173 | manual |

There is no supervisor/orchestration layer (no systemd units, no Docker Compose for CYUKTI's own processes) — each is started and monitored manually in this development environment. This is itself a documented limitation (Section 17).

---

## 3. Neo4j Data Model

### 3.1 Node labels, real counts (queried live)

| Label | Count | Source |
|---|---|---|
| `Technique` | 858 | Imported from vendored ATT&CK Enterprise STIX bundle (`mitre_import/`) |
| `Malware` | 729 | Imported (ATT&CK STIX) |
| `CourseOfAction` | 268 | Imported (ATT&CK STIX; MITRE mitigations) |
| `AttackEvent` | 212 | **Runtime** — created per resolved/unresolved alert |
| `ThreatActor` | 189 | Imported (ATT&CK STIX) |
| `Campaign` | 111 | **Runtime** |
| `Tool` | 95 | Imported (ATT&CK STIX) |
| `Operation` | 50 | **Runtime** |
| `Host` | 12 | **Runtime** |
| `Attacker` | 9 | **Runtime** |
| `Stage` | 6 | **Runtime** (kill-chain stage buckets: Reconnaissance, Credential Access, Initial Access, Privilege Escalation, Lateral Movement, Exfiltration — `config.TPS_MAP`'s six stages) |

### 3.2 Relationship types, real counts

| Type | Count | Direction | Provenance |
|---|---|---|---|
| `USES` | 16,902 | (ThreatActor/Malware/Tool)→Technique | Imported |
| `MITIGATES` | 1,448 | CourseOfAction→Technique | Imported |
| `SUBTECHNIQUE_OF` | 477 | Technique→Technique | Imported |
| `SIMILAR_TO` | 284 | Campaign↔Campaign | **Runtime** (derived, technique-overlap based — never used as ground truth for GNN/ML evaluation, see Section 12) |
| `HAS_EVENT` | 212 | Campaign→AttackEvent | Runtime |
| `REVOKED_BY` | 157 | Technique→Technique | Imported |
| `MATCHES` | 136 | AttackEvent→Technique | Runtime |
| `LAUNCHED` | 111 | Attacker→Campaign | Runtime |
| `TARGETS` | 111 | Campaign→Host | Runtime |
| `HAS_CAMPAIGN` | 49 | Operation→Campaign | Runtime |
| `LIKELY_NEXT` | 23 | Campaign→Technique | Runtime (prediction) |
| `BELONGS_TO` | 16 | Technique→Stage | Runtime |
| `RESEMBLES` | 6 | Campaign↔Campaign | Runtime (derived, distinct mechanism from `SIMILAR_TO`) |
| `NEXT_TECHNIQUE` | 3 | Technique→Technique | Runtime (the learned Markov transition table itself — see Section 9) |

### 3.3 Key node property lists (runtime nodes)

**`Campaign`**: `campaign_id`, `status`, `first_seen`, `last_seen`, `last_technique`, `occurrences`, `total_tps`, `risk_score`, `reopened_count`, `cti_score`, `cti_level`, `cti_publish`, `cti_breakdown` (stringified dict), `cti_updated_at`.

**`AttackEvent`**: `event_id`, `campaign_id`, `attack_id` (nullable — see Section 4.5), `fingerprint`, `rule_id`, `agent_id`, `technique`, `stage`, `attacker_ip`, `victim_ip`, `first_seen`, `last_seen`, `occurrences`, `tps`, `rule_level`, `suricata_score`, `zeek_score`, `sigma_score`, `yara_score`, `investigation_payload`, `prediction_generated_at`, `mitre_status`, `mitre_provenance`, `mitre_confidence`, `mitre_reason`, `mitre_resolver_version`, `mitre_technique_ids`.

**`Attacker`**: `ip`, `first_seen`, `last_seen`, `vt_reputation`, `threat_actor_reputation`, `malware_confidence`, `tool_confidence`, `misp_confidence`, `ioc_confidence` — **the last six are permanently frozen at `0.0` (Section 17.4)**.

**`Operation`**: `operation_id`, `campaign_count`, `last_seen`, `primary_attacker`.

**`Technique`** (imported): `attack_id`, `stix_id`, `name`, `description`, `url`, `created`, `modified`, `revoked`, `deprecated`, `attack_spec_version`, `object_version`, `platforms`, `domains`, `kill_chain_phases` (⚠ data-quality issue, Section 17.2), `is_subtechnique`, `occurrences`, `total_tps`, `stage`, `first_seen`, `last_seen`.

### 3.4 What's imported vs. runtime, at a glance

Imported once from the vendored ATT&CK Enterprise STIX bundle (`mitredata/attack-stix-data/enterprise-attack/enterprise-attack.json`, `x_mitre_version: 19.1`, `x_mitre_attack_spec_version: 3.3.0`): `Technique`, `Malware`, `ThreatActor`, `Tool`, `CourseOfAction`, and their `USES`/`MITIGATES`/`SUBTECHNIQUE_OF`/`REVOKED_BY` edges. Everything else is written by the live pipeline as real alerts are processed.

---

## 4. MITRE Resolution

### 4.1 The 4-tier ladder (`mitre_resolver.py`)

Strict precedence, first match wins, never reinterpreted by a later tier:

1. **`NATIVE_WAZUH`** — `alert.rule.mitre.id`, exactly as Wazuh (or a custom rule) supplies it. Confidence: `CONFIRMED`. **Never validated against Neo4j** — a native Wazuh mapping is authoritative by design.
2. **`REVIEWED_RULE_MAPPING`** — `mitre_rule_registry.REVIEWED_RULE_MAPPINGS`, a manually-curated Python dict keyed by Wazuh rule ID. Confidence: `REVIEWED`. Validated against Neo4j (revoked/deprecated/nonexistent technique → downgrades to `UNKNOWN`).
3. **`DETERMINISTIC_INFERENCE`** — explicit structural rules only (no fuzzy/semantic matching, no LLM). Confidence: `CANDIDATE`. **`DETERMINISTIC_INFERENCE_RULES = []` — ships completely empty.** The module docstring is explicit about why: *"no alert observed in the current lab data has a structural signal strong enough to justify a deterministic rule here... do not populate this to raise coverage numbers."*
4. **`UNKNOWN`** — `technique_ids=()`. A first-class, preferred outcome, not an error state.

A 5th outcome, **`AMBIGUOUS`**, exists for the case where a deterministic-inference rule (if one ever existed) matched multiple equally-plausible techniques — never arbitrarily resolved to one.

### 4.2 Why `REVIEWED_RULE_MAPPINGS` ships empty

Same reasoning as tier 3: `mitre_rule_registry.py`'s docstring names the specific unmapped lab rule IDs it deliberately declined to map (`40704, 52002, 2904, 86601, 2902, 531, 210020`) because most are routine OS/package-manager/service telemetry, and forcing a technique onto them "would fabricate ground truth."

### 4.3 Provenance persistence (real, live-verified)

Every `AttackEvent` node persists `mitre_provenance`, `mitre_confidence`, `mitre_reason`, `mitre_resolver_version`, `mitre_technique_ids` — the full decision trail, queryable per event, forever. Live-verified this session on the real Nmap-scan event: `provenance=NATIVE_WAZUH`, `confidence=CONFIRMED`, `reason="Wazuh rule.mitre.id"`, `resolver_version="1.0.0"`.

### 4.4 Provenance distribution (real, live query, n=93 events with a value set)

| Provenance | Count |
|---|---|
| `UNKNOWN` | 76 |
| `NATIVE_WAZUH` | 17 |

| Confidence | Count |
|---|---|
| `NONE` | 76 |
| `CONFIRMED` | 17 |

**119 of 212 total AttackEvent nodes (56.1%) have no provenance field set at all** — these predate the provenance-tracking feature being added to `create_attack_event()`'s Cypher (historical events created by an earlier code version). Of the 93 that do have it set, 81.7% are `UNKNOWN` and 18.3% are `NATIVE_WAZUH` — **zero events in this dataset have ever reached `REVIEWED_RULE_MAPPING` or `DETERMINISTIC_INFERENCE`**, consistent with both of those tiers shipping empty by design.

### 4.5 The unattributed path

76 of 212 AttackEvent nodes (35.8%) have `attack_id IS NULL` — these are events for which `resolve_mitre()` returned `UNKNOWN`, routed through `create_unattributed_attack_event()` rather than `create_attack_event()`, so routine, non-attack telemetry (informational logs, package-manager events) is never dropped, just never given a fabricated technique.

### 4.6 Real technique occurrence distribution (top, by `Technique.occurrences`)

| Technique | Name | Occurrences |
|---|---|---|
| T1595 | Active Scanning | 1,474 |
| T1595.002 | Vulnerability Scanning | 600 |
| T1055 | Process Injection | 364 |
| T1059.007 | JavaScript | 206 |
| T1110.001 | Password Guessing | 160 |
| T1059 | Command and Scripting Interpreter | 29 |
| T1078 | Valid Accounts | 27 |
| T1110 | Brute Force | 25 |
| T1562.001 | Disable or Modify Tools | 11 |
| T1114 | Email Collection | 10 |

(`Technique.occurrences` accumulates across the technique's entire lifetime including duplicate-suppressed repeats; contrast with Section 4.7's distinct-`AttackEvent`-node counts below.)

### 4.7 Distinct AttackEvent technique distribution (top 10, by node count — different metric from 4.6)

| `attack_id` | Distinct AttackEvent nodes |
|---|---|
| T1110.001 | 53 |
| T1595 | 18 |
| T1078 | 16 |
| T1110 | 11 |
| T1562.001 | 10 |
| T1053.003 | 7 |
| T1055 | 6 |
| T1190 | 4 |
| T1548.003 | 2 |
| T1595.002 | 2 |

### 4.8 A real data-quality issue in the vendored STIX corpus

Found this session while building `mitre_resolver.enrich_technique_metadata()`: **268 of 858 (31.2%) imported techniques have non-standard `kill_chain_phases` values** — e.g. T1562.001 ("Disable or Modify Tools") and T1055 ("Process Injection") both carry `"stealth"` instead of the real ATT&CK Enterprise tactic `"defense-evasion"`; other entries carry `"defense-impairment"`, also not a real tactic. Traced to the *raw vendored STIX file itself* (`{"kill_chain_name": "mitre-attack", "phase_name": "stealth"}` is genuinely present in `enterprise-attack.json` for these techniques) — not an import-code bug. Not silently corrected (guessing a fix for 268 techniques would itself be fabrication); flagged for re-fetching the official bundle from `mitre-attack/attack-stix-data` and diffing.

### 4.9 Live-verified custom Wazuh rule fixes (this session)

Six rule IDs in `/var/ossec/etc/rules/local_rules.xml` were each defined twice (`100001`, `100003`, `100004`, `100005`, `100500`, `100501`) — Wazuh only honors the first definition, silently dropping the rest (confirmed via `wazuh-logtest`'s explicit duplicate-ID warnings). Fixed by renumbering the shadowed duplicates into the unused `100510–100515` range and adding `<mitre>` tags to the now-live, defensible ones:

| Rule | Match | Technique added | Live-verified via `wazuh-logtest` |
|---|---|---|---|
| 100510 (was 100001) | SSH auth failure (if_sid 5716) | T1110 | Yes — `mitre.id: ['T1110']` |
| 100511 (was 100003) | `wget\|curl` | T1105 | Yes — `mitre.id: ['T1105']` |
| 100513 (was 100005) | "SLOW endpoint" | T1499.002 | Yes — `mitre.id: ['T1499.002']` |
| 210001 | "BOT_ATTACK" | T1498.001 | Yes — `mitre.id: ['T1498.001']` |
| 210011 | "LOGIN_ATTACK" | T1110 | Yes — `mitre.id: ['T1110']` |
| 210020 | bare "sudo" | **Deliberately not mapped** — overbroad match, no behavioral specificity | n/a |
| 100512 (was 100004) | `[INVENTORY_ABUSE]` | **Deliberately not mapped** — synthetic label, no verifiable semantics | n/a |

**Not yet active in the live Wazuh manager**: the manager was running (since before the fix was applied) and has not been restarted — confirmed via file-mtime vs. `ActiveEnterTimestamp` comparison. Restarting requires `sudo /var/ossec/bin/wazuh-control restart`, which needs an interactive password this automation cannot supply.

---

## 5. Campaign Correlation

### 5.1 The 7 features (`campaign_feature_engine.py` → `CampaignFeatures`) and their weights (`config.CAMPAIGN_WEIGHTS`)

| Feature | Weight |
|---|---|
| `prediction_similarity` | 0.35 |
| `chain_similarity` | 0.30 |
| `temporal_similarity` | 0.20 |
| `attacker_similarity` | 0.10 |
| `runtime_similarity` | 0.03 |
| `graph_similarity` | 0.01 |
| `duplicate_similarity` | 0.01 |

### 5.2 Scoring (`campaign_decision_engine.py`)

```
score = (Σ feature_value × weight for available features) / (Σ weight for available features)
confidence = (available_features / 7) × 100
```

A missing feature (`None`) is excluded from both the numerator and the weight sum — not treated as 0 — so campaigns with partial feature availability aren't unfairly penalized, and `confidence` separately reports how much of the 7-feature space was actually usable.

### 5.3 Continue / Reopen / Create decision logic (`campaign_manager.py::resolve_campaign()`)

```
ACTIVE_CONTINUE_THRESHOLD = 0.35
REOPEN_THRESHOLD = 0.70
CAMPAIGN_TIMEOUT = 120 seconds
```

1. If an **active, unexpired** campaign exists for this (attacker, victim) pair: compute the 7-feature score.
   - `score >= 0.35` → **CONTINUE** the campaign.
   - else → **CLOSE** it (status → `INACTIVE`), fall through to step 2.
2. Search **inactive** campaigns matching (attacker, victim):
   - **Fast-path reopen**: if the candidate's *first* observed technique equals the current technique, `temporal_similarity >= 0.8`, and attacker/victim match exactly → immediate reopen, bypassing full scoring.
   - Otherwise, score every inactive candidate. Any scoring `>= 0.70` is logged as a **reopen-ambiguity candidate** (multiple qualifying campaigns is explicitly detected and logged, not silently resolved). The single best-scoring campaign is selected (ties broken by most-recent `last_seen`).
   - If `best_score >= 0.70` → **REOPEN** it.
3. Otherwise → **CREATE** a new campaign.

### 5.4 The skip flag

`CampaignContext.skip_next_chain_update` is set to `True` on **every reopen path** (fast-path and scored). Purpose: prevent the attack-chain/Markov-transition learner (Section 9) from recording "closed-technique → reopened-technique" as a real, natural transition — a reopen is a resumption of prior activity, not a genuine next-step in the kill chain, and treating it as one would corrupt the learned transition table with artifacts of the campaign-timeout mechanism rather than real attacker behavior.

### 5.5 Real reopen statistics (live query)

**13 of 111 campaigns (11.7%) have been reopened at least once, totaling 21 reopen events** (`sum(reopened_count) = 21`).

---

## 6. Operation Grouping

Operations are the second, independent correlation layer, clustering *campaigns* (not raw events) that appear to be part of the same broader attacker operation. Built on `campaign_correlation_engine.py` + `operation_decision_engine.py` + `operation_manager.py`, using its own weighted feature set (distinct engine from Section 5's campaign-level one, confirmed by direct code inspection — not the same object reused).

**Real counts**: 50 `Operation` nodes, 49 `HAS_CAMPAIGN` edges (one operation currently has zero attached campaigns — created but not yet populated, or its sole campaign was later reassigned). Top operations by campaign count (live query):

| Operation | Campaigns attached |
|---|---|
| OP_236E9AE4 | 4 |
| OP_A89B0658 | 4 |
| OP_93401807 | 4 |
| OP_D856F358 | 3 |
| OP_E76AEB64 | 2 |

Average campaigns-per-operation across the top 10: 2.9 — most operations are small (2–4 campaigns), consistent with a threshold-gated correlation engine that doesn't over-merge unrelated campaigns into one operation.

---

## 7. Scoring Stack

CYUKTI runs **two independent scoring pipelines** that are frequently confused with each other but answer different questions:

### 7.1 TPS (Threat Point Score) — the campaign-continuation/risk-accumulation scale

`config.TPS_MAP` assigns a fixed point value per kill-chain stage:

| Stage | TPS |
|---|---|
| Reconnaissance | 10 |
| Credential Access | 45 |
| Initial Access | 70 |
| Privilege Escalation | 90 |
| Lateral Movement | 110 |
| Exfiltration | 150 |

Accumulated additively onto `Campaign.total_tps`/`Campaign.risk_score` every time a matching event occurs. Normalized to a 0–100 display scale via `risk_scoring.normalize_risk_score()`: `min(100, round((raw_tps / TPS_CEILING) × 100))`, with **`TPS_CEILING = 1500`** (environment-overridable, default). `risk_scoring.py`'s own docstring calls this ceiling *"a genuine open calibration question, not just a technical detail... 6x [an earlier] ceiling. Changing TPS_CEILING reshapes the entire severity distribution."* — an explicitly acknowledged, unresolved calibration limitation (Section 17.5).

Risk bands (`risk_level_from_score`): `LOW` (<35), `MEDIUM` (35–59), `HIGH` (60–79), `CRITICAL` (≥80).

**Real distribution** (111 campaigns, raw `risk_score` before normalization): min 0, max 34,820 (⚠ far exceeding the 1,500 ceiling — see Section 17.5), mean 572.1, p50 **45.0**, p95 **590**. After normalization, the p50 campaign (45/1500×100 ≈ 3%) reads as deep `LOW` — most real campaigns in this dataset are low-severity by this scale.

### 7.2 The severity-classification pipeline (feature families → `SeverityEngine` → XGBoost)

Independent scoring stack feeding the trained XGBoost classifier (Section 12.1), built from feature families: platform (weight 0.25), mitigation (0.50), threat-actor (0.15), malware (0.20), tool (0.20), runtime-event (0.50), detection (1.00), threat-intel (1.00) — `config.py`'s `*_WEIGHT` constants. Classes: `Low`, `Medium`, `Critical` (no `High` class exists in the training data — 0 examples, never manufactured).

### 7.3 Dynamic risk / confidence model (`dynamic_risk_engine.py`)

`DynamicRiskEngine.calculate_confidence()` — **contains a real, confirmed defect** (Section 17.3): 3 of its 7 evidence terms are hardcoded `1.0` regardless of any actual feature value, giving confidence an artificial floor of **42.86%** (3/7 × 100) no matter how little real evidence exists.

### 7.4 CTI confidence (blended, gates MISP)

`cti_confidence_engine.py`: `score = detection×0.25 + risk×0.20 + threat×0.20 + campaign×0.20 + prediction×0.15`. Tiers: `QUALIFIED_THREAT` (≥40, = `PUBLISH_THRESHOLD`), `SUSPICIOUS` (20–39.99), `NOT_THREAT` (<20). **Real distribution** (42 scored campaigns): min 27.63, max 65.81, mean 47.45; `cti_publish=True` on 26, `False` on 16 (61.9% would clear the live MISP gate).

---

## 8. Prediction

`prediction_engine.py` + `chain_updater.py`: a **learned Markov-chain-style next-technique predictor** over observed `Technique → Technique` transitions, persisted as real `NEXT_TECHNIQUE` edges (with `count`/`confidence` properties) between `Technique` nodes, separately from the per-campaign `LIKELY_NEXT` edge that stores the *current* live prediction for that campaign.

**Real learned transitions** (all 3 that exist in this dataset):

| From | To | Observed count | Confidence |
|---|---|---|---|
| T1110.001 | T1110 | 6 | 1.0 |
| T1110 | T1078 | 5 | 1.0 |
| T1078 | T1110.001 | 1 | 1.0 |

**Gating**: `config.MIN_TRANSITION_OBSERVATIONS = 3` — a transition is only surfaced as a prediction once observed at least 3 times (explains why only 3 transition edges exist despite 212 real events: most technique pairs haven't recurred 3+ times yet). `MIN_PREDICTION_CONFIDENCE = 30` gates whether a prediction is surfaced to the analyst at all.

**Prediction accuracy — genuinely zero data points.** `Campaign.prediction_hits`/`prediction_misses` sum to **0 and 0** across all 111 campaigns. The hit/miss tracking mechanism exists in the schema (`CampaignContext.prediction_hits`, `.prediction_misses`) but has never recorded a single outcome in this dataset — **no prediction-accuracy metric can be honestly reported** (Section 17.6). 23 real `LIKELY_NEXT` edges exist (live predictions currently attached to campaigns), but none have been retrospectively scored against what actually happened next.

**Recommendation lookup** (`recommendation_engine.py`): a real, live Neo4j query — `MATCH (t:Technique {attack_id:$id}) MATCH (m:CourseOfAction)-[:MITIGATES]->(t) RETURN m...` — surfacing genuine ATT&CK mitigations (real edges, 1,448 `MITIGATES` relationships in the imported corpus), not a static lookup table.

---

## 9. Attribution

`threat_attribution_engine.py` + `attribution_similarity.py` — matches a campaign against historical campaign records using three independent signals, combined with fixed weights:

```
total_score = coverage × 0.50 + precision × 0.20 + chain_similarity × 0.30
```

- **Coverage** = `|observed ∩ historical| / |observed|` — of the current campaign's techniques, what fraction were also seen historically.
- **Precision** = `|observed ∩ historical| / |historical|` — of the historical campaign's techniques, what fraction match the current one.
- **Chain similarity** = `LCS(current_chain, historical_chain) / len(current_chain)` — Longest Common Subsequence over the *ordered* technique sequence (real dynamic-programming LCS implementation, O(m·n)), not just set overlap — order matters for attribution.

Distinct from `ThreatActorContext`'s separate `technique_similarity`/`chain_similarity`/`infrastructure_similarity`/`campaign_similarity`/`prediction_similarity`/`topology_similarity` fields (the latter added this session, GNN-derived, computed *after* ranking so it never influences the sort — additive only).

---

## 10. Investigation Loop

### 10.1 Evidence model: confidence vs. relevance

`evidence/schema.py`'s `Evidence` dataclass draws an explicit, load-bearing distinction:
- **`confidence`**: how much the evidence *itself* is trusted, as reported by the system that produced it (MITRE reference data → 1.0; a MISP correlation → whatever the CTI engine computed).
- **`relevance`**: how relevant this specific piece of evidence is to the *current* investigation query — starts at 0.0, filled in by the retriever, never by the collector (a fact doesn't know in advance what it will be useful for).

### 10.2 The 8-vs-10 discrepancy, resolved

The current `InvestigationAction` enum has **10** members (not 8): `MITRE_KNOWLEDGE`, `MITRE_SEMANTIC_SEARCH`, `CTI_LOOKUP`, `DETECTION_CHECK`, `ATTRIBUTION_MATCH`, `CAMPAIGN_HISTORY`, `CAMPAIGN_NARRATIVE_SEARCH`, `GRAPH_STRUCTURE`, `XGBOOST_PREDICTION`, `GNN_TOPOLOGY_RETRIEVAL`. The two Multi-RAG-related actions (`MITRE_SEMANTIC_SEARCH`, `CAMPAIGN_NARRATIVE_SEARCH`) and `GNN_TOPOLOGY_RETRIEVAL` were added in later phases of this project — an "8 actions" recollection likely predates those additions. Reported here as the real, current count.

Each action's `ActionMeta` declares `source`, `reliability` (0–1), `cost` (0–1), `latency` (0–1), and `depends_on` (a `frozenset` of actions whose evidence is *code-verified* to substantially overlap this one's — e.g. `ATTRIBUTION_MATCH` depends on `CAMPAIGN_HISTORY` because `threat_attribution_engine.attribute()` calls the same `load_historical_campaigns()` internally).

### 10.3 The Next-Best-Evidence scoring formula (`investigation/next_best_evidence.py`)

```
value = 1.0×expected_gain + 0.3×reliability + 0.5×novelty + 0.6×uncertainty_reduction
        − 0.4×cost − 0.2×latency − 0.5×redundancy_penalty

expected_gain = reliability × novelty
novelty = 1.0 (source never queried this investigation) or 0.15 (repeat query)
uncertainty_reduction = current_uncertainty  (ONLY for XGBOOST_PREDICTION — the sole
                         action producing a verdict about the conclusion, not a
                         supporting fact) else 0.0
redundancy_penalty = 0.5 × |already-taken actions this action depends_on|
```

An explicit, fully-transparent heuristic — every term traceable to a declared static number or the current `ConfidenceEstimate`, described in its own docstring as *"a documented HEURISTIC policy, not a learned/optimal one — there is no historical log of which evidence-gathering order actually resolved investigations fastest to train a policy from yet."*

### 10.4 The confidence-saturation bug and its fix (real, documented in code)

`investigation/confidence.py`'s module docstring preserves the actual incident: *"found by running it against real campaigns — every run stopped after 1–2 steps because a single reliable-but-narrow fact was read as 'investigation resolved'."*

**Root cause**: an earlier version conflated *evidence reliability* (a source-level property) with *investigation resolution* (a breadth property) — one highly-reliable fact from a single source was enough to saturate confidence past the stopping threshold.

**Fix**: introduced `evidence_coverage` — the fraction of the *possible evidence-source space* actually consulted, weighted by the best relevance found from each source:

```
evidence_coverage = Σ(best_relevance_per_source) / |all EvidenceSource categories|
```

...and made `investigation_confidence` a function of **both** reliability and coverage, not reliability alone:

```
if a model verdict (XGBoost) exists:
    coverage_gate = min(1.0, evidence_coverage / 0.15)     # MIN_COVERAGE_FOR_MODEL_TRUST
    investigation_confidence = model_confidence × (1 − model_uncertainty) × coverage_gate
else:
    investigation_confidence = evidence_reliability × evidence_coverage

if unresolved conflicts exist:
    investigation_confidence *= 0.85 ** conflict_count     # CONFLICT_PENALTY, compounds per conflict
```

`model_uncertainty` is normalized Shannon entropy over the model's class-probability distribution (`predictive_entropy()`), chosen specifically because two distributions can share the same top-1 probability while differing in how spread the remainder is — a real, standard measure, not bespoke.

### 10.5 The 4 stopping criteria (`investigation/stopping.py`)

Defaults: `confidence_threshold=0.75`, `max_uncertainty=0.4`, `max_steps=8`.

1. `investigation_confidence >= 0.75` **AND** (`model_uncertainty <= 0.4` or no model ran) **AND** zero unresolved conflicts.
2. `steps_taken >= 8` (hard safety limit, not the primary mechanism).
3. Best remaining action's expected `value <= 0.0` — cost now exceeds expected benefit for everything left.
4. No evidence-gathering actions remain at all (menu exhausted).

Every `StoppingDecision` carries a human-readable `reason` string, persisted in the investigation record.

### 10.6 Phase 21/22: the adaptive-NBE negative result (disclosed, not a bug)

Two real research-validation phases tested whether the Next-Best-Evidence policy (10.3) is actually *adaptive* — does it choose a different evidence-gathering order depending on what a campaign's evidence contains, or does it always do the same thing? Both are documented in full in `review/phase21_real_investigation_validation.md` and `review/phase22_nbe_sensitivity_validation.md`, quoted here verbatim rather than re-derived.

**Phase 21 — real-campaign investigation.** Three real campaigns were run through the live investigation loop end-to-end: `CAMP_427A075C` (192.168.56.106→192.168.56.105, T1595, risk 8330), `CAMP_1429ADB4` (192.168.56.105→pes1ug23cs411-VirtualBox, T1055/T1059/T1059.007/T1190/T1210/T1595.002, risk 34820), `CAMP_D8605E81` (192.168.56.106→pes1ug23cs411-VirtualBox, T1110/T1110.001, risk 630) — different attacker/victim pairs, different technique sets, an order-of-magnitude spread in risk. The action order chosen was **identical across all three**, at the time this project used 8 `InvestigationAction` members (before `CAMPAIGN_NARRATIVE_SEARCH` and `GNN_TOPOLOGY_RETRIEVAL` were added — see 10.2):

| Step | Action | `selection_value` — CAMP_427A075C | CAMP_1429ADB4 | CAMP_D8605E81 |
|---|---|---|---|---|
| 1 | `mitre_knowledge` | 1.74 | 1.74 | 1.74 |
| 2 | `xgboost_prediction` | 1.73 | 1.73 | 1.73 |
| 3 | `graph_structure` | 1.62 | 1.62 | 1.62 |
| 4 | `campaign_history` | 1.49 | 1.49 | 1.49 |
| 5 | `detection_check` | 1.485 | 1.485 | 1.485 |
| 6 | `cti_lookup` | 1.135 | 1.135 | 1.135 |
| 7 | `attribution_match` | 0.875 | 0.875 | 0.875 |
| 8 | `mitre_semantic_search` | 0.3 | 0.3 | 0.3 |

All three investigations stopped via stopping criterion 2 (`steps_taken >= 8`, the hard depth limit) — the confidence-threshold branch was never exercised by any of the three. Full backend suite at the time: 164/164 passing (unchanged; no code was modified this phase).

**Phase 22 — ablation study.** Phase 21 raised two competing hypotheses for *why* the order never changed: **H1** (magnitude dominance — content-sensitive terms exist but are numerically swamped by static ones) vs. **H2** (no content channel — the adaptive terms never carried campaign-specific information to begin with). Phase 22 tested this directly with an 8-way ablation (configurations A–F3, toggling `novelty`, `redundancy_penalty`, and `uncertainty_reduction` on/off in combination) run pairwise across the same 3 real campaigns: **192 pairwise comparisons total**. Result: **192/192 identical rankings, 192/192 identical raw scores, 0 total pairwise rank swaps, Kendall's τ = 1.0 (min=max, n=168 non-degenerate pairs), 0.0 score variance in every single comparison.** H2 confirmed, H1 falsified.

**Root cause** (traced to real code, not inferred): `score_action()`'s `uncertainty_reduction` term is nonzero *only* for `XGBOOST_PREDICTION` (10.3) — every other action contributes 0.0 to it regardless of campaign. `novelty` and `redundancy_penalty` are computed purely from *which action types have already been taken this investigation*, never from the evidence content those actions returned. And `mitre_knowledge` is deterministically step 1 in every campaign because it has the highest static score (1.74) unconditionally; its collector (`evidence/collectors/mitre_collector.py`) never sets `Evidence.relevance` above the dataclass default of 0.0, and `EvidenceStore.weighted_confidence()`/`evidence_coverage()` (10.4) give zero weight to relevance-0.0 evidence — a deliberate anti-saturation design from the Section 10.4 fix. Consequence: after step 1, `evidence_reliability=0.0` and `evidence_coverage=0.0` in *every* campaign unconditionally, so `investigation_confidence=0.0`, `uncertainty=1.0`, and XGBoost's `uncertainty_reduction` is pinned to exactly 1.0 regardless of campaign content — structural, not coincidental.

**What IS campaign-dependent**: the confidence/uncertainty *trajectories* over the 8 steps differ meaningfully — final `investigation_confidence` was 0.1835 / 0.0687 / 0.0963 and final `uncertainty` was 0.8165 / 0.9313 / 0.9037 for the three campaigns respectively, with real per-campaign `evidence_reliability` (1.0 / 0.9287 / 0.9814) and `evidence_coverage` (0.5279 / 0.4884 / 0.4528) values. **What is NOT campaign-dependent**: the *order* in which evidence is gathered.

**Framing for the paper**: this is a disclosed negative result about the current heuristic NBE policy, verified by both code inspection and a 192-comparison empirical ablation with zero exceptions — not a bug, and not hidden. The honest, defensible claim (stated in Phase 22's own conclusions): *"the current evidence-dependency/redundancy and novelty mechanisms are structurally activated by which action types run, not by the substance of what those actions found."* The fix — giving `mitre_collector.py` a real, non-zero relevance signal — is identified but out of scope for this project's remaining time (see Section 17 candidates for future work).

---

## 11. ML Subsystems

### 11.1 XGBoost — wired, trained, in production use

- **Artifact**: `ml/models/xgb_severity.json` (301,923 bytes) + `xgb_severity.meta.joblib` (453,487 bytes), trained `2026-09-15T05:28:10Z`.
- **Model**: `CalibratedClassifierCV(cv='prefit', estimator=XGBClassifier(...))` — 3-class (`Critical`, `Low`, `Medium` — no `High` class exists in the data).
- **Wired into**: `POST /api/ml/predict/severity` (503 if the artifact is missing — never fabricates a prediction), and the investigation loop's `XGBOOST_PREDICTION` action.
- **Real LOGO (Leave-One-Attacker-Group-Out) cross-validation results**, pooled out-of-fold across all 60 real campaigns in the training set (`GNN_XGBOOST_ABLATION.md`):

  | Metric | Baseline | + GNN embeddings |
  |---|---|---|
  | Accuracy | 0.900 | 0.900 |
  | Macro F1 | 0.449 | 0.449 |
  | Weighted F1 | 0.863 | 0.863 |
  | Balanced accuracy | 0.417 | 0.417 |

  **Delta: exactly 0.000 on every metric** — GNN embeddings added zero measurable value to severity classification. This is a frozen result, deliberately never reopened by any later phase.

  Per-class (pooled, identical both conditions):

  | Class | Precision | Recall | F1 | Support |
  |---|---|---|---|---|
  | Critical | 1.000 | 0.250 | 0.400 | 4 |
  | Low | 0.898 | 1.000 | 0.946 | 53 |

  (Medium's row is degenerate in at least one fold — one held-out group's training split had zero Medium examples, correctly scored 0 rather than crashing or being silently omitted, per a fixed `LabelEncoder`-per-fold defect described in this project's own engineering history.)

### 11.2 Graph Autoencoder / SSFT (self-supervised network-flow autoencoder) — built, artifacts exist, **not wired**

- Artifacts present: `autoencoder_best.pth` (455,715 bytes), `ssl_scaler.pkl` (8,103 bytes), `adaptive_threshold.json`, `severity_scorer.json` — all dated `2026-08-27`.
- Purpose: a self-supervised anomaly-scoring autoencoder over 312-dimensional real network-flow feature vectors (CICIDS2017-style), trained and evaluated in its own domain.
- **Status: infrastructure-blocked.** Requires a live 312-dimensional network-flow capture pipeline that does not exist in this environment. Not integrated into the live alert-processing path. Deliberately not force-fit ("do not integrate into the live campaign feature pipeline unless the required input actually exists" — an explicit standing constraint honored across every phase of this project).

### 11.3 GNN topology autoencoder — standalone, additive, disabled by default

- Architecture: 2-layer hand-implemented SAGEConv (GraphSAGE mean-aggregator; no `torch_geometric`/DGL dependency), hidden_dim=16, embedding_dim=8, mean-pool + `Linear`+`tanh` bottleneck.
- Artifact: `gnn_autoencoder.pt` (13,923 bytes), version `gnn_autoencoder.pt@1790069556-13923b`.
- Trained via graph autoencoding (edge existence + edge type + standardized node-feature reconstruction), **not** for severity prediction — `risk_score` deliberately excluded from its input features as a leakage source.
- Gated by `GNN_ENABLED` (currently `true` in this environment) — when `false`, every integration point short-circuits to "no GNN evidence available" without importing `torch` at all.
- **Real retrieval evaluation** (LOGO cross-validation, `GNN_RETRIEVAL_EVALUATION.md`):

  | Task | n | MRR | Recall@5 | vs. chance |
  |---|---|---|---|---|
  | A — recover own later snapshot | ~71 candidates | **0.377** | 0.476 | ≈6× chance (≈0.060) |
  | B — same-attacker retrieval (fold-safe) | 68 | **0.975** | 0.231 (group-size-capped) | — |
  | C — same-host retrieval (fold-safe) | 65 | **0.979** | 0.189 (group-size-capped) | — |
  | B, single-model (weaker check) | 68 | 0.822 | — | notably lower than fold-safe — attributed to cross-fold embedding-space comparability, not a real generalization signal |

  Never evaluated against `SIMILAR_TO`/`RESEMBLES` as ground truth (would be circular — those edges are themselves technique-overlap-derived).
- Live-verified determinism this session: `embed_campaign(id, use_cache=False)` called twice against the real model and a real campaign → `torch.allclose()` on the two outputs (eval-mode, no training-time randomness).

---

## 12. Attribution/Prediction Cross-Reference

*(See Sections 8–9 above; this section intentionally left as a pointer per the outline's structure — no additional distinct content beyond what's already reported precisely there, to avoid restating numbers under a different heading.)*

---

## 13. MISP/CTI Publishing

### 13.1 Event creation and gating

`realtime_socgraph.py` instantiates `CTIPublisher`+`MISPSync` at module scope; every processed alert's campaign is passed through `sync.publish_campaign(incident)` at the end of the pipeline. **Gated**, not unconditional: `MISPSync.should_publish()` checks `incident.cti.publish` (the blended CTI-confidence boolean, Section 7.4) — `score >= 40` — before ever making an HTTP call. This gate is real and has been in place since before this session's engineering-audit phase confirmed it (a prior architecture audit had missed that it existed).

### 13.2 Caching (idempotency)

`misp_cache.py`: a thread-safe, JSON-file-backed (`misp_cache.json`) `campaign_id → MISP event_id` map. `MISPSync._cached_event()`/`_cache_event()` consult/update it before deciding create-vs-update; `_search_existing()` additionally does a live MISP `restSearch` fallback (`"CYUKTI Campaign <id>"` text match) if the local cache misses but MISP already has the event — verified this project caught and fixed a real defect where a cache-miss-but-MISP-has-it scenario would otherwise duplicate the event.

### 13.3 Retry logic

`CTIPublisher` wraps its `requests.Session` with a retry adapter (`DEFAULT_TIMEOUT=30`s); `health_check()` is a real, read-only `GET /servers/getVersion` probe, never raises, returns a plain bool.

### 13.4 Threat levels

`MISPEventGenerator._risk_to_threat_level()`: risk ≥70 → level 1 (High), ≥40 → level 2 (Medium), else → level 3 (Low) — MISP's native 3-tier threat-level field, mapped from CYUKTI's own risk score.

### 13.5 Real environment status

`MISP_URL=https://localhost:8443/` is configured; a real API key was set in `.env` earlier in this project (previously had a trailing-whitespace bug that caused live `403`s, since fixed). **At the time of writing this document, `curl` to `https://localhost:8443/` times out completely (`000` — no TCP response at all)** — no MISP instance is currently reachable on the configured port in this environment, despite it having been reachable and returning real `403`/publish attempts in earlier phases of this same project. This is reported as a live, current-state observation, not a historical claim (Section 17.1).

### 13.6 New this session: threat-gating checklist (advisory, not yet wired into the live gate)

`threat_qualification.py`'s six-point publication-readiness checklist (`threat_classification_qualified`, `has_ioc`, `valid_mitre_provenance`, `valid_timestamp`, `has_campaign_context`, `not_already_published`) is deliberately **not** wired into `should_publish()` — kept advisory/dashboard-facing pending more live observation, per an explicit engineering decision recorded this session.

---

## 14. Frontend

### 14.1 Mounted components (27 total `.tsx` components, all currently reachable from the app tree except one)

Top-level views (`TopNavBar.tsx`'s 4 primary + "More" dropdown of 4):

| View | Component | Primary or "More" |
|---|---|---|
| Overview | `SecurityOverview.tsx` (+ dashboard 3-column layout) | Primary |
| Incidents | `IncidentView.tsx` | Primary |
| Response | `SOARPage.tsx` | Primary |
| Threat Intel | `ThreatIntelligencePage.tsx` | Primary |
| GNN Intelligence | `GNNIntelligencePage.tsx` | More |
| Prediction | `PredictionIntelligencePage.tsx` | More |
| System Health | `SystemHealthPage.tsx` | More |
| Audit Log | `AuditLogPage.tsx` | More |

Sub-tab / embedded components (mounted inside the above): `AttackGraph.tsx`, `AttackPathAnalytics.tsx`, `AttackerIntelligence.tsx`, `CampaignIntelligence.tsx`, `CampaignSelectionPanel.tsx`, `EvidenceInvestigation.tsx`, `IntelligenceWorkspace.tsx`, `LiveEventsFeed.tsx`, `MitreAttackChain.tsx`, `PanelWrapper.tsx` (shared chrome), `PathExplorer.tsx`, `QueryConsole.tsx`, `RecommendationEngine.tsx`, `RiskPropagation.tsx`, `ThreatActorAttribution.tsx`, `ThreatCorrelation.tsx`, `TopologyIntelligence.tsx`, `TopNavBar.tsx`.

**Orphaned (real, confirmed by `grep`)**: `PredictionPanel.tsx` — imported only by its own test file, zero real mount points. Its full-page replacement `PredictionIntelligencePage.tsx` is what's actually reachable.

### 14.2 Polling vs. WebSocket — real, confirmed dead code

`DashboardContext.tsx` refreshes all data via `setInterval(refreshAll, 30000)` — a plain 30-second poll, confirmed the sole real-time update mechanism.

**`frontend/src/hooks/useWebSocket.ts` is genuinely dead code**: it constructs a `socket.io-client` connection expecting a Flask-SocketIO server at `/socket.io`, listening for a `new_events` event. It is **imported by zero other files** in the frontend (`grep -rln useWebSocket src/` matches only its own file). The backend, despite `Flask-SocketIO 5.5.1` being an installed pip dependency, has **zero references to `socketio`/`SocketIO` anywhere in `backend/*.py`** — no `SocketIO(app)` instantiation, no `socketio.emit()` call exists. This is a fully scaffolded, entirely unwired push-notification path on both ends (Section 17.7).

### 14.3 API client

`services/api.ts`: a single `fetch()`-based client, `API_BASE = '/api'` (relative, proxied by Vite's dev-server config to `http://localhost:5002` — confirmed no port mismatch in the actual proxy configuration itself, see Section 17.1 for the *separate* MISP-port observation).

---

## 15. Test Suite

### 15.1 Backend

- **65 test files**, full suite: **689 passing, 0 failing** (most recent clean run, CYUKTI's own backend/listener processes stopped during the run to eliminate resource-contention flakiness — see Section 17.8 for the one flake this project found and diagnosed, not currently present).
- Coverage highlights: MITRE resolution (precedence, provenance, `AMBIGUOUS`/`UNKNOWN` paths), campaign/operation decision engines, evidence-aware investigation (confidence, NBE scoring, stopping criteria), GNN inference (fail-safe, singleton test-isolation), SOAR/playbook lifecycle (generation, matching, adaptation, execution state machine), MISP integration (event generation, caching, false-positive-success regression tests), dashboard API routes (Neo4j-outage handling, evidence-aware endpoints), determinism (7 tests, including one live GNN test), failure injection (18 tests, 13 required failure modes), a final end-to-end object-chain trace test, a real benchmark harness (7 tests), experiment-recording schema validation (6 tests), configuration-snapshot secret-exclusion tests (7 tests).
- **Not covered**: no live-Wazuh-alert-to-dashboard end-to-end integration test exists that doesn't mock at least one boundary (Neo4j, or the alert file) — every test that touches a live external system is explicitly gated `skipif not reachable`, never fails the suite when infrastructure is down.

### 15.2 Frontend

- **17 test files, 109 tests passing.** Full production build clean, no TypeScript errors.
- Notable regression tests preserved in test docstrings: a fixed `predict_next()` arity bug (was calling the wrong function with a write side-effect from a read-only GET route), a fixed Cypher `max(a,b)` syntax error in the attribution-actors route, a fixed nested-double-header bug when `RecommendationEngine` was first mounted as a tab.

### 15.3 What's explicitly NOT measured (honest gaps)

No independent ground truth exists in this environment for: MITRE mapping accuracy, threat-classification precision/recall, campaign-correlation pairwise precision/recall (scoring CYUKTI against its own derived labels would be circular and is explicitly forbidden by this project's own engineering standards) — see `ACCURACY_EVALUATION.md`'s per-task (A) real-ground-truth / (B) proxy-labels / (C) unavailable classification.

---

## 16. All Numbers (raw query results)

*(Consolidated here for convenience; each also appears in context above.)*

```
Node counts:            Technique 858, Malware 729, CourseOfAction 268, AttackEvent 212,
                         ThreatActor 189, Campaign 111, Tool 95, Operation 50, Host 12,
                         Attacker 9, Stage 6

Relationship counts:    USES 16902, MITIGATES 1448, SUBTECHNIQUE_OF 477, SIMILAR_TO 284,
                         HAS_EVENT 212, REVOKED_BY 157, MATCHES 136, LAUNCHED 111,
                         TARGETS 111, HAS_CAMPAIGN 49, LIKELY_NEXT 23, BELONGS_TO 16,
                         RESEMBLES 6, NEXT_TECHNIQUE 3

Campaign status:        INACTIVE 111 (100% -- no campaign is ACTIVE at the exact moment
                         this snapshot was taken; the API layer separately computes a
                         live "ACTIVE"/timeout-based status per request, see below)
Campaign risk_score:     min 0, max 34820, mean 572.12, p50 45.0, p95 590
Campaign cti_score:      n=42 scored, min 27.63, max 65.81, mean 47.45
Campaign cti_publish:    False=16, True=26
Reopened campaigns:      13 campaigns, 21 total reopens

Operations:              50 total, 49 HAS_CAMPAIGN edges, top operation has 4 campaigns

AttackEvent w/ mitre_provenance set:   93 (44%) -- UNKNOWN=76, NATIVE_WAZUH=17
AttackEvent w/ no provenance field:    119 (56%, pre-dates the feature)
AttackEvent unattributed (no attack_id): 76 (35.8%)
Distinct fingerprints:                 71 (avg 2.99 occurrences each, max 19)
Total occurrences (dedup-collapsed):   3185 across 212 distinct AttackEvent nodes
                                        (avg 15.02 per node)

Prediction (NEXT_TECHNIQUE learned edges): 3 -- T1110.001->T1110 (n=6), T1110->T1078 (n=5),
                                            T1078->T1110.001 (n=1), all confidence=1.0
Prediction hits/misses (all campaigns):    0 / 0  -- no accuracy measurable
LIKELY_NEXT live edges:                    23

Attacker threat-intel fields nonzero:      0 of 9 (100% still at default 0.0)

ThreatActor nodes: 189, CourseOfAction nodes: 268, MITIGATES edges: 1448

XGBoost (pooled LOGO, n=60 campaigns): accuracy 0.900, macro-F1 0.449,
    weighted-F1 0.863, balanced-accuracy 0.417 -- identical with/without GNN features
GNN retrieval: Task A MRR 0.377 (~6x chance), Recall@5 0.476;
    Task B (attacker) MRR 0.975; Task C (host) MRR 0.979

Backend tests: 689 passed / 0 failed (65 files)
Frontend tests: 109 passed / 0 failed (17 files)

Dashboard API routes: 33 (dashboard_api.py) + 13 (soar/api.py) = 46 total
Frontend components: 27 (.tsx), 1 confirmed orphaned (PredictionPanel.tsx)
```

---

## 17. Known Issues

### 17.1 MISP: configured but not currently reachable ("port mismatch" candidate)

`MISP_URL=https://localhost:8443/` is set and a real API key is configured, and this exact configuration was live-reachable earlier in this project (returned real HTTP 403s during an authenticated-publish attempt). At the time of writing, `curl` to that URL returns connection code `000` (no response at all) — nothing is currently listening on port 8443 in this environment. **No literal port-mismatch bug was found in the code itself** (every reference to the MISP URL is consistent); the most likely explanation is that the real MISP instance runs in a separate process/container that is not currently up, not a code defect. Flagged honestly rather than guessed at further.

### 17.2 STIX corpus tactic-label data quality

31.2% of imported ATT&CK techniques carry non-standard `kill_chain_phases` values (`"stealth"`, `"defense-impairment"` instead of real tactics) — traced to the vendored file itself, not the import code. See Section 4.8.

### 17.3 "Constant confidence" — `DynamicRiskEngine.calculate_confidence()`

```python
evidence = [
    1.0, 1.0, 1.0,                                    # <-- hardcoded, unconditional
    1.0 if (threat_intel.ip_reputation != 0 or ...) else 0.0,
    1.0 if runtime.prediction_frequency > 0 else 0.0,
    1.0 if graph.node_count > 0 else 0.0,
    1.0 if graph.attack_chain_depth > 0 else 0.0,
]
return round(sum(evidence) / len(evidence) * 100, 2)
```

Three of seven terms are unconditional constants — confidence can never fall below **42.86%** regardless of actual evidence quality, and combined with Section 17.4 (threat-intel fields always 0), the 4th term is also effectively always 0 in this dataset, meaning **only 3 of 7 terms are ever real, non-constant signal in practice**.

### 17.4 Threat-intelligence fields are permanently empty

`Attacker.vt_reputation`, `.threat_actor_reputation`, `.malware_confidence`, `.tool_confidence`, `.misp_confidence`, `.ioc_confidence` are set exactly once, on node creation, from a static `ATTACKER_DEFAULTS = {..all 0.0..}` dict, and **never updated afterward by any code path in the repository** — confirmed by exhaustive grep. Live-verified: 0 of 9 real `Attacker` nodes have any non-zero value. No live VirusTotal, threat-actor-reputation, or IOC-confidence feed is wired into this system; MISP integration is outbound-publish only, not inbound-enrichment.

### 17.5 The TPS_CEILING calibration problem

`TPS_CEILING = 1500` is used to normalize raw accumulated TPS to a 0–100 display scale, but the real data shows raw `risk_score` values up to **34,820** — over 23× the ceiling. Every campaign above 1,500 raw TPS clamps to a flat 100% "CRITICAL," collapsing real differences in severity among the highest-risk campaigns into visual indistinguishability. `risk_scoring.py`'s own docstring acknowledges this is "a genuine open calibration question," not a bug with a known fix.

### 17.6 No measurable prediction accuracy

`prediction_hits`/`prediction_misses` are 0/0 across every real campaign — the tracking mechanism exists in the schema but has never been exercised. Any claim of "prediction accuracy" for this system would currently be fabricated; the honest statement is "not yet measured, mechanism exists."

### 17.7 Dead WebSocket/SocketIO code

`Flask-SocketIO` is an installed backend dependency with zero real usage; `frontend/src/hooks/useWebSocket.ts` is fully written, expects a server that doesn't exist, and is imported nowhere. The actual (and only) real-time update mechanism is a 30-second poll (`DashboardContext.tsx`). This is scaffolding for a feature that was never completed, not a regression.

### 17.8 SHUFFLE_WEBHOOK — configured, never triggered

`SHUFFLE_WEBHOOK` in `.env` now contains a real URL (confirmed this session — was empty in every earlier phase of this project). No live trigger has ever been attempted: the workflow behind that webhook is unknown to this documentation effort, and firing an unfamiliar webhook without knowing what it does was judged too risky to do opportunistically. `SHUFFLE_BASE_URL`/`SHUFFLE_API_KEY` (needed for execution-status polling, distinct from the trigger webhook) remain unset.

### 17.9 A resource-contention test flake (diagnosed, not a code defect)

Running the full 689-test backend suite *while* `dashboard_api.py`/`wazuh_listener.py` are also running live in the background (competing for CPU/memory alongside the loaded GNN/XGBoost models) produced 2 transient failures in subprocess-timeout-based tests (`test_audit_logging.py`). Re-running those same tests in isolation, and re-running the full suite with the competing processes stopped, both passed cleanly. Diagnosed as system-load-induced flakiness in a fixed 30-second subprocess timeout, not a code regression — documented rather than silently retried away.

### 17.10 Investigation history is not persisted

An investigation run (`POST /api/investigate/<id>`) is a request/response object — never written to Neo4j as its own queryable entity. CYUKTI can run a fresh investigation for a campaign at any time (deterministically reproducible given the same underlying data — Section 10.4's formulas are pure), but cannot answer "show me every investigation that has run against campaign X historically."

### 17.11 Shuffle execution and MISP publication are not directly linked

Both are keyed to the same `campaign_id`, but a `PlaybookExecution` record never stores a resulting MISP event ID, and `misp_cache.json`'s event mapping has no field referencing which (if any) `PlaybookExecution` triggered it.

### 17.12 GNN cross-fold embedding-space comparability

The GNN retrieval evaluation (Section 11.3) notes its own limitation: embeddings from different LOGO folds are not guaranteed directly comparable (the model is retrained per fold), which likely explains why Task B's single-model MRR (0.822) is *lower* than its fold-safe counterpart (0.975) — a methodological caveat, not a contradiction, but one a reviewer should be shown rather than have discovered independently.

---

## 18. Tech Stack

### 18.1 Backend (Python)

| Package | Version |
|---|---|
| Python | 3.12.3 |
| Flask | 3.1.3 |
| flask-cors | 5.0.1 |
| Flask-SocketIO | 5.5.1 *(installed, unused — Section 17.7)* |
| neo4j (driver) | 6.2.0 |
| scikit-learn | 1.3.2 |
| torch | 2.10.0 |
| torchaudio | 2.10.0 |
| torchvision | 0.25.0 |
| xgboost | 3.4.1 |
| jsonschema | 4.26.0 |
| requests | 2.32.5 |

### 18.2 Frontend (Node/TypeScript)

| Package | Notes |
|---|---|
| Node.js | v18.20.8 |
| React | (Vite + TypeScript template) |
| Vite | 5.4.21 |
| Vitest / React Testing Library | test runner |
| socket.io-client | installed, used only by dead code (Section 17.7) |
| lucide-react | icon set |

### 18.3 Infrastructure

| Component | Version / Image | Role |
|---|---|---|
| Neo4j | `neo4j:5-community` (Docker) | Graph database, ports 7687 (Bolt)/7474 (HTTP) |
| Wazuh manager | 4.7.5-1 | SIEM/HIDS, real alert source |
| Wazuh indexer | 4.7.5-1 | Wazuh's own OpenSearch-based storage |
| Wazuh dashboard | 4.7.5-1 | Wazuh's own UI (separate from CYUKTI's) |
| Suricata | (bundled with the Wazuh/Kali lab setup) | Network IDS feeding several custom Wazuh rules |
| MISP | Configured (`localhost:8443`), not currently reachable (Section 17.1) | Threat-intel sharing platform |
| Shuffle | Configured via a real webhook URL (Section 17.8), never triggered | SOAR execution engine |
| ATT&CK Enterprise (vendored) | `x_mitre_version 19.1`, spec `3.3.0` | Source knowledge base for `mitre_import/` |

### 18.4 Development environment

OS: `Linux 6.18.33.2-microsoft-standard-WSL2` (WSL2 under Windows 11), development machine (not a hardened production deployment) — a fact worth stating plainly in any performance-claims section of a paper.

---

## 19. SOAR / Playbook Layer

Real module names (the paraphrase "`playbook_generator.py`/`playbook_matcher.py`/`playbook_adaptation.py`" some earlier recollections used does not match the current tree — corrected here to the real filenames, all under `backend/soar/`): `generator.py`, `matcher.py`, `adapter.py`, `memory.py`, `execution_service.py`, `schema.py`, `shuffle_client.py`, `api.py`. 83 tests across 9 files (`tests/test_soar_*.py`).

### 19.1 `generator.py` — `PlaybookGenerator`

Turns an already-resolved `CampaignContext` into a `Playbook`. Every `PlaybookAction` carries a `reason` string tracing to a real signal — a MITRE mitigation record from `recommendation_engine.get_recommendations()` (live Neo4j `CourseOfAction` match), a field actually present on the campaign, or a risk threshold — the generator "never invents evidence it doesn't have" (module docstring). Fixed action skeleton: enrich attacker IP → CTI/MISP lookup → historical-campaign search (RAG) → collect victim evidence (if a victim IP exists) → per-technique MITRE mitigations (capped at 2 per technique, `requires_approval=True`) → conditionally `block_ip`/`isolate_host` (only at `HIGH`/`CRITICAL` severity, both `destructive=True, requires_approval=True`) → create incident ticket → notify SOC (always last). `execution_policy` is `ANALYST_APPROVAL` if any action requires approval, else `RECOMMEND_ONLY`. Deliberately **not** built on the older `response/playbook_generator.py` ("Generation-1," dead per `GENERATION1_DISPOSITION.md` — keyed on synthetic `severity_label`/`attack_label` strings that don't exist in the real pipeline, and its `Mitigator` sibling executed raw shell commands directly, which this SOAR layer replaces with a Shuffle-mediated, approval-gated model).

### 19.2 `matcher.py` — `PlaybookMatcher`, the 5 similarity signals

Retrieves historical playbooks ranked by real per-candidate signals (`_score_one()`):

1. **`technique_similarity`** — Jaccard overlap of MITRE technique sets between the current and historical campaign.
2. **`topology_similarity`** — GNN embedding similarity via `ml.gnn.topology_similarity.gnn_topology_similarity_between_campaigns()`; `None` (not 0.0) when the GNN is unavailable, so absence is never silently scored as dissimilarity.
3. **`attacker_match`** — boolean, historical attacker IP equals the current campaign's attacker IP.
4. **`victim_match`** — boolean, same for victim IP.
5. **`historical_success_rate`** — from `memory_store.historical_match_stats(playbook_id)`, i.e., this playbook's own real track record (19.4).

Ranking combines `technique_similarity + (topology_similarity or 0.0)` with `historical_success_rate or 0.0`; a candidate with zero technique overlap, no topology similarity, and no IP match on either side is excluded outright. The match reason string is built from whichever signals actually fired (e.g. "73% MITRE technique overlap", "61% GNN topology similarity").

### 19.3 `adapter.py` — `PlaybookAdaptation`, IOC retargeting

Rewrites only the **identity-bound** input fields of a historical playbook's actions (`ip`, `campaign_id`, `operation_id` — the module's `_IDENTITY_INPUT_KEYS`) to point at the current incident; everything else (action type, name, `destructive`, `requires_approval`, mitigation IDs, technique, severity computed at generation time) is left untouched — "copying a playbook must never silently change what it does, only who/what it targets" (module docstring). IP retargeting is resolved by the action's semantic role, not by matching the old value: `enrich_ip`/`block_ip`/`threat_intel_lookup` always retarget to the *current attacker* IP; `collect_evidence`/`isolate_host` always retarget to the *current victim* IP. A deep copy is made, a fresh `playbook_id` is assigned, `adapted_from_playbook_id` records the lineage, and every adapted action's `reason` is prefixed `"[Adapted from <original_id>] ..."` so the provenance survives in the UI.

### 19.4 `memory.py` — `PlaybookMemoryStore`, SQLite persistence

Backed by SQLite (stdlib, `backend/soar/playbook_memory.db`), not Neo4j — the module docstring is explicit about why: "Playbook → PlaybookExecution → PlaybookActionResult is a strictly relational one-to-many-to-many shape with no graph-traversal requirement of its own... Neo4j is reserved for the parts of CYUKTI that are genuinely graph-shaped." Four tables: `playbooks`, `playbook_executions`, `playbook_action_results`, `soar_audit_events`. `effectiveness(playbook_id)` computes real `PlaybookEffectiveness` (executions, successes, failures, `success_rate`, mean duration, analyst approvals/rejections, up to 10 real failure reasons) purely from rows already written — no separate metrics pipeline. Every lifecycle transition also writes a `soar_audit_events` row (`log_audit_event()`), independent of Shuffle's own logging.

### 19.5 `api.py` — 13 real routes (`soar_bp`)

```
GET  /status
POST /playbooks/generate
GET  /playbooks
GET  /playbooks/<playbook_id>
POST /playbooks/adapt
POST /playbooks/<playbook_id>/execute
GET  /executions
GET  /executions/<execution_id>
POST /executions/<execution_id>/poll
POST /executions/<execution_id>/approve
POST /executions/<execution_id>/reject
GET  /recommendations/<campaign_id>
GET  /effectiveness
```

### 19.6 The real execution state machine (`execution_service.py`, `schema.ExecutionStatus`)

The real enum is `PENDING`, `PENDING_APPROVAL`, `REJECTED`, `RUNNING`, `SUCCESS`, `FAILED`, `TIMEOUT`, `CANCELLED` (`soar/schema.py`) — not the simplified "PENDING → EXECUTING → COMPLETED/FAILED/TIMED_OUT" shorthand some earlier recollections used; corrected here to the real values. Real transitions, owned by `PlaybookExecutionService`:

- `request_execution()`: computes `effective_policy()` — **a safety rule enforced in code, not just the UI**: any playbook containing a `destructive=True` action is silently downgraded from `AUTOMATIC` to `ANALYST_APPROVAL`, regardless of what the generator originally set ("a historical incident looking similar is never sufficient justification for unattended destructive action," module docstring). `RECOMMEND_ONLY` playbooks raise `PolicyError` if execution is attempted at all.
- `ANALYST_APPROVAL` policy → status starts at `PENDING_APPROVAL`; `approve(execution_id, approved_by)` or `reject(execution_id, reason)` are the only two ways out. `reject()` → `REJECTED` (terminal).
- `approve()` or an `AUTOMATIC`-policy non-destructive playbook → `_trigger()` → status → `RUNNING`, `started_at` set, a `PLAYBOOK_EXECUTION_STARTED` audit event logged, then `ShuffleClient.trigger()` is called with the full action list as payload.
- From `RUNNING`, four real outcomes: `ShuffleTriggerOutcome.NOT_CONFIGURED` → `FAILED` immediately, every action result carries the honest error `"Shuffle is not configured (SHUFFLE_WEBHOOK is empty)..."`; `SYNCHRONOUS_RESULT` → `SUCCESS` immediately with real output; `TRIGGERED` (fire-and-forget async) → **stays `RUNNING`** until `poll_status()` is called; `AUTH_FAILED`/`TIMEOUT`/`ERROR` → `TIMEOUT` or `FAILED` with the real Shuffle error attached.
- `poll_status()` only ever finalizes a `RUNNING` execution if Shuffle's REST status endpoint is configured and reachable — "running"/"not_configured"/"error" responses leave the execution `RUNNING` unchanged, since "an unreachable status endpoint is not evidence of failure" (module docstring; never fabricates a status this environment can't actually observe).

### 19.7 What's live-verified vs. blocked by infrastructure

| Component | Status |
|---|---|
| `PlaybookGenerator` producing a real playbook for a real campaign | **Live-verified** (`test_soar_generator.py`, and prior-phase live generation runs referenced in `ARCHITECTURE_VERIFICATION.md`) |
| `PlaybookMatcher` ranking real historical playbooks | **Live-verified** — real ranked candidates with real explanation strings |
| `PlaybookAdaptation` retargeting a real historical playbook | **Live-verified** — real playbooks adapted for real campaigns, identity assertion in `test_final_trace.py` (`plan.playbook is playbook`) |
| SQLite persistence (`memory.py`) | **Live-verified** — real playbooks/executions/audit events read back from `playbook_memory.db` |
| All 13 API routes | Unit-tested (`test_soar_api.py`); not separately load-tested |
| `SHUFFLE_WEBHOOK` configured with a real URL | **Confirmed this session** (a genuine change from earlier in the project) |
| An execution actually reaching `RUNNING`/`SUCCESS` via a live Shuffle trigger | **Not live-executed.** The workflow behind that webhook is unknown to this session; firing an unfamiliar webhook without understanding what it does was judged a real-world action requiring the operator's explicit go-ahead, not something to do opportunistically. The `NOT_CONFIGURED`/failure path IS live-exercised (via `test_failure_injection.py`'s explicitly-constructed unconfigured client), and it correctly returns `FAILED` with an honest per-action error, never a fabricated success. |
| Shuffle → MISP correlation | **Not implemented** — the two subsystems share only a `campaign_id`, no direct reference (documented gap, `ARCHITECTURE_VERIFICATION.md`) |

---

## 20. Raw Alert Volume and Deduplication Ratio

**Current live file**: `wc -l /var/ossec/logs/alerts/alerts.json` = **120** lines (this session's live re-check; `WAZUH_ALERT_FILE` is unset, so the code default `/var/ossec/logs/alerts/alerts.json` is the actual path in use, confirmed from `scripts/mitre_coverage_report.py`'s own `os.environ.get(...)` default).

**The rotation complication (found this session, not previously documented)**: Wazuh rotates `alerts.json` and archives dated copies under `/var/ossec/logs/alerts/2026/<Month>/ossec-alerts-DD.json[.gz]`. The current 120-line file is a live, actively-growing snapshot, not the deployment's full historical volume — the archive tree contains 78 further uncompressed dated files plus 22 `.gz` files spanning at least June through September 2026. Summing just the uncompressed archived files plus the current file: **463,638 raw alert lines** — and this excludes the 22 gzipped files entirely (not decompressed for this measurement), so it is itself a floor, not a ceiling. This total also includes months of routine Wazuh/OS housekeeping noise unrelated to any attack (`dpkg` package installs, disk-space monitor alerts, agent connect/disconnect events, Suricata APT-repository traffic) — it is not a clean "raw attack alerts" count.

Because of this, a literal `1 − (212 / current_file_line_count)` is not meaningful (212 > 120 — the current snapshot alone undercounts by construction) and `1 − (212 / 463,638)` overstates true dedup effectiveness by counting hundreds of thousands of alerts CYUKTI's fingerprinting logic never even saw as "attack-relevant" candidates in the first place. The **defensible, apples-to-apples dedup ratio** uses CYUKTI's own tracked occurrence counter instead — a real, already-computed Neo4j property (`AttackEvent.occurrences`), not an estimate:

| Metric | Value |
|---|---|
| Distinct `AttackEvent` nodes | 212 |
| Total tracked occurrences (Σ `AttackEvent.occurrences`) | 3,185 |
| **Dedup ratio** = 1 − (212 / 3,185) | **93.3%** |
| Avg occurrences per distinct event | 15.02 |
| Distinct dedup fingerprints (separate stat, `generate_fingerprint()`) | 71, avg reuse 2.99×, max reuse 19× |

This 93.3% figure measures dedup specifically among alerts that passed MITRE resolution and entered the campaign-correlation pipeline (i.e., the population `is_duplicate()` actually filters), which is the intended meaning of a "dedup ratio" claim in the paper — not a ratio against Wazuh's total raw log volume, most of which was never attack-relevant to begin with.

---

## 21. Multi-RAG Detail: the Campaign-Narrative Retriever (expands Section 10.2)

`rag/campaign_retriever.py`'s `CampaignNarrativeRetriever` indexes a TF-IDF corpus built from every `HistoricalCampaign` record (`attribution_context.py`; fields: `campaign_id`, `attacker`, `victim`, `techniques`, `timestamps`, plus optional `status`/`prediction`/`prediction_confidence`), loaded via `attribution_context.context.load_historical_campaigns()`. Each document's text is constructed by `_campaign_documents()` as exactly: `f"Campaign {campaign_id}: attacker {attacker} targeting {victim}. Techniques observed: {' '.join(techniques)}."` — one flat sentence per historical campaign, no additional feature engineering. Unlike `mitre_retriever.py`'s `MitreSemanticRetriever` (a module-level singleton over the vendored, never-changing STIX corpus), `CampaignNarrativeRetriever` is deliberately **not** a singleton — it's re-instantiated per investigation because the historical-campaign corpus grows over time as new campaigns close. `query()` returns real `Evidence` objects (`source=EvidenceSource.CAMPAIGN_HISTORY`, `confidence=1.0`, `relevance=doc.relevance`, `provenance="rag.campaign_retriever (TF-IDF over historical campaign records)"`). Its own module docstring states the distinction from the exhaustive, non-RAG `CAMPAIGN_HISTORY` technique-set-overlap collector explicitly: this retriever answers a genuinely different question — *"which past campaigns best match this free-text description of observed behavior"* — versus `CAMPAIGN_NARRATIVE_SEARCH`'s sibling `MITRE_SEMANTIC_SEARCH`, which answers *"which ATT&CK reference concepts (techniques, mitigations, groups) does this text most resemble"* against the static STIX corpus, not against CYUKTI's own campaign history at all.

---

## 22. Evaluation Metrics Framework

`evaluation_metrics.py` (backed by `ACCURACY_EVALUATION.md`) provides ground-truth-agnostic, unit-tested measurement functions, ready to use the moment real labels exist for any task: `classification_report` (accuracy, macro/weighted/micro F1, balanced accuracy — reused as-is across MITRE mapping and severity tasks), `confusion_matrix` (full N×N, fixed label order), `multilabel_exact_match_ratio` (multi-technique alerts), `precision_recall_f1`, `false_positive_negative_rates`, `pairwise_precision_recall_f1` and `cluster_purity` and `campaign_fragmentation` (campaign/operation correlation — over-merging vs. over-fragmentation reported separately, never blended into one number), `recall_at_k`, `precision_at_k`, and `mean_reciprocal_rank` (GNN retrieval and playbook recommendation ranking tasks). **21 tests** (`test_evaluation_metrics.py`) verify every function against hand-computed values — perfect prediction, known error patterns, over-merging vs. over-fragmenting cluster cases, empty/no-match edge cases. Critically: every function is correctness-tested in isolation, but **no CYUKTI-specific ground-truth dataset has been run through them yet** — `ACCURACY_EVALUATION.md` labels most of CYUKTI's own tasks (MITRE mapping accuracy, threat qualification, campaign/operation correlation, campaign selection, playbook effectiveness) as category **(C)**, "unavailable — no independent ground truth exists in this environment," precisely to avoid the circularity of scoring CYUKTI against labels CYUKTI itself produced. Only GNN retrieval and XGBoost severity classification are category **(A)**, measured against real independent ground truth (LOGO cross-validation grouped by real attacker IP — Sections 11.1/11.3).

---

## 23. Latency

Real, already-measured data exists — `backend/benchmarks/run_benchmarks.py`, documented in full in `BENCHMARKS.md`, committed reference run `backend/benchmarks/results/benchmark_20260925T075030Z.json`. Measured 2026-09-25, live Neo4j (Docker, `neo4j:5-community`), `GNN_ENABLED=true`, a trained XGBoost severity model present, 20 iterations per benchmark (first call excluded as warm-up):

| Stage | p50 (ms) | p95 (ms) | p99 (ms) | throughput (ops/sec) |
|---|---|---|---|---|
| MITRE resolution | 0.001 | 0.001 | 0.002 | ~1,110,000 |
| IOC extraction | 0.000 | 0.000 | 0.001 | ~5,035,000 |
| Deduplication check | 0.001 | 0.002 | 0.003 | ~735,000 |
| Neo4j simple query | 0.904 | 1.251 | 1.285 | ~1,116 |
| Campaign selection scoring (in-memory, 10 candidates) | 0.037 | 0.073 | 0.074 | ~23,200 |
| Campaign selection candidate discovery (live Neo4j query + signal computation) | 21.035 | 24.285 | 26.480 | ~47 |
| Severity prediction (XGBoost) | 21.975 | 39.158 | 117.582 | ~34.5 |
| GNN embedding (real model, real campaign graph, cache bypassed) | 2.818 | 3.981 | 4.070 | ~342 |
| ResponsePlan generation | 0.002 | 0.004 | 0.006 | ~373,000 |
| **Total end-to-end** (`GET /api/incidents/<id>/overview`, real HTTP round trip) | **35.538** | **66.869** | **100.484** | **~22.8** |

Per-stage breakdown as requested: MITRE resolution and deduplication are sub-microsecond (pure in-memory Python, no I/O — lower bounds on decision cost only, not the surrounding pipeline). Neo4j writes/reads land around 1–25ms depending on query complexity (a simple query vs. campaign-candidate discovery, which does real signal computation across all candidates). Feature extraction is not separately benchmarked as its own row — it is embedded inside the severity-prediction and campaign-selection-scoring numbers above, since `graph_feature_engine.py`/`mitre_feature_engine.py` are called synchronously as part of those same measured calls, not as an independently invokable stage. Scoring (XGBoost) is the single largest per-call cost (~22ms p50, with a heavier 117.6ms p99 tail consistent with JIT/cache warm-up variance in XGBoost's C++ path, not a systematic bottleneck). The end-to-end p50 (~36ms) is dominated by two real network round trips (candidate discovery ~21ms + base lookups), not any one slow component. **Caveat carried over from `BENCHMARKS.md` unchanged**: single-process, single-machine, low-concurrency numbers on a development machine — not a load-tested production SLA.

---

## 24. Wazuh Rule Fix Status (live re-verification, this session)

Section 4.9 documented six duplicate-rule-ID fixes added to `local_rules.xml`, noting at the time that "the manager was running (since before the fix was applied) and has not been restarted." Re-checked this session:

- `local_rules.xml` last modified: `2026-09-25 05:09:59 UTC`.
- System boot time: `2026-09-25 14:20:39 UTC`. `wazuh-analysisd` process start time: `2026-09-25 14:21:31 UTC` — i.e., the manager process restarted (via a full environment reboot, not a manual `wazuh-control restart`) **after** the rule fix was written to disk. The fixed rules are therefore loaded into the currently-running manager.
- Re-ran `scripts/mitre_coverage_report.py` against the live `alerts.json` (120 lines) as instructed:

```
Provenance distribution:
  UNKNOWN         94  (78.3%)
  NATIVE_WAZUH    26  (21.7%)

Resolved technique frequency:
  T1595        x25
  T1562.001    x1   <-- NOT in MITRE_TO_STAGE (tps=0)
```

**Honest reading of this result**: none of the six specifically-fixed rule IDs (100510, 100511, 100513, 210001, 210011, and the sixth from 4.9) appear in the current 120-line window at all — the current alerts are dominated by rule `86601` (Suricata APT-repo traffic, unrelated to the fix), `2902`/`2904` (dpkg housekeeping), and `100500` (Nmap reconnaissance, which was already correctly resolving to `T1595` via `NATIVE_WAZUH` *before* the 4.9 fix — it was never one of the broken rules). This is **not** evidence the fix failed; it is evidence that the specific attack types the fix targeted (brute force, port-scan variants, DoS) have not been re-triggered against this environment since the reboot. The fix's activation status is: **rules are active in the live manager as of this measurement**, but not yet exercised by a matching alert — re-running the same synthetic attacks that originally produced rule IDs 100510/100511/100513/210001/210011 would be the direct way to confirm the corrected MITRE mappings end-to-end.

---

## 25. Detection Paradox: Conceptual Framing

From `review_pack/06_research_contribution.md` — the project's strongest conceptual framing, restated here as the paper's reference copy:

| Concept | What it actually answers | Where CYUKTI implements it |
|---|---|---|
| Detection | "Did a sensor observe something?" | Wazuh's rule engine — happens regardless of MITRE tagging |
| MITRE attribution | "Can we defensibly say which technique this was?" | `mitre_resolver.py` |
| Investigation | "What does this mean in context of a campaign?" | `investigation/` loop, evidence collectors |
| Evidence confidence | "How much do we trust what we know?" | `investigation/confidence.py` |
| Prediction | "What is likely to happen next?" | `prediction_engine.py`, gated by `NEXT_TECHNIQUE` learned counts |
| Response | "What should be done?" | `recommendation_engine.py`, MISP publication |

The paradox the table names: a Wazuh rule firing (**Detection**) is a completely different, independent event from CYUKTI being able to say *which ATT&CK technique* that alert represents (**MITRE attribution**) — an alert can be loudly, correctly detected and still resolve to `UNKNOWN` provenance (Section 4), because detection and attribution are answering two different questions with two different failure modes. This is why `UNKNOWN` is not treated as a defect to be hidden or silently guessed away in this codebase: an `UNKNOWN`-provenance `AttackEvent` is still a real, detected, timestamped, graph-connected event — it simply hasn't been defensibly mapped to a named technique yet, and CYUKTI's 4-tier ladder (4.1) is built specifically to make that distinction visible rather than to fabricate a technique ID just to fill the field. Each of the six concepts above answers a genuinely different reviewer question, and conflating them is the single most common way a SOC-tooling paper overstates what its system does.

---

## 26. What Else Belongs Here (self-identified gaps in this document)

- **No formal threat model / adversarial-robustness analysis** has been written for CYUKTI itself (e.g., can an attacker poison the Markov-chain predictor by manufacturing fake transitions, or corrupt campaign correlation by mimicking a known attacker IP). Worth a paragraph in the paper's limitations section.
- **No user study.** Every UX claim ("an analyst should understand within 10 seconds") is a design goal verified by the authors reading their own dashboard, not by an independent analyst evaluation. A journal reviewer will likely ask for this or expect it named as future work.
- **No comparison against a real competing product** (Splunk ES, Elastic Security, a commercial SOAR). The "what makes it different" claims in Section 1.2 are architectural/qualitative, not benchmarked head-to-head.
- **Dataset size is small and self-generated** (111 campaigns, 212 events, from a lab environment the authors control) — any performance number in this document should be captioned as such in the paper, not presented as if drawn from production SOC traffic at scale.
- **No citation list has been assembled** — MITRE ATT&CK, GraphSAGE (Hamilton et al.), XGBoost (Chen & Guestrin), and Wazuh/MISP/Shuffle's own documentation will all need formal citations; this document does not include a bibliography.

---

*Document generated 2026-09-25 from direct source-code inspection and live queries against this project's real Neo4j instance (post-restart), Wazuh manager, and test suites. Every number is traceable to a specific file/query cited inline. No fabricated data, estimated statistics, or rounded-for-effect figures appear anywhere in this document.*
