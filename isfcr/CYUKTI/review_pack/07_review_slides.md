# CYUKTI — Suggested Review Slide Deck (10-12 slides)

> Last verified: 2026-09-25. Counts reflect live Neo4j query and pytest run from this date.

## Slide 1 — Problem
- Real Wazuh deployments generate mostly-untagged telemetry; most rules have no native ATT&CK mapping.
- Naive SOC pipelines require MITRE attribution before ingestion — discarding most real alerts.
- A security research pipeline that fabricates attribution to "solve" this corrupts its own ground truth.
- CYUKTI's goal: ingest everything, attribute only what's defensible, never fabricate.
- Diagram: none needed.
- **Say**: "Most security telemetry doesn't come pre-labeled with an attack technique — the question is what you do about the rest."

## Slide 2 — The Detection Paradox
- Detection ≠ Attribution ≠ Investigation ≠ Prediction ≠ Response — six distinct questions, often conflated.
- "No MITRE mapping" does not mean "irrelevant" — it means "not yet defensibly attributable."
- Fabricating a technique contaminates every downstream consumer (chain learning, risk scoring, ML ground truth).
- Show: the six-row concept table from `06_research_contribution.md`.
- **Say**: "The paradox is that requiring attribution before ingestion destroys the evidence you'd need to ever attribute it later."

## Slide 3 — CYUKTI Architecture
- Wazuh → Listener → MITRE Resolver → {Resolved, Unknown} → Neo4j → downstream modules.
- Neo4j as the central knowledge graph: Campaign, AttackEvent, Technique, Attacker, Host, Operation.
- Real ATT&CK STIX corpus vendored locally (v19.1, 858 techniques) — no runtime internet dependency.
- Diagram: `cyukti_architecture.mmd` (high-level).
- **Say**: "Everything downstream — prediction, attribution, RAG, dashboard — reads from one graph that both resolved and unresolved evidence feed into."

## Slide 4 — Data Flow
- Alert → dedup → MITRE resolution → {create_attack_event, create_unattributed_attack_event} → campaign attachment → chain/prediction/correlation (resolved only).
- Live-verified (2026-09-25): 212 AttackEvents, 111 Campaigns (was 124/65 on 2026-09-12; orphan count from 2026-09-12, 0/124, not re-run at the new scale).
- Diagram: `cyukti_detailed_architecture.mmd`.
- **Say**: "Every alert reaches the graph — the fork is only in what it's allowed to do once it's there."

## Slide 5 — Knowledge Graph / Campaign Reconstruction
- 858 real Technique nodes, imported from vendored STIX (not invented).
- Campaign reconstruction: repaired 44 real orphaned events into 27 campaigns, idempotent, evidence-only (never guesses).
- Metric to show: `check_integrity()` — 0 orphans currently.
- **Say**: "We found a real data-integrity defect in production — 44 events with no parent campaign — and fixed it without guessing or fabricating provenance."

## Slide 6 — Prediction / ML
- XGBoost severity classifier: real trained artifact, in-sample accuracy 0.917 (n=60, **not held-out** — say this explicitly).
- SSL autoencoder trained on real CICIDS2017 windows; hand-built GNN (no torch_geometric).
- NEXT_TECHNIQUE learning: only 3 real edges — own audit concluded `INSUFFICIENT_FOR_SUPERVISED_ML`.
- Metric to show: confusion matrix; the `INSUFFICIENT_FOR_SUPERVISED_ML` verdict.
- **Say**: "We built the full ML pipeline correctly, then used our own validity gate to conclude the current dataset can't yet support a real predictive claim — that's a finding, not a failure."

## Slide 7 — Threat Attribution
- `threat_attribution_engine.py` implemented, evidence-collector-tested.
- No ground-truth attribution dataset exists — accuracy is `NOT MEASURED`.
- **Say**: "Attribution is implemented and exercised in the evidence pipeline, but we don't have a labeled dataset to report an accuracy number against, and we're not going to invent one."

## Slide 8 — Evidence / RAG / MISP
- 6 evidence-source collectors (MITRE, graph, detection, CTI, campaign history, attribution).
- RAG retriever indexes 500+ real ATT&CK techniques via TF-IDF over the vendored STIX corpus.
- MISP client initializes against the local instance; live publish success not confirmed this session.
- **Say**: "Evidence is multi-source and grounded in a real ATT&CK corpus, not a toy dataset."

## Slide 9 — Phase 20: MITRE Resolution
- 4-tier precedence: NATIVE_WAZUH > REVIEWED_RULE_MAPPING > DETERMINISTIC_INFERENCE > UNKNOWN.
- Live coverage (2026-09-25, post rule-fix + reboot, 120 alerts): 21.7% NATIVE_WAZUH, 78.3% UNKNOWN, 0% reviewed/inferred/ambiguous. (Earlier snapshot, 2026-08-31, 391 alerts: 10.0%/90.0%.)
- Live-verified: 0 fabricated attack_id, 0 Technique contamination, NEXT_TECHNIQUE unchanged.
- **Say**: "The large majority unresolved is expected and correct — it's Wazuh's own out-of-the-box coverage gap, and we chose to preserve that evidence rather than fake an answer."

## Slide 10 — Results & Metrics (consolidated dashboard)

| Component | Implementation Status | Dataset/Test Size | Primary Metric | Result | Evidence |
|---|---|---|---|---|---|
| Wazuh ingestion | Complete, live | 120 live alerts (current snapshot) | Ingestion success (post-Phase 20) | 100% ingested (resolved or preserved-unknown) | Live listener log |
| Neo4j graph | Complete, live | 111 Campaigns / 212 AttackEvents (2,539 total nodes, 20,804 total relationships) | Orphan rate | 0/124 (2026-09-12 check; not re-run at current scale) | Live Cypher query, 2026-09-25 |
| Campaign reconstruction | Complete, validated | 44 real orphaned events repaired | Orphans after repair | 0 (as of 2026-09-12) | `campaign_reconstruction.py`, 10 tests |
| Attack chain (NEXT_TECHNIQUE) | Implemented, data-limited | 3 learned edges | Evaluable prediction accuracy | 4/12 correct (33%) | Phase 18 rebuild |
| MITRE mapping (stage) | Complete, 1 known gap | 13 technique→stage entries | T1548.003 coverage | Missing | `mitre_mapper.py`, live check |
| MITRE resolver (Phase 20) | Complete, live-validated | 120 live alerts (post rule-fix/reboot) | Resolution coverage | 21.7% native / 78.3% unknown | `mitre_coverage_report.py`, 2026-09-25 |
| UNKNOWN handling | Complete, live-validated | 5 real UNKNOWN events | Fabricated attack_id | 0 | Direct Neo4j inspection |
| Prediction engine | Implemented, not viable yet | 12 evaluable transitions | Own validity verdict | `INSUFFICIENT_FOR_SUPERVISED_ML` | Phase 18 audit |
| Attribution | Implemented, unvalidated | — | Accuracy | NOT MEASURED | No ground-truth dataset |
| RAG | Complete, tested | 500+ real ATT&CK docs | Index size | >500 documents | `test_rag.py` |
| MISP | Implemented, unconfirmed live | — | Publish success | NOT MEASURED | Client initializes; not confirmed |
| SOAR/Playbook layer | Complete, unit-tested | 83 tests, 9 files | Live Shuffle trigger | Not live-executed (webhook configured, unfired) | `tests/test_soar_*.py` |
| Dashboard/API | Complete | 46 routes (33 core + 13 SOAR) | — | Passing | `test_dashboard_api.py`, `test_soar_api.py` |
| Dataset generation | Complete, frozen | 60 campaigns | Validity gate | `NOT_READY_FOR_CALIBRATION` | Phase 17 report |
| ML (XGBoost) | Complete, in-sample only | 60 rows, no held-out split | Accuracy (in-sample) | 0.917 | `evaluate_model.py`, 2026-08-31 |
| ML (SSL autoencoder) | Complete, trained | Real CICIDS2017 windows | Convergence | Confirmed by test | `test_ssl_pipeline.py` |
| ML (GNN) | Implemented, synthetic-tested only | Synthetic graphs | Real-data benchmark | NOT MEASURED | `test_gnn.py` |
| Latency (end-to-end) | Measured | Real HTTP round trip | p50 / p95 | 35.5ms / 66.9ms | `run_benchmarks.py`, 2026-09-25 |
| Test suite | Complete | 798 tests (689 backend, 65 files + 109 frontend, 17 files) | Pass rate | 798/798 | Fresh run, 2026-09-25 |

- **Say**: "This is the full component-by-component status — note that we distinguish complete-and-tested from complete-but-unbenchmarked throughout, deliberately."

## Slide 11 — Limitations
- Dataset frozen at 60 campaigns, 3 attackers, 2 victims — Phase 17's own verdict: `NOT_READY_FOR_CALIBRATION`.
- NEXT_TECHNIQUE learning: only 3 edges — `INSUFFICIENT_FOR_SUPERVISED_ML`.
- No held-out ML evaluation split exists anywhere.
- A recurring, non-blocking exception in the maintenance thread (`campaign_manager.expire_active_campaigns()`, `context.last_seen=None`) — known, not fixed, does not affect ingestion correctness.
- `100500`/`100501` duplicate local rule IDs — formally unresolved (needs root `wazuh-analysisd -t`); new live evidence suggests the Nmap-detection definition is active, not confirmed.
- **Say**: "We're disclosing every open issue we know about, including one we found live yesterday and deliberately have not fixed yet."

## Slide 12 — Future Work
- Deliberate dataset expansion (Phase 19 spec): 6-8 attackers, 5+ victims, ≥1 real High-severity example, ≥20 technique-set compositions.
- Add reviewed/inferred MITRE mappings only where independently justifiable — not to raise coverage numbers.
- Fix the maintenance-worker `last_seen` bug and the `100500`/`100501` duplicate-rule ambiguity.
- Held-out evaluation split once the dataset expansion makes one meaningful.
- Real-campaign GNN benchmark.
- **Say**: "The roadmap is expand-then-recalibrate, not fabricate-to-look-finished."
