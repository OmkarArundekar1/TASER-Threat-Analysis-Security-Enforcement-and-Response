# CYUKTI — Full System Integration Audit

Companion to `ARCHITECTURE_AUDIT.md` (which this supersedes for
component-integration purposes — that document's GNN research history
remains the authoritative record of the research phases; this document
is the current, live inventory of what actually executes and what the
dashboard actually shows). Built by direct inspection of this
repository's real files this phase, not from memory of prior reports —
every classification below traces to a specific import, route, or
mount point cited inline.

**Rule applied throughout**: a component is marked FULLY INTEGRATED
only if it participates in a real execution path reachable from either
the live alert pipeline (`realtime_socgraph.py` / `listener/wazuh_listener.py`)
or the dashboard (`dashboard_api.py` → a mounted React component). A
`.py` file existing, or a class being importable, is not integration.

---

## 1. Backend — core pipeline

| Component | Purpose | Current implementation | Integration | Status |
|---|---|---|---|---|
| `listener/wazuh_listener.py` | Tails Wazuh's alert JSON file, feeds `realtime_socgraph.process_alert` | Real, threaded queue worker + maintenance worker | **Correction (2026-09-24, live-verified):** a real Wazuh manager IS installed and running in this environment (`wazuh-apid.py` processes live since boot, `/var/ossec/logs/alerts/alerts.json` actively growing) and the listener process was observed running live, tailing it, creating real campaigns end-to-end. The "INFRASTRUCTURE BLOCKED" classification below was written from an earlier session's state and was wrong for this session — corrected here rather than left stale. | **FULLY INTEGRATED AND LIVE** — confirmed via a running listener process's own log: real alert ingestion, real campaign creation (e.g. `CAMP_E8E9F042`), real MISP publish attempts observed in flight. |
| `realtime_socgraph.py` | Per-alert orchestration: MITRE resolve → IOC → dedup → campaign → operation → Neo4j → severity → CTI → MISP sync | Real, calls every module below in sequence | Called by the listener above | FULLY INTEGRATED AND LIVE, confirmed running |
| `mitre_resolver.py`, `mitre_mapper.py`, `mitre_rule_registry.py`, `mitre_feature_engine.py` | MITRE ATT&CK technique resolution, stage mapping, Neo4j-backed feature lookup | Real, called from `realtime_socgraph.py` and `investigation/loop.py`'s `MITRE_KNOWLEDGE` action | Live pipeline + investigation | FULLY INTEGRATED |
| `dedup_engine.py` (+ `duplicate_buffer`) | Fingerprint-based enterprise deduplication | Real, background flush worker | Live pipeline | FULLY INTEGRATED |
| `campaign_manager.py`, `campaign_context.py`, `campaign_feature_engine.py`, `campaign_decision_engine.py` | Campaign lifecycle (create/continue/close), 7-feature continuation decision | Real | Live pipeline; `CampaignDecisionEngine`'s real (non-stub) `graph_similarity` weighted 1% (`config.CAMPAIGN_WEIGHTS`) | FULLY INTEGRATED |
| `campaign_correlation_engine.py`, `operation_manager.py`, `operation_context.py`, `operation_feature_engine.py`, `operation_decision_engine.py`, `operation_repository.py`, `operation_schema.py` | Operation-level correlation (7-dim weighted decision, 0.70 threshold) | Real; `gnn_topology_similarity` additive field (Section 5 below) | Live pipeline; dashboard shows correlation via `/api/correlation/campaigns/<id>` (a **separate**, standalone Cypher similarity query — traced live, confirmed distinct from this engine) | FULLY INTEGRATED (pipeline); operation-level detail (why campaigns were grouped) **NOT exposed to the dashboard** — no `/api/operations` route exists. *Planned:* add an `/api/operations/<id>` route surfacing the correlation breakdown once a dashboard consumer for it is scoped. |
| `neo4j_client.py` | All Cypher read/write | Real, single driver instance | Everywhere | FULLY INTEGRATED |
| `graph_feature_engine.py`, `graph_schema.py`, `runtime_graph_feature_engine.py` | Graph structural analytics (19 scalar features), runtime graph features | Real | XGBoost features, investigation `GRAPH_STRUCTURE` action, `/api/graph` | FULLY INTEGRATED |
| `dynamic_tps.py`, `dynamic_risk_engine.py`, `severity_engine.py`, `risk_scoring.py` | TPS/risk/severity scoring | Real | Live pipeline; risk/severity shown per-campaign in `/api/campaigns` | FULLY INTEGRATED |
| `prediction_engine.py`, `chain_updater.py`, `attack_chain_builder.py` | Next-technique prediction (Markov-chain style), attack chain persistence | Real | `/api/predictions`, `/api/predict`, `/api/attack-chain` | FULLY INTEGRATED (backend + API); frontend consumer is `PredictionPanel.tsx` — see Section 4, **currently unmounted** |
| `ml/train_xgboost.py`, `ml/runtime_predictor.py`, `ml/dataset_utils.py`, `ml/dataset_builder.py`, `ml/dataset_writer.py`, `ml/dataset_validator.py`, `ml/label_generator.py`, `ml/feature_schema.py`, `ml/feature_extractors.py` | Severity classifier, training pipeline | Real; `ml/models/xgb_severity.json` present | `/api/ml/predict/severity`, investigation `XGBOOST_PREDICTION` | FULLY INTEGRATED |
| `threat_attribution_engine.py`, `attribution_context.py`, `attribution_models.py`, `attribution_similarity.py`, `threat_actor_context.py` | Campaign-to-historical-campaign attribution (coverage/precision/chain), + additive `topology_similarity` | Real | Investigation `ATTRIBUTION_MATCH` action; **not** the same mechanism as `/api/attribution/actors/<id>` (that route matches against MITRE `ThreatActor` STIX nodes — a genuinely different, real, separate mechanism, both traced live) | FULLY INTEGRATED (both mechanisms, distinctly) |
| `detection_confidence_engine.py`, `threat_intelligence_engine.py`, `cti_confidence_engine.py` | Per-alert detection confidence, threat-intel confidence, blended CTI confidence | Real | Live pipeline; investigation `DETECTION_CHECK`/`CTI_LOOKUP` | FULLY INTEGRATED |
| `recommendation_engine.py` | MITRE mitigation lookup ("playbook") | Real, live Neo4j `CourseOfAction` query | `/api/recommendations`; frontend `RecommendationEngine.tsx` (mounted as "Playbook" tab, this session) | FULLY INTEGRATED |
| `datetime_utils.py`, `feature_orchestrator.py`, `config.py` | Shared utilities, config | Real | Everywhere | FULLY INTEGRATED |

## 2. Backend — CTI / MISP (real gap found this phase)

| Component | Purpose | Current implementation | Integration | Status |
|---|---|---|---|---|
| `misp_event_generator.py` | Builds a MISP event payload from an incident | Real | Called by `misp_sync.py` | FULLY INTEGRATED (generation) |
| `misp_sync.py` | Cache-aware create/update/search orchestration (`MISPSync.synchronize`/`.statistics()`) | Real | Instantiated in `realtime_socgraph.py` (verified: `from misp_sync import MISPSync`, `sync = MISPSync(publisher)`) | FULLY INTEGRATED (pipeline) |
| `misp_cache.py` | Persisted `campaign_id -> MISP event_id` map (`misp_cache.json`) | Real, thread-safe | Read/written by `misp_sync.py` | FULLY INTEGRATED (pipeline) |
| `cti_publisher.py` | Real HTTP client (`CTIPublisher`), retries, `.health_check()` (real `/servers/getVersion` probe, read-only) | Real | Instantiated in `realtime_socgraph.py` | FULLY INTEGRATED (pipeline) |
| `campaign_cti_builder.py` | Builds `IncidentContext`/CTI payload for a campaign | Real | Feeds `misp_sync` | FULLY INTEGRATED (pipeline) |
| **Authenticated MISP round-trip** | — | `CTIPublisher.health_check()` confirms it | **Update (2026-09-24):** a real MISP admin auth key was provided and configured in `backend/.env` this session. A live, running listener process was observed attempting real `POST /events/add` calls against it and getting **403** — the key it's using was loaded at process start, before a trailing-whitespace bug in the `.env` value was fixed, so that in-flight process is running on a corrupted key. Both `dashboard_api.py` and `wazuh_listener.py` need a restart to pick up the corrected key; not yet re-verified live as of this writing. *Planned:* restart both processes and re-verify the authenticated round trip in the next session. | **CREDENTIAL CONFIGURED, pending restart to verify** (was CREDENTIAL BLOCKED earlier this session) |
| **Dashboard visibility of any of the above** | — | — | **Zero** — grep of `dashboard_api.py` finds no `/api/misp/*` or `/api/cti/*` route at all | **ISOLATED (real gap) → addressed this phase, Section 6** |

## 3. Backend — evidence, investigation, RAG, GNN

| Component | Purpose | Current implementation | Integration | Status |
|---|---|---|---|---|
| `evidence/schema.py`, `evidence/store.py`, `evidence/orchestrator.py`, `evidence/collectors/*` | Evidence representation, per-source collectors | Real; `EvidenceSource.GNN_TOPOLOGY` added (production-integration phase) | `/api/investigate/<id>`'s `evidence` array | FULLY INTEGRATED |
| `investigation/actions.py`, `investigation/loop.py`, `investigation/confidence.py`, `investigation/next_best_evidence.py`, `investigation/stopping.py` | The evidence-aware investigation loop, 9-action menu, NBE selection, confidence/uncertainty, stopping criteria | Real | `/api/investigate/<id>`; frontend `EvidenceInvestigation.tsx` (mounted, "Investigate" tab) | FULLY INTEGRATED |
| `rag/mitre_retriever.py` | TF-IDF over the MITRE STIX corpus | Real | `/api/rag/mitre/search` (standalone-queryable) + investigation `MITRE_SEMANTIC_SEARCH` | FULLY INTEGRATED |
| `rag/campaign_retriever.py` | TF-IDF over historical campaign narratives | Real | Investigation `CAMPAIGN_NARRATIVE_SEARCH` **only** — no standalone API route | PARTIALLY INTEGRATED (investigation-only, not independently queryable). *Planned:* a standalone `/api/rag/campaign/search` route, mirroring the MITRE source's own route, once justified by a real consumer. |
| `rag/gnn_topology_retriever.py` | GNN embedding-similarity ranking over historical campaigns | Real | Investigation `GNN_TOPOLOGY_RETRIEVAL` + `/api/gnn/topology/<id>` (standalone) | FULLY INTEGRATED |
| `rag/retriever.py` | Generic TF-IDF engine both retrievers above build on | Real | Shared library code | FULLY INTEGRATED |
| **Unified Multi-RAG query surface** | Query all 3 sources at once, tagged by source | — | Does not exist as a single endpoint before this phase | **ISOLATED (real gap) → addressed this phase, Section 5** |
| `ml/gnn/inference.py`, `ml/gnn/topology_similarity.py` | Production GNN inference (frozen artifact, fail-safe) | Real | Operation correlation, attribution, retrieval, investigation, `/api/gnn/*`, dashboard `GNN Intelligence` page + `Topology` tab | FULLY INTEGRATED, **disabled by default** (`GNN_ENABLED`) |
| `ml/gnn/campaign_graphs.py`, `graph_encoder.py`, `layers.py`, `autoencoder_model.py`, `train_autoencoder.py` | Graph extraction, encoding, autoencoder architecture/training | Real | Produces the frozen artifact `inference.py` loads | FULLY INTEGRATED (offline/training path; not re-run at request time, by design — Section 3 of `GNN_PRODUCTION_INTEGRATION.md`) |
| `ml/gnn/temporal_graphs.py`, `evaluate_autoencoder.py`, `retrieval_evaluation.py`, `xgboost_ablation.py`, `train_gnn.py`, `synthetic_graphs.py`, `model.py` | Research-phase evaluation/ablation tooling, severity-classification GNN (unused, Option C) | Real, tested | **Deliberately not production-integrated** — `train_gnn.py`/`model.py`'s severity-classification path is Option C, rejected in favor of Option E (topology representation); the evaluation scripts are one-shot research tools, correctly not part of any live request path | OBSOLETE FOR PRODUCTION BY DESIGN, not dead code — preserved as the research record (`GNN_XGBOOST_ABLATION.md`, `GNN_RETRIEVAL_EVALUATION.md`) |
| `ml/ssl_pipeline.py`, `ml/ssft.py` | Self-supervised CICIDS2017 network-flow autoencoder + runtime transformer | Real, trained, tested in its own domain | Requires a live 312-dim network-flow capture pipeline that does not exist | **INFRASTRUCTURE BLOCKED** (unchanged from all prior phases; re-verified no new pipeline exists). *Planned:* revisit once a live network-flow/packet-capture pipeline producing a real per-campaign window exists — an infrastructure question, not an engineering one. |

## 4. Frontend — every component, mount status verified against the real `App.tsx`/`IntelligenceWorkspace.tsx` tree

| Component | Displays | Backend endpoint | Mounted? | Status |
|---|---|---|---|---|
| `TopNavBar.tsx` | Health, overview counts, view switcher | `/api/health`, `/api/overview` | Yes (`App.tsx`) | FULLY INTEGRATED |
| `SecurityOverview.tsx` | Stat cards | `/api/overview` | Yes | FULLY INTEGRATED |
| `LiveEventsFeed.tsx` | Alert stream | `/api/events`, `/api/events/<id>` | Yes | FULLY INTEGRATED |
| `AttackGraph.tsx` | Neo4j graph (Cytoscape) | `/api/graph`, `/api/graph/expand` | Yes | FULLY INTEGRATED |
| `MitreAttackChain.tsx` | Attack chain / learned transitions | `/api/attack-chain` | Yes | FULLY INTEGRATED |
| `CampaignIntelligence.tsx` | Campaign list + inline prediction/recs | `/api/campaigns`, `/api/predictions`, `/api/recommendations` | Yes | FULLY INTEGRATED |
| `QueryConsole.tsx` | Read-only Cypher console | `/api/query` | Yes | FULLY INTEGRATED |
| `AttackerIntelligence.tsx` | Attacker list | `/api/attackers` | Yes | FULLY INTEGRATED |
| `IntelligenceWorkspace.tsx` (tab container) | — | — | Yes | FULLY INTEGRATED |
| `ThreatCorrelation.tsx` | Standalone Cypher campaign similarity + `gnn_topology_similarity` | `/api/correlation/campaigns/<id>` | Yes (tab) | FULLY INTEGRATED |
| `AttackPathAnalytics.tsx` | Path analytics | `/api/analytics/paths` | Yes (tab) | FULLY INTEGRATED |
| `ThreatActorAttribution.tsx` | MITRE ThreatActor matches | `/api/attribution/actors/<id>` | Yes (tab) | FULLY INTEGRATED |
| `RiskPropagation.tsx` | Risk propagation | `/api/risk/propagation/<id>` | Yes (tab) | FULLY INTEGRATED |
| `TopologyIntelligence.tsx` | GNN topology (in-panel) | `/api/gnn/status`, `/api/gnn/topology/<id>` | Yes (tab) | FULLY INTEGRATED |
| `RecommendationEngine.tsx` | Playbook | `/api/recommendations` | Yes (tab, this session) | FULLY INTEGRATED |
| `EvidenceInvestigation.tsx` | Investigation + Multi-RAG source bar | `/api/investigate/<id>`, `/api/rag/mitre/search`, `/api/ml/predict/severity` | Yes (tab) | FULLY INTEGRATED |
| `GNNIntelligencePage.tsx` | Full-page GNN view | `/api/gnn/status`, `/api/gnn/topology/<id>` | Yes (top-level) | FULLY INTEGRATED |
| `PathExplorer.tsx` | Path exploration overlay | `/api/graph/paths` | Yes (conditional) | FULLY INTEGRATED |
| `PanelWrapper.tsx` | Shared chrome (collapse/fullscreen) | — | Yes (used by every panel) | UTILITY |
| **`PredictionPanel.tsx`** | Severity/next-technique prediction gauge | `/api/predictions` (data already fetched into context) | **No — grep of the entire `src/` tree confirms zero import sites outside its own file and test** | **ISOLATED (real gap) → addressed this phase, Section 7** |

## 5. Implemented this phase — Multi-RAG unified query

`POST /api/rag/search` (`backend/dashboard_api.py`): queries MITRE semantic search and campaign-narrative search (when `query` is given) and GNN topology retrieval (when `campaign_id` is given) **in one call**, each result tagged under its own `sources.<name>` key, never blended into a single ranking. Per-source failures are isolated (`{"error": ...}` for that source only, not a 500 for the whole request). 5 backend tests in `tests/test_dashboard_api_integration_routes.py`. Frontend: `api.ragSearch()` added to `services/api.ts` (not yet consumed by a dedicated UI this phase — the existing per-source surfaces, `/api/rag/mitre/search` via `EvidenceInvestigation.tsx` and `/api/gnn/topology/<id>` via `TopologyIntelligence.tsx`/`GNNIntelligencePage.tsx`, already give each source real visibility).

## 6. Implemented this phase — MISP/CTI status

`GET /api/misp/status` (`backend/dashboard_api.py`): reports credential-configured state, a real (read-only) `CTIPublisher.health_check()` probe, and the real `misp_cache.all()` campaign→event mapping — never fabricates a successful publish, never returns the API key (verified by an explicit test asserting the configured key string never appears in the response, including on the exception path). New top-level `ThreatIntelligencePage.tsx` renders it, reachable from `TopNavBar`'s view switcher ("Threat Intel"). Live-verified in this environment: `credential_configured: false` (no `MISP_API_KEY` set), matching the known state from `MISP_INTEGRATION.md`.

## 7. Implemented this phase — System Health, Audit Log, Prediction pages

- `GET /api/system/health` (`backend/dashboard_api.py`): five independent checks -- Neo4j connectivity, XGBoost artifact presence, GNN availability (respects `GNN_ENABLED`), MISP credential presence, and listener-log recency (`recently_active` if modified within 5 minutes) -- `status: degraded` only when Neo4j itself is down; every other subsystem reports its own state without affecting the others'. New top-level `SystemHealthPage.tsx`, auto-refreshes every 15s.
- `GET /api/audit/logs` (`backend/dashboard_api.py`): tails and parses the real `backend/logs/prerana_listener.log` file (`TIMESTAMP | LEVEL | message` format, confirmed against the actual live file, which had 1,580 real lines from today's session at the time of writing), paginated via `?limit=`, gracefully returns an empty result if the file doesn't exist rather than erroring. New top-level `AuditLogPage.tsx` with a manual refresh button.
- New top-level `PredictionIntelligencePage.tsx`: lists every real prediction currently held by `DashboardContext` (from the existing, already-working `/api/predictions` -- the gap was frontend-only: `PredictionPanel.tsx` had zero import sites anywhere in `src/`). This new page is a full-page, multi-prediction view; the original `PredictionPanel.tsx` component is left as-is (still unmounted, still has its own passing tests) rather than force-fit into a page it wasn't designed for.
- All four pages wired into `TopNavBar.tsx`'s view switcher (extended `TopLevelView` union: `dashboard | gnn | prediction | threat-intel | system | audit`) and `App.tsx`'s conditional render, following the exact pattern already established for the GNN page.
- Backend: 15 new tests (`tests/test_dashboard_api_integration_routes.py`), full suite re-run clean (457 passed / 26 skipped pre-existing-skip / 2 pre-existing failures unrelated to this phase -- both are live-Neo4j-dependent tests failing because no Neo4j instance is currently running in this environment, not a regression from this phase's changes; neither test touches any file this phase modified).
- Frontend: 10 new tests across the 4 new page components, full suite 92/92 passing (up from the 82 baseline), `npm run build` clean with no new TypeScript errors.

**Not done this phase** (explicitly, not fabricated as complete): the ~15-page dashboard reorganization described in the original request (Overview/Live Alerts/Campaign Explorer/Operation View/Attack Graph/Investigation Workspace/Evidence Explorer as *separate* dedicated pages, MITRE explorer page, Settings page), full security audit pass, and the comprehensive 30-point final report were not attempted this session given the scope -- the four real, previously-invisible gaps (MISP visibility, System Health, Audit Log, Prediction page) were prioritized and completed end-to-end (backend route → test → frontend page → test → build verification) rather than spreading effort thin across all ~40 sections superficially. *Planned:* pick up the remaining dashboard reorganization, security audit pass, and final report items in a subsequent phase scoped for that work specifically.

---

## Summary counts

| Classification | Count (approx., backend modules) |
|---|---|
| FULLY INTEGRATED (incl. Wazuh listener + live MISP publish attempts, corrected 2026-09-24) | ~47 |
| PARTIALLY INTEGRATED | 1 (`rag/campaign_retriever.py` — investigation-only, no standalone route; not changed this phase, since Multi-RAG's unified endpoint (Section 5) now covers this same retriever's output through a new path) |
| INFRASTRUCTURE BLOCKED | 1 (SSL/SSFT — Wazuh reclassified as live, see Section 1's correction) |
| CREDENTIAL CONFIGURED, pending restart to verify | 1 (MISP authenticated publish — real key set in `.env`, running processes need a restart to load it; was CREDENTIAL BLOCKED earlier this session) |
| OBSOLETE FOR PRODUCTION BY DESIGN | ~7 (GNN Option C / research-only tooling, Generation-1 per `GENERATION1_DISPOSITION.md` — unchanged, not touched this phase) |
| ISOLATED, now fixed this phase | 3 (`PredictionPanel.tsx`, MISP dashboard visibility, unified Multi-RAG endpoint) |

**Generation-1 code** (`agents/`, `soc_engine/explainer.py`, `classifiers/`, `response/`, `kill_chain/`, ~1,660 lines): unchanged from `GENERATION1_DISPOSITION.md`'s prior audit — proven dead, not deleted, not touched this phase (out of this phase's scope per the operating principle).
