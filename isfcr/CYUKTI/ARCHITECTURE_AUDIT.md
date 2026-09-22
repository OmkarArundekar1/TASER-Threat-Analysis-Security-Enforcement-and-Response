# CYUKTI — Architecture Gap Audit

Produced after six prior BUILD→INTEGRATE→STABILIZE sessions (ML/NBE,
audit logging, Campaign Correlation, Threat Attribution, MISP,
Dashboard API, frontend testing) brought the system to 397 passing
tests (336 backend + 61 frontend) with each named subsystem
behaviorally and, where infrastructure allowed, live-verified. This
document answers: **given what CYUKTI actually implements today, what
is genuinely still missing, and what should be built next** — not
another test-count exercise, not a UI redesign, not a research
validation experiment.

**Phase 21/22 findings remain frozen and untouched by this audit or
its implementation work** — see the explicit boundary noted under
Multi-RAG below.

---

## A. Current architecture (as actually traced, not as documented)

Tracing real imports and real call sites — not assuming
`review_pack/`'s diagrams are current — surfaced a fact not previously
recorded in this project's own documentation: **there are two
architectural generations in this repository**, and only one of them
is live.

### Generation 2 (current, live, production path)

```
Wazuh alerts.json
   -> listener/wazuh_listener.py (offset-poll, dedup, queue)
   -> realtime_socgraph.process_alert()
        -> mitre_resolver.py (4-tier provenance-tracked resolution)
        -> campaign_manager.py (campaign identity, CAMPAIGN_TIMEOUT)
        -> campaign_correlation_engine.py -> operation_manager.py (operation identity)
        -> mitre_feature_engine / graph_feature_engine / threat_intelligence_engine
           / detection_confidence_engine (evidence-source engines)
        -> dynamic_risk_engine / cti_confidence_engine / risk_scoring.py
        -> threat_attribution_engine.py (campaign-similarity attribution)
        -> misp_event_generator.py -> misp_sync.py -> cti_publisher.py -> MISP
   -> Neo4j (Campaign / AttackEvent / Technique / Operation / ThreatActor graph)

dashboard_api.py (separate process)
   -> reads the same Neo4j graph directly (22 routes)
   -> investigation/loop.py (evidence-aware adaptive investigation:
        NBE action selection -> evidence collectors -> confidence/
        uncertainty -> stopping -> optional XGBoost severity verdict)
   -> ml/runtime_predictor.py -> ml/models/xgb_severity.json (real trained artifact)
   -> rag/mitre_retriever.py, rag/campaign_retriever.py (Multi-RAG, this session)

frontend/ (React 19 + Vite 5)
   -> src/services/api.ts -> dashboard_api.py's 22 routes
   -> src/context/DashboardContext.tsx (30s polling, Promise.allSettled resilience)
   -> 17 components (dashboard, campaign, investigation, attribution,
      correlation, graph, prediction, query console, ...)
```

This is the architecture every prior session in this project verified
and is documented per-subsystem in `backend/ML_NBE_INTEGRATION.md`,
`backend/CAMPAIGN_CORRELATION.md`, `backend/THREAT_ATTRIBUTION.md`,
`backend/MISP_INTEGRATION.md`, `backend/DASHBOARD_API.md`,
`backend/AUDIT_LOGGING.md`, `frontend/TESTING.md`.

### Generation 1 (superseded, real code, zero runtime consumers)

Discovered this session by tracing actual import graphs, not assumed
from directory names:

```
agents/soc_agent.py (SOCAgent — a complete, different orchestrator)
   -> soc_engine/model.py, trainer.py, scorer.py, threshold.py, explainer.py, temporal_engine.py
   -> classifiers/attack_stage_mapper.py, classifiers/lightgbm_classifier.py
   -> response/playbook_generator.py, response/mitigator.py
   -> agents/output_schema.py (ThreatIntelReport)
   -> expects its own models/ dir + config.yaml (neither exists)

kill_chain/kill_chain_detector.py, attack_graph.py, event_sequence_buffer.py
   -> zero consumers outside kill_chain/ itself
```

**Zero test coverage references any of this. Zero production code path
(`realtime_socgraph.py`, `dashboard_api.py`, `listener/`) imports
`SOCAgent`, `kill_chain_detector`, `playbook_generator`, or the
LightGBM classifier.** `config.yaml` (which `SOCAgent.from_config()`
requires to be anything other than all-defaults) does not exist
anywhere in the repository.

**What survived from Generation 1 into Generation 2**: only the raw ML
building blocks — `soc_engine/model.py`'s `Autoencoder`,
`trainer.py`'s `AutoencoderTrainer`, `scorer.py`'s `SeverityScorer`,
`threshold.py`'s `AdaptiveThreshold`. These were "already implemented
... but had never actually been run" (verbatim from
`ml/ssl_pipeline.py`'s own docstring) until a later session wrote
`ml/ssl_pipeline.py` to actually train them on real CICIDS2017 data,
producing the real artifacts in `ml/models/`. `classifiers/attack_stage_mapper.py`
is the one partial exception with a real external consumer
(`kill_chain/kill_chain_detector.py`) — but that consumer itself has
no consumers, so the chain is still a dead end.

**Classification (superseded by a later, file-level disposition
audit — see `GENERATION1_DISPOSITION.md` for the full, evidence-based
breakdown)**: the blanket "Category E" label above conflated two
different things per-file. The disposition audit split it into:

- **Reusable Generation-2 primitive** (kept, actively imported):
  `soc_engine/model.py`, `trainer.py`, `scorer.py` (the `SeverityScorer`
  class only), `threshold.py` — real dependencies of
  `ml/ssl_pipeline.py`/`ml/ssft.py`.
- **Documentation/example artifact** (kept, not Generation-2, but not
  dead either): `soc_engine/temporal_engine.py` — imported by the
  capstone demo notebooks (`notebooks/capstone_demo*.ipynb`), not by
  any Generation-2 production code.
- **Historical Generation-1 prototype, proven dead** (not deleted, see
  below): `agents/*`, `soc_engine/explainer.py`, `soc_engine/model.py`'s
  `load_autoencoder` function, `soc_engine/scorer.py`'s `compute_scores`
  function, `classifiers/*`, `response/*`, `kill_chain/*`. Zero
  consumers found across production code, tests, scripts, notebooks,
  and service/config/CI files — see `GENERATION1_DISPOSITION.md` for
  the search basis per file, not just "no imports found."

**One genuine architectural cross-contamination found and fixed this
session**: `soc_engine/__init__.py` eagerly re-exported
`FeatureExplainer` (dead) and `TemporalSmoother` (notebook-only) at
package level, which meant Generation-2's real, production import
(`from soc_engine.model import Autoencoder`, used by
`ml/ssl_pipeline.py`/`ml/ssft.py`) transitively imported both as an
unrequested side effect on every load. Fixed by removing only those
two re-exports from `soc_engine/__init__.py` — neither file was
deleted, both remain fully importable via their existing direct
submodule path (`from soc_engine.explainer import FeatureExplainer`,
`from soc_engine.temporal_engine import TemporalSmoother`), which is
how `agents/soc_agent.py` and the notebooks already consume them.
Regression-tested in `backend/tests/test_soc_engine_package_boundary.py`.

This is not a claim that the dead code is bad — it reads as competent,
complete, plausibly an earlier capstone-phase design that got
superseded when the project pivoted to the Neo4j-graph-centric
architecture (Wazuh → resolver → graph → investigation) that every
later session actually built on. **Still not deleted** (~1,660 lines
across 4 directories, now fully proven dead with file-level evidence)
— see `GENERATION1_DISPOSITION.md`'s "Remaining Decision Points" and
Engineering Gaps below for why this remains a recorded decision point,
not an autonomous deletion.

---

## B. Capability matrix

| Capability | Intended | Implemented | Integrated | Tested | Live verified | Status |
|---|---|---|---|---|---|---|
| Wazuh ingestion / dedup / offset-poll | Yes | Yes | Yes | Yes | Yes (live PID, real alerts) | **A** |
| MITRE provenance-tracked resolution | Yes | Yes | Yes | Yes (unit+integration) | Yes (391 real alerts) | **A** |
| Campaign identity / reconstruction | Yes | Yes | Yes | Yes | Yes (44-event repair, 0 orphans) | **A** |
| Campaign Correlation / Operation matching | Yes | Yes | Yes | Yes (this project's session 4) | Yes | **A** |
| Evidence-aware investigation loop (NBE, confidence, stopping) | Yes | Yes | Yes | Yes (extensive) | Yes (real campaigns, Phase 21) | **A**, with a frozen negative sub-finding (content-insensitivity of NBE ranking — see Detection Paradox section) |
| Threat Attribution (campaign-similarity) | Yes | Yes | Yes | Yes (this project's session 4) | Yes | **A** for engineering correctness; **B** for accuracy (no ground-truth dataset, by design not attempted yet) |
| Threat Attribution (ThreatActor-node) | Yes | Yes | Yes (dashboard route + Neo4j write) | Partial (route tested; Cypher logic needs live Neo4j) | Yes (this session's `curl` check) | **B** |
| MISP event generation / publish | Yes | Yes | Yes | Yes (this project's session 5) | Partial (connectivity live-verified; no API key to complete an authenticated publish) | **B/F** (infrastructure-blocked on a credential, not code) |
| Dashboard API (22 routes) | Yes | Yes | Yes | Yes (this project's session 6) | Yes (live Neo4j) | **A** |
| React frontend | Yes | Yes | Yes | Yes (this project's session 7) | HTTP-boundary only (no browser automation in repo) | **B** |
| ML — XGBoost severity | Yes | Yes | Yes | Yes | Yes (real artifact, real predictions) | **A** engineering; **B** accuracy (in-sample only, disclosed) |
| ML — SSL autoencoder | Yes | Yes | Yes (feeds SSFT) | Yes | Real artifact trained on real CICIDS2017 | **A** |
| ML — SSFT (self-supervised feature transform) | Yes | Yes, incl. a real single-window runtime path (`SSFTRuntimeTransformer`, added in the SSFT phase following this audit) | Batch + runtime paths integrated *within SSFT's own domain*; cross-domain integration into XGBoost/investigation is **Outcome C** (genuine architectural incompatibility — see `backend/SSFT_RUNTIME.md`), not attempted | Yes (round-trip + 7 new runtime/parity tests) | Real artifact exists | **A** for the transform itself; **C** (proven, documented incompatibility, not a gap) for feeding CYUKTI's live campaign-severity model |
| ML — GNN model/training (`ml/gnn/model.py`, `train_gnn.py`) | Yes | Yes (hand-rolled GraphSAGE) | No | Yes (synthetic-graph unit tests only) | No | **D** — unchanged; a real dataset extractor now exists (see next row) but real training was not attempted, per `GNN_FEASIBILITY.md`'s evidence that current real data (71 campaigns, 3 of 4 severity classes observed, 70% single-event subgraphs) cannot support a defensible split |
| ML — GNN real graph extraction (`ml/gnn/campaign_graphs.py`, new) | N/A (feasibility phase) | Yes | Yes (reuses XGBoost's own `graph_analytics`/label-generation paths, no new Neo4j query) | Yes (fake-driver + 2 live-Neo4j integration tests) | Yes (`python -m ml.gnn.campaign_graphs` run live, real output in `GNN_FEASIBILITY.md`) | **A** for the extraction layer itself; feeds a training decision that remains **E** (open, see `GNN_FEASIBILITY.md` Section 13-14) |
| Multi-RAG — MITRE source | Yes | Yes | Yes | Yes | Yes (858 real techniques) | **A** |
| Multi-RAG — second source (campaign narratives) | Yes (per `rag/retriever.py`'s own design docstring) | **Yes, this session** | **Yes, this session** | **Yes, this session** | **Yes, this session** | **A** (newly) |
| Detection Paradox (structural non-contamination) | Yes | Yes | Yes | Yes | Yes (5 live UNKNOWN events, 2 sessions apart) | **A** — strongest, most defensible claim in the project (frozen) |
| Generation-1 orchestrator (`SOCAgent`, kill-chain, playbooks) | Was, historically | Yes | **No** | No | No | **C** (proven dead, file-level evidence — see `GENERATION1_DISPOSITION.md`) |
| Audit logging | Yes | Yes | Yes | Yes (this project's session 3) | Yes | **A** |
| Maintenance-worker crash (`campaign_manager.py`) | N/A (bug) | N/A | N/A | **Fixed + tested this session** | Root-caused and confirmed present before the fix | **Resolved** |

---

## C. Original capstone reconciliation

### SSL — Self-Supervised Learning

**Meaning in this project**: an `Autoencoder` (`soc_engine/model.py`)
trained with a pure reconstruction objective — `minimize ||x -
decode(encode(x))||²` — on real, unlabeled CICIDS2017 sliding-window
network-flow features (`ml/data_prep/sliding_window_preprocess.py`,
21,176 real windows). No attack/severity labels used anywhere in this
training step. **Status: A — implemented, integrated, real trained
artifact (`ml/models/autoencoder_best.pth`), tested.**

### SSFT — ambiguous acronym, two different real meanings, only one integrated

Genuine finding this session: **"SSFT" means two different things in
this codebase**, from two different architectural generations, using
the same three letters:

1. `ml/ssft.py` — **Self-Supervised Feature Transformation**: the
   trained SSL encoder transforms raw 312-dim windows into 32-dim
   learned latent vectors + reconstruction-error scores. Unsupervised,
   no labels. **Update (SSFT-focused follow-up session)**: this
   module's own batch/runtime asymmetry (no single-window inference
   path existed, unlike XGBoost's `RuntimeCampaignPredictor`) has been
   fixed — see `backend/SSFT_RUNTIME.md`. That same session also
   proved, from code and from the real production artifacts (not
   assumed), that feeding its 32-dim window-latent output into
   CYUKTI's live campaign-level XGBoost/investigation pipeline is a
   **genuine architectural incompatibility** (57 campaign-graph
   features vs. 312 network-flow-window features — different feature
   spaces, no live join key, no packet-capture pipeline in this
   repository), not a wiring gap this repository's code can close
   without fabrication. Status: **A** for the transform itself
   (batch + runtime, parity-verified); the cross-domain "feed XGBoost"
   integration this document originally described as a small wiring
   task was re-classified **C** (proven incompatibility) after deeper
   inspection — an audit correction, not a reversal of what was built.
2. `agents/soc_agent.py`'s `export_ssft_dataset()` — **Semi-Supervised
   Fine-Tuning**: pseudo-labels (HIGH→1, NORMAL→0, ambiguous excluded)
   for later supervised fine-tuning. Part of the disconnected
   Generation-1 `SOCAgent` orchestrator. Status: **E**.

### GNN — Graph Neural Network

**Intended graph representation**: campaign subgraphs (per
`ml/gnn/synthetic_graphs.py`'s structure — node types matching the
real Neo4j schema: Campaign/AttackEvent/Technique/Attacker/Host).
**Node/edge features**: `ml/gnn/graph_encoder.py`'s own curated
property list (not `ml/feature_schema.py`'s XGBoost schema directly —
correcting this document's earlier framing). **Model**: a
hand-implemented GraphSAGE (`ml/gnn/layers.py`, `model.py`,
`graph_encoder.py` — no `torch_geometric` dependency, self-contained).
**Training code**: `ml/gnn/train_gnn.py` exists and is exercised by
`tests/test_gnn.py`, on synthetic data only. **No model artifact
exists. Training was not attempted on real data this phase either —
see below for why.**

**Update (GNN feasibility phase)**: a real (non-synthetic) extraction
path now exists — `ml/gnn/campaign_graphs.py`, reusing the exact same
`graph_feature_engine.graph_analytics` (`GraphSnapshotLoader` +
`GraphBuilder`) XGBoost's own feature pipeline already depends on, plus
the exact same `risk_scoring`-based severity-labeling functions
`ml/label_generator.py` already uses for XGBoost. Run live against this
environment's real Neo4j: all 71 real campaigns extract successfully,
but the real severity-label distribution is `{Low: 64, Medium: 3,
Critical: 4, High: 0}` and 70% of real campaigns have only a single
`AttackEvent` — real data that cannot support a defensible train/val
split (no split strategy survives a class with 3 members and another
with zero, see `GNN_FEASIBILITY.md` Section 11). A real leakage source
was also found and fixed in the same phase: `graph_encoder.py`'s node
features included `risk_score`, the exact same value `severity` is a
deterministic function of — mirroring a leakage fix `ml/dataset_utils.py`
had already made for XGBoost, now applied to the GNN's encoder too.

**Status: D, unchanged for the model/training itself** — designed,
implemented at the unit level, never run on real data, and per
`GNN_FEASIBILITY.md`'s real-data evidence should not be yet. **The
extraction layer is newly A** — real, tested (including 2 live-Neo4j
integration tests), reproducible (`python -m ml.gnn.campaign_graphs`).
Node/edge encoding now also carries one-hot relationship-type
`edge_attr` (`LAUNCHED`/`HAS_EVENT`/`MATCHES`/`TARGETS`/`Unknown`) —
additive, not yet consumed by the model. Graph *features* (not
embeddings) still substitute for GNN functionality in the real
pipeline — `graph_feature_engine.py`'s hand-computed structural metrics
(density, clustering, betweenness, etc., already feeding both XGBoost
and the investigation loop's `GRAPH_STRUCTURE` evidence) are what a
GNN's learned embeddings would eventually replace or augment, not what
they currently are.

**Update (objective-decision + representation-design phases, later
sessions)**: the project owner has selected **Option E — cross-campaign
graph representation learning** (not severity classification) as the
GNN's learning objective; see `GNN_OBJECTIVE_DECISION.md` for the
comparative analysis and `GNN_REPRESENTATION_DESIGN.md` for the
resulting formal representation/objective/evaluation design. Neither
document trains a model or changes this section's status. Two
corrections/refinements to this document's and `GNN_FEASIBILITY.md`'s
earlier claims, found while tracing the code for that analysis (real
evidence, not assumption — see the two documents above for full
detail): (1) `GraphSnapshotLoader`'s query is untyped and already
incidentally captures `SIMILAR_TO`/`HAS_CAMPAIGN`/`RESEMBLES`/
`LIKELY_NEXT` for 42 of 71 real campaigns (59%) — this already affects
XGBoost's live graph features today, not only a future GNN's; (2) a
*second*, real (non-stub) `graph_similarity` field exists
(`CampaignFeatureEngine`/`CampaignDecisionEngine`, weight `0.01` in
`config.CAMPAIGN_WEIGHTS`, feeding `campaign_manager.py`'s active-
campaign continue/close decision) — distinct from the well-known
`OperationFeatures.graph_similarity` stub (`return 0.0`, weight `0.00`,
feeding operation-correlation) this document already described.
**Update (representation-implementation phase, later session)**: the
Objective A (graph autoencoding) design from `GNN_REPRESENTATION_DESIGN.md`
has now actually been trained on the real 71-campaign dataset, and a
separate, real temporal-snapshot dataset (21 campaigns, real
`AttackEvent` timestamps only) has been built for Objective C. Full
detail, exact configuration, and honest (non-overclaiming) results in
`GNN_REPRESENTATION_IMPLEMENTATION.md`. Summary, factual, not a
production-readiness claim:

```text
GNN extraction              done (prior phase)
GNN encoding                done (prior phase; edge-type encoding added)
GNN training infrastructure done (this phase: autoencoder_model.py, train_autoencoder.py)
Real graph autoencoder      done — trained, 59/12 attacker-group split, converges
                             (train loss 3.62 -> 0.665 / 200 epochs; val edge-existence
                             AUC 0.735, real generalization gap disclosed, not hidden)
Temporal representation     partial — 21/71 real campaigns have a usable temporal
                             dataset (2+ distinctly-timestamped events); snapshot
                             extraction is real and verified monotonic/leakage-free;
                             no dedicated temporal model was trained, only the
                             autoencoder's embeddings were checked for temporal
                             self-consistency (20/21 campaigns pass, one exception)
Non-circular evaluation     done — identity-aware retrieval (real, above chance-
                             baseline result) and temporal self-consistency; explicitly
                             does NOT include SIMILAR_TO/RESEMBLES anywhere
Production integration      not done, not attempted — graph_similarity (either slot)
                             remains unpopulated; no decision engine, XGBoost feature,
                             or dashboard consumes these embeddings
Research validation         partial — honest reconstruction/embedding diagnostics
                             exist; no ablation against XGBoost's 19 scalar features
                             beyond a single correlation number (r=0.581, disclosed as
                             evidence, not proof); no downstream-task improvement
                             measured (none was in scope this phase)
```

None of this changes `GraphSnapshotLoader`, XGBoost, campaign
correlation, or either `graph_similarity` slot — verified via `git
status` showing only new files added this phase (`ARCHITECTURE_AUDIT.md`'s
own update and `GNN_REPRESENTATION_IMPLEMENTATION.md` included).
Backend: 401 passed (376 baseline + 25 new, zero regressions).

### Multi-RAG

**Before this session**: one real concrete source (`rag/mitre_retriever.py`)
on top of a genuinely multi-source-capable generic engine
(`rag/retriever.py`, whose own docstring explicitly named this gap).
**After this session**: a second real source
(`rag/campaign_retriever.py`, over real historical campaign records),
wired into the same evidence-aware investigation action menu the first
source already uses, following the exact pattern the first source
established. **Status: A** (newly, this session — see `backend/MULTI_RAG.md`
for the full write-up). A third source (CTI/MISP event narratives,
which `rag/retriever.py`'s docstring also names) remains unbuilt —
genuinely possible with the same pattern, deliberately not built this
session to avoid scope creep once the "at least a second source"
capstone gap was closed with a real, tested, live-verified
implementation.

### Detection Paradox

See the dedicated mapping in Section on Detection Paradox status,
below — this is CYUKTI's strongest, already-complete, already-frozen
research contribution. Nothing was added or changed here this session.

---

## Detection Paradox status (factual mapping, no new claims)

```
Requirement: most real Wazuh alerts lack a native MITRE tag; naive
pipelines either discard them or fabricate attribution for them.
        |
CYUKTI mechanism: mitre_resolver.py's 4-tier precedence
(native > reviewed > deterministic-inference > UNKNOWN), where a
lower-confidence tier can never override a higher one.
        |
Implementation: complete. Structural non-contamination proven by
control-flow analysis (the UNKNOWN branch cannot reach Technique
creation, chain_updater, predict_next, or MISP publish).
        |
Integration status: fully live — this is the actual, only, real
attribution-resolution path in the production listener.
        |
Existing evidence: 391 real alerts (39 native, 0 reviewed, 0
inferred, 352 UNKNOWN); 5 live UNKNOWN AttackEvents inspected across
two sessions separated by a real infrastructure outage, 0 fabricated
attack_id, 0 attack-chain contamination.
        |
What remains to be demonstrated: comparative improvement over a
naive discard-or-fabricate baseline (no such experiment exists, none
should be claimed) — this is explicitly the boundary
review/novelty_argument.md already draws, unchanged by this session.
```

**Frozen sub-finding this audit did not touch**: Phase 22 root-caused
*why* the evidence-aware NBE loop's action ranking was observed
invariant across 3 real campaigns (`review/research_claims_matrix.md`,
claim 14) to `evidence/collectors/mitre_collector.py` never setting
`Evidence.relevance` (staying at its 0.0 default), which pins
`evidence_reliability`/`evidence_coverage` at exactly 0.0 after the
first step in every campaign, unconditionally — making the one
content-sensitive NBE term (`uncertainty_reduction`) content-*insensitive*
in practice. **This audit identified this as a real, precisely
diagnosed root cause but deliberately did not touch
`mitre_collector.py`'s relevance-setting logic** — it is the specific
mechanism a frozen, cited Phase 22 finding explains; changing it would
make that finding's numbers unreproducible against current code
without the finding itself being wrong. This is recorded here as a
known, real, fixable defect — explicitly deferred to a future session
that is prepared to re-run and update the Phase 21/22 narrative
alongside the fix, not silently carried out under an "obvious
engineering fix" framing.

---

## D. Frontend dead-code analysis

### `PredictionPanel.tsx`

Investigated: imports (real, correct — `useDashboard()`'s `predictions`
state, populated by `api.predictions()`, itself calling a real,
tested, live-verified backend route), API service usage (correct),
type dependencies (correct, matches `Prediction` in `types/index.ts`),
layout (never referenced in `App.tsx` or any other component — the
*only* reference to `PredictionPanel` in the entire `src/` tree is the
component's own file), git history (added already-unused in the
single squashed initial-import commit, so no evidence of prior wiring
followed by removal).

**Determination: Possibility B is partially right and C is partially
right, not A.** It is not unfinished/broken functionality (the
component itself is complete, correct, and has 4 passing tests from
this project's frontend session) — but it's not fully "obsolete" as a
duplicate either: `CampaignIntelligence.tsx`'s detail view already
shows the *same underlying prediction data* inline as a small card
when a campaign is selected, which covers the majority of what
`PredictionPanel` would add, but is not identical (no gauge
visualization, no standalone view for browsing predictions without
selecting a campaign first). **Integrating it requires a real layout
decision**: `App.tsx`'s 3-column, 2-row grid is already full; adding a
new panel means either shrinking existing panels' flex ratios (a
genuine visual-design call) or replacing one. Per this phase's explicit
instruction not to make that call silently: **left untouched, recorded
as a decision point.** If a human decides in favor, the panel itself
needs no further engineering work — it is ready to mount.

### `hooks/useWebSocket.ts`

Investigated: `backend/requirements.txt` declares `Flask-SocketIO==5.5.1`,
but **zero Python files anywhere in `backend/` reference `socketio`,
`SocketIO`, or `flask_socketio`** — the dependency is installed and
unused. `dashboard_api.py` is a plain `Flask` app with no `SocketIO(app)`
wrapper, no `@socketio.on(...)` handler, no `emit(...)` call anywhere.
The hook itself (`useWebSocket.ts`) is real, correct client code
expecting exactly this server (`io('/', {path: '/socket.io'})`,
listening for a `new_events` event, emitting `subscribe_events` on
connect) — but has no server to connect to, and (confirmed this
session, same as `PredictionPanel`) zero component imports it anywhere
in `src/`.

**Determination: Possibility "designed but not implemented," not
obsolete.** The frontend's actual real-time mechanism today is
**polling** — `DashboardContext.tsx`'s `setInterval(refreshAll, 30000)`.
`Flask-SocketIO` being a declared-but-unused dependency is direct
evidence that push-based real-time updates were a planned upgrade over
polling, not abandoned. **Not implemented this session**: building the
backend half is a real, non-trivial engineering task requiring an
architectural decision this audit will not make silently — `dashboard_api.py`
and `listener/wazuh_listener.py` are **separate processes** in the
real deployment topology (documented in `dashboard_api.py`'s own
`_load_campaign_context` docstring); a WebSocket push mechanism needs
a decision about which process hosts the Socket.IO server and how a
new-alert event crosses the process boundary (an in-process emit won't
reach a client connected to a different process's Flask instance —
this needs either merging the processes, a shared pub-sub layer, or
the listener hosting its own minimal Socket.IO server). **Classified
as: genuine future engineering work, blocked on an architecture
decision, not obsolete, not this session's scope.**

---

## E. Dependency graph

```
Graph Representation (Neo4j Campaign/AttackEvent/Technique schema — DONE)
        |
Real campaign-subgraph extraction into GNN-consumable form
(ml/gnn/campaign_graphs.py — DONE, GNN feasibility phase; reuses
XGBoost's own graph_analytics + label_generator paths, run live
against real Neo4j — see GNN_FEASIBILITY.md)
        |
  [defensible learning objective + sufficient/balanced real labels —
   MISSING, evidenced not assumed: real severity distribution is
   {Low: 64, Medium: 3, Critical: 4, High: 0} across 71 campaigns, and
   70% of real campaigns are single-event subgraphs — see
   GNN_FEASIBILITY.md Sections 8-11 for why no split strategy survives
   this, and Sections 4-5 for the open objective-choice decision]
        |
GNN training on real campaigns (model/training code exists, still only
ever run on synthetic data — correctly not attempted on real data this
phase, per the blocker above)
        |
Graph Embedding (32-or-N-dim learned vector per node — does not exist yet)
        |
Investigation Features / NBE (currently consumes hand-computed graph_feature_engine.py metrics instead)


Evidence Sources (6 real collectors — DONE)
        |
Retrieval Layer (rag/retriever.py's generic SemanticRetriever — DONE)
        |
Multi-RAG (2 real sources: MITRE + campaign narratives — DONE, this session)
        |
Evidence Quality (evidence/schema.py's confidence/relevance — DONE)
        |
Investigation (investigation/loop.py's NBE selection, confidence, stopping — DONE)


Self-Supervised Representation (SSL autoencoder, real, trained — DONE)
        |
SSFT / Self-Supervised Feature Transformation (ml/ssft.py, real
artifact, now with a genuine batch + single-window runtime path — DONE,
SSFT-focused follow-up session)
        |
  [PROVEN INCOMPATIBLE, not merely missing: feeding the 32-dim
   window-latent vector into XGBoost/GNN's campaign-graph feature set
   would require either fabricated data (feeding 57 unrelated
   campaign-graph numbers through a scaler/encoder fit on 312
   flow-statistics columns) or new live packet/flow-capture
   infrastructure this repository doesn't have — see
   backend/SSFT_RUNTIME.md for the code-level proof. Not a wiring task.]
        |
Prediction / Investigation (currently trained on the raw 57-feature
schema; will remain so until a real per-campaign network-flow data
source exists — an infrastructure question, not an engineering one)
```

**Reading this graph**: the GNN chain's first link (real
campaign-subgraph extraction) is now built and real-data-verified
(GNN feasibility phase). What blocks everything downstream of it is no
longer an engineering gap — it's the second link: a defensible
learning objective and a real label distribution that can support a
train/val split, neither of which exists yet on 71 real campaigns from
very few attacker/victim identities (the same dataset-scale problem
`review_pack/10_limitations_and_future_work.md` already disclosed for
XGBoost, now confirmed to apply at least as strongly here — see
`GNN_FEASIBILITY.md`). The Multi-RAG chain is now fully built. The
SSFT chain's remaining link is **not**, on closer inspection, a small
wiring task as this document originally characterized it — it is
blocked on the same missing live network-flow/packet-capture
infrastructure the GNN chain would also need for any campaign-level
network-traffic signal, correctly classified as an infrastructure gap
now, not an engineering one — see Engineering Gaps and Infrastructure
Gaps below (updated).

---

## F. Engineering gaps (implementable from the existing repository, no research/product decision required)

1. **Maintenance-worker crash in `campaign_manager.py`** — **fixed this
   session** (see below).
2. **SSFT's batch/runtime asymmetry — fixed in a SSFT-focused
   follow-up session.** `ml/ssft.py` previously had no single-window
   inference path, only a whole-file batch one (unlike XGBoost's
   `RuntimeCampaignPredictor`). Now has `SSFTRuntimeTransformer`,
   sharing one transform code path with the existing batch function,
   parity-verified by test. **What that session also established,
   correcting this document's original framing**: "wiring the 32-dim
   latent vector into `ml/train_xgboost.py`/`ml/gnn/train_gnn.py`" is
   **not** pure engineering — those models operate on a completely
   different feature space (57 real campaign-graph features vs. 312
   real CICIDS2017 network-flow-window features), with no live data
   source in this repository that produces a network-flow window for
   any real campaign. Concatenating the two would be fabrication, not
   integration. See `backend/SSFT_RUNTIME.md` for the full proof. Moved
   to Infrastructure Gaps below — it is genuinely blocked on data
   infrastructure that doesn't exist, not on an engineering decision.
3. **Generation-1 dead code cleanup** (`agents/`, most of `classifiers/`,
   `response/playbook_generator.py`, `response/mitigator.py`,
   `kill_chain/*`) — safe to delete by every technical measure (zero
   test coverage, zero runtime consumers, a config file dependency
   that doesn't exist) but ~1,660 lines is a large enough deletion,
   and carries enough ambiguity about whether it represents intended
   future work (e.g., real-time supervised fine-tuning via its own
   `export_ssft_dataset`) rather than pure abandonment, that this audit
   records it as a decision point rather than an autonomous removal.
   A dedicated disposition audit (`GENERATION1_DISPOSITION.md`) has
   since proven this dead-code claim file-by-file with the actual
   search basis (not just "no imports found"), and fixed the one real
   architectural cross-contamination this created (`soc_engine/__init__.py`
   eagerly re-exporting the dead `explainer.py` into Generation-2's own
   real `soc_engine.model` import path) — but the delete-vs-archive
   decision itself is still open, unchanged from this line's original
   framing.
4. **`PredictionPanel.tsx` / `useWebSocket.ts`** — see Section D; both
   require a product/architecture decision, not touched.

## G. Research gaps (require choosing a model/algorithm/hypothesis, then experimental evaluation)

1. **GNN on real campaigns** — needs a labeling/objective decision (what
   is the supervised or self-supervised target at the campaign-graph
   level?) before training can even start. **Update (GNN feasibility
   phase)**: this is no longer a hypothetical — real extraction now
   exists and real data was measured: 71 campaigns, severity
   distribution `{Low: 64, Medium: 3, Critical: 4, High: 0}`, 70%
   single-event subgraphs. The dataset-scale problem already disclosed
   for XGBoost applies here at least as strongly, now with concrete
   numbers rather than an inferred analogy — see `GNN_FEASIBILITY.md`
   for the full analysis and the two candidate objectives it leaves
   open (severity classification vs. cross-campaign representation
   learning).
2. **Attribution accuracy benchmark** — no ground-truth actor-identity
   dataset exists; building one is a data-collection/labeling
   decision, not an engineering task.
3. **RAG retrieval-quality benchmark** (either source) — needs a
   labeled query set that doesn't exist.
4. **A third Multi-RAG source over CTI/MISP narratives** — engineering-only
   once a decision is made about which real CTI text field to index
   (MISP event descriptions require a live, authenticated MISP
   instance this environment doesn't have — see Infrastructure Gaps).
5. **Any Detection Paradox comparative claim** (vs. naive
   discard-or-fabricate) — requires designing and running a comparison
   experiment; explicitly out of scope per this and every prior
   session's instructions.

## H. Infrastructure gaps (blocked on environment/data/service, not code)

1. **MISP authenticated publish** — connectivity is live-verified; no
   API key exists in this environment's `.env` (confirmed by reading
   config, not assumed) to complete a real create/update/search round
   trip.
2. **Real browser end-to-end frontend testing** — no Playwright/Cypress
   in this repository; not introduced (per explicit instruction not to
   add browser automation "unless genuinely necessary").
3. **Atomic Red Team / automated attack-execution data** — confirmed
   absent from the repository entirely (no files anywhere reference it
   beyond the word appearing in review documentation as a planned,
   never-executed capability).
4. **SSFT → campaign-level model integration** (moved here from
   Engineering Gaps after the SSFT-focused follow-up session's deeper
   inspection) — genuinely blocked on a live network-flow/packet-capture
   pipeline producing a real window per real campaign, which does not
   exist. Proven from code, not assumed: see `backend/SSFT_RUNTIME.md`.
5. **GNN real-data training** — the extraction engineering task
   (Section E) is now done (`ml/gnn/campaign_graphs.py`, GNN feasibility
   phase). What remains is not an infrastructure gap but a data-scale
   and class-balance one: 71 real campaigns is enough to pass
   `train_gnn.py`'s own 30-graph minimum, but the real severity labels
   are `{Low: 64, Medium: 3, Critical: 4, High: 0}` — no train/val
   split is defensible on that distribution regardless of extraction.
   Resolves only as CYUKTI accumulates more, more class-diverse real
   campaigns — see `GNN_FEASIBILITY.md` Sections 9-11.

---

## I. Work performed this session

Two safely implementable, clearly justified gaps were identified and
completed:

### 1. Maintenance-worker crash fix (`campaign_manager.py`)

**Root cause** (already precisely documented in
`review_pack/10_limitations_and_future_work.md`, "Current Issue 1,"
confirmed still present by reading the current code before touching
it): `create_campaign_context()` — the Phase 20 UNKNOWN-first-event
path, which deliberately bypasses `activate_campaign()` since that
function also calls `append_technique()`, which must never run for
UNKNOWN events — never set `CampaignContext.first_seen`/`last_seen`,
leaving them at the dataclass default of `None`. The maintenance
worker's `expire_active_campaigns()` then crashed every ~5 seconds
computing `now - None` for as long as such a context stayed cached.

**Fix**: `create_campaign_context()` now sets `first_seen`/`last_seen`
to the creation timestamp directly, mirroring what
`create_campaign_db()` already does server-side
(`first_seen:datetime(), last_seen:datetime()`). No change to
`activate_campaign()`, `append_technique()`, or any Phase 20
UNKNOWN-path safety guarantee.

**Files changed**: `backend/campaign_manager.py`.

**Tests**: `backend/tests/test_campaign_manager_maintenance.py` (3
tests) — reproduces the exact documented `TypeError` against the
pre-fix code (verified via `git stash`), confirms the fix resolves it,
and confirms the maintenance loop's actual expiry logic now functions
correctly for this path (previously unreachable, since the crash
happened before any comparison completed).

### 2. Second Multi-RAG source (`rag/campaign_retriever.py`)

See `backend/MULTI_RAG.md` for the full write-up. Summary: a second
real, concrete Multi-RAG source (real historical campaign records, TF-IDF,
the same generic `SemanticRetriever` engine the first source already
uses), wired into the evidence-aware investigation action menu as a
ninth action (`CAMPAIGN_NARRATIVE_SEARCH`), following the exact
architectural precedent the first source (`MITRE_SEMANTIC_SEARCH`)
already established. Does not touch the NBE scoring formula, does not
re-run or alter any frozen Phase 21/22 finding.

**Files changed**: `backend/rag/campaign_retriever.py` (new),
`backend/investigation/actions.py`, `backend/investigation/loop.py`,
`backend/tests/test_rag.py` (+4 tests), `backend/tests/test_default_wiring.py`
(+2 tests, +1 fixture-class fix).

**Integration verification**: live-verified against this
environment's real Neo4j (a direct query returned 3 real ranked
historical campaigns) and via a full real `run_investigation()` call
against a real campaign, which naturally selected all 9 actions
including the new one, in a sensible NBE order, with zero errors.

### Regression

```
cd backend && python -m pytest tests/ -q
# 345 passed (336 prior baseline + 3 maintenance-worker + 6 Multi-RAG)
```

Frontend baseline (61 tests) untouched — no frontend files were
modified this session. `review/` and `review_pack/` untouched.

---

## J. Next implementation phase (factual dependency ordering, not a subjective ranking)

**Superseded by the SSFT-focused follow-up session**: item 1 below was
originally "SSFT → model consumption wiring," characterized as pure
engineering. Deeper inspection in that session proved this is a
genuine infrastructure gap (no live network-flow data source for any
real campaign), not an engineering task — see `backend/SSFT_RUNTIME.md`.
SSFT's own internal gap (no single-window runtime path) *was* real,
in-scope engineering, and has been fixed. The ordering below is
updated accordingly.

1. **Generation-1 cleanup decision** (Section F, item 3 — renumbered
   after the SSFT correction above) — resolving whether
   `agents/`/`soc_engine`'s unused parts/`classifiers/`/`response/playbook_generator.py`/`kill_chain/`
   represent abandoned code (delete) or a deferred real-time
   fine-tuning capability (keep, and eventually connect). This decision
   doesn't block other work but should be made before the codebase
   grows further around either assumption. Now the clearest
   remaining item that is genuinely just a decision, not an
   infrastructure or research blocker. **Update (Generation-1
   disposition audit)**: the dead-code claim is now proven file-by-file
   with search evidence, and the one real architectural side effect
   (`soc_engine/__init__.py` pulling the dead `explainer.py` into
   Generation-2's real import path) has been fixed without deleting
   anything — see `GENERATION1_DISPOSITION.md`. The underlying
   delete-vs-archive decision itself is unchanged and still open.
2. **`PredictionPanel.tsx` layout decision** (Section D) — small,
   contained, zero prerequisite engineering once decided.
3. **WebSocket push architecture decision** (Section D) — the largest
   decision-gated item; genuinely useful (replaces 30s polling
   latency) but requires settling the listener/dashboard-API
   process-boundary question first.
4. **GNN real-data path** — the extraction engineering task is now done
   (`ml/gnn/campaign_graphs.py`, GNN feasibility phase). What remains is
   a labeling/objective research decision (see `GNN_FEASIBILITY.md`
   Sections 4-5, 14) *and* more class-diverse real campaign data than
   currently exists (real severity distribution `{Low: 64, Medium: 3,
   Critical: 4, High: 0}` across 71 campaigns — Section 9, 11).
5. **A live network-flow/packet-capture pipeline** (Infrastructure
   Gaps, item 4) — the actual prerequisite for both SSFT→campaign-model
   integration and any campaign-level network-traffic signal for GNN;
   correctly last, since it's the one item that is infrastructure
   before it is either engineering or research.

None of items 1-3 require a research decision to *implement* — each is
blocked on a product/architecture choice this audit is not making
silently, per this phase's explicit instructions. Item 4 is the one
genuine research-track item. Item 5 is the one genuine
infrastructure-track item that other work (SSFT's cross-domain
integration, and part of GNN's real-data path) is actually waiting on.
