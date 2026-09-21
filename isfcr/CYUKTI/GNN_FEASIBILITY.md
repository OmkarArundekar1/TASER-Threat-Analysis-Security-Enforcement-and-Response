# GNN Feasibility — Graph Extraction & Dataset Design

Companion to `ARCHITECTURE_AUDIT.md`, `GENERATION1_DISPOSITION.md`, and
(later session) `GNN_OBJECTIVE_DECISION.md` — that document analyzes
the Section 14/15 "which learning objective" open decision in depth
and, while doing so, found and corrected one claim in this document's
own Section 2 (see the strikethrough there). Read
`GNN_OBJECTIVE_DECISION.md` Section 8 before relying on this document's
description of what `GraphSnapshotLoader` does or does not capture.
Covers the first four stages of the intended GNN path (`Existing Graph
→ Graph Extraction → Graph Dataset → Node/Edge Features`) and answers,
with real-data evidence rather than assumption, whether the remaining
stages (`Learning Objective/Labels → GNN Model → Training → Artifact →
Runtime Inference`) can honestly begin yet. **They cannot, on current
real data — see Section 13.** This is documented as the correct,
successful outcome of this phase, not as a failure.

All numbers in this document were queried live against this
environment's real Neo4j instance and reproduced via
`python -m ml.gnn.campaign_graphs` (a new, real, non-synthetic
extraction command this phase adds — see Section 6). No number here is
assumed or copied from a prior document without re-verification.

---

## 1. Existing GNN inventory

`backend/ml/gnn/` (all real code, no stubs):

| File | Contents | Classification |
|---|---|---|
| `layers.py` | `SAGEConvLayer` — hand-implemented GraphSAGE mean-aggregator (`h_v = ReLU(W_self·x_v + W_neigh·mean(x_u))`), no `torch_geometric`/DGL dependency | **B** — correct, reusable, real message-passing primitive |
| `graph_encoder.py` | `encode_graph(nx.DiGraph) -> EncodedGraph` — converts any networkx graph (real or synthetic) into `(x, edge_index)` tensors; one-hot node type + curated numeric properties | **B**, with one bug fixed this phase — see Section 10 |
| `model.py` | `CampaignGNN` — stacks `SAGEConvLayer`s, mean-pools to a graph embedding, classifies severity; `batch_graphs` for multi-graph batching | **B** — correct, real, mechanically tested |
| `synthetic_graphs.py` | Generates fake campaign subgraphs + noisy severity labels, explicitly for pipeline validation only | **C** for production use, **A** for its stated purpose (pipeline testing) |
| `train_gnn.py` | `train_gnn()` — real training loop (train/val split, Adam, cross-entropy, `MIN_TRAINING_GRAPHS = 30` guard), `load_gnn()` | **B** — mechanically correct, never run against real data (see Section 8-9 for why) |
| `campaign_graphs.py` | **New this phase.** Real (non-synthetic) `(EncodedGraph, severity)` dataset builder from live Neo4j | **A** — real, tested, run against live data (Section 6-9) |
| `tests/test_gnn.py` | Encoder/layer/model/training-loop correctness, all against hand-built or synthetic graphs | Real tests, **synthetic-data scope only** (mechanics, not real-world accuracy — stated in its own docstring) |
| `tests/test_campaign_graphs.py` | **New this phase.** Extraction-layer tests: fake-driver unit tests + live-Neo4j integration tests | Real tests, **real-data scope** (see Section 6) |

**Search performed**: grepped `GNN|GCN|GAT|GraphSAGE|torch_geometric|dgl|message.passing|node classification|edge classification|graph classification` across every `.py`/`.md`/`.ipynb` file, `requirements.txt`, and `pyproject.toml` (none exists in this repo). No `torch_geometric`/DGL import or reference exists anywhere. No other GNN implementation, partial or otherwise, exists outside `ml/gnn/`.

**Runtime consumers**: **zero**, outside `ml/gnn/` itself and its own tests, before and after this phase. Nothing in `dashboard_api.py`, `investigation/`, `prediction_engine.py`, `ml/train_xgboost.py`, or `ml/runtime_predictor.py` imports anything from `ml.gnn`. This matches `ARCHITECTURE_AUDIT.md`'s prior "D — designed/prototype/synthetic-only" classification for GNN's *model/training* status, which this phase does not change (see Section 15). What changes is that the *graph extraction* status is no longer synthetic-only — see Section 6.

---

## 2. Actual CYUKTI graph schema

Traced from `neo4j_client.py` and verified live against this
environment's real database (queried via `MATCH (a)-[r]->(b) RETURN
labels(a), type(r), labels(b), count(*)` and a label-count query — not
assumed from documentation):

### Nodes (label : count : key properties, as actually observed on a real node)

| Label | Count | Key properties (sampled from a real node) |
|---|---|---|
| `Technique` | 858 | `attack_id`, `name`, `kill_chain_phases`, `platforms`, `revoked`, `is_subtechnique`, `stix_id` (MITRE ATT&CK STIX corpus — same data Multi-RAG's MITRE source indexes) |
| `Malware` | 729 | (MITRE STIX corpus) |
| `CourseOfAction` | 268 | (MITRE STIX corpus — mitigations) |
| `ThreatActor` | 189 | `actor_id`, `aliases`, `name`, `stix_id` (MITRE STIX corpus, plus attribution-linked entries) |
| `AttackEvent` | 130 | `attack_id`, `event_id`, `occurrences`, `tps`, `rule_level`, `sigma_score`, `suricata_score`, `yara_score`, `zeek_score`, `stage`, `first_seen`, `last_seen` |
| `Tool` | 95 | (MITRE STIX corpus) |
| `Campaign` | 71 | `campaign_id`, `risk_score`, `risk_level`, `dynamic_risk`, `total_tps`, `occurrences`, `status`, `predicted_next`, `prediction_confidence`, `first_seen`, `last_seen` |
| `Operation` | 43 | `operation_id`, `correlation_score`, `confidence`, `predicted_goal`, `primary_attacker`, `primary_target`, `status` |
| `Host` | 9 | `ip`, `first_seen`, `last_seen` |
| `Attacker` | 7 | `ip`, `vt_reputation`, `threat_actor_reputation`, `malware_confidence`, `tool_confidence`, `misp_confidence`, `ioc_confidence` |
| `Stage` | 6 | (kill-chain stage taxonomy nodes) |

### Relationships (start label -[type]-> end label : count, live-verified)

| Relationship | Count | Meaning |
|---|---|---|
| `Malware -[USES]-> Technique` | 10,342 | MITRE STIX corpus (Multi-RAG's MITRE source, not live-campaign data) |
| `ThreatActor -[USES]-> Technique` | 4,546 | MITRE STIX corpus |
| `CourseOfAction -[MITIGATES]-> Technique` | 1,448 | MITRE STIX corpus |
| `Tool -[USES]-> Technique` | 869 | MITRE STIX corpus |
| `ThreatActor -[USES]-> Malware` | 673 | MITRE STIX corpus |
| `Technique -[SUBTECHNIQUE_OF]-> Technique` | 477 | MITRE STIX corpus |
| `ThreatActor -[USES]-> Tool` | 472 | MITRE STIX corpus |
| `Technique -[REVOKED_BY]-> Technique` | 149 | MITRE STIX corpus |
| **`Campaign -[HAS_EVENT]-> AttackEvent`** | **130** | **Live campaign data** — a campaign's own attack chain |
| **`Campaign -[SIMILAR_TO]-> Campaign`** | **122** | **Live campaign data** — cross-campaign correlation (campaign_correlation_engine.py) |
| **`AttackEvent -[MATCHES]-> Technique`** | **119** | **Live campaign data** — MITRE resolution (11 of 130 events, ~8.5%, have no match — the frozen Detection Paradox's UNKNOWN tier) |
| **`Attacker -[LAUNCHED]-> Campaign`** | **71** | **Live campaign data** |
| **`Campaign -[TARGETS]-> Host`** | **71** | **Live campaign data** |
| **`Operation -[HAS_CAMPAIGN]-> Campaign`** | **35** | **Live campaign data** — only 35 of 71 campaigns (49%) belong to a correlated Operation |
| **`Campaign -[LIKELY_NEXT]-> Technique`** | **18** | **Live campaign data** — prediction_engine's stored next-technique prediction |
| `Technique -[BELONGS_TO]-> Stage` | 16 | Kill-chain taxonomy |
| `ThreatActor -[REVOKED_BY]-> ThreatActor` | 6 | MITRE STIX corpus |
| **`Campaign -[RESEMBLES]-> ThreatActor`** | **5** | **Live campaign data** — threat_attribution_engine's output |
| `Technique -[NEXT_TECHNIQUE]-> Technique` | 3 | (sparse, likely test/legacy data) |
| `Malware -[REVOKED_BY]-> {Malware,Tool}` | 2 | MITRE STIX corpus |

**Bold rows are the live, campaign-derived graph** — the rest is the
static MITRE ATT&CK STIX corpus, already fully covered by Multi-RAG's
MITRE source, not campaign-specific structure a GNN would reason over.

### What the existing extraction path (`GraphSnapshotLoader` + `GraphBuilder`, `graph_feature_engine.py`) already captures

Real, live, production code — not built this phase — already extracts
a **per-campaign subgraph** from Neo4j via one Cypher query:
`(Attacker)->(Campaign)->(AttackEvent)->(Technique)`, `(Campaign)->(Host)`.
This is the exact same extraction `campaign_feature_engine.py` already
depends on for XGBoost's 57 tabular features (via
`graph_analytics.extract_features`), and the same shape
`ml/gnn/graph_encoder.py`'s `encode_graph` was already written to
consume (confirmed by reading its docstring against the real query).

~~**Not captured by this per-campaign extraction**: `Operation
-[HAS_CAMPAIGN]-> Campaign`, `Campaign -[SIMILAR_TO]-> Campaign`,
`Campaign -[LIKELY_NEXT]-> Technique`, `Campaign -[RESEMBLES]->
ThreatActor`~~ — **correction, `GNN_OBJECTIVE_DECISION.md` Section 8
(later session)**: this claim was wrong. `GraphSnapshotLoader`'s
`(a)-[r1]->(c)` and `(c)-[r4]->(h)` Cypher patterns are untyped (no
relationship-type or node-label filter), so all four of these
relationship types are already incidentally captured whenever they
exist — verified concretely on a real campaign (5 `SIMILAR_TO` + 1
`HAS_CAMPAIGN` returned by the unmodified loader) and quantified (42 of
71 real campaigns, 59%, affected). This does **not** mean the
cross-campaign correlation structure (Section 4's reasoning for what a
GNN could add) is cleanly available today — the capture is accidental,
1-hop only, untyped-until-the-edge-type-encoding-phase (now generically
`Unknown`), and asymmetric (only *incoming* `SIMILAR_TO`/`HAS_CAMPAIGN`
and *outgoing* `RESEMBLES`/`LIKELY_NEXT` — never a campaign's own
outgoing `SIMILAR_TO` edges). It also means **this was never a
GNN-only question**: `campaign_feature_engine.py`'s XGBoost graph
features already inherit this same inconsistency today, for the same
59% of campaigns, since both call the identical `graph_analytics`
singleton. See `GNN_OBJECTIVE_DECISION.md` Section 8 for the full
analysis and Section 10 for loader-design alternatives (still not
implemented). `GraphSnapshotLoader`'s query was **not modified** in
either phase — it is a live, working, shared dependency of XGBoost's
own feature pipeline; extending or correcting its scope is a design
question about what a "campaign graph" should mean, not a safe,
narrowly-scoped engineering change (see Section 14).

---

## 3. Proposed GNN role

`ml/gnn/model.py`'s own docstring (written when this code was
originally authored, before this phase) already states an intended
role, in two parts:

1. **Primary, supervised**: predict campaign severity from the
   attack-subgraph, "the same target XGBoost predicts from tabular
   features, so the two models' predictions can be compared honestly
   on the same label."
2. **Secondary, the docstring calls "more important"**: the graph
   **embedding** itself, for "embedding-space similarity between the
   current campaign and historical ones as a structural relevance
   signal that's independent of (and complements) the pure
   technique-overlap coverage `threat_attribution_engine.py` already
   computes."

This phase did not invent this role — it was already declared in code
before this phase began. What this phase adds is real-data evidence
for whether either half is currently well-founded (Section 4-5).

---

## 4. Candidate learning objectives

Evaluated against the real schema (Section 2), not chosen for ease:

| Option | Description | Evidence for/against |
|---|---|---|
| **A — Node-level** | Classify individual nodes (e.g. Technique risk within a campaign) | No per-node label exists anywhere in the schema for any node type. Would require inventing one. Not evidenced. |
| **B — Edge-level** | Predict relationship probability (e.g. will this Attacker→Campaign edge exist) | No task in CYUKTI's stated purpose calls for this; campaign/attacker linkage is already deterministic (Wazuh alert → attacker IP), not something to predict. Not evidenced. |
| **C — Graph-level (severity)** | Campaign subgraph → severity class | **This is what `model.py`/`train_gnn.py` already implement.** But: (a) the label is a deterministic function of `risk_score`, itself computed largely from the same `AttackEvent`/`Technique` structure being encoded — real risk of redundancy, not new signal (Section 8, 10); (b) real severity labels are severely imbalanced (Section 9); (c) the per-campaign subgraph this targets (`GraphSnapshotLoader`'s scope) is nearly identical in information content to the `GraphFeatures` summary statistics (density, degree, clustering, etc.) `graph_feature_engine.py` already computes and feeds to XGBoost — a GNN reaching the same conclusion from the same subgraph via learned aggregation instead of hand-written formulas is a real but narrow value proposition (see Section 5). |
| **D — Link prediction (next-technique)** | Current graph → likely next technique/edge | **A real, already-persisted signal exists**: `Campaign -[LIKELY_NEXT]-> Technique` (18 live edges) is prediction_engine's *existing*, non-GNN next-technique prediction (a Markov-chain-style `predict_next`/`predict_next_readonly`, per `ml/label_generator.py`'s own use of it). A GNN version of this would need to beat or meaningfully complement that existing baseline — not evaluated this phase (would require a research decision, Section 14). |
| **E — Representation learning** | Campaign graph → embedding, used for similarity (not classification) | **The `model.py` docstring's own stated "more important" role.** Does not require a severity label at all — sidesteps the label-imbalance/leakage risk of Option C entirely. Closest in spirit to CYUKTI's existing SSL/SSFT precedent (`ml/ssl_pipeline.py`, `ml/ssft.py` — self-supervised, no attack/severity labels). **Best-evidenced candidate for what a GNN could add that XGBoost genuinely cannot** — see Section 5. |

**This phase does not silently pick one.** Option C is already coded
and has the clearest implementation path, but Section 9-10's real-data
findings mean it cannot defensibly train yet regardless. Option E is
the best-evidenced complementary role but has not been implemented as
a self-supervised training objective anywhere in this repository — it
would be new research/engineering work, not a decision this phase
should make unilaterally. Option D has a real existing non-GNN
baseline to compare against, which is itself a benchmarking task
explicitly out of scope for this phase (`GNN_FEASIBILITY.md` Section
18 of the mission — no benchmarking). **This is recorded as an open
decision in Section 15.**

---

## 5. Recommended / unresolved objective

**Recommendation, not a decision made on the project's behalf**: Option
E (representation learning for cross-campaign structural similarity,
likely over the `SIMILAR_TO`/`RESEMBLES`/`HAS_CAMPAIGN` correlation
edges Section 2 shows are **not** currently captured by
`GraphSnapshotLoader`) is better evidenced than continuing to pursue
Option C (severity classification) on the current per-campaign subgraph,
for two concrete reasons:

1. **Non-redundancy**: Option C's current subgraph scope
   (`Attacker→Campaign→AttackEvent→Technique`, `Campaign→Host`) is the
   same subgraph `graph_feature_engine.GraphFeatures` already
   summarizes into ~20 hand-engineered scalar features
   (`graph_density`, `average_degree`, `attack_chain_depth`,
   `campaign_complexity`, `structural_risk`, ...) that XGBoost already
   consumes. A GNN learning severity from the *same* subgraph is asking
   whether a learned aggregation beats a hand-engineered one on the
   *same* information — a real, testable question, but a narrower one
   than "what can a GNN see that XGBoost cannot."
2. **A GNN over the correlation graph** (`Campaign-SIMILAR_TO-Campaign`,
   122 real edges; `Operation-HAS_CAMPAIGN-Campaign`, 35 real edges;
   `Campaign-RESEMBLES-ThreatActor`, 5 real edges) would see structure
   genuinely absent from any single campaign's tabular feature row —
   this is the one form of "graph" information CYUKTI's existing
   XGBoost/graph-feature pipeline architecturally cannot see, because
   it operates per-campaign, not across campaigns.

**This is not adopted as a decision.** Choosing between "improve
Option C" and "build Option E on a wider graph scope" is exactly the
kind of research decision this phase's own operating principle
forbids making silently. Recorded in Section 15 as the primary open
decision point.

---

## 6. Graph extraction design

**New this phase**: `backend/ml/gnn/campaign_graphs.py` —
`list_campaign_ids()`, `build_campaign_graph_sample(campaign_id)`,
`build_real_campaign_dataset()`. Deliberately thin: it does not
re-implement extraction. It calls the same two already-production
paths XGBoost's own pipeline depends on:

```
Neo4j (live)
   |  neo4j_client.get_campaign_context_data(campaign_id)   [real risk_score]
   |  graph_feature_engine.graph_analytics.load_graph(campaign_id)
   |    -> GraphSnapshotLoader.load_campaign_graph -> GraphBuilder.build
   v
networkx.DiGraph (real campaign subgraph)
   |  ml/gnn/graph_encoder.encode_graph
   v
EncodedGraph (x: node-feature tensor, edge_index)
   |  risk_scoring.normalize_risk_score / risk_level_from_score
   |    (the exact functions ml/label_generator.py already uses for XGBoost)
   v
CampaignGraphSample(campaign_id, graph, severity, risk_score, num_events)
```

Reproducible command (fits the existing `ml/gnn/` module-with-`__main__`
convention `train_gnn.py` already uses):

```bash
cd backend
python -m ml.gnn.campaign_graphs
```

Real output against this environment's live database, this phase:

```json
{
  "campaigns_in_neo4j": 71,
  "samples_built": 71,
  "severity_distribution": {"Low": 64, "Medium": 3, "Critical": 4},
  "events_per_campaign_distribution": {"1": 50, "2": 10, "3": 2, "4": 2, "5": 2, "6": 3, "8": 1, "10": 1}
}
```

Every one of the 71 real campaigns has a `risk_score` and produces a
valid sample — extraction itself has zero missing-data failures on
real data. The dataset-sufficiency problem (Section 9) is about label
diversity and subgraph richness, not extraction reliability.

Deliberately **not** included in this phase: an export/persistence
step (e.g. writing an `.npz`/parquet dataset to disk, as
`ml/dataset_builder.py`/`ml/ssft.py`'s batch path do). Persisting a
dataset implies it is ready to train on; Section 9-10 show it is not
yet. Adding persistence is a one-function addition once the label
question (Section 15) is resolved — not built prematurely here.

---

## 7. Feature design

**Node features**: reused, not re-derived — `ml/gnn/graph_encoder.py`
was already written (before this phase) to read exactly the properties
`neo4j_client.py` actually writes (Section 2's property tables confirm
this against real nodes): a one-hot node-type vector (`Attacker`,
`Campaign`, `AttackEvent`, `Technique`, `Host`, `Unknown`) plus 10
curated numeric properties (`occurrences`, `total_tps`, `tps`,
`rule_level`, `vt_reputation`, `threat_actor_reputation`,
`malware_confidence`, `tool_confidence`, `misp_confidence`,
`ioc_confidence` — `risk_score` removed this phase, Section 10).
Missing/non-numeric values default to `0.0` rather than raising, since
not every node type carries every property (verified true against
Section 2's real property lists — e.g. `Host` nodes have none of the
10 numeric properties).

**No new feature-extraction logic was written this phase** — this
phase only removed one leakage-causing feature (Section 10). One
authoritative mapping continues to exist in `graph_encoder.py`; nothing
duplicates it.

**Edge features**: at the time this phase concluded, none existed —
`edge_index` carried structure only, with the real relationship *type*
(`HAS_EVENT` vs `MATCHES` vs `TARGETS`) discarded. **Addendum (added in
a later session, does not change this phase's verdict):** this was the
one item Section 14 flagged as a pure engineering gap rather than a
research decision, and it has since been closed —
`graph_encoder.py` now emits `edge_attr` (`EDGE_TYPES = ["LAUNCHED",
"HAS_EVENT", "MATCHES", "TARGETS", "Unknown"]`, one-hot per edge,
row-aligned with `edge_index`, symmetrized edges keeping their forward
type rather than inventing a reverse one). Edges are still symmetrized
for message passing (`encode_graph`'s existing, pre-this-phase
behavior). `edge_attr` is **not yet consumed** by `SAGEConvLayer`/
`CampaignGNN` — the mean aggregator in `layers.py` remains
edge-type-agnostic — so this is additive plumbing, not a model change,
and does not alter Section 13's training-readiness verdict (labels and
dataset size are still the blocker, not feature availability). 5 new
regression tests in `tests/test_gnn.py`; live re-verification via
`python -m ml.gnn.campaign_graphs` against this environment's real
Neo4j reproduced the identical 71-sample/severity-distribution output
in Section 6, confirming the change is transparent to existing
consumers.

---

## 8. Label availability

Traced from code (`ml/label_generator.py`, reused verbatim by this
phase's `campaign_graphs.py` — Section 6), not assumed:

1. **What labels exist?** Only `severity` (`Low`/`Medium`/`High`/`Critical`),
   the same label XGBoost trains on.
2. **Where do they come from?** `risk_scoring.risk_level_from_score(risk_scoring.normalize_risk_score(campaign.risk_score))`.
3. **Are they ground truth?** **No.** `risk_score` is an unbounded
   running sum CYUKTI itself accumulates from `AttackEvent.tps`
   (`neo4j_client.create_attack_event`: `c.risk_score = coalesce(c.risk_score,0) + $tps`)
   — a heuristic derived from the same detection pipeline being
   modeled, not an independently verified outcome (e.g. an analyst's
   confirmed verdict, or a labeled research dataset). Same caveat
   `ml/dataset_utils.py` already documents for XGBoost's identical
   label — not a new limitation this phase introduces.
4. **Derived from MITRE?** Indirectly — `tps` inputs to `risk_score`
   partly reflect MITRE-resolved technique severity, but the label
   itself is a numeric-threshold function, not a MITRE taxonomy lookup.
5. **Inferred?** Yes, in the sense of (3) — a heuristic proxy, not a
   verified ground truth.
6. **Campaign-level or event-level?** Campaign-level only. No
   per-event or per-technique label exists.
7. **Temporal?** No — `severity` is a single, non-time-indexed label
   per campaign (whatever `risk_score` happens to be as of the
   extraction call — see Section 6's `force_reload` default of
   `False`, meaning cached graphs can be extracted from stale state
   unless explicitly refreshed).
8. **Class diversity — real, measured**: **3 of 4 classes observed**
   (`Low`: 64, `Medium`: 3, `Critical`: 4) — **`High` has zero real
   examples** in this environment's current data.
9. **Graph diversity — real, measured**: 50 of 71 campaigns (70%) have
   exactly **one** `AttackEvent` (Section 6's `events_per_campaign_distribution`)
   — a near-star-shaped 4-5-node subgraph
   (`Attacker→Campaign→AttackEvent→Technique`, `Campaign→Host`), with
   almost no multi-hop structure for message passing to exploit. Only
   9 campaigns (13%) have 4+ events.

---

## 9. Dataset size

**Measured live this phase**, not assumed from a prior session's
snapshot (which is explicitly why the mission requires re-querying):
**71 Campaign nodes, 130 AttackEvent nodes, 43 Operation nodes, 858
Technique nodes** (Section 2). `train_gnn.py`'s own
`MIN_TRAINING_GRAPHS = 30` gate would technically pass (71 ≥ 30) — but
that gate checks raw count only, not per-class count or split
feasibility, which is where the real problem is (Section 8, item 8;
Section 11).

---

## 10. Leakage risks

**One real leakage source found and fixed this phase** (not merely
documented — see the Operating Principle's "implement safe engineering"
instruction, and the direct precedent already set for XGBoost):

`ml/gnn/graph_encoder.py`'s `NUMERIC_PROPS` list included `"risk_score"`
as a `Campaign`-node feature. Because `severity` (the label
`train_gnn.py`/`model.py` are wired to predict) is a **deterministic
thresholded function of that exact same `risk_score`**
(`risk_scoring.risk_level_from_score` — Section 8), encoding it as a
node feature would let the GNN trivially reconstruct the label from
one input number instead of learning from graph structure — the
identical leakage `ml/dataset_utils.py`'s `LEAKAGE_COLUMNS = ["risk_score"]`
already documents having found and excluded for XGBoost's tabular
features (`# risk_score is not a label, but severity is a deterministic
function of it ... discovered while auditing real campaign data:
training on risk_score let the model trivially reconstruct the label`).
The GNN's node-feature encoder had not yet had the equivalent fix
applied. **Fixed this phase**: `risk_score` removed from
`NUMERIC_PROPS`, with a regression test
(`test_encode_graph_excludes_risk_score_as_a_leakage_source` in
`tests/test_gnn.py`) and an end-to-end guard
(`test_build_campaign_graph_sample_does_not_leak_risk_score_into_graph_features`
in `tests/test_campaign_graphs.py`) asserting an implausible sentinel
`risk_score` value never appears in the encoded feature matrix, even
though the real Campaign node genuinely carries it.

**Other leakage risks considered, not yet applicable (dataset not
trained yet) but relevant once it is**:

- **Same campaign in train and test**: not currently a risk — each
  sample is one whole campaign graph, so a campaign-level split
  (Section 11) naturally avoids within-campaign leakage. Would become a
  risk only if extraction were changed to sample sub-windows of a
  single campaign's graph.
- **Same attacker/victim pair crossing splits**: a real risk *once
  training starts* — Section 2 shows only 7 real `Attacker` nodes and 9
  `Host` nodes across 71 campaigns, meaning many campaigns necessarily
  share an attacker or host. A random campaign-level split could still
  let the same attacker's `vt_reputation`/`threat_actor_reputation`
  appear identically in both train and val, letting the model
  memorize attacker identity rather than generalize. Not fixed this
  phase (no split has been implemented — Section 11) but flagged for
  whoever implements the split.
- **Temporal leakage**: `severity` is read as of extraction time, not
  pinned to campaign closure — extracting the same campaign at two
  different points in its lifecycle would yield two different labels
  for structurally similar (early) graphs. Relevant once a real
  training loop exists; not applicable to a one-shot dataset snapshot.
- **MITRE-derived labels**: not applicable — no label in this dataset
  is derived from MITRE technique identity directly (see Section 8,
  item 4).

---

## 11. Train/validation/test split strategy

**Not implemented this phase** — correctly so, given Section 8-9's
findings. Analysis for whoever picks this up next:

- **Random per-sample split** (what `train_gnn.py` currently does via
  `sklearn.train_test_split(..., stratify=y)`): would very likely fail
  outright or produce a meaningless split on the real label
  distribution — stratifying 3 classes (`High` has zero real examples,
  so effectively 3, not 4) with sizes 64/4/3 at `val_fraction=0.2`
  leaves the `Medium` class (3 examples) needing `0.6` of an example in
  the validation set, which `sklearn` will either round to 0 (making
  that class untestable) or raise on directly, depending on version.
- **Campaign-level split**: not meaningfully different from
  per-sample here, since one sample already *is* one campaign — the
  real risk is the shared-attacker/shared-host leakage noted in
  Section 10, not within-campaign leakage.
- **Temporal split** (train on earlier campaigns, validate on later
  ones): more defensible for a system CYUKTI's own investigation loop
  treats as real-time, but with only 71 campaigns total and 4-7
  `Critical`/`Medium` examples, a temporal cutoff would likely put
  entire minority classes on only one side of the split.
- **Conclusion**: at 71 total samples with a 64/3/4/0 class split, **no
  split strategy makes the validation metric meaningful** — this is
  the same conclusion `ml/dataset_utils.py`'s own `MIN_TRAINING_ROWS =
  30` guard and its surrounding disclosure ("in-sample only") already
  reached for XGBoost, now confirmed to apply at least as strongly to
  a graph-level classifier with an even smaller effective minority-class
  count.

---

## 12. Framework/dependency status

**No change made or needed this phase.** `requirements.txt` already
has `torch==2.10.0` and `networkx==3.6.1`; no `torch-geometric` or
`dgl` entry exists, and `ml/gnn/layers.py`'s own docstring already
documents the decision *not* to add one ("not installed / fragile to
pin against this environment's torch build ... a hand implementation is
more robust than a finicky optional dependency") — a decision made
before this phase, re-verified as still correct: mean-aggregator
message passing is a few lines of `index_add_`, and introducing a large
new dependency is not justified for that. This phase introduces no new
third-party dependency.

---

## 13. Can GNN training begin? — Engineering blockers vs. infrastructure vs. research decisions

Per the mission's explicit gate (all of these must be true before
training):

| Requirement | Status |
|---|---|
| Real graph extraction works | **Yes** (Section 6, new this phase) |
| Node/edge features defined | **Yes** — node features (Section 7); edge/relationship-type features added in a later session (Section 7 addendum) — neither changes the remaining rows below |
| Learning objective is defensible | **No** — genuinely open, Section 4-5 |
| Labels are real/defensible | **No** — heuristic proxy label, severely imbalanced, one class entirely unobserved (Section 8) |
| Leakage is controlled | **Yes, for the one found source** (Section 10) — but the split-strategy leakage risks (Section 10, attacker/host sharing) are unaddressed because no split exists yet |
| Dataset size sufficient for a meaningful proof of concept | **No** — 71 total, 3/4 classes present, minority classes of 3 and 4 (Section 9, 11) |
| Train/val/test strategy defensible | **No** — no strategy survives the real class distribution (Section 11) |
| Framework available or safely introduced | **Yes** (Section 12) |

### Engineering blockers
- ~~Edge-type/edge-feature encoding does not exist~~ **Closed in a later
  session** — `encode_graph` now emits one-hot relationship-type
  `edge_attr` (Section 7 addendum). Did not change the training-readiness
  verdict, as predicted.
- No dataset persistence/export step exists yet (deliberately, Section 6).

### Infrastructure blockers
- None found that block extraction itself — live Neo4j is reachable in
  this environment and every real campaign extracts successfully
  (Section 6). The infrastructure gap is one level up: too few real
  campaigns exist yet, and the ones that do skew heavily toward `Low`
  severity with single-event attack chains (Section 8-9). This is a
  data-accumulation problem, not a missing-system problem — it
  resolves as CYUKTI ingests more/more-severe real campaigns over
  time, not through engineering work in this repository.

### Research decisions (this phase does not make these)
1. **Which learning objective** (Section 4-5) — continue Option C
   (severity classification on the existing per-campaign subgraph, real
   but partially redundant with XGBoost's existing graph-summary
   features) vs. pursue Option E (self-supervised representation
   learning over the cross-campaign correlation graph, better-evidenced
   as genuinely complementary to XGBoost but unimplemented, and would
   require deciding what "similarity" should mean without a label).
2. **Whether to extend `GraphSnapshotLoader`'s query scope** to include
   `SIMILAR_TO`/`RESEMBLES`/`HAS_CAMPAIGN` edges (Section 2, 5) — a
   design decision about what a GNN's "graph" should mean, touching a
   shared, working, live production dependency of XGBoost's own
   feature pipeline.
3. **What counts as "enough" real data** to revisit this — no
   specific campaign count or class-balance threshold is set here; that
   is itself a research/statistical decision, not an engineering one.

## Answer: **NO** — GNN training cannot defensibly begin yet.

Justification, strictly from repository evidence: real labels are
usable-but-marginal in *count* (71 ≥ `MIN_TRAINING_GRAPHS`) and
unusable in *distribution* (one of four classes has zero real
examples; the smallest present class has 3), and 70% of real campaign
subgraphs are near-trivial single-event structures offering little for
message passing to exploit. This is a dataset-sufficiency conclusion,
not a code-quality one — every mechanical piece of the pipeline
(extraction, encoding, model, training loop) is real, tested, and
correct on the data shape it's given (Section 1, 6-7).

---

## 14. What requires a human/research decision (do not silently resolve)

1. **Objective choice** — Section 4-5, 13, item 1.
2. **`GraphSnapshotLoader` scope extension** — Section 2, 13, item 2.
   Not attempted this phase because it modifies a live, shared,
   working dependency of XGBoost's feature pipeline outside this
   phase's "minimal safe engineering" mandate.
3. **Whether/when to revisit** once more real campaign data has
   accumulated, and what class-balance bar should be required before
   training is attempted — Section 13, item 3.
4. ~~**Edge-type encoding design**~~ (Section 7, 13) — resolved in a
   later session: one-hot over the 4 relationship types this
   extraction scope actually produces (`LAUNCHED`, `HAS_EVENT`,
   `MATCHES`, `TARGETS`) plus an `Unknown` fallback, symmetrized edges
   keep their forward type. As anticipated, this did not require
   revisiting the overall verdict.

## 15. What can safely be implemented now (and was)

- Real, live-Neo4j graph extraction reusing existing production paths
  (`ml/gnn/campaign_graphs.py`, Section 6) — safe because it only reads
  data and calls already-tested existing functions, makes no design
  decision about scope beyond what `GraphSnapshotLoader` already
  extracts.
- The `risk_score` leakage fix in `graph_encoder.py` (Section 10) —
  safe because it is a direct, precedented mirror of a fix already made
  and disclosed for the structurally identical XGBoost case, not a new
  judgment call.
- Regression tests for both, plus a drift guard tying
  `campaign_graphs.py`'s severity mapping to `ml/label_generator.py`'s
  (Section 6, 10).

---

## Tests

```bash
cd backend
python -m pytest tests/ -q
```

**371 passed** at the end of this phase (356 baseline at the start +
15 new: 13 in `tests/test_campaign_graphs.py`, 1 replaced + 1 added in
`tests/test_gnn.py`). Zero regressions. Two of the 13 new tests are
live-Neo4j integration tests (`requires_live_neo4j`, same skip
convention as `tests/test_dashboard_api_routes.py`) — both ran (not
skipped) and passed against this environment's real database.

**Addendum (later session, edge-type encoding):** **376 passed** (371
+ 5 new in `tests/test_gnn.py` covering `edge_attr` shape/alignment,
real-type reflection, symmetrized-edge type consistency, and the
`Unknown` fallback for both an unrecognized real type and a missing
`relationship` attr). `python -m ml.gnn.campaign_graphs` re-run against
this environment's live Neo4j reproduced the identical 71-sample output
in Section 6, confirming the addition is transparent to existing
consumers.

## Runtime verification

`ml.gnn.campaign_graphs`, `ml.gnn.graph_encoder`, `ml.gnn.model`,
`ml.gnn.train_gnn` all import cleanly. `python -m ml.gnn.campaign_graphs`
runs end-to-end against live Neo4j (Section 6's real output). No
Generation-1 import is required anywhere in this phase's code. Phase
21/22 frozen research findings and the Detection Paradox were not read
or modified.
