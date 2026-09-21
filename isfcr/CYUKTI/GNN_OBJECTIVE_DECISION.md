# GNN Objective Decision — Option C vs Option E

Companion to `GNN_FEASIBILITY.md` (which established that GNN training
cannot defensibly begin on current data — that conclusion is not
revisited here). This document exists to give the project owner
evidence-based material for the one open question that phase
deliberately did not resolve: **which learning objective, if any,
should the eventual GNN work pursue — Option C (severity
classification) or Option E (cross-campaign representation learning)?**

**This document does not choose.** It traces the real code, quantifies
the real dataset, and surfaces one previously-undocumented correction
to `GNN_FEASIBILITY.md`'s own claims, discovered while tracing
`GraphSnapshotLoader` for this analysis. No production code was
changed. No model was trained. No label was fabricated.

All numbers below were re-queried live against this environment's real
Neo4j instance during this phase (via throwaway, uncommitted scripts
that called only existing, already-tested production functions —
`ml.gnn.campaign_graphs.build_real_campaign_dataset`,
`graph_feature_engine.GraphSnapshotLoader`, and direct read-only Cypher
— then deleted; `git status` at the end of this phase shows no
production file touched). Backend test suite: **376 passed**, i.e.
unchanged from the end of the edge-type-encoding phase — confirmed by
re-running `python -m pytest tests/ -q` after this phase's (read-only)
work.

---

## 1. Executive Summary

- XGBoost already consumes **19 raw hand-engineered graph-summary
  statistics** (density, degree, clustering, community structure, etc.
  — Section 3) plus one further composite (`graph_similarity`) computed
  from the *identical* per-campaign subgraph the GNN would encode.
  Option C (GNN severity classification on that same subgraph) is
  therefore substantially — not entirely — redundant with what already
  exists; **Section 4** quantifies exactly which structural signal is
  and isn't already captured by hand-engineered features.
- Every real "similarity" mechanism already in production —
  `OperationDecisionEngine` (campaign→operation attach),
  `ThreatAttributionEngine` (campaign→actor), the Neo4j-native
  `SIMILAR_TO`/`RESEMBLES` writers (called from `realtime_socgraph.py`
  on every alert), and the campaign-narrative TF-IDF retriever — computes
  similarity from a **flat set or sequence of technique IDs** (plus
  attacker/victim/time identity). **None of them reason over graph
  topology.** The one field explicitly reserved for that,
  `OperationFeatures.graph_similarity`, is a literal `return 0.0` stub
  with a `0.00` weight in `OperationDecisionEngine.weights` — present in
  the architecture, never implemented (Section 6, Section 9).
- **Correction to `GNN_FEASIBILITY.md` Section 2, discovered this
  phase**: `GraphSnapshotLoader.load_campaign_graph`'s Cypher patterns
  `(a)-[r1]->(c)` and `(c)-[r4]->(h)` are **untyped** — no relationship
  or node-label filter — so they already incidentally pull
  `SIMILAR_TO`, `HAS_CAMPAIGN`, `RESEMBLES`, and `LIKELY_NEXT`
  edges/nodes into the "per-campaign" subgraph whenever they exist.
  Verified concretely on a real campaign (`CAMP_7331E223`): the
  unmodified loader returned 5 `SIMILAR_TO` and 1 `HAS_CAMPAIGN`
  relationship alongside the intended `LAUNCHED`/`HAS_EVENT`/`MATCHES`/
  `TARGETS`. A direct Cypher count confirms **42 of 71 real campaigns
  (59%) are affected**. Because XGBoost's graph features and the GNN's
  dataset both call the same `graph_analytics` singleton, **this is not
  a hypothetical for a future GNN — it already affects XGBoost's
  currently-deployed graph features today.** Section 8 covers this in
  full; it changes both options' analysis and is this document's single
  most important finding.
- Option E's central appeal — "a GNN sees cross-campaign structure
  XGBoost architecturally cannot" — is **weaker than
  `GNN_FEASIBILITY.md` assumed**, because that structure is not
  cleanly absent from the current pipeline; it's present but
  *accidentally, inconsistently, and only 1-hop*. Option E's central
  **risk** — discovered this phase, Section 9 — is that the only
  real "ground truth" available for what "similar campaigns" means
  (`SIMILAR_TO`, `RESEMBLES`, or operation co-membership) is itself
  derived from the same technique-overlap heuristic Option E would need
  to be evaluated against, making a naive contrastive/similarity
  objective circular by construction.
- Neither option is free of open research questions. Both are
  documented in full below without a recommendation.

---

## 2. Current GNN Architecture

Unchanged since `GNN_FEASIBILITY.md` plus this phase's addendum there:
`ml/gnn/layers.py` (hand-implemented GraphSAGE mean-aggregator, no
`torch_geometric`), `ml/gnn/graph_encoder.py` (`encode_graph`: node
one-hot type + 10 curated numeric properties, `risk_score` excluded as
a leakage source; **as of the edge-type-encoding phase**, edges now
carry a one-hot `edge_attr` over `{LAUNCHED, HAS_EVENT, MATCHES,
TARGETS, Unknown}`, not yet consumed by the model), `ml/gnn/model.py`
(`CampaignGNN`: stacked `SAGEConvLayer` → mean-pool → severity
classifier head, plus the embedding itself as a first-class output),
`ml/gnn/train_gnn.py` (real train/val loop, `MIN_TRAINING_GRAPHS = 30`
guard), `ml/gnn/campaign_graphs.py` (real, live-Neo4j
`(EncodedGraph, severity)` dataset builder — **the module whose
Cypher-adjacent behavior this phase re-examined**, Section 8). Zero
runtime consumers outside `ml/gnn/` and its own tests — unchanged.

---

## 3. Existing CYUKTI Graph Signals — Traced From Code, Not Filenames

`ml/feature_schema.py`'s `CampaignDatasetRecord` (66 fields; 3
identifiers + 5 labels + 1 leakage column (`risk_score`) excluded by
`ml/dataset_utils.py`'s `FEATURE_COLUMNS` = the real **57** XGBoost
trains on — counted directly from the dataclass, not assumed) contains
these graph-derived fields, traced to their producing function:

| Existing signal | Current representation | Producing code | Used by XGBoost? | Structural information preserved? |
|---|---|---|---|---|
| `node_count`, `edge_count` | raw scalar counts | `graph_feature_engine.GraphAnalytics.node_count/edge_count` | Yes (`FEATURE_COLUMNS`) | Size only, no shape |
| `graph_density`, `graph_connectivity`, `average_degree` | scalar | same, `nx.density`/`nx.connected_components`/degree mean | Yes | Coarse global shape, no local structure |
| `attacker_degree`, `victim_degree`, `technique_degree` | scalar | same, per-label-type degree | Yes | Per-role connectivity, not per-node identity |
| `attack_chain_depth`, `average_path_length`, `graph_diameter`, `branching_factor` | scalar | same | Yes | Chain/path shape, summarized to one number |
| `average_clustering`, `average_betweenness`, `average_closeness` | scalar (mean over all nodes) | same, `nx.clustering`/betweenness/closeness centrality | Yes | Real topology metrics, but **averaged away** — two structurally different graphs can produce the same mean |
| `community_count`, `largest_community` | scalar | same, `nx.community.greedy_modularity_communities` | Yes | Coarse partition summary only |
| `campaign_complexity`, `structural_risk`, `evolution_rate` | scalar, composite of the above | same, hand-written formulas | Yes | Derived from the same underlying stats, not new information |
| `graph_similarity` (a `CampaignFeatures` field, distinct dataclass from `GraphFeatures`) | scalar, `(graph_density + min(chain_depth/10,1) + min(complexity/100,1)) / 3` | `campaign_feature_engine.CampaignFeatureEngine.graph_similarity` | Yes | A hand-picked 3-feature composite, not the graph itself |
| Node identity / adjacency (who connects to whom, in what order) | *no field carries this* | — | **No** — none of the 57 columns encode which specific nodes exist or how they're wired beyond the scalars above | **Not preserved** — this is what a GNN's learned aggregation could see that hand-engineered scalars cannot |
| Relationship *type* structure (`HAS_EVENT` vs `MATCHES` vs `TARGETS` as distinct semantics) | not distinguished — `GraphFeatures` treats all edges as one undifferentiated graph for `nx` algorithms | — | No | Not preserved (this is exactly what the edge-type-encoding phase added to the *GNN's* encoder, not to XGBoost's) |

**Conclusion, directly answering the task's framing question**: XGBoost
already sees **19 real scalar summaries** of this subgraph's topology
(not zero — `GNN_FEASIBILITY.md`'s Section 5 slightly understates this
by calling it "hand-engineered summary statistics" without enumerating
them). What it provably does not see is **node-level identity and
exact wiring** — e.g., it cannot distinguish "one attacker with 5
techniques against one host" from "five different (attacker,
technique, host) triples with the same aggregate density," because
`average_degree`/`graph_density`/etc. collapse both to similar numbers.
A GNN's per-node embeddings, before pooling, are the one architectural
mechanism that could keep that distinction — *if* Option C's severity
target actually depends on it, which Section 5 questions.

---

## 4. Option C — Severity Classification: Deep Analysis

**Unchanged from `GNN_FEASIBILITY.md` Sections 8-11, re-affirmed, not
re-litigated** (per this document's own operating instruction not to
contradict prior findings without new evidence — there is none here):

- **Labels**: only `severity` (`Low`/`Medium`/`High`/`Critical`),
  `risk_scoring.risk_level_from_score(risk_scoring.normalize_risk_score(...))`
  — a deterministic threshold on `risk_score`, itself an unbounded
  running sum of `AttackEvent.tps`. Not independently verified ground
  truth (same caveat as XGBoost's identical label).
- **Class distribution** (reconfirmed live this phase via
  `build_real_campaign_dataset()`): `Low: 64, Medium: 3, Critical: 4`,
  **`High` still has zero real examples.**
- **Graph complexity** (reconfirmed, with new granularity this phase):
  node-count-per-subgraph distribution across the 71 real samples —
  `4→8, 5→15, 6→11, 7→4, 8→2, 10→3, 11→5, 12→2, 14→4, 17→2, 18→2, 19→1,
  20→5, 21→3, 22→2, 24→2`. The modal subgraphs (5 and 6 nodes, 26 of 71
  campaigns) are the near-star shapes `GNN_FEASIBILITY.md` already
  described; only 21 of 71 (30%) reach 10+ nodes.
- **Split strategy**: no strategy survives 64/3/4/0 — reconfirmed, not
  re-derived.

**New this phase (Section 8's finding, applied to Option C)**: even if
the label-imbalance problem were somehow resolved (e.g., after months
of further real-data accumulation), Option C's subgraphs currently
carry an **undocumented confound**: 42 of 71 (59%) samples include
1-hop `Unknown`-typed leaf nodes/edges (other campaigns, an operation,
occasionally a threat actor or a predicted-next technique) that are
*accidental artifacts of an untyped Cypher pattern*, not deliberate
"this campaign's severity depends on its neighborhood" modeling — and
the other 41% don't have them at all. A severity classifier trained on
this population would be learning from inconsistently-scoped inputs
without that inconsistency being a deliberate design choice. This is
an additional blocker layered on top of the label problem, not a
replacement for it.

**What would make Option C defensible**: (1) the label problem resolves
as CYUKTI accumulates more real, more severity-diverse campaigns — an
infrastructure/data-accumulation question, no code changes; (2) the
loader-scope confound (Section 8, 10) gets resolved one way or the
other — either deliberately keep the current per-campaign scope (and
fix the untyped-pattern bug so it's actually clean) or deliberately
adopt a wider, typed scope consistently. Neither is an engineering
change that changes *this* document's verdict on its own.

---

## 5. Option E — Representation Learning: Deep Analysis

The task's own framing rightly rejects "use embeddings" as an
objective. Four concrete formulations, evaluated against what CYUKTI's
real graph/label data can actually support:

### E1 — Graph autoencoding (embedding → reconstruct node features/adjacency)
- **Supervision required**: none — self-supervised, consistent with
  CYUKTI's existing `ml/ssl_pipeline.py`/`ml/ssft.py` precedent (no
  attack/severity label needed).
- **Real risk**: 26 of 71 (37%) subgraphs are 5-6-node near-stars
  (Section 4) — reconstructing a near-deterministic star shape is close
  to trivial and may not force the encoder to learn anything beyond
  node-type one-hots it's already given. The signal-richness question
  `GNN_FEASIBILITY.md` raised for Option C (limited multi-hop structure)
  applies here too, for a different reason: there isn't much to
  *reconstruct* that isn't already implied by node type.
- **Evaluation**: reconstruction loss is measurable without any label,
  but a *low* loss doesn't by itself demonstrate the embedding is
  *useful* for anything downstream — needs a second, separate
  evaluation (see E3's circularity problem below, which applies to
  validating any of E1/E2/E4's output too).

### E2 — Contrastive graph learning (positive/negative graph pairs → similar embeddings)
- **Supervision required**: positive/negative pairs. The only real
  candidates in the schema: `SIMILAR_TO` edges (122 real, Section 8) or
  same-`Operation` membership (`HAS_CAMPAIGN`, 35 real edges across 31
  of 71 campaigns) as positive pairs.
- **Real risk — the most important finding for Option E
  (Section 9 develops this fully)**: `SIMILAR_TO` is computed as `60%
  technique-Jaccard + 20% shared-attacker + 20% shared-host`
  (`neo4j_client.update_campaign_similarity`, verified by direct code
  read, Section 9). Training a GNN to reproduce `SIMILAR_TO` as its
  contrastive target means training it to reproduce a technique-overlap
  formula — not to find structure beyond it. This is not merely a risk
  to manage; as posed, **it is circular**, for the identical reason
  `risk_score` was excluded as a node feature (Section 10 of
  `GNN_FEASIBILITY.md`): the target and half the candidate supervision
  signal are the same underlying computation.
- **Class balance**: 122 positive edges out of `C(71,2) = 2485` possible
  pairs — a ~5% positive rate, a real but not unusual imbalance for
  contrastive setups (unlike Option C's problem, this one has standard
  mitigations — hard-negative mining, weighted loss).

### E3 — Cross-campaign similarity learning (metric-learning framing of E2)
- Same data source, same circularity risk as E2. Framing it as
  "learn a distance function" instead of "binary classify pairs" does
  not remove the circularity — the training signal is still
  `SIMILAR_TO`/`HAS_CAMPAIGN`.

### E4 — Unsupervised representation learning, evaluated qualitatively (no reconstruction or contrastive loss at all)
- **Supervision required**: none, but then there is also no
  quantitative training objective — this is closer to "run the encoder
  with fixed/lightly-trained weights and inspect the embeddings" than
  "train a GNN." Evaluation would be qualitative (nearest-neighbor case
  studies, visualization) unless anchored to an external criterion,
  which reintroduces E2/E3's circularity if that criterion is
  `SIMILAR_TO`/severity.

**What would make Option E defensible**: an independent ground truth
for "these two campaigns are truly related" that is **not** derived
from technique-ID overlap or attacker/host identity — e.g., analyst-
confirmed campaign linkage, or a real threat-intel-confirmed actor
attribution distinct from `RESEMBLES`'s own technique-overlap
computation. No such source currently exists in this repository's
schema (Section 3, 6). This is a data/process gap, not a code gap.

---

## 6. Comparison Table (Factual, No Winner Declared)

| Dimension | Option C (severity classification) | Option E (representation learning) |
|---|---|---|
| Scientific novelty | Low-to-moderate — targets a label XGBoost already predicts from a summary of the same subgraph (Section 3) | Moderate-to-high *if* a non-circular training/eval signal exists (Section 5); low if trained/evaluated against `SIMILAR_TO`/`RESEMBLES` |
| Architectural fit | Drop-in comparison point next to XGBoost's existing severity output | No existing consumer expects a "campaign embedding" as input today — would be new plumbing into `investigation/` or `threat_attribution_engine.py` |
| Data requirements | Per-campaign subgraph only (current `GraphSnapshotLoader` scope, once its untyped-pattern issue is resolved one way or another, Section 8) | Ideally the *cross*-campaign correlation graph (`SIMILAR_TO`/`HAS_CAMPAIGN`/`RESEMBLES`) as first-class, deliberate structure — currently only accidentally reachable (Section 8) |
| Label requirements | Yes — `severity`, heuristic proxy (Section 4) | E1/E4: none. E2/E3: yes, and the only real candidates are circular (Section 5, 9) |
| Dataset sufficiency (71 samples) | No — 3/4 classes, minority classes of 3-4 (`GNN_FEASIBILITY.md` Section 9, 11, reaffirmed) | Untested — no established minimum sample count for graph-level contrastive/autoencoding at this problem's scale exists in this repository; 71 is small by general ML convention regardless of objective |
| Class imbalance | Severe, unresolved (Section 4) | E2/E3: ~5% positive-pair rate (122/2485) — present but more standard; E1/E4: N/A (no classes) |
| Leakage risk | `risk_score`→node-feature leakage already fixed (`GNN_FEASIBILITY.md` Section 10); new confound from untyped loader patterns (Section 8, this doc) | Circularity between training/eval signal and `SIMILAR_TO`/`RESEMBLES`'s own formula (Section 5, 9) — a different *kind* of leakage, not yet fixed anywhere because the objective doesn't exist yet |
| Evaluation requirements | Accuracy/macro-F1 on a class-stratified split — currently impossible to construct meaningfully (Section 4) | Reconstruction loss (E1, measurable but not proof of usefulness) or pair-classification metrics (E2/E3, circular per above) or qualitative-only (E4) |
| Runtime usefulness | Would sit beside `runtime_predictor.py`'s existing XGBoost severity output — unclear what a second severity predictor changes for the SOC operator | Embedding could feed `threat_attribution_engine.py` (currently pure technique-set overlap, Section 3) or `investigation/` NBE evidence-selection as a new candidate signal — but only once/if it's shown to add non-circular information |
| Integration complexity | Low — `model.py`/`train_gnn.py` already implement this path | Higher — no training loop, loss function, or downstream consumer exists yet for any of E1-E4 |
| Reproducibility | Yes, mechanically (deterministic dataset extraction already proven, `GNN_FEASIBILITY.md` Section 6) | Yes for extraction; the objective itself (once chosen) would need its own new reproducibility proof |
| Research defensibility | Weak today — see Section 4's "what would make it defensible" | Weak today for E2/E3 (circularity); E1/E4 defensible only as an exploratory/descriptive contribution, not a "beats baseline X" claim, absent new ground truth |

---

## 7. Does a GNN Add Genuine Capability? — Direct Answers

**Option C vs. `graph features + XGBoost`**: XGBoost already consumes
19 real scalar summaries of the identical subgraph (Section 3). A GNN
here would be testing "does a learned aggregation beat hand-written
formulas on the *same* information," a real but narrow question — not
"can a GNN see something XGBoost cannot." `GNN_FEASIBILITY.md`'s
Section 5 already reached this conclusion; Section 3 above is the
missing enumeration that makes it checkable line-by-line rather than
asserted.

**Option E vs. `campaign correlation + operation similarity + TF-IDF
retrieval + Neo4j structural relationships`**: every one of those four
existing mechanisms (Section 1's summary, Sections traced in full
below) computes similarity from technique-ID sets/sequences plus
attacker/victim/temporal identity — **never from subgraph topology**.
`OperationFeatures.graph_similarity` is a literal `return 0.0`
(`operation_feature_engine.py`, verified by direct read) with a `0.00`
weight (`operation_decision_engine.py`'s `weights` dict) — the
architecture already reserved a slot for exactly this signal and never
filled it. **This is real, checkable evidence that a genuine
topology-aware similarity signal does not exist anywhere in production
today** — the strongest single piece of evidence in Option E's favor.
It is qualified, not negated, by Section 8's finding that raw
topological *adjacency* to this correlation structure already leaks
into the per-campaign subgraph incidentally (not the same as a
deliberate, evaluated embedding).

---

## 8. GraphSnapshotLoader Analysis (Includes a Correction to `GNN_FEASIBILITY.md`)

Traced directly from `graph_feature_engine.py`'s `GraphSnapshotLoader.load_campaign_graph`:

```cypher
MATCH (c:Campaign {campaign_id:$campaign_id})
OPTIONAL MATCH (a)-[r1]->(c)          -- UNTYPED: any node, any relationship, into c
OPTIONAL MATCH (c)-[r2]->(e:AttackEvent)
OPTIONAL MATCH (e)-[r3]->(t)
OPTIONAL MATCH (c)-[r4]->(h)          -- UNTYPED: any relationship, to any node, from c
```

1. **What relationships currently exist that touch a Campaign node**:
   `LAUNCHED` (in), `TARGETS` (out), `HAS_EVENT` (out) — the intended
   scope — plus `SIMILAR_TO` (in, from another Campaign), `HAS_CAMPAIGN`
   (in, from an Operation), `RESEMBLES` (out, to a ThreatActor),
   `LIKELY_NEXT` (out, to a Technique).
2. **Why are the latter four "currently excluded"?** They are **not**
   excluded — `GNN_FEASIBILITY.md` Section 2 stated they were, based on
   reading the query's evident *intent* (`(Attacker)->(Campaign)->
   (AttackEvent)->(Technique)`, `(Campaign)->(Host)`) rather than its
   literal Cypher, which does not filter `r1`'s or `r4`'s relationship
   type or the other endpoint's label. **Verified concretely**: calling
   the real, unmodified `GraphSnapshotLoader.load_campaign_graph` on
   real campaign `CAMP_7331E223` returns node labels
   `['Campaign','Campaign','Campaign','Campaign','Campaign','Operation',
   'Attacker','Campaign','AttackEvent','Technique','Host','AttackEvent']`
   and relationship types
   `['SIMILAR_TO','SIMILAR_TO','SIMILAR_TO','SIMILAR_TO','SIMILAR_TO',
   'HAS_CAMPAIGN','LAUNCHED','HAS_EVENT','MATCHES','TARGETS','HAS_EVENT']`
   — 5 `SIMILAR_TO` and 1 `HAS_CAMPAIGN` relationship, unrequested by
   the query's evident design but returned by its actual text. A direct
   Cypher count confirms this touches **42 of the 71 real campaigns
   (59%)** (`EXISTS` on incoming `SIMILAR_TO`/`HAS_CAMPAIGN` or outgoing
   `RESEMBLES`/`LIKELY_NEXT`).
3. **Which components consume `GraphSnapshotLoader`?** Exactly one
   instantiation site in the whole repository
   (`graph_feature_engine.py:133-134`, inside `GraphAnalytics.__init__`)
   — re-verified this phase by grep, not assumed. Its `graph_analytics`
   singleton is imported by: `campaign_feature_engine.py` (XGBoost's
   `graph_similarity` feature and, via `GraphFeatures`, 19 more of
   XGBoost's 57 columns), `ml/gnn/campaign_graphs.py` (this GNN work),
   `evidence/orchestrator.py`, `investigation/loop.py`,
   `feature_orchestrator.py`.
4. **Would adding them [deliberately] change existing XGBoost
   behavior?** They are **already** silently affecting it, for 59% of
   campaigns, every time `campaign_feature_engine.graph_similarity()`
   or any of the 19 raw `GraphFeatures` columns are computed
   live via `graph_analytics.refresh_campaign()`/`extract_features()`
   at prediction/investigation time (confirmed:
   `campaign_feature_engine.py`'s `graph_similarity` method calls
   `graph_analytics.refresh_campaign(context.campaign_id)` then
   `extract_features(...)`, not a cached/frozen snapshot). A
   *deliberate* scope change (e.g., explicitly and consistently
   including or excluding this structure, rather than the current
   untyped-pattern accident) would change these 19+1 feature values for
   the affected 42 campaigns going forward — a real train/serve
   consideration for any XGBoost artifact trained on historically
   accumulated rows (Section 10).
5. **Would it change feature dimensions?** Not for XGBoost's 57 columns
   (`GraphFeatures`'s fields are fixed scalars regardless of subgraph
   content) — only their *values* would shift for affected campaigns.
   For the GNN's `graph_encoder.py`, dimensions are already fixed by
   one-hot schemes (`NODE_TYPES`, `EDGE_TYPES`); an `Operation` or
   `ThreatActor` node currently falls into the `Unknown` node-type
   bucket and an untyped edge into the `Unknown` edge-type bucket
   (both exist post the edge-type-encoding phase) — present but
   currently indistinguishable from each other.
6. **Would it affect existing model artifacts?** The production
   `xgb_severity.json`/`.meta.joblib` artifact was trained on
   historically accumulated dataset rows (`ml/dataset_writer.py`) whose
   graph-feature columns already reflect this untyped-pattern behavior
   for whatever fraction of *those* historical campaigns were affected
   at accumulation time — this is not a new risk introduced by
   *changing* anything; it is a pre-existing, previously undocumented
   characteristic of the artifact's current training data.
7. **Would it break training/runtime parity?** Not today (nothing has
   been changed) — but *any* future change to this query's typing
   (deliberate widening, or a bug fix that removes the accidental
   capture) would, unless the artifact is retrained on the new feature
   distribution.
8. **Would Option C require [deliberately] resolving this?** Yes,
   eventually (Section 4) — training on inconsistently-scoped subgraphs
   without acknowledging why is a confound.
9. **Would Option E require [deliberately] resolving this?** Yes, more
   fundamentally — Option E's premise is that cross-campaign structure
   is worth modeling *as first-class, typed, intentionally-scoped
   structure* (e.g., recursively including a similar campaign's own
   attack chain, not just its bare node as an untyped leaf) — the
   current accidental 1-hop leak does not achieve that.
10. **Could the loader support optional graph scopes safely?** Yes in
    principle — see Section 9's L1/L2/L3 comparison, not implemented
    this phase.

---

## 9. Similarity Definition Analysis — What Does "Similar" Mean in CYUKTI Today?

Traced from real code, not assumed:

| Mechanism | File | Formula | Topology-aware? |
|---|---|---|---|
| `OperationDecisionEngine` (campaign→operation attach) | `operation_decision_engine.py`, `operation_feature_engine.py` | `0.20·attacker_id_match + 0.15·victim_id_match + 0.30·technique_Jaccard + 0.20·temporal_bucket + 0.10·chain_LCS + 0.05·prediction_match + 0.00·graph_similarity` (weights sum to 1.00; `graph_similarity` hard-coded `return 0.0`) | **No** — the one reserved slot for it is unimplemented |
| `ThreatAttributionEngine` (campaign→actor) | `threat_attribution_engine.py` | `0.50·technique_coverage + 0.20·technique_precision + 0.30·chain_LCS`, all technique-ID set/sequence operations | No |
| `SIMILAR_TO` writer (Neo4j-native, called every alert from `realtime_socgraph.py`) | `neo4j_client.update_campaign_similarity` | `60%·technique_Jaccard + 20%·shared_attacker(bool) + 20%·shared_host(bool)`, threshold ≥75 | No |
| `RESEMBLES` writer (Neo4j-native, same caller) | `neo4j_client.update_actor_attribution` | `technique_overlap_ratio·100 + shared_count·2` (capped 99), threshold ≥50 | No |
| Campaign-narrative retriever (Multi-RAG source 2) | `rag/campaign_retriever.py` | TF-IDF over `"attacker {X} targeting {Y}. Techniques: {list}"` — bag-of-words on formatted identity/technique text | No |

**Every mechanism reduces a campaign to a set or sequence of technique
IDs (plus attacker/victim/time identity) before comparing it to
another.** None consult node degree, path structure, branching, or any
of the 19 `GraphFeatures` scalars XGBoost already has, let alone
per-node embeddings.

**Does GNN similarity add information beyond the existing weighted
engine?** In principle, yes — it is the only mechanism that could
reason over *how* nodes connect, not just *which* technique IDs
appear. **In practice, evaluating it honestly requires ground truth
CYUKTI does not have**: the only labeled "these are similar" signal in
the schema (`SIMILAR_TO`) is itself 60% the same technique-Jaccard a
GNN would need to be shown to beat. Training or validating a GNN
similarity objective against `SIMILAR_TO` would be measuring the GNN
against a close cousin of itself, not an independent standard — this
is the single most load-bearing finding for Option E's defensibility
and is why Section 5 could not simply recommend "do E2."

---

## 10. Loader Design Alternatives (Not Implemented)

| Option | Description | Compatibility | Maintainability | Risk to XGBoost | Code duplication | Reproducibility | Architecture clarity |
|---|---|---|---|---|---|---|---|
| **L1** — change `GraphSnapshotLoader` globally (e.g., type the `r1`/`r4` patterns, or deliberately widen them) | One shared query, new semantics for every caller | Breaks silently for any caller relying on current (even if accidental) behavior | Single source of truth, easy to reason about *once* changed | **Highest** — `campaign_feature_engine.py`'s live `graph_similarity`/`GraphFeatures` recomputation (Section 8, item 4) would shift for the 42 affected campaigns without retraining the deployed XGBoost artifact | None | High (single path) | Clear once done, but silently changes a live production dependency outside this phase's scope |
| **L2** — add an optional relationship-type scope parameter (e.g. `load_campaign_graph(campaign_id, relationship_types=None)`), defaulting to *current* (untyped) behavior | XGBoost's call sites unaffected by default; GNN work opts into an explicit, typed scope | Two behaviors to reason about, but the default preserves existing behavior | Moderate — one function, one new parameter | **Low** — nothing changes unless a caller opts in | None (one function, branching logic) | High | Slightly less clear (one function serving two scopes) but explicit about the choice |
| **L3** — separate GNN-specific loader (new query, reuses existing `GraphBuilder` unchanged since it only turns a `GraphSnapshot` into `nx.DiGraph` regardless of how the snapshot was queried) | Fully decoupled from XGBoost's path | Two loaders to maintain, but each single-purpose | **None** — production path untouched | Low (a new Cypher query text, `GraphBuilder` already generic) | Highest (separate query, though not separate graph-building logic) | Clearest separation of "what XGBoost sees" vs "what a GNN sees" |

All three are evaluated factually here; none is implemented, per this
phase's scope.

---

## 11. Engineering Requirements

- **For Option C**: resolve the loader-scope confound (Section 8, 10 —
  a design decision, not pure engineering, since it decides what a
  "campaign graph" means); everything else (extraction, encoding, model,
  training loop) already exists and is tested (`GNN_FEASIBILITY.md`
  Section 1, 6-7).
- **For Option E**: a training loop and loss function for whichever of
  E1-E4 is chosen do not exist yet; a downstream consumer (e.g.
  `threat_attribution_engine.py` accepting an embedding-similarity
  input alongside its existing technique-overlap score) does not exist
  yet; the loader-scope question (Section 8, 10) is more central here
  since Option E's whole premise depends on what "the graph" includes.

## 12. Research Requirements

- Which objective (C vs E, and if E, which of E1-E4) — this document's
  central open question.
- What independent, non-circular ground truth (if any) could validate
  Option E's output — currently absent from the schema (Section 5, 9).
- Whether Option C is worth pursuing at all once dataset sufficiency
  improves, given Section 3's redundancy finding, or whether that
  effort is better spent elsewhere.
- Whether `GraphSnapshotLoader`'s scope should be fixed (typed) or
  deliberately widened, and for whom (Section 8, 10).

## 13. Infrastructure Requirements

- More real, more severity-diverse campaigns for Option C
  (`GNN_FEASIBILITY.md` Section 13 — a data-accumulation problem, not a
  missing system).
- An independent similarity/attribution ground-truth source for Option
  E's evaluation (Section 5, 9) — does not exist today in any form
  (live telemetry or otherwise).

---

## 14. Decision Criteria (Framework Only, No Score, No Winner)

1. Does it solve a real CYUKTI problem? (Section 7)
2. Does it add capability not already present? (Sections 3, 6, 7, 9)
3. Does current data support it? (Sections 4, 5)
4. Can it be evaluated honestly? (Section 5's circularity finding is
   the sharpest test of this for Option E; Section 4's split-infeasibility
   is the sharpest test for Option C)
5. Can it integrate into the existing architecture? (Section 11)
6. Does it contribute meaningfully to the capstone?
7. Can it produce a defensible research contribution? (Sections 4, 5)
8. Does it avoid unnecessary duplication of XGBoost? (Section 3, 7)
9. Does it preserve the Detection Paradox research direction? (Neither
   option was found to touch Phase 21/22's frozen findings; not
   evaluated further here as out of this document's scope)
10. Can the work realistically be completed with available
    dataset/infrastructure? (Sections 4, 5, 13)

## 15. Open Questions

1. Is the untyped `(a)-[r1]->(c)` / `(c)-[r4]->(h)` pattern in
   `GraphSnapshotLoader` a bug to fix (tighten to the originally
   intended `LAUNCHED`/`HAS_EVENT`/`MATCHES`/`TARGETS` scope) or an
   accidental head start on Option E's "wider graph" idea to deliberately
   keep and formalize? This document takes no position — Section 8, 10.
2. If Option E is chosen, is a circular evaluation against
   `SIMILAR_TO`/`RESEMBLES` acceptable as a first descriptive pass (with
   that circularity disclosed), or does the project require holding out
   for independent ground truth before publishing any result?
3. Is a second, redundant severity signal (Option C) valuable to the
   capstone narrative even if it doesn't beat XGBoost — e.g., as a
   methodological comparison — or only if it demonstrably adds
   information?
4. Does the project want to invest in L2/L3 (Section 10) regardless of
   which objective is chosen, simply to stop the current accidental
   inconsistency (Section 8) from silently affecting XGBoost's live
   features?

## 16. Recommended Decision Process

Not a decision — a suggested sequence for the project owner to reach
one:

1. Read Section 8 first — it changes the premise both options were
   evaluated under in `GNN_FEASIBILITY.md`.
2. Decide Open Question 1 (loader scope) independently of C vs E, since
   it affects both.
3. If leaning toward Option E, treat Open Question 2 (circularity
   disclosure vs. holding out) as the actual decision — it determines
   whether E2/E3 are usable at all versus only E1/E4 in a purely
   descriptive mode.
4. If leaning toward Option C, treat Section 4's "what would make it
   defensible" as a go/no-go gate tied to real data accumulation, not a
   engineering task to schedule now.
5. Only after 2-4: scope the actual engineering (Section 11) for
   whichever path is chosen.

---

## 17. Testing

```bash
cd backend
python -m pytest tests/ -q
```

**376 passed** — unchanged from the end of the edge-type-encoding
phase. This phase added no test files and changed no production code;
all analysis in this document was produced by throwaway, uncommitted
scripts that called only existing, already-tested functions, deleted
after use (`git status` confirms a clean working tree for `backend/`
at the end of this phase, aside from this document itself and its
sibling addendum in `GNN_FEASIBILITY.md`).
