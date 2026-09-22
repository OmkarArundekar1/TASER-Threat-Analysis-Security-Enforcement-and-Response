# GNN Representation Design — Option E Formalization

Companion to `GNN_FEASIBILITY.md` and `GNN_OBJECTIVE_DECISION.md`. The
project owner has selected **Option E — cross-campaign graph
representation learning**. This document converts that selection into
a precise representation, an enumerated set of candidate self-
supervised objectives (each classified, none silently chosen), an
isolated analysis of the `GraphSnapshotLoader` finding, and a future
runtime interface — **without training a model, fabricating a label,
or changing any production behavior.**

All numbers were re-queried live against this environment's real Neo4j
instance this phase, via throwaway, uncommitted scripts that called
only existing, already-tested production functions
(`ml.gnn.campaign_graphs.build_real_campaign_dataset`, direct read-only
Cypher), then deleted (`git status` at the end of this phase shows no
production file touched — verified below). Backend: **376 passed**,
unchanged.

---

## 1. Selected Direction

Cross-campaign graph representation learning: a GNN encoder produces a
fixed-size embedding per campaign (or, where the data supports it,
per operation) that a downstream consumer can compare, retrieve, or
combine with other signals — **not** a severity classifier. The
representation, not a classification head, is the deliverable.

---

## 2. Representation Definition

Formally: `G = (V, E, X, A)` where `V` is the node set, `E` the edge
set, `X ∈ R^(|V| × F_node)` the node-feature matrix, `A` the (typed)
adjacency. A GNN encoder `f_θ` produces `z_G = f_θ(G) ∈ R^d`.

**What already exists, concretely (traced from `ml/gnn/graph_encoder.py`,
current code)**:
- `X`: one-hot node type (`Attacker, Campaign, AttackEvent, Technique,
  Host, Unknown` — `NODE_TYPES`) concatenated with 10 curated numeric
  properties (`occurrences, total_tps, tps, rule_level, vt_reputation,
  threat_actor_reputation, malware_confidence, tool_confidence,
  misp_confidence, ioc_confidence`), `risk_score` deliberately excluded
  as a leakage source. `F_node = 16`.
- `A`/edge typing: one-hot relationship type (`LAUNCHED, HAS_EVENT,
  MATCHES, TARGETS, Unknown` — `EDGE_TYPES`, added the phase before
  this one), symmetrized for message passing, not yet consumed by
  `SAGEConvLayer`.
- Graph-level features: not part of the GNN's input today — these are
  `graph_feature_engine.GraphFeatures`'s 19 scalars, computed
  separately and already fed to XGBoost (Section 3).
- Temporal information: **not encoded anywhere in the GNN path today.**
  `campaign_graphs.py` extracts one static snapshot per campaign at
  call time (`force_reload` default `False`). Section 6C and Section 8
  analyze whether/how to add this.
- Campaign metadata used as features: `occurrences`, `total_tps`,
  `risk_score` (excluded, Section 10 of `GNN_FEASIBILITY.md`) — no
  campaign-level scalar feature vector is concatenated onto the graph
  embedding today; the embedding is derived purely from node/edge
  content and structure.
- Operation metadata: **not part of the per-campaign extraction at
  all** except incidentally, per Section 10 below.

**What `z_G` should preserve** (per Section 7 of `GNN_OBJECTIVE_DECISION.md`,
restated as a design target): node-level identity and exact wiring —
*which* attacker, technique, and host co-occur and *how* they connect —
information the 19 `GraphFeatures` scalars provably collapse into
aggregate statistics (Section 3).

**What `z_G` should deliberately not memorize**: `risk_score` (already
excluded, same leakage precedent as XGBoost's `LEAKAGE_COLUMNS`); raw
node identifiers (`campaign_id`, IP strings) — the encoder never sees
these as features (`node_ids` is bookkeeping only, not part of `X`);
and, per this phase's central constraint, **the `SIMILAR_TO`/
`RESEMBLES` derived scores themselves**, if they end up incidentally
present as `Unknown`-typed neighbor nodes (Section 10) — an embedding
that trivially reconstructs `SIMILAR_TO`'s own technique-Jaccard
formula from a leaked neighbor has not learned anything the existing
Cypher writer doesn't already compute in one line.

**Campaign-level or operation-level?** Campaign-level is what the
current extraction produces and is the only level with a real per-
node feature/label pipeline today (`campaign_graphs.py`). Operation-
level embeddings (pooling several campaigns' embeddings, or encoding
an `Operation`+its `HAS_CAMPAIGN` neighborhood directly) are not
extractable today without deliberately extending scope (Section 10) —
recorded as an open question (Section 13), not decided here.

**Are temporal snapshots needed?** Only if Objective C (temporal
representation learning, Section 6) is chosen — and only for the 21 of
71 real campaigns with 2+ real `AttackEvent`s (Section 8 quantifies
this precisely). Not needed for Objective A (autoencoding) at all.

**Embedding dimension `d`**: not fixed here — `CampaignGNN`'s existing
`hidden_dim` default (32, `ml/gnn/model.py`) is a synthetic-data-era
default, not evidence-justified for real data. Given the real dataset
is 71 samples with 34 distinct node-type-multiset shapes (Section 8), a
dimension anywhere near or above the sample count risks trivial
memorization regardless of objective; this document does not pick a
number, since the right value depends on which objective (Section 6)
is approved and how it's regularized — recorded as an open question
(Section 13).

---

## 3. Current Graph Schema — Signals, Traced From Code

Unchanged from `GNN_OBJECTIVE_DECISION.md` Section 3's table (XGBoost's
19 `GraphFeatures` scalars + 1 composite, node identity/adjacency not
preserved by any of them) — not re-derived here. One addition found
this phase, material to Section 4:

`CampaignFeatureEngine.graph_similarity` (`campaign_feature_engine.py`)
is a **second**, distinct, already-real (not a stub) composite:
`(graph_density + min(attack_chain_depth/10, 1) + min(campaign_complexity/100, 1)) / 3`
— built from 3 of the same 19 `GraphFeatures` scalars, so still no new
information relative to Section 3's table, but functionally different
from `OperationFeatures.graph_similarity` (Section 5): it is real,
computed, and weighted `0.01` in `config.CAMPAIGN_WEIGHTS`, feeding
`CampaignDecisionEngine` — not zero, not a stub, just very low-weight
and still just a hand-engineered scalar, not a learned representation.

---

## 4. Existing Similarity Mechanisms

| Mechanism | Input | Representation | Similarity basis | Topology-aware? | Circular risk (as a training/eval target for Option E) |
|---|---|---|---|---|---|
| `OperationDecisionEngine` (campaign→operation attach) | `OperationFeatures` (7 scalars) | Weighted sum, threshold 0.70 | `0.30·technique_Jaccard + 0.20·attacker_id_match + 0.20·temporal_bucket + 0.15·victim_id_match + 0.10·chain_LCS + 0.05·prediction_match + 0.00·graph_similarity` (stub) | No (reserved slot unfilled) | High if used directly (`HAS_CAMPAIGN` membership derived from this) — Section 7 |
| `CampaignDecisionEngine` (active-campaign continue/close) | `CampaignFeatures` (7 scalars) | Weighted sum ÷ available weight | `0.35·prediction + 0.30·chain_LCS + 0.20·temporal + 0.10·attacker_id + 0.03·runtime + 0.01·graph_similarity (real, Section 3) + 0.01·duplicate` | Marginally (1% weight on a real hand-engineered composite) | Low as a target (this decision isn't persisted as a graph relationship at all — it only closes/continues a campaign, it doesn't label a *pair* of campaigns as similar) but **not usable as positive/negative pair supervision** either, for the same reason |
| `ThreatAttributionEngine` (campaign→actor) | Technique-ID sets + chain | `0.50·coverage + 0.20·precision + 0.30·chain_LCS`, all set/sequence ops | No | High if used directly (`RESEMBLES` derived from a closely related formula) |
| `SIMILAR_TO` writer (Neo4j-native, every alert) | Technique-ID sets + attacker/host identity | `60%·technique_Jaccard + 20%·shared_attacker + 20%·shared_host`, threshold ≥75 | No | **Highest** — the only edge-labeled "these campaigns are similar" signal in the schema, 60% technique-overlap by construction |
| `RESEMBLES` writer (Neo4j-native, every alert) | Technique-ID sets | `technique_overlap_ratio·100 + shared_count·2`, threshold ≥50 | No | High, same reason as `SIMILAR_TO` |
| Campaign-narrative TF-IDF (Multi-RAG source 2) | Formatted attacker/victim/technique text | Bag-of-words | No | Low as direct supervision (it's a retrieval index, not a stored pairwise label) but shares the same underlying technique-ID vocabulary, so not independent either |

**Capability gap, stated precisely**: every mechanism above that
produces a *storable, reusable* similarity judgment (`SIMILAR_TO`,
`RESEMBLES`, and by extension `HAS_CAMPAIGN` membership via
`OperationDecisionEngine`) is dominated by technique-ID set overlap.
None encode *how* a campaign's nodes are wired — degree, branching,
multi-hop reachability. A GNN embedding is the only mechanism
architecturally capable of that. **The gap is real; the available
ground truth to prove it's being filled is not** (Section 6-7).

---

## 5. `OperationFeatures.graph_similarity` — Verified Architectural Role

- **Why it currently returns 0.0**: `operation_feature_engine.py`'s
  `extract_graph_similarity(self, campaign_context, operation_context):
  return 0.0` — a literal, unconditional stub, not a bug in a
  computation (there is no computation).
- **Explicitly reserved?** Yes — it is a named field in the
  `OperationFeatures` dataclass, has a named (if zero) weight in
  `OperationDecisionEngine.weights`, and appears in the `breakdown`
  dict returned to callers — the architecture carved out a slot and
  never filled it, rather than the field being an afterthought.
- **Where its output would flow, if populated**: into
  `OperationDecisionEngine.evaluate()`'s weighted `score` (only once
  its weight is *also* changed from `0.00` — populating the field
  alone does nothing, since `value × 0.00 = 0` regardless of value) →
  `decision` (`ATTACH_TO_OPERATION` at `score ≥ 0.70`) →
  `CorrelationResult.matched`/`.score`/`.breakdown`, returned to
  `realtime_socgraph.py`'s `operation_engine.correlate(context)` call.
  Traced this phase: only `.matched`, `.score`, `.confidence`, and
  `.candidate_count` are actually consumed there today (feeding
  `cti_engine.calculate(campaign_confidence=result.confidence, ...)`);
  `.breakdown` itself (where `graph_similarity`'s value would be
  visible) is not read downstream in production — it exists for
  diagnostics/tests (`tests/test_campaign_correlation.py` asserts
  against it directly).
- **A second, sibling slot exists** (found this phase, Section 3):
  `CampaignFeatures.graph_similarity` in `campaign_decision_engine.py`
  is real (not a stub) and already flows into
  `campaign_manager.py`'s `decision.score >= ACTIVE_CONTINUE_THRESHOLD`
  (0.35) check — a live, production, active-campaign continue/close
  decision. It is a different field, a different engine, and a
  different decision than `OperationFeatures.graph_similarity`; a
  future embedding-based signal would have **two** distinct,
  independently-weighted places it could enter, not one.
- **Could a future GNN embedding legitimately populate the
  `OperationFeatures` slot?** In principle yes — that is exactly the
  kind of signal the slot's name and reserved weight anticipate. It
  would need to be reduced to a scalar (e.g. cosine similarity between
  two campaigns' embeddings) to fit the existing weighted-sum
  architecture, which is a real design compromise (a scalar collapses
  the embedding back down, discarding exactly the per-node information
  Section 2 says the embedding should preserve) — noted, not resolved.
- **Would populating it change operation-correlation behavior?** Yes,
  necessarily, the moment its weight is changed from `0.00` — this is
  precisely why it is **not done in this phase** (Section 13's
  Operating Principle: no correlation-behavior change).

---

## 6. Candidate Learning Objectives — Classified, Not Chosen

### A — Graph autoencoding (embedding → reconstruct node features/structure)
- **Supervision**: none — self-supervised, consistent with CYUKTI's
  existing `ml/ssl_pipeline.py`/`ml/ssft.py` precedent.
- **Real risk, quantified this phase**: 34 distinct node-type-multiset
  "shapes" exist across the 71 real campaigns, but the single most
  common shape (the intended 5-node star — one each of `Attacker`,
  `Campaign`, `AttackEvent`, `Host`, `Technique`) accounts for 15 of 71
  (21%) alone, and the top 3 shapes account for 30 of 71 (42%).
  Reconstructing a shape this repetitive risks the encoder learning
  "which of ~5 common templates is this" rather than fine-grained
  structure.
- **Classification: feasible now**, with the above caveat disclosed
  as a limitation, not hidden.

### B — Self-supervised contrastive learning via graph augmentation
- **Supervision**: augmented views of the same graph as positive
  pairs. Candidate augmentations and their security-semantic validity,
  checked against the real schema:
  - *Node masking* (drop a node): on the 21% of graphs that are a
    5-node star, masking any one of the 5 nodes removes a *distinct*
    semantic role (there is no redundant node to drop) — not
    meaning-preserving at this graph size. Only defensible on the
    minority of larger graphs (21 of 71 have 4+ nodes beyond the
    minimum, Section 8).
  - *Feature masking* (zero out a numeric property): defensible in
    principle (analogous to standard graph-SSL practice) but has not
    been checked against which of the 10 numeric properties, if any,
    the small real graphs depend on to remain distinguishable from
    each other — not evaluated this phase.
  - *Edge masking*: removing `LAUNCHED`, `HAS_EVENT`, `MATCHES`, or
    `TARGETS` from a 4-5 node star can disconnect the graph entirely
    (e.g. removing the one `HAS_EVENT` edge severs `Technique` from
    everything else) — not meaning-preserving at this size either.
  - *Subgraph sampling*: undefined on graphs this small — there is
    often no proper subgraph left that isn't trivial.
- **Classification: research-risky.** The standard graph-contrastive
  augmentation toolbox assumes graphs large enough that dropping a
  node/edge still leaves a meaningful structure; the real dataset's
  modal shape (Section 8) does not meet that assumption. Usable, if at
  all, only on the minority of larger real graphs — and even then, no
  augmentation was validated as security-semantics-preserving this
  phase (per this phase's explicit instruction not to implement
  arbitrary augmentations).

### C — Temporal representation learning
- **Investigated concretely, using only already-stored real
  timestamps — no fabrication, no waiting in real time.** Each real
  `AttackEvent` carries a real `first_seen`. For a campaign with `k`
  real events, the events can be ordered by timestamp and the graph
  reconstructed incrementally (as of event 1, as of events 1-2, ...,
  as of events 1-k) entirely from data already in Neo4j. Consecutive
  snapshots of the *same* campaign are a positive pair (same real
  entity, not a derived similarity score); snapshots from *different*
  campaigns are negatives.
- **This does not use `SIMILAR_TO` at all** — "same campaign, different
  point in its own real history" is an identity fact (the
  `Campaign.campaign_id` is literally the same), not a heuristic
  judgment. This is the objective with the cleanest non-circularity
  argument of the four.
- **Quantified this phase**: 21 of 71 real campaigns (30%) have ≥2 real
  `AttackEvent`s with genuinely distinct `first_seen` timestamps
  (verified: all 21 have real chronological ordering, not simultaneous
  events) — these can support a 2-point-minimum temporal pair. Only 9
  of 71 (13%) have ≥4 events for a richer multi-point sequence.
- **Classification: feasible with additional data** for a robust
  version (9 campaigns is a thin basis for anything beyond a proof of
  concept); a **minimal 2-snapshot proof of concept is feasible now**
  on the 21-campaign subset, disclosed as a small-sample pilot, not a
  general result.

### D — Downstream task evaluation as the training/selection signal
- **Investigated**: could embeddings improve `OperationDecisionEngine`'s
  attach decision, measured against an independently defined evaluation
  set? Traced this phase: no independent, analyst-confirmed
  "campaign X should/shouldn't have attached to operation Y" label
  exists anywhere in this schema — the only thing "improve" could be
  measured against is the same weighted-formula threshold (0.70) that
  produced `HAS_CAMPAIGN` in the first place (Section 4).
- **Classification: insufficiently measurable today** — not circular in
  the same direct sense as training against `SIMILAR_TO` (D doesn't
  propose using it as a *training* target at all, only as a downstream
  *consumer* to evaluate against), but there is no independent
  "improved at what, exactly" criterion available. Becomes feasible
  the moment any independent ground truth (analyst review, confirmed
  intel) exists — an infrastructure/process gap, not an engineering one.

### Explicit elimination
**Using `SIMILAR_TO` or `RESEMBLES` directly as a contrastive positive-
pair label (a literal reading of Objective B/C using those edges
instead of augmentations/timestamps) is eliminated as circular**, per
this phase's explicit constraint and `GNN_OBJECTIVE_DECISION.md`
Section 5/9's finding: `SIMILAR_TO` is 60% the same technique-Jaccard
computation a GNN would need to be shown to exceed.

**Summary classification**:

| Objective | Classification |
|---|---|
| A — Autoencoding | Feasible now (with the shape-repetition caveat disclosed) |
| B — Contrastive w/ augmentation | Research-risky (augmentation validity unresolved at this graph size) |
| C — Temporal (own-history snapshots) | Feasible now as a small pilot (21 campaigns); feasible with additional data for a robust version (only 9 have 4+ events) |
| D — Downstream-task evaluation | Insufficiently measurable today (no independent ground truth) |
| Contrastive against `SIMILAR_TO`/`RESEMBLES` directly | **Circular — eliminated** |

No objective is selected here. Section 6 of the final report below
distinguishes a technical recommendation from the project owner's
actual decision, per this phase's instructions.

---

## 7. Circularity Analysis — `SIMILAR_TO` in Full

Restated with one addition: `SIMILAR_TO`'s formula
(`neo4j_client.update_campaign_similarity`, verified by direct code
read) is `60% technique_Jaccard + 20% shared_attacker(bool) + 20%
shared_host(bool)`, threshold ≥75. `HAS_CAMPAIGN` (via
`OperationDecisionEngine`, Section 4) is less purely technique-based
(30% technique, 45% attacker+victim combined, 20% temporal, 10% chain,
5% prediction, 0% graph) but still not independent of the same
underlying identity/technique signals. **Neither is usable as a
training or evaluation ground truth for "the GNN learned meaningful
structural similarity"** without disclosing that the ground truth
itself is majority-derived from the same category of signal (technique
overlap, or attacker/victim identity) the GNN is meant to go beyond.

---

## 8. Dataset Requirements — Quantified From Live Data

- 71 real campaign samples (unchanged).
- **34 distinct node-type-multiset shapes** — not 71 distinct graphs in
  any meaningful structural sense; several shapes repeat (top shape:
  15 of 71, second: 8 of 71, third: 7 of 71).
- **Direct evidence of the `GraphSnapshotLoader` leak's severity**: one
  real shape (2 campaigns) shows `Campaign: 15` — i.e. at least one
  real campaign's "per-campaign" subgraph contains **15 other
  Campaign-labeled nodes it never asked for**, purely from the untyped
  `(a)-[r1]->(c)` pattern capturing `SIMILAR_TO` fan-in (Section 10).
  This is not a rare, single-digit anomaly — it materially changes what
  "the campaign's graph" contains for the affected samples.
- Node-count-per-subgraph distribution: unchanged from
  `GNN_OBJECTIVE_DECISION.md` Section 4 (`4→8, 5→15, 6→11, ...,
  24→2`).
- Temporal diversity: 21 of 71 (30%) have ≥2 real, distinctly-timestamped
  events; 9 of 71 (13%) have ≥4.
- Attacker/host reuse (`GNN_FEASIBILITY.md`/`GNN_OBJECTIVE_DECISION.md`,
  reaffirmed, not re-queried this phase since no new evidence
  contradicts it): 4 of 7 real attacker IPs and 3 of 9 real host IPs
  are shared by more than one campaign.
- **Conclusion**: Objective A can proceed on all 71 (with the
  shape-repetition caveat above disclosed); Objective C's honest scope
  is 21 campaigns for a pilot, 9 for anything more; Objective B is not
  currently supportable without further augmentation-validity work;
  Objective D needs new, non-existent ground truth. **No objective
  needs to wait for a larger raw campaign count per se — the
  limitations are about structural diversity, event-count depth, and
  ground-truth availability, not sample count alone.**

---

## 9. Evaluation Requirements — Without Circularity

Two concrete strategies this repository can actually support today,
neither dependent on `SIMILAR_TO`/`RESEMBLES`/`HAS_CAMPAIGN`:

1. **Raw-identity retrieval sanity check**: build a relevance set
   directly from graph facts, not derived scores — "campaign X's
   nearest neighbor by embedding should, at above-chance rate, share
   the *same* `Attacker` node or the *same* `Host` node" (both are raw
   `LAUNCHED`/`TARGETS` edges, objectively true or false, not a
   weighted judgment). This is a **low bar** (an embedding that just
   encodes attacker/host node one-hots would pass trivially) but is
   fully non-circular and cheap to run as a first sanity check before
   trusting anything else.
2. **Temporal self-consistency** (ties to Objective C): for the 21
   campaigns with 2+ timestamped events, check whether embeddings of
   *earlier* snapshots of a campaign are closer to its own *later*
   snapshots than to any other campaign's snapshots — a same-entity-
   over-time criterion, not a similarity judgment borrowed from
   `SIMILAR_TO`.

Not currently supportable: retrieval against an analyst-defined
relevance set (no such set exists in this repository or process today
— an infrastructure gap, not something this phase can construct
without fabricating one).

---

## 10. `GraphSnapshotLoader` Findings — Isolated, Not Fixed

Restating `GNN_OBJECTIVE_DECISION.md` Section 8's finding with the
task's specific sub-questions answered directly:

1. **Intended relationships**: `LAUNCHED` (in), `HAS_EVENT` (out),
   `MATCHES` (out, via `AttackEvent`), `TARGETS` (out) — evident from
   the query's node-label-typed hops (`c:Campaign`, `e:AttackEvent`).
2. **Actually retrieved**: the above, **plus** `SIMILAR_TO` (in),
   `HAS_CAMPAIGN` (in), `RESEMBLES` (out), `LIKELY_NEXT` (out) whenever
   they exist for that campaign, because `(a)-[r1]->(c)` and
   `(c)-[r4]->(h)` carry no type/label filter. Verified concretely
   (real campaign `CAMP_7331E223`: 5 `SIMILAR_TO` + 1 `HAS_CAMPAIGN`
   returned by the unmodified loader) and quantified (42 of 71
   campaigns, 59%; at least one campaign carries 15 extra `Campaign`
   nodes this way, Section 8).
3. **XGBoost features affected**: `campaign_feature_engine.py`'s
   `graph_similarity` and, via `graph_analytics.extract_features`, all
   19 `GraphFeatures` scalars (Section 3) — every one of them is
   computed over whatever this query returns, for every campaign, live.
4. **Does current (persisted) training data include this?** Yes —
   the deployed model's training data (`ml/datasets/campaign_dataset.csv`,
   60 rows, matching the previously-reported "60 campaigns" baseline)
   was accumulated via this same, single, unchanged code path
   (`git log --follow` on `graph_feature_engine.py` shows exactly one
   commit in this repository's history, the initial CYUKTI import — no
   evidence of a mid-history change). **This means the behavior is
   consistent across all of training and serving, not skewed between
   them** — a more benign framing than "a change would cause skew";
   the real characteristic is an *unacknowledged, undocumented, uniform
   confound*, not a train/serve mismatch, unless someone changes the
   query later without retraining.
5. **Runtime inference**: yes — `campaign_feature_engine.graph_similarity()`
   calls `graph_analytics.refresh_campaign()` (bypassing the cache) then
   `extract_features()` live, at investigation/prediction time, so
   currently-served predictions are computed under the identical
   behavior as training data.
6. **Would changing the query change feature values?** Yes, for the 42
   affected campaigns' worth of future computations — not retroactively
   for already-persisted rows.
7. **Does the existing model artifact depend on this behavior?**
   Implicitly yes, in the sense that its training data was shaped by
   it — but since the behavior has been constant throughout this
   repository's tracked history, the artifact is not *more* wrong today
   than it always was; it simply was never trained on the "clean"
   4-relationship-type graph `GNN_FEASIBILITY.md` assumed existed.
8. **Documented anywhere before this?** No — `GNN_FEASIBILITY.md`
   Section 2 (prior phase) stated the opposite; corrected in place with
   a strikethrough and pointer to `GNN_OBJECTIVE_DECISION.md` Section 8
   during that phase.
9. **Accidental leak or intentional wider-graph design?** No evidence
   either way was found in code, comments, commit history (a single
   squashed initial-import commit, no incremental history to inspect),
   or any existing documentation. **This cannot be called a bug without
   evidence of original intent — there is none, in either direction —
   so this document calls it an unautheticated/unverified behavior, not
   a "bug," and leaves the fix-vs-formalize decision to the project
   owner (Section 13).**

---

## 11. XGBoost Compatibility — Explicit Guarantee Check

Per this phase's hard constraint: **`GraphSnapshotLoader` was not
modified.** Verified: `git status` at the end of this phase shows only
this document, `GNN_REPRESENTATION_DESIGN.md`, and (if approved for
commit) `ARCHITECTURE_AUDIT.md`'s factual update — no `.py` file
touched. Therefore, trivially:

- Training behavior: unchanged (nothing executed against the persisted
  dataset).
- Runtime behavior: unchanged (the same query still runs).
- Feature dimensions: unchanged (`FEATURE_COLUMNS` untouched).
- Feature semantics: unchanged (no redefinition attempted).
- Existing XGBoost artifact validity: unchanged (not retrained, not
  touched).

All five guarantees hold by construction, not by proof-after-the-fact
— this phase made no attempt to change the loader, precisely because
Section 8's finding could not be resolved with full confidence about
original intent (item 9 above).

---

## 12. GNN-Specific Graph Scope — Design Alternatives (Not Implemented)

Mapped onto this task's A-D framing (equivalent to
`GNN_OBJECTIVE_DECISION.md` Section 10's L1-L3, extended with a 4th,
fully-independent option):

| Option | Description | Compatibility | Code duplication | Maintainability | Model-artifact stability | Runtime consistency |
|---|---|---|---|---|---|---|
| **A** — type the existing loader's `r1`/`r4` patterns (fixes the leak globally) | One shared, now-typed query | Breaks any caller (XGBoost) relying on current behavior, even if accidental | None | High once done | **At risk** — future feature values shift for 42 campaigns unless retrained | Changes runtime feature computation the moment it ships |
| **B** — optional relationship-scope parameter, default = current behavior | Same function, opt-in typed/widened scope | XGBoost unaffected unless it opts in | Low (one function, branching) | Moderate | Unaffected by default | Unaffected by default |
| **C** — dedicated GNN loader class, reusing the existing `GraphBuilder` (`GraphSnapshot` → `nx.DiGraph` conversion is already generic) | New Cypher query, same downstream types | Fully decoupled from XGBoost's call path | Low-moderate (only the query differs) | Two loaders, each single-purpose | Untouched | Untouched |
| **D** — fully independent GNN extraction stack (own snapshot/graph-building types, no shared code with `graph_feature_engine.py` at all) | Maximum isolation | Fully decoupled | Highest | Two independent stacks to maintain | Untouched | Untouched |

Not implemented this phase (per the explicit instruction to design,
not build, unless "purely additive and cannot affect XGBoost" — even
Option C/D, while low-risk, are left as design only, since this phase's
completion criteria list analysis, not implementation, as the bar).

---

## 13. Remaining Research Decisions

1. **Is the `GraphSnapshotLoader` untyped-pattern behavior (Section 10)
   a bug to fix (Option A/B, tightening scope) or a starting point to
   formalize (Option C/D, deliberately widening and typing what's
   currently accidental)?** No evidence of original intent either way.
2. **Which of Objectives A/C (the two not eliminated as circular or
   presently-infeasible) should the first real implementation target,
   or both in sequence?** (Section 6.)
3. **Is a low-bar, non-circular sanity evaluation (Section 9, raw
   attacker/host identity retrieval) an acceptable first published
   result, or does the project require holding out for a downstream-
   task improvement (Objective D) once independent ground truth exists?**
4. **What embedding dimension and regularization strategy avoids
   memorizing 71 samples across 34 shapes** (Section 2, 8) — a
   research/statistical decision once an objective is chosen, not
   answerable in the abstract.
5. **Should the two `graph_similarity` slots (Section 5) ever be
   populated by a future embedding-derived scalar, and if so, does that
   require re-tuning `OperationDecisionEngine.weights`/
   `config.CAMPAIGN_WEIGHTS`** — explicitly out of scope for this
   phase, flagged for whenever training is approved.

---

## 14. Testing

```bash
cd backend
python -m pytest tests/ -q
```

**376 passed** — unchanged. This phase added no test files (no safe
additive test was identified that wouldn't either be vacuous or require
touching production code) and changed no production code; all analysis
was produced by throwaway, uncommitted scripts calling only existing,
already-tested functions, deleted after use.
