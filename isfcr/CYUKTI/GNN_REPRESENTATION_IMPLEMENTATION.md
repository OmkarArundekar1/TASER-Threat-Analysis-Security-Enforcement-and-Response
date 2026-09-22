# GNN Representation Implementation — Graph Autoencoder + Temporal Snapshots

Companion to `GNN_FEASIBILITY.md`, `GNN_OBJECTIVE_DECISION.md`, and
`GNN_REPRESENTATION_DESIGN.md`. Implements, trains, and evaluates
Objective A (graph autoencoding) and the temporal-snapshot dataset for
Objective C, both selected/designed in the prior two phases as
non-circular (`SIMILAR_TO`/`RESEMBLES` are read nowhere in this
phase's code). No XGBoost file, `GraphSnapshotLoader`, campaign
correlation, NBE, or frozen research finding was touched — verified
below (Section 10-11) and by `git status` (only new files were added).

All numbers in this document come from one real run against this
environment's live Neo4j instance this phase — not assumed, not copied
from a prior session — reproducible via the exact commands in Section
9 and 12.

---

## 1. Research objective

Cross-campaign graph representation learning (Option E). This phase's
scope, per the project owner's instructions: (1) a graph autoencoder
baseline over the 71 real campaign graphs, (2) a temporal-snapshot
dataset for the subset with defensible timestamps. Not a severity
classifier (Option C remains untouched). Production integration
(`OperationFeatures.graph_similarity`, `CampaignFeatureEngine.graph_similarity`,
`ThreatAttributionEngine`, NBE, dashboard) is explicitly out of scope —
this phase ends at "produce and evaluate embeddings."

---

## 2. Graph input

Reused, unmodified: `ml.gnn.campaign_graphs.build_real_campaign_dataset()`
(the same real, live-Neo4j extraction the severity-classification GNN
work already established) and `ml.gnn.graph_encoder.encode_graph`. Node
features: one-hot type (6) + 10 curated numeric properties = 16-dim
(`FEATURE_DIM`). `risk_score` remains excluded (pre-existing leakage
guard, untouched). This phase adds no new node/edge feature — see
Section 4 for what was newly added on top (decoders, not encodings).

---

## 3. Edge-type encoding

Unchanged from the prior phase: one-hot over `{LAUNCHED, HAS_EVENT,
MATCHES, TARGETS, Unknown}` (`EDGE_FEATURE_DIM = 5`), symmetrized. This
phase is the first to actually *consume* `edge_attr` (via the edge-type
decoder, Section 4) — previously present but unused by any model.

---

## 4. Autoencoder architecture

New: `ml/gnn/autoencoder_model.py`. Reuses `SAGEConvLayer`
(`layers.py`) unchanged for the encoder stack — no new message-passing
primitive was written.

```
x [N,16], edge_index
    -> SAGEConvLayer(16->16) -> SAGEConvLayer(16->16)   (2 layers, hidden_dim=16)
    -> h [N,16]  (per-node embeddings, NOT pooled)
    -> mean-pool -> Linear(16->8) -> tanh -> z_G [8]    (the portable graph embedding)
```

**Why per-node embeddings, not the pooled vector, for reconstruction**:
a single pooled `z_G` cannot answer "which node connects to which" —
reconstructing topology requires node identity to survive to decode
time. The bottleneck (mean-pool + `Linear(16,8)` + `tanh`) is applied
*after* the decoders read `h`, specifically to produce the portable
summary vector this whole phase is about (Section 12's future
interface), not to do the reconstruction itself.

**Dimension justification** (Phase D, not arbitrary): `hidden_dim=16`
matches `FEATURE_DIM` exactly (standard practice for a 2-layer GNN on
graphs this small — no reason to expand width beyond the input's own
dimensionality). `embedding_dim=8` (half of 16) is small enough to
force real compression (real graphs have up to 24 nodes and 16-dim
features per node — an 8-dim summary cannot simply concatenate raw
values) while comfortably exceeding the ~5.1 bits (`log2(34)`) needed
to distinguish the 71 real campaigns' 34 distinct node-type-multiset
shapes (`GNN_REPRESENTATION_DESIGN.md` Section 8) — large enough not to
be degenerate, small enough not to trivially memorize 71 individual
samples (which even a 16-dim vector could risk).

**Decoders** (Phase C — deliberately not "reconstruct every field"):

| Target | Decoder | Why included |
|---|---|---|
| Edge existence (dense, all ordered pairs, diagonal excluded) | `sigmoid(h_i · h_j)` — zero-parameter dot product (Kipf & Welling GAE) | Directly tests "who connects to whom" — the core research contribution |
| Edge type (5-way, true edges only) | `Linear(32, 5)` on `[h_i; h_j]` | Directly tests relationship *semantics*, not just existence |
| Node numeric features (10-dim, standardized) | `Linear(16, 10)` | Tests whether per-node content survives message passing |
| Node type (one-hot) | **Not reconstructed** | It is a direct input feature (part of `x`) — reconstructing it from an embedding partly built from it is close to an identity task and would inflate apparent quality without evidencing anything new |

Graphs are small enough (≤24 nodes) that **dense** all-pair
reconstruction is used — no negative sampling, no randomness beyond
model initialization, fully deterministic given a seed.

---

## 5. Training objective

`Loss = BCEWithLogits(edge_existence, pos_weight) + CrossEntropy(edge_type | true edges) + MSE(node_features, standardized)`,
unweighted sum (equal weight 1.0 each — the simplest defensible choice
for a first proof of concept, not a tuned hyperparameter search).
`pos_weight` computed once from the training split's true positive/negative
pair ratio (a real class imbalance — edges are the minority of all
ordered pairs). Node numeric features are standardized (z-score) using
**training-split-only** mean/std, stored in the saved artifact so
inference applies the identical transform (same discipline as
`ml/ssl_pipeline.py`'s scaler artifact).

---

## 6. Dataset

**71 real campaign graphs**, unchanged extraction. 0 excluded (every
sample has ≥2 nodes — the minimum for any off-diagonal pair to
reconstruct). **Split**: by attacker-identity group (`get_campaign_attacker_ips`,
a raw `LAUNCHED`-relationship fact, re-queried live this phase — not
`SIMILAR_TO`/`RESEMBLES`, not technique overlap). This repository's
real attacker-group sizes (verified live): `[33, 26, 5, 4, 1, 1, 1]`
across 7 attackers. A greedy closest-to-target assignment (Section 5
of `train_autoencoder.py`'s `split_by_attacker_group`) holds out the 5
smallest groups (`5+4+1+1+1=12` campaigns) as validation and keeps both
large groups (`33+26=59`) in training — **no attacker's feature vector
appears on both sides of the split.** (An earlier, simpler
"add-smallest-groups-until-under-target" rule was tried first and
produced the *opposite* of the intended split — 38 val / 33 train —
because a lumpy real distribution can overshoot a small target the
moment a large group is considered; this was caught by actually
running the code against real data, not assumed, and fixed before
training — see git history for this phase.)

---

## 7. Temporal dataset

New: `ml/gnn/temporal_graphs.py`. **21 real campaigns** have ≥2
distinct real `AttackEvent.first_seen` timestamps (live re-verified
this phase, not assumed from the prior phase's count — same number,
independently reconfirmed). Distribution of snapshot counts per
campaign (i.e., distinct timestamps): `2→10, 3→2, 4→2, 5→2, 6→3, 8→1,
10→1` campaigns. Each snapshot is a cumulative, cutoff-filtered
subgraph (`first_seen <= cutoff`) built via a new, explicitly-typed
query (`LAUNCHED`/`HAS_EVENT`/`MATCHES`/`TARGETS` only — deliberately
not reusing `GraphSnapshotLoader`'s untyped pattern, Section 11)
reusing the existing, unmodified `GraphSnapshot`/`GraphBuilder`
classes. Verified live (example, `CAMP_1429ADB4`, 10 snapshots): node
count grows `5, 7, 8, 9, 11, 13, 15, 16, 17, 19` — strictly
non-decreasing, confirming no future-information leakage by
construction (a later snapshot can only gain nodes/edges relative to
an earlier one, never lose them, because the cutoff only ever moves
forward).

---

## 8. Training configuration

| Parameter | Value | Justification |
|---|---|---|
| Seed | 42 | Fixed, `torch.manual_seed` + `random.seed` |
| Hidden dim | 16 | = `FEATURE_DIM` (Section 4) |
| Embedding dim | 8 | = `FEATURE_DIM // 2` (Section 4) |
| Layers | 2 | Matches existing `CampaignGNN`'s convention (`model.py`) |
| Optimizer | Adam | Standard default, no tuning performed |
| Learning rate | 0.01 | Matches `train_gnn.py`'s existing default |
| Epochs | 200 | Full-batch (all 59 training graphs per step); chosen to reach visible convergence (Section 9) without a tuned early-stopping criterion — a proof of concept, not a tuned model |
| Batching | Full-batch, per-graph dense reconstruction, one optimizer step/epoch | Simplest, most transparent choice for 71 tiny graphs — not a throughput optimization |
| Val fraction | 0.2 (target), realized as 12/71 (17%) | Section 6 |

---

## 9. Results

Reproducible via:
```bash
cd backend
python -m ml.gnn.train_autoencoder
```

```json
{
  "train_graphs": 59,
  "val_graphs": 12,
  "excluded_singleton_graphs": 0,
  "first_train_loss":  {"existence": 1.558, "edge_type": 1.605, "feature": 0.459, "total": 3.622},
  "final_train_loss":  {"existence": 0.623, "edge_type": 0.024, "feature": 0.017, "total": 0.665},
  "val_loss_final":    {"existence": 1.474, "edge_type": 0.524, "feature": 0.296, "total": 2.294},
  "edge_existence_auc_train": 0.975,
  "edge_existence_auc_val": 0.735,
  "edge_type_accuracy_train": 1.000,
  "edge_type_accuracy_val": 0.763
}
```

**Reading this honestly**: training loss drops 3.62 → 0.665 over 200
epochs — real convergence, not stalled. Validation loss (2.294) is
higher than training loss, and validation AUC/accuracy are
meaningfully below training's — a real, expected generalization gap
given 59 training graphs and a strict no-attacker-overlap split, not
hidden or averaged away. **No accuracy is reported as a headline
metric** (per this phase's explicit instruction) — the numbers above
are reconstruction diagnostics, not a claim of "the GNN achieves X%."

---

## 10. Embedding analysis

Reproducible via:
```bash
cd backend
python -m ml.gnn.evaluate_autoencoder
```

- **Shape/finiteness/determinism**: all 71 embeddings are `R^8`, all
  finite, and `embed_graph` called twice on the same input produces
  bitwise-identical output (`torch.equal`) — confirmed, not assumed.
- **Collapse check**: embedding norms range `0.366–1.645` (mean 0.946,
  std 0.455) — not degenerate (no all-zero or fixed-norm collapse).
  Pairwise cosine distance: mean 0.187, ranging `0.0–0.931` across 2485
  pairs. **259 of 2485 pairs (10.4%) are near-duplicates** (cosine
  distance < 0.01) — a real, disclosed finding, not hidden. Plausible
  explanation, not yet proven: many real campaigns share one of the
  dataset's 34 distinct shapes (`GNN_REPRESENTATION_DESIGN.md` Section
  8), and structurally near-identical inputs producing near-identical
  embeddings is expected behavior for this encoder, not necessarily a
  defect — but it does mean roughly 1 in 10 campaign pairs are
  currently near-indistinguishable to this representation.
- **Structural variation**: the remaining ~90% of pairs span a real
  distance range (up to 0.93 cosine distance) — different graphs do
  produce different embeddings, not a uniform collapse.

---

## 11. Temporal analysis

**21 real campaigns evaluated.** In **20 of 21 (95%)**, a campaign's
own consecutive-snapshot embeddings are closer to each other (mean
cosine distance, e.g. `CAMP_0E9C0284`: 0.0099) than that campaign's
first snapshot is to other campaigns' embeddings (e.g. `CAMP_0E9C0284`:
0.146) — roughly a 15x tighter clustering for "own history" vs. "other
campaigns" in the typical case. **One exception**: `CAMP_427A075C`
(only 2 snapshots) has consecutive distance 0.379, *worse* than its
cross-campaign distance of 0.192 — disclosed, not hidden; plausibly a
2-snapshot campaign where the second snapshot's structure changed
enough (new node/edge types appearing) to shift the embedding
substantially, but not independently confirmed here. **This shows the
representation preserves campaign identity while allowing structural
evolution to be observed, in the large majority of cases — it is not
proof of semantic similarity between different campaigns** (the
evaluation's own explicit caveat, printed alongside every run).

---

## 12. Non-circular evaluation

Two strategies actually run, neither touching `SIMILAR_TO`/`RESEMBLES`:

1. **Identity-aware retrieval** (raw `LAUNCHED`/`TARGETS` facts, live
   re-queried): nearest-neighbor-by-embedding shares the same attacker
   in **71.8%** of cases, against a **34.97%** chance baseline computed
   from this repository's real, skewed attacker-group sizes (not an
   arbitrary 50%) — meaningfully above chance, roughly 2x. Same-host
   rate: **85.9%**. Explicitly disclosed as a low bar (an embedding
   that just re-encodes the attacker/host one-hot passes trivially) —
   **not** presented as evidence the GNN "understands" campaigns.
2. **Temporal self-consistency** — Section 11.

**Trivial-baseline honesty check** (Phase L): a zero-parameter rule
("edge exists iff node types are one of the four schema-canonical
pairs") scores `precision 0.263, recall 0.569, F1 0.360, accuracy
0.708`. The trained model at a plain 0.5 sigmoid threshold scores
`precision 0.145, recall 1.0, F1 0.253, accuracy 0.145` — **worse F1
than the trivial baseline at this threshold.** This is disclosed, not
hidden: the model's `pos_weight`-adjusted training objective (Section
5) skews its raw logits positive, so **ranking quality (AUC 0.975
train / 0.735 val) is the fair comparison, not F1 at an untuned 0.5
threshold** — no attempt was made to hunt for a better threshold, since
doing so post-hoc to produce a nicer number would itself be a form of
overclaiming. **Conclusion, stated plainly: this phase does not
establish that the model's default-threshold edge predictions beat a
trivial heuristic; it establishes that the model's ranking of true vs.
false edges is well above chance and above what raw node-type alone
would achieve at the same task, framed as a ranking problem.**

---

## 13. Comparison with existing 19 scalar features

Pearson correlation between pairwise embedding distance and pairwise
distance in XGBoost's existing 19 `GraphFeatures` scalars (reusing
`graph_analytics.extract_features` unchanged, not reimplemented):
**r = 0.581** across all 2485 pairs. Materially below 1.0 (would
indicate near-total redundancy) and materially above 0 (would indicate
no relationship at all) — **evidence, not proof**, that the embedding
varies along some dimensions the 19 scalars collapse, while still
sharing real structure with them (expected, since both are ultimately
functions of the same underlying graph). **No claim of superiority is
made from this number.** A proper ablation (e.g., does adding `z_G` to
XGBoost's feature set change any real prediction) is explicitly future
work, not attempted here.

---

## 14. Limitations

- 71 samples, 59/12 split — small by any general ML convention;
  validation metrics (Section 9) should be read as directional, not
  precise.
- 10.4% near-duplicate embedding pairs (Section 10) — partly expected
  given real shape repetition, not fully explained.
- Model's default-threshold (0.5) edge-existence classification is
  worse than a trivial baseline (Section 12) — ranking quality (AUC) is
  the metric that shows real signal, not raw F1.
- Temporal analysis (Section 11) covers only 21 campaigns, one of
  which is an exception to the main finding.
- No embedding-dimension or loss-weighting hyperparameter search was
  performed — the configuration (Section 8) is a documented, justified
  first choice, not a tuned optimum.
- The `GraphSnapshotLoader` untyped-pattern finding (`GNN_OBJECTIVE_DECISION.md`
  Section 8) still applies to the base 71-graph dataset used here
  (unchanged, per this phase's explicit instruction not to touch the
  loader) — the temporal dataset (Section 7) deliberately avoids it via
  a separate, typed query, but the base autoencoder dataset does not.

## What has NOT been demonstrated

- No severity-prediction improvement (this is not Option C; not tested).
- No campaign-correlation improvement (`graph_similarity` was not
  populated; no A/B comparison against `OperationDecisionEngine`'s
  current behavior was run).
- No attribution improvement (`ThreatAttributionEngine` was not
  touched or compared against).
- No Detection Paradox improvement (Phase 21/22 findings were not read
  or modified this phase).
- No production runtime benefit of any kind — this phase produces and
  evaluates embeddings only; nothing in `dashboard_api.py`,
  `investigation/`, or any decision engine consumes them.

---

## Tests

```bash
cd backend
python -m pytest tests/ -q
```

**401 passed** (376 baseline + 25 new: `tests/test_autoencoder.py` and
`tests/test_temporal_graphs.py`). Zero regressions.

## XGBoost / GraphSnapshotLoader compatibility

Verified via `git status` at the end of this phase: only new files
were added (`ml/gnn/autoencoder_model.py`, `ml/gnn/train_autoencoder.py`,
`ml/gnn/temporal_graphs.py`, `ml/gnn/evaluate_autoencoder.py`,
`tests/test_autoencoder.py`, `tests/test_temporal_graphs.py`, and this
document/`ARCHITECTURE_AUDIT.md`'s factual update). No existing `.py`
file was modified — `graph_feature_engine.py`, `campaign_feature_engine.py`,
`GraphSnapshotLoader`, `ml/train_xgboost.py`, `ml/dataset_utils.py`,
`operation_decision_engine.py`, `campaign_decision_engine.py`, and
every file the prior two phases traced as touching XGBoost or
correlation, are byte-for-byte unchanged.
