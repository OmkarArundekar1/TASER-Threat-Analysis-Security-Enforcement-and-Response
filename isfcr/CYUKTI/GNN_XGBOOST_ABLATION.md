# GNN Embedding Ablation Against Existing XGBoost

Companion to `GNN_FEASIBILITY.md`, `GNN_OBJECTIVE_DECISION.md`,
`GNN_REPRESENTATION_DESIGN.md`, and `GNN_REPRESENTATION_IMPLEMENTATION.md`.
Tests, rather than assumes, whether the trained graph embedding `z_G`
adds measurable predictive information to CYUKTI's existing 57-feature
XGBoost severity classifier. **No production model, artifact, or
pipeline was changed.** All numbers below come from one real run
against this environment's live Neo4j instance and the real, persisted
`ml/datasets/campaign_dataset.csv` this phase — not assumed, not
copied from a prior session.

---

## 1. Research Question

> Does adding the learned graph representation `z_G` to CYUKTI's
> existing campaign feature representation provide measurable
> additional predictive information for severity classification?

This is an ablation, not a redesign of the GNN and not a production
integration. `SIMILAR_TO`/`RESEMBLES` are read nowhere in this phase's
code (verified by grep of `ml/gnn/xgboost_ablation.py`).

---

## 2. Existing XGBoost Baseline

Traced from `ml/train_xgboost.py`, `ml/dataset_utils.py`,
`ml/feature_schema.py` (not assumed):

- `FEATURE_COLUMNS`: 57 columns (`ml.dataset_utils.FEATURE_COLUMNS`,
  reverified this phase — `CampaignDatasetRecord`'s 66 fields minus 3
  identifiers, 5 labels, 1 leakage column).
- Production hyperparameters (`XGBoostCampaignClassifier.train`
  defaults): `n_estimators=200, max_depth=6, learning_rate=0.1,
  objective="multi:softprob" (>2 classes), eval_metric="mlogloss",
  random_state=42, n_jobs=-1`, `compute_sample_weight("balanced",
  y_train)`, plus an internal 80/20 split with
  `CalibratedClassifierCV` for probability calibration.
- Persisted dataset: `ml/datasets/campaign_dataset.csv`, **60 rows**,
  66 columns (reverified this phase, matches prior sessions' reports).
  Severity distribution: `Low: 53, Critical: 4, Medium: 3` (**High: 0**,
  unchanged from every prior phase's finding).
- Runtime path: `campaign_id → 57 features (7 of them, e.g.
  prediction_similarity/chain_similarity, require live alert-time
  context via CampaignFeatureEngine.extract(context, current_technique,
  attacker_ip) — not recomputable after the fact from campaign_id
  alone) → XGBoost → severity`.

**Nothing in this section's traced pipeline was modified.**

---

## 3. GNN Representation

Unchanged from `GNN_REPRESENTATION_IMPLEMENTATION.md`: 2-layer
`SAGEConvLayer` encoder (hidden_dim=16), mean-pool + `Linear(16,8)` +
`tanh` bottleneck, `embedding_dim=8`. This phase reuses
`ml.gnn.train_autoencoder.fit_autoencoder`/`embed_samples` (extracted
from `train_autoencoder()` this phase via a behavior-preserving
refactor — its own existing tests and a byte-identical reproduction of
its previously-reported `edge_existence_auc_train=0.9746163608077244`
were both reverified after the refactor) — no new encoder architecture
was written for this ablation.

---

## 4. Dataset Mapping

**Traced explicitly, not assumed** (`ml.gnn.xgboost_ablation.load_ablation_dataset`):

| Set | Count |
|---|---|
| Campaigns in the persisted CSV | 60 |
| Campaigns in the live 71-graph GNN extraction | 71 |
| **Intersection (the ablation-usable set)** | **60** |
| In CSV but not live GNN | 0 |
| In live GNN but not CSV | **11** |

The CSV is a strict subset of the live GNN extraction — **every**
persisted-dataset campaign still exists live in Neo4j. The 11
GNN-only campaigns (`CAMP_0053B82E, CAMP_18582D74, CAMP_54643C88,
CAMP_57160EE8, CAMP_740C421F, CAMP_7AFACCAB, CAMP_8165192E,
CAMP_9F5B5001, CAMP_A2F3DC2D, CAMP_D1DCCF3E, CAMP_E2601E7B`) accumulated
in Neo4j after the CSV was last written and have no persisted 57-feature
row (7 of those features require replaying live alert-time context that
no longer exists) — **excluded, disclosed, not silently dropped.**
**Ablation dataset: exactly the 60 CSV campaigns.**

---

## 5. Leakage Prevention

For each evaluation fold (Section 6): a **fresh** `GraphAutoencoder` is
fit via `fit_autoencoder(train_samples, ...)` using **only** that
fold's training campaigns. `feature_mean`/`feature_std` (numeric-property
standardization) and `pos_weight` are computed exclusively from
`train_samples` — mechanically verified this phase
(`test_fold_safe_feature_stats_computed_only_from_train_samples`: an
implausible sentinel value (9999.0) planted on a held-out sample's
features does not appear in the fitted mean). Held-out campaigns are
embedded via `embed_samples(model, test_samples, ...)` — inference
only (`model.eval()`, `torch.no_grad()` inside `embed_graph`), never
influencing the encoder's parameters, the standardization statistics,
or the reconstruction loss. The held-out campaigns' graphs are **never
passed to `fit_autoencoder`** — the function's signature makes this
structurally impossible, not just a runtime discipline.

---

## 6. Split Strategy

**Leave-One-Attacker-Group-Out (LOGO) cross-validation**, not a single
holdout — chosen over Option A (a single grouped holdout, as used for
the standalone GNN training in the prior phase) because the 60-campaign
ablation population has only **3** distinct attacker groups (not the
full 71-campaign population's 7 — traced explicitly, not assumed:
the 11 excluded GNN-only campaigns account for all 4 of the smaller
attacker groups from the prior phase's 71-sample split). With only 3
groups, LOGO gives 3 folds using every campaign as held-out test data
exactly once, which is more informative than a single small holdout
would be on a population this size.

**Real group composition (reverified this phase)**:

| Attacker group | Size | Severity composition |
|---|---|---|
| `192.168.56.106` | 33 | 27 Low, 3 Medium, 3 Critical |
| `pes1ug23cs411-VirtualBox` | 23 | 23 Low (single class) |
| `192.168.56.105` | 4 | 3 Low, 1 Critical |

**A real, disclosed limitation of this split, not hidden**: the fold
that holds out `192.168.56.106` (the only group with Medium examples)
trains on the other two groups — 27 campaigns containing **zero**
Medium and only **1** Critical example. XGBoost's sklearn wrapper
requires contiguous present-class labels for `multi:softprob`
(reproduced and unit-tested this phase,
`test_fit_and_eval_xgb_handles_class_missing_from_training`) — handled
via a per-fold `LabelEncoder` fit on training labels only, with
predictions mapped back to the fixed 3-label space before scoring, so
a class absent from a fold's training data correctly scores 0
precision/recall/F1 on that fold rather than crashing or being
silently omitted. **This means that fold's Medium/Critical performance
is expected to be near-floor regardless of features — a property of
this tiny dataset's group structure, not of either model.**

No `SIMILAR_TO`/`RESEMBLES`/technique-overlap was used to construct
groups or folds — attacker identity (`LAUNCHED`, a raw fact) only.

---

## 7. Baseline Model

`57 existing CampaignDatasetRecord features → XGBoost (production
hyperparameters, Section 2, no calibration step — see Section 9's
disclosed deviation) → severity`, fit fresh per fold on that fold's
training rows.

---

## 8. GNN-Augmented Model

`57 existing features + 8-dimensional z_G (fold-safe, Section 5) →
identical XGBoost hyperparameters → severity`, fit fresh per fold, same
training rows plus 8 columns. **The only difference between the two
conditions in any fold is these 8 columns** — same rows, same labels,
same seed, same hyperparameters (Section 9).

---

## 9. Metrics

Accuracy, macro F1, weighted F1, balanced accuracy, per-class
precision/recall/F1, confusion matrix — computed both **per fold** and
**pooled** (out-of-fold predictions across all 3 folds combined into
one 60-campaign confusion matrix, since some individual folds are too
small/label-starved for a stable per-fold macro-F1 alone). **No High
class exists in this dataset (0 examples anywhere) — not reported,
not manufactured** (consistent with every prior phase's finding).

**One disclosed methodological deviation from production, applied
identically to both conditions**: no `CalibratedClassifierCV` step —
calibration requires yet another internal split on data already this
scarce for some folds (e.g. the 4-campaign fold). Raw
`predict`/`predict_proba` are compared directly for both baseline and
augmented, so the comparison stays fair even though it differs from
the deployed artifact's own calibration behavior.

**No hyperparameter search was performed for either condition** —
production's exact `n_estimators=200, max_depth=6, learning_rate=0.1`
reused unchanged.

---

## 10. Results

Reproducible via:
```bash
cd backend
python -m ml.gnn.xgboost_ablation
```

### Per-fold

| Held-out group | Train / Test size | Train label counts | Test label counts | Baseline acc / macro-F1 | Augmented acc / macro-F1 |
|---|---|---|---|---|---|
| `192.168.56.105` | 56 / 4 | Low 50, Medium 3, Critical 3 | Low 3, Critical 1 | 1.000 / 0.667 | 1.000 / 0.667 |
| `192.168.56.106` | 27 / 33 | Low 26, Critical 1 (**no Medium**) | Low 27, Medium 3, Critical 3 | 0.818 / 0.300 | 0.818 / 0.300 |
| `pes1ug23cs411-VirtualBox` | 37 / 23 | Low 30, Medium 3, Critical 4 | Low 23 (single class) | 1.000 / 0.333 | 1.000 / 0.333 |

**Every metric, every fold, is bit-identical between baseline and
augmented** — not approximately equal, exactly equal (same accuracy,
same macro/weighted F1, same balanced accuracy, same confusion matrix,
same per-class precision/recall/F1 in all three folds).

### Pooled (out-of-fold, all 60 campaigns)

| Metric | Baseline | Augmented |
|---|---|---|
| Accuracy | 0.900 | 0.900 |
| Macro F1 | 0.449 | 0.449 |
| Weighted F1 | 0.863 | 0.863 |
| Balanced accuracy | 0.417 | 0.417 |

**Delta: exactly 0 on every metric.**

---

## 11. Per-Class Analysis

Pooled, identical for both conditions:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Critical | 1.000 | 0.250 | 0.400 | 4 |
| Low | 0.898 | 1.000 | 0.946 | 53 |
| Medium | 0.000 | 0.000 | 0.000 | 3 |

**Medium is never correctly predicted by either model** — directly
explained by Section 6's finding (the only fold with Medium in its
test set has zero Medium in its training set). **Critical recall
(0.25) is low but non-zero** — 1 of 4 Critical campaigns correctly
identified, again identical between conditions.

**No class improved or degraded** — the augmented model made
byte-identical predictions to the baseline on every single held-out
campaign, in every fold.

---

## 12. Embedding Diagnostics

**Verified this phase, not assumed**: the embeddings themselves are
real, finite, and non-degenerate in every fold — e.g. the
`192.168.56.106`-held-out fold's 27 training embeddings have per-
dimension means ranging `-0.70` to `0.56` and per-dimension standard
deviations `0.077`–`0.222` (no collapsed/constant dimension, no NaN).

**Direct mechanistic explanation for the zero delta**: inspecting
`XGBClassifier.feature_importances_` for the augmented model (same
fold) shows **all 8 embedding columns at exactly `0.0` importance** —
summing to `0.0` out of a total `1.0`. XGBoost's greedy per-node
split selection **never once chose to split on any embedding
dimension**, in this fold (and, given the pooled predictions are
identical across all three folds, plausibly in every fold). **This is
not a bug in the ablation code — it is the mechanism by which the
result occurred**: with 57 existing features already available,
`max_depth=6`/`n_estimators=200` trees found every split they needed
without the embedding, on this dataset's severity distribution.

### 10.4%-near-duplicate-embedding diagnostic (not the main task)

Not re-investigated in depth this phase (explicitly out of scope,
"diagnostic only, do not modify the model because of it" per this
phase's instructions). Given the mechanistic finding above (zero
feature importance, not "embeddings too similar to discriminate"), the
near-duplicate rate from the prior phase is not the primary explanation
for this phase's null result — the null result held even in folds
where the embeddings show real per-dimension spread (this section).

---

## 13. Limitations

- Only 3 attacker groups exist in the 60-campaign ablation population
  — LOGO here means 3 folds, not a large-N cross-validation.
- One fold's training data has zero Medium and one Critical example —
  that fold's minority-class metrics are structurally floor-bound
  regardless of any feature added.
- 60 campaigns, 3 usable severity classes (High absent) — small by any
  general ML convention; a null result here should not be read as a
  definitive, permanent answer.
- No hyperparameter search, no calibration step (Section 9) — a
  differently-tuned model (for either condition) was not explored.
- The embedding was trained via an unsupervised, severity-oblivious
  objective (graph autoencoding) — this ablation tests exactly that
  embedding, not a severity-supervised or contrastively-trained
  alternative representation.
- `GraphSnapshotLoader`'s untyped-pattern finding
  (`GNN_OBJECTIVE_DECISION.md` Section 8) still applies to the
  underlying 60/71-campaign extraction used here, unchanged.

## 14. Interpretation

**What the results support**: in this experiment — this dataset, this
split, this embedding, these hyperparameters — adding `z_G` to
XGBoost's existing 57 features produced **no measurable change** in
severity-prediction performance, on any metric, in any fold. The
mechanistic reason is directly observable (zero feature importance on
every embedding column), not inferred: XGBoost's tree-building never
found a split on any embedding dimension more useful than the splits
already available from the existing features, on this label
distribution.

**What this does not mean**: it does not mean the embedding carries no
information (Section 12 shows real, non-degenerate variation across
campaigns; `GNN_REPRESENTATION_IMPLEMENTATION.md`'s r=0.581 correlation
with the 19 scalar features already showed partial non-redundancy). It
means that whatever non-redundant information the embedding carries
was **not useful for this specific task** (3-class severity
prediction) **under this specific split and hyperparameter
configuration**.

## 15. What The Experiment Does NOT Prove

> **A null result on this ablation does not prove the GNN failed to
> learn meaningful structure, and — symmetrically — a positive result
> would not have proven the GNN learned semantically meaningful
> cyber-attack topology.** Improvement (or its absence) in XGBoost
> severity-prediction performance is evidence about one narrow,
> specific question — incremental utility for this one downstream
> task, under this one split, with these specific hyperparameters — not
> a general verdict on whether the representation is "good" or
> "understands" attack campaigns.

Additionally, not established by this experiment:
- Whether a differently-configured XGBoost (deeper trees, feature
  selection, regularization tuned for 65 vs. 57 columns) would behave
  differently — not tested, per this phase's explicit "do not tune"
  instruction.
- Whether the embedding would help a different downstream task
  (campaign correlation, retrieval, attribution) — not tested here.
- Whether more real data (resolving the group/class scarcity in
  Section 6) would change this outcome — not testable with current data.
- Causality of any kind — only an observed, reproducible association
  (or, here, its exact absence) under a controlled comparison.

---

## Tests

```bash
cd backend
python -m pytest tests/ -q
```

New: `tests/test_xgboost_ablation.py` (12 tests — dataset-mapping
mismatch reporting, attacker grouping, fold-safety of feature
standardization, missing-training-class handling, artifact-isolation
guard, determinism, plus 2 live-Neo4j end-to-end tests). No existing
test was weakened.

## Reproducibility record

Dataset version: `ml/datasets/campaign_dataset.csv` (60 rows, as it
exists in this repository at the time of this phase) joined with the
live Neo4j instance's 71-campaign GNN extraction (Section 4). Seed 42
throughout (GNN encoder fitting and XGBoost `random_state`). GNN
architecture/hyperparameters: Section 3 (unchanged from
`GNN_REPRESENTATION_IMPLEMENTATION.md`). XGBoost configuration:
Section 2/9. Split: Section 6 (LOGO over 3 real attacker groups,
re-derivable live via `ml.gnn.xgboost_ablation.build_attacker_groups`).
Feature ordering: `ml.dataset_utils.FEATURE_COLUMNS` (57, unchanged)
followed by embedding dimensions `z_0..z_7` in the order
`GraphAutoencoder.pooled_embedding` produces them. Metric definitions:
`sklearn.metrics` (`accuracy_score`, `f1_score` average="macro"/"weighted",
`balanced_accuracy_score`, `classification_report`, `confusion_matrix`),
all with `labels=["Critical","Low","Medium"]` and `zero_division=0`
explicitly fixed for reproducibility across environments/sklearn
versions.
