# GNN Retrieval / Correlation Evaluation

Companion to `GNN_FEASIBILITY.md`, `GNN_OBJECTIVE_DECISION.md`,
`GNN_REPRESENTATION_DESIGN.md`, `GNN_REPRESENTATION_IMPLEMENTATION.md`,
and `GNN_XGBOOST_ABLATION.md` (which produced a clean severity-prediction
null result — **not reopened here**). Tests a narrower, independent
hypothesis: does `z_G` provide topology-aware value for
retrieval/correlation, a task the severity model was never meant to
serve? No production system was modified. `SIMILAR_TO`/`RESEMBLES` are
read nowhere in this phase's code (`ml/gnn/retrieval_evaluation.py`
contains no reference to either).

All numbers below come from one real run against this environment's
live Neo4j instance this phase (`python -m ml.gnn.retrieval_evaluation`,
~7 minutes — 7-fold Leave-One-Attacker-Group-Out, one fresh
autoencoder trained per fold) — not assumed, not copied. Current
backend baseline at the start of this phase: **413 passed** (the
handoff's stated "401" was one phase stale — the XGBoost ablation
phase already added 12 tests; noted here as a factual correction, not
a discrepancy in this phase's own work).

---

## 1. Research Question

> Does CYUKTI's learned graph representation `z_G` provide useful
> topology-aware information for campaign/operation retrieval or
> correlation, independent of whether it helps severity prediction (it
> does not — `GNN_XGBOOST_ABLATION.md`)?

---

## 2. Existing Similarity Mechanisms

Re-verified this phase against current code (unchanged since the last
phase traced them — no new evidence contradicts prior findings):

| Mechanism | Input | Similarity basis | Ground truth | Topology-aware? |
|---|---|---|---|---|
| `OperationDecisionEngine` (campaign→operation) | `OperationFeatures` (7 scalars) | `0.30·technique_Jaccard + 0.20·attacker_id + 0.20·temporal + 0.15·victim_id + 0.10·chain_LCS + 0.05·prediction + 0.00·graph_similarity` | Threshold 0.70 on the weighted sum itself (no external label) | **No** — `graph_similarity` is `operation_feature_engine.py`'s literal `return 0.0` |
| `CampaignDecisionEngine` (active-campaign continue/close) | `CampaignFeatures` (7 scalars) | `0.35·prediction + 0.30·chain_LCS + 0.20·temporal + 0.10·attacker_id + 0.03·runtime + 0.01·graph_similarity (real) + 0.01·duplicate` | Threshold 0.35 | Marginally — 1% weight on a real (not stub) 3-scalar composite (`graph_density`/`attack_chain_depth`/`campaign_complexity`) |
| `ThreatAttributionEngine` (campaign→actor) | Technique-ID sets + chain | `0.50·coverage + 0.20·precision + 0.30·chain_LCS` | Top-K ranking, no external label | No |
| `SIMILAR_TO` writer (`neo4j_client.update_campaign_similarity`) | Technique sets + attacker/host identity | `60%·technique_Jaccard + 20%·shared_attacker + 20%·shared_host`, threshold ≥75 | Self-referential (the formula defines the edge) | No |
| `RESEMBLES` writer (`neo4j_client.update_actor_attribution`) | Technique sets | `overlap_ratio·100 + shared_count·2`, threshold ≥50 | Self-referential | No |
| Campaign-narrative TF-IDF (`rag/campaign_retriever.py`) | Formatted attacker/victim/technique text | Bag-of-words cosine | Query-specific relevance, no fixed label | No |

**Both `graph_similarity` slots distinguished, independently traced (not
conflated), unchanged from prior phases**: `OperationFeatures.graph_similarity`
is a true zero-weight stub feeding operation correlation.
`CampaignFeatures.graph_similarity` is real but tiny-weight (1%),
feeding `campaign_manager.py`'s active-campaign continue/close
decision, and is itself only 3 of the 19 existing hand-engineered
scalar features (`graph_density`, `attack_chain_depth`,
`campaign_complexity`) — **not** the same Neo4j information the GNN
consumes at the per-node level; both read the same underlying graph via
`graph_analytics`, but one collapses it to 3 numbers, the other
preserves per-node structure. **Neither slot was touched this phase.**

---

## 3. GNN Representation

Unchanged (`GNN_REPRESENTATION_IMPLEMENTATION.md`). Two embedding sets
used this phase, kept explicitly separate per this phase's own
instruction:

1. **Fold-safe / generalization** (primary evidence): Leave-One-Attacker-Group-Out
   over **all 71 real campaigns' 7 attacker groups** (not the 60-campaign,
   3-group subset `GNN_XGBOOST_ABLATION.md` used — retrieval needs only
   graphs, not the persisted 57-feature CSV, so the full population is
   used here). A fresh `GraphAutoencoder` is fit per fold on that fold's
   training campaigns only (`fit_autoencoder`, reused unchanged); every
   campaign's embedding comes from a model that never saw its graph.
2. **Single-model / identity-preservation only** (explicitly weaker,
   used only where labeled as such): the one model trained on all 71
   campaigns together in a prior phase (`ml/models/gnn_autoencoder.pt`,
   **reused, not retrained**, per this phase's explicit instruction).
   59 of 71 campaigns' embeddings here come from data the model
   directly trained on.

**A real, disclosed methodological limitation of (1)**: the 7 fold-specific
models are independently trained (same seed/initialization, different
training data per fold), so pooling their embeddings into one ranking
space is not a single jointly-fit geometry — a genuine caveat on how
much weight to put on the fold-safe retrieval numbers below, addressed
directly in Section 12.

---

## 4. Evaluation Tasks

Four tasks investigated, per this phase's own framing:

- **A — Temporal retrieval** (fold-safe): query = an early snapshot's
  embedding; relevant = that same real campaign's later snapshot,
  among a pool of ~70 other real campaigns' embeddings. Campaign ID is
  never used as a feature (it isn't one — `graph_encoder.py`'s `x`
  contains no identifier).
- **B — Same-attacker retrieval**: relevant = other real campaigns
  sharing the same `LAUNCHED` attacker IP.
- **C — Same-host retrieval**: relevant = other real campaigns sharing
  the same `TARGETS` host IP.
- **D — Structural diagnostic**: real campaign pairs at the extremes of
  (technique overlap, topology similarity) disagreement — no invented
  label, only measurable statistics (node-type multiset "shape",
  already established in `GNN_OBJECTIVE_DECISION.md` Section 8).

---

## 5. Ground Truth Definition

Every ground truth used is a raw, independently-observable fact, never
a derived similarity score:

| Task | Ground truth | Why non-circular |
|---|---|---|
| A (temporal) | "this snapshot and that snapshot belong to the same `Campaign.campaign_id`" | An identity fact, not a similarity judgment |
| B (attacker) | `Attacker -[LAUNCHED]-> Campaign` | A raw relationship, not a weighted score |
| C (host) | `Campaign -[TARGETS]-> Host` | Same |
| D (structural) | Node-type-multiset "shape" (already-computed graph statistic) vs. real `MATCHES`-derived technique sets | Both are raw extracted facts, compared to each other, not to a manufactured label |

---

## 6. Circularity Controls

Verified by direct inspection of `ml/gnn/retrieval_evaluation.py`:
zero occurrences of `SIMILAR_TO` or `RESEMBLES` anywhere in the file.
`similar_to_formula_score` recomputes `neo4j_client.update_campaign_similarity`'s
**formula** fresh in Python from raw technique/attacker/host facts — it
is a candidate *representation* to compare, exactly like technique-Jaccard
or the 19 scalar features, never read as a stored edge and never used
as a label for anything (unit-tested:
`test_similar_to_formula_score_matches_real_neo4j_client_weights`).

---

## 7. Metrics

Mean Reciprocal Rank (MRR) and Recall@5, computed only for tasks with
legitimate ground truth (Section 5) — no metric was computed against
`SIMILAR_TO`/`RESEMBLES`.

---

## 8. Results

Reproducible via:
```bash
cd backend
python -m ml.gnn.retrieval_evaluation
```

### Task A — Temporal retrieval (fold-safe, 21 campaigns, ~71-candidate pool each)

| Metric | Value | Chance baseline (≈ln(N)/N, N≈71) |
|---|---|---|
| MRR | **0.377** | ≈0.060 |
| Recall@5 | **0.476** | ≈0.070 |

**≈6x above chance on MRR.** The correct "own later snapshot" is
typically found around rank 2–3 out of ~71 real, unrelated candidates.

### Task B — Same-attacker retrieval (fold-safe, 68 campaigns with a peer)

| Metric | Value |
|---|---|
| MRR | **0.975** |
| Recall@5 | 0.231 (capped low because the two dominant attacker groups have 33/26 members — recall@5 cannot exceed 5/33 or 5/26 for those queries) |

### Task C — Same-host retrieval (fold-safe, 65 campaigns with a peer)

| Metric | Value |
|---|---|
| MRR | **0.979** |
| Recall@5 | 0.189 (same group-size ceiling effect) |

### Task B, weaker check — Same-attacker retrieval (single-model, identity-preservation only)

| Metric | Value |
|---|---|
| MRR | 0.822 |

**Notably lower than the fold-safe result (0.975)** — see Section 12
for why this is not read as "generalization beats memorization" in
general; it is more likely a symptom of the cross-fold embedding-space
comparability limitation (Section 3).

---

## 9. Comparison With Existing Representations

Representation 4 (`z_G`) vs. Representations 1–3, per Section 4's
Task D (the only task where all four are cleanly computable from
already-extracted real facts):

**Task D result — the central finding of this phase, from real
campaign pairs, not synthetic data**:

- **82 real pairs** found with **identical node-type shape** but
  **technique_Jaccard = 0.0** (completely disjoint or absent technique
  sets). Example: `CAMP_2EDE8EF9` vs. `CAMP_54643C88` —
  `technique_jaccard = 0.0`, **`embedding_cosine_distance = 0.0004`**
  (essentially identical embeddings despite zero technique overlap).
- **226 real pairs** found with **technique_Jaccard > 0.8** (near-identical
  technique sets) but **different node-type shape**. Example:
  `CAMP_71320238` vs. `CAMP_4CF07810` — `technique_jaccard = 1.0`,
  **`embedding_cosine_distance = 0.443`** (meaningfully different
  embeddings despite identical technique sets).

**Mechanistic explanation, not just correlation**: `graph_encoder.py`'s
node features (`NODE_TYPES` one-hot + `NUMERIC_PROPS`) contain **no
feature that encodes which specific ATT&CK technique a `Technique` node
represents** — only that a node *is* a Technique, plus unrelated
numeric properties. **The GNN structurally cannot see technique
identity at all.** This is not a learned preference for topology over
technique overlap — it is a direct, mechanical consequence of what
information the encoder was ever given. The result above is real and
useful evidence (the embedding does track structure, and does not
track technique identity), but it should not be read as "the model
learned that topology matters more" — it never had the option to do
otherwise. **This is disclosed prominently, not glossed over.**
*Planned:* extend `NODE_TYPES`/`NUMERIC_PROPS` with technique-identity
features in a future phase if the GNN is meant to jointly capture
topology and technique identity, rather than topology alone.

Representations 1 (technique-Jaccard) and 2 (`similar_to_formula_score`,
which is 60% Representation 1) would each treat the 82 same-shape/
zero-overlap pairs as maximally dissimilar (score 0) — exactly where
`z_G` says "nearly identical." This is the concrete, real-data form of
the capability gap `GNN_OBJECTIVE_DECISION.md` Section 7 predicted
technique-based mechanisms would have.

---

## 10. Temporal Consistency

Already established in `GNN_REPRESENTATION_IMPLEMENTATION.md` (20/21
campaigns, own-history closer than cross-campaign, qualitative). This
phase adds the formal retrieval-metric version (Section 8, Task A) —
**not** re-run via retraining (per this phase's explicit instruction);
the fold-safe LOGO embeddings computed fresh this phase for the
retrieval tasks were reused for this, since they are the
generalization-appropriate embedding set the prior phase's single
all-71-trained model was not.

---

## 11. Near-Duplicate Analysis (Previously Deferred Diagnostic)

Performed this phase, using the single-model embeddings (259
near-duplicate pairs, `cosine distance < 0.01`, as originally found):

| Correlate | Fraction of near-duplicate pairs |
|---|---|
| Same node-type shape | 35.5% |
| Same attacker | 53.7% |
| Same host | **67.6%** (highest single correlate) |
| Identical technique set | 41.3% |

**No single factor dominates or exclusively explains near-duplication**
— host-sharing is the strongest single correlate but still leaves
32.4% of near-duplicate pairs with *different* hosts. This is a mixed,
honest picture: near-duplicates are not purely trivial identity
memorization along any one dimension, though shared identity
(attacker/host) correlates more than shared technique set or shape
alone.

---

## 12. Limitations

- **Cross-fold embedding-space comparability** (Section 3): the 7
  fold-specific models share initialization/architecture but are
  independently optimized on different data — ranking embeddings
  pooled across them is not a strictly single, jointly-fit space. The
  strong, consistent Task A/B/C results are encouraging under this
  caveat, not proof it doesn't matter.
  *Planned:* investigate a post-hoc cross-fold alignment/calibration
  step, or a jointly-fit embedding space once enough real campaigns
  exist to support one without leaking test folds.
- **Recall@5's ceiling effect** (Section 8, Tasks B/C): low Recall@5
  values for large attacker/host groups are an artifact of group size
  exceeding k, not evidence of weak retrieval — MRR is the more
  informative metric for these two tasks.
- **Task D's mechanism is partly definitional** (Section 9): the GNN
  cannot see technique identity by construction, so "GNN ignores
  technique identity" is guaranteed, not purely discovered.
- Only 21 campaigns support Task A; only 68/65 support Tasks B/C
  (campaigns without a same-group peer are correctly excluded, not
  padded).
  *Planned:* the Phase 19 dataset expansion is expected to increase the
  number of campaigns available for Task A and similar retrieval
  evaluations.
- No comparison of Representations 1–3 against `z_G` was attempted for
  Tasks A/B/C (only Task D) — computing per-snapshot technique sets
  and 19-scalar features would require new plumbing not built this
  phase (disclosed, not silently skipped).
  *Planned:* build that plumbing in a future phase to extend the
  Representations 1–3 vs. `z_G` comparison to Tasks A/B/C.

## 13. What The Results Do NOT Prove

- They do not prove the GNN "understands" attack campaigns or attacker
  behavior — same-attacker/host retrieval success is substantially
  explained by shared numeric reputation features
  (`vt_reputation`/`threat_actor_reputation`) being literally identical
  across a shared attacker's campaigns, a mechanistically easy task,
  consistent with this document's own "low bar" framing throughout.
- They do not prove production retrieval/correlation would improve —
  no downstream consumer was touched; this is representation-level
  evidence only.
- Task D's finding does not prove the GNN's topology signal is
  *correct* or *security-meaningful* — only that it is *real,
  measurable, and different* from what technique-overlap-based
  mechanisms would say on the same real pairs.
- The temporal result (Task A) is the least mechanically-trivial and
  most novel finding, but 21 campaigns is a small evidentiary base for
  a general claim.

---

## 14. Production Integration Decision

**Not made here, per this phase's explicit instruction.** Evidence
summary for whoever makes it: the temporal retrieval result (Task A) is
the strongest, least-trivially-explained finding in this document and
plausibly promising; the attacker/host results, while numerically
strong, are largely explained by shared literal input features rather
than learned structure; the structural diagnostic (Task D) is real but
partly definitional. **Evidence level: promising, narrowly scoped
(temporal identity), and requiring further validation (particularly
resolving the cross-fold embedding-space limitation) before any
integration decision** — not insufficient, not sufficiently strong for
an unqualified yes.

---

## Tests

```bash
cd backend
python -m pytest tests/ -q
```

New: `tests/test_retrieval_evaluation.py` (17 tests covering the pure
metric/formula/diagnostic functions and the two DB-boundary-mocked
query helpers — the two expensive full-LOGO live functions are
exercised via the module's own `__main__` run, consistent with this
repository's convention of keeping the routine test suite fast). No
existing test was weakened. No production file was modified this
phase (verified via `git status`: only new files added).
