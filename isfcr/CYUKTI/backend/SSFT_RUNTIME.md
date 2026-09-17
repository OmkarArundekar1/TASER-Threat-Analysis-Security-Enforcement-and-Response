# SSFT Runtime Integration

This documents the actual, code-verified status of Self-Supervised
Feature Transformation (`ml/ssft.py`) after this session's inspection
and implementation — correcting and sharpening a prior session's
architecture audit, which characterized the remaining SSFT gap as pure
feature-schema wiring. It is not. **No accuracy/quality claim is made
anywhere in this document** — this is an engineering integration
report.

## The naming collision (preserved, not resolved)

"SSFT" means two unrelated things in this codebase, from two different
architectural generations:

1. **Self-Supervised Feature Transformation** (`ml/ssft.py`) — the
   subject of this document. Unsupervised: transforms raw network-flow
   window features into a compact learned latent representation via
   the SSL autoencoder, no labels involved anywhere.
2. **Semi-Supervised Fine-Tuning** (`agents/soc_agent.py`'s
   `export_ssft_dataset()`) — part of the disconnected Generation-1
   `SOCAgent` orchestrator (see `../ARCHITECTURE_AUDIT.md`). Pseudo-labels
   data for later supervised fine-tuning. Not touched, not relevant to
   this mission, explicitly out of scope.

## The SSFT contract (as implemented, verified by reading the code and the real trained artifacts — not assumed)

```
Raw CICIDS2017 flow window (100 rows x 78 columns of raw flow stats)
        |  ml/data_prep/sliding_window_preprocess.py (mean/std/max/min per column)
        v
312-dim raw window feature vector
        |  StandardScaler (ml/models/ssl_scaler.pkl, fit on real CICIDS2017 data)
        v
312-dim scaled vector
        |  Autoencoder.encode() (soc_engine/model.py, trained by
        |  ml/ssl_pipeline.py on real data, pure reconstruction loss)
        v
32-dim latent vector  +  scalar reconstruction error
```

Verified directly against the real, currently-committed production
artifact (not assumed from a docstring):

```python
>>> import joblib
>>> joblib.load('ml/models/ssl_scaler.pkl').n_features_in_
312
```

## Was SSFT already integrated? — No, and the reason is more specific than "not wired up"

**Root cause, proven from code, not inferred:** SSFT's input domain
(312-dim CICIDS2017 network-flow *window* statistics) and CYUKTI's
live production severity model's input domain (57-dim *campaign-graph*
features — `ml/feature_schema.py`: `campaign_size`, `graph_density`,
`wazuh_level`, ...) are **different feature spaces with no shared join
key in this repository**:

```python
>>> from ml.dataset_utils import FEATURE_COLUMNS
>>> len(FEATURE_COLUMNS)
57
```

A live Neo4j `Campaign` has no associated "network-flow window" —
`ml/data_prep/sliding_window_preprocess.py`'s only real input is the
offline, historical CICIDS2017 research CSVs in `datasets/` (2017
academic-testbed IPs), entirely unrelated to the live Wazuh-derived
attacker/victim IPs in the production graph.
`detection_confidence_engine.py` (which DOES read a "Zeek"/"Suricata"
score per real `AttackEvent`) reads **pre-stored scalar properties**
already written onto the Neo4j node — not raw flow feature vectors —
confirmed by reading `neo4j_client.get_detection_confidence()`'s
return shape; there is no live packet/flow-capture pipeline in this
repository feeding either the SSL encoder or a per-campaign window.

**Forcing the two together would mean one of:**
- **Fabrication** — running the 57 campaign-graph numbers through a
  scaler/encoder fit on 312 unrelated flow-statistics columns. This
  would execute without error and produce a 32-dim vector that means
  nothing — exactly the "meaningless SSL/SSFT wrapper" this project's
  own engineering discipline exists to avoid. Not built.
- **New infrastructure** — a live network-flow/packet-capture pipeline
  producing a real window per campaign. Does not exist; building one is
  a genuine infrastructure project, out of this session's scope.
- **SSL retraining** on the campaign-graph feature space instead of
  CICIDS2017. Explicitly out of scope for this session.

**This is Outcome C** (a genuine architectural incompatibility,
proven from code) **for the specific integration the mission
described** ("SSL → SSFT → RuntimePredictor → XGBoost → Investigation").
No workaround was fabricated to force a green checkmark on that box.

## What genuinely was disconnected, and what this session fixed

Within SSFT's own correct domain (network-flow windows), a real gap
existed: `ml/ssft.py` had **only a batch/offline entry point**
(`transform_to_ssft_dataset()`, reading an entire pre-built
`window_features.npy` file and writing one combined `.npz`) — no
single-instance **runtime** inference path existed, unlike
`ml.train_xgboost.XGBoostCampaignClassifier`, which already has exactly
this distinction via `ml.runtime_predictor.RuntimeCampaignPredictor`
(load once, predict cheaply per call). That asymmetry — a real gap,
not a misunderstanding — is what this session closed.

**`SSFTRuntimeTransformer`** (`ml/ssft.py`, new this session): loads
the trained scaler + encoder once, then `.transform(window)` accepts
either a single window (shape `(312,)`, returns a plain `dict` with a
`(32,)` latent array and a `float` reconstruction error) or a batch
(shape `(N, 312)`, returns arrays — the exact shape
`transform_to_ssft_dataset` already wrote to `ssft_dataset.npz`).
`transform_to_ssft_dataset()` was refactored to call this same class
internally, rather than duplicating the scaler/encoder logic — **there
is exactly one transform code path**, used by both the batch and the
new single-instance entry point. No behavior change to
`transform_to_ssft_dataset()`'s existing output, parameters, or error
messages (`test_ssft_transform_requires_trained_model` /
`test_ssft_transform_round_trip`, unmodified, still pass).

## Train/runtime parity — verified directly, not assumed

`test_transform_to_ssft_dataset_and_runtime_transformer_agree_on_the_same_data`
runs the same 20 synthetic windows through both entry points and
asserts bit-identical `latent`/`reconstruction_error` arrays.
`test_runtime_transformer_batch_call_matches_single_window_calls`
confirms calling the class once per window (genuine single-instance
use) agrees with calling it once for the whole batch.
`test_runtime_transformer_is_deterministic` confirms no randomness
(dropout, etc.) leaks into inference — repeated calls on the identical
window are bit-identical.

## Model compatibility

The production XGBoost model (`ml/models/xgb_severity.json`) was **not
touched, not retrained, not regenerated** this session — it was never
compatible with SSFT's output in the first place (57-dim campaign
features vs. 32-dim window-latent features are not interchangeable or
concatenable without redefining what a "sample" even is for XGBoost —
today one row is one *campaign*; an SSFT-derived feature would need to
be one row per *window*, a different unit of analysis entirely).
Classified per the mission's own framework: making XGBoost consume
SSFT-adjacent features would require **a research decision** (what
does "a campaign's window" even mean, given no live flow-capture
exists) before any migration engineering could start — not an
engineering task this session could complete honestly.

## Known environment limitation (unrelated to this change, disclosed for completeness)

`tests/test_ssl_pipeline.py` fails when run in isolation
(`python -m pytest tests/test_ssl_pipeline.py`) in this session's
Windows environment — `torch`'s DLL loader
(`OSError: ... c10.dll ...`) fails to initialize when this is the
first/only test file collected, but succeeds when run as part of the
full suite. Verified via `git stash` that this reproduces identically
against the **unmodified, pre-session** `ml/ssft.py` — it is a
pre-existing Windows torch-DLL-loading order quirk, not caused by this
session's changes. The authoritative regression check
(`cd backend && python -m pytest tests/ -q`) is unaffected and passes
cleanly; always use the full-suite command, not an isolated single-file
run, to check this specific file's tests in this environment.

## Tests

```bash
cd backend
python -m pytest tests/ -q   # full suite — 352 passed (345 prior baseline + 7 new)
```

New tests (all in `tests/test_ssl_pipeline.py`, alongside the existing
SSL/SSFT mechanics tests, unmodified): `test_runtime_transformer_requires_a_trained_model`,
`test_runtime_transformer_single_window_shape_and_type`,
`test_runtime_transformer_rejects_wrong_dimensional_input`,
`test_runtime_transformer_handles_nan_and_inf_input_without_crashing`,
`test_runtime_transformer_is_deterministic`,
`test_runtime_transformer_batch_call_matches_single_window_calls`,
`test_transform_to_ssft_dataset_and_runtime_transformer_agree_on_the_same_data`.
