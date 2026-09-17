# Generation-1 Architecture Disposition

A file-level audit of every component under `backend/agents/`,
`backend/soc_engine/`, `backend/classifiers/`, `backend/response/`, and
`backend/kill_chain/` — the directories `ARCHITECTURE_AUDIT.md`
previously classified in bulk as "Generation 1, superseded." This audit
re-derives that classification per file, with the actual search basis
recorded (not "no imports found" asserted without evidence), splits
files that were incorrectly grouped together (some are live Generation-2
dependencies; some are dead; one is neither — it's demo/notebook code),
and fixes the one real architectural cross-contamination this
grouping created. **No files were deleted.** Companion document to
`ARCHITECTURE_AUDIT.md`; read that first for the overall system picture.

## 1. Executive Summary

"Generation 1" is not one thing. It is three:

1. **A handful of raw ML building blocks** (`soc_engine/model.py`,
   `trainer.py`, `scorer.py`'s `SeverityScorer` class, `threshold.py`)
   that Generation 2 actually imports and runs today, in production
   training and inference code. These are not legacy — they are live.
2. **One file used only by the capstone demo notebooks**
   (`soc_engine/temporal_engine.py`) — not Generation-2 production
   code, but not dead either.
3. **A complete, self-contained, disconnected orchestrator**
   (`agents/soc_agent.py` and everything it alone pulls in:
   `soc_engine/explainer.py`, all of `classifiers/`, `response/`,
   `kill_chain/`) with zero consumers anywhere else in the repository.
   This is genuinely dead code — proven below, file by file — but is
   **not deleted** in this audit; see Section 8.

The only code change made this session was removing two eager
re-exports from `soc_engine/__init__.py` (`FeatureExplainer`,
`TemporalSmoother`) that caused Generation-2's real import
(`from soc_engine.model import Autoencoder`) to transitively load
dead/demo-only code as a side effect. Both files remain intact and
importable by their existing consumers via their direct submodule
path. Regression suite: 356/356 passing (352 baseline + 4 new tests
added this session), zero regressions.

## 2. Generation-1 Inventory

| Component | Classification | Runtime (Gen-2) Consumer | Test Consumer | Other Consumer | Action |
|---|---|---|---|---|---|
| `soc_engine/model.py` (`Autoencoder`, `get_device`) | **B** reusable primitive | `ml/ssl_pipeline.py`, `ml/ssft.py` | `tests/test_ssl_pipeline.py` | `agents/soc_agent.py` (dead) | Kept, unchanged |
| `soc_engine/model.py`'s `load_autoencoder` function | **C** dead | None | None | None | Kept (file-level, not deletable independently of the class above) |
| `soc_engine/trainer.py` (`AutoencoderTrainer`) | **B** reusable primitive | `ml/ssl_pipeline.py` | `tests/test_ssl_pipeline.py` | `agents/soc_agent.py` (dead) | Kept, unchanged |
| `soc_engine/scorer.py`'s `SeverityScorer` class | **B** reusable primitive | `ml/ssl_pipeline.py` | `tests/test_ssl_pipeline.py` | `agents/soc_agent.py` (dead) | Kept, unchanged |
| `soc_engine/scorer.py`'s `compute_scores` function | **C** dead | None | None | None | Kept (same file as the class above) |
| `soc_engine/threshold.py` (`AdaptiveThreshold`) | **B** reusable primitive | `ml/ssl_pipeline.py` | `tests/test_ssl_pipeline.py` | `agents/soc_agent.py` (dead) | Kept, unchanged |
| `soc_engine/temporal_engine.py` (`TemporalSmoother`, `apply_temporal_smoothing`) | **D** documentation/example artifact | None | None | `notebooks/capstone_demo.ipynb`, `notebooks/capstone_demo_phase2.ipynb` | Kept, unchanged; **no longer eagerly re-exported at package level** (see Section 5) |
| `soc_engine/explainer.py` (`FeatureExplainer`) | **C** dead | None | None | None (checked notebooks too — zero references) | Kept; **no longer eagerly re-exported at package level** (see Section 5) |
| `soc_engine/__init__.py` | package boundary | (see Section 5) | `tests/test_soc_engine_package_boundary.py` (new) | — | **Edited this session** |
| `agents/soc_agent.py` (`SOCAgent`) | **C** dead | None | None | None | Kept, undeleted |
| `agents/output_schema.py` (`ThreatIntelReport`, `AnomalyResult`, `ClassificationResult`) | **C** dead | None | None | `agents/soc_agent.py` (dead) only | Kept, undeleted |
| `agents/__init__.py` | package boundary, no cross-contamination (see Section 5) | — | — | — | Unchanged |
| `classifiers/lightgbm_classifier.py` (`AttackClassifier`) | **C** dead | None | None | `agents/soc_agent.py` (dead), gated behind an unused flag | Kept, undeleted |
| `classifiers/attack_stage_mapper.py` (`AttackStageMapper`) | **C** dead | None | None | `agents/soc_agent.py` (dead), `kill_chain/kill_chain_detector.py` (dead) | Kept, undeleted |
| `response/playbook_generator.py` (`PlaybookGenerator`) | **C** dead | None | None | `agents/soc_agent.py` (dead) only | Kept, undeleted |
| `response/mitigator.py` (`Mitigator`) | **C** dead | None | None | **None at all** — not even referenced by `soc_agent.py`, only named in a docstring | Kept, undeleted |
| `kill_chain/event_sequence_buffer.py` (`EventBuffer`) | **C** dead | None | None | `kill_chain/kill_chain_detector.py` (dead) only | Kept, undeleted |
| `kill_chain/attack_graph.py` (`AttackGraphBuilder`) | **C** dead | None | None | `kill_chain/kill_chain_detector.py` (dead) only | Kept, undeleted |
| `kill_chain/kill_chain_detector.py` (`KillChainDetector`) | **C** dead | None | None | None outside `kill_chain/` itself | Kept, undeleted |
| `kill_chain/__init__.py` | package boundary, no cross-contamination | — | — | — | Unchanged |

**Search basis for every "None" cell above** (per the mission's
requirement to not merely assert "no imports found"): grepped the
exact class/function names and their module paths (`soc_agent`,
`SOCAgent`, `soc_engine.explainer`, `FeatureExplainer`,
`classifiers.`, `AttackClassifier`, `AttackStageMapper`, `response.`,
`PlaybookGenerator`, `Mitigator`, `kill_chain.`, `KillChainDetector`,
`AttackGraphBuilder`, `EventBuffer`) across: every `backend/*.py` file
(including the three real production entry points —
`dashboard_api.py`, `realtime_socgraph.py`, `listener/wazuh_listener.py`),
every file in `backend/tests/`, every file in `backend/scripts/`
(none exist referencing these), every `.sh`/`.yml`/`.yaml`/Dockerfile/
systemd unit in the repository, every notebook in `notebooks/`, and
`requirements.txt`/`config.py` for any config-driven feature-flag path
that could reach `soc_agent.py` conditionally. One false-positive match
was found and excluded after inspection:
`realtime_socgraph.py`'s matches for `response\.` are a local
`requests.Response` object variable literally named `response`, not
the `response/` package.

## 3. Shared Primitives (must not be deleted)

`soc_engine/model.py` (`Autoencoder` class), `soc_engine/trainer.py`
(`AutoencoderTrainer`), `soc_engine/scorer.py` (`SeverityScorer`
class), `soc_engine/threshold.py` (`AdaptiveThreshold`) are real,
currently-imported dependencies of `ml/ssl_pipeline.py` and
`ml/ssft.py` — the code that trains and runs CYUKTI's real SSL
autoencoder on real CICIDS2017 data and produces the artifacts in
`ml/models/` (`autoencoder_best.pth`, `ssl_scaler.pkl`). Any future
cleanup of this directory must keep these four files, and must keep
`soc_engine/__init__.py` re-exporting `Autoencoder`, `load_autoencoder`,
`SeverityScorer`, `compute_scores`, `AdaptiveThreshold`,
`AutoencoderTrainer` exactly as it does after this session's edit —
`ml/ssft.py`'s `SSFTRuntimeTransformer.__init__` does
`from soc_engine.model import Autoencoder` directly against the
submodule, so even the package-level re-export of these four is not
strictly required by that one call site, but is left in place as it
introduces no cross-contamination (none of these four modules import
`explainer.py` or `temporal_engine.py`) and other, currently-hypothetical
callers may reasonably use the package-level path.

## 4. Cross-Generation Dependencies (before this session's fix)

`ml/ssl_pipeline.py` and `ml/ssft.py` (both Generation 2) import
`from soc_engine.model import Autoencoder`. Python package imports are
eager: importing any submodule of a package first executes that
package's `__init__.py` in full. Before this session,
`soc_engine/__init__.py` unconditionally executed
`from .explainer import FeatureExplainer` and
`from .temporal_engine import TemporalSmoother` as part of that
`__init__.py`, regardless of which submodule the caller actually
wanted. The practical effect: every real Generation-2 training/runtime
invocation of the SSL pipeline transitively loaded `explainer.py` (a
module with zero consumers anywhere in the repository) and
`temporal_engine.py` (a module used only by the demo notebooks) into
memory as an unrequested side effect — accidental coupling introduced
purely by the package's `__init__.py` re-export list, not by any
actual need in the SSL/SSFT code path.

This was the one instance of Generation-2 → Generation-1 architectural
cross-contamination found in this audit. No other package boundary
(`agents/__init__.py`, `classifiers/__init__.py`, `response/__init__.py`,
`kill_chain/__init__.py`) is imported by any Generation-2 code at all,
so none of them can cause this kind of side effect regardless of what
they re-export.

### Remediation applied (per the 4-step process this audit required)

1. **Is it necessary?** No — `ml/ssl_pipeline.py`/`ml/ssft.py` only
   ever reference `Autoencoder`/`load_autoencoder`/`SeverityScorer`/
   `AdaptiveThreshold`/`AutoencoderTrainer`, never `FeatureExplainer`
   or `TemporalSmoother`.
2. **Reusable primitive or accidental coupling?** Accidental — nothing
   in Generation 2 accesses `soc_engine.FeatureExplainer` or
   `soc_engine.TemporalSmoother` via the package-level path; the two
   real consumers of these classes (`agents/soc_agent.py` and the demo
   notebooks) already use direct submodule imports
   (`from soc_engine.temporal_engine import TemporalSmoother` /
   `apply_temporal_smoothing`), which do not depend on `__init__.py`
   re-exporting them.
3. **Safely decoupled**: `soc_engine/__init__.py` no longer imports
   `explainer` or `temporal_engine`. Neither file was deleted, moved,
   or edited — only the package-level re-export lines and their
   `__all__` entries were removed. `agents/soc_agent.py` and the
   notebooks are unaffected because they never used the package-level
   path in the first place.
4. **Regression test added**: `backend/tests/test_soc_engine_package_boundary.py`
   — asserts `soc_engine.FeatureExplainer`/`soc_engine.TemporalSmoother`
   no longer exist at package level, asserts the four real primitives
   are still re-exported, asserts both `explainer.py`/`temporal_engine.py`
   remain directly importable via their submodule path, and (the actual
   proof, run in a fresh subprocess) asserts that importing
   `soc_engine.model` — the exact statement `ml/ssl_pipeline.py`/
   `ml/ssft.py` execute — no longer causes
   `soc_engine.explainer`/`soc_engine.temporal_engine` to appear in
   `sys.modules`.

## 5. Removed Components

None. No file was deleted this session. See Section 8 for why.

## 6. Retained Historical Components

Every file listed as Category C or D in Section 2 remains in place,
unedited (with the sole exception of `soc_engine/__init__.py`'s
re-export list, Section 4). Rationale for retaining rather than
deleting, per component group:

- **`agents/`, `classifiers/`, `response/`, `kill_chain/` (Category
  C, ~1,660 lines total)**: proven dead by exhaustive search (Section
  2), but this audit's own operating instructions require deletion to
  be a recorded human decision, not an autonomous action, because the
  code is competent and complete (not broken/abandoned mid-write) and
  could plausibly represent intended-but-deferred future work (e.g.
  `soc_agent.py`'s `export_ssft_dataset()` toward real-time supervised
  fine-tuning) rather than pure abandonment. See Section 8.
- **`soc_engine/explainer.py` (Category C)**: same reasoning — dead,
  but small, self-contained, and low-cost to leave in place pending
  the same human decision as the rest of Category C.
- **`soc_engine/temporal_engine.py` (Category D)**: **must** be
  retained regardless of the Category-C decision above — it is a real,
  current dependency of the capstone demo notebooks
  (`notebooks/capstone_demo.ipynb`, `notebooks/capstone_demo_phase2.ipynb`),
  which are very likely used in the capstone's own presentation/demo
  material. Deleting it would break a deliverable outside this
  session's authority to assess.

## 7. Architectural Boundary

```
Generation 2 (live, production)
  ml/ssl_pipeline.py, ml/ssft.py
        |
        v  from soc_engine.model import Autoencoder   <-- the only Gen-2 -> soc_engine edge
        |
  soc_engine/__init__.py  (re-exports: Autoencoder, load_autoencoder,
        |                   SeverityScorer, compute_scores,
        |                   AdaptiveThreshold, AutoencoderTrainer)
        |
        +-- model.py, trainer.py, scorer.py, threshold.py      [Category B -- shared primitive]
        |
        +-- (no longer re-exported here) --------------------- [boundary enforced by this session's fix]
            |
            +-- explainer.py        [Category C -- dead, reachable only via
            |                        direct submodule import from agents/soc_agent.py, also dead]
            +-- temporal_engine.py  [Category D -- reachable only via direct
                                     submodule import from the demo notebooks]

Generation 1 (historical, disconnected, zero Gen-2 consumers)
  agents/soc_agent.py
        +-- classifiers/lightgbm_classifier.py, attack_stage_mapper.py
        +-- response/playbook_generator.py, mitigator.py
        +-- soc_engine/explainer.py (direct submodule import)
        +-- agents/output_schema.py

  kill_chain/kill_chain_detector.py
        +-- kill_chain/event_sequence_buffer.py, attack_graph.py
        +-- classifiers/attack_stage_mapper.py
  (kill_chain has zero consumers of its own, including from soc_agent.py)
```

The boundary is now exactly what it should be: Generation 2 only ever
touches the four Category-B files, and only via `soc_engine.model`
(and, incidentally, the package-level re-export of the other three —
harmless, since none of those three import anything Category-C/D
themselves). Nothing in Generation 2 can any longer transitively load
Category-C or Category-D code merely by importing its real
dependencies.

## 8. Remaining Decision Points

**The one genuine, unresolved human/product decision from this
audit**: whether to eventually **delete** or **formally archive**
(e.g. move to a `legacy/` directory with a README) the ~1,660 lines of
now-fully-proven-dead Category-C code (`agents/`, `classifiers/`,
`response/`, `kill_chain/`, `soc_engine/explainer.py`). This audit
did not make that decision, per its own operating instructions:

- **For deletion**: every technical measure supports it — zero
  production imports, zero test imports, zero executable entry points,
  zero service/config/CI references, zero notebook references, a
  `config.yaml` dependency (`SOCAgent.from_config()`) that doesn't
  exist anywhere in the repository.
- **Against immediate deletion**: the code is competent and complete,
  not broken or partially written, which is more consistent with "an
  earlier design phase that got superseded" than "abandoned mid-task."
  `soc_agent.py`'s `export_ssft_dataset()` specifically anticipates a
  real-time supervised fine-tuning capability that Generation 2 has
  not built and may still want — deleting it destroys that specific
  historical design reference, not just inert code.

No action was taken on this decision beyond documenting it here and in
`ARCHITECTURE_AUDIT.md`. `soc_engine/temporal_engine.py` is excluded
from this decision entirely — it has a real, current consumer (the
demo notebooks) and must be retained regardless of what happens to the
rest of Category C.

## 9. Tests

```bash
cd backend
python -m pytest tests/ -q   # 356 passed (352 baseline + 4 new), zero regressions
```

New tests, all in `tests/test_soc_engine_package_boundary.py`:
`test_soc_engine_package_does_not_reexport_generation1_only_helpers`,
`test_soc_engine_reusable_generation2_primitives_still_reexported`,
`test_explainer_and_temporal_engine_remain_directly_importable`,
`test_generation2_import_path_no_longer_pulls_in_generation1_only_modules`
(the actual proof of decoupling — run in a fresh subprocess).

## 10. Runtime Verification

Confirmed all real Generation-2 entry points still import cleanly
after the `soc_engine/__init__.py` change, without any external
credentials: `dashboard_api`, `neo4j_client`, `investigation.loop`,
`investigation.actions`, `investigation.next_best_evidence`,
`ml.runtime_predictor`, `ml.train_xgboost`, `ml.ssft`,
`campaign_manager`, `mitre_resolver`, `rag.retriever`,
`rag.mitre_retriever`, `rag.campaign_retriever` — all `OK`.

`listener.wazuh_listener` fails to import when imported as a
dotted module path from `backend/` (`cannot import name
'generate_alert_id' from 'utils'`) — verified via `git stash` that
this reproduces identically on the unmodified, pre-session tree. Root
cause: `wazuh_listener.py` does `from utils import generate_alert_id`
before adding `BACKEND_DIR` to `sys.path`, relying on its own
directory (`listener/`, which has its own `utils.py`) being first on
`sys.path` — true when run as a script (`python wazuh_listener.py`
from inside `listener/`) or under the test suite's own path setup, but
not when imported as `listener.wazuh_listener` from `backend/`, where
`utils` instead resolves to the unrelated top-level `backend/utils/`
package. Pre-existing, unrelated to this session's change, not fixed
here (out of scope — this session touches only `soc_engine/__init__.py`).
The full regression suite (`tests/`, run via its normal `conftest.py`
path setup) already covers `wazuh_listener.py` and passes.
