"""soc_engine/__init__.py — SOC Anomaly Detection Engine package.

This package is a genuine, mixed-use Generation-2 dependency: `model.py`
(Autoencoder), `trainer.py` (AutoencoderTrainer), `scorer.py`
(SeverityScorer), and `threshold.py` (AdaptiveThreshold) are real,
currently-imported building blocks of the live SSL/SSFT training and
runtime path (ml/ssl_pipeline.py, ml/ssft.py — see
../GENERATION1_DISPOSITION.md for the full evidence). `temporal_engine.py`
has a real consumer too, just not a Generation-2 one — the capstone
demo notebooks (notebooks/capstone_demo*.ipynb) import
`apply_temporal_smoothing` from it directly.

`explainer.py` has no consumer anywhere in this repository (production
code, tests, or notebooks) other than the disconnected Generation-1
`agents/soc_agent.py` orchestrator — see ../GENERATION1_DISPOSITION.md.
It is deliberately NOT re-exported here: package `__init__.py` imports
are eager, so `from soc_engine.model import Autoencoder` (what
ml/ssl_pipeline.py and ml/ssft.py actually do) previously also
transitively imported explainer.py as an unnecessary side effect —
Generation-2's own live training/runtime path accidentally pulling in
Generation-1-only code purely because of this file's re-export list,
not because anything needed it. `temporal_engine.py` is excluded from
this package-level re-export for the same reason (nothing accesses it
via `soc_engine.TemporalSmoother`; the demo notebooks already import it
via `soc_engine.temporal_engine.apply_temporal_smoothing`, a direct
submodule import unaffected by this list). Neither file was deleted —
both remain fully importable via their own submodule path exactly as
before; only the accidental package-level eager import was removed.
"""
from .model import Autoencoder, load_autoencoder
from .scorer import SeverityScorer, compute_scores
from .threshold import AdaptiveThreshold
from .trainer import AutoencoderTrainer

__all__ = [
    "Autoencoder", "load_autoencoder",
    "SeverityScorer", "compute_scores",
    "AdaptiveThreshold",
    "AutoencoderTrainer",
]
