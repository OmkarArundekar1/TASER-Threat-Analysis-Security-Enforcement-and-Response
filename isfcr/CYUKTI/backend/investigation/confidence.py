"""
investigation/confidence.py
==============================
Separates the concepts the loop was conflating (found by running it
against real campaigns — every run stopped after 1-2 steps because a
single reliable-but-narrow fact was read as "investigation resolved"):

    evidence_reliability — mean trust in the evidence GATHERED so far
        (a source-level property: "is this observation true").
    evidence_coverage    — how much of the *available evidence-source
        space* has actually been consulted, weighted by how relevant
        what was found there was. A single maximally-relevant,
        maximally-reliable fact from ONE source yields low coverage
        (1 of 6 possible sources) even though evidence_reliability
        alone would read as 1.0 — this is precisely what stopped the
        saturation bug from recurring in a different form.
    model_probabilities / model_confidence / model_uncertainty — about
        the actual investigative CONCLUSION (e.g. predicted severity),
        from the real trained XGBoost model, not from any Evidence
        item's reliability.
    investigation_confidence — the single score the stopping policy
        acts on. Deliberately NOT a fixed-weight blend of the above:
          - if a model verdict exists: driven primarily by the model
            (model_confidence * (1 - model_uncertainty)), because that
            IS confidence in the conclusion; evidence_coverage acts as
            a gate (see note in estimate_confidence) so the loop can't
            declare victory on a model verdict before consulting a
            reasonable breadth of supporting evidence.
          - if no model verdict exists yet: evidence_reliability *
            evidence_coverage — reliability alone can no longer
            saturate confidence; coverage must also be non-trivial.
    uncertainty — 1 - investigation_confidence, surfaced explicitly so
        callers don't have to invert it themselves; model_uncertainty
        (entropy) remains available separately since it answers a
        narrower, model-specific question.

Predictive entropy (Shannon, normalized to [0, 1] by dividing by
log(num_classes)) is used for model_uncertainty specifically because
top-1 probability alone can't distinguish a concentrated distribution
from a spread one at the same top-class value (see docstring on
predictive_entropy for the worked example) — this is the standard,
well-understood measure, not a bespoke one invented for this project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from evidence.schema import EvidenceSource
from evidence.store import EvidenceStore

CONFLICT_PENALTY = 0.85  # multiplicative penalty per investigation when conflicts are unresolved
MIN_COVERAGE_FOR_MODEL_TRUST = 0.15  # see note in estimate_confidence


def predictive_entropy(probabilities: dict[str, float]) -> float:
    """Normalized Shannon entropy over a class-probability distribution, in [0, 1].

    0.0 = fully certain (all mass on one class); 1.0 = maximally
    uncertain (uniform over all classes). Two distributions can share
    the same top-1 probability (e.g. [0.5, 0.4, 0.1] vs [0.5, 0.25,
    0.25]) while differing in how uncertain the remainder is — entropy
    captures that, top-1 probability alone does not.
    """
    ps = [p for p in probabilities.values() if p > 0]
    if len(ps) <= 1:
        return 0.0

    entropy = -sum(p * math.log(p) for p in ps)
    max_entropy = math.log(len(probabilities))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def evidence_coverage(store: EvidenceStore) -> float:
    """Fraction of the possible evidence-source space actually consulted,
    weighted by the best relevance found from each source.

    Deliberately independent of evidence_reliability: a single
    maximally-relevant, maximally-reliable fact from ONE of the six
    EvidenceSource categories should NOT read as a broadly-resolved
    investigation — it's one data point, not corroboration.
    """
    all_sources = list(EvidenceSource)
    if not all_sources:
        return 0.0

    best_relevance_per_source: dict[EvidenceSource, float] = {}
    for e in store.all():
        best_relevance_per_source[e.source] = max(best_relevance_per_source.get(e.source, 0.0), e.relevance)

    return sum(best_relevance_per_source.values()) / len(all_sources)


@dataclass
class ConfidenceEstimate:
    # evidence-level (source reliability / breadth — NOT conclusion confidence)
    evidence_reliability: float
    evidence_coverage: float
    conflict_count: int

    # model-level (about the actual investigative conclusion)
    model_probabilities: dict[str, float] | None = None
    model_confidence: float | None = None
    model_uncertainty: float | None = None

    # investigation-level (what the stopping policy acts on)
    investigation_confidence: float = 0.0
    uncertainty: float = 1.0


def estimate_confidence(
    store: EvidenceStore,
    model_probabilities: dict[str, float] | None = None,
) -> ConfidenceEstimate:
    reliability = store.weighted_confidence()
    coverage = evidence_coverage(store)
    conflicts = store.detect_conflicts()

    if model_probabilities:
        model_confidence = max(model_probabilities.values())
        model_uncertainty = predictive_entropy(model_probabilities)
        # The model's own verdict-confidence, adjusted for how spread
        # out the rest of the distribution is. Gated by evidence_coverage
        # below MIN_COVERAGE_FOR_MODEL_TRUST: a model run against almost
        # no supporting context (e.g. before any MITRE/CTI/graph lookup)
        # shouldn't be trusted at full strength — this is a floor, not a
        # blend, so a genuinely confident model with real supporting
        # coverage is never artificially suppressed.
        model_term = model_confidence * (1.0 - model_uncertainty)
        coverage_gate = min(1.0, coverage / MIN_COVERAGE_FOR_MODEL_TRUST) if MIN_COVERAGE_FOR_MODEL_TRUST > 0 else 1.0
        investigation_confidence = model_term * coverage_gate
    else:
        model_confidence = None
        model_uncertainty = None
        # No model verdict yet: reliability alone must not be able to
        # saturate confidence — coverage must also be non-trivial, so a
        # single relevant-and-reliable fact from one source reads as
        # partial, not resolved.
        investigation_confidence = reliability * coverage

    if conflicts:
        investigation_confidence *= CONFLICT_PENALTY

    investigation_confidence = round(max(0.0, min(1.0, investigation_confidence)), 4)

    return ConfidenceEstimate(
        evidence_reliability=round(reliability, 4),
        evidence_coverage=round(coverage, 4),
        conflict_count=len(conflicts),
        model_probabilities={k: round(v, 4) for k, v in model_probabilities.items()} if model_probabilities else None,
        model_confidence=round(model_confidence, 4) if model_confidence is not None else None,
        model_uncertainty=round(model_uncertainty, 4) if model_uncertainty is not None else None,
        investigation_confidence=investigation_confidence,
        uncertainty=round(1.0 - investigation_confidence, 4),
    )
