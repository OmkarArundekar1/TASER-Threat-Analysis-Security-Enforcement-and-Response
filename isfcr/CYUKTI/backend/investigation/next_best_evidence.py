"""
investigation/next_best_evidence.py
=======================================
Scores each not-yet-exhausted InvestigationAction by an explicit,
explainable value formula implementing the detection-paradox trade-off
CYUKTI's design calls for:

    value = expected_gain + reliability_term + novelty_term
            + uncertainty_reduction_term
            - cost_term - latency_term

This is a documented HEURISTIC policy, not a learned/optimal one — there
is no historical log of "which evidence-gathering order actually
resolved investigations fastest" to train a policy from yet. It is
deliberately transparent (every term traceable to either ActionMeta's
declared numbers or the current ConfidenceEstimate) so it can be
replaced or calibrated once real investigation outcomes accumulate,
rather than a black-box score.

uncertainty_reduction_term is new: only InvestigationAction.XGBOOST_PREDICTION
carries it, because it is the only action in the menu that produces a
verdict about the actual investigative conclusion (predicted severity)
rather than a supporting fact — a real, non-arbitrary distinction, not
tuning to hit a target step count. Its size scales with how much
uncertainty currently exists: once a model has run and uncertainty is
low, running it again earns no further credit (matching the existing
novelty discount for repeat queries elsewhere in this file).
"""

from __future__ import annotations

from dataclasses import dataclass

from evidence.store import EvidenceStore
from investigation.actions import ACTION_METADATA, InvestigationAction

# Weights are intentionally simple integers/halves, not fitted
# coefficients — there is no labeled outcome data to fit them against.
W_GAIN = 1.0
W_RELIABILITY = 0.3
W_NOVELTY = 0.5
W_UNCERTAINTY_REDUCTION = 0.6
W_COST = 0.4
W_LATENCY = 0.2

REPEAT_QUERY_NOVELTY = 0.15  # re-querying an already-covered source can still update it, just less valuable

# The only action producing a verdict about the investigative conclusion
# itself, rather than a supporting fact about the campaign/attacker/technique.
CONCLUSION_ACTIONS = frozenset({InvestigationAction.XGBOOST_PREDICTION})


@dataclass
class ActionValue:
    action: InvestigationAction
    expected_gain: float
    reliability: float
    novelty: float
    uncertainty_reduction: float
    cost: float
    latency: float
    value: float


def score_action(
    action: InvestigationAction,
    store: EvidenceStore,
    current_uncertainty: float = 1.0,
) -> ActionValue:
    """current_uncertainty: the investigation's current `uncertainty`
    (from ConfidenceEstimate), in [0, 1]. Defaults to 1.0 (fully
    uncertain) when no estimate is available yet — e.g. the very first
    action of an investigation, before any confidence has been computed.
    """
    meta = ACTION_METADATA[action]
    already_covered = len(store.by_source(meta.source)) > 0
    novelty = REPEAT_QUERY_NOVELTY if already_covered else 1.0
    expected_gain = meta.reliability * novelty

    uncertainty_reduction = current_uncertainty if action in CONCLUSION_ACTIONS else 0.0

    value = (
        W_GAIN * expected_gain
        + W_RELIABILITY * meta.reliability
        + W_NOVELTY * novelty
        + W_UNCERTAINTY_REDUCTION * uncertainty_reduction
        - W_COST * meta.cost
        - W_LATENCY * meta.latency
    )

    return ActionValue(
        action=action,
        expected_gain=round(expected_gain, 4),
        reliability=meta.reliability,
        novelty=round(novelty, 4),
        uncertainty_reduction=round(uncertainty_reduction, 4),
        cost=meta.cost,
        latency=meta.latency,
        value=round(value, 4),
    )


def select_next_best_evidence(
    store: EvidenceStore,
    available_actions: list[InvestigationAction],
    current_uncertainty: float = 1.0,
) -> ActionValue | None:
    if not available_actions:
        return None

    scored = [score_action(a, store, current_uncertainty) for a in available_actions]
    return max(scored, key=lambda av: av.value)
