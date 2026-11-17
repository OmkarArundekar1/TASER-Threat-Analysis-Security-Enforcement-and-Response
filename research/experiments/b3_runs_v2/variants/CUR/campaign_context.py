from dataclasses import dataclass, field
from datetime import datetime
from feature_orchestrator import FeatureVector

@dataclass(frozen=True)
class TransitionContext:
    """Immutable record of one resolved event's position in its campaign,
    captured by CampaignManager at the moment the event is attached to the
    campaign -- BEFORE any campaign state (in memory or in Neo4j) is
    updated with the event's own technique.

    Exists because the "previous technique" of a transition cannot be
    re-derived afterwards from Campaign.last_technique: that property is
    overwritten with the current technique by update_campaign_activity()
    and create_attack_event() before attack-chain learning runs (see
    research/STEP1_CHAIN_LEARNING.md). Passing this snapshot explicitly to
    chain_updater.update_attack_chain() removes the read-after-write
    dependency instead of reordering around it.

    previous_technique: the campaign's most recent RESOLVED technique
        before this event, or None if there is none to transition from.
    segment_start: True for the first resolved event of a newly created
        or reopened campaign. No transition is recorded for it -- this
        preserves the pre-existing learning policy unchanged; whether the
        transition that caused a split/reopen should be recorded is a
        separate, later decision (feedback (a)ii, Step 4).
    """
    campaign_id: str
    current_technique: str
    previous_technique: str | None
    segment_start: bool


@dataclass
class CampaignContext:
    campaign_id: str
    attacker_ip: str
    victim_ip: str
    status: str = "ACTIVE"
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    last_technique: str | None = None
    observed_chain: list = field(default_factory=list)
    predicted_next: str | None = None
    prediction_confidence: float = 0.0
    prediction_hits: int = 0
    prediction_misses: int = 0
    reopened_count: int = 0
    prediction_generated_at: datetime | None = None
    risk_score: float = 0.0
    skip_next_chain_update: bool = False
    techniques: set = field(default_factory=set)
    attack_chain: list = field(default_factory=list)
    features: FeatureVector | None = None