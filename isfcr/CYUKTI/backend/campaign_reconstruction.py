"""
campaign_reconstruction.py
=============================
Repairs AttackEvent nodes orphaned by a historical gap in Campaign
persistence: every orphaned event already carries campaign_id,
attacker_ip, victim_ip, and full technique/timestamp data as its own
properties (see backend/ml/../../scripts/orphan_investigation.py) —
Campaign creation and AttackEvent creation are two separate Neo4j
transactions (create_campaign_db, then create_attack_event), so a crash
or early-pipeline bug between the two could leave events pointing at a
campaign_id whose Campaign node was never actually created.

Evidence ruling out "duplicate, discard" as the repair: every orphaned
fingerprint also appears on a properly-linked event, but always dated
weeks LATER (Aug 3+ vs. the orphans' Jul 17-20) — these are separate
real occurrences of the same recurring attack tool, not double-counted
copies of the same event.

This module RECONSTRUCTS the missing Campaign node from data already
present on its own orphaned children — it does not guess anything not
already recorded. Idempotent: reconstructing a campaign_id that already
has a Campaign node is a no-op (checked before any write).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neo4j_client import driver
from risk_recalculation import _tps_for


@dataclass
class OrphanCampaignPlan:
    campaign_id: str
    attacker_ip: str
    victim_ip: str
    event_ids: list[str] = field(default_factory=list)
    first_seen: object = None
    last_seen: object = None
    last_technique: str = ""
    risk_score: float = 0.0


def find_orphan_campaign_ids() -> list[str]:
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (e:AttackEvent)
            WHERE NOT EXISTS { MATCH (:Campaign)-[:HAS_EVENT]->(e) }
            RETURN DISTINCT e.campaign_id AS campaign_id
            """
        ).data()
    return [r["campaign_id"] for r in rows]


def plan_reconstruction(campaign_id: str) -> OrphanCampaignPlan | None:
    """Read-only: builds the reconstruction plan from the orphaned
    events' own properties. Returns None if a Campaign node already
    exists for this campaign_id (nothing to reconstruct) or if the
    orphaned events don't have a single consistent attacker/victim pair
    (would violate the one-campaign-one-pair invariant — refuse rather
    than guess which pair is correct)."""
    with driver.session() as session:
        existing = session.run(
            "MATCH (c:Campaign {campaign_id: $cid}) RETURN c", cid=campaign_id
        ).single()
        if existing is not None:
            return None  # already reconstructed or never was orphaned — no-op

        rows = session.run(
            """
            MATCH (e:AttackEvent {campaign_id: $cid})
            WHERE NOT EXISTS { MATCH (:Campaign)-[:HAS_EVENT]->(e) }
            RETURN e.event_id AS event_id, e.attacker_ip AS attacker_ip,
                   e.victim_ip AS victim_ip, e.attack_id AS attack_id,
                   e.occurrences AS occurrences, e.first_seen AS first_seen,
                   e.last_seen AS last_seen
            ORDER BY e.first_seen
            """,
            cid=campaign_id,
        ).data()

    if not rows:
        return None

    attackers = {r["attacker_ip"] for r in rows}
    victims = {r["victim_ip"] for r in rows}
    if len(attackers) != 1 or len(victims) != 1:
        raise ValueError(
            f"{campaign_id}: orphaned events don't share a single attacker/victim pair "
            f"(attackers={attackers}, victims={victims}) — refusing to guess, needs manual review"
        )

    total_risk = sum(_tps_for(r["attack_id"])[1] * (r["occurrences"] or 0) for r in rows)

    return OrphanCampaignPlan(
        campaign_id=campaign_id,
        attacker_ip=attackers.pop(),
        victim_ip=victims.pop(),
        event_ids=[r["event_id"] for r in rows],
        first_seen=rows[0]["first_seen"],
        last_seen=rows[-1]["last_seen"],
        last_technique=rows[-1]["attack_id"],
        risk_score=total_risk,
    )


def apply_reconstruction(plan: OrphanCampaignPlan) -> None:
    """Creates the missing Campaign node and its LAUNCHED/TARGETS/HAS_EVENT
    relationships to the already-existing orphaned AttackEvent nodes.
    Does not create new AttackEvent nodes — only links existing ones.
    Idempotent: uses MERGE throughout, and plan_reconstruction() already
    refuses to re-plan a campaign_id that has a Campaign node."""
    with driver.session() as session:
        session.run(
            """
            MERGE (c:Campaign {campaign_id: $campaign_id})
            ON CREATE SET
                c.attacker_ip = $attacker_ip,
                c.victim_ip = $victim_ip,
                c.status = 'INACTIVE',
                c.first_seen = $first_seen,
                c.last_seen = $last_seen,
                c.last_technique = $last_technique,
                c.occurrences = size($event_ids),
                c.total_tps = $risk_score,
                c.risk_score = $risk_score,
                c.reconstructed = true,
                c.reconstructed_at = datetime()

            WITH c
            MERGE (a:Attacker {ip: $attacker_ip})
            MERGE (h:Host {ip: $victim_ip})
            MERGE (a)-[:LAUNCHED]->(c)
            MERGE (c)-[:TARGETS]->(h)

            WITH c
            UNWIND $event_ids AS eid
            MATCH (e:AttackEvent {event_id: eid})
            MERGE (c)-[:HAS_EVENT]->(e)
            """,
            campaign_id=plan.campaign_id,
            attacker_ip=plan.attacker_ip,
            victim_ip=plan.victim_ip,
            first_seen=plan.first_seen,
            last_seen=plan.last_seen,
            last_technique=plan.last_technique,
            event_ids=plan.event_ids,
            risk_score=plan.risk_score,
        )


@dataclass
class IntegrityReport:
    total_events: int
    linked_events: int
    orphaned_events: int
    orphan_campaign_ids: list[str]

    @property
    def is_clean(self) -> bool:
        return self.orphaned_events == 0


def check_integrity() -> IntegrityReport:
    """The data invariant this project relies on: every AttackEvent
    belongs to exactly one Campaign (every query path — dataset builder,
    feature engines, evidence collectors — reaches events through
    Campaign-[:HAS_EVENT]->AttackEvent; nothing treats a standalone
    AttackEvent as meaningful). This check makes a violation of that
    invariant detectable rather than silently ignored, the way the 44
    orphans discovered during this audit were until now."""
    with driver.session() as session:
        total = session.run("MATCH (e:AttackEvent) RETURN count(e) AS n").single()["n"]
        linked = session.run(
            "MATCH (:Campaign)-[:HAS_EVENT]->(e:AttackEvent) RETURN count(DISTINCT e) AS n"
        ).single()["n"]
        orphan_ids = [
            r["campaign_id"] for r in session.run(
                "MATCH (e:AttackEvent) WHERE NOT EXISTS { MATCH (:Campaign)-[:HAS_EVENT]->(e) } "
                "RETURN DISTINCT e.campaign_id AS campaign_id"
            )
        ]

    return IntegrityReport(
        total_events=total, linked_events=linked,
        orphaned_events=total - linked, orphan_campaign_ids=orphan_ids,
    )


def reconstruct_all_orphans() -> list[OrphanCampaignPlan]:
    """Runs plan + apply for every currently-orphaned campaign_id.
    Returns the plans that were actually applied (skips no-ops)."""
    applied = []
    for campaign_id in find_orphan_campaign_ids():
        plan = plan_reconstruction(campaign_id)
        if plan is None:
            continue
        apply_reconstruction(plan)
        applied.append(plan)
    return applied
