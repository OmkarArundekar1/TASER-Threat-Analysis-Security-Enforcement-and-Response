"""
soar/matcher.py
==================
PlaybookMatcher: given the CURRENT campaign, finds prior playbooks
generated for structurally/technically similar historical campaigns and
returns every signal separately (Phase 7's explicit requirement --
never collapse technique overlap, GNN topology similarity, and
attacker/victim identity into one blind score).

Reuses real, already-existing CYUKTI signals rather than inventing a
new similarity metric:
  - technique_similarity: Jaccard overlap of MITRE technique sets
    (CampaignContext.techniques, already real per-campaign data).
  - topology_similarity: ml.gnn.topology_similarity's existing
    gnn_topology_similarity_between_campaigns() -- None (not zero) when
    GNN is disabled/unavailable, never fabricated.
  - attacker_ip_match / victim_ip_match: exact identity facts.
  - historical execution stats: from soar.memory's PlaybookMemoryStore,
    CYUKTI's own record, never Shuffle's.
"""

from __future__ import annotations

import logging

from campaign_context import CampaignContext
from soar.memory import PlaybookMemoryStore
from soar.schema import HistoricalPlaybookMatch, Playbook

logger = logging.getLogger(__name__)


def _technique_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class PlaybookMatcher:
    def __init__(self, memory_store: PlaybookMemoryStore):
        self.memory_store = memory_store

    def find_matches(
        self,
        current_campaign: CampaignContext,
        neo4j_session=None,
        top_k: int = 5,
    ) -> list[HistoricalPlaybookMatch]:
        """neo4j_session: an open `driver.session()` used to load each
        candidate playbook's source campaign for technique comparison --
        passed in by the caller (soar/api.py) rather than opened here,
        so this module has no direct Neo4j driver dependency of its
        own and stays trivially unit-testable with a fake session."""
        candidates = [
            p for p in self.memory_store.list_playbooks()
            if p.source_campaign_id and p.source_campaign_id != current_campaign.campaign_id
        ]

        matches: list[HistoricalPlaybookMatch] = []
        for playbook in candidates:
            match = self._score_one(playbook, current_campaign, neo4j_session)
            if match is not None:
                matches.append(match)

        matches.sort(
            key=lambda m: (
                m.technique_similarity + (m.topology_similarity or 0.0),
                m.historical_success_rate or 0.0,
            ),
            reverse=True,
        )
        return matches[:top_k]

    def _score_one(
        self, playbook: Playbook, current_campaign: CampaignContext, neo4j_session
    ) -> HistoricalPlaybookMatch | None:
        source_campaign_id = playbook.source_campaign_id
        historical_techniques: set[str] = set(playbook.mitre_techniques or [])
        historical_attacker_ip = None
        historical_victim_ip = None

        if neo4j_session is not None:
            try:
                from dashboard_api import _load_campaign_context
                historical_context = _load_campaign_context(neo4j_session, source_campaign_id)
                if historical_context is not None:
                    historical_techniques = historical_context.techniques or historical_techniques
                    historical_attacker_ip = historical_context.attacker_ip
                    historical_victim_ip = historical_context.victim_ip
            except Exception:
                logger.exception("PlaybookMatcher: failed loading historical campaign %s", source_campaign_id)

        technique_similarity = _technique_jaccard(current_campaign.techniques or set(), historical_techniques)

        topology_similarity = None
        try:
            from ml.gnn.topology_similarity import gnn_topology_similarity_between_campaigns
            topology_similarity = gnn_topology_similarity_between_campaigns(
                current_campaign.campaign_id, source_campaign_id
            )
        except Exception:
            logger.exception("PlaybookMatcher: GNN topology similarity failed for %s", source_campaign_id)

        attacker_match = bool(historical_attacker_ip) and historical_attacker_ip == current_campaign.attacker_ip
        victim_match = bool(historical_victim_ip) and historical_victim_ip == current_campaign.victim_ip

        executions, successes, failures, success_rate = self.memory_store.historical_match_stats(playbook.playbook_id)

        if technique_similarity == 0.0 and not attacker_match and not victim_match and not (topology_similarity or 0):
            return None  # no real signal ties this playbook to the current incident at all

        reason_parts = []
        if technique_similarity > 0:
            reason_parts.append(f"{technique_similarity:.0%} MITRE technique overlap")
        if topology_similarity is not None and topology_similarity > 0:
            reason_parts.append(f"{topology_similarity:.0%} GNN topology similarity")
        if attacker_match:
            reason_parts.append("same attacker IP")
        if victim_match:
            reason_parts.append("same victim IP")
        if success_rate is not None:
            reason_parts.append(f"{success_rate:.0%} historical success rate over {executions} execution(s)")
        reason = "; ".join(reason_parts) if reason_parts else "weak historical signal"

        return HistoricalPlaybookMatch(
            playbook_id=playbook.playbook_id,
            playbook_name=playbook.name,
            source_campaign_id=source_campaign_id,
            technique_similarity=technique_similarity,
            topology_similarity=topology_similarity,
            attacker_ip_match=attacker_match,
            victim_ip_match=victim_match,
            historical_executions=executions,
            historical_successes=successes,
            historical_failures=failures,
            historical_success_rate=success_rate,
            recommendation_reason=reason,
        )
