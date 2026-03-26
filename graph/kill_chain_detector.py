"""
graph/kill_chain_detector.py
==============================
Infers current kill-chain phase from an attack graph.
Combines graph structure analysis + attack stage progression scoring.
"""

from __future__ import annotations

from typing import Any

from classifiers.attack_stage_mapper import AttackStageMapper, _PHASE_ORDER

_mapper = AttackStageMapper()


class KillChainDetector:
    """
    Detects kill-chain progression from an attack graph + event history.

    Args:
        events:  List of enriched event dicts with 'attack_label' field.
        graph:   Graph dict from AttackGraphBuilder.build().
        src_ip:  Attacker IP (for report metadata only).
    """

    def detect(
        self,
        events: list[dict[str, Any]],
        graph: dict[str, Any],
        src_ip: str = "unknown",
    ) -> dict[str, Any]:
        """
        Run kill-chain inference.

        Returns::

            {
                "attacker_ip": str,          # audit only
                "detected_stages": [...],
                "current_phase": str,
                "progression_score": float,
                "is_active_campaign": bool,
                "is_multi_stage": bool,
                "highest_severity": float,
                "recommended_priority": str
            }
        """
        if not events:
            return self._empty_result(src_ip)

        labels = [ev.get("attack_label", "Other") for ev in events]
        severities = [float(ev.get("severity_score", 0.0)) for ev in events]

        progression = _mapper.infer_campaign_progression(labels)
        unique_phases = progression["observed_phases"]
        max_order = progression["max_kill_chain_order"]
        is_multi_stage = graph.get("is_multi_stage", len(unique_phases) > 1)

        highest_severity = max(severities) if severities else 0.0
        priority = self._priority(max_order, highest_severity)

        return {
            "attacker_ip": src_ip,                  # audit only
            "detected_stages": unique_phases,
            "current_phase": progression["max_kill_chain_phase"],
            "progression_score": progression["progression_score"],
            "is_active_campaign": progression["is_active_campaign"],
            "is_multi_stage": is_multi_stage,
            "highest_severity": round(highest_severity, 4),
            "recommended_priority": priority,
            "event_count": len(events),
        }

    @staticmethod
    def _priority(phase_order: int, severity: float) -> str:
        if phase_order >= 6 or severity > 10:
            return "P1_CRITICAL"
        if phase_order >= 4 or severity > 6:
            return "P2_HIGH"
        if phase_order >= 2 or severity > 3:
            return "P3_MEDIUM"
        return "P4_LOW"

    @staticmethod
    def _empty_result(src_ip: str) -> dict[str, Any]:
        return {
            "attacker_ip": src_ip,
            "detected_stages": [],
            "current_phase": "None",
            "progression_score": 0.0,
            "is_active_campaign": False,
            "is_multi_stage": False,
            "highest_severity": 0.0,
            "recommended_priority": "P4_LOW",
            "event_count": 0,
        }
