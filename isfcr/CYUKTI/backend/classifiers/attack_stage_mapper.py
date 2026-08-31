"""
classifiers/attack_stage_mapper.py
=====================================
Maps attack labels to MITRE ATT&CK tactics and Kill-Chain phases.
"""

from __future__ import annotations

from typing import Any

# MITRE ATT&CK + Kill-Chain mapping
_ATTACK_MAP: dict[str, dict[str, str]] = {
    "BENIGN": {
        "mitre_tactic": "None",
        "kill_chain_phase": "None",
        "description": "Normal network traffic",
        "risk_level": "NONE",
    },
    "PortScan": {
        "mitre_tactic": "Discovery",
        "kill_chain_phase": "Reconnaissance",
        "description": "Active network scanning / host discovery",
        "risk_level": "LOW",
    },
    "DDoS": {
        "mitre_tactic": "Impact",
        "kill_chain_phase": "Actions on Objectives",
        "description": "Distributed denial-of-service attack",
        "risk_level": "HIGH",
    },
    "DoS": {
        "mitre_tactic": "Impact",
        "kill_chain_phase": "Actions on Objectives",
        "description": "Denial-of-service attack",
        "risk_level": "HIGH",
    },
    "BruteForce": {
        "mitre_tactic": "Credential Access",
        "kill_chain_phase": "Exploitation",
        "description": "Password brute-force / credential stuffing",
        "risk_level": "HIGH",
    },
    "WebAttack": {
        "mitre_tactic": "Initial Access",
        "kill_chain_phase": "Delivery",
        "description": "Web-based attack (SQLi / XSS / Brute Force)",
        "risk_level": "HIGH",
    },
    "Infiltration": {
        "mitre_tactic": "Command and Control",
        "kill_chain_phase": "Command & Control",
        "description": "Network infiltration / C2 communication",
        "risk_level": "CRITICAL",
    },
    "Bot": {
        "mitre_tactic": "Execution",
        "kill_chain_phase": "Installation",
        "description": "Bot / malware beaconing activity",
        "risk_level": "CRITICAL",
    },
    "Exploit": {
        "mitre_tactic": "Execution",
        "kill_chain_phase": "Exploitation",
        "description": "Exploit / vulnerability exploitation",
        "risk_level": "CRITICAL",
    },
    "Other": {
        "mitre_tactic": "Unknown",
        "kill_chain_phase": "Unknown",
        "description": "Unclassified traffic",
        "risk_level": "LOW",
    },
}

# Kill-Chain phase ordering (earlier = lower number)
_PHASE_ORDER = {
    "None": 0,
    "Reconnaissance": 1,
    "Weaponization": 2,
    "Delivery": 3,
    "Exploitation": 4,
    "Installation": 5,
    "Command & Control": 6,
    "Actions on Objectives": 7,
    "Unknown": 0,
}


class AttackStageMapper:
    """
    Maps attack_label → MITRE ATT&CK tactic + Kill-Chain phase.

    Usage::
        mapper = AttackStageMapper()
        info = mapper.map("DDoS")
        # -> {"mitre_tactic": "Impact", "kill_chain_phase": "Actions on Objectives", ...}
    """

    def map(self, attack_label: str) -> dict[str, Any]:
        """Return full mapping for a given attack label."""
        return dict(_ATTACK_MAP.get(attack_label, _ATTACK_MAP["Other"]))

    def get_phase_order(self, attack_label: str) -> int:
        """Return numeric kill-chain phase order (for progression scoring)."""
        phase = _ATTACK_MAP.get(attack_label, {}).get("kill_chain_phase", "Unknown")
        return _PHASE_ORDER.get(phase, 0)

    def get_risk_level(self, attack_label: str) -> str:
        return _ATTACK_MAP.get(attack_label, _ATTACK_MAP["Other"])["risk_level"]

    def infer_campaign_progression(self, labels: list[str]) -> dict[str, Any]:
        """
        Given a sequence of attack labels over time, infer attack campaign stage.
        Returns the highest phase observed + a progression score.
        """
        orders = [self.get_phase_order(l) for l in labels]
        max_order = max(orders) if orders else 0
        unique_phases = list({_ATTACK_MAP.get(l, _ATTACK_MAP["Other"])["kill_chain_phase"] for l in labels})

        return {
            "max_kill_chain_order": max_order,
            "max_kill_chain_phase": [k for k, v in _PHASE_ORDER.items() if v == max_order][0],
            "observed_phases": unique_phases,
            "is_active_campaign": max_order >= 4,  # Exploitation or beyond
            "progression_score": round(max_order / 7.0, 3),
        }
