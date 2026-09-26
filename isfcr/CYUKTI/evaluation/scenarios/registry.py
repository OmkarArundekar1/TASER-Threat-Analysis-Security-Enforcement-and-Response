"""
scenarios/registry.py
========================
AttackScenario: a formal, pre-declared expected outcome for a real
attack scenario this project has actually run, defined from
independent sources (lab topology, the synthetic traffic generator's
own declared intent, rule text) -- never from what CYUKTI subsequently
computed for it.

scenarios.json is loaded at import time. Every entry in it traces to a
specific already-written project document (see `evidence_sources`) --
none were invented for this task. Two provenance tiers exist and are
labeled honestly per scenario:
  - "manual_live_attack": a human ran a real tool (nmap, etc.) against
    a real Kali->Ubuntu VM pair, observed directly in this project's
    history (INCIDENT_VIEW.md, MITRE_MAPPING.md).
  - "synthetic_generator_declared_intent": this lab's traffic generator
    (see ACCURACY_EVALUATION.md's "synthetic traffic generator
    (LOGIN_ATTACK, BOT_ATTACK, etc.)") produced the traffic, and the
    generator's OWN declared attack-type label is used as the
    independent signal -- independent of CYUKTI because the generator
    is a separate upstream tool that commits to an attack type before
    Wazuh/CYUKTI ever sees the traffic, but weaker than a manual,
    directly-observed attack and disclosed as such everywhere it's used.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios.json")


@dataclass
class AttackScenario:
    scenario_id: str
    scenario_name: str
    attacker: str
    victim: str
    attack_family: str
    expected_techniques: list[str]
    expected_threat_status: str
    benign_or_malicious: str
    provenance: str                     # "manual_live_attack" | "synthetic_generator_declared_intent"
    evidence_sources: list[str] = field(default_factory=list)
    expected_attribution: str | None = None
    expected_prediction_transition: str | None = None
    execution_command_reference: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_registry() -> list[AttackScenario]:
    if not os.path.exists(_REGISTRY_PATH):
        return []
    with open(_REGISTRY_PATH) as f:
        raw = json.load(f)
    return [AttackScenario(**s) for s in raw]


def get_scenario(scenario_id: str) -> AttackScenario | None:
    for s in load_registry():
        if s.scenario_id == scenario_id:
            return s
    return None
