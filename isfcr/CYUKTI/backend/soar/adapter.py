"""
soar/adapter.py
==================
PlaybookAdaptation: takes a historical Playbook (found by
PlaybookMatcher) and rewrites the IDENTITY-bound fields of each action
(attacker IP, victim IP, campaign/operation ID) to the current
incident, without touching action semantics (type, name, destructive,
requires_approval) -- copying a playbook must never silently change
what it does, only who/what it targets.
"""

from __future__ import annotations

import copy

from campaign_context import CampaignContext
from soar.schema import Playbook, _new_id

# Input keys that name an identity the historical playbook targeted --
# only these get rewritten. Anything else (mitigation_id, technique,
# severity computed at generation time) is left as historical fact,
# not silently re-derived.
_IDENTITY_INPUT_KEYS = {"ip", "campaign_id", "operation_id"}


class PlaybookAdaptation:
    def adapt(
        self,
        historical_playbook: Playbook,
        current_campaign: CampaignContext,
        operation_id: str | None = None,
    ) -> Playbook:
        adapted = copy.deepcopy(historical_playbook)
        adapted.playbook_id = _new_id("pb")
        adapted.version = 1
        adapted.adapted_from_playbook_id = historical_playbook.playbook_id
        adapted.source_campaign_id = current_campaign.campaign_id
        adapted.name = f"{historical_playbook.name}_ADAPTED"
        adapted.trigger_conditions = {
            **historical_playbook.trigger_conditions,
            "adapted_from_campaign": historical_playbook.source_campaign_id,
        }

        for action in adapted.actions:
            new_inputs = dict(action.inputs)
            for key in list(new_inputs.keys()):
                if key not in _IDENTITY_INPUT_KEYS:
                    continue
                if key == "ip":
                    # Historical actions targeting the historical attacker
                    # become the current attacker; historical victim-target
                    # actions become the current victim. Distinguish by
                    # which historical identity the value equals -- if it
                    # matches neither (e.g. a third-party IOC IP), leave it.
                    if new_inputs[key] and new_inputs[key] not in (
                        current_campaign.attacker_ip, current_campaign.victim_ip,
                    ):
                        continue
                elif key == "campaign_id":
                    new_inputs[key] = current_campaign.campaign_id
                    continue
                elif key == "operation_id":
                    new_inputs[key] = operation_id
                    continue
            # IP re-targeting needs the action's own semantic role
            # (attacker- vs victim-directed), which the action_type
            # encodes -- rewrite by role, not by matching the old value.
            if "ip" in new_inputs:
                if action.action_type in ("enrich_ip", "block_ip", "threat_intel_lookup"):
                    new_inputs["ip"] = current_campaign.attacker_ip
                elif action.action_type in ("collect_evidence", "isolate_host"):
                    new_inputs["ip"] = current_campaign.victim_ip
            action.inputs = new_inputs
            action.reason = f"[Adapted from {historical_playbook.playbook_id}] {action.reason}"

        return adapted
