"""
Regression tests for mitre_mapper.py's technique->stage mapping,
following the real-data audit that found:
  - 9 of the 15 techniques observed in real campaigns were missing
    entirely (silently falling to "Unknown" -> tps=0)
  - T1053.003 mapped to "Cron abuse", a string that isn't a TPS_MAP key
Fixed: 5 of the 9 missing techniques have an unambiguous real ATT&CK
tactic covered by TPS_MAP and are now mapped; T1053.003 is corrected;
4 remain intentionally unmapped (documented in mitre_mapper.py) because
their real tactics (Discovery, Execution, Collection, Defense Evasion)
have no TPS_MAP entry at all — forcing them into one of the 6 existing
stages would be fabricating a mapping, not fixing one.
"""

from config import TPS_MAP
from mitre_mapper import MITRE_TO_STAGE

# All 15 techniques observed in the real 33-campaign dataset (see
# scripts/technique_hierarchy_analysis.py). Split by whether TPS_MAP
# has ANY entry for their real ATT&CK tactic(s).
MAPPABLE_OBSERVED_TECHNIQUES = [
    "T1595", "T1595.002", "T1110", "T1110.001", "T1053.003",
    "T1078", "T1021.004", "T1055", "T1190", "T1210",
]
INTENTIONALLY_UNMAPPED_TECHNIQUES = [
    "T1057",       # Discovery — no TPS_MAP entry
    "T1059",       # Execution — no TPS_MAP entry
    "T1059.007",   # Execution (subtechnique of T1059) — no TPS_MAP entry
    "T1114",       # Collection — no TPS_MAP entry
    "T1562.001",   # Defense Evasion — no TPS_MAP entry
]


def test_every_mappable_observed_technique_has_an_explicit_mapping():
    for technique_id in MAPPABLE_OBSERVED_TECHNIQUES:
        assert technique_id in MITRE_TO_STAGE, f"{technique_id} should have an explicit stage mapping"


def test_intentionally_unmapped_techniques_fall_through_to_unknown():
    for technique_id in INTENTIONALLY_UNMAPPED_TECHNIQUES:
        assert technique_id not in MITRE_TO_STAGE
        assert MITRE_TO_STAGE.get(technique_id, "Unknown") == "Unknown"


def test_every_mapped_stage_is_a_valid_tps_map_key():
    for technique_id, stage in MITRE_TO_STAGE.items():
        assert stage in TPS_MAP, (
            f"{technique_id} maps to '{stage}', which is not a TPS_MAP key — "
            "this technique would silently contribute tps=0 despite having an entry"
        )


def test_t1053_003_no_longer_maps_to_invalid_cron_abuse_string():
    assert MITRE_TO_STAGE["T1053.003"] != "Cron abuse"
    assert MITRE_TO_STAGE["T1053.003"] == "Privilege Escalation"


def test_newly_added_techniques_have_expected_stages():
    # each grounded in the real ATT&CK tactic pulled from the STIX corpus
    # (scripts/build_stage_mapping.py) — see mitre_mapper.py docstring
    assert MITRE_TO_STAGE["T1055"] == "Privilege Escalation"   # real tactics include privilege-escalation
    assert MITRE_TO_STAGE["T1190"] == "Initial Access"          # real tactic: initial-access
    assert MITRE_TO_STAGE["T1210"] == "Lateral Movement"        # real tactic: lateral-movement
    assert MITRE_TO_STAGE["T1595.002"] == "Reconnaissance"      # real tactic: reconnaissance


def test_parent_subtechnique_stage_consistency():
    # subtechnique must map to the same stage as its parent where both are mapped
    assert MITRE_TO_STAGE["T1595.002"] == MITRE_TO_STAGE["T1595"]
    assert MITRE_TO_STAGE["T1110.001"] == MITRE_TO_STAGE["T1110"]
    assert MITRE_TO_STAGE["T1021.004"] == MITRE_TO_STAGE["T1021"]


def test_unmapped_subtechnique_pair_consistent_absence():
    # T1059 and T1059.007 both lack a TPS_MAP-covered tactic (execution) —
    # both should be absent, not just one of them
    assert "T1059" not in MITRE_TO_STAGE
    assert "T1059.007" not in MITRE_TO_STAGE


def test_unmapped_future_technique_falls_back_to_unknown():
    assert MITRE_TO_STAGE.get("T9999.999", "Unknown") == "Unknown"
