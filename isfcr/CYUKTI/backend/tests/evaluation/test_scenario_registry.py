from scenarios.registry import load_registry, get_scenario


def test_registry_loads_real_committed_scenarios():
    scenarios = load_registry()
    ids = {s.scenario_id for s in scenarios}
    assert "SCN-NMAP-001" in ids
    assert "SCN-SSHBRUTE-001" in ids
    assert len(scenarios) >= 4


def test_get_scenario_returns_none_for_unknown_id():
    assert get_scenario("SCN-DOES-NOT-EXIST") is None


def test_get_scenario_returns_the_right_scenario():
    scenario = get_scenario("SCN-NMAP-001")
    assert scenario is not None
    assert scenario.attacker == "192.168.56.106"
    assert "T1595" in scenario.expected_techniques


def test_session_boundary_scenarios_have_no_mitre_ground_truth():
    """Anti-circularity check on the registry data itself: the two
    session-boundary scenarios must not carry a technique list, since
    their only recorded technique observations are CYUKTI's own
    resolved output and must never be used as MITRE ground truth."""
    for scenario_id in ("SCN-SESSION-BOUNDARY-001", "SCN-SESSION-BOUNDARY-002"):
        scenario = get_scenario(scenario_id)
        assert scenario is not None
        assert scenario.expected_techniques == []
