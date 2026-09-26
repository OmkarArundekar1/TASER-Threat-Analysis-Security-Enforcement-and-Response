import pytest

from active_response.correlation import derive_correlation_id, require_correlation_id, MissingCorrelationID


def test_campaign_id_is_reused_verbatim_as_correlation_id():
    assert derive_correlation_id(campaign_id="CAMP_ABC123") == "CAMP_ABC123"


def test_attack_event_id_used_when_no_campaign_exists_yet():
    corr = derive_correlation_id(attack_event_id="AE-1")
    assert corr == "PRECAMPAIGN:AE-1"


def test_campaign_id_takes_precedence_over_attack_event_id():
    corr = derive_correlation_id(campaign_id="CAMP_X", attack_event_id="AE-1")
    assert corr == "CAMP_X"


def test_fresh_id_minted_when_neither_exists():
    corr = derive_correlation_id()
    assert corr.startswith("corr_")


def test_two_different_incidents_get_distinct_correlation_ids():
    a = derive_correlation_id(campaign_id="CAMP_A")
    b = derive_correlation_id(campaign_id="CAMP_B")
    assert a != b


def test_same_campaign_always_derives_the_same_correlation_id():
    a = derive_correlation_id(campaign_id="CAMP_A")
    b = derive_correlation_id(campaign_id="CAMP_A")
    assert a == b


def test_require_correlation_id_passes_through_a_real_value():
    assert require_correlation_id("corr-1") == "corr-1"


def test_require_correlation_id_rejects_none():
    with pytest.raises(MissingCorrelationID):
        require_correlation_id(None)


def test_require_correlation_id_rejects_empty_string():
    with pytest.raises(MissingCorrelationID):
        require_correlation_id("")
