from active_response.containment_actions import (
    ContainmentAction, EXECUTABLE_ACTIONS, is_executable, is_destructive,
)


def test_only_block_source_ip_is_executable():
    assert is_executable(ContainmentAction.BLOCK_SOURCE_IP) is True
    for action in ContainmentAction:
        if action is not ContainmentAction.BLOCK_SOURCE_IP:
            assert is_executable(action) is False, f"{action} must not be executable this phase"


def test_executable_set_has_exactly_one_member():
    assert EXECUTABLE_ACTIONS == frozenset({ContainmentAction.BLOCK_SOURCE_IP})


def test_all_actions_are_classified_destructive():
    for action in ContainmentAction:
        assert is_destructive(action) is True


def test_future_actions_exist_as_enum_but_are_declared_unsupported():
    assert ContainmentAction.ISOLATE_HOST in ContainmentAction
    assert not is_executable(ContainmentAction.ISOLATE_HOST)
