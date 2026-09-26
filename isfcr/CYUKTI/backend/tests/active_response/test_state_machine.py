import pytest

from active_response.state_machine import (
    ResponseState, ResponseStateMachine, InvalidStateTransition,
)


def test_full_happy_path_transition_sequence():
    sm = ResponseStateMachine()
    sm.transition(ResponseState.INVESTIGATE)
    sm.transition(ResponseState.RECOMMEND)
    sm.transition(ResponseState.CONTAIN)
    sm.transition(ResponseState.VERIFY, evidence="firewall rule confirmed present")
    sm.transition(ResponseState.VERIFIED, evidence="post-attack connection blocked, pre-attack succeeded")
    assert sm.current_state == ResponseState.VERIFIED
    assert len(sm.history) == 5


def test_cannot_jump_directly_from_contain_to_verified():
    sm = ResponseStateMachine()
    sm.transition(ResponseState.INVESTIGATE)
    sm.transition(ResponseState.RECOMMEND)
    sm.transition(ResponseState.CONTAIN)
    with pytest.raises(InvalidStateTransition):
        sm.transition(ResponseState.VERIFIED, evidence="anything")


def test_cannot_jump_from_observe_to_contain():
    sm = ResponseStateMachine()
    with pytest.raises(InvalidStateTransition):
        sm.transition(ResponseState.CONTAIN)


def test_verify_transition_requires_non_empty_evidence():
    sm = ResponseStateMachine()
    sm.transition(ResponseState.INVESTIGATE)
    sm.transition(ResponseState.RECOMMEND)
    sm.transition(ResponseState.CONTAIN)
    with pytest.raises(ValueError, match="requires non-empty evidence"):
        sm.transition(ResponseState.VERIFY, evidence="")


def test_verified_transition_requires_non_empty_evidence():
    sm = ResponseStateMachine()
    sm.transition(ResponseState.INVESTIGATE)
    sm.transition(ResponseState.RECOMMEND)
    sm.transition(ResponseState.CONTAIN)
    sm.transition(ResponseState.VERIFY, evidence="checked")
    with pytest.raises(ValueError, match="requires non-empty evidence"):
        sm.transition(ResponseState.VERIFIED, evidence="")


def test_contain_can_fail_directly():
    sm = ResponseStateMachine()
    sm.transition(ResponseState.INVESTIGATE)
    sm.transition(ResponseState.RECOMMEND)
    sm.transition(ResponseState.CONTAIN)
    sm.transition(ResponseState.FAILED, evidence="firewall backend returned an error")
    assert sm.current_state == ResponseState.FAILED
    assert sm.is_terminal()


def test_verify_can_fail():
    sm = ResponseStateMachine()
    sm.transition(ResponseState.INVESTIGATE)
    sm.transition(ResponseState.RECOMMEND)
    sm.transition(ResponseState.CONTAIN)
    sm.transition(ResponseState.VERIFY, evidence="checking")
    sm.transition(ResponseState.FAILED, evidence="post-attack connection still succeeded")
    assert sm.current_state == ResponseState.FAILED


def test_terminal_states_reject_any_further_transition():
    sm = ResponseStateMachine()
    sm.transition(ResponseState.INVESTIGATE)
    sm.transition(ResponseState.RECOMMEND)
    sm.transition(ResponseState.CONTAIN)
    sm.transition(ResponseState.FAILED, evidence="x")
    with pytest.raises(InvalidStateTransition):
        sm.transition(ResponseState.VERIFY, evidence="y")


def test_verified_can_expire_or_roll_back():
    sm = ResponseStateMachine()
    sm.transition(ResponseState.INVESTIGATE)
    sm.transition(ResponseState.RECOMMEND)
    sm.transition(ResponseState.CONTAIN)
    sm.transition(ResponseState.VERIFY, evidence="checked")
    sm.transition(ResponseState.VERIFIED, evidence="confirmed")
    sm.transition(ResponseState.EXPIRED, evidence="TTL elapsed")
    assert sm.is_terminal()


def test_history_records_are_ordered_and_immutable_in_place():
    sm = ResponseStateMachine()
    sm.transition(ResponseState.INVESTIGATE)
    first_history = list(sm.history)
    sm.transition(ResponseState.RECOMMEND)
    assert len(first_history) == 1
    assert len(sm.history) == 2
