"""
Regression test for a real, live-confirmed defect documented in
review_pack/10_limitations_and_future_work.md ("Current Issue 1"):
`create_campaign_context()` (the Phase 20 UNKNOWN-first-event path,
which deliberately bypasses `activate_campaign()` since that function
also calls `append_technique()`, which must never run for UNKNOWN
events) never set `CampaignContext.first_seen`/`last_seen`, leaving
them at the dataclass default of `None`. The maintenance worker's
`expire_active_campaigns()` then crashed every cycle computing
`now - None` for as long as that context stayed cached --
`TypeError: unsupported operand type(s) for -: 'datetime.datetime' and 'NoneType'`,
live-observed in review_pack's own audit and explicitly left unfixed
there pending separate authorization for a broader session.

This is pure operational/maintenance-loop technical debt with zero
relationship to the frozen Phase 21/22 NBE research findings (a
different subsystem entirely) -- fixing it does not reopen or alter
any frozen finding.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from campaign_manager import CampaignManager


@pytest.fixture()
def manager(monkeypatch):
    monkeypatch.setattr("campaign_manager.create_campaign_db", lambda attacker_ip, victim_ip: "CAMP_TEST_UNKNOWN")
    monkeypatch.setattr("campaign_manager.expire_stale_campaigns_db", lambda timeout: [])
    monkeypatch.setattr("campaign_manager.close_campaign_db", lambda campaign_id: None)
    return CampaignManager()


def test_create_campaign_context_populates_first_and_last_seen(manager):
    """The direct regression: this path must never leave first_seen/last_seen
    at the CampaignContext dataclass default of None."""
    context = manager.create_campaign_context("10.0.0.5", "10.0.0.9", current_technique=None)

    assert context.first_seen is not None
    assert context.last_seen is not None
    assert isinstance(context.last_seen, datetime)
    assert context.last_seen.tzinfo is not None  # timezone-aware, matching the rest of the codebase


def test_expire_active_campaigns_does_not_crash_on_an_unknown_first_event_context(manager):
    """The actual crash reproduction: a context created via the UNKNOWN
    path (never routed through activate_campaign()) must survive a real
    maintenance-worker sweep without TypeError -- this raised before the
    fix, on every real listener process, every ~5 seconds, for as long
    as such a context stayed cached."""
    manager.create_campaign_context("10.0.0.5", "10.0.0.9", current_technique=None)

    # Must not raise -- this is the exact call the listener's
    # maintenance_worker() makes every 5 seconds (listener/wazuh_listener.py).
    manager.expire_active_campaigns()


def test_expire_active_campaigns_actually_expires_a_genuinely_stale_unknown_context(manager, monkeypatch):
    """Not just "doesn't crash" -- confirms the timeout logic now
    functions correctly for this path too, since it previously could
    never be reached at all (the crash happened before any comparison
    completed)."""
    from config import CAMPAIGN_TIMEOUT

    context = manager.create_campaign_context("10.0.0.5", "10.0.0.9", current_technique=None)
    context.last_seen = datetime.now(timezone.utc) - timedelta(seconds=CAMPAIGN_TIMEOUT + 30)

    closed = []
    monkeypatch.setattr("campaign_manager.close_campaign_db", lambda campaign_id: closed.append(campaign_id))

    manager.expire_active_campaigns()

    assert context.status == "INACTIVE"
    assert closed == ["CAMP_TEST_UNKNOWN"]
