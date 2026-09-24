import requests

from soar.shuffle_client import ShuffleClient, ShuffleTriggerOutcome


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text_body=""):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text_body

    def json(self):
        if self._json_body is None:
            raise ValueError("no json")
        return self._json_body


def test_trigger_returns_not_configured_when_no_webhook_url():
    client = ShuffleClient(webhook_url="")
    result = client.trigger({"a": 1})
    assert result.outcome == ShuffleTriggerOutcome.NOT_CONFIGURED


def test_trigger_reports_synchronous_result_when_shuffle_returns_content(monkeypatch):
    client = ShuffleClient(webhook_url="https://shuffle.example/hooks/abc")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **kw: _FakeResponse(200, json_body={"execution_id": "ex1", "action_output": "blocked"}),
    )
    result = client.trigger({"campaign_id": "CAMP_1"})
    assert result.outcome == ShuffleTriggerOutcome.SYNCHRONOUS_RESULT
    assert result.shuffle_execution_id == "ex1"
    assert result.output["action_output"] == "blocked"


def test_trigger_reports_triggered_when_body_is_ack_only(monkeypatch):
    client = ShuffleClient(webhook_url="https://shuffle.example/hooks/abc")
    monkeypatch.setattr("requests.post", lambda *a, **kw: _FakeResponse(200, json_body={"success": True}))
    result = client.trigger({"campaign_id": "CAMP_1"})
    assert result.outcome == ShuffleTriggerOutcome.TRIGGERED


def test_trigger_reports_auth_failed_on_401(monkeypatch):
    client = ShuffleClient(webhook_url="https://shuffle.example/hooks/abc")
    monkeypatch.setattr("requests.post", lambda *a, **kw: _FakeResponse(401))
    result = client.trigger({})
    assert result.outcome == ShuffleTriggerOutcome.AUTH_FAILED


def test_trigger_reports_timeout(monkeypatch):
    client = ShuffleClient(webhook_url="https://shuffle.example/hooks/abc")

    def _raise(*a, **kw):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr("requests.post", _raise)
    result = client.trigger({})
    assert result.outcome == ShuffleTriggerOutcome.TIMEOUT


def test_trigger_reports_error_on_network_failure(monkeypatch):
    client = ShuffleClient(webhook_url="https://shuffle.example/hooks/abc")

    def _raise(*a, **kw):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr("requests.post", _raise)
    result = client.trigger({})
    assert result.outcome == ShuffleTriggerOutcome.ERROR
    assert "refused" in result.error


def test_get_execution_status_not_configured_without_base_url_and_key():
    client = ShuffleClient(webhook_url="https://shuffle.example/hooks/abc")
    result = client.get_execution_status("wf1", "ex1")
    assert result.outcome == "not_configured"


def test_get_execution_status_maps_shuffle_states(monkeypatch):
    client = ShuffleClient(webhook_url="", base_url="https://shuffle.example", api_key="secret-key")
    monkeypatch.setattr("requests.get", lambda *a, **kw: _FakeResponse(200, json_body={"status": "FINISHED"}))
    result = client.get_execution_status("wf1", "ex1")
    assert result.outcome == "success"


def test_get_execution_status_never_leaks_api_key_in_headers_assertion(monkeypatch):
    client = ShuffleClient(webhook_url="", base_url="https://shuffle.example", api_key="super-secret")
    captured = {}

    def _fake_get(url, headers=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse(200, json_body={"status": "EXECUTING"})

    monkeypatch.setattr("requests.get", _fake_get)
    result = client.get_execution_status("wf1", "ex1")
    assert result.outcome == "running"
    assert captured["headers"]["Authorization"] == "Bearer super-secret"


def test_health_check_false_without_base_url():
    client = ShuffleClient(webhook_url="https://shuffle.example/hooks/abc")
    assert client.health_check() is False
