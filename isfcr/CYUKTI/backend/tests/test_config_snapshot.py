"""
Tests for config_snapshot.py -- the explicit-allowlist safe
configuration report (CONFIGURATION_SNAPSHOT.md). The core property
under test: no matter what config.py contains, a credential field
never appears in the snapshot's keys or values.
"""

import config_snapshot
import dashboard_api


def test_snapshot_never_includes_credential_field_names():
    snapshot = config_snapshot.get_safe_config_snapshot()
    forbidden_keys = {"NEO4J_USERNAME", "NEO4J_PASSWORD", "MISP_API_KEY", "SHUFFLE_API_KEY"}
    assert forbidden_keys.isdisjoint(snapshot.keys())


def test_snapshot_never_includes_the_real_credential_values(monkeypatch):
    monkeypatch.setattr("config.MISP_API_KEY", "super-secret-key-value", raising=False)
    monkeypatch.setattr("config.SHUFFLE_API_KEY", "another-secret-token", raising=False)
    monkeypatch.setattr("config.NEO4J_PASSWORD", "db-password-value", raising=False)

    import json
    snapshot = config_snapshot.get_safe_config_snapshot()
    serialized = json.dumps(snapshot)

    assert "super-secret-key-value" not in serialized
    assert "another-secret-token" not in serialized
    assert "db-password-value" not in serialized


def test_snapshot_reports_credential_configured_as_a_boolean_not_the_value():
    import config as real_config
    original = real_config.MISP_API_KEY
    try:
        real_config.MISP_API_KEY = "some-real-key"
        snapshot = config_snapshot.get_safe_config_snapshot()
        assert snapshot["misp_credential_configured"] is True
        assert isinstance(snapshot["misp_credential_configured"], bool)
    finally:
        real_config.MISP_API_KEY = original


def test_snapshot_includes_real_software_versions():
    versions = config_snapshot.get_software_versions()
    assert versions["python"]
    assert "." in versions["python"]  # a real version string like "3.12.3"


def test_snapshot_never_raises_when_a_listed_package_cannot_be_queried(monkeypatch):
    """importlib.metadata.version() raises PackageNotFoundError for an
    uninstalled package -- must be reported as None, never propagate."""
    import importlib.metadata

    def _raise(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)
    versions = config_snapshot.get_software_versions()
    assert versions["flask"] is None
    assert versions["torch"] is None


def test_snapshot_gracefully_omits_a_config_field_that_does_not_exist(monkeypatch):
    import config
    monkeypatch.delattr(config, "GNN_MODEL_PATH", raising=False)
    snapshot = config_snapshot.get_safe_config_snapshot()
    assert "GNN_MODEL_PATH" not in snapshot


def test_dashboard_route_returns_the_snapshot_as_json(monkeypatch):
    dashboard_api.app.config["TESTING"] = True
    with dashboard_api.app.test_client() as client:
        resp = client.get("/api/system/config-snapshot")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "software_versions" in data
        assert "MISP_API_KEY" not in str(data)
