# CYUKTI Configuration Snapshot

`backend/config_snapshot.py`, exposed via `GET /api/system/config-snapshot`.

## Design: allowlist, not denylist

`_SAFE_SCALAR_FIELDS` names exactly which `config.py` attributes are safe to report verbatim (`NEO4J_URI`, `MISP_URL`, `GNN_ENABLED`, `GNN_MODEL_PATH`, timeouts/windows, `LOG_LEVEL`). A new secret added to `config.py` in the future is excluded from the snapshot **by default** — it would have to be explicitly added to the allowlist to ever appear, rather than requiring someone to remember to add it to an exclusion list. `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `MISP_API_KEY`, `SHUFFLE_API_KEY` are never in the allowlist.

`MISP_API_KEY` / `SHUFFLE_WEBHOOK` / `SHUFFLE_API_KEY` are reported only as booleans (`misp_credential_configured`, `shuffle_webhook_configured`, `shuffle_api_configured`) — `SHUFFLE_WEBHOOK`'s URL path segment is a Shuffle-generated hook identifier, effectively a bearer token for that specific workflow, so even the URL itself is treated as sensitive, not just an API key field.

## Example (this environment, credentials-configured state omitted deliberately)

```json
{
  "NEO4J_URI": "bolt://localhost:7687",
  "MISP_URL": "https://localhost:8443",
  "GNN_ENABLED": true,
  "CAMPAIGN_TIMEOUT": 120,
  "misp_credential_configured": true,
  "shuffle_webhook_configured": false,
  "shuffle_api_configured": false,
  "software_versions": { "python": "3.12.3", "torch": "2.10.0", "...": "..." }
}
```

## Tests

`backend/tests/test_config_snapshot.py` (7): asserts credential field *names* never appear in the snapshot's keys, asserts real credential *values* injected via monkeypatch never appear anywhere in the serialized JSON (not just checking the allowlisted keys — a full-string search), and verifies the mechanism degrades honestly (an uninstalled package reports `None`, a removed config field is silently omitted, never a crash).
