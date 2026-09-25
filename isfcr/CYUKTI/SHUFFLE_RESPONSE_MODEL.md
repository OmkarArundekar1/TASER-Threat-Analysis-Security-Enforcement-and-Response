# CYUKTI Shuffle Response Model

## ResponsePlan (`soar/response_plan.py`)

Composes the analyst-facing "threat detected, here's what should happen" narrative from three already-real pieces, never generating a new opinion of its own:

1. `threat_qualification.ThreatQualificationResult` — why this is/isn't a threat.
2. `campaign_selection.SelectionResult` — which historical campaign matched, and why.
3. `soar.schema.Playbook` (from `soar.generator.PlaybookGenerator`, built in the previous SOAR phase) — what should happen next, with every action carrying its own `reason`, `destructive`, and `requires_approval` flags.

Any of the three inputs may be `None` — a campaign with no CTI confidence computed yet, or no comparable historical campaign found, produces a `ResponsePlan` that honestly says so in `why_threat`/`why_selected`, rather than fabricating a narrative.

```json
{
  "campaign_id": "CAMP_1",
  "threat_summary": "Campaign CAMP_1: attacker 1.2.3.4 -> victim 10.0.0.5, technique(s) T1110, severity HIGH.",
  "why_threat": ["Threat classification: QUALIFIED_THREAT (CTI confidence score 90.0).", "✓ Attacker IP: 1.2.3.4", ...],
  "selected_historical_campaign": "CAMP_OLD",
  "why_selected": "CAMP_OLD ranked highest (composite score 0.87) because its topology similarity (94%) ...",
  "misp_status": "READY",
  "misp_reason": "All publication-readiness checks passed.",
  "playbook": { ... }
}
```

## Shuffle integration (unchanged from the SOAR phase, `soar/shuffle_client.py`)

Two real, documented Shuffle mechanisms, no invented endpoints:

1. **Webhook trigger** (`SHUFFLE_WEBHOOK`) — POST a JSON payload; a "wait for response" workflow returns its final output inline.
2. **REST polling** (`SHUFFLE_BASE_URL` + `SHUFFLE_API_KEY`) — for asynchronously finalizing a triggered-but-not-yet-complete execution.

The conceptual workflow shape (`CYUKTI webhook → validate → enrich IOC → CTI lookup → condition → block/contain/collect/notify if malicious, else record+stop → return result`) is Shuffle's own responsibility to build visually — CYUKTI's payload (the `Playbook`'s ordered `actions`, each with `action_type`, `inputs`, `destructive`, `requires_approval`) gives Shuffle everything it needs to implement that branching without CYUKTI reimplementing a parallel SOAR engine.

## Approval, not blind execution

`soar.execution_service.PlaybookExecutionService` (built in the prior phase) already enforces: a playbook with any destructive action can never execute under `AUTOMATIC` policy — it's silently downgraded to `ANALYST_APPROVAL`, server-side, not just hidden in the UI. This phase's `ResponsePlan` surfaces exactly which actions are safe vs. approval-required vs. destructive so the analyst sees the "what requires approval" answer before clicking anything.

## Built this phase: the unified Incident View

`GET /api/incidents/<campaign_id>/response-plan` now exposes `ResponsePlanGenerator` directly (reuses any already-stored playbook for the campaign, or generates a fresh candidate; composes real threat qualification + campaign selection via the same shared helpers `/api/incidents/<id>/overview` uses). `IncidentView.tsx`'s "Response Plan" section renders it end-to-end: threat summary → why → selected historical campaign + explanation → MISP readiness badge → per-action SAFE/DESTRUCTIVE and AUTO/APPROVAL badges — exactly the incident-view mockup from the original spec, assembled from real, already-tested pieces rather than duplicated logic.

Live-verified against real data: `GET /api/incidents/CAMP_4FDA1A87/response-plan` returned a real generated playbook with real per-action reasons, correctly reported `misp_status: BLOCKED` (this campaign's real CTI score of 29.15 is SUSPICIOUS, not QUALIFIED_THREAT).

## Tests

`test_soar_response_plan.py` (7), `test_dashboard_api_incident_views.py`'s response-plan cases (3), `IncidentView.test.tsx` (5).
