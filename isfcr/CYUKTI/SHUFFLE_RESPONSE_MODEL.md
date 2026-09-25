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

## What's not built this phase

- A dedicated dashboard page rendering `ResponsePlan` end-to-end as the mockup in the original spec shows (threat summary → why → evidence → recommended response → MISP status → Shuffle status, all in one incident view). The underlying data is fully available via `soar.response_plan.ResponsePlanGenerator` and the existing `/api/soar/*` + `/api/threat-qualification/*` + `/api/campaign-selection/*` endpoints; assembling one unified page from them is a reasonable next step given the remaining scope this phase.
- A dashboard API route that calls `ResponsePlanGenerator` directly (it's currently a library function, exercised by `test_soar_response_plan.py`, not yet exposed as its own endpoint — the pieces it composes are each independently exposed already).

## Tests

`test_soar_response_plan.py` (7).
