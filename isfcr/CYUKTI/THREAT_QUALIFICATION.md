# CYUKTI Threat Qualification

## What counts as a threat

CYUKTI distinguishes three tiers, not a binary:

- **NOT_THREAT** — essentially no corroborating signal (CTI confidence score < 20).
- **SUSPICIOUS** — some real signal, not enough to act on automatically (score 20–39.99).
- **QUALIFIED_THREAT** — score ≥ 40 (`PUBLISH_THRESHOLD`, `cti_confidence_engine.py`).

**MITRE mapping alone is never sufficient.** A resolved technique describes *behavior*, not intent — `cti_confidence_engine.CTIConfidence` blends detection confidence, dynamic risk, threat-intelligence confidence, campaign consistency, and prediction confidence (weights: 0.25/0.20/0.20/0.20/0.15) into the score that actually decides the tier.

## What already existed vs. what this phase added

`misp_sync.should_publish()` already gated live MISP publication on `CTIConfidence.publish` (`score >= 40`) — CYUKTI has **never** sent every alert or every campaign to MISP; this was a real, working, already-tested gate (`test_misp_integration.py`) that a previous audit phase (`FULL_SYSTEM_INTEGRATION_AUDIT.md`) had not surfaced. This phase does not replace or touch that gate — rule 16 ("preserve existing CYUKTI behavior").

Added this phase, purely additively:
1. `CTIConfidence.threat_classification` — the same score, now also labeled as one of the three tiers above (instead of collapsing NOT_THREAT and SUSPICIOUS into the same `publish=False`). `publish`'s own value/condition is unchanged and covered by a regression test (`test_cti_confidence_classification.py`).
2. `threat_qualification.ThreatQualificationEngine` — a separate, more granular, explainable layer on top, producing a checklist (not one opaque boolean):
   - `threat_classification_qualified` — is the tier `QUALIFIED_THREAT`?
   - `has_ioc` — is there an attacker IP?
   - `valid_mitre_provenance` — is the technique resolved (not `UNKNOWN`)?
   - `valid_timestamp` — is there a timestamp?
   - `has_campaign_context` — is there a campaign ID?
   - `not_already_published` — has this campaign already produced a MISP event?

   `may_publish_to_misp` is true only if **all six** pass. This is consumed by `soar.response_plan.ResponsePlanGenerator` and `GET /api/threat-qualification/<campaign_id>` — **not** by the live publish path itself (see below for why).

## Why the live gate wasn't rewired to use the new checklist

`should_publish()`'s existing gate is exercised by ~20 passing tests built in a prior phase and is the thing actually running in the live alert pipeline right now. Swapping it for the new, stricter six-point checklist risked silently changing which real campaigns get published without a chance to observe that in this environment first (Neo4j/MISP weren't both live at the same time during this phase to verify end-to-end). The new engine is deliberately additive and dashboard/response-plan-facing; wiring it into the live gate is a reasonable, low-effort next step once its behavior has been observed against real campaigns for a while (see `MISP_PUBLICATION_POLICY.md`'s "Exact next action").

## Retroactive qualification for an already-resolved campaign

`GET /api/threat-qualification/<campaign_id>` reconstructs a qualification view for a campaign that's already been through the live pipeline, using **real, persisted** state (`Campaign.cti_score`/`cti_publish`, written by `neo4j_client.store_cti_confidence` during live alert processing) rather than requiring a live, in-memory `IncidentContext` (which only exists transiently during `realtime_socgraph.py`'s own processing and is never itself persisted). `threat_classification` is re-derived from the stored score using the exact same thresholds as `cti_confidence_engine.py` — not a new heuristic.

**Known limitation**: a campaign that never went through live CTI-confidence computation (e.g. seeded directly into Neo4j, or processed before this field existed) has no stored `cti_score` — the endpoint honestly reports `NOT_THREAT` / "no CTI confidence computed yet" rather than fabricating a score.
*Planned:* add a backfill utility that runs `cti_confidence_engine.py` against such campaigns on demand so this UNKNOWN state can be resolved without waiting for a new live event.

## Dashboard (added this phase)

`IncidentView.tsx`'s "Threat Qualification" section renders all six checks with an explicit PASS/FAIL/UNKNOWN state per check (not just true/false) — `UNKNOWN` is shown specifically for `threat_classification_qualified` when `cti_score` is `null` (no CTI confidence computed yet), distinguishing "we checked and it failed" from "we don't know yet," per the explicit "do not hide uncertainty" requirement. Every other check remains a real PASS/FAIL since it's checking presence of a concrete field, not a probabilistic judgment.

Also composed into `GET /api/incidents/<id>/overview` and `GET /api/incidents/<id>/response-plan` via a shared `_build_threat_qualification()` helper (one implementation, reused by the standalone route and both aggregate endpoints).

## Tests

`test_cti_confidence_classification.py` (4), `test_threat_qualification.py` (10), `test_dashboard_api_threat_and_selection_routes.py`'s threat-qualification cases (4), `test_dashboard_api_incident_views.py` (7, covering the aggregate endpoints).
