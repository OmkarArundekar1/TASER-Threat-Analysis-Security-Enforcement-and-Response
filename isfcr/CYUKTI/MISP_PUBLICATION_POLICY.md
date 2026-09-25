# CYUKTI MISP Publication Policy

## What gets sent to MISP

Only campaigns that pass `misp_sync.should_publish()` — gated on `CTIConfidence.publish` (blended detection/risk/threat/campaign/prediction confidence score ≥ 40, `cti_confidence_engine.py`). This is a real, already-implemented gate from a prior phase, exercised by ~20 passing tests — CYUKTI has never published every alert or every campaign to MISP. See `THREAT_QUALIFICATION.md` for the full history of this finding.

## The publication-readiness checklist (this phase, `threat_qualification.py`)

A more granular, explainable layer on top of the score gate, six independently-reported checks:

```
[ ] threat classification = QUALIFIED_THREAT   (score >= 40)
[ ] has_ioc                                    (attacker IP present)
[ ] valid_mitre_provenance                     (technique resolved, not UNKNOWN)
[ ] valid_timestamp                            (timestamp present)
[ ] has_campaign_context                       (campaign ID present)
[ ] not_already_published                      (no existing MISP event for this campaign)
```

`may_publish_to_misp` requires all six. Consumed by `ResponsePlan.misp_status` (`READY`/`BLOCKED`/`NOT_APPLICABLE`) and `GET /api/threat-qualification/<campaign_id>` — **advisory/dashboard-facing**, not yet wired into the live `should_publish()` gate itself (see `THREAT_QUALIFICATION.md` for why, and the next-action note below).

## MISP event content (unchanged from prior phases, `misp_event_generator.py`)

Campaign ID, operation ID, MITRE technique(s), attacker/victim IPs, detection source, CTI confidence, severity, timestamp — never a secret, never unnecessary personal data. `cti_publisher.py`'s `CTIPublisher` never logs or returns the configured API key (verified this session and prior sessions via explicit "key never in response" test assertions).

## Live verification status

- **Threat-gate logic**: verified internally (unit + route tests) using both real stored scores and synthetic edge cases — `BENIGN → BLOCKED`, `SUSPICIOUS → BLOCKED`, `QUALIFIED_THREAT → READY`, exactly matching the required behavior.
- **Live authenticated MISP round trip**: a real MISP admin auth key was configured in `backend/.env` in a prior phase of this session; a live listener process was observed attempting real publishes and getting HTTP 403, traced to a trailing-whitespace bug in the configured key (already fixed, pending a process restart to take effect — see that phase's summary). **Not re-verified live this phase** — Neo4j and a running listener were not both available simultaneously during this specific phase's work to observe a fresh end-to-end publish. No publication success is claimed.
- **API key exposure**: never observed in any response, log line, or Playbook Memory record, in this phase's tests or the prior phase's.

## Exact next action

1. Confirm (via `/api/misp/status`) that the corrected API key authenticates successfully once the backend/listener processes are restarted.
2. Trigger one real qualified-threat campaign through the live pipeline and confirm in `logs/prerana_listener.log` that the MISP publish now returns 200/201, not 403.
3. Once that's confirmed working end-to-end, consider wiring `threat_qualification.engine.qualify()`'s six-point checklist into `misp_sync.should_publish()` as an additional AND-condition (currently deliberately not done — see `THREAT_QUALIFICATION.md`), since by then its behavior will have been observed against real campaigns.
