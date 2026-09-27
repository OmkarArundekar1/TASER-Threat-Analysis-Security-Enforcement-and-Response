# Operations View

The Overview's campaign list is now operation-centric: campaigns are
grouped under the real Operation that correlated them, rather than
shown as one flat list of Campaigns. This document covers what's real,
what's verified, and a genuine gap this surfaced in the live-listener
pipeline that this change does **not** attempt to fix.

## Why

`Campaign` and `Operation` are two different, pre-existing real
concepts in the Neo4j graph (`operation_manager.py`, `operation_context.py`,
`operation_decision_engine.py`): a Campaign is a single correlated
attacker→victim burst; an Operation groups multiple Campaigns believed
to be the same actor's ongoing activity (`OperationDecisionEngine`
decides `ATTACH_TO_OPERATION` vs `CREATE_NEW_OPERATION` per campaign
based on attacker/victim/technique/temporal/chain similarity). Before
this change, the frontend only ever showed the flat Campaign list —
Operations existed in the graph and were used in exactly one place
(`operation_id` displayed on the Incident header) but were never a
first-class browsing/grouping unit anywhere in the GUI.

## What's new

**`GET /api/operations`** (`dashboard_api.py`) — lists every real
`Operation` node plus the `campaign_ids` it has an actual `HAS_CAMPAIGN`
relationship to:

```json
{
  "operations": [{
    "operation_id": "OP_C3F42E7B",
    "primary_attacker": "192.168.56.106",
    "primary_target": "pes1ug23cs411-VirtualBox",
    "status": "INACTIVE",
    "created_at": "2026-09-26T10:24:53.776Z",
    "last_seen": "2026-09-26T11:55:14.609Z",
    "closed_at": "2026-09-26T11:57:14.728Z",
    "confidence": 0.0,
    "correlation_score": 0.0,
    "predicted_goal": null,
    "campaign_ids": ["CAMP_2FE069A5", "CAMP_EA30167F", "..."],
    "campaign_count": 8
  }],
  "count": 57
}
```

`campaign_count`/`campaign_ids` are recomputed live from the real
`HAS_CAMPAIGN` relationship on every call (`count(DISTINCT c)`), not
trusted from the Operation node's own stored `campaign_count` property
-- see "The gap" below for why that stored counter can't be trusted as
the source of truth right now.

The frontend (`CampaignIntelligence.tsx`) cross-references this against
the existing, unchanged `/api/campaigns` list client-side: for each
Operation, it shows the campaign_ids that are also in the current
(already time/attacker-filtered) campaigns list as its members. An
Operation with zero currently-visible members (e.g. filtered out) is
not shown at all -- it is never rendered as a misleading empty group.

## The gap this surfaced (real, disclosed, not fixed here)

Querying live Neo4j directly (2026-09-27): **154 Campaigns exist, but
only 66 are covered by an Operation's `HAS_CAMPAIGN` relationship — 88
(roughly 60%) have no Operation at all**, including campaigns created
minutes before this was checked. This is not stale/legacy data from
before Operations existed; it is an ongoing gap in the live pipeline.

`realtime_socgraph.py`'s alert-processing loop does call
`attach_campaign_to_operation()` / `create_operation_db()`
unconditionally for every campaign it processes (lines ~674-701) — on
paper, every campaign should get an Operation. But the listener's own
log (`backend/logs/prerana_listener.log`) never once shows this code
path's own `"[OPERATION] Attached campaign ..."` / `"[OPERATION]
Created ..."` print statements, across the entire log history. Whatever
is happening (an exception being swallowed upstream, a stale running
process, or something else) has not been root-caused as part of this
change — that investigation, and any fix to the real-time
operation-attachment pipeline, is intentionally out of scope here and
tracked separately. This document exists specifically so that gap
isn't silently rediscovered later without context.

**Because of this gap, the frontend never assumes every campaign has an
Operation.** Any campaign_id absent from every Operation's
`campaign_ids` renders as its own entry in the Overview list, using its
own real data, clearly a single campaign rather than a fabricated
Operation — it is never hidden and never given a synthetic
`operation_id`.

## Tests

`backend/tests/test_dashboard_api_routes.py` — mocked-tier tests for
the response contract, the empty-result case, and the `[None]`-filtering
edge case from the `OPTIONAL MATCH` when an Operation has no attached
campaigns; a live-tier test (`requires_live_neo4j`) that every
campaign_id an operation claims actually exists in `/api/campaigns` (no
phantom references), skipped rather than failed when Neo4j isn't
reachable, matching this project's established live-test pattern.

`frontend/src/components/CampaignIntelligence.test.tsx` — grouping
behavior (collapsed by default, expand reveals members, selecting a
member calls the existing `selectCampaign`), the ungrouped-campaign
case, and the empty-visible-members case.
