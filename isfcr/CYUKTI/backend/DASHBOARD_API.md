# Dashboard / API

Status before this session: 22 routes in `dashboard_api.py`, only the
newest evidence-aware endpoints (`/api/investigate`,
`/api/ml/predict/severity`, `/api/rag/mitre/search`) and two prior
bugfixes (`_load_campaign_context`, `/api/query`'s driver-API mismatch)
had dedicated tests. This documents what this session verified/fixed.
**No load/performance testing was performed or is claimed** — that is
explicitly deferred, per the mission for this phase.
*Planned:* extend the benchmark harness to a sustained-load ingestion
test once a suitable traffic generator is in place.

## API inventory

All 22 routes are consumed by the real frontend
(`frontend/src/services/api.ts` is the authoritative list, cross-checked
by `test_all_frontend_consumed_routes_are_registered`) — there are no
dead/unused routes to prune.

| Route | Method | Backend dependency |
|---|---|---|
| `/api/health` | GET | Neo4j connectivity check |
| `/api/overview` | GET | Neo4j |
| `/api/events`, `/api/events/<id>` | GET | Neo4j, `prediction_engine`, `recommendation_engine` |
| `/api/campaigns` | GET | Neo4j |
| `/api/correlation/campaigns/<id>` | GET | Neo4j (standalone Cypher similarity — not `CampaignCorrelationEngine`) |
| `/api/attribution/actors/<id>` | GET | Neo4j (ThreatActor-node mechanism — not `ThreatAttributionEngine`, see THREAT_ATTRIBUTION.md) |
| `/api/risk/propagation/<id>` | GET | Neo4j |
| `/api/campaigns/<id>/timeline` | GET | Neo4j |
| `/api/attackers` | GET | Neo4j |
| `/api/graph/expand`, `/api/graph/paths`, `/api/graph` | GET | Neo4j |
| `/api/query` | POST | Neo4j (read-only Cypher console, mutation keywords blocked) |
| `/api/analytics/paths` | GET | Neo4j |
| `/api/attack-chain` | GET | Neo4j |
| `/api/predictions` | GET | Neo4j, `prediction_engine` |
| `/api/recommendations` | GET | `recommendation_engine`, Neo4j (fallback) |
| `/api/predict` | POST | `prediction_engine` |
| `/api/investigate/<id>` | POST | Real investigation loop (`investigation.loop.run_investigation`) |
| `/api/rag/mitre/search` | POST | Real MITRE STIX corpus (no Neo4j) |
| `/api/ml/predict/severity` | POST | Real trained XGBoost model + Neo4j |

## Defects found (via live testing) and fixed this session

Live Neo4j turned out to be reachable in this environment (a genuine,
useful discovery — see "Live verification" below). Running the real
test client against real data surfaced three real bugs that mocked
tests alone did not catch, because the mocks were built from an
assumed-correct understanding of each callee's signature/query syntax:

**1. No uniform JSON-error contract.** Only the newest endpoints
(`_try_load_campaign_context`) handled a Neo4j outage gracefully;
every other route queried `driver.session()` directly with no
try/except, so an outage (or any other uncaught exception — a
malformed `int` query param, an internal engine bug) fell through to
Flask's default HTML error page. **Fixed** with four app-level
`@app.errorhandler`s (added once, near the top of `dashboard_api.py`,
covering every route uniformly rather than touching 15+ routes
individually):
- `ServiceUnavailable` → 503, `{"error": "Database unavailable: ..."}`
- `Neo4jError` (a **sibling** exception type, not a subclass of
  `ServiceUnavailable` — the driver couldn't reach the server vs. the
  server responded but rejected the query) → 500, `{"error":
  "Database query error: ..."}`. Kept distinct deliberately: conflating
  them would have reported "unavailable" for defect #2 below, which is
  misleading for anyone debugging from the API response alone — this
  is not a hypothetical, it's what actually happened before the fix.
- `HTTPException` (Flask/werkzeug's own aborts — e.g. an unmatched
  route) → JSON with the real status code, not an HTML page.
- Bare `Exception` (anything else) → 500, `{"error": "Internal server
  error"}`. The real exception is logged server-side
  (`logger.exception`); the client never sees a stack trace, internals,
  or secrets.

**2. `/api/attribution/actors/<id>`: invalid Cypher, live-crashing on
every real call.** Its query used `max(shared_tech_count, 1)` as a
two-scalar function — not valid Cypher (`max()` is aggregate-only,
takes one collection/grouped argument). Live-confirmed error:
`Neo.ClientError.Statement.SyntaxError: Too many parameters for
function 'max'`. This was masked before this session: with no error
handling, every real call to this endpoint would have silently
produced an HTML 500, and nothing exercised it against live data
before now. **Fixed** by removing the guard: the preceding `WHERE
shared_tech_count > 0` already guarantees `total_tech_count` (always
≥ `shared_tech_count`, being a union-minus-overlap count) is never 0
at that point, so the division-by-zero protection the `max()` call was
attempting was unneeded, not just invalid.

**3. Three call sites used the wrong `prediction_engine` function.**
`event_detail()`, `predictions()`, and `predict()` all called
`prediction_engine.predict_next(technique)` — but that function's real
signature is `predict_next(campaign_id, technique)`, raising a live
`TypeError` on every call from all three (none of them had a
`campaign_id` in scope in a form suitable to pass, and `/api/predict`'s
request contract has never included one at all). This was more than
an arity bug: `predict_next` also has a **Neo4j write side effect**
(`campaign_manager.update_prediction(...)`, overwriting the campaign's
live `LIKELY_NEXT`/`predicted_next` state) — intentional for the live
investigation loop, but wrong for these three, which are all read-only
lookups (two GETs, one stateless POST). **Fixed** by calling
`prediction_engine.predict_next_readonly(technique)` instead at all
three sites — a function that already existed, purpose-built for
exactly this ("Offline evaluation instead calls this function directly
... without disturbing live state", per its own docstring), just never
wired into the dashboard API.

No other defects were found in the 22 routes; response shapes,
parameter validation, and existing error paths (`/api/risk/propagation`'s
404, `/api/graph/expand`'s 400, `/api/query`'s mutation-keyword
rejection) matched their own documented behavior in every scenario
tested.

## Frontend compatibility

`frontend/src/services/api.ts` was read in full and used as the source
of truth for every request/response shape tested. One real
staleness issue found: `frontend/src/types/index.ts`'s
`EvidenceItem`/`InvestigationStepResult`/`SeverityPrediction`
interfaces were missing fields the backend has returned since the
ML/NBE and Phase 22 sessions (`derived_from`, `why_selected`,
`candidate_actions`, `action_scores`, `previous_confidence`,
`previous_uncertainty`, `candidate_hypotheses`, `top_k`,
`model_metadata`, `prediction_context`). Not a runtime bug (extra
untyped fields are harmless in JS), but a real integration-correctness
gap — TypeScript couldn't express data the backend actually sends, so
no `.tsx` component could type-safely use it. **Fixed**: added the
missing fields as typed, documented interface members (additive only —
no existing field changed or removed, so this cannot break any
existing consumer). No `.tsx` rendering code was touched — this is a
type-contract fix, not a UI change.

One pre-existing, harmless staleness left as-is: `Prediction.empty_reason`
is declared optional in the frontend type but never set by the backend
— since it's optional, this is not a compatibility problem, just an
unused aspirational field.

`npm install`/`tsc` could not be run to mechanically verify the type
file in this session (this environment's npm fails on the WSL UNC
path's `node_modules` — `EPERM`/`EISDIR` errors unrelated to this
change); the edit was verified by inspection (additive interface
fields only, syntactically valid TypeScript).
*Planned:* re-run `tsc` mechanically once `npm install` succeeds in an
environment without the WSL UNC path issue.

## Live verification

**Live Neo4j is reachable in this environment** (71 Campaigns, 130
AttackEvents, 43 Operations at time of testing) — a genuine, useful
discovery this session, not assumed. `tests/test_dashboard_api_routes.py`
uses this for real end-to-end verification wherever possible, each
test `pytest.skip`-ing (not failing) if Neo4j is unreachable, so the
suite stays portable to environments without it — identifiers
(campaign IDs, event IDs) are discovered from the live database at
test time, never hardcoded.

Routes genuinely exercised against live data this session:
`/api/health`, `/api/overview`, `/api/campaigns`, `/api/events`,
`/api/events/<id>` (including the fixed prediction call),
`/api/investigate/<id>` (a real investigation, real evidence, real
confidence state, real JSON), `/api/ml/predict/severity` (the real
production XGBoost artifact), `/api/attribution/actors/<id>` (the
fixed Cypher), `/api/correlation/campaigns/<id>`,
`/api/risk/propagation/<id>`, `/api/attackers`, `/api/graph`,
`/api/graph/expand`, `/api/query`, `/api/predictions`,
`/api/analytics/paths`.

## Test procedure

```bash
cd backend
python -m pytest tests/test_dashboard_api_routes.py -q   # this session's new coverage (65 tests)
python -m pytest tests/test_dashboard_api.py tests/test_dashboard_api_campaign_context.py \
    tests/test_query_console_execution.py tests/test_dashboard_api_routes.py -q  # full dashboard suite
python -m pytest tests/ -q                                 # full backend suite (336 tests)
```

## Remaining limitations

Load/performance/throughput testing is explicitly out of scope for
this phase (deferred, per the mission), not attempted.
*Planned:* extend the benchmark harness to a sustained-load ingestion
test once a suitable traffic generator is in place.

`frontend`
TypeScript compilation could not be mechanically verified in this
environment (see above) — verified by inspection instead.
*Planned:* re-run `tsc` mechanically once `npm install` succeeds in an
environment without the WSL UNC path issue.

Several
Neo4j-heavy routes (`/api/graph/paths`, `/api/analytics/paths`,
`/api/query`'s general Cypher-console surface) received live smoke
verification but not exhaustive parameter-combination coverage, in
line with proportioning effort to value across 22 routes rather than
exhaustively testing every one to the same depth.
*Planned:* expand parameter-combination test coverage for these routes
as time allows, prioritized by real usage.
