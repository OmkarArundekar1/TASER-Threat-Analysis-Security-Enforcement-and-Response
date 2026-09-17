# MISP Integration / CTI Publishing

Status before this session: module_status.md rated this **C** —
"`CTIPublisher` initializes successfully... live publish success not
confirmed — `MISP_API_KEY` was empty as of Phase 19." This document
covers what was verified/fixed this session. **No accuracy/quality
claim is made anywhere here** — this is engineering correctness only
(does CYUKTI reliably generate, serialize, publish, and handle MISP
events per its own contract).

## Architecture

```
Investigation (realtime_socgraph.py, per resolved alert)
        |
threat_attribution_engine.engine.attribute(context) -> ThreatAttributionResult
        |
IncidentContext (misp_event_generator.py) -- campaign/operation/technique/
        |         detection/risk/CTI/attribution, assembled per alert
        v
MISPEventGenerator.generate(incident) -> {"Event": {info, distribution,
        |         threat_level_id, analysis, published, date, Attribute, Tag}}
        v
MISPSync.publish_campaign(incident)
        |  should_publish() gates on incident.cti.publish (CTI policy,
        |  computed elsewhere -- this integration doesn't set that policy,
        |  only respects it)
        v
MISPSync.synchronize(incident)  -- 3-tier idempotent lookup:
        |  1. local cache (misp_cache.json: campaign_id -> event_id)
        |  2. if cached, publisher.event_exists() (live GET) -- evict + fall
        |     through to (3) if the cached event was deleted externally
        |  3. publisher.search_campaign() -- MISP-side text search on
        |     "CYUKTI Campaign <campaign_id>" (the same string
        |     MISPEventGenerator sets as the event's `info` field -- an
        |     implicit but real coupling: changing the info format would
        |     silently break re-discovery of existing events)
        |  -> update if found by either path, else create
        v
CTIPublisher.create_event() / update_event()  -- POST to MISP REST API
        v
requests.Session (3 retries, backoff 1.5x, on 429/500/502/503/504;
        NOT retried: 400/401/403 -- auth/client errors aren't retried)
```

There is **no dashboard/API path** for MISP — `dashboard_api.py` has no
MISP references at all. The entire pipeline above runs only inside the
listener's alert-processing flow (`realtime_socgraph.process_alert()`).

## Event generation contract

`MISPEventGenerator.generate()` reads campaign/operation identifiers,
technique/stage, detection/risk/CTI scores, recommendations, and —
when present — the top-ranked `ThreatAttributionEngine` candidate
(actor, score, coverage, precision, chain similarity, rationale
strings) into MISP `Attribute`/`Tag` entries. The only real IOC CYUKTI
currently tracks end to end is the attacker IP
(`realtime_socgraph.extract_iocs()`), which becomes a properly-typed
`ip-src` attribute; the victim IP is carried as a `comment` attribute,
not typed as an IOC (a deliberate distinction — the victim isn't
itself an indicator to action on). No fields were added this session;
the schema is exactly what the existing implementation already
produced.

## Publisher contract

`CTIPublisher._request()` returns one uniform shape for every outcome
— `{"success", "status", "response", "latency"}` — whether the call
succeeded, returned an HTTP error, or raised a `requests` exception
(timeout, connection failure); the exception path never propagates
out of `_request()`. `create_event()`/`update_event()` add `event_id`
extracted from the response.

## Defects found and fixed this session

**1. False-positive success on malformed 2xx responses.**
`create_event()`/`update_event()` reported `success: True` for *any*
2xx HTTP status, even when the response body didn't match MISP's
actual event schema (no `Event.id` on create, no `Event` object on
update) — e.g. a misconfigured reverse proxy, an inline MISP error
wrapped in a 200, or a future API schema change. A caller checking
only `result["success"]` would believe an event was created/updated
when it wasn't. **Fixed**: both methods now downgrade `success` to
`False` when the response body doesn't contain the schema a genuine
success always has. Regression tests:
`test_create_event_2xx_with_malformed_body_does_not_report_success`,
`test_create_event_2xx_with_non_json_body_does_not_report_success`,
`test_update_event_2xx_with_malformed_body_does_not_report_success`.

**2. Ambiguous create/update action labeling.**
`MISPSync.publish_campaign()` inferred which action had run from
`result.get("event_id") is None` — but `update_event()` never set that
key either, so a **create** whose `event_id` failed to parse was
mislabeled `"updated"`. **Fixed**: `_create()`/`_update()` now set
`result["action"]` explicitly at the source (the only place that
unambiguously knows which one actually ran), and
`publish_campaign()` no longer infers it. Regression test:
`test_misp_sync_action_label_is_unambiguous_for_a_create_with_parse_failure`
(this scenario now also correctly reports `success: False`, per fix 1
above, rather than a mislabeled success).

No other defects were found; the retry/cache/error-handling design
otherwise matched its own documented behavior in every scenario
tested.

## Idempotency / duplicate publication

**Intended and verified.** Identity is `campaign_id`, resolved through
the cache → live-existence-check → text-search chain described above.
Verified behaviorally: first publish creates exactly one event and
caches it; a second publish for the same campaign updates that event
(no second `/events/add` call); a stale cache entry (event deleted
externally) is evicted and a fresh lookup performed rather than failing
or looping; a cache miss where MISP already has a matching event (found
via search) updates it rather than duplicating.

**One known, understood, unmitigated edge case** (documented, not
"fixed" — doing so properly would need MISP-side idempotency keys or a
larger architectural change, out of this session's stabilization
scope): `create_event()`'s POST to `/events/add` is included in the
retry-on-5xx set. POST is not inherently idempotent — if MISP actually
processed a create but failed to return a clean response before a
retryable status/timeout, a retry could create a second event. The
existing search-before-create fallback in `synchronize()` provides a
real (if not airtight) safety net for the *next* publish attempt on
the same campaign to converge on one event via `search_campaign()`
rather than duplicating further, but a single retry-storm within one
`create_event()` call is not fully guarded against.

## Security (credential handling)

Verified, not just inspected — `test_api_key_never_appears_in_log_output`
captures all log output (including an exception path) at DEBUG level
around a request built with a real-looking secret string and asserts
it never appears; `test_api_key_never_appears_in_success_or_error_response_dicts`
and `test_generated_misp_event_never_embeds_the_api_key` do the same
for the returned result dict and the generated MISP event payload.
`_headers()` sends the raw key only in the `Authorization` header,
never logged, never embedded in event content. `MISP_API_KEY` is loaded
from environment/`.env` (`config.py`, via `python-dotenv`) — never
hard-coded. `VERIFY_SSL=False` is a deliberate default for the local
self-signed dev MISP instance, not something this session changed.

## Live verification

**LIVE MISP UNAVAILABLE FOR AUTHENTICATED VERIFICATION — CONNECTIVITY
LIVE-VERIFIED, MOCKED INTEGRATION VERIFIED FOR EVERYTHING ELSE.**

This environment's `backend/.env` has `MISP_URL=https://localhost:8443`
reachable and confirmed to be a genuine, live MISP instance (real
login page, MISP's characteristic security headers and static assets).
`MISP_API_KEY` is genuinely empty in this environment (confirmed by
reading `config.MISP_API_KEY`, not assumed) — matching the historical
Phase 19 note.

What was verified for real against the live instance (read-only,
unauthenticated, nothing published): `CTIPublisher.health_check()`,
called against the real server through the real `CTIPublisher`, sent a
real HTTP request and received a real MISP response — `403` with
MISP's actual JSON error body ("Authentication failed. Please make
sure you pass the API key of an API enabled user along in the
Authorization header.") — and `health_check()` correctly returned
`False`, not a false positive. This is genuine live evidence that the
transport layer (TLS, headers, JSON parsing, status handling) works
correctly end to end against a real MISP server, and that the
missing-credential failure path degrades safely and observably rather
than reporting success or crashing.

What could **not** be verified live: a full authenticated
create/search/update round trip (`create_event`, `search_campaign`,
`update_event`, `event_exists`) — these all require a valid MISP
automation key, which this session does not have and must not request
or expose. **Smallest required human action**: populate
`MISP_API_KEY=` in `backend/.env` with a valid MISP automation key
(MISP web UI → user profile → Auth keys) and re-run the live check
below.

Every other publisher/generator/sync behavior in this document is
verified against a mocked HTTP transport
(`tests/test_misp_integration.py`) — genuinely exercising the real
`CTIPublisher`/`MISPSync`/`MISPEventGenerator` implementations, with
only `requests.Session.request` replaced.

To attempt full live verification once a key is configured:

```bash
cd backend
python -c "
from cti_publisher import CTIPublisher
from config import MISP_URL, MISP_API_KEY, VERIFY_MISP_SSL
pub = CTIPublisher(MISP_URL, MISP_API_KEY, VERIFY_MISP_SSL)
print('health_check:', pub.health_check())
"
```

## Tests

`backend/tests/test_misp_integration.py` — 29 tests: event generation
(basic structure, attribution enrichment, campaign/operation
propagation, technique tags, the attacker-IP IOC, JSON serialization),
publisher success/auth-failure/5xx/4xx/timeout/connection-failure/
missing-config/malformed-URL paths, the two fixed defects above,
credential-leakage checks, the full idempotency/dedup matrix (create,
update, stale-cache self-heal, cache-miss-but-MISP-has-it, CTI-policy
gating), and one full-pipeline test using real
`CampaignContext`/`ThreatAttributionEngine`/`MISPEventGenerator`/
`MISPSync`/`CTIPublisher` together with only Neo4j and the MISP HTTP
transport mocked.

Run: `cd backend && python -m pytest tests/test_misp_integration.py -q`
