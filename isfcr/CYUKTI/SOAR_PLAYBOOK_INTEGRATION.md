# CYUKTI SOAR / Playbook Execution and Memory Layer

Companion to `FULL_SYSTEM_INTEGRATION_AUDIT.md` and `GNN_PRODUCTION_INTEGRATION.md`
-- same rigor: real code, real tests, honest about what's live-verified
versus what's built-but-blocked by missing infrastructure.

## 1. Architecture

```
WAZUH -> CYUKTI INGESTION -> MITRE+IOC+DEDUP -> CAMPAIGN -> OPERATION
   -> NEO4J GRAPH -> RISK+PREDICTION+ATTRIBUTION
   -> EVIDENCE-AWARE INVESTIGATION -> NBE -> MULTI-RAG
   -> [NEW] PLAYBOOK GENERATOR -> [NEW] PLAYBOOK MEMORY <-------------------+
   -> [NEW] SHUFFLE -> WORKFLOW EXECUTION -> ACTION RESULTS                |
   -> [NEW] EXECUTION OUTCOME (persisted back into Playbook Memory) -------+
   -> [NEW] EFFECTIVENESS -> FUTURE SIMILAR INCIDENT
   -> [NEW] HISTORICAL PLAYBOOK RETRIEVAL (PlaybookMatcher)
   -> [NEW] ADAPTATION (PlaybookAdaptation) -> APPROVAL -> SHUFFLE
```

**Principle preserved throughout**: CYUKTI is the intelligence/memory
layer; Shuffle only executes and reports results back. Nothing in
Playbook Memory is derived FROM Shuffle except execution status/output
-- the playbook definition, the matching signals, and the adaptation
logic are all CYUKTI's own.

## 2. Phase 1 audit findings (what already existed)

- **`response/playbook_generator.py` / `response/mitigator.py`**: proven
  dead Generation-1 code (`GENERATION1_DISPOSITION.md`, "C — dead",
  zero real consumers). Keyed on synthetic `severity_label`/`attack_label`
  strings that don't exist anywhere in CYUKTI's real pipeline;
  `Mitigator.execute_playbook()` runs raw `iptables`/shell commands
  directly via `subprocess` with no approval gate at all — exactly the
  un-auditable execution model this new layer replaces with Shuffle +
  an explicit approval gate. Not reused, not deleted (per the standing
  Generation-1 preservation rule), not duplicated: the new
  `soar.generator.PlaybookGenerator` is a different class in a
  different package, built on the real `CampaignContext`/MITRE technique
  schema instead.
- **Shuffle integration**: `realtime_socgraph.py` already POSTs a
  payload to `SHUFFLE_WEBHOOK` after every alert (line ~730), but the
  module hardcodes `SHUFFLE_WEBHOOK = ""` at line 76 — it **never**
  reads `config.SHUFFLE_WEBHOOK`. This existing "integration" has
  always silently no-op'd (POST to an empty string, caught by its own
  `except`). Left untouched (out of scope to fix a Generation-2 file
  for this phase beyond the new SOAR layer; noted here for accuracy
  rather than silently building over it).
- **Where the final investigation result becomes available**: 
  `dashboard_api.py`'s `/api/investigate/<campaign_id>` route, after
  `investigation.loop.run_investigation()` returns. `soar/generator.py`
  takes the same `CampaignContext` this route already loads
  (`_load_campaign_context`) as its primary input, so playbook
  generation is a natural next step from the same data, not a
  redundant re-fetch.

## 3. Files created

```
backend/soar/__init__.py
backend/soar/schema.py              -- Playbook / PlaybookAction / PlaybookExecution /
                                        PlaybookActionResult / HistoricalPlaybookMatch /
                                        PlaybookEffectiveness / ExecutionPolicy / ExecutionStatus
backend/soar/memory.py              -- PlaybookMemoryStore (SQLite), audit event log
backend/soar/shuffle_client.py      -- ShuffleClient (webhook trigger + optional REST polling)
backend/soar/generator.py           -- PlaybookGenerator
backend/soar/matcher.py             -- PlaybookMatcher (technique/topology/identity signals)
backend/soar/adapter.py             -- PlaybookAdaptation
backend/soar/execution_service.py   -- PlaybookExecutionService (the state machine + policy gate)
backend/soar/api.py                 -- Flask Blueprint, 12 routes
backend/tests/test_soar_schema.py            (5 tests)
backend/tests/test_soar_memory.py             (10 tests)
backend/tests/test_soar_shuffle_client.py     (9 tests)
backend/tests/test_soar_generator.py          (9 tests)
backend/tests/test_soar_matcher.py            (7 tests)
backend/tests/test_soar_adapter.py            (7 tests)
backend/tests/test_soar_execution_service.py  (13 tests)
backend/tests/test_soar_api.py                (16 tests)
frontend/src/components/SOARPage.tsx
frontend/src/components/SOARPage.test.tsx     (6 tests)
```

## 4. Files modified

- `backend/config.py`: added `SHUFFLE_BASE_URL`, `SHUFFLE_API_KEY` (both empty by default).
- `backend/dashboard_api.py`: registers `soar_bp` blueprint.
- `backend/.gitignore` / `.gitignore`: `backend/soar/playbook_memory.db`, `misp_cache.json` added to ignore list (runtime state, mirrors the existing `logs/` pattern).
- `frontend/src/types/index.ts`: SOAR types.
- `frontend/src/services/api.ts`: 13 new client methods.
- `frontend/src/components/TopNavBar.tsx`, `App.tsx`: new "SOAR" top-level view.

Nothing else was touched. XGBoost, GNN, MISP, existing investigation/evidence code, and all prior tests are unmodified and still pass (see Section 13).

## 5. Playbook schema

See `soar/schema.py` in full. Matches the requested spec exactly:
`Playbook` (playbook_id, name, version, description, trigger_conditions,
campaign_type, mitre_techniques, severity, risk, required_evidence,
actions, execution_policy, shuffle_workflow_id/version, created_at,
updated_at, status) with `PlaybookAction` (action_id, action_type,
name, description, order, inputs, expected_output, destructive,
requires_approval, timeout_seconds) plus a non-fabrication field this
implementation adds: **`reason`** -- every action traces to a real
signal (a MITRE mitigation record, a field actually present on the
campaign, a risk threshold), never left blank.

## 6. Shuffle integration method

Two real, distinct mechanisms (`soar/shuffle_client.py`):

1. **Webhook trigger** (`SHUFFLE_WEBHOOK`) — the only mechanism this
   environment has ever had configured. POSTing JSON starts the
   workflow; if that workflow's webhook trigger is set to "wait for
   response", Shuffle returns the final output inline
   (`SYNCHRONOUS_RESULT`); otherwise it just acknowledges
   (`TRIGGERED`) and the workflow runs server-side.
2. **REST API polling** (`SHUFFLE_BASE_URL` + `SHUFFLE_API_KEY`) —
   `GET /api/v1/workflows/<id>/executions/<execution_id>`, for
   asynchronously finalizing a `TRIGGERED` execution later
   (`PlaybookExecutionService.poll_status`). Not usable in this
   environment (no base URL/API key configured) — reported as
   `not_configured`, never faked.

No endpoint was invented; both are Shuffle's own documented mechanisms.
Credentials only ever come from `config.py`/`.env`, are never logged,
never stored in Playbook Memory, and never appear in any dashboard API
response (verified by `test_soar_shuffle_client.py`'s explicit
API-key-never-leaked assertions).

## 7. Execution lifecycle

```
request_execution(playbook, campaign_id)
  -> RECOMMEND_ONLY:        raises PolicyError -- API never offers /execute for this policy
  -> ANALYST_APPROVAL:      PENDING_APPROVAL  (Shuffle NOT called yet)
  -> AUTOMATIC (no destructive action): -> _trigger() immediately
  -> AUTOMATIC (has destructive action): silently downgraded to ANALYST_APPROVAL
                                          (safety rule enforced in the service, not just the UI)

approve(execution_id) -> _trigger()
reject(execution_id)  -> REJECTED (terminal, Shuffle never called)

_trigger():
  RUNNING -> shuffle_client.trigger(payload)
    NOT_CONFIGURED     -> FAILED (every action result carries the real reason)
    SYNCHRONOUS_RESULT -> SUCCESS immediately, output persisted per action
    TRIGGERED          -> stays RUNNING, shuffle_execution_id recorded
    AUTH_FAILED/TIMEOUT/ERROR -> FAILED / TIMEOUT

poll_status(execution_id):  # only finalizes if SHUFFLE_BASE_URL+KEY configured
  RUNNING + shuffle says "success" -> SUCCESS
  RUNNING + shuffle says "failed"  -> FAILED
  otherwise -> left RUNNING (an unreachable status endpoint is not evidence of failure)
```

Every transition writes one `soar_audit_events` row (Phase 16):
`PLAYBOOK_GENERATED`, `PLAYBOOK_APPROVAL_REQUESTED`, `PLAYBOOK_APPROVED`,
`PLAYBOOK_REJECTED`, `PLAYBOOK_EXECUTION_STARTED`,
`PLAYBOOK_EXECUTION_COMPLETED`, `PLAYBOOK_EXECUTION_FAILED`,
`PLAYBOOK_ADAPTED` -- retrievable per-execution via
`GET /api/soar/executions/<id>` (`audit_events` field).

## 8. Playbook memory design

SQLite (stdlib, no new dependency), not Neo4j: `Playbook ->
PlaybookExecution -> PlaybookActionResult` is a strictly relational
one-to-many-to-many shape with no graph-traversal need of its own.
Campaign/operation/investigation association is stored as plain
columns. `soar/memory.py`'s `PlaybookMemoryStore.effectiveness()`
computes `executions / successful / failed / success_rate /
average_execution_seconds / analyst_approvals / analyst_rejections /
last_execution_at / failure_reasons` directly from stored execution
rows — real aggregation, not fabricated statistics. Mirrors
`misp_cache.py`'s existing pattern of a small, thread-safe, file-backed
store living next to the module that owns it.

## 9. Historical matching design

`PlaybookMatcher.find_matches()` (Phase 7) computes, per candidate
stored playbook, **every signal kept separate**:
- `technique_similarity`: Jaccard overlap of real MITRE technique sets.
- `topology_similarity`: the existing, already-shipped
  `ml.gnn.topology_similarity.gnn_topology_similarity_between_campaigns()`
  — `None` (not zero) when GNN is disabled/unavailable, never fabricated.
- `attacker_ip_match` / `victim_ip_match`: exact identity facts.
- historical execution stats: from `PlaybookMemoryStore`, CYUKTI's own record.

A playbook with zero real signal to the current incident (no technique
overlap, no topology similarity, no shared identity) is excluded
entirely rather than padded in with a fabricated low score.

## 10. Adaptation design

`PlaybookAdaptation.adapt()` (Phase 9) deep-copies the historical
playbook, assigns it a new `playbook_id` with `adapted_from_playbook_id`
lineage, and rewrites only identity-bound action inputs (attacker IP,
victim IP, campaign ID) by the action's own role (`enrich_ip`/`block_ip`
-> current attacker; `collect_evidence`/`isolate_host` -> current
victim) — action semantics (`destructive`, `requires_approval`, `order`)
are never altered.

## 11. Approval / security model

- Destructive action + `AUTOMATIC` policy is **always** downgraded to
  `ANALYST_APPROVAL` inside `PlaybookExecutionService`, not just hidden
  in the UI — verified by
  `test_automatic_policy_with_destructive_action_is_downgraded_to_approval`.
- `RECOMMEND_ONLY` playbooks reject execution attempts with a typed
  `PolicyError` (400), enforced server-side.
- Credentials: see Section 6 — never logged, never in Playbook Memory,
  never in a dashboard response.
- **Not implemented this phase** (honest gap, Phase 17's checklist item):
  duplicate/concurrent-execution prevention for the same
  playbook+campaign pair. Two `/execute` calls in quick succession would
  currently create two independent `PlaybookExecution` records. Given
  the scope of this phase, this was deprioritized versus building the
  full lifecycle end-to-end; a real fix would check
  `memory_store.executions_for_campaign(campaign_id)` for an existing
  non-terminal execution of the same playbook before creating a new one.

## 12. Dashboard changes

New top-level "SOAR" view (`SOARPage.tsx`, wired into `TopNavBar`
alongside the other 6 views added this session) with 5 sub-tabs:
**Recommendations** (campaign picker -> historical matches + a freshly
generated candidate playbook, each action showing a Safe/Approval/
Destructive badge and its real `reason`), **Library** (every stored
playbook, Execute button hidden for `recommend_only`), **Active
Executions** (approve/reject/poll controls), **History** (terminal
executions), **Effectiveness** (per-playbook success rate table).

## 13. API endpoints

```
GET  /api/soar/status
POST /api/soar/playbooks/generate            {campaign_id}
GET  /api/soar/playbooks
GET  /api/soar/playbooks/<id>
POST /api/soar/playbooks/adapt               {source_playbook_id, campaign_id}
POST /api/soar/playbooks/<id>/execute        {campaign_id?}
GET  /api/soar/executions
GET  /api/soar/executions/<id>               (includes audit_events)
POST /api/soar/executions/<id>/poll
POST /api/soar/executions/<id>/approve       {approved_by}
POST /api/soar/executions/<id>/reject        {reason}
GET  /api/soar/recommendations/<campaign_id>
GET  /api/soar/effectiveness
```

## 14. Tests

Backend: 76 new tests across 8 files (schema, memory, shuffle_client,
generator, matcher, adapter, execution_service, api). Full backend
suite re-run: **561 passed, 0 failed** (up from 470 at the start of
this session's larger integration phase). Frontend: 6 new tests
(`SOARPage.test.tsx`), full suite **98/98 passing** (up from 92),
`npm run build` clean.

## 15. Live Shuffle verification (Phase 19)

**SHUFFLE_INFRASTRUCTURE_BLOCKED.** `config.SHUFFLE_WEBHOOK` is empty
in this environment's `.env` — there is no reachable Shuffle instance
to trigger a real harmless workflow against. `ShuffleClient.trigger()`
correctly reports `NOT_CONFIGURED` when called this way (verified: this
is exactly what `test_trigger_marks_failed_when_shuffle_not_configured`
exercises), and `PlaybookExecutionService` marks any such execution
`FAILED` with an honest, real reason string — not fabricated as
successful. No destructive or non-destructive live workflow was
attempted against a real Shuffle instance this phase.

## 16. Historical playbook reuse verification

Not live-verified end-to-end against a real Shuffle execution (blocked
by Section 15), but the **retrieval/adaptation logic itself** is real
and tested against real CYUKTI signals: `test_soar_matcher.py` and
`test_soar_adapter.py` exercise the actual Jaccard/GNN-topology/identity
scoring and the actual field-rewriting logic, not mocks of the
matching/adaptation behavior itself — only Shuffle's HTTP boundary is
mocked anywhere in this test suite.

## 17. Remaining blockers

- **SHUFFLE_INFRASTRUCTURE_BLOCKED** (Section 15) — no reachable Shuffle instance/webhook configured in this environment.
- MISP publish remains credential-pending-restart per `FULL_SYSTEM_INTEGRATION_AUDIT.md` Section 6 (unrelated to this phase, noted for completeness since Playbook `create_incident`/`notify_soc` actions are conceptually adjacent).

## 18. Remaining limitations

- Duplicate/concurrent execution prevention not implemented (Section 11).
- `shuffle_workflow_id`/`shuffle_workflow_version` on `Playbook` are real schema fields but nothing currently populates them automatically — an operator would set them manually per playbook once a specific Shuffle workflow is built for it, since this environment has no Shuffle workflows to introspect.
- Per-action result granularity from Shuffle depends entirely on that workflow's own webhook response shape; a `SYNCHRONOUS_RESULT` with no per-action breakdown is stored as one shared output attached to every action, honestly, not invented as separate results.
- `PlaybookGenerator`'s action catalog (`enrich_ip`, `threat_intel_lookup`, `historical_campaign_search`, `collect_evidence`, `apply_mitigation`, `block_ip`, `isolate_host`, `create_incident`, `notify_soc`) is a generic contract for a Shuffle workflow to interpret — this environment has no configured Shuffle apps to verify those `action_type` strings map to a specific real integration (e.g., a specific firewall API); the workflow-side mapping is Shuffle's responsibility, consistent with the "Shuffle executes, CYUKTI decides" architecture.

## 19. Exact end-to-end data flow

`CampaignContext` (real, resolved) -> `PlaybookGenerator.generate()` ->
`Playbook` (persisted via `PlaybookMemoryStore.save_playbook`) ->
analyst reviews in `SOARPage` "Library"/"Recommendations" tab ->
`POST /execute` -> `PlaybookExecutionService.request_execution()` ->
(approval gate if required) -> `ShuffleClient.trigger()` -> Shuffle
workflow runs -> result persisted as `PlaybookExecution` +
`PlaybookActionResult` rows -> `PlaybookMemoryStore.effectiveness()`
recalculated on next read -> next similar incident's
`PlaybookMatcher.find_matches()` sees this execution's real outcome ->
`PlaybookAdaptation.adapt()` retargets it -> loop closes.

## 20. Exact next action

If/when a real Shuffle instance becomes reachable: set
`SHUFFLE_WEBHOOK` (and optionally `SHUFFLE_BASE_URL`/`SHUFFLE_API_KEY`
for polling) in `backend/.env`, build one harmless test workflow in
Shuffle (e.g. a no-op that just echoes its input back with "wait for
response" enabled), generate a playbook for any real campaign via the
SOAR page, and execute it — that first real
`PLAYBOOK_EXECUTION_COMPLETED` audit event is the point at which
Section 15/16 above can be updated from "blocked" to "live-verified."
