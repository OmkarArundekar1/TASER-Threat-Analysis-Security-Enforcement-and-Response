# CYUKTI — Phase X: Active Containment & Closed-Loop Response Report

> 2026-09-26. Extends CYUKTI from DETECT→...→RECOMMEND to include CONTAIN→VERIFY→AUDIT→LEARN. **This phase does not touch, promote, or reinterpret any evaluation-framework result** — MITRE/attribution/campaign-correlation/threat-qualification remain exactly `MEASURED_PRELIMINARY`, prediction remains `UNMEASURABLE`, and nothing was locked or human-reviewed. Verified via `git status`/`git diff` before committing (Section S).

## A. Executive summary

Built a new `backend/active_response/` package (policy engine, decision model, explicit state machine, allowlisted containment actions, firewall abstraction, client response agent, independent containment verifier, rollback/expiry, audit logging, response metrics) plus a backward-compatible extension of the existing `soar/response_plan.py::ResponsePlan`. **No live containment was executed against any real system this session** — the architecture, decision logic, safety gates, and failure paths are real and tested (76 new tests, 838/838 backend passing); actual enforcement against the lab's Kali/Ubuntu VMs is `BLOCKED_BY_ENVIRONMENT` because this session has no shell access to either VM. `>>> ATTACK BLOCKED <<<` cannot appear anywhere in this codebase without passing through `ContainmentVerifier.verify()` returning `VerificationStatus.VERIFIED`, which itself requires real evidence, not an HTTP-200-style success signal (structurally enforced and tested — see Section I).

## B. Files created/modified

**New**: `backend/active_response/{__init__,containment_actions,decision,policy,state_machine,firewall_backend,client_agent,verification,rollback,audit,metrics}.py` (11 modules); `backend/tests/active_response/` (10 test files, 76 tests).
**Modified**: `backend/config.py` (+`AUTO_CONTAIN`, `+CONTAINMENT_NEVER_BLOCK_IPS`, both env-driven, both default to the safe/off state); `backend/soar/response_plan.py` (10 new optional fields on `ResponsePlan`, `generate()` gained 6 new optional kwargs — verified backward-compatible against all 4 existing test files that call it); `backend/soar/api.py` (+1 route, `GET /api/soar/response-audit/<correlation_id>`).
**Untouched**: `evaluation/`, all `review/*.md` evaluation docs, `backend/soar/{generator,matcher,adapter,memory,execution_service,schema,shuffle_client}.py` (reused as-is, not modified), the entire detection/correlation/investigation pipeline (`realtime_socgraph.py`, `campaign_manager.py`, `investigation/*.py`, etc.).

## C. Response architecture

```
ResponsePolicyEngine.decide(PolicyInput) -> ResponseDecision
                                                |
                    (if CONTAIN)                v
                          soar.execution_service.PlaybookExecutionService
                          (REUSED AS-IS -- already fails closed on
                           unconfigured Shuffle, see Section H)
                                                |
                                                v
                          ClientResponseAgent.handle(ContainmentRequest)
                          -> FirewallBackend.block() + independent
                             is_blocked() re-check -> ContainmentResult
                                                |
                                                v
                          ContainmentVerifier.verify(...) -> ContainmentVerification
                          (the ONLY function allowed to return VERIFIED)
                                                |
                                                v
                          active_response.audit.log_response_event(...)
                          (reuses soar.memory's existing SQLite store)
```

`ResponsePlan` (existing, extended) sits alongside this as the analyst-facing narrative — it *describes* a `ResponseDecision`, it never makes one.

## D. Response state machine

`ResponseState`: `OBSERVE, INVESTIGATE, RECOMMEND, CONTAIN, VERIFY, VERIFIED, FAILED, ROLLED_BACK, EXPIRED`. Adjacency list in `state_machine.py::_VALID_TRANSITIONS` — structurally, there is no edge into `VERIFIED` except from `VERIFY`, and no edge into `VERIFY` except from `CONTAIN`; `CONTAIN -> VERIFIED` directly raises `InvalidStateTransition` (tested: `test_cannot_jump_directly_from_contain_to_verified`). Transitions into `VERIFY`/`VERIFIED` additionally require a non-empty `evidence` string or raise `ValueError` (tested). **IMPLEMENTED, TESTED.**

## E. Policy decisions

`ResponsePolicyEngine._evaluate()` gates, in order: threat_class (NOT_THREAT/None → OBSERVE) → evidence_sufficient (False → INVESTIGATE) → investigation_confidence >= 0.5 (below → RECOMMEND) → threat_class == SUSPICIOUS (→ RECOMMEND, never auto-contains regardless of confidence) → requested_action present and executable (→ RECOMMEND if not) → source IP not on `CONTAINMENT_NEVER_BLOCK_IPS` (→ RECOMMEND if it is) → `AUTO_CONTAIN` config true (→ RECOMMEND if false) → **CONTAIN**. Severity/risk_level is carried on the decision for audit purposes but is never itself a gate — `test_severity_alone_is_not_sufficient_for_containment` asserts CRITICAL severity with low confidence/insufficient evidence still does not reach CONTAIN. **IMPLEMENTED, TESTED (13 tests in `test_policy.py`).**

## F. Containment action allowlist

`ContainmentAction` enum has 5 members; `EXECUTABLE_ACTIONS = frozenset({BLOCK_SOURCE_IP})` is the only set any code may consult before calling a firewall backend (`is_executable()`). `ISOLATE_HOST`, `BLOCK_NETWORK_FLOW`, `DISABLE_COMPROMISED_ACCOUNT`, `TERMINATE_MALICIOUS_SESSION` exist in the enum (declared future scope) but are asserted non-executable by `test_future_actions_exist_as_enum_but_are_declared_unsupported`, and the policy engine independently downgrades any request for one of them to RECOMMEND (`test_unsupported_action_is_never_auto_selected_for_containment`). No arbitrary-command field exists anywhere (`ContainmentRequest` has no `command`/`cmd`/`executable`/`script` field — statically asserted by `test_containment_request_has_no_command_or_executable_path_field`). **IMPLEMENTED, TESTED.**

## G. Client response mechanism

`ClientResponseAgent.handle()` performs, in order: authenticate (constant-time HMAC comparison, fails closed on empty/missing/wrong token) → correlation_id present → duplicate-correlation-id check → action-on-allowlist check → source-not-on-never-block-list check → IP-format validation → `firewall.block()` → **independent** `firewall.is_blocked()` re-check (never trusts `block()`'s own return value alone) → structured `ContainmentResult`. Every one of these gates has a dedicated failing-case test in `test_client_agent.py` (11 tests) and `test_failure_paths.py` (4 tests). Never accepts a raw command, never exposes an unauthenticated path (the `Authenticator` is mandatory in the constructor), never bypasses the allowlist. **IMPLEMENTED, TESTED. LIVE-VERIFIED only against `InMemoryFirewallBackend`** (a real, stateful rule table, not a stub) — never against a real host firewall this session (Section T).

## H. Shuffle integration status

**Reused, not rebuilt.** `soar.execution_service.PlaybookExecutionService` already implements exactly the contract Section 6 asks for: `ShuffleTriggerOutcome.NOT_CONFIGURED` → the execution is marked `FAILED` with an honest per-action error (`"Shuffle is not configured..."`), never silently succeeds. This was true before this phase and remains true — `SHUFFLE_WEBHOOK` is configured with a real URL (per `journal_ready_data.md` Section 19.7) but has never been live-triggered, in this phase or any prior one. **Status: BLOCKED_BY_ENVIRONMENT for live execution; the fail-closed contract itself is IMPLEMENTED, TESTED (pre-existing, `test_soar_execution_service.py`).**

## I. Verification mechanism

`ContainmentVerifier.verify()` is the single function in the entire new package allowed to return `VerificationStatus.VERIFIED`, and only when `pre_attack_reachable=True` AND `post_attack_reachable=False` AND the firewall backend's own independent `is_blocked()` also confirms it. Critically, **firewall-state confirmation alone (no real connection-test evidence) returns `INSUFFICIENT_EVIDENCE`, never `VERIFIED`** — `test_no_connection_evidence_but_firewall_confirms_is_insufficient_not_verified` is the test that directly encodes Section 10's core requirement ("never convert uncertainty into success"). A later, independent verification call is never a cached copy of the execution-time result (`test_execution_succeeded_but_later_independent_verification_disagrees`). **IMPLEMENTED, TESTED. The strongest evidence path (a real attacker-side connection attempt before/after) is `BLOCKED_BY_ENVIRONMENT`** — this session cannot drive a real connection attempt from the Kali VM (no shell access to it).

## J. Correlation-ID trace

`correlation_id` is a first-class field on `ResponseDecision`, `ResponsePlan` (new), `ContainmentRequest`, `ContainmentResult`, `ContainmentVerification`, `RollbackRecord`, and every `active_response.audit.log_response_event()` call — `audit_trail_for_correlation()` reconstructs the full response-side lifecycle for one ID (tested end-to-end in `test_audit.py`, exposed via the new `GET /api/soar/response-audit/<correlation_id>` route, tested in `test_response_audit_api.py`). **Honest limitation, not fabricated as done**: this phase did **not** modify the upstream detection pipeline (`realtime_socgraph.py`, `campaign_manager.py`, `investigation/*.py`) to generate or thread a `correlation_id` from the original Wazuh alert onward — doing so would touch live production ingestion code, which was out of this phase's stated scope and carries real regression risk this late in the session. The practical bridge today: `campaign_id` (already a stable, real identifier threaded through the existing pipeline) is the natural `correlation_id` value a caller would pass in when invoking the policy engine from real campaign data — demonstrated in `test_response_plan_extension.py`, not yet wired as an automatic default anywhere in the live pipeline. **IMPLEMENTED + TESTED from ResponseDecision downstream; NOT_MEASURED/deferred upstream of it.**

## K. Audit design

Reuses `soar.memory.memory_store`'s existing SQLite `soar_audit_events` table (no schema migration, no second database) — `active_response.audit.log_response_event()` packs the fields that table doesn't have first-class columns for (`correlation_id`, `decision_id`, `response_plan_id`, `incident_id`, `attack_event_id`) into the existing `detail` JSON column. `_scrub()` strips any key matching `password/api_key/secret/token/credential` (case-insensitive) before writing, tested (`test_log_response_event_never_persists_secret_looking_keys`). **IMPLEMENTED, TESTED.**

## L. Rollback/expiry

`RollbackManager.rollback()` calls `agent.rollback()` (unblock + independent `is_blocked()` re-check) and only sets `verified_rolled_back=True` when that re-check confirms the IP is no longer blocked — never inferred from `unblock()`'s own success flag alone. `is_expired()` is a pure function against a real TTL timestamp. **IMPLEMENTED, TESTED (8 tests across `test_rollback.py`/`test_client_agent.py`/`test_failure_paths.py`).**

## M. Tests

**838/838 backend tests passing** (was 762 before this phase; **+76 new**, 0 regressions, 0 skipped). New test files: `test_containment_actions.py` (4), `test_state_machine.py` (10), `test_policy.py` (13), `test_firewall_backend.py` (7), `test_client_agent.py` (11), `test_verification.py` (6), `test_metrics.py` (5), `test_rollback.py` (4), `test_audit.py` (2), `test_safety_audit.py` (4), `test_response_plan_extension.py` (4), `test_failure_paths.py` (4), `test_response_audit_api.py` (2) = 76. Frontend tests not re-run (no frontend files touched).

## N. Live-lab validation

**`ACTIVE_RESPONSE_LIVE_TEST = BLOCKED_BY_ENVIRONMENT`.** This session has shell access only to the Wazuh-manager/backend host, not to the separate Kali attacker VM or Ubuntu victim VM consoles (consistent with every prior phase's own disclosure of this same boundary). No real attack was simulated and presented as live; no firewall command was executed against any real host, lab or otherwise — `InMemoryFirewallBackend` (a real, stateful, non-fake rule table) was used for every test in this phase. `IptablesFirewallBackend` exists, is code-complete, and is unit-tested with `subprocess.run` mocked, but has never been invoked for real.

## O. Failure experiments

All of Section 16's list are covered except two not applicable to this architecture as designed: "missing campaign ID"/"missing operation ID" are optional audit metadata on `ResponseDecision`, not required inputs to the policy engine (the engine's actual required inputs — `threat_class`, `evidence_sufficient`, `investigation_confidence` — are exercised by dedicated missing/insufficient tests). Covered: agent unavailable → N/A this phase (no network service was stood up; the agent is a library called in-process, matching this phase's scope); Shuffle unavailable → reused, pre-existing, tested; invalid source IP → tested; invalid/unsupported containment action → tested; unauthorized request → tested; expired request → tested; firewall failure → tested; verification failure (both `NOT_VERIFIED` and `INSUFFICIENT_EVIDENCE`) → tested; duplicate request → tested; malformed request → tested; missing correlation ID → tested; insufficient evidence → tested; NOT_THREAT → tested; SUSPICIOUS → tested; `AUTO_CONTAIN=false` → tested; allowlisted source → tested; execution-succeeds-but-later-verification-fails → tested. **No test path produces a false "ATTACK BLOCKED."**

## P. Response latency metrics

`active_response/metrics.py`'s `latency_stats()`/`rate_stats()`/`compute_lifecycle_latencies()` are real, tested (5 tests) pure functions — p50/p95/p99 via linear-interpolation percentiles, rates via `evaluation_metrics.wilson_confidence_interval` (reused, not duplicated). **Status: NOT_MEASURED against real data** — zero real containment actions have ever executed in this environment, so there are zero real timestamp pairs to compute a latency from. Reported honestly as `NOT_MEASURED` with the reason ("no observations recorded yet"), never a fabricated number.

## Q. Response-memory integration

Light integration via the shared audit trail (Section K): querying `audit_trail_for_correlation()` for a past incident's `correlation_id` returns its full response history, which is the retrieval mechanism Section 18 asks for. **Not done this phase**: a dedicated `PlaybookEffectiveness`-style aggregate specifically over containment-verification outcomes (as opposed to Shuffle-execution outcomes, which `soar.memory.PlaybookMemoryStore.effectiveness()` already covers) — flagged as a real, disclosed follow-up rather than fabricated as complete. No failed/unverified action is silently recorded as a successful playbook memory: `soar.memory`'s existing `effectiveness()` only counts `ExecutionStatus.SUCCESS` as a success, and this phase's own audit events log `CONTAINMENT_FAILED`/verification outcomes as distinct, separately-queryable event types, never overwriting or reclassifying them.

## R. Dashboard integration

**Backend only, this phase**: `GET /api/soar/response-audit/<correlation_id>` (new), tested, returns the full structured audit trail a dashboard component would render. **No new frontend component was built** — this is a disclosed scope decision (this phase's budget went to the response architecture and its test coverage), not a claim of visual integration. The response states (`REQUESTED`/`EXECUTING`/`VERIFIED`/`FAILED`/`NOT_VERIFIED`/`BLOCKED_BY_ENVIRONMENT`) that a future dashboard view must visually distinguish are already the real, structured values (`ContainmentResult.containment_status`, `ContainmentVerification.status`) the new API route returns — the data contract is ready; the UI is not built. **NOT_MEASURED / deferred.**

## S. Security/safety audit

`test_safety_audit.py` (4 tests, static analysis over the package's own source): no `os.system`, `shell=True`, `eval`, `exec`, `paramiko`, or bare `subprocess.Popen` anywhere in `active_response/`; `subprocess.run` appears only in `firewall_backend.py`'s fixed-argument-list iptables calls; `ContainmentRequest` has no command/executable/script field; no `/execute?command=` pattern string exists anywhere. `git status`/`git diff` reviewed before committing: no secrets, no credentials, no historical evaluation document touched, no evaluation number changed (Section §21 verified directly — `git diff review/ evaluation/` shows zero lines changed by this phase).

## T. Environment limitations

1. No shell access to the Kali attacker VM or Ubuntu victim VM from this session — blocks Sections N, I's strongest evidence path, and any real `IptablesFirewallBackend` exercise.
2. Shuffle has never been live-triggered in this project's history (pre-existing condition, unchanged by this phase).
3. No frontend build/UI work was attempted (scope decision, Section R).
4. Correlation-ID generation is not yet wired into the live detection pipeline upstream of `ResponseDecision` (Section J).

## U. Research limitations

Response metrics are entirely `NOT_MEASURED` against real data (zero real executions exist to measure). The policy engine's `MIN_INVESTIGATION_CONFIDENCE_FOR_CONTAIN = 0.5` threshold is a reasonable, documented default, not empirically derived or validated against any labeled outcome set — a real, disclosed limitation for any paper claim about this specific number.

## V. Exact status of every capability

| Capability | Status |
|---|---|
| ResponsePolicyEngine | IMPLEMENTED, TESTED |
| ResponseDecision model | IMPLEMENTED, TESTED |
| Containment action allowlist | IMPLEMENTED, TESTED |
| Response state machine | IMPLEMENTED, TESTED |
| ResponsePlan extension | IMPLEMENTED, TESTED |
| Shuffle integration (fail-closed contract) | IMPLEMENTED, TESTED (reused pre-existing code) — live execution BLOCKED_BY_ENVIRONMENT |
| ClientResponseAgent | IMPLEMENTED, TESTED against InMemoryFirewallBackend only |
| BLOCK_SOURCE_IP (in-memory) | IMPLEMENTED, TESTED, LIVE-VERIFIED (against the real in-process rule table, not a stub) |
| BLOCK_SOURCE_IP (real iptables) | IMPLEMENTED (code), TESTED (mocked subprocess) — BLOCKED_BY_ENVIRONMENT for live use |
| ContainmentVerification | IMPLEMENTED, TESTED — strongest (connection-test) evidence path BLOCKED_BY_ENVIRONMENT |
| Correlation ID (decision-downstream) | IMPLEMENTED, TESTED |
| Correlation ID (pipeline-upstream) | NOT_MEASURED / deferred |
| Audit logging | IMPLEMENTED, TESTED |
| Rollback/expiry | IMPLEMENTED, TESTED |
| Response metrics | IMPLEMENTED, TESTED (against synthetic data) — NOT_MEASURED against real data |
| Response memory (audit-trail retrieval) | IMPLEMENTED, TESTED — deeper effectiveness-aggregate integration NOT done |
| Dashboard (API layer) | IMPLEMENTED, TESTED |
| Dashboard (UI) | NOT_MEASURED / deferred |
| Controlled lab demonstration | BLOCKED_BY_ENVIRONMENT |
| Evaluation-framework truth status | UNCHANGED (verified via git diff) |
