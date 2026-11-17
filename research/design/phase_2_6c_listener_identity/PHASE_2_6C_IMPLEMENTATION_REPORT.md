# Phase 2.6C implementation report

Listener alert identity and deduplication remediation. Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Labels:** **[V]** verified from code, **[M]** measured offline, **[T]** reproduced in tests, **[O]** modeled behaviour, **[N]** not yet validated in production.

## Result

**[M] The confirmed alert-loss defect is fixed on the sample.** The unmodified code refuses 70 of 9,015 alerts. The fixed code accepts all 9,015, and no alert with a different Wazuh `id` can be refused because of its content.

**Replay is safe against loss under the tested conditions. It isn't safe against repeats across a restart.** After a restart the duplicate cache is empty, so re-read alerts are delivered and processed again. See `OFFSET_AND_QUEUE_SAFETY.md`.

**[N] The identity contract isn't validated on your deployed Wazuh.** See `PRODUCTION_VALIDATION_GAPS.md`.

**Other alert-loss paths remain,** and are reported rather than fixed: alerts in the memory queue at a crash, and three rotation or file-content cases.

## Acceptance criteria

| Criterion | Status | Evidence |
|---|---|---|
| 70-alert loss reproduced before, prevented after | **Met** | `evidence/replay_before_fix.txt`, `evidence/replay_after_fix.txt` |
| Source identity exact and namespace-aware | **Met** | `D1_SOURCE_IDENTITY_IMPLEMENTATION.md` |
| Different source ids can't be collapsed by content collisions | **Met** | `D2_ALERT_DEDUPLICATION_FIX.md` |
| Missing identity can't silently discard an alert | **Met** | `REPLAY_RESULTS.md` |
| Offset and queue behaviour tested for the crash windows | **Met,** with known gaps reported | `OFFSET_AND_QUEUE_SAFETY.md` |
| First-pass and second-pass results | **Met** | `REPLAY_RESULTS.md` |
| Attributed and unattributed handling correct | **Met in tests, with stubs and a model** | `D2_ALERT_DEDUPLICATION_FIX.md` |
| Existing listener regression tests pass | **Met,** after one fixture edit, disclosed below | `TEST_AND_FAULT_RESULTS.md` |
| Full backend tests run | **Met:** 2,543 passed, 29 skipped, 0 failed | `evidence/test_full_suite_output.txt` |
| No journal, ledger, coordinator or Neo4j writer change | **Met** | `evidence/preservation_audit.txt` |
| Protected evidence and prior archives unchanged | **Met** | `evidence/preservation_audit.txt` |

## Source files changed

| File | Change | sha256 before | sha256 after |
|---|---|---|---|
| `listener/utils.py` | Rewritten: source identity replaces `generate_alert_id` | `evidence/source_hashes_before.txt` | `evidence/source_hashes_after.txt` |
| `listener/wazuh_listener.py` | Identity in the reader and `accept_alert`, one reader fix, atomic offset save | Same | Same |
| `tests/test_listener_alert_acceptance.py` | **Fixture only:** `_alert` gains `id` and `manager` | Same | Same |
| `tests/test_listener_source_identity.py` | New: 157 tests | | Same |

The listener's CRLF line endings are kept. Copies of the code before the change are in `evidence/before_fix/`.

## Identity in one paragraph

[V] `listener/utils.py:extract_source_identity` L57-85 reads only `alert["id"]` and `alert["manager"]["name"]`. A valid alert gets `wazuh:<manager>/<id>`, which `listener/wazuh_listener.py:accept_alert` L128-162 uses as its duplicate key. An alert with no usable identity gets no key. It's delivered, nothing is recorded, and it's counted and logged.

## What your brief required that differs from Phase 2.6A

| Phase 2.6A's remediation contract | Your brief | Implemented |
|---|---|---|
| Without identity: fall back to today's content formula | No weak fallback. Disable deduplication for that alert | **Your brief** |

## Decisions I made that you may want to review

| # | Decision | Why | Alternative |
|---|---|---|---|
| 1 | **A valid `id` with no usable `manager.name` counts as no identity,** so no suppression | Your brief says to include the namespace when reliable, and not to invent one. 2.6A's D1 says never default. A key without a namespace would collide across managers | Suppress by `id` alone, since one listener reads one file today. 0 sample alerts are affected either way |
| 2 | **An `id` must match `^[0-9]+\.[0-9]+$`, up to 64 characters** | Your brief requires rejecting malformed ids. **If the deployed Wazuh uses another form, every alert loses suppression.** Nothing is lost, and the log says so | Accept any non-empty string |
| 3 | **Three narrow reader and offset changes** beyond identity: see `OFFSET_AND_QUEUE_SAFETY.md` | Each closed a demonstrated way to skip alerts permanently | Leave them, and report them |
| 4 | `request_key` exists but nothing calls it | Your brief asks for the format with the operation separate, for later use | Omit it until the coordinator |

## Changes to existing tests

**One fixture, no assertion.** `tests/test_listener_alert_acceptance.py` built alerts with no `id`, and four of its tests assumed a repeat of those is suppressed. That's the rule your brief removes. The fixture's alerts now carry an `id` and manager, so those tests exercise the same behaviour through the real identity.

## Tests

[M] Final recorded runs, all in an empty network namespace, with no service started:

| Run | Result |
|---|---|
| New listener identity tests | 157 passed |
| Existing listener, audit, failure, safety and routing tests | 185 passed |
| Journal and ledger tests | 712 passed |
| Full backend suite | **2,543 passed, 29 skipped, 0 failed** |
| Mutation check | 54 of 54 caught, control passes |

The previous full-suite baseline was 2,386 passed and 29 skipped. The difference of 157 is exactly the new tests.

## Performance, in one finding

[M] The listener sustains 377 alerts a second with room to spare: it reads about 1,800 a second. **`process_alert` decides whether the peak can be met.** At 5 ms of service per alert, the queue fills and alerts wait, though none were lost in the run. See `PERFORMANCE_RESULTS.md`.

## Not done, as instructed

Coordinator, journal or ledger integration, markers, constraints, W07U migration, live services and production processing. The earlier 2.6B questions are still open.

## Next authorization needed

D6 marker testing or coordinator implementation, each with your explicit go. Separately, `PRODUCTION_VALIDATION_GAPS.md` lists the data that would validate this identity.
