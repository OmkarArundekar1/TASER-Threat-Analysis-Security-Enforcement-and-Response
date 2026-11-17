# Phase 2.6A findings

Decision resolution and implementation readiness. **Read-only investigation and design: nothing is implemented, and every decision remains yours.** Inspected at HEAD `3afe62c` plus the uncommitted working tree. Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Labels:** **[V]** verified current behaviour, **[M]** measured evidence, **[P]** proposed behaviour, **[A]** assumption, **[D]** open decision, **[R]** requires production validation.

## Reading order

| Document | Purpose |
|---|---|
| This one | The main results, and corrections to Phase 2.5 |
| `DECISION_REGISTER_D1_D15.md` | **Start here to decide.** Every decision and approval |
| `D1_SOURCE_IDENTITY_DECISION.md` | The source identity contract |
| `D2_ALERT_DEDUPLICATION_INVESTIGATION.md` | The confirmed alert-loss defect |
| `D3_JOURNAL_LEDGER_API_CONTRACT.md` | AC-1, AC-2, AC-3 |
| `D4_RETRY_AFTER_RESOLUTION.md` | The retry model and crash windows |
| `D6_APPLIED_WRITE_MARKER_DESIGN.md` | The marker |
| `D13_FIRST_WRITER_SELECTION.md` | The first writer |
| `PHASE_2_6_IMPLEMENTATION_READINESS.md` | Dependencies, blockers, order |
| `EVIDENCE_INDEX.md` | Where each claim comes from |

## Main results

1. **[M] The listener defect is confirmed offline.** The real `accept_alert` and `generate_alert_id` refuse 70 of 9,015 sample alerts, all with distinct Wazuh IDs and different content. The loss happens before any write, so **no coordinator can protect it.** One of the 70 would have been a new event in the routing model.
2. **[V] The Wazuh `id` is `<event seconds>.<file offset>`** (Wazuh v4.9.0 source). It has no manager part, and it's not a designed unique ID. The deployed version wasn't checked.
3. **[M] A content hash isn't a safe substitute, even though it looks fine here.** It gives 9,015 distinct values in this sample only because of the timestamp and a Wazuh counter. **I withdraw Phase 2.5's hash fallback.**
4. **[P] The three journal changes are sufficient to link the stores,** with four tighter rules, and the ledger needs no change for them.
5. **[P] The new-intent retry holds** in every crash window checked, with two state pairs and two small rules added. **Its unhandled risk is a person's wrong "not committed" on an additive write,** which only a marker can catch.
6. **[P] The marker is feasible on paper, and unproven.** Written as a separate statement it would give false evidence, so it must share the mutation's `MATCH`. **Its absence isn't proof of non-commit.**
7. **[V] W07U stops before the Campaign,** verified along the whole call path and by the fact that `MATCHES` is created in one place only. **[R]** A real server has not confirmed it.
8. **[M] W07U stays the recommended first writer, with three new conditions:** it's the busiest path, at 7,552 of 9,015 alerts with a peak of 377 a second; re-delivery across operations bypasses it; and a retry may overwrite a newer payload.
9. **Phase 2.6 isn't ready to implement.** See `PHASE_2_6_IMPLEMENTATION_READINESS.md`.

## Corrections to Phase 2.5

Phase 2.5's files are unchanged. These corrections are recorded here.

| # | Phase 2.5 said | Correction | Evidence |
|---|---|---|---|
| 1 | 70 alerts share an `alert_id` while their "`id` and content differ" | **The method was weak:** its comparison included `id`, which always differs. Recomputed without it, the claim **holds:** 0 of the 70 differ only in `id` | `evidence/d2_replay_output.txt` |
| 2 | D1 recommended a `sha:` content-hash fallback | **Withdrawn.** It merges real occurrences in principle | `D1_SOURCE_IDENTITY_DECISION.md` |
| 3 | Example keys were `wazuh:<id>` | **The `id` has no manager part,** so a namespace is added | Same |
| 4 | The `id`'s form "suggests" time plus offset | Now read from Wazuh's source, for v4.9.0 | `EVIDENCE_INDEX.md` |
| 5 | After a crash, "the offset has already moved past" queued alerts | **Only partly true.** The offset is saved after the read loop, so a crash can also re-deliver | `listener/wazuh_listener.py:AlertFileHandler.read_new_lines` L228-306 |
| 6 | State tables | Two legal pairs were missing | `D4_RETRY_AFTER_RESOLUTION.md` |
| 7 | "A marker commits atomically with its write" | True only if it depends on the same `MATCH`. A separate statement could commit alone | `D6_APPLIED_WRITE_MARKER_DESIGN.md` |
| 8 | W07's duplicate risk: "event, technique and campaign" | True for the attributed path. On W07U it's the event only | `D13_FIRST_WRITER_SELECTION.md` |

## What this phase did not establish

| Open question | Status |
|---|---|
| Wazuh `id` uniqueness across restart, rotation, restore, reinstall and multiple managers | **Unresolved:** one sample, none of those events |
| Deployed Wazuh version | **Unresolved** |
| Sample's provenance | **Unresolved:** gitignored, with no record |
| Production collision rate | **Unresolved** |
| Marker behaviour, cost and storage | **Unresolved:** no server was started |
| Whether "no transaction open" can be established on Community | **Unresolved** |
| Coordinator latency | **Unresolved** |
| Production Neo4j edition and constraints | **Unresolved** |

## Safety

No production code, test, journal, ledger, helper or frozen evidence changed. No service, listener, database, replay experiment or test ran. Four read-only analysis scripts and one document checker ran over existing files, and they're in `evidence/`.
