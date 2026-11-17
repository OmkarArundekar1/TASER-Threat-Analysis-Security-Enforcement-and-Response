# D13: first writer selection

Phase 2.6A, evidence and proposal only. **No writer was migrated.** Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Labels:** **[V]** verified current behaviour, **[M]** measured evidence, **[P]** proposed behaviour, **[A]** assumption, **[D]** open decision, **[R]** requires production validation.

## Recommendation

**[P] Keep `update_duplicate_event` on the unattributed path (W07U) as the first writer,** with three conditions Phase 2.5 didn't have:

1. **Measure its throughput first.** It's the busiest path, at 83.8% of sample alerts, with bursts of 377 a second.
2. **Accept that it can't close all re-delivery on its own.** An alert that originally created an event comes back as a duplicate, under a different key.
3. **Decide on retried payloads.** A retry can write an older `investigation_payload` over a newer one.

## Call chain

[V] Verified from the code:

1. `listener/wazuh_listener.py:AlertFileHandler.read_new_lines` L228-306 parses the line, and `listener/wazuh_listener.py:accept_alert` L125-148 queues it.
2. `listener/wazuh_listener.py:queue_worker` L150-167, one thread, calls `realtime_socgraph.py:process_alert` L289-833.
3. `resolve_mitre` returns `UNKNOWN` or `AMBIGUOUS`, so L341 calls `realtime_socgraph.py:_process_unattributed_alert` L170-286.
4. `dedup_engine.py:DeduplicationEngine.is_duplicate` L358-398, with `technique_id=None`, returns true.
5. `dedup_engine.py:DeduplicationEngine.get_metadata` L427-428 gives the `event_id`.
6. L206 calls `neo4j_client.py:update_duplicate_event` L712-748.

**All callers of `update_duplicate_event`:**

| Line | Path | When |
|---|---|---|
| `realtime_socgraph.py` L206 | Unattributed duplicate | Every one with metadata. **This is W07U** |
| `realtime_socgraph.py` L461 | Attributed duplicate | Only when the buffer is full |
| `realtime_socgraph.py` L470 | Attributed duplicate | Only with `ENABLE_DUPLICATE_BUFFER` false. It's true today (`config.py` L35), so **unreachable by default** |

## Does the unattributed path stop before the Campaign?

**[V] Yes, by the call path as well as the function body.** It holds only if the event has no `MATCHES` relationship, so each link was checked:

| Link | Evidence |
|---|---|
| The Cypher stops if `MATCHES` finds nothing | `MATCH (e)-[:MATCHES]->(t:Technique)` at `neo4j_client.py` L727 sits before the Campaign `MATCH` at L736. Zero rows ends the query, and the earlier `SET` stands |
| The `event_id` comes from an event with this fingerprint | Cache, or `neo4j_client.py:find_recent_duplicate` L1380-1405, which matches on fingerprint only |
| The fingerprint can only belong to an unattributed event | `dedup_engine.py:DeduplicationEngine.generate_fingerprint` L304-320 hashes the technique, and this path passes `None`. [M] No sample alert has a native technique spelled `None` |
| Unattributed events have no `MATCHES` | `neo4j_client.py:create_unattributed_attack_event` L588-709 creates none |
| Nothing adds one later | [V] The **only** `MERGE`/`CREATE` of `MATCHES` in backend production code is `neo4j_client.py` L541, inside `neo4j_client.py:create_attack_event` L412-585. Every other match in the search is a read |

**[R] Reading isn't running.** The behaviour of the stopped query needs one real-server test before this is relied on (`D6_APPLIED_WRITE_MARKER_DESIGN.md`, test 9).

**[V] A consequence to preserve or question:** the first unattributed event updates `Campaign.last_seen` and `occurrences` (L650-652). Its duplicates don't. So a campaign seeing only repeats of an unattributed alert doesn't stay active on them. Migration must keep this, and whether it's desirable is a separate matter.

## Verified facts about W07U

| Criterion | Finding | Label |
|---|---|---|
| Function | `update_duplicate_event(event_id, tps, investigation_payload, event_time)` | [V] |
| Parameters | `event_id` from metadata, `tps=0`, `compact_payload`, `event_time.value` | [V] |
| Updates one AttackEvent? | **Yes:** `last_seen`, `occurrences`, `tps`, `investigation_payload` | [V] |
| Effect on Technique and Campaign | **None,** as above | [V] |
| Event identity | Random `uuid4` from the creating write, via cache or graph | [V] |
| Duplicate counting | `occurrences + 1`, once per alert | [V] |
| Called | Once per duplicate alert, never repeated by code | [V] |
| Additive or absolute | `occurrences` is additive. `last_seen` is monotonic. `tps + 0` has no effect. `investigation_payload` is absolute, **last write wins** | [V] |
| Payload determinism | **7,552 of 7,552** gave identical serialization and fingerprint across two parses, with `event_id` held fixed. All fingerprints distinct | [M] |
| Source identity survives? | **Yes.** `alert` is an argument of `_process_unattributed_alert`, so the key can be built at L206 with no new plumbing | [V] |
| Dependencies | An earlier W06 event. Nothing later reads W07U's result, and the helper's return can be ignored | [V] |
| Marker protection | Feasible, inside the first `MATCH` | [P] |
| Return value | None today, and none needed | [V] |

**Where determinism can fail:**

| Input | Effect | Evidence |
|---|---|---|
| `event_id` | Re-delivery can resolve to a **newer event** for the same fingerprint, since `find_recent_duplicate` takes the latest. So a different payload: **`IdempotencyConflictError`** | [M] 18 of 27 modelled unattributed fingerprints have more than one event. Safe, but needs a person |
| `event_time` | The default `EVENT_TIME_POLICY` is `processing_time_fallback` (`config.py` L22). A missing or bad timestamp uses the current time, so a re-delivered alert differs: conflict | [V]. [M] 0 such alerts in the sample |
| Query text | Changes with code, which is deliberate | [V] |

### Three limits to accept

**1. Re-delivery across operations.** [V] An alert that created an event via W06 is re-read after a crash. The cache is empty, so `find_recent_duplicate` finds **its own event** and routes it as a duplicate. Its W07U key is new, so the ledger allows it, and `occurrences` gains 1. **Neither ledger nor marker can catch this,** because the key differs by operation. It's today's behaviour, not a new risk, but **the W07U pilot won't remove it.** Migrating W06 lets the coordinator check for it. [M] The sample models 152 such creators.

**2. Retried payload overwrite.** [V] Between an uncertain attempt and its retry, later duplicates can update the event. The retry would then write an older `investigation_payload`. `last_seen` is guarded and the payload isn't. **[D]** Accept it for the pilot, or guard the payload with the same time check, which changes behaviour slightly. I recommend accepting it, because the field is a display copy of the latest alert.

**3. Throughput.** [M] W07U's busiest second has 377 requests, and 27 seconds hold 100 or more. Each coordinated request needs several durable SQLite commits plus a Neo4j transaction. **That cost is unmeasured.** [V] A full queue delays rather than loses, by re-reading (`wazuh_listener.py` L274-280), but a long backlog could outrun log rotation. The pilot must measure this first.

## Comparison

[M] Rates are from the routing model in `evidence/d2_replay_output.txt`, on the sample.

| Criterion | **W07U** | W06 `create_unattributed_attack_event` | W07 attributed | W08 `batch_update_duplicates` |
|---|---|---|---|---|
| Modelled alerts | 7,552 | 152 | 1,201, nearly all via W08 | Those 1,201, batched |
| Risk reduction | **Moderate:** `occurrences` on one node | Higher: stops a duplicate event | Moderate, and it's rarely reached | **Highest:** the reproduced double count |
| Blast radius if the migration is wrong | **One property set on one node** | Event, Campaign, Attacker, Host | Event, Technique, Campaign, risk | All of those, for many alerts at once |
| Identity confidence | **High** after D1 | High for the key | High | **Unresolved** (D8) |
| Payload determinism | **Measured,** given `event_id` | **No:** a new `uuid4` per call (D7) | Medium: `tps` is computed from state | **No:** merges over time |
| Graph idempotency | Feasible | Feasible, with D7 | Feasible | Only with a batch identity |
| Transaction boundary | One statement | One statement, and a return value is needed | One statement | One statement, with restore logic around it |
| Testability | **High:** assert one node | Medium | Hard to reach | Hard: timing and restore |
| Rollback and migration | **Simple:** one call site | Medium: id format changes, and consumers see ids | Simple | **Hard:** interacts with your ambiguous-flush decision |
| Throughput exposure | **Highest** | Low | Very low | Batched |

**Why not W08, which has the highest value?** Nothing has a defensible key until D8 is decided. **Why not W06?** It needs D7 first, and it changes id formats that dashboards see. It has a lower rate, so it's **the natural second writer,** and it removes limit 1.

**Why not choose something rarer, to avoid throughput risk?** W07 attributed is barely reached, so a pilot on it would prove little. The risk can be measured safely in a benchmark before anything runs live.

## Pilot acceptance

| # | Evidence | Pass |
|---|---|---|
| 1 | Real-server test of the stopped query | Campaign and Technique unchanged |
| 2 | Benchmark on disposable stores and server | Sustains a replayed peak without unbounded backlog. Threshold to be set from what's measured |
| 3 | Fault tests for each outcome, using the existing proxy | No double count, no lost record |
| 4 | Replay the sample before and after | Same `occurrences` per event |
| 5 | Crash tests in all windows | Per `D4_RETRY_AFTER_RESOLUTION.md` |
| 6 | Conflict rate on re-delivery | Reported, and bounded |
| 7 | Feature flag | Off restores today's path, tested |

## If rejected

Another first writer needs D7 or D8 settled first, which delays Phase 2.6.
