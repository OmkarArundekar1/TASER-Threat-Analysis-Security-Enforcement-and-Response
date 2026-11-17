# D2: alert deduplication fix

Backend paths are relative to `isfcr/CYUKTI/backend/`. **[V]** verified from code, **[M]** measured offline, **[T]** reproduced in tests, **[O]** modeled behaviour, **[N]** not yet validated in production.

## The defect, reproduced before the fix

[M] The real, unmodified functions, kept in `evidence/before_fix/`, on the sample:

| Measure | Before | After |
|---|---|---|
| Accepted, first pass | 8,945 | **9,015** |
| Refused as repeats, first pass | 70 | **0** |
| The 70 reaching `process_alert` | 0 | **70** |
| The 7 naming a different attacker or victim, reaching `process_alert` | 0 | **7** |

## The fix

**The primary contract is source identity, not a better hash.** [V] `listener/wazuh_listener.py:accept_alert` L128-162 compares `alert_id`, which is now `wazuh:<manager>/<id>` or nothing. **No content takes part in the decision.** `generate_alert_id` is gone, so nothing can call it by accident.

| Case | Result | Test |
|---|---|---|
| Same manager, same id | Duplicate | [T] |
| Same id, different managers | Both accepted | [T] |
| Different ids, identical content | Both accepted | [T] |
| Same identity, differing `firedtimes`, attacker or timestamp | Duplicate: **identity decides** | [T] |
| Different `firedtimes`, different ids | Both accepted | [T] |
| Repeated timestamp, different ids | All accepted | [T] |
| Replayed file, one process | All duplicates | [T] |
| Missing, empty or malformed id | All delivered | [T] |

**Note the fourth row.** Two alerts with the same manager and id are treated as one, even with different content. That's the contract, and it's why id reuse is a production question.

## Field names

[M] The sample's real schema: `data.srcip` on 8,004 alerts, `data.src_ip` on 189, neither on 822. `data.dest_ip` on 189, `data.dstip` on none. **No alert has both spellings, and none has them at the top level.**

[V] `listener/utils.py:network_endpoints` L118-129 handles all four, and the top-level forms the old code read:

| Value | Order |
|---|---|
| Source | `data.src_ip`, `data.srcip`, top-level `src_ip`, top-level `srcip` |
| Destination | `data.dest_ip`, `data.dstip`, top-level `dest_ip`, top-level `dstip` |

- **Why this order:** it matches `realtime_socgraph.py:extract_iocs` L80-105, which decides the attacker and victim. [T] A test compares the two directly.
- **Conflicts:** both spellings disagreeing is detected and logged, with no values. **Both originals stay on the alert.** [T]
- **Empty first form:** falls through to the next, as `extract_iocs` does. [T]

**These fields no longer decide delivery.** Their only remaining use in the listener is the conflict warning and the diagnostic fingerprint. The attacker and victim are still decided downstream by `extract_iocs`, which was already correct and wasn't touched.

## What's preserved

| Requirement | Result |
|---|---|
| Different attackers or victims not merged | [T] Delivered separately with their own values, including the sample's 7 |
| Legitimate duplicate updates | [T] `process_alert`'s own fingerprint dedup is untouched. Duplicates there are a separate layer |
| Stable across replay and restart | [T] Derived only from the alert's own fields |

## Two layers of "duplicate"

These are different, and the fix touches only the first.

| Layer | Question | Where | Changed? |
|---|---|---|---|
| Source-level | Have I already accepted this exact Wazuh alert? | `listener/wazuh_listener.py:accept_alert` L128-162 | **Yes** |
| Event-level | Does this new alert repeat a recent event? | `dedup_engine.py:DeduplicationEngine.is_duplicate` L358-398 | No |

## Attributed and unattributed paths

[T] Using the real `process_alert` with its dependencies stubbed, as the existing integration tests do:

| Path | Test | Result |
|---|---|---|
| Attributed, new | `test_the_identity_fields_do_not_change_routing` | Same writer, same arguments, with or without the identity fields |
| Unattributed, new | Same | Same |
| Unattributed, duplicate | `test_a_duplicate_on_the_unattributed_path_still_updates_its_event` | `DUPLICATE_UNATTRIBUTED`, `update_duplicate_event` called, `tps` 0, no identity in the payload |
| Attributed, duplicate | `test_a_duplicate_on_the_attributed_path_still_goes_to_the_buffer` | `DUPLICATE`, goes to the buffer |

**[O] Modeled routing, which isn't a production fact.** The same model as Phase 2.6A, with no Neo4j, over what reached `process_alert`:

| Category | Before | After | Model of all 9,015 |
|---|---|---|---|
| Attributed duplicate | 1,198 | 1,201 | 1,201 |
| Attributed new event | 110 | 110 | 110 |
| Unattributed duplicate | 7,486 | 7,552 | 7,552 |
| Unattributed new event | 151 | 152 | 152 |

So the fix restores 69 modeled duplicate updates and 1 modeled new event. **More alerts now reach the pipeline, so counts in a live system will shift upward by about this share,** 0.78% on this sample.

**Whether each of the 69 joins the right event is decided by `process_alert` and the graph, not by this model.** No Neo4j was involved.

## Research replays

[V] `replay/harness.py` calls `process_alert` directly, so it never had this defect. Offline replays and the live listener should now agree on which alerts arrive. Frozen evidence is unaffected.
