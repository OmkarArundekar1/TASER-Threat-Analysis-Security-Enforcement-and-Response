# Production Neo4j writer inventory

Phase 2.5, investigation only. Inspected at HEAD `3afe62c` plus the uncommitted working tree. Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Labels:** **[V]** verified current behaviour, **[P]** proposed behaviour, **[D]** open decision, **[A]** assumption requiring verification, **[L]** known limitation. Decision numbers refer to `PHASE_2_5_SCOPE_AND_FINDINGS.md`.

## Method

[V] Three sweeps, then a caller trace. The scripts and raw output are listed in `EVIDENCE_INDEX.md`.

1. An AST scan of every backend `.py` file outside `tests/` for string constants holding Cypher write clauses.
2. An AST scan for driver entry points: `session.run`, `tx.run`, `execute_write`, `execute_read`, `begin_transaction`, and any `.run(` on any receiver, including non-literal queries.
3. A list of every module importing the driver or `neo4j_client`.
4. A caller trace of each writer by AST, confirming entry points by hand.

Names decided nothing: `create_operation_db` is a live writer, while `OperationRepository.create_operation` is dead code.

## Summary

[V] **20 production writers**, all reached from one long-running process, `listener/wazuh_listener.py`. Nothing else in production writes to Neo4j.

| Tier | Meaning | Count | Writers |
|---|---|---|---|
| 1 | High duplicate or corruption risk | 7 | W01, W05, W06, W07, W08, W10, W16 |
| 2 | Moderate: replay reverses or inflates state | 4 | W02, W03, W04, W15 |
| 3 | Lower: absolute or convergent | 9 | W09, W11, W12, W13, W14, W17, W18, W19, W20 |
| Excluded | Not a production write path | 14 | X01 to X14 |

| Identity confidence | Count | Writers |
|---|---|---|
| High, given decision D1 | 4 | W05, W06, W07, W10 |
| Medium: the key is sound, but the payload or a dependency needs work | 13 | W01, W02, W03, W04, W09, W11, W12, W13, W14, W15, W16, W17, W18 |
| **Unresolved** | 3 | W08, W19, W20 |

Every confidence rating assumes a source identity from decision D1. **Today, no writer has a usable stable identity at its call site**, because `process_alert` doesn't receive or use one.

### Facts that apply to many writers

- **[V] One thread per path.** Alert writers run in `queue_worker` (`listener/wazuh_listener.py:queue_worker` L150-167). W03 (one path), W19 and W20 run in `maintenance_worker` (`listener/wazuh_listener.py:maintenance_worker` L169-190). W08 runs in the flush worker.
- **[V] Autocommit.** Every writer calls `session.run` directly. None uses a transaction function, and none uses the Phase 2.2 helper. Only W05, W06, W19 and W20 read their result. For the rest, the driver can swallow a failure at session close (Phase 2.1 finding, reproduced on a real server).
- **[V] Writes per alert aren't one transaction.** A resolved new alert runs up to 13 separate writer calls with no shared boundary (`realtime_socgraph.py:process_alert` L289-833).
- **[V] Failure handling.** `process_alert` catches every exception, prints it and returns `False`. `queue_worker` logs that. Neither retries (`realtime_socgraph.py:process_alert` L289-833, `listener/wazuh_listener.py:queue_worker` L150-167).
- **[V] No retry exists,** except W08's restore-to-buffer and the APOC fallback in W11.
- **[V] The offset moves first.** `offset.dat` advances when an alert enters the queue, before processing (`listener/wazuh_listener.py:AlertFileHandler.read_new_lines` L228-306). A crash can lose queued alerts, so re-delivery isn't normally how a crashed alert returns.
- **[L] Side effects outside Neo4j** in the same flow aren't covered by anything here: MISP publication, the Shuffle webhook, and the in-memory campaign and dedup caches.

## Writers

Key notation: `<src>` is the source event identity from decision D1. The full key format is in `REQUEST_IDENTITY_MAPPING.md`. All line ranges are in `neo4j_client.py` unless a path is given.

### W01: `create_campaign_db`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:create_campaign_db` L982-1038 |
| Callers | `campaign_manager.py:CampaignManager.create_campaign_context` L424-471, from `campaign_manager.py:CampaignManager.resolve_campaign` L523-735 and `realtime_socgraph.py:_process_unattributed_alert` L170-286 |
| Logical operation | Open a new campaign for an attacker and victim pair |
| Neo4j effects | `CREATE (:Campaign)` with defaults |
| Statements, boundary | One, autocommit, result discarded |
| Input identity | Attacker and victim IPs, event time. `campaign_id` is a new random value made inside the function |
| Retry behaviour | None |
| Idempotency | [V] None. Each run creates a new node with a new id |
| Duplicate risk | A second Campaign for the same pair, splitting later events. This is the over-splitting problem the project is correcting |
| Failure handling | Discarded result, so a failure can go unnoticed, and the caller then uses an id that doesn't exist |
| Candidate key | `<src>` + `campaign.create` |
| Identity confidence | Medium: the id is random, so two runs have different payloads and would conflict (D7) |
| Migration risk | **High:** needs deterministic ids and changes who allocates the id |

### W02: `reopen_campaign_db`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:reopen_campaign_db` L963-979 |
| Callers | `campaign_manager.py:CampaignManager.resolve_campaign` L523-735 (two sites) |
| Logical operation | Reactivate a closed campaign for a new alert |
| Neo4j effects | Campaign: status, monotonic `last_seen`, prediction cleared, `reopened_count + 1` |
| Statements, boundary | One, autocommit, discarded |
| Input identity | `campaign_id`, event time |
| Retry behaviour | None |
| Idempotency | [V] No: `reopened_count` is additive |
| Duplicate risk | Inflated `reopened_count`, and it clears a newer prediction |
| Failure handling | Discarded result; the in-memory context still says ACTIVE |
| Candidate key | `<src>` + `campaign.reopen` + `campaign_id` |
| Identity confidence | Medium: `campaign_id` comes from a decision over mutable state, though it's stable once recorded |
| Migration risk | Medium |

### W03: `close_campaign_db`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:close_campaign_db` L1057-1072 |
| Callers | `campaign_manager.py:CampaignManager.close_expired_context` L192-211, `campaign_manager.py:CampaignManager.expire_active_campaigns` L473-521 (maintenance), `campaign_manager.py:CampaignManager.resolve_campaign` L523-735 |
| Logical operation | Mark a campaign inactive |
| Neo4j effects | Campaign: status INACTIVE, `closed_at` |
| Statements, boundary | One, autocommit, discarded |
| Input identity | `campaign_id`, `closed_at` |
| Retry behaviour | None |
| Idempotency | [V] Absolute values, so a repeat right away changes nothing. A delayed repeat would close a campaign reopened in between |
| Duplicate risk | A late replay reverses a reopen |
| Failure handling | Discarded; the cache says INACTIVE regardless |
| Candidate key | Alert path: `<src>` + `campaign.close` + `campaign_id`. Maintenance path: `campaign.close` + `campaign_id` + `closed_at` |
| Identity confidence | Medium. The maintenance form relies on `closed_at = last_seen + timeout`, which is derived but stable only while `last_seen` is |
| Migration risk | Medium: two paths, two threads |

### W04: `update_campaign_activity`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:update_campaign_activity` L1040-1055 |
| Callers | `campaign_manager.py:CampaignManager.activate_campaign_with_transition` L748-788 |
| Logical operation | Attach a resolved alert's activity to its campaign |
| Neo4j effects | Campaign: monotonic `last_seen`, status ACTIVE, `last_technique` |
| Statements, boundary | One, autocommit, discarded |
| Input identity | `campaign_id`, technique, event time |
| Retry behaviour | None |
| Idempotency | [V] Partly. `last_seen` is monotonic, but status and `last_technique` aren't protected |
| Duplicate risk | A late replay reactivates a closed campaign and rewinds `last_technique` |
| Failure handling | Discarded |
| Candidate key | `<src>` + `campaign.activity` + `campaign_id` |
| Identity confidence | Medium |
| Migration risk | Medium |

### W05: `create_attack_event`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:create_attack_event` L412-585 |
| Callers | `realtime_socgraph.py:process_alert` L289-833 |
| Logical operation | Record a new attributed attack event |
| Neo4j effects | MERGE Attacker, Host, Technique, Stage. CREATE AttackEvent. Additive: Campaign `occurrences`, `total_tps`, `risk_score`; Technique `occurrences`, `total_tps`. MERGE five relationships. RETURN `event_id` |
| Statements, boundary | One, autocommit, consumed with `.single()` |
| Input identity | Fingerprint, `campaign_id`, event time. `event_id` is random, made inside the function |
| Retry behaviour | None |
| Idempotency | [V] None |
| Duplicate risk | **A second event, and every additive counter doubled.** This feeds risk and CTI |
| Failure handling | Errors surface. If the campaign is missing there's no row, and `record["event_id"]` raises `TypeError` |
| Candidate key | `<src>` + `attack_event.create` |
| Identity confidence | High for the key. Its payload needs deterministic `event_id` (D7) |
| Migration risk | **High:** the most effects, returns a value later steps need, and depends on W01 |

### W06: `create_unattributed_attack_event`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:create_unattributed_attack_event` L588-709 |
| Callers | `realtime_socgraph.py:_process_unattributed_alert` L170-286 |
| Logical operation | Preserve an alert with no defensible ATT&CK mapping |
| Neo4j effects | MERGE Attacker, Host. CREATE AttackEvent. Campaign `occurrences + 1`. MERGE three relationships. RETURN `event_id` |
| Statements, boundary | One, autocommit, consumed |
| Input identity | As W05, without a technique |
| Retry behaviour | None |
| Idempotency | [V] None |
| Duplicate risk | A second event, and `occurrences` doubled |
| Failure handling | As W05 |
| Candidate key | `<src>` + `attack_event.create_unattributed` |
| Identity confidence | High for the key; needs deterministic `event_id` (D7) |
| Migration risk | **High,** but simpler than W05 |

### W07: `update_duplicate_event`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:update_duplicate_event` L712-748 |
| Callers | `realtime_socgraph.py:_process_unattributed_alert` L170-286 (always), `realtime_socgraph.py:process_alert` L289-833 (when the buffer is full or disabled). [V] `ENABLE_DUPLICATE_BUFFER = True` (`config.py` L35) |
| Logical operation | Apply one duplicate alert to its existing event |
| Neo4j effects | AttackEvent: monotonic `last_seen`, `occurrences + 1`, `tps + $tps`, payload. Then Technique and Campaign additive |
| Statements, boundary | One, autocommit, discarded |
| Input identity | `event_id` from the dedup cache, `tps`, payload, event time. No generated ids |
| Retry behaviour | None |
| Idempotency | [V] None: additive |
| Duplicate risk | Double counting on the event, technique and campaign |
| Failure handling | Discarded: a failure can vanish silently |
| Candidate key | `<src>` + `duplicate_event.update` |
| Identity confidence | **High:** each duplicate alert is its own logical request, and nothing in the payload is random |
| Migration risk | Medium. **Recommended first candidate,** see below |

[V] **Finding.** The statement continues with `MATCH (e)-[:MATCHES]->(t:Technique)`. Unattributed events have no `MATCHES` relationship (`neo4j_client.py:create_unattributed_attack_event` L588-709), so for them the statement stops there, and the Campaign counters aren't updated. That comes from reading the query, not from running it. Whether it's intended is outside this phase.

### W08: `batch_update_duplicates`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:batch_update_duplicates` L749-785 |
| Callers | `dedup_engine.py:_write_pending` L124-141, from `dedup_engine.py:DuplicateFlushWorker._flush_once` L263-285 and `dedup_engine.py:DuplicateFlushWorker.stop` L186-258 |
| Logical operation | Apply up to 200 aggregated duplicate records (`config.py` L34) |
| Neo4j effects | Per record: as W07, with `r.occurrences` and `r.total_tps` |
| Statements, boundary | One UNWIND per batch, autocommit, discarded |
| Input identity | Aggregated records keyed by fingerprint. **Contributing alerts aren't retained** (`dedup_engine.py:DuplicateBuffer.add_duplicate` L40-77) |
| Retry behaviour | [V] On an exception the batch returns to the buffer, and it can merge with new duplicates (`dedup_engine.py:DuplicateBuffer.restore` L91-112) |
| Idempotency | [V] None |
| Duplicate risk | **The reproduced double count:** a committed batch reported as failed is restored and applied again (Phase 2.1). A batch can hold many alerts' worth of counts |
| Failure handling | Restore and log |
| Candidate key | **None defensible today.** A batch's content changes as it merges, and fingerprint alone isn't a request |
| Identity confidence | **UNRESOLVED** (D8) |
| Migration risk | **High** |

### W09: `store_dynamic_risk`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:store_dynamic_risk` L1711-1750 |
| Callers | `realtime_socgraph.py:process_alert` L289-833 |
| Logical operation | Store the campaign's current risk |
| Neo4j effects | Campaign: risk fields, `risk_trend`, `risk_updated_at` |
| Statements, boundary | One read then one write, separate autocommit transactions |
| Input identity | `campaign_id`, computed risk |
| Retry behaviour | None |
| Idempotency | [V] Mostly absolute, except `risk_trend`, computed from a previous read |
| Duplicate risk | The trend says STABLE on a repeat. Low impact |
| Failure handling | Discarded |
| Candidate key | `<src>` + `campaign.dynamic_risk` + `campaign_id` |
| Identity confidence | Medium: the payload holds a value read from the graph, so it must be computed once and kept |
| Migration risk | Low to medium |

### W10: `update_attack_chain`

| Field | Value |
|---|---|
| Module, function | `chain_updater.py:update_attack_chain` L31-155 |
| Callers | `realtime_socgraph.py:process_alert` L289-833 |
| Logical operation | Learn one observed technique transition |
| Neo4j effects | NEXT_TECHNIQUE: MERGE, `count + 1`, `last_seen`. Then `confidence` recomputed across the source's edges |
| Statements, boundary | **Two writes in separate autocommit transactions,** then one read, in one session |
| Input identity | Previous and current technique, from an immutable `TransitionContext` |
| Retry behaviour | None |
| Idempotency | [V] None: count is additive |
| Duplicate risk | **Corrupts the global learned model used for prediction,** across all campaigns. A crash between the two writes leaves `confidence` stale |
| Failure handling | Exceptions are caught and printed, returning `None` |
| Candidate key | `<src>` + `chain.transition` + previous + current |
| Identity confidence | High |
| Migration risk | Medium to high: should become one transaction, which is a real behaviour change |

### W11: `update_campaign_similarity`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:update_campaign_similarity` L794-848 |
| Callers | `realtime_socgraph.py:process_alert` L289-833 |
| Logical operation | Recompute SIMILAR_TO edges |
| Neo4j effects | Deletes and re-creates edges above a score threshold |
| Statements, boundary | One, with a second attempt without APOC on failure |
| Input identity | `campaign_id` |
| Retry behaviour | [V] The only built-in retry: an APOC fallback in the same session |
| Idempotency | [V] Convergent: recomputed from the graph |
| Duplicate risk | Low. A repeat gives the current answer, not the earlier one |
| Failure handling | Caught and printed at the caller |
| Candidate key | `<src>` + `campaign.similarity` + `campaign_id` |
| Identity confidence | Medium: it may not need a ledger entry (D11) |
| Migration risk | Medium: `query.replace` fallback, and cost scales with campaigns |

### W12: `update_actor_attribution`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:update_actor_attribution` L850-886 |
| Callers | `realtime_socgraph.py:process_alert` L289-833 |
| Logical operation | Recompute RESEMBLES edges to threat actors |
| Neo4j effects | Deletes and re-creates edges |
| Statements, boundary | One, autocommit, discarded |
| Input identity | `campaign_id` |
| Retry behaviour | None |
| Idempotency | [V] Convergent |
| Duplicate risk | Low |
| Failure handling | Caught and printed at the caller |
| Candidate key | `<src>` + `campaign.attribution` + `campaign_id` |
| Identity confidence | Medium (D11) |
| Migration risk | Low to medium |

### W13: `store_prediction`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:store_prediction` L1265-1303 |
| Callers | `campaign_manager.py:CampaignManager.update_prediction` L246-273, from `prediction_engine.py:predict_next` L87-109, from `realtime_socgraph.py:process_alert` L289-833 |
| Logical operation | Store the campaign's next-technique prediction |
| Neo4j effects | Replaces LIKELY_NEXT; Campaign prediction fields |
| Statements, boundary | One, autocommit, discarded |
| Input identity | `campaign_id`, predicted technique, confidence |
| Retry behaviour | None |
| Idempotency | [V] Absolute, apart from server timestamps |
| Duplicate risk | A late replay restores an old prediction |
| Failure handling | Discarded |
| Candidate key | `<src>` + `campaign.prediction` + `campaign_id` |
| Identity confidence | Medium |
| Migration risk | Low to medium |

[A] The query's second `MERGE (c)-[r:LIKELY_NEXT]->(t)` runs after an `OPTIONAL MATCH` that can leave `t` null, which Neo4j would normally reject. With a discarded result, that failure could go unnoticed. Check it on a disposable server before migrating.

### W14: `store_cti_confidence`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:store_cti_confidence` L1752-1770 |
| Callers | `realtime_socgraph.py:process_alert` L289-833 |
| Logical operation | Store the campaign's CTI score |
| Neo4j effects | Campaign: `cti_*` fields, `cti_updated_at` |
| Statements, boundary | One, autocommit, discarded |
| Input identity | `campaign_id`, computed score |
| Retry behaviour | None |
| Idempotency | [V] Absolute |
| Duplicate risk | Low: a late replay restores an old score |
| Failure handling | Discarded |
| Candidate key | `<src>` + `campaign.cti` + `campaign_id` |
| Identity confidence | Medium |
| Migration risk | Low |

### W15: `reopen_operation_db`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:reopen_operation_db` L1163-1178 |
| Callers | `campaign_correlation_engine.py:CampaignCorrelationEngine.correlate` L37-171 |
| Logical operation | Reactivate an operation |
| Neo4j effects | Operation: status, monotonic `last_seen`, removes `closed_at` |
| Statements, boundary | One, autocommit, discarded |
| Input identity | `operation_id`, event time |
| Retry behaviour | None |
| Idempotency | [V] Absolute at once; a late replay reverses a close |
| Duplicate risk | A closed operation reactivated |
| Failure handling | Discarded |
| Candidate key | `<src>` + `operation.reopen` + `operation_id` |
| Identity confidence | Medium |
| Migration risk | Low to medium |

### W16: `create_operation_db`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:create_operation_db` L1103-1133 |
| Callers | `realtime_socgraph.py:process_alert` L289-833 |
| Logical operation | Open a new operation for a campaign no operation matched |
| Neo4j effects | `CREATE (:Operation)` |
| Statements, boundary | One, autocommit, discarded |
| Input identity | `campaign_context`. `operation_id` is random, made inside the function |
| Retry behaviour | None |
| Idempotency | [V] None |
| Duplicate risk | A second Operation, splitting correlation |
| Failure handling | Discarded, then W17 attaches to a node that may not exist |
| Candidate key | `<src>` + `operation.create` |
| Identity confidence | Medium: needs a deterministic id (D7) |
| Migration risk | **High:** the id is needed by W17 straight after |

### W17: `attach_campaign_to_operation`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:attach_campaign_to_operation` L1180-1209 |
| Callers | `realtime_socgraph.py:process_alert` L289-833 (two sites) |
| Logical operation | Link a campaign to its operation |
| Neo4j effects | MERGE HAS_CAMPAIGN; `campaign_count + 1` only ON CREATE; `last_seen`; primary fields if empty |
| Statements, boundary | One, autocommit, discarded |
| Input identity | `operation_id`, `campaign_id`, last seen |
| Retry behaviour | None |
| Idempotency | [V] Yes for the link and the count. `last_seen` is overwritten, not monotonic |
| Duplicate risk | Low: a late replay can move `last_seen` backwards |
| Failure handling | Discarded |
| Candidate key | `<src>` + `operation.attach` + `operation_id` + `campaign_id` |
| Identity confidence | Medium |
| Migration risk | Low |

### W18: `update_operation_activity`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:update_operation_activity` L1135-1149 |
| Callers | `realtime_socgraph.py:process_alert` L289-833 |
| Logical operation | Record operation activity |
| Neo4j effects | Operation: `last_seen` (not monotonic), status ACTIVE |
| Statements, boundary | One, autocommit, discarded |
| Input identity | `operation_id`, last seen |
| Retry behaviour | None |
| Idempotency | [V] Absolute at once |
| Duplicate risk | A late replay moves `last_seen` back and reactivates |
| Failure handling | Discarded |
| Candidate key | `<src>` + `operation.activity` + `operation_id` |
| Identity confidence | Medium |
| Migration risk | Low |

### W19: `expire_stale_campaigns_db`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:expire_stale_campaigns_db` L1074-1101 |
| Callers | `campaign_manager.py:CampaignManager.expire_active_campaigns` L473-521, from `listener/wazuh_listener.py:maintenance_worker` L169-190, every 5 s |
| Logical operation | Close every ACTIVE campaign past its timeout |
| Neo4j effects | Campaign: status, `closed_at`, for all matching rows. RETURNs ids |
| Statements, boundary | One, autocommit, iterated |
| Input identity | Timeout and an event-time watermark, which changes constantly |
| Retry behaviour | Runs again in 5 s regardless |
| Idempotency | [V] Convergent: the `status = 'ACTIVE'` condition means a repeat finds nothing |
| Duplicate risk | Low. A lost result leaves the cache unaware until its own sweep |
| Failure handling | Caught and logged by the worker |
| Candidate key | **None.** A sweep has no logical identity, and a watermark key would make a new request every 5 s |
| Identity confidence | **UNRESOLVED,** recommended out of the ledger (D9) |
| Migration risk | Low through the helper alone |

### W20: `expire_stale_operations_db`

| Field | Value |
|---|---|
| Module, function | `neo4j_client.py:expire_stale_operations_db` L1211-1229 |
| Callers | `operation_manager.py:OperationManager.expire_active_operations` L82-91, from `listener/wazuh_listener.py:maintenance_worker` L169-190 |
| Logical operation | Close every ACTIVE operation past its timeout |
| Neo4j effects | Operation: status, `closed_at` |
| Statements, boundary | One, autocommit, iterated |
| Input identity | As W19 |
| Retry behaviour | As W19 |
| Idempotency | [V] Convergent |
| Duplicate risk | Low |
| Failure handling | As W19 |
| Candidate key | **None** |
| Identity confidence | **UNRESOLVED,** recommended out of the ledger (D9) |
| Migration risk | Low through the helper alone |

## Excluded

| ID | Code | Why excluded |
|---|---|---|
| X01 | `operation_repository.py:OperationRepository` L13-40 | [V] No module imports it. It's also inconsistent with what it wraps: `create_operation_db()` is called with no argument. Dead |
| X02 | `neo4j_client.py:close_operation_db` L1151-1161 | [V] Its only caller is X01. Dead |
| X03 | `neo4j_client.py:archive_campaign_db` L1339-1354 | [V] Reached only through `campaign_manager.py:CampaignManager.archive_old_campaigns` L852-873 and `cleanup_cache`, and `cleanup_cache` has no callers. Dead |
| X04 | `neo4j_client.py:store_likely_next` L787-792 | [V] No callers. Dead wrapper over W13 |
| X05 | `risk_recalculation.py:apply_recalculation` L111-135 | [V] No caller in production or tests. Manual maintenance tool; absolute values by design |
| X06 | `campaign_reconstruction.py:apply_reconstruction` L109-151 | [V] Its module isn't imported by anything. Manual repair; MERGE throughout |
| X07 | `attack_chain_builder.py:build_attack_chain` L11-28 | [V] A `__main__` seed script, not imported. Additive, so running it twice inflates counts |
| X08 to X13 | `mitre_import/import_techniques.py`, `import_threat_actors.py`, `import_courses.py`, `import_tools.py`, `import_malware.py`, `relationships.py` | [V] Setup, run through `mitre_import/import_all.py`. MERGE on stable STIX ids. Own driver |
| X14 | Reads that look like writes | [V] `dashboard_api.py` runs its one user-supplied query through `execute_read`, after refusing mutating commands. `soar/api.py`, `recommendation_engine.py`, `graph_feature_engine.py`, `ml/gnn/*`, `replay/harness.py`, `benchmarks/run_benchmarks.py` and `_verify_neo4j.py` hold no write clauses. The harness and benchmarks reach writers only by calling `process_alert` |

[L] X05, X06 and X07 are writes a person can run by hand. They're outside the coordinator's plan, and running them alongside it isn't protected.

[V] `listener/wazuh_listener.py` imports `batch_update_duplicates` but never calls it. It isn't a second caller.

## Prioritization

The tiers rank what a duplicate does:

- **Tier 1** can't be undone without analysis. Counters, the learned chain model, or identity itself end up wrong, and the damage feeds risk scores, CTI and predictions.
- **Tier 2** reverses or inflates state, but the values are bounded and later alerts usually correct them.
- **Tier 3** recomputes or overwrites, so the next alert generally repairs it.

## First migration candidate

[P] **W07 `update_duplicate_event`, on the unattributed path first** (`realtime_socgraph.py:_process_unattributed_alert` L170-286), subject to D13.

| Why | Evidence |
|---|---|
| Real risk, not just ease | It's additive, so it's the double-count class Phase 2.1 reproduced, and today's discarded result can lose it silently |
| Clean identity | One duplicate alert is one logical request. The key needs only D1 |
| Stable payload | No random ids, no value read back from the graph. `event_id` comes from the dedup cache and is carried in the journal |
| No result needed | Nothing later uses its output, so `ALREADY_COMPLETED` can't starve a later step |
| One statement | It fits `write_query` as it stands, with no transaction rewrite |
| Lowest blast radius in Tier 1 | On the unattributed path `tps = 0`, and the statement stops before Technique and Campaign, so a wrong outcome affects one event's count |

Why not the others:

| Writer | Why not first |
|---|---|
| W14 `store_cti_confidence` | Easiest, but absolute values mean it wouldn't test double-count protection. The result would prove little |
| W08 | The biggest risk, but its identity is unresolved |
| W05, W06, W01, W16 | They need deterministic ids and return values used by later steps |
| W10 | Needs a transaction rewrite, and it's a global learned model, so a mistake has a wide effect |
