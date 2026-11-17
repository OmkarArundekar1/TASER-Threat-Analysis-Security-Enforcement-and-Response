# Logical request identity contract

Phase 2.5, design only. **Nothing here is implemented.** Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Labels:** **[V]** verified current behaviour, **[P]** proposed behaviour, **[D]** open decision, **[A]** assumption requiring verification, **[L]** known limitation.

## The kinds of identity

These are different things, and confusing them causes real errors.

| Identity | What it names | Current source | Example |
|---|---|---|---|
| Event (source) | One alert from Wazuh | [V] Wazuh's own `id` exists but isn't used | `1773199644.1234` |
| Event (graph) | One AttackEvent node | [V] Random `event_id`, and many duplicate alerts share one | `e.event_id` |
| Campaign | A group of events for an attacker and victim | [V] Random `CAMP_…` | `c.campaign_id` |
| Operation | A group of campaigns | [V] Random `OP_…` | `o.operation_id` |
| Logical request | One intended graph change | **[P] New: the ledger key** | see below |
| Execution attempt | One claim to run a request | [V] `attempt_count` in the ledger | `3` |
| Transaction try | One driver retry inside an attempt | [V] `WriteResult.attempts` | `2` |
| Statement | One query in a batch | [V] Position in the batch, digest in the ledger | `statement 1` |

- **A campaign id can't be a request key.** A campaign receives many separate, legitimate writes.
- **A graph event id can't be a request key.** Many duplicate alerts legitimately update one event.
- **A fingerprint can't be a request key.** Two duplicate alerts can have identical payloads and still both be real.
- **Attempts aren't transaction tries.** One ledger attempt can contain several driver retries, and they're counted separately.

## The source identity problem

[V] `generate_alert_id` exists (`listener/utils.py:generate_alert_id` L3-27), but it can't anchor request identity:

1. **It's dropped.** The listener sets `alert["alert_id"]` (`listener/wazuh_listener.py:AlertFileHandler.read_new_lines` L228-306) and uses it only for `accept_alert`. `process_alert` never reads it, and it never reaches the graph.
2. **The replay path never sets it.** `replay/harness.py` calls `process_alert` directly (`replay/harness.py:replay` L127-175).
3. **It collides.** It hashes timestamp, rule, agent, `srcip`, `dstip` and location, and it doesn't read `src_ip`, the Suricata spelling that `extract_iocs` does read (`realtime_socgraph.py:extract_iocs` L80-105).

[V] **Measured on the repository's sample, `isfcr/CYUKTI/data/raw_logs/alerts.json`:**

| Measure | Result |
|---|---|
| Alerts | 9,015 |
| With a top-level Wazuh `id` | 9,015 |
| Distinct Wazuh `id` values | 9,015 |
| Distinct `alert_id` values | 8,945 |
| Alerts sharing an `alert_id` while their Wazuh `id` and content differ | **70** |

[V] Colliding alerts share a timestamp, so they fall inside `accept_alert`'s 600-second window (`listener/wazuh_listener.py:accept_alert` L125-148). A listener given this file would discard those 70 as "duplicate raw alert". I measured the collision, and I didn't run the listener. This is an existing alert-loss issue, separate from this design (D2).

[A] Wazuh's `id` is unique and stable in production, including across log rotation. It's unique in this sample, which is one file. Its form looks like a time plus a file offset, which suggests but doesn't prove uniqueness. **Verify against production-shaped data before relying on it.**

**[P] Recommended source identity (D1):**

1. `alert["id"]` when present.
2. Otherwise SHA-256 of the raw alert's canonical JSON, marked as `sha:` in the key so it can't be mistaken for a Wazuh id.

A retried alert keeps its identity because the journal holds the original payload. Nothing recomputes it.

## Key format

[P] A ledger key is:

```
cyukti.v1|<source_kind>:<source_id>|<operation>[|<discriminator>...]
```

| Part | Meaning | Rule |
|---|---|---|
| `cyukti.v1` | The key format version | Changes only if the format itself changes |
| `source_kind` | Where the identity comes from | `wazuh` or `sha`; `maint` for W03's maintenance path |
| `source_id` | The source event | From D1, never a timestamp or a value made at call time |
| `operation` | The logical operation, from a fixed registry | For example `attack_event.create`. Not a function name, which could be renamed |
| `discriminator` | Only when one event legitimately makes several of the same operation | For example a campaign id, or previous and current technique |

[V] It fits the ledger's rules: one string, 1 to 512 characters, no control characters or surrounding whitespace, matched exactly (`neo4j_idempotency_ledger.py:_checked_key` L377-388). Wazuh ids and these operation names are short enough, and a test should pin the longest real key.

[P] **Keys are built in one place,** a registry function per operation. Callers don't assemble strings. A missing source identity raises an error, and it never falls back to something generated.

[P] **Never used in a key:** timestamps taken at call time, random values, process ids, execution order, counters, or anything read back from the graph at retry time.

## Payload fingerprint

[V] `fingerprint_payload` covers the operation name, `return_rows` and the ordered statements with exact types, under a frozen versioned format (`neo4j_idempotency_ledger.py:fingerprint_payload` L320-356). The ledger stores digests per statement and no values.

[V] **A consequence that shapes the migration: random ids inside a payload break identity.** `create_attack_event` puts a new `uuid4` in its parameters, so two runs of the same alert have different fingerprints. A retry through the same call path would be reported as a conflict. So:

- [P] A retry always uses the payload **stored in the journal,** and it's never rebuilt by the caller.
- [P] Writers that create ids need those ids derived from the key (D7), for example `uuid5(namespace, key)`. Then re-delivery of an alert also produces the same payload.

[V] Query text is in the fingerprint with no normalization, and server-side `datetime()` isn't a parameter, so it doesn't affect it.

## What makes two invocations the same request

| Case | Meaning | Result |
|---|---|---|
| Same key, same payload | The same logical request | [V] `DUPLICATE`, grants nothing. [P] The coordinator acts on the state it finds |
| Same key, different payload | A bug, a code change, or a routing change | [V] `IdempotencyConflictError`, audited, existing request untouched. [P] Never executes; needs a person |
| Different keys, same payload | Separate legitimate requests | [V] Both `NEW` |
| Same event, different operations | Separate requests | [P] `operation` differs in the key |
| Same operation retried after an uncertain outcome | The same request, a later attempt | [P] Same key, new attempt, only after explicit resolution and authorization. See `JOURNAL_LEDGER_COORDINATION_CONTRACT.md` |

**Payload equality doesn't mean request equality.** Two different duplicate alerts with the same `event_id`, `tps` and payload are two real occurrences, and both must count.

## How identity survives a restart

[V] The ledger and journal are durable, and their keys and ids persist. The source id doesn't survive on its own:

- [V] After a crash, queued alerts are gone from memory, and the offset has already moved past them.
- [P] So the **key must be persisted before anything can be lost:** `register` happens before execution, and the journal holds the payload.
- [L] An alert that dies in the queue before any `register` call has no record anywhere. Identity can't fix that: it needs offset integration (D10).

## Versions and schema changes

- [P] The key's `v1` covers only the key format.
- [P] Changing a query changes the fingerprint, so an old request presented with new text **conflicts.** That's deliberate: it stops the new code quietly running against an uncertain old request. A deployment should drain or resolve open requests first.
- [V] A stored request under another fingerprint version raises `FingerprintVersionError` and isn't treated as new (`neo4j_idempotency_ledger.py:IdempotencyLedger._require_comparable` L899-903).

## Multi-statement requests

[P] One logical request can hold several statements in one transaction, through `write_statements`, and the ledger digests each one. **Statements don't get their own keys.**

W10 is the example: its two writes should become one request. Keeping them as two would preserve today's stale-`confidence` window.

[P] Separate steps of `process_alert` are separate requests, because each already commits on its own today. Making an alert atomic would be a much larger behaviour change (D10).

## Conflicting reuse and unknown identity

- [P] **Conflict:** `register` raises. The coordinator returns the error, doesn't execute, and doesn't touch the journal. The audit row names which part differs. An operator investigates.
- [P] **Unknown source identity:** the registry refuses to build a key. The write doesn't silently go through unprotected unless D14's bypass policy says so during migration.
- [P] **Unresolved writers (W08, W19, W20):** not ledgered until D8 and D9 are decided.

## Examples from real writers

Keys below use an example Wazuh id, `1773199644.1234`.

**1. W05 `create_attack_event`, a new attributed alert**

```
cyukti.v1|wazuh:1773199644.1234|attack_event.create
```
One alert makes one event. Re-delivery of the alert gives `DUPLICATE`. Depends on D7 for a stable `event_id`.

**2. W07 `update_duplicate_event`, two duplicate alerts on one event**

```
cyukti.v1|wazuh:1773199644.1234|duplicate_event.update
cyukti.v1|wazuh:1773199651.9876|duplicate_event.update
```
Same `event_id`, possibly identical payloads, different keys: **both count.** This is why a fingerprint can't be the key.

**3. W10 `update_attack_chain`, one transition**

```
cyukti.v1|wazuh:1773199644.1234|chain.transition|T1110|T1078
```
Two statements, one request. A second alert making the same transition has a different key and legitimately adds to `count`.

**4. Same event, different operations**

```
cyukti.v1|wazuh:1773199644.1234|attack_event.create
cyukti.v1|wazuh:1773199644.1234|campaign.cti|CAMP_ABC12345
```
One alert, several independent requests. One can complete while another is uncertain, and they're resolved separately.

**5. W03 `close_campaign_db`, two paths**

```
cyukti.v1|wazuh:1773199644.1234|campaign.close|CAMP_ABC12345
cyukti.v1|maint:CAMP_ABC12345|campaign.close|2026-03-11T03:29:24+00:00
```
The alert path has a source event. The maintenance path doesn't, so it uses the derived `closed_at`. If `last_seen` moved between them, the second is a separate request, and that's the correct reading.

**6. W01 `create_campaign_db`, a retry after an uncertain outcome**

```
cyukti.v1|wazuh:1773199644.1234|campaign.create
```
Ledger attempt 1 is uncertain. If someone finds it didn't commit and authorizes a retry, attempt 2 uses the same key and the journal's stored payload, so the same `campaign_id`. A new key would risk two campaigns.

**7. W08 `batch_update_duplicates`, no key**

There's no defensible key today. The batch's contents change as it merges, and contributing alerts aren't kept. See D8. **Marked UNRESOLVED rather than invented.**
