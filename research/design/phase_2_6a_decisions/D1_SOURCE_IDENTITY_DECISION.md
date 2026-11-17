# D1: source identity

Phase 2.6A, evidence and proposal only. **Nothing here is implemented, and this decision is yours.** Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Labels:** **[V]** verified current behaviour, **[M]** measured evidence, **[P]** proposed behaviour, **[A]** assumption, **[D]** open decision, **[R]** requires production validation.

## Recommendation

**[P] Use `cyukti.v1|wazuh:<manager>/<id>|<operation>[|<discriminator>...]`, and withdraw the content-hash fallback that Phase 2.5 recommended.** An alert without a valid `id` and `manager.name` gets no key. What happens to it is D14's question.

The Wazuh `id` is the best identity available, but it's a by-product rather than a designed unique ID. It should be adopted only after the production checks at the end of this document.

## What the Wazuh `id` is

[V, external source, read through v4.9.0] Wazuh builds it in `src/analysisd/format/to_json.c`:

```c
snprintf(alert_id, 22, "%ld.%ld", (long int)lf->time.tv_sec, __crt_ftell);
cJSON_AddStringToObject(root, "id", alert_id);
```

It is added only when `lf->time.tv_sec` is non-zero. `__crt_ftell` is the `ftell` of the alerts log, or of the JSON log when that's the only output (`src/analysisd/analysisd.c`, writer threads).

So **`id` is `<event seconds>.<file offset>`.** It has no manager component.

[A] **The deployed Wazuh version is unknown.** No document found says this format is a stability guarantee, so it's an implementation detail that could change.

[M] The sample matches this reading (`evidence/d1_wazuh_id_analysis_output.txt`):

| Measure, `data/raw_logs/alerts.json` | Result |
|---|---|
| Alerts with an `id` | 9,015 of 9,015 |
| Distinct `id` values | 9,015 |
| Match `seconds.offset` | 9,015 |
| Seconds part equals the alert's `timestamp` second | 9,015 |
| Offset in file order | rose 9,014 times, never fell, never repeated |
| Busiest second | 409 alerts |
| `manager.name` present / distinct values | 9,015 / 1 |
| `cluster.name` or `cluster.node` present | 0 |
| Span | 58,274 s (0.67 days) on one day |

**What this sample can't show:** it holds one manager, no restart, no rotation, no restore and no reinstall. Its uniqueness says nothing about those cases.

## The eight questions

| # | Question | Answer | Label |
|---|---|---|---|
| 1 | Unique within one manager? | Expected while the clock doesn't step back, because the offset rises within a second. Not designed as unique. The offset is one global shared by two writer threads, and the source doesn't show the read is guarded, so two alerts could share one in principle. 0 repeats were measured across a 409-alert second | [V] construction, [M] sample, **[R] unresolved** |
| 1 | Unique across managers? | **No.** Nothing in the `id` names the manager, so two managers can produce the same pair | [V] |
| 1 | Other scope? | One manager's alert log | [V] |
| 2 | Same alert replayed, same ID? | Re-reading a stored line keeps its ID, because the ID is part of the line. If Wazuh itself re-analyses an event, it would write a new line with a new offset, and so a new ID. That second case wasn't tested | [V] first, [R] second |
| 3 | Reused after restart, restore or reinstall? | Reuse needs the same second **and** the same offset. A restart or reinstall moves time forward, so it shouldn't reuse one. A clock stepped back, or a restored machine running with an old clock, could. Rotation resets the offset, but the second differs except at the boundary | [V] reasoning, **[R] unresolved, none of it observed** |
| 4 | Manager identity always available to the listener? | In this sample, yes: the listener passes the whole parsed line onward. In v4.9.0 `manager.name` comes from `gethostname`, so it's always written. `cluster.*` is absent throughout the sample, so it **can't be relied on** | [M], [V], [R] |
| 5 | If absent? | No key is built, and it's never defaulted. A default such as `local` would silently merge managers. The B3 synthetic alerts lack both fields (8 of 8), so they need an explicit policy | [M], [P] |
| 6 | Available before dedup? | **Yes.** It's in the dict at `listener/wazuh_listener.py:AlertFileHandler.read_new_lines` L228-306, before `alert_id` is set, and at `realtime_socgraph.py:process_alert` L289-833 before `is_duplicate`. The Wazuh `id` is simply not used today | [V] |
| 7 | Survives restart and archival replay? | The identity does, because it's stored in the line. **Alerts in memory don't.** `replay/harness.py:replay` L127-175 passes each stored alert with its fields to `process_alert` | [V] |
| 8 | Separates two legitimate occurrences with identical content? | **Yes, when both have IDs:** the offsets differ. That's what content can't do | [V] |

## Candidates

| Candidate | Stable on replay | Separates identical occurrences | Across managers | Verdict |
|---|---|---|---|---|
| Wazuh `id` alone | Yes | Yes | **No** | Rejected: silently merges managers |
| Manager + `id` | Yes | Yes | Yes, if names differ | The core of the recommendation |
| `id` + source identity (agent, location) | Yes | Yes | **No:** agents and locations repeat across managers | Rejected: adds no manager separation, and the extra fields can vary |
| Content hash fallback | Yes | **No, in principle** | **No** | **Rejected, see below** |
| Composite with namespace and version | Yes | Yes | Yes | **Recommended** |

### Why a content hash isn't a substitute

[M] On this sample a content hash happens to work: hashing everything except `id` gives 9,015 distinct values. That result is misleading, for two reasons:

1. **It works because of two volatile fields.** Remove `timestamp` as well and 23 alerts merge into another (15 groups). What remains distinct depends mostly on `rule.firedtimes`, an in-memory Wazuh counter that's present on all 9,015 alerts here.
2. **It can't work in principle.** Two real occurrences with equal content have one hash, and nothing can tell them apart. Merging them is the silent loss this work exists to prevent.

The narrower hash the listener already uses shows the result in practice: `generate_alert_id` merges 70 of these alerts (`D2_ALERT_DEDUPLICATION_INVESTIGATION.md`).

## Proposed format

[P] All of this is a proposal. Example, with an illustrative manager name:

```
cyukti.v1|wazuh:manager-01/1773199644.1234|duplicate_event.update
```

| Part | Rule |
|---|---|
| Version | `cyukti.v1`, the format version only |
| Kind | `wazuh`. Other kinds must be declared, never inferred. `maint` remains for W03's maintenance path |
| Manager | `alert["manager"]["name"]`. Exact type `str`, non-empty, percent-encoded as UTF-8 with only `A-Z a-z 0-9 - . _ ~` left bare. No case folding. At most 200 characters once encoded |
| ID | `alert["id"]`. Exact type `str`, matching `^[0-9]+\.[0-9]+$`. **Kept as the exact string:** never parsed to a number, so `1.10` and `1.1` stay different |
| Operation | From the fixed registry, never a function name |
| Discriminator | Only where one alert makes several of the same operation |

**Canonical serialization:** plain concatenation with these separators. There's no JSON step, so nothing varies with key order. Encoding the manager guarantees `|`, `/` and `:` can't be forged by a field value.

[M] Keys proposed for the W07U pilot are 59 to 63 characters, and the ledger's real check accepts all 7,552 (`neo4j_idempotency_ledger.py:_checked_key` L377-388), which allows up to 512 characters. The sample's manager name needs no encoding.

### Missing identity

| Case | Result |
|---|---|
| `id` missing, not a string, or badly formed | Key registry raises. D14 decides what the writer does |
| `manager.name` missing, not a string, or empty | The same. **Never defaulted** |
| Test fixtures and B3 synthetic data | Need their own declared kind, for example `fixture:<dataset hash>/<line>`, supplied by the harness. Never inferred from content |
| Maintenance writes with no source alert | `maint`, per Phase 2.5 |

### Collisions

| Case | Result | Note |
|---|---|---|
| Same key, same payload | `DUPLICATE`. The coordinator acts on the existing state | **[V]** `neo4j_idempotency_ledger.py:IdempotencyLedger.register` L972-1012 |
| Same key, different payload | `IdempotencyConflictError`. It's audited, and nothing runs. A person investigates | **[V]** same function. Never re-keyed automatically |
| A reused ID, on a different alert | Normally a conflict, because `event_time` and the payload are in the fingerprint | [P] An exact repeat of content under a reused ID can't be detected. That's accepted, and listed for production validation |

**A conflict can also have innocent causes:** a code change, or re-delivery that resolves to a newer event (`D13_FIRST_WRITER_SELECTION.md`). The answer is the same, because a person can tell them apart and the store can't.

### Versions

- [P] `cyukti.v1` changes only if the format itself changes.
- [P] **A new format can't recognize old keys,** so an upgrade must drain or resolve open requests first. Old requests are never rewritten.
- [P] If Wazuh changes its `id` format, the pattern fails and those alerts get no key. That failure is loud, and that's the intent. New kinds or formats need a new decision.

## Evidence required before production adoption

[R] I can't collect these: `/var/ossec` is out of bounds and the deployed manager was not inspected. Each needs a copy you provide, or your authorization.

| # | Evidence | Pass |
|---|---|---|
| 1 | Deployed Wazuh version and its `id` construction | Same `seconds.offset` meaning |
| 2 | A copy of `alerts.json` across one rotation | No repeated `manager/id` |
| 3 | A copy across one manager restart | No repeated `manager/id`, and the fields are unchanged |
| 4 | Peak-load copy | No repeated `id`, which tests the shared-offset concern |
| 5 | `alerts_log` and `jsonout_output` settings | Understood, since the offset's source depends on them |
| 6 | Hostnames of every manager this could read, and any plans to rebuild them | Distinct names, or a decision on pinning them |
| 7 | Clock discipline on the manager | A stated answer on backward steps |
| 8 | Sample's own provenance | Which manager and version produced `data/raw_logs/alerts.json`. It's gitignored and undocumented |

**[V] Today there's one listener, reading one fixed path** (`listener/wazuh_listener.py` L26). So multi-manager collisions can't occur in the current layout. The namespace protects a move to a different one, and costs one field.

## If rejected

No writer can migrate, and D2's remediation has to choose its key alone.
