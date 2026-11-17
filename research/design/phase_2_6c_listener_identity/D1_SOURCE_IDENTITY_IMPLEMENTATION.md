# D1: source identity implementation

Backend paths are relative to `isfcr/CYUKTI/backend/`. **[V]** verified from code, **[M]** measured offline, **[T]** reproduced in tests, **[N]** not yet validated in production.

## Format

| Form | Value | Used by |
|---|---|---|
| Source key | `wazuh:<manager>/<id>` | [V] The listener's duplicate check, as `alert["alert_id"]` |
| Request key | `cyukti.v1|wazuh:<manager>/<id>|<operation>` | Nothing yet. For a later coordinator |

| Part | Source | Rule |
|---|---|---|
| `<id>` | `alert["id"]` | Exact `str`, matching `^[0-9]+\.[0-9]+$` (ASCII digits only), at most 64 characters. **Kept as the exact string:** never parsed, so `1.10` and `1.1` stay different |
| `<manager>` | `alert["manager"]["name"]` | Exact `str`, not empty. Percent-encoded as UTF-8, leaving `A-Z a-z 0-9 - . _ ~`, at most 200 characters encoded |
| `<operation>` | Supplied by a caller | `[A-Za-z0-9_.-]+`. Never part of the source identity |

## Code

| Item | Location |
|---|---|
| Extraction | `listener/utils.py:extract_source_identity` L57-85 |
| The value | `listener/utils.py:SourceIdentity` L38-54 |
| Placing it on the alert | `listener/utils.py:assign_source_identity` L88-93 |
| Request key | `listener/utils.py:request_key` L96-102 |
| Called by the reader | `listener/wazuh_listener.py:AlertFileHandler.read_new_lines` L279-360 |

## Field trace

[V] From the raw line to `process_alert`:

1. `read_new_lines` parses one line with `safe_json_load`, giving the whole alert as a dict.
2. `assign_source_identity` adds `alert["source_identity"]` and `alert["alert_id"]`. Nothing else on the alert changes. [T]
3. `listener/wazuh_listener.py:accept_alert` L128-162 decides on `alert_id`, and puts the same dict on `alert_queue`.
4. `listener/wazuh_listener.py:queue_worker` L164-181 passes that dict to `process_alert`.

[T] A test confirms `source_identity` reaches `process_alert`. [V] Nothing in the backend dumps or iterates a whole alert, and `investigation_payload` takes named fields only, so the new keys don't reach the graph. [T] Tested on both routes.

## What's established about the fields

[M] From the 9,015-alert sample:

| Field | Result |
|---|---|
| `id` | A `str` on 9,015. All distinct. All match the pattern |
| `manager` | A dict on 9,015, holding only `name` |
| `manager.name` | A `str` on 9,015, one distinct value |
| `cluster` | Absent on all, so it isn't used |

**[N] This is the sample, not your deployment.** Its provenance is undocumented, and the deployed version wasn't inspected, because `/var/ossec` is off limits. Phase 2.6A's reading of the v4.9.0 source isn't relied on here for behaviour: the code checks the shape and handles anything else.

## Manager identity

[V] Only `alert["manager"]["name"]` is used. **Nothing is derived from an agent name, hostname, location or cluster field.** [T] Tests set those fields to the manager's name with the manager block missing or empty, and require no identity.

[M] `agent.name` equals `manager.name` on 341 sample alerts, all with agent id `000`, the manager's own agent. That's consistent with `manager.name` being the manager, and isn't proof of it.

## When identity is unavailable

| Status | Cause |
|---|---|
| `MISSING_ID` | No `id`, or null |
| `EMPTY_ID` | `""` |
| `MALFORMED_ID` | Not a string, wrong shape, or too long |
| `MISSING_MANAGER` | No manager block, no name, null or empty |
| `MALFORMED_MANAGER` | Block or name of the wrong type, a lone surrogate, or too long |

| Behaviour | Detail |
|---|---|
| Delivered | [V] `listener/wazuh_listener.py:accept_alert` L128-162 queues it, and records nothing |
| Never refused for its content | [T] Identical alerts with no identity are all delivered |
| Queue failure | [T] Retained and retried, the same as any alert |
| Counted | `identity_unavailable`, by status |
| Logged | [V] `listener/wazuh_listener.py:_note_identity_unavailable` L206-216, at the 1st and every 1000th of each status, with a content fingerprint and no alert values |

**The trade-off, stated plainly:** such an alert has no suppression, so a re-read delivers it again. That's preferred to a guess that can discard a real alert. No random or unstable id is introduced.

[V] `listener/utils.py:content_fingerprint` L132-141 exists only for that log line, so a reader can tell whether such alerts repeat. It decides nothing. [T]

## Exactness and determinism

[T] Tests show:

- The same result from a re-parsed copy.
- Only those two fields matter: changing the timestamp, rule, agent, location, data or `full_log` changes nothing.
- A manager name can't forge separators: `a/b` becomes `a%2Fb`, and `%2F` becomes `%252F`.
- `[M]` Every key the sample produces is distinct.
- Request keys pass the ledger's real `_checked_key`, unchanged.

## Exposure

[V] A source key holds a host name, so logs carry `listener/utils.py:source_ref` L105-108, a 12-character hash. [T] Tested: no manager name or id reaches the log.

## What this doesn't claim

**No global uniqueness of Wazuh ids.** Restart, rotation, clock step-back, restore, multiple managers and peak load are all unvalidated. See `PRODUCTION_VALIDATION_GAPS.md`.

If an id were reused within the cache's lifetime, the second alert would be refused as a repeat. The duplicate check has no way to notice. **That's the residual risk of adopting this identity.** [M] It hasn't been observed: there are 0 repeats in the sample.
