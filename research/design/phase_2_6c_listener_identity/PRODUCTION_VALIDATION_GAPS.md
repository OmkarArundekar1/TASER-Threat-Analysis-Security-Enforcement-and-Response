# Production validation gaps

**Everything here is [N]: not yet validated in production.** The fix is verified on one archived sample. I can't collect this evidence: `/var/ossec` and the live listener are off limits. Each item needs a copy you provide, or your authorization.

## What's known

| Fact | Source | Strength |
|---|---|---|
| The sample's `id` and `manager.name` have the expected shape | `evidence/replay_after_fix.txt` | Measured, **on one sample of undocumented origin** |
| Wazuh v4.9.0 builds `id` from event seconds and a file offset | Phase 2.6A, source reading | **Not the deployed version, as far as anyone has checked** |
| One listener reads one fixed file | `listener/wazuh_listener.py` L26 | Verified from code |

## Evidence still required

| # | Question | Evidence | Pass | If it fails |
|---|---|---|---|---|
| 1 | Which Wazuh version is deployed, and how does it build `id`? | Version, and a few alerts | `seconds.offset` form | **Every alert loses suppression, loudly, with none lost.** Revise the pattern |
| 2 | Do deployed alerts have `id` and `manager.name` on every line? | Copy of `alerts.json` | 100%, or an explained remainder | Those alerts have no suppression |
| 3 | Is `id` unique across **log rotation**? | Copy across one rotation | No repeated manager and id | A real alert could be refused |
| 4 | Across a **manager restart**? | Copy across one restart | The same | The same |
| 5 | Under **peak load**? The offset is one global shared by two writer threads, in v4.9.0 | Copy from a busy period | The same | The same, and this is the likeliest place |
| 6 | After a **clock step-back**? | Clock discipline on the manager, or a controlled test | Stated, or no repeats | The same |
| 7 | After a **restore or rebuild**? | Procedure review | No reuse within the cache's lifetime | The same, though the cache's 600 s limits the exposure |
| 8 | With **multiple managers?** | Hostnames, and any plans | Distinct names | The namespace would merge them |
| 9 | Is `manager.name` stable? | Deployment facts | Stable, or a change understood | A rename makes a re-read look new |
| 10 | Is the sample what production looks like? | Its origin | Stated | These results may not transfer |
| 11 | **How often did the old defect really fire?** | Count of `Duplicate raw alert ignored` in the listener's log, which I may not open | A number | Tells you how much the fix changes live counts |
| 12 | Are there non-UTF-8 lines? | Copy of `alerts.json` | None | The reader stops at them: see `OFFSET_AND_QUEUE_SAFETY.md` |
| 13 | How often does rotation happen with a backlog? | Rotation settings and real alert rates | Understood | Rotation gaps become real losses |
| 14 | Real `process_alert` time | Disposable-server benchmark | Under about 2.65 ms for the peak, or an accepted backlog | Sustained bursts queue |

## What a simple check would look like

A read-only scan of a copied `alerts.json`, run with `evidence/replay_listener.py` or similar, reporting: lines, valid identities, statuses of the rest, repeated keys, malformed lines. It needs no service, and writes nothing outside a temporary directory.

## Before enabling live

1. **Live counts will rise,** by 0.78% on the sample, because alerts that were discarded now arrive. Expect it, and don't read it as a regression.
2. **Watch the new diagnostic.** `Alert has no usable source identity` in the log means alerts are running without suppression. Any at all deserves a look, and a flood means the deployed format differs.
3. **Upgrade timing:** the cache starts empty, as at any restart, so it needs no migration.
4. **Rollback:** restore `listener/utils.py` and `listener/wazuh_listener.py` from `evidence/before_fix/`. The offset format is unchanged, so `offset.dat` carries over. **That would bring the defect back.**

The listener hasn't been started by this work, so none of this has been exercised live.
