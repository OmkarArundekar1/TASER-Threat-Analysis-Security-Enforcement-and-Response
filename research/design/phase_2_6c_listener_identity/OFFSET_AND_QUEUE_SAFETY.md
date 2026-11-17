# Offset and queue safety

Backend paths are relative to `isfcr/CYUKTI/backend/`. **[V]** verified from code, **[T]** reproduced in tests, **[M]** measured offline. Crash tests use real SIGKILL on a subprocess running the real listener source.

## Four different things

| Stage | Means | What guarantees it |
|---|---|---|
| Source-level suppression | This process already accepted this manager and id | Memory only. Up to 600 s and 10,000 entries. **Gone at restart** |
| Queue delivery | The alert entered the queue | Memory only. **Gone at a crash** |
| Processing completion | `process_alert` finished | Nothing durable |
| Neo4j write completion | The graph holds it | The future coordinator. **Not this phase** |

**This phase offers no exactly-once processing.** After a crash, an alert may be redelivered, or lost if it was only in the queue.

## Changes made, and why each was necessary

Each closes a demonstrated way to skip alerts for good. None changes the offset format, or adds a queue.

| Change | Before | After | Evidence |
|---|---|---|---|
| **Complete lines that can't parse no longer stop the reader.** In `listener/wazuh_listener.py:AlertFileHandler.read_new_lines` L279-360 | [M] A truncated line ending in a newline stopped the reader for good, because the code waited for a line that was already complete. Nothing behind it was read | Failed explicitly: counted in `unreadable_lines`, logged, passed over | `evidence/replay_before_fix.txt`, `evidence/replay_after_fix.txt` |
| **JSON that isn't an object no longer stops the reader** | [M] An array or number raised on every poll, the same outcome | The same as above | Same |
| **The offset is saved atomically.** `listener/wazuh_listener.py:_save_offset` L224-232 | [V] `open("w")` truncated first. A crash there left an empty file, which means "start at the end of the log", skipping every unread alert | Temporary file, fsync, `os.replace`. [T] Killed mid-save, the old value stays valid | Test |
| **A first start saves its starting point at once.** `listener/wazuh_listener.py:AlertFileHandler.__init__` L236-277 | [V] A crash before the first save meant "the end of the log" again at next start, past what arrived in between | Saved at once. A failure is logged, and doesn't stop the start | Test |

The offset is also now rewritten only when it moved, because an fsync on every 0.5 s poll would be wasted.

## The crash windows

| Window | What happens to each alert | Test | Result |
|---|---|---|---|
| Killed while reading | Queued ones are **redelivered.** Unread ones are **read** | `test_killed_while_reading_nothing_unread_is_skipped_and_what_was_queued_is_delivered_again` | No loss |
| Killed after queue submission, before the offset save | **Redelivered** | `test_killed_after_queue_submission_before_the_offset_is_saved_alerts_are_redelivered_not_lost` | No loss |
| Killed during the offset save | **Redelivered,** from a valid older offset | `test_killed_in_the_middle_of_saving_the_offset_leaves_a_valid_older_value` | No loss |
| Killed after a first start | Later alerts are **read** | `test_killed_after_a_first_start_what_arrives_later_is_not_skipped` | No loss |
| **Killed after the offset save, with alerts in the queue** | **Lost** | `test_known_gap_killed_after_the_offset_is_saved_alerts_still_in_the_queue_are_lost` | **Known gap, below** |
| Clean restart | Only new alerts are read | `test_a_clean_restart_at_the_saved_offset_delivers_only_what_is_new` | No loss, no repeat |
| Restart, replaying the whole file | All **redelivered** | `test_restart_replay_of_the_whole_file_delivers_each_alert_again_because_the_cache_is_memory` | No loss. **Repeats** |

## Other required verifications

| Requirement | Result |
|---|---|
| A duplicate refusal was already accounted for | [T] `test_a_refused_alert_was_already_accepted_under_that_identity`. [V] An id is recorded only after `put_nowait` succeeds |
| Missing identity doesn't discard | [T] Delivered, and the reader moves on |
| A queue failure can't pass an unaccepted alert | [T] `test_a_queue_failure_retains_the_alert_and_a_retry_accepts_it_once`, for `Full` and other errors, with and without identity |
| Partial final line | [T] Retained until complete, then delivered once, including a line ending in `}` |
| Offset format | [V] Unchanged: plain digits. [T] Existing values still read |
| A failed save | [T] The file stays whole and older, and the next poll saves |

## Known gaps, not fixed

Each has a test named `known_gap`, so that any change to it gets noticed.

| Gap | Effect | Why it wasn't fixed | Bounded proposal |
|---|---|---|---|
| **Alerts in the memory queue at a crash** | **Permanent loss** of up to 1,000 accepted, unprocessed alerts. The offset has already passed them | It needs a durable queue or crash resume, which you excluded. **It's D10** | Save the offset only as alerts complete, with the coordinator design |
| **Rotation to a file already larger than the old offset** | The start of the new file is **skipped** | The offset can't tell files apart, and fixing that changes its format | Store the file's inode beside the offset |
| **Alerts appended just before rotation** | **Lost:** the reader never looks at the old file | It needs the same file identity, and reading rotated files | With the above |
| **A line that isn't valid UTF-8** | The reader **stops** there, with a logged exception each poll, until someone acts. [M] The sample has none | Replacing bytes alters alert content, which your brief forbids. Skipping needs a decision | Read in binary, and fail only that line |
| **An unusable offset file** | Means "the end of the log", **skipping** what's unread. This code can no longer produce an empty one, but other causes could | It's the existing rule, and its alternative replays history | A decision for you |

## Repeats across a restart

[M] Replaying the sample after a restart delivers all 9,015 again, because the cache is memory.

**They then reach `process_alert`,** where the event-level dedup may treat them as duplicates and **count them again.** So a crash can inflate occurrence counts by what was re-read: **at most one read loop's worth,** since the offset is saved after each loop. This is existing behaviour, and the coordinator design addresses it.

**Alerts without identity** have no suppression, even within one process. A re-read in one process needs a failure in the read loop, because the offset otherwise stays ahead.

## Concurrency

[V] The listener has **one reader and one queue worker,** and `accept_alert` has one caller. [T] A test pins that.

| Test | Result |
|---|---|
| Real reader and worker, queue of 5, 1,000 alerts | Each delivered once, in order |
| Writer appending in pieces while the reader polls | No loss, no repeat, and no partial line treated as malformed |
| 16 threads presenting one identity | One acceptance. Beyond the real model: it checks the lock |

**[T] These show the behaviour in those runs. They can't force every interleaving.**
