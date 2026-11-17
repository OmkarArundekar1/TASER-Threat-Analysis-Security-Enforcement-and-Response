# Test and fault results

All recorded runs used an empty network namespace and started no service. **[M]** measured.

## Totals

| Run | Result | Output |
|---|---|---|
| `tests/test_listener_source_identity.py` | **157 passed** | `evidence/test_new_output.txt` |
| Same, 3 more repeats | 157 passed each | `evidence/test_repeat_stability.txt` |
| Existing listener, audit logging, failure injection, production safety and MITRE routing | **185 passed** | `evidence/test_existing_listener_output.txt` |
| Journal and ledger, to show no effect | **712 passed** | `evidence/test_journal_ledger_output.txt` |
| **Full backend** | **2,543 passed, 29 skipped, 0 failed** | `evidence/test_full_suite_output.txt` |

The 29 skips are the same count as before this phase. The 4 sample-based tests ran in these results. They'd skip on a machine without the gitignored sample.

## Coverage against the brief's list

**Deduplication,** in `tests/test_listener_source_identity.py`:

| Case | Test |
|---|---|
| `srcip`/`dstip`, `src_ip`/`dest_ip`, mixed naming, both with conflicting values | `test_network_fields_follow_one_order_and_values_are_unchanged`, `test_source_identity_alone_decides_what_is_a_repeat` |
| Same id and manager, different managers, different ids with identical content | `test_source_identity_alone_decides_what_is_a_repeat` |
| Missing, empty, malformed | `test_without_identity_identical_alerts_are_all_delivered_never_collapsed`, `test_unusable_identity_is_named_and_gives_no_key` |
| Replayed alert | `test_replaying_a_file_in_one_process_recognizes_every_identity` |
| Repeated timestamp | `test_repeated_timestamps_are_not_repeats` |
| Different `rule.firedtimes` | `test_source_identity_alone_decides_what_is_a_repeat` |
| Attributed and unattributed | `test_the_identity_fields_do_not_change_routing`, `test_a_duplicate_on_the_unattributed_path_still_updates_its_event`, `test_a_duplicate_on_the_attributed_path_still_goes_to_the_buffer` |

**Offset and queue:**

| Case | Outcome asserted | Test |
|---|---|---|
| Normal complete read | Accepted | `test_a_normal_complete_read_accepts_all_and_saves_the_end` |
| Crash before offset save | Retained, redelivered | `test_killed_while_reading_nothing_unread_is_skipped_and_what_was_queued_is_delivered_again` |
| Crash after queue submission | Redelivered | `test_killed_after_queue_submission_before_the_offset_is_saved_alerts_are_redelivered_not_lost` |
| Queue submission failure | Retained for retry | `test_a_queue_failure_retains_the_alert_and_a_retry_accepts_it_once` |
| Duplicate refusal | Rejected, legitimately | `test_a_refused_alert_was_already_accepted_under_that_identity` |
| Missing id | Accepted, no suppression | `test_an_alert_with_no_id_field_at_all_is_delivered` |
| Restart and replay of one file | Redelivered | `test_restart_replay_of_the_whole_file_delivers_each_alert_again_because_the_cache_is_memory` |
| Partial final line | Retained | `test_a_partial_final_line_is_retained_and_delivered_once_complete` |
| Truncated or malformed JSON | Failed explicitly | `test_a_complete_line_that_can_never_be_an_alert_is_failed_explicitly_and_does_not_stop_the_reader` |
| Rotation, controlled fixture | Accepted, and two known gaps pinned | `test_rotation_to_a_smaller_file_starts_it_from_the_beginning`, `test_known_gap_a_new_file_already_larger_than_the_old_offset_is_read_from_the_wrong_place`, `test_known_gap_alerts_appended_just_before_rotation_are_not_read` |

**Sample:**

| Requirement | Test |
|---|---|
| The old key really collapsed 70, in 39 groups | `test_the_old_key_really_did_collapse_seventy_of_them` |
| All accepted through the real reader, then all recognized, and the 70 kept their ids | `test_the_sample_through_the_real_reader_all_accepted_then_all_recognized` |
| The seven keep their own values | `test_the_seven_with_a_different_attacker_or_victim_reach_the_pipeline_with_their_own_values` |

## Mutation check

[M] 54 deliberate bugs, each on a copy in a temporary overlay. **54 caught, 0 survived,** and the no-op control passes. Script: `evidence/mutation_check.py`. Output: `evidence/mutation_output.txt`.

**It found a real gap, now fixed:** "manager taken from the agent" survived, because my test removed the whole manager block and never tried a block with no name. `test_a_manager_block_without_a_name_is_never_filled_from_the_agent` now covers it.

**Earlier test errors I also corrected:** four of my first tests passed `manager=` through a helper that wrapped the value, so they didn't test what they named. They failed, and I fixed the tests rather than the code.

**Limits:**

- The mutants are ones I thought of.
- **The 4 sample-based tests skip in the overlay,** because the overlay has no sample, so those didn't contribute to catching mutants. The replay evidence covers that ground separately.

## Network attempts

[M] The full run reports 16 lines of unexpected network attempts, the same older set. None comes from the new tests. They remain untraced.

## Limitations

- **The live listener never ran.** The real source ran with stubs, and no `start_listener` call in these tests other than existing isolated ones.
- **`process_alert` paths use stubs,** as the existing integration tests do.
- **Power loss isn't tested.** The offset's fsync rests on the filesystem. SIGKILL keeps the page cache.
- **The parent directory isn't fsynced after the rename,** so power loss could bring back an older offset. That's safe in the direction of repeating rather than skipping.
- **Race tests show those runs,** and can't force every interleaving.
