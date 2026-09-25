# CYUKTI Experiment Recording

A reproducible mechanism for recording what actually happened when a real (or explicitly-marked synthetic) scenario was run through CYUKTI — never a place to insert fabricated attack results.

## Structure

```
experiments/
  schema.json        -- JSON Schema (Draft-07) an experiment record must satisfy
  record_experiment.py  -- validates a record against schema.json, then saves it
  examples/          -- fixed, committed example(s), for reference
  records/           -- where record_experiment.py actually saves validated records
```

## The `data_provenance` field is mandatory-in-spirit

Every record should state one of:
- `real_live_environment` — every field was observed from a real running system, at the timestamp given.
- `synthetic_reconstruction` — inputs are real but some downstream fields were reconstructed/estimated rather than freshly re-observed live; the record must say which fields and why.
- `unit_test_fixture` — entirely synthetic, for testing the recording mechanism itself; must never be cited as evidence of real system behavior.

## The real example (`examples/EXP-2026-09-25-nmap-scan-01.json`)

Populated with actual data from this repository's own investigation of a real live Nmap scan (see `INCIDENT_VIEW.md`'s "Real-world validation" and `MITRE_MAPPING.md`) — real attacker/victim IPs, the real Wazuh alert, the real resolved MITRE technique, the real campaign ID, and the real threat-qualification/campaign-selection API responses queried directly from the live backend. Fields the investigation didn't specifically measure for this record (`evidence`, `next_best_evidence`, `multi_rag`, `response_plan`, `latency`) are honestly `null`, not fabricated.

Note this example also records a real, useful negative result: the `expected_behavior` field states the original (wrong) assumption that this reconnaissance-only event would NOT reach `QUALIFIED_THREAT` — and `actual_behavior` records that it did, with the explanation. An experiment log should be honest about corrected assumptions, not just confirmed ones.

## Recording a new experiment

1. Run the real scenario (a real Wazuh alert, a real API call, a real test).
2. Fill in a JSON file matching `schema.json`, using only what was actually observed.
3. Validate and save:

```bash
cd experiments
python record_experiment.py path/to/your_experiment.json
```

`record_experiment.validate_record()` and `.save_record()` are also importable directly for scripted/automated experiment recording.

## Tests

`backend/tests/test_experiment_recording.py` (6) — validates the recorder's own logic (accepts valid records, rejects specific defects, allows the documented `null` fields) and confirms the committed real example itself validates cleanly against the schema.
