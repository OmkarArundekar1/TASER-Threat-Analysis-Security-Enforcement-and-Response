"""
experiments/record_experiment.py
===================================
Validates an experiment record against schema.json and saves it under
experiments/records/. Deliberately does NOT generate any of the
record's content -- it only validates and stores what the caller
already observed. See experiments/README.md.

Usage:
    python record_experiment.py path/to/my_experiment.json
"""

from __future__ import annotations

import json
import os
import sys

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCHEMA_PATH = os.path.join(_HERE, "schema.json")
_RECORDS_DIR = os.path.join(_HERE, "records")


def _load_schema() -> dict:
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


def validate_record(record: dict) -> list[str]:
    """Returns a list of validation problems (empty if valid). Falls
    back to checking only the schema's `required` fields if the
    `jsonschema` package isn't installed, rather than silently skipping
    validation."""
    schema = _load_schema()

    if _HAS_JSONSCHEMA:
        validator = jsonschema.Draft7Validator(schema)
        return [str(e.message) for e in validator.iter_errors(record)]

    problems = []
    for field in schema.get("required", []):
        if field not in record:
            problems.append(f"Missing required field: {field}")
    if "errors" not in record:
        problems.append("Missing required field: errors (use [] if genuinely none)")
    return problems


def save_record(record: dict, path: str | None = None) -> str:
    problems = validate_record(record)
    if problems:
        raise ValueError("Experiment record failed validation:\n" + "\n".join(f"  - {p}" for p in problems))

    os.makedirs(_RECORDS_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(_RECORDS_DIR, f"{record['experiment_id']}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def main():
    if len(sys.argv) != 2:
        print("Usage: python record_experiment.py path/to/experiment.json", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        record = json.load(f)

    problems = validate_record(record)
    if problems:
        print("INVALID experiment record:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    saved_path = save_record(record)
    print(f"Valid. Saved to {saved_path}")


if __name__ == "__main__":
    main()
