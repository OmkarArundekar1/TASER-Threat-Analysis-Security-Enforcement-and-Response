"""A pytest plugin for the network-attempt audit. It adds a stack to each attempt the existing test guard refuses, and suppresses, mocks and allows
nothing: the guard still blocks every one. Loaded with -p, from outside the backend, with nothing in the backend changed.
Writes one JSON line per attempt to the file named by CYUKTI_NETWORK_AUDIT_LOG, naming the code that led there."""
import json
import os
import traceback

LOG = os.environ["CYUKTI_NETWORK_AUDIT_LOG"]


def pytest_collection_finish(session):
    from tests import _production_safety as safety
    real = safety.NetworkGuard._record

    def recording(self, kind, target):
        real(self, kind, target)
        record = self.records[-1]
        frames = [f for f in traceback.extract_stack() if "/backend/" in f.filename and "/tests/_production_safety" not in f.filename]
        record_out = {"test": record["test"], "operation": kind, "destination": record["destination"],
                      "stack": [f"{f.filename.split('/backend/')[-1]}:{f.lineno}:{f.name}" for f in frames][-8:]}
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record_out) + "\n")
    safety.NetworkGuard._record = recording
