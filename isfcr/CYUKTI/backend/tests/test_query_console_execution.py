"""
Regression test for a real bug found running /api/query against the
live Neo4j instance: the validated handler called session.read_transaction(),
which was removed in the installed neo4j driver (6.2.0) — every
legitimate read-only query failed with
"'Session' object has no attribute 'read_transaction'", even though
mutation rejection worked. Fixed to session.execute_read().
"""

import dashboard_api


class _FakeSession:
    def __init__(self, return_value):
        self._return_value = return_value

    def execute_read(self, fn, query):
        # mirror the real driver's contract: call fn(tx, query) and return its result
        return fn(self, query)

    def run(self, query):
        return _FakeResult()


class _FakeResult:
    def keys(self):
        return ["n"]

    def __iter__(self):
        return iter([{"n": 33}])


def test_query_console_uses_execute_read_not_removed_api(monkeypatch):
    fake_session = _FakeSession(return_value=None)

    class _FakeDriver:
        def session(self):
            class _Ctx:
                def __enter__(self_):
                    return fake_session

                def __exit__(self_, *a):
                    return False

            return _Ctx()

    monkeypatch.setattr(dashboard_api, "driver", _FakeDriver())

    client = dashboard_api.app.test_client()
    resp = client.post("/api/query", json={"query": "MATCH (c:Campaign) RETURN count(c) AS n"})

    assert resp.status_code == 200
    assert not hasattr(fake_session, "read_transaction")
