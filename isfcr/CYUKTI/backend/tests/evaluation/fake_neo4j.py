"""A minimal fake Neo4j driver/session for evaluator unit tests, so
these tests never touch a real database. Supports exactly the
.session()/.run()/.single() surface the evaluators use."""


class FakeRecord(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key)


class FakeResult:
    def __init__(self, rows):
        self._rows = [FakeRecord(r) for r in rows]

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None


class FakeSession:
    def __init__(self, query_router):
        self._router = query_router

    def run(self, query, **params):
        return FakeResult(self._router(query, params))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeDriver:
    """query_router(query_text, params) -> list[dict] rows."""

    def __init__(self, query_router):
        self._router = query_router

    def session(self):
        return FakeSession(self._router)
