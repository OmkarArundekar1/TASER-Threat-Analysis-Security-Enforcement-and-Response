"""
Tests for ml/gnn/temporal_graphs.py — the temporal-snapshot extraction
layer for Objective C (GNN_REPRESENTATION_DESIGN.md Section 6),
real-timestamp-only, no SIMILAR_TO/RESEMBLES. Mocks only the database
boundary (neo4j_client.driver) for the two Cypher-issuing helpers, and
mocks the snapshot-graph builder itself (`_build_snapshot_graph`) for
the ordering/monotonicity/leakage tests -- those test this module's
own sequencing logic, not GraphBuilder/encode_graph (already covered
elsewhere).
"""

import networkx as nx
import pytest

import ml.gnn.temporal_graphs as temporal_graphs
from ml.gnn.temporal_graphs import (
    _distinct_event_timestamps,
    build_temporal_snapshots,
    list_temporal_campaign_ids,
)


class _FakeIterResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def run(self, query, **kwargs):
        return _FakeIterResult(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _FakeSession(self._rows)


# ---------------------------------------------------------------------------
# list_temporal_campaign_ids / _distinct_event_timestamps — DB boundary only
# ---------------------------------------------------------------------------

def test_list_temporal_campaign_ids_returns_query_order(monkeypatch):
    monkeypatch.setattr(
        "neo4j_client.driver", _FakeDriver([{"id": "camp-a"}, {"id": "camp-b"}]),
    )
    assert list_temporal_campaign_ids() == ["camp-a", "camp-b"]


def test_list_temporal_campaign_ids_empty(monkeypatch):
    monkeypatch.setattr("neo4j_client.driver", _FakeDriver([]))
    assert list_temporal_campaign_ids() == []


def test_distinct_event_timestamps_returns_query_order(monkeypatch):
    monkeypatch.setattr(
        "neo4j_client.driver", _FakeDriver([{"ts": "t1"}, {"ts": "t2"}, {"ts": "t3"}]),
    )
    assert _distinct_event_timestamps("camp-1") == ["t1", "t2", "t3"]


# ---------------------------------------------------------------------------
# build_temporal_snapshots — ordering, monotonicity, no future leakage
# ---------------------------------------------------------------------------

def test_build_temporal_snapshots_empty_for_fewer_than_two_timestamps(monkeypatch):
    monkeypatch.setattr(temporal_graphs, "_distinct_event_timestamps", lambda cid: [])
    assert build_temporal_snapshots("camp-1") == []

    monkeypatch.setattr(temporal_graphs, "_distinct_event_timestamps", lambda cid: ["t1"])
    assert build_temporal_snapshots("camp-1") == []


def test_build_temporal_snapshots_calls_builder_with_cutoffs_in_order(monkeypatch):
    timestamps = ["t1", "t2", "t3"]
    calls = []

    def fake_builder(campaign_id, cutoff):
        calls.append((campaign_id, cutoff))
        return nx.DiGraph()

    monkeypatch.setattr(temporal_graphs, "_distinct_event_timestamps", lambda cid: timestamps)
    monkeypatch.setattr(temporal_graphs, "_build_snapshot_graph", fake_builder)

    snapshots = build_temporal_snapshots("camp-1")

    assert calls == [("camp-1", "t1"), ("camp-1", "t2"), ("camp-1", "t3")]
    assert [s.cutoff_timestamp for s in snapshots] == ["t1", "t2", "t3"]
    assert [s.snapshot_index for s in snapshots] == [0, 1, 2]
    assert [s.num_distinct_timestamps_included for s in snapshots] == [1, 2, 3]


def test_build_temporal_snapshots_node_count_is_non_decreasing(monkeypatch):
    """Mechanical proxy for "no future leakage": each successive cutoff
    can only grow the graph (a real campaign's AttackEvents accumulate,
    never disappear), so snapshot i's node count must never exceed
    snapshot i+1's."""
    timestamps = ["t1", "t2", "t3", "t4"]

    def fake_builder(campaign_id, cutoff):
        # simulate real growth: node count = index of cutoff + 2
        idx = timestamps.index(cutoff)
        g = nx.DiGraph()
        for i in range(idx + 2):
            g.add_node(f"n{i}", labels=["Campaign"])
        return g

    monkeypatch.setattr(temporal_graphs, "_distinct_event_timestamps", lambda cid: timestamps)
    monkeypatch.setattr(temporal_graphs, "_build_snapshot_graph", fake_builder)

    snapshots = build_temporal_snapshots("camp-1")
    node_counts = [s.graph.num_nodes for s in snapshots]
    assert node_counts == sorted(node_counts)  # non-decreasing
    assert node_counts[0] < node_counts[-1]  # and genuinely grows, not flat


def test_build_temporal_snapshots_never_queries_cutoff_beyond_its_own_index(monkeypatch):
    """A snapshot's builder call must only ever receive cutoffs at or
    before its own position in the sorted timestamp list -- the
    concrete meaning of "no future information leak" (Phase I)."""
    timestamps = ["t1", "t2", "t3"]
    seen_cutoffs_at_call = []

    def fake_builder(campaign_id, cutoff):
        seen_cutoffs_at_call.append(cutoff)
        return nx.DiGraph()

    monkeypatch.setattr(temporal_graphs, "_distinct_event_timestamps", lambda cid: timestamps)
    monkeypatch.setattr(temporal_graphs, "_build_snapshot_graph", fake_builder)

    build_temporal_snapshots("camp-1")

    for call_index, cutoff in enumerate(seen_cutoffs_at_call):
        assert timestamps.index(cutoff) == call_index  # never a later timestamp than its own turn


# ---------------------------------------------------------------------------
# live Neo4j (optional -- skipped if unreachable), same convention as
# tests/test_campaign_graphs.py
# ---------------------------------------------------------------------------

def _neo4j_reachable():
    try:
        from neo4j_client import driver
        driver.verify_connectivity()
        return True
    except Exception:
        return False


requires_live_neo4j = pytest.mark.skipif(
    not _neo4j_reachable(),
    reason="No live Neo4j instance reachable in this environment.",
)


@requires_live_neo4j
def test_build_temporal_snapshots_against_live_neo4j_is_reproducible():
    ids = list_temporal_campaign_ids()
    if not ids:
        pytest.skip("No real campaigns with 2+ distinct event timestamps in this environment.")
    campaign_id = ids[0]
    first = build_temporal_snapshots(campaign_id)
    second = build_temporal_snapshots(campaign_id)
    assert [s.cutoff_timestamp for s in first] == [s.cutoff_timestamp for s in second]
    assert len(first) >= 2


@requires_live_neo4j
def test_build_temporal_snapshots_against_live_neo4j_grows_monotonically():
    ids = list_temporal_campaign_ids()
    if not ids:
        pytest.skip("No real campaigns with 2+ distinct event timestamps in this environment.")
    snapshots = build_temporal_snapshots(ids[0])
    node_counts = [s.graph.num_nodes for s in snapshots]
    assert node_counts == sorted(node_counts)
