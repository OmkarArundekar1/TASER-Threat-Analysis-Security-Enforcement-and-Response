"""
Phase Z Section 14: race conditions in state transitions. A real,
concurrent (not just sequential-mocked) test proving the duplicate-
request guard cannot be defeated by two simultaneous requests sharing
one correlation_id -- this test would have been flaky/failing against
the pre-fix check-then-act implementation (found and fixed during this
phase's own security audit).
"""

import threading

from active_response.client_agent import Authenticator, ClientResponseAgent, ContainmentRequest
from active_response.containment_actions import ContainmentAction
from active_response.firewall_backend import InMemoryFirewallBackend


def test_concurrent_identical_requests_only_one_executes():
    agent = ClientResponseAgent(InMemoryFirewallBackend(), Authenticator("secret"))
    request = ContainmentRequest("corr-race", ContainmentAction.BLOCK_SOURCE_IP, "10.0.0.5", "dec-1", "secret")

    results = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()  # maximize the chance of a real overlap, not just interleaving
        result = agent.handle(request)
        with results_lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    executed = [r for r in results if r.containment_status == "EXECUTED"]
    rejected = [r for r in results if r.containment_status == "REJECTED"]
    assert len(executed) == 1, f"expected exactly 1 real execution, got {len(executed)}"
    assert len(rejected) == 19


def test_concurrent_requests_with_different_correlation_ids_all_execute_independently():
    agent = ClientResponseAgent(InMemoryFirewallBackend(), Authenticator("secret"))
    results = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(10)

    def worker(i: int):
        barrier.wait()
        req = ContainmentRequest(f"corr-{i}", ContainmentAction.BLOCK_SOURCE_IP, f"10.0.0.{i}", f"dec-{i}", "secret")
        result = agent.handle(req)
        with results_lock:
            results.append(result)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r.containment_status == "EXECUTED" for r in results)
    assert len({r.correlation_id for r in results}) == 10
