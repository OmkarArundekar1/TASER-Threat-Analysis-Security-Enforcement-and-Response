#!/usr/bin/env python3
"""Live API integration test script"""
import requests
import json
import time
import sys

def test_api():
    print("Waiting for backend to be ready...")
    time.sleep(3)

    base_url = "http://localhost:5002/api"
    results = []

    tests = [
        ("GET", f"{base_url}/health", "Health check"),
        ("GET", f"{base_url}/soar/status", "SOAR status"),
        ("GET", f"{base_url}/soar/playbooks", "List playbooks"),
        ("GET", f"{base_url}/soar/executions", "List executions"),
        ("GET", f"{base_url}/soar/effectiveness", "Effectiveness metrics"),
    ]

    for method, url, desc in tests:
        try:
            r = requests.request(method, url, timeout=5)
            status = "✅ PASS" if r.status_code == 200 else f"❌ FAIL ({r.status_code})"
            results.append((desc, status, r.status_code, r.json()))
            print(f"{status} - {desc}")
            if r.status_code == 200:
                data = r.json()
                if 'playbooks' in data:
                    print(f"  → {len(data['playbooks'])} playbooks")
                elif 'executions' in data:
                    print(f"  → {len(data['executions'])} executions")
                elif 'effectiveness' in data:
                    print(f"  → {len(data['effectiveness'])} effectiveness records")
        except requests.exceptions.ConnectionError:
            print(f"❌ FAIL - {desc} - Connection refused")
            results.append((desc, "FAIL", "Connection", None))
            return False
        except Exception as e:
            print(f"❌ FAIL - {desc} - {e}")
            results.append((desc, "FAIL", str(e), None))

    print("\n" + "="*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {sum(1 for _, status, _, _ in results if 'PASS' in status)}")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)
