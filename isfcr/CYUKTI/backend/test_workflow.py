#!/usr/bin/env python3
"""Test a complete SOAR workflow: generate playbook, check recommendations"""
import requests
import json

BASE_URL = "http://localhost:5002/api/soar"
CAMPAIGN_ID = "CAMP_AC736F50"

print("=" * 60)
print("SOAR WORKFLOW TEST")
print("=" * 60)

# Test 1: Generate playbook
print(f"\n1. Generating playbook for campaign {CAMPAIGN_ID}...")
try:
    response = requests.post(
        f"{BASE_URL}/playbooks/generate",
        json={"campaign_id": CAMPAIGN_ID},
        timeout=30
    )
    print(f"   Status: {response.status_code}")
    if response.ok:
        playbook = response.json()
        playbook_id = playbook.get("playbook_id")
        print(f"   ✅ Generated playbook: {playbook_id}")
        print(f"   Actions: {len(playbook.get('actions', []))}")
    else:
        print(f"   ❌ Error: {response.json()}")
        playbook_id = None
except Exception as e:
    print(f"   ❌ Exception: {e}")
    playbook_id = None

# Test 2: Get recommendations
print(f"\n2. Getting recommendations for campaign {CAMPAIGN_ID}...")
try:
    response = requests.get(
        f"{BASE_URL}/recommendations/{CAMPAIGN_ID}",
        timeout=10
    )
    print(f"   Status: {response.status_code}")
    if response.ok:
        recs = response.json()
        print(f"   ✅ Recommendations: {json.dumps(recs, indent=2)[:200]}...")
    else:
        print(f"   ❌ Error: {response.json()}")
except Exception as e:
    print(f"   ❌ Exception: {e}")

# Test 3: Get response state for campaign
print(f"\n3. Checking response state for {CAMPAIGN_ID}...")
try:
    response = requests.get(
        f"{BASE_URL}/response-state/{CAMPAIGN_ID}",
        timeout=10
    )
    print(f"   Status: {response.status_code}")
    if response.ok:
        state = response.json()
        print(f"   ✅ Current state: {state.get('current_state')}")
        print(f"   Event count: {state.get('event_count')}")
    else:
        print(f"   ❌ Error: {response.json()}")
except Exception as e:
    print(f"   ❌ Exception: {e}")

# Test 4: If we generated a playbook, get its details
if playbook_id:
    print(f"\n4. Retrieving generated playbook {playbook_id}...")
    try:
        response = requests.get(
            f"{BASE_URL}/playbooks/{playbook_id}",
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.ok:
            playbook = response.json()
            print(f"   ✅ Playbook name: {playbook.get('playbook_name')}")
            print(f"   Source campaign: {playbook.get('source_campaign_id')}")
            print(f"   Rationale: {playbook.get('rationale', '')[:100]}...")
        else:
            print(f"   ❌ Error: {response.json()}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")

print("\n" + "=" * 60)
print("WORKFLOW TEST COMPLETE")
print("=" * 60)
