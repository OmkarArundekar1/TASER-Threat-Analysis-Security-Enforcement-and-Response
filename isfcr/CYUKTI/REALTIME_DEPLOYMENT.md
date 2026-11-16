# CYUKTI Real-Time SOC Deployment Guide

**Last Updated**: 2024-11-15  
**Feature**: End-to-End Real-Time Detection, Investigation & Response

---

## Overview

This guide covers starting CYUKTI with real-time capabilities:

- **Wazuh → Backend**: Live alert ingestion via `wazuh_listener.py`
- **Backend → Neo4j**: Immediate persistence with campaign/operation correlation
- **Backend → Frontend**: WebSocket event streaming (< 1 second latency)
- **Dashboard → Backend**: Manual/policy-controlled attacker containment
- **Backend → Firewall**: Verified IP blocking with audit trail

---

## Prerequisites

### Required Services

1. **Neo4j** (Bolt port 7687)
   ```bash
   docker start neo4j-soc
   # OR
   systemctl start neo4j
   ```

2. **Wazuh Manager** (with agents reporting)
   ```bash
   sudo systemctl status wazuh-manager
   # Should show: active (running)
   ```

3. **Python Environment**
   ```bash
   source ~/capstone/soc_env/bin/activate
   ```

### Python Dependencies

All required packages are already installed in `soc_env`:
- Flask 3.1.3
- Flask-SocketIO 5.5.1
- python-socketio 5.16.3
- neo4j driver
- All existing CYUKTI dependencies

---

## Startup Sequence

### 1. Start Neo4j

```bash
# If using Docker
docker start neo4j-soc

# Verify connection
docker exec -it neo4j-soc cypher-shell -u neo4j -p neo4j_password

# Test query
MATCH (n) RETURN count(n) as node_count;
```

### 2. Start CYUKTI Backend (with WebSocket support)

```bash
cd ~/capstone/isfcr/CYUKTI/backend
source ~/capstone/soc_env/bin/activate
python dashboard_api.py
```

**Expected output:**
```
INFO:websocket_events:SocketIO initialized successfully
INFO:werkzeug:Starting CYUKTI dashboard with WebSocket support on port 5002
INFO:werkzeug: * Running on http://0.0.0.0:5002
```

**Key indicators:**
- ✓ "SocketIO initialized successfully"
- ✓ "WebSocket support on port 5002"
- ✗ If you see only "Running on http://0.0.0.0:5002" without SocketIO messages, check `websocket_events.py` import

### 3. Start Wazuh Listener (Real-Time Ingestion)

**In a new terminal:**

```bash
cd ~/capstone/isfcr/CYUKTI/backend
source ~/capstone/soc_env/bin/activate
python listener/wazuh_listener.py
```

**Expected output:**
```
INFO | Wazuh listener started
INFO | Monitoring: /var/ossec/logs/alerts/alerts.json
INFO | Backend ingestion: process_alert()
```

**Verify only ONE listener is running:**
```bash
ps aux | grep wazuh_listener | grep -v grep
# Should show exactly 1 process
```

### 4. Start Frontend Dashboard

**In a new terminal:**

```bash
cd ~/capstone/isfcr/CYUKTI/frontend
npm run dev
```

**Expected output:**
```
VITE v5.x.x ready in XXX ms
➜ Local:   http://localhost:5173/
➜ Network: use --host to expose
```

### 5. Verify WebSocket Connection

Open browser to `http://localhost:5173`

**Check browser console (F12):**
```
WebSocket connected to CYUKTI backend
Subscribed to events: {status: 'subscribed', timestamp: '...'}
```

**Dashboard indicators:**
- Top-right: Green WiFi icon = WebSocket connected
- Amber WiFi icon with diagonal line = Polling fallback
- Live SOC Feed shows green pulsing dot

---

## Testing the Pipeline

### Test 1: Backend Health

```bash
curl http://localhost:5002/api/health
```

**Expected:**
```json
{
  "status": "healthy",
  "neo4j": "connected",
  "timestamp": "2024-11-15T..."
}
```

### Test 2: Run Integration Tests

```bash
cd ~/capstone/isfcr/CYUKTI/backend
source ~/capstone/soc_env/bin/activate
python test_realtime_pipeline.py
```

This verifies:
- ✓ Backend responding
- ✓ Events API working
- ✓ WebSocket endpoint accessible
- ✓ Containment API available
- ✓ Protected IP safety checks

### Test 3: Generate Test Alert

**Simulate SSH brute force attack:**

```bash
# Create test alert JSON
cat > /tmp/test_alert.json <<'EOF'
{"timestamp":"2024-11-15T12:00:00Z","rule":{"id":"60204","description":"Multiple Failed SSH Login Attempts","level":10,"mitre":{"technique":["Brute Force"],"id":["T1110"]}},"agent":{"name":"test-agent","ip":"192.168.56.106"},"data":{"srcip":"203.0.113.50","dstip":"192.168.56.105"},"full_log":"sshd: Failed password for invalid user admin from 203.0.113.50"}
EOF

# Append to Wazuh alerts (requires sudo)
sudo bash -c 'cat /tmp/test_alert.json >> /var/ossec/logs/alerts/alerts.json'
```

**Expected flow (< 1 second total):**
1. Listener detects new line in `alerts.json`
2. Backend processes via `realtime_socgraph.process_alert()`
3. Neo4j persists AttackEvent
4. WebSocket emits `new_events` to all clients
5. Dashboard displays event WITHOUT page refresh

**Verify:**
- Backend console: `[OPERATION] Created OP_...` or `[OPERATION] Attached campaign...`
- Browser console: `Received 1 new event(s) via WebSocket`
- Dashboard: New row appears at top of Live SOC Feed

### Test 4: Manual Containment

**From browser console (F12):**

```javascript
// Get the latest event ID
const events = await fetch('http://localhost:5002/api/events?page=1&page_size=1')
  .then(r => r.json());
const eventId = events.events[0].id;

// Request containment
fetch('http://localhost:5002/api/containment/block', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    event_id: eventId,
    operator: 'analyst-test',
    reason: 'manual',
    ttl_seconds: 300
  })
}).then(r => r.json()).then(console.log);
```

**Expected response:**
```json
{
  "success": true,
  "event_id": "...",
  "attacker_ip": "203.0.113.50",
  "status": "BLOCKED",
  "verification": "confirmed",
  "timestamp": "..."
}
```

**Backend console should show:**
- `Containment request for 203.0.113.50`
- `Firewall rule applied: ...`
- `Verification: IP is blocked`

---

## Troubleshooting

### WebSocket Not Connecting

**Symptom:** Browser shows amber WiFi icon, console shows connection errors

**Checks:**
1. Backend running with SocketIO?
   ```bash
   curl http://localhost:5002/socket.io/?EIO=4&transport=polling
   # Should return: 0{"sid":"..."}
   ```

2. Check backend startup logs:
   ```
   # Good:
   INFO:websocket_events:SocketIO initialized successfully
   
   # Bad:
   WARNING:websocket_events:SocketIO already initialized
   # OR no SocketIO message at all
   ```

3. Restart backend:
   ```bash
   # Kill existing process
   pkill -f dashboard_api.py
   
   # Start fresh
   python dashboard_api.py
   ```

### Events Not Appearing

**Symptom:** Wazuh alerts generated but dashboard doesn't update

**Checks:**
1. Listener running?
   ```bash
   ps aux | grep wazuh_listener
   ```

2. Listener processing alerts?
   ```bash
   tail -f ~/capstone/isfcr/CYUKTI/backend/listener/logs/prerana_listener.log
   # Should show new lines when alerts arrive
   ```

3. Neo4j accepting writes?
   ```cypher
   MATCH (ae:AttackEvent)
   RETURN ae.event_id, ae.first_seen
   ORDER BY ae.first_seen DESC
   LIMIT 5
   ```

4. Backend emitting WebSocket events?
   ```bash
   # Check backend console for:
   DEBUG:websocket_events:Emitted attack event ... to WebSocket clients
   ```

### Containment Fails

**Symptom:** Block request returns `"status": "FAILED"`

**Checks:**
1. Is the IP protected?
   ```python
   # In backend console:
   from containment_api import is_ip_protected
   is_ip_protected("127.0.0.1")  # Should be True
   ```

2. Check firewall backend availability:
   ```bash
   # Does the environment support iptables/firewalld?
   sudo iptables -L | head
   # OR
   sudo firewall-cmd --state
   ```

3. Check active_response configuration:
   ```python
   # In Python:
   from active_response.firewall_backend import FirewallBackend
   fw = FirewallBackend()
   # Should not raise ImportError
   ```

4. Review containment audit trail:
   ```bash
   curl http://localhost:5002/api/soar/response-audit/CAMPAIGN_ID
   ```

### Duplicate Listeners

**Symptom:** Events processed twice, duplicate Neo4j nodes

**Fix:**
```bash
# Kill all listeners
pkill -f wazuh_listener

# Verify none running
ps aux | grep wazuh_listener | grep -v grep
# Should return empty

# Start exactly one
python listener/wazuh_listener.py
```

### Frontend Polling Instead of WebSocket

**Symptom:** Events eventually appear but with 30-second delay

**This is the fallback behavior.** WebSocket failures don't break the dashboard.

**To restore WebSocket:**
1. Check backend logs for SocketIO errors
2. Verify port 5002 is not firewalled
3. Check browser console for connection errors
4. Try different browser (WebSocket support)

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                      WAZUH MANAGER                          │
│              /var/ossec/logs/alerts/alerts.json             │
└────────────────────────┬────────────────────────────────────┘
                         │ tail -f
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              listener/wazuh_listener.py                     │
│  - Deduplication buffer                                     │
│  - Source identity tracking                                 │
│  - Calls: realtime_socgraph.process_alert()                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              realtime_socgraph.py                           │
│  - MITRE resolution (resolver + curated + reviewed)         │
│  - Campaign correlation                                     │
│  - Operation grouping                                       │
│  - Prediction engine                                        │
│  - Risk scoring                                             │
│  - CTI confidence                                           │
│  - Threat attribution                                       │
│  - Neo4j persistence (AttackEvent, Campaign, Operation)     │
│  - WebSocket emission (websocket_events.emit_attack_event)  │
└───────────┬─────────────────────────────────────────────────┘
            │                                │
            ▼                                ▼
┌───────────────────────┐      ┌────────────────────────────┐
│   Neo4j Database      │      │ Flask-SocketIO Server      │
│   - AttackEvent       │      │ - new_events channel       │
│   - Campaign          │      │ - response_status_update   │
│   - Operation         │      └──────────┬─────────────────┘
│   - Technique         │                 │ WebSocket
│   - Attacker          │                 ▼
│   - Host              │      ┌────────────────────────────┐
└───────────────────────┘      │    Frontend Dashboard      │
                               │  - useWebSocket() hook     │
                               │  - Live SOC Feed           │
                               │  - Real-time updates       │
                               │  - Manual containment UI   │
                               └──────────┬─────────────────┘
                                          │ HTTP POST
                                          ▼
                               ┌────────────────────────────┐
                               │  containment_api.py        │
                               │  - Validate event ID       │
                               │  - Extract attacker IP     │
                               │  - Check protected IPs     │
                               │  - Execute via            │
                               │    active_response         │
                               │  - Verify firewall state   │
                               │  - Persist audit trail     │
                               │  - Emit status update      │
                               └────────────────────────────┘
```

---

## Measured Latencies

**Expected End-to-End Performance:**

| Stage | Target Latency |
|-------|---------------|
| Wazuh alert → Listener detection | < 100ms (tail -f) |
| Listener → process_alert() call | < 10ms |
| process_alert() → Neo4j write | 50-200ms (campaign correlation) |
| Neo4j write → WebSocket emit | < 5ms |
| WebSocket emit → Browser render | 10-50ms (network + React) |
| **TOTAL: Alert → Dashboard** | **< 500ms typical** |
| Manual block request → Firewall | 100-500ms |
| Firewall → Verification | 50-100ms |
| **TOTAL: Block Click → Verified** | **< 1 second** |

**Polling fallback latency:** 30 seconds (acceptable degradation)

---

## Safety Controls

### Protected Infrastructure

**These IPs CANNOT be blocked:**
- 127.0.0.1 (localhost)
- Wazuh manager IP
- Neo4j host IP
- CYUKTI backend host IP
- SOC analyst workstation IPs

**Configuration:**
Edit `isfcr/CYUKTI/backend/containment_api.py`:
```python
PROTECTED_IPS = {
    "127.0.0.1",
    "192.168.56.1",  # Add your infrastructure IPs
    "10.0.2.15",
}
```

### Automatic Response Policy

**Default: DISABLED**

To enable policy-controlled automatic blocking:

Edit `isfcr/CYUKTI/backend/containment_api.py`:
```python
AUTO_BLOCK_ENABLED = True
AUTO_BLOCK_MIN_CONFIDENCE = 95.0  # Require very high confidence
AUTO_BLOCK_MIN_SEVERITY = "HIGH"   # Only HIGH/CRITICAL
AUTO_BLOCK_TTL_SECONDS = 3600      # 1 hour expiry
```

**⚠️ WARNING:** Only enable in isolated lab environments with tested configurations.

### Rollback/Unblock

```bash
curl -X POST http://localhost:5002/api/containment/unblock \
  -H "Content-Type: application/json" \
  -d '{"attacker_ip": "203.0.113.50", "operator": "analyst"}'
```

---

## Maintenance

### View Active Blocks

```bash
# If using iptables
sudo iptables -L -n -v | grep 203.0.113

# If using firewalld
sudo firewall-cmd --list-all
```

### Check Audit Trail

```bash
# For a specific campaign
curl http://localhost:5002/api/soar/response-audit/CAMP_XXXXXXXX | jq

# For an event
curl http://localhost:5002/api/containment/status/EVENT_ID | jq
```

### Clear Test Data

```cypher
// Delete test events (Neo4j)
MATCH (ae:AttackEvent)
WHERE ae.event_id CONTAINS 'test'
DETACH DELETE ae;

// Delete test campaigns
MATCH (c:Campaign)
WHERE c.campaign_id CONTAINS 'TEST'
DETACH DELETE c;
```

---

## Production Deployment Notes

**This implementation is designed for LAB environments.**

For production deployment, consider:

1. **Authentication**: Add JWT/OAuth2 to containment endpoints
2. **Authorization**: Role-based access control for block/unblock
3. **Rate Limiting**: Prevent containment request flooding
4. **HTTPS**: TLS for WebSocket and API connections
5. **Distributed Deployment**: Multiple backend instances with load balancer
6. **Firewall Integration**: Production-grade firewall API (not iptables)
7. **SIEM Integration**: Forward audit trail to enterprise SIEM
8. **Monitoring**: Prometheus/Grafana for latency tracking
9. **Backup**: Automated Neo4j backups with event retention policy

---

## Support

**Issues:**
- Check backend console for Python exceptions
- Check browser console for WebSocket errors
- Review `listener/logs/prerana_listener.log`
- Run `test_realtime_pipeline.py` for diagnostics

**Git Status:**
This feature adds/modifies:
- `backend/websocket_events.py` (new)
- `backend/containment_api.py` (new)
- `backend/dashboard_api.py` (modified - SocketIO init)
- `backend/realtime_socgraph.py` (modified - WebSocket emission)
- `frontend/src/hooks/useWebSocket.ts` (modified - proper connection)
- `frontend/src/App.tsx` (modified - hook activation)
- `frontend/src/components/LiveEventsFeed.tsx` (modified - connection indicator)

**Audit Phase Constraints:**
- No modifications to:
  - Campaign/Operation correlation logic
  - ATT&CK resolution tiers
  - MITRE mappings
  - Neo4j schema
  - Existing test suites

All changes are additive real-time features, not fixes to audited systems.
