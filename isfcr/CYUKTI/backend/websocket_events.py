"""
websocket_events.py
===================
Real-time event broadcasting for CYUKTI dashboard using Flask-SocketIO.

This module provides WebSocket event emission after successful Neo4j persistence.
WebSocket failures never block database writes or alert ingestion.

Broadcasting is idempotent - duplicate emissions are suppressed using event IDs.
"""

import logging
import json
from datetime import datetime
from threading import Lock
from collections import OrderedDict
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Global SocketIO instance - initialized by dashboard_api.py
socketio = None

# Recently emitted event IDs for deduplication (thread-safe)
_emitted_events = OrderedDict()
_emitted_events_lock = Lock()
MAX_EMITTED_CACHE = 1000

def initialize_socketio(app, cors_allowed_origins="*"):
    """
    Initialize Flask-SocketIO instance with the Flask app.
    Called once at startup by dashboard_api.py.
    
    Args:
        app: Flask application instance
        cors_allowed_origins: CORS origins for WebSocket connections
    
    Returns:
        SocketIO instance
    """
    global socketio
    
    if socketio is not None:
        logger.warning("SocketIO already initialized")
        return socketio
    
    try:
        from flask_socketio import SocketIO
        from flask import request as flask_request
        
        socketio = SocketIO(
            app,
            cors_allowed_origins=cors_allowed_origins,
            logger=False,  # Use our own logging
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25
        )
        
        # Register connection handlers
        @socketio.on('connect')
        def handle_connect():
            try:
                sid = flask_request.sid if hasattr(flask_request, 'sid') else 'unknown'
                logger.info(f"Client connected: {sid}")
            except:
                logger.info("Client connected")
        
        @socketio.on('disconnect')
        def handle_disconnect():
            try:
                sid = flask_request.sid if hasattr(flask_request, 'sid') else 'unknown'
                logger.info(f"Client disconnected: {sid}")
            except:
                logger.info("Client disconnected")
        
        @socketio.on('subscribe_events')
        def handle_subscribe(data):
            client_id = data.get('client', 'unknown')
            logger.info(f"Client subscribed to events: {client_id}")
            # Acknowledge subscription
            return {'status': 'subscribed', 'timestamp': datetime.utcnow().isoformat()}
        
        logger.info("SocketIO initialized successfully")
        return socketio
        
    except Exception as e:
        logger.exception(f"Failed to initialize SocketIO: {e}")
        return None


def emit_attack_event(
    event_id: str,
    campaign_id: Optional[str],
    attacker_ip: str,
    victim_ip: str,
    technique_id: Optional[str],
    technique_name: Optional[str],
    rule_id: str,
    severity: str,
    stage: Optional[str],
    timestamp: str,
    agent_name: Optional[str] = None,
    confidence: Optional[float] = None,
    operation_id: Optional[str] = None,
    risk_score: Optional[float] = None,
    **kwargs
):
    """
    Emit a new attack event to all connected dashboard clients.
    
    This function is called AFTER successful Neo4j persistence.
    Emission failures are logged but never raise exceptions.
    
    Args:
        event_id: Unique Neo4j AttackEvent ID
        campaign_id: Associated campaign ID (can be None for unattributed events)
        attacker_ip: Source IP address
        victim_ip: Destination IP address
        technique_id: MITRE ATT&CK technique ID (can be None for UNKNOWN)
        technique_name: Human-readable technique name
        rule_id: Wazuh rule ID
        severity: Event severity (CRITICAL, HIGH, MEDIUM, LOW)
        stage: Attack stage (e.g., Initial Access, Discovery)
        timestamp: ISO-8601 timestamp
        agent_name: Wazuh agent name
        confidence: Detection confidence (0-100)
        operation_id: Associated operation ID
        risk_score: Dynamic risk score
        **kwargs: Additional metadata
    
    Returns:
        bool: True if emitted successfully, False otherwise
    """
    global socketio
    
    if socketio is None:
        logger.debug("SocketIO not initialized, skipping event emission")
        return False
    
    # Check if already emitted (idempotency)
    with _emitted_events_lock:
        if event_id in _emitted_events:
            logger.debug(f"Event {event_id} already emitted, skipping duplicate")
            return True
        
        # Add to emitted cache
        _emitted_events[event_id] = datetime.utcnow()
        
        # Maintain cache size
        if len(_emitted_events) > MAX_EMITTED_CACHE:
            _emitted_events.popitem(last=False)
    
    try:
        # Construct event payload matching frontend contract
        payload = {
            "id": event_id,
            "event_type": "attack_detected",
            "timestamp": timestamp,
            "attacker_ip": attacker_ip,
            "victim_ip": victim_ip,
            "technique_id": technique_id or "UNKNOWN",
            "technique_name": technique_name or "Unknown Technique",
            "rule_id": rule_id,
            "severity": severity,
            "stage": stage or "Unknown",
            "campaign_id": campaign_id,
            "operation_id": operation_id,
            "agent_name": agent_name,
            "confidence": confidence,
            "risk_score": risk_score,
            "response_status": "NOT_REQUESTED",
        }
        
        # Add any additional metadata
        if kwargs:
            payload["metadata"] = kwargs
        
        # Broadcast to all connected clients
        socketio.emit('new_events', {'events': [payload]}, namespace='/')
        
        logger.debug(f"Emitted attack event {event_id} to WebSocket clients")
        return True
        
    except Exception as e:
        logger.error(f"Failed to emit event {event_id}: {e}", exc_info=True)
        return False


def emit_response_status_update(
    event_id: str,
    attacker_ip: str,
    response_status: str,
    response_action: Optional[str] = None,
    timestamp: Optional[str] = None,
    **kwargs
):
    """
    Emit a response status update to connected clients.
    
    Called when containment state changes (e.g., BLOCKED, FAILED, UNBLOCKED).
    
    Args:
        event_id: Attack event ID
        attacker_ip: Affected IP address
        response_status: New status (PENDING, BLOCKED, FAILED, UNBLOCKED, etc.)
        response_action: Action taken (block_ip, unblock_ip, etc.)
        timestamp: ISO-8601 timestamp of status change
        **kwargs: Additional response metadata
    
    Returns:
        bool: True if emitted successfully, False otherwise
    """
    global socketio
    
    if socketio is None:
        logger.debug("SocketIO not initialized, skipping status update")
        return False
    
    try:
        payload = {
            "event_id": event_id,
            "event_type": "response_status_update",
            "attacker_ip": attacker_ip,
            "response_status": response_status,
            "response_action": response_action,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
        }
        
        if kwargs:
            payload.update(kwargs)
        
        socketio.emit('response_status_update', payload, namespace='/')
        
        logger.info(f"Emitted response status update for {event_id}: {response_status}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to emit response status for {event_id}: {e}", exc_info=True)
        return False


def get_connection_count() -> int:
    """Get the number of currently connected WebSocket clients."""
    global socketio
    
    if socketio is None:
        return 0
    
    try:
        # Flask-SocketIO doesn't expose connection count directly
        # Return -1 to indicate "available but count unknown"
        return -1
    except:
        return 0
