"""
soar/api.py
=============
Flask Blueprint for the SOAR/Playbook dashboard API (Phase 14). Kept as
its own blueprint rather than appended directly into dashboard_api.py
(which is already large) -- registered once via
`app.register_blueprint(soar_bp)` in dashboard_api.py. Every route
below follows dashboard_api.py's own conventions (JSON errors, the same
Neo4j error handlers apply since they're registered on the same Flask
app instance).
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from soar.adapter import PlaybookAdaptation
from soar.execution_service import PlaybookExecutionService, PolicyError
from soar.generator import PlaybookGenerator
from soar.matcher import PlaybookMatcher
from soar.memory import memory_store
from soar.shuffle_client import build_default_client

logger = logging.getLogger(__name__)

soar_bp = Blueprint("soar", __name__, url_prefix="/api/soar")

_shuffle_client = build_default_client()
_execution_service = PlaybookExecutionService(memory_store, _shuffle_client)
_matcher = PlaybookMatcher(memory_store)
_generator = PlaybookGenerator()
_adapter = PlaybookAdaptation()


def _load_context_or_error(campaign_id: str):
    """Local wrapper so this module's routes don't import dashboard_api
    at module scope (dashboard_api imports this blueprint to register
    it -- importing it back here at module load time would be
    circular). The lazy, function-body import below only runs once
    Flask is already fully initialized."""
    from dashboard_api import _try_load_campaign_context
    return _try_load_campaign_context(campaign_id)


@soar_bp.route("/response-audit/<correlation_id>")
def response_audit_trail(correlation_id: str):
    """Active-containment dashboard data source (backend/active_response/,
    review/phaseX_active_response_report.md): the full auditable
    lifecycle for one correlation_id -- policy decision, containment
    request/execution, verification outcome, rollback/expiry -- exactly
    as recorded by active_response.audit.log_response_event(), never
    reconstructed or inferred. An empty list is a real, honest answer
    (no response action has ever been logged under this ID), not an error."""
    from active_response.audit import audit_trail_for_correlation
    events = audit_trail_for_correlation(correlation_id)
    return jsonify({"correlation_id": correlation_id, "event_count": len(events), "events": events})


@soar_bp.route("/status")
def soar_status():
    import config
    return jsonify({
        "shuffle_webhook_configured": bool(config.SHUFFLE_WEBHOOK),
        "shuffle_api_configured": bool(config.SHUFFLE_BASE_URL and config.SHUFFLE_API_KEY),
        "shuffle_reachable": _shuffle_client.health_check() if config.SHUFFLE_BASE_URL else None,
        "stored_playbooks": len(memory_store.list_playbooks()),
        "stored_executions": len(memory_store.list_executions(limit=100000)),
    })


@soar_bp.route("/playbooks/generate", methods=["POST"])
def generate_playbook():
    body = request.get_json(silent=True) or {}
    campaign_id = body.get("campaign_id")
    if not campaign_id:
        return jsonify({"error": "Missing campaign_id"}), 400

    context, error_response = _load_context_or_error(campaign_id)
    if error_response:
        return error_response
    if context is None:
        return jsonify({"error": f"Campaign {campaign_id} not found"}), 404

    playbook = _generator.generate(context)
    memory_store.save_playbook(playbook)
    memory_store.log_audit_event(
        "PLAYBOOK_GENERATED", campaign_id=campaign_id, playbook_id=playbook.playbook_id,
    )
    return jsonify(playbook.to_dict())


@soar_bp.route("/playbooks", methods=["GET"])
def list_playbooks():
    return jsonify({"playbooks": [p.to_dict() for p in memory_store.list_playbooks()]})


@soar_bp.route("/playbooks/<playbook_id>", methods=["GET"])
def get_playbook(playbook_id):
    playbook = memory_store.get_playbook(playbook_id)
    if playbook is None:
        return jsonify({"error": f"Playbook {playbook_id} not found"}), 404
    return jsonify(playbook.to_dict())


@soar_bp.route("/playbooks/adapt", methods=["POST"])
def adapt_playbook():
    body = request.get_json(silent=True) or {}
    source_playbook_id = body.get("source_playbook_id")
    campaign_id = body.get("campaign_id")
    if not source_playbook_id or not campaign_id:
        return jsonify({"error": "Missing source_playbook_id or campaign_id"}), 400

    source_playbook = memory_store.get_playbook(source_playbook_id)
    if source_playbook is None:
        return jsonify({"error": f"Playbook {source_playbook_id} not found"}), 404

    context, error_response = _load_context_or_error(campaign_id)
    if error_response:
        return error_response
    if context is None:
        return jsonify({"error": f"Campaign {campaign_id} not found"}), 404

    adapted = _adapter.adapt(source_playbook, context, operation_id=body.get("operation_id"))
    memory_store.save_playbook(adapted)
    memory_store.log_audit_event(
        "PLAYBOOK_ADAPTED", campaign_id=campaign_id, playbook_id=adapted.playbook_id,
        detail={"adapted_from": source_playbook_id},
    )
    return jsonify(adapted.to_dict())


@soar_bp.route("/playbooks/<playbook_id>/execute", methods=["POST"])
def execute_playbook(playbook_id):
    body = request.get_json(silent=True) or {}
    playbook = memory_store.get_playbook(playbook_id)
    if playbook is None:
        return jsonify({"error": f"Playbook {playbook_id} not found"}), 404

    campaign_id = body.get("campaign_id") or playbook.source_campaign_id
    if not campaign_id:
        return jsonify({"error": "No campaign_id available for this playbook"}), 400

    try:
        execution = _execution_service.request_execution(
            playbook, campaign_id=campaign_id,
            operation_id=body.get("operation_id"), investigation_id=body.get("investigation_id"),
        )
    except PolicyError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(execution.to_dict())


@soar_bp.route("/executions", methods=["GET"])
def list_executions():
    limit = int(request.args.get("limit", 100))
    return jsonify({"executions": [e.to_dict() for e in memory_store.list_executions(limit=limit)]})


@soar_bp.route("/executions/<execution_id>", methods=["GET"])
def get_execution(execution_id):
    execution = memory_store.get_execution(execution_id)
    if execution is None:
        return jsonify({"error": f"Execution {execution_id} not found"}), 404
    payload = execution.to_dict()
    payload["audit_events"] = memory_store.list_audit_events(execution_id=execution_id)
    return jsonify(payload)


@soar_bp.route("/executions/<execution_id>/poll", methods=["POST"])
def poll_execution(execution_id):
    try:
        execution = _execution_service.poll_status(execution_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify(execution.to_dict())


@soar_bp.route("/executions/<execution_id>/approve", methods=["POST"])
def approve_execution(execution_id):
    body = request.get_json(silent=True) or {}
    try:
        execution = _execution_service.approve(execution_id, approved_by=body.get("approved_by", "analyst"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(execution.to_dict())


@soar_bp.route("/executions/<execution_id>/reject", methods=["POST"])
def reject_execution(execution_id):
    body = request.get_json(silent=True) or {}
    try:
        execution = _execution_service.reject(
            execution_id, reason=body.get("reason", "Rejected by analyst"), rejected_by=body.get("rejected_by"),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(execution.to_dict())


@soar_bp.route("/recommendations/<campaign_id>", methods=["GET"])
def recommendations(campaign_id):
    context, error_response = _load_context_or_error(campaign_id)
    if error_response:
        return error_response
    if context is None:
        return jsonify({"error": f"Campaign {campaign_id} not found"}), 404

    from neo4j_client import driver
    try:
        with driver.session() as session:
            matches = _matcher.find_matches(context, neo4j_session=session)
    except Exception:
        logger.exception("soar recommendations: matcher failed for %s", campaign_id)
        matches = []

    candidate_playbook = _generator.generate(context)

    adapted_playbook = None
    if matches:
        top_match_playbook = memory_store.get_playbook(matches[0].playbook_id)
        if top_match_playbook is not None:
            adapted_playbook = _adapter.adapt(top_match_playbook, context).to_dict()

    return jsonify({
        "campaign_id": campaign_id,
        "historical_matches": [m.to_dict() for m in matches],
        "candidate_playbook": candidate_playbook.to_dict(),
        "adapted_playbook": adapted_playbook,
    })


@soar_bp.route("/effectiveness", methods=["GET"])
def effectiveness():
    return jsonify({"effectiveness": [e.to_dict() for e in memory_store.effectiveness_all()]})
