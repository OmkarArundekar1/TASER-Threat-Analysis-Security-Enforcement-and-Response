/* ──────────────────────────────────────────────────────────────
   WatchDog SOC Dashboard — Data Types
   Aligned with dashboard_api.py and the actual Prerana Neo4j schema
   ────────────────────────────────────────────────────────────── */

export interface OverviewMetrics {
  total_events: number;
  active_campaigns: number;
  unique_attackers: number;
  techniques_detected: number;
  learned_transitions: number;
  total_hosts: number;
  timestamp: string;
}

export interface AlertEvent {
  id?: string;
  timestamp: string;
  first_seen?: string;
  attacker_ip: string;
  victim_ip: string;
  technique_id: string;
  technique_name: string;
  stage: string;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  occurrences?: number;
  campaign_id: string;
  mitre_tactic?: string;
  tps?: number;
  recommendations?: RecommendationItem[] | string[];
  predicted_next_attack?: string | null;
  prediction_confidence?: number | null;
  prediction_generated_at?: string | null;
  investigation_payload?: InvestigationPayload;
  raw_wazuh_event?: Record<string, unknown>;
}

export interface InvestigationPayload {
  rule_id?: string | number;
  rule_description?: string;
  rule_level?: string | number;
  rule_groups?: string[];
  rule_firedtimes?: number;
  mitre?: Record<string, unknown>;
  network?: Record<string, unknown>;
  http?: Record<string, unknown>;
  suricata?: Record<string, unknown>;
  agent?: Record<string, unknown>;
  location?: string;
  timestamp?: string;
}

export interface Campaign {
  campaign_id: string;
  campaign_label?: string;
  attacker_ip: string;
  victim_ip: string;
  first_seen: string;
  last_seen: string;
  event_count: number;
  risk_score: number;
  raw_tps?: number;
  risk_level: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  latest_technique: string;
  predicted_technique?: string;
  prediction_confidence?: number;
}

export interface Prediction {
  campaign_id: string;
  current_technique: string;
  predicted_technique: string;
  predicted_name?: string;
  confidence: number;
  risk_level: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  stage: string | null;
  generated_at: string;
  source?: 'LIKELY_NEXT' | 'NEXT_TECHNIQUE';
  empty_reason?: string;
}

export interface Attacker {
  attacker_ip: string;
  campaign_count: number;
  event_count: number;
  risk_score: number;
  raw_tps?: number;
  risk_level?: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  first_seen: string;
  last_seen: string;
  techniques_observed?: string[];
  predicted_technique?: string | null;
}

export interface GraphNode {
  id: string;
  label: string;
  full_label?: string;
  type: 'Attacker' | 'Campaign' | 'AttackEvent' | 'Technique' | 'Host' | 'Stage' | string;
  attack_id?: string;
  campaign_id?: string;
  [key: string]: unknown;
}

export interface GraphLink {
  source: string;
  target: string;
  label: string;
  confidence?: number;
  count?: number;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface AttackChainStep {
  technique_id: string;
  name: string;
  stage: string;
  first_seen?: string;
  detection_count: number;
  tps: number;
}

export interface ChainTransition {
  from: string;
  from_name: string;
  from_stage: string;
  to: string;
  to_name: string;
  to_stage: string;
  count: number;
  confidence: number;
}

export interface RecommendationGroup {
  technique_id: string;
  recommendations: RecommendationItem[] | string[];
  campaign_id?: string;
  technique_name?: string;
  confidence?: number;
}

export interface RecommendationItem {
  recommendation: string;
  priority?: string;
  mitre_mitigation?: string;
  reason?: string;
  predicted_technique?: string;
  traceability?: string;
}

export interface QueryResult {
  records: Record<string, unknown>[];
  columns: string[];
  count: number;
  graph: GraphData;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded';
  neo4j: 'connected' | 'disconnected';
  timestamp: string;
}

/* ── Evidence-aware investigation (backend/investigation) ──────────── */

export interface EvidenceItem {
  evidence_id: string;
  source: 'siem' | 'mitre' | 'cti' | 'graph' | 'campaign_history' | 'attribution' | 'gnn_topology';
  source_id: string;
  timestamp: string;
  type: string;
  content: Record<string, unknown>;
  confidence: number;
  relevance: number;
  provenance: string;
  relationships: string[];
  // evidence_ids of other Evidence already in the store that this item's
  // computation drew on (investigation/loop.py's ActionMeta.depends_on) —
  // empty unless a real, code-verified dependency exists.
  derived_from: string[];
}

/* ── GNN topology (backend/ml/gnn, additive -- see GNN_PRODUCTION_INTEGRATION.md) ── */

export interface GNNModelMetadata {
  model_type: string;
  architecture: string;
  hidden_dim: number;
  embedding_dim: number;
  num_layers: number;
  node_feature_schema: string[];
  edge_feature_schema: string[];
  normalization: string;
  training_seed: number;
  model_version: string;
  training_dataset_description: string;
  artifact_scope: string;
}

export interface GNNStatus {
  gnn_available: boolean;
  gnn_model_version: string | null;
  gnn_metadata: GNNModelMetadata | null;
}

export interface GNNTopologyResponse {
  campaign_id: string;
  gnn_available: boolean;
  gnn_model_version?: string;
  // Each neighbor is a real Evidence item (evidence/schema.py's
  // Evidence.to_dict()) -- same shape the investigation panel renders,
  // reused here rather than duplicated.
  topology_neighbors: EvidenceItem[];
}

/* ── MISP/CTI, system health, audit log (dashboard visibility for
   pipeline state that already existed but had no frontend surface) ── */

export interface MISPStatus {
  misp_url: string;
  credential_configured: boolean;
  authenticated: boolean;
  cached_campaigns: number;
  campaign_event_map: Record<string, number>;
  status_message: string;
}

export interface SystemHealthResponse {
  status: 'healthy' | 'degraded';
  timestamp: string;
  subsystems: Record<string, { status: string; [key: string]: unknown }>;
}

export interface AuditLogEntry {
  timestamp: string | null;
  level: string | null;
  message: string;
}

export interface AuditLogResponse {
  entries: AuditLogEntry[];
  total_lines: number;
  log_path: string;
}

export interface RagSearchResponse {
  query: string | null;
  campaign_id: string | null;
  sources: Record<string, EvidenceItem[] | { error: string } | { gnn_available: boolean; results: EvidenceItem[] }>;
}

export interface ActionScore {
  action: string;
  value: number;
  expected_gain: number;
  reliability: number;
  novelty: number;
  uncertainty_reduction: number;
  redundancy_penalty: number;
  cost: number;
  latency: number;
}

export interface InvestigationStepResult {
  step_index: number;
  action_taken: string;
  action_value: number;
  // human-readable rationale for why this action beat the other candidates
  why_selected: string;
  // every action considered this step, not just the one taken
  candidate_actions: string[];
  action_scores: ActionScore[];
  evidence_added: number;
  // confidence/uncertainty as they stood BEFORE this step's action ran —
  // pairs with the (unprefixed) fields below, which are AFTER
  previous_confidence: number;
  previous_uncertainty: number;
  // model verdict (about the actual investigative conclusion, e.g. severity) —
  // null until the xgboost_prediction action has run this investigation
  model_probabilities: Record<string, number> | null;
  model_confidence: number | null;
  model_uncertainty: number | null;
  // evidence-level (source reliability / breadth — NOT conclusion confidence)
  evidence_reliability: number;
  evidence_coverage: number;
  // investigation-level (what the stopping policy acts on)
  investigation_confidence: number;
  uncertainty: number;
  // every hypothesis with nonzero model probability, ranked highest-first —
  // the full competing-hypothesis picture, not just the argmax label
  candidate_hypotheses: [string, number][];
}

export interface InvestigationResult {
  campaign_id: string;
  steps: InvestigationStepResult[];
  final_confidence: number | null;
  final_model_probabilities: Record<string, number> | null;
  stopping_reason: string;
  total_evidence: number;
  evidence: EvidenceItem[];
}

export interface MitreSearchResult {
  query: string;
  results: EvidenceItem[];
}

export interface ModelMetadata {
  target_column: string;
  classes: string[];
  feature_schema_version: string;
  n_features: number;
  trained_at: string;
}

export interface PredictionContext {
  campaign_id: string;
  attack_id: string;
  event_id: string;
}

export interface SeverityPrediction {
  campaign_id: string;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  // classes ranked by probability, highest first (same data as
  // `probabilities`, pre-sorted for convenience)
  top_k: { label: string; probability: number }[];
  model_metadata: ModelMetadata;
  // traces this prediction back to the investigation state it was computed from
  prediction_context: PredictionContext;
}

/* ── SOAR / Playbook layer (backend/soar, see SOAR_PLAYBOOK_INTEGRATION.md) ── */

export type ExecutionPolicy = 'recommend_only' | 'analyst_approval' | 'automatic';
export type ExecutionStatus =
  | 'pending' | 'pending_approval' | 'rejected' | 'running' | 'success' | 'failed' | 'timeout' | 'cancelled';

export interface PlaybookAction {
  action_id: string;
  action_type: string;
  name: string;
  description: string;
  order: number;
  inputs: Record<string, unknown>;
  expected_output: string;
  destructive: boolean;
  requires_approval: boolean;
  timeout_seconds: number;
  reason: string;
}

export interface Playbook {
  playbook_id: string;
  name: string;
  version: number;
  description: string;
  trigger_conditions: Record<string, unknown>;
  campaign_type: string;
  mitre_techniques: string[];
  severity: string;
  risk: number;
  required_evidence: string[];
  actions: PlaybookAction[];
  execution_policy: ExecutionPolicy;
  shuffle_workflow_id: string | null;
  shuffle_workflow_version: string | null;
  created_at: string;
  updated_at: string;
  status: 'active' | 'deprecated';
  source_campaign_id: string | null;
  adapted_from_playbook_id: string | null;
  has_destructive_action: boolean;
}

export interface PlaybookActionResult {
  action_result_id: string;
  action_id: string;
  status: ExecutionStatus;
  input: Record<string, unknown>;
  output: Record<string, unknown> | string | null;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface PlaybookExecution {
  execution_id: string;
  playbook_id: string;
  playbook_version: number;
  campaign_id: string;
  operation_id: string | null;
  investigation_id: string | null;
  status: ExecutionStatus;
  shuffle_execution_id: string | null;
  started_at: string | null;
  completed_at: string | null;
  action_results: PlaybookActionResult[];
  approved_by: string | null;
  approval_decision_at: string | null;
  rejection_reason: string | null;
  created_at: string;
  audit_events?: SoarAuditEvent[];
}

export interface SoarAuditEvent {
  event_type: string;
  timestamp: string;
  campaign_id: string | null;
  operation_id: string | null;
  investigation_id: string | null;
  playbook_id: string | null;
  execution_id: string | null;
  detail: Record<string, unknown>;
}

export interface HistoricalPlaybookMatch {
  playbook_id: string;
  playbook_name: string;
  source_campaign_id: string;
  technique_similarity: number;
  topology_similarity: number | null;
  attacker_ip_match: boolean;
  victim_ip_match: boolean;
  historical_executions: number;
  historical_successes: number;
  historical_failures: number;
  historical_success_rate: number | null;
  recommendation_reason: string;
}

export interface PlaybookEffectiveness {
  playbook_id: string;
  playbook_name: string;
  executions: number;
  successful_executions: number;
  failed_executions: number;
  success_rate: number | null;
  average_execution_seconds: number | null;
  analyst_approvals: number;
  analyst_rejections: number;
  last_execution_at: string | null;
  failure_reasons: string[];
}

export interface SoarStatus {
  shuffle_webhook_configured: boolean;
  shuffle_api_configured: boolean;
  shuffle_reachable: boolean | null;
  stored_playbooks: number;
  stored_executions: number;
}

export interface SoarRecommendationsResponse {
  campaign_id: string;
  historical_matches: HistoricalPlaybookMatch[];
  candidate_playbook: Playbook;
  adapted_playbook: Playbook | null;
}
