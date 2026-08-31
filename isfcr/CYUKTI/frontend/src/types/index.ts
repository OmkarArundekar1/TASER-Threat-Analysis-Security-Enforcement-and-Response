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
  source: 'siem' | 'mitre' | 'cti' | 'graph' | 'campaign_history' | 'attribution';
  source_id: string;
  timestamp: string;
  type: string;
  content: Record<string, unknown>;
  confidence: number;
  relevance: number;
  provenance: string;
  relationships: string[];
}

export interface InvestigationStepResult {
  step_index: number;
  action_taken: string;
  action_value: number;
  evidence_added: number;
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

export interface SeverityPrediction {
  campaign_id: string;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
}
