/**
 * Realistic backend response fixtures for frontend tests.
 *
 * Every field here traces to a real backend response shape verified in
 * prior CYUKTI sessions (dashboard_api.py, investigation/loop.py's
 * InvestigationRecord.to_dict(), ml/train_xgboost.py's predict()) --
 * see ../../DASHBOARD_API.md, ../../../backend/ML_NBE_INTEGRATION.md.
 * Nothing here is invented: a field is only present if the real backend
 * actually returns it.
 */

import type {
  InvestigationResult, SeverityPrediction, Campaign, OverviewMetrics,
  GraphData, Prediction, HealthStatus,
} from '../types';

/** investigation/loop.py's InvestigationRecord.to_dict() -- two real
 * steps (a fact-gathering action, then the XGBoost prediction action),
 * matching the real shape including the fields this session found the
 * UI was not yet rendering (why_selected, candidate_hypotheses, etc). */
export const investigationResultFixture: InvestigationResult = {
  campaign_id: 'CAMP_7331E223',
  steps: [
    {
      step_index: 1,
      action_taken: 'campaign_history',
      action_value: 1.49,
      why_selected: 'highest-scoring of 8 candidate action(s), value=1.49 vs. next-best mitre_knowledge=1.4',
      candidate_actions: ['campaign_history', 'mitre_knowledge', 'graph_structure', 'attribution_match'],
      action_scores: [
        {
          action: 'campaign_history', value: 1.49, expected_gain: 0.9, reliability: 0.9,
          novelty: 1.0, uncertainty_reduction: 0.0, redundancy_penalty: 0.0, cost: 0.3, latency: 0.3,
        },
      ],
      evidence_added: 2,
      previous_confidence: 0.0,
      previous_uncertainty: 1.0,
      model_probabilities: null,
      model_confidence: null,
      model_uncertainty: null,
      evidence_reliability: 0.9,
      evidence_coverage: 0.17,
      investigation_confidence: 0.15,
      uncertainty: 0.85,
      candidate_hypotheses: [],
    },
    {
      step_index: 2,
      action_taken: 'xgboost_prediction',
      action_value: 1.71,
      why_selected: 'highest-scoring of 7 candidate action(s), value=1.71 vs. next-best attribution_match=1.02',
      candidate_actions: ['xgboost_prediction', 'attribution_match', 'graph_structure'],
      action_scores: [
        {
          action: 'xgboost_prediction', value: 1.71, expected_gain: 0.0, reliability: 0.0,
          novelty: 0.0, uncertainty_reduction: 0.85, redundancy_penalty: 0.0, cost: 0.0, latency: 0.0,
        },
      ],
      evidence_added: 0,
      previous_confidence: 0.15,
      previous_uncertainty: 0.85,
      model_probabilities: { Critical: 0.62, High: 0.21, Medium: 0.12, Low: 0.05 },
      model_confidence: 0.62,
      model_uncertainty: 0.41,
      evidence_reliability: 0.9,
      evidence_coverage: 0.17,
      investigation_confidence: 0.56,
      uncertainty: 0.41,
      candidate_hypotheses: [['Critical', 0.62], ['High', 0.21], ['Medium', 0.12], ['Low', 0.05]],
    },
  ],
  final_confidence: 0.56,
  final_model_probabilities: { Critical: 0.62, High: 0.21, Medium: 0.12, Low: 0.05 },
  stopping_reason: 'max_steps reached',
  total_evidence: 2,
  evidence: [
    {
      evidence_id: 'campaign_history:CAMP_PAST_1:historical_match',
      source: 'campaign_history',
      source_id: 'CAMP_PAST_1',
      timestamp: '2026-01-01T00:00:00+00:00',
      type: 'historical_match',
      content: { campaign_id: 'CAMP_PAST_1', attacker: '203.0.113.7', techniques: ['T1110', 'T1078'] },
      confidence: 1.0,
      relevance: 0.8,
      provenance: 'attribution_context (historical campaign record)',
      relationships: ['CAMP_PAST_1'],
      derived_from: [],
    },
    {
      evidence_id: 'attribution:CAMP_PAST_1:attribution_match',
      source: 'attribution',
      source_id: 'CAMP_PAST_1',
      timestamp: '2026-01-15T12:00:00+00:00',
      type: 'attribution_match',
      content: {
        candidate_campaign_id: 'CAMP_PAST_1', matched_techniques: ['T1110', 'T1078'],
        technique_similarity: 100.0, chain_similarity: 100.0, coverage: 100.0, precision: 100.0,
        notes: ['Coverage : 100.00%', 'Precision: 100.00%', 'Matched 2 ATT&CK techniques'], rank: 0,
      },
      confidence: 1.0,
      relevance: 1.0,
      provenance: 'threat_attribution_engine (historical campaign similarity)',
      relationships: ['CAMP_PAST_1'],
      // populated because CAMPAIGN_HISTORY ran first this investigation --
      // investigation/loop.py's ActionMeta.depends_on -- the real
      // provenance-tracing mechanism this fixture must not fabricate a
      // second version of.
      derived_from: ['campaign_history:CAMP_PAST_1:historical_match'],
    },
  ],
};

export const noModelInvestigationResultFixture: InvestigationResult = {
  ...investigationResultFixture,
  final_model_probabilities: null,
  final_confidence: 0.15,
  steps: [investigationResultFixture.steps[0]],
  stopping_reason: 'no positive-value action remains',
  total_evidence: 2,
};

/** ml/train_xgboost.py's predict() + ml/runtime_predictor.py's
 * prediction_context -- the real fields this session found
 * SeverityPrediction was missing before the type fix. */
export const severityPredictionFixture: SeverityPrediction = {
  campaign_id: 'CAMP_7331E223',
  label: 'Critical',
  confidence: 0.62,
  probabilities: { Critical: 0.62, High: 0.21, Medium: 0.12, Low: 0.05 },
  top_k: [
    { label: 'Critical', probability: 0.62 },
    { label: 'High', probability: 0.21 },
    { label: 'Medium', probability: 0.12 },
    { label: 'Low', probability: 0.05 },
  ],
  model_metadata: {
    target_column: 'severity',
    classes: ['Critical', 'High', 'Low', 'Medium'],
    feature_schema_version: 'campaign_dataset_v1',
    n_features: 57,
    trained_at: '2026-09-15T05:28:10.967262+00:00',
  },
  prediction_context: { campaign_id: 'CAMP_7331E223', attack_id: 'T1078', event_id: 'evt-1' },
};

export const campaignFixture: Campaign = {
  campaign_id: 'CAMP_7331E223',
  campaign_label: 'Campaign …7331E223',
  attacker_ip: '185.220.101.7',
  victim_ip: '10.20.0.15',
  first_seen: '2026-01-15T10:00:00+00:00',
  last_seen: '2026-01-15T12:00:00+00:00',
  event_count: 5,
  risk_score: 72,
  raw_tps: 620,
  risk_level: 'HIGH',
  latest_technique: 'T1078',
  predicted_technique: 'T1053.003',
  prediction_confidence: 64,
};

export const campaignsFixture: { campaigns: Campaign[]; count: number } = {
  campaigns: [campaignFixture],
  count: 1,
};

export const overviewFixture: OverviewMetrics = {
  total_events: 130,
  active_campaigns: 71,
  unique_attackers: 12,
  techniques_detected: 34,
  learned_transitions: 8,
  total_hosts: 9,
  timestamp: '2026-01-15T12:00:00+00:00',
};

export const healthFixture: HealthStatus = {
  status: 'healthy',
  neo4j: 'connected',
  timestamp: '2026-01-15T12:00:00+00:00',
};

export const predictionFixture: Prediction = {
  campaign_id: 'CAMP_7331E223',
  current_technique: 'T1078',
  predicted_technique: 'T1053.003',
  confidence: 64,
  risk_level: 'HIGH',
  stage: 'Persistence',
  generated_at: '2026-01-15T12:00:00+00:00',
  source: 'LIKELY_NEXT',
};

/** dashboard_api.py's attribution_actors() -- the ThreatActor-node
 * mechanism (distinct from ThreatAttributionEngine's campaign-similarity
 * evidence in investigationResultFixture above) -- see THREAT_ATTRIBUTION.md. */
export const attributionActorsFixture = {
  campaign_id: 'CAMP_7331E223',
  attribution: [
    {
      actor_name: 'APT-TestGroup', confidence: 87, description: 'Known for credential access campaigns.',
      shared_techniques: ['T1110', 'T1078'], malware: ['TestMalware'], tools: ['Mimikatz'],
    },
  ],
};

export const correlationCampaignsFixture = {
  campaign_id: 'CAMP_7331E223',
  similar_campaigns: [
    {
      campaign_id: 'CAMP_PAST_1', campaign_label: 'Campaign …CAMP_PAST_1', similarity_score: 78,
      shared_techniques: ['T1110', 'T1078'], shared_tactics: ['Credential Access'],
      shared_attackers: ['185.220.101.7'], shared_hosts: [],
    },
  ],
};

export const graphDataFixture: GraphData = {
  nodes: [
    { id: 'n1', type: 'Attacker', label: '185.220.101.7' },
    { id: 'n2', type: 'Campaign', label: 'Campaign #1', campaign_id: 'CAMP_7331E223' },
    { id: 'n3', type: 'Technique', label: 'Brute Force', attack_id: 'T1110' },
  ],
  links: [
    { source: 'n1', target: 'n2', label: 'LAUNCHED' },
    { source: 'n2', target: 'n3', label: 'USES_TECHNIQUE' },
  ],
};

export const emptyGraphDataFixture: GraphData = { nodes: [], links: [] };
