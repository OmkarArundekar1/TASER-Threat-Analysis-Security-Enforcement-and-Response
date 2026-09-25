import { render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi, beforeEach } from 'vitest';
import { IncidentView } from './IncidentView';
import { api } from '../services/api';
import { useDashboard } from '../context/DashboardContext';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
vi.mock('../services/api', () => ({
  api: {
    incidentOverview: vi.fn(),
    incidentResponsePlan: vi.fn(),
    campaignTimeline: vi.fn(),
    ragSearch: vi.fn(),
    soarRecommendations: vi.fn(),
    investigate: vi.fn(),
  },
}));

const mockUseDashboard = vi.mocked(useDashboard);
const mockApi = vi.mocked(api);
const selectCampaign = vi.fn();

const campaigns = [
  { campaign_id: 'CAMP_1', campaign_label: 'Campaign #1', attacker_ip: '1.1.1.1', victim_ip: '2.2.2.2', first_seen: 't', last_seen: 't', event_count: 3, risk_score: 10, risk_level: 'LOW' as const, latest_technique: 'T1078' },
];

function setDashboard(selectedCampaign: string | null) {
  mockUseDashboard.mockReturnValue({
    campaigns, selectedCampaign, selectCampaign,
    investigationResult: null, isInvestigating: false, runInvestigation: vi.fn(),
  } as unknown as ReturnType<typeof useDashboard>);
}

const sampleOverview = {
  campaign: {
    campaign_id: 'CAMP_1', attacker_ip: '1.2.3.4', victim_ip: '10.0.0.5', status: 'ACTIVE',
    first_seen: '2026-01-01T00:00:00Z', last_seen: '2026-01-01T01:00:00Z', last_technique: 'T1110',
    techniques: ['T1110'], risk_score: 1300, predicted_next: null, prediction_confidence: null,
  },
  operation_id: 'OP_1',
  mitre: [{ mitre_id: 'T1110', technique_name: 'Brute Force', tactic: ['Credential Access'], mapping_source: 'X', mapping_confidence: 'N/A', mapping_reason: 'r' }],
  severity: { label: 'CRITICAL', risk_score: 1300 },
  threat_qualification: {
    classification: 'QUALIFIED_THREAT' as const, cti_score: 90,
    checks: [{ name: 'has_ioc', passed: true, detail: 'Attacker IP: 1.2.3.4' }],
    may_publish_to_misp: true, reason: 'All publication-readiness checks passed.',
  },
  campaign_selection: {
    campaign_id: 'CAMP_1',
    ranked_candidates: [{
      campaign_id: 'CAMP_OLD', composite_score: 0.9,
      signals: { topology_similarity: 0.9, technique_similarity: 0.8, temporal_similarity: 0.1, attacker_similarity: 1, host_similarity: 1 },
      historical_playbook_success_rate: null, historical_playbook_executions: 0,
    }],
    selected: {
      campaign_id: 'CAMP_OLD', composite_score: 0.9,
      signals: { topology_similarity: 0.9, technique_similarity: 0.8, temporal_similarity: 0.1, attacker_similarity: 1, host_similarity: 1 },
      historical_playbook_success_rate: null, historical_playbook_executions: 0,
    },
    alternatives: [], confidence: 'HIGH' as const, score_gap: 0.5, explanation: 'CAMP_OLD ranked highest.',
  },
  gnn: { gnn_available: true, gnn_model_version: 'v1' },
  soar: { playbooks: [], executions: [] },
  investigation_available: true,
  rag_available: true,
};

beforeEach(() => {
  vi.clearAllMocks();
  mockApi.campaignTimeline.mockResolvedValue({ campaign_id: 'CAMP_1', events: [] });
  mockApi.ragSearch.mockResolvedValue({ query: null, campaign_id: 'CAMP_1', sources: {} });
  mockApi.soarRecommendations.mockResolvedValue({
    campaign_id: 'CAMP_1', historical_matches: [], candidate_playbook: {} as any, adapted_playbook: null,
  });
  mockApi.incidentResponsePlan.mockResolvedValue({
    campaign_id: 'CAMP_1', threat_summary: 'summary', why_threat: [], mitre_techniques: [],
    severity: 'CRITICAL', risk_score: 1300, selected_historical_campaign: null, why_selected: null,
    selection_confidence: null, playbook: null, historical_playbook_success_rate: null,
    historical_playbook_executions: 0, misp_status: 'NOT_APPLICABLE', misp_reason: 'r',
  });
});

it('prompts for an incident selection when none is selected', () => {
  setDashboard(null);
  render(<IncidentView />);
  expect(screen.getByText(/Select an incident above/)).toBeInTheDocument();
});

it('renders the incident header with severity and threat classification badges', async () => {
  setDashboard('CAMP_1');
  mockApi.incidentOverview.mockResolvedValue(sampleOverview as any);

  render(<IncidentView />);

  await waitFor(() => expect(screen.getByText('CRITICAL')).toBeInTheDocument());
  expect(screen.getByText('QUALIFIED THREAT')).toBeInTheDocument();
  expect(screen.getByText('1.2.3.4')).toBeInTheDocument();
  expect(screen.getByText('OP_1', { exact: false })).toBeInTheDocument();
});

it('renders the campaign selection tree with the selected candidate marked', async () => {
  setDashboard('CAMP_1');
  mockApi.incidentOverview.mockResolvedValue(sampleOverview as any);

  render(<IncidentView />);

  await waitFor(() => expect(screen.getByText('CAMP_OLD')).toBeInTheDocument());
  expect(screen.getByText('Selected')).toBeInTheDocument();
  expect(screen.getByText(/CAMP_OLD ranked highest/)).toBeInTheDocument();
});

it('renders the threat qualification checklist with PASS/FAIL labels', async () => {
  setDashboard('CAMP_1');
  mockApi.incidentOverview.mockResolvedValue(sampleOverview as any);

  render(<IncidentView />);

  await waitFor(() => expect(screen.getByText('has ioc')).toBeInTheDocument());
  expect(screen.getByText('PASS')).toBeInTheDocument();
});

it('shows an error state when the overview request fails', async () => {
  setDashboard('CAMP_1');
  mockApi.incidentOverview.mockRejectedValue(new Error('boom'));

  render(<IncidentView />);
  await waitFor(() => expect(screen.getByText('boom')).toBeInTheDocument());
});
