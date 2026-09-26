import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { expect, it, vi, beforeEach } from 'vitest';
import { SOARPage } from './SOARPage';
import { api } from '../services/api';
import { useDashboard } from '../context/DashboardContext';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
vi.mock('../services/api', () => ({
  api: {
    soarStatus: vi.fn(), listPlaybooks: vi.fn(), listExecutions: vi.fn(), soarEffectiveness: vi.fn(),
    soarRecommendations: vi.fn(), generatePlaybook: vi.fn(), executePlaybook: vi.fn(), adaptPlaybook: vi.fn(),
    approveExecution: vi.fn(), rejectExecution: vi.fn(), pollExecution: vi.fn(), responseState: vi.fn(),
  },
}));

const mockUseDashboard = vi.mocked(useDashboard);
const mockApi = vi.mocked(api);
const selectCampaign = vi.fn();

const campaigns = [
  { campaign_id: 'CAMP_1', campaign_label: 'Campaign #1', attacker_ip: '1.1.1.1', victim_ip: '2.2.2.2', first_seen: 't', last_seen: 't', event_count: 3, risk_score: 10, risk_level: 'LOW' as const, latest_technique: 'T1078' },
];

function setDashboard(selectedCampaign: string | null) {
  mockUseDashboard.mockReturnValue({ campaigns, selectedCampaign, selectCampaign } as unknown as ReturnType<typeof useDashboard>);
}

const samplePlaybook = {
  playbook_id: 'pb_1', name: 'TEST_RESPONSE', version: 1, description: 'desc',
  trigger_conditions: {}, campaign_type: 'x', mitre_techniques: ['T1110'], severity: 'HIGH', risk: 900,
  required_evidence: [], execution_policy: 'analyst_approval' as const,
  shuffle_workflow_id: null, shuffle_workflow_version: null, created_at: 't', updated_at: 't',
  status: 'active' as const, source_campaign_id: 'CAMP_1', adapted_from_playbook_id: null,
  has_destructive_action: true,
  actions: [
    { action_id: 'act_1', action_type: 'block_ip', name: 'Block IP', description: '', order: 1,
      inputs: {}, expected_output: '', destructive: true, requires_approval: true, timeout_seconds: 60,
      reason: 'high severity' },
  ],
};

beforeEach(() => {
  vi.clearAllMocks();
  mockApi.soarStatus.mockResolvedValue({
    shuffle_webhook_configured: false, shuffle_api_configured: false, shuffle_reachable: null,
    stored_playbooks: 0, stored_executions: 0,
  });
  mockApi.listPlaybooks.mockResolvedValue({ playbooks: [] });
  mockApi.listExecutions.mockResolvedValue({ executions: [] });
  mockApi.soarEffectiveness.mockResolvedValue({ effectiveness: [] });
  mockApi.responseState.mockResolvedValue({
    correlation_id: 'CAMP_1', current_state: 'NO_RESPONSE_ACTIVITY', event_count: 0, events: [],
  });
});

it('shows the shuffle-not-configured indicator', async () => {
  setDashboard(null);
  render(<SOARPage />);
  await waitFor(() => expect(screen.getByText('Shuffle not configured')).toBeInTheDocument());
});

it('recommendations tab shows a candidate playbook once a campaign is selected', async () => {
  setDashboard('CAMP_1');
  mockApi.soarRecommendations.mockResolvedValue({
    campaign_id: 'CAMP_1', historical_matches: [], candidate_playbook: samplePlaybook, adapted_playbook: null,
  });

  render(<SOARPage />);
  await waitFor(() => expect(screen.getByText('TEST_RESPONSE')).toBeInTheDocument());
  expect(screen.getByText('1. Block IP')).toBeInTheDocument();
  expect(screen.getByText('DESTRUCTIVE')).toBeInTheDocument();
});

it('recommendations tab shows historical matches with an Adapt & Use action', async () => {
  setDashboard('CAMP_1');
  mockApi.soarRecommendations.mockResolvedValue({
    campaign_id: 'CAMP_1',
    historical_matches: [{
      playbook_id: 'pb_old', playbook_name: 'OLD_PB', source_campaign_id: 'CAMP_OLD',
      technique_similarity: 0.8, topology_similarity: 0.6, attacker_ip_match: false, victim_ip_match: false,
      historical_executions: 3, historical_successes: 3, historical_failures: 0, historical_success_rate: 1.0,
      recommendation_reason: '80% MITRE technique overlap',
    }],
    candidate_playbook: samplePlaybook, adapted_playbook: null,
  });
  mockApi.adaptPlaybook.mockResolvedValue(samplePlaybook);

  render(<SOARPage />);
  await waitFor(() => expect(screen.getByText('OLD_PB')).toBeInTheDocument());
  fireEvent.click(screen.getByText('Adapt & Use'));
  await waitFor(() => expect(mockApi.adaptPlaybook).toHaveBeenCalledWith('pb_old', 'CAMP_1'));
});

it('library tab lists stored playbooks and can execute one', async () => {
  setDashboard(null);
  mockApi.listPlaybooks.mockResolvedValue({ playbooks: [samplePlaybook] });
  mockApi.executePlaybook.mockResolvedValue({
    execution_id: 'exec_1', playbook_id: 'pb_1', playbook_version: 1, campaign_id: 'CAMP_1',
    operation_id: null, investigation_id: null, status: 'pending_approval', shuffle_execution_id: null,
    started_at: null, completed_at: null, action_results: [], approved_by: null, approval_decision_at: null,
    rejection_reason: null, created_at: 't',
  });

  render(<SOARPage />);
  fireEvent.click(screen.getByText('Library'));
  await waitFor(() => expect(screen.getByText('TEST_RESPONSE')).toBeInTheDocument());

  fireEvent.click(screen.getByText('Execute'));
  await waitFor(() => expect(mockApi.executePlaybook).toHaveBeenCalledWith('pb_1', 'CAMP_1'));
});

it('active executions tab shows approve/reject controls for pending_approval executions', async () => {
  setDashboard(null);
  mockApi.listExecutions.mockResolvedValue({
    executions: [{
      execution_id: 'exec_1', playbook_id: 'pb_1', playbook_version: 1, campaign_id: 'CAMP_1',
      operation_id: null, investigation_id: null, status: 'pending_approval', shuffle_execution_id: null,
      started_at: null, completed_at: null, action_results: [], approved_by: null, approval_decision_at: null,
      rejection_reason: null, created_at: 't',
    }],
  });
  mockApi.approveExecution.mockResolvedValue({} as never);

  render(<SOARPage />);
  fireEvent.click(screen.getByText('Active Executions'));
  await waitFor(() => expect(screen.getByText('Approve')).toBeInTheDocument());

  fireEvent.click(screen.getByText('Approve'));
  await waitFor(() => expect(mockApi.approveExecution).toHaveBeenCalledWith('exec_1'));
});

it('effectiveness tab shows success rate per playbook', async () => {
  setDashboard(null);
  mockApi.soarEffectiveness.mockResolvedValue({
    effectiveness: [{
      playbook_id: 'pb_1', playbook_name: 'TEST_RESPONSE', executions: 4, successful_executions: 3,
      failed_executions: 1, success_rate: 0.75, average_execution_seconds: 12.5, analyst_approvals: 4,
      analyst_rejections: 0, last_execution_at: 't', failure_reasons: [],
    }],
  });

  render(<SOARPage />);
  fireEvent.click(screen.getByText('Effectiveness'));
  await waitFor(() => expect(screen.getByText('75%')).toBeInTheDocument());
});

it('response tab prompts for a campaign when none is selected', async () => {
  setDashboard(null);
  render(<SOARPage />);
  fireEvent.click(screen.getByText('Active Response'));
  await waitFor(() => expect(screen.getByText(/Select a campaign to see its active-response lifecycle/)).toBeInTheDocument());
  expect(mockApi.responseState).not.toHaveBeenCalled();
});

it('response tab shows NO_RESPONSE_ACTIVITY honestly when nothing has happened', async () => {
  setDashboard('CAMP_1');
  render(<SOARPage />);
  fireEvent.click(screen.getByText('Active Response'));
  await waitFor(() => expect(mockApi.responseState).toHaveBeenCalledWith('CAMP_1'));
  await waitFor(() => expect(screen.getByText(/No response activity/)).toBeInTheDocument());
});

it('response tab never renders an unverified containment as blocked', async () => {
  setDashboard('CAMP_1');
  mockApi.responseState.mockResolvedValue({
    correlation_id: 'CAMP_1', current_state: 'CONTAINMENT_EXECUTED', event_count: 1,
    events: [{ event_type: 'CONTAINMENT_EXECUTED', timestamp: 't', campaign_id: 'CAMP_1',
               operation_id: null, investigation_id: null, playbook_id: null, execution_id: null, detail: {} }],
  });

  render(<SOARPage />);
  fireEvent.click(screen.getByText('Active Response'));
  await waitFor(() => expect(screen.getByText(/Containment executed \(not yet verified\)/)).toBeInTheDocument());
  expect(screen.queryByText(/ATTACK BLOCKED/i)).not.toBeInTheDocument();
});

it('response tab shows CONTAINMENT VERIFIED only when the backend actually reports it', async () => {
  setDashboard('CAMP_1');
  mockApi.responseState.mockResolvedValue({
    correlation_id: 'CAMP_1', current_state: 'CONTAINMENT_VERIFIED', event_count: 1,
    events: [{ event_type: 'CONTAINMENT_VERIFIED', timestamp: 't', campaign_id: 'CAMP_1',
               operation_id: null, investigation_id: null, playbook_id: null, execution_id: null, detail: {} }],
  });

  render(<SOARPage />);
  fireEvent.click(screen.getByText('Active Response'));
  await waitFor(() => expect(screen.getByText(/CONTAINMENT VERIFIED/)).toBeInTheDocument());
});
