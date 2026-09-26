import { render, screen, waitFor, fireEvent, within } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
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
    soarStatus: vi.fn(),
    responseState: vi.fn(),
    generatePlaybook: vi.fn(),
    executePlaybook: vi.fn(),
    approveExecution: vi.fn(),
    rejectExecution: vi.fn(),
    pollExecution: vi.fn(),
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

const samplePlaybook = {
  playbook_id: 'PB_1', name: 'Contain Brute Force Campaign', version: 1, description: 'd',
  trigger_conditions: {}, campaign_type: 'brute_force', mitre_techniques: ['T1110'],
  severity: 'CRITICAL', risk: 90, required_evidence: [],
  actions: [
    { action_id: 'A1', action_type: 'BLOCK_SOURCE_IP', name: 'Contain source IP', description: 'd', order: 1,
      inputs: {}, expected_output: 'o', destructive: true, requires_approval: true, timeout_seconds: 60, reason: 'Active brute-force source' },
  ],
  execution_policy: 'analyst_approval' as const,
  shuffle_workflow_id: null, shuffle_workflow_version: null,
  created_at: 't', updated_at: 't', status: 'active' as const,
  source_campaign_id: 'CAMP_1', adapted_from_playbook_id: null, has_destructive_action: true,
};

const noResponseState = { correlation_id: 'CAMP_1', current_state: 'NO_RESPONSE_ACTIVITY', event_count: 0, events: [] };

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
    selection_confidence: null, playbook: samplePlaybook as any, historical_playbook_success_rate: null,
    historical_playbook_executions: 0, misp_status: 'NOT_APPLICABLE', misp_reason: 'r',
  });
  mockApi.soarStatus.mockResolvedValue({
    shuffle_webhook_configured: false, shuffle_api_configured: false, shuffle_reachable: null,
    stored_playbooks: 0, stored_executions: 0,
  });
  mockApi.responseState.mockResolvedValue(noResponseState as any);
});

it('prompts for an incident selection when none is selected', () => {
  setDashboard(null);
  render(<IncidentView />);
  expect(screen.getByText(/Select an incident above/)).toBeInTheDocument();
});

it('renders the compact incident header with severity and threat classification badges', async () => {
  setDashboard('CAMP_1');
  mockApi.incidentOverview.mockResolvedValue(sampleOverview as any);

  render(<IncidentView />);

  await waitFor(() => expect(screen.getByText('CRITICAL SEVERITY')).toBeInTheDocument());
  expect(screen.getAllByText('QUALIFIED THREAT').length).toBeGreaterThan(0);
  expect(screen.getByText('1.2.3.4')).toBeInTheDocument();
  expect(screen.getByText(/OP_1/)).toBeInTheDocument();
});

it('renders the Current Assessment panel generated from real qualification checks, not hardcoded', async () => {
  setDashboard('CAMP_1');
  mockApi.incidentOverview.mockResolvedValue(sampleOverview as any);

  render(<IncidentView />);

  await waitFor(() => expect(screen.getByText('Attacker IP: 1.2.3.4')).toBeInTheDocument());
  expect(screen.getByText('PASS')).toBeInTheDocument();
  expect(screen.getByText('Current evidence is sufficient to treat this as a qualified threat.')).toBeInTheDocument();
});

it('shows an UNKNOWN state instead of a hard FAIL when no CTI score has been computed yet', async () => {
  setDashboard('CAMP_1');
  mockApi.incidentOverview.mockResolvedValue({
    ...sampleOverview,
    threat_qualification: {
      classification: 'NOT_THREAT', cti_score: null,
      checks: [{ name: 'threat_classification_qualified', passed: false, detail: 'No CTI confidence has been computed for this incident yet.' }],
      may_publish_to_misp: false, reason: 'Blocked on: threat_classification_qualified.',
    },
  } as any);

  render(<IncidentView />);

  await waitFor(() => expect(screen.getByText('UNKNOWN')).toBeInTheDocument());
});

it('renders the campaign selection tree and opens a comparison panel on click', async () => {
  setDashboard('CAMP_1');
  mockApi.incidentOverview.mockResolvedValue(sampleOverview as any);

  render(<IncidentView />);

  await waitFor(() => expect(screen.getByText('CAMP_OLD')).toBeInTheDocument());
  expect(screen.getByText('Selected')).toBeInTheDocument();
  expect(screen.getByText(/CAMP_OLD ranked highest/)).toBeInTheDocument();

  fireEvent.click(screen.getByText('CAMP_OLD'));
  await waitFor(() => expect(screen.getByText('Values are similarity signals, not probabilities.')).toBeInTheDocument());
});

it('opens the investigation drawer when "Open Investigation" is clicked', async () => {
  setDashboard('CAMP_1');
  mockApi.incidentOverview.mockResolvedValue(sampleOverview as any);

  render(<IncidentView />);
  await waitFor(() => expect(screen.getByText('Open Investigation')).toBeInTheDocument());

  fireEvent.click(screen.getByText('Open Investigation'));
  const dialog = await screen.findByRole('dialog');
  expect(within(dialog).getByText(/Investigation — CAMP_1/)).toBeInTheDocument();
});

it('expands an attack-story step to show real detail on click', async () => {
  setDashboard('CAMP_1');
  mockApi.incidentOverview.mockResolvedValue(sampleOverview as any);

  render(<IncidentView />);
  await waitFor(() => expect(screen.getByText('MITRE')).toBeInTheDocument());

  fireEvent.click(screen.getByText('MITRE'));
  await waitFor(() => expect(screen.getByText(/tactic unknown|Credential Access/)).toBeInTheDocument());
});

it('shows an error state when the overview request fails', async () => {
  setDashboard('CAMP_1');
  mockApi.incidentOverview.mockRejectedValue(new Error('boom'));

  render(<IncidentView />);
  await waitFor(() => expect(screen.getByText('boom')).toBeInTheDocument());
});

describe('Response Execution', () => {
  it('never shows the old dead-end "Execution unavailable" message for a RECOMMENDED playbook', async () => {
    setDashboard('CAMP_1');
    mockApi.incidentOverview.mockResolvedValue(sampleOverview as any); // soar.playbooks: []

    render(<IncidentView />);
    await waitFor(() => expect(screen.getByText('RECOMMENDED')).toBeInTheDocument());
    expect(screen.queryByText(/Execution unavailable/)).not.toBeInTheDocument();
  });

  it('A: recommendation exists, no playbook -> Generate Playbook is the obvious next action', async () => {
    setDashboard('CAMP_1');
    mockApi.incidentOverview.mockResolvedValue(sampleOverview as any);

    render(<IncidentView />);
    await waitFor(() => expect(screen.getByText('Contain source IP')).toBeInTheDocument()); // the recommendation, from the response plan
    expect(screen.getByRole('button', { name: /Generate Playbook/ })).toBeInTheDocument();
    expect(screen.getByText('Waiting for playbook.')).toBeInTheDocument();
  });

  it('B: generating a playbook updates the card to show it, with a Review Playbook action', async () => {
    setDashboard('CAMP_1');
    mockApi.incidentOverview.mockResolvedValue(sampleOverview as any);
    mockApi.generatePlaybook.mockResolvedValue(samplePlaybook as any);

    render(<IncidentView />);
    // The button renders immediately but stays disabled until the response
    // plan (which gates it) has loaded -- wait for the plan's own content
    // first so the click below lands on an enabled button.
    await waitFor(() => expect(screen.getByText('Contain source IP')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByRole('button', { name: /Generate Playbook/ })).toBeEnabled());

    fireEvent.click(screen.getByRole('button', { name: /Generate Playbook/ }));
    await waitFor(() => expect(mockApi.generatePlaybook).toHaveBeenCalledWith('CAMP_1'));

    await waitFor(() => expect(screen.getByText('Contain Brute Force Campaign')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Review Playbook' })).toBeInTheDocument();
    expect(screen.getByText('READY FOR REVIEW')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Review Playbook' }));
    const dialog = await screen.findByRole('dialog');
    expect(within(dialog).getByText('1. Contain source IP')).toBeInTheDocument();
  });

  it('shows a playbook-generation failure without losing the recommendation', async () => {
    setDashboard('CAMP_1');
    mockApi.incidentOverview.mockResolvedValue(sampleOverview as any);
    mockApi.generatePlaybook.mockRejectedValue(new Error('boom'));

    render(<IncidentView />);
    await waitFor(() => expect(screen.getByText('Contain source IP')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByRole('button', { name: /Generate Playbook/ })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: /Generate Playbook/ }));

    await waitFor(() => expect(screen.getByText('Unable to generate playbook. The recommendation is still available.')).toBeInTheDocument());
    expect(screen.getByText('Contain source IP')).toBeInTheDocument(); // recommendation still visible
  });

  it('C: an already-generated playbook with Shuffle unreachable shows UNAVAILABLE, never a fake Execute button', async () => {
    setDashboard('CAMP_1');
    mockApi.incidentOverview.mockResolvedValue({
      ...sampleOverview,
      soar: { playbooks: [samplePlaybook], executions: [] },
    } as any);
    mockApi.soarStatus.mockResolvedValue({
      shuffle_webhook_configured: true, shuffle_api_configured: false, shuffle_reachable: null,
      stored_playbooks: 1, stored_executions: 0,
    });

    render(<IncidentView />);
    await waitFor(() => expect(screen.getByText('Shuffle UNAVAILABLE')).toBeInTheDocument());
    expect(screen.getByText(/Playbook generated successfully\. Execution is currently unavailable/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Execute$/ })).not.toBeInTheDocument();
    // still inspectable
    expect(screen.getByRole('button', { name: 'Review Playbook' })).toBeInTheDocument();
  });

  it('executing a playbook that requires approval shows Approve & Execute / Reject, not immediate execution', async () => {
    setDashboard('CAMP_1');
    mockApi.incidentOverview.mockResolvedValue({
      ...sampleOverview,
      soar: { playbooks: [samplePlaybook], executions: [] },
    } as any);
    mockApi.soarStatus.mockResolvedValue({
      shuffle_webhook_configured: true, shuffle_api_configured: true, shuffle_reachable: true,
      stored_playbooks: 1, stored_executions: 0,
    });
    mockApi.executePlaybook.mockResolvedValue({
      execution_id: 'EX_1', playbook_id: 'PB_1', playbook_version: 1, campaign_id: 'CAMP_1',
      operation_id: null, investigation_id: null, status: 'pending_approval', shuffle_execution_id: null,
      started_at: null, completed_at: null, action_results: [], approved_by: null,
      approval_decision_at: null, rejection_reason: null, created_at: 't',
    } as any);

    render(<IncidentView />);
    await waitFor(() => expect(screen.getByRole('button', { name: /^Execute$/ })).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /^Execute$/ }));
    await waitFor(() => expect(mockApi.executePlaybook).toHaveBeenCalledWith('PB_1', 'CAMP_1'));

    await waitFor(() => expect(screen.getByText('APPROVAL REQUIRED')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /Approve & Execute/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Reject' })).toBeInTheDocument();
    // never claim containment before it's actually confirmed
    expect(screen.queryByText(/ATTACK BLOCKED/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^CONTAINED$/)).not.toBeInTheDocument();
  });

  it('D: execution failure is shown explicitly and never claimed as successful containment', async () => {
    setDashboard('CAMP_1');
    mockApi.incidentOverview.mockResolvedValue({
      ...sampleOverview,
      soar: {
        playbooks: [samplePlaybook],
        executions: [{
          execution_id: 'EX_1', playbook_id: 'PB_1', playbook_version: 1, campaign_id: 'CAMP_1',
          operation_id: null, investigation_id: null, status: 'failed', shuffle_execution_id: 'SH_1',
          started_at: 't', completed_at: 't', action_results: [], approved_by: 'analyst',
          approval_decision_at: 't', rejection_reason: null, created_at: 't',
        }],
      },
    } as any);

    render(<IncidentView />);
    await waitFor(() => expect(screen.getByText('FAILED')).toBeInTheDocument());
    expect(screen.getByText('Execution failed. No successful containment is claimed.')).toBeInTheDocument();
    expect(screen.queryByText(/ATTACK BLOCKED/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^CONTAINED$/)).not.toBeInTheDocument();
  });

  it('E: a successful execution whose containment was not independently verified says so explicitly', async () => {
    setDashboard('CAMP_1');
    mockApi.incidentOverview.mockResolvedValue({
      ...sampleOverview,
      soar: {
        playbooks: [samplePlaybook],
        executions: [{
          execution_id: 'EX_1', playbook_id: 'PB_1', playbook_version: 1, campaign_id: 'CAMP_1',
          operation_id: null, investigation_id: null, status: 'success', shuffle_execution_id: 'SH_1',
          started_at: 't', completed_at: 't', action_results: [], approved_by: 'analyst',
          approval_decision_at: 't', rejection_reason: null, created_at: 't',
        }],
      },
    } as any);
    mockApi.responseState.mockResolvedValue({
      correlation_id: 'CAMP_1', current_state: 'CONTAINMENT_NOT_VERIFIED', event_count: 2, events: [],
    } as any);

    render(<IncidentView />);
    await waitFor(() => expect(screen.getByText('EXECUTED')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText('Response was submitted, but containment could not be verified.')).toBeInTheDocument());
    expect(screen.queryByText(/ATTACK BLOCKED/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^CONTAINED$/)).not.toBeInTheDocument();
  });

  it('shows independently VERIFIED containment only when the real verification pipeline confirms it', async () => {
    setDashboard('CAMP_1');
    mockApi.incidentOverview.mockResolvedValue({
      ...sampleOverview,
      soar: {
        playbooks: [samplePlaybook],
        executions: [{
          execution_id: 'EX_1', playbook_id: 'PB_1', playbook_version: 1, campaign_id: 'CAMP_1',
          operation_id: null, investigation_id: null, status: 'success', shuffle_execution_id: 'SH_1',
          started_at: 't', completed_at: 't', action_results: [], approved_by: 'analyst',
          approval_decision_at: 't', rejection_reason: null, created_at: 't',
        }],
      },
    } as any);
    mockApi.responseState.mockResolvedValue({
      correlation_id: 'CAMP_1', current_state: 'CONTAINMENT_VERIFIED', event_count: 3, events: [],
    } as any);

    render(<IncidentView />);
    await waitFor(() => expect(screen.getByText('Containment independently verified.')).toBeInTheDocument());
  });

  it('a recommend_only playbook never offers execution', async () => {
    setDashboard('CAMP_1');
    mockApi.incidentOverview.mockResolvedValue({
      ...sampleOverview,
      soar: { playbooks: [{ ...samplePlaybook, execution_policy: 'recommend_only' }], executions: [] },
    } as any);

    render(<IncidentView />);
    await waitFor(() => expect(screen.getByText('READY FOR REVIEW')).toBeInTheDocument());
    expect(screen.getByText(/recommendation-only/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Execute$/ })).not.toBeInTheDocument();
  });
});
