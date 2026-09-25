import { render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi, beforeEach } from 'vitest';
import { CampaignSelectionPanel } from './CampaignSelectionPanel';
import { api } from '../services/api';
import { useDashboard } from '../context/DashboardContext';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
vi.mock('../services/api', () => ({ api: { campaignSelection: vi.fn() } }));

const mockUseDashboard = vi.mocked(useDashboard);
const mockApi = vi.mocked(api);

beforeEach(() => {
  vi.clearAllMocks();
});

it('prompts for a campaign selection when none is selected', () => {
  mockUseDashboard.mockReturnValue({ selectedCampaign: null } as unknown as ReturnType<typeof useDashboard>);
  render(<CampaignSelectionPanel />);
  expect(screen.getByText(/Select a campaign/)).toBeInTheDocument();
});

it('shows ranked candidates with the selected one highlighted', async () => {
  mockUseDashboard.mockReturnValue({ selectedCampaign: 'CAMP_1' } as unknown as ReturnType<typeof useDashboard>);
  mockApi.campaignSelection.mockResolvedValue({
    campaign_id: 'CAMP_1',
    ranked_candidates: [
      {
        campaign_id: 'CAMP_A', composite_score: 0.9,
        signals: { topology_similarity: 0.95, technique_similarity: 0.8, temporal_similarity: 0.7, attacker_similarity: 1.0, host_similarity: null },
        historical_playbook_success_rate: 0.83, historical_playbook_executions: 6,
      },
      {
        campaign_id: 'CAMP_B', composite_score: 0.4,
        signals: { topology_similarity: 0.3, technique_similarity: 0.2, temporal_similarity: 0.1, attacker_similarity: 0, host_similarity: 0 },
        historical_playbook_success_rate: null, historical_playbook_executions: 0,
      },
    ],
    selected: {
      campaign_id: 'CAMP_A', composite_score: 0.9,
      signals: { topology_similarity: 0.95, technique_similarity: 0.8, temporal_similarity: 0.7, attacker_similarity: 1.0, host_similarity: null },
      historical_playbook_success_rate: 0.83, historical_playbook_executions: 6,
    },
    alternatives: [],
    confidence: 'HIGH',
    score_gap: 0.5,
    explanation: 'CAMP_A ranked highest because its topology similarity matched most closely.',
  });

  render(<CampaignSelectionPanel />);

  await waitFor(() => expect(screen.getByText('CAMP_A')).toBeInTheDocument());
  expect(screen.getByText('CAMP_B')).toBeInTheDocument();
  expect(screen.getByText('HIGH confidence')).toBeInTheDocument();
  expect(screen.getByText(/ranked highest because its topology/)).toBeInTheDocument();
  expect(screen.getByText(/Historical response success: 83%/)).toBeInTheDocument();
});

it('shows an empty state when no candidates are found', async () => {
  mockUseDashboard.mockReturnValue({ selectedCampaign: 'CAMP_1' } as unknown as ReturnType<typeof useDashboard>);
  mockApi.campaignSelection.mockResolvedValue({
    campaign_id: 'CAMP_1', ranked_candidates: [], selected: null, alternatives: [],
    confidence: 'NONE', score_gap: null, explanation: 'No candidate campaigns were available to select from.',
  });

  render(<CampaignSelectionPanel />);
  await waitFor(() => expect(screen.getByText(/No comparable historical campaigns/)).toBeInTheDocument());
});
