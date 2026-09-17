/**
 * Behavioral tests for ThreatCorrelation.tsx -- dashboard_api.py's
 * /api/correlation/campaigns/<id> (a standalone Cypher similarity query,
 * distinct from CampaignCorrelationEngine's operation-matching -- see
 * ../../../backend/CAMPAIGN_CORRELATION.md).
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { expect, it, vi, beforeEach } from 'vitest';
import { ThreatCorrelation } from './ThreatCorrelation';
import { api } from '../services/api';
import { useDashboard } from '../context/DashboardContext';
import { correlationCampaignsFixture } from '../test/fixtures';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
vi.mock('../services/api', () => ({ api: { campaignCorrelation: vi.fn() } }));

const mockUseDashboard = vi.mocked(useDashboard);
const mockApi = vi.mocked(api);
const selectCampaign = vi.fn();

function setSelectedCampaign(campaignId: string | null) {
  mockUseDashboard.mockReturnValue({ selectedCampaign: campaignId, selectCampaign } as unknown as ReturnType<typeof useDashboard>);
}

beforeEach(() => {
  vi.clearAllMocks();
});

it('prompts campaign selection when none is selected', () => {
  setSelectedCampaign(null);
  render(<ThreatCorrelation />);
  expect(screen.getByText('Select a Campaign')).toBeInTheDocument();
  expect(mockApi.campaignCorrelation).not.toHaveBeenCalled();
});

it('renders similar campaigns with shared techniques and tactics', async () => {
  setSelectedCampaign('CAMP_7331E223');
  mockApi.campaignCorrelation.mockResolvedValue(correlationCampaignsFixture);

  render(<ThreatCorrelation />);

  await waitFor(() => expect(screen.getByText('Campaign …CAMP_PAST_1')).toBeInTheDocument());
  expect(screen.getByText('78%')).toBeInTheDocument();
  expect(screen.getByText('Credential Access')).toBeInTheDocument();
  expect(screen.getByText('T1110')).toBeInTheDocument();
});

it('selects the correlated campaign when its label is clicked', async () => {
  setSelectedCampaign('CAMP_7331E223');
  mockApi.campaignCorrelation.mockResolvedValue(correlationCampaignsFixture);
  render(<ThreatCorrelation />);

  await waitFor(() => expect(screen.getByText('Campaign …CAMP_PAST_1')).toBeInTheDocument());
  fireEvent.click(screen.getByText('Campaign …CAMP_PAST_1'));

  expect(selectCampaign).toHaveBeenCalledWith('CAMP_PAST_1');
});

it('shows the no-similar-campaigns state for an empty result (matches the documented "no 404" contract)', async () => {
  setSelectedCampaign('CAMP_7331E223');
  mockApi.campaignCorrelation.mockResolvedValue({ campaign_id: 'CAMP_7331E223', similar_campaigns: [] });

  render(<ThreatCorrelation />);

  await waitFor(() => expect(screen.getByText(/no similar campaigns found/i)).toBeInTheDocument());
});
