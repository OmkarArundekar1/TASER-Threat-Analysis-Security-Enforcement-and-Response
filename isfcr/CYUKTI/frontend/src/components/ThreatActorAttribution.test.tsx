/**
 * Behavioral tests for ThreatActorAttribution.tsx -- the ThreatActor-node
 * attribution mechanism (dashboard_api.py's /api/attribution/actors/<id>,
 * distinct from ThreatAttributionEngine's evidence -- see
 * ../../../backend/THREAT_ATTRIBUTION.md). Verifies the component renders
 * this specific real contract correctly: successful attribution, no
 * attribution, loading, and error, without conflating it with the other
 * attribution mechanism.
 */
import { render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi, beforeEach } from 'vitest';
import { ThreatActorAttribution } from './ThreatActorAttribution';
import { api } from '../services/api';
import { useDashboard } from '../context/DashboardContext';
import { attributionActorsFixture } from '../test/fixtures';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
vi.mock('../services/api', () => ({ api: { campaignAttribution: vi.fn() } }));

const mockUseDashboard = vi.mocked(useDashboard);
const mockApi = vi.mocked(api);

function setSelectedCampaign(campaignId: string | null) {
  mockUseDashboard.mockReturnValue({ selectedCampaign: campaignId, selectCampaign: vi.fn() } as unknown as ReturnType<typeof useDashboard>);
}

beforeEach(() => {
  vi.clearAllMocks();
});

it('prompts campaign selection when none is selected, without calling the API', () => {
  setSelectedCampaign(null);
  render(<ThreatActorAttribution />);
  expect(screen.getByText('Select a Campaign')).toBeInTheDocument();
  expect(mockApi.campaignAttribution).not.toHaveBeenCalled();
});

it('renders real ThreatActor candidates with confidence, malware, and tools', async () => {
  setSelectedCampaign('CAMP_7331E223');
  mockApi.campaignAttribution.mockResolvedValue(attributionActorsFixture);

  render(<ThreatActorAttribution />);

  await waitFor(() => expect(screen.getByText('APT-TestGroup')).toBeInTheDocument());
  expect(screen.getByText('87%')).toBeInTheDocument();
  expect(screen.getByText('Mimikatz')).toBeInTheDocument();
  expect(screen.getByText('T1110')).toBeInTheDocument();
});

it('shows the no-match state for zero candidates, not an empty screen', async () => {
  setSelectedCampaign('CAMP_7331E223');
  mockApi.campaignAttribution.mockResolvedValue({ campaign_id: 'CAMP_7331E223', attribution: [] });

  render(<ThreatActorAttribution />);

  await waitFor(() => expect(screen.getByText(/no known threat actors matched/i)).toBeInTheDocument());
});

it('does not crash and clears data when the attribution fetch fails', async () => {
  setSelectedCampaign('CAMP_7331E223');
  mockApi.campaignAttribution.mockRejectedValue(new Error('Database unavailable'));
  const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

  render(<ThreatActorAttribution />);

  await waitFor(() => expect(screen.getByText(/no known threat actors matched/i)).toBeInTheDocument());
  consoleError.mockRestore();
});

it('re-fetches when the selected campaign changes', async () => {
  setSelectedCampaign('CAMP_1');
  mockApi.campaignAttribution.mockResolvedValue({ campaign_id: 'CAMP_1', attribution: [] });
  const { rerender } = render(<ThreatActorAttribution />);
  await waitFor(() => expect(mockApi.campaignAttribution).toHaveBeenCalledWith('CAMP_1'));

  setSelectedCampaign('CAMP_2');
  mockApi.campaignAttribution.mockResolvedValue(attributionActorsFixture);
  rerender(<ThreatActorAttribution />);

  await waitFor(() => expect(mockApi.campaignAttribution).toHaveBeenCalledWith('CAMP_2'));
});
