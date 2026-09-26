/**
 * Behavioral tests for CampaignIntelligence.tsx -- the campaign
 * list/detail view. Verifies list rendering, empty state, selection
 * interaction, and the detail view's composition of campaign +
 * prediction + recommendation + timeline data.
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { CampaignIntelligence } from './CampaignIntelligence';
import { api } from '../services/api';
import { useDashboard } from '../context/DashboardContext';
import { campaignFixture, predictionFixture } from '../test/fixtures';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
vi.mock('../services/api', () => ({ api: { campaignTimeline: vi.fn() } }));

const mockUseDashboard = vi.mocked(useDashboard);
const mockApi = vi.mocked(api);

const selectCampaign = vi.fn();

function setState(overrides: Partial<ReturnType<typeof useDashboard>>) {
  mockUseDashboard.mockReturnValue({
    campaigns: [], selectedCampaign: null, selectCampaign, predictions: [], recommendations: [],
    ...overrides,
  } as ReturnType<typeof useDashboard>);
}

beforeEach(() => {
  vi.clearAllMocks();
  mockApi.campaignTimeline.mockResolvedValue({ events: [], campaign_id: 'CAMP_7331E223' });
});

describe('CampaignIntelligence -- list view', () => {
  it('shows the empty state when there are no campaigns', () => {
    setState({ campaigns: [] });
    render(<CampaignIntelligence />);
    expect(screen.getByText(/no active campaigns/i)).toBeInTheDocument();
  });

  it('renders the canonical campaign_id (never the campaign_label) plus the other real fields', () => {
    setState({ campaigns: [campaignFixture] });
    render(<CampaignIntelligence />);
    expect(screen.getByText(campaignFixture.campaign_id)).toBeInTheDocument();
    expect(screen.queryByText(campaignFixture.campaign_label!)).not.toBeInTheDocument();
    expect(screen.getByText(new RegExp(`${campaignFixture.attacker_ip}.*${campaignFixture.victim_ip}`))).toBeInTheDocument();
    expect(screen.getByText(campaignFixture.latest_technique)).toBeInTheDocument();
    expect(screen.getByText(campaignFixture.predicted_technique!)).toBeInTheDocument();
  });

  it('selects a campaign when its card is clicked', () => {
    setState({ campaigns: [campaignFixture] });
    render(<CampaignIntelligence />);
    fireEvent.click(screen.getByText(campaignFixture.campaign_id));
    expect(selectCampaign).toHaveBeenCalledWith(campaignFixture.campaign_id);
  });
});

describe('CampaignIntelligence -- detail view', () => {
  it('renders the campaign summary, prediction, and recommendations for the selected campaign', async () => {
    setState({
      campaigns: [campaignFixture],
      selectedCampaign: campaignFixture.campaign_id,
      predictions: [predictionFixture],
      recommendations: [{ technique_id: 'T1053.003', recommendations: ['Restrict scheduled task creation'] }],
    });
    render(<CampaignIntelligence />);

    expect(screen.getByText(campaignFixture.attacker_ip)).toBeInTheDocument();
    expect(screen.getByText(`${campaignFixture.risk_score}/100 · ${campaignFixture.risk_level}`)).toBeInTheDocument();
    expect(screen.getByText(predictionFixture.predicted_technique)).toBeInTheDocument();
    expect(screen.getByText('Restrict scheduled task creation')).toBeInTheDocument();

    await waitFor(() => expect(mockApi.campaignTimeline).toHaveBeenCalledWith(campaignFixture.campaign_id));
  });

  it('renders the observed timeline once it loads', async () => {
    mockApi.campaignTimeline.mockResolvedValue({
      events: [{ first_seen: '2026-01-15T10:00:00Z', occurrences: 3, technique_name: 'Brute Force', technique_id: 'T1110', stage: 'Credential Access' }],
      campaign_id: campaignFixture.campaign_id,
    });
    setState({ campaigns: [campaignFixture], selectedCampaign: campaignFixture.campaign_id });
    render(<CampaignIntelligence />);

    await waitFor(() => expect(screen.getByText('Brute Force')).toBeInTheDocument());
    expect(screen.getByText('x3')).toBeInTheDocument();
  });

  it('clears the selection when the back button is clicked', async () => {
    setState({ campaigns: [campaignFixture], selectedCampaign: campaignFixture.campaign_id });
    render(<CampaignIntelligence />);
    await waitFor(() => expect(mockApi.campaignTimeline).toHaveBeenCalled());
    fireEvent.click(screen.getByRole('button')); // ArrowLeft back button is the first/only button in this header
    expect(selectCampaign).toHaveBeenCalledWith(null);
  });

  it('does not crash when the selected campaign has no matching entry in the campaigns list', async () => {
    setState({ campaigns: [], selectedCampaign: 'CAMP_NOT_IN_LIST' });
    expect(() => render(<CampaignIntelligence />)).not.toThrow();
    await waitFor(() => expect(mockApi.campaignTimeline).toHaveBeenCalled());
  });
});
