import { render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi, beforeEach } from 'vitest';
import { ThreatIntelligencePage } from './ThreatIntelligencePage';
import { api } from '../services/api';

vi.mock('../services/api', () => ({ api: { mispStatus: vi.fn() } }));
const mockApi = vi.mocked(api);

beforeEach(() => {
  vi.clearAllMocks();
});

it('shows the honest not-connected state when no credential is configured', async () => {
  mockApi.mispStatus.mockResolvedValue({
    misp_url: 'https://localhost:8443', credential_configured: false, authenticated: false,
    cached_campaigns: 0, campaign_event_map: {},
    status_message: 'MISP publication requires configured authentication (MISP_API_KEY is not set).',
  });

  render(<ThreatIntelligencePage />);

  await waitFor(() => expect(screen.getByText('NOT CONNECTED')).toBeInTheDocument());
  expect(screen.getByText(/MISP_API_KEY is not set/)).toBeInTheDocument();
});

it('renders the campaign-to-event mapping when campaigns are cached', async () => {
  mockApi.mispStatus.mockResolvedValue({
    misp_url: 'https://localhost:8443', credential_configured: true, authenticated: true,
    cached_campaigns: 1, campaign_event_map: { CAMP_1: 42 }, status_message: 'Connected',
  });

  render(<ThreatIntelligencePage />);

  await waitFor(() => expect(screen.getByText('CONNECTED')).toBeInTheDocument());
  expect(screen.getByText('CAMP_1')).toBeInTheDocument();
  expect(screen.getByText('Event #42')).toBeInTheDocument();
});
