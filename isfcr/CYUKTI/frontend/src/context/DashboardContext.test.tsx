/**
 * Integration-style tests for DashboardContext.tsx -- the real
 * DashboardProvider, the real api.ts service layer, with ONLY the HTTP
 * boundary (global fetch) mocked. This is the mission's explicit
 * "component -> real API service -> mock HTTP boundary -> realistic
 * backend response -> component state/render" pattern, applied to
 * CYUKTI's central data-loading logic: refreshAll()'s Promise.allSettled
 * handling of 8 concurrent endpoints, and the loading/error/partial-
 * failure states that logic produces.
 */
import { act, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { DashboardProvider, useDashboard } from './DashboardContext';
import { healthFixture, overviewFixture, campaignsFixture } from '../test/fixtures';

function routeFor(url: string): { body: unknown; ok: boolean; status?: number } {
  if (url.startsWith('/api/health')) return { body: healthFixture, ok: true };
  if (url.startsWith('/api/overview')) return { body: overviewFixture, ok: true };
  if (url.startsWith('/api/events')) return { body: { events: [], total: 0, page: 1, page_size: 50, total_pages: 0 }, ok: true };
  if (url.startsWith('/api/graph')) return { body: { nodes: [], links: [] }, ok: true };
  if (url.startsWith('/api/campaigns')) return { body: campaignsFixture, ok: true };
  if (url.startsWith('/api/predictions')) return { body: { predictions: [], count: 0 }, ok: true };
  if (url.startsWith('/api/recommendations')) return { body: { recommendations: [] }, ok: true };
  if (url.startsWith('/api/attackers')) return { body: { attackers: [], count: 0 }, ok: true };
  return { body: { error: `unexpected route in test: ${url}` }, ok: false, status: 404 };
}

function mockAllEndpointsHealthy() {
  vi.stubGlobal('fetch', vi.fn((url: string) => {
    const { body, ok, status } = routeFor(url);
    return Promise.resolve({ ok, status: status ?? 200, statusText: 'OK', json: () => Promise.resolve(body) });
  }));
}

function Consumer() {
  const { loading, error, health, overview, campaigns } = useDashboard();
  return (
    <div>
      <div data-testid="loading">{String(loading)}</div>
      <div data-testid="error">{error ?? 'none'}</div>
      <div data-testid="health-status">{health?.status ?? 'none'}</div>
      <div data-testid="overview-events">{overview?.total_events ?? 'none'}</div>
      <div data-testid="campaign-count">{campaigns.length}</div>
    </div>
  );
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

describe('DashboardProvider -- refreshAll loading/success/error states', () => {
  it('starts loading, then reflects real parsed data from every endpoint on success', async () => {
    mockAllEndpointsHealthy();
    render(<DashboardProvider><Consumer /></DashboardProvider>);

    expect(screen.getByTestId('loading').textContent).toBe('true');

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));

    expect(screen.getByTestId('error').textContent).toBe('none');
    expect(screen.getByTestId('health-status').textContent).toBe('healthy');
    expect(screen.getByTestId('overview-events').textContent).toBe('130');
    expect(screen.getByTestId('campaign-count').textContent).toBe('1');
  });

  it('reports a single clear error when every endpoint fails, without crashing', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false, status: 503, statusText: 'Service Unavailable',
      json: () => Promise.resolve({ error: 'Database unavailable: connection refused' }),
    }));

    render(<DashboardProvider><Consumer /></DashboardProvider>);

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));
    expect(screen.getByTestId('error').textContent).toMatch(/all api calls failed/i);
    expect(screen.getByTestId('health-status').textContent).toBe('none');
  });

  it('degrades gracefully on a PARTIAL failure: successful endpoints populate state, no top-level error is shown', async () => {
    vi.stubGlobal('fetch', vi.fn((url: string) => {
      if (url.startsWith('/api/campaigns')) {
        return Promise.resolve({
          ok: false, status: 500, statusText: 'Internal Server Error',
          json: () => Promise.resolve({ error: 'Internal server error' }),
        });
      }
      const { body, ok, status } = routeFor(url);
      return Promise.resolve({ ok, status: status ?? 200, statusText: 'OK', json: () => Promise.resolve(body) });
    }));

    render(<DashboardProvider><Consumer /></DashboardProvider>);

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));
    // Real Promise.allSettled semantics: one rejected promise among eight
    // must not be reported as total failure, and the endpoints that DID
    // succeed must still be reflected in state -- this is the actual
    // resilience contract refreshAll() implements, verified end to end
    // through the real provider rather than asserted from reading the
    // source.
    expect(screen.getByTestId('error').textContent).toBe('none');
    expect(screen.getByTestId('health-status').textContent).toBe('healthy');
    expect(screen.getByTestId('campaign-count').textContent).toBe('0'); // the one that failed keeps its prior (empty) value
  });

  it('does not leave the UI stuck in loading forever if fetch itself throws synchronously-rejecting promises', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    render(<DashboardProvider><Consumer /></DashboardProvider>);

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));
    expect(screen.getByTestId('error').textContent).toMatch(/all api calls failed/i);
  });
});

describe('DashboardProvider -- selection triggers a real re-fetch', () => {
  function SelectorConsumer() {
    const { selectCampaign, selectedCampaign, loading } = useDashboard();
    return (
      <div>
        <div data-testid="selected">{selectedCampaign ?? 'none'}</div>
        <div data-testid="loading">{String(loading)}</div>
        <button onClick={() => selectCampaign('CAMP_7331E223')}>select</button>
      </div>
    );
  }

  it('re-runs refreshAll (a real fetch round trip) when selectCampaign is called', async () => {
    mockAllEndpointsHealthy();
    render(<DashboardProvider><SelectorConsumer /></DashboardProvider>);
    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));

    const fetchMock = fetch as unknown as ReturnType<typeof vi.fn>;
    const callsBeforeSelection = fetchMock.mock.calls.length;

    await act(async () => {
      screen.getByRole('button', { name: 'select' }).click();
    });

    expect(screen.getByTestId('selected').textContent).toBe('CAMP_7331E223');
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThan(callsBeforeSelection));
  });
});
