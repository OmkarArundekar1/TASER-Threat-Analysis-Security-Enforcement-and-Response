/**
 * Behavioral tests for the API service layer (api.ts) -- the real
 * exported functions, not mocks of them. Only the HTTP boundary
 * (global fetch) is mocked, with realistic response bodies matching
 * dashboard_api.py's actual contract (see ../test/fixtures.ts and
 * ../../DASHBOARD_API.md).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from './api';
import {
  campaignsFixture, investigationResultFixture, overviewFixture,
  severityPredictionFixture, healthFixture,
} from '../test/fixtures';

function mockFetchOnce(body: unknown, init: { ok?: boolean; status?: number; statusText?: string } = {}) {
  const { ok = true, status = 200, statusText = 'OK' } = init;
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok, status, statusText,
    json: () => Promise.resolve(body),
  }));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('api service -- success paths', () => {
  it('health() calls GET /api/health and returns the parsed body', async () => {
    mockFetchOnce(healthFixture);
    const result = await api.health();
    expect(fetch).toHaveBeenCalledWith('/api/health', expect.objectContaining({
      headers: { 'Content-Type': 'application/json' },
    }));
    expect(result).toEqual(healthFixture);
  });

  it('overview() encodes time_range as a query parameter', async () => {
    mockFetchOnce(overviewFixture);
    await api.overview({ time_range: '24h' });
    expect(fetch).toHaveBeenCalledWith('/api/overview?time_range=24h', expect.anything());
  });

  it('overview() omits the query string entirely when no time_range given', async () => {
    mockFetchOnce(overviewFixture);
    await api.overview();
    expect(fetch).toHaveBeenCalledWith('/api/overview', expect.anything());
  });

  it('events() serializes only the defined parameters', async () => {
    mockFetchOnce({ events: [], total: 0, page: 1, page_size: 50, total_pages: 0 });
    await api.events({ page: 2, campaign: 'CAMP_1', attacker: undefined });
    const calledUrl = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    const params = new URLSearchParams(calledUrl.split('?')[1]);
    expect(params.get('page')).toBe('2');
    expect(params.get('campaign')).toBe('CAMP_1');
    expect(params.has('attacker')).toBe(false);
  });

  it('campaigns() returns the real parsed campaign list shape', async () => {
    mockFetchOnce(campaignsFixture);
    const result = await api.campaigns();
    expect(result.count).toBe(1);
    expect(result.campaigns[0].campaign_id).toBe('CAMP_7331E223');
  });

  it('investigate() POSTs to the campaign-scoped URL with the options as the JSON body', async () => {
    mockFetchOnce(investigationResultFixture);
    const result = await api.investigate('CAMP_7331E223', { max_steps: 5 });
    expect(fetch).toHaveBeenCalledWith(
      '/api/investigate/CAMP_7331E223',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ max_steps: 5 }) }),
    );
    expect(result.steps).toHaveLength(2);
    expect(result.steps[1].why_selected).toContain('highest-scoring');
  });

  it('investigate() URL-encodes a campaign id containing special characters', async () => {
    mockFetchOnce(investigationResultFixture);
    await api.investigate('CAMP/with space');
    expect(fetch).toHaveBeenCalledWith(
      '/api/investigate/CAMP%2Fwith%20space',
      expect.anything(),
    );
  });

  it('predictSeverity() POSTs campaign_id/attack_id/event_id and returns the full prediction contract', async () => {
    mockFetchOnce(severityPredictionFixture);
    const result = await api.predictSeverity('CAMP_7331E223', 'T1078', 'evt-1');
    expect(fetch).toHaveBeenCalledWith(
      '/api/ml/predict/severity',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ campaign_id: 'CAMP_7331E223', attack_id: 'T1078', event_id: 'evt-1' }),
      }),
    );
    // the fields this session's backend work added that the frontend
    // type previously did not declare -- confirming the service layer
    // actually returns them to callers, not just that the type exists.
    expect(result.top_k).toHaveLength(4);
    expect(result.model_metadata.feature_schema_version).toBe('campaign_dataset_v1');
    expect(result.prediction_context.campaign_id).toBe('CAMP_7331E223');
  });

  it('predict() POSTs current_technique', async () => {
    mockFetchOnce({ current: 'T1110', predicted: 'T1078', confidence: 50 });
    await api.predict('T1110');
    expect(fetch).toHaveBeenCalledWith('/api/predict', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ current_technique: 'T1110' }),
    }));
  });

  it('query() POSTs the raw Cypher string under the "query" key', async () => {
    mockFetchOnce({ columns: ['n'], records: [], count: 0, graph: { nodes: [], links: [] } });
    await api.query('MATCH (n) RETURN n LIMIT 1');
    expect(fetch).toHaveBeenCalledWith('/api/query', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ query: 'MATCH (n) RETURN n LIMIT 1' }),
    }));
  });
});

describe('api service -- error paths', () => {
  it('propagates the backend error message on a 404', async () => {
    mockFetchOnce({ error: 'Campaign not found' }, { ok: false, status: 404 });
    await expect(api.campaignTimeline('does-not-exist')).rejects.toThrow('Campaign not found');
  });

  it('propagates the backend error message on a 500', async () => {
    mockFetchOnce({ error: 'Internal server error' }, { ok: false, status: 500 });
    await expect(api.overview()).rejects.toThrow('Internal server error');
  });

  it('propagates the backend error message on a 503 (matches dashboard_api.py Neo4j-unavailable contract)', async () => {
    mockFetchOnce({ error: 'Database unavailable: connection refused' }, { ok: false, status: 503 });
    await expect(api.campaigns()).rejects.toThrow('Database unavailable');
  });

  it('falls back to statusText when the error response body is not valid JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false, status: 500, statusText: 'Internal Server Error',
      json: () => Promise.reject(new Error('Unexpected token < in JSON')),
    }));
    await expect(api.overview()).rejects.toThrow('Internal Server Error');
  });

  it('falls back to a generic "API Error: <status>" when neither a JSON error field nor statusText is available', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false, status: 500, statusText: '',
      json: () => Promise.reject(new Error('Unexpected token < in JSON')),
    }));
    await expect(api.overview()).rejects.toThrow('API Error: 500');
  });

  it('does not silently convert an HTTP failure into a resolved value', async () => {
    mockFetchOnce({ error: 'boom' }, { ok: false, status: 400 });
    let succeeded = false;
    try {
      await api.predict('T1110');
      succeeded = true;
    } catch {
      // expected
    }
    expect(succeeded).toBe(false);
  });

  it('propagates a network failure (fetch rejecting) rather than swallowing it', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await expect(api.health()).rejects.toThrow('Failed to fetch');
  });
});
