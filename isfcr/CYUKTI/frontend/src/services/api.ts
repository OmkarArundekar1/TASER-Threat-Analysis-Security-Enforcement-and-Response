const API_BASE = '/api';

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API Error: ${res.status}`);
  }
  return res.json();
}

export interface EventsParams {
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_dir?: string;
  campaign?: string;
  severity?: string;
  attacker?: string;
  victim?: string;
  technique?: string;
  time_range?: string;
  q?: string;
}

export const api = {
  health: () => fetchApi<import('../types').HealthStatus>('/health'),

  overview: (params: { time_range?: string } = {}) => {
    const query = params.time_range ? `?time_range=${params.time_range}` : '';
    return fetchApi<import('../types').OverviewMetrics>(`/overview${query}`);
  },

  events: (params: EventsParams = {}) => {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) searchParams.set(key, String(value));
    });
    return fetchApi<{
      events: import('../types').AlertEvent[];
      total: number;
      page: number;
      page_size: number;
      total_pages: number;
    }>(`/events?${searchParams}`);
  },

  eventDetail: (eventId: string) =>
    fetchApi<import('../types').AlertEvent>(`/events/${encodeURIComponent(eventId)}`),

  graph: (campaign?: string, viewMode?: string, attacker?: string) => {
    const params = new URLSearchParams();
    if (campaign) params.set('campaign', campaign);
    if (attacker) params.set('attacker', attacker);
    if (viewMode) params.set('view_mode', viewMode);
    return fetchApi<import('../types').GraphData>(`/graph?${params.toString()}`);
  },

  graphExpand: (nodeId: string, depth: number = 1) => {
    const params = new URLSearchParams({ node_id: nodeId, depth: String(depth) });
    return fetchApi<import('../types').GraphData>(`/graph/expand?${params.toString()}`);
  },

  pathGraph: (campaignId?: string, timeframe?: string) => {
    let url = '/graph/paths';
    const params = new URLSearchParams();
    if (campaignId) params.append('campaign_id', campaignId);
    if (timeframe) params.append('timeframe', timeframe);
    const qs = params.toString();
    if (qs) url += '?' + qs;
    return fetchApi<any>(url);
  },

  campaigns: (params: { time_range?: string; attacker?: string } = {}) => {
    const searchParams = new URLSearchParams();
    if (params.time_range) searchParams.set('time_range', params.time_range);
    if (params.attacker) searchParams.set('attacker', params.attacker);
    const query = searchParams.toString() ? `?${searchParams.toString()}` : '';
    return fetchApi<{ campaigns: import('../types').Campaign[]; count: number }>(`/campaigns${query}`);
  },

  campaignTimeline: (campaignId: string) => 
    fetchApi<{ events: any[]; campaign_id: string }>(`/campaigns/${encodeURIComponent(campaignId)}/timeline`),

  campaignCorrelation: (campaignId: string) => 
    fetchApi<{ similar_campaigns: any[]; campaign_id: string }>(`/correlation/campaigns/${encodeURIComponent(campaignId)}`),

  analyticsPaths: () => 
    fetchApi<{ paths: any[] }>('/analytics/paths'),

  campaignAttribution: (campaignId: string) => 
    fetchApi<{ attribution: any[]; campaign_id: string }>(`/attribution/actors/${encodeURIComponent(campaignId)}`),

  riskPropagation: (campaignId: string) => 
    fetchApi<{ propagation: any[]; campaign_id: string }>(`/risk/propagation/${encodeURIComponent(campaignId)}`),

  attackers: () =>
    fetchApi<{ attackers: import('../types').Attacker[]; count: number }>('/attackers'),

  attackChain: (campaign?: string) => {
    const params = campaign ? `?campaign=${campaign}` : '';
    return fetchApi<{
      chain?: import('../types').AttackChainStep[];
      transitions?: import('../types').ChainTransition[];
      campaign_id?: string;
    }>(`/attack-chain${params}`);
  },

  predictions: () =>
    fetchApi<{ predictions: import('../types').Prediction[]; count: number }>('/predictions'),

  predict: (technique: string) =>
    fetchApi<{ current: string; predicted: string | null; confidence: number }>('/predict', {
      method: 'POST',
      body: JSON.stringify({ current_technique: technique }),
    }),

  recommendations: (technique?: string) => {
    const params = technique ? `?technique=${technique}` : '';
    return fetchApi<{
      recommendations: import('../types').RecommendationGroup[] | string[];
      technique?: string;
    }>(`/recommendations${params}`);
  },

  query: (cypher: string) =>
    fetchApi<import('../types').QueryResult>('/query', {
      method: 'POST',
      body: JSON.stringify({ query: cypher }),
    }),

  investigate: (
    campaignId: string,
    options: { attack_id?: string; event_id?: string; confidence_threshold?: number; max_steps?: number } = {}
  ) =>
    fetchApi<import('../types').InvestigationResult>(`/investigate/${encodeURIComponent(campaignId)}`, {
      method: 'POST',
      body: JSON.stringify(options),
    }),

  ragMitreSearch: (query: string, topK: number = 5) =>
    fetchApi<import('../types').MitreSearchResult>('/rag/mitre/search', {
      method: 'POST',
      body: JSON.stringify({ query, top_k: topK }),
    }),

  predictSeverity: (campaignId: string, attackId?: string, eventId?: string) =>
    fetchApi<import('../types').SeverityPrediction>('/ml/predict/severity', {
      method: 'POST',
      body: JSON.stringify({ campaign_id: campaignId, attack_id: attackId, event_id: eventId }),
    }),
};
