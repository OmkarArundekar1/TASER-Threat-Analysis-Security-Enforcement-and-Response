import { createContext, useContext, useState, useCallback, useEffect } from 'react';
import type { ReactNode } from 'react';
import { api } from '../services/api';
import type {
  HealthStatus, OverviewMetrics, AlertEvent, GraphData,
  Campaign, Prediction, RecommendationGroup, Attacker
} from '../types';

interface DashboardState {
  health: HealthStatus | null;
  overview: OverviewMetrics | null;
  events: AlertEvent[];
  eventsTotal: number;
  eventsPage: number;
  graphData: GraphData | null;
  campaigns: Campaign[];
  predictions: Prediction[];
  recommendations: RecommendationGroup[];
  attackers: Attacker[];
  selectedCampaign: string | null;
  selectedAttacker: string | null;
  selectedTechnique: string | null;
  globalTimeRange: string;
  graphLayer: 'campaign' | 'investigation' | 'threat_intel';
  resetKey: number;
  loading: boolean;
  error: string | null;
  isPathExplorerOpen: boolean;
}

interface DashboardContextType extends DashboardState {
  selectCampaign: (id: string | null) => void;
  selectAttacker: (ip: string | null) => void;
  selectTechnique: (id: string | null) => void;
  setGlobalTimeRange: (range: string) => void;
  setGraphLayer: (layer: 'campaign' | 'investigation' | 'threat_intel') => void;
  setEventsPage: (page: number) => void;
  setPathExplorerOpen: (isOpen: boolean) => void;
  resetDashboard: () => void;
  refreshAll: () => Promise<void>;
  refreshEvents: () => Promise<void>;
  refreshGraph: () => Promise<void>;
  addEvents: (events: AlertEvent[]) => void;
}

const DashboardContext = createContext<DashboardContextType | null>(null);

export function useDashboard() {
  const ctx = useContext(DashboardContext);
  if (!ctx) throw new Error('useDashboard must be used within DashboardProvider');
  return ctx;
}

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<DashboardState>({
    health: null, overview: null, events: [], eventsTotal: 0, eventsPage: 1,
    graphData: null, campaigns: [], predictions: [], recommendations: [],
    attackers: [], selectedCampaign: null, selectedAttacker: null,
    selectedTechnique: null, globalTimeRange: '24h', graphLayer: 'campaign', resetKey: 0, loading: true, error: null,
    isPathExplorerOpen: false,
  });

  const selectCampaign = useCallback((id: string | null) => {
    setState(s => ({ ...s, selectedCampaign: id }));
  }, []);

  const selectAttacker = useCallback((ip: string | null) => {
    setState(s => ({ ...s, selectedAttacker: ip }));
  }, []);

  const selectTechnique = useCallback((id: string | null) => {
    setState(s => ({ ...s, selectedTechnique: id }));
  }, []);

  const setGlobalTimeRange = useCallback((range: string) => {
    setState(s => ({ ...s, globalTimeRange: range }));
  }, []);

  const setGraphLayer = useCallback((layer: 'campaign' | 'investigation' | 'threat_intel') => {
    setState(s => {
      let selectedCamp = s.selectedCampaign;
      if (layer !== 'campaign' && !selectedCamp && s.campaigns.length > 0) {
        const highestRisk = [...s.campaigns].sort((a, b) => b.risk_score - a.risk_score)[0];
        selectedCamp = highestRisk.campaign_id;
      }
      return { ...s, graphLayer: layer, selectedCampaign: selectedCamp };
    });
  }, []);

  const setEventsPage = useCallback((page: number) => {
    setState(s => ({ ...s, eventsPage: page }));
  }, []);

  const setPathExplorerOpen = useCallback((isOpen: boolean) => {
    setState(s => ({ ...s, isPathExplorerOpen: isOpen }));
  }, []);

  const resetDashboard = useCallback(() => {
    setState(s => ({ 
      ...s, 
      selectedCampaign: null, 
      selectedAttacker: null, 
      selectedTechnique: null, 
      graphLayer: 'campaign', 
      isPathExplorerOpen: false,
      eventsPage: 1,
      resetKey: s.resetKey + 1,
    }));
    window.dispatchEvent(new Event('watchdog-reset'));
  }, []);

  const addEvents = useCallback((newEvents: AlertEvent[]) => {
    setState(s => ({ ...s, events: [...newEvents, ...s.events].slice(0, 200) }));
  }, []);

  const refreshEvents = useCallback(async () => {
    try {
      const data = await api.events({
        page: state.eventsPage,
        page_size: 50,
        campaign: state.selectedCampaign || undefined,
        attacker: state.selectedAttacker || undefined,
        technique: state.selectedTechnique || undefined,
        time_range: state.globalTimeRange,
      });
      setState(s => ({ ...s, events: data.events, eventsTotal: data.total }));
    } catch (e) {
      console.error('refreshEvents failed', e);
    }
  }, [state.eventsPage, state.selectedCampaign, state.selectedAttacker, state.selectedTechnique, state.globalTimeRange]);

  const refreshGraph = useCallback(async () => {
    try {
      const data = await api.graph(state.selectedCampaign || undefined, state.graphLayer, state.selectedAttacker || undefined);
      setState(s => ({ ...s, graphData: data }));
    } catch (e) {
      console.error('refreshGraph failed', e);
    }
  }, [state.selectedCampaign, state.graphLayer, state.selectedAttacker]);

  const refreshAll = useCallback(async () => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const results = await Promise.allSettled([
        api.health(),
        api.overview({ time_range: state.globalTimeRange }),
        api.events({
          page: state.eventsPage,
          page_size: 50,
          campaign: state.selectedCampaign || undefined,
          attacker: state.selectedAttacker || undefined,
          technique: state.selectedTechnique || undefined,
          time_range: state.globalTimeRange,
        }),
        api.graph(state.selectedCampaign || undefined, state.graphLayer, state.selectedAttacker || undefined),
        api.campaigns({ time_range: state.globalTimeRange, attacker: state.selectedAttacker || undefined }),
        api.predictions(),
        api.recommendations(),
        api.attackers(),
      ]);

      const [healthRes, overviewRes, eventsRes, graphRes, campaignsRes,
             predictionsRes, recsRes, attackersRes] = results;

      const allFailed = results.every(r => r.status === 'rejected');

      setState(s => ({
        ...s,
        health: healthRes.status === 'fulfilled' ? healthRes.value : s.health,
        overview: overviewRes.status === 'fulfilled' ? overviewRes.value : s.overview,
        events: eventsRes.status === 'fulfilled' ? eventsRes.value.events : s.events,
        eventsTotal: eventsRes.status === 'fulfilled' ? eventsRes.value.total : s.eventsTotal,
        graphData: graphRes.status === 'fulfilled' ? graphRes.value : s.graphData,
        campaigns: campaignsRes.status === 'fulfilled' ? campaignsRes.value.campaigns : s.campaigns,
        predictions: predictionsRes.status === 'fulfilled' ? predictionsRes.value.predictions : s.predictions,
        recommendations: recsRes.status === 'fulfilled' && Array.isArray((recsRes.value as { recommendations: RecommendationGroup[] }).recommendations)
          ? (recsRes.value as { recommendations: RecommendationGroup[] }).recommendations : s.recommendations,
        attackers: attackersRes.status === 'fulfilled' ? attackersRes.value.attackers : s.attackers,
        loading: false,
        error: allFailed ? 'All API calls failed. Is dashboard_api.py running on port 5002?' : null,
      }));
    } catch (e) {
      setState(s => ({
        ...s,
        loading: false,
        error: e instanceof Error ? e.message : 'Unknown error',
      }));
    }
  }, [state.eventsPage, state.selectedCampaign, state.selectedAttacker, state.selectedTechnique, state.globalTimeRange]);

  // Initial load
  useEffect(() => { refreshAll(); }, []);

  // Re-fetch when filters change
  useEffect(() => {
    refreshAll();
  }, [state.selectedCampaign, state.selectedAttacker, state.selectedTechnique, state.eventsPage, state.globalTimeRange, state.graphLayer]);

  // Auto-refresh every 30s
  useEffect(() => {
    const interval = setInterval(refreshAll, 30000);
    return () => clearInterval(interval);
  }, [refreshAll]);

  return (
    <DashboardContext.Provider value={{
      ...state, selectCampaign, selectAttacker, selectTechnique, setGlobalTimeRange, setGraphLayer, resetDashboard,
      setEventsPage, setPathExplorerOpen, refreshAll, refreshEvents, refreshGraph, addEvents,
    }}>
      {children}
    </DashboardContext.Provider>
  );
}
