import { Shield, Clock } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { useEffect, useState } from 'react';

export function TopNavBar() {
  const { health, overview, globalTimeRange, setGlobalTimeRange, resetDashboard, selectedCampaign, selectedAttacker, selectedTechnique } = useDashboard();
  const [time, setTime] = useState('');

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toLocaleTimeString('en-US', { hour12: false }));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const isHealthy = health?.status === 'healthy';
  const hasActiveFilters = selectedCampaign || selectedAttacker || selectedTechnique;

  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-[#1e2d4a] bg-[#0c1220]/80 backdrop-blur-md sticky top-0 z-50">
      <div className="flex items-center gap-4">
        <Shield className="w-6 h-6 text-indigo-500" />
        <div>
          <h1 className="text-lg font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent leading-none mb-1">
            WatchDog SOC
          </h1>
          <div className="flex items-center gap-2 text-[10px] text-slate-400 font-mono leading-none">
            <span>Prerana Engine</span>
            <span className={isHealthy ? 'text-green-400' : 'text-red-400'}>
              {isHealthy ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button 
          onClick={resetDashboard}
          disabled={!hasActiveFilters}
          className={`text-xs font-semibold px-3 py-1 rounded-md transition-colors border ${
            hasActiveFilters 
              ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20 border-red-500/30' 
              : 'bg-slate-800/50 text-slate-500 border-slate-700 cursor-not-allowed'
          }`}
        >
          RESET DASHBOARD
        </button>

        <div className="flex items-center gap-2">
          <select 
            value={globalTimeRange}
            onChange={(e) => setGlobalTimeRange(e.target.value)}
            className="bg-[#131c2e] text-xs text-slate-300 font-mono border border-[#1e2d4a] rounded px-2 py-1 outline-none focus:border-indigo-500"
          >
            <option value="15m">Last 15 minutes</option>
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="all">All Time</option>
          </select>
        </div>

        <div className="flex gap-2">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#131c2e] border border-[#1e2d4a]">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Camp</span>
            <span className="text-sm font-mono font-bold text-orange-400">{overview?.active_campaigns || 0}</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#131c2e] border border-[#1e2d4a]">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Atk</span>
            <span className="text-sm font-mono font-bold text-red-400">{overview?.unique_attackers || 0}</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#131c2e] border border-[#1e2d4a]">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Evt</span>
            <span className="text-sm font-mono font-bold text-blue-400">{(overview?.total_events || 0).toLocaleString()}</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#131c2e] border border-[#1e2d4a]">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Nodes</span>
            <span className="text-sm font-mono font-bold text-fuchsia-400">{useDashboard().graphData?.nodes?.length || 0}</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#131c2e] border border-[#1e2d4a]">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Edges</span>
            <span className="text-sm font-mono font-bold text-indigo-400">{useDashboard().graphData?.links?.length || 0}</span>
          </div>
        </div>

        <div className="h-6 w-px bg-[#1e2d4a]"></div>

        <div className="flex items-center gap-2 text-slate-300 font-mono text-xs bg-[#131c2e] px-2 py-1 rounded-md border border-[#1e2d4a]">
          <Clock className="w-3 h-3 text-cyan-400" />
          {time}
        </div>
      </div>
    </div>
  );
}
