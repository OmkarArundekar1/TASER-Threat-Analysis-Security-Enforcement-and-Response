import { Shield, Clock, LayoutDashboard, GitBranch, Zap, Radio, Activity, ScrollText, Bot, ShieldAlert, ChevronDown, MoreHorizontal } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { useEffect, useRef, useState } from 'react';

export type TopLevelView = 'dashboard' | 'incident' | 'gnn' | 'prediction' | 'threat-intel' | 'system' | 'audit' | 'soar';

// Primary navigation: the small set of views an analyst actually works
// from day to day. Everything else (research/engineering surfaces) is
// one click away under "More" rather than competing for top-level
// attention -- see the UX-redesign phase's explicit instruction not to
// expose every backend subsystem as a top-level nav item.
const PRIMARY_VIEWS: { id: TopLevelView; label: string; icon: typeof LayoutDashboard; activeClass: string }[] = [
  { id: 'dashboard', label: 'Overview', icon: LayoutDashboard, activeClass: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40' },
  { id: 'incident', label: 'Incidents', icon: ShieldAlert, activeClass: 'bg-red-500/20 text-red-300 border-red-500/40' },
  { id: 'soar', label: 'Response', icon: Bot, activeClass: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40' },
  { id: 'threat-intel', label: 'Threat Intel', icon: Radio, activeClass: 'bg-orange-500/20 text-orange-300 border-orange-500/40' },
];

const MORE_VIEWS: { id: TopLevelView; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'gnn', label: 'GNN Intelligence', icon: GitBranch },
  { id: 'prediction', label: 'Prediction', icon: Zap },
  { id: 'system', label: 'System Health', icon: Activity },
  { id: 'audit', label: 'Audit Log', icon: ScrollText },
];

interface TopNavBarProps {
  activeView: TopLevelView;
  onChangeView: (view: TopLevelView) => void;
}

function MoreMenu({ activeView, onChangeView }: TopNavBarProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const isActive = MORE_VIEWS.some((v) => v.id === activeView);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', onClickOutside);
    return () => document.removeEventListener('mousedown', onClickOutside);
  }, []);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        aria-haspopup="menu"
        aria-expanded={open}
        className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-semibold transition-colors whitespace-nowrap ${
          isActive ? 'bg-slate-500/20 text-slate-200 border border-slate-500/40' : 'text-slate-400 hover:text-slate-200 border border-transparent'
        }`}
      >
        <MoreHorizontal className="w-3.5 h-3.5" /> More <ChevronDown className="w-3 h-3" />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute left-0 top-full mt-1 w-48 bg-[#0c1220] border border-[#1e2d4a] rounded-md shadow-xl py-1 z-50"
        >
          {MORE_VIEWS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              role="menuitem"
              onClick={() => { onChangeView(id); setOpen(false); }}
              className={`w-full flex items-center gap-2 px-3 py-1.5 text-xs text-left transition-colors ${
                activeView === id ? 'text-cyan-300 bg-cyan-500/10' : 'text-slate-300 hover:bg-white/5'
              }`}
            >
              <Icon className="w-3.5 h-3.5" /> {label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function TopNavBar({ activeView, onChangeView }: TopNavBarProps) {
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

        <div className="flex items-center gap-1 bg-[#131c2e] border border-[#1e2d4a] rounded-md p-0.5 ml-2">
          {PRIMARY_VIEWS.map(({ id, label, icon: Icon, activeClass }) => (
            <button
              key={id}
              onClick={() => onChangeView(id)}
              className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-semibold transition-colors whitespace-nowrap ${
                activeView === id ? `${activeClass} border` : 'text-slate-400 hover:text-slate-200 border border-transparent'
              }`}
            >
              <Icon className="w-3.5 h-3.5" /> {label}
            </button>
          ))}
          <MoreMenu activeView={activeView} onChangeView={onChangeView} />
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
