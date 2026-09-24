import { useEffect, useState } from 'react';
import { Activity, Database, Cpu, Radio, FileText, CheckCircle2, XCircle, AlertCircle } from 'lucide-react';
import { api } from '../services/api';
import type { SystemHealthResponse } from '../types';

const ICONS: Record<string, typeof Database> = {
  neo4j: Database,
  gnn: Cpu,
  xgboost_severity: Activity,
  misp: Radio,
  wazuh_listener: FileText,
};

const OK_STATUSES = new Set(['connected', 'available', 'recently_active', 'credential_configured']);
const WARN_STATUSES = new Set(['idle', 'disabled', 'not_trained', 'enabled_but_unavailable', 'credential_missing']);

/**
 * Top-level System Health page -- distinct from the minimal Neo4j-only
 * /api/health this dashboard already used for the top-nav CONNECTED
 * indicator. Reports each subsystem's own real state independently
 * (GET /api/system/health), never a single fabricated "all green".
 */
export function SystemHealthPage() {
  const [health, setHealth] = useState<SystemHealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = () => api.systemHealth().then(setHealth).catch((err) => setError(err.message || 'Failed to load system health'));
    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-4 md:p-6 bg-[#060a13]">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
            <Activity className="w-6 h-6 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">System Health</h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Independent, real status per subsystem. Refreshes every 15 seconds.
            </p>
          </div>
          {health && (
            <div className="ml-auto">
              <span
                className={`text-xs font-semibold px-3 py-1.5 rounded-md border ${
                  health.status === 'healthy'
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                    : 'bg-red-500/10 text-red-400 border-red-500/30'
                }`}
              >
                {health.status.toUpperCase()}
              </span>
            </div>
          )}
        </div>

        {error ? (
          <div className="glass-card p-8 flex items-center justify-center text-red-400 text-sm">{error}</div>
        ) : !health ? (
          <div className="glass-card p-8 flex items-center justify-center text-slate-500 text-sm animate-pulse">
            Loading system health...
          </div>
        ) : (
          <div className="grid md:grid-cols-2 gap-3">
            {Object.entries(health.subsystems).map(([name, info]) => {
              const Icon = ICONS[name] || Activity;
              const isOk = OK_STATUSES.has(info.status);
              const isWarn = WARN_STATUSES.has(info.status);
              const StatusIcon = isOk ? CheckCircle2 : isWarn ? AlertCircle : XCircle;
              const color = isOk ? 'text-emerald-400' : isWarn ? 'text-amber-400' : 'text-red-400';
              return (
                <div key={name} className="glass-card p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Icon className="w-4 h-4 text-slate-400" />
                    <span className="text-sm font-semibold text-slate-200 capitalize">
                      {name.replace(/_/g, ' ')}
                    </span>
                    <StatusIcon className={`w-4 h-4 ml-auto ${color}`} />
                  </div>
                  <div className={`text-xs font-mono ${color} mb-1`}>{info.status}</div>
                  {Object.entries(info)
                    .filter(([k]) => k !== 'status')
                    .map(([k, v]) => (
                      <div key={k} className="text-[10px] text-slate-500 truncate">
                        {k}: <span className="text-slate-400">{String(v)}</span>
                      </div>
                    ))}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
