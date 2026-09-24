import { useEffect, useState } from 'react';
import { ScrollText, RefreshCw } from 'lucide-react';
import { api } from '../services/api';
import type { AuditLogEntry } from '../types';

const LEVEL_COLOR: Record<string, string> = {
  INFO: 'text-slate-400',
  WARNING: 'text-amber-400',
  ERROR: 'text-red-400',
  DEBUG: 'text-cyan-400',
};

/**
 * Top-level Audit Log page -- tails the real
 * listener/wazuh_listener.py log file (backend/logs/prerana_listener.log),
 * previously a plain file with zero API/dashboard exposure. Read-only.
 */
export function AuditLogPage() {
  const [entries, setEntries] = useState<AuditLogEntry[]>([]);
  const [totalLines, setTotalLines] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = () => {
    setLoading(true);
    api
      .auditLogs(200)
      .then((res) => {
        setEntries(res.entries);
        setTotalLines(res.total_lines);
        setError(null);
      })
      .catch((err) => setError(err.message || 'Failed to load audit log'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-4 md:p-6 bg-[#060a13]">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-slate-500/10 border border-slate-500/30">
            <ScrollText className="w-6 h-6 text-slate-300" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">Audit Log</h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Real listener activity log (listener/wazuh_listener.py) -- {totalLines.toLocaleString()} total lines,
              most recent {entries.length} shown.
            </p>
          </div>
          <button
            onClick={load}
            className="ml-auto flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-md border bg-slate-800/50 text-slate-300 border-slate-700 hover:bg-slate-700/50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} /> Refresh
          </button>
        </div>

        {error ? (
          <div className="glass-card p-8 flex items-center justify-center text-red-400 text-sm">{error}</div>
        ) : entries.length === 0 ? (
          <div className="glass-card p-8 flex items-center justify-center text-slate-500 text-sm">
            No log entries found yet.
          </div>
        ) : (
          <div className="glass-card p-3 font-mono text-[11px] max-h-[70vh] overflow-y-auto custom-scrollbar space-y-0.5">
            {entries.map((entry, idx) => (
              <div key={idx} className="flex gap-2 py-0.5 border-b border-[#131c2e] last:border-0">
                <span className="text-slate-600 shrink-0">{entry.timestamp || '—'}</span>
                <span className={`shrink-0 w-14 ${entry.level ? LEVEL_COLOR[entry.level] || 'text-slate-400' : 'text-slate-500'}`}>
                  {entry.level || ''}
                </span>
                <span className="text-slate-300 break-all">{entry.message}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
