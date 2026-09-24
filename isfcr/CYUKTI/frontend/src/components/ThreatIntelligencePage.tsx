import { useEffect, useState } from 'react';
import { Radio, Link2, ShieldOff, CheckCircle2, XCircle } from 'lucide-react';
import { api } from '../services/api';
import type { MISPStatus } from '../types';

/**
 * Top-level MISP/CTI page -- realtime_socgraph.py already instantiates
 * CTIPublisher/MISPSync and attempts to sync every resolved campaign
 * to MISP (see FULL_SYSTEM_INTEGRATION_AUDIT.md Section 2), but that
 * state had zero dashboard visibility before /api/misp/status. Never
 * publishes anything from here -- read-only status surface.
 */
export function ThreatIntelligencePage() {
  const [status, setStatus] = useState<MISPStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.mispStatus().then(setStatus).catch((err) => setError(err.message || 'Failed to load MISP status'));
  }, []);

  const cachedEntries = status ? Object.entries(status.campaign_event_map) : [];

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-4 md:p-6 bg-[#060a13]">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-orange-500/10 border border-orange-500/30">
            <Radio className="w-6 h-6 text-orange-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">Threat Intelligence (MISP/CTI)</h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Live status of the CTI publication pipeline (cti_publisher.py / misp_sync.py) --
              already wired into every resolved campaign, surfaced here for the first time.
            </p>
          </div>
          {status && (
            <div className="ml-auto">
              <span
                className={`text-xs font-semibold px-3 py-1.5 rounded-md border flex items-center gap-1.5 ${
                  status.authenticated
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                    : 'bg-slate-800/50 text-slate-500 border-slate-700'
                }`}
              >
                {status.authenticated ? <CheckCircle2 className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
                {status.authenticated ? 'CONNECTED' : 'NOT CONNECTED'}
              </span>
            </div>
          )}
        </div>

        {error ? (
          <div className="glass-card p-8 flex items-center justify-center text-red-400 text-sm">{error}</div>
        ) : !status ? (
          <div className="glass-card p-8 flex items-center justify-center text-slate-500 text-sm animate-pulse">
            Loading MISP status...
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <StatCard label="MISP URL" value={status.misp_url} mono small />
              <StatCard label="Credential Configured" value={status.credential_configured ? 'Yes' : 'No'} />
              <StatCard label="Cached Campaigns" value={`${status.cached_campaigns}`} />
            </div>

            <div className="glass-card p-4">
              <p className="text-sm text-slate-300">{status.status_message}</p>
              {!status.credential_configured && (
                <p className="text-xs text-slate-500 mt-2">
                  Set <code className="text-cyan-400 bg-[#0a0e17] px-1.5 py-0.5 rounded">MISP_API_KEY</code> in
                  the backend's <code className="text-cyan-400 bg-[#0a0e17] px-1.5 py-0.5 rounded">.env</code>{' '}
                  to enable authenticated publishing. Nothing here is fabricated -- this environment has no
                  credential configured, so publication has genuinely never been exercised end-to-end.
                </p>
              )}
            </div>

            {cachedEntries.length === 0 ? (
              <div className="glass-card p-8 flex flex-col items-center text-center gap-2 text-slate-500">
                <ShieldOff className="w-8 h-8 opacity-30" />
                No campaigns have been cached against a MISP event yet.
              </div>
            ) : (
              <div className="space-y-2">
                <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1">
                  Campaign → MISP Event Mapping
                </h2>
                <div className="glass-card divide-y divide-[#1e2d4a]">
                  {cachedEntries.map(([campaignId, eventId]) => (
                    <div key={campaignId} className="flex items-center justify-between px-4 py-2.5">
                      <span className="font-mono text-xs text-cyan-400">{campaignId}</span>
                      <span className="flex items-center gap-1.5 text-xs text-slate-400">
                        <Link2 className="w-3 h-3" /> Event #{eventId}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, small, mono }: { label: string; value: string; small?: boolean; mono?: boolean }) {
  return (
    <div className="glass-card px-3 py-3 flex flex-col gap-1.5">
      <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-500">{label}</span>
      <span className={`${small ? 'text-xs' : 'text-lg'} ${mono ? 'font-mono' : ''} font-bold text-white truncate`} title={value}>
        {value}
      </span>
    </div>
  );
}
