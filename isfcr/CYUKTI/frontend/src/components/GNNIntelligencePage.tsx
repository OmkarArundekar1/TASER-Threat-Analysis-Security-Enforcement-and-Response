import { useState, useEffect } from 'react';
import { GitBranch, Cpu, Layers, Hash, ShieldOff, Network } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';
import type { GNNStatus, EvidenceItem } from '../types';

/**
 * Full top-level page for the GNN topology-aware representation --
 * not a tab nested inside another panel. Sibling to the main SOC
 * dashboard (see App.tsx's activeView switch), reachable from
 * TopNavBar. Reuses the same /api/gnn/status and /api/gnn/topology/<id>
 * endpoints TopologyIntelligence.tsx (the smaller, in-panel version
 * still available under Intelligence Workspace -> Topology) already
 * consumes -- this page just gives the same real data much more room.
 */
export function GNNIntelligencePage() {
  const { campaigns, selectedCampaign, selectCampaign } = useDashboard();
  const [status, setStatus] = useState<GNNStatus | null>(null);
  const [neighbors, setNeighbors] = useState<EvidenceItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.gnnStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  useEffect(() => {
    if (!selectedCampaign || status?.gnn_available === false) {
      setNeighbors([]);
      return;
    }
    if (!status) return;
    setLoading(true);
    setError(null);
    api.gnnTopology(selectedCampaign, 12)
      .then((res) => setNeighbors(res.topology_neighbors || []))
      .catch((err) => setError(err.message || 'Topology lookup failed'))
      .finally(() => setLoading(false));
  }, [selectedCampaign, status]);

  const meta = status?.gnn_metadata;

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-4 md:p-6 bg-[#060a13]">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-cyan-500/10 border border-cyan-500/30">
            <GitBranch className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">GNN Topology Intelligence</h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Graph autoencoder embedding similarity -- structural, learned, independent of shared
              technique/attacker/host identity. See GNN_PRODUCTION_INTEGRATION.md.
            </p>
          </div>
          <div className="ml-auto">
            <span className={`text-xs font-semibold px-3 py-1.5 rounded-md border ${
              status?.gnn_available
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : 'bg-slate-800/50 text-slate-500 border-slate-700'
            }`}>
              {status?.gnn_available ? 'MODEL ACTIVE' : 'MODEL DISABLED'}
            </span>
          </div>
        </div>

        {/* Model stat cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard icon={Cpu} label="Architecture" value={meta?.architecture || '—'} small />
          <StatCard icon={Layers} label="Embedding Dim" value={meta ? `${meta.embedding_dim}` : '—'} />
          <StatCard icon={Hash} label="Encoder Layers" value={meta ? `${meta.num_layers}` : '—'} />
          <StatCard icon={Network} label="Model Version" value={status?.gnn_model_version || '—'} small mono />
        </div>

        {!status?.gnn_available ? (
          <div className="glass-card p-8 flex flex-col items-center text-center gap-3">
            <ShieldOff className="w-10 h-10 text-slate-600" />
            <p className="text-sm font-semibold text-slate-300">GNN Topology Engine is disabled</p>
            <p className="text-xs text-slate-500 max-w-md">
              Set <code className="text-cyan-400 bg-[#0a0e17] px-1.5 py-0.5 rounded">GNN_ENABLED=true</code> in
              the backend's <code className="text-cyan-400 bg-[#0a0e17] px-1.5 py-0.5 rounded">.env</code> and
              restart the API to activate the trained graph autoencoder. Every other CYUKTI mechanism --
              correlation, attribution, severity prediction -- runs identically either way; this is a purely
              additive signal, off by default.
            </p>
          </div>
        ) : (
          <>
            {/* Campaign picker */}
            <div className="glass-card p-3 flex items-center gap-3">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider shrink-0">
                Campaign
              </label>
              <select
                value={selectedCampaign || ''}
                onChange={(e) => selectCampaign(e.target.value || null)}
                className="flex-1 bg-[#0a0e17] text-sm text-slate-200 font-mono border border-[#1e2d4a] rounded px-2 py-1.5 outline-none focus:border-cyan-500"
              >
                <option value="">Select a campaign to embed...</option>
                {campaigns.map((c) => (
                  <option key={c.campaign_id} value={c.campaign_id}>
                    {c.campaign_label || c.campaign_id}
                  </option>
                ))}
              </select>
            </div>

            {/* Results */}
            {!selectedCampaign ? (
              <div className="glass-card p-8 flex flex-col items-center text-center gap-2 text-slate-500">
                <GitBranch className="w-8 h-8 opacity-20" />
                <p className="text-sm">
                  Select a campaign above -- its attack graph will be embedded by the trained encoder and
                  ranked against every other real campaign by learned structural similarity.
                </p>
              </div>
            ) : loading ? (
              <div className="glass-card p-12 flex items-center justify-center text-cyan-400 animate-pulse text-sm">
                Encoding attack graph...
              </div>
            ) : error ? (
              <div className="glass-card p-8 flex items-center justify-center text-red-400 text-sm">{error}</div>
            ) : neighbors.length === 0 ? (
              <div className="glass-card p-8 flex flex-col items-center text-center gap-2 text-slate-500">
                <ShieldOff className="w-8 h-8 opacity-30" />
                No structurally comparable campaigns found in real graph data yet.
              </div>
            ) : (
              <div className="space-y-2">
                <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1">
                  Topology-Nearest Campaigns ({neighbors.length})
                </h2>
                <div className="grid md:grid-cols-2 gap-3">
                  {neighbors.map((n, idx) => {
                    const similarity = (n.content.topology_similarity as number | undefined) ?? n.relevance;
                    const pct = Math.max(0, Math.min(100, similarity * 100));
                    const campaignId =
                      (n.content.campaign_id as string | undefined) ||
                      (n.content.candidate_campaign_id as string | undefined) ||
                      n.source_id;
                    return (
                      <div
                        key={n.evidence_id}
                        onClick={() => selectCampaign(campaignId)}
                        className="glass-card p-4 cursor-pointer hover:border-cyan-500/50 transition-colors border border-transparent"
                      >
                        <div className="flex justify-between items-center mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-mono text-slate-600">#{idx + 1}</span>
                            <span className="font-mono text-cyan-400 font-bold hover:underline">{campaignId}</span>
                          </div>
                          <div className="text-2xl font-mono font-bold text-white">{pct.toFixed(1)}%</div>
                        </div>
                        <div className="h-2 rounded-full bg-[#131c2e] overflow-hidden mb-2">
                          <div
                            className="h-full bg-gradient-to-r from-cyan-600 to-cyan-400 rounded-full transition-all"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        {typeof n.content.attacker === 'string' && (
                          <div className="text-[11px] text-slate-500">
                            Attacker: <span className="text-red-400 font-mono">{n.content.attacker}</span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function StatCard({
  icon: Icon, label, value, small, mono,
}: { icon: typeof Cpu; label: string; value: string; small?: boolean; mono?: boolean }) {
  return (
    <div className="glass-card px-3 py-3 flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5 text-slate-500">
        <Icon className="w-3.5 h-3.5" />
        <span className="text-[10px] uppercase tracking-wider font-semibold">{label}</span>
      </div>
      <span className={`${small ? 'text-xs' : 'text-lg'} ${mono ? 'font-mono' : ''} font-bold text-white truncate`} title={value}>
        {value}
      </span>
    </div>
  );
}
