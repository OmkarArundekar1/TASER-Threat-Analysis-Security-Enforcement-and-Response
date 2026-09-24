import { useState, useEffect } from 'react';
import { GitBranch, ShieldOff, Cpu } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';
import type { GNNStatus, EvidenceItem } from '../types';

/**
 * Dedicated GNN showcase panel -- distinct from ThreatCorrelation's
 * technique/attacker/host overlap (which also carries a small
 * gnn_topology_similarity annotation per candidate, computed from the
 * same source, but this panel is the actual home for the signal: a
 * real graph autoencoder embedding-similarity ranking, sourced from
 * rag/gnn_topology_retriever.py via /api/gnn/topology/<id>, with model
 * metadata surfaced so it's clear this is a learned representation, not
 * a hand-written formula.
 */
export function TopologyIntelligence() {
  const { selectedCampaign, selectCampaign } = useDashboard();
  const [status, setStatus] = useState<GNNStatus | null>(null);
  const [neighbors, setNeighbors] = useState<EvidenceItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.gnnStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  useEffect(() => {
    if (!selectedCampaign || status?.gnn_available === false) {
      // Skip the request entirely once status is known to be disabled --
      // the backend route itself handles this fail-safe either way, but
      // there's no reason to make the round trip when we already know
      // the answer.
      setNeighbors([]);
      return;
    }
    if (!status) return; // wait for gnnStatus() to resolve first
    setLoading(true);
    setError(null);
    api.gnnTopology(selectedCampaign, 8)
      .then((res) => setNeighbors(res.topology_neighbors || []))
      .catch((err) => setError(err.message || 'Topology lookup failed'))
      .finally(() => setLoading(false));
  }, [selectedCampaign, status]);

  if (status && !status.gnn_available) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-4 text-center gap-2">
        <ShieldOff className="w-8 h-8 opacity-20" />
        <p className="text-sm font-semibold text-slate-400">GNN Topology Engine Disabled</p>
        <p className="text-xs max-w-[260px]">
          Set <code className="text-cyan-400">GNN_ENABLED=true</code> in the backend environment to
          activate the graph autoencoder. Every other CYUKTI mechanism (correlation, attribution,
          severity) is completely unaffected either way.
        </p>
      </div>
    );
  }

  if (!selectedCampaign) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-4 text-center">
        <GitBranch className="w-8 h-8 mb-3 opacity-20" />
        <p className="text-sm font-semibold text-slate-400">Select a Campaign</p>
        <p className="text-xs mt-2 max-w-[240px]">
          The graph neural network will embed this campaign's attack graph and rank every other real
          campaign by learned structural similarity -- independent of shared technique, attacker, or
          host identity.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden animate-slide-in bg-[#060a13]">
      {status?.gnn_metadata && (
        <div className="shrink-0 px-3 py-2 border-b border-[#1e2d4a] bg-[#0a0e17] flex items-center justify-between text-[10px]">
          <div className="flex items-center gap-1.5 text-slate-400">
            <Cpu className="w-3 h-3 text-cyan-400" />
            <span>{status.gnn_metadata.architecture}</span>
            <span className="text-slate-600">·</span>
            <span>{status.gnn_metadata.embedding_dim}-dim embedding</span>
          </div>
          <span className="font-mono text-slate-600 truncate max-w-[140px]" title={status.gnn_model_version || ''}>
            {status.gnn_model_version}
          </span>
        </div>
      )}

      <div className="flex-1 overflow-auto custom-scrollbar p-3 space-y-2">
        {loading ? (
          <div className="flex items-center justify-center h-32 text-cyan-400 animate-pulse text-xs">
            Encoding attack graph...
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-32 text-red-400 text-xs">{error}</div>
        ) : neighbors.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-slate-500 text-xs gap-2">
            <ShieldOff className="w-6 h-6 opacity-50" />
            No structurally comparable campaigns found in real graph data yet.
          </div>
        ) : (
          neighbors.map((n, idx) => {
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
                className="bg-[#0a0e17] border border-[#1e2d4a] rounded-lg p-3 hover:border-cyan-500/50 transition-colors cursor-pointer"
              >
                <div className="flex justify-between items-center mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-[9px] font-mono text-slate-600">#{idx + 1}</span>
                    <span className="font-mono text-cyan-400 font-bold text-sm hover:underline">
                      {campaignId}
                    </span>
                  </div>
                  <div className="text-xl font-mono font-bold text-white">{pct.toFixed(1)}%</div>
                </div>
                <div className="h-1.5 rounded-full bg-[#131c2e] overflow-hidden mb-2">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-600 to-cyan-400 rounded-full transition-all"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-slate-500">
                  {typeof n.content.attacker === 'string' && (
                    <span>Attacker: <span className="text-red-400 font-mono">{n.content.attacker}</span></span>
                  )}
                  <span className="text-slate-600 truncate max-w-[160px]" title={n.provenance}>
                    learned structural embedding
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
