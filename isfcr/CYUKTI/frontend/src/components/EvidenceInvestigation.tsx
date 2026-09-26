import { useState } from 'react';
import { Search, PlayCircle, Gauge, FileSearch, AlertCircle } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';
import type { EvidenceItem, InvestigationResult, MitreSearchResult, SeverityPrediction } from '../types';

const SOURCE_COLOR: Record<string, string> = {
  mitre: 'border-indigo-500/30 bg-indigo-500/5 text-indigo-400',
  cti: 'border-purple-500/30 bg-purple-500/5 text-purple-400',
  siem: 'border-yellow-500/30 bg-yellow-500/5 text-yellow-400',
  graph: 'border-blue-500/30 bg-blue-500/5 text-blue-400',
  campaign_history: 'border-emerald-500/30 bg-emerald-500/5 text-emerald-400',
  attribution: 'border-red-500/30 bg-red-500/5 text-red-400',
  gnn_topology: 'border-cyan-500/30 bg-cyan-500/5 text-cyan-400',
};

function EvidenceCard({ e }: { e: EvidenceItem }) {
  const colorClass = SOURCE_COLOR[e.source] || 'border-slate-500/30 bg-slate-500/5 text-slate-400';
  // Populated by both rag/gnn_topology_retriever.py (source: gnn_topology)
  // and threat_attribution_engine.py's additive field on ATTRIBUTION
  // evidence (see GNN_PRODUCTION_INTEGRATION.md) -- same content key,
  // same meaning (GNN embedding cosine similarity), two different
  // sources. Not present (undefined) when GNN is disabled/unavailable.
  const topologySimilarity = e.content.topology_similarity;
  const matchedCampaignId =
    (e.content.campaign_id as string | undefined) ||
    (e.content.candidate_campaign_id as string | undefined);

  return (
    <div className={`border rounded-md p-2 text-xs ${colorClass}`}>
      <div className="flex justify-between items-center mb-1">
        <span className="uppercase tracking-wider text-[10px] font-bold">{e.source}</span>
        <span className="font-mono text-[10px] text-slate-400">
          conf {e.confidence.toFixed(2)} · rel {e.relevance.toFixed(2)}
        </span>
      </div>
      <div className="text-slate-300 truncate" title={e.provenance}>{e.provenance}</div>
      {matchedCampaignId && (
        <div className="text-[10px] text-slate-400 mt-1 font-mono truncate">
          matched: {matchedCampaignId}
        </div>
      )}
      {typeof topologySimilarity === 'number' && (
        <div className="text-[10px] text-cyan-400 mt-1 font-mono">
          topology similarity: {(topologySimilarity * 100).toFixed(1)}%
        </div>
      )}
      {e.derived_from.length > 0 && (
        <div className="text-[10px] text-slate-500 mt-1" title={e.derived_from.join(', ')}>
          derived from {e.derived_from.length} item{e.derived_from.length > 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
}

const SOURCE_DOT: Record<string, string> = {
  mitre: 'bg-indigo-400',
  cti: 'bg-purple-400',
  siem: 'bg-yellow-400',
  graph: 'bg-blue-400',
  campaign_history: 'bg-emerald-400',
  attribution: 'bg-red-400',
  gnn_topology: 'bg-cyan-400',
};

/**
 * Makes CYUKTI's Multi-RAG design (multiple independent retrieval
 * sources feeding one evidence store -- MITRE semantic search,
 * campaign-narrative TF-IDF, GNN topology embeddings, plus SIEM/CTI/
 * attribution) visible at a glance, rather than only discoverable by
 * reading through every individual evidence card's badge.
 */
function MultiRagSourceBar({ evidence }: { evidence: EvidenceItem[] }) {
  if (evidence.length === 0) return null;
  const counts = new Map<string, number>();
  for (const e of evidence) counts.set(e.source, (counts.get(e.source) || 0) + 1);
  const sources = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-1 pb-1 text-[10px] text-slate-400">
      <span className="text-slate-600 uppercase tracking-wider">Multi-RAG:</span>
      {sources.map(([source, count]) => (
        <span key={source} className="flex items-center gap-1">
          <span className={`w-1.5 h-1.5 rounded-full ${SOURCE_DOT[source] || 'bg-slate-400'}`} />
          {source} <span className="font-mono text-slate-500">{count}</span>
        </span>
      ))}
    </div>
  );
}

export function EvidenceInvestigation() {
  const { selectedCampaign } = useDashboard();
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<InvestigationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<SeverityPrediction | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<MitreSearchResult | null>(null);

  const runInvestigation = async () => {
    if (!selectedCampaign) return;
    setRunning(true);
    setError(null);
    try {
      const res = await api.investigate(selectedCampaign);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Investigation failed');
      setResult(null);
    } finally {
      setRunning(false);
    }
  };

  const predictSeverity = async () => {
    if (!selectedCampaign) return;
    setPredicting(true);
    setPredictionError(null);
    try {
      const res = await api.predictSeverity(selectedCampaign);
      setPrediction(res);
    } catch (err) {
      setPredictionError(err instanceof Error ? err.message : 'Prediction unavailable');
      setPrediction(null);
    } finally {
      setPredicting(false);
    }
  };

  const runSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const res = await api.ragMitreSearch(searchQuery, 5);
      setSearchResult(res);
    } catch (err) {
      console.error('MITRE semantic search failed', err);
      setSearchResult(null);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden animate-slide-in bg-[#060a13]">
      {/* MITRE semantic search — independent of campaign selection */}
      <div className="p-2 border-b border-[#1e2d4a] flex-none">
        <div className="flex items-center gap-1.5 mb-1.5">
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') runSearch(); }}
            placeholder="Describe observed behavior (e.g. repeated password guessing)"
            className="flex-1 bg-[#0c1220] border border-[#1e2d4a] rounded px-2 py-1 text-xs text-slate-200 placeholder:text-slate-600"
          />
          <button
            onClick={runSearch}
            disabled={searching}
            className="p-1.5 bg-indigo-500 hover:bg-indigo-600 text-white rounded"
            title="Semantic search over MITRE ATT&CK"
          >
            <Search className="w-3.5 h-3.5" />
          </button>
        </div>
        {searchResult && (
          <div className="space-y-1 max-h-28 overflow-auto custom-scrollbar">
            {searchResult.results.length === 0 && (
              <div className="text-[10px] text-slate-500">No techniques matched.</div>
            )}
            {searchResult.results.map((r) => (
              <div key={r.evidence_id} className="flex justify-between text-[11px] px-1.5 py-1 bg-[#0c1220] rounded border border-[#1e2d4a]">
                <span className="text-slate-300 truncate">{(r.content.name as string) || r.source_id}</span>
                <span className="font-mono text-indigo-400">{r.relevance.toFixed(2)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {!selectedCampaign ? (
        <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-4 text-center">
          <FileSearch className="w-8 h-8 mb-3 opacity-20" />
          <p className="text-sm font-semibold text-slate-400">Select a Campaign</p>
          <p className="text-xs mt-2 max-w-[220px]">
            Run the evidence-aware investigation loop, or a severity prediction, against a selected campaign.
          </p>
        </div>
      ) : (
        <div className="flex-1 overflow-auto custom-scrollbar p-2 space-y-2">
          <div className="flex gap-1.5">
            <button
              onClick={runInvestigation}
              disabled={running}
              className="flex-1 flex items-center justify-center gap-1.5 bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 text-white px-2 py-1.5 rounded text-xs font-semibold"
            >
              <PlayCircle className="w-3.5 h-3.5" /> {running ? 'Investigating…' : 'Run Investigation'}
            </button>
            <button
              onClick={predictSeverity}
              disabled={predicting}
              className="flex-1 flex items-center justify-center gap-1.5 bg-[#1e2d4a] hover:bg-[#28395c] disabled:opacity-50 text-slate-200 px-2 py-1.5 rounded text-xs font-semibold"
            >
              <Gauge className="w-3.5 h-3.5" /> {predicting ? 'Predicting…' : 'Predict Severity'}
            </button>
          </div>

          {error && (
            <div className="flex items-center gap-1.5 text-red-400 text-[11px]">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" /> {error}
            </div>
          )}
          {predictionError && (
            <div className="flex items-center gap-1.5 text-slate-500 text-[11px]">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" /> {predictionError}
            </div>
          )}
          {prediction && (
            <div className="border border-[#1e2d4a] rounded p-2 text-xs space-y-1">
              <div>
                <span className="text-slate-400">Predicted severity: </span>
                <span className="font-bold text-slate-100">{prediction.label}</span>
                <span className="text-slate-500"> ({(prediction.confidence * 100).toFixed(1)}%)</span>
              </div>
              {prediction.top_k.length > 1 && (
                <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-slate-500">
                  {prediction.top_k.map((k) => (
                    <span key={k.label}>{k.label} {(k.probability * 100).toFixed(1)}%</span>
                  ))}
                </div>
              )}
              <div className="text-[10px] text-slate-600" title={`trained ${prediction.model_metadata.trained_at}`}>
                model schema {prediction.model_metadata.feature_schema_version}
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-400">Investigation confidence</span>
                <span className="font-mono font-bold text-slate-100">
                  {result.final_confidence !== null ? result.final_confidence.toFixed(2) : '—'}
                </span>
              </div>
              {result.final_model_probabilities && (
                <div className="text-[11px] space-y-0.5">
                  <div className="text-slate-500 uppercase tracking-wider text-[10px]">Model verdict (severity)</div>
                  {Object.entries(result.final_model_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, prob]) => (
                      <div key={cls} className="flex justify-between px-1.5">
                        <span className="text-slate-300">{cls}</span>
                        <span className="font-mono text-slate-400">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>
              )}
              <div className="text-[11px] text-slate-500 italic">{result.stopping_reason}</div>

              <div className="space-y-1">
                {result.steps.map((s) => (
                  <div key={s.step_index} className="text-[11px] px-1.5 py-1 bg-[#0c1220] rounded border border-[#1e2d4a]">
                    <div className="flex justify-between">
                      <span className="text-slate-300">#{s.step_index} {s.action_taken}</span>
                      <span className="font-mono text-slate-400">
                        conf→{s.investigation_confidence.toFixed(2)} unc→{s.uncertainty.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-500 text-[10px] mt-0.5">
                      <span>reliability={s.evidence_reliability.toFixed(2)} coverage={s.evidence_coverage.toFixed(2)}</span>
                      {s.model_confidence !== null && (
                        <span>model={s.model_confidence.toFixed(2)} (unc={s.model_uncertainty?.toFixed(2)})</span>
                      )}
                    </div>
                    <div className="text-slate-600 text-[10px] mt-0.5 italic truncate" title={s.why_selected}>
                      {s.why_selected}
                    </div>
                    {s.candidate_hypotheses.length > 1 && (
                      <div className="flex flex-wrap gap-x-2 text-[10px] text-slate-500 mt-0.5">
                        {s.candidate_hypotheses.map(([label, prob]) => (
                          <span key={label}>{label} {(prob * 100).toFixed(0)}%</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="text-[10px] uppercase tracking-wider text-slate-500 pt-1">
                Evidence ({result.total_evidence})
              </div>
              <MultiRagSourceBar evidence={result.evidence} />
              <div className="space-y-1">
                {result.evidence.map((e) => <EvidenceCard key={e.evidence_id} e={e} />)}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
