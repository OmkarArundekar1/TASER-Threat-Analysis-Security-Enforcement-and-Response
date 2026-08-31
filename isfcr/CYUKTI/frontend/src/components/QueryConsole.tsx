import { useState, useCallback, useEffect, useRef } from 'react';
import { Terminal, Play, Save, Clock, Table as TableIcon, Network } from 'lucide-react';
import { api } from '../services/api';
import ForceGraph2D from 'react-force-graph-2d';
import type { QueryResult } from '../types';
import { useDashboard } from '../context/DashboardContext';
import { PanelWrapper } from './PanelWrapper';

const SAVED_QUERIES_KEY = 'watchdog_saved_queries';
const HISTORY_KEY = 'watchdog_query_history';

export function QueryConsole() {
  const { resetKey } = useDashboard();
  const [query, setQuery] = useState('MATCH (c:Campaign)-[r:LIKELY_NEXT]->(t:Technique)\nRETURN c.campaign_id, t.attack_id, r.confidence, r.generated_at\nLIMIT 10');
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'table' | 'graph'>('table');
  const [history, setHistory] = useState<string[]>([]);
  const [saved, setSaved] = useState<string[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const graphRef = useRef<HTMLDivElement>(null);
  const [graphSize, setGraphSize] = useState({ width: 600, height: 200 });

  useEffect(() => {
    setHistory(JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'));
    setSaved(JSON.parse(localStorage.getItem(SAVED_QUERIES_KEY) || '[]'));
  }, []);

  useEffect(() => {
    setQuery('');
    setResult(null);
    setError(null);
    setViewMode('table');
  }, [resetKey]);

  useEffect(() => {
    const el = graphRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setGraphSize({ width: el.clientWidth, height: el.clientHeight });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [viewMode]);

  const executeQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.query(query);
      setResult(res);
      const nextHistory = [query, ...history.filter((h) => h !== query)].slice(0, 20);
      setHistory(nextHistory);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(nextHistory));
      
      if (res.graph && res.graph.nodes && res.graph.nodes.length > 0) {
        setViewMode('graph');
      } else {
        setViewMode('table');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const saveQuery = () => {
    if (!query.trim()) return;
    const next = [...new Set([query, ...saved])].slice(0, 10);
    setSaved(next);
    localStorage.setItem(SAVED_QUERIES_KEY, JSON.stringify(next));
  };

  const drawNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const label = node.label || node.id;
    const fontSize = 10 / globalScale;
    ctx.font = `${fontSize}px Inter, monospace`;
    ctx.beginPath();
    
    let color = '#6366f1';
    switch (node.type) {
      case 'Attacker': color = '#ef4444'; break;
      case 'Campaign': color = '#f97316'; break;
      case 'Technique': color = '#a855f7'; break;
      case 'Host': color = '#3b82f6'; break;
      case 'AttackEvent': color = '#eab308'; break;
      case 'ThreatActor': color = '#dc2626'; break;
      case 'Malware': color = '#ec4899'; break;
      case 'Tool': color = '#8b5cf6'; break;
      case 'Mitigation': color = '#10b981'; break;
      case 'Tactic': color = '#14b8a6'; break;
    }

    ctx.arc(node.x, node.y, 5, 0, 2 * Math.PI, false);
    ctx.fillStyle = color;
    ctx.fill();
    if (globalScale > 0.8) {
      ctx.fillStyle = '#e2e8f0';
      ctx.textAlign = 'center';
      ctx.fillText(String(label).slice(0, 20), node.x, node.y + 12);
    }
  }, []);

  return (
    <PanelWrapper title="Query Console" icon={<Terminal className="w-4 h-4 text-indigo-400" />} className="h-full">
      <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
        <div className="flex items-center gap-2 px-3 py-2 border-b border-[#1e2d4a] bg-[#0a0e17] flex-none">
          <div className="relative">
            <button onClick={() => setShowHistory(!showHistory)} className="p-1.5 text-slate-400 hover:text-white rounded" title="Query History">
              <Clock className="w-4 h-4" />
            </button>
            {showHistory && (
              <div className="absolute left-0 top-full mt-1 w-80 max-h-40 overflow-auto bg-[#0c1220] border border-[#1e2d4a] rounded z-50 p-2 text-xs">
                <div className="text-slate-500 uppercase text-[10px] mb-1">History</div>
                {history.map((h, i) => (
                  <button key={i} onClick={() => { setQuery(h); setShowHistory(false); }} className="block w-full text-left truncate hover:bg-[#1a2540] p-1 rounded text-slate-300">{h.split('\n')[0]}</button>
                ))}
                <div className="text-slate-500 uppercase text-[10px] mt-2 mb-1">Saved</div>
                {saved.map((s, i) => (
                  <button key={i} onClick={() => { setQuery(s); setShowHistory(false); }} className="block w-full text-left truncate hover:bg-[#1a2540] p-1 rounded text-emerald-300">{s.split('\n')[0]}</button>
                ))}
              </div>
            )}
          </div>
          <button onClick={saveQuery} className="p-1.5 text-slate-400 hover:text-white rounded" title="Save Query"><Save className="w-4 h-4" /></button>
          <button onClick={executeQuery} disabled={loading} className="ml-auto flex items-center gap-1 bg-indigo-500 hover:bg-indigo-600 text-white px-3 py-1 rounded text-xs font-semibold">
            <Play className="w-3.5 h-3.5" /> RUN (Ctrl+Enter)
          </button>
        </div>

        <div className="p-2 border-b border-[#1e2d4a] bg-[#0c1220] flex-none">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) executeQuery(); }}
            className="cypher-editor h-16"
            spellCheck={false}
          />
          {error && <div className="mt-1 text-red-400 text-xs">{error}</div>}
        </div>

        <div className="flex border-b border-[#1e2d4a] flex-none">
          <button onClick={() => setViewMode('table')} className={`px-3 py-1.5 text-xs ${viewMode === 'table' ? 'text-indigo-400 border-b-2 border-indigo-400' : 'text-slate-400'}`}><TableIcon className="w-3 h-3 inline mr-1" />Table</button>
          {result?.graph?.nodes?.length ? (
            <button onClick={() => setViewMode('graph')} className={`px-3 py-1.5 text-xs ${viewMode === 'graph' ? 'text-indigo-400 border-b-2 border-indigo-400' : 'text-slate-400'}`}><Network className="w-3 h-3 inline mr-1" />Graph</button>
          ) : null}
          {result && <span className="ml-auto px-3 py-1.5 text-[10px] text-slate-500">{result.count} rows</span>}
        </div>

        <div className="flex-1 min-h-0 overflow-hidden bg-[#0a0e17]">
          {viewMode === 'table' && result && (
            <div className="h-full overflow-auto">
              <table className="soc-table w-full text-xs">
                <thead><tr>{result.columns.map((col) => <th key={col}>{col}</th>)}</tr></thead>
                <tbody>
                  {result.records.map((row, i) => (
                    <tr key={i}>
                      {result.columns.map((col) => {
                        const val = typeof row[col] === 'object' ? JSON.stringify(row[col]) : String(row[col]);
                        
                        // Detect if this looks like a Technique or Campaign ID to make it clickable
                        const isTechnique = val.startsWith('T') && val.length >= 5 && !val.includes(' ');
                        const isCampaign = val.includes('_') && val.split('_').length >= 2 && val.length > 10;
                        const { selectTechnique, selectCampaign } = useDashboard();
                        
                        return (
                          <td key={col} className="max-w-[200px] truncate">
                            {isTechnique ? (
                              <button onClick={() => selectTechnique(val)} className="text-indigo-400 hover:underline">{val}</button>
                            ) : isCampaign ? (
                              <button onClick={() => selectCampaign(val)} className="text-orange-400 hover:underline">{val}</button>
                            ) : (
                              val
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {viewMode === 'graph' && (
            <div ref={graphRef} className="h-full w-full">
              {result?.graph.nodes.length ? (
                <ForceGraph2D width={graphSize.width} height={graphSize.height} graphData={result.graph} nodeCanvasObject={drawNode} backgroundColor="#0a0e17" />
              ) : (
                <div className="h-full flex items-center justify-center text-slate-500 text-xs">Run a query returning nodes for graph view</div>
              )}
            </div>
          )}
          {!result && !loading && <div className="h-full flex items-center justify-center text-slate-500 text-xs">Execute Cypher to see results</div>}
        </div>
      </div>
    </PanelWrapper>
  );
}
