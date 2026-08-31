import React, { useState, useEffect } from 'react';
import { Play, Pause, Search, Filter, ChevronDown, ChevronRight, Activity, AlertTriangle, FileJson, ShieldCheck } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';
import type { AlertEvent } from '../types';

export function LiveEventsFeed() {
  const { events, eventsTotal, eventsPage, setEventsPage, refreshEvents, resetKey } = useDashboard();
  const [isPaused, setIsPaused] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedEventId, setExpandedEventId] = useState<string | null>(null);
  const [eventDetail, setEventDetail] = useState<AlertEvent | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  useEffect(() => {
    setSearchTerm('');
    setExpandedEventId(null);
    setEventDetail(null);
    setIsPaused(false);
  }, [resetKey]);
  
  // Columns config
  const [columns, setColumns] = useState({
    timestamp: true,
    attacker: true,
    victim: true,
    technique_id: true,
    technique: true,
    stage: true,
    severity: true,
    campaign: true,
  });

  const [showColConfig, setShowColConfig] = useState(false);

  const getSeverityClass = (severity: string) => {
    switch (severity) {
      case 'CRITICAL': return 'bg-red-500/20 text-red-400 border border-red-500/30';
      case 'HIGH': return 'bg-orange-500/20 text-orange-400 border border-orange-500/30';
      case 'MEDIUM': return 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30';
      case 'LOW': return 'bg-blue-500/20 text-blue-400 border border-blue-500/30';
      default: return 'bg-slate-800 text-slate-300 border border-slate-600';
    }
  };

  const formatDate = (isoString: string) => {
    const d = new Date(isoString);
    return `${d.toLocaleDateString()} ${d.toLocaleTimeString('en-US', { hour12: false })}`;
  };

  const handleRowClick = async (event: AlertEvent) => {
    // We use timestamp + attacker + victim + technique as a pseudo-ID if id is missing
    const id = event.id || `${event.timestamp}-${event.attacker_ip}-${event.technique_id}`;
    if (expandedEventId === id) {
      setExpandedEventId(null);
      setEventDetail(null);
      return;
    }
    
    setExpandedEventId(id);
    if (event.id) {
      setLoadingDetail(true);
      try {
        const detail = await api.eventDetail(event.id);
        setEventDetail(detail);
      } catch (e) {
        console.error("Failed to load event details", e);
        setEventDetail(event); // fallback to basic details
      } finally {
        setLoadingDetail(false);
      }
    } else {
      setEventDetail(event);
    }
  };

  const toggleColumn = (col: keyof typeof columns) => {
    setColumns(prev => ({ ...prev, [col]: !prev[col] }));
  };

  const filteredEvents = events.filter(e => 
    e.technique_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    e.technique_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    e.attacker_ip.includes(searchTerm) ||
    e.victim_ip.includes(searchTerm)
  );

  return (
    <div className="glass-card flex flex-col h-full overflow-hidden relative">
      <div className="p-4 border-b border-[#1e2d4a] flex items-center justify-between bg-[#131c2e]/50">
        <h2 className="text-sm font-semibold uppercase tracking-wider flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isPaused ? 'bg-amber-500' : 'bg-green-500 animate-pulse-dot'}`}></div>
          Live SOC Feed
        </h2>
        
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-2.5 top-2 text-slate-400" />
            <input 
              type="text" 
              placeholder="Filter events locally..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-[#0a0e17] border border-[#1e2d4a] rounded-md pl-8 pr-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 w-48 transition-colors"
            />
          </div>
          <div className="relative">
            <button 
              onClick={() => setShowColConfig(!showColConfig)}
              className={`p-1.5 rounded transition-colors ${showColConfig ? 'bg-[#1e2d4a] text-white' : 'text-slate-400 hover:text-white hover:bg-[#1e2d4a]'}`} 
              title="Filter columns"
            >
              <Filter className="w-4 h-4" />
            </button>
            {showColConfig && (
              <div className="absolute right-0 top-full mt-2 w-48 bg-[#0c1220] border border-[#1e2d4a] rounded-md shadow-xl z-50 p-2 flex flex-col gap-1">
                <div className="text-[10px] uppercase text-slate-500 font-semibold mb-1 px-1">Visible Columns</div>
                {Object.entries(columns).map(([key, isVisible]) => (
                  <label key={key} className="flex items-center gap-2 px-2 py-1 hover:bg-[#1a2540] rounded cursor-pointer text-xs text-slate-300">
                    <input type="checkbox" checked={isVisible} onChange={() => toggleColumn(key as keyof typeof columns)} className="accent-indigo-500" />
                    <span className="capitalize">{key.replace('_', ' ')}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
          <button 
            onClick={() => setIsPaused(!isPaused)}
            className={`p-1.5 rounded transition-colors flex items-center gap-1 text-xs font-semibold ${
              isPaused 
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' 
                : 'bg-[#1e2d4a] text-slate-300 hover:text-white border border-transparent'
            }`}
          >
            {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            {isPaused ? 'RESUME' : 'PAUSE'}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto relative">
        <table className="soc-table w-full relative z-10">
          <thead className="sticky top-0 bg-[#060a13] z-20">
            <tr>
              <th className="w-8"></th>
              {columns.timestamp && <th className="w-32">Timestamp</th>}
              {columns.attacker && <th className="w-32">Attacker IP</th>}
              {columns.victim && <th className="w-32">Victim Host</th>}
              {columns.technique_id && <th className="w-24">ID</th>}
              {columns.technique && <th className="w-48">Technique</th>}
              {columns.stage && <th className="w-32">Stage</th>}
              {columns.severity && <th className="w-24">Severity</th>}
              {columns.campaign && <th className="w-32 text-right">Campaign</th>}
            </tr>
          </thead>
          <tbody>
            {filteredEvents.length === 0 ? (
              <tr>
                <td colSpan={10} className="text-center py-8 text-slate-500 italic">No events matching criteria</td>
              </tr>
            ) : (
              filteredEvents.map((event, idx) => {
                const id = event.id || `${event.timestamp}-${event.attacker_ip}-${event.technique_id}`;
                const isExpanded = expandedEventId === id;
                return (
                  <React.Fragment key={`${id}-${idx}`}>
                    <tr 
                      onClick={() => handleRowClick(event)}
                      className={`cursor-pointer transition-colors ${isExpanded ? 'bg-indigo-500/10' : 'hover:bg-[#1a2540]'}`}
                    >
                      <td className="text-slate-500 text-center">
                        {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                      </td>
                      {columns.timestamp && <td className="text-slate-400 text-xs">{formatDate(event.timestamp)}</td>}
                      {columns.attacker && <td className="text-red-400 font-mono text-xs">{event.attacker_ip}</td>}
                      {columns.victim && <td className="text-blue-400 font-mono text-xs">{event.victim_ip}</td>}
                      {columns.technique_id && <td className="text-indigo-400 font-bold font-mono text-xs">{event.technique_id}</td>}
                      {columns.technique && <td className="text-slate-200 text-xs truncate max-w-[200px]" title={event.technique_name}>{event.technique_name}</td>}
                      {columns.stage && <td className="text-emerald-400 text-xs">{event.stage}</td>}
                      {columns.severity && (
                        <td>
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${getSeverityClass(event.severity)}`}>
                            {event.severity}
                          </span>
                        </td>
                      )}
                      {columns.campaign && (
                        <td className="text-right text-orange-400 font-mono text-xs truncate max-w-[120px]" title={event.campaign_id}>
                          {event.campaign_id.replace('CAMP_', '')}
                        </td>
                      )}
                    </tr>
                    
                    {/* Expanded Drawer Details */}
                    {isExpanded && (
                      <tr className="bg-[#0a0e17] border-b border-[#1e2d4a]">
                        <td colSpan={10} className="p-0">
                          <div className="p-6 grid grid-cols-3 gap-6 animate-slide-in">
                            {/* Meta Info */}
                            <div className="col-span-1 flex flex-col gap-4">
                              <div>
                                <h4 className="text-xs font-semibold text-slate-500 uppercase flex items-center gap-1 mb-2">
                                  <AlertTriangle className="w-3.5 h-3.5" /> Threat Intelligence
                                </h4>
                                <div className="bg-[#131c2e] border border-[#1e2d4a] rounded p-3 text-xs space-y-2">
                                  <div className="flex justify-between">
                                    <span className="text-slate-400">First Seen:</span>
                                    <span className="text-slate-200">{formatDate(eventDetail?.first_seen || event.timestamp)}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-slate-400">Last Seen:</span>
                                    <span className="text-slate-200">{formatDate(eventDetail?.timestamp || event.timestamp)}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-slate-400">Occurrences:</span>
                                    <span className="text-slate-200 font-mono bg-[#1a2540] px-1 rounded">{eventDetail?.occurrences || 1}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-slate-400">Rule ID:</span>
                                    <span className="font-mono text-indigo-400">{eventDetail?.technique_id || event.technique_id}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-slate-400">Description:</span>
                                    <span className="text-slate-300 truncate max-w-[120px]" title={eventDetail?.technique_name || event.technique_name}>
                                      {eventDetail?.technique_name || event.technique_name}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-slate-400">Severity:</span>
                                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${getSeverityClass(eventDetail?.severity || event.severity)}`}>
                                      {eventDetail?.severity || event.severity}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-slate-400">Campaign:</span>
                                    <span className="font-mono text-orange-400">{eventDetail?.campaign_id || event.campaign_id}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-slate-400">Attacker IP:</span>
                                    <span className="font-mono text-red-400">{eventDetail?.attacker_ip || event.attacker_ip}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-slate-400">Victim IP:</span>
                                    <span className="font-mono text-blue-400">{eventDetail?.victim_ip || event.victim_ip}</span>
                                  </div>
                                </div>
                              </div>
                              
                              {eventDetail?.predicted_next_attack && (
                                <div>
                                  <h4 className="text-xs font-semibold text-slate-500 uppercase flex items-center gap-1 mb-2">
                                    <Activity className="w-3.5 h-3.5" /> Predicted Next Step
                                  </h4>
                                  <div className="bg-cyan-500/10 border border-cyan-500/30 rounded p-3">
                                    <div className="font-mono text-cyan-400 font-bold text-lg">{eventDetail.predicted_next_attack}</div>
                                    <div className="text-xs text-slate-400 mt-1">Confidence: {eventDetail.prediction_confidence}%</div>
                                  </div>
                                </div>
                              )}
                              
                              {eventDetail?.recommendations && eventDetail.recommendations.length > 0 && (
                                <div>
                                  <h4 className="text-xs font-semibold text-slate-500 uppercase flex items-center gap-1 mb-2">
                                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> Playbook
                                  </h4>
                                  <div className="bg-emerald-500/10 border border-emerald-500/30 rounded p-3 text-xs space-y-1">
                                    {eventDetail.recommendations.map((rec, i) => {
                                      const text = typeof rec === 'string' ? rec : rec.recommendation;
                                      return (
                                      <div key={i} className="flex items-start gap-1 text-slate-300">
                                        <ChevronRight className="w-3 h-3 text-emerald-400 flex-shrink-0 mt-0.5" />
                                        <span>{text}</span>
                                      </div>
                                    )})}
                                  </div>
                                </div>
                              )}
                            </div>

                            {/* SOC Investigation Evidence Panel */}
                            <div className="col-span-2 flex flex-col gap-4">
                              <h4 className="text-xs font-semibold text-slate-500 uppercase flex items-center gap-1 mb-2">
                                <FileJson className="w-3.5 h-3.5" /> SOC Investigation Evidence Panel
                              </h4>
                              
                              {loadingDetail ? (
                                <div className="bg-[#060a13] border border-[#1e2d4a] rounded flex-1 p-4 flex items-center justify-center">
                                  <div className="text-slate-500 text-sm italic animate-pulse">Loading deep investigation evidence...</div>
                                </div>
                              ) : (
                                <div className="space-y-3 overflow-auto max-h-[400px] pr-2 custom-scrollbar">
                                  {/* Rule Summary */}
                                  {eventDetail?.investigation_payload?.rule_description && (
                                    <div className="bg-[#131c2e] border border-[#1e2d4a] rounded p-3">
                                      <div className="text-xs text-slate-400 mb-1">Rule Detail</div>
                                      <div className="text-sm text-slate-200">{eventDetail.investigation_payload.rule_description}</div>
                                      <div className="flex gap-4 mt-2">
                                        <span className="text-xs font-mono text-indigo-400 bg-indigo-500/10 px-1.5 py-0.5 rounded">ID: {eventDetail.investigation_payload.rule_id}</span>
                                        <span className="text-xs font-mono text-red-400 bg-red-500/10 px-1.5 py-0.5 rounded">Level: {eventDetail.investigation_payload.rule_level}</span>
                                        {eventDetail.investigation_payload.location && <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded truncate max-w-[200px]" title={eventDetail.investigation_payload.location}>Src: {eventDetail.investigation_payload.location}</span>}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Network Evidence */}
                                  {eventDetail?.investigation_payload?.network && Object.keys(eventDetail.investigation_payload.network).length > 0 && (
                                    <details className="bg-[#060a13] border border-blue-500/30 rounded p-3 group">
                                      <summary className="text-[10px] font-bold text-blue-400 uppercase tracking-wider mb-2 cursor-pointer list-none flex items-center gap-2">
                                        <ChevronRight className="w-3 h-3 transition-transform group-open:rotate-90" />
                                        Network Evidence
                                      </summary>
                                      <pre className="text-[10px] text-slate-300 font-mono whitespace-pre-wrap mt-2">
                                        {JSON.stringify(eventDetail.investigation_payload.network, null, 2)}
                                      </pre>
                                    </details>
                                  )}
                                  
                                  {/* HTTP Evidence */}
                                  {eventDetail?.investigation_payload?.http && Object.keys(eventDetail.investigation_payload.http).length > 0 && (
                                    <details className="bg-[#060a13] border border-purple-500/30 rounded p-3 group">
                                      <summary className="text-[10px] font-bold text-purple-400 uppercase tracking-wider mb-2 cursor-pointer list-none flex items-center gap-2">
                                        <ChevronRight className="w-3 h-3 transition-transform group-open:rotate-90" />
                                        HTTP Context
                                      </summary>
                                      <pre className="text-[10px] text-slate-300 font-mono whitespace-pre-wrap mt-2">
                                        {JSON.stringify(eventDetail.investigation_payload.http, null, 2)}
                                      </pre>
                                    </details>
                                  )}

                                  {/* Suricata Evidence */}
                                  {eventDetail?.investigation_payload?.suricata && Object.keys(eventDetail.investigation_payload.suricata).length > 0 && (
                                    <details className="bg-[#060a13] border border-red-500/30 rounded p-3 group">
                                      <summary className="text-[10px] font-bold text-red-400 uppercase tracking-wider mb-2 cursor-pointer list-none flex items-center gap-2">
                                        <ChevronRight className="w-3 h-3 transition-transform group-open:rotate-90" />
                                        Suricata IPS Context
                                      </summary>
                                      <pre className="text-[10px] text-slate-300 font-mono whitespace-pre-wrap mt-2">
                                        {JSON.stringify(eventDetail.investigation_payload.suricata, null, 2)}
                                      </pre>
                                    </details>
                                  )}

                                  {/* Fallback Raw JSON if none of the above exist */}
                                  {(!eventDetail?.investigation_payload || Object.keys(eventDetail.investigation_payload).length === 0) && (
                                    <details className="bg-[#060a13] border border-[#1e2d4a] rounded p-3 group">
                                      <summary className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2 cursor-pointer list-none flex items-center gap-2">
                                        <ChevronRight className="w-3 h-3 transition-transform group-open:rotate-90" />
                                        Raw Alert Fallback
                                      </summary>
                                      <pre className="text-[10px] text-slate-300 font-mono whitespace-pre-wrap mt-2">
                                        {JSON.stringify(eventDetail?.raw_wazuh_event || eventDetail, null, 2)}
                                      </pre>
                                    </details>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })
            )}
          </tbody>
        </table>
        
        {!isPaused && (
          <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent animate-[scan-line_4s_linear_infinite] pointer-events-none z-30"></div>
        )}
      </div>

      {/* Pagination Controls */}
      <div className="p-3 border-t border-[#1e2d4a] flex items-center justify-between bg-[#131c2e]/50">
        <div className="text-xs text-slate-400">
          Showing {events.length} of {eventsTotal} total events
        </div>
        <div className="flex items-center gap-2">
          <button 
            disabled={eventsPage === 1}
            onClick={() => { setEventsPage(eventsPage - 1); refreshEvents(); }}
            className="px-3 py-1 text-xs bg-[#1e2d4a] hover:bg-indigo-500/30 text-white rounded transition-colors disabled:opacity-50"
          >
            Previous
          </button>
          <span className="text-xs text-slate-300">Page {eventsPage}</span>
          <button 
            disabled={events.length < 50}
            onClick={() => { setEventsPage(eventsPage + 1); refreshEvents(); }}
            className="px-3 py-1 text-xs bg-[#1e2d4a] hover:bg-indigo-500/30 text-white rounded transition-colors disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
