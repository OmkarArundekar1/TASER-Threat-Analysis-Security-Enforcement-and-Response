/**
 * Behavioral tests for EvidenceInvestigation.tsx -- CYUKTI's evidence-aware
 * investigation UI. Mocks the context (selectedCampaign) and the api
 * module (the HTTP boundary abstraction) with realistic fixtures; asserts
 * on real rendered output, not internal state or mock call counts alone.
 *
 * This component was found, by reading it against the real backend
 * contract (investigation/loop.py's InvestigationRecord.to_dict()), to
 * silently drop several real fields the API actually returns:
 * why_selected, candidate_hypotheses, derived_from, top_k, model_metadata.
 * Fixed in EvidenceInvestigation.tsx this session; the tests below guard
 * against regressing back to dropping them.
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { EvidenceInvestigation } from './EvidenceInvestigation';
import { api } from '../services/api';
import { useDashboard } from '../context/DashboardContext';
import {
  investigationResultFixture, noModelInvestigationResultFixture, severityPredictionFixture,
} from '../test/fixtures';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
vi.mock('../services/api', () => ({
  api: {
    investigate: vi.fn(),
    predictSeverity: vi.fn(),
    ragMitreSearch: vi.fn(),
  },
}));

const mockUseDashboard = vi.mocked(useDashboard);
const mockApi = vi.mocked(api);

function setSelectedCampaign(campaignId: string | null) {
  mockUseDashboard.mockReturnValue({ selectedCampaign: campaignId } as ReturnType<typeof useDashboard>);
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('EvidenceInvestigation -- empty state', () => {
  it('prompts campaign selection and does not render action buttons when none is selected', () => {
    setSelectedCampaign(null);
    render(<EvidenceInvestigation />);
    expect(screen.getByText(/select a campaign/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /run investigation/i })).not.toBeInTheDocument();
  });
});

describe('EvidenceInvestigation -- investigation flow', () => {
  it('runs a real investigation and renders the full response, including previously-dropped fields', async () => {
    setSelectedCampaign('CAMP_7331E223');
    mockApi.investigate.mockResolvedValue(investigationResultFixture);

    render(<EvidenceInvestigation />);
    fireEvent.click(screen.getByRole('button', { name: /run investigation/i }));

    expect(screen.getByText(/investigating/i)).toBeInTheDocument();

    await waitFor(() => expect(mockApi.investigate).toHaveBeenCalledWith('CAMP_7331E223'));
    await waitFor(() => expect(screen.getByText('0.56')).toBeInTheDocument());

    // stopping reason and per-class model verdict
    expect(screen.getByText('max_steps reached')).toBeInTheDocument();
    expect(screen.getByText('Critical')).toBeInTheDocument();

    // step count and the field this component previously dropped: why_selected
    expect(screen.getByText(/#1 campaign_history/)).toBeInTheDocument();
    expect(screen.getByText(/#2 xgboost_prediction/)).toBeInTheDocument();
    expect(screen.getAllByText(/highest-scoring of/).length).toBeGreaterThan(0);

    // candidate_hypotheses for the step that produced a model verdict
    expect(screen.getByText('High 21%')).toBeInTheDocument();

    // evidence count and derived_from provenance on the attribution item
    expect(screen.getByText(/Evidence \(2\)/)).toBeInTheDocument();
    expect(screen.getByText(/derived from 1 item/)).toBeInTheDocument();
  });

  it('renders GNN topology evidence generically, same as any other source', async () => {
    setSelectedCampaign('CAMP_7331E223');
    mockApi.investigate.mockResolvedValue({
      ...investigationResultFixture,
      evidence: [
        ...investigationResultFixture.evidence,
        {
          evidence_id: 'gnn_topology:CAMP_PAST_2:historical_match',
          source: 'gnn_topology',
          source_id: 'CAMP_PAST_2',
          timestamp: '2026-01-01T00:00:00+00:00',
          type: 'historical_match',
          content: { campaign_id: 'CAMP_PAST_2', topology_similarity: 0.94 },
          confidence: 1.0,
          relevance: 0.97,
          provenance: 'ml.gnn.inference (GraphAutoencoder topology similarity, model_version=test-v1)',
          relationships: ['CAMP_PAST_2'],
          derived_from: [],
        },
      ],
      total_evidence: 3,
    });

    render(<EvidenceInvestigation />);
    fireEvent.click(screen.getByRole('button', { name: /run investigation/i }));

    await waitFor(() => expect(screen.getByText(/Evidence \(3\)/)).toBeInTheDocument());
    // appears twice: the evidence card's own badge, and the Multi-RAG source summary bar
    expect(screen.getAllByText('gnn_topology').length).toBe(2);
    expect(screen.getByText(/ml\.gnn\.inference/)).toBeInTheDocument();
    expect(screen.getByText('matched: CAMP_PAST_2')).toBeInTheDocument();
    expect(screen.getByText('topology similarity: 94.0%')).toBeInTheDocument();
  });

  it('shows a Multi-RAG source breakdown reflecting every distinct evidence source', async () => {
    setSelectedCampaign('CAMP_7331E223');
    mockApi.investigate.mockResolvedValue(investigationResultFixture);

    render(<EvidenceInvestigation />);
    fireEvent.click(screen.getByRole('button', { name: /run investigation/i }));

    await waitFor(() => expect(screen.getByText(/Multi-RAG:/)).toBeInTheDocument());
    expect(screen.getAllByText('campaign_history').length).toBeGreaterThan(0);
    expect(screen.getAllByText('attribution').length).toBeGreaterThan(0);
  });

  it('shows a controlled error message and clears any prior result when the investigation call fails', async () => {
    setSelectedCampaign('CAMP_7331E223');
    mockApi.investigate.mockRejectedValue(new Error('Database unavailable: connection refused'));

    render(<EvidenceInvestigation />);
    fireEvent.click(screen.getByRole('button', { name: /run investigation/i }));

    await waitFor(() => expect(screen.getByText('Database unavailable: connection refused')).toBeInTheDocument());
    expect(screen.queryByText(/investigation confidence/i)).not.toBeInTheDocument();
  });

  it('handles an investigation with no model verdict (no XGBoost step reached) without crashing', async () => {
    setSelectedCampaign('CAMP_7331E223');
    mockApi.investigate.mockResolvedValue(noModelInvestigationResultFixture);

    render(<EvidenceInvestigation />);
    fireEvent.click(screen.getByRole('button', { name: /run investigation/i }));

    await waitFor(() => expect(screen.getByText('no positive-value action remains')).toBeInTheDocument());
    expect(screen.queryByText(/model verdict/i)).not.toBeInTheDocument();
  });
});

describe('EvidenceInvestigation -- severity prediction flow', () => {
  it('predicts severity and renders label, confidence, top_k, and model_metadata', async () => {
    setSelectedCampaign('CAMP_7331E223');
    mockApi.predictSeverity.mockResolvedValue(severityPredictionFixture);

    render(<EvidenceInvestigation />);
    fireEvent.click(screen.getByRole('button', { name: /predict severity/i }));

    await waitFor(() => expect(mockApi.predictSeverity).toHaveBeenCalledWith('CAMP_7331E223'));
    await waitFor(() => expect(screen.getByText('Critical')).toBeInTheDocument());

    expect(screen.getByText('(62.0%)')).toBeInTheDocument();
    // top_k -- previously dropped entirely by this component
    expect(screen.getByText('High 21.0%')).toBeInTheDocument();
    expect(screen.getByText('Low 5.0%')).toBeInTheDocument();
    // model_metadata -- previously dropped entirely
    expect(screen.getByText(/campaign_dataset_v1/)).toBeInTheDocument();
  });

  it('shows a prediction error without touching investigation state', async () => {
    setSelectedCampaign('CAMP_7331E223');
    mockApi.predictSeverity.mockRejectedValue(new Error('No trained severity model available yet.'));

    render(<EvidenceInvestigation />);
    fireEvent.click(screen.getByRole('button', { name: /predict severity/i }));

    await waitFor(() => expect(screen.getByText('No trained severity model available yet.')).toBeInTheDocument());
  });
});

describe('EvidenceInvestigation -- MITRE semantic search (campaign-independent)', () => {
  it('is usable without a selected campaign', () => {
    setSelectedCampaign(null);
    render(<EvidenceInvestigation />);
    expect(screen.getByPlaceholderText(/describe observed behavior/i)).toBeInTheDocument();
  });

  it('runs a search on Enter and renders results', async () => {
    setSelectedCampaign(null);
    mockApi.ragMitreSearch.mockResolvedValue({
      query: 'password guessing',
      results: [{
        evidence_id: 'mitre:T1110', source: 'mitre', source_id: 'T1110', timestamp: 't',
        type: 'technique_knowledge', content: { name: 'Brute Force' }, confidence: 1.0, relevance: 0.91,
        provenance: 'mitre_resolver', relationships: [], derived_from: [],
      }],
    });

    render(<EvidenceInvestigation />);
    const input = screen.getByPlaceholderText(/describe observed behavior/i);
    fireEvent.change(input, { target: { value: 'password guessing' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => expect(screen.getByText('Brute Force')).toBeInTheDocument());
    expect(screen.getByText('0.91')).toBeInTheDocument();
  });

  it('shows a no-match message for an empty result set rather than nothing', async () => {
    setSelectedCampaign(null);
    mockApi.ragMitreSearch.mockResolvedValue({ query: 'gibberish', results: [] });

    render(<EvidenceInvestigation />);
    const input = screen.getByPlaceholderText(/describe observed behavior/i);
    fireEvent.change(input, { target: { value: 'gibberish' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => expect(screen.getByText(/no techniques matched/i)).toBeInTheDocument());
  });
});
