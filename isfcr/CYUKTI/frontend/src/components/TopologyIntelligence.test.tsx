/**
 * Behavioral tests for TopologyIntelligence.tsx -- the dedicated GNN
 * showcase panel backed by /api/gnn/status and /api/gnn/topology/<id>
 * (rag/gnn_topology_retriever.py). Distinct from ThreatCorrelation's
 * small inline gnn_topology_similarity annotation.
 */
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { expect, it, vi, beforeEach } from 'vitest';
import { TopologyIntelligence } from './TopologyIntelligence';
import { api } from '../services/api';
import { useDashboard } from '../context/DashboardContext';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
vi.mock('../services/api', () => ({ api: { gnnStatus: vi.fn(), gnnTopology: vi.fn() } }));

const mockUseDashboard = vi.mocked(useDashboard);
const mockApi = vi.mocked(api);
const selectCampaign = vi.fn();

function setSelectedCampaign(campaignId: string | null) {
  mockUseDashboard.mockReturnValue({ selectedCampaign: campaignId, selectCampaign } as unknown as ReturnType<typeof useDashboard>);
}

beforeEach(() => {
  vi.clearAllMocks();
  mockApi.gnnStatus.mockResolvedValue({
    gnn_available: true, gnn_model_version: 'gnn_autoencoder.pt@123-456b',
    gnn_metadata: {
      model_type: 'graph_autoencoder', architecture: '2-layer SAGEConv encoder, mean-pool + Linear+tanh bottleneck',
      hidden_dim: 16, embedding_dim: 8, num_layers: 2, node_feature_schema: [], edge_feature_schema: [],
      normalization: '', training_seed: 42, model_version: 'gnn_autoencoder.pt@123-456b',
      training_dataset_description: '', artifact_scope: '',
    },
  });
});

it('shows the disabled state when GNN is not available, without calling the topology endpoint', async () => {
  mockApi.gnnStatus.mockResolvedValue({ gnn_available: false, gnn_model_version: null, gnn_metadata: null });
  setSelectedCampaign('CAMP_1');

  render(<TopologyIntelligence />);

  await waitFor(() => expect(screen.getByText(/GNN Topology Engine Disabled/i)).toBeInTheDocument());
  expect(mockApi.gnnTopology).not.toHaveBeenCalled();
});

it('prompts campaign selection when GNN is available but no campaign is selected', async () => {
  setSelectedCampaign(null);
  render(<TopologyIntelligence />);
  await waitFor(() => expect(screen.getByText('Select a Campaign')).toBeInTheDocument());
});

it('renders ranked topology neighbors with similarity percentages and model metadata', async () => {
  setSelectedCampaign('CAMP_7331E223');
  mockApi.gnnTopology.mockResolvedValue({
    campaign_id: 'CAMP_7331E223', gnn_available: true, gnn_model_version: 'gnn_autoencoder.pt@123-456b',
    topology_neighbors: [
      {
        evidence_id: 'gnn_topology:CAMP_A:historical_match', source: 'gnn_topology', source_id: 'CAMP_A',
        timestamp: 't', type: 'historical_match',
        content: { campaign_id: 'CAMP_A', attacker: '10.0.0.1', topology_similarity: 0.994 },
        confidence: 1.0, relevance: 0.997, provenance: 'ml.gnn.inference', relationships: ['CAMP_A'], derived_from: [],
      },
    ],
  });

  render(<TopologyIntelligence />);

  await waitFor(() => expect(screen.getByText('CAMP_A')).toBeInTheDocument());
  expect(screen.getByText('99.4%')).toBeInTheDocument();
  expect(screen.getByText(/2-layer SAGEConv/)).toBeInTheDocument();
  expect(screen.getByText('8-dim embedding')).toBeInTheDocument();
});

it('selects the matched campaign when a result is clicked', async () => {
  setSelectedCampaign('CAMP_7331E223');
  mockApi.gnnTopology.mockResolvedValue({
    campaign_id: 'CAMP_7331E223', gnn_available: true,
    topology_neighbors: [{
      evidence_id: 'gnn_topology:CAMP_A:historical_match', source: 'gnn_topology', source_id: 'CAMP_A',
      timestamp: 't', type: 'historical_match', content: { campaign_id: 'CAMP_A', topology_similarity: 0.9 },
      confidence: 1.0, relevance: 0.95, provenance: 'ml.gnn.inference', relationships: ['CAMP_A'], derived_from: [],
    }],
  });

  render(<TopologyIntelligence />);
  await waitFor(() => expect(screen.getByText('CAMP_A')).toBeInTheDocument());
  fireEvent.click(screen.getByText('CAMP_A'));

  expect(selectCampaign).toHaveBeenCalledWith('CAMP_A');
});

it('shows an empty state when no structurally comparable campaigns are found', async () => {
  setSelectedCampaign('CAMP_7331E223');
  mockApi.gnnTopology.mockResolvedValue({ campaign_id: 'CAMP_7331E223', gnn_available: true, topology_neighbors: [] });

  render(<TopologyIntelligence />);
  await waitFor(() => expect(screen.getByText(/no structurally comparable campaigns/i)).toBeInTheDocument());
});

it('shows a controlled error message if the topology lookup fails', async () => {
  setSelectedCampaign('CAMP_7331E223');
  mockApi.gnnTopology.mockRejectedValue(new Error('Database unavailable'));

  render(<TopologyIntelligence />);
  await waitFor(() => expect(screen.getByText('Database unavailable')).toBeInTheDocument());
});
