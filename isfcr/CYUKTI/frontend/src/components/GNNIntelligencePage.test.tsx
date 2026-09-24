/**
 * Behavioral tests for GNNIntelligencePage.tsx -- the top-level GNN
 * Intelligence view (TopNavBar -> "GNN Intelligence"), distinct from
 * the smaller in-panel TopologyIntelligence.tsx tab. Same backend
 * contract (/api/gnn/status, /api/gnn/topology/<id>), a full-page
 * layout with its own campaign picker instead of relying on the graph
 * selection.
 */
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { expect, it, vi, beforeEach } from 'vitest';
import { GNNIntelligencePage } from './GNNIntelligencePage';
import { api } from '../services/api';
import { useDashboard } from '../context/DashboardContext';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
vi.mock('../services/api', () => ({ api: { gnnStatus: vi.fn(), gnnTopology: vi.fn() } }));

const mockUseDashboard = vi.mocked(useDashboard);
const mockApi = vi.mocked(api);
const selectCampaign = vi.fn();

const campaigns = [
  { campaign_id: 'CAMP_1', campaign_label: 'Campaign #1', attacker_ip: '1.1.1.1', victim_ip: '2.2.2.2', first_seen: 't', last_seen: 't', event_count: 3, risk_score: 10, risk_level: 'LOW' as const, latest_technique: 'T1078' },
];

function setDashboard(selectedCampaign: string | null) {
  mockUseDashboard.mockReturnValue({
    campaigns, selectedCampaign, selectCampaign,
  } as unknown as ReturnType<typeof useDashboard>);
}

beforeEach(() => {
  vi.clearAllMocks();
});

it('shows the disabled state with an explanation when GNN is off', async () => {
  mockApi.gnnStatus.mockResolvedValue({ gnn_available: false, gnn_model_version: null, gnn_metadata: null });
  setDashboard(null);

  render(<GNNIntelligencePage />);

  await waitFor(() => expect(screen.getByText('MODEL DISABLED')).toBeInTheDocument());
  expect(screen.getByText(/GNN_ENABLED=true/)).toBeInTheDocument();
  expect(mockApi.gnnTopology).not.toHaveBeenCalled();
});

it('shows model metadata and a campaign picker once GNN is available', async () => {
  mockApi.gnnStatus.mockResolvedValue({
    gnn_available: true, gnn_model_version: 'gnn_autoencoder.pt@1-2b',
    gnn_metadata: {
      model_type: 'graph_autoencoder', architecture: '2-layer SAGEConv encoder, mean-pool + Linear+tanh bottleneck',
      hidden_dim: 16, embedding_dim: 8, num_layers: 2, node_feature_schema: [], edge_feature_schema: [],
      normalization: '', training_seed: 42, model_version: 'gnn_autoencoder.pt@1-2b',
      training_dataset_description: '', artifact_scope: '',
    },
  });
  setDashboard(null);

  render(<GNNIntelligencePage />);

  await waitFor(() => expect(screen.getByText('MODEL ACTIVE')).toBeInTheDocument());
  expect(screen.getByText('8')).toBeInTheDocument(); // embedding dim
  expect(screen.getByRole('option', { name: 'Campaign #1' })).toBeInTheDocument();
});

it('fetches and ranks topology neighbors for the selected campaign, full-width cards', async () => {
  mockApi.gnnStatus.mockResolvedValue({
    gnn_available: true, gnn_model_version: 'v1',
    gnn_metadata: {
      model_type: 'graph_autoencoder', architecture: 'x', hidden_dim: 16, embedding_dim: 8, num_layers: 2,
      node_feature_schema: [], edge_feature_schema: [], normalization: '', training_seed: 42,
      model_version: 'v1', training_dataset_description: '', artifact_scope: '',
    },
  });
  mockApi.gnnTopology.mockResolvedValue({
    campaign_id: 'CAMP_1', gnn_available: true,
    topology_neighbors: [{
      evidence_id: 'gnn_topology:CAMP_2:historical_match', source: 'gnn_topology', source_id: 'CAMP_2',
      timestamp: 't', type: 'historical_match',
      content: { campaign_id: 'CAMP_2', attacker: '9.9.9.9', topology_similarity: 0.87 },
      confidence: 1.0, relevance: 0.9, provenance: 'ml.gnn.inference', relationships: ['CAMP_2'], derived_from: [],
    }],
  });
  setDashboard('CAMP_1');

  render(<GNNIntelligencePage />);

  await waitFor(() => expect(mockApi.gnnTopology).toHaveBeenCalledWith('CAMP_1', 12));
  await waitFor(() => expect(screen.getByText('CAMP_2')).toBeInTheDocument());
  expect(screen.getByText('87.0%')).toBeInTheDocument();
  expect(screen.getByText('9.9.9.9')).toBeInTheDocument();
});

it('selecting a campaign from the dropdown updates shared dashboard selection', async () => {
  mockApi.gnnStatus.mockResolvedValue({ gnn_available: true, gnn_model_version: 'v1', gnn_metadata: null });
  mockApi.gnnTopology.mockResolvedValue({ campaign_id: 'CAMP_1', gnn_available: true, topology_neighbors: [] });
  setDashboard(null);

  render(<GNNIntelligencePage />);
  await waitFor(() => expect(screen.getByText('MODEL ACTIVE')).toBeInTheDocument());

  fireEvent.change(screen.getByRole('combobox'), { target: { value: 'CAMP_1' } });
  expect(selectCampaign).toHaveBeenCalledWith('CAMP_1');
});
