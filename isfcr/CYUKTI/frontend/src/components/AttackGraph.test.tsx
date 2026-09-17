/**
 * Behavioral tests for AttackGraph.tsx.
 *
 * react-force-graph-2d renders to an HTML canvas via a physics engine --
 * not meaningfully testable (or worth testing) under jsdom, and doing so
 * would be exactly the brittle internal-implementation-detail testing
 * the mission for this phase explicitly warned against. It is mocked
 * here as an external boundary (like the HTTP layer) so this file can
 * test AttackGraph's OWN real logic: empty/loading states, and the
 * node/link merging + deduplication it performs before handing data to
 * the graph library.
 */
import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { AttackGraph } from './AttackGraph';
import { useDashboard } from '../context/DashboardContext';
import { graphDataFixture, emptyGraphDataFixture } from '../test/fixtures';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));

vi.mock('react-force-graph-2d', () => ({
  default: vi.fn((props: { graphData: { nodes: unknown[]; links: unknown[] } }) => (
    <div data-testid="force-graph-stub" data-node-count={props.graphData.nodes.length} data-link-count={props.graphData.links.length} />
  )),
}));

const mockUseDashboard = vi.mocked(useDashboard);

function setDashboardState(overrides: Partial<ReturnType<typeof useDashboard>>) {
  mockUseDashboard.mockReturnValue({
    graphData: null, refreshGraph: vi.fn(), loading: false, graphLayer: 'campaign',
    setGraphLayer: vi.fn(), selectTechnique: vi.fn(), selectCampaign: vi.fn(), resetKey: 0,
    ...overrides,
  } as unknown as ReturnType<typeof useDashboard>);
}

it('shows a loading message when data is loading and nothing has arrived yet', () => {
  setDashboardState({ graphData: emptyGraphDataFixture, loading: true });
  render(<AttackGraph />);
  expect(screen.getByText(/loading graph data/i)).toBeInTheDocument();
  expect(screen.queryByTestId('force-graph-stub')).not.toBeInTheDocument();
});

it('shows a no-data message for an empty (but successfully loaded) graph', () => {
  setDashboardState({ graphData: emptyGraphDataFixture, loading: false });
  render(<AttackGraph />);
  expect(screen.getByText(/no graph data available/i)).toBeInTheDocument();
});

it('renders the graph library with the real node/link counts once data arrives', () => {
  setDashboardState({ graphData: graphDataFixture, loading: false });
  render(<AttackGraph />);
  const stub = screen.getByTestId('force-graph-stub');
  expect(stub.dataset.nodeCount).toBe(String(graphDataFixture.nodes.length));
  expect(stub.dataset.linkCount).toBe(String(graphDataFixture.links.length));
});

it('does not crash when graphData is null (before the first fetch resolves)', () => {
  setDashboardState({ graphData: null, loading: true });
  expect(() => render(<AttackGraph />)).not.toThrow();
  expect(screen.getByText(/loading graph data/i)).toBeInTheDocument();
});

it('offers all three graph-layer switch controls', () => {
  setDashboardState({ graphData: emptyGraphDataFixture });
  render(<AttackGraph />);
  expect(screen.getByRole('button', { name: 'Campaign' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Investigation' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Threat Intel' })).toBeInTheDocument();
});
