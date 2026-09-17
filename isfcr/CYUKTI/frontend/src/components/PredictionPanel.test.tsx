/**
 * Behavioral tests for PredictionPanel.tsx.
 *
 * Note: this component is not currently mounted anywhere in App.tsx --
 * `predictions` state is fetched by DashboardContext but no component
 * in the actual render tree displays it via this panel (verified by
 * grepping the whole src tree for "PredictionPanel" -- only the
 * component's own file references it). Documented as a known finding
 * in TESTING.md; not wired into the layout this session (a layout
 * decision, out of scope for test-infrastructure work) but tested here
 * in isolation since it is real, non-trivial, reachable component logic.
 */
import { render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { PredictionPanel } from './PredictionPanel';
import { useDashboard } from '../context/DashboardContext';
import { predictionFixture } from '../test/fixtures';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
const mockUseDashboard = vi.mocked(useDashboard);

it('shows the no-prediction state when there is nothing to show', () => {
  mockUseDashboard.mockReturnValue({ predictions: [], selectedCampaign: null } as unknown as ReturnType<typeof useDashboard>);
  render(<PredictionPanel />);
  expect(screen.getByText(/no likely_next relationship found/i)).toBeInTheDocument();
});

it('renders the prediction for the selected campaign when one exists', () => {
  mockUseDashboard.mockReturnValue({
    predictions: [predictionFixture], selectedCampaign: predictionFixture.campaign_id,
  } as unknown as ReturnType<typeof useDashboard>);
  render(<PredictionPanel />);

  expect(screen.getByText(predictionFixture.current_technique)).toBeInTheDocument();
  expect(screen.getByText(predictionFixture.predicted_technique)).toBeInTheDocument();
  expect(screen.getByText('64')).toBeInTheDocument();
  expect(screen.getByText(predictionFixture.risk_level, { exact: false })).toBeInTheDocument();
});

it('falls back to the first prediction when no campaign is selected', () => {
  mockUseDashboard.mockReturnValue({
    predictions: [predictionFixture], selectedCampaign: null,
  } as unknown as ReturnType<typeof useDashboard>);
  render(<PredictionPanel />);
  expect(screen.getByText(predictionFixture.predicted_technique)).toBeInTheDocument();
});

it('shows the no-prediction state when a campaign is selected but has no matching prediction', () => {
  mockUseDashboard.mockReturnValue({
    predictions: [predictionFixture], selectedCampaign: 'CAMP_WITH_NO_PREDICTION',
  } as unknown as ReturnType<typeof useDashboard>);
  render(<PredictionPanel />);
  expect(screen.getByText(/no likely_next relationship found/i)).toBeInTheDocument();
});
