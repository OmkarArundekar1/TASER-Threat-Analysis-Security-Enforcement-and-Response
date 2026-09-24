import { render, screen, fireEvent } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { PredictionIntelligencePage } from './PredictionIntelligencePage';
import { useDashboard } from '../context/DashboardContext';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
const mockUseDashboard = vi.mocked(useDashboard);
const selectCampaign = vi.fn();

it('shows an empty state when there are no predictions', () => {
  mockUseDashboard.mockReturnValue({
    predictions: [], selectedCampaign: null, selectCampaign,
  } as unknown as ReturnType<typeof useDashboard>);

  render(<PredictionIntelligencePage />);
  expect(screen.getByText('No predictions available yet')).toBeInTheDocument();
});

it('lists every real prediction, not just one', () => {
  mockUseDashboard.mockReturnValue({
    predictions: [
      { campaign_id: 'CAMP_1', current_technique: 'T1078', predicted_technique: 'T1021', confidence: 72, risk_level: 'HIGH', stage: null, generated_at: '2026-01-01T00:00:00Z', source: 'LIKELY_NEXT' },
      { campaign_id: 'CAMP_2', current_technique: 'T1059', predicted_technique: 'T1105', confidence: 40, risk_level: 'LOW', stage: null, generated_at: '2026-01-01T00:00:00Z' },
    ],
    selectedCampaign: null,
    selectCampaign,
  } as unknown as ReturnType<typeof useDashboard>);

  render(<PredictionIntelligencePage />);
  expect(screen.getByText('CAMP_1')).toBeInTheDocument();
  expect(screen.getByText('CAMP_2')).toBeInTheDocument();
  expect(screen.getByText('2 active predictions')).toBeInTheDocument();
});

it('selecting a prediction card updates the shared campaign selection', () => {
  mockUseDashboard.mockReturnValue({
    predictions: [
      { campaign_id: 'CAMP_1', current_technique: 'T1078', predicted_technique: 'T1021', confidence: 72, risk_level: 'HIGH', stage: null, generated_at: '2026-01-01T00:00:00Z' },
    ],
    selectedCampaign: null,
    selectCampaign,
  } as unknown as ReturnType<typeof useDashboard>);

  render(<PredictionIntelligencePage />);
  fireEvent.click(screen.getByText('CAMP_1'));
  expect(selectCampaign).toHaveBeenCalledWith('CAMP_1');
});
