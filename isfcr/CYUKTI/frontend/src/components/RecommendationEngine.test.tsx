/**
 * Behavioral tests for RecommendationEngine.tsx ("Defensive Playbook") --
 * built and functional but previously never mounted in App.tsx, so
 * these are its first tests. Backed by recommendation_engine.py's real
 * MITRE-mitigation lookup, already fetched into DashboardContext.
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { expect, it, vi, beforeEach } from 'vitest';
import { RecommendationEngine } from './RecommendationEngine';
import { useDashboard } from '../context/DashboardContext';

vi.mock('../context/DashboardContext', () => ({ useDashboard: vi.fn() }));
const mockUseDashboard = vi.mocked(useDashboard);

function setDashboard(overrides: Partial<ReturnType<typeof useDashboard>>) {
  mockUseDashboard.mockReturnValue({
    recommendations: [], selectedCampaign: null, predictions: [], resetKey: 0,
    ...overrides,
  } as unknown as ReturnType<typeof useDashboard>);
}

beforeEach(() => {
  vi.clearAllMocks();
});

it('shows the empty state when no recommendations are available', () => {
  setDashboard({});
  render(<RecommendationEngine />);
  expect(screen.getByText(/no automated recommendations available/i)).toBeInTheDocument();
});

it('renders mitigation items for the selected campaign\'s predicted technique', () => {
  setDashboard({
    selectedCampaign: 'CAMP_1',
    predictions: [{ campaign_id: 'CAMP_1', current_technique: 'T1078', predicted_technique: 'T1110', predicted_name: 'Brute Force', confidence: 80, risk_level: 'HIGH', stage: null, generated_at: 't' }],
    recommendations: [{
      technique_id: 'T1110',
      recommendations: [
        { recommendation: 'Enforce account lockout policy', priority: 'HIGH', mitre_mitigation: 'M1036', reason: 'Limits brute force attempts', traceability: 'MITRE ATT&CK → M1036' },
      ],
    }],
  });

  render(<RecommendationEngine />);
  expect(screen.getByText('Enforce account lockout policy')).toBeInTheDocument();
  expect(screen.getByText('M1036')).toBeInTheDocument();
  expect(screen.getByText('Brute Force')).toBeInTheDocument();
});

it('toggles an item as done when clicked', () => {
  setDashboard({
    selectedCampaign: 'CAMP_1',
    predictions: [{ campaign_id: 'CAMP_1', current_technique: 'T1078', predicted_technique: 'T1110', confidence: 80, risk_level: 'HIGH', stage: null, generated_at: 't' }],
    recommendations: [{ technique_id: 'T1110', recommendations: ['Patch the affected service'] }],
  });

  render(<RecommendationEngine />);
  const item = screen.getByText('Patch the affected service');
  expect(item.className).not.toContain('line-through');
  fireEvent.click(item);
  expect(item.className).toContain('line-through');
});

it('normalizes plain string recommendations (legacy shape) into full items', () => {
  setDashboard({
    selectedCampaign: null,
    predictions: [],
    recommendations: [{ technique_id: 'T1595', recommendations: ['Deploy network segmentation'] }],
  });

  render(<RecommendationEngine />);
  expect(screen.getByText('Deploy network segmentation')).toBeInTheDocument();
  expect(screen.getByText('MEDIUM')).toBeInTheDocument(); // default priority for normalized strings
});
