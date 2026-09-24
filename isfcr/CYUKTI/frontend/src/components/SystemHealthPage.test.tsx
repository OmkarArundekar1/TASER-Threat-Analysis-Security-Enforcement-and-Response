import { render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi, beforeEach, afterEach } from 'vitest';
import { SystemHealthPage } from './SystemHealthPage';
import { api } from '../services/api';

vi.mock('../services/api', () => ({ api: { systemHealth: vi.fn() } }));
const mockApi = vi.mocked(api);

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  vi.useRealTimers();
});

it('shows overall healthy status and each subsystem independently', async () => {
  mockApi.systemHealth.mockResolvedValue({
    status: 'healthy',
    timestamp: '2026-01-01T00:00:00Z',
    subsystems: {
      neo4j: { status: 'connected' },
      gnn: { status: 'disabled' },
      misp: { status: 'credential_missing' },
      xgboost_severity: { status: 'available', model_path: 'ml/models/xgb_severity.json' },
      wazuh_listener: { status: 'no_log_found' },
    },
  });

  render(<SystemHealthPage />);

  await waitFor(() => expect(screen.getByText('HEALTHY')).toBeInTheDocument());
  expect(screen.getByText('connected')).toBeInTheDocument();
  expect(screen.getByText('disabled')).toBeInTheDocument();
  expect(screen.getByText('credential_missing')).toBeInTheDocument();
});

it('shows degraded status when a subsystem is down', async () => {
  mockApi.systemHealth.mockResolvedValue({
    status: 'degraded',
    timestamp: '2026-01-01T00:00:00Z',
    subsystems: { neo4j: { status: 'disconnected', detail: 'refused' } },
  });

  render(<SystemHealthPage />);

  await waitFor(() => expect(screen.getByText('DEGRADED')).toBeInTheDocument());
});
