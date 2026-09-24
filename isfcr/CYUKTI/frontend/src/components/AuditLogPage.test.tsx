import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { expect, it, vi, beforeEach } from 'vitest';
import { AuditLogPage } from './AuditLogPage';
import { api } from '../services/api';

vi.mock('../services/api', () => ({ api: { auditLogs: vi.fn() } }));
const mockApi = vi.mocked(api);

beforeEach(() => {
  vi.clearAllMocks();
});

it('shows an empty state when there are no entries', async () => {
  mockApi.auditLogs.mockResolvedValue({ entries: [], total_lines: 0, log_path: 'logs/prerana_listener.log' });

  render(<AuditLogPage />);

  await waitFor(() => expect(screen.getByText('No log entries found yet.')).toBeInTheDocument());
});

it('renders parsed log entries with timestamp, level, and message', async () => {
  mockApi.auditLogs.mockResolvedValue({
    entries: [
      { timestamp: '2026-09-24 08:35:18,013', level: 'INFO', message: 'Alert ID: 5b8961dae8c8' },
    ],
    total_lines: 1580,
    log_path: 'logs/prerana_listener.log',
  });

  render(<AuditLogPage />);

  await waitFor(() => expect(screen.getByText('Alert ID: 5b8961dae8c8')).toBeInTheDocument());
  expect(screen.getByText('INFO')).toBeInTheDocument();
  expect(screen.getByText(/1,580 total lines/)).toBeInTheDocument();
});

it('refresh button re-fetches the log', async () => {
  mockApi.auditLogs.mockResolvedValue({ entries: [], total_lines: 0, log_path: 'x' });
  render(<AuditLogPage />);
  await waitFor(() => expect(mockApi.auditLogs).toHaveBeenCalledTimes(1));

  fireEvent.click(screen.getByText('Refresh'));
  await waitFor(() => expect(mockApi.auditLogs).toHaveBeenCalledTimes(2));
});
