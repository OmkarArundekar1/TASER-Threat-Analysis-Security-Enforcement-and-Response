import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { OperationId } from './OperationId';

it('renders the raw operation id untruncated', () => {
  render(<OperationId id="OP_10407C1A" />);
  expect(screen.getByText('OP_10407C1A')).toBeInTheDocument();
});

it('does not add any prefix or suffix text around the id', () => {
  render(<OperationId id="OP_ABCDEF12" />);
  expect(screen.getByText('OP_ABCDEF12').textContent).toBe('OP_ABCDEF12');
});

it('renders long ids without truncating the text content', () => {
  const longId = 'OP_' + 'A'.repeat(64);
  render(<OperationId id={longId} />);
  expect(screen.getByText(longId)).toBeInTheDocument();
});
