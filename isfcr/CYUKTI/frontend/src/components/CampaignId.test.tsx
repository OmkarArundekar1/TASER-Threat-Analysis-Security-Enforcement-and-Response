import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { CampaignId } from './CampaignId';

it('renders the raw campaign id untruncated', () => {
  render(<CampaignId id="CAMP_10407C1A" />);
  expect(screen.getByText('CAMP_10407C1A')).toBeInTheDocument();
});

it('does not add any prefix or suffix text around the id', () => {
  render(<CampaignId id="CAMP_ABCDEF12" />);
  expect(screen.getByText('CAMP_ABCDEF12').textContent).toBe('CAMP_ABCDEF12');
});

it('renders long ids without truncating the text content', () => {
  const longId = 'CAMP_' + 'A'.repeat(64);
  render(<CampaignId id={longId} />);
  expect(screen.getByText(longId)).toBeInTheDocument();
});
