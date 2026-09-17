/**
 * Behavioral tests for PanelWrapper.tsx -- the shared collapse/fullscreen
 * chrome every panel in the dashboard uses. Real component, no mocks.
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { expect, it } from 'vitest';
import { PanelWrapper } from './PanelWrapper';

it('renders children by default', () => {
  render(<PanelWrapper title="Test Panel">child content</PanelWrapper>);
  expect(screen.getByText('child content')).toBeInTheDocument();
});

it('hides children when collapsed, and can be re-expanded', () => {
  render(<PanelWrapper title="Test Panel">child content</PanelWrapper>);
  fireEvent.click(screen.getByTitle('Collapse panel'));
  expect(screen.queryByText('child content')).not.toBeInTheDocument();

  fireEvent.click(screen.getByTitle('Expand panel'));
  expect(screen.getByText('child content')).toBeInTheDocument();
});

it('respects defaultCollapsed', () => {
  render(<PanelWrapper title="Test Panel" defaultCollapsed>child content</PanelWrapper>);
  expect(screen.queryByText('child content')).not.toBeInTheDocument();
});

it('toggles fullscreen and exits on Escape', () => {
  render(<PanelWrapper title="Test Panel">child content</PanelWrapper>);
  fireEvent.click(screen.getByTitle('Fullscreen'));
  expect(screen.getByTitle('Exit fullscreen')).toBeInTheDocument();

  fireEvent.keyDown(window, { key: 'Escape' });
  expect(screen.getByTitle('Fullscreen')).toBeInTheDocument();
});

it('renders the provided title and headerExtra content', () => {
  render(<PanelWrapper title="Test Panel" headerExtra={<span>extra</span>}>child</PanelWrapper>);
  expect(screen.getByText('Test Panel')).toBeInTheDocument();
  expect(screen.getByText('extra')).toBeInTheDocument();
});
