import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// React Testing Library does not auto-cleanup outside of Jest's global
// afterEach hook -- Vitest needs this wired explicitly, or DOM nodes and
// React trees leak between tests in the same file.
afterEach(() => {
  cleanup();
});
