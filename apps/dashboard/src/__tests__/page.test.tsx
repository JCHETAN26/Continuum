import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { afterEach, expect, test, vi } from 'vitest';
import Home from '../app/page';

// Mock Recharts since it requires a real DOM layout to render SVG properly
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  LineChart: () => <div>LineChart</div>,
  Line: () => <div>Line</div>,
  XAxis: () => <div>XAxis</div>,
  YAxis: () => <div>YAxis</div>,
  CartesianGrid: () => <div>CartesianGrid</div>,
  Tooltip: () => <div>Tooltip</div>,
}));

test('renders overview page with expected elements', () => {
  render(<Home />);
  expect(screen.getByText('Overview')).toBeDefined();
  expect(screen.getByText('Current Drift Score')).toBeDefined();
  expect(screen.getByText('Active Model')).toBeDefined();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

// Each endpoint gets the shape the page expects: the snapshot handlers call .reverse() on
// the list responses, so one generic object breaks them in a way that has nothing to do
// with what is under test.
function stubBackend(demoResponse: unknown, calls: string[]) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push(`${init?.method ?? 'GET'} ${url}`);
      let body: unknown = [];
      if (url.includes('/v1/demo/')) {
        body = demoResponse;
      } else if (url.includes('/summary')) {
        body = {};
      } else if (url.includes('/projection')) {
        body = { points: [] };
      }
      return { ok: true, json: async () => body };
    }),
  );
}

test('run demo button posts to the ingest service', async () => {
  const calls: string[] = [];
  stubBackend({ phase: 'baseline', ingested: 10, total: 1200, error: null }, calls);

  render(<Home />);
  fireEvent.click(screen.getByRole('button', { name: /run demo/i }));

  await waitFor(() => expect(calls).toContain('POST http://localhost:8000/v1/demo/seed'));
});

test('a backend that is down leaves the button usable', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => {
      throw new Error('connection refused');
    }),
  );

  render(<Home />);
  const button = screen.getByRole('button', { name: /run demo/i });
  fireEvent.click(button);

  // The click swallows the failure rather than surfacing an unhandled rejection, the same
  // way the page's snapshot fetches do.
  // jest-dom matchers are not installed here, so this checks the property directly.
  await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false));
});
