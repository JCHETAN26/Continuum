import { render, screen } from '@testing-library/react';
import type { ReactNode } from 'react';
import { expect, test, vi } from 'vitest';
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
