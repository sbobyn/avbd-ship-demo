import assert from 'node:assert/strict';
import { test } from 'node:test';
import { clothWeights } from '../src/ship/geometry.ts';

test('cloth binding is continuous across cell boundaries and normalized at edges', () => {
  const grid = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  const deform = (x: number, y: number): number => clothWeights(x, y, 3, 3, grid).reduce((sum, w) => sum + [0, 4, -3, 2, -5, 1, 3, 6, -1][w.body] * w.weight, 0);
  for (let t = 0; t <= 2; t += 0.1) {
    assert.ok(Math.abs(deform(1 - 1e-6, t) - deform(1 + 1e-6, t)) < 0.00003);
    assert.ok(Math.abs(deform(t, 1 - 1e-6) - deform(t, 1 + 1e-6)) < 0.00003);
  }
  for (const [x, y] of [[-2, -2], [5, 5], [0.3, 0.7], [1, 1]]) {
    const weights = clothWeights(x, y, 3, 3, grid);
    assert.ok(Math.abs(weights.reduce((sum, w) => sum + w.weight, 0) - 1) < 1e-12);
    assert.ok(weights.every(w => w.weight > 0 && w.body >= 0 && w.body <= 8));
  }
});
