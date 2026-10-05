// Smooth spectral-energy windows; adjacent bands form a partition of unity.
export function bandWeight(k: number, min = 0, max = Infinity): number {
  const above = (edge: number) => {
    if (edge === 0) return 1;
    if (!Number.isFinite(edge)) return 0;
    const t = Math.max(0, Math.min(1, (k / edge - 0.8) / 0.4));
    return t * t * (3 - 2 * t);
  };
  return Math.max(0, above(min) - above(max));
}
