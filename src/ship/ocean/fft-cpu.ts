// CPU complex-array DFT and radix-2 FFT, float64. The direct DFT is the O(N^4) gate reference (small N, exact up
// to floating point); the FFT is the CPU twin of the GPU compute pipeline in fft-gpu.ts — same bit-reversal and
// butterfly index math, so a bug in one is very likely to show up as a mismatch against the other.

export interface ComplexGrid { readonly n: number; re: Float64Array; im: Float64Array }
export const makeGrid = (n: number): ComplexGrid => ({ n, re: new Float64Array(n * n), im: new Float64Array(n * n) });
export const cloneGrid = (g: ComplexGrid): ComplexGrid => ({ n: g.n, re: g.re.slice(), im: g.im.slice() });

/**
 * Direct 2D DFT (forward or inverse) by double summation, O(N^4). Array indices only — the mapping from a grid
 * index to a signed physical wavenumber lives in spectrum.ts, not here.
 */
export function directDFT2D(input: ComplexGrid, inverse: boolean): ComplexGrid {
  const { n } = input, out = makeGrid(n), sign = inverse ? 1 : -1;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let sumRe = 0, sumIm = 0;
      for (let ky = 0; ky < n; ky++) {
        for (let kx = 0; kx < n; kx++) {
          const angle = sign * 2 * Math.PI * (kx * x / n + ky * y / n);
          const c = Math.cos(angle), s = Math.sin(angle);
          const re = input.re[ky * n + kx]!, im = input.im[ky * n + kx]!;
          sumRe += re * c - im * s;
          sumIm += re * s + im * c;
        }
      }
      out.re[y * n + x] = sumRe;
      out.im[y * n + x] = sumIm;
    }
  }
  if (inverse) {
    const scale = 1 / (n * n);
    for (let i = 0; i < n * n; i++) { out.re[i]! *= scale; out.im[i]! *= scale; }
  }
  return out;
}

function bitReverseIndices(n: number): Uint32Array {
  const bits = Math.log2(n);
  if (!Number.isInteger(bits)) throw new Error('FFT size must be a power of two.');
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let x = i, r = 0;
    for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; }
    out[i] = r;
  }
  return out;
}
const reversalCache = new Map<number, Uint32Array>();
const reversalFor = (n: number) => reversalCache.get(n) ?? (reversalCache.set(n, bitReverseIndices(n)).get(n)!);

/** In-place radix-2 iterative FFT (Cooley–Tukey, decimation in time) on one row of length N. */
export function fft1D(re: Float64Array, im: Float64Array, inverse: boolean) {
  const n = re.length, bits = Math.log2(n);
  if (!Number.isInteger(bits)) throw new Error('FFT size must be a power of two.');
  const rev = reversalFor(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i]!;
    if (j > i) {
      const tr = re[i]!; re[i] = re[j]!; re[j] = tr;
      const ti = im[i]!; im[i] = im[j]!; im[j] = ti;
    }
  }
  const sign = inverse ? 1 : -1;
  for (let s = 0; s < bits; s++) {
    const half = 1 << s, m = half << 1;
    for (let k = 0; k < n; k += m) {
      for (let j = 0; j < half; j++) {
        const angle = sign * 2 * Math.PI * j / m, c = Math.cos(angle), sn = Math.sin(angle);
        const i0 = k + j, i1 = i0 + half;
        const tre = re[i1]! * c - im[i1]! * sn, tim = re[i1]! * sn + im[i1]! * c;
        re[i1] = re[i0]! - tre; im[i1] = im[i0]! - tim;
        re[i0] = re[i0]! + tre; im[i0] = im[i0]! + tim;
      }
    }
  }
  if (inverse) { for (let i = 0; i < n; i++) { re[i]! /= n; im[i]! /= n; } }
}

/** Separable 2D FFT: every row, then every column. Mutates and returns `grid`. */
export function fft2D(grid: ComplexGrid, inverse: boolean): ComplexGrid {
  const { n, re, im } = grid;
  const rowRe = new Float64Array(n), rowIm = new Float64Array(n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) { rowRe[x] = re[y * n + x]!; rowIm[x] = im[y * n + x]!; }
    fft1D(rowRe, rowIm, inverse);
    for (let x = 0; x < n; x++) { re[y * n + x] = rowRe[x]!; im[y * n + x] = rowIm[x]!; }
  }
  const colRe = new Float64Array(n), colIm = new Float64Array(n);
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { colRe[y] = re[y * n + x]!; colIm[y] = im[y * n + x]!; }
    fft1D(colRe, colIm, inverse);
    for (let y = 0; y < n; y++) { re[y * n + x] = colRe[y]!; im[y * n + x] = colIm[y]!; }
  }
  return grid;
}
