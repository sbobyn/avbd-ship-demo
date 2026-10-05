import { bandWeight } from './bands.ts';
// CPU-only ocean synthesis: h0(k) tables, time evolution, and inverse transform (direct DFT or FFT). This is the
// reference the GPU compute pipeline (fft-gpu.ts) is checked against, and the mount's "CPU sum-of-sines" import
// point (Water lab's model.ts) is compared with elsewhere — this module never touches the Water lab.
import { directDFT2D, fft2D, makeGrid, type ComplexGrid } from './fft-cpu.ts';
import { dispersion, gaussianPair, mulberry32, spectrum2D, type SpectrumParams } from './spectrum.ts';
import type { CascadeParams } from './model.ts';

/** Signed wavenumber-grid index: 0..N/2 stay positive, N/2+1..N-1 wrap to negative (standard DFT frequency order). */
export const signedIndex = (i: number, n: number) => (i <= n >> 1 ? i : i - n);
const mirror = (i: number, n: number) => (n - i) % n;

export interface CascadeTables {
  readonly n: number; readonly length: number;
  h0: ComplexGrid; h0c: ComplexGrid;
  /** Per texel: kx, kz, 1/|k| (0 at k=0), ω(k). */
  kx: Float64Array; kz: Float64Array; invK: Float64Array; omega: Float64Array;
}

function toParams(c: CascadeParams): SpectrumParams {
  return { kind: c.kind, wind: c.wind, amplitude: c.amplitude, fetch: c.fetch, depth: c.depth, tma: c.tma };
}

/** Builds h0(k), its conjugate-at-−k twin, and the per-texel wavenumber/frequency tables. Deterministic in `seed`. */
export function buildTables(cascade: CascadeParams): CascadeTables {
  const { n, length } = cascade, params = toParams(cascade);
  const rng = mulberry32(cascade.seed);
  const h0 = makeGrid(n);
  const dk = (2 * Math.PI) / length;
  const kx = new Float64Array(n * n), kz = new Float64Array(n * n), invK = new Float64Array(n * n), omega = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const kzv = (2 * Math.PI * signedIndex(j, n)) / length;
    for (let i = 0; i < n; i++) {
      const idx = j * n + i;
      const kxv = (2 * Math.PI * signedIndex(i, n)) / length;
      const k = Math.hypot(kxv, kzv);
      kx[idx] = kxv; kz[idx] = kzv; invK[idx] = k > 1e-9 ? 1 / k : 0; omega[idx] = dispersion(k, params.depth);
      const [xi1, xi2] = gaussianPair(rng);
      // h0(k) = (1/√2)(ξ1+iξ2)·√(P(k)·Δk²): the Δk² turns spectrum2D's continuous spectral density (built so
      // ∫∫P(k)d²k is a physical variance, via the JONSWAP/Phillips → Cartesian Jacobian in spectrum.ts) into a
      // per-mode variance for this grid's finite Δk = 2π/length. synthesize() below then sums modes unnormalized
      // (Tessendorf's h(x,t) = Σ_k h̃(k,t)e^{ik·x}), which is what makes the two consistent — see its comment.
      const amplitude = Math.sqrt((spectrum2D(kxv, kzv, params) * bandWeight(k, cascade.minWavenumber, cascade.maxWavenumber) / 2) * dk * dk);
      h0.re[idx] = xi1 * amplitude; h0.im[idx] = xi2 * amplitude;
    }
  }
  const h0c = makeGrid(n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const idx = j * n + i, src = mirror(j, n) * n + mirror(i, n);
    h0c.re[idx] = h0.re[src]!; h0c.im[idx] = -h0.im[src]!; // conj(h0(−k))
  }
  return { n, length, h0, h0c, kx, kz, invK, omega };
}

/** h̃(k, t) = h0(k)·e^{iωt} + h0*(−k)·e^{−iωt} (Tessendorf eq. 19). */
export function evolveHeightSpectrum(t: CascadeTables, time: number): ComplexGrid {
  const out = makeGrid(t.n);
  for (let idx = 0; idx < t.n * t.n; idx++) {
    const wt = t.omega[idx]! * time, c = Math.cos(wt), s = Math.sin(wt);
    const a0 = t.h0.re[idx]!, b0 = t.h0.im[idx]!, a1 = t.h0c.re[idx]!, b1 = t.h0c.im[idx]!;
    // h0·e^{i·wt} = (a0 c − b0 s) + i(a0 s + b0 c); h0c·e^{−i·wt} = (a1 c + b1 s) + i(b1 c − a1 s)
    out.re[idx] = a0 * c - b0 * s + a1 * c + b1 * s;
    out.im[idx] = a0 * s + b0 * c + b1 * c - a1 * s;
  }
  return out;
}

const scaled = (h: ComplexGrid, weight: (idx: number) => number): ComplexGrid => {
  const out = makeGrid(h.n);
  for (let idx = 0; idx < h.n * h.n; idx++) { const w = weight(idx); out.re[idx] = h.re[idx]! * w; out.im[idx] = h.im[idx]! * w; }
  return out;
};
const rotated = (h: ComplexGrid, weight: (idx: number) => number): ComplexGrid => {
  // Multiplies by i·weight: i·(a + bi)·w = −b·w + i·a·w.
  const out = makeGrid(h.n);
  for (let idx = 0; idx < h.n * h.n; idx++) { const w = weight(idx); out.re[idx] = -h.im[idx]! * w; out.im[idx] = h.re[idx]! * w; }
  return out;
};

// A spectral derivative multiplies by i·k, which needs k(−k) = −k(k) to keep the result Hermitian (real after the
// inverse transform). That antisymmetry fails at the Nyquist row/column of an even-size grid, where the mirror
// index is its own pair (self-paired, not negated) — the classic even-length-FFT differentiation pitfall. Those
// bins carry negligible energy (they are the grid's very shortest representable wave); zeroing them keeps the
// derivative fields exactly Hermitian instead of leaking a residual imaginary part.
const nyquistSafe = (t: CascadeTables, idx: number) => {
  const i = idx % t.n, j = (idx - i) / t.n, half = t.n >> 1;
  return i !== half && j !== half;
};
export const slopeSpectrum = (H: ComplexGrid, t: CascadeTables, axis: 'x' | 'z') =>
  rotated(H, i => (nyquistSafe(t, i) ? (axis === 'x' ? t.kx[i]! : t.kz[i]!) : 0));
export const displacementSpectrum = (H: ComplexGrid, t: CascadeTables, axis: 'x' | 'z') =>
  // −i·(k/|k|)·H: same rotation as the slope, negated, scaled by 1/|k| instead of k.
  scaled(rotated(H, i => (nyquistSafe(t, i) ? (axis === 'x' ? t.kx[i]! : t.kz[i]!) * t.invK[i]! : 0)), () => -1);
export const jacobianSpectrum = (H: ComplexGrid, t: CascadeTables, which: 'xx' | 'zz' | 'xz') =>
  scaled(H, i => {
    if (!nyquistSafe(t, i)) return 0; // same Nyquist bins the derivative fields drop; see nyquistSafe above
    return which === 'xx' ? t.kx[i]! ** 2 * t.invK[i]!
      : which === 'zz' ? t.kz[i]! ** 2 * t.invK[i]!
        : t.kx[i]! * t.kz[i]! * t.invK[i]!;
  });

/**
 * The Jacobian of the *choppy* horizontal map (x, z) ↦ (x + λDx(x,z), z + λDz(x,z)), λ = choppiness: J =
 * (1+λJxx)(1+λJzz) − (λJxz)², where Jxx/Jzz/Jxz are jacobianSpectrum's unscaled spectral second derivatives.
 * Scaling by λ before combining (not after) is what fft-gpu.ts's pack() now does too — see its comment for why the
 * unscaled version (this file's previous behaviour, before STE-1736) under-reports folding on a low-choppiness
 * cascade and over-reports it on a high one: λ = 0 must give J ≡ 1 (a flat map never folds), which only the scaled
 * form guarantees.
 */
export function jacobianField(H: ComplexGrid, t: CascadeTables, choppiness: number, method: Method = 'dft'): Float64Array {
  const jxx = synthesize(jacobianSpectrum(H, t, 'xx'), method).re;
  const jzz = synthesize(jacobianSpectrum(H, t, 'zz'), method).re;
  const jxz = synthesize(jacobianSpectrum(H, t, 'xz'), method).re;
  const out = new Float64Array(jxx.length);
  for (let i = 0; i < out.length; i++) {
    const a = choppiness * jxx[i]!, b = choppiness * jzz[i]!, c = choppiness * jxz[i]!;
    out[i] = (1 + a) * (1 + b) - c * c;
  }
  return out;
}

export type Method = 'dft' | 'fft';
/**
 * Inverse-transforms a spectrum to the spatial domain; `.im` should be ~0 (Hermitian symmetry) — see tests.
 * fft-cpu.ts's inverse DFT/FFT is the textbook-normalized one (1/N² for a 2D transform), but Tessendorf's
 * synthesis h(x,t) = Σ_k h̃(k,t)·e^{ik·x} is an unnormalized sum. Multiplying by N² here cancels that built-in
 * normalization — paired with the Δk² baked into h0(k) in buildTables, this is what makes a physical spectrum
 * (JONSWAP/Phillips, calibrated so ∫∫P(k)d²k is a variance) come out as a physical wave height instead of one
 * scaled down by the grid resolution. fft-gpu.ts's `normalize` kernel is the GPU twin: it does not divide at all,
 * which is the same cancellation (its own pipeline supplies the matching 1/N² fft-cpu.ts has).
 */
export function synthesize(spectrum: ComplexGrid, method: Method): ComplexGrid {
  const out = method === 'dft' ? directDFT2D(spectrum, true) : fft2D({ n: spectrum.n, re: spectrum.re.slice(), im: spectrum.im.slice() }, true);
  const scale = spectrum.n * spectrum.n;
  for (let i = 0; i < out.re.length; i++) { out.re[i]! *= scale; out.im[i]! *= scale; }
  return out;
}

export interface OceanSample { height: Float64Array; slopeX: Float64Array; slopeZ: Float64Array; dispX: Float64Array; dispZ: Float64Array; maxImaginary: number }
/** The full CPU pipeline: build tables → evolve → derive → inverse transform → real parts. */
export function oceanSurface(cascade: CascadeParams, time: number, method: Method): OceanSample {
  const t = buildTables(cascade), H = evolveHeightSpectrum(t, time);
  const height = synthesize(H, method);
  const slopeX = synthesize(slopeSpectrum(H, t, 'x'), method), slopeZ = synthesize(slopeSpectrum(H, t, 'z'), method);
  const dispX = synthesize(displacementSpectrum(H, t, 'x'), method), dispZ = synthesize(displacementSpectrum(H, t, 'z'), method);
  let maxImaginary = 0;
  for (const g of [height, slopeX, slopeZ, dispX, dispZ]) for (const v of g.im) maxImaginary = Math.max(maxImaginary, Math.abs(v));
  const scaleBy = (a: Float64Array, s: number) => { const out = new Float64Array(a.length); for (let i = 0; i < a.length; i++) out[i] = a[i]! * s; return out; };
  const choppy = cascade.choppiness;
  return { height: height.re, slopeX: slopeX.re, slopeZ: slopeZ.re, dispX: scaleBy(dispX.re, choppy), dispZ: scaleBy(dispZ.re, choppy), maxImaginary };
}
