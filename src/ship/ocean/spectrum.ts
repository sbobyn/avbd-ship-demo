// Ocean wave spectra and the dispersion relation, float64, shared by the CPU reference (reference.ts) and the
// h0(k)/ω(k) tables the GPU compute pipeline is handed (fft-gpu.ts uploads what this module produces; it does not
// recompute spectra on the GPU). Formulas follow J. Tessendorf, "Simulating Ocean Water" (Phillips) and the
// JONSWAP/TMA/directional-spreading forms summarised in C. J. Horvath, "Empirical Directional Wave Spectra for
// Computer Graphics" (DigiPro 2015).

export const GRAVITY = 9.81; // m/s^2

/** Deterministic seeded PRNG (mulberry32), so a cascade's h0(k) is reproducible from its seed alone. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller: one independent standard-normal pair per two uniform draws. */
export function gaussianPair(rng: () => number): readonly [number, number] {
  let u1 = 0;
  while (u1 <= Number.EPSILON) u1 = rng();
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1)), theta = 2 * Math.PI * u2;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

export interface Wind {
  /** m/s, at the reference height used by the spectrum. */
  speed: number;
  /** Radians; the direction waves travel toward, measured from +x toward +z. */
  direction: number;
}

/** ω² = g·k·tanh(k·h). `depth` of Infinity (or ≥ ~ a few / k) gives the deep-water ω² = g·k. */
export function dispersion(k: number, depth: number): number {
  if (k <= 0) return 0;
  const kh = k * depth;
  const shallow = Number.isFinite(depth) && kh < 20; // tanh saturates to 1 well before this
  return Math.sqrt(GRAVITY * k * (shallow ? Math.tanh(kh) : 1));
}
/** dω/dk at fixed depth, from the same ω(k); used to convert a frequency spectrum S(ω) to S(k). */
export function groupSpeedDk(k: number, depth: number, epsilon = 1e-4): number {
  const dk = Math.max(epsilon, k * 1e-4);
  return (dispersion(k + dk, depth) - dispersion(Math.max(k - dk, 1e-9), depth)) / (2 * dk);
}

/**
 * Tessendorf's Phillips spectrum: the wind's own waves (largest wavelength L = V²/g) windowed by a directional
 * factor |k̂·ŵ|² and a small-wavelength suppression term so capillary-scale noise does not alias the grid.
 */
export function phillips(kx: number, kz: number, wind: Wind, amplitude: number, minWave = 0.05): number {
  const k2 = kx * kx + kz * kz;
  if (k2 < 1e-12) return 0;
  const k = Math.sqrt(k2), L = (wind.speed * wind.speed) / GRAVITY;
  const wx = Math.cos(wind.direction), wz = Math.sin(wind.direction);
  const directional = ((kx / k) * wx + (kz / k) * wz) ** 2;
  const damping = Math.exp(-k2 * minWave * minWave);
  return amplitude * Math.exp(-1 / (k2 * L * L)) / (k2 * k2) * directional * damping;
}

export interface JonswapParams { wind: Wind; fetch: number /* metres */; gamma?: number }

/** JONSWAP frequency spectrum S(ω) (Hasselmann et al. 1973), peak-enhanced by γ (default 3.3). */
export function jonswapOmega(omega: number, p: JonswapParams): number {
  if (omega <= 0) return 0;
  const g = GRAVITY, gamma = p.gamma ?? 3.3;
  const omegaP = 22 * (g * g / (p.wind.speed * p.fetch)) ** (1 / 3);
  const alpha = 0.076 * (p.wind.speed ** 2 / (p.fetch * g)) ** 0.22;
  const sigma = omega <= omegaP ? 0.07 : 0.09;
  const r = Math.exp(-((omega - omegaP) ** 2) / (2 * sigma * sigma * omegaP * omegaP));
  const pm = (alpha * g * g) / omega ** 5 * Math.exp(-1.25 * (omegaP / omega) ** 4);
  return pm * gamma ** r;
}
export const jonswapPeakOmega = (p: JonswapParams) => 22 * (GRAVITY * GRAVITY / (p.wind.speed * p.fetch)) ** (1 / 3);

/**
 * TMA depth attenuation (Bouws et al. 1985 / Kitaigorodskii), applied to JONSWAP: a shallower sea flattens the
 * spectrum's peak. `omegaH = ω·sqrt(h/g)`; the transfer function saturates to 1 (deep water) above omegaH ≈ 2.
 */
export function tmaFactor(omega: number, depth: number): number {
  if (!Number.isFinite(depth)) return 1;
  const omegaH = omega * Math.sqrt(depth / GRAVITY);
  if (omegaH <= 1) return 0.5 * omegaH * omegaH;
  if (omegaH <= 2) return 1 - 0.5 * (2 - omegaH) ** 2;
  return 1;
}

/**
 * Mitsuyasu-style directional spreading, cos^{2s}(θ/2) about the wind direction, with s(ω/ωp) after Horvath (a
 * steeper, narrower lobe at the spectral peak; broader off-peak). Numerically normalised over [-π, π) rather than
 * via the Gamma-function closed form, so the normalisation is correct regardless of the exact exponent formula
 * (checked by the "integrates to one" unit test).
 */
export function spreadExponent(omega: number, omegaP: number, windSpeed: number): number {
  const peak = 11.5 * (GRAVITY / (omegaP * windSpeed)) ** 2.5;
  const ratio = omega / omegaP;
  return ratio < 1 ? peak * ratio ** 5 : peak * ratio ** -2.5;
}
const SPREAD_SAMPLES = 720;
const spreadNormCache = new Map<number, number>();
function spreadNormalization(s: number): number {
  const key = Math.round(s * 256);
  const cached = spreadNormCache.get(key);
  if (cached !== undefined) return cached;
  let sum = 0;
  const dtheta = (2 * Math.PI) / SPREAD_SAMPLES;
  for (let i = 0; i < SPREAD_SAMPLES; i++) {
    const theta = -Math.PI + (i + 0.5) * dtheta;
    sum += Math.cos(theta / 2) ** (2 * s) * dtheta;
  }
  const norm = 1 / sum;
  spreadNormCache.set(key, norm);
  return norm;
}
/** D(θ) with ∫ D dθ = 1 over one turn; `theta` and `windDirection` in radians. */
export function directionalSpreading(theta: number, windDirection: number, s: number): number {
  const relative = wrapAngle(theta - windDirection);
  return spreadNormalization(s) * Math.cos(relative / 2) ** (2 * s);
}
function wrapAngle(a: number): number {
  let x = a % (2 * Math.PI);
  if (x > Math.PI) x -= 2 * Math.PI;
  if (x < -Math.PI) x += 2 * Math.PI;
  return x;
}

export type SpectrumKind = 'phillips' | 'jonswap';
export interface SpectrumParams {
  kind: SpectrumKind;
  wind: Wind;
  amplitude: number; // Phillips amplitude constant A
  fetch: number; // metres, JONSWAP
  gamma?: number;
  depth: number; // metres; Infinity for deep water
  tma: boolean; // depth-limited TMA correction (JONSWAP only)
  minWave?: number;
}

/** The 2D wave-energy density P(kx, kz) feeding h0(k), in whichever units make |h0|² a wave-height variance. */
export function spectrum2D(kx: number, kz: number, p: SpectrumParams): number {
  const k2 = kx * kx + kz * kz;
  if (k2 < 1e-12) return 0;
  if (p.kind === 'phillips') return phillips(kx, kz, p.wind, p.amplitude, p.minWave);
  const k = Math.sqrt(k2), omega = dispersion(k, p.depth);
  if (omega <= 0) return 0;
  const omegaP = jonswapPeakOmega(p);
  let sOmega = jonswapOmega(omega, p) * (p.tma ? tmaFactor(omega, p.depth) : 1);
  const s = spreadExponent(omega, omegaP, p.wind.speed);
  const theta = Math.atan2(kz, kx);
  const direction = directionalSpreading(theta, p.wind.direction, s);
  // Polar (k, θ) → Cartesian (kx, kz): S(k,θ) = S(ω)·(dω/dk)·D(θ)/k.
  return sOmega * groupSpeedDk(k, p.depth) * direction / k;
}
