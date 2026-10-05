// Cascade definitions shared by the CPU reference (reference.ts) and the GPU compute pipeline (fft-gpu.ts). A
// cascade is one N×N spectral grid tiled at world size `length` metres; 2–3 cascades with non-commensurate
// lengths cover different wavenumber bands (Δk = 2π/L differs per cascade) and their tiling seams never coincide.
import type { SpectrumKind, Wind } from './spectrum.ts';

export interface CascadeParams {
  name: string;
  minWavenumber?: number;
  maxWavenumber?: number;
  /** Grid resolution; must be a power of two (the GPU and CPU FFTs are both radix-2). */
  n: number;
  /** World size of one tile, metres. Non-commensurate across cascades (no common integer ratio) hides repetition. */
  length: number;
  kind: SpectrumKind;
  wind: Wind;
  amplitude: number; // Phillips amplitude constant
  fetch: number; // metres, JONSWAP
  depth: number; // metres; Infinity for deep water
  tma: boolean;
  seed: number;
  /** Choppy horizontal displacement scale (Tessendorf's λ); 0 disables choppiness. */
  choppiness: number;
}

const WIND: Wind = { speed: 9, direction: 0.35 };

/** Swell → chop → ripples. Lengths are mutually irrational-ish (250, 43, 9 m) so tiling seams never line up. */
export const CASCADES: readonly CascadeParams[] = [
  { name: 'swell', maxWavenumber: 2*Math.PI/12, n: 64, length: 250, kind: 'jonswap', wind: WIND, amplitude: 3.5e-6, fetch: 200000, depth: Infinity, tma: false, seed: 1729, choppiness: 1.2 },
  { name: 'chop', minWavenumber: 2*Math.PI/12, maxWavenumber: 2*Math.PI/1.2, n: 64, length: 43, kind: 'phillips', wind: WIND, amplitude: 5e-6, fetch: 40000, depth: Infinity, tma: false, seed: 4173, choppiness: 1.0 },
  { name: 'ripples', minWavenumber: 2*Math.PI/1.2, n: 64, length: 9, kind: 'phillips', wind: { ...WIND, speed: 6 }, amplitude: 6e-6, fetch: 5000, depth: Infinity, tma: false, seed: 917, choppiness: 0.6 },
];

/** A small, fast cascade for the CPU-DFT-vs-GPU-FFT gate and the FFT-vs-DFT unit tests (O(N^4) direct DFT). */
export const CHECK_CASCADE: CascadeParams = { name: 'check', n: 32, length: 60, kind: 'jonswap', wind: WIND, amplitude: 4e-6, fetch: 80000, depth: Infinity, tma: false, seed: 2026, choppiness: 1 };

/** Depth-limited variant of the swell cascade, for the TMA option. */
export const SHALLOW_CASCADE: CascadeParams = { ...CASCADES[0]!, name: 'shallow', kind: 'jonswap', depth: 6, tma: true, seed: 55 };
