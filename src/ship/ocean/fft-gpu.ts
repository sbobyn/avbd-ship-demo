// WebGPU compute twin of fft-cpu.ts / reference.ts: the same radix-2 2D inverse FFT, the same h̃(k,t) evolution,
// and the same spectral derivatives for slope, choppy displacement and the Jacobian fold diagnostic. h0(k)/h0*(−k)/ω(k)
// are NOT recomputed on the GPU: reference.ts builds them once in float64 and this module uploads the bytes, so both
// sides start from literally the same spectrum (mirrors the clouds lab's noise-gpu.ts / noise.ts split).
//
// Every field stays on the GPU. One frame, for all cascades at once, is one renderer.compute() call — one compute
// pass, one submit — of 2·log₂N + 2 dispatches: evolve → log₂N butterfly stages (row pass) → the same stages again
// (column pass) → pack. The render graph (graph.ts) then reads the packed `surface` storage buffer directly, as
// shallow-water/mount.ts reads its simulation state. WebGPU orders a queue's submits and makes each dispatch's writes
// visible to the next, so the render pass that follows sees this frame's fields — see docs/fft-ocean-validation.md
// ("Root cause") and repro-compute-sample.ts for why the old per-frame CPU readback was never needed.
import * as THREE from 'three/webgpu';
import { cos, float, Fn, instanceIndex, instancedArray, sin, uint, uniform, vec2, vec4 } from 'three/tsl';
import type { CascadeParams } from './model.ts';
import { buildTables, type CascadeTables } from './reference.ts';
import { releaseStorage } from './gpu-resources.ts';

type N<T extends string = 'float'> = THREE.Node<T>;
type ComputeNode = THREE.ComputeNode;
type ComplexBuffer = THREE.StorageBufferNode<'vec2'>;

function reverseBits(x: N<'uint'>, bits: number): N<'uint'> {
  let r: N<'uint'> = uint(0), src = x;
  for (let b = 0; b < bits; b++) { r = r.shiftLeft(uint(1)).bitOr(src.bitAnd(uint(1))); src = src.shiftRight(uint(1)); }
  return r;
}

/**
 * A radix-2 inverse FFT over `slices` independent N×N complex grids stored back to back in one 'vec2' buffer (slice s
 * at offset s·N²), with a scratch buffer of the same shape. One 1D pass along every row of every slice is log₂N
 * dispatches: stage 0 gathers its inputs from `data` at bit-reversed columns (the permutation, fused) and writes
 * scratch; the middle stages run in place on scratch (each thread owns one butterfly pair nothing else touches that
 * dispatch); the last stage writes `data` transposed. Running the same stages twice therefore does rows, transpose,
 * rows, transpose — the separable 2D transform, back in the original orientation. N must be at least 4, so the first
 * and last stages differ (a single stage would read and write `data` in one dispatch).
 *
 * All reads go through toVar() before the writes: a plain TSL element node is a lazy expression re-read at each use,
 * which is exactly the bug that once left ~44% of an in-place swap wrong (repro-compute-sample.ts, 'NoVar').
 */
export interface BatchedFFT { data: ComplexBuffer; scratch: ComplexBuffer; stages: ComputeNode[]; dispatches: ComputeNode[] }

export function createBatchedFFT(data: ComplexBuffer, n: number, slices: number): BatchedFFT {
  const bits = Math.log2(n);
  if (!Number.isInteger(bits) || bits < 2) throw new Error('FFT size must be a power of two, at least 4.');
  const s = uint(n), n2 = uint(n * n);
  const scratch = instancedArray(slices * n * n, 'vec2');
  const stages = Array.from({ length: bits }, (_, stage) => {
    const half = 1 << stage, m = half * 2, first = stage === 0, last = stage === bits - 1;
    return Fn(() => {
      const idx = instanceIndex, perSlice = uint(n * n / 2), perRow = uint(n / 2);
      const slice = idx.div(perSlice), q = idx.mod(perSlice);
      const row = q.div(perRow), p = q.mod(perRow);
      const group = p.div(uint(half)), j = p.mod(uint(half));
      const col0 = group.mul(uint(m)).add(j), col1 = col0.add(uint(half));
      const base = slice.mul(n2), rowBase = base.add(row.mul(s));
      const a = (first ? data.element(rowBase.add(reverseBits(col0, bits))) : scratch.element(rowBase.add(col0))).toVar();
      const b = (first ? data.element(rowBase.add(reverseBits(col1, bits))) : scratch.element(rowBase.add(col1))).toVar();
      const angle = float(j).mul(2 * Math.PI / m); // inverse FFT: +angle (fft-cpu.ts uses sign=+1 for inverse)
      const c = cos(angle), sn = sin(angle);
      const tre = b.x.mul(c).sub(b.y.mul(sn)), tim = b.x.mul(sn).add(b.y.mul(c));
      const out0 = vec2(a.x.add(tre), a.y.add(tim)), out1 = vec2(a.x.sub(tre), a.y.sub(tim));
      if (last) {
        data.element(base.add(col0.mul(s)).add(row)).assign(out0);
        data.element(base.add(col1.mul(s)).add(row)).assign(out1);
      } else {
        scratch.element(rowBase.add(col0)).assign(out0);
        scratch.element(rowBase.add(col1)).assign(out1);
      }
    })().compute(slices * n * n / 2);
  });
  return { data, scratch, stages, dispatches: [...stages, ...stages] };
}

const nyquistSafe = (t: CascadeTables, idx: number) => {
  const i = idx % t.n, j = (idx - i) / t.n, half = t.n >> 1;
  return i !== half && j !== half;
};

/** Packs h0/h0* into one upload array and (Nyquist-safe) kx, kz, 1/|k|, ω into another, at `offset` texels. Float32
 * is the GPU's own working precision, so this is the same rounding a WebGPU implementation would apply to a float64
 * CPU table. */
function packUploadBuffers(tables: CascadeTables, spec: Float32Array, k: Float32Array, offset: number) {
  const n2 = tables.n * tables.n;
  for (let idx = 0; idx < n2; idx++) {
    const o = 4 * (offset + idx);
    spec[o] = tables.h0.re[idx]!; spec[o + 1] = tables.h0.im[idx]!;
    spec[o + 2] = tables.h0c.re[idx]!; spec[o + 3] = tables.h0c.im[idx]!;
    const safe = nyquistSafe(tables, idx);
    k[o] = safe ? tables.kx[idx]! : 0; k[o + 1] = safe ? tables.kz[idx]! : 0;
    k[o + 2] = tables.invK[idx]!; k[o + 3] = tables.omega[idx]!;
  }
}

/** Complex slices per cascade: each holds two real fields as A + i·B (both inverse transforms are real, because every
 * spectrum here is Hermitian — so one complex FFT yields both, A in .x and B in .y). */
export const PACKED_FIELDS = [['dispX', 'dispZ'], ['height', 'jxz'], ['slopeX', 'slopeZ'], ['jxx', 'jzz']] as const;

export interface OceanGPU {
  readonly n: number;
  readonly cascades: readonly { length: number; choppiness: number }[];
  time: ReturnType<typeof uniform>;
  /** Per cascade c, texel t (= row·N + col): surface[2(c·N² + t)] = (choppy dispX, height, choppy dispZ, 1) and
   *  surface[2(c·N² + t) + 1] = (slopeX, slopeZ, Jacobian, 0). The only buffer the render graph reads. */
  surface: THREE.StorageBufferNode<'vec4'>;
  fft: BatchedFFT;
  evolve: ComputeNode; pack: ComputeNode;
  /** Everything one frame dispatches, in order, as one renderer.compute() call. */
  dispatches: ComputeNode[];
  inputs: THREE.StorageBufferNode<'vec4'>[]; // spec and k, uploaded once
}

/** Builds the GPU pipeline for a set of cascades sharing one grid size: uploads h0, its −k conjugate, k and ω from the
 * CPU tables, then the evolve, batched FFT and pack kernels. */
export function createOceanGPU(cascades: readonly CascadeParams[]): OceanGPU {
  const n = cascades[0]?.n ?? 0, count = cascades.length, n2 = n * n;
  if (count === 0 || cascades.some(c => c.n !== n)) throw new Error('All cascades must share one grid size.');
  const spec = new Float32Array(count * n2 * 4), k = new Float32Array(count * n2 * 4);
  cascades.forEach((c, i) => packUploadBuffers(buildTables(c), spec, k, i * n2));
  const specBuffer = instancedArray(spec, 'vec4'), kBuffer = instancedArray(k, 'vec4');
  const slices = count * PACKED_FIELDS.length;
  const data = instancedArray(slices * n2, 'vec2');
  const fft = createBatchedFFT(data, n, slices);
  const surface = instancedArray(count * n2 * 2, 'vec4');
  const time = uniform(0);
  const un2 = uint(n2);

  // h̃(k,t) = h0·e^{iωt} + h0*(−k)·e^{−iωt}, then the spectral derivatives for slope, choppy displacement and the
  // Jacobian fold — mirrors reference.ts evolveHeightSpectrum / slopeSpectrum / displacementSpectrum /
  // jacobianSpectrum. Three bindings (spec, k, data), well under the 8-storage-buffer-per-stage limit that once forced
  // one-buffer-per-field kernels to be split in two.
  const evolve = Fn(() => {
    const idx = instanceIndex, c = idx.div(un2), cell = idx.mod(un2);
    const spec4 = specBuffer.element(idx), k4 = kBuffer.element(idx);
    const wt = k4.w.mul(time), cw = cos(wt), sw = sin(wt);
    const a0 = spec4.x, b0 = spec4.y, a1 = spec4.z, b1 = spec4.w;
    const h = vec2(a0.mul(cw).sub(b0.mul(sw)).add(a1.mul(cw)).add(b1.mul(sw)), a0.mul(sw).add(b0.mul(cw)).add(b1.mul(cw)).sub(a1.mul(sw))).toVar();
    const kx = k4.x, kz = k4.y, invK = k4.z;
    const iw = (w: N) => vec2(h.y.negate().mul(w), h.x.mul(w)); // i·w·H
    const fields = {
      height: h, slopeX: iw(kx), slopeZ: iw(kz),
      dispX: iw(kx.mul(invK)).negate(), dispZ: iw(kz.mul(invK)).negate(), // −i·(k/|k|)·H
      jxx: h.mul(kx.mul(kx).mul(invK)), jzz: h.mul(kz.mul(kz).mul(invK)), jxz: h.mul(kx.mul(kz).mul(invK)),
    };
    const base = c.mul(uint(PACKED_FIELDS.length)).mul(un2).add(cell);
    PACKED_FIELDS.forEach(([a, b], slot) => {
      const A = fields[a], B = fields[b]; // A + i·B
      data.element(base.add(uint(slot * n2))).assign(vec2(A.x.sub(B.y), A.y.add(B.x)));
    });
  })().compute(count * n2);

  // Real parts back out of the packed slices, choppiness applied, J = (1+Jxx)(1+Jzz) − Jxz² (unchanged from the
  // DataTexture version). Choppiness is a per-cascade constant, picked by a short select chain.
  const pack = Fn(() => {
    const idx = instanceIndex, c = idx.div(un2), cell = idx.mod(un2);
    const base = c.mul(uint(PACKED_FIELDS.length)).mul(un2).add(cell);
    const slot = (i: number) => data.element(base.add(uint(i * n2))).toVar();
    const disp = slot(0), heightJxz = slot(1), slope = slot(2), jj = slot(3);
    let chop: N = float(cascades[count - 1]!.choppiness);
    for (let i = count - 2; i >= 0; i--) chop = c.equal(uint(i)).select(float(cascades[i]!.choppiness), chop);
    // Jacobian of the *choppy* map (x, z) ↦ (x + λDx, z + λDz), λ = this cascade's choppiness: its partials are
    // 1 + λ·∂Dx/∂x etc., the same λ the displacement itself is scaled by a few lines above (disp.x.mul(chop)). jj.x/
    // jj.y/heightJxz.y are the *unscaled* spectral second derivatives (Jxx, Jzz, Jxz); multiplying each by λ before
    // combining is the fix docs/fft-ocean-validation.md records — a λ = 0 cascade (no choppiness) now correctly
    // never folds (J ≡ 1), where the unscaled version could still report J < 1 from height curvature alone.
    const jxx = jj.x.mul(chop), jzz = jj.y.mul(chop), jxz = heightJxz.y.mul(chop);
    const jacobian = float(1).add(jxx).mul(float(1).add(jzz)).sub(jxz.mul(jxz));
    surface.element(idx.mul(uint(2))).assign(vec4(disp.x.mul(chop), heightJxz.x, disp.y.mul(chop), 1));
    surface.element(idx.mul(uint(2)).add(uint(1))).assign(vec4(slope.x, slope.y, jacobian, 0));
  })().compute(count * n2);

  return {
    n, cascades: cascades.map(c => ({ length: c.length, choppiness: c.choppiness })), time, surface, fft, evolve, pack,
    dispatches: [evolve, ...fft.dispatches, pack], inputs: [specBuffer, kBuffer],
  };
}

/** Queues one frame of GPU work for every cascade — one compute pass, one submit, no readback. The render that
 * follows on the same queue samples the result. */
export function updateOceanGPU(renderer: THREE.WebGPURenderer, ocean: OceanGPU, time: number) {
  ocean.time.value = time;
  renderer.compute(ocean.dispatches);
}

/** Validation only (never per frame): copies `surface` back to the CPU, with a timeout. */
export async function readOceanSurface(renderer: THREE.WebGPURenderer, ocean: OceanGPU, timeoutMs = 20000) {
  let timer = 0;
  const timeout = new Promise<never>((_, reject) => { timer = window.setTimeout(() => reject(new Error('Surface readback timed out.')), timeoutMs); });
  try {
    return new Float32Array((await Promise.race([renderer.getArrayBufferAsync(ocean.surface.value as THREE.BufferAttribute), timeout])).slice(0));
  } finally { clearTimeout(timer); }
}

/** Frees the compute pipelines and the storage buffers. A dropped ComputeNode keeps its bind groups until dispose(),
 *  and three only deletes a bare storage attribute's GPUBuffer through a geometry's dispose, so the buffers are
 *  released from the renderer directly (like clouds/noise-gpu.ts disposes its one-off kernel). */
export function disposeOceanGPU(ocean: OceanGPU, renderer?: THREE.WebGPURenderer | null) {
  for (const node of [ocean.evolve, ocean.pack, ...ocean.fft.stages]) node.dispose();
  releaseStorage(renderer, [...ocean.inputs, ocean.fft.data, ocean.fft.scratch, ocean.surface].map(b => b.value as THREE.BufferAttribute));
}
