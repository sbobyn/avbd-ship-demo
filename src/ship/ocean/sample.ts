// Adapted from the Web GPU Gems FFT ocean: bilinear GPU storage sampling.
import { floor, int, mix, storage } from 'three/tsl';
import type * as THREE from 'three/webgpu';
import type { OceanGPU } from './fft-gpu.ts';
type N<T extends string = 'float'> = THREE.Node<T>;
export function surfaceSampler(ocean: OceanGPU) {
  const n = ocean.n, n2 = n * n, mask = int(n - 1), one = int(1);
  const surface = storage(ocean.surface.value, 'vec4', ocean.cascades.length * n2 * 2).toReadOnly();
  return (cascade: number, uv: N<'vec2'>, record: 0 | 1): N<'vec4'> => {
    const t = uv.mul(n).sub(0.5), cell = floor(t), f = t.sub(cell);
    const i0 = int(cell.x).bitAnd(mask), j0 = int(cell.y).bitAnd(mask);
    const i1 = i0.add(one).bitAnd(mask), j1 = j0.add(one).bitAnd(mask);
    const at = (i: N<'int'>, j: N<'int'>) => surface.element(j.mul(int(n)).add(i).add(int(cascade * n2)).mul(int(2)).add(int(record)));
    return mix(mix(at(i0, j0), at(i1, j0), f.x), mix(at(i0, j1), at(i1, j1), f.x), f.y);
  };
}

