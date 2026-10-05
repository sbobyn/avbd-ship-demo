// Breaking-crest foam with a finite lifetime, sampled in each cascade's own tile.
import * as THREE from 'three/webgpu';
import { exp, float, floor, Fn, instanceIndex, instancedArray, int, mix, smoothstep, storage, uniform, uint, vec2 } from 'three/tsl';
import type { OceanGPU } from './fft-gpu.ts';

export function createWhitecaps(ocean: OceanGPU, wave: THREE.Node<'float'>, activity:THREE.Node<'float'>=float(1)) {
  const n=ocean.n, count=n*n*ocean.cascades.length;
  const coverage=instancedArray(count,'vec2'), dt=uniform(0);
  const surface=storage(ocean.surface.value,'vec4',count*2).toReadOnly();
  const step=Fn(() => {
    const i=instanceIndex;
    // Differentiate the displaced surface at its rendered amplitude. Unscaled
    // spectral compression would keep emitting foam after the waves calm down.
    const cascade=i.div(n*n), cell=i.mod(n*n), x=cell.mod(n), y=cell.div(n);
    const at=(a: THREE.Node<'uint'>,b: THREE.Node<'uint'>) => surface.element(cascade.mul(n*n).add(b.bitAnd(uint(n-1)).mul(n)).add(a.bitAnd(uint(n-1))).mul(2));
    let spacing: THREE.Node<'float'>=float(ocean.cascades[ocean.cascades.length-1].length/n);
    for(let c=ocean.cascades.length-2;c>=0;c--) spacing=cascade.equal(c).select(float(ocean.cascades[c].length/n),spacing);
    const dx=at(x.add(1),y).xz.sub(at(x.add(n-1),y).xz).mul(wave).div(spacing.mul(2));
    const dy=at(x,y.add(1)).xz.sub(at(x,y.add(n-1)).xz).mul(wave).div(spacing.mul(2));
    const jacobian=float(1).add(dx.x).mul(float(1).add(dy.y)).sub(dx.y.mul(dy.x));
    const height=surface.element(i.mul(2)).y.mul(wave);
    const neighbourhood=at(x.add(1),y).y.add(at(x.add(n-1),y).y).add(at(x,y.add(1)).y).add(at(x,y.add(n-1)).y).mul(wave).mul(0.25);
    // Small, convex breaking peaks emit foam; broad swell shoulders do not.
    const peak=smoothstep(spacing.mul(0.005),spacing.mul(0.05),height.sub(neighbourhood));
    const crest=smoothstep(0.025,0.11,height).mul(peak).mul(cascade.greaterThan(0).select(float(1),float(0)));
    // Curvature catches narrow overturning tips before the displacement folds.
    // Compression alone can peak on a trough in a mixed spectral wave field.
    const breaking=peak.max(float(1).sub(smoothstep(0.65,0.9,jacobian)));
    const compression=breaking.mul(crest).mul(activity).mul(smoothstep(0.08,0.35,wave));
    const previous=coverage.element(i).toVar();
    // Density and its age moment persist after the breaking crest passes.
    // New deposition renews a patch; old membranes thin out over several seconds.
    const decay=exp(dt.mul(-0.65)), birth=float(1).sub(exp(dt.mul(-3).mul(compression)));
    const carried=previous.x.mul(decay), density=carried.add(float(1).sub(carried).mul(birth)).clamp(0,1);
    const moment=previous.y.mul(decay).add(carried.mul(dt)).mul(float(1).sub(birth)).min(density.mul(30));
    coverage.element(i).assign(vec2(density,moment));
  })().compute(count);
  const read=storage(coverage.value,'vec2',count).toReadOnly();
  const sampleState=(cascade:number,point:THREE.Node<'vec2'>)=>{
    const p=point.div(ocean.cascades[cascade].length).add(0.5).mul(n).sub(0.5);
    const cell=floor(p), f=p.sub(cell), mask=int(n-1);
    const x=int(cell.x), y=int(cell.y);
    const at=(i: THREE.Node<'int'>,j: THREE.Node<'int'>) => read.element(j.bitAnd(mask).mul(n).add(i.bitAnd(mask)).add(cascade*n*n));
    return mix(mix(at(x,y),at(x.add(1),y),f.x),mix(at(x,y.add(1)),at(x.add(1),y.add(1)),f.x),f.y);
  };
  return {coverage,sampleState,
    update(renderer: THREE.WebGPURenderer, delta: number) { dt.value=Math.min(Math.max(delta,0),0.1); renderer.compute(step); },
    sample:(cascade:number,point:THREE.Node<'vec2'>)=>sampleState(cascade,point).x,
  };
}
