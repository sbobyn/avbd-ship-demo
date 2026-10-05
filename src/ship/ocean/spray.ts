import type { BowImpacts } from './impact.ts';
import { whitewaterDetail } from './whitewater.ts';
import * as THREE from 'three/webgpu';
import { cameraPosition, cameraViewMatrix, cameraWorldMatrix, dFdx, dFdy, dot, exp, float, Fn, If, instanceIndex, instancedArray, mix, normalize, perspectiveDepthToViewZ, positionLocal, positionWorld, positionView, screenUV, sin, smoothstep, texture, uniform, uniformArray, uint, uv, vec2, vec3, vec4 } from 'three/tsl';

// Fixed GPU pool: ballistic drops and drag-dominated mist share one draw. Inspired
// by web-gpu-gems' spray pool/soft-depth pattern; no particle positions are read back.
const DROPS=6144, MIST=1024, COUNT=DROPS+MIST;
export function createShipSpray(camera: THREE.PerspectiveCamera, depth: THREE.DepthTexture,
  surface: (point: THREE.Node<'vec2'>) => THREE.Node<'float'>,
  sunlight: THREE.Node<'vec3'>, ambient: THREE.Node<'vec3'>, sunDirection: THREE.Node<'vec3'>,
  bow?: Pick<BowImpacts,'sources'|'launches'|'count'> & {activeCount?:THREE.Node<'uint'>;starts?:THREE.StorageBufferNode<'vec4'>;ends?:THREE.StorageBufferNode<'vec4'>;outsideHull?:(point:THREE.Node<'vec3'>)=>THREE.Node<'bool'>},
  deposit?: (point:THREE.Node<'vec2'>,energy:THREE.Node<'float'>)=>void) {
  const positions=instancedArray(COUNT,'vec4'), velocities=instancedArray(COUNT,'vec4');
  const sourceValues=Array.from({length:4},()=>new THREE.Vector4());
  const sources=uniformArray<'vec4'>(sourceValues,'vec4');
  const footprint=uniform(1), heading=uniform(new THREE.Vector2(1,0));
  const dt=uniform(0), frame=uniform(0), travel=uniform(0), wind=uniform(new THREE.Vector3());
  const hash=(salt:number) => sin(float(instanceIndex).mul(127.1).add(frame.mul(311.7)).add(salt)).mul(43758.5453).fract();
  const integrate=Fn(() => {
    const p=positions.element(instanceIndex), v=velocities.element(instanceIndex);
    const mist=instanceIndex.greaterThanEqual(DROPS), count=bow?.activeCount?float(bow.activeCount).max(1):float(bow?.count??4);
    const index=bow?.activeCount?uint(hash(17.1).mul(count)):instanceIndex.mod(uint(count));
    const source=bow?bow.sources.element(index):sources.element(index);
    If(p.w.lessThanEqual(0), () => {
      // Fine mist needs a breaking impact; gentle contact produces drops without
      // a persistent fog collar around an otherwise glassy sea.
      const front=bow?.starts?bow.starts.element(index).w:float(0);
      const impact=bow?.ends?bow.ends.element(index).w:float(0);
      const emission=bow?mist.select(smoothstep(0.25,1.5,source.w).mul(source.w),source.w):source.w;
      const weight=bow?.activeCount?bow.launches.element(index).w:float(1);
      const valid=bow?.activeCount?bow.activeCount.greaterThan(0).select(float(1),float(0)):float(1);
      const focus=bow?.activeCount?mist.select(front.mul(0.75).add(0.25),float(1)):float(1);
      const rate=mist.select(float(54),float(1100)).mul(focus).mul(front.mul(mist.select(float(3),float(2))).add(1)).mul(emission).mul(weight).mul(valid);
      const burst=bow?.activeCount?sin(source.x.mul(1.13).add(source.y.mul(0.79)).add(frame.mul(0.047)))
        .mul(sin(source.y.mul(1.71).sub(frame.mul(0.031)))).mul(0.35).add(0.65).mul(impact.mul(1.4).add(0.6)):float(1);
      const chance=rate.mul(burst).mul(dt).div(mist.select(float(MIST).div(count),float(DROPS).div(count)));
      If(source.w.greaterThan(0.001).and(hash(3.1).lessThan(chance)), () => {
        const intensity=front.mul(source.w.sqrt().add(impact)).mul(0.5).clamp(0,1.5);
        const life=mist.select(hash(8.9).mul(1.4).add(1.6).add(intensity),hash(8.9).mul(0.4).add(0.7).add(impact.mul(0.3)));
        const launch=source.w.min(4).sqrt();
        const root=bow?.starts && bow.ends?mix(bow.starts.element(index).xyz,bow.ends.element(index).xyz,hash(13.7)).add(vec3(0,0,0.04)):source.xyz.add(vec3(hash(13.7).sub(0.5).mul(bow?0.32:0.9).mul(footprint),hash(21.3).sub(0.5).mul(bow?0.25:0.5).mul(footprint),0.07));
        p.assign(vec4(root,life));
        if(bow) {
          const shared=bow.launches.element(index).xyz;
          v.assign(vec4(shared.mul(mist.select(float(0.65),hash(5.1).mul(0.7).add(0.65)))
            .add(vec3(hash(7.2).sub(0.5).mul(front.mul(2).add(1)),hash(9.4).sub(0.5).mul(front.mul(2).add(1)),hash(3.7).mul(front.mul(0.8).add(0.8)))),life));
        } else {
          const forward=travel.mul(0.4).add(hash(5.1).sub(0.5));
          const lateral=instanceIndex.mod(2).equal(0).select(float(1),float(-1)).mul(hash(7.2).mul(1.8).add(1.1)).mul(launch);
          v.assign(vec4(forward.mul(heading.x).sub(lateral.mul(heading.y)),
            forward.mul(heading.y).add(lateral.mul(heading.x)),
            mist.select(float(1.1),hash(9.4).mul(1.8).add(1.3)).mul(launch),life));
        }
      });
    }).Else(() => {
      // Exact linear-drag step. Fine mist follows the air; drops fall ballistically.
      const drag=mist.select(float(1.7),float(0.18)), gravity=mist.select(float(-0.55),float(-9.81));
      const equilibrium=wind.add(vec3(0,0,gravity.div(drag)));
      const decay=exp(drag.mul(dt).negate()), difference=v.xyz.sub(equilibrium).toVar();
      const previous=p.xyz.toVar();
      p.xyz.addAssign(equilibrium.mul(dt).add(difference.mul(float(1).sub(decay).div(drag))));
      v.xyz.assign(equilibrium.add(difference.mul(decay)));
      p.w.subAssign(dt);
      // Swept samples catch fast drops crossing a thin hull within one step.
      if(bow?.outsideHull) {
        const outside=bow.outsideHull;
        If(outside(previous),()=>{
          for(let step=1;step<=8;step++) If(outside(mix(previous,p.xyz,float(step/8))).not(),()=>{ p.w.assign(0); });
        }).Else(()=>{ If(outside(p.xyz).not(),()=>{ p.w.assign(0); }); });
      }
      If(p.z.lessThan(surface(p.xy)), () => {
        if(deposit) If(mist.not(),()=>{ deposit(p.xy,v.z.abs().mul(0.004).min(0.06)); });
        p.w.assign(0);
      });
    });
  })().compute(COUNT);
  const clear=Fn(() => { positions.element(instanceIndex).assign(vec4(0)); velocities.element(instanceIndex).assign(vec4(0)); })().compute(COUNT);
  const p=positions.element(instanceIndex), v=velocities.element(instanceIndex);
  const mist=instanceIndex.greaterThanEqual(DROPS);
  const age=float(1).sub(p.w.div(v.w.max(0.001))).clamp(0,1);
  const variation=float(instanceIndex).mul(0.618).fract();
  // Impact mist has a longer lifetime; reuse that stored scalar for its
  // optical scale instead of adding a ninth compute-stage storage binding.
  const strength=mist.select(v.w.sub(2.8).clamp(0,1.5),float(0));
  const size=mist.select(age.mul(1.8).add(0.28).mul(strength.mul(0.3).add(1)),float(instanceIndex).mul(0.317).fract().pow(2).mul(0.03).add(0.005)).mul(variation.mul(0.7).add(0.65));
  const material=new THREE.MeshBasicNodeMaterial({transparent:true,depthWrite:false,side:THREE.DoubleSide});
  // Velocity-aligned streaks catch light in the ballistic fan; mist stays soft.
  const projected=cameraViewMatrix.mul(vec4(v.xyz,0)).xy;
  const direction=projected.add(vec2(0.0001,0)).normalize();
  const perpendicular=vec2(direction.y.negate(),direction.x);
  const stretch=mist.select(variation.mul(0.6).add(1.15).add(age.mul(0.5)),projected.length().mul(0.07).add(1).min(2.2));
  const billboard=perpendicular.mul(positionLocal.x).add(direction.mul(positionLocal.y.mul(stretch))).mul(size);
  material.positionNode=cameraWorldMatrix.mul(vec4(billboard,0,0)).xyz.add(p.xyz);
  const radius=uv().sub(0.5).length();
  const edge=dFdx(radius).abs().add(dFdy(radius).abs()).max(0.01);
  const soft=float(1).sub(smoothstep(float(0.45).sub(edge),float(0.5).add(edge),radius));
  const sceneZ=perspectiveDepthToViewZ(texture(depth,screenUV).r,float(camera.near),float(camera.far));
  const intersection=smoothstep(0,0.3,positionView.z.sub(sceneZ));
  const detail=whitewaterDetail(uv(),instanceIndex);
  const mistDensity=detail.b.mul(smoothstep(age.mul(0.25),age.mul(0.25).add(0.45),detail.a));
  const mistOpacity=float(1).sub(exp(mistDensity.mul(float(1).sub(age.mul(0.5))).mul(strength.mul(3).add(1)).mul(bow?.activeCount?-0.006:-0.06)));
  const fade=smoothstep(0,0.08,age).mul(float(1).sub(smoothstep(0.5,1,age)));
  if(bow?.outsideHull) material.maskNode=bow.outsideHull(positionWorld);
  material.opacityNode=p.w.greaterThan(0).select(fade.mul(intersection).mul(mist.select(mistOpacity,soft.mul(variation.mul(0.25).add(0.5)))),float(0));
  // One lighting response for the pool keeps the faint overlapping sprites stable
  // without sorting or a pair of full-screen OIT render targets.
  const cosine=dot(normalize(cameraPosition.sub(p.xyz)),sunDirection).clamp(-1,1);
  // Bounded Henyey–Greenstein forward scattering for mist, broader glints for
  // drops. Thickness shades each wisp without marching a volume per particle.
  const phase=float(0.64).div(float(1.36).sub(cosine.mul(1.2)).pow(1.5)).mul(0.16).min(1);
  const sphere=uv().sub(0.5).mul(2), nz=float(1).sub(sphere.dot(sphere)).max(0.001).sqrt();
  const dropletNormal=normalize(cameraWorldMatrix.mul(vec4(sphere,nz,0)).xyz);
  const view=normalize(cameraPosition.sub(p.xyz)), halfway=normalize(view.add(sunDirection));
  const highlight=dot(dropletNormal,halfway).max(0).pow(48).mul(2);
  const fresnel=float(1).sub(nz).pow(5).mul(0.5);
  const dropLight=dot(dropletNormal,sunDirection).max(0).mul(0.3).add(highlight).add(fresnel).add(0.12);
  material.colorNode=ambient.mul(mist.select(detail.g.mul(0.25).add(0.5),float(0.65)))
    .add(sunlight.mul(mist.select(phase.add(0.18),dropLight))).mul(vec3(0.8,0.9,0.92));
  const mesh=new THREE.InstancedMesh(new THREE.PlaneGeometry(1,1),material,COUNT);
  for(let i=0;i<COUNT;i++) mesh.setMatrixAt(i,new THREE.Matrix4());
  mesh.frustumCulled=false;
  const contacts=Array.from({length:4},()=>new THREE.Vector3());
  const impacts=new Float32Array(4), valid=new Uint8Array(4);
  return { mesh, positions, velocities, dropCount:DROPS,
    setHullBeam(halfBeam:number) { footprint.value=THREE.MathUtils.clamp(halfBeam/2,0.5,3); },
    contact(index:number, point:THREE.Vector3, impact:number, waveHeight=0) {
      contacts[index].copy(point); valid[index]=1; impacts[index]=Math.max(impacts[index],Math.min(6,impact*impact*0.65*(1+Math.min(waveHeight,3)*0.4)));
    },
    reset(renderer:THREE.WebGPURenderer) { renderer.compute(clear); valid.fill(0); impacts.fill(0); sourceValues.forEach(v=>v.set(0,0,0,0)); },
    update(renderer:THREE.WebGPURenderer, elapsed:number, speed:number, hull:THREE.Vector3, air:THREE.Vector3, yaw=0) {
      if(elapsed<=0) return;
      dt.value=Math.min(elapsed,0.05); frame.value++; heading.value.set(Math.cos(yaw),Math.sin(yaw)); travel.value=speed; wind.value.copy(air);
      contacts.forEach((point,i)=>{
        const steady=i>=2?THREE.MathUtils.smoothstep(speed,0.45,2)*0.7:0;
        sourceValues[i].set(point.x+hull.x,point.y+hull.y,point.z,valid[i]?(steady+impacts[i])*footprint.value:0);
        impacts[i]*=Math.exp(-elapsed/0.4);
      });
      renderer.compute(integrate);
    },
  };
}
