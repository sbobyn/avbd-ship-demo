import * as THREE from 'three/webgpu';
import { cameraViewMatrix, cross, dFdx, dFdy, exp, float, floor, Fn, fwidth, If, instanceIndex, instancedArray, mix, positionLocal, positionWorld, sin, smoothstep, uniform, uniformArray, uint, uv, vec2, vec3, vec4 } from 'three/tsl';
import type { BowContact } from './hull.ts';
import { whitewaterDetail } from './whitewater.ts';

type Surface = (point: THREE.Node<'vec2'>) => THREE.Node<'float'>;
const SOURCES=8, SHEETS=SOURCES*6;

// Two joined parameter grids form a closed lens. The shader collapses their
// thickness at the boundary, so fans have real side profiles without rim cards.
function splashGeometry() {
  const columns=15, rows=11, positions:number[]=[], normals:number[]=[], uvs:number[]=[], indices:number[]=[];
  for(const side of [1,-1]) {
    const base=positions.length/3;
    for(let y=0;y<rows;y++) for(let x=0;x<columns;x++) {
      positions.push(x/(columns-1)-0.5,y/(rows-1),side);
      normals.push(0,0,side); uvs.push(x/(columns-1),y/(rows-1));
    }
    for(let y=0;y<rows-1;y++) for(let x=0;x<columns-1;x++) {
      const a=base+y*columns+x,b=a+1,c=a+columns,d=c+1;
      indices.push(...(side===1?[a,b,c,b,d,c]:[a,c,b,b,c,d]));
    }
  }
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  geometry.setAttribute('normal',new THREE.Float32BufferAttribute(normals,3));
  geometry.setAttribute('uv',new THREE.Float32BufferAttribute(uvs,2));
  geometry.setIndex(indices);
  return geometry;
}

// Eight measured bow stations share their contact energy and launch velocity
// with sheets and droplets. All positions stay in world space after emission.
export function createBowImpacts(surface:Surface) {
  const stations=Array.from({length:SOURCES},(_,i)=>new THREE.Vector4(6-i*0.2,i%2?-1:1,0,i%2?-1:1));
  const outline=uniformArray<'vec4'>(stations,'vec4'), transform=uniform(new THREE.Matrix4());
  const dt=uniform(1/60), clock=uniform(0), ready=uniform(0);
  const previous=instancedArray(SOURCES,'vec4');
  const sources=instancedArray(SOURCES,'vec4'), launches=instancedArray(SOURCES,'vec4');
  const contact=Fn(()=>{
    const station=outline.element(instanceIndex), old=previous.element(instanceIndex);
    const point=transform.mul(vec4(station.xy,-0.45,1)).xyz.toVar();
    const outward=transform.mul(vec4(station.zw,0,0)).xyz.normalize();
    const height=surface(point.xy).toVar();
    const velocity=ready.greaterThan(0).select(point.sub(old.xyz).div(dt.max(0.001)),vec3(0)).toVar();
    const immersion=height.sub(point.z);
    const wet=smoothstep(-0.12,0.12,immersion).mul(float(1).sub(smoothstep(0.6,1.2,immersion)));
    const rising=height.sub(old.w).div(dt.max(0.001)).sub(velocity.z).clamp(0,6);
    const cutting=velocity.dot(outward).clamp(0,6);
    const energy=cutting.pow2().mul(0.075).add(rising.pow2().mul(0.38)).mul(wet).min(5);
    const source=sources.element(instanceIndex);
    const response=float(1).sub(exp(dt.mul(-10)));
    source.assign(vec4(point.xy.add(outward.xy.mul(0.08)),height.add(0.04),ready.greaterThan(0).select(mix(source.w,energy,response),float(0))));
    const strength=rising.add(cutting.mul(0.35)).min(5);
    launches.element(instanceIndex).assign(vec4(velocity.xy.mul(0.3).add(outward.xy.mul(strength.mul(0.45).add(0.65))),strength.mul(0.7).add(1.2),0));
    old.assign(vec4(point,height));
  })().compute(SOURCES);
  const sheets=createImpactSheets(sources,launches);
  const clearContacts=Fn(()=>{
    sources.element(instanceIndex).assign(vec4(0)); launches.element(instanceIndex).assign(vec4(0)); previous.element(instanceIndex).assign(vec4(0));
  })().compute(SOURCES);
  return {...sheets,sources,launches,count:SOURCES,
    setHull(contacts:BowContact[]) {
      if(contacts.length!==SOURCES) throw new Error('Bow requires eight waterline stations.');
      contacts.forEach((p,i)=>stations[i].set(p.x,p.y,p.nx,p.ny)); ready.value=0;
    },
    reset(renderer:THREE.WebGPURenderer) { sheets.reset(renderer); renderer.compute(clearContacts); ready.value=0; },
    update(renderer:THREE.WebGPURenderer,elapsed:number,time:number,matrix:THREE.Matrix4) {
      if(elapsed<=0) return;
      dt.value=Math.min(elapsed,0.1); clock.value=time; transform.value.copy(matrix);
      renderer.compute(contact); sheets.update(renderer,elapsed,time); ready.value=1;
    },
  };
}
export type BowImpacts = ReturnType<typeof createBowImpacts>;

export function createImpactSheets(sources:THREE.StorageBufferNode<'vec4'>, launches:THREE.StorageBufferNode<'vec4'>, activeCount?:THREE.Node<'uint'>, starts?:THREE.StorageBufferNode<'vec4'>, ends?:THREE.StorageBufferNode<'vec4'>, surface?:(point:THREE.Node<'vec2'>)=>THREE.Node<'float'>, outsideHull?:(point:THREE.Node<'vec3'>)=>THREE.Node<'bool'>) {
  const count=activeCount?384:SHEETS, dt=uniform(1/60), clock=uniform(0);
  const roots=instancedArray(count,'vec4'), vectors=instancedArray(count,'vec4');
  const widths=instancedArray(count,'vec2');
  const sheetStep=Fn(()=>{
    const root=roots.element(instanceIndex), launch=vectors.element(instanceIndex);
    const index=activeCount?uint(sin(float(instanceIndex).mul(127.1).add(floor(clock.div(0.18)).mul(311.7))).mul(43758.5453).fract().mul(float(activeCount).max(1))):instanceIndex.mod(SOURCES);
    const source=sources.element(index), velocity=launches.element(index);
    const bow=starts?starts.element(index).w:float(0);
    root.w.assign(root.w.sub(dt).max(0));
    const slot=floor(clock.add(float(instanceIndex.mod(SOURCES)).mul(0.037)).div(0.18)).mod(6);
    // A tessellated contact can be centimetres long. Sampling every segment as
    // a full-width continuous film overfilled the stern with overlapping sheets.
    const chance=activeCount?velocity.w.mul(source.w).mul(float(activeCount)).mul(dt).mul(bow.mul(4).add(1)).mul(2/count).min(1):float(1);
    const emission=sin(float(instanceIndex).mul(127.1).add(clock.mul(109.5))).mul(43758.5453).fract();
    If(root.w.lessThanEqual(0).and(activeCount?activeCount.greaterThan(0).and(emission.lessThan(chance)):slot.equal(float(instanceIndex.div(SOURCES)))).and(source.w.greaterThan(0.28)),()=>{
      const life=source.w.sqrt().mul(0.12).add(0.48).add(bow.mul(0.25)).min(1.05);
      const along=sin(float(instanceIndex).mul(73.7).add(clock.mul(53.1))).mul(43758.5453).fract();
      const point=starts && ends?mix(starts.element(index).xyz,ends.element(index).xyz,along):source.xyz;
      root.assign(vec4(point,life)); launch.assign(vec4(velocity.xyz,life));
      widths.element(instanceIndex).assign(vec2(activeCount?velocity.w.mul(2.5).clamp(0.18,1.1).add(bow.mul(source.w.sqrt()).mul(0.65)).min(2.4):velocity.z.mul(0.22).add(0.55),bow.mul(0.55).add(source.w.mul(0.12)).clamp(0,1)));
    });
  })().compute(count);
  const clear=Fn(()=>{
    roots.element(instanceIndex).assign(vec4(0)); vectors.element(instanceIndex).assign(vec4(0)); widths.element(instanceIndex).assign(vec2(0));
  })().compute(count);
  const root=roots.element(instanceIndex), launch=vectors.element(instanceIndex);
  const age=launch.w.sub(root.w).max(0), fraction=age.div(launch.w.max(0.001)).clamp(0,1);
  const u=uv().x.sub(0.5), v=uv().y;
  const detail=whitewaterDetail(vec2(uv().x,v.mul(0.8).add(fraction.mul(0.15))),instanceIndex);
  const tangent=vec3(launch.y.negate(),launch.x.add(0.0001),0).normalize();
  const width=widths.element(instanceIndex).x;
  const aeration=widths.element(instanceIndex).y;
  const opening=smoothstep(0,0.22,age);
  // Young films rise as a connected fan; the trailing root joins the ballistic
  // trajectory as the curtain opens. No world-pinned splash edge.
  const flight=age.mul(v.mul(0.85).add(fraction.mul(0.15)));
  // Low-frequency geometric ripples stay smooth on the fixed grid; the finer
  // atlas breakup belongs in opacity/normals rather than jagged vertex spikes.
  const ripple=sin(uv().x.mul(Math.PI*2).add(sin(float(instanceIndex).mul(73.7)).mul(Math.PI)))
    .mul(v.pow2()).mul(fraction).mul(width).mul(0.07);
  // Roll the lip outward and vary its height across the fan. This gives the
  // connected water mass a curved silhouette before it opens into droplets.
  const lip=sin(v.mul(Math.PI)).mul(u.pow2()).mul(width).mul(aeration).mul(opening);
  const trajectory=launch.xyz.add(vec3(0,0,flight.mul(-9.81)));
  const across=cross(tangent,trajectory).add(vec3(0.00001,0,0)).normalize();
  const section=sin(uv().x.mul(Math.PI)).max(0).mul(sin(v.mul(Math.PI)).max(0).pow(0.7));
  const folds=sin(uv().x.mul(Math.PI*4).add(sin(float(instanceIndex).mul(17.3)).mul(3)))
    .mul(sin(v.mul(Math.PI*2))).mul(0.22).add(1);
  const bulk=width.mul(0.13).add(0.055).mul(aeration.mul(0.5).add(0.5)).mul(opening)
    .mul(float(1).sub(fraction.mul(0.55))).mul(section).mul(folds);
  const material=new THREE.MeshStandardNodeMaterial({transparent:true,depthWrite:false,side:THREE.FrontSide,roughness:0.5,metalness:0});
  material.positionNode=root.xyz.add(launch.xyz.mul(flight)).add(vec3(0,0,flight.pow2().mul(-4.905)))
    .add(tangent.mul(u.mul(width).mul(v.mul(1.2).add(0.05)).mul(opening)))
    .add(launch.xyz.normalize().mul(lip.mul(0.35))).add(vec3(0,0,ripple.add(lip)))
    .add(across.mul(bulk.mul(positionLocal.z)));
  const normal=cross(dFdx(positionWorld),dFdy(positionWorld)).normalize();
  const relief=detail.a.mul(0.015);
  material.normalNode=cameraViewMatrix.mul(vec4(cross(dFdx(positionWorld).add(normal.mul(dFdx(relief))),dFdy(positionWorld).add(normal.mul(dFdy(relief)))).normalize(),0)).xyz;
  // Coherent films open into irregular holes from the free edge as they age.
  // Variant selection and packed thickness avoid a repeating sinusoidal lattice.
  const tear=fraction.pow2().mul(v.mul(0.7).add(0.3)).mul(0.72).add(v.pow2().mul(0.12));
  const edgeWidth=fwidth(detail.r).max(0.06);
  const breakup=smoothstep(tear.sub(edgeWidth),tear.add(edgeWidth),detail.r);
  const fringe=smoothstep(0.45,1,v).mul(fraction).add(float(1).sub(breakup).mul(0.35)).max(aeration).clamp(0,1);
  material.roughnessNode=mix(float(0.28),float(0.72),fringe).add(detail.a.mul(0.12)).min(0.85);
  material.colorNode=mix(vec3(0.65,0.78,0.78),vec3(0.88,0.93,0.9),fringe).mul(detail.g.mul(0.18).add(0.82));
  // Mist uses the atlas' soft billow channel. Water films instead retain a
  // dense young membrane, with a scalloped rim and age-dependent perforation.
  const scallop=detail.r.sub(0.5).mul(0.11).mul(v);
  const edge=float(1).sub(smoothstep(float(0.34).add(scallop),float(0.49).add(scallop),u.abs()));
  const lipFade=float(1).sub(smoothstep(float(0.83).add(scallop),float(1).add(scallop),v));
  const thickness=detail.g.mul(0.4).add(0.6);
  const opacity=root.w.greaterThan(0).select(smoothstep(0,0.09,fraction).mul(float(1).sub(smoothstep(0.65,1,fraction)))
    .mul(edge).mul(smoothstep(0,0.06,v)).mul(lipFade).mul(breakup).mul(thickness).mul(aeration.mul(0.5).add(0.35)),float(0));
  if(outsideHull) material.maskNode=outsideHull(positionWorld);
  material.opacityNode=surface?opacity.mul(smoothstep(0.015,0.16,positionWorld.z.sub(surface(positionWorld.xy)))):opacity;
  const mesh=new THREE.InstancedMesh(splashGeometry(),material,count);
  for(let i=0;i<count;i++) mesh.setMatrixAt(i,new THREE.Matrix4());
  mesh.frustumCulled=false;
  return {mesh,roots,vectors,widths,
    reset(renderer:THREE.WebGPURenderer) { renderer.compute(clear); },
    update(renderer:THREE.WebGPURenderer,elapsed:number,time:number) {
      dt.value=Math.min(elapsed,0.1); clock.value=time; renderer.compute(sheetStep);
    },
  };
}
