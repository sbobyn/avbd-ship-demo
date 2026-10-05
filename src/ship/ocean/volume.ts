import * as THREE from 'three/webgpu';
import { cameraPosition, cameraViewMatrix, cross, dot, exp, float, Fn, If, instanceIndex, instancedArray, Loop, mix, normalize, perspectiveDepthToViewZ, positionLocal, positionWorld, screenUV, sin, smoothstep, texture, texture3D, uint, uniform, vec2, vec3, vec4 } from 'three/tsl';
import type { HullContacts } from './contact.ts';

const COUNT=128;
const CORES=COUNT/2;

// One small, periodic density texture shared by all world-space impact volumes.
// Its coherent octaves sculpt aerated sheets rather than independent fog balls.
function densityTexture() {
  const n=64, data=new Uint8Array(n*n*n);
  const hash=(x:number,y:number,z:number,period:number)=>{
    let h=Math.imul((x+period)%period,374761393)^Math.imul((y+period)%period,668265263)^Math.imul((z+period)%period,1274126177);
    h=Math.imul(h^(h>>>13),1274126177); return ((h^(h>>>16))>>>0)/4294967296;
  };
  const noise=(x:number,y:number,z:number,p:number)=>{
    const ix=Math.floor(x),iy=Math.floor(y),iz=Math.floor(z);
    const ease=(v:number)=>v*v*(3-2*v), u=ease(x-ix),v=ease(y-iy),w=ease(z-iz);
    let value=0;
    for(let dz=0;dz<2;dz++) for(let dy=0;dy<2;dy++) for(let dx=0;dx<2;dx++) value+=hash(ix+dx,iy+dy,iz+dz,p)*(dx?u:1-u)*(dy?v:1-v)*(dz?w:1-w);
    return value;
  };
  for(let z=0;z<n;z++) for(let y=0;y<n;y++) for(let x=0;x<n;x++) data[(z*n+y)*n+x]=Math.round(255*(noise(x*4/n,y*4/n,z*4/n,4)*0.35+noise(x*12/n,y*12/n,z*12/n,12)*0.4+noise(x*24/n,y*24/n,z*24/n,24)*0.25));
  const map=new THREE.Data3DTexture(data,n,n,n);
  map.format=THREE.RedFormat; map.minFilter=map.magFilter=THREE.LinearFilter;
  map.wrapS=map.wrapT=map.wrapR=THREE.RepeatWrapping; map.needsUpdate=true;
  return map;
}

export function createWhitewaterVolumes(camera:THREE.PerspectiveCamera, depth:THREE.DepthTexture,
  contacts:HullContacts, sunlight:THREE.Node<'vec3'>, ambient:THREE.Node<'vec3'>, sunDirection:THREE.Node<'vec3'>, deposit?:(point:THREE.Node<'vec2'>,energy:THREE.Node<'float'>)=>void) {
  const positions=instancedArray(COUNT,'vec4'), velocities=instancedArray(COUNT,'vec4'), profiles=instancedArray(COUNT,'vec4');
  const dt=uniform(1/60), frame=uniform(0), air=uniform(new THREE.Vector3());
  const hash=(salt:number)=>sin(float(instanceIndex).mul(127.1).add(frame.mul(311.7)).add(salt)).mul(43758.5453).fract();
  const step=Fn(()=>{
    const p=positions.element(instanceIndex), v=velocities.element(instanceIndex), profile=profiles.element(instanceIndex);
    If(p.w.lessThanEqual(0),()=>{
      const index=uint(hash(17.1).mul(float(contacts.activeCount).max(1)));
      const source=contacts.sources.element(index), launch=contacts.launches.element(index);
      const start=contacts.starts.element(index), end=contacts.ends.element(index);
      const chance=source.w.mul(launch.w).mul(start.w).mul(float(contacts.activeCount)).mul(dt).mul(12/COUNT).min(1);
      If(contacts.activeCount.greaterThan(0).and(source.w.greaterThan(0.5)).and(hash(3.1).lessThan(chance)),()=>{
        const life=source.w.mul(0.12).add(1.1).min(1.7);
        p.assign(vec4(mix(start.xyz,end.xyz,hash(13.7)).add(vec3(0,0,0.12)),life));
        v.assign(vec4(launch.xyz.mul(0.75),life));
        profile.assign(vec4(source.w,start.w,hash(9.4),0));
      });
    }).Else(()=>{
      const age=float(1).sub(p.w.div(v.w.max(0.001))).clamp(0,1);
      const drag=mix(float(0.7),float(1.7),age), gravity=mix(float(-9.81),float(-2.5),smoothstep(0.55,0.95,age));
      const equilibrium=air.add(vec3(0,0,gravity.div(drag))), decay=exp(drag.mul(dt).negate());
      const difference=v.xyz.sub(equilibrium).toVar();
      const previous=p.xyz.toVar();
      p.xyz.addAssign(equilibrium.mul(dt).add(difference.mul(float(1).sub(decay).div(drag))));
      v.xyz.assign(equilibrium.add(difference.mul(decay))); p.w.subAssign(dt);
      const height=contacts.surface(p.xy), radius=profile.x.sqrt().mul(0.24).add(0.15).add(age.mul(1.35));
      const settling=float(1).sub(smoothstep(0,radius,p.z.sub(height))).mul(v.z.negate().max(0));
      profile.w.assign(settling.mul(profile.x.sqrt()).mul(dt).mul(0.12).min(0.12));
      If(contacts.outsideHull(previous),()=>{
        for(let sample=1;sample<=8;sample++) If(contacts.outsideHull(mix(previous,p.xyz,float(sample/8))).not(),()=>{ p.w.assign(0); profile.w.assign(0); });
      }).Else(()=>{ If(contacts.outsideHull(p.xyz).not(),()=>{ p.w.assign(0); profile.w.assign(0); }); });
      If(p.z.lessThan(height.sub(radius)),()=>{p.w.assign(0);});
    });
  })().compute(COUNT);
  // Reuse half of the existing raymarch pool for dense breaking cores on the
  // actual film trajectories. A separate six-buffer dispatch preserves the
  // portable eight-binding limit of the contact emitter above.
  const followFilms=Fn(()=>{
    const index=instanceIndex.mul(contacts.mesh.count/CORES);
    const root=contacts.roots.element(index), launch=contacts.vectors.element(index), width=contacts.widths.element(index);
    const p=positions.element(instanceIndex), v=velocities.element(instanceIndex), profile=profiles.element(instanceIndex);
    const seconds=launch.w.sub(root.w).max(0), fraction=seconds.div(launch.w.max(0.001)).clamp(0,1);
    const flight=seconds.mul(float(0.72).add(fraction.mul(0.15)));
    p.assign(vec4(root.xyz.add(launch.xyz.mul(flight)).add(vec3(0,0,flight.pow2().mul(-4.905))),root.w));
    v.assign(vec4(launch.xyz.add(vec3(0,0,flight.mul(-9.81))),launch.w));
    profile.assign(vec4(width.x.mul(0.5).add(width.y.mul(2)).clamp(0,5),width.x.mul(-0.5).sub(0.001),float(index).mul(0.731).fract(),0));
  })().compute(CORES);
  // Separate from contact emission to stay within eight storage bindings.
  const settle=deposit?Fn(()=>{
    const p=positions.element(instanceIndex), profile=profiles.element(instanceIndex);
    If(profile.w.greaterThan(0),()=>{
      const radius=profile.x.sqrt().mul(0.25).add(0.3);
      for(let y=-1;y<=1;y++) for(let x=-1;x<=1;x++) {
        const weight=(x===0?2:1)*(y===0?2:1)/16;
        deposit(p.xy.add(vec2(x,y).mul(radius)),profile.w.mul(weight));
      }
      profile.w.assign(0);
    });
  })().compute(COUNT):undefined;
  const clear=Fn(()=>{positions.element(instanceIndex).assign(vec4(0)); velocities.element(instanceIndex).assign(vec4(0)); profiles.element(instanceIndex).assign(vec4(0));})().compute(COUNT);
  const p=positions.element(instanceIndex), v=velocities.element(instanceIndex), profile=profiles.element(instanceIndex);
  const age=float(1).sub(p.w.div(v.w.max(0.001))).clamp(0,1);
  const core=profile.y.lessThan(0);
  const opening=smoothstep(0,0.22,v.w.sub(p.w).max(0));
  const size=profile.x.sqrt().mul(0.24).add(0.15).add(age.mul(core.select(float(0.25),float(1.35))))
    .mul(core.select(opening.mul(0.75).add(0.25),float(1)));
  const radii=vec3(size.mul(core.select(float(0.7),float(1.7))),core.select(profile.y.negate().mul(opening).max(0.08),size),size.mul(core.select(float(0.45),mix(float(0.35),float(0.85),age))));
  const axis=normalize(core.select(v.xyz,vec3(v.xy,0)).add(vec3(0.001,0,0)));
  const side=normalize(cross(axis,vec3(0,0,1)).add(vec3(0,0.001,0)));
  const up=cross(side,axis).normalize();
  const material=new THREE.MeshBasicNodeMaterial({transparent:true,depthTest:false,depthWrite:false,side:THREE.BackSide,toneMapped:false,fog:false,
    blending:THREE.CustomBlending,blendSrc:THREE.OneFactor,blendDst:THREE.OneFactor,blendEquation:THREE.AddEquation});
  material.positionNode=p.xyz.add(axis.mul(positionLocal.x.mul(radii.x))).add(side.mul(positionLocal.y.mul(radii.y))).add(up.mul(positionLocal.z.mul(radii.z)));
  const map=densityTexture();
  const densityAt=Fn(([q]:[THREE.Node<'vec3'>])=>{
    const world=p.xyz.add(axis.mul(q.x.mul(radii.x))).add(side.mul(q.y.mul(radii.y))).add(up.mul(q.z.mul(radii.z)));
    const coarse=texture3D(map,world.mul(0.19)).level(float(0)).r;
    const fine=texture3D(map,world.mul(0.67).add(profile.z.mul(0.07))).level(float(0)).r;
    const erosion=smoothstep(float(0.12),age.mul(0.12).add(0.72),coarse.mul(0.65).add(fine.mul(0.35)));
    const fragments=smoothstep(0.38,0.6,fine).mul(0.7).add(0.3);
    // Fresh whitewater forms a folded membrane before dispersing into aerosol.
    // Neighbouring parcels share the warp, so their sheets join rather than
    // drawing a row of disconnected soft ellipsoid silhouettes.
    const membrane=float(1).sub(smoothstep(0.25,0.65,q.z.add(coarse.sub(0.5).mul(0.8)).abs()));
    const structure=core.select(fragments.mul(0.4).add(0.6),mix(fragments,float(1),smoothstep(0.35,0.9,age)).mul(mix(membrane,float(1),smoothstep(0.15,0.6,age))));
    return float(1).sub(smoothstep(0.55,1,q.length())).mul(erosion).mul(structure);
  });
  const toLocal=(q:THREE.Node<'vec3'>)=>vec3(dot(q,axis),dot(q,side),dot(q,up)).div(radii);
  const march=Fn(()=>{
    const result=vec4(0).toVar();
    If(p.w.greaterThan(0),()=>{
      const ray=normalize(positionWorld.sub(cameraPosition)), origin=toLocal(cameraPosition.sub(p.xyz)), direction=toLocal(ray);
      const a=dot(direction,direction), b=dot(origin,direction), c=dot(origin,origin).sub(1);
      const discriminant=b.pow2().sub(a.mul(c));
      If(discriminant.greaterThan(0),()=>{
        const root=discriminant.max(0).sqrt();
        const near=b.negate().sub(root).div(a).max(0);
        const viewRay=cameraViewMatrix.mul(vec4(ray,0)).xyz;
        const opaque=perspectiveDepthToViewZ(texture(depth,screenUV).r,float(camera.near),float(camera.far)).div(viewRay.z.min(-0.00001));
        const far=b.negate().add(root).div(a).min(opaque);
        const distance=far.sub(near).max(0), stride=distance.div(12);
        const fade=core.select(float(1).sub(smoothstep(0.55,1,age)),float(1).sub(age).pow2());
        const extinction=profile.x.sqrt().mul(core.select(float(2),float(0.85))).mul(smoothstep(0,0.12,age)).mul(fade).div(size.add(0.3));
        const sunlightLocal=toLocal(sunDirection).mul(size.mul(0.25));
        const cosine=dot(ray.negate(),sunDirection).clamp(-1,1), g=mix(float(0.2),float(0.6),age);
        const phase=float(1).sub(g.pow2()).div(float(1).add(g.pow2()).sub(g.mul(cosine).mul(2)).pow(1.5)).mul(0.16).min(1);
        const color=vec3(0).toVar(), tau=float(0).toVar();
        Loop(12,({i})=>{
          const t=near.add(float(i).add(0.5).mul(stride)), point=cameraPosition.add(ray.mul(t));
          const q=origin.add(direction.mul(t));
          If(point.z.greaterThan(contacts.surface(point.xy)).and(contacts.outsideHull(point)),()=>{
            const density=densityAt(q), optical=density.mul(extinction).mul(stride);
            const shadow=exp(densityAt(q.add(sunlightLocal)).add(densityAt(q.add(sunlightLocal.mul(2)))).mul(extinction).mul(size).mul(-0.25));
            const lighting=ambient.mul(0.75).add(sunlight.mul(phase.mul(0.5).add(0.25)).mul(shadow));
            color.addAssign(lighting.mul(optical)); tau.addAssign(optical);
          });
        });
        // Add optical depth and its lighting moment. Resolve once so overlapping
        // plumes do not depend on particle order or become stacked opaque cards.
        result.assign(vec4(color.mul(0.96),tau));
      });
    });
    return result;
  })();
  material.outputNode=march;
  const mesh=new THREE.InstancedMesh(new THREE.BoxGeometry(2,2,2),material,COUNT);
  for(let i=0;i<COUNT;i++) mesh.setMatrixAt(i,new THREE.Matrix4());
  mesh.frustumCulled=false;
  const scene=new THREE.Scene(); scene.add(mesh);
  const target=new THREE.RenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false});
  const image=texture(target.texture), texel=uniform(new THREE.Vector2(1,1));
  const resolve=(beauty:THREE.Node<'vec4'>)=>Fn(()=>{
    const sum=vec4(0).toVar(), weights=float(0).toVar();
    const center=perspectiveDepthToViewZ(texture(depth,screenUV).r,float(camera.near),float(camera.far));
    for(const [x,y] of [[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5],[0.5,0.5]]) {
      const point=screenUV.add(texel.mul(vec2(x,y)));
      const neighbour=perspectiveDepthToViewZ(texture(depth,point).r,float(camera.near),float(camera.far));
      const weight=float(1).div(center.sub(neighbour).abs().mul(3).add(1));
      sum.addAssign(texture(image,point).mul(weight)); weights.addAssign(weight);
    }
    const value=sum.div(weights.max(0.00001)), opacity=float(1).sub(exp(value.a.min(8).negate()));
    return vec4(beauty.rgb.mul(float(1).sub(opacity)).add(value.rgb.div(value.a.max(0.00001)).mul(opacity)),beauty.a);
  })();
  const renderSize=new THREE.Vector2();
  return {mesh,positions,velocities,profiles,target,resolve,
    reset(renderer:THREE.WebGPURenderer) {renderer.compute(clear);},
    update(renderer:THREE.WebGPURenderer,elapsed:number,wind:THREE.Vector3) {
      if(elapsed<=0) return;
      dt.value=Math.min(elapsed,0.05); frame.value++; air.value.copy(wind); renderer.compute(step); if(settle) renderer.compute(settle); renderer.compute(followFilms);
    },
    render(renderer:THREE.WebGPURenderer) {
      renderer.getDrawingBufferSize(renderSize);
      const w=Math.max(1,Math.floor(renderSize.x/2)),h=Math.max(1,Math.floor(renderSize.y/2));
      if(target.width!==w || target.height!==h) {target.setSize(w,h); texel.value.set(1/w,1/h);}
      const previous=renderer.getRenderTarget(), color=renderer.getClearColor(new THREE.Color()), alpha=renderer.getClearAlpha();
      renderer.setClearColor(0,0); renderer.setRenderTarget(target); renderer.render(scene,camera);
      renderer.setRenderTarget(previous); renderer.setClearColor(color,alpha);
    },
  };
}
