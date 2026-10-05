import * as THREE from 'three/webgpu';
import { atomicAdd, atomicLoad, atomicStore, cameraViewMatrix, cross, dFdx, dFdy, exp, float, Fn, If, instanceIndex, instancedArray, mix, positionLocal, positionWorld, sin, smoothstep, storage, texture, uint, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';
import { contactTriangles, waterlineEnvelope } from './hull.ts';
import { createImpactSheets } from './impact.ts';
import { createHullExclusion } from './exclusion.ts';
import { whitewaterDetail } from './whitewater.ts';

// The visible water geometry is rasterized to height, so choppiness, triangle
// interpolation and wake displacement agree at contact. This is a visual boundary
// coupling, not a volumetric fluid solve. Height resolves 8.6 cm; whitewater coverage resolves 4.3 cm per texel.
const SIZE=512, FOAM_SIZE=1024, EXTENT=44, CAPACITY=65536;
export function createHullContacts() {
  const exclusion=createHullExclusion();
  const origin=uniform(new THREE.Vector2()), oldOrigin=uniform(new THREE.Vector2());
  const inverseTransform=uniform(new THREE.Matrix4());
  const transform=uniform(new THREE.Matrix4()), oldTransform=uniform(new THREE.Matrix4());
  const clock=uniform(0), dt=uniform(1/60), ready=uniform(0), fieldReady=uniform(0);
  const makeTarget=(size=SIZE)=>new THREE.RenderTarget(size,size,{type:THREE.HalfFloatType,minFilter:THREE.LinearMipmapLinearFilter,magFilter:THREE.LinearFilter,generateMipmaps:true});
  let heightTarget=makeTarget(), oldHeightTarget=makeTarget(), fieldTarget=makeTarget(FOAM_SIZE), oldFieldTarget=makeTarget(FOAM_SIZE);
  const heightTexture=texture(heightTarget.texture), previousHeightTexture=texture(oldHeightTarget.texture);
  const fieldTexture=texture(fieldTarget.texture), previousFieldTexture=texture(oldFieldTarget.texture);
  const coordinates=(p:THREE.Node<'vec2'>,previous=false)=>p.sub(previous?oldOrigin:origin).div(EXTENT).add(0.5);
  const surface=(p:THREE.Node<'vec2'>,previous=false)=>texture(previous?previousHeightTexture:heightTexture,coordinates(p,previous)).level(float(0)).r;
  // Match the pressure footprint to the 0.5 m wake grid. Sampling the 4 cm
  // contact map too sharply aliases thin rudder/hull edges into height impulses.
  const sample=(p:THREE.Node<'vec2'>,previous=false,coarse=false)=>{
    const q=coordinates(p,previous);
    return fieldReady.greaterThan(0).and(q.x.greaterThan(0)).and(q.x.lessThan(1)).and(q.y.greaterThan(0)).and(q.y.lessThan(1))
      .select(texture(previous?previousFieldTexture:fieldTexture,q).level(float(coarse?Math.log2(0.5*FOAM_SIZE/EXTENT):0)),vec4(0));
  };
  // Filter coverage before its nonlinear shading response. Four overlapping
  // mip footprints remove raster-cell corners without blurring the simulation.
  const sampleFoam=(p:THREE.Node<'vec2'>)=>{
    // The field is written in world XY. Material texture sampling applies a
    // render-target Y flip; undo it here to match the compute-side field reads.
    const q=coordinates(p).flipY();
    // Use the actual anisotropic pixel footprint. An isotropic world-axis
    // square grows sideways at grazing angles and makes the hull collar swim.
    const dx=dFdx(q), dy=dFdy(q);
    const minor=dx.length().min(dy.length()).mul(FOAM_SIZE).max(2);
    const lod=minor.log2().clamp(1,8);
    const coverage=texture(fieldTexture,q.add(dx.add(dy).mul(0.25))).level(lod).y
      .add(texture(fieldTexture,q.add(dx.sub(dy).mul(0.25))).level(lod).y)
      .add(texture(fieldTexture,q.sub(dx.add(dy).mul(0.25))).level(lod).y)
      .add(texture(fieldTexture,q.sub(dx.sub(dy).mul(0.25))).level(lod).y).mul(0.25);
    return fieldReady.greaterThan(0).and(q.x.greaterThan(0)).and(q.x.lessThan(1)).and(q.y.greaterThan(0)).and(q.y.lessThan(1))
      .select(coverage.max(0),float(0));
  };
  const counter=instancedArray(1,'uint').toAtomic();
  const countBuffer=instancedArray(1,'uint');
  const activeCount=storage(countBuffer.value,'uint',1).toReadOnly().element(0);
  const sources=instancedArray(CAPACITY,'vec4'), launches=instancedArray(CAPACITY,'vec4');
  const starts=instancedArray(CAPACITY,'vec4'), ends=instancedArray(CAPACITY,'vec4');
  const sheets=createImpactSheets(sources,launches,activeCount,starts,ends,surface,exclusion.outsideSplash);
  const curtainGeometry=new THREE.PlaneGeometry(1,1,1,6);
  const drawBuffer=new THREE.IndirectStorageBufferAttribute(new Uint32Array([curtainGeometry.index!.count,0,0,0,0]),1);
  const draw=storage(drawBuffer,'uint',5);
  curtainGeometry.setIndirect(drawBuffer);
  const clearCount=Fn(()=>{ atomicStore(counter.element(0),uint(0)); countBuffer.element(0).assign(uint(0)); draw.element(1).assign(uint(0)); })().compute(1);
  const publish=Fn(()=>{ countBuffer.element(0).assign(atomicLoad(counter.element(0))); draw.element(1).assign(atomicLoad(counter.element(0))); })().compute(1);
  const clear=Fn(()=>{ sources.element(instanceIndex).assign(vec4(0)); launches.element(instanceIndex).assign(vec4(0)); starts.element(instanceIndex).assign(vec4(0)); ends.element(instanceIndex).assign(vec4(0)); })().compute(CAPACITY);
  let contactStep:THREE.ComputeNode | undefined;
  const waterScene=new THREE.Scene(), fieldScene=new THREE.Scene();
  const camera=new THREE.OrthographicCamera(-EXTENT/2,EXTENT/2,EXTENT/2,-EXTENT/2,0.1,100);
  camera.up.set(0,1,0);
  let waterMesh:THREE.Mesh | undefined, hullMesh:THREE.Mesh | undefined;
  const hullMaterial=new THREE.MeshBasicNodeMaterial({side:THREE.DoubleSide,toneMapped:false,fog:false});
  const submerged=surface(positionWorld.xy).sub(positionWorld.z);
  hullMaterial.maskNode=submerged.greaterThan(0);
  hullMaterial.outputNode=vec4(submerged.mul(0.3).clamp(0,0.7),0,0,0);
  const ribbonMaterial=new THREE.MeshBasicNodeMaterial({transparent:true,depthWrite:false,depthTest:false,side:THREE.DoubleSide,toneMapped:false,fog:false,
    blending:THREE.CustomBlending,blendSrc:THREE.OneFactor,blendDst:THREE.OneFactor,blendEquation:THREE.AddEquation});
  const a=starts.element(instanceIndex), b=ends.element(instanceIndex), source=sources.element(instanceIndex), launch=launches.element(instanceIndex);
  const direction=b.xy.sub(a.xy).add(vec2(0.00001,0)).normalize();
  const width=float(0.14).add(source.w.sqrt().mul(0.035)).add(a.w.mul(source.w.sqrt()).mul(0.24));
  ribbonMaterial.positionNode=mix(a.xyz,b.xyz,uv().x).add(vec3(direction.y.negate(),direction.x,0).mul(positionLocal.y.mul(width)));
  // A constant-intensity strip ends at raster coverage, so sub-texel motion
  // toggles whole bright cells. Taper energy and its flow moment together.
  const ribbonEnergy=source.w.mul(float(1).sub(smoothstep(0.1,0.5,uv().y.sub(0.5).abs()))).mul(1.5);
  ribbonMaterial.outputNode=instanceIndex.lessThan(activeCount).select(vec4(0,ribbonEnergy,launch.xy.mul(ribbonEnergy)),vec4(0));
  const ribbons=new THREE.InstancedMesh(new THREE.PlaneGeometry(1,1),ribbonMaterial,CAPACITY);
  ribbons.frustumCulled=false; ribbons.renderOrder=1; fieldScene.add(ribbons);
  for(let i=0;i<CAPACITY;i++) ribbons.setMatrixAt(i,new THREE.Matrix4());
  // A connected low curtain follows each exact waterline segment. Detached
  // films handle the larger bursts; indirect drawing visits only live segments.
  const curtainMaterial=new THREE.MeshStandardNodeMaterial({transparent:true,depthWrite:false,side:THREE.DoubleSide,roughness:0.5,metalness:0});
  const root=mix(a.xyz,b.xyz,uv().x), v=uv().y;
  const pulse=sin(root.x.mul(2.7).add(root.y.mul(3.1)).sub(clock.mul(7))).mul(0.15).add(0.85);
  // Sample the continuous pressure field instead of giving each tiny hull
  // triangle a different fan direction and height (visible white comb seams).
  const energy=sample(root.xy,false,true).y.clamp(0,5);
  const gradient=vec2(sample(root.xy.add(vec2(0.4,0)),false,true).x.sub(sample(root.xy.sub(vec2(0.4,0)),false,true).x),
    sample(root.xy.add(vec2(0,0.4)),false,true).x.sub(sample(root.xy.sub(vec2(0,0.4)),false,true).x));
  const outward=gradient.negate().add(vec2(0.00001,0)).normalize();
  const reach=energy.sqrt().mul(0.28).add(0.1).mul(a.w);
  const fan=outward.mul(v.mul(reach));
  const point=root.xy.add(fan);
  // Keep the low froth roll connected to the displaced water at both edges.
  // Ballistic films supply the airborne jets instead of penetrating the prow.
  const height=surface(point).add(sin(v.mul(Math.PI)).mul(energy.sqrt().mul(0.08).add(0.025)).mul(pulse));
  curtainMaterial.positionNode=vec3(point,height.add(0.025));
  const filmNormal=cross(dFdx(positionWorld),dFdy(positionWorld)).normalize();
  curtainMaterial.normalNode=cameraViewMatrix.mul(vec4(filmNormal,0)).xyz;
  const film=whitewaterDetail(vec2(root.x.mul(0.53).add(root.y.mul(0.71)).fract(),v.mul(0.85)),uint(root.x.add(root.y).mul(0.17).floor().abs()));
  const density=float(1).sub(exp(energy.mul(-0.65))).mul(a.w);
  const tear=smoothstep(v.pow2().mul(0.38),v.pow2().mul(0.38).add(0.18),film.r);
  curtainMaterial.colorNode=vec3(0.87,0.93,0.9).mul(film.g.mul(0.2).add(0.8));
  const coverage=float(1).sub(exp(sampleFoam(positionWorld.xy).mul(-0.65)));
  curtainMaterial.maskNode=exclusion.outsideSplash(positionWorld);
  curtainMaterial.opacityNode=density.min(coverage).mul(tear).mul(float(1).sub(v).pow2()).mul(0.16);
  const curtain=new THREE.InstancedMesh(curtainGeometry,curtainMaterial,CAPACITY);
  curtain.frustumCulled=false;
  for(let i=0;i<CAPACITY;i++) curtain.setMatrixAt(i,new THREE.Matrix4());
  let initialized=false, lastTime=0;
  const capture=(renderer:THREE.WebGPURenderer,scene:THREE.Scene,target:THREE.RenderTarget)=>{
    const previous=renderer.getRenderTarget(), color=renderer.getClearColor(new THREE.Color()), alpha=renderer.getClearAlpha();
    renderer.setClearColor(0,0); renderer.setRenderTarget(target); renderer.render(scene,camera); renderer.setRenderTarget(previous); renderer.setClearColor(color,alpha);
  };
  return {...sheets,curtain,drawBuffer,sources,launches,starts,ends,activeCount,counter,count:CAPACITY,
    localPoint:(p:THREE.Node<'vec3'>)=>inverseTransform.mul(vec4(p,1)).xyz,
    sample:(p:THREE.Node<'vec2'>)=>sample(p), sampleFoam, outsideHull:exclusion.outsideSplash, outsideSolid:exclusion.outside, outsideWater:exclusion.outsideWater,
    sampleWake:(p:THREE.Node<'vec2'>,previous=false)=>sample(p,previous,true), surface,
    setWater(geometry:THREE.BufferGeometry,displacement:THREE.Node<'vec3'>) {
      const material=new THREE.MeshBasicNodeMaterial({side:THREE.DoubleSide,toneMapped:false,fog:false});
      material.positionNode=displacement; material.outputNode=vec4(positionWorld.z,0,0,1);
      waterMesh=new THREE.Mesh(geometry,material); waterMesh.frustumCulled=false; waterScene.add(waterMesh);
    },
    setHull(geometries:THREE.BufferGeometry[]) {
      exclusion.setHull(geometries);
      const normals:number[]=[];
      const triangles=contactTriangles(geometries,0.5,normals), count=triangles.length/12;
      const envelope=waterlineEnvelope(geometries);
      if(count>CAPACITY) throw new Error('Hull contact triangle capacity exceeded.');
      const input=instancedArray(triangles,'vec4');
      const read=storage(input.value,'vec4',count*3).toReadOnly();
      const normalInput=instancedArray(new Float32Array(normals),'vec4');
      const readNormals=storage(normalInput.value,'vec4',count*3).toReadOnly();
      const geometry=new THREE.BufferGeometry();
      const xyz=new Float32Array(count*9);
      for(let i=0;i<count*3;i++) xyz.set(triangles.subarray(i*4,i*4+3),i*3);
      geometry.setAttribute('position',new THREE.BufferAttribute(xyz,3));
      hullMesh=new THREE.Mesh(geometry,hullMaterial); hullMesh.matrixAutoUpdate=false; hullMesh.frustumCulled=false; fieldScene.add(hullMesh);
      contactStep=Fn(()=>{
        const va=read.element(instanceIndex.mul(3)), vb=read.element(instanceIndex.mul(3).add(1)), vc=read.element(instanceIndex.mul(3).add(2));
        const a=transform.mul(va).xyz, b=transform.mul(vb).xyz, c=transform.mul(vc).xyz;
        const na=readNormals.element(instanceIndex.mul(3)).xyz, nb=readNormals.element(instanceIndex.mul(3).add(1)).xyz, nc=readNormals.element(instanceIndex.mul(3).add(2)).xyz;
        const da=surface(a.xy).sub(a.z), db=surface(b.xy).sub(b.z), dc=surface(c.xy).sub(c.z);
        const first=vec3(0).toVar(), second=vec3(0).toVar(), localFirst=vec3(0).toVar(), localSecond=vec3(0).toVar(), hits=uint(0).toVar();
        const firstNormal=vec3(0).toVar(), secondNormal=vec3(0).toVar();
        for(const [p,q,lp,lq,np,nq,dp,dq] of [[a,b,va.xyz,vb.xyz,na,nb,da,db],[b,c,vb.xyz,vc.xyz,nb,nc,db,dc],[c,a,vc.xyz,va.xyz,nc,na,dc,da]]) {
          If(dp.greaterThanEqual(0).select(float(1),float(0)).notEqual(dq.greaterThanEqual(0).select(float(1),float(0))),()=>{
            // Secant refinement follows the actual rasterized water triangles;
            // flat water resolves exactly on the first interpolation.
            const low=float(0).toVar(), high=float(1).toVar(), dl=dp.toVar(), dh=dq.toVar(), t=dp.div(dp.sub(dq)).clamp(0,1).toVar();
            for(let i=0;i<3;i++) {
              const point=mix(p,q,t), residual=surface(point.xy).sub(point.z);
              If(residual.mul(dl).greaterThanEqual(0),()=>{ low.assign(t); dl.assign(residual); }).Else(()=>{ high.assign(t); dh.assign(residual); });
              t.assign(dl.abs().add(dh.abs()).greaterThan(0.000001).select(mix(low,high,dl.abs().div(dl.abs().add(dh.abs()).max(0.000001))),t));
            }
            If(hits.equal(0),()=>{ first.assign(mix(p,q,t)); localFirst.assign(mix(lp,lq,t)); firstNormal.assign(mix(np,nq,t)); }).Else(()=>{ second.assign(mix(p,q,t)); localSecond.assign(mix(lp,lq,t)); secondNormal.assign(mix(np,nq,t)); });
            hits.addAssign(1);
          });
        }
        If(hits.equal(2).and(first.distance(second).greaterThan(0.0001)),()=>{
          const index=atomicAdd(counter.element(0),uint(1));
          const center=first.add(second).mul(0.5), localCenter=localFirst.add(localSecond).mul(0.5);
          const old=oldTransform.mul(vec4(localCenter,1)).xyz;
          const velocity=ready.greaterThan(0).select(center.sub(old).div(dt),vec3(0));
          const smooth=transform.mul(vec4(firstNormal.add(secondNormal),0)).xyz;
          const normal=smooth.length().greaterThan(0.0001).select(smooth.normalize(),cross(b.sub(a),c.sub(a)).normalize());
          // Filter the silhouette pressure gradient over small hull bevels. The
          // original vertex normal remains the fallback before a field exists.
          const gradient=vec2(sample(old.xy.add(vec2(0.4,0)),true,true).x.sub(sample(old.xy.sub(vec2(0.4,0)),true,true).x),
            sample(old.xy.add(vec2(0,0.4)),true,true).x.sub(sample(old.xy.sub(vec2(0,0.4)),true,true).x));
          const boundary=gradient.negate().div(gradient.length().max(0.00001));
          const outward=vec3(mix(normal.xy,boundary,smoothstep(0.002,0.06,gradient.length()).mul(0.85)),0).add(vec3(0.00001,0,0)).normalize();
          const rising=ready.greaterThan(0).select(old.z.sub(surface(old.xy,true)).div(dt).clamp(0,6),float(0));
          const normalSpeed=velocity.dot(outward), cutting=normalSpeed.clamp(0,6);
          const separation=normalSpeed.negate().clamp(0,6), shear=velocity.xy.length().min(6);
          const energy=ready.greaterThan(0).select(cutting.pow2().mul(0.075).add(rising.pow2().mul(0.38)).add(separation.pow2().mul(0.015)).add(shear.pow2().mul(0.004)).min(5),float(0));
          const strength=rising.add(cutting.mul(0.35)).min(5);
          // Local waterline extent identifies the forward shoulder even through
          // turns. Closing speed gates its curtain; a reversing stern stays quiet.
          const forward=smoothstep(envelope.stern+(envelope.bow-envelope.stern)*0.55,envelope.stern+(envelope.bow-envelope.stern)*0.9,localCenter.x);
          const forwardSpeed=velocity.dot(transform.mul(vec4(1,0,0,0)).xyz.normalize());
          const bow=forward.mul(smoothstep(0.4,2,forwardSpeed));
          const impact=rising.div(4).clamp(0,1);
          const fan=strength.mul(0.45).add(0.65).add(bow.mul(cutting.mul(0.3).add(rising.mul(0.35))));
          sources.element(index).assign(vec4(center,energy));
          launches.element(index).assign(vec4(velocity.xy.mul(0.3).add(outward.xy.mul(fan)),strength.mul(0.7).add(1.2).add(bow.mul(rising.mul(0.5).add(cutting.mul(0.2)))),first.distance(second)));
          starts.element(index).assign(vec4(first,bow)); ends.element(index).assign(vec4(second,impact));
        });
      })().compute(count);
      initialized=false; fieldReady.value=0;
    },
    reset(renderer:THREE.WebGPURenderer) { renderer.compute([clearCount,clear]); sheets.reset(renderer); initialized=false; ready.value=0; fieldReady.value=0; },
    update(renderer:THREE.WebGPURenderer,elapsed:number,time:number,matrix:THREE.Matrix4) {
      if(elapsed<=0 || !contactStep || !waterMesh || !hullMesh) return;
      const jump=Math.hypot(matrix.elements[12]-transform.value.elements[12],matrix.elements[13]-transform.value.elements[13]);
      if(initialized && (time<lastTime || elapsed>0.5 || jump>5)) { sheets.reset(renderer); initialized=false; }
      lastTime=time; clock.value=time; dt.value=Math.min(elapsed,0.1); ready.value=initialized?1:0;
      oldOrigin.value.copy(origin.value); oldTransform.value.copy(transform.value);
      exclusion.update(matrix); inverseTransform.value.copy(matrix).invert();
      origin.value.set(matrix.elements[12],matrix.elements[13]); transform.value.copy(matrix);
      [heightTarget,oldHeightTarget]=[oldHeightTarget,heightTarget]; [fieldTarget,oldFieldTarget]=[oldFieldTarget,fieldTarget];
      heightTexture.value=heightTarget.texture; previousHeightTexture.value=oldHeightTarget.texture;
      fieldTexture.value=fieldTarget.texture; previousFieldTexture.value=oldFieldTarget.texture;
      camera.position.set(origin.value.x,origin.value.y,50); camera.lookAt(origin.value.x,origin.value.y,0);
      waterMesh.position.set(origin.value.x,origin.value.y,-0.45); hullMesh.matrix.copy(matrix);
      capture(renderer,waterScene,heightTarget);
      if(!initialized) { oldOrigin.value.copy(origin.value); oldTransform.value.copy(matrix); capture(renderer,waterScene,oldHeightTarget); }
      renderer.compute([clearCount,contactStep,publish]);
      capture(renderer,fieldScene,fieldTarget);
      if(!initialized) capture(renderer,fieldScene,oldFieldTarget);
      sheets.update(renderer,elapsed,time); initialized=true; fieldReady.value=1;
    },
  };
}
export type HullContacts=ReturnType<typeof createHullContacts>;
