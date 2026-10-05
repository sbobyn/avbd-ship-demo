import * as THREE from 'three/webgpu';
import { atomicAdd, atomicLoad, atomicStore, cameraPosition, cameraProjectionMatrix, cameraViewMatrix, cameraWorldMatrix, convertToTexture, exp, float, fract, fwidth, Fn, If, instanceIndex, instancedArray, positionGeometry, positionWorld, screenSize, screenUV, smoothstep, storage, uint, uniform, uv, vec2, vec3, vec4, wgslFn } from 'three/tsl';

const impacts=wgslFn(`
fn stormImpacts(p:vec2f,t:f32,rain:f32)->vec2f {
  if(rain<0.001){return vec2f(0.0); }
  let cell=floor(p*1.4); var result=vec2f(0.0);
  for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
    let c=cell+vec2f(f32(x),f32(y));
    let h=fract(sin(dot(c,vec2f(127.1,311.7)))*43758.5453);
    let j=fract(sin(dot(c,vec2f(269.5,183.3)))*23718.123);
    let age=fract(t*1.7+h); let delta=p*1.4-c-vec2f(h,j);
    let r=length(delta); let front=r-age*0.7;
    let ring=sin(front*70.0)*exp(-front*front*140.0)*pow(1.0-age,3.0);
    result+=delta/max(r,0.02)*ring*0.025;
  }} return result;
}`);

// Independently authored droplet profiles inspired by the layered refraction in
// https://godotshaders.com/shader/rain-on-glass/ (Gerardo LCDF).
const lensDrops=wgslFn(`
fn lensDrops(p:vec2f,t:f32,wet:f32)->vec4f {
  var result=vec4f(0.0);
  if(wet<0.00001){return result;}
  for(var layer=0;layer<3;layer++) {
    let moving=layer>0;
    let scale=select(vec2f(76.0),select(vec2f(23.0,6.0),vec2f(37.0,9.0),layer==2),moving);
    var q=p*scale;
    let column=fract(sin(floor(q.x)*73.17+f32(layer)*19.3)*17341.23);
    if(moving){q.y-=t*(0.035+column*0.045);}
    let cell=floor(q);
    let seed=dot(cell,vec2f(127.1,311.7))+f32(layer)*97.3;
    let h=fract(sin(vec3f(seed,seed+13.7,seed+41.9))*vec3f(43758.54,23718.12,19283.17));
    // Sparse independent layers, rather than a uniform grid of identical beads.
    let occupancy=smoothstep(h.z-0.025,h.z+0.025,wet*select(0.27,0.13,moving))*min(1.0,wet*4.0);
    var d=fract(q)-(vec2f(0.5)+(h.xy-0.5)*0.48);
    let shape=select(vec2f(1.0),vec2f(1.0,3.7),moving);
    d*=shape;
    let radius=0.1+h.x*0.095;
    // A rounded head stretches slightly into a tapered neck during drainage.
    let neck=select(1.0,1.0+max(-d.y,0.0)*1.8,moving);
    let v=d*vec2f(neck,1.0)/radius;
    let r2=dot(v,v);
    let edge=max(0.0,1.0-r2);
    let phase=fract(t*0.065+h.y);
    let life=select(smoothstep(0.0,0.12,phase)*(1.0-smoothstep(0.65,1.0,phase)),1.0,moving);
    let bead=edge*edge*occupancy*life;
    let normal=-v*edge*occupancy*life;
    let trail=select(0.0,exp(-d.x*d.x*1900.0)*smoothstep(-1.1,-0.15,d.y)*(1.0-smoothstep(-0.15,0.02,d.y))*occupancy*0.16,moving);
    result+=vec4f(normal,bead,trail);
  }
  return result;
}`);

/** Camera-relative rain and a small, deterministic storm event controller. */
export function createStorm(scene:THREE.Scene) {
  const intensity=uniform(0),clock=uniform(0),flash=uniform(0);
  const lensWetness=instancedArray(1,'float'), exposure=instancedArray(1,'uint').toAtomic();
  const wetRead=storage(lensWetness.value,'float',1).toReadOnly().element(0);
  const lensCamera=uniform(new THREE.Vector3()), lensForward=uniform(new THREE.Vector3(0,0,-1)), lensDt=uniform(0);
  let wetting:THREE.ComputeNode | undefined, settle:THREE.ComputeNode | undefined;
  const clearExposure=Fn(()=>{atomicStore(exposure.element(0),uint(0));})().compute(1);
  let seed=9317;
  const random=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return (seed>>>0)/4294967296;};
  const wind=uniform(new THREE.Vector3());
  const count=18000,seeds=instancedArray(Float32Array.from({length:count*4},random),'vec4');
  const rainMaterial=new THREE.MeshBasicNodeMaterial({transparent:true,depthWrite:false,color:'#a9c6da'});
  const rainSeed=seeds.element(instanceIndex);
  const rainExtent=instanceIndex.mod(3).equal(0).select(float(40),instanceIndex.mod(3).equal(1).select(float(90),float(180)));
  const fall=rainSeed.w.mul(6).add(8);
  const velocity=vec3(wind.xy.mul(rainSeed.w.mul(0.25).add(0.75)),fall.negate());
  // World-stable toroidal volumes retain parallax while the ship/camera moves.
  const rainCenter=cameraPosition.add(fract(rainSeed.xyz.add(velocity.mul(clock).sub(cameraPosition).div(vec3(rainExtent,rainExtent,36))))
    .sub(0.5).mul(vec3(rainExtent,rainExtent,36)));
  const depth=cameraViewMatrix.mul(vec4(rainCenter,1)).z.abs();
  const physicalWidth=rainSeed.w.pow2().mul(0.006).add(0.0015);
  const resolvedWidth=depth.mul(0.75).div(screenSize.y.mul(cameraProjectionMatrix.mul(vec4(0,1,0,0)).y.abs()).max(1));
  const width=physicalWidth.max(resolvedWidth);
  rainMaterial.positionNode=Fn(()=>{
    const right=cameraWorldMatrix.mul(vec4(1,0,0,0)).xyz;
    // 10–28 ms shutter exposure gives short variable streaks at real fall speeds.
    const shutter=rainSeed.x.mul(0.018).add(0.01);
    return rainCenter.add(right.mul(positionGeometry.x.mul(width)))
      .sub(velocity.mul(positionGeometry.y.mul(shutter)));
  })();
  const edge=fwidth(uv().x).max(0.08);
  const crossSection=float(1).sub(smoothstep(float(0.5).sub(edge),float(0.5),uv().x.sub(0.5).abs()));
  const ends=uv().y.mul(float(1).sub(uv().y)).mul(4).pow(0.65);
  const distanceFade=exp(depth.mul(-0.009)).mul(smoothstep(0.35,2,depth));
  rainMaterial.opacityNode=intensity.mul(crossSection).mul(ends).mul(distanceFade)
    .mul(physicalWidth.div(width)).mul(rainSeed.y.mul(0.24).add(0.22)).mul(flash.mul(1.5).add(1));
  const rain=new THREE.Mesh(new THREE.PlaneGeometry(1,1),rainMaterial); rain.count=count; rain.frustumCulled=false; scene.add(rain);
  const lightning=new THREE.Group();scene.add(lightning);
  const core=new THREE.MeshBasicMaterial({color:'#e9f5ff',transparent:true,blending:THREE.AdditiveBlending,depthWrite:false});
  const halo=new THREE.MeshBasicMaterial({color:'#608fff',transparent:true,opacity:0.2,blending:THREE.AdditiveBlending,depthWrite:false});
  let timer=3,age=100,manual=false;
  const sound={strike:0,strikePosition:new THREE.Vector3()};
  let pulses:[number,number,number][]=[];
  const channel=(points:THREE.Vector3[],radius:number,material:THREE.Material)=>{
    const path=new THREE.CurvePath<THREE.Vector3>();
    for(let i=1;i<points.length;i++) path.add(new THREE.LineCurve3(points[i-1],points[i]));
    return new THREE.Mesh(new THREE.TubeGeometry(path,points.length*3,radius,4,false),material);
  };
  const strike=(origin:THREE.Vector3,forced=false)=>{
    for(const child of [...lightning.children]) {lightning.remove(child);(child as THREE.Mesh).geometry.dispose();}
    const height=190+random()*130,count=23+Math.floor(random()*16);
    const leanX=(random()-0.5)*70,leanY=(random()-0.5)*50;
    const points=Array.from({length:count},(_,i)=>{
      const t=i/(count-1);
      return new THREE.Vector3(leanX*t+(random()-0.5)*14,leanY*t+(random()-0.5)*12,height*(1-t));
    });
    const radius=0.07+random()*0.1;
    lightning.add(channel(points,radius,core),channel(points,radius*4,halo));
    const forks=3+Math.floor(random()*7);
    for(let fork=0;fork<forks;fork++) {
      const start=points[3+Math.floor(random()*(count-8))];
      const direction=new THREE.Vector3((random()-0.5)*55,(random()-0.5)*55,-20-random()*60);
      const branch=Array.from({length:8},(_,i)=>start.clone().addScaledVector(direction,i/7)
        .add(i?new THREE.Vector3((random()-0.5)*6,(random()-0.5)*6,0):new THREE.Vector3()));
      lightning.add(channel(branch,radius*0.45,core));
    }
    // Place forced strikes in front of the camera so the demo button is useful.
    const forward=new THREE.Vector3(lensForward.value.x,lensForward.value.y,0).normalize();
    const right=new THREE.Vector3(-forward.y,forward.x,0);
    lightning.position.copy(forced?lensCamera.value:origin);
    if(forced) lightning.position.addScaledVector(forward,180).addScaledVector(right,(random()-0.5)*100);
    else lightning.position.add(new THREE.Vector3((random()-0.5)*360,120+random()*220,0));
    lightning.position.z=0;
    sound.strike++; sound.strikePosition.copy(lightning.position);
    pulses=Array.from({length:2+Math.floor(random()*3)},(_,i)=>[i===0?0:i*0.11+random()*0.025,i===0?0.085:0.055+random()*0.025,i===0?1:0.55+random()*0.4]);
    age=0;timer=8+random()*17;manual=forced;
  };
  const lens=(image:THREE.Node<'vec4'>)=>{
    const input=convertToTexture(image);
    const aspect=screenSize.x.div(screenSize.y);
    const drops=lensDrops(screenUV.sub(0.5).mul(vec2(aspect,1)),clock,wetRead) as THREE.Node<'vec4'>;
    const offset=drops.xy.mul(vec2(float(0.0022).div(aspect),0.0022));
    const refractedUV=screenUV.add(offset).clamp(0.001,0.999);
    const refracted=input.sample(refractedUV);
    // Only wet pixels soften; the dry scene stays sharp and retains its colour.
    const blur=drops.z.mul(0.12).add(drops.w.mul(0.3)).clamp(0,0.2);
    const pixel=vec2(1).div(screenSize);
    const softened=input.sample(refractedUV.add(pixel.mul(1.5))).rgb
      .add(input.sample(refractedUV.sub(pixel.mul(1.5))).rgb).mul(0.5);
    return vec4(refracted.rgb.mul(float(1).sub(blur)).add(softened.mul(blur)),1);
  };
  return {intensity,flash,lens,lensWetness,wind,sound,
    triggerLightning(origin:THREE.Vector3) {strike(origin,true);},
    setLensContacts(contacts:{count:number;activeCount:THREE.Node<'uint'>;sources:THREE.StorageBufferNode<'vec4'>;launches:THREE.StorageBufferNode<'vec4'>},surface:(p:THREE.Node<'vec2'>)=>THREE.Node<'float'>) {
      wetting=Fn(()=>{
        If(instanceIndex.lessThan(contacts.activeCount),()=>{
          const source=contacts.sources.element(instanceIndex), launch=contacts.launches.element(instanceIndex);
          const delta=source.xyz.sub(lensCamera), distance=delta.length();
          const facing=smoothstep(-0.2,0.5,delta.dot(lensForward).div(distance.max(0.1)));
          const close=exp(distance.pow2().mul(-1/24));
          const waterDistance=lensCamera.z.sub(surface(lensCamera.xy)).abs();
          const low=float(1).sub(smoothstep(0.5,6,waterDistance));
          const dose=source.w.mul(launch.w).mul(close).mul(facing).mul(low).min(8);
          atomicAdd(exposure.element(0),uint(dose.mul(4096)));
        });
      })().compute(contacts.count);
      settle=Fn(()=>{
        const spray=float(1).sub(exp(float(atomicLoad(exposure.element(0))).div(4096).mul(-0.3)));
        const target=intensity.mul(0.38).max(spray.mul(0.75));
        const old=lensWetness.element(0);
        const rate=target.greaterThan(old).select(float(1.8),float(0.12));
        old.addAssign(target.sub(old).mul(float(1).sub(exp(lensDt.mul(rate).negate()))));
      })().compute(1);
    },
    updateLens(renderer:THREE.WebGPURenderer) {if(wetting&&settle) renderer.compute([clearExposure,wetting,settle]);},
    setBoundary(surface:(p:THREE.Node<'vec2'>)=>THREE.Node<'float'>,outside:(p:THREE.Node<'vec3'>)=>THREE.Node<'bool'>) {
      rainMaterial.maskNode=outside(positionWorld).and(positionWorld.z.greaterThan(surface(positionWorld.xy)));
    },ripples:(p:THREE.Node<'vec2'>)=>(impacts(p,clock,intensity) as THREE.Node<'vec2'>).mul(intensity),
    update(dt:number,origin:THREE.Vector3,camera?:THREE.Camera){
      lensDt.value=Math.max(0,Math.min(0.1,dt));
      if(camera) {camera.getWorldPosition(lensCamera.value);camera.getWorldDirection(lensForward.value);}
      clock.value+=dt;age+=dt;
      if(intensity.value>0.65) {timer-=dt;if(timer<=0) strike(origin);} else timer=3;
      const pulse=pulses.reduce((value,[start,duration,strength])=>Math.max(value,age>=start&&age<start+duration?strength:0),Math.exp(-age*20)*0.18);
      flash.value=Math.min(1,pulse)*(manual?1:intensity.value); lightning.visible=flash.value>0.015;
      core.opacity=Math.min(1,flash.value*2);halo.opacity=flash.value*0.2;
      rain.visible=intensity.value>0.001;
    },
  };
}
