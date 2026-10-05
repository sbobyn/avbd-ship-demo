import assert from 'node:assert/strict';
import { createHullExclusion } from '../src/ship/ocean/exclusion.ts';
import * as THREE from 'three/webgpu';
import { atomicAdd, Fn, If, float, instanceIndex, instancedArray, uniform, positionLocal, uint, vec2, vec3, vec4 } from 'three/tsl';
import { createHullContacts } from '../src/ship/ocean/contact.ts';
import { createBowImpacts, createImpactSheets } from '../src/ship/ocean/impact.ts';
import { createShipSpray } from '../src/ship/ocean/spray.ts';
import { createWhitewaterVolumes } from '../src/ship/ocean/volume.ts';
import { createWake } from '../src/ship/ocean/wake.ts';
import { createWhitecaps } from '../src/ship/ocean/whitecaps.ts';
import { createOceanGPU, disposeOceanGPU, updateOceanGPU } from '../src/ship/ocean/fft-gpu.ts';
import { CASCADES } from '../src/ship/ocean/model.ts';
import { createStorm } from '../src/ship/storm.ts';
import { createWeather } from '../src/ship/weather.ts';
import { gpuTest } from './device.ts';

// Exercise the actual composed material, not just the FFT buffers: diagnostic
// branches once left shared normal calculations out of the beauty shader.
gpuTest('ship water: clear sea retains wave detail without white coverage or brightness jumps', async device => {
  const saved = ['self', 'document'].map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)] as const);
  const gpuDescriptor = Object.getOwnPropertyDescriptor(navigator, 'gpu');
  let nextFrame: ((time: number) => void) | undefined;
  Object.defineProperty(globalThis, 'self', { configurable: true, value: {
    requestAnimationFrame: (fn: (time: number) => void) => { nextFrame=fn; return 0; }, cancelAnimationFrame() {},
  } });
  Object.defineProperty(globalThis, 'document', { configurable: true, value: { body: { dataset: {} } } });
  Object.defineProperty(navigator, 'gpu', { configurable: true, value: { getPreferredCanvasFormat: () => 'bgra8unorm' } });
  const canvas = { width: 256, height: 160, style: {}, addEventListener() {}, removeEventListener() {}, getContext() { return { configure() {} }; } };
  const renderer = new THREE.WebGPURenderer({ canvas: canvas as unknown as HTMLCanvasElement, device });
  const target = new THREE.RenderTarget(256,160,{type:THREE.FloatType});
  try {
    await renderer.init();
    renderer.setSize(256,160,false); renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.shadowMap.enabled=true;
    const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(38,256/160,0.1,8000);
    camera.up.set(0,0,1); camera.position.set(33,-45,16); camera.lookAt(0,0,9);
    const sun=new THREE.DirectionalLight(); sun.castShadow=true; scene.add(sun);
    const fill=new THREE.HemisphereLight(), hull=new THREE.Group(); scene.add(fill,hull);
    const weather=createWeather(renderer,scene,camera,sun,fill,hull);
    weather.setQuality(false); renderer.setRenderTarget(target);
    let previous: number[] | undefined, motion=0;
    for(let frame=0;frame<12;frame++) {
      nextFrame!(frame*1000/60); weather.update(frame/60,7,0.15); weather.render();
      const pixels=await renderer.readRenderTargetPixelsAsync(target,0,0,256,160) as Float32Array;
      const luminance: number[]=[];
      for(let y=100;y<150;y++) for(let x=10;x<246;x++) {
        const i=(y*256+x)*4;
        assert.ok(Number.isFinite(pixels[i]+pixels[i+1]+pixels[i+2]));
        luminance.push(pixels[i]*0.2126+pixels[i+1]*0.7152+pixels[i+2]*0.0722);
      }
      assert.ok(luminance.filter(v=>v>0.8).length/luminance.length<0.01,'clear sea must not become a white sheet');
      const mean=luminance.reduce((a,b)=>a+b,0)/luminance.length;
      const variance=luminance.reduce((a,b)=>a+(b-mean)**2,0)/luminance.length;
      assert.ok(variance>0.00005,'water must retain spatial wave shading');
      if(previous) {
        const change=luminance.reduce((a,b,i)=>a+Math.abs(b-previous![i]),0)/luminance.length;
        assert.ok(change<0.025,`water lighting jumped by ${change}`); motion+=change;
      }
      previous=luminance;
    }
    assert.ok(motion>0.0001,'wave shading must evolve rather than freeze');

    // Render the actual composed wake behind a moving hull and thin rudder.
    // Check temporal coverage separately from reflection/specular variation.
    const testHull=new THREE.BoxGeometry(14,4,3), rudder=new THREE.BoxGeometry(1,0.12,2).translate(-7.4,0,-0.5);
    weather.setHull({stern:-7.9,bow:7,centerY:0,beam:2},[testHull,rudder]);
    weather.inspectWake('foam');
    let wakePixels:Float32Array|undefined, peakChange=0, visibleCoverage=0;
    for(let frame=0;frame<90;frame++) {
      const t=0.2+frame/60, x=frame*6/60;
      hull.position.set(x,0,Math.sin(t*2)*0.1);
      camera.position.set(x-19,-10,7); camera.lookAt(x-9,0,-0.4);
      nextFrame!(t*1000); weather.update(t,7,0.15);
      if(frame<60) continue;
      weather.render();
      const pixels=await renderer.readRenderTargetPixelsAsync(target,0,0,256,160) as Float32Array;
      assert.ok(pixels.every(Number.isFinite),'moving stern wake renders finite pixels');
      let change=0, coverage=0;
      for(let i=0;i<pixels.length;i+=4) {
        coverage+=pixels[i];
        if(wakePixels) change+=Math.abs(pixels[i]-wakePixels[i]);
      }
      visibleCoverage=Math.max(visibleCoverage,coverage/(256*160));
      peakChange=Math.max(peakChange,change/(256*160)); wakePixels=pixels;
    }
    assert.ok(visibleCoverage>0.005,'the rendered stern wake remains visible');
    assert.ok(peakChange<0.025,`stern foam coverage must not flash between frames: ${peakChange}`);
    weather.inspectWake('beauty'); testHull.dispose(); rudder.dispose();

    // Keep simulation time frozen: atmosphere transitions use wall-clock dt.
    const dayPosition=sun.position.clone();
    for(const preset of ['dusk','storm','night','dawn','day'] as const) {
      const before=sun.position.clone(), exposure=renderer.toneMappingExposure;
      weather.apply(preset);
      assert.ok(sun.position.equals(before),'selecting a preset must not teleport the sun');
      assert.equal(renderer.toneMappingExposure,exposure,'exposure must not snap');
      for(let frame=0;frame<8;frame++) {
        const direction=sun.position.clone().normalize();
        weather.update(0.2+89/60,7,0.15,1/60); weather.render();
        assert.ok(direction.angleTo(sun.position.clone().normalize())<0.05,'sun motion remains continuous when retargeted');
      }
      const pixels=await renderer.readRenderTargetPixelsAsync(target,0,0,256,160) as Float32Array;
      assert.ok(pixels.every(Number.isFinite),'transition rendering must remain finite');
    }
    assert.ok(sun.position.distanceTo(dayPosition)>0.01,'weather advances while simulation is paused');

    weather.setQuality(true); weather.render();
    const cinematicPixels=await renderer.readRenderTargetPixelsAsync(target,0,0,256,160) as Float32Array;
    assert.ok(cinematicPixels.every(Number.isFinite),'volumes and light shafts compose finite pixels');
    assert.ok(cinematicPixels.some((v,i)=>i%4!==3 && v>0.01),'cinematic composition preserves the beauty image');
    // Camera wetting uses contact energy on the GPU, with smooth drying and
    // no storm required. A distant/high camera must stay dry.
    const lensScene=new THREE.Scene(), lensStorm=createStorm(lensScene);
    const lensCamera=new THREE.PerspectiveCamera();lensCamera.up.set(0,0,1);
    const lensSources=instancedArray(new Float32Array([0,0,0,3]),'vec4');
    const lensLaunches=instancedArray(new Float32Array([0,0,0,4]),'vec4');
    lensStorm.setLensContacts({count:1,activeCount:uint(1),sources:lensSources,launches:lensLaunches},()=>float(0));
    const lensValue=async()=>new Float32Array(await renderer.getArrayBufferAsync(lensStorm.lensWetness.value as THREE.BufferAttribute))[0];
    lensCamera.position.set(0,-30,10);lensCamera.lookAt(0,0,0);
    for(let frame=0;frame<30;frame++){lensStorm.update(0.1,new THREE.Vector3(),lensCamera);lensStorm.updateLens(renderer);}
    assert.equal(await lensValue(),0,'distant camera stays dry in dry weather');
    lensCamera.position.set(0,-1,1);lensCamera.lookAt(0,0,0);
    for(let frame=0;frame<30;frame++){lensStorm.update(0.1,new THREE.Vector3(),lensCamera);lensStorm.updateLens(renderer);}
    const nearWet=await lensValue();assert.ok(nearWet>0.4&&nearWet<=0.75,'nearby energetic splash wets the lens');
    lensCamera.position.set(0,-30,10);lensCamera.lookAt(0,0,0);
    lensStorm.update(0.1,new THREE.Vector3(),lensCamera);lensStorm.updateLens(renderer);
    const drying=await lensValue();assert.ok(drying<nearWet&&drying>nearWet*0.9,'wet lens dries gradually instead of popping off');
    for(let frame=0;frame<300;frame++){lensStorm.update(0.1,new THREE.Vector3(),lensCamera);lensStorm.updateLens(renderer);}
    assert.ok(await lensValue()<nearWet*0.04,'lens clears after leaving the spray');
    for(const child of lensScene.children) if(child instanceof THREE.Mesh){child.geometry.dispose();(child.material as THREE.Material).dispose();}
    // Real spectral breaking must create varied partial coverage in trade wind,
    // more in swell, and none on a glass surface. Age travels with that density.
    const capOcean=createOceanGPU(CASCADES.map((c,i)=>({...c,n:128,choppiness:c.choppiness*1.15,fetch:c.fetch,amplitude:c.amplitude*(i===0?1.35:i===1?112:32)})));
    const capWave=uniform(0.55), capActivity=uniform(0.316), swellWave=uniform(0.85);
    const tradeCaps=createWhitecaps(capOcean,capWave,capActivity), swellCaps=createWhitecaps(capOcean,swellWave);
    const glassCaps=createWhitecaps(capOcean,uniform(0.025),float(0));
    for(let frame=0;frame<180;frame++) {
      updateOceanGPU(renderer,capOcean,frame/30);
      for(const caps of [tradeCaps,swellCaps,glassCaps]) caps.update(renderer,1/30);
    }
    const capFields=await Promise.all([tradeCaps,swellCaps,glassCaps].map(async caps=>new Float32Array(await renderer.getArrayBufferAsync(caps.coverage.value as THREE.BufferAttribute))));
    const capMass=capFields.map(values=>values.reduce((sum,v,i)=>sum+(i%2===0?v:0),0));
    assert.ok(capMass[0]>1,`trade wind creates crest foam (${capMass[0]})`);
    assert.ok(capMass[1]>capMass[0]*1.5,`swell has stronger breaking coverage (${capMass})`);
    assert.equal(capMass[2],0,'glass does not acquire ambient whitecaps');
    assert.ok(capFields[1].slice(0,capOcean.n*capOcean.n*2).every(v=>v===0),'broad swell shoulders do not produce sprawling foam membranes');
    for(const values of capFields) for(let i=0;i<values.length;i+=2) {
      assert.ok(Number.isFinite(values[i]+values[i+1]) && values[i]>=0 && values[i]<=1);
      assert.ok(values[i+1]>=0 && values[i+1]<=values[i]*30+0.00001,'foam age stays bounded by its density');
    }
    assert.ok(capFields[1].filter((v,i)=>i%2===0 && v>0.1).length<capFields[1].length/4,'whitecaps remain localized rather than covering the sea');
    capWave.value=0; capActivity.value=0;
    for(let frame=0;frame<300;frame++) tradeCaps.update(renderer,1/30);
    const expiredCaps=new Float32Array(await renderer.getArrayBufferAsync(tradeCaps.coverage.value as THREE.BufferAttribute));
    const expiredMass=expiredCaps.reduce((sum,v,i)=>sum+(i%2===0?v:0),0);
    assert.ok(expiredMass<capMass[0]*0.2,'crest foam decays after breaking ceases');
    disposeOceanGPU(capOcean,renderer);
    const wake=createWake(), samples=instancedArray(64*32,'vec4'), histories=instancedArray(64*32,'vec4'), sampleOrigin=uniform(0), sampleY=uniform(0);
    const sample=Fn(() => {
      const point=vec2(float(instanceIndex.mod(64)).mul(0.5).sub(16).add(sampleOrigin),float(instanceIndex.div(64)).mul(0.5).sub(8).add(sampleY));
      samples.element(instanceIndex).assign(wake.sample(point));
      histories.element(instanceIndex).assign(wake.sampleHistory(point));
    })().compute(64*32);
    let peakWakeHeight=0, aerationSum=0;
    const foamSum=async () => {
      renderer.compute(sample);
      const values=new Float32Array(await renderer.getArrayBufferAsync(samples.value as THREE.BufferAttribute));
      const metadata=new Float32Array(await renderer.getArrayBufferAsync(histories.value as THREE.BufferAttribute));
      assert.ok(metadata.every(Number.isFinite),'foam flow and age remain finite');
      let sum=0; peakWakeHeight=0; aerationSum=0;
      for(let i=0;i<values.length;i+=4) {
        assert.ok(values.slice(i,i+4).every(Number.isFinite),'wake must remain finite');
        assert.ok(values[i+2]>=0 && values[i+2]<=1,'bounded foam density');
        assert.ok(values[i+3]>=0 && values[i+3]<=1,'bounded entrained bubbles');
        aerationSum+=values[i+3];
        assert.ok(metadata[i+2]>=0 && metadata[i+2]<=values[i+2]*120+0.001,'bounded density-weighted foam age');
        assert.ok(metadata[i+3]>=0 && metadata[i+3]<=1,'bounded foam agitation');
        assert.ok(Math.hypot(metadata[i],metadata[i+1])<3,'bounded transported flow');
        peakWakeHeight=Math.max(peakWakeHeight,Math.abs(values[i]));
        assert.ok(Math.abs(values[i])<2,`wake geometry displacement remains bounded: ${values[i]} m`);
        sum+=values[i+2];
      }
      return sum;
    };
    wake.update(renderer,0,0);
    assert.equal(await foamSum(),0,'stationary hull does not paint foam');
    for(let frame=1;frame<=180;frame++) wake.update(renderer,frame/30,frame/30);
    const moving=await foamSum();
    assert.ok(moving>1,'moving hull emits a persistent wake');
    assert.ok(peakWakeHeight>0.025,`wake moves geometry, peak height ${peakWakeHeight}`);
    for(let frame=181;frame<=660;frame++) wake.update(renderer,frame/30,6);
    const lingering=await foamSum();
    const older=new Float32Array(await renderer.getArrayBufferAsync(histories.value as THREE.BufferAttribute));
    const oldDensity=new Float32Array(await renderer.getArrayBufferAsync(samples.value as THREE.BufferAttribute));
    const retainedAge=older.filter((_,i)=>i%4===2).reduce((a,b)=>a+b,0)/oldDensity.filter((_,i)=>i%4===2).reduce((a,b)=>a+b,0);
    assert.ok(retainedAge>8,'retained foam ages instead of staying permanently fresh');
    assert.ok(lingering>moving*0.15,'wake stays visible sixteen seconds after stopping');
    for(let frame=661;frame<=9000;frame++) wake.update(renderer,frame/30,6);
    assert.ok(await foamSum()<moving*0.15,'long-lived foam eventually dissipates after stopping');
    wake.update(renderer,0,0);
    assert.equal(await foamSum(),0,'reset clears old whitewater');
    const resetHistory=new Float32Array(await renderer.getArrayBufferAsync(histories.value as THREE.BufferAttribute));
    assert.ok(resetHistory.every(v=>v===0),'reset clears old flow and foam age');
    // Emit two narrow contact ribbons, then stop production. Recorded flow
    // must carry both fronts into empty water, rather than pinning their edges.
    const ribbonSource=uniform(1), ribbonX=uniform(0);
    const spreading=createWake({sampleWake:point=>{
      const intensity=point.x.sub(ribbonX).pow2().mul(-8).exp()
        .mul(point.y.abs().sub(2).pow2().mul(-8).exp()).mul(ribbonSource);
      return vec4(0,intensity,0,point.y.div(point.y.abs().add(0.001)).mul(intensity));
    }});
    spreading.update(renderer,0,0);
    for(let frame=1;frame<=60;frame++) { ribbonX.value=frame/10; spreading.update(renderer,frame/60,frame/10); }
    const spreadSamples=instancedArray(64*64,'vec4');
    const spreadProbe=Fn(()=>{ spreadSamples.element(instanceIndex).assign(spreading.sample(vec2(float(instanceIndex.mod(64)).mul(0.5).sub(8),float(instanceIndex.div(64)).mul(0.5).sub(16)))); })().compute(64*64);
    const ribbonWidth=async()=>{
      renderer.compute(spreadProbe);
      const pixels=new Float32Array(await renderer.getArrayBufferAsync(spreadSamples.value as THREE.BufferAttribute));
      const mass=[0,0],distance=[0,0];
      for(let i=0;i<64*64;i++) {
        const foam=pixels[i*4+2], y=Math.floor(i/64)*0.5-16, side=y<0?0:1;
        mass[side]+=foam; distance[side]+=foam*Math.abs(y);
      }
      assert.ok(mass.every(v=>v>1),'both spreading trails retain visible foam');
      return distance.map((v,i)=>v/mass[i]);
    };
    const emittedWidth=await ribbonWidth(); ribbonSource.value=0;
    for(let frame=61;frame<=300;frame++) spreading.update(renderer,frame/60,frame/10,0,frame>120?Math.PI/2:0);
    const widened=await ribbonWidth();
    assert.ok(widened.every((v,i)=>v>emittedWidth[i]+1.5),'both contact trails separate in their recorded world direction after emission stops and the hull turns');
    // Cover every preset's travel speed plus the top of the manual wind range.
    let slowFoam=0, slowAeration=0;
    for(const velocity of [0.24,0.84,1.44,2.16,3.3,6]) {
      wake.update(renderer,0,0);
      for(let frame=1;frame<=180;frame++) wake.update(renderer,frame/30,velocity*frame/30);
      sampleOrigin.value=velocity*6;
      const amount=await foamSum();
      assert.ok(amount>1,`visible persistent foam at ${velocity} m/s`);
      if(velocity===0.24) { slowFoam=amount; slowAeration=aerationSum; }
      else {
        assert.ok(amount>slowFoam,'faster hull produces more whitewater than Glass');
        assert.ok(aerationSum>slowAeration,'cruising hull entrains more bubbles than Glass travel');
      }
    }
    wake.update(renderer,0,0);
    for(let frame=1;frame<=1800;frame++) wake.update(renderer,frame/30,frame/15);
    sampleOrigin.value=20; // 100 metres behind the current hull.
    assert.ok(await foamSum()>1,'wake remains in world space more than 100 metres behind the hull');
    wake.update(renderer,0,0); sampleOrigin.value=0;
    for(let frame=1;frame<=60;frame++) wake.update(renderer,frame/30,0);
    assert.equal(await foamSum(),0,'a stationary hull does not continuously emit whitewater');

    // Sustained travel near the wake field's propagation speed used to build
    // a resonant water wall; six-second high-speed checks missed the buildup.
    for(const velocity of [4.5,5,5.6,6]) {
      wake.update(renderer,0,0);
      for(let frame=1;frame<=1800;frame++) {
        wake.update(renderer,frame/30,velocity*frame/30);
        if(frame%90===0) {
          sampleOrigin.value=velocity*frame/30;
          await foamSum();
        }
      }
    }
    // Match the recording's kilometre-scale position, diagonal course, and
    // variable render cadence while accelerating through the resonant speed.
    let voyageTime=0, voyageX=3600, voyageY=-1200;
    wake.update(renderer,0,voyageX,voyageY,0.5);
    for(let frame=0;frame<3600;frame++) {
      const dt=[1/120,1/120,1/30,1/60][frame%4];
      voyageTime+=dt;
      const heading=0.5+0.2*Math.sin(voyageTime*0.12), velocity=4+2*(1-Math.exp(-voyageTime/6));
      voyageX+=Math.cos(heading)*velocity*dt; voyageY+=Math.sin(heading)*velocity*dt;
      wake.update(renderer,voyageTime,voyageX,voyageY,heading);
      if(frame%120===119) {
        sampleOrigin.value=voyageX; sampleY.value=voyageY;
        assert.ok(await foamSum()>1,'wake stays with the hull under variable cadence');
      }
    }
    sampleOrigin.value=0; sampleY.value=0;

    // Rotate the source through a turn; the old world-space trail must survive.
    wake.update(renderer,0,0);
    for(let frame=1;frame<=180;frame++) wake.update(renderer,frame/30,frame/30);
    const beforeTurn=await foamSum();
    for(let frame=181;frame<=360;frame++) {
      const angle=(frame-180)/180*Math.PI/2;
      wake.update(renderer,frame/30,6+6*Math.sin(angle),6*(1-Math.cos(angle)),angle);
    }
    assert.ok(await foamSum()>beforeTurn*0.25,'turning retains foam at the old track instead of rotating or clearing it');

    // A heading change must not reorient foam already left behind.
    wake.update(renderer,0,0); sampleOrigin.value=0; sampleY.value=0;
    for(let frame=1;frame<=180;frame++) wake.update(renderer,frame/30,frame/30);
    await foamSum();
    const beforeFlow=new Float32Array(await renderer.getArrayBufferAsync(histories.value as THREE.BufferAttribute));
    const flowMagnitude=beforeFlow.reduce((sum,value,i)=>sum+(i%4<2?Math.abs(value):0),0);
    assert.ok(flowMagnitude>1,'stern-quarter emission imparts world-space flow');
    wake.update(renderer,181/30,6,0,Math.PI/2);
    await foamSum();
    const afterFlow=new Float32Array(await renderer.getArrayBufferAsync(histories.value as THREE.BufferAttribute));
    const flowChange=afterFlow.reduce((sum,value,i)=>sum+(i%4<2?Math.abs(value-beforeFlow[i]):0),0);
    assert.ok(flowChange<flowMagnitude*0.1,'old wake flow keeps its direction through a new hull heading');

    const spray=createShipSpray(camera,new THREE.DepthTexture(1,1),()=>float(-0.45),vec3(1),vec3(0.5),vec3(0,0,1));
    spray.reset(renderer);
    const inspectSpray=async (pool=spray) => {
      const values=new Float32Array(await renderer.getArrayBufferAsync(pool.positions.value as THREE.BufferAttribute));
      assert.ok(values.every(Number.isFinite),'particle pool stays finite');
      let drops=0,mist=0;
      for(let i=0;i<values.length;i+=4) if(values[i+3]>0) {
        assert.ok(values[i+2]>-0.45 && values[i+2]<8,'live particles remain above water with bounded launch height');
        if(i/4<pool.dropCount) drops++; else mist++;
      }
      return {drops,mist};
    };
    for(let i=0;i<4;i++) spray.contact(i,new THREE.Vector3(0,i%2?2:-2,-0.45),0);
    for(let i=0;i<60;i++) spray.update(renderer,1/60,0,new THREE.Vector3(),new THREE.Vector3());
    assert.deepEqual(await inspectSpray(),{drops:0,mist:0},'no spray without speed or impact');
    for(let i=0;i<4;i++) spray.contact(i,new THREE.Vector3(0,i%2?2:-2,-0.45),0.5,0.2);
    for(let i=0;i<20;i++) spray.update(renderer,1/60,0,new THREE.Vector3(),new THREE.Vector3());
    const gentle=await inspectSpray();
    spray.reset(renderer);
    for(let i=0;i<4;i++) spray.contact(i,new THREE.Vector3(0,i%2?2:-2,-0.45),2,1.5);
    for(let i=0;i<20;i++) spray.update(renderer,1/60,0,new THREE.Vector3(),new THREE.Vector3());
    const crashing=await inspectSpray();
    assert.ok(crashing.drops>gentle.drops*2 && crashing.mist>gentle.mist*2,'larger wave impacts produce substantially more spray and mist');
    for(let i=0;i<4;i++) spray.contact(i,new THREE.Vector3(0,i%2?2:-2,-0.45),2);
    for(let i=0;i<90;i++) spray.update(renderer,1/60,1.8,new THREE.Vector3(),new THREE.Vector3(0.4,0,0));
    const active=await inspectSpray();
    assert.ok(active.drops>0 && active.mist>0,'bow travel emits both drops and mist');
    for(let i=0;i<480;i++) spray.update(renderer,1/60,0,new THREE.Vector3(),new THREE.Vector3());
    assert.deepEqual(await inspectSpray(),{drops:0,mist:0},'particles die after motion and impacts stop');
    spray.reset(renderer);
    assert.deepEqual(await inspectSpray(),{drops:0,mist:0},'reset clears the particle pool');
    spray.mesh.geometry.dispose(); (spray.mesh.material as THREE.Material).dispose();

    // Contact energy, sheet launches and drop landings share the same GPU sources.
    // Keep the wake hull still so all foam in this check comes from landing drops.
    const level=uniform(-0.45), impacts=createBowImpacts(()=>level);
    impacts.setHull(Array.from({length:8},(_,i)=>({x:6-Math.floor(i/2)*0.7,y:i%2?-1:1,nx:Math.SQRT1_2,ny:(i%2?-1:1)*Math.SQRT1_2})));
    const coupled=createShipSpray(camera,new THREE.DepthTexture(1,1),()=>level,vec3(1),vec3(0.5),vec3(0,0,1),impacts,wake.deposit);
    impacts.reset(renderer); coupled.reset(renderer);
    const matrix=new THREE.Matrix4().makeTranslation(3000,-1200,0), hullPoint=new THREE.Vector3(3000,-1200,0), air=new THREE.Vector3();
    sampleOrigin.value=3006; sampleY.value=-1200; wake.update(renderer,0,3000,-1200);
    const scatter=Fn(()=>{ wake.deposit(vec2(3006,-1200),float(0.25)); })().compute(1);
    renderer.compute(scatter); wake.update(renderer,1/60,3000,-1200);
    const depositedFoam=await foamSum();
    assert.ok(depositedFoam>0,'atomic landing accumulator feeds foam once');
    wake.update(renderer,2/60,3000,-1200);
    assert.ok(await foamSum()<depositedFoam,'consumed landing does not emit again on the next step');
    wake.update(renderer,0,3000,-1200);
    const crowdedLanding=Fn(()=>{ wake.deposit(vec2(3006,-1200),float(0.01)); })().compute(400);
    renderer.compute(crowdedLanding); wake.update(renderer,1/60,3000,-1200);
    assert.ok(Math.abs(await foamSum()-(1-Math.exp(-0.6)))<0.001,'concurrent landing writes accumulate and saturate safely');
    wake.update(renderer,0,3000,-1200);
    for(let frame=1;frame<=30;frame++) {
      impacts.update(renderer,1/60,frame/60,matrix);
      coupled.update(renderer,1/60,0,hullPoint,air);
    }
    assert.deepEqual(await inspectSpray(coupled),{drops:0,mist:0},'resting GPU contacts do not emit');
    let steadyEnergy=0;
    for(let frame=31;frame<=150;frame++) {
      const t=frame/60; matrix.makeTranslation(3000+(frame-30)/10,-1200,0);
      wake.update(renderer,t,3000,-1200); impacts.update(renderer,1/60,t,matrix);
      coupled.update(renderer,1/60,6,hullPoint,air);
    }
    const sourceValues=new Float32Array(await renderer.getArrayBufferAsync(impacts.sources.value as THREE.BufferAttribute));
    for(let i=0;i<sourceValues.length;i+=4) { assert.ok(sourceValues[i]>3000); steadyEnergy+=sourceValues[i+3]; }
    const liveSheets=new Float32Array(await renderer.getArrayBufferAsync(impacts.roots.value as THREE.BufferAttribute));
    assert.ok(liveSheets.filter((_,i)=>i%4===3 && liveSheets[i]>0).length>0,'cutting bow launches coherent sheets');
    const sharedActive=await inspectSpray(coupled);
    assert.ok(sharedActive.drops>0 && sharedActive.mist>0,'shared sources emit both particle phases');
    assert.ok(await foamSum()>0,'landed spray deposits world-space foam without moving wake hull');
    level.value=0.05;
    impacts.update(renderer,1/60,151/60,matrix);
    const hit=new Float32Array(await renderer.getArrayBufferAsync(impacts.sources.value as THREE.BufferAttribute));
    assert.ok(hit.filter((_,i)=>i%4===3).reduce((a,b)=>a+b,0)>steadyEnergy,'rising water strengthens bow contact');
    // Sustained heave and turns at high speed, far from the origin.
    for(let frame=152;frame<=1952;frame++) {
      const t=frame/60, yaw=t*0.08;
      level.value=-0.45+Math.sin(t*2.1)*0.3;
      matrix.makeRotationZ(yaw); matrix.setPosition(3000+75*Math.sin(yaw),-1200+75*(1-Math.cos(yaw)),Math.sin(t*1.7)*0.2);
      impacts.update(renderer,1/60,t,matrix);
      coupled.update(renderer,1/60,6,hullPoint,air,yaw);
      if(frame%120===0) {
        for(const field of [impacts.sources,impacts.launches,impacts.roots,impacts.vectors,coupled.positions,coupled.velocities]) {
          const values=new Float32Array(await renderer.getArrayBufferAsync(field.value as THREE.BufferAttribute));
          assert.ok(values.every(Number.isFinite),'contact, sheet and spray buffers remain finite during swell and turns');
          if(field===impacts.sources) for(let i=3;i<values.length;i+=4) assert.ok(values[i]>=0 && values[i]<=5);
          if(field===impacts.roots) for(let i=3;i<values.length;i+=4) assert.ok(values[i]>=0 && values[i]<=0.75);
        }
      }
    }
    impacts.reset(renderer); coupled.reset(renderer);
    const cleared=new Float32Array(await renderer.getArrayBufferAsync(impacts.roots.value as THREE.BufferAttribute));
    assert.ok(cleared.every(v=>v===0),'reset removes all old sheet trajectories');
    assert.deepEqual(await inspectSpray(coupled),{drops:0,mist:0});
    for(const mesh of [impacts.mesh,coupled.mesh]) { mesh.geometry.dispose(); (mesh.material as THREE.Material).dispose(); }

    // Refining an identical waterline must not multiply the emitted films.
    // All segments share the same energy/velocity; only their lengths differ.
    const sheetPools=[1,32,1].map((segments,poolIndex)=>{
      const sources=instancedArray(segments,'vec4'), launches=instancedArray(segments,'vec4');
      renderer.compute(Fn(()=>{
        sources.element(instanceIndex).assign(vec4(0,0,0,1));
        launches.element(instanceIndex).assign(vec4(1,0,2,1/segments));
      })().compute(segments));
      const starts=instancedArray(segments,'vec4'), ends=instancedArray(segments,'vec4');
      renderer.compute(Fn(()=>{ starts.element(instanceIndex).assign(vec4(0,0,0,1)); ends.element(instanceIndex).assign(vec4(0,0,0,0)); })().compute(segments));
      return createImpactSheets(sources,launches,uint(segments),poolIndex===2?starts:undefined,poolIndex===2?ends:undefined);
    });
    for(const pool of sheetPools) pool.reset(renderer);
    const filmOccupancy=instancedArray(3,'uint').toAtomic();
    const accumulate=sheetPools.map((pool,index)=>Fn(()=>{
      If(pool.roots.element(instanceIndex).w.greaterThan(0),()=>{ atomicAdd(filmOccupancy.element(index),uint(1)); });
    })().compute(pool.mesh.count));
    for(let frame=1;frame<=600;frame++) for(let index=0;index<sheetPools.length;index++) {
      sheetPools[index].update(renderer,1/60,frame/60); renderer.compute(accumulate[index]);
    }
    const occupancy=new Uint32Array(await renderer.getArrayBufferAsync(filmOccupancy.value as THREE.BufferAttribute));
    const films=await Promise.all(sheetPools.map(async pool=>new Float32Array(await renderer.getArrayBufferAsync(pool.roots.value as THREE.BufferAttribute))));
    assert.ok(films[0].some((value,i)=>i%4===3 && value>0),'weighted films still emit on an energetic contact');
    assert.deepEqual(films[0],films[1],'subdividing a contact preserves sheet emission and trajectories');
    assert.ok(occupancy[0]>0 && occupancy[0]===occupancy[1],'contact subdivision preserves cumulative film occupancy');
    assert.ok(occupancy[2]>occupancy[0],'forward shoulder produces denser films over the full emission interval at equal contact energy');
    for(let index=0;index<sheetPools.length;index++) {
      const pool=sheetPools[index], widths=new Float32Array(await renderer.getArrayBufferAsync(pool.widths.value as THREE.BufferAttribute));
      for(let i=0;i<widths.length/2;i++) if(films[index][i*4+3]>0) {
        assert.ok(Math.abs(widths[i*2]-(index===0?1.1:index===1?0.18:1.75))<0.00001,'film widths follow contact lengths with bounded limits');
      }
      if(index===2) {
        // A strong young bow film must render visible water mass, not just a fog
        // sprite. Render an isolated ballistic fan against transparent black.
        const fan=sheetPools[2], fanScene=new THREE.Scene(), fanCamera=camera.clone();
        renderer.compute(Fn(()=>{
          fan.roots.element(instanceIndex).assign(instanceIndex.equal(0).select(vec4(0,0,0,0.6),vec4(0)));
          fan.vectors.element(instanceIndex).assign(vec4(2,0,6,0.9));
          fan.widths.element(instanceIndex).assign(vec2(2.4,1));
        })().compute(fan.mesh.count));
        fanCamera.position.set(5,-7,4); fanCamera.lookAt(0.4,0,0.8); fanCamera.aspect=1; fanCamera.updateProjectionMatrix();
        fanScene.add(fan.mesh,new THREE.HemisphereLight(0xffffff,0x708090,2));
        const fanTarget=new THREE.RenderTarget(128,128,{type:THREE.HalfFloatType});
        const clearColor=renderer.getClearColor(new THREE.Color()), clearAlpha=renderer.getClearAlpha();
        renderer.setClearColor(0,0); renderer.setRenderTarget(fanTarget); renderer.render(fanScene,fanCamera);
        const fanPixels=await renderer.readRenderTargetPixelsAsync(fanTarget,0,0,128,128) as Uint16Array;
        assert.ok(fanPixels.every(v=>(v&0x7c00)!==0x7c00),'curved whitewater film renders finite pixels');
        assert.ok(fanPixels.filter((v,i)=>i%4===3 && THREE.DataUtils.fromHalfFloat(v)>0.5).length>40,'young aerated splash contains dense connected water coverage');
        fanCamera.position.set(0.4,-7,0.8); fanCamera.lookAt(0.4,0,0.8);
        renderer.render(fanScene,fanCamera);
        const sidePixels=await renderer.readRenderTargetPixelsAsync(fanTarget,0,0,128,128) as Uint16Array;
        assert.ok(sidePixels.every(v=>(v&0x7c00)!==0x7c00),'side profile of splash volume remains finite');
        assert.ok(sidePixels.filter((v,i)=>i%4===3 && THREE.DataUtils.fromHalfFloat(v)>0.25).length>12,'rounded splash retains visible water mass from the side');
        fanScene.remove(fan.mesh); renderer.setClearColor(clearColor,clearAlpha); renderer.setRenderTarget(target); fanTarget.dispose();
      }
      pool.reset(renderer);
      for(const buffer of [pool.widths]) assert.ok(new Float32Array(await renderer.getArrayBufferAsync(buffer.value as THREE.BufferAttribute)).every(value=>value===0),'reset clears film width and aeration');
      pool.mesh.geometry.dispose(); (pool.mesh.material as THREE.Material).dispose();
    }

    // The new path intersects actual hull triangles with the rendered water mesh.
    const contacts=createHullContacts(), box=new THREE.BoxGeometry(18,5,4).translate(2,0.4,0);
    const waterGeometry=new THREE.PlaneGeometry(44,44,16,16), waterLevel=uniform(0), waterTilt=uniform(0);
    contacts.setWater(waterGeometry,vec3(positionLocal.xy,waterLevel.add(positionLocal.x.mul(waterTilt))));
    contacts.setHull([box]);
    const contactWake=createWake(contacts);
    const exactSpray=createShipSpray(camera,new THREE.DepthTexture(1,1),contacts.surface,vec3(1),vec3(0.5),vec3(0,0,1),contacts,contactWake.deposit);
    const volumeCamera=camera.clone();
    volumeCamera.position.set(23,-16,6); volumeCamera.lookAt(13,0,0);
    const volumeDepth=new THREE.RenderTarget(256,160,{depthTexture:new THREE.DepthTexture(256,160)});
    const volumeScene=new THREE.Scene();
    const settlingWake=createWake(); settlingWake.update(renderer,0,0);
    const volumes=createWhitewaterVolumes(volumeCamera,volumeDepth.depthTexture!,contacts,vec3(1),vec3(0.5),vec3(0,0,1),settlingWake.deposit);
    volumes.reset(renderer);
    const exclusionChecks=instancedArray(4,'uint');
    const queryHull=Fn(()=>{
      const points=[vec3(2,0.4,0),vec3(2,4,0),vec3(12,0.4,0),vec3(2,0.4,3)];
      for(let i=0;i<points.length;i++) exclusionChecks.element(i).assign(contacts.outsideHull(points[i]).select(uint(1),uint(0)));
    })().compute(1);
    renderer.compute(queryHull);
    assert.deepEqual(Array.from(new Uint32Array(await renderer.getArrayBufferAsync(exclusionChecks.value as THREE.BufferAttribute))),[0,1,1,0],'splash exclusion rejects hull interior and the air above the deck, preserving surrounding air');
    const thinDeck=new THREE.PlaneGeometry(4,4).translate(2,0.4,1);
    const deckExclusion=createHullExclusion(); deckExclusion.setHull([thinDeck]);
    renderer.compute(Fn(()=>{
      const points=[vec3(2,0.4,0),vec3(2,4,0),vec3(12,0.4,0),vec3(2,0.4,3)];
      for(let i=0;i<points.length;i++) exclusionChecks.element(i).assign(deckExclusion.outside(points[i]).select(uint(1),uint(0)));
    })().compute(1));
    assert.deepEqual(Array.from(new Uint32Array(await renderer.getArrayBufferAsync(exclusionChecks.value as THREE.BufferAttribute))),[0,1,1,1],'a zero-thickness horizontal deck culls splashes underneath while preserving air above and outside');
    renderer.compute(Fn(()=>{
      exclusionChecks.element(0).assign(deckExclusion.outsideSplash(vec3(2,0.4,3)).select(uint(1),uint(0)));
      exclusionChecks.element(1).assign(deckExclusion.outsideSplash(vec3(2,4,3)).select(uint(1),uint(0)));
    })().compute(1));
    const deckClip=new Uint32Array(await renderer.getArrayBufferAsync(exclusionChecks.value as THREE.BufferAttribute));
    assert.equal(deckClip[0],0,'airborne splash over the deck footprint must be clipped');
    assert.equal(deckClip[1],1,'airborne splash outside the hull remains visible');
    renderer.compute(Fn(()=>{
      exclusionChecks.element(0).assign(deckExclusion.outsideWater(vec3(2,0.4,0)).select(uint(1),uint(0)));
      exclusionChecks.element(1).assign(deckExclusion.outsideWater(vec3(2,0.4,3)).select(uint(1),uint(0)));
    })().compute(1));
    const waterClip=new Uint32Array(await renderer.getArrayBufferAsync(exclusionChecks.value as THREE.BufferAttribute));
    assert.equal(waterClip[0],1,'deck overhang must not remove open water below the deck');
    assert.equal(waterClip[1],0,'water crests above the deck must be clipped');
    thinDeck.dispose();
    const crossingDrop=Fn(()=>{
      exactSpray.positions.element(0).assign(vec4(2,-4,1,1));
      exactSpray.velocities.element(0).assign(vec4(0,160,0,1));
    })().compute(1);
    renderer.compute(crossingDrop);
    exactSpray.update(renderer,0.05,0,new THREE.Vector3(),new THREE.Vector3());
    const crossed=new Float32Array(await renderer.getArrayBufferAsync(exactSpray.positions.value as THREE.BufferAttribute));
    assert.equal(crossed[3],0,'swept drop crossing the entire hull is killed even when its endpoint is outside');
    const inspectContacts=async()=>{
      const counter=new Uint32Array(await renderer.getArrayBufferAsync(contacts.counter.value as THREE.BufferAttribute))[0];
      assert.ok(counter>0 && counter<=contacts.count,'box has a compact, bounded intersection curve');
      const draw=new Uint32Array(await renderer.getArrayBufferAsync(contacts.drawBuffer));
      assert.equal(draw[1],counter,'indirect curtain draw visits exactly the live waterline segments');
      const values=new Float32Array(await renderer.getArrayBufferAsync(contacts.sources.value as THREE.BufferAttribute)).slice(0,counter*4);
      const starts=new Float32Array(await renderer.getArrayBufferAsync(contacts.starts.value as THREE.BufferAttribute)).slice(0,counter*4);
      const ends=new Float32Array(await renderer.getArrayBufferAsync(contacts.ends.value as THREE.BufferAttribute)).slice(0,counter*4);
      assert.ok(values.every(Number.isFinite) && starts.every(Number.isFinite) && ends.every(Number.isFinite));
      return {values,starts,ends,counter};
    };
    contacts.reset(renderer); exactSpray.reset(renderer);
    matrix.identity(); contacts.update(renderer,1/60,0,matrix); contactWake.update(renderer,0,0);
    let curve=await inspectContacts();
    assert.ok(curve.values.filter((_,i)=>i%4===3).every(v=>v===0),'first contact does not create a velocity spike');
    for(const endpoints of [curve.starts,curve.ends]) for(let i=0;i<endpoints.length;i+=4) {
      assert.ok(Math.abs(endpoints[i+2]+0.45)<0.002,'contact sits on the actual water plane');
      const face=Math.min(Math.abs(endpoints[i]+7),Math.abs(endpoints[i]-11),Math.abs(endpoints[i+1]+2.1),Math.abs(endpoints[i+1]-2.9));
      assert.ok(face<0.002,'intersection lies on a real box face, not an ellipse');
    }
    waterLevel.value=-0.05; contacts.update(renderer,1/60,1/60,matrix);
    const aligned=await inspectContacts();
    for(let i=2;i<aligned.starts.length;i+=4) assert.ok(Math.abs(aligned.starts[i]+0.5)<0.002,'waterlines exactly on subdivided vertices retain the intersection curve');
    waterLevel.value=0.7; waterTilt.value=0.02;
    matrix.makeRotationX(0.08); matrix.setPosition(0,0,0.25);
    contacts.update(renderer,1/60,1/60,matrix);
    curve=await inspectContacts();
    for(const endpoints of [curve.starts,curve.ends]) for(let i=0;i<endpoints.length;i+=4) {
      assert.ok(Math.abs(endpoints[i+2]-(0.25+endpoints[i]*0.02))<0.004,'contact tracks the tilted rendered water and moving hull');
      const local=new THREE.Vector3(endpoints[i],endpoints[i+1],endpoints[i+2]).applyMatrix4(matrix.clone().invert());
      assert.ok(Math.min(Math.abs(local.x+7),Math.abs(local.x-11),Math.abs(local.y+2.1),Math.abs(local.y-2.9))<0.004,'refined root stays on its hull triangle');
    }
    contacts.reset(renderer); exactSpray.reset(renderer); waterLevel.value=0; waterTilt.value=0; matrix.identity();
    contacts.update(renderer,1/60,0,matrix); contactWake.update(renderer,0,0);
    for(let frame=1;frame<=45;frame++) {
      matrix.makeTranslation(frame/10,0,0); contacts.update(renderer,1/60,frame/60,matrix);
      settlingWake.update(renderer,frame/60,0); volumes.update(renderer,1/60,air);
      contactWake.update(renderer,frame/60,frame/10); exactSpray.update(renderer,1/60,6,new THREE.Vector3(frame/10,0,0),air);
    }
    assert.ok((await inspectSpray(exactSpray)).drops>0,'mesh intersection launches spray');
    const volumePositions=new Float32Array(await renderer.getArrayBufferAsync(volumes.positions.value as THREE.BufferAttribute));
    assert.ok(volumePositions.every(Number.isFinite) && volumePositions.some((v,i)=>i%4===3 && v>0),'actual bow contacts populate finite live volumes');
    assert.ok(volumePositions.filter((_,i)=>i%4===3).every(v=>v<=1.7),'volumes have bounded lifetimes');
    const filmRoots=new Float32Array(await renderer.getArrayBufferAsync(contacts.roots.value as THREE.BufferAttribute));
    const filmLaunches=new Float32Array(await renderer.getArrayBufferAsync(contacts.vectors.value as THREE.BufferAttribute));
    let liveCores=0;
    for(let i=0;i<64;i++) if(volumePositions[i*4+3]>0) {
      const j=i*6*4, life=filmLaunches[j+3], seconds=life-filmRoots[j+3];
      const flight=seconds*(0.72+seconds/Math.max(life,0.001)*0.15);
      assert.equal(volumePositions[i*4+3],filmRoots[j+3],'dense core lifetime follows its actual splash');
      for(let axis=0;axis<3;axis++) {
        const expected=filmRoots[j+axis]+filmLaunches[j+axis]*flight+(axis===2?-4.905*flight*flight:0);
        assert.ok(Math.abs(volumePositions[i*4+axis]-expected)<0.0001,'dense whitewater stays on the ballistic sheet trajectory');
      }
      liveCores++;
    }
    assert.ok(liveCores>0,'bow splash films have connected volumetric cores');
    renderer.setRenderTarget(volumeDepth); renderer.render(volumeScene,volumeCamera); volumes.render(renderer);
    const volumePixels=await renderer.readRenderTargetPixelsAsync(volumes.target,0,0,128,80) as Uint16Array;
    assert.ok(volumePixels.every(v=>(v&0x7c00)!==0x7c00),'ray-marched optical depth and lighting stay finite');
    assert.ok(volumePixels.some((v,i)=>i%4===3 && v>0),'live plumes render volumetric optical depth');
    const blocker=new THREE.Mesh(new THREE.BoxGeometry(40,40,40),new THREE.MeshBasicMaterial());
    blocker.position.copy(volumeCamera.position).add(new THREE.Vector3(13,0,0).sub(volumeCamera.position).normalize().multiplyScalar(25));
    volumeScene.add(blocker);
    renderer.render(volumeScene,volumeCamera); volumes.render(renderer);
    const blockedPixels=await renderer.readRenderTargetPixelsAsync(volumes.target,0,0,128,80) as Uint16Array;
    assert.ok(blockedPixels.filter((_,i)=>i%4===3).every(v=>v===0),'opaque hull depth clips the entire plume when occluded');
    blocker.geometry.dispose(); blocker.material.dispose(); volumeScene.remove(blocker); renderer.setRenderTarget(target);
    const movingCurve=await inspectContacts();
    let forwardContacts=0, aftContacts=0;
    for(let i=0;i<movingCurve.starts.length;i+=4) {
      const x=movingCurve.values[i]-4.5, weight=movingCurve.starts[i+3];
      assert.ok(weight>=0 && weight<=1,'bow classification stays bounded');
      if(x>10.9) { assert.ok(weight>0.99,'leading hull face gets bow emission'); forwardContacts++; }
      if(x<0) { assert.equal(weight,0,'aft contacts do not inherit the bow curtain'); aftContacts++; }
    }
    assert.ok(forwardContacts>0 && aftContacts>0,'classification covers both ends of actual geometry');
    const sprayLaunches=new Float32Array(await renderer.getArrayBufferAsync(exactSpray.velocities.value as THREE.BufferAttribute));
    const mistLifetimes=sprayLaunches.slice(exactSpray.dropCount*4).filter((_,i)=>i%4===3);
    assert.ok(mistLifetimes.every(v=>Number.isFinite(v) && v>=0 && v<=4.5),'impact mist lifetimes remain bounded');
    assert.ok(mistLifetimes.some(v=>v>3),'bow impacts sustain a larger mist plume');
    const contactSamples=instancedArray(128,'vec4'), contactProbeOrigin=uniform(new THREE.Vector2());
    const contactProbe=Fn(()=>{
      const point=vec2(float(instanceIndex).mul(0.2).sub(8),0.4).add(contactProbeOrigin);
      contactSamples.element(instanceIndex).assign(contactWake.sample(point));
    })().compute(128);
    renderer.compute(contactProbe);
    const generated=new Float32Array(await renderer.getArrayBufferAsync(contactSamples.value as THREE.BufferAttribute));
    assert.ok(generated.some((v,i)=>i%4===0 && Math.abs(v)>0.01),'submerged triangle footprint displaces wake geometry');
    assert.ok(generated.some((v,i)=>i%4===2 && v>0.01),'the same contact field propagates foam');
    assert.ok(generated[22*4+2]>0.01,'released hull footprint generates foam behind the stern center, beyond the side contact ribbons');
    const curtainScene=new THREE.Scene(), curtainCamera=camera.clone();
    curtainCamera.position.set(23,-16,6); curtainCamera.lookAt(13,0,0); curtainCamera.aspect=1; curtainCamera.updateProjectionMatrix();
    curtainScene.add(contacts.curtain,new THREE.HemisphereLight(0xffffff,0x708090,2));
    const curtainTarget=new THREE.RenderTarget(128,128,{type:THREE.HalfFloatType});
    renderer.setRenderTarget(curtainTarget); renderer.render(curtainScene,curtainCamera);
    const curtainPixels=await renderer.readRenderTargetPixelsAsync(curtainTarget,0,0,128,128) as Uint16Array;
    assert.ok(curtainPixels.every(v=>(v&0x7c00)!==0x7c00),'indirect curtain renders finite half-float pixels');
    assert.ok(curtainPixels.some((v,i)=>i%4!==3 && v>0),'live contact curtain reaches the actual rendered image');
    curtainScene.remove(contacts.curtain); renderer.setRenderTarget(target); curtainTarget.dispose();
    matrix.makeTranslation(4.5,0,5); contacts.update(renderer,1/60,46/60,matrix);
    const dryCount=new Uint32Array(await renderer.getArrayBufferAsync(contacts.counter.value as THREE.BufferAttribute))[0];
    assert.equal(dryCount,0,'dry hull has no water intersections or emission sources');
    assert.equal(new Uint32Array(await renderer.getArrayBufferAsync(contacts.drawBuffer))[1],0,'dry hull produces no curtain draw');
    // Once the hull is dry no new plume can emit. Existing airborne whitewater
    // must settle into a footprint that persists after all volume lifetimes end.
    for(let frame=47;frame<=210;frame++) {
      contacts.update(renderer,1/60,frame/60,matrix);
      settlingWake.update(renderer,frame/60,0); volumes.update(renderer,1/60,new THREE.Vector3());
    }
    const landedFoam=instancedArray(32*24,'vec4');
    renderer.compute(Fn(()=>{
      landedFoam.element(instanceIndex).assign(settlingWake.sample(vec2(float(instanceIndex.mod(32)).mul(0.5).add(5),float(instanceIndex.div(32)).mul(0.5).sub(6))));
    })().compute(32*24));
    const landedValues=new Float32Array(await renderer.getArrayBufferAsync(landedFoam.value as THREE.BufferAttribute));
    assert.ok(landedValues.every(Number.isFinite),'settling whitewater leaves a finite foam field');
    assert.ok(landedValues.some((v,i)=>i%4===2 && v>0.001),'airborne whitewater transitions into persistent surface foam after the hull stops emitting');
    assert.ok(new Float32Array(await renderer.getArrayBufferAsync(volumes.positions.value as THREE.BufferAttribute)).filter((_,i)=>i%4===3).every(v=>v<=0),'settled or expired plumes leave no lingering airborne density');

    // Sustained high-speed contact under roll, rising water, turns and mixed cadence.
    let contactTime=1, contactX=3000, contactY=-1200;
    contactWake.update(renderer,0,contactX,contactY);
    for(let frame=1;frame<=180;frame++) {
      const elapsed=[1/30,1/60,1/120][frame%3], heading=frame*0.005;
      contactTime+=elapsed; contactX+=Math.cos(heading)*6*elapsed; contactY+=Math.sin(heading)*6*elapsed;
      waterLevel.value=Math.sin(contactTime*2.1)*0.6;
      matrix.makeRotationZ(heading).multiply(new THREE.Matrix4().makeRotationX(Math.sin(contactTime)*0.12));
      matrix.setPosition(contactX,contactY,Math.sin(contactTime*1.7)*0.25);
      contacts.update(renderer,elapsed,contactTime,matrix);
      contactWake.update(renderer,contactTime,contactX,contactY,heading);
      volumes.update(renderer,elapsed,air);
      exactSpray.update(renderer,elapsed,6,new THREE.Vector3(contactX,contactY,0),air,heading);
      if(frame%30===0) {
        const curve=await inspectContacts();
        for(let i=3;i<curve.values.length;i+=4) assert.ok(curve.values[i]>=0 && curve.values[i]<=5,'intersection energy remains bounded');
        contactProbeOrigin.value.set(contactX,contactY); renderer.compute(contactProbe);
        const samples=new Float32Array(await renderer.getArrayBufferAsync(contactSamples.value as THREE.BufferAttribute));
        assert.ok(samples.every(Number.isFinite),'mesh-driven wake remains finite through swell and turns');
        for(let i=0;i<samples.length;i+=4) { assert.ok(Math.abs(samples[i])<2); assert.ok(samples[i+2]>=0 && samples[i+2]<=1); }
      }
    }
    const sustainedVolumes=new Float32Array(await renderer.getArrayBufferAsync(volumes.positions.value as THREE.BufferAttribute));
    assert.ok(sustainedVolumes.every(Number.isFinite),'volume advection remains finite through swell, turns and mixed cadence');
    volumes.reset(renderer);
    assert.ok(new Float32Array(await renderer.getArrayBufferAsync(volumes.profiles.value as THREE.BufferAttribute)).every(v=>v===0),'reset removes all volume density profiles');
    volumes.mesh.geometry.dispose(); (volumes.mesh.material as THREE.Material).dispose(); volumes.target.dispose(); volumeDepth.dispose();
    exactSpray.reset(renderer);
    assert.ok(new Float32Array(await renderer.getArrayBufferAsync(exactSpray.velocities.value as THREE.BufferAttribute)).every(v=>v===0),'reset clears stored spray lifetime');
    contacts.reset(renderer);
    const contactFieldSamples=instancedArray(1,'vec4');
    renderer.compute(Fn(()=>{contactFieldSamples.element(0).assign(contacts.sample(vec2(contactX,contactY)));})().compute(1));
    const resetField=new Float32Array(await renderer.getArrayBufferAsync(contactFieldSamples.value as THREE.BufferAttribute));
    assert.ok(resetField.every(v=>v===0),'reset removes immediate hull contact foam');
    // A physical point on the hull must retain the same contact texture UV
    // through translation, steering and roll; a camera never enters this map.
    const hullTexturePoint=uniform(new THREE.Vector3()), hullTextureResult=instancedArray(1,'vec4');
    const sampleHullTexture=Fn(()=>{hullTextureResult.element(0).assign(vec4(contacts.localPoint(hullTexturePoint),1));})().compute(1);
    const localTexturePoint=new THREE.Vector3(2.7,-1.8,0.1);
    for(let pose=0;pose<4;pose++) {
      matrix.makeRotationZ(pose*0.73).multiply(new THREE.Matrix4().makeRotationX(pose*0.08));
      matrix.setPosition(300+pose*0.5,-120,0.2);
      contacts.update(renderer,1/60,20+pose/60,matrix);
      hullTexturePoint.value.copy(localTexturePoint).applyMatrix4(matrix);renderer.compute(sampleHullTexture);
      const uv=new Float32Array(await renderer.getArrayBufferAsync(hullTextureResult.value as THREE.BufferAttribute));
      for(let axis=0;axis<3;axis++) assert.ok(Math.abs(uv[axis]-localTexturePoint.getComponent(axis))<0.0001,'contact texture follows the full hull pose');
    }
    // Material-side sampling must agree with the world-space contact sources,
    // including an asymmetric hull after yaw (a mirrored field looks plausible at zero yaw).
    const contactPixel=uniform(new THREE.Vector2());
    const probeMaterial=new THREE.MeshBasicNodeMaterial();
    probeMaterial.colorNode=vec3(contacts.sampleFoam(contactPixel));probeMaterial.toneMapped=false;
    const probeScene=new THREE.Scene(), probeCamera=new THREE.OrthographicCamera(-1,1,1,-1,0.1,10);
    probeCamera.position.z=2;
    const probeQuad=new THREE.Mesh(new THREE.PlaneGeometry(2,2),probeMaterial);probeScene.add(probeQuad);
    const probeTarget=new THREE.RenderTarget(4,4,{type:THREE.FloatType});
    for(let pose=1;pose<=3;pose++) {
      matrix.makeRotationZ(pose*0.61);matrix.setPosition(300+pose*0.15,-120,0);
      contacts.update(renderer,1/60,21+pose/60,matrix);
      const curve=await inspectContacts();
      let strongest=0;
      for(let i=4;i<curve.values.length;i+=4) if(curve.values[i+3]>curve.values[strongest+3]) strongest=i;
      assert.ok(curve.values[strongest+3]>0.1,'rotated hull has an energetic contact');
      contactPixel.value.set(curve.values[strongest],curve.values[strongest+1]);
      renderer.setRenderTarget(probeTarget);renderer.render(probeScene,probeCamera);
      const pixel=await renderer.readRenderTargetPixelsAsync(probeTarget,1,1,1,1) as Float32Array;
      assert.ok(pixel[0]>0.05,`rendered contact foam must cover its measured hull contact at yaw ${pose*0.61}, got ${pixel[0]}`);
    }
    renderer.setRenderTarget(target);probeTarget.dispose();probeQuad.geometry.dispose();probeMaterial.dispose();
    for(const mesh of [contacts.mesh,contacts.curtain,exactSpray.mesh]) { mesh.geometry.dispose(); (mesh.material as THREE.Material).dispose(); }
    box.dispose(); waterGeometry.dispose();

  } finally {
    target.dispose(); renderer.dispose();
    for(const [name,descriptor] of saved) {
      if(descriptor) Object.defineProperty(globalThis,name,descriptor); else Reflect.deleteProperty(globalThis,name);
    }
    if(gpuDescriptor) Object.defineProperty(navigator,'gpu',gpuDescriptor); else Reflect.deleteProperty(navigator,'gpu');
  }
});
