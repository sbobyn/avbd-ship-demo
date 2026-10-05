import { createStorm } from './storm.ts';
import { SmoothTransition, nearestAngle, nextDayAngle } from './transition.ts';
import { createHullContacts } from './ocean/contact.ts';
import { createShipSpray } from './ocean/spray.ts';
import { createWhitewaterVolumes } from './ocean/volume.ts';
import { createWake } from './ocean/wake.ts';
import { createWhitecaps } from './ocean/whitecaps.ts';
import { createOceanGPU, updateOceanGPU } from './ocean/fft-gpu.ts';
import { CASCADES } from './ocean/model.ts';
import { surfaceSampler } from './ocean/sample.ts';
import * as THREE from 'three/webgpu';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import { godrays } from 'three/addons/tsl/display/GodraysNode.js';
import { bilateralBlur } from 'three/addons/tsl/display/BilateralBlurNode.js';
import { depthAwareBlend } from 'three/addons/tsl/display/depthAwareBlend.js';
import { cubeTexture, Loop, uint, BRDF_GGX, attribute, cross, dFdx, dFdy, cameraPosition, cameraWorldMatrix, cameraViewMatrix, cameraProjectionMatrix, cameraProjectionMatrixInverse, getViewPosition, reflect, dot, exp, float, Fn, instancedArray, fwidth, mix, normalize, pass, positionLocal, positionGeometry, positionWorld, screenCoordinate, smoothstep, uniform, uniformArray, reflector, screenUV, texture, perspectiveDepthToViewZ, positionView, vec2, vec3, vec4, wgslFn } from 'three/tsl';

export const ATMOSPHERES = {
  day: { elevation: 34, azimuth: -55, sun: '#fff2dc', power: 3.2, exposure: 0.85, cover: 0.43, density: 1.1, ambient: '#a4bed0', fog: '#9dbbc7', sea: '#063c50', wave: 0.65, rays: 0.025 },
  dawn: { elevation: 6, azimuth: 135, sun: '#ffb66a', power: 2.2, exposure: 0.95, cover: 0.40, density: 1.3, ambient: '#997e91', fog: '#be9c8e', sea: '#153940', wave: 0.55, rays: 0.07 },
  dusk: { elevation: 2, azimuth: 135, sun: '#f89c71', power: 1.35, exposure: 1.1, cover: 0.55, density: 1.5, ambient: '#a1a4bd', fog: '#403b49', sea: '#182839', wave: 0.65, rays: 0.04 },
  night: { elevation: -12, azimuth: -40, sun: '#9dbce8', power: 0.9, exposure: 0.95, cover: 0.35, density: 1.0, ambient: '#8297b8', fog: '#111d33', sea: '#071322', wave: 0.6, rays: 0.008 },
  storm: { elevation: 18, azimuth: -55, sun: '#bac8d1', power: 0.55, exposure: 1.1, cover: 0.73, density: 2.2, ambient: '#91a5b5', fog: '#4c626d', sea: '#102e37', wave: 1.15, rays: 0.035 },
} as const;
export type Atmosphere = keyof typeof ATMOSPHERES;

// A bounded slab march: multi-scale density, Beer extinction, and short sun marches.
// Inspired by the density/transmittance separation in the Web GPU Gems cloud/shafts labs.
const clouds = wgslFn(`
fn shipClouds(ray: vec3f, eye: vec3f, sun: vec3f, clock: f32, cover: f32, density: f32, light: vec3f, ambient: vec3f, pixel: vec2f) -> vec4f {
  if (ray.z < 0.015) { return vec4f(0.0); }
  let entry = max(0.0, (180.0 - eye.z) / ray.z);
  let exit = min((340.0 - eye.z) / ray.z, 14000.0);
  let step = min((exit - entry) / 40.0, 160.0);
  var trans = 1.0;
  var result = vec3f(0.0);
  for (var i = 0; i < 40; i++) {
    let p = eye + ray * (entry + (f32(i) + shipHash(vec3f(pixel, 3.7))) * step);
    let d = shipDensity(p, clock, cover);
    if (d > 0.001) {
      var optical = 0.0;
      for (var j = 1; j <= 3; j++) {
        optical += shipDensity(p + sun * f32(j) * 26.0, clock, cover) * 26.0;
      }
      let visibility = exp(-optical * density * 0.035);
      let phase = 0.35 + 0.65 * pow(max(dot(ray, sun), 0.0), 12.0);
      let shade = ambient * (0.32 + 0.50 * clamp((p.z - 180.0) / 160.0, 0.0, 1.0)) + light * visibility * phase;
      let alpha = 1.0 - exp(-d * density * step * 0.035);
      result += trans * alpha * shade;
      trans *= 1.0 - alpha;
      if (trans < 0.015) { break; }
    }
  }
  let fade = smoothstep(0.015, 0.12, ray.z);
  return vec4f(result * 0.35 / max(1.0 - trans, 0.001), (1.0 - trans) * fade);
}
fn shipHash(p: vec3f) -> f32 {
  var q = fract(p * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}
fn shipNoise(p: vec3f) -> f32 {
  let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(shipHash(i), shipHash(i + vec3f(1,0,0)),u.x), mix(shipHash(i + vec3f(0,1,0)),shipHash(i + vec3f(1,1,0)),u.x),u.y),
    mix(mix(shipHash(i + vec3f(0,0,1)),shipHash(i + vec3f(1,0,1)),u.x),mix(shipHash(i + vec3f(0,1,1)),shipHash(i + vec3f(1,1,1)),u.x),u.y),u.z);
}
fn shipDensity(p: vec3f, clock: f32, cover: f32) -> f32 {
  let q = (p + vec3f(clock * 2.5, clock * 0.7, 0.0)) * vec3f(0.006,0.006,0.012);
  let shape = shipNoise(q) * 0.60 + shipNoise(q * 2.13 + 17.3) * 0.25 + shipNoise(q * 4.31 + 7.1) * 0.15;
  let h = (p.z - 180.0) / 160.0;
  let envelope = smoothstep(0.0,0.16,h) * (1.0-smoothstep(0.65,1.0,h));
  let erosion=shipNoise(q*8.7+vec3f(9.1,3.7,11.3))*0.065;
  return smoothstep(1.0-cover, 1.22-cover, shape-erosion*(1.0-shape)) * envelope;
}
`);

export function createWeather(renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera, sun: THREE.DirectionalLight, fill: THREE.HemisphereLight, hull: THREE.Group) {
  // Mip-filtered periodic value noise avoids narrow procedural contour aliases.
  const noiseSize=256, noiseData=new Uint8Array(noiseSize*noiseSize*4);
  let noiseSeed=7319;
  const random=() => { noiseSeed^=noiseSeed<<13; noiseSeed^=noiseSeed>>>17; noiseSeed^=noiseSeed<<5; return (noiseSeed>>>0)/4294967296; };
  const grid=Float32Array.from({length:32*32},random);
  for(let y=0;y<noiseSize;y++) for(let x=0;x<noiseSize;x++) {
    const gx=x/8, gy=y/8, ix=Math.floor(gx), iy=Math.floor(gy);
    const fx=gx-ix, fy=gy-iy, u=fx*fx*(3-2*fx), v=fy*fy*(3-2*fy);
    const at=(a:number,b:number)=>grid[(b&31)*32+(a&31)];
    const n=(at(ix,iy)*(1-u)+at(ix+1,iy)*u)*(1-v)+(at(ix,iy+1)*(1-u)+at(ix+1,iy+1)*u)*v;
    const offset=(y*noiseSize+x)*4;
    // Periodic cellular ridges in G give foam membranes around irregular holes.
    // R remains smooth noise; B stores fine grain. All channels share mipmaps.
    let first=Infinity, second=Infinity;
    for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++) {
      const ax=ix+dx, ay=iy+dy;
      const distance=Math.hypot(gx-ax-0.15-at(ax,ay)*0.7,gy-ay-0.15-at(ax+11,ay+7)*0.7);
      if(distance<first) { second=first; first=distance; } else second=Math.min(second,distance);
    }
    noiseData[offset]=Math.round(n*255);
    noiseData[offset+1]=Math.round(Math.max(0,1-(second-first)*5)*255);
    noiseData[offset+2]=Math.round(random()*255); noiseData[offset+3]=255;
  }
  const foamTexture=new THREE.DataTexture(noiseData,noiseSize,noiseSize);
  foamTexture.wrapS=foamTexture.wrapT=THREE.RepeatWrapping;
  foamTexture.magFilter=THREE.LinearFilter; foamTexture.minFilter=THREE.LinearMipmapLinearFilter;
  foamTexture.generateMipmaps=true; foamTexture.needsUpdate=true;
  const foamNoise=(p: THREE.Node<'vec2'>) => texture(foamTexture,p.mul(1/32)).r;
  const hullPosition = uniform(new THREE.Vector3()), speed = uniform(0);
  scene.add(sun.target);
  const storm=createStorm(scene);
  const time = uniform(0), wave = uniform(0.65), cover = uniform(0.43), density = uniform(1.1), foamWind=uniform(7);
  const light = uniform(new THREE.Color()), ambient = uniform(new THREE.Color());
  const sunDirection = uniform(new THREE.Vector3());
  const skyScene = new THREE.Scene(), sky = new SkyMesh();
  sky.scale.setScalar(4500); sky.cloudCoverage.value = 0;
  skyScene.add(sky);
  const cube = new THREE.CubeRenderTarget(1024, { type: THREE.HalfFloatType });
  const cubeCamera = new THREE.CubeCamera(0.1, 6000, cube);
  const environmentCube=new THREE.CubeRenderTarget(128,{type:THREE.HalfFloatType});
  const environmentCamera=new THREE.CubeCamera(0.1,6000,environmentCube);
  const pmrem = new THREE.PMREMGenerator(renderer);
  let environment: THREE.RenderTarget | undefined;
  scene.backgroundRotation.x = Math.PI / 2;
  scene.environmentRotation.x = Math.PI / 2;
  scene.background = cube.texture;

  // Fixed celestial directions; tiny, varied stars render behind the cloud slab.
  const starVisibility=uniform(0), starPositions:number[]=[], starColors:number[]=[], starSizes:number[]=[];
  for(let i=0;i<4200;i++) {
    const z=random(), angle=random()*Math.PI*2, radius=Math.sqrt(1-z*z);
    starPositions.push(Math.cos(angle)*radius*4900,Math.sin(angle)*radius*4900,z*4900);
    const brightness=(0.2+Math.pow(random(),4)*0.8)*Math.min(1,z*8);
    const warmth=random();
    starColors.push(brightness*(0.78+warmth*0.22),brightness*0.88,brightness*(1-warmth*0.2));
    starSizes.push(1.1+Math.pow(random(),5)*1.3);
  }
  const starGeometry=new THREE.BufferGeometry();
  starGeometry.setAttribute('position',new THREE.Float32BufferAttribute(starPositions,3));
  starGeometry.setAttribute('color',new THREE.Float32BufferAttribute(starColors,3));
  starGeometry.setAttribute('starSize',new THREE.Float32BufferAttribute(starSizes,1));
  const starMaterial=new THREE.PointsNodeMaterial({transparent:true,depthWrite:false,fog:false,sizeAttenuation:false,toneMapped:true});
  starMaterial.sizeNode=attribute('starSize','float'); starMaterial.opacityNode=starVisibility;
  // HDR radiance survives Fresnel attenuation and the mirror texture filter.
  starMaterial.colorNode=attribute('color','vec3').mul(3);
  const stars=new THREE.Points(starGeometry,starMaterial);
  stars.frustumCulled=false; stars.renderOrder=-11; scene.add(stars);

  const cloudMaterial = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, transparent: true, depthWrite: false, fog: false });
  cloudMaterial.outputNode = clouds(normalize(positionWorld.sub(cameraPosition)), cameraPosition, sunDirection, time, cover, density, light, ambient, screenCoordinate);
  const cloudMesh = new THREE.Mesh(new THREE.SphereGeometry(5000, 24, 12), cloudMaterial);
  cloudMesh.frustumCulled = false; cloudMesh.renderOrder = -10; scene.add(cloudMesh);

  // Concentrate vertices around the ship; distant water only needs its silhouette.
  const geometry = new THREE.PlaneGeometry(2, 2, 384, 384);
  const points = geometry.getAttribute('position');
  for (let i = 0; i < points.count; i++) {
    const warp = (v: number): number => Math.sign(v) * (Math.abs(v) * 45 + Math.abs(v) ** 4 * 7000);
    points.setXYZ(i, warp(points.getX(i)), warp(points.getY(i)), 0);
  }
  const fftOcean=createOceanGPU(CASCADES.map((c,i) => ({...c, n:128, choppiness:c.choppiness*1.15, fetch:c.fetch, amplitude:c.amplitude*(i===0?1.35:i===1?112:32)})));
  const sampleSurface=surfaceSampler(fftOcean);
  const whitecaps=createWhitecaps(fftOcean,wave,smoothstep(4,12,foamWind));
  const hullContacts=createHullContacts();
  const wakeField=createWake(hullContacts);
  const oceanUV=positionGeometry.xy.add(hullPosition.xy);
  const fields=(point: THREE.Node<'vec2'>) => fftOcean.cascades.map((c,i) => ({
    displacement: sampleSurface(i,point.div(c.length).add(0.5),0),
    normal: sampleSurface(i,point.div(c.length).add(0.5),1),
  }));
  // Reuse the 10 Hz buoyancy readback for contact-driven sound, without another GPU sync.
  const probeBuffer=instancedArray(4,'vec4');
  const probeOffsets=[[7,0],[-7,0],[0,2],[0,-2]];
  const probePoints=Array.from({length:4},()=>new THREE.Vector3());
  const probeLocations=uniformArray<'vec3'>(probePoints,'vec3');
  const probeStep=Fn(() => {
    probeOffsets.forEach((_,i) => {
      const target=probeLocations.element(i).xy;
      const q=target.toVar();
      for(let iteration=0;iteration<3;iteration++) {
        const horizontal=vec2(0).toVar();
        for(const field of fields(q)) horizontal.addAssign(field.displacement.xz.mul(wave));
        q.assign(target.sub(horizontal));
      }
      const height=float(0).toVar();
      for(const field of fields(q)) height.addAssign(field.displacement.y.mul(wave));
      height.addAssign(wakeField.sample(target).x);
      probeBuffer.element(i).assign(vec4(height,0,0,0));
    });
    const contact=float(0).toVar(), impact=float(0).toVar();
    Loop({start:uint(0),end:bowImpacts.activeCount,type:'uint',condition:'<'},({i})=>{
      const energy=bowImpacts.sources.element(i).w.mul(bowImpacts.launches.element(i).w);
      contact.addAssign(energy);
      impact.addAssign(energy.mul(bowImpacts.ends.element(i).w));
    });
    probeBuffer.element(0).y.assign(contact); probeBuffer.element(0).z.assign(impact);
  })().compute(1);
  const sound={rain:0,contact:0,impact:0,sample:0,strike:0,strikePosition:storm.sound.strikePosition};
  const waterPose=new THREE.Vector3();
  let probeBusy=false, probeTime=-1, probeGeneration=0;
  const displacement=Fn(() => {
    const p=positionLocal.toVar();
    for (const field of fields(oceanUV)) p.addAssign(field.displacement.xzy.mul(wave));
    p.z.addAssign(wakeField.sample(p.xy.add(hullPosition.xy)).x);
    return p;
  });
  const shadingFields=fields(oceanUV).map(field=>({displacement:field.displacement.toVar(),normal:field.normal.toVar()}));
  const waveFrame=Fn(() => {
    const slope=vec2(0).toVar(), horizontal=vec2(0).toVar(), variance=float(0).toVar();
    shadingFields.forEach((field,i) => {
      const footprint=fwidth(oceanUV).length().mul(fftOcean.n/fftOcean.cascades[i].length);
      const resolved=float(1).sub(smoothstep(0.5,2,footprint));
      slope.addAssign(field.normal.xy.mul(wave).mul(resolved));
      horizontal.addAssign(field.displacement.xz.mul(wave).mul(resolved));
      variance.addAssign(field.normal.xy.dot(field.normal.xy).mul(wave.pow2()).mul(float(1).sub(resolved.pow2())));
    });
    slope.addAssign(storm.ripples(oceanUV));
    const wakeSlope=vec2(
      wakeField.sample(positionWorld.xy.add(vec2(0.5,0))).x.sub(wakeField.sample(positionWorld.xy.sub(vec2(0.5,0))).x),
      wakeField.sample(positionWorld.xy.add(vec2(0,0.5))).x.sub(wakeField.sample(positionWorld.xy.sub(vec2(0,0.5))).x));
    // Screen derivatives recover the horizontal choppiness Jacobian without
    // additional FFT taps. Height slopes are differentiated in their own domain;
    // the wake is differentiated at the displaced world position.
    const dx=vec3(dFdx(oceanUV.add(horizontal)),dot(slope,dFdx(oceanUV)).add(dot(wakeSlope,dFdx(positionWorld.xy))));
    const dy=vec3(dFdy(oceanUV.add(horizontal)),dot(slope,dFdy(oceanUV)).add(dot(wakeSlope,dFdy(positionWorld.xy))));
    const n=cross(dx,dy);
    const oriented=n.z.lessThan(0).select(n.negate(),n);
    const normal=oriented.length().greaterThan(0.0000001).select(normalize(oriented),normalize(vec3(slope.add(wakeSlope).negate(),1)));
    return vec4(normal,variance);
  });
  const wakeInspection=uniform(0);
  const water=new THREE.MeshBasicNodeMaterial({color:'#063c50'});
  water.positionNode=displacement();
  // Displaced water/wake crests also obey the ship exclusion, not only spray.
  water.maskNode=hullContacts.outsideWater(positionWorld);
  const frame=waveFrame(), waterNormal=frame.xyz;
  // Unresolved normal detail becomes a wider highlight rather than flickering
  // glints. Reuse Three's GGX/Smith/Schlick implementation in world space.
  const normalWidth=fwidth(waterNormal);
  const roughness=float(0.035).pow(4).add(frame.w.mul(0.035)).add(dot(normalWidth,normalWidth).mul(0.12)).pow(0.25).clamp(0.055,0.38);
  const crestHeight=Fn(() => {
    const height=float(0).toVar();
    for(const field of shadingFields) height.addAssign(field.displacement.y.mul(wave));
    return height;
  })();
  // A real scene reflection, perturbed by the FFT slope, retains the hull and sky.
  const reflection=reflector({resolutionScale:0.5,bounces:false,generateMipmaps:true});
  // Distort by the wave perturbation, not the flat plane's view-space normal.
  const viewNormal=cameraViewMatrix.mul(vec4(waterNormal.sub(vec3(0,0,1)),0)).xyz;
  // Project the change in reflected ray direction. A constant UV offset made
  // metre-scale waves barely perturb the sky and left the hull mirror-smooth.
  const eye=normalize(cameraPosition.sub(positionWorld));
  const reflectionOffset=Fn(() => {
    const ray=reflect(eye.negate(),waterNormal);
    const flatRay=reflect(eye.negate(),vec3(0,0,1));
    const project=(direction: THREE.Node<'vec3'>) => {
      const mirrored=vec3(direction.xy,direction.z.negate());
      const view=cameraViewMatrix.mul(vec4(mirrored,0));
      const clip=cameraProjectionMatrix.mul(vec4(view.x.negate(),view.yz,0));
      return clip.xy.div(clip.w.max(0.1)).mul(vec2(0.5,-0.5));
    };
    const offset=project(ray).sub(project(flatRay));
    return offset.div(float(1).add(offset.length().div(0.12)));
  })();
  // Screen UVs are valid only on the flat reflector plane. Project the actual
  // displaced water point through the mirrored view; otherwise an orbit slides
  // the hull/splash silhouette sideways over the wave surface.
  const reflectedPoint=vec3(positionWorld.xy,float(-0.9).sub(positionWorld.z));
  const mirroredView=cameraViewMatrix.mul(vec4(reflectedPoint,1));
  const reflectedClip=cameraProjectionMatrix.mul(vec4(mirroredView.x.negate(),mirroredView.yz,1));
  const reflectionUV=reflectedClip.xy.div(reflectedClip.w.max(0.001)).mul(vec2(0.5,-0.5)).add(0.5);
  const reflectedUV=reflectionUV.add(reflectionOffset);
  reflection.uvNode=reflectedUV.clamp(0.002,0.998);
  const edges=reflectedUV.min(vec2(1).sub(reflectedUV));
  const edge=edges.x.min(edges.y);
  const reflectionConfidence=smoothstep(0.005,0.06,edge).mul(smoothstep(0.01,0.2,reflectedClip.w));
  const reflectedRay=reflect(eye.negate(),waterNormal);
  const skyBrightness=uniform(1);
  const skyReflection=cubeTexture(cube.texture,vec3(reflectedRay.x,reflectedRay.z,reflectedRay.y.negate())).rgb.mul(skyBrightness);
  // Preserve subpixel starlight on glass; rough seas still use the filtered footprint.
  const reflectionClarity=uniform(1);
  const reflectionMip=smoothstep(0.15,2,fwidth(oceanUV).length()).mul(2.5).add(roughness.mul(4)).mul(reflectionClarity);
  const filteredReflection=mix(skyReflection,reflection.level(reflectionMip).rgb,reflectionConfidence);
  const sceneTarget=new THREE.RenderTarget(1,1,{type:THREE.HalfFloatType,depthTexture:new THREE.DepthTexture(1,1)});
  sceneTarget.depthTexture!.type=THREE.FloatType;
  const screenDepth=texture(sceneTarget.depthTexture!,screenUV);
  const backgroundZ=perspectiveDepthToViewZ(screenDepth.r,float(camera.near),float(camera.far));
  const thickness=positionView.z.sub(backgroundZ).max(0).min(120);
  const refractedUV=screenUV.add(viewNormal.xy.mul(thickness.min(2)).mul(0.018)).clamp(0.001,0.999);
  const displacedZ=perspectiveDepthToViewZ(texture(sceneTarget.depthTexture!,refractedUV).r,float(camera.near),float(camera.far));
  const safeUV=displacedZ.greaterThan(positionView.z).select(screenUV,refractedUV);
  // Absorption must use the depth at the accepted refraction sample. Foreground
  // hull pixels are not underwater: reject them instead of smearing their silhouette.
  const acceptedZ=perspectiveDepthToViewZ(texture(sceneTarget.depthTexture!,safeUV).r,float(camera.near),float(camera.far));
  const opticalDepth=acceptedZ.greaterThanEqual(positionView.z).select(float(120),positionView.z.sub(acceptedZ).min(120));
  const depthConfidence=float(1).sub(smoothstep(0.15,1.5,fwidth(opticalDepth)));
  const sampleDepth=texture(sceneTarget.depthTexture!,safeUV).r;
  const sampleWorld=cameraWorldMatrix.mul(vec4(getViewPosition(safeUV,sampleDepth,cameraProjectionMatrixInverse),1)).xyz;
  const submerged=smoothstep(0.02,0.25,positionWorld.z.sub(sampleWorld.z));
  const transmission=exp(vec3(-0.8,-0.38,-0.24).mul(opticalDepth)).mul(depthConfidence).mul(submerged);
  const refracted=texture(sceneTarget.texture,safeUV).rgb.mul(transmission)
    .add(vec3(1).sub(transmission).mul(uniform(water.color)));
  const fresnel=float(0.0204).add(float(1).sub(dot(waterNormal,eye).max(0)).pow(5).mul(0.9796));
  const backlight=dot(eye.negate(),sunDirection).max(0).pow(3);
  const crestLight=smoothstep(-0.15,0.9,crestHeight).mul(float(0.22).add(backlight.mul(0.78)));
  const scattering=vec3(0.015,0.22,0.16).mul(crestLight).mul(vec3(light.r,light.g,light.b));
  const troughShade=smoothstep(-1.1,0.4,crestHeight).mul(0.45).add(0.55);
  const specular=(normal:THREE.Node<'vec3'>,surfaceRoughness:THREE.Node<'float'>,f0:number) => {
    const parameters={lightDirection:sunDirection,viewDirection:eye,normalView:normal,f0:vec3(f0),f90:float(1),roughness:surfaceRoughness};
    // The installed declarations predate BRDF_GGX's normalView override and
    // its TSL-proxied vec3 return; both are present in Three's implementation.
    return (BRDF_GGX(parameters) as unknown as THREE.Node<'vec3'>)
      .mul(dot(normal,sunDirection).max(0)).mul(vec3(light.r,light.g,light.b));
  };
  const glint=specular(waterNormal,roughness,0.0204);
  // Advected, domain-warped foam: narrow filaments with holes, not painted white blobs.
  const wakeSample=wakeField.sample(positionWorld.xy), foamHistory=wakeField.sampleHistory(positionWorld.xy);
  const foamAge=foamHistory.z.div(wakeSample.z.max(0.0001)).clamp(0,120);
  // Density already advects in the solver. Reconstructing texture travel from
  // instantaneous velocity * age teleports the pattern when young/old water
  // mixes behind the rudder. Keep the detail domain continuous in world space.
  const drift=positionWorld.xy;
  const aged=smoothstep(2,18,foamAge);
  const warp=vec2(foamNoise(drift.mul(0.31)),foamNoise(drift.mul(0.27).add(41.7))).sub(0.5);
  const secondaryWarp=vec2(foamNoise(drift.mul(1.37).add(17.3)),foamNoise(drift.mul(1.91).sub(9.1))).sub(0.5);
  const foamUV=drift.add(warp.mul(3.7)).add(secondaryWarp.mul(0.65));
  const patches=foamNoise(foamUV.mul(1.6));
  const breakup=foamNoise(foamUV.mul(11));
  const filaments=foamNoise(foamUV.mul(40));
  const farFilter=smoothstep(0.2,0.9,fwidth(foamUV.mul(40)).length());
  const bubbles=mix(filaments,float(0.33),farFilter).mul(0.65).add(breakup.mul(0.35));
  const wake=wakeSample.z;
  // Whitecaps follow spectral compression instead of arbitrary height/noise patches.
  const crestState=Fn(() => {
    const state=vec2(0).toVar();
    for (let i=0;i<fftOcean.cascades.length;i++) {
      const resolved=float(1).sub(smoothstep(1,4,fwidth(oceanUV).length().mul(fftOcean.n/fftOcean.cascades[i].length)));
      const candidate=whitecaps.sampleState(i,oceanUV).mul(resolved).toVar();
      state.assign(candidate.x.greaterThan(state.x).select(candidate,state));
    }
    return state;
  })();
  const crestAge=crestState.y.div(crestState.x.max(0.0001));
  const crestYoung=float(1).sub(smoothstep(0.45,3,crestAge));
  const crestCoverage=float(1).sub(exp(crestState.x.pow(1.35).mul(-3.5)));
  // Coverage is controlled once by the transported density. Old foam opens
  // into cellular membranes; fresh hull contact keeps a dense, granular core.
  // Incommensurate, rotated samples and smoothly varying blend weights break
  // the repeated cellular lattice without tile boundaries or extra textures.
  // Keep the detail frame fixed in world space. Rotating absolute world UVs
  // by instantaneous flow made stern vortices flip the entire foam pattern
  // whenever the rudder's opposing streams crossed through zero velocity.
  const stretched=foamUV;
  const rotated=vec2(stretched.x.mul(0.8).sub(stretched.y.mul(0.6)),stretched.x.mul(0.6).add(stretched.y.mul(0.8)));
  const detailA=texture(foamTexture,stretched.mul(5/32));
  const detailB=texture(foamTexture,rotated.mul(6.87/32).add(vec2(0.317,0.731)));
  const foamDetail=mix(detailA,detailB,foamNoise(foamUV.mul(0.17)).mul(0.7).add(0.15));
  // Fine crest membranes follow the displaced crest domain. Two rotated,
  // incommensurate samples open into branching holes without a visible tile grid.
  const crestUV=foamUV.add(shadingFields[0].displacement.xz.mul(wave).mul(0.4)).sub(vec2(0.035,0.012).mul(time));
  const crestA=texture(foamTexture,crestUV.mul(3.4/32));
  const crestB=texture(foamTexture,vec2(crestUV.x.mul(0.6).sub(crestUV.y.mul(0.8)),crestUV.x.mul(0.8).add(crestUV.y.mul(0.6))).mul(5.17/32).add(vec2(0.413,0.173)));
  const membrane=crestA.g.max(crestB.g.mul(0.85));
  const membraneAA=fwidth(membrane).max(0.08);
  const strands=smoothstep(float(0.35).sub(membraneAA),float(0.35).add(membraneAA),membrane);
  const crestHoles=smoothstep(0.2,0.6,crestA.r.mul(0.6).add(crestB.r.mul(0.4)));
  const crestPattern=mix(strands.mul(crestHoles.mul(0.65).add(0.35)),float(0.4).add(crestB.g.mul(0.35)).add(foamDetail.b.mul(0.2)),crestYoung);
  const crestFroth=crestCoverage.mul(crestPattern).mul(0.75);
  const cells=smoothstep(0.25,0.7,foamDetail.g.mul(0.35).add(detailB.r.mul(0.4)).add(breakup.mul(0.25)));
  const holes=smoothstep(aged.mul(0.2).add(0.18),aged.mul(0.2).add(0.5),patches.mul(0.7).add(detailB.r.mul(0.3)));
  // Transported bubbles survive the whole wave cycle. Gating their opacity on
  // instantaneous height made the short stern waves flash the foam on and off.
  const wakeDensity=wake.mul(1.15).clamp(0,1);
  // Even low-energy tangential contact (including the thin rudder) creates
  // fresh microbubbles. Zero contact still produces no permanent white collar.
  const contactPressure=hullContacts.sampleFoam(positionWorld.xy);
  const contactFoam=float(1).sub(exp(contactPressure.mul(-3.4))).mul(smoothstep(0,0.035,contactPressure)).clamp(0,1);
  const densityMask=wakeDensity;
  const breakupSignal=patches.mul(0.6).add(breakup.mul(0.4));
  const breakupAA=fwidth(breakupSignal).max(0.07);
  const breakupCenter=float(0.75).sub(densityMask.mul(0.5));
  const breakupMask=smoothstep(breakupCenter.sub(breakupAA),breakupCenter.add(breakupAA),breakupSignal);
  const fresh=float(1).sub(smoothstep(0.8,6,foamAge)).mul(smoothstep(0.15,0.65,densityMask)).max(crestYoung.mul(crestCoverage)).max(contactFoam);
  const lace=mix(cells.mul(0.85).add(bubbles.mul(0.15)).mul(holes),float(0.35).add(foamDetail.b.mul(0.4)).add(breakup.mul(0.2)),fresh);
  const transported=densityMask.mul(breakupMask).mul(lace).mul(float(1).sub(aged.mul(0.25)));
  // Fresh contact belongs to the moving hull frame, not the world-anchored
  // texture of water already left behind. Full inverse pose includes roll/yaw.
  const contactUV=hullContacts.localPoint(positionWorld).xy;
  const contactDetail=texture(foamTexture,contactUV.mul(5/32));
  const contactFine=foamNoise(contactUV.mul(11));
  const contactGrains=smoothstep(0.22,0.7,contactDetail.g.mul(0.5).add(contactFine.mul(0.5))).mul(0.65).add(0.3);
  const contactFroth=contactFoam.mul(contactGrains).mul(0.9);
  const foam=float(1).sub(float(1).sub(transported).mul(float(1).sub(contactFroth)).mul(float(1).sub(crestFroth))).clamp(0,0.96);
  // Millimetre foam relief has its own bump frame and lighting. Derivatives use
  // the already sampled mip-filtered grains; no per-bubble geometry is needed.
  const surfaceDetail=mix(foamDetail,contactDetail,contactFoam);
  const relief=surfaceDetail.g.mul(0.004).add(surfaceDetail.r.mul(0.007)).add(crestA.g.mul(crestFroth).mul(0.006)).mul(float(1).sub(farFilter));
  const r1=cross(dFdy(positionWorld),waterNormal), r2=cross(waterNormal,dFdx(positionWorld));
  const determinant=dot(dFdx(positionWorld),r1);
  const safeDet=determinant.lessThan(0).select(determinant.abs().max(0.000001).negate(),determinant.abs().max(0.000001));
  const gradient=r1.mul(dFdx(relief)).add(r2.mul(dFdy(relief))).div(safeDet);
  const foamNormal=normalize(waterNormal.sub(gradient.div(float(1).add(gradient.length()))));
  const cavity=surfaceDetail.g.mul(0.22).add(0.78).mul(mix(float(0.92),float(1.06),fresh));
  const foamDiffuse=dot(foamNormal,sunDirection).max(0).mul(0.65).add(0.12);
  const illumination=vec3(ambient.r,ambient.g,ambient.b).mul(0.8).add(vec3(light.r,light.g,light.b).mul(foamDiffuse));
  const foamColor=vec3(0.86,0.92,0.88).mul(illumination).mul(cavity)
    .add(specular(foamNormal,mix(float(0.85),float(0.42),fresh.mul(float(1).sub(foamHistory.w.mul(0.35)))),0.025).mul(0.35));
  // Entrained bubbles scatter beneath the surface, so reflections remain over
  // the milky blue-green trail rather than being replaced by a white decal.
  const aerationGaps=smoothstep(0.2,0.72,patches.mul(0.65).add(breakup.mul(0.35))).mul(0.8).add(0.2);
  const aeration=float(1).sub(exp(wakeSample.w.mul(-1.6))).mul(aerationGaps)
    .mul(foamHistory.w.mul(0.65).add(0.3));
  const underWater=mix(refracted.mul(troughShade).add(scattering),vec3(0.12,0.34,0.31).mul(illumination),aeration);
  const waterBeauty=mix(mix(underWater,filteredReflection,fresnel).add(glint),foamColor,foam);
  water.colorNode=mix(waterBeauty,vec3(foam),wakeInspection);
  const ocean = new THREE.Mesh(geometry,water); ocean.position.z=-0.45; ocean.receiveShadow=true; ocean.frustumCulled=false; scene.add(ocean); ocean.add(reflection.target);

  hullContacts.setWater(geometry,displacement());
  storm.setBoundary(hullContacts.surface,hullContacts.outsideSolid);
  storm.setLensContacts(hullContacts,hullContacts.surface);
  // Drops collide against the same captured, piecewise-triangle water surface.
  const spraySurface=Fn(([point]: [THREE.Node<'vec2'>])=>hullContacts.surface(point));
  const bowImpacts=hullContacts; scene.add(bowImpacts.mesh,bowImpacts.curtain);
  const particles=createShipSpray(camera,sceneTarget.depthTexture!,spraySurface,
    vec3(light.r,light.g,light.b),vec3(ambient.r,ambient.g,ambient.b),sunDirection,bowImpacts,wakeField.deposit);
  const spray=particles.mesh; scene.add(spray);
  const sprayWind=new THREE.Vector3();
  const volumes=createWhitewaterVolumes(camera,sceneTarget.depthTexture!,bowImpacts,vec3(light.r,light.g,light.b),vec3(ambient.r,ambient.g,ambient.b),sunDirection,wakeField.deposit);

  const beauty = pass(scene,camera), depth = beauty.getTextureNode('depth');
  const rays = godrays(depth,camera,sun); rays.raymarchSteps.value=32; rays.resolutionScale=0.5; rays.maxDensity.value=0.025;
  const blur = bilateralBlur(rays.getTextureNode());
  const pipeline = new THREE.RenderPipeline(renderer);
  const volumeBeauty=volumes.resolve(beauty.getTextureNode('output'));
  const composite = volumes.resolve(depthAwareBlend(beauty.getTextureNode('output'),blur.getTextureNode(),depth,camera,{blendColor: light}));
  const stormComposite=storm.lens(composite),stormBeauty=storm.lens(volumeBeauty);
  pipeline.outputNode = stormComposite;
  // Model-space centre of the stern lantern's glass chamber. Parenting keeps
  // the light aligned through heave, roll and navigation without a readback.
  const sternLantern=new THREE.PointLight('#ffad59',0,6,2);
  sternLantern.name='Stern lantern'; sternLantern.position.set(-10.08,0.003,5.60); hull.add(sternLantern);
  const lanternGlow=uniform(0);
  const settings=(name: Atmosphere) => {
    const p=ATMOSPHERES[name], sunlight=new THREE.Color(p.sun), skyFill=new THREE.Color(p.ambient);
    const sea=new THREE.Color(p.sea), fog=new THREE.Color(p.fog);
    return {
      elevation:(name==='night'?28:p.elevation)*Math.PI/180, azimuth:p.azimuth*Math.PI/180,
      sunR:sunlight.r,sunG:sunlight.g,sunB:sunlight.b,power:p.power,exposure:p.exposure,
      ambientR:skyFill.r,ambientG:skyFill.g,ambientB:skyFill.b,
      seaR:sea.r,seaG:sea.g,seaB:sea.b,fogR:fog.r,fogG:fog.g,fogB:fog.b,
      cover:p.cover,density:p.density,wave:p.wave,rays:p.rays,
      rain:name==='storm'?1:0,
      stars:name==='night'?1:0,
      lantern:name==='day'?0:18,
      fill:name==='night'?0.5:name==='storm'?1.1:name==='dusk'?0.95:0.45,
      cloudLight:name==='night'?0.18:p.power*0.6,
      cloudAmbient:name==='night'?0.18:name==='storm'?1.0:name==='dusk'?1.1:0.85,
      fogNear:name==='storm'?100:450,fogFar:name==='storm'?1700:5000,
      turbidity:name==='storm'?15:3,
      background:name==='night'?0.014:name==='storm'?0.25:name==='dusk'?0.28:0.18,
      environment:name==='night'?0.055:name==='storm'?0.35:name==='dusk'?0.32:0.12,
    };
  };
  const transition=new SmoothTransition(settings('day'));
  let atmosphere:Atmosphere='day';
  const seaState=new SmoothTransition({ scale: 1 });
  const fog=new THREE.Fog('#9dbbc7',450,5000); scene.fog=fog;
  let lastSkyCapture=-Infinity, skyCapturePending=false;
  const updateAtmosphere=(changed=true) => {
    skyCapturePending ||= changed;
    const p=transition.values;
    storm.intensity.value=p.rain; starVisibility.value=p.stars;
    sternLantern.intensity=p.lantern; lanternGlow.value=p.lantern/18;
    sunDirection.value.set(Math.cos(p.azimuth)*Math.cos(p.elevation),Math.sin(p.azimuth)*Math.cos(p.elevation),Math.sin(p.elevation));
    sun.position.copy(sun.target.position).addScaledVector(sunDirection.value,50);
    sun.color.setRGB(p.sunR,p.sunG,p.sunB); sun.intensity=p.power;
    fill.color.setRGB(p.ambientR,p.ambientG,p.ambientB); fill.groundColor.setRGB(p.seaR,p.seaG,p.seaB); fill.intensity=p.fill;
    light.value.copy(sun.color).multiplyScalar(p.cloudLight);
    ambient.value.copy(fill.color).multiplyScalar(p.cloudAmbient);
    cover.value=p.cover; density.value=p.density; water.color.copy(fill.groundColor);
    renderer.toneMappingExposure=p.exposure;
    fog.color.setRGB(p.fogR,p.fogG,p.fogB); fog.near=p.fogNear; fog.far=p.fogFar;
    sky.sunPosition.value.set(sunDirection.value.x,sunDirection.value.z,-sunDirection.value.y);
    sky.turbidity.value=p.turbidity; sky.rayleigh.value=2;
    // Reuse the filtered target; a smaller lighting cube keeps animated captures affordable.
    const captureTime=performance.now();
    if(skyCapturePending && captureTime-lastSkyCapture>=150) {
      lastSkyCapture=captureTime; skyCapturePending=false;
      sky.showSunDisc.value=0; environmentCamera.update(renderer,skyScene);
      environment=pmrem.fromCubemap(environmentCube.texture,environment); scene.environment=environment.texture;
      sky.showSunDisc.value=1; cubeCamera.update(renderer,skyScene);
    }
    scene.backgroundIntensity=p.background; scene.environmentIntensity=p.environment;
    rays.density.value=p.rays;
  };
  const apply=(name: Atmosphere): void => {
    const target=settings(name);
    target.azimuth=(atmosphere==='night'&&name==='dawn'?nextDayAngle:nearestAngle)(transition.values.azimuth,target.azimuth);
    atmosphere=name;
    transition.setTarget(target);
    document.body.dataset.atmosphere=name;
  };
  updateAtmosphere(); document.body.dataset.atmosphere='day';
  let splashVisible=true, qualityLevel=1;
  const renderSize=new THREE.Vector2();
  return { apply, sound, inspectWake(mode:string) {wakeInspection.value=mode==='foam'?1:0;splashVisible=mode==='beauty';}, triggerLightning:()=>storm.triggerLightning(hull.position), waterPose, lanternGlow, setHull(envelope: Parameters<typeof wakeField.setHull>[0],geometries:THREE.BufferGeometry[]) {
    wakeField.setHull(envelope); particles.setHullBeam(envelope.beam);
    hullContacts.setHull(geometries);
  }, render: () => {
    renderer.getDrawingBufferSize(renderSize);
    const w=Math.max(1,Math.floor(renderSize.x)), h=Math.max(1,Math.floor(renderSize.y));
    if(sceneTarget.width!==w || sceneTarget.height!==h) sceneTarget.setSize(w,h);
    ocean.visible=false; spray.visible=false; bowImpacts.mesh.visible=false; bowImpacts.curtain.visible=false;
    const cloudsVisible=cloudMesh.visible; cloudMesh.visible=false;
    const previous=renderer.getRenderTarget();
    renderer.setRenderTarget(sceneTarget); renderer.render(scene,camera); renderer.setRenderTarget(previous);
    ocean.visible=true; spray.visible=splashVisible; bowImpacts.mesh.visible=splashVisible; bowImpacts.curtain.visible=splashVisible; cloudMesh.visible=cloudsVisible;
    volumes.mesh.visible=splashVisible; volumes.render(renderer); pipeline.render();
  }, update(t: number, wind: number, heave: number, elapsed?: number, glass = false, windDirection = -160, heading = 0) {
    if(t<time.value) { probeGeneration++; probeTime=-1; waterPose.set(0,0,0); bowImpacts.reset(renderer);
      particles.reset(renderer); volumes.reset(renderer); speed.value=0; }
    const dt=Math.max(0,Math.min(0.1,t-time.value)), blend=1-Math.exp(-dt*1.5);
    const atmosphereChanged=transition.advance(Math.max(0,Math.min(0.1,elapsed??dt)));
    if(atmosphereChanged || skyCapturePending) updateAtmosphere(atmosphereChanged);
    const hullSpeed=dt>0?Math.hypot(hull.position.x-hullPosition.value.x,hull.position.y-hullPosition.value.y)/dt:0;
    if(dt>0) speed.value+=(Math.min(hullSpeed,4)-speed.value)*blend;
    hullPosition.value.copy(hull.position);
    ocean.position.x=hull.position.x; ocean.position.y=hull.position.y;
    // Keep the shadow volume level and texel-stable while the hull heaves/rolls.
    const shadowTexel=50/sun.shadow.mapSize.x;
    sun.target.position.set(Math.round(hull.position.x/shadowTexel)*shadowTexel,Math.round(hull.position.y/shadowTexel)*shadowTexel,0);
    sun.position.copy(sun.target.position).addScaledVector(sunDirection.value,50);
    seaState.setTarget({ scale: glass ? 0.025 : 1 }); seaState.advance(dt);
    // Tiny celestial points disappear in the half-resolution mirror. Only glass
    // pays for full resolution; the same scene stars remain cloud-occluded.
    reflection.reflector.resolutionScale=qualityLevel===0?0.25:qualityLevel===1?(glass?0.65:0.4):(glass?1:0.5);
    reflectionClarity.value=0.15+0.85*seaState.values.scale;
    wave.value+=((transition.values.wave*(0.6+wind/30)+heave*0.3)*seaState.values.scale-wave.value)*blend;
    foamWind.value=wind;
    updateOceanGPU(renderer,fftOcean,t);
    hull.updateMatrixWorld(); hullContacts.update(renderer,dt,t,hull.matrixWorld);
    whitecaps.update(renderer,dt);
    wakeField.update(renderer,t,hull.position.x,hull.position.y,heading);
    if(!probeBusy && t-probeTime>=0.1) {
      probeBusy=true; probeTime=t; const generation=probeGeneration;
      probeOffsets.forEach(([x,y],i)=>probePoints[i].set(x,y,-0.45).applyQuaternion(hull.quaternion).add(hull.position));
      renderer.compute(probeStep);
      void renderer.getArrayBufferAsync(probeBuffer.value as THREE.BufferAttribute).then(buffer => {
        if(generation!==probeGeneration) return;
        const h=new Float32Array(buffer);
        if(Number.isFinite(h[1])&&Number.isFinite(h[2])) {sound.contact=h[1];sound.impact=h[2];sound.sample++;}
        if([h[0],h[4],h[8],h[12]].every(Number.isFinite)) {
          waterPose.set((h[0]+h[4]+h[8]+h[12])/4,
            Math.atan2(h[8]-h[12],4),-Math.atan2(h[0]-h[4],14));
        }
      }).catch(error => console.warn('Ocean probe readback failed',error)).finally(() => { probeBusy=false; });
    }
    particles.update(renderer,dt,speed.value,hull.position,sprayWind.set(Math.cos(windDirection*Math.PI/180)*wind*0.12,Math.sin(windDirection*Math.PI/180)*wind*0.12,0),heading);
    volumes.update(renderer,dt,sprayWind);
    storm.wind.value.copy(sprayWind).multiplyScalar(3);
    storm.update(Math.max(0,Math.min(0.1,elapsed??dt)),hull.position,camera);
    storm.updateLens(renderer);
    sound.rain=storm.intensity.value; sound.strike=storm.sound.strike;
    light.value.setRGB(transition.values.sunR,transition.values.sunG,transition.values.sunB).multiplyScalar(transition.values.cloudLight).addScalar(storm.flash.value*5);
    sun.intensity=transition.values.power+storm.flash.value*7;
    fill.intensity=transition.values.fill+storm.flash.value*0.7;
    scene.backgroundIntensity=transition.values.background+storm.flash.value*0.16;
    skyBrightness.value=scene.backgroundIntensity;
    time.value=t;
    cloudMesh.position.copy(camera.position); stars.position.copy(camera.position);
  }, setPerformance(level:number) {qualityLevel=level;}, setQuality(high: boolean) { cloudMesh.visible=high; pipeline.outputNode=high?stormComposite:stormBeauty; pipeline.needsUpdate=true; } };
}
