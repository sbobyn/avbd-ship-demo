import { revealShots, revealCamera, createRevealComposite, recordReveal } from './reveal.ts';
import { createPirateFlag } from './flag.ts';
import { createShipAudio } from './audio.ts';
import type { PilotControls } from './navigation.ts';
import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ShipPhysics, type ShipControls } from './physics.ts';
import { ShipGpuSkin } from './gpu-skin.ts';
import { createWeather, type Atmosphere } from './weather.ts';
import './style.css';
import { windViewBasis, projectWind, unprojectWind } from './wind-view.ts';
import { color, positionLocal, smoothstep, texture, vec2, vec3 } from 'three/tsl';
import { waterlineEnvelope } from './ocean/hull.ts';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('scene');
const gpuProfiling=new URL(location.href).searchParams.has('gpuProfile');
const renderer = new THREE.WebGPURenderer({ canvas, antialias: true, trackTimestamp:gpuProfiling });
let gpuTiming='GPU render timing disabled (add ?gpuProfile=1).';
let resolvingTimings=false;
let qualityLevel=1, qualityFrames=0, qualityMs=0;
const renderRatio=()=>Math.min(devicePixelRatio,[0.85,1.25,2][qualityLevel],Math.sqrt([921600,1600000,3686400][qualityLevel]/(innerWidth*innerHeight)));
renderer.setPixelRatio(renderRatio());
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#d8e7e8');
scene.fog = new THREE.Fog('#d8e7e8', 85, 260);
const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.1, 8000);
camera.up.set(0, 0, 1);
camera.position.set(33, -45, 16);
const orbit = new OrbitControls(camera, canvas);
orbit.target.set(0, 0, 9);
orbit.enableDamping = true;
orbit.minDistance = 7; orbit.maxDistance = 90;
orbit.maxPolarAngle = Math.PI * 0.49;
const fill = new THREE.HemisphereLight('#e8f3ff', '#376463', 0.55); scene.add(fill);
const sun = new THREE.DirectionalLight('#fff2df', 3.0);
sun.position.set(-20, -25, 40); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -25, right: 25, top: 25, bottom: -25, near: 1, far: 100 });
sun.shadow.bias = -0.00015;
sun.shadow.normalBias = 0.025;
scene.add(sun);
let weather: ReturnType<typeof createWeather>;
const audio=createShipAudio();
let soundEnabled=false;
$('sound').onclick=async()=>{
  const button=$<HTMLButtonElement>('sound');button.disabled=true;
  try {
    await audio.setEnabled(!soundEnabled);soundEnabled=!soundEnabled;
    button.textContent=soundEnabled?'Sound on':'Enable sound';button.setAttribute('aria-pressed',String(soundEnabled));
  } catch(error) {button.textContent='Retry sound';button.title=String(error);console.warn(error);}
  finally {button.disabled=false;}
};
const audioForward=new THREE.Vector3(), audioUp=new THREE.Vector3();
let glass = false;
let controls: ShipControls = { wind: 7, direction: -35, gust: 0.35, heave: 0.15, roll: 2, period: 7 };
const pilot: PilotControls = {rudder:0,power:1,autoTrim:true,trim:0};
const held=new Set<string>();
let followShip=true;
let cleanView=false, cinematic=false, cinematicPhase=0, framingOffset=0;
function toggleCleanView(): void {
  cleanView=!cleanView; document.body.classList.toggle('clean-view',cleanView);
  $('clean-view').textContent=cleanView?'Show controls · H':'Hide controls · H';
  $('restore-controls').hidden=!cleanView;
}
function toggleCinematic(): void {
  cinematic=!cinematic;
  cinematicPhase=Math.atan2(camera.position.y-orbit.target.y,camera.position.x-orbit.target.x)-(physics?.navigation.heading??0);
  $('cinematic-camera').textContent=cinematic?'Stop cinematic orbit':'Cinematic orbit';
  $('cinematic-camera').setAttribute('aria-pressed',String(cinematic));
  orbit.enabled=!cinematic;
}
$('clean-view').onclick=toggleCleanView;
$('restore-controls').onclick=toggleCleanView;
$('cinematic-camera').onclick=toggleCinematic;
canvas.addEventListener('pointerdown',()=>{if(cinematic) toggleCinematic();});
addEventListener('keydown',event=>{
  if(event.repeat||event.ctrlKey||event.metaKey||event.altKey) return;
  if(event.target instanceof HTMLElement&&event.target.closest('input,select,textarea,[tabindex]')) return;
  if(event.key.toLowerCase()==='h') {event.preventDefault();toggleCleanView();}
  if(event.key.toLowerCase()==='c') {event.preventDefault();toggleCinematic();}
  if(event.key==='Escape') {if(cleanView) toggleCleanView();if(cinematic) toggleCinematic();}
});
$('follow-ship').addEventListener('change',()=>{
  followShip=$<HTMLInputElement>('follow-ship').checked;
  if(followShip && physics) {
    const offset=new THREE.Vector3(physics.positions[0].x-orbit.target.x,physics.positions[0].y-orbit.target.y,0);
    camera.position.add(offset); orbit.target.add(offset);
  }
});
function syncPilot(): void {
  $<HTMLInputElement>('rudder').value=String(pilot.rudder);
  $('rudder-value').textContent=Math.abs(pilot.rudder)<0.01?'Amidships':`${Math.round(Math.abs(pilot.rudder)*100)}% ${pilot.rudder<0?'port':'starboard'}`;
  $<HTMLInputElement>('sail-power').value=String(pilot.power);
  $('sail-power-value').textContent=`${Math.round(pilot.power*100)}%`;
  $<HTMLInputElement>('auto-trim').checked=pilot.autoTrim;
  $<HTMLInputElement>('yard-trim').disabled=pilot.autoTrim;
  $<HTMLInputElement>('yard-trim').value=String(pilot.trim);
  $('yard-trim-value').textContent=pilot.autoTrim?'Auto':`${Math.round(pilot.trim)}°`;
}
$('rudder').addEventListener('input',()=>{ pilot.rudder=Number($<HTMLInputElement>('rudder').value); syncPilot(); });
$('sail-power').addEventListener('input',()=>{ pilot.power=Number($<HTMLInputElement>('sail-power').value); syncPilot(); });
$('yard-trim').addEventListener('input',()=>{ pilot.trim=Number($<HTMLInputElement>('yard-trim').value); syncPilot(); });
$('auto-trim').addEventListener('change',()=>{ pilot.autoTrim=$<HTMLInputElement>('auto-trim').checked; syncPilot(); });
$('center-rudder').onclick=()=>{ pilot.rudder=0; held.clear(); syncPilot(); };
addEventListener('keydown',event=>{
  if(event.target instanceof HTMLElement && event.target.closest('input,button,summary,[tabindex],textarea,select')) return;
  if(event.ctrlKey || event.metaKey || event.altKey || !'adwsqe'.includes(event.key.toLowerCase()) || event.key.length!==1) return;
  event.preventDefault(); held.add(event.key.toLowerCase());
});
addEventListener('keyup',event=>{ const key=event.key.toLowerCase(); held.delete(key); if(key==='a'||key==='d') {pilot.rudder=0;syncPilot();} });
const releaseHelm=()=>{ held.clear(); pilot.rudder=0; syncPilot(); };
addEventListener('blur',releaseHelm);
document.addEventListener('visibilitychange',()=>{if(document.hidden) releaseHelm();});
function advanceHelm(dt:number): void {
  if(!held.size) return;
  if(held.has('a') || held.has('d')) pilot.rudder=Number(held.has('d'))-Number(held.has('a'));
  pilot.power=THREE.MathUtils.clamp(pilot.power+(Number(held.has('w'))-Number(held.has('s')))*dt*0.25,0,1);
  const trim=Number(held.has('e'))-Number(held.has('q'));
  if(trim) {pilot.autoTrim=false;pilot.trim=THREE.MathUtils.clamp(pilot.trim+trim*dt*15,-45,45);}
  syncPilot();
}
const suffix: Record<keyof ShipControls, string> = { wind: ' m/s', direction: '°', gust: '%', heave: ' m', roll: '°', period: ' s' };
const sliderKeys = ['gust', 'heave', 'roll', 'period'] as const;
let physics: ShipPhysics, device: GPUDevice;
const dial = $('wind-dial');
let windBasis:readonly number[]=[1,0,0,-1];
function updateWindDial(): void {
  camera.updateMatrixWorld();
  windBasis=windViewBasis(camera.matrixWorldInverse);
  const [a,b,c,d]=windBasis;
  dial.querySelector('svg')!.setAttribute('viewBox','0 0 200 200');
  document.getElementById('wind-plane')!.setAttribute('transform',`matrix(${a} ${b} ${c} ${d} ${100-100*a-100*c} ${100-100*b-100*d})`);
  const angle=controls.direction*Math.PI/180;
  const [dx,dy]=projectWind(windBasis,Math.cos(angle)*controls.wind/18*80,Math.sin(angle)*controls.wind/18*80);
  for(const [id,ax,ay] of [['wind-line','x2','y2'],['wind-handle','cx','cy']]) {
    document.getElementById(id)!.setAttribute(ax,String(100+dx));
    document.getElementById(id)!.setAttribute(ay,String(100+dy));
  }
  document.getElementById('dial-heading')!.setAttribute('transform',`rotate(${(physics?.navigation.heading??0)*180/Math.PI} 100 100)`);
}
const sync = (): void => {
  for (const key of sliderKeys) {
    $<HTMLInputElement>(key).value = String(controls[key]);
    $<HTMLOutputElement>(`${key}-value`).value = `${key === 'gust' ? Math.round(controls[key] * 100) : controls[key]}${suffix[key]}`;
  }
  updateWindDial();
  $('wind-readout').textContent = `${controls.wind.toFixed(1)} m/s · ${Math.round(controls.direction)}°`;
  dial.setAttribute('aria-label', `Wind vector: ${controls.wind.toFixed(1)} metres per second, ${Math.round(controls.direction)} degrees; camera-aligned compass. Arrow keys adjust; Home calms`);
};
for (const key of sliderKeys) $<HTMLInputElement>(key).addEventListener('input', e => {
  glass = false;
  controls[key] = Number((e.target as HTMLInputElement).value); sync();
  document.querySelectorAll('[data-preset]').forEach(b => b.classList.remove('selected'));
});
const setWind = (x: number, y: number): void => {
  glass = false;
  controls.wind = Math.min(18, Math.hypot(x, y));
  if (controls.wind > 0.05) controls.direction = Math.atan2(y, x) * 180 / Math.PI;
  else controls.wind = 0;
  sync();
  document.querySelectorAll('[data-preset]').forEach(b => b.classList.remove('selected'));
};
const dragWind = (event: PointerEvent): void => {
  updateWindDial();
  const svg=dial.querySelector('svg')!, transform=svg.getScreenCTM();
  if(!transform) return;
  const point=new DOMPoint(event.clientX,event.clientY).matrixTransform(transform.inverse());
  setWind(...unprojectWind(windBasis,(point.x-100)/80*18,(point.y-100)/80*18));
};
dial.addEventListener('pointerdown', event => {
  if (!event.isPrimary || event.button !== 0) return;
  dial.focus(); dial.setPointerCapture(event.pointerId); dragWind(event);
});
dial.addEventListener('pointermove', event => { if (dial.hasPointerCapture(event.pointerId)) dragWind(event); });
dial.addEventListener('pointerup', event => {
  if (dial.hasPointerCapture(event.pointerId)) { dragWind(event); dial.releasePointerCapture(event.pointerId); }
});
dial.addEventListener('keydown', event => {
  const step = event.shiftKey ? 2 : 0.5;
  const delta: Record<string, [number, number]> = { ArrowRight: [step, 0], ArrowLeft: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
  if (event.key === 'Home') { event.preventDefault(); setWind(0, 0); }
  else if (delta[event.key]) {
    event.preventDefault();
    const angle = controls.direction * Math.PI / 180;
    updateWindDial();
    const projected=projectWind(windBasis,Math.cos(angle)*controls.wind,Math.sin(angle)*controls.wind);
    setWind(...unprojectWind(windBasis,projected[0]+delta[event.key][0],projected[1]-delta[event.key][1]));
  }
});
const presets: Record<string, ShipControls> = {
  glass: { wind: 2, direction: -35, gust: 0, heave: 0, roll: 0, period: 10 },
  calm: { wind: 3, direction: -35, gust: 0, heave: 0, roll: 0, period: 7 },
  breeze: { wind: 7, direction: -35, gust: 0.35, heave: 0.15, roll: 2, period: 7 },
  swell: { wind: 12, direction: -45, gust: 0.65, heave: 0.5, roll: 6, period: 5 },
};
document.querySelectorAll<HTMLButtonElement>('[data-preset]').forEach(button => button.addEventListener('click', () => {
  glass = button.dataset.preset === 'glass';
  controls = { ...presets[button.dataset.preset!] }; sync();
  document.querySelectorAll('[data-preset]').forEach(b => b.classList.toggle('selected', b === button));
}));
sync();
let paused = false, busy = false, resetting = false, failed = false;
let gpuSkin: ShipGpuSkin;
let sourceMaterials: THREE.MeshStandardMaterial[];
let ready = false, lastInspection = 0;
let loadingStatus = 'Waiting for the browser’s WebGPU device…';
let sails: THREE.Mesh, rigging: THREE.Mesh;
let originalSails: THREE.BufferGeometry, originalRigging: THREE.BufferGeometry;
const hullGroup = new THREE.Group(); scene.add(hullGroup);
let pirateFlag:ReturnType<typeof createPirateFlag>|undefined;
// Recycle only distant grid cells; visible markers never follow the hull.
const markerBoxes=new THREE.InstancedMesh(new THREE.BoxGeometry(4,4,6),new THREE.MeshStandardMaterial({color:'#e9a34c',roughness:0.72}),1089);
markerBoxes.frustumCulled=false;
scene.add(markerBoxes);
let markerCell='';
const markerMatrix=new THREE.Matrix4();
function updateMarkers(): void {
  markerBoxes.visible=$<HTMLInputElement>('landmarks').checked;
  if(!markerBoxes.visible) return;
  const spacing=Number($<HTMLSelectElement>('marker-spacing').value);
  const cx=Math.floor(hullGroup.position.x/spacing), cy=Math.floor(hullGroup.position.y/spacing);
  const cell=`${spacing}:${cx}:${cy}`;
  if(cell===markerCell) return;
  markerCell=cell;
  const radius=Math.ceil(400/spacing), width=radius*2+1;
  markerBoxes.count=width*width;
  for(let y=-radius;y<=radius;y++) for(let x=-radius;x<=radius;x++) {
    markerMatrix.makeTranslation((cx+x)*spacing,(cy+y+0.5)*spacing,2.55);
    markerBoxes.setMatrixAt((y+radius)*width+x+radius,markerMatrix);
  }
  markerBoxes.instanceMatrix.needsUpdate=true;
}
$('landmarks').addEventListener('change',()=>{
  $<HTMLSelectElement>('marker-spacing').disabled=!$<HTMLInputElement>('landmarks').checked;
  updateMarkers();
});
$('marker-spacing').addEventListener('change',updateMarkers);
updateMarkers();
let last = performance.now(), accumulator = 0, fps = 0, fpsTime = last, previousSimTime = 0;
function fail(error: unknown): void {
  failed = true; $('error').hidden = false; $('error').textContent = error && typeof error === 'object' && 'message' in error ? String(error.message) : String(error);
  $('live').textContent = 'STOPPED'; $('status').textContent = 'Simulation stopped'; console.error(error);
}
$('pause').onclick = () => { paused = !paused; $('pause').textContent = paused ? 'Resume' : 'Pause'; $('live').textContent = paused ? 'PAUSED' : 'LIVE'; accumulator = 0; };
$('reset').onclick = () => { resetting = true; };
const reviewViews = [
  { name:'ship', label:'Ship', position:[33,-45,16], target:[0,0,9] },
  { name:'sails', label:'Sails', position:[17,-22,14], target:[3,0,9] },
  { name:'stern', label:'Rear yard', position:[-15,-16,12], target:[-6,0,8] },
  { name:'bow', label:'Bow rigging', position:[20,-13,9], target:[10,0,5] },
  { name:'impact', label:'Bow splash', position:[20,-18,5], target:[9,0,0.5] },
  { name:'wake', label:'Stern wake', position:[-32,-24,12], target:[-6,0,1] },
] satisfies { name:string; label:string; position:[number,number,number]; target:[number,number,number] }[];
let reviewView=Math.max(0,reviewViews.findIndex(view=>view.name===new URL(location.href).searchParams.get('view')));
function applyReviewView(): void {
  const view=reviewViews[reviewView];
  // Consume residual orbit damping before assigning an exact saved view.
  const damping=orbit.enableDamping; orbit.enableDamping=false; orbit.update();
  camera.position.set(...view.position); orbit.target.set(...view.target);
  if(physics) {
    const heading=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),physics.navigation.heading);
    const offset=new THREE.Vector3(physics.positions[0].x,physics.positions[0].y,0);
    camera.position.applyQuaternion(heading).add(offset); orbit.target.applyQuaternion(heading).add(offset);
  }
  orbit.update(); orbit.enableDamping=damping;
  $('camera').textContent=reviewViews[(reviewView+1)%reviewViews.length].label+' ↗';
}
applyReviewView();
$('camera').onclick=()=>{
  if(cinematic) toggleCinematic();
  reviewView=(reviewView+1)%reviewViews.length; applyReviewView();
  const url=new URL(location.href); url.searchParams.set('view',reviewViews[reviewView].name);
  history.replaceState(null,'',url);
};
const frameTimes:number[]=[];
let cadenceSamples:number[] | null=null;
let cadenceWarmup=0;
$('cadence-check').onclick=()=>{
  cadenceSamples=[]; cadenceWarmup=5;
  $<HTMLButtonElement>('cadence-check').disabled=true;
  $('cadence-result').textContent='Measuring 120 idle frames; drawing and simulation are temporarily frozen…';
};
addEventListener('pagehide', event => { if (!event.persisted) renderer.dispose(); });
addEventListener('resize', () => { renderer.setPixelRatio(renderRatio()); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
function reset(heading = 0): void {
  if (physics) { const offset=new THREE.Vector3(physics.positions[0].x,physics.positions[0].y,0); camera.position.sub(offset); orbit.target.sub(offset); }
  releaseHelm();
  physics?.destroy();
  gpuSkin?.dispose();
  sails.geometry.dispose(); rigging.geometry.dispose();
  sails.geometry = originalSails.clone(); rigging.geometry = originalRigging.clone();
  if(pirateFlag) {pirateFlag.mesh.geometry.dispose();pirateFlag.mesh.geometry=pirateFlag.restGeometry.clone();}
  physics = new ShipPhysics(device, sails.geometry, rigging.geometry, count => {
    gpuSkin = new ShipGpuSkin(renderer, count);
    return gpuSkin.buffer;
  }, heading, pirateFlag?.mesh.geometry);
  gpuSkin.bind(physics, [sails, rigging, ...(pirateFlag?[pirateFlag.mesh]:[])], sourceMaterials);
  hullGroup.position.set(0, 0, 0); hullGroup.quaternion.identity();
  previousSimTime = 0; fpsTime = performance.now(); fps = 0;
  accumulator = 0; failed = false; $('error').hidden = true; resetting = false;
  $('live').textContent = paused ? 'PAUSED' : 'LIVE';
  applyReviewView();
}
let revealComposite:ReturnType<typeof createRevealComposite>|undefined;
let revealStart=0, revealRecording:MediaRecorder|undefined, revealShot=-1, revealLightning=false;
$('reveal').onclick=async()=>{
  if(!ready||!weather||revealRecording) return;
  const button=$<HTMLButtonElement>('reveal');button.disabled=true;
  try {
    await audio.setEnabled(false);soundEnabled=false;$('sound').textContent='Enable sound';$('sound').setAttribute('aria-pressed','false');
    paused=false; cinematic=false; orbit.enabled=false;
    reset(Math.PI*0.75);
    if(cleanView) toggleCleanView();
    document.querySelector<HTMLDetailsElement>('.debug-panel')!.open=false;
    revealComposite=createRevealComposite(canvas);
    // Render above delivery resolution so fine rigging survives the 1080p downsample.
    renderer.setPixelRatio(1.5);renderer.setSize(1920,1080,false);
    camera.aspect=1920/1080;camera.setViewOffset(1920,1080,140,0,1920,1080);camera.updateProjectionMatrix();
    controls={...presets.calm,direction:0};glass=false;weather.apply('dawn');
    revealShot=-1;revealLightning=false;
    // Settle the opening light before capture, with the shot already framed.
    revealStart=performance.now()+4000;
    $('reveal-status').textContent='Preparing dawn…';
    setTimeout(()=>{
      revealRecording=recordReveal(revealComposite!.canvas,undefined,url=>{
        const link=$<HTMLAnchorElement>('reveal-download');link.href=url;link.hidden=false;
        $('reveal-status').textContent='45-second film ready';button.disabled=false;

      });
    },4000);
  } catch(error) {button.disabled=false;$('reveal-status').textContent=String(error);}
};
function advanceReveal(now:number):void {
  if(!revealStart||!physics||!weather) return;
  const t=Math.max(0,(now-revealStart)/1000);
  const index=revealShots.reduce((current,shot,i)=>t>=shot.at?i:current,0), shot=revealShots[index];
  if(index!==revealShot) {
    revealShot=index;
    document.querySelector<HTMLButtonElement>(`[data-weather="${shot.weather}"]`)!.click();
    document.querySelector<HTMLButtonElement>(`[data-preset="${shot.sea}"]`)!.click();
    controls.direction+=physics.navigation.heading*180/Math.PI;sync();
  }
  if(t<4.5) {
    const phase=Math.max(0,Math.min(1,(t-0.3)/3.8));
    const angle=physics.navigation.heading+(-35+75*Math.sin(phase*Math.PI*2))*Math.PI/180;
    const speed=3+5*Math.sin(phase*Math.PI);
    setWind(Math.cos(angle)*speed,Math.sin(angle)*speed);
  }
  pilot.rudder=t>18&&t<32?0.15:0;pilot.autoTrim=true;
  const frame=revealCamera(t);
  const cameraAngle=physics.navigation.heading+Math.PI+frame.angle;
  camera.position.set(hullGroup.position.x+Math.cos(cameraAngle)*frame.radius,hullGroup.position.y+Math.sin(cameraAngle)*frame.radius,frame.z);
  orbit.target.set(hullGroup.position.x+Math.cos(physics.navigation.heading)*frame.lead,hullGroup.position.y+Math.sin(physics.navigation.heading)*frame.lead,frame.target);camera.lookAt(orbit.target);
  if(t>24.5&&!revealLightning){weather.triggerLightning();revealLightning=true;}
  if(t>=45&&revealRecording) {
    revealRecording.stop();revealRecording=undefined;revealComposite=undefined;revealStart=0;pilot.rudder=0;orbit.enabled=true;
    renderer.setPixelRatio(renderRatio());renderer.setSize(innerWidth,innerHeight);
    camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();
    if(cleanView) toggleCleanView();
  }
}
function applyQuality():void {
  renderer.setPixelRatio(renderRatio());renderer.setSize(innerWidth,innerHeight);
  const shadowSize=qualityLevel===0?1024:2048;
  if(sun.shadow.mapSize.x!==shadowSize) {
    sun.shadow.map?.dispose(); sun.shadow.map=null;
    sun.shadow.mapSize.set(shadowSize,shadowSize);
  }
  weather?.setPerformance(qualityLevel);
  const clouds=qualityLevel>0;
  $<HTMLInputElement>('cinematic').checked=clouds;weather?.setQuality(clouds);
  $('quality-status').textContent=`${['Low','Balanced','High'][qualityLevel]} · ${Math.round(innerWidth*renderRatio())} × ${Math.round(innerHeight*renderRatio())}`;
  qualityFrames=0;qualityMs=0;
}
$('quality').addEventListener('change',()=>{
  qualityLevel=Math.max(0,['low','balanced','high'].indexOf($<HTMLSelectElement>('quality').value));
  if($<HTMLSelectElement>('quality').value==='auto') qualityLevel=1;
  applyQuality();
});
function animate(now: number): void {
  const frameMs=now-last;
  const elapsed = Math.min(frameMs / 1000, 0.05); last = now;
  if (document.hidden) { accumulator = 0; frameTimes.length=0; return; }
  if(ready && !paused && !failed) { frameTimes.push(frameMs); if(frameTimes.length>240) frameTimes.shift(); }
  if(ready&&!paused&&!revealStart&&$<HTMLSelectElement>('quality').value==='auto'&&frameMs<1000) {
    qualityMs+=frameMs;qualityFrames++;
    if(qualityFrames>=180) {
      if(qualityMs/qualityFrames>35&&qualityLevel>0) {qualityLevel--;applyQuality();}
      qualityFrames=0;qualityMs=0;
    }
  }
  if(cadenceSamples) {
    accumulator=0;
    if(cadenceWarmup>0) { cadenceWarmup--; return; }
    cadenceSamples.push(frameMs);
    if(cadenceSamples.length>=120) {
      const sorted=cadenceSamples.slice().sort((a,b)=>a-b);
      const mean=cadenceSamples.reduce((a,b)=>a+b,0)/cadenceSamples.length;
      $('cadence-result').textContent=`Idle: ${(1000/mean).toFixed(1)} fps · mean ${mean.toFixed(1)} ms · p95 ${sorted[113].toFixed(1)} ms. Drawing and simulation resumed.`;
      cadenceSamples=null; frameTimes.length=0;
      $<HTMLButtonElement>('cadence-check').disabled=false;
      fps=0; fpsTime=now; previousSimTime=physics?.time ?? 0;
    }
    return;
  }
  orbit.update();
  if (physics && !busy && resetting) reset();
  if (physics && !paused && !failed) accumulator = Math.min(accumulator + elapsed, 1 / 15);
  if (physics && !paused && !failed) {
    const steps = Math.min(4, Math.floor(accumulator * 120));
    if (steps) {
      accumulator -= steps / 120;
      try {
        advanceHelm(steps/120); Object.assign(physics.pilot,pilot);
        const oldPosition = physics.positions[0].clone();
        if(weather) physics.surfaceTarget.copy(weather.waterPose);
        physics.step(controls, steps);
        const travel=physics.positions[0].clone().sub(oldPosition); travel.z=0;
        if(followShip) {camera.position.add(travel); orbit.target.add(travel);}
        hullGroup.position.copy(physics.positions[0]); hullGroup.quaternion.copy(physics.rotations[0]);
        $('pilot-readout').textContent=`${(physics.navigation.speed*1.94384).toFixed(1)} kn · heading ${Math.round((physics.navigation.heading*180/Math.PI%360+360)%360)%360}° · travelled ${physics.navigation.distance.toFixed(0)} m`;
      } catch (error) { fail(error); }
    }
    if (!busy && now - lastInspection > 1000) {
      busy = true; lastInspection = now;
      void physics.inspect().catch(fail).finally(() => { busy = false; });
    }
  }
  if(cinematic&&physics) {
    cinematicPhase+=elapsed*0.065;
    const heading=physics.navigation.heading, angle=cinematicPhase+heading;
    const radius=58+Math.sin(cinematicPhase*1.7)*4;
    const desired=new THREE.Vector3(physics.positions[0].x+Math.cos(angle)*radius,physics.positions[0].y+Math.sin(angle)*radius,17+Math.sin(cinematicPhase*1.3)*3);
    camera.position.lerp(desired,1-Math.exp(-elapsed*1.4));
    orbit.target.lerp(new THREE.Vector3(physics.positions[0].x,physics.positions[0].y,10),1-Math.exp(-elapsed*1.4));
    camera.lookAt(orbit.target);
  }
  advanceReveal(now);
  const frameTarget=cleanView?0:Math.min(145,innerWidth*0.16);
  framingOffset+=(frameTarget-framingOffset)*(1-Math.exp(-elapsed*5));
  if(!revealStart) camera.setViewOffset(innerWidth,innerHeight,framingOffset,0,innerWidth,innerHeight);
  updateWindDial();
  updateMarkers();
  weather?.update(physics?.time ?? 0, controls.wind, controls.heave, elapsed, glass, controls.direction, physics?.navigation.heading ?? 0);
  if(weather) {
    camera.getWorldDirection(audioForward);audioUp.set(0,1,0).applyQuaternion(camera.quaternion);
    audio.update(elapsed,weather.sound,controls.wind,controls.direction,camera.position,audioForward,audioUp,hullGroup.position,!paused&&!failed&&!resetting,glass);
  }
  if (now - fpsTime > 1000) {
    $('audio-status').textContent=audio.status();
    $('status').textContent = physics ? `${physics.sailCount} sails · ${physics.ropeCount} rope sections · ${physics.ladderCount} rope ladders · ${physics.reference.bodies.length.toLocaleString()} bodies · ${Math.round(fps * 1000 / (now - fpsTime))} fps · ${((physics.time - previousSimTime) * 1000 / (now - fpsTime)).toFixed(2)}× sim` : loadingStatus;
    const sorted=frameTimes.slice().sort((a,b)=>a-b);
    const mean=frameTimes.length ? frameTimes.reduce((a,b)=>a+b,0)/frameTimes.length : 0;
    const p95=sorted[Math.max(0,Math.ceil(sorted.length*0.95)-1)] ?? 0;
    if (physics) $('metrics').textContent = `Simulated ${physics.time.toFixed(1)} s · max sail displacement ${physics.maxDisplacement.toFixed(2)} m relative to hull. Frame mean ${mean.toFixed(1)} ms · p95 ${p95.toFixed(1)} ms (${frameTimes.length} frames). Submit ${physics.timings.submit.toFixed(1)} ms · GPU mesh deformation (no per-frame readback). ${physics.profile}. ${gpuTiming}`;
    previousSimTime = physics?.time ?? 0;
    fps = 0; fpsTime = now;
  }
  fps++;
  if (ready && weather) {
    weather.render();
    if(revealComposite) revealComposite.draw(Math.max(0,(now-revealStart)/1000));
    if(gpuProfiling && renderer.hasFeature('timestamp-query') && !resolvingTimings) {
      resolvingTimings=true;
      void Promise.all([renderer.resolveTimestampsAsync('render'),renderer.resolveTimestampsAsync('compute')])
        .then(([render,compute])=>{
          gpuTiming=`GPU draw passes ${render?.toFixed(2) ?? 'unavailable'} ms · Three.js compute ${compute?.toFixed(2) ?? 'unavailable'} ms (excludes AVBD solver).`;
        }).catch(()=>{ gpuTiming='GPU timestamp readback unavailable.'; })
        .finally(()=>{ resolvingTimings=false; });
    }
  }
}
try {
  if (!navigator.gpu) throw new Error('This demo needs a browser with WebGPU enabled. Try a current Chrome, Edge, or Safari.');
  await renderer.init();
  if(gpuProfiling && !renderer.hasFeature('timestamp-query')) gpuTiming='GPU timestamp queries unsupported; frame timings remain available.';
  renderer.setAnimationLoop(animate);
  if (!(renderer.backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend) throw new Error('This demo requires a WebGPU rendering device.');
  device = (renderer.backend as unknown as { device: GPUDevice }).device;
  loadingStatus = 'Preparing the ocean and atmosphere…';
  weather = createWeather(renderer, scene, camera, sun, fill, hullGroup);
  document.querySelectorAll<HTMLButtonElement>('[data-weather]').forEach(button => button.addEventListener('click', () => {
    weather.apply(button.dataset.weather as Atmosphere);
    document.querySelectorAll('[data-weather]').forEach(b => b.classList.toggle('selected', b === button));
  }));
  $('wake-inspection').addEventListener('change',()=>weather.inspectWake($<HTMLSelectElement>('wake-inspection').value));
  $('lightning').onclick=()=>weather.triggerLightning();
  $<HTMLInputElement>('cinematic').addEventListener('change', event => weather.setQuality((event.target as HTMLInputElement).checked));
  applyQuality();
  ready = true;
  device.addEventListener('uncapturederror', e => fail(e.error));
  void device.lost.then(info => fail(`GPU device lost: ${info.message}. Reload to restart.`));
  loadingStatus = 'Loading the ship model…';
  const gltf = await new GLTFLoader().loadAsync('/ship/ship.gltf');
  gltf.scene.updateMatrixWorld(true);
  const meshes: THREE.Mesh[] = [];
  gltf.scene.traverse(object => { if (object instanceof THREE.Mesh) meshes.push(object); });
  for (const mesh of meshes) {
    mesh.geometry = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld).rotateX(Math.PI / 2);
    mesh.position.set(0, 0, 0); mesh.rotation.set(0, 0, 0); mesh.scale.set(1, 1, 1);
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials as THREE.MeshStandardMaterial[]) {
      // Poly Haven ARM packs occlusion in red; glTF only declares roughness/metalness.
      material.aoMap = material.roughnessMap;
      material.aoMapIntensity = mesh.name.includes('sails') ? 0.65 : 1;
      for (const texture of [material.map, material.normalMap, material.roughnessMap, material.metalnessMap]) {
        if (texture) texture.anisotropy = 8;
      }
    }
    mesh.castShadow = true; mesh.receiveShadow = true; mesh.frustumCulled = false;
    if (mesh.name.includes('sails')) { sails = mesh; scene.add(mesh); }
    else if (mesh.name.includes('rigging')) { rigging = mesh; scene.add(mesh); }
    else {
      // Illuminate the existing lantern glass, retaining its textured cage and cap.
      const glass=smoothstep(5.40,5.44,positionLocal.z).mul(smoothstep(5.80,5.76,positionLocal.z))
        .mul(smoothstep(0.26,0.22,positionLocal.xy.sub(vec2(-10.08,0.003)).length()));
      mesh.material=(materials as THREE.MeshStandardMaterial[]).map(material => {
        const lit=new THREE.MeshStandardNodeMaterial().copy(material);
        lit.emissiveNode=color('#ffad59').mul(weather.lanternGlow).mul(4).mul(glass)
          .mul(material.map?texture(material.map).rgb:vec3(1));
        return lit;
      });
      if(materials.length===1) mesh.material=mesh.material[0];
      hullGroup.add(mesh);
    }
  }
  const hullGeometries=meshes.filter(mesh => mesh !== sails && mesh !== rigging).map(mesh => mesh.geometry);
  weather.setHull(waterlineEnvelope(hullGeometries),hullGeometries);
  pirateFlag=createPirateFlag(meshes.map(mesh=>mesh.geometry));
  if (!sails! || !rigging!) throw new Error('The model is missing its named sail or rigging mesh.');
  scene.add(pirateFlag.mesh);
  sourceMaterials = [sails.material, rigging.material, pirateFlag.material] as THREE.MeshStandardMaterial[];
  originalSails = sails.geometry.clone(); originalRigging = rigging.geometry.clone();
  reset();
} catch (error) { fail(error); }
