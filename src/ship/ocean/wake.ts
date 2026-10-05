import * as THREE from 'three/webgpu';
import { atomicAdd, atomicLoad, atomicStore, If, uint, exp, float, floor, Fn, instanceIndex, instancedArray, int, mix, smoothstep, storage, uniform, sin, cos, vec2, vec4 } from 'three/tsl';

// Local linear wave field. Integer-cell recentering preserves the world-space trail.
const WIDTH=512, HEIGHT=512, CELL=0.5, STEP=1/60, C2=25;
export function createWake(contacts?: {sampleWake:(point:THREE.Node<'vec2'>,previous?:boolean)=>THREE.Node<'vec4'>}) {
  const landings=instancedArray(WIDTH*HEIGHT,'uint').toAtomic();
  const state=instancedArray(WIDTH*HEIGHT,'vec4'), next=instancedArray(WIDTH*HEIGHT,'vec4');
  // World flow, density-weighted age, and agitation travel with the foam.
  const history=instancedArray(WIDTH*HEIGHT,'vec4'), nextHistory=instancedArray(WIDTH*HEIGHT,'vec4');
  const contactStepScale=uniform(1);
  const origin=uniform(new THREE.Vector2()), shift=uniform(new THREE.Vector2());
  const pose=uniform(new THREE.Vector4()), previousPose=uniform(new THREE.Vector4()), clock=uniform(0), speed=uniform(0);
  const hull=uniform(new THREE.Vector4(-6.5,7.5,0,2));
  const fetch=(x: THREE.Node<'int'>,y: THREE.Node<'int'>,buffer=state) => {
    const valid=x.greaterThanEqual(0).and(x.lessThan(WIDTH)).and(y.greaterThanEqual(0)).and(y.lessThan(HEIGHT));
    return valid.select(buffer.element(int(float(y).clamp(0,HEIGHT-1)).mul(WIDTH).add(int(float(x).clamp(0,WIDTH-1)))),vec4(0));
  };
  const advected=(p: THREE.Node<'vec2'>,buffer=state) => {
    const base=floor(p), f=p.sub(base), x=int(base.x), y=int(base.y);
    return mix(mix(fetch(x,y,buffer),fetch(x.add(1),y,buffer),f.x),mix(fetch(x,y.add(1),buffer),fetch(x.add(1),y.add(1),buffer),f.x),f.y);
  };
  const solve=Fn(() => {
    const x=int(instanceIndex).mod(WIDTH), y=int(instanceIndex).div(WIDTH);
    const sx=x.add(int(shift.x)), sy=y.add(int(shift.y));
    const q=fetch(sx,sy).toVar();
    const l=fetch(sx.sub(1),sy).x, r=fetch(sx.add(1),sy).x;
    const b=fetch(sx,sy.sub(1)).x, t=fetch(sx,sy.add(1)).x;
    const px=float(x).mul(CELL).add(origin.x), py=float(y).mul(CELL).add(origin.y);
    const local=(boat: THREE.Node<'vec4'>) => {
      const dx=px.sub(boat.x), dy=py.sub(boat.y);
      return vec2(dx.mul(boat.z).add(dy.mul(boat.w)),dy.mul(boat.z).sub(dx.mul(boat.w)));
    };
    const pressure=(boat: THREE.Node<'vec4'>) => {
      const p=local(boat);
      const bow=p.x.sub(hull.y), stern=p.x.sub(hull.x), across=p.y.sub(hull.z);
      return exp(bow.mul(bow).div(-2.5).add(across.mul(across).div(hull.w.mul(hull.w).negate()))).mul(1.9)
        .sub(exp(stern.mul(stern).div(-4).add(across.mul(across).div(hull.w.mul(hull.w).mul(-0.7)))).mul(0.95));
    };
    const edge=float(x).min(float(y)).min(float(WIDTH-1).sub(float(x))).min(float(HEIGHT-1).sub(float(y)));
    const steepness=vec2(r.sub(l),t.sub(b)).length().div(2*CELL);
    // A linear field otherwise stores the resonant bow energy indefinitely near
    // its 5 m/s propagation speed. Breaking crests dissipate both height and velocity.
    const breaking=smoothstep(0.25,0.8,steepness).mul(3).add(smoothstep(0.45,1.1,q.x.abs()).mul(6));
    const damping=float(1).sub(smoothstep(0,14,edge)).mul(4).add(0.12).add(breaking);
    const velocity=q.y.add(l.add(r).add(b).add(t).sub(q.x.mul(4)).mul(C2*STEP/(CELL*CELL))).mul(exp(damping.mul(-STEP)));
    const meshSource=contacts?.sampleWake(vec2(px,py));
    const disturbance=contacts?contacts.sampleWake(vec2(px,py),true).x.sub(meshSource!.x).mul(contactStepScale):pressure(previousPose).sub(pressure(pose));
    const height=q.x.add(velocity.mul(STEP)).add(disturbance).mul(exp(damping.mul(-STEP)));
    const localPoint=local(pose), localX=localPoint.x, across=localPoint.y.sub(hull.z);
    const along=localX.sub(hull.x).div(hull.y.sub(hull.x)).clamp(0,1);
    const width=hull.w.mul(float(1).sub(along.mul(2).sub(1).pow2()).max(0).sqrt());
    const ends=smoothstep(hull.x.sub(1.5),hull.x.add(1),localX).mul(float(1).sub(smoothstep(hull.y.sub(0.5),hull.y.add(1),localX)));
    const shoulder=exp(across.abs().sub(width).pow2().mul(-6)).mul(ends);
    // Two narrow stern-quarter streams rather than one broad propeller-like patch.
    const stern=exp(localX.sub(hull.x).pow2().div(-1.2)
      .add(across.abs().sub(hull.w.mul(0.55)).pow2().div(-0.22)));
    const bow=shoulder.mul(smoothstep(0.65,0.95,along));
    // Stern-quarter jets are stamped with their emission heading. Old trails
    // retain that world flow through turns instead of following today's rudder.
    const pulseA=sin(clock.mul(1.71).add(across.mul(1.3))), pulseB=sin(clock.mul(0.63).sub(across.mul(2.17)));
    const pulses=pulseA.mul(pulseB).mul(0.32).add(0.68);
    const sourceFlow=vec2(stern.mul(speed).mul(-0.24),across.div(across.abs().add(0.3))
      .mul(stern.add(shoulder.mul(0.25))).mul(speed).mul(0.12))
      .add(vec2(pulseB,pulseA).mul(stern).mul(speed.min(3)).mul(0.16));
    // Released submerged footprint identifies the actual stern water exit,
    // including turns, without painting an analytic stern ellipse onto the sea.
    const release=contacts?disturbance.max(0).div(STEP):float(0);
    const lateralFlow=vec2(pose.w.negate(),pose.z).mul(across.div(across.abs().add(0.3))).mul(speed.mul(0.25));
    const releaseFlow=vec2(pose.z,pose.w).mul(speed.mul(-0.2)).add(lateralFlow);
    const releaseWeight=release.div(release.add(meshSource?.y??float(0)).add(0.0001));
    const sourceVelocity=contacts?mix(meshSource!.zw.div(meshSource!.y.max(0.0001)).mul(0.5).add(lateralFlow),releaseFlow,releaseWeight):vec2(sourceFlow.x.mul(pose.z).sub(sourceFlow.y.mul(pose.w)),sourceFlow.y.mul(pose.z).add(sourceFlow.x.mul(pose.w)));
    const worldFlow=sourceVelocity.mul(float(2.8).div(sourceVelocity.length().max(2.8)));
    const oldHistory=fetch(sx,sy,history);
    // Extrapolate recorded world flow into empty edge cells. Without it, a
    // semi-Lagrangian backtrace cannot move a ribbon into its dry neighbour.
    const neighbours=[[sx.sub(1),sy],[sx.add(1),sy],[sx,sy.sub(1)],[sx,sy.add(1)]];
    const mass=q.z.mul(4).toVar(), momentum=oldHistory.xy.mul(q.z).mul(4).toVar();
    const neighbourState=vec4(0).toVar(), neighbourHistory=vec4(0).toVar();
    for(const [nx,ny] of neighbours) {
      const n=fetch(nx,ny), h=fetch(nx,ny,history);
      mass.addAssign(n.z); momentum.addAssign(h.xy.mul(n.z));
      neighbourState.addAssign(n.mul(0.25)); neighbourHistory.addAssign(h.mul(0.25));
    }
    const edgeFlow=momentum.div(mass.max(0.0001));
    // Small divergence-free eddies in world space keep aerated ribbons alive
    // without tying their motion to the current hull or rotating old tracks.
    const phaseX=px.mul(0.73).add(clock.mul(0.17)), phaseY=py.mul(0.61).sub(clock.mul(0.11));
    const curl=vec2(sin(phaseX).mul(cos(phaseY)).mul(0.61),cos(phaseX).mul(sin(phaseY)).mul(-0.73)).mul(oldHistory.w).mul(0.2);
    const backtrace=vec2(float(sx),float(sy)).sub(edgeFlow.add(curl).mul(STEP/CELL));
    const carried=advected(backtrace).toVar(), carriedHistory=advected(backtrace,history).toVar();
    const foamAge=oldHistory.z.div(q.z.max(0.0001));
    const diffusion=smoothstep(1,15,foamAge).mul(0.035).add(0.025);
    // Diffuse foam, bubbles and their moments together so spreading does not
    // rejuvenate old foam or leave an unmoving underwater stripe behind it.
    carried.zw.assign(mix(carried.zw,neighbourState.zw,diffusion));
    carriedHistory.zw.assign(mix(carriedHistory.zw,neighbourHistory.zw,diffusion));
    // Saturating production is tied to travelled water, not a permanent painted
    // hull mask. A slow displacement hull leaves thin ribbons, never prop wash.
    const production=(contacts?meshSource!.y.mul(2.8).add(release.mul(0.8)):bow.mul(2.8).add(shoulder.mul(0.45)).add(stern.mul(2.8).mul(pulses)).mul(speed.min(3)))
      .add(smoothstep(0.18,0.5,steepness).mul(0.18));
    const landingInside=sx.greaterThanEqual(0).and(sy.greaterThanEqual(0)).and(sx.lessThan(WIDTH)).and(sy.lessThan(HEIGHT));
    const landing=landingInside.select(float(atomicLoad(landings.element(int(float(sy).clamp(0,HEIGHT-1)).mul(WIDTH).add(int(float(sx).clamp(0,WIDTH-1)))))).div(4096).min(0.6),float(0));
    const birth=float(1).sub(exp(production.mul(-STEP).sub(landing)));
    const boundary=exp(float(1).sub(smoothstep(0,12,edge)).mul(-STEP*2));
    const oldFoam=carried.z.mul(Math.exp(-STEP/45));
    const injected=float(1).sub(oldFoam).mul(birth);
    const foam=oldFoam.add(injected).clamp(0,1).mul(boundary);
    const renewal=injected.div(oldFoam.add(injected).max(0.0001));
    const flow=mix(mix(carriedHistory.xy,edgeFlow,float(1).sub(smoothstep(0.001,0.03,carried.z))).mul(Math.exp(-STEP/18)),worldFlow,renewal);
    const ageMoment=carriedHistory.z.mul(Math.exp(-STEP/45)).add(oldFoam.mul(STEP)).mul(boundary).min(foam.mul(120));
    const agitation=mix(carriedHistory.w.mul(Math.exp(-STEP/10)),(contacts?meshSource!.y.mul(0.2).add(release.mul(0.3)):speed.div(6)).add(landing.mul(2)).clamp(0,1),renewal);
    nextHistory.element(instanceIndex).assign(vec4(flow,ageMoment,agitation));
    // Subsurface aeration has a softer footprint and outlives surface foam.
    const entrainment=speed.div(3).clamp(0.05,1);
    const bubbleBirth=float(1).sub(exp(production.mul(entrainment).mul(-STEP*0.8).sub(landing.mul(1.4))));
    const bubbles=carried.w.mul(Math.exp(-STEP/70)).add(float(1).sub(carried.w).mul(bubbleBirth)).clamp(0,1).mul(boundary);
    next.element(instanceIndex).assign(vec4(height,velocity,foam,bubbles));
  })().compute(WIDTH*HEIGHT);
  const copy=Fn(() => { state.element(instanceIndex).assign(next.element(instanceIndex)); history.element(instanceIndex).assign(nextHistory.element(instanceIndex)); atomicStore(landings.element(instanceIndex),uint(0)); })().compute(WIDTH*HEIGHT);
  const clear=Fn(() => { state.element(instanceIndex).assign(vec4(0)); next.element(instanceIndex).assign(vec4(0)); history.element(instanceIndex).assign(vec4(0)); nextHistory.element(instanceIndex).assign(vec4(0)); atomicStore(landings.element(instanceIndex),uint(0)); })().compute(WIDTH*HEIGHT);
  const read=storage(state.value,'vec4',WIDTH*HEIGHT).toReadOnly();
  const historyRead=storage(history.value,'vec4',WIDTH*HEIGHT).toReadOnly();
  const sample=(point: THREE.Node<'vec2'>,buffer=read) => {
    const p=point.sub(origin).div(CELL), cell=floor(p), fraction=p.sub(cell);
    // Continuous first derivatives prevent visible grid seams in foam relief.
    const f=fraction.mul(fraction).mul(vec2(3).sub(fraction.mul(2)));
    const x=int(cell.x),y=int(cell.y);
    const at=(i: THREE.Node<'int'>,j: THREE.Node<'int'>) => buffer.element(int(float(j).clamp(0,HEIGHT-1)).mul(WIDTH).add(int(float(i).clamp(0,WIDTH-1))));
    const inside=p.x.greaterThanEqual(1).and(p.y.greaterThanEqual(1)).and(p.x.lessThan(WIDTH-2)).and(p.y.lessThan(HEIGHT-2));
    return inside.select(mix(mix(at(x,y),at(x.add(1),y),f.x),mix(at(x,y.add(1)),at(x.add(1),y.add(1)),f.x),f.y),vec4(0));
  };
  // Scatter landing energy without readback. Consumed once by the next wake step;
  // clear in the separate copy dispatch so neighbouring reads cannot race clears.
  const deposit=(point:THREE.Node<'vec2'>,energy:THREE.Node<'float'>):void => {
    const grid=point.sub(origin).div(CELL), cell=floor(grid), fraction=grid.sub(cell), x=int(cell.x), y=int(cell.y);
    If(x.greaterThanEqual(1).and(y.greaterThanEqual(1)).and(x.lessThan(WIDTH-2)).and(y.lessThan(HEIGHT-2)),()=>{
      for(const [dx,dy] of [[0,0],[1,0],[0,1],[1,1]]) {
        const weight=(dx?fraction.x:float(1).sub(fraction.x)).mul(dy?fraction.y:float(1).sub(fraction.y));
        atomicAdd(landings.element(y.add(dy).mul(WIDTH).add(x.add(dx))),uint(energy.clamp(0,1).mul(weight).mul(4096)));
      }
    });
  };
  let lastTime=0,lastX=0,lastY=0,lastHeading=0,simulationX=0,simulationY=0,simulationHeading=0,accumulator=0,initialized=false;
  return { sample:(point:THREE.Node<'vec2'>)=>sample(point), sampleHistory:(point:THREE.Node<'vec2'>)=>sample(point,historyRead), deposit, setHull(envelope: {stern:number;bow:number;centerY:number;beam:number}) {
    hull.value.set(envelope.stern,envelope.bow,envelope.centerY,envelope.beam);
  }, update(renderer: THREE.WebGPURenderer,time: number,x: number,y=0,heading=0) {
    const elapsed=time-lastTime, dx=x-lastX, dy=y-lastY;
    if(!initialized || elapsed<0 || elapsed>0.5 || Math.hypot(dx,dy)>5) {
      renderer.compute(clear); accumulator=0; simulationX=x; simulationY=y; simulationHeading=heading;
      origin.value.set(Math.floor((x-WIDTH*CELL/2)/CELL)*CELL,Math.floor((y-HEIGHT*CELL/2)/CELL)*CELL);
      pose.value.set(x,y,Math.cos(heading),Math.sin(heading)); previousPose.value.copy(pose.value);
      initialized=true; lastTime=time; lastX=x; lastY=y; lastHeading=heading; return;
    }
    contactStepScale.value=elapsed>0?Math.min(1,STEP/elapsed):0;
    accumulator+=elapsed;
    const vx=elapsed>0?dx/elapsed:0, vy=elapsed>0?dy/elapsed:0;
    const angular=elapsed>0?Math.atan2(Math.sin(heading-lastHeading),Math.cos(heading-lastHeading))/elapsed:0;
    speed.value=Math.hypot(vx,vy); clock.value=time;
    while(accumulator>=STEP) {
      previousPose.value.copy(pose.value);
      // Sample the authoritative hull segment at this fixed-step time. Integrating
      // the latest frame velocity across leftover time accumulates position error.
      const fraction=elapsed>0?Math.max(0,Math.min(1,(elapsed-accumulator+STEP)/elapsed)):1;
      simulationX=lastX+dx*fraction; simulationY=lastY+dy*fraction;
      simulationHeading=lastHeading+angular*elapsed*fraction;
      pose.value.set(simulationX,simulationY,Math.cos(simulationHeading),Math.sin(simulationHeading));
      const ox=Math.floor((simulationX-WIDTH*CELL/2)/CELL)*CELL, oy=Math.floor((simulationY-HEIGHT*CELL/2)/CELL)*CELL;
      shift.value.set(Math.round((ox-origin.value.x)/CELL),Math.round((oy-origin.value.y)/CELL)); origin.value.set(ox,oy);
      renderer.compute([solve,copy]); accumulator-=STEP;
    }
    lastTime=time;lastX=x;lastY=y;lastHeading=heading;
  } };
}
