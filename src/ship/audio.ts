export interface ShipSoundState {
  rain:number; contact:number; impact:number; sample:number;
  strike:number; strikePosition:{x:number;y:number;z:number};
}

/** Detect contact bursts rather than playing an impact continuously in a steady wake. */
export class SplashEnvelope {
  private baseline=0;
  private cooldown=0;
  private previous=0;
  reset():void {this.baseline=0;this.previous=0;this.cooldown=0.8;}
  update(energy:number,dt:number):number {
    this.cooldown=Math.max(0,this.cooldown-dt);
    const rising=energy>this.previous+0.1;this.previous=energy;
    const burst=Math.max(0,energy-this.baseline);
    this.baseline+=(energy-this.baseline)*(1-Math.exp(-dt*3));
    if(!rising||this.cooldown>0||burst<0.3) return 0;
    this.cooldown=0.95;
    return burst/(burst+8);
  }
}

/** Continuous crossfades also support manually adjusted wind between presets. */
export function oceanMix(wind:number,glass:boolean) {
  const calm=1-Math.min(1,Math.max(0,(wind-3)/4));
  const swell=Math.min(1,Math.max(0,(wind-7)/5));
  return {'ocean-glass':glass?0.025:0,'ocean-calm':glass?0:Math.sqrt(calm)*0.18,
    'ocean-trade':glass?0:Math.sqrt(1-calm-swell)*0.45,'ocean-swell':glass?0:Math.sqrt(swell)*0.72};
}
export const thunderDelay=(distance:number):number=>Math.max(0.25,Math.min(0.85,distance/343));
const loopNames=['ocean-glass','ocean-calm','ocean-trade','ocean-swell','rain','rigging'] as const;
const names=[...loopNames,'hull-impact','thunder'] as const;
type Sound=typeof names[number];
type Point={x:number;y:number;z:number};

export function createShipAudio() {
  let context:AudioContext|undefined, master:GainNode, enabled=false, loading:Promise<void>|undefined;
  const buffers=new Map<Sound,AudioBuffer>(), loops=new Map<Sound,GainNode>();
  const voices=new Set<AudioBufferSourceNode>(), splash=new SplashEnvelope();
  let capture:MediaStreamAudioDestinationNode;
  let impactsPlayed=0, thunderPlayed=0;
  let sample=-1, strike=-1, lastWind:number|undefined, lastDirection=0, gust=0;
  const silence=()=>{if(context) master.gain.setTargetAtTime(enabled&&!document.hidden?0.8:0,context.currentTime,0.15);};
  document.addEventListener('visibilitychange',()=>{silence();if(document.hidden) {for(const voice of voices) voice.stop();voices.clear();}});
  function loopBuffer(input:AudioBuffer):AudioBuffer {
    const fade=Math.min(Math.floor(input.sampleRate*0.75),Math.floor(input.length/4));
    const output=context!.createBuffer(input.numberOfChannels,input.length-fade,input.sampleRate);
    for(let channel=0;channel<input.numberOfChannels;channel++) {
      const source=input.getChannelData(channel), target=output.getChannelData(channel);
      target.set(source.subarray(fade,input.length-fade));
      for(let i=0;i<fade;i++) {
        const t=i/fade;
        target[target.length-fade+i]=source[input.length-fade+i]*Math.cos(t*Math.PI/2)+source[i]*Math.sin(t*Math.PI/2);
      }
    }
    return output;
  }
  async function load():Promise<void> {
    const decoded=await Promise.all(names.map(async name=>{
      const response=await fetch(`/ship/audio/${name}.mp3?v=2`);
      if(!response.ok) throw new Error(`Could not load ${name} audio (${response.status})`);
      return [name,await context!.decodeAudioData(await response.arrayBuffer())] as const;
    }));
    for(const [name,buffer] of decoded) buffers.set(name,buffer);
    for(const name of loopNames) {
      const source=context!.createBufferSource(), gain=context!.createGain();
      source.buffer=loopBuffer(buffers.get(name)!);source.loop=true;
      gain.gain.value=0;source.connect(gain).connect(master);source.start();loops.set(name,gain);
    }
  }
  function play(name:Sound,level:number,position:Point,delay=0):void {
    if(!context||!enabled||document.hidden) return;
    if(voices.size>=8) {
      if(name!=='thunder') return;
      const oldest=voices.values().next().value;
      if(oldest) {oldest.stop();voices.delete(oldest);}
    }
    const buffer=buffers.get(name);if(!buffer) return;
    const source=context.createBufferSource(), gain=context.createGain(), panner=context.createPanner();
    source.buffer=buffer;source.playbackRate.value=name==='hull-impact'?1.02-Math.min(1,level)*0.15+Math.random()*0.06:0.94+Math.random()*0.08;
    panner.panningModel='HRTF';panner.distanceModel='inverse';panner.refDistance=name==='thunder'?1000:80;panner.rolloffFactor=name==='thunder'?0.15:0.35;panner.maxDistance=1500;
    panner.positionX.value=position.x;panner.positionY.value=position.y;panner.positionZ.value=position.z;
    gain.gain.value=level;source.connect(gain).connect(panner).connect(master);voices.add(source);
    source.onended=()=>{voices.delete(source);source.disconnect();gain.disconnect();panner.disconnect();};
    source.start(context.currentTime+delay);
    if(name==='thunder') thunderPlayed++;else if(name==='hull-impact') impactsPlayed++;
  }
  return {
    stream:()=>capture.stream,
    status:()=>`${buffers.size}/${names.length} sounds loaded · ${context?.state??'off'} · ${impactsPlayed} splashes · ${thunderPlayed} thunder cues`,
    async setEnabled(value:boolean):Promise<void> {
      enabled=value;
      if(!value) {silence();for(const voice of voices) voice.stop();voices.clear();return;}
      if(!context) {
        context=new AudioContext();master=context.createGain();master.gain.value=0;
        const limiter=context.createDynamicsCompressor();limiter.threshold.value=-3;limiter.knee.value=3;limiter.ratio.value=12;limiter.attack.value=0.003;limiter.release.value=0.25;
        master.connect(limiter).connect(context.destination);
        capture=context.createMediaStreamDestination();limiter.connect(capture);
      }
      await context.resume();
      try {await (loading??=load());splash.reset();silence();}
      catch(error) {loading=undefined;enabled=false;silence();throw error;}
    },
    update(dt:number,state:ShipSoundState,wind:number,direction:number,camera:Point,forward:Point,up:Point,hull:Point,active:boolean,glass=false):void {
      if(!context||!loops.size) return;
      const now=context.currentTime, listener=context.listener;
      for(const [parameter,value] of [[listener.positionX,camera.x],[listener.positionY,camera.y],[listener.positionZ,camera.z],
        [listener.forwardX,forward.x],[listener.forwardY,forward.y],[listener.forwardZ,forward.z],
        [listener.upX,up.x],[listener.upY,up.y],[listener.upZ,up.z]] as const) parameter.setTargetAtTime(value,now,0.03);
      const distance=Math.hypot(camera.x-hull.x,camera.y-hull.y,camera.z-hull.z), proximity=1/(1+distance/70);
      const turn=Math.abs(Math.atan2(Math.sin((direction-lastDirection)*Math.PI/180),Math.cos((direction-lastDirection)*Math.PI/180)));
      if(lastWind!==undefined) gust=Math.min(1,gust+Math.abs(wind-lastWind)*0.08+turn*0.4);
      lastWind=wind;lastDirection=direction;gust*=Math.exp(-dt*0.9);
      const audible=enabled&&active&&!document.hidden;
      if(!active&&voices.size) {for(const voice of voices) voice.stop();voices.clear();}
      const levels={...oceanMix(wind,glass),rain:state.rain*0.32,rigging:proximity*(0.025+Math.min(wind/18,1)*0.085+gust*0.1)};
      for(const name of loopNames) loops.get(name)!.gain.setTargetAtTime(audible?levels[name]:0,now,0.35);
      if(state.sample!==sample) {
        sample=state.sample;
        const strength=splash.update(state.impact,0.1);
        if(audible&&strength>0) play('hull-impact',Math.pow(strength,0.65)*(glass?0.06:1.35),hull);
      }
      if(state.strike!==strike) {
        if(strike>=0&&audible) {
          const p=state.strikePosition;
          play('thunder',1.5,p,thunderDelay(Math.hypot(p.x-camera.x,p.y-camera.y,p.z-camera.z)));
        }
        strike=state.strike;
      }
    },
  };
}
