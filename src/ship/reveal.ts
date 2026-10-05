import type { Atmosphere } from './weather.ts';

// Heading-relative rear-quarter offsets: keep the rigging in view through every turn.
export const revealShots = [
  {at:0,weather:'dawn',sea:'calm',angle:0.235,radius:52,z:17,target:11,lead:2},
  {at:4.5,weather:'day',sea:'breeze',angle:0.235,radius:34,z:11,target:9,lead:2},
  {at:9,weather:'day',sea:'breeze',angle:-0.65,radius:23,z:10,target:8,lead:0},
  {at:14,weather:'day',sea:'breeze',angle:-1.05,radius:17.5,z:10,target:8,lead:0},
  {at:18,weather:'day',sea:'breeze',angle:-0.75,radius:29,z:12,target:11,lead:1},
  {at:22,weather:'storm',sea:'swell',angle:-0.45,radius:50,z:10,target:11,lead:2},
  {at:32,weather:'dusk',sea:'breeze',angle:0.35,radius:50,z:10,target:11,lead:2},
  {at:38,weather:'night',sea:'glass',angle:0.15,radius:52,z:12,target:11,lead:2},
  {at:42,weather:'night',sea:'glass',angle:0.08,radius:52,z:12,target:11,lead:2},
  {at:45,weather:'night',sea:'glass',angle:0,radius:52,z:12,target:11,lead:2},
] as const satisfies readonly {at:number;weather:Atmosphere;sea:string;angle:number;radius:number;z:number;target:number;lead:number}[];

export function revealCamera(t:number) {
  const index=revealShots.reduce((current,shot,i)=>t>=shot.at?i:current,0);
  const a=revealShots[Math.max(0,index)], b=revealShots[Math.min(revealShots.length-1,index+1)];
  const u=Math.max(0,Math.min(1,(t-a.at)/Math.max(0.001,b.at-a.at)));
  const ease=u*u*(3-2*u), mix=(x:number,y:number)=>x+(y-x)*ease;
  return {angle:mix(a.angle,b.angle),radius:mix(a.radius,b.radius),z:mix(a.z,b.z),target:mix(a.target,b.target),lead:mix(a.lead,b.lead)};
}

/** Export companion to the DOM controls, reading their real selected state and dial geometry. */
export function createRevealComposite(source:HTMLCanvasElement) {
  const canvas=document.createElement('canvas');canvas.width=1920;canvas.height=1080;
  const ctx=canvas.getContext('2d')!;
  ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';
  const panel=(x:number,y:number,w:number,h:number)=>{ctx.fillStyle='rgba(243,248,244,0.94)';ctx.beginPath();ctx.roundRect(x,y,w,h,20);ctx.fill();};
  const text=(label:string,x:number,y:number,size=18,color='#21454b')=>{ctx.font=`${size}px system-ui`;ctx.fillStyle=color;ctx.fillText(label,x,y);};
  return {canvas,draw(t:number){
    ctx.drawImage(source,0,0,1920,1080);
    panel(1460,28,432,206);text('Conditions',1484,65,22);text('LIVE',1830,65,12,'#61817f');
    const buttons=(selector:string,y:number,width:number)=>document.querySelectorAll<HTMLButtonElement>(selector).forEach((button,i)=>{
      const x=1484+i*(width+6), active=button.classList.contains('selected');
      ctx.fillStyle=active?'#244b50':'#e6eeeb';ctx.beginPath();ctx.roundRect(x,y,width,36,7);ctx.fill();
      ctx.textAlign='center';text(button.textContent!.trim(),x+width/2,y+23,13,active?'#fff':'#35535a');ctx.textAlign='left';
    });
    buttons('[data-preset]',80,94);text('LIGHT & WEATHER',1484,141,11,'#6b8583');buttons('[data-weather]',153,74);
    text('Debug ▸',1484,220,12,'#6b8583');
    panel(1572,748,320,304);text('Wind vector',1596,784,19);text(document.getElementById('wind-readout')!.textContent!,1596,811,14,'#61817f');
    const cx=1732,cy=927,r=82;
    ctx.strokeStyle='#aec6c0';ctx.lineWidth=1;ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.stroke();
    ctx.setLineDash([4,5]);ctx.beginPath();ctx.arc(cx,cy,r/2,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
    ctx.beginPath();ctx.moveTo(cx-r,cy);ctx.lineTo(cx+r,cy);ctx.moveTo(cx,cy-r);ctx.lineTo(cx,cy+r);ctx.stroke();
    const handle=document.getElementById('wind-handle')!, x=cx+(Number(handle.getAttribute('cx'))-100)*r/80,y=cy+(Number(handle.getAttribute('cy'))-100)*r/80;
    ctx.strokeStyle='#2a6360';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);ctx.stroke();
    ctx.fillStyle='#2a6360';ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff';ctx.stroke();
    if(t>0.25&&t<4.2){ctx.strokeStyle='rgba(42,99,96,0.55)';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,15,0,Math.PI*2);ctx.stroke();}
    text('CAMERA-ALIGNED · DRAG TO SET WIND',1596,1030,11,'#61817f');
  }};
}

/** Record the rendered canvas and an optional Web Audio mix. */
export function recordReveal(canvas:HTMLCanvasElement,audio:MediaStream|undefined,onDone:(url:string)=>void):MediaRecorder {
  const video=canvas.captureStream(30);
  const stream=new MediaStream([...video.getVideoTracks(),...(audio?.getAudioTracks()??[])]);
  const mimeType=['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm'].find(type=>MediaRecorder.isTypeSupported(type));
  if(!mimeType) throw new Error('This browser cannot record WebM video.');
  const recorder=new MediaRecorder(stream,{mimeType,videoBitsPerSecond:18000000,audioBitsPerSecond:192000});
  const chunks:Blob[]=[];
  recorder.ondataavailable=event=>{if(event.data.size) chunks.push(event.data);};
  recorder.onstop=()=>{video.getTracks().forEach(track=>track.stop());onDone(URL.createObjectURL(new Blob(chunks,{type:mimeType})));};
  recorder.start(1000);return recorder;
}
