import * as THREE from 'three/webgpu';

/** Rest mesh for the AVBD cloth, in the ship's authored coordinate system. */
export function createFlagGeometry(anchor:THREE.Vector3) {
  const geometry=new THREE.PlaneGeometry(2.7,1.75,24,14);
  const p=geometry.getAttribute('position');
  for(let row=0;row<=14;row++) for(let column=0;column<=24;column++)
    p.setXYZ(row*25+column,anchor.x+column/24*2.7,anchor.y,anchor.z-row/14*1.75);
  geometry.computeVertexNormals();
  return geometry;
}

export function createPirateFlag(geometries:THREE.BufferGeometry[]) {
  let top=-Infinity;
  for(const geometry of geometries) {
    const p=geometry.getAttribute('position');
    for(let i=0;i<p.count;i++) top=Math.max(top,p.getZ(i));
  }
  const anchor=new THREE.Vector3();let count=0;
  for(const geometry of geometries) {
    const p=geometry.getAttribute('position');
    for(let i=0;i<p.count;i++) if(p.getZ(i)>top-0.06) {anchor.add(new THREE.Vector3(p.getX(i),p.getY(i),p.getZ(i)));count++;}
  }
  anchor.divideScalar(Math.max(1,count));anchor.z=top;
  const canvas=document.createElement('canvas');canvas.width=768;canvas.height=512;
  const ctx=canvas.getContext('2d')!;
  ctx.fillStyle='#111316';ctx.fillRect(0,0,768,512);
  ctx.translate(384,256);ctx.fillStyle='#ded9c8';ctx.strokeStyle='#ded9c8';
  // Crossed bones and a deliberately simple, readable Jolly Roger silhouette.
  for(const angle of [-0.48,0.48]) {
    ctx.save();ctx.translate(0,65);ctx.rotate(angle);ctx.lineWidth=20;ctx.lineCap='round';
    ctx.beginPath();ctx.moveTo(-145,0);ctx.lineTo(145,0);ctx.stroke();
    for(const x of [-145,145]) for(const y of [-9,9]) {ctx.beginPath();ctx.arc(x,y,14,0,Math.PI*2);ctx.fill();}
    ctx.restore();
  }
  ctx.beginPath();ctx.ellipse(0,-70,76,83,0,0,Math.PI*2);ctx.fill();
  ctx.fillRect(-48,-30,96,62);
  ctx.fillStyle='#111316';
  for(const x of [-29,29]) {ctx.beginPath();ctx.ellipse(x,-66,20,25,x<0?-0.22:0.22,0,Math.PI*2);ctx.fill();}
  ctx.beginPath();ctx.moveTo(0,-44);ctx.lineTo(-12,-20);ctx.lineTo(12,-20);ctx.closePath();ctx.fill();
  for(let x=-32;x<=32;x+=16) ctx.fillRect(x,8,5,25);
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;texture.anisotropy=8;
  const geometry=createFlagGeometry(anchor);
  const material=new THREE.MeshStandardMaterial({map:texture,side:THREE.DoubleSide,roughness:0.93,metalness:0});
  const mesh=new THREE.Mesh(geometry,material);mesh.name='Masthead Jolly Roger';
  mesh.castShadow=true;mesh.receiveShadow=true;mesh.frustumCulled=false;
  return {mesh,material,restGeometry:geometry.clone()};
}
