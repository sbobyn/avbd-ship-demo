import * as THREE from 'three/webgpu';
import { float, texture, uniform, vec4 } from 'three/tsl';

// Conservative solid cross-sections from the actual hull triangles. Built once;
// the inverse ship transform keeps the field attached during roll and steering.
export function createHullExclusion() {
  const size=1024, data=new Float32Array(size*size*4);
  const map=new THREE.DataTexture(data,size,size,THREE.RGBAFormat,THREE.FloatType);
  map.minFilter=map.magFilter=THREE.LinearFilter;
  const deckData=new Float32Array(size*size*4);
  const deckMap=new THREE.DataTexture(deckData,size,size,THREE.RGBAFormat,THREE.FloatType);
  deckMap.minFilter=deckMap.magFilter=THREE.LinearFilter;
  const deckBounds=uniform(new THREE.Vector4());
  const inverse=uniform(new THREE.Matrix4()), bounds=uniform(new THREE.Vector4(0,-8,1,16)), ready=uniform(0);
  const outside=(world:THREE.Node<'vec3'>,splash=false,water=false)=>{
    const p=inverse.mul(vec4(world,1)).xyz;
    const q=p.xz.sub(bounds.xy).div(bounds.zw);
    const span=texture(map,q).level(float(0));
    const inside=ready.greaterThan(0).and(q.x.greaterThanEqual(0)).and(q.x.lessThanEqual(1))
      .and(q.y.greaterThanEqual(0)).and(q.y.lessThanEqual(1)).and(span.z.greaterThan(0))
      .and(p.y.greaterThanEqual(span.x.sub(0.08))).and(p.y.lessThanEqual(span.y.add(0.08)));
    const deckUV=p.xy.sub(deckBounds.xy).div(deckBounds.zw);
    const deck=texture(deckMap,deckUV).level(float(0));
    const deckColumn=ready.greaterThan(0).and(deckUV.x.greaterThanEqual(0)).and(deckUV.x.lessThanEqual(1))
      .and(deckUV.y.greaterThanEqual(0)).and(deckUV.y.lessThanEqual(1)).and(deck.y.greaterThan(0))
      .and(p.z.greaterThanEqual(-8));
    if(water) return inside.or(deckColumn.and(p.z.greaterThanEqual(deck.x.sub(0.12)))).not();
    const underDeck=deckColumn.and(splash?float(1).greaterThan(0):p.z.lessThanEqual(deck.x.add(0.12)));
    return inside.or(underDeck).not();
  };
  return {outside, outsideWater:(world:THREE.Node<'vec3'>)=>outside(world,false,true), outsideSplash:(world:THREE.Node<'vec3'>)=>outside(world,true), update(matrix:THREE.Matrix4) { inverse.value.copy(matrix).invert(); },
    setHull(geometries:THREE.BufferGeometry[]) {
      let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
      for(const geometry of geometries) { geometry.computeBoundingBox(); minX=Math.min(minX,geometry.boundingBox!.min.x); maxX=Math.max(maxX,geometry.boundingBox!.max.x); minY=Math.min(minY,geometry.boundingBox!.min.y); maxY=Math.max(maxY,geometry.boundingBox!.max.y); }
      bounds.value.set(minX-0.1,-8,maxX-minX+0.2,16);
      deckBounds.value.set(minX-0.1,minY-0.1,maxX-minX+0.2,maxY-minY+0.2);
      data.fill(0); deckData.fill(0);
      const dx=bounds.value.z/size,dz=bounds.value.w/size;
      for(const geometry of geometries) {
        const p=geometry.getAttribute('position'), index=geometry.getIndex();
        for(let i=0;i<(index?.count??p.count);i+=3) {
          const ids=[0,1,2].map(j=>index?index.getX(i+j):i+j);
          const [a,b,c]=ids.map(j=>[p.getX(j),p.getY(j),p.getZ(j)]);
          // Horizontal deck triangles have no area in the hull's X/Z map.
          // Rasterize their height separately and treat the space below as solid.
          const deckDen=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
          const nx=(b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1]);
          const ny=(b[2]-a[2])*(c[0]-a[0])-(b[0]-a[0])*(c[2]-a[2]);
          if(Math.abs(deckDen)>1e-10 && Math.abs(deckDen)>Math.hypot(nx,ny) && Math.max(a[2],b[2],c[2])<7) {
            const sx=deckBounds.value.z/size,sy=deckBounds.value.w/size;
            const x0=Math.max(0,Math.floor((Math.min(a[0],b[0],c[0])-deckBounds.value.x)/sx)-1);
            const x1=Math.min(size-1,Math.ceil((Math.max(a[0],b[0],c[0])-deckBounds.value.x)/sx)+1);
            const y0=Math.max(0,Math.floor((Math.min(a[1],b[1],c[1])-deckBounds.value.y)/sy)-1);
            const y1=Math.min(size-1,Math.ceil((Math.max(a[1],b[1],c[1])-deckBounds.value.y)/sy)+1);
            for(let y=y0;y<=y1;y++) for(let x=x0;x<=x1;x++) {
              const px=deckBounds.value.x+(x+0.5)*sx,py=deckBounds.value.y+(y+0.5)*sy;
              const u=((b[1]-c[1])*(px-c[0])+(c[0]-b[0])*(py-c[1]))/deckDen;
              const v=((c[1]-a[1])*(px-c[0])+(a[0]-c[0])*(py-c[1]))/deckDen,w=1-u-v;
              if(Math.min(u,v,w)<-0.025) continue;
              const offset=(y*size+x)*4,z=u*a[2]+v*b[2]+w*c[2];
              deckData[offset]=deckData[offset+1]?Math.max(deckData[offset],z):z;
              deckData[offset+1]=1;
            }
          }
          const denominator=(b[2]-c[2])*(a[0]-c[0])+(c[0]-b[0])*(a[2]-c[2]);
          if(Math.abs(denominator)<1e-10) continue;
          const x0=Math.max(0,Math.floor((Math.min(a[0],b[0],c[0])-bounds.value.x)/dx)-1);
          const x1=Math.min(size-1,Math.ceil((Math.max(a[0],b[0],c[0])-bounds.value.x)/dx)+1);
          const z0=Math.max(0,Math.floor((Math.min(a[2],b[2],c[2])+8)/dz)-1);
          const z1=Math.min(size-1,Math.ceil((Math.max(a[2],b[2],c[2])+8)/dz)+1);
          for(let z=z0;z<=z1;z++) for(let x=x0;x<=x1;x++) {
            const px=bounds.value.x+(x+0.5)*dx,pz=-8+(z+0.5)*dz;
            const u=((b[2]-c[2])*(px-c[0])+(c[0]-b[0])*(pz-c[2]))/denominator;
            const v=((c[2]-a[2])*(px-c[0])+(a[0]-c[0])*(pz-c[2]))/denominator,w=1-u-v;
            if(Math.min(u,v,w)<-0.025) continue;
            const y=u*a[1]+v*b[1]+w*c[1], offset=(z*size+x)*4;
            if(!data[offset+2]) { data[offset]=y; data[offset+1]=y; data[offset+2]=1; }
            else { data[offset]=Math.min(data[offset],y); data[offset+1]=Math.max(data[offset+1],y); }
          }
        }
      }
      // Close holes in the deck footprint between its port/starboard edges.
      // Airborne splash is clipped over this footprint, not just below its skin.
      for(let x=0;x<size;x++) {
        let first=size,last=-1,top=-8;
        for(let y=0;y<size;y++) {const i=(y*size+x)*4;if(deckData[i+1]) {first=Math.min(first,y);last=y;top=Math.max(top,deckData[i]);}}
        for(let y=first;y<=last;y++) {const i=(y*size+x)*4;deckData[i]=top;deckData[i+1]=1;}
      }
      deckMap.needsUpdate=true; map.needsUpdate=true; ready.value=1;
    },
  };
}
