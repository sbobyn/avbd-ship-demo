import * as THREE from 'three/webgpu';
import { float, floor, texture, vec2 } from 'three/tsl';

// Eight deterministic breakup/billow variants, shared by sheets and mist. Packed
// noise, thickness, soft density and relief stay mip-filtered in one small atlas.
let atlas: THREE.DataTexture | undefined;
function whitewaterAtlas() {
  if(atlas) return atlas;
  const tile=128, width=tile*4, height=tile*2, data=new Uint8Array(width*height*4);
  const hash=(x:number,y:number,seed:number)=>{
    let h=Math.imul(x,374761393)^Math.imul(y,668265263)^Math.imul(seed,1274126177);
    h=Math.imul(h^(h>>>13),1274126177); return ((h^(h>>>16))>>>0)/4294967296;
  };
  const noise=(x:number,y:number,seed:number)=>{
    const ix=Math.floor(x), iy=Math.floor(y), fx=x-ix, fy=y-iy;
    const u=fx*fx*(3-2*fx), v=fy*fy*(3-2*fy);
    return (hash(ix,iy,seed)*(1-u)+hash(ix+1,iy,seed)*u)*(1-v)
      +(hash(ix,iy+1,seed)*(1-u)+hash(ix+1,iy+1,seed)*u)*v;
  };
  for(let variant=0;variant<8;variant++) for(let y=0;y<tile;y++) for(let x=0;x<tile;x++) {
    const u=x/(tile-1), v=y/(tile-1), seed=variant*19+7;
    const coarse=noise(u*4.7,v*4.7,seed), fine=noise(u*17.3,v*17.3,seed+1);
    const n=coarse*0.65+fine*0.25+noise(u*43.1,v*43.1,seed+2)*0.1;
    // Uneven overlapping lobes with a zero border: no square atlas edges or
    // uniform disc silhouette, even as low mip levels merge the small wisps.
    let billow=0;
    for(let lobe=0;lobe<5;lobe++) {
      const cx=0.26+hash(lobe,3,seed)*0.48, cy=0.24+hash(lobe,5,seed)*0.52;
      const rx=0.13+hash(lobe,9,seed)*0.13, ry=0.1+hash(lobe,11,seed)*0.15;
      billow+=Math.exp(-(((u-cx)/rx)**2+((v-cy)/ry)**2))*(0.4+hash(lobe,7,seed)*0.6);
    }
    const border=Math.min(u,v,1-u,1-v), edge=Math.min(1,Math.max(0,border/0.12));
    const density=Math.max(0,Math.min(1,billow*(0.35+n*0.85)-(1-fine)*0.12))*edge*edge;
    // Fine cellular rims interrupt the broad lobes with bubbles and ligaments.
    let distance=2;
    const bx=u*18, by=v*18;
    for(let j=-1;j<=1;j++) for(let i=-1;i<=1;i++) {
      const cx=Math.floor(bx)+i,cy=Math.floor(by)+j;
      distance=Math.min(distance,Math.hypot(bx-cx-hash(cx,cy,seed+3),by-cy-hash(cx,cy,seed+4)));
    }
    const rim=Math.exp(-(((distance-0.32)/0.07)**2));
    const offset=((Math.floor(variant/4)*tile+y)*width+(variant%4)*tile+x)*4;
    data[offset]=Math.round(n*255);
    data[offset+1]=Math.round((0.25+coarse*0.5+rim*0.25)*255);
    data[offset+2]=Math.round(density*255);
    data[offset+3]=Math.round((fine*0.65+rim*0.35)*255);
  }
  atlas=new THREE.DataTexture(data,width,height);
  atlas.minFilter=THREE.LinearMipmapLinearFilter; atlas.magFilter=THREE.LinearFilter;
  atlas.generateMipmaps=true; atlas.needsUpdate=true;
  return atlas;
}

export function whitewaterDetail(point:THREE.Node<'vec2'>, index:THREE.Node<'uint'>) {
  const variant=index.mod(8);
  const cell=vec2(float(variant.mod(4)),floor(float(variant).div(4)));
  return texture(whitewaterAtlas(),point.clamp(0.008,0.992).add(cell).mul(vec2(0.25,0.5)));
}
