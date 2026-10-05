import type { BufferGeometry } from 'three';

export type BowContact = { x:number; y:number; nx:number; ny:number };

/** Rest-waterline envelope from triangle/plane intersections, in ship-local metres. */
export function waterlineEnvelope(geometries: BufferGeometry[], level = -0.45) {
  const segments: number[][]=[];
  let stern=Infinity, bow=-Infinity, port=-Infinity, starboard=Infinity;
  for (const geometry of geometries) {
    const p=geometry.getAttribute('position'), indices=geometry.getIndex();
    const count=indices?.count ?? p.count;
    for(let i=0;i<count;i+=3) {
      const crossings: number[][]=[];
      for(let edge=0;edge<3;edge++) {
        const a=indices?indices.getX(i+edge):i+edge;
        const b=indices?indices.getX(i+(edge+1)%3):i+(edge+1)%3;
        const za=p.getZ(a)-level, zb=p.getZ(b)-level;
        if(za*zb>0 || za===zb) continue;
        const t=za/(za-zb), x=p.getX(a)+(p.getX(b)-p.getX(a))*t, y=p.getY(a)+(p.getY(b)-p.getY(a))*t;
        crossings.push([x,y]);
        stern=Math.min(stern,x); bow=Math.max(bow,x); starboard=Math.min(starboard,y); port=Math.max(port,y);
      }
      if(crossings.length===2) segments.push([...crossings[0],...crossings[1]]);
    }
  }
  if(![stern,bow,port,starboard].every(Number.isFinite) || bow<=stern || port<=starboard) throw new Error('Ship hull does not intersect the rest waterline.');
  const bowContacts: BowContact[]=[];
  for(const fraction of [0.04,0.09,0.16,0.23]) {
    const x=bow-(bow-stern)*fraction;
    for(const side of [1,-1]) {
      let selected: BowContact | undefined;
      for(const [ax,ay,bx,by] of segments) {
        if(x<Math.min(ax,bx) || x>Math.max(ax,bx) || Math.abs(bx-ax)<1e-8) continue;
        const y=ay+(by-ay)*(x-ax)/(bx-ax), length=Math.hypot(bx-ax,by-ay);
        let nx=(by-ay)/length, ny=(ax-bx)/length;
        if(ny*side<0) { nx=-nx; ny=-ny; }
        if(!selected || y*side>selected.y*side) selected={x,y,nx,ny};
      }
      if(!selected) throw new Error('Ship bow waterline is not continuous.');
      bowContacts.push(selected);
    }
  }
  return { stern, bow, centerY:(port+starboard)/2, beam:(port-starboard)/2, bowContacts };
}

/** Exact input triangles; no convex hull or waterline ellipse is substituted. */
export function contactTriangles(geometries:BufferGeometry[],maximumEdge=Infinity, normals?:number[]) {
  if(!(maximumEdge>0)) throw new RangeError('Contact edge length must be positive.');
  const vertices:number[]=[];
  type Point=[number,number,number,number,number,number];
  const append=(a:Point,b:Point,c:Point):void=>{
    const edges=[[a,b,c],[b,c,a],[c,a,b]] as const;
    const lengths=edges.map(([p,q])=>(p[0]-q[0])**2+(p[1]-q[1])**2+(p[2]-q[2])**2);
    const longest=lengths.indexOf(Math.max(...lengths));
    // Refine the operating waterline band without spending vertices on spars or
    // deck fittings that are well clear of the water. Every child stays on its
    // original triangle; the input model's silhouette is unchanged.
    if(Math.min(a[2],b[2],c[2])<2.5 && Math.max(a[2],b[2],c[2])>-2.5 && lengths[longest]>maximumEdge**2) {
      const [p,q,r]=edges[longest], mid:Point=[(p[0]+q[0])/2,(p[1]+q[1])/2,(p[2]+q[2])/2,(p[3]+q[3])/2,(p[4]+q[4])/2,(p[5]+q[5])/2];
      append(p,mid,r); append(mid,q,r);
    } else for(const p of [a,b,c]) { vertices.push(p[0],p[1],p[2],1); normals?.push(p[3],p[4],p[5],0); }
  };
  for(const geometry of geometries) {
    const p=geometry.getAttribute('position'), n=geometry.getAttribute('normal'), indices=geometry.getIndex();
    const count=indices?.count??p.count;
    for(let i=0;i<count;i+=3) {
      const points=Array.from({length:3},(_,j):Point=>{ const v=indices?indices.getX(i+j):i+j; return [p.getX(v),p.getY(v),p.getZ(v),n?.getX(v)??0,n?.getY(v)??0,n?.getZ(v)??0]; });
      append(points[0],points[1],points[2]);
    }
  }
  return new Float32Array(vertices);
}
