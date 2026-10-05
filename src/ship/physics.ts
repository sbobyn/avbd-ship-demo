import { Navigation, type PilotControls } from './navigation.ts';
import { SmoothTransition } from './transition.ts';
import { Conditions } from './conditions.ts';
import { BufferGeometry, Matrix4, Quaternion, Vector3 } from 'three';
import { NO_COLOR } from '../avbd2d/gpu/layout.ts';
import { Solver } from '../avbd3d/ref/solver.ts';
import { Rigid } from '../avbd3d/ref/body.ts';
import { Joint, Spring } from '../avbd3d/ref/forces.ts';
import { sail } from '../avbd3d/shapes.ts';
import { GpuSolver3D } from '../avbd3d/gpu/solver.ts';
import { B_ANGVEL, B_POS, B_ROT, B_VEL, BODY_FLOATS } from '../avbd3d/gpu/layout.ts';
import { clothWeights, components, ropeNetworks, ropeSupports, subdivideRopes, type RopeNetwork } from './geometry.ts';

interface Influence { body: number; local: Vector3; weight: number }
interface Skin { geometry: BufferGeometry; weights: Influence[][]; bindings?: Float32Array; offsets?: Uint32Array }
export interface ShipControls { wind: number; direction: number; gust: number; heave: number; roll: number; period: number }
const frame = new Quaternion().setFromUnitVectors(new Vector3(0, 0, 1), new Vector3(1, 0, 0));
const STEP = 1 / 120;

/** Thin rigid patches and ball joints, matching the engine's existing flag demo. */
export class ShipPhysics {
  readonly yards: { stowed:boolean; body:number; motor:number; pivot:Vector3; axis:Vector3; halfLength:number; vertices:number[] }[] = [];
  private readonly bracing = new SmoothTransition({ angle: 0 }, 1.2);
  readonly reference = new Solver();
  readonly hull: Rigid;
  readonly skins: Skin[] = [];
  readonly sailBodies: number[] = [];
  readonly furledBodies: number[][] = [];
  readonly gpu: GpuSolver3D;
  readonly ropeCount: number;
  readonly ladderCount: number;
  readonly riggingBodies: number[] = [];
  readonly ladderBodies: number[] = [];
  readonly ladderRungSpans: number[][] = [];
  readonly ropeSupports: { body: number; point: Vector3; local: Vector3 }[] = [];
  readonly sailCount: number;
  readonly yardHardware: { body:number; vertices:number[]; points:Vector3[] }[] = [];
  readonly hangingBlocks: { point: Vector3; body: number; points:Vector3[]; vertices:number[]; mount?:Vector3 }[] = [];
  readonly sailHardware: { point: Vector3; body: number; vertices: number[] }[] = [];
  readonly positions: Vector3[];
  readonly rotations: Quaternion[];
  readonly initialPositions: Vector3[];
  profile = '';
  readonly timings = { submit: 0, readback: 0, skin: 0 };
  time = 0;
  readonly conditions = new Conditions();
  readonly navigation = new Navigation();
  readonly pilot: PilotControls = {rudder:0,power:1,autoTrim:true,trim:0};
  maxDisplacement = 0;
  readonly surfaceTarget = new Vector3();
  private readonly surfaceMid = new Vector3();
  private readonly surfacePose = new Vector3();
  private state: Float32Array;
  private readonly matrices: Float32Array;
  private readonly matrix = new Matrix4();
  private readonly unitScale = new Vector3(1, 1, 1);

  readonly device: GPUDevice;

  constructor(device: GPUDevice, sails: BufferGeometry, rigging: BufferGeometry, makeBuffer?: (count: number) => GPUBuffer, heading = 0, flagGeometry?: BufferGeometry) {
    this.device = device;
    this.reference.dt = STEP;
    this.reference.iterations = 12;
    this.reference.gravity = -9.81;
    this.reference.alpha = 0.95;
    this.hull = new Rigid(this.reference, [0.1, 0.1, 0.1], 0, 0, [0, 0, 0]);
    const sailGroups = components(sails);
    const yardGroups=components(rigging), yardPositions=rigging.getAttribute('position');
    for(const seed of [33911,34079,34503,34671,13614,13910]) {
      const vertices=yardGroups.find(group=>group.includes(seed))!;
      const points=vertices.map(v=>new Vector3().fromBufferAttribute(yardPositions,v));
      let a=points[0], b=points[1], longest=0;
      for(const start of points) for(const end of points) {
        const distance=start.distanceToSquared(end);
        if(distance>longest) { longest=distance; a=start; b=end; }
      }
      if(b.y<a.y) [a,b]=[b,a];
      // Vertex density is uneven along these tapered meshes. Its average is
      // not the spar centre and would put the aft hinge a metre off the mast.
      const pivot=a.clone().add(b).multiplyScalar(0.5);
      const axis=b.clone().sub(a).normalize(), halfLength=a.distanceTo(b)/2;
      const body=this.reference.bodies.length;
      const yard=new Rigid(this.reference,[0.2,halfLength*2,0.2],450,0.4,pivot.toArray());
      yard.positionAng.set(new Quaternion().setFromUnitVectors(new Vector3(0,1,0),axis).toArray());
      // Two points on the mast axis make a revolute joint: yaw remains free.
      for(const z of [-3,3]) this.connect(this.hull,yard,pivot.clone().add(new Vector3(0,0,z)));
      const stowed=seed===13910;
      // The furled lateen yard is secured by its mast mount and a restraining
      // stay. It must not be braced through the working square sails' yaw range.
      if(stowed) this.connect(this.hull,yard,pivot.clone().addScaledVector(axis,2));
      const motor=this.reference.bodies.length, tip=pivot.clone().addScaledVector(axis,2);
      const target=new Rigid(this.reference,[0.1,0.1,0.1],0,0,tip.toArray());
      new Spring(this.reference,target,yard,[0,0,0],tip.clone().sub(pivot).applyQuaternion(new Quaternion().fromArray(yard.positionAng).invert()).toArray(),50000,0);
      this.yards.push({stowed,body,motor,pivot,axis,halfLength,vertices});
    }
    // Authored component seeds in the checksum-pinned Poly Haven glTF: these two
    // canvas surfaces share the rigging material instead of the sails material.
    const riggingGroups = components(rigging);
    const furled = riggingGroups.filter(vertices => vertices.includes(1883) || vertices.includes(2072));
    if (furled.length !== 2) throw new Error('The model is missing its two rigging-material furled sails.');
    this.sailCount = sailGroups.length + furled.length;
    const sailWeights: Influence[][] = Array.from({ length: sails.getAttribute('position').count }, () => []);
    for (const vertices of sailGroups) this.buildSail(sails, vertices, sailWeights);
    this.skins.push({ geometry: sails, weights: sailWeights });
    const networks = ropeNetworks(rigging);
    subdivideRopes(rigging, networks);
    const ropeWeights: Influence[][] = Array.from({ length: rigging.getAttribute('position').count }, (_, i) => [this.influence(0, new Vector3().fromBufferAttribute(rigging.getAttribute('position'), i), 1)]);
    for(const yard of this.yards) {
      for(const group of riggingGroups) {
        if(group[0]!==yard.vertices[0] && !group.every(v=>this.yardAt(new Vector3().fromBufferAttribute(rigging.getAttribute('position'),v),0.3)?.body===yard.body)) continue;
        for(const vertex of group) ropeWeights[vertex]=[this.influence(yard.body,new Vector3().fromBufferAttribute(rigging.getAttribute('position'),vertex),1)];
      }
    }
    // Complete yard-end blocks, including their separate eyes/strops. Proximity
    // to the timber alone excludes their outer vertices and leaves floating parts.
    for(const [yardSeed, ...seeds] of [
      [33911,33655,33685,33719,33749],
      [34079,33783,33813,33847,33871],
      [34503,34247,34281,34311,34337],
      [34671,34375,34407,34439,34471],
      [13614,13782,13806,13846,13870,32500,32533,32575,32576,32593,32626,32668,32669],
    ]) {
      const yard=this.yards.find(yard=>yard.vertices.includes(yardSeed))!;
      const vertices=riggingGroups.filter(group=>seeds.includes(group[0])).flat();
      const points=vertices.map(vertex=>new Vector3().fromBufferAttribute(rigging.getAttribute('position'),vertex));
      vertices.forEach((vertex,i)=>ropeWeights[vertex]=[this.influence(yard.body,points[i],1)]);
      this.yardHardware.push({body:yard.body,vertices,points});
    }
    for (const vertices of furled) this.buildFurled(rigging, vertices, ropeWeights);
    // Four clew blocks, including their separate eyes and UV-seam pieces, in the
    // checksum-pinned asset. They travel with the sail, unlike deck/yard hardware.
    const clewSeeds = [35427, 35520, 35717, 35810];
    const blockParts = new Set([35427, 35460, 35502, 35503, 35520, 35553, 35595, 35596,
      35717, 35750, 35792, 35793, 35810, 35843, 35885, 35886]);
    const rp = rigging.getAttribute('position'), sp = sails.getAttribute('position');
    for (const seed of clewSeeds) {
      const group = riggingGroups.find(vertices => vertices.includes(seed))!;
      const point = group.reduce((sum, v) => sum.add(new Vector3().fromBufferAttribute(rp, v)), new Vector3()).multiplyScalar(1 / group.length);
      let nearest = 0, distance = Infinity;
      for (let v = 0; v < sp.count; v++) {
        const d = point.distanceToSquared(new Vector3().fromBufferAttribute(sp, v));
        if (d < distance) { distance = d; nearest = v; }
      }
      const body = sailWeights[nearest].reduce((best, influence) => influence.weight > best.weight ? influence : best).body;
      const vertices = riggingGroups.filter(part => blockParts.has(part[0]) && new Vector3().fromBufferAttribute(rp, part[0]).distanceTo(point) < 0.3).flat();
      for (const vertex of vertices) ropeWeights[vertex] = [this.influence(body, new Vector3().fromBufferAttribute(rp, vertex), 1)];
      this.sailHardware.push({ point, body, vertices });
    }
    // Bowsprit and aft-sail tackle: suspended blocks are supported by their ropes.
    // Their separate wooden cheeks and eyes must share one rigid transform.
    for (const seeds of [[10592, 10620], [10656, 10676], [3241, 3274, 3316, 3317],
      [2873, 2906, 2948, 2949], [2966, 2999, 3041, 3042],
      [10772,10804], [10854,10886], [12208,12240], [12290,12322],
      [32128,32161,32203,32204], [32221,32254,32296,32297]]) {
      const vertices = riggingGroups.filter(group => seeds.some(seed => group.includes(seed))).flat();
      const lo = new Vector3(Infinity, Infinity, Infinity), hi = lo.clone().negate();
      for (const vertex of vertices) { const p = new Vector3().fromBufferAttribute(rp, vertex); lo.min(p); hi.max(p); }
      const point = lo.clone().add(hi).multiplyScalar(0.5);
      const size = hi.clone().sub(lo).max(new Vector3(0.06, 0.06, 0.06));
      new Rigid(this.reference, size.toArray(), 450, 0.4, point.toArray());
      const body = this.reference.bodies.length - 1;
      for (const vertex of vertices) ropeWeights[vertex] = [this.influence(body, new Vector3().fromBufferAttribute(rp, vertex), 1)];
      const points=vertices.map(v=>new Vector3().fromBufferAttribute(rp,v));
      // Lower halyard and sheet blocks are secured to the ship; the upper
      // blocks remain rope-supported. Mount the lower eye, leaving swivel free.
      const mount=[10772,12290,32128,32221].includes(seeds[0]) ? points.reduce((lowest,p)=>p.z<lowest.z?p:lowest).clone() : undefined;
      if(mount) this.connect(this.hull,this.reference.bodies[body],mount);
      this.hangingBlocks.push({ point, body, vertices, points, mount });
    }
    for (const network of networks) this.buildNetwork(network, rigging, ropeWeights);
    this.ladderCount = networks.filter(n => n.ladder).length;
    this.ropeCount = networks.length - this.ladderCount;
    this.skins.push({ geometry: rigging, weights: ropeWeights });
    if(flagGeometry) this.buildFlag(flagGeometry);
    this.initialPositions = this.reference.bodies.map(b => new Vector3().fromArray(b.positionLin));
    this.positions = this.initialPositions.map(p => p.clone());
    this.rotations = this.reference.bodies.map(b => new Quaternion().fromArray(b.positionAng));
    this.matrices = new Float32Array(this.positions.length * 16);
    for (const skin of this.skins) {
      if (!skin.geometry.hasAttribute('normal')) skin.geometry.computeVertexNormals();
      const normal = skin.geometry.getAttribute('normal');
      const data: number[] = [], offsets = new Uint32Array(skin.weights.length + 1);
      skin.weights.forEach((weights, vertex) => {
        offsets[vertex] = data.length;
        for (const influence of weights) {
          const localNormal = new Vector3().fromBufferAttribute(normal, vertex).applyQuaternion(this.rotations[influence.body].clone().invert());
          data.push(influence.body, influence.weight, ...influence.local.toArray(), ...localNormal.toArray());
        }
      });
      offsets[skin.weights.length] = data.length;
      skin.bindings = new Float32Array(data); skin.offsets = offsets;
    }
    // Place the complete constrained assembly before upload. Turning only the
    // kinematic hull at startup violently drags the canvas out of its rest pose.
    const initialRotation = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), heading);
    this.navigation.heading = heading;
    this.reference.bodies.forEach((body, i) => {
      this.positions[i].applyQuaternion(initialRotation);
      this.rotations[i].premultiply(initialRotation);
      body.positionLin.set(this.positions[i].toArray());
      body.positionAng.set(this.rotations[i].toArray());
    });
    this.gpu = new GpuSolver3D(device, this.reference, { spatialSort: false, bodyBuffer: makeBuffer?.(this.reference.bodies.length) });
    // This showcase isolates cloth/rope constraints: no self-contact or hull collisions.
    const indices = this.reference.bodies.map((_, i) => i);
    this.gpu.setFilters(indices, indices.map(() => 1), indices.map(() => 0));
    // No contacts or changing topology: colour the joint graph once, rather than
    // running 16 colouring rounds and up to 24 primal dispatches per iteration.
    const neighbours = this.reference.bodies.map(() => new Set<number>());
    const bodyIndex = new Map(this.reference.bodies.map((body, i) => [body, i]));
    for (const force of this.reference.forces) if (force.bodyA && force.bodyA.mass > 0 && force.bodyB.mass > 0) {
      const a = bodyIndex.get(force.bodyA)!, b = bodyIndex.get(force.bodyB)!;
      neighbours[a].add(b); neighbours[b].add(a);
    }
    const colors = new Uint32Array(indices.length).fill(NO_COLOR);
    for (const i of indices.filter(i => this.reference.bodies[i].mass > 0).sort((a, b) => neighbours[b].size - neighbours[a].size)) {
      const used = new Set([...neighbours[i]].map(j => colors[j]));
      let color = 0;
      while (used.has(color)) color++;
      colors[i] = color;
    }
    this.gpu.fixedColors = colors;
    this.state = new Float32Array(indices.length * BODY_FLOATS);
  }

  private buildFlag(geometry: BufferGeometry): void {
    const p=geometry.getAttribute('position'), anchor=new Vector3().fromBufferAttribute(p,0);
    const cols=13, rows=8, dx=2.7/(cols-1), dz=1.75/(rows-1), ids:number[]=[];
    const orientation=new Quaternion().setFromAxisAngle(new Vector3(1,0,0),Math.PI/2);
    for(let row=0;row<rows;row++) for(let col=0;col<cols;col++) {
      const center=anchor.clone().add(new Vector3(col*dx,0,-row*dz));
      const body=sail(new Rigid(this.reference,[dx,dz,0.02],0.22/0.02,0.4,center.toArray()));
      body.positionAng.set(orientation.toArray());ids.push(this.reference.bodies.length-1);
      if(col===0) for(const side of [-1,1]) this.connect(this.hull,body,center.clone().add(new Vector3(0,0,side*dz*0.35)));
    }
    for(let row=0;row<rows;row++) for(let col=0;col<cols;col++) {
      const a=this.reference.bodies[ids[row*cols+col]];
      for(const [dc,dr] of [[1,0],[0,1],[1,1],[1,-1]]) {
        if(col+dc>=cols||row+dr<0||row+dr>=rows) continue;
        const b=this.reference.bodies[ids[(row+dr)*cols+col+dc]];
        this.connect(a,b,new Vector3().fromArray(a.positionLin).add(new Vector3().fromArray(b.positionLin)).multiplyScalar(0.5),0.002);
      }
    }
    const weights:Influence[][]=[];
    for(let i=0;i<p.count;i++) {
      const point=new Vector3().fromBufferAttribute(p,i);
      weights.push(clothWeights((point.x-anchor.x)/dx,(anchor.z-point.z)/dz,cols,rows,ids).map(({body,weight})=>this.influence(body,point,weight)));
    }
    this.sailBodies.push(...ids);
    this.skins.push({geometry,weights});
  }

  private yardAt(point:Vector3, radius=0.4) {
    return this.yards.find(yard=>{
      const offset=point.clone().sub(yard.pivot), along=offset.dot(yard.axis);
      return Math.abs(along)<yard.halfLength+0.2 && offset.addScaledVector(yard.axis,-along).length()<radius;
    });
  }

  private influence(body: number, p: Vector3, weight: number): Influence {
    const b = this.reference.bodies[body];
    return { body, local: p.clone().sub(new Vector3().fromArray(b.positionLin)).applyQuaternion(new Quaternion().fromArray(b.positionAng).invert()), weight };
  }

  private connect(a: Rigid, b: Rigid, point: Vector3, angular = 0): void {
    const local = (body: Rigid): number[] => point.clone().sub(new Vector3().fromArray(body.positionLin)).applyQuaternion(new Quaternion().fromArray(body.positionAng).invert()).toArray();
    new Joint(this.reference, a, b, local(a), local(b), Infinity, angular);
  }

  private buildSail(geometry: BufferGeometry, vertices: number[], weights: Influence[][]): void {
    const p = geometry.getAttribute('position');
    const bins = new Map<string, { points: Vector3[]; y: number; z: number; body?: Rigid }>();
    const lo = new Vector3(Infinity, Infinity, Infinity), hi = lo.clone().negate();
    for (const v of vertices) { const point = new Vector3().fromBufferAttribute(p, v); lo.min(point); hi.max(point); }
    const topCenter=lo.clone().add(hi).multiplyScalar(0.5); topCenter.z=hi.z;
    const mountingYard=this.yards.reduce((best,yard)=>yard.pivot.distanceToSquared(topCenter)<best.pivot.distanceToSquared(topCenter)?yard:best);
    const spacing = 0.42;
    for (const v of vertices) {
      const point = new Vector3().fromBufferAttribute(p, v);
      const y = Math.floor((point.y - lo.y) / spacing), z = Math.floor((point.z - lo.z) / spacing), key = `${y},${z}`;
      if (!bins.has(key)) bins.set(key, { points: [], y, z });
      bins.get(key)!.points.push(point);
    }
    const ids: number[] = [];
    for (const bin of bins.values()) {
      const center = bin.points.reduce((s, v) => s.add(v), new Vector3()).multiplyScalar(1 / bin.points.length);
      // 0.6 kg/m² canvas; local z is the aerodynamic normal.
      const body = sail(new Rigid(this.reference, [spacing * 0.92, spacing * 0.92, 0.035], 0.6 / 0.035, 0.4, center.toArray()));
      body.positionAng.set(frame.toArray());
      bin.body = body;
      ids.push(this.reference.bodies.length - 1);
    }
    this.sailBodies.push(...ids);
    for (const bin of bins.values()) {
      for (const [dy, dz] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
        const other = bins.get(`${bin.y + dy},${bin.z + dz}`);
        if (other) this.connect(bin.body!, other.body!, new Vector3().fromArray(bin.body!.positionLin).add(new Vector3().fromArray(other.body!.positionLin)).multiplyScalar(0.5), 0.3);
      }
      // Attach the upper edge along the spar, not a freely spinning patch centre.
      // Two anchors leave the intended hinge around the spar but prevent twist.
      if (![...bins.values()].some(other => other.y === bin.y && other.z > bin.z)) {
        const edge = bin.points.reduce((best, point) => point.z > best.z ? point : best);
        const center = new Vector3().fromArray(bin.body!.positionLin);
        center.z = edge.z;
        for (const side of [-1, 1]) this.connect(this.reference.bodies[mountingYard.body], bin.body!, center.clone().add(new Vector3(0, side * spacing * 0.35, 0)));
      }
    }
    const cols = Math.max(...[...bins.values()].map(bin => bin.y)) + 1;
    const rows = Math.max(...[...bins.values()].map(bin => bin.z)) + 1;
    const entries = [...bins.values()];
    const grid = Array.from({ length: cols * rows }, (_, i) => {
      const y = i % cols, z = Math.floor(i / cols);
      const bin = bins.get(`${y},${z}`) ?? entries.reduce((best, candidate) => (candidate.y - y) ** 2 + (candidate.z - z) ** 2 < (best.y - y) ** 2 + (best.z - z) ** 2 ? candidate : best);
      return this.reference.bodies.indexOf(bin.body!);
    });
    for (const vertex of vertices) {
      const point = new Vector3().fromBufferAttribute(p, vertex);
      weights[vertex] = clothWeights((point.y - lo.y) / spacing - 0.5, (point.z - lo.z) / spacing - 0.5, cols, rows, grid)
        .map(({ body, weight }) => this.influence(body, point, weight));
    }
  }

  private buildFurled(geometry: BufferGeometry, vertices: number[], weights: Influence[][]): void {
    const mountingYard=this.yards.find(yard=>yard.vertices.includes(vertices.includes(1883)?13910:13614))!;
    const p = geometry.getAttribute('position');
    const points = vertices.map(i => new Vector3().fromBufferAttribute(p, i));
    // Find the spar direction; the aft sail is diagonal, the bowsprit sail transverse.
    let a = points[0], b = points[1], distance = 0;
    for (const start of points) for (const end of points) if (start.distanceToSquared(end) > distance) {
      distance = start.distanceToSquared(end); a = start; b = end;
    }
    const u = b.clone().sub(a).normalize();
    const approximateNormal = vertices.includes(1883) ? new Vector3(0, 1, 0) : new Vector3(1, 0, 0);
    const v = new Vector3().crossVectors(approximateNormal, u).normalize();
    if (v.z < 0) v.negate();
    const normal = new Vector3().crossVectors(u, v).normalize();
    const rotation = new Quaternion().setFromRotationMatrix(new Matrix4().makeBasis(u, v, normal));
    const lengths = points.map(point => point.dot(u)), lo = Math.min(...lengths), hi = Math.max(...lengths);
    const count = Math.max(2, Math.ceil((hi - lo) / 0.35)), step = (hi - lo) / count;
    const grid: Rigid[][] = [], ids: number[] = [], widths: { lower: number; width: number }[] = [];
    for (let col = 0; col < count; col++) {
      const along = lo + (col + 0.5) * step;
      const nearby = [...points].sort((a, b) => Math.abs(a.dot(u) - along) - Math.abs(b.dot(u) - along)).slice(0, 12);
      const cross = nearby.map(point => point.dot(v));
      const lower = Math.min(...cross), upper = Math.max(...cross), width = Math.max(0.06, upper - lower);
      widths.push({ lower, width });
      const depth = nearby.reduce((sum, point) => sum + point.dot(normal), 0) / nearby.length;
      const column: Rigid[] = [];
      for (let row = 0; row < 3; row++) {
        const center = u.clone().multiplyScalar(along).addScaledVector(v, lower + width * row / 2).addScaledVector(normal, depth);
        const body = sail(new Rigid(this.reference, [step, width / 3, 0.025], 0.6 / 0.025, 0.4, center.toArray()));
        body.positionAng.set(rotation.toArray());
        ids.push(this.reference.bodies.length - 1); column.push(body);
        for (const neighbour of [row ? column[row - 1] : undefined, col ? grid[col - 1][row] : undefined]) if (neighbour) {
          this.connect(neighbour, body, center.clone().add(new Vector3().fromArray(neighbour.positionLin)).multiplyScalar(0.5), 0.015);
        }
        if (row === 2) for (const side of [-1, 1]) {
          this.connect(this.reference.bodies[mountingYard.body], body, center.clone().addScaledVector(u, side * step * 0.35));
        }
      }
      grid.push(column);
    }
    this.sailBodies.push(...ids); this.furledBodies.push(ids);
    const bodyGrid = Array.from({ length: count * 3 }, (_, i) => ids[(i % count) * 3 + Math.floor(i / count)]);
    for (const vertex of vertices) {
      const point = new Vector3().fromBufferAttribute(p, vertex);
      const along = Math.max(0, Math.min(count - 1, (point.dot(u) - lo) / step - 0.5));
      const left = Math.floor(along), right = Math.min(count - 1, left + 1), t = along - left;
      const lower = widths[left].lower * (1 - t) + widths[right].lower * t;
      const width = widths[left].width * (1 - t) + widths[right].width * t;
      weights[vertex] = clothWeights(along, (point.dot(v) - lower) / width * 2, count, 3, bodyGrid)
        .map(({ body, weight }) => this.influence(body, point, weight));
    }
  }

  private buildNetwork(network: RopeNetwork, geometry: BufferGeometry, weights: Influence[][]): void {
    // Short eyes/strops belong to the block assembly. Treating their open ribbon
    // ends as deck anchors leaves hooks suspended in object space beside a moving block.
    const assembly=!network.ladder ? this.hangingBlocks.find(block=>network.nodes.every(point=>point.distanceTo(block.point)<0.45))
      ?? this.yardHardware.find(block=>network.nodes.every(point=>block.points.some(p=>p.distanceToSquared(point)<0.04))) : undefined;
    if(assembly) {
      const position=geometry.getAttribute('position');
      const vertices=[...network.vertices,...network.rings!.flatMap(ring=>ring.vertices)];
      if('point' in assembly) assembly.vertices.push(...vertices);
      for(const vertex of vertices) weights[vertex]=[this.influence(assembly.body,new Vector3().fromBufferAttribute(position,vertex),1)];
      return;
    }
    const incident: { body: Rigid; id: number; rung: boolean }[][] = network.nodes.map(() => []);
    const segments: { a: Vector3; b: Vector3; id: number; edge: number }[] = [];
    for (const [edge, [a, b]] of network.edges.entries()) {
      const start = network.nodes[a], end = network.nodes[b];
      const length = start.distanceTo(end);
      const rung = network.ladder && Math.abs(end.z - start.z) < length * 0.45;
      // A single ratline span must not gain a free central hinge merely because
      // it crosses the rope subdivision threshold. Its ends still move freely.
      const count = rung ? 1 : Math.max(1, Math.ceil(length / 0.45));
      const span: number[] = [];
      if (rung) this.ladderRungSpans.push(span);
      const rotation = new Quaternion().setFromUnitVectors(new Vector3(1, 0, 0), end.clone().sub(start).normalize());
      let previous: Rigid | undefined;
      for (let i = 0; i < count; i++) {
        const p = start.clone().lerp(end, i / count), q = start.clone().lerp(end, (i + 1) / count);
        const body = sail(new Rigid(this.reference, [length / count, 0.025, 0.025], 550, 0.4, p.clone().add(q).multiplyScalar(0.5).toArray()));
        body.positionAng.set(rotation.toArray());
        const id = this.reference.bodies.length - 1;
        this.riggingBodies.push(id);
        span.push(id);
        if (network.ladder) this.ladderBodies.push(id);
        segments.push({ a: p, b: q, id, edge });
        if (previous) this.connect(previous, body, p, network.ladder ? 0.5 : 0);
        if (i === 0) incident[a].push({ body, id, rung: network.ladder && Math.abs(end.z - start.z) < length * 0.45 });
        if (i === count - 1) incident[b].push({ body, id, rung: network.ladder && Math.abs(end.z - start.z) < length * 0.45 });
        previous = body;
      }
    }
    const heights = network.nodes.map(p => p.z), bottom = Math.min(...heights), top = Math.max(...heights);
    const supports = ropeSupports(network);
    incident.forEach((links, i) => {
      // Rungs share constraints with the side ropes; no individually fixed ladder rungs.
      for (let j = 1; j < links.length; j++) this.connect(links[0].body, links[j].body, network.nodes[i]);
      // A ratline crossing a shroud is continuous rope, not a freely folding hinge.
      const rungs = links.filter(link => link.rung);
      for (let j = 1; j < rungs.length; j++) {
        this.connect(rungs[0].body, rungs[j].body, network.nodes[i]);
        this.connect(rungs[0].body, rungs[j].body, network.nodes[i].clone().add(new Vector3(0, 0, 0.08)));
      }
      const hanging = !network.ladder ? this.hangingBlocks.filter(block => block.points.some(p=>p.distanceToSquared(network.nodes[i])<0.0225))
        .sort((a, b) => a.point.distanceToSquared(network.nodes[i]) - b.point.distanceToSquared(network.nodes[i]))[0] : undefined;
      const anchor = hanging !== undefined || (network.ladder
        ? network.nodes[i].z < bottom + 0.12 || network.nodes[i].z > top - 0.12
        : links.length === 1 || supports.has(i));
      if (anchor && links.length) {
        const point = network.nodes[i];
        const hardware = this.sailHardware.find(block => block.point.distanceTo(point) < 0.3);
        const yard=this.yardHardware.find(block=>block.points.some(p=>p.distanceToSquared(point)<0.04)) ?? this.yardAt(point);
        const sailId = hanging?.body ?? hardware?.body ?? yard?.body ?? this.sailBodies.find(id => new Vector3().fromArray(this.reference.bodies[id].positionLin).distanceTo(point) < 0.35);
        this.connect(sailId === undefined || network.ladder || (supports.has(i) && !hardware && !hanging && !yard) ? this.hull : this.reference.bodies[sailId], links[0].body, point);
        if (supports.has(i) && !hardware && !hanging && !yard) this.ropeSupports.push({
          body: links[0].id, point: point.clone(),
          local: point.clone().sub(new Vector3().fromArray(links[0].body.positionLin)).applyQuaternion(new Quaternion().fromArray(links[0].body.positionAng).invert()),
        });
      }
    });
    // Closed short lanyards need one attachment, while open ropes attach at their ends.
    if (!network.ladder && !network.nodes.some(point => this.hangingBlocks.some(block => block.points.some(p=>p.distanceToSquared(point)<0.0225))) && incident.every(links => links.length !== 1)) this.connect(this.reference.bodies[this.yardAt(network.nodes[0])?.body ?? 0], incident[0][0].body, network.nodes[0]);
    const p = geometry.getAttribute('position');
    // Skin each round cross-section as a unit. Blending independently rotating
    // links can collapse a circle into a ribbon (the candy-wrapper artifact).
    // Restrict ownership to this edge so ladder rungs cannot bind to side ropes.
    for (const ring of network.rings!) {
      const candidates = segments.filter(segment => segment.edge === ring.edge);
      const nearest = candidates.reduce((best, segment) => {
        const distance = (s: typeof segment): number => ring.center.distanceToSquared(s.a.clone().add(s.b).multiplyScalar(0.5));
        return distance(segment) < distance(best) ? segment : best;
      });
      for (const vertex of ring.vertices) {
        const point = new Vector3().fromBufferAttribute(p, vertex);
        weights[vertex] = [this.influence(nearest.id, point, 1)];
      }
    }
  }

  step(controls: ShipControls, steps: number): void {
    const started = performance.now();
    for (let i = 0; i < steps; i++) {
      const smooth = this.conditions.advance(controls, STEP);
      Object.assign(this.gpu.params, { windSpeed: smooth.wind, windAngle: smooth.direction * Math.PI / 180, windGust: smooth.gust });
      const t = this.time + STEP, phase = this.conditions.phase;
      const ramp = Math.min(t / 3, 1);
      const response=1-Math.exp(-STEP*3);
      this.surfaceMid.lerp(this.surfaceTarget,response);
      this.surfacePose.lerp(this.surfaceMid,response);
      this.navigation.advance(smooth.wind,smooth.direction,this.pilot,STEP);
      const position = new Vector3(this.navigation.x, this.navigation.y, this.surfacePose.x + Math.sin(phase) * smooth.heave * ramp);
      const rotation = new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), this.surfacePose.y + Math.sin(phase * 0.81) * smooth.roll * Math.PI / 180 * ramp);
      rotation.multiply(new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), this.surfacePose.z + Math.sin(phase * 0.67) * smooth.roll * Math.PI / 540 * ramp));
      rotation.premultiply(new Quaternion().setFromAxisAngle(new Vector3(0,0,1),this.navigation.heading));
      const velocity = position.clone().sub(this.positions[0]).multiplyScalar(1 / STEP);
      const delta = rotation.clone().multiply(this.rotations[0].clone().invert());
      const angular = new Vector3(delta.x, delta.y, delta.z).multiplyScalar((delta.w < 0 ? -2 : 2) / STEP);
      // Rebase the kinematic anchor to the authoritative previous pose each step.
      // Integrating f32 velocity forever drifts from the CPU-rendered hull on long voyages.
      this.device.queue.writeBuffer(this.gpu.bodyBuffer, B_POS * 4, new Float32Array(this.positions[0].toArray()));
      this.device.queue.writeBuffer(this.gpu.bodyBuffer, B_ROT * 4, new Float32Array(this.rotations[0].toArray()));
      // Same velocity-driven fixed-body convention as the public Body.moveTo API.
      this.device.queue.writeBuffer(this.gpu.bodyBuffer, B_VEL * 4, new Float32Array(velocity.toArray()));
      this.device.queue.writeBuffer(this.gpu.bodyBuffer, B_ANGVEL * 4, new Float32Array(angular.toArray()));
      // Automatic bracing is a soft position servo, not a teleport of the yard.
      const relative=smooth.direction*Math.PI/180-this.navigation.heading;
      const angle=Math.atan2(Math.sin(relative),Math.cos(relative));
      const broadside=Math.atan2(Math.sin(angle*2),Math.cos(angle*2))/2;
      this.bracing.setTarget({angle:this.pilot.autoTrim ? Math.max(-Math.PI/4,Math.min(Math.PI/4,broadside))*Math.min(1,smooth.wind/2) : Math.max(-45,Math.min(45,this.pilot.trim))*Math.PI/180});
      this.bracing.advance(STEP);
      const yaw=new Quaternion().setFromAxisAngle(new Vector3(0,0,1),this.bracing.values.angle);
      for(const yard of this.yards) {
        const target=yard.pivot.clone().add((yard.stowed ? yard.axis.clone() : yard.axis.clone().applyQuaternion(yaw)).multiplyScalar(2)).applyQuaternion(rotation).add(position);
        const previous=this.positions[yard.motor];
        this.device.queue.writeBuffer(this.gpu.bodyBuffer,(yard.motor*BODY_FLOATS+B_POS)*4,new Float32Array(previous.toArray()));
        this.device.queue.writeBuffer(this.gpu.bodyBuffer,(yard.motor*BODY_FLOATS+B_VEL)*4,new Float32Array(target.clone().sub(previous).multiplyScalar(1/STEP).toArray()));
        previous.copy(target);
      }
      if (Math.round(this.time / STEP) % 120 === 0) this.gpu.profileNextStep(p => { this.profile = Object.entries(p).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(' · '); });
      this.gpu.step();
      this.positions[0].copy(position); this.rotations[0].copy(rotation);
      this.time = t;
    }
    this.timings.submit = performance.now() - started;
  }

  /** Infrequent diagnostics only: never holds up rendering or stepping. */
  async inspect(): Promise<void> {
    const state = await this.gpu.readBodies();
    if (!state.every(Number.isFinite)) throw new Error('Non-finite physics state; reset the simulation.');
    const hullPosition = new Vector3().fromArray(state, B_POS), hullRotation = new Quaternion().fromArray(state, B_ROT);
    this.maxDisplacement = Math.max(...this.sailBodies.map(i => new Vector3().fromArray(state, i * BODY_FLOATS + B_POS).distanceTo(this.initialPositions[i].clone().applyQuaternion(hullRotation).add(hullPosition))));
    const counters = await this.gpu.readCounters();
    if (counters.overflow || counters.clashes) throw new Error(`Physics capacity exceeded: ${JSON.stringify(counters)}`);
  }

  /** CPU deformation path retained for headless state/geometry validation. */
  async advance(controls: ShipControls, steps: number): Promise<void> {
    this.step(controls, steps);
    const submitted = performance.now();
    this.state = await this.gpu.readBodies();
    const read = performance.now();
    this.timings.readback = read - submitted;
    if (!this.state.every(Number.isFinite)) throw new Error('Non-finite physics state; reset the simulation.');
    for (let i = 0; i < this.positions.length; i++) {
      this.positions[i].fromArray(this.state, i * BODY_FLOATS + B_POS);
      this.rotations[i].fromArray(this.state, i * BODY_FLOATS + B_ROT);
    }
    this.maxDisplacement = Math.max(...this.sailBodies.map(i => this.positions[i].distanceTo(this.initialPositions[i].clone().applyQuaternion(this.rotations[0]).add(this.positions[0]))));
    for (let i = 0; i < this.positions.length; i++) {
      this.matrix.compose(this.positions[i], this.rotations[i], this.unitScale);
      this.matrices.set(this.matrix.elements, i * 16);
    }
    const m = this.matrices;
    for (const skin of this.skins) {
      const p = skin.geometry.getAttribute('position'), normal = skin.geometry.getAttribute('normal');
      const positions = p.array, normals = normal.array, data = skin.bindings!, offsets = skin.offsets!;
      for (let vertex = 0; vertex < p.count; vertex++) {
        let x = 0, y = 0, z = 0, nx = 0, ny = 0, nz = 0;
        for (let j = offsets[vertex]; j < offsets[vertex + 1]; j += 8) {
          const k = data[j] * 16, w = data[j + 1], a = data[j + 2], b = data[j + 3], c = data[j + 4];
          x += (m[k] * a + m[k + 4] * b + m[k + 8] * c + m[k + 12]) * w;
          y += (m[k + 1] * a + m[k + 5] * b + m[k + 9] * c + m[k + 13]) * w;
          z += (m[k + 2] * a + m[k + 6] * b + m[k + 10] * c + m[k + 14]) * w;
          const u = data[j + 5], v = data[j + 6], t = data[j + 7];
          nx += (m[k] * u + m[k + 4] * v + m[k + 8] * t) * w;
          ny += (m[k + 1] * u + m[k + 5] * v + m[k + 9] * t) * w;
          nz += (m[k + 2] * u + m[k + 6] * v + m[k + 10] * t) * w;
        }
        const o = vertex * 3, length = Math.hypot(nx, ny, nz) || 1;
        positions[o] = x; positions[o + 1] = y; positions[o + 2] = z;
        normals[o] = nx / length; normals[o + 1] = ny / length; normals[o + 2] = nz / length;
      }
      p.needsUpdate = true; normal.needsUpdate = true;
    }
    this.timings.skin = performance.now() - read;
  }

  destroy(): void { this.gpu.destroy(); }
}
