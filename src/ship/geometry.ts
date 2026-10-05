import { BufferGeometry, Float32BufferAttribute, Vector3 } from 'three';

/** Weld UV seams for connectivity, without changing the render mesh or its UVs. */
export function components(geometry: BufferGeometry): number[][] {
  const p = geometry.getAttribute('position');
  const parent = Array.from({ length: p.count }, (_, i) => i);
  const root = (i: number): number => {
    while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; }
    return i;
  };
  const join = (a: number, b: number): void => { parent[root(a)] = root(b); };
  const seen = new Map<string, number>();
  for (let i = 0; i < p.count; i++) {
    const key = [p.getX(i), p.getY(i), p.getZ(i)].map(x => Math.round(x * 1e4)).join(',');
    const prev = seen.get(key);
    if (prev !== undefined) join(i, prev);
    seen.set(key, i);
  }
  const ix = geometry.index;
  for (let i = 0; i < (ix?.count ?? p.count); i += 3) {
    const a = ix ? ix.getX(i) : i;
    join(a, ix ? ix.getX(i + 1) : i + 1);
    join(a, ix ? ix.getX(i + 2) : i + 2);
  }
  const groups = new Map<number, number[]>();
  for (let i = 0; i < p.count; i++) {
    const key = root(i);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(i);
  }
  return [...groups.values()];
}

export interface RopeNetwork {
  vertices: number[];
  nodes: Vector3[];
  edges: [number, number][];
  ladder: boolean;
  rings?: { vertices: number[]; center: Vector3; edge: number }[];
}

/** The asset's cordage is made of narrow open ribbons; spars are closed tubes.
 * Contract only ribbon-width edges. Faces then collapse into a centreline graph,
 * including the existing rung/side intersections of a rope ladder. */
export function ropeNetworks(geometry: BufferGeometry): RopeNetwork[] {
  const position = geometry.getAttribute('position'), index = geometry.index!;
  const groups = components(geometry);
  const groupOf = new Int32Array(position.count);
  groups.forEach((vertices, g) => vertices.forEach(v => { groupOf[v] = g; }));
  const triangles: number[][] = groups.map(() => []);
  for (let i = 0; i < index.count; i += 3) triangles[groupOf[index.getX(i)]].push(index.getX(i), index.getX(i + 1), index.getX(i + 2));
  const networks: RopeNetwork[] = [];
  groups.forEach((vertices, g) => {
    // This checksum-pinned connector is an ordered two-vertex ribbon. Its densely
    // sampled pulley eye has longitudinal edges shorter than its width: generic
    // short-edge contraction would merge the whole eye and reject the rope.
    if(vertices[0]===33425 && vertices.length===74) {
      const nodes:Vector3[]=[];
      for(let i=0;i<vertices.length;i+=2) nodes.push(new Vector3().fromBufferAttribute(position,vertices[i])
        .add(new Vector3().fromBufferAttribute(position,vertices[i+1])).multiplyScalar(0.5));
      networks.push({vertices,nodes,edges:nodes.slice(1).map((_,i)=>[i,i+1]),ladder:false});
      return;
    }
    const unique = new Map<string, number>(), weld = new Map<number, number>(), points: Vector3[] = [];
    for (const v of vertices) {
      const point = new Vector3().fromBufferAttribute(position, v);
      const key = point.toArray().map(x => Math.round(x * 1e4)).join(',');
      if (!unique.has(key)) { unique.set(key, points.length); points.push(point); }
      weld.set(v, unique.get(key)!);
    }
    const faces = new Map<string, number[]>(), edges = new Map<string, { a: number; b: number; count: number }>();
    const tris = triangles[g];
    for (let i = 0; i < tris.length; i += 3) {
      const face = tris.slice(i, i + 3).map(v => weld.get(v)!).sort((a, b) => a - b);
      faces.set(face.join(','), face);
    }
    for (const [a, b, c] of faces.values()) for (const [i, j] of [[a, b], [a, c], [b, c]]) {
      const key = `${i},${j}`;
      if (!edges.has(key)) edges.set(key, { a: i, b: j, count: 0 });
      edges.get(key)!.count++;
    }
    if (![...edges.values()].some(e => e.count === 1)) return; // closed wooden/metal hardware
    const parent = points.map((_, i) => i);
    const root = (i: number): number => {
      while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; }
      return i;
    };
    for (const { a, b } of edges.values()) if (points[a].distanceTo(points[b]) < 0.035) parent[root(a)] = root(b);
    // A surface that retains triangles is not narrow cordage.
    if ([...faces.values()].some(face => new Set(face.map(root)).size === 3)) return;
    const bins = new Map<number, Vector3[]>();
    points.forEach((point, i) => {
      const r = root(i);
      if (!bins.has(r)) bins.set(r, []);
      bins.get(r)!.push(point);
    });
    if (bins.size < 2 || [...bins.values()].filter(b => b.length === 1).length > 2) return;
    const nodes = [...bins.values()].map(bin => bin.reduce((p, v) => p.add(v), new Vector3()).multiplyScalar(1 / bin.length));
    // Reject broad pieces reduced by transitive short edges, while allowing rope knots.
    if ([...bins.values()].some((bin, i) => bin.some(p => p.distanceTo(nodes[i]) > 0.12))) return;
    const nodeOf = new Map([...bins.keys()].map((r, i) => [r, i]));
    const links = new Map<string, [number, number]>();
    for (const edge of edges.values()) {
      const a = nodeOf.get(root(edge.a))!, b = nodeOf.get(root(edge.b))!;
      if (a !== b) links.set([a, b].sort((x, y) => x - y).join(','), [a, b]);
    }
    const graph = [...links.values()];
    if (graph.reduce((length, [a, b]) => length + nodes[a].distanceTo(nodes[b]), 0) < 0.1) return;
    const degree = nodes.map(() => 0);
    graph.forEach(([a, b]) => { degree[a]++; degree[b]++; });
    networks.push({ vertices, nodes, edges: graph, ladder: degree.filter(d => d >= 3).length > 8 });
  });
  return networks;
}

/** Replace the asset's ribbon faces with round cordage on the same centreline graph.
 * Preserve source vertices (including hardware/sail indices) and sample the original
 * ribbon UVs at each end so the existing atlas still supplies the rope material. */
export function subdivideRopes(geometry: BufferGeometry, networks: RopeNetwork[]): void {
  const position = geometry.getAttribute('position'), uv = geometry.getAttribute('uv');
  const positions = Array.from(position.array), uvs = uv ? Array.from(uv.array) : null;
  const owner = new Set(networks.flatMap(network => network.vertices));
  const indices: number[] = [], index = geometry.index!;
  for (let i = 0; i < index.count; i += 3) {
    if (!owner.has(index.getX(i))) indices.push(index.getX(i), index.getX(i + 1), index.getX(i + 2));
  }
  const radialNormals = new Map<number, Vector3>();
  const ropeRadii = new Map<number, number>();
  const sides = 12;
  for (const network of networks) {
    const source = [...network.vertices];
    const samples = network.nodes.map(node => {
      const nearest = source.map(vertex => ({ vertex, distance: new Vector3().fromBufferAttribute(position, vertex).distanceToSquared(node) }))
        .sort((a, b) => a.distance - b.distance).slice(0, 2);
      return { radius: Math.max(0.01, Math.min(0.03, Math.sqrt(nearest[0].distance) * 1.5)),
        uv: nearest.map(({ vertex }) => uv ? [uv.getX(vertex), uv.getY(vertex)] : [0, 0]) };
    });
    network.vertices = [];
    network.rings = [];
    for (const [edge, [a, b]] of network.edges.entries()) {
      const start = network.nodes[a], end = network.nodes[b];
      const axis = end.clone().sub(start).normalize();
      const side = new Vector3().crossVectors(axis, Math.abs(axis.z) < 0.9 ? new Vector3(0, 0, 1) : new Vector3(0, 1, 0)).normalize();
      const up = new Vector3().crossVectors(axis, side);
      const segments = Math.max(1, Math.ceil(start.distanceTo(end) / 0.4));
      const base = positions.length / 3;
      for (let ring = 0; ring <= segments; ring++) {
        const t = ring / segments, center = start.clone().lerp(end, t);
        const ringVertices: number[] = [];
        network.rings.push({ vertices: ringVertices, center, edge });
        // Ratlines are thinner than load-bearing shrouds, but must survive pixel filtering.
        const minimum = network.ladder ? (Math.abs(axis.z) < 0.45 ? 0.012 : 0.02) : 0.01;
        const radius = Math.max(minimum, samples[a].radius * (1 - t) + samples[b].radius * t);
        for (let j = 0; j <= sides; j++) {
          const angle = j / sides * Math.PI * 2;
          const normal = side.clone().multiplyScalar(Math.cos(angle)).addScaledVector(up, Math.sin(angle));
          const vertex = positions.length / 3;
          positions.push(...center.clone().addScaledVector(normal, radius).toArray());
          ropeRadii.set(vertex,radius);
          radialNormals.set(vertex, normal); network.vertices.push(vertex); ringVertices.push(vertex);
          if (uvs) {
            // Fold the ribbon's width around the circumference, staying inside its atlas island.
            const across = 1 - Math.abs(2 * j / sides - 1);
            for (let k = 0; k < 2; k++) {
              const ua = samples[a].uv[0][k] * (1 - across) + samples[a].uv[1][k] * across;
              const ub = samples[b].uv[0][k] * (1 - across) + samples[b].uv[1][k] * across;
              uvs.push(ua * (1 - t) + ub * t);
            }
          }
          if (ring < segments && j < sides) {
            const x = base + ring * (sides + 1) + j, y = x + sides + 1;
            indices.push(x, x + 1, y, x + 1, y + 1, y);
          }
        }
      }
    }
  }
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  if (uvs) geometry.setAttribute('uv', new Float32BufferAttribute(uvs, 2));
  const radii=new Float32Array(positions.length/3);
  for(const [vertex,radius] of ropeRadii) radii[vertex]=radius;
  geometry.setAttribute('ropeRadius',new Float32BufferAttribute(radii,1));
  geometry.deleteAttribute('normal'); geometry.setIndex(indices); geometry.computeVertexNormals();
  const normals = geometry.getAttribute('normal');
  for (const [vertex, normal] of radialNormals) normals.setXYZ(vertex, normal.x, normal.y, normal.z);
}

/** Rope ends are not the only supports: a line can return around deck or yard hardware.
 * Preserve sharp routing turns and the far turn of out-and-back lines whose ends meet
 * at one attachment. Ordinary catenary curves are not pinned along their span. */
export function ropeSupports(network: RopeNetwork): Set<number> {
  const adjacent = network.nodes.map(() => [] as number[]);
  for (const [a, b] of network.edges) { adjacent[a].push(b); adjacent[b].push(a); }
  const supports = new Set<number>();
  if (network.ladder) return supports;
  network.nodes.forEach((point, i) => {
    if (adjacent[i].length !== 2) return;
    const a = network.nodes[adjacent[i][0]].clone().sub(point).normalize();
    const b = network.nodes[adjacent[i][1]].clone().sub(point).normalize();
    if (a.dot(b) > -0.5) supports.add(i);
  });
  const ends = adjacent.flatMap((links, i) => links.length === 1 ? [i] : []);
  if (ends.length === 2 && network.nodes[ends[0]].distanceTo(network.nodes[ends[1]]) < 0.2) {
    const origin = network.nodes[ends[0]].clone().add(network.nodes[ends[1]]).multiplyScalar(0.5);
    let farthest = ends[0];
    network.nodes.forEach((point, i) => {
      if (point.distanceToSquared(origin) > network.nodes[farthest].distanceToSquared(origin)) farthest = i;
    });
    if (network.nodes[farthest].distanceTo(origin) > 1) supports.add(farthest);
  }
  return supports;
}

/** Bilinear weights have zero contribution when a corner leaves the cell, avoiding
 * the discontinuities of a fixed-count nearest-neighbour skin. Sparse boundary cells
 * extend the closest occupied grid point, with the same lookup for adjacent cells. */
export function clothWeights(x: number, y: number, cols: number, rows: number, grid: readonly number[]): { body: number; weight: number }[] {
  const u = Math.max(0, Math.min(cols - 1, x)), v = Math.max(0, Math.min(rows - 1, y));
  const a = Math.floor(u), b = Math.floor(v), fu = u - a, fv = v - b;
  const weights = new Map<number, number>();
  for (const [dx, dy, w] of [[0, 0, (1 - fu) * (1 - fv)], [1, 0, fu * (1 - fv)], [0, 1, (1 - fu) * fv], [1, 1, fu * fv]]) {
    const body = grid[Math.min(rows - 1, b + dy) * cols + Math.min(cols - 1, a + dx)];
    if (w > 0) weights.set(body, (weights.get(body) ?? 0) + w);
  }
  return [...weights].map(([body, weight]) => ({ body, weight }));
}
