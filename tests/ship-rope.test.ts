import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BufferGeometry, Float32BufferAttribute, Vector3 } from 'three';
import { subdivideRopes } from '../src/ship/geometry.ts';

test('rope ribbons become round textured geometry without moving the centreline', () => {
  const geometry = new BufferGeometry()
    .setAttribute('position', new Float32BufferAttribute([0, -0.01, 0, 0, 0.01, 0, 1, -0.01, 0, 1, 0.01, 0], 3))
    .setAttribute('uv', new Float32BufferAttribute([0.2, 0.3, 0.25, 0.3, 0.2, 0.8, 0.25, 0.8], 2))
    .setIndex([0, 2, 1, 1, 2, 3]);
  const network = { vertices: [0, 1, 2, 3], nodes: [new Vector3(), new Vector3(1, 0, 0)], edges: [[0, 1] as [number, number]], ladder: false };
  subdivideRopes(geometry, [network]);
  const p = geometry.getAttribute('position'), uv = geometry.getAttribute('uv');
  assert.ok(Array.from(geometry.index!.array).every(i => i >= 4), 'flat source faces are removed');
  let depth = 0;
  for (const i of network.vertices) {
    assert.ok(Math.abs(Math.hypot(p.getY(i), p.getZ(i)) - 0.015) < 1e-7);
    assert.ok(uv.getX(i) >= 0.19999 && uv.getX(i) <= 0.25001);
    assert.ok(uv.getY(i) >= 0.29999 && uv.getY(i) <= 0.80001);
    depth = Math.max(depth, Math.abs(p.getZ(i)));
  }
  const radii=geometry.getAttribute('ropeRadius');
  assert.equal(radii.count,p.count,'every render vertex has a radius mask');
  for(const i of network.vertices) assert.ok(radii.getX(i)>0,'round cordage is marked for subpixel visibility');
  assert.equal(radii.getX(0),0,'source geometry is not expanded');
  assert.ok(depth > 0.008, 'rope has thickness perpendicular to the original ribbon');
});
