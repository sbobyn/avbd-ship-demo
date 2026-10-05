import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Vector3 } from 'three';
import { createFlagGeometry } from '../src/ship/flag.ts';

test('flag rest mesh keeps its hoist at the mast and preserves canvas dimensions',()=>{
  const geometry=createFlagGeometry(new Vector3(2,3,30));
  const p=geometry.getAttribute('position');
  for(let row=0;row<=14;row++) {
    assert.equal(p.getX(row*25),2);assert.equal(p.getY(row*25),3);
    assert.ok(Math.abs(p.getZ(row*25)-(30-row/14*1.75))<1e-5);
  }
  assert.ok(Math.abs(p.getX(24)-4.7)<1e-5);
  geometry.dispose();
});
