import assert from 'node:assert/strict';
import { test } from 'node:test';
import { PerspectiveCamera } from 'three/webgpu';
import { projectWind,unprojectWind,windViewBasis } from '../src/ship/wind-view.ts';

test('wind compass stays circular at every pitch and drag preserves direction and speed',()=>{
  const camera=new PerspectiveCamera();camera.up.set(0,0,1);
  for(const position of [[0,-20,12],[20,0,12],[-10,18,1],[0,0,20]]) {
    camera.position.fromArray(position);camera.lookAt(0,0,0);camera.updateMatrixWorld();
    const basis=windViewBasis(camera.matrixWorldInverse);
    const right=projectWind(basis,basis[0],basis[2]);
    assert.ok(Math.abs(right[0]-1)<1e-10&&Math.abs(right[1])<1e-10);
    for(const [x,y] of [[12,0],[0,7],[-8,5]]) {

      const projected=projectWind(basis,x,y), restored=unprojectWind(basis,...projected);
      assert.ok(Math.abs(Math.hypot(...projected)-Math.hypot(x,y))<1e-10,'camera pitch must not squash the dial');
      assert.ok(Math.hypot(restored[0]-x,restored[1]-y)<1e-10,'drag preserves the world direction and speed');
    }
  }
  assert.ok(unprojectWind([1,0,0,0],3,1).every(Number.isFinite),'edge-on fallback stays finite');
});
