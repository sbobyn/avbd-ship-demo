import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Matrix4, PerspectiveCamera, Vector4 } from 'three';
import { mirrorWaterProjection } from '../src/ship/reflection.ts';

test('water reflection covers both screen edges with an off-axis camera',()=>{
  for(const [width,height] of [[936,938],[1468,1960],[1920,1080]]) {
    for(const offset of [0,145,-145]) {
      const camera=new PerspectiveCamera(45,width/height,0.1,6000);
      camera.setViewOffset(width,height,offset,0,width,height);
      const original=camera.projectionMatrix.clone();
      const reflected=mirrorWaterProjection(original,new Matrix4());
      for(const x of [-0.999,0,0.999]) {
        const view=new Vector4(x,0.25,0.5,1).applyMatrix4(original.clone().invert());
        view.x=-view.x;
        const clip=view.applyMatrix4(reflected);
        assert.ok(Math.abs(clip.x/clip.w+x)<1e-10,'reflection must stay within the mirrored viewport');
        assert.ok(Math.abs(clip.y/clip.w-0.25)<1e-10);
      }
      assert.deepEqual(camera.projectionMatrix.elements,original.elements,'main camera stays unchanged');
    }
  }
});
