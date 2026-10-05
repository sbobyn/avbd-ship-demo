import assert from 'node:assert/strict';
import { test } from 'node:test';
import { nearestAngle, nextDayAngle, SmoothTransition } from '../src/ship/transition.ts';

test('weather transitions ease from rest and converge without overshooting', () => {
  const transition=new SmoothTransition({exposure:0.7,clouds:0.8});
  transition.setTarget({exposure:1.1,clouds:0.3});
  assert.deepEqual(transition.values,{exposure:0.7,clouds:0.8});
  transition.advance(1/60);
  assert.ok(transition.values.exposure-0.7<0.001);
  for(let i=0;i<600;i++) {
    transition.advance(1/60);
    assert.ok(transition.values.exposure>=0.7 && transition.values.exposure<=1.1);
    assert.ok(transition.values.clouds>=0.3 && transition.values.clouds<=0.8);
  }
  assert.deepEqual(transition.values,{exposure:1.1,clouds:0.3});
  assert.equal(transition.advance(1/60),false,'settled sky needs no additional captures');
});

test('retargeting preserves the current value and rate', () => {
  const transition=new SmoothTransition({x:0}); transition.setTarget({x:1}); transition.advance(0.7);
  const delta=1e-6, a=transition.values.x; transition.advance(delta);
  const b=transition.values.x, velocity=(b-a)/delta;
  transition.setTarget({x:-1});
  assert.equal(transition.values.x,b);
  transition.advance(delta);
  assert.ok(Math.abs((transition.values.x-b)/delta-velocity)<0.0001);
});

test('weather response is independent of frame subdivisions', () => {
  const coarse=new SmoothTransition({x:0}), fine=new SmoothTransition({x:0});
  coarse.setTarget({x:1}); fine.setTarget({x:1}); coarse.advance(1);
  for(let i=0;i<120;i++) fine.advance(1/120);
  assert.ok(Math.abs(coarse.values.x-fine.values.x)<1e-12);
  const before=fine.values.x; fine.advance(0); assert.equal(fine.values.x,before);
});

test('sun azimuth takes the short route across the angle seam', () => {
  const radians=(degrees: number)=>degrees*Math.PI/180;
  assert.ok(Math.abs(nearestAngle(radians(179),radians(-179))-radians(181))<1e-12);
  assert.ok(Math.abs(nearestAngle(radians(-179),radians(179))-radians(-181))<1e-12);
});

test('night to dawn continues around the far side, including accumulated turns', () => {
  const radians=(degrees:number)=>degrees*Math.PI/180;
  for(const turns of [-2,0,3]) {
    const current=radians(-40+360*turns), target=radians(135);
    assert.ok(Math.abs(nextDayAngle(current,target)-radians(-225+360*turns))<1e-12);
    assert.ok(nextDayAngle(current,target)<current);
    assert.ok(nearestAngle(current,target)>current);
  }
});
