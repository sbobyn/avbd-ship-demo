import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Conditions } from '../src/ship/conditions.ts';

test('late preset switches preserve phase and bound hull velocity', () => {
  const c = new Conditions();
  const calm={wind:0,direction:179,gust:0,heave:0,roll:0,period:12};
  let z=0, maxSpeed=0;
  for(let i=0;i<120*80;i++) {
    const target=i<120*60?calm:{wind:18,direction:-179,gust:1,heave:0.7,roll:8,period:3};
    const before=c.phase, smooth=c.advance(target,1/120);
    assert.ok(c.phase>before && c.phase-before<=2*Math.PI/3/120+1e-10);
    const next=Math.sin(c.phase)*smooth.heave;
    maxSpeed=Math.max(maxSpeed,Math.abs(next-z)*120); z=next;
  }
  assert.ok(maxSpeed<1.6, `preset change created a hull velocity spike: ${maxSpeed}`);
});
