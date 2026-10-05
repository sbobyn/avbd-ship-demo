import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Scene, Vector3 } from 'three/webgpu';
import { createStorm } from '../src/ship/storm.ts';

test('storm lightning has bounded short pulses, quiet intervals and no dry-weather flashes',()=>{
  const scene=new Scene(),storm=createStorm(scene),origin=new Vector3(120,-40,0);
  for(let i=0;i<100;i++) storm.update(0.05,origin);
  assert.equal(storm.flash.value,0);
  storm.intensity.value=1;
  const pulses:number[]=[];
  for(let i=0;i<600;i++) {storm.update(0.05,origin);pulses.push(storm.flash.value);}
  assert.ok(pulses.every(v=>Number.isFinite(v)&&v>=0&&v<=1));
  assert.ok(pulses.filter(v=>v>0.5).length>=4,'both the first strike and a later strike must flash');
  assert.ok(pulses.filter(v=>v>0.01).length<60,'flashes must be brief rather than continuously oscillating');
  storm.intensity.value=0;storm.update(0.05,origin);
  assert.equal(storm.flash.value,0,'returning to dry weather extinguishes lightning');
});


test('manual lightning works in dry weather and produces varied bounded channels',()=>{
  const scene=new Scene(),storm=createStorm(scene),origin=new Vector3();
  const sizes=new Set<number>();
  for(let strike=0;strike<6;strike++) {
    storm.triggerLightning(origin);
    sizes.add(scene.children[1].children.length);
    storm.update(0.02,origin);
    assert.ok(storm.flash.value>0&&storm.flash.value<=1,'manual strike flashes without enabling rain');
    assert.equal(storm.intensity.value,0);
    for(let i=0;i<30;i++) storm.update(0.05,origin);
    assert.ok(storm.flash.value<0.001,'manual flash clears');
  }
  assert.ok(sizes.size>1,'strikes vary their branch count');
});
