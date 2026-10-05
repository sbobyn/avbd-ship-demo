import assert from 'node:assert/strict';
import {test} from 'node:test';
import {SplashEnvelope,oceanMix,thunderDelay} from '../src/ship/audio.ts';

test('hull audio triggers a new contact burst but not continuous steady whitewater',()=>{
  const envelope=new SplashEnvelope();
  assert.equal(envelope.update(0,0.1),0);
  assert.ok(envelope.update(4,0.1)>0);
  for(let i=0;i<60;i++) assert.equal(envelope.update(4,0.1),0);
  for(let i=0;i<20;i++) envelope.update(0,0.1);
  assert.ok(envelope.update(8,0.1)>0);
});

test('hull audio warmup suppresses stale contact on enabling sound',()=>{
  const envelope=new SplashEnvelope();envelope.reset();
  for(let i=0;i<8;i++) assert.equal(envelope.update(3,0.1),0);
});


test('ocean conditions have distinct beds and increase in level with sea energy',()=>{
  const states=[oceanMix(2,true),oceanMix(3,false),oceanMix(7,false),oceanMix(12,false)];
  const levels=states.map(state=>Object.values(state).reduce((a,b)=>a+b,0));
  assert.ok(levels.every((v,i)=>i===0||v>levels[i-1]));
  assert.equal(states[0]['ocean-swell'],0);
  for(let wind=0;wind<=18;wind+=0.1) assert.ok(Object.values(oceanMix(wind,false)).every(v=>Number.isFinite(v)&&v>=0));
  assert.ok(Math.abs(oceanMix(7-0.001,false)['ocean-trade']-oceanMix(7+0.001,false)['ocean-trade'])<0.001);
});

test('thunder follows the flash with a short bounded cinematic delay',()=>{
  assert.equal(thunderDelay(0),0.25);
  assert.equal(thunderDelay(343/2),0.5);
  assert.equal(thunderDelay(3000),0.85);
});

test('larger splash bursts produce stronger impact cues',()=>{
  assert.ok(new SplashEnvelope().update(6,0.1)>new SplashEnvelope().update(1,0.1));
});
