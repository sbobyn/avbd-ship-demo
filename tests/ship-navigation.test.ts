import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Navigation } from '../src/ship/navigation.ts';

const pilot={rudder:0,power:1,autoTrim:true,trim:0};
test('helm turns continuously, coasts to rest, and cannot turn a stopped ship',()=>{
  const n=new Navigation();
  for(let i=0;i<1200;i++) n.advance(0,0,{...pilot,rudder:1},1/120);
  assert.equal(n.heading,0);
  for(let i=0;i<3600;i++) {
    const old=n.heading;
    n.advance(12,0,{...pilot,rudder:1},1/120);
    assert.ok(Math.abs(n.heading-old)<=0.09/120+1e-9);
  }
  assert.ok(n.heading < -1 && n.y < -5,'starboard helm curves the trajectory');
  const before=n.speed;
  for(let i=0;i<6000;i++) n.advance(12,0,{...pilot,power:0},1/120);
  assert.ok(n.speed<before*0.001 && Math.abs(n.turnRate)<0.001);
});
test('navigation is consistent across step sizes and opposite rudders mirror the track',()=>{
  const run=(dt:number,rudder:number)=>{
    const n=new Navigation();
    for(let t=0;t<20-dt/2;t+=dt) n.advance(10,0,{...pilot,rudder},dt);
    return n;
  };
  const a=run(1/120,1),b=run(1/240,1),port=run(1/120,-1);
  assert.ok(Math.hypot(a.x-b.x,a.y-b.y)<0.04);
  assert.ok(Math.abs(a.x-port.x)<1e-9 && Math.abs(a.y+port.y)<1e-9);
});

test('sailing presets provide useful speed without headwind drive or instant acceleration',()=>{
  const cruise=new Navigation(),headwind=new Navigation(),storm=new Navigation(),swell=new Navigation(),calm=new Navigation(),glass=new Navigation();
  for(let i=0;i<7200;i++) {
    const previous=cruise.speed;
    cruise.advance(7,-35,pilot,1/120);
    headwind.advance(7,180,pilot,1/120);
    storm.advance(18,-45,pilot,1/120);
    swell.advance(12,-45,pilot,1/120);
    calm.advance(3,-35,pilot,1/120);
    glass.advance(2,-35,pilot,1/120);
    assert.ok(cruise.speed-previous<0.009,'acceleration is gradual');
  }
  assert.ok(cruise.speed*1.94384>10 && cruise.speed*1.94384<12,'trade wind reaches 10–12 knots');
  assert.ok(headwind.speed<0.01,'no forward drive directly into wind');
  assert.ok(storm.speed*1.94384<=20,'high wind respects the 20 knot cap'
  );
  assert.ok(swell.speed*1.94384>=15 && swell.speed*1.94384<=20,'rolling swell reaches 15–20 knots');
  assert.ok(calm.speed*1.94384>4 && calm.speed<cruise.speed,'calm makes steady headway');
  assert.ok(glass.speed*1.94384>3 && glass.speed<calm.speed,'glass cruises gently below calm');
});
