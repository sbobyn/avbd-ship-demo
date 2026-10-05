import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BufferAttribute, BufferGeometry, Quaternion, Vector3 } from 'three';
import { ShipPhysics, type ShipControls } from '../src/ship/physics.ts';
import { ropeNetworks, subdivideRopes } from '../src/ship/geometry.ts';
import { Joint } from '../src/avbd3d/ref/forces.ts';
import { gpuTest } from './device.ts';
import { B_POS, B_ROT, BODY_FLOATS } from '../src/avbd3d/gpu/layout.ts';

/** Read only mesh data: no DOM, images, network, or alternative physics implementation. */
function geometry(mesh: number): BufferGeometry {
  const gltf = JSON.parse(readFileSync(new URL('../public/ship/ship.gltf', import.meta.url), 'utf8'));
  const bin = readFileSync(new URL('../public/ship/dutch_ship_medium.bin', import.meta.url));
  const primitive = gltf.meshes[mesh].primitives[0];
  const attribute = (index: number): BufferAttribute => {
    const a = gltf.accessors[index], view = gltf.bufferViews[a.bufferView];
    const itemSize = a.type === 'VEC3' ? 3 : 1;
    const offset = (view.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const bytes = a.componentType === 5123 ? 2 : 4;
    const buffer = bin.buffer.slice(bin.byteOffset + offset, bin.byteOffset + offset + a.count * itemSize * bytes);
    return new BufferAttribute(a.componentType === 5126 ? new Float32Array(buffer) : a.componentType === 5123 ? new Uint16Array(buffer) : new Uint32Array(buffer), itemSize);
  };
  return new BufferGeometry().setAttribute('position', attribute(primitive.attributes.POSITION)).setIndex(attribute(primitive.indices)).rotateX(Math.PI / 2);
}
const calm: ShipControls = { wind: 0, direction: 0, gust: 0, heave: 0, roll: 0, period: 5 };

gpuTest('ship: real asset sails and ropes remain finite, react to wind and follow moving attachments', async device => {
  const run = async (controls: ShipControls, batches = 90): Promise<{ sails: Vector3[]; furled: Vector3[][]; ladders: Vector3[]; hull: Vector3; rotation: number; deviation: number; yards: Vector3[] }> => {
    const rigging=geometry(0);
    const networks=ropeNetworks(rigging);
    const connector=networks.find(network=>network.vertices.includes(33425));
    assert.ok(connector,'the front connector must be extracted instead of remaining object-space geometry');
    assert.equal(connector.nodes.length,37,'preserve the tightly curved eye and complete connector centreline');
    subdivideRopes(rigging.clone(),networks);
    const ship = new ShipPhysics(device, geometry(2), rigging);
    try {
      assert.equal(ship.sailCount, 6);
      assert.equal(ship.hangingBlocks.length, 11);
      assert.equal(ship.hangingBlocks.filter(block=>block.mount).length,4,'lower halyard and sheet blocks have ship-mounted swivel eyes');
      for(const seed of [10772,12290]) {
        const owner=ship.skins[1].weights[seed][0].body;
        assert.ok(ship.hangingBlocks.find(block=>block.body===owner)?.mount,'lower mast tackle must attach to the ship, not only its paired ropes');
      }
      const aftBlocks=ship.hangingBlocks.filter(block => block.point.x < -7);
      assert.equal(aftBlocks.length,2,'both rear tackle blocks are dynamic');
      for(const seed of [2873,2906,2948,2949,2966,2999,3041,3042]) {
        assert.ok(aftBlocks.some(block => ship.skins[1].weights[seed][0].body===block.body),'rear pulley cheeks and eyes move with the simulated block');
      }
      const connectorOwners=new Set(connector.rings!.flatMap(ring=>ring.vertices.map(vertex=>ship.skins[1].weights[vertex][0].body)));
      assert.ok([...connectorOwners].every(body=>body!==0),'entire connector render mesh follows simulated links');
      for(const point of [connector.nodes[0],connector.nodes.at(-1)!]) {
        const block=ship.hangingBlocks.find(block=>block.points.some(p=>p.distanceToSquared(point)<0.0225));
        assert.ok(block,'both connector ends overlap their actual pulley assemblies');
        const body=ship.reference.bodies[block.body];
        assert.ok(ship.reference.forces.some(force=>force instanceof Joint && force.bodyA===body && (()=>{
          const anchor=new Vector3().fromArray(force.rA).applyQuaternion(new Quaternion().fromArray(body.positionAng)).add(new Vector3().fromArray(body.positionLin));
          return anchor.distanceTo(point)<1e-5;
        })()),'connector endpoint has a joint on the moving pulley, not a hull anchor');
      }
      for (const block of ship.hangingBlocks) {
        const body = ship.reference.bodies[block.body];
        assert.ok(body.mass > 0);
        for(const vertex of block.vertices) assert.equal(ship.skins[1].weights[vertex][0].body,block.body,'every pulley cheek, eye and strop shares its rope-constrained rigid body');
        const joints = ship.reference.forces.filter(f => f.bodyA === body || f.bodyB === body);
        assert.ok(joints.length > 0, 'suspended pulley has rope attachments');
        if(!block.mount) assert.ok(joints.every(f => f.bodyA !== ship.hull && f.bodyB !== ship.hull), 'suspended pulley must not be fixed to the hull');
        else assert.ok(joints.some(f=>f.bodyA===ship.hull && f.bodyB===body),'lower sheet block swivels at its deck-mounted eye');
      }
      for(const seed of [13806,13870]) assert.equal(ship.skins[1].weights[seed][0].body,ship.yards.find(yard=>yard.vertices.includes(13614))!.body,'front yard pulley eyes must not remain hull-fixed');
      assert.equal(ship.yardHardware.length,5);
      for(const block of ship.yardHardware) {
        assert.ok(block.vertices.length>100,'yard-end block includes separate strop and eye pieces');
        for(const vertex of block.vertices) assert.equal(ship.skins[1].weights[vertex][0].body,block.body,'entire pulley follows the yard instead of leaving hull-fixed fragments');
        const body=ship.reference.bodies[block.body];
        const attached=ship.reference.forces.filter(force=>{
          if(!(force instanceof Joint)||force.bodyA!==body) return false;
          const point=new Vector3().fromArray(force.rA).applyQuaternion(new Quaternion().fromArray(body.positionAng)).add(new Vector3().fromArray(body.positionLin));
          return ship.riggingBodies.includes(ship.reference.bodies.indexOf(force.bodyB)) && block.points.some(p=>p.distanceToSquared(point)<0.04);
        });
        assert.ok(attached.length>=2,'both yard-end blocks carry attached simulated ropes');
      }
      assert.equal(ship.skins[1].weights[3241][0].body,ship.skins[1].weights[3274][0].body,'both wooden cheeks belong to the same suspended front pulley');
      assert.equal(ship.sailHardware.length, 4);
      for (const block of ship.sailHardware) {
        assert.ok(ship.sailBodies.includes(block.body));
        assert.ok(block.vertices.length > 80, 'include the pulley eye and separate seam pieces');
        for (const vertex of block.vertices) assert.equal(ship.skins[1].weights[vertex][0].body, block.body);
      }
      assert.equal(ship.furledBodies.length, 2);
      assert.equal(ship.ladderCount, 10, 'all ten rope ladders must be simulated');
      assert.ok(ship.ladderRungSpans.length > 100);
      assert.ok(ship.ladderRungSpans.every(span => span.length === 1), 'wide ratline spans must not introduce a central hinge');
      assert.equal(ship.ropeCount, 426, 'all original rope sections plus the restored front connector');
      assert.ok(ship.ladderBodies.length > 1000);
      // The two deck return blocks shown in the reported dangling-line regression.
      for (const side of [-1, 1]) assert.ok(ship.ropeSupports.some(s => Math.abs(s.point.x + 5.55) < 0.02 && Math.abs(s.point.y - side * 1.24) < 0.02 && Math.abs(s.point.z - 3.92) < 0.02), 'deck return block has a hull constraint');
      assert.ok(ship.sailBodies.length > 100);
      // Main sails must have edge attachments, not single-point spinning mounts.
      const mainSails = new Set(ship.sailBodies.filter(id => !ship.furledBodies.some(group => group.includes(id))).map(id => ship.reference.bodies[id]));
      const yardBodies=new Set(ship.yards.map(yard=>ship.reference.bodies[yard.body]));
      assert.equal(ship.yards.length,6);
      const aftYard=ship.yards.find(yard=>yard.vertices.includes(13910))!;
      assert.ok(aftYard.pivot.distanceTo(new Vector3(-6.228,-0.161,7.851))<0.04,'aft hinge is at the authored mast crossing, not the vertex-density centroid');
      const mounts = new Map<object, Joint[]>();
      for (const force of ship.reference.forces) if (force instanceof Joint && force.bodyA && yardBodies.has(force.bodyA) && mainSails.has(force.bodyB)) {
        const joints = mounts.get(force.bodyB) ?? [];
        joints.push(force); mounts.set(force.bodyB, joints);
      }
      assert.ok(mounts.size > 0);
      for (const joints of mounts.values()) {
        assert.equal(joints.length, 2, 'spar mount must constrain twist');
        assert.ok(new Vector3().fromArray(joints[0].rA).distanceTo(new Vector3().fromArray(joints[1].rA)) > 0.2);
      }
      for (const force of ship.reference.forces) if (force.bodyA && force.bodyA.mass > 0 && force.bodyB.mass > 0) {
        const a = ship.reference.bodies.indexOf(force.bodyA), b = ship.reference.bodies.indexOf(force.bodyB);
        assert.notEqual(ship.gpu.fixedColors![a], ship.gpu.fixedColors![b], 'connected dynamic bodies cannot share a solver colour');
      }
      // Every dynamic node must be connected to the hull through joints.
      const reached = new Set([ship.hull]);
      for (let pass = 0; pass < ship.reference.bodies.length; pass++) for (const f of ship.reference.forces) {
        if (f.bodyA && reached.has(f.bodyA)) reached.add(f.bodyB);
        if (f.bodyA && reached.has(f.bodyB)) reached.add(f.bodyA);
      }
      assert.equal(reached.size, ship.reference.bodies.length, 'no floating/unattached cloth islands');
      let maxRoll = 0;
      for (let i = 0; i < batches; i++) {
        await ship.advance(controls, 4);
        maxRoll = Math.max(maxRoll, ship.rotations[0].angleTo(new Quaternion()));
      }
      for(const yard of ship.yards) {
        const pivot=yard.pivot.clone().applyQuaternion(ship.rotations[0]).add(ship.positions[0]);
        assert.ok(ship.positions[yard.body].distanceTo(pivot)<0.05,'yard centre stays on the moving mast pivot');
        const axis=new Vector3(0,1,0).applyQuaternion(ship.rotations[yard.body]).applyQuaternion(ship.rotations[0].clone().invert());
        assert.ok(Math.abs(axis.z-yard.axis.z)<0.015,`yard ${yard.vertices[0]} tilt ${axis.z-yard.axis.z}: must keep authored elevation`);
        for(const vertex of yard.vertices) assert.equal(ship.skins[1].weights[vertex][0].body,yard.body,'original yard geometry follows its dynamic body');
      }
      for(const block of ship.hangingBlocks) {
        if(block.mount) {
          const local=block.mount.clone().sub(ship.initialPositions[block.body]);
          const actual=local.applyQuaternion(ship.rotations[block.body]).add(ship.positions[block.body]);
          const expected=block.mount.clone().applyQuaternion(ship.rotations[0]).add(ship.positions[0]);
          assert.ok(actual.distanceTo(expected)<0.05,'deck-mounted pulley eye cannot swing away from the ship');
        }
        const hullFixed=ship.initialPositions[block.body].clone().applyQuaternion(ship.rotations[0]).add(ship.positions[0]);
        assert.ok(ship.positions[block.body].distanceTo(hullFixed)>0.001,'every suspended pulley must move relative to the hull');
      }
      for (const support of ship.ropeSupports) {
        const actual = support.local.clone().applyQuaternion(ship.rotations[support.body]).add(ship.positions[support.body]);
        const expected = support.point.clone().applyQuaternion(ship.rotations[0]).add(ship.positions[0]);
        assert.ok(actual.distanceTo(expected) < 0.05, `rope escaped its hardware support: ${actual.distanceTo(expected)} m`);
      }
      let jointError = 0;
      for (const force of ship.reference.forces) if (force instanceof Joint && force.bodyA) {
        const a = ship.reference.bodies.indexOf(force.bodyA), b = ship.reference.bodies.indexOf(force.bodyB);
        const pa = new Vector3().fromArray(force.rA).applyQuaternion(ship.rotations[a]).add(ship.positions[a]);
        const pb = new Vector3().fromArray(force.rB).applyQuaternion(ship.rotations[b]).add(ship.positions[b]);
        jointError = Math.max(jointError, pa.distanceTo(pb));
      }
      assert.ok(jointError < 0.3, `detached/stretching rigging: ${jointError} m joint error`);
      const counters = await ship.gpu.readCounters();
      assert.equal(counters.overflow, 0); assert.equal(counters.clashes, 0);
      assert.ok(ship.maxDisplacement < 5, `unbounded cloth displacement ${ship.maxDisplacement}`);
      for (const skin of ship.skins) assert.ok(Array.from(skin.geometry.getAttribute('position').array).every(Number.isFinite));
      return { yards:ship.yards.map(yard=>new Vector3(0,1,0).applyQuaternion(ship.rotations[yard.body]).applyQuaternion(ship.rotations[0].clone().invert())), furled: ship.furledBodies.map(ids => ids.map(i => ship.positions[i].clone())), sails: ship.sailBodies.map(i => ship.positions[i].clone()), ladders: ship.ladderBodies.map(i => ship.positions[i].clone()), hull: ship.positions[0].clone(), rotation: maxRoll, deviation: ship.maxDisplacement };
    } finally { ship.destroy(); }
  };
  const still = await run(calm);
  const wind = await run({ ...calm, wind: 12 });
  const port=await run({...calm,wind:8,direction:60});
  const starboard=await run({...calm,wind:8,direction:-60});
  console.log('yard response degrees',port.yards.map((axis,i)=>axis.angleTo(starboard.yards[i])*180/Math.PI));
  port.yards.slice(0,5).forEach((axis,i)=>assert.ok(axis.angleTo(starboard.yards[i])>0.5,
    `yard ${i} must visibly rotate with opposing wind: ${axis.angleTo(starboard.yards[i])} radians`));
  const windResponse = Math.max(...still.sails.map((p, i) => p.distanceTo(wind.sails[i])));
  assert.ok(windResponse > 0.05, `wind must change actual physics poses: ${windResponse}`);
  still.furled.forEach((points, group) => {
    const response = Math.max(...points.map((p, i) => p.distanceTo(wind.furled[group][i])));
    assert.ok(response > 0.001, `furled sail ${group} must respond to wind: ${response}`);
  });
  const ladderResponse = Math.max(...still.ladders.map((p, i) => p.distanceTo(wind.ladders[i])));
  assert.ok(ladderResponse > 0.01, `wind must deform the original rope ladders: ${ladderResponse}`);
  const swell = await run({ ...calm, wind: 18, gust: 1, heave: 0.7, roll: 8, period: 3 }, 300);
  assert.ok(swell.rotation > 0.02, 'hull roll is applied');
  assert.ok(swell.sails.some((p, i) => p.distanceTo(still.sails[i]) > 0.1), 'moving attachments affect the sails');
  console.log({ windResponse, ladderResponse, calmDisplacement: still.deviation, windDisplacement: wind.deviation, swellDisplacement: swell.deviation });
});

gpuTest('ship: render-loop stepping needs no readback to keep moving rigging stable', async device => {
  const ship = new ShipPhysics(device, geometry(2), geometry(0));
  try {
    const controls = { ...calm, wind: 18, gust: 1, heave: 0.7, roll: 8, period: 3 };
    for (let i = 0; i < 600; i++) ship.step(controls, 1);
    await ship.inspect();
    assert.ok(Math.abs(ship.time - 5) < 1e-8);
    assert.ok(ship.maxDisplacement > 0.1 && ship.maxDisplacement < 5);
  } finally { ship.destroy(); }
});


gpuTest('ship: wave-following hull rendering stays aligned with GPU rigging anchors', async device => {
  const ship = new ShipPhysics(device, geometry(2), geometry(0));
  try {
    // Start kilometres from the origin, as after a long-running sailing session.
    ship.navigation.x = 2000;
    ship.pilot.rudder=1; ship.pilot.autoTrim=false; ship.pilot.trim=30;
    const initial = await ship.gpu.readBodies();
    for (let i = 0; i < ship.positions.length; i++) {
      ship.positions[i].x += 2000;
      initial[i * BODY_FLOATS + B_POS] += 2000;
    }
    device.queue.writeBuffer(ship.gpu.bodyBuffer, 0, new Float32Array(initial));
    const controls = { ...calm, wind: 12, direction: -35, heave: 0.3, roll: 6 };
    for (let frame = 0; frame < 900; frame++) {
      ship.surfaceTarget.set(Math.sin(frame / 70) * 0.5, Math.sin(frame / 95) * 0.1, Math.cos(frame / 80) * 0.08);
      ship.step(controls, 4);
    }
    assert.ok(ship.navigation.heading < -0.2 && Math.abs(ship.navigation.y)>1,'helm turns the actual hull');
    const state = await ship.gpu.readBodies();
    const gpuPosition = new Vector3().fromArray(state, B_POS);
    const gpuRotation = new Quaternion().fromArray(state, B_ROT);
    assert.ok(gpuPosition.distanceTo(ship.positions[0]) < 0.002, `hull position drift: ${gpuPosition.distanceTo(ship.positions[0])}`);
    assert.ok(gpuRotation.angleTo(ship.rotations[0]) < 0.002, `hull rotation drift: ${gpuRotation.angleTo(ship.rotations[0])}`);
    for(const yard of ship.yards) {
      const position=new Vector3().fromArray(state,yard.body*BODY_FLOATS+B_POS);
      assert.ok(position.distanceTo(yard.pivot.clone().applyQuaternion(ship.rotations[0]).add(ship.positions[0]))<0.06,'yard remains attached during steering and manual trim');
      const local=new Quaternion().fromArray(state,yard.body*BODY_FLOATS+B_ROT).premultiply(ship.rotations[0].clone().invert());
      const axis=new Vector3(0,1,0).applyQuaternion(local);
      if(yard.stowed) assert.ok(axis.distanceTo(yard.axis)<0.03,'aft spar remains stowed while turning');
      else assert.ok(axis.distanceTo(yard.axis)>0.15,'manual trim rotates the working yards');
    }
  } finally { ship.destroy(); }
});

gpuTest('ship: stowed rear yard stays secured through a full wind sweep and abrupt sea changes', async device => {
  const ship=new ShipPhysics(device,geometry(2),geometry(0));
  try {
    const yard=ship.yards.find(yard=>yard.stowed)!;
    let previous: Quaternion | undefined;
    for(const direction of [0,45,90,135,180,225,270,315,360,180,0]) {
      for(const rough of [false,true]) {
        const controls={wind:rough?18:0.2,direction,gust:rough?1:0,heave:rough?0.7:0,roll:rough?8:0,period:rough?3:7};
        for(let frame=0;frame<60;frame++) {
          await ship.advance(controls,4);
          const local=ship.rotations[0].clone().invert().multiply(ship.rotations[yard.body]);
          const rest=new Quaternion().fromArray(ship.reference.bodies[yard.body].positionAng);
          assert.ok(local.angleTo(rest)<0.025,'furled rear yard stays in its secured orientation');
          assert.ok(ship.positions[yard.body].distanceTo(yard.pivot.clone().applyQuaternion(ship.rotations[0]).add(ship.positions[0]))<0.05,'stowed yard stays at the mast');
          if(previous) assert.ok(local.angleTo(previous)<0.02,'no rear yard snapping during retargeting');
          previous=local;
          for(const moving of ship.yards) {
            const axis=new Vector3(0,1,0).applyQuaternion(ship.rotations[moving.body]).applyQuaternion(ship.rotations[0].clone().invert());
            assert.ok(Math.abs(axis.z-moving.axis.z)<0.025,'working yard does not tip over');
          }
        }
      }
    }
  } finally { ship.destroy(); }
});

gpuTest('ship: reveal heading starts the whole rig at rest without a hull teleport', async device => {
  const ship = new ShipPhysics(device, geometry(2), geometry(0), undefined, Math.PI * 0.75);
  try {
    const rotation = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI * 0.75);
    for (const id of ship.sailBodies) assert.ok(ship.positions[id].distanceTo(ship.initialPositions[id].clone().applyQuaternion(rotation)) < 1e-6);
    await ship.advance({ wind: 0, direction: 135, gust: 0, heave: 0, roll: 0, period: 8 }, 4);
    assert.ok(ship.maxDisplacement < 0.15, `startup canvas displacement ${ship.maxDisplacement} m`);
    assert.ok(ship.rotations[0].angleTo(rotation) < 0.001);
  } finally { ship.destroy(); }
});

gpuTest('ship: AVBD flag droops under gravity with a constrained hoist and responds to wind', async device => {
  const {createFlagGeometry}=await import('../src/ship/flag.ts');
  const flag=createFlagGeometry(new Vector3(0,0,30));
  const ship=new ShipPhysics(device,geometry(2),geometry(0),undefined,0,flag);
  try {
    const skin=ship.skins[2], tip=skin.weights[24][0].body, hoist=skin.weights[0][0].body;
    const start=ship.positions[tip].clone();
    const calm={wind:0,direction:90,gust:0,heave:0,roll:0,period:8};
    for(let i=0;i<60;i++) await ship.advance(calm,4);
    assert.ok(ship.positions[tip].z<start.z-0.3,'unforced free edge must fall');
    assert.ok(ship.positions[hoist].distanceTo(new Vector3(0,0,30))<0.08,'hoist stays on mast');
    const resting=ship.positions[tip].clone();
    for(let i=0;i<60;i++) await ship.advance({...calm,wind:12},4);
    assert.ok(ship.positions[tip].distanceTo(resting)>0.2,'wind drives the simulated free edge');
  } finally {ship.destroy();flag.dispose();}
});
