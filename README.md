# Windward — AVBD Ship Demo

Interactive sailing-ship experiment with GPU rope and cloth physics, FFT ocean waves, hull-contact foam and spray, changing daylight, stars, rain and lightning.

## Run

Use Node 24+ and pnpm 10.

```sh
pnpm install
pnpm dev
pnpm check
```

Open http://localhost:5319. Production: `pnpm build` and `pnpm preview`.
A WebGPU-capable browser/device is required. GPU tests explicitly skip when no adapter is available.

## Controls

Drag the scene to orbit; scroll to zoom. Choose sea conditions and lighting, or drag the camera-aligned wind compass. Debug contains steering, yard trim, lightning, physics and rendering controls. Audio starts muted.

Rendering defaults to **Auto**: a bounded pixel budget, reduced reflection resolution, and a downgrade to Low when sustained frame times exceed 35 ms. Low disables volumetric clouds/light shafts and lowers the resolution budget. Balanced and High can be selected manually. Physics stays unchanged between quality tiers. This is a demanding WebGPU demo; Low is not a guarantee of smooth performance on every phone or older GPU.

## Implementation and scope

Ship-specific code is in `src/ship`. The local AVBD solver modules are vendored from [three-avbd](https://github.com/sbobyn/three-avbd), under the included MIT license, to preserve the tested internal API (engine revision `b3675dea83c78aba9644b059975f48285bf02a47`, plus the ship demo implementation). The flag, sails and rigging use AVBD; hull navigation/buoyancy and water effects are a real-time approximation, not a full fluid simulation. Cloth self-collision is not enabled.

The mirror blends to the sky cubemap outside valid projected reflection coordinates, avoiding stretched edge texels when orbiting at night. Sky lighting captures are throttled during preset transitions; the final state is captured after settling.

## Deployment

Import this repository into Vercel (Vite preset), or run `vercel --prod`. No runtime secrets or external asset service is required. `/ship.html` redirects to the root demo.

## Credits

Ship: [Dutch Ship Medium, Poly Haven](https://polyhaven.com/a/dutch_ship_medium), CC0. Authors and asset hashes are in `public/ship/README.md` and `public/ship/manifest.json`. See `THIRD_PARTY_NOTICES.md` for all third-party notices. Engine source: [three-avbd](https://github.com/sbobyn/three-avbd).
