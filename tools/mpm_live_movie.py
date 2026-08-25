#!/usr/bin/env python
"""Render an MPM run to mp4 LIVE, straight off the GPU, without ever writing a trajectory.

WHY THIS EXISTS. At 100 M particles a single recorded frame is 100e6 x 3 x 4 B = 1.2 GB, so the
normal route -- `Plexus_Main.py -o generate` writes `trajectory.npz`, then `tools/cell_panels.py`
reads it -- would need 72 GB on disk for a 60-frame clip before any renderer saw a pixel. It is not
that the renderer is too slow; it is that the intermediate does not fit. So the picture is taken
while the state is still in GPU memory and only the picture is kept.

TWO SUBSAMPLINGS, AND THEY ARE DIFFERENT THINGS.
  * `--render-n` draws a fixed random subset of the particles. The SIMULATION still runs all of
    them -- this only bounds what VTK is asked to draw, because a point cloud of 100 M vertices is
    ~2.4 GB of VTK memory and tens of seconds a frame. A uniform random subset of a uniform-density
    fluid looks like the fluid; it is a sampling of the picture, not of the physics.
  * `--stride` renders every k-th simulation frame. This is a frame rate choice.
Both are printed on the movie, because a subsampled render that does not say so is a lie about how
many particles ran.

THIS IS NOT THE BENCHMARK AND MUST NEVER BECOME IT. `tools/mpm_bench.py` renders nothing and its
ms/frame figures are the simulation alone; the `ms/frame` stamped on this movie INCLUDES the
per-frame device->host copy and the VTK render, and is therefore slower by construction. Quote the
bench for throughput, this for pictures.

    python tools/mpm_live_movie.py --spec config/material/material_3d_water_bench_100m.yaml \
        --frames 90 --device cuda:1 --out graphs_data/cell/mpm_100m/movie.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

FLAT = dict(render_points_as_spheres=True, lighting=False, ambient=1.0, diffuse=0.0, specular=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--render-n", type=int, default=400_000,
                    help="particles DRAWN per frame; the run still simulates all of them")
    ap.add_argument("--stride", type=int, default=1, help="render every k-th simulation frame")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--px", type=int, default=1280)
    ap.add_argument("--dot", type=float, default=1.4)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-58.0)
    a = ap.parse_args()

    import numpy as np
    import torch
    import yaml

    import plexus.operators  # noqa: F401
    import plexus.operators.mpm_warp  # noqa: F401
    from plexus.render_vtk import offscreen
    from plexus.schema import load
    from plexus import engine as E

    offscreen()
    import pyvista as pv
    pv.OFF_SCREEN = True

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    spec_path = a.spec if os.path.isabs(a.spec) else os.path.join(root, a.spec)
    spec = yaml.safe_load(open(spec_path))
    spec["general"]["n_frames"] = a.frames
    spec["general"]["record_cap"] = 2                 # the recorder is switched OFF in all but name
    name = spec["general"]["name"]
    world = spec["general"].get("world", [1.0, 1.0, 1.0])
    up = int((spec.get("plotting") or {}).get("up_axis", 2))

    import tempfile
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(spec, f); f.close()
    sim = load(f.name); os.unlink(f.name)

    out = a.out or os.path.join(root, "graphs_data", "cell", name, "movie.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    px = int(a.px) // 16 * 16                          # ffmpeg's macro_block_size, see cell_panels
    p = pv.Plotter(off_screen=True, window_size=(px, px), border=False)
    p.set_background("black")
    p.enable_anti_aliasing("msaa", multi_samples=8)

    lo = np.zeros(3); hi = np.array([float(w) for w in world])
    p.add_mesh(pv.Box((lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])).extract_all_edges(),
               color="#4a4a4a", line_width=1.0, lighting=False)
    centre, radius = 0.5 * (lo + hi), float((hi - lo).max()) * 0.55
    e, az = np.radians(a.elev), np.radians(a.azim)
    ax_h = [i for i in range(3) if i != up]
    d = np.zeros(3)
    d[ax_h[0]], d[ax_h[1]], d[up] = np.cos(e) * np.cos(az), np.cos(e) * np.sin(az), np.sin(e)
    p.camera.position = tuple(centre + d * radius * 6.0)
    p.camera.focal_point = tuple(centre)
    u = np.zeros(3); u[up] = 1.0
    p.camera.up = tuple(u)
    p.camera.parallel_projection = True
    p.camera.parallel_scale = radius * 1.45

    p.open_movie(out, framerate=a.fps, quality=8)

    st = {"cloud": None, "idx": None, "n": 0, "drawn": 0, "t0": None, "rendered": 0}

    def on_frame(H, tick):
        lvl = H.level("mpm_particle")
        if st["cloud"] is None:
            st["n"] = int(lvl.n)
            # SEEDED, and drawn ONCE. A subset that changes between frames makes the fluid boil:
            # every dot would be a different particle, so nothing would appear to move. Fixing the
            # subset is what makes the render a movie of the same material rather than of noise.
            gen = torch.Generator(device="cpu").manual_seed(0)
            k = min(a.render_n, st["n"])
            st["idx"] = torch.randperm(st["n"], generator=gen)[:k].to(lvl.state.device)
            st["drawn"] = k
            pos = lvl.get("pos")[st["idx"]].detach().cpu().numpy().astype(np.float64)
            st["cloud"] = pv.PolyData(pos)
            # coloured by HEIGHT ALONG THE UP AXIS at t=0, carried with the particle: the colour is
            # then a material label and the mixing is visible, which a colour recomputed per frame
            # would hide entirely.
            h = (pos[:, up] - lo[up]) / max(hi[up] - lo[up], 1e-9)
            rgb = np.stack([np.clip(1.4 - 1.6 * h, 0, 1), np.clip(0.35 + 0.5 * h, 0, 1),
                            np.clip(0.25 + 1.1 * h, 0, 1)], 1)
            st["cloud"]["rgb"] = (rgb * 255).astype(np.uint8)
            p.add_mesh(st["cloud"], scalars="rgb", rgb=True, **FLAT, point_size=a.dot)
            st["t0"] = time.perf_counter()
            return
        if tick % a.stride:
            return
        st["cloud"].points = lvl.get("pos")[st["idx"]].detach().cpu().numpy().astype(np.float64)
        el = time.perf_counter() - st["t0"]
        p.add_text(f"{name}\n{st['n']:,} particles simulated, {st['drawn']:,} drawn\n"
                   f"frame {tick}/{a.frames}   {el / max(tick, 1) * 1000:.0f} ms/frame",
                   position="upper_left", font_size=11, color="white", name="hdr")
        p.write_frame()
        st["rendered"] += 1

    torch.cuda.init(); torch.zeros(1, device=a.device)   # the stats API needs a live context first
    torch.cuda.reset_peak_memory_stats(a.device)
    E.run(sim, out_path=None, device=a.device, on_frame=on_frame, progress=True)
    p.close()
    print(f"\n  {out}\n  {st['n']:,} particles simulated, {st['drawn']:,} drawn, "
          f"{st['rendered']} frames @ {a.fps} fps, "
          f"peak {torch.cuda.max_memory_allocated(a.device) / 2 ** 30:.2f} GiB")


if __name__ == "__main__":
    main()
