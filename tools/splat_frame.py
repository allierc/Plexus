#!/usr/bin/env python
"""Render ONE frame of a spec with the GPU splat, and write it as a PNG.

    python tools/splat_frame.py --spec si_material/si_bench_1b --frame 0 --out render.png

WHY. At a billion particles a movie costs an hour and a spec mistake costs all of it. This builds
the hierarchy, advances `--frame` frames, renders once and exits -- so framing, colour and the
bounding box can be checked for the price of the build (~100 s at 1 B) instead of the run.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="<type>/<name>")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="render.png")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--res", type=int, default=0, help="override plotting.splat_res")
    a = ap.parse_args()

    import imageio.v3 as iio
    import torch

    import plexus.operators                                          # noqa: F401
    from plexus.engine import run
    from plexus.paths import graphs_data_path
    from plexus.schema import load
    from plexus.splat_movie import SplatMovie

    torch.cuda.set_device(a.device)
    typ, name = a.spec.split("/", 1)
    sim = load(os.path.join(ROOT, "config", typ, name + ".yaml"))
    sim.n_frames = max(1, int(a.frame))
    sim.save_data = False
    style = dict(sim.plotting or {})
    if a.res:
        style["splat_res"] = int(a.res)
    out_dir = graphs_data_path(typ, sim.name)
    os.makedirs(out_dir, exist_ok=True)
    out_png = a.out if os.path.isabs(a.out) else os.path.join(out_dir, a.out)

    sm = SplatMovie(out=os.path.join(out_dir, "_frame.mp4"), world=list(sim.world_size),
                    n_frames=sim.n_frames, up=int(style.get("up_axis", 2)), name=sim.name,
                    sim=sim, style=style, max_frames=1, stills=0)
    got = {}

    def hook(H, tick):
        if tick == a.frame or (a.frame == 0 and tick == 0):
            lvl = H.level("mpm_particle")
            pos = lvl.get("pos")
            got["img"] = sm._image(pos, sm._colours(H, lvl, pos.device))
            got["n"] = int(lvl.n)
            got["lo"] = pos.amin(0).tolist()
            got["hi"] = pos.amax(0).tolist()

    with contextlib.redirect_stdout(io.StringIO()):
        run(sim, out_path=None, device=a.device, progress=False, on_frame=hook)
    if "img" not in got:
        raise SystemExit(f"frame {a.frame} never reached")
    iio.imwrite(out_png, got["img"])
    print(f"\n  {sim.name}  frame {a.frame}   {got['n']:,} particles")
    print(f"  world  {[round(float(w), 3) for w in sim.world_size]}")
    print(f"  extent lo {[round(v, 4) for v in got['lo']]}  hi {[round(v, 4) for v in got['hi']]}")
    inside = all(l >= -1e-6 for l in got["lo"]) and all(
        h <= float(w) + 1e-6 for h, w in zip(got["hi"], sim.world_size))
    print(f"  every particle inside the box: {'YES' if inside else 'NO'}")
    print(f"  wrote {out_png}\n")


if __name__ == "__main__":
    main()
