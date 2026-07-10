#!/usr/bin/env python
"""run_attractors -- four strange attractors as strict-Plexus 3D flows, rendered to glowing mp4s.

Loads each `specs/<name>.yaml` (a genuine Plexus spec: a `cloud` set seeded in a tiny cube +
the registered `attractor_flow` operator), runs it through the Plexus ENGINE (forward-Euler
integration of dx/dt = f(x)), then renders the recorded [T,N,3] point cloud with the
from-scratch additive-glow 3D renderer (`viz3d`) into archive/<name>/:

    movie.mp4   orbiting-camera neon 3D movie
    strip.png   5-stage development strip (seed cube -> unfolded attractor)
    fig_final.png
    spec.yaml   the exact spec that produced it
    diag.json   cloud observables (extent, fractal-ish occupancy, chaos check)

Split across both GPUs, then montage:

    python run_attractors.py --rank 0 --nproc 2 --device cuda:0 &
    python run_attractors.py --rank 1 --nproc 2 --device cuda:1 &
    wait; python run_attractors.py --montage

Run with:  /workspace/.conda_envs/neural-graph-linux/bin/python run_attractors.py ...
"""
from __future__ import annotations
import os, sys, argparse, glob, json, shutil, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np

import plexus.operators          # noqa: F401  self-register the core operator library
import attractors_ops            # noqa: F401  self-register attractor_flow
from plexus.schema import load as load_spec
from plexus.engine import run as engine_run
import viz3d

OUT = os.path.join(HERE, "archive")
ATTRACTORS = ["halvorsen", "lorenz", "aizawa", "sprott_b",
              "thomas", "rossler", "dadras", "chen", "chua", "rabinovich_fabrikant"]


def _sanitize(pos):
    """Guard the render stats against a rare basin escapee: drop non-finite frames-points to
    the cloud median and clip absurd magnitudes, so one runaway point can't poison autoscale."""
    pos = np.asarray(pos, np.float32)
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(pos.reshape(-1, 3), axis=0)
    big = np.abs(pos - med[None, None]) > 1e4
    if big.any():
        pos = np.where(big, med[None, None], pos)
    return pos


def diagnostics(pos):
    """Chaos / attractor observables from the cloud. `spread_growth` = final vs initial cloud
    radius (chaos stretches a tiny seed by orders of magnitude); `occupancy` = fraction of a
    coarse 3D grid the final cloud touches (a fractal fills a set of measure zero -> low but
    nonzero); extent = final bounding box."""
    p0, pT = pos[0], pos[-1]
    r0 = float(np.linalg.norm(p0 - p0.mean(0), axis=1).mean() + 1e-9)
    rT = float(np.linalg.norm(pT - pT.mean(0), axis=1).mean())
    lo, hi = pT.min(0), pT.max(0)
    # coarse 32^3 occupancy of the final cloud (a crude box-counting proxy)
    g = 32
    q = np.floor((pT - lo) / (hi - lo + 1e-9) * (g - 1e-6)).astype(int)
    occ = len(set(map(tuple, q))) / g ** 3
    return dict(spread_growth=round(rT / r0, 1),
                extent=[round(float(e), 2) for e in (hi - lo)],
                occupancy_32=round(occ, 4),
                n_points=int(pT.shape[0]))


def run_one(name, device):
    odir = os.path.join(OUT, name)
    if os.path.exists(os.path.join(odir, "movie.mp4")):
        print(f"[{name}] skip (movie exists)", flush=True)
        return
    spec_path = os.path.join(HERE, "specs", f"{name}.yaml")
    sim = load_spec(spec_path)
    os.makedirs(odir, exist_ok=True)
    shutil.copy2(spec_path, os.path.join(odir, "spec.yaml"))

    print(f"[{name}] engine.run on {device} ...", flush=True)
    _, out = engine_run(sim, device=device, progress=True)
    pos = _sanitize(out["sets"]["cloud"]["pos"])          # [T, N, 3]

    style = dict(sim.plotting or {})
    color = style.pop("color", [1.0, 1.0, 1.0])
    viz3d.render(pos, odir, name, color, style=style, device=device)

    diag = diagnostics(pos)
    json.dump(diag, open(os.path.join(odir, "diag.json"), "w"), indent=1)
    print(f"[{name}] done  {diag}", flush=True)


def run_share(rank, nproc, device):
    mine = ATTRACTORS[rank::nproc]
    print(f"[rank {rank}] {mine} on {device}", flush=True)
    for name in mine:
        try:
            run_one(name, device)
        except Exception:
            print(f"[rank {rank}] {name} FAILED\n{traceback.format_exc()}", flush=True)
    print(f"[rank {rank}] done", flush=True)


def montage():
    from PIL import Image
    strips = [os.path.join(OUT, n, "strip.png") for n in ATTRACTORS
              if os.path.exists(os.path.join(OUT, n, "strip.png"))]
    ims = [Image.open(f).convert("RGB") for f in strips]
    if ims:
        w = max(i.width for i in ims); h = sum(i.height for i in ims)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for i in ims:
            sheet.paste(i, (0, y)); y += i.height
        sheet.save(os.path.join(OUT, "_montage.png"))
    done = [n for n in ATTRACTORS if os.path.exists(os.path.join(OUT, n, "diag.json"))]
    with open(os.path.join(OUT, "_summary.md"), "w") as fh:
        fh.write(f"# {len(done)} strange attractors — Plexus `attractor_flow` (dx/dt = f(x))\n\n")
        fh.write("| attractor | spread_growth | extent (x,y,z) | occupancy_32 |\n")
        fh.write("|--|--|--|--|\n")
        for n in done:
            d = json.load(open(os.path.join(OUT, n, "diag.json")))
            fh.write(f"| {n} | {d.get('spread_growth','?')}× | {d.get('extent','?')} "
                     f"| {d.get('occupancy_32','?')} |\n")
        fh.write("\n_rabinovich_fabrikant leaks ~20% of its cloud to infinity (its chaotic "
                 "attractor coexists with escaping orbits), so its extent/spread stats reflect "
                 "the escapees; the movie frames the bounded urchin core (view_quantile)._\n")
    print(f"montage: {len(ims)} strips -> {OUT}/_montage.png + _summary.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default=None, help="run a single attractor by name")
    ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    if a.montage:
        montage()
    elif a.only:
        run_one(a.only, a.device)
    else:
        run_share(a.rank, a.nproc, a.device)


if __name__ == "__main__":
    main()
