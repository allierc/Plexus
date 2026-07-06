#!/usr/bin/env python
"""galaxy_sweep -- search N-body galaxy IC/parameter space for good SPIRAL disks.

Runs the strict-Plexus N-body galaxy (galaxy_ops: nbody_gravity + disk_ic) over a grid
of physical levers, on BOTH GPUs, with FIXED-FRAME rendering (the free-boundary auto-zoom
was hiding the disk) and per-config diagnostics -> the "understanding":

  retain  = fraction of stars still within the view (few ejections = stable disk)
  A2      = m=2 Fourier amplitude of the star angles in a mid annulus (bar / two-arm / spiral
            strength; ~0 = featureless axisymmetric disk, high = strong bar/arms)
  r_half  = half-mass radius (disk spread; collapse vs puff-up)

Levers swept: spin (circular-velocity fraction), m_bh (central black hole), disk mass
(self-gravity strength ~ Toomre Q), velocity jitter (warm disk = arm-forming vs
fragmenting), softening. Each candidate -> archive/galaxy_sweep/<name>/ (fig + movie),
plus a montage + a scored summary table.

    python galaxy_sweep.py --rank 0 --nproc 2 --device cuda:0 &
    python galaxy_sweep.py --rank 1 --nproc 2 --device cuda:1 &
    wait; python galaxy_sweep.py --montage
"""
from __future__ import annotations
import os, sys, argparse, glob, json, tempfile, traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators           # noqa: F401
import galaxy_ops                 # noqa: F401  nbody_gravity + disk_ic
import plexus.schema as S
from plexus.engine import run as engine_run

HERE = os.path.dirname(os.path.abspath(__file__))
FINE = False                                       # set by --fine: zoom near the coarse winner
OUT = os.path.join(HERE, "archive", "galaxy_sweep")
WORLD = 8.0                                        # square world; galaxy centred at (4,4)


def configs():
    """Sweep the TOOMRE stability transition, where smooth spiral arms live between a
    cold heavy disk (fragments into clumps, Q<1) and a hot light one (featureless, Q>2).
    Levers: disk self-gravity (mass), velocity dispersion (jitter ~ Q), central bulge
    (m_bh, sets the shear that swing-amplifies arms), and the rotation fraction (spin).

    --fine ZOOMS around the coarse-sweep winner (small BH + heavy cold disk, bh0.5_m2_j0.06)
    to pin the strongest-spiral corner more precisely."""
    C = []
    N, disc_R, dt, nf = 25000, 1.5, 0.004, 10000    # 4x frames: spiral evolves further (movie stays ~40s)
    if FINE:
        bhs, masses, jitters, spins = (0.25, 0.5, 0.75), (1.5, 2.0, 2.5), (0.04, 0.06, 0.09), (0.88, 0.92)
    else:
        bhs, masses, jitters, spins = (0.5, 2.0, 5.0), (0.5, 1.0, 2.0), (0.06, 0.14, 0.28), (0.9, 1.0)
    for m_bh in bhs:                             # central mass -> rotation shear
        for mass_tot in masses:                 # disk self-gravity strength
            for jitter in jitters:              # velocity dispersion (Toomre Q)
                for spin in spins:
                    name = f"bh{m_bh:g}_m{mass_tot:g}_j{jitter:g}_s{spin:g}"
                    C.append((name, dict(N=N, disc_R=disc_R, dt=dt, nframes=nf,
                                         spin=spin, m_bh=m_bh, mass=mass_tot / N,
                                         jitter=jitter, soft=0.08)))
    return C


def make_sim(p):
    cfg = {
        "general": {"name": "g", "seed": 2, "n_frames": p["nframes"], "dt": p["dt"],
                    "boundary": "free", "world": [WORLD, WORLD]},
        "sets": {"star": {"n": p["N"], "types": {"s": {"fraction": 1.0, "mass": p["mass"]}}}},
        "fields": {},
        "operators": [
            {"op": "disk_ic", "at": "star", "G": 1.0, "softening": p["soft"],
             "disc_radius": p["disc_R"], "spin": p["spin"], "m_bh": p["m_bh"],
             "vel_jitter": p["jitter"], "before_frame": 1},
            {"op": "nbody_gravity", "at": "star", "G": 1.0, "softening": p["soft"],
             "compile": True},          # fused force -> feasible at N=25k (0.01 GB, ~23x faster)
        ],
        "schedule": ["disk_ic", "nbody_gravity"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f); path = f.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def diagnostics(pos_last, center, view):
    d = pos_last - center; r = np.sqrt((d ** 2).sum(-1))
    retain = float((r < view).mean())
    r_half = float(np.median(r[r < view])) if retain > 0 else float("nan")
    ann = (r > 0.15 * view) & (r < 0.7 * view)
    if ann.sum() > 10:
        th = np.arctan2(d[ann, 1], d[ann, 0])
        A2 = float(np.abs(np.exp(2j * th).mean()))
    else:
        A2 = 0.0
    return dict(retain=round(retain, 3), A2=round(A2, 3), r_half=round(r_half, 3))


def render(pos, outdir, name, view, seconds=40.0, max_frames=1000):
    """Matplotlib movie -- the profiled pipeline bottleneck (~38 ms/frame). We DECIMATE the
    recorded trajectory to <=`max_frames` (so render cost is bounded by movie length, not
    sim length) and then set fps = frames/`seconds` so the movie is ALWAYS ~`seconds` long."""
    os.makedirs(outdir, exist_ok=True)
    c = np.array([WORLD / 2, WORLD / 2])
    lo, hi = c - view, c + view
    T = pos.shape[0]
    stride = max(1, -(-T // max_frames))               # ceil(T/max_frames) -> <= max_frames frames
    idx = list(range(0, T, stride))
    fps = max(1, round(len(idx) / seconds))            # lock duration to ~`seconds`

    def frame(ax, xy):
        ax.clear(); ax.set_facecolor("black")
        ax.scatter(xy[:, 0], xy[:, 1], s=1.0, c="#dbe6ff", linewidths=0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1)
    frame(ax, pos[-1]); fig.savefig(os.path.join(outdir, "fig_final.png"), dpi=110, facecolor="black")
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=90):
        for t in idx:
            frame(ax, pos[t]); w.grab_frame()
    plt.close(fig)


def run_share(rank, nproc, device):
    cfgs = configs()[rank::nproc]
    print(f"[rank {rank}] {len(cfgs)} configs on {device}", flush=True)
    for i, (name, p) in enumerate(cfgs):
        odir = os.path.join(OUT, name)
        if os.path.exists(os.path.join(odir, "fig_final.png")):
            print(f"[rank {rank}] skip {name}", flush=True); continue
        try:
            print(f"[rank {rank}] ({i+1}/{len(cfgs)}) {name}", flush=True)
            sim, cfg = make_sim(p)
            os.makedirs(odir, exist_ok=True)
            with open(os.path.join(odir, "spec.yaml"), "w") as sf:      # self-documenting archive
                yaml.safe_dump(cfg, sf, sort_keys=False)
            _, out = engine_run(sim, device=device)
            pos = out["sets"]["star"]["pos"]                    # [T, N, 2]
            view = 2.4 * p["disc_R"]
            render(pos, odir, name, view)
            diag = diagnostics(pos[-1], np.array([WORLD / 2, WORLD / 2]), view)
            diag.update(p)
            json.dump(diag, open(os.path.join(odir, "diag.json"), "w"), indent=1)
        except Exception:
            print(f"[rank {rank}] {name} FAILED\n{traceback.format_exc()}", flush=True)
    print(f"[rank {rank}] done", flush=True)


def montage():
    from PIL import Image, ImageDraw
    figs = sorted(glob.glob(os.path.join(OUT, "*", "fig_final.png")))
    rows = []
    for f in figs:
        d = os.path.dirname(f); name = os.path.basename(d)
        try:
            diag = json.load(open(os.path.join(d, "diag.json")))
        except Exception:
            diag = {}
        rows.append((name, diag, f))
    # rank by a "good spiral" score: structured (A2) AND retained (not dispersed/collapsed)
    def score(diag):
        return diag.get("A2", 0) * min(diag.get("retain", 0) / 0.8, 1.0)
    rows.sort(key=lambda r: -score(r[1]))
    # montage (sorted best-first)
    thumbs = []
    for name, diag, f in rows:
        im = Image.open(f).convert("RGB").resize((240, 240))
        dd = ImageDraw.Draw(im)
        dd.text((4, 4), name, fill=(255, 255, 255))
        dd.text((4, 224), f"A2={diag.get('A2','?')} ret={diag.get('retain','?')}", fill=(150, 220, 150))
        thumbs.append(im)
    cols = 6; R = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, R * 240), (0, 0, 0))
    for k, t in enumerate(thumbs):
        sheet.paste(t, ((k % cols) * 240, (k // cols) * 240))
    sheet.save(os.path.join(OUT, "_montage.png"))
    # understanding table
    with open(os.path.join(OUT, "_summary.md"), "w") as fh:
        fh.write("# galaxy sweep — ranked by spiral score (A2 · retain)\n\n")
        fh.write("| rank | config | A2 (arm/bar) | retain | r_half | score |\n|--|--|--|--|--|--|\n")
        for i, (name, diag, _) in enumerate(rows):
            fh.write(f"| {i+1} | {name} | {diag.get('A2','?')} | {diag.get('retain','?')} "
                     f"| {diag.get('r_half','?')} | {score(diag):.3f} |\n")
    print(f"montage: {len(thumbs)} candidates -> {OUT}/_montage.png + _summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--montage", action="store_true")
    ap.add_argument("--fine", action="store_true")
    a = ap.parse_args()
    global FINE, OUT
    if a.fine:
        FINE = True; OUT = os.path.join(HERE, "archive", "galaxy_sweep_fine")
    montage() if a.montage else run_share(a.rank, a.nproc, a.device)


if __name__ == "__main__":
    main()
