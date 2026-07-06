#!/usr/bin/env python
"""run_vesicle -- cell_polarity case 2: 3D lumen vesicle -> monolayer<->multilayer stratification.

A hollow spherical monolayer of cells around a lumen; sweeping the cortical-tension knob (the shell
radius) drives the SimuCell3D Fig. 3 transition: a large shell holds all cells in one MONOLAYER,
a shrunk shell (high tension) can't tile them, so they STRATIFY into a multilayer. 2x1 panel:
a 3D vesicle (cells coloured by radius) + an equatorial CROSS-SECTION showing the layer thickness.

    python run_vesicle.py --device cuda:0
    python run_vesicle.py --montage
"""
from __future__ import annotations
import os, sys, argparse, glob, json, tempfile, math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators           # noqa: F401  radius_graph
import vesicle_ops                 # noqa: F401  vesicle_seed + vesicle_mechanics
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive_vesicle")
N = 300
WORLD = [16.0, 16.0, 16.0]
CEN = np.array([8.0, 8.0, 8.0])


def presets():
    """The cortical-tension knob = shell radius: large -> monolayer, shrunk -> multilayer."""
    return [
        ("monolayer", dict(shell=6.5)),   # roomy shell -> single layer
        ("mid",       dict(shell=4.2)),    # near the transition
        ("multilayer", dict(shell=3.0)),   # tension shrinks the shell -> stratifies
        ("strong",    dict(shell=2.3)),    # strong tension -> thick multilayer
    ]


def make_sim(p, frames=500, dt=1.0):
    cfg = {
        "general": {"name": "vesicle", "seed": 1, "n_frames": frames, "dt": dt,
                    "boundary": "free", "dim": 3, "world": WORLD},
        "sets": {"cell": {"n": N, "spawn": "random", "types": {"a": {"fraction": 1.0}}}},
        "fields": {},
        "operators": [
            {"op": "vesicle_seed", "at": "cell", "radius": p["shell"], "before_frame": 1},
            {"op": "radius_graph", "at": "cell", "radius": 1.4},
            {"op": "vesicle_mechanics", "at": "cell", "shell": p["shell"], "adhesion": 1.0,
             "sigma": 0.9, "r_adh": 1.4, "k_rep": 40.0, "k_r": 0.6, "mu": 0.02, "noise": 0.01},
        ],
        "schedule": ["vesicle_seed", "radius_graph", "vesicle_mechanics"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def diagnostics(pos):
    xy = pos[-1] - CEN
    r = np.linalg.norm(xy, axis=1)
    thickness = float(np.percentile(r, 95) - np.percentile(r, 5))     # radial spread of the shell
    layers = round(thickness / 0.9 + 1)                               # ~cells across the shell
    return dict(mean_radius=round(float(r.mean()), 2), thickness=round(thickness, 2),
                layers=int(layers), state=("monolayer" if thickness < 1.1 else "multilayer"))


def render(pos, outdir, name, diag, seconds=14.0, max_frames=120):
    os.makedirs(outdir, exist_ok=True)
    T = pos.shape[0]
    idx = list(range(0, T, max(1, -(-T // max_frames))))
    fps = max(1, round(len(idx) / seconds))

    def draw(fig, t, azim):
        fig.clf(); fig.patch.set_facecolor("black")
        p = pos[t] - CEN
        r = np.linalg.norm(p, axis=1)
        col = plt.cm.coolwarm(np.clip((r - 3) / 4, 0, 1))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d"); ax1.set_facecolor("black")
        ax1.scatter(p[:, 0], p[:, 1], p[:, 2], s=90, c=col, alpha=0.55, depthshade=True, edgecolors="none")
        lim = 7; ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim); ax1.set_zlim(-lim, lim)
        ax1.set_box_aspect((1, 1, 1)); ax1.set_axis_off(); ax1.view_init(elev=18, azim=azim)
        ax1.set_title("vesicle", color="white", fontsize=9)
        # equatorial cross-section: cells near z=0
        ax2 = fig.add_subplot(1, 2, 2); ax2.set_facecolor("black")
        sl = np.abs(p[:, 2]) < 0.9
        ax2.scatter(p[sl, 0], p[sl, 1], s=140, c=col[sl], alpha=0.9, edgecolors="white", linewidths=0.5)
        ax2.add_patch(plt.Circle((0, 0), 0.6, color="black"))     # lumen marker
        ax2.set_xlim(-7, 7); ax2.set_ylim(-7, 7); ax2.set_aspect("equal"); ax2.axis("off")
        info = (f"{name}\nshell R={diag.get('shell','?')}\nthickness={diag['thickness']}"
                f"\nlayers~{diag['layers']}\n{diag['state']}")
        ax2.text(0.02, 0.98, info, transform=ax2.transAxes, color="white", fontsize=6,
                 va="top", ha="left", family="monospace")

    fig = plt.figure(figsize=(8.6, 4.5)); fig.patch.set_facecolor("black")
    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
    sfig = plt.figure(figsize=(4 * 2.3, 2.4)); sfig.patch.set_facecolor("black")
    for k, tt in enumerate(picks):
        sax = sfig.add_subplot(1, 4, k + 1); sax.set_facecolor("black")
        p = pos[tt] - CEN; r = np.linalg.norm(p, axis=1); sl = np.abs(p[:, 2]) < 0.9
        sax.scatter(p[sl, 0], p[sl, 1], s=60, c=plt.cm.coolwarm(np.clip((r[sl] - 3) / 4, 0, 1)),
                    alpha=0.9, edgecolors="white", linewidths=0.4)
        sax.set_xlim(-7, 7); sax.set_ylim(-7, 7); sax.set_aspect("equal"); sax.axis("off")
        sax.set_title(f"{int(100*tt/max(T-1,1))}%", color="white", fontsize=9)
    sfig.subplots_adjust(0.01, 0.01, 0.99, 0.9, wspace=0.05)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=110, facecolor="black"); plt.close(sfig)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=100):
        for j, t in enumerate(idx):
            draw(fig, t, 20 + 90 * j / max(len(idx) - 1, 1)); w.grab_frame()
    plt.close(fig)


def run_share(rank, nproc, device):
    P = presets()[rank::nproc]
    print(f"[rank {rank}] {len(P)} presets on {device}", flush=True)
    for i, (name, p) in enumerate(P):
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        print(f"[rank {rank}] ({i+1}/{len(P)}) {name}", flush=True)
        sim, cfg = make_sim(p)
        with open(os.path.join(odir, "spec.yaml"), "w") as sf:
            yaml.safe_dump(cfg, sf, sort_keys=False)
        _, out = engine_run(sim, device=device)
        pos = out["sets"]["cell"]["pos"]
        diag = diagnostics(pos); diag["shell"] = p["shell"]
        render(pos, odir, name, diag)
        json.dump(diag, open(os.path.join(odir, "diag.json"), "w"), indent=1)
    print(f"[rank {rank}] done", flush=True)


def montage():
    from PIL import Image
    strips = sorted(glob.glob(os.path.join(OUT, "*", "strip.png")))
    ims = [(os.path.basename(os.path.dirname(f)), Image.open(f).convert("RGB")) for f in strips]
    if ims:
        w = max(i.width for _, i in ims); h = sum(i.height for _, i in ims)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, im in ims:
            sheet.paste(im, (0, y)); y += im.height
        sheet.save(os.path.join(OUT, "_montage.png"))
    with open(os.path.join(OUT, "_summary.md"), "w") as fh:
        fh.write("# cell_polarity case 2 -- lumen vesicle monolayer<->multilayer (SimuCell3D Fig.3)\n\n")
        fh.write("| preset | shell R | thickness | layers | state |\n|--|--|--|--|--|\n")
        for f in sorted(glob.glob(os.path.join(OUT, "*", "diag.json"))):
            d = json.load(open(f)); name = os.path.basename(os.path.dirname(f))
            fh.write(f"| {name} | {d.get('shell','?')} | {d.get('thickness','?')} "
                     f"| {d.get('layers','?')} | {d.get('state','?')} |\n")
    print(f"montage: {len(ims)} -> {OUT}/_montage.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    montage() if a.montage else run_share(a.rank, a.nproc, a.device)


if __name__ == "__main__":
    main()
