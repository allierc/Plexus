#!/usr/bin/env python
"""run_turing3d -- Stage 4 (Fig. 3): discrete Turing RD on a STATIC 3D cell aggregate.

Reproduces Okuda et al. 2018 Fig. 3 -- chemical patterns on a 3D cell aggregate without
tissue deformation, cells coloured by activator (red). A closed sphere has no boundary
(SO(3) symmetry), so the Turing spots are cleaner than on the 2D disc. `chi` sets the
spatial scale (paper: chi=0.1 -> ~9-cell spots on a monolayer shell; chi=0.05 -> ~6-cell
spots on a compacted ball). Same plexus2 operators as 2D (turing_ops.py); only the seed
mode and the renderer are 3D.

Each preset is archived to archive/<name>/ : spec.yaml, strip.png, movie.mp4 (rotating),
diag.json.

    python run_turing3d.py
    python run_turing3d.py --montage
"""
from __future__ import annotations
import os, sys, argparse, glob, json, tempfile, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

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

import plexus.operators   # noqa: F401
import turing_ops         # noqa: F401
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
FRAMES, DT = 7000, 1.0
PANEL = 4.6


def presets():
    # Brusselator activator-inhibitor -> ROUND Turing spots (paper Fig. 3). chi is the
    # spatial-scale knob: larger chi -> larger activator domains (paper: chi=0.1 monolayer
    # ~9-cell spots; chi=0.05 compacted ~6-cell spots).
    return [
        # name              mode    N     R    chi  (normalized Laplacian: d_h*chi<0.5 is stable)
        ("shell_big_spots", "shell", 1400, 8.0, 0.80),   # monolayer, larger spots  (~Fig 3a, chi=0.1)
        ("shell_fine_spots","shell", 1400, 8.0, 0.40),   # monolayer, finer spots
        ("ball_chi_small",  "ball",  2400, 8.0, 0.60),   # compacted aggregate      (~Fig 3b, chi=0.05)
    ]


def make_spec(name, mode, N, R, chi, react=None, frames=FRAMES):
    W = 2.6 * R
    r = {"op": "react", "at": "cell", "model": "brusselator",
         "gamma": 0.1, "A": 1.0, "B": 3.0}
    cfg = {
        "general": {"name": f"turing3d_{name}", "seed": 0, "n_frames": frames, "dt": DT,
                    "boundary": "free", "dim": 3, "world": [W, W, W]},
        "sets": {"cell": {"n": N, "state": {
            "chem": {"width": 2, "integration": "first_order", "boundary": "free"},
            "xyz":  {"width": 3, "integration": "none", "boundary": "free"}}}},
        "fields": {},
        "operators": [
            {"op": "seed_aggregate", "at": "cell", "mode": mode, "seed_mode": "noise",
             "radius": R, "k": 6, "a0": 1.0, "h0": 3.0, "noise": 0.03, "before_frame": 1},
            {"op": "graph_diffuse", "at": "cell", "d_a": 0.05, "d_h": 0.5, "chi": chi, "norm": True},
            r,
        ],
        "schedule": ["seed_aggregate", "graph_diffuse", "react"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def render(xyz, act, outdir, name, diag, seconds=12.0, max_frames=180):
    os.makedirs(outdir, exist_ok=True)
    T = act.shape[0]
    vmax = max(0.05, float(np.percentile(act, 99.5)))
    idx = list(range(0, T, max(1, -(-T // max_frames))))
    fps = max(1, round(len(idx) / seconds))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    box = float(np.ptp(np.stack([x, y, z]), axis=1).max()) + 1.0
    # size dots by the PROJECTED areal density (cells overlap in depth): the sphere/ball
    # projects into a disc, so use sqrt(N) for both -- volume spacing (N^1/3) is too big.
    spacing = box / (len(x) ** 0.5)
    s = (PANEL * 72.0 / box * spacing * 0.85) ** 2

    def draw(ax, t, azim):
        ax.clear(); ax.set_facecolor("black")
        ax.scatter(x, y, z, c=act[t], cmap="Reds", vmin=0, vmax=vmax, s=s, edgecolors="none", depthshade=True)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=18, azim=azim)
        pct = int(100 * t / max(T - 1, 1))
        ax.text2D(0.02, 0.98, f"{name}\nt={t} ({pct}%)\nactivator (red)\nhi cells={diag['hi_cells']}",
                  transform=ax.transAxes, color="white", fontsize=7, va="top", family="monospace")

    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
    sfig = plt.figure(figsize=(len(picks) * PANEL, PANEL)); sfig.patch.set_facecolor("black")
    for i, t in enumerate(picks):
        ax = sfig.add_subplot(1, len(picks), i + 1, projection="3d"); ax.set_facecolor("black")
        draw(ax, t, azim=30)
    sfig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.02)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)

    fig = plt.figure(figsize=(PANEL, PANEL)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(projection="3d"); ax.set_facecolor("black"); fig.subplots_adjust(0, 0, 1, 1)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=120):
        for k, t in enumerate(idx):
            draw(ax, t, azim=30 + 360.0 * k / len(idx))       # rotate while the pattern forms
            w.grab_frame()
    plt.close(fig)


def diagnostics(act):
    vT = act[-1]
    return dict(v_max=round(float(vT.max()), 3), v_std=round(float(vT.std()), 3),
                hi_cells=int((vT > 0.2).sum()),
                patterned=bool(vT.std() > 0.04 and vT.max() > 0.2))


def run_all():
    for name, mode, N, R, chi in presets():
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        print(f"[turing3d] {name}: mode={mode} N={N} chi={chi}", flush=True)
        rec = {"mode": mode, "N": N, "chi": chi}
        try:
            sim, cfg = make_spec(name, mode, N, R, chi)
            yaml.safe_dump(cfg, open(os.path.join(odir, "spec.yaml"), "w"), sort_keys=False)
            _, out = engine_run(sim, device="cpu")
            act = out["sets"]["cell"]["state"]["chem"][..., 0]
            xyz = out["sets"]["cell"]["state"]["xyz"][0]
            diag = diagnostics(act); rec.update(diag)
            render(xyz, act, odir, name, diag)
            print(f"           -> patterned={diag['patterned']} v_std={diag['v_std']} hi={diag['hi_cells']}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def montage():
    from PIL import Image
    names = [p[0] for p in presets()]
    files = [os.path.join(OUT, n, "strip.png") for n in names]
    strips = [(n, Image.open(f).convert("RGB")) for n, f in zip(names, files) if os.path.exists(f)]
    if strips:
        w = max(i.width for _, i in strips); h = sum(i.height for _, i in strips)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, im in strips:
            sheet.paste(im, (0, y)); y += im.height
        sheet.save(os.path.join(OUT, "_montage_turing3d.png"))
    print(f"[turing3d] montage -> {OUT}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    montage() if a.montage else (run_all(), montage())


if __name__ == "__main__":
    main()
