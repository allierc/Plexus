#!/usr/bin/env python
"""run_embryo_cell_sorting -- differential-adhesion cell sorting as a strict-Plexus sim.

An initially MIXED aggregate of cell types (contact graph = `radius_graph`, overdamped motion
from the `differential_adhesion` operator) sorts over time: like cells cluster, and the most
cohesive type is engulfed by the less cohesive (Steinberg). Renders cells coloured by type,
moving, on black. Sweeps the adhesion matrix (differential vs equal control; 2-type sort vs
3-type engulfment hierarchy).

    python run_embryo_cell_sorting.py --device cuda:1
    python run_embryo_cell_sorting.py --montage
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

import plexus.operators           # noqa: F401  radius_graph
import embryo_cell_sorting_ops     # noqa: F401  differential_adhesion
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
RED = [0.90, 0.30, 0.22]; BLUE = [0.25, 0.48, 0.95]; YELLOW = [0.95, 0.85, 0.20]


def presets():
    """Levers: the adhesion matrix. Differential (like>unlike) -> sorting; equal -> no sorting
    (control). A 3-type ascending hierarchy (blue<red<yellow self-adhesion) -> engulfment."""
    two = [("A", 0.5, RED), ("B", 0.5, BLUE)]
    three = [("blue", 0.34, BLUE), ("red", 0.33, RED), ("yellow", 0.33, YELLOW)]
    P = [
        # name, types, adhesion matrix (row-major, T*T)
        ("sort2",   two,   [1.0, 0.3, 0.3, 1.0]),                 # strong differential -> 2 domains
        ("weak2",   two,   [1.0, 0.7, 0.7, 1.0]),                 # weak contrast -> partial sort
        ("engulf2", two,   [1.5, 0.4, 0.4, 0.6]),                 # A very cohesive -> A engulfed by B
        ("control", two,   [1.0, 1.0, 1.0, 1.0]),                 # equal adhesion -> stays mixed
        ("hier3",   three, [0.5, 0.4, 0.3,                        # blue<red<yellow self-adhesion
                            0.4, 1.0, 0.5,
                            0.3, 0.5, 1.5]),
    ]
    return [(n, dict(types=t, adhesion=a, N=900, frames=1500)) for n, t, a in P]


def make_sim(p):
    types = {nm: {"fraction": fr} for nm, fr, _ in p["types"]}
    cfg = {
        "general": {"name": "cs", "seed": 1, "n_frames": p["frames"], "dt": 1.0,
                    "boundary": "free", "world": [1.0, 1.0]},
        "sets": {"cell": {"n": p["N"], "spawn": "disc", "spawn_radius": 0.32, "types": types}},
        "fields": {},
        "operators": [
            {"op": "radius_graph", "at": "cell", "radius": 0.048},
            {"op": "differential_adhesion", "at": "cell", "adhesion": p["adhesion"],
             "sigma": 0.03, "r_adh": 0.048, "k_rep": 90.0, "mu": 0.0008,
             "confine": 0.001, "noise": 0.0004},
        ],
        "schedule": ["radius_graph", "differential_adhesion"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def render(pos, node_type, colors, outdir, name, seconds=16.0, max_frames=400):
    os.makedirs(outdir, exist_ok=True)
    nt = np.asarray(node_type)
    cmap = np.array(colors)
    c = cmap[nt]                                            # [N,3] fixed per-cell colour
    T = pos.shape[0]
    stride = max(1, -(-T // max_frames)); idx = list(range(0, T, stride))
    fps = max(1, round(len(idx) / seconds))
    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.1, 0.35, 0.7, 1.0)]
    fig, ax = plt.subplots(1, len(picks), figsize=(len(picks) * 2.2, 2.4)); fig.patch.set_facecolor("black")
    for a, t in zip(ax, picks):
        a.scatter(pos[t, :, 0], pos[t, :, 1], s=7, c=c, linewidths=0); a.set_facecolor("black")
        a.set_title(f"{int(100*t/max(T-1,1))}%", color="white", fontsize=9)
        a.set_xlim(0.1, 0.9); a.set_ylim(0.1, 0.9); a.set_aspect("equal"); a.axis("off")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.9, wspace=0.05)
    fig.savefig(os.path.join(outdir, "strip.png"), dpi=110, facecolor="black"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4.2)); fig.patch.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1); ax.set_facecolor("black"); ax.axis("off"); ax.set_aspect("equal")
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0.1, 0.9)
    sc = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=10, c=c, linewidths=0)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=100):
        for t in idx:
            sc.set_offsets(pos[t]); w.grab_frame()
    plt.close(fig)


def diagnostics(pos, node_type, types, radius=0.048):
    xy = pos[-1]; nt = np.asarray(node_type)
    # homotypic neighbour fraction (sorting index) at the final frame
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
    nb = d < radius
    same = 0; tot = 0
    for k in range(len(nt)):
        idx = np.where(nb[k])[0]
        if len(idx):
            same += (nt[idx] == nt[k]).sum(); tot += len(idx)
    homotypic = round(float(same / max(tot, 1)), 3)
    c = np.array([0.5, 0.5])
    r = np.sqrt(((xy - c) ** 2).sum(-1))
    radii = {nm: round(float(r[nt == ti].mean()), 3) for ti, (nm, _, _) in enumerate(types)}
    return dict(homotypic=homotypic, mean_radius_by_type=radii)


def run_share(rank, nproc, device):
    P = presets()[rank::nproc]
    print(f"[rank {rank}] {len(P)} presets on {device}", flush=True)
    for i, (name, p) in enumerate(P):
        odir = os.path.join(OUT, name)
        if os.path.exists(os.path.join(odir, "movie.mp4")):
            print(f"[rank {rank}] skip {name}", flush=True); continue
        try:
            print(f"[rank {rank}] ({i+1}/{len(P)}) {name}", flush=True)
            sim, cfg = make_sim(p)
            os.makedirs(odir, exist_ok=True)
            with open(os.path.join(odir, "spec.yaml"), "w") as sf:
                yaml.safe_dump(cfg, sf, sort_keys=False)
            _, out = engine_run(sim, device=device)
            cell = out["sets"]["cell"]
            pos = cell["pos"]; nt = cell["node_type"]
            colors = [c for _, _, c in p["types"]]
            render(pos, nt, colors, odir, name)
            diag = diagnostics(pos, nt, p["types"]); diag["adhesion"] = p["adhesion"]
            json.dump(diag, open(os.path.join(odir, "diag.json"), "w"), indent=1)
        except Exception:
            print(f"[rank {rank}] {name} FAILED\n{traceback.format_exc()}", flush=True)
    print(f"[rank {rank}] done", flush=True)


def montage():
    from PIL import Image
    strips = sorted(glob.glob(os.path.join(OUT, "*", "strip.png")))
    ims = []
    for fpath in strips:
        name = os.path.basename(os.path.dirname(fpath))
        im = Image.open(fpath).convert("RGB"); im = im.resize((560, int(560 * im.height / im.width)))
        ims.append((name, im))
    if ims:
        w = max(i.width for _, i in ims); h = sum(i.height for _, i in ims)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, i in ims:
            sheet.paste(i, (0, y)); y += i.height
        sheet.save(os.path.join(OUT, "_montage.png"))
    with open(os.path.join(OUT, "_summary.md"), "w") as fh:
        fh.write("# embryo_cell_sorting — differential adhesion (Steinberg)\n\n")
        fh.write("| preset | homotypic (sorting) | mean radius by type | adhesion |\n|--|--|--|--|\n")
        for fpath in sorted(glob.glob(os.path.join(OUT, "*", "diag.json"))):
            d = json.load(open(fpath)); name = os.path.basename(os.path.dirname(fpath))
            fh.write(f"| {name} | {d.get('homotypic','?')} | {d.get('mean_radius_by_type','?')} "
                     f"| {d.get('adhesion','?')} |\n")
    print(f"montage: {len(ims)} presets -> {OUT}/_montage.png + _summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    montage() if a.montage else run_share(a.rank, a.nproc, a.device)


if __name__ == "__main__":
    main()
