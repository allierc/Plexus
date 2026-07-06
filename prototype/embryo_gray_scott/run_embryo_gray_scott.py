#!/usr/bin/env python
"""run_embryo3 -- Turing / Gray-Scott reaction-diffusion morphogenesis as a strict-Plexus field sim.

Builds Plexus specs (a 2-channel `grid` field + the `gray_scott` field operator), runs each
through the Plexus engine, and renders the autocatalyst morphogen B (inferno on black) to an
mp4 + a development strip in archive/<name>/. Sweeps the Pearson (f, k) map -- each pair is a
different morphogenetic CLASS (spots, stripes, mazes, self-replicating "mitosis", solitons).
Splits across GPUs like galaxy_sweep / run_embryo2:

    python run_embryo3.py --rank 0 --nproc 2 --device cuda:0 &
    python run_embryo3.py --rank 1 --nproc 2 --device cuda:1 &
    wait; python run_embryo3.py --montage
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
import matplotlib.cm as cm
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators           # noqa: F401
import embryo_gray_scott_ops                 # noqa: F401  gray_scott + rd_seed
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
RES = 200
CMAP = cm.get_cmap("inferno")
VMAX = 0.4                                          # fixed B normalisation -> consistent look


def presets():
    """The Pearson (f, k) map -- each pair selects a distinct morphogenetic class. Values
    from Pearson (1993) / Munafo's Gray-Scott taxonomy; all in the living (non-extinct) region."""
    P = [
        ("mitosis",  0.0367, 0.0649),   # self-replicating spots (division-like)
        ("spots",    0.0350, 0.0650),   # stable spot lattice
        ("coral",    0.0545, 0.0620),   # branching coral / fingerprints
        ("worms",    0.0580, 0.0630),   # labyrinthine worms
        ("maze",     0.0290, 0.0570),   # maze / stripes
        ("holes",    0.0390, 0.0580),   # negative spots (holes)
        ("chaos",    0.0260, 0.0510),   # spatiotemporal chaos
        ("waves",    0.0180, 0.0500),   # travelling / oscillating fronts (needs strong ignition)
        ("solitons", 0.0300, 0.0560),   # moving localised spots
        ("uskate",   0.0620, 0.0610),   # u-skate gliders
        ("stripes",  0.0220, 0.0490),   # parallel stripes
        ("default",  0.0600, 0.0620),   # the reference (worms/coral edge)
    ]
    # low-feed presets need a stronger initial ignition (bigger central seed + more noise)
    strong = {"waves", "stripes", "maze"}
    out = []
    for n, f, k in P:
        d = dict(f=f, k=k, frames=700, substeps=20)
        if n in strong:
            d.update(seed_frac=0.20, influence=0.12)
        out.append((n, d))
    return out


def make_sim(p):
    cfg = {
        "general": {"name": "embryo3", "seed": 0, "n_frames": p["frames"], "dt": 1.0,
                    "boundary": "periodic", "world": [1.0, 1.0]},
        "sets": {"seed_cell": {"n": 1, "types": {"a": {"fraction": 1.0}}}},  # dummy set (engine needs >=1)
        "fields": {"rd": {"frame": "grid", "res": RES, "components": 2}},
        "operators": [
            {"op": "rd_seed", "at": "rd", "before_frame": 1,
             **({"seed_frac": p["seed_frac"]} if "seed_frac" in p else {}),
             **({"influence": p["influence"]} if "influence" in p else {})},
            {"op": "gray_scott", "at": "rd", "f": p["f"], "k": p["k"],
             "substeps": p["substeps"], "dt": 1.0},
        ],
        "schedule": ["rd_seed", "gray_scott"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _rgb(gridB):
    """B channel [nx, ny] -> inferno RGB on black."""
    return CMAP(np.clip(gridB / VMAX, 0.0, 1.0))[..., :3]


def render(grids, outdir, name, seconds=22.0, max_frames=560):
    """grids: [T, 2, nx, ny]. Morphogen-B movie (inferno/black, ~`seconds`) + a development strip."""
    os.makedirs(outdir, exist_ok=True)
    T = grids.shape[0]
    stride = max(1, -(-T // max_frames))
    idx = list(range(0, T, stride))
    fps = max(1, round(len(idx) / seconds))
    picks = [int(round(fr * (T - 1))) for fr in (0.02, 0.12, 0.35, 0.65, 1.0)]
    fig, ax = plt.subplots(1, len(picks), figsize=(len(picks) * 2.1, 2.4))
    fig.patch.set_facecolor("black")
    for a, t in zip(ax, picks):
        a.imshow(_rgb(grids[t, 1]), origin="lower"); a.set_title(f"{int(100*t/(T-1))}%",
                                                                 color="white", fontsize=9)
        a.axis("off")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.9, wspace=0.05)
    fig.savefig(os.path.join(outdir, "strip.png"), dpi=110, facecolor="black")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4)); fig.patch.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1); ax.axis("off")
    im = ax.imshow(_rgb(grids[0, 1]), origin="lower")
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=100):
        for t in idx:
            im.set_data(_rgb(grids[t, 1])); w.grab_frame()
    plt.close(fig)


def diagnostics(grids):
    """Pattern observables from morphogen B: coverage, contrast, and whether it survived."""
    B = grids[:, 1]                                            # [T, nx, ny]
    final = B[-1]
    cover = float((final > 0.2).mean())
    return dict(B_mean_final=round(float(final.mean()), 4),
                B_coverage=round(cover, 4),
                B_contrast=round(float(final.std()), 4),
                alive=bool(final.max() > 0.15))


def run_share(rank, nproc, device):
    P = presets()[rank::nproc]
    print(f"[rank {rank}] {len(P)} presets on {device}", flush=True)
    for i, (name, p) in enumerate(P):
        odir = os.path.join(OUT, name)
        if os.path.exists(os.path.join(odir, "movie.mp4")):
            print(f"[rank {rank}] skip {name}", flush=True); continue
        try:
            print(f"[rank {rank}] ({i+1}/{len(P)}) {name} f={p['f']} k={p['k']}", flush=True)
            sim, cfg = make_sim(p)
            os.makedirs(odir, exist_ok=True)
            with open(os.path.join(odir, "spec.yaml"), "w") as sf:
                yaml.safe_dump(cfg, sf, sort_keys=False)
            _, out = engine_run(sim, device=device)
            grids = out["fields"]["rd"]["grid"]               # [T, 2, nx, ny]
            render(grids, odir, name)
            diag = diagnostics(grids); diag.update({k: p[k] for k in ("f", "k")})
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
        im = Image.open(fpath).convert("RGB")
        im = im.resize((560, int(560 * im.height / im.width)))
        ims.append((name, im))
    if ims:
        w = max(i.width for _, i in ims); h = sum(i.height for _, i in ims)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, i in ims:
            sheet.paste(i, (0, y)); y += i.height
        sheet.save(os.path.join(OUT, "_montage.png"))
    with open(os.path.join(OUT, "_summary.md"), "w") as fh:
        fh.write("# embryo3 (Turing / Gray-Scott) — Pearson (f,k) pattern classes\n\n")
        fh.write("| preset | f | k | coverage | contrast | alive |\n|--|--|--|--|--|--|\n")
        for fpath in sorted(glob.glob(os.path.join(OUT, "*", "diag.json"))):
            d = json.load(open(fpath)); name = os.path.basename(os.path.dirname(fpath))
            fh.write(f"| {name} | {d.get('f','?')} | {d.get('k','?')} | {d.get('B_coverage','?')} "
                     f"| {d.get('B_contrast','?')} | {d.get('alive','?')} |\n")
    print(f"montage: {len(ims)} presets -> {OUT}/_montage.png + _summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    montage() if a.montage else run_share(a.rank, a.nproc, a.device)


if __name__ == "__main__":
    main()
