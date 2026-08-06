#!/usr/bin/env python
"""run_sheet -- cell_polarity case 1: apical constriction -> INVAGINATION of an epithelial sheet.

A flat apical/basal monolayer (`seed_sheet`) is given apico-basal polarity (`cell_polarity`: a
central patch gets high apical cortical tension) and relaxed under the epithelial shape energy
(`epithelium`). The constricting patch wedges its cells so the sheet buckles into a furrow --
the canonical gastrulation / neural-tube fold. Renders the folding cells (patch highlighted) with
values printed top-left. Sweeps the constriction strength; a no-constriction control stays flat.

    python run_sheet.py --device cuda:0
    python run_sheet.py --montage
"""
from __future__ import annotations
import os, sys, argparse, glob, json, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators           # noqa: F401
import sheet_ops                   # noqa: F401  seed_sheet + cell_polarity + epithelium
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive_sheet")
NC = 48                                                 # cells; set size = 2*(NC+1)
WORLD = [10.0, 6.0]
WIDTH, H0 = 8.0, 1.0
A0 = (WIDTH / NC) * H0                                  # preferred area = initial cell area


def presets():
    return [
        ("fold",      dict(constrict=0.5, elongate=0.8, patch_half=0.14)),   # apical constriction -> furrow
        ("fold_wide", dict(constrict=0.4, elongate=0.6, patch_half=0.24)),   # wider, shallower domain
        ("fold_deep", dict(constrict=0.6, elongate=1.4, patch_half=0.12)),   # strong -> deep narrow furrow
        ("control",   dict(constrict=0.0, elongate=0.0, patch_half=0.14)),   # no polarity -> stays flat
    ]


def make_sim(p, frames=1000, dt=0.02):
    n = 2 * (NC + 1)
    cfg = {
        "general": {"name": "sheet", "seed": 0, "n_frames": frames, "dt": dt,
                    "boundary": "free", "world": WORLD},
        "sets": {"cell": {"n": n, "spawn": "random", "types": {"a": {"fraction": 1.0}}}},
        "fields": {},
        "operators": [
            {"op": "seed_sheet", "at": "cell", "width": WIDTH, "height": H0, "bow": 0.3, "before_frame": 1},
            {"op": "cell_polarity", "at": "cell", "constrict": p["constrict"], "elongate": p["elongate"],
             "patch_center": 0.5, "patch_half": p["patch_half"]},
            {"op": "epithelium", "at": "cell", "K_A": 2.0, "A0": A0, "k_ap": 3.0, "k_ba": 2.0,
             "k_lat": 1.0, "h0": H0, "mu": 0.4, "pin": "basal"},
        ],
        "schedule": ["seed_sheet", "cell_polarity", "epithelium"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _quads(frame):
    Ncp = NC + 1
    A = frame[:Ncp]; B = frame[Ncp:]
    return [np.array([A[i], A[i + 1], B[i + 1], B[i]]) for i in range(NC)]


def _patch_mask(p):
    frac = np.linspace(0, 1, NC)
    return (np.abs(frac - 0.5) < p["patch_half"]) & ((p["constrict"] > 0) | (p.get("elongate", 0) > 0))


def render(pos, outdir, name, p, diag, seconds=14.0, max_frames=280):
    os.makedirs(outdir, exist_ok=True)
    T = pos.shape[0]
    idx = list(range(0, T, max(1, -(-T // max_frames))))
    fps = max(1, round(len(idx) / seconds))
    patch = _patch_mask(p)
    cols = np.where(patch[:, None], np.array([[0.95, 0.35, 0.2]]), np.array([[0.3, 0.55, 0.9]]))

    def draw(ax, t):
        ax.clear(); ax.set_facecolor("black")
        pc = PolyCollection(_quads(pos[t]), facecolors=cols, edgecolors=(1, 1, 1, 0.4), linewidths=0.4)
        ax.add_collection(pc)
        ax.set_xlim(1, 9); ax.set_ylim(5.5, 0.5); ax.set_aspect("equal"); ax.axis("off")   # flipped upside down
        info = (f"{name}\napical constrict {p['constrict']:.0%}\npatch={2*p['patch_half']:.0%} of sheet"
                f"\nfurrow depth={diag['furrow_depth']:.2f}")
        ax.text(0.02, 0.98, info, transform=ax.transAxes, color="white", fontsize=6,
                va="top", ha="left", family="monospace")

    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.2, 0.5, 1.0)]
    sfig, sax = plt.subplots(1, len(picks), figsize=(len(picks) * 2.6, 2.0)); sfig.patch.set_facecolor("black")
    for a, t in zip(sax, picks):
        draw(a, t); a.set_title(f"{int(100*t/max(T-1,1))}%", color="white", fontsize=9)
    sfig.subplots_adjust(0.01, 0.01, 0.99, 0.88, wspace=0.05)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)
    fig, ax = plt.subplots(figsize=(6, 3.6)); fig.patch.set_facecolor("black"); fig.subplots_adjust(0, 0, 1, 1)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=110):
        for t in idx:
            draw(ax, t); w.grab_frame()
    plt.close(fig)


def diagnostics(pos):
    Ncp = NC + 1
    a0 = pos[0][:Ncp]; aT = pos[-1][:Ncp]
    flat = float(a0[0, 1])                                 # flank apical level (flat)
    depth = float(flat - aT[:, 1].min())                  # how far the apical surface dipped below flat
    # apical patch length change
    ap_len0 = np.linalg.norm(np.diff(a0, axis=0), axis=1).sum()
    ap_lenT = np.linalg.norm(np.diff(aT, axis=0), axis=1).sum()
    return dict(furrow_depth=round(depth, 3),
                apical_len_ratio=round(float(ap_lenT / ap_len0), 3),
                folded=bool(depth > 0.3))


def run_share(rank, nproc, device):
    P = presets()[rank::nproc]
    print(f"[rank {rank}] {len(P)} presets on {device}", flush=True)
    for i, (name, p) in enumerate(P):
        odir = os.path.join(OUT, name)
        os.makedirs(odir, exist_ok=True)
        print(f"[rank {rank}] ({i+1}/{len(P)}) {name}", flush=True)
        sim, cfg = make_sim(p)
        with open(os.path.join(odir, "spec.yaml"), "w") as sf:
            yaml.safe_dump(cfg, sf, sort_keys=False)
        _, out = engine_run(sim, device=device)
        pos = out["sets"]["cell"]["pos"]
        diag = diagnostics(pos); diag["constrict"] = p["constrict"]
        render(pos, odir, name, p, diag)
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
        fh.write("# cell_polarity case 1 -- apical constriction -> invagination\n\n")
        fh.write("| preset | constrict | furrow depth | apical len ratio | folded |\n|--|--|--|--|--|\n")
        for f in sorted(glob.glob(os.path.join(OUT, "*", "diag.json"))):
            d = json.load(open(f)); name = os.path.basename(os.path.dirname(f))
            fh.write(f"| {name} | {d.get('constrict','?')} | {d.get('furrow_depth','?')} "
                     f"| {d.get('apical_len_ratio','?')} | {d.get('folded','?')} |\n")
    print(f"montage: {len(ims)} -> {OUT}/_montage.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    montage() if a.montage else run_share(a.rank, a.nproc, a.device)


if __name__ == "__main__":
    main()
