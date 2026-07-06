#!/usr/bin/env python
"""run_embryo_vertex -- Self-Propelled Voronoi tissue as a strict-Plexus sim.

Confluent tissue = the Voronoi tessellation of cell centres; mechanics from the shape energy
E = Σ[K_A(A−A₀)² + K_P(P−P₀)²] via the `vertex_tension` operator (autodiff force + self-propulsion).
Sweeps the target shape index p₀ across the rigidity transition (p₀*≈3.81): below it the tissue is
a SOLID (jammed, low MSD, hexagonal), above a FLUID (flowing via T1s, high MSD, irregular). Renders
the live Voronoi tessellation coloured by cell shape index, on black. Reports MSD/effective
diffusion, mean shape index, and the T1 (neighbour-exchange) rate.

    python run_embryo_vertex.py --device cuda:1
    python run_embryo_vertex.py --montage
"""
from __future__ import annotations
import os, sys, argparse, glob, json, tempfile, traceback, math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators           # noqa: F401
import embryo_vertex_ops as V      # vertex_tension + geometry helpers
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
N = 256
L = round(math.sqrt(N))                                 # density 1 (A0 = 1 -> mean area 1)
CMAP = cm.get_cmap("plasma")


def presets():
    """Sweep the target shape index p₀ across the rigidity transition (p₀*≈3.81). A v₀=0 control
    (no self-propulsion) shows the passive ground state; the rest are motile (v₀=0.2)."""
    P = [
        ("p0_360", dict(p0=3.60, v0=0.2)),   # deep solid
        ("p0_375", dict(p0=3.75, v0=0.2)),   # solid
        ("p0_381", dict(p0=3.81, v0=0.2)),   # at the transition
        ("p0_390", dict(p0=3.90, v0=0.2)),   # fluid
        ("p0_410", dict(p0=4.10, v0=0.2)),   # deep fluid
        ("passive", dict(p0=3.90, v0=0.0)),  # control: no motility -> frozen even above p0*
        ("grow",    dict(p0=3.85, v0=0.10, n0=90, buffer=300, div_rate=0.025)),  # cell_divide -> growing tissue
    ]
    return [(n, dict(N=N, frames=800, dt=0.05, Dr=1.0, **d)) for n, d in P]


def make_sim(p):
    setcfg = {"n": p.get("n0", N), "spawn": "random", "types": {"a": {"fraction": 1.0}}}
    ops = [{"op": "vertex_tension", "at": "cell", "p0": p["p0"], "v0": p["v0"],
            "Dr": p["Dr"], "K_A": 1.0, "K_P": 1.0, "A0": 1.0, "mu": 1.0, "dt": p["dt"]}]
    sched = ["vertex_tension"]
    if p.get("div_rate"):                                     # proliferation: cell_divide on the vertex tissue
        setcfg["buffer"] = p.get("buffer", 300)
        ops.append({"op": "cell_divide", "at": "cell", "rate": p["div_rate"], "offset": 0.25, "max_occ": 0.95})
        sched.append("cell_divide")
    cfg = {
        "general": {"name": "spv", "seed": 1, "n_frames": p["frames"], "dt": p["dt"],
                    "boundary": "periodic", "world": [float(L), float(L)]},
        "sets": {"cell": setcfg},
        "fields": {},
        "operators": ops,
        "schedule": sched,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def render(pos, outdir, name, occ=None, seconds=16.0, max_frames=300):
    os.makedirs(outdir, exist_ok=True)
    T = pos.shape[0]
    stride = max(1, -(-T // max_frames)); idx = list(range(0, T, stride))
    fps = max(1, round(len(idx) / seconds))

    def draw(ax, t):
        ax.clear(); ax.set_facecolor("black")
        xy = pos[t]
        live = (occ[t] > 0) if occ is not None else np.ones(xy.shape[0], bool)
        xy = xy[live]
        n = xy.shape[0]
        polys, area, perim, ok = V.cell_polygons(xy.astype(np.float64) % L, L, n)
        verts, cols = [], []
        for i in range(n):
            if polys[i] is not None and area[i] > 1e-6:
                verts.append(polys[i])
                q = perim[i] / math.sqrt(max(area[i], 1e-9))          # per-cell shape index
                cols.append(CMAP(np.clip((q - 3.7) / 0.7, 0, 1)))
        pc = PolyCollection(verts, facecolors=cols, edgecolors="black", linewidths=0.4)
        ax.add_collection(pc)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    fig, ax = plt.subplots(figsize=(4.6, 4.6)); fig.patch.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1)
    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.25, 0.5, 0.75, 1.0)]
    sfig, sax = plt.subplots(1, len(picks), figsize=(len(picks) * 2.2, 2.3)); sfig.patch.set_facecolor("black")
    for a, t in zip(sax, picks):
        draw(a, t); a.set_title(f"{int(100*t/max(T-1,1))}%", color="white", fontsize=9)
    sfig.subplots_adjust(0.01, 0.01, 0.99, 0.9, wspace=0.05)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=110, facecolor="black"); plt.close(sfig)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=100):
        for t in idx:
            draw(ax, t); w.grab_frame()
    plt.close(fig)


def diagnostics(pos, occ=None):
    T = pos.shape[0]
    if occ is not None and int(occ[0].sum()) != int(occ[-1].sum()):    # growing tissue
        live = occ[-1] > 0
        p = pos[-1][live].astype(np.float64) % L
        n = int(live.sum())
        _, area, perim, ok = V.cell_polygons(p, L, n)
        q = float((perim[ok > 0] / np.sqrt(np.clip(area[ok > 0], 1e-9, None))).mean())
        return dict(n_cells=n, shape_index=round(q, 3), state="growing")
    disp = np.diff(pos, axis=0); disp -= L * np.round(disp / L)
    unwrap = np.concatenate([pos[:1], pos[:1] + np.cumsum(disp, axis=0)], axis=0)
    half = T // 2
    msd = float(((unwrap[-1] - unwrap[half]) ** 2).sum(-1).mean())
    deff = msd / (T - half)                                        # effective diffusion (per frame)
    # shape index + T1 rate at the final state
    _, area, perim, ok = V.cell_polygons(pos[-1].astype(np.float64) % L, L, N)
    q = float((perim[ok > 0] / np.sqrt(np.clip(area[ok > 0], 1e-9, None))).mean())
    n0 = V.delaunay_neighbors(pos[max(0, T - 40)].astype(np.float64) % L, L, N)
    n1 = V.delaunay_neighbors(pos[-1].astype(np.float64) % L, L, N)
    t1 = len(n0 ^ n1)                                             # neighbour-pair changes over 40 frames
    return dict(msd=round(msd, 4), deff=round(deff, 6), shape_index=round(q, 3),
                t1_changes=int(t1), state=("fluid" if deff > 5e-4 else "solid"))


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
            pos = out["sets"]["cell"]["pos"]; occ = out["sets"]["cell"]["occ"]
            render(pos, odir, name, occ=occ)
            diag = diagnostics(pos, occ); diag.update({k: p[k] for k in ("p0", "v0")})
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
        im = Image.open(fpath).convert("RGB"); im = im.resize((580, int(580 * im.height / im.width)))
        ims.append((name, im))
    if ims:
        w = max(i.width for _, i in ims); h = sum(i.height for _, i in ims)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, i in ims:
            sheet.paste(i, (0, y)); y += i.height
        sheet.save(os.path.join(OUT, "_montage.png"))
    with open(os.path.join(OUT, "_summary.md"), "w") as fh:
        fh.write("# embryo_vertex — Self-Propelled Voronoi rigidity transition (p0*≈3.81)\n\n")
        fh.write("| preset | p0 | v0 | shape_index | MSD | Deff | T1 changes | state |\n|--|--|--|--|--|--|--|--|\n")
        for fpath in sorted(glob.glob(os.path.join(OUT, "*", "diag.json"))):
            d = json.load(open(fpath)); name = os.path.basename(os.path.dirname(fpath))
            fh.write(f"| {name} | {d.get('p0','?')} | {d.get('v0','?')} | {d.get('shape_index','?')} "
                     f"| {d.get('msd','?')} | {d.get('deff','?')} | {d.get('t1_changes','?')} "
                     f"| {d.get('state','?')} |\n")
    print(f"montage: {len(ims)} presets -> {OUT}/_montage.png + _summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--montage", action="store_true")
    a = ap.parse_args()
    montage() if a.montage else run_share(a.rank, a.nproc, a.device)


if __name__ == "__main__":
    main()
