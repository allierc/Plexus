#!/usr/bin/env python
"""run_embryo_vertex_3d -- 3D Self-Propelled Voronoi tissue as a strict-Plexus sim.

3D confluent tissue = the Voronoi tessellation of cell centres in a periodic box; mechanics from
the 3D shape energy E = Σ[K_V(V−V₀)² + K_S(S−S₀)²] via `vertex_tension_3d`. Sweeps the 3D target
shape index s₀ across the rigidity transition (s₀*≈5.41): below → solid (jammed), above → fluid
(flowing). Renders a TRANSPARENT 3D "colloidal" view — cells as translucent spheres coloured by
their shape index, under a slowly-rotating camera, on black — so you can see through the tissue.

    python run_embryo_vertex_3d.py --device cuda:1
    python run_embryo_vertex_3d.py --montage
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators           # noqa: F401
import embryo_vertex_3d_ops as V3  # vertex_tension_3d + geometry helpers
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive_3d")
N = 125
L = round(N ** (1.0 / 3.0))                             # density 1 (V0=1 -> mean volume 1); L=5
CMAP = matplotlib.colormaps["plasma"]


def presets():
    """Sweep the 3D target shape index s₀ across the rigidity transition (s₀*≈5.41)."""
    P = [
        ("s0_500", dict(s0=5.00, v0=0.15)),   # solid
        ("s0_530", dict(s0=5.30, v0=0.15)),   # solid, near transition
        ("s0_541", dict(s0=5.41, v0=0.15)),   # at the transition
        ("s0_560", dict(s0=5.60, v0=0.15)),   # fluid
        ("s0_590", dict(s0=5.90, v0=0.15)),   # deep fluid
        ("passive", dict(s0=5.60, v0=0.0)),   # control: no motility -> frozen
    ]
    return [(n, dict(frames=300, dt=0.05, Dr=1.0, **d)) for n, d in P]


def make_sim(p):
    cfg = {
        "general": {"name": "spv3", "seed": 1, "n_frames": p["frames"], "dt": p["dt"],
                    "boundary": "periodic", "dim": 3, "world": [float(L)] * 3},
        "sets": {"cell": {"n": N, "spawn": "random", "types": {"a": {"fraction": 1.0}}}},
        "fields": {},
        "operators": [{"op": "vertex_tension_3d", "at": "cell", "s0": p["s0"], "v0": p["v0"],
                       "Dr": p["Dr"], "K_V": 1.0, "K_S": 1.0, "V0": 1.0, "mu": 1.0, "dt": p["dt"]}],
        "schedule": ["vertex_tension_3d"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _shape_index(xy):
    _, vol, surf, ok = V3.cell_polyhedra(xy.astype(np.float64) % L, L, N)
    q = np.full(N, np.nan)
    m = ok > 0
    q[m] = surf[m] / np.clip(vol[m], 1e-9, None) ** (2.0 / 3.0)
    return q, vol


def render(pos, outdir, name, seconds=16.0, max_frames=140):
    os.makedirs(outdir, exist_ok=True)
    T = pos.shape[0]
    stride = max(1, -(-T // max_frames)); idx = list(range(0, T, stride))
    fps = max(1, round(len(idx) / seconds))

    def draw(ax, t, azim):
        ax.clear(); ax.set_facecolor("black")
        faces, svals = V3.cell_faces(pos[t].astype(np.float64) % L, L, N)
        # keep cells whose centre is inside the box (the central tissue), drop boundary-crossers
        tris, cols = [], []
        for cell_tris, s in zip(faces, svals):
            centroid = cell_tris.reshape(-1, 3).mean(0)
            if np.any(centroid < -0.5) or np.any(centroid > L + 0.5):
                continue
            c = CMAP(np.clip((s - 5.0) / 1.0, 0, 1))
            rgba = (c[0], c[1], c[2], 0.30)                   # translucent -> see through the tissue
            for tri in cell_tris:
                tris.append(tri); cols.append(rgba)
        pc = Poly3DCollection(tris, facecolors=cols, edgecolors=(1, 1, 1, 0.12), linewidths=0.15)
        ax.add_collection3d(pc)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=22, azim=azim)

    fig = plt.figure(figsize=(4.8, 4.8)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1)
    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
    sfig = plt.figure(figsize=(4 * 2.4, 2.5)); sfig.patch.set_facecolor("black")
    for k, t in enumerate(picks):
        sax = sfig.add_subplot(1, 4, k + 1, projection="3d")
        draw(sax, t, 30 + 20 * k); sax.set_title(f"{int(100*t/max(T-1,1))}%", color="white", fontsize=9)
    sfig.subplots_adjust(0.01, 0.01, 0.99, 0.9, wspace=0.02)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=110, facecolor="black"); plt.close(sfig)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=100):
        for j, t in enumerate(idx):
            draw(ax, t, 20 + 100 * j / max(len(idx) - 1, 1))     # slow camera orbit
            w.grab_frame()
    plt.close(fig)


def diagnostics(pos):
    T = pos.shape[0]
    disp = np.diff(pos, axis=0); disp -= L * np.round(disp / L)
    unwrap = np.concatenate([pos[:1], pos[:1] + np.cumsum(disp, axis=0)], axis=0)
    half = T // 2
    msd = float(((unwrap[-1] - unwrap[half]) ** 2).sum(-1).mean())
    deff = msd / (T - half)
    q, vol = _shape_index(pos[-1])
    return dict(msd=round(msd, 4), deff=round(deff, 6), shape_index=round(float(np.nanmean(q)), 3),
                mean_vol=round(float(np.nanmean(vol)), 3),
                state=("fluid" if deff > 3e-3 else "solid"))


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
            pos = out["sets"]["cell"]["pos"]
            render(pos, odir, name)
            diag = diagnostics(pos); diag.update({k: p[k] for k in ("s0", "v0")})
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
        fh.write("# embryo_vertex 3D -- Self-Propelled Voronoi rigidity transition (s0*~=5.41)\n\n")
        fh.write("| preset | s0 | v0 | shape_index | mean_vol | MSD | Deff | state |\n|--|--|--|--|--|--|--|--|\n")
        for fpath in sorted(glob.glob(os.path.join(OUT, "*", "diag.json"))):
            d = json.load(open(fpath)); name = os.path.basename(os.path.dirname(fpath))
            fh.write(f"| {name} | {d.get('s0','?')} | {d.get('v0','?')} | {d.get('shape_index','?')} "
                     f"| {d.get('mean_vol','?')} | {d.get('msd','?')} | {d.get('deff','?')} "
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
