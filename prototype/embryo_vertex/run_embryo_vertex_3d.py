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
    return [(n, dict(frames=600, dt=0.0125, Dr=1.0, **d)) for n, d in P]   # dt/4: slower, smoother 3D


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


def _slice_polygon(tris, z0):
    """Cross-section polygon of a convex polyhedron (given as hull triangles [F,3,3]) by the
    plane z=z0: intersect each triangle edge, collect points, order around the centroid."""
    pts = []
    for tri in tris:
        z = tri[:, 2]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (z[a] - z0) * (z[b] - z0) < 0:
                t = (z0 - z[a]) / (z[b] - z[a])
                pts.append((tri[a] + t * (tri[b] - tri[a]))[:2])
    if len(pts) < 3:
        return None
    pts = np.array(pts); c = pts.mean(0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]


def render(pos, outdir, name, diag=None, seconds=28.0, max_frames=250):
    """2x1 panel per frame: LEFT = transparent 3D Voronoi tissue, RIGHT = a Voronoi cross-section
    slice at z=L/2, both coloured by cell shape index; a caption prints the live values."""
    os.makedirs(outdir, exist_ok=True)
    T = pos.shape[0]
    stride = max(1, -(-T // max_frames)); idx = list(range(0, T, stride))
    fps = max(1, round(len(idx) / seconds))
    z0 = L / 2.0
    dtxt = "" if diag is None else f"   Deff={diag.get('deff','?')}   {diag.get('state','?')}"

    def draw(fig, t, azim):
        fig.clf(); fig.patch.set_facecolor("black")
        faces, svals = V3.cell_faces(pos[t].astype(np.float64) % L, L, N)
        # LEFT: transparent 3D polyhedra
        ax1 = fig.add_subplot(1, 2, 1, projection="3d"); ax1.set_facecolor("black")
        tris3d, cols3d = [], []
        for ct, s in zip(faces, svals):
            cen = ct.reshape(-1, 3).mean(0)
            if np.any(cen < -0.5) or np.any(cen > L + 0.5):
                continue
            c = CMAP(np.clip((s - 5.0) / 1.0, 0, 1)); rgba = (c[0], c[1], c[2], 0.30)
            for tri in ct:
                tris3d.append(tri); cols3d.append(rgba)
        ax1.add_collection3d(Poly3DCollection(tris3d, facecolors=cols3d, edgecolors=(1, 1, 1, 0.10),
                                              linewidths=0.12))
        ax1.set_xlim(0, L); ax1.set_ylim(0, L); ax1.set_zlim(0, L)
        ax1.set_box_aspect((1, 1, 1)); ax1.set_axis_off(); ax1.view_init(elev=20, azim=azim)
        # RIGHT: cross-section at z0
        ax2 = fig.add_subplot(1, 2, 2); ax2.set_facecolor("black")
        ncut = 0
        for ct, s in zip(faces, svals):
            poly = _slice_polygon(ct, z0)
            if poly is not None and len(poly) >= 3:
                c = CMAP(np.clip((s - 5.0) / 1.0, 0, 1))
                ax2.fill(poly[:, 0], poly[:, 1], facecolor=c, alpha=0.9, edgecolor="white", lw=0.5)
                ncut += 1
        ax2.set_xlim(0, L); ax2.set_ylim(0, L); ax2.set_aspect("equal"); ax2.axis("off")
        # params printed in panel-2 top-left, small font (not in a title)
        q = float(np.nanmean(svals))
        info = (f"{name}\ns0={diag.get('s0','?') if diag else '?'}  v0={diag.get('v0','?') if diag else '?'}"
                f"\n<s>={q:.3f}   z={z0:.1f}\ncells cut={ncut}"
                f"\nDeff={diag.get('deff','?') if diag else '?'}\n{diag.get('state','') if diag else ''}")
        ax2.text(0.02, 0.98, info, transform=ax2.transAxes, color="white", fontsize=6,
                 va="top", ha="left", family="monospace")

    fig = plt.figure(figsize=(8.6, 4.5)); fig.patch.set_facecolor("black")
    # strip: cross-section snapshots over time
    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
    sfig, sax = plt.subplots(1, len(picks), figsize=(len(picks) * 2.3, 2.4)); sfig.patch.set_facecolor("black")
    for a, tt in zip(sax, picks):
        a.set_facecolor("black")
        fc, sv = V3.cell_faces(pos[tt].astype(np.float64) % L, L, N)
        for ct, s in zip(fc, sv):
            poly = _slice_polygon(ct, z0)
            if poly is not None and len(poly) >= 3:
                a.fill(poly[:, 0], poly[:, 1], facecolor=CMAP(np.clip((s - 5.0) / 1.0, 0, 1)),
                       alpha=0.9, edgecolor="white", lw=0.4)
        a.set_xlim(0, L); a.set_ylim(0, L); a.set_aspect("equal"); a.axis("off")
        a.set_title(f"{int(100*tt/max(T-1,1))}%", color="white", fontsize=9)
    sfig.subplots_adjust(0.01, 0.01, 0.99, 0.9, wspace=0.05)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=110, facecolor="black"); plt.close(sfig)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=100):
        for j, t in enumerate(idx):
            draw(fig, t, 20 + 100 * j / max(len(idx) - 1, 1))
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
            diag = diagnostics(pos); diag.update({k: p[k] for k in ("s0", "v0")})
            render(pos, odir, name, diag=diag)
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
