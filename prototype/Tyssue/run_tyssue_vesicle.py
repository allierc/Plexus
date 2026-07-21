#!/usr/bin/env python
"""run_tyssue_vesicle -- Goal 1bis: the 3D (surface) vertex model. A closed epithelial VESICLE (a
spherical half-edge mesh) relaxes to force balance under the 3D AVM shape energy + a lumen-volume
constraint (tyssue_ops3d.py): seed_mesh_3d -> shape_energy_3d. A jittered sphere relaxes to a
uniform epithelial shell; the lumen term keeps it inflated against surface tension. This is the
true-vertex-model sibling of Turing_vertex's spherical-Voronoi vesicle, and the substrate for 3D
budding/tubulation. 3D render (rotating), strip + movie.

    python run_tyssue_vesicle.py
    python run_tyssue_vesicle.py --montage
"""
from __future__ import annotations
import os, sys, argparse, json, tempfile, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators   # noqa: F401
import tyssue_ops3d        # noqa: F401  seed_mesh_3d + shape_energy_3d
from tyssue_ops3d import build_sphere_mesh, face_polygons_3d
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_specfmt import write_spec

OUT = os.path.join(HERE, "archive")
NCELLS, RADIUS, JITTER, SEED = 150, 5.0, 0.18, 0
FRAMES = 120


def presets():
    #     name            p0    grow    divide n_cells frames
    return [("vesicle_3d",    3.72, 0.0,     False, 150,  120),
            ("vesicle_grow",  3.72, 0.003,   False, 150,  120),   # grow quasi-statically; tension keeps it smooth
            ("vesicle_divide", 3.72, 0.003,  True,  150,  220),   # grow AND divide gradually -> proliferating vesicle
            ("vesicle_fig4",  3.72, 0.00055, True,  600, 1000)]   # Turing fig4 scale: 600->~3000 (5x) over 1000 frames


def make_spec(name, p0, buf, grow_rate, divide, n_cells, frames):
    ops = [
        {"op": "seed_mesh_3d", "at": "vertex", "n_cells": n_cells, "radius": RADIUS,
         "jitter": JITTER, "p0": p0, "seed": SEED, "before_frame": 1},
    ]
    sched = ["seed_mesh_3d"]
    if grow_rate > 0:
        ops.append({"op": "vesicle_growth", "at": "vertex", "rate": grow_rate, "every": 1})
        sched.append("vesicle_growth")           # ramp the per-cell TARGETS before the force step reads them
    # Expansion EMERGES from the per-cell volume elasticity: growth only ramps each cell's target volume,
    # and shape_energy_3d's force balance inflates every cell locally (no vertex moved by hand). During
    # growth give it more relax iters per frame so it tracks the ramping targets quasi-statically.
    relax = 26 if divide else (20 if grow_rate > 0 else 8)   # division needs extra relaxation to round daughters
    lam = 0.5 if divide else 0.1                  # line tension (physical surface energy) rounds the daughters
    gam = 0.1 if divide else 0.0                  # cortical contractility (1/2)Gamma*P^2 -> rounds cells (emergent)
    ops.append({"op": "shape_energy_3d", "at": "vertex", "p0": p0, "K_A": 1.0, "K_P": 1.0,
                "Lambda": lam, "Gamma": gam, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": 1.0,
                "relax_iters": relax, "eta": 0.08, "cap_frac": 0.12})
    sched.append("shape_energy_3d")
    if divide:
        mxd = max(10, n_cells // 20)              # divisions/call scale with tissue size (clears each wave)
        ops.append({"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "p0": p0,
                    "every": 2, "max_div": mxd})  # gradual, staggered volume-doubling cell cycle (Turing fig4)
        sched.append("divide_3d")                # split cells whose wedge volume doubled (edge-midpoint septum)
        ops.append({"op": "topo_snapshot_3d", "at": "vertex"})
        sched.append("topo_snapshot_3d")         # record the (changing) mesh each frame for rendering
    cfg = {
        "general": {"name": f"tyssue_ves_{name}", "seed": SEED, "n_frames": frames or FRAMES, "dt": 1.0,
                    "boundary": "free", "dim": 3, "world": [6 * RADIUS, 6 * RADIUS, 6 * RADIUS]},
        "sets": {"vertex": {"n": buf}},
        "fields": {},
        "operators": ops,
        "schedule": sched,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _mesh_from_build(n_cells=NCELLS):
    verts, es, et, ef, nF = build_sphere_mesh(n_cells, RADIUS, JITTER, SEED)
    return dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, verts0=verts, Nv=verts.shape[0])


INNER = 0.82                                                    # basal radius fraction (thin monolayer wall)


def _draw(ax, pos, mesh, p0, azim, act=None, inner=INNER, Lbox=None):
    """3D monolayer: each cell is a prism -- an apical face (outer), a basal face (inner), and lateral
    walls. Cells are coloured by ACTIVATION with the Turing white->red LUT (activation 0 -> white); the
    apical faces are bevelled and edged black so the cells read as raised 3D blocks (many faces each)."""
    ax.clear(); ax.set_facecolor("black")
    polys, area, perim, shape = face_polygons_3d(pos, mesh)
    if act is None:
        act = np.zeros(len(polys))                              # activation = 0 -> white (Goal 2 RD lights the red)
    cmap = plt.cm.Reds
    faces3d, cols = [], []
    for f, ap in enumerate(polys):
        base = cmap(float(np.clip(act[f], 0.0, 1.0)))
        bp = ap * inner                                          # basal ring (apical scaled toward the sphere centre)
        wall = tuple(0.72 * np.array(base[:3])) + (1.0,)        # lateral wall: lightly shadowed (only shows at silhouette)
        k = len(ap)
        for i in range(k):                                      # lateral walls first (drawn behind the apical cap)
            faces3d.append(np.array([ap[i], ap[(i + 1) % k], bp[(i + 1) % k], bp[i]])); cols.append(wall)
        faces3d.append(ap); cols.append(base)                   # apical face: shared mesh vertices -> tiles edge-to-edge,
        #                                                         a closed OPAQUE surface (no bevel gaps to see through)
    pc = Poly3DCollection(faces3d, facecolors=cols, edgecolors=(0, 0, 0, 1.0), linewidths=0.25)
    ax.add_collection3d(pc)
    L = Lbox if Lbox is not None else RADIUS * 1.0              # tight 3D box so the sphere renders to scale with the 2D ring
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
    ax.set_box_aspect((1, 1, 1)); ax.axis("off"); ax.view_init(elev=18, azim=azim)


def _draw_cross(ax, pos, mesh, p0, level=0.0, act=None, inner=INNER, Lbox=None):
    """Cross-section of the MONOLAYER at z=level. Each junction edge crossing the plane gives one
    apical (outer) point; its basal counterpart is that point scaled inward. Connecting them draws the
    LATERAL wall of the junction, so the band is partitioned into individual cells (white, activation
    LUT), edged black; the hollow centre is the lumen."""
    from matplotlib.patches import Polygon as MplPoly
    ax.clear(); ax.set_facecolor("black")
    es, et = mesh["E_srce"], mesh["E_trgt"]
    pts = []
    for e in range(len(es)):
        a = pos[int(es[e])]; b = pos[int(et[e])]
        if (a[2] - level) * (b[2] - level) < 0:
            fr = (level - a[2]) / (b[2] - a[2]); pts.append((a + fr * (b - a))[:2])
    if pts:
        ap = np.array(pts); ap = ap[np.argsort(np.arctan2(ap[:, 1], ap[:, 0]))]
        ba = ap * inner                                                   # basal = apical scaled inward
        for i in range(len(ap)):                                          # one filled quad per cell (white), edged black
            j = (i + 1) % len(ap)
            quad = np.array([ba[i], ap[i], ap[j], ba[j]])
            ax.add_patch(MplPoly(quad, closed=True, facecolor="white", edgecolor="black", lw=0.5, zorder=1))
    L = Lbox if Lbox is not None else RADIUS * 2.1              # wider box so the ring renders to scale with the 3D sphere (mplot3d shrinks it)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.axis("off")


def diagnostics(pos_traj, mesh, p0):
    def stats(pos):
        _, area, _, shape = face_polygons_3d(pos, mesh)
        # sphericity of the shell: std/mean of vertex radius
        rad = np.linalg.norm(pos, axis=1)
        return float(np.std(rad) / max(np.mean(rad), 1e-9)), float(np.nanmean(shape)), \
            float(np.nanstd(shape))
    r0, q0, sd0 = stats(pos_traj[0]); rT, qT, sdT = stats(pos_traj[-1])
    return dict(rough_start=round(r0, 4), rough_end=round(rT, 4),
                shape_start=round(q0, 4), shape_end=round(qT, 4),
                shape_std_start=round(sd0, 4), shape_std_end=round(sdT, 4))


def run_all(only=None):
    for name, p0, grow_rate, divide, n_cells, frames in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        mesh0 = _mesh_from_build(n_cells); Nv0 = mesh0["Nv"]
        buf = int(Nv0 * 5.5) if divide else Nv0          # vertex-buffer headroom for division (~5x cells)
        print(f"[tyssue_ves] {name}: p0={p0} grow={grow_rate} divide={divide}  (Nv={Nv0}, cells={mesh0['nF']}, buf={buf}, frames={frames})", flush=True)
        rec = {"name": name, "p0": p0, "grow_rate": grow_rate, "divide": divide, "Nv": Nv0, "cells": mesh0["nF"]}
        try:
            sim, cfg = make_spec(name, p0, buf, grow_rate, divide, n_cells, frames)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            Hf, out = engine_run(sim, device="cpu")
            emesh = Hf.level("vertex")._mesh
            hist = emesh.get("hist")                      # per-frame mesh snapshots (divide preset), else None
            pos_full = out["sets"]["vertex"]["pos"]       # [T, buf, 3]
            T = pos_full.shape[0]

            def frame(t):                                 # (mesh, positions) at frame t, sliced to that frame's Nv
                mt = hist[min(t, len(hist) - 1)] if hist else mesh0
                return mt, pos_full[t][:mt["Nv"]].astype(np.float64)

            def rough(pf):
                rad = np.linalg.norm(pf, axis=1); return float(np.std(rad) / max(np.mean(rad), 1e-9))

            def qmean(mt, pf):
                _, _, _, sh = face_polygons_3d(pf, mt); return float(np.nanmean(sh))

            m0, p0f = frame(0); mT, pTf = frame(T - 1)
            rec.update(rough_start=round(rough(p0f), 4), rough_end=round(rough(pTf), 4),
                       shape_start=round(qmean(m0, p0f), 4), shape_end=round(qmean(mT, pTf), 4),
                       radius_start=round(float(np.linalg.norm(p0f, axis=1).mean()), 3),
                       radius_end=round(float(np.linalg.norm(pTf, axis=1).mean()), 3),
                       cells_start=int(m0["nF"]), cells_end=int(mT["nF"]), n_div=int(emesh.get("n_div", 0)))
            if hist:                                    # verify the closed surface survived every division (Euler=2)
                from tyssue_topology_ops3d import rings_from_flat_3d, _check_closed
                okc, Vc, Ec, Fc, euc = _check_closed(
                    rings_from_flat_3d(mT["E_srce"], mT["E_trgt"], mT["E_face"], mT["nF"]))
                rec.update(closed=bool(okc), euler=int(euc))
            keep = np.linspace(0, T - 1, min(T, 120)).astype(int)
            # frame the view to the LARGEST extent over the trajectory (a growing shell never clips);
            # keep the 3D:2D box ratio (1.06 : 2.23) so the sphere and the ring stay to scale with each other
            Rmax = max(float(np.linalg.norm(frame(t)[1], axis=1).max()) for t in keep)
            L3, L2 = Rmax * 1.06, Rmax * 2.23
            np.savez_compressed(os.path.join(odir, "traj.npz"), pos=pos_full[keep].astype("float32"), p0=p0)
            # strip: two rows -- top = 3D shell, bottom = 2D cross-section through the lumen, 4 times
            fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
            picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
            for i, t in enumerate(picks):
                mt, pt = frame(t)
                ax3 = fig.add_subplot(2, 4, i + 1, projection="3d")
                _draw(ax3, pt, mt, p0, azim=30, Lbox=L3)
                ax2 = fig.add_subplot(2, 4, 4 + i + 1)
                _draw_cross(ax2, pt, mt, p0, Lbox=L2)
            fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02)
            fig.savefig(os.path.join(odir, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
            # movie: evolve + rotate
            figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black")
            axm = figm.add_subplot(111, projection="3d"); figm.subplots_adjust(0, 0, 1, 1)
            mkeep = keep[::max(1, len(keep) // 60)]      # cap the movie at ~60 frames (large meshes render slowly)
            wri = FFMpegWriter(fps=max(1, round(len(mkeep) / 8.0)), metadata={"title": name})
            with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=110):
                for j, t in enumerate(mkeep):
                    mt, pt = frame(t)
                    _draw(axm, pt, mt, p0, azim=(2 * j) % 360, Lbox=L3)
                    wri.grab_frame()
            plt.close(figm)
            extra = f"  cells {rec['cells_start']}->{rec['cells_end']} (+{rec['n_div']} div)" if divide else ""
            print(f"           -> R {rec['radius_start']}->{rec['radius_end']}  roughness "
                  f"{rec['rough_start']}->{rec['rough_end']}  <q> {rec['shape_start']}->{rec['shape_end']}{extra}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def montage():
    print("[tyssue_ves] (single preset)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.montage:
        montage()
    else:
        run_all(a.only)


if __name__ == "__main__":
    main()
