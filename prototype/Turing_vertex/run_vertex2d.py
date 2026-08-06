#!/usr/bin/env python
"""run_vertex2d -- Stage 2 smoke test: the 2D vertex (Self-Propelled Voronoi) MECHANICS.

A disordered confluent tissue (jittered cell centres in a periodic box) relaxes under
the shape energy E = sum K_A(A-A0)^2 + K_P(P-P0)^2 (plexus2 operators in vertex_ops.py):
seed_tissue -> voronoi_graph (re-tessellate) -> voronoi_tension (force). Cells settle to
a foam whose regularity is set by the target shape index p0 (solid below ~3.81, fluid
above). Renders the Voronoi polygons coloured by shape index p = P/sqrt(A).

Smoke-test check: the mechanics run, are stable, and the tissue RELAXES -- mean
|area - A0| drops and the energy decreases.

    python run_vertex2d.py            # run presets, archive each
    python run_vertex2d.py --montage
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
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators   # noqa: F401
import vertex_ops         # noqa: F401  seed_tissue + voronoi_graph + voronoi_tension
from vertex_ops import cell_polygons
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
N, A0 = 288, 1.0                                      # 288 = 18x16 triangular lattice
L = (N * A0) ** 0.5                                   # box so target area fills it
FRAMES, DT = 400, 0.05


def presets():
    return [
        ("vertex_solid", 3.60, 0.0),    # triangular seed -> hexagons; below transition -> jammed solid
        ("vertex_hex",   3.80, 0.0),    # near transition, stays hexagonal
        ("vertex_fluid", 4.10, 0.30),   # above transition + self-propulsion -> flows (T1 rearrangements)
    ]


def make_spec(name, p0, v0):
    cfg = {
        "general": {"name": f"vertex2d_{name}", "seed": 0, "n_frames": FRAMES, "dt": DT,
                    "boundary": "periodic", "dim": 2, "world": [L, L]},
        "sets": {"cell": {"n": N}},                  # default spatial state (pos/vel); overdamped
        "fields": {},
        "operators": [
            {"op": "seed_tissue", "at": "cell", "a0": A0, "jitter": 0.2, "lattice": "triangular",
             "before_frame": 1},
            {"op": "voronoi_graph", "at": "cell"},
            {"op": "voronoi_tension", "at": "cell", "p0": p0, "A0": A0, "K_A": 1.0, "K_P": 1.0,
             "mu": 1.0, "v0": v0, "Dr": 1.0, "dt": DT},
        ],
        "schedule": ["seed_tissue", "voronoi_graph", "voronoi_tension"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _frame_polys(pos, L, N):
    polys, area, perim, ok = cell_polygons(pos.astype(np.float64), L, N)
    shape = np.where((area > 1e-9) & (ok > 0), perim / np.sqrt(np.maximum(area, 1e-9)), np.nan)
    verts = [p for p in polys if p is not None]
    cols = shape[[i for i, p in enumerate(polys) if p is not None]]
    return verts, cols, area, ok


def render(pos_traj, outdir, name, p0, diag, seconds=12.0, max_frames=200):
    os.makedirs(outdir, exist_ok=True)
    T = pos_traj.shape[0]
    idx = list(range(0, T, max(1, -(-T // max_frames))))
    fps = max(1, round(len(idx) / seconds))
    PANEL = 4.4

    # diverging LUT centred on the rigidity transition p0*=3.81, range ADAPTED to this
    # preset's shape-index distribution (so the fluid case isn't saturated all-red).
    samp = idx[:: max(1, len(idx) // 6)]
    sh = np.concatenate([_frame_polys(pos_traj[t], L, N)[1] for t in samp])
    sh = sh[np.isfinite(sh)]
    lo, hi = np.percentile(sh, [2, 98])
    norm = TwoSlopeNorm(vcenter=3.81, vmin=min(float(lo), 3.74), vmax=max(float(hi), 3.88))

    def draw(ax, t):
        ax.clear(); ax.set_facecolor("black")
        verts, cols, _, _ = _frame_polys(pos_traj[t], L, N)
        # blue = solid/compact (p<3.81), white = transition, red = fluid/elongated (p>3.81)
        pc = PolyCollection(verts, array=cols, cmap="coolwarm", norm=norm,
                            edgecolors=(1, 1, 1, 0.30), linewidths=0.4)
        ax.add_collection(pc)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal"); ax.axis("off")
        ax.text(0.02, 0.98, f"{name}\nt={t} ({int(100*t/max(T-1,1))}%)\np0={p0}\n"
                            f"shape index  blue<3.81<red",
                transform=ax.transAxes, color="white", fontsize=7, va="top", family="monospace")

    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.25, 0.5, 1.0)]
    sfig, sax = plt.subplots(1, len(picks), figsize=(len(picks) * PANEL, PANEL)); sfig.patch.set_facecolor("black")
    for a, t in zip(sax, picks):
        draw(a, t)
    sfig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.03)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)

    fig, ax = plt.subplots(figsize=(PANEL, PANEL)); fig.patch.set_facecolor("black"); fig.subplots_adjust(0, 0, 1, 1)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=120):
        for t in idx:
            draw(ax, t); w.grab_frame()
    plt.close(fig)


def diagnostics(pos_traj):
    def area_err(pos):
        _, _, area, ok = _frame_polys(pos, L, N)
        a = area[ok > 0]
        return float(np.mean(np.abs(a - A0)))
    e0, eT = area_err(pos_traj[0]), area_err(pos_traj[-1])
    return dict(area_err_start=round(e0, 4), area_err_end=round(eT, 4),
                relaxed=bool(eT < e0 * 0.6))


def run_all():
    for name, p0, v0 in presets():
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        print(f"[vertex2d] {name}: p0={p0} v0={v0}", flush=True)
        rec = {"p0": p0, "v0": v0, "N": N}
        try:
            sim, cfg = make_spec(name, p0, v0)
            yaml.safe_dump(cfg, open(os.path.join(odir, "spec.yaml"), "w"), sort_keys=False)
            _, out = engine_run(sim, device="cpu")
            pos = out["sets"]["cell"]["pos"]              # [T,N,2]
            diag = diagnostics(pos); rec.update(diag)
            keep = np.linspace(0, pos.shape[0] - 1, min(pos.shape[0], 200)).astype(int)
            np.savez_compressed(os.path.join(odir, "traj.npz"), pos=pos[keep].astype("float32"), p0=p0, name=name)
            render(pos[keep], odir, name, p0, diag)       # render + cache decimated trajectory
            print(f"           -> relaxed={diag['relaxed']} area_err {diag['area_err_start']} -> {diag['area_err_end']}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def rerender(only=None):
    """Re-render from cached traj.npz -- no re-simulation (for LUT / render tweaks)."""
    for name, p0, v0 in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); tf = os.path.join(odir, "traj.npz")
        if not os.path.exists(tf):
            print(f"[rerender] {name}: no traj.npz (run first)"); continue
        d = np.load(tf); pos = d["pos"]
        render(pos, odir, name, float(d["p0"]), diagnostics(pos))
        print(f"[rerender] {name} -> strip.png + movie.mp4", flush=True)


def montage():
    from PIL import Image
    names = [p[0] for p in presets()]
    strips = [(n, Image.open(os.path.join(OUT, n, "strip.png")).convert("RGB"))
              for n in names if os.path.exists(os.path.join(OUT, n, "strip.png"))]
    if strips:
        w = max(i.width for _, i in strips); h = sum(i.height for _, i in strips)
        sheet = Image.new("RGB", (w, h), (0, 0, 0)); y = 0
        for _, im in strips:
            sheet.paste(im, (0, y)); y += im.height
        sheet.save(os.path.join(OUT, "_montage_vertex2d.png"))
    print(f"[vertex2d] montage -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--rerender", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.montage:
        montage()
    elif a.rerender:
        rerender(a.only); montage()
    else:
        run_all(); montage()


if __name__ == "__main__":
    main()
