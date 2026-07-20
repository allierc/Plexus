#!/usr/bin/env python
"""run_coupled2d -- Stage 4: the FULL coupled Turing x vertex model (Okuda et al. 2018), 2D.

Closes the loop across the two halves on ONE `cell` set carrying three independently
integrated state blocks (the engine's multi-block extension):

    pos  (coordinate)  <- vertex_tension_2d   shape-energy force toward the growing a0
    chem = [act,inh]   <- graph_diffuse + react   Turing RD on the Voronoi/Delaunay graph
    a0   (target area) <- growth   activator (mitogen) grows a0 (Hill, Eqs 7-8)

and division on a FIXED BUFFER with occupancy: divide_2x wakes a dormant slot when a0
reaches 2x its base (Okuda v_th=(4/3)v_ref). Free (finite, ghost-bounded) tissue, so it
grows and deforms.

    Fig 4 (hysteresis / patterning on a growing tissue) first, then Fig 5 (the full couple).

    python run_coupled2d.py            # run presets, archive each
    python run_coupled2d.py --rerender # re-render from cached traj.npz
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
from matplotlib.collections import PolyCollection
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators   # noqa: F401
import turing_ops         # noqa: F401  graph_diffuse + react (INTEGRAND="chem")
import coupled_ops        # noqa: F401  coupled_seed_2d + voronoi_graph_2d + growth + divide_2x + vertex_tension_2d
from coupled_ops import cell_polygons_2d
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive")
PANEL = 4.4
A0 = 1.0                                           # base target area (division at 2x)


def presets():
    # Brusselator RD (round activator spots, robust from noise). chi sets the spot scale.
    # growth lam_ref sets how fast the activator inflates a0; divide_2x at ratio*A0.
    #  - fig4_grow  : activator-INDEPENDENT uniform growth (rho_lam=1, lam by activator off) + patterning
    #  - fig5_couple: activator-DRIVEN growth (mitogen) -> non-uniform expansion, deformation
    BR = {"implementation": "brusselator", "gamma": 0.05, "A": 1.0, "B": 3.0}
    # 2D filled-disc analog of Fig 4: activator-INDEPENDENT uniform growth + Turing RD. Validates the
    # coupling + growing-domain patterning. (The literal Fig 4/5 are 3D hollow monolayer vesicles --
    # see run_coupled3d.py; tubulation is intrinsically 3D so the mitogen/differential-growth variant
    # is done there, not on a flat disc.)
    def GS(F, kk):
        return {"implementation": "gray_scott", "F": F, "kk": kk}
    # all share the couple2d_grow coupling (uniform growth + division); only the RD REGIME varies,
    # to show the growing-disc machinery reproduces the static-shell Turing patterns (archive/shell_*).
    # Each regime is emitted twice: plain, and `_lloyd` (planar Lloyd relaxation -> uniform hex cells).
    grow = dict(n0=400, buffer=5200, radius=11.0, dt=0.2, p0=3.9,
                lam_ref=0.004, rho_lam=1.0, a_sw=1.0, hill=2.0, ratio=2.0)
    regimes = [
        dict(name="grow",      frames=4000, chi=0.30, d_a=0.05, d_h=0.5,  react=BR),                        # round spots
        dict(name="bigspots",  frames=4000, chi=0.75, d_a=0.05, d_h=0.5,  react=BR),                        # larger wavelength
        dict(name="coral",     frames=9000, chi=0.28, d_a=0.08, d_h=0.16, react=GS(0.058, 0.063), seed_mode="scatter"),
        dict(name="labyrinth", frames=9000, chi=0.28, d_a=0.08, d_h=0.16, react=GS(0.029, 0.054), seed_mode="scatter"),
        dict(name="holes",     frames=9000, chi=0.28, d_a=0.08, d_h=0.16, react=GS(0.039, 0.058), seed_mode="scatter"),
        dict(name="chi_small", frames=9000, chi=0.20, d_a=0.08, d_h=0.16, react=GS(0.037, 0.060), seed_mode="scatter"),  # small scale
    ]
    out = []
    for r in regimes:
        rr = dict(grow, **r); nm = rr.pop("name")
        out.append(dict(rr, name=f"couple2d_{nm}"))                            # plain (vertex_tension only)
        out.append(dict(rr, name=f"couple2d_{nm}_lloyd", k_lloyd=2.0))         # + planar Lloyd -> uniform cells
    return out


def _ops2d(p, A, B, seed_mode):
    ops = [
        {"op": "coupled_seed_2d", "at": "cell", "radius": p["radius"], "a0": A0,
         "a_mean": A, "h_mean": B / A, "noise": 0.04, "seed_mode": seed_mode,
         "seed_frac": p.get("seed_frac", 0.04), "before_frame": 1},
        {"op": "voronoi_graph_2d", "at": "cell"},
        {"op": "graph_diffuse", "at": "cell", "d_a": p["d_a"], "d_h": p["d_h"], "chi": p["chi"],
         "norm": p.get("norm", False)},
        {"op": "react", "at": "cell", **p["react"]},
        {"op": "growth", "at": "cell", "lam_ref": p["lam_ref"], "rho_lam": p["rho_lam"],
         "a_sw": p["a_sw"], "hill": p["hill"]},
        {"op": "divide_2x", "at": "cell", "ratio": p["ratio"]},
        {"op": "vertex_tension_2d", "at": "cell", "p0": p["p0"], "K_A": 1.0, "K_P": 0.5, "mu": 0.2},
    ]
    if p.get("k_lloyd", 0) > 0:                      # planar Lloyd -> uniform hexagonal cells
        ops.append({"op": "lloyd_2d", "at": "cell", "k_lloyd": p["k_lloyd"]})
    return ops


def make_spec(p):
    W = 6.0 * p["radius"]                            # generous free box; tissue grows into it
    A = p["react"].get("A", 1.0); B = p["react"].get("B", 3.0)
    seed_mode = p.get("seed_mode", "noise")         # noise (Brusselator/GM) | scatter (Gray-Scott)
    cfg = {
        "general": {"name": f"coupled2d_{p['name']}", "seed": 0, "n_frames": p["frames"], "dt": p["dt"],
                    "boundary": "free", "dim": 2, "world": [W, W]},
        "sets": {"cell": {"n": p["n0"], "buffer": p["buffer"], "state": {
            "pos":  {"width": 2, "integration": "second_order_coordinate", "boundary": "free"},
            "vel":  {"width": 2, "integration": "second_order_rate", "boundary": "free"},
            "chem": {"width": 2, "integration": "first_order", "boundary": "free"},
            "a0":   {"width": 1, "integration": "first_order", "boundary": "free"}}}},
        "fields": {},
        "operators": _ops2d(p, A, B, seed_mode),
        "schedule": [o["op"] for o in _ops2d(p, A, B, seed_mode)],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def _frame(pos, act, occ, pad=0.4):
    """Voronoi polygons of the LIVE cells, coloured by activator."""
    live = occ > 0
    pl = pos[live].astype(np.float64)
    polys, area, ok = cell_polygons_2d(pl, pad)
    a = act[live]
    verts = [p for p in polys if p is not None]
    cols = a[[i for i, p in enumerate(polys) if p is not None]]
    return verts, cols


def render(pos_t, act_t, occ_t, outdir, name, seconds=12.0, max_frames=180):
    os.makedirs(outdir, exist_ok=True)
    T = pos_t.shape[0]
    idx = list(range(0, T, max(1, -(-T // max_frames))))
    fps = max(1, round(len(idx) / seconds))
    # fixed extent = final tissue span (so growth reads as growth, not zoom)
    lf = occ_t[-1] > 0
    c = pos_t[-1][lf].mean(0)
    box = float(np.ptp(pos_t[-1][lf], axis=0).max()) * 1.15 + 2.0
    amax = float(np.percentile(act_t[occ_t > 0], 99))
    amin = float(np.percentile(act_t[occ_t > 0], 1))

    def draw(ax, t):
        ax.clear(); ax.set_facecolor("black")
        verts, cols = _frame(pos_t[t], act_t[t], occ_t[t])
        # white->red activator LUT (as in the Turing archive), BLACK cell-delimiting lines on top
        pc = PolyCollection(verts, array=cols, cmap="Reds", edgecolors=(0, 0, 0, 0.55), linewidths=0.35)
        pc.set_clim(amin, amax); ax.add_collection(pc)
        ax.set_xlim(c[0] - box / 2, c[0] + box / 2); ax.set_ylim(c[1] - box / 2, c[1] + box / 2)
        ax.set_aspect("equal"); ax.axis("off")
        n = int((occ_t[t] > 0).sum())
        ax.text(0.02, 0.98, f"{name}\nt={t} ({int(100*t/max(T-1,1))}%)\nN={n} cells\nactivator (red)",
                transform=ax.transAxes, color="white", fontsize=8, va="top", family="monospace")

    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
    sfig, sax = plt.subplots(1, 4, figsize=(4 * PANEL, PANEL)); sfig.patch.set_facecolor("black")
    for a_, t in zip(sax, picks):
        draw(a_, t)
    sfig.subplots_adjust(0.005, 0.005, 0.995, 0.995, wspace=0.03)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=120, facecolor="black"); plt.close(sfig)

    fig, ax = plt.subplots(figsize=(PANEL, PANEL)); fig.patch.set_facecolor("black"); fig.subplots_adjust(0, 0, 1, 1)
    w = FFMpegWriter(fps=fps, metadata={"title": name})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=120):
        for t in idx:
            draw(ax, t); w.grab_frame()
    plt.close(fig)


def diagnostics(act_t, occ_t):
    n0, nT = int((occ_t[0] > 0).sum()), int((occ_t[-1] > 0).sum())
    aT = act_t[-1][occ_t[-1] > 0]
    return dict(n_start=n0, n_end=nT, grew=bool(nT > n0 * 1.2),
                v_std=round(float(aT.std()), 3), patterned=bool(aT.std() > 0.05 and not np.isnan(aT).any()),
                nan=bool(np.isnan(aT).any()))


def run_all(only=None):
    for p in presets():
        if only and p["name"] not in only:
            continue
        odir = os.path.join(OUT, p["name"]); os.makedirs(odir, exist_ok=True)
        print(f"[coupled2d] {p['name']}: n0={p['n0']} chi={p['chi']} lam_ref={p['lam_ref']}", flush=True)
        rec = {k: p[k] for k in ("n0", "buffer", "chi", "dt", "frames", "lam_ref", "rho_lam", "ratio")}
        try:
            sim, cfg = make_spec(p)
            yaml.safe_dump(cfg, open(os.path.join(odir, "spec.yaml"), "w"), sort_keys=False)
            _, out = engine_run(sim, device="cpu")
            cell = out["sets"]["cell"]                     # coordinate 'pos' is top-level; extra blocks under 'state'
            pos = cell["pos"]; act = cell["state"]["chem"][..., 0]; occ = cell["occ"]
            diag = diagnostics(act, occ); rec.update(diag)
            T = pos.shape[0]; keep = np.linspace(0, T - 1, min(T, 180)).astype(int)
            np.savez_compressed(os.path.join(odir, "traj.npz"), pos=pos[keep].astype("float32"),
                                act=act[keep].astype("float32"), occ=occ[keep].astype("float32"), name=p["name"])
            render(pos[keep], act[keep], occ[keep], odir, p["name"])
            print(f"     -> N {diag['n_start']}->{diag['n_end']} grew={diag['grew']} "
                  f"patterned={diag['patterned']}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


def rerender(only=None):
    for p in presets():
        if only and p["name"] not in only:
            continue
        odir = os.path.join(OUT, p["name"]); tf = os.path.join(odir, "traj.npz")
        if not os.path.exists(tf):
            print(f"[rerender] {p['name']}: no traj.npz"); continue
        d = np.load(tf)
        render(d["pos"], d["act"], d["occ"], odir, str(d["name"]))
        print(f"[rerender] {p['name']} -> strip.png + movie.mp4", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerender", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    if a.rerender:
        rerender(a.only)
    else:
        run_all(a.only)


if __name__ == "__main__":
    main()
