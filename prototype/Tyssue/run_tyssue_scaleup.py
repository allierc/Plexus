#!/usr/bin/env python
"""Scale-up driver (cluster-ready, --only <preset>): take the SOLVED homogenised+rounded recipe
(H1-H3 uniformity vol_cv~0.12 + H10 surface-tension rounding shape_idx~3.81) to Fig-5 scale (~2000
cells). Each preset writes archive/<preset>/{spec.yaml, strip.png, movie.mp4, diag.json}. Submit with
    TV_SCRIPT=run_tyssue_scaleup.py python cluster_gen.py n2000_round n2000_coral
Runs on a dedicated L4 node (sims are scipy/CPU-bound; the win is no devcontainer core contention)."""
from __future__ import annotations
import os, sys, json, tempfile, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross, make_movie_axes, draw_movie_frame
from tyssue_diag import hollow_metric, hollow_flags
from tyssue_topology_ops3d import rings_from_flat_3d
import torch

R, J, SEED = 5.0, 0.18, 0
P0, LAM, GAM, K_V = 3.5, 3.0, 0.4, 2.0                       # H10 rounded-cell winner

# preset: n_cells, frames, coral(bool). ~2000-cell Fig-5 start; homogenised + rounded, RD is pure colouring.
# SCALE FINDING (2026-07-22): n2000_round/coral both blew up identically (->7128 cells, vol_cv 1.63,
# hollow 0.33) -- the recipe over-proliferates (rho=1 runaway) and the fixed relax_iters=30 can't keep
# 7000+ cells uniform/unbuckled. Fig5 does morphogenesis on ~2000 cells, NOT a 4x bulk proliferation.
# Next hypotheses (small steps): H-bound = stop proliferation near ~2500 (fewer frames) -> stays clean;
# H-relax = more relaxation per frame handles the higher DOF; H-stiff = radial stiffness resists buckling.
PRESETS = {
    "n2000_round": dict(n=2000, frames=220, coral=False),
    "n2000_coral": dict(n=2000, frames=220, coral=True),
    "n2000_bound": dict(n=2000, frames=70, coral=False),                   # H-bound: bounded proliferation
    "n2000_relax": dict(n=2000, frames=220, coral=False, relax=80),        # H-relax: 80 iters (vs 30)
    "n2000_stiff": dict(n=2000, frames=220, coral=False, relax=80, K_R=2.0),  # H-stiff: + radial anti-buckle
    # SEED SWEEP (smaller steps, 2026-07-22): where does uniformity break vs initial cell count? Fixed
    # frames=150 so each seed gets the same homogenisation treatment; read vol_cv/hollow/shape_idx trend.
    "seed500":  dict(n=500,  frames=150, coral=False),
    "seed1000": dict(n=1000, frames=150, coral=False),
    "seed1500": dict(n=1500, frames=150, coral=False),
    "seed2000": dict(n=2000, frames=150, coral=False),
    # H-slow (2026-07-22): the seed sweep + n2000_bound showed vol_cv tracks CELL COUNT / proliferation
    # rate, not relaxation (relax/stiff falsified). The old clean recipe hit vol_cv 0.12 at 1778 cells via
    # SLOW growth (150 seed, 500 frames). So reach ~2000 cells slowly and it should stay clean:
    "grow2k":   dict(n=300,  frames=400, coral=False),                  # small seed, many frames -> ~2100
    "gentle2k": dict(n=2000, frames=150, coral=False, mdf=0.008),       # full seed, GENTLE divisions/frame
}


def make(n, frames, coral, relax=30, K_R=0.4, mdf=0.03):
    verts, es, et, ef, nF = build_sphere_mesh(n, R, J, SEED); Nv = verts.shape[0]
    dt = 0.02 if coral else 1.0
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": n, "radius": R, "jitter": J, "p0": P0, "seed": SEED, "before_frame": 1, "vseed_cv": 0.15},
           {"op": "cell_geometry_3d", "at": "cell"}]
    sched = ["seed_mesh_3d", "cell_geometry_3d"]
    if coral:
        ops += [{"op": "cell_adjacency", "at": "cell"},
                {"op": "seed_cell_rd", "at": "cell", "seed": SEED, "before_frame": 3, "mode": "scatter", "seed_frac": 0.06},
                {"op": "cell_diffuse", "at": "cell", "d_a": 0.08, "d_h": 0.16, "chi": 1.3},
                {"op": "cell_react", "at": "cell", "model": "gray_scott", "F": 0.055, "kk": 0.062, "rate": 1.0}]
        sched += ["cell_adjacency", "seed_cell_rd", "cell_diffuse", "cell_react"]
    ops += [{"op": "morphogen_growth_3d", "at": "vertex", "cell_set": "cell", "rate": 0.03, "a_sw": 50.0, "hill": 4.0, "rho": 1.0, "vth_frac": 1.4},
            {"op": "shape_energy_3d", "at": "vertex", "p0": P0, "K_A": 1.0, "K_P": 1.0, "Gamma": GAM, "Lambda": LAM, "K_V": K_V, "K_R": K_R, "mu": 1.0, "dt": dt, "relax_iters": relax, "eta": 0.08, "cap_frac": 0.12},
            {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 60},
            {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.15, "p0": P0, "every": 2, "max_div": 60, "max_div_frac": mdf, "cell_set": "cell", "min_cycle": 4, "max_cycle": 12},
            {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched += ["morphogen_growth_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": f"tyssue_su", "seed": SEED, "n_frames": frames, "dt": dt, "record_cap": frames + 2, "boundary": "free", "dim": 3, "world": [10 * R] * 3},
           "sets": {"vertex": {"n": int(Nv * 6)}, "cell": {"n": int(nF * 6), "state": {"chem": {"width": 2, "integration": "first_order"}, "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, cfg, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


def metrics(mt, pt):
    es, et, ef, nF = np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"]), mt["nF"]
    area, perim, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF)
    area, perim, vf = area.numpy(), perim.numpy(), vf.numpy(); ok = area > 1e-9
    shape = float(np.median(perim[ok] / np.sqrt(area[ok] + 1e-12)))
    rings = rings_from_flat_3d(es, et, ef, nF); nsides = np.array([len(r) if r is not None else 0 for r in rings])
    sliver = float((nsides[ok & (nsides > 0)] <= 4).mean())
    vc = float(vf[np.abs(vf) > 1e-9].std() / (np.abs(vf[np.abs(vf) > 1e-9].mean()) + 1e-9))
    return shape, sliver, vc


def do(preset):
    p = PRESETS[preset]; OUT = os.path.join(HERE, "archive", preset); os.makedirs(OUT, exist_ok=True)
    sim, cfg, mesh0 = make(p["n"], p["frames"], p["coral"], p.get("relax", 30), p.get("K_R", 0.4), p.get("mdf", 0.03)); write_spec(cfg, os.path.join(OUT, "spec.yaml"))
    rec = {"name": preset, **{k: p[k] for k in p}}
    try:
        Hf, out = engine_run(sim, device="cpu")
        emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
        def frame(t):
            mt = hist[min(t, len(hist) - 1)] if hist else mesh0
            return mt, posf[t][:mt["Nv"]].astype(np.float64), chemf[t][:mt["nF"], 0]
        mtT, pT, aT = frame(T - 1); sh, sl, vc = metrics(mtT, pT)
        rec.update(cells_end=int(mtT["nF"]), shape_idx=round(sh, 3), sliver_frac=round(sl, 3), vol_cv=round(vc, 3),
                   hollow_frac=round(float(hollow_flags(pT, mtT)[2]["frac"]), 3), rounded=True)
        Rmax = max(float(np.linalg.norm(frame(t)[1], axis=1).max()) for t in np.linspace(0, T - 1, 20).astype(int)); L3, L2 = Rmax * 1.06, Rmax * 2.23
        if p["coral"]:
            lo, hi = float(np.percentile(aT, 5)), float(np.percentile(aT, 99) + 1e-6)
            col = lambda mt, pt, a: np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        else:
            def col(mt, pt, a):
                _, ar, _ = hollow_metric(pt, mt); med = np.median(ar[ar > 0]) if (ar > 0).any() else 1.0
                return np.clip(np.abs(ar[:mt["nF"]] - med) / (med + 1e-9), 0, 1)
        fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
        for i, t in enumerate([int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]):
            mt, pt, a = frame(t); c = col(mt, pt, a)
            ax3 = fig.add_subplot(2, 4, i + 1, projection="3d"); _draw(ax3, pt, mt, P0, azim=30, act=c, Lbox=L3)
            ax2 = fig.add_subplot(2, 4, 4 + i + 1); _draw_cross(ax2, pt, mt, P0, act=c, Lbox=L2)
        fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02); fig.savefig(os.path.join(OUT, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
        figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black"); axm, axin = make_movie_axes(figm)
        keep = np.arange(0, T, max(1, T // 150)); wri = FFMpegWriter(fps=12, metadata={"title": preset})
        with wri.saving(figm, os.path.join(OUT, "movie.mp4"), dpi=110):
            for j, t in enumerate(keep):
                mt, pt, a = frame(int(t)); draw_movie_frame(axm, axin, pt, mt, P0, (2 * j) % 360, col(mt, pt, a), L3, L2); wri.grab_frame()
        plt.close(figm)
        print(f"[{preset}] cells {p['n']}->{rec['cells_end']}  shape_idx={rec['shape_idx']} sliver={rec['sliver_frac']} "
              f"vol_cv={rec['vol_cv']} hollow={rec['hollow_frac']}", flush=True)
    except Exception as e:
        rec["error"] = repr(e); traceback.print_exc()
    json.dump(rec, open(os.path.join(OUT, "diag.json"), "w"), indent=1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--only":
        do(args[1])
    else:
        for k in (args or list(PRESETS)):
            do(k)
