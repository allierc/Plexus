#!/usr/bin/env python
"""run_tyssue_fig5 -- Okuda Turing_Vertex Fig 5/6: MULTIPLE tubes from MULTIPLE Turing spots (coral +
tubing combined). A ~2000-cell vesicle runs a Brusselator RD to a handful of steady round activator
spots; morphogen_growth_3d then grows each cell's target volume where the activator is high, so every
spot sprouts a tube, and divide_3d (propagating chem to daughters) + reconnect_t1_3d let the tissue
proliferate and flow into the tubes. Stochastic cell cycle (cv) + live-scaled division cap keep cell
sizes uniform (Fig 5 cells stay even). ALIGNED recording (topo every=1). Strip/movie coloured by
activator (white->red), showing tubes at the red tips.

Phases (overnight plan):
  --only calib     spot-count calibration: 2000c, RD only (no growth), chi sweep -> pick ~5 spots
  --only fig5_2k   2000c multi-tube (moderate)      fig5_4k  ~4000c (better aspect)
  --only fig5_5k   ~5000c (very elongated)          fig5_combo coral+tubing showcase
"""
from __future__ import annotations
import os, sys, argparse, json, tempfile, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_t1_ops3d, tyssue_rd_ops  # noqa
from tyssue_ops3d import build_sphere_mesh
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_specfmt import write_spec
from run_tyssue_vesicle import _draw, _draw_cross
from tyssue_diag import hollow_flags
from tyssue_topology_ops3d import rings_from_flat_3d

OUT = os.path.join(HERE, "archive")
RADIUS, JITTER, SEED = 5.0, 0.16, 0


def presets():
    #  CLUSTER param SEARCH (small/short 800c/300f) for uniform-cell TUBES: the tension is v_eq cap
    #  (uniform) vs protrusion (needs bulge/localized proliferation). Grid over vth (bulge cap), rho
    #  (baseline growth), cone (spot size). cols: name n_cells frames n_spots cone grow cv rho vth K_V min max
    g = []
    for vth in (2.0, 3.0, 4.0):
        for rho in (0.0, 0.08):
            for cone in (10.0, 14.0):
                nm = f"sw_v{int(vth * 10)}_r{int(rho * 100)}_c{int(cone)}"
                g.append((nm, 800, 300, 5, cone, 0.025, 0.4, rho, vth, 3.0, 3, 14))
    g.append(("smoke_hom", 900, 200, 3, 18.0, 0.03, 0.4, 0.15, 1.35, 3.0, 3, 14))
    return g


def make_spec(name, n_cells, frames, n_spots, cone_deg, grow, cv, rho, vth, K_V, min_cyc, max_cyc, buf, cbuf):
    dt = 1.0    # cones define the activator directly (no Turing reaction) -> no CFL constraint
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": n_cells, "radius": RADIUS, "jitter": JITTER,
            "p0": 3.90, "seed": SEED, "before_frame": 1, "vseed_cv": cv},
           {"op": "cell_geometry_3d", "at": "cell"},                    # per-cell centroid (for the cones)
           {"op": "cell_rd_seed", "at": "cell", "mode": "cones", "n_spots": n_spots, "cone_deg": cone_deg},  # re-seed each frame -> tracks the N growing tips
           {"op": "morphogen_growth_3d", "at": "vertex", "cell_set": "cell", "rate": grow,
            "a_sw": 0.5, "hill": 4.0, "rho": rho, "vth_frac": vth},     # OKUDA: rate*(rho+Hill), v_eq capped -> uniform
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.90, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.05,
            "Lambda": 0.2, "K_V": K_V, "K_R": 0.02, "mu": 1.0, "dt": dt, "relax_iters": 30,
            "eta": 0.08, "cap_frac": 0.12},                            # fluid; radial ~off so tubes leave the sphere
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.28, "every": 1, "max_flips": max(40, n_cells // 8)},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": cv,
            "p0": 3.90, "every": 2, "max_div": 12, "max_div_frac": 0.03, "cell_set": "cell",
            "min_cycle": min_cyc, "max_cycle": max_cyc},                # volume-primary + bounded duration
           {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]     # ALIGNED recording
    sched = ["seed_mesh_3d", "cell_geometry_3d", "cell_rd_seed", "morphogen_growth_3d",
             "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": f"tyssue_fig5_{name}", "seed": SEED, "n_frames": frames, "dt": dt,
                       "record_cap": frames + 2, "boundary": "free", "dim": 3, "world": [16 * RADIUS] * 3},
           "sets": {"vertex": {"n": buf},
                    "cell": {"n": cbuf, "state": {"chem": {"width": 2, "integration": "first_order"},
                                                  "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, cfg


def count_spots(a, mesh, thr):
    """Connected-component count of activator > thr on the cell adjacency (number of Turing spots)."""
    es, et, ef = np.asarray(mesh["E_srce"]), np.asarray(mesh["E_trgt"]), np.asarray(mesh["E_face"])
    nF = int(mesh["nF"]); rings = rings_from_flat_3d(es, et, ef, nF)
    from collections import defaultdict, deque
    byedge = defaultdict(list)
    for k in range(len(ef)):
        byedge[(min(int(es[k]), int(et[k])), max(int(es[k]), int(et[k])))].append(int(ef[k]))
    nbr = defaultdict(set)
    for fs in byedge.values():
        for x in fs:
            for y in fs:
                if x != y:
                    nbr[x].add(y)
    hot = set(np.where(a > thr)[0].tolist()); seen = set(); nc = 0
    for s in list(hot):
        if s in seen:
            continue
        nc += 1; q = deque([s]); seen.add(s)
        while q:
            u = q.popleft()
            for v in nbr[u]:
                if v in hot and v not in seen:
                    seen.add(v); q.append(v)
    return nc


def run_all(only=None):
    for name, n_cells, frames, n_spots, cone_deg, grow, cv, rho, vth, K_V, min_cyc, max_cyc in presets():
        if only and name not in only:
            continue
        odir = os.path.join(OUT, name); os.makedirs(odir, exist_ok=True)
        verts, es, et, ef, nF = build_sphere_mesh(n_cells, RADIUS, JITTER, SEED); Nv = verts.shape[0]
        buf = int(Nv * 4.0); cbuf = int(nF * 4.0)   # ~4x seed headroom; bounds every-frame-recording memory
        print(f"[fig5] {name}: n={n_cells} frames={frames} spots={n_spots} cone={cone_deg} grow={grow} rho={rho} vth={vth} K_V={K_V} dur=({min_cyc},{max_cyc})  (Nv={Nv}, cells={nF})", flush=True)
        rec = {"name": name, "n_cells": n_cells, "frames": frames, "n_spots": n_spots, "grow": grow,
               "cv": cv, "rho": rho, "vth": vth, "K_V": K_V, "min_cyc": min_cyc, "max_cyc": max_cyc}
        try:
            sim, cfg = make_spec(name, n_cells, frames, n_spots, cone_deg, grow, cv, rho, vth, K_V, min_cyc, max_cyc, buf, cbuf)
            write_spec(cfg, os.path.join(odir, "spec.yaml"))
            Hf, out = engine_run(sim, device="cpu")
            emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist")
            posf = out["sets"]["vertex"]["pos"]; chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]

            def frame(t):
                mt = hist[min(t, len(hist) - 1)] if hist else dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)
                return mt, posf[t][:mt["Nv"]].astype(np.float64), chemf[t][:mt["nF"], 0]

            mtT, pT, aT = frame(T - 1); ext = pT.max(0) - pT.min(0)
            thr = float(np.percentile(aT, 70))
            spots = count_spots(aT, mtT, thr)
            _, _, hstat = hollow_flags(pT, mtT)
            from tyssue_diag import hollow_metric
            _, area, _ = hollow_metric(pT, mtT); area = area[area > 0]     # per-cell areas -> uniformity CV
            area_cv = float(area.std() / (area.mean() + 1e-9)) if area.size else 0.0
            # PROTRUSION: how far the tissue sticks out radially (tube stick-out). 95th pct / median radius.
            from tyssue_topology_ops3d import rings_from_flat_3d as _rff
            _rings = _rff(np.asarray(mtT["E_srce"]), np.asarray(mtT["E_trgt"]), np.asarray(mtT["E_face"]), mtT["nF"])
            _rad = np.array([np.linalg.norm(pT[r].mean(0)) if (r is not None and len(r)) else 0 for r in _rings]); _rad = _rad[_rad > 0]
            protr = float(np.percentile(_rad, 95) / (np.median(_rad) + 1e-9)) if _rad.size else 1.0
            rec.update(cells_end=int(mtT["nF"]), n_div=int(emesh.get("n_div", 0)), n_t1=int(emesh.get("n_t1", 0)),
                       extent=[round(float(x), 2) for x in ext], aspect=round(float(ext.max() / max(ext.min(), 1e-6)), 3),
                       spots=int(spots), a_std=round(float(aT.std()), 3), hollow_frac=round(hstat["frac"], 3),
                       area_cv=round(area_cv, 3), protr=round(protr, 3))
            Rmax = max(float(np.abs(frame(t)[1]).max()) for t in np.linspace(0, T - 1, 20).astype(int))
            L3 = Rmax * 1.06
            picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
            fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
            for i, t in enumerate(picks):
                mt, pt, a = frame(t); act = np.clip((a - np.percentile(a, 5)) / (np.percentile(a, 99) - np.percentile(a, 5) + 1e-9), 0, 1)
                ax3 = fig.add_subplot(2, 4, i + 1, projection="3d"); _draw(ax3, pt, mt, 3.90, azim=30, act=act, Lbox=L3)
                ax2 = fig.add_subplot(2, 4, 4 + i + 1); _draw_cross(ax2, pt, mt, 3.90, act=act, Lbox=L3, axis=1)
            fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02)
            fig.savefig(os.path.join(odir, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)
            figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black")
            axm = figm.add_subplot(111, projection="3d"); figm.subplots_adjust(0, 0, 1, 1)
            keep = np.arange(0, T, max(1, T // 150))
            wri = FFMpegWriter(fps=12, metadata={"title": name})
            with wri.saving(figm, os.path.join(odir, "movie.mp4"), dpi=110):
                for j, t in enumerate(keep):
                    mt, pt, a = frame(int(t)); act = np.clip((a - np.percentile(a, 5)) / (np.percentile(a, 99) - np.percentile(a, 5) + 1e-9), 0, 1)
                    _draw(axm, pt, mt, 3.90, azim=(2 * j) % 360, act=act, Lbox=L3); wri.grab_frame()
            plt.close(figm)
            print(f"       -> cells {nF}->{rec['cells_end']} (+{rec['n_div']} div, {rec['n_t1']} T1)  "
                  f"spots={rec['spots']} aspect={rec['aspect']} extent={rec['extent']} hollow={rec['hollow_frac']}", flush=True)
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        json.dump(rec, open(os.path.join(odir, "diag.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None)
    run_all(ap.parse_args().only)
