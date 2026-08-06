#!/usr/bin/env python
"""Metric-driven PARALLEL parameter sweep for Fig-5 multi-tube morphogenesis (64-core node). Each config
runs the cones->morphogen-growth->divide->T1 pipeline (aligned recording, no render) and is scored by
DEDICATED METRICS:
    area_cv  cell-area uniformity (LOWER better; Okuda Fig 5 ~ 0.25-0.35)
    protr    radial protrusion 95pct/median (HIGHER better; tubes stick out; ~1 = sphere, >1.5 = tubes)
    hollow   mesh cleanliness (LOWER better, <0.05)
    spots    activator foci that survived
    cells    final count
GOAL score rewards low area_cv + high protr + low hollow. Small/short runs for fast search; scale the
winners. Runs N configs concurrently via ProcessPoolExecutor.

    python sweep_fig5.py                 # run the default grid
    python sweep_fig5.py --workers 8 --cells 800 --frames 300
"""
from __future__ import annotations
import os, sys, json, argparse, itertools, tempfile, traceback
os.environ.setdefault("OMP_NUM_THREADS", "6"); os.environ.setdefault("MKL_NUM_THREADS", "6")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
from concurrent.futures import ProcessPoolExecutor, as_completed

RADIUS, JITTER, SEED = 5.0, 0.16, 0


def _one(cfgp):
    """Worker: build + run one config, return metrics (no render). Runs in its own process."""
    import numpy as np, yaml
    import plexus.operators  # noqa
    import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d  # noqa
    import plexus.schema as S
    from plexus.engine import run as engine_run
    from tyssue_ops3d import build_sphere_mesh
    from tyssue_diag import hollow_flags, hollow_metric
    from tyssue_topology_ops3d import rings_from_flat_3d
    import run_tyssue_fig5 as F
    name, P = cfgp["name"], cfgp
    try:
        n_cells, frames = P["cells"], P["frames"]
        verts, es, et, ef, nF = build_sphere_mesh(n_cells, RADIUS, JITTER, SEED); Nv = verts.shape[0]
        buf, cbuf = int(Nv * 4.0), int(nF * 4.0)
        ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": n_cells, "radius": RADIUS, "jitter": JITTER,
                "p0": 3.90, "seed": SEED, "before_frame": 1, "vseed_cv": 0.4},
               {"op": "cell_geometry_3d", "at": "cell"},
               {"op": "seed_cell_rd", "at": "cell", "mode": "cones", "n_spots": P["spots"], "cone_deg": P["cone"]},
               {"op": "morphogen_growth_3d", "at": "vertex", "cell_set": "cell", "rate": P["grow"],
                "a_sw": 0.5, "hill": 4.0, "rho": P["rho"], "vth_frac": P["vth"]},
               {"op": "shape_energy_3d", "at": "vertex", "p0": 3.90, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.05,
                "Lambda": 0.2, "K_V": P["K_V"], "K_R": 0.02, "mu": 1.0, "dt": 1.0, "relax_iters": 30, "eta": 0.08, "cap_frac": 0.12},
               {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.28, "every": 1, "max_flips": max(40, n_cells // 8)},
               {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.4, "p0": 3.90,
                "every": 2, "max_div": 12, "max_div_frac": 0.03, "cell_set": "cell",
                "min_cycle": P["min_cycle"], "max_cycle": P["max_cycle"]},
               {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
        sched = ["seed_mesh_3d", "cell_geometry_3d", "seed_cell_rd", "morphogen_growth_3d",
                 "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
        cfg = {"general": {"name": f"sw_{name}", "seed": SEED, "n_frames": frames, "dt": 1.0, "record_cap": frames + 2,
                           "boundary": "free", "dim": 3, "world": [16 * RADIUS] * 3},
               "sets": {"vertex": {"n": buf}, "cell": {"n": cbuf, "state": {"chem": {"width": 2, "integration": "first_order"},
                                                                            "cen": {"width": 3}, "area": {"width": 1}}}},
               "fields": {}, "operators": ops, "schedule": sched}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            yaml.safe_dump(cfg, fh); path = fh.name
        sim = S.load(path); os.unlink(path)
        Hf, out = engine_run(sim, device="cpu")
        emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]
        mt = hist[-1] if hist else dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)
        pt = posf[T - 1][:mt["Nv"]].astype(np.float64)
        _, area, _ = hollow_metric(pt, mt); area = area[area > 0]
        area_cv = float(area.std() / (area.mean() + 1e-9))
        rings = rings_from_flat_3d(np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"]), mt["nF"])
        rad = np.array([np.linalg.norm(pt[r].mean(0)) if (r is not None and len(r)) else 0 for r in rings]); rad = rad[rad > 0]
        protr = float(np.percentile(rad, 95) / (np.median(rad) + 1e-9))
        hollow = float(hollow_flags(pt, mt)[2]["frac"])
        ext = pt.max(0) - pt.min(0)
        # GOAL score: reward protrusion + uniformity + cleanliness (want protr high, area_cv low, hollow low)
        score = protr - 1.5 * area_cv - 2.0 * hollow
        return dict(name=name, cells=int(mt["nF"]), area_cv=round(area_cv, 3), protr=round(protr, 3),
                    hollow=round(hollow, 3), aspect=round(float(ext.max() / max(ext.min(), 1e-6)), 2),
                    score=round(score, 3), **{k: P[k] for k in ("vth", "rho", "cone", "grow", "K_V", "min_cycle", "max_cycle")})
    except Exception as e:
        return dict(name=name, error=repr(e)[:120], score=-99)


def grid(cells, frames):
    cfgs = []
    for vth, rho, cone, grow, K_V, dur in itertools.product(
            [1.5, 2.5, 4.0], [0.0, 0.1], [10.0, 16.0], [0.025], [3.0], [(3, 14), (0, 10 ** 9)]):
        mn, mx = dur
        cfgs.append(dict(name=f"v{vth}_r{rho}_c{int(cone)}_d{mn}", cells=cells, frames=frames, spots=5,
                         vth=vth, rho=rho, cone=cone, grow=grow, K_V=K_V, min_cycle=mn, max_cycle=mx))
    # dedup names
    seen = {};
    for c in cfgs:
        seen[c["name"]] = c
    return list(seen.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--cells", type=int, default=800)
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    cfgs = grid(a.cells, a.frames)
    print(f"[sweep] {len(cfgs)} configs, {a.workers} workers, {a.cells}c/{a.frames}f", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one, c): c for c in cfgs}
        for fu in as_completed(futs):
            r = fu.result(); results.append(r)
            print(f"  {r.get('name'):22s} score={r.get('score')}  area_cv={r.get('area_cv')} "
                  f"protr={r.get('protr')} hollow={r.get('hollow')} cells={r.get('cells')} "
                  f"{'ERR '+r['error'] if 'error' in r else ''}", flush=True)
    results.sort(key=lambda r: r.get("score", -99), reverse=True)
    json.dump(results, open(os.path.join(HERE, "sweep_fig5_results.json"), "w"), indent=1)
    print("\n=== TOP 6 (by score = protr - 1.5*area_cv - 2*hollow) ===")
    for r in results[:6]:
        print(f"  {r['name']:22s} score={r.get('score')} protr={r.get('protr')} area_cv={r.get('area_cv')} "
              f"hollow={r.get('hollow')} cells={r.get('cells')}  vth={r.get('vth')} rho={r.get('rho')} cone={r.get('cone')} dur={r.get('min_cycle')}")
