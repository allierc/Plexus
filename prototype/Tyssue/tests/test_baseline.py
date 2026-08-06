#!/usr/bin/env python
"""test_baseline -- the IMMOVABLE regression baseline for the Tyssue AVM prototype operators.

Two short, fully deterministic end-to-end runs on the closed spherical half-edge mesh:

  vesicle3d : seed_mesh_3d -> vesicle_growth -> shape_energy_3d -> reconnect_t1_3d -> divide_3d
              -> cell_geometry_3d -> topo_snapshot_3d
  rd        : the same mechanics + the live Turing RD on the cell set
              (cell_adjacency -> seed_cell_rd -> cell_diffuse -> cell_react), with divide_3d
              propagating the morphogen to daughters (cell_set: cell)

For each run we record, at frame 0, the midpoint and the final frame:
  * the live cell count nF and vertex count Nv,
  * the closed-surface invariants (closed, V, E, F, Euler characteristic) recomputed from the
    face rings -- Euler must be 2 for a closed sphere,
  * the FULL tissue_analysis.frame_metrics dict (hollow cells, size CVs, protrusion, tube
    diameter/length/count, cell census),
  * a sha256 HASH of the live float64 vertex positions plus scalar position invariants,
  * the mesh event counters n_div / n_t1 and (rd) the activator statistics.

Usage
    PYTHONPATH=/workspace/Plexus/src python tests/test_baseline.py --record   # write the baseline
    PYTHONPATH=/workspace/Plexus/src python tests/test_baseline.py            # compare, exit 1 on drift

Tolerances.  Everything TOPOLOGICAL / COUNTED (nF, Nv, V, E, F, euler, closed, n_div, n_t1,
hollow_n, n_tubes, n_tip, n_red, cells) is compared EXACTLY -- a behaviour-preserving refactor
cannot change an integer.  Floats are compared with rtol=1e-9 / atol=1e-12: the runs are CPU,
deterministic torch reductions with seeded numpy RNGs, so a refactor that only moves data
between containers must reproduce them bit-for-bit; the tolerance exists solely to absorb
reassociation of a scalar that is read out of a dict instead of an attribute.  Most of the tube
metrics are already rounded to 3 decimals inside tissue_analysis, so they are far coarser than
that anyway.  The position HASH is exact; `--allow-hash-drift` downgrades it to a warning so a
last-ULP change can be distinguished from a change in the physics (the float invariants recorded
alongside it stay hard checks either way).

LIVENESS.  The test also asserts that each run actually DID something (divisions fired, T1s
fired, cells grew, the RD patterned).  A silently-inert operator that still returns metrics is
the worst failure mode this prototype has: it looks like a valid negative result.  If those
assertions fire, the baseline is meaningless and the test says so instead of recording zeros.

NOTE (clock double-gating).  divide_3d / reconnect_t1_3d carry a PRIVATE `every` gate on top of
the engine's own `every` gate, so their effective period here is every^2 = 4 ticks, not 2.  The
baseline records the CURRENT behaviour.  Removing the private gates is a deliberate behaviour
CHANGE: it will move n_div / n_t1 and everything downstream, and requires a documented
re-record, not a tolerance widening.

NOTE (why vesicle_growth is in both runs).  Without a target-volume ramp no cell ever reaches
divide_3d's 2x-birth-volume trigger, so divide_3d would sit silently inert and the baseline
would pin down nothing about it.  The growth operator is therefore part of both baselines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROTO, "..", "..", "src"))   # core plexus
sys.path.insert(0, PROTO)                                    # the prototype modules

import numpy as np
import yaml

import plexus.operators        # noqa: F401  registers the 52 core operators
import tyssue_ops3d            # noqa: F401  seed_mesh_3d / shape_energy_3d / vesicle_growth / divide_3d / topo_snapshot_3d
import tyssue_t1_ops3d         # noqa: F401  reconnect_t1_3d
import tyssue_rd_ops           # noqa: F401  cell_geometry_3d / cell_adjacency / seed_cell_rd / cell_diffuse / cell_react
import plexus.schema as S
from plexus.engine import run as engine_run
from tissue_analysis import frame_metrics
from tyssue_topology_ops3d import rings_from_flat_3d, _check_closed

BASE = os.path.join(HERE, "_baseline")

# --- the two runs (small + short, but large enough that division and T1 both fire) ------------
N_CELLS = 120
FRAMES = 60
RADIUS, JITTER, SEED, P0 = 5.0, 0.16, 0, 3.72
GROW = 0.006                 # target-volume ramp (see the module docstring)

RTOL, ATOL = 1e-9, 1e-12
# fields whose value is a count / a topological invariant -> must match exactly
_EXACT = {"cells", "nF", "Nv", "V", "E", "F", "euler", "closed", "n_div", "n_t1",
          "hollow_n", "n_tubes", "n_tip", "n_red", "frame", "T", "n_hist"}


# --------------------------------------------------------------------------------------------
#  spec construction (mirrors run_tyssue_vesicle.make_spec / run_tyssue_rd.make_spec)
# --------------------------------------------------------------------------------------------
GS = dict(react=dict(implementation="gray_scott", F=0.055, kk=0.062, rate=1.0),
          diffuse=dict(d_a=0.08, d_h=0.16, chi=1.3),
          seed=dict(mode="scatter", seed_frac=0.06))


def make_spec(name, rd, n_cells, frames, buf, cbuf):
    """seed -> growth -> mechanics -> T1 -> divide -> (RD) -> snapshot. `rd` None = the plain
    vesicle run. topo_snapshot_3d runs at every=1 and record_cap >= frames+2, so the recorded
    position frames and the topology history are the SAME length (see the length assertion in
    run_one -- pairing position frame t with topology frame min(t, len-1) is the mis-pairing
    bug that produced a phantom 97%-hollow-cell result)."""
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": n_cells, "radius": RADIUS,
            "jitter": JITTER, "p0": P0, "seed": SEED, "before_frame": 1},
           {"op": "vesicle_growth", "at": "vertex", "rate": GROW, "every": 1},
           {"op": "shape_energy_3d", "at": "vertex", "p0": P0, "K_A": 1.0, "K_P": 1.0,
            "Gamma": 0.1, "Lambda": 0.5, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": 1.0,
            "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 2,
            "max_flips": max(20, n_cells // 15)},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "p0": P0,
            "every": 2, "max_div": max(10, n_cells // 20),
            **({"cell_set": "cell"} if rd else {})},
           {"op": "cell_geometry_3d", "at": "cell"}]
    sched = ["seed_mesh_3d", "vesicle_growth", "shape_energy_3d", "reconnect_t1_3d",
             "divide_3d", "cell_geometry_3d"]
    if rd:
        ops += [{"op": "cell_adjacency", "at": "cell"},
                {"op": "seed_cell_rd", "at": "cell", "seed": SEED, "before_frame": 3, **rd["seed"]},
                {"op": "cell_diffuse", "at": "cell", **rd["diffuse"]},
                {"op": "cell_react", "at": "cell", **rd["react"]}]
        sched += ["cell_adjacency", "seed_cell_rd", "cell_diffuse", "cell_react"]
    ops.append({"op": "topo_snapshot_3d", "at": "vertex", "every": 1})
    sched.append("topo_snapshot_3d")
    cell_state = {"cen": {"width": 3}, "area": {"width": 1}}
    if rd:
        cell_state["chem"] = {"width": 2, "integration": "first_order"}
    cfg = {"general": {"name": f"tyssue_baseline_{name}", "seed": SEED, "n_frames": frames,
                       "dt": 1.0, "record_cap": frames + 2, "boundary": "free", "dim": 3,
                       "world": [8 * RADIUS, 8 * RADIUS, 8 * RADIUS]},
           "sets": {"vertex": {"n": buf}, "cell": {"n": cbuf, "state": cell_state}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim


# --------------------------------------------------------------------------------------------
#  measurement
# --------------------------------------------------------------------------------------------
def _hash(pos):
    return hashlib.sha256(np.ascontiguousarray(pos, dtype=np.float64).tobytes()).hexdigest()[:32]


def snapshot(pt, mt, act):
    """Everything we pin down for one frame: topology invariants + the full tube metric dict +
    the position hash and its scalar invariants."""
    ok, V, E, F, euler = _check_closed(rings_from_flat_3d(
        np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"]), int(mt["nF"])))
    rad = np.linalg.norm(pt, axis=1)
    rec = dict(nF=int(mt["nF"]), Nv=int(mt["Nv"]), closed=bool(ok), V=int(V), E=int(E), F=int(F),
               euler=int(euler), pos_hash=_hash(pt),
               pos_sum=float(pt.sum()), pos_absum=float(np.abs(pt).sum()),
               rad_mean=float(rad.mean()), rad_std=float(rad.std()), rad_max=float(rad.max()))
    rec["metrics"] = frame_metrics(pt, mt, act)
    if act is not None:
        rec["act"] = dict(mean=float(np.mean(act)), std=float(np.std(act)),
                          min=float(np.min(act)), max=float(np.max(act)), sum=float(np.sum(act)))
    return rec


def run_one(name, rd):
    n_cells = N_CELLS
    verts, _, _, _, nF0 = tyssue_ops3d.build_sphere_mesh(n_cells, RADIUS, JITTER, SEED)
    Nv0 = verts.shape[0]
    buf, cbuf = int(Nv0 * 6), int(nF0 * 6)
    sim = make_spec(name, rd, n_cells, FRAMES, buf, cbuf)
    Hf, out = engine_run(sim, device="cpu")
    mesh = Hf.level("vertex")._mesh
    hist = mesh.get("hist")
    posf = out["sets"]["vertex"]["pos"]
    T = posf.shape[0]
    # HARD alignment check: mis-pairing late positions with stale topology is what manufactured
    # the phantom "97% hollow cell" result. Never silently clamp the index.
    assert hist is not None, "topo_snapshot_3d recorded no history"
    assert len(hist) == T, (f"topology history ({len(hist)}) and recorded position frames ({T}) "
                            f"have different lengths -- frames cannot be paired")
    chemf = out["sets"]["cell"]["state"].get("chem") if out["sets"]["cell"]["state"] else None

    def frame(t):
        mt = hist[t]
        pt = posf[t][:int(mt["Nv"])].astype(np.float64)
        act = chemf[t][:int(mt["nF"]), 0].astype(np.float64) if chemf is not None else None
        return pt, mt, act

    rec = dict(name=name, n_cells_cfg=n_cells, frames=FRAMES, T=int(T), n_hist=int(len(hist)),
               buf=int(buf), cbuf=int(cbuf), Nv0=int(Nv0), nF0=int(nF0),
               n_div=int(mesh.get("n_div", 0)), n_t1=int(mesh.get("n_t1", 0)),
               gscale=float(mesh.get("gscale", 1.0)))
    rec["frames_checked"] = {}
    for tag, t in (("first", 0), ("mid", T // 2), ("last", T - 1)):
        pt, mt, act = frame(t)
        s = snapshot(pt, mt, act); s["frame"] = int(t)
        rec["frames_checked"][tag] = s
    # LIVENESS: a baseline of "nothing happened" is worse than no baseline at all.
    assert rec["n_div"] > 0, f"{name}: divide_3d never fired -- the baseline would be inert"
    assert rec["n_t1"] > 0, f"{name}: reconnect_t1_3d never fired -- the baseline would be inert"
    assert rec["frames_checked"]["last"]["nF"] > rec["frames_checked"]["first"]["nF"], \
        f"{name}: cell count did not grow"
    if rd:
        a = rec["frames_checked"]["last"]["act"]
        assert a["std"] > 1e-6, f"{name}: the RD activator is flat -- the RD operators were inert"
    return rec


RUNS = (("vesicle3d", None), ("rd", GS))


# --------------------------------------------------------------------------------------------
#  compare
# --------------------------------------------------------------------------------------------
def diff(base, new, path="", allow_hash_drift=False, out=None):
    out = [] if out is None else out
    if isinstance(base, dict):
        if not isinstance(new, dict):
            out.append(f"{path}: type changed dict -> {type(new).__name__}"); return out
        for k in sorted(set(base) | set(new)):
            if k not in new:
                out.append(f"{path}/{k}: MISSING in new run")
            elif k not in base:
                out.append(f"{path}/{k}: NEW key (not in baseline) = {new[k]!r}")
            else:
                diff(base[k], new[k], f"{path}/{k}", allow_hash_drift, out)
        return out
    key = path.rsplit("/", 1)[-1]
    if key == "pos_hash":
        if base != new and not allow_hash_drift:
            out.append(f"{path}: hash {base} -> {new}")
        elif base != new:
            print(f"  [warn] {path}: hash {base} -> {new} (allowed)")
        return out
    if isinstance(base, bool) or key in _EXACT or isinstance(base, str):
        if base != new:
            out.append(f"{path}: {base!r} -> {new!r}")
        return out
    if isinstance(base, (int, float)):
        if not isinstance(new, (int, float)):
            out.append(f"{path}: type changed -> {new!r}"); return out
        if abs(float(new) - float(base)) > ATOL + RTOL * abs(float(base)):
            out.append(f"{path}: {base!r} -> {new!r}  (delta {float(new) - float(base):.3e})")
        return out
    if base != new:
        out.append(f"{path}: {base!r} -> {new!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="write the baseline instead of comparing")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--allow-hash-drift", action="store_true",
                    help="downgrade a position-hash mismatch to a warning (float checks stay hard)")
    a = ap.parse_args()
    os.makedirs(BASE, exist_ok=True)
    failed = []
    for name, rd in RUNS:
        if a.only and name not in a.only:
            continue
        print(f"\n=== baseline run: {name} ===", flush=True)
        rec = run_one(name, rd)
        path = os.path.join(BASE, f"{name}.json")
        if a.record:
            json.dump(rec, open(path, "w"), indent=1, sort_keys=True)
            print(f"[record] wrote {path}")
            print(json.dumps(rec, indent=1, sort_keys=True))
            continue
        if not os.path.exists(path):
            print(f"[FAIL] no baseline at {path}; run with --record first"); failed.append(name); continue
        base = json.load(open(path))
        # json round-trips through the same float64 repr, so this is an exact re-read
        d = diff(base, json.loads(json.dumps(rec)), path=name, allow_hash_drift=a.allow_hash_drift)
        if d:
            print(f"[FAIL] {name}: {len(d)} field(s) drifted")
            for line in d:
                print("   " + line)
            failed.append(name)
        else:
            print(f"[OK] {name}: matches baseline "
                  f"(cells {rec['frames_checked']['first']['nF']} -> {rec['frames_checked']['last']['nF']}, "
                  f"{rec['n_div']} div, {rec['n_t1']} T1)")
    if failed and not a.record:
        print(f"\nBASELINE MISMATCH: {', '.join(failed)}")
        return 1
    if not a.record:
        print("\nBASELINE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
