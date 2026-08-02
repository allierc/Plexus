"""Differ for `regulate:connectionist`: score the Plexus operator against the reference.

Unlike the `mwc` sibling (which returns a bare rate the engine Euler-steps, forcing an
integrator-matched comparison), the connectionist operator SELF-SOLVES each macro-step with
fixed-step RK4 (substeps=64) and returns (g(dt)-g0)/dt, so the engine's g += dt*delta recovers
g(dt). Its self-solve is an accurate integration of the SAME ODE the reference integrates with
adaptive diffrax Dopri5, so we can compare the Plexus engine trajectory DIRECTLY against the
reference's own Dopri5 output -- the honest end-to-end test of "does our operator reproduce the
reference's behaviour". The only admissible disagreement is the integrator gap (RK4-64 vs Dopri5
rtol=1e-4) plus cross-backend float32 rounding; a wrong reaction law diverges by O(1) in a few
frames.

    python diff_gene_network_connectionist.py score   # after run_spec.py gene_network_connectionist

The oracle (reference.npz) and the Plexus spec were both generated from ONE parameter set by
_oracle/scripts/_gen_gene_network_connectionist.py, so the two sides cannot drift apart.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))

ORACLE_RUN = os.path.join(HERE, "_oracle", "runs", "diff_gene_network_connectionist")
REF_NPZ = os.path.join(ORACLE_RUN, "reference.npz")
THRESHOLD = 5.0e-3


def _zarr_gene():
    """The engine's recorded gene trajectory [n_rec, N, n_gene], from simulation.zarr."""
    import zarr
    from plexus.paths import graphs_data_path
    zpath = os.path.join(graphs_data_path("atlas", "gene_network_connectionist"),
                         "simulation.zarr")
    if not os.path.exists(zpath):
        raise SystemExit(f"no engine trajectory at {zpath} -- run "
                         f"run_spec.py gene_network_connectionist first")
    root = zarr.open_group(zpath, mode="r")
    return np.asarray(root["cell"]["state"]["gene"]), zpath


def score():
    r = dict(np.load(REF_NPZ))
    G_ref = np.asarray(r["gene"])                    # [21, 6, 5] Dopri5 reference trajectory
    G_eng, zpath = _zarr_gene()                      # [21, 6, 5] Plexus RK4-64 self-solve
    if G_eng.shape != G_ref.shape:
        raise SystemExit(f"shape mismatch: engine {G_eng.shape} vs reference {G_ref.shape} -- "
                         f"the comparison is only meaningful frame-for-frame")

    traj_err = np.abs(G_eng.astype(np.float32) - G_ref.astype(np.float32))
    value = float(traj_err.max())
    finite = bool(np.isfinite(G_eng).all())
    passed = bool(finite and value < THRESHOLD)

    # where does the max deviation sit? An integrator-limited gap concentrates at the LATEST,
    # largest-magnitude frames; an early-frame O(1) spike would betray a reaction-law bug.
    k, i, j = np.unravel_index(int(traj_err.argmax()), traj_err.shape)
    per_frame_max = traj_err.reshape(traj_err.shape[0], -1).max(axis=1)
    # a relative view at the argmax, for context against the ~1e-4 reference rtol
    rel_at_max = float(value / (abs(float(G_ref[k, i, j])) + 1e-6))

    out = {
        "threshold": THRESHOLD,
        "metric": "max_abs_gene_trajectory_deviation",
        "value": value,
        "passed": passed,
        "trajectory_finite": finite,
        "argmax_frame_cell_gene": [int(k), int(i), int(j)],
        "argmax_gene_ref_value": float(G_ref[k, i, j]),
        "argmax_gene_eng_value": float(G_eng[k, i, j]),
        "relative_deviation_at_argmax": rel_at_max,
        "per_frame_max_abs_dev": per_frame_max.tolist(),
        "first_frame_max_abs_dev": float(per_frame_max[0]),
        "final_frame_max_abs_dev": float(per_frame_max[-1]),
        "gene_ref_final_cell0": G_ref[-1, 0].tolist(),
        "gene_eng_final_cell0": G_eng[-1, 0].tolist(),
        "n_frames": int(G_ref.shape[0]), "n_cells": int(G_ref.shape[1]),
        "n_gene": int(G_ref.shape[2]),
        "engine_trajectory": zpath,
        "reference_npz": REF_NPZ,
    }
    with open(os.path.join(ORACLE_RUN, "diff.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nmax|dev| occurs at frame {k}, cell {i}, gene {j}: "
          f"ref {G_ref[k,i,j]:.5f} vs eng {G_eng[k,i,j]:.5f}  "
          f"(rel {rel_at_max:.2e})")
    print(f"first-frame max|dev| = {per_frame_max[0]:.3e}  "
          f"final-frame max|dev| = {per_frame_max[-1]:.3e}  "
          f"(monotone growth => integrator-limited, not a reaction-law bug)")
    print(f"\nvalue = {value:.3e}   threshold = {THRESHOLD:g}   "
          f"=> {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score"])
    a = ap.parse_args()
    if a.mode == "score":
        sys.exit(score())
