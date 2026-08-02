"""diff_brownian_dynamics -- score the Plexus `agitate` bath against the jax-morph
BrownianDynamics(potential=None) free-diffusion reference.

Runs in the PLEXUS (torch) env, NOT the oracle venv. `agitate` is the thermal leg of
BrownianDynamics with the drift delegated to a separate pair-potential; here it is scheduled ALONE
(pure bath), so the differential test isolates exactly the piece the frozen language lacked.

Because the JAX and torch PRNG streams differ, a pathwise trajectory can never match sample by
sample; the invariant is the free-diffusion constant D read from the cloud's radius of gyration:

    Rg^2(t) = mean over alive cells of |r_i(t) - c(t)|^2  ->  Rg^2 = 2 * n_dim * D * t,
    D_hat   = slope(Rg^2 vs t through origin) / (2 * n_dim),

with each recorded frame carried at its TRUE elapsed time (engine steps once per tick over ticks
0..n_frames, so recorded frame k is after k+1 steps: t_k = (k+1)*dt). D is measured at dt in
{1.0, 0.5, 0.25} at fixed total time T ~ 40; it is dt-INVARIANT iff the noise carries the Wiener
sqrt(dt) scaling. value = max over dt of |D_plexus(dt) - D_ref(dt)| / D_ref(dt); PASS iff
value <= THRESHOLD (0.03).

NEGATIVE CONTROL (pre-registered). A bath that scales noise with dt instead of sqrt(dt) -- exactly
what drag/glide/attraction_repulsion's deterministic-integrated `noise*randn` term does -- is
rolled through the identical engine integration at dt=0.25. Its D collapses to dt*kT/gamma = 0.025
(a 75% error, ~25x the threshold), proving the metric rejects a mis-scaled bath, AND a Wiener-
scaled roll on the SAME harness recovers D ~ 0.1, proving the failure is the scaling, not the roll.

  /workspace/.conda_envs/neural-graph-linux/bin/python atlas_jax_morph/diff_brownian_dynamics.py
"""
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))

THRESHOLD = 0.03
NDIM = 2
KT, GAMMA = 0.1, 1.0
D_THEORY = KT / GAMMA
REF = os.path.join(HERE, "_oracle", "runs", "diff_brownian_dynamics", "reference.npz")
SPECS = {                                   # dt -> spec name (config/atlas/<name>.yaml)
    1.0: "brownian_dynamics",
    0.5: "brownian_dynamics_dt05",
    0.25: "brownian_dynamics_dt025",
}

# --- load the atlas anti-chamber so the specs can name `agitate` ----------------------------- #
import plexus.operators  # noqa: F401  self-registers the core library
import importlib
import plexus.operators.candidates as C
for fn in sorted(os.listdir(os.path.dirname(C.__file__))):
    if fn.startswith(("jax_morph_", "atlas_")) and fn.endswith(".py"):
        importlib.import_module(f"plexus.operators.candidates.{fn[:-3]}")

from plexus.schema import load
from plexus.engine import run, build


# ------------------------------------------------------------------------------------------- #
#  estimators (identical on both sides)
# ------------------------------------------------------------------------------------------- #
def rg2_of(pos, occ):
    """Rg^2(t) = mean over alive cells of |r - centroid|^2, per frame. pos [T,N,D], occ [T,N]."""
    out = np.empty(pos.shape[0])
    for t in range(pos.shape[0]):
        live = occ[t].astype(bool)
        p = pos[t][live]
        c = p.mean(0)
        out[t] = float(((p - c) ** 2).sum(1).mean())
    return out


def slope_D(t, rg2):
    """D from the least-squares slope of Rg^2 vs t THROUGH THE ORIGIN, and the fit R^2."""
    t = np.asarray(t, float); y = np.asarray(rg2, float)
    b = float((t * y).sum() / (t * t).sum())                 # slope through origin
    resid = y - b * t
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return b / (2.0 * NDIM), r2


def endpoint_D(t, rg2):
    return float(rg2[-1] / (2.0 * NDIM * t[-1]))


# ------------------------------------------------------------------------------------------- #
#  the reference (oracle side, precomputed): rebuild D_ref from the saved Rg^2 trajectories
# ------------------------------------------------------------------------------------------- #
ref = np.load(REF, allow_pickle=True)
ref_D = {}
for dt in SPECS:
    key = f"dt{dt}".replace(".", "p")
    t = np.asarray(ref[f"{key}_t"], float)                   # oracle history times: k*dt, k=0..n_steps
    rg2 = np.asarray(ref[f"{key}_rg2"], float)
    Ds, r2 = slope_D(t, rg2)
    ref_D[dt] = {"D_slope": Ds, "D_endpoint": endpoint_D(t, rg2), "fit_r2": r2,
                 "rg2_final": float(rg2[-1]), "T": float(t[-1]), "n_frames": int(len(t) - 1)}

# ------------------------------------------------------------------------------------------- #
#  the Plexus operator AS THE ENGINE RUNS IT
# ------------------------------------------------------------------------------------------- #
plx_D = {}
per_dt = {}
for dt, name in SPECS.items():
    sim = load(os.path.join(PLEXUS, "config", "atlas", name + ".yaml"))
    assert abs(float(sim.dt) - dt) < 1e-12, (sim.dt, dt)
    _, out = run(sim, out_path=None, device="cpu", progress=False)
    pos = np.asarray(out["sets"]["cell"]["pos"], float)      # [n_rec, N, D]
    occ = np.asarray(out["sets"]["cell"]["occ"]).astype(bool)
    t = (np.arange(pos.shape[0]) + 1) * dt                   # frame k is after k+1 steps
    rg2 = rg2_of(pos, occ)
    Ds, r2 = slope_D(t, rg2)
    De = endpoint_D(t, rg2)
    plx_D[dt] = Ds
    rel_slope = abs(Ds - ref_D[dt]["D_slope"]) / ref_D[dt]["D_slope"]
    rel_end = abs(De - ref_D[dt]["D_endpoint"]) / ref_D[dt]["D_endpoint"]
    per_dt[str(dt)] = {
        "dt": dt, "n_frames": int(sim.n_frames), "steps_applied": int(pos.shape[0]),
        "T_plexus": float(t[-1]),
        "D_plexus_slope": Ds, "D_plexus_endpoint": De, "plexus_fit_r2": r2,
        "D_ref_slope": ref_D[dt]["D_slope"], "D_ref_endpoint": ref_D[dt]["D_endpoint"],
        "rel_err_slope": rel_slope, "rel_err_endpoint": rel_end,
        "rel_err_vs_theory": abs(Ds - D_THEORY) / D_THEORY,
    }
    print(f"dt={dt:<5} D_plexus(slope)={Ds:.5f}  D_ref(slope)={ref_D[dt]['D_slope']:.5f}  "
          f"rel={rel_slope:.4f}  R2={r2:.5f}  (theory {D_THEORY})", flush=True)

value = max(per_dt[str(dt)]["rel_err_slope"] for dt in SPECS)
passed = bool(value <= THRESHOLD)

# dt-invariance on the Plexus side: spread of D across dt (should be ~0 iff Wiener sqrt(dt))
plx_vals = np.array([plx_D[dt] for dt in SPECS])
plx_spread = float(plx_vals.max() - plx_vals.min())

# ------------------------------------------------------------------------------------------- #
#  NEGATIVE CONTROL -- a dt-scaled bath (the bug) vs a Wiener bath, on the SAME roll harness
# ------------------------------------------------------------------------------------------- #
def roll(spec_name, wiener):
    """Manually roll a bath through the engine's first-order integration x += dt*v on the spec's
    Hierarchy. wiener=True  -> v = sqrt(2 kT/(gamma dt)) * xi  (agitate's correct 1/sqrt(dt)),
    wiener=False -> v = sqrt(2 kT/gamma)      * xi  (dt-INDEPENDENT amplitude -> displacement ~ dt,
    the mis-scaled `noise*randn` bug). Returns (t, Rg^2(t))."""
    sim = load(os.path.join(PLEXUS, "config", "atlas", spec_name + ".yaml"))
    H = build(sim, "cpu")
    cell = H.level("cell")
    px0, px1 = cell.state_schema["pos"]
    dt = float(sim.dt)
    N = cell.state.shape[0]
    amp = math.sqrt(2.0 * KT / (GAMMA * dt)) if wiener else math.sqrt(2.0 * KT / GAMMA)
    ts, rg2 = [], []
    for tick in range(sim.n_frames + 1):
        xi = torch.randn(N, NDIM, generator=H.rng)
        v = amp * xi * cell.occ[:, None]
        new = cell.state.clone()
        new[:, px0:px1] = new[:, px0:px1] + dt * v            # engine first-order step
        cell.state = new
        p = cell.get("pos")[cell.occ > 0]
        c = p.mean(0)
        rg2.append(float(((p - c) ** 2).sum(1).mean())); ts.append((tick + 1) * dt)
    return np.asarray(ts), np.asarray(rg2)

t_w, rg2_w = roll("brownian_dynamics_dt025", wiener=True)     # harness check: must recover ~0.1
D_wiener_roll, _ = slope_D(t_w, rg2_w)
t_n, rg2_n = roll("brownian_dynamics_dt025", wiener=False)    # the bug: must collapse to ~0.025
D_naive_roll, _ = slope_D(t_n, rg2_n)
nc_rel_err = abs(D_naive_roll - ref_D[0.25]["D_slope"]) / ref_D[0.25]["D_slope"]

# ------------------------------------------------------------------------------------------- #
result = {
    "operator": "agitate (BrownianDynamics thermal leg, potential=None)",
    "threshold": THRESHOLD,
    "value": value,
    "passed": passed,
    "metric": "max over dt of |D_plexus - D_ref| / D_ref, D = slope(Rg^2 vs t)/(2 n_dim)",
    "D_theory_kT_over_gamma": D_THEORY,
    "per_dt": per_dt,
    "plexus_D_spread_over_dt": plx_spread,           # ~0 iff Wiener sqrt(dt) scaling holds
    "negative_control": {
        "desc": "dt=0.25 dt-scaled bath (noise ~ dt, the bolt-on `noise*randn` bug)",
        "D_naive_roll": D_naive_roll,
        "D_naive_theory_dt_kT_over_gamma": 0.25 * KT / GAMMA,
        "nc_rel_err_vs_ref": nc_rel_err,             # must be >> threshold
        "nc_rejected": bool(nc_rel_err > THRESHOLD),
        "harness_check_wiener_roll_D": D_wiener_roll,  # SAME roll, correct scaling -> ~0.1
    },
    "oracle_run": "diff_brownian_dynamics",
    "N_cells": 20000, "kT": KT, "gamma": GAMMA,
    "specs": {str(dt): f"config/atlas/{n}.yaml" for dt, n in SPECS.items()},
}

out_dir = os.path.join(PLEXUS, "log", "atlas_jax", "brownian_dynamics")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "diff.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
print(f"\n-> {os.path.join(out_dir, 'diff.json')}")
print(f"NEGATIVE CONTROL: D_naive(dt=0.25)={D_naive_roll:.5f} (rel {nc_rel_err:.2f} = "
      f"{nc_rel_err / THRESHOLD:.0f}x threshold); wiener-roll harness check D={D_wiener_roll:.5f}")
print("PASS" if passed else "FAIL", f"  value={value:.4f}  threshold={THRESHOLD}")
