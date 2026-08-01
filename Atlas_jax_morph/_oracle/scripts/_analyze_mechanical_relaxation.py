"""Score the relax differential: the Plexus `relax` equilibrium vs the jax-morph MechanicalRelaxation
reference, per cell. Runs in the Plexus (torch) env; no re-simulation.

  reference : Atlas_jax_morph/_oracle/runs/diff_mechanical_relaxation/reference.npz
              (jxm MechanicalRelaxation(Morse), FIRE to f_tol, quasistatic x*)
  plexus    : graphs_data/atlas/mechanical_relaxation/trajectory.npz
              (config/atlas/mechanical_relaxation.yaml, engine `relax` operator)

Primary metric  D_eq = max over LIVE cells i of ||x*_plx,i - x*_ref,i||_2 / sigma at the relaxed
                equilibrium (frame 1). sigma = r_i + r_j = 1.0.
Threshold       1.0e-3 (sigma), pre-registered in the record before this diff was computed.
Corroborators   force-balance cross-check (the operator's OWN Morse energy evaluated at each side's
                equilibrium -> |grad U|_inf, a gauge-free test that both are force balances of the
                SAME energy), rigid-gauge-removed (Kabsch) D_eq, fixed-point idempotence of the
                Plexus no-op plateau, frame-0 == IC on both sides, dead-slot immobility, and the
                misaligned x_plx(1) vs IC magnitude. Writes diff.json into log/atlas/mechanical_relaxation/.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))
REF = os.path.join(HERE, "..", "runs", "diff_mechanical_relaxation", "reference.npz")
PLX = os.path.join(PLEXUS, "graphs_data", "atlas", "mechanical_relaxation", "trajectory.npz")
LOG = os.path.join(PLEXUS, "log", "atlas", "mechanical_relaxation")

# the operator's OWN Morse energy helpers -- so the force cross-check uses the identical energy the
# `relax` operator relaxes, not a re-derivation.
from plexus.operators.candidates.jax_morph_mechanical_relaxation import _pair_energy, _safe_norm

r = np.load(REF)
p = np.load(PLX)

ref = r["position"].astype(np.float64)      # [T+1, CAP, 2]  index t = IC after t macro-steps
plx = p["cell__pos"].astype(np.float64)     # [T+1, CAP, 2]  frame t = IC after t steps (after_frame:1)
occ = p["cell__occ"].astype(bool)           # [T+1, CAP]
aliv = r["alive"].astype(bool)              # [T+1, CAP]
radius = r["radius"].astype(np.float64)     # [CAP]  (0.5 live, 0 dead)
sigma = float(r["sigma"]); f_tol = float(r["f_tol"]); eps = float(r["eps"]); alpha = float(r["alpha"])
N = int(r["N"]); CAP = int(r["CAP"]); T = int(r["NSTEPS"])
ref_residual = float(r["residual_at_xstar"])

assert ref.shape == plx.shape == (T + 1, CAP, 2), (ref.shape, plx.shape)
assert np.array_equal(occ, aliv), "live masks differ between Plexus and reference"
live = aliv[0]                              # [CAP] fixed live set
n_live = int(live.sum())
EQ = 1                                      # frame index of the relaxed equilibrium (relax after_frame:1)


# --- force-balance cross-check: the operator's own Morse energy, autodiffed to |grad U|_inf ---- #
def force_infnorm(x_np):
    """max over LIVE cells of |grad_x U|_inf at configuration x, using the operator's Morse energy
    (free boundary, sigma = r_i + r_j, live non-self pairs) -- a gauge-free equilibrium test."""
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    rad = torch.tensor(radius, dtype=torch.float64)
    sig = rad[:, None] + rad[None, :]
    al = torch.tensor(live)
    eye = torch.eye(CAP, dtype=torch.bool)
    pair_mask = al[:, None] & al[None, :] & ~eye
    disp = x[:, None, :] - x[None, :, :]                 # free boundary -> no minimum image
    rr = _safe_norm((disp * disp).sum(-1))
    u = _pair_energy("morse", rr, sig, eps, alpha, 1.5, 2.5)
    u = torch.where(pair_mask, u, torch.zeros_like(u))
    energy = 0.5 * u.sum()
    (g,) = torch.autograd.grad(energy, x)
    return float(g[torch.tensor(live)].abs().max())


# --- Kabsch: best rigid (rotation+translation, no scale) alignment of plx-live onto ref-live ---- #
def kabsch_resid(A, B):
    """max per-point ||A' - B|| after the optimal rigid transform A -> A' minimizing it (units of A)."""
    ca, cb = A.mean(0), B.mean(0)
    Ac, Bc = A - ca, B - cb
    U, _, Vt = np.linalg.svd(Ac.T @ Bc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    Aal = Ac @ R.T + cb
    return float(np.linalg.norm(Aal - B, axis=-1).max())


# --- PRIMARY metric: per-cell equilibrium discrepancy over live cells, at frame 1 -------------- #
diff_eq = np.linalg.norm(plx[EQ, live] - ref[EQ, live], axis=-1) / sigma      # [n_live]
D_eq = float(diff_eq.max())
# robustness: max over ALL post-relax frames (the equilibrium should be a held fixed point)
diff_all = np.linalg.norm(plx[EQ:, live] - ref[EQ:, live], axis=-1) / sigma   # [T, n_live]
D_eq_allframes = float(diff_all.max())

# gauge-removed (rigid) equilibrium discrepancy -- if ~ raw D_eq, no global drift inflates it
D_eq_kabsch = kabsch_resid(plx[EQ, live], ref[EQ, live]) / sigma

# --- alignment sanity: frame 0 must be the pristine IC on BOTH sides -------------------------- #
p0 = r["p0"].astype(np.float64)
frame0_ref_is_ic = float(np.abs(ref[0] - p0).max())
frame0_plx_is_ic = float(np.abs(plx[0] - p0).max())
# misaligned (WRONG) comparison x_plx(equilibrium) vs the IC -> ~ the relaxation displacement,
# orders above D_eq, proving the equilibrium is genuinely reached, not an accidental IC match.
mis_vs_ic = float(np.linalg.norm(plx[EQ, live] - ref[0, live], axis=-1).max() / sigma)

# --- force-balance cross-check at each side's equilibrium ------------------------------------- #
res_ref = force_infnorm(ref[EQ])           # |grad U| at the reference equilibrium (~ ref_residual)
res_plx = force_infnorm(plx[EQ])           # |grad U| at the Plexus equilibrium (the real test)

# --- fixed-point idempotence: after frame 1, relax is a no-op on both sides ------------------- #
plateau_plx = float(np.abs(plx[EQ + 1:, live] - plx[EQ:EQ + 1, live]).max()) if T > EQ else 0.0
plateau_ref = float(np.abs(ref[EQ + 1:, live] - ref[EQ:EQ + 1, live]).max()) if T > EQ else 0.0

# --- dead slots never moved on either side --------------------------------------------------- #
dead = ~live
dead_move_plx = float(np.abs(plx[:, dead] - plx[0:1, dead]).max()) if dead.any() else 0.0
dead_move_ref = float(np.abs(ref[:, dead] - ref[0:1, dead]).max()) if dead.any() else 0.0

# the magnitude the metric must resolve far under: the relaxation displacement itself
relax_disp = float(np.linalg.norm(ref[EQ, live] - ref[0, live], axis=-1).max() / sigma)

THRESHOLD = 1.0e-3
passed = bool(np.isfinite(D_eq) and D_eq < THRESHOLD
              and frame0_ref_is_ic == 0.0 and frame0_plx_is_ic == 0.0
              and dead_move_plx == 0.0
              and res_plx <= 5.0 * f_tol)          # the Plexus equilibrium is a genuine force balance

out = {
    "metric": "D_eq = max_{live i} ||x*_plx - x*_ref||_2 / sigma at the relaxed equilibrium (frame 1)",
    "value": D_eq, "threshold": THRESHOLD, "passed": passed,
    "n_live": n_live, "frames": T + 1, "sigma": sigma, "eps": eps, "alpha": alpha, "f_tol": f_tol,
    "D_eq_allframes": D_eq_allframes,
    "D_eq_kabsch_gauge_removed": D_eq_kabsch,
    "force_infnorm_plexus_equilibrium": res_plx,
    "force_infnorm_reference_equilibrium": res_ref,
    "reference_residual_at_xstar": ref_residual,
    "fixed_point_plateau_plexus": plateau_plx,
    "fixed_point_plateau_reference": plateau_ref,
    "frame0_ref_is_ic_resid": frame0_ref_is_ic, "frame0_plx_is_ic_resid": frame0_plx_is_ic,
    "misaligned_vs_ic": mis_vs_ic,
    "relaxation_displacement": relax_disp,
    "dead_move_plx": dead_move_plx, "dead_move_ref": dead_move_ref,
    "per_cell_diff_eq": diff_eq.tolist(),
}
os.makedirs(LOG, exist_ok=True)
with open(os.path.join(LOG, "diff.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("\nPASS" if passed else "\nFAIL", f" D_eq={D_eq:.3e}  threshold={THRESHOLD:.1e}")
