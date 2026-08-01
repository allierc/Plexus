"""Score the adhere/lennard_jones differential: Plexus engine trajectory vs the jax-morph
LennardJones overdamped reference, per cell, per frame. Runs in the Plexus (torch) env; no
re-simulation.

  reference : Atlas_jax_morph/_oracle/runs/diff_lennard_jones/reference.npz  (jxm BrownianDynamics kT=0)
  plexus    : graphs_data/atlas/lennard_jones/trajectory.npz                 (config/atlas/lennard_jones.yaml)

Primary metric  D_pos = max over frames t=0..100 and LIVE cells i of ||x_plx-x_ref||_2 / sigma.
Threshold       1.0e-3 (sigma).  Corroborators: single-step IC force residual, pair-separation
trajectory agreement, the SoftSphere negative control (LJ diverges from it), dead-slot immobility,
frame-0 == IC on both sides, and the oracle's own scan-vs-analytic guard. Writes diff.json into
log/atlas/lennard_jones/.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REF = os.path.join(HERE, "..", "runs", "diff_lennard_jones", "reference.npz")
PLX = os.path.join(PLEXUS, "graphs_data", "atlas", "lennard_jones", "trajectory.npz")
ORC_SUMMARY = os.path.join(HERE, "..", "runs", "diff_lennard_jones", "summary.json")
LOG = os.path.join(PLEXUS, "log", "atlas", "lennard_jones")

r = np.load(REF)
p = np.load(PLX)
orc = json.load(open(ORC_SUMMARY))

ref = r["position"].astype(np.float64)      # [T+1, CAP, 2]  index t = IC after t steps
plx = p["cell__pos"].astype(np.float64)     # [T+1, CAP, 2]  frame t = IC after t steps (after_frame:1)
occ = p["cell__occ"].astype(bool)           # [T+1, CAP]
aliv = r["alive"].astype(bool)              # [T+1, CAP]
pos_ss = r["pos_ss"].astype(np.float64)     # SoftSphere negative-control trajectory
sigma = float(r["sigma"]); dt = float(r["dt"]); eps = float(r["eps"])
N = int(r["N"]); CAP = int(r["CAP"]); T = int(r["NSTEPS"])

assert ref.shape == plx.shape == (T + 1, CAP, 2), (ref.shape, plx.shape)
assert np.array_equal(occ, aliv), "live masks differ between Plexus and reference"
live = aliv[0]                              # [CAP] fixed live set
n_live = int(live.sum())

# --- alignment sanity: frame 0 must be the pristine IC on BOTH sides ------------------------- #
p0 = r["p0"].astype(np.float64)
frame0_ref_is_ic = float(np.abs(ref[0] - p0).max())
frame0_plx_is_ic = float(np.abs(plx[0] - p0).max())

# --- PRIMARY metric: per-cell position discrepancy over live cells, all frames --------------- #
diff = np.linalg.norm(plx[:, live] - ref[:, live], axis=-1) / sigma   # [T+1, n_live]
per_frame_max = diff.max(axis=1)                                      # [T+1]
D_pos = float(diff.max())
argt = int(per_frame_max.argmax())
D_pos_final = float(per_frame_max[-1])

# WRONG alignment plx[t] vs ref[t-1] -- should be clearly LARGER (proves the aligned convention). #
mis = float(np.linalg.norm(plx[1:, live] - ref[:-1, live], axis=-1).max() / sigma)

# --- corroborator 1: single-step IC force residual (raw force law, pre-compounding) ---------- #
F_ref = r["force_ic"].astype(np.float64)[live]           # [n_live,2]  -grad U at IC
F_plx = ((plx[1] - plx[0]) / dt)[live]                   # [n_live,2]  first-step velocity == mobility*F
force_res = float(np.linalg.norm(F_plx - F_ref, axis=-1).max() / (eps / sigma))
force_ref_max = float(np.linalg.norm(F_ref, axis=-1).max())

# --- corroborator 2: per-pair separation trajectory agreement (LJ-specific) ------------------ #
def pair_seps(pos):
    return np.array([[np.linalg.norm(pos[t, 2 * k] - pos[t, 2 * k + 1]) for k in range(N // 2)]
                     for t in range(T + 1)])             # [T+1, npairs]
sep_ref, sep_plx = pair_seps(ref), pair_seps(plx)
pair_sep_maxdiff = float(np.abs(sep_plx - sep_ref).max() / sigma)
pair_sep_final_ref = sep_ref[-1].tolist()               # tail pairs -> ~1.0, cutoff pair frozen
pair_sep_final_plx = sep_plx[-1].tolist()

# --- corroborator 3: SoftSphere NEGATIVE CONTROL -- Plexus tracks LJ, NOT the adhesion-off law - #
plx_vs_lj = float(np.linalg.norm(plx[:, live] - ref[:, live], axis=-1).max() / sigma)     # == D_pos
plx_vs_ss = float(np.linalg.norm(plx[:, live] - pos_ss[:, live], axis=-1).max() / sigma)  # should be ~0.35
neg_control_ratio = plx_vs_ss / max(plx_vs_lj, 1e-12)   # how many x closer to LJ than to SoftSphere

# --- corroborator 4: dead slots never moved on either side ----------------------------------- #
dead = ~live
dead_move_plx = float(np.abs(plx[:, dead] - plx[0:1, dead]).max()) if dead.any() else 0.0
dead_move_ref = float(np.abs(ref[:, dead] - ref[0:1, dead]).max()) if dead.any() else 0.0

THRESHOLD = 1.0e-3
passed = bool(np.isfinite(D_pos) and D_pos < THRESHOLD
              and frame0_ref_is_ic == 0.0 and frame0_plx_is_ic == 0.0
              and dead_move_plx == 0.0)

out = {
    "metric": "D_pos = max_{t,live i} ||x_plx-x_ref||_2 / sigma",
    "value": D_pos, "threshold": THRESHOLD, "passed": passed,
    "n_live": n_live, "frames": T + 1, "sigma": sigma, "dt": dt, "eps": eps,
    "D_pos_argframe": argt, "D_pos_final_frame": D_pos_final,
    "misaligned_alt_Dpos": mis,
    "frame0_ref_is_ic_resid": frame0_ref_is_ic, "frame0_plx_is_ic_resid": frame0_plx_is_ic,
    "force_residual_norm": force_res, "force_ref_max": force_ref_max,
    "pair_sep_trajectory_maxdiff_over_sigma": pair_sep_maxdiff,
    "pair_sep_final_ref": pair_sep_final_ref, "pair_sep_final_plx": pair_sep_final_plx,
    "neg_control_plx_vs_softsphere_over_sigma": plx_vs_ss,
    "neg_control_plx_vs_lj_over_sigma": plx_vs_lj,
    "neg_control_ratio_softsphere_over_lj": neg_control_ratio,
    "dead_move_plx": dead_move_plx, "dead_move_ref": dead_move_ref,
    "max_live_displacement_ref": float(np.linalg.norm(ref[-1, live] - ref[0, live], axis=-1).max()),
    "oracle_scan_ref_vs_analytic_max": orc.get("scan_ref_vs_analytic_max_below_ron"),
    "oracle_force_at_contact": orc.get("force_at_contact"),
    "oracle_force_beyond_cutoff": orc.get("force_beyond_cutoff_2p6sigma"),
    "oracle_euler_matches_bit_for_bit": orc.get("euler_convention_matches_bit_for_bit"),
    "per_frame_max_first5": per_frame_max[:5].tolist(),
    "per_frame_max_last5": per_frame_max[-5:].tolist(),
}
os.makedirs(LOG, exist_ok=True)
with open(os.path.join(LOG, "diff.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("\nPASS" if passed else "\nFAIL", f" D_pos={D_pos:.3e}  threshold={THRESHOLD:.1e}")
