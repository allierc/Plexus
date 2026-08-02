"""Score the adhere/harmonic differential: Plexus engine trajectory vs the jax-morph Harmonic
overdamped reference, per cell, per frame. Runs in the Plexus (torch) env; no re-simulation.

  reference : atlas_jax_morph/_oracle/runs/diff_harmonic/reference.npz  (jxm BrownianDynamics kT=0,
              Harmonic(k=1.0, r_cutoff_frac=2.5))
  plexus    : graphs_data/atlas/harmonic/trajectory.npz                 (config/atlas/harmonic.yaml)

PRIMARY metric  D_pos = max over frames t=0..160 and LIVE cells i of ||x_plx-x_ref||_2 / sigma.
Threshold       1.0e-3 (sigma).  This tests the FORCE LAW including the ADHESIVE TAIL and the hard
C0 cutoff (the feature distinguishing harmonic from the already-validated purely-repulsive
soft_sphere/hertzian). Corroborators: single-step IC force residual, gyration rel-err + SHAPE
(adhesion CONTRACTS the cluster vs soft_sphere's expansion), dead-slot immobility, frame-0 == IC on
both sides, misaligned-frame control. NEGATIVE CONTROL (from the oracle): a purely-repulsive
SoftSphere with the matched core diverges the trajectory by O(1) sigma -- reported here to show the
metric resolves the adhesive tail. Writes diff.json into log/atlas_jax/harmonic/.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REF = os.path.join(HERE, "..", "runs", "diff_harmonic", "reference.npz")
PLX = os.path.join(PLEXUS, "graphs_data", "atlas", "harmonic", "trajectory.npz")
LOG = os.path.join(PLEXUS, "log", "atlas_jax", "harmonic")

r = np.load(REF)
p = np.load(PLX)

ref = r["position"].astype(np.float64)      # [T+1, CAP, 2]  index t = IC after t steps
plx = p["cell__pos"].astype(np.float64)     # [T+1, CAP, 2]  frame t = IC after t steps (after_frame:1)
occ = p["cell__occ"].astype(bool)           # [T+1, CAP]
aliv = r["alive"].astype(bool)              # [T+1, CAP]
sigma = float(r["sigma"]); dt = float(r["dt"]); k = float(r["k"])
N = int(r["N"]); CAP = int(r["CAP"]); T = int(r["NSTEPS"])

assert ref.shape == plx.shape == (T + 1, CAP, 2), (ref.shape, plx.shape)
# liveness must agree and be fixed (no birth/death): compare on the shared live set
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

# alternative (WRONG) alignment plx[t] vs ref[t-1]; should be clearly LARGER -> proves the aligned
# convention is the right one and not accidentally matching a shifted trajectory.
mis = float(np.linalg.norm(plx[1:, live] - ref[:-1, live], axis=-1).max() / sigma)

# --- corroborator 1: single-step IC force residual (raw force law, pre-compounding) ---------- #
# overdamped: dx over the first step = dt * F(IC); recover F_plx = (x1 - x0)/dt, compare to F_ref.
F_ref = r["force_ic"].astype(np.float64)[live]           # [n_live,2]  -grad U at IC (Harmonic)
F_plx = ((plx[1] - plx[0]) / dt)[live]                   # [n_live,2]  first-step velocity
force_res = float(np.linalg.norm(F_plx - F_ref, axis=-1).max() / (k * sigma))
force_ref_max = float(np.linalg.norm(F_ref, axis=-1).max())


# --- corroborator 2: radius-of-gyration trajectory (rel err + SHAPE: adhesion contracts) ----- #
def gyr(traj):
    out = []
    for t in range(T + 1):
        q = traj[t, live]; c = q.mean(0)
        out.append(np.sqrt(((q - c) ** 2).sum(1).mean()))
    return np.asarray(out)


g_ref, g_plx = gyr(ref), gyr(plx)
gyr_rel = float(np.abs(g_plx - g_ref).max() / max(g_ref.max(), 1e-9))
adhesion_contracts = bool(g_ref[-1] < g_ref[0])          # Harmonic pulls the cluster IN (adhesion on)

# --- corroborator 3: dead slots never moved on either side ---------------------------------- #
dead = ~live
dead_move_plx = float(np.abs(plx[:, dead] - plx[0:1, dead]).max()) if dead.any() else 0.0
dead_move_ref = float(np.abs(ref[:, dead] - ref[0:1, dead]).max()) if dead.any() else 0.0

# --- NEGATIVE CONTROL (oracle-side): SoftSphere (adhesion OFF, matched core) on the SAME IC --- #
# with k = eps = 1, sigma = 1 the repulsive cores are identical, so this isolates the adhesive tail.
ref_ss = r["position_ss"].astype(np.float64)
nc_traj_Dpos = float((np.linalg.norm(ref_ss[:, live] - ref[:, live], axis=-1) / sigma).max())
F_ss = r["force_ic_ss"].astype(np.float64)
nc_force_rel = float(np.abs(F_ss - r["force_ic"].astype(np.float64)).max()
                     / max(1e-12, np.abs(r["force_ic"].astype(np.float64)).max()))
g_ss = gyr(ref_ss)

# --- oracle self-guard carried through: 2-cell scan vs analytic k*(sigma-r)|_{r<r_c} ---------- #
scan_ref_vs_analytic_max = float(np.abs(r["f_ref_scan"] - r["f_an_scan"]).max())

THRESHOLD = 1.0e-3
passed = bool(np.isfinite(D_pos) and D_pos < THRESHOLD
              and frame0_ref_is_ic == 0.0 and frame0_plx_is_ic == 0.0
              and dead_move_plx == 0.0)

out = {
    "metric": "D_pos = max_{t,live i} ||x_plx-x_ref||_2 / sigma",
    "value": D_pos, "threshold": THRESHOLD, "passed": passed,
    "n_live": n_live, "frames": T + 1, "sigma": sigma, "dt": dt, "k": k,
    "r_cutoff_frac": float(r["r_cutoff_frac"]), "r_c": float(r["r_c"]),
    "D_pos_argframe": argt, "D_pos_final_frame": D_pos_final,
    "misaligned_alt_Dpos": mis,
    "frame0_ref_is_ic_resid": frame0_ref_is_ic, "frame0_plx_is_ic_resid": frame0_plx_is_ic,
    "force_residual_norm": force_res, "force_ref_max": force_ref_max,
    "gyration_rel_err": gyr_rel,
    "gyration_ref_first": float(g_ref[0]), "gyration_ref_last": float(g_ref[-1]),
    "gyration_plx_first": float(g_plx[0]), "gyration_plx_last": float(g_plx[-1]),
    "adhesion_contracts_cluster": adhesion_contracts,
    "dead_move_plx": dead_move_plx, "dead_move_ref": dead_move_ref,
    "max_live_displacement_ref": float(np.linalg.norm(ref[-1, live] - ref[0, live], axis=-1).max()),
    "negctrl_softsphere_traj_Dpos": nc_traj_Dpos,
    "negctrl_softsphere_force_rel": nc_force_rel,
    "negctrl_softsphere_gyration_last": float(g_ss[-1]),
    "oracle_scan_ref_vs_analytic_max": scan_ref_vs_analytic_max,
    "per_frame_max_first5": per_frame_max[:5].tolist(),
    "per_frame_max_last5": per_frame_max[-5:].tolist(),
}
os.makedirs(LOG, exist_ok=True)
with open(os.path.join(LOG, "diff.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("\nPASS" if passed else "\nFAIL", f" D_pos={D_pos:.3e}  threshold={THRESHOLD:.1e}",
      f"  (neg-ctrl SoftSphere D_pos={nc_traj_Dpos:.3f} = {nc_traj_Dpos/THRESHOLD:.0f}x threshold)")
