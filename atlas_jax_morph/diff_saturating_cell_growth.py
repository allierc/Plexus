"""diff_saturating_cell_growth -- score Plexus `grow_radius` against the jax-morph
SaturatingCellGrowth reference trajectory.

Runs in the PLEXUS (torch) env, NOT the oracle venv. Growth is ISOLATED (no relaxation, no
division) so the two sides keep the SAME cells in the SAME array slots and the radius trajectories
align cell-for-cell, frame-for-frame. Division is deliberately absent: its hazard
p = 1-exp(-division_rate*dt) never reads radius (division.py _dist), so division -- not growth -- is
what makes the anchor's cell counts diverge (124 vs 82); isolating growth is the faithful test of
THIS operator's contract, the per-cell radius ODE. dt = 2.0 (not 1) makes the mean-rate convention
observable (grow_radius returns delta = dr/dt; the engine recovers radius += dt*delta = dr).

Three comparisons on the reference's exact per-cell IC (radius r0, growth_rate k):

  PRIMARY  (D_max_A): the engine trajectory. Load config/atlas/saturating_cell_growth.yaml, run it
      through plexus.engine.run (out_path=None), read the recorded `radius` block
      out["sets"]["cell"]["state"]["radius"] and diff it against the reference scenario A over every
      frame, every live cell. This is grow_radius AS THE ENGINE RUNS IT.

  SECONDARY (D_max_B): the heterogeneous 6x6 grid. build() a 36-cell Hierarchy, set per-cell radius
      and growth_rate from the reference grid, and roll 20 macro-steps with the real engine
      first-order integration (radius += dt*delta). Spans k=0 (byte no-op), small-k near-linear
      growth, large-k near-instant saturation (Euler would overshoot at dt=2), r0==R (dr~0) and
      r0>R (relaxation DOWN to R, the no-clamp claim).

  NEGATIVE CONTROL (nc_B): re-roll scenario B with the /dt convention DROPPED (as if the op returned
      dr instead of dr/dt, so the engine applies radius += dt*dr = dt^2*delta each step). Its D vs
      the reference must be orders of magnitude above threshold -- proof the metric would catch a
      wiring defect (and that dt=2, where /dt and *dt do NOT cancel, exposes it).

evidence.value = max(D_max_A, D_max_B). PASS iff value <= threshold (1e-5).
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))

THRESHOLD = 1.0e-5
REF = os.path.join(HERE, "_oracle", "runs", "diff_saturating_cell_growth", "reference.npz")
SPEC_A = os.path.join(PLEXUS, "config", "atlas", "saturating_cell_growth.yaml")
SPEC_B = os.path.join(HERE, "saturating_cell_growth_gridB.yaml")

# --- load the atlas anti-chamber so the specs can name grow_radius / seed_state -------------- #
import plexus.operators  # noqa: F401  self-registers the core library
import importlib
import plexus.operators.candidates as C
for fn in sorted(os.listdir(os.path.dirname(C.__file__))):
    if fn.startswith(("jax_morph_", "atlas_")) and fn.endswith(".py"):
        importlib.import_module(f"plexus.operators.candidates.{fn[:-3]}")

from plexus.schema import load
from plexus.engine import run, build
from plexus.models.registry import get_operator

ref = np.load(REF, allow_pickle=True)
radiusA = ref["radiusA"].astype(np.float64)          # [T+1, 4]
r0B = ref["r0B"].astype(np.float32)                  # [36]
kB = ref["kB"].astype(np.float32)                    # [36]
radiusB = ref["radiusB"].astype(np.float64)          # [T+1, 36]
R = float(ref["R"]); DT = float(ref["dt"]); T = int(ref["T"])

# ------------------------------------------------------------------------------------------- #
#  PRIMARY -- the engine trajectory (uniform IC), exactly as run_spec ran it
# ------------------------------------------------------------------------------------------- #
sim = load(SPEC_A)
assert sim.n_frames == T, (sim.n_frames, T)
assert abs(float(sim.dt) - DT) < 1e-12, (sim.dt, DT)
_, out = run(sim, out_path=None, device="cpu", progress=False)
rad_eng = np.asarray(out["sets"]["cell"]["state"]["radius"], dtype=np.float64)   # [T+1, 8, 1]
occ_eng = np.asarray(out["sets"]["cell"]["occ"]).astype(bool)                    # [T+1, 8]
rad_eng = rad_eng[..., 0]                                                        # [T+1, 8]
n_live = radiusA.shape[1]                                                        # 4
assert rad_eng.shape[0] == radiusA.shape[0], (rad_eng.shape, radiusA.shape)

diff_A = np.abs(rad_eng[:, :n_live] - radiusA)                                   # live cells only
per_frame_A = np.array([diff_A[t][occ_eng[t, :n_live]].max() if occ_eng[t, :n_live].any() else 0.0
                        for t in range(diff_A.shape[0])])
D_max_A = float(per_frame_A.max())
frame0_is_ic = float(np.abs(rad_eng[0, :n_live] - radiusA[0]).max())            # frame 0 must be the seeded IC

# ------------------------------------------------------------------------------------------- #
#  SECONDARY -- heterogeneous 6x6 grid, real operator + real engine integration
# ------------------------------------------------------------------------------------------- #
def roll(scale):
    """Roll grow_radius for T steps on the 36-cell grid; `scale` multiplies the applied delta
    (scale=1 -> the correct radius += dt*delta; scale=dt -> the negative-control dt*dr bug)."""
    simB = load(SPEC_B)
    H = build(simB, "cpu")
    cell = H.level("cell")
    ra, rb = cell.state_schema["radius"]
    ga, gb = cell.state_schema["growth_rate"]
    st = cell.state.clone()
    st[:36, ra:rb] = torch.as_tensor(r0B, dtype=st.dtype).reshape(-1, 1)
    st[:36, ga:gb] = torch.as_tensor(kB, dtype=st.dtype).reshape(-1, 1)
    cell.state = st
    op = get_operator("grow_radius")({"max_radius": R, "_at": "cell"}, "cpu")
    traj = [cell.get("radius")[:36, 0].clone().numpy()]                          # frame 0 = IC
    for _ in range(T):
        d = op.forward(H)["cell"]                                                # [36, 1] mean-rate delta
        new = cell.state.clone()
        new[:, ra:rb] = new[:, ra:rb] + DT * (scale * d)                         # engine first-order step
        cell.state = new
        traj.append(cell.get("radius")[:36, 0].clone().numpy())
    return np.asarray(traj, dtype=np.float64)                                    # [T+1, 36]

G_B = roll(scale=1.0)
diff_B = np.abs(G_B - radiusB)
D_max_B = float(diff_B.max())

# per-cell no-clamp / no-op sanity on the Plexus side
k0_noop_drift = float(np.abs(G_B[:, kB == 0.0] - r0B[kB == 0.0][None, :]).max())
aboveR_last_min = float(G_B[-1, r0B > R].min())                                  # should relax to ~R from above

# negative control: the dropped-/dt wiring bug (radius += dt*dr = dt^2*delta)
G_nc = roll(scale=DT)
nc_B = float(np.abs(G_nc - radiusB).max())

value = max(D_max_A, D_max_B)
passed = bool(value <= THRESHOLD)

result = {
    "threshold": THRESHOLD,
    "D_max_A_uniform_engine": D_max_A,
    "D_max_B_grid": D_max_B,
    "value": value,
    "passed": passed,
    "frame0_is_ic_residual": frame0_is_ic,
    "negative_control_nc_B_drop_dt": nc_B,
    "k0_noop_drift": k0_noop_drift,
    "aboveR_last_min_should_be_R": aboveR_last_min,
    "R": R, "dt": DT, "T": T, "n_cells_grid": int(r0B.shape[0]),
    "oracle_run": "diff_saturating_cell_growth",
    "spec_A": os.path.relpath(SPEC_A, PLEXUS),
    "spec_B": os.path.relpath(SPEC_B, PLEXUS),
}

out_dir = os.path.join(PLEXUS, "log", "atlas_jax", "saturating_cell_growth")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "diff.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
print(f"\n-> {os.path.join(out_dir, 'diff.json')}")
print("PASS" if passed else "FAIL", f"  value={value:.3e}  threshold={THRESHOLD:.1e}")
