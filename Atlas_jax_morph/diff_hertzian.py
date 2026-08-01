"""diff_hertzian -- score Plexus `adhere:hertzian` against the jax-morph Hertzian reference.

Runs in the PLEXUS (torch) env, NOT the oracle venv. Hertzian is a pure FORCE law, so the test
isolates the force. Two comparisons on the reference's exact initial conditions, plus a negative
control:

  PRIMARY (force_field_rel_err): the heterogeneous force FIELD. Build the 7-cell cluster from
      config/atlas/hertzian.yaml (positions POS7 + [20,20]), register HETEROGENEOUS radius (0.40-0.70)
      and PER-CELL epsilon as buffers, and call adhere:hertzian.forward(H) ONCE. The emitted velocity
      at mobility=1 IS the operator's autodiff force. Diff vs the reference Hertzian(epsilon).forces
      over the 7 live cells and both components, relative to max|F_ref|. Exercises sigma = r_i + r_j
      (size-consistency) and the arithmetic-mean per-cell epsilon mix. No integrator confound.

  SECONDARY (traj_pos_max_abs): the deterministic overdamped-Euler trajectory. Load
      config/atlas/hertzian.yaml (uniform radius 0.5, epsilon 2.0, dt=0.1, 40 frames), run it through
      plexus.engine.run (out_path=None) exactly as run_spec does, centroid-align, and diff the
      per-cell per-frame positions vs the reference BrownianDynamics(Hertzian, kT=0) trajectory.

  NEGATIVE CONTROL (nc_*): SoftSphere (exponent 2) vs Hertzian (exponent 5/2) on the same config --
      the wrong-exponent defect the verdict hinges on. Two forms: the reference-internal
      SoftSphere.forces vs Hertzian.forces, and the Plexus operator's F vs the SoftSphere reference.
      Both must land >> THRESHOLD_FORCE, proving the metric resolves the neighbouring implementation.

evidence.value = force_field_rel_err (PRIMARY). passed = (force_field_rel_err <= 1e-4) AND
(traj_pos_max_abs <= 1e-3).
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))

THRESHOLD_FORCE = 1.0e-4
THRESHOLD_TRAJ = 1.0e-3
REF = os.path.join(HERE, "_oracle", "runs", "diff_hertzian", "reference.npz")
SPEC = os.path.join(PLEXUS, "config", "atlas", "hertzian.yaml")

# --- load the atlas anti-chamber so the spec can name adhere:hertzian ------------------------ #
import plexus.operators  # noqa: F401
import importlib
import plexus.operators.candidates as C
for fn in sorted(os.listdir(os.path.dirname(C.__file__))):
    if fn.startswith(("jax_morph_", "atlas_")) and fn.endswith(".py"):
        importlib.import_module(f"plexus.operators.candidates.{fn[:-3]}")

from plexus.schema import load
from plexus.engine import run, build
from plexus.models.registry import get_operator

ref = np.load(REF, allow_pickle=True)
POS7 = ref["pos7"].astype(np.float32)               # [7,2] origin-centred
RAD_HET = ref["rad_het"].astype(np.float32)         # [7]
EPS_HET = ref["eps_het"].astype(np.float32)         # [7]
F_ref_H = ref["F_ref_H"].astype(np.float64)         # [7,2] Hertzian reference force
F_ss_H = ref["F_ss_H"].astype(np.float64)           # [7,2] SoftSphere (wrong-exponent control)
posU = ref["posU"].astype(np.float64)               # [T+1,7,2] reference trajectory
DT = float(ref["dt"]); T = int(ref["T"])

# ------------------------------------------------------------------------------------------- #
#  PRIMARY -- heterogeneous force field, adhere:hertzian.forward called directly
# ------------------------------------------------------------------------------------------- #
simH = load(SPEC)
H = build(simH, "cpu")
cell = H.level("cell")
buf = cell.n                                        # 16
px0, px1 = cell.state_schema["pos"]
# assert the built live positions are POS7 translated by +[20,20] (forces are translation-invariant)
pos_built = cell.state[:7, px0:px1].numpy()
offset = pos_built - POS7
assert np.allclose(offset, offset[0], atol=1e-5), ("built IC is not a pure translation of POS7", offset)

# register HETEROGENEOUS radius + PER-CELL epsilon as buffers the operator reads (live 7; dead padded)
rad_buf = torch.full((buf,), 0.5); rad_buf[:7] = torch.as_tensor(RAD_HET)
eps_buf = torch.full((buf,), 1.0); eps_buf[:7] = torch.as_tensor(EPS_HET)
cell.register_buffer("radius", rad_buf)
cell.register_buffer("epsilon", eps_buf)

op = get_operator("adhere", "hertzian")({"epsilon_field": "epsilon", "mobility": 1.0, "_at": "cell"}, "cpu")
vel = op.forward(H)["cell"].detach().numpy()        # [16,2]; live 7 = mobility*force = force
F_plexus_H = vel[:7].astype(np.float64)

force_field_rel_err = float(np.abs(F_plexus_H - F_ref_H).max() / max(1e-12, np.abs(F_ref_H).max()))
newton_ref = float(np.abs(F_ref_H.sum(0)).max())    # sum of internal forces ~ 0
newton_plexus = float(np.abs(F_plexus_H.sum(0)).max())

# negative controls (the metric MUST fire on the wrong exponent)
nc_ref_softsphere_vs_hertzian = float(np.abs(F_ss_H - F_ref_H).max() / max(1e-12, np.abs(F_ref_H).max()))
nc_plexus_vs_softsphere = float(np.abs(F_plexus_H - F_ss_H).max() / max(1e-12, np.abs(F_ss_H).max()))

# ------------------------------------------------------------------------------------------- #
#  SECONDARY -- the deterministic overdamped-Euler trajectory, exactly as run_spec runs it
# ------------------------------------------------------------------------------------------- #
simU = load(SPEC)
assert simU.n_frames == T, (simU.n_frames, T)
assert abs(float(simU.dt) - DT) < 1e-6, (simU.dt, DT)   # DT is stored float32 in the reference npz
_, out = run(simU, out_path=None, device="cpu", progress=False)
pos_eng = np.asarray(out["sets"]["cell"]["pos"], dtype=np.float64)[:, :7, :]   # [T+1,7,2]
occ_eng = np.asarray(out["sets"]["cell"]["occ"]).astype(bool)[:, :7]
assert pos_eng.shape[0] == posU.shape[0], (pos_eng.shape, posU.shape)


def centroid_align(p):
    """Remove the constant translation by subtracting the frame-0 centroid (conserved: sum F = 0)."""
    return p - p[0].mean(0, keepdims=True)


pe, po = centroid_align(pos_eng), centroid_align(posU)
per_frame = np.array([np.abs(pe[t] - po[t])[occ_eng[t]].max() if occ_eng[t].any() else 0.0
                      for t in range(pe.shape[0])])
traj_pos_max_abs = float(per_frame.max())
frame0_ic_residual = float(np.abs(pe[0] - po[0]).max())            # frame 0 must be the shared IC
traj_final_pos = float(np.abs(pe[-1] - po[-1]).max())

value = force_field_rel_err
passed = bool(force_field_rel_err <= THRESHOLD_FORCE and traj_pos_max_abs <= THRESHOLD_TRAJ)

result = {
    "threshold_force": THRESHOLD_FORCE, "threshold_traj": THRESHOLD_TRAJ,
    "force_field_rel_err": force_field_rel_err,
    "traj_pos_max_abs": traj_pos_max_abs,
    "value": value, "passed": passed,
    "H_force_ref_maxabs": float(np.abs(F_ref_H).max()),
    "H_force_plexus_maxabs": float(np.abs(F_plexus_H).max()),
    "newton_thirdlaw_ref": newton_ref, "newton_thirdlaw_plexus": newton_plexus,
    "negative_control_ref_softsphere_vs_hertzian": nc_ref_softsphere_vs_hertzian,
    "negative_control_plexus_vs_softsphere": nc_plexus_vs_softsphere,
    "traj_frame0_ic_residual": frame0_ic_residual,
    "traj_final_pos_max_abs": traj_final_pos,
    "dt": DT, "T": T, "n_live": 7,
    "oracle_run": "diff_hertzian",
    "spec": os.path.relpath(SPEC, PLEXUS),
}

out_dir = os.path.join(PLEXUS, "log", "atlas", "hertzian")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "diff.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
print(f"\n-> {os.path.join(out_dir, 'diff.json')}")
print("PASS" if passed else "FAIL",
      f"  force_rel={force_field_rel_err:.3e} (thr {THRESHOLD_FORCE:.0e})  "
      f"traj={traj_pos_max_abs:.3e} (thr {THRESHOLD_TRAJ:.0e})")
