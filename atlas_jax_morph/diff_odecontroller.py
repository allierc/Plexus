"""diff_odecontroller -- score the Plexus `regulate` (ode_generic) operator against the
jax-morph GeneNetworkConnectionist reference trajectory.

Runs in the PLEXUS (torch) environment, NOT the oracle venv. Two comparisons, both on the
IDENTICAL circuit / initial condition / frozen drive the oracle used:

  PRIMARY  (D_max_uniform): the engine trajectory. Load config/atlas/odecontroller.yaml and
      run it through plexus.engine.run (out_path=None -> no zarr); read the recorded `gene`
      block out["sets"]["cell"]["state"]["gene"] [22, 8, 3] and diff it against the reference
      y_uniform over every recorded frame, every LIVE cell, every gene component. This is the
      operator AS THE ENGINE RUNS IT (self-solved adaptive Dopri5, then gene += dt*delta).

  SECONDARY (D_max_distinct): the maps=[] / intracellular corroboration. The uniform run
      shares one drive across cells, so it cannot see cell-to-cell coupling. Build the same
      operator on a hand-set Hierarchy with a DIFFERENT frozen drive per cell (the reference
      u_distinct) and roll 21 steps with the real engine integration; diff against y_distinct.
      If the operator leaked any cell-to-cell coupling (a violation of the maps=[] identity
      that separates `regulate` from `signal`), the four per-cell trajectories would diverge
      from the reference here. Reported, not the pass gate.

evidence.value = max(D_max_uniform, D_max_distinct). PASS iff <= threshold (5e-3).
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))

THRESHOLD = 5.0e-3
REF = os.path.join(HERE, "_oracle", "runs", "diff_odecontroller", "reference.npz")
SPEC = os.path.join(PLEXUS, "config", "atlas", "odecontroller.yaml")

# --- load the atlas anti-chamber so the spec can name `regulate` / `seed_state` ------------- #
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
y_uniform = ref["y_uniform"].astype(np.float64)      # [22, 8, 3]
y_distinct = ref["y_distinct"].astype(np.float64)    # [22, 8, 3]
alive = ref["alive"].astype(bool)                    # [22, 8]
u_distinct = ref["u_distinct"].astype(np.float32)    # [4, 2]
n_live = int(ref["n_live"])                          # 4
n_steps = int(ref["n_steps"])                        # 21
gene_order = [str(x) for x in ref["gene_order"]]

sim = load(SPEC)
assert sim.n_frames == n_steps, (sim.n_frames, n_steps)

# ------------------------------------------------------------------------------------------- #
#  PRIMARY -- the engine trajectory (uniform drive), exactly as run_spec ran it
# ------------------------------------------------------------------------------------------- #
_, out = run(sim, out_path=None, device="cpu", progress=False)
G_eng = np.asarray(out["sets"]["cell"]["state"]["gene"], dtype=np.float64)   # [22, 8, 3]
assert G_eng.shape == y_uniform.shape, (G_eng.shape, y_uniform.shape)

live_mask = alive.copy()                              # [22, 8] -- compare only live slots
# frame 0 must be the seeded initial condition (all genes 0) on BOTH sides
diff_u = np.abs(G_eng - y_uniform)
per_frame_u = np.array([diff_u[t][live_mask[t]].max() if live_mask[t].any() else 0.0
                        for t in range(diff_u.shape[0])])
D_max_uniform = float(per_frame_u.max())
frame0_is_ic = float(np.abs(G_eng[0][:n_live]).max())          # engine's initial gene block (should be 0)

# ------------------------------------------------------------------------------------------- #
#  SECONDARY -- distinct per-cell drive, real operator + real integration (no cell coupling)
# ------------------------------------------------------------------------------------------- #
def regulate_opspec(sim):
    for o in sim.operators:
        if o.op == "regulate":
            return o
    raise SystemExit("no regulate op in spec")

o = regulate_opspec(sim)
H = build(sim, "cpu")
cell = H.level("cell")
ga, gb = cell.state_schema["gene"]
da, db = cell.state_schema["drive"]
st = cell.state.clone()
st[:n_live, ga:gb] = 0.0                                        # genes 0 (y0)
st[:n_live, da:db] = torch.as_tensor(u_distinct, dtype=st.dtype)  # a DIFFERENT frozen drive per cell
cell.state = st

op = get_operator("regulate", o.impl)(
    {**o.params, "to": o.to, "from": o.frm, "_at": "cell"}, "cpu")

dt = float(sim.dt)
traj = [cell.get("gene")[:n_live].clone().numpy()]             # frame 0 = initial (0)
for _ in range(n_steps):
    d = op.forward(H)["cell"]                                  # [8, 3] mean-rate delta (self-solved)
    new = cell.state.clone()
    new[:, ga:gb] = new[:, ga:gb] + dt * d                     # engine's first-order step gene += dt*delta
    cell.state = new
    traj.append(cell.get("gene")[:n_live].clone().numpy())
G_dist = np.asarray(traj, dtype=np.float64)                    # [22, 4, 3]

live_d = alive[:, :n_live]
diff_d = np.abs(G_dist - y_distinct[:, :n_live])
per_frame_d = np.array([diff_d[t][live_d[t]].max() if live_d[t].any() else 0.0
                        for t in range(diff_d.shape[0])])
D_max_distinct = float(per_frame_d.max())

value = max(D_max_uniform, D_max_distinct)
passed = bool(value <= THRESHOLD)

result = {
    "threshold": THRESHOLD,
    "D_max_uniform": D_max_uniform,
    "D_max_distinct": D_max_distinct,
    "value": value,
    "passed": passed,
    "frame0_engine_ic_maxabs": frame0_is_ic,
    "n_frames": int(y_uniform.shape[0]),
    "n_live": n_live,
    "n_gene": int(y_uniform.shape[-1]),
    "gene_order": gene_order,
    "per_frame_max_uniform": per_frame_u.tolist(),
    "engine_gene_last_cell0": G_eng[-1][0].tolist(),
    "ref_gene_last_cell0": y_uniform[-1][0].tolist(),
    "distinct_engine_last": G_dist[-1].tolist(),
    "distinct_ref_last": y_distinct[-1][:n_live].tolist(),
    "reference_run": os.path.relpath(os.path.dirname(REF), PLEXUS),
    "engine_evidence": "log/atlas/odecontroller/",
}
OUT = os.path.join(HERE, "_oracle", "runs", "diff_odecontroller", "diff.json")
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps({k: v for k, v in result.items()
                  if k not in ("per_frame_max_uniform",)}, indent=2))
print(f"\nwrote {OUT}")
print("PASS" if passed else "FAIL", f"  value={value:.3e}  threshold={THRESHOLD:.1e}")
