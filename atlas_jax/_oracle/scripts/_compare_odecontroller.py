"""Compute the `regulate` differential metric: Plexus engine gene trajectory vs the
jax-morph GeneNetworkConnectionist reference (_oracle/runs/diff_odecontroller/reference.npz).

Runs in the PLEXUS (torch) env, NOT the oracle venv -- it diffs the reference arrays against
the Plexus engine's own recorded gene block. Two comparisons:

  UNIFORM   -- the real engine.run of config/atlas_jax/odecontroller.yaml (the spec run_spec.py
               executes). Primary metric D_inf over 22 frames x 4 live cells x 3 genes.
  DISTINCT  -- a per-cell frozen drive (u_distinct), stepped through the engine's own
               _integrate (gene += dt*delta) exactly as engine.run does, since seed_state
               cannot express a per-cell drive. Corroboration that the operator leaks NO
               cell-to-cell coupling (the maps=[] intracellular identity).

Writes diff.json into the oracle run folder next to reference.npz.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS = os.path.abspath(os.path.join(HERE, "..", ".."))            # .../atlas_jax
sys.path.insert(0, ATLAS)                                          # for run_spec (adds src to path)

import run_spec                                                    # noqa: E402  (inserts src on import)
run_spec.load_atlas_candidates()                                  # register `regulate`/`seed_state`
import plexus.operators                                           # noqa: E402,F401
from plexus.schema import load                                    # noqa: E402
from plexus import engine                                         # noqa: E402
from plexus.engine import build, _integrate, _resolve_emit        # noqa: E402
from plexus.models.registry import get_operator                   # noqa: E402

RUN = os.path.join(ATLAS, "_oracle", "runs", "diff_odecontroller")
SPEC = os.path.join(ATLAS, "..", "config", "atlas_jax", "odecontroller.yaml")

REF = np.load(os.path.join(RUN, "reference.npz"), allow_pickle=True)
y_uniform = REF["y_uniform"].astype(np.float64)                   # [22, 8, 3]
y_distinct = REF["y_distinct"].astype(np.float64)                 # [22, 8, 3]
n_live = int(REF["n_live"])                                       # 4
n_steps = int(REF["n_steps"])                                     # 21
u_distinct = np.asarray(REF["u_distinct"], np.float64)            # [4, 2]
gene_order = [str(x) for x in REF["gene_order"]]

# --- PRIMARY: uniform drive, through the REAL engine.run (the run_spec path) ------------------ #
sim = load(SPEC)
_, out = engine.run(sim, out_path=None, device="cpu", progress=False)
g_eng_u = np.asarray(out["sets"]["cell"]["state"]["gene"], np.float64)     # [22, 8, 3]
assert g_eng_u.shape == y_uniform.shape, (g_eng_u.shape, y_uniform.shape)
du = np.abs(g_eng_u[:, :n_live, :] - y_uniform[:, :n_live, :])
D_max_uniform = float(du.max())
per_frame_u = du.reshape(du.shape[0], -1).max(axis=1)
frame0_ic = float(np.abs(g_eng_u[0, :n_live, :]).max())           # engine's seeded IC (should be 0)

# --- CORROBORATION: distinct per-cell drive, manual engine stepping --------------------------- #
sim_d = load(SPEC)
Hd = build(sim_d, device="cpu")
Hd.emit_order = _resolve_emit(sim_d, Hd)
cell = Hd.level("cell")
a_g, b_g = cell.state_schema["gene"]
a_d, b_d = cell.state_schema["drive"]
st = cell.state.clone()
st[:, a_g:b_g] = 0.0                                              # gene init 0 (same IC as reference)
st[:, a_d:b_d] = 0.0
st[:n_live, a_d:b_d] = torch.as_tensor(u_distinct, dtype=st.dtype)  # per-cell frozen drive
cell.state = st
opspec = next(o for o in sim_d.operators if o.op == "regulate")
op = get_operator("regulate", opspec.impl)({**opspec.params, "_at": "cell"}, "cpu")
rec = np.zeros((n_steps + 1, n_live, b_g - a_g), np.float64)
rec[0] = cell.get("gene")[:n_live].cpu().numpy()
for t in range(1, n_steps + 1):
    Hd.frame = t
    Hd.zero_delta()
    Hd.add_delta("cell", op(Hd, cell.active)["cell"], op.INTEGRAND)
    _integrate(Hd, sim_d.dt)
    rec[t] = cell.get("gene")[:n_live].cpu().numpy()
dd = np.abs(rec - y_distinct[:, :n_live, :])
D_max_distinct = float(dd.max())

THRESHOLD = 2e-4
value = max(D_max_uniform, D_max_distinct)
passed = bool(value <= THRESHOLD)

diff = {
    "diff_metric": "D_inf = max|engine_gene - reference_gene| over 22 frames x 4 live cells x 3 genes",
    "threshold": THRESHOLD,
    "value": value,                                              # primary = uniform; reported as the max of the two
    "D_max_uniform": D_max_uniform,
    "D_max_distinct": D_max_distinct,
    "passed": passed,
    "frame0_engine_ic_maxabs": frame0_ic,
    "n_frames": int(y_uniform.shape[0]),
    "n_live": n_live,
    "n_gene": len(gene_order),
    "gene_order": gene_order,
    "per_frame_max_uniform": per_frame_u.tolist(),
    "engine_gene_last_cell0": g_eng_u[-1, 0].tolist(),
    "ref_gene_last_cell0": y_uniform[-1, 0].tolist(),
    "distinct_engine_last": rec[-1].tolist(),
    "distinct_ref_last": y_distinct[-1, :n_live].tolist(),
    "reference_run": "atlas_jax/_oracle/runs/diff_odecontroller",
    "engine_evidence": "log/atlas_jax/odecontroller/",
}
with open(os.path.join(RUN, "diff.json"), "w") as f:
    json.dump(diff, f, indent=2)

print(json.dumps({k: diff[k] for k in
                  ("threshold", "value", "D_max_uniform", "D_max_distinct", "passed",
                   "frame0_engine_ic_maxabs")}, indent=2))
print("per-frame max (uniform):", [f"{x:.2e}" for x in per_frame_u])
print("wrote", os.path.join(RUN, "diff.json"))
