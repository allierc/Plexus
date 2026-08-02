"""Differential test for regulate:neural_ode -- the Plexus torch operator vs the JAX reference,
on the IDENTICAL operator (same MLP weights) and the IDENTICAL initial condition.

NeuralODE's behaviour IS its MLP vector field, and an MLP cannot cross the JAX/torch boundary
through a YAML spec -- so the oracle (`_oracle/scripts/neural_ode.py`) built the MLP, integrated
the reference controller over one macro-step on a fixed per-cell IC, and exported the exact
per-layer weights + the reference endpoint. Here we load those weights VERBATIM into the torch
operator, drive it from the same g0/u/dt THROUGH THE REAL ENGINE `_integrate` (g += dt*delta),
and measure how far the operator's macro-step endpoint lands from the reference's.

Pre-registered metric (see atlas_record.yaml, evidence.diff_metric):
    D_max = max over cells, evolving components, dt in {0.5,1,2}, and both circuit shapes, of
            | y_plexus(dt) - y_ref(dt) |,   threshold 3e-3 (~3x the reference's measured 9.9e-4
            truncation-from-truth at dt=2; a tighter 1e-3 could false-fail on integrator jitter alone).
Reported alongside:
  * a NET-EQUALITY check (torch MLP vs the exported JAX MLP on a probe) -- proves the field is
    identical, so any endpoint gap is the INTEGRATOR, not the network;
  * the ground-truth-anchored errors |y_plexus - y_true| and |y_ref - y_true| -- the reference's
    own rtol=1e-4 endpoint is ~1e-3 from truth at dt=2 (a property of the loose tolerance on a
    single big step, not a defect), so this separates operator error from controller-path error;
  * a NEGATIVE CONTROL (endpoint reconstructed as g0 + delta, i.e. the /dt mean-rate conversion
    dropped) at dt=2, to show the threshold discriminates a real wiring bug.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))

import torch
import torch.nn as nn

import plexus.operators  # noqa: F401
import plexus.operators.candidates.jax_morph_odecontroller  # noqa: F401  keep connectionist the default
import plexus.operators.candidates.jax_morph_neural_ode  # noqa: F401  self-registers regulate:neural_ode
from plexus.models.registry import get_operator
from plexus.models.base import Hierarchy, Level
from plexus.models.state import (
    Block, StateSchema, NONE, FIRST_ORDER, SECOND_ORDER_COORDINATE, SECOND_ORDER_RATE,
    BOUNDARY_WORLD, BOUNDARY_FREE,
)
from plexus.engine import _integrate

ORACLE = os.path.join(HERE, "_oracle", "runs", "diff_neural_ode")
OUTDIR = os.path.join(PLEXUS, "log", "atlas_jax", "neural_ode")
DTYPE = torch.float64
DTS = [0.5, 1.0, 2.0]
CONFIGS = [("A", 0, 3), ("B", 2, 2)]           # (tag, hidden_size, n_out)  -- must match the oracle


class _Cfg:
    def __init__(self, dt):
        self.dt = dt


def build_net(ref, tag):
    """Rebuild the exported MLP in torch: Linear/ReLU/.../Linear (eqx.nn.MLP topology, identity
    final). eqx and torch both store Linear weight as (out, in), so weights copy directly."""
    i = 0
    weights = []
    while f"{tag}__W{i}" in ref:
        weights.append((ref[f"{tag}__W{i}"], ref[f"{tag}__b{i}"]))
        i += 1
    layers = []
    for j, (w, b) in enumerate(weights):
        lin = nn.Linear(w.shape[1], w.shape[0]).to(DTYPE)
        with torch.no_grad():
            lin.weight.copy_(torch.as_tensor(w, dtype=DTYPE))
            lin.bias.copy_(torch.as_tensor(b, dtype=DTYPE))
        layers.append(lin)
        if j != len(weights) - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def make_H(N, n_in, n_gene, dt):
    """A one-cell-set Hierarchy with a spatial coordinate (pos/vel) plus the driver block `u`
    and the evolving first-order `gene` block -- the same shape the operator sees in the engine."""
    schema = StateSchema([
        Block("pos", 2, role="coordinate", integration=SECOND_ORDER_COORDINATE, boundary=BOUNDARY_WORLD),
        Block("vel", 2, role="rate", integration=SECOND_ORDER_RATE, record=False),
        Block("u", n_in, integration=NONE, boundary=BOUNDARY_FREE),
        Block("gene", n_gene, integration=FIRST_ORDER, boundary=BOUNDARY_FREE),
    ])
    state = torch.zeros(N, schema.dim, dtype=DTYPE)
    cell = Level("cell", state=state, state_schema=schema)
    H = Hierarchy()
    H.add_level(cell)
    H.config = _Cfg(dt)
    H.world_size = torch.tensor([1e9, 1e9], dtype=DTYPE)   # free gene block; box irrelevant, pos unused
    H.dim = 2
    H.emit_order = {}                                      # no coordinate operator on cell here
    return H, cell, schema


def set_block(cell, schema, name, value):
    a, b = schema[name]
    cell.state[:, a:b] = torch.as_tensor(value, dtype=DTYPE)


def get_block(cell, schema, name):
    a, b = schema[name]
    return cell.state[:, a:b].clone()


def endpoint_through_engine(op, H, cell, schema, dt):
    """Run the operator and integrate its delta through the REAL engine `_integrate`
    (gene += dt*delta), returning the operator's macro-step endpoint y_plexus(dt)."""
    H.zero_delta()
    deltas = op.forward(H)
    for lvl, d in deltas.items():
        H.add_delta(lvl, d, op.INTEGRAND)                 # engine main-loop routing (INTEGRAND='gene')
    _integrate(H, dt)
    return get_block(cell, schema, "gene").numpy()


def main():
    torch.set_grad_enabled(False)   # a forward differential check; the operator is DIFFERENTIABLE=True
    ref = np.load(os.path.join(ORACLE, "reference.npz"))
    N = 16
    n_in = 2
    op_cls = get_operator("regulate", "neural_ode")

    rows = []
    net_eq_max = 0.0
    d_ref_max = 0.0
    d_true_max = 0.0
    ref_true_max = 0.0
    neg_control = {}

    for tag, hidden, n_out in CONFIGS:
        n_gene = hidden + n_out
        net = build_net(ref, tag)

        # NET EQUALITY: torch MLP must reproduce the exported JAX MLP exactly (identical field).
        with torch.no_grad():
            pin = torch.as_tensor(ref[f"{tag}__probe_in"], dtype=DTYPE)
            pout = net(pin).numpy()
        net_eq = float(np.abs(pout - ref[f"{tag}__probe_out"]).max())
        net_eq_max = max(net_eq_max, net_eq)

        g0 = ref[f"{tag}__g0"]
        u0 = ref[f"{tag}__u0"]

        for dt in DTS:
            H, cell, schema = make_H(N, n_in, n_gene, dt)
            set_block(cell, schema, "gene", g0)
            set_block(cell, schema, "u", u0)
            # a fresh operator per (config) -- reuse across dt is fine, but a fresh net-bound op
            # per config keeps the injected net unambiguous.
            op = op_cls({"_at": "cell", "state": "gene", "inputs": "u",
                         "hidden_size": hidden, "net": net}, device="cpu")
            y_plexus = endpoint_through_engine(op, H, cell, schema, dt)

            y_ref = ref[f"{tag}__y_ref_dt{dt}"]
            y_true = ref[f"{tag}__y_true_dt{dt}"]
            d_ref = float(np.abs(y_plexus - y_ref).max())
            d_true = float(np.abs(y_plexus - y_true).max())
            r_true = float(np.abs(y_ref - y_true).max())
            d_ref_max = max(d_ref_max, d_ref)
            d_true_max = max(d_true_max, d_true)
            ref_true_max = max(ref_true_max, r_true)
            rows.append({"config": tag, "dt": dt, "n_gene": n_gene,
                         "plexus_vs_reference": d_ref, "plexus_vs_truth": d_true,
                         "reference_vs_truth": r_true})

            # NEGATIVE CONTROL at dt=2: drop the /dt mean-rate conversion (endpoint = g0 + delta,
            # not g0 + dt*delta). A correct operator's delta is (y-g0)/dt, so g0+delta mis-scales
            # the increment by 1/dt -- the exact bug the /dt convention exists to avoid.
            if dt == 2.0:
                H2, cell2, schema2 = make_H(N, n_in, n_gene, dt)
                set_block(cell2, schema2, "gene", g0)
                set_block(cell2, schema2, "u", u0)
                op2 = op_cls({"_at": "cell", "state": "gene", "inputs": "u",
                              "hidden_size": hidden, "net": net}, device="cpu")
                delta = op2.forward(H2)["cell"].numpy()
                y_nodt = g0 + delta                       # WRONG reconstruction (no dt)
                neg_control[tag] = float(np.abs(y_nodt - y_ref).max())

    threshold = 3.0e-3   # ~3x the reference's measured 9.9e-4 truncation-from-truth (dt=2); see record
    passed = bool(d_ref_max <= threshold)
    result = {
        "metric": "D_max = max |y_plexus(dt) - y_ref(dt)| over cells, components, dt, configs",
        "threshold": threshold,
        "D_max_plexus_vs_reference": d_ref_max,
        "passed": passed,
        "net_equality_max": net_eq_max,
        "plexus_vs_truth_max": d_true_max,
        "reference_vs_truth_max": ref_true_max,
        "negative_control_no_dt_at_dt2_plexus_vs_reference": neg_control,
        "rows": rows,
    }
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "diff.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nD_max (plexus vs reference) = {d_ref_max:.3e}   threshold {threshold:.0e}   "
          f"-> {'PASS' if passed else 'FAIL'}")
    print(f"net-equality (torch MLP vs JAX MLP)  = {net_eq_max:.3e}  (should be ~1e-12: identical field)")
    print(f"plexus vs truth = {d_true_max:.3e}   reference vs truth = {ref_true_max:.3e}")
    print(f"negative control (dropped /dt) at dt=2 = {neg_control}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
