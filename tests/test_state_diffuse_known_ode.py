"""state_diffuse[model: known_ode] (exp17): connectome-gnn's known ODE on the multi-level mesh, current and conductance."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

CLS = get_contract("state_diffuse").implementations["known_ode"]


def setup(tmp, synapse="current", n=400):
    g = np.random.default_rng(0)
    P = (g.uniform(0, 100, (n, 3)) * [1.0, 1.6, 0.5]).astype(np.float32)
    f = os.path.join(tmp, "pos.npz")
    np.savez(f, pos_um=P)
    o = CLS({"_at": "neuron", "block": "dff", "positions": "xyz", "mesh_spacing": 16.0, "mesh_levels": 3, "inputs": 1,
             "substeps": 4, "positions_file": f, "synapse": synapse, "forcing": "stimulus.u", "forcing_dim": 22})
    nb = {"tau": torch.full((n, 1), -2.97), "rest": torch.zeros(n, 1), "gain": torch.full((n, 1), 0.3),
          "input": 0.1 * torch.randn(n, 22)}
    return o, torch.as_tensor(P), nb


def test_step_and_gradients_current_and_conductance(tmp_path):
    for syn in ("current", "conductance"):
        o, P, nb = setup(str(tmp_path), syn)
        names = ["W", "mesh_rate", "mesh_rest"] + (["mesh_reversal"] if syn == "conductance" else [])
        for k in names:
            setattr(o, k, getattr(o, k).clone().requires_grad_(True))
        for k in nb:
            nb[k].requires_grad_(True)
        d = o.step(torch.rand(len(P), 1), P, None, torch.randn(22, 1), nb=nb)
        assert d.shape == (len(P), 1) and torch.isfinite(d).all()
        d.pow(2).sum().backward()
        for k in names:
            assert float(getattr(o, k).grad.norm()) > 0, (syn, k)
        for k in nb:
            assert float(nb[k].grad.norm()) > 0, (syn, k)


def test_a_history_is_refused(tmp_path):
    g = np.random.default_rng(0)
    f = os.path.join(str(tmp_path), "pos.npz")
    np.savez(f, pos_um=g.uniform(0, 100, (200, 3)).astype(np.float32))
    try:
        CLS({"_at": "neuron", "block": "dff", "positions": "xyz", "mesh_spacing": 16.0, "inputs": 6, "positions_file": f})
    except ValueError:
        return
    raise AssertionError("a known ODE accepted a history")
