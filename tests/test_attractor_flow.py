"""Promotion tests for the `attractor_flow` operator (strange-attractor ODE flows dx/dt = f(x)).

Covers the paper's promotion requirements (Section "From prototypes to the Plexus core"):
an analytic correctness check of the vector field, schema validation of the live config specs,
a minimal live engine run (finite + chaos actually spreads the seed), and the operator contract
(occupancy respected, |v| clamp, 3D-only via SUPPORTED_DIMS).
"""
import glob
import os
import tempfile

import numpy as np
import torch

import plexus.operators  # noqa: F401  self-registers attractor_flow
import plexus.schema as S
from plexus.schema import Spec, OpSpec, Selector
from plexus.engine import build, run as engine_run
from plexus.models.registry import get_operator
from plexus.operators.attractor_flow import attractor_velocity, ATTRACTOR_SYSTEMS

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "..", "config", "attractors")


def test_vector_field_matches_closed_form():
    """f(x) reproduces the published equations exactly (Lorenz + Sprott B at a test point)."""
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    # Lorenz sigma=10, rho=28, beta=8/3 at (1,2,3): [10(2-1), 1(28-3)-2, 1*2-8/3*3]
    lz = attractor_velocity("lorenz", pos)
    assert torch.allclose(lz, torch.tensor([[10.0, 23.0, 2.0 - 8.0]]), atol=1e-5)
    # Sprott B (a=1): [y z, x-y, 1-x y] = [6, -1, -1]
    sb = attractor_velocity("sprott_b", pos)
    assert torch.allclose(sb, torch.tensor([[6.0, -1.0, -1.0]]), atol=1e-5)


def test_all_systems_registered_and_finite():
    """Every declared system evaluates to a finite [N,3] velocity."""
    assert len(ATTRACTOR_SYSTEMS) == 10
    pos = torch.randn(64, 3)
    for sysname in ATTRACTOR_SYSTEMS:
        v = attractor_velocity(sysname, pos)
        assert v.shape == (64, 3) and torch.isfinite(v).all(), sysname


def test_config_specs_validate():
    """Every promoted config/attractors/*.yaml passes schema validation and uses the operator."""
    specs = sorted(glob.glob(os.path.join(CONFIG, "*.yaml")))
    assert len(specs) == 10
    for p in specs:
        sim = S.load(p)
        assert sim.dim == 3
        assert [o.op for o in sim.operators] == ["attractor_flow"]


def _tiny_lorenz(n=400, n_frames=2500):
    sets = {"cloud": {"n": n, "start": [0.6, 0.6, 14.6, 1.4, 1.4, 15.4]}}
    ops = [OpSpec(op="attractor_flow", on=Selector("cloud"), to=None, frm=None,
                  params={"system": "lorenz"})]
    return Spec(name="lorenz_t", seed=0, n_frames=n_frames, dt=0.004, sets=sets, fields={},
                operators=ops, schedule=["attractor_flow"], boundary="free", dim=3,
                world=50.0, world_size=[50.0, 50.0, 50.0])


def test_engine_runs_and_chaos_spreads():
    """A minimal live run: the engine integrates the flow, stays finite, and the chaos stretches
    the tiny seed cube onto the (much larger) attractor."""
    sim = _tiny_lorenz()
    _, out = engine_run(sim, device="cpu", progress=False)
    pos = out["sets"]["cloud"]["pos"]                       # [T, N, 3]
    assert np.isfinite(pos).all()
    r0 = np.linalg.norm(pos[0] - pos[0].mean(0), axis=1).mean()
    rT = np.linalg.norm(pos[-1] - pos[-1].mean(0), axis=1).mean()
    assert rT > 5.0 * r0                                    # sensitive dependence: the seed fans out
    assert np.ptp(pos[-1, :, 2]) > 10.0                    # spans the butterfly in z


def test_occupancy_and_clamp():
    """Dormant nodes get zero velocity; `clamp` caps |v| without touching the direction."""
    sim = _tiny_lorenz(n=8, n_frames=1)
    H = build(sim, device="cpu")
    lvl = H.level("cloud")
    lvl.occ[0] = 0.0                                        # retire node 0
    op = get_operator("attractor_flow")({"system": "lorenz", "_at": "cloud", "clamp": 1.0}, "cpu")
    vel = op(H, None)["cloud"]
    assert torch.allclose(vel[0], torch.zeros(3))          # dormant -> no motion
    live = vel[1:]
    assert (live.norm(dim=-1) <= 1.0 + 1e-5).all()         # |v| clamped to 1.0


def test_2d_spec_is_rejected():
    """SUPPORTED_DIMS=[3]: a dim=2 spec must fail schema validation, not at runtime."""
    raw = ("general: {name: lz2d, seed: 0, n_frames: 1, dt: 0.01, boundary: free, dim: 2, world: [1.0, 1.0]}\n"
           "sets: {cloud: {n: 4}}\n"
           "fields: {}\n"
           "operators: [{op: attractor_flow, at: cloud, system: lorenz}]\n"
           "schedule: [attractor_flow]\n")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(raw); path = fh.name
    try:
        raised = False
        try:
            S.load(path)
        except ValueError as e:
            raised = True
            assert "dim" in str(e).lower()
        assert raised, "dim=2 attractor_flow spec should be rejected by the schema"
    finally:
        os.unlink(path)
