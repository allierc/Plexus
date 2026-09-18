"""`config/neural/zf_eyeG_285.yaml` against the forward loop it was transcribed from.

The reference is `ZebrafishCircuitRNN` in `prototype/dot_tracking/train_zebra_eyeG.py` on
connectome-gnn's feat/oculomotor: the measured 285-cell oculomotor pool, sign-locked to the
connectome, driving the SAME frozen eye the free 64-unit ctRNN rig drives. Three claims are
checked, and each one is a thing a spec can get wrong while still running:

    1. SIGN-LOCK. The message uses |w| times the PRESYNAPTIC cell's Dale sign, eq:sign-lock, so
       a magnitude may move under training and a sign may not. The connectome on disk is already
       100% consistent with the type table, which makes `dale: true` a no-op on the initial
       weights -- so a spec that silently dropped it would agree at frame 0 and diverge only
       after the optimiser first pushed a weight through zero. Checked directly, on the operator.

    2. FOUR MUSCLES AT EXACTLY ZERO. SR, IR, SO and IO are driven by OMN, which is not among
       these 285 cells, so nothing in the pool can reach them. The reference holds them at
       exactly zero and NOT at softplus(0) = 0.693, which would assert a tonic contraction that
       does not exist. In the spec that is two masked `readout`s, and the failure mode it guards
       is a masked readout writing its whole block: the second would erase the first, leaving
       one muscle driven and the run still plausible.

    3. THE ARITHMETIC OF THE WHOLE CHAIN, retina -> AF5 -> pool -> AMN/AIN -> LR/MR -> eye,
       against a numpy transcription of the reference's loop.
"""
import os

import numpy as np
import pytest

from plexus import engine
from plexus.operators.muscle_ops import PAIRS
from plexus.paths import graphs_data_path
from plexus.schema import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "config", "neural", "zf_eyeG_285.yaml")
EDGES = ["zebrafish_om_285_edges.npz", "zf_eyeG_285_win.npz",
         "zf_eyeG_285_lr.npz", "zf_eyeG_285_mr.npz"]
N, T = 285, 240
LR, MR = 0, 2                                   # muscle_ops.MUSCLES = (LR, SR, MR, IR, SO, IO)


def _have_edges():
    d = os.path.join(graphs_data_path(), "neural")
    return all(os.path.exists(os.path.join(d, f)) for f in EDGES)


def _dense(fname, rows, cols):
    z = np.load(os.path.join(graphs_data_path(), "neural", fname))
    M = np.zeros((rows, cols))
    M[z["edge_index"][1], z["edge_index"][0]] = z["weights"]
    return M


pytestmark = pytest.mark.skipif(not _have_edges(),
                                reason="the zebrafish edge sets are not on this host")


def _run():
    engine.quiet(True)
    try:
        sim = load(SPEC)
        sim.n_frames = T
        H, _ = engine.run(sim, device="cpu", progress=False)
    finally:
        engine.quiet(False)
    return sim, H


def test_only_LR_and_MR_are_driven():
    """The other four muscles are at EXACTLY zero, not softplus(0)."""
    _, H = _run()
    m = H.level("muscle").get("drive")[:, 0].detach().numpy()
    silent = [i for i in range(6) if i not in (LR, MR)]
    assert np.all(m[silent] == 0.0), (
        f"SR/IR/SO/IO carry drive {m[silent]}; nothing in the 285-cell pool projects to them, so "
        f"any nonzero value here is a tonic contraction the circuit did not command.")
    assert m[LR] > 0.1 and m[MR] > 0.1, (
        f"LR {m[LR]:.4f} and MR {m[MR]:.4f} must BOTH be driven -- a zero here means the schedule "
        f"ran one `readout` where two are declared, or one erased the other's block.")


def test_the_message_is_sign_locked_to_the_presynaptic_type():
    """|w| times the sender's Dale sign, and the sign survives a weight of the wrong sign."""
    import torch
    sim, H = _run()
    op = dict(zip(H.operator_names, H.operators))["neuron_signal"]
    es = H.level("synapse")
    sign = op._dale_sign(H.level("neuron"), es).squeeze(-1).numpy()

    types = load(SPEC).sets["neuron"]["types"]
    lo, per_cell = 0, np.zeros(N)
    for t in types.values():
        per_cell[lo:lo + int(t["count"])] = 1.0 if t["sign"] == "E" else -1.0
        lo += int(t["count"])
    assert np.array_equal(sign, per_cell[es.pre.numpy()]), \
        "the per-edge sign is not the presynaptic cell's type sign"

    # THE LOCK MUST HOLD AGAINST A WEIGHT OF THE WRONG SIGN, which is the only state in which it
    # does anything: the connectome on disk already agrees with the type table on all 5,013
    # edges, so a spec that dropped `dale:` would pass every check made at the initial weights.
    w = es.get("w").clone()
    with torch.no_grad():
        es.state[:, es.state_schema["w"][0]:es.state_schema["w"][1]] = -w
    w_eff = (es.get("w").abs() * op._dale_sign(H.level("neuron"), es)).squeeze(-1).numpy()
    assert np.array_equal(np.sign(w_eff[w_eff != 0]), sign[w_eff != 0]), \
        "flipping every measured weight flipped the effective sign; the lock is not applied"


def test_the_spec_reproduces_ZebrafishCircuitRNN():
    """The whole chain against a transcription of the reference's forward loop."""
    sim, H = _run()
    ops = {o.op: o.params for o in sim.operators}
    beta = np.asarray(ops["muscle_pose_map"]["beta"], float)
    C, K = (np.asarray(ops["organ_mechanics"][k], float) for k in ("C", "K"))
    dt = float(sim.dt)
    tau = float(H.level("neuron").get("tau")[0, 0])
    # The retina drive is read from the yaml's own seed line rather than from `sim`: `Spec.seed`
    # is `general.seed`, the RNG seed, and the seed OPERATORS live elsewhere -- two different
    # things wearing one word.
    import yaml
    raw = yaml.safe_load(open(SPEC))
    drive_in = float(next(s for s in raw["seed"] if s["block"] == "signal")["lo"])

    S = _dense(EDGES[0], N, N)                        # measured, already Dale-consistent
    Win = _dense(EDGES[1], N, 2)                      # zero outside the 41 AF5 cells
    w_lr = _dense(EDGES[2], 6, N)[LR]                 # AMN columns only
    w_mr = _dense(EDGES[3], 6, N)[MR]                 # AIN columns only
    b_v = H.level("neuron").get("bias")[:, 0].detach().numpy().astype(np.float64)
    b_m = H.level("muscle").get("bias")[:, 0].detach().numpy().astype(np.float64)

    # eq:sign-lock, as a dense matrix: |S_ij| carrying the sign of the PRESYNAPTIC cell j.
    sign_col = np.zeros(N)
    lo = 0
    for t in load(SPEC).sets["neuron"]["types"].values():
        sign_col[lo:lo + int(t["count"])] = 1.0 if t["sign"] == "E" else -1.0
        lo += int(t["count"])
    What = np.abs(S) * sign_col[None, :]

    minv = np.linalg.inv(np.eye(3) + dt * C)
    I = Win @ np.full(2, drive_in)
    v, u, ud = np.zeros(N), np.zeros(3), np.zeros(3)
    # `n_frames + 1` STEPS, NOT `n_frames`. `engine.run` iterates `range(sim.n_frames + 1)`
    # (engine.py:2363) and every tick integrates, so a run of 240 frames advances the state 241
    # times -- `n_frames` names the number of recorded rows AFTER the first, not the number of
    # Euler steps. Getting this wrong costs a whole step of drift, which on a settled run is
    # invisible (the ctRNN rig reaches steady state and one extra step barely moves it) and on a
    # transient is the difference between agreeing to 1e-16 and to 1e-3.
    for _ in range(T + 1):
        r = np.tanh(v)
        m = np.zeros(6)                                # the four unreachable muscles stay at zero
        m[LR] = np.log1p(np.exp(w_lr @ r + b_m[LR]))
        m[MR] = np.log1p(np.exp(w_mr @ r + b_m[MR]))
        cross = np.array([m[i] * m[j] for i, j in PAIRS])
        u_inf = np.concatenate([m, m ** 2, cross]) @ beta
        ud = (ud + dt * (K @ (u_inf - u))) @ minv.T
        u = u + dt * ud
        v = v + dt * ((1.0 / tau) * (-v + What @ r + I) + b_v)

    got = H.level("eye").get("pose")[0].detach().numpy().astype(np.float64)
    err = np.abs(got - u).max()
    assert abs(u[0]) > 0.1, f"the rig barely moves the eye (theta {u[0]:.3f} deg); vacuous"
    assert err < 1e-5, (
        f"the spec departs from ZebrafishCircuitRNN by {err:.3e} deg of gaze over {T} frames "
        f"(spec {np.round(got, 4)}, reference {np.round(u, 4)}).")
