"""The oculomotor plant, against the reference rollout it was transcribed from.

The two operators of `plexus.operators.muscle_ops` reproduce one specific implementation:
`prototype/dot_tracking/train_eyeG.py` in the connectome-gnn repository, whose `EyeG.equilibrium`
is the quadratic static map and whose `rollout` is the damped second-order body, semi-implicit in
the spring and implicit in the damping. A controller trained through THAT eye can only be replayed
through THIS one if the two agree numerically, so the reference is transcribed here rather than
described, and the test is a comparison and not a smoke check.

The coefficients are synthetic. A real fit (`eye_fit_XL.json`, 440 MB of characterisation behind
it) is not in this repository, and the agreement being tested is a property of the DISCRETISATION,
not of any particular C, K and beta -- so the test draws its own, with the same shapes, the same
symmetry and the same order of magnitude as the measured ones, and additionally checks the
measured eye's own stability margin as a stated number rather than a remembered one.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import plexus.operators                                        # noqa: F401  self-registers
from plexus.models.registry import get_operator
from plexus.operators.muscle_ops import MUSCLES, N_MUSCLE, N_QUAD, PAIRS

DT = 1.0 / 60.0                    # the rate the reference controller is trained at
T = 240                            # 4 s, long enough for the plant's ~1 Hz ringing to show


def _synthetic_fit(tmp_path, seed=0):
    """A well-posed eye: symmetric positive-definite C and K of the measured magnitude.

    K's eigenvalues land near the measured 33-53 per second squared and C's near 4.6-5.7 per
    second, so the run exercises the same underdamped regime (damping ratio ~0.4) the real eye
    sits in rather than an overdamped one where every discretisation agrees trivially.
    """
    rng = np.random.RandomState(seed)
    def spd(scale):
        a = rng.randn(3, 3) * 0.08
        return (np.eye(3) * scale + (a + a.T) / 2 * scale).tolist()
    spec = {"beta": (rng.randn(N_QUAD, 3) * 3.0).tolist(),
            "C": spd(5.1), "K": spd(42.0)}
    p = tmp_path / "eye_fit_synth.json"
    p.write_text(json.dumps(spec))
    return str(p), spec


def _reference(beta, C, K, m, dt):
    """`train_eyeG.EyeG.equilibrium` then `train_eyeG.rollout`, in float64 numpy.

    u_inf = g(m), the 27-term quadratic; then u_ddot + C u_dot + K u = K u_inf stepped as
    `v = (v + dt K (u_inf - u)) (I + dt C)^-1` followed by `u += dt v` -- the spring taken at the
    current position, the damping taken implicitly.
    """
    beta, C, K = np.asarray(beta), np.asarray(C), np.asarray(K)
    cross = np.stack([m[:, i] * m[:, j] for i, j in PAIRS], -1)
    u_inf = np.concatenate([m, m ** 2, cross], -1) @ beta
    minv = np.linalg.inv(np.eye(3) + dt * C)
    u, v, out = np.zeros(3), np.zeros(3), []
    for t in range(u_inf.shape[0]):
        v = (v + dt * ((u_inf[t] - u) @ K.T)) @ minv.T
        u = u + dt * v
        out.append(u.copy())
    return np.stack(out)


class _Lvl:
    def __init__(self, n, schema, width, name):
        self.n, self.name = n, name
        self.state = torch.zeros(n, width)
        self.state_schema, self.occ = schema, torch.ones(n)

    def get(self, block):
        a, z = self.state_schema[block]
        return self.state[:, a:z]


class _H:
    """The smallest Hierarchy these two operators need: one eye, its six muscles, a dt."""
    def __init__(self, eye, muscle, dt):
        self.levels = {"eye": eye, "muscle": muscle}
        self.config = type("C", (), {"dt": dt})()

    def level(self, n):
        return self.levels[n]

    def ancestors(self, n):
        return ["eye"] if n == "muscle" else []

    def lift_index(self, child, parent):
        return torch.arange(self.levels[child].n) // N_MUSCLE


def _run(fit, m, dt, implementation=None):
    """Step the two operators the way the engine steps a second-order set."""
    eye = _Lvl(1, {"gaze": (0, 3), "gaze_rate": (3, 6), "gaze_inf": (6, 9)}, 9, "eye")
    mus = _Lvl(N_MUSCLE, {"drive": (0, 1)}, 1, "muscle")
    H = _H(eye, mus, dt)
    gmap = get_operator("muscle_gaze_map")({"fit": fit, "_at": "muscle", "parent": "eye"})
    mech = get_operator("eye_mechanics", implementation=implementation)({"fit": fit, "_at": "eye"})
    out = []
    for t in range(m.shape[0]):
        mus.state[:, 0:1] = torch.as_tensor(m[t], dtype=torch.float32)[:, None]
        gmap.forward(H)                                     # -> gaze_inf on the parent
        a = mech.forward(H)["eye"]                          # the emitted acceleration
        eye.state[:, 3:6] = eye.state[:, 3:6] + dt * a      # v += dt a
        eye.state[:, 0:3] = eye.state[:, 0:3] + dt * eye.state[:, 3:6]   # u += dt v
        out.append(eye.state[0, 0:3].clone().numpy())
    return np.stack(out)


def _drives(seed=0):
    """Six sinusoids at independent phases, in [0, 1] -- every muscle on, every cross term live."""
    rng = np.random.RandomState(seed)
    return np.clip(0.5 + 0.4 * np.sin(np.linspace(0, 6 * np.pi, T)[:, None] + rng.rand(6) * 6.28),
                   0, 1)


def test_the_plant_reproduces_the_reference_rollout(tmp_path):
    """THE PARITY CLAIM. Agreement to float32 rounding over a full trial, not merely 'close'."""
    fit, spec = _synthetic_fit(tmp_path)
    m = _drives()
    ref = _reference(spec["beta"], spec["C"], spec["K"], m, DT)
    got = _run(fit, m, DT)
    err = np.abs(got - ref).max()
    assert np.ptp(ref, axis=0).min() > 1.0, (
        f"the test drive barely moves the eye (per-axis range {np.ptp(ref, axis=0)}), so agreement would "
        f"be vacuous")
    assert err < 1e-3, (
        f"the operators depart from train_eyeG.rollout by {err:.3e} deg over {T} frames "
        f"({T * DT:.1f} s). Anything above float32 rounding means a controller trained through "
        f"the reference eye cannot be replayed through this one.")


def test_the_eye_is_underdamped_so_the_discretisation_is_actually_tested(tmp_path):
    """The regime matters: every scheme agrees on an overdamped plant."""
    _, spec = _synthetic_fit(tmp_path)
    wn = np.sqrt(np.linalg.eigvalsh(np.asarray(spec["K"])))
    zeta = np.linalg.eigvalsh(np.asarray(spec["C"])) / (2 * wn)
    assert zeta.max() < 1.0, f"damping ratios {zeta} -- overdamped, so this tests nothing"
    assert 0.5 < wn.min() / (2 * np.pi) < 3.0, (
        f"natural frequencies {wn / (2 * np.pi)} Hz are outside the ~1 Hz band the measured eye "
        f"sits in")


def test_explicit_damping_is_a_separate_implementation_and_differs(tmp_path):
    """The two bodies are the same contract and genuinely different numbers."""
    fit, spec = _synthetic_fit(tmp_path)
    m = _drives()
    ref = _reference(spec["beta"], spec["C"], spec["K"], m, DT)
    d_impl = np.abs(_run(fit, m, DT, "explicit") - ref).max()
    d_dflt = np.abs(_run(fit, m, DT) - ref).max()
    assert d_dflt < 1e-3 < d_impl, (
        f"the default is meant to match the reference ({d_dflt:.3e} deg) and `explicit` is meant "
        f"to be a visibly different discretisation ({d_impl:.3e} deg); if they agree the "
        f"implementation axis is recording a distinction that does not exist")


def test_the_measured_eye_is_far_inside_the_explicit_stability_limit():
    """The number the docstrings quote, as an assertion rather than a memory.

    Explicit damping is stable only while dt * max(eig C) < 2. The measured fit's C has a largest
    eigenvalue of 5.741 per second, so at dt = 1/60 s the product is 0.096 -- a factor of 21
    inside the limit. This is why the implicit default is chosen for PARITY with the reference and
    not, on this eye, to prevent a divergence.
    """
    max_eig_c = 5.741                                  # per second, from the measured fit
    assert DT * max_eig_c == pytest.approx(0.0957, abs=1e-3)
    assert DT * max_eig_c < 2.0 / 20, "the quoted factor-of-20 margin no longer holds"


def test_the_muscle_count_is_checked(tmp_path):
    """A fit is a map from EXACTLY six drives, in a stated order."""
    fit, _ = _synthetic_fit(tmp_path)
    eye = _Lvl(1, {"gaze": (0, 3), "gaze_rate": (3, 6), "gaze_inf": (6, 9)}, 9, "eye")
    mus = _Lvl(N_MUSCLE - 1, {"drive": (0, 1)}, 1, "muscle")      # five muscles
    H = _H(eye, mus, DT)
    gmap = get_operator("muscle_gaze_map")({"fit": fit, "_at": "muscle", "parent": "eye"})
    with pytest.raises(ValueError, match="muscles for"):
        gmap.forward(H)
    assert len(MUSCLES) == N_MUSCLE == 6
    assert N_QUAD == 27, "6 linear + 6 square + 15 cross"


def test_a_missing_or_malformed_fit_is_refused(tmp_path):
    """Both operators read the same file, so a wrong one must not be read as a plausible eye."""
    with pytest.raises(FileNotFoundError, match="eye fit"):
        get_operator("eye_mechanics")({"fit": str(tmp_path / "nope.json"), "_at": "eye"})
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"beta": np.zeros((N_QUAD, 3)).tolist(),
                               "C": np.eye(3).tolist(), "K": np.eye(2).tolist()}))
    with pytest.raises(ValueError, match="expected"):
        get_operator("eye_mechanics")({"fit": str(bad), "_at": "eye"})


# --------------------------------------------------------------------------- #
#  the whole rig, as a spec
# --------------------------------------------------------------------------- #
def test_the_ctrnn_rig_spec_reproduces_CTRNNEyeG():
    """config/neural/ctrnn_eyeG_rig.yaml against the forward loop it was transcribed from.

    The spec is the FIRST rig of the oculomotor work -- `CTRNNEyeG` in
    `prototype/dot_tracking/train_eyeG.py`: a free 64-unit continuous-time circuit between a
    two-channel velocity input and the six muscle drives of the fitted eye. It is the whole
    chain in one file, W_in -> circuit -> W_out -> organ, and the point of the test is that the
    chain is ARITHMETICALLY the reference's, not merely shaped like it.

    Skipped where the edge-set npz are absent, since they live in graphs_data rather than in the
    repository; nothing else here needs them.
    """
    import os
    import numpy as np
    from plexus.schema import load
    from plexus import engine
    from plexus.paths import graphs_data_path

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec_path = os.path.join(root, "config", "neural", "ctrnn_eyeG_rig.yaml")
    d = os.path.join(graphs_data_path(), "neural")
    files = [f"ctrnn_eyeG_rig_{k}.npz" for k in ("win", "rec", "wout")]
    if not all(os.path.exists(os.path.join(d, f)) for f in files):
        pytest.skip("the rig's edge sets are not on this host")

    def mat(f, r, c):
        z = np.load(os.path.join(d, f))
        M = np.zeros((r, c))
        M[z["edge_index"][1], z["edge_index"][0]] = z["weights"]
        return M

    Win, W, Wout = mat(files[0], 64, 2), mat(files[1], 64, 64), mat(files[2], 6, 64)
    sim = load(spec_path)
    # The coefficients are IN the spec, so the reference reads them from there rather than from
    # a characterisation file -- which is the property being relied on, not a convenience.
    ops = {o.op: o.params for o in sim.operators}
    beta = np.asarray(ops["muscle_gaze_map"]["beta"], float)
    C, K = (np.asarray(ops["eye_mechanics"][k], float) for k in ("C", "K"))
    dt, tau, T = float(sim.dt), 0.5, int(sim.n_frames)
    drive_in = float(sim.seed[0].params["lo"]) if getattr(sim, "seed", None) else 0.35

    minv = np.linalg.inv(np.eye(3) + dt * C)
    I = Win @ np.full(2, drive_in)
    v, u, ud = np.zeros(64), np.zeros(3), np.zeros(3)
    for _ in range(T):
        r = np.tanh(v)                                   # the RATES are what W_out sees
        m = np.log1p(np.exp(Wout @ r))                   # softplus: a muscle pulls or does nothing
        cross = np.array([m[i] * m[j] for i, j in PAIRS])
        u_inf = np.concatenate([m, m ** 2, cross]) @ beta
        ud = (ud + dt * (K @ (u_inf - u))) @ minv.T
        u = u + dt * ud
        v = v + dt * ((1.0 / tau) * (-v + W @ r + I))     # a = g = 1/tau, the spec's p row

    H, _ = engine.run(sim, device="cpu", progress=False)
    got = H.level("eye").get("gaze")[0].detach().numpy().astype(np.float64)
    err = np.abs(got - u).max()
    assert abs(u[0]) > 1.0, f"the rig barely moves the eye (theta {u[0]:.3f} deg); vacuous"
    assert err < 1e-3, (
        f"the spec departs from CTRNNEyeG by {err:.3e} deg of gaze over {T} frames. The chain "
        f"is W_in -> circuit -> W_out -> organ; a mismatch here means one link is not the "
        f"reference's arithmetic.")
