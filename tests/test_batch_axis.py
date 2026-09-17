"""A batched run of B trials reproduces B separate single runs.

THIS IS THE ONLY THING THAT PROVES THE BATCH AXIS IS INERT. The axis was added by moving every
state indexing from the left (`state[i, a:b]`) to the right (`state[..., a:b]`) and by reading
every element count from the right (`Level.n`), which is a change to some forty call sites across
the engine, the level, the incidence maps and four operators. A site that was missed does not
raise: it indexes trial 0 and broadcasts, so all B trials get trial 0's value and the run looks
fine -- lower loss, even, because the variance across trials is gone. Only a trial-by-trial
comparison against separate runs can see that.

DIFFERENT INPUT PER TRIAL is the point. With the same stimulus in every row a missed `...` would
be invisible, because trial 0's value IS every trial's value. Each row here gets its own retina
drive, so any site still reading axis 0 collapses the spread and the comparison misses by the full
scale of the gaze -- degrees -- not by a rounding amount.

THE REAL GATE IS THE FLOAT64 ONE. In float32 the two paths do not agree bit-for-bit, and should
not be expected to: a batched `design @ beta` and a batched gather dispatch to different BLAS
kernels than their [N, W] counterparts, summing the same terms in a different order. Measured over
120 steps of the eye rig that costs 2.8e-7 of the gaze's own scale, about 2 units in the last
place of float32 (eps = 1.2e-7). In float64 the same comparison closes to 1.8e-15 degrees on a
gaze of 7 degrees -- again ~2 ulp. An error that shrinks with the mantissa is arithmetic; a leaked
batch axis would not move at all. So the float64 case carries a tight absolute bound and the
float32 case a relative one, five orders of magnitude below what any real leak produces.
"""
import os

import pytest
import torch

from plexus import engine
from plexus.schema import load

SPEC = os.path.join(os.path.dirname(__file__), "..", "config", "neural", "ctrnn_eyeG_rig.yaml")
B, T = 8, 120


def _drive(u):
    """A per-frame hook writing `u` into retina.signal: [T] for one trial, [B, T] for a batch."""
    def hook(H, frame):
        lvl = H.level("retina")
        a, b = lvl.state_schema["signal"]
        t = min(frame, u.shape[-1] - 1)
        st = lvl.state.clone()
        st[..., a:b] = u[..., t, None, None]              # [.., 1, 1] -> [.., N, 1]
        lvl.state = st
    return hook


def _gaze(u, batch):
    """The eye's three pose angles every frame: [T, 1, 3] at B = 1, [T, B, 1, 3] batched."""
    sim = load(SPEC)
    sim.n_frames = T
    trace = []
    base = _drive(u)

    def hook(H, frame):
        base(H, frame)
        trace.append(H.level("eye").get("pose").clone())

    engine.quiet(True)
    try:
        engine.run(sim, device="cpu", progress=False, grad=False, on_frame=hook, batch=batch)
    finally:
        engine.quiet(False)
    return torch.stack(trace)


def _compare(dtype, tol, relative):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        torch.manual_seed(0)
        u = (0.35 + 0.30 * torch.randn(B, T)).to(dtype)    # a different drive per trial
        many = _gaze(u, batch=B)
        assert many.shape[1] == B, f"batched run lost its trial axis: {tuple(many.shape)}"

        worst = 0.0
        for i in range(B):
            one = _gaze(u[i], batch=1)
            err = (many[:, i] - one).abs().max().item()
            if relative:
                err /= max(one.abs().max().item(), 1e-12)
            worst = max(worst, err)
            assert err < tol, (f"trial {i} in {dtype}: batched gaze differs by {err:.3e} "
                               f"({'relative' if relative else 'degrees'}) from its own run")

        # AND THE TRIALS MUST ACTUALLY DIFFER. If `expand_batch` shared storage instead of
        # copying, every row would hold the same trajectory and the loop above would pass
        # vacuously -- eight identical runs agree with themselves.
        spread = (many - many.mean(dim=1, keepdim=True)).abs().max().item()
        assert spread > 1e-3, f"all {B} trials are identical (max deviation {spread:.3e} deg)"
        return worst
    finally:
        torch.set_default_dtype(old)


@pytest.mark.skipif(not os.path.exists(SPEC), reason="rig spec absent")
def test_batched_equals_separate_runs_float64():
    """1e-9 degrees of gaze, on a gaze of ~7 degrees. Nothing but the arithmetic is left."""
    _compare(torch.float64, 1e-9, relative=False)


@pytest.mark.skipif(not os.path.exists(SPEC), reason="rig spec absent")
def test_batched_equals_separate_runs_float32():
    """1e-5 of the gaze's own scale -- 84x the measured float32 disagreement, 1e5x below a leak."""
    _compare(torch.float32, 1e-5, relative=True)
