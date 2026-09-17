"""The gradient of a batched loss equals the gradient accumulated over separate trials.

THE FORWARD GATE IS NOT ENOUGH. `test_batch_axis` proves B trials run side by side without
contaminating each other, which is a statement about the state. Training is a statement about the
TAPE: the fitted weights are written into `[N, W]` state and then `repeat`ed to `[B, N, W]`, so
every one of the B trials reads the SAME leaf and the backward pass has to sum B contributions
into it. If `expand_batch` had broken that -- a `.detach()`, a re-created leaf, a `.data` write --
the forward would still be right and the fit would silently train on one trial's gradient.

WHAT IS COMPARED. `mean over frames and trials of (y - target)^2` differentiated in one batched
rollout, against the same loss assembled as `(1/B) * sum_i` of B single-trial rollouts, each
backward'd into the same leaf. These are the same scalar function of the same weights, so their
gradients agree up to the arithmetic -- and the arithmetic is the only thing left, which is why
this runs in float64. The relative bound is on the gradient's own norm, because the three fitted
blocks differ in scale by an order of magnitude and an absolute bound would be a bound on the
largest one only.
"""
import os

import pytest
import torch

from plexus import engine
from plexus.tasks.spec_trainer import _mse, build, rollout

RUN = os.path.join(os.path.dirname(__file__), "..", "config", "run", "eye_rig_fit.yaml")
B, T = 4, 90
IO = dict(drive_set="retina", drive_block="signal", read_set="eye", read_block="pose")


def _run(path, device="cpu"):
    from plexus.tasks.spec_trainer import load_run
    run = load_run(path)
    sim = build(run, device)
    sim.n_frames = T
    return sim


@pytest.mark.skipif(not os.path.exists(RUN), reason="eye rig run spec absent")
def test_batched_gradient_equals_accumulated_gradient():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    engine.quiet(True)
    try:
        torch.manual_seed(0)
        u = (0.35 + 0.30 * torch.randn(B, T, 1)).double()   # a different drive per trial
        y_t = torch.randn(B, T, 1).double() * 5.0           # a stand-in target, in degrees

        # --- one batched rollout ---------------------------------------------------- #
        sim = _run(RUN)
        _, y = rollout(sim, u, IO["drive_set"], IO["drive_block"], IO["read_set"],
                       IO["read_block"], grad=True)
        _mse(y, y_t, 0).backward()
        g_batch = {k: p.grad.clone() for k, p in sim.fitted.items()}
        assert g_batch, "nothing was fitted, so there is no gradient to compare"

        # --- B single rollouts, accumulated into the same leaves -------------------- #
        # `sim.fitted` is created by the first run, so it is INSTALLED here rather than copied
        # into: the same route `_restore` uses to load a checkpoint. Both specs would start from
        # the same edges file anyway; installing makes "the same point in weight space" a fact of
        # the test rather than of the spec.
        sim2 = _run(RUN)
        sim2.fitted = {k: torch.nn.Parameter(p.detach().clone()) for k, p in sim.fitted.items()}
        for i in range(B):
            _, yi = rollout(sim2, u[i], IO["drive_set"], IO["drive_block"], IO["read_set"],
                            IO["read_block"], grad=True)
            (_mse(yi, y_t[i], 0) / B).backward()
        g_acc = {k: p.grad.clone() for k, p in sim2.fitted.items()}

        for k in g_batch:
            num = (g_batch[k] - g_acc[k]).norm().item()
            den = max(g_acc[k].norm().item(), 1e-30)
            assert num / den < 1e-9, (
                f"{k}: batched gradient differs from the accumulated one by {num / den:.3e} "
                f"relative (norms {g_batch[k].norm():.4e} vs {g_acc[k].norm():.4e})")
            # AND THE GRADIENT MUST NOT BE ZERO, or the comparison above is two zeros agreeing.
            assert g_acc[k].norm().item() > 0, f"{k} received no gradient at all"
    finally:
        engine.quiet(False)
        torch.set_default_dtype(old)
