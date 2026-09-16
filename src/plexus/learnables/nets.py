"""The learnable families, in the two shapes an operator can have.

    mlp      intrinsic | relational   the baseline that has to be beaten
    siren    intrinsic | relational   smooth laws of few inputs, high frequency
    table    intrinsic | relational   a value per TYPE, no smoothness assumed at all

Each is registered twice where it makes sense -- `mlp` and `mlp_edge` -- because the shape is
part of what may stand in for what, and a family that silently switched shape depending on where
it was used would defeat the check it is subject to.

WHY THESE THREE FIRST. They fail differently, which is the only useful property of a starting set.
`mlp` is smooth and low-frequency-biased, so it under-fits a sharp transfer function and says so
by being beaten. `siren` has a sinusoidal first layer and represents high frequency cheaply, which
is what a saturating synaptic nonlinearity needs. `table` assumes nothing -- one free value per
cell type -- so it is the upper bound on what any per-type law can achieve, and a smooth net that
matches it has lost nothing by being smooth. `ngp` and `gnn` are the next two and need no new
interface.

WHAT A LEARNABLE MUST DECLARE, or it cannot be registered:

    SHAPE       intrinsic (reads its own set's blocks) or relational (gathers along an edge set)
    READS       the blocks it consumes -- may narrow the operator's, never widen
    EMIT        "inherit" unless it genuinely emits something else, which is almost always wrong
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from plexus.learnables import INTRINSIC, RELATIONAL, register_learnable


class Learnable(nn.Module):
    """Base: a net plus the contract it claims, and the fit that starts it at a known law.

    `EMIT`/`INTEGRAND`/`WRITES` default to "inherit", meaning "whatever the operator I replace
    declares" -- the common and almost always correct case, since a substitution that changes
    what the engine integrates is a different model rather than a fitted one.
    """

    SHAPE = None
    EMIT = "inherit"
    INTEGRAND = "inherit"
    READS: list = []
    WRITES: list = []

    def __init__(self, n_in, n_out, **params):
        super().__init__()
        self.n_in, self.n_out = int(n_in), int(n_out)

    def fit_to(self, fn, n_samples=4096, iters=800, lo=-3.0, hi=3.0, lr=1e-3, seed=0,
               verbose=False):
        """Pre-fit to the closed-form law `fn` the learnable replaces, so training STARTS THERE.

        This is `init_from: operator`, and it is the difference between a residual that means
        something and one that does not. Begin at the analytic law and whatever the net does
        afterwards is what the data asked for beyond it; begin at noise and the first thousand
        steps are spent rediscovering the law, with nothing separating "the model was wrong" from
        "the fit had not converged".

        Returns the final sampling loss, so a substitution that could NOT be initialised to its
        operator is visible rather than silently starting somewhere arbitrary.
        """
        g = torch.Generator().manual_seed(int(seed))
        x = torch.rand(n_samples, self.n_in, generator=g) * (hi - lo) + lo
        with torch.no_grad():
            y = fn(x)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss = torch.tensor(float("nan"))
        for i in range(iters):
            loss = ((self(x) - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if verbose and i % 200 == 0:
                print(f"    [init_from] {i:5d}  {float(loss):.3e}")
        return float(loss.detach())


# --------------------------------------------------------------------------- #
#  mlp
# --------------------------------------------------------------------------- #
class _MLP(Learnable):
    """tanh-activated fully connected. The baseline: anything that cannot beat it is not earning
    its parameters, and anything it beats was not a useful hypothesis."""

    def __init__(self, n_in, n_out, hidden=64, layers=3, seed=0, **_):
        super().__init__(n_in, n_out)
        g = torch.Generator().manual_seed(int(seed))
        dims = [n_in] + [int(hidden)] * (int(layers) - 1) + [n_out]
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(a, b)
            with torch.no_grad():
                lin.weight.copy_(torch.randn(b, a, generator=g) / math.sqrt(a))
                lin.bias.zero_()
            mods += [lin, nn.Tanh()]
        self.net = nn.Sequential(*mods[:-1])              # no activation on the output

    def forward(self, x):
        return self.net(x)


@register_learnable("mlp", family="dense")
class MLPIntrinsic(_MLP):
    SHAPE = INTRINSIC


@register_learnable("mlp_edge", family="dense")
class MLPRelational(_MLP):
    SHAPE = RELATIONAL


# --------------------------------------------------------------------------- #
#  siren
# --------------------------------------------------------------------------- #
class _Siren(Learnable):
    """Sinusoidal first layer: sin(omega0 W x + b), then tanh layers.

    `omega0` is the frequency the first layer is initialised at, and it is the one knob that
    matters: a smooth net needs many layers to represent a sharp saturation, while a sinusoidal
    one represents it in the first. 30 is the value the SIREN paper uses for signals normalised to
    [-1, 1]; a state block with a wider range wants a smaller one, which is why it is exposed.

    The first layer's weights are drawn U(-1/n, 1/n) and the rest U(-sqrt(6/n)/omega0, +...), the
    initialisation that keeps the pre-activation distribution stable through depth -- without it a
    sinusoidal net's output is white noise at depth 3 and the fit never starts.

    Reference: Sitzmann, V. et al. (2020). Implicit Neural Representations with Periodic
    Activation Functions. NeurIPS.
    """

    def __init__(self, n_in, n_out, hidden=64, layers=3, omega0=30.0, seed=0, **_):
        super().__init__(n_in, n_out)
        g = torch.Generator().manual_seed(int(seed))
        self.omega0 = float(omega0)
        dims = [n_in] + [int(hidden)] * (int(layers) - 1) + [n_out]
        self.lins = nn.ModuleList()
        for k, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            lin = nn.Linear(a, b)
            with torch.no_grad():
                if k == 0:
                    lin.weight.uniform_(-1.0 / a, 1.0 / a, generator=g)
                else:
                    w = math.sqrt(6.0 / a) / self.omega0
                    lin.weight.uniform_(-w, w, generator=g)
                lin.bias.zero_()
            self.lins.append(lin)

    def forward(self, x):
        h = torch.sin(self.omega0 * self.lins[0](x))
        for lin in self.lins[1:-1]:
            h = torch.tanh(lin(h))
        return self.lins[-1](h)


@register_learnable("siren", family="implicit")
class SirenIntrinsic(_Siren):
    SHAPE = INTRINSIC


@register_learnable("siren_edge", family="implicit")
class SirenRelational(_Siren):
    SHAPE = RELATIONAL


# --------------------------------------------------------------------------- #
#  table
# --------------------------------------------------------------------------- #
class _Table(Learnable):
    """A free value per bin of the first input. No smoothness assumed anywhere.

    THE UPPER BOUND, and that is what it is for. A table with enough bins can represent any law
    of one variable, so a smooth net that MATCHES it has lost nothing by being smooth, and one
    that falls short of it has been limited by its own inductive bias rather than by the data.
    Running a smooth family and a table on the same task is how you tell those apart, which no
    single fit can.

    Linear interpolation between bins rather than nearest, so the derivative exists and gradient
    descent has something to follow -- a piecewise-constant table trains only the bins the data
    lands in and leaves the rest at their initial value forever.
    """

    def __init__(self, n_in, n_out, bins=64, lo=-3.0, hi=3.0, seed=0, **_):
        super().__init__(n_in, n_out)
        self.bins, self.lo, self.hi = int(bins), float(lo), float(hi)
        self.values = nn.Parameter(torch.zeros(int(bins), n_out))

    def forward(self, x):
        u = ((x[:, :1] - self.lo) / (self.hi - self.lo)).clamp(0, 1) * (self.bins - 1)
        i0 = u.floor().long().clamp(0, self.bins - 2)
        w = (u - i0.to(u.dtype))
        v0 = self.values[i0[:, 0]]
        v1 = self.values[i0[:, 0] + 1]
        return v0 * (1 - w) + v1 * w


@register_learnable("table", family="lookup")
class TableIntrinsic(_Table):
    SHAPE = INTRINSIC


@register_learnable("table_edge", family="lookup")
class TableRelational(_Table):
    SHAPE = RELATIONAL
