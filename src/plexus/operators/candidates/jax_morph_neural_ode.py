"""regulate:neural_ode -- cell <- (cell's own fields). A per-cell internal regulatory ODE
whose reaction law is a free-form neural network.

This is the `neural_ode` implementation of the SAME `regulate` contract as the shipped
`connectionist` circuit (`jax_morph_odecontroller.py`) and the `mwc` sibling: a
cell-autonomous controller that carries an evolving internal state `g` (the source's
`y = concat(hidden, outputs)`, one first-order block here), reads a FIXED sensed drive `u`
held constant across the macro-step, SELF-SOLVES the ODE over [0, dt] with an adaptive
Dopri5 integrator, and returns the EXACT integrated increment `g(dt) - g(0)` as the cell's
gene-state delta. The three implementations share signature, routing, and solver and differ
in EXACTLY ONE place -- the vector field:

    connectionist : dg/dt = sigma( W_gene @ g + W_in @ u + b ) - gamma * g
    mwc           : dg/dt = rho * sigma( log-occupancy drive ) - g / tau
    neural_ode    : dg/dt = MLP( concat(u, g) )                        <-- this file

The MLP maps `n_in + n_gene -> n_gene` (the source's `make_mlp`: in_size + hidden + out_size
-> hidden + out_size). It is the generic, uninterpretable member: no explicit regulation
matrix, bias, or degradation -- those are whatever the network learns. THIS is where the
trainable parameters live.

BIOLOGY vs PAPER (source wins, rule 5): the paper (Deshpande, Mottes et al. 2025, eq. (4),
p. 16 / p. 10 fig. 1b) describes ONLY the structured gene-regulatory ODE; the free-form MLP
member has NO paper counterpart -- it is a library generalization of the identical
integration machinery. A paper-only reimplementer would build `connectionist`, never this;
that absence is recorded as a surprise, not as scope.

Routing (identical to the connectionist sibling, so the two are interchangeable):

* `kind=exchange, family=fields, set=cell, maps=[]` -- INTRACELLULAR: each cell integrates
  its own circuit in isolation, no cell-to-cell edge, no gather/scatter map (that is what
  separates `regulate` from `signal`, a connectome morphism on a `synapse` edge-set).
* The evolving state is ONE first-order block `state:` (default `gene`). `hidden_size` is a
  documented latent/output split that does NOT change the integration -- the whole block is
  solved as one coupled vector, exactly as the source concatenates hidden and outputs into
  `y0`. The sensed driver block `inputs:` is read-only and frozen for the whole solve.
* SELF-SOLVED INCREMENT, NOT A RATE. The operator does the whole adaptive integration
  internally; its result is the exact change `g(dt) - g(0)`. In the JAX source the DYNAMIC
  step returns that increment and the Model ADDS it. Plexus integrates a first-order block as
  `g += dt * delta`, so we return the effective mean rate `delta = (g(dt) - g(0)) / dt`; the
  engine's `dt *` then recovers the exact endpoint. The dt cancels -- it is NOT a second
  integration, it is the faithful adaptation to Plexus's `x += dt*delta` convention.
* `EMIT="velocity"` (first-order block) with class `INTEGRAND="gene"` so `_resolve_emit` sees
  a non-`pos` integrand and never constrains the cell's spatial integration order; instance
  `self.INTEGRAND` routes the delta to the configured block.

Reference: Deshpande, Mottes et al., "Engineering morphogenesis of cell clusters with
differentiable programming", Nat. Comput. Sci. (2025), eq. (4) p. 16 (the ODE-controller
family this generalizes); translated from papers/jax-morph/jax_morph/control/ode.py:281
(NeuralODE.vector_field), :328 (make_mlp) and :161 (ODEController.__call__).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


# Dormand-Prince 5(4) tableau -- the diffrax `Dopri5` the source integrates with, and the
# SHARED solver of every `regulate` implementation (identical to the connectionist sibling);
# only the vector field `_field` differs between them. The field is AUTONOMOUS (`del t` in the
# source), so the `c` nodes never enter f.
_A = (
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)          # 5th-order weights
_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)  # embedded 4th-order

_ACTIVATIONS = {                          # eqx.nn.MLP defaults to jax.nn.relu
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}


def _net_io(net: nn.Module):
    """(in_features, out_features) of an MLP by its first/last Linear layer."""
    lins = [m for m in net.modules() if isinstance(m, nn.Linear)]
    return lins[0].in_features, lins[-1].out_features


@register_operator("regulate", family="fields", set="cell", kind="exchange",
                   implementation="neural_ode")
class NeuralODERegulate(Exchange):
    """The `neural_ode` implementation of the `regulate` contract: an MLP per-cell vector
    field, self-solved over the macro-step. Same signature, routing, and adaptive Dopri5
    solver as the `connectionist`/`mwc` siblings -- only the reaction law differs. Translated
    to torch from papers/jax-morph/jax_morph/control/ode.py (NeuralODE + ODEController)."""

    EMIT = "velocity"                  # the gene block is first-order; the delta is dg/dt-equivalent (inc/dt)
    INTEGRAND = "gene"                 # writes a NON-coordinate block (the evolving gene vector), not pos
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["gene", "drive"]          # evolving gene vector g (state) + fixed driver u (inputs, read-only)
    WRITES = ["gene"]                  # the dt-increment of the gene vector
    MAPS = []                          # intracellular: no gather/scatter, zero cell-to-cell coupling
    SUPPORTED_DIMS = [2, 3]            # acts on per-cell state; ignores spatial dimension
    DIFFERENTIABLE = True              # the RK steps are plain torch ops -> autograd flows (matches diffrax/equinox)
    REQUIRES_PARAMS = []               # all params optional (an untrained MLP is a valid, inert-ish circuit)
    MECHANISM_TAGS = ["gene_regulatory_network", "internal_state_ode",
                      "genotype_phenotype_map", "self_solved_macrostep", "neural_vector_field"]
    PARAM_ROLES = {
        "state": "evolving_gene_block", "inputs": "fixed_driver_block",
        "hidden_size": "latent_regulator_width", "width": "mlp_hidden_width",
        "depth": "mlp_depth", "activation": "mlp_nonlinearity", "seed": "init_seed",
        "rtol": "solver_rel_tol", "atol": "solver_abs_tol",
    }
    REFERENCE = ("Deshpande, Mottes et al. (2025), Nat. Comput. Sci., eq. (4) p. 16 & fig. 1b; "
                 "papers/jax-morph/jax_morph/control/ode.py:281 (NeuralODE.vector_field), "
                 ":328 (make_mlp) & :161 (ODEController.__call__).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.state_block = params.get("state", "gene")          # the evolving gene vector (hidden ++ outputs)
        self.input_block = params.get("inputs", None)           # the fixed sensed drive u (None -> autonomous)
        self.hidden_size = int(params.get("hidden_size", 0))    # leading latent columns (documented split; whole block is integrated)
        if self.hidden_size < 0:                                # mirror the source's __init__ guard
            raise ValueError(f"hidden_size must be non-negative, got {self.hidden_size}")
        # MLP vector-field hyperparameters (mirror NeuralODE.make_mlp: width=64, depth=2, relu).
        self.width = int(params.get("width", 64))
        self.depth = int(params.get("depth", 2))
        act = str(params.get("activation", "relu")).lower()
        if act not in _ACTIVATIONS:
            raise ValueError(f"activation {act!r} not in {sorted(_ACTIVATIONS)}")
        self.activation = act
        self.seed = params.get("seed", None)
        self.rtol = float(params.get("rtol", 1e-4))             # source PIDController rtol
        self.atol = float(params.get("atol", 1e-6))             # source PIDController atol
        self.max_steps = int(params.get("max_steps", 4096))     # adaptive-solver safety cap
        # a pre-built vector-field module may be injected (mirrors NeuralODE taking a pre-sized
        # mlp); otherwise it is built lazily on first forward, once block widths are known.
        self.net = params.get("net", None)
        self._net_shape = None                                  # (in, out) once validated/built
        # instance INTEGRAND: route the delta into the configured gene block (engine reads this
        # off the instance). The class INTEGRAND stays "gene" so _resolve_emit sees a non-`pos`
        # integrand and does not constrain the coordinate's integration order.
        self.INTEGRAND = self.state_block

    # --- the MLP vector field (eqx.nn.MLP(in, out, width, depth) mirrored in torch) ------ #
    def _build_net(self, in_dim: int, out_dim: int) -> nn.Module:
        """depth+1 Linear layers with `activation` after every layer but the last (identity
        final activation) -- the eqx.nn.MLP topology. depth<=0 is a single linear map."""
        act = _ACTIVATIONS[self.activation]
        if self.depth <= 0:
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = [nn.Linear(in_dim, self.width)]
            for _ in range(self.depth - 1):
                layers += [act(), nn.Linear(self.width, self.width)]
            layers += [act(), nn.Linear(self.width, out_dim)]
        net = nn.Sequential(*layers)
        if self.seed is not None:                               # reproducible init without disturbing global RNG
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(int(self.seed))
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
            torch.random.set_rng_state(rng_state)
        return net

    def _ensure_net(self, n_in: int, n_gene: int, device, dtype) -> nn.Module:
        want = (n_in + n_gene, n_gene)                          # MLP: in_size + n_gene -> n_gene
        if self.net is not None and self._net_shape is None:    # validate an injected net once (reference's ctor check)
            got = _net_io(self.net)
            if got != want:
                raise ValueError(
                    f"regulate:neural_ode net must map {want[0]} -> {want[1]} "
                    f"(in_size + hidden + out_size -> hidden + out_size), got {got[0]} -> {got[1]}")
            self.net = self.net.to(device=device, dtype=dtype)
            self._net_shape = want
        if self.net is None:                                    # lazy build once widths are known
            self.net = self._build_net(*want).to(device=device, dtype=dtype)
            self._net_shape = want
        return self.net

    def _field(self, g, u):
        """dg/dt for every cell: MLP(concat(u, g)). `u` is the frozen driver, closed over for
        the whole solve; when there is no driver block the field is autonomous, MLP(g)."""
        inp = g if u is None else torch.cat([u, g], dim=1)
        return self.net(inp)

    # --- adaptive Dopri5 over [0, dt] (diffrax Dopri5 + PIDController(rtol, atol)) ------ #
    def _solve(self, y0, u, dt):
        """Integrate dy/dt = field(y, u) from 0 to dt, returning y(dt). Batched over cells
        with ONE shared adaptive step sequence (RMS error norm over all elements), matching
        the source's single `diffeqsolve` on the stacked per-cell state. First step = dt (the
        source's dt0=dt). This is the SHARED solver of every `regulate` implementation."""
        t, y, h = 0.0, y0, float(dt)
        safety, minfac, maxfac = 0.9, 0.2, 5.0
        for _ in range(self.max_steps):
            if t >= dt:
                break
            h = min(h, dt - t)                                  # never overshoot the endpoint
            k1 = self._field(y, u)
            k2 = self._field(y + h * (_A[0][0] * k1), u)
            k3 = self._field(y + h * (_A[1][0] * k1 + _A[1][1] * k2), u)
            k4 = self._field(y + h * (_A[2][0] * k1 + _A[2][1] * k2 + _A[2][2] * k3), u)
            k5 = self._field(y + h * (_A[3][0] * k1 + _A[3][1] * k2 + _A[3][2] * k3
                                      + _A[3][3] * k4), u)
            k6 = self._field(y + h * (_A[4][0] * k1 + _A[4][1] * k2 + _A[4][2] * k3
                                      + _A[4][3] * k4 + _A[4][4] * k5), u)
            y5 = y + h * (_B5[0] * k1 + _B5[2] * k3 + _B5[3] * k4 + _B5[4] * k5 + _B5[5] * k6)
            k7 = self._field(y5, u)                             # FSAL node for the embedded estimate
            y4 = y + h * (_B4[0] * k1 + _B4[2] * k3 + _B4[3] * k4 + _B4[4] * k5
                          + _B4[5] * k6 + _B4[6] * k7)
            # per-element scale atol + rtol*max(|y|,|y5|); RMS norm (diffrax's default).
            scale = self.atol + self.rtol * torch.maximum(y.abs(), y5.abs())
            err = torch.sqrt(torch.mean(((y5 - y4) / scale) ** 2)).item()
            if err <= 1.0:                                      # accept the step
                t += h
                y = y5
            fac = maxfac if err == 0.0 else min(maxfac, max(minfac, safety * err ** -0.2))
            h = h * fac
        return y

    def forward(self, H, mask=None):
        cell = H.level(self.at)
        g0 = cell.get(self.state_block)                         # [N, n_gene] the evolving gene vector
        u = cell.get(self.input_block) if self.input_block else None   # [N, n_in] frozen drive, or None
        n_gene = g0.shape[-1]
        n_in = u.shape[-1] if u is not None else 0
        self._ensure_net(n_in, n_gene, g0.device, g0.dtype)
        dt = float(getattr(H.config, "dt", 1.0))
        if dt <= 0.0:
            return {self.at: torch.zeros_like(g0)}
        y_end = self._solve(g0, u, dt)
        # the EXACT integrated increment, expressed as the effective mean rate so the engine's
        # first-order step (g += dt*delta) recovers y(dt). dt cancels.
        delta = (y_end - g0) / dt
        delta = delta * cell.occ[:, None]                       # dormant cells hold their gene state
        if mask is not None:
            delta = delta * mask[:, None].float()
        return {self.at: delta}
