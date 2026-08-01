"""regulate -- cell <- (cell's own fields). A per-cell gene-regulatory ODE.

Each cell carries an internal regulatory circuit: a vector of gene concentrations `g`
that evolves in continuous time under its own recurrent regulation plus a FIXED sensed
drive `u` (the chemical / mechanical signals the cell reads from its environment). Over
one macro-step [0, dt] the operator SELF-SOLVES

    dg/dt = sigma( W_gene @ g + W_in @ u + b ) - gamma * g          (u held fixed)

with an adaptive Dopri5 integrator, and returns the EXACT integrated increment
`g(dt) - g(0)` as the cell's gene-state delta. This is the genotype->phenotype decision
function: the evolved output genes downstream set division rate, secretion, adhesion,
and `hidden_size` of the genes are latent regulators that couple the circuit but are
never read directly. The gene vector persists as heritable cell state.

Routing (the `regulate` contract, kind=exchange, family=fields, set=cell, maps=[]):
each cell integrates its OWN circuit in isolation -- `W_gene`/`W_in` are dense
per-operator matrices, there is NO cell-to-cell edge and NO gather/scatter map. That is
what separates `regulate` from `signal` (a connectome morphism on a `synapse` edge-set):
`regulate` is INTRACELLULAR. The evolving state is one first-order block `state:` (the
paper's `y = concat(hidden, outputs)`, integrated as one coupled system); the sensed
driver block `inputs:` is read-only and frozen across the solve.

Two subtleties a reimplementer must not miss (both from the source):

* SELF-SOLVED INCREMENT, NOT A RATE. Unlike a typical first-order operator that returns
  an instantaneous dg/dt for the engine to Euler-step, this operator does the whole
  adaptive integration internally and its result is the exact change over the interval.
  The Plexus engine integrates a first-order block as `g += dt * delta`, so we return the
  effective mean rate `delta = (g(dt) - g(0)) / dt`; the engine's `dt *` then recovers the
  exact endpoint `g(dt)`. (In the JAX source the DYNAMIC step returns `g(dt) - g(0)`
  directly and the Model adds it; dividing by dt here is the faithful adaptation to
  Plexus's `x += dt*delta` convention, not a second integration -- the dt cancels.)

* PAPER vs CODE contradiction on the forcing input; SOURCE WINS. The paper (p. 10,
  "Genetic regulatory interactions") writes dg_i/dt = phi(sum_j W_ij g_j + b_i) + I_i
  - k_i g_i, with the sensed input I_i ADDITIVE and OUTSIDE the sigmoid. The shipped
  circuit `GeneNetworkConnectionist` puts the drive INSIDE the sigmoid via `W_in @ u`
  and has no separate additive term. We implement the CODE (this is the
  `connectionist` implementation of `regulate`), because the differential test compares
  us to the running source, not the prose.

The saturation is the ALGEBRAIC sigmoid `0.5 + 0.5 * x / sqrt(1 + x^2)` (the source's
`_rescaled_sigmoid`), NOT the logistic -- the `GeneNetworkMWC` sibling implementation
uses the logistic instead, two saturations under one paper symbol.

Reference: Deshpande, Mottes et al., "Engineering morphogenesis of cell clusters with
differentiable programming", Nat. Comput. Sci. (2025), p. 10 & fig. 1b; translated from
papers/jax-morph/jax_morph/control/ode.py:46 (ODEController base) and :202
(GeneNetworkConnectionist.vector_field, L261).
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


# Dormand-Prince 5(4) tableau (the diffrax `Dopri5` the source uses). The field is
# AUTONOMOUS (`del t` in the source vector fields), so the `c` nodes never enter f.
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


# implementation="ode_generic", NOT "connectionist": the concrete paper variant lives in
# jax_morph_gene_network_connectionist.py (it uses the source's RESCALED sigmoid). Both agents
# reached for the same implementation label, and the clash was invisible until the modules were
# imported into ONE interpreter -- which is now part of verify_impl.py.
@register_operator("regulate", family="fields", set="cell", kind="exchange",
                   implementation="ode_generic")
class Regulate(Exchange):
    """The `connectionist` implementation of the `regulate` contract: a per-cell gene
    circuit with a sigmoid-saturated linear regulatory drive and linear degradation,
    self-solved over the macro-step. `mwc` (thermodynamic log-occupancy drive) and
    `neural_ode` (an MLP vector field) are sibling implementations of the SAME contract --
    same signature, same self-solve, differing only in the reaction law."""

    EMIT = "velocity"                  # the gene block is first-order; the delta is dg/dt-equivalent (inc/dt)
    INTEGRAND = "gene"                 # writes a NON-coordinate block (the evolving gene vector), not pos
    # typed signature (Plexus2 sec. 2.1): a morphism cell -> cell that reads the cell's
    # own gene state + its fixed sensed drive and writes the gene state. MAPS=[] is
    # load-bearing: there is no incidence map and no neighbour edge -- each cell integrates
    # in isolation (the intracellular identity that distinguishes it from `signal`).
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["gene", "drive"]          # evolving gene vector g (state) + fixed driver u (inputs, read-only)
    WRITES = ["gene"]                  # the dt-increment of the gene vector
    MAPS = []                          # intracellular: no gather/scatter, zero cell-to-cell coupling
    SUPPORTED_DIMS = [2, 3]            # acts on per-cell state; ignores spatial dimension
    REQUIRES_PARAMS = []               # all params optional (zeros defaults, like the source's _resolve_param)
    MECHANISM_TAGS = ["gene_regulatory_network", "internal_state_ode",
                      "genotype_phenotype_map", "self_solved_macrostep"]
    PARAM_ROLES = {
        "state": "evolving_gene_block", "inputs": "fixed_driver_block",
        "hidden_size": "latent_regulator_width", "W_gene": "gene_interaction_matrix",
        "W_in": "input_mixing_matrix", "b": "basal_drive", "gamma": "degradation_rate",
        "rtol": "solver_rel_tol", "atol": "solver_abs_tol",
    }
    REFERENCE = ("Deshpande, Mottes et al. (2025), Nat. Comput. Sci., p. 10 & fig. 1b; "
                 "papers/jax-morph/jax_morph/control/ode.py:46 (ODEController) & :261 "
                 "(GeneNetworkConnectionist.vector_field).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.state_block = params.get("state", "gene")          # the evolving gene vector (hidden ++ outputs)
        self.input_block = params.get("inputs", None)           # the fixed sensed drive u (None -> autonomous)
        self.hidden_size = int(params.get("hidden_size", 0))    # leading latent columns (documented split; whole block is integrated)
        if self.hidden_size < 0:                                # mirror the source's __init__ guard
            raise ValueError(f"hidden_size must be non-negative, got {self.hidden_size}")
        # the connectionist reaction-law params -- kept raw here, materialised lazily in
        # forward() once the block widths (n_gene, n_in) are known from the actual state.
        self._raw = {k: params.get(k) for k in ("W_gene", "W_in", "b")}
        self.gamma = params.get("gamma", 0.1)                   # scalar or per-gene; source default 0.1
        self.rtol = float(params.get("rtol", 1e-4))             # source PIDController rtol
        self.atol = float(params.get("atol", 1e-6))             # source PIDController atol
        self.max_steps = int(params.get("max_steps", 4096))     # adaptive-solver safety cap
        # instance INTEGRAND: route the delta into the configured gene block (engine reads
        # this off the instance). The class INTEGRAND stays "gene" so _resolve_emit sees a
        # non-`pos` integrand and does not constrain the coordinate's integration order.
        self.INTEGRAND = self.state_block
        self._params = None                                     # (W_gene, W_in, b, gamma) tensors, built on first forward

    # --- the reaction law (GeneNetworkConnectionist.vector_field, autonomous) ---------- #
    @staticmethod
    def _sigmoid(x):
        """The algebraic sigmoid 0.5 + 0.5 x/sqrt(1+x^2) (the source's _rescaled_sigmoid);
        `hypot(1, x)` computes sqrt(1+x^2) without overflowing x*x."""
        return 0.5 * (1.0 + x / torch.hypot(torch.ones_like(x), x))

    def _field(self, g, u):
        """dg/dt for every cell: sigma(g @ W_gene^T + u @ W_in^T + b) - gamma * g. `u` is
        the frozen driver, closed over for the whole solve (the quasistatic-chemistry
        assumption)."""
        W_gene, W_in, b, gamma = self._params
        drive = g @ W_gene.t() + b
        if u is not None and W_in.shape[1] > 0:
            drive = drive + u @ W_in.t()
        return self._sigmoid(drive) - gamma * g

    def _materialise(self, n_gene, n_in, device, dtype):
        """Build (W_gene, W_in, b, gamma) tensors once the widths are known; unset
        matrices default to zeros (an inert circuit), exactly as the source does."""
        def as_tensor(v, shape):
            if v is None:
                return torch.zeros(shape, device=device, dtype=dtype)
            t = torch.as_tensor(v, device=device, dtype=dtype)
            if tuple(t.shape) != shape:
                raise ValueError(f"regulate param shape {tuple(t.shape)} != required {shape}")
            return t
        W_gene = as_tensor(self._raw["W_gene"], (n_gene, n_gene))
        W_in = as_tensor(self._raw["W_in"], (n_gene, n_in))
        b = as_tensor(self._raw["b"], (n_gene,))
        gamma = torch.as_tensor(self.gamma, device=device, dtype=dtype)   # scalar or (n_gene,)
        self._params = (W_gene, W_in, b, gamma)

    # --- adaptive Dopri5 over [0, dt] (diffrax Dopri5 + PIDController(rtol, atol)) ------ #
    def _solve(self, y0, u, dt):
        """Integrate dy/dt = field(y, u) from 0 to dt, returning y(dt). Batched over cells
        with ONE shared adaptive step sequence (RMS error norm over all elements), matching
        the source's single `diffeqsolve` on the stacked per-cell state. First step = dt
        (the source's dt0=dt)."""
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
        if self._params is None:
            self._materialise(n_gene, n_in, g0.device, g0.dtype)
        dt = float(getattr(H.config, "dt", 1.0))
        y_end = self._solve(g0, u, dt)
        # the EXACT integrated increment, expressed as the effective mean rate so the
        # engine's first-order step (g += dt*delta) recovers y(dt). dt cancels.
        delta = (y_end - g0) / dt
        delta = delta * cell.occ[:, None]                       # dormant cells hold their gene state
        if mask is not None:
            delta = delta * mask[:, None].float()
        return {self.at: delta}
