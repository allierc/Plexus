"""regulate:connectionist -- a per-cell gene-regulatory ODE with a linear (connectionist) drive.

BIOLOGY. Every cell carries an INTERNAL regulatory circuit: a vector of gene
concentrations `g` (optional hidden/latent regulator genes concatenated AHEAD of the
observed output genes, so `g = concat(hidden, outputs)` and `n_gene = hidden_size +
out_size`). Over one macro-step [0, dt] the circuit integrates the autonomous per-cell ODE

    dg/dt = sigma( W_gene @ g + W_in @ u + b ) - gamma * g          (u held fixed)

where `u` is a FIXED quasistatic snapshot of the driver signals the cell senses
(chemistry / mechanics, sensed onto the cell by an upstream step and frozen for the
solve), `W_gene` is the dense WITHIN-cell gene->gene regulatory matrix, `W_in` the
learnable input coupling, `b` the basal transcription bias, `gamma` the per-gene linear
degradation, and `sigma` the ALGEBRAIC (rescaled) sigmoid `0.5 + 0.5 x / sqrt(1 + x^2)`
-- NOT the logistic. The evolved output genes are the genotype->phenotype decision
variables downstream steps (division / secretion / adhesion) read; the whole gene vector
persists as heritable cell state.

ROUTING (the `regulate` contract: kind=exchange, family=fields, set=cell, maps=[]). Each
cell integrates its OWN circuit in isolation -- `W_gene`/`W_in` are dense per-operator
matrices, there is NO cell-to-cell edge and NO gather/scatter map. That maps=[] is what
separates `regulate` from `signal` (a connectome morphism on a `synapse` EDGE-SET with
pre/post maps): `regulate` is INTRACELLULAR. This is the CONNECTIONIST (linear `W*g`
drive) implementation of the contract; `mwc` (thermodynamic log-occupancy drive) and
`neural_ode` (a black-box MLP vector field) are sibling implementations of the SAME
signature, differing ONLY in the reaction law -- so the ODEController subclasses collapse
to one `regulate` contract with interchangeable implementations, selected by
`implementation:`.

INTEGRATION -- SELF-SOLVED INCREMENT, NOT A RATE. The source is an ODEController: it
integrates the ODE over the WHOLE macro-step with an adaptive diffrax Dopri5 solver
(PIDController rtol=1e-4, atol=1e-6, dt0=dt) and its DYNAMIC step returns the exact
integrated increment `g(dt) - g(0)` (a sparse delta the Model accumulates), never a
first-order rate*dt. The Plexus engine, by contrast, integrates a first-order block with
a single Euler step per tick (`g += dt * delta`). To reproduce the reference ENDPOINT we
do the whole integration internally over [0, dt] (fixed-step RK4, `substeps`) and return
the effective MEAN rate `(g(dt) - g(0)) / dt`; the engine's `dt *` then recovers the
exact `g(dt)`. The dt cancels -- this is the faithful adaptation to Plexus's `x += dt*delta`
convention, NOT a second integration. (Fixed-step RK4 here vs adaptive Dopri5 in the
reference is a numerics choice within the one contract; raise `substeps` to tighten it.)

PAPER vs CODE contradiction on the forcing input; SOURCE WINS. The paper (p. 10,
"Genetic regulatory interactions", inspired by Hiscock 2019 [24]) writes
`dg_i/dt = phi(sum_j W_ij g_j + b_i) + I_i - k_i g_i`, with the sensed input `I_i`
ADDITIVE and OUTSIDE the sigmoid. The shipped `GeneNetworkConnectionist` puts the input
INSIDE the sigmoid via `W_in @ u` (a full trainable coupling) with no separate additive
term, so the input drive saturates together with the regulatory drive. We implement the
CODE, because the differential test compares us to the running source, not the prose.

Translated from papers/jax-morph/jax_morph/control/ode.py:202-278
(GeneNetworkConnectionist.vector_field, L261-278) and :161-199 (ODEController.__call__,
the macro-step solve + sparse delta). Torch, not JAX.

Reference: Deshpande, Mottes, Vidal Saez, Kicheva & Hiscock (2025), "Engineering
morphogenesis of cell clusters with differentiable programming", Nat. Comput. Sci.,
p. 10 & fig. 1b.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


def _rescaled_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """The algebraic sigmoid `0.5 + 0.5 x / sqrt(1 + x^2)` (the source's `_rescaled_sigmoid`,
    ode.py:27-35), overflow-safe: clip x to the dtype's finite range, then rescale by
    max(1, |x|) so `x*x` never overflows float32 before the solver would diverge. Range
    (0, 1); sigma(0) = 0.5; asymptotes to 0/1 as x -> -/+ inf. NOT the logistic (the `mwc`
    sibling uses `torch.sigmoid`; two saturations under one paper symbol phi)."""
    limit = torch.finfo(x.dtype).max
    finite_x = x.clamp(-limit, limit)
    scale = finite_x.abs().clamp(min=1.0)
    scaled_x = finite_x / scale
    scaled_one = 1.0 / scale
    ratio = scaled_x / torch.sqrt(scaled_one * scaled_one + scaled_x * scaled_x)
    return 0.5 + 0.5 * ratio


@register_operator("regulate", family="fields", set="cell", kind="exchange",
                   implementation="connectionist")
class RegulateConnectionist(Exchange):
    """The `connectionist` implementation of the `regulate` contract: a per-cell gene
    circuit with a sigmoid-saturated LINEAR regulatory drive and linear degradation,

        dg/dt = sigma(W_gene @ g + W_in @ u + b) - gamma * g,

    self-solved over the macro-step and returned as the gene-state delta. `mwc`
    (thermodynamic log-occupancy drive) and `neural_ode` (an MLP vector field) are sibling
    implementations of the SAME contract -- same signature, same self-solve, differing
    only in the reaction law."""

    EMIT = "velocity"                  # the gene block is first-order; the delta is the dt-increment expressed as inc/dt
    INTEGRAND = "gene"                 # writes a NON-coordinate block (the evolving gene vector), not the spatial pos
    # typed signature (Plexus2 sec. 2.1): a morphism cell -> cell that reads the cell's own
    # gene state + its fixed sensed drive and writes the gene state. MAPS=[] is load-bearing:
    # no incidence map, no neighbour edge -- each cell integrates in isolation (the
    # intracellular identity that distinguishes `regulate` from the lateral `signal`).
    INPUTS = ["cell", "drive"]
    OUTPUTS = ["cell"]
    READS = ["gene", "drive"]   # evolving gene vector g (state) + fixed driver u (read-only)
    WRITES = ["gene"]                         # the dt-increment of the gene vector
    MAPS = []                                 # intracellular: no gather/scatter, zero cell-to-cell coupling
    SUPPORTED_DIMS = [2, 3]                    # acts on per-cell state; ignores spatial dimension
    DIFFERENTIABLE = True                     # pure-torch vector field + RK4; grads flow through for the inverse problem
    REQUIRES_PARAMS = []                       # all params optional (zeros defaults, like the source's _resolve_param)
    MECHANISM_TAGS = ["gene_regulatory_network", "connectionist_drive", "internal_state_ode",
                      "genotype_phenotype_map", "recurrent", "self_solved_macrostep"]
    PARAM_ROLES = {
        "gene": "evolving_gene_block",
        "inputs": "fixed_driver_blocks",
        "hidden_size": "latent_regulator_count",
        "W_gene": "regulatory_coupling_matrix",
        "W_in": "input_sensitivity_matrix",
        "b": "basal_production_bias",
        "gamma": "degradation_rate",
        "substeps": "macro_step_rk4_substeps",
    }
    REFERENCE = (
        "Deshpande, Mottes, Vidal Saez, Kicheva & Hiscock (2025), 'Engineering "
        "morphogenesis of cell clusters with differentiable programming', Nat. Comput. Sci., "
        "p. 10 & fig. 1b (dg_i/dt = phi(sum_j W_ij g_j + b_i) + I_i - k_i g_i; inspired by "
        "Hiscock 2019 [24]). Translated from papers/jax-morph/jax_morph/control/ode.py:261-278 "
        "(GeneNetworkConnectionist.vector_field) & :161-199 (ODEController.__call__). "
        "SOURCE WINS: the code mixes the sensed input INSIDE the sigmoid via a trainable W_in; "
        "the paper adds I_i OUTSIDE it. sigma is the algebraic 0.5+0.5 x/sqrt(1+x^2), not logistic."
    )

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.gene_block = params.get("gene", params.get("block", "gene"))   # the evolving gene vector (hidden ++ outputs)
        # the fixed sensed drive u: a block name or a list of names, concatenated in order
        # (the reference packs input_specs). None / [] -> autonomous circuit (in_size = 0).
        inp = params.get("inputs", params.get("from"))
        if inp is None:
            self.input_blocks = []
        elif isinstance(inp, str):
            self.input_blocks = [inp]
        else:
            self.input_blocks = list(inp)
        self.hidden_size = int(params.get("hidden_size", 0))     # leading latent columns (documented split; whole block integrated)
        if self.hidden_size < 0:                                 # mirror the source's __init__ guard
            raise ValueError(f"hidden_size must be non-negative, got {self.hidden_size}")
        self.substeps = max(1, int(params.get("substeps", 8)))   # fixed RK4 substeps over the macro-step
        # instance INTEGRAND: route the delta into the configured gene block (the engine reads
        # this off the instance). The class INTEGRAND stays "gene" so _resolve_emit sees a
        # non-`pos` integrand and does not constrain the coordinate's integration order.
        self.INTEGRAND = self.gene_block
        # the connectionist reaction-law params -- kept raw here, materialised lazily on the
        # first forward once the widths (n_gene, in_size) are known from the actual state.
        self._raw = {k: params.get(k) for k in ("W_gene", "W_in", "b")}
        self.gamma = params.get("gamma", 0.1)                    # scalar or per-gene; source default 0.1 (stored verbatim, NOT shape-checked)
        self._params = None                                      # (W_gene, W_in, b, gamma) tensors, built on first use

    # --- parameter materialization (None -> zeros, else shape-checked) ---------- #
    def _ensure_params(self, n_gene, in_size, device, dtype):
        """Build (W_gene, W_in, b, gamma) tensors once the widths are known; an unset
        matrix defaults to zeros (an inert-INTERACTION circuit), exactly as the source's
        `_resolve_param` does. `gamma` is stored verbatim and NOT shape-checked -- the one
        regulatory parameter the source does not route through `_resolve_param`."""
        if self._params is not None:
            return
        def as_tensor(v, shape, name):
            if v is None:
                return torch.zeros(shape, device=device, dtype=dtype)
            t = torch.as_tensor(v, device=device, dtype=dtype)
            if tuple(t.shape) != shape:
                raise ValueError(
                    f"regulate:connectionist param {name!r} must have shape {shape}, got {tuple(t.shape)}")
            return t
        W_gene = as_tensor(self._raw["W_gene"], (n_gene, n_gene), "W_gene")
        W_in = as_tensor(self._raw["W_in"], (n_gene, in_size), "W_in")
        b = as_tensor(self._raw["b"], (n_gene,), "b")
        gamma = torch.as_tensor(self.gamma, device=device, dtype=dtype)     # scalar or (n_gene,); broadcast, not shape-checked
        self._params = (W_gene, W_in, b, gamma)

    # --- the reaction law (GeneNetworkConnectionist.vector_field, autonomous) ---- #
    def vector_field(self, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """dg/dt for every cell: `sigma(g @ W_gene^T + u @ W_in^T + b) - gamma * g`. The
        field is AUTONOMOUS (no explicit time) and `u` (the frozen driver snapshot) is held
        constant for the whole solve. `g` is [N, n_gene]; `u` is [N, in_size] (in_size may
        be 0). Materialises the parameters against the widths on first call."""
        self._ensure_params(g.shape[-1], u.shape[-1], g.device, g.dtype)
        W_gene, W_in, b, gamma = self._params
        drive = g @ W_gene.transpose(-2, -1) + b
        if W_in.shape[-1] > 0:
            drive = drive + u @ W_in.transpose(-2, -1)
        return _rescaled_sigmoid(drive) - gamma * g

    # --- self-solve over [0, dt] with fixed-step RK4 ---------------------------- #
    def _solve(self, g0: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        """Integrate `dg/dt = vector_field(g, u)` from 0 to dt and return g(dt). Classic
        RK4 with `substeps` fixed steps of h = dt/substeps; the field is autonomous, so the
        stage times never enter. Batched over all cells; `u` is closed over (frozen)."""
        h = dt / self.substeps
        g = g0
        for _ in range(self.substeps):
            k1 = self.vector_field(g, u)
            k2 = self.vector_field(g + 0.5 * h * k1, u)
            k3 = self.vector_field(g + 0.5 * h * k2, u)
            k4 = self.vector_field(g + h * k3, u)
            g = g + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return g

    def forward(self, H, mask=None):
        cell = H.level(self.at)
        g0 = cell.get(self.gene_block)                           # [N, n_gene]  y0, the ODE initial condition
        # pack the fixed sensed drivers u (per-cell state blocks), concatenated in order;
        # empty -> a [N, 0] tensor (autonomous circuit).
        if self.input_blocks:
            u = torch.cat([cell.get(nm) for nm in self.input_blocks], dim=-1)
        else:
            u = g0.new_zeros((g0.shape[0], 0))
        dt = float(getattr(getattr(H, "config", None), "dt", 1.0))
        g_end = self._solve(g0, u, dt)
        # the EXACT integrated increment g(dt) - g(0), expressed as the effective mean rate
        # so the engine's first-order step (g += dt*delta) recovers g(dt). The dt cancels.
        delta = (g_end - g0) / dt
        delta = delta * cell.occ[:, None]                        # dormant cells hold their gene state
        if mask is not None:
            delta = delta * mask[:, None].float()
        return {self.at: delta}
