"""regulate:mwc -- a per-cell gene-regulatory ODE with a thermodynamic (MWC) drive.

The `regulate` contract is a cell's INTERNAL regulatory dynamical system: a heritable
per-cell vector of gene concentrations `g` whose autonomous ODE is integrated over the
macro-step, reading the cell's own genes plus a FIXED snapshot of sensed driver inputs
`u` (chemistry / mechanics, already sensed onto the cell by an upstream `sense`/`chemotax`
step), and emitting the gene-expression change that division / secretion / adhesion steps
downstream read. It is the intracellular genotype->phenotype decision function.

This is the THERMODYNAMIC (Monod-Wyman-Changeux, statistical-mechanics log-occupancy)
IMPLEMENTATION of that contract. Each gene evolves as saturating production minus
first-order decay,

    dg_i/dt = rho_i * sigmoid(F_i)  -  g_i / tau_i ,

with a log-occupancy regulatory drive

    F_i = F0_i
          + sum_j H_gene_ij * ln(1 + g_j / K_gene_ij)      # gene -> gene occupancy
          + sum_k H_in_ik   * ln(1 + u_k / K_in_ik) .       # driver -> gene occupancy

The signed interaction weights `H` set activation(+)/inhibition(-); the positive
quantities rho (production ceiling), tau (lifetime), K (half-occupancy threshold) are
stored in LOG space and exponentiated within the dtype's finite range. This is the same
`regulate` signature as the connectionist (linear `W*g`) and neural-ODE (black-box MLP)
drives -- the three siblings differ ONLY in the vector field, so they collapse to one
contract with three interchangeable implementations, selected by `implementation:`.

Routing (the Plexus decomposition): like the `signal` connectome ODE, the operator returns
the first-order vector field `dg/dt` on the cell's `gene` state block (EMIT=velocity,
INTEGRAND=gene) and the ENGINE integrates it (x += dt*delta). The engine owns the
time-stepping; the operator owns only the biology (the drive law). Its sole difference
from the paper's connectionist form is that difference in `f`; per the record's source-wins
rule the MWC drive is code-only (the paper's eq. 4 states the linear form), and is recorded
here as the third implementation of the same biological role.

Sensed inputs `u` are held FIXED across the step (read, never written), matching the
reference's quasistatic-chemistry assumption; only the gene block evolves.

Translated from papers/jax-morph/jax_morph/control/ode.py:364-475 (GeneNetworkMWC /
ODEController). Torch, not JAX; no diffrax -- the engine's Euler step replaces the
reference's internal Dopri5 sub-solve (Axis A: integration is the engine's concern).
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


def _positive_from_log(log_value: torch.Tensor) -> torch.Tensor:
    """Exponentiate a log-parameter within its dtype's finite positive range.

    Mirrors the reference `_positive_from_log` (ode.py:38): clip the log to
    [log(tiny), log(max) - log(4)] before exp, so rho/tau/K stay strictly positive and
    finite with headroom (the `- log(4)` leaves room for subsequent arithmetic before
    overflow). Default log 0 -> exp(0) = 1, so an unset rho/tau/K is unity, not zero.
    """
    info = torch.finfo(log_value.dtype)
    min_log = math.log(float(info.tiny))
    max_log = math.log(float(info.max)) - math.log(4.0)
    return torch.exp(log_value.clamp(min_log, max_log))


@register_operator("regulate", family="fields", set="cell", kind="exchange",
                   implementation="mwc")
class RegulateMWC(Exchange):
    r"""MWC (thermodynamic log-occupancy) implementation of the `regulate` contract.

    dg/dt = rho * sigmoid(F0 + sum_j H_gene*ln(1+g_j/K_gene) + sum_k H_in*ln(1+u_k/K_in))
            - g / tau.
    Returns dg/dt on the cell's gene block; the engine integrates it first-order.
    """

    EMIT = "velocity"                    # first-order gene ODE (dg/dt); engine integrates the `gene` block
    INTEGRAND = "gene"                   # the delta targets the gene state block, NOT the spatial coordinate
    # typed signature (Plexus2 sec. 2.1): a per-cell morphism (cell, sensed drivers) -> cell,
    # reading the gene state + a fixed sensed-input snapshot, writing the gene derivative.
    # No maps and no cross-cell coupling: H_gene is a dense WITHIN-cell gene->gene matrix
    # applied independently per cell (the reference's vmap).
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["gene", "drive"]           # own gene concentrations + the fixed sensed driver block
    WRITES = ["gene"]                    # dg/dt on the gene block
    MAPS = []                            # intracellular: no edge-set, no gather/scatter
    SUPPORTED_DIMS = [2, 3]              # gene state is scalar-per-gene; spatial dimension is irrelevant
    DIFFERENTIABLE = True               # pure-torch vector field; grads flow through for the inverse problem
    REQUIRES_PARAMS = []                # every knob optional (all params default to zeros = an inert circuit)
    MECHANISM_TAGS = ["gene_regulatory_network", "thermodynamic_occupancy", "mwc",
                      "saturating_production", "recurrent", "intracellular_controller"]
    PARAM_ROLES = {
        "gene": "gene_state_block",
        "sensed": "fixed_driver_input_block",
        "hidden_size": "latent_regulator_count",
        "log_rho": "max_production_rate_log",
        "log_tau": "molecule_lifetime_log",
        "F0": "basal_activation_bias",
        "H_gene": "gene_coupling_strength_signed",
        "log_K_gene": "gene_binding_threshold_log",
        "H_in": "input_coupling_strength_signed",
        "log_K_in": "input_binding_threshold_log",
    }
    REFERENCE = (
        "Deshpande, Mottes, Vidal Saez, Kicheva & Hiscock (2025), 'Engineering "
        "morphogenesis of cell clusters with differentiable programming', Nat Comput Sci. "
        "Monod, Wyman & Changeux (1965), J. Mol. Biol. 12:88-118 (thermodynamic occupancy). "
        "Translated from papers/jax-morph/jax_morph/control/ode.py:364-475 (GeneNetworkMWC). "
        "NB: the MWC log-occupancy drive is code-only; the paper's eq. 4 states a LINEAR "
        "connectionist drive (see the `regulate` implementation `connectionist`)."
    )

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.gene_block = params.get("gene", params.get("block", "gene"))
        self.input_block = params.get("sensed", params.get("input"))   # None -> no sensed drivers (n_in = 0)
        self.hidden_size = int(params.get("hidden_size", 0))           # metadata: hidden|output split (math is uniform)
        self.at = params.get("_at", "cell")
        self.INTEGRAND = self.gene_block                               # instance override: route to the gene block
        # store the raw parameter values verbatim; materialize (and shape-check) against the
        # actual block widths on the first forward, like the reference `_resolve_param`.
        self._raw = {k: params.get(k) for k in
                     ("log_rho", "log_tau", "F0", "H_gene", "log_K_gene", "H_in", "log_K_in")}
        self._resolved = None            # cache: (n_gene, n_in) -> dict of tensors

    # --- parameter resolution (None -> zeros, else shape-checked) ------------- #
    def _param(self, key, shape, device, dtype):
        v = self._raw.get(key)
        if v is None:
            return torch.zeros(shape, device=device, dtype=dtype)
        t = torch.as_tensor(v, device=device, dtype=dtype)
        if tuple(t.shape) != tuple(shape):
            raise ValueError(
                f"regulate:mwc param {key!r} must have shape {tuple(shape)}, got {tuple(t.shape)}")
        return t

    def _resolve(self, n_gene, n_in, device, dtype):
        key = (n_gene, n_in, device, dtype)
        if self._resolved is not None and self._resolved["_key"] == key:
            return self._resolved
        r = {
            "_key": key,
            "log_rho":    self._param("log_rho",    (n_gene,),        device, dtype),
            "log_tau":    self._param("log_tau",    (n_gene,),        device, dtype),
            "F0":         self._param("F0",         (n_gene,),        device, dtype),
            "H_gene":     self._param("H_gene",     (n_gene, n_gene), device, dtype),
            "log_K_gene": self._param("log_K_gene", (n_gene, n_gene), device, dtype),
            "H_in":       self._param("H_in",       (n_gene, n_in),   device, dtype),
            "log_K_in":   self._param("log_K_in",   (n_gene, n_in),   device, dtype),
        }
        self._resolved = r
        return r

    # --- the vector field (ode.py:449-475), the biology of this implementation - #
    def vector_field(self, evolving: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """dg/dt for every cell, given the cell's genes `evolving` [N, n_gene] and its
        FIXED sensed drivers `inputs` [N, n_in]. Production uses the CLAMPED (>=0) genes
        and drivers inside the occupancy logs (ln(1+.) would NaN on negatives); the decay
        term uses the RAW evolving state -- an intentional asymmetry that makes decay
        restorative toward 0 even for a (transiently) negative concentration."""
        n_gene = evolving.shape[1]
        n_in = inputs.shape[1]
        device, dtype = evolving.device, evolving.dtype
        p = self._resolve(n_gene, n_in, device, dtype)

        genes = evolving.clamp(min=0.0)                          # [N, n_gene]  clamped for occupancy
        drivers = inputs.clamp(min=0.0)                          # [N, n_in]
        K_gene = _positive_from_log(p["log_K_gene"])             # [n_gene, n_gene]
        K_in = _positive_from_log(p["log_K_in"])                 # [n_gene, n_in]
        # overflow guard (ode.py:465-466): a tiny threshold makes g/K overflow to +inf; cap
        # the ratio at the finite dtype max so a mixed-sign interaction row cannot form
        # (+inf) + (-inf) = NaN.
        max_ratio = float(torch.finfo(dtype).max)
        gene_occ = torch.log1p((genes[:, None, :] / K_gene).clamp(max=max_ratio))   # [N, n_gene, n_gene]
        gene_drive = (p["H_gene"] * gene_occ).sum(dim=-1)                           # [N, n_gene]
        if n_in > 0:
            input_occ = torch.log1p((drivers[:, None, :] / K_in).clamp(max=max_ratio))  # [N, n_gene, n_in]
            input_drive = (p["H_in"] * input_occ).sum(dim=-1)                            # [N, n_gene]
        else:
            input_drive = torch.zeros_like(gene_drive)
        rho = _positive_from_log(p["log_rho"])                  # [n_gene]
        tau = _positive_from_log(p["log_tau"])                  # [n_gene]
        production = rho * torch.sigmoid(gene_drive + input_drive + p["F0"])   # [N, n_gene]  in (0, rho)
        return production - evolving / tau                      # RAW evolving in the decay

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        evolving = lvl.get(self.gene_block)                     # [N, n_gene]  y0, the ODE initial condition
        if self.input_block is not None and self.input_block in lvl.state_schema:
            inputs = lvl.get(self.input_block)                  # [N, n_in]  fixed sensed drivers u
        else:
            inputs = evolving.new_zeros(evolving.shape[0], 0)   # no sensed drivers
        dg = self.vector_field(evolving, inputs)
        dg = dg * lvl.occ[:, None]                              # dormant cells do not evolve
        if mask is not None:
            dg = dg * mask[:, None].float()
        return {self.at: dg}
