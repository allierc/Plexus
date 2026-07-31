"""morphogen:free_space_greens_function -- cell -> cell. The steady-state morphogen field.

The developmental MORPHOGEN gradient as a single quasistatic operator: map every cell's
per-species secretion rate `S` to the equilibrium concentration `c` of the secreted,
diffusible, degradable signal, sampled back at every cell. `c` solves the screened-diffusion
(modified Helmholtz) steady state

    D grad^2 c - K c + S = 0     (dc/dt = 0),

where `D` (`diffusion`) is the diffusion coefficient and `K` (`degradation`) the first-order
degradation / uptake rate. It is QUASISTATIC: `dt` is ignored and the field is OVERWRITTEN each
macro-step with its t=infinity solution -- a CONSTRAINT solve, not a time step. So it is a derived
readout of the `chemical` state block (like `aggregate`'s centroid), mutated in place; returns {}.

THIS IMPLEMENTATION -- free-space Green's functions (the source code). Each alive cell is a finite
source of radius `a = radius_j` (a sphere in 3-D, disk in 2-D, segment in 1-D) emitting at rate
`S_j`; the field superposes their unbounded, open-boundary screened-diffusion Green's functions,

    c_i = sum_j  alive_j * G_dim(r_ij, a_j) * S_j ,   then  c_i *= alive_i ,

a dense all-pairs cell -> cell distance kernel (`r_ij = |pos_i - pos_j|`, inverse screening length
`kappa = sqrt(K/D)`). The near / self field is regularized by clamping the receiver distance to the
source surface, `r_eff = max(r_ij, a_j)`, so the `i == j` diagonal contributes the on-surface value
-- a cell reads its OWN secretion. Dimension-selected kernels (segment / disk / sphere); the 2-D
disk needs the modified Bessel functions K0/K1 (ported here, differentiable, via the
Abramowitz-and-Stegun series so gradients flow for the inverse problem).

PAPER CONTRADICTION (source wins; recorded in the atlas `why`): the paper (Methods, p. 15) solves
the SAME steady PDE on the GRAPH LAPLACIAN of the cell-center lattice -- c = (K I - D L)^{-1} S with
explicit closed/permeable boundaries and a ghost sink node, no source radius. That graph-Laplacian
inverse is a SIBLING implementation of the same `morphogen` contract (open vs. bounded boundaries;
two numerical methods, one biological operator). This is the free-space form the code actually runs.

Translated from papers/jax-morph/jax_morph/physics/diffusion.py:92 (FreeScreenedDiffusion). Torch,
not JAX; the JAX `vmap` over species becomes a Python loop over the (small, static) species axis --
the biology (the analytic steady field) is identical.
"""
from __future__ import annotations

import math

import torch

from plexus.geometry import minimum_image
from plexus.models.base import Exchange
from plexus.models.registry import register_operator

_EPS = 1e-12                                            # radius / kappa floor: keep the kernel finite on dead slots


def _safe_norm(x, dim=-1):
    """Euclidean norm with value AND gradient zero at the zero vector (the reference
    `safe_norm`, core/ad_utils.py:26): the `i == j` diagonal has r = 0 exactly, and a bare
    `sqrt(sum sq)` there is finite but its gradient is NaN. Guard both with a `where`."""
    sq = (x * x).sum(dim=dim)
    safe = torch.where(sq == 0.0, torch.ones_like(sq), sq)
    return torch.where(sq == 0.0, torch.zeros_like(sq), torch.sqrt(safe))


def _k0(x):
    """Modified Bessel K0 (Abramowitz & Stegun 9.8.5 / 9.8.6); differentiable, finite for x > 0.
    Port of diffusion.py:21 using torch.special.i0 (itself differentiable)."""
    xs = torch.minimum(x, x.new_tensor(2.0))
    y = (xs / 2.0) ** 2
    k_small = -torch.log(xs / 2.0) * torch.special.i0(xs) + (
        -0.57721566
        + y * (0.42278420
        + y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740)))))
    )
    xl = torch.maximum(x, x.new_tensor(2.0))
    z = 2.0 / xl
    k_large = (
        torch.exp(-xl) / torch.sqrt(xl)
        * (1.25331414
        + z * (-0.07832358
        + z * (0.02189568 + z * (-0.01062446 + z * (0.00587872 + z * (-0.00251540 + z * 0.00053208))))))
    )
    return torch.where(x <= 2.0, k_small, k_large)


def _k1(x):
    """Modified Bessel K1 (Abramowitz & Stegun 9.8.7 / 9.8.8); differentiable, finite for x > 0.
    Port of diffusion.py:55 using torch.special.i1."""
    xs = torch.minimum(x, x.new_tensor(2.0))
    y = (xs / 2.0) ** 2
    k1_small = torch.log(xs / 2.0) * torch.special.i1(xs) + (1.0 / xs) * (
        1.0
        + y * (0.15443144
        + y * (-0.67278579 + y * (-0.18156897 + y * (-0.01919402 + y * (-0.00110404 + y * -0.00004686)))))
    )
    xl = torch.maximum(x, x.new_tensor(2.0))
    z = 2.0 / xl
    k1_large = (
        torch.exp(-xl) / torch.sqrt(xl)
        * (1.25331414
        + z * (0.23498619
        + z * (-0.03655620 + z * (0.01504268 + z * (-0.00780353 + z * (0.00325614 + z * -0.00068245))))))
    )
    return torch.where(x <= 2.0, k1_small, k1_large)


@register_operator("morphogen", family="fields", set="cell", kind="exchange",
                   implementation="free_space_greens_function")
class MorphogenFreeSpace(Exchange):
    r"""Free-space Green's-function implementation of the `morphogen` contract.

    Overwrites the per-cell `chemical` block with the superposed steady screened-diffusion field
    c_i = sum_j alive_j G(r_ij, a_j) S_j (then zeroed on dead receivers). Quasistatic derived
    readout: EMIT=None, mutates state in place, returns {}."""

    EMIT = None                                 # quasistatic OVERWRITE of the `chemical` block; returns {} — no integrable delta
    MAY_MUTATE_INTEGRATED_STATE = True          # derived readout: writes a state block in place (like aggregate's centroid)
    # typed signature (Plexus2 sec. 2.1): a per-cell morphism cell -> cell over a DENSE all-pairs
    # cell->cell distance kernel. Reads the source rate + geometry (pos/radius/alive); writes the
    # concentration block. The reference under-declares its reads (only the source field); the
    # contract lists every input the field actually depends on.
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["secretion_rate", "pos", "radius", "alive"]
    WRITES = ["chemical"]
    MAPS = ["pairwise"]                         # dense all-pairs superposition (no edge set, no grid)
    SUPPORTED_DIMS = [1, 2, 3]                  # dimension-selected kernel (segment / disk / sphere); 2-D/3-D are the Plexus worlds
    DIFFERENTIABLE = True                       # pure-torch (incl. the ported Bessel series); grads flow for the inverse problem
    REQUIRES_PARAMS = []                        # every knob optional (all default: D=K=1, secretion_rate -> chemical)
    MECHANISM_TAGS = ["morphogen_gradient", "screened_diffusion", "modified_helmholtz",
                      "steady_state_field", "greens_function", "reaction_diffusion_equilibrium"]
    PARAM_ROLES = {
        "diffusion": "diffusion_coefficient",
        "degradation": "degradation_uptake_rate",
        "source_name": "source_rate_field_plumbing",
        "field_name": "output_concentration_field_plumbing",
        "radius": "per_source_finite_size_or_uniform_default",
        "n_space_dim": "kernel_dimension_selector_assert",
    }
    REFERENCE = (
        "Deshpande, Mottes, Vidal Saez, Kicheva & Hiscock (2025), 'Engineering morphogenesis of "
        "cell clusters with differentiable programming', Nat Comput Sci (steady screened diffusion "
        "D grad^2 c - K c + S = 0, Methods p. 15). Translated from "
        "papers/jax-morph/jax_morph/physics/diffusion.py:92 (FreeScreenedDiffusion). "
        "NB (source wins): the paper solves this on the graph Laplacian c = (K I - D L)^-1 S with "
        "explicit boundaries; this is the code's FREE-SPACE Green's-function form (a sibling "
        "implementation of the same contract). Bessel series: Abramowitz & Stegun 9.8.5-9.8.8."
    )

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.source_name = params.get("source_name", params.get("source", "secretion_rate"))
        self.field_name = params.get("field_name", params.get("field", "chemical"))
        self.diffusion = params.get("diffusion", 1.0)          # D_k: scalar or per-species (n_species,)
        self.degradation = params.get("degradation", 1.0)      # K_k: scalar or per-species
        # per-cell source radius `a`: read the named block/buffer if the cell carries it, else fall
        # back to a uniform default (the engine's spawn `radius` default). The reference always has
        # state.radius; the fallback lets a cell set without an explicit radius still run.
        self.radius_name = params.get("radius_name", "radius")
        self.default_radius = float(params.get("radius", 0.02))
        # optional static dim assert (the reference RAISES on mismatch): keep the surprise reproducible.
        nsd = params.get("n_space_dim", None)
        self.n_space_dim = None if nsd is None else int(nsd)

    # --- per-source finite radius `a` (block, else buffer, else uniform default) ---------------- #
    def _radii(self, lvl, N, dev, dtype):
        name = self.radius_name
        if name in lvl.state_schema:
            return lvl.get(name)[:, 0].to(dtype)
        buf = getattr(lvl, name, None)
        if torch.is_tensor(buf) and buf.shape[0] == N:
            return buf.reshape(N).to(dtype)
        return torch.full((N,), self.default_radius, device=dev, dtype=dtype)

    # --- per-cell source rate S (a state block; the field it produces is written back) ---------- #
    def _source(self, lvl):
        if self.source_name in lvl.state_schema:
            return lvl.get(self.source_name)
        buf = getattr(lvl, self.source_name, None)
        if torch.is_tensor(buf):
            return buf
        raise ValueError(
            f"morphogen: source field {self.source_name!r} is not a state block or buffer on set "
            f"{self.at!r}; declare it (the per-cell secretion rate S this step reads).")

    def _kernel(self, r, a, kappa, D, dim):
        """Screened Green's function of a finite source of radius `a`, clamped to its surface
        (r_eff = max(r, a)). `kappa` and `D` are per-species scalars; `r`/`a` are [N, N] (a is a
        broadcast source-row). Faithful port of diffusion.py:191 (`_kernel`)."""
        a = a.clamp(min=_EPS)                              # dead cell a=0 would make 1/r_eff or K1(0) singular
        r_eff = torch.maximum(r, a)
        if dim == 1:                                       # 1-D finite segment
            kappa = kappa.clamp(min=_EPS)
            return torch.exp(-kappa * (r_eff - a)) / (2.0 * D * kappa)
        if dim == 3:                                       # 3-D finite sphere (bounded at kappa=0; no floor)
            return torch.exp(-kappa * (r_eff - a)) / (4.0 * math.pi * D * r_eff * (1.0 + kappa * a))
        # 2-D finite disk (needs the modified Bessel K0/K1)
        kappa = kappa.clamp(min=_EPS)
        return _k0(kappa * r_eff) / (2.0 * math.pi * D * a * kappa * _k1(kappa * a))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")                               # [N, D]
        N, D = pos.shape[0], pos.shape[1]
        dev, dtype = pos.device, pos.dtype
        if self.n_space_dim is not None and self.n_space_dim != D:
            raise ValueError(
                f"morphogen was built with n_space_dim={self.n_space_dim} (it selects the "
                f"segment/disk/sphere kernel) but the set has spatial dimension {D}; they must match.")

        # dense pairwise center-to-center distances r_ij = |pos_i - pos_j| (minimum-image if the
        # world is periodic -- note: free-space kernel + periodic box is a modeling error the user owns).
        disp = pos[:, None, :] - pos[None, :, :]           # [N, N, D]  disp[i, j] = pos_i - pos_j
        disp = minimum_image(disp, getattr(H, "periodic", False),
                             getattr(H, "world_size", getattr(H, "world_width", 1.0)))
        r = _safe_norm(disp, dim=-1)                       # [N, N]

        a_src = self._radii(lvl, N, dev, dtype)[None, :]   # [1, N] SOURCE radii (broadcast over receiver rows i)
        occ = lvl.occ.to(dtype)                            # [N] liveness (the reference `alive`)
        S = self._source(lvl)                              # [N, n_species]
        n_species = S.shape[1]
        Dc = torch.as_tensor(self.diffusion, dtype=dtype, device=dev).broadcast_to((n_species,))
        Kc = torch.as_tensor(self.degradation, dtype=dtype, device=dev).broadcast_to((n_species,))
        if D in (1, 2) and bool((Kc <= 0).any()):
            raise ValueError(
                f"morphogen in {D}-D needs degradation > 0 (kappa = sqrt(K/D); the unscreened "
                f"low-dimension field is unbounded).")

        cols = []
        for c in range(n_species):
            kappa = torch.sqrt(Kc[c] / Dc[c])
            G = self._kernel(r, a_src, kappa, Dc[c], D) * occ[None, :]   # mask SOURCES (columns j)
            cols.append(G @ S[:, c])                                     # [N] superposition over live sources
        field = torch.stack(cols, dim=1) * occ[:, None]    # [N, n_species]; mask RECEIVERS (rows i)

        # quasistatic OVERWRITE of the `chemical` block (a derived readout). `mask` (the `at:`
        # selector) gates which receivers take the fresh field; unselected LIVE cells keep their
        # previous value, so a subset selector does not stomp the rest.
        if self.field_name not in lvl.state_schema:
            raise ValueError(
                f"morphogen: output field {self.field_name!r} is not a state block on set "
                f"{self.at!r}; declare it (the per-cell concentration this step writes).")
        c0, c1 = lvl.state_schema[self.field_name]
        new = lvl.state.clone()
        if mask is not None:
            keep = (~mask.to(torch.bool)) & (lvl.occ > 0)               # live but not selected -> keep old
            new[:, c0:c1] = torch.where(keep[:, None], lvl.state[:, c0:c1], field)
        else:
            new[:, c0:c1] = field
        lvl.state = new
        return {}
