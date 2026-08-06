"""turing_ops -- discrete Turing (reaction--diffusion) on a cell aggregate (2D or 3D).

Plexus2 prototype of Okuda et al. 2018, "Combining Turing and 3D vertex models"
(Sci. Rep. 8:2386): the SIGNALLING half, on a cell aggregate. Each cell is one node
carrying two morphogens; molecules flux between neighbouring cells through junctions
(the discrete Turing model, Eq. 2) as a graph Laplacian over the cell adjacency, and
react by an activator/autocatalyst kinetics. We build up 2D first (Turing -> vertex ->
coupled), then reuse the same operators in 3D (the paper's Fig. 3 aggregates).

Fully plexus2: operators are typed morphisms (INPUTS/OUTPUTS/READS/WRITES/MAPS) with
capabilities (SUPPORTED_DIMS, DIFFERENTIABLE); `react` is one CONTRACT with three
interchangeable IMPLEMENTATIONS (gray_scott, gierer_meinhardt, brusselator) chosen by
`implementation:` in the spec.

  cell state:  chem = [activator, substrate/inhibitor]   (first-order integrand)
               xyz  = cell centre (2 or 3 wide)           (frozen; adjacency + render)
  cell map:    edge_index -- the cell--cell adjacency (the diffusion graph)
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Lateral, Structural
from plexus.models.registry import register_operator


# --------------------------------------------------------------------------- #
#  Seed: place cells (2D disc / 3D shell / 3D ball), seed the pattern, build adjacency
# --------------------------------------------------------------------------- #
@register_operator("seed_aggregate", set="cell", kind="seed", family="growth")
class AggregateSeed(Structural):
    """Frame-0 IC (gate with `before_frame: 1`): place N cells in an aggregate, seed
    the morphogens (substrate full, a small central activator patch + noise -> spots
    nucleate), and build the cell--cell adjacency graph (symmetric kNN) the discrete
    Turing diffusion runs on. Modes: `disc` (2D) | `shell`,`ball` (3D)."""
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = False                       # a discrete IC, not part of the differentiable rollout
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["mode"]
    MECHANISM_TAGS = ["cell_aggregate", "initial_condition", "adjacency_graph"]
    PARAM_ROLES = {"mode": "disc|shell|ball", "radius": "aggregate_radius", "k": "n_neighbours",
                   "seed_frac": "activator_patch_fraction"}
    REFERENCE = "Plexus (this work); cf. Okuda, S. et al. (2018). Sci. Rep. 8:2386 (cell aggregate initial condition)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.mode = params.get("mode", "disc")
        self.seed_mode = params.get("seed_mode", "patch")       # patch (one central nucleus) | scatter (many nuclei)
        self.radius = float(params.get("radius", 6.0))
        self.k = int(params.get("k", 6))
        self.seed_frac = float(params.get("seed_frac", 0.06))   # patch: radius fraction | scatter: fraction of cells
        self.noise = float(params.get("noise", 0.02))
        self.a0 = float(params.get("a0", 1.0))                  # noise-seed: activator homogeneous state
        self.h0 = float(params.get("h0", 3.0))                  # noise-seed: inhibitor homogeneous state

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        N = lvl.state.shape[0]
        dev = lvl.state.device
        Dsp = H.dim
        c = 0.5 * H.world_size[:Dsp].to(dev)
        g = torch.Generator(device="cpu"); g.manual_seed(0)
        if self.mode == "disc":                              # 2D: even disc (sunflower)
            idx = torch.arange(N).float() + 0.5
            r = torch.sqrt(idx / N) * self.radius
            th = math.pi * (1.0 + 5.0 ** 0.5) * idx
            pos = torch.stack([c[0] + r * torch.cos(th), c[1] + r * torch.sin(th)], 1).to(dev)
        elif self.mode == "shell":                           # 3D: Fibonacci sphere (monolayer)
            k = torch.arange(N).float()
            phi = math.pi * (3.0 - math.sqrt(5.0))
            z = 1.0 - 2.0 * (k + 0.5) / N
            rho = torch.sqrt((1.0 - z * z).clamp(min=0))
            dirs = torch.stack([rho * torch.cos(phi * k), rho * torch.sin(phi * k), z], 1)
            pos = (c + dirs.to(dev) * self.radius)
        else:                                                # 3D: solid ball (compacted aggregate)
            d = torch.randn(N, 3, generator=g)
            d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
            rr = torch.rand(N, generator=g).pow(1.0 / 3.0) * self.radius
            pos = (c + (d * rr[:, None]).to(dev))

        # chem = [activator, inhibitor/substrate]
        if self.seed_mode == "noise":                       # activator-inhibitor (Brusselator/GM): steady state + noise
            v = self.a0 + self.noise * torch.randn(N, generator=g).to(dev)
            u = self.h0 + self.noise * torch.randn(N, generator=g).to(dev)
        else:                                               # Gray-Scott: full substrate + activator nuclei
            v = 0.02 * torch.rand(N, generator=g).to(dev)
            u = torch.ones(N, device=dev)
            if self.seed_mode == "scatter":                 # many random nuclei -> discrete fronts
                nuc = torch.rand(N, generator=g).to(dev) < self.seed_frac
                v[nuc] = 0.5
            else:                                           # one central patch -> a spreading front
                patch = (pos - c).norm(dim=1) < self.seed_frac * self.radius
                v[patch] = 0.5
            v = (v + self.noise * torch.randn(N, generator=g).to(dev)).clamp(min=0.0)

        xa, xb = lvl.state_schema["xyz"]; ca, _ = lvl.state_schema["chem"]
        st = lvl.state.clone()
        st[:, xa:xb] = pos
        st[:, ca] = v; st[:, ca + 1] = u
        lvl.state = st

        # symmetric kNN adjacency on the (static) positions = the diffusion graph
        D = torch.cdist(pos, pos); D.fill_diagonal_(float("inf"))
        nn = D.topk(self.k, largest=False).indices
        i = torch.arange(N, device=dev)[:, None].expand(-1, self.k).reshape(-1)
        j = nn.reshape(-1)
        ei = torch.stack([torch.cat([i, j]), torch.cat([j, i])])
        lvl.edge_index = torch.unique(ei, dim=1)
        return {}


# --------------------------------------------------------------------------- #
#  Diffusion: a graph Laplacian over the cell adjacency (the discrete transport)
# --------------------------------------------------------------------------- #
@register_operator("graph_diffuse", set="cell", kind="lateral", family="fields")
class CellDiffuse(Lateral):
    """Discrete diffusion of the two morphogens between neighbouring cells: a graph
    Laplacian on the cell adjacency (Okuda Eq. 2, junctional flux). Per-species
    coefficients (`d_a` activator, `d_h` substrate) give the Turing separation;
    `chi` scales the diffusion length -> the spatial scale of the pattern.
    First-order in the chemistry, so EMIT=velocity."""
    SUPPORTED_DIMS = [2, 3]
    EMIT = "velocity"
    INTEGRAND = "chem"                                        # coupled: -> the chem block (pure-Turing: chem IS the coordinate)
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["d_a", "d_h", "chi"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["chem"]; WRITES = ["chem"]
    MAPS = ["edge_index"]                                    # the cell--cell adjacency relation
    MECHANISM_TAGS = ["diffusion", "graph_laplacian", "junctional_transport", "turing"]
    PARAM_ROLES = {"d_a": "activator_diffusivity", "d_h": "substrate_diffusivity", "chi": "spatial_scale"}
    REFERENCE = ("Okuda, S. et al. (2018). Combining Turing and 3D vertex models... Sci. Rep. 8:2386 (Eq. 2, "
                 "junctional transport); Turing, A. M. (1952). Phil. Trans. R. Soc. B 237:37-72.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.d_a = float(params["d_a"]); self.d_h = float(params["d_h"]); self.chi = float(params["chi"])
        self.norm = bool(params.get("norm", False))          # degree-normalise the Laplacian (stable at any degree)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")                               # [N, 2] = (activator, inhibitor)
        ei = lvl.edge_index; i, j = ei[0], ei[1]
        N = chem.shape[0]
        agg = torch.zeros_like(chem).index_add_(0, i, chem[j])
        deg = torch.zeros(N, device=chem.device).index_add_(0, i, torch.ones_like(i, dtype=chem.dtype))
        if self.norm:                                        # neighbour-MEAN minus self: eigenvalues in [-2,0]
            lap = agg / deg.clamp(min=1)[:, None] - chem     #   -> stability independent of neighbour count
        else:                                                # raw sum_j (chem_j - chem_i): degree-weighted
            lap = agg - deg[:, None] * chem
        coef = torch.tensor([self.d_a, self.d_h], device=chem.device) * self.chi
        return {self.at: (coef[None, :] * lap) * lvl.occ[:, None]}


# --------------------------------------------------------------------------- #
#  Reaction: one contract, two interchangeable kinetics
# --------------------------------------------------------------------------- #
@register_operator("react", set="cell", kind="lateral", family="fields",
                   implementation="gray_scott")
class GrayScott(Lateral):
    """`gray_scott` implementation of the `react` contract: activator/substrate
    autocatalysis (a robust Turing spot-former; the autocatalyst is the activator).

        dv/dt =  u v^2 - (F + kk) v        (v = activator / autocatalyst)
        du/dt = -u v^2 + F (1 - u)         (u = substrate)
    """
    SUPPORTED_DIMS = [2, 3]
    EMIT = "velocity"
    INTEGRAND = "chem"
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["F", "kk"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["chem"]; WRITES = ["chem"]
    MAPS = []                                                # local (no map traversed)
    MECHANISM_TAGS = ["reaction", "autocatalysis", "turing", "gray_scott", "self_replication"]
    PARAM_ROLES = {"F": "feed_rate", "kk": "kill_rate", "rate": "reaction_time_scale"}
    REFERENCE = ("Pearson, J. E. (1993). Complex patterns in a simple system. Science 261:189-192; "
                 "Gray, P. & Scott, S. K. (1984). Chem. Eng. Sci. 39:1087-1097.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.F = float(params["F"]); self.kk = float(params["kk"])
        self.rate = float(params.get("rate", 1.0))          # time-scale the WHOLE RD (with diffuse chi) so the
        #                                                     pattern develops at a fine mechanics dt -- pure time
        #                                                     rescale, identical pattern (Turing/vertex separation)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        v = chem[:, 0]; u = chem[:, 1]
        uvv = u * v * v
        dv = uvv - (self.F + self.kk) * v
        du = -uvv + self.F * (1.0 - u)
        return {self.at: self.rate * torch.stack([dv, du], dim=1) * lvl.occ[:, None]}


@register_operator("react", set="cell", kind="lateral", family="fields",
                   implementation="gierer_meinhardt")
class GiererMeinhardt(Lateral):
    """`gierer_meinhardt` implementation of the same `react` contract: the
    classic activator--inhibitor kinetics with saturation (matches the paper's
    activator--inhibitor language).

        da/dt = rho ( a^2 / (h (1 + kappa a^2)) - a ) + rho0
        dh/dt = rho ( a^2 - h )
    """
    SUPPORTED_DIMS = [2, 3]
    EMIT = "velocity"
    INTEGRAND = "chem"
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["rho"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["chem"]; WRITES = ["chem"]
    MAPS = []
    MECHANISM_TAGS = ["reaction", "activator_inhibitor", "turing", "gierer_meinhardt"]
    PARAM_ROLES = {"rho": "reaction_rate", "kappa": "saturation", "rho0": "activator_baseline"}
    REFERENCE = "Gierer, A. & Meinhardt, H. (1972). A theory of biological pattern formation. Kybernetik 12:30-39."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.rho = float(params["rho"]); self.kappa = float(params.get("kappa", 0.1))
        self.rho0 = float(params.get("rho0", 0.01))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        a = chem[:, 0].clamp(min=1e-4); h = chem[:, 1].clamp(min=1e-4)
        a2 = a * a
        da = self.rho * (a2 / (h * (1.0 + self.kappa * a2)) - a) + self.rho0
        dh = self.rho * (a2 - h)
        return {self.at: torch.stack([da, dh], dim=1) * lvl.occ[:, None]}


@register_operator("react", set="cell", kind="lateral", family="fields",
                   implementation="brusselator")
class Brusselator(Lateral):
    """`brusselator` implementation of the `react` contract: the classic
    activator--inhibitor Turing kinetics. Non-stiff (polynomial), so it patterns
    robustly from noise into ROUND spots (unlike Gray-Scott's stripes/labyrinths) --
    the right kinetics for the paper's Fig. 3 activator spots. Seed with
    `seed_mode: noise`, `a0=A`, `h0=B/A`.

        da/dt = gamma ( A - (B+1) a + a^2 h )
        dh/dt = gamma ( B a - a^2 h )
    Turing-unstable for inhibitor faster than activator (d_h >> d_a) and B > 1 + A^2.
    """
    SUPPORTED_DIMS = [2, 3]
    EMIT = "velocity"
    INTEGRAND = "chem"
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["gamma", "A", "B"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["chem"]; WRITES = ["chem"]
    MAPS = []
    MECHANISM_TAGS = ["reaction", "activator_inhibitor", "turing", "brusselator", "round_spots"]
    PARAM_ROLES = {"gamma": "reaction_rate", "A": "feed", "B": "conversion"}
    REFERENCE = "Prigogine, I. & Lefever, R. (1968). Symmetry breaking instabilities in dissipative systems. II. J. Chem. Phys. 48:1695-1700."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.gamma = float(params["gamma"]); self.A = float(params["A"]); self.B = float(params["B"])

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        chem = lvl.get("chem")
        a = chem[:, 0]; h = chem[:, 1]
        a2h = a * a * h
        da = self.gamma * (self.A - (self.B + 1.0) * a + a2h)
        dh = self.gamma * (self.B * a - a2h)
        return {self.at: torch.stack([da, dh], dim=1) * lvl.occ[:, None]}
