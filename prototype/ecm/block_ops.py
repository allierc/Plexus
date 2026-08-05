"""block_ops -- the two blocks as an ELASTIC MATERIAL rather than a rigid surface.

WHY THIS EXISTS. `plate_confine_3d` is a projection: the plate is infinitely rigid, so it cannot be
seen to do anything. It holds the tissue and the matrix out of a half-space and that is the whole of
its behaviour -- no dent where the tissue presses hardest, no bulge beside it, no way to tell a stiff
block from an immovable one. Asked to SEE the block deform, the honest answer is that a projection has
nothing to show, and the block has to become a material.

SO IT BECOMES ONE, THROUGH THE SAME GRID. The blocks are a second MPM particle set with a much larger
Young's modulus, scattering into the SAME background grid as the matrix -- the pattern
`prototype/eye/` already runs for the globe and its muscles, and `mpm_scatter[accumulate]` is that
prototype's operator, imported rather than rewritten. The coupling is then momentum exchange on a
shared grid, not a contact model anyone had to invent: the tissue squeezes the matrix, the matrix
presses on the block, and the block deforms by as much as its stiffness allows. Nothing holds the
blocks in place except the domain wall behind them, which is the correct boundary condition for a slab
resting against the outside of the box.

WHAT SETS THE STIFFNESS RATIO, AND WHAT IT COSTS. MPM is explicit: dt < dx / sqrt(E/rho). At
dx = 1/48 and dt_sub = 2e-4 the ceiling is E ~ 10,000, so a block at E = 2,000 against a matrix at
E = 15 is a ratio of 130 -- stiff enough to read as solid, soft enough to visibly dent, and inside the
stable range without shrinking the substep. Raising it to a truly rigid 10^5 would need a substep 3x
smaller and a run 3x longer to show LESS.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral, Structural
from plexus.models.entities import MPMParticle
from plexus.models.registry import register_entity, register_operator

# The block's own strain, per frame, kept separately from the matrix's. `ecm_ops.STRESS_HISTORY` is a
# single module-level list, so a second `ecm_stress` instance would interleave two sets' rows into it
# and the renderer would colour each set with the other's numbers -- silently, and looking plausible.
BLOCK_STRESS: list = []


# --------------------------------------------------------------------------- the entity
@register_entity(
    "mpm_block", depth=0,
    state_schema={"pos": (0, 2), "vel": (2, 4)},
    render={"color_by": "node_type", "arrows": None},
)
class MPMBlock:
    """A material point of a solid block. Identical continuum state to the matrix's `mpm_particle`
    (F, C, mass, Lame mu/la, p_vol); the stock provision allocates it.

    THE ENTITY IS RESOLVED BY SET NAME, which is the thing that has to be known and is easy to miss:
    `_entity_meta` looks the SET's name up in the entity registry and falls back to a bare pos/vel
    schema for anything unregistered. So a set called `mpm_block` with every MPM operator pointed at it
    still has no `F`, and the run dies in `mpm_strain` with `'Level' object has no attribute 'F'` --
    which reads like a bug in the operator and is a missing registration. `prototype/eye`'s
    `MuscleParticle` is the same three lines for the same reason.
    """
    provision = MPMParticle.provision


@register_operator("block_seed", family="growth", set="particle", kind="structural")
class BlockSeed(Structural):
    """Fill the two slabs beyond `gap_half` with particles, once, at frame 0."""

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["gap_half"]
    MECHANISM_TAGS = ["solid_obstacle", "material_seeding"]
    PARAM_ROLES = {"gap_half": "free_half_gap", "axis": "confined_axis",
                   "centre": "domain_centre", "margin": "wall_clearance"}
    REFERENCE = "Sulsky, D. et al. (1994) Comput. Methods Appl. Mech. Eng. 118:179 (MPM)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_block")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.axis = int(params.get("axis", 2))
        self.gap_half = float(params["gap_half"])
        # A WALL CLEARANCE, not a cosmetic margin. `mpm_gather` treats the outer `wall_contact` shell
        # specially and a particle seeded exactly on the boundary starts inside that shell, so the
        # block's outermost layer would begin the run in a contact correction it never leaves.
        self.margin = float(params.get("margin", 0.012))
        self.seed = int(params.get("seed", 0))
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        n, D = pos.shape
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        ax, c = self.axis, self.centre[self.axis]
        lo, hi = self.margin, 1.0 - self.margin

        # A JITTERED LATTICE, NOT UNIFORM NOISE. MPM resolves a material by how evenly its particles
        # fill the cells: uniform random sampling leaves holes and clumps at this density, and a clump
        # is a stiffness the material was not given. One particle per lattice site plus a fraction of
        # the spacing is what the stock block provisions do.
        half = n // 2
        per = []
        for sgn in (+1.0, -1.0):
            t0, t1 = c + sgn * self.gap_half, c + sgn * (0.5 - self.margin)
            th = abs(t1 - t0)
            # cube-root split of the slab's aspect ratio, so the lattice is roughly isotropic
            k = max(1, int(round((half * th * th / max((hi - lo) ** 2, 1e-9)) ** (1.0 / 3.0))))
            nx = max(1, int(round((half / max(k, 1)) ** 0.5)))
            grid = torch.stack(torch.meshgrid(
                torch.linspace(lo, hi, nx), torch.linspace(lo, hi, nx),
                torch.linspace(min(t0, t1), max(t0, t1), k), indexing="ij"), -1).reshape(-1, 3)
            per.append(grid)
        p = torch.cat(per)
        if p.shape[0] < n:
            p = torch.cat([p, p[: n - p.shape[0]]])
        p = p[:n]
        # reorder the columns so column `ax` is the slab-normal one
        if ax != 2:
            idx = [0, 1, 2]; idx[2], idx[ax] = idx[ax], idx[2]
            p = p[:, idx]
        spacing = (hi - lo) / max(int(round(n ** (1.0 / 3.0))), 1)
        p = p + (torch.rand(p.shape, generator=g) - 0.5) * spacing * 0.35
        p[:, ax] = p[:, ax].clamp(min(lo, c - 0.5 + self.margin), max(hi, c + 0.5 - self.margin))
        p = p.clamp(lo, hi)
        # anything that landed in the free gap is pushed back into its own slab
        d = p[:, ax] - c
        bad = d.abs() < self.gap_half
        if bad.any():
            p[bad, ax] = c + torch.sign(torch.where(d[bad] == 0, torch.ones_like(d[bad]), d[bad])) \
                * (self.gap_half + spacing * 0.5)
        lvl.get("pos")[:] = p.to(pos.device, pos.dtype)
        self._done = True
        print(f"[block_seed] {n} particles in two slabs beyond +/-{self.gap_half:.4g} of "
              f"{c:.3g} on axis {ax} ({100 * (1 - 2 * self.gap_half):.0f}% of the box)", flush=True)
        return {}


@register_operator("block_stress", family="hierarchy", set="particle", kind="lateral")
class BlockStress(Lateral):
    """The block's own |J-1|, banded, recorded per frame -- so its deformation is visible."""

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diagnostic", "strain_visualisation"]
    PARAM_ROLES = {"scale": "strain_full_scale", "bands": "colour_bands"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_block")
        # A DIFFERENT FULL SCALE FROM THE MATRIX'S, and it has to be: the block is ~130x stiffer, so
        # the same load strains it ~130x less. At the matrix's 0.08 the block would read as
        # uniformly unstrained in every frame, which is the claim "it is rigid" -- the claim this
        # operator exists to test.
        self.scale = float(params.get("scale", 0.004))
        self.bands = int(params.get("bands", 8))
        # Same three options as `ecm_stress`; `vonmises` reads the Cauchy stress the accumulate scatter
        # cached (`store_stress: true`) rather than re-deriving anything from F.
        self.measure = str(params.get("measure", "vol"))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        F = getattr(lvl, "F", None)
        if F is None:
            return {}
        sig = getattr(lvl, "sigma", None) if self.measure == "vonmises" else None
        if sig is not None:
            tr = sig.diagonal(dim1=-2, dim2=-1).sum(-1)
            eye = torch.eye(sig.shape[-1], device=sig.device, dtype=sig.dtype)
            dv = sig - (tr / 3.0)[:, None, None] * eye
            s = torch.sqrt((1.5 * (dv * dv).sum((-1, -2))).clamp_min(0.0)) / max(self.scale, 1e-9)
        else:
            if self.measure == "vonmises" and not getattr(self, "_warned", False):
                print("[block_stress] measure=vonmises but no `sigma` buffer -- falling back to "
                      "|J-1|, a DIFFERENT quantity", flush=True)
                self._warned = True
            J = torch.linalg.det(F)
            s = (J - 1.0).abs() / max(self.scale, 1e-9)
        band = (s.clamp(0, 1) * (self.bands - 1)).round().long()
        BLOCK_STRESS.append(band.detach().to("cpu", torch.uint8).numpy())
        return {}
