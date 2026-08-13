"""plate_ops -- two rigid solid blocks, top and bottom, that the growing tissue cannot get past.

THE QUESTION THIS ANSWERS. A vesicle growing in an unconfined matrix stays a sphere: the matrix in
these runs is soft, the coupling is one-way, and nothing breaks the symmetry. Put a SOLID block above
and below it and the symmetry is broken by geometry -- the cells' volume has to go somewhere, and the
only directions left are the two free ones. So the sphere becomes an oblate ovoid, and the aspect
ratio is a prediction the run either meets or does not.

WHY A RIGID BLOCK AND NOT MORE MATRIX. A block is the one obstacle whose answer is known before the
run: it does not deform, so whatever shape the tissue takes is the TISSUE's mechanics responding to a
fixed boundary, not two soft materials negotiating. That makes it the right first experiment. The
fibre-mediated version -- the matrix itself resisting growth, which is what `ecm_load` does -- is
the interesting one, and it has to be read against this one to mean anything.

ONE OPERATOR, TWO LEVELS. `set=` in the registry is an enumeration tag, not a constraint, so this same
operator confines the VERTEX mesh in the tissue's own 50-unit world (`centre: 0`) and the MATRIX
particles in the unit MPM box (`centre: 0.5`). The blocks are one physical object; two instances of one
operator is the honest way to say so. Writing two operators would let the two descriptions of the same
slab drift apart.

A CLAMP, NOT A FORCE, AND THAT IS A CHOICE WITH A COST. `cell_mechanics` owns the vertex force loop
and there is no term to add a wall to from outside it, so the constraint is applied by moving
violating vertices back after the relaxation -- a projection. `stiff < 1` moves them back only
partway, which is a stiff spring rather than an infinitely hard wall, and it matters: at `stiff = 1`
every vertex touching a plate is at EXACTLY the same coordinate, the wedges there are coplanar, and
coplanar wedges have zero volume, which is a degenerate cell the AVM will report as broken. Partial
projection leaves a contact layer with thickness.
"""
from __future__ import annotations

import torch

from plexus.models.base import Structural
from plexus.models.registry import register_operator

# WHO IS TOUCHING THE PLATES, PER FRAME. Same reason `ecm_ops.STRESS_HISTORY` exists: the recorder
# keeps state, not per-frame diagnostics, and "the tissue reached the plate at frame N" is the event
# the elongation is supposed to start from. Without it, the aspect ratio is a curve with no cause on
# it. Cleared per run by whoever runs the run.
PLATE_CONTACT: list = []


@register_operator("plate_confine", family="mechanics", set="vertex", kind="structural")
class PlateConfine3D(Structural):
    """Confine a set between two rigid plates normal to `axis`, at `centre` +/- `gap_half`."""

    EMIT = None                        # moves positions in place; no integrable delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["gap_half"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["rigid_confinement", "anisotropic_boundary", "solid_obstacle"]
    PARAM_ROLES = {"gap_half": "free_half_gap", "stiff": "projection_fraction",
                   "axis": "confined_axis", "centre": "gap_centre_on_axis"}
    REFERENCE = "Plexus (this work); the confinement geometry of Okuda, S. et al. (2018) Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.axis = int(params.get("axis", 2))
        self.centre = float(params.get("centre", 0.0))
        self.gap_half = float(params["gap_half"])
        self.stiff = float(params.get("stiff", 0.6))
        self.damp_normal = bool(params.get("damp_normal", True))
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        ax = self.axis
        # LIVE ENTITIES ONLY, when the level knows how many it has. A vertex buffer is mostly empty
        # slots; projecting them is harmless but it makes the contact COUNT meaningless, and that
        # count is the only evidence of when confinement began.
        m = getattr(lvl, "_mesh", None)
        n = int(m["Nv"]) if (m is not None and "Nv" in m) else pos.shape[0]

        z = pos[:n, ax] - self.centre
        over = z.abs() - self.gap_half
        hit = over > 0
        n_hit = int(hit.sum())
        if n_hit:
            pos[:n, ax] = torch.where(
                hit, pos[:n, ax] - self.stiff * torch.sign(z) * over.clamp_min(0.0), pos[:n, ax])
            # `lvl.state` IS THE TENSOR; `lvl.state_schema` is what knows the block names. Probing
            # the tensor calls Tensor.__contains__ with a string, which raises -- so the vertex set,
            # which has no `vel` block at all, took down the whole run at frame 1.
            if self.damp_normal and "vel" in lvl.state_schema:
                v = lvl.get("vel")
                # KILL ONLY THE INTO-PLATE COMPONENT. Zeroing the whole velocity would apply a
                # friction the plate does not have, and a matrix particle sliding ALONG a plate is
                # exactly what the free directions are for.
                vn = v[:n, ax]
                v[:n, ax] = torch.where(hit & (vn * torch.sign(z) > 0),
                                        torch.zeros_like(vn), vn)
        PLATE_CONTACT.append((n_hit, float(over[hit].max()) if n_hit else 0.0))
        if not self._said:
            print(f"[plate_confine] {self.at}: rigid plates at {self.centre:+.4g} "
                  f"+/- {self.gap_half:.4g} along axis {ax}, stiff={self.stiff}; "
                  f"{n_hit} of {n} in contact at frame 0", flush=True)
            self._said = True
        return {}


def block_fraction(gap_half, half_extent):
    """Fraction of the domain's volume the two blocks occupy, for a box of half-width `half_extent`.

    Reported rather than specified, because the two things a caller wants to control -- how squashed
    the tissue is, and how much of the box is solid -- are the SAME number seen twice, and only one of
    them can be set. See `PLATES.md` for the arithmetic that pins them together.
    """
    free = min(1.0, max(0.0, gap_half / half_extent))
    return 1.0 - free
