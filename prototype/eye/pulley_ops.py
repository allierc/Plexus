"""pulley_ops -- a single, LOCALIZED anatomical pulley (PROTOTYPE-LOCAL, not promoted).

    op: muscle_pulley_ring
    at: muscle_particle
    k: 8000.0
    s_ring: 0.5          # fibre coordinate the ring sits at (0.5 = mid-belly, "the equator")
    half_width: 0.06      # the band of particles counted as "at the ring"

`muscle_sleeve` (this file's sibling) is a DISTRIBUTED transverse constraint over the
proximal 70% of the muscle's length, each particle held near ITS OWN rest lateral position --
closer to a fascial sheath running the whole path than to the discrete structure Demer's
active-pulley hypothesis actually describes. Raising its `k` was tried as a test of "does the
pulley mechanism fix the buckling" and did not help (see git history / run_forced_duction.py's
docstring) -- which does not settle the question, because a stronger BROAD spring is not the
same experiment as a real pulley: the anatomical structure is a small fibrous ring, fixed to
the orbital wall, that the belly threads through at roughly ONE location, not a rail it
follows for 70% of its length.

This operator is that narrower claim, tested on its own: a NARROW band of particles near
`s_ring` is pulled toward a SINGLE SHARED anchor point (that band's own rest centroid, one
point per muscle) instead of each particle's own individual rest position. Sliding along the
fibre through the ring is free (same `d - d_par` decomposition `muscle_sleeve` uses); only
the perpendicular excursion of the banded particles, relative to the ring's own fixed point,
is restrained. Meant to run WITHOUT `muscle_sleeve` (k_sleeve=0) so any change in the buckling
is attributable to the ring itself, not conflated with the broader constraint.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("muscle_pulley_ring", family="mechanics", set="muscle_particle",
                   kind="lateral")
class MusclePulleyRing(Lateral):
    """One fixed ring per muscle: a narrow s-band pulled toward its own rest centroid."""

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["k"]
    INPUTS = ["muscle_particle"]
    OUTPUTS = ["muscle_particle"]
    READS = ["pos", "vel"]
    WRITES = []
    MECHANISM_TAGS = ["muscle_pulley", "transverse_constraint", "path_stabilisation",
                      "anti_buckling", "localized_constraint"]
    PARAM_ROLES = {"k": "ring_stiffness", "c": "ring_damping", "s_ring": "ring_location",
                   "half_width": "ring_capture_band"}
    REFERENCE = "Demer, J. L. et al. (1995). Invest. Ophthalmol. Vis. Sci. 36:1125."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle_particle")
        self.k = float(params["k"])
        self.c = float(params.get("c", 30.0))
        self.s_ring = float(params.get("s_ring", 0.5))
        self.half_width = float(params.get("half_width", 0.06))
        self._anchor = None                 # [n_muscle, 3], set once from rest geometry

    def forward(self, H, mask=None):
        p = H.level(self.at)
        if not hasattr(p, "rest") or not hasattr(p, "fibre") or not hasattr(p, "s"):
            return {}
        band = (p.s - self.s_ring).abs() <= self.half_width
        if not band.any():
            return {}
        if self._anchor is None:
            n_muscle = int(p.parent.max()) + 1
            dev = p.state.device
            anchor = torch.zeros(n_muscle, 3, device=dev)
            for mi in range(n_muscle):
                sel = band & (p.parent == mi)
                if sel.any():
                    anchor[mi] = p.rest[sel].mean(0)
            self._anchor = anchor
        f = p.fibre
        target = self._anchor[p.parent]
        d = p.get("pos") - target
        v = p.get("vel")
        d_perp = d - (d * f).sum(1, keepdim=True) * f
        v_perp = v - (v * f).sum(1, keepdim=True) * f
        acc = torch.zeros_like(d)
        acc[band] = -self.k * d_perp[band] - self.c * v_perp[band]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
