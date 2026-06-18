"""Neighbour alignment -- a second-derivative Lateral operator (boids rule 2 / Vicsek).

The alignment term of the classic boids law, isolated as its own one-concern
operator. A receiver particle i steers its acceleration toward the *mean
heading* of its neighbours j (the edges left by a `rewire` operator such as
`radius_graph`), weighted per neighbour by a gate w_ij:

    acc_i = a * (sum_j w_ij (vel_j - vel_i)) / (sum_j w_ij)

A single operator covers two regimes via the `gate` option -- they differ only
in how each neighbour is weighted:

  gate: none        every graph neighbour counts equally (w_ij = 1).  The
                    weighted mean collapses to a plain mean over neighbours --
                    this *is* the boids alignment rule.  Add `per_type: true`
                    (below) to also fold in boids' per-type alignment strength,
                    reproducing boids' alignment term exactly.

  gate: contact     (default) a smooth (smoothstep) *contact* gate: i only
                    matches heading with neighbours it is (almost) touching, so
                    the force has no discontinuity as particles drift in/out of
                    contact.  w_ij falls from 1 (full contact) to 0 at radius r:

                        w_ij = smoothstep over |pos_j - pos_i|
                             = 1               for |d| <= (1 - softness)*r
                             -> 0 (Hermite)    out to |d| = r

Params (all overridable via the spec `params:` block):
  a         alignment scale (default 5e-4, the canonical PDE_B constant)
  gate      "contact" (default) | "none"
  r         contact radius, world units (default 0.05; gate="contact" only)
  softness  falloff band in [0,1] (default 0.5; 0 = hard cutoff; gate="contact" only)
  per_type  if true, multiply each receiver's alignment by its per-type weight
            p[:, type_index] (default false).  Requires the set's `types:` to
            carry `p`.  Off by default so the operator needs no `types`.
  type_index  which slot of p holds the alignment weight (default 1, matching
            boids' p = [cohesion, alignment, separation]).

The graph radius should be >= r so the contact gate has its candidates. The mean
is taken over (weighted) neighbours -> a particle with no neighbour in contact
gets zero alignment.

Second-derivative law -> returns an acceleration (set `prediction:
second_derivative`); the engine integrates vel += dt*acc, pos += dt*vel. It has
no 1/|d|^2 term, so it needs no `min_radius` on the graph, and it composes
cleanly with `attraction_repulsion`, `drag`, etc.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


@register_operator("alignment", level="particle", kind="lateral")
class Alignment(Lateral):
    PREDICTION = "second_derivative"            # emits an acceleration

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("a", 5e-4))         # alignment scale
        self.gate = str(params.get("gate", "contact"))  # "contact" | "none"
        self.r = float(params.get("r", 0.05))         # contact radius (world units)
        self.softness = float(params.get("softness", 0.5))  # falloff band, [0,1]; 0 = hard cutoff
        self.per_type = bool(params.get("per_type", False))  # fold in boids per-type weight
        self.type_index = int(params.get("type_index", 1))   # slot of p (1 = alignment, boids order)
        self.at = params.get("_at", "particle")
        if self.gate not in ("contact", "none"):
            raise ValueError(f"alignment: gate must be 'contact' or 'none', got {self.gate!r}")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        vel = lvl.get("vel")
        occ = lvl.occ
        N = vel.shape[0]
        ei = lvl.edge_index                                 # row0 = receiver i, row1 = neighbour j
        if ei.numel() == 0:
            return {self.at: torch.zeros(N, 2, device=vel.device)}
        i, j = ei[0], ei[1]

        if self.gate == "contact":
            d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                              getattr(H, "world_width", 1.0))   # j - i  [E, 2]
            dist = d.norm(dim=-1)                                # |d|  [E]
            # smoothstep contact gate: 1 for dist <= r_in, 0 at dist >= r, Hermite in between.
            r_in = (1.0 - self.softness) * self.r
            denom = max(self.r - r_in, 1e-12)                    # softness=0 -> denom~0 -> hard step
            t = ((self.r - dist) / denom).clamp(0.0, 1.0)
            gate = t * t * (3.0 - 2.0 * t)                       # smoothstep weight  [E]
        else:                                                    # gate == "none": all neighbours equal
            gate = torch.ones_like(occ[j])
        w = gate * occ[j]                                        # gate & mask empty neighbour slots  [E]

        dv = vel[j] - vel[i]                                     # match neighbour heading  [E, 2]
        msg = self.a * dv * w[:, None]
        if self.per_type:
            pt = lvl.type_params[lvl.node_type[i], self.type_index]   # per-receiver alignment weight  [E]
            msg = msg * pt[:, None]

        acc = torch.zeros(N, 2, device=vel.device)
        acc.index_add_(0, i, msg)
        deg = torch.zeros(N, device=vel.device).index_add_(0, i, w)
        acc = acc / deg.clamp(min=1.0)[:, None]                 # (weighted) mean over neighbours
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
