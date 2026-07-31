"""reorient -- cell -> cell (heading). Single-body rotational diffusion of a cell's
own orientation: a zero-drift Brownian wander of the heading, with NO target and NO
neighbour coupling.

This is the ONE uncovered leg of jax-morph's ActiveBrownianDynamics2D (an overdamped
active-Brownian step). That step is a COMPOSITE of three mechanisms, two of which the
promoted language already owns:

    passive drift F/gamma  -> a pair potential under an overdamped mobility (attraction_
                              repulsion / cohesion / ...); already a contract.
    v0*e + translational   -> `glide` (self-propulsion along the heading) + its `noise`;
      noise                   already a contract ("glide + noise = an active Brownian walker").
    dtheta = sqrt(2 D_r dt) -> THIS operator: the persistent heading's rotational
      * xi                    diffusion. No registered operator performs it.

The rotational-diffusion leg is the ingredient that gives an active Brownian walker its
finite PERSISTENCE LENGTH and its ballistic -> diffusive crossover: propulsion pushes the
cell forward along `heading`, and this operator slowly randomizes that `heading` so the
straight run decorrelates. In the textbook ABP limit (kT = 0, no translational noise) it
is the ONLY source of wandering. Per alive cell i over a macro-step dt:

    dtheta_i = sqrt(2 * rot_diffusion * dt) * xi_i,   xi_i ~ N(0, 1)   # zero-drift angle
    heading_i <- R(dtheta_i) . heading_i                              # rotate by dtheta_i

Routing (the polarity family owns every heading write):

* `kind=exchange, family=polarity, set=cell` -- the sibling of `polarity_align` /
  `polarity_flow_align`, and like them it MUTATES `heading` in place and returns `{}`
  (`EMIT=None`): heading is auxiliary control state the engine does not integrate, so a
  heading writer steers it directly. Schedule it alongside the alignment steer (both act
  on `heading` before the next `glide` emits the propulsion velocity).
* NOT `glide`. glide is factored to own ONE concern -- translational propulsion (EMIT=
  velocity; it READS heading and never writes it). The prototype `candidates/motility.py`
  bundled propulsion AND heading rotational diffusion in one class; the promotion to
  `glide` split off the propulsion and DROPPED the diffusion. `reorient` is exactly the
  dropped half, kept as its own single-concern operator so it composes with the polarity
  family rather than grafting heading dynamics onto a position operator.
* NOT `polarity_align` / `polarity_flow_align`. Those are SOCIAL, DETERMINISTIC steers
  that relax the heading toward a TARGET (neighbour-mean heading, or the tissue flow) and
  keep the heading when isolated. `reorient` is the opposite: a single-body STOCHASTIC
  decorrelation with no target and no coupling.

Faithful representation. The source carries `active_heading` as a scalar angle theta and
adds `dtheta` to it as a DYNAMIC (additive) delta, so a second dynamic writer (e.g. an
alignment torque) accumulates into the same field. In Plexus `heading` is a unit VECTOR
[N, D] (read by glide / bounce / sense), so "add dtheta to the angle" becomes "rotate the
heading vector by dtheta" -- exactly equivalent, and norm-preserving by construction (a
planar rotation keeps |heading| = 1, so no renormalization is needed; the single concern
is to rotate). The additive-delta composition survives the change of representation:
planar rotations compose additively, R(a) . R(b) = R(a+b), so applying `reorient` in place
and then another rotational heading writer equals summing the two angle increments -- the
same accumulation the source gets from its dynamic delta.

2-D ONLY. The heading is a scalar rotation angle, so this is planar rotation; the source
raises for n_space_dim != 2. A 3-D active version needs a rotation AXIS (a different
mechanism), not this class -- hence SUPPORTED_DIMS = [2].

SOURCE vs PAPER (rule 5, SOURCE WINS): the paper's mechanics (Deshpande, Mottes et al.
2025, SI I 'Mechanical Interactions', p. 14) is passive and even noise-free -- deterministic
gradient-descent energy minimization of a Morse potential. No self-propulsion, active
heading, or rotational diffusion appears anywhere in the paper; the whole active step, this
leg included, is a library-side active-matter generalization. The recorded disagreement is
that the paper is silent on active/self-propelled dynamics.

Reference: jax-morph ActiveBrownianDynamics2D, papers/jax-morph/jax_morph/physics/mechanics/
dynamics.py:L199 (rotational leg: std_r = sqrt(2 D_r dt) at :L317, dtheta = std_r * xi at
:L362). Physics: Howse et al. (2007) Phys. Rev. Lett. 99:048102 (active-Brownian-particle
rotational diffusion). Paper: Deshpande, Mottes et al., "Engineering morphogenesis of cell
clusters with differentiable programming", Nat. Comput. Sci. (2025) -- the active/rotational-
diffusion step is ABSENT (paper mechanics is passive Morse minimization; SOURCE WINS).
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("reorient", family="polarity", set="cell", kind="exchange")
class Reorient(Exchange):
    EMIT = None                                 # writes `heading` in place (rotational diffusion); returns {} — not an integrable delta
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["heading"]
    WRITES = ["heading"]
    MAPS = []                                   # single-body: no gather map, no neighbour coupling
    SUPPORTED_DIMS = [2]                        # planar rotation of a scalar heading angle (3-D needs an axis)
    REQUIRES_PARAMS = []                        # no required params — rot_diffusion optional (default in __init__)
    MECHANISM_TAGS = ["rotational_diffusion", "orientational_decorrelation", "active_brownian", "persistence"]
    PARAM_ROLES = {"rot_diffusion": "rotational_diffusion_rate"}
    REFERENCE = (
        "jax-morph ActiveBrownianDynamics2D, physics/mechanics/dynamics.py:L199 "
        "(std_r = sqrt(2 D_r dt) at :L317, dtheta = std_r * xi at :L362); "
        "Howse et al. (2007) Phys. Rev. Lett. 99:048102; "
        "Deshpande, Mottes et al. (2025) Nat. Comput. Sci. (active step ABSENT — SOURCE WINS)."
    )

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")                     # the set this acts on (engine-injected)
        self.rot_diffusion = float(params.get("rot_diffusion", 1.0))   # D_r; 0 fixes the heading (no crash)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        h = lvl.heading                                         # [N, D] unit heading vector
        if h.shape[-1] != 2:                                    # 2-D only: a scalar rotation angle
            raise ValueError(
                "reorient is 2-D only (rotational diffusion of a scalar heading angle); "
                f"heading has {h.shape[-1]} components — a 3-D version needs a rotation axis."
            )
        dt = float(getattr(H.config, "dt", 1.0))
        std_r = math.sqrt(2.0 * self.rot_diffusion * dt)        # heading noise scale (NO gamma; its own rate D_r)
        # Active/occupied gate: dead or masked slots draw dtheta = 0 -> identity rotation
        # (heading unchanged). Noise is sized to the full padded capacity, like the source.
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        dtheta = std_r * torch.randn(N, generator=getattr(H, "rng", None), device=dev) * m   # [N] zero-drift angle
        c, s = torch.cos(dtheta), torch.sin(dtheta)
        hx, hy = h[:, 0], h[:, 1]
        # rotate each heading by its own dtheta: R(dtheta) . h. A planar rotation is exactly
        # norm-preserving, so |heading| = 1 is maintained without renormalization.
        lvl.heading = torch.stack([c * hx - s * hy, s * hx + c * hy], dim=-1)
        return {}
