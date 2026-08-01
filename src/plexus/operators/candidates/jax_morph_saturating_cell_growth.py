"""saturating cell growth -- a per-cell radius ODE that relaxes toward a maximum size.

Each cell carries a geometric size scalar `radius`; this operator grows it toward the module
target `max_radius` (R) by the SATURATING (von Bertalanffy) law

    dr/dt = k * (1 - r/R),

where the per-cell rate k is read from the heritable state field `growth_rate`. Growth is
fastest at small r and relaxes smoothly to zero as r -> R -- a decelerating approach, not a
constant-speed ramp. The rate lives in the STATE (not a constructor knob) precisely so an
upstream controller (a gene network / MLP) can rewrite it each macro-step and gradients flow
back through it from a morphology objective; default 0 means no growth until a rate is supplied.

EXACT FLOW, NOT EULER. The increment applied over a step dt is the closed-form integral of the
linear ODE, not forward Euler:

    dr = (R - r) * (1 - exp(-k*dt/R)),

which is unconditionally stable and monotone for any dt and any k >= 0 (a naive dr = k(1-r/R)dt
can overshoot R for large dt; this cannot). The source returns this dr directly as a dt-scaled
DYNAMIC delta and the model adds it. Plexus integrates a first-order block as `x += dt*delta`,
so -- exactly as the `regulate:neural_ode` sibling does for its self-solved gene increment -- we
return the effective mean RATE `delta = dr/dt`; the engine's `dt *` then recovers the exact
endpoint radius. The dt cancels; it is faithful adaptation to Plexus's convention, not a second
integration. The exact-flow property holds while this is the sole dynamic writer of `radius`;
with a co-writer it degrades to a stable additive operator-split contribution.

ROUTING. A per-cell, self-contained relaxation with NO neighbour coupling and NO field -- the
same shape as `attractor_flow`/`signal` (a within-set autonomous ODE), so `kind=lateral`. But it
advances a NON-coordinate first-order block (`radius`), not the spatial coordinate, so it sets
`EMIT="velocity"` with `INTEGRAND="radius"` (like `neural_ode`'s `gene` block): `_resolve_emit`
sees a non-`pos` integrand and never constrains the cell's spatial integration order, and the
delta lands in the `radius` accumulator (`x += dt*delta`), summed with any other radius writer.

BIOLOGY vs PAPER (source wins, rule 5). The paper's Methods state CONSTANT-rate linear growth
with a hard clamp, R_i(t+dt) = min(R_i(t) + dR, R_max) (Deshpande, Mottes et al. 2025, p. 9
"Cell Growth", restated p. 14; intro "cells grow at a fixed rate until reaching a maximum
radius"). The CODE implements the saturating exponential relaxation above with NO clamp
(saturation is intrinsic). Different trajectories -- constant speed then an abrupt stop vs a
smooth asymptote -- so a reimplementer following the prose would produce the wrong dynamics. We
translate the code, because the differential test compares us to the running source.

REFINEMENT of the registered 'cell_grow' contract -- and why it CANNOT co-register under that
name. This is the same biology as the shipped 'cell_grow' (a cell grows toward a maximum size),
so the atlas files it as a REFINEMENT, not `new`. But the refinement FLIPS the kind: shipped
'cell_grow' is `structural` (EMIT=None, `forward()` returns {}) -- it advances a rest-VOLUME
multiplier and REALIZES it by waking dormant MPM reserve particles -- whereas this mechanism is
delta-emitting (it relaxes a scalar `radius` and mutates no state directly). Plexus's registry
FORBIDS a second implementation of one contract from differing in kind (registry.py:131), so
`@register_operator("cell_grow", kind="lateral", ...)` raises at import next to the shipped
structural 'cell_grow'. That rejection is not an obstacle to route around -- it IS the breaking
change the ledger exists to surface (a refinement that widens a signature silently invalidates
every existing caller; here `kind` widening is hard-rejected). So the candidate registers under
the distinct name `grow_radius`; promoting it is a curator decision about whether 'cell_grow'
should widen to admit a delta-emitting radius realization. Note also that the entry's
contract.kind "field" reads as "delta-emitting"; the concrete Plexus `field` kind is grid
self-dynamics that returns {} (diffuse/decay) and cannot emit this per-cell delta -- `lateral`
is the faithful kind.

Reference: Deshpande, Mottes et al., "Engineering morphogenesis of cell clusters with
differentiable programming", Nat. Comput. Sci. (2025), Methods "Cell Growth" (p. 9, restated
p. 14); translated to torch from papers/jax-morph/jax_morph/physics/growth.py:23
(SaturatingCellGrowth.__call__, L81-83) and :20 (the GROWTH_RATE heritable state field).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("grow_radius", family="growth", set="cell", kind="lateral")
class GrowRadius(Lateral):
    """Saturating (von Bertalanffy) per-cell radius growth as an exact-flow delta. A refinement
    of the shipped 'cell_grow' contract that cannot co-register under that name (kind flips
    structural -> delta-emitting; see the module docstring). Translated from
    papers/jax-morph/jax_morph/physics/growth.py:23."""

    EMIT = "velocity"                  # first-order block: the delta is dr/dt-equivalent (dr/dt), engine does radius += dt*delta
    INTEGRAND = "radius"               # writes a NON-coordinate first-order block (the geometric size), not pos
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["radius", "growth_rate"]  # current size r + the heritable per-cell rate k (a STATE field, default 0)
    WRITES = ["radius"]                # the dt-increment of the radius
    MAPS = []                          # per-cell autonomous ODE: no gather/scatter, zero cell-to-cell coupling
    SUPPORTED_DIMS = [2, 3]            # acts on a per-cell scalar; independent of spatial dimension
    DIFFERENTIABLE = True              # pure torch (exp/mul) -> autograd flows through r, k, and the target R
    REQUIRES_PARAMS = []               # all optional: no growth_rate + rate=0 -> k=0 -> byte no-op
    MECHANISM_TAGS = ["cell_growth", "rest_size_growth", "von_bertalanffy", "saturating_growth",
                      "controllable_growth_rate"]
    PARAM_ROLES = {"max_radius": "asymptotic_target_size",
                   "rate": "uniform_fallback_growth_rate",
                   "radius": "size_state_block", "growth_rate": "rate_state_block"}
    REFERENCE = ("Deshpande, Mottes et al. (2025), Nat. Comput. Sci., Methods 'Cell Growth' "
                 "(p. 9, restated p. 14) -- paper states constant-rate + min-clamp, CODE is the "
                 "saturating von Bertalanffy flow (source wins, see surprises); translated from "
                 "papers/jax-morph/jax_morph/physics/growth.py:23 (SaturatingCellGrowth.__call__).")


    # LEARNABLE TUNABLES. A parameter kept as a Python float is a CONSTANT in the autograd graph,
    # so `d(loss)/d(param)` does not exist -- measured, not assumed (grad_probe.py step 2). Passing
    # a tensor keeps it on the tape. This mirrors the reference's own rule: "store as a jax.Array
    # anything you want to learn or vary without recompiling". Forward runs are unaffected: a
    # float stays a float.
    @staticmethod
    def _tunable(v, device):
        import torch as _t
        if isinstance(v, _t.Tensor):
            return v.to(device)
        return float(v)

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.max_radius = self._tunable(params.get("max_radius", 1.0), device)   # R: asymptotic target size
        if float(self.max_radius) <= 0.0:                            # R divides the exponent; a target size must be positive
            raise ValueError(f"grow_radius: max_radius must be > 0, got {self.max_radius}")
        self.rate = float(params.get("rate", 0.0))                  # uniform fallback k if no per-cell growth_rate (0 -> no-op)
        self.radius_block = str(params.get("radius", "radius"))     # the size state block this grows
        self.rate_block = str(params.get("growth_rate", "growth_rate"))  # the heritable per-cell rate block/buffer
        # instance INTEGRAND routes the delta into the configured size block (engine reads it off
        # the instance in _run_token). The class INTEGRAND stays "radius" so _resolve_emit sees a
        # non-`pos` integrand and does not constrain the coordinate's integration order.
        self.INTEGRAND = self.radius_block

    def _read_scalar(self, lvl, name):
        """A per-cell [N, w] view of a named quantity: a state block if the schema declares one,
        else a registered per-node buffer of that name, else None."""
        if name in lvl.state_schema:
            return lvl.get(name)
        buf = getattr(lvl, name, None)
        if buf is not None and torch.is_tensor(buf):
            return buf if buf.dim() == 2 else buf[:, None]
        return None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        dt = float(getattr(H.config, "dt", 1.0))

        r = self._read_scalar(lvl, self.radius_block)               # current per-cell radius [N, w]
        if r is None:
            raise ValueError(
                f"grow_radius: set {self.at!r} has no size block/buffer {self.radius_block!r} to "
                f"grow (declare it in the set's `state:` schema or provision it as a buffer).")

        # per-cell growth rate k: prefer the heritable `growth_rate` STATE (block or buffer), else
        # the uniform `rate` param (default 0 -> k=0 -> decay=1 -> dr=0 -> byte-identical no-op).
        k = self._read_scalar(lvl, self.rate_block)
        if k is None:
            k = torch.full_like(r, self.rate)

        if dt <= 0.0:                                               # nothing integrates over a zero step
            return {self.at: torch.zeros_like(r)}

        # exact flow of dr/dt = k(1 - r/R) over dt: dr = (R - r)(1 - exp(-k*dt/R)). Saturating,
        # monotone, unconditionally stable; never overshoots R for k >= 0 (no min/clamp needed).
        R = self.max_radius
        decay = torch.exp(-k * dt / R)
        dr = (R - r) * (1.0 - decay)
        # return the mean RATE dr/dt so the engine's first-order step (radius += dt*delta) recovers
        # the exact increment dr; the dt cancels (same convention as regulate:neural_ode).
        delta = dr / dt
        delta = delta * lvl.occ[:, None]                           # dormant cells hold their size
        if mask is not None:                                       # `at:` may restrict growth to a subset (e.g. cell[type=bud])
            delta = delta * mask[:, None].float()
        return {self.at: delta}
