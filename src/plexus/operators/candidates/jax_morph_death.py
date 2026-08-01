"""apoptose (cell set, structural): stochastic cell death -- cell_divide's biological inverse.

Where `cell_divide` WAKES a dormant slot (occ 0 -> 1) to add a cell, `apoptose` RETIRES a live
one (occ 1 -> 0): each cell alive at the death decision independently draws a Bernoulli death
event with the SAME per-cell hazard the division step uses,

    p_i = 1 - exp(-clip(death_rate_i, 0) * dt)      # -expm1(-clip(rate,0)*dt)

and a cell that draws 1 has its `alive` mask flipped False. The tunable is `death_rate`, a
HERITABLE per-cell STATE field (not a constructor arg) -- an initial condition or an upstream
controller sets it, exactly like the per-type `div_rate` that `cell_divide` reads. The `rate`
param is only a uniform fallback when no `death_rate` buffer is present (default 0 -> inert).

Routing (the `apoptose` contract, kind=structural, family=growth, set=cell, maps=[]): a single-set
morphism cell -> cell. It mutates occupancy in place and returns {} -- no integrable delta, like
every structural op. `EMIT=None`.

Two things a reimplementer must get right (both from the source):

* THE PERSISTENT `death` RECORD IS NOT THE ALIVE MASK. `apoptose` writes a per-cell float `death`
  (0/1) recording the slots removed THIS macro-step, for downstream lineage postprocessing
  (`reconstruct_lineage`). It is OVERWRITTEN each step (zeroed, then set on the newly-dead), never
  accumulated -- reading a single final `death` array as a lifetime death count is wrong. It is a
  float so it can be summed/averaged/differentiated; `alive` (occupancy) stays boolean. The die
  mask is re-AND'd with the LIVE mask (a cell already dormant cannot "newly die"), so we never mark
  an already-dead slot as freshly dead.

* THE FREED SLOT IS NOT RECYCLED WITHIN THE STEP. Setting occ 1 -> 0 makes the slot dormant, but
  `cell_divide`'s allocator (`occ == 0` slots) reuses it only on a LATER macro-step. Composition
  order is load-bearing: run `cell_divide` BEFORE `apoptose` (divide-then-die) so a mother and her
  newborn daughters may all die in one step while no death-freed slot is repopulated that step --
  which keeps the per-step `death` lineage record intact.

PAPER CONTRADICTION (source wins). The paper does NOT model cell death: "death"/"apoptosis" appear
zero times, and the forward model's cell capabilities (p. 2) are exactly division, growth, stress
sensing, and morphogen excretion/detection -- no death. Its only cell removal is an EXTERNAL
post-hoc robustness ablation ("Loss of the final state after removal, at random, of a fraction of
cells", p. 8/21), not a per-cell hazard dynamic. `Death` is thus a shipped-library step with no
paper counterpart; we translate the CODE, not the prose, because the differential test compares us
to the running source. See the atlas entry's `surprises` for the full account.

Gradient-estimator note. In jax-morph the forward `died` draw is a straight-through Bernoulli whose
physical effect is a HARD boolean mask, so survival/death objectives take the SCORE-FUNCTION
(REINFORCE) gradient through `Death.logp`, never a pathwise gradient through `alive`. Plexus's
engine runs the forward EFFECT (the source's `replay`); the trace/`logp` scoring layer is not
modelled here, exactly as `cell_divide` realises only the forward proliferation event.

Distinct from the efflux-boundary `death` operator (candidates/death.py), which removes a cell when
its centroid crosses a spatial exit line -- that is a geometric sink, this is a stochastic hazard.

Reference: Deshpande, Mottes et al., "Engineering morphogenesis of cell clusters with
differentiable programming", Nat. Comput. Sci. (2025) -- death is ABSENT from the paper (see
surprises); translated from papers/jax-morph/jax_morph/physics/death.py:26 (Death.replay / _dist).
"""
from __future__ import annotations

import torch

from plexus.models.base import Structural
from plexus.models.registry import register_operator


@register_operator("apoptose", family="growth", set="cell", kind="structural")
class Apoptose(Structural):
    EMIT = None                                       # structural: retires live slots, mutates occ in place; returns {} — no integrable delta
    # typed signature (Plexus2 sec. 2.1): a morphism cell -> cell that reads the heritable
    # per-cell death hazard and writes the alive mask (occupancy) + the per-step death record.
    # MAPS=[] -- each cell draws in isolation, no gather/scatter, no cell-to-cell coupling.
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["death_rate"]                            # the heritable per-cell hazard rate (a STATE field, not a param)
    WRITES = ["alive", "death"]                       # flips occupancy; writes the per-step float death record
    MAPS = []
    SUPPORTED_DIMS = [2, 3]                            # acts on occupancy/hazard; dimension-agnostic
    REQUIRES_PARAMS = []                              # no required params — `rate` falls back to per-cell death_rate else 0 (no-op)
    MECHANISM_TAGS = ["apoptosis", "cell_death", "stochastic_removal", "population_decline"]
    PARAM_ROLES = {"rate": "fallback_death_hazard_rate"}
    REFERENCE = ("Deshpande, Mottes et al. (2025), Nat. Comput. Sci. — death is ABSENT from the "
                 "paper (see atlas surprises); papers/jax-morph/jax_morph/physics/death.py:26 "
                 "(Death.replay / Death._dist).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.rate = float(params.get("rate", 0.0))        # uniform fallback hazard if no per-cell death_rate buffer

    def forward(self, H, mask=None):
        lvl = H.level(self.at); dev = lvl.state.device
        dt = float(getattr(H.config, "dt", 1.0))
        buf = lvl.occ.shape[0]
        live = lvl.occ > 0

        # per-cell hazard rate: prefer the heritable `death_rate` STATE buffer (the source's
        # DEATH_RATE field), else the scalar `rate` param (0 -> p=0 -> inert no-op).
        rate = getattr(lvl, "death_rate", None)
        rate = rate if rate is not None else torch.ones(buf, device=dev) * self.rate  # ones*p, not full(p): `full` rejects a tensor fill value, blocking a learnable knob
        # p = 1 - exp(-clip(rate, 0) * dt). The clip guards a negative controller output: it would
        # otherwise give p < 0 and NaN a score; here it just means "no death" (p = 0).
        p = -torch.expm1(-rate.clamp(min=0.0) * dt)

        # eligibility = alive AT the death decision, restricted to the `at:` selection. This IS the
        # source's `die_eligible = alive` mask; re-AND with `live` guards an already-dormant slot.
        elig = live
        if mask is not None:
            elig = elig & mask.to(torch.bool)
        p = p * elig.to(p.dtype)

        # exact Bernoulli death draw (same generator convention as cell_divide), masked to eligible.
        draw = torch.rand(buf, generator=getattr(H, "rng", None), device=dev)
        die = (draw < p) & elig

        # persistent per-step `death` record (float 0/1), OVERWRITTEN each macro-step, not
        # accumulated. Lazily provisioned like cell_grow's grow_V.
        if getattr(lvl, "death", None) is None:
            lvl.register_buffer("death", torch.zeros(buf, device=dev, dtype=p.dtype))
        lvl.death.zero_()
        lvl.death[die] = 1.0

        # EFFECT: flip alive -> dead by retiring the slot (occ 1 -> 0). The slot is NOT freed within
        # the step -- cell_divide reuses `occ == 0` slots only on a LATER macro-step, so under the
        # divide-then-die ordering the death record survives for lineage reconstruction.
        # occ is CLONED before the write. `grow_radius` multiplies its delta by
        # `lvl.occ[:, None]`, so the live mask is on the autograd tape; mutating it in
        # place invalidates that multiply and the whole rollout stops being
        # differentiable. Found by grad_probe.py, not by any forward run -- a forward run
        # cannot tell the two apart.
        occ = lvl.occ.clone()
        occ[die] = 0.0
        lvl.occ = occ
        return {}
