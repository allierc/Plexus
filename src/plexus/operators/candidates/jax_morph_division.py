"""agent_divide:volume_conserving -- oriented, volume-conserving stochastic cleavage.

A second IMPLEMENTATION of the registered `agent_divide` contract (structural / growth / cell),
translated from jax-morph's `Division` step. The forward biology is the SAME divide primitive the
promoted isotropic `agent_divide` already realizes -- one alive cell becomes two as an independent
Bernoulli event with the per-cell hazard

    p_i = 1 - exp(-clip(division_rate_i, 0) * dt)      # = -expm1(-clip(rate,0)*dt)

on a fixed capacity buffer, the daughter WAKING a dormant slot (occ 0 -> 1) and inheriting the
mother's heritable per-cell fields. What this implementation REFINES over the default (the reason
the entry files it `refinement` of agent_divide, not a bare `alias`):

* VOLUME CONSERVATION, not mass doubling. The default agent_divide is confluent proliferation --
  the daughter appears at the mother's radius, so tissue AREA grows per division. jax-morph is
  CLEAVAGE-stage division: the embryo does not grow, the cells just get smaller. BOTH daughters
  take radius r*m with m = 2^(-1/d) (`n_space_dim` = the world dim sets the exponent), so each
  daughter has HALF the mother's d-volume and the two together conserve it. m ~ 0.707 in 2D,
  ~0.794 in 3D -- a hardcoded 1/2, or a 3D factor used in 2D, silently breaks the conservation.

* ORIENTED placement (Hertwig's rule). The default jitters the daughter isotropically; here the
  split direction blends an optional per-cell unit axis `division_axis` with unit-RMS isotropic
  noise at amplitude signal-to-noise ratio `orientation_snr`,

    a_hat = division_axis / (||division_axis|| + 1e-12)
    xi    ~ Normal(0, I_d) / sqrt(d)                    # unit-RMS isotropic
    dir   = (orientation_snr * a_hat + xi) / (||...|| + 1e-12)

  orientation_snr = 0 OR a zero axis falls back to pure isotropic placement (the additive default),
  so no promoted operator reads a `division_axis` today -- the oriented capability is genuinely new.

* JUST-TOUCHING, symmetric placement. The offset uses the NEW (halved) radius, offset = (r*m)*dir;
  the mother moves to x + offset and the daughter to x - offset, so the two daughters sit exactly
  touching (centre gap 2*r*m = sum of their radii) and the pair is centred on the mother's old
  position. Using the mother's OLD radius would overlap or gap them.

* LINEAGE. Beyond a `born` new-daughter flag we record `mother` = the parent slot index (sentinel
  -1 elsewhere), so clonal ancestry is reconstructable (jax-morph `reconstruct_lineage`). `born`
  and `mother` reset every macro-step (a per-step record); Plexus's own `Level.lineage` buffer
  captures the same parent-slot provenance.

* CAPACITY is a hard wall, not a crash. If more cells divide than there are free slots the surplus
  dividers are silently DROPPED and tallied into the GLOBAL, running `division_overflow` counter --
  a reimplementer who RAISES on overflow diverges. This is numerics (a full-buffer guard), not
  biology; the default agent_divide handles a full buffer by simply stopping.

NOT modelled (excluded so the refinement is not over-claimed): the straight-through / pathwise
differentiability and the `logp` score are gradient-ESTIMATOR machinery -- an engine concern in
Plexus, ruled out_of_scope for the shipped stochastic steps (see the `apoptose` sibling). This
operator realizes only the forward EFFECT (the source's `replay`), like every structural op.

SOURCE vs PAPER (rule 5, source wins): the paper (Deshpande, Mottes et al. 2025, p.2) chooses the
division plane UNIFORMLY at random and states only "two daughter cells, each with half the volume
of a fully grown mother cell" -- no oriented axis. The CODE adds `division_axis` + `orientation_snr`
oriented placement. We translate the code; either reading is still stochastic cell division, so the
contradiction is recorded, not verdict-changing.

Reference: Deshpande, Mottes et al., "Engineering morphogenesis of cell clusters with
differentiable programming", Nat. Comput. Sci. (2025), p.2 (division = half-volume daughters,
uniform plane); translated to torch from papers/jax-morph/jax_morph/physics/division.py:116
(Division._dist), :121 (sample_trace, the oriented direction) and :148 (Division.replay).
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Structural
from plexus.models.registry import register_operator


@register_operator("agent_divide", family="population", set="cell", kind="structural",
                   implementation="volume_conserving")
class CellDivideVolumeConserving(Structural):
    EMIT = None                                       # structural: wakes dormant slots, mutates occ/state in place; returns {} — no integrable delta
    # typed signature (Plexus2 sec. 2.1): a morphism cell -> cell. The widened read/write set the
    # entry costs -- reads the heritable per-cell hazard + orientation axis + radius/position, writes
    # position/radius/alive, the lineage records, and the global overflow diagnostic. MAPS=[] --
    # each cell draws in isolation (the slot allocation is bookkeeping, not a cell-to-cell coupling).
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = ["division_rate", "division_axis", "radius", "pos", "alive"]
    WRITES = ["pos", "radius", "alive", "celltype", "born", "mother", "division_overflow"]
    MAPS = []
    SUPPORTED_DIMS = [2, 3]                            # m = 2^(-1/d) reads the world dim; jax-morph also allows 1D
    REQUIRES_PARAMS = []                              # no required params — `rate` falls back to per-cell division_rate else 0 (inert)
    MECHANISM_TAGS = ["proliferation", "mitosis", "growth", "oriented_division",
                      "volume_conserving_division", "cleavage", "cell_lineage"]
    PARAM_ROLES = {"orientation_snr": "division_axis_alignment_strength",
                   "rate": "fallback_division_hazard_rate", "radius": "fallback_cell_radius"}
    REFERENCE = ("Deshpande, Mottes et al. (2025), Nat. Comput. Sci., p.2 (half-volume daughters, "
                 "uniform division plane); papers/jax-morph/jax_morph/physics/division.py:116 "
                 "(Division._dist), :121 (sample_trace) & :148 (Division.replay).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.rate = float(params.get("rate", 0.0))              # uniform fallback hazard if no per-cell division_rate
        self.orientation_snr = float(params.get("orientation_snr", 0.0))  # 0 -> isotropic; the sole physics knob
        self.radius0 = float(params.get("radius", 1.0))         # fill for a missing `radius` buffer (birth size)

    @staticmethod
    def _block(lvl, name):
        """The per-cell vector for `name`, wherever the spec chose to keep it.

        A spec may declare `radius` / `division_rate` as STATE BLOCKS (so the engine records them
        and an upstream operator can integrate them) or leave them as Level buffers. This
        operator originally read buffers only -- so in a spec whose `radius` was a state block it
        silently read a buffer it had itself created, at the constructor default, and
        `division_rate` fell back to 0. The run completed, the movie looked fine, and the cells
        never divided: 4 cells after 40 frames against the reference's 82. Nothing raised.
        `grow_radius` used the state block at the same time, so the two operators of this same
        atlas disagreed about where a cell's radius lives.
        """
        if name in lvl.state_schema:
            a, b = lvl.state_schema[name]
            return lvl.state[:, a:b].reshape(-1) if b - a == 1 else lvl.state[:, a:b]
        return getattr(lvl, name, None)

    @staticmethod
    def _set_block_on(lvl, st, name, idx, value):
        """Write into the CLONE `st` when the block is state; buffers are written directly."""
        if name in lvl.state_schema:
            a, b = lvl.state_schema[name]
            st[idx, a:b] = value.reshape(-1, b - a) if value.dim() else value
        else:
            getattr(lvl, name)[idx] = value

    def forward(self, H, mask=None):
        lvl = H.level(self.at); dev = lvl.state.device
        dt = float(getattr(H.config, "dt", 1.0))
        buf = lvl.occ.shape[0]
        d = H.dim
        m = 2.0 ** (-1.0 / d)                                   # volume-conserving radius factor r -> r*m
        px0, px1 = lvl.state_schema["pos"]
        rng = getattr(H, "rng", None)

        # --- lazily provision the state this contract owns ------------------------------------ #
        # radius: the heritable per-cell size the split halves; division_rate/division_axis are the
        # heritable drivers (like apoptose's death_rate). born/mother are the per-step lineage record;
        # division_overflow is a GLOBAL running counter (NOT reset each step).
        if "radius" not in lvl.state_schema and getattr(lvl, "radius", None) is None:
            lvl.register_buffer("radius", torch.full((buf,), self.radius0, device=dev))
        if getattr(lvl, "born", None) is None:
            lvl.register_buffer("born", torch.zeros(buf, device=dev))
        if getattr(lvl, "mother", None) is None:
            lvl.register_buffer("mother", torch.full((buf,), -1, dtype=torch.long, device=dev))
        if getattr(lvl, "division_overflow", None) is None:
            lvl.register_buffer("division_overflow", torch.zeros((), device=dev))

        # per-step lineage record resets to its default every macro-step (jax-morph re-emits `born`
        # as zeros and `mother` as -1 on every replay); the global overflow counter accumulates.
        lvl.born.zero_()
        lvl.mother.fill_(-1)

        live = lvl.occ > 0
        if not bool(live.any()):
            return {}

        # --- per-cell Bernoulli hazard p = 1 - exp(-clip(rate,0)*dt) -------------------------- #
        rate = self._block(lvl, "division_rate")
        rate = rate if rate is not None else torch.full((buf,), self.rate, device=dev)
        elig = live
        if mask is not None:
            elig = elig & mask.to(torch.bool)
        p = -torch.expm1(-rate.clamp(min=0.0) * dt) * elig.to(rate.dtype)   # clip: a negative rate -> no division
        draw = torch.rand(buf, generator=rng, device=dev)
        movers = (draw < p).nonzero(as_tuple=True)[0]                       # cells that drew a division this step
        if movers.numel() == 0:
            return {}

        # --- capacity is a hard wall: the k-th divider takes the k-th free slot, surplus dropped -- #
        free = (~live).nonzero(as_tuple=True)[0]
        cap = min(int(movers.numel()), int(free.numel()))
        dropped = int(movers.numel()) - cap                                # committed dividers with no free slot
        if dropped:
            lvl.division_overflow += float(dropped)                        # GLOBAL, accumulates across macro-steps
        if cap == 0:                                                       # buffer full: every divider dropped
            return {}
        parents = movers[:cap]
        slots = free[:cap]

        # FUNCTIONAL UPDATE, not in-place. Every write below lands on a CLONE of the state,
        # which is then assigned back. In-place writes are cheaper and, in a forward-only run,
        # invisible -- but they make the rollout non-differentiable: autograd refuses a tensor
        # that was mutated after being read ("modified by an inplace operation ... version 4;
        # expected version 3"). The reference has this discipline for free because JAX has no
        # in-place at all; torch lets us skip it, and the inverse half of Plexus is what pays.
        # Measured with grad_probe.py: with this clone, a gradient survives 20 frames INCLUDING
        # division events; without it, the tape dies at the first division.
        st = lvl.state.clone()
        pos = st[:, px0:px1]
        r_old = self._block(lvl, "radius")[parents].clone()                # [cap]
        x_old = pos[parents].clone()                                       # [cap, d]

        # --- oriented placement direction: normalize(snr * a_hat + unit-RMS isotropic noise) --- #
        xi = torch.randn(cap, d, generator=rng, device=dev) / math.sqrt(d)  # unit-RMS isotropic
        axis = getattr(lvl, "division_axis", None)
        if axis is not None:
            a = axis[parents]
            a_hat = a / (a.norm(dim=1, keepdim=True) + 1e-12)               # zero axis -> a_hat = 0 -> isotropic
            biased = self.orientation_snr * a_hat + xi
        else:
            biased = xi                                                    # no axis field -> pure isotropic
        dirv = biased / (biased.norm(dim=1, keepdim=True) + 1e-12)
        offset = (r_old * m)[:, None] * dirv                               # offset uses the NEW radius r*m

        # --- inherit every heritable per-cell buffer into the daughter slots ------------------- #
        # (celltype, division_rate, division_axis, radius, node_type, ...). born/mother are the
        # lineage record we set explicitly below, so skip them here.
        st[slots] = st[parents].clone()
        for name, b in lvl.per_node_buffers():
            # `state` is itself a registered per-node buffer, so it comes back from
            # per_node_buffers() -- and writing it here would be an IN-PLACE write on
            # `lvl.state`, the tensor this operator has just cloned into `st` and is about to
            # replace. Forward-only that is invisible (the write is superseded a few lines
            # later); with a gradient it is fatal, because an upstream operator that read a
            # SLICE of `state` this tick has that view on the tape. `regulate` writing an
            # integrated `growth_rate` is exactly such an operator, and the composed
            # sense->regulate->grow spec is what surfaced it. The state copy is already done
            # on the clone by the line above.
            if name in ("born", "mother", "state"):
                continue
            b[slots] = b[parents].clone()

        # --- volume-conserving radii: both daughters -> r*m (each half the mother's d-volume) --- #
        self._set_block_on(lvl, st, "radius", parents, r_old * m)
        self._set_block_on(lvl, st, "radius", slots, r_old * m)

        # --- symmetric just-touching placement: mother -> x+offset, daughter -> x-offset ------- #
        pos[parents] = x_old + offset
        pos[slots] = x_old - offset

        # --- wake the daughter slot and record lineage ----------------------------------------- #
        # occ is CLONED before the write. `grow_radius` multiplies its delta by
        # `lvl.occ[:, None]`, so the live mask is on the autograd tape; mutating it in
        # place invalidates that multiply and the whole rollout stops being
        # differentiable. Found by grad_probe.py, not by any forward run -- a forward run
        # cannot tell the two apart.
        occ = lvl.occ.clone()
        occ[slots] = 1.0
        lvl.occ = occ
        if hasattr(lvl, "birth"):
            lvl.birth[slots] = 1.0                                         # Plexus occupancy baseline (mass-trigger; harmless here)
        lvl.born[slots] = 1.0
        lvl.mother[slots] = parents
        lvl.state = st                 # publish the clone: the functional update, completed
        return {}
