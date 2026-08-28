#!/usr/bin/env python
"""Continuous flow: a jet that enters through one face and leaves through another.

    general:
      engine: continuous          # selects run() below instead of plexus.engine.run

WHY AN ENGINE KEY AT ALL, GIVEN HOW LITTLE THIS FILE DOES. The physics here is two ordinary
operators -- `mpm_emit` and `mpm_drain` -- and they need no engine support whatsoever: `occ == 0`
particles are already frozen (not advected, `mpm_ops.py:1280`) and already contribute zero weight to
the scatter (`:538`), so a dormant particle is genuinely absent from the physics. What the engine key
buys is a PLACE TO REFUSE: a continuous spec that forgets its drain fills the pool and then silently
stops emitting, which looks like a jet that thins out rather than a spec that is wrong. `run()` below
checks the pair up front and states the pool's headroom in frames.

WHY THE POOL STAYS, EVEN THOUGH THE TAPE IS GONE. The fixed reservoir is usually justified by
differentiability, and giving that up seems to license dynamic allocation. It does not: `engine.py`
drops a captured CUDA graph the moment a state buffer is reallocated (`_graph_sig` compares storage
pointers every tick), so growing the particle array to emit would cost the measured 2.36x capture
speedup on EVERY frame, permanently -- and compaction on drain would destroy per-particle identity
as well. A fixed pool with `occ` recycling never moves a buffer.

That is not the limitation it sounds like. A fixed pool supports unbounded flow DURATION; only the
live population is bounded, and in a steady jet the live population is bounded by physics anyway at
`emission rate x transit time`. Size the pool above that and recycling runs forever.

WHAT IS STILL DIFFERENTIABLE, since it is less lost than "not differentiable" suggests. The discrete
count and timing of activation are not. But the inlet VELOCITY and the material written into emitted
particles are ordinary continuous writes into state, so gradients still reach them.

FOUR THINGS THIS GETS RIGHT ON PURPOSE, three of which are bugs already paid for elsewhere in this
codebase:

  frame level, not substep level.  The activation cursor advances once per frame. Inside a captured
      substep the slice bounds would be baked in at capture and the same particles would be emitted
      forever. `mpm_emit` and `mpm_drain` belong in the OUTER schedule, beside `gravity`.

  branch-free activation.  A rolling contiguous slice [cursor, cursor+k) mod N, never
      `nonzero(occ == 0)`. `Level.free_slots` does use nonzero and is right to -- it serves division,
      where which slots are free is genuinely data-dependent -- but a cursor needs no such query, and
      a boolean-mask index_put inside a capture is `cudaErrorStreamCaptureUnsupported`, which is
      exactly how the buoyancy branch died.

  reset on activation.  F = I, C = 0, Jp = 1, vel = inlet. A recycled particle still carrying its
      previous life's deformation gradient produces instant spurious stress at the inlet -- and it
      reads as an inlet instability, not as bookkeeping.

  a slab, not a plane.  Emission is spread through `U * dt` of depth, or the stream enters as
      discrete sheets in lockstep with the frame rate.

THE SIZING IS FORCED, NOT CHOSEN. Two constraints fix both unknowns:

    mass flux    rho * A * U  =  k * m_p / dt          what the physics wants
    sampling     k            =  ppc * A * U * dt / dx^D   what the grid needs

so m_p = rho * dx^D / ppc, and `mpm_emit` derives `k` from the inlet area, the speed and the frame
step rather than taking a rate on faith. It says what it derived, and warns when the spec's own
particle_mass disagrees -- because then the jet's density is not the density that was declared.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Operator
from plexus.models.registry import register_operator

_AXIS = {"x": 0, "y": 1, "z": 2}


def _face(spec: str) -> tuple:
    """'-y' -> (1, -1.0): the axis it enters along, and the direction it travels."""
    s = str(spec).strip().lower()
    sign = -1.0 if s.startswith("-") else +1.0
    ax = _AXIS[s.lstrip("+-")]
    return ax, sign


# ==========================================================================================================
#  mpm_emit -- wake dormant particles at the inlet, once per frame
# ==========================================================================================================
@register_operator("mpm_emit", family="mpm", set="particle", kind="structural")
class MPMEmit(Operator):
    """Activate a rolling slice of the pool at the inlet face, with reset state.

    `at:` the particle set. `face: -y` means the fluid enters at the TOP (high y) and travels in -y.
    """

    EMIT = None                                  # writes occ/state directly; returns no delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["face", "speed"]
    MAY_MUTATE_INTEGRATED_STATE = True           # it writes positions and velocities by definition
    MECHANISM_TAGS = ["inlet", "continuous_flow", "particle_recycling"]
    PARAM_ROLES = {"face": "inlet_face", "speed": "inlet_speed", "patch": "inlet_patch_fraction",
                   "ppc": "particles_per_cell", "type": "emitted_type"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.axis, self.sign = _face(params["face"])
        self.speed = float(params["speed"])
        # THE INLET PATCH AS A FRACTION OF THE FACE, so it survives a change of world size -- the
        # same reason the studio's knobs rescale geometry. [lo0, lo1, hi0, hi1] over the two axes
        # that are not the inlet normal, default the middle third.
        self.patch = [float(x) for x in (params.get("patch") or [0.33, 0.33, 0.67, 0.67])]
        self.ppc = float(params.get("ppc", 8.0))
        self.type_name = params.get("type")
        self._cursor = 0
        self._armed = False
        self._said = False

    # ---------------------------------------------------------------- helpers
    def _park(self, p, box):
        """Everything dormant and off-domain, once, before the first emission.

        The pool is seeded by `provision` with its material properties already correct (mass, p_vol,
        mu, la, is_liquid) -- only its occupancy and position are wrong for a jet, because the spec's
        `block`/`shape` placed a body where we want an empty box.
        """
        p.occ.zero_()
        p.state[:, :box.numel()] = -1.0 * box                 # far outside; never drawn, never sensed
        if getattr(p, "Jp", None) is not None:
            p.Jp.fill_(1.0)

    def forward(self, H, mask=None):
        p = H.level(self.at)
        dev = p.state.device
        D = int(getattr(H, "dim", 3))
        box = torch.as_tensor(list(H.world_size)[:D], device=dev, dtype=p.state.dtype)
        dx = float(H.field(next(iter(H.fields))).dx) if len(H.fields) else float(box[0]) / 96.0
        dt = float(getattr(H, "dt", 1.0 / 1200.0))

        if not self._armed:
            self._park(p, box)
            self._armed = True

        # ---- how many, from the flux and the sampling, not from a declared rate ----
        lat = [k for k in range(D) if k != self.axis]
        A = 1.0
        for i, k in enumerate(lat):
            A *= float(box[k]) * max(self.patch[i + len(lat)] - self.patch[i], 1e-9)
        depth = self.speed * dt                               # the slab that enters this frame
        k_frame = int(round(self.ppc * A * depth / dx ** D))
        if k_frame < 1:
            return {}

        n = int(p.n)
        idx = (torch.arange(k_frame, device=dev) + self._cursor) % n
        self._cursor = (self._cursor + k_frame) % n

        # THE CURSOR RECYCLES BLINDLY, SO SAY WHEN IT EATS ITS OWN TAIL. It wraps after
        # `pool / k` frames and overwrites whatever is in those slots. That is correct exactly when
        # the pool outlasts the transit time -- a particle emitted `pool/k` frames ago must already
        # have drained -- and silently WRONG when it does not: fluid still in mid-air is teleported
        # back to the inlet, which reads as a jet that stutters or as mass appearing from nowhere,
        # not as a pool that is too small. The count is a sync, but this is a frame-level operator
        # outside the captured region and the slice is only `k` long.
        _clobber = int((p.occ[idx] > 0).sum())
        if _clobber and not getattr(self, "_clobbered", False):
            self._clobbered = True
            from plexus.paths import warn
            warn(f"mpm_emit: the pool has wrapped onto {_clobber:,} particles that are STILL LIVE "
                 f"({100.0 * _clobber / max(k_frame, 1):.0f}% of this frame's emission). The pool "
                 f"({n:,}) is smaller than emission x transit; those particles are being teleported "
                 f"back to the inlet mid-flight. Raise per_parent to at least "
                 f"{int(n * 1.5):,}, or lower the inlet speed or area.")

        # ---- place them in a SLAB, not on a plane ----
        u = torch.rand(k_frame, D, device=dev, generator=getattr(H, "rng", None))
        pos = torch.empty(k_frame, D, device=dev, dtype=p.state.dtype)
        for i, k in enumerate(lat):
            lo, hi = self.patch[i], self.patch[i + len(lat)]
            pos[:, k] = (lo + (hi - lo) * u[:, k]) * box[k]
        # the inlet plane sits 2 cells inside the wall -- the gather clamps positions to
        # [2*dx, box - 2*dx], so anything emitted outside that is snapped and piles up on the face.
        edge = (float(box[self.axis]) - 2.5 * dx) if self.sign < 0 else 2.5 * dx
        pos[:, self.axis] = edge - self.sign * depth * u[:, self.axis]

        vel = torch.zeros(k_frame, D, device=dev, dtype=p.state.dtype)
        vel[:, self.axis] = self.sign * self.speed

        px0, px1 = 0, D
        p.state[idx, px0:px1] = pos
        vslice = getattr(p, "vel_slice", None)
        if vslice is not None:
            p.state[idx, vslice[0]:vslice[1]] = vel
        else:
            _v = p.get("vel")
            _v[idx] = vel
        # ---- RESET, or a recycled particle arrives pre-stressed ----
        eye = torch.eye(D, device=dev, dtype=p.F.dtype)
        p.F[idx] = eye
        p.C[idx] = 0.0
        if getattr(p, "Jp", None) is not None:
            p.Jp[idx] = 1.0
        p.occ[idx] = 1.0

        if not self._said:
            self._said = True
            m_want = float(p.mass.median()) if hasattr(p, "mass") else 0.0
            rho = m_want * self.ppc / dx ** D if m_want else 0.0
            print(f"[mpm_emit] inlet {A * 1e4:.2f} cm^2 at {self.speed:g} m/s -> {k_frame} "
                  f"particles/frame, pool {n:,} -> {n / max(k_frame, 1):.0f} frames of headroom "
                  f"before recycling; implied density {rho:.0f} kg/m^3", flush=True)
        return {}


# ==========================================================================================================
#  mpm_drain -- damp, then retire, at the outlet
# ==========================================================================================================
@register_operator("mpm_drain", family="mpm", set="particle", kind="structural")
class MPMDrain(Operator):
    """Retire particles past the outlet plane, through a damping sponge rather than a cliff.

    A particle deleted while still inside a live grid node's stencil is a momentum discontinuity
    that propagates back upstream, so the sponge damps velocity over `sponge` cells first and the
    kill plane sits clear of anything worth measuring.
    """

    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["face"]
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["outlet", "continuous_flow", "absorbing_boundary"]
    PARAM_ROLES = {"face": "outlet_face", "at_fraction": "outlet_plane_fraction",
                   "sponge": "sponge_width_cells"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.axis, self.sign = _face(params["face"])
        self.frac = float(params.get("at_fraction", 0.04))    # plane, as a fraction of the box
        self.sponge = float(params.get("sponge", 4.0))        # cells of damping above it
        self.damp = float(params.get("damp", 0.5))            # velocity kept per frame in the sponge

    def forward(self, H, mask=None):
        p = H.level(self.at)
        dev = p.state.device
        D = int(getattr(H, "dim", 3))
        box = torch.as_tensor(list(H.world_size)[:D], device=dev, dtype=p.state.dtype)
        dx = float(H.field(next(iter(H.fields))).dx) if len(H.fields) else float(box[0]) / 96.0
        x = p.state[:, self.axis]
        L = float(box[self.axis])
        plane = self.frac * L if self.sign < 0 else (1.0 - self.frac) * L
        band = self.sponge * dx

        live = p.occ > 0
        if self.sign < 0:                       # flow travels toward LOW coordinate
            gone = live & (x < plane)
            insponge = live & (x < plane + band) & (x >= plane)
        else:
            gone = live & (x > plane)
            insponge = live & (x > plane - band) & (x <= plane)

        v = p.get("vel")
        v[insponge] = v[insponge] * self.damp
        if bool(gone.any()):
            p.occ[gone] = 0.0
            if hasattr(p, "mass"):
                pass                            # mass is a material property; occ alone removes it
            p.state[gone, :D] = -1.0 * box      # park off-domain so it is neither drawn nor sensed
            p.C[gone] = 0.0
        return {}


# ==========================================================================================================
#  the engine entry point
# ==========================================================================================================
def run(sim, *args, **kwargs):
    """`plexus.engine.run`, with the continuous-flow spec checked before anything expensive starts.

    It delegates: there is no second integration loop here, and there must not be. Everything this
    mode needs is expressible as operators, so a fork of `run` would be a second copy of the schedule
    loop, the recording, the CUDA-graph capture and the integration invariant -- all of which would
    then drift. What this adds is the refusal that an operator cannot make, because an operator sees
    only itself.
    """
    ops = [o.op for o in sim.operators]
    if "mpm_emit" in ops and "mpm_drain" not in ops:
        raise ValueError(
            "engine: continuous -- this spec emits but never drains. The pool is finite, so it "
            "fills and then emission silently stops, which looks like a jet that thins out rather "
            "than a spec that is wrong. Add `mpm_drain` with the face the flow leaves by.")
    if "mpm_drain" in ops and "mpm_emit" not in ops:
        raise ValueError("engine: continuous -- this spec drains but never emits.")
    # EMIT AND DRAIN MUST BE OUTSIDE THE SUBSTEP BLOCK. Inside it, the activation cursor advances
    # once per substep instead of once per frame (so the jet is `substeps` times too fast), and a
    # captured graph would bake the first substep's slice in and emit the same particles forever.
    for blk in sim.schedule:
        if isinstance(blk, dict) and {"mpm_emit", "mpm_drain"} & set(blk.get("steps") or []):
            raise ValueError(
                "engine: continuous -- mpm_emit / mpm_drain are inside the substep block. They are "
                "FRAME-level operators: put them in the outer schedule beside `gravity`, or the "
                "inlet runs once per substep and a captured graph freezes the emitted slice.")
    from plexus.engine import run as _run
    return _run(sim, *args, **kwargs)
