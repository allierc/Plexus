"""MLS-MPM, as one module: the grid, the four-step cycle, and the two forces on it.

    mpm_grid            the background FIELD and the quadratic B-spline kernel (not an operator)
    mpm_scatter (p2g)   particle -> grid: mass, momentum, and the internal stress impulse
    mpm_grid_update     grid -> grid: the solve, gravity, and the wall conditions
    mpm_gather (g2p)    grid -> particle: velocity, the affine C, and advection
    mpm_strain          particle -> particle: F update and the material's response
    mpm_turgor          particle -> particle: an isotropic outward pressure (a cell against its cortex)
    mpm_anchor          a spring to a rest position, for a body that must not drift
    mpm_spin            a prescribed angular velocity
    apply_material_map  a per-particle material assignment from a map
    mls_mpm_mechanics   the FENCED transitional oracle: the whole cycle in one operator

THE ORACLE IS STILL HERE AND IS STILL FENCED. `mls_mpm_mechanics` does in one operator what the
four above do in four, and it exists so the decomposition can be checked against something. It is
not the recommended path and it is not what a new spec should schedule.

WHY THE GRID IS IN THE SAME FILE. `stencil_offsets`, `bspline` and `sub_dt` were imported from
`mpm_grid` by seven other files, so the kernel that defines the discretisation was a private
detail of one of nine siblings. Every MPM operator's substep -- and the CFL ceiling that bounds it,
dt < dx / sqrt(E/rho) -- is now readable in one place.

TWO REJECTED NEIGHBOURS ARE NOT HERE. `mpm_boundary` (kinematic, momentum not conserved, standoff
set by the stencil width) and `bm_strain` stay in discovery_okuda; see membrane_ops and AUDIT.md.
"""
from __future__ import annotations
import functools
import itertools
import torch
from plexus.models.base import Field
from plexus.models.registry import register_field
from plexus.models.base import Exchange
from plexus.models.registry import register_operator
# (was `from plexus.operators.mpm_grid import stencil_offsets, bspline, sub_dt`) -- same module now
from plexus.models.base import FieldUpdate
# (was `from plexus.operators.mpm_grid import sub_dt`) -- same module now
from plexus.models.base import Lateral
import os
from plexus.models.base import Field, Exchange
from plexus.models.registry import register_field, register_operator
from plexus.paths import graphs_data_path


# ==========================================================================================================
# FROM `discovery_okuda/ops/mpm_grid.py` -- mpm_grid -- the Eulerian background grid FIELD + the shared transfer kernel for the
# ==========================================================================================================
@functools.lru_cache(maxsize=None)
def _perm_index(device, perm) -> torch.Tensor:
    """A constant column permutation as a DEVICE tensor, built once. See `_polar_higham`."""
    return torch.tensor(list(perm), dtype=torch.long, device=device)


@functools.lru_cache(maxsize=None)
def _stencil_cached(dim: int, devstr: str) -> torch.Tensor:
    return torch.tensor(list(itertools.product(range(3), repeat=dim)),
                        dtype=torch.float32, device=devstr)


def _const_any(ob, attr, t) -> bool:
    """A RUN-CONSTANT tensor predicate, resolved ONCE per operator instance.

    `bool(t.any())` inside a python `if` is `.item()`: a GPU->CPU round trip that drains the
    launch queue and stalls the CPU that should be running ahead queueing kernels. There are
    seven of them across the four MPM operators -- "does this set contain liquid / snow /
    viscoelastic particles", "are there wall obstacles" -- and every one asks a question whose
    answer is fixed for the whole run: the material masks are written once at seeding
    (models/entities.py) and the obstacle set does not move.

    Twelve evaluations per substep on a two-set spec, measured at 19.7 ms of a 156 ms frame.
    They are also, on their own, the reason a substep cannot be stream-captured at all.

    FIRST-CALL CACHE rather than a constructor argument, because the operator is built before
    the level it acts on has been seeded. The first call still syncs; every later one does not,
    and the answer is identical either way -- so the eager path stays bit-for-bit unchanged.
    """
    v = getattr(ob, attr, None)
    if v is None:
        v = bool(t is not None and bool(t.any()))
        setattr(ob, attr, v)
    return v


def stencil_offsets(dim: int, device="cpu") -> torch.Tensor:
    """The 3^dim quadratic-B-spline stencil offsets, row-major (last axis fastest).
    2D -> [9,2] == `[[i,j] for i in 0..2 for j in 0..2]`; 3D -> [27,3] (matches the
    MPM_3D offset ordering: idx//9, (idx%9)//3, idx%3)."""
    # MEMOISED. Built from a python list, this is a PAGEABLE HOST->DEVICE COPY -- a host sync in
    # eager, and outright illegal inside a CUDA stream capture. It depends only on (dim, device)
    # and is called four times per substep, so it was rebuilding a constant 27x3 table 90 times a
    # frame. Measured at 19 ms of a 156 ms frame, by leaving THIS ONE call unmemoised with every
    # other host sync already gone -- which is the only way to measure it, because removing syncs
    # is all-or-nothing: any single survivor drains the launch queue and hides the rest.
    # The returned tensor is SHARED and must be treated as read-only; today every caller only
    # reads it (`offsets.long()`, arithmetic) and none mutates it in place.
    return _stencil_cached(int(dim), str(device))


# 2D stencil kept as a module constant for back-compat (p2g/g2p now build per-dim)
_OFFSETS = stencil_offsets(2)


@register_field("mpm_grid")
class MPMGrid(Field):
    """MLS-MPM background grid over the world box, with square cells dx = world_size[1] / n_grid.
    Channels: m (mass), mv (momentum [.,dim]), c (liquid colour for CSF), v (velocity
    [.,dim]). Pure scratch: p2g zeroes + scatters into it each substep, grid_update
    solves on it, g2p reads it back.

    THE CELL SIZE COMES FROM THE WORLD, which it did not before. This used to read
    `dx = 1.0 / n_grid` with axes 1.. assumed to span [0,1] and only axis 0 scaled by `width`, so
    `n_grid` meant "cells per unit length". On a box that is not 1 unit across -- a 0.1 m water
    scene, say -- that puts a 1.0-wide grid on a 0.1-wide world: the cell is 10x too large, only
    ~10 of 96 cells per axis carry any mass, and the failure is not subtle. `MPMGather` clamps
    positions into [2*dx, box[k] - 2*dx] with `box` correct and `dx` wrong, which crushes a 0.1 m
    cube into a 0.058 m slab on the first substep.

    `n_grid` now means CELLS ACROSS AXIS 1. The two readings coincide when world_size[1] == 1.0,
    and they always did here: of the 1,744 specs under config/, 152 are MPM, 94 declare
    `general.world` and ALL 94 are exactly [1.0, 1.0, 1.0]; the other 58 are 2D and omit it, which
    schema.py defaults to [1.0, 1.0]. So every existing spec keeps the identical dx and the
    identical node count, and no spec sets `width` on this field.

    inv_dx IS NOT 1.0/dx. `1.0 / (1.0 / n) != float(n)` for 640 of the first 4,096 integers -- the
    first offender is n = 49 -- so the reciprocal round-trip is exact only by luck at the n_grid
    values in use (48, 64, 96, 128, 192). Deriving it from the integer directly keeps byte-identity
    on the corpus, and keeps it for a spec that picks n_grid = 49 tomorrow.
    """

    RECORD = False                                   # transient scratch -- not recorded/rendered

    def __init__(self, name, width=1.0, n_grid=128, dim=2, device="cpu", world_size=None, **kw):
        super().__init__(name)
        self.dim = int(dim)
        # `width` is the legacy axis-0 scalar and stays the fallback: a caller that does not pass
        # the per-axis box gets exactly the old geometry, [width] x [1] x [1].
        box = [float(w) for w in world_size] if world_size else \
              [float(width)] + [1.0] * (self.dim - 1)
        if len(box) != self.dim:
            raise ValueError(f"MPMGrid: world_size has {len(box)} entries but dim={self.dim}")
        self.world_size = box
        self.width = box[0]
        self.n_grid = int(n_grid)                    # cells across axis 1
        self.inv_dx = float(n_grid) / box[1]         # NOT 1.0/dx -- see the note above
        self.dx = box[1] / float(n_grid)
        n_k = [max(1, int(round(box[k] * self.inv_dx))) for k in range(self.dim)]
        self.nx, self.ny = n_k[0], n_k[1]
        if self.dim == 2:
            self.shape = (self.nx, self.ny)
        else:
            self.nz = n_k[2]
            self.shape = (self.nx, self.ny, self.nz)
        n = 1
        for s in self.shape:
            n *= s
        self.n_cells = n
        self.register_buffer("m", torch.zeros(n, device=device))
        self.register_buffer("mv", torch.zeros(n, self.dim, device=device))
        self.register_buffer("c", torch.zeros(n, device=device))
        self.register_buffer("v", torch.zeros(n, self.dim, device=device))

    @property
    def grid(self):                                  # [1,*shape] view for the recorder (mass density)
        return self.m.view((1,) + self.shape)


def bspline(X, inv_dx, offsets, shape, periodic):
    """Quadratic B-spline weights of each particle over its 3^dim grid stencil.
    Returns (fx [N,D], weight [N,S], flat [N*S]) where S = 3^D. Dimension-generic; the
    2D call reduces bit-identically to the original `w[:,oi,0]*w[:,oj,1]`, `gx*ny+gy`."""
    D = X.shape[1]
    base = (X * inv_dx - 0.5).floor().long()                          # [N,D]
    fx = X * inv_dx - base.float()                                    # [N,D]
    w = torch.stack([0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1) ** 2,
                     0.5 * (fx - 0.5) ** 2], dim=1)                   # [N,3,D]
    oidx = offsets.long()                                            # [S,D]
    weight = torch.ones(X.shape[0], offsets.shape[0], device=X.device)
    for k in range(D):                                               # prod_k w[:, o_k, k]
        weight = weight * w[:, oidx[:, k], k]
    gpos = base[:, None, :] + oidx[None]                             # [N,S,D]
    comps = []
    for k in range(D):
        comps.append(gpos[..., k] % shape[k] if periodic
                     else gpos[..., k].clamp(0, shape[k] - 1))
    flat = comps[0]                                                  # row-major flatten over `shape`
    for k in range(1, D):
        flat = flat * shape[k] + comps[k]
    return fx, weight, flat.reshape(-1)


def sub_dt(H, fallback):
    """The current substep dt: the schedule's `{substep: N, dt}` loop sets `H.sub_dt`;
    fall back to the operator's own `dt_sub` param if run outside a substep loop."""
    sd = getattr(H, "sub_dt", None)
    return float(sd if sd is not None else fallback)


# ==========================================================================================================
# FROM `discovery_okuda/ops/mpm_scatter.py` -- mpm_scatter (was p2g) (particle -> mpm_grid): the MLS-MPM particle-to-grid scatter.
# ==========================================================================================================
def _polar_higham(F, iters=6):
    """Orthogonal polar factor R of F = R S, by Newton's iteration  R <- (R + R^-T)/2.

    Quadratically convergent from R0 = F whenever F is non-singular, which a valid
    deformation gradient is. Six iterations reach float32 on deformations of the size
    an eye muscle produces; the caller can ask for fewer.

    Only the ROTATION is wanted here -- the singular values the SVD also returns are
    not used by the fixed-corotated stress below -- so this is a drop-in, not an
    approximation of a different quantity.

    det R follows sign(det F), so an INVERTED particle would give an improper R where
    the SVD path forces a proper rotation. That case is caught rather than hidden: an
    inverted deformation gradient means the simulation has already failed.
    """
    D = F.shape[-1]
    R = F.clone()
    if D == 3:
        for _ in range(iters):
            # inverse-transpose by the ADJUGATE, not by a solve. `torch.linalg.solve`
            # RAISES on a singular batch element, and one degenerate particle out of
            # 58,200 then kills the whole run -- which is exactly what happened on the
            # SR staircase, 65 minutes in. The cofactor form cannot raise: the only
            # division is by the determinant, and clamping that away from zero leaves a
            # degenerate particle with a finite (meaningless) rotation instead of taking
            # the simulation down with it. A collapsed deformation gradient is a failure
            # to report, not a reason to lose the other 58,199.
            # `R[:, :, [1, 2, 0]]` LOOKS free and is not: a python list index builds a CPU
            # int tensor and copies it to the device, six times per call. In eager that is a
            # pageable H2D per iteration; inside a stream capture it is fatal, and measured as
            # exactly that -- list index FAILS with cudaErrorStreamCaptureInvalidated where a
            # cached device index tensor captures cleanly. index_select on a memoised index is
            # the same gather with the constant hoisted out.
            _i1 = _perm_index(R.device, (1, 2, 0))
            _i2 = _perm_index(R.device, (2, 0, 1))
            c = torch.cross(R.index_select(2, _i1), R.index_select(2, _i2), dim=1)
            det = (R[:, :, 0] * c[:, :, 0]).sum(1)[:, None, None]
            det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
            R = 0.5 * (R + c / det)
    else:
        eyeT = torch.eye(D, device=F.device, dtype=F.dtype)
        for _ in range(iters):
            R = 0.5 * (R + torch.linalg.solve(R, eyeT).transpose(-2, -1))
    return R


# ==========================================================================================================
# DIMENSIONAL CONSTANTS, EXPRESSED RELATIVE TO THEIR OWN NATURAL SCALE
# ==========================================================================================================
_REF_DX, _REF_RHO = 1.0 / 96, 1.0          # the box every one of these numbers was chosen against

#   name -> (historical value, exponent of density, exponent of length)
#   a mass is rho * L^3;  a length is L;  a reciprocal length is L^-1.
_CONST_DIMS = {
    "mass_floor":     (1e-10, 1, 3),
    "csf_mass_floor": (1e-8, 1, 3),
    "ring":           (0.04, 0, 1),
    "csf_eps":        (1e-6, 0, -1),
}


def _scale_constant(name, dx, rho=1.0):
    """The value a dimensional constant takes at this (dx, rho).

    WHY THESE ARE NOT NUMBERS. Each was chosen against a UNIT BOX at n_grid 96 with density 1, and
    each means something else the moment any of the three changes. `wall_contact: 0.04` -- the first
    one converted -- was the plain case: in a 0.1 m box it selected everything but a 0.02 m sliver,
    so the whole fluid read as permanently in wall contact and was permanently damped.

    The others are quieter and one of them is already costing accuracy. The CSF regulariser `eps` in
    n = grad(c)/(|grad c| + eps) has units of 1/LENGTH, and it is why `csf_rho` is not a pure gain
    rescaling: at the parity tension sigma = 120/192^2, material_two_drops_st reproduces only to
    2.376% RMS in CSF force, 32.7% once `csf_band` is on, because an absolute epsilon added to
    |grad c| bites differently once the colour is divided by rho*dx^D.

    CALIBRATED TO REPRODUCE THE HISTORICAL NUMBER EXACTLY at the reference, so no existing spec
    moves: at rho = 1 and dx = 1/96 both ratios are exactly 1.0 and the result is the original float,
    not a value close to it.

    NOT EVERY CONSTANT BELONGS HERE, and the ones left out are left out for a reason:
      a_max 200   an acceleration. The natural scale is `g`, which DIFFERS BETWEEN SPECS (14, 16,
                  9.81), so no single multiple reproduces them all -- converting it would be a
                  behaviour change dressed as a refactor.
      vmax 1e9    a velocity, and inert: the binding cap is min(vmax, 0.4*dx/dt) in the gather, so
                  the absolute default never applies.
      spin_k 30   a rate, and there is no natural time scale in the operator to divide it by.
    """
    v0, a, b = _CONST_DIMS[name]
    return v0 * (float(rho) / _REF_RHO) ** a * (float(dx) / _REF_DX) ** b


def _hand_body_force_to_grid(op, H, a_ext, dev, D):
    """WHERE A BODY FORCE BELONGS. Canonical MLS-MPM applies gravity ON THE GRID, as an
    acceleration, AFTER the momentum has been divided by nodal mass -- Taichi's mpm88/mpm99 read

        if grid_m[I] > 0:  grid_v[I] = (1 / grid_m[I]) * grid_v[I];  grid_v[I][1] -= dt * gravity

    This operator instead folds it into the PARTICLE velocity before P2G, so it rides through the
    scatter as momentum and comes back out of a division it never needed to be inside.

    IN EXACT ARITHMETIC THE TWO ARE THE SAME THING: sum_p w m (v + a dt) / sum_p w m = v_bar + a dt,
    because the mass cancels. The placement only starts to matter when something breaks that
    cancellation, and exactly one thing does -- `gm.clamp(min=mass_floor)` in the grid solve. On a
    node lighter than the floor the whole momentum, gravity's share included, is scaled by
    gm/mass_floor < 1, and a particle there cannot accelerate.

    MEASURED, before assuming that is what ails anything: a lone block dropped in an empty box
    accelerates at 0.9996 OF g with drag 0, and 0.9902 with the spec's Stokes drag 0.1 -- the
    1% being the drag, exactly. So on this spec family the floor does NOT bind and gravity IS
    delivered. Turning `mass_floor` down from 1e-10 to 1e-14 changes the suspended-particle count
    from 224 to 224 on material_3d_water_st000. This path is therefore the CORRECT ARCHITECTURE,
    not a cure for the haze, and it is offered as `body_force: grid` so the two can be compared
    rather than argued about.

    WHAT MOVES AND WHAT CANNOT. Only the PARENT-LEVEL acceleration -- gravity, buoyancy, anything
    emitted `at:` the parent set -- is a field quantity with a grid representation, and only when
    every parent carries the SAME vector (checked once, cached). Per-particle body forces (turgor,
    active stress, per-particle drag) have no grid representation and stay on the particle, where
    the clamp still reaches them; that is an argument for fixing the clamp, not for moving them.
    """
    if getattr(op, "body_force", "particle") != "grid":
        return a_ext
    if getattr(op, "_c_bf", None) is None:
        # a device sync, so it happens ONCE and is cached: is the parent acceleration uniform?
        amax = (a_ext - a_ext[:1]).abs().max() if a_ext.numel() else torch.zeros((), device=dev)
        op._c_bf = bool(float(amax) < 1e-12)
        if not op._c_bf:
            import warnings
            warnings.warn(
                "mpm_scatter: body_force='grid' needs a body force that is the same for every "
                "particle, and this one is not (per-particle accelerations have no grid "
                "representation). Falling back to the particle path for all of it.",
                RuntimeWarning, stacklevel=2)
        op._bf_buf = torch.zeros(D, device=dev)
    if not op._c_bf:
        return a_ext
    op._bf_buf.copy_(a_ext[0])              # persistent buffer -> safe inside a captured graph
    H._mpm_body_accel = op._bf_buf          # consumed by MPMGridUpdate after normalisation
    return a_ext - op._bf_buf                # particle keeps only what the grid cannot carry


@register_operator("mpm_scatter", "p2g", family="mpm", set="particle", kind="exchange")
class MPMScatter(Exchange):                 # (alias `p2g`, one migration cycle)
    EMIT = None                 # particle->grid: writes the mpm_grid field in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []        # no required params — `to` defaults to mpm_grid, all knobs optional
    REQUIRES_TYPE_PROPS = ["youngs"]
    # a liquid has no Young's modulus (nu -> 1/2 makes E = 3K(1-2nu) -> 0); it has a bulk modulus,
    # and for mu = 0 that IS lambda. Either spelling satisfies the requirement.
    TYPE_PROP_ALTERNATIVES = {"youngs": ("bulk_modulus",)}
    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress", "shared_grid_accumulate"]
    PARAM_ROLES = {"dt_sub": "MLS-MPM substep dt", "drag": "Stokes drag coefficient",
                   "a_max": "external-acceleration clamp",
                   "body_force": "body_force_application_site",
                   "store_stress": "cache Cauchy stress to a per-particle buffer"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM P2G); Sulsky, D. et al. (1994)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.to = params.get("to", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.drag = float(params.get("drag", 0.0))
        self.a_max = float(params.get("a_max", 200.0))
        # WHERE THE BODY FORCE IS APPLIED. "particle" (default, and every existing run) folds it
        # into the particle velocity before P2G; "grid" hands the uniform part to the grid solve,
        # which is where canonical MLS-MPM puts it. See _hand_body_force_to_grid for why the two
        # are algebraically identical until the mass clamp binds, and for the measurement showing
        # that on this spec family it does not.
        self.body_force = str(params.get("body_force", "particle"))
        if self.body_force not in ("particle", "grid"):
            raise ValueError(f"mpm_scatter: body_force must be 'particle' or 'grid', "
                             f"got {self.body_force!r}")
        # HOW THE POLAR ROTATION IS FOUND, in 3-D. The fixed-corotated stress needs R from
        # F = R S, and the obvious way to get it is an SVD -- but `torch.linalg.svd` on a
        # batch of 3x3 matrices costs about a microsecond EACH, and this operator runs once
        # per particle per substep: 45,000 particles x 25 substeps is 1.1 million 3x3 SVDs a
        # frame, and on the zebrafish eye that single call measured 44.7 ms of the operator's
        # 46.4 ms. "higham" replaces it with the Newton polar iteration
        # R <- (R + R^-T)/2, which converges quadratically from F and costs 6.4 ms for the
        # same batch -- 7x -- agreeing with the SVD rotation to 1.5e-6 with an orthogonality
        # error of 2.4e-7, i.e. to float32. Default stays "svd": identical numbers unless asked.
        # DEFAULT `higham`, CHANGED 24 AUGUST from `svd`. The polar factor R of the deformation
        # gradient is the same rotation either way; these are two numerical routes to it.
        #
        #   svd     `torch.linalg.svd` -> a cuSOLVER call. It is an EXTERN kernel: no compiler
        #           fuses it, and -- decisively -- it CANNOT BE CAPTURED into a CUDA graph, so a
        #           spec using it forfeits the substep capture that is worth ~2x on its own.
        #   higham  a Newton iteration built from matmuls and a cofactor cross product. Captures,
        #           fuses, and on a two-set spec with the host syncs already gone it is faster.
        #
        # WHAT THE CHANGE COSTS, measured on cell_02 (a bouncing elastic nucleus) rather than
        # asserted: max|diff| in position 2.4e-07 at 50 frames, 1.2e-06 at 200, 2.8e-05 at 600, and
        # the two centres of mass stay within 3.8e-07 -- 9e-06 of the body radius. Bounded, not
        # amplifying. 2D specs are UNAFFECTED: `forward` takes the analytic 2x2 rotation branch and
        # never consults this parameter at all.
        #
        # `polar: svd` remains available for a spec that must reproduce a stored 3D result byte for
        # byte. The promotion twins are unaffected either way, because okuda schedules these same
        # core operators -- both sides of the comparison move together.
        self.polar = str(params.get("polar", "higham")).lower()
        self.polar_iters = int(params.get("polar_iters", 6))
        # KEEP THE CAUCHY STRESS, OPTIONALLY. The fixed-corotated law below produces the Kirchhoff
        # stress tau = J.sigma, uses it to build the affine momentum matrix, and then overwrites the
        # variable with its dt-scaled form -- so the one tensor in the solver that says what the
        # material is actually carrying is computed 8,000 times a run and discarded every time. With
        # `store_stress: true` it is cached to a per-particle `sigma` buffer (Cauchy, i.e. tau/J) that
        # diagnostics and colourings can read instead of re-deriving a proxy from F.
        #
        # DEFAULT OFF, and the guard is what makes this safe to add to a shared operator: when off,
        # nothing is allocated and the only cost is one branch per substep. The cached value is read,
        # never written back, so the mechanics cannot be changed by asking for it.
        self.store_stress = bool(params.get("store_stress", False))

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.to); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        nx, ny, inv_dx, dx = g.nx, g.ny, g.inv_dx, g.dx
        D = p.F.shape[-1]
        periodic = bool(getattr(H, "periodic", False))
        offsets = stencil_offsets(D, dev)
        X, V = p.get("pos"), p.get("vel")
        # external per-cell acceleration from the parent set's accumulated delta (gravity)
        pn = getattr(p, "parent_name", None)
        if pn is not None:
            a_cell = H.delta(pn)
            a_cell = torch.nan_to_num(a_cell, posinf=self.a_max, neginf=-self.a_max).clamp(-self.a_max, self.a_max)
            a_ext = a_cell[p.parent]
        else:
            a_ext = torch.zeros(p.n, D, device=dev)
        part_accel = getattr(H, "part_accel", None)
        if part_accel is not None:
            a_ext = a_ext + part_accel
        # per-particle body force from particle-level force operators (e.g. pulse_to_contraction,
        # drag) -- the symmetric counterpart of the parent-delta path above (gravity).
        a_ext = a_ext + torch.nan_to_num(H.delta(p.name))
        a_ext = _hand_body_force_to_grid(self, H, a_ext, dev, D)
        V = V + dt * (a_ext - self.drag * V)                       # body force + Stokes drag (local; G2P resets V)

        F, C, mass = p.F, p.C, p.mass
        # RESIDUAL STRESS / PRESTRESS (optional, default OFF): compute the fixed-corotated stress
        # relative to a non-identity per-particle REST tensor F_res (multiplicative morphoelastic split
        # F = Fe . F_res, so Fe = F @ F_res_inv is the elastic part). At the mesh rest state F=I this
        # leaves Fe = F_res_inv != I -> a STANDING PRELOAD; an incompatible F_res(x,y) holds a
        # self-equilibrated residual-stress field. Absent buffer -> byte-identical; F_res=I (alpha=0) ->
        # F @ I = F exactly, so the operator truly ablates. Only the STRESS reference shifts; the
        # kinematic F (updated in mpm_strain) is untouched.
        Fres_inv = getattr(p, "F_res_inv", None)
        if Fres_inv is not None:
            F = F @ Fres_inv
        eye = torch.eye(D, device=dev).expand(p.n, D, D)
        if D == 2:
            a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
            J = a * d - b * c
        else:
            J = torch.linalg.det(F)
        mu, la = p.mu, p.la
        snow = getattr(p, "is_snow", None)
        if _const_any(self, "_c_snow", snow):                      # snow hardening from the plastic ratio Jp
            h = torch.exp((10.0 * (1.0 - p.Jp)).clamp(-6.0, 6.0))
            mu = torch.where(snow, p.mu * h, p.mu)
            la = torch.where(snow, p.la * h, p.la)
        if D == 2:                                                 # analytic 2D polar rotation (bit-identical)
            cs, sn = (F[:, 0, 0] + F[:, 1, 1]), (F[:, 1, 0] - F[:, 0, 1])
            r = torch.sqrt(cs * cs + sn * sn) + 1e-9
            cs, sn = cs / r, sn / r
            R = torch.stack([torch.stack([cs, -sn], -1), torch.stack([sn, cs], -1)], -2)
        elif self.polar == "higham":                               # Newton polar iteration
            R = _polar_higham(F, self.polar_iters)
        else:                                                      # SVD polar rotation R = U Vh (proper rotation)
            # THE SIGN FLIP IS A MULTIPLY, NOT A BOOLEAN MASK, and the two are bit-for-bit the
            # same: `torch.equal` True, max|diff| 0.000e+00 over 1,500 matrices. What changes is
            # the SHAPE of the work. `U[det(U) < 0, :, -1] *= -1` is `aten.nonzero` -- the number
            # of rows it writes depends on how many matrices are improper rotations, which is
            # ~50% here and drifts as the body deforms. Under torch.compile that is a
            # data-dependent shape: dynamo cuts the graph in half at this line and then recompiles
            # on every new count until it hits the recompile limit and falls back to eager for the
            # rest of the run ("___stack2 size mismatch at index 0, expected 13, actual 15").
            # Multiplying the last column by +-1 touches every row unconditionally, so the shape
            # is static and the scatter can be one graph -- which is also the precondition for
            # capturing the substep as a CUDA graph.
            #
            # IN PLACE ON THE CLONE, not `torch.cat` of the columns. Reassembling with `cat`
            # changes the memory layout, the following matmul picks a different kernel, and the
            # result moves by 6e-8 -- small, but enough to break the promotion suite's
            # byte-identity gate for no gain. Writing the column back keeps U contiguous.
            U, sig, Vh = torch.linalg.svd(F)
            U = U.clone(); Vh = Vh.clone()
            sgnU = torch.where(torch.det(U) < 0, -1.0, 1.0)
            sgnV = torch.where(torch.det(Vh) < 0, -1.0, 1.0)
            U[:, :, -1] = U[:, :, -1] * sgnU[:, None]
            Vh[:, -1, :] = Vh[:, -1, :] * sgnV[:, None]
            R = U @ Vh
        stress = 2 * mu[:, None, None] * ((F - R) @ F.transpose(-2, -1)) \
            + eye * (la * J * (J - 1))[:, None, None]
        # optional MLS-MPM ACTIVE STRESS (-A n n^T from pulse_to_active_stress), added to the
        # fixed-corotated elastic stress before the affine scatter. Default off (absent -> None ->
        # pure elastic); same units / scaling / scatter as the elastic stress. Same H side-channel
        # idiom as part_accel; it feeds the tissue through stress divergence, not a pointwise force.
        act = getattr(H, "active_stress", None)
        if act is not None:
            stress = stress + act
        # THE SAME CHANNEL, FOR A DIFFERENT MECHANISM. `extra_stress` is any additive Kirchhoff
        # stress written by an operator upstream in the substep -- today the Newtonian viscous
        # stress from `mpm_viscosity`. Kept separate from `active_stress` so the two compose
        # rather than clobber, and so a spec that reads `active_stress` back means what it says.
        xtr = getattr(H, "extra_stress", None)
        if xtr is not None:
            stress = stress + xtr
        # TURGOR / OSMOTIC PRESSURE (optional, default OFF): an isotropic OUTWARD pressure carried
        # by this set, written as a per-particle `turgor` buffer by the `mpm_turgor` operator. The
        # sign is the one that makes a cell a cell: tau here is Kirchhoff (positive = tension), a
        # fluid at pressure P has sigma = -P.I, so a POSITIVE `turgor` SUBTRACTS from tau and pushes
        # the material outward -- the same sign the liquid's own `la*J*(J-1)` takes when J < 1
        # (compressed -> pushes out). Absent buffer -> byte-identical; the buffer lives on the LEVEL,
        # so `mpm_turgor at: cytosol` pressurises the cytosol and leaves nucleus/membrane untouched
        # even though all three scatter into the same grid.
        turg = getattr(p, "turgor", None)
        if turg is not None:
            stress = stress - eye * turg[:, None, None]
        if self.store_stress:
            # CAUCHY, NOT KIRCHHOFF. What the lines above build is tau = J.sigma (the fixed-corotated
            # first Piola P times F^T), which is the form MLS-MPM scatters; sigma = tau / J is the
            # stress per unit CURRENT area, which is what "Cauchy stress" means and what a von Mises
            # invariant is normally quoted from. Captured here, after any active stress has been added
            # and BEFORE the dt / p_vol rescale on the next line, so it is the material's stress and
            # not a momentum increment.
            sig = stress / J.abs().clamp_min(1e-9)[:, None, None]
            if getattr(p, "sigma", None) is None or p.sigma.shape != sig.shape:
                p.register_buffer("sigma", torch.zeros_like(sig))
            p.sigma.copy_(sig.detach())
        stress = (-dt * 4 * inv_dx * inv_dx) * p.p_vol[:, None, None] * stress
        affine = stress + mass[:, None, None] * C

        fx, weight, flat = bspline(X, inv_dx, offsets, g.shape, periodic)
        # DORMANT particles (occ==0, e.g. a agent_grow reserve) contribute NOTHING to the grid:
        # mask the scatter weights by occupancy. Byte-identical when all particles are live.
        occ = getattr(p, "occ", None)
        if occ is not None:
            weight = weight * (occ > 0).to(weight.dtype)[:, None]
        dpos_phys = (offsets[None] - fx[:, None, :]) * dx
        mom = mass[:, None, None] * V[:, None, :] + (affine[:, None] @ dpos_phys[..., None]).squeeze(-1)
        # THE GRID IS SHARED BY EVERY SET THAT SCATTERS INTO IT, so only the FIRST scatter of a
        # micro-step may zero it. This used to allocate fresh zeros unconditionally and assign
        # them, which meant a spec with two particle sets kept only the LAST one: nucleus momentum
        # was deposited, then thrown away by the cytosol's scatter, and the grid solve ran on the
        # cytosol alone.
        #
        # WHAT IT LOOKED LIKE, because like the substep-binding defect it does not announce itself.
        # A nucleus falling inside a cytosol shell fell SLOWER than gravity (z = 0.7061 at the tick
        # free-fall puts at 0.6835) and was crushed from an rms thickness of 0.0248 to 0.0093 while
        # still in mid-air, where nothing was touching it. The same nucleus with no cytosol tracked
        # free-fall to four decimals and held 0.0248 exactly. It reads as "the nucleus is too soft"
        # and it is actually "the nucleus is gathering a velocity field it never contributed to":
        # inside the cavity the cytosol deposits no mass, so `gmv / gm.clamp(1e-10)` there is the
        # B-spline tails divided by nearly nothing.
        #
        # WHY IT SURVIVED THIS LONG. Every MPM spec in the corpus scatters ONE set -- multi-material
        # is done with `types:` inside a single `mpm_particle` set (see
        # config/material/material_3d_multimaterial.yaml: jelly + water + snow, one set, one
        # scatter). Multi-SET MPM is what the composed cell introduced, and it is the only thing
        # this ever affected. With one set the stamp is always stale and the zeros are always
        # fresh, so single-set runs stay bit-identical.
        # STEP 3: WHO ZEROES THE GRID IS STATIC, so it is not asked at run time. This used to read
        # `H.micro`, a python int the engine advances every substep, and compare it to a stamp on
        # the field. That works, but it is a per-substep side effect no CUDA-graph replay can
        # reproduce -- and it made dynamo recompile on every substep, because it specialises on
        # integer attributes of an nn.Module. The engine binds the i-th OCCURRENCE of a token to
        # the i-th INSTANCE, so occurrence 0 of `mpm_scatter` for a given grid is ALWAYS the first
        # one in a substep: the engine stamps `_zeroes_grid` on it when `inst` is built.
        _fresh = getattr(self, "_zeroes_grid", True)
        # STEP 2: WRITE INTO THE FIELD'S OWN BUFFERS, never a fresh allocation. Assigning
        # `g.m = torch.zeros(...)` rebinds the attribute to new storage every substep; a captured
        # graph holds the address it saw at capture time, so the replay silently writes somewhere
        # the rest of the run no longer reads. Measured: capture SUCCEEDS with this left as it was,
        # raises nothing, and produces a 6-tick checksum of 22816.79 against the correct 22132.22.
        gm, gmv, gc = g.m, g.mv, g.c
        if _fresh:
            gm.zero_(); gmv.zero_(); gc.zero_()
        gm.index_add_(0, flat, (weight * mass[:, None]).reshape(-1))
        gmv.index_add_(0, flat, (weight[..., None] * mom).reshape(-1, D))
        liquid = getattr(p, "is_liquid", None)
        if _const_any(self, "_c_liquid", liquid):                  # liquid colour for the CSF surface tension
            lw = (weight * (mass * liquid.to(mass.dtype))[:, None]).reshape(-1)
            gc.index_add_(0, flat, lw)
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/mpm_grid_update.py` -- mpm_grid_update (mpm_grid -> mpm_grid): the MLS-MPM grid solve.
# ==========================================================================================================
@register_operator("mpm_grid_update", family="mpm", set="field", kind="field")
class MPMGridUpdate(FieldUpdate):
    EMIT = None                                 # field->field grid solve: writes grid velocity in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []                        # no required params — all optional (grid from `at:`, engine-injected)
    MECHANISM_TAGS = ["grid_solve", "surface_tension", "boundary_conditions"]
    PARAM_ROLES = {"dt_sub": "substep_timestep", "surface_tension": "interface_cohesion",
                   "plate_gap_half": "free_half_gap", "plate_gap_half_end": "final_free_half_gap",
                   "plate_close_from": "frame_closing_starts", "plate_close_to": "frame_closing_ends",
                   "plate_axis": "confined_axis", "plate_centre": "gap_centre_on_axis",
                   "wall_damp": "wall_restitution", "csf_rho": "liquid_reference_density",
                   "csf_band": "interface_fraction_band", "csf_smooth": "colour_mollify_passes"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150; Sulsky, D. et al. (1994)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.surface_tension = float(params.get("surface_tension", 0.0))
        self.mass_floor = params.get("mass_floor", None)      # None -> derived from dx, rho
        # THE SAME DEFECT, A HUNDRED TIMES LARGER. The CSF surface-tension term converts a nodal
        # force to an acceleration with `dx^D / gm.clamp(min=1e-8)`, and 1e-8 is TEN THOUSAND times
        # the per-particle mass of a 10M-particle run (1.4e-09). Measured on that run at frame 500:
        # the floor binds on 59,932 of 122,868 occupied nodes -- 48.8% -- and scales the surface
        # tension there by a median factor of 0.056. Surface tension acts AT THE INTERFACE, which is
        # exactly where nodal mass is lowest, so the floor attenuates the term precisely where it is
        # supposed to work, and worse the more particles a spec uses. NOT, HOWEVER, WHY THE SIGMA
        # SWEEP DID NOTHING: that was measured to be the missing colour normalisation below, a factor
        # of 1/(rho*dx^D) ~ 1e6, against this floor's median 18x on half the nodes. Both are real;
        # they differ by five orders of magnitude, and setting `csf_rho` makes this floor inert.
        self.csf_mass_floor = params.get("csf_mass_floor", None)   # None -> derived
        # WHAT "COLOUR = 1" MEANS -- WHICH THIS BLOCK NEVER KNEW, AND THE REASON THE PARAGRAPH ABOVE
        # IS TREATING A SYMPTOM. `mpm_scatter` deposits `weight * mass * is_liquid`, so `gc` is a
        # liquid MASS PER NODE in absolute units, not the dimensionless volume fraction (0 in air,
        # 1 in liquid) that Brackbill's f = sigma * kappa * grad(c) is written for. On an all-liquid
        # spec the "colour" is bitwise the mass field: max |gc - gm| / gm over occupied nodes is
        # 2.7e-7 on material_3d_water_drop and 2.5e-7 on material_3d_water_dam_20m.
        #
        # Setting `csf_rho` to the liquid's density divides it by the mass of one FULL liquid cell,
        # rho * dx^D, and that single division is the repair. MEASURED COST OF NOT DOING IT: the
        # delivered acceleration is short by exactly 1/(rho * dx^D) -- 8.887e5 at n_grid 96 against
        # the predicted 96^3 = 884,736 (0.4%), and 7.356e6 at 192 against 192^3 = 7.078e6 (3.9%).
        # So `surface_tension` is not a tension: it is a tension divided by a cell mass, and its
        # PHYSICAL MEANING CHANGES WITH `n_grid`. `surface_tension: 60` on the dam at 192^3 is a
        # physical sigma of 8.5e-6 against rho*g*R^2 = 0.092 for its 0.076-half-width blocks --
        # BOND NUMBER 10,900, gravity beating capillarity by four orders of magnitude. That is the
        # whole content of "sigma 0 -> 150 moves the centre of mass by 0.0001 of the box".
        #
        # DEFAULT 0.0 KEEPS THE OLD PATH BIT-IDENTICAL, because the branch is a static Python `if`
        # and `range(0)` emits no ops, so CUDA-graph capture is unchanged. A spec that sets it MUST
        # re-fit its own sigma -- sigma_new = sigma_old * rho * dx^D for parity, or from a Bond
        # number for physics: rho*g*L^2 gives Bond 1 (0.64 on the drop at 96^3, 0.092 on the dam).
        self.csf_rho = float(params.get("csf_rho", 0.0))
        # THE INTERFACE TEST NEEDS THE SAME REFERENCE, and for want of it selects the bulk.
        # `gmag > 0.02 * gmag.max()` is a percentile of a running maximum with no absolute scale,
        # and the P2G shot noise at 8 particles per cell gives the BULK a |grad c| of the same order
        # as the interface's, so a 2% cut admits everything: measured 98.1% OF OCCUPIED NODES on
        # material_3d_water_drop and 97.7% on material_3d_water_dam_20m. It is an occupancy mask
        # wearing an interface's name. An absolute band on the volume fraction cuts it to 21.7% and
        # 25.8% of occupied at `csf_band: 0.2`.
        #
        # THE UPPER BOUND IS THE LOAD-BEARING HALF. At sigma 0.64 (Bond 1) on the drop, the band
        # 0.2 < c < 0.8 gives spread r90 0.311 with gm_max 2.2e-6 against one full cell's 1.1e-6 --
        # clean -- while `c > 0.1` with NO upper bound, which is what the relative threshold
        # effectively does, implodes to r90 0.0006 on 422 surviving occupied nodes. The bulk noise
        # force is harmless today only because everything is 1e6 too small; it detonates the moment
        # the gain is right, so the two parameters have to move together.
        self.csf_band = float(params.get("csf_band", 0.0))
        # MOLLIFY BEFORE THE SECOND DERIVATIVE -- worth little at 96^3 and a lot at 192^3. Band-median
        # kappa on a B-spline-splatted sphere with real 8-ppc shot noise, as a fraction of the exact
        # 2/R: normalised colour reads 1.07 unsmoothed / 1.02 with two passes at 96^3, and 1.27 /
        # 1.06 at 192^3. The SHIPPED raw colour reads 0.70 at 96^3 and 0.08 at 192^3, because
        # noise-driven |div n| grows as alpha/dx while the true curvature stays at 2/R. It does not
        # change the trajectory (r90 0.302 unsmoothed vs 0.306 smoothed at Bond 1) and it leaks
        # colour into empty cells, which is why the band below carries an explicit mass clause.
        self.csf_smooth = int(params.get("csf_smooth", 0))
        self.csf_eps = params.get("csf_eps", None)                 # None -> derived (a 1/length)
        # A BAND WITHOUT A REFERENCE DENSITY IS A SILENT OFF-SWITCH: `c` would still be a mass, of
        # order rho*dx^D ~ 1e-6, so `c > 0.2` is false everywhere, the mask is empty and the term
        # vanishes without a word. Refuse the combination rather than run it.
        # SAY SO, LOUDLY, RATHER THAN LET A MEANINGLESS NUMBER THROUGH. A spec that asks for surface
        # tension without a reference density gets the legacy path, which is what keeps old runs
        # reproducible -- but nothing about the yaml says the number is a tension divided by a cell
        # mass, and nothing warns that it will mean something different if `n_grid` is touched.
        if self.surface_tension > 0.0 and self.csf_rho <= 0.0:
            import warnings
            warnings.warn(
                f"mpm_grid_update: surface_tension={self.surface_tension:g} is being applied to an "
                f"UNNORMALISED colour field (csf_rho unset), so it is a tension divided by the mass "
                f"of one cell, rho*dx^D. It is ~1/(rho*dx^D) too small in physical units and its "
                f"meaning CHANGES WITH n_grid. Set csf_rho to the liquid's density and csf_band "
                f"(0.2) and re-fit sigma from a Bond number: sigma = rho*g*L^2 / Bond.",
                RuntimeWarning, stacklevel=2)
        if self.csf_band > 0.0 and self.csf_rho <= 0.0:
            raise ValueError(
                "mpm_grid_update: csf_band needs csf_rho -- the band is a test on the liquid "
                "VOLUME FRACTION, and without csf_rho the colour is still a mass (~rho*dx^D), so "
                "the band would select nothing and surface tension would be silently disabled. "
                "Set csf_rho to the liquid's density.")
        # BUOYANCY, IN THE SAME PLACE AND FOR THE SAME REASON AS SURFACE TENSION: it depends on the
        # LOCAL DENSITY, and only the grid knows what a particle is displacing.
        #
        # IT CANNOT BE A PER-PARTICLE FORCE HERE, and that was measured rather than assumed. The
        # `gravity` operator emits an ACCELERATION the substep consumes as `a_ext`, so every
        # particle receives the same `a` and mass cancels out of the grid velocity entirely. A
        # one-cell run with two liquid species at densities 0.6 and 1.8, mixed, gave a light-heavy
        # separation of -0.0020 over 1,200 frames -- zero, and faintly the wrong sign. Nothing
        # sorts under a uniform acceleration.
        #
        # THE FORM IS ARCHIMEDES ON THE NODE. With rho = m_node / dx^D, a node heavier than the
        # reference falls and a node lighter than it rises:
        #     dv = dt * g * (rho - rho_ref) / rho
        # so a node AT the reference density feels nothing, which is what makes this buoyancy and
        # not a second gravity. `rho_ref` is the fluid the body is displacing -- for a mixture, its
        # mean density -- and leaving it at 0 reduces the term to plain gravity, so the default is
        # off (`buoyancy: 0`) and no existing run changes.
        self.buoyancy = float(params.get("buoyancy", 0.0))
        self.rho_ref = float(params.get("rho_ref", 1.0))
        _bd = params.get("buoyancy_dir")
        self.buoy_dir = [float(x) for x in _bd] if _bd else None
        self.wall_damp = float(params.get("wall_damp", 1.0))
        # A MOVING PLATEN, AS A GRID VELOCITY BOUNDARY CONDITION.
        #
        # WHY IT CANNOT BE A POSITION PROJECTION, which is what `plate_confine` does and what the
        # first squash experiment used. In MPM the stress a material carries comes from its
        # deformation gradient F, and `mpm_strain` updates F from the VELOCITY GRADIENT read off
        # the grid. Teleporting a particle's position never enters that path: the constitutive
        # model is never told it has been compressed, so no stress builds, nothing pushes back,
        # and nothing bulges sideways. Measured on cell_12 -- half-height fell 2.8x while the
        # equatorial radius moved by less than 0.1% and the volume proxy r^2*h fell 3.0x. That is
        # a CROP, not a squash: matter displaced in z and never returned laterally.
        #
        # Imposed here instead, in the same place and the same way as the reflective domain walls:
        # clamp the normal grid velocity of every node beyond the plate to the PLATE'S OWN
        # velocity. Compression then enters the velocity field, F contracts in z, the elastic
        # model generates stress, and the lateral bulge and volume conservation come out of the
        # physics rather than being hoped for.
        #
        # A MOVING PLATEN, IMPOSED ON THE GRID. `plate_confine` projects particle POSITIONS after
        # the substep block, and a projection is invisible to the constitutive model: stress comes
        # from F, F is updated from the velocity gradient, and teleporting a particle changes
        # neither. Measured on cell_12: half-height fell 2.8x while the equatorial radius moved by
        # 0.1% and r^2*h -- volume -- fell 3x. The cell was being cropped, not squashed.
        #
        # A rigid obstacle cannot fix it either: `_walls3d` zeroes grid velocity inside a solid,
        # which stops material entering but cannot expel what is already there. A plate has to
        # CARRY a velocity, so the nodes it covers move at the plate's speed, the field near it is
        # compressive, F contracts along the axis, and the elastic response pushes material out
        # sideways. That is the difference between squashing and clipping.
        self.plate_axis = params.get("plate_axis", None)
        self.plate_axis = None if self.plate_axis is None else int(self.plate_axis)
        self.plate_centre = float(params.get("plate_centre", 0.5))
        self.plate_gap_half = float(params.get("plate_gap_half", 0.0) or 0.0)
        _pge = params.get("plate_gap_half_end", None)
        self.plate_gap_half_end = None if _pge is None else float(_pge)
        self.plate_close_from = int(params.get("plate_close_from", 0))
        self.plate_close_to = int(params.get("plate_close_to", 0))
        self._wall_key = None; self._wall_cache = None
        self._wall3d_key = None; self._wall3d_cache = None

    def _plate_bc(self, H, g, gv, dev, dt):
        """Two rigid plates closing along `plate_axis`, as a velocity condition on the grid.

        Frictionless and one-sided, the standard MPM rigid-body collision: a node the plate has
        reached may not move further into it, and is carried at the plate's own speed. Everything
        else -- the compression, the lateral bulge, the volume -- follows from the solve rather
        than being imposed.
        """
        ax = self.plate_axis
        gap, v_plate = self.plate_state(H)
        shape = g.shape
        idx = (torch.arange(shape[ax], device=dev) + 0.5) * g.dx
        shp = [1] * g.dim
        shp[ax] = shape[ax]
        coord = (idx - self.plate_centre).view(shp)                  # signed distance from the centre

        gvv = gv.view(*shape, g.dim)
        c = gvv[..., ax]
        # Top plate moves at -|v|, bottom at +|v|; a covered node is clamped so it cannot outrun
        # the plate outward. Clamping rather than assigning leaves material free to move INWARD
        # faster than the plate, which is what being squeezed out from between them looks like.
        top = coord >= gap
        bot = coord <= -gap
        c = torch.where(top, c.clamp(max=-abs(v_plate)), c)
        c = torch.where(bot, c.clamp(min=abs(v_plate)), c)
        gvv[..., ax] = c
        return gvv.view(g.n_cells, g.dim)

    def plate_state(self, H):
        """(current half-gap, plate closing speed in world units per second)."""
        g0, g1 = self.plate_gap_half, self.plate_gap_half_end
        if g1 is None or self.plate_close_to <= self.plate_close_from:
            return g0, 0.0
        span = self.plate_close_to - self.plate_close_from
        # THE FRAME IS READ AS A TENSOR, and that is not a stylistic choice. A captured CUDA graph
        # bakes python constants in at capture time, so a gap computed from `int(H.frame)` freezes
        # at whatever it was on the capture tick: the plates would sit still for the whole run and
        # nothing would say so -- the same silent shape as the `H.micro` recompile defect.
        # `H.frame_t` is a device scalar the engine fills in place each tick, OUTSIDE the graph,
        # so the kernels read the current value on every replay. Everything downstream of `u` is
        # therefore a tensor op; `torch.where` rather than a python branch for the same reason.
        fr = getattr(H, "frame_t", None)
        if fr is None:                                  # engine too old / called outside run()
            fr = torch.as_tensor(float(getattr(H, "frame", 0)))
        u = ((fr.double() - self.plate_close_from) / span).clamp(0.0, 1.0)
        gap = g0 + u * (g1 - g0)
        # Per FRAME in the schedule, per SECOND on the grid: the grid solve advances by a substep
        # dt, so the closing rate has to be divided by the frame's own dt to become a velocity.
        # Zero once the plates have arrived, or they keep pushing on a gap that no longer changes.
        dt_frame = float(getattr(H, "dt", 0.0) or 0.0) or 1.0
        rate = ((g1 - g0) / span) / dt_frame
        v = torch.where(u < 1.0, torch.full_like(u, rate), torch.zeros_like(u))
        return gap, v

    def _walls3d(self, H, g, dev):
        """Rasterize 3D obstacles onto the grid (cached). Formats: a box
        [x0,y0,z0,x1,y1,z1] (6 values) or a sphere [cx,cy,cz,r] (4 values). Returns a
        flat [n_cells] bool mask of solid (obstacle-occupied) cells."""
        key = g.shape
        if self._wall3d_key == key and self._wall3d_cache is not None:
            return self._wall3d_cache
        nx, ny, nz = g.shape
        walls = torch.zeros(nx, ny, nz, dtype=torch.bool, device=dev)
        obs = list(getattr(H, "obstacles", []) or [])
        if obs:
            xs = (torch.arange(nx, device=dev) + 0.5) * g.dx
            ys = (torch.arange(ny, device=dev) + 0.5) * g.dx
            zs = (torch.arange(nz, device=dev) + 0.5) * g.dx
            gx = xs[:, None, None]; gy = ys[None, :, None]; gz = zs[None, None, :]
            for o in obs:
                v = [float(x) for x in o]
                if len(v) == 6:                                  # box [x0,y0,z0,x1,y1,z1]
                    walls = walls | ((gx >= v[0]) & (gx <= v[3]) & (gy >= v[1]) & (gy <= v[4])
                                     & (gz >= v[2]) & (gz <= v[5]))
                elif len(v) == 4:                                # sphere [cx,cy,cz,r]
                    walls = walls | (((gx - v[0]) ** 2 + (gy - v[1]) ** 2 + (gz - v[2]) ** 2) <= v[3] ** 2)
        walls = walls.reshape(-1)
        self._wall3d_key = key; self._wall3d_cache = walls
        return walls

    def _walls(self, H, g, dev):
        key = (g.nx, g.ny)
        if self._wall_key == key and self._wall_cache is not None:
            return self._wall_cache
        walls = torch.zeros(g.nx, g.ny, dtype=torch.bool, device=dev)
        obs = list(getattr(H, "obstacles", []) or [])
        if obs:
            xs = (torch.arange(g.nx, device=dev) + 0.5) * g.dx
            ys = (torch.arange(g.ny, device=dev) + 0.5) * g.dx
            gx = xs[:, None].expand(g.nx, g.ny); gy = ys[None, :].expand(g.nx, g.ny)
            for rect in obs:
                v = [float(x) for x in rect]
                if len(v) == 4:
                    walls = walls | ((gx >= v[0]) & (gx <= v[2]) & (gy >= v[1]) & (gy <= v[3]))
                elif len(v) == 3:
                    walls = walls | (((gx - v[0]) ** 2 + (gy - v[1]) ** 2) <= v[2] ** 2)
        walls = walls.reshape(-1)
        self._wall_key = key; self._wall_cache = walls
        return walls

    def _wall_bc_2d(self, gv, nx, ny, dev, wd):
        """Reflective domain walls, 2D. EXTRACTED VERBATIM so `default` is unchanged, and made a
        method so an implementation can replace THIS and nothing else.

        It is 8 of the 113 aten calls a 2D grid solve issues and 66% of a 2D frame, because
        `gv[lox, :, 0] = ...` is boolean-mask indexing: it lowers to `index_put_`, which goes
        through `nonzero`, which SYNCHRONISES. Eight of those per call at 18 substeps is ~144
        pipeline drains a frame. The 3D branch expresses the identical condition with `torch.where`
        on broadcast masks and has none -- see `mpm_grid_update[implementation: nosync]`.
        """
        gv = gv.view(nx, ny, 2)
        ix = torch.arange(nx, device=dev); iy = torch.arange(ny, device=dev); bnd = 3
        lox, hix = ix < bnd, ix > nx - bnd
        loy, hiy = iy < bnd, iy > ny - bnd
        gv[lox, :, 0] = gv[lox, :, 0].clamp(min=0); gv[hix, :, 0] = gv[hix, :, 0].clamp(max=0)
        gv[:, loy, 1] = gv[:, loy, 1].clamp(min=0); gv[:, hiy, 1] = gv[:, hiy, 1].clamp(max=0)
        if wd != 1.0:
            gl = gv[lox, :, 1]; gv[lox, :, 1] = torch.where(gl > 0, gl * wd, gl)
            gh = gv[hix, :, 1]; gv[hix, :, 1] = torch.where(gh > 0, gh * wd, gh)
            gv[:, loy, 0] = gv[:, loy, 0] * wd
            gv[:, hiy, 0] = gv[:, hiy, 0] * wd
        return gv.view(nx * ny, 2)

    def _const(self, name, g):
        """A declared value wins; otherwise the constant is derived from THIS grid's cell size and
        the liquid's own density. `csf_rho` is that density when it is given (it is, by definition,
        the divisor that turns deposited mass into a volume fraction); `rho_ref` otherwise, whose
        default 1.0 is exactly the reference these numbers were chosen against."""
        v = getattr(self, name)
        if v is not None:
            return float(v)
        rho = self.csf_rho if self.csf_rho > 0.0 else self.rho_ref
        return _scale_constant(name, float(g.dx), rho)

    def forward(self, H, mask=None):
        g = H.field(self.at); dev = g.m.device
        dt = sub_dt(H, self.dt_sub)
        _mass_floor = self._const("mass_floor", g)
        _csf_floor = self._const("csf_mass_floor", g)
        _csf_eps = self._const("csf_eps", g)
        nx, ny, inv_dx, dx = g.nx, g.ny, g.inv_dx, g.dx
        D = g.dim
        periodic = bool(getattr(H, "periodic", False))
        wd = self.wall_damp
        gm, gmv, gc = g.m, g.mv, g.c
        # THE MASS FLOOR IS ABSOLUTE AND THE PARTICLE MASS IS NOT. `p_vol = body_volume /
        # per_parent`, so per-particle mass falls as 1/N: 1.5e-08 at 945k particles, 1.4e-09 at
        # 10M, 1.4e-10 at 100M. A node carrying one particle at a typical tail weight of the
        # B-spline holds 1.05e-09 at 945k -- comfortably above the floor -- and 9.9e-11 at 10M,
        # BELOW it. Once the floor binds, `gmv / 1e-10` is smaller than the true `gmv / gm`, so an
        # isolated particle's gathered velocity is scaled DOWN every substep and it can never
        # accelerate under gravity. It hangs in mid-air, which is what a 10M run showed: 21,826
        # particles suspended up to y = 0.60 above a bulk surface at y = 0.115, descending 0.13 in
        # 3.6 s where free fall at g = 16 would be 104.
        #
        # Kept as a PARAMETER at its historical value so no existing run changes; a spec whose
        # particle mass has outgrown it sets `mass_floor` smaller.
        gv = gmv / gm.clamp(min=_mass_floor)[:, None]
        # THE BODY FORCE, IF mpm_scatter HANDED IT OVER (`body_force: grid`). Applied here and not
        # in the scatter because this is after the division by nodal mass: as a pure addition to a
        # velocity it carries no mass factor, so `gm.clamp` cannot attenuate it and a node holding
        # one lone particle receives exactly the same `dt * a` as a node in the bulk. That is the
        # canonical MLS-MPM ordering (Hu et al. 2018; Taichi mpm88/mpm99 apply gravity immediately
        # after `grid_v = (1/grid_m) * grid_v` and before the wall conditions, which is the order
        # kept here). Absent the handover the attribute is None and this is not even a tensor op.
        _bf = getattr(H, "_mpm_body_accel", None)
        if _bf is not None:
            gv = gv + dt * _bf

        if D == 2:                                                  # --- 2D: verbatim (bit-identical) ---
            surf = self.surface_tension
            # CACHED, not dropped. Unlike the material masks this one is not run-constant in
            # PRINCIPLE -- gc is rebuilt every substep -- but it is settled by the first grid
            # solve, which always runs after every scatter of the substep. Deleting the guard
            # instead would turn gv into gv + 0.0 on a no-liquid spec and flip -0.0 to +0.0.
            if getattr(self, "_c_csf", None) is None:
                self._c_csf = bool(surf > 0.0 and bool((gc > 0).any()))
            if self._c_csf:                                        # CSF continuum surface force
                # SAME MISSING REFERENCE AS 3D, SAME REPAIR, and the 2D corpus is the reason it is
                # opt-in rather than applied: material_two_drops_st runs g = 0, so its Bond number
                # is ZERO and a tension 1/(rho*dx^2) = 36,864x too small is still the only force in
                # the scene -- measured Rg 0.13406 -> 0.10341 and r90 0.19205 -> 0.14136 over 4000
                # frames, against a sigma = 0 twin that does not move at all (Rg identical to five
                # decimals for 601 frames). That demo is correct AS WRITTEN and must stay so.
                # What the defect costs is scenes where gravity competes, and the resolution
                # dependence: the same yaml number is a different physical tension at every n_grid.
                if self.csf_rho > 0.0:
                    c = (gc / (self.csf_rho * dx * dx)).view(nx, ny)   # liquid VOLUME FRACTION
                else:
                    c = gc.view(nx, ny)                                # legacy: raw liquid MASS
                for _ in range(self.csf_smooth):        # separable [1/4,1/2,1/4]; no-op when 0
                    for k in range(2):
                        c = 0.25 * torch.roll(c, 1, k) + 0.5 * c + 0.25 * torch.roll(c, -1, k)
                cx = (torch.roll(c, -1, 0) - torch.roll(c, 1, 0)) * (0.5 * inv_dx)
                cy = (torch.roll(c, -1, 1) - torch.roll(c, 1, 1)) * (0.5 * inv_dx)
                gmag = torch.sqrt(cx * cx + cy * cy); eps = _csf_eps
                nxg, nyg = cx / (gmag + eps), cy / (gmag + eps)
                kappa = -((torch.roll(nxg, -1, 0) - torch.roll(nxg, 1, 0)) * (0.5 * inv_dx)
                          + (torch.roll(nyg, -1, 1) - torch.roll(nyg, 1, 1)) * (0.5 * inv_dx))
                _gain2 = 1.0 / max(1.0 - 2.0 * self.csf_band, 1e-6) if self.csf_band > 0 else 1.0
                if self.csf_band > 0.0:
                    _mfull = self.csf_rho * dx * dx
                    fmask = ((c > self.csf_band) & (c < 1.0 - self.csf_band)
                             & (gm.view(nx, ny) > self.csf_band * _mfull)).to(c.dtype) * _gain2
                else:
                    fmask = (gmag > 0.02 * gmag.max()).to(c.dtype)
                stfx = (surf * kappa * cx * fmask).view(-1); stfy = (surf * kappa * cy * fmask).view(-1)
                inv_m = (dx * dx) / gm.clamp(min=_csf_floor)
                gv = gv + dt * torch.stack([stfx * inv_m, stfy * inv_m], dim=1)

            if not periodic:
                gv = self._wall_bc_2d(gv, nx, ny, dev, wd)
            walls = self._walls(H, g, dev)
            if wd != 1.0 and _const_any(self, "_c_walls2d", walls):  # friction in cells touching obstacles
                w2 = walls.view(nx, ny)
                near = (torch.roll(w2, 1, 0) | torch.roll(w2, -1, 0)
                        | torch.roll(w2, 1, 1) | torch.roll(w2, -1, 1)) & ~w2
                gvv = gv.view(nx, ny, 2); gx_ = gvv[..., 0]; gy_ = gvv[..., 1]
                gvv[..., 0] = torch.where(near, gx_ * wd, gx_)
                gvv[..., 1] = torch.where(near & (gy_ > 0), gy_ * wd, gy_)
                gv = gvv.view(nx * ny, 2)
            gv = torch.where(walls[:, None], torch.zeros_like(gv), gv)  # interior wall BC
        else:                                                       # --- 3D: CSF + reflective box walls ---
            # SURFACE TENSION IN 3D. Until now this whole term lived inside the `D == 2` branch
            # above, so `surface_tension: 60.0` in a `dim: 3` spec was read, stored, and never
            # used. It cost a rung of the cell ladder: a liquid cytosol dropped onto the floor
            # spread into a one-particle-thick puddle covering the entire domain, which reads as
            # "the liquid is too soft" and is actually "there is no cohesion term at all". A liquid
            # in MLS-MPM has mu = 0 by construction -- it resists volume change and nothing else --
            # so surface tension is not a refinement on top of the constitutive model, it is the
            # ONLY thing that makes a droplet a droplet.
            #
            # SAME CSF AS 2D, WRITTEN OVER D AXES. Brackbill's continuum surface force: the liquid
            # colour `gc` deposited by the scatter is smooth across the interface, its gradient is
            # the interface normal times the interface's sharpness, and the divergence of the unit
            # normal is the curvature. f = sigma * kappa * grad(c) then pulls a bulge in and pushes
            # a dimple out. The 2D code above is left untouched rather than generalised, because it
            # is on the promotion path and must stay bit-identical.
            surf = self.surface_tension
            if getattr(self, "_c_csf", None) is None:
                self._c_csf = bool(surf > 0.0 and bool((gc > 0).any()))
            if self._c_csf:
                if self.csf_rho > 0.0:
                    c = (gc / (self.csf_rho * dx ** D)).view(*g.shape)   # liquid VOLUME FRACTION
                else:
                    c = gc.view(*g.shape)                                # legacy: raw liquid MASS
                for _ in range(self.csf_smooth):        # separable [1/4,1/2,1/4]; no-op when 0
                    for k in range(D):
                        c = 0.25 * torch.roll(c, 1, k) + 0.5 * c + 0.25 * torch.roll(c, -1, k)
                grad = [(torch.roll(c, -1, k) - torch.roll(c, 1, k)) * (0.5 * inv_dx)
                        for k in range(D)]
                gmag = torch.sqrt(sum(gk * gk for gk in grad))
                eps = _csf_eps
                nrm = [gk / (gmag + eps) for gk in grad]
                kappa = -sum((torch.roll(nrm[k], -1, k) - torch.roll(nrm[k], 1, k)) * (0.5 * inv_dx)
                             for k in range(D))
                # ONLY WHERE THERE IS AN INTERFACE -- the comment was always right and the test never
                # enforced it. `gmag > 0.02 * gmag.max()` selects 98.1% of OCCUPIED nodes on
                # material_3d_water_drop; the band selects 21.7%. The mass clause is what makes
                # `csf_mass_floor` inert rather than load-bearing: the lightest node admitted holds
                # csf_band * rho * dx^D = 2.3e-7 at band 0.2 on the drop, 23x the 1e-8 floor, so
                # masked nodes with gm == 0 go 2.6% -> 0.0% and floor-binding 16.6% -> 0.0%.
                # THE BAND IS A GAIN, NOT ONLY A FILTER -- COMPENSATE FOR IT.
                # The CSF force is f = sigma*kappa*grad(c), and the TOTAL impulse across an
                # interface telescopes: int grad(c) dx = c_in - c_out. Restricting the force to
                # `band < c < 1-band` therefore delivers exactly (1 - 2*band) OF THE TENSION,
                # whatever the shape of the profile -- it is the fundamental theorem, not an
                # approximation. The default band 0.2 was throwing away 40% of sigma by
                # construction.
                #
                # MEASURED on a Young-Laplace ladder (a sphere in zero gravity compresses until
                # K(1-J) = 2 sigma/R, so mean(J) = 1 - 2 sigma/(R K), nothing fitted), R = 10 mm,
                # sigma 0.072, K 1e4, delivered fraction of the declared tension:
                #     band 0.20 -> 0.472      band 0.10 -> 0.736
                #     band 0.05 -> 0.887      band 0.02 -> 0.972
                # against the predicted 0.60 / 0.80 / 0.90 / 0.96. The trend is the band's, and it
                # goes to 1 as the band closes.
                #
                # Dividing by (1 - 2*band) restores the magnitude while keeping the band doing its
                # real job, which is to keep the force OFF the bulk: the interface test was never
                # about how much tension to apply, only about where.
                _gain = 1.0 / max(1.0 - 2.0 * self.csf_band, 1e-6) if self.csf_band > 0 else 1.0
                if self.csf_band > 0.0:
                    _mfull = self.csf_rho * dx ** D
                    fmask = ((c > self.csf_band) & (c < 1.0 - self.csf_band)
                             & (gm.view(*g.shape) > self.csf_band * _mfull)).to(c.dtype) * _gain
                else:
                    fmask = (gmag > 0.02 * gmag.max()).to(c.dtype)
                inv_m = (dx ** D) / gm.clamp(min=_csf_floor)   # force -> acceleration
                gv = gv + dt * torch.stack(
                    [(surf * kappa * grad[k] * fmask).view(-1) * inv_m for k in range(D)], dim=1)
            if not periodic:
                shape = g.shape; bnd = 3
                gv = gv.view(*shape, D)
                for k in range(D):
                    n_k = shape[k]
                    idx = torch.arange(n_k, device=dev)
                    shp = [1] * D; shp[k] = n_k
                    lo_m = (idx < bnd).view(shp); hi_m = (idx > n_k - bnd).view(shp)
                    ck = gv[..., k]
                    ck = torch.where(lo_m, ck.clamp(min=0), ck)     # don't penetrate the wall
                    ck = torch.where(hi_m, ck.clamp(max=0), ck)
                    gv[..., k] = ck
                    if wd != 1.0:                                   # tangential friction on the wall slabs
                        slab = lo_m | hi_m
                        for j in range(D):
                            if j == k:
                                continue
                            cj = gv[..., j]
                            gv[..., j] = torch.where(slab, cj * wd, cj)
                gv = gv.view(g.n_cells, D)
            if self.plate_axis is not None and self.plate_gap_half > 0.0:
                gv = self._plate_bc(H, g, gv, dev, dt)
            walls = self._walls3d(H, g, dev)                        # solid 3D obstacles (box / sphere)
            if _const_any(self, "_c_walls3d", walls):
                gv = torch.where(walls[:, None], torch.zeros_like(gv), gv)   # no-slip: zero grid velocity inside
        # BUOYANCY IS NOT A DIMENSION BRANCH, and putting it here made it look like one. Inserted
        # in 40e1d0c9 between `if D == 2:` and its `else:`, it STOLE THAT ELSE: the 3D wall code
        # then ran whenever buoyancy was zero -- for a 2D spec too, where `_walls3d` unpacks
        # `nx, ny, nz = g.shape` and dies -- and was SKIPPED whenever buoyancy was on, so every 3D
        # buoyant run had no wall boundary condition at all. Both halves of that were invisible
        # until a 2D spec was written without buoyancy. It now sits after the branch, in both
        # dimensions, which is what a body force on the grid is.
        if self.buoyancy != 0.0:
            # rho at each node from the mass the particles deposited there. `gm` is a node mass and
            # dx^D its volume, so this is a genuine density and the comparison with `rho_ref` is
            # dimensionally honest rather than a tuned ratio.
            _rho = gm / (dx ** D)
            _act = _rho > 1e-9                                  # empty nodes have no buoyancy
            _f = torch.zeros_like(_rho)
            _f[_act] = (_rho[_act] - self.rho_ref) / _rho[_act]
            _dir = torch.zeros(D, device=dev, dtype=gv.dtype)
            if self.buoy_dir is not None:
                _dir[:len(self.buoy_dir)] = torch.as_tensor(self.buoy_dir, device=dev, dtype=gv.dtype)
            else:
                _dir[1 if D == 2 else 2] = -1.0                 # "down" is -y in 2D, -z in 3D
            gv = gv + dt * self.buoyancy * _f[:, None] * _dir[None, :]
        g.v.copy_(gv)                       # in place: a captured graph holds this address
        return {}



# ==========================================================================================================
# `mpm_grid_update[implementation: nosync]` -- the 2D wall BC without boolean-mask indexing.
#
# WHY THIS EXISTS, MEASURED. In 3D `mpm_grid_update` is 2.5% of a frame at 5M particles and 12% at
# 945k. In 2D it is 66% -- 163 ms of 246 on `material_dam_break`, 109 of 163 on
# `material_active_swirl` -- and 2D is 58 of the 78 specs in config/material. The cause is not
# arithmetic: one 2D grid solve issues 113 aten calls, and 8 of them are `index_put_` from
# `gv[lox, :, 0] = ...`. Boolean-mask indexing lowers through `nonzero`, which SYNCHRONISES, so
# every one drains the pipeline; at 18 substeps that is ~144 drains a frame.
#
# THE MASKS ARE RUN-CONSTANT. `lox = ix < bnd` depends only on the grid extent, so nothing here is
# data-dependent -- the sync buys nothing at all. The 3D branch already writes the identical
# condition as `torch.where` on broadcast masks; this is the 2D branch written the same way.
#
# IT SHOULD BE BIT-IDENTICAL TO `default`, which is a much stronger claim than the `warp`
# implementations can make and is gated as such: same operations, same order, same reads. The
# ordering matters and is preserved -- the friction block reads the velocity AFTER the clamps, so
# `vy` carries steps 3-4 before step 5 uses it, and a row that is both `lox` and `hix` sees the
# first update before the second, exactly as the in-place version does.
# ==========================================================================================================
@register_operator("mpm_grid_update", implementation="nosync", family="mpm",
                   set="field", kind="field")
class MPMGridUpdateNoSync(MPMGridUpdate):
    """`mpm_grid_update` with a sync-free 2D wall BC. Identical in 3D, where the default already is."""

    MECHANISM_TAGS = ["grid_solve", "surface_tension", "boundary_conditions", "sync_free"]

    def _wall_bc_2d(self, gv, nx, ny, dev, wd):
        gv = gv.view(nx, ny, 2)
        bnd = 3
        ix = torch.arange(nx, device=dev).view(nx, 1)      # broadcast over y
        iy = torch.arange(ny, device=dev).view(1, ny)      # broadcast over x
        lox, hix = ix < bnd, ix > nx - bnd
        loy, hiy = iy < bnd, iy > ny - bnd
        vx, vy = gv[..., 0], gv[..., 1]
        vx = torch.where(lox, vx.clamp(min=0), vx)         # do not penetrate the x walls
        vx = torch.where(hix, vx.clamp(max=0), vx)
        vy = torch.where(loy, vy.clamp(min=0), vy)         # ... or the y walls
        vy = torch.where(hiy, vy.clamp(max=0), vy)
        if wd != 1.0:                                      # tangential friction on the wall slabs
            vy = torch.where(lox & (vy > 0), vy * wd, vy)
            vy = torch.where(hix & (vy > 0), vy * wd, vy)
            vx = torch.where(loy, vx * wd, vx)
            vx = torch.where(hiy, vx * wd, vx)
        return torch.stack([vx, vy], dim=-1).view(nx * ny, 2)


# ==========================================================================================================
# FROM `discovery_okuda/ops/mpm_gather.py` -- mpm_gather (was g2p) (mpm_grid -> particle): the MLS-MPM grid-to-particle gather + advection.
# ==========================================================================================================
@register_operator("mpm_gather", "g2p", family="mpm", set="particle", kind="exchange")
class MPMGather(Exchange):                  # (alias `g2p`, one migration cycle)
    EMIT = None                                    # advects pos/vel inside the MPM substep (MAY_MUTATE_INTEGRATED_STATE); returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []                           # no required params — all optional (source grid defaults to `mpm_grid`)
    MAY_MUTATE_INTEGRATED_STATE = True             # advects pos/vel inside the substep (like the oracle)
    MECHANISM_TAGS = ["grid_to_particle", "advection"]
    PARAM_ROLES = {"dt_sub": "substep_timestep", "wall_damp": "wall_restitution",
                   "wall_contact": "contact_layer_thickness", "vmax": "speed_cap",
                   "wall_damp_mode": "restitution_application_rate"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM G2P); Sulsky, D. et al. (1994)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.frm = params.get("from", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.wall_damp = float(params.get("wall_damp", 1.0))
        # A LENGTH, AND THEREFORE A TRAP IN ANY BOX THAT IS NOT 1 UNIT WIDE. The contact test is
        # `(x < cb) | (x > box[k] - cb)`, so in a 0.1 m box the historical 0.04 selects everything
        # but a 0.02 m sliver -- the entire fluid reads as permanently in wall contact and is
        # permanently damped. `wall_contact_cells` states it in the only scale the grid has:
        # 0.04 / (1/96) = 3.84 cells, so the default reproduces 0.04 exactly at n_grid 96 and
        # follows the world everywhere else. An explicit `wall_contact` still wins, for the specs
        # that tuned it.
        self.wall_contact_cells = float(params.get("wall_contact_cells", 3.84))
        self.wall_contact = params.get("wall_contact", None)
        if self.wall_contact is not None:
            self.wall_contact = float(self.wall_contact)
        # HOW OFTEN `wall_damp` IS APPLIED, and it is the difference between a restitution
        # coefficient and a decay rate. See the note in `forward`. `per_substep` is the historical
        # behaviour and stays the default so no existing run changes.
        self.wall_damp_mode = str(params.get("wall_damp_mode", "per_substep")).lower()
        if self.wall_damp_mode not in ("per_substep", "per_impact"):
            raise ValueError(f"mpm_gather: wall_damp_mode must be 'per_substep' or 'per_impact', "
                             f"got {self.wall_damp_mode!r}")
        self.vmax = float(params.get("vmax", 1e9))

    def _contact_band(self, g):
        """The contact-layer thickness, in world length. An explicit `wall_contact` wins; otherwise
        it is `wall_contact_cells * dx`, which follows the box instead of assuming it is 1 wide."""
        return self.wall_contact if self.wall_contact is not None \
            else self.wall_contact_cells * float(g.dx)

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.frm); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        inv_dx, dx = g.inv_dx, g.dx
        D = p.F.shape[-1]
        periodic = bool(getattr(H, "periodic", False))
        # CACHED, because it is constant for the whole run and `float()` on a tensor element is a
        # host sync. Read fresh every call it cost a GPU->CPU round trip per substep, and under
        # torch.compile it broke the graph in the middle of the gather on every single call.
        # Computed once, the branch is False on every later call and never enters the traced graph.
        if getattr(self, "_box", None) is None:
            self._box = [float(b) for b in
                         getattr(H, "world_size", torch.tensor([g.width, 1.0]))][:D]
        box = self._box
        offsets = stencil_offsets(D, dev); S = offsets.shape[0]
        X, V = p.get("pos"), p.get("vel")
        fx, weight, flat = bspline(X, inv_dx, offsets, g.shape, periodic)
        gvn = g.v[flat].view(p.n, S, D)
        new_V = (weight[..., None] * gvn).sum(1)
        dpos_grid = offsets[None] - fx[:, None, :]
        new_C = 4 * inv_dx * (weight[..., None, None] * (gvn[..., :, None] @ dpos_grid[..., None, :])).sum(1)
        new_V = torch.nan_to_num(new_V)
        if self.wall_damp != 1.0 and not periodic:                 # inelastic wall contact (solids)
            cb = self._contact_band(g)
            near = torch.zeros(p.n, dtype=torch.bool, device=dev)
            for k in range(D):
                near = near | (X[:, k] < cb) | (X[:, k] > box[k] - cb)
            liquid = getattr(p, "is_liquid", None)
            if liquid is not None:
                near = near & ~liquid
            # ONCE PER IMPACT, NOT ONCE PER SUBSTEP. `wall_damp` reads as a restitution coefficient
            # -- "keep 60% of the velocity on a bounce" -- and under `per_substep` it is not one:
            # the multiplier lands on every substep the particle spends inside `wall_contact`, so
            # one impact removes wall_damp ** (substeps in the layer). That exponent grows with grid
            # resolution, because a finer grid resolves the contact more stiffly and the body
            # lingers. MEASURED on material_3d_ball_drop, energy removed by the wall:
            #
            #     n_grid  64   2.4%      n_grid  96  71.0%      n_grid 128  70.8%
            #
            # -- the same spec, the same wall_damp, 68.6 percentage points apart. It is not even
            # monotonic across scenes: genA_code_star_ball loses MORE at 64 (84.1%) than at 96
            # (67.5%). A parameter whose meaning depends on the discretisation cannot be calibrated
            # once, which is why retuning every spec was the wrong fix.
            #
            # `per_impact` applies it on the RISING EDGE of contact only -- the substep where the
            # particle enters the layer -- so one impact costs exactly one multiplication whatever
            # the substep count. The edge state is a persistent per-particle buffer written IN
            # PLACE, so a captured graph keeps reading the address it saw.
            if self.wall_damp_mode == "per_impact":
                prev = getattr(p, "_wall_near", None)
                if prev is None:
                    p.register_buffer("_wall_near",
                                      torch.zeros(p.n, dtype=torch.bool, device=dev))
                    prev = p._wall_near
                near, _keep = near & ~prev, near
                prev.copy_(_keep)
            new_V = torch.where(near[:, None], new_V * self.wall_damp, new_V)
        sp = new_V.norm(dim=1, keepdim=True).clamp(min=1e-9)
        vmax = min(self.vmax, 0.4 * dx / dt)                       # CFL velocity cap
        new_V = new_V * (sp.clamp(max=vmax) / sp)
        new_C = torch.nan_to_num(new_C)
        Xn = torch.nan_to_num(X + dt * new_V, nan=0.5)
        if periodic:
            Xn = torch.stack([torch.remainder(Xn[:, k], box[k]) for k in range(D)], dim=1)
        else:
            Xn = torch.stack([Xn[:, k].clamp(2 * dx, box[k] - 2 * dx) for k in range(D)], dim=1)
        # DORMANT particles (occ==0, a agent_grow reserve) are FROZEN -- not advected -- so they sit as a
        # compact reservoir until agent_grow activates + repositions them. Byte-identical when all are live.
        occ = getattr(p, "occ", None)
        if occ is not None:
            live = occ > 0
            Xn = torch.where(live[:, None], Xn, X)
            new_V = torch.where(live[:, None], new_V, V)
            new_C = torch.where(live[:, None, None], new_C, p.C)
        # IN PLACE. Every read of X / V above happens before this write, and this operator declares
        # MAY_MUTATE_INTEGRATED_STATE, so the engine's tick-0 integration-invariant guard does not
        # apply to it. The clone-and-rebind it replaces gave `p.state` a new address every substep.
        pa, pb = p.state_schema["pos"]; va, vb = p.state_schema["vel"]
        p.state[:, pa:pb] = Xn
        p.state[:, va:vb] = new_V
        p.C.copy_(new_C)
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/mpm_strain.py` -- mpm_strain (particle -> particle): the MLS-MPM deformation-gradient + material update.
# ==========================================================================================================
@register_operator("mpm_strain", family="mpm", set="particle", kind="lateral")
class MPMStrain(Lateral):
    EMIT = None                 # particle->particle: updates F + material in place; returns {} — no delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []        # no required params — all knobs optional (defaults in __init__)
    MECHANISM_TAGS = ["elastic_strain", "plastic_flow", "incompressible_volume"]
    PARAM_ROLES = {"dt_sub": "MLS-MPM substep dt"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM); Sulsky, D. et al. (1994). Comput. Methods Appl. Mech. Eng. 118:179-196."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.dt_sub = float(params.get("dt_sub", 2e-4))

    def forward(self, H, mask=None):
        p = H.level(self.at); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        D = p.F.shape[-1]
        eye = torch.eye(D, device=dev).expand(p.n, D, D)
        F = (eye + dt * p.C) @ p.F
        if D == 2:
            a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
            J = a * d - b * c
        else:
            J = torch.linalg.det(F)
        liquid = getattr(p, "is_liquid", None)
        if _const_any(self, "_c_liquid", liquid):                  # LIQUID: drop shape memory
            Jc = J.clamp(min=1e-6)
            Jl = torch.sqrt(Jc) if D == 2 else Jc.pow(1.0 / D)     # volume-preserving isotropic reset
            F = torch.where(liquid[:, None, None], eye * Jl[:, None, None], F)
        visco = getattr(p, "is_visco", None)
        if _const_any(self, "_c_visco", visco):                    # VISCOELASTIC (Maxwell): PARTIAL shape reset
            vm = visco                                             # relax F toward isotropic with time-constant tau,
            Fv = F[vm]                                             # keeping VOLUME (J) -> stress builds then decays
            U, sig, Vh = torch.linalg.svd(Fv)                      # SVD: sig = principal stretches
            Jl = sig.prod(-1).clamp(min=1e-6).pow(1.0 / D)         # isotropic (volume-preserving) target stretch
            a = torch.exp(-dt / p.visco_tau[vm].clamp(min=1e-6))   # memory retained this substep: a->1 elastic, a->0 liquid
            sig = Jl[:, None] + (sig - Jl[:, None]) * a[:, None]   # pull stretches toward isotropic (shear relaxes, volume kept)
            F = F.clone()
            F[vm] = U @ torch.diag_embed(sig) @ Vh
        snow = getattr(p, "is_snow", None)
        if _const_any(self, "_c_snow", snow):                      # SNOW: clamp singular values, harden via Jp
            sm = snow; Fs = F[sm]
            if Fs.shape[0] > 0:
                U, sig, Vh = torch.linalg.svd(Fs)
                if D == 3:                                          # proper-rotation sign fix (MPM_3D)
                    U = U.clone(); sig = sig.clone(); Vh = Vh.clone()
                    negU = torch.det(U) < 0
                    U[negU, :, -1] *= -1; sig[negU, -1] *= -1
                    negV = torch.det(Vh) < 0
                    Vh[negV, -1, :] *= -1; sig[negV, -1] *= -1
                sig_c = sig.clamp(1.0 - 2.5e-2, 1.0 + 7.5e-3)
                F = F.clone(); F[sm] = U @ torch.diag_embed(sig_c) @ Vh
                ratio = sig.prod(-1) / sig_c.prod(-1).clamp(min=1e-6)
                Jp = p.Jp.clone(); Jp[sm] = (Jp[sm] * ratio).clamp(0.6, 20.0)
                p.Jp.copy_(Jp)
        # DORMANT PARTICLES DO NOT DEFORM. `mpm_scatter` masks its weights by occupancy and
        # `mpm_gather` freezes occ==0 rather than advecting it, but this operator integrated F for the
        # reserve regardless -- so a particle waiting to be spawned accumulated an arbitrary deformation
        # for as long as it waited, and was then promoted into real material carrying it. Byte-identical
        # when every particle is live, which is every composition that has no reserve.
        occ = getattr(p, "occ", None)
        if occ is not None:
            live = (occ > 0)[:, None, None]
            F = torch.where(live, F, p.F)
        p.F.copy_(F)                        # in place; every read of p.F above precedes it
        return {}


# ==========================================================================================================
# mpm_turgor -- the isotropic outward pressure a cell's interior holds against its cortex.
# ==========================================================================================================
@register_operator("mpm_turgor", "osmotic_pressure", family="mpm", set="particle", kind="lateral")
class MPMTurgor(Lateral):
    """Give a set an isotropic OUTWARD pressure -- turgor / excess osmotic pressure -- by writing a
    per-particle `turgor` buffer that `mpm_scatter` subtracts from the Kirchhoff stress.

    WHY THIS IS A MECHANISM AND NOT A TUNING KNOB. An MLS-MPM liquid has `mu = 0` by construction, so
    its whole constitutive law is `tau = la*J*(J-1)*I`: it resists departures from the volume it was
    BORN at, and nothing else. A cytosol built that way sits at `J = 1` forever -- it has no reason to
    press on anything -- so a membrane drawn around it carries zero tension and is a floppy bag that
    the interior drains out of. Real cells are not like that: the interior is held above the outside
    pressure by the solutes trapped in it, the cortex is in tension because of it, and the two are
    related by Laplace, `P = 2*gamma/R`. Turgor is the term that makes a membrane a membrane.

    WHY IT IS NOT SURFACE TENSION. `mpm_grid_update`'s CSF term (`surface_tension:`) minimises
    interface AREA -- `f = sigma * kappa * grad(c)` pulls a bulge in and pushes a dimple out. It can
    round a droplet up and it can hold one together, but its sign is inward everywhere the interface
    is convex, so it can never inflate a cell against a shell. The two are complementary, not
    alternatives: surface tension is the cortex pulling in, turgor is the cytosol pushing out, and a
    cell at rest is the balance of them.

    WHY IT IS A REST-VOLUME SHIFT AND THEREFORE SELF-LIMITING. Adding `-P.I` to `tau` moves the zero
    of the liquid's own equation of state: the material stops expanding at the `J*` that solves
    `la*J*(J-1) = P`, which for the small strains a cell runs at is `J* ~= 1 + P/la`. So `pressure` is
    NOT an unbounded body force that blows the cell up -- it names an inflated rest volume, and the
    liquid's bulk modulus is what holds it there. Sizing it is one division: to swell a cytosol by a
    fraction `eps` of ITS OWN INITIAL VOLUME, ask for `pressure ~= la*eps`, where
    `la = E*nu/((1+nu)(1-2nu))`, which at the shared `nu = 0.2` is `la = E/3.6`.

    THE UNITS ARE THE STRESS UNITS OF THE SPEC, the same ones `youngs` is in. The number that matters
    is the RATIO to two other pressures in the same run: the liquid's bulk modulus `la` (which sets
    how far it swells) and the hydrostatic head `rho*g*R` of the cell's own weight (which is what the
    turgor has to beat for the interior not to puddle in the bottom of the membrane).

    `mode: constant` (the default) holds `P` fixed -- the osmotic exchange across a real membrane is
    orders of magnitude slower than the mechanics, so over a run of this length the solute count, not
    the pressure, is what a van 't Hoff term would have to carry; that form is a strictly larger
    operator and is deliberately not built until something needs a cell to shrink in hypertonic
    medium.
    """

    EMIT = None                 # particle->particle: writes the `turgor` buffer in place; no delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["pressure"]
    MECHANISM_TAGS = ["osmotic_pressure", "turgor", "volume_regulation"]
    PARAM_ROLES = {"pressure": "isotropic_outward_pressure", "mode": "pressure_law"}
    REFERENCE = ("Stewart, M.P. et al. (2011). Nature 469:226-230 (hydrostatic pressure and the "
                 "actomyosin cortex in mitotic rounding); Jiang, H. & Sun, S.X. (2013). Biophys. J. "
                 "105:609-619 (cell volume and osmotic pressure).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.pressure = float(params["pressure"])           # excess of interior over exterior pressure
        self.mode = str(params.get("mode", "constant"))
        self.at = params.get("_at", "mpm_particle")
        if self.mode != "constant":
            raise ValueError(f"mpm_turgor: unknown mode {self.mode!r} (only 'constant' is built)")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        P = torch.full((p.n,), self.pressure, device=p.state.device)
        # DORMANT PARTICLES CARRY NO PRESSURE. `mpm_scatter` masks a reserve particle's weights to
        # zero so it deposits nothing, but `store_stress` and any diagnostic reading `turgor` would
        # otherwise report a pressurised particle that is not in the material yet.
        occ = getattr(p, "occ", None)
        if occ is not None:
            P = P * (occ > 0).to(P.dtype)
        if mask is not None:
            P = P * mask.to(P.dtype)
        if getattr(p, "turgor", None) is None or p.turgor.shape != P.shape:
            p.register_buffer("turgor", P)
        else:
            p.turgor.copy_(P)
        return {}


# ==========================================================================================================
# mpm_viscosity -- the Newtonian viscous stress a liquid in this MPM does not otherwise have.
# ==========================================================================================================
@register_operator("mpm_viscosity", family="mpm", set="particle", kind="lateral")
class MPMViscosity(Lateral):
    """WHY A LIQUID HERE NEVER STOPS MOVING. `material: liquid` sets mu = 0, so the deviatoric
    stress is IDENTICALLY ZERO and nothing in the constitutive model resists or dissipates shear.
    The only sinks in the whole scheme are `wall_damp` and MLS-MPM's own numerical dissipation --
    and APIC exists precisely to minimise the latter. A real drop comes to rest; this one has no
    mechanism to.

    MEASURED, on material_3d_water_st050 (290k particles, 2400 frames): kinetic energy falls 500x
    between frames 400 and 1600 and then STOPS FALLING -- 7e-5, 8e-5, 8e-5, 11e-5 at frames 1600,
    1800, 2000, 2200. A plateau, not a decay. Of the residual at frame 2201, 29% IS SUB-GRID
    JITTER (rms speed 0.0107, of which 0.0090 survives coarse-graining onto 4-cell boxes and
    0.0058 does not), which is exactly the scale a viscosity acts on.

    THE STRESS. Water is Newtonian: sigma = -p I + 2 mu_dyn D, with D the strain-rate tensor
    (1/2)(grad v + grad v^T) and mu_dyn ~ 1.0e-3 Pa s. In MLS-MPM the affine matrix C IS the
    velocity gradient, so D = (1/2)(C + C^T) and the Kirchhoff viscous stress is

        tau_visc = mu_dyn * (C + C^T)

    added to the elastic stress with the SAME sign convention (positive = tension): under
    extension a viscous fluid pulls back, exactly as an elastic one does. It costs one 3x3
    symmetrisation per particle -- C is already carried -- and no extra P2G pass, because it goes
    through the additive-stress channel `mpm_scatter` already reads for `pulse_to_active_stress`.
    That channel is live on the warp path too (the kernel takes ACT as a mat33), so this operator
    works at `implementation: warp` without a kernel change.

    HOW BIG SHOULD IT BE, AND WHAT IT IS HONEST TO CLAIM. Quote it as a Reynolds number or it means
    nothing. Re = U L / nu with L the body's own size. In the water-drop scene (L ~ 0.2, U ~ 3.9 at
    impact, U ~ 0.01 once settled) a kinematic nu of 3e-4 gives Re ~ 2300 DURING THE IMPACT -- high
    enough that the splash is barely touched -- and Re ~ 6 ONCE SETTLED, where it dominates and the
    jitter dies. That asymmetry is the physics doing the work, not a schedule: Re falls as U does.
    A real centimetre-scale water drop lands at Re 1e4-1e5, so a splash genuinely IS near-inviscid
    and real water genuinely does slosh; what stops a real puddle is viscosity acting on the small
    scales over many seconds. Pushed far past the Reynolds-matched value this term becomes
    numerical damping wearing a physics name, and should be called sub-grid or artificial
    viscosity when it is used that way.

    WHAT IT IS NOT. It is NOT `drag`, which is a body force -dragl*v and therefore damps UNIFORM
    TRANSLATION -- it slows free fall. A viscous stress depends on the velocity GRADIENT, so a body
    in rigid translation has C = 0, tau_visc = 0, and falls at exactly g. That is the gate.
    It also will not stop the slow compaction of a drop (a volume error, not a momentum one), nor
    the surface-tension implosion at high sigma (the CSF overpowering the bulk modulus).

    STABILITY. An explicit viscous stress carries its own diffusion limit, dt < rho dx^2 / (2 D mu).
    At mu 3e-4, rho 1, dx 1/96, D 3 that is 6.0e-2 against a 2e-4 substep -- 300x of headroom -- and
    even mu 1e-2 leaves 9x. The operator reports the margin so a spec cannot walk past it silently.
    """

    EMIT = None                 # writes H.extra_stress, consumed by mpm_scatter in the same substep
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["eta"]
    MECHANISM_TAGS = ["viscous_stress", "momentum_diffusion", "dissipation"]
    PARAM_ROLES = {"eta": "dynamic_viscosity", "liquid_only": "restrict_to_liquid_particles"}
    REFERENCE = ("Batchelor, G.K. (1967). An Introduction to Fluid Dynamics, ch. 3 (Newtonian "
                 "stress); Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM, C = grad v).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.eta = float(params["eta"])                    # DYNAMIC viscosity mu_dyn
        # ONLY THE LIQUID, BY DEFAULT. A solid already carries a deviatoric stress through mu > 0;
        # adding a viscous one on top makes it Kelvin-Voigt, which is a different material and
        # should be asked for rather than inherited.
        self.liquid_only = bool(params.get("liquid_only", True))
        if self.eta < 0:
            raise ValueError(f"mpm_viscosity: eta must be >= 0, got {self.eta}")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        C = p.C
        tau = self.eta * (C + C.transpose(-2, -1))
        if self.liquid_only:
            liq = getattr(p, "is_liquid", None)
            if _const_any(self, "_c_liquid", liq):
                tau = tau * liq[:, None, None].to(tau.dtype)
        occ = getattr(p, "occ", None)
        if occ is not None:
            tau = tau * (occ > 0).to(tau.dtype)[:, None, None]
        if mask is not None:
            tau = tau * mask.to(tau.dtype)[:, None, None]
        prev = getattr(H, "extra_stress", None)
        if prev is None or prev.shape != tau.shape:
            H.extra_stress = tau
        else:
            prev.copy_(tau)                     # persistent buffer -> safe inside a captured graph
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/mpm_anchor.py` -- mpm_anchor -- a substrate/boundary rest-anchor body force for MLS-MPM particles.
# ==========================================================================================================
@register_operator("mpm_anchor", family="mechanics", set="particle", kind="lateral")
class MPMAnchor(Lateral):
    EMIT = "mpm_acceleration"   # consumed by the MPM substep as a_ext, not engine-integrated
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["k"]
    MECHANISM_TAGS = ["substrate_anchor", "boundary_condition", "rest_restoring"]
    PARAM_ROLES = {"k": "anchor_stiffness", "ring": "boundary_width", "mode": "anchor_extent"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = float(params["k"])
        self.mode = str(params.get("mode", "boundary"))       # "boundary" ring | "substrate" all
        self.ring = params.get("ring", None)                  # None -> derived (a length)
        self.at = params.get("_at", "particle")
        self._rest = None
        self._sel = None

    def _init(self, lvl, H=None):
        # THE BAND IS A LENGTH, AND ITS NATURAL SCALE IS THE CELL. An anchor ring narrower than a
        # couple of cells cannot be resolved by the grid that carries the force, so `dx` is the
        # right yardstick -- 3.84 cells, which is exactly 0.04 at n_grid 96 and therefore leaves
        # every existing spec where it was. Falls back to the historical constant when no MPM grid
        # is in the run (the operator does not require one).
        self._rest = lvl.get("pos").clone()                   # undeformed sheet (frame 0)
        if self.ring is None:
            _g = (H.fields.get("mpm_grid") if H is not None and getattr(H, "fields", None)
                  else None)
            self._ring = (_scale_constant("ring", float(_g.dx)) if _g is not None
                          else _CONST_DIMS["ring"][0])
        else:
            self._ring = float(self.ring)
        if self.mode == "substrate":
            self._sel = torch.ones(self._rest.shape[0], dtype=torch.bool, device=self._rest.device)
        else:                                                 # outer ring of the tissue's rest extent
            lo = self._rest.min(0).values
            hi = self._rest.max(0).values
            near = ((self._rest - lo) < self._ring) | ((hi - self._rest) < self._ring)  # [N,2]
            self._sel = near.any(dim=1)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self._rest is None:
            self._init(lvl, H)
        acc = self.k * (self._rest - lvl.get("pos")) * (self._sel * lvl.occ)[:, None].float()
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


# ==========================================================================================================
# FROM `discovery_okuda/ops/mpm_spin.py` -- mpm_spin -- drive an MLS-MPM body toward slow solid-body rotation (a body force).
# ==========================================================================================================
@register_operator("mpm_spin", family="mechanics", set="particle", kind="lateral")
class MPMSpin(Lateral):
    EMIT = "mpm_acceleration"   # consumed by the MPM substep as a_ext, not engine-integrated
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["omega"]
    MECHANISM_TAGS = ["solid_body_rotation", "swirl"]
    PARAM_ROLES = {"omega": "angular_velocity", "spin_k": "spin_gain"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.omega = float(params["omega"])                # target angular velocity (rad / time)
        self.spin_k = float(params.get("spin_k", 30.0))    # controller gain toward v_rot
        self.center = params.get("center", None)           # rotation centre; default = domain centre
        self.axis = params.get("axis", [0.0, 0.0, 1.0])    # 3D rotation axis
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at); dev = lvl.state.device
        X = lvl.get("pos"); V = lvl.get("vel")
        D = X.shape[1]
        if self.center is not None:
            c = torch.tensor([float(x) for x in self.center][:D], device=dev)
        else:                                              # domain centre: axis 0 = width, rest = 1
            box = [float(b) for b in getattr(H, "world_size", [getattr(H, "world_width", 1.0)] + [1.0] * (D - 1))][:D]
            c = 0.5 * torch.tensor(box, device=dev)
        rel = X - c
        if D == 2:
            v_rot = self.omega * torch.stack([-rel[:, 1], rel[:, 0]], dim=1)
        else:
            ax = torch.tensor([float(a) for a in self.axis][:3], device=dev)
            ax = ax / ax.norm().clamp(min=1e-9)
            v_rot = self.omega * torch.cross(ax.expand_as(rel), rel, dim=1)
        acc = self.spin_k * (v_rot - V) * lvl.occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


# ==========================================================================================================
# FROM `discovery_okuda/ops/material_map.py` -- material_map -- a static image FIELD read from a TIFF, plus `apply_material_map`,
# ==========================================================================================================
@register_field("image", frame="image")
class ImageField(Field):
    """A 1-channel scalar field read from a 2D image (TIFF/PNG), normalised to [0,1].
    A STATIC map (no dynamics): it holds only its grid `[1, nx, ny]` and the
    world<->pixel geometry, sampled by `apply_material_map`. Same orientation
    convention as `PrescribedField` (flip vertical so image-top maps to domain-top)."""

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu",
                 normalize=True, **kw):
        super().__init__(name)                                  # binds to no set (no couples_to)
        if source is None:
            raise ValueError(f"image field {name!r} needs a `source:` (path to a .tif/.png)")
        import tifffile
        path = source if os.path.isabs(source) else graphs_data_path(source)
        img = tifffile.imread(path).astype("float32")          # [ny, nx] (image rows top->bottom)
        if img.ndim == 3:                                      # collapse any channels to grayscale
            img = img.mean(axis=-1)
        img = img[::-1, :].copy()                              # flip vertical: image-top -> domain-top
        if normalize:
            lo, hi = float(img.min()), float(img.max())
            img = (img - lo) / (hi - lo + 1e-9)                # -> [0,1]
        v = torch.tensor(img, device=device).permute(1, 0).contiguous()   # [ny,nx] -> [nx,ny]
        self.C = 1
        self.nx, self.ny = int(v.shape[0]), int(v.shape[1])
        self.width = float(width)
        self.R = self.nx / self.width                          # pixels per world unit (x)
        self.register_buffer("grid", v[None])                  # [1, nx, ny]

    def pix(self, x, y):
        gx = (x.clamp(0, self.width - 1e-6) / self.width * self.nx).long().clamp(0, self.nx - 1)
        gy = (y.clamp(0, 1 - 1e-6) * self.ny).long().clamp(0, self.ny - 1)
        return gx, gy


@register_field("vector_grid", frame="vector_grid")
class VectorGrid(Field):
    """A 2-channel UNIT-VECTOR field d(x) = (dx, dy) read from a TIFF -- the contraction
    DIRECTION / active-stress-orientation map. A 2-channel TIFF `[ny,nx,2]` is read as
    (dx, dy); a 1-channel TIFF as an angle theta in [0,1]->[0,2pi) -> (cos, sin). Every
    vector is normalised to unit length. Same vertical-flip convention as ImageField."""

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu", **kw):
        super().__init__(name)
        if source is None:
            raise ValueError(f"vector_grid field {name!r} needs a `source:` (path to a .tif)")
        import tifffile
        import numpy as np
        path = source if os.path.isabs(source) else graphs_data_path(source)
        img = tifffile.imread(path).astype("float32")
        img = img[::-1].copy()                                 # flip vertical (image-top -> domain-top)
        if img.ndim == 2:                                      # angle map theta in [0,1] -> [0,2pi)
            th = img * (2 * np.pi)
            dx, dy = np.cos(th), np.sin(th)
        else:                                                  # [ny,nx,2] vector map (dx, dy), [-1,1]
            dx, dy = img[..., 0], img[..., 1]
        v = np.stack([dx, dy], axis=0)                         # [2, ny, nx]
        n = np.sqrt(v[0] ** 2 + v[1] ** 2); n[n < 1e-9] = 1.0
        v = (v / n).astype("float32")                          # unit vectors
        vt = torch.tensor(v, device=device).permute(0, 2, 1).contiguous()   # [2, nx, ny]
        self.C = 2
        self.nx, self.ny = int(vt.shape[1]), int(vt.shape[2])
        self.width = float(width)
        self.R = self.nx / self.width
        self.register_buffer("grid", vt)                       # [2, nx, ny]


@register_operator("apply_material_map", family="mpm", set="particle", kind="exchange")
class ApplyMaterialMap(Exchange):
    """field -> set: sample the map at each particle and write a per-particle material
    parameter. `target: youngs` maps intensity in [0,1] to E in [min,max] and sets the
    Lame buffers mu/la (the MPM stress law reads them); any other `target` is written as
    a per-particle buffer of that name. Mutates per-particle buffers, returns {}."""

    EMIT = None                              # sets material, emits no force
    REQUIRES_PARAMS = ["from", "target"]
    SUPPORTED_DIMS = [2, 3]
    MECHANISM_TAGS = ["material_map", "heterogeneous_stiffness", "symmetry_breaking"]
    PARAM_ROLES = {"min": "param_lo", "max": "param_hi", "target": "material_parameter"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM material model)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.target = str(params.get("target", "youngs"))
        self.lo = float(params.get("min", 20.0))
        self.hi = float(params.get("max", 200.0))
        self.channel = int(params.get("channel", 0))
        self.at = params.get("_at", "mpm_particle")

    def _sample(self, H, lvl):
        """Bilinear-sample the map field at the particle positions -> intensity [N] in [0,1]."""
        return H.fields[self.field_name].sample(lvl.get("pos"), self.channel).clamp(0.0, 1.0)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        mapped = self.lo + self._sample(H, lvl) * (self.hi - self.lo)   # intensity -> [lo, hi]
        if self.target == "youngs":
            from plexus.models.entities import _lame
            mu, la = _lame(mapped)
            liquid = getattr(lvl, "is_liquid", None)
            if liquid is not None:                                 # liquid keeps zero shear modulus
                mu = torch.where(liquid, torch.zeros_like(mu), mu)
            lvl.mu, lvl.la = mu, la                                # MPM stress reads these
            if "youngs" in getattr(lvl, "_buffers", {}):
                lvl.youngs = mapped
            else:
                lvl.register_buffer("youngs", mapped)
        else:
            if self.target in getattr(lvl, "_buffers", {}):
                setattr(lvl, self.target, mapped)
            else:
                lvl.register_buffer(self.target, mapped)
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/mpm.py` -- mls_mpm_mechanics -- a FENCED TRANSITIONAL operator wrapping the MLS-MPM solver.
# ==========================================================================================================
_OFFSETS = torch.tensor([[i, j] for i in range(3) for j in range(3)], dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  Backend kernel: one MLS-MPM substep (pure -> compilable). NOT a Plexus
#  primitive -- the fenced operator below is the only thing that calls it.
# --------------------------------------------------------------------------- #
def mls_mpm_substep(X, V, C, F, mass, mu, la, a_ext, offsets,
                    nx, ny, dx, inv_dx, dt, p_vol, drag, walls_flat, vmax_user, periodic, width,
                    wall_damp, wall_contact, liquid_mask, snow_mask, Jp, surf):
    """One MLS-MPM substep. All tensors batched over particles. Pure -> compilable.
    Grid is [nx, ny] of square cells (dx); the world is [0,width]x[0,1]."""
    N = X.shape[0]
    eye = torch.eye(2, device=X.device).expand(N, 2, 2)

    # external cell-level accel + Stokes drag (overdamped tissue): V saturates at a/drag
    V = V + dt * (a_ext - drag * V)

    # deformation gradient update
    F = (eye + dt * C) @ F
    a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
    J = a * d - b * c                                   # det(F)

    if liquid_mask is not None:                         # LIQUID: drop shape memory, keep only volume J.
        Jl = torch.sqrt(J.clamp(min=1e-6))              # F := sqrt(J)*I  -> isotropic, no shear/rotation.
        F = torch.where(liquid_mask[:, None, None], eye * Jl[:, None, None], F)
        a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]   # (mu=0 for liquid -> stress is pure pressure)

    mu_e, la_e = mu, la
    if snow_mask is not None:                           # SNOW: plastic flow -> clamp singular values of F,
        sm = snow_mask                                  # accumulate plastic volume Jp, harden mu/la with Jp.
        Fs = F[sm]
        if Fs.shape[0] > 0:
            U, sig, Vh = torch.linalg.svd(Fs)           # F = U diag(sig) Vh   (per snow particle)
            sig_c = sig.clamp(1.0 - 2.5e-2, 1.0 + 7.5e-3)   # snow yield: theta_c compress, theta_s stretch
            Fp = U @ torch.diag_embed(sig_c) @ Vh
            F = F.clone(); F[sm] = Fp
            ratio = sig.prod(-1) / sig_c.prod(-1).clamp(min=1e-6)   # volume pushed into plastic part
            Jp = Jp.clone(); Jp[sm] = (Jp[sm] * ratio).clamp(0.6, 20.0)
            a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
            J = a * d - b * c
        h = torch.exp((10.0 * (1.0 - Jp)).clamp(-6.0, 6.0))   # Jp<1 (packed) -> harder; Jp>1 -> softer
        mu_e = torch.where(sm, mu * h, mu)
        la_e = torch.where(sm, la * h, la)

    # analytic 2x2 polar rotation R (closest rotation to F)
    cs, sn = (a + d), (c - b)
    r = torch.sqrt(cs * cs + sn * sn) + 1e-9
    cs, sn = cs / r, sn / r
    R = torch.stack([torch.stack([cs, -sn], -1),
                     torch.stack([sn, cs], -1)], -2)    # [N,2,2]

    # fixed-corotated stress -> affine momentum matrix  (mu_e/la_e carry snow hardening)
    FmR = F - R
    stress = 2 * mu_e[:, None, None] * (FmR @ F.transpose(-2, -1)) \
        + eye * (la_e * J * (J - 1))[:, None, None]
    pv = p_vol[:, None, None] if torch.is_tensor(p_vol) else p_vol
    stress = (-dt * 4 * inv_dx * inv_dx) * pv * stress
    affine = stress + mass[:, None, None] * C

    # --- P2G ---
    base = (X * inv_dx - 0.5).floor().long()            # [N,2]
    fx = X * inv_dx - base.float()
    w = torch.stack([0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1) ** 2,
                     0.5 * (fx - 0.5) ** 2], dim=1)      # [N,3,2]
    oi = offsets[:, 0].long(); oj = offsets[:, 1].long()             # [9]
    weight = w[:, oi, 0] * w[:, oj, 1]                               # [N,9]
    gpos = base[:, None, :] + offsets.long()[None]                  # [N,9,2]
    if periodic:
        gpos = torch.stack([gpos[..., 0] % nx, gpos[..., 1] % ny], dim=-1)
    else:
        gpos = torch.stack([gpos[..., 0].clamp(0, nx - 1), gpos[..., 1].clamp(0, ny - 1)], dim=-1)
    dpos_phys = (offsets[None] - fx[:, None, :]) * dx               # [N,9,2]

    mom = mass[:, None, None] * V[:, None, :] \
        + (affine[:, None] @ dpos_phys[..., None]).squeeze(-1)      # [N,9,2]
    flat = (gpos[..., 0] * ny + gpos[..., 1]).reshape(-1)          # [N*9]  (row-major nx x ny)
    grid_m = torch.zeros(nx * ny, device=X.device)
    grid_mv = torch.zeros(nx * ny, 2, device=X.device)
    grid_m.index_add_(0, flat, (weight * mass[:, None]).reshape(-1))
    grid_mv.index_add_(0, flat, (weight[..., None] * mom).reshape(-1, 2))

    # grid velocity
    gv = grid_mv / grid_m.clamp(min=1e-10)[:, None]

    if surf > 0.0 and liquid_mask is not None:        # SURFACE TENSION as a CSF (continuum surface force):
        # 1) liquid colour field c on the grid (scatter liquid mass via the same P2G weights)
        lw = (weight * (mass * liquid_mask.to(mass.dtype))[:, None]).reshape(-1)
        c = torch.zeros(nx * ny, device=X.device).index_add_(0, flat, lw).view(nx, ny)
        # 2) normal n = grad(c)/|grad(c)|  (central differences in physical units)
        cx = (torch.roll(c, -1, 0) - torch.roll(c, 1, 0)) * (0.5 * inv_dx)
        cy = (torch.roll(c, -1, 1) - torch.roll(c, 1, 1)) * (0.5 * inv_dx)
        gmag = torch.sqrt(cx * cx + cy * cy)
        eps = 1e-6
        nxg, nyg = cx / (gmag + eps), cy / (gmag + eps)
        # 3) curvature kappa = -div(n)
        kappa = -((torch.roll(nxg, -1, 0) - torch.roll(nxg, 1, 0)) * (0.5 * inv_dx)
                  + (torch.roll(nyg, -1, 1) - torch.roll(nyg, 1, 1)) * (0.5 * inv_dx))
        # 4) surface force density f = surf * kappa * grad(c)  (acts only where |grad c|>0: the interface)
        fmask = (gmag > 0.02 * gmag.max()).to(c.dtype)
        stfx = (surf * kappa * cx * fmask).view(-1)        # surface-tension force, x (not the P2G fx!)
        stfy = (surf * kappa * cy * fmask).view(-1)
        # 5) apply as grid acceleration a = f * cell_area / grid_mass, carried to particles by G2P
        inv_m = (dx * dx) / grid_m.clamp(min=1e-8)
        gv = gv + dt * torch.stack([stfx * inv_m, stfy * inv_m], dim=1)

    if not periodic:                                  # reflective domain walls (toroidal otherwise)
        gv = gv.view(nx, ny, 2)
        ix = torch.arange(nx, device=X.device); iy = torch.arange(ny, device=X.device); bnd = 3
        lox, hix = ix < bnd, ix > nx - bnd
        loy, hiy = iy < bnd, iy > ny - bnd
        gv[lox, :, 0] = gv[lox, :, 0].clamp(min=0); gv[hix, :, 0] = gv[hix, :, 0].clamp(max=0)
        gv[:, loy, 1] = gv[:, loy, 1].clamp(min=0); gv[:, hiy, 1] = gv[:, hiy, 1].clamp(max=0)
        if wall_damp != 1.0:                          # tangential wall FRICTION (kills wall jets), but
            # at SIDE walls damp only UPWARD flow -> jets die yet gravity still drains stuck droplets down
            gl = gv[lox, :, 1]; gv[lox, :, 1] = torch.where(gl > 0, gl * wall_damp, gl)
            gh = gv[hix, :, 1]; gv[hix, :, 1] = torch.where(gh > 0, gh * wall_damp, gh)
            gv[:, loy, 0] = gv[:, loy, 0] * wall_damp   # floor/ceiling: horizontal tangential (symmetric ok)
            gv[:, hiy, 0] = gv[:, hiy, 0] * wall_damp
        gv = gv.view(nx * ny, 2)
    if wall_damp != 1.0 and walls_flat.any():     # friction in the fluid cells touching any INTERIOR
        w2 = walls_flat.view(nx, ny)              # obstacle wall (general: works for any obstacle shape)
        near = (torch.roll(w2, 1, 0) | torch.roll(w2, -1, 0)
                | torch.roll(w2, 1, 1) | torch.roll(w2, -1, 1)) & ~w2
        gvv = gv.view(nx, ny, 2); gx = gvv[..., 0]; gy = gvv[..., 1]
        gvv[..., 0] = torch.where(near, gx * wall_damp, gx)              # horizontal: full friction
        gvv[..., 1] = torch.where(near & (gy > 0), gy * wall_damp, gy)   # vertical: damp only upward -> gravity drains
        gv = gvv.view(nx * ny, 2)
    gv = torch.where(walls_flat[:, None], torch.zeros_like(gv), gv)   # interior wall BC

    # --- G2P ---
    gvn = gv[flat].view(N, 9, 2)                                    # [N,9,2]
    new_V = (weight[..., None] * gvn).sum(1)
    dpos_grid = offsets[None] - fx[:, None, :]                      # [N,9,2]
    new_C = 4 * inv_dx * (weight[..., None, None]
                          * (gvn[..., :, None] @ dpos_grid[..., None, :])).sum(1)
    # robustness: bound velocity (CFL) and sanitize NaN/inf so a bad design can't
    # poison the CUDA context -- it just produces a poor (low-food) trajectory.
    new_V = torch.nan_to_num(new_V)
    if wall_damp != 1.0 and not periodic:            # inelastic walls: bleed kinetic energy from the
        cb = wall_contact                            # SOLID layer in contact with a wall (bounce restitution)
        near = ((X[:, 0] < cb) | (X[:, 0] > width - cb)
                | (X[:, 1] < cb) | (X[:, 1] > 1.0 - cb))
        if liquid_mask is not None:                  # liquids are handled by the asymmetric grid wall
            near = near & ~liquid_mask               # friction -> don't pin them here (else they can't drain)
        new_V = torch.where(near[:, None], new_V * wall_damp, new_V)
    sp = new_V.norm(dim=1, keepdim=True).clamp(min=1e-9)
    vmax = min(vmax_user, 0.4 * dx / dt)             # user cap, never above CFL
    new_V = new_V * (sp.clamp(max=vmax) / sp)
    new_C = torch.nan_to_num(new_C)
    F = torch.nan_to_num(F, nan=1.0)
    X = torch.nan_to_num(X + dt * new_V, nan=0.5)
    if periodic:
        X = torch.stack([torch.remainder(X[:, 0], width),            # bc_pos: wrap onto the torus
                         torch.remainder(X[:, 1], 1.0)], dim=1)
    else:
        X = torch.stack([X[:, 0].clamp(2 * dx, width - 2 * dx),
                         X[:, 1].clamp(2 * dx, 1 - 2 * dx)], dim=1)
    return X, new_V, new_C, F, Jp


# --------------------------------------------------------------------------- #
#  The fenced transitional operator
# --------------------------------------------------------------------------- #
@register_operator("mls_mpm_mechanics", family="mpm", set="particle", kind="exchange")
class MLSMPMMechanics(Exchange):
    """Compound MLS-MPM mechanics at the particle level (P2G -> grid solve -> G2P ->
    advect). Cell shape/rigidity emerge from the particles' elastic stress; per-cell-
    type `youngs` sets the per-particle Lame parameters (mu, la).

    FENCED TRANSITIONAL operator -- breaks the one-concern + integration-invariant
    rules on purpose, behind the `TRANSITIONAL` fence. See module docstring and
    `ARCHITECTURAL_DEBT` for the decomposition roadmap.
    """

    EMIT = None                          # substep advects pos/vel in place (MAY_MUTATE_INTEGRATED_STATE); returns {} — no integrable delta
    SUPPORTED_DIMS = [2]                  # the MLS-MPM kernel hard-codes 2D (eye(2), 3x3 stencil, nx*ny grid)
    REQUIRES_PARAMS = []                  # no required params — all knobs optional (defaults in __init__)

    # --- the fence ------------------------------------------------------- #
    TRANSITIONAL = True
    MAY_MUTATE_INTEGRATED_STATE = True   # the substep advects particles in place (opt out of the guard)
    ARCHITECTURAL_DEBT = [
        "mutates integrated state (advects pos/vel inside the substep, not via a returned delta)",
        "bundles many mechanisms (P2G, grid solve, fixed-corotated/liquid/snow stress, "
        "surface tension, wall BCs, G2P) in one operator",
        "wraps the legacy MLS-MPM numerical kernel `mls_mpm_substep`",
    ]
    # Long-term decomposition target (each line -> one ideal registered primitive):
    #   p2g              exchange   particle -> grid scatter
    #   mpm_grid_solve   field      grid momentum + boundary BCs
    #   mpm_material     field      fixed-corotated / liquid / snow stress (state -> affine)
    #   surface_tension  field      CSF on the grid
    #   g2p              exchange   grid -> particle gather
    #   (advection)      engine     pos/vel integration of the returned G2P delta

    # --- declared dependencies (no longer hidden inside the substep) ----- #
    REQUIRES_TYPE_PROPS = ["youngs"]
    # a liquid has no Young's modulus (nu -> 1/2 makes E = 3K(1-2nu) -> 0); it has a bulk modulus,
    # and for mu = 0 that IS lambda. Either spelling satisfies the requirement.
    TYPE_PROP_ALTERNATIVES = {"youngs": ("bulk_modulus",)}                      # per-cell-type stiffness -> mu, la
    REQUIRES_BUFFERS = ["C", "F", "mass", "mu", "la", "p_vol"]  # per-particle (mpm_particle entity provisions them)
    REQUIRES_HSTATE = []                                  # body force = the PARENT set's accumulated delta (H.delta)

    # --- mechanism-search metadata --------------------------------------- #
    MECHANISM_TAGS = ["elastic_mechanics", "material_point_method",
                      "fixed_corotated_stress", "incompressible_volume",
                      "surface_tension", "plastic_flow"]
    PARAM_ROLES = {
        "n_grid": "background_grid_resolution",
        "substeps": "cfl_subcycling",
        "dt_sub": "substep_timestep",
        "drag": "overdamped_friction",
        "wall_damp": "wall_restitution",
        "surface_tension": "interface_cohesion",
        "vmax": "speed_cap",
    }
    REFERENCE = "Hu, Y. et al. (2018). A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. 37(4):150; Sulsky, D., Chen, Z. & Schreyer, H. L. (1994). Comput. Methods Appl. Mech. Eng. 118:179-196."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")           # the set this operator acts on (engine-injected)
        self.n_grid = int(params.get("n_grid", 128))
        self.substeps = int(params.get("substeps", 10))
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.a_max = float(params.get("a_max", 200.0))
        # WHERE THE BODY FORCE IS APPLIED. "particle" (default, and every existing run) folds it
        # into the particle velocity before P2G; "grid" hands the uniform part to the grid solve,
        # which is where canonical MLS-MPM puts it. See _hand_body_force_to_grid for why the two
        # are algebraically identical until the mass clamp binds, and for the measurement showing
        # that on this spec family it does not.
        self.body_force = str(params.get("body_force", "particle"))
        if self.body_force not in ("particle", "grid"):
            raise ValueError(f"mpm_scatter: body_force must be 'particle' or 'grid', "
                             f"got {self.body_force!r}")    # clamp broadcast accel
        self.drag = float(params.get("drag", 40.0))       # Stokes drag (overdamped)
        self.wall_damp = float(params.get("wall_damp", 1.0))  # 1.0=elastic wall; <1 loses energy on bounce
        # A LENGTH, AND THEREFORE A TRAP IN ANY BOX THAT IS NOT 1 UNIT WIDE. The contact test is
        # `(x < cb) | (x > box[k] - cb)`, so in a 0.1 m box the historical 0.04 selects everything
        # but a 0.02 m sliver -- the entire fluid reads as permanently in wall contact and is
        # permanently damped. `wall_contact_cells` states it in the only scale the grid has:
        # 0.04 / (1/96) = 3.84 cells, so the default reproduces 0.04 exactly at n_grid 96 and
        # follows the world everywhere else. An explicit `wall_contact` still wins, for the specs
        # that tuned it.
        self.wall_contact_cells = float(params.get("wall_contact_cells", 3.84))
        self.wall_contact = params.get("wall_contact", None)
        if self.wall_contact is not None:
            self.wall_contact = float(self.wall_contact)  # contact-layer thickness damped on bounce
        self.surface_tension = float(params.get("surface_tension", 0.0))  # liquid cohesion (CSF coefficient)
        self.vmax = float(params.get("vmax", 1e9))        # max cell speed (default: CFL only)
        self.dx = 1.0 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.compiled = None
        self._wall_key = None; self._wall_cache = None    # cached obstacle raster (per grid resolution)

    def _wall_mask(self, H, nx, ny, device):
        """Rasterize the world's obstacle rectangles/discs onto the MPM background
        grid (flat nx*ny, row-major i*ny+j to match the kernel's index). Cached per
        grid resolution. Obstacles come from the generic `general: obstacles:` list
        (H.obstacles) -- the same wall geometry the `bounce` operator reads -- so the
        MPM subsystem does not invent its own domain notion."""
        key = (nx, ny)
        if self._wall_key == key and self._wall_cache is not None:
            return self._wall_cache
        walls = torch.zeros(nx, ny, dtype=torch.bool, device=device)
        obs = list(getattr(H, "obstacles", []) or [])
        if obs:
            xs = (torch.arange(nx, device=device) + 0.5) * self.dx     # square cells dx=1/ny
            ys = (torch.arange(ny, device=device) + 0.5) * self.dx
            gx = xs[:, None].expand(nx, ny); gy = ys[None, :].expand(nx, ny)
            for rect in obs:
                v = [float(x) for x in rect]
                if len(v) == 4:                                        # wall rectangle [x0,y0,x1,y1]
                    walls = walls | ((gx >= v[0]) & (gx <= v[2]) & (gy >= v[1]) & (gy <= v[3]))
                elif len(v) == 3:                                      # disc obstacle [cx,cy,r]
                    walls = walls | (((gx - v[0]) ** 2 + (gy - v[1]) ** 2) <= v[2] ** 2)
        walls = walls.reshape(-1)
        self._wall_key = key; self._wall_cache = walls
        return walls

    def _require(self, H, p) -> None:
        """Fail loudly if the engine/entity has not provisioned this transitional
        operator's declared dependencies. These are engine-provisioned (not spec
        keys), so the schema cannot catch them -- but a precise error here beats an
        AttributeError deep inside a substep (the contract's 'fail before the run'
        spirit, applied to a transitional op's engine requirements)."""
        missing_buf = [b for b in self.REQUIRES_BUFFERS if not hasattr(p, b)]
        if missing_buf:
            raise RuntimeError(
                f"operator {type(self).__name__!r} requires per-particle buffer(s) {missing_buf} on "
                f"set {self.at!r} (REQUIRES_BUFFERS={self.REQUIRES_BUFFERS}); the engine/entity build "
                f"must allocate them (mass, deformation F, affine C, Lame mu/la, particle volume p_vol).")
        missing_h = [s for s in self.REQUIRES_HSTATE if getattr(H, s, None) is None]
        if missing_h:
            raise RuntimeError(
                f"operator {type(self).__name__!r} requires Hierarchy state {missing_h} "
                f"(REQUIRES_HSTATE={self.REQUIRES_HSTATE}).")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        self._require(H, p)                                # declared deps present, or fail loudly
        dev = p.state.device
        # external per-cell acceleration = the PARENT set's accumulated delta. A cell-level
        # force operator (e.g. gravity) returns {cell: g}; the engine accumulates it and --
        # since the cell has no EMIT -- never integrates it, so the MPM substep is free
        # to consume it here as a body force (no bespoke `H.cell_accel`).
        pn = getattr(p, "parent_name", None)
        if pn is not None:
            a_cell = H.delta(pn)                            # [Nc,2] accumulated parent force (zeros if none)
            a_cell = torch.nan_to_num(a_cell, posinf=self.a_max, neginf=-self.a_max).clamp(-self.a_max, self.a_max)
            a_ext = a_cell[p.parent]                        # broadcast down  [Np,2]
        else:
            a_ext = torch.zeros(p.n, 2, device=dev)
        part_accel = getattr(H, "part_accel", None)        # optional per-particle external accel
        if part_accel is not None:
            a_ext = a_ext + part_accel                     # (e.g. per-cell cohesion for identity)
        offsets = _OFFSETS.to(dev)
        width = float(getattr(H, "world_width", 1.0))      # rectangular world [0,width]x[0,1]
        ny = self.n_grid; nx = int(round(width * ny))      # square cells dx = 1/ny
        walls = self._wall_mask(H, nx, ny, dev)            # interior obstacles rasterized to the grid
        periodic = bool(getattr(H, "periodic", False))

        fn = self.compiled or mls_mpm_substep
        # read integrated state THROUGH THE SCHEMA (not hard-coded `p.state[:, :2]`)
        X, V = p.get("pos"), p.get("vel")
        C, F = p.C, p.F
        liquid = getattr(p, "is_liquid", None)             # per-particle liquid material mask (or None)
        if liquid is not None and not liquid.any():
            liquid = None                                  # all-solid -> skip the liquid branch entirely
        snow = getattr(p, "is_snow", None)                 # per-particle snow/plastic mask (or None)
        if snow is not None and not snow.any():
            snow = None                                    # no snow -> skip the SVD plasticity branch
        Jp = getattr(p, "Jp", None)

        # surface tension is injected as a proper CSF (continuum surface force) on the grid
        # inside the substep (see mls_mpm_substep); pass the coefficient through.
        surf = self.surface_tension if (self.surface_tension > 0 and liquid is not None) else 0.0
        for _ in range(self.substeps):
            X, V, C, F, Jp = fn(X, V, C, F, p.mass, p.mu, p.la, a_ext, offsets,
                                nx, ny, self.dx, self.inv_dx, self.dt_sub, p.p_vol, self.drag, walls,
                                self.vmax, periodic, width, self.wall_damp, self.wall_contact,
                                liquid, snow, Jp, surf)

        # write the integrated state back THROUGH THE SCHEMA (fenced direct mutation:
        # MAY_MUTATE_INTEGRATED_STATE=True -- the substep already integrated pos/vel).
        new_state = p.state.clone()
        pa, pb = p.state_schema["pos"]; va, vb = p.state_schema["vel"]
        new_state[:, pa:pb] = X
        new_state[:, va:vb] = V
        p.state = new_state
        p.C, p.F = C, F
        if Jp is not None:
            p.Jp = Jp
        return {}
