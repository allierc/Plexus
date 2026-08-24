"""MLS-MPM, as one module: the grid, the four-step cycle, and the two forces on it.

    mpm_grid            the background FIELD and the quadratic B-spline kernel (not an operator)
    mpm_scatter (p2g)   particle -> grid: mass, momentum, and the internal stress impulse
    mpm_grid_update     grid -> grid: the solve, gravity, and the wall conditions
    mpm_gather (g2p)    grid -> particle: velocity, the affine C, and advection
    mpm_strain          particle -> particle: F update and the material's response
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
def stencil_offsets(dim: int, device="cpu") -> torch.Tensor:
    """The 3^dim quadratic-B-spline stencil offsets, row-major (last axis fastest).
    2D -> [9,2] == `[[i,j] for i in 0..2 for j in 0..2]`; 3D -> [27,3] (matches the
    MPM_3D offset ordering: idx//9, (idx%9)//3, idx%3)."""
    return torch.tensor(list(itertools.product(range(3), repeat=dim)),
                        dtype=torch.float32, device=device)


# 2D stencil kept as a module constant for back-compat (p2g/g2p now build per-dim)
_OFFSETS = stencil_offsets(2)


@register_field("mpm_grid")
class MPMGrid(Field):
    """MLS-MPM background grid on [0,width]x[0,1](x[0,1]) with square cells dx = 1/n_grid.
    Channels: m (mass), mv (momentum [.,dim]), c (liquid colour for CSF), v (velocity
    [.,dim]). Pure scratch: p2g zeroes + scatters into it each substep, grid_update
    solves on it, g2p reads it back."""

    RECORD = False                                   # transient scratch -- not recorded/rendered

    def __init__(self, name, width=1.0, n_grid=128, dim=2, device="cpu", **kw):
        super().__init__(name)
        self.dim = int(dim)
        self.ny = int(n_grid)                        # cells per unit length (axes 1..)
        self.nx = int(round(float(width) * self.ny)) # axis 0 spans the world width
        self.width = float(width)
        self.dx = 1.0 / self.ny
        self.inv_dx = float(self.ny)
        if self.dim == 2:
            self.shape = (self.nx, self.ny)
        else:                                        # 3D cube: axes 1,2 span [0,1]
            self.nz = self.ny
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
            c = torch.cross(R[:, :, [1, 2, 0]], R[:, :, [2, 0, 1]], dim=1)
            det = (R[:, :, 0] * c[:, :, 0]).sum(1)[:, None, None]
            det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
            R = 0.5 * (R + c / det)
    else:
        eyeT = torch.eye(D, device=F.device, dtype=F.dtype)
        for _ in range(iters):
            R = 0.5 * (R + torch.linalg.solve(R, eyeT).transpose(-2, -1))
    return R


@register_operator("mpm_scatter", "p2g", family="mpm", set="particle", kind="exchange")
class MPMScatter(Exchange):                 # (alias `p2g`, one migration cycle)
    EMIT = None                 # particle->grid: writes the mpm_grid field in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []        # no required params — `to` defaults to mpm_grid, all knobs optional
    REQUIRES_TYPE_PROPS = ["youngs"]
    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress"]
    PARAM_ROLES = {"dt_sub": "MLS-MPM substep dt", "drag": "Stokes drag coefficient",
                   "a_max": "external-acceleration clamp",
                   "store_stress": "cache Cauchy stress to a per-particle buffer"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM P2G); Sulsky, D. et al. (1994)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.to = params.get("to", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.drag = float(params.get("drag", 0.0))
        self.a_max = float(params.get("a_max", 200.0))
        # HOW THE POLAR ROTATION IS FOUND, in 3-D. The fixed-corotated stress needs R from
        # F = R S, and the obvious way to get it is an SVD -- but `torch.linalg.svd` on a
        # batch of 3x3 matrices costs about a microsecond EACH, and this operator runs once
        # per particle per substep: 45,000 particles x 25 substeps is 1.1 million 3x3 SVDs a
        # frame, and on the zebrafish eye that single call measured 44.7 ms of the operator's
        # 46.4 ms. "higham" replaces it with the Newton polar iteration
        # R <- (R + R^-T)/2, which converges quadratically from F and costs 6.4 ms for the
        # same batch -- 7x -- agreeing with the SVD rotation to 1.5e-6 with an orthogonality
        # error of 2.4e-7, i.e. to float32. Default stays "svd": identical numbers unless asked.
        self.polar = str(params.get("polar", "svd")).lower()
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
        if snow is not None and snow.any():                        # snow hardening from the plastic ratio Jp
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
            U, sig, Vh = torch.linalg.svd(F)
            U = U.clone(); Vh = Vh.clone()
            negU = torch.det(U) < 0; U[negU, :, -1] *= -1
            negV = torch.det(Vh) < 0; Vh[negV, -1, :] *= -1
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
        gm = torch.zeros(g.n_cells, device=dev); gmv = torch.zeros(g.n_cells, D, device=dev)
        gm.index_add_(0, flat, (weight * mass[:, None]).reshape(-1))
        gmv.index_add_(0, flat, (weight[..., None] * mom).reshape(-1, D))
        gc = torch.zeros(g.n_cells, device=dev)
        liquid = getattr(p, "is_liquid", None)
        if liquid is not None and liquid.any():                    # liquid colour for the CSF surface tension
            lw = (weight * (mass * liquid.to(mass.dtype))[:, None]).reshape(-1)
            gc.index_add_(0, flat, lw)
        g.m, g.mv, g.c = gm, gmv, gc
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
                   "wall_damp": "wall_restitution"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150; Sulsky, D. et al. (1994)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.surface_tension = float(params.get("surface_tension", 0.0))
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
        self._wall_key = None; self._wall_cache = None
        self._wall3d_key = None; self._wall3d_cache = None

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

    def forward(self, H, mask=None):
        g = H.field(self.at); dev = g.m.device
        dt = sub_dt(H, self.dt_sub)
        nx, ny, inv_dx, dx = g.nx, g.ny, g.inv_dx, g.dx
        D = g.dim
        periodic = bool(getattr(H, "periodic", False))
        wd = self.wall_damp
        gm, gmv, gc = g.m, g.mv, g.c
        gv = gmv / gm.clamp(min=1e-10)[:, None]

        if D == 2:                                                  # --- 2D: verbatim (bit-identical) ---
            surf = self.surface_tension
            if surf > 0.0 and bool((gc > 0).any()):                # CSF continuum surface force
                c = gc.view(nx, ny)
                cx = (torch.roll(c, -1, 0) - torch.roll(c, 1, 0)) * (0.5 * inv_dx)
                cy = (torch.roll(c, -1, 1) - torch.roll(c, 1, 1)) * (0.5 * inv_dx)
                gmag = torch.sqrt(cx * cx + cy * cy); eps = 1e-6
                nxg, nyg = cx / (gmag + eps), cy / (gmag + eps)
                kappa = -((torch.roll(nxg, -1, 0) - torch.roll(nxg, 1, 0)) * (0.5 * inv_dx)
                          + (torch.roll(nyg, -1, 1) - torch.roll(nyg, 1, 1)) * (0.5 * inv_dx))
                fmask = (gmag > 0.02 * gmag.max()).to(c.dtype)
                stfx = (surf * kappa * cx * fmask).view(-1); stfy = (surf * kappa * cy * fmask).view(-1)
                inv_m = (dx * dx) / gm.clamp(min=1e-8)
                gv = gv + dt * torch.stack([stfx * inv_m, stfy * inv_m], dim=1)

            if not periodic:                                        # reflective domain walls
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
                gv = gv.view(nx * ny, 2)
            walls = self._walls(H, g, dev)
            if wd != 1.0 and walls.any():                          # friction in fluid cells touching obstacles
                w2 = walls.view(nx, ny)
                near = (torch.roll(w2, 1, 0) | torch.roll(w2, -1, 0)
                        | torch.roll(w2, 1, 1) | torch.roll(w2, -1, 1)) & ~w2
                gvv = gv.view(nx, ny, 2); gx_ = gvv[..., 0]; gy_ = gvv[..., 1]
                gvv[..., 0] = torch.where(near, gx_ * wd, gx_)
                gvv[..., 1] = torch.where(near & (gy_ > 0), gy_ * wd, gy_)
                gv = gvv.view(nx * ny, 2)
            gv = torch.where(walls[:, None], torch.zeros_like(gv), gv)  # interior wall BC
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
        else:                                                       # --- 3D: reflective box walls + friction ---
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
            walls = self._walls3d(H, g, dev)                        # solid 3D obstacles (box / sphere)
            if walls.any():
                gv = torch.where(walls[:, None], torch.zeros_like(gv), gv)   # no-slip: zero grid velocity inside
        g.v = gv
        return {}


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
                   "wall_contact": "contact_layer_thickness", "vmax": "speed_cap"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM G2P); Sulsky, D. et al. (1994)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.frm = params.get("from", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.wall_damp = float(params.get("wall_damp", 1.0))
        self.wall_contact = float(params.get("wall_contact", 0.04))
        self.vmax = float(params.get("vmax", 1e9))

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.frm); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        inv_dx, dx = g.inv_dx, g.dx
        D = p.F.shape[-1]
        periodic = bool(getattr(H, "periodic", False))
        box = [float(b) for b in getattr(H, "world_size", torch.tensor([g.width, 1.0]))][:D]
        offsets = stencil_offsets(D, dev); S = offsets.shape[0]
        X, V = p.get("pos"), p.get("vel")
        fx, weight, flat = bspline(X, inv_dx, offsets, g.shape, periodic)
        gvn = g.v[flat].view(p.n, S, D)
        new_V = (weight[..., None] * gvn).sum(1)
        dpos_grid = offsets[None] - fx[:, None, :]
        new_C = 4 * inv_dx * (weight[..., None, None] * (gvn[..., :, None] @ dpos_grid[..., None, :])).sum(1)
        new_V = torch.nan_to_num(new_V)
        if self.wall_damp != 1.0 and not periodic:                 # inelastic wall contact (solids)
            cb = self.wall_contact
            near = torch.zeros(p.n, dtype=torch.bool, device=dev)
            for k in range(D):
                near = near | (X[:, k] < cb) | (X[:, k] > box[k] - cb)
            liquid = getattr(p, "is_liquid", None)
            if liquid is not None:
                near = near & ~liquid
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
        new = p.state.clone()
        pa, pb = p.state_schema["pos"]; va, vb = p.state_schema["vel"]
        new[:, pa:pb] = Xn; new[:, va:vb] = new_V
        p.state = new
        p.C = new_C
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
        if liquid is not None and liquid.any():                    # LIQUID: drop shape memory
            Jc = J.clamp(min=1e-6)
            Jl = torch.sqrt(Jc) if D == 2 else Jc.pow(1.0 / D)     # volume-preserving isotropic reset
            F = torch.where(liquid[:, None, None], eye * Jl[:, None, None], F)
        visco = getattr(p, "is_visco", None)
        if visco is not None and visco.any():                      # VISCOELASTIC (Maxwell): PARTIAL shape reset
            vm = visco                                             # relax F toward isotropic with time-constant tau,
            Fv = F[vm]                                             # keeping VOLUME (J) -> stress builds then decays
            U, sig, Vh = torch.linalg.svd(Fv)                      # SVD: sig = principal stretches
            Jl = sig.prod(-1).clamp(min=1e-6).pow(1.0 / D)         # isotropic (volume-preserving) target stretch
            a = torch.exp(-dt / p.visco_tau[vm].clamp(min=1e-6))   # memory retained this substep: a->1 elastic, a->0 liquid
            sig = Jl[:, None] + (sig - Jl[:, None]) * a[:, None]   # pull stretches toward isotropic (shear relaxes, volume kept)
            F = F.clone()
            F[vm] = U @ torch.diag_embed(sig) @ Vh
        snow = getattr(p, "is_snow", None)
        if snow is not None and snow.any():                        # SNOW: clamp singular values, harden via Jp
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
                p.Jp = Jp
        # DORMANT PARTICLES DO NOT DEFORM. `mpm_scatter` masks its weights by occupancy and
        # `mpm_gather` freezes occ==0 rather than advecting it, but this operator integrated F for the
        # reserve regardless -- so a particle waiting to be spawned accumulated an arbitrary deformation
        # for as long as it waited, and was then promoted into real material carrying it. Byte-identical
        # when every particle is live, which is every composition that has no reserve.
        occ = getattr(p, "occ", None)
        if occ is not None:
            live = (occ > 0)[:, None, None]
            F = torch.where(live, F, p.F)
        p.F = F
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
        self.ring = float(params.get("ring", 0.04))           # ring width (world units) for mode=boundary
        self.at = params.get("_at", "particle")
        self._rest = None
        self._sel = None

    def _init(self, lvl):
        self._rest = lvl.get("pos").clone()                   # undeformed sheet (frame 0)
        if self.mode == "substrate":
            self._sel = torch.ones(self._rest.shape[0], dtype=torch.bool, device=self._rest.device)
        else:                                                 # outer ring of the tissue's rest extent
            lo = self._rest.min(0).values
            hi = self._rest.max(0).values
            near = ((self._rest - lo) < self.ring) | ((hi - self._rest) < self.ring)   # [N,2]
            self._sel = near.any(dim=1)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self._rest is None:
            self._init(lvl)
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
    REQUIRES_TYPE_PROPS = ["youngs"]                      # per-cell-type stiffness -> mu, la
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
        self.a_max = float(params.get("a_max", 200.0))    # clamp broadcast accel
        self.drag = float(params.get("drag", 40.0))       # Stokes drag (overdamped)
        self.wall_damp = float(params.get("wall_damp", 1.0))  # 1.0=elastic wall; <1 loses energy on bounce
        self.wall_contact = float(params.get("wall_contact", 0.04))  # contact-layer thickness damped on bounce
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
