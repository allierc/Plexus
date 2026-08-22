"""membrane_ops -- MOVED to `plexus.operators.membrane_ops`.

Kept as a re-export because thirty files import it by bare module name -- `run_one.py`,
`instrument.py`, `vtk_render.py`, `metrics.py` and twenty archive/analysis scripts -- and the
campaign is still running against them. PRIVATE NAMES ARE RE-EXPORTED TOO: `_carry_face_state`,
`_engine_owns_clock` and friends are called across module boundaries in okuda, so a shim that
exported only the public surface would break at the first T1.

New code should import from `plexus.operators.membrane_ops`.
"""
from plexus.operators.membrane_ops import *          # noqa: F401,F403


# =============================================================================================
# NOT PROMOTED: `BasementMembraneContinuumStrain`, `MPMTissueBoundary`. `AUDIT.md` rejects them, so they were cut out of the module that moved to
# `src/plexus/operators/` and left here. They are still registered -- an archived spec that names
# one still runs -- but no spec can reach them from the core registry, and there is no alias to
# find them by. A rejection that lives only in a markdown file is a rejection that the next reader
# re-promotes by accident.
# =============================================================================================

@register_operator("bm_strain", family="mpm", set="particle", kind="lateral")
class BasementMembraneContinuumStrain(Lateral):
    """Per-particle strain read from the deformation gradient, for a membrane with no crosslinks.

    WHY IT EXISTS. Every membrane diagnostic in this prototype is bond-based -- crosslink strain colours
    the renderer, and `lcc` / `mean_degree_z` / `bonds_end` are the integrity measures. A basement
    membrane carried as an MPM continuum has no bonds at all, so those quantities do not become bad, they
    stop existing. This publishes the continuum equivalent into the same channel the renderer already
    reads, so the switch costs no plumbing downstream.

    THE MEASURE is the largest principal stretch minus one, sigma_max(F) - 1: the tensile strain in the
    direction the sheet is actually being pulled. det(F) is the wrong choice for a growing shell, because
    a sheet stretching biaxially while it thins is close to volume-preserving and would read J ~ 1 while
    being stretched to failure.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["basement_membrane", "continuum_strain"]
    PARAM_ROLES = {}
    REFERENCE = "Hu, Y. et al. (2018) ACM Trans. Graph. 37(4):150 (MLS-MPM)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        F = getattr(lvl, "F", None)
        if F is None:
            return {}
        alive = getattr(H, "membrane_alive", None)
        s = torch.linalg.svdvals(F)[:, 0] - 1.0        # largest principal stretch, as a strain
        # MASK, DO NOT MULTIPLY. The unsecreted reserve is parked at the tissue centre with zero mass and
        # a degenerate F, so its stretch is NaN -- and NaN * 0 is NaN, which poisons the whole array and
        # surfaces as the -1 sentinel after nan_to_num. `where` drops them instead of scaling them.
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        if alive is not None:
            s = torch.where(alive, s, torch.zeros_like(s))
        # CLAMPED BECAUSE THE CHANNEL IS float16. A parked reserve particle can integrate a runaway F,
        # and anything past 65504 becomes inf on the cast, then -1 through the reader's nan_to_num --
        # a sentinel that looks like a failed run rather than an unused particle.
        s = s.clamp(-1.0, 50.0)
        MEMBRANE_STRAIN.append(s.detach().to("cpu", torch.float16).numpy())
        if not self._said:
            print("[bm_strain] no crosslinks: colouring by sigma_max(F) - 1, "
                  "the continuum strain, in place of bond strain", flush=True)
            self._said = True
        return {}


@register_operator("mpm_boundary", family="boundary", set="field", kind="field")
class MPMTissueBoundary(FieldUpdate):
    """The growing epithelium as a MOVING no-slip boundary on the MPM grid.

    WHY THIS OPERATOR EXISTS, as the measurement that forced it. Run 88 carried the basement membrane as
    a pure MPM continuum and it tracked the spheroid perfectly -- R 0.0875 -> 0.2985, coverage 1.0 -- while
    reporting a peak strain of 7e-4 against a true in-plane stretch of 3.4x. The sheet was a DECAL. The
    log said why: 18,134 particles per frame were being repositioned by `cell_exclude`, a hard
    positional projection. Moving a particle by hand does not touch its deformation gradient, so the
    material never learns it was stretched, and a body at zero strain cannot tear, cannot resist growth,
    and cannot load the stroma.

    THE FIX IS WHERE MPM PUTS BOUNDARIES: on the grid. `mpm_grid_update` already zeroes grid velocity
    inside rasterized obstacles -- correct for a wall, wrong for a growing tissue, which is an obstacle
    with a VELOCITY. Setting the grid velocity inside the tissue to the surface's own expansion velocity
    makes the growth enter as momentum: particles gather it, the affine gradient C becomes non-zero, and
    F integrates the stretch that a projection threw away. This is the standard collision-object
    treatment (Stomakhin 2013), not a new mechanism.

    IT USES THE REAL TISSUE, not a sphere fitted to it: the same `smap` table the membrane is seeded on
    and the integrin anchors read, so the boundary and the sheet cannot disagree about where the surface
    is. R is looked up per direction per frame, and Rdot from the difference between consecutive frames.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["moving_boundary", "no_slip", "tissue_grid_coupling"]
    PARAM_ROLES = {"surface": "tissue_surface_radius_map", "dt_frame": "frame_dt_for_Rdot",
                   "scale": "surface_rescale", "shell": "band_outside_R_also_driven"}
    REFERENCE = ("Stomakhin, A. et al. (2013) ACM Trans. Graph. 32(4):102 (collision objects imposed on "
                 "the grid); Hu, Y. et al. (2018) ACM Trans. Graph. 37(4):150 (MLS-MPM).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_grid")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.dt_frame = float(params.get("dt_frame", 4.0e-3))
        self.band = float(params.get("band", 2.0))        # surface band width, in grid cells
        self.recover = float(params.get("recover", 2.0))  # frames to clear an overlap; 0 = velocity only
        import numpy as _np
        z = _np.load(str(params["surface"]))
        self.smap = torch.as_tensor(_np.asarray(z["smap"], _np.float32)) * self.scale
        self.T = int(self.smap.shape[0])
        self._xyz = None
        self._ang = None          # bin indices + bilinear weights: constant, the grid does not move
        self._said = False

    def _node_dirs(self, g, dev, dt_):
        """Unit direction and radius of every grid node, cached -- the grid does not move."""
        if self._xyz is not None:
            return self._xyz
        nx, ny, nz = g.nx, g.ny, getattr(g, "nz", g.ny)
        dx = g.dx
        ii = torch.arange(nx, device=dev, dtype=dt_)
        jj = torch.arange(ny, device=dev, dtype=dt_)
        kk = torch.arange(nz, device=dev, dtype=dt_)
        X, Y, Z = torch.meshgrid(ii, jj, kk, indexing="ij")
        P = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3) * dx
        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        d = P - c
        r = d.norm(dim=-1).clamp_min(1e-9)
        self._xyz = (d / r[:, None], r)
        return self._xyz

    def forward(self, H, mask=None):
        g = H.field(self.at)
        dev, dt_ = g.m.device, g.mv.dtype
        f = int(getattr(H, "frame", 0) or 0)
        t = min(self.T - 1, max(0, f))
        tn = min(self.T - 1, t + 1)
        u, r = self._node_dirs(g, dev, dt_)
        M = self.smap[t].to(dev, dt_)
        Mn = self.smap[tn].to(dev, dt_)
        nth, nph = M.shape
        # THE GRID DOES NOT MOVE, so every node's direction -- and therefore its bin indices and its
        # bilinear weights -- is constant for the whole run. Recomputing acos/atan2/floor over 110,592
        # nodes at every substep was 24% of all operator time across 420 calls per 20 frames. Cached,
        # the arithmetic is unchanged, so results stay bit-identical.
        if self._ang is None:
            th_ = torch.acos(u[:, 2].clamp(-1.0, 1.0))
            ph_ = torch.atan2(u[:, 1], u[:, 0])
            fth_ = (th_ / math.pi * nth - 0.5).clamp(0, nth - 1)
            fph_ = ((ph_ + math.pi) / (2 * math.pi) * nph - 0.5) % nph
            t0_ = fth_.floor().long().clamp(0, nth - 1); t1_ = (t0_ + 1).clamp(0, nth - 1)
            p0_ = fph_.floor().long() % nph; p1_ = (p0_ + 1) % nph
            self._ang = (t0_, t1_, p0_, p1_,
                         (fth_ - t0_.to(fth_.dtype)), (fph_ - p0_.to(fph_.dtype)))
        t0, t1, p0, p1, wt, wp = self._ang
        # BILINEAR, NOT NEAREST-BIN. The map is 32x64, so a bin subtends a fixed ANGLE and therefore a
        # growing LENGTH: at the last frame its equatorial width is 1.40 grid cells. Read by nearest bin
        # that is a staircase whose steps outgrow the grid, and the membrane -- which is smooth -- ends
        # up inside the step. Measured on 94: the fraction of the sheet lying inside the surface goes
        # 0.0% -> 7.8% -> 31.3% -> 46.6% over the run, exactly as the steps widen.
        #
        # This removes the staircase; it does not add information. With ~2,800 vertices against 2,048
        # bins the map is at its resolution limit, so a sheet that must follow the true apical surface
        # needs the FACES, not a finer radius map.
        def _bil(A):
            return ((A[t0, p0] * (1 - wp) + A[t0, p1] * wp) * (1 - wt)
                    + (A[t1, p0] * (1 - wp) + A[t1, p1] * wp) * wt)
        R = _bil(M)
        Rdot = (_bil(Mn) - R) / max(self.dt_frame, 1e-9)
        # ON NODES THAT CARRY MASS, IN A BAND AROUND THE SURFACE -- not on the nodes strictly inside.
        # Run 90 imposed the velocity inside r < R and the membrane did not move one step (R stayed
        # 0.0875 for 402 frames) for a reason that is obvious afterwards: the LUMEN IS EMPTY. There are
        # no MPM particles inside the epithelium, so those nodes have zero mass, `mv = v*m` writes
        # nothing, and G2P gathers nothing from them. The boundary was being imposed exactly where
        # there is no material to impose it on. The material is in the shell just OUTSIDE the surface,
        # so that is where the constraint has to act.
        band = self.band * g.dx
        # A SHELL, NOT A BALL. `r < R + band` is every node inside the tissue as well, which was
        # harmless while the constraint only topped velocity up to Rdot -- bounded -- and is not harmless
        # with a penetration term: a node near the centre has pen = R, so it was handed an enormous
        # outward velocity, and membrane particles sampling it through the B-spline stencil were flung
        # out. That is the constant standoff visible from the very first frame.
        near = (r > R - band) & (r < R + band) & (g.m > 1.0e-9)
        if bool(near.any()):
            gm = g.m.clamp(min=1e-10)
            # `g.v`, NOT `g.mv`. `mpm_grid_update` ends by writing the solved velocity into `g.v`, and
            # `mpm_gather` reads `g.v[flat]` -- so an operator that edits momentum after the grid solve
            # is editing a field nobody downstream looks at. That single mistake is the whole of runs
            # 89, 90 and 91: the boundary condition ran every substep (140 calls in 6 frames, verified)
            # and was invisible, which reads exactly like a physical null.
            gv = g.v if getattr(g, "v", None) is not None else g.mv / gm[:, None]
            # SEPARATING, NOT NO-SLIP. Only the outward NORMAL component is corrected, and only when the
            # material is moving slower than the surface -- so the tissue pushes material out of its way
            # and never pulls it back in or drags it tangentially. Full no-slip would weld the sheet to
            # the epithelium, which is what the integrin anchor is for and is a different experiment.
            vn = (gv * u).sum(-1)
            # VELOCITY ALONE CANNOT RECOVER LOST GROUND. Topping the normal velocity up to Rdot keeps
            # material moving WITH the surface but never pushes back material the surface has already
            # overtaken -- so the stroma's back-pressure squeezes the sheet inward between corrections
            # and the standoff decays monotonically: +0.0081 at frame 100 to +0.0006 at 402, with 47% of
            # the sheet ending up inside the epithelium. A wider band hides it by catching the sheet
            # earlier, which is what band 2.0 was doing, at the cost of a visible gap.
            #
            # The penetration term makes it a real non-penetration constraint: anything inside R is given
            # the extra outward speed that clears the overlap over `recover` frames, on top of Rdot.
            # Zero reproduces the previous behaviour exactly.
            pen = (R - r).clamp(0.0, band)      # clamped: at most one band's worth of overlap per step
            v_want = Rdot + pen / max(self.recover * self.dt_frame, 1e-9)
            push = near & (vn < v_want)
            gv_new = torch.where(push[:, None], gv + (v_want - vn)[:, None] * u, gv)
            # THE OTHER HALF OF THE COUPLING, and it costs one subtraction. The momentum the boundary
            # must INJECT to hold its prescribed motion is exactly what the material is exerting back on
            # the tissue: dp = m*(v_bc - v_free), summed over the constrained nodes. Pass 2 replays a
            # tissue recorded in pass 1 and so cannot be pushed by it -- but this is the force that would
            # push it, measured rather than assumed, and it is what pass 1 would have to be given for a
            # genuinely two-way run. Its radial component is the one that matters: negative = the
            # matrix and membrane are resisting the growth.
            dp = torch.nan_to_num(((gv_new - gv) * gm[:, None])[near], nan=0.0,
                                  posinf=0.0, neginf=0.0)
            fr = (dp * u[near]).sum(-1)
            BOUNDARY_REACTION.append((int(getattr(H, "frame", 0) or 0),
                                      float(dp.norm(dim=-1).sum()), float(fr.sum()),
                                      int(near.sum())))
            g.v = gv_new
            g.mv = gv_new * gm[:, None]
        if not self._said:
            print(f"[mpm_boundary] tissue imposed on the grid: {int(near.sum())} massed nodes "
                  f"in the surface band of {r.numel()} at frame {f}, max Rdot {float(Rdot.max()):.4g}",
                  flush=True)
            self._said = True
        return {}

