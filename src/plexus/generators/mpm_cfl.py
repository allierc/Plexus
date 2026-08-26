"""Grid-dt CFL auto-correction for MPM specs -- applied to the YAML, not in code.

The MLS-MPM substep is explicit, so it must obey the Courant (grid-dt) condition: an
elastic wave may not cross more than ~`cfl` of a background-grid cell per substep,

    dt_sub <= cfl * dx / c ,   c = sqrt((lambda + 2*mu) / rho)   (elastic P-wave speed),

with dx = 1/n_grid. c is set by the STIFFEST material (largest `youngs`); liquids
(mu = 0) are slow and usually already safe. When the micro-timestep exceeds the limit
this WARNS and -- per the 'the correction lives in the spec, not in the engine' rule --
rewrites the spec.yaml in place, lowering it to the stable value while PRESERVING the
per-frame simulated time (same physics, now stable). Idempotent: a corrected spec is
left unchanged on re-run.

THE ELASTIC WAVE IS NOT THE ONLY WAVE. A liquid with `surface_tension` also carries CAPILLARY
waves, and the shortest one the mesh admits (wavelength 2*dx) has its own explicit limit,

    dt_cap <= cfl * sqrt(rho_liquid * dx^D / (2*pi*sigma))    (Brackbill/Kothe/Zemach 1992),

which is INDEPENDENT of the elastic one and can be much smaller. It only became reachable when
`mpm_grid_update` learned to normalise its colour field (`csf_rho`): before that `surface_tension`
was a tension divided by a cell mass, 1e6 too small, and no spec could get near this limit. So the
capillary term is computed ONLY when `csf_rho` is set, and is +inf otherwise -- an unnormalised
`surface_tension` is not a physical tension and putting it in this formula would produce a
nonsense limit for every legacy spec.

CALIBRATION, MEASURED, NOT ASSUMED (material_3d_water_drop, 290k particles, n_grid 96, 60 frames,
csf_rho 1.0, csf_band 0.2, at a FIXED sigma of 1.12 = Bond 0.5 so only dt varies): the peak node
mass, in units of one FULL liquid cell rho*dx^D, reads 1.80 / 10.7 / 60.8 at dt/dt_cap_raw =
0.50 / 0.80 / 1.00. A clean run sits at ~1.8 full cells (the sigma = 0 twin reads 2.0), so the raw
Brackbill limit is already 34x into pile-up and half of it is clean -- hence the same `cfl` safety
factor the elastic limit uses (0.4 by default) is applied to it, leaving 1.25x of margin on the
measured edge.

It handles BOTH MPM spec forms:
  * the ORACLE op `mls_mpm_mechanics`: micro-timestep `dt_sub` + an explicit `substeps`
    on the op line -- it lowers `dt_sub` and raises `substeps` (per-frame = substeps*dt_sub).
  * the DECOMPOSED cycle: a schedule substep block `{substep_dt: X, steps: [mpm_strain,
    mpm_scatter, mpm_grid_update, mpm_gather]}` -- the substep COUNT is implicit
    (round(general.dt / substep_dt)), so it lowers only `substep_dt` and the count auto-rises.
Either way it edits only the relevant token(s), so comments and layout are preserved.
"""
from __future__ import annotations

import math
import os
import re

import yaml

from plexus.paths import warn

_NU = 0.2                                    # Poisson ratio (matches the mpm_particle entity)


def _lame(E, nu=_NU):
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, la


def _max_wave_speed(spec, rho_default=1.0) -> float:
    """Largest elastic P-wave speed over EVERY set's materials (mu = 0 for liquid).

    WALKS ALL SETS, NOT `sets["cell"]`. This used to read the type table of one hard-coded set
    name and take the density from another (`sets["mpm_particle"]["density"]`, defaulting to 1.0).
    Both are wrong for a composed body, where the materials live on the CHILD sets: for the cell
    ladder it was reading the parent `cell` set's placeholder `youngs: 40` and never seeing the
    nucleus at 300 or the membrane at 1500, then dividing by rho = 1.0 instead of 2.6 or 1.1.

    THE FAILURE DIRECTION IS THE DANGEROUS ONE. Reading a soft placeholder makes c too SMALL, so
    the computed limit is too LOOSE and the guard silently passes a substep that should have been
    tightened. It has not bitten yet only because every cell spec was hand-set to 1e-4, well under
    both the wrong limit and the right one -- see the headroom column this now reports.

    Density is per SET, overridable per TYPE, exactly as `models/entities.py` assigns it.
    """
    # THE DENSITY AND THE MATERIAL CAN LIVE ON DIFFERENT SETS, and pairing them within one set is
    # wrong for exactly the composition this repo uses most: the 27 water blocks declare `youngs`
    # and `material` on the PARENT (`cell`) types, while `density` sits on the CHILD
    # (`mpm_particle`) set. Walking sets independently then gave the parent rho = 1.0 (it has no
    # density) and the child no types (it has none), so a spec at density 0.25 was evaluated at
    # 1.0 -- c came out 7.46 instead of 14.9, the limit was twice too loose, and
    # material_3d_water_dam_20m_rho0p25 was passed as stable and blew up: by frame 1 its material
    # spanned [0.010, 0.990] on all three axes. That is the failure direction this function's own
    # docstring calls the dangerous one, in a case it did not cover.
    #
    # So: for every set that declares a density, find the sets whose types describe ITS material --
    # itself, and its parent if it names one -- and pair them.
    sets = spec.get("sets") or {}
    # DECLARED DENSITIES FIRST, DEFAULTS ONLY WHERE NOTHING WAS DECLARED. The previous form did
    # `setdefault(sname, st.get("density", rho_default))` and then `min(existing, child_density)`;
    # since `cell` is listed before `mpm_particle`, the parent got the DEFAULT 1.0 first and the
    # min() then kept it. Invisible while every spec used density 1.0, and worth a factor of
    # sqrt(1000) = 31.6 in the wave speed the moment one does not: an SI water spec at rho 1000
    # with K 1e5 read c = 316.2 m/s instead of 10.0 and was re-timed to 633 substeps a frame
    # instead of 21.
    rho_decl = {sn: float(st["density"]) for sn, st in sets.items()
                if isinstance(st, dict) and "density" in st}
    rho_for = {}
    for sname, st in sets.items():
        if not isinstance(st, dict):
            continue
        rho_for[sname] = rho_decl.get(sname, rho_default)
        par = st.get("parent")
        if par in sets and sname in rho_decl:
            # the child's particles carry the child's density, and their material comes from the
            # parent's type table; attribute the lower of the DECLARED densities to the parent so a
            # heavy child is not missed -- but never let an undeclared parent's default win.
            cand = [rho_decl[sname]] + ([rho_decl[par]] if par in rho_decl else [])
            rho_for[par] = min(cand)
    cmax = 0.0
    for sname, st in sets.items():
        if not isinstance(st, dict):
            continue
        rho_set = rho_for.get(sname, float(st.get("density", rho_default)))
        for t in (st.get("types") or {}).values():
            if not isinstance(t, dict):
                continue
            rho = float(t.get("density", rho_set))          # per-type density wins (light/heavy cytosol)
            layers = list(t.get("layers") or [{"youngs": t.get("youngs"),
                                               "bulk_modulus": t.get("bulk_modulus"),
                                               "material": t.get("material", "elastic")}])
            if t.get("core"):
                layers.append(t["core"])
            for L in layers:
                mat = L.get("material", t.get("material", "elastic"))
                # `bulk_modulus` IS lambda for a liquid (mu = 0), so it goes straight in; without it
                # the pass would fall back to the youngs default of 100 and report a wave speed for
                # a material the spec never declared -- which is how an SI spec first read c = 5.3
                # instead of 10.0.
                K = L.get("bulk_modulus", t.get("bulk_modulus"))
                if K is not None and mat == "liquid":
                    mu, la = 0.0, float(K)
                else:
                    E = L.get("youngs", t.get("youngs"))
                    mu, la = _lame(float(E if E is not None else 100.0))
                    if mat == "liquid":
                        mu = 0.0             # liquid carries no shear modulus
                cmax = max(cmax, math.sqrt(max(la + 2 * mu, 0.0) / max(rho, 1e-12)))
    return cmax


def _cell_size(spec, n_grid: int, dim: int) -> float:
    """The background cell size, `world_size[1] / n_grid`, matching MPMGrid.

    THIS USED TO BE `1.0 / n_grid`, with a comment asserting "world is unit-width". That is true of
    every spec written so far -- of 1,744 specs, the 94 MPM ones that declare `general.world` all
    declare [1.0, 1.0, 1.0] -- and it is exactly the assumption a metres-and-seconds spec breaks. It
    matters here more than almost anywhere else in the codebase: dx enters this guard as
    `dt <= cfl * dx / c`, so a dx ten times too large BLESSES A SUBSTEP TEN TIMES TOO LARGE. The one
    pass whose job is to stop a blow-up would wave it through.

    Reads the same `general.world` that schema.py parses, with the same scalar/list handling.
    """
    w = (spec.get("general") or {}).get("world", 1.0)
    if isinstance(w, (list, tuple)):
        box = [float(x) for x in w]
    else:
        box = [float(w)] + [1.0] * (dim - 1)         # scalar: axis-0 width, the rest unit
    return box[1] / float(n_grid) if len(box) > 1 else box[0] / float(n_grid)


def _capillary_limit(spec, dx: float, dim: int):
    """The Brackbill/Kothe/Zemach capillary-wave limit, or (inf, 0, 0) when it does not apply.

        dt_cap_raw = sqrt(rho_liquid * dx^D / (2*pi*sigma))

    is the period of the shortest capillary wave the mesh can hold (wavelength 2*dx) divided by
    2*pi; an explicit scheme that steps past it grows that wave instead of propagating it. It is
    the liquid's OWN limit and has nothing to do with `youngs`, so a pure-liquid spec -- exactly
    the case where the elastic limit is loosest -- is where it bites hardest.

    ONLY WHEN THE COLOUR IS NORMALISED. `mpm_grid_update` applies f = sigma*kappa*grad(c) to a
    colour `c` that is a liquid MASS PER NODE unless `csf_rho` is set, so without `csf_rho` the
    yaml's `surface_tension` is a tension divided by rho*dx^D -- 1e6 too small at n_grid 96, and a
    different physical tension at every n_grid. Feeding that number to this formula would invent a
    limit ~1e3 too small and re-time every legacy spec on the corpus for a force that is not there.
    So: no `csf_rho`, no capillary term. This function is a no-op for every spec written before the
    normalisation existed, which is all of them but the water_st ladder.

    rho is the LIQUID's density and `csf_rho` IS that density by definition (it is the divisor that
    turns the deposited mass into a volume fraction), so it is read from the same place rather than
    re-derived from the sets -- one number, one meaning. Brackbill writes rho_avg = (rho_1+rho_2)/2
    for two fluids; here the second phase is vacuum, so rho_avg = rho_liquid/2 would be the strict
    reading and using rho_liquid is the LOOSER of the two by sqrt(2). The `cfl` factor applied by
    the caller (0.4) is 3.5x tighter than that sqrt(2), so nothing is lost by the simpler choice.

    Several `mpm_grid_update` operators (a spec may carry more than one) -> the SMALLEST limit."""
    dt_cap, sig_at, rho_at = float("inf"), 0.0, 0.0
    for o in (spec.get("operators") or []):
        if not isinstance(o, dict):
            continue
        try:
            sigma = float(o.get("surface_tension", 0.0) or 0.0)
            rho_l = float(o.get("csf_rho", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if sigma <= 0.0 or rho_l <= 0.0:
            continue                                 # legacy / unnormalised / tension off
        d = math.sqrt(rho_l * dx ** dim / (2.0 * math.pi * sigma))
        if d < dt_cap:
            dt_cap, sig_at, rho_at = d, sigma, rho_l
    return dt_cap, sig_at, rho_at


def _liquid_scale(spec, dim: int):
    """(softest liquid bulk modulus K, smallest liquid body radius R) -- (inf, inf) if no liquid.

    A liquid in MLS-MPM has mu = 0, so its bulk modulus IS lambda = E*nu/((1+nu)(1-2nu)); at the
    corpus default nu = 0.2 that is 0.2778*E, i.e. `youngs: 200` buys a bulk modulus of 55.6 and
    NOT 200. The SOFTEST liquid and the SMALLEST body are taken because both are the worst case
    for the ratio below: least resistance, largest curvature.

    R is an equivalent radius from the declared extent (`block`) or `radius`, so it is the body as
    WRITTEN, before it falls and spreads. Curvature only grows as a drop breaks up, so this is
    again the optimistic end and the report below is a floor on the real number."""
    sets = spec.get("sets") or {}
    K, R = float("inf"), float("inf")
    for sname, st in sets.items():
        if not isinstance(st, dict):
            continue
        rad_default = float(st.get("radius", 0.05) or 0.05)
        for t in (st.get("types") or {}).values():
            if not isinstance(t, dict):
                continue
            layers = list(t.get("layers") or [{"youngs": t.get("youngs"),
                                               "bulk_modulus": t.get("bulk_modulus"),
                                               "material": t.get("material", "elastic")}])
            if t.get("core"):
                layers.append(t["core"])
            liq = [L for L in layers
                   if L.get("material", t.get("material", "elastic")) == "liquid"]
            if not liq:
                continue
            for L in liq:
                # `bulk_modulus` says it outright; `youngs` reaches the same number through a
                # modulus a liquid does not have. Either way mu = 0, so the bulk modulus IS lambda.
                _K = L.get("bulk_modulus", t.get("bulk_modulus"))
                if _K is None:
                    _E = L.get("youngs", t.get("youngs"))
                    _mu, _K = _lame(float(_E if _E is not None else 100.0))
                K = min(K, float(_K))
            b = t.get("block")
            if b and len(b) >= 2 * dim:
                v = 1.0
                for k in range(dim):
                    v *= abs(float(b[dim + k]) - float(b[k]))
                r = (math.sqrt(v / math.pi) if dim == 2
                     else (3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0))
            else:
                r = float(t.get("radius", rad_default))
            R = min(R, r) if r > 0 else R
    return K, R


def _snap_to_frame(dt_target: float, dt_frame: float) -> float:
    """The largest step <= `dt_target` that divides `dt_frame` EXACTLY.

    Both constraints at once, which is the point. `dt_frame / round(dt_frame/dt_target)` divides
    exactly but can land ABOVE the stability limit -- on material_3d_multimaterial it suggested
    1.231e-04 against a CFL limit of 1.226e-04, i.e. it recommended an unstable step in order to
    fix a clock error. Taking the CEILING of the substep count instead can only make the step
    smaller, so the result is stable by construction and exact by construction.
    """
    if not dt_frame or dt_target <= 0.0:
        return dt_target
    n = max(1, int(math.ceil(dt_frame / dt_target - 1e-9)))
    return dt_frame / n


def Courant_Friedrichs_Lewy_condition(yaml_path: str, write: bool = True):
    """Check (and, if needed, correct in place) the grid-dt CFL of an MPM spec.

    Returns (changed: bool, info: dict|None). A spec with no `mls_mpm_mechanics`
    operator is left untouched (returns (False, None))."""
    # Read the raw YAML TEXT (not just the parsed dict): the correction is a targeted
    # text rewrite so comments/formatting survive; `spec` is the parsed view we compute from.
    text = open(yaml_path).read()
    spec = yaml.safe_load(text)

    # --- locate the MPM micro-timestep. Two spec forms carry it ---------------- #
    # (1) the ORACLE `mls_mpm_mechanics` op: `dt_sub` + an explicit `substeps` on the op line.
    # (2) the DECOMPOSED cycle: a schedule substep block `{substep_dt: X, steps: [mpm_strain,
    #     mpm_scatter, mpm_grid_update, mpm_gather]}` -- here the substep COUNT is implicit
    #     (round(general.dt / substep_dt)), so we only shrink `substep_dt` and the count auto-rises.
    MPM_STEPS = {"mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather",
                 "p2g", "g2p", "agent_scatter", "agent_gather"}        # incl. transitional aliases
    op = next((o for o in spec.get("operators", []) if o.get("op") == "mls_mpm_mechanics"), None)
    if op is not None:
        token = "dt_sub"                                              # which YAML key we rewrite
        micro_dt = float(op.get("dt_sub", 2e-4))                      # the explicit MICRO-timestep
        substeps = int(op.get("substeps", 10))                       # explicit; per-frame time = substeps*dt_sub
        n_grid = int(op.get("n_grid", 128))
        cfl = float(op.get("cfl", 0.4))
    else:
        blk = next((s for s in spec.get("schedule", [])              # the decomposed substep micro-loop
                    if isinstance(s, dict) and "substep_dt" in s
                    and MPM_STEPS & set(s.get("steps", []))), None)
        if blk is None:
            return False, None                                       # neither form -> not an MPM spec, nothing to do
        token = "substep_dt"
        micro_dt = float(blk.get("substep_dt", 2e-4))
        substeps = None                                              # implicit: round(general.dt / substep_dt)
        cfl = float(blk.get("cfl", 0.4))
        # grid resolution lives on the mpm_grid FIELD (`fields: {mpm_grid: {n_grid: 128}}`)
        n_grid = next((int(fc["n_grid"]) for fc in spec.get("fields", {}).values()
                       if isinstance(fc, dict) and "n_grid" in fc), 128)

    # --- the frame clock: substep_dt MUST divide general.dt ---------------------- #
    # THE SUBSTEP COUNT IS ROUNDED AND THE STEP SIZE IS NOT. `engine.run` computes
    # `count = round(general.dt / substep_dt)` and then takes `count` steps of `substep_dt`, so the
    # frame advances `count * substep_dt` -- which equals `general.dt` only when the division is
    # exact. It is silent otherwise, and it does not look like a clock error: measured on cell_02,
    # substep_dt 2.0e-4 against dt 1.5e-3 rounds 7.5 up to 8 and advances 1.6e-3, 6.7% too much,
    # which shows up as a 13.9% deviation in the fall trajectory and reads exactly like the
    # integrator losing accuracy. Neighbouring values that DO divide evenly deviate by 0.1-0.2%.
    _dtf = float((spec.get("general") or {}).get("dt", 0.0)) if substeps is None else None
    if _dtf:
        _n = max(1, round(_dtf / micro_dt))
        _err = abs(_n * micro_dt - _dtf) / _dtf
        # 1e-5, NOT 1e-6, AND SIX DIGITS ON THE WAY OUT. The corrected value is written back as a
        # decimal literal, so it cannot divide `general.dt` to infinite precision; at four
        # significant figures the residual was 1.2e-05 and this branch re-fired on its own output
        # every single run. Six digits bound the round-off at ~5e-07, comfortably inside a 1e-05
        # gate -- which is still 0.001% of a frame, three orders below the 6.7% error that first
        # motivated this check.
        _divide_fix = _snap_to_frame(micro_dt, _dtf) if _err > 1e-5 else None

    # --- the stability limit (same physics either form) ------------------------ #
    dim = int((spec.get("general") or {}).get("dim", 2))
    dx = _cell_size(spec, n_grid, dim)           # grid CELL SIZE, from the WORLD BOX
    cmax = _max_wave_speed(spec)                 # FASTEST elastic P-wave over EVERY set, c = sqrt((la+2mu)/rho)
    # TWO WAVES, TWO LIMITS, AND THE SMALLER ONE IS THE LIMIT. Elastic is the P-wave; capillary is
    # the shortest surface wave the mesh holds, and it exists only where the CSF colour has been
    # normalised (`csf_rho`) -- see `_capillary_limit`. `cap_raw` is Brackbill's bare limit; the
    # SAME `cfl` safety factor is applied to it as to the elastic one, because at the bare limit
    # the drop already piles 60.8 full cells of mass on one node against 1.8 for a clean run.
    cap_raw, sigma, rho_liq = _capillary_limit(spec, dx, dim)
    dt_cap = cfl * cap_raw
    if cmax <= 0.0 and dt_cap == float("inf"):
        return False, None                       # no elastic material AND no tension -> nothing to bound
    # Courant limit: a wave may cross at most `cfl` of a cell per micro-step. A `micro_dt` larger
    # than this makes the explicit MPM substep UNSTABLE (blows up).
    dt_el = cfl * dx / cmax if cmax > 0.0 else float("inf")
    dt_cfl = min(dt_el, dt_cap)                  # THE binding limit
    binds = "capillary" if dt_cap < dt_el else "elastic"
    # by what margin one limit beats the other -- 1.0 means they coincide
    _ratio = (max(dt_el, dt_cap) / min(dt_el, dt_cap)
              if min(dt_el, dt_cap) > 0 and math.isfinite(max(dt_el, dt_cap)) else float("inf"))
    _why = ""
    if math.isfinite(dt_cap):
        _why = (f"; {binds} binds"
                + (f" by {_ratio:.2f}x over " + ("elastic" if binds == "capillary" else "capillary")
                   if math.isfinite(_ratio) else "")
                + f" [elastic {dt_el:.2e} (c_max={cmax:.1f}), capillary {dt_cap:.2e} "
                  f"(sigma={sigma:g}, rho_liq={rho_liq:g})]")
    name = spec.get("general", {}).get("name", yaml_path)

    # ------------------------------------------------------------------------------------------
    # THE OTHER CAPILLARY LIMIT, AND IT HAS NO dt IN IT. Shrinking the substep does not save a
    # drop whose surface tension is stronger than its own liquid. `mpm_strain` gives a liquid
    # mu = 0, so the ONLY thing resisting the CSF's inward pull is the bulk modulus K = lambda,
    # and the pull is the Laplace pressure 2*sigma/R (1*sigma/R in 2D). Their ratio
    #
    #     La = 2*sigma / (R * K)        the fraction of its own volume the tension asks for
    #
    # is a CONSTITUTIVE number, not a stability one: past La ~ 0.4 the drop crushes to a point no
    # matter how the run is timed. MEASURED on material_3d_water_drop (290k particles, n_grid 96,
    # 40 frames, csf_rho 1.0, csf_band 0.2, substep_dt 6.4e-5 = 0.19-0.25 of the RAW Brackbill
    # limit, so the capillary CFL above is nowhere near binding), R = 0.214 (the equivalent radius
    # of the 0.32 x 0.40 x 0.32 block, exactly as `_liquid_scale` computes it), reading mean(det F)
    # over all particles at the end -- 1.0 is the birth volume:
    #
    #     La  0.19 (sigma 1.12, K 55.6) -> mean J 0.937,  peak node mass 1.7 full cells    clean
    #     La  0.27 (sigma 1.60, K 55.6) -> mean J 0.902,  1.9 full cells                   clean
    #     La  0.38 (sigma 2.24, K 55.6) -> mean J 0.757,  13.7 full cells                  degraded
    #     La  0.47 (sigma 2.80, K 55.6) -> mean J 1.0e-6, 12,729 full cells                COLLAPSE
    #     La  0.31 (sigma 3.73, K 111 ) -> mean J 0.810,  5.6 full cells                   marginal
    #     La  0.38 (sigma 4.48, K 111 ) -> mean J 0.732,  46.7 full cells                  degraded
    #     La  0.47 (sigma 5.60, K 111 ) -> mean J 1.0e-6, 13,630 full cells                COLLAPSE
    #
    # THE TWO K FAMILIES AGREE ON La AND DISAGREE ON sigma BY 2x, which is what makes La the number:
    # doubling `youngs` 200 -> 400 turned the sigma-2.8 collapse into a clean run (mean J 0.875)
    # with no change to the timestep, while five substeps from 2.0e-4 down to 6.4e-5 all collapsed
    # alike at sigma 2.8. Particle count is NOT the lever either -- 145k and 580k particles (0.5x
    # and 2x of 290k) both collapsed at sigma 2.8. So this is REPORTED, never corrected: the cures
    # (stiffer liquid, weaker tension, bigger drop, `csf_smooth`, a wider `csf_band`) each change
    # what the run IS. Same discipline as the particles-per-cell report below.
    _La = 0.0
    if sigma > 0.0 and rho_liq > 0.0:
        _K, _R = _liquid_scale(spec, dim)
        if math.isfinite(_K) and math.isfinite(_R) and _K > 0 and _R > 0:
            _La = (dim - 1) * sigma / (_R * _K)      # 2*sigma/(R*K) in 3D, sigma/(R*K) in 2D
            if _La > 0.35:
                warn(f"{name}: surface tension asks for more compression than this liquid can "
                     f"refuse -- Laplace/bulk La = {(dim - 1)}*sigma/(R*K) = {_La:.2f} "
                     f"(sigma={sigma:g}, R={_R:.3g}, K=lambda={_K:.4g}, from youngs at nu={_NU}). "
                     f"Measured on material_3d_water_drop at n_grid 96: La 0.38 costs 24-27% of "
                     f"the drop's birth volume and La 0.47 collapses it to a point (mean det F = "
                     f"1e-6, 13,000 full cells of mass on one node). LOWERING substep_dt DOES NOT "
                     f"HELP -- five steps from 2.0e-4 to 6.4e-5 collapsed alike, and so did 0.5x "
                     f"and 2x the particle count. Raise `youngs` (K scales with it), lower "
                     f"`surface_tension`, or mollify the curvature: `csf_smooth: 4` took that same "
                     f"collapsing run to mean J 0.875, and `csf_band: 0.35` to mean J 0.878.")
            elif _La > 0.15:
                print(f"[grid-CFL] {name}: Laplace/bulk La = {_La:.2f} (sigma={sigma:g}, "
                      f"R={_R:.3g}, K={_K:.4g}) -- the tension asks for {_La * 100:.0f}% of the "
                      f"liquid's volume at first order; measured 6.3% loss of mean det F at "
                      f"La 0.19, 9.8% at La 0.27, 24% at La 0.38, total collapse at La 0.47.",
                      flush=True)

    # Already stable? Leave it. The 1% slack makes this IDEMPOTENT: it absorbs the 3-sig-fig
    # rounding of a value WE wrote last run, so a corrected spec is not re-bumped every generate.
    # THE CLOCK FIX IS APPLIED, NOT ANNOUNCED. This used to `warn(...)` and leave the spec alone,
    # so every run re-printed a complaint about a value this function had itself written on the
    # previous run -- and the frame advanced by the wrong amount the whole time. Snapping to a step
    # that divides `general.dt` can only SHRINK it, so it cannot break the stability bound below.
    if _dtf and _divide_fix is not None:
        _n_new = max(1, round(_dtf / _divide_fix))
        print(f"[grid-CFL] {name if 'name' in dir() else spec.get('general', {}).get('name', yaml_path)}: "
              f"{token}={micro_dt:.3e} did not divide general.dt={_dtf:.3e} "
              f"({_err * 100:.1f}% of a frame lost per frame); corrected -> {_divide_fix:.4e} "
              f"for exactly {_n_new} substeps.", flush=True)
        micro_dt = _divide_fix
        if write:
            text = re.sub(rf"(\b{token}:\s*)[0-9.eE+\-]+",
                          lambda m: f"{m.group(1)}{_divide_fix:.6e}", text, count=1)
            open(yaml_path, "w").write(text)

    if micro_dt <= dt_cfl * 1.01:
        # STABLE -- and now it SAYS BY HOW MUCH. This branch was silent, so a spec running four
        # times more substeps than stability requires looked exactly like one running at the
        # limit. Substeps are a straight linear multiplier on frame time, so unreported headroom
        # is unclaimed speed: cell_02 and cell_03 were at 3.7x. It is only reported, never acted
        # on -- CFL bounds STABILITY, not accuracy, and shortening a run is the caller's call.
        head = dt_cfl / micro_dt
        # A CAPILLARY-BOUND SPEC ALWAYS REPORTS, at any headroom. The 1.5x gate above exists to
        # keep a routine elastic pass quiet, and it would have hidden the whole new limit: on
        # material_3d_water_st560 the step sits 1.13x under the capillary limit, i.e. inside the
        # noise of "silent", while the same spec at 5x the tension is 1.97x OVER it. The margin is
        # the number the reader needs and 1.13 is exactly the value worth printing.
        if head >= 1.5 or binds == "capillary":
            # substeps: explicit on the oracle form, implicit as round(general.dt / substep_dt)
            dt_frame = (substeps * micro_dt if substeps is not None
                        else float((spec.get("general") or {}).get("dt", 0.0)))
            now = round(dt_frame / micro_dt) if dt_frame else None
            could = max(1, round(dt_frame / dt_cfl)) if dt_frame else None
            extra = (f"; {now} substeps/frame where {could} would be stable ({now / could:.1f}x)"
                     if now and could and head >= 1.5 else "")
            print(f"[grid-CFL] {name}: {token}={micro_dt:.2e} is {head:.2f}x BELOW the stability "
                  f"limit {dt_cfl:.2e} (c_max={cmax:.1f}){extra}{_why}.", flush=True)
        return False, {"name": name, "cmax": cmax, "micro_dt": micro_dt, "dt_cfl": dt_cfl,
                       "ok": True, "headroom": head, "La": _La, "binds": binds, "dt_elastic": dt_el,
                       "dt_capillary": dt_cap, "sigma": sigma, "csf_rho": rho_liq}

    # --- too big: shrink the micro-timestep to the limit, KEEPING per-frame time ---------- #
    # oracle: per-frame time = substeps*dt_sub, so raise substeps as dt_sub falls (new_sub).
    # decomposed: per-frame time = general.dt (fixed) and substeps = round(general.dt/substep_dt)
    #             is implicit, so shrinking substep_dt raises the count automatically -> no rewrite.
    # DECOMPOSED FORM: snap to a step that also divides general.dt, so this correction does not
    # itself create the clock error the block above then has to repair on the next run.
    dt_new = dt_cfl if substeps is not None else _snap_to_frame(dt_cfl, _dtf)
    new_sub = int(math.ceil(substeps * micro_dt / dt_cfl)) if substeps is not None else None
    info = {"name": name, "cmax": cmax, "token": token, "dt_old": micro_dt, "dt_new": dt_new,
            "sub_old": substeps, "sub_new": new_sub, "La": _La, "binds": binds, "dt_elastic": dt_el,
            "dt_capillary": dt_cap, "sigma": sigma, "csf_rho": rho_liq,
            "over_by": micro_dt / dt_cfl}
    sub_note = (f", substeps {substeps}->{new_sub}" if new_sub is not None
                else f" ({round(_dtf / dt_new)} substeps, dividing general.dt exactly)" if _dtf
                else " (substeps auto = round(dt/substep_dt))")
    print(f"[grid-CFL] {name}: {token}={micro_dt:.2e} > limit {dt_cfl:.2e} "
          f"({micro_dt / dt_cfl:.2f}x OVER) (c_max={cmax:.1f}, dx={dx:.2e}, cfl={cfl})"
          f"{_why}; correcting spec -> "
          f"{token}={dt_new:.4e}{sub_note} (per-frame time preserved).", flush=True)
    if write:
        # Rewrite ONLY the relevant token(s) in the raw text (regex, count=1) so comments/layout survive.
        text = re.sub(rf"(\b{token}:\s*)[0-9.eE+\-]+", lambda m: f"{m.group(1)}{dt_new:.6e}", text, count=1)
        if new_sub is not None:                  # oracle form only: also bump the explicit substeps
            text = re.sub(r"(\bsubsteps:\s*)\d+", lambda m: f"{m.group(1)}{new_sub}", text, count=1)
        open(yaml_path, "w").write(text)
    return True, info


# ==========================================================================================================
# PARTICLES PER CELL -- the other half of an MPM discretisation, and the one with no error message.
#
# `substep_dt` has the CFL check above; the grid resolution had nothing, so `n_grid` could be raised
# without touching the particle count and the run would go quietly wrong. It cost a whole batch:
# `material_3d_multimaterial` went from n_grid 64 to 192 at a fixed 100k particles per body, which
# divides particles-per-cell by (192/64)^3 = 27 -- 43.3 to 1.61 -- and its snow block, which had
# held its shape as a slightly compacted cube, collapsed into a flat pancake. That reads as a
# material-parameter problem and is a sampling problem.
#
# WHY 8. MPM carries the material on particles and solves on the grid, so a cell needs enough
# particle samples to determine the local deformation: the convention is 2 per axis, hence 2^3 = 8
# in 3D and 2^2 = 4 in 2D. Below about half of that the deformation gradient in a cell is
# underdetermined, the grid velocity there is essentially one particle's own velocity, and the body
# loses stiffness and fractures numerically.
#
# REPORTED, NEVER ACTED ON. Both cures -- coarsen the grid or add particles -- change what the run
# IS, so neither is this function's decision to make. Same discipline as the CFL headroom message.
# ==========================================================================================================
PPC_TARGET = 8.0                 # 2 particles per axis in 3D (4.0 in 2D; set below from `dim`)
PPC_FLOOR = 0.5                  # fraction of target below which the material is under-sampled


def _body_volumes(spec, dim):
    """(name, volume) for every declared body type, from its `block` extent or its radius."""
    out = []
    sets = spec.get("sets") or {}
    for sname, sv in sets.items():
        if not isinstance(sv, dict) or "per_parent" not in sv:
            continue
        parent = sets.get(sv.get("parent"), {}) or {}
        types = parent.get("types") or {}
        rad_default = float(sv.get("radius", parent.get("radius", 0.05)) or 0.05)
        n_par = int(parent.get("n", 1))
        if not types:
            v = (math.pi * rad_default ** 2 if dim == 2
                 else (4.0 / 3.0) * math.pi * rad_default ** 3)
            out.append((sname, v, int(sv["per_parent"]), n_par))
            continue
        # THE ENGINE'S EXACT ALLOCATION, replicated rather than approximated: engine.py:610 gives
        # the LAST type the remainder (`total - start`) so per-type rounding never leaves cells
        # unassigned. Applying round() to every type -- including the last -- reported `snow`
        # absent when it in fact receives every cell the others rounded away.
        _names = list(types)
        _alloc, _start = {}, 0
        for _i, _tn in enumerate(_names):
            _k = ((n_par - _start) if _i == len(_names) - 1
                  else int(round(float(types[_tn].get("fraction", 1.0 / len(_names))) * n_par)))
            _alloc[_tn] = max(_k, 0); _start += _k
        for tn, t in types.items():
            b = t.get("block")
            if b:
                v = 1.0
                for k in range(dim):
                    v *= abs(float(b[dim + k]) - float(b[k]))
            else:
                r = float(t.get("radius", rad_default))
                v = math.pi * r ** 2 if dim == 2 else (4.0 / 3.0) * math.pi * r ** 3
            # NOT scaled by `fraction`. `fraction` selects how many PARENT CELLS take this type
            # (engine.py:610, k = round(fraction * n_cells)); every cell that does gets the FULL
            # `per_parent` particles, and p_vol = vol/per_parent, so each body is fully dense
            # whatever the split. Multiplying here reported 3.85 p/cell for a body that actually
            # had 38.5, and warned about specs that were fine.
            #
            # A TYPE THAT ROUNDS TO ZERO CELLS IS NOT A SAMPLING PROBLEM, IT IS AN ABSENT BODY, and
            # it gets its own warning below because it is much easier to misread: the material
            # simply is not in the scene.
            frac = float(t.get("fraction", 1.0 / max(len(types), 1)))
            if _alloc.get(tn, 0) == 0 and n_par > 0:
                warn(f"{(spec.get('general') or {}).get('name', '?')}: type {tn!r} has "
                     f"fraction {frac} of {n_par} parent cells -> round({frac * n_par:.2f}) = ZERO "
                     f"cells, so this material is ABSENT from the run. Fractions quantise to "
                     f"1/{n_par} here; use a multiple of {1.0 / n_par:.3f} or raise the parent "
                     f"count. (The LAST type absorbs whatever the others round away, so it gets "
                     f"more cells than its fraction asks for.)")
                continue
            out.append((f"{sname}/{tn} x{_alloc[tn]}", v, int(sv["per_parent"]), n_par))
    return out


def particles_per_cell(yaml_path: str) -> list:
    """Warn when a spec's grid and particle count disagree about how finely it is sampled."""
    try:
        spec = yaml.safe_load(open(yaml_path))
    except Exception:
        return []
    if not isinstance(spec, dict):
        return []
    dim = int((spec.get("general") or {}).get("dim", 2))
    n_grid = next((int(fc["n_grid"]) for fc in (spec.get("fields") or {}).values()
                   if isinstance(fc, dict) and "n_grid" in fc), None)
    if n_grid is None:
        return []
    target = PPC_TARGET if dim == 3 else 4.0
    name = (spec.get("general") or {}).get("name", os.path.basename(yaml_path))
    rows = []
    _dupes = []
    # DEDUPED. A spec with 27 identical blobs would otherwise emit 27 identical warnings, and a
    # message repeated 27 times is one nobody reads.
    seen = {}
    # CELLS FROM THE CELL SIZE, not from n_grid^dim -- the two agree only on a unit box. On a 0.1 m
    # box a 1e-3 m^3 body holds 884,736 cells and `vol * n_grid**dim` reports 884, so a spec that is
    # correctly sampled would be told it is oversampled by 1000x.
    _dx_ppc = _cell_size(spec, n_grid, dim)
    for label, vol, n_per_parent, _n_par in _body_volumes(spec, dim):
        cells = vol / (_dx_ppc ** dim)
        if cells <= 0:
            continue
        ppc = n_per_parent / cells
        rows.append((label, ppc, cells, n_per_parent))
        key = (round(ppc, 3), round(cells))
        if key in seen:
            seen[key][1] += 1
            continue
        seen[key] = [label, 1]
        if ppc < target * PPC_FLOOR:
            # `n_grid` for the target, and the particle count for the target, so the reader can pick
            want_grid = int(round((n_per_parent / (target * vol)) ** (1.0 / dim)))
            want_n = int(target * cells)
            warn(f"{name}: {label} has {ppc:.2f} particles per grid cell "
                 f"({n_per_parent:,.0f} particles over {cells:,.0f} cells at n_grid={n_grid}) -- "
                 f"UNDER-SAMPLED, MPM wants ~{target:.0f}. The body will lose stiffness and may "
                 f"fracture numerically. Either n_grid={want_grid} or "
                 f"{want_n:,} particles for that body.")
            _dupes.append(key)
        elif ppc > target * 16:
            print(f"[grid-ppc] {name}: {label} has {ppc:.0f} particles per cell "
                  f"(~{target:.0f} is the convention) -- the grid cannot resolve detail the "
                  f"particles carry, so this is cost without fidelity; n_grid="
                  f"{int(round((n_per_parent / (target * vol)) ** (1.0 / dim)))} would use them.",
                  flush=True)
    for k in _dupes:                      # "... and 26 more just like it", once
        if seen.get(k, [None, 1])[1] > 1:
            print(f"[grid-ppc] {name}: ... and {seen[k][1] - 1} further bodies identical to "
                  f"{seen[k][0]}.", flush=True)
    return rows
