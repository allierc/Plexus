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
    rho_for = {}
    for sname, st in sets.items():
        if not isinstance(st, dict):
            continue
        rho_for.setdefault(sname, float(st.get("density", rho_default)))
        par = st.get("parent")
        if par in sets and "density" in st:
            # the child's particles carry the child's density, and their material comes from the
            # parent's type table; attribute the lower of the two to the parent so it is not missed
            rho_for[par] = min(rho_for.get(par, float("inf")), float(st["density"]))
    cmax = 0.0
    for sname, st in sets.items():
        if not isinstance(st, dict):
            continue
        rho_set = rho_for.get(sname, float(st.get("density", rho_default)))
        for t in (st.get("types") or {}).values():
            if not isinstance(t, dict):
                continue
            rho = float(t.get("density", rho_set))          # per-type density wins (light/heavy cytosol)
            layers = list(t.get("layers") or [{"youngs": t.get("youngs", 100.0),
                                               "material": t.get("material", "elastic")}])
            if t.get("core"):
                layers.append(t["core"])
            for L in layers:
                E = float(L.get("youngs", t.get("youngs", 100.0)))
                mu, la = _lame(E)
                if L.get("material", t.get("material", "elastic")) == "liquid":
                    mu = 0.0                 # liquid carries no shear modulus
                cmax = max(cmax, math.sqrt(max(la + 2 * mu, 0.0) / max(rho, 1e-12)))
    return cmax


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
    dx = 1.0 / n_grid                            # grid CELL SIZE (world is unit-width)
    cmax = _max_wave_speed(spec)                 # FASTEST elastic P-wave over EVERY set, c = sqrt((la+2mu)/rho)
    if cmax <= 0.0:
        return False, None                       # no elastic material (pure liquid, mu=0) -> no wave, nothing to bound
    # Courant limit: a wave may cross at most `cfl` of a cell per micro-step. A `micro_dt` larger
    # than this makes the explicit MPM substep UNSTABLE (blows up).
    dt_cfl = cfl * dx / cmax
    name = spec.get("general", {}).get("name", yaml_path)
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
        if head >= 1.5:
            # substeps: explicit on the oracle form, implicit as round(general.dt / substep_dt)
            dt_frame = (substeps * micro_dt if substeps is not None
                        else float((spec.get("general") or {}).get("dt", 0.0)))
            now = round(dt_frame / micro_dt) if dt_frame else None
            could = max(1, round(dt_frame / dt_cfl)) if dt_frame else None
            extra = (f"; {now} substeps/frame where {could} would be stable ({now / could:.1f}x)"
                     if now and could else "")
            print(f"[grid-CFL] {name}: {token}={micro_dt:.2e} is {head:.1f}x BELOW the stability "
                  f"limit {dt_cfl:.2e} (c_max={cmax:.1f}){extra}.", flush=True)
        return False, {"name": name, "cmax": cmax, "micro_dt": micro_dt, "dt_cfl": dt_cfl,
                       "ok": True, "headroom": head}

    # --- too big: shrink the micro-timestep to the limit, KEEPING per-frame time ---------- #
    # oracle: per-frame time = substeps*dt_sub, so raise substeps as dt_sub falls (new_sub).
    # decomposed: per-frame time = general.dt (fixed) and substeps = round(general.dt/substep_dt)
    #             is implicit, so shrinking substep_dt raises the count automatically -> no rewrite.
    # DECOMPOSED FORM: snap to a step that also divides general.dt, so this correction does not
    # itself create the clock error the block above then has to repair on the next run.
    dt_new = dt_cfl if substeps is not None else _snap_to_frame(dt_cfl, _dtf)
    new_sub = int(math.ceil(substeps * micro_dt / dt_cfl)) if substeps is not None else None
    info = {"name": name, "cmax": cmax, "token": token, "dt_old": micro_dt, "dt_new": dt_new,
            "sub_old": substeps, "sub_new": new_sub}
    sub_note = (f", substeps {substeps}->{new_sub}" if new_sub is not None
                else f" ({round(_dtf / dt_new)} substeps, dividing general.dt exactly)" if _dtf
                else " (substeps auto = round(dt/substep_dt))")
    print(f"[grid-CFL] {name}: {token}={micro_dt:.2e} > limit {dt_cfl:.2e} "
          f"(c_max={cmax:.1f}, dx={dx:.2e}, cfl={cfl}); correcting spec -> "
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
    for label, vol, n_per_parent, _n_par in _body_volumes(spec, dim):
        cells = vol * n_grid ** dim
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
