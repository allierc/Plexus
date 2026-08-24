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
    cmax = 0.0
    for sname, st in (spec.get("sets") or {}).items():
        if not isinstance(st, dict):
            continue
        rho_set = float(st.get("density", rho_default))
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
        if _err > 1e-6:
            warn(f"{spec.get('general', {}).get('name', yaml_path)}: {token}={micro_dt:.3e} does "
                 f"not divide general.dt={_dtf:.3e} -- {_n} substeps advance "
                 f"{_n * micro_dt:.4e}, i.e. {_err * 100:.1f}% {'more' if _n * micro_dt > _dtf else 'less'} "
                 f"simulated time per frame than the spec declares. Use "
                 f"{token}={_dtf / _n:.3e} for exactly {_n} substeps.")

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
    new_sub = int(math.ceil(substeps * micro_dt / dt_cfl)) if substeps is not None else None
    info = {"name": name, "cmax": cmax, "token": token, "dt_old": micro_dt, "dt_new": dt_cfl,
            "sub_old": substeps, "sub_new": new_sub}
    sub_note = f", substeps {substeps}->{new_sub}" if new_sub is not None else " (substeps auto = round(dt/substep_dt))"
    print(f"[grid-CFL] {name}: {token}={micro_dt:.2e} > limit {dt_cfl:.2e} "
          f"(c_max={cmax:.1f}, dx={dx:.2e}, cfl={cfl}); correcting spec -> "
          f"{token}={dt_cfl:.3e}{sub_note} (per-frame time preserved).", flush=True)
    if write:
        # Rewrite ONLY the relevant token(s) in the raw text (regex, count=1) so comments/layout survive.
        text = re.sub(rf"(\b{token}:\s*)[0-9.eE+\-]+", lambda m: f"{m.group(1)}{dt_cfl:.3e}", text, count=1)
        if new_sub is not None:                  # oracle form only: also bump the explicit substeps
            text = re.sub(r"(\bsubsteps:\s*)\d+", lambda m: f"{m.group(1)}{new_sub}", text, count=1)
        open(yaml_path, "w").write(text)
    return True, info
