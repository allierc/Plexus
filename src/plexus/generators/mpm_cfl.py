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

_NU = 0.2                                    # Poisson ratio (matches the mpm_particle entity)


def _lame(E, nu=_NU):
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, la


def _max_wave_speed(spec, rho) -> float:
    """Largest elastic P-wave speed over all cell-type materials (mu = 0 for liquid)."""
    cmax = 0.0
    for t in spec["sets"]["cell"].get("types", {}).values():
        layers = list(t.get("layers") or [{"youngs": t.get("youngs", 100.0), "material": "elastic"}])
        if t.get("core"):
            layers.append(t["core"])
        for L in layers:
            E = float(L.get("youngs", t.get("youngs", 100.0)))
            mu, la = _lame(E)
            if L.get("material", "elastic") == "liquid":
                mu = 0.0                     # liquid carries no shear modulus
            c = math.sqrt(max(la + 2 * mu, 0.0) / max(rho, 1e-12))
            cmax = max(cmax, c)
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

    # --- the stability limit (same physics either form) ------------------------ #
    dx = 1.0 / n_grid                            # grid CELL SIZE (world is unit-width)
    rho = float(spec["sets"].get("mpm_particle", {}).get("density", 1.0))   # material density
    cmax = _max_wave_speed(spec, rho)            # FASTEST elastic P-wave over all materials, c = sqrt((la+2mu)/rho)
    if cmax <= 0.0:
        return False, None                       # no elastic material (pure liquid, mu=0) -> no wave, nothing to bound
    # Courant limit: a wave may cross at most `cfl` of a cell per micro-step. A `micro_dt` larger
    # than this makes the explicit MPM substep UNSTABLE (blows up).
    dt_cfl = cfl * dx / cmax
    name = spec.get("general", {}).get("name", yaml_path)
    # Already stable? Leave it. The 1% slack makes this IDEMPOTENT: it absorbs the 3-sig-fig
    # rounding of a value WE wrote last run, so a corrected spec is not re-bumped every generate.
    if micro_dt <= dt_cfl * 1.01:
        return False, {"name": name, "cmax": cmax, "micro_dt": micro_dt, "dt_cfl": dt_cfl, "ok": True}

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
