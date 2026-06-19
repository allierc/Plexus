"""Grid-dt CFL auto-correction for MPM specs -- applied to the YAML, not in code.

The MLS-MPM substep is explicit, so it must obey the Courant (grid-dt) condition: an
elastic wave may not cross more than ~`cfl` of a background-grid cell per substep,

    dt_sub <= cfl * dx / c ,   c = sqrt((lambda + 2*mu) / rho)   (elastic P-wave speed),

with dx = 1/n_grid. c is set by the STIFFEST material (largest `youngs`); liquids
(mu = 0) are slow and usually already safe. When a spec's `dt_sub` exceeds the limit
this WARNS and -- per the 'the correction lives in the spec, not in the engine' rule --
rewrites the spec.yaml in place: it lowers `dt_sub` to the stable value and raises
`substeps` so the per-frame simulated time (substeps*dt_sub) is preserved (same
physics, now stable). Idempotent: a corrected spec is left unchanged on re-run.

It edits only the `dt_sub:`/`substeps:` tokens (which appear solely on the mpm
operator line), so comments and the rest of the spec text are preserved.
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


def enforce_grid_cfl(yaml_path: str, write: bool = True):
    """Check (and, if needed, correct in place) the grid-dt CFL of an MPM spec.

    Returns (changed: bool, info: dict|None). A spec with no `mls_mpm_mechanics`
    operator is left untouched (returns (False, None))."""
    text = open(yaml_path).read()
    spec = yaml.safe_load(text)
    op = next((o for o in spec.get("operators", []) if o.get("op") == "mls_mpm_mechanics"), None)
    if op is None:
        return False, None
    dt_sub = float(op.get("dt_sub", 2e-4)); substeps = int(op.get("substeps", 10))
    n_grid = int(op.get("n_grid", 128)); cfl = float(op.get("cfl", 0.4))
    dx = 1.0 / n_grid
    rho = float(spec["sets"].get("mpm_particle", {}).get("density", 1.0))
    cmax = _max_wave_speed(spec, rho)
    if cmax <= 0.0:
        return False, None
    dt_cfl = cfl * dx / cmax
    name = spec.get("general", {}).get("name", yaml_path)
    # 1% tolerance: absorbs the 3-sig-fig rounding of a previously-written dt_sub so the
    # correction is IDEMPOTENT (a corrected spec is not re-bumped on the next generate).
    if dt_sub <= dt_cfl * 1.01:
        return False, {"name": name, "cmax": cmax, "dt_sub": dt_sub, "dt_cfl": dt_cfl, "ok": True}
    new_sub = int(math.ceil(substeps * dt_sub / dt_cfl))
    info = {"name": name, "cmax": cmax, "dt_old": dt_sub, "dt_new": dt_cfl,
            "sub_old": substeps, "sub_new": new_sub}
    print(f"[grid-CFL] {name}: dt_sub={dt_sub:.2e} > limit {dt_cfl:.2e} "
          f"(c_max={cmax:.1f}, dx={dx:.2e}, cfl={cfl}); correcting spec -> "
          f"dt_sub={dt_cfl:.3e}, substeps {substeps}->{new_sub} (per-frame time preserved).", flush=True)
    if write:
        text = re.sub(r"(\bdt_sub:\s*)[0-9.eE+\-]+", lambda m: f"{m.group(1)}{dt_cfl:.3e}", text, count=1)
        text = re.sub(r"(\bsubsteps:\s*)\d+", lambda m: f"{m.group(1)}{new_sub}", text, count=1)
        open(yaml_path, "w").write(text)
    return True, info
