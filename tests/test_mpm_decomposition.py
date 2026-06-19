"""Phase-3 regression: the decomposed MPM (mpm_strain -> p2g -> mpm_grid_update -> g2p
under a substep loop) must reproduce the fenced oracle `mls_mpm_mechanics` frame-by-frame.
The oracle is KEPT as the reference target. CPU for determinism (index_add_).

Run:  PYTHONPATH=src python tests/test_mpm_decomposition.py
"""
import os, tempfile
import numpy as np
import plexus.operators  # noqa: F401  (registers operators + fields)
from plexus import schema
from plexus.engine import run

_CELL = ("cell: {{n: {n}, start: {start}, types: {{{types}}}}}\n"
         "  mpm_particle: {{parent: cell, per_parent: {ppc}, radius: {rad}, density: 1.0}}")

def _mono(case):
    return f"""
general: {{name: {case['name']}_mono, seed: 0, n_frames: {case['frames']}, dt: 1.0, boundary: wall{case['obs']}}}
sets:
  {_CELL.format(**case)}
fields: {{}}
operators:
  - {{op: aggregate, at: cell}}
  - {{op: gravity, at: cell, g: {case['g']}}}
  - {{op: mls_mpm_mechanics, at: mpm_particle, n_grid: 64, substeps: {case['sub']}, dt_sub: 2.0e-4, a_max: 200, drag: {case['drag']}, wall_damp: {case['wd']}, wall_contact: 0.05, surface_tension: {case['st']}}}
schedule: [aggregate, gravity, mls_mpm_mechanics]
"""

def _dec(case):
    return f"""
general: {{name: {case['name']}_dec, seed: 0, n_frames: {case['frames']}, dt: 1.0, boundary: wall{case['obs']}}}
sets:
  {_CELL.format(**case)}
fields:
  mpm_grid: {{frame: mpm_grid, n_grid: 64}}
operators:
  - {{op: aggregate, at: cell}}
  - {{op: gravity, at: cell, g: {case['g']}}}
  - {{op: mpm_strain, at: mpm_particle, dt_sub: 2.0e-4}}
  - {{op: p2g, at: mpm_particle, to: mpm_grid, dt_sub: 2.0e-4, drag: {case['drag']}, a_max: 200}}
  - {{op: mpm_grid_update, at: mpm_grid, dt_sub: 2.0e-4, surface_tension: {case['st']}, wall_damp: {case['wd']}, wall_contact: 0.05}}
  - {{op: g2p, at: mpm_particle, from: mpm_grid, dt_sub: 2.0e-4, wall_damp: {case['wd']}, wall_contact: 0.05, vmax: 1.0e9}}
schedule:
  - aggregate
  - gravity
  - {{substep: {case['sub']}, dt: 2.0e-4, steps: [mpm_strain, p2g, mpm_grid_update, g2p]}}
"""

def _run(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False); f.write(text); f.close()
    sp = schema.load(f.name); os.unlink(f.name)
    _, out = run(sp, device="cpu")
    return out["sets"]["mpm_particle"]["pos"]

LIQ = "water: {fraction: 1.0, youngs: 300, block: [0.1,0.05,0.55,0.6], layers: [{frac: 1.0, youngs: 300, material: liquid}]}"
SNOW = "snow: {fraction: 1.0, youngs: 250, layers: [{frac: 1.0, youngs: 250, material: snow}]}"
ELAS = "ball: {fraction: 1.0, youngs: 500, layers: [{frac: 1.0, youngs: 500, material: elastic}]}"

CASES = [
  dict(name="liq_grav",  types=LIQ,  start="[0.3,0.5,0.3,0.5]", n=1, ppc=1500, rad=0.2, frames=20, g=12, drag=0.1, wd=0.5, st=0,  obs="", sub=14),
  dict(name="csf",       types=LIQ,  start="[0.3,0.5,0.3,0.5]", n=1, ppc=1500, rad=0.2, frames=20, g=0,  drag=0.1, wd=1.0, st=30, obs="", sub=14),
  dict(name="snow",      types=SNOW, start="[0.5,0.7,0.5,0.7]",  n=1, ppc=1200, rad=0.16, frames=25, g=12, drag=0.2, wd=0.5, st=0, obs="", sub=14),
  dict(name="elastic",   types=ELAS, start="[0.5,0.7,0.5,0.7]",  n=1, ppc=1200, rad=0.12, frames=25, g=12, drag=0.2, wd=0.6, st=0, obs="", sub=14),
  dict(name="obstacle",  types=LIQ,  start="[0.3,0.6,0.3,0.6]",  n=1, ppc=1500, rad=0.2, frames=25, g=12, drag=0.2, wd=0.5, st=0,
       obs=", obstacles: [[0.45,0.0,0.55,0.4]]", sub=14),
]

print(f"{'case':12s} {'frames':>6s} {'final max|Δ|':>13s} {'worst max|Δ|':>13s}  verdict")
ok = True
for c in CASES:
    a = _run(_mono(c)); b = _run(_dec(c))
    dmax = np.abs(a - b).max(axis=(1, 2))
    good = dmax.max() < 1e-4
    ok = ok and good
    print(f"{c['name']:12s} {c['frames']:6d} {dmax[-1]:13.2e} {dmax.max():13.2e}  {'MATCH' if good else 'MISMATCH'}")
print("\nALL MATCH -- decomposition reproduces the oracle" if ok else "\nFAIL: decomposition diverges from oracle")
assert ok
