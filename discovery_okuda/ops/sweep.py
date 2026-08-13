#!/usr/bin/env python
"""sweep -- twenty ECM runs, each asking one question, all logged to log/okuda_ECM/NN_*.

    python sweep.py --device cuda:0

FIVE QUESTIONS, FOUR RUNS EACH, ONE VARIABLE AT A TIME. Every run shares a baseline and differs
from it in exactly one number, so a difference between two movies is attributable. That is the
same one-edit rule the okuda loop enforces on its own batches, and it exists for the same reason:
a sweep that moves two things at once produces twenty results and no comparisons.

  01-04  STIFFNESS      how stiff can the matrix be before it resists the ball -- and before the
                        explicit substep stops being stable. Both ends are findings: a matrix
                        that never yields tells us the ball cannot invade, and a matrix that
                        explodes tells us the substep, not the biology, set the limit.
  05-08  CAVITY SHAPE   a sphere confines equally and the ball stays a ball. A thin disc pinches
                        one axis and frees two. This is the axis the whole experiment turns on,
                        so it gets a run at each of four thicknesses from thin to spherical.
  09-12  FIBRE ALIGNMENT  isotropic to strongly aligned. HONEST CAVEAT: the fibres are geometry,
                        not yet mechanics -- MPM interpolates to a continuum grid, so aligned
                        fibres of an isotropic material still respond isotropically. Any
                        difference across these four is the SEEDING (where the material happens
                        to be dense), not anisotropy. Run because the null result is worth
                        having on record before an anisotropic term is written.
  13-16  GROWTH RATE    how fast the ball demands room. Slow growth lets the matrix relax between
                        increments; fast growth is a shock. Same final radius, different arrival.
  17-20  CONTACT + FIBRE GEOMETRY  the penalty stiffness, and fibre length/count at fixed particle
                        number -- i.e. many short fibres against few long ones.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, "..", "..", "src"))):
    if p not in sys.path:
        sys.path.insert(0, p)

BASE = dict(n_frames=260, n_particles=48000, n_grid=48, youngs=40.0,
            cavity_r=0.22, cavity_h=0.07, align=0.0, growth=0.0016,
            k_contact=900.0, n_fibres=900, fibre_len=0.16, substep_dt=2.0e-4)

RUNS = [
    # ---- stiffness ------------------------------------------------------------------
    ("01_soft_E10",        dict(youngs=10.0)),
    ("02_baseline_E40",    dict()),
    ("03_stiff_E120",      dict(youngs=120.0)),
    ("04_stiff_E300",      dict(youngs=300.0)),
    # ---- cavity shape ---------------------------------------------------------------
    ("05_disc_thin_h004",  dict(cavity_h=0.04)),
    ("06_disc_h007",       dict(cavity_h=0.07)),
    ("07_disc_thick_h012", dict(cavity_h=0.12)),
    ("08_sphere_h022",     dict(cavity_h=0.22)),          # h == r: a spherical cavity
    # ---- fibre alignment ------------------------------------------------------------
    ("09_align_000",       dict(align=0.0)),
    ("10_align_030",       dict(align=0.3)),
    ("11_align_060",       dict(align=0.6)),
    ("12_align_090",       dict(align=0.9)),
    # ---- growth rate ----------------------------------------------------------------
    ("13_grow_slow",       dict(growth=0.0006, n_frames=420)),
    ("14_grow_med",        dict(growth=0.0016)),
    ("15_grow_fast",       dict(growth=0.0032, n_frames=180)),
    ("16_grow_veryfast",   dict(growth=0.0060, n_frames=140)),
    # ---- contact stiffness + fibre geometry -----------------------------------------
    ("17_contact_k300",    dict(k_contact=300.0)),
    ("18_contact_k2500",   dict(k_contact=2500.0)),
    ("19_fibres_short",    dict(n_fibres=2400, fibre_len=0.06)),
    ("20_fibres_long",     dict(n_fibres=300,  fibre_len=0.34)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default=None, help="substring filter over run names")
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    import ecm_spec as ES
    import run_ecm as R

    rows, t0 = [], time.time()
    todo = [(n, o) for n, o in RUNS if (a.only is None or a.only in n)]
    print(f"[sweep] {len(todo)} run(s), device={a.device}", flush=True)
    for i, (name, over) in enumerate(todo, 1):
        cfg = dict(BASE); cfg.update(over)
        print(f"\n[sweep {i}/{len(todo)}] {name}  " +
              "  ".join(f"{k}={v}" for k, v in over.items() or [("baseline", "")]), flush=True)
        try:
            spec = ES.build_spec(name, **cfg)
            m = R.run(name, spec, device=a.device, movie=not a.no_movie)
            m["varied"] = over
            rows.append(m)
        except Exception as e:
            traceback.print_exc()
            print(f"[sweep] {name} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
            rows.append({"name": name, "varied": over, "error": f"{type(e).__name__}: {e}"})
        json.dump(rows, open(os.path.join(R.LOG, "sweep.json"), "w"), indent=1)

    print(f"\n[sweep] {len(rows)} run(s) in {(time.time()-t0)/60:.0f} min -> {R.LOG}\n")
    hdr = f"{'run':22}{'contact':>8}{'strained':>10}{'max_disp':>10}{'exploded':>10}{'wall_s':>8}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['name'][:21]:22}{'ERROR':>8}  {r['error'][:44]}")
            continue
        print(f"{r['name'][:21]:22}{str(r.get('contact_frame')):>8}"
              f"{(r.get('strained_frac_end') or 0):>10.3f}{(r.get('max_disp') or 0):>10.3f}"
              f"{str(r.get('exploded')):>10}{(r.get('wall_s') or 0):>8.0f}")


if __name__ == "__main__":
    main()
