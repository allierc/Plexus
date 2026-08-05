#!/usr/bin/env python
"""epi2_sweep -- ten ECM runs against a tissue that never stops dividing, in a tighter, denser,
more compliant matrix.

    python epi2_sweep.py --device cuda:0 --runs 29,30,31
    python epi2_sweep.py --device cuda:1 --runs 32,33

WHAT CHANGED FROM BATCH 1 (24-28), and why each one:

  RESERVOIR x4      Batch 1's tissue stopped dividing at ~7 seconds of a 9-second movie: the vertex
                    buffer filled at 6,396 and `divide_3d` refused 1,723 divisions, so the last
                    quarter of every movie showed an epithelium that had stopped proliferating
                    because of an ARRAY. At x4 the run reports ZERO refusals and the cell count is
                    still changing at frame 399: 200 -> 5,968 cells instead of 200 -> 3,170.
                    Nothing mechanical was touched -- a reservoir is an allocation, not a parameter.
  DENSER FIBRES     6,000 fibres instead of 2,600, at 140,000 particles instead of 110,000. The
                    particle count moves WITH the fibre count on purpose: fibres at fixed particle
                    number means fewer particles per strand, and past a point the "fibre" is a line
                    of dots the MPM grid cannot resolve. Density here means more material, not
                    thinner material.
  TIGHT CAVITY      0.095 of the box against a tissue whose apical surface starts at 0.087, so the
                    matrix is already almost touching at frame 0 and contact lands around frame 28
                    of 402 rather than 136. Almost, not quite: a cavity SMALLER than the starting
                    tissue overlaps at frame 0, and then "first contact" is not an event the run can
                    report -- which is how runs 01-23 all came to log `contact_frame: 0`.
  MORE COMPLIANT    E = 15 rather than 40, with the batch spanning 5 -> 150. A softer matrix strains
                    further for the same load, which is what makes the deformation and the fibre
                    splaying visible rather than inferable.
  stress_scale      0.08 rather than 0.05. A tighter cavity and a bigger tissue mean far more strain,
                    and at 0.05 the top band saturated. FIXED ACROSS THIS BATCH, so the ten runs are
                    comparable to each other -- but NOT colour-comparable with 24-28, which used
                    0.05. Two runs at different palette scales cannot be put side by side, and
                    saying so is cheaper than someone discovering it from the pictures.

ONE EDIT PER RUN from `29_epi2_tight_dense_E15`, so any difference between two movies is
attributable. 33/34 move two numbers because they are one question (fibre count at fixed particle
count -- many thin strands against few thick ones), which is stated rather than hidden.

HONEST CAVEAT, unchanged and load-bearing: the coupling is ONE-WAY. The epithelium pushes the
matrix; the matrix does not push back. So a tight cavity does NOT flatten or slow the tissue -- the
tissue grows the same way in all ten runs, and everything that differs is the matrix's response.
Two-way needs both solvers in one world, which needs the vertex model recalibrated to the unit box;
`combine.py` records why that is a project and not a flag.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "prototype", "Tyssue"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

BUFFER_X = 4

BASE = dict(n_particles=140000, n_grid=48, youngs=15.0, k_contact=900.0,
            cavity_r=0.095, cavity_h=0.095, axis=2, n_fibres=6000, fibre_len=0.16,
            align=0.0, substep_dt=2.0e-4, stress_scale=0.08)

RUNS = [
    ("29_epi2_tight_dense_E15", dict()),
    # ---- how compliant the matrix is: the whole point of "more elastic" -------------------
    ("30_epi2_soft_E5",         dict(youngs=5.0)),
    ("31_epi2_E40",             dict(youngs=40.0)),
    ("32_epi2_stiff_E150",      dict(youngs=150.0)),
    # ---- how dense the fibre network is --------------------------------------------------
    ("33_epi2_fibres_12k",      dict(n_fibres=12000)),
    ("34_epi2_fibres_3k",       dict(n_fibres=3000)),
    # ---- how tightly it starts against the cells -----------------------------------------
    ("35_epi2_cavity_snug",     dict(cavity_r=0.089, cavity_h=0.089)),
    ("36_epi2_cavity_loose",    dict(cavity_r=0.160, cavity_h=0.160)),
    # ---- how hard the tissue presses, and an anisotropic cavity --------------------------
    ("37_epi2_contact_k2500",   dict(k_contact=2500.0)),
    ("38_epi2_disc_tight",      dict(cavity_r=0.200, cavity_h=0.095)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs", default=None, help="comma-separated leading numbers, e.g. 29,30")
    ap.add_argument("--cell-frames", type=int, default=401)
    ap.add_argument("--movie-frames", type=int, default=60)
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    import combine as C
    import run_ecm as R
    import tissue as TIS

    want = None if a.runs is None else {s.strip() for s in a.runs.split(",")}
    todo = [(n, o) for n, o in RUNS if want is None or n.split("_")[0] in want]
    npz = TIS.load_or_build(frames=a.cell_frames, device=a.device, buffer_x=BUFFER_X)

    rows, t0 = [], time.time()
    print(f"[epi2] {len(todo)} run(s) on {a.device}: {', '.join(n for n, _ in todo)}", flush=True)
    for i, (name, over) in enumerate(todo, 1):
        cfg = dict(BASE); cfg.update(over)
        print(f"\n[epi2 {i}/{len(todo)}] {name}  " +
              ("  ".join(f"{k}={v}" for k, v in over.items()) or "baseline"), flush=True)
        try:
            spec, info = C.build(name, npz, **cfg)
            out_dir = os.path.join(R.LOG, name)
            os.makedirs(out_dir, exist_ok=True)
            info["varied"] = over
            info["buffer_x"] = BUFFER_X
            json.dump(info, open(os.path.join(out_dir, "pass1.json"), "w"), indent=1)
            m = R.run(name, spec, device=a.device, movie=not a.no_movie,
                      render_kw={"movie_frames": a.movie_frames})
            m["varied"] = over
            rows.append(m)
        except Exception as e:
            traceback.print_exc()
            print(f"[epi2] {name} FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)
            rows.append({"name": name, "varied": over, "error": f"{type(e).__name__}: {e}"})
        # PER-WORKER FILE. Four of these run at once and a shared json would have each overwrite
        # the others' rows with its own partial view. `summarise.py` merges them.
        tag = (a.runs or "all").replace(",", "-")
        json.dump(rows, open(os.path.join(R.LOG, f"epi2_sweep_{tag}.json"), "w"), indent=1)

    print(f"\n[epi2] {len(rows)} run(s) in {(time.time()-t0)/60:.0f} min\n")
    for r in rows:
        if "error" in r:
            print(f"  {r['name']:26} ERROR {r['error'][:60]}")
        else:
            print(f"  {r['name']:26} contact={str(r.get('contact_frame')):>4}  "
                  f"strained_end={(r.get('strained_frac_end') or 0):.3f}  "
                  f"max_disp={(r.get('max_disp') or 0):.3f}  "
                  f"exploded={r.get('exploded')}")


if __name__ == "__main__":
    main()
