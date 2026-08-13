#!/usr/bin/env python
"""epi_sweep -- five ECM runs, all loading the SAME cellfix_B_new epithelium.

    python epi_sweep.py --device cuda:0 --runs 24,25,26
    python epi_sweep.py --device cuda:1 --runs 27,28

ONE TISSUE, FIVE MATRICES. The epithelium is pass 1 (`tissue.py`) and is built once into
`log/okuda_ECM/_tissue/`, so all five runs replay a BIT-IDENTICAL tissue: the same 200 -> 3,170
cells, the same divisions, the same apical surface at every frame. Every difference between two of
these movies is therefore a difference in the MATRIX, which is the only way a five-run sweep is a
comparison rather than five anecdotes. It also means the tissue costs 69 seconds once instead of
six minutes.

ONE EDIT PER RUN, from `25_epi_ecm_E40`:

  24  youngs 10      a soft gel. The front should be broad and slow and the displacement large:
                     soft material yields rather than storing stress.
  25  --             the baseline. E = 40, a spherical cavity at 0.14 of the box, 2,600 fibres.
  26  youngs 150     a stiff gel. Same loading, less strain, a sharper and further-travelling
                     front -- stress that a soft matrix would have relaxed away instead.
  27  fibre geometry 700 long fibres instead of 2,600 short ones, at the SAME particle count. Two
                     numbers move because they are one question: many short strands against few
                     long ones. HONEST CAVEAT, the same one `ecm_ops` states: MPM interpolates every
                     particle onto a continuum grid, so a fibrous ARRANGEMENT of an isotropic
                     material still responds isotropically. Any difference here is where the
                     material happens to be DENSE, not fibre mechanics.
  28  disc cavity    r 0.22, h 0.11 about the vertical axis: the matrix is thin above and below the
                     tissue and thick around its equator. Contact is two events at different
                     frames rather than one. Note what this CANNOT show, because the coupling is
                     one-way: the tissue does not flatten. The anisotropy appears in the matrix's
                     stress field only.

WHAT TO LOOK FOR, decided before the runs rather than after: `contact_frame` should be ~129 for the
spherical-cavity runs (the frame the recorded apical surface first crosses r = 0.14) and EARLIER for
28, whose cavity wall is at 0.11 along the pinched axis. A run reporting contact at 0 is seeded
wrong and its movie is not evidence of anything -- which is what every run 01-23 in this folder
reported, and why the cavity is bigger than the tissue's starting radius now.
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
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "discovery_okuda", "ops"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

BASE = dict(n_particles=110000, n_grid=48, youngs=40.0, k_contact=900.0,
            cavity_r=0.14, cavity_h=0.14, axis=2, n_fibres=2600, fibre_len=0.16,
            align=0.0, substep_dt=2.0e-4)

RUNS = [
    ("24_epi_ecm_soft_E10",      dict(youngs=10.0)),
    ("25_epi_ecm_E40",           dict()),
    ("26_epi_ecm_stiff_E150",    dict(youngs=150.0)),
    ("27_epi_ecm_fibres_long",   dict(n_fibres=700, fibre_len=0.34)),
    ("28_epi_ecm_cavity_disc",   dict(cavity_r=0.22, cavity_h=0.11)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs", default=None, help="comma-separated leading numbers, e.g. 24,25")
    ap.add_argument("--cell-frames", type=int, default=401)
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    import combine as C
    import run_ecm as R
    import tissue as TIS

    want = None if a.runs is None else {s.strip() for s in a.runs.split(",")}
    todo = [(n, o) for n, o in RUNS if want is None or n.split("_")[0] in want]
    npz = TIS.load_or_build(frames=a.cell_frames, device=a.device)

    rows, t0 = [], time.time()
    print(f"[epi_sweep] {len(todo)} run(s) on {a.device}: "
          f"{', '.join(n for n, _ in todo)}", flush=True)
    for i, (name, over) in enumerate(todo, 1):
        cfg = dict(BASE); cfg.update(over)
        print(f"\n[epi_sweep {i}/{len(todo)}] {name}  " +
              ("  ".join(f"{k}={v}" for k, v in over.items()) or "baseline"), flush=True)
        try:
            spec, info = C.build(name, npz, **cfg)
            out_dir = os.path.join(R.LOG, name)
            os.makedirs(out_dir, exist_ok=True)
            info["varied"] = over
            json.dump(info, open(os.path.join(out_dir, "pass1.json"), "w"), indent=1)
            m = R.run(name, spec, device=a.device, movie=not a.no_movie)
            m["varied"] = over
            rows.append(m)
        except Exception as e:
            traceback.print_exc()
            print(f"[epi_sweep] {name} FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)
            rows.append({"name": name, "varied": over, "error": f"{type(e).__name__}: {e}"})
        # PER-DEVICE FILE. Two of these run at once, one per GPU, and a shared `sweep.json` would
        # have each process overwrite the other's rows with its own view of the world.
        tag = a.device.replace(":", "")
        json.dump(rows, open(os.path.join(R.LOG, f"epi_sweep_{tag}.json"), "w"), indent=1)

    print(f"\n[epi_sweep] {len(rows)} run(s) in {(time.time()-t0)/60:.0f} min -> {R.LOG}\n")
    hdr = (f"{'run':26}{'contact':>8}{'strained_end':>14}{'front_r95':>11}"
           f"{'max_disp':>10}{'exploded':>10}{'wall_s':>8}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['name'][:25]:26}{'ERROR':>8}  {r['error'][:52]}")
            continue
        print(f"{r['name'][:25]:26}{str(r.get('contact_frame')):>8}"
              f"{(r.get('strained_frac_end') or 0):>14.3f}{(r.get('front_r95_end') or 0):>11.3f}"
              f"{(r.get('max_disp') or 0):>10.3f}{str(r.get('exploded')):>10}"
              f"{(r.get('wall_s') or 0):>8.0f}")


if __name__ == "__main__":
    main()
