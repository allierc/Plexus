#!/usr/bin/env python
"""plate_sweep -- two solid blocks squash the growing epithelium into an ovoid, and the matrix in
between records what that costs.

    python plate_sweep.py --device cuda:0 --runs 39,40
    python plate_sweep.py --device cuda:1 --runs 41,42

THE ARITHMETIC THAT PINS THIS EXPERIMENT DOWN, because it decides what is askable. Two numbers are
wanted independently and cannot be: how SQUASHED the tissue gets, and how much of the box is SOLID.
The tissue's widest extent is pinned to `fit` = 0.30 of the box half-width -- larger and the matrix has
nowhere to be, which is what the experiment is about. So the plate gap, in box units, is
`gap_tissue * fit / r_eq`, and the block fraction is whatever that leaves:

    plate gap (tissue units)   13.5    11.1     9.2     7.0
    aspect ratio at frame 401  1.21    1.41    1.65    ~2.0
    blocks, % of box volume     52%     60%     67%     75%

A CONSEQUENCE WORTH STATING BEFORE ANYONE ASKS FOR IT: blocks filling 1/3 of the volume -- the natural
thing to ask for -- CANNOT squash this tissue. 1/3 solid means a free gap of 0.333 box units, which at
this scale is 18.5 tissue units, and the unconfined tissue only ever reaches 16.5. It never touches
them. `39_plate_null` is that run, and it is included because a null with a reason on it is worth more
than an omission: the tissue has to be able to reach the plates, and at `fit` 0.30 that needs the
blocks to take more than half the box.

WHAT THE PLATES ARE, AND ARE NOT. They are RIGID: they do not deform, so the shape the tissue takes is
its own mechanics answering a fixed boundary. That makes them the right FIRST experiment and the wrong
LAST one -- a real matrix resists by deforming, and `ecm_load_3d` is that version. Read against these
runs it says how much of the ovoid a soft matrix can actually produce.

THE SECOND RESULT, WHICH WAS NOT THE POINT AND IS MORE INTERESTING THAN THE FIRST. Confinement
SUPPRESSES PROLIFERATION here: 5,968 cells unconfined, 5,470 at gap 11.1, 4,951 at gap 9.2. Nothing in
the spec tells cells to stop dividing when squeezed. `divide_3d` fires on volume doubling and
`shape_energy_3d` charges a cell for being compressed, so a confined cell reaches its division volume
later -- mechanical feedback on the cell cycle, emerging from two operators that were not written to
produce it. It is measured, not modelled, and it is why `n_cells` is reported per run.
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

# The epi2 matrix, unchanged, so a plate run is comparable to `29_epi2_tight_dense_E15` -- which is
# this experiment's no-plate control and was already run.
# k_contact RAISED 900 -> 4000 and a_max 200 -> 800. Matrix particles were ending up INSIDE the
# epithelium: the contact is a penalty (k * depth) and `mpm_scatter` clamps the resulting acceleration,
# so the two together set a ceiling on how hard the tissue can push a particle out -- at 900/200 that
# ceiling arrived at depth 0.22, most of the tissue's radius. `cell_exclude_3d` is the hard backstop
# that guarantees non-penetration; these two are what keep the force physical on the way there.
BASE = dict(n_particles=140000, n_grid=48, youngs=15.0, k_contact=1200.0, a_max=300.0,
            cavity_r=0.095, cavity_h=0.095, axis=2, n_fibres=6000, fibre_len=0.16,
            align=0.0, substep_dt=2.0e-4,
            # von Mises of the stored Cauchy stress, so 43 is comparable with 47/48 rather than with the
            # \|J-1\| runs it was first measured against -- see log/okuda_ECM/DEFECTS.md.
            stress_scale=0.008, stress_measure="vonmises")

RUNS = [
    ("39_plate_null_blk33",  18.5),      # blocks 1/3 of the box: never reached. The null.
    ("40_plate_blk52",       13.5),
    ("41_plate_blk60",       11.1),
    ("42_plate_blk67",        9.2),
    ("43_plate_blk75",        7.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--cell-frames", type=int, default=401)
    ap.add_argument("--movie-frames", type=int, default=90)
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    import combine as C
    import run_ecm as R
    import tissue as TIS

    want = None if a.runs is None else {s.strip() for s in a.runs.split(",")}
    todo = [(n, g) for n, g in RUNS if want is None or n.split("_")[0] in want]

    rows, t0 = [], time.time()
    print(f"[plate] {len(todo)} run(s) on {a.device}: {', '.join(n for n, _ in todo)}", flush=True)
    for i, (name, gap) in enumerate(todo, 1):
        print(f"\n[plate {i}/{len(todo)}] {name}  plate_gap={gap} tissue units", flush=True)
        try:
            # PASS 1 PER GAP, cached: the tissue's SHAPE is what the gap changes, so every plate run
            # needs its own tissue. This is the one thing in the project that is not shared across a
            # sweep, and it is 80 seconds.
            npz = TIS.load_or_build(frames=a.cell_frames, device=a.device, buffer_x=BUFFER_X,
                                    plate_gap=gap)
            spec, info = C.build(name, npz, **dict(BASE))
            out_dir = os.path.join(R.LOG, name)
            os.makedirs(out_dir, exist_ok=True)
            info["varied"] = {"plate_gap": gap}
            info["buffer_x"] = BUFFER_X
            json.dump(info, open(os.path.join(out_dir, "pass1.json"), "w"), indent=1)
            print(f"[plate] {name}: aspect {info['aspect_end']:.2f} "
                  f"(r_eq {info['tissue_r_eq_end']:.2f} / r_ax {info['tissue_r_ax_end']:.2f}), "
                  f"{info['cells_end']} cells, plates at +/-{info['plate_gap_box']:.3f} of the box "
                  f"= {100 * info['block_volume_frac']:.0f}% solid", flush=True)
            m = R.run(name, spec, device=a.device, movie=not a.no_movie,
                      render_kw={"movie_frames": a.movie_frames})
            m["varied"] = info["varied"]
            m["aspect_end"] = info["aspect_end"]
            m["block_volume_frac"] = info["block_volume_frac"]
            m["cells_end"] = info["cells_end"]
            rows.append(m)
        except Exception as e:
            traceback.print_exc()
            print(f"[plate] {name} FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)
            rows.append({"name": name, "error": f"{type(e).__name__}: {e}"})
        tag = (a.runs or "all").replace(",", "-")
        json.dump(rows, open(os.path.join(R.LOG, f"plate_sweep_{tag}.json"), "w"), indent=1)

    print(f"\n[plate] {len(rows)} run(s) in {(time.time()-t0)/60:.0f} min\n")
    for r in rows:
        if "error" in r:
            print(f"  {r['name']:24} ERROR {r['error'][:60]}")
        else:
            print(f"  {r['name']:24} aspect={r.get('aspect_end', 0):.2f}  "
                  f"cells={r.get('cells_end')}  solid={100 * (r.get('block_volume_frac') or 0):.0f}%"
                  f"  contact={r.get('contact_frame')}  "
                  f"strained_end={(r.get('strained_frac_end') or 0):.3f}  "
                  f"exploded={r.get('exploded')}")


if __name__ == "__main__":
    main()
