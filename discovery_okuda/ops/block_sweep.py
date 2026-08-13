#!/usr/bin/env python
"""block_sweep -- the blocks as an elastic MATERIAL, so their deformation is visible too.

    python block_sweep.py --device cuda:0 --runs 45,46

TWO RUNS, AND THEY ASK DIFFERENT QUESTIONS.

  45  blocks 25% of the box (free gap 0.375), tissue UNCONFINED. The tissue never reaches the blocks --
      it only ever gets to 0.30 -- so everything the blocks feel arrives THROUGH the matrix. This is the
      run for watching the ECM get squeezed between a growing sphere and a solid, and for watching the
      solid answer: force transmitted through fibres, which is the thing a rigid plate can never show.
  46  blocks 75% of the box (free gap 0.126), tissue CONFINED to an ovoid of aspect 2.08. Here the
      tissue is pressing almost directly on the blocks through a thin layer of matrix, so the block
      deformation is large and localised under the flat faces of the ovoid.

THE INCONSISTENCY IN 46, STATED. The tissue's ovoid shape was produced in pass 1 by a RIGID
`plate_confine` at the same gap, because a replayed tissue cannot feel an elastic block -- pass 1
finished first. So run 46 draws a tissue shaped by an infinitely stiff plate against a block that then
visibly yields. The two are not the same boundary, and the difference is exactly the size of the block's
dent. It is the same one-way limitation `combine.py` documents, showing up in a new place; `ecm_load`
is the direction out of it.
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

BUFFER_X = 4
BASE = dict(n_particles=140000, n_grid=48, youngs=15.0, k_contact=1200.0, a_max=300.0,
            cavity_r=0.095, cavity_h=0.095, axis=2, n_fibres=6000, fibre_len=0.16,
            # COLOURED BY VON MISES OF THE CAUCHY STRESS the solver itself computed, not by |J-1|.
            # Measured on a 260-frame calibration run: von Mises has median 0.00094, p99 0.019,
            # max 0.054, so at scale 0.008 the matrix runs 64% above band 0 with 7.6% saturated -- a
            # readable field. The same run's |J-1| put 0.6% above band 0. The stress was always there;
            # a volumetric measure cannot see a material that is being SHEARED, and MLS-MPM's
            # fixed-corotated law resists volume change far more stiffly than shape change.
            align=0.0, substep_dt=2.0e-4, stress_scale=0.008, stress_measure="vonmises",
            # The block is ~130x stiffer, so the same load leaves it at a far lower stress -- its own
            # scale, or it reads as uniformly unloaded in every frame.
            block_youngs=2000.0, block_particles=60000, block_stress_scale=0.05,
            block_measure="vonmises")

# `block gap = None` means USE THE GAP THE TISSUE WAS GROWN AGAINST, converted to box units by the one
# scale. That is the self-consistent choice: the elastic block then occupies exactly the region the
# rigid plate of pass 1 occupied, so the tissue's ovoid and the block's dent are the same boundary
# rather than two boundaries that happen to be near each other.
#
# THE GEOMETRY THAT MATTERS, MEASURED. `44_plate_blk25_cushion` put the blocks at 25% of the volume --
# a free gap of 0.75 of the box against a tissue that only reaches 0.30 -- and there was nothing left
# to compress: no stress in the matrix and no ovoid. The free gap has to be comparable to the tissue,
# not to the box:
#
#     free gap (box)   0.75    0.65    0.40    0.33    0.25
#     blocks (% vol)    25%     35%     60%     67%     75%
#     tissue aspect    1.00    1.06    1.41    1.65    2.08
#
#      name                        block gap (box)   tissue plate gap (tissue units, None = free)
RUNS = [("47_block_elastic_g33",         None,       8.4),    # free gap ~1/3.3 of the box, ~70% solid
        ("48_block_elastic_g40",         None,      11.1)]    # free gap ~0.40 of the box, 60% solid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--cell-frames", type=int, default=401)
    ap.add_argument("--movie-frames", type=int, default=90)
    a = ap.parse_args()

    import combine as C
    import run_ecm as R
    import tissue as TIS

    want = None if a.runs is None else {s.strip() for s in a.runs.split(",")}
    todo = [r for r in RUNS if want is None or r[0].split("_")[0] in want]
    rows, t0 = [], time.time()
    print(f"[block] {len(todo)} run(s) on {a.device}", flush=True)
    for i, (name, bgap, pgap) in enumerate(todo, 1):
        print(f"\n[block {i}/{len(todo)}] {name}  block_gap={bgap} box  tissue_plate={pgap}",
              flush=True)
        try:
            npz = TIS.load_or_build(frames=a.cell_frames, device=a.device, buffer_x=BUFFER_X,
                                    plate_gap=pgap)
            cfg = dict(BASE)
            # `plate_box=None`: the ELASTIC block replaces the rigid projection in the matrix. Keeping
            # both would constrain the matrix twice, with the projection winning every time, and the
            # block would be a decoration on a rigid wall.
            # Two passes over `build`: the first only to learn what the pass-1 plate gap maps to in box
            # units, the second with the elastic block placed exactly there.
            if bgap is None:
                _, probe = C.build(name, npz, plate_box=None, **cfg)
                bgap = float(probe["plate_gap_box"])
            cfg["block_gap"] = bgap
            spec, info = C.build(name, npz, plate_box=None, **cfg)
            out_dir = os.path.join(R.LOG, name)
            os.makedirs(out_dir, exist_ok=True)
            info["varied"] = {"block_gap_box": bgap, "block_youngs": BASE["block_youngs"],
                              "tissue_plate_gap": pgap}
            info["block_volume_frac"] = 1.0 - 2.0 * bgap
            json.dump(info, open(os.path.join(out_dir, "pass1.json"), "w"), indent=1)
            print(f"[block] {name}: elastic blocks E={BASE['block_youngs']:g} beyond "
                  f"+/-{bgap} = {100 * (1 - 2 * bgap):.0f}% of the box; tissue aspect "
                  f"{info['aspect_end']:.2f}, r_eq {info['tissue_r_end_box']:.3f} of the box",
                  flush=True)
            m = R.run(name, spec, device=a.device, movie=True,
                      render_kw={"movie_frames": a.movie_frames})
            m["varied"] = info["varied"]
            m["aspect_end"] = info["aspect_end"]
            rows.append(m)
        except Exception as e:
            traceback.print_exc()
            print(f"[block] {name} FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)
            rows.append({"name": name, "error": f"{type(e).__name__}: {e}"})
        json.dump(rows, open(os.path.join(R.LOG, "block_sweep.json"), "w"), indent=1)
    print(f"\n[block] {len(rows)} run(s) in {(time.time()-t0)/60:.0f} min")
    for r in rows:
        print(f"  {r['name']:26} " + (f"ERROR {r['error'][:60]}" if "error" in r else
              f"aspect={r.get('aspect_end', 0):.2f}  contact={r.get('contact_frame')}  "
              f"strained_end={(r.get('strained_frac_end') or 0):.3f}  "
              f"exploded={r.get('exploded')}"))


if __name__ == "__main__":
    main()
