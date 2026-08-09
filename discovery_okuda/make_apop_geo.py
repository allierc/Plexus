#!/usr/bin/env python
"""Dedicated geometry for testing cell death, where death is the ONLY thing happening.

Cedric, 9 August: "can you build dedicated geometry to test cell death now."

WHY A SUBSTRATE OF ITS OWN. Every test so far ran death inside a campaign composition -- chemistry
reacting, cells growing, division firing -- so anything death did arrived mixed with everything
else, and a rule that marked cells but never extruded them was indistinguishable from a rule that
marked nothing. `competition` on r010_12 is exactly that case: zero deaths, yet P4 broken, because
it marks cells that shrink for the rest of the run without ever reaching the triangle the collapse
needs. That is a pathway failure wearing the costume of a quiet selector.

So these carry six operators and no others:

    seed_mesh_3d -> cell_geometry_3d -> apoptosis_3d -> shape_energy_3d
                 -> reconnect_t1_3d -> topo_snapshot_3d

No chemistry, no growth, no division. The cell count can only ever go DOWN, so `n_apop` and the
count must agree exactly -- on a growing tissue they cannot be cross-checked at all, which is how
the death counter came to be needed in the first place.

400 CELLS, NOT 2,000. One death is 0.25% of the sheet rather than 0.05%, so a single extrusion is
visible in the movie; and the runs are minutes rather than tens of minutes, which is what makes a
failed pathway cheap to find.

THE FIVE GEOMETRIES, ordered by how much they ask of the topology:

    one     a single named cell. The correctness case: euler 2 throughout, exactly one death.
    cap     a 22.8-degree cone, ~24 cells here. Does a dying patch pull the surface INWARD --
            the `invagination` metric exists because global descriptors could not see this.
    ring    an 8-degree equatorial band. Does the sphere constrict where the band is removed;
            measured 1.009 -> 1.155 gyr_prolate on the 2,000-cell version.
    rings9  nine bands, ~45% of the sheet. On the large version the vesicle did NOT collapse --
            it closed over every gap and stayed a sphere at reduced_volume 0.981. Worth knowing
            whether that survives at a quarter of the cell count.
    half    everything above the equator. The extreme: the largest single hole the collapse
            machinery is asked to close, and the case most likely to break `_check_closed`.

    python make_apop_geo.py            write the specs
    python make_apop_geo.py --check    also run the static premises and the unread-key gate
"""
import argparse
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(os.path.dirname(HERE), "config", "okuda")

N_CELLS = 400
FRAMES = 600
# The mechanics that let a marked cell actually be shed. `Lambda` is deliberately LOW: it charges
# for edge length, so a high value resists the very neighbour-exchange the pathway depends on --
# Route A measured grip collapsing 0.051 -> 0.002 across Lambda 0 -> 2 for the same reason.
MECH = dict(K_A=1.0, K_P=1.0, K_V=20.0, K_R=0.0, Lambda=0.5, Gamma=0.4, p0=3.5,
            mu=1.0, dt=1.0, relax_iters=30, eta=0.08, cap_frac=0.12)

GEO = {
    "one":    dict(mode="list", cells=[137]),
    "cap":    dict(mode="cone", cone_deg=22.8),
    "ring":   dict(mode="band", band_deg=8.0, n_bands=1),
    "rings9": dict(mode="band", band_deg=4.0, n_bands=9),
    "half":   dict(mode="band", band_deg=45.0, n_bands=1),
}


def build(tag, apo):
    ops = [
        {"op": "seed_mesh_3d", "at": "vertex", "cell_set": "cell", "before_frame": 1,
         "n_cells": N_CELLS, "seed": 0, "vseed_cv": 0.15, "radius": 5.0, "jitter": 0.15,
         "p0": 3.5},
        {"op": "cell_geometry_3d", "at": "cell"},
        # DEATH BEFORE THE MECHANICS, so a cell extruded this frame is relaxed this frame rather
        # than leaving a raw hole for one. The delta-renumbering fix means the position no longer
        # affects correctness -- it did until 9 August, and that is worth not relying on twice.
        {"op": "apoptosis_3d", "at": "vertex", "cell_set": "cell", "p0": 3.5,
         "shrink_rate": 0.05, "critical_frac": 0.15, "min_age": 0, **apo},
        {"op": "shape_energy_3d", "at": "vertex", **MECH},
        {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.28, "every": 1,
         "max_flips": 300},
        {"op": "topo_snapshot_3d", "at": "vertex", "every": 1},
    ]
    return {
        "general": {"name": tag, "seed": 0, "n_frames": FRAMES, "dt": 1.0,
                    "record_cap": FRAMES + 2, "record_every": 1, "boundary": "free", "dim": 3,
                    "world": [80.0, 80.0, 80.0]},
        "_run": {"target_cells": 4000, "seed_cells": N_CELLS},
        "sets": {"vertex": {"n": 8000},
                 "cell": {"n": 4000,
                          "state": {"chem": {"width": 2, "integration": "first_order"},
                                    "cen": {"width": 3}, "area": {"width": 1}}}},
        "fields": {},
        "operators": ops,
        "schedule": [o["op"] for o in ops],
        "_apopgeo": {"geometry": tag.replace("apopgeo_", ""), "selector": apo,
                     "why": "death is the only mechanism in this composition, so the cell count "
                            "can only fall and n_apop must equal 400 - cells_final exactly"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    print(f"{'spec':<20}{'selector':<44}{'ops':>4}  {'gate'}")
    bad = 0
    for g, apo in GEO.items():
        tag = f"apopgeo_{g}"
        spec = build(tag, apo)
        with open(os.path.join(CONFIG, f"{tag}.yaml"), "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
        note = ""
        if a.check:
            import biologist as B
            from make_basis import _unread
            fails = [r.pid for r in B.check(spec) if r.status == "fail"] + _unread(spec)
            bad += bool(fails)
            note = "BROKEN " + ",".join(fails) if fails else "ok"
        print(f"{tag:<20}{str(apo)[:43]:<44}{len(spec['operators']):>4}  {note}")
    print(f"\n{len(GEO)} specs -> {CONFIG}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
