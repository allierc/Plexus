#!/usr/bin/env python
"""The apico-basal thickness a spec's ENERGY wants, against the `h0` its seed declares.

WHY THIS EXISTS. `seed_mesh[apicobasal]` writes a uniform cell thickness `h0` and the energy then
decides the real one. Nothing checks the two against each other, and on every spec in the tree they
disagree -- `mech_shell_free` seeds 1.8 and settles at 0.799, `divide_growing_ball` seeds 0.4 and
rises. So the run opens with a ramp: on `mech_shell_free` the thickness takes about 300 frames to
arrive and the mean radius drifts 5.03 -> 7.47 on the way, because `_rest_offset` zeroes the RADIAL
force on the seeded mesh and the balance moves as the thickness collapses.

WHAT THAT COST, and it is the reason this tool is worth having rather than a one-off measurement.
Every reference the division machinery ever tried to take -- seed-time, first-call, settled,
frozen-at-1% -- was a measurement taken somewhere on that ramp, and each produced a different
population: 920, 12,543, 236, 200. None of them was wrong about the number it measured. They were
all measuring a shell that had not finished moving.

HOW IT MEASURES. The spec is stripped to `seed_mesh` + `cell_geometry` + `cell_mechanics` -- no
growth, no division, no T1 -- because the equilibrium is a property of the SEEDED GEOMETRY and the
ENERGY, and anything that changes cell volume confounds it. Then it runs until the median thickness
stops moving: relative drift under `--tol` across a trailing window, or `--max-frames`, whichever
comes first, and it SAYS which one it hit. A spec that has not converged is reported as such rather
than having its last value quoted as an equilibrium.

A FROZEN SEPARATION HAS NO EQUILIBRIUM TO FIND. `sep_mu: 0` pins the thickness at `h0` by
construction, so those specs are skipped and named.

    PYTHONPATH=src python tools/equilibrium_h.py --group tissue
    PYTHONPATH=src python tools/equilibrium_h.py --specs mech_shell_free divide_growing_ball
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# ONLY THESE THREE SURVIVE THE STRIP. `cell_geometry` is kept because `cell_mechanics` reads the
# cell set's `area`/`centroid`; everything else either changes the volume (`cell_grow`, `cell_divide`)
# or the topology (`edge_flip`), and both move the thing being measured.
KEEP = ("seed_mesh", "mesh_seed", "cell_geometry", "cell_mechanics", "topo_record")


def equilibrium(name, group="tissue", device="cuda:0", tol=5e-4, window=25, max_frames=1500):
    """(h0, h_eq, frames, converged) for one spec, or None if it has no apico-basal separation."""
    import plexus.operators                                            # noqa: F401  registry
    from plexus import schema
    from plexus.engine import run
    sim = schema.load(os.path.join(ROOT, "config", group, f"{name}.yaml"))
    seeds = [o for o in (sim.seed_ops or []) if o.op in ("seed_mesh", "mesh_seed")]
    # THE VARIANT IS `OpSpec.impl`, NOT A PARAM. The spec writes `model:` or `implementation:` and
    # the schema folds both into one field, because the engine instantiates a class either way.
    if not seeds or getattr(seeds[0], "impl", None) != "apicobasal":
        return None
    mech = [o for o in sim.operators if o.op == "cell_mechanics"]
    if mech and float(mech[0].params.get("sep_mu", 0.0)) == 0.0:
        return ("frozen", float(seeds[0].params.get("h0", 0.4)), None, 0, False)
    sim.operators = [o for o in sim.operators if o.op in KEEP]
    sim.schedule = [t for t in sim.schedule if (t if isinstance(t, str) else t.get("op")) in KEEP]
    sim.n_frames = int(max_frames)
    hs = []
    stop = {"at": None}

    def hook(H, t, *a, **k):
        lvl = H.level(sim_mesh_set(sim))
        nv = int(lvl.mesh["Nv"])
        hs.append(float(lvl.get("sep")[:nv].detach().norm(dim=1).median()) * 2.0)
        if stop["at"] is None and len(hs) > window:
            a0, a1 = hs[-window], hs[-1]
            if abs(a1 - a0) <= tol * max(abs(a0), 1e-12):
                stop["at"] = t
                raise StopIteration                       # the engine does not catch this; see below

    try:
        run(sim, out_path=None, device=device, on_frame=hook)
    except StopIteration:
        pass
    h0 = float(seeds[0].params.get("h0", 0.4))
    return ("ok", h0, hs[-1] if hs else float("nan"), len(hs) - 1, stop["at"] is not None)


def sim_mesh_set(sim):
    for n, s in (sim.sets or {}).items():
        if isinstance(s, dict) and s.get("mesh"):
            return n
    return "vertex"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="tissue")
    ap.add_argument("--specs", nargs="*", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tol", type=float, default=5e-4, help="relative drift over the window")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--max-frames", type=int, default=1500)
    a = ap.parse_args()
    names = a.specs or sorted(os.path.basename(p)[:-5]
                              for p in glob.glob(os.path.join(ROOT, "config", a.group, "*.yaml")))
    print(f"{'spec':<26} {'h0':>8} {'h_eq':>8} {'h_eq/h0':>8} {'frames':>7}  state")
    for n in names:
        try:
            r = equilibrium(n, a.group, a.device, a.tol, a.window, a.max_frames)
        except Exception as e:                                          # noqa: BLE001
            print(f"{n:<26} {'':>8} {'':>8} {'':>8} {'':>7}  FAILED {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        kind, h0, heq, frames, conv = r
        if kind == "frozen":
            print(f"{n:<26} {h0:8.4f} {'--':>8} {'--':>8} {'--':>7}  sep_mu 0: the thickness is "
                  f"pinned at h0 by construction")
            continue
        print(f"{n:<26} {h0:8.4f} {heq:8.4f} {heq / max(h0, 1e-12):8.3f} {frames:7d}  "
              + ("converged" if conv else f"NOT CONVERGED in {frames} frames -- h_eq is a floor, "
                                          f"not an answer"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
