#!/usr/bin/env python
"""test_new_ops -- a pass/fail battery for the two new levels. Pass 1 only, so it costs no run folder.

    python test_new_ops.py --device cuda:0

WHY THESE TESTS AND NOT OTHERS. Every one of them is aimed at a failure mode this project has ALREADY
shipped at least once, because those are the failures with a demonstrated ability to look like results:

  T1 ABLATION      beta=0, activity=1 must be BIT-IDENTICAL to no myosin operator at all. This is the
                   only test that can catch "the hook is wired to the wrong array" or "the energy term
                   is being applied twice" -- both of which would still produce a plausible tissue.
                   Bit-identity is achievable because multiplying by exactly 1.0 is exact in IEEE, so
                   anything other than equality is a real difference.
  T2 MONOTONICITY  more myosin must mean more line tension must mean a smaller tissue. A non-monotone
                   response would mean the sign is wrong somewhere, which no single run can reveal.
  T3 PERSISTENCE   new junctions per frame must be a small multiple of the DIVISION rate, not the whole
                   edge count. If the topological keying fails, every junction reads as new every frame
                   and myosin silently resets -- the run still completes and still looks fine.
  T4 DETERMINISM   the same configuration twice must give the same answer. Cheap, and it is the test
                   that catches state leaking between runs through a module-level list, which this
                   codebase has done before (`STRESS_HISTORY` needed clearing per run for exactly that).
  T5 EXTREMES      activity 0 and 3 must not produce NaN or an inverted mesh. A model that only behaves
                   in the middle of its range is one whose sweeps cannot be trusted at the ends.
  T6 CACHE KEY     two configurations differing only in a myosin parameter must land in DIFFERENT cache
                   files. Shipped twice already; now it has a test.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "prototype", "Tyssue"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}", flush=True)
    return ok


def fingerprint(npz):
    """A few reduced numbers that any real change to the mechanics would move."""
    z = np.load(npz)
    return dict(cells=int(z["n_cells"][-1]),
                r_apical=float(z["r_apical"][-1]),
                r_xyz=tuple(float(v) for v in np.asarray(z["r_xyz"])[-1]),
                smap_sum=float(np.asarray(z["smap"], np.float64).sum()),
                path=npz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=120)
    a = ap.parse_args()
    import tissue as TIS
    import junction_ops
    F = a.frames

    print("\nT1  ABLATION: beta=0, activity=1 must be bit-identical to no myosin at all")
    base = fingerprint(TIS.load_or_build(frames=F, device=a.device, buffer_x=4))
    abl = fingerprint(TIS.load_or_build(frames=F, device=a.device, buffer_x=4,
                                        myosin=1.0, myo_beta=0.0, myo_new=1.0))
    same = (base["cells"] == abl["cells"] and base["r_apical"] == abl["r_apical"]
            and base["r_xyz"] == abl["r_xyz"] and base["smap_sum"] == abl["smap_sum"])
    check("T1 ablation is bit-identical", same,
          f"cells {base['cells']} vs {abl['cells']}, r {base['r_apical']:.9g} vs "
          f"{abl['r_apical']:.9g}, smap {base['smap_sum']:.12g} vs {abl['smap_sum']:.12g}")

    print("\nT2  MONOTONICITY: more myosin -> more line tension -> smaller tissue")
    rows = []
    for act in (0.25, 0.5, 1.0, 2.0):
        f = fingerprint(TIS.load_or_build(frames=F, device=a.device, buffer_x=4, myosin=act))
        rows.append((act, f["r_apical"], f["cells"]))
        print(f"    activity {act:<5} r_apical {f['r_apical']:.4f}  cells {f['cells']}")
    r = [x[1] for x in rows]
    check("T2 radius decreases with myosin", all(r[i] >= r[i + 1] for i in range(len(r) - 1)),
          f"radii {[round(x,4) for x in r]}")

    print("\nT3  PERSISTENCE: new junctions/frame must track DIVISIONS, not the edge count")
    junction_ops.MYOSIN_TRACE.clear()
    npz = TIS.load_or_build(frames=F, device=a.device, buffer_x=4, myosin=1.0, myo_beta=1.5,
                            rebuild=True)
    tr = np.asarray(junction_ops.MYOSIN_TRACE, float)
    z = np.load(npz)
    dcells = int(z["n_cells"][-1]) - int(z["n_cells"][0])
    new_total, edges_end = float(tr[1:, 4].sum()), float(tr[-1, 0])
    # a division adds a handful of half-edges; the whole edge count per frame would be ~100x more
    check("T3 new junctions are a small multiple of divisions",
          new_total < 30 * max(dcells, 1) and new_total > 0,
          f"{new_total:.0f} new over the run for {dcells} new cells "
          f"({new_total / max(dcells,1):.1f} per division; the edge count is {edges_end:.0f}/frame)")

    print("\nT4  DETERMINISM: the same configuration twice")
    p1 = fingerprint(TIS.load_or_build(frames=F, device=a.device, buffer_x=4, myosin=0.7,
                                       rebuild=True))
    p2 = fingerprint(TIS.load_or_build(frames=F, device=a.device, buffer_x=4, myosin=0.7,
                                       rebuild=True))
    check("T4 deterministic", p1["cells"] == p2["cells"] and p1["smap_sum"] == p2["smap_sum"],
          f"cells {p1['cells']}/{p2['cells']}, smap {p1['smap_sum']:.12g}/{p2['smap_sum']:.12g}")

    print("\nT5  EXTREMES: activity 0 and 3 must stay finite and non-degenerate")
    for act in (0.0, 3.0):
        f = fingerprint(TIS.load_or_build(frames=F, device=a.device, buffer_x=4, myosin=act))
        ok = (np.isfinite(f["r_apical"]) and f["r_apical"] > 0.5 * 4.66 and f["cells"] > 200)
        check(f"T5 activity {act} finite and grew", ok,
              f"r {f['r_apical']:.3f}, cells {f['cells']}")

    print("\nT6  CACHE KEY: a myosin parameter must change the cache file")
    pa = TIS.load_or_build(frames=F, device=a.device, buffer_x=4, myosin=1.0, myo_tau=20.0)
    pb = TIS.load_or_build(frames=F, device=a.device, buffer_x=4, myosin=1.0, myo_tau=60.0)
    check("T6 distinct cache per myosin config", os.path.basename(pa) != os.path.basename(pb),
          f"{os.path.basename(pa)} vs {os.path.basename(pb)}")

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n=== {n_pass}/{len(RESULTS)} passed ===")
    for nm, ok, d in RESULTS:
        if not ok:
            print(f"  FAILED: {nm}  {d}")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
