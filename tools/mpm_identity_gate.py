#!/usr/bin/env python
"""Byte-identity gate for the MPM operators: save a reference, then prove a change did not move it.

WHY THIS EXISTS SEPARATELY FROM `promotion_identical.py`. That harness compares okuda against the
core, freshly generated on both sides, and it is the right tool for promotion. It is the wrong tool
for a same-tree refactor, where the question is narrower and asked far more often: "I rewrote the
inside of an operator for speed -- did any bit move?" That question wants a reference captured
BEFORE the edit and re-checked after, in one command, on a machine with no cluster.

WHAT IT COVERS, and why these specs. The MPM operators branch on things a single spec cannot
exercise at once, and every one of those branches has hidden a defect this month:

  cell_02_nucleus_bounce     3D, ONE particle set          -- the plain scatter/gather path
  cell_03_nucleus_cytosol    3D, TWO sets, liquid + solid  -- the shared-grid ACCUMULATE path,
                                                              and the CSF surface-tension term
  cell_05_membrane           3D, THREE sets                -- accumulate with more than two, which
                                                              is what makes "who zeroes the grid"
                                                              a question rather than a tautology
  material_3d_multimaterial  3D, one set, THREE types      -- jelly + water + snow: the snow
                                                              hardening and liquid branches
  material_two_drops_st      2D                            -- the entirely separate 2D wall BC and
                                                              analytic 2D polar rotation

A change that is identical on cell_02 and wrong on cell_05 is the exact shape of the shared-grid
bug: one set scattering is a special case in which overwriting and accumulating agree.

WHAT IS COMPARED. Final-frame positions per set, byte for byte (`tobytes()`), plus a float64
checksum for the report. Not a tolerance -- a tolerance turns "this refactor is a no-op" into "this
refactor is small", and those are different claims.

    python tools/mpm_identity_gate.py --save  /tmp/ref.npz          # before the edit
    python tools/mpm_identity_gate.py --check /tmp/ref.npz          # after; exit 1 on any mismatch
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SPECS = [
    ("config/cell/cell_02_nucleus_bounce.yaml", 40),
    ("config/cell/cell_03_nucleus_cytosol.yaml", 30),
    ("config/cell/cell_05_membrane.yaml", 25),
    ("config/material/material_3d_multimaterial.yaml", 30),
    ("config/material/material_two_drops_st.yaml", 30),
]


def collect(device: str, frames_scale: float = 1.0) -> dict:
    import plexus.operators  # noqa: F401  self-register
    from plexus.schema import load
    from plexus import engine as E

    out = {}
    for rel, n in SPECS:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", rel)
        sim = load(path)
        sim.n_frames = max(2, int(n * frames_scale))
        # THE PERFORMANCE FLAGS ARE STRIPPED. A spec may carry `capture: true` or `compile: true`
        # while someone is experimenting; the gate's subject is the OPERATORS, so it always runs
        # the eager path. (`capture` is separately proven bit-identical to eager by its own gate;
        # `compile` is not, by construction.) `polar:` is NOT stripped -- that is a real numerical
        # choice a spec makes, and a reference taken before it changed SHOULD fail.
        for st in sim.schedule:
            if isinstance(st, dict):
                st.pop("capture", None)
                st.pop("compile", None)
                st.pop("compile_mode", None)
        _H, tr = E.run(sim, out_path=None, device=device, progress=False)
        name = os.path.basename(rel)[:-5]
        for sname, blk in tr["sets"].items():
            pos = np.asarray(blk["pos"])
            if pos.ndim != 3 or pos.shape[1] <= 1:      # skip the parent point set
                continue
            out[f"{name}::{sname}"] = pos[-1].astype(np.float32, copy=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", metavar="REF")
    g.add_argument("--check", metavar="REF")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every spec's frame count")
    a = ap.parse_args()

    got = collect(a.device, a.scale)
    if a.save:
        np.savez(a.save, **got)
        print(f"\n  reference saved: {a.save}   ({len(got)} set(s) over {len(SPECS)} specs)")
        for k, v in got.items():
            print(f"    {k:<44} {v.shape[0]:>7,} particles   checksum {v.astype(np.float64).sum():.6f}")
        return 0

    ref = np.load(a.check)
    keys = sorted(set(got) | set(ref.files))
    bad = 0
    print(f"\n  {'spec::set':<44}{'particles':>11}{'verdict':>14}{'max|diff|':>13}")
    print("  " + "-" * 82)
    for k in keys:
        if k not in got or k not in ref.files:
            print(f"  {k:<44}{'':>11}{'MISSING':>14}{'':>13}")
            bad += 1
            continue
        A, B = np.asarray(ref[k]), got[k]
        if A.shape != B.shape:
            print(f"  {k:<44}{B.shape[0]:>11,}{'SHAPE':>14}{'':>13}")
            bad += 1
        elif A.tobytes() == B.tobytes():
            print(f"  {k:<44}{B.shape[0]:>11,}{'IDENTICAL':>14}{0.0:>13.1e}")
        else:
            print(f"  {k:<44}{B.shape[0]:>11,}{'*** DIFFERS':>14}{np.abs(A - B).max():>13.3e}")
            bad += 1
    print("  " + "-" * 82)
    print(f"  {'PASS' if not bad else f'FAIL -- {bad} of {len(keys)} moved'}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
