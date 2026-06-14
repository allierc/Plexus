"""Verification for the 2-level MPM scenario — the main concern.

  1. REPEATABLE  : same spec+seed -> bit-identical particle trajectory.
  2. WELL-FORMED : particles finite & in-domain; field finite.
  3. CELLS COHERE: MPM elasticity holds each cell together (particles stay near
                   their centroid) -> the hierarchy's containment is physical.
  4. INTENT MET  : soft cells (which make & follow the field) aggregate -- their
                   nearest-neighbour distance drops sharply -- while stiff cells
                   (which ignore it) stay dispersed -> "the field guides only the
                   first population".

    python verify2.py [scenario_mpm.yaml] [--cpu]
"""

import sys
import numpy as np
import torch

from scenario_schema import load
from engine2 import run


def _coh(pp, cp, par, fr):
    d = pp[fr] - cp[fr][par]
    return float(np.sqrt((d ** 2).sum(-1)).mean())


def _nnd(p):
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sc = load(args[0] if args else "scenario_mpm.yaml")
    dev = "cpu" if "--cpu" in sys.argv else ("cuda" if torch.cuda.is_available() else "cpu")

    _, a = run(sc, "/tmp/tg2_a.zarr", device=dev)
    _, b = run(sc, "/tmp/tg2_b.zarr", device=dev)
    pp, cp, par = a["particle_pos"], a["cell_pos"], a["parent"]
    t, names = a["cell_type"], a["type_names"]
    soft, stiff = t == names.index("soft"), t == names.index("stiff")
    checks = []

    checks.append(("repeatable (identical run twice)",
                   np.array_equal(a["particle_pos"], b["particle_pos"])))
    checks.append(("particles finite", bool(np.isfinite(pp).all())))
    checks.append(("particles in [0,1]", bool((pp >= 0).all() and (pp <= 1).all())))
    checks.append(("field finite", bool(np.isfinite(a["field"]).all())))

    c0, cT = _coh(pp, cp, par, 0), _coh(pp, cp, par, -1)
    checks.append((f"cells cohere (radius {c0:.4f}->{cT:.4f}, <2x)", cT < 2 * c0))

    # soft cells aggregate (NN distance drops); stiff stay dispersed
    s0, sT = _nnd(cp[0][soft]), _nnd(cp[-1][soft])
    st0, stT = _nnd(cp[0][stiff]), _nnd(cp[-1][stiff])
    soft_drop = 1 - sT / s0
    stiff_drop = 1 - stT / st0
    checks.append((f"soft aggregates (NNd {s0:.4f}->{sT:.4f}, -{soft_drop*100:.0f}%)",
                   soft_drop > 0.20))
    checks.append((f"stiff stays dispersed (NNd {st0:.4f}->{stT:.4f}, -{stiff_drop*100:.0f}%)",
                   stiff_drop < 0.12))

    print(f"\n=== verify2: {sc.name}  ({sc.sets['cell']['n']} cells x "
          f"{sc.sets['particle']['per_parent']} particles, {dev}) ===")
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and bool(passed)
    print(f"=== {'ALL PASS' if ok else 'FAILURES PRESENT'} ===\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
