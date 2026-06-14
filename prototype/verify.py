"""Verification harness — the main concern.

Three classes of check:
  1. REPEATABLE : same spec + seed -> bit-identical trajectory.
  2. WELL-FORMED : trajectory finite, in-domain; field non-negative.
  3. INTENT MET  : the *only* population told to follow the field (soft) aggregates,
                   while the other (stiff) does not. This checks the scenario did
                   what the sentence asked, not merely that it ran.

    python verify.py [scenario.yaml]
"""

import sys
import numpy as np

from scenario_schema import load
from engine import run


def _mean_nn_dist(p):
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1).mean()


def main():
    sc = load(sys.argv[1] if len(sys.argv) > 1 else "scenario.yaml")
    _, a = run(sc, "/tmp/tg_verify_a.zarr")
    _, b = run(sc, "/tmp/tg_verify_b.zarr")
    checks = []

    # 1. repeatable
    same = np.array_equal(a["cell_pos"], b["cell_pos"])
    checks.append(("repeatable (identical run twice)", same))

    # 2. well-formed
    pos = a["cell_pos"]
    checks.append(("trajectory finite", bool(np.isfinite(pos).all())))
    checks.append(("positions in [0,1]", bool((pos >= 0).all() and (pos <= 1).all())))
    checks.append(("field non-negative", bool((a["field"] >= 0).all())))

    # 3. intent met: soft (follows field) aggregates more than stiff (does not)
    types, names = a["cell_type"], a["type_names"]
    soft, stiff = (types == names.index("soft")), (types == names.index("stiff"))
    soft_0, soft_T = _mean_nn_dist(pos[0][soft]), _mean_nn_dist(pos[-1][soft])
    stiff_0, stiff_T = _mean_nn_dist(pos[0][stiff]), _mean_nn_dist(pos[-1][stiff])
    soft_aggregates = soft_T < 0.9 * soft_0
    soft_tighter = soft_T < stiff_T
    checks.append((f"soft aggregates (nn {soft_0:.4f}->{soft_T:.4f})", bool(soft_aggregates)))
    checks.append((f"soft tighter than stiff ({soft_T:.4f} < {stiff_T:.4f})", bool(soft_tighter)))

    print(f"\n=== verify: {sc.name} ===")
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"=== {'ALL PASS' if ok else 'FAILURES PRESENT'} ===\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
