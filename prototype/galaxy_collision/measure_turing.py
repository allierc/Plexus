"""Numbers for the three Turing cards, read off the runs themselves.

Each card's caption quotes what this prints, in the run's own units:
  cells      how many cells carry the chemistry
  domains    connected components of the ACTIVATOR-HIGH cells over the same radius graph the
             simulation runs on (radius 0.95): the count of stripes or spots the pattern settles on
  covered    fraction of cells that end activator-high -- how much of the disc the pattern occupies
  overlap    (coupled run only) fraction of cells high in BOTH chemistries at once; the interesting
             number, because two chemistries on one cell set could either coincide or interleave

Run:  python prototype/galaxy_collision/measure_turing.py
"""
from __future__ import annotations
import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

LOG = "/workspace/Plexus/log/promotion"
RUNS = ["TURING_atlas_turing2d_gs_kappa", "TURING_atlas_turing2d_gs_eta",
        "TURING_atlas_turing2d_coupled"]
RADIUS = 0.95                     # the spec's own `radius_graph` radius


def _domains(pos: np.ndarray, high: np.ndarray) -> int:
    """Connected components of the activator-high cells, over the simulation's own graph."""
    idx = np.flatnonzero(high)
    if idx.size == 0:
        return 0
    pairs = cKDTree(pos[idx]).query_pairs(RADIUS, output_type="ndarray")
    if len(pairs) == 0:
        return int(idx.size)
    n = idx.size
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return int(connected_components(g, directed=False)[0])


def _high(chem: np.ndarray) -> np.ndarray:
    """Activator-high mask: above the midpoint of the field's own 5th--95th percentile range.
    A relative threshold, because the two Gray-Scott regimes do not share an absolute scale."""
    lo, hi = np.percentile(chem, 5), np.percentile(chem, 95)
    return chem > 0.5 * (lo + hi)


def main():
    for run in RUNS:
        d = np.load(os.path.join(LOG, run, "B", "trajectory.npz"))
        pos, chem = d["cell__pos"][-1], d["cell__chem"][-1]
        n = pos.shape[0]
        a0 = _high(chem[:, 0])
        line = (f"{run}\n   {n:,} cells, {d['cell__pos'].shape[0]} recorded frames | "
                f"activator domains {_domains(pos, a0)}, covering {100 * a0.mean():.0f}% of the disc")
        if chem.shape[1] >= 4:                       # the coupled run carries two A/H pairs
            a1 = _high(chem[:, 2])
            both = float((a0 & a1).mean())
            line += (f"\n   second chemistry: domains {_domains(pos, a1)}, covering "
                     f"{100 * a1.mean():.0f}% | both high at once: {100 * both:.1f}% of cells")
        print(line)


if __name__ == "__main__":
    main()
