"""Numbers for the two galaxy-encounter cards, read off the runs themselves.

Every figure quoted on the minisite comes from here, in the run's own dimensionless units:
  passage       time of closest approach between the two centres of mass, and that separation
  r50           median stellar radius about its own galaxy's centre (spawn radius is 1.2)
  stripped      fraction of stars that end farther than 4x the spawn radius from the pair centre
                -- the stars the encounter took out of both discs
  reach         95th-percentile radius at the last frame: how far the tails/debris get
  mixed         fraction of the remnant's inner stars (within 2x spawn radius of the pair centre)
                that came from the OTHER galaxy -- 0.5 = fully mixed

Run:  python prototype/galaxy_collision/measure.py [run ...]
"""
from __future__ import annotations
import os
import sys

import numpy as np

GD = "/groups/saalfeld/home/allierc/GraphData/graphs_data/inverse_square"
R_SPAWN = 1.2


def measure(run: str, dt: float = 0.004, n_frames: int = 10000) -> dict:
    d = np.load(os.path.join(GD, run, "trajectory.npz"))
    pos = d["star__pos"]; nt = d["star__node_type"]
    T = pos.shape[0]
    t = np.linspace(0.0, dt * n_frames, T)
    a, b = nt == 0, nt == 1
    ca = pos[:, a].mean(1); cb = pos[:, b].mean(1)
    sep = np.linalg.norm(ca - cb, axis=1)
    k = int(np.argmin(sep[: T // 2]))                  # the FIRST passage (the later minima are returns)
    last = pos[-1]
    c_all = last.mean(0); rad = np.linalg.norm(last - c_all, axis=1)
    inner = rad < 2 * R_SPAWN
    # "mixed": in the inner remnant, how even the two galaxies' contributions are (0.5 = even)
    frac_a = float(a[inner].mean()) if inner.any() else float("nan")
    out = dict(
        run=run, frames=T,
        passage_t=float(t[k]), passage_sep=float(sep[k]),
        sep_start=float(sep[0]), sep_end=float(sep[-1]), sep_max_after=float(sep[k:].max()),
        r50_a=float(np.median(np.linalg.norm(last[a] - last[a].mean(0), axis=1))),
        r50_b=float(np.median(np.linalg.norm(last[b] - last[b].mean(0), axis=1))),
        stripped=float((rad > 4 * R_SPAWN).mean()),
        reach=float(np.percentile(rad, 95)),
        mixed=min(frac_a, 1 - frac_a) * 2.0,           # 1.0 = evenly mixed, 0 = one galaxy only
        n=int(pos.shape[1]),
    )
    return out


def main():
    runs = sys.argv[1:] or ["galaxy_collision_3d", "galaxy_merger_3d"]
    for r in runs:
        m = measure(r)
        print(f"{m['run']}  N={m['n']}  {m['frames']} recorded frames")
        print(f"   first passage at t={m['passage_t']:.1f}, centres {m['passage_sep']:.2f} apart "
              f"(started {m['sep_start']:.1f}); after it the centres reach {m['sep_max_after']:.1f} "
              f"and end {m['sep_end']:.2f} apart")
        print(f"   final r50 red={m['r50_a']:.2f} blue={m['r50_b']:.2f} (spawn radius {R_SPAWN})")
        print(f"   stripped beyond 4 spawn radii: {100 * m['stripped']:.0f}%   95% inside "
              f"{m['reach']:.1f}   inner mixing {m['mixed']:.2f} (1.0 = even)")


if __name__ == "__main__":
    main()
