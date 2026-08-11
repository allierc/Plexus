"""
06_spheroid_bm -- THE BM HALF OF 06, AT ITS KNOWN-GOOD POINT.

This runs 05b's nominal rig and nothing else: no density sweep, no rupture, no control. It exists
so that 06 has a spheroid+BM folder whose movie is sound, next to 06_spheroid_ecm's spheroid+matrix.

WHAT THE SPHEROID IS HERE, SAID PLAINLY. It is a driven icosphere -- 642 vertices, dilating
UNIFORMLY under k_drive toward a prescribed radius. It has no cells, no division, no T1, and its
vertex count never changes. The plaques follow it perfectly because every point they hold moves
radially by the same factor. That is why this run works, and it is the whole distance between this
folder and 06_three_bodies, where the driver is the replayed vertex model: 396 vertices at the
seeding frame growing to thousands, moving non-uniformly, changing topology underneath the bonds.

So this folder is evidence about the BM MACHINERY -- plaque force, standoff, momentum, slip -- and
is NOT evidence that the BM tracks a real epithelium. The one-variable experiment that would decide
that is this rig with the driver swapped and everything else held at these certified values.

Imports 05b rather than copying it: test_05b_plaque.py belongs to the sheet session.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_05b_plaque as B                                              # noqa: E402


def main():
    def arg(flag, cast, default):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 401)
    name = arg("--name", str, "06_spheroid_bm")
    d = os.path.join(B.LOG, name)
    os.makedirs(d, exist_ok=True)

    T = 2.0e-3
    P = dict(subdiv=4, subdiv_epi=3, E=400.0, thickness=T, nu=0.3, kn=5.0, xi=0.0,
             l0=0.3 * T, zeta=20.0, s_target=1.0, k_drive=50.0, dev=dev)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 160))).astype(int).tolist())

    rig = B.Rig05b(**P)
    print(f"[{name}] {rig.sheet.n} bm nodes / {rig.x_epi.shape[0]} epi vertices (DRIVEN ICOSPHERE, "
          f"not the vertex model) / {rig.n_plaque} plaques", flush=True)
    kept, reached = B.run(rig, frames, keep=keep, label=name)
    if reached != frames:
        print(f"[{name}] diverged at {reached}/{frames} -- no movie", flush=True)
        return

    s_hi = float(np.percentile(np.concatenate([k[2] for k in kept[::4]]), 99))
    B.render(kept, rig.sheet.Fc.cpu().numpy(), rig.F_epi.cpu().numpy(), d, name, s_hi)
    print(f"[{name}] movie -> {os.path.join(d, 'movie.mp4')}", flush=True)


if __name__ == "__main__":
    main()
