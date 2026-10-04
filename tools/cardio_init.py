#!/usr/bin/env python
"""The STARTING hypothesis of a cardiac-sheet fit, read off its recording.

    PYTHONPATH=src python tools/cardio_init.py healthy --beats 1,2,3 --modes 2 --seed 0
    -> graphs_data/cardio/<specimen>_init.npz

Every learnable starts at the value the model states, and for a sheet fitted from motion alone
the model states what the recording offers before any fit: each cell's contraction AXIS and its
peak shortening (`fibre_init`, at the frame of its largest deformation in the first fitted beat),
the sheet's 4-number CLOCK fitted to the mean shortening curve (`clock_init`), a uniform stiffness
log(80), and zero for every per-cell departure (delay, across-fibre strain, time-course scales,
adhesion). The two temporal modes start as small seeded random walks and their per-cell weights
as small seeded noise -- prototype/cardio_mpm/strain/fit.py's rules, draw for draw.

The file has the keys `seed_state_from_file` (per-cell blocks) and `active_strain`'s `fit:`
(clock, psi) read, so a model names it twice and starts there.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from plexus.paths import graphs_data_path            # noqa: E402
from plexus.tasks import recording as R               # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("specimen")
ap.add_argument("--beats", default="1,2,3")
ap.add_argument("--modes", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--youngs", type=float, default=80.0)
a = ap.parse_args()
rec = R.load(a.specimen)
wins = [R.beat_window(rec, int(b)) for b in a.beats.split(",")]
A_rec, _u = R.window_affine(rec, wins[0])
phi, amp, _ = R.fibre_init(A_rec)
ck = R.clock_init(A_rec)
C, K, T = rec["n_cells"], a.modes, max(len(w["frames"]) for w in wins)
gen = torch.Generator().manual_seed(a.seed)
psi = 0.02 * torch.randn((max(K, 1), max(T, 2)), generator=gen).cumsum(1)
psi[:, 0] = 0.0
amode = 0.1 * torch.randn((C, max(K, 1)), generator=gen)
z = dict(g=np.full(C, float(amp.median()), np.float32), phi=phi.numpy().astype(np.float32),
         logE=np.full(C, math.log(a.youngs), np.float32),
         clock=np.array([ck["t0"], math.log(ck["tau_r"]), math.log(ck["dur"]), math.log(ck["tau_d"])],
                        np.float32),
         psi=psi.numpy().astype(np.float32), amode=amode.numpy().astype(np.float32),
         logkappa0=np.float32(0.0), n_modes=np.int64(K), clock_mode=np.array("sigmoid"))
for k in ("delay", "g2", "logtau", "logtr", "logdur", "logkappa"):
    z[k] = np.zeros(C, np.float32)
out = graphs_data_path("cardio", f"{a.specimen}_init.npz")
np.savez(out, **z)
print(f"{a.specimen}: {C} cells, g0 {float(amp.median()):.4f} (median peak shortening of beat "
      f"{wins[0]['k']}), clock t0 {ck['t0']:.2f} rise {ck['tau_r']:.2f} plateau {ck['dur']:.2f} "
      f"decay {ck['tau_d']:.2f} frames, psi {tuple(psi.shape)} -> {out}")
