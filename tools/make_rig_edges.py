"""Write the three edge files the ctRNN + eye rig reads, from a declared law and a seed.

    python tools/make_rig_edges.py                 # report what is on disk, write nothing
    python tools/make_rig_edges.py --write         # write, refusing to clobber
    python tools/make_rig_edges.py --write --force --tag v2 --rec-gain 1.0

WHY THIS EXISTS. `config/neural/ctrnn_eyeG_rig.yaml` names three `edges_file:` npz's -- W_in, the
recurrent matrix and W_out -- and nothing in the repository made them. They were inputs with no
provenance: a fit could be re-run but not rebuilt, and "the initial weights" was a claim no one
could check. This file is the missing half of the spec.

THE STRUCTURE IS ALL-TO-ALL AND IS NOT A CHOICE HERE. The rig has no measured connectome: 2
retina cells onto 64 neurons (128 edges), 64 neurons onto each other with no self-loop (64*63 =
4032), 64 neurons onto 6 muscles (384). Every pair exists and training decides the weight, which
is what makes this a rig and not a biological model -- the point of it is the TRAINING PATH, and
a sparsity pattern would be one more thing to be wrong about. `src/plexus/learnables/` is where a
measured connectome enters, through a substitution that must keep the relation it was given.

THE INITIAL RECURRENT WEIGHT IS ZERO IN THE FILES ON DISK, and that is worth knowing rather than
inheriting. A circuit with W = 0 has no recurrence at epoch 0: the neurons only leak, and every
bit of the dynamics has to be built by the optimiser from a flat start. It is not a symmetric
saddle -- W_in and W_out are non-zero, so the gradient on W is non-zero from the first step, and
the recorded fit did reach 0.043 of target variance from there. But the standard ctRNN start is
W ~ N(0, g^2/N) with g near 1, which begins on the edge of chaos rather than at a fixed point,
and `--rec-gain` is here so the two can be compared rather than assumed.

The defaults reproduce the LAW of the files the recorded runs used, not their values -- the
original draws were made without a recorded seed, which is the defect this file closes. A new
`--tag` writes new files and leaves the old ones alone.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from plexus.paths import graphs_data_path

N_RETINA, N_NEURON, N_MUSCLE = 2, 64, 6

# The uniform half-widths the files on disk were drawn with, read back off them: W_in has
# |w| <= 0.35355 = 1/sqrt(8) and W_out |w| <= 0.125 = 1/sqrt(64) = 1/sqrt(n_neuron), the usual
# "one over the square root of the fan-in" so that a neuron's summed input has unit-ish scale
# whatever its in-degree.
WIN_SCALE = 1.0 / np.sqrt(8.0)
WOUT_SCALE = 1.0 / np.sqrt(N_NEURON)


def _dense(n_pre: int, n_post: int, self_loops: bool = True) -> np.ndarray:
    """Every (pre, post) pair, post-major -- the order the files on disk are in."""
    pre, post = [], []
    for j in range(n_post):
        for i in range(n_pre):
            if not self_loops and i == j:
                continue
            pre.append(i); post.append(j)
    return np.asarray([pre, post], dtype=np.int64)


def build(seed: int = 0, rec_gain: float = 0.0) -> dict:
    """The three edge sets. `rec_gain` is g in W ~ N(0, g^2/N); g = 0 gives the zero start."""
    rng = np.random.default_rng(seed)
    win = _dense(N_RETINA, N_NEURON)
    rec = _dense(N_NEURON, N_NEURON, self_loops=False)
    wout = _dense(N_NEURON, N_MUSCLE)
    return {
        "win": (win, rng.uniform(-WIN_SCALE, WIN_SCALE, win.shape[1])),
        # g / sqrt(N) IS THE SCALE THAT MATTERS, not the variance: the spectral radius of a random
        # N x N matrix with entries of standard deviation s is s*sqrt(N), so s = g/sqrt(N) puts it
        # at g -- g = 1 is the edge of chaos and g = 0 is no recurrence at all.
        "rec": (rec, rng.normal(0.0, rec_gain / np.sqrt(N_NEURON), rec.shape[1])
                     if rec_gain > 0 else np.zeros(rec.shape[1])),
        "wout": (wout, rng.uniform(-WOUT_SCALE, WOUT_SCALE, wout.shape[1])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the files (default: report only)")
    ap.add_argument("--force", action="store_true", help="overwrite files that already exist")
    ap.add_argument("--tag", default="", help="suffix, e.g. --tag v2 -> ctrnn_eyeG_rig_v2_win.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rec-gain", type=float, default=0.0,
                    help="g in W ~ N(0, g^2/N); 0 reproduces the zero recurrent start on disk")
    a = ap.parse_args()

    sets = build(a.seed, a.rec_gain)
    tag = f"_{a.tag}" if a.tag else ""
    base = graphs_data_path("neural")
    os.makedirs(base, exist_ok=True)
    for key, (ei, w) in sets.items():
        p = os.path.join(base, f"ctrnn_eyeG_rig{tag}_{key}.npz")
        here = f"{ei.shape[1]:5d} edges  sd {w.std():.4f}  |max| {np.abs(w).max():.4f}"
        if not a.write:
            old = np.load(p) if os.path.exists(p) else None
            was = (f"  (on disk: sd {old['weights'].std():.4f})" if old is not None
                   else "  (absent)")
            print(f"[dry] {os.path.basename(p):32s} {here}{was}")
            continue
        if os.path.exists(p) and not a.force:
            raise SystemExit(
                f"{p} exists. It is an INPUT TO RECORDED FITS, and overwriting it would leave "
                f"every logged result describing weights that are no longer there. Pass --tag to "
                f"write new files, or --force if you mean it.")
        np.savez(p, edge_index=ei, weights=w.astype(np.float32))
        print(f"[write] {os.path.basename(p):32s} {here}")
    if not a.write:
        print("nothing written; pass --write")


if __name__ == "__main__":
    main()
