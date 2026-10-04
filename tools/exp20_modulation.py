"""exp20: THE LEARNED MODULATION Omega_i(t) as a movie, exp17's slide 17 (tools/exp17_modulation.py, called unchanged)
on a gut-brain SIREN run: left the recorded dF/F, right Omega per cell, below the brain means.

    PYTHONPATH=src:tools python tools/exp20_modulation.py <run> [<run> ...]

exp17's tool draws with exp17_ablation._brain_view (ZAPBench's orientation) and a 0.914-s frame; both are set here, in
memory, for exp20: every gut-brain recording lies along x with its head at +x (provenance "head", checked on all 24
fish 2026-10-04), drawn mirrored so the head is LEFT; the frame period is the recording's own. Writes, beside exp17's
outputs (<run>/results/<run>_omega.npz, movie_omega.mp4 / .png), experiments/exp20_gutbrain_graphcast/data/omega_<run>.json:
Omega's mean, 5th-95th percentiles over all cell-frames, the range of its brain mean over time, the share below 1.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")


def head_left(pos):
    """exp20's drawing: mirror x (the head at +x drawn LEFT), as trace_recording.load's pos_view."""
    pos = np.asarray(pos, np.float64)
    return np.stack([-pos[:, 0], pos[:, 1], pos[:, 2]], 1)


def patch():
    import exp17_ablation
    exp17_ablation._brain_view = head_left


def render(name, device="cuda:0"):
    import exp17_modulation as M
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    patch()
    rec = TR.load(T.load(name)["task"]["reference"]["trace_recording"])
    M.FRAME_S = float(np.median(np.diff(rec["t_s"])))
    path = M.render(name, device)
    z = np.load(os.path.join(T.out_dir(T.load(name), None), "results", f"{name}_omega.npz"))
    om = z["omega"].astype(np.float32)
    bm = om.mean(1)
    st = {"run": name, "mean": float(om.mean()), "p5": float(np.percentile(om, 5)), "p95": float(np.percentile(om, 95)),
          "brain_mean_min": float(bm.min()), "brain_mean_max": float(bm.max()), "frac_below_1": float((om < 1).mean()),
          "frames": int(om.shape[0]), "cells": int(om.shape[1])}
    json.dump(st, open(os.path.join(EXP, "data", f"omega_{name}.json"), "w"), indent=1)
    return path


if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        render(n_)
