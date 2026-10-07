"""exp17: THE BALANCED INPUT MASK (Cedric, 2026-10-06: "the visual input neurons equally distributed left and right; at
least 50 neurons per block, but 0 in open loop and dark"). Local, model-free.

From a mask npz of tools/exp17_stim_coherence.py (its `coherence_by_feature` [N, F_live] and `features`, the 13
features that change within their condition), per BLOCK (the condition each feature belongs to):
  the block's coherence of a neuron   its highest band coherence with one of the block's features
  the block's count                   its natural count -- the neurons whose block coherence clears the global cut that
                                      keeps the top `--top` of the neurons by best coherence (the unbalanced mask) --
                                      and at least `--min` (50); 0 for a block with no changing feature (open loop, dark)
  the selection                       half of the count on each side of the midline (the head-up view, split at the
                                      neurons' median), each side its neurons with the highest block coherence
  the per-feature mask                a selected neuron reads the block's features whose coherence clears the block's
                                      own cut on its side (its best feature of the block always)
The mask is the union over the blocks (a neuron may be picked by two). Writes graphs_data/zebrafish/<out>: mask,
mask_by_input [N, all stimulus columns], block_counts.

    PYTHONPATH=src:tools python tools/exp17_balanced_mask.py --source input_mask_destripe_vis_coh20.npz --top 0.2 \
        --out input_mask_destripe_bal20.npz
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]


def build(source, top, out, n_min=50, recording="zapbench_destripe"):
    from plexus.paths import graphs_data_path
    from exp17_ablation import _brain_view
    z = np.load(graphs_data_path("zebrafish", f"{recording}_recording.npz"))
    U, off, names = z["stimulus"], z["offsets"], [str(x) for x in z["names"]]
    src = np.load(graphs_data_path("zebrafish", source))
    C, live = src["coherence_by_feature"], np.asarray(src["features"])
    N = len(C)
    fblock = np.array([next(k for k in range(len(off) - 1) if np.abs(U[off[k]:off[k + 1], j]).max() > 0) for j in live])
    P = _brain_view(z["pos_um"].astype(np.float64))
    left = P[:, 1] < np.median(P[:, 1])                       # the movies' head-up view: its left half
    best = C.max(1)
    cut = np.quantile(best, 1 - top)                          # the unbalanced mask's global cut
    mask = np.zeros(N, bool)
    mbi = np.zeros((N, U.shape[1]), np.float32)
    counts = {}
    for b in range(len(names)):
        fb = np.where(fblock == b)[0]
        if not len(fb):                                       # no changing feature: no input neuron (open loop, dark)
            counts[names[b]] = {"total": 0, "left": 0, "right": 0}
            continue
        cb = C[:, fb].max(1)
        n_b = max(int((cb >= cut).sum()), n_min)
        tot = {"total": 0, "left": 0, "right": 0}
        for side, m_ in (("left", left), ("right", ~left)):
            ids = np.where(m_)[0]
            pick = ids[np.argsort(cb[ids])[::-1][:n_b // 2 + (n_b % 2 if side == "left" else 0)]]
            mask[pick] = True
            cut_b = cb[pick].min()                            # the block's own cut on this side
            for j in fb:
                mbi[pick, live[j]] = np.maximum(mbi[pick, live[j]], (C[pick, j] >= cut_b).astype(np.float32))
            bj = live[fb[np.argmax(C[pick][:, fb], 1)]]       # each picked neuron's best feature of the block, always
            mbi[pick, bj] = 1.0
            tot[side] = int(len(pick))
            tot["total"] += int(len(pick))
        counts[names[b]] = tot
    np.savez(graphs_data_path("zebrafish", out), mask=mask.astype(np.float32), mask_by_input=mbi,
             block_counts=json.dumps(counts), source=source, top=top, n_min=n_min)
    nf = (mbi[mask] > 0).sum(1)
    print(f"[balanced mask] {out}: {int(mask.sum()):,} input neurons ({100 * mask.mean():.1f} %), left {int((mask & left).sum()):,}"
          f" / right {int((mask & ~left).sum()):,}; features per input neuron {nf.mean():.2f}")
    for k, v in counts.items():
        print(f"   {k:10s} total {v['total']:6d}  left {v['left']:6d}  right {v['right']:6d}")
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--top", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min", type=int, default=50)
    a = ap.parse_args()
    build(a.source, a.top, a.out, a.min)
