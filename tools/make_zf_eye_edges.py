"""Write the three edge files that couple the 285-cell oculomotor pool to the eye plant.

    python tools/make_zf_eye_edges.py             # report, write nothing
    python tools/make_zf_eye_edges.py --write

WHAT THIS ADDS TO THE CONNECTOME. `neural/zebrafish_om_285_edges.npz` is the measured thing: 5,013
synapses among 285 cells, already 100% consistent with the type table's Dale assignment (checked:
sign(w) == sign of the presynaptic type's `sign: E|I` on every edge). It has no input and no
output, because a connectome does not have those -- they are a claim about what the circuit is
FOR. The three files here are that claim, and they are the reference's eq:input-map and
eq:output-map (`train_zebra_eyeG.py` on connectome-gnn's feat/oculomotor):

    zf_eyeG_285_win.npz    2 retina cells -> the 41 AF5 cells, all-to-all (82 edges)
    zf_eyeG_285_lr.npz     the 92 AMN cells -> muscle LR   (lateral rectus, muscle index 0)
    zf_eyeG_285_mr.npz     the 35 AIN cells -> muscle MR   (medial rectus, muscle index 2)

ONLY TWO OF SIX MUSCLES ARE REACHABLE FROM THIS POOL, and that is the scientific content rather
than a shortcut. SR, IR, SO and IO are driven by OMN, which is not among these 285 cells, so there
is no cell in the pool that could drive them. The reference holds those four at EXACTLY zero drive
-- not softplus(0) = 0.693, which would assert a tonic contraction that does not exist -- and the
Plexus spec gets the same by running one `readout` per reachable muscle under `at: muscle[type=..]`
and leaving the other four at their seeded zero. So the controller can command horizontal gaze and
NOTHING ELSE: vertical gaze and torsion follow only through whatever `muscle_pose_map`'s cross
terms put there.

TWO READOUTS AND NOT ONE, for the same reason the reference's W_out is a (2, N) matrix masked so
that row LR sees only AMN columns and row MR only AIN. Here the mask IS the edge set: an AMN cell
has no edge to MR, so it cannot drive it, and that is a structural fact of the spec rather than a
zero someone has to keep re-imposing.

The cell-type index ranges are read from the spec's own `types:` table in declared order, because
`type_layout: ordered` is what assigns them -- so this file cannot drift from the spec it serves.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

from plexus.paths import graphs_data_path

SPEC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "config", "neural", "zebrafish_om_285.yaml")
N_RETINA = 2
# THE TASK RIG'S SENSORY BANK IS WIDER THAN THE EYE RIG'S AND FIXED. A t-task corpus has one
# stimulus channel, and a teacher-varying grid adds a one-hot of the condition cell on top --
# six more for `t3_lowpass_order`, so seven lines at the widest. Eight covers every task in the
# battery, and one that uses fewer simply leaves the spare lines at the zero they were seeded to.
# One spec then serves the whole battery instead of one spec per input width.
N_SENSOR = 8
MUSCLE_INDEX = {"LR": 0, "SR": 1, "MR": 2, "IR": 3, "SO": 4, "IO": 5}   # muscle_ops.MUSCLES order


def type_ranges(spec_path=SPEC) -> dict:
    """{type name: (first index, last index + 1)} over the 285 cells, in declared order."""
    s = yaml.safe_load(open(spec_path))
    out, i = {}, 0
    for name, t in s["sets"]["neuron"]["types"].items():
        out[name] = (i, i + int(t["count"]))
        i += int(t["count"])
    return out


def _all_to_all(pre_idx, post_idx) -> np.ndarray:
    """Every (pre, post) pair, post-major."""
    pre, post = np.meshgrid(np.asarray(pre_idx), np.asarray(post_idx), indexing="xy")
    return np.stack([pre.ravel(), post.ravel()]).astype(np.int64)


def build(seed: int = 0, spec_path=SPEC) -> dict:
    """The three maps. Weight draws follow `torch.nn.Linear`'s default, U(-1/sqrt(fan_in), +),
    which is what the reference's `nn.Linear(2, N)` and `nn.Linear(N, 2)` start from."""
    rng = np.random.default_rng(seed)
    r = type_ranges(spec_path)
    af5 = np.arange(r["AF5_ipsi"][0], r["AF5_contra"][1])       # the two AF5 types are contiguous
    amn = np.arange(*r["AMN"])
    ain = np.arange(*r["AIN"])

    win = _all_to_all(np.arange(N_RETINA), af5)
    lr = _all_to_all(amn, [MUSCLE_INDEX["LR"]])
    mr = _all_to_all(ain, [MUSCLE_INDEX["MR"]])
    # fan-in is the number of SENDERS reaching one receiver, which for an all-to-all map is the
    # size of the presynaptic set -- 2 retina cells into a neuron, 92 AMN cells into LR.
    # the task rig: a wider sensory bank onto the same AF5 cells, and a LINEAR readout taking
    # every output cell (AMN and AIN together) onto one signed scalar -- a task target is signed,
    # where a muscle pulls or does nothing, so there is no rectifier and no per-muscle split.
    out_cells = np.concatenate([amn, ain])
    tin = _all_to_all(np.arange(N_SENSOR), af5)
    tout = _all_to_all(out_cells, [0])
    return {
        "win": (win, rng.uniform(-1, 1, win.shape[1]) / np.sqrt(N_RETINA) * 0.5),
        "task_in": (tin, rng.uniform(-1, 1, tin.shape[1]) / np.sqrt(N_SENSOR) * 0.5),
        "task_out": (tout, rng.uniform(-1, 1, tout.shape[1]) / np.sqrt(len(out_cells))),
        "lr": (lr, rng.uniform(-1, 1, lr.shape[1]) / np.sqrt(len(amn))),
        "mr": (mr, rng.uniform(-1, 1, mr.shape[1]) / np.sqrt(len(ain))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    r = type_ranges()
    print("[cells] " + "  ".join(f"{k} [{v[0]}, {v[1]})" for k, v in r.items()))
    base = graphs_data_path("neural")
    os.makedirs(base, exist_ok=True)
    for key, (ei, w) in build(a.seed).items():
        p = os.path.join(base, f"zf_eyeG_285_{key}.npz")
        line = (f"{os.path.basename(p):26s} {ei.shape[1]:5d} edges  "
                f"pre {ei[0].min()}..{ei[0].max()}  post {ei[1].min()}..{ei[1].max()}  "
                f"|w|max {np.abs(w).max():.4f}")
        if not a.write:
            print(f"[dry] {line}"); continue
        if os.path.exists(p) and not a.force:
            raise SystemExit(f"{p} exists; pass --force to overwrite.")
        np.savez(p, edge_index=ei, weights=w.astype(np.float32))
        print(f"[write] {line}")
    if not a.write:
        print("nothing written; pass --write")


if __name__ == "__main__":
    main()
