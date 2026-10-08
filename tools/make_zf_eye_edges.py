"""Write the three edge files that couple the 285-cell oculomotor pool to the eye plant.

    python tools/make_zf_eye_edges.py             # report, write nothing
    python tools/make_zf_eye_edges.py --write
    python tools/make_zf_eye_edges.py --eyes 2 --write      # the two-eye maps (config/neural/zf_eyeG2_285.yaml)

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


# ============================================================================== two eyes
SPEC2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "neural", "zf_eyeG2_285.yaml")
EM_REGION = ("neural_regions", "zf_oculomotor_285")


def sides() -> np.ndarray:
    """Each model row's side, 'left' or 'right', from the EM annotation (Cedric, 2026-10-08: "a two-eye system").

    `neural/zebrafish_om_285_edges.npz` and the EM files (`neural_regions/zf_oculomotor_285/region.npz`,
    `neurons.npz`: type, hemi, body id per cell) hold THE SAME GRAPH UNDER A RELABELLING of the cells -- not the same
    order. The relabelling is recovered here as a graph isomorphism within the type blocks: each cell's fingerprint
    (type, out-degree, in-degree, its targets' and sources' types) is unique for 279 of the 285; the 3 pairs left
    are settled by requiring every one of the 5,013 edges to land on an EM edge. Two of those pairs (AMN) are
    automorphic -- either assignment is an isomorphism -- and are taken in the order that keeps each type's sides
    contiguous, which is the order the rows were written in (every type: its left cells, then its right)."""
    import itertools
    from collections import Counter, defaultdict
    e = np.load(graphs_data_path("neural", "zebrafish_om_285_edges.npz"))["edge_index"]
    r = np.load(graphs_data_path(*EM_REGION, "region.npz"), allow_pickle=True)["edge_index"]
    nz = np.load(graphs_data_path(*EM_REGION, "neurons.npz"), allow_pickle=True)
    t_em, hemi = nz["type_id"], nz["hemi"]
    t_mod = np.repeat(np.arange(len(nz["type_names"])), [int((t_em == k).sum()) for k in range(len(nz["type_names"]))])
    n = len(t_mod)

    def fp(ei, t):
        pre, post = ei
        return [(int(t[i]), int((pre == i).sum()), int((post == i).sum()),
                 tuple(sorted(Counter(t[post[pre == i]]).items())), tuple(sorted(Counter(t[pre[post == i]]).items())))
                for i in range(n)]
    ga, gb = defaultdict(list), defaultdict(list)
    for i, f in enumerate(fp(e, t_mod)):
        ga[f].append(i)
    for i, f in enumerate(fp(r, t_em)):
        gb[f].append(i)
    if Counter({k: len(v) for k, v in ga.items()}) != Counter({k: len(v) for k, v in gb.items()}):
        raise SystemExit("the model's synapse graph is not the EM graph relabelled -- no side can be read")
    base = {ga[f][0]: gb[f][0] for f in ga if len(ga[f]) == 1}
    amb = [(ga[f], gb[f]) for f in ga if len(ga[f]) > 1]
    em = set(map(tuple, r.T))
    found = []
    for combo in itertools.product(*[list(itertools.permutations(b_)) for _, b_ in amb]):
        p_ = dict(base)
        for (a_, _), perm in zip(amb, combo):
            p_.update(zip(a_, perm))
        if set((p_[a], p_[b]) for a, b in e.T) == em:
            found.append(np.array([p_[i] for i in range(n)]))
    if not found:
        raise SystemExit("no relabelling maps every model edge onto an EM edge")
    for perm in found:                                   # the side-contiguous one (see the docstring)
        h = hemi[perm]
        if all(np.all(np.diff((h[t_mod == k] == "right").astype(int)) >= 0) for k in range(int(t_mod.max()) + 1)):
            return h.astype(str)
    raise SystemExit("no relabelling keeps every type's sides contiguous")


def build2(seed: int = 0, spec_path=SPEC2) -> dict:
    """The two-eye maps on `zf_eyeG2_285` (types split by side): eye 0 the LEFT eye, eye 1 the RIGHT, muscle index
    6 x eye + MUSCLE_INDEX. The abducens circuit: AMN_L -> the left LR, AMN_R -> the right LR; the internuclear
    neurons cross -- AIN_L -> the right MR, AIN_R -> the left MR (through the MLF to the contralateral oculomotor
    nucleus, folded here into one edge). The retinas cross too: retina 0 (left eye) -> the right AF5 cells,
    retina 1 (right eye) -> the left ones."""
    rng = np.random.default_rng(seed)
    r = type_ranges(spec_path)
    # THE TWO RETINAS, CROSSED (Cedric, 2026-10-08: "fix the input neurons for the two eyes"): retina cell 0 is the
    # LEFT eye, cell 1 the RIGHT; the larval zebrafish retina projects only contralaterally, so the left eye reaches
    # the right AF5 cells and the right eye the left ones -- 41 edges, one per AF5 cell
    af5_R = np.concatenate([np.arange(*r[k]) for k in ("AF5_ipsi_R", "AF5_contra_R")])
    af5_L = np.concatenate([np.arange(*r[k]) for k in ("AF5_ipsi_L", "AF5_contra_L")])
    win = np.concatenate([_all_to_all([0], af5_R), _all_to_all([1], af5_L)], 1)
    m = lambda eye, mu: 6 * eye + MUSCLE_INDEX[mu]                       # noqa: E731
    lr = np.concatenate([_all_to_all(np.arange(*r["AMN_L"]), [m(0, "LR")]),
                         _all_to_all(np.arange(*r["AMN_R"]), [m(1, "LR")])], 1)
    mr = np.concatenate([_all_to_all(np.arange(*r["AIN_L"]), [m(1, "MR")]),
                         _all_to_all(np.arange(*r["AIN_R"]), [m(0, "MR")])], 1)
    fan = lambda k: np.sqrt(len(np.arange(*r[k])))                       # noqa: E731
    w_lr = np.concatenate([rng.uniform(-1, 1, r["AMN_L"][1] - r["AMN_L"][0]) / fan("AMN_L"),
                           rng.uniform(-1, 1, r["AMN_R"][1] - r["AMN_R"][0]) / fan("AMN_R")])
    w_mr = np.concatenate([rng.uniform(-1, 1, r["AIN_L"][1] - r["AIN_L"][0]) / fan("AIN_L"),
                           rng.uniform(-1, 1, r["AIN_R"][1] - r["AIN_R"][0]) / fan("AIN_R")])
    return {"win": (win, rng.uniform(-1, 1, win.shape[1]) * 0.5), "lr": (lr, w_lr), "mr": (mr, w_mr)}   # fan-in 1


def check_sides(spec_path=SPEC2):
    """The two-eye spec's _L / _R type blocks against the EM sides, row by row."""
    h, r = sides(), type_ranges(spec_path)
    bad = [k for k, (a_, b_) in r.items() if set(h[a_:b_]) != {"left" if k.endswith("_L") else "right"}]
    if bad:
        raise SystemExit(f"{spec_path}: these types' rows are not all on their side: {bad}")
    print(f"[sides] {os.path.basename(spec_path)}: every _L / _R block is on its EM side ({len(h)} cells)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eyes", type=int, default=1, choices=[1, 2],
                    help="2: the two-eye maps zf_eyeG2_285_{win,lr,mr}.npz on config/neural/zf_eyeG2_285.yaml")
    a = ap.parse_args()
    if a.eyes == 2:
        check_sides()
        base = graphs_data_path("neural")
        for key, (ei, w) in build2(a.seed).items():
            p = os.path.join(base, f"zf_eyeG2_285_{key}.npz")
            line = (f"{os.path.basename(p):26s} {ei.shape[1]:5d} edges  pre {ei[0].min()}..{ei[0].max()}  "
                    f"post {sorted(set(ei[1].tolist()))[:12]}  |w|max {np.abs(w).max():.4f}")
            if not a.write:
                print(f"[dry] {line}"); continue
            if os.path.exists(p) and not a.force:
                raise SystemExit(f"{p} exists; pass --force to overwrite.")
            np.savez(p, edge_index=ei, weights=w.astype(np.float32))
            print(f"[write] {line}")
        return

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
