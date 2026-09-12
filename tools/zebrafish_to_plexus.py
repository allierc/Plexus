#!/usr/bin/env python
"""The 285-cell zebrafish oculomotor pool as a Plexus spec -- the same selection, sort and sign
convention as connectome-gnn's `prototype/dot_tracking/zebrafish_circuit.py`, written as a set
of typed neurons and a synapse edge-set with `edges_file:`.

    PYTHONPATH=src python tools/zebrafish_to_plexus.py \\
        --pkl /workspace/connectome-gnn/config/zebrafish/Oculomotor_sortedData_081126.pkl \\
        --out config/neural/zebrafish_om_285.yaml

What it writes: `<out>` and, under graphs_data/neural/, `<name>_edges.npz` (edge_index [2, E]
pre -> post, weights [E] = the synapse size, signed by the presynaptic cell type's Dale sign and
rescaled so the signed matrix has spectral radius `--rho`). The eight cell types, their roles
and signs are the oculomotor note's claims, copied from
`config/zebrafish/zebrafish_om_intg_285_v1.yaml` there; nothing here decides a sign.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

# name -> (role, sign): the circuit owner's claims (zebrafish_om_intg_285_v1.yaml)
CELL_TYPES = [
    ("AF5_ipsi", "afferent", "E"), ("AF5_contra", "afferent", "E"),
    ("INTG_ipsi_m", "recurrent", "E"), ("INTG_ipsi_i", "recurrent", "E"),
    ("INTG_contra_m", "recurrent", "I"), ("INTG_contra_i", "recurrent", "I"),
    ("AMN", "output", "E"), ("AIN", "output", "E"),
]
ROLE_ORDER = ("afferent", "recurrent", "output")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default="/workspace/connectome-gnn/config/zebrafish/Oculomotor_sortedData_081126.pkl")
    ap.add_argument("--weights", default="size", choices=("size", "counts"))
    ap.add_argument("--rho", type=float, default=0.9, help="spectral radius of the signed matrix")
    ap.add_argument("--out", default=os.path.join(REPO, "config", "neural", "zebrafish_om_285.yaml"))
    ap.add_argument("--n-frames", type=int, default=600)
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--tau0", type=float, default=0.1, help="membrane time constant, s (a = 1/tau0)")
    a = ap.parse_args()
    from plexus.paths import graphs_data_path

    d = pickle.load(open(a.pkl, "rb"))
    types = np.asarray(d["Type"]).astype(str); hemi = np.asarray(d["Hemi"]).astype(str)
    A = np.asarray(d[f"adjacency_matrix_{a.weights}"], np.float64)          # pre x post
    names_decl = [n for n, _, _ in CELL_TYPES]
    role_of = {n: r for n, r, _ in CELL_TYPES}; sign_of = {n: s for n, _, s in CELL_TYPES}
    sel = np.where(np.isin(types, names_decl))[0]
    key = [(ROLE_ORDER.index(role_of[types[i]]), names_decl.index(types[i]),
            0 if str(hemi[i]).lower().startswith("l") else 1) for i in sel]
    sel = sel[np.lexsort(([k[2] for k in key], [k[1] for k in key], [k[0] for k in key]))]
    W = A[np.ix_(sel, sel)]                                                   # pre x post, sorted
    nm = types[sel]; N = len(sel)
    sign = np.array([1.0 if sign_of[t] == "E" else -1.0 for t in nm])
    signed = W * sign[:, None]                                                # the pre row's sign
    max_re = float(np.linalg.eigvals(signed.T).real.max())
    scale = a.rho / max_re if max_re > 0 else 1.0
    pre, post = np.nonzero(W)
    w = (signed[pre, post] * scale).astype(np.float32)
    name = os.path.splitext(os.path.basename(a.out))[0]
    npz_rel = os.path.join("neural", f"{name}_edges.npz")
    npz = graphs_data_path(npz_rel)
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    np.savez(npz, edge_index=np.stack([pre, post]).astype(np.int64), weights=w)
    # rows by role then type then hemisphere, on three arcs so the picture reads afferent ->
    # recurrent -> output left to right
    pos = []
    for i, t in enumerate(nm):
        r = ROLE_ORDER.index(role_of[t])
        x0 = (0.15, 0.5, 0.85)[r]
        k = np.sum(np.array([role_of[u] for u in nm[:i]]) == role_of[t])
        tot = np.sum(np.array([role_of[u] for u in nm]) == role_of[t])
        pos.append([round(float(x0 + 0.08 * np.sin(2 * np.pi * k / tot)), 5), round(float(0.1 + 0.8 * k / max(tot - 1, 1)), 5)])
    counts = {t: int(np.sum(nm == t)) for t in names_decl if np.sum(nm == t)}
    tspec = {}
    for t in names_decl:
        if t not in counts:
            continue
        role, s = role_of[t], sign_of[t]
        # tau_i dv = -v + W r + I with tau0 = 0.1 s (train_zebra_eyeG's init): a = 1/tau0 in 1/s,
        # no self-coupling, unit gain; the psi width and threshold are the reference's defaults.
        tspec[t] = {"count": counts[t], "sign": s, "role": role, "p": [1.0 / a.tau0, 0.0, 1.0, 0.0, 1.0, 0.0]}
    spec = {
        "general": {"name": name, "seed": 0, "n_frames": a.n_frames, "dt": a.dt, "dim": 2, "world": 1.0, "boundary": "wall"},
        "sets": {
            "brain": {"n": 1},
            "neuron": {"n": N, "start": pos, "type_layout": "ordered", "types": tspec},
            "synapse": {"parent": "brain", "edge_set": True, "pre": "neuron", "post": "neuron", "edges_file": npz_rel},
        },
        "fields": {"omega": {"frame": "grid", "res": 64, "components": 1}},
        "operators": [
            {"op": "pacemaker", "at": "omega", "period": 240, "duration": 120},
            {"op": "activation_pulse", "at": "omega", "period": 240, "duration": 120, "radius": 0.2, "center": [0.15, 0.5]},
            {"op": "neuron_drive", "at": "neuron[type=AF5_ipsi]", "from": "omega", "gain": 1.0},
            {"op": "neuron_update", "at": "neuron", "model": "leaky_tanh", "noise": 0.0},
            {"op": "neuron_signal", "at": "neuron", "model": "type_pairwise", "edge_set": "synapse", "activation": "tanh"},
        ],
        "schedule": ["pacemaker", "activation_pulse", "neuron_drive", "neuron_update", "neuron_signal"],
        "plotting": {"renderer": "neural_panel", "background": "black",
                     "panel": {"input": ["AF5_ipsi", "AF5_contra"], "output": ["AMN", "AIN"], "kino_frames": 300},
                     "max_frames": 300, "stills": 10, "keep_stills": True},
    }
    with open(a.out, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=None, width=120)
    print(f"[zebrafish] {N} cells {counts}; {len(w):,} synapses; spectral scale {scale:.3g} (rho {a.rho}) -> {a.out}, {npz}")


if __name__ == "__main__":
    main()
