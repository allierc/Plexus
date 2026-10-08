"""The ARTR / head-direction twin of the oculomotor eye rig (Cedric, 2026-10-08: "the twin prototyping for ARTR, restricted
to angle integration"): the 917-cell fish-2 HD / IPN circuit of Cedric's note "Connectome-constrained self-motion
integration in the larval zebrafish hindbrain" (connectome-gnn-cx docs/zebrafish.tex), built as its code builds it
(connectome-gnn-cx circuit zebrafish_HD_IPN_917_artr_pt1, config zebrafish_hd_si_ipn_917_v1_selfmotion_rotation;
audited 2026-10-08) and written as Plexus edge files and a model spec.

    python tools/make_zf_hd.py                 # report, write nothing
    python tools/make_zf_hd.py --write         # the CSVs imported, the edge files, config/neural/zf_hd_917.yaml

THE CIRCUIT (917 cells, 30,851 edges, weights the synapse contact areas):
  ring (inhibitory, 700)     dIPN (IPNd*, IPNds*: 408), IPN12 (IPN12_a / _b: 108), IPN-core (IPN28-IPN36: 184) -- the
                             GABAergic r1pi integrator; every outgoing weight negative (the reference's
                             _ZHD_INH_PREFIXES, which its connectome section and code use; its Methods text names only
                             IPNd* / IPNds*). The readout reads all 700, IPN-core included, as the code does.
  afferents (excitatory, 217) ARTR (RIPN01/02/03_a/03_b: 76), pt-IPN1 (51), motor_efferent (RIPN11/12_a/12_c: 65),
                             other (RIPN05/12_b: 25). For angle integration only the ARTR is driven.
  scale                      J [post, pre] (signed contact area) scaled so its largest eigenvalue REAL PART is 0.9 (the
                             code's normalisation; the spectral radius is then ~0.98); the 3 self-edges dropped (the
                             reference zeroes the diagonal).
  bias                       the reference's b sits inside (1/tau)(...): its b = 1 start is 10 here, where the bias
                             adds to the derivative directly.
  input                      omega (deg/s) -> ARTR only, SIGN-LOCKED: -> ARTR_L through edges bounded <= 0, -> ARTR_R
                             through edges bounded >= 0 (the reference ties each side to one scalar -softplus(v_L) /
                             +softplus(v_R), initialised at 0.01 per deg/s; here each edge has its own magnitude, the sign
                             locked by the training spec's `bounds`). The starting heading's cue (cos, sin theta0, frame
                             0 only) reaches every cell, the ring included (free weights, randn / 100), as in the code.
  readout                    the 700 ring cells -> (cos theta, sin theta), all-to-all, Kaiming-uniform start.
Types are (group, side) blocks: ARTR, ptIPN1, motor, other, dIPN, IPN12, IPNcore, each _L then _R; within a type the
cells are sorted by their soma's angle on the IPN ring, so a travelling bump runs down the rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import numpy as np

from plexus.paths import graphs_data_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "/workspace/connectome-gnn-cx/figures/zebrafish/zebrafish_connectome_HD_IPN_917"
DATA = ("neural", "zf_hd_ipn917")
ORDER = ("ARTR", "ptIPN1", "motor", "other", "dIPN", "IPN12", "IPNcore")
RING = ("dIPN", "IPN12", "IPNcore")
TARGET_RE = 0.9


def group(t: str) -> str:
    if t in ("RIPN01", "RIPN02", "RIPN03_a", "RIPN03_b"):
        return "ARTR"
    if t.startswith("pt-IPN1"):
        return "ptIPN1"
    if t in ("RIPN11", "RIPN12_a", "RIPN12_c"):
        return "motor"
    if t.startswith("RIPN"):
        return "other"
    if t.startswith("IPNd"):                      # IPNd* and IPNds*
        return "dIPN"
    if t.startswith("IPN12"):
        return "IPN12"
    if t.startswith("IPN"):                       # IPN28 .. IPN36
        return "IPNcore"
    raise ValueError(f"cell type {t!r} has no group")


def _dir():
    d = graphs_data_path(*DATA)
    return d if os.path.exists(os.path.join(d, "neurons.csv")) else SRC


def build(seed: int = 0) -> dict:
    import pandas as pd
    d = _dir()
    n = pd.read_csv(os.path.join(d, "neurons.csv"))
    c = pd.read_csv(os.path.join(d, "connections.csv"))
    n["group"] = n["type"].astype(str).map(group)
    n["block"] = n["group"] + "_" + n["side"].astype(str)
    rank = {f"{g}_{s}": 2 * k + (s == "R") for k, g in enumerate(ORDER) for s in "LR"}
    n = n.assign(r_=n["block"].map(rank)).sort_values(["r_", "angle", "bodyId"], kind="stable").reset_index(drop=True)
    N = len(n)
    idx = {b: i for i, b in enumerate(n["bodyId"])}
    pre = c["bodyId_pre"].map(idx).values
    post = c["bodyId_post"].map(idx).values
    area = c["weight"].values.astype(np.float64)
    inh = n["group"].isin(RING).values
    sgn = np.where(inh[pre], -1.0, 1.0)
    J = np.zeros((N, N))
    np.add.at(J, (post, pre), sgn * area)
    scale = TARGET_RE / np.max(np.linalg.eigvals(J).real)          # the largest real part -> 0.9
    off = post != pre                                                # the self-edges, dropped
    w = np.abs(J[post[off], pre[off]]) * scale
    blocks, first, i = [], {}, 0
    for b, g in n.groupby("block", sort=False):
        grp = b.rsplit("_", 1)[0]
        blocks.append((b, len(g), "I" if grp in RING else "E",
                       "afferent" if grp == "ARTR" else "output" if grp in RING else "recurrent"))
        first[b] = (i, i + len(g))
        i += len(g)
    rng = np.random.default_rng(seed)
    aL, aR = np.arange(*first["ARTR_L"]), np.arange(*first["ARTR_R"])
    ring = np.concatenate([np.arange(*first[f"{g}_{s}"]) for g in RING for s in "LR"])
    allc = np.arange(N)
    e = lambda pre_, post_: np.stack(np.meshgrid(pre_, post_, indexing="xy")).reshape(2, -1).astype(np.int64)   # noqa: E731
    out_e = e(ring, [0, 1])
    return {"n": n, "N": N, "blocks": blocks, "first": first, "scale": scale,
            "max_re": float(np.max(np.linalg.eigvals(J * scale).real)), "rho": float(np.max(np.abs(np.linalg.eigvals(J * scale)))),
            "files": {
                "edges": (np.stack([pre[off], post[off]]).astype(np.int64), w.astype(np.float32)),
                "omegaL": (e([0], aL), np.full(len(aL), -0.01, np.float32)),
                "omegaR": (e([0], aR), np.full(len(aR), 0.01, np.float32)),
                "cue": (e([1, 2], allc), (rng.standard_normal(2 * N) / 100).astype(np.float32)),
                "out": (out_e, rng.uniform(-1, 1, out_e.shape[1]).astype(np.float32) / np.sqrt(len(ring))),
            },
            "n_inh": int(inh.sum()), "n_edges": int(off.sum())}


def spec_yaml(B) -> str:
    types = "".join(f"      {b}:\n        count: {k}\n        sign: {sg}\n        role: {role}\n"
                    f"        p: [10.0, 0.0, 10.0, 0.0, 1.0, 0.0]\n" for b, k, sg, role in B["blocks"])
    f = lambda k: f"neural/zf_hd_{B['N']}_{k}.npz"                     # noqa: E731
    return f"""general:
  name: zf_hd_{B['N']}
  seed: 0
  n_frames: 1000
  dt: 0.01
  dim: 3
  world: [1.0, 1.0, 1.0]
  boundary: wall
sets:
  brain: {{n: 1}}
  selfmotion:
    parent: brain
    per_parent: 3
    state:
      signal: {{width: 1, integration: none, boundary: free}}
  neuron:
    parent: brain
    per_parent: {B['N']}
    type_layout: ordered
    types:
{types}    state:
      pos: {{width: 3, role: geometry, integration: none, boundary: world}}
      voltage: {{width: 1, role: coordinate, integration: first_order, boundary: free}}
      omega: {{width: 1, role: modulation, integration: none, boundary: free, record: false}}
      bias: {{width: 1, integration: none, boundary: free, record: false}}
  heading:
    parent: brain
    per_parent: 2
    type_layout: ordered
    types:
      cos: {{count: 1}}
      sin: {{count: 1}}
    state:
      value: {{width: 1, integration: none, boundary: free}}
      bias: {{width: 1, integration: none, boundary: free, record: false}}
  synapse: {{parent: brain, edge_set: true, entity: connection, pre: neuron, post: neuron, edges_file: {f('edges')}}}
  omega_L: {{parent: brain, edge_set: true, entity: connection, pre: selfmotion, post: neuron, edges_file: {f('omegaL')}}}
  omega_R: {{parent: brain, edge_set: true, entity: connection, pre: selfmotion, post: neuron, edges_file: {f('omegaR')}}}
  cue: {{parent: brain, edge_set: true, entity: connection, pre: selfmotion, post: neuron, edges_file: {f('cue')}}}
  readout: {{parent: brain, edge_set: true, entity: connection, pre: neuron, post: heading, edges_file: {f('out')}}}
fields: {{}}
seed:
- {{op: seed_state_random, at: neuron, block: bias, lo: 10.0, hi: 10.0}}
- {{op: seed_state_random, at: heading, block: bias, lo: 0.0, hi: 0.0}}
operators:
- {{op: project, at: neuron, edge_set: omega_L, block: signal, gain: 10.0}}
- {{op: project, at: neuron, edge_set: omega_R, block: signal, gain: 10.0}}
- {{op: project, at: neuron, edge_set: cue, block: signal, gain: 10.0, bias: bias}}
- {{op: neuron_update, at: neuron, model: leaky_tanh, noise: 0.0}}
- {{op: neuron_signal, at: neuron, model: shared, edge_set: synapse, activation: sigmoid, dale: true}}
- {{op: readout, at: heading, edge_set: readout, block: voltage, into: value, send: sigmoid, bias: bias}}
schedule: [project, project, project, neuron_update, neuron_signal, readout]
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    B = build(a.seed)
    print(f"[hd] {B['N']} cells, {B['n_edges']:,} edges (self-edges dropped), {B['n_inh']} inhibitory; scale "
          f"{B['scale']:.3g} per contact area -> largest real part {B['max_re']:.3f}, spectral radius {B['rho']:.3f}")
    for b, k, sg, role in B["blocks"]:
        print(f"   {b:10s} {k:4d}  {sg}  {role}")
    if not a.write:
        print("nothing written; pass --write")
        return
    dd = graphs_data_path(*DATA)
    os.makedirs(dd, exist_ok=True)
    if _dir() == SRC:                                                # import the circuit once, with its hashes
        prov = {}
        for f_ in ("neurons.csv", "connections.csv"):
            shutil.copy(os.path.join(SRC, f_), os.path.join(dd, f_))
            prov[f_] = hashlib.sha256(open(os.path.join(dd, f_), "rb").read()).hexdigest()
        json.dump({"source": SRC, "what": "fish2 HD/IPN circuit, 917 cells, contact-area weights", "sha256": prov},
                  open(os.path.join(dd, "provenance.json"), "w"), indent=1)
    B["n"][["bodyId", "type", "instance", "side", "angle", "group", "block"]].to_csv(
        os.path.join(dd, f"cells_{B['N']}.csv"), index=False)
    base = graphs_data_path("neural")
    for key, (ei, w) in B["files"].items():
        p = os.path.join(base, f"zf_hd_{B['N']}_{key}.npz")
        if os.path.exists(p) and not a.force:
            raise SystemExit(f"{p} exists; pass --force")
        np.savez(p, edge_index=ei, weights=w)
        print(f"[write] {os.path.basename(p)}: {ei.shape[1]:,} edges")
    sp = os.path.join(ROOT, "config", "neural", f"zf_hd_{B['N']}.yaml")
    if os.path.exists(sp) and not a.force:
        raise SystemExit(f"{sp} exists; pass --force")
    open(sp, "w").write(spec_yaml(B))
    print(f"[write] {sp}")


if __name__ == "__main__":
    main()
