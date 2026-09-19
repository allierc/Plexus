#!/usr/bin/env python
"""Correct the direction of the Platynereis connectome: the adjacency matrix was read transposed.

    python tools/platynereis_fix_direction.py --check     # say what is wrong, change nothing
    python tools/platynereis_fix_direction.py --apply     # swap, keeping the original beside it

WHAT IS WRONG. `graphs_data/platynereis/build_region.py:191-201` reads
`full_connectome_adjacency_matrix.csv` taking each ROW LABEL as the postsynaptic cell and each
COLUMN HEADER as the presynaptic one, and writes `edge_index = (pre, post)` accordingly. The
source matrix has it the other way round, so every edge in `connectome.npz` points backwards and
`n_pre`/`n_post` in `neurons.npz`, which were derived from it, are swapped.

HOW IT IS KNOWN. The source file settles it directly, and three independent checks agree.

  0. THE CSV ITSELF, at
     /workspace/connectome-gnn/graphs_data/remote/candidates/Platynereis_dumerilii_3_day_old_
     nectochaete_larva_annelid_trochophore_lineage_RANK_1_THE_PICK/source/ -- 1,720 x 1,720, the
     corner cell reading "Neurons". ROW `MC` holds the prototroch entries (prototroch_1P 27,
     prototroch_5A 16, prototroch_11-12A 13, ...) and COLUMN `MC` holds inputs from sensory cells
     (doCRunp; SN, hCR2_r; SN, antPUc_1r; SN). MC drives the prototroch and is driven by sensory
     neurons, so the ROW is the PREsynaptic cell. `build_region.py` takes the row as post.

  1. Muscle cells hold 319 edges with muscle in row 0 and ZERO in row 1, and the ciliary band 311
     against 1. Read as written, that is 840 muscle cells and 74 ciliary cells that send synapses
     to the nervous system and receive none. An effector with no postsynaptic sites is not a
     thing.

  2. Sensory neurons come out receiving 1,817 and sending 1,006. Under the swap they send 1,817
     and receive 1,006, which is what a sensory cell does.

  3. THE MC CELL, which is the same fact read back out of the region. There is exactly one cell of type `MC` in the compendium, and the file gives it
     23 edges with all 23 prototroch cells, 340 synapses, with prototroch in row 0 and MC in row
     1. MC is the larva's giant head ciliomotor neuron and it INNERVATES the prototroch --
     Verasztó et al. (2017), eLife 6:e26000, "Ciliomotor circuitry underlying whole-body
     coordination of ciliary activity in the Platynereis larva". MC is presynaptic. Therefore
     row 0 is the POSTsynaptic cell, and the two rows must be exchanged.

WHY THE WEIGHTS DO NOT CHANGE. `weights` is per-edge and rides along. The scale that set the
spectral radius to 0.9 was computed on W, and swapping the rows makes the dense matrix W^T, whose
eigenvalues are exactly those of W -- so the radius is unchanged and no ratio between any two
connections moves. Only the direction of every arrow does.

WHAT IT TOUCHES. `connectome.npz` and `neurons.npz` of the region, IN PLACE, keeping the originals
as `*_as_built_transposed.npz`. This is shared data: `config/neural/platynereis_larva.yaml` and
the copy under /workspace/connectome-gnn read the same region, and both were running the circuit
backwards. `build_region.py` is patched too, so a rebuild does not reintroduce it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

REGION = "platynereis_larva_4117"


def region_dir() -> str:
    from plexus.paths import graphs_data_path
    return os.path.join(graphs_data_path(), "neural_regions", REGION)


def evidence(R: str) -> dict:
    """The three tests, run on whatever is on disk now."""
    z = np.load(os.path.join(R, "neurons.npz"), allow_pickle=True)
    c = np.load(os.path.join(R, "connectome.npz"), allow_pickle=True)
    ei, w = np.asarray(c["edge_index"]), np.asarray(c["contacts"])
    cn, ci, ct = list(z["cell_class_names"]), np.asarray(z["cell_class_id"]), z["celltype"]
    out = {}
    for cls in ("muscle", "ciliary band", "Sensory neuron"):
        m = ci == cn.index(cls)
        out[cls] = (int(m[ei[0]].sum()), int(m[ei[1]].sum()))     # (in row 0, in row 1)
    mc = np.array([str(t) == "MC" for t in ct])
    pt = np.array([str(t) == "prototroch" for t in ct])
    out["MC->prototroch"] = int((mc[ei[0]] & pt[ei[1]]).sum())
    out["prototroch->MC"] = int((pt[ei[0]] & mc[ei[1]]).sum())
    out["MC_synapses"] = int(w[(pt[ei[0]] & mc[ei[1]]) | (mc[ei[0]] & pt[ei[1]])].sum())
    return out


def report(e: dict) -> bool:
    """Print the evidence; return True if the file still needs the swap."""
    print("  effectors, as the file reads today (edges with the class in row 0 / in row 1):")
    for k in ("muscle", "ciliary band", "Sensory neuron"):
        print(f"    {k:16s} row0 {e[k][0]:5d}   row1 {e[k][1]:5d}")
    print(f"\n  the MC cell and the prototroch, {e['MC_synapses']} synapses over 23 contacts:")
    print(f"    MC in row 0, prototroch in row 1 : {e['MC->prototroch']}")
    print(f"    prototroch in row 0, MC in row 1 : {e['prototroch->MC']}")
    bad = e["prototroch->MC"] > e["MC->prototroch"]
    print(f"\n  MC innervates the prototroch (Veraszto et al. 2017, eLife 6:e26000), so MC must be "
          f"in the PRE row.\n  -> the file is {'BACKWARDS and needs the swap' if bad else 'correct'}.")
    return bad


def apply(R: str) -> None:
    for name in ("connectome.npz", "neurons.npz"):
        src = os.path.join(R, name)
        keep = os.path.join(R, name.replace(".npz", "_as_built_transposed.npz"))
        if not os.path.exists(keep):
            shutil.copyfile(src, keep)
            print(f"  kept the original as {os.path.basename(keep)}")

    p = os.path.join(R, "connectome.npz")
    c = dict(np.load(p, allow_pickle=True))
    ei = np.asarray(c["edge_index"])
    c["edge_index"] = ei[::-1].copy()                 # (post, pre) -> (pre, post)
    c["meta_source"] = np.asarray(
        str(c["meta_source"]) + "  [direction corrected 2026-09-19: the CSV's rows are "
        "PREsynaptic and build_region.py read them as post; see tools/platynereis_fix_direction.py]")
    np.savez(p, **c)
    print(f"  connectome.npz: {ei.shape[1]:,} edges reversed")

    p = os.path.join(R, "neurons.npz")
    z = dict(np.load(p, allow_pickle=True))
    # n_pre and n_post were COUNTED off the wrong rows, so they are each other.
    z["n_pre"], z["n_post"] = np.asarray(z["n_post"]).copy(), np.asarray(z["n_pre"]).copy()
    np.savez(p, **z)
    print("  neurons.npz: n_pre and n_post exchanged")

    note = os.path.join(R, "DIRECTION.md")
    with open(note, "w") as f:
        f.write(
            "# The edge direction of this region was corrected on 2026-09-19\n\n"
            "`build_region.py` read `full_connectome_adjacency_matrix.csv` taking each ROW as the\n"
            "POSTsynaptic cell. The source has rows PREsynaptic, so every edge pointed backwards.\n\n"
            "It was found because the file gave 840 muscle cells and 74 ciliary-band cells that\n"
            "SEND synapses to the nervous system and receive none, and because the one MC cell --\n"
            "the larva's giant head ciliomotor neuron, which innervates the prototroch (Veraszto\n"
            "et al. 2017, eLife 6:e26000) -- appeared as the postsynaptic partner of all 23\n"
            "prototroch cells rather than their presynaptic one.\n\n"
            "`edge_index` has been reversed and `n_pre`/`n_post` exchanged. The per-edge `weights`\n"
            "did not change: swapping the rows transposes the dense matrix, and a matrix and its\n"
            "transpose have the same eigenvalues, so the spectral radius of 0.9 still holds and no\n"
            "ratio between two connections moved.\n\n"
            "The files as originally built are kept beside these as\n"
            "`connectome_as_built_transposed.npz` and `neurons_as_built_transposed.npz`.\n\n"
            "ANYTHING THAT RAN AGAINST THIS REGION BEFORE THIS DATE RAN THE CIRCUIT BACKWARDS,\n"
            "including `config/neural/platynereis_larva.yaml` and the copy under\n"
            "/workspace/connectome-gnn.\n")
    print(f"  wrote {os.path.basename(note)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    R = region_dir()
    print(f"{REGION} at {R}\n")
    bad = report(evidence(R))
    if not a.apply:
        print("\n(--check only; pass --apply to swap)")
    elif not bad:
        print("\nnothing to do.")
    else:
        print()
        apply(R)
        print("\nafter:")
        report(evidence(R))
