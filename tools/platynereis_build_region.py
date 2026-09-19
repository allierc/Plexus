#!/usr/bin/env python
"""Freeze the whole-body larva of *Platynereis dumerilii* as a Plexus neural region.

The animal is a 3-day nectochaete larva, ~150 um long, and it is a genuine ciliary microswimmer:
80 multiciliated band cells in five girdles (prototroch, akrotroch, metatroch, paratroch, nuchal)
drive it at a Reynolds number near 0.3, and a small ciliomotor circuit -- one cholinergic MC cell
that arrests every cilium at once, two serotonergic Ser-h1 cells that raise the beat frequency --
starts and stops them.

Source: Veraszto C, Jasek S, Guhmann M, Bezares-Calderon LA, Williams EA, Shahidi R, Jekely G
(2025) "Whole-body connectome of a segmented annelid larva", eLife 13:RP97964,
doi 10.7554/eLife.97964. Research compendium github.com/JekelyLab/Platynereis_3D_connectome_2024.

WHAT MAKES THIS DIFFERENT FROM THE CTENOPHORE, which is the reason it was chosen. There, a cell's
position existed only if it bore an annotated synapse, so 271 of 903 cells could be placed and the
rest had to be invented. Here the geometry comes from the cell-type compendium, which is rendered
from the FULL TRACED SKELETONS: a cell has a position because it was traced, not because it was
wired. 4,117 somata are placed, including 944 of the paper's 966 neurons and every one of the 80
ciliary band cells, and each soma arrives with its own smoothed arbour rather than as a dot.

WHERE THE GEOMETRY IS READ FROM, and why it needs decoding at all. The compendium is 291 HTML
pages, one per cell type, each an rgl/htmlwidget scene carrying a base64 glTF-style buffer. Inside
one page the objects are, in order: 35 `linestrip` objects that every page redraws as background
(the acicula and a bounding scaffold), one `spheres` object holding one soma per cell of that
type, then one `linestrip` per cell -- IN SPHERE ORDER. That ordering is the join between a soma
and its arbour, and it is checked rather than assumed: every cell's arbour must begin within one
nanometre of its own soma, or the build fails.

WHAT THE EDGES ARE. `full_connectome_adjacency_matrix.csv` is 1,720 x 1,720 over NAMED cells
(`PRC_al3`, `prototroch_pl2`), and 1,196 of the 4,117 somata carry such a name, so the wiring is
joined by name and the region records which cells are wired and which are only drawn. Nothing is
extrapolated: a cell without a name gets no edges rather than a guessed one.

    python build_region.py            # writes ../neural_regions/platynereis_larva_<N>/
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.abspath(os.path.join(HERE, "..", "candidates"))
PICK = os.path.join(CAND, "Platynereis_dumerilii_3_day_old_nectochaete_larva_annelid_trochophore_lineage_RANK_1_THE_PICK")
FULL = os.path.join(CAND, "Platynereis_dumerilii_3_day_old_nectochaete_larva_whole_body_150_um_long_")
RGL = os.path.join(FULL, "repo", "supplements", "celltype_compendium_website",
                   "celltype_compendium", "celltype_RGLs")
REGION_ROOT = os.path.abspath(os.path.join(HERE, "..", "neural_regions"))
NAME = "platynereis_larva"

N_BACKGROUND = 35      # linestrips every page redraws before the per-cell arbours; verified below

# THE CILIARY BAND CELLS, named so the model can find its own effectors. These 80 cells are what
# propels the animal, and they are the reason this organism was chosen over a larger connectome
# that does not swim. Kept as a set rather than inferred from a name substring, because
# `crescentcell` does not contain "troch" and would be missed.
CILIARY_BANDS = {"prototroch", "akrotroch", "metatroch", "paratroch", "nuchal", "crescentcell"}


def soma_table() -> list:
    """The 4,117 placed somata, with every label the compendium and the connectome agree on."""
    with open(os.path.join(PICK, "soma_positions.csv")) as f:
        return list(csv.DictReader(f))


def _geometry(path):
    sys.path.insert(0, os.path.join(FULL, "tools"))
    import extract_geometry as G                                   # noqa: E402
    return G.geometry(path)


def arbours() -> dict:
    """(celltype_annotation, index_in_type) -> the cell's own arbour, an (n, 3) array in nm.

    NaN triples are rgl's separators between the disjoint pieces of one cell's tree; they are kept
    here, because dropping them would join the end of one neurite to the start of the next and
    draw a segment through empty tissue. `_swc` below turns each run between separators into its
    own chain.
    """
    out, checked, failed = {}, 0, []
    pages = sorted(p for p in os.listdir(RGL) if p.endswith(".html"))
    for i, page in enumerate(pages):
        ann = page[:-5]
        try:
            S, _R, PL, _M = _geometry(os.path.join(RGL, page))
        except Exception as e:                                       # noqa: BLE001
            failed.append((ann, f"{type(e).__name__}: {e}"))
            continue
        arb = PL[N_BACKGROUND:]
        if len(S) == 0 or len(arb) != len(S):
            # An overview page draws cells from many types and has no one-to-one arbour list; it
            # contributes somata elsewhere and no morphology here.
            continue
        for k in range(len(S)):
            a = np.asarray(arb[k], dtype=np.float64)
            good = a[~np.isnan(a).any(axis=1)]
            if len(good) == 0:
                continue
            # THE JOIN, CHECKED. The arbour list is in sphere order only because the generator
            # plots them that way; if that ever stops being true every cell gets a neighbour's
            # morphology and the picture is wrong in a way nothing else would catch.
            d = float(np.linalg.norm(good - S[k], axis=1).min())
            if d > 1.0:
                failed.append((f"{ann}[{k}]", f"arbour starts {d:.1f} nm from its soma"))
                continue
            out[(ann, k)] = a
            checked += 1
        if (i % 40) == 0:
            print(f"  {i}/{len(pages)} pages, {checked} arbours", flush=True)
    if failed:
        print(f"  {len(failed)} pages/cells skipped, first few: {failed[:3]}")
    return out


def _swc(arb: np.ndarray, radius_nm: float) -> str:
    """One cell's polyline set as an `.swc`: a chain per run, parent = the previous row.

    An swc row is (id, type, x, y, z, radius, parent) and a parent of -1 opens a new tree, which
    is exactly what a separator between two disjoint neurite pieces means. Radius is the soma's,
    uniformly: the compendium's linestrips carry no per-point thickness, and inventing a taper
    would put a measurement-shaped number on something never measured.
    """
    rows, rid = [], 0
    prev = -1
    for p in arb:
        if np.isnan(p).any():
            prev = -1
            continue
        rid += 1
        rows.append(f"{rid} 3 {p[0]:.1f} {p[1]:.1f} {p[2]:.1f} {radius_nm:.1f} {prev}")
        prev = rid
    return "\n".join(rows) + "\n"


def build(write_morphology: bool = True) -> dict:
    rows = soma_table()
    print(f"soma table: {len(rows)} placed cells")

    # ROW ORDER: by cell class, then cell type, then the type's own index. Classes first because
    # `type_layout: ordered` needs whatever the picture colours by to be contiguous, and the class
    # (sensory / inter / motor / ciliary band / muscle / epidermis ...) is the coarsest useful cut.
    def cls(r):
        c = (r.get("celltype_class") or r.get("cell_class") or "").strip()
        if r["celltype_name"].strip() in CILIARY_BANDS:
            return "ciliary band"
        return c or "unassigned"

    order = sorted(range(len(rows)), key=lambda i: (cls(rows[i]), rows[i]["celltype_annotation"],
                                                    int(rows[i]["idx_in_type"])))
    rows = [rows[i] for i in order]
    xyz = np.array([[float(r["x_nm"]), float(r["y_nm"]), float(r["z_nm"])] for r in rows])

    combined = [f'{cls(r)}__{r["celltype_name"].strip() or r["celltype_annotation"]}' for r in rows]
    tn, seen = [], set()
    for c in combined:
        if c not in seen:
            seen.add(c); tn.append(c)
    t_index = {c: i for i, c in enumerate(tn)}
    type_id = np.array([t_index[c] for c in combined], dtype=np.int64)

    def codes(vals):
        names = sorted({v for v in vals})
        idx = {v: i for i, v in enumerate(names)}
        return np.array([idx[v] for v in vals], np.int64), np.array(names, dtype=object)

    class_id, class_names = codes([cls(r) for r in rows])
    region_id, region_names = codes([(r.get("region") or "").strip() or "unstated" for r in rows])
    seg_id, seg_names = codes([(r.get("segment") or "").strip() or "unstated" for r in rows])
    side_id, side_names = codes([(r.get("side") or "").strip() or "unstated" for r in rows])

    cell_name = [r.get("cell_name", "").strip() for r in rows]
    # `body_id` IS THE ROW INDEX, and the CATMAID skid rides alongside as `skid`. Two reasons, and
    # the first is not aesthetic: `morphology_seed` resolves a cell's skeleton file by
    # `morphology_index[str(body_id)]`, so whatever keys that index must be unique and must exist
    # for EVERY cell -- and only 2,896 of the 4,117 cells have a skid. Keying by skid would
    # silently drop 1,221 cells' morphology, which is most of the epidermis and muscle, i.e. most
    # of what the animal looks like. Second, -1 repeated 1,221 times is not an identity.
    skid = np.array([int(r["skid"]) if r.get("skid") else -1 for r in rows], np.int64)
    body_id = np.arange(len(rows), dtype=np.int64)

    # ------------------------------------------------------------------ the measured wiring
    # Joined by CELL NAME, which is the only key the adjacency matrix and the geometry share.
    with open(os.path.join(PICK, "source", "full_connectome_adjacency_matrix.csv")) as f:
        rd = list(csv.reader(f))
    # THE ROWS ARE PRESYNAPTIC AND THE COLUMNS POSTSYNAPTIC. This was the other way round here
    # until 2026-09-19, and it reversed every edge in the region: 840 muscle cells and 74
    # ciliary-band cells came out SENDING synapses to the nervous system and receiving none, and
    # the one MC cell -- the larva's giant head ciliomotor neuron -- appeared as the postsynaptic
    # partner of all 23 prototroch cells instead of their presynaptic one. The source settles it:
    # in `full_connectome_adjacency_matrix.csv` the ROW `MC` carries the prototroch entries
    # (prototroch_1P 27, prototroch_5A 16, ...) while the COLUMN `MC` carries inputs from sensory
    # cells. MC drives the prototroch and is driven by sensory neurons.
    # Ref: Veraszto et al. (2017), eLife 6:e26000.
    hdr = rd[0][1:]
    name_row = {n: i for i, n in enumerate(cell_name) if n}
    post_cols = [(j, hdr[j]) for j in range(len(hdr)) if hdr[j] in name_row]
    ei, w = [], []
    for line in rd[1:]:
        pre = line[0]
        if pre not in name_row:
            continue
        i_pre = name_row[pre]
        for j, post in post_cols:
            v = line[j + 1]
            if v and v not in ("0", "0.0"):
                ei.append((i_pre, name_row[post])); w.append(float(v))
    ei = np.asarray(ei, np.int64).T if ei else np.zeros((2, 0), np.int64)
    contacts = np.asarray(w, np.float32)
    print(f"connectome: {ei.shape[1]} directed edges over {len(name_row)} named cells "
          f"({int(contacts.sum())} synapses)")

    # THE WEIGHTS ARE NOT THE SYNAPSE COUNTS. Same argument as the ctenophore region: a count is
    # anatomy, a weight is a gain, and nothing in the data says what gain one synapse is worth.
    # One global factor puts the spectral radius -- the gain of the network's most amplified mode
    # -- at 0.9, so the linear part contracts and the circuit settles instead of saturating. No
    # ratio between any two connections changes.
    n = len(rows)
    W = np.zeros((n, n))
    if ei.shape[1]:
        W[ei[1], ei[0]] = contacts
    rho = float(np.abs(np.linalg.eigvals(W)).max()) if n and ei.shape[1] else 0.0
    w_scale = (0.9 / rho) if rho > 1e-12 else 1.0
    ew = (contacts * w_scale).astype(np.float32)

    deg = collections.Counter(ei[1].tolist()) if ei.shape[1] else {}
    out_deg = collections.Counter(ei[0].tolist()) if ei.shape[1] else {}
    n_in = np.array([deg.get(i, 0) for i in range(n)], np.float32)
    n_out = np.array([out_deg.get(i, 0) for i in range(n)], np.float32)
    radius = np.array([float(r.get("soma_radius_nm") or 2000.0) for r in rows], np.float32)

    # ------------------------------------------------------------------ the cube
    lo = xyz.min(axis=0)
    side = float(np.ceil((xyz.max(axis=0) - lo).max() * 1.05 / 100.0) * 100.0)
    lo = lo - 0.5 * (side - (xyz.max(axis=0) - lo).max())

    out = os.path.join(REGION_ROOT, f"{NAME}_{n}")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "neurons.npz"), xyz=xyz, bounds_lo=lo, bounds_side=np.float64(side),
             body_id=body_id, skid=skid, type_id=type_id, type_names=np.array(tn, dtype=object),
             soma_radius=radius, n_pre=n_out.astype(np.int64), n_post=n_in.astype(np.int64),
             cell_class_id=class_id, cell_class_names=class_names,
             region_id=region_id, region_names=region_names,
             segment_id=seg_id, segment_names=seg_names,
             side_id=side_id, side_names=side_names,
             cell_name=np.array(cell_name, dtype=object),
             celltype=np.array([r["celltype_name"].strip() for r in rows], dtype=object))
    np.savez(os.path.join(out, "connectome.npz"), edge_index=ei, weights=ew, contacts=contacts,
             meta_n_neurons=np.int64(n), meta_n_edges=np.int64(ei.shape[1]),
             meta_measured=np.bool_(True), meta_spectral_radius=np.float64(0.9),
             meta_weight_scale=np.float64(w_scale), meta_raw_spectral_radius=np.float64(rho),
             meta_source=np.array("Veraszto et al. 2025 eLife RP97964, "
                                  "full_connectome_adjacency_matrix.csv joined by cell name",
                                  dtype=object))

    # ------------------------------------------------------------------ the arbours
    n_skel = 0
    if write_morphology:
        print("decoding arbours from 294 compendium pages ...")
        arb = arbours()
        sk_dir = os.path.join(out, "skeletons")
        os.makedirs(sk_dir, exist_ok=True)
        index = {}
        for i, r in enumerate(rows):
            key = (r["celltype_annotation"], int(r["idx_in_type"]))
            a = arb.get(key)
            if a is None:
                continue
            # KEYED BY ROW INDEX, not by skid: only 2,896 of the 4,117 cells have a skid, and a
            # region whose morphology index skipped every unnamed cell would drop a third of the
            # animal -- including muscle and epidermis, which is most of what a body looks like.
            fn = f"{i}.swc"                      # i IS the body_id, by construction above
            with open(os.path.join(sk_dir, fn), "w") as f:
                f.write(_swc(a, float(radius[i])))
            index[str(i)] = f"skeletons/{fn}"
            n_skel += 1
        with open(os.path.join(out, "morphology_index.json"), "w") as f:
            json.dump({"skeletons": index, "meshes": {}, "keyed_by": "body_id, which is the row index in neurons.npz"}, f)
        print(f"  wrote {n_skel} skeletons")

    # the body surface, copied in so the region is self-contained
    src_obj = os.path.join(FULL, "geometry", "body_outline_yolk.obj")
    if os.path.exists(src_obj):
        import shutil
        shutil.copy(src_obj, os.path.join(out, "body_outline.obj"))

    man = {
        "source": {"dataset": "Veraszto et al. 2025, eLife RP97964 -- whole-body connectome of a "
                              "3-day Platynereis dumerilii nectochaete larva",
                   "server": "none -- decoded offline from the cell-type compendium",
                   "query": "every cell of the compendium that carries a soma sphere"},
        "region": {"shape": "measured", "n_neurons": n,
                   "bounds_lo_nm": [float(v) for v in lo], "side_nm": side,
                   "side_um": side / 1000.0, "voxel_size_nm": None},
        "n_edges": int(ei.shape[1]),
        "types": {k: int((type_id == v).sum()) for k, v in t_index.items()},
        "cell_classes": {str(k): int((class_id == i).sum()) for i, k in enumerate(class_names)},
        "segments": {str(k): int((seg_id == i).sum()) for i, k in enumerate(seg_names)},
        "sides": {str(k): int((side_id == i).sum()) for i, k in enumerate(side_names)},
        "morphology": {"n_skeletons": n_skel, "n_meshes": 0, "skeleton_dir": "skeletons",
                       "keyed_by": "body_id == row index", "body_outline": "body_outline.obj"},
        "wiring": {"cells_with_a_connectome_name": int(sum(1 for c in cell_name if c)),
                   "cells_with_a_skid": int((skid >= 0).sum()),
                   "note": "edges join the geometry by CELL NAME, the only key the adjacency "
                           "matrix and the compendium share. A cell without a name has no edges "
                           "rather than a guessed one, so degree 0 means unjoined, not unwired."},
        "swimmer": {"ciliary_band_cells": int(sum(1 for r in rows
                                                  if r["celltype_name"].strip() in CILIARY_BANDS)),
                    "note": "prototroch, akrotroch, metatroch, paratroch, nuchal and crescentcell "
                            "-- the multiciliated girdles that propel the larva"},
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    man["dir"] = out
    return man


if __name__ == "__main__":
    m = build(write_morphology="--no-morphology" not in sys.argv)
    print(json.dumps({k: v for k, v in m.items() if k != "types"}, indent=1)[:2600])
    print("\nwrote", m["dir"])
