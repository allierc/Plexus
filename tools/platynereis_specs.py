#!/usr/bin/env python
"""Write the Platynereis ladder into `config/platynereis/` -- one spec per rung.

    python tools/platynereis_specs.py r1          # the anatomy, layer by layer
    python tools/platynereis_specs.py --list      # what it would write

WHY A GENERATOR AND NOT ELEVEN HAND-WRITTEN FILES. The rungs differ in ONE thing: which cell
classes are drawn. Everything else -- the 4,117 somata, their 18 classes in the data's own row
order, the world box, the units, the body outline behind them -- is identical, and eleven copies
of it would drift. The specs it writes are the artefact and are checked in; this file is how they
are kept consistent, not a way around running them.

THE ANATOMY ORDER IS THE VIDEO'S. `graphs_data/platynereis/elife-97964-video1.mp4` builds the
larva up structure by structure, and `video_frames/contact_sheet.png` is that video sampled: body
outline, yolk, chaetae, endoderm, [synapses], sensory neurons, interneurons, ciliary bands, glia,
developing, neuropodium, epidermis. The non-neural ones are R1; the neurons arrive with the
connectome at R3, so they are last here and drawn all at once.

HOW A LAYER IS HIDDEN, and it needs no new renderer code: `plotting.dot_radius` is a mapping from
TYPE NAME to a world radius, and `live_movie._glyph_types` skips any type that has no entry
(`if r is None: continue`). A class absent from `dot_radius` is therefore not drawn at all -- not
drawn small, not drawn dark, not drawn behind something. When any `dot_radius` is present the
plain point cloud is replaced by the glyph actors entirely (`live_movie.py:1185`), so nothing
leaks through underneath.

Reference for the animal: Veraszto et al. (2025), eLife RP97964 -- the whole-body connectome of a
three-day Platynereis dumerilii nectochaete larva.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

REGION = "platynereis_larva_4117"
OUT = os.path.join(REPO, "config", "platynereis")

# THE RUNGS OF R1, each naming the cell classes it ADDS. Cumulative: rung k draws every class
# named at or before k, so the animal accretes exactly as the video builds it.
#
# The video's words on the left, the dataset's `cell_class_names` on the right. Two of them need a
# note. The video's "endoderm" is the gut lining, and the compendium's nearest classes are
# `coelothelium` (the coelomic lining, 101 cells) and `mesoderm` (24); neither is literally
# endoderm, and the spec says so rather than pretending the label matched. The video's
# "neuropodium" is a REGION of the parapodium, not a cell class -- its cells are already in
# `chaetal complex` and `muscle`, so it gets no rung of its own.
R1 = [
    ("outline",   [],                                              "the body's own surface, and nothing inside it yet"),
    ("yolk",      ["yolk"],                                        "the yolk mass the larva still lives off"),
    ("chaetae",   ["chaetal complex"],                             "the bristles, their aciculae and the follicle cells that hold them"),
    ("endoderm",  ["coelothelium", "mesoderm"],                    "the coelomic lining and mesoderm -- the video's 'endoderm' layer"),
    ("ciliary",   ["ciliary band", "multiciliated cell"],          "the ciliary bands: prototroch, metatroch, paratroch, akrotroch"),
    ("glia",      ["glia cell"],                                   "the glia"),
    ("muscle",    ["muscle"],                                      "the musculature: longitudinal, transverse, oblique, pharyngeal"),
    ("pigment",   ["pigment cell"],                                "the pigment cells, including the eyespots"),
    ("glands",    ["gland", "gland cell", "excretory system",
                   "microvillar", "macrophage-like"],              "glands, nephridia and the odds and ends"),
    ("epidermis", ["epidermis"],                                   "the epidermis, 1,120 cells, which closes the animal"),
    ("neurons",   ["Sensory neuron", "Interneuron", "Motoneuron"], "and the nervous system: 944 neurons, the subject of everything after R1"),
]

# ONE COLOUR PER CLASS, taken from the video's own palette so the two can be compared side by
# side. The video is a black-background montage and so is this.
COLOR = {
    "yolk":               "#d8d2c4",   # pale, almost the outline's grey
    # THE VIDEO DRAWS THE BRISTLES BLACK ON WHITE. This montage is black on black, and #2b2b30
    # made the whole 657-cell chaetal rung invisible -- a layer that cannot be seen has not been
    # added. Inverted to the pale the same ink would be on this background.
    "chaetal complex":    "#c9c9d2",
    "coelothelium":       "#d98a3d",   # the orange clusters
    "mesoderm":           "#b06a2c",
    "ciliary band":       "#e24a6a",   # the red bands
    "multiciliated cell": "#f07a94",
    "glia cell":          "#1f9e8f",   # teal
    "muscle":             "#c94040",
    "pigment cell":       "#6a6a93",
    "gland":              "#8e6fc4",
    "gland cell":         "#a98ad4",
    "excretory system":   "#5fb3d9",
    "microvillar":        "#7fd4c4",
    "macrophage-like":    "#9c9c6e",
    "epidermis":          "#6d7f96",   # the blue-grey shell
    "Sensory neuron":     "#e0a53a",   # gold, as the video draws them
    "Interneuron":        "#2f6fb5",   # blue
    "Motoneuron":         "#d45fa8",   # pink
}

# THE DRAWN RADIUS OF A SOMA, IN MICROMETRES, PER CLASS.
#
# It is NOT read from the dataset, and that has to be said out loud: `soma_radius` in neurons.npz
# is the constant 2000 nm for all 4,117 cells -- a placeholder the region builder wrote, not a
# measurement. Drawing every class at one radius would make the five yolk cells vanish inside a
# thousand epidermal ones, so these are chosen to match what the video shows: yolk and pigment
# large, epidermis and the neurons small. They are a DRAWING choice and nothing reads them as
# physics.
RADIUS_UM = {
    "yolk": 9.0, "pigment cell": 3.0, "coelothelium": 2.6, "mesoderm": 2.6,
    "chaetal complex": 2.0, "ciliary band": 2.4, "multiciliated cell": 2.4,
    "glia cell": 2.0, "muscle": 1.8, "gland": 2.4, "gland cell": 2.2,
    "excretory system": 2.2, "microvillar": 2.0, "macrophage-like": 2.2,
    "epidermis": 1.6, "Sensory neuron": 1.6, "Interneuron": 1.6, "Motoneuron": 1.8,
}


def region_dir() -> str:
    from plexus.paths import graphs_data_path
    return os.path.join(graphs_data_path(), "neural_regions", REGION)


def facts() -> dict:
    """What the frozen region says about itself: the classes in ROW ORDER with their counts, the
    cube that maps it into the unit box, and whether the classes are contiguous.

    The contiguity check is not decoration. `type_layout: ordered` hands rows to types in the
    order the types are declared, so declaring 18 classes only places them correctly if each
    class occupies ONE unbroken run of rows. If that ever stops being true the classes would be
    silently mislabelled -- every picture after it would be wrong and none of them would look it.
    """
    R = region_dir()
    z = np.load(os.path.join(R, "neurons.npz"), allow_pickle=True)
    man = json.load(open(os.path.join(R, "manifest.json")))
    names, cid = list(z["cell_class_names"]), np.asarray(z["cell_class_id"])
    runs = int((np.diff(cid) != 0).sum()) + 1
    if runs != len(set(cid.tolist())):
        raise SystemExit(
            f"the {len(set(cid.tolist()))} cell classes occupy {runs} runs of rows, so they are "
            f"NOT contiguous; `type_layout: ordered` would mislabel every cell. Sort the region "
            f"by cell class, or declare types per row instead.")
    order = [names[i] for i in sorted(set(cid.tolist()), key=lambda i: int(np.argmax(cid == i)))]
    return {
        "n": int(cid.shape[0]),
        "order": order,
        "count": {names[i]: int((cid == i).sum()) for i in set(cid.tolist())},
        "side_nm": float(man["region"]["side_nm"]),
        "side_um": float(man["region"]["side_um"]),
        "lo_nm": [float(v) for v in man["region"]["bounds_lo_nm"]],
        "n_edges": int(man["n_edges"]),
    }


def spec(rung: int, f: dict) -> dict:
    """The spec for one rung: every cell seeded, the first `rung` layers drawn."""
    shown = [c for _, cs, _ in R1[:rung + 1] for c in cs]
    name = f"plat_r1_{rung:02d}_{R1[rung][0]}"
    um = f["side_um"]                                    # one world unit = the cube's edge, in um
    return {
        "general": {
            "name": name, "seed": 0,
            # NO CLOCK AND NO GRAVITY. R1 is an assembly, not a simulation: one frame, seeded and
            # looked at. The animal is not allowed to fall until the anatomy is complete (R2).
            "n_frames": 0, "dt": 0.01, "dim": 3, "world": [1.0, 1.0, 1.0],
            "boundary": "wall", "record_cap": 2,
            # THE WORLD BOX IS THE REGION'S OWN CUBE, so one world unit is `side_um` micrometres
            # and every length in the picture can be read in the animal's units. `neural_seed`
            # CHECKS this against the manifest rather than assuming it.
            "units": {"length_um": um, "time_s": 1.0},
        },
        "sets": {
            "brain": {"n": 1},
            "neuron": {
                "parent": "brain", "per_parent": f["n"], "type_layout": "ordered",
                # ALL 4,117 ARE ALWAYS SEEDED, whatever the rung draws. `neural_seed` refuses a
                # count that disagrees with the manifest rather than truncating, and it is right
                # to: seeding the first N rows of a different animal is indistinguishable, in the
                # output, from seeding the right one.
                "types": {c: {"count": f["count"][c]} for c in f["order"]},
            },
        },
        "seed": [{"op": "neural_seed", "at": "neuron", "region": REGION,
                  "v0_mean": 0.0, "v0_sd": 0.0}],
        "operators": [],
        "schedule": [],
        "fields": {},
        "plotting": {
            "renderer": "vtk_points", "background": "black", "box_frame": False, "up_axis": 2,
            "dot_shading": True, "max_frames": 1, "stills": 1, "keep_stills": True,
            # THE LAYER SWITCH. Only the classes this rung has reached carry a radius, and a class
            # with no radius is not drawn (`live_movie._glyph_types`). See the module docstring.
            "dot_radius": {c: round(RADIUS_UM[c] / um, 6) for c in shown},
            # AND NOTHING BUT THOSE. Without it an empty `dot_radius` -- rung 00, which draws the
            # outline and no cells -- falls through to the plain point cloud and every one of the
            # 4,117 somata appears in the default blue, which is the opposite of that rung.
            "glyphs_only": True,
            "colors": {c: COLOR[c] for c in shown},
            # THE ANIMAL'S OWN SURFACE, segmented once and never simulated: scenery, drawn behind
            # everything so a layer can be seen sitting INSIDE the body rather than floating.
            "static_mesh": {
                "file": f"neural_regions/{REGION}/body_outline.obj",
                "color": "#aeb6c2", "opacity": 0.16,
                "to_world": {"origin": f["lo_nm"], "scale": 1.0 / f["side_nm"]},
            },
        },
    }


def write(rungs: list, f: dict, dry: bool = False) -> list:
    import yaml
    os.makedirs(OUT, exist_ok=True)
    out = []
    for k in rungs:
        s = spec(k, f)
        p = os.path.join(OUT, s["general"]["name"] + ".yaml")
        shown = sum(f["count"][c] for _, cs, _ in R1[:k + 1] for c in cs)
        print(f"  {os.path.relpath(p, REPO):48s} {shown:5,} cells drawn   {R1[k][2]}")
        if not dry:
            with open(p, "w") as fh:
                fh.write(f"# R1.{k:02d} -- {R1[k][2]}\n"
                         f"# Written by tools/platynereis_specs.py; the ladder is the video's own order.\n")
                yaml.safe_dump(s, fh, sort_keys=False, default_flow_style=False, width=100)
        out.append(p)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rung", nargs="?", default="r1")
    ap.add_argument("--list", action="store_true", help="say what would be written, write nothing")
    a = ap.parse_args()
    F = facts()
    print(f"{REGION}: {F['n']:,} cells, {len(F['order'])} classes, {F['n_edges']:,} edges, "
          f"cube {F['side_um']:g} um\n")
    write(list(range(len(R1))), F, dry=a.list)
