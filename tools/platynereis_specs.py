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

import math

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
            "camera_roll": 180.0,
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


# THE MATERIAL OF EACH CLASS, a first guess and stated as such: nothing here has been fitted.
# Young's modulus in pascals, density in kg/m3. The chaetae are stiff because they are chitin
# rods; the yolk is soft and the densest thing in the animal because it is stored lipid and
# protein; muscle and epidermis sit between.
MATERIAL = {
    "Interneuron": (1500.0, 1050.0), "Motoneuron": (1500.0, 1050.0),
    "Sensory neuron": (1500.0, 1050.0), "chaetal complex": (8000.0, 1200.0),
    "ciliary band": (1200.0, 1050.0), "coelothelium": (1200.0, 1040.0),
    "epidermis": (2500.0, 1060.0), "excretory system": (1200.0, 1040.0),
    "gland": (1000.0, 1040.0), "gland cell": (1000.0, 1040.0),
    "glia cell": (1200.0, 1040.0), "macrophage-like": (900.0, 1030.0),
    "mesoderm": (1500.0, 1050.0), "microvillar": (1200.0, 1040.0),
    "multiciliated cell": (1200.0, 1050.0), "muscle": (3000.0, 1060.0),
    "pigment cell": (1800.0, 1060.0), "yolk": (400.0, 1150.0),
}


def positions(f: dict, scale: float, offset: list) -> list:
    """The 4,117 measured soma positions, mapped into the world exactly as `neural_seed` maps them.

    WHY THEY ARE WRITTEN INTO THE SPEC AND NOT SEEDED. `neural_seed` runs in the `seed:` block,
    which is AFTER the hierarchy is built -- and an `mpm_particle` set is filled around its
    parent's position at BUILD time. So a spec that seeds neurons from the region and hangs
    material points off them gets its cells moved to their measured positions and their matter
    left behind at the origin: the whole animal renders as one small blob, which is exactly what
    happened the first time. Writing the positions as the set's own `start:` puts the cells where
    they belong before any particle is placed.

    The affine is `neural_seed`'s, to the digit: `offset + scale * (xyz - bounds_lo) / side`. The
    provenance it would have recorded is written into the spec's header comment instead.
    """
    import numpy as _np
    z = _np.load(os.path.join(region_dir(), "neurons.npz"), allow_pickle=True)
    xyz = _np.asarray(z["xyz"], _np.float64)
    unit = (xyz - _np.asarray(f["lo_nm"])) / f["side_nm"]
    w = _np.asarray(offset)[None, :] + scale * unit
    return [[round(float(v), 6) for v in row] for row in w]


def spec_r2(f: dict, scale: float = 0.55, offset=(0.22, 0.22, 0.74),
            per_cell: int = 24, soma_um: float = 1.9) -> dict:
    """R2: the assembled anatomy as matter, dropped.

    Every cell becomes a parent of `per_cell` MPM material points, so the animal is a heap of
    4,117 soft balls sharing one background grid -- which is what lets them push on each other.
    The particle mass fixes each cell's VOLUME (`V = per_parent * particle_mass / density`), and
    it is chosen so a cell of density 1050 is a ball of `soma_um` radius. It is in WORLD units,
    not kilogrammes: `p_vol` is read as a world-cube volume.
    """
    um = f["side_um"]
    r = soma_um / um                                       # the soma radius, in world units
    pm = 1050.0 * (4.0 / 3.0) * math.pi * r ** 3 / per_cell
    start = positions(f, scale, list(offset))
    types = {c: {"count": f["count"][c], "shape": "ball", "material": "elastic",
                 "youngs": MATERIAL[c][0], "density": MATERIAL[c][1]} for c in f["order"]}
    return {
        "general": {
            "name": "plat_r2_fall", "seed": 0, "n_frames": 300, "dt": 0.005, "dim": 3,
            "world": [1.0, 1.0, 1.6], "boundary": "wall", "save_data": True, "record_cap": 301,
            # `length_um` IS HERE BECAUSE `neural_seed` CHECKS IT, not because the dynamics are
            # in SI. The engine reads `gz`, `youngs` and `density` as RAW numbers in world units
            # per time unit -- it does not rescale them by this block -- and the first version of
            # this rung proved it the hard way: `gz: -9.81` with a 195.9 um world unit is 9.81
            # WORLD units per second squared, so in 4.8 ms the animal fell 21 NANOMETRES, which
            # is what the trajectory measured. See the note above the operators.
            "units": {"length_um": um, "time_s": 1.0},
        },
        "sets": {
            "cell": {"n": f["n"], "start": start, "type_layout": "ordered", "types": types},
            "mpm_particle": {"parent": "cell", "per_parent": per_cell, "density": 1050.0,
                             "radius": round(r, 6), "particle_mass": float(f"{pm:.4g}")},
        },
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 96}},
        "operators": [
            # GRAVITY, IN WORLD UNITS PER TIME UNIT SQUARED, and the whole scene with it.
            #
            # 0.5 is not 0.5 m/s2 and is not meant to be. The scene is stated consistently in the
            # engine's own units and reads as follows: the animal starts 0.34 of a world unit (67
            # um, a third of its own length) clear of the face it will land on and reaches it
            # after 1.17 s of scene time, arriving at 0.58 world units per second -- inside the
            # 1.5 s the run lasts, with time left to watch it settle.
            #
            # IT FALLS ALONG +z, AND THAT IS A STATEMENT ABOUT THE PICTURE, NOT THE PHYSICS. The
            # region's z runs HEAD to TAIL, so every view of this animal is rolled 180 degrees to
            # put its head up, which also puts LOW z at the top of the screen. Gravity along -z
            # would then read as the animal falling upwards out of frame, which is what the first
            # landing looked like. Along +z it falls toward the bottom of the screen and lands on
            # the z = 1.6 face, which `boundary: wall` treats like any other wall. Head up, falling
            # down; nothing about the mechanics differs. The stiffest tissue here has a sound speed sqrt(E/rho) = sqrt(8000/1200)
            # = 2.6 world units per second, so the impact is Mach 0.23 -- an elastic body landing,
            # not a hypersonic one shattering.
            #
            # The alternative, real SI throughout, needs Young's moduli near 1e10 to keep the same
            # Mach number once a world unit is 196 um, and the substep that then satisfies the
            # Courant condition makes the run hours long for no extra truth.
            #
            # ALONG -z, NAMED AXIS BY AXIS: a bare `g:` pulls along -y, which is right for a scene
            # whose up axis is y and wrong here, where the animal's long axis runs head to tail
            # along z -- a bare `g` would drag it sideways through its own body.
            {"op": "gravity", "at": "cell", "gx": 0.0, "gy": 0.0, "gz": 0.5},
            {"op": "mpm_strain", "at": "mpm_particle", "implementation": "warp"},
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid", "drag": 0.0,
             "a_max": 200.0, "implementation": "warp", "polar": "higham"},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 0.6, "wall_friction": 0.4},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "vmax": 1.0e9, "implementation": "warp"},
        ],
        "schedule": ["gravity", {"substep_dt": 1.0e-3,
                                 "steps": ["mpm_strain", "mpm_scatter", "mpm_grid_update",
                                           "mpm_gather"]}],
        "plotting": {
            "renderer": "vtk_points", "background": "black", "box_frame": True, "up_axis": 2,
            "dot_shading": True, "max_frames": 300, "stills": 4, "keep_stills": True,
            # The movie is rolled with the stills, so the two pictures of one run agree. Without
            # it the film showed the larva tail-up and falling toward the top of the frame.
            "camera_roll": 180.0,
            "dot_size": 3.0, "colors": {c: COLOR[c] for c in f["order"]},
        },
    }


def spec_r3(f: dict) -> dict:
    """R3: the connectome, carried as an edge set and drawn over the cells that wear it.

    THE SET IS DECLARED, NOT LOADED BY AN OPERATOR. `edges_file:` on a set with `edge_set: true`
    is a property of the set: the 4,664 edges arrive with the hierarchy and the `synapse` set's
    `pre`/`post` legs point into the neuron set, so `neuron_signal` and `readout` can both walk
    them without anything else being declared.

    ONLY 1,009 OF THE 4,117 CELLS CARRY AN EDGE, and that is a property of the source, not a
    defect here: the adjacency matrix joins the geometry BY CELL NAME, which only 1,196 cells
    have. Degree zero means unjoined, not unwired. The wired classes are the interesting ones --
    98% of motoneurons, 99% of the ciliary band, 89% of interneurons -- so the circuit this rung
    needs is present even though most of the epidermis is not.

    The cells are drawn small and dim and the synapses bright, because the subject of this rung is
    the wiring and not the bodies.
    """
    um = f["side_um"]
    return {
        "general": {
            "name": "plat_r3_connectome", "seed": 0, "n_frames": 0, "dt": 0.05, "dim": 3,
            "world": [1.0, 1.0, 1.0], "boundary": "wall", "record_cap": 2,
            "units": {"length_um": um, "time_s": 1.0},
        },
        "sets": {
            "brain": {"n": 1},
            "neuron": {"parent": "brain", "per_parent": f["n"], "type_layout": "ordered",
                       "types": {c: {"count": f["count"][c]} for c in f["order"]}},
            "synapse": {"parent": "brain", "edge_set": True, "entity": "connection",
                        "pre": "neuron", "post": "neuron",
                        "edges_file": f"neural_regions/{REGION}/connectome.npz"},
        },
        "seed": [{"op": "neural_seed", "at": "neuron", "region": REGION,
                  "v0_mean": 0.0, "v0_sd": 0.0}],
        "operators": [],
        "schedule": [],
        "fields": {},
        "plotting": {
            "renderer": "vtk_points", "background": "black", "box_frame": False, "up_axis": 2,
            "dot_shading": True, "max_frames": 1, "stills": 1, "keep_stills": True,
            "camera_roll": 180.0,
            "dot_radius": {c: round(0.9 / um, 6) for c in f["order"]},
            "glyphs_only": True,
            "colors": dict({c: COLOR[c] for c in f["order"]}, synapse="#f2f2f2"),
            # `always: true` because a circuit's synapses ARE the picture, not its closing
            # statement -- the default shows each relation once, over the last few frames.
            # 0.22 and not 0.55: at 4,664 edges over a 200 um animal the lines saturate into a
            # white sheet and the cells vanish behind their own wiring. Faint enough that density
            # reads as density.
            "graph_overlay": {"sets": ["synapse"], "always": True, "line_width": 1.0,
                              "opacity": 0.22, "max_edges": 8000},
            "static_mesh": {
                "file": f"neural_regions/{REGION}/body_outline.obj",
                "color": "#aeb6c2", "opacity": 0.10,
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
    ap.add_argument("rung", nargs="?", default="all", choices=["r1", "r2", "r3", "all"])
    ap.add_argument("--list", action="store_true", help="say what would be written, write nothing")
    a = ap.parse_args()
    F = facts()
    print(f"{REGION}: {F['n']:,} cells, {len(F['order'])} classes, {F['n_edges']:,} edges, "
          f"cube {F['side_um']:g} um\n")
    if a.rung in ("r1", "all"):
        write(list(range(len(R1))), F, dry=a.list)
    if a.rung in ("r3", "all"):
        import yaml
        sp = spec_r3(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} {F['n_edges']:,} synapses over {F['n']:,} cells")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R3 -- the connectome: 4,664 measured synapses, carried as an edge set\n"
                         "# and drawn over the cells that wear them. Direction verified: see\n"
                         "# tools/platynereis_fix_direction.py and tests/test_platynereis.py.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r2", "all"):
        import yaml
        sp = spec_r2(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        n_p = F["n"] * sp["sets"]["mpm_particle"]["per_parent"]
        print(f"  {os.path.relpath(p, REPO):48s} {F['n']:,} cells x "
              f"{sp['sets']['mpm_particle']['per_parent']} points = {n_p:,} material points")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write(
                    "# R2 -- the assembled anatomy, dropped.\n#\n"
                    "# Every one of the 4,117 cells is a parent of MPM material points, so the\n"
                    "# animal is a heap of soft balls sharing one background grid. Then gravity is\n"
                    "# turned on for the first time. What it tests is whether the measured anatomy\n"
                    "# is MECHANICALLY coherent: a cloud of dots falls the same way whatever the\n"
                    "# dots mean, a body does not.\n#\n"
                    "# The soma positions in `start:` are the region's own, mapped by exactly the\n"
                    "# affine `neural_seed` uses -- offset + scale * (xyz - bounds_lo) / side, with\n"
                    "# bounds_lo " + str(F["lo_nm"]) + " nm and side " + f"{F['side_nm']:.0f}"
                    + " nm.\n"
                    "# Source: Veraszto et al. 2025, eLife RP97964, whole-body connectome of a\n"
                    "# 3-day Platynereis dumerilii nectochaete larva.\n"
                    "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
