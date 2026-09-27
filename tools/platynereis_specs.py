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
            {"op": "mpm_gather", "at": "body_point", "from": "mpm_grid", "wall_damp": 1.0,
             "vmax": 1.0e9, "implementation": "warp"},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "vmax": 1.0e9, "implementation": "warp"},
            # THE CELL'S POSITION IS ITS MATERIAL'S CENTRE OF MASS, and without this the cell set
            # never moves at all. The MPM moves the material POINTS; a parent's `pos` is only its
            # seed position unless something aggregates it, so the first take measured the cells'
            # radial excursion as exactly 0.0000 um for all 4,117 while their particles were
            # travelling tens of micrometres. It also matters for the run and not only for the
            # measurement: `radial_polarity` is a seed, but anything later that reads a cell's
            # position -- a neighbour relation, a field sample, a second animal -- would be
            # reading frame 0 forever.
            {"op": "aggregate_centroid", "at": "cell", "child": "mpm_particle"},
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


# THE PER-CLASS DYNAMICS, as the parameter vector `neuron_ops` reads:
#
#     p = [a, b, g, s, w, h]   leak, offset, gain, self-coupling, width, threshold
#
# and the law those six sit in (neuron_ops.py:8) is
#
#     dx_i/dt = -a_i x_i + b_i + s_i tanh(x_i) + g_i * sum_j W_ij psi_ij(x_j) + eta_i
#
# with x_i the cell's membrane state (dimensionless -- these are NOT millivolts), a_i = 1/tau_i
# its leak rate per time unit, b_i a constant offset, s_i a self-coupling, g_i the gain on the
# synaptic sum, W_ij the measured connectome weight, psi the synaptic transfer function and eta
# noise of the declared standard deviation per step.
#
# NOTHING HERE IS FITTED, and the existing 291-type spec is no better off -- every one of its
# types carries the same placeholder. What IS said deliberately: the neurons are given the
# self-coupling s = 0.5 that makes a cell more than a leak, and the ciliary band is given twice
# the leak (a = 2.0) and half the width (w = 0.5), so an effector follows its drive rather than
# integrating it. Everything else takes the neuron values.
P_NEURON = [1.0, 0.0, 1.2, 0.5, 1.0, 0.0]
P_EFFECTOR = [2.0, 0.0, 1.2, 0.5, 0.5, 0.0]
P_CLASS = {"ciliary band": P_EFFECTOR, "multiciliated cell": P_EFFECTOR, "muscle": P_EFFECTOR}


def spec_r4(f: dict, n_frames: int = 600, dt: float = 0.05) -> dict:
    """R4: the connectome doing something -- the membrane law and the synaptic law, and a viz.

    `neuron_update` is the local half (phi) and `neuron_signal` the synaptic half (psi); both are
    already registered, so this rung declares rather than writes. `model: type_pairwise` lets the
    RECEIVER set the width and the sender the threshold, which is the richest of the three
    without introducing a parameter the data cannot constrain.

    WHAT MAKES IT MOVE. Nothing is driving this circuit yet: the activity is noise of standard
    deviation 0.01 per step, relaxed through a connectome scaled to spectral radius 0.9. A
    contracting network fed noise is the honest first picture -- it shows the wiring transporting
    something, without a stimulus whose shape would be the thing you were really watching. The
    drive arrives at R7, when there is a ciliary beat to tune.
    """
    um = f["side_um"]
    return {
        "general": {
            "name": "plat_r4_activity", "seed": 0, "n_frames": n_frames, "dt": dt, "dim": 3,
            "world": [1.0, 1.0, 1.0], "boundary": "wall", "save_data": True,
            "record_cap": n_frames + 1, "units": {"length_um": um, "time_s": 1.0},
        },
        "sets": {
            "brain": {"n": 1},
            "neuron": {"parent": "brain", "per_parent": f["n"], "type_layout": "ordered",
                       "types": {c: {"count": f["count"][c],
                                     "p": P_CLASS.get(c, P_NEURON)} for c in f["order"]}},
            "synapse": {"parent": "brain", "edge_set": True, "entity": "connection",
                        "pre": "neuron", "post": "neuron",
                        "edges_file": f"neural_regions/{REGION}/connectome.npz"},
        },
        # v0_sd 0.5 and not 0: v = 0 is a fixed point of the whole system, so a population seeded
        # exactly there sits at it and the run is a picture of nothing.
        "seed": [{"op": "neural_seed", "at": "neuron", "region": REGION,
                  "v0_mean": 0.0, "v0_sd": 0.5}],
        "operators": [
            {"op": "neuron_update", "at": "neuron", "model": "leaky_tanh", "noise": 0.01},
            {"op": "neuron_signal", "at": "neuron", "model": "type_pairwise",
             "edge_set": "synapse", "activation": "tanh"},
        ],
        "schedule": ["neuron_update", "neuron_signal"],
        "fields": {},
        "plotting": {
            "renderer": "vtk_points", "background": "black", "box_frame": False, "up_axis": 2,
            # EVERY FRAME KEPT, AND A BEAT THE FILM CAN RESOLVE. At omega 10 rad/s one cycle is
            # 12.6 frames and the movie's stride of 2 sampled it 6.3 times -- barely above
            # Nyquist, so a stroke that is perfectly smooth in the data reads on screen as a
            # jitter, and what you are watching is the sampling rather than the cilium. At 2.5
            # rad/s (0.40 Hz) a cycle is 50 frames, `max_frames` equals `n_frames` so the stride
            # is 1, and every one of those 50 is in the film: 8 beats over the run at 20 fps,
            # two and a half seconds of screen time each.
            "dot_shading": True, "max_frames": n_frames, "stills": 6, "keep_stills": True,
            "fps": 20.0, "camera_roll": 180.0, "dot_size": 7.0,
            # THE VIZ. Every soma coloured by its own membrane state, on a fixed range so two
            # frames of the run can be compared -- an autoscaled range makes a quiet moment look
            # exactly like a loud one.
            # THE RANGE IS THE ONE THE RUN ACTUALLY OCCUPIES. +-1.5 is the range the membrane
            # law could reach and not the one it does: measured over the last sixth of a run, the
            # wired population sits at |v| 0.25 and the loudest class, the ciliary band, at 0.58,
            # so +-1.5 renders the whole animal as almost-white and the activity is invisible.
            # Fixed and not autoscaled, so two frames stay comparable.
            "color_field": "voltage", "field_cmap": "coolwarm", "color_range": [-0.8, 0.8],
            "static_mesh": {
                "file": f"neural_regions/{REGION}/body_outline.obj",
                "color": "#aeb6c2", "opacity": 0.08,
                "to_world": {"origin": f["lo_nm"], "scale": 1.0 / f["side_nm"]},
            },
        },
    }


def spec_r5(f: dict, scale: float = 0.85, offset=(0.08, 0.08, 0.08),
            per_cell: int = 24, soma_um: float = 1.9, n_frames: int = 600) -> dict:
    """R5: the motor pathway -- the circuit's own output makes the ciliary band beat.

    THE PATHWAY IS ALREADY IN THE DATA. The connectome carries 255 motoneuron-to-ciliary-band
    edges, including the single MC cell contacting all 23 prototroch cells across 340 synapses,
    so the ciliary cells' membrane state IS the motor command -- R4 measured them as the loudest
    class in the animal at |v| 0.58, because they are the network's sink. No junction had to be
    invented and no edge file written: `neuron_signal` already delivers it.

    WHAT WAS MISSING was the last step, and it was missing from both repositories: no operator
    anywhere turned a membrane state into a mechanical quantity. `polar_active_stress[driven]`
    (src/plexus/operators/cilia_ops.py) is that step -- the existing beat with its amplitude read
    from the cell's own `voltage` block instead of from a constant in the spec.

    WHAT THIS RUNG DOES NOT CLAIM. A cilium is a slender appendage projecting from the cell, and
    that is not what is modelled here: the ciliary-band CELLS themselves extend and retract along
    their own outward radius. It is the coarsest possible reading of a ciliary stroke and it is
    the right first one, because it makes the whole chain -- connectome, membrane law, synaptic
    law, phase, polarity, active stress, MLS-MPM -- run end to end and be measured. The slender
    appendage comes after the chain is known to work.

    THE STROKE AXIS is each band cell's own outward radial direction about the animal's head-tail
    axis, written once by `radial_polarity` and ONLY on the ciliary band: every other cell keeps
    the zero polarity it was provisioned with, and a cell with no stroke axis does not stroke.
    A ring of cells all stroking the same way would push the animal sideways; a ring stroking
    outward is a girdle.
    """
    um = f["side_um"]
    r = soma_um / um
    pm = 1050.0 * (4.0 / 3.0) * math.pi * r ** 3 / per_cell
    start = positions(f, scale, list(offset))
    types = {c: {"count": f["count"][c], "shape": "ball", "material": "elastic",
                 "youngs": MATERIAL[c][0], "density": MATERIAL[c][1],
                 "p": P_CLASS.get(c, P_NEURON)} for c in f["order"]}
    return {
        "general": {
            "name": "plat_r5_beat", "seed": 0, "n_frames": n_frames, "dt": 0.05, "dim": 3,
            "world": [1.0, 1.0, 1.0], "boundary": "wall", "save_data": True,
            "record_cap": n_frames + 1, "units": {"length_um": um, "time_s": 1.0},
        },
        "sets": {
            # A ROOT FOR THE EDGE SET TO HANG FROM. `cell` is parentless because its 4,117
            # positions are declared outright, and an edge set still needs a parent -- `parent:
            # null` is refused, which is right: every set belongs somewhere in the containment map.
            "brain": {"n": 1},
            # THE BODY. `pos` is its coordinate and the material points hang off it.
            "cell": {
                "n": f["n"], "start": start, "type_layout": "ordered", "types": types,
                "state": {
                    "pos": {"width": 3, "role": "coordinate",
                            "integration": "second_order_coordinate", "boundary": "world"},
                    "vel": {"width": 3, "role": "rate", "integration": "second_order_rate",
                            "boundary": "free"},
                    "phase": {"width": 1, "integration": "first_order", "boundary": "free"},
                    "polarity": {"width": 3, "integration": "none", "boundary": "free",
                                 "record": False},
                },
            },
            # THE NEURON, ONE PER CELL, AND IT IS A SEPARATE SET FOR A REASON THAT IS NOT
            # BOOKKEEPING. `voltage` IS a neuron's coordinate -- the neuron entity declares it
            # `role: coordinate` and `pos` as `integration: none`, so the engine integrates the
            # membrane state and leaves the position alone. A body's coordinate is `pos`. One set
            # cannot have two, and declaring `voltage` by hand beside an integrated `pos`
            # produced a membrane state that sat at exactly 0.0000 for 600 frames while every
            # operator ran and nothing complained. Containment joins them: neuron i lives in cell
            # i, `per_parent: 1`, and `polar_active_stress[driven]` reads the drive across it.
            # ONE TYPE, AND THAT IS A REAL LOSS, STATED. A child set's type counts are PER
            # PARENT, so a set with `per_parent: 1` can carry exactly one type and the 18 classes
            # cannot be spelled here. The alternative was a flat neuron set beside the cells,
            # corresponding to them BY ROW INDEX and nothing else -- an undeclared correspondence
            # that would break silently the first time either set was reordered. Containment is
            # the correspondence the language has for this, so it is the one used, and the
            # per-class parameter vectors go. They were never fitted: R4's 18 classes differed
            # only where this file chose to make them differ, so what is lost is a guess.
            "neuron": {"parent": "cell", "per_parent": 1, "entity": "neuron",
                       "types": {"cell": {"fraction": 1.0, "p": P_NEURON}}},
            "synapse": {"parent": "brain", "edge_set": True, "entity": "connection",
                        "pre": "neuron", "post": "neuron",
                        "edges_file": f"neural_regions/{REGION}/connectome.npz"},
            "mpm_particle": {"parent": "cell", "per_parent": per_cell, "density": 1050.0,
                             "radius": round(r, 6), "particle_mass": float(f"{pm:.4g}")},
        },
        "seed": [
            {"op": "radial_polarity", "at": "cell[type=ciliary band]", "axis": 2,
             "centre": [0.5, 0.5, 0.0]},
            {"op": "seed_state_random", "at": "cell", "block": "phase", "lo": 0.0, "hi": 6.2832},
            # v = 0 IS A FIXED POINT OF THE WHOLE SYSTEM, so a population seeded exactly there
            # sits at it and the circuit is a picture of nothing.
            {"op": "seed_state_random", "at": "neuron", "block": "voltage", "lo": -0.5, "hi": 0.5},
        ],
        "operators": [
            {"op": "neuron_update", "at": "neuron", "model": "leaky_tanh", "noise": 0.01},
            {"op": "neuron_signal", "at": "neuron", "model": "type_pairwise",
             "edge_set": "synapse", "activation": "tanh"},
            # THE BEAT'S OWN CLOCK. Each cell carries its own angle and its own rate, so the band
            # does not beat in lockstep -- `jitter` is the spread of rates, and a girdle whose
            # cells are all exactly in phase pumps once and then fights itself.
            {"op": "phase_clock", "at": "cell", "block": "phase", "omega": 3.0, "jitter": 0.12,
             "seed": 7},
            {"op": "polar_active_stress", "at": "mpm_particle", "model": "driven",
             "cell_set": "cell", "block": "polarity", "phase_block": "phase",
             "drive_set": "neuron", "drive": "voltage",
             # THE STROKE'S SIZE, AND IT IS SMALL ON PURPOSE. `amplitude_frac` states a TARGET
             # STRAIN: the strain a stress of A produces is roughly A / (lambda + 2 mu), so 0.06
             # is a cell reaching 6% of its own length. The first take asked for 0.30 and the
             # gain multiplied it -- the band's measured drive of 0.95 against `v_scale` 0.45
             # gives 1.8 -- so the real request was a strain over 50%, and band particles ended
             # up displaced 41 um on average with one reaching 307 um, further than the animal
             # is long. `v_scale` is raised to 0.95, the drive the band actually reaches, so a
             # fully driven band cell asks for exactly `amplitude_frac` and not a multiple of it.
             "v0": 0.15, "v_scale": 0.95, "rectify": "relu", "gain_max": 1.5,
             "amplitude_frac": 0.06, "deviatoric": True},
            # AN ANCHOR, BECAUSE THIS ANIMAL HAS NO ADHESION AND NO MATRIX. Its 4,117 cells are
            # separate elastic balls coupled only through contact on the shared grid, and R2
            # already measured what that costs: landing on a floor squashed the body to 55% of
            # its height while each cell kept 97% of its own radius. With gravity off there is
            # nothing at all holding it, and the first take of this rung dispersed the whole
            # animal until it filled the 196 um box on every axis within 150 frames.
            #
            # The spring is a SCAFFOLD standing in for the cell-cell adhesion and extracellular
            # matrix the model does not yet have, and it is named as one rather than tuned until
            # the picture looks right. k = 400 per second squared gives a restoring time of
            # 1/sqrt(k) = 50 ms, short against the beat's own 2 s period, so the body is held
            # while the stroke is not.
            {"op": "mpm_anchor", "at": "mpm_particle", "k": 400.0},
            {"op": "mpm_strain", "at": "mpm_particle", "implementation": "warp"},
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid", "drag": 0.0,
             "a_max": 200.0, "implementation": "warp", "polar": "higham"},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 0.9, "wall_friction": 0.3},
            {"op": "mpm_gather", "at": "body_point", "from": "mpm_grid", "wall_damp": 1.0,
             "vmax": 1.0e9, "implementation": "warp"},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "vmax": 1.0e9, "implementation": "warp"},
            # THE CELL'S POSITION IS ITS MATERIAL'S CENTRE OF MASS, and without this the cell set
            # never moves at all. The MPM moves the material POINTS; a parent's `pos` is only its
            # seed position unless something aggregates it, so the first take measured the cells'
            # radial excursion as exactly 0.0000 um for all 4,117 while their particles were
            # travelling tens of micrometres. It also matters for the run and not only for the
            # measurement: `radial_polarity` is a seed, but anything later that reads a cell's
            # position -- a neighbour relation, a field sample, a second animal -- would be
            # reading frame 0 forever.
            {"op": "aggregate_centroid", "at": "cell", "child": "mpm_particle"},
        ],
        # NO GRAVITY. R2 established that the animal falls; this rung is about whether the circuit
        # moves it, and a body that is also falling makes the two impossible to tell apart.
        "schedule": ["neuron_update", "neuron_signal", "phase_clock",
                     {"substep_dt": 0.005,
                      "steps": ["polar_active_stress", "mpm_anchor", "mpm_strain", "mpm_scatter",
                                "mpm_grid_update", "mpm_gather"]},
                     "aggregate_centroid"],
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 96}},
        "plotting": {
            "renderer": "vtk_points", "background": "black", "box_frame": False, "up_axis": 2,
            # EVERY FRAME KEPT, AND A BEAT THE FILM CAN RESOLVE. At omega 10 rad/s one cycle is
            # 12.6 frames and the movie's stride of 2 sampled it 6.3 times -- barely above
            # Nyquist, so a stroke that is perfectly smooth in the data reads on screen as a
            # jitter, and what you are watching is the sampling rather than the cilium. At 2.5
            # rad/s (0.40 Hz) a cycle is 50 frames, `max_frames` equals `n_frames` so the stride
            # is 1, and every one of those 50 is in the film: 8 beats over the run at 20 fps,
            # two and a half seconds of screen time each.
            "dot_shading": True, "max_frames": n_frames, "stills": 6, "keep_stills": True,
            "fps": 20.0, "camera_roll": 180.0, "dot_size": 2.5,
            "colors": {c: COLOR[c] for c in f["order"]},
        },
    }


def spec_r6(f: dict, n_water: int = 140000, n_frames: int = 600) -> dict:
    """R6: water, and whether the beat moves it.

    R5's beat is real and periodic but it happens in a vacuum. This puts the animal in a fluid and
    asks the only question that matters for a swimmer: does the stroke move anything that is not
    the animal.

    HOW THE TWO SETS TALK, and it needs no new operator. `mpm_scatter` zeroes the grid only on the
    substep's FIRST scatter (mpm_ops.py:717, tagged `shared_grid_accumulate`); every later one
    accumulates. So two particle sets that each scatter into `mpm_grid`, are solved together by
    one `mpm_grid_update`, and each gather back, exert forces on each other through the grid's
    momentum -- which is exactly how the eye prototype's six muscles move a globe, and how
    config/cell/adh_base.yaml runs fifteen sets at once. The animal pushes the water because they
    share a grid, not because anything applies a force `to the water`.

    THE WATER IS A LIQUID AND SAYS SO: `material: liquid` with a `bulk_modulus` instead of a
    Young's modulus, and `mpm_viscosity` on it. Without the viscosity a "liquid" here is a
    pressure-only gas that transmits a stroke instantly and shears not at all, and the wake the
    rung exists to see is a shear wake.

    ITS PARTICLES ARE SMALLER THAN THE ANIMAL'S. `particle_mass` is per set, so the water can be
    resolved finely without the cells being cut up: 140,000 points over the box against 98,808
    over the animal.
    """
    base = spec_r5(f, n_frames=n_frames)
    base["general"]["name"] = "plat_r6_water"
    sets = base["sets"]
    # THE POOL, as a block filling the box around the animal. The cells occupy 1.6% of the world
    # volume, so the overlap at frame 0 is small and the first substep resolves it as pressure.
    sets["water"] = {"n": 1, "start": [[0.5, 0.5, 0.5]],
                     "types": {"seawater": {"count": 1, "material": "liquid",
                                            "bulk_modulus": 20000.0, "density": 1025.0,
                                            "eta": 0.001,
                                            "block": [0.02, 0.02, 0.02, 0.98, 0.98, 0.98]}}}
    # `entity: mpm_particle` IS WHAT MAKES A SET MATERIAL POINTS. Without it the set is provided
    # with positions and nothing else -- no deformation gradient, no affine velocity -- and the
    # run dies on the first strain step with `'Level' object has no attribute 'F'`. The NAME
    # `mpm_particle` is not the trigger and was never meant to be; the entity is, which is what
    # lets one model carry fifteen material-point sets under fifteen different names.
    sets["water_particle"] = {"parent": "water", "per_parent": n_water, "entity": "mpm_particle",
                              "density": 1025.0, "radius": 0.5}
    ops = base["operators"]
    # The water's own four steps, into the SAME grid. `mpm_anchor` is deliberately not among them:
    # the animal is held to its rest shape, the water is free.
    ops += [
        {"op": "mpm_strain", "at": "water_particle", "implementation": "warp"},
        {"op": "mpm_viscosity", "at": "water_particle", "eta": 0.001},
        {"op": "mpm_scatter", "at": "water_particle", "to": "mpm_grid", "drag": 0.0,
         "a_max": 200.0, "implementation": "warp", "polar": "higham"},
        {"op": "mpm_gather", "at": "water_particle", "from": "mpm_grid", "wall_damp": 0.9,
         "vmax": 1.0e9, "implementation": "warp"},
    ]
    base["schedule"] = [
        "neuron_update", "neuron_signal", "phase_clock",
        # ONE SUBSTEP BLOCK FOR BOTH SETS. Both scatters must land between the same grid zero and
        # the same grid solve, or the second set would be solved against a grid the first had
        # already consumed -- which is a scene where the water feels the animal and the animal
        # does not feel the water.
        {"substep_dt": 0.005,
         "steps": ["polar_active_stress", "mpm_anchor", "mpm_strain", "mpm_strain",
                   "mpm_viscosity", "mpm_scatter", "mpm_scatter", "mpm_grid_update",
                   "mpm_gather", "mpm_gather"]},
        "aggregate_centroid",
    ]
    base["plotting"]["colors"]["seawater"] = "#1b3f6b"
    base["plotting"]["dot_size"] = 2.0
    # THE ANIMAL IS THE SUBJECT, SAID OUT LOUD. The renderer draws the BIGGEST particle set unless
    # told otherwise, and the pool is larger than the animal, so the first take rendered 140,000
    # water points in one hue and no larva at all.
    base["plotting"]["subject"] = "mpm_particle"
    return base


def spec_r7(f: dict, scale: float = 0.85, offset=(0.08, 0.08, 0.08), per_cell: int = 24,
            soma_um: float = 1.9, n_frames: int = 900, period: float = 8.0) -> dict:
    """R7: the rhythm comes from the NEURONS, because this connectome cannot make one itself.

    WHAT THE SPECTRUM SAID, measured in tools/platynereis_spectrum.py and not guessed. Every one
    of the 4,664 weights is a synapse COUNT and therefore non-negative, and by Perron-Frobenius a
    non-negative matrix has its spectral radius as a REAL eigenvalue. So the first mode to go
    unstable as the gain rises is real: the network LATCHES, every cell saturating together
    against the tanh, and no gain produces a beat. Measured: leading eigenvalue +0.900, real; the
    leading complex pair at 0.294 +- 0.072i needs gain 3.40, reached 2.9x later. Putting Dale
    signs back on half the interneurons does not rescue it either -- the leading mode stays real
    and the complex pair is still reached 1.8x late.

    SO THE RHYTHM IS PUT WHERE THE ANIMAL KEEPS IT. Verasztó et al. (2017), eLife 6:e26000: the
    single cholinergic MC cell fires periodically and ARRESTS the prototroch, while serotonergic
    Ser-h1 drives beating; the larva's ciliary closures come from cells that oscillate, not from
    a network that does. `neuron_pacemaker` gives the motoneuron pool an intrinsic rhythm and the
    measured connectome carries it: the question this rung answers is whether the BAND's drive
    then oscillates at the pool's frequency, which is a claim about the wiring and not about the
    pacemaker.

    TWO TIMESCALES, AND THAT IS THE BIOLOGY. A cilium beats fast and the ciliomotor circuit gates
    it slowly -- closures every several seconds over a beat of tens of hertz. So `phase_clock`
    keeps the fast stroke and the circuit supplies the slow envelope, which is what
    `polar_active_stress[driven]` reading the band's voltage already means.

    THE CONNECTOME IS THE SIGNED ONE. `connectome_dale.npz`, written by the same tool: the MC
    cell inhibitory on the paper's evidence, 30% of the other interneurons inhibitory as a STATED
    ASSUMPTION rather than a measurement. Without any inhibition the pacemaker's rhythm arrives
    at the band rectified into a constant.
    """
    um = f["side_um"]
    r = soma_um / um
    pm = 1050.0 * (4.0 / 3.0) * math.pi * r ** 3 / per_cell
    start = positions(f, scale, list(offset))
    types = {c: {"count": f["count"][c], "shape": "ball", "material": "elastic",
                 "youngs": MATERIAL[c][0], "density": MATERIAL[c][1]} for c in f["order"]}
    return {
        "general": {
            "name": "plat_r7_rhythm", "seed": 0, "n_frames": n_frames, "dt": 0.05, "dim": 3,
            "world": [1.0, 1.0, 1.0], "boundary": "wall", "save_data": True,
            "record_cap": n_frames + 1, "units": {"length_um": um, "time_s": 1.0},
        },
        "sets": {
            "brain": {"n": 1},
            "cell": {
                "n": f["n"], "start": start, "type_layout": "ordered", "types": types,
                "state": {
                    "pos": {"width": 3, "role": "coordinate",
                            "integration": "second_order_coordinate", "boundary": "world"},
                    "vel": {"width": 3, "role": "rate", "integration": "second_order_rate",
                            "boundary": "free"},
                    "phase": {"width": 1, "integration": "first_order", "boundary": "free"},
                    "polarity": {"width": 3, "integration": "none", "boundary": "free",
                                 "record": False},
                },
            },
            # THE NEURONS FLAT, BESIDE THE CELLS RATHER THAN INSIDE THEM, and that is forced: a
            # child set's type counts are PER PARENT, so a `per_parent: 1` neuron set carries
            # exactly one type and this rung has to name a class -- the pacemaker goes to the
            # motoneurons and not to the other 3,993 cells. Flat, the 18 classes can be declared.
            # The two sets then correspond by ROW, which is true of this dataset by construction
            # (both are built from the same region in the same order) and is checked at run time:
            # `polar_active_stress[driven]` refuses a `drive_set` of a different length.
            "neuron": {"parent": "brain", "per_parent": f["n"], "type_layout": "ordered",
                       "types": {c: {"count": f["count"][c], "p": P_CLASS.get(c, P_NEURON)}
                                 for c in f["order"]}},
            "synapse": {"parent": "brain", "edge_set": True, "entity": "connection",
                        "pre": "neuron", "post": "neuron",
                        "edges_file": f"neural_regions/{REGION}/connectome_dale.npz"},
            "mpm_particle": {"parent": "cell", "per_parent": per_cell, "density": 1050.0,
                             "radius": round(r, 6), "particle_mass": float(f"{pm:.4g}")},
        },
        "seed": [
            {"op": "neural_seed", "at": "neuron", "region": REGION, "v0_mean": 0.0, "v0_sd": 0.3,
             "scale": scale, "offset": list(offset)},
            {"op": "radial_polarity", "at": "cell[type=ciliary band]", "axis": 2,
             "centre": [0.5, 0.5, 0.0]},
            {"op": "seed_state_random", "at": "cell", "block": "phase", "lo": 0.0, "hi": 6.2832},
        ],
        "operators": [
            # THE RHYTHM, IN THE POOL THAT HAS ONE. `jitter` spreads the periods by 8%, because a
            # pool of identical pacemakers is one pacemaker with a louder voice -- they never
            # drift apart and nothing downstream can tell a population from a single cell.
            {"op": "neuron_pacemaker", "at": "neuron[type=Motoneuron]", "period": period,
             "amplitude": 1.2, "waveform": "sine", "jitter": 0.08},
            {"op": "neuron_update", "at": "neuron", "model": "leaky_tanh", "noise": 0.01},
            {"op": "neuron_signal", "at": "neuron", "model": "type_pairwise",
             "edge_set": "synapse", "activation": "tanh"},
            {"op": "phase_clock", "at": "cell", "block": "phase", "omega": 3.0, "jitter": 0.12,
             "seed": 7},
            {"op": "polar_active_stress", "at": "mpm_particle", "model": "driven",
             "cell_set": "cell", "block": "polarity", "phase_block": "phase",
             "drive_set": "neuron", "drive": "voltage",
             "v0": 0.05, "v_scale": 0.60, "rectify": "relu", "gain_max": 2.0,
             "amplitude_frac": 0.10, "deviatoric": True},
            {"op": "mpm_anchor", "at": "mpm_particle", "k": 400.0},
            {"op": "mpm_strain", "at": "mpm_particle", "implementation": "warp"},
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid", "drag": 0.0,
             "a_max": 200.0, "implementation": "warp", "polar": "higham"},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 0.9, "wall_friction": 0.3},
            {"op": "mpm_gather", "at": "body_point", "from": "mpm_grid", "wall_damp": 1.0,
             "vmax": 1.0e9, "implementation": "warp"},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "vmax": 1.0e9, "implementation": "warp"},
            {"op": "aggregate_centroid", "at": "cell", "child": "mpm_particle"},
        ],
        "schedule": ["neuron_pacemaker", "neuron_update", "neuron_signal", "phase_clock",
                     {"substep_dt": 0.005,
                      "steps": ["polar_active_stress", "mpm_anchor", "mpm_strain", "mpm_scatter",
                                "mpm_grid_update", "mpm_gather"]},
                     "aggregate_centroid"],
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 96}},
        "plotting": {
            "renderer": "vtk_points", "background": "black", "box_frame": False, "up_axis": 2,
            # EVERY FRAME KEPT, AND A BEAT THE FILM CAN RESOLVE. At omega 10 rad/s one cycle is
            # 12.6 frames and the movie's stride of 2 sampled it 6.3 times -- barely above
            # Nyquist, so a stroke that is perfectly smooth in the data reads on screen as a
            # jitter, and what you are watching is the sampling rather than the cilium. At 2.5
            # rad/s (0.40 Hz) a cycle is 50 frames, `max_frames` equals `n_frames` so the stride
            # is 1, and every one of those 50 is in the film: 8 beats over the run at 20 fps,
            # two and a half seconds of screen time each.
            "dot_shading": True, "max_frames": n_frames, "stills": 6, "keep_stills": True,
            "fps": 20.0, "camera_roll": 180.0, "dot_size": 2.5, "subject": "mpm_particle",
            "colors": {c: COLOR[c] for c in f["order"]},
        },
    }


def spec_r7w(f: dict, n_water: int = 140000, n_frames: int = 900) -> dict:
    """R7 in water: the rhythm-driven stroke, and how far it now reaches.

    R6 measured a clocked beat's reach into the fluid at 31 um. This is the same pool around the
    same animal with the same coupling -- one shared `mpm_grid` -- and the only thing changed is
    that the stroke is now gated by the pacemakers' rhythm carried through the connectome, and is
    six times larger for it. The comparison is therefore a clean one: two runs differing in the
    drive and in nothing else.
    """
    base = spec_r7(f, n_frames=n_frames)
    base["general"]["name"] = "plat_r7_water"
    sets = base["sets"]
    sets["water"] = {"n": 1, "start": [[0.5, 0.5, 0.5]],
                     "types": {"seawater": {"count": 1, "material": "liquid",
                                            "bulk_modulus": 20000.0, "density": 1025.0,
                                            "eta": 0.001,
                                            "block": [0.02, 0.02, 0.02, 0.98, 0.98, 0.98]}}}
    sets["water_particle"] = {"parent": "water", "per_parent": n_water, "entity": "mpm_particle",
                              "density": 1025.0, "radius": 0.5}
    base["operators"] += [
        {"op": "mpm_strain", "at": "water_particle", "implementation": "warp"},
        {"op": "mpm_viscosity", "at": "water_particle", "eta": 0.001},
        {"op": "mpm_scatter", "at": "water_particle", "to": "mpm_grid", "drag": 0.0,
         "a_max": 200.0, "implementation": "warp", "polar": "higham"},
        {"op": "mpm_gather", "at": "water_particle", "from": "mpm_grid", "wall_damp": 0.9,
         "vmax": 1.0e9, "implementation": "warp"},
    ]
    base["schedule"] = [
        "neuron_pacemaker", "neuron_update", "neuron_signal", "phase_clock",
        {"substep_dt": 0.005,
         "steps": ["polar_active_stress", "mpm_anchor", "mpm_strain", "mpm_strain",
                   "mpm_viscosity", "mpm_scatter", "mpm_scatter", "mpm_grid_update",
                   "mpm_gather", "mpm_gather"]},
        "aggregate_centroid",
    ]
    base["plotting"]["colors"]["seawater"] = "#1b3f6b"
    return base


def spec_r8(f: dict, wavenumber: float = 6.0, n_water: int = 140000,
            n_frames: int = 900) -> dict:
    """R8: a metachronal wave, and whether a non-reciprocal band transports fluid.

    THE TEST THIS RUNG IS. R6 and R7 are the two controls and they are already run: a clocked
    band beating in random phase moved water 0.049 um at its peak, and gating that band with a
    neural rhythm made its cells move 5.7x FURTHER while moving the water 2.3x LESS. Neither is
    swimming, and the reason is the stroke's SHAPE: `polar_active_stress` extends and retracts
    along one axis, which is reciprocal, and a reciprocal stroke moves no fluid at low Reynolds
    number whatever its amplitude (Purcell 1977).

    The only thing changed here is WHERE EACH CELL STARTS IN THE CYCLE. `metachronal_phase`
    replaces the random `phase_clock` seed with a phase that winds `wavenumber` times around the
    girdle, so at any instant one arc is extending while the arc behind it retracts and the band
    carries a travelling deformation. The individual stroke is as reciprocal as it ever was; the
    BAND is not. If the measured transport rises against R6 and R7, the difference is metachrony
    and nothing else -- same animal, same pool, same drive, same amplitude, same coupling.

    `phase_clock` still runs, because the wave has to travel: the seed sets the phase OFFSETS and
    the clock advances them all together at omega.
    """
    base = spec_r7w(f, n_water=n_water, n_frames=n_frames)
    base["general"]["name"] = "plat_r8_metachronal"
    # The seed order matters: the wave is written after the random phase, so it replaces it on
    # the band and leaves every other cell's phase alone.
    base["seed"] = [s for s in base["seed"]] + [
        {"op": "metachronal_phase", "at": "cell[type=ciliary band]",
         "wavenumber": wavenumber, "axis": 2, "centre": [0.5, 0.5, 0.0]},
    ]
    return base


def spec_r9(f: dict, wavenumber: float = 6.0, n_water: int = 140000,
            n_frames: int = 900) -> dict:
    """R9: a TANGENTIAL stroke under the metachronal wave -- the direction a cilium sweeps.

    THREE CONTROLS ARE ALREADY RUN AND ALL THREE FAILED TO SWIM.

      R6  random phase, clocked drive, radial stroke    peak 0.049 um, reach 31.1 um
      R7  random phase, rhythm-gated, radial stroke     peak 0.021 um, reach 25.7 um
      R8  METACHRONAL wave,            radial stroke    peak 0.020 um, reach 25.7 um

    R8 is the informative one: adding the wave changed nothing at all. A radial stroke extends
    and retracts along the same outward line, so every cell pushes fluid out and pulls it
    straight back, and winding the phase around the girdle changes only WHEN each cell pushes --
    never in which direction. Metachrony cannot rescue a stroke pointed the wrong way.

    A ciliary power stroke does not point outward, it SWEEPS along the surface. `direction:
    tangential` sets each band cell's stroke axis to the cross product of the body axis with its
    own outward radius, which is tangent to the girdle; a travelling wave of tangential strokes
    then carries fluid around the ring the way a peristaltic wave carries it down a tube. This is
    the one change from R8: the same animal, the same pool, the same rhythm, the same wavenumber,
    the same amplitude.
    """
    base = spec_r8(f, wavenumber=wavenumber, n_water=n_water, n_frames=n_frames)
    base["general"]["name"] = "plat_r9_tangential"
    for op in base["seed"]:
        if op.get("op") == "radial_polarity":
            op["direction"] = "tangential"
    return base


def spec_r10(f: dict, soma_um: float = 4.0, per_cell: int = 48, n_frames: int = 600) -> dict:
    """R10: cells that actually FILL the animal, and no anchor holding it up.

    WHY EVERY RUNG SO FAR NEEDED A SCAFFOLD. At a soma radius of 1.9 um the 4,117 cells occupy
    6.0% of the animal's bounding box -- measured, not estimated. That is not a tissue, it is a
    sparse cloud of balls in the shape of a larva, and it behaves like one: R2 dropped it and the
    body squashed to 55% of its height while every cell kept 97% of its own radius, and R5 onward
    could only stand it up with an `mpm_anchor` pinning every material point to where it started.
    An anchored animal cannot swim by construction, which is why R6-R9 compared four stroke
    designs and found them indistinguishable.
    
    The fix is not an adhesion operator. It is that 1.9 um was a DRAWING choice, taken from a
    `soma_radius` field that is the constant 2000 nm for all 4,117 cells -- a placeholder the
    region builder wrote, not a measurement. At 4.0 um the same cells fill 63% of the bounding
    box, which for a body that is not a box is most of its actual volume: they touch, the MLS-MPM
    continuum is continuous, and the elasticity that was always there holds the animal together
    with nothing pinning it. Platynereis cells are 3-6 um across, so this is the more defensible
    number as well as the one that works.

    48 points per cell and not 24, because the points must still resolve the grid: a 4.0 um ball
    holding 48 points spaces them 1.8 um apart against a grid cell of 2.04 um, so every grid cell
    that the animal covers sees at least one particle.

    NO WATER HERE. This rung asks one question -- does the body hold itself up -- and a pool
    around it would both slow the run and give the answer somewhere to hide.
    """
    um = f["side_um"]
    r = soma_um / um
    pm = 1050.0 * (4.0 / 3.0) * math.pi * r ** 3 / per_cell
    base = spec_r7(f, per_cell=per_cell, soma_um=soma_um, n_frames=n_frames)
    base["general"]["name"] = "plat_r10_tissue"
    base["sets"]["mpm_particle"] = {"parent": "cell", "per_parent": per_cell, "density": 1050.0,
                                    "radius": round(r, 6), "particle_mass": float(f"{pm:.4g}")}
    # THE ANCHOR GOES. That is the whole experiment.
    base["operators"] = [o for o in base["operators"] if o.get("op") != "mpm_anchor"]
    base["schedule"] = ["neuron_pacemaker", "neuron_update", "neuron_signal", "phase_clock",
                        {"substep_dt": 0.005,
                         "steps": ["polar_active_stress", "mpm_strain", "mpm_scatter",
                                   "mpm_grid_update", "mpm_gather"]},
                        "aggregate_centroid"]
    for op in base["seed"]:
        if op.get("op") == "radial_polarity":
            op["direction"] = "tangential"
    base["seed"] = list(base["seed"]) + [
        {"op": "metachronal_phase", "at": "cell[type=ciliary band]", "wavenumber": 6.0,
         "axis": 2, "centre": [0.5, 0.5, 0.0]}]
    return base


def spec_r11(f: dict, n_water: int = 120000, n_frames: int = 900) -> dict:
    """R11: the free animal in water -- the first run in which swimming is even possible.

    R10 took the anchor out: with its cells filling it at 4.0 um the animal holds its own shape,
    100.0% of its extent over 600 unanchored frames. Every earlier water run pinned every
    material point to where it started, so the only thing that could ever have been measured was
    local stirring. This one can move.

    THE MEASUREMENT CHANGES WITH IT. `tools/platynereis_flow.py` scores water by distance from
    the band, which is the right question for a stirrer; for a swimmer the question is where the
    ANIMAL went, and `tools/platynereis_swim.py` asks that -- the displacement of the body's own
    centre of mass, against the three things that would fake it (the pool's initial pressure
    equilibration, a drift shared with the water, and the walls).
    """
    base = spec_r10(f, n_frames=n_frames)
    base["general"]["name"] = "plat_r11_swim"
    base["sets"]["water"] = {"n": 1, "start": [[0.5, 0.5, 0.5]],
                             "types": {"seawater": {"count": 1, "material": "liquid",
                                                    "bulk_modulus": 20000.0, "density": 1025.0,
                                                    "eta": 0.001,
                                                    "block": [0.02, 0.02, 0.02,
                                                              0.98, 0.98, 0.98]}}}
    base["sets"]["water_particle"] = {"parent": "water", "per_parent": n_water,
                                      "entity": "mpm_particle", "density": 1025.0, "radius": 0.5}
    base["operators"] += [
        {"op": "mpm_strain", "at": "water_particle", "implementation": "warp"},
        {"op": "mpm_viscosity", "at": "water_particle", "eta": 0.001},
        {"op": "mpm_scatter", "at": "water_particle", "to": "mpm_grid", "drag": 0.0,
         "a_max": 200.0, "implementation": "warp", "polar": "higham"},
        {"op": "mpm_gather", "at": "water_particle", "from": "mpm_grid", "wall_damp": 0.9,
         "vmax": 1.0e9, "implementation": "warp"},
    ]
    base["schedule"] = [
        "neuron_pacemaker", "neuron_update", "neuron_signal", "phase_clock",
        {"substep_dt": 0.005,
         "steps": ["polar_active_stress", "mpm_strain", "mpm_strain", "mpm_viscosity",
                   "mpm_scatter", "mpm_scatter", "mpm_grid_update", "mpm_gather", "mpm_gather"]},
        "aggregate_centroid",
    ]
    base["plotting"]["colors"]["seawater"] = "#1b3f6b"
    base["plotting"]["subject"] = "mpm_particle"
    return base


def spec_r12(f: dict, n_water: int = 140000, n_frames: int = 400, per_cilium: int = 20,
             cilium_um: float = 38.0, soma_um: float = 4.0, per_cell: int = 48,
             sweep_deg: float = 45.0, omega: float = 2.5,
             wavenumber: float = 1.0, body_youngs: float = 12000.0) -> dict:
    """R12: TRUE CILIA, driven open-loop, in water. Does a real appendage move the fluid?

    Every rung up to here made a ciliary-band CELL swell and shrink. That is not a cilium. A
    cilium is a slender shaft rooted in a cell, several times longer than the cell itself, and it
    beats by SWINGING -- which is why the source video's chaetae look like bristles and nothing
    in these specs did.

    Here each band cell grows a real one: a 38 um shaft of 20 material points rooted at the cell's
    surface, on a 4 um cell. A Platynereis prototroch cilium is some 20-40 um long, so the
    appendage is nearly TEN TIMES the cell that bears it -- and that ratio is the whole point,
    because the tip's speed is the angular rate times the length and it is the tip that does the
    work on the fluid. The points are spaced 1.9 um apart against a grid cell of 2.04 um, so the
    shaft is resolved along its whole length rather than being a line of isolated lumps.

    WHAT IS MADE OF MATTER. The body, the yolk, the cilia and the water. The body is the 4,112
    non-yolk cells at eight points each -- the size R10 measured as the one at which cells touch
    and the animal holds its own shape -- because a cilium rooted at a band cell needs a SURFACE
    under it. The first take made only the yolk out of matter and the picture showed why that
    fails: the five yolk cells span 15 x 12 x 19 um in the middle of the animal while the band
    spans 97 x 115 x 128 um at its surface, a median 54.5 um apart, so the shafts floated in
    open water with nothing beneath them.

    THE DRIVE IS PRESCRIBED, NOT WIRED, AND THAT IS THE POINT OF THIS RUNG. The circuit is left
    out entirely and every cilium is given `drive = 1`. If the water does not move with the
    effector commanded flat out and nothing else in the way, then wiring a connectome to it would
    only be adding an explanation for a thing that is not happening. The connectome goes back in
    at the next rung, and then the comparison is against THIS.

    NO GRAVITY, for the reason R5 had none: a body that is also falling makes it impossible to
    say whether the beat moved anything.

    The two stages are the eye's, per cilium:
        cilium.drive + cilium.phase --(cilium_pose_map)--> pose_target      a commanded angle
        pose_target                 --(organ_mechanics)--> pose             a damped plant
        pose                        --(cilium_kinematics)--> the shaft's pos AND vel
    and the shaft shares the water's `mpm_grid`, which is what carries the momentum across.
    """
    um = f["side_um"]
    r = soma_um / um
    pm = 1050.0 * (4.0 / 3.0) * math.pi * r ** 3 / per_cell
    start = positions(f, 0.85, [0.08, 0.08, 0.08])
    types = {c: {"count": f["count"][c], "shape": "ball", "material": "elastic",
                 "youngs": MATERIAL[c][0], "density": MATERIAL[c][1]} for c in f["order"]}
    # A CILIUM POINT'S MASS. The shaft is prescribed kinematically, so its mass never enters its
    # own motion -- but it is exactly what it hands the grid, and therefore the water. A shaft of
    # 20 um and 0.25 um radius at the tissue's own density is the honest number.
    cil_vol = math.pi * (0.25 / um) ** 2 * (cilium_um / um)
    cil_pm = 1050.0 * cil_vol / per_cilium
    return {
        "general": {
            "name": "plat_r12_cilia", "seed": 0, "n_frames": n_frames, "dt": 0.05, "dim": 3,
            "world": [1.0, 1.0, 1.0], "boundary": "wall", "save_data": True,
            "record_cap": n_frames + 1, "units": {"length_um": um, "time_s": 1.0},
        },
        "sets": {
            "cell": {
                "n": f["n"], "start": start, "type_layout": "ordered", "types": types,
                "state": {
                    "pos": {"width": 3, "role": "coordinate",
                            "integration": "second_order_coordinate", "boundary": "world"},
                    "vel": {"width": 3, "role": "rate", "integration": "second_order_rate",
                            "boundary": "free"},
                    "phase": {"width": 1, "integration": "first_order", "boundary": "free"},
                    "polarity": {"width": 3, "integration": "none", "boundary": "free",
                                 "record": False},
                },
            },
            # THE BODY, AND THE YOLK INSIDE IT, AS TWO SETS.
            #
            # The first take made only the yolk out of matter, and the picture showed why that
            # cannot work: the five yolk cells span 15 x 12 x 19 um and sit in the middle of the
            # animal, while the ciliary band spans 97 x 115 x 128 um at its SURFACE. The median
            # band cell is 54.5 um from the nearest yolk cell, so a 12 um yolk ball leaves a 43 um
            # gap and the cilia float in open water with nothing under them. The yolk is an
            # internal mass; it is not the body.
            #
            # So the body is the CELLS -- all 4,112 of the non-yolk ones, eight material points
            # each at a 4.0 um radius, which R10 measured as the size at which they touch and the
            # animal holds its own shape. 32,896 points for a body, against the 197,616 a
            # 48-point version cost, because this rung needs the cilia to have a surface to sit
            # on and not a finite-element study of the tissue.
            "body_point": {"parent": "cell", "entity": "mpm_particle", "density": 1050.0,
                           "radius": round(4.0 / um, 6),
                           # STIFFER THAN THE TISSUE'S NOMINAL 2 kPa, and the reason is the
                           # cilia. 74 shafts stirring at hundreds of micrometres per second
                           # deliver real momentum into the shared grid, and a body at 2 kPa
                           # cannot take it: measured, its bounding volume grew 2.94x over 400
                           # frames while the zero-sweep control held at exactly 1.00. 12 kPa
                           # raises the sound speed sqrt(E/rho) from 1.4 to 3.4 world units per
                           # second, so the body transmits the load instead of being torn by it.
                           "types": {"body": {"fraction": 1.0, "material": "elastic",
                                              "youngs": body_youngs, "density": 1050.0}},
                           "per_parent": {c: (0 if c == "yolk" else 8) for c in f["order"]}},
            # The yolk keeps its own set because it is a different thing: five big cells, 400
            # points each at 12 um, dense and soft. Its own set is also what lets it keep its own
            # colour while the rest of the animal is white.
            "mpm_particle": {"parent": "cell", "entity": "mpm_particle", "density": 1150.0,
                             "radius": round(12.0 / um, 6),
                             "types": {"yolk_mass": {"fraction": 1.0, "material": "elastic",
                                                     "youngs": 400.0, "density": 1150.0}},
                             "per_parent": {c: (400 if c == "yolk" else 0) for c in f["order"]}},
            # A CILIUM EXISTS ONLY WHERE A CELL BEARS ONE. With `per_parent: 1` the set held
            # 4,117 cilia of which 74 were real, and their 48,516 dead shaft points still
            # scattered into the MPM grid as lumps of matter at every cell in the animal --
            # invisible in the picture and pushing the water all the same. The per-type mapping
            # makes them not exist.
            "cilium": {
                "parent": "cell", "entity": "organ",
                "per_parent": {c: (1 if c == "ciliary band" else 0) for c in f["order"]},
                "state": {
                    # A PLACE AS WELL AS AN ANGLE. The `organ` entity carries `pose` and nothing
                    # spatial -- an eye's coordinate is where it LOOKS, not where it is -- but a
                    # cilium's shaft has to hang off something, and the child set is scattered
                    # about its parent's `pos` at build. Integrated by nothing: the cilium sits
                    # where its cell is, and `cilium_seed` writes it there.
                    "pos": {"width": 3, "role": "coordinate", "integration": "none",
                            "boundary": "free"},
                    "pose": {"width": 3, "role": "coordinate",
                             "integration": "second_order_coordinate", "boundary": "free"},
                    "pose_rate": {"width": 3, "role": "rate", "integration": "second_order_rate",
                                  "boundary": "free"},
                    "pose_target": {"width": 3, "role": "readout", "integration": "none",
                                    "boundary": "free"},
                    "drive": {"width": 1, "role": "readout", "integration": "none",
                              "boundary": "free"},
                    # AN OFFSET, NOT A STATE. `organ_mechanics` integrates `pose` at second
                    # order and a set carries one integrator, so a first-order `phase_clock`
                    # beside it is refused. The offset is written once by `metachronal_phase` and
                    # `cilium_pose_map` advances the cycle from the run's own clock.
                    "phase": {"width": 1, "integration": "none", "boundary": "free"},
                },
            },
            "cilium_point": {"parent": "cilium", "per_parent": per_cilium,
                             "entity": "mpm_particle", "density": 1050.0,
                             # The shaft is driven KINEMATICALLY, so its own elasticity never
                             # enters its motion -- `cilium_kinematics` overwrites pos and vel
                             # every substep and its deformation gradient stays a rotation, which
                             # carries no stress. The modulus is here because `mpm_scatter`
                             # requires one, and it is the tissue's so the number is not a
                             # stranger to the scene.
                             "types": {"cilium_shaft": {"fraction": 1.0, "material": "elastic",
                                                        "youngs": 2000.0, "density": 1050.0}},
                             "radius": round(0.25 / um, 6),
                             "particle_mass": float(f"{cil_pm:.4g}")},
            "water": {"n": 1, "start": [[0.5, 0.5, 0.5]],
                      "types": {"seawater": {"count": 1, "material": "liquid",
                                             "bulk_modulus": 20000.0, "density": 1025.0,
                                             "eta": 0.001,
                                             "block": [0.02, 0.02, 0.02, 0.98, 0.98, 0.98]}}},
            "water_particle": {"parent": "water", "per_parent": n_water,
                               "entity": "mpm_particle", "density": 1025.0, "radius": 0.5},
        },
        "seed": [
            # RADIAL, because this is the direction the cilium POINTS, not the direction it
            # sweeps. The sweep is a rotation ABOUT an axis perpendicular to the shaft, and
            # `cilium_seed` computes that axis itself from the body's geometry.
            {"op": "radial_polarity", "at": "cell[type=ciliary band]", "axis": 2,
             "centre": [0.5, 0.5, 0.0], "direction": "radial"},
            {"op": "cilium_seed", "at": "cilium_point", "cilium_set": "cilium",
             "cell_set": "cell", "length_um": cilium_um, "soma_um": soma_um,
             "beat": "tangential", "axis": 2},
            # THE PRESCRIBED DRIVE. Every cilium flat out, so this rung tests the mechanism and
            # not a circuit's ability to reach it.
            {"op": "seed_state_random", "at": "cilium", "block": "drive", "lo": 1.0, "hi": 1.0},
            # WAVENUMBER 1, NOT 6, AND THAT IS A SAMPLING LIMIT RATHER THAN A TASTE.
            #
            # A metachronal wave needs NEIGHBOURING cilia close in phase; past about 90 degrees
            # per step the ring is aliased and adjacent shafts sit in antiphase, which on screen
            # is indistinguishable from random. The girdles are small: the metatroch has 8 cells
            # and the akrotroch 8, so at k = 6 the metatroch steps 160 degrees between
            # neighbours and the prototroch 59. At k = 1 the steps are 5 to 27 degrees across
            # every girdle and the wave is a wave.
            {"op": "metachronal_phase", "at": "cilium", "wavenumber": wavenumber, "axis": 2,
             "centre": [0.5, 0.5, 0.0], "block": "phase"},
            # THE WATER THE ANIMAL DISPLACES. A block of water does not know a body is standing
            # in it, so a uniform seed puts thousands of particles inside the tissue -- and
            # MLS-MPM resolves that as a first-frame pressure explosion that every later
            # measurement of "how far the water moved" would be reading instead of the beat.
            {"op": "exclude_overlap", "at": "water_particle",
             "inside": ["body_point", "mpm_particle", "cilium_point"], "res": 96, "dilate": 1},
        ],
        "operators": [
            {"op": "cilium_pose_map", "at": "cilium", "sweep_deg": sweep_deg, "omega": omega,
             "waveform": "stroke", "duty": 0.3, "d0": 0.0, "d_scale": 1.0},
            # STAGE TWO IS THE EYE'S OWN PLANT, unchanged. K and C are per second squared and
            # per second: sqrt(eig K) = 40 rad/s is a corner four times the beat's own 10 rad/s,
            # so the shaft FOLLOWS its command rather than being low-passed out of existence --
            # which is the thing to get right before asking whether the water moved, because a
            # plant that cannot track its input makes every amplitude in the spec a fiction. The
            # damping ratio C / (2 sqrt K) = 0.75 keeps it from ringing on the stroke's corner.
            {"op": "organ_mechanics", "at": "cilium",
             "K": [[1600.0, 0, 0], [0, 1600.0, 0], [0, 0, 1600.0]],
             "C": [[60.0, 0, 0], [0, 60.0, 0], [0, 0, 60.0]]},
            {"op": "cilium_kinematics", "at": "cilium_point", "cilium_set": "cilium",
             "cell_set": "cell"},
            {"op": "mpm_strain", "at": "body_point", "implementation": "warp"},
            {"op": "mpm_strain", "at": "mpm_particle", "implementation": "warp"},
            {"op": "mpm_scatter", "at": "body_point", "to": "mpm_grid", "drag": 0.0,
             "a_max": 200.0, "implementation": "warp", "polar": "higham"},
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid", "drag": 0.0,
             "a_max": 200.0, "implementation": "warp", "polar": "higham"},
            {"op": "mpm_scatter", "at": "cilium_point", "to": "mpm_grid", "drag": 0.0,
             "a_max": 200.0, "implementation": "warp", "polar": "higham"},
            {"op": "mpm_strain", "at": "water_particle", "implementation": "warp"},
            {"op": "mpm_viscosity", "at": "water_particle", "eta": 0.001},
            {"op": "mpm_scatter", "at": "water_particle", "to": "mpm_grid", "drag": 0.0,
             "a_max": 200.0, "implementation": "warp", "polar": "higham"},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 0.9, "wall_friction": 0.3},
            {"op": "mpm_gather", "at": "body_point", "from": "mpm_grid", "wall_damp": 1.0,
             "vmax": 1.0e9, "implementation": "warp"},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "vmax": 1.0e9, "implementation": "warp"},
            {"op": "mpm_gather", "at": "water_particle", "from": "mpm_grid", "wall_damp": 0.9,
             "vmax": 1.0e9, "implementation": "warp"},
            # THE CELL FOLLOWS ITS OWN MATTER, without which the anchor above is anchored to a
            # frame-0 ghost: the MPM moves the body's POINTS and a parent keeps its seed position
            # unless something aggregates it.
            {"op": "aggregate_centroid", "at": "cell", "child": "body_point"},
        ],
        # NO GRAVITY. And no `mpm_gather` on the shaft: it is kinematic, its motion is its
        # circuit's, and reading the grid back into it would be the fluid pushing the cilium
        # around -- a different model, and one this rung is not testing.
        "schedule": ["cilium_pose_map", "organ_mechanics",
                     {"substep_dt": 0.005,
                      "steps": ["cilium_kinematics", "mpm_strain", "mpm_strain", "mpm_strain",
                                "mpm_viscosity", "mpm_scatter", "mpm_scatter", "mpm_scatter",
                                "mpm_scatter", "mpm_grid_update", "mpm_gather", "mpm_gather",
                                "mpm_gather"]},
                     "aggregate_centroid"],
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": 96}},
        "plotting": {
            "renderer": "vtk_points", "background": "black", "box_frame": True, "up_axis": 2,
            # EVERY FRAME KEPT, AND A BEAT THE FILM CAN RESOLVE. At omega 10 rad/s one cycle is
            # 12.6 frames and the movie's stride of 2 sampled it 6.3 times -- barely above
            # Nyquist, so a stroke that is perfectly smooth in the data reads on screen as a
            # jitter, and what you are watching is the sampling rather than the cilium. At 2.5
            # rad/s (0.40 Hz) a cycle is 50 frames, `max_frames` equals `n_frames` so the stride
            # is 1, and every one of those 50 is in the film: 8 beats over the run at 20 fps,
            # two and a half seconds of screen time each.
            "dot_shading": True, "max_frames": n_frames, "stills": 6, "keep_stills": True,
            "fps": 20.0, "camera_roll": 180.0,
            # THE WATER IS THE CLOUD, AT A DOT SIZE YOU CAN ACTUALLY SEE. Only one set can be
            # the subject and the subject is what gets drawn as points, so the water takes it --
            # it is the thing whose motion this rung is about, and at 1.1 px it read as a faint
            # haze. The body and the cilia are glyphs over it: the body gold and nearly
            # transparent so it shows its silhouette without hiding the water behind it, the
            # cilia white and solid because they are what is being watched.
            #
            # THE YOLK IS NOT DRAWN. It is still MATTER -- it scatters into the grid and
            # displaces water like everything else -- but a type with no `dot_radius` is not
            # drawn at all.
            "subject": "water_particle", "dot_size": 2.8,
            "dot_opacity": {"cilium_shaft": 1.0, "body": 0.14},
            "dot_radius": {"cilium_shaft": round(0.9 / um, 6),
                           "body": round(0.9 / um, 6)},
            "colors": dict({c: COLOR[c] for c in f["order"]},
                           seawater="#3d6ea8", cilium_shaft="#ffffff",
                           body="#d8b45c", yolk_mass="#e8c33a"),
        },
    }


def spec_r13(f: dict, n_frames: int = 400, torque: float = 3.0e-4, omega: float = 2.5) -> dict:
    """R13: the cilium feels the fluid. An elastic shaft driven by a TORQUE, momentum conserved.

    R12 drove each shaft kinematically -- `cilium_kinematics` overwrote its position AND velocity
    every substep -- and that is not a model of an actuator, it is momentum appearing from
    nowhere. Measured on the trajectory: the scene's total momentum ran 0.58 -> 45.9 -> 2.4 from
    a standing start, the body's momentum sat 9 degrees from the water's at frame 100 instead of
    180, the body inflated to 168% of its own radius rather than being propelled, and stiffening
    it six-fold changed the blow-up by 4% because stiffness cannot absorb a source.

    HERE THE SHAFT IS ORDINARY ELASTIC MATTER. It has `mpm_strain`, it scatters, it gathers, the
    fluid pushes back on it and its root is held by the same grid that holds the body. The only
    thing added is a PURE COUPLE at the root -- forces that sum to zero by construction, so a
    driven shaft cannot accelerate its own centre of mass however hard it is driven -- and the
    equal and opposite couple on the body points nearest that root, so the reaction goes where a
    basal body's reaction goes. Both are exact to the last bit of float64 and tested
    (tests/test_platynereis.py).

    WHAT THIS COSTS, and it is worth stating before the run rather than after. A real cilium is
    0.25 um thick and the grid cell here is 2.04 um: MLS-MPM cannot hold a body thinner than its
    own cell, so the shaft is widened to 1.2 um radius -- 2.4 um across, a little over one cell.
    That is a fatter cilium than the animal's, and the alternative is a finer grid at eight times
    the cost per halving. The shaft is also SOFT (2 kPa) so that a torque can actually bend it;
    a stiff rod would rotate rigidly and the stroke's shape would be the command's, not the
    fluid's.

    `organ_mechanics` is gone. The plant IS the shaft's own dynamics now -- its inertia, its
    elasticity and the water's drag -- so there is no second-order model of a body travelling to
    a commanded angle, because the body is there.
    """
    base = spec_r12(f, n_frames=n_frames, omega=omega)
    base["general"]["name"] = "plat_r13_torque"
    sets = base["sets"]
    # WIDE ENOUGH FOR THE GRID TO HOLD IT, and soft enough for a torque to bend it.
    sets["cilium_point"]["radius"] = round(1.2 / f["side_um"], 6)
    sets["cilium_point"]["types"]["cilium_shaft"]["youngs"] = 2000.0
    sets["cilium_point"].pop("particle_mass", None)

    ops = [o for o in base["operators"]
           if o.get("op") not in ("cilium_kinematics", "organ_mechanics")]
    # the shaft is matter: it strains, and it takes the grid's answer back
    ops = ([{"op": "cilium_pose_map", "at": "cilium", "sweep_deg": torque, "omega": omega,
             "waveform": "stroke", "duty": 0.3, "d0": 0.0, "d_scale": 1.0}]
           + [o for o in ops if o.get("op") != "cilium_pose_map"])
    ops += [
        {"op": "cilium_torque", "at": "cilium_point", "cilium_set": "cilium",
         "body_set": "body_point", "n_react": 24, "check": True},
        {"op": "mpm_strain", "at": "cilium_point", "implementation": "warp"},
        {"op": "mpm_gather", "at": "cilium_point", "from": "mpm_grid", "wall_damp": 0.9,
         "vmax": 1.0e9, "implementation": "warp"},
    ]
    base["operators"] = ops
    # EVERY INSTANCE MUST APPEAR, AND THE COUNT IS THE WHOLE POINT.
    #
    # The engine binds the i-th occurrence of a token in the schedule to the i-th instance of
    # that operator in the list. There are now FOUR material sets -- body, yolk, water and the
    # shafts -- so there are four `mpm_strain`, four `mpm_scatter` and four `mpm_gather`, and a
    # schedule naming three of each silently drops one set out of the physics entirely. That is
    # what happened: the shafts never scattered into the grid and never gathered from it, so they
    # had no dynamics at all, their momentum was 0.00000 at every frame, and widening them from
    # 1.2 um to 6 um changed nothing because the width was never the problem.
    #
    # Counted from the sets rather than written out, so adding a fifth material set cannot
    # reintroduce it.
    n_mat = sum(1 for k, v in base["sets"].items() if str(v.get("entity", "")) == "mpm_particle"
                or k in ("body_point", "mpm_particle", "water_particle", "cilium_point"))
    base["schedule"] = [
        "cilium_pose_map",
        {"substep_dt": 0.005,
         "steps": (["cilium_torque"] + ["mpm_strain"] * n_mat + ["mpm_viscosity"]
                   + ["mpm_scatter"] * n_mat + ["mpm_grid_update"] + ["mpm_gather"] * n_mat)},
        "aggregate_centroid",
    ]
    base["plotting"]["dot_radius"]["cilium_shaft"] = round(1.2 / f["side_um"], 6)
    return base


def spec_r14(f: dict, n_frames: int = 400, torque: float = 0.05, omega: float = 2.5,
             paddle_um: float = 6.0, omega_vis: float = 60.0) -> dict:
    """R14: an actuator the GRID CAN MOVE, and water coloured by what it is doing.

    R13 conserved momentum to 118 times better than R12 and kept the body at 101% of its own
    radius -- and its shafts had exactly 0.00000 momentum at every frame. MLS-MPM carries ONE
    velocity per grid node, so a body narrower than a cell shares every node it touches with the
    fluid and cannot move relative to it. The cell is 2.04 um; R13's shaft was 1.18 cells across
    and was simply advected. A real cilium at 0.25 um is a quarter of a cell and would need a
    2000-cube grid on this domain.

    SO THE ACTUATOR IS WIDENED UNTIL THE GRID CAN HOLD IT: 6 um radius, 5.9 cells across. That is
    not a cilium and is not called one -- it is a PADDLE, and it stands for the collective action
    of the tuft of cilia a band cell actually carries, which is the thing that moves water at this
    scale anyway. Naming it honestly is the difference between a coarse model and a wrong one.

    AND THE WATER IS COLOURED BY ITS SPEED. 125,000 blue dots are a fog whatever is happening
    behind them; on a dark colour map still water is BLACK and only moving water is visible, so
    the picture stops being a box of dots and becomes the flow itself. `color_range` is fixed so
    two frames can be compared -- an autoscaled range makes a quiet moment look like a loud one.
    """
    base = spec_r13(f, n_frames=n_frames, torque=torque, omega=omega)
    base["general"]["name"] = "plat_r14_paddle"
    um = f["side_um"]
    cp = base["sets"]["cilium_point"]
    cp["radius"] = round(paddle_um / um, 6)
    # A PADDLE HAS TO BE STIFF ENOUGH TO PUSH. At 2 kPa a 6 um shaft folds under its own drag
    # instead of sweeping; 30 kPa is still soft against chitin and holds its shape over a stroke.
    cp["types"]["cilium_shaft"]["youngs"] = 30000.0
    base["plotting"]["dot_radius"]["cilium_shaft"] = round(paddle_um / um, 6)
    # THE WATER IS THE PICTURE NOW: black where it is still, bright where it moves.
    base["plotting"]["color_field"] = "speed"
    base["plotting"]["field_cmap"] = "inferno"
    base["plotting"]["color_range"] = [0.0, 0.35]
    base["plotting"]["dot_size"] = 2.2
    base["plotting"]["colors"]["body"] = "#4a5a70"
    base["plotting"]["dot_opacity"] = {"body": 0.10, "cilium_shaft": 1.0}
    return base


def spec_r15(f: dict, n_frames: int = 400, torque: float = 0.2, omega: float = 1.25,
             n_water: int = 400000, n_grid: int = 48, dt_sub: float = 0.05 / 28,
             blade_um: float = 38.0, width_um: float = 12.0,
             n_along: int = 24, n_across: int = 9, per_body: int = 32,
             youngs: float = 12000.0) -> dict:
    """R15: a scene that is actually resolved. The body stopped breaking for arithmetic reasons.

    EVERY RUNG UP TO HERE RAN AN MLS-MPM SCENE THAT WAS NOT A CONTINUUM, and no amount of tuning
    the cilia could have fixed it. Two independent faults, both found by the repository's own
    checkers rather than by looking at pictures:

      THE TIMESTEP WAS 6.76x OVER THE COURANT LIMIT. `mpm_cfl` on plat_r14_paddle:
      substep_dt = 5.00e-03 against a limit of 7.40e-04 (c_max 5.63 world/s from the 30 kPa
      blade, dx = 1.04e-02, safety 0.4). An explicit MPM substep past Courant does not drift, it
      PUMPS: the elastic wave crosses more than a cell per step, the grid transfer re-reads its
      own overshoot, and the energy that appears has to go somewhere. It went into the body.

      EVERY BODY WAS A DUST. `particles_per_cell` on the same spec: body 0.25 particles per grid
      cell, yolk 0.47, shaft 0.19, water 0.18 -- against the ~8 MLS-MPM is written for. The
      checker's own words are "the body will lose stiffness and may fracture numerically", which
      is the observation this model has been chasing since R12 and treating as a modulus to be
      raised. It was never a modulus. A solid sampled at a quarter of a particle per cell has
      almost no cell in which the grid can assemble a stress at all, so it has no stiffness to
      lose, and stiffening it six-fold moved the blow-up by 4% exactly as measured.

    SO THIS RUNG BUYS RESOLUTION AND SPENDS THE TORQUE BUDGET NOWHERE ELSE.

      grid       96 -> 48 cells (2.04 -> 4.08 um). Eight times the particles per cell for free,
                 and it doubles the Courant limit rather than costing anything.
      substep    5.00e-03 -> 1.79e-03, 28 substeps a frame. Now BELOW the limit, which the water
                 sets (K = 20 kPa, c = 4.42) once the blade is brought down to the body's 12 kPa.
      body       8 -> 32 points a cell, 131,584 in all: 2.0 particles per cell at n_grid 48.
      yolk       400 -> 800 points a cell.
      water      140,000 -> 400,000: 4.1 particles per cell, over MLS-MPM's practical floor of 4.
      blade      20 points in a LINE -> 24 x 9 on a rectangle, 216 points over ~27 cells.

    THE SHAFT BECOMES A BLADE WITH A FACE, which is the other thing that was never true. A line
    of points is one-dimensional: it occupies a tube one cell across and drives a thread of water
    rather than a sheet. Measured on R14, its tip moved 0.8 times as far as its own ROOT -- the
    shafts were not beating at all, they were being carried. `width_um` and `n_across` lay the
    points on a rectangle spanning length x beat-axis, so the flat face meets the sweep.

    AND THE BLADE IS SOFTER, NOT STIFFER (30 -> 12 kPa, the body's own). R14 stiffened it on the
    argument that a soft blade folds under its drag. That argument was about a line, which has no
    second moment of area to resist anything; a rectangle 12 um across does. The stiffness was
    also what set c_max and therefore the Courant limit, so it was buying folding resistance with
    a timestep the run could not afford.

    WHAT IS STILL WRONG AND IS NOT FIXED HERE. The animal is 165 um in a 196 um box, so the walls
    are some 15 um off its flanks and the near field is reflecting off them. Periodic boundaries
    are not the escape: `_resolve_default_impl` (engine.py:986) refuses the warp path for a
    periodic world -- 973.8 ms a frame against 31.8 -- and these specs name `implementation: warp`
    explicitly, so a periodic world would keep the fast kernel and silently CLAMP at the wall
    instead of wrapping. A wrong answer at full speed. The box has to grow instead, and that is
    the next rung's business, after this one has shown a resolved scene behaves.
    """
    base = spec_r14(f, n_frames=n_frames, torque=torque, omega=omega)
    base["general"]["name"] = "plat_r15_resolved"
    um = f["side_um"]
    sets = base["sets"]

    # ---------------------------------------------------------------- the grid, and the timestep
    base["fields"]["mpm_grid"]["n_grid"] = n_grid
    for st in base["schedule"]:
        if isinstance(st, dict) and "substep_dt" in st:
            st["substep_dt"] = round(dt_sub, 8)

    # ---------------------------------------------------------------- sampling, per material set
    sets["body_point"]["per_parent"] = {c: (0 if c == "yolk" else per_body) for c in f["order"]}
    sets["body_point"]["types"]["body"]["youngs"] = youngs
    sets["mpm_particle"]["per_parent"] = {c: (800 if c == "yolk" else 0) for c in f["order"]}
    sets["water_particle"]["per_parent"] = n_water

    # ---------------------------------------------------------------- the blade
    per_cil = n_along * n_across
    sets["cilium_point"]["per_parent"] = per_cil
    sets["cilium_point"]["types"]["cilium_shaft"]["youngs"] = youngs
    # THE PARTICLE VOLUME IS THE BLADE'S OWN, DIVIDED BY ITS POINTS, and getting this from the
    # geometry rather than from a round number is what makes the blade fill the cells it sits in
    # instead of being a ghost in them. A blade of length x width x ONE CELL thick is the thinnest
    # slab this grid can represent, so that is the thickness, stated rather than implied.
    thick_um = um / n_grid
    v_blade = (blade_um * width_um * thick_um) / um ** 3
    sets["cilium_point"]["radius"] = round((v_blade * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0), 6)
    base["plotting"]["dot_radius"]["cilium_shaft"] = round(1.4 / um, 6)

    for o in base["seed"]:
        if o.get("op") == "cilium_seed":
            o["length_um"] = blade_um
            o["width_um"] = width_um
            o["n_across"] = n_across
        if o.get("op") == "exclude_overlap":
            o["res"] = n_grid          # the carve must be at the grid's own resolution, not 96
    for o in base["operators"]:
        if o.get("op") == "cilium_torque":
            # THE HINGE IS THE BLADE'S ROOT ROW, not the one corner that index 0 now is.
            o["root_pts"] = n_across
            # THE REACTION IS SPREAD OVER A COMPARABLE PATCH OF BODY, and `n_react` counts POINTS,
            # so quadrupling the body's sampling without touching it would have shrunk the patch
            # to under one cell's worth and concentrated the whole couple on a few points.
            o["n_react"] = 64

    # ---------------------------------------------------------------- what can be recorded
    # 400,000 water particles over 401 frames is 1.9 GB of trajectory. The stride is what the
    # `record_cap` is FOR, and at omega 1.25 rad/s a beat is 100 frames, so every fourth frame
    # still samples the stroke 25 times -- more than R12 had before the omega came down.
    base["general"]["record_cap"] = n_frames // 4 + 1
    base["plotting"]["max_frames"] = n_frames // 4
    return base


def spec_r16(f: dict, n_frames: int = 400, torque: float = 0.2, omega: float = 1.25,
             blade_um: float = 25.0, width_um: float = 10.0,
             n_along: int = 16, n_across: int = 7, v_max: float = 0.06, **kw) -> dict:
    """R16: the girdle points the right way, the blade is the size of the band it stands for.

    R15 made the scene a continuum and the numbers came right -- body and water opposing at 161
    degrees, the body holding 101% of its own radius -- and the PICTURE was still a thicket. Two
    reasons, both anatomy rather than physics, and both measurable.

    THE GIRDLE WAS POLARISED ABOUT THE WRONG AXIS. `radial_polarity` and `metachronal_phase` were
    both given `centre: [0.5, 0.5, 0.0]`, the middle of the world box -- but the animal does not
    sit in the middle of its box. Its centroid is at (0.353, 0.475, 0.487) and it spans x from
    0.100 to 0.633, so x = 0.5 is near its +x EDGE, 28.8 um off the axis it was meant to name.
    The consequence is measurable without running anything: the mean of the 74 blades' unit
    directions has magnitude 0.399 about the spec's centre and 0.010 about the animal's own. A
    girdle radiates evenly; 0.399 is most of a band pointing one way.

    Both operators already default to the level's own centroid when `centre` is absent. So the
    fix is to DELETE the parameter, not to compute a better triple and write it down -- a hard
    coded centre is a number that stops being true the moment `positions()` changes its scale or
    its margin, and nothing would say so.

    THE BLADE WAS BIGGER THAN THE BAND. It was 38 x 12 um, 74 of them, on an animal whose cells
    lie a mean 61.4 um from its centroid: 135,000 um^3 of paddle against roughly 589,000 um^3 of
    animal, 23% of the larva made of cilia. Measured on the region instead of guessed, the band's
    cells sit a median 9.9 um apart (10th percentile 6.7, 90th 16.3), and per girdle the spacing
    along the ring is 9.7 um for the paratroch, 16.0 for the prototroch, 24.4 for the metatroch
    and 31.9 for the akrotroch. So a 10 um blade TILES the band one per cell, which is what a
    ciliary band is, and 12 um was already overlapping its neighbours.

    The length comes down from 38 um to 25 um for the same reason: Platynereis prototroch cilia
    run some 20-25 um, and a 38 um blade rooted 4 um out from a cell at radius 58 um reached 96
    um on a body whose own outer radius is 70 to 93 -- a fringe half again as long as the animal
    is wide.

    AND THE COLOUR RANGE WAS FIFTEEN TIMES TOO WIDE, which is most of why the flow was invisible.
    `color_range: [0, 0.35]` is in world units per second; the water's measured median speed is
    4.487 um/s = 0.0229 world/s, so the median pixel sat at 6.5% of an inferno map, which is
    black. It is set from the measurement now: 0.06 puts the median at 38% and the 95th
    percentile (9.7 um/s) near the top.

    `cutaway` OPENS THE FLUID INSTEAD OF DRAWING ALL OF IT. 307,000 water dots are an opaque fog
    whatever colour they carry. Cutting along the view direction discards the half nearest the
    camera, so what is drawn is the flow BEHIND and AROUND the animal, seen square on, by the
    same renderer and the same palette. The animal itself is a glyph set and is not cut.
    """
    base = spec_r15(f, n_frames=n_frames, torque=torque, omega=omega,
                    blade_um=blade_um, width_um=width_um,
                    n_along=n_along, n_across=n_across, **kw)
    base["general"]["name"] = "plat_r16_girdle"

    # ---------------------------------------------------------------- the axis the girdle is on
    # DELETED, NOT CORRECTED. Both operators fall back to the level's own centroid, which follows
    # the animal wherever `positions()` puts it; a triple written here would not.
    for o in base["seed"]:
        if o.get("op") in ("radial_polarity", "metachronal_phase"):
            o.pop("centre", None)

    # ---------------------------------------------------------------- what can actually be seen
    p = base["plotting"]
    p["color_range"] = [0.0, v_max]
    p["cutaway"] = {"axis": "view", "at": 0.5}
    p["dot_opacity"] = {"body": 0.18, "cilium_shaft": 1.0}
    p["dot_size"] = 2.6
    return base


def spec_r17(f: dict, n_frames: int = 400, omega_n: float = 40.0, zeta: float = 1.0,
             n_across: int = 7, **kw) -> dict:
    """R17: the blade is attached to the animal. Eighteen of seventy-four had been flying off.

    R13 made the shaft ordinary elastic matter so the fluid could push back on it, which is what
    conserves momentum -- and in doing so it deleted `cilium_kinematics`, which had been the only
    thing holding the shaft to its cell. What was left was the hope that a root would share enough
    MLS-MPM grid nodes with the body to be dragged along by it.

    MEASURED ON plat_r16_girdle, that hope holds for most blades and fails for a fifth of them.
    The MEDIAN root stayed 3.3 um from the body point it began nearest, so the majority were fine;
    but 25 of 74 ended more than 10 um away, 18 more than 25 um, and the worst finished 232 um
    off -- clear across a 196 um box. Most broke free inside the first 30 frames. On R15 it was 11
    of 74. A blade held by node-sharing alone is held by an accident of where its root landed.

    `cilium_anchor` IS A HINGE, NOT A CLAMP. It springs the blade's root ROW to the live centroid
    of its own cell's material points, and puts the equal and opposite force on those same points,
    so sum(m a) over the pair is identically zero. Only the root's POSITION is held: the blade's
    orientation is left entirely to `cilium_torque` and the water, which is what a basal body
    does -- it says where the cilium is attached, not which way it points.

    NOT `mpm_anchor`, AND THE DIFFERENCE WOULD HAVE BEEN CATASTROPHIC. That operator springs a
    particle back to the position it was SEEDED at, a fixed point in the world. The animal swims
    44.5 um over this run, so a root pinned to its frame-0 position would drag the larva back to
    where it started and tear it apart on the way. This is the frozen-base mistake `cilium_seed`
    already made once; the fix is the same one -- store the offset, read the body live.

    `omega_n` IS WHERE THE MODELLING IS. It is the attachment's corner frequency in radians per
    second, and it has to sit well above the beat or the anchor low-passes the stroke it is meant
    to be the fulcrum of. At a beat of 1.25 rad/s, 40 rad/s is 32 times faster: over one time
    constant of the anchor the blade has swept 1.8 degrees, so the root is a hinge on the stroke's
    own timescale. Critically damped (zeta = 1) so it does not ring on the stroke's corner.
    """
    base = spec_r16(f, n_frames=n_frames, n_across=n_across, **kw)
    base["general"]["name"] = "plat_r17_anchored"
    ops = base["operators"]
    i = next(i for i, o in enumerate(ops) if o.get("op") == "cilium_torque")
    ops.insert(i + 1, {"op": "cilium_anchor", "at": "cilium_point", "cilium_set": "cilium",
                       "cell_set": "cell", "body_set": "body_point",
                       "omega_n": omega_n, "zeta": zeta, "n_root": n_across, "check": True})
    for st in base["schedule"]:
        if isinstance(st, dict) and "steps" in st:
            j = st["steps"].index("cilium_torque")
            st["steps"].insert(j + 1, "cilium_anchor")
    return base


def spec_r18(f: dict, n_frames: int = 400, grow: float = 2.0, n_grid: int = 96,
             n_water: int = 3200000, record: int = 101, **kw) -> dict:
    """R18: the same animal in a box twice as wide, to find out what the walls were worth.

    THE QUESTION THIS RUNG EXISTS TO ANSWER, and it was asked as a measurement first. The animal
    comes within about 19 um of five of its six walls, so the objection "the animal is in a small
    box" is obviously true geometrically -- but obvious is not the same as important. Measured on
    plat_r15_resolved: the water within 20 um of a wall moves at 1.39 um/s while the water within
    1.5 body radii of the animal moves at 7.46. That is 19%: the flow HAS decayed before it
    arrives, and the walls are damping what reaches them rather than reflecting it back at full
    strength. Real, not dominant, and worth one run to bound.

    PERIODIC BOUNDARIES ARE NOT THE ALTERNATIVE, and the reason is in the engine rather than in
    the physics. `_resolve_default_impl` (engine.py:986) excludes a periodic world from the warp
    path -- "the warp gather clamps at the box and does not wrap" -- which costs 973.8 ms a frame
    against 31.8. Worse, these specs name `implementation: warp` EXPLICITLY rather than leaving it
    to the default, so the exclusion would never fire: the fast kernel would run and silently
    clamp at the wall instead of wrapping. A wrong answer at full speed is the one failure mode
    worth more than a 30x slowdown, so the box grows instead.

    HOW THE BOX GROWS, since the world is always the unit cube. `length_um` doubles and the
    animal's scale halves, so the animal keeps its size IN MICROMETRES and everything denominated
    in micrometres -- the 25 x 10 um blade, the 4 um cell, the 12 um yolk -- follows it down. The
    grid doubles with the box so the cell stays 4.08 um and the sampling per cell is unchanged.
    The animal is also CENTRED now, which it never was: `positions(f, 0.85, [0.08]*3)` put its
    centroid at (0.353, 0.475, 0.487) rather than at the middle, wasting clearance on one side
    while crowding the other.

    WHAT IT COSTS, stated before the run. The grid goes 48^3 -> 96^3, eight times the cells, so
    the water goes 400,000 -> 3,200,000 to hold 4 particles per cell. And the Courant limit halves
    with dx, so the substep halves too: 1.79e-03 -> 8.93e-04, 56 a frame. Together that is about
    eleven times R15's work per frame.

    `save_data` HAS TO COME OUT OF THE SPEC FIRST. It overrides `record_cap` outright
    (engine.py:1736: "when `save_data` is given it wins: True saves EVERY frame"), which is why
    R15 wrote a 2.9 GB trajectory after being asked for every fourth frame. At 3.2M particles
    every frame would be 15 GB; `record_cap: 101` makes it every fourth and 3.9 GB.

    WHAT THIS RUNG DOES NOT CLAIM. The animal is half its former size in WORLD units while the
    moduli are unchanged in world units, so this is not a controlled twin of R17 -- it is the same
    animal, in micrometres, with more water around it. The comparison to make is the wall
    statistic, which is dimensionless, and not the swim speed.
    """
    f2 = dict(f, side_um=f["side_um"] * grow)
    # THE GRID GOES DOWN THE CHAIN, NOT ON AFTERWARDS. `spec_r15` sizes the blade's particle
    # volume as length x width x ONE CELL, so a grid set here rather than there would leave the
    # blade carrying a cell's worth of thickness from the WRONG grid -- measured, 3.73 particles
    # per cell instead of 7.5, because the slab came out twice as thick as the cells holding it.
    # The substep goes with it for the same reason: the Courant limit follows dx = 1 / n_grid.
    base = spec_r17(f2, n_frames=n_frames, n_grid=n_grid,
                    dt_sub=(0.05 / 28.0) * 48.0 / n_grid, **kw)
    base["general"]["name"] = "plat_r18_openwater"
    um = f2["side_um"]

    # THE ANIMAL, THE SAME SIZE IN MICROMETRES AND CENTRED IN THE BIGGER BOX. `positions` maps
    # `offset + scale * unit`, so keeping the physical size means scaling by 1/grow, and centring
    # means solving offset = 0.5 - scale * (the animal's own centroid in unit coordinates).
    import numpy as _np
    scale = 0.85 / grow
    _u = _np.asarray(positions(f, 1.0, [0.0, 0.0, 0.0])).mean(0)
    off = [round(float(0.5 - scale * c), 6) for c in _u]
    base["sets"]["cell"]["start"] = positions(f, scale, off)

    base["sets"]["water_particle"]["per_parent"] = n_water
    for o in base["seed"]:
        if o.get("op") == "exclude_overlap":
            o["res"] = n_grid
    # THE COURANT LIMIT FOLLOWS dx, AND dx IS IN WORLD UNITS. A finer grid on the same unit cube
    # halves dx and therefore halves the stable substep; the physical cell size being unchanged is
    # irrelevant to the solver, which only ever sees 1 / n_grid.
    # `save_data` WINS OVER `record_cap` (engine.py:1736), so it has to go for the cap to bite.
    base["general"].pop("save_data", None)
    base["general"]["record_cap"] = record
    base["plotting"]["max_frames"] = record - 1
    return base


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
    ap.add_argument("rung", nargs="?", default="all", choices=["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r7w", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "all"])
    ap.add_argument("--list", action="store_true", help="say what would be written, write nothing")
    a = ap.parse_args()
    F = facts()
    print(f"{REGION}: {F['n']:,} cells, {len(F['order'])} classes, {F['n_edges']:,} edges, "
          f"cube {F['side_um']:g} um\n")
    if a.rung in ("r1", "all"):
        write(list(range(len(R1))), F, dry=a.list)
    if a.rung in ("r18", "all"):
        import yaml
        sp = spec_r18(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} the same animal in a box twice as wide, to "
              f"bound what the walls were worth")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R18 -- open water. Measured on R15, the shell within 20 um of a wall\n"
                         "# moves at 1.39 um/s against 7.46 in the shell the cilia stir: 19%.\n"
                         "# Periodic is not the alternative -- engine.py:986 excludes a periodic\n"
                         "# world from the warp path, and these specs name warp explicitly, so it\n"
                         "# would clamp at the wall instead of wrapping. A wrong answer at full\n"
                         "# speed. So length_um doubles, the animal's scale halves to keep its\n"
                         "# size in micrometres, the grid doubles to keep the cell at 4.08 um,\n"
                         "# and the substep halves with dx. Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r17", "all"):
        import yaml
        sp = spec_r17(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} the blades are ATTACHED: a momentum-exact hinge "
              f"to the cell's own matter")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R17 -- cilium_anchor. R13 made the shaft ordinary matter so the fluid\n"
                         "# could push back on it, and thereby deleted cilium_kinematics, which\n"
                         "# was the only thing holding it to its cell. On R16 that cost 18 of 74\n"
                         "# blades, the worst ending 232 um away in a 196 um box, most gone\n"
                         "# inside 30 frames. The anchor is a momentum-exact hinge to the live\n"
                         "# centroid of the cell's own material points -- NOT mpm_anchor, which\n"
                         "# springs to a frame-0 position and would drag the swimming larva back.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r16", "all"):
        import yaml
        sp = spec_r16(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} the girdle polarised about the ANIMAL's axis, "
              f"a blade the size of the band, a colour range you can see")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R16 -- anatomy, after R15 fixed the physics. radial_polarity and\n"
                         "# metachronal_phase were centred on [0.5, 0.5], the middle of the BOX,\n"
                         "# while the animal's centroid is at (0.353, 0.475): the mean of the 74\n"
                         "# blade directions had magnitude 0.399 where a girdle radiating evenly\n"
                         "# gives 0.010. The blade comes down to 25 x 10 um -- the band's own\n"
                         "# median cell spacing is 9.9 um, so 10 um tiles it one per cell -- and\n"
                         "# color_range drops 0.35 -> 0.06 world/s against a measured median water\n"
                         "# speed of 0.0229, which is why the flow rendered black.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r15", "all"):
        import yaml
        sp = spec_r15(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} a RESOLVED scene: CFL-legal substep, ~4 "
              f"particles per cell, and a blade with a face")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R15 -- the scene is actually a continuum. R14 ran 6.76x over the\n"
                         "# Courant limit (substep 5.0e-03 against 7.4e-04) with every body at\n"
                         "# 0.18 to 0.47 particles per grid cell against the ~8 MLS-MPM wants.\n"
                         "# The body was not breaking for want of stiffness; it was breaking\n"
                         "# because there was no continuum there to be stiff. Grid 96 -> 48,\n"
                         "# substep -> 1.79e-03 (28 a frame), water 140k -> 400k, body 8 -> 32\n"
                         "# points a cell, and the shaft becomes a 24 x 9 BLADE with a face.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r14", "all"):
        import yaml
        sp = spec_r14(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} a 6 um PADDLE the grid can move (5.9 cells), "
              f"water coloured by speed")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R14 -- an actuator the grid can move. R13's 1.2 um shaft was 1.18\n"
                         "# cells across and had exactly zero momentum: MPM carries one velocity\n"
                         "# per node, so a sub-cell body is advected by the fluid it shares nodes\n"
                         "# with. 6 um is 5.9 cells. It is a PADDLE, not a cilium, and is named\n"
                         "# as one. Water is coloured by speed so still fluid is black.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r13", "all"):
        import yaml
        sp = spec_r13(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} the shaft as ELASTIC matter, driven by a pure "
              f"couple -- momentum conserved by construction")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R13 -- the cilium feels the fluid. An elastic shaft driven by a\n"
                         "# TORQUE at its root, with the equal and opposite couple on the body.\n"
                         "# Forces sum to zero by construction, so a driven shaft cannot\n"
                         "# accelerate its own centre of mass -- the invariant R12 broke.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r12", "all"):
        import yaml
        sp = spec_r12(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} TRUE CILIA: {F['count']['ciliary band']} shafts "
              f"of 20 um, driven open-loop, in water, no gravity")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R12 -- true cilia. A slender shaft rooted in each band cell, with an\n"
                         "# ANGLE driven through the eye's own two-stage plant. The drive is\n"
                         "# PRESCRIBED here, not wired: if the water does not move with the\n"
                         "# effector flat out, wiring a connectome to it would only be adding an\n"
                         "# explanation for a thing that is not happening.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r11", "all"):
        import yaml
        sp = spec_r11(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} the FREE animal in water -- swimming is "
              f"possible for the first time")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R11 -- the free animal in water. Every earlier water run pinned every\n"
                         "# material point to where it started, so only local stirring could be\n"
                         "# measured. R10 removed the anchor; this one can move.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r10", "all"):
        import yaml
        sp = spec_r10(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} cells at 4.0 um fill the body; "
              f"{F['n'] * sp['sets']['mpm_particle']['per_parent']:,} points, NO anchor")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R10 -- a tissue rather than a heap. At 1.9 um the cells filled 6.0%\n"
                         "# of the animal's bounding box and needed an anchor to stand up; at\n"
                         "# 4.0 um they fill 63% and touch, so the elasticity that was always\n"
                         "# there holds the body together with nothing pinning it.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r9", "all"):
        import yaml
        sp = spec_r9(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} a TANGENTIAL stroke under the same wave")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R9 -- the stroke sweeps along the surface instead of pointing out of\n"
                         "# it. R8 showed that a metachronal wave over a RADIAL stroke changes\n"
                         "# nothing: winding the phase changes when each cell pushes, never which\n"
                         "# way. One change from R8, and three controls already run.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r8", "all"):
        import yaml
        sp = spec_r8(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} a metachronal wave on the band, "
              f"against R6 and R7 as controls")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R8 -- a metachronal wave. R6 and R7 showed that a bigger stroke moves\n"
                         "# LESS water, because the stroke is reciprocal and a reciprocal stroke\n"
                         "# transports nothing at low Reynolds number. The only change here is\n"
                         "# where each band cell starts in the cycle.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r7w", "all"):
        import yaml
        sp = spec_r7w(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} the rhythm-driven stroke, in the R6 pool")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R7w -- R7's rhythm-driven stroke in R6's pool. The same animal, the\n"
                         "# same water, the same shared grid; only the drive differs, so the\n"
                         "# reach can be compared against R6's 31 um directly.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r7", "all"):
        import yaml
        sp = spec_r7(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} {F['count']['Motoneuron']} pacemakers -> "
              f"{F['count']['ciliary band']} band cells, over a signed connectome")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R7 -- the rhythm comes from the neurons. This connectome cannot make\n"
                         "# one itself: every weight is a non-negative synapse count, so by\n"
                         "# Perron-Frobenius its leading mode is REAL and rising gain makes it\n"
                         "# latch, never beat. The animal keeps its rhythm in cells that have\n"
                         "# one (Veraszto et al. 2017, eLife 6:e26000).\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r6", "all"):
        import yaml
        sp = spec_r6(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} "
              f"{F['n'] * sp['sets']['mpm_particle']['per_parent']:,} animal points + "
              f"{sp['sets']['water_particle']['per_parent']:,} water points, one grid")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R6 -- water. Two MPM particle sets sharing one grid, which is how\n"
                         "# the beat reaches the fluid: mpm_scatter zeroes the grid only on the\n"
                         "# substep's first scatter and accumulates thereafter.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r5", "all"):
        import yaml
        sp = spec_r5(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} {F['count']['ciliary band']} band cells beating, "
              f"driven through {F['n_edges']:,} synapses")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R5 -- the motor pathway: the connectome's own output makes the\n"
                         "# ciliary band beat. The 255 motoneuron-to-ciliary-band edges are\n"
                         "# already in the data; the missing step was an operator that turns a\n"
                         "# membrane state into a stress, which is polar_active_stress[driven].\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
    if a.rung in ("r4", "all"):
        import yaml
        sp = spec_r4(F)
        p = os.path.join(OUT, sp["general"]["name"] + ".yaml")
        print(f"  {os.path.relpath(p, REPO):48s} {sp['general']['n_frames']} frames of "
              f"{F['n']:,} cells over {F['n_edges']:,} synapses")
        if not a.list:
            os.makedirs(OUT, exist_ok=True)
            with open(p, "w") as fh:
                fh.write("# R4 -- the connectome doing something: the membrane law, the synaptic\n"
                         "# law, and every soma coloured by its own membrane state.\n"
                         "# Written by tools/platynereis_specs.py.\n")
                yaml.safe_dump(sp, fh, sort_keys=False, default_flow_style=False, width=110)
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
