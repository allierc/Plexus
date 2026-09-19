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
            "dot_shading": True, "max_frames": 300, "stills": 6, "keep_stills": True,
            "camera_roll": 180.0, "dot_size": 7.0,
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
            "dot_shading": True, "max_frames": 300, "stills": 6, "keep_stills": True,
            "camera_roll": 180.0, "dot_size": 2.5,
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
            "dot_shading": True, "max_frames": 300, "stills": 6, "keep_stills": True,
            "camera_roll": 180.0, "dot_size": 2.5, "subject": "mpm_particle",
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
    ap.add_argument("rung", nargs="?", default="all", choices=["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r7w", "r8", "r9", "r10", "r11", "all"])
    ap.add_argument("--list", action="store_true", help="say what would be written, write nothing")
    a = ap.parse_args()
    F = facts()
    print(f"{REGION}: {F['n']:,} cells, {len(F['order'])} classes, {F['n_edges']:,} edges, "
          f"cube {F['side_um']:g} um\n")
    if a.rung in ("r1", "all"):
        write(list(range(len(R1))), F, dry=a.list)
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
