#!/usr/bin/env python
"""Write the STATIC scaffold scene of the C. jejuni motor from PDB 9HMF, as measured.

    python tools/bfm_scaffold_spec.py [name]     # -> config/bacterium/<name>.yaml (default bfm_c01_scaffold_9hmf)

The scene holds the periplasmic scaffold that Drobnic et al. 2025 (Nat Microbiol 10:1723) built
into the whole-motor map: six proteins, each its OWN SET of points placed by `cloud_seed` from
the entry's alpha-carbon trace after its 17-fold symmetry expansion (shapes/9hmf/points.npz,
written by tools/bfm_anatomy_from_structure.py --cloud 9hmf). Nothing here moves and nothing is
scheduled: this is the anatomy the human validates before a rotor is placed inside it.

The frame is the one the rotor scene will use, so the two compose: the world unit is 140 nm, the
inner-membrane plane is at world z 0.30, and the scaffold's own z = 0 (the mean of its atoms)
sits Z_OFFSET_NM above that plane -- FliL and the MotB periplasmic domain reach down to z -8.8 nm
in the scaffold's frame and they sit ON the membrane, so the membrane is 8.5 nm below the mean.
The 17 stator units are marked as spheres under the MotB dimers' centroids (360/17 degrees apart,
the angles of record step 0010) at the membrane, r 26.5 nm: see STATOR_R_NM below.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

BOX_NM = 140.0
Z_MEMBRANE = 0.30
Z_OFFSET_NM = 8.5                    # the scaffold's mean-atom plane above the inner membrane
# THE STATOR SPHERES ARE MotA, WHICH 9HMF DOES NOT CONTAIN. MotB's periplasmic domain is in the
# model (cyan points, centroids at r 29.19 nm, z -2.0 nm); the MotA pentamer it belongs to sits in
# the inner membrane beneath it and is what pushes the C-ring, at the lever arm the paper's torque
# implies: 3,288 pN nm = 17 x 7.3 pN x 26.5 nm. So each sphere (4.5 nm, the ~9 nm stator complex)
# is drawn at r 26.5 nm and z -10 nm -- 1.5 nm into the membrane, whose periplasmic face is the
# scaffold's z -8.5 -- under its own MotB, at the MotB angles.
STATOR_R_NM, STATOR_Z_NM, STATOR_RADIUS_NM = 26.5, -10.0, 4.5
N_STATOR = 17
STATOR_PHASE_DEG = 9.6               # the first MotB centroid's angle in the entry's frame (step 0010)
COLORS = {"FliL": [0.75, 0.22, 0.17], "PflA": [0.18, 0.44, 0.71], "PflB": [0.29, 0.49, 0.35],
          "PflCD": [0.88, 0.51, 0.08], "Lipo": [0.48, 0.35, 0.65], "MotB": [0.09, 0.75, 0.81]}


# TWO LOOKS FOR THE SAME ANATOMY, `--cryo` picks the second:
#   dots  every alpha carbon a lit matte sprite on black, eye-dome lighting for the outline and
#         ambient occlusion for the shadow -- the data as deposited, nothing fitted.
#   cryo  the structural-biology figure: a matte surface contoured around each protein's atoms
#         (0.5 nm grid, blurred 1.5 cells = 0.75 nm, the look of an 8 A map; 1.2 left holes in the loops), a black silhouette where one
#         body's depth jumps against another's (ChimeraX `graphics silhouettes`), ambient
#         occlusion, no specular anywhere -- the rendering of every cryo-EM panel in Drobnic
#         et al. 2025, on black by the human's choice (2026-09-23; `background: white` gives the
#         paper's page). The surface is a DRAWING of the atoms, not a model; the sets are
#         still the atoms.
LOOKS = {
    "dots": {"background": "black", "dot_shading": True, "dot_ambient": 0.35, "dot_diffuse": 0.75,
             "dot_specular": 0.0, "edl": True, "ssao": True},
    "cryo": {"background": "black", "ssao": True, "ssao_radius_frac": 0.03, "legend": False,
             "silhouette": {"color": "black", "width": 1.5},
             "surface_specular": 0.0, "surface_ambient": 0.32, "surface_diffuse": 0.72,
             "light": "studio"},
}
SURFACE = {
    "dots": {"render": "dots", "point_size": 3.0},
    "cryo": {"render": "surface", "spacing": round(0.5 / 140.0, 6), "blur": 1.5, "smooth": 20, "iso_frac": 0.3,
             "specular": 0.0, "ambient": 0.32, "diffuse": 0.72},
}


def nm(x):
    return x / BOX_NM


def scaffold(look="dots", z_membrane=Z_MEMBRANE, box_nm=BOX_NM):
    """The six scaffold sets, their seeds, and the plotting entries that draw them, for a box of
    `box_nm` whose inner-membrane plane is at world z `z_membrane`. The motor generator
    (tools/bfm_motor_spec.py --scaffold) composes its rotor into the same frame."""
    from plexus.paths import graphs_data_path
    pts = np.load(os.path.join(graphs_data_path(), "shapes", "9hmf", "points.npz"))
    origin = [0.5, 0.5, round(z_membrane + Z_OFFSET_NM / box_nm, 6)]
    scale = 1.0 / (box_nm * 1e-9)                       # world units per metre
    sets, seeds = {}, []
    for part in COLORS:
        sets[part] = {"n": int(len(pts[part])), "start": [origin]}
        seeds.append({"op": "cloud_seed", "at": part, "cloud": f"9hmf/{part}", "origin": origin, "scale": scale})
    plot = {"compartment_sets": list(COLORS), "surface": {p: dict(SURFACE[look]) for p in COLORS},
            "opacity": {p: 1.0 for p in COLORS}, "colors": dict(COLORS)}
    return sets, seeds, plot, origin


def stator_ring(origin_z, r_nm=STATOR_R_NM, z_nm=STATOR_Z_NM, box_nm=BOX_NM):
    """The 17 stator points at the MotB angles of PDB 9HMF, r and z in nm about the scaffold's
    origin (its mean-atom plane)."""
    th = [math.radians(STATOR_PHASE_DEG) + 2.0 * math.pi * i / N_STATOR for i in range(N_STATOR)]
    return [[round(0.5 + r_nm / box_nm * math.cos(a), 6), round(0.5 + r_nm / box_nm * math.sin(a), 6),
             round(origin_z + z_nm / box_nm, 6)] for a in th]


def main():
    look = "cryo" if "--cryo" in sys.argv else "dots"
    sets, ops, plot, origin = scaffold(look)
    sets["stator_unit"] = {"n": N_STATOR, "start": stator_ring(origin[2])}
    args_ = [x for x in sys.argv[1:] if not x.startswith("--")]
    name_spec = args_[0] if args_ else ("bfm_c01_scaffold_9hmf" + ("_cryo" if look == "cryo" else ""))
    spec = {
        "general": {"name": name_spec, "seed": 0, "n_frames": 10, "dt": 0.002, "boundary": "wall",
                    "dim": 3, "world": [1.0, 1.0, 1.0], "save_data": True, "record_cap": 301,
                    "units": {"length_um": BOX_NM * 1e-3, "time_s": 0.005, "force_nN": 0.5047076668008149}},
        "sets": sets,
        "fields": {},
        "seed": ops,
        "operators": [],
        "schedule": [],
        "plotting": {
            "renderer": "vtk_points", "background": "black", "up_axis": 2, "box_frame": True,
            "hide_sets": ["stator_unit"],
            "render_3d": "compartments", **plot,
            "spheres": {"set": "stator_unit", "radius": round(nm(STATOR_RADIUS_NM), 6), "color": "#e8b04a",
                        "specular": 0.0, "ambient": 0.25, "diffuse": 0.8},
            "subject": "PflA", "keep_stills": True, "stills": 4, "max_frames": 10,
            "real_time": False, "duration_s": 10.0, "zoom": 1.6,
            **LOOKS[look],
        },
    }
    out = os.path.join(REPO, "config", "bacterium", name_spec + ".yaml")
    with open(out, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=None, width=100)
    print(f"wrote {os.path.relpath(out, REPO)}: " + ", ".join(f"{p} {sets[p]['n']:,}" for p in COLORS)
          + f"; origin {origin}, {N_STATOR} stator spheres at r {STATOR_R_NM} nm")


if __name__ == "__main__":
    main()
