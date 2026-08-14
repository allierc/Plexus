"""eye_spec -- build the Plexus2 `spec.yaml` for the zebrafish oculomotor plant.

The spec is the deliverable: sets / fields / operators / schedule, validated by the stock
`plexus.schema.load` and run by the stock `plexus.engine.run`. Nothing about the eye lives
in the engine; the operators are registered by importing `eye_ops` + `muscle_ops`.

    sets       eye (organ) / mpm_particle (globe tissue) /
               muscle (x6) / muscle_particle (muscle tissue) / orbit (socket)
    fields     mpm_grid -- ONE background grid, shared by both particle sets, which is
               what couples the muscles to the globe
    schedule   pose -> geometry -> drive -> contract -> anchor -> contact -> drag
               -> [ MLS-MPM substep loop, run over BOTH bodies ]

Note the operator order inside the substep: `mpm_scatter` appears twice, first for the globe
(the stock implementation, which RESETS the grid for the substep) and then for the muscles
(`implementation: accumulate`, which adds to it). One schedule token runs both.
"""
from __future__ import annotations

import copy

import numpy as np
import yaml

import eye_anatomy as EA
import fish_anatomy as FA


# --------------------------------------------------------------------------- #
#  gaze programs: [frame, horizontal, vertical, torsion] in degrees
# --------------------------------------------------------------------------- #
def _sequence(holds, hold=58):
    """[frame, h, v, t] waypoints from a list of (h, v, t) commands, one every `hold` frames."""
    return [[i * hold] + list(map(float, c)) for i, c in enumerate(holds)]


def _oscillate(a, b, cycles):
    """Back and forth between two commands, `cycles` times."""
    return [c for _ in range(cycles) for c in (a, b)]


P = (0.0, 0.0, 0.0)
PROGRAMS = {
    # the full tour, with each antagonist pair driven BACK AND FORTH several times so the
    # movie shows the plant working rather than a single excursion
    "atlas": _sequence(
        [P]
        + _oscillate((26, 0, 0), (-26, 0, 0), 3)        # abduction  <-> adduction   LR / MR
        + [P]
        + _oscillate((0, 16, 0), (0, -16, 0), 3)        # elevation  <-> depression  SR+IO / IR+SO
        + [P]
        + _oscillate((0, 0, 9), (0, 0, -9), 3)          # intorsion  <-> extorsion   SO / IO
        + [P]                                           # (+-9 deg: ocular torsion really is
        + _oscillate((22, 14, 0), (-22, -14, 0), 2)     #  a small excursion, unlike gaze)
        + _oscillate((22, -14, 0), (-22, 14, 0), 2)
        + [P]),
    # the zebrafish optokinetic response: slow tracking ramps with fast reset saccades
    "okr": [[0, 0, 0, 0]] + [
        row for i in range(10)
        for row in ([130 * i + 10, -18, 0, 0], [130 * i + 32, -9, 0, 0],
                    [130 * i + 54, 0, 0, 0], [130 * i + 76, 9, 0, 0],
                    [130 * i + 98, 18, 0, 0], [130 * i + 112, -18, 0, 0])
    ],
    # a short calibration run: abduction, adduction, elevation, intorsion, back
    "probe": _sequence([P, (25, 0, 0), (-25, 0, 0), (0, 16, 0), (0, 0, 9), P], hold=65),
}

PRESETS = {
    "atlas": dict(program="atlas", n_frames=int(PROGRAMS["atlas"][-1][0]) + 90),
    "okr": dict(program="okr", n_frames=int(PROGRAMS["okr"][-1][0]) + 60),
    "probe": dict(program="probe", n_frames=int(PROGRAMS["probe"][-1][0]) + 70),
}


def _strengths(oblique_strength=None):
    """Per-muscle strength factor for `muscle_contract`. The obliques reach the globe over a
    much shorter post-pulley path than the recti, so they were given a boost -- but a short,
    sharply curved strap is also the one that BUCKLES first, so the boost is a knob, not a
    constant."""
    w = [float(x) for x in EA.peak_tensions()]
    if oblique_strength is not None:
        w[4] = w[5] = float(oblique_strength)
    return w


def build_spec(name="eye_zebrafish", preset="atlas", n_particles=45000, n_muscle_particles=2600,
               n_grid=128, dt=0.003, substep_dt=None, drag=5.0, muscle_drag=6.0,
               contract=26.0, stretch_activation=0.0,
               kp=0.10, ki=0.0, kd=0.010, tonic=0.20, gain=1.2, tau=0.020,
               k_socket=5000.0, k_fat=4000.0, c_fat=90.0, k_bone=9000.0, c_bone=60.0,
               k_sleeve=2600.0, c_sleeve=30.0, sleeve_free=(0.70, 0.88),
               n_frames=None, sclera_youngs=420.0, vitreous_youngs=45.0, choroid_youngs=130.0,
               muscle_youngs=60.0, mus_width=0.034, mus_thickness=0.021, mus_arc=30.0,
               mus_gap=0.038, mus_embed=-0.014, mus_frac=0.88, oblique_strength=None,
               plant="mammal", axial_ratio=None, program=None, seed=0):
    """The full spec as a plain dict, ready for `yaml.safe_dump` + `plexus.schema.load`.

    `plant` selects WHOSE ANATOMY the operators are handed:

        "mammal"      the original guess in `eye_anatomy`: four recti from an annulus
                      of Zinn, obliques behind the equator, a trochlea, globe 0.82
        "fish_larva"  measured off Fig. 12.1A of Tulenko & Currie -- obliques from the
                      rostral orbit onto the dorsal and ventral faces, SR/IR/MR from
                      one caudal plate, LR from outside the orbit onto the caudal
                      sclera, globe 0.676, per-muscle widths
        "fish_adult"  the same globe with Kasprick's adult insertions: all six on the
                      sclera-corneal junction, SO sharing SR's station and IO sharing
                      IR's

    Nothing about the plant reaches the operators except through these params, so a
    plant is a set of numbers in the spec, not a branch in the code.
    """
    p = dict(PRESETS.get(preset, PRESETS["atlas"]))
    n_frames = int(n_frames if n_frames is not None else p["n_frames"])
    prog = program if program is not None else PROGRAMS[p["program"]]
    cx, cy, cz = EA.GLOBE_CENTER

    fish = plant.startswith("fish")
    if fish:
        stage = plant.split("_", 1)[1]
        ins_dirs = FA.insertion_dirs(stage)
        org_world = FA.origins_world(stage)
        mus_width = [float(v) for v in FA.strap_widths()]
        mus_thickness = float(FA.strap_thickness())
        # the fish's origins are real and distinct bones, so the whole muscle is
        # drawn: `frac` existed only to stop four mammalian recti piling onto one apex
        mus_frac = 1.0
        # let each muscle's wrap be set by where its bone is (tangent construction),
        # and ride closer to the sclera: these bellies are a third the thickness the
        # mammalian straps were, so a 0.365 a_eq stand-off held them out in mid-orbit
        mus_arc = None
        mus_gap = 0.0161
        mus_embed = -0.013
        strengths = [float(v) for v in FA.peak_tensions()]
        lens = FA.lens()
        lens_center = [0.0, 0.0, float(lens["center_axial"] * FA.axial_ratio())]
        lens_radius = float(lens["radius"])
        ratio = float(FA.axial_ratio() if axial_ratio is None else axial_ratio)
    else:
        ins_dirs = EA.insertion_dirs()
        org_world = EA.origins_world()
        strengths = _strengths(oblique_strength)
        lens_center = list(EA.LENS_CENTER)
        lens_radius = EA.LENS_RADIUS
        ratio = float(EA.AXIAL_RATIO if axial_ratio is None else axial_ratio)
    belly = [[round(float(v), 4) for v in b]
             for b in 0.5 * (np.asarray(org_world)
                             + np.asarray(EA.GLOBE_CENTER)[None, :] + EA.A_EQ * ins_dirs)]

    spec = {
        "general": {
            "name": name, "seed": int(seed), "n_frames": n_frames, "dt": float(dt),
            "boundary": "wall", "dim": 3, "world": [1.0, 1.0, 1.0],
            "record_cap": 4000, "field_record_cap": 1,
        },

        # ---- SETS: the biological decomposition ------------------------------- #
        "sets": {
            # the GLOBE as one organ: a centroid + the (h, v, t) orientation readout
            "eye": {
                "n": 1,
                "start": [[cx, cy, cz]],
                "state": {
                    "pos": {"width": 3, "integration": "second_order_coordinate", "boundary": "world"},
                    "vel": {"width": 3, "integration": "second_order_rate", "boundary": "free",
                            "record": False},
                    "gaze": {"width": 3, "integration": "none", "boundary": "free"},
                },
                # Radial bands: a soft vitreous gel inside a stiff scleral shell -- what makes
                # the eye DEFORMABLE but not floppy. The ~9x contrast from core to shell is what
                # the strain panel shows; the ABSOLUTE values matter just as much. An earlier
                # version had the vitreous at E = 9, a shear modulus of 3.8, which is very nearly
                # a fluid: under a strong tendon pull the interior simply flowed and the globe
                # came apart (archive/t18_q_b). The fixed-corotated law has no failure criterion,
                # so "holds together" is entirely a matter of having enough shear modulus
                # EVERYWHERE, not only in the shell.
                "types": {
                    "globe": {
                        "fraction": 1.0, "youngs": float(sclera_youngs),
                        "layers": [
                            {"frac": EA.R_VITREOUS, "youngs": float(vitreous_youngs)},
                            {"frac": EA.R_INNER, "youngs": float(choroid_youngs)},
                            {"frac": 1.0, "youngs": float(sclera_youngs)},
                        ],
                    },
                },
            },
            "mpm_particle": {"parent": "eye", "per_parent": int(n_particles),
                             "radius": EA.A_EQ, "density": 1.0},

            # the six extraocular muscles: activation is the integrated block; length and
            # tension are readouts aggregated from their own material points
            "muscle": {
                "n": EA.N_MUSCLE,
                "start": belly,
                "state": {
                    "pos": {"width": 3, "integration": "second_order_coordinate", "boundary": "world"},
                    "vel": {"width": 3, "integration": "second_order_rate", "boundary": "free",
                            "record": False},
                    "act": {"width": 1, "integration": "first_order", "boundary": "free"},
                    "tension": {"width": 1, "integration": "none", "boundary": "free"},
                    "length": {"width": 1, "integration": "none", "boundary": "free"},
                },
                "types": {"muscle": {"fraction": 1.0, "youngs": float(muscle_youngs)}},
            },
            "muscle_particle": {"parent": "muscle", "per_parent": int(n_muscle_particles),
                                "radius": 0.02, "density": 1.0},

            # the bony socket
            "orbit": {"n": 1, "start": [[cx, cy, cz]]},
        },

        # ---- FIELDS: one grid, shared -- this is the mechanical coupling -------- #
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": int(n_grid)}},

        # ---- OPERATORS --------------------------------------------------------- #
        "operators": [
            # anatomy: once, at frame 0
            {"op": "eye_anatomy", "at": "mpm_particle", "before_frame": 1,
             "center": [cx, cy, cz], "a_eq": EA.A_EQ, "axial_ratio": ratio,
             "lens_youngs": EA.LENS_YOUNGS, "cornea_youngs": 320.0,
             "lens_center": [float(v) for v in lens_center],
             "lens_radius": float(lens_radius)},
            {"op": "muscle_morphogenesis", "at": "muscle_particle", "before_frame": 1,
             "center": [cx, cy, cz], "a_eq": EA.A_EQ, "c_ax": EA.A_EQ * ratio,
             "width": mus_width if isinstance(mus_width, list) else float(mus_width),
             "thickness": (mus_thickness if isinstance(mus_thickness, list)
                           else float(mus_thickness)),
             "arc_deg": None if mus_arc is None else float(mus_arc), "gap": float(mus_gap), "embed": float(mus_embed),
             "frac": float(mus_frac), "youngs": float(muscle_youngs),
             "insertions": [[round(float(v), 5) for v in d] for d in ins_dirs],
             "origins": [[round(float(v), 5) for v in o] for o in org_world]},
            # readouts
            {"op": "eye_pose", "at": "eye", "child": "mpm_particle", "shell_min": 0.86},
            {"op": "muscle_geometry", "at": "muscle", "child": "muscle_particle", "eye": "eye"},
            # command -> contraction
            {"op": "oculomotor_drive", "at": "muscle", "eye": "eye", "program": prog,
             "kp": float(kp), "ki": float(ki), "kd": float(kd), "tonic": float(tonic),
             "gain": float(gain),
             "tau": float(tau)},
            {"op": "muscle_contract", "at": "muscle_particle", "muscles": "muscle",
             "amplitude": float(contract), "stretch_activation": float(stretch_activation),
             "strength": strengths},
            # boundary conditions
            {"op": "bone_anchor", "at": "muscle_particle", "k": float(k_bone), "c": float(c_bone)},
            {"op": "muscle_sleeve", "at": "muscle_particle", "k": float(k_sleeve),
             "c": float(c_sleeve), "free_from": float(sleeve_free[0]),
             "free_to": float(sleeve_free[1])},
            {"op": "orbit_socket", "at": "mpm_particle", "orbit": "orbit",
             "k": float(k_socket), "damp": 12.0, "radius": EA.A_EQ + 0.007,
             "aperture": EA.CUP_APERTURE_DEG, "k_fat": float(k_fat), "c_fat": float(c_fat)},
            {"op": "drag", "at": "mpm_particle", "k": float(drag), "emit": "mpm_acceleration"},
            {"op": "drag", "at": "muscle_particle", "k": float(muscle_drag),
             "emit": "mpm_acceleration"},
            # the decomposed MLS-MPM cycle, over BOTH bodies, into ONE grid
            {"op": "mpm_strain", "at": "mpm_particle"},
            {"op": "mpm_strain", "at": "muscle_particle"},
            # `polar: higham` -- the fixed-corotated stress needs the rotation from F = R S,
            # and a batched 3x3 SVD costs a microsecond per particle per substep: measured
            # on this plant it was 44.7 ms of mpm_scatter's 46.4 ms, i.e. most of the run.
            # The Newton polar iteration gives the same rotation to float32 (the gaze
            # trajectory agrees to 1e-6 deg) at half the wall clock for the whole simulation.
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid",       # resets the grid
             "drag": 0.0, "a_max": 200, "polar": "higham"},
            {"op": "mpm_scatter", "at": "muscle_particle", "to": "mpm_grid",    # adds to it
             "implementation": "accumulate", "drag": 0.0, "a_max": 200, "polar": "higham"},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 1.0},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "wall_contact": 0.02, "vmax": 1000000000.0},
            {"op": "mpm_gather", "at": "muscle_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "wall_contact": 0.02, "vmax": 1000000000.0},
        ],

        # ---- SCHEDULE ---------------------------------------------------------- #
        "schedule": [
            "eye_anatomy",
            "muscle_morphogenesis",
            "eye_pose",
            "muscle_geometry",
            "oculomotor_drive",
            "muscle_contract",
            "bone_anchor",
            "muscle_sleeve",
            "orbit_socket",
            "drag",
            {"substep_dt": float(substep_dt or 1.2e-4),
             "steps": ["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"]},
        ],

        "plotting": {"background": "black", "fps": 30},
    }
    return spec


def write_spec(spec: dict, path: str) -> str:
    with open(path, "w") as f:
        f.write("# Plexus2 spec -- zebrafish eyeball in its orbit, moved by six extraocular\n"
                "# muscles that are themselves contracting MPM bodies.\n"
                "# Sets: eye / mpm_particle / muscle x6 / muscle_particle / orbit.\n"
                "# Operators registered by prototype/eye/{eye_ops,muscle_ops}.py.\n")
        yaml.safe_dump(copy.deepcopy(spec), f, sort_keys=False, width=100)
    return path


def cfl_limit(spec: dict) -> float:
    """The MLS-MPM Courant limit: dt_sub <= 0.4*dx/c, c = sqrt((la+2mu)/rho) over the
    STIFFEST material in EITHER body. The stock helper only understands a set literally
    named `cell`, so the check lives here (prototype-local, nothing promoted)."""
    import math
    nu, rho = 0.2, 1.0
    n_grid = int(next(iter(spec["fields"].values()))["n_grid"])
    ys = [EA.LENS_YOUNGS]
    for sname in ("eye", "muscle"):
        for t in spec["sets"][sname].get("types", {}).values():
            ys.append(float(t.get("youngs", 100.0)))
            for L in t.get("layers", []):
                ys.append(float(L.get("youngs", 100.0)))
    E = max(ys)
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    return 0.4 * (1.0 / n_grid) / math.sqrt((la + 2 * mu) / rho)
