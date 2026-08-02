"""eye_spec -- build the Plexus2 `spec.yaml` for the zebrafish oculomotor plant.

The spec is the deliverable: sets / fields / operators / schedule, validated by the stock
`plexus.schema.load` and run by the stock `plexus.engine.run`. Nothing about the eye lives
in the engine; the six new operators are registered by importing `eye_ops`.

    general      dim 3, a unit world, the frame dt and the frame count
    sets         eye (organ) / mpm_particle (material points) / muscle (x6) / orbit (socket)
    fields       mpm_grid -- the transient MLS-MPM background grid
    operators    2 anatomy (rewire, frame 0) + 4 plant + the 4 decomposed MLS-MPM steps
    schedule     pose -> drive -> traction -> contact -> drag -> [MPM substep loop]
"""
from __future__ import annotations

import copy

import yaml

import eye_anatomy as EA


# --------------------------------------------------------------------------- #
#  gaze programs: [frame, horizontal, vertical, torsion] in degrees
# --------------------------------------------------------------------------- #
PROGRAMS = {
    # a systematic tour of the plant: each command recruits a different muscle group
    "atlas": [
        [0,     0,   0,   0],     # primary position (tonic co-contraction only)
        [70,   26,   0,   0],     # abduction            -> lateral rectus
        [150, -26,   0,   0],     # adduction            -> medial rectus
        [230,   0,  19,   0],     # elevation            -> superior rectus + inferior oblique
        [310,   0, -19,   0],     # depression           -> inferior rectus + superior oblique
        [390,   0,   0,  15],     # intorsion            -> superior oblique (+ superior rectus)
        [470,   0,   0, -15],     # extorsion            -> inferior oblique (+ inferior rectus)
        [550,  22,  15,   0],     # oblique up-and-out   -> a combination
        [630, -22, -15,   0],     # oblique down-and-in
        [700,   0,   0,   0],     # back to primary
    ],
    # the zebrafish optokinetic response: slow tracking ramp, fast reset saccade
    "okr": [[0, 0, 0, 0]] + [
        row for i in range(6)
        for row in ([120 * i + 10, -18, 0, 0], [120 * i + 30, -9, 0, 0],
                    [120 * i + 50, 0, 0, 0], [120 * i + 70, 9, 0, 0],
                    [120 * i + 90, 18, 0, 0], [120 * i + 100, -18, 0, 0])
    ],
    # a short calibration run: one abduction, one elevation, one intorsion
    "probe": [[0, 0, 0, 0], [40, 25, 0, 0], [110, 0, 18, 0], [180, 0, 0, 14], [250, 0, 0, 0]],
}

PRESETS = {
    "atlas": dict(program="atlas", n_frames=760),
    "okr": dict(program="okr", n_frames=740),
    "probe": dict(program="probe", n_frames=300),
}


def build_spec(name="eye_zebrafish", preset="atlas", n_particles=45000, n_grid=96,
               dt=0.003, substep_dt=1.5e-4, amplitude=0.030, drag=5.0,
               kp=0.10, kd=0.010, tonic=0.20, gain=1.2, tau=0.020,
               k_socket=5000.0, k_fat=260.0, c_fat=18.0, n_frames=None,
               sclera_youngs=300.0, vitreous_youngs=9.0, choroid_youngs=40.0,
               program=None, seed=0):
    """The full spec as a plain dict, ready for `yaml.safe_dump` + `plexus.schema.load`."""
    p = dict(PRESETS.get(preset, PRESETS["atlas"]))
    n_frames = int(n_frames if n_frames is not None else p["n_frames"])
    prog = program if program is not None else PROGRAMS[p["program"]]

    cx, cy, cz = EA.GLOBE_CENTER
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
                "types": {
                    # radial bands of the globe: a soft vitreous gel inside a stiff scleral
                    # shell. This is what makes the eye DEFORMABLE but not floppy -- the
                    # tendons dimple the sclera and the strain runs into the interior.
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
            # the material points of the globe (MLS-MPM). Seeded as a uniform ball of
            # radius A_EQ; `eye_anatomy` squashes it into the ovoid at frame 0.
            "mpm_particle": {"parent": "eye", "per_parent": int(n_particles),
                             "radius": EA.A_EQ, "density": 1.0},
            # the six extraocular muscles: activation is the integrated state, tension a readout
            "muscle": {
                "n": EA.N_MUSCLE,
                "state": {
                    "act": {"width": 1, "integration": "first_order", "boundary": "free"},
                    "tension": {"width": 1, "integration": "none", "boundary": "free"},
                },
            },
            # the bony socket
            "orbit": {"n": 1, "start": [[cx, cy, cz]]},
        },

        # ---- FIELDS ------------------------------------------------------------ #
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": int(n_grid)}},

        # ---- OPERATORS --------------------------------------------------------- #
        "operators": [
            # anatomy: run once, at frame 0
            {"op": "eye_anatomy", "at": "mpm_particle", "before_frame": 1,
             "center": [cx, cy, cz], "a_eq": EA.A_EQ, "axial_ratio": EA.AXIAL_RATIO,
             "lens_youngs": EA.LENS_YOUNGS, "cornea_youngs": 260.0},
            {"op": "muscle_insertion", "at": "muscle", "before_frame": 1,
             "particles": "mpm_particle", "patch_deg": EA.INSERTION_PATCH_DEG},
            # the plant, every frame
            {"op": "eye_pose", "at": "eye", "child": "mpm_particle", "shell_min": 0.86},
            {"op": "oculomotor_drive", "at": "muscle", "eye": "eye", "program": prog,
             "kp": float(kp), "kd": float(kd), "tonic": float(tonic), "gain": float(gain),
             "tau": float(tau)},
            {"op": "muscle_traction", "at": "mpm_particle", "muscles": "muscle",
             "amplitude": float(amplitude), "wrap": EA.WRAP},
            {"op": "orbit_socket", "at": "mpm_particle", "orbit": "orbit",
             "k": float(k_socket), "damp": 12.0, "radius": EA.CUP_RADIUS,
             "aperture": EA.CUP_APERTURE_DEG, "k_fat": float(k_fat), "c_fat": float(c_fat)},
            {"op": "drag", "at": "mpm_particle", "k": float(drag), "emit": "mpm_acceleration"},
            # the decomposed MLS-MPM cycle (stock operators)
            {"op": "mpm_strain", "at": "mpm_particle"},
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid", "drag": 0.0, "a_max": 200},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 1.0},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 1.0,
             "wall_contact": 0.02, "vmax": 1000000000.0},
        ],

        # ---- SCHEDULE ---------------------------------------------------------- #
        # pose readout -> gaze error -> innervation -> tension -> traction -> mechanics
        "schedule": [
            "eye_anatomy",
            "muscle_insertion",
            "eye_pose",
            "oculomotor_drive",
            "muscle_traction",
            "orbit_socket",
            "drag",
            {"substep_dt": float(substep_dt),
             "steps": ["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"]},
        ],

        "plotting": {"background": "black", "fps": 30},
    }
    return spec


def write_spec(spec: dict, path: str) -> str:
    with open(path, "w") as f:
        f.write("# Plexus2 spec -- zebrafish eyeball in its orbit, moved by six extraocular\n"
                "# muscles.  Sets: eye (organ) / mpm_particle (material points) / muscle (x6) /\n"
                "# orbit (socket).  The six eye operators are registered by prototype/eye/eye_ops.py.\n")
        yaml.safe_dump(copy.deepcopy(spec), f, sort_keys=False, width=100)
    return path


def cfl_limit(spec: dict) -> float:
    """The MLS-MPM Courant limit for this spec: dt_sub <= 0.4*dx/c, c = sqrt((la+2mu)/rho)
    over the STIFFEST material. The stock helper only understands a set literally named
    `cell`, so the check is done here (prototype-local, nothing promoted)."""
    import math
    nu = 0.2
    rho = float(spec["sets"]["mpm_particle"].get("density", 1.0))
    n_grid = int(next(iter(spec["fields"].values()))["n_grid"])
    ys = [EA.LENS_YOUNGS]
    for t in spec["sets"]["eye"]["types"].values():
        ys.append(float(t.get("youngs", 100.0)))
        for L in t.get("layers", []):
            ys.append(float(L.get("youngs", 100.0)))
    E = max(ys)
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    c = math.sqrt((la + 2 * mu) / rho)
    return 0.4 * (1.0 / n_grid) / c
