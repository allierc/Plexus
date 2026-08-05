"""ecm_spec -- build the spec for "a cell ball grows inside a fibrous matrix".

The spec is the deliverable, as everywhere else in Plexus: sets / fields / operators / schedule,
validated by the stock `plexus.schema.load` and run by the stock `plexus.engine.run`. Nothing
here lives in the engine; importing `ecm_ops` registers the three new operators.

    sets       cell        -- the parent the matrix particles hang off (MPM needs a parent set)
               mpm_particle -- the MATRIX itself: one particle set, seeded into fibres
    fields     mpm_grid    -- the background grid; when the real cell mesh arrives it scatters
                              into THIS grid, which is what makes the coupling two-way
    schedule   aggregate -> ecm_seed -> cell_to_ecm -> ecm_stress
               -> [ MLS-MPM substep loop ]

WHY THE MATERIAL IS SOFT. MPM is explicit, so the substep must satisfy dt < dx/c with
c = sqrt(E/rho): stiffness is what costs wall-clock. The stock 3D material specs run E ~ 190 at
dt_sub 2e-4. A matrix is a gel, not a rubber, and choosing E in the tens is both the biologically
right answer and the one that buys a substep large enough to run a sweep. The relationship is
measured in `sweep.py` rather than assumed -- if the run explodes, the number to move is E or
dt_sub, and which one is not a matter of taste.

EIGHT TYPES, ONE MATERIAL. `plotting.color_by` is a palette INDEX, not a continuous map, so the
stress bands are types. Every type carries identical material properties, which is what makes the
colour decoration that cannot alter the physics it draws: `ecm_stress` writes the band each frame
and the mechanics never reads it.
"""
from __future__ import annotations

import copy

# Eight steps of `inferno`, dark -> bright: unstrained matrix is nearly black and disappears into
# the background, and the stress front is the only bright thing in the frame. A qualitative
# palette would make every band equally loud and the propagation would not read at all.
# BAND 0 MUST BE VISIBLE. The first palette put unstrained matrix at near-black on a black
# background, on the theory that the stress front should be the only bright thing in the frame.
# The result was a frame containing nothing: you cannot watch a front propagate INTO fibres you
# cannot see. The rest state is now a dim slate -- clearly matrix, clearly unstressed -- and the
# ramp climbs to white through the inferno hues.
STRESS_COLORS = [
    [0.16, 0.19, 0.30], [0.35, 0.15, 0.42], [0.58, 0.13, 0.42], [0.78, 0.20, 0.33],
    [0.92, 0.35, 0.18], [0.99, 0.55, 0.10], [1.00, 0.76, 0.22], [1.00, 0.97, 0.75],
]


def build_spec(name, n_frames=320, dt=0.004, substep_dt=2.0e-4, n_grid=64,
               n_particles=48000, youngs=40.0, density=1.0,
               # THE CAVITY MUST BE BIGGER THAN THE TISSUE IS WHEN THE CLOCK STARTS. Runs 21-23 had
               # `cavity_h=0.05` against a tissue whose apical radius began at 0.088 of the box, so
               # the epithelium was already overlapping the matrix at frame 0 and every one of them
               # reported `contact_frame: 0` -- the "moment of first contact" the experiment exists
               # to observe could not happen inside it. Spherical (h == r) by default: with a
               # one-way coupling an anisotropic cavity cannot reshape the tissue, so it belongs in
               # a run that says it is testing the matrix's anisotropy, not in the baseline.
               cavity_r=0.14, cavity_h=0.14, axis=2,
               n_fibres=900, fibre_len=0.16, align=0.0, align_dir=(1.0, 0.0, 0.0),
               r0=0.05, r_max=0.30, growth=0.0009, k_contact=900.0, damp=0.0,
               # MEASURED, not guessed. At 0.03 the band histogram of 23_cellfix_vertex_visible
               # ended with 39% of the matrix pinned at band 7 -- saturated, so the last quarter of
               # the run showed a uniformly white matrix and the front stopped being visible exactly
               # where it was strongest. 0.05 keeps the top band for the material that is genuinely
               # at the front.
               stress_scale=0.05, drag=0.0, wall_damp=0.6, seed=0, fps=10):
    """The whole experiment as a plain dict, ready for yaml.safe_dump + schema.load."""
    types = {f"s{i}": {"fraction": 1.0 / len(STRESS_COLORS), "youngs": youngs}
             for i in range(len(STRESS_COLORS))}
    spec = {
        "general": {"name": name, "seed": seed, "n_frames": int(n_frames), "dt": float(dt),
                    "boundary": "wall", "dim": 3, "world": [1.0, 1.0, 1.0]},
        "sets": {
            # The parent exists because the MPM provision hangs particles off a parent set; it
            # carries no physics here and is hidden in the render.
            "cell": {"n": 1, "start": [[0.5, 0.5, 0.5]], "types": types},
            # THE TYPES ARE DECLARED ON THE PARTICLE SET, NOT ONLY ON THE PARENT. Without them
            # `node_type` is a buffer of the PARENT -- the provision registers it per set
            # (base.py:348) -- so a child particle set has no per-particle type, every particle
            # inherits one colour from its parent, and a stress band cannot be drawn at all.
            # Declaring the same eight types here gives each particle its own slot. They carry
            # identical material, so which slot a particle is in changes nothing but its colour.
            "mpm_particle": {"parent": "cell", "per_parent": int(n_particles),
                             "radius": 0.48, "density": float(density), "types": types},
        },
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": int(n_grid)}},
        "operators": [
            {"op": "aggregate", "at": "cell"},
            {"op": "ecm_seed", "at": "mpm_particle", "centre": [0.5, 0.5, 0.5],
             "cavity_r": float(cavity_r), "cavity_h": float(cavity_h), "axis": int(axis),
             "n_fibres": int(n_fibres), "fibre_len": float(fibre_len),
             "align": float(align), "align_dir": list(align_dir), "seed": int(seed)},
            {"op": "cell_to_ecm", "at": "mpm_particle", "centre": [0.5, 0.5, 0.5],
             "k": float(k_contact), "r0": float(r0), "r_max": float(r_max),
             "growth": float(growth), "damp": float(damp)},
            {"op": "ecm_stress", "at": "mpm_particle", "scale": float(stress_scale),
             "bands": len(STRESS_COLORS)},
            {"op": "mpm_strain", "at": "mpm_particle"},
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid",
             "drag": float(drag), "a_max": 200},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": float(wall_damp)},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid",
             "wall_damp": float(wall_damp), "wall_contact": 0.04, "vmax": 1.0e9},
        ],
        "schedule": [
            "aggregate",
            "ecm_seed",                      # once, at frame 0
            "cell_to_ecm",                   # the ball's push, as an external acceleration
            "ecm_stress",                    # recolour by |J-1| BEFORE the frame is recorded
            {"substep_dt": float(substep_dt),
             "steps": ["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"]},
        ],
        "plotting": {
            "background": "black", "up_axis": 1, "box_frame": True, "fps": int(fps),
            "render_3d": "dots", "dot_size": 0.9, "hide_sets": ["cell"],
            "color_by": "node_type",         # the stress band written by ecm_stress
            "colors": {f"s{i}": c for i, c in enumerate(STRESS_COLORS)},
            "camera_elev": 1.05, "camera_turns": 0.0, "camera_zoom": 0.0,
        },
    }
    return copy.deepcopy(spec)
