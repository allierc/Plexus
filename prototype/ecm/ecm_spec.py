"""ecm_spec -- build the spec for "a cell ball grows inside a fibrous matrix".

The spec is the deliverable, as everywhere else in Plexus: sets / fields / operators / schedule,
validated by the stock `plexus.schema.load` and run by the stock `plexus.engine.run`. Nothing
here lives in the engine; importing `ecm_ops` registers the three new operators.

    sets       cell        -- the parent the matrix particles hang off (MPM needs a parent set)
               mpm_particle -- the MATRIX itself: one particle set, seeded into fibres
    fields     mpm_grid    -- the background grid; when the real cell mesh arrives it scatters
                              into THIS grid, which is what makes the coupling two-way
    schedule   aggregate -> seed_ecm -> cell_to_ecm -> ecm_stress
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
# AND THE LOW END HAS TO SEPARATE FROM BAND 0. The first version ramped 0 -> 1 -> 2 as
# slate -> dark purple -> dark magenta: three dark colours, so the ONSET of stress -- the outermost edge
# of the front, where most of the strained material is -- was a hue shift inside a dark fog and read as
# nothing. The ramp now brightens immediately: band 1 is a clearly lit violet, band 2 a bright magenta,
# and the top four keep the inferno hues they had, so a run whose strain sits in bands 1-2 is legible
# without changing what the top of the scale means.
STRESS_COLORS = [
    [0.16, 0.19, 0.30], [0.46, 0.31, 0.72], [0.78, 0.26, 0.72], [0.95, 0.28, 0.45],
    [0.99, 0.45, 0.20], [1.00, 0.62, 0.12], [1.00, 0.80, 0.28], [1.00, 0.97, 0.78],
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
               # a DENSER cone about  -- see ecm_ops.ECMSeed
               dense_axis=2, dense_cone_deg=0.0, dense_boost=1.0,
               # RIGIDITY OF THE CELL-MATRIX CONTACT, raised from 900. Matrix particles were visibly
               # ending up inside the lumen, and `cell_exclude_3d` is the hard backstop; this is the soft
               # part of the same fix, the one that keeps the stress physical.
               #
               # A CORRECTION TO AN EARLIER COMMENT HERE, which said `mpm_scatter` clamps this penalty at
               # `a_max` so that k and a_max together set a maximum useful depth. THAT WAS WRONG. Read
               # `mpm_scatter.forward`: `a_max` clamps only the PARENT set's delta -- gravity -- while the
               # per-particle delta this operator emits, `H.delta(p.name)`, is passed through
               # `nan_to_num` and never clamped. So the penalty was never truncated, and the ceiling that
               # explanation invoked does not exist. The MEASUREMENTS below stand; the mechanism offered
               # for them did not, and a plausible mechanism fitted to a real number afterwards is the
               # more dangerous of the two errors.
               #
               # WHAT IS MEASURED. At k = 4000 (with a_max = 800, which turns out to be irrelevant) the matrix
               # stopped being pushed and started being FLICKED: a huge acceleration on the contact
               # layer alone accelerates that layer out of the contact zone before it can transmit
               # anything to the material behind it, so the median particle displacement collapsed from
               # 0.085 to 0.0005 and the strained fraction from 0.97 to 0.21. A piston pushes bulk; an
               # impulse does not. 1200/300 is firm enough to keep penetration rare and gentle enough
               # to transmit a sustained pressure, with `cell_exclude_3d` guaranteeing the constraint.
               r0=0.05, r_max=0.30, growth=0.0009, k_contact=1200.0, damp=0.0,
               # MEASURED, not guessed. At 0.03 the band histogram of 23_cellfix_vertex_visible
               # ended with 39% of the matrix pinned at band 7 -- saturated, so the last quarter of
               # the run showed a uniformly white matrix and the front stopped being visible exactly
               # where it was strongest. 0.05 keeps the top band for the material that is genuinely
               # at the front.
               stress_scale=0.05, stress_measure="vol", drag=0.0, wall_damp=0.6, seed=0,
               fps=10, a_max=300.0,
               # AN ELASTIC BLOCK instead of a rigid plate: a SECOND MPM set, ~130x stiffer than the
               # matrix, scattering into the same grid (`prototype/eye`'s two-body pattern). None =
               # no block. A rigid projection cannot be seen to do anything; a material can.
               block_gap=None, block_youngs=2000.0, block_particles=60000,
               block_stress_scale=0.004, block_measure="vol",
               # THE BASEMENT MEMBRANE: a third MPM set -- one stiff CROSSLINKED shell just outside the
               # epithelium, between it and the stroma. `membrane` is the path to the tissue cache whose
               # frame-0 surface it is laid on; None = absent. See membrane_ops for what it is and what
               # this discretisation cannot claim about it.
               membrane=None, membrane_particles=30000, membrane_youngs=400.0,
               membrane_bond_k=2.0e5, membrane_cutoff=0.020, membrane_break=0.35,
               membrane_offset=0.004, membrane_thickness=0.002, membrane_adhesion=2.0e4,
               membrane_jitter=0.35,
               # crosslink turnover, in frames. Large = a membrane that cannot keep up with growth and
               # must fragment; small = one that remodels and stays intact. 0 or None disables it.
               membrane_tau=60.0, membrane_reserve=0.0, membrane_secrete_rate=0.02,
               membrane_secrete_targeted=1.0, membrane_deposit="uniform", membrane_rebond_every=20, membrane_remodel_target="own",
               membrane_mesh_w=1.0, membrane_relax_new=4, membrane_relax_every=20, membrane_relax_sweeps=3,
               # l* is the mesh size the network is BUILT to, held fixed rather than tracking the sheet's
               # own mean; 0 freezes it from frame 0. `repel_w` is k_repel/k_bond -- a ratio, so it
               # transfers between calibrations, and it is bounded above by the integrator, not the sheet.
               membrane_l_star=0.0, membrane_repel_w=0.0, membrane_repel_range=1.0,
               # "linear" = one-sided spring, range 1 l*. "ar" = Plexus's attraction_repulsion law with
               # its attractive term dropped, range ~3 l* -- the form CGI uses to scatter points evenly
               # over a surface, and the one that took the frozen sheet from d/hex 0.677 to 0.895.
               membrane_repel_law="linear", membrane_repel_sigma=0.7, membrane_repel_k=0.0,
               # False = NO CROSSLINK NETWORK. The sheet is then a pure MPM continuum: elasticity from
               # the deformation gradient, contact with epithelium and stroma through the shared grid,
               # separation where material thins past the grid's support. Nothing new is added -- this is
               # the mechanics that was already underneath the springs and invisible while they ran.
               membrane_springs=True,
               # where the unsecreted reserve waits. Outside the box by default; (0.5,0.5,0.5) is the
               # old tissue-centre parking, kept so the two can be compared directly.
               membrane_park=(-0.25, -0.25, -0.25),
               # how far outside the surface the boundary condition reaches, in grid cells. It sets the
               # standoff the sheet settles at, and 2.0 is what put a visible gap between the epithelium
               # and everything around it -- the matrix as well as the membrane, since the constraint
               # acts on every massed node. Measured at the end of the run, BM radius minus tissue
               # radius:  band 2.0 -> +0.0181,  band 1.0 -> +0.0015,  band 0.5 -> -0.0062.
               # 1.0 sits the sheet on the surface; 0.5 is too weak to hold it out and it sinks inside.
               membrane_band=1.0,
               # frames over which material found INSIDE the surface is pushed back out. 0 = the old
               # velocity-only constraint, which cannot recover ground already lost.
               membrane_recover=2.0,
               # per-particle non-penetration instead of the grid boundary condition. 0 = off.
               membrane_contact_k=0.0,
               # fraction of the sheet that carries an integrin anchor. Adhesion is punctate in vivo --
               # hemidesmosome plaques with membrane spanning between -- so < 1 is the faithful case.
               membrane_adhesion_fraction=1.0,
               # HEMIDESMOSOMES AS THEIR OWN SET. n_adhesions = 0 keeps the old field-on-the-membrane
               # tether; > 0 replaces it with discrete plaques that form, bear load, rupture and re-bind.
               n_adhesions=0, adhesion_k=1.0e4, adhesion_gamma=1.0, adhesion_rupture=0.0,
               # THE HYBRID. MPM keeps the sheet's internal material response -- which is what gives a
               # continuum its homogeneity, coverage 1.000 with no bond network -- while adhesion and
               # contact act DIRECTLY on the particle, engine-integrated and overdamped, exactly as they
               # do in graph mode. That distinction is what run 70 demonstrates: at the SAME grid and
               # the same sheet thickness its membrane sits properly on the surface, because in graph
               # mode the sheet never touches the grid and its springs move particles at float
               # precision. In MPM every force is routed scatter -> grid solve -> gather, so even a
               # per-particle contact is smeared to cell scale before it acts. The grid cannot IMPOSE a
               # sub-cell position; it has no trouble holding one.
               membrane_direct_forces=False,
               # adhesions rupture under load: `detach` is the displacement at which one lets go.
               membrane_detach=0.0,
               # THE TISSUE AS A MOVING BOUNDARY ON THE GRID, rather than a positional projection.
               # 88 showed the projection carries no strain -- see membrane_ops.MPMTissueBoundary.
               membrane_grid_bc=False,
               membrane_repel_aggr="",
               membrane_tau_adh=0.0, membrane_aniso=1.0, membrane_record_hoop=False, membrane_surface_level=False,
               membrane_impl="mpm", membrane_drag=40.0, membrane_inertial=False,
               membrane_gamma=2.0e3):
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
            {"op": "seed_ecm", "at": "mpm_particle", "centre": [0.5, 0.5, 0.5],
             "cavity_r": float(cavity_r), "cavity_h": float(cavity_h), "axis": int(axis),
             "n_fibres": int(n_fibres), "fibre_len": float(fibre_len),
             "align": float(align), "align_dir": list(align_dir), "seed": int(seed),
             "dense_axis": int(dense_axis), "dense_cone_deg": float(dense_cone_deg),
             "dense_boost": float(dense_boost)},
            {"op": "cell_to_ecm", "at": "mpm_particle", "centre": [0.5, 0.5, 0.5],
             "k": float(k_contact), "r0": float(r0), "r_max": float(r_max),
             "growth": float(growth), "damp": float(damp)},
            {"op": "ecm_stress", "at": "mpm_particle", "scale": float(stress_scale),
             "bands": len(STRESS_COLORS), "measure": str(stress_measure)},
            {"op": "mpm_strain", "at": "mpm_particle"},
            {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid",
             "drag": float(drag), "a_max": float(a_max),
             "store_stress": (stress_measure == "vonmises")},
            {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": float(wall_damp)},
            {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid",
             "wall_damp": float(wall_damp), "wall_contact": 0.04, "vmax": 1.0e9},
        ],
        "schedule": [
            "aggregate",
            "seed_ecm",                      # once, at frame 0
            "cell_to_ecm",                   # the ball's push, as an external acceleration
            "ecm_stress",                    # recolour by |J-1| BEFORE the frame is recorded
            {"substep_dt": float(substep_dt),
             "steps": (["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_tissue_boundary",
                        "mpm_gather"] if (membrane and membrane_grid_bc) else
                       ["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"])},
        ],
        "plotting": {
            "background": "black", "up_axis": 1, "box_frame": True, "fps": int(fps),
            "render_3d": "dots", "dot_size": 0.9, "hide_sets": ["cell"],
            "color_by": "node_type",         # the stress band written by ecm_stress
            "colors": {f"s{i}": c for i, c in enumerate(STRESS_COLORS)},
            "camera_elev": 1.05, "camera_turns": 0.0, "camera_zoom": 0.0,
        },
    }
    if membrane is not None:
        import membrane_ops                                           # noqa: F401  register it
        import surface_ops                                            # noqa: F401
        spec["sets"]["basement_membrane_particle"] = {
            "parent": "cell", "per_parent": int(membrane_particles), "radius": 0.48,
            "density": float(density),
            "types": {f"m{i}": {"fraction": 1.0 / len(STRESS_COLORS),
                                "youngs": float(membrane_youngs)}
                      for i in range(len(STRESS_COLORS))}}
        spec["operators"] += [
            {"op": "seed_basement_membrane", "at": "basement_membrane_particle",
             "centre": [0.5, 0.5, 0.5], "surface": str(membrane), "scale": 1.0,
             "offset": float(membrane_offset), "thickness": float(membrane_thickness),
             "seed": int(seed), "jitter": float(membrane_jitter),
             "reserve": float(membrane_reserve), "park": list(membrane_park)},
            # INTEGRINS FIRST: without them the sheet slides over the epithelium and its bonds never
            # feel the growth. `membrane_adhesion = 0` reproduces the unanchored (wrong) loading path.
            {"op": "integrin_adhesion", "at": "basement_membrane_particle",
             "centre": [0.5, 0.5, 0.5], "surface": str(membrane), "scale": 1.0,
             "k": float(membrane_adhesion), "offset": float(membrane_offset),
             "detach": float(membrane_detach),
             "tau_adh": float(membrane_tau_adh),
             "fraction": float(membrane_adhesion_fraction)},
            {"op": "basement_membrane_bond", "at": "basement_membrane_particle",
             "k": float(membrane_bond_k), "cutoff": float(membrane_cutoff),
             "aniso": float(membrane_aniso), "record_hoop": bool(membrane_record_hoop),
             "max_neighbours": 6},
            {"op": "basement_membrane_remodel", "at": "basement_membrane_particle",
             "tau": float(membrane_tau), "cap": 0.02,
             "target": str(membrane_remodel_target), "mesh_w": float(membrane_mesh_w),
             "l_star": float(membrane_l_star)},
            # EXCLUDED VOLUME, beside the springs rather than instead of them. Ordered AFTER the bond
            # operator because it borrows that operator's neighbour search, and after the remodel because
            # the remodel is what freezes l*.
            {"op": "basement_membrane_repel", "at": "basement_membrane_particle",
             # absolute if given, otherwise the ratio w * k_bond. The absolute form exists so the
             # repulsion survives a run with the crosslink springs turned off, where w * k_bond is 0.
             "k": (float(membrane_repel_k) if membrane_repel_k > 0.0
                   else float(membrane_repel_w) * float(membrane_bond_k)),
             "every": int(membrane_rebond_every),
             "max_neighbours": 6 if membrane_repel_law == "linear" else 18,
             "range_scale": float(membrane_repel_range),
             "law": str(membrane_repel_law), "sigma_scale": float(membrane_repel_sigma),
             **({"aggr": str(membrane_repel_aggr)} if membrane_repel_aggr else {})},
            {"op": "surface_track", "at": "surface", "centre": [0.5, 0.5, 0.5],
             "surface": str(membrane), "scale": 1.0, "k": 6,
             "seed": int(seed), "jitter": float(membrane_jitter)},
            {"op": "basement_membrane_secrete", "at": "basement_membrane_particle",
             "centre": [0.5, 0.5, 0.5], "rate": float(membrane_secrete_rate),
             "targeted": float(membrane_secrete_targeted),
             "deposit": str(membrane_deposit),
             "relax_new": int(membrane_relax_new),
             "relax_every": int(membrane_relax_every),
             "relax_sweeps": int(membrane_relax_sweeps)},
            {"op": "basement_membrane_crosslink", "at": "basement_membrane_particle",
             "every": int(membrane_rebond_every), "cutoff": float(membrane_cutoff),
             "max_neighbours": 6},
            {"op": "basement_membrane_bond_break", "at": "basement_membrane_particle",
             "break_strain": float(membrane_break), "components_every": 40},
            # the MLS-MPM cycle for the third body. APPENDED, so its scatter accumulates AFTER the
            # stroma's has reset the grid -- the ordering `mpm_scatter[accumulate]` depends on.
            {"op": "mpm_strain", "at": "basement_membrane_particle"},
            {"op": "mpm_scatter", "at": "basement_membrane_particle", "to": "mpm_grid",
             "implementation": "accumulate", "drag": float(drag), "a_max": float(a_max)},
            {"op": "mpm_gather", "at": "basement_membrane_particle", "from": "mpm_grid",
             "wall_damp": float(wall_damp), "wall_contact": 0.04, "vmax": 1.0e9},
        ]
        i = spec["schedule"].index("seed_ecm") + 1
        spec["schedule"].insert(i, "seed_basement_membrane")
        if not membrane_springs:
            # THE MEMBRANE AS A CONTINUUM, with no crosslink network at all. Everything the sheet does
            # then comes from MPM: the deformation gradient carries the elastic response, the shared grid
            # carries contact with both the epithelium and the stroma, and material that thins past the
            # grid's support separates on its own. Nothing here is a new mechanism -- it is the mechanics
            # that was already present underneath the springs and could not be seen while they ran.
            drop = {"basement_membrane_bond", "basement_membrane_remodel",
                    "basement_membrane_crosslink", "basement_membrane_bond_break",
                    "basement_membrane_repel"}
            if membrane_adhesion <= 0.0:
                drop.add("integrin_adhesion")
            spec["operators"] = [o for o in spec["operators"] if o["op"] not in drop]
            spec["operators"].append({"op": "basement_membrane_continuum_strain",
                                      "at": "basement_membrane_particle"})
            if n_adhesions > 0:
                # its own SET, hung off the same parent the other particle sets use
                spec["sets"]["adhesion"] = {"parent": "cell", "per_parent": int(n_adhesions),
                                            "radius": 0.48, "density": float(density)}
                spec["operators"] += [
                    {"op": "adhesion_seed", "at": "adhesion", "centre": [0.5, 0.5, 0.5],
                     "surface": str(membrane), "scale": 1.0, "seed": int(seed),
                     "membrane_set": "basement_membrane_particle"},
                    {"op": "adhesion_pull", "at": "basement_membrane_particle", "adhesion_set": "adhesion",
                     "centre": [0.5, 0.5, 0.5], "surface": str(membrane), "scale": 1.0,
                     "k": float(adhesion_k), "gamma": float(adhesion_gamma),
                     "offset": float(membrane_offset)},
                    {"op": "adhesion_turnover", "at": "basement_membrane_particle",
                     "centre": [0.5, 0.5, 0.5], "rupture": float(adhesion_rupture)},
                ]
                i2 = spec["schedule"].index("seed_basement_membrane") + 1
                spec["schedule"].insert(i2, "adhesion_seed")
                j2 = spec["schedule"].index("ecm_stress")
                spec["schedule"].insert(j2, "adhesion_pull")
                spec["schedule"].insert(j2 + 1, "adhesion_turnover")
                # the adhesion set is a marker, not a material: keep it out of the MPM cycle
                spec["operators"] = [o for o in spec["operators"]
                                     if not (o.get("at") == "adhesion" and o["op"].startswith("mpm_"))]
            if membrane_contact_k > 0.0:
                spec["operators"].append(
                    {"op": "basement_membrane_contact", "at": "basement_membrane_particle",
                     "centre": [0.5, 0.5, 0.5], "surface": str(membrane), "scale": 1.0,
                     "k": float(membrane_contact_k), "offset": float(membrane_offset)})
                # INSIDE THE SUBSTEP LOOP, not at frame level. An mpm_acceleration computed once per
                # frame is re-applied at every substep, so the governing timestep for a stiff contact is
                # dt_frame and not dt_sub: dt_frame*sqrt(k) is 0.40 at k=1e4 and 0.89 at 5e4 (the last
                # stable run), then 2.53 at 4e5 and 5.66 at 2e6 -- both of which blew the sheet apart.
                # That capped the contact near k = 6e4 while closing the gap needs ~4e5. Evaluated at
                # the substep the limit becomes dt_sub*sqrt(k) < 1, i.e. k up to 2.5e7.
                for _st in spec["schedule"]:
                    if isinstance(_st, dict) and "substep_dt" in _st:
                        _st["steps"].insert(0, "basement_membrane_contact")
                        break
            if membrane_grid_bc:
                spec["operators"].append(
                    {"op": "mpm_tissue_boundary", "at": "mpm_grid", "centre": [0.5, 0.5, 0.5],
                     "surface": str(membrane), "scale": 1.0, "dt_frame": float(dt),
                     "band": float(membrane_band), "recover": float(membrane_recover)})
            j = spec["schedule"].index("ecm_stress")
            if "integrin_adhesion" not in drop:
                spec["schedule"].insert(j, "integrin_adhesion"); j += 1
            spec["schedule"].insert(j, "basement_membrane_continuum_strain")
        if membrane_springs:
            # the bond force is a DYNAMICS operator (EMIT mpm_acceleration): it must run before the substep
            # block so the engine has its delta to integrate, and the break check after it.
            i = spec["schedule"].index("ecm_stress")
            spec["schedule"].insert(i, "integrin_adhesion")
            spec["schedule"].insert(i + 1, "basement_membrane_bond")
            if membrane_tau and membrane_tau > 0:
                spec["schedule"].insert(i + 2, "basement_membrane_remodel")
                spec["schedule"].insert(i + 3, "basement_membrane_crosslink")
                spec["schedule"].insert(i + 4, "basement_membrane_bond_break")
            else:
                spec["operators"] = [o for o in spec["operators"]
                                     if o["op"] != "basement_membrane_remodel"]
                spec["schedule"].insert(i + 2, "basement_membrane_bond_break")
            if membrane_repel_w > 0.0 or membrane_repel_k > 0.0:
                # after the remodel, which is what freezes l*, and after the bond operator, whose neighbour
                # search it borrows. w = 0 leaves the operator out entirely, so every run to date is
                # unchanged bit-for-bit.
                after = ("basement_membrane_remodel" if "basement_membrane_remodel" in spec["schedule"]
                         else "basement_membrane_bond")
                spec["schedule"].insert(spec["schedule"].index(after) + 1, "basement_membrane_repel")
            else:
                spec["operators"] = [o for o in spec["operators"] if o["op"] != "basement_membrane_repel"]
            if membrane_direct_forces:
                for o in spec["operators"]:
                    if o["op"] in ("integrin_adhesion", "basement_membrane_contact"):
                        o["emit"] = "velocity"          # x += dt*(F/gamma): first-order, no inertia
                        o["overdamped_gamma"] = float(membrane_gamma)
                        o["graph_mode"] = False
                # and out of the substep block: an engine-integrated delta is applied once per frame
                for _st in spec["schedule"]:
                    if isinstance(_st, dict) and "substep_dt" in _st:
                        _st["steps"] = [x for x in _st["steps"] if x != "basement_membrane_contact"]
        if membrane_impl == "graph":
                # THE MEMBRANE AS A SPRING GRAPH, not a continuum body. Nothing about the sheet's mechanics
                # was ever coming from MPM: the crosslinks hold it together, the integrin springs hold it in
                # place, bonds breaking fragment it, and every figure colours it by crosslink strain. MPM was
                # buying exactly one thing -- momentum exchange with the matrix through the shared grid --
                # and at n_grid=48 that grid cannot resolve a 0.002-thick sheet anyway.
                #
                # `emit: acceleration` hands the bond and integrin forces to the ENGINE (v += dt*a; x += dt*v)
                # instead of routing them into the MPM substep as an external body force. `drag` replaces the
                # damping the grid was providing implicitly -- an undamped spring network rings forever, and
                # the grid transfer was quietly acting as a low-pass filter on it.
                #
                # The MPM buffers (F, C, mass, p_vol) are still provisioned and now unused. That is deliberate
                # rather than tidy: `mass` is what parks the unsecreted reserve, and dropping the entity would
                # rename the set and touch every reference to it in the spec, the renderer and the analysis.
                keep = {"mpm_strain", "mpm_scatter", "mpm_gather"}
                spec["operators"] = [o for o in spec["operators"]
                                     if not (o["op"] in keep and o["at"] == "basement_membrane_particle")]
                # OVERDAMPED, NOT INERTIAL. `emit: acceleration` integrates v += dt*a; x += dt*v with unit
                # mass -- a term with no physical basis at Re ~ 1e-10, where the equation of motion is
                # gamma*x_dot = F. Everything section 5 and 7 report about the sheet -- the undamped spring
                # oscillating about a moving anchor, critical damping c = 2*sqrt(k), the tracking lag, the
                # sinking, the stability reversal above k_adh = 1e5 -- is the phenomenology of that inertia.
                # In the overdamped limit none of it exists. `emit: velocity` gives x += dt*(F/gamma), which
                # is the correct low-Reynolds motion and removes both the oscillation and the ceiling that
                # came with it. `membrane_inertial=True` restores the old path for comparison only.
                emit = "acceleration" if membrane_inertial else "velocity"
                for o in spec["operators"]:
                    if o["op"] in ("basement_membrane_bond", "integrin_adhesion",
                                   "basement_membrane_repel"):
                        o["emit"] = emit
                        o["overdamped_gamma"] = 0.0 if membrane_inertial else float(membrane_gamma)
                        # EXPLICIT, not inferred from `emit`. The spec's `emit` key is consumed by the
                        # engine's emit resolution and does not survive into the round-tripped yaml, so an
                        # operator that keys its stability check off it silently skips the check -- which is
                        # what happened: k = 2e5 ran to completion in graph mode and returned an infinite
                        # strain with no warning.
                        o["graph_mode"] = True
                        if o["op"] == "basement_membrane_bond":
                            o["k_adhesion_hint"] = float(membrane_adhesion)
                if membrane_inertial:
                    # `drag` only makes sense against inertia. Overdamped, the dissipation IS gamma.
                    spec["operators"].append({"op": "drag", "at": "basement_membrane_particle",
                                              "k": float(membrane_drag)})
                    i = spec["schedule"].index("basement_membrane_bond")
                    spec["schedule"].insert(i + 1, "drag")
                else:
                    spec["operators"] = [o for o in spec["operators"] if o["op"] != "drag"]
                # (no schedule surgery: the substep block dispatches mpm steps BY OP NAME, so dropping the
                # three operator entries above is what removes them from the cycle.)
        if membrane_surface_level:
            # `per_parent`, not `n`: a set with a parent is provisioned per parent, and `n` is
            # simply not read -- the build dies on a KeyError several frames of setup later.
            spec["sets"]["surface"] = {"parent": "cell", "per_parent": int(membrane_particles),
                                       "radius": 0.48}
            # FIRST, AND BEFORE THE SEED. The Level is authoritative: it owns the lattice, and both
            # the seed and the adhesion read it. The dependency runs one way, so there is no second
            # place where a direction or a radius can be computed slightly differently.
            spec["schedule"].insert(0, "surface_track")
            for o in spec["operators"]:
                if o["op"] in ("integrin_adhesion", "seed_basement_membrane"):
                    o["surface_set"] = "surface"
        else:
            spec["operators"] = [o for o in spec["operators"] if o["op"] != "surface_track"]
        if membrane_reserve and membrane_reserve > 0:
            # LAST IN THE FRAME. `basement_membrane_bond` must have run at least once so
            # `H.membrane_bonds` exists to choose a strained bond from, and the rebuild it triggers is
            # picked up at the top of the next frame, so new material is bonded in before it is loaded.
            spec["schedule"].append("basement_membrane_secrete")
        else:
            spec["operators"] = [o for o in spec["operators"]
                                 if o["op"] != "basement_membrane_secrete"]
    if block_gap is not None:
        # ONE SET, ONE MATERIAL. The block's stiffness is a property of its TYPE, which is why it has
        # to be a separate set rather than extra types on the matrix: `ecm_stress` rewrites
        # `node_type` every frame to carry the stress band, so a type that meant "stiff" would have
        # its material reassigned by the colouring. Sets are the boundary the material lives behind.
        spec["sets"]["mpm_block"] = {
            "parent": "cell", "per_parent": int(block_particles), "radius": 0.48,
            "density": float(density),
            "types": {f"b{i}": {"fraction": 1.0 / len(STRESS_COLORS),
                                "youngs": float(block_youngs)}
                      for i in range(len(STRESS_COLORS))}}
        spec["operators"] += [
            {"op": "seed_block", "at": "mpm_block", "centre": [0.5, 0.5, 0.5],
             "axis": int(axis), "gap_half": float(block_gap), "seed": int(seed)},
            {"op": "block_stress", "at": "mpm_block", "scale": float(block_stress_scale),
             "bands": len(STRESS_COLORS), "measure": str(block_measure)},
            # APPENDED, WHICH IS WHAT PUTS THEM SECOND. One schedule token runs every operator of
            # that name in spec order, and `mpm_scatter`'s default implementation RESETS the grid
            # while `accumulate` adds to it -- so the matrix's scatter must come first or the block
            # would wipe the matrix out of the grid every substep.
            {"op": "mpm_strain", "at": "mpm_block"},
            {"op": "mpm_scatter", "at": "mpm_block", "to": "mpm_grid",
             "implementation": "accumulate", "drag": float(drag), "a_max": float(a_max),
             "store_stress": (block_measure == "vonmises")},
            {"op": "mpm_gather", "at": "mpm_block", "from": "mpm_grid",
             "wall_damp": float(wall_damp), "wall_contact": 0.04, "vmax": 1.0e9},
        ]
        i = spec["schedule"].index("seed_ecm") + 1
        spec["schedule"].insert(i, "seed_block")
        i = spec["schedule"].index("ecm_stress") + 1
        spec["schedule"].insert(i, "block_stress")
    return copy.deepcopy(spec)
