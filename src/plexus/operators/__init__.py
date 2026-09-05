"""The operator library. Importing this package self-registers every operator.

The modules are grouped BY MECHANISM, not one file per operator. The alternative is unreadable in
a specific way rather than merely long: finding out what acts on an MPM particle would mean
opening eight files, and the eight could never be compared because they would never be on the
screen together. The two implementations of `cell_mechanics` -- the 3D vertex model and the
monolayer -- settle it. They are one contract with two bodies, and reading them one after the
other is how anyone can tell which one a specification is getting.

    vertex_ops           the 3D vertex model: seed, geometry, mechanics, grow, divide, die, T1
    diffusion_reaction   chemistry on the cell GRAPH, and the two shape-chemistry couplings
    junction_ops         myosin, on junctions and across the apex, plus the cytokinetic ring
    ecm_ops              the matrix as MPM material, and the stiff blocks that confine it
    membrane_ops         the basement membrane, its crosslink network, and the integrin links
    contact_ops          where a triangulated surface meets a continuum, both directions
    mpm_ops              MLS-MPM: the grid, the four-step cycle, and the forces on it
    motion_ops           single-body motion: drag, glide, sediment, walls, gravity
    encoding_ops         fields that REPRESENT rather than simulate: hash_encoding, voxelize
    interaction_ops      pairwise laws, and the relation they act over
    field_ops            a continuum bound to a set: deposit / diffuse / decay / sense
    neural               a recurrent circuit: the seed that places a connectome region, the
                         local update, the signalling through W, and the field modulating it

Every module opens with its own contracts listed IN THE ORDER THEY APPEAR IN THE FILE, then the
models and implementations of those contracts. Each contract's docstring states the mechanism, its
typed morphism, the equation it implements, every symbol in that equation with its units, and the
publication it comes from where there is one.

There are no per-operator module names. A module that re-exports one operator reads as a separate
mechanism when it is one line of a file, so an import of a name that used to exist raises rather
than resolving -- which is the correct answer and not a regression.
"""
from __future__ import annotations

# --- the mechanism modules -------------------------------------------------------------------
from . import encoding_ops          # noqa: F401  hash_encoding -- a learnable field from its
                                   #               own coordinates; voxelize -- a set, splatted
from . import interaction_ops       # noqa: F401  radius_graph, attraction_repulsion, squared_law,
#                                                 cohesion, separation, velocity_align, stillinger_weber
from . import motion_ops            # noqa: F401  drag, glide, velocity_cruise, sediment,
#                                                 attractor_flow, gravity, bounce
from . import field_ops             # noqa: F401  the grid field, deposit, diffuse, decay, sense,
#                                                 chemotax, playback, pacemaker, activation_pulse, signal
from . import mpm_ops               # noqa: F401  mpm_grid + p2g/grid_update/g2p/strain, anchor,
#                                                 spin, apply_material_map, and the fenced oracle
from . import vertex_ops            # noqa: F401  seed_mesh, cell_mechanics, cell_divide, cell_die,
#                                                 edge_flip, topo_record
from . import diffusion_reaction    # noqa: F401  seed_cell_chem, cell_chem_diffuse/react,
#                                                 cell_geometry, cell_grow, cell_chem_from_shape,
#                                                 cell_shape_probe, interface_tension/push
from . import junction_ops          # noqa: F401  junction_myosin (default|two_pool), junction_sync,
#                                                 medioapical_myosin, cytokinetic_ring
from . import ecm_ops               # noqa: F401  ecm_seed/stress/from_cell, cell_exclude,
#                                                 block_seed/stress
from . import membrane_ops          # noqa: F401  bm_*, adhesion_*, integrin_*
from . import contact_ops           # noqa: F401  mesh_contact, mesh_inside, surface_track,
#                                                 plate_confine, bm_sense, ecm_load, ecm_gate_growth
from . import neural                # noqa: F401  neural_seed, neuron_update (phi), neuron_signal
#                                                 (psi: shared | type_pre | type_pairwise),
#                                                 neuron_field_input (Omega)
# The continuous-flow operators live with their engine but must still register here. `mpm_emit`
# and `mpm_drain` are defined in plexus/continuous_engine.py, beside the run() that refuses a
# specification using one without the other. Importing them only when that engine is selected
# would be too late: `schema.load` validates every operator NAME before anything chooses an
# engine, so a specification naming mpm_emit would fail before the engine key was read. That
# module imports models.base and models.registry only, so there is no cycle back into this one.
from plexus import continuous_engine   # noqa: F401  mpm_emit / mpm_drain

__all__ = ["encoding_ops", "interaction_ops", "motion_ops", "field_ops", "mpm_ops",
           "vertex_ops", "diffusion_reaction", "junction_ops", "ecm_ops", "membrane_ops",
           "contact_ops", "neural", "continuous_engine"]
