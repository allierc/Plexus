"""The operator library. Importing this package self-registers every operator.

TEN MODULES, GROUPED BY MECHANISM, where there were fifty-five files holding one operator each.
That directory was unreadable in a specific way rather than merely long: to find out what acts on an
MPM particle you opened eight files, and the eight could not be compared because they were never on
the screen together. The two implementations of `cell_mechanics` -- the 3D AVM and the monolayer --
are the case that settles it: they are one contract with two bodies, and reading them one after the
other is how anyone can tell which one a spec is getting.

    vertex_ops           the 3D vertex model: seed, geometry, mechanics, grow, divide, die, T1
    diffusion_reaction   chemistry on the cell GRAPH, and the two shape<->chemistry couplings
    junction_ops         myosin, on junctions and across the apex, plus the cytokinetic ring
    ecm_ops              the matrix as MPM material, and the stiff blocks that confine it
    membrane_ops         the basement membrane, its crosslink network, and the integrin links
    contact_ops          where a triangulated surface meets a continuum, both directions
    mpm_ops              MLS-MPM: the grid, the four-step cycle, and the forces on it
    motion_ops           how a body moves with nothing acting on it: drag, glide, walls, gravity
    interaction_ops      pairwise laws, and the relation they act over
    field_ops            a continuum bound to a set: deposit / diffuse / decay / sense
    agent_ops            agents in a material: the two-way coupling, population, scale maps
    neural               a recurrent circuit: the local update, the signalling through W, and
                         the external field that modulates it

EVERY OLD MODULE NAME STILL IMPORTS. `plexus.operators.drag`, `plexus.operators.mpm_grid` and the
rest are re-export shims, because five prototypes reach for them by name -- `prototype/eye`,
`prototype/cardio_cells` (three files) and `prototype/inverse_slime`. They are not imported here:
each is covered by the module it points at, and importing both would register nothing twice but
would put the old names back in the reader's way.

WHAT IS NOT IN THE CORE. `mpm_boundary` and `bm_strain` are registered in `discovery_okuda` only.
`AUDIT.md` rejects both -- the first overwrites grid-node velocity, so the constraint is kinematic
and its standoff is set by the B-spline stencil width; the second is "not a mechanism" -- and a
rejection that lives only in a markdown file is one the next reader re-promotes by accident.
"""
from __future__ import annotations

# --- the mechanism modules -------------------------------------------------------------------
from . import interaction_ops       # noqa: F401  radius_graph, attraction_repulsion, squared_law,
#                                                 cohesion, separation, velocity_align, stillinger_weber
from . import motion_ops            # noqa: F401  drag, glide, velocity_cruise, sediment,
#                                                 attractor_flow, gravity, bounce
from . import field_ops             # noqa: F401  the grid field, deposit, diffuse, decay, sense,
#                                                 chemotax, playback, pacemaker, activation_pulse, signal
from . import mpm_ops               # noqa: F401  mpm_grid + p2g/grid_update/g2p/strain, anchor,
#                                                 spin, apply_material_map, and the fenced oracle
from . import agent_ops             # noqa: F401  agent_scatter/gather/remodel, agent_divide/grow,
#                                                 polarity, active force+stress, aggregate/broadcast,
#                                                 seed_from_segmentation
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
from . import neural                # noqa: F401  neuron_update (phi), neuron_signal (psi: shared |
#                                                 type_pre | type_pairwise), neuron_field_input (Omega)

__all__ = ["interaction_ops", "motion_ops", "field_ops", "mpm_ops", "agent_ops",
           "vertex_ops", "diffusion_reaction", "junction_ops", "ecm_ops", "membrane_ops",
           "contact_ops", "neural"]
