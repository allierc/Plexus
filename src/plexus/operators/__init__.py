"""Operator library. Importing this package self-registers every operator into
the registry (each module calls `@register_operator`). The engine imports this so
a spec's operator names resolve; `registry.catalog_summary()` then lists them.

Start small: the analytic attraction_repulsion law only. Port more (mpm,
secrete/sense, the grow/divide line) into new one-concern modules here as we
scale up.
"""
from __future__ import annotations

from . import graph                 # noqa: F401  registers radius_graph (rewire)
from . import aggregate             # noqa: F401  children -> parent reduction (centroid)
from . import broadcast             # noqa: F401  parent -> children lift (containment)
from . import attraction_repulsion  # noqa: F401  registers attraction_repulsion (lateral, 1st-derivative)
from . import squared_law                # noqa: F401  registers squared_law (lateral, 2nd-derivative: Coulomb electrostatics OR Newtonian gravity)
from . import stillinger_weber       # noqa: F401  SW two+three-body tetrahedral potential (mW water/Si/Ge; 1st many-body force)
from . import attractor_flow         # noqa: F401  registers attractor_flow (lateral, 1st-derivative: strange-attractor ODE flow dx/dt = f(x))
from . import cohesion              # noqa: F401  boids steering rule (lateral, 2nd-derivative)
from . import velocity_align       # noqa: F401  Vicsek velocity alignment (nominal); boids = special case (was alignment)
from . import separation            # noqa: F401  boids steering rule (lateral, 2nd-derivative)
from . import velocity_cruise      # noqa: F401  Vicsek self-propulsion to speed v0 (2nd-order; was cruise)
from . import drag                  # noqa: F401  registers drag (lateral, 2nd-derivative)
# field-coupled primitives (the slime/Physarum decomposition: 1 set + 1 scalar field)
from . import scalar_field          # noqa: F401  registers the `grid` scalar field
from . import deposit               # noqa: F401  set -> field
from . import diffuse               # noqa: F401  field -> field
from . import decay                 # noqa: F401  field -> field
from . import sense                 # noqa: F401  field -> set, sensor-fan steering (2D/3D, vector heading)
from . import glide                # noqa: F401  slime self-propulsion: glide along the heading (1st-order, overdamped)
from . import bounce                # noqa: F401  set -> wall/obstacle reflection (2D/3D specular re-head)
from . import prescribed_field      # noqa: F401  registers the `prescribed` field + playback
from . import chemotax              # noqa: F401  field -> set gradient coupling; emit: velocity|mpm_acceleration (merges chemotaxis+chemo_force)
from . import gravity               # noqa: F401  cell-level body force (feeds the MPM substep)
from . import sediment              # noqa: F401  agent-level per-type directional drift (differential sedimentation)
# active-stimulus decomposition (clock -> activation field -> contraction -> MPM):
from . import pacemaker             # noqa: F401  periodic scalar clock p(t) -> H.signals (field)
from . import activation_pulse      # noqa: F401  clocked activation field: shared-clock profile OR per-pixel delayed wave (merges pulse_stimulus+phase_delay_pulse)
from . import active_force          # noqa: F401  activation gradient -> per-particle force (exchange; alias pulse_to_contraction)
from . import active_stress         # noqa: F401  activation -> per-particle active stress -A nn^T (exchange; alias pulse_to_active_stress)
from . import mpm_spin              # noqa: F401  drive MPM body toward slow solid-body rotation (lateral)
from . import mpm_anchor            # noqa: F401  substrate/boundary rest-anchor k*(rest-pos) (lateral)
from . import material_map          # noqa: F401  image field + apply_material_map (per-particle stiffness)
from . import signal               # noqa: F401  passive connectome signalling (lateral, 1st-order voltage ODE)
from . import mpm                   # noqa: F401  FENCED TRANSITIONAL oracle: MLS-MPM mechanics (mls_mpm_mechanics)
# Phase-3 decomposition of the oracle -- one file per operator + the shared grid field:
from . import mpm_grid              # noqa: F401  the mpm_grid background FIELD + B-spline kernel
from . import mpm_strain            # noqa: F401  particle -> particle  (F + material update)
from . import mpm_scatter        # noqa: F401  particle -> mpm_grid   (scatter; was p2g)
from . import mpm_grid_update       # noqa: F401  mpm_grid -> mpm_grid    (grid solve + BCs)
from . import mpm_gather         # noqa: F401  mpm_grid -> particle    (gather + advect; was g2p)
# active-matter <-> MPM two-way coupling (agents dragged/confined by + deforming the material):
from . import agent_scatter      # noqa: F401  agent set -> mpm_grid   (agents deform material; was agent_to_mpm)
from . import agent_gather       # noqa: F401  mpm_grid  -> agent set  (material drags + confines agents; was mpm_to_agent)
from . import agent_remodel         # noqa: F401  agent set -> mpm stiffness (cells soften/rigidify tissue)
from . import polarity_flow_align  # noqa: F401  mpm_grid -> agent heading (polarity-flow alignment; was flow_align)
from . import polarity_align       # noqa: F401  agent -> agent heading (1st-order Vicsek polar alignment; was heading_align)
from . import cell_divide           # noqa: F401  agent set structural: proliferation on a fixed buffer (occ)
from . import cell_grow             # noqa: F401  mpm_particle structural: tissue growth by material-point addition
from . import segmentation_seed     # noqa: F401  a measured instance segmentation -> the CELL level

__all__ = ["graph", "aggregate", "broadcast", "attraction_repulsion", "squared_law", "attractor_flow",
           "cohesion", "velocity_align", "separation", "velocity_cruise", "drag",
           "scalar_field", "deposit", "diffuse", "decay", "sense", "glide", "bounce",
           "prescribed_field", "chemotax", "gravity", "sediment",
           "pacemaker", "activation_pulse", "active_force", "active_stress",
           "mpm_spin", "mpm_anchor", "material_map", "mpm",
           "mpm_grid", "mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather",
           "agent_scatter", "agent_gather", "agent_remodel", "polarity_flow_align", "polarity_align"]
