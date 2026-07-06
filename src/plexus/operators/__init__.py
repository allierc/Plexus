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
from . import squared_law                # noqa: F401  registers squared_law (lateral, 2nd-derivative, charged particles)
from . import cohesion              # noqa: F401  boids steering rule (lateral, 2nd-derivative)
from . import alignment             # noqa: F401  Vicsek velocity alignment (nominal); boids = special case
from . import separation            # noqa: F401  boids steering rule (lateral, 2nd-derivative)
from . import cruise               # noqa: F401  Vicsek self-propulsion: cruise to speed v0 (2nd-order; active matter, The Well)
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
from . import pulse_to_contraction  # noqa: F401  activation gradient -> per-particle force (exchange)
from . import pulse_to_active_stress  # noqa: F401  activation -> per-particle active stress -A nn^T (exchange)
from . import mpm_spin              # noqa: F401  drive MPM body toward slow solid-body rotation (lateral)
from . import mpm_anchor            # noqa: F401  substrate/boundary rest-anchor k*(rest-pos) (lateral)
from . import material_map          # noqa: F401  image field + apply_material_map (per-particle stiffness)
from . import mpm                   # noqa: F401  FENCED TRANSITIONAL oracle: MLS-MPM mechanics (mls_mpm_mechanics)
# Phase-3 decomposition of the oracle -- one file per operator + the shared grid field:
from . import mpm_grid              # noqa: F401  the mpm_grid background FIELD + B-spline kernel
from . import mpm_strain            # noqa: F401  particle -> particle  (F + material update)
from . import p2g                   # noqa: F401  particle -> mpm_grid   (scatter)
from . import mpm_grid_update       # noqa: F401  mpm_grid -> mpm_grid    (grid solve + BCs)
from . import g2p                   # noqa: F401  mpm_grid -> particle    (gather + advect)
# active-matter <-> MPM two-way coupling (agents dragged/confined by + deforming the material):
from . import agent_to_mpm          # noqa: F401  agent set -> mpm_grid   (agents deform material)
from . import mpm_to_agent          # noqa: F401  mpm_grid  -> agent set  (material drags + confines agents)
from . import agent_remodel         # noqa: F401  agent set -> mpm stiffness (cells soften/rigidify tissue)
from . import flow_align            # noqa: F401  mpm_grid -> agent heading (polarity-velocity/flow alignment)
from . import heading_align         # noqa: F401  agent -> agent heading (1st-order Vicsek polar alignment)
from . import cell_divide           # noqa: F401  agent set structural: proliferation on a fixed buffer (occ)
from . import cell_grow             # noqa: F401  mpm_particle structural: tissue growth by material-point addition

__all__ = ["graph", "aggregate", "broadcast", "attraction_repulsion", "squared_law",
           "cohesion", "alignment", "separation", "cruise", "drag",
           "scalar_field", "deposit", "diffuse", "decay", "sense", "glide", "bounce",
           "prescribed_field", "chemotax", "gravity", "sediment",
           "pacemaker", "activation_pulse", "pulse_to_contraction", "pulse_to_active_stress",
           "mpm_spin", "mpm_anchor", "material_map", "mpm",
           "mpm_grid", "mpm_strain", "p2g", "mpm_grid_update", "g2p",
           "agent_to_mpm", "mpm_to_agent", "agent_remodel", "flow_align", "heading_align"]
