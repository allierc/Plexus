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
from . import Coulomb                # noqa: F401  registers Coulomb (lateral, 2nd-derivative, charged particles)
from . import cohesion              # noqa: F401  boids steering rule (lateral, 2nd-derivative)
from . import alignment             # noqa: F401  boids steering rule (lateral, 2nd-derivative)
from . import separation            # noqa: F401  boids steering rule (lateral, 2nd-derivative)
from . import drag                  # noqa: F401  registers drag (lateral, 2nd-derivative)
# field-coupled primitives (the slime/Physarum decomposition: 1 set + 1 scalar field)
from . import scalar_field          # noqa: F401  registers the `grid` scalar field
from . import deposit               # noqa: F401  set -> field
from . import diffuse               # noqa: F401  field -> field
from . import decay                 # noqa: F401  field -> field
from . import sense                 # noqa: F401  field -> set (turn heading)
from . import advance               # noqa: F401  set -> self-propelled move
from . import bounce                # noqa: F401  set -> wall/obstacle reflection (re-head)
from . import sense_3d              # noqa: F401  field -> set, 3D cone sensing (vector heading)
from . import advance_3d            # noqa: F401  set -> self-propelled move along a 3D heading
from . import bounce_3d             # noqa: F401  set -> 3D wall reflection (re-head)
from . import video_field           # noqa: F401  registers the `video` field + playback
from . import chemotaxis            # noqa: F401  field -> set gradient coupling
from . import gravity               # noqa: F401  cell-level body force (feeds the MPM substep)
# active-stimulus decomposition (clock -> activation field -> contraction -> MPM):
from . import pacemaker             # noqa: F401  periodic scalar clock p(t) -> H.signals (field)
from . import pulse_stimulus        # noqa: F401  clock x Gaussian -> activation field (field)
from . import pulse_to_contraction  # noqa: F401  activation gradient -> per-particle force (exchange)
from . import mpm_drag              # noqa: F401  viscous body drag -k*v as a particle force (lateral)
from . import material_map          # noqa: F401  image field + apply_material_map (per-particle stiffness)
from . import mpm                   # noqa: F401  FENCED TRANSITIONAL oracle: MLS-MPM mechanics (mls_mpm_mechanics)
# Phase-3 decomposition of the oracle -- one file per operator + the shared grid field:
from . import mpm_grid              # noqa: F401  the mpm_grid background FIELD + B-spline kernel
from . import mpm_strain            # noqa: F401  particle -> particle  (F + material update)
from . import p2g                   # noqa: F401  particle -> mpm_grid   (scatter)
from . import mpm_grid_update       # noqa: F401  mpm_grid -> mpm_grid    (grid solve + BCs)
from . import g2p                   # noqa: F401  mpm_grid -> particle    (gather + advect)

__all__ = ["graph", "aggregate", "broadcast", "attraction_repulsion", "Coulomb",
           "cohesion", "alignment", "separation", "drag",
           "scalar_field", "deposit", "diffuse", "decay", "sense", "advance", "bounce",
           "sense_3d", "advance_3d", "bounce_3d",
           "video_field", "chemotaxis", "gravity",
           "pacemaker", "pulse_stimulus", "pulse_to_contraction", "mpm_drag", "material_map", "mpm",
           "mpm_grid", "mpm_strain", "p2g", "mpm_grid_update", "g2p"]
