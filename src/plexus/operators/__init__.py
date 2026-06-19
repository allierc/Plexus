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
from . import video_field           # noqa: F401  registers the `video` field + playback
from . import chemotaxis            # noqa: F401  field -> set gradient coupling

__all__ = ["graph", "aggregate", "broadcast", "attraction_repulsion",
           "cohesion", "alignment", "separation", "drag",
           "scalar_field", "deposit", "diffuse", "decay", "sense", "advance", "bounce",
           "video_field", "chemotaxis"]
