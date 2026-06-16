"""Operator library. Importing this package self-registers every operator into
the registry (each module calls `@register_operator`). The engine imports this so
a spec's operator names resolve; `registry.catalog_summary()` then lists them.

Start small: the analytic attraction_repulsion law only. Port more (mpm,
secrete/sense, the grow/divide line) into new one-concern modules here as we
scale up.
"""
from __future__ import annotations

from . import graph                 # noqa: F401  registers radius_graph (rewire)
from . import attraction_repulsion  # noqa: F401  registers attraction_repulsion (lateral, 1st-derivative)
from . import boids                 # noqa: F401  registers boids (lateral, 2nd-derivative)
from . import drag                  # noqa: F401  registers drag (lateral, 2nd-derivative)

__all__ = ["graph", "attraction_repulsion", "boids", "drag"]
