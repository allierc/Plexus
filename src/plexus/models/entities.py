"""Entity semantics: the per-node STATE SCHEMA and RENDER hints, in the registry.

This is the single source of truth for "what a node's state columns mean". The
engine reads `state_schema` to size and slice a set's state; operators read named
blocks (`lvl.get('pos')`, `lvl.get('vel')`) instead of hardcoding `[:, :2]`; the
plotter reads `render` (color-by, arrows) so it draws generically. A new entity
declares its layout here once and everyone downstream just works.

  state_schema : {block_name: (start_col, end_col)}  -- contiguous, defines the dim
  render       : {color_by: <per-node int field>, arrows: <vector block or None>}

Importing this module registers the entities. The engine imports it alongside the
operator library.
"""
from __future__ import annotations

from plexus.models.registry import register_entity


@register_entity(
    "particle", level=0,
    state_schema={"pos": (0, 2), "vel": (2, 4)},
    render={"color_by": "node_type", "arrows": "vel"},
)
class Particle:
    """A point with position + velocity (the interacting-particle / boid leaf)."""


@register_entity(
    "cell", level=1,
    state_schema={"pos": (0, 2), "vel": (2, 4)},
    render={"color_by": "node_type", "arrows": "vel"},
)
class Cell:
    """A set of particles/molecules; its position is an aggregate of its children."""


# default for any set whose name is not a registered entity
DEFAULT_STATE_SCHEMA = {"pos": (0, 2), "vel": (2, 4)}
DEFAULT_RENDER = {"color_by": "node_type", "arrows": None}
