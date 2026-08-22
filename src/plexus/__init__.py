"""Plexus: hierarchical sets + operators for differentiable tissue."""

from plexus.models import base, registry  # noqa: F401
# NOTE: the scaffolding-era `models/catalog.py` stub menu is GONE -- the registry is the
# catalogue, and `plexus.operators` is the library it was a worklist for.
# Real operators (e.g. the prototype) register into the same registry without
# colliding with stubs.

__all__ = ["base", "registry"]
