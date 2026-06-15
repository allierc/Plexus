"""Plexus models: the hierarchical container, registries, and catalog."""

from plexus.models import base, registry  # noqa: F401
# catalog is opt-in (stub menu); real op modules register into the same registry.

__all__ = ["base", "registry"]
