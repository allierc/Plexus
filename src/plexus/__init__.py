"""Plexus: hierarchical sets + operators for differentiable tissue."""

from plexus.models import base, registry  # noqa: F401
# NOTE: catalog (the stub menu) is opt-in: `import plexus.models.catalog`.
# Real operators (e.g. the prototype) register into the same registry without
# colliding with stubs.

__all__ = ["base", "registry"]
