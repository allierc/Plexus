"""Operator library. Importing this package self-registers every operator into
the registry (each module calls `@register_operator`). The engine imports this so
a spec's operator names resolve; `registry.catalog_summary()` then lists them.

Start small: attraction / repulsion only. Port more (mpm, secrete/sense, the
grow/divide line) into new one-concern modules here as we scale up.
"""
from __future__ import annotations

from . import interaction  # noqa: F401  registers attract / repulse

__all__ = ["interaction"]
