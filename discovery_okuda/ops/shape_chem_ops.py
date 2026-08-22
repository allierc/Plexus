"""shape_chem_ops -- MOVED to `plexus.operators.diffusion_reaction`.

Kept as a re-export because thirty files import it by bare module name -- `run_one.py`,
`instrument.py`, `vtk_render.py`, `metrics.py` and twenty archive/analysis scripts -- and the
campaign is still running against them. PRIVATE NAMES ARE RE-EXPORTED TOO: `_carry_face_state`,
`_engine_owns_clock` and friends are called across module boundaries in okuda, so a shim that
exported only the public surface would break at the first T1.

New code should import from `plexus.operators.diffusion_reaction`.
"""
from plexus.operators.diffusion_reaction import *          # noqa: F401,F403
from plexus.operators.diffusion_reaction import (          # noqa: F401  the underscored names okuda reaches for
    _ShapeToChemBase,
    _cell_adjacency,
    _np,
    _standardise)
