"""t1_ops -- MOVED to `plexus.operators.vertex_ops`.

Kept as a re-export because thirty files import it by bare module name -- `run_one.py`,
`instrument.py`, `vtk_render.py`, `metrics.py` and twenty archive/analysis scripts -- and the
campaign is still running against them. PRIVATE NAMES ARE RE-EXPORTED TOO: `_carry_face_state`,
`_engine_owns_clock` and friends are called across module boundaries in okuda, so a shim that
exported only the public surface would break at the first T1.

New code should import from `plexus.operators.vertex_ops`.
"""
from plexus.operators.vertex_ops import *          # noqa: F401,F403
from plexus.operators.vertex_ops import (          # noqa: F401  the underscored names okuda reaches for
    _boundary_de,
    _face_ok_3d,
    _insert_after,
    _insert_before,
    _local_manifold_ok,
    _polygon_simple_2d,
    _ring_ok,
    _seg_cross,
    _vertex_faces)
