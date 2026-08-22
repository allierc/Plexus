"""attractor_flow -- MOVED to `plexus.operators.motion_ops`.

Kept as a re-export because thirty files import it by bare module name -- `run_one.py`,
`instrument.py`, `vtk_render.py`, `metrics.py` and twenty archive/analysis scripts -- and the
campaign is still running against them. PRIVATE NAMES ARE RE-EXPORTED TOO: `_carry_face_state`,
`_engine_owns_clock` and friends are called across module boundaries in okuda, so a shim that
exported only the public surface would break at the first T1.

New code should import from `plexus.operators.motion_ops`.
"""
from plexus.operators.motion_ops import *          # noqa: F401,F403
from plexus.operators.motion_ops import (          # noqa: F401  the underscored names okuda reaches for
    _FIELDS,
    _aizawa,
    _chen,
    _chua,
    _dadras,
    _halvorsen,
    _lorenz,
    _rabinovich_fabrikant,
    _rossler,
    _sprott_b,
    _thomas)
