"""The experiment rulers of exp06-15: one module per experiment, each registering its whole-run
measures ("run" kind) into `plexus.measures.MEASURES` on import. `tools/exp_gate_score.py` imports
this package and reads the measures by name, `expNN.<measure>`.

    from exp_measures import run_measure
    run_measure("exp11.bud", "tissue/exp11_hole_s1", axis=[0, 0, 1])
"""
from __future__ import annotations

import importlib
import pkgutil

from .common import open_run  # noqa: F401

# ONE BROKEN MODULE MUST NOT HIDE THE OTHERS: each experiment's rulers are imported on their own, and
# a failure is recorded (and warned) rather than raised, so exp06's syntax error cannot stop exp11's
# judge. `run_measure` on a measure whose module failed raises with that module's error.
IMPORT_ERRORS: dict[str, str] = {}
for _m in pkgutil.iter_modules(__path__):
    if _m.name.startswith("exp"):
        try:
            importlib.import_module(f"{__name__}.{_m.name}")
        except Exception as _e:                                              # noqa: BLE001
            IMPORT_ERRORS[_m.name] = f"{type(_e).__name__}: {_e}"
            import warnings
            warnings.warn(f"exp_measures.{_m.name} failed to import: {IMPORT_ERRORS[_m.name]}")


def run_measure(name: str, spec_or_traj, **kw) -> dict:
    """Evaluate a registered run measure on a run (`group/name`, a directory, or an open Traj)."""
    from plexus.measures import MEASURES
    if name not in MEASURES:
        mod = name.split(".")[0]
        raise KeyError(f"no run measure {name!r}" + (f" -- exp_measures.{mod} failed to import: {IMPORT_ERRORS[mod]}"
                                                    if mod in IMPORT_ERRORS else ""))
    m = MEASURES[name]
    if m.kind != "run":
        raise ValueError(f"{name} is a {m.kind} measure, not a run measure")
    T = open_run(spec_or_traj) if isinstance(spec_or_traj, str) else spec_or_traj
    return m.fn(T, **kw)
