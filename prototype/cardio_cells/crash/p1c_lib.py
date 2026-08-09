#!/usr/bin/env python
"""p1c_lib -- the shared harness for PROBE C, the adversarial attack on "E is unidentifiable".

Nothing here defines a measurement. It builds systems, rolls them out and hands the resulting
[G, M, 2] loop arrays to `accept.score_one`, which reads only the four certified instruments.
The one thing it adds over `boxprior.py` is that it can change the SPEC -- drag coefficient,
pacemaker period, active-force amplitude -- so the claim can be tested outside the single
configuration it was measured in.

The spec is changed by writing a patched copy of the yaml and pointing `assemble.CONFIG` at it.
No file under version control is edited, and `System.__init__` re-reads CONFIG on every build.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
for p in ("/workspace/Plexus/src", ALG, DISC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import assemble as ASM                                                    # noqa: E402
import crash_test as CT                                                   # noqa: E402
import accept as ACC                                                      # noqa: E402
import metrics as MET                                                     # noqa: E402

BASE_CONFIG = "/workspace/Plexus/config/material/material_cardio_cells.yaml"
INSTRUMENTS = tuple(ACC.CERTIFIED)


# --------------------------------------------------------------------------------------------- #
def default_args(**kw):
    a = argparse.Namespace(device="cuda:0", cells=100, per_parent=100, n_grid=128,
                           warmup=180, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def patched_config(drag_k=None, period=None, duration=None, amplitude=None, radius=None,
                   tag="p1c"):
    """Write a copy of the spec with the named knobs changed; return its path.

    Returns the BASE path unchanged when nothing is asked for, so the unmodified case is provably
    the same file the rest of the campaign used.
    """
    if all(v is None for v in (drag_k, period, duration, amplitude, radius)):
        return BASE_CONFIG
    raw = yaml.safe_load(open(BASE_CONFIG))
    for o in raw["operators"]:
        if o.get("op") == "drag" and drag_k is not None:
            o["k"] = float(drag_k)
        if o.get("op") == "pacemaker":
            if period is not None:
                o["period"] = float(period)
            if duration is not None:
                o["duration"] = float(duration)
        if o.get("op") == "active_force" and amplitude is not None:
            o["amplitude"] = float(amplitude)
        if o.get("op") == "activation_pulse" and radius is not None:
            o["radius"] = float(radius)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix=f"{tag}_cfg_")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(raw, f)
    return path


class Rig:
    """A planted, warmed system plus the reading surface and the reference rollout."""

    def __init__(self, args, drag_k=None, period=None, duration=None, amplitude=None,
                 radius=None, log=print, quiet=True):
        self.args = args
        self.spec = {"drag_k": drag_k, "period": period, "duration": duration,
                     "amplitude": amplitude, "radius": radius}
        old = ASM.CONFIG
        ASM.CONFIG = patched_config(drag_k, period, duration, amplitude, radius)
        try:
            self.sy, _ = CT.plant_and_warm(args, (lambda *_: None) if quiet else log)
        finally:
            ASM.CONFIG = old
        sy = self.sy
        self.t0, self.G = args.warmup, args.window
        self.tracers = {m: CT.tracer_indices(sy.x0, CT.probe_points(m)) for m in (10, 20)}
        self.C = sy.C
        self.E_true = sy.E_true[1:].clone()
        self.gain_true = sy.gain_true[1:].clone()
        self.theta_true = sy.theta_true.clone()

    # -- parameter vectors -------------------------------------------------------------------- #
    def theta(self, E=None, gain=None):
        sy = self.sy
        if E is None:
            e = self.E_true.clone()
        elif np.isscalar(E):
            e = torch.full((self.C,), float(E), device=sy.device, dtype=sy.dtype)
        else:
            e = torch.as_tensor(np.asarray(E, float), device=sy.device, dtype=sy.dtype)
        if gain is None:
            g = self.gain_true.clone()
        elif np.isscalar(gain):
            g = torch.full((self.C,), float(gain), device=sy.device, dtype=sy.dtype)
        else:
            g = torch.as_tensor(np.asarray(gain, float), device=sy.device, dtype=sy.dtype)
        return torch.cat([e, g])

    # -- where each cell IS, so a spatial pattern can be a spatial pattern ---------------------- #
    def centroids(self):
        """[C, 2] world centroid of each cell's particles at the snapshot tick."""
        if getattr(self, "_cen", None) is None:
            sy = self.sy
            cid = sy.cid.long()
            x = sy.x0
            s = torch.zeros(self.C + 1, 2, device=sy.device, dtype=sy.dtype)
            n = torch.zeros(self.C + 1, device=sy.device, dtype=sy.dtype)
            s.index_add_(0, cid, x)
            n.index_add_(0, cid, torch.ones_like(cid, dtype=sy.dtype))
            self._cen = (s[1:] / n[1:, None].clamp(min=1)).detach().cpu().numpy()
        return self._cen

    def checker(self, lo, hi, block=0.1, by="space"):
        """A two-valued per-cell field. `space`: a spatial checkerboard of side `block`.
        `index`: the same two values assigned by cell INDEX, which is spatially scrambled --
        the control that says whether an effect is the composition or the arrangement.
        """
        if by == "space":
            c = self.centroids()
            par = (np.floor(c[:, 0] / block).astype(int)
                   + np.floor(c[:, 1] / block).astype(int)) % 2
        elif by == "index":
            par = np.arange(self.C) % 2
        else:
            raise ValueError(by)
        return np.where(par == 0, lo, hi).astype(float)

    # -- one rollout, as loops ------------------------------------------------------------------ #
    def roll(self, theta, margin=20, G=None):
        tr, *_ = CT.rollout(self.sy, theta, self.t0, int(G or self.G), self.tracers)
        return tr[margin].detach().cpu().numpy()

    def free(self):
        del self.sy
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------- #
#  scoring -- every number below comes out of accept.score_one, i.e. out of cite()
# --------------------------------------------------------------------------------------------- #
_FLOORS = None


def floors():
    global _FLOORS
    if _FLOORS is None:
        _FLOORS = ACC.working_floors()
    return _FLOORS


def steps_row(sim, real):
    """The four certified instruments in steps, plus the objective (reported, never cited)."""
    r = ACC.score_one(sim, real, floors())
    out = {n: (None if r[n]["steps"] is None else float(r[n]["steps"])) for n in INSTRUMENTS}
    vals = {n: (None if r[n]["value"] is None else float(r[n]["value"])) for n in INSTRUMENTS}
    try:
        out["loopscore"] = float(MET.REGISTRY["loopscore"](sim, real))
    except Exception as e:
        out["loopscore"] = f"{type(e).__name__}"
    live = [v for k, v in out.items() if k in INSTRUMENTS and isinstance(v, float)]
    out["STAT"] = float(max(live)) if live else None
    out["_values"] = vals
    return out


def null_row():
    return {n: float(v) for n, v in ACC.null_steps(floors()).items()}


def amp_reading(loops):
    return float(MET.REGISTRY["peak_excursion"].reading(loops))


def path_reading(loops):
    return float(MET.REGISTRY["path_length"].reading(loops))
