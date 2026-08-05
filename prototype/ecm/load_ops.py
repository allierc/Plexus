"""load_ops -- the other half of the coupling: the matrix pushing back on the cells.

WHAT WAS MISSING. `cell_to_ecm` computes the force the growing epithelium puts on the matrix. That
force has an equal and opposite partner on the epithelium, and a REPLAY had nowhere to put it -- pass 1
finished before pass 2 began -- so every run so far showed a tissue that loaded a matrix and was not
touched by it. The plates (`plate_confine_3d`) get the ovoid by fiat: they are rigid, so the shape is
the tissue's mechanics answering a boundary condition that was decided before the run. THIS operator
is the version where the material does it: the fibres resist, and the resistance is what shapes the
tissue.

HOW IT CLOSES THE LOOP -- A STAGGERED SCHEME, NOT A SINGLE-WORLD SOLVE. `mpm_grid` is hard-coded to the
unit box and the vertex model cannot be rescaled into it (`combine.py` has the measurements), so the
two solvers cannot share a timestep. Instead they alternate, which is a standard partitioned coupling:

    iteration 0    tissue alone            -> surface S0
                   matrix loaded by S0     -> pressure map P0(theta, phi, t)
    iteration 1    tissue + load P0        -> surface S1        <- THIS OPERATOR
                   matrix loaded by S1     -> pressure map P1
    ...            until the surface stops moving

Each half-step is a real simulation of a real operator stack. What the scheme gives up is
simultaneity: within one iteration the tissue feels the PREVIOUS iteration's matrix, so a converged
fixed point is a true two-way solution and an unconverged one is not. `feedback.py` measures the
iteration-to-iteration change and reports it, because "it converged" is the only thing that makes the
result mean what it looks like.

THE GAIN IS A CALIBRATION CONSTANT AND IT IS NOT DERIVED. This is the honest limit. The pressure comes
out of the matrix in MPM units over a unit box; the vertex model works in AVM energy units over a
50-unit world. Converting one into the other rigorously IS the dimensional calibration that the two-pass
structure exists to avoid, so `gain` sets how hard the matrix pushes and its value is chosen rather than
computed. What is NOT invented is the SHAPE and the TIMING of the load: where on the surface it acts and
how it grows are measured from the matrix, not assumed. So a gain sweep answers "what stiffness of
matrix would flatten the tissue this much", and does not answer "what does E = 15 kPa do". Reporting the
sweep instead of a single number is the difference between those two claims.

AN OVERDAMPED DISPLACEMENT, WHICH IS WHAT THE SOLVER ITSELF DOES. `shape_energy_3d` owns the vertex
force loop and there is no term to add a load to from outside it, so the load is applied as a
displacement after the relaxation: dx = -(gain * P / mu) * dt * n. That is not a shortcut for a force,
it IS the force under the overdamped integration `shape_energy_3d` already uses (its own `mu` and `dt`),
applied at the same point in the frame. Capped per frame by `cap_frac` of the local radius for the same
reason the shape solver caps its own step: one bad frame should not be able to invert a cell.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Structural
from plexus.models.registry import register_operator

# Per frame: (vertices loaded, max inward displacement, mean pressure applied). The load is invisible
# in a still -- it acts inward on a surface that is also growing outward -- so without this there is no
# way to tell "the matrix pushed and the tissue resisted" from "the operator did nothing".
LOAD_TRACE: list = []


@register_operator("ecm_load_3d", family="mechanics", set="vertex", kind="structural")
class ECMLoad3D(Structural):
    """Push the vertex mesh inward with a recorded matrix pressure map P(theta, phi, t)."""

    EMIT = None                        # moves positions in place; no integrable delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["load"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["matrix_to_cell_feedback", "mechanical_resistance", "partitioned_coupling"]
    PARAM_ROLES = {"gain": "load_coupling_gain", "mu": "vertex_mobility",
                   "dt": "frame_timestep", "cap_frac": "max_step_as_radius_fraction"}
    REFERENCE = "Plexus (this work); the reaction to Okuda, S. et al. (2018) Sci. Rep. 8:2386 contact."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as _np
        self.at = params.get("_at", "vertex")
        z = _np.load(str(params["load"]))
        P = _np.asarray(z["pmap"], _np.float32)
        # NORMALISED BY ITS OWN PEAK, so `gain` means the same thing across matrices of different
        # stiffness. Without this, raising E in pass 2 would raise the pressure AND therefore the
        # effective gain, and a gain sweep would be measuring two things at once.
        self.pk = float(max(P.max(), 1e-12))
        self.P = torch.as_tensor(P / self.pk, dtype=torch.float32)
        self.T = int(self.P.shape[0])
        self.gain = float(params.get("gain", 1.0))
        self.mu = float(params.get("mu", 1.0))
        self.dt = float(params.get("dt", 1.0))
        self.cap_frac = float(params.get("cap_frac", 0.04))
        self._frame, self._t, self._said = -1, 0, False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        m = getattr(lvl, "_mesh", None)
        n = int(m["Nv"]) if (m is not None and "Nv" in m) else pos.shape[0]
        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
            self._t = min(self.T - 1, max(0, f))
        M = self.P[self._t].to(dev, dt_)
        nth, nph = M.shape

        # CENTROID-REFERENCED, because the recorded map is: `tissue.apical_map` subtracts the vertex
        # centroid before binning, and the vesicle drifts. Binning against the world origin instead
        # would rotate the load off the surface it was measured on, a little more every frame.
        p = pos[:n]
        c = p.mean(0)
        d = p - c
        r = d.norm(dim=1).clamp_min(1e-9)
        u = d / r[:, None]
        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        it = (th / math.pi * nth).long().clamp(0, nth - 1)
        ip = (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)
        press = M[it, ip]

        step = (self.gain * press / max(self.mu, 1e-12)) * self.dt
        step = torch.minimum(step, self.cap_frac * r)          # never more than a slice of the radius
        pos[:n] = p - step[:, None] * u                        # inward, along the surface normal
        nz = int((step > 0).sum())
        LOAD_TRACE.append((nz, float(step.max()) if n else 0.0, float(press.mean())))
        if not self._said:
            print(f"[ecm_load_3d] {self.at}: recorded matrix load, {self.T} frames, peak pressure "
                  f"{self.pk:.4g} (normalised to 1); gain={self.gain}, cap={self.cap_frac} of r; "
                  f"{nz} of {n} vertices loaded at frame 0", flush=True)
            self._said = True
        return {}
