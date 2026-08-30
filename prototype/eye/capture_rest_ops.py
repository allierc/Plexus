"""capture_rest_ops -- re-seed the muscle set's REST REFERENCE from a captured frame's
actual geometry, instead of the blend mesh (PROTOTYPE-LOCAL, not promoted).

    op: muscles_from_capture
    at: muscle_particle
    capture: archive/eye_H/duction_h_L_m20_final_curves.npz
    frame_idx: 167          # the captured index (not the raw sim frame) to re-seed from
    youngs: 240.0

Every muscle-stress panel in this project measures deformation and stress against `p.rest`,
which `blend_muscles` sets to the ARTIST'S drawn geometry at construction. That is a
reasonable default, but it is a CHOICE, not a law: whether a muscle at −20deg forced gaze
carries genuine tension or is sitting at its own natural length is an empirical question
(see run_zero_stress_return.py's own investigation), and answering it requires being able to
put the ZERO of the stress scale somewhere OTHER than the construction mesh.

This operator does exactly that: it loads a captured frame's actual particle positions (the
SAME simulation, mid-run, already correctly resolved) and registers THAT as `p.rest`, so `p.F`
starts at the identity there -- zero strain, zero stress, BY CONSTRUCTION, not by inspection --
then recomputes the fibre direction (the local tangent can rotate under load, even though the
fibre COORDINATE `s` -- a material label -- cannot) and each muscle's rest length the same way
`blend_muscles` does, off the new positions.

Meant to run immediately AFTER `blend_muscles` in the same `before_frame` seed slot: that
supplies `parent`, `p_vol`, `mass`, `mu`, `la` (intrinsic material properties, unchanged by
which configuration is called "rest") and this operator overrides only the geometric fields
(`pos`, `rest`, `fibre`, `anchored`, `tendon`, and the parent muscle's `rest_length`). `s`
itself is untouched -- it is a material coordinate (WHICH particle, not WHERE it is) and does
not change when the reference configuration does.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from plexus.models.base import Seed
from plexus.models.registry import register_operator

from blend_mpm_ops import fibre_from_s, binned_length


@register_operator("muscles_from_capture", family="anatomy", set="muscle_particle",
                   kind="seed")
class MusclesFromCapture(Seed):
    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["capture", "frame_idx"]
    MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["muscle_particle"]
    OUTPUTS = ["muscle_particle"]
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["morphogenesis_static", "reference_configuration_override"]
    PARAM_ROLES = {"capture": "source_curves_npz", "frame_idx": "captured_frame_index",
                   "cap": "attachment_cap_fraction", "k": "knn_for_fibre_regrad"}
    REFERENCE = ("This work; tests whether a forced-duction extremum is closer to a muscle's "
                "true unloaded length than the construction mesh is.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle_particle")
        self.capture = str(params["capture"])
        self.frame_idx = int(params["frame_idx"])
        self.cap = float(params.get("cap", 0.10))
        self.k = int(params.get("k", 12))
        # isolate ONE muscle: the other five keep their geometry (so `parent`/indexing stay
        # intact) but get mass=0, p_vol=0 -- a zero-mass MPM particle scatters nothing onto
        # the shared grid, so it cannot contact-couple with the target muscle or the globe
        # while still being present in the arrays. None = no isolation, all six stay live.
        self.target = params.get("target", None)
        self.target = int(self.target) if self.target is not None else None
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        p = H.level(self.at)
        dev = p.state.device
        d = np.load(self.capture)
        X = d["mus_pos"][self.frame_idx].astype(np.float64)
        par = d["mus_parent"]
        s = d["mus_s"].astype(np.float64)
        M = int(par.max()) + 1
        if X.shape[0] != p.n:
            raise ValueError(f"muscles_from_capture: capture has {X.shape[0]} muscle "
                             f"particles, spec has {p.n}")

        fib = np.zeros_like(X)
        rest_len = np.zeros(M)
        for mi in range(M):
            sel = par == mi
            pts = X[sel]
            k = min(self.k, len(pts) - 1)
            _, idx = cKDTree(pts).query(pts, k=k + 1)
            fib[sel] = fibre_from_s(pts, s[sel], idx)
            rest_len[mi], _ = binned_length(pts, s[sel])

        new = p.state.clone()
        pa, pb = p.state_schema["pos"]
        new[:, pa:pb] = torch.as_tensor(X, dtype=torch.float32, device=dev)
        p.state = new
        p.register_buffer("fibre", torch.as_tensor(fib, dtype=torch.float32, device=dev))
        p.register_buffer("rest", torch.as_tensor(X, dtype=torch.float32, device=dev))
        p.register_buffer("anchored", torch.as_tensor(s < self.cap, device=dev))
        p.register_buffer("tendon", torch.as_tensor(s > 1.0 - self.cap, device=dev))
        m = H.level(p.parent_name)
        m.register_buffer("rest_length",
                          torch.as_tensor(rest_len, dtype=torch.float32, device=dev))

        if self.target is not None:
            par_t = torch.as_tensor(par, device=dev)
            silent = par_t != self.target
            p.mass = p.mass.clone()
            p.p_vol = p.p_vol.clone()
            p.mass[silent] = 0.0
            p.p_vol[silent] = 0.0
            print(f"[muscles_from_capture] isolated muscle index {self.target}: "
                 f"{int(silent.sum())} of {p.n} particles zeroed (mass=0, p_vol=0), "
                 f"cannot scatter onto the shared grid", flush=True)

        print(f"[muscles_from_capture] re-seeded rest from {self.capture} frame_idx="
             f"{self.frame_idx}; new rest lengths = {np.round(rest_len, 4).tolist()}",
             flush=True)
        self._done = True
        return {}
