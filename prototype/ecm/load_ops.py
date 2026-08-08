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
        # NORMALISED BY THE p99 OF THE NONZERO MAP -- same reason and same measurement as
        # `ecm_growth_gate_3d` below: the peak is a single bin in a single frame, an order of magnitude
        # above anything typical, so normalising by it silently scales the coupling to nothing.
        nz = P[P > 0]
        self.pk = max(float(_np.percentile(nz, 99)) if nz.size else 1.0, 1e-12)
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


@register_operator("ecm_growth_gate_3d", family="growth", set="vertex", kind="structural")
class ECMGrowthGate3D(Structural):
    """The matrix's stress slows the CELL CYCLE where it presses hardest.

    THE MECHANISM, AND WHY IT IS STRONGER THAN A FORCE. `ecm_load_3d` pushes the vertices inward: a
    mechanical correction that fights the growth every frame and is bounded by how hard you dare push
    before cells invert. This operator instead gates the RATE -- a cell facing a stressed matrix grows
    its target volume more slowly, so it reaches `divide_3d`'s volume-doubling threshold later and
    DIVIDES LESS OFTEN. That difference integrates over 400 frames, so a few percent of stress
    anisotropy becomes a visible shape anisotropy, which a force of the same size cannot do.

    It is also the biology: proliferation under mechanical load is suppressed, not just deformed
    (Helmlinger 1997; Montel 2011 measured spheroids stalling under external pressure). The tissue is
    not being pushed into an ovoid -- it is GROWING into one, because the directions differ in how much
    the matrix objects.

    HOW IT INTERCEPTS WITHOUT REWRITING THE GROWTH OPERATOR. `grow_3d` keeps a per-cell
    cumulative scale `mg_scale` and multiplies it by (1 + rate.(rho + Hill(a))) each tick, then derives
    A0/P0/V0f/R0 from it. This operator runs AFTER it, remembers the scale it left behind last frame,
    and reads the factor growth just applied as `f = s_now / s_prev`. The gated scale is then
    `s_prev * (1 + gate.(f - 1))` -- exact, and it needs to know nothing about `rate`, `rho` or the Hill
    function, so it cannot drift out of step with them. Five lines of A0/P0/V0f/R0 bookkeeping are
    duplicated from that operator, which is the price of not editing a shared file; they are marked.

    THE GATE IS A CHOICE OF FUNCTIONAL FORM, and it is stated rather than buried: a Hill in the
    normalised pressure, `gate = floor + (1 - floor) / (1 + (P/p_half)^n)`. `floor` matters -- with
    floor 0 a cell in the most stressed direction stops dividing entirely and the tissue can only grow
    the other way, which produces a dramatic shape for a reason that is closer to a wall than to
    mechanosensing.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["load"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["mechanosensitive_growth", "contact_inhibition",
                      "matrix_to_cell_feedback", "anisotropic_growth"]
    PARAM_ROLES = {"p_half": "half_suppression_pressure", "hill": "gate_sharpness",
                   "floor": "minimum_growth_fraction"}
    REFERENCE = ("Helmlinger, G. et al. (1997) Nat. Biotechnol. 15:778; "
                 "Montel, F. et al. (2011) Phys. Rev. Lett. 107:188102.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as np
        self.at = params.get("_at", "vertex")
        z = np.load(str(params["load"]))
        P = np.asarray(z["pmap"], np.float32)

        # SMOOTHED, AND THIS WAS THE LAST THING IN THE WAY. The recorded map is a CONTACT LEDGER, not a
        # stress field: at frame 250 only 24% of its bins are nonzero, because 140,000 particles binned
        # into 2,048 directions against a moving surface touch a different quarter of it every frame. A
        # zero bin gates to 1.0 -- no suppression -- so three quarters of the polar cells were
        # unsuppressed in any given frame and a different three quarters the next, which time-averages to
        # a nearly UNIFORM weak suppression. That is why a 15x instantaneous directional contrast moved
        # the aspect ratio by 3%: the contrast was real and it was not persistent for any given cell.
        #
        # The patchiness is a sampling artefact of the estimator, not a feature of the field, so it is
        # averaged out: a running mean over `smooth_frames` and a box over `smooth_phi_deg` of longitude
        # and one row of colatitude. `smooth_phi_deg = 360` makes the map AXISYMMETRIC, which is the
        # right estimator when the matrix is (dense polar caps are), and is wrong the moment the
        # experiment puts structure in longitude -- so it is a parameter and not a default of the code.
        sf = int(params.get("smooth_frames", 25))
        sp = float(params.get("smooth_phi_deg", 360.0))
        if sf > 1 and P.shape[0] > sf:
            k = np.ones(sf, np.float32) / sf
            flat = P.reshape(P.shape[0], -1)
            P = np.stack([np.convolve(flat[:, j], k, mode="same") for j in range(flat.shape[1])],
                         axis=1).reshape(P.shape).astype(np.float32)
        if sp >= 359.0:
            P = np.repeat(P.mean(axis=2, keepdims=True), P.shape[2], axis=2)
        elif sp > 0:
            w = max(1, int(round(sp / 360.0 * P.shape[2])))
            ker = np.ones(w, np.float32) / w
            P = np.stack([[np.convolve(np.tile(P[t, i], 3), ker, mode="same")[P.shape[2]:2 * P.shape[2]]
                           for i in range(P.shape[1])] for t in range(P.shape[0])]).astype(np.float32)
        # one row of colatitude, clamped at the poles
        P = (P + np.pad(P, ((0, 0), (1, 0), (0, 0)), mode="edge")[:, :-1]
             + np.pad(P, ((0, 0), (0, 1), (0, 0)), mode="edge")[:, 1:]) / 3.0
        # NORMALISED so `p_half` means the same thing whatever the matrix's stiffness was -- otherwise
        # stiffening the matrix moves both the pressure and the gate's operating point and a stiffness
        # sweep measures two things at once.
        #
        # BY THE p99 OF THE NONZERO MAP, NOT BY THE PEAK, and that was measured the hard way. The peak
        # is ONE bin in ONE frame: on 49_aniso_i0_fibres it is 4.0e5 while the per-axis mean pressures
        # are 1448-2176, so peak-normalisation put the whole run at press ~0.005 and `p_half = 0.25` left
        # the gate at 1.000 -- an operator that ran 400 times and did nothing. Against the p99 (4.6e4)
        # the same run spans mean 0.007 -> 0.159 with the loaded directions reaching p90 0.45, which is a
        # range a Hill function can actually act on. Only bins that ever saw contact count: three
        # quarters of the map is identically zero, and including it would drag the scale toward the
        # tissue's own solid angle rather than the pressures it produced.
        nz = P[P > 0]
        self.pk = max(float(np.percentile(nz, 99)) if nz.size else 1.0, 1e-12)
        self.P = torch.as_tensor(P / self.pk, dtype=torch.float32)
        self.T = int(self.P.shape[0])
        # SELF-CALIBRATING OPERATING POINT. `p_half` in units of the p99 was a hand-picked number and it
        # put the gate on the wrong part of its own curve: a real map's typical pressures sit far below
        # its p99, which is set by a few hot contact bins. With dense polar caps giving poles 7234 and
        # equator 2631 -- a genuine 2.75x pattern -- both landed at 0.145 and 0.053 against p_half 0.10,
        # i.e. on the flat foot of the Hill, for a rate difference of only 1.7x where the clean synthetic
        # map gave 5x. `auto` puts the half-suppression point at the MEDIAN of the pressure the surface
        # actually carries late in the run, so half the loaded tissue sits either side of it and the
        # pattern falls across the steep part of the curve instead of under it.
        # `relative` IS A DIFFERENT MECHANISM AND IS LABELLED ONE. The absolute gate is the physical
        # reading -- a cell responds to the stress it carries -- and it has a timing problem that is the
        # MATRIX's, not the gate's: the measured pressure only passes the half-point around frame 350 of
        # 402, so a 15x directional rate difference arrives with 50 frames left to shape anything. In
        # `relative` mode the reference is the current frame's own mean over loaded directions, so the
        # gate reads the PATTERN and ignores the amplitude, and it acts from first contact. That is
        # adaptive mechanosensing (cells habituating to ambient stress and responding to the excess),
        # which is a real mechanism and a WEAKER claim than the absolute one -- it cannot be reported as
        # "the matrix's stress shaped the tissue" without saying which mode produced it.
        self.relative = str(params.get("p_half", "")).lower() == "relative"
        ph = params.get("p_half", "auto")
        if self.relative:
            self.p_half = float(params.get("rel_half", 1.0))
        elif isinstance(ph, str) and ph.lower() == "auto":
            late = P[int(0.75 * P.shape[0]):]
            lnz = late[late > 0]
            self.p_half = float(np.median(lnz) / self.pk) if lnz.size else 0.10
        else:
            self.p_half = float(ph)
        # SHARPER THAN 2, AND THAT IS A CLAIM ABOUT MECHANOSENSING, not a fitting knob. A linear-ish
        # gate cannot turn a 2.75x stress pattern into a shape: it needs to be switch-like, which is what
        # a Hill exponent of 4-6 is and what mechanotransduction actually looks like (a threshold, not a
        # proportionality). Stated here because the alternative reading -- "the exponent was raised until
        # the picture worked" -- is the one a reader should be able to rule out from the numbers above.
        self.hill = float(params.get("hill", 4.0))
        self.floor = float(params.get("floor", 0.15))
        self._prev = None
        self._frame, self._t, self._said = -1, 0, False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        if m is None or "mg_scale" not in m:
            return {}                      # growth has not run yet this run; nothing to gate
        nF = int(m["nF"])
        s_now = m["mg_scale"]
        dev, dt_ = s_now.device, s_now.dtype
        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
            self._t = min(self.T - 1, max(0, f))

        # PER-CELL DIRECTION -> PRESSURE BIN. Centroid-referenced, matching how the map was built.
        pos = lvl.get("pos")
        es, ef = m["E_srce"], m["E_face"]
        live = ef < nF
        e_s, e_f = es[live].long(), ef[live].long()
        cnt = torch.zeros(nF, device=dev, dtype=dt_).index_add_(
            0, e_f, torch.ones_like(e_f, dtype=dt_))
        cen = torch.zeros(nF, 3, device=dev, dtype=dt_).index_add_(0, e_f, pos[e_s].to(dt_))
        ok = cnt > 0
        cen[ok] /= cnt[ok, None]
        origin = cen[ok].mean(0) if ok.any() else torch.zeros(3, device=dev, dtype=dt_)
        d = cen - origin
        r = d.norm(dim=1).clamp_min(1e-9)
        u = d / r[:, None]
        M = self.P[self._t].to(dev, dt_)
        nth, nph = M.shape
        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        press = M[(th / math.pi * nth).long().clamp(0, nth - 1),
                  (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
        press = torch.where(ok, press, torch.zeros_like(press))
        ref = self.p_half
        if self.relative:
            # THE CURRENT FRAME'S OWN MEAN OVER LOADED DIRECTIONS. Zero bins are excluded: three
            # quarters of the map is untouched surface early on, and averaging that in would put the
            # reference far below anything real and saturate the gate everywhere.
            live = M[M > 0]
            if live.numel() < 8:
                return {}                      # nothing in contact yet: nothing to be relative to
            ref = float(live.mean()) * self.p_half
        gate = self.floor + (1.0 - self.floor) / (
            1.0 + (press / max(ref, 1e-12)).clamp_min(0.0) ** self.hill)

        # ---- GATE THE TARGET VOLUME, NOT `mg_scale` -------------------------------------------------
        # THE TRAP THIS AVOIDS, MEASURED. `grow_3d` reallocates `mg_scale` to ONES and
        # re-bases its A0_init/P0_init/V0f_init snapshots from the current values whenever `nF` changes
        # -- and with `divide_3d` firing every 4 frames, nF changes on most frames. A gate that reads
        # `s_now / s_prev` therefore saw a size change almost every frame, skipped its correction, and
        # ran 400 times without altering anything: 5,933 cells against 5,968 ungated and the same final
        # radius to two decimals. `V0f` is CONTINUOUS across that re-base (only its reference moves), so
        # gating its increment works where gating the scale cannot.
        V = m["V0f"]
        prev = self._prev
        if prev is None or prev.shape[0] != nF:
            p2 = V.detach().clone()
            if prev is not None:
                k = min(prev.shape[0], nF)
                p2[:k] = prev[:k]
            prev = p2
        dV = V - prev
        # A DIVISION IS NOT GROWTH. A daughter's target volume is set by `divide_3d`, not by an
        # increment, so its `dV` is a large jump in either direction; gating it would shrink or inflate a
        # cell that had just been created. Growth is ~0.3% per frame, so anything past 20% is a topology
        # event and is passed through untouched -- and re-based, so the next frame gates normally.
        topo = dV.abs() > 0.2 * prev.clamp_min(1e-12)
        Vg = torch.where(topo, V, prev + gate * dV)
        q = (Vg / V.clamp_min(1e-12)).clamp(0.2, 1.0)      # how much of the wanted growth was allowed
        m["V0f"] = V * q
        m["V0"] = float(m["V0f"].sum())
        # A0 ~ R^2 and P0 ~ R with R ~ V^(1/3), so one volume ratio sets all three consistently. Applied
        # to the CURRENT values rather than to the *_init snapshots, which is what makes this survive the
        # re-base above.
        m["A0"] = m["A0"] * q.pow(2.0 / 3.0)
        m["P0"] = m["P0"] * q.pow(1.0 / 3.0)
        m["mg_scale"] = s_now * q.pow(1.0 / 3.0)
        if "R0" in m:
            m["R0"] = float((3.0 * max(m["V0"], 1e-12) / (4.0 * math.pi)) ** (1.0 / 3.0))
        self._prev = m["V0f"].detach().clone()
        LOAD_TRACE.append((int((gate < 0.99).sum()), float(gate.min()), float(press.mean())))
        if not self._said:
            print(f"[ecm_growth_gate_3d] recorded matrix load, {self.T} frames, p99 {self.pk:.4g}; "
                  f"p_half={self.p_half} hill={self.hill} floor={self.floor}; gate range "
                  f"[{float(gate.min()):.3f}, {float(gate.max()):.3f}] over {nF} cells at frame {f}",
                  flush=True)
            self._said = True
        return {}

