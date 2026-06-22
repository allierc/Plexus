#!/usr/bin/env python
"""Data-facing cardio prototype: a Plexus spec that reproduces the real 4-cycle
trajectory data, NOT a mechanism demo.

100x100 fixed tissue-material grid. No cell segmentation -- the nodes are material
sites. Each site carries position, velocity, (u,w), a smooth material STIFFNESS and a
smooth ACTIVE-CONTRACTION GAIN. Symmetry is broken by a deterministic, smooth
heterogeneity field (wavelength ~10 nodes), never a random texture.

Closed forward loop on one `tissue_particle` set:

  trigger_pulse -> excitable_nagumo -> signal_to_mpm_force -> mls_mpm_mechanics
   (4 paced S1)      (u,w wave)          (u -> active stress)   (inertial elastic deformation)

Optimised for the observed pattern: four repeated cycles with similar curved-string
node trajectories -- not planar conduction, not a generic rotor (those are debug
tests in cardio_stage1.py).

Outputs, archived into archive/<name>/:
  <name>_activity.mp4      multicolor activity/mechanics movie (u on moving nodes; no microscope bg)
  <name>_trajectories.mp4  amplified node displacement paths     (cf. cog_trajectories.mp4; no bg)
  <name>_panels.png        multipanel: phases (rest->activation->curve->recovery) + heterogeneity field
  spec.yaml                the exact, reproducible spec

Run:
  cd /workspace/Plexus
  PYTHONPATH=src .../python prototype/cardio/cardio_stage2.py prototype/cardio/specs/coupled4.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from plexus.models.base import Hierarchy, Level, Lateral   # noqa: E402
from cardio_stage1 import GridGraph, TriggerPulse, ExcitableNagumo, FFMPEG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ_REAL = os.path.join(HERE, "cardio_real.npz")
# state layout: pos, vel, u, w
SCHEMA = {"pos": (0, 2), "vel": (2, 4), "u": (4, 5), "w": (5, 6)}


def real_traj(n_out, shape, path=NPZ_REAL):
    """Real cardio node trajectory (137^2, normalized) resampled to n_out frames by
    linear time-stretch. Node ordering matches the sim grid (index = y*137 + x)."""
    P = np.load(path)["pos"]                                  # [T, N, 2]
    if P.shape[1] != shape[0] * shape[1]:
        raise ValueError(f"real data has {P.shape[1]} nodes, sim grid is {shape} "
                         f"({shape[0]*shape[1]}); boundary anchoring needs them equal")
    T = P.shape[0]
    idx = np.linspace(0, T - 1, n_out)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, T - 1)
    w = (idx - lo)[:, None, None]
    return ((1 - w) * P[lo] + w * P[hi]).astype(np.float32)   # [n_out, N, 2]


# --------------------------------------------------------------------------- #
#  Deterministic smooth heterogeneity (wavelength ~ lam nodes; no RNG)
# --------------------------------------------------------------------------- #
def aniso_field(Hy, Wx, wl, angle=0.0, phase=0.0):
    """Smooth scalar in [0,1]. wl scalar or [wx,wy] (anisotropic); pattern rotated by
    `angle` (rad). Deterministic -- the symmetry-breaking knob lives entirely here."""
    yy, xx = np.meshgrid(np.arange(Hy), np.arange(Wx), indexing="ij")
    ca, sa = np.cos(angle), np.sin(angle)
    xr, yr = ca * xx + sa * yy, -sa * xx + ca * yy
    wx, wy = (wl if isinstance(wl, (list, tuple)) else (wl, wl))
    f = (np.cos(2 * np.pi * xr / wx + phase) * np.cos(2 * np.pi * yr / wy + 0.5 * phase)
         + 0.5 * np.cos(2 * np.pi * (xr / wx + yr / wy) + phase))
    return ((f - f.min()) / (f.max() - f.min() + 1e-9)).astype(np.float32)


def fiber_field(Hy, Wx, cfg):
    """Per-node fibre angle (rad). mode: 'smooth' (band-limited), 'swirl' (tangential
    around a centre), or 'radial' -- each breaks the loop-orientation symmetry differently."""
    mode = cfg.get("mode", "smooth")
    if mode in ("swirl", "radial"):
        yy, xx = np.meshgrid(np.linspace(0, 1, Hy), np.linspace(0, 1, Wx), indexing="ij")
        cx, cy = cfg.get("center", [0.5, 0.5])
        ang = np.arctan2(yy - cy, xx - cx)
        if mode == "swirl":
            ang = ang + cfg.get("offset", np.pi / 2)        # tangential
        return ang.astype(np.float32)
    base = cfg.get("base", 0.0); amp = cfg.get("amp", np.pi)
    return (base + amp * aniso_field(Hy, Wx, cfg.get("wavelength", 10),
                                     cfg.get("angle", 0.0), cfg.get("phase", 0.0))).astype(np.float32)


def _patterns(sset):
    """Resolve the property-pattern config (new `patterns:` block, or legacy keys)."""
    if "patterns" in sset:
        return sset["patterns"]
    lam = sset.get("pattern_wavelength", 10)
    return {"stiffness": {"lo": sset.get("k_lo", 0.6), "hi": sset.get("k_hi", 1.8), "wavelength": lam, "phase": 0.7},
            "gain": {"lo": sset.get("g_lo", 0.4), "hi": sset.get("g_hi", 1.6), "wavelength": lam, "phase": 0.0},
            "fiber": {"mode": "smooth", "wavelength": lam, "amp": float(np.pi)}}


# --------------------------------------------------------------------------- #
#  Set builder
# --------------------------------------------------------------------------- #
def build_tissue(shape, pat, a=0.7, b=0.8, device="cpu", bwidth=1):
    Hy, Wx = shape
    N = Hy * Wx
    ys, xs = np.meshgrid(np.linspace(0, 1, Hy), np.linspace(0, 1, Wx), indexing="ij")
    state = torch.zeros(N, 6, device=device)
    state[:, 0] = torch.tensor(xs.ravel(), dtype=torch.float32)
    state[:, 1] = torch.tensor(ys.ravel(), dtype=torch.float32)
    u0 = -1.1994
    state[:, 4] = u0
    state[:, 5] = (u0 + a) / b
    lvl = Level("tissue_particle", 0, state, state_schema=SCHEMA)
    lvl.shape = (Hy, Wx)
    sc, gc, fc = pat["stiffness"], pat["gain"], pat["fiber"]
    het_k = aniso_field(Hy, Wx, sc.get("wavelength", 10), sc.get("angle", 0.0), sc.get("phase", 0.7)).ravel()
    het_g = aniso_field(Hy, Wx, gc.get("wavelength", 10), gc.get("angle", 0.0), gc.get("phase", 0.0)).ravel()
    fiber = fiber_field(Hy, Wx, fc).ravel()
    for nm, arr in [("I_ext", np.zeros(N)), ("act", np.zeros(N)),
                    ("stiff", sc["lo"] + (sc["hi"] - sc["lo"]) * het_k),
                    ("gain", gc["lo"] + (gc["hi"] - gc["lo"]) * het_g),
                    ("fiber", fiber)]:
        lvl.register_buffer(nm, torch.tensor(np.asarray(arr, np.float32), device=device))
    lvl.register_buffer("X", state[:, :2].clone())
    # boundary ring (the cultured sheet is attached at its edges) -- pinned to rest,
    # so interior twitches spring back to baseline between beats (a real silent period)
    bnd = np.zeros((Hy, Wx), bool)
    bw = max(1, int(bwidth))
    bnd[:bw, :] = bnd[-bw:, :] = bnd[:, :bw] = bnd[:, -bw:] = True
    lvl.register_buffer("boundary", torch.tensor(bnd.ravel(), device=device))
    H = Hierarchy(); H.add_level(lvl); H.dim = 2
    return H


# --------------------------------------------------------------------------- #
#  Operators new to Stage 2
# --------------------------------------------------------------------------- #
class SignalToMpmForce(Lateral):
    """Local: active contraction stress  a_i = gain_i * sigmoid((u_i - theta)/eta)."""
    KIND = "lateral"
    PREDICTION = None

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "tissue_particle")
        self.theta = float(params.get("theta", 0.0))
        self.eta = float(params.get("eta", 0.3))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        u = lvl.get("u").squeeze(1)
        lvl.act = lvl.gain * torch.sigmoid((u - self.theta) / self.eta)
        return {}


class MpmMechanics(Lateral):
    """Inertial, damped, anisotropic active elastic sheet on the grid edges.

    Decomposed stand-in for `mls_mpm_mechanics` with the same interface: activation
    shrinks each edge's target length along the local fibre, scaled by per-node
    stiffness; a substrate anchor bounds the motion; inertia gives curved (looped)
    per-node beat trajectories. Internal substeps for stability. Updates pos+vel.
    """
    KIND = "lateral"
    PREDICTION = None

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "tissue_particle")
        self.beta = float(params.get("beta", 0.2))       # max active shortening
        self.k_anch = float(params.get("k_anchor", 1.0))
        self.gamma = float(params.get("gamma", 2.0))     # overdamped drag (timescale)
        self.aniso = float(params.get("aniso", 0.85))
        self.nsub = int(params.get("substeps", 4))
        self.dt = float(params.get("dt", 0.1)) / self.nsub
        self._geo = None

    def _geo_build(self, lvl):
        i, j = lvl.edge_index
        d = lvl.X[j] - lvl.X[i]
        L0 = d.norm(dim=1).clamp(min=1e-6)
        edir = d / L0[:, None]
        fvec = torch.stack([torch.cos(lvl.fiber[i]), torch.sin(lvl.fiber[i])], 1)
        align = (edir * fvec).sum(1) ** 2
        ani = (1 - self.aniso) + self.aniso * align
        kedge = 0.5 * (lvl.stiff[i] + lvl.stiff[j])
        self._geo = (i, j, L0, ani, kedge)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self._geo is None:
            self._geo_build(lvl)
        i, j, L0, ani, kedge = self._geo
        a_edge = 0.5 * (lvl.act[i] + lvl.act[j])
        Lt = L0 * (1.0 - self.beta * a_edge * ani)
        x0 = lvl.get("pos").clone()
        for _ in range(self.nsub):
            x = lvl.get("pos")
            d = x[j] - x[i]; L = d.norm(dim=1).clamp(min=1e-6)
            fvec = (kedge * (L - Lt) / L)[:, None] * d            # spring along edge
            F = torch.zeros_like(x).index_add_(0, i, fvec)
            F = F + self.k_anch * (lvl.X - x)                     # substrate anchor -> self-relaxation
            lvl.state[:, 0:2] = x + (self.dt / self.gamma) * F    # overdamped Euler (no inertia)
        lvl.state[:, 2:4] = (lvl.get("pos") - x0) / (self.dt * self.nsub)   # velocity readout
        return {}


class PulseField(Lateral):
    """Global synchronous stimulus -- a uniform 'pulse field' over the whole set.

    The real cardiomyocyte sheet beats synchronously (no propagating front; see the
    real-data onset analysis), so instead of a localised trigger that launches a
    wave, this raises u on ALL nodes together at a fixed period. The grid_graph
    coupling in excitable_nagumo still interconnects the nodes (it would resync any
    perturbation), but the activation itself is global, not a wave. Conceptually a
    spatially-uniform field source delivered to the set; written in place (returns {}).
    """
    KIND = "lateral"
    PREDICTION = None

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "tissue_particle")
        self.period = int(params.get("period", 180))
        self.dur = int(params.get("dur", 6))
        self.amp = float(params.get("amp", 2.0))
        self.t0 = int(params.get("t_start", 1))
        self.mode = params.get("mode", "set")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        t = int(H.tick) - self.t0
        if t >= 0 and (t % self.period) < self.dur:
            uc = lvl.state_schema["u"][0]
            if self.mode == "set":
                lvl.state[:, uc] = self.amp                 # all nodes together
            else:
                lvl.I_ext += self.amp
        return {}


class BoundaryData(Lateral):
    """Drive the boundary ring with the REAL data (a time-dependent Dirichlet BC).
    The interior is predicted by the mechanics; the edges follow the measured tissue.
    Runs last in the schedule so it overrides the boundary each tick. Returns {}."""
    KIND = "rewire"
    PREDICTION = None
    MAY_MUTATE_INTEGRATED_STATE = True

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "tissue_particle")
        real = real_traj(int(params["n_frames"]), tuple(params["shape"]))
        self.real = torch.tensor(real, device=device)        # [n_frames, N, 2]

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        t = min(int(H.tick), self.real.shape[0] - 1)
        bnd = lvl.boundary
        lvl.state[bnd, 0:2] = self.real[t][bnd]
        return {}


OPS = {"grid_graph": GridGraph, "trigger_pulse": TriggerPulse, "pulse_field": PulseField,
       "excitable_nagumo": ExcitableNagumo, "signal_to_mpm_force": SignalToMpmForce,
       "mls_mpm_mechanics": MpmMechanics, "boundary_data": BoundaryData}


# --------------------------------------------------------------------------- #
#  Engine
# --------------------------------------------------------------------------- #
def run(spec, device="cpu"):
    g = spec["general"]
    sset = spec["sets"]["tissue_particle"]
    nag = next(o for o in spec["operators"] if o["op"] == "excitable_nagumo")
    H = build_tissue(tuple(sset["shape"]), _patterns(sset),
                     a=nag.get("a", 0.7), b=nag.get("b", 0.8), device=device,
                     bwidth=int(sset.get("boundary_width", 1)))
    lvl = H.level("tissue_particle")
    ops = {o["op"]: OPS[o["op"]]({**o, "_at": "tissue_particle", "dt": g["dt"],
                                  "n_frames": g["n_frames"], "shape": sset["shape"]}, device)
           for o in spec["operators"]}
    ops["grid_graph"].forward(H)
    rec = int(g.get("record_every", 2))
    pos, un = [], []
    for t in range(int(g["n_frames"])):
        H.tick = t
        for name in spec["schedule"]:
            ops[name].forward(H)
        if t % rec == 0:
            pos.append(lvl.get("pos").detach().cpu().numpy().copy())
            un.append(lvl.get("u").detach().cpu().numpy().squeeze(1).copy())
    return np.stack(pos), np.stack(un), lvl


# --------------------------------------------------------------------------- #
#  Intent metrics
# --------------------------------------------------------------------------- #
def metrics(pos, un, n_cycles, period_frames):
    """Cycle period (frames), and similarity of per-node displacement paths across cycles."""
    P = period_frames
    sims = []
    # per-cycle displacement RELATIVE to that cycle's start -> measures beat SHAPE,
    # drift-invariant (the real question: are the 4 beats' trajectories similar?)
    cyc = [pos[c * P:(c + 1) * P] - pos[c * P] for c in range(n_cycles)
           if (c + 1) * P <= pos.shape[0]]
    base = cyc[0].reshape(P, -1) if cyc else None
    for seg in cyc[1:]:
        seg = seg.reshape(P, -1)
        num = (base * seg).sum()
        den = np.linalg.norm(base) * np.linalg.norm(seg) + 1e-9
        sims.append(float(num / den))
    # count activation cycles: upward crossings of the excited-fraction trace
    act = (un > 0.0).mean(1)                             # fraction excited per frame
    thr = 0.03
    crossings = int(((act[1:] > thr) & (act[:-1] <= thr)).sum())
    return {"period_frames": P,
            "cross_cycle_similarity_mean": float(np.mean(sims)) if sims else float("nan"),
            "cross_cycle_similarity": [round(s, 3) for s in sims],
            "max_excited_fraction": round(float(act.max()), 3),
            "n_activation_cycles": crossings}


# --------------------------------------------------------------------------- #
#  Renderers (dark background, no microscope image)
# --------------------------------------------------------------------------- #
def _writer(out, fps):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    if os.path.exists(FFMPEG):
        plt.rcParams["animation.ffmpeg_path"] = FFMPEG
    return FFMpegWriter(fps=fps, bitrate=5000)


def render_nodes(pos, out, fps=48, stride=2, real=None):
    """Synthetic node dots (green) on black; optional real-data overlay (blue)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7)); ax.set_position([0, 0, 1, 1]); ax.axis("off")
    fig.set_facecolor("black"); ax.set_facecolor("black")
    rsc = ax.scatter(real[0][:, 0], real[0][:, 1], s=1.6, c="lime",
                     linewidths=0, alpha=0.55) if real is not None else None
    sc = ax.scatter(pos[0][:, 0], pos[0][:, 1], s=1.6, c="red", linewidths=0, alpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    w = _writer(out, fps)
    frames = range(0, pos.shape[0], stride)
    with w.saving(fig, out, dpi=128):
        for f in frames:
            sc.set_offsets(pos[f])
            if rsc is not None:
                rsc.set_offsets(real[f])
            w.grab_frame()
    plt.close(fig)
    print(f"saved {out}  ({os.path.getsize(out)/1e6:.1f} MB, {len(list(frames))} frames @ {fps}fps)")


def _sel_amp(pos, sel, amp):
    P = pos[:, sel]; rest = P[0]
    return rest[None] + amp * (P - rest[None])


def _traj_sel(shape, grid_n, margin_frac=10.0 / 137.0):
    """g x g INTERIOR node indices. Default margin = cog's select_grid_nodes (margin 10 on the
    137 grid), so the displayed crop matches cog_trajectories.mp4 / render_compare.png exactly.
    The anchored boundary band sits OUTSIDE this crop (anchoring still happens in train+render;
    it's just not the displayed outer ring). margin_frac=0 would span edge-to-edge instead."""
    Hy, Wx = shape
    my, mx = round(margin_frac * (Hy - 1)), round(margin_frac * (Wx - 1))
    ii = np.linspace(my, Hy - 1 - my, grid_n).round().astype(int)
    jj = np.linspace(mx, Wx - 1 - mx, grid_n).round().astype(int)
    return (ii[:, None] * Wx + jj[None, :]).ravel()


def render_trajectories(pos, out, shape, grid_n=10, amp=12.0, tail=90, fps=24, stride=1, real=None):
    """Synthetic 10x10 trajectories (green); optional real-data trajectories (blue)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    Hy, Wx = shape
    sel = _traj_sel(shape, grid_n)
    P = _sel_amp(pos, sel, amp)
    R = _sel_amp(real, sel, amp) if real is not None else None
    fig, ax = plt.subplots(figsize=(7, 7)); ax.set_position([0, 0, 1, 1]); ax.axis("off")
    fig.set_facecolor("black"); ax.set_facecolor("black")
    M = 0.12   # margin so amplified edge-node loops are not cropped at the frame
    ax.set_xlim(-M, 1 + M); ax.set_ylim(1 + M, -M); ax.set_aspect("equal")   # Y down (cog)
    rlc = LineCollection([], linewidths=1.1, colors=(0.2, 1.0, 0.2, 0.7)) if R is not None else None
    if rlc is not None:
        ax.add_collection(rlc)
    lc = LineCollection([], linewidths=1.3); ax.add_collection(lc)
    # small green dot tracking the TRUE (real) head, beside the red learned head
    rhead = ax.scatter(R[0, :, 0], R[0, :, 1], s=6, c="lime", zorder=2) if R is not None else None
    head = ax.scatter(P[0, :, 0], P[0, :, 1], s=13, c="red", edgecolors="black",
                      linewidths=0.4, zorder=3)

    def trail(A, t):
        win = A[max(0, t - tail):t + 1]
        if win.shape[0] < 2:
            return None
        return np.stack([win[:-1], win[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)

    w = _writer(out, fps)
    with w.saving(fig, out, dpi=128):
        for t in range(0, P.shape[0], stride):
            segs = trail(P, t)
            if segs is not None:
                age = np.tile(np.linspace(0, 1, segs.shape[0] // P.shape[1]), P.shape[1])
                col = np.zeros((age.size, 4), np.float32); col[:, 0] = 1.0; col[:, 3] = 0.2 + 0.8 * age  # red=learned
                lc.set_segments(list(segs)); lc.set_color(col)
            if rlc is not None:
                rs = trail(R, t)
                if rs is not None:
                    rlc.set_segments(list(rs))
            if rhead is not None:
                rhead.set_offsets(R[t])
            head.set_offsets(P[t]); w.grab_frame()
    plt.close(fig)
    print(f"saved {out}  ({os.path.getsize(out)/1e6:.1f} MB, {grid_n}x{grid_n} nodes, amp={amp}"
          f"{', +real(green)' if real is not None else ''})")


def render_properties(lvl, out):
    """Multi-panel map of the smooth mechanical-property fields that break symmetry."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Hy, Wx = lvl.shape
    stiff = lvl.stiff.cpu().numpy().reshape(Hy, Wx)
    gain = lvl.gain.cpu().numpy().reshape(Hy, Wx)
    fib = lvl.fiber.cpu().numpy().reshape(Hy, Wx)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    im0 = axs[0].imshow(stiff, cmap="viridis"); axs[0].set_title("stiffness")
    fig.colorbar(im0, ax=axs[0], fraction=0.046)
    im1 = axs[1].imshow(gain, cmap="magma"); axs[1].set_title("active contraction gain")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)
    im2 = axs[2].imshow((fib % np.pi), cmap="twilight"); axs[2].set_title("fibre angle (mod π)")
    fig.colorbar(im2, ax=axs[2], fraction=0.046)
    s = max(1, Hy // 26)
    yy, xx = np.meshgrid(np.arange(0, Hy, s), np.arange(0, Wx, s), indexing="ij")
    axs[3].imshow(gain, cmap="gray", alpha=0.4)
    axs[3].quiver(xx, yy, np.cos(fib[::s, ::s]), np.sin(fib[::s, ::s]),
                  color="red", pivot="mid", scale=26)
    axs[3].set_title("fibre directions over gain"); axs[3].set_aspect("equal")
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def render_traj_png(pos, out, shape, grid_n=10, amp=12.0, real=None):
    """Static 10x10 amplified node trajectories (full run); synthetic green, real blue."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    Hy, Wx = shape
    sel = _traj_sel(shape, grid_n)
    P = _sel_amp(pos, sel, amp)                                # [T, M, 2]
    fig, ax = plt.subplots(figsize=(7, 7)); ax.set_position([0, 0, 1, 1]); ax.axis("off")
    M = 0.12   # margin so amplified edge-node loops are not cropped at the frame
    ax.set_facecolor("black"); ax.set_xlim(-M, 1 + M)
    ax.set_ylim(1 + M, -M)                                     # image convention (Y down), matches cog
    ax.set_aspect("equal")
    if real is not None:
        R = _sel_amp(real, sel, amp)
        rsegs = np.stack([R[:-1], R[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
        ax.add_collection(LineCollection(list(rsegs), colors=(0.2, 1.0, 0.2, 0.6), linewidths=1.0))
    segs = np.stack([P[:-1], P[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    age = np.tile(np.linspace(0, 1, P.shape[0] - 1), P.shape[1])
    col = np.zeros((age.size, 4), np.float32); col[:, 0] = 1.0; col[:, 3] = 0.15 + 0.85 * age  # red=learned
    ax.add_collection(LineCollection(list(segs), colors=col, linewidths=1.0))
    ax.scatter(P[0, :, 0], P[0, :, 1], s=10, c="red", edgecolors="black", linewidths=0.4)
    fig.patch.set_facecolor("black")
    fig.savefig(out, dpi=130, facecolor="black"); plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    name = spec["general"]["name"]
    shape = tuple(spec["sets"]["tissue_particle"]["shape"])
    fps = int(spec["general"].get("fps", 24))
    n_cyc = int(spec["general"]["n_cycles"])
    rec = int(spec["general"].get("record_every", 2))
    period_f = int(spec["general"]["cycle_frames"] // rec)

    pos, un, lvl = run(spec, device=args.device)
    disp = np.linalg.norm(pos - pos[0], axis=2)
    met = metrics(pos, un, n_cyc, period_f)
    print(f"displacement mean {disp.mean():.4f} max {disp.max():.4f} | "
          f"cycles {met['n_activation_cycles']} max_excited {met['max_excited_fraction']} | "
          f"cross-cycle sim {met['cross_cycle_similarity']} mean {met['cross_cycle_similarity_mean']:.3f}")

    step = spec["general"].get("step")
    dname = f"{step}_{name}" if step is not None else name
    d = os.path.join(HERE, "archive", dname); os.makedirs(d, exist_ok=True)
    vfps = int(spec["general"].get("video_fps", 96))   # both videos at one rate (= current node fps x2)
    vstride = int(spec["general"].get("video_stride", 1))
    # real (true) trajectory in blue as a background reference (always, if grid matches);
    # on the mp4s only when the boundary is actually anchored to the real data.
    try:
        real_rec = real_traj(pos.shape[0], shape)
    except (ValueError, FileNotFoundError):
        real_rec = None
    anchored = any(o["op"] == "boundary_data" for o in spec["operators"])
    render_properties(lvl, os.path.join(d, f"{name}_properties.png"))
    render_traj_png(pos, os.path.join(d, f"{name}_trajectories.png"), shape, real=real_rec)
    render_nodes(pos, os.path.join(d, f"{name}_nodes.mp4"), fps=vfps, stride=vstride,
                 real=real_rec if anchored else None)
    render_trajectories(pos, os.path.join(d, f"{name}_trajectories.mp4"), shape, fps=vfps,
                        stride=vstride, real=real_rec if anchored else None)
    with open(os.path.join(d, "spec.yaml"), "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    with open(os.path.join(d, "metrics.yaml"), "w") as f:
        yaml.safe_dump(met, f, sort_keys=False)
    # per-run log line (the process record; see LOGBOOK.md for the narrative)
    ops = {o["op"]: {k: v for k, v in o.items() if k not in ("op",)} for o in spec["operators"]}
    with open(os.path.join(d, "run.log"), "w") as f:
        f.write(f"name={name} step={step}  shape={shape}\n")
        f.write(f"displacement mean={disp.mean():.4f} max={disp.max():.4f}\n")
        f.write(f"cycles={met['n_activation_cycles']} max_excited={met['max_excited_fraction']}\n")
        f.write(f"cross_cycle_similarity={met['cross_cycle_similarity']} "
                f"mean={met['cross_cycle_similarity_mean']:.3f}\n")
        f.write(f"pulse_field={ops.get('pulse_field')}\n")
        f.write(f"excitable_nagumo={ops.get('excitable_nagumo')}\n")
        f.write(f"mls_mpm_mechanics={ops.get('mls_mpm_mechanics')}\n")
    print(f"archived -> {d}")


if __name__ == "__main__":
    main()
