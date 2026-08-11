#!/usr/bin/env python
"""test_05_sheet -- 05a: does a codimension-1 basement membrane carry its OWN deformation?

    python test_05_sheet.py [--device cuda:0] [--frames 401] [--subdiv 4]
                                                          ->  log/okuda_ECM/05a_sheet/

THE ONE QUESTION. Every archived version of this sheet reported a stretch that was not the stretch the
geometry had applied to it. Run 130 moved the sheet by a factor of 3.44 in radius and its material
reported 0.31 of the 2.44 strain that implies -- 13% -- because a particle moved by an engine delta
never passes through the MPM grid whose velocity gradient advects F. Run 121 routed the same force
through the grid, recovered 2.25 of 2.30 (98%), and paid for it with the standoff. THE COUPLING TEST IS
WORTHLESS ON A SHEET THAT LAUNDERS STRAIN, which is why the plaque (05b) waits for this run.

On a mesh there is nothing to launder: F is measured per triangle from where its nodes are against where
they were seeded (`bm_ops.cauchy_green`), so it is blind to what moved them. That is a claim about a
formula and it is certified before this rig runs -- `bm_ops.selftest()` applies a rigid motion, a
uniform dilation and an anisotropic affine map and reports the error against their known singular
values. The run below is the same measure on a load it cannot check in closed form.

THE LOAD IS THE REAL EPITHELIUM, not a sphere with a growth law. `smap` in the tissue cache is the
recorded boundary of the 200 -> 3170 cell spheroid, 32x64 directions x 402 frames, and every node of the
sheet is tethered to its own frozen direction in it. So the sheet is driven by exactly the surface it
will face in `04`, at 0.0398 -> 0.1351 box units, a radius ratio of 3.39.

WHAT IS MEASURED, and each of these can come back wrong:

  1. STRETCH FIDELITY   mean lambda_geo from the sheet's own F against R(t)/R(0) from the drive.
                        The number to beat is 130's 0.13 and 121's 0.98. A mesh should give 1.00, and
                        if it does not, the sheet is being moved by something this rig does not model.
  2. THE STANDOFF       where the sheet sits relative to the surface it is tethered to, against the
                        tracking-lag prediction dR/dt / (M kappa). The archived line's four best runs
                        put it at -0.016 to +0.004 and the gap between prediction and measurement is
                        the sheet's OWN HOOP STRESS pulling it inward -- HANDOVER's "at realistic
                        stiffness the sheet's hoop stress beats a stable tether", now with the term
                        separated rather than inferred.
  3. REMODELLING        tau_r finite lets the reference metric creep toward the current one, so
                        lambda_el (what the material feels) decouples from lambda_geo (what the tissue
                        did). If the standoff in 2 is hoop stress, remodelling must recover it, and
                        that is the prediction HANDOVER left unrun: "the next step is 155 with
                        remodelling on, and it is the first time turning it on is a mechanism".
  4. STABILITY          a massless sheet's limit is a RATE, not a wave speed: dt*M*lambda_max < 2 with
                        lambda_max the largest eigenvalue of the elastic Hessian, MEASURED by power
                        iteration rather than asserted. The sweep runs 0.25 to 4 across it.
  5. AREAL DENSITY      rho = rho0 / J with no secretion, so a sheet enclosing a tripling radius thins
                        by ~11x. This is the baseline `bm_secrete` has to beat and the reason a fixed
                        particle budget was never going to work.
  6. TWO CONTROLS       kappa = 0 (nothing pulls: lambda must stay 1.000, or the measure invents
                        strain) and a rigid translation of the whole drive (lambda must stay 1.000, or
                        the measure confuses motion with deformation).

WHAT THIS IS NOT. No plaque -- the tether here is a one-sided spring to a FROZEN DIRECTION, which is
precisely what note S9 says an adhesion is not, and it returns no reaction to anything. No matrix, no
secretion, no proteolysis, no bending stiffness. The sheet cannot yet lose material, gain it, or fold.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.animation import FFMpegWriter                           # noqa: E402
from matplotlib.colors import ListedColormap                           # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection                # noqa: E402

import bm_ops as BM                                                     # noqa: E402
import ecm_spec as ES                                                   # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
TISSUE = os.path.join(LOG, "_tissue", "cellfix_B_new_f401.npz")
SCALE = 0.00853007254279858          # tissue units -> box units, the same number `04c` runs at
CMAP = ListedColormap(ES.STRESS_COLORS)
TARGET_C = "#8fb8de"

# =============================================================================================
#  THE PHYSICAL SCALE OF THESE RUNS, declared once (plexus/units.py) instead of assumed per figure.
#  Three base scales, everything else derived.
#
#  LENGTH, and it is MEASURED rather than chosen. At the last frame the cache holds 3170 cells on an
#  apical sphere of radius 0.13567 box units, so a cell covers 7.30e-5 box^2 and is 8.54e-3 box across.
#  ONE assumption enters -- an epithelial cell is ~10 um -- and it is the only one: 1 box = 1171 um,
#  which puts the spheroid at 318 um diameter, the size a spheroid is.
#
#  TIME. 200 -> 3170 cells is 3.99 doublings over 401 frames, i.e. 100.6 frames per doubling; at a
#  12-24 h cell cycle a frame is 429-859 s. The rigs advance dt = 1 per frame, so 1 time unit = 600 s
#  (10 min) sits in the middle of that range.
#
#  FORCE is NOT declared, and that is a statement rather than an omission: nothing in this prototype
#  fixes a force scale, so only force RATIOS (E_bm/E_ecm, E(lambda)/E0) are meaningful and the loader
#  will say so. Quoting a modulus in MPa here would be quoting a number as a pressure.
#
#  WHAT THE DECLARATION IMMEDIATELY EXPOSES: T = 2e-3 box = 2.34 um against the ~0.1 um a basement
#  membrane actually is -- 23x too thick -- and since l0 = 0.3T and Sigma = 7T are defined IN T, every
#  sheet length is too large by the same factor. The proportions are right; the absolute scale is not.
#  That was invisible while the model was dimensionless, which is the whole argument for declaring it.
# =============================================================================================
UNITS = dict(length_um=1171.0, time_s=600.0)
T_REAL_UM = 0.1                      # a basement membrane, from the literature


# =============================================================================================
#  the drive: the recorded epithelial surface, read at the sheet's own directions
# =============================================================================================
class SurfaceReplay:
    """R(u, t) for a fixed set of directions, from the pass-1 recording -- `surface_ops` without the
    engine around it. Distance-weighted over the k nearest map directions, so there are no bin edges
    for a strain field to remember (which is the defect that promoting `surface` to a Level fixed)."""

    def __init__(self, u, path=TISSUE, scale=SCALE, k=6, dev="cuda:0", dtype=torch.float64):
        z = np.load(path, allow_pickle=True)
        self.smap = torch.as_tensor(np.asarray(z["smap"]), dtype=dtype, device=dev) * scale
        self.T, nth, nph = self.smap.shape
        th = (torch.arange(nth, dtype=dtype, device=dev) + 0.5) / nth * math.pi
        ph = (torch.arange(nph, dtype=dtype, device=dev) + 0.5) / nph * 2 * math.pi
        T2, P2 = torch.meshgrid(th, ph, indexing="ij")
        mu = torch.stack([torch.sin(T2) * torch.cos(P2), torch.sin(T2) * torch.sin(P2),
                          torch.cos(T2)], -1).reshape(-1, 3)
        cs = u @ mu.T
        self.nb = torch.topk(cs, k, dim=1).indices
        w = (1.0 - torch.gather(cs, 1, self.nb)).clamp_min(1e-6)
        self.w = (1.0 / w) / (1.0 / w).sum(1, keepdim=True)
        self.flat = self.smap.reshape(self.T, -1)

    def R(self, t):
        t = int(min(self.T - 1, max(0, t)))
        return (self.flat[t][self.nb] * self.w).sum(1)


# =============================================================================================
#  the rig
# =============================================================================================
class Rig05a:
    """A codim-1 sheet tethered to the recorded surface, one frame at a time.

    `kappa` is a stiffness PER NODE against its own anchor and enters the stability bound additively
    with the elastic Hessian, which is why the rig measures lambda_max of the elastic part and adds
    kappa to it rather than guessing a combined number.
    """

    def __init__(self, subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, beta=0.0, kappa=5.0,
                 delta=6.0e-4, zeta=20.0, s_target=1.0, refresh=10, tau_r=0.0, dev="cuda:0",
                 drive="replay", dtype=torch.float64):
        self.dev, self.dtype = dev, dtype
        V, Fc, Ed = BM.icosphere(subdiv, device=dev, dtype=dtype)
        self.rep = SurfaceReplay(V, dev=dev, dtype=dtype)
        R_seed = self.rep.R(0) + delta
        self.sheet = BM.Sheet(subdiv=subdiv, R0=1.0, E=E, thickness=thickness, nu=nu, beta=beta,
                              tau_r=tau_r, dev=dev, dtype=dtype)
        # SEED ON THE RECORDED SURFACE, not on a sphere: the reference metric must be the metric of the
        # shape the sheet is actually born on, or frame 0 already reports a strain the tissue never
        # applied. (`Sheet` is built at R0 = 1 and re-seeded here, which rebuilds its reference frame.)
        self.sheet.reseed(self.sheet.c + V * R_seed[:, None])
        self.u0, self.delta, self.kappa = V, float(delta), float(kappa)
        self.drive = drive
        # THE MOBILITY IS SET BY A RATE, not chosen. zeta is how many e-folds of elastic relaxation the
        # sheet gets per FRAME: "does the sheet relax faster than the tissue grows" is the only question
        # a massless sheet's stiffness can answer, and it is this number.
        self.lam_el, self._pv = self.sheet.spectral_rate(return_vec=True)
        self.lam_el_seed = self.lam_el
        self.lam_tot = self.lam_el + self.kappa
        self.sheet.M = float(zeta) / self.lam_tot
        # THE SUBSTEP COUNT IS SET BY THE MEASUREMENT, EVERY `refresh` FRAMES, and this is not a
        # convenience. A StVK membrane's tangent stiffens as it stretches, so lambda_max at the seeded
        # configuration is an underestimate of lambda_max at 3.4x the radius: holding n_sub fixed at
        # the seeded bound put the sweep's divergence threshold at 1.8 rather than the 2 the overdamped
        # theory gives, and the entire discrepancy was the Hessian having grown under the run's own
        # load. Held this way, the group `s_target` is the same number at every frame and the sweep is
        # a test of the bound rather than of how much the sheet stiffened.
        self.s_target, self.refresh = float(s_target), int(refresh)
        self.n_sub = self._nsub()
        self.stability = self.s_target
        self.R0_mean = float(self.rep.R(0).mean())
        self.res = {k: [] for k in ("lam_geo", "lam_geo_p99", "lam_el", "lam_min", "R_sheet",
                                    "R_target", "standoff", "rho", "energy", "fmax", "area",
                                    "Y_over_Y0", "J", "lam_hess", "n_sub")}

    def _nsub(self):
        return max(1, int(math.ceil(self.sheet.M * (self.lam_el + self.kappa) / self.s_target)))

    def anchors(self, t):
        R = self.rep.R(t) + self.delta
        a = self.sheet.c + self.u0 * R[:, None]
        if self.drive == "rigid":
            # THE RIGID CONTROL: the drive translates and rotates but never dilates, so a measure that
            # confuses motion with deformation reports a stretch and a correct one reports 1.000.
            th = 0.004 * t
            Rz = torch.tensor([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0],
                               [0.0, 0.0, 1.0]], device=self.dev, dtype=self.dtype)
            a0 = self.sheet.c + self.u0 * (self.rep.R(0) + self.delta)[:, None]
            a = (a0 - self.sheet.c) @ Rz.T + self.sheet.c \
                + torch.tensor([2.0e-4 * t, 0.0, 1.0e-4 * t], device=self.dev, dtype=self.dtype)
        return a, R

    def frame(self, t):
        a, Rt = self.anchors(t)
        f = None
        if t % self.refresh == 0 and torch.isfinite(self.sheet.x).all():
            self.lam_el, self._pv = self.sheet.spectral_rate(iters=25, v0=self._pv, return_vec=True)
            self.n_sub = self._nsub()
        dt_sub = 1.0 / self.n_sub
        for _ in range(self.n_sub):
            ft = self.kappa * (a - self.sheet.x)
            f = self.sheet.step(dt_sub, extra_force=ft)
        l1, l2 = self.sheet.stretch_geo()
        e1, _ = self.sheet.stretch_elastic()
        r = (self.sheet.x - self.sheet.c).norm(dim=1)
        self.res["lam_geo"].append(float(l1.mean()))
        self.res["lam_geo_p99"].append(float(torch.quantile(l1, 0.99)))
        self.res["lam_min"].append(float(l2.mean()))
        self.res["lam_el"].append(float(e1.mean()))
        self.res["J"].append(float((l1 * l2).mean()))
        self.res["R_sheet"].append(float(r.mean()))
        self.res["R_target"].append(float(Rt.mean()))
        self.res["standoff"].append(float(r.mean() - Rt.mean()))
        self.res["rho"].append(float(self.sheet.areal_density().mean()))
        self.res["energy"].append(float(self.sheet.energy(self.sheet.x)))
        self.res["fmax"].append(float(f.norm(dim=1).max()))
        self.res["area"].append(float(self._area()))
        self.res["Y_over_Y0"].append(float(self.sheet.Y2[self.sheet.live].mean())
                                     / (self.sheet.E0 * self.sheet.T))
        self.res["lam_hess"].append(float(self.lam_el))
        self.res["n_sub"].append(int(self.n_sub))
        return l1

    def _area(self):
        x = self.sheet.x
        v0, v1, v2 = x[self.sheet.Fc[:, 0]], x[self.sheet.Fc[:, 1]], x[self.sheet.Fc[:, 2]]
        return 0.5 * torch.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1).sum()

    def alive(self):
        return bool(torch.isfinite(self.sheet.x).all())


def run(rig, frames, keep=None, label=""):
    """Step the rig, keeping (t, positions, per-face stretch) on the frames the movie wants."""
    keep = set() if keep is None else keep
    kept, t0 = [], time.time()
    for t in range(frames):
        l1 = rig.frame(t)
        if not rig.alive():
            print(f"[{label}] DIVERGED at frame {t} (stability group "
                  f"{rig.stability:.3g})", flush=True)
            return kept, t
        if t in keep:
            kept.append((t, rig.sheet.x.detach().float().cpu().numpy(),
                         l1.detach().float().cpu().numpy(),
                         (rig.rep.R(t) + rig.delta).detach().float().cpu().numpy()))
    if label:
        print(f"[{label}] {frames} frames in {time.time()-t0:.1f}s -- "
              f"lambda_geo {rig.res['lam_geo'][-1]:.4f}, lambda_el {rig.res['lam_el'][-1]:.4f}, "
              f"standoff {rig.res['standoff'][-1]:+.3e}", flush=True)
    return kept, frames


# =============================================================================================
#  rendering
# =============================================================================================
def _tri_collection(x, faces, col, cmap, vmin, vmax):
    tri = x[faces]
    pc = Poly3DCollection(tri, linewidths=0.0)
    pc.set_facecolor(cmap((np.clip(col, vmin, vmax) - vmin) / max(vmax - vmin, 1e-12)))
    return pc


def render(kept, faces, d, name, s_hi, fps=20):
    """A movie whose colour scale is one p99 over the whole run, fixed before the first frame is drawn
    -- the convention every artefact in this prototype uses, and the reason 02's first movie was a
    saturated block."""
    fig = plt.figure(figsize=(11.6, 5.8), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": name})
    strip = []
    strip_at = set(np.round(np.linspace(0, len(kept) - 1, 8)).astype(int).tolist())
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for i, (t, X, L, Rt) in enumerate(kept):
            fig.clf()
            c = X.mean(0) * 0 + 0.5
            lim = 0.165
            ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
            ax.set_facecolor("black"); ax.axis("off")
            # THE FAR HALF ONLY, so the camera looks INTO the lumen. Drawn whole, a closed shell is
            # indistinguishable from a filled ball, and the one thing this rig has to show is that the
            # sheet is a surface enclosing nothing.
            keepf = X[faces][:, :, 1].mean(1) > c[1]
            ax.add_collection3d(_tri_collection(X, faces[keepf], L[keepf], CMAP, 1.0, s_hi))
            ax.set_xlim(c[0] - lim, c[0] + lim); ax.set_ylim(c[1] - lim, c[1] + lim)
            ax.set_zlim(c[2] - lim, c[2] + lim)
            try:
                ax.set_box_aspect((1, 1, 1), zoom=1.55)
            except TypeError:
                ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=16, azim=-58)
            ax.text2D(0.02, 0.97, f"{name}   frame {t}\n"
                                  f"$\\lambda_{{geo}}$ mean {L.mean():.3f}   colour to {s_hi:.2f}",
                      transform=ax.transAxes, color="white", fontsize=10.5, va="top")
            a2 = fig.add_subplot(1, 2, 2, facecolor="black")
            sl = np.abs(X[:, 1] - c[1]) < 0.004
            nl = np.zeros(X.shape[0]); cnt = np.zeros(X.shape[0])
            np.add.at(nl, faces.reshape(-1), np.repeat(L, 3)); np.add.at(cnt, faces.reshape(-1), 1)
            nl = nl / np.maximum(cnt, 1)
            a2.scatter(X[sl][:, 0], X[sl][:, 2], s=11, c=np.clip(nl[sl], 1.0, s_hi), cmap=CMAP,
                       vmin=1.0, vmax=s_hi, marker=".", linewidths=0)
            th = np.linspace(0, 2 * np.pi, 200)
            a2.plot(c[0] + Rt.mean() * np.cos(th), c[2] + Rt.mean() * np.sin(th), "--",
                    color=TARGET_C, lw=1.0)
            a2.set_xlim(c[0] - lim, c[0] + lim); a2.set_ylim(c[2] - lim, c[2] + lim)
            a2.set_aspect("equal"); a2.axis("off")
            a2.text(0.02, 0.98, "section; dashed = the recorded epithelial surface it is tethered to",
                    transform=a2.transAxes, color="white", fontsize=9.5, va="top")
            wri.grab_frame()
            if i in strip_at:
                strip.append((t, X.copy(), nl.copy(), Rt.mean()))
    fig.savefig(os.path.join(d, "3d.png"), dpi=115, facecolor="black")
    plt.close(fig)
    figs = plt.figure(figsize=(3.0 * len(strip), 3.3), facecolor="black")
    for i, (t, X, nl, Rm) in enumerate(strip):
        a = figs.add_subplot(1, len(strip), i + 1, facecolor="black")
        sl = np.abs(X[:, 1] - 0.5) < 0.004
        a.scatter(X[sl][:, 0], X[sl][:, 2], s=7, c=np.clip(nl[sl], 1.0, s_hi), cmap=CMAP, vmin=1.0,
                  vmax=s_hi, marker=".", linewidths=0)
        th = np.linspace(0, 2 * np.pi, 200)
        a.plot(0.5 + Rm * np.cos(th), 0.5 + Rm * np.sin(th), "--", color=TARGET_C, lw=0.9)
        a.set_xlim(0.335, 0.665); a.set_ylim(0.335, 0.665); a.set_aspect("equal"); a.axis("off")
        a.text(0.03, 0.97, f"frame {t}", transform=a.transAxes, color="white", fontsize=11, va="top")
    figs.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02)
    figs.savefig(os.path.join(d, "strip.png"), dpi=115, facecolor="black")
    plt.close(figs)


def metrics_png(nom, ctrl_free, ctrl_rigid, remodel, stab, stiff, d, frames):
    fig, ax = plt.subplots(2, 4, figsize=(17.4, 6.8), facecolor="white")
    t = np.arange(len(nom["lam_geo"]))
    # a: the headline -- measured stretch against the stretch the drive applied
    geo = np.asarray(nom["R_target"]) / nom["R_target"][0]
    ax[0, 0].plot(t, nom["lam_geo"], color="#1a1a1a", lw=1.8, label=r"measured $\lambda_{geo}$")
    ax[0, 0].plot(t, geo, color="#1f8a5c", lw=1.4, ls="--", label=r"$R(t)/R(0)$ of the drive")
    ax[0, 0].plot(t, ctrl_free["lam_geo"], color="#b03030", lw=1.2,
                  label=r"control: $\kappa=0$ (nothing pulls)")
    ax[0, 0].plot(t, ctrl_rigid["lam_geo"], color="#c07820", lw=1.2, ls=":",
                  label="control: rigid drive")
    ax[0, 0].set_ylabel(r"principal stretch $\lambda_1$")
    ax[0, 0].set_title(f"fidelity {nom['lam_geo'][-1]/geo[-1]:.4f} of the applied stretch\n"
                       f"(MPM: 0.13 direct-force, 0.98 through the grid)", fontsize=8.5)
    ax[0, 0].legend(fontsize=7, frameon=False)
    # b: the standoff, and what the tracking lag alone predicts
    ax[0, 1].plot(t, np.asarray(nom["standoff"]) * 1e3, color="#1a1a1a", lw=1.6, label="measured")
    ax[0, 1].axhline(0.0, color="#999", lw=0.8)
    for lab, r in remodel.items():
        ax[0, 1].plot(t[:len(r["standoff"])], np.asarray(r["standoff"]) * 1e3, lw=1.3,
                      label=lab)
    ax[0, 1].set_ylabel(r"$\langle r\rangle - \langle R\rangle$  ($10^{-3}$ box units)")
    ax[0, 1].set_title("where the sheet sits, against the surface it is tethered to", fontsize=8.5)
    ax[0, 1].legend(fontsize=7, frameon=False)
    # c: the two stretches, which is the whole point of remodelling
    ax[0, 2].plot(t, nom["lam_geo"], color="#1a1a1a", lw=1.6, label=r"$\lambda_{geo}$, no remodelling")
    ax[0, 2].plot(t, nom["lam_el"], color="#1a1a1a", lw=1.6, ls="--",
                  label=r"$\lambda_{el}$, no remodelling")
    for lab, r in remodel.items():
        ax[0, 2].plot(t[:len(r["lam_el"])], r["lam_el"], lw=1.3, ls="--", label=r"$\lambda_{el}$, " + lab)
    ax[0, 2].set_ylabel("stretch")
    ax[0, 2].set_title(r"what the tissue did ($\lambda_{geo}$) vs what the material feels "
                       r"($\lambda_{el}$)", fontsize=8.5)
    ax[0, 2].legend(fontsize=7, frameon=False)
    # d: areal density -- the case for a mass balance
    ax[1, 0].plot(t, nom["rho"], color="#1a1a1a", lw=1.6)
    ax[1, 0].set_ylabel(r"areal density $\rho/\rho_0$")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title(f"with no secretion the sheet thins to 1/{1/nom['rho'][-1]:.1f} of what it\n"
                       f"was seeded with: this is what `bm_secrete` has to hold flat", fontsize=8.5)
    # e: stability -- measured, against the bound
    s = np.asarray([r["s"] for r in stab]); ok = np.asarray([r["ok"] for r in stab])
    ax[1, 1].scatter(s[ok], np.ones(ok.sum()), s=70, marker="o", color="#1f8a5c", label="finite")
    ax[1, 1].scatter(s[~ok], np.zeros((~ok).sum()), s=70, marker="x", color="#b03030",
                     label="diverged")
    ax[1, 1].axvline(2.0, color="#1a1a1a", ls="--", lw=1.3)
    ax[1, 1].set_xscale("log"); ax[1, 1].set_yticks([0, 1]); ax[1, 1].set_yticklabels(["NaN", "ran"])
    ax[1, 1].set_xlabel(r"$\Delta t\, M\, \lambda_{max}$ (Hessian $\lambda_{max}$ measured)")
    ax[1, 1].set_title("the overdamped bound is a RATE: predicted 2, dashed", fontsize=8.5)
    ax[1, 1].legend(fontsize=7, frameon=False)
    # f: energy -- what a sheet that cannot forget is storing
    ax[0, 3].semilogy(t, np.maximum(nom["energy"], 1e-30), color="#1a1a1a", lw=1.6,
                      label="no remodelling")
    for lab, r in remodel.items():
        ax[0, 3].semilogy(t[:len(r["energy"])], np.maximum(r["energy"], 1e-30), lw=1.3, label=lab)
    ax[0, 3].set_ylabel("elastic energy of the sheet")
    pos_e = [v for v in nom["energy"] if v > 0]
    if pos_e:                       # frame 0 is exactly 0 and a log axis would draw down to 1e-30
        ax[0, 3].set_ylim(min(pos_e) * 0.5, max(nom["energy"]) * 2)
    ax[0, 3].set_title("a sheet that cannot forget stores everything it is\nstretched by", fontsize=8.5)
    ax[0, 3].legend(fontsize=7, frameon=False)
    # g: the tangent stiffness is not a constant, which is why the substep count is not either
    ax[1, 2].plot(t, nom["lam_hess"], color="#1a1a1a", lw=1.6, label=r"$\lambda_{max}$(Hessian)")
    a2 = ax[1, 2].twinx()
    a2.plot(t, nom["n_sub"], color="#2b6cb0", lw=1.3, ls="--")
    a2.set_ylabel("substeps per frame", color="#2b6cb0")
    ax[1, 2].set_ylabel(r"$\lambda_{max}$ of the elastic Hessian")
    ax[1, 2].set_title(f"the tangent stiffens {nom['lam_hess'][-1]/nom['lam_hess'][0]:.1f}x over the "
                       f"run: a bound taken\nat the seeded state would be wrong by that factor",
                       fontsize=8.5)
    ax[1, 2].legend(fontsize=7, frameon=False, loc="upper left")
    # h: the stiffening law, as a ratio -- MPa is not a quantity this box has
    ax[1, 3].plot(t, stiff["Y_over_Y0"], color="#7a3b9a", lw=1.7,
                  label=r"$\beta=5$: $E(\lambda)/E_0$")
    ax[1, 3].plot(t, nom["Y_over_Y0"], color="#1a1a1a", lw=1.3, ls="--", label=r"$\beta=0$")
    ax[1, 3].set_ylabel(r"$E(\lambda)/E_0$, the tangent modulus ratio")
    ax[1, 3].set_title(r"Candiello 2007 measures 0.4 $\to$ 3 MPa on native BM," "\n"
                       r"i.e. a ratio of 7.5 -- the only form the box can carry", fontsize=8.5)
    ax[1, 3].legend(fontsize=7, frameon=False)
    for a in ax.reshape(-1):
        a.set_xlabel("frame"); a.spines[["top", "right"]].set_visible(False)
    ax[1, 1].set_xlabel(r"$\Delta t\, M\, \lambda_{max}$ (Hessian $\lambda_{max}$ measured)")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "metrics.png"), dpi=150, facecolor="white")
    plt.close(fig)


# =============================================================================================
def main():
    def arg(flag, cast, default):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 401)
    subdiv = arg("--subdiv", int, 4)
    name = arg("--name", str, "05a_sheet")
    d = os.path.join(LOG, name)
    os.makedirs(d, exist_ok=True)

    # ---- CERTIFY THE MEASURE BEFORE RUNNING ANYTHING ON IT ------------------------------------
    cert = BM.selftest(dev=dev, subdiv=subdiv)
    assert cert["rigid_max_err"] < 1e-6 and cert["affine_max_err"] < 1e-10, cert

    P = dict(subdiv=subdiv, E=400.0, thickness=2.0e-3, nu=0.3, kappa=5.0, delta=6.0e-4,
             zeta=20.0, s_target=1.0, dev=dev)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 160))).astype(int).tolist())

    nom = Rig05a(**P)
    print(f"[{name}] {nom.sheet.n} nodes, {nom.sheet.m} faces; elastic lambda_max "
          f"{nom.lam_el:.4g} + kappa {nom.kappa:g}; M = {nom.sheet.M:.4g} (zeta = {P['zeta']:g} "
          f"e-folds per frame); stability group {nom.stability:.3g} (bound 2)", flush=True)
    kept, _ = run(nom, frames, keep=keep, label=name)

    # ---- the two controls, same rig, one thing removed ---------------------------------------
    free = Rig05a(**{**P, "kappa": 0.0})
    run(free, frames, label="control kappa=0")
    rigid = Rig05a(**P, drive="rigid")
    run(rigid, frames, label="control rigid drive")

    # ---- remodelling: the mechanism HANDOVER predicted and never ran --------------------------
    remodel = {}
    for tau in (100.0, 25.0):
        r = Rig05a(**{**P, "tau_r": tau})
        run(r, frames, label=f"tau_r = {tau:g} frames")
        remodel[rf"$\tau_r$ = {tau:g} frames"] = r.res
    # ---- the stiffening law, measured as a ratio and not in MPa -------------------------------
    stiff = Rig05a(**{**P, "beta": 5.0})
    run(stiff, frames, label="beta = 5 (strain stiffening)")

    # ---- stability: either side of the measured bound -----------------------------------------
    stab = []
    n_stab = min(200, frames)
    for s_target in (0.25, 0.5, 1.0, 1.5, 1.9, 2.1, 2.5, 3.0, 4.0):
        # the physics is identical in every one of these; the ONLY difference is the group the
        # substep count is held at, which is what makes the divergence threshold a test of the bound.
        r = Rig05a(**{**P, "s_target": s_target})
        _, reached = run(r, n_stab)
        stab.append(dict(s=s_target, n_sub_final=r.n_sub, ok=bool(reached == n_stab),
                         reached=int(reached)))
        print(f"  stability group {s_target:5.3f} (n_sub ends at {r.n_sub:4d}): "
              f"{'ran' if stab[-1]['ok'] else f'diverged at frame {reached}'}", flush=True)

    # ---- artefacts ----------------------------------------------------------------------------
    s_hi = float(np.percentile(np.concatenate([k[2] for k in kept[::4]]), 99))
    render(kept, nom.sheet.Fc.cpu().numpy(), d, name, s_hi)
    metrics_png(nom.res, free.res, rigid.res, remodel, stab, stiff.res, d, frames)

    geo_final = nom.res["R_target"][-1] / nom.res["R_target"][0]
    lag_pred = (nom.res["R_target"][-1] - nom.res["R_target"][0]) / frames / (nom.sheet.M * nom.kappa)
    out = dict(
        run=name, frames=frames, nodes=nom.sheet.n, faces=nom.sheet.m,
        certification=cert,
        drive=dict(source=os.path.basename(TISSUE), scale=SCALE,
                   R_first=nom.res["R_target"][0], R_last=nom.res["R_target"][-1],
                   radius_ratio=geo_final),
        rig=dict(**{k: v for k, v in P.items() if k != "dev"}, mobility=nom.sheet.M,
                 lambda_max_elastic_seeded=nom.lam_el_seed,
                 lambda_max_elastic_final=nom.res["lam_hess"][-1],
                 hessian_growth=nom.res["lam_hess"][-1] / nom.lam_el_seed,
                 substeps_first=nom.res["n_sub"][0], substeps_final=nom.res["n_sub"][-1],
                 stability_group=nom.stability, Y2_2d_modulus=P["E"] * P["thickness"]),
        # 1. THE HEADLINE
        stretch=dict(measured_final=nom.res["lam_geo"][-1], applied_final=geo_final,
                     fidelity=nom.res["lam_geo"][-1] / geo_final,
                     fidelity_mpm_direct_force_run130=0.31 / 2.44 + 1 - 1,
                     lambda_el_final_no_remodel=nom.res["lam_el"][-1],
                     J_final=nom.res["J"][-1], J_expected=geo_final ** 2),
        # 2. THE STANDOFF
        standoff=dict(final=nom.res["standoff"][-1], min=min(nom.res["standoff"]),
                      tracking_lag_prediction=-lag_pred,
                      hoop_excess=nom.res["standoff"][-1] + lag_pred),
        # 3. REMODELLING
        remodelling={k.replace("$", "").replace("\\", ""):
                     dict(lam_el_final=v["lam_el"][-1], lam_geo_final=v["lam_geo"][-1],
                          standoff_final=v["standoff"][-1], energy_final=v["energy"][-1],
                          energy_ratio_vs_elastic=v["energy"][-1] / max(nom.res["energy"][-1], 1e-30))
                     for k, v in remodel.items()},
        # 4. STABILITY
        stability=stab,
        stability_largest_stable=max([r["s"] for r in stab if r["ok"]] or [0.0]),
        stability_smallest_unstable=min([r["s"] for r in stab if not r["ok"]] or [float("inf")]),
        # 5. AREAL DENSITY
        density=dict(rho_final=nom.res["rho"][-1], thinning_factor=1.0 / nom.res["rho"][-1],
                     area_ratio=nom.res["area"][-1] / nom.res["area"][0]),
        # 6. CONTROLS
        controls=dict(kappa0_lambda_final=free.res["lam_geo"][-1],
                      rigid_lambda_final=rigid.res["lam_geo"][-1],
                      rigid_lambda_max=max(rigid.res["lam_geo"])),
        stiffening=dict(beta=5.0, Y_over_Y0_final=stiff.res["Y_over_Y0"][-1],
                        lam_el_final=stiff.res["lam_el"][-1],
                        lam_geo_final=stiff.res["lam_geo"][-1]),
        series={k: [float(x) for x in v] for k, v in nom.res.items()},
    )
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    np.savez_compressed(
        os.path.join(d, "traj.npz"),
        frames=np.asarray([k[0] for k in kept]),
        pos=np.stack([k[1] for k in kept]).astype(np.float32),
        lam=np.stack([k[2] for k in kept]).astype(np.float32),
        R_target=np.stack([k[3] for k in kept]).astype(np.float32),
        faces=nom.sheet.Fc.cpu().numpy().astype(np.int32),
        edges=nom.sheet.Ed.cpu().numpy().astype(np.int32),
        u0=nom.u0.float().cpu().numpy(),
        **{f"series_{k}": np.asarray(v, dtype=np.float32) for k, v in nom.res.items()})
    yaml.safe_dump(dict(
        units=dict(**UNITS, force_nN=None,
                   note="declared per plexus/units.py: length MEASURED from the cache (a cell is "
                        "8.54e-3 box across; assuming 10 um gives 1 box = 1171 um and a 318 um "
                        "spheroid), time from 3.99 doublings over 401 frames at a 12-24 h cycle. "
                        "Force is NOT declared: nothing here fixes a force scale, so only ratios "
                        "are meaningful.",
                   exposes=f"thickness T = 2e-3 box = {2e-3*UNITS['length_um']:.2f} um against a real "
                           f"basement membrane of {T_REAL_UM} um -- {2e-3*UNITS['length_um']/T_REAL_UM:.0f}x "
                           f"too thick; l0 and Sigma are defined in T and are too large by the same "
                           f"factor. The proportions are right, the absolute scale is not."),
        what="05a -- a codimension-1 basement membrane tethered to the recorded epithelial surface",
        question="does a sheet on a mesh carry the deformation the tissue applies to it, which no MPM "
                 "version of it did (run 130: 13%, run 121: 98% at the cost of the standoff)",
        sheet=dict(model="St Venant-Kirchhoff membrane, one constant-strain triangle per face",
                   massless=True, integrator="overdamped explicit Euler, x += dt*M*f",
                   E=P["E"], thickness=P["thickness"], nu=P["nu"],
                   Y2="E*thickness, a force per unit length", subdiv=subdiv,
                   nodes=nom.sheet.n, faces=nom.sheet.m),
        drive=dict(kind="tether to a FROZEN DIRECTION on the recorded surface -- one-sided, returns "
                        "no reaction; this is what 05b replaces with a plaque",
                   surface=os.path.basename(TISSUE), scale=SCALE, kappa=P["kappa"],
                   delta=P["delta"]),
        schedule=dict(frames=frames,
                      substeps_per_frame=f"adaptive, {nom.res['n_sub'][0]} at frame 0 -> "
                                         f"{nom.res['n_sub'][-1]} at the last frame",
                      zeta_efolds_per_frame=P["zeta"], stability_group=nom.stability,
                      bound="dt*M*lambda_max < 2, lambda_max re-measured by power iteration every "
                            "10 frames because the tangent stiffens as the sheet stretches"),
        measures=["stretch fidelity against the drive", "standoff vs the tracking-lag prediction",
                  "lambda_el vs lambda_geo under remodelling", "stability either side of the bound",
                  "areal density with no secretion", "kappa=0 and rigid-drive controls"],
        not_modelled=["plaque and its reaction", "the matrix", "secretion", "proteolysis",
                      "bending stiffness", "bond breaking"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    print(f"[{name}] stretch {out['stretch']['measured_final']:.4f} measured against "
          f"{geo_final:.4f} applied = {out['stretch']['fidelity']:.4f} fidelity; "
          f"standoff {out['standoff']['final']:+.3e} (tracking lag alone predicts "
          f"{out['standoff']['tracking_lag_prediction']:+.3e}); "
          f"controls {out['controls']['kappa0_lambda_final']:.5f} / "
          f"{out['controls']['rigid_lambda_final']:.5f}; thinning "
          f"1/{out['density']['thinning_factor']:.1f} -> {d}", flush=True)


if __name__ == "__main__":
    main()
