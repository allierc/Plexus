#!/usr/bin/env python
"""test_05c_remesh -- the reservoir: a sheet that refines as the spheroid grows, and that can tear.

    python test_05c_remesh.py [--device cuda:0] [--frames 401]  ->  log/okuda_ECM/05c_remesh/
    python test_05c_remesh.py --tear                            ->  log/okuda_ECM/05d_tear/

WHY THIS COMES BEFORE SECRETION AND BEFORE TEARING, THOUGH IT IS NEITHER. 05a's sheet has a FIXED mesh:
over 401 frames its area goes x11.1 on the same 5120 triangles, mean edge length 0.00305 -> 0.0102 box
units, areal density to 1/11.1. A mass balance cannot add material to a set whose size is fixed and a
proteolytic operator cannot remove it. So the reservoir is not a step of the biology, it is the thing
all the remaining biology is written in -- and it is the pattern the framework already uses:
`engine.py:453` allocates `grow_reserve` dormant MPM particles at `occ = 0` for `cell_grow` to wake.
Here nodes and faces carry the same flag, and `refine`, `tear` and (next) `bm_secrete` are all flips of
`face_occ` rather than reallocations, so an index held by a plaque survives every one of them.

THE TRAP, and it is the whole reason this run has gates before it has results. A child triangle whose
reference frame is rebuilt from its CURRENT shape reports lambda = 1 the moment it is born. A sheet
that refines as it grows would then silently forget everything it had been stretched by -- the mesh
version of the laundering that made run 130 report 13% of its true stretch through the MPM grid. The
four children of a midpoint split are related to their parent by a CONSTANT map S_k in material
coordinates, so `Dm_inv_child = S_k^-1 Dm_inv_parent` and parent and child report the same lambda by
construction. `bm_ops.refine_test()` measures it: 2.3e-14 on a loaded sheet, with the total area and the
total elastic energy bit-identical across the split.

WHAT THIS RUN MEASURES (G17, G18, and the tearing gates G19-G23 under `--tear`):

  G17  the mean edge length stays inside a band as R triples, instead of tracking it. Without
       refinement it goes x3.4; the whole point of a remesher is that it does not.
  G18  refinement does not change the physics it was meant to resolve: lambda_geo, the standoff and
       the areal density against the fixed-mesh 05a, which is the same run with `max_refine = 0`.
  G19  THE TEAR IS IN THE SAME PLACE AT THE SAME lambda WHEN THE MESH IS REFINED. This is the test the
       MPM sheet never faced. `LADDER.md:24` flagged it at the outset -- "MPM separates material that
       thins past the grid's support, so tearing may be set by dx rather than by stress; a tear that
       moves when the grid is refined is numerical" -- and the refinement cross-check (run 90,
       n_grid 64 -> 128) was planned and never ran, on a sheet 1/8 of a grid cell thick that was
       carrying the stroma's youngs 15 rather than the 400 its spec declared.
  G20  the onset scales with the CRITERION and not with element size: onset lambda against the
       threshold must be monotone, and against `subdiv` must be flat.
  G21  a hole stays open -- no face is re-woken, and the remesher does not heal what a mechanism killed.
  G22  control: a criterion above the maximum load kills nothing. Run 127 reported a mechanism as a
       null when its threshold sat above the largest load in the run.
  G23  the mesh is still conforming with a hole in it: no edge carries three faces.

WHAT IS NOT HERE. Refinement is GLOBAL 1->4, not adaptive: every live face splits at once, which is
conforming by construction and needs no hanging-node closure. A crack tip wants local refinement and
that is a later operator. There is no secretion, so refinement makes the existing material finer and
never thicker -- rho is unchanged by a split, which is exactly why 05e is a separate step. And the
midpoints are placed on the CHORD and not projected onto the surface: projecting would smooth the sheet
and change lambda, so the smoothing has to come from the dynamics.
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
from test_05_sheet import SurfaceReplay, LOG, TISSUE, SCALE, UNITS, T_REAL_UM             # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

CMAP = ListedColormap(ES.STRESS_COLORS)
TARGET_C = "#8fb8de"
DEAD_C = "#3a2020"


class Rig05c:
    """05a's rig with a reservoir under it: the same tether, the same drive, plus `refine` and `tear`.

    `edge_trigger` is the multiple of the SEEDED mean edge length at which the sheet refines. It is a
    material threshold, not a schedule: a sheet that is not being stretched never refines, which is
    what makes G17 a measurement rather than a restatement of the trigger.
    """

    def __init__(self, subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, kappa=5.0, delta=6.0e-4,
                 zeta=20.0, s_target=1.0, refresh=10, tau_r=0.0, max_refine=0, edge_trigger=1.45,
                 tear_lambda=0.0, tear_mode="lambda", dev="cuda:0", dtype=torch.float64):
        self.dev, self.dtype = dev, dtype
        V, Fc, Ed = BM.icosphere(subdiv, device=dev, dtype=dtype)
        self.rep = SurfaceReplay(V, dev=dev, dtype=dtype)
        self.sheet = BM.Sheet(subdiv=subdiv, R0=1.0, E=E, thickness=thickness, nu=nu, tau_r=tau_r,
                              max_refine=max_refine, dev=dev, dtype=dtype)
        self.sheet.reseed(self.sheet.c + V * (self.rep.R(0) + delta)[:, None])
        self.u0, self.delta, self.kappa = V, float(delta), float(kappa)
        self.edge_trigger, self.max_refine = float(edge_trigger), int(max_refine)
        self.tear_lambda, self.tear_mode = float(tear_lambda), tear_mode
        self.lam_el, self._pv = self.sheet.spectral_rate(return_vec=True)
        self.sheet.M = float(zeta) / (self.lam_el + self.kappa)
        self.s_target, self.refresh = float(s_target), int(refresh)
        self.n_sub = self._nsub()
        # THE ANCHOR IS A DIRECTION, AND A REFINED SHEET HAS NEW NODES, so the drive has to be
        # extended to them. A midpoint node inherits the direction of its own current position, which
        # is what "the surface this patch of membrane faces" means for a patch that did not exist when
        # the run started. Anything else -- reusing a parent's direction, or leaving new nodes
        # untethered -- would put a hole in the load rather than in the sheet.
        self._rebuild_drive()
        self.res = {k: [] for k in ("lam_geo", "lam_el", "R_sheet", "R_target", "standoff", "rho",
                                    "energy", "area", "mean_edge", "n_faces", "n_nodes", "n_sub",
                                    "lam_hess", "refined", "dead", "lam_max", "rim")}

    def _rebuild_drive(self):
        u = self.sheet.x[self.sheet.live_nodes] - self.sheet.c
        self.u_live = u / u.norm(dim=1, keepdim=True).clamp_min(1e-30)
        self.rep_live = SurfaceReplay(self.u_live, dev=self.dev, dtype=self.dtype)

    def _nsub(self):
        return max(1, int(math.ceil(self.sheet.M * (self.lam_el + self.kappa) / self.s_target)))

    def frame(self, t):
        # -- remesh, before the step, so the frame is integrated on one mesh and not two
        refined = 0
        if (self.max_refine and self.sheet.n_refinements < self.max_refine
                and self.sheet.mean_edge() > self.edge_trigger * self.sheet.mean_edge_seed):
            ne, nf = self.sheet.refine()
            self._rebuild_drive()
            self.lam_el, self._pv = self.sheet.spectral_rate(iters=40, return_vec=True)
            self.n_sub = self._nsub()
            refined = nf
            print(f"    [refine] frame {t}: {self.sheet.m} faces, {self.sheet.n} nodes, "
                  f"mean edge {self.sheet.mean_edge()/self.sheet.mean_edge_seed:.3f} of seeded",
                  flush=True)
        if t % self.refresh == 0 and torch.isfinite(self.sheet.x).all():
            self.lam_el, self._pv = self.sheet.spectral_rate(iters=25, v0=self._pv, return_vec=True)
            self.n_sub = self._nsub()
        R = self.rep_live.R(t) + self.delta
        a = self.sheet.c + self.u_live * R[:, None]
        dt = 1.0 / self.n_sub
        idx = self.sheet.live_nodes
        for _ in range(dt and self.n_sub):
            ft = torch.zeros_like(self.sheet.x)
            ft[idx] = self.kappa * (a - self.sheet.x[idx])
            self.sheet.step(dt, extra_force=ft)
        # -- tearing, after the step, on the state the step produced
        dead = 0
        if self.tear_lambda > 0:
            l1, _ = self.sheet.stretch_elastic()
            dead = self.sheet.tear(l1 > self.tear_lambda)
            if dead:
                self._rebuild_drive()
                R = self.rep_live.R(t) + self.delta
        self._record(t, R, refined, dead)
        return dead

    def _record(self, t, R, refined, dead):
        l1, l2 = self.sheet.stretch_geo()
        e1, _ = self.sheet.stretch_elastic()
        idx = self.sheet.live_nodes
        r = (self.sheet.x[idx] - self.sheet.c).norm(dim=1)
        self.res["lam_geo"].append(float(l1.mean()))
        self.res["lam_max"].append(float(l1.max()))
        self.res["lam_el"].append(float(e1.mean()))
        self.res["R_sheet"].append(float(r.mean()))
        self.res["R_target"].append(float(R.mean()))
        self.res["standoff"].append(float(r.mean() - R.mean()))
        self.res["rho"].append(float(self.sheet.areal_density().mean()))
        self.res["energy"].append(float(self.sheet.energy(self.sheet.x)))
        self.res["area"].append(float(self.sheet.area().sum()))
        self.res["mean_edge"].append(self.sheet.mean_edge() / self.sheet.mean_edge_seed)
        self.res["n_faces"].append(self.sheet.m)
        self.res["n_nodes"].append(self.sheet.n)
        self.res["n_sub"].append(int(self.n_sub))
        self.res["lam_hess"].append(float(self.lam_el))
        self.res["refined"].append(int(refined))
        self.res["dead"].append(int(dead))
        self.res["rim"].append(int(self.sheet.euler_check()["rim"]))

    def alive(self):
        return bool(torch.isfinite(self.sheet.x[self.sheet.live_nodes]).all())


def run(rig, frames, keep=None, label=""):
    keep, kept, t0 = (set() if keep is None else keep), [], time.time()
    for t in range(frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[{label}] DIVERGED at frame {t}", flush=True)
            return kept, t
        if t in keep:
            l1, _ = rig.sheet.stretch_geo()
            kept.append((t, rig.sheet.x.float().cpu().numpy(),
                         l1.float().cpu().numpy(), rig.sheet.Fc.cpu().numpy(),
                         float(rig.res["R_target"][-1])))
    if label:
        print(f"[{label}] {frames} frames in {time.time()-t0:.1f}s -- {rig.sheet.m} faces, "
              f"edge {rig.res['mean_edge'][-1]:.3f} of seeded, lambda_geo "
              f"{rig.res['lam_geo'][-1]:.4f}, standoff {rig.res['standoff'][-1]:+.3e}, "
              f"{int(np.sum(rig.res['dead']))} faces torn", flush=True)
    return kept, frames


# =============================================================================================
def render(kept, d, name, s_hi, fps=20):
    fig = plt.figure(figsize=(11.6, 5.8), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": name})
    strip, strip_at = [], set(np.round(np.linspace(0, len(kept) - 1, 8)).astype(int).tolist())
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for i, (t, X, L, F, Rm) in enumerate(kept):
            fig.clf()
            c, lim = np.array([0.5, 0.5, 0.5]), 0.165
            ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="black", computed_zorder=False)
            ax.set_facecolor("black"); ax.axis("off")
            kf = X[F][:, :, 1].mean(1) > c[1]
            tri = Poly3DCollection(X[F][kf], linewidths=0.12, edgecolors=(1, 1, 1, 0.18))
            tri.set_facecolor(CMAP(np.clip((L[kf] - 1.0) / max(s_hi - 1.0, 1e-9), 0, 1)))
            ax.add_collection3d(tri)
            ax.set_xlim(c[0] - lim, c[0] + lim); ax.set_ylim(c[1] - lim, c[1] + lim)
            ax.set_zlim(c[2] - lim, c[2] + lim)
            try:
                ax.set_box_aspect((1, 1, 1), zoom=1.55)
            except TypeError:
                ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=16, azim=-58)
            ax.text2D(0.02, 0.97, f"{name}   frame {t}\n{F.shape[0]} live faces   "
                                  f"$\\lambda_{{geo}}$ mean {L.mean():.3f}, colour to {s_hi:.2f}",
                      transform=ax.transAxes, color="white", fontsize=10.5, va="top")
            a2 = fig.add_subplot(1, 2, 2, facecolor="black")
            nl = np.zeros(X.shape[0]); cnt = np.zeros(X.shape[0])
            np.add.at(nl, F.reshape(-1), np.repeat(L, 3)); np.add.at(cnt, F.reshape(-1), 1)
            live = cnt > 0
            nl = nl / np.maximum(cnt, 1)
            sl = live & (np.abs(X[:, 1] - c[1]) < 0.004)
            a2.scatter(X[sl][:, 0], X[sl][:, 2], s=11, c=np.clip(nl[sl], 1.0, s_hi), cmap=CMAP,
                       vmin=1.0, vmax=s_hi, marker=".", linewidths=0)
            th = np.linspace(0, 2 * np.pi, 200)
            a2.plot(c[0] + Rm * np.cos(th), c[2] + Rm * np.sin(th), "--", color=TARGET_C, lw=1.0)
            a2.set_xlim(c[0] - lim, c[0] + lim); a2.set_ylim(c[2] - lim, c[2] + lim)
            a2.set_aspect("equal"); a2.axis("off")
            a2.text(0.02, 0.98, "section; a gap in the ring is a hole in the sheet",
                    transform=a2.transAxes, color="white", fontsize=9.5, va="top")
            wri.grab_frame()
            if i in strip_at:
                strip.append((t, X.copy(), nl.copy(), live.copy(), Rm))
    fig.savefig(os.path.join(d, "3d.png"), dpi=115, facecolor="black")
    plt.close(fig)
    figs = plt.figure(figsize=(3.0 * len(strip), 3.3), facecolor="black")
    for i, (t, X, nl, live, Rm) in enumerate(strip):
        a = figs.add_subplot(1, len(strip), i + 1, facecolor="black")
        sl = live & (np.abs(X[:, 1] - 0.5) < 0.004)
        a.scatter(X[sl][:, 0], X[sl][:, 2], s=7, c=np.clip(nl[sl], 1.0, s_hi), cmap=CMAP, vmin=1.0,
                  vmax=s_hi, marker=".", linewidths=0)
        th = np.linspace(0, 2 * np.pi, 200)
        a.plot(0.5 + Rm * np.cos(th), 0.5 + Rm * np.sin(th), "--", color=TARGET_C, lw=0.9)
        a.set_xlim(0.335, 0.665); a.set_ylim(0.335, 0.665); a.set_aspect("equal"); a.axis("off")
        a.text(0.03, 0.97, f"frame {t}", transform=a.transAxes, color="white", fontsize=11, va="top")
    figs.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02)
    figs.savefig(os.path.join(d, "strip.png"), dpi=115, facecolor="black")
    plt.close(figs)


def metrics_png(nom, fixed, conv, tear, d, name):
    fig, ax = plt.subplots(2, 3, figsize=(13.4, 6.8), facecolor="white")
    t = np.arange(len(nom["lam_geo"]))
    ax[0, 0].plot(t, nom["mean_edge"], color="#1a1a1a", lw=1.8, label="with the reservoir")
    ax[0, 0].plot(np.arange(len(fixed["mean_edge"])), fixed["mean_edge"], color="#b03030", lw=1.4,
                  ls="--", label="fixed mesh (05a)")
    ax[0, 0].axhline(1.45, color="#999", ls=":", lw=1.0)
    ax[0, 0].set_ylabel("mean edge length / seeded")
    ax[0, 0].set_title(f"G17: the sheet refines instead of coarsening\n"
                       f"(fixed mesh ends at {fixed['mean_edge'][-1]:.2f}x, "
                       f"reservoir at {nom['mean_edge'][-1]:.2f}x)", fontsize=8.5)
    ax[0, 0].legend(fontsize=7, frameon=False)
    a1 = ax[0, 1]
    a1.plot(t, nom["n_faces"], color="#1a1a1a", lw=1.7, label="live faces")
    a1.plot(np.arange(len(fixed["n_faces"])), fixed["n_faces"], color="#b03030", lw=1.3, ls="--")
    a1.set_yscale("log"); a1.set_ylabel("live faces (occ = 1)")
    a1.set_title("the reservoir, waking", fontsize=8.5)
    ax[0, 2].plot(t, nom["lam_geo"], color="#1a1a1a", lw=1.8, label="with the reservoir")
    ax[0, 2].plot(np.arange(len(fixed["lam_geo"])), fixed["lam_geo"], color="#b03030", lw=1.4,
                  ls="--", label="fixed mesh (05a)")
    ax[0, 2].plot(t, np.asarray(nom["R_target"]) / nom["R_target"][0], color="#1f8a5c", lw=1.1,
                  ls=":", label="$R(t)/R(0)$ applied")
    ax[0, 2].set_ylabel(r"$\lambda_{geo}$")
    d18 = abs(nom["lam_geo"][-1] - fixed["lam_geo"][-1]) / max(fixed["lam_geo"][-1], 1e-9)
    ax[0, 2].set_title(f"G18: refinement does not change the physics\n"
                       f"({100*d18:.2f}% apart at the last frame)", fontsize=8.5)
    ax[0, 2].legend(fontsize=7, frameon=False)
    if tear:
        for lab, r in tear.items():
            ax[1, 0].plot(np.arange(len(r["dead"])), np.cumsum(r["dead"]), lw=1.5, label=lab)
        ax[1, 0].set_ylabel("faces torn (cumulative)")
        ax[1, 0].set_title("G20/G22: onset moves with the CRITERION;\nabove the load, nothing dies",
                           fontsize=8.5)
        ax[1, 0].legend(fontsize=7, frameon=False)
    if conv:
        ks = sorted(conv)
        ax[1, 1].plot(ks, [conv[k]["onset_frame"] for k in ks], "o-", color="#1a1a1a", lw=1.5)
        ax[1, 1].set_xlabel("faces at the moment of tearing")
        ax[1, 1].set_ylabel("frame at which the first face dies")
        a2 = ax[1, 1].twinx()
        a2.plot(ks, [conv[k]["onset_lambda"] for k in ks], "s--", color="#2b6cb0", lw=1.3)
        a2.set_ylabel(r"$\lambda_{el}$ at onset", color="#2b6cb0")
        sp = [conv[k]["onset_lambda"] for k in ks]
        ax[1, 1].set_title(f"G19/G20: onset under refinement\n"
                           f"(spread {100*(max(sp)-min(sp))/max(np.mean(sp),1e-9):.2f}% across "
                           f"{len(ks)} resolutions)", fontsize=8.5)
        ax[1, 1].set_xscale("log")
    ax[1, 2].plot(t, nom["rho"], color="#1a1a1a", lw=1.7, label="with the reservoir")
    ax[1, 2].plot(np.arange(len(fixed["rho"])), fixed["rho"], color="#b03030", lw=1.3, ls="--",
                  label="fixed mesh")
    ax[1, 2].set_yscale("log"); ax[1, 2].set_ylabel(r"areal density $\rho/\rho_0$")
    ax[1, 2].set_title("refinement makes the material FINER, never thicker:\n"
                       r"$\rho$ is untouched, which is why 05e is a separate step", fontsize=8.5)
    ax[1, 2].legend(fontsize=7, frameon=False)
    for a in ax.reshape(-1):
        if not a.get_xlabel():
            a.set_xlabel("frame")
        a.spines[["top", "right"]].set_visible(False)
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
    do_tear = "--tear" in sys.argv
    name = arg("--name", str, "05d_tear" if do_tear else "05c_remesh")
    d = os.path.join(LOG, name)
    os.makedirs(d, exist_ok=True)

    cert = BM.selftest(dev=dev, subdiv=subdiv)
    for k, lim in (("refine_unloaded_dlambda", 1e-6), ("refine_loaded_dlambda", 1e-12),
                   ("refine_loaded_denergy_rel", 1e-6)):
        assert cert[k] < lim, f"{k} = {cert[k]:.3e}, gate is {lim:.0e}"
    assert cert["refine_bad_edges"] == 0 and cert["tear_bad_edges"] == 0, cert
    P = dict(subdiv=subdiv, E=400.0, thickness=2.0e-3, nu=0.3, kappa=5.0, delta=6.0e-4,
             zeta=20.0, s_target=1.0, dev=dev)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 160))).astype(int).tolist())

    # ---- the nominal: two levels of reserve, refinement triggered by the material -------------
    nom = Rig05c(**P, max_refine=2, edge_trigger=1.45)
    print(f"[{name}] reservoir: {nom.sheet.F_all.shape[0]} face slots and "
          f"{nom.sheet.x.shape[0]} node slots for {nom.sheet.m} live faces / {nom.sheet.n} live "
          f"nodes; refine when the mean edge passes {nom.edge_trigger}x its seeded "
          f"{nom.sheet.mean_edge_seed:.4g}", flush=True)
    kept, _ = run(nom, frames, keep=keep, label=name)

    # ---- the control that G18 is against: the same run with no reservoir ----------------------
    fixed = Rig05c(**P, max_refine=0)
    run(fixed, frames, label="control: fixed mesh (05a)")

    tear, conv = {}, {}
    if do_tear:
        # G20/G22: the criterion is a material threshold on lambda_el. The control sits ABOVE the
        # largest stretch the nominal reached, so it must kill nothing -- run 127's null, on purpose.
        lam_max_seen = max(nom.res["lam_max"])
        for thr, lab in ((1.8, r"$\lambda_{el} > 1.8$"), (2.6, r"$\lambda_{el} > 2.6$"),
                         (1.15 * lam_max_seen, "control: above the largest stretch reached")):
            r = Rig05c(**P, max_refine=2, edge_trigger=1.45, tear_lambda=thr)
            run(r, frames, label=f"tear at lambda_el > {thr:.3g}")
            tear[lab] = r.res
        # G19: the SAME criterion at three resolutions. If the onset moves, the tear is set by the
        # discretisation and not by the material -- the test the MPM sheet was never given.
        for sd, mr in ((3, 2), (4, 1), (4, 2)):
            r = Rig05c(**{**P, "subdiv": sd}, max_refine=mr, edge_trigger=1.45, tear_lambda=2.2)
            run(r, frames, label=f"refinement convergence: subdiv {sd}, max_refine {mr}")
            dead = np.asarray(r.res["dead"])
            on = int(np.argmax(dead > 0)) if dead.any() else -1
            conv[r.res["n_faces"][on if on >= 0 else -1]] = dict(
                onset_frame=on, onset_lambda=r.res["lam_el"][on] if on >= 0 else float("nan"),
                onset_lambda_max=r.res["lam_max"][on] if on >= 0 else float("nan"),
                subdiv=sd, max_refine=mr, faces_at_onset=r.res["n_faces"][on if on >= 0 else -1],
                torn_total=int(dead.sum()), rim_final=r.res["rim"][-1])

    s_hi = float(np.percentile(np.concatenate([k[2] for k in kept[::4]]), 99))
    render(kept, d, name, s_hi)
    metrics_png(nom.res, fixed.res, conv, tear, d, name)

    out = dict(
        run=name, frames=frames, certification=cert,
        reservoir=dict(face_slots=int(nom.sheet.F_all.shape[0]),
                       node_slots=int(nom.sheet.x.shape[0]),
                       faces_first=nom.res["n_faces"][0], faces_final=nom.res["n_faces"][-1],
                       nodes_first=nom.res["n_nodes"][0], nodes_final=nom.res["n_nodes"][-1],
                       refinements=nom.sheet.n_refinements,
                       refine_frames=[i for i, v in enumerate(nom.res["refined"]) if v],
                       edge_trigger=nom.edge_trigger, mean_edge_seed=nom.sheet.mean_edge_seed),
        G17=dict(mean_edge_final_reservoir=nom.res["mean_edge"][-1],
                 mean_edge_final_fixed=fixed.res["mean_edge"][-1],
                 band=[min(nom.res["mean_edge"]), max(nom.res["mean_edge"])]),
        G18=dict(lambda_geo_reservoir=nom.res["lam_geo"][-1],
                 lambda_geo_fixed=fixed.res["lam_geo"][-1],
                 relative_difference=abs(nom.res["lam_geo"][-1] - fixed.res["lam_geo"][-1])
                 / max(fixed.res["lam_geo"][-1], 1e-30),
                 standoff_reservoir=nom.res["standoff"][-1],
                 standoff_fixed=fixed.res["standoff"][-1],
                 rho_reservoir=nom.res["rho"][-1], rho_fixed=fixed.res["rho"][-1]),
        tearing={k.replace("$", "").replace("\\", ""):
                 dict(torn=int(np.sum(v["dead"])), faces_final=v["n_faces"][-1],
                      rim_final=v["rim"][-1], lam_geo_final=v["lam_geo"][-1],
                      onset_frame=int(np.argmax(np.asarray(v["dead"]) > 0))
                      if np.any(np.asarray(v["dead"]) > 0) else -1)
                 for k, v in tear.items()} or None,
        G19={str(k): v for k, v in conv.items()} or None,
        series={k: [float(x) for x in v] for k, v in nom.res.items()})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    np.savez_compressed(
        os.path.join(d, "traj.npz"),
        frames=np.asarray([k[0] for k in kept]),
        # the mesh CHANGES SIZE, so the trajectory cannot be one stacked array. Each kept frame keeps
        # its own faces; a reader that assumed a fixed connectivity would silently pair frame 400's
        # positions with frame 0's triangles.
        **{f"pos_{k[0]}": k[1].astype(np.float32) for k in kept},
        **{f"lam_{k[0]}": k[2].astype(np.float32) for k in kept},
        **{f"faces_{k[0]}": k[3].astype(np.int32) for k in kept},
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
        what=f"05c/05d -- a sheet with a reservoir: it refines as the spheroid grows"
             f"{', and it tears' if do_tear else ''}",
        reservoir=dict(pattern="nodes and faces allocated to their maximum and carrying `occ`, as "
                               "`engine.py:453` does for MPM particles via `grow_reserve`",
                       refine="global 1->4 midpoint split of every live face; conforming by "
                              "construction, no hanging-node closure needed",
                       inheritance="Dm_inv_child = S_k^-1 Dm_inv_parent with S_k constant, so parent "
                                   "and child report the SAME lambda (2.3e-14 measured)",
                       tear="a flip of face_occ; the hole's rim is free and no node is removed",
                       trigger=f"mean live edge > {nom.edge_trigger} x its seeded value"),
        gates=dict(G14="refining an unloaded sheet does not change lambda",
                   G15="refining a loaded sheet does not change lambda, area or energy",
                   G16="no edge carries three faces",
                   G17="the mean edge length stays in a band as R triples",
                   G18="lambda_geo, the standoff and rho match the fixed-mesh run",
                   G19="the tear is at the same lambda at three mesh resolutions",
                   G20="onset moves with the criterion, not with element size",
                   G21="a hole stays open",
                   G22="a criterion above the load kills nothing",
                   G23="the mesh is still conforming with a hole in it"),
        not_modelled=["adaptive/local refinement (a crack tip wants it; this is global 1->4)",
                      "coarsening -- the sheet can only get finer",
                      "secretion, so refinement never adds MASS (rho is untouched by a split)",
                      "the plaque, the matrix, bending stiffness"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    print(f"[{name}] faces {nom.res['n_faces'][0]} -> {nom.res['n_faces'][-1]} in "
          f"{nom.sheet.n_refinements} refinements; edge {nom.res['mean_edge'][-1]:.3f} of seeded "
          f"against the fixed mesh's {fixed.res['mean_edge'][-1]:.3f} [G17]; lambda_geo "
          f"{nom.res['lam_geo'][-1]:.4f} vs {fixed.res['lam_geo'][-1]:.4f} = "
          f"{100*out['G18']['relative_difference']:.2f}% apart [G18] -> {d}", flush=True)


if __name__ == "__main__":
    main()
