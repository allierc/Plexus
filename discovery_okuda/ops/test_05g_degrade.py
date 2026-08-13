#!/usr/bin/env python
"""test_05g_degrade -- proteolysis: two fields, one tethered enzyme, and where a hole gets its size.

    python test_05g_degrade.py [--device cuda:0] [--frames 401]  ->  log/okuda_ECM/05g_degrade/

THE THREE SPECIES, AND WHY ONLY TWO ARE FIELDS (protease_ops.py has the arithmetic):

  MMP      soluble, ~72 kDa, D ~ 10-100 um^2/s. Cuts collagen IV.            FIELD on bm_face
  TIMP     soluble inhibitor, binds MMP 1:1 and inactivates it.              FIELD on bm_face
  MT1-MMP  membrane-type (MMP14): TRANSMEMBRANE, so it never enters the      per-CELL state
           extracellular space and stays where its cell put it.

In one 600 s frame a soluble protease travels sqrt(4Dt) = 155-490 um against a 318 um spheroid: it is
UNIFORM over the whole sheet within a frame, and a uniform protease cannot open a hole anywhere in
particular. Localising to a cell would need D < 0.042 um^2/s, 240-2400x slower than a protein of that
size. That is not a tuning range, it is a different molecule -- and it is MT1-MMP, which is tethered.

So this run has two mechanisms that are deliberately not the same shape, and the point is to see which
one makes a hole:
  * the DIFFUSE arm: cells secrete soluble MMP, TIMP inhibits it, and the balance gives the field a
    length scale sqrt(D/k). Without the sink the field fills the sphere.
  * the TETHERED arm: a subset of cells carry MT1-MMP, which acts only on the faces they touch.

REFERENCES. The species, their roles and the reaction topology are from Karagiannis & Popel (2004)
J. Biol. Chem. 279(37):39105 -- the standard ODE model for MT1-MMP / TIMP-2 / proMMP-2 / MMP-2 --
with the tethered-activator argument from Sato & Takino (2010) Cancer Sci. 101:843. NOTE that this run
uses TIMP as a PURE INHIBITOR, which is only half of what that literature describes: TIMP-2 is also the
bridging adaptor required for activation, so the true dose response is biphasic. This run is therefore
the monotonic CONTROL that 05h's ternary model has to beat.

THE GATES:
  G34  a hole opens only where protease is         zero deaths outside the source's support
  G35  the breach has a SIZE, and it follows sqrt(D/k_eff) rather than filling the sheet
  G36  control, soluble only: NO localisation -- the sheet thins everywhere, which is what the
       arithmetic predicts and is the reason MT1-MMP exists
  G37  control, no protease at all: nothing dies
  G38  the diffusion solve conserves the field (it is a flux form, so it must)
"""
from __future__ import annotations

import json, math, os, sys
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection  # noqa: E402

import bm_ops as BM                                                     # noqa: E402
import ecm_spec as ES                                                   # noqa: E402
from protease_ops import face_laplacian, diffuse_implicit               # noqa: E402
from test_05_sheet import LOG, UNITS                                    # noqa: E402
from test_05f_secrete import Rig05f                                     # noqa: E402
import test_05e_conserve as E5                                          # noqa: E402
from rerender_05 import unroll, attachment_points                       # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

CMAP = ListedColormap(ES.STRESS_COLORS)
UM = 1171.0
EPI_C = "#e8dcc0"


class Rig05g(Rig05f):
    """05f's mass balance, plus two diffusible species and one tethered enzyme."""

    def __init__(self, *a, D_mmp=30.0, D_timp=30.0, k_inhib=200.0, s_mmp=0.0, s_timp=0.0,
                 s_mt1=0.0, mt1_frac=0.03, k_deg=3.0, tau_mmp=8.0, tau_timp=8.0, **kw):
        # the box carries no seconds, so the rates are per FRAME: D in box^2/frame from um^2/s
        self.D_mmp = D_mmp * UNITS["time_s"] / (UM ** 2)
        self.D_timp = D_timp * UNITS["time_s"] / (UM ** 2)
        self.k_inhib, self.k_deg = float(k_inhib), float(k_deg)
        # CLEARANCE, and its absence was a real defect rather than a missing refinement. With a source
        # and no first-order sink, TIMP simply accumulated -- 0.0318 at frame 20, 0.0438 at frame 25,
        # rising without bound -- because its only sink was the 1:1 reaction with an MMP that was
        # 20x smaller. There was no steady state for the pair to reach. And the length scale
        # sqrt(D/k) that justifies solving the field at all IS this k: proteins are cleared, taken up
        # and degraded, and without that term a source fills the sphere no matter what D is. The
        # reaction alone cannot provide it, because a 1:1 reaction is limited by the scarcer species.
        self.tau_mmp, self.tau_timp = float(tau_mmp), float(tau_timp)
        self.s_mmp, self.s_timp, self.s_mt1 = float(s_mmp), float(s_timp), float(s_mt1)
        self.mt1_frac = float(mt1_frac)
        super().__init__(*a, **kw)
        m = self.sheet.F_all.shape[0]
        self.mmp = torch.zeros(m, device=self.dev, dtype=self.dtype)
        self.timp = torch.zeros(m, device=self.dev, dtype=self.dtype)
        # WHICH CELLS EXPRESS MT1-MMP. A fixed random subset, frozen at seeding: the tethered enzyme
        # is what gives a hole a place, so which cells carry it has to be a stated fact of the run
        # and not something that drifts.
        g = torch.Generator(device="cpu").manual_seed(11)
        n_c = self.F_epi.shape[0]
        self.mt1 = torch.zeros(n_c, device=self.dev, dtype=self.dtype)
        pick = torch.randperm(n_c, generator=g)[:max(1, int(self.mt1_frac * n_c))]
        self.mt1[pick.to(self.dev)] = 1.0
        self._lap = None
        for k in ("mmp_total", "timp_total", "mmp_max", "dead_cum", "hole_frac", "field_conserved",
                  "dead_under_mt1", "dead_elsewhere", "mmp_mean", "timp_mean"):
            self.res[k] = []
        self._dead_cum = 0

    def _lapl(self):
        if self._lap is None or self._lap[4] != self.sheet.m:
            i, j, w, a = face_laplacian(self.sheet.x, self.sheet.F_all, self.sheet.live)
            self._lap = (i, j, w, a, self.sheet.m)
        return self._lap

    def _mt1_on_faces(self):
        """The tethered enzyme, mapped from the cells that carry it onto the faces they touch. It
        does NOT diffuse: that is the whole point of it."""
        v = torch.zeros(self.sheet.x.shape[0], device=self.dev, dtype=self.dtype)
        v.index_add_(0, self.ct_node, self.mt1[self.ct_face])
        cnt = torch.zeros_like(v)
        cnt.index_add_(0, self.ct_node, torch.ones_like(self.ct_node, dtype=self.dtype))
        v = v / cnt.clamp_min(1.0)
        return v[self.sheet.Fc].mean(1)

    def fields(self, dt=1.0):
        i, j, w, a, _ = self._lapl()
        mmp = self.mmp[self.sheet.live]
        timp = self.timp[self.sheet.live]
        before = float((mmp * a).sum())
        # THE SOURCES ARE NOT THE SAME SHAPE, and the first version made them identical -- both
        # spatially uniform, both at the same rate, both with the same D, quenching each other 1:1.
        # The two fields then solved the same equation from the same initial condition and came out
        # BIT-IDENTICAL (corr 1.000000, max difference exactly 0), uniform to 2e-16. They would have
        # been uniform at D = 0: nothing ever broke the symmetry, so the run could not test the
        # diffusion argument it was built to test.
        # The biology that breaks it: MT1-MMP is the ACTIVATOR of proMMP-2 at the cell surface, so
        # soluble MMP is produced WHERE THE TETHERED ENZYME IS -- a patchy source -- while TIMP comes
        # from the stroma and arrives everywhere. MMP then has a halo of size sqrt(D/k) around each
        # expressing cell, larger than the cell itself, and THAT is the length the breach inherits.
        src = self._mt1_on_faces()
        src = src / src.mean().clamp_min(1e-30) if float(src.mean()) > 0 else src
        mmp = mmp + dt * self.s_mmp * src
        timp = timp + dt * self.s_timp
        # diffusion, semi-implicit -- explicit would cost 1,300-13,400 substeps and quadruple at
        # every refinement
        mmp_d = diffuse_implicit(mmp, self.D_mmp, dt, i, j, w, a)
        timp_d = diffuse_implicit(timp, self.D_timp, dt, i, j, w, a)
        cons = abs(float((mmp_d * a).sum()) - float((mmp * a).sum())) / max(float((mmp * a).sum()), 1e-30)
        # the reaction that gives the field a length scale: MMP + TIMP -> inactive, 1:1
        r = (self.k_inhib * mmp_d * timp_d * dt).clamp(max=torch.minimum(mmp_d, timp_d))
        # clearance: first order, so each species has a steady state s*tau even with no reaction
        mmp_d = (mmp_d - r) * math.exp(-dt / self.tau_mmp) if self.tau_mmp > 0 else (mmp_d - r)
        timp_d = (timp_d - r) * math.exp(-dt / self.tau_timp) if self.tau_timp > 0 else (timp_d - r)
        self.mmp[self.sheet.live] = mmp_d.clamp_min(0.0)
        self.timp[self.sheet.live] = timp_d.clamp_min(0.0)
        self.res_cons = cons
        return before

    def degrade(self):
        """`bm_degrade`: mass is cut by the ACTIVE soluble MMP and by the tethered enzyme, and a face
        dies when its areal density falls below critical. Two arms, deliberately different in shape."""
        live = self.sheet.live
        active = self.mmp[live] + self.s_mt1 * self._mt1_on_faces()
        self.sheet.mass[live] = (self.sheet.mass[live]
                                 * torch.exp(-self.k_deg * active)).clamp_min(0.0)
        if self.rho_crit <= 0:
            return 0
        rho = self.sheet.areal_density() / self.sheet.rho0
        mask = rho < self.rho_crit
        # where did they die? under a cell that expresses MT1, or away from one -- G34
        mt1f = self._mt1_on_faces()
        self._dead_mt1 = int((mask & (mt1f > 0.05)).sum())
        self._dead_else = int((mask & (mt1f <= 0.05)).sum())
        n = self.sheet.tear(mask)
        self._dead_cum += n
        return n

    def frame(self, t):
        self._dead_mt1 = self._dead_else = 0
        super().frame(t)          # 05f: refine, integrate, secrete, then its own degrade (rho_crit)
        self.fields()
        self.res["mmp_total"].append(float(self.mmp[self.sheet.live].sum()))
        self.res["timp_total"].append(float(self.timp[self.sheet.live].sum()))
        self.res["mmp_max"].append(float(self.mmp[self.sheet.live].max()) if self.sheet.m else 0.0)
        self.res["dead_cum"].append(self._dead_cum)
        self.res["hole_frac"].append(1.0 - self.sheet.m / max(self.res["n_faces"][0], 1))
        self.res["field_conserved"].append(getattr(self, "res_cons", 0.0))
        li = self.sheet.live
        self.res["mmp_mean"].append(float(self.mmp[li].mean()) if self.sheet.m else 0.0)
        self.res["timp_mean"].append(float(self.timp[li].mean()) if self.sheet.m else 0.0)
        self.res["dead_under_mt1"].append(self._dead_mt1)
        self.res["dead_elsewhere"].append(self._dead_else)


# =============================================================================================
def render_2x2(kept, d, name, l0, fps=20):
    """The 2x2: three heatmaps of the three species/quantities, and the section top right.

    WHY THESE FOUR. A run with two fields cannot be read from one picture -- the question is where the
    protease is RELATIVE to where the sheet is dying, and that is two maps and a difference. So: the
    active enzyme, the inhibitor that gives it a length scale, and the material it is eating, with the
    section keeping its job of saying which side of the epithelium the sheet is on. Each panel carries
    its OWN colour scale, printed, because a shared one across three unrelated quantities is a lie.
    """
    fig = plt.figure(figsize=(11.8, 8.4), facecolor="black")
    wri = FFMpegWriter(fps=fps, metadata={"title": name})
    lim, C = 0.165, np.array([0.5, 0.5, 0.5])
    hi = {k: float(np.percentile(np.concatenate([f[k][::5] for f in [x[5] for x in kept[::3]]]), 99))
          for k in ("mmp", "timp", "mt1", "rho")}
    for k in hi:
        hi[k] = hi[k] if hi[k] > 0 else 1.0
    strip, strip_at = [], set(np.round(np.linspace(0, len(kept) - 1, 8)).astype(int).tolist())
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for idx, (t, X, L, F, XE, flds, nod) in enumerate(kept):
            fig.clf()
            fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                                wspace=0.02, hspace=0.05)
            def rgb_panel(pos):
                """ONE PANEL FOR THREE SPECIES, each on its own channel: MMP red, TIMP green,
                MT1-MMP blue, each normalised by its own p99. The colour then says the relationship
                rather than the amount -- yellow is MMP and TIMP together, magenta is MMP over a
                tethered source, white is all three, and a uniformly yellow sphere is the failure this
                run actually shipped (two fields that were bit-identical). A composite makes that
                visible at a glance where three separate greyscales did not."""
                ax = fig.add_subplot(2, 2, pos, projection="3d", facecolor="black",
                                     computed_zorder=False)
                ax.set_facecolor("black"); ax.axis("off")
                kf = X[F][:, :, 1].mean(1) > C[1]
                rgb = np.stack([np.clip(flds["mmp"][kf] / hi["mmp"], 0, 1),
                                np.clip(flds["timp"][kf] / hi["timp"], 0, 1),
                                np.clip(flds["mt1"][kf] / max(hi["mt1"], 1e-30), 0, 1)], 1)
                tri = Poly3DCollection(X[F][kf], linewidths=0.0)
                tri.set_facecolor(np.concatenate([rgb, np.ones((rgb.shape[0], 1))], 1))
                ax.add_collection3d(tri)
                ax.set_xlim(C[0] - lim, C[0] + lim); ax.set_ylim(C[1] - lim, C[1] + lim)
                ax.set_zlim(C[2] - lim, C[2] + lim)
                try:
                    ax.set_box_aspect((1, 1, 1), zoom=2.1)
                except TypeError:
                    ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=16, azim=-58)
                ax.text2D(0.02, 0.99,
                          "the three species, one channel each",
                          transform=ax.transAxes, color="white", fontsize=10, va="top")
                for n, (lab, col, key) in enumerate((("MMP", "#ff5050", "mmp"),
                                                     ("TIMP", "#50ff50", "timp"),
                                                     ("MT1-MMP", "#7090ff", "mt1"))):
                    ax.text2D(0.02, 0.90 - 0.05 * n, f"{lab}  0 to {hi[key]:.3g}",
                              transform=ax.transAxes, color=col, fontsize=9, va="top")
                return ax

            def panel(pos, key, label, cmap=CMAP):
                ax = fig.add_subplot(2, 2, pos, projection="3d", facecolor="black",
                                     computed_zorder=False)
                ax.set_facecolor("black"); ax.axis("off")
                kf = X[F][:, :, 1].mean(1) > C[1]
                tri = Poly3DCollection(X[F][kf], linewidths=0.0)
                v = np.clip(flds[key][kf] / hi[key], 0, 1)
                tri.set_facecolor(cmap(v))
                ax.add_collection3d(tri)
                ax.set_xlim(C[0] - lim, C[0] + lim); ax.set_ylim(C[1] - lim, C[1] + lim)
                ax.set_zlim(C[2] - lim, C[2] + lim)
                try:
                    ax.set_box_aspect((1, 1, 1), zoom=2.1)
                except TypeError:
                    ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=16, azim=-58)
                ax.text2D(0.02, 0.97, f"{label}\n0 to {hi[key]:.3g}", transform=ax.transAxes,
                          color="white", fontsize=10, va="top")
                return ax
            # THE SOURCE AND THE FIELD, not the field twice. The first version drew MMP and TIMP,
            # which were the same array, and omitted the tethered enzyme that was doing all the work.
            rgb_panel(1)
            panel(3, "mmp", "active MMP alone (its own scale)")
            panel(4, "rho", r"$\rho/\rho_0$ -- gaps are dead faces")
            a2 = fig.add_subplot(2, 2, 2, facecolor="black")
            Rc = float(np.linalg.norm(XE - C, axis=1).mean())
            half = max(22.0 * l0, 0.13 * Rc)
            edge = float(np.linalg.norm(X[F[:, 1]] - X[F[:, 0]], axis=1).mean())
            band = 0.60 * edge
            se = (np.abs(XE[:, 1] - C[1]) < band) & (XE[:, 0] > C[0])
            if se.sum() > 2:
                ex, ez = unroll(XE[se]); o = np.argsort(np.arctan2(ez - C[2], ex - C[0]))
                a2.plot(ex[o], ez[o], "-", color=EPI_C, lw=1.6, zorder=2)
            liv = np.zeros(X.shape[0], bool); liv[F.reshape(-1)] = True
            sl = liv & (np.abs(X[:, 1] - C[1]) < band) & (X[:, 0] > C[0])
            if sl.sum() > 2:
                sx, sz = unroll(X[sl]); o2 = np.argsort(np.arctan2(sz - C[2], sx - C[0]))
                a2.plot(sx[o2], sz[o2], "-", color="#9ad2ff", lw=1.2, zorder=5)
                a2.scatter(sx, sz, s=9, c="#9ad2ff", marker="o", linewidths=0, zorder=6)
            gap = float(np.linalg.norm(X[liv] - C, axis=1).mean() - Rc)
            a2.set_xlim(C[0] + Rc - half, C[0] + Rc + half)
            a2.set_ylim(C[2] - half, C[2] + half)
            a2.set_aspect("equal"); a2.axis("off")
            a2.text(0.02, 0.98, f"{name}   frame {t}\nsection, {2*half*UM:.1f} um across\n"
                                f"{F.shape[0]} live faces\n"
                                f"sheet is {'outside' if gap > 0 else 'inside'} by "
                                f"{abs(gap)*UM:.3f} um",
                    transform=a2.transAxes, color="white", fontsize=9.5, va="top")
            wri.grab_frame()
            if idx in strip_at:
                strip.append((t, X.copy(), F.copy(), flds["rho"].copy()))
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)
    figs = plt.figure(figsize=(3.0 * len(strip), 3.3), facecolor="black")
    for i, (t, X, F, rho) in enumerate(strip):
        a = figs.add_subplot(1, len(strip), i + 1, facecolor="black")
        liv = np.zeros(X.shape[0], bool); liv[F.reshape(-1)] = True
        sl = liv & (np.abs(X[:, 1] - 0.5) < 0.004)
        a.scatter(X[sl][:, 0], X[sl][:, 2], s=6, c="#9ad2ff", marker=".", linewidths=0)
        a.set_xlim(0.335, 0.665); a.set_ylim(0.335, 0.665); a.set_aspect("equal"); a.axis("off")
        a.text(0.03, 0.97, f"frame {t}\n{F.shape[0]} faces", transform=a.transAxes, color="white",
               fontsize=10, va="top")
    figs.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02)
    figs.savefig(os.path.join(d, "strip.png"), dpi=110, facecolor="black")
    plt.close(figs)


def run_g(rig, frames, keep, label):
    kept = []
    for t in range(frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[{label}] stopped at frame {t} ({rig.sheet.m} faces left)", flush=True)
            return kept, t
        if t in keep:
            l1, _ = rig.sheet.stretch_geo()
            flds = dict(mmp=rig.mmp[rig.sheet.live].float().cpu().numpy(),
                        timp=rig.timp[rig.sheet.live].float().cpu().numpy(),
                        mt1=rig._mt1_on_faces().float().cpu().numpy(),
                        rho=(rig.sheet.areal_density() / rig.sheet.rho0).float().cpu().numpy())
            kept.append((t, rig.sheet.x.float().cpu().numpy(), l1.float().cpu().numpy(),
                         rig.sheet.Fc.cpu().numpy(), rig.x_epi.float().cpu().numpy(), flds,
                         rig.ct_node.cpu().numpy()))
    print(f"[{label}] {frames} frames -- {rig.sheet.m} faces, {rig._dead_cum} dead, "
          f"MMP max {rig.res['mmp_max'][-1]:.3g}, hole {100*rig.res['hole_frac'][-1]:.1f}%",
          flush=True)
    return kept, frames


def main():
    def arg(f, c, dflt):
        return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else dflt
    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 200)
    name = arg("--name", str, "05g_degrade")
    ratio = arg("--ratio", float, 1.0)      # s_mmp : s_timp, the invasion switch
    d = os.path.join(LOG, name); os.makedirs(d, exist_ok=True)
    cert = BM.selftest(dev=dev, subdiv=4)

    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, sigma_T=7.0, zeta=20.0, s_target=1.0,
             k_drive=50.0, dev=dev)
    A = dict(kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3)
    S = dict(s_mode="homeostatic", tau_bm=40.0, rho_crit=0.35, max_refine=0, reseed=False)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 140))).astype(int).tolist())

    runs, kept_nom = {}, None
    sM, sT = 2.0e-3 * ratio, 2.0e-3
    plan = [("both arms", dict(s_mmp=sM, s_timp=sT, s_mt1=0.35)),
            ("soluble only", dict(s_mmp=sM, s_timp=sT, s_mt1=0.0)),
            ("tethered only", dict(s_mmp=0.0, s_timp=sT, s_mt1=0.35)),
            ("no protease", dict(s_mmp=0.0, s_timp=sT, s_mt1=0.0)),
            ("no inhibitor", dict(s_mmp=sM, s_timp=0.0, s_mt1=0.35))]
    for lab, extra in plan:
        r = Rig05g(**P, **A, **S, **extra)
        kp = keep if lab == "both arms" else set()
        kept, _ = run_g(r, frames, kp, f"{name}: {lab}")
        runs[lab] = r.res
        if lab == "both arms":
            kept_nom = kept
            render_2x2(kept, d, f"{name}: both arms", r.l0)

    def steady(v):
        n = max(4, len(v) // 4)
        a, b = float(np.mean(v[-2 * n:-n])), float(np.mean(v[-n:]))
        return abs(b - a) / max(abs(b), 1e-30)
    out = dict(run=name, frames=frames, source_ratio=ratio, certification=cert,
               G53={k: dict(mmp_drift=steady(v["mmp_mean"]), timp_drift=steady(v["timp_mean"]),
                            mmp_final=v["mmp_mean"][-1], timp_final=v["timp_mean"][-1],
                            steady=bool(steady(v["mmp_mean"]) < 0.01
                                        and steady(v["timp_mean"]) < 0.01))
                    for k, v in runs.items()},
               rig=dict(**{k: v for k, v in P.items() if k != "dev"}, **A, **S),
               species=dict(MMP="soluble, a FIELD on bm_face, D ~ 30 um^2/s",
                            TIMP="soluble inhibitor, a FIELD on bm_face",
                            MT1_MMP="membrane-tethered, a per-CELL state, NO field"),
               G34={k: dict(dead_under_mt1=int(np.sum(v["dead_under_mt1"])),
                            dead_elsewhere=int(np.sum(v["dead_elsewhere"])),
                            dead_total=int(v["dead_cum"][-1]),
                            hole_frac=v["hole_frac"][-1]) for k, v in runs.items()},
               G38=dict(field_conservation_max=float(max(max(v["field_conserved"])
                                                         for v in runs.values()))),
               series={k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                       for k, v in runs.items()})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    yaml.safe_dump(dict(
        what="05g -- proteolysis: two diffusible species, one tethered enzyme",
        units=dict(**UNITS, force_nN=None),
        species=dict(
            MMP="soluble matrix metalloproteinase (MMP-2/9). D ~ 10-100 um^2/s -> travels 155-490 um "
                "per 600 s frame against a 318 um spheroid, so it is UNIFORM unless a sink gives it a "
                "length scale. A FIELD on bm_face.",
            TIMP="tissue inhibitor. Binds MMP 1:1. The sink that gives the field its length scale "
                 "sqrt(D/k). A FIELD on bm_face.",
            MT1_MMP="membrane-type MMP14, TRANSMEMBRANE: never enters the extracellular space, stays "
                    "where its cell put it. A per-CELL state. This is what localises a breach, and "
                    "the arithmetic says nothing soluble can."),
        solve="semi-implicit (I + dt D L)c = c_old by CG on a finite-volume face Laplacian. Explicit "
              "would need 1,300-13,400 substeps/frame and QUADRUPLE at every 1->4 refinement.",
        gates=dict(G34="a hole opens only where protease is", G35="the breach has a size ~ sqrt(D/k)",
                   G36="soluble-only does NOT localise -- the prediction that motivates MT1-MMP",
                   G37="no protease, nothing dies", G38="the diffusion solve conserves the field")),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    for k, v in out["G34"].items():
        print(f"[{name}] {k:15s} dead {v['dead_total']:6d}  under MT1 {v['dead_under_mt1']:6d}  "
              f"elsewhere {v['dead_elsewhere']:6d}  hole {100*v['hole_frac']:.1f}%", flush=True)
    print(f"[{name}] G38 field conservation {out['G38']['field_conservation_max']:.2e} -> {d}",
          flush=True)


if __name__ == "__main__":
    main()
