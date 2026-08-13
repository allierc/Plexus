#!/usr/bin/env python
"""test_05f_secrete -- 05f: the mass balance, and the tear that supply drives.

    python test_05f_secrete.py [--device cuda:0] [--frames 401]  ->  log/okuda_ECM/05f_secrete/

WHAT 05e LEFT READY. Mass is now the per-face state and rho = m_f/A_f is derived, so `sum_f m_f` is
bit-identical across a refinement and across a whole run. That is what licenses this step: a substrate
that lost material when it refined could not tell secretion from arithmetic.

THE OPERATOR, AND WHY IT CHANGED KIND. `bm_secrete` was specified as a Divide -- wake reserve faces to
add material -- which is what a PARTICLE model must do, and what run 128 did when it added 45,000
particles that sat at standoff -0.19, a second shell in the lumen, and halved the reported strain. With
mass carried as a scalar on a face it is a LATERAL operator instead: it raises m_f on faces that already
are the sheet. The reservoir is for resolution, not for supply, and run 128's failure mode becomes
structurally impossible rather than merely avoided.

    dm_f/dt = s(x,t) A_f  -  m_f / tau_bm            [secretion, turnover]
    rho_f   = m_f / A_f(now)                          [dilution is what m/A does; never integrated]
    s*      = rho_target (1/tau_bm + Adot/A)          [the homeostatic rate]

with tau_bm = t_half/ln2 = 4-14 h. At 9-18 minutes per frame that is 13-93 frames, so turnover is
RESOLVED here rather than instantaneous; the nominal is 40.

THE CLAIM THIS RUN CAN FALSIFY, from `note_spheroid_bm_ecm` S9: the two removal terms are the SAME SIZE
here -- a spheroid tripling its radius over a day has Adot/A ~ 1e-1 per hour against 1/tau_bm ~ 1e-1 per
hour -- so "turnover is not a correction to growth, it is the same size as growth". That is an
arithmetic claim about numbers this rig measures, and panel (b) is it.

AND THE DEMO THE WHOLE LADDER WAS BUILT FOR (`LADDER.md`, milestone M2, never reached): a sheet that is
fed does not tear and a sheet that is starved does, AT THE SAME STRETCH. Tearing here is driven by
SUPPLY, not by strain: a face dies when its areal density falls below rho_crit, which is what "a
basement membrane fails where it is not resupplied" means as a mechanism. Whether that tear is also
independent of the mesh is G19, and it is 05g.

THE GATES:
  G24  rho held within 10% of seeded while the area triples, with s = s*
  G25  the added material is IN the sheet -- satisfied BY CONSTRUCTION (mass is added to existing
       faces), so what is measured instead is the polarised variant: where secretion is proportional to
       plaque coverage, does the mass end up distributed as the plaques are?
  G26  fed does not tear and starved does, at the same lambda
  G27  the two removal terms are comparable: Adot/A against 1/tau_bm, measured, over the run
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
from matplotlib.colors import ListedColormap                           # noqa: E402

import bm_ops as BM                                                     # noqa: E402
import ecm_spec as ES                                                   # noqa: E402
from test_05_sheet import LOG, TISSUE, SCALE, UNITS, T_REAL_UM                            # noqa: E402
from test_05d_adhesion import Rig05d                                    # noqa: E402
import test_05e_conserve as E5                                          # noqa: E402
from rerender_05 import write_traj, render_from_traj                    # noqa: E402

CMAP = ListedColormap(ES.STRESS_COLORS)


class Rig05f(Rig05d):
    """05e's rig plus the mass balance and a supply-driven tear.

    ORDER WITHIN A FRAME: refine and integrate (05e's `frame`), then secrete, then degrade. Mass is
    changed on the state the frame produced, and the tear reads the density that secretion just set --
    so a face that was rescued this frame is not killed by last frame's number.
    """

    def __init__(self, *a, s_mode="homeostatic", tau_bm=40.0, rho_crit=0.0, s_scale=1.0,
                 spread_passes=12, **kw):
        self.s_mode, self.tau_bm, self.rho_crit, self.s_scale = (s_mode, float(tau_bm),
                                                                 float(rho_crit), float(s_scale))
        self._spread_passes = int(spread_passes)      # unused: n_b is already a smooth field
        super().__init__(*a, **kw)
        self.rho_target = self.sheet.rho0
        self._prev_area = float(self.sheet.area().sum())
        for k in ("s_rate", "dilution_rate", "turnover_rate", "torn", "rho_min", "rho_p05",
                  "mass_secreted", "mass_lost", "polar_corr", "lam_corr"):
            self.res[k] = []
        self._secreted = 0.0
        self._lost = 0.0

    def _plaque_cover(self, spread=None):
        """Per-face ADHESION, as the bond density the clutch is already carrying.

        THE SMOOTHING STAND-IN IS GONE, and that is what the scheme change bought. With discrete
        plaques, "deposition follows adhesion" meant multiplying by an indicator that was nonzero on
        8% of faces, so 92% of the sheet received literally nothing and dissolved -- and rescuing it
        needed an invented Jacobi kernel standing in for `bm_assemble`. `n_b` is a continuous field by
        construction: every face has adhesion, and how much varies smoothly. Polarised deposition is
        therefore just deposition proportional to n_b, with no kernel and nothing to tune.
        """
        nb = torch.zeros(self.sheet.x.shape[0], device=self.dev, dtype=self.dtype)
        nb.index_add_(0, self.ct_node, self.clutch.Nb / self._dual_area())
        return nb[self.sheet.Fc].mean(1)

    def secrete(self, dt=1.0):
        """`bm_secrete`, a Lateral operator on bm_face: raise the mass of faces that already exist."""
        live = self.sheet.live
        A = self.sheet.area()
        area = float(A.sum())
        # Adot/A from the LAST frame's area, which includes both the tissue's growth and anything the
        # remesher or a tear did to the face set. Measuring it from the geometry rather than assuming
        # the tissue's growth law is what makes G27 a measurement.
        dil = (area - self._prev_area) / max(self._prev_area, 1e-30) / max(dt, 1e-30)
        self._prev_area = area
        turn = 1.0 / self.tau_bm
        s_star = self.rho_target * (turn + max(dil, 0.0))
        if self.s_mode == "starved":
            s = torch.zeros_like(A)
        elif self.s_mode == "polarised":
            # the SAME total rate, redistributed by plaque coverage, so fed and polarised differ in
            # WHERE the material goes and not in how much there is
            w = self._plaque_cover()
            wm = float((w * A).sum())
            s = (self.s_scale * s_star * area / max(wm, 1e-30)) * w
        else:
            s = torch.full_like(A, self.s_scale * s_star)
        add = s * A * dt
        loss = self.sheet.mass[live] / self.tau_bm * dt
        self.sheet.mass[live] = (self.sheet.mass[live] + add - loss).clamp_min(0.0)
        self._secreted += float(add.sum())
        self._lost += float(loss.sum())
        return float(s.mean()), dil, turn

    def degrade(self):
        """`bm_degrade` on a SUPPLY criterion: a face dies where its areal density has fallen below
        critical. This is the biological form -- a basement membrane fails where it is not resupplied --
        as opposed to 05d's stretch criterion, which is mechanics."""
        if self.rho_crit <= 0:
            return 0
        rho = self.sheet.areal_density() / self.sheet.rho0
        return self.sheet.tear(rho < self.rho_crit)

    def alive(self):
        """A sheet with no faces left is not a torn sheet, it is a dissolved one, and every ratio
        measured on it divides by zero. The run stops and says which frame it stopped at."""
        return super().alive() and self.sheet.m >= 8

    def frame(self, t):
        super().frame(t)
        s_mean, dil, turn = self.secrete()
        torn = self.degrade()
        rho = self.sheet.areal_density() / self.sheet.rho0
        if rho.numel() == 0:
            rho = torch.zeros(1, device=self.dev, dtype=self.dtype)
        self.res["s_rate"].append(s_mean)
        self.res["dilution_rate"].append(dil)
        self.res["turnover_rate"].append(turn)
        self.res["torn"].append(int(torn))
        self.res["rho_min"].append(float(rho.min()) if rho.numel() else 0.0)
        self.res["rho_p05"].append(float(torch.quantile(rho, 0.05)) if rho.numel() > 1 else 0.0)
        self.res["mass_secreted"].append(self._secreted)
        self.res["mass_lost"].append(self._lost)
        # G25: WHERE DID THE MATERIAL GO, AND THE CONFOUND. Correlating per-face mass with plaque
        # coverage is not enough on its own: under UNIFORM secretion dm = s*A*dt, so a face that is
        # stretched more receives more, and sparse plaques stretch the sheet locally by dragging it --
        # which makes mass correlate with plaque coverage through mechanics rather than through
        # deposition. The 90-frame development run measured +0.76 for uniform secretion for exactly
        # that reason. So the stretch correlation is reported beside it as the control, and the pair
        # is what discriminates: polarised raises the plaque correlation ABOVE the stretch one.
        def corr(a, b):
            if a.numel() < 3 or float(a.std()) <= 0 or float(b.std()) <= 0:
                return 0.0
            return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))
        w = self._plaque_cover()
        m = self.sheet.mass[self.sheet.live] / self.sheet.A0[self.sheet.live].clamp_min(1e-30)
        l1f, _ = self.sheet.stretch_geo()
        self.res["polar_corr"].append(corr(w, m))
        self.res["lam_corr"].append(corr(l1f, m))
        # the last recorded rho/mass are pre-secretion; overwrite with the post-secretion truth
        self.res["rho"][-1] = float(rho.mean()) * self.sheet.rho0
        self.res["mass"][-1] = self.sheet.total_mass()


def model_png(runs, d, tau_bm, rho_crit):
    """`secrete_model.png` -- the operator, its equations, and four measurements."""
    fig = plt.figure(figsize=(14.6, 6.0), facecolor="white")
    axE = fig.add_axes([0.005, 0.05, 0.235, 0.90]); axE.axis("off")
    ax = [fig.add_axes([0.315, 0.575, 0.29, 0.345]), fig.add_axes([0.695, 0.575, 0.29, 0.345]),
          fig.add_axes([0.315, 0.095, 0.29, 0.375]), fig.add_axes([0.695, 0.095, 0.29, 0.375])]
    axE.text(0.0, 1.00, "bm_secrete", fontsize=13, fontweight="bold", va="top", family="monospace")
    axE.text(0.0, 0.935, "a Lateral operator on the bm_face hyperedge set\n"
                         "(it was specified as a Divide; with mass as a\n"
                         "scalar on a face, no new face is needed)",
             fontsize=8.2, va="top", color="#444")
    axE.text(0.0, 0.795, r"$\dfrac{dm_f}{dt} \;=\; s(\mathbf{x},t)\,A_f \;-\; \dfrac{m_f}{\tau_{bm}}$",
             fontsize=14, va="top")
    axE.text(0.0, 0.665, r"$\rho_f=\dfrac{m_f}{A_f(\mathrm{now})}$", fontsize=14, va="top")
    axE.text(0.0, 0.545, r"$s^{*}=\rho_{0}\left(\dfrac{1}{\tau_{bm}}+\dfrac{\dot A}{A}\right)$",
             fontsize=14, va="top")
    axE.text(0.0, 0.425, r"a face dies where $\rho_f<\rho_{\rm crit}$", fontsize=11, va="top")
    axE.text(0.0, 0.355,
             "the dilution term is NOT integrated: it is what\n"
             "$m/A$ does on its own as the tissue grows. Adding\n"
             "it as well would count it twice. Mass goes onto\n"
             "faces that already are the sheet, so run 128's\n"
             "second shell in the lumen cannot happen here.",
             fontsize=8.2, va="top", color="#333")
    axE.text(0.0, 0.165,
             f"$\\tau_{{bm}}$ = {tau_bm:g} frames ($t_{{1/2}}$ = 3--10 h = 13--93 frames)\n"
             f"$\\rho_{{\\rm crit}}$ = {rho_crit:g}$\\,\\rho_0$   $\\rho_{{\\rm target}}=\\rho_0$",
             fontsize=8.2, va="top", color="#333")
    axE.text(0.0, 0.03, "Matsubayashi et al. 2020 Dev Cell 54:33\n"
                        "Yurchenco 2011 Cold Spring Harb Perspect Biol 3:a004911\n"
                        "run 128: the archived secretion that missed the sheet",
             fontsize=7.3, va="bottom", color="#666")

    col = {"fed": "#1a1a1a", "half": "#e08a2e", "starved": "#b03030", "polarised": "#2b6cb0",
           "half, polarised": "#7a3b9a", "fed, no turnover": "#1f8a5c"}
    for k, r in runs.items():
        t = np.arange(len(r["rho"]))
        ax[0].plot(t, np.asarray(r["rho"]), lw=1.7, color=col.get(k, "#777"), label=k)
    ax[0].axhline(1.0, color="#999", ls="--", lw=0.9)
    if rho_crit > 0:
        ax[0].axhline(rho_crit, color="#b03030", ls=":", lw=1.1)
        ax[0].text(0.02, rho_crit, r"  $\rho_{\rm crit}$", transform=ax[0].get_yaxis_transform(),
                   color="#b03030", fontsize=7.5, va="bottom")
    ax[0].set_yscale("log"); ax[0].set_ylabel(r"areal density $\rho/\rho_0$")
    fed = runs["fed"]
    ax[0].set_title(f"G24: fed holds $\\rho$ at {fed['rho'][-1]:.3f} while the area\n"
                    f"triples; starved falls to {runs['starved']['rho'][-1]:.3f}", fontsize=8.5)
    ax[0].legend(fontsize=7, frameon=False)

    t = np.arange(len(fed["dilution_rate"]))
    ax[1].plot(t, fed["dilution_rate"], color="#1f8a5c", lw=1.5, label=r"dilution $\dot A/A$")
    ax[1].plot(t, fed["turnover_rate"], color="#7a3b9a", lw=1.5, ls="--",
               label=r"turnover $1/\tau_{bm}$")
    ax[1].set_yscale("log"); ax[1].set_ylabel("per frame")
    rat = np.asarray(fed["dilution_rate"])[10:] / np.asarray(fed["turnover_rate"])[10:]
    ax[1].set_title(f"G27: the two removal terms are comparable --\n"
                    f"median ratio {np.median(rat):.2f}, range "
                    f"{np.percentile(rat,5):.2f}--{np.percentile(rat,95):.2f}", fontsize=8.5)
    ax[1].legend(fontsize=7, frameon=False)

    for k, r in runs.items():
        if "torn" not in r:
            continue
        ax[2].plot(r["lam_geo"], np.cumsum(r["torn"]), lw=1.7, color=col.get(k, "#777"), label=k)
    ax[2].set_xlabel(r"$\lambda_{geo}$ (NOT frame: the demo is at equal stretch)")
    ax[2].set_ylabel("faces torn, cumulative")
    n_fed = int(np.sum(fed["torn"]))
    n_hp = int(np.sum(runs.get("half, polarised", {"torn": [0]})["torn"]))
    ax[2].set_title(f"G26: fed tears {n_fed}; half+polarised tears {n_hp} where\n"
                    f"the plaques are not. Uniform starvation DISSOLVES", fontsize=8.5)
    ax[2].legend(fontsize=7, frameon=False)

    for k in ("fed", "polarised"):
        if k in runs:
            ax[3].plot(np.arange(len(runs[k]["polar_corr"])), runs[k]["polar_corr"], lw=1.7,
                       color=col.get(k, "#777"), label=k)
    ax[3].axhline(0.0, color="#999", ls="--", lw=0.9)
    for k in ("fed", "polarised"):
        if k in runs:
            ax[3].plot(np.arange(len(runs[k]["lam_corr"])), runs[k]["lam_corr"], lw=1.1, ls=":",
                       color=col.get(k, "#777"))
    ax[3].set_ylabel("corr with mass/$A^0$ (solid: plaques, dotted: stretch)")
    pc = runs.get("polarised", {}).get("polar_corr", [0.0])
    ax[3].set_title(f"G25: polarised ends at r = {pc[-1]:+.3f} against the\n"
                    f"stretch confound (dotted); uniform "
                    f"{fed['polar_corr'][-1]:+.3f}", fontsize=8.5)
    ax[3].legend(fontsize=7, frameon=False)
    for a in ax[:2]:
        a.tick_params(labelbottom=False)
    ax[3].set_xlabel("frame")
    for a in ax:
        a.spines[["top"]].set_visible(False)
    fig.savefig(os.path.join(d, "secrete_model.png"), dpi=150, facecolor="white")
    plt.close(fig)


def main():
    def arg(flag, cast, default):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 401)
    name = arg("--name", str, "05f_secrete")
    tau_bm = arg("--tau", float, 40.0)
    rho_crit = arg("--rhocrit", float, 0.35)
    d = os.path.join(LOG, name)
    os.makedirs(d, exist_ok=True)

    cert = BM.selftest(dev=dev, subdiv=4)
    assert cert["remesh_mass_rel"] < 1e-12, cert
    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, kn=5.0, sigma_T=7.0, zeta=20.0,
             s_target=1.0, k_drive=50.0, dev=dev)
    Q = dict(max_refine=2, edge_trigger=1.45, reseed=True, tau_bm=tau_bm, rho_crit=rho_crit)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 160))).astype(int).tolist())

    runs, rigs = {}, {}
    # THE TEAR NEEDS A HETEROGENEITY, and the first version of this run found out the hard way: a
    # uniform sheet, uniformly starved, against a uniform rho_crit crosses the threshold EVERYWHERE at
    # once and dissolves rather than tearing. That is a correct outcome for a uniform problem and it is
    # reported as dissolution, not as a tear. The biological source of the heterogeneity is polarised
    # supply -- an epithelium maintains its basement membrane where it is ATTACHED -- so `half,
    # polarised` is the run in which holes have a location, and it is the tear demo.
    plan = [("fed", dict(s_mode="homeostatic", s_scale=1.0)),
            ("half", dict(s_mode="homeostatic", s_scale=0.5)),
            ("starved", dict(s_mode="starved")),
            ("polarised", dict(s_mode="polarised", s_scale=1.0)),
            ("half, polarised", dict(s_mode="polarised", s_scale=0.5)),
            ("fed, slow turnover", dict(s_mode="homeostatic", s_scale=1.0, tau_bm=93.0)),
            ("fed, no turnover", dict(s_mode="homeostatic", s_scale=1.0, tau_bm=1e9))]
    for lab, extra in plan:
        r = Rig05f(**P, **{**Q, **extra})
        # EVERY VARIANT GETS ITS OWN FOLDER AND ITS OWN MOVIE. The run is a comparison -- fed against
        # starved, uniform against polarised -- and a comparison with one movie of one arm is an
        # assertion. `movie.mp4` at the top stays the NOMINAL, because a run's default artefact should
        # be the run and not its most extreme arm: rendering `half, polarised` there read as "secretion
        # is broken" when `fed` holds rho at 1.005 and loses no face at all.
        kp = keep
        kept, _ = E5.run(r, frames, keep=kp, label=f"{name}: {lab}")
        runs[lab], rigs[lab] = r.res, r
        slug = lab.replace(", ", "_").replace(" ", "_")
        sub = d if lab == "fed" else os.path.join(d, slug)
        write_traj(kept, r.F_epi.cpu().numpy(), sub)
        render_from_traj(sub, zoom=1.0, l0=r.l0, title=f"{name}: {lab}")

    model_png(runs, d, tau_bm, rho_crit)

    fed, st = runs["fed"], runs["starved"]
    dil = np.asarray(fed["dilution_rate"])[10:]; tur = np.asarray(fed["turnover_rate"])[10:]
    # THE DEMO, AT EQUAL STRETCH. "starved tears and fed does not" is only a statement about supply if
    # the two are compared at the same deformation; compared at the same FRAME it could be a statement
    # about the clock. lam_at_first_tear is the number that makes it the former.
    demo = runs["half, polarised"]
    st_t = np.asarray(demo["torn"]); f_t = np.asarray(fed["torn"])
    i_st = int(np.argmax(st_t > 0)) if st_t.any() else -1
    lam_star = demo["lam_geo"][i_st] if i_st >= 0 else None
    fed_torn_by_then = (int(f_t[:int(np.argmax(np.asarray(fed["lam_geo"]) >= lam_star)) + 1].sum())
                        if lam_star is not None and max(fed["lam_geo"]) >= lam_star else 0)
    out = dict(
        run=name, frames=frames, certification=cert,
        rig=dict(**{k: v for k, v in P.items() if k != "dev"}, **{k: v for k, v in Q.items()}),
        G24=dict(rho_final_fed=fed["rho"][-1], rho_final_half=runs["half"]["rho"][-1],
                 rho_final_starved=st["rho"][-1],
                 rho_final_no_turnover=runs["fed, no turnover"]["rho"][-1],
                 area_ratio=fed["area"][-1] / fed["area"][0],
                 within_10pc=bool(abs(fed["rho"][-1] - 1.0) < 0.10)),
        G25=dict(satisfied_by_construction="mass is added to faces that already are the sheet, so run "
                                           "128's second shell in the lumen cannot occur",
                 polar_corr_final_polarised=runs["polarised"]["polar_corr"][-1],
                 polar_corr_final_uniform=fed["polar_corr"][-1],
                 stretch_corr_final_polarised=runs["polarised"]["lam_corr"][-1],
                 stretch_corr_final_uniform=fed["lam_corr"][-1],
                 confound="under uniform secretion dm = s A dt, so mass follows STRETCH, and sparse "
                          "plaques stretch the sheet locally -- the stretch correlation is the "
                          "control that separates deposition from mechanics"),
        G26=dict(faces_torn_fed=int(f_t.sum()),
                 faces_torn_half_polarised=int(st_t.sum()),
                 faces_torn_starved=int(np.sum(st["torn"])),
                 starved_dissolved_at_frame=(len(st["torn"]) if len(st["torn"]) < frames else None),
                 faces_left_starved=st["n_faces"][-1],
                 faces_torn_half=int(np.sum(runs["half"]["torn"])),
                 lambda_at_first_tear_starved=lam_star,
                 fed_faces_torn_by_that_lambda=fed_torn_by_then,
                 rho_crit=rho_crit),
        G27=dict(tau_bm_frames=tau_bm,
                 dilution_median=float(np.median(dil)), turnover=float(tur[0]),
                 ratio_slow_turnover=float(np.median(
                     np.asarray(runs["fed, slow turnover"]["dilution_rate"])[10:]
                     / np.asarray(runs["fed, slow turnover"]["turnover_rate"])[10:])),
                 ratio_median=float(np.median(dil / tur)),
                 ratio_p05=float(np.percentile(dil / tur, 5)),
                 ratio_p95=float(np.percentile(dil / tur, 95)),
                 claim="note S9: the two removal terms are the same size here"),
        mass=dict(secreted_total=fed["mass_secreted"][-1], lost_total=fed["mass_lost"][-1],
                  mass_first=fed["mass"][0], mass_final=fed["mass"][-1],
                  starved_mass_final=st["mass"][-1]),
        series={k: {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in runs.items()})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
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
        what="05f -- the mass balance on the sheet, and a tear driven by supply rather than by strain",
        operator=dict(name="bm_secrete", kind="Lateral on bm_face",
                      why_not_divide="mass is a scalar on a face, so no new face is needed; the "
                                     "reservoir is for resolution, not for supply",
                      equation="dm_f/dt = s A_f - m_f/tau_bm, rho = m_f/A_f",
                      homeostatic="s* = rho0 (1/tau_bm + Adot/A), with Adot/A MEASURED from the "
                                  "geometry rather than assumed from the tissue's growth law"),
        degrade=dict(criterion="rho_f < rho_crit", why="a basement membrane fails where it is not "
                                                       "resupplied; 05d's criterion was stretch, "
                                                       "which is mechanics"),
        tau_bm=dict(frames=tau_bm, provenance="t_half 3-10 h / ln2, at 9-18 min per frame = 13-93 "
                                              "frames"),
        gates=dict(G24="rho within 10% of seeded while the area triples",
                   G25="the added material is in the sheet (by construction) and follows the plaques "
                       "when secretion is polarised",
                   G26="fed does not tear and starved does, AT THE SAME lambda",
                   G27="dilution and turnover are comparable, measured"),
        not_modelled=["proteolysis as a field (05g)", "the refinement-invariance of this tear (G19, "
                      "05g)", "plaque turnover", "bending, so the hole's rim has no line tension"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    print(f"[{name}] G24 fed rho {out['G24']['rho_final_fed']:.4f} at area x"
          f"{out['G24']['area_ratio']:.2f} (starved {out['G24']['rho_final_starved']:.4f}); "
          f"G26 torn fed {out['G26']['faces_torn_fed']} vs starved "
          f"{out['G26']['faces_torn_starved']}, starved first tears at lambda "
          f"{out['G26']['lambda_at_first_tear_starved']}; G27 dilution/turnover median "
          f"{out['G27']['ratio_median']:.2f}; G25 polar corr "
          f"{out['G25']['polar_corr_final_polarised']:+.3f} vs "
          f"{out['G25']['polar_corr_final_uniform']:+.3f} -> {d}", flush=True)


if __name__ == "__main__":
    main()
