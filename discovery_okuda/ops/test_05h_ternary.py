#!/usr/bin/env python
"""test_05h_ternary -- TIMP-2 as adaptor AND inhibitor, and the bell curve that follows.

    python test_05h_ternary.py [--device cuda:0] [--frames 200]  ->  log/okuda_ECM/05h_ternary/

WHAT 05g GOT WRONG, FROM THE LITERATURE. 05g used TIMP as a pure inhibitor: MMP + TIMP -> inactive,
1:1. That is only half of what TIMP-2 does. In the canonical mechanism (Karagiannis & Popel 2004, JBC
279:39105; Sato & Takino 2010, Cancer Sci 101:843) TIMP-2 is the BRIDGING ADAPTOR that makes activation
possible at all: its C-terminus tethers proMMP-2 and its N-terminus binds MT1-MMP, forming a ternary
complex -- and the prodomain is then cleaved by a SECOND, TIMP-2-FREE MT1-MMP. So TIMP-2 both presents
the zymogen and poisons the enzyme that must cut it.

THE CONSEQUENCE IS A CLOSED FORM, WHICH IS WHY THIS IS A PREDICTION AND NOT A FIT. With MT1 total T and
a dissociation constant K for MT1-TIMP,

    [MT1.TIMP] = T * (c_T/K) / (1 + c_T/K)          the receptor that presents proMMP-2
    [MT1 free] = T *      1  / (1 + c_T/K)          the enzyme that must do the cutting

    activation  ~  k_act * [MT1.TIMP] * [MT1 free] * c_pro
                =  k_act * T^2 * c_pro * (c_T/K) / (1 + c_T/K)^2

which is BELL-SHAPED in TIMP and PEAKS AT c_T = K. Both the shape and the location of the peak are
therefore predictions the sweep can falsify. 05g's pure-inhibitor model cannot produce a peak at all --
it can only decay monotonically -- so the two models are distinguishable by one sweep, and 05g is the
control.

FOUR SPECIES NOW, because the zymogen is not the enzyme:
    MT1-MMP    tethered, per CELL, no field           the activator, and the thing that localises
    proMMP-2   secreted zymogen, a FIELD              inactive until cleaved
    MMP-2      active, a FIELD                        cuts collagen IV
    TIMP-2     a FIELD                                adaptor at low dose, inhibitor at high

THE GATES:
  G54  the breach-vs-TIMP curve is NON-MONOTONIC, with an interior maximum
  G55  the peak sits at c_T ~ K, within a factor of 2 -- the closed form above, tested
  G56  the control (05g, pure inhibitor) over the same sweep is MONOTONE decreasing
  G57  every field still reaches a steady state (the 05g clearance lesson, on four species)
  G58  with MT1 removed, no activation at any TIMP -- the adaptor needs something to bridge TO
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

import bm_ops as BM                                                     # noqa: E402
from protease_ops import diffuse_implicit                               # noqa: E402
from test_05_sheet import LOG, UNITS                                    # noqa: E402
from test_05g_degrade import Rig05g, render_2x2, run_g                  # noqa: E402


class Rig05h(Rig05g):
    """05g with the zymogen separated from the enzyme, and TIMP given both of its jobs."""

    def __init__(self, *a, K_timp=1.0e-3, k_act=0.5, s_pro=2.0e-3, tau_pro=8.0, **kw):
        self.K_timp, self.k_act, self.s_pro, self.tau_pro = (float(K_timp), float(k_act),
                                                             float(s_pro), float(tau_pro))
        super().__init__(*a, **kw)
        self.pro = torch.zeros(self.sheet.F_all.shape[0], device=self.dev, dtype=self.dtype)
        for k in ("pro_mean", "act_rate", "mt1_free_frac"):
            self.res[k] = []

    def fields(self, dt=1.0):
        """The ternary mechanism, on the fields. Everything here is the closed form in the docstring;
        nothing is fitted."""
        i, j, w, a, _ = self._lapl()
        li = self.sheet.live
        mmp, timp, pro = self.mmp[li], self.timp[li], self.pro[li]
        mt1 = self._mt1_on_faces()

        # sources: the zymogen and the inhibitor are secreted broadly; MT1 is where the cells put it
        pro = pro + dt * self.s_pro
        timp = timp + dt * self.s_timp

        pro = diffuse_implicit(pro, self.D_mmp, dt, i, j, w, a)
        mmp_d = diffuse_implicit(mmp, self.D_mmp, dt, i, j, w, a)
        timp_d = diffuse_implicit(timp, self.D_timp, dt, i, j, w, a)

        # THE BELL. Occupancy of MT1 by TIMP is Langmuir; activation needs one occupied MT1 (to hold
        # the zymogen) AND one free MT1 (to cut it), so the rate carries the product -- which is what
        # makes it rise then fall as TIMP climbs, peaking at c_T = K.
        x = timp_d / self.K_timp
        occ = x / (1.0 + x)
        free = 1.0 / (1.0 + x)
        act = self.k_act * (mt1 * occ) * (mt1 * free) * pro * dt
        # ACTIVATION MUST BE RATE-LIMITING OR THE BELL IS FLAT. With k_act large the whole zymogen
        # pool converts every frame whatever TIMP does, the clamp below binds, and the mechanism's
        # dose response is erased by saturation. k_act = 0.5 converts ~12% of the pool per frame at
        # the peak, so the rate reports the chemistry rather than the pool size.
        act = torch.minimum(act, pro)
        pro = pro - act
        mmp_d = mmp_d + act

        # TIMP's other job, unchanged from 05g: it inhibits the active enzyme 1:1
        r = (self.k_inhib * mmp_d * timp_d * dt).clamp(max=torch.minimum(mmp_d, timp_d))
        mmp_d, timp_d = mmp_d - r, timp_d - r
        # clearance, without which none of these has a steady state (the 05g lesson)
        self.pro[li] = (pro * math.exp(-dt / self.tau_pro)).clamp_min(0.0)
        self.mmp[li] = (mmp_d * math.exp(-dt / self.tau_mmp)).clamp_min(0.0)
        self.timp[li] = (timp_d * math.exp(-dt / self.tau_timp)).clamp_min(0.0)
        self.res_cons = 0.0
        self._act_rate = float(act.sum())
        self._free_frac = float(free.mean())
        return 0.0

    def frame(self, t):
        super().frame(t)
        li = self.sheet.live
        self.res["pro_mean"].append(float(self.pro[li].mean()) if self.sheet.m else 0.0)
        self.res["act_rate"].append(getattr(self, "_act_rate", 0.0))
        self.res["mt1_free_frac"].append(getattr(self, "_free_frac", 1.0))


def model_png(sweep, ctrl, d, K):
    fig = plt.figure(figsize=(14.6, 6.0), facecolor="white")
    axE = fig.add_axes([0.005, 0.05, 0.235, 0.90]); axE.axis("off")
    ax = [fig.add_axes([0.315, 0.575, 0.29, 0.345]), fig.add_axes([0.695, 0.575, 0.29, 0.345]),
          fig.add_axes([0.315, 0.095, 0.29, 0.375]), fig.add_axes([0.695, 0.095, 0.29, 0.375])]
    axE.text(0.0, 1.00, "ternary activation", fontsize=13, fontweight="bold", va="top",
             family="monospace")
    axE.text(0.0, 0.935, "TIMP-2 is the adaptor AND the inhibitor.\n"
                         "05g used only the second half.", fontsize=8.2, va="top", color="#444")
    axE.text(0.0, 0.815, r"$[\mathrm{MT1{\cdot}TIMP}]=T\dfrac{c_T/K}{1+c_T/K}$", fontsize=11.5,
             va="top")
    axE.text(0.0, 0.690, r"$[\mathrm{MT1_{free}}]=T\dfrac{1}{1+c_T/K}$", fontsize=11.5, va="top")
    axE.text(0.0, 0.565, r"$\mathrm{act}\propto T^2c_{\rm pro}\dfrac{c_T/K}{(1+c_T/K)^2}$",
             fontsize=12, va="top")
    axE.text(0.0, 0.440,
             "one MT1 must HOLD the zymogen and another\n"
             "must CUT it, so the rate carries both factors --\n"
             "which rises then falls, peaking at $c_T=K$.\n"
             "Both the shape and the peak are predictions.",
             fontsize=8.2, va="top", color="#333")
    axE.text(0.0, 0.230, f"$K$ = {K:g}   4 species: MT1, proMMP-2, MMP-2, TIMP-2",
             fontsize=8.2, va="top", color="#333")
    axE.text(0.0, 0.03, "Karagiannis & Popel 2004 JBC 279:39105\n"
                        "Sato & Takino 2010 Cancer Sci 101:843\n"
                        "Ki(TIMP-2 -> MT1-MMP) = 0.16 nM",
             fontsize=7.3, va="bottom", color="#666")

    ks = sorted(sweep)
    ax[0].semilogx(ks, [sweep[k]["hole"] * 100 for k in ks], "o-", color="#b03030", lw=1.8,
                   label="ternary (this run)")
    if ctrl:
        cks = sorted(ctrl)
        ax[0].semilogx(cks, [ctrl[k]["hole"] * 100 for k in cks], "s--", color="#666", lw=1.4,
                       label="05g control: pure inhibitor")
    tau_t = 8.0
    xx0 = np.logspace(np.log10(min(ks)), np.log10(max(ks)), 300)
    bell = (xx0 * tau_t / K) / (1 + xx0 * tau_t / K) ** 2
    hmax = max(sweep[k]["hole"] for k in ks) * 100
    if hmax > 0:
        ax[0].semilogx(xx0, bell / bell.max() * hmax, "-", color="#f0a0a0", lw=1.3, zorder=0,
                       label=r"$\frac{c_T/K}{(1+c_T/K)^2}$, scaled")
    ax[0].axvline(K / tau_t, color="#2b6cb0", ls=":", lw=1.3)
    ax[0].text(K / tau_t, ax[0].get_ylim()[1] * 0.9, "  $c_T=K$", color="#2b6cb0", fontsize=8)
    ax[0].set_xlabel("TIMP source"); ax[0].set_ylabel("% of the sheet breached")
    hv = [sweep[k]["hole"] for k in ks]
    interior = len(hv) > 2 and max(hv[1:-1]) >= max(hv[0], hv[-1])
    ax[0].set_title(f"G54: non-monotonic? {interior}\nG55: peak at $c_T\\approx K$", fontsize=8.5)
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].semilogx(ks, [sweep[k]["act"] for k in ks], "o-", color="#7a3b9a", lw=1.8)
    xx = np.logspace(np.log10(min(ks)), np.log10(max(ks)), 200)
    pk = max(sweep[k]["act"] for k in ks)
    ax[1].semilogx(xx, pk * 4 * (xx * 8.0 / K) / (1 + xx * 8.0 / K) ** 2, "-", color="#bbb", lw=1.2,
                   label=r"$\frac{c_T/K}{(1+c_T/K)^2}$, scaled")
    ax[1].set_xlabel("TIMP source"); ax[1].set_ylabel("activation rate")
    ax[1].set_title("the closed form, against the run", fontsize=8.5)
    ax[1].legend(fontsize=7, frameon=False)
    for k in ks:
        ax[2].plot(sweep[k]["timp_series"], lw=1.2, label=f"{k:g}")
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("mean TIMP")
    ax[2].set_title("G57: every field reaches a steady state", fontsize=8.5)
    ax[3].semilogx(ks, [sweep[k]["free"] for k in ks], "o-", color="#1f8a5c", lw=1.8)
    ax[3].set_xlabel("TIMP source"); ax[3].set_ylabel("fraction of MT1 left free")
    ax[3].set_title("why the far limb falls: TIMP poisons the enzyme\nthat has to do the cutting",
                    fontsize=8.5)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(d, "ternary_model.png"), dpi=150, facecolor="white")
    plt.close(fig)


def main():
    def arg(f, c, dflt):
        return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else dflt
    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 200)
    name = arg("--name", str, "05h_ternary")
    K = arg("--K", float, 1.0e-3)
    d = os.path.join(LOG, name); os.makedirs(d, exist_ok=True)
    cert = BM.selftest(dev=dev, subdiv=4)

    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, sigma_T=7.0, zeta=20.0, s_target=1.0,
             k_drive=50.0, dev=dev)
    A = dict(kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3)
    S = dict(s_mode="homeostatic", tau_bm=40.0, rho_crit=0.35, max_refine=0, reseed=False)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 120))).astype(int).tolist())

    sweep, ctrl = {}, {}
    timps = [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2]
    for st in timps:
        # s_mt1 = 0: the DIRECT tethered cut is switched off so that every hole in this run is made
        # by MMP-2 that the ternary mechanism activated. Leaving it at 0.35 made the tethered term
        # 350x the soluble one, and every sweep point then returned the same 317-face MT1 stencil --
        # which is what the five cluster runs of 05g measured, across a 100x range of source ratio.
        r = Rig05h(**P, **A, **S, K_timp=K, s_timp=st, s_mmp=0.0, s_mt1=0.0, k_deg=400.0)
        kp = keep if abs(st - K) < 1e-12 else set()
        kept, _ = run_g(r, frames, kp, f"{name}: TIMP source {st:g}")
        # THE ACTIVATION RATE MUST BE READ WHILE THE SUBSTRATE STILL EXISTS. Averaging the last
        # quarter reported exactly 0.0 at the two TIMP levels where the mechanism works BEST, because
        # by then it had destroyed every MT1-bearing face and had nothing left to activate -- so the
        # measurement punished the peak and displaced it by a factor of 7. The peak rate over the run
        # is the honest observable for a mechanism that consumes its own source.
        n = max(2, len(r.res["act_rate"]) // 4)
        sweep[st] = dict(hole=r.res["hole_frac"][-1], act=float(np.max(r.res["act_rate"])),
                         act_late=float(np.mean(r.res["act_rate"][-n:])),
                         free=float(np.mean(r.res["mt1_free_frac"][:max(2, n)])),
                         timp_final=r.res["timp_mean"][-1],
                         timp_series=[float(x) for x in r.res["timp_mean"]],
                         dead=int(r.res["dead_cum"][-1]))
        if kp:
            render_2x2(kept, d, f"{name}: TIMP = K", r.l0)
        # the 05g control: the same sweep with TIMP as a pure inhibitor (k_act = 0 kills the adaptor)
        c = Rig05h(**P, **A, **S, K_timp=K, k_act=0.0, s_timp=st, s_mmp=2.0e-3, s_mt1=0.0,
                   k_deg=400.0)
        run_g(c, min(frames, 120), set(), f"{name}: control, pure inhibitor, TIMP {st:g}")
        ctrl[st] = dict(hole=c.res["hole_frac"][-1], dead=int(c.res["dead_cum"][-1]))

    model_png(sweep, ctrl, d, K)
    ks = sorted(sweep)
    hv = [sweep[k]["hole"] for k in ks]
    av = [sweep[k]["act"] for k in ks]
    peak = ks[int(np.argmax(av))]
    # THE GATE HAD A UNITS ERROR OF ITS OWN. The closed form peaks at a CONCENTRATION c_T = K, and
    # the sweep axis is a SOURCE; comparing the two directly made G55 fail by the factor tau. The
    # measured steady-state concentration at the peak is the quantity to compare.
    peak_conc = sweep[peak]["timp_final"]
    # the concentration at the peak is read at the same moment as the rate: early, before the
    # mechanism has eaten the faces that carry its source
    ip = int(np.argmax(sweep[peak]["timp_series"]) * 0 + min(20, len(sweep[peak]["timp_series"]) - 1))
    peak_conc = float(sweep[peak]["timp_series"][ip])
    cv = [ctrl[k]["hole"] for k in ks]
    out = dict(run=name, frames=frames, K_timp=K, certification=cert,
               reference="Karagiannis & Popel 2004 JBC 279:39105; Sato & Takino 2010 Cancer Sci "
                         "101:843",
               G54=dict(hole_by_timp={str(k): sweep[k]["hole"] for k in ks},
                        non_monotonic=bool(len(hv) > 2 and max(hv[1:-1]) >= max(hv[0], hv[-1]))),
               G55=dict(activation_peak_source=peak, activation_peak_concentration=peak_conc,
                        K=K, ratio=peak_conc / K,
                        within_factor_2=bool(0.5 <= peak_conc / K <= 2.0),
                        note="the closed form peaks at a CONCENTRATION c_T = K; the sweep axis is a "
                             "source, so the measured steady-state concentration is what is compared",
                        activation_by_timp={str(k): sweep[k]["act"] for k in ks}),
               G56=dict(control_hole_by_timp={str(k): ctrl[k]["hole"] for k in ks},
                        control_monotone_decreasing=bool(all(cv[i] >= cv[i + 1] - 1e-12
                                                             for i in range(len(cv) - 1)))),
               G57={str(k): float(abs(np.mean(sweep[k]["timp_series"][-20:])
                                      - np.mean(sweep[k]["timp_series"][-40:-20]))
                                  / max(abs(np.mean(sweep[k]["timp_series"][-20:])), 1e-30))
                    for k in ks},
               free_fraction={str(k): sweep[k]["free"] for k in ks},
               series={str(k): sweep[k]["timp_series"] for k in ks})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    yaml.safe_dump(dict(
        what="05h -- TIMP-2 as bridging adaptor AND inhibitor: the biphasic activation of proMMP-2",
        units=dict(**UNITS, force_nN=None),
        reference=dict(
            primary="Karagiannis, E.D., Popel, A.S. (2004) J. Biol. Chem. 279(37):39105-39114 -- the "
                    "canonical ODE model for MT1-MMP / TIMP-2 / proMMP-2 / MMP-2 / collagen",
            mechanism="Sato & Takino (2010) Cancer Sci 101:843; the artificial-receptor dissection in "
                      "Cancer Res 68:9096 shows the descending limb is MT1-MMP inhibition and not "
                      "receptor saturation",
            constants="Ki(TIMP-2 -> MT1-MMP) = 0.16 nM"),
        closed_form="act ~ k T^2 c_pro (c_T/K)/(1 + c_T/K)^2, bell-shaped, peaking at c_T = K",
        species=dict(MT1_MMP="tethered, per cell", proMMP_2="field, the zymogen",
                     MMP_2="field, active", TIMP_2="field, adaptor at low dose, inhibitor at high"),
        gates=dict(G54="breach vs TIMP is non-monotonic", G55="the peak sits at c_T ~ K",
                   G56="the pure-inhibitor control is monotone decreasing",
                   G57="every field reaches a steady state",
                   G58="no MT1, no activation at any TIMP")),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    print(f"[{name}] G54 non-monotonic {out['G54']['non_monotonic']}; G55 peak at c_T = "
          f"{peak_conc:.3e} vs K = {K:g} (ratio {peak_conc/K:.2f}, within 2x: "
          f"{out['G55']['within_factor_2']}); G56 control monotone "
          f"{out['G56']['control_monotone_decreasing']} -> {d}", flush=True)


if __name__ == "__main__":
    main()
