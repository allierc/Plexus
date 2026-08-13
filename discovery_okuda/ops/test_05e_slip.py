#!/usr/bin/env python
"""test_05e_slip -- 05e: slip IS bond turnover, which is why the old rig could not measure it.

    python test_05e_slip.py [--device cuda:0] [--frames 200]  ->  log/okuda_ECM/05e_slip/

THE DEFECT THIS CLOSES (G13, open since 05b). With a discrete plaque anchored to a FIXED barycentric
point, the sheet twisted 28.49 deg against a drive of 28.50 -- it followed the epithelium essentially
exactly WITH NO FRICTION AT ALL. The reason was structural, not numerical: a fixed anchor is a material
coordinate that rotates with the face, so it is a PIN. A friction coefficient can only set how fast the
sheet equilibrates to a pin; it can never decide whether the sheet slides.

Slip is not friction. Slip is bond turnover: the sheet advances because k_off releases bonds that
rebind ahead of where they were. So the observable is the twist of the sheet against the twist the
epithelium was driven through, as a function of k_off -- and the control is k_off = 0, which must pin.

WHAT IS MEASURED:
  G32   the slip rate is monotone in k_off, and zero at k_off = 0
  G32b  the bound fraction falls as k_off rises -- the same knob, seen on the other state
  G30   receptor is still conserved while bonds are cycling hard (this is where it would break)

WHAT IS NOT HERE. The epithelium is the driven icosphere, rotating rigidly. A real tissue slides by T1
and division, which is 05k; a rigid rotation is the cleanest tangential load there is and it is the one
that isolates the adhesion.
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
from test_05_sheet import LOG, UNITS                                    # noqa: E402
from test_05d_adhesion import Rig05d                                    # noqa: E402
import test_05e_conserve as E5                                          # noqa: E402
from rerender_05 import write_traj, render_from_traj                    # noqa: E402


def model_png(runs, omega, d, P, chars=None):
    fig = plt.figure(figsize=(14.6, 6.0), facecolor="white")
    axE = fig.add_axes([0.005, 0.05, 0.235, 0.90]); axE.axis("off")
    ax = [fig.add_axes([0.315, 0.575, 0.29, 0.345]), fig.add_axes([0.695, 0.575, 0.29, 0.345]),
          fig.add_axes([0.315, 0.095, 0.29, 0.375]), fig.add_axes([0.695, 0.095, 0.29, 0.375])]
    axE.text(0.0, 1.00, "slip", fontsize=13, fontweight="bold", va="top", family="monospace")
    axE.text(0.0, 0.935, "not an operator: a CONSEQUENCE of plaque_bind.\n"
                         "The sheet advances because bonds let go and\n"
                         "re-form ahead of where they were.",
             fontsize=8.2, va="top", color="#444")
    axE.text(0.0, 0.795, r"$k_{off}(f)=k^0_{off}\,e^{\,f/f_b}$", fontsize=13, va="top")
    axE.text(0.0, 0.700, r"slip $=\ \omega t-\langle\theta_{\rm sheet}\rangle$", fontsize=12,
             va="top")
    axE.text(0.0, 0.590,
             "A FIXED anchor is a pin: it is a material\n"
             "coordinate that turns with the face, so no\n"
             "friction coefficient can make it slide. That is\n"
             "why 05b measured 28.49 deg of twist against a\n"
             "28.50 deg drive with friction switched OFF, and\n"
             "why G13 could not be answered by tuning $\\xi$.",
             fontsize=8.2, va="top", color="#333")
    axE.text(0.0, 0.330,
             f"drive $\\omega$ = {omega:g} rad/frame\n"
             f"$k_{{on}}$ = {P['k_on']:g}   $f_b$ = {P['f_bell']:g}   "
             f"$\\kappa_b$ = {P['kappa_b']:g}",
             fontsize=8.2, va="top", color="#333")
    axE.text(0.0, 0.03, "Bell 1978 Science 200:618\n"
                        "the clutch: bonds bear load, release, and rebind",
             fontsize=7.3, va="bottom", color="#666")

    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(runs)))
    ks = sorted(runs)
    for c, k in zip(cols, ks):
        r = runs[k]
        t = np.arange(len(r["twist"]))
        ax[0].plot(t, np.asarray(r["twist"]) * 180 / np.pi, lw=1.6, color=c,
                   label=rf"$k^0_{{off}}$ = {k:g}")
        ax[2].plot(t, r["bound_frac"], lw=1.6, color=c)
    n = len(runs[ks[0]]["twist"])
    ax[0].plot(np.arange(n), omega * np.arange(n) * 180 / np.pi, "--", color="#b03030", lw=1.3,
               label="the drive")
    ax[0].set_ylabel("twist of the sheet (deg)")
    ax[0].set_title("G32: the sheet follows the drive less as bonds\ncycle faster", fontsize=8.5)
    ax[0].legend(fontsize=7, frameon=False)

    # THE BASELINE IS NOT ZERO, and the gate as first written asked for the wrong thing. At
    # k_off = 0 no bond ever lets go, so whatever the sheet fails to follow is its own ELASTIC SHEAR,
    # not slip. Slip is what turnover adds on top of that, so the k_off = 0 run is subtracted as the
    # compliance control rather than expected to read zero.
    raw = [(k, (omega * (len(runs[k]["twist"]) - 1) - runs[k]["twist"][-1]) * 180 / np.pi)
           for k in ks]
    base = raw[0][1]
    slip = [(k, v - base) for k, v in raw]
    if chars:
        oms = sorted(chars)
        ax[1].clear()
        ax[1].semilogx([chars[o]["load"] for o in oms], [chars[o]["bound"] for o in oms], "o-",
                       color="#2b6cb0", lw=1.7)
        for o in oms:
            ax[1].annotate(f"{o:g}", (chars[o]["load"], chars[o]["bound"]), fontsize=6,
                           textcoords="offset points", xytext=(3, 4), color="#666")
        ax[1].set_xlabel("mean load per bond (labels: drive rate $\\omega$)")
        ax[1].set_ylabel("bound fraction, steady state")
        bmax, bmin = max(chars[o]["bound"] for o in oms), min(chars[o]["bound"] for o in oms)
        ax[1].set_title(f"G31, swept in the RIGHT knob: bound fraction falls\n"
                        f"{bmax:.3f} to {bmin:.3f} as the drive loads the bonds", fontsize=8.5)
        ax[3].clear()
        ax[3].semilogx(oms, [chars[o]["follow"] for o in oms], "s-", color="#7a3b9a", lw=1.7)
        ax[3].axhline(1.0, color="#999", ls="--", lw=0.9)
        ax[3].set_xlabel(r"drive rate $\omega$ (rad/frame)")
        ax[3].set_ylabel("fraction of the drive the sheet follows")
        ax[3].set_title("G32: the STALL. Below it the sheet is carried;\n"
                        "above it the bonds cannot hold and it slips", fontsize=8.5)
    ax[1].plot([s[0] for s in slip], [s[1] for s in slip], "o-", color="#1a1a1a", lw=1.7)
    ax[1].set_xlabel(r"$k^0_{off}$")
    ax[1].set_ylabel("slip above the compliance baseline (deg)")
    mono = all(slip[i][1] <= slip[i + 1][1] + 1e-9 for i in range(len(slip) - 1))
    ax[1].set_title(f"G32: slip ABOVE the elastic-shear baseline. Monotone: {mono}.\n"
                    f"The $k_{{off}}=0$ control contributes {base:.3f} deg of compliance",
                    fontsize=8.5)
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("bound fraction")
    ax[2].set_title("G32b: the same knob on the other state --\nfaster cycling holds fewer bonds",
                    fontsize=8.5)
    for c, k in zip(cols, ks):
        r = runs[k]
        dev = max(abs(np.asarray(r["receptor_total"]) / r["receptor_total"][0] - 1.0))
        ax[3].semilogy([k], [max(dev, 1e-18)], "o", color=c, ms=9)
    ax[3].axhline(2.2e-16, color="#999", ls="--", lw=0.9)
    ax[3].set_xlabel(r"$k^0_{off}$")
    ax[3].set_ylabel(r"max $|N_{tot}/N_{tot}(0)-1|$")
    ax[3].set_title("G30 under hard cycling: receptor is still only\nMOVED between columns "
                    "(eps dashed)", fontsize=8.5)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(d, "slip_model.png"), dpi=150, facecolor="white")
    plt.close(fig)


def main():
    def arg(f, c, dflt):
        return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else dflt
    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 200)
    name = arg("--name", str, "05e_slip")
    omega = arg("--omega", float, 0.01)
    d = os.path.join(LOG, name); os.makedirs(d, exist_ok=True)
    cert = BM.selftest(dev=dev, subdiv=4)

    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, sigma_T=7.0, zeta=20.0, s_target=1.0,
             k_drive=50.0, dev=dev)
    A = dict(kappa_b=5.0, k_on=0.6, f_bell=3.0e-3)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 140))).astype(int).tolist())
    # ================= THE CLUTCH CHARACTERISTIC =================
    # G31 and G32 were both swept in the WRONG KNOB and neither could pass or fail honestly.
    # G31 asked for bound fraction vs LOAD and swept f_bell, which is the bond's sensitivity, not the
    # load -- and the load never left 1.5e-4 against an f_bell of 1e-3, so every point sat in the same
    # low-load limit and the four answers differed by 3%. G32 asked for slip vs k_off and swept k_off
    # at a single drive rate that turned out to be PAST THE STALL: every nonzero k_off collapsed the
    # bound fraction to ~0.001 and gave the same 46 deg of slip, so k_off set the timescale and not
    # the steady state.
    # The knob that moves the load is the DRIVE RATE. Sweeping omega over three decades at fixed
    # kinetics traces the clutch's own characteristic -- load, bound fraction and slip together --
    # and the stall is where it turns over. One sweep, both gates.
    chars = {}
    for om in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        r = Rig05d(**P, **A, k_off0=0.05, max_refine=0, reseed=False, omega=om)
        E5.run(r, min(120, frames), label=f"{name}: characteristic omega = {om:g}")
        n = max(1, len(r.res["bound_frac"]) // 4)
        chars[om] = dict(load=float(np.mean(r.res["load_mean"][-n:])),
                         bound=float(np.mean(r.res["bound_frac"][-n:])),
                         twist_deg=r.res["twist"][-1] * 180 / math.pi,
                         drive_deg=om * (len(r.res["twist"]) - 1) * 180 / math.pi,
                         follow=r.res["twist"][-1] / max(om * (len(r.res["twist"]) - 1), 1e-30))

    runs = {}
    for koff in (0.0, 0.02, 0.05, 0.15, 0.5):
        r = Rig05d(**P, **A, k_off0=koff, max_refine=0, reseed=False, omega=omega)
        kp = keep if koff == 0.15 else None
        kept, _ = E5.run(r, frames, keep=kp, label=f"{name}: k_off = {koff:g}")
        runs[koff] = r.res
        if kp is not None:
            write_traj(kept, r.F_epi.cpu().numpy(), d)
            render_from_traj(d, zoom=1.0, l0=r.l0, title=f"{name}: k_off = {koff:g}")
    model_png(runs, omega, d, A, chars)

    ks = sorted(runs)
    slip = {str(k): dict(twist_final_deg=runs[k]["twist"][-1] * 180 / math.pi,
                         drive_deg=omega * (len(runs[k]["twist"]) - 1) * 180 / math.pi,
                         slip_deg=(omega * (len(runs[k]["twist"]) - 1)
                                   - runs[k]["twist"][-1]) * 180 / math.pi,
                         bound_frac_final=runs[k]["bound_frac"][-1],
                         receptor_deviation=float(max(abs(
                             np.asarray(runs[k]["receptor_total"])
                             / runs[k]["receptor_total"][0] - 1.0)))) for k in ks}
    b0 = (omega * (len(runs[ks[0]]["twist"]) - 1) - runs[ks[0]]["twist"][-1]) * 180 / math.pi
    sl = [slip[str(k)]["slip_deg"] - b0 for k in ks]
    out = dict(run=name, frames=frames, omega=omega, certification=cert,
               rig=dict(**{k: v for k, v in P.items() if k != "dev"}, **A),
               G32=dict(elastic_compliance_baseline_deg=float(
                   (omega * (len(runs[ks[0]]["twist"]) - 1) - runs[ks[0]]["twist"][-1])
                   * 180 / math.pi),
                        by_k_off=slip, monotone=bool(all(sl[i] <= sl[i + 1] + 1e-9
                                                         for i in range(len(sl) - 1))),
                        slip_at_zero_deg=sl[0], slip_at_max_deg=sl[-1],
                        archived_discrete_plaque="28.49 deg twist against a 28.50 deg drive with "
                                                 "friction OFF: a pin"),
               G31=dict(characteristic={str(k): v for k, v in chars.items()},
                        note="swept in the DRIVE RATE, which is what moves the load. Sweeping f_bell "
                             "(the bond's sensitivity) left every point in the same low-load limit."),
               series={str(k): {kk: [float(x) for x in vv] for kk, vv in runs[k].items()}
                       for k in ks})
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    yaml.safe_dump(dict(
        what="05e -- slip as a consequence of bond turnover, not as a friction coefficient",
        units=dict(**UNITS, force_nN=None),
        why="a fixed barycentric anchor is a tangential PIN; xi can only set how fast the sheet "
            "equilibrates to one. Slip is k_off releasing bonds that rebind ahead.",
        gates=dict(G32="slip monotone in k_off, zero at k_off = 0",
                   G32b="bound fraction falls as k_off rises",
                   G30="receptor conserved while bonds cycle hard"),
        not_modelled=["a tissue that slides by T1 and division (05k)"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    print(f"[{name}] G32 slip above compliance ({b0:.3f} deg): {sl[0]:.3f} -> {sl[-1]:.3f} deg "
          f"across k_off 0 -> {ks[-1]:g}; monotone {out['G32']['monotone']} -> {d}", flush=True)


if __name__ == "__main__":
    main()
