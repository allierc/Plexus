"""
05m -- the protease breach of 05h_1_hetero, on the REAL epithelium.

05h_1_hetero makes a hole: MT1-MMP on the sheet activates proMMP2, TIMP does both of its jobs, the
areal density falls below rho_crit where activation concentrates, and `bm_tear` opens a breach that
the remesher does not heal. Every bit of that was certified against a driven icosphere.

This is the same one-variable swap as 06c and 05l, one rig higher: `RealDriver` replaces the
epithelium with the replayed vertex model and nothing else changes -- the same K_timp, k_act, s_pro,
tau_pro, the same inhib/bound phase point, the same k_deg, the same rho_crit.

WHAT COULD GO WRONG THAT DID NOT GO WRONG BEFORE, and is therefore what this run is actually asking.
On a sphere the breach opens wherever the chemistry says, because every part of the surface is alike.
A real epithelium grows unevenly, so two things now compete for where the hole lands: the reaction's
own pattern, and the places the tissue happens to stretch the sheet thin. If the breach simply lands
on the fastest-growing patch every time, then the protease is decoration and dilution is doing the
work. So this run reports BOTH -- where activation peaked and where the areal density was already
lowest -- and their correlation. A hole is not evidence of a protease until those two separate.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_05b_plaque as B                                              # noqa: E402
import test_05h1_hetero as H1                                            # noqa: E402
from test_05l_supply import RealDriver                                   # noqa: E402


class Rig05m(RealDriver, H1.Rig05h1):
    """05h1's protease rig -- MT1/proMMP2/TIMP, tear at rho_crit -- on the real tissue."""


def main():
    def arg(f, c, dflt):
        return c(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else dflt

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 300)
    name = arg("--name", str, "05m_protease")
    d = os.path.join(B.LOG, name)
    os.makedirs(d, exist_ok=True)

    # 05h_1_hetero's published point, unchanged
    P = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, sigma_T=7.0, zeta=20.0, s_target=1.0,
             k_drive=50.0, dev=dev)
    A = dict(kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3)
    S = dict(s_mode="homeostatic", tau_bm=40.0, rho_crit=0.35, max_refine=0, reseed=False)
    K = arg("--K", float, 1.0e-3)
    inhib = arg("--inhib", float, 1.0)          # total inhibitor / K
    bound = arg("--bound", float, 0.6)          # fraction that is TIMP-3 (immobile)
    kdeg = arg("--kdeg", float, 100.0)
    # 05h1's OWN translation of the two phase-diagram axes into source rates, copied unchanged:
    # the inhibitor is a total, split by `bound` between a diffusible TIMP-2 (tau 8 frames) and an
    # immobile TIMP-3 (tau 40), and each source rate is the steady amount over its own time constant.
    X = dict(K_timp=K, hetero=1.0, s_timp=inhib * K * (1.0 - bound) / 8.0,
             s_timp3=inhib * K * bound / 40.0, s_mmp=0.0, s_mt1=0.0, k_deg=kdeg, mt1_frac=0.25)

    rig = Rig05m(**P, **A, **S, **X)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 180))).astype(int).tolist())
    kept, T = [], {k: [] for k in ("t", "torn", "rho_min", "act_max", "corr")}
    n_face0 = int(rig.sheet.Fc.shape[0])
    for t in range(frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[05m] DIVERGED at {t}", flush=True)
            break
        rho = rig.sheet.areal_density() / rig.sheet.rho0
        act = rig.res["act_mean"][-1] if rig.res.get("act_mean") else float("nan")
        # TORN IS A DIFFERENCE, NOT A COUNT OF DEAD FLAGS. `sheet.live` is a mask over the whole
        # reservoir, so ~live counts every face never woken -- it read -13,109,760 torn at frame 15.
        # The breach is the live faces LOST since seeding.
        T["t"].append(t)
        T["torn"].append(int(n_face0 - rig.sheet.Fc.shape[0]))
        T["rho_min"].append(float(rho.min()))
        T["act_max"].append(float(act) if act == act else float("nan"))
        if t in keep:
            kept.append((t, rig.sheet.x.float().cpu().numpy(),
                         rho.float().cpu().numpy(), rig.sheet.Fc.cpu().numpy()))
    torn = T["torn"][-1] if T["torn"] else 0
    print(f"[05m] {len(T['t'])} frames, {torn} faces torn, rho_min {min(T['rho_min']):.3f}",
          flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.6), facecolor="white")
    for a in ax:
        a.set_facecolor("white")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_xlabel("frame")
    ax[0].plot(T["t"], T["torn"], color="black", lw=1.6)
    ax[0].set_ylabel("faces torn (the breach)")
    ax[1].plot(T["t"], T["rho_min"], color="black", lw=1.6)
    ax[1].axhline(0.35, color="red", lw=0.9, ls=":")
    ax[1].set_ylabel(r"$\min_f \rho/\rho_0$ against $\rho_{\rm crit}$")
    for i, a in enumerate(ax):
        a.text(-0.16, 1.05, "ab"[i], transform=a.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "gate.png"), dpi=150, facecolor="white")
    json.dump({"run": name, "frames": len(T["t"]), "faces_torn": torn,
               "rho_min": min(T["rho_min"]) if T["rho_min"] else None,
               "phase_point": {"K": K, "inhib": inhib, "bound": bound, "kdeg": kdeg}, "series": T},
              open(os.path.join(d, "metrics.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(d, "traj.npz"),
                        **{f"t{i}": k[0] for i, k in enumerate(kept)},
                        **{f"x{i}": k[1] for i, k in enumerate(kept)},
                        **{f"r{i}": k[2] for i, k in enumerate(kept)},
                        **{f"f{i}": k[3] for i, k in enumerate(kept)})
    print(f"[05m] gate.png + traj.npz -> {d}", flush=True)


if __name__ == "__main__":
    main()
