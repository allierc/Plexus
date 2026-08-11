"""
One folder per gate: 05k_G40 .. 05k_G46, each with its own gate.png and a movie.mp4.

The five gates are read off ONE run -- 05b's rig with the real-tissue driver -- because they are five
readouts of the same experiment, not five experiments. Saying that plainly matters: the movie in
G41's folder and the movie in G42's folder are the same 401 frames, and what differs between the
folders is the measurement drawn from them. G40 alone needs more than one run, and its three extra
stiffnesses are read from 05k_gates/metrics.json rather than recomputed.

Each figure is the ONE curve its gate is about, with its threshold drawn on it: white ground, no box,
no title, the letter in the corner. A gate you cannot watch is a number someone has to trust.
"""
import json
import math
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_05b_plaque as B                                              # noqa: E402
from test_05k_gates import quality, tri_area                             # noqa: E402
from test_06c_real_driver import Rig06c                                  # noqa: E402


def axes(ylab, xlab="frame"):
    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor="white")
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return fig, ax


def save(fig, ax, letter, d, verdict):
    ax.text(-0.17, 1.05, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.text(0.98, 1.05, verdict, transform=ax.transAxes, fontsize=10, ha="right",
            color=("green" if verdict.startswith("PASS") else "red"))
    fig.tight_layout()
    fig.savefig(os.path.join(d, "gate.png"), dpi=150, facecolor="white")
    plt.close(fig)


def main():
    def arg(flag, cast, default):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dev = arg("--device", str, "cuda:0" if torch.cuda.is_available() else "cpu")
    frames = arg("--frames", int, 401)

    T = 2.0e-3
    P = dict(subdiv=4, subdiv_epi=3, E=400.0, thickness=T, nu=0.3, kn=5.0, xi=0.0,
             l0=0.3 * T, zeta=20.0, s_target=1.0, k_drive=50.0, dev=dev)
    keep = set(np.round(np.linspace(0, frames - 1, min(frames, 160))).astype(int).tolist())

    rig = Rig06c(**P)
    e0 = float((rig.sheet.x[rig.sheet.Ed[:, 1]] - rig.sheet.x[rig.sheet.Ed[:, 0]]).norm(dim=1).mean())
    A0 = float(tri_area(rig.x_epi, rig.F_epi).sum())
    S = {k: [] for k in ("t", "lam", "area", "q", "edge", "gap")}
    kept = []
    for t in range(frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[05k] DIVERGED at {t}", flush=True)
            break
        X = rig.sheet.x
        l1, _ = rig.sheet.stretch_geo()
        S["t"].append(t)
        S["lam"].append(float(l1.mean()))
        S["area"].append(math.sqrt(float(tri_area(rig.x_epi, rig.F_epi).sum()) / A0))
        S["q"].append(float(quality(X, rig.sheet.Fc).min()))
        S["edge"].append(float((X[rig.sheet.Ed[:, 1]] - X[rig.sheet.Ed[:, 0]]).norm(dim=1).mean()) / e0)
        S["gap"].append(float((X - rig.c).norm(dim=1).mean()
                              - (rig.x_epi[:rig.nv0] - rig.c).norm(dim=1).mean()))
        if t in keep:
            kept.append((t, X.float().cpu().numpy(), l1.float().cpu().numpy(),
                         rig.x_epi.float().cpu().numpy(),
                         rig.plq.node[rig.plq.bound].cpu().numpy(),
                         (rig.x_epi[rig.F_epi[rig.plq.face]] * rig.plq.w[:, :, None]).sum(1)
                         [rig.plq.bound].float().cpu().numpy()))
    print(f"[05k] {len(S['t'])} frames, {len(kept)} kept", flush=True)

    # the movie once, into a scratch folder, then copied into each gate's folder
    scratch = os.path.join(B.LOG, "05k_gates")
    os.makedirs(scratch, exist_ok=True)
    s_hi = float(np.percentile(np.concatenate([k[2] for k in kept[::4]]), 99))
    B.render(kept, rig.sheet.Fc.cpu().numpy(), rig.F_epi.cpu().numpy(), scratch,
             "05q real driver", s_hi)
    mp4 = os.path.join(scratch, "movie.mp4")

    sweep = {2.5: 3.6412, 5.0: 3.7134, 10.0: 3.7689, 20.0: 3.8173}       # from 05k_gates
    lam, ar = S["lam"][-1], S["area"][-1]
    v40 = (max(sweep.values()) - min(sweep.values())) / np.mean(list(sweep.values()))
    v41 = abs(lam - ar) / ar
    spec = [
        ("05k_G40_stiffness", "a", v40 < 0.05,
         f"PASS {v40*100:.2f}%" if v40 < 0.05 else f"FAIL {v40*100:.2f}%"),
        ("05k_G41_tracking", "b", v41 < 0.05,
         f"PASS {v41*100:.2f}%" if v41 < 0.05 else f"FAIL {v41*100:.2f}%"),
        ("05k_G42_quality", "c", min(S["q"]) > 0.2,
         f"PASS {min(S['q']):.3f}" if min(S["q"]) > 0.2 else f"FAIL {min(S['q']):.3f}"),
        ("05k_G44_refine_demand", "d", 0.8 <= S["edge"][-1] <= 1.7,
         f"FAIL {S['edge'][-1]:.2f}x"),
        ("05k_G46_standoff", "e", min(S["gap"]) > 0.0,
         f"FAIL {min(S['gap']):.2e}"),
    ]
    for folder, letter, ok, verdict in spec:
        d = os.path.join(B.LOG, folder)
        os.makedirs(d, exist_ok=True)
        if folder.endswith("G40_stiffness"):
            fig, ax = axes(r"$\lambda^{\rm geo}$ at the last frame", r"$\kappa_n$")
            ax.plot(list(sweep), list(sweep.values()), "o-", color="black", lw=1.0)
            ax.set_xscale("log")
        elif folder.endswith("G41_tracking"):
            fig, ax = axes(r"$\lambda^{\rm geo}$ (black), $\sqrt{A_{\rm ep}/A_{\rm ep}(0)}$ (green)")
            ax.plot(S["t"], S["lam"], color="black", lw=1.6)
            ax.plot(S["t"], S["area"], color="green", lw=1.2, ls="--")
        elif folder.endswith("G42_quality"):
            fig, ax = axes("worst triangle quality")
            ax.plot(S["t"], S["q"], color="black", lw=1.4)
            ax.axhline(0.2, color="red", lw=0.9, ls=":")
            ax.set_ylim(0.0, 1.05)
        elif folder.endswith("G44_refine_demand"):
            fig, ax = axes("mean edge / seeded")
            ax.plot(S["t"], S["edge"], color="black", lw=1.6)
            ax.axhspan(0.8, 1.7, color="green", alpha=0.12)
        else:
            fig, ax = axes("mean radial gap sheet $-$ epithelium (box units)")
            ax.plot(S["t"], S["gap"], color="black", lw=1.6)
            ax.axhline(0.0, color="red", lw=0.9, ls=":")
        save(fig, ax, letter, d, verdict)
        shutil.copy(mp4, os.path.join(d, "movie.mp4"))
        json.dump({"gate": folder, "verdict": verdict, "pass": bool(ok),
                   "movie": "the same 401-frame run in every folder; the gates are five readouts "
                            "of one experiment", "series": S},
                  open(os.path.join(d, "metrics.json"), "w"), indent=1)
        print(f"[05k] {folder}: {verdict}", flush=True)


if __name__ == "__main__":
    main()
