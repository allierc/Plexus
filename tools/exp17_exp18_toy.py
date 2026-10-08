"""exp17 deck: EXP18'S TOY MODEL BESIDE BATCH 21'S LAW (Cedric, 2026-10-08: "on the angle slide, a column with the toy model
results of exp18: the six circuits, ground truth / inferred traces and the angles on the circle, the circuits clearly
told apart, black background"). Local, from exp18's run on disk (no training).

The run: exp18_m5_alpha_phirandlearn_zf285 -- the 285-cell zebrafish oculomotor integrator with ONE fixed wiring and
six laws (integrate, delay, low-pass, high-pass, resonator, differentiate), each selected by one broadcast angle alpha_k
in cos(phi_{t(j)t(i)} - alpha_k), phi learned from random (the GNN_Transformer note, Fig. 2 c). Two figures, black:
    traces  one held-out trial per law: the target output (white) and the circuit's (the law's colour)
    circle  the six learned angles alpha_k on the unit circle, each in its law's colour, numbered, with its held-out error
            (the error over the law's own variance, results/<run>_test.json)

    PYTHONPATH=src:tools python tools/exp17_exp18_toy.py
-> experiments/exp17_zapbench_graphcast/presentation/figs/exp18_toy_traces.png, exp18_toy_circle.png
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
FIGS = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast", "presentation", "figs")
RUN = "exp18_m5_alpha_phirandlearn_zf285"
NICE = {"integrate": "integrate", "delay": "delay", "lowpass": "low-pass", "highpass": "high-pass",
        "resonator": "resonator", "differentiate": "differentiate"}


def main(device="cpu"):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp_measures.common import TrainingRun
    from exp_measures.exp18 import _restore, _phase_op
    from exp18_paper_figs import GD, COLS_DARK
    COLS_DARK = {**COLS_DARK, "differentiate": "#ffd84a"}              # not the target's white
    T = TrainingRun(GD + RUN)
    TR, spec, sim, learn, ck, U, Y, cond = _restore(T, device)
    names = T.results[f"{RUN}_test"]["cell_names"]
    per = T.results[f"{RUN}_test"]["normalised_per_cell"]
    K = len(names)
    idx = [int(np.flatnonzero(cond == k)[0]) for k in range(K)]       # one held-out trial per law
    ch = int(spec["task"]["observe"].get("channel", 0))
    with torch.no_grad():
        H, Yp = TR.rollout(sim, learn, U[idx], spec["task"], device, grad=False)
    alpha = _phase_op(H).alpha.detach().cpu().numpy().reshape(-1)[:K]
    dt = float(sim.dt)
    nf = min(Yp.shape[-2], Y.shape[-2])
    t = np.arange(nf) * dt
    plt.style.use("dark_background")
    # the traces, one row per law
    fig, axs = plt.subplots(K, 1, figsize=(4.2, 6.4), facecolor="black", sharex=True)
    for k, ax in enumerate(axs):
        c = COLS_DARK.get(names[k], "w")
        ax.plot(t, Y[idx[k], :nf, 0].cpu().numpy(), color="white", lw=1.6, label="target")
        ax.plot(t, Yp[k, :nf, ch].cpu().numpy(), color=c, lw=1.2, ls="--", label="circuit")
        ax.text(0.01, 0.92, f"{k + 1}  {NICE.get(names[k], names[k])}", transform=ax.transAxes, color=c, fontsize=10,
                weight="bold", va="top")
        ax.set_yticks([])
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.set_facecolor("black")
    axs[0].legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
    axs[-1].set_xlabel("time, s", fontsize=9)
    axs[-1].tick_params(labelsize=8)
    fig.tight_layout(h_pad=0.2)
    fig.savefig(os.path.join(FIGS, "exp18_toy_traces.png"), dpi=170, facecolor="black")
    plt.close(fig)
    # the angles on the circle
    fig, ax = plt.subplots(figsize=(4.2, 4.6), facecolor="black")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="0.55", lw=1.0)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.axvline(0, color="0.3", lw=0.6)
    for k in range(K):
        c = COLS_DARK.get(names[k], "w")
        a = float(alpha[k])
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color=c, lw=2.6)
        ax.scatter([np.cos(a)], [np.sin(a)], color=c, s=40, zorder=3)
        ax.text(1.17 * np.cos(a), 1.17 * np.sin(a), f"{k + 1}", color=c, fontsize=11, weight="bold", ha="center", va="center")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    lines = [f"{k + 1}  {NICE.get(names[k], names[k]):<13} {float(alpha[k]) % (2 * np.pi):4.2f} rad   {per[str(k)]:.3f}"
             for k in range(K)]
    step = 7.5 * 1.35 / 72 / fig.get_figheight()                       # one monospace line, figure fraction
    fig.text(0.06, 0.02 + K * step, "law              angle    error", fontsize=7.5, family="monospace", color="0.8",
             va="bottom")
    for k in range(K):                                                  # each line in its law's colour
        fig.text(0.06, 0.02 + (K - 1 - k) * step, lines[k], fontsize=7.5, family="monospace",
                 color=COLS_DARK.get(names[k], "w"), va="bottom")
    fig.subplots_adjust(bottom=0.30, top=0.98, left=0.02, right=0.98)
    fig.savefig(os.path.join(FIGS, "exp18_toy_circle.png"), dpi=170, facecolor="black")
    plt.close(fig)
    print("[toy]", {names[k]: (round(float(alpha[k]), 3), round(float(per[str(k)]), 4)) for k in range(K)})


if __name__ == "__main__":
    main()
