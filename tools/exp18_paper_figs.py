"""Figure 1 of GNN_Transformer.tex Part V: the learned angles, the circuits they make, and the activity they drive.

    PYTHONPATH=src:tools /workspace/.conda_envs/neural-graph-linux/bin/python tools/exp18_paper_figs.py

Three runs of exp18 (columns): two laws with varphi learned from 0; six laws with varphi learned from 0; six laws with
varphi learned from random. Three rows:

    a-c  the broadcast angles alpha_k on S^1, one line per law, with its held-out error (results/<run>_test.json)
    d-f  the eigenvalues of the coupling Jacobian J(alpha_k) = diag(G) [W o cos(Phi - alpha_k)] diag(rho'_k) of each
         law (GNN_Transformer.tex Eq. exp_J), rho'_k = 1 - tanh(v)^2 averaged over ONE trial driven under law k's
         context -- each law linearised at its own operating point; the three largest real parts drawn large
    g-i  the frequency response of the circuit under each law's angle, |H(f)| = |sum Y conj U / sum |U|^2| over the
         law's held-out trials (solid), and of the target law through the same estimate (dashed), up to 2 Hz

Every number is computed here from the trained checkpoints (`models/best.pt`) through the rulers' own restore
(`exp_measures.exp18._restore`), in the devcontainer.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
GD = "/groups/saalfeld/home/allierc/GraphData/log/training/neural/"
OUT = os.path.join(ROOT, "papers", "figs_gnn_transformer", "exp18_angles")
RUNS = [("exp18_m3_alpha_zf285", "two laws, $\\varphi$ learned from 0"),
        ("exp18_m5_alpha_zf285", "six laws, $\\varphi$ learned from 0"),
        ("exp18_m5_alpha_phirandlearn_zf285", "six laws, $\\varphi$ learned from random")]
F_MAX = 2.0     # Hz: the stimulus is band-limited noise up to 2 Hz (config/task/m*_*.yaml)
COLS = {"integrate": "#1f4e9c", "delay": "#c0392b", "lowpass": "#8e44ad", "highpass": "#e67e22",
        "resonator": "#16a085", "differentiate": "#555555"}


def circuit(run: str, device: str):
    """Per law: its angle, J(alpha_k) eigenvalues, and the voltages [T, N] under one shared stimulus."""
    import torch
    from exp_measures.common import TrainingRun
    from exp_measures.exp18 import _restore, _phase_op, coupling
    T = TrainingRun(GD + run)
    TR, spec, sim, learn, ck, U, Y, cond = _restore(T, device)
    names = T.results[f"{run}_test"]["cell_names"]
    per = T.results[f"{run}_test"]["normalised_per_cell"]
    K = len(names)
    u0 = U[int(np.flatnonzero(cond == 0)[0])].clone()                 # one stimulus trace, [T, 1 + K]
    # THE FREQUENCY RESPONSE per law from its OWN held-out trials: H(f) = sum Y(f) conj U(f) / sum |U(f)|^2 over the
    # trials, for the circuit's output and for the target alike, at the FFT bins inside the stimulus band.
    n = min(int(spec["task"]["reference"].get("n_test", 24)), U.shape[0])
    ch = int(spec["task"]["observe"].get("channel", 0))
    with torch.no_grad():
        _, Yp = TR.rollout(sim, learn, U[:n], spec["task"], device, grad=False)
    nf = min(Yp.shape[-2], Y.shape[-2])
    dt = float(sim.dt)
    freqs = np.fft.rfftfreq(nf, dt)
    band = (freqs > 0) & (freqs <= F_MAX)
    resp = {}
    for k in range(K):
        idx = np.flatnonzero(cond[:n] == k)
        u = U[idx, :nf, 0].cpu().numpy()
        Fu = np.fft.rfft(u - u.mean(1, keepdims=True), axis=1)
        H = {}
        for tag, y in (("circuit", Yp[idx, :nf, ch].cpu().numpy()), ("target", Y[idx, :nf, 0].cpu().numpy())):
            Fy = np.fft.rfft(y - y.mean(1, keepdims=True), axis=1)
            H[tag] = (Fy * Fu.conj()).sum(0) / (np.abs(Fu) ** 2).sum(0)
        resp[k] = {"f": freqs[band], "circuit": np.abs(H["circuit"][band]), "target": np.abs(H["target"][band])}
    out = []
    for k in range(K):
        u = u0.clone()
        u[:, 1:1 + K] = 0.0
        u[:, 1 + k] = 1.0                                              # only the context changes
        vs = []
        with torch.no_grad():
            H, _ = TR.rollout(sim, learn, u, spec["task"], device, grad=False,
                              watch=lambda H: vs.append(H.level("neuron").get("voltage").squeeze(-1).clone()))
        op = _phase_op(H)
        lvl, es = H.level("neuron"), H.level(op.edge_set)
        n = lvl.n
        w = es.get("w").detach().reshape(-1)
        if op.dale:
            w = w.abs() * op._dale_sign(lvl, es).reshape(-1).to(w.dtype)
        Wm = np.zeros((n, n))
        Wm[es.post.cpu().numpy(), es.pre.cpu().numpy()] = w.cpu().numpy()
        nt = lvl.node_type.cpu().numpy()
        g = lvl.type_params[lvl.node_type][:, 2].detach().cpu().numpy()
        V = torch.stack(vs).cpu().numpy()                              # [T, N]
        slope = (1.0 - np.tanh(V) ** 2).mean(0)
        a = op.alpha.detach().cpu().numpy()[k]
        J = coupling(Wm, op.phi.detach().cpu().numpy(), nt, nt, a, g, slope)
        out.append({"name": names[k], "alpha": float(a), "err": float(per[str(k)]),
                    "eig": np.linalg.eigvals(J), "V": V, "H": resp[k]})
    return out


COLS_DARK = {"integrate": "#5b8ff9", "delay": "#ff6b5b", "lowpass": "#c38df0", "highpass": "#ffa14a",
             "resonator": "#3ddbb5", "differentiate": "#d0d0d0"}
DECK = os.path.join(ROOT, "experiments", "exp18_phase_modulation", "presentation", "figs")


def draw(ax, kind, c, title, laws, dark=False):
    """One panel: kind 'angles' / 'eig' / 'freq', column c (letter), for one run's laws."""
    from matplotlib.lines import Line2D
    cols = COLS_DARK if dark else COLS
    grey, light, edge = ("0.55", "0.3", "white") if dark else ("0.6", "0.88", "black")
    if kind == "angles":
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), color=grey, lw=1)
        ax.plot([-1.05, 1.05], [0, 0], color=light, lw=0.6)
        ax.plot([0, 0], [-1.05, 1.05], color=light, lw=0.6)
        hs = []
        for L in laws:
            ax.plot([0, np.cos(L["alpha"])], [0, np.sin(L["alpha"])], color=cols[L["name"]], lw=2.2)
            hs.append(Line2D([0], [0], color=cols[L["name"]], lw=2.2,
                             label=f"{L['name']:<13} {L['alpha']:+.2f} rad   {L['err']:.4f}"))
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.axis("off")
        ax.set_title(f"{'abc'[c]}   {title}", fontsize=9.5, loc="left")
        ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.0), frameon=False,
                  prop={"family": "monospace", "size": 7},
                  title="law, $\\alpha_k$, held-out error / own variance", title_fontsize=7)
    elif kind == "eig":
        for L in laws:
            e = L["eig"]
            ax.scatter(e.real, e.imag, s=4, color=cols[L["name"]], alpha=0.35, lw=0)
            top = e[np.argsort(-e.real)[:3]]
            ax.scatter(top.real, top.imag, s=42, color=cols[L["name"]], edgecolor=edge, lw=0.5, zorder=5)
        ax.axvline(0, color=light, lw=0.6)
        ax.axhline(0, color=light, lw=0.6)
        ax.set_xlabel(r"Re $\lambda$ of $J(\alpha_k)$ (1/s)", fontsize=8)
        ax.set_ylabel(r"Im $\lambda$ (1/s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{'def'[c]}   eigenvalues of $J(\\alpha_k)$; large: three leading", fontsize=9, loc="left")
    else:
        for L in laws:
            ax.loglog(L["H"]["f"], L["H"]["target"], color=cols[L["name"]], lw=1.0, ls="--", alpha=0.8)
            ax.loglog(L["H"]["f"], L["H"]["circuit"], color=cols[L["name"]], lw=1.8)
        ax.set_xticks([0.2, 0.5, 1.0, 2.0])
        ax.set_xticklabels(["0.2", "0.5", "1", "2"])
        ax.minorticks_off()
        ax.set_xlabel("frequency (Hz)", fontsize=8)
        ax.set_ylabel("gain $|H(f)|$, output / stimulus", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{'ghi'[c]}   frequency response: circuit (solid), target law (dashed)", fontsize=9, loc="left")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    data = [circuit(r, dev) for r, _ in RUNS]
    # THE PAPER: one 3 x 3 figure, white
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 12.6), gridspec_kw={"height_ratios": [1.25, 1, 1]})
    for c, ((run, title), laws) in enumerate(zip(RUNS, data)):
        for r, kind in enumerate(("angles", "eig", "freq")):
            draw(axes[r, c], kind, c, title, laws)
    fig.tight_layout()
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + ".png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    # THE DECK: one row per slide, black (Cedric 2026-10-04)
    os.makedirs(DECK, exist_ok=True)
    with plt.style.context("dark_background"):
        for kind, h in (("angles", 5.2), ("eig", 3.9), ("freq", 3.9)):
            fig, axes = plt.subplots(1, 3, figsize=(12.0, h), facecolor="black")
            for c, ((run, title), laws) in enumerate(zip(RUNS, data)):
                draw(axes[c], kind, c, title, laws, dark=True)
            fig.tight_layout()
            fig.savefig(os.path.join(DECK, f"fig1_{kind}.png"), dpi=150, facecolor="black", bbox_inches="tight")
            plt.close(fig)
    # the numbers the slides' conclusion lines quote, computed here (the one rule)
    summary = {}
    for (run, _), laws in zip(RUNS, data):
        a = np.array([L["alpha"] for L in laws])
        d = np.abs(np.angle(np.exp(1j * (a[:, None] - a[None, :]))))
        np.fill_diagonal(d, np.inf)
        g = [float(np.sqrt(np.mean((np.log(L["H"]["circuit"]) - np.log(L["H"]["target"])) ** 2))) for L in laws]
        summary[run] = {"n": len(laws), "worst": max(L["err"] for L in laws), "closest_pair_rad": float(d.min()),
                        "lead_re": [float(L["eig"].real.max()) for L in laws], "gain_log_rms_max": max(g),
                        "names": [L["name"] for L in laws], "errs": [L["err"] for L in laws]}
    json.dump(summary, open(os.path.join(DECK, "fig1_summary.json"), "w"), indent=1)
    print(json.dumps({r: {k: v for k, v in x.items() if k in ("worst", "closest_pair_rad", "gain_log_rms_max")} for r, x in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
