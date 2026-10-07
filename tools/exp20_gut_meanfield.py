"""exp20: THE GUT RESPONSE, THE GRAPH AGAINST ONE BRAIN-WIDE SIGNAL (Cedric, 2026-10-06: "after slides 26-27, show the
difference in gut response between brain-wide and not brain-wide"). Batch 7 on glucose fish 1: per arm (A the atlas's
area postrema + vagal ganglia, B + the dorsal vagal complex) the 3-level-mesh graph, its mean-field twin (every cell
hears the same brain mean of tanh z, through its own gain), its no-W twin, and the graph with W = 0 at inference.

    PYTHONPATH=src:tools python tools/exp20_gut_meanfield.py [--device cuda:0]

Each law's free rollout of the whole session (tools/exp20_freetrial.py's `rollout_windows`, driven by the stimuli
only), read around the 12 gut pulses (every full-window pulse off site 1):
  traces  the gut-responsive cells' mean dF/F (baselines `responsive`, 2,538 cells), 10 s before to 55 s after the
          pulse, mean and SEM over the pulses -- recorded, graph, graph W = 0, mean field, no W
  maps    every cell's evoked change, 0-20 s after the pulse minus the 10 s before, mean over the gut pulses --
          recorded, graph, mean field -- from above, head left
Writes data/gut_meanfield_b7.npz and presentation/figs/gut_meanfield_b7.png.
"""
import argparse
import contextlib
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
ARMS = (("A", "area postrema + vagal ganglia", "gb_b7_glucose_f1_mesh3_anat_apvg"),
        ("B", "area postrema + vagal ganglia + dorsal vagal complex", "gb_b7_glucose_f1_mesh3_anat_dvc"))
CTRL_SITE = 1


def law_rollouts(name, device, w0=False):
    """{arm: {onset: pred [pre + post + 1, N]}} for the law `name` (and its W = 0 when `w0`), the recording's X, the
    gut-pulse onsets and the window."""
    from plexus import engine
    from plexus import trainer as T
    import exp17_ablation as A
    import exp20_freetrial as FT
    engine.quiet(True)
    spec = T.load(name)
    ck = torch.load(os.path.join(T.out_dir(spec, None), "models", "best.pt"), weights_only=False, map_location=device)
    learn = T.Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = T._trace_setup(spec, device)
    X, rec = box["X"], box["rec"]
    dt = float(np.median(np.diff(rec["t_s"])))
    pre, post, ev = int(round(FT.PRE_S / dt)), int(np.ceil(FT.POST_S / dt)), int(round(FT.EVOKED_S / dt))
    tr = rec["trials"]
    gut = [int(f) for f, s, full in zip(tr[:, 0], tr[:, 2], tr[:, 5]) if full and int(s) != CTRL_SITE]
    windows = {f: (f - pre, f + post) for f in gut}
    keys = {e["param"][2:]: T.Learnables.key(e) for e in spec["learnable"]
            if e.get("op") == "state_diffuse" and str(e.get("param", "")).startswith("W_")}
    arms = {"full": contextlib.nullcontext()}
    if w0 and keys:
        arms["W0"] = A.weights_zeroed(learn, keys, {s: None for s in keys})
    out = {}
    for arm, cm in arms.items():
        with cm:
            out[arm], _ = FT.rollout_windows(T, spec, learn, box, device, windows)
    return out, X, gut, (pre, post, ev), dt


def evoked(W, pre, ev):
    """[N]: 0-ev frames after the onset minus the pre frames before, mean over the pulses; W [P, pre + post + 1, N]."""
    return (W[:, pre:pre + ev].mean(1) - W[:, :pre].mean(1)).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    resp = np.load(os.path.join(EXP, "data", "baselines_gutbrain_glucose_f1_cells.npz"))["responsive"]
    ri = torch.as_tensor(np.where(resp)[0], device=a.device)
    D = {}
    for arm, _, g in ARMS:
        for kind, run in (("graph", g), ("mean field", g.replace("mesh3", "mf")), ("no W", g.replace("mesh3", "now"))):
            R, X, gut, (pre, post, ev), dt = law_rollouts(run, a.device, w0=(kind == "graph"))
            if "rec" not in D:
                Wr = torch.stack([X[f - pre:f + post + 1] for f in gut])                    # [P, L, N]
                D["rec_evoked"] = evoked(Wr, pre, ev).cpu().numpy()
                tr_ = Wr[:, :, ri].mean(2)                                                  # [P, L]
                D["rec_trace"], D["rec_sem"] = tr_.mean(0).cpu().numpy(), (tr_.std(0) / np.sqrt(len(gut))).cpu().numpy()
                D["t_s"] = (np.arange(-pre, post + 1)) * dt
                D["rec"] = True
            for sub, preds in R.items():
                lab = f"{arm} {kind}" + (", W = 0" if sub == "W0" else "")
                W = torch.stack([preds[f] for f in gut])
                tr_ = W[:, :, ri].mean(2)
                D[f"{lab}|trace"] = tr_.mean(0).cpu().numpy()
                D[f"{lab}|sem"] = (tr_.std(0) / np.sqrt(len(gut))).cpu().numpy()
                D[f"{lab}|evoked"] = evoked(W, pre, ev).cpu().numpy()
                e_r = D["rec_evoked"]
                e_l = D[f"{lab}|evoked"]
                ok = np.isfinite(e_l)
                print(f"[gut] {lab}: gut-responsive cells' evoked {float(np.nanmean(e_l[resp])):+.4f} "
                      f"(recorded {float(e_r[resp].mean()):+.4f}); map r with the recorded over every cell "
                      f"{np.corrcoef(e_l[ok], e_r[ok])[0, 1]:+.2f}", flush=True)
            del R
            torch.cuda.empty_cache()
    D.pop("rec")
    np.savez_compressed(os.path.join(EXP, "data", "gut_meanfield_b7.npz"), responsive=resp, n_pulses=len(gut), **D)
    figure(D, resp, len(gut))


def figure(D, resp, n_pulses):
    """Cedric, 2026-10-06: each arm's maps under its own trace panel, the recorded map on the left under the legend;
    both panels read the SAME gut-responsive cells, the arms differ only in where the UV input enters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plexus.tasks import trace_recording as TR
    P = np.asarray(TR.load("gutbrain_glucose_f1")["pos_view"], np.float64)                  # head left (drawing only)
    o = np.argsort(P[:, 2])
    t = D["t_s"]
    COL = {"graph": "white", "graph, W = 0": "#ff5252", "mean field": "#ffa726", "no W": "#42a5f5"}
    LAB = {"graph": "graph", "graph, W = 0": "graph, W = 0", "mean field": "mean field", "no W": "no W"}
    NIN = {"A": ("527", "area postrema + vagal ganglia"), "B": ("2,037", "area postrema + vagal ganglia + dorsal vagal complex")}
    vmax = float(np.percentile(np.abs(D["rec_evoked"]), 99.5))
    fig = plt.figure(figsize=(16, 8.0), facecolor="black")
    gs = fig.add_gridspec(2, 3, width_ratios=[0.62, 1, 1], height_ratios=[1.0, 0.55], hspace=0.34, wspace=0.20)

    def brain(ax, e, title):
        ax.set_facecolor("black"); ax.axis("off"); ax.set_aspect("equal")
        im_ = ax.scatter(P[o, 0], P[o, 1], c=np.nan_to_num(e[o]), s=0.25, cmap="coolwarm", vmin=-vmax, vmax=vmax, lw=0)
        ax.set_title(title, color="white", fontsize=10)
        return im_
    # left column: the legend, the recorded map under it
    axl = fig.add_subplot(gs[0, 0]); axl.axis("off")
    hs = [Line2D([], [], color="#4caf50", lw=2.4)] + [Line2D([], [], color=COL[k], lw=1.8, ls="--" if "W = 0" in k else "-")
                                                      for k in COL]
    axl.legend(hs, ["recorded"] + [LAB[k] for k in COL], loc="upper left", frameon=False, labelcolor="white", fontsize=10)
    axl.text(0.0, 0.0, f"both panels: the same {int(resp.sum()):,}\ngut-responsive cells (the\npaper's rule); the arms differ\n"
             "only in where the UV input\nenters. Mean field: every cell\nhears one brain-wide signal", color="0.8", fontsize=9,
             transform=axl.transAxes, va="bottom")
    im = brain(fig.add_subplot(gs[1, 0]), D["rec_evoked"], "recorded")
    for j, (arm, _, _) in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, 1 + j])
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_color("0.6")
        ax.tick_params(colors="0.85", labelsize=8)
        ax.fill_between(t, D["rec_trace"] - D["rec_sem"], D["rec_trace"] + D["rec_sem"], color="#4caf50", alpha=0.25, lw=0)
        ax.plot(t, D["rec_trace"], color="#4caf50", lw=2.2)
        for kind in COL:
            k = f"{arm} {kind}"
            if f"{k}|trace" in D:
                ax.plot(t, D[f"{k}|trace"], color=COL[kind], lw=1.8 if kind == "graph" else 1.4, ls="--" if "W = 0" in kind else "-")
        ax.axvline(0, color="#ffd54f", lw=0.8, ls=":")
        ax.set_title(f"arm {arm}: the UV input enters {NIN[arm][0]} cells\n{NIN[arm][1]}", color="white", fontsize=10.5, loc="left")
        ax.set_xlabel("s from a gut pulse", color="0.9", fontsize=9)
        if j == 0:
            ax.set_ylabel(f"mean dF/F, gut-responsive cells\n(mean $\\pm$ SEM over {n_pulses} gut pulses)", color="0.9", fontsize=9)
        sub = gs[1, 1 + j].subgridspec(1, 2, wspace=0.04)
        brain(fig.add_subplot(sub[0, 0]), D[f"{arm} graph|evoked"], f"arm {arm}, graph")
        brain(fig.add_subplot(sub[0, 1]), D[f"{arm} mean field|evoked"], f"arm {arm}, mean field")
    cax = fig.add_axes([0.905, 0.09, 0.007, 0.22])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(colors="0.85", labelsize=8)
    cb.set_label("dF/F change, 0-20 s after a gut\npulse minus the 10 s before", color="0.9", fontsize=7.5)
    out = os.path.join(EXP, "presentation", "figs", "gut_meanfield_b7.png")
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("[gut]", out)


if __name__ == "__main__":
    if "--figure" in sys.argv:                         # redraw from data/gut_meanfield_b7.npz, no rollout
        z = np.load(os.path.join(EXP, "data", "gut_meanfield_b7.npz"))
        figure({k: z[k] for k in z.files if k not in ("responsive", "n_pulses")}, z["responsive"], int(z["n_pulses"]))
    else:
        main()
