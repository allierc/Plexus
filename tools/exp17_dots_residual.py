"""exp17: WHAT THE LAW MISSES DURING THE DOTS (Cedric, 2026-10-04: the learned brain mean stays flat through the dots
block while the recorded one rises: missing visual input -- neurons tuned to the dots' texture, flicker, local motion,
which the two dots features f2 / f3 do not carry -- or a missing internal state?). Local, from a run's free rollout.

For each neuron i, from the 800 frames of the free rollout (results/<run>_movie.npz) and the recording at those frames:
  residual   r_i = < xhat_i - x_i >_dots - < xhat_i - x_i >_other blocks      (learned minus recorded; the neuron's
             offset outside the dots removed, so only what goes wrong DURING the dots remains; negative = too low)
  recorded   d_i = < x_i >_dots - < x_i >_other blocks                          (the dots response itself)
each drawn on the brain from above (head left) and from the side; r(residual, recorded) over the neurons says whether the
law misses the dots response where it occurs. Below: the brain-mean dF/F, recorded (green) and learned (white), the dots
block shaded.

    PYTHONPATH=src:tools python tools/exp17_dots_residual.py zap_e15_cur_siren
Writes presentation/figs/dots_residual_<run>.png, data/dots_residual_<run>.json (+ png/).
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
G = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "log", "training", "zapbench")


def main(run, block="dots"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import _brain_view
    spec = T.load(run)
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    z = np.load(os.path.join(G, run, "results", f"{run}_movie.npz"))
    fr = z["frames"]
    pred = z["pred"].astype(np.float64)
    X = rec["dff"][fr].astype(np.float64)
    names = [str(n) for n in rec["names"]]
    cond = rec["condition"][fr]
    b = names.index(block)
    inb, oth = cond == b, cond != b
    R = pred - X
    res = R[inb].mean(0) - R[oth].mean(0)
    dots = X[inb].mean(0) - X[oth].mean(0)
    r_ = float(np.corrcoef(res, dots)[0, 1])
    bv = _brain_view(np.asarray(rec["pos_um"], np.float64))
    top, side = bv[:, :2], np.stack([bv[:, 0], bv[:, 2]], 1)
    o_top, o_side = np.argsort(bv[:, 2]), np.argsort(bv[:, 1])
    doc = {"run": run, "block": block, "frames_in_block": int(inb.sum()), "frames_other": int(oth.sum()),
           "r_residual_vs_recorded_response": r_,
           "brain_mean_in_block": {"recorded": float(X[inb].mean()), "learned": float(pred[inb].mean())},
           "brain_mean_other": {"recorded": float(X[oth].mean()), "learned": float(pred[oth].mean())},
           "neurons_too_low_frac": float((res < 0).mean())}
    from matplotlib.colors import LinearSegmentedColormap
    cm_res = LinearSegmentedColormap.from_list("bkr", ["#4aa8ff", "#10304a", "black", "#4a1010", "#ff4a3a"])
    cm_dot = LinearSegmentedColormap.from_list("pko", ["#b48cff", "#2a1a44", "black", "#4a2a08", "#ffa030"])
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9.2), facecolor="black")
    vr = np.percentile(np.abs(res), 98)
    vd = np.percentile(np.abs(dots), 98)
    for j, (v, ttl, vm, cm) in enumerate(((res, "the law's error during the dots (learned - recorded), the neuron's offset "
                                           "elsewhere removed", vr, cm_res),
                                          (dots, "the recorded dots response (dots - other blocks)", vd, cm_dot))):
        for k, (P, o, nm, y0, h) in enumerate(((top, o_top, "from above", 0.47, 0.42), (side, o_side, "from the side", 0.22, 0.22))):
            ax = fig.add_axes([0.01 + 0.5 * j, y0, 0.46, h])
            ax.set_facecolor("black")
            o = np.argsort(np.abs(v))                                     # the strongest neurons drawn last
            sc = ax.scatter(P[o, 0], P[o, 1], c=v[o], s=0.5, cmap=cm, vmin=-vm, vmax=vm, linewidths=0)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.text(0.01 + 0.5 * j, y0 + h + 0.005, f"{nm}" + (f": {ttl}" if k == 0 else ""), color="white",
                     fontsize=10 if k == 0 else 9)
        cax = fig.add_axes([0.475 + 0.5 * j, 0.50, 0.006, 0.30])
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(colors="0.8", labelsize=7)
        cb.set_label("dF/F", color="0.8", fontsize=8)
    m = fig.add_axes([0.05, 0.05, 0.90, 0.11])
    m.set_facecolor("black")
    tm = np.asarray(z["r2_t"]) * 0.914 / 60
    m.plot(tm, z["mean_obs_all"], color="#2ca02c", lw=0.8)
    m.plot(tm, z["mean_pred_all"], color="white", lw=0.8)
    tb = fr[inb] * 0.914 / 60
    m.axvspan(tb.min(), tb.max(), color="0.35", alpha=0.5, lw=0)
    m.text((tb.min() + tb.max()) / 2, 1.02, block, color="0.85", fontsize=9, ha="center", transform=m.get_xaxis_transform())
    m.set_xlim(tm[0], tm[-1])
    m.set_yticks([])
    for sp in ("top", "right"):
        m.spines[sp].set_visible(False)
    m.set_xlabel("time, min", fontsize=8)
    fig.text(0.05, 0.175, "brain-mean dF/F: recorded (green), learned (white); the free rollout of the whole 2 h",
             color="0.8", fontsize=9)
    path = os.path.join(EXP, "presentation", "figs", f"{block}_residual_{run}.png")
    fig.savefig(path, dpi=120, facecolor="black", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    json.dump(doc, open(os.path.join(EXP, "data", f"{block}_residual_{run}.json"), "w"), indent=1)
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
