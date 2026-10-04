"""exp20: THE TWO INPUT MASKS SIDE BY SIDE, from above (Cedric, 2026-10-04: "slide 127 is really not clear"): rows the
batch 1-5 mask and the batch-6 mask closer to biology, columns the inputs (the UV pulse with the beam position, the
grating, the swim), each panel glucose fish 1's brain from above, head left, every cell a faint dot and the cells
that input enters coloured, with the rule and the count above; in the UV panels the 2,538 gut-responsive cells (the
paper's selection, tools/gutbrain_baselines.py) in yellow on top.

    PYTHONPATH=src:tools python tools/exp20_mask_compare.py
Writes presentation/figs/mask_compare_f1.png and a copy in png/.
"""
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
ZF = os.path.join(os.environ["GNN_OUTPUT_ROOT"], "graphs_data", "zebrafish")


def main(rec="gutbrain_glucose_f1"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks import trace_recording as TR
    P = np.asarray(TR.load(rec)["pos_view"], np.float64)
    o = np.argsort(P[:, 2])
    old, bio = (np.load(os.path.join(ZF, f"input_mask_{rec}{t}.npz")) for t in ("", "_bio"))
    resp = np.load(os.path.join(EXP, "data", f"baselines_{rec}_cells.npz"))["responsive"]
    rows = [("batch 1-5 mask", old["mask_by_input"],
             ("UV pulse + beam: |t| of the change after EVERY training pulse, the off-fish control included",
              "grating: coherence with the grating speed", "swim: coherence with the swim power")),
            ("batch 6 mask, closer to biology", bio["mask_by_input"],
             ("UV pulse + beam: EXCITED after the gut pulses (t, top 10 %), NOT after the control (t < 2)",
              "grating: coherence with the grating speed (the same cells)", "swim: none -- a motor output, not an input"))]
    cols = [(0, "#ff4040"), (3, "#4fc3f7"), (4, "#81c784")]
    fig, ax = plt.subplots(2, 3, figsize=(16, 6.4), facecolor="black")
    for i, (lab, by, titles) in enumerate(rows):
        for j, ((c, col), t) in enumerate(zip(cols, titles)):
            a = ax[i, j]
            a.set_facecolor("black"); a.axis("off"); a.set_aspect("equal")
            a.scatter(P[o[::6], 0], P[o[::6], 1], s=0.15, c="0.28", lw=0)
            m = by[o, c] > 0
            a.scatter(P[o][m, 0], P[o][m, 1], s=0.35, c=col, lw=0)
            if c == 0:
                a.scatter(P[resp, 0], P[resp, 1], s=0.8, c="#ffeb3b", lw=0)
                inside = int((by[resp, 0] > 0).sum())
                t2 = f"{int(m.sum()):,} cells; yellow: the {int(resp.sum()):,} gut-responsive cells, {100 * inside / resp.sum():.0f} % inside"
            else:
                t2 = f"{int(m.sum()):,} cells"
            a.set_title(f"{t}\n{t2}", color="white", fontsize=8.5)
        fig.text(0.005, 0.73 - 0.47 * i, lab, color="white", fontsize=11, rotation=90, va="center", weight="bold")
    x0, y0 = P[:, 0].min(), P[:, 1].min() - 25
    ax[1, 0].plot([x0, x0 + 100], [y0, y0], color="white", lw=2)
    ax[1, 0].text(x0 + 50, y0 - 8, "100 µm", color="white", fontsize=8, ha="center", va="top")
    fig.text(0.5, 0.01, "glucose fish 1 from above, head left; each input enters only its own cells in batch 6 (a per-input "
             "mask); top 10 % = 19,035 of the 190,346 cells", color="0.75", fontsize=9, ha="center")
    fig.tight_layout(rect=(0.015, 0.03, 1, 1))
    out = os.path.join(EXP, "presentation", "figs", "mask_compare_f1.png")
    fig.savefig(out, dpi=150, facecolor="black")
    plt.close(fig)
    shutil.copy(out, os.path.join(EXP, "png", os.path.basename(out)))
    print("[mask_compare]", out)


if __name__ == "__main__":
    main()
