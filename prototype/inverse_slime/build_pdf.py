"""build_pdf.py -- the LIVING visual report for the slime inverse problem.

Re-run any time to refresh inverse_gt_vs_stages.pdf. Every page is the page-1/2
VISUAL style: GROUND TRUTH on top + up to 2 recovery methods below (3 rows/page),
5 timepoints across, SQUARE panels. Pages alternate CELL and FIELD evolution. Add a
recovery method to STAGES -> it shows up as a new test, in the same visual format.

    PYTHONPATH=../../src python build_pdf.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DR = "/groups/saalfeld/home/allierc/GraphData/graphs_data/slime"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "inverse_gt_vs_stages.pdf")

GT = ("slime_gt", "GROUND TRUTH")
# recovery methods (the "tests"), best first so it sits on page 1 right under GT
METHODS = [
    ("slime_gd8d_v2", "8D POSITION-loss (best)"),      # the new test -- de-biased sensor_angle
    ("slime_gd8d", "8D heading-loss"),
    ("slime_gd", "grad-desc 4D (field+move)"),
    ("slime_recovered", "UCB (black-box)"),
]
FRAMES = [0, 5, 10, 20, 50, 100, 200]                         # real-time columns (t)
TMAX = 200                                                    # max real time in a trajectory
NCOL = len(FRAMES)


def frame_indices(n):
    """Map the real-time FRAMES to indices of an array with `n` saved frames spanning 0..TMAX."""
    return [int(round(t * (n - 1) / TMAX)) for t in FRAMES]


def have(folder):
    return os.path.exists(os.path.join(DR, folder, "trajectory.npz"))


# per-type palette (matches the slime configs: green / red / blue / yellow)
PAL = np.array([[0.2, 0.95, 0.65], [1.0, 0.35, 0.45], [0.4, 0.65, 1.0], [1.0, 0.85, 0.3]])


def draw_row(axes_row, folder, label, channel):
    d = np.load(os.path.join(DR, folder, "trajectory.npz"))
    nt = np.asarray(d["cell__node_type"]) if "cell__node_type" in d else None
    multitype = nt is not None and int(nt.max()) > 0
    arr = np.asarray(d["cell__pos"]) if channel == "cell" else np.asarray(d["chemical__grid"])
    vmax = float(arr.max() ** 0.7) if channel != "cell" else 0
    fr = frame_indices(arr.shape[0])
    for c, t in enumerate(fr):
        ax = axes_row[c]
        if channel == "cell":
            ax.set_facecolor("black")
            cols = PAL[nt % 4] if multitype else "#33f0a6"
            ax.scatter(arr[t, :, 0], arr[t, :, 1], s=0.25, c=cols, linewidths=0)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        else:
            g = arr[t]                                         # [C,nx,ny]
            if g.shape[0] == 1:
                ax.imshow(g[0].T ** 0.7, origin="lower", cmap="magma", vmin=0, vmax=max(1e-6, vmax))
            else:                                              # multi-type: per-channel colour composite
                rgb = np.clip(np.tensordot((g ** 0.7), PAL[:g.shape[0]], axes=(0, 0)), 0, 1)  # [nx,ny,3]
                ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower")
        ax.set_aspect("equal", adjustable="box")              # SQUARE panels
        ax.set_xticks([]); ax.set_yticks([])
    axes_row[0].set_ylabel(label, fontsize=9)


def page(pdf, channel, methods):
    """One page: GT row + up to 2 method rows (3 rows total), `channel` evolution."""
    rows = [GT] + methods
    fig, axes = plt.subplots(len(rows), NCOL, figsize=(NCOL * 2.0, len(rows) * 2.0),
                             facecolor="white", squeeze=False)
    ch = "Cell" if channel == "cell" else "Chemical field"
    fig.suptitle(f"{ch} evolution -- GROUND TRUTH vs recovery", fontsize=13, y=0.99)
    for r, (folder, label) in enumerate(rows):
        if have(folder):
            draw_row(axes[r], folder, label, channel)
        else:
            for c in range(NCOL):
                axes[r][c].axis("off")
            axes[r][0].text(0.0, 0.5, f"{label}\n(not generated yet)", fontsize=8)
    # column-time titles on the top row
    for c in range(NCOL):
        axes[0][c].set_title(f"t={FRAMES[c]}", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig)


def target_page(pdf, channel, ti):
    """GT vs one-step-position recovery for target ti (3 rows: GT, recovery, [blank])."""
    gtf, recf = f"slime_gt_t{ti}", f"slime_pos_t{ti}"
    if not (have(gtf) and have(recf)):
        return False
    rows = [(gtf, f"GROUND TRUTH (target {ti})"), (recf, f"POSITION-loss recovery (target {ti})")]
    fig, axes = plt.subplots(len(rows), NCOL, figsize=(NCOL * 2.0, len(rows) * 2.0),
                             facecolor="white", squeeze=False)
    ch = "Cell" if channel == "cell" else "Chemical field"
    fig.suptitle(f"{ch} -- target {ti}: GT vs position-loss recovery", fontsize=13, y=0.99)
    for r, (folder, label) in enumerate(rows):
        draw_row(axes[r], folder, label, channel)
    for c in range(NCOL):
        axes[0][c].set_title(f"t={FRAMES[c]}", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig)
    return True


def description_page(pdf):
    """A one-page written description: the inverse problem vs Plexus, method, results."""
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.text(0.5, 0.965, "The slime inverse problem in Plexus", ha="center", fontsize=17, weight="bold")
    fig.text(0.5, 0.945, "recovering the generating spec from a simulation", ha="center",
             fontsize=11, style="italic", color="0.3")

    blocks = [
        ("Plexus  —  the forward model", [
            "Plexus compiles a small typed spec (sets - fields - operators - schedule) into a",
            "simulation:   spec  ->  engine  ->  trajectory.   The slime spec is 8 continuous numbers:",
            "field side  {deposit.amount, diffuse.rate, decay.rate, sense.cross}  and per-cell motion",
            "{move_speed, turn_speed, sensor_angle, sensor_dist}.  Registered operators (deposit /",
            "diffuse / decay / sense / bounce / advance) make cells deposit, then sense and climb, a",
            "self-made chemical field -> the filament network seen in the GROUND TRUTH rows.",
        ]),
        ("The inverse problem", [
            "Given ONLY a simulation (the cell + chemical-field evolution over time), recover the spec",
            "that produced it -- inverting Plexus's forward arrow:  trajectory  ->  spec.  This is the",
            "system-identification / recognition problem under the paper's mechanism-discovery program.",
        ]),
        ("Method", [
            "Mirror each engine operator with a DIFFERENTIABLE 'learnable operator', each validated",
            "EXACT against the engine (deposit/diffuse/decay/advance match to ~1e-7).  Fit parameters by",
            "gradient descent on a teacher-forced one-step loss:",
            "  - field params + move_speed:  one-step field / displacement loss  ->  recovered exactly.",
            "  - sense params:  the engine's `sense` is non-differentiable (hard branch + random turn),",
            "    so a differentiable surrogate is fit on particle motion.  KEY RESULT: putting the loss",
            "    on PARTICLE POSITIONS (not reconstructed heading) removes a systematic sensor_angle",
            "    bias -- diagnosed with likelihood profiles; one-step beats a recurrent rollout here.",
        ]),
        ("Results", [
            "  field params (amount/diffuse/decay) + move_speed :  u-err ~ 0.000  (EXACT)",
            "      vs the UCB black-box baseline:  mean u-err 0.283 -> 0.001   (~224x more accurate)",
            "  sense params (one-step position loss), across 4 random targets:",
            "      sensor_angle u-err   t0 0.015   t1 0.158   t2 0.092   t3 0.051",
            "      turn_speed ~ 0.05-0.07,   sensor_dist ~ 0.03-0.06",
            "  The recovered specs reproduce the GT morphology (following pages: GT vs recovery).",
            "  Caveat: the LATENT field is NOT identifiable from particle motion alone (flat profile)",
            "  -- it needs its own observation; relevant for the cardio (latent-electrics) problem.",
        ]),
    ]
    y = 0.90
    for title, lines in blocks:
        fig.text(0.07, y, title, fontsize=12.5, weight="bold", color="#1f4e79")
        y -= 0.028
        for ln in lines:
            fig.text(0.09, y, ln, fontsize=9.3, family="monospace")
            y -= 0.0205
        y -= 0.016
    fig.text(0.5, 0.03, "prototype/inverse_slime  -  scientific log: runs/logbook2.md", ha="center",
             fontsize=8, color="0.5")
    pdf.savefig(fig); plt.close(fig)


def main():
    methods = [m for m in METHODS]                            # showcase target (target0)
    chunks = [methods[i:i + 2] for i in range(0, len(methods), 2)] or [[]]
    npages = 0
    with PdfPages(OUT) as pdf:
        description_page(pdf); npages += 1                     # written cover/description page
        # showcase target: method comparison (GT + 2 methods per page)
        for ch in ("cell", "field"):
            for chunk in chunks:
                page(pdf, ch, chunk); npages += 1
        # additional targets: a GT-vs-recovery page each (more pages as they generate)
        for ti in (1, 2, 3):
            for ch in ("cell", "field"):
                if target_page(pdf, ch, ti):
                    npages += 1
        # multi-type (2 & 4 types): GT vs position-loss recovery, cross now recovered
        for tag, lab in [("slime2", "2-type"), ("slime4", "4-type")]:
            if have(f"{tag}_gt") and have(f"{tag}_pos"):
                for ch in ("cell", "field"):
                    rows = [(f"{tag}_gt", f"GROUND TRUTH ({lab})"),
                            (f"{tag}_pos", f"POSITION-loss recovery ({lab}, incl. cross)")]
                    fig, axes = plt.subplots(len(rows), NCOL, figsize=(NCOL * 2.0, len(rows) * 2.0),
                                             facecolor="white", squeeze=False)
                    cn = "Cell" if ch == "cell" else "Chemical field"
                    fig.suptitle(f"{cn} -- {lab} slime: GT vs recovery (cross identifiable)", fontsize=13, y=0.99)
                    for r, (folder, label) in enumerate(rows):
                        draw_row(axes[r], folder, label, ch)
                    for cc in range(NCOL):
                        axes[0][cc].set_title(f"t={FRAMES[cc]}", fontsize=8)
                    fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig); npages += 1
    print("wrote", OUT, "-- pages:", npages)


if __name__ == "__main__":
    main()
