"""export_figs.py -- export every visual page of inverse_gt_vs_stages.pdf as a standalone
figure (figures/gal_*.pdf) and write figures/_autofigs.tex, so the paper reproduces them
all as-is. Reuses build_pdf's type-aware draw_row + layout.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pdf as B

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
NCOL = B.NCOL


def page_fig(rows, channel, title):
    """A page = GROUND TRUTH on top + recovery rows; returns the matplotlib fig."""
    rows = [r for r in rows if B.have(r[0])]
    if len(rows) < 2:
        return None
    fig, axes = plt.subplots(len(rows), NCOL, figsize=(NCOL * 2.0, len(rows) * 2.0),
                             facecolor="white", squeeze=False)
    fig.suptitle(title, fontsize=12, y=0.99)
    for r, (folder, label) in enumerate(rows):
        B.draw_row(axes[r], folder, label, channel)
    fr = np.linspace(0, 200, NCOL).astype(int)
    for c in range(NCOL):
        axes[0][c].set_title(f"t={fr[c]}", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    pages = []                                              # (rows, channel, title, shortcaption)
    # showcase target (single-type): method comparison
    for chunk, tag in [(B.METHODS[:2], "best methods"), (B.METHODS[2:4], "baselines")]:
        for ch in ("cell", "field"):
            cn = "cell" if ch == "cell" else "chemical field"
            pages.append(([B.GT] + chunk, ch,
                          f"Showcase target -- {cn} evolution ({tag})",
                          f"Showcase single-type target, {cn} evolution: ground truth vs {tag}."))
    # per-target GT vs position-loss recovery
    for ti in (1, 2, 3):
        for ch in ("cell", "field"):
            cn = "cell" if ch == "cell" else "chemical field"
            pages.append(([(f"slime_gt_t{ti}", f"GT (t{ti})"), (f"slime_pos_t{ti}", f"recovered (t{ti})")],
                          ch, f"Target {ti} -- {cn}: GT vs position-loss recovery",
                          f"Target {ti}, {cn} evolution: ground truth vs position-loss recovery."))
    # multi-type
    for tag, lab in [("slime2", "2-type"), ("slime4", "4-type")]:
        for ch in ("cell", "field"):
            cn = "cell" if ch == "cell" else "chemical field"
            pages.append(([(f"{tag}_gt", f"GT ({lab})"), (f"{tag}_pos", f"recovered ({lab})")],
                          ch, f"{lab} slime -- {cn}: GT vs recovery (cross identifiable)",
                          f"{lab} slime, {cn} evolution: ground truth vs recovery (cross now recovered)."))

    tex = []
    n = 0
    for rows, ch, title, cap in pages:
        fig = page_fig(rows, ch, title)
        if fig is None:
            continue
        name = f"gal_{n:02d}"
        fig.savefig(os.path.join(FIG, name + ".pdf")); plt.close(fig)
        tex.append("\\begin{figure}[p]\\centering\n"
                   f"\\includegraphics[width=\\textwidth]{{figures/{name}.pdf}}\n"
                   f"\\caption{{{cap}}}\n\\end{{figure}}")
        n += 1
    open(os.path.join(FIG, "_autofigs.tex"), "w").write("\n".join(tex) + "\n")
    print(f"exported {n} gallery figures -> figures/gal_*.pdf + figures/_autofigs.tex")


if __name__ == "__main__":
    main()
