"""make_loopscore_sensitivity.py -- which morphology dimensions does LoopScore (LS) actually reward?

Answers the STRUCTURAL question in the instruction. Takes the real 10x10 GT loops and perturbs ONE
morphology dimension at a time across a controlled sweep, measuring LS(perturbed vs original). A dimension
LS is SENSITIVE to drops LS fast; a dimension LS is invariant to keeps LS~1. Output:

  loopscore_sensitivity.png  -- 2 rows x 6 dims (black bg):
      row 1: a representative GT loop (green) vs a perturbed copy (red) at a moderate magnitude, titled LS
      row 2: LS (mean over the 100 loops) vs perturbation magnitude -- the sensitivity curve
  + a printed RANKING table (sensitivity = 1 - min LS over the sweep).

Run:  PYTHONPATH=../../src python make_loopscore_sensitivity.py
"""
import os
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cardio_harmonic as H
from harmonic_montage import load_gt_loops, BG, GREEN, RED          # reuse GT loader + dark palette

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- per-dimension perturbations
def p_size(L, f):                                                   # scale about centroid
    c = L.mean(0, keepdim=True); return c + f * (L - c)

def p_position(L, d):                                               # translate by d x (per-loop extent)
    ext = (L.amax(0) - L.amin(0)); return L + (d * ext)[None]

def p_orient(L, deg):                                               # rotate about centroid
    th = np.deg2rad(deg); ca, sa = np.cos(th), np.sin(th)
    c = L.mean(0, keepdim=True); X = L - c
    return c + torch.stack([ca * X[..., 0] - sa * X[..., 1], sa * X[..., 0] + ca * X[..., 1]], -1)

def p_phase(L, tau):                                               # time-shift the beat
    return torch.roll(L, int(tau), dims=0)

def p_chir(L, cc):                                                 # cc=1 orig, 0 zero-area fold, -1 flipped
    rev = torch.flip(L, dims=[0])
    return ((1 + cc) / 2) * L + ((1 - cc) / 2) * rev

def p_aspect(L, f):                                               # squash the MINOR axis (openness/aspect) per loop
    c = L.mean(0, keepdim=True); X = L - c; out = X.clone()
    for n in range(X.shape[1]):
        Xn = X[:, n, :]
        cov = (Xn.t() @ Xn) / Xn.shape[0]
        _, evec = torch.linalg.eigh(cov)                          # cols: [minor, major]
        proj = Xn @ evec; proj[:, 0] = proj[:, 0] * f             # scale minor coord
        out[:, n, :] = proj @ evec.t()
    return c + out


# (name, perturb fn, sweep magnitudes, the 'no-change' value, x-label, representative magnitude for row 1)
DIMS = [
    ("size",        p_size,     [0.4, 0.6, 0.8, 1.0, 1.25, 1.6],   1.0,  "scale f",       0.6),
    ("openness/\naspect", p_aspect, [0.2, 0.4, 0.6, 1.0, 1.5, 2.0], 1.0, "minor-axis f",  0.4),
    ("chirality",   p_chir,     [1.0, 0.5, 0.0, -0.5, -1.0],       1.0,  "area scale c",  0.0),
    ("axis\norientation", p_orient, [0, 15, 30, 45, 70, 90],        0.0,  "rotation (deg)", 45),
    ("temporal\nphase", p_phase, [0, 2, 4, 8, 13, 26],              0.0,  "time-shift (fr)", 13),
    ("position",    p_position, [0.0, 0.1, 0.25, 0.5, 0.8, 1.2],   0.0,  "translate x ext", 0.5),
]


def ls_of(sim, real, mov):
    return H.harmonic_stats(sim, real, mov)[0]                     # mean LoopScore


def main():
    beat = load_gt_loops()                                         # [G,100,2] real loops
    mov = torch.ones(beat.shape[1], dtype=torch.bool)
    # representative loop for row 1 = the one with the largest |signed area| (clearest loop)
    area = H.harmonic_descriptor(beat, 4)["area"].sum(0).abs()
    rep = int(area.argmax())

    rows = []
    fig, axs = plt.subplots(2, len(DIMS), figsize=(3.0 * len(DIMS), 6.4), facecolor=BG)
    for j, (name, fn, sweep, nochange, xlab, repmag) in enumerate(DIMS):
        ls = [ls_of(fn(beat, m), beat, mov) for m in sweep]
        sens = 1.0 - min(ls)                                       # max degradation over the sweep
        rows.append((name.replace("\n", " "), sens, min(ls)))

        # row 1: representative loop, original (green) vs perturbed at repmag (red)
        ax = axs[0, j]; ax.set_facecolor(BG); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#333")
        o = beat[:, rep]; pert = fn(beat, repmag)[:, rep]
        ax.plot(*torch.cat([o, o[:1]]).T, color=GREEN, lw=1.6)
        ax.plot(*torch.cat([pert, pert[:1]]).T, color=RED, lw=1.3, alpha=0.9)
        ax.set_title(name, color="white", fontsize=11, fontweight="bold")
        ax.text(0.04, 0.97, f"LS={ls_of(fn(beat, repmag), beat, mov):+.2f}", transform=ax.transAxes,
                color="white", fontsize=10, fontweight="bold", ha="left", va="top")

        # row 2: LS vs magnitude
        ax2 = axs[1, j]; ax2.set_facecolor(BG)
        ax2.plot(sweep, ls, "-o", color="#66ccff", lw=1.8, ms=4)
        ax2.axvline(nochange, color="#55ff55", ls=":", lw=1); ax2.axhline(1.0, color="#444", ls=":", lw=1)
        ax2.set_ylim(-1.05, 1.05); ax2.set_xlabel(xlab, color="#ccc", fontsize=8)
        ax2.set_title(f"sensitivity {sens:.2f}", color="white", fontsize=10)
        ax2.tick_params(colors="#aaa", labelsize=7)
        for sp in ax2.spines.values(): sp.set_color("#444")
        if j == 0: ax2.set_ylabel("LoopScore", color="#ccc", fontsize=9)

    rank = sorted(rows, key=lambda r: -r[1])
    rank_str = "  |  ".join(f"{n}:{s:.2f}" for n, s, _ in rank)
    fig.suptitle("LoopScore sensitivity — which morphology dimensions does LS reward?  (green=GT, red=perturbed)\n"
                 "sensitivity = 1 − min LS over the sweep (higher = LS more sensitive).   RANK:  " + rank_str,
                 color="white", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(HERE, "loopscore_sensitivity.png")
    fig.savefig(p, dpi=120, facecolor=BG, bbox_inches="tight"); plt.close(fig)

    print("\n=== LoopScore sensitivity ranking (1 - min LS over sweep) ===")
    for n, s, mn in rank:
        print(f"  {n:18s} sensitivity={s:.3f}   (min LS over sweep = {mn:+.3f})")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
