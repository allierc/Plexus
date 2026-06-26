"""harmonic_montage.py -- demonstrate that the HARMONIC loop-loss tracks trajectory morphology where
interior R² fails. Produces 3 PNGs in this directory:

  harmonic_montage_1_generated.png  -- 10x10 generated zoo: 10 trajectory TOPOLOGIES (rows) x 10
       perturbations (cols). Each cell prints R² and LoopScore; the value is GREEN when the metric's
       verdict matches the ground-truth (shape-preserving vs shape-broken), RED when it is FOOLED.
       At a glance: LoopScore is green across the grid; R² goes red in the offset/time-shift columns
       (says BAD when shape is fine) and the radial-dominated columns (says GOOD when the loop is wrong).
  harmonic_montage_2_gt_descriptor.png -- the real 10x10 GT beat loops (green) with the harmonic
       FUNDAMENTAL ellipse (orange) overlaid; titled with chirality and signed area -> the descriptor
       reads real morphology spatially.
  harmonic_montage_3_gt_radial_fooled.png -- the trainer's actual failure on the REAL field: each red
       sim loop is the real loop COLLAPSED toward its radial line (loop/area removed). R² barely
       notices (stays high), LoopScore catches it (drops). Per-cell + aggregate.
"""
import os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cardio_harmonic as H

HERE = os.path.dirname(os.path.abspath(__file__))
G = 53                                                            # frames in the fit beat
TH = torch.linspace(0, 2 * np.pi, G + 1)[:-1]                    # one period


# --------------------------------------------------------------------------- topology generators
def _xy(x, y): return torch.stack([x, y], -1)

def _rot(c, a):
    ca, sa = np.cos(a), np.sin(a)
    return _xy(ca * c[:, 0] - sa * c[:, 1], sa * c[:, 0] + ca * c[:, 1])

def circle():     return _xy(torch.cos(TH), torch.sin(TH))
def ellipse():    return _rot(_xy(torch.cos(TH), 0.5 * torch.sin(TH)), 0.5)
def peanut():     r = 1 + 0.8 * torch.cos(2 * TH); return _xy(r * torch.cos(TH), r * torch.sin(TH))     # concave waist
def fig8():       return _xy(torch.cos(TH), torch.sin(TH) * torch.cos(TH))                              # self-intersecting
def crescent():   return _xy(torch.cos(TH) - 0.4 * torch.cos(2 * TH), torch.sin(TH) - 0.4 * torch.sin(2 * TH))  # concave banana
def rtriangle():  r = 1 + 0.35 * torch.cos(3 * TH); return _xy(r * torch.cos(TH), r * torch.sin(TH))    # convex-ish polygon
def teardrop():   return _xy(torch.sin(TH), torch.cos(TH) * (1 - torch.cos(TH)) * 0.6)                  # teardrop
def bean():       return _xy(torch.cos(TH) + 0.25 * torch.cos(2 * TH), torch.sin(TH))                   # asymmetric bean
def doubleloop(): return _xy(torch.cos(TH) + 0.3 * torch.cos(3 * TH), torch.sin(TH) + 0.3 * torch.sin(3 * TH))  # lobed
def noisy():      g = torch.Generator().manual_seed(0); return ellipse() + 0.08 * torch.randn(G, 2, generator=g)

TOPOS = [("circle", circle), ("ellipse", ellipse), ("peanut\n(concave)", peanut), ("figure-8", fig8),
         ("crescent", crescent), ("r-triangle", rtriangle), ("teardrop", teardrop), ("bean", bean),
         ("double-loop", doubleloop), ("noisy", noisy)]


# --------------------------------------------------------------------------- perturbations (real, sim)
# 'broken' = does the perturbation change the LOOP MORPHOLOGY (ground-truth: should score BAD)?
BIGLINE = _xy(2.0 * torch.cos(TH), 0.0 * TH)                      # shared radial breathing

def perturb(name, base):
    b = base
    if name == "match":          return b, b.clone(), False
    if name == "offset·sm":      return b, b + torch.tensor([0.4, 0.2]), False
    if name == "offset·lg":      return b, b + torch.tensor([1.2, 0.7]), False
    if name == "tshift·G/4":     return b, torch.roll(b, G // 4, 0), False
    if name == "chir·flip":      return b, _xy(b[:, 0], -b[:, 1]), True
    if name == "radial:loop→0":  return BIGLINE + 0.3 * b, BIGLINE.clone(), True          # loop removed under big line
    if name == "radial:chir":    return BIGLINE + 0.3 * b, BIGLINE + 0.3 * _xy(b[:, 0], -b[:, 1]), True  # loop flipped under big line
    if name == "scramble":       g = torch.Generator().manual_seed(1); return b, torch.cumsum(0.3 * torch.randn(G, 2, generator=g), 0), True
    raise ValueError(name)

PERTS = ["match", "offset·sm", "offset·lg", "tshift·G/4", "chir·flip", "radial:loop→0", "radial:chir", "scramble"]
MOV1 = torch.ones(1, dtype=torch.bool)


def _cell_metrics(real, sim):
    r = real[:, None, :]; s = sim[:, None, :]
    return H.interior_r2(s, r, MOV1), H.harmonic_score(s, r, MOV1)


# dark palette (black background montages)
BG, TXT, GREEN, RED, ORANGE = "black", "#dddddd", "#33dd33", "#ff5555", "#ffaa33"

def _dark(ax):
    ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#333333")


def montage1():
    nR, nC = len(TOPOS), len(PERTS)
    fig, axs = plt.subplots(nR, nC, figsize=(2.05 * nC, 2.15 * nR), facecolor=BG)
    for i, (tname, gen) in enumerate(TOPOS):
        base = gen()
        for j, pname in enumerate(PERTS):
            ax = axs[i, j]; ax.set_aspect("equal"); _dark(ax)
            real, sim, broken = perturb(pname, base)
            r2, hm = _cell_metrics(real, sim)
            ax.plot(*torch.cat([real, real[:1]]).T, color=GREEN, lw=1.6)
            ax.plot(*torch.cat([sim, sim[:1]]).T, color=RED, lw=1.1, alpha=0.9)
            good = not broken                                    # ground-truth: should the metric say GOOD?
            def col(v): return "#33dd33" if (v > 0.5) == good else "#ff5555"   # green if verdict correct, red if fooled
            ax.set_title(f"R²={r2:+.2f}  LS={hm:+.2f}", fontsize=13, color="white", fontweight="bold")
            ax.text(0.02, 0.98, "R²", color=col(r2), fontsize=9, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
            ax.text(0.22, 0.98, "LS", color=col(hm), fontsize=9, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
            if j == 0: ax.set_ylabel(tname, fontsize=8, color=TXT)
            if i == nR - 1: ax.annotate(pname, (0.5, -0.16), xycoords="axes fraction", ha="center", va="top", fontsize=7, rotation=20, color=TXT)
    fig.suptitle("LoopScore (loop morphology) vs interior R²  —  generated topology zoo (green=real, red=sim)\n"
                 "metric label GREEN = verdict correct, RED = FOOLED.  R² fails on offset/time-shift (shape fine) "
                 "and radial:* (loop wrong); LoopScore is correct throughout.", fontsize=12, y=0.995, color=TXT)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    p = os.path.join(HERE, "harmonic_montage_1_generated.png")
    fig.savefig(p, dpi=110, facecolor=BG, bbox_inches="tight"); plt.close(fig); print("wrote", p)


# --------------------------------------------------------------------------- GT 10x10 loops
def load_gt_loops():
    sys.path.insert(0, os.path.join(HERE, "..", "cardio"))
    from cardio_real_render import select_grid_nodes
    import cardio_mpm_data as D
    P = np.load(D.NPZ)["pos"].astype(np.float32)                 # [F, 137^2, 2]
    sel = select_grid_nodes(10, 10)                              # 100 flat indices, row-major
    Pm = (D.DOM_LO + D.DOM * P)[:, sel]                          # [F, 100, 2] in sheet domain
    onset, glen = 152, G                                         # the trainer's fit beat (onsets[-2]=152)
    beat = Pm[onset:onset + glen] - Pm[onset]                    # [G, 100, 2] displacement loops
    return torch.tensor(beat)                                    # [G,100,2]


def _grid10(beat, redfn, title, fname, descr=False):
    fig, axs = plt.subplots(10, 10, figsize=(20, 20.6), facecolor=BG)
    r2s, hms = [], []
    for k in range(100):
        ax = axs[k // 10, k % 10]; ax.set_aspect("equal"); _dark(ax)
        real = beat[:, k]                                        # [G,2]
        ax.plot(*torch.cat([real, real[:1]]).T, color=GREEN, lw=1.4)
        if descr:
            area = float((H.harmonic_descriptor(real[:, None, :], 4)["area"]).sum())
            ell = H.fundamental_ellipse(real)
            ax.plot(*torch.cat([ell, ell[:1]]).T, color=ORANGE, lw=1.0, alpha=0.9)
            ax.set_title(f"{'CCW' if area>0 else 'CW'}  S={area:+.1e}", fontsize=9, color="white", fontweight="bold")
        else:
            sim = redfn(real)
            ax.plot(*torch.cat([sim, sim[:1]]).T, color=RED, lw=1.0, alpha=0.9)
            r2, hm = _cell_metrics(real, sim); r2s.append(r2); hms.append(hm)
            ax.set_title(f"R²={r2:+.2f} LS={hm:+.2f}", fontsize=13, color="white", fontweight="bold")
    if not descr:
        title += f"\naggregate over 100 nodes:  R²={np.mean(r2s):+.3f}   LoopScore={np.mean(hms):+.3f}"
    fig.suptitle(title, fontsize=14, y=0.997, color=TXT)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    p = os.path.join(HERE, fname); fig.savefig(p, dpi=95, facecolor=BG, bbox_inches="tight"); plt.close(fig); print("wrote", p)


def mistimed(real):
    """A SHAPE-PRESERVING degradation: the SAME real loop, phase-lagged in time and slightly displaced
    -- exactly what imperfect pulse timing / anchoring produces. Morphology is identical; only timing
    and placement differ. The ideal metric should rate this HIGH (right loop), R² rates it LOW."""
    shift = G // 5
    off = 0.20 * (real.amax(0) - real.amin(0))                  # ~20% of the loop's own extent
    return torch.roll(real, shift, 0) + off


if __name__ == "__main__":
    montage1()
    beat = load_gt_loops()
    _grid10(beat, None, "Real 10x10 GT beat loops (green) + harmonic FUNDAMENTAL ellipse (orange)  —  "
            "the descriptor reads real morphology (chirality CCW/CW, signed area S)",
            "harmonic_montage_2_gt_descriptor.png", descr=True)
    _grid10(beat, mistimed, "Real 10x10 GT loops (green) vs a SHAPE-PRESERVING mistimed+displaced sim (red): "
            "same loop, wrong timing/placement.\nThe morphology is correct, so the ideal metric should say GOOD "
            "-- R² is FOOLED LOW (punishes timing/placement), LoopScore correctly stays HIGH.",
            "harmonic_montage_3_gt_timing_fooled.png", descr=False)
