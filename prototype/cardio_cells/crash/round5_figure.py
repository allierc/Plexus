"""round5_figure.py -- ROUND 5, stage 5.  The figure a person should look at in the morning.

Six panels, all measured, no schematics:
  a  recovered vs planted Young's modulus, unconstrained against box-constrained (3 seeds)
  b  per-cell signed error map, unconstrained -- where the tail lives
  c  per-cell signed error map, box-constrained -- the same fit with the box
  d  three margin-20 tracer loops over the scored beat: reference vs the two predictions
  e  per-frame rms position error, in grid cells, over the 150-frame free rollout
  f  the acceptance statistic (held-out one-frame residual) against the gauged loopscore, with the
     zero-information band shaded

usage: PYTHONPATH=/workspace/Plexus/src python round5_figure.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GT = "#33dd66"          # ground truth
PRED = "#ffffff"        # the prediction under test
ALT = "#ff5555"         # the control it has to beat
NUL = "#8899aa"


def lab(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, color="w", fontsize=11, fontweight="bold",
            va="top", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="s90210/T8/eiv_box")
    ap.add_argument("--alt", default="s90210/T8/naive")
    ap.add_argument("--out", default="round5.png")
    a = ap.parse_args()

    rep = json.load(open(os.path.join(HERE, "round5_report.json")))
    rows = rep["rows"]
    sh = [json.load(open(os.path.join(HERE, f"round5_score_s{s}.json"))) for s in (0, 1)
          if os.path.exists(os.path.join(HERE, f"round5_score_s{s}.json"))]
    cand = {}
    for s in sh:
        for k, v in s["candidates"].items():
            cand.setdefault(k, v)
    cx = np.array(sh[0]["cell_centroid_x"], float)
    cy = np.array(sh[0]["cell_centroid_y"], float)
    Z = np.load(os.path.join(HERE, "theta_round5.npz"))
    th = Z["theta_true"]
    C = th.size // 2
    E = th[:C]
    TR = {}
    for s in (0, 1):
        p = os.path.join(HERE, f"tracks_round5_score_s{s}.npz")
        if os.path.exists(p):
            z = np.load(p)
            for k in z.files:
                TR.setdefault(k, z[k])
    real20 = TR["real20"]

    p40 = os.path.join(HERE, "theta_round5_box40k.npz")
    Z40 = np.load(p40) if os.path.exists(p40) else None

    def theta_of(name):
        seed, T, k = name.split("/")
        pref = {"clean": "round5_norm_clean", "hiF": "round5_norm_s90210_sF0.0327"}.get(
            seed, f"round5_norm_{seed}_sF0.0039")
        if k.endswith("40k"):
            return Z40[f"{pref}|{T}|{k[:-3]}"]
        return Z[f"{pref}|{T}|{k}"]

    plt.rcParams.update({"font.size": 9, "text.color": "w", "axes.labelcolor": "w",
                         "xtick.color": "w", "ytick.color": "w", "axes.edgecolor": "#777777"})
    fig, AX = plt.subplots(2, 3, figsize=(15.5, 8.6), facecolor="k")
    for ax in AX.ravel():
        ax.set_facecolor("k")

    # ---- a: recovered vs planted -------------------------------------------------------------
    ax = AX[0, 0]
    for i, sd in enumerate((90210, 555, 777)):
        for k, col, mk in (("naive", ALT, "o"), ("eiv_box", "#4da6ff", "^")):
            nm = f"s{sd}/T8/{k}"
            if nm not in rows:
                continue
            t = theta_of(nm)
            ax.scatter(E, t[:C], s=9, c=col, marker=mk, alpha=0.55, linewidths=0,
                       label=(k if i == 0 else None))
    ax.plot([0, 230], [0, 230], color=GT, lw=1.2, ls="--", label="identity")
    ax.set_xlim(30, 230)
    ax.set_ylim(-30, 260)
    ax.set_xlabel("planted $E_c$")
    ax.set_ylabel(r"recovered $\hat E_c$")
    lg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.90), frameon=False, fontsize=8)
    for t_ in lg.get_texts():
        t_.set_color("w")
    nneg = sum(int((theta_of(f"s{sd}/T8/naive")[:C] < 0).sum()) for sd in (90210, 555, 777))
    nout = sum(int((theta_of(f"s{sd}/T8/eiv_box")[:C] > 260).sum()) for sd in (90210, 555, 777))
    ax.text(0.98, 0.04, f"unconstrained: {nneg}/300 cells negative, slope 0.33\n"
                        f"box-constrained: 0 negative, {nout}/300 above the axis",
            transform=ax.transAxes, color="w", fontsize=8, ha="right")
    lab(ax, "a   3 measurement draws, T=8 frames, $\\sigma_F$=3.9e-3")

    # ---- b, c: per-cell error maps ------------------------------------------------------------
    for j, nm in enumerate((a.alt, a.best)):
        ax = AX[0, 1 + j]
        t = theta_of(nm)
        r = (t[:C] - E) / E
        sc = ax.scatter(cx, cy, c=np.clip(r, -1, 1), s=70, cmap="coolwarm", vmin=-1, vmax=1,
                        edgecolors="none")
        bad = np.abs(r) > 1
        ax.scatter(cx[bad], cy[bad], s=170, facecolors="none", edgecolors="w", linewidths=1.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(r"$(\hat E_c - E_c)/E_c$", color="w")
        cb.ax.yaxis.set_tick_params(color="w")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="w")
        rr = rows[nm]
        ax.text(0.02, 0.88,
                f"med|dE/E| {rr['med_E']:.3f}, |err|>1 in {int(bad.sum())}/{C} cells (circled)\n"
                f"negative $E$: {rr['n_negE']},  gauged loopscore {rr['gauged_loop']:.4f}\n"
                f"held-out 1-frame residual {rr['holdout_cleanF']:.4f}",
                transform=ax.transAxes, color="w", fontsize=8, va="top")
        lab(ax, f"{'bc'[j]}   {nm}")

    # ---- d: loops ------------------------------------------------------------------------------
    ax = AX[1, 0]
    amp = np.linalg.norm(real20 - real20[0], axis=-1).max(0)
    pick = np.argsort(-amp)[[0, 3, 7]]
    kb = TR.get(f"{a.best}|gauged")
    ka = TR.get(f"{a.alt}|gauged")
    kn = TR.get(f"{rep['floor']['loop_band_top_member']}|gauged")
    dxs = 1.0 / 128.0
    for i, p in enumerate(pick):
        def cen(A):
            return (A[:, p, :] - A[:, p, :].mean(0)) / dxs
        r = cen(real20)
        off = np.array([i * 3.6, 0.0])
        ax.plot(r[:, 0] + off[0], r[:, 1], color=GT, lw=2.2,
                label="reference" if i == 0 else None, zorder=1)
        for A, col, lb, z in ((kb, PRED, a.best, 3), (ka, ALT, a.alt, 2),
                              (kn, NUL, rep["floor"]["loop_band_top_member"], 2)):
            if A is None:
                continue
            c = cen(A)
            ax.plot(c[:, 0] + off[0], c[:, 1], color=col, lw=1.0,
                    label=lb if i == 0 else None, zorder=z)
    ax.set_aspect("equal")
    ax.set_xlabel("x, per-node centred, in grid cells (3 tracers, offset)")
    ax.set_ylabel("y, per-node centred (dx)")
    lg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=7.5,
                   ncol=2)
    for t_ in lg.get_texts():
        t_.set_color("w")
    ax.set_ylim(-1.6, 1.9)
    lab(ax, "d   3 margin-20 tracer loops, 150 frames (gauged)")

    # ---- e: rms per frame -----------------------------------------------------------------------
    ax = AX[1, 1]
    show = [(a.best, "#4da6ff"), (a.alt, ALT),
            (rep["floor"]["loop_band_top_member"], NUL), ("clean/T8/naive", GT)]
    for nm, col in show:
        if nm not in cand:
            continue
        v = cand[nm]["gauged"]["coarse"]["rms_pos_err_dx_per_frame"]
        ax.plot(np.arange(1, len(v) + 1), v, color=col, lw=1.3, label=nm)
    ax.set_yscale("log")
    ax.set_xlabel("frame of the scored beat")
    ax.set_ylabel("rms position error / dx")
    lg = ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    for t_ in lg.get_texts():
        t_.set_color("w")
    lab(ax, "e   free rollout error (interior particles)")

    # ---- f: acceptance statistic vs score -------------------------------------------------------
    ax = AX[1, 2]
    band = rep["floor"]["loop_band"]
    ax.axhspan(band[0], band[1], color="#334455", alpha=0.8, zorder=0)
    ax.text(0.98, band[1], " zero-information band", color=NUL, fontsize=8, ha="right",
            va="bottom", transform=ax.get_yaxis_transform())
    for nm, r in rows.items():
        if not isinstance(r["gauged_loop"], float):
            continue
        if nm.startswith("bank") or nm.startswith("null"):
            col, mk = NUL, "s"
        elif "box" in nm:
            col, mk = "#4da6ff", "^"
        elif nm == "theta_true" or nm.startswith("clean"):
            col, mk = GT, "*"
        else:
            col, mk = ALT, "o"
        ax.scatter(max(r["holdout_cleanF"], 1e-3), r["gauged_loop"], c=col, marker=mk, s=44,
                   zorder=3, linewidths=0)
    for nm, dxy in (("theta_true", (6, -9)), (a.best, (6, 2)), (a.alt, (6, -2)),
                    (rep["floor"]["loop_band_top_member"], (-6, 6)),
                    ("null_med0_rand45", (-6, 6)), ("hiF/T8/eiv_box", (-6, -12)),
                    ("s90210/T8/eiv_snr0", (-6, 4))):
        if nm in rows and isinstance(rows[nm]["gauged_loop"], float):
            ax.annotate(nm, (max(rows[nm]["holdout_cleanF"], 1e-3), rows[nm]["gauged_loop"]),
                        textcoords="offset points", xytext=dxy, color="w", fontsize=7,
                        ha="right" if dxy[0] < 0 else "left")
    ax.set_ylim(-0.45, 1.12)
    dn = rep["nulls"]["do_nothing"]["loopscore"]
    rp = rep["nulls"]["replay_previous_beat"]["loopscore"]
    for y, t_, c in ((dn, "do nothing", NUL), (rp, "replay previous beat", NUL)):
        ax.axhline(y, color=c, lw=0.8, ls=":")
        ax.text(0.02, y, f" {t_} {y:.3f}", transform=ax.get_yaxis_transform(), color=c,
                fontsize=7.5, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("held-out 1-frame residual (clean F)  — available on the recording")
    ax.set_ylabel("gauged loopscore")
    lab(ax, "f   acceptance statistic vs score")

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, a.out), dpi=150, facecolor="k")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
