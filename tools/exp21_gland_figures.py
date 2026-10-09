"""The figures of the gland's surface-cell motion (exp 21), from `tools/exp21_gland_surface_motion.py`'s JSON and cache
and Wang 2021's tracks, each with a sidecar `<stem>.txt` (headline first, then the numbers and how they were made),
and the watcher record rebuilt after each one (`tools/exp21_data_record.py`).

    python tools/exp21_gland_figures.py [--only flow,tracks3d,...]

Colours: red = surface layer (or movie 25x-4), blue = interior (or movie 25x-2); the three faces of the surface --
bottom (glass side), top (filter side), rim (lateral bud surface) -- purple, ochre, teal. No green.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                    # noqa: E402
import numpy as np                                                                 # noqa: E402
import pandas as pd                                                                # noqa: E402
from matplotlib.collections import LineCollection                                  # noqa: E402
from scipy.spatial import cKDTree                                                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import exp21_gland_surface_motion as M                                             # noqa: E402

FIGS = os.path.join(M.EXP, "figs")
PY = "/workspace/.conda_envs/neural-graph-linux/bin/python"
FFMPEG = "/workspace/.conda_envs/neural-graph-linux/bin/ffmpeg"
RED, BLUE = "#c0522a", "#1f6fb8"
FACE_C = {"bottom": "#7a4fa0", "top": "#9a7d1a", "rim": "#1a8a8a"}
INK, MUTED = "0.15", "0.45"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK, "legend.frameon": False,
                     "legend.fontsize": 8, "savefig.facecolor": "white", "figure.facecolor": "white"})


def hours(frame):
    """Movie clock (Tracks-all Time.csv: frame 25 = 2.00 h)."""
    return (np.asarray(frame) - 1) * M.FRAME_MIN / 60.0


def panel(ax, letter, label=None, xlabel=None, box=False):
    """Bold letter at the top-left corner, the y quantity as a label ABOVE the axis, no title, no box."""
    if not box:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-16, 7), textcoords="offset points",
                fontweight="bold", fontsize=13, ha="left", va="bottom")
    if label:
        ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7), textcoords="offset points",
                    fontsize=9, ha="left", va="bottom", color=INK)
    if xlabel:
        ax.set_xlabel(xlabel)


def save(fig, stem, text):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, stem + ".png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(FIGS, stem + ".txt"), "w") as fh:
        fh.write(text.strip() + "\n")
    subprocess.run([PY, os.path.join(ROOT, "tools/exp21_data_record.py")], cwd=ROOT, check=False,
                   env={**os.environ, "PYTHONPATH": "src:tools"}, capture_output=True)
    print("wrote", stem)


# ============================================================================ data
class Data:
    def __init__(self):
        self.J = json.load(open(os.path.join(M.DATA, "wang_surface_motion.json")))
        self.cache = np.load(os.path.join(M.DATA, "wang_surface_motion_cache.npz"), allow_pickle=False)
        self.df = M.add_normals(M.load(*M.MOVIES["25x-4"]))
        self.D, self.C = M.frame_drift(self.df)
        self.m4, self.m2 = self.J["25x-4"], self.J["25x-2"]


# ============================================================================ 1: the surface layer and its sampling
def fig_layer(S):
    df, m = S.df, S.m4
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.45, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    e = np.array(m["d_hist"]["edges"]); c = np.array(m["d_hist"]["counts"])
    ax.bar(e[:-1] + 1, c / 1000, width=1.8, color="0.6")
    ax.axvspan(-15, 13, color=RED, alpha=0.12, lw=0); ax.axvspan(-8, 13, color=RED, alpha=0.15, lw=0)
    ax.axvspan(-52, -20, color=BLUE, alpha=0.10, lw=0)
    ax.text(-14.5, c.max() / 1000 * 0.97, "S15", color=RED, fontsize=8); ax.text(-7.5, c.max() / 1000 * 0.88, "S8",
                                                                              color=RED, fontsize=8)
    ax.text(-50, c.max() / 1000 * 0.97, "interior (< -20 um)", color=BLUE, fontsize=8)
    panel(ax, "a", "nuclei, thousands (all frames)", "signed distance to the epithelial surface d (um)")
    t = 157
    f = df[df["t"] == t]
    cg, kap = M.outline(df[(df["t"] >= t - 12) & (df["t"] < t + 12)][["x", "y"]].values)
    cl, tp = M.clefts_tips(cg, kap)
    for k, (a, b, lab) in enumerate((("x", "y", "y (um)"), ("x", "z", "z (um)"))):
        ax = fig.add_subplot(gs[0, 1 + k])
        g = f[f["face"] == ""]
        ax.scatter(g[a], g[b], s=2, c="0.8", lw=0)
        for face, col in FACE_C.items():
            g = f[f["face"] == face]
            ax.scatter(g[a], g[b], s=5, c=col, lw=0, label=f"{face} ({len(g)})")
        if b == "y":
            ax.plot(np.r_[cg[:, 0], cg[:1, 0]], np.r_[cg[:, 1], cg[:1, 1]], color=INK, lw=0.8)
            ax.scatter(cg[cl, 0], cg[cl, 1], marker="v", s=40, c=INK, zorder=5, label="cleft")
            ax.scatter(cg[tp, 0], cg[tp, 1], marker="o", s=30, facecolors="none", edgecolors=INK, zorder=5,
                       label="tip")
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
        ax.set_aspect("equal")
        panel(ax, "bc"[k], f"{lab}, frame {t} (movie {hours(t):.1f} h)", "x (um)")
    # g(r) of the bottom face in the plane
    ax = fig.add_subplot(gs[1, 0])
    edges = np.arange(0, 41, 2.0); G = np.zeros(len(edges) - 1); nf = 0; ratio = []
    for tt in range(40, 280, 20):
        f2 = df[df["t"] == tt]
        cg2, _ = M.outline(f2[["x", "y"]].values)
        area = 0.5 * abs(np.sum(cg2[:, 0] * np.roll(cg2[:, 1], -1) - np.roll(cg2[:, 0], -1) * cg2[:, 1]))
        P = f2[f2["face"] == "bottom"][["x", "y"]].values
        pr = cKDTree(P).query_pairs(40, output_type="ndarray")
        r = np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1)
        h, _ = np.histogram(r, edges)
        G += 2 * h / (len(P) * (len(P) / area) * np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)); nf += 1
        rho = len(P) / area
        ratio.append((np.median(cKDTree(P).query(P, k=2)[0][:, 1]), 0.5 / np.sqrt(rho),
                      np.sqrt(2 / (np.sqrt(3) * rho)), len(P), area))
    ratio = np.array(ratio)
    ax.plot(0.5 * (edges[1:] + edges[:-1]), G / nf, color=FACE_C["bottom"], marker="o", ms=3)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.text(11, 0.3, f"tracked bottom-face nuclei: {ratio[:, 3].mean():.0f} on {ratio[:, 4].mean() / 1e3:.0f}k um2\n"
                      f"nearest neighbour {ratio[:, 0].mean():.1f} um\nrandom (Poisson) {ratio[:, 1].mean():.1f} um\n"
                      f"full hexagonal layer {ratio[:, 2].mean():.1f} um", fontsize=7.5)
    panel(ax, "d", "pair correlation g(r), bottom face in (x, y)", "r (um)")
    # the layer's spacing and the numbers that depend on it
    ax = fig.add_subplot(gs[1, 1])
    k1 = m["d_retention"]["S15_kept_1h_vs_fixed_d0_um"]; k2 = S.m2["d_retention"]["S15_kept_1h_vs_fixed_d0_um"]
    x = np.array([float(a) for a in k1])
    ax.plot(x, list(k1.values()), color=RED, marker="o", ms=4, label="movie 25x-4")
    ax.plot(x, list(k2.values()), color=BLUE, marker="s", ms=4, label="movie 25x-2")
    ax.axvline(13.9, color=MUTED, ls=":", lw=0.8); ax.text(14.2, 0.3, "S8 spacing 13.9 um\n= model cell 14 um",
                                                           fontsize=7)
    ax.axvline(7.3, color=MUTED, ls="--", lw=0.8); ax.text(7.6, 0.05, "nuclear spacing,\nSMG2 segmentation\n7.3 um",
                                                           fontsize=7, va="bottom")
    ax.set_ylim(0, 1.02); ax.legend(loc="lower right")
    panel(ax, "e", "neighbour pairs kept at 1 h (S15 layer)", "d0 handed to the estimator (um)")
    ax = fig.add_subplot(gs[1, 2])
    nz = df["nz"].values[df["d"].values > M.S15]
    ax.hist(nz[np.isfinite(nz)], bins=40, color="0.6")
    for x0 in (-M.NZ_FACE, M.NZ_FACE):
        ax.axvline(x0, color=MUTED, ls=":", lw=0.8)
    ax.text(-0.98, ax.get_ylim()[1] * 0.9, "bottom", color=FACE_C["bottom"]); ax.text(-0.1, ax.get_ylim()[1] * 0.9,
                                                                                     "rim", color=FACE_C["rim"])
    ax.text(0.75, ax.get_ylim()[1] * 0.9, "top", color=FACE_C["top"])
    panel(ax, "f", "S15 nuclei (all frames)", "n_z of the outward normal (grad d)")
    sh = m["normals"]["face_share_of_S15_spots"]
    save(fig, "gland_surface_layer", f"""
The gland's surface layer: a peak of nuclei 8-10 um inside the surface, three faces, and tracked nuclei that are a sparse sample of it.
a: signed distance d of every tracked epithelial nucleus (movie 25x-4, frames 25-289, 476,648 spots); the outer layer peaks at d -10..-8 um, trough -18..-14 um; S15 = d > -15 um (Wang's own 15 um), S8 = d > -8 um (the exp 21 targets so far: the outer quarter of the layer), interior = d < -20 um.
b-c: frame {t}: the gland is a lobed pancake ~430 x 350 x 100 um between filter (top) and glass (bottom); S15 nuclei by face from the outward normal grad d (local least squares over 20 nuclei within 30 um; |grad d| median {m['normals']['grad_d_norm_surface_median']:.2f}, 7 deg from PCA normals): bottom {sh['bottom']:.0%}, top {sh['top']:.0%}, rim {sh['rim']:.0%} of S15 spots; clefts/tips from the curvature of the x-y outline.
d: bottom-face nuclei have a 6-um hard core and no lattice peak (g max 1.26): NN {ratio[:, 0].mean():.1f} um vs random {ratio[:, 1].mean():.1f} um vs full hexagonal layer {ratio[:, 2].mean():.1f} um at their density -- the tracks sample roughly 1/3 of the layer's nuclei.
e: hence neighbour retention depends on the absolute d0 handed to the estimator, not on which nuclei are used: kept_1h 0.58 / 0.73 / 0.88 at d0 8 / 10 / 14 um (25x-4), 0.44 / 0.62 / 0.81 (25x-2); S8 and S15 agree at fixed d0 (0.884 vs 0.877 at 13.9 um).
""")


# ============================================================================ 2: speed, tangential vs normal
def fig_speed(S):
    df, C, m = S.df, S.C, S.m4
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), gridspec_kw={"wspace": 0.45})
    i0, i1 = M.lag_pairs(df, 1)
    V = M.displacement(df, C, i0, i1); sp = np.linalg.norm(V, axis=1) / M.DT_H
    d0 = df["d"].values[i0]
    ax = axs[0]
    bins = np.linspace(0, 60, 41)
    for lab, msk, col in (("surface layer S15", d0 > M.S15, RED), ("interior d < -20 um", d0 < M.DEEP, BLUE)):
        ax.hist(sp[msk], bins=bins, density=True, histtype="step", color=col, lw=1.6,
                label=f"{lab}:\nmedian {np.median(sp[msk]):.1f}")
    ax.legend(loc="center right", fontsize=7)
    panel(ax, "a", "density, 5-min steps, drift removed", "speed (um/h)")
    ax = axs[1]
    a = m["a_speed_tn"]
    lags = [5, 10, 15, 30, 60, 120]
    for g, col, lw, ls in (("S15", RED, 2.0, "-"), ("S15_bottom", FACE_C["bottom"], 1.0, "-"),
                           ("S15_top", FACE_C["top"], 1.0, "-"), ("S15_rim", FACE_C["rim"], 1.0, "-"),
                           ("interior", BLUE, 2.0, "-")):
        y = [a[g][f"{L}min"]["tangential_share"] for L in lags]
        ax.plot(lags, y, color=col, lw=lw, ls=ls, marker="o", ms=3, label=g.replace("S15_", "S15 "))
    ax.axhline(2 / 3, color=MUTED, ls=":", lw=0.9); ax.text(5, 0.672, "isotropic 2/3", fontsize=7, color=MUTED)
    ax.set_xscale("log"); ax.set_xticks(lags); ax.set_xticklabels([str(x) for x in lags]); ax.legend(loc="lower right")
    panel(ax, "b", "tangential share of squared displ.", "lag (min)")
    ax = axs[2]
    for g, col in (("S15", RED), ("interior", BLUE)):
        c = m["c_msd"][g]
        lag = np.array(c["lag_h"])
        ax.plot(lag, c["msd_t_um2"], color=col, lw=1.8, label=f"{g}: along the surface (tangential)")
        ax.plot(lag, c["msd_dd_um2"], color=col, lw=1.8, ls="--", label=f"{g}: change of depth d")
    ax.legend(loc="upper left", fontsize=7)
    panel(ax, "c", "mean squared displacement (um2)", "lag (h)")
    ax = axs[3]
    g = df[df["d"] > M.S15].groupby("tid")["t"].agg(["min", "size"])
    rng = np.random.default_rng(3)
    long_ = g[g["size"] >= 72].index.values
    pick = rng.choice(long_, size=min(25, len(long_)), replace=False)
    for k, tid in enumerate(pick):
        tr = df[df["tid"] == tid]
        ax.plot(hours(tr["t"]), tr["d"], color=RED, lw=0.6, alpha=0.6)
    ax.axhspan(-15, 5, color=RED, alpha=0.08, lw=0); ax.axhline(M.DEEP, color=BLUE, ls=":", lw=0.8)
    ax.set_ylim(-45, 5)
    panel(ax, "d", "d (um), 25 surface tracks of >= 6 h", "movie time (h)")
    s15, it = a["S15"], a["interior"]
    cm = m["c_msd"]
    save(fig, "gland_speed_tangential", f"""
Surface cells are as fast as interior cells and stay at their depth: over 30 min - 2 h, 75-79 % of their squared displacement is along the surface and their depth change is confined.
a: 5-min step speed with the gland's per-frame median step removed (wang_smg_stats.py's estimator): surface layer S15 median {s15['5min']['speed_um_h_median']:.1f} um/h, interior {it['5min']['speed_um_h_median']:.1f} um/h (S8 {a['S8']['5min']['speed_um_h_median']:.1f}, the 15.6 of the exp 21 target); localisation noise ~0.53 um per axis (PRW fit per axis) makes ~2/3 of the 5-min squared step.
b: displacement split on the outward normal at the step's start (grad d): tangential share {s15['5min']['tangential_share']:.2f} at 5 min (noise, isotropic 2/3), {s15['30min']['tangential_share']:.2f} at 30 min, {s15['60min']['tangential_share']:.2f} at 1 h, {s15['120min']['tangential_share']:.2f} at 2 h; the interior, against its nearest-surface direction, {it['30min']['tangential_share']:.2f} / {it['60min']['tangential_share']:.2f} / {it['120min']['tangential_share']:.2f} (the pancake flattens everyone's z motion).
c: what separates them is depth: the change of d of S15 cells is subdiffusive (log-log slope 30-120 min {cm['S15']['alpha_dd_30_120min']:.2f}; {cm['S15']['msd_dd_um2'][11]:.1f} um2 at 1 h, {cm['S15']['msd_dd_um2'][35]:.1f} at 3 h) while their tangential MSD reaches {cm['S15']['msd_t_um2'][35]:.0f} um2 at 3 h; interior depth diffuses (slope {cm['interior']['alpha_dd_30_120min']:.2f}, {cm['interior']['msd_dd_um2'][35]:.0f} um2 at 3 h).
d: 25 random surface tracks of >= 6 h: depth fluctuates within the layer (shaded, d > -15 um) with occasional dives below -20 um (dotted).
""")


# ============================================================================ 3: flow fields
def fig_flow(S):
    df = S.df
    X, V, t0, face = S.cache["X"], S.cache["V"], S.cache["t0"], S.cache["face"]
    wins = [25, 121, 217]
    fig, axs = plt.subplots(3, 3, figsize=(15, 13), gridspec_kw={"hspace": 0.18, "wspace": 0.08})
    allX = df[["x", "y"]].values; allt = df["t"].values
    lim_x = (allX[:, 0].min() - 15, allX[:, 0].max() + 15); lim_y = (allX[:, 1].min() - 15, allX[:, 1].max() + 15)
    h = 18.0
    stats = []
    for j, tw in enumerate(wins):
        cg, kap = M.outline(allX[(allt >= tw) & (allt < tw + 24)])
        cl, tp = M.clefts_tips(cg, kap)
        for i, fc in enumerate(("bottom", "top", "rim")):
            ax = axs[i, j]
            m = (t0 == tw) & (face == fc)
            P, Q = X[m, :2], V[m, :2]
            ax.plot(np.r_[cg[:, 0], cg[:1, 0]], np.r_[cg[:, 1], cg[:1, 1]], color="0.55", lw=0.8)
            ax.scatter(cg[cl, 0], cg[cl, 1], marker="v", s=35, c=INK, zorder=5)
            ax.scatter(P[:, 0], P[:, 1], s=1, c=FACE_C[fc], alpha=0.25, lw=0)
            ix = np.floor((P - [lim_x[0], lim_y[0]]) / h).astype(int)
            key = ix[:, 0] * 1000 + ix[:, 1]
            g = pd.DataFrame({"k": key, "x": P[:, 0], "y": P[:, 1], "u": Q[:, 0], "v": Q[:, 1]}).groupby("k")
            mm = g.mean()[g.size() >= 8]
            ax.quiver(mm["x"], mm["y"], mm["u"], mm["v"], color=FACE_C[fc], angles="xy", scale_units="xy",
                      scale=0.25, width=0.004, headwidth=3.5)
            spd = np.hypot(mm["u"], mm["v"])
            stats.append((fc, tw, float(spd.median()), float(np.sqrt((Q ** 2).sum(1).mean()))))
            ax.set_xlim(*lim_x); ax.set_ylim(*lim_y); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            lab = f"{fc} face" if fc != "rim" else "rim (x-y part)"
            panel(ax, "abcdefghi"[i * 3 + j], f"{lab}, movie {hours(tw):.0f}-{hours(tw + 24):.0f} h", box=True)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0 and j == 0:
                ax.quiver([lim_x[0] + 20], [lim_y[0] + 15], [10.0], [0.0], color=INK, angles="xy",
                          scale_units="xy", scale=0.25, width=0.004)
                ax.text(lim_x[0] + 20, lim_y[0] + 25, "10 um/h", fontsize=8)
                ax.plot([lim_x[1] - 120, lim_x[1] - 20], [lim_y[0] + 15] * 2, color=INK, lw=1.5)
                ax.text(lim_x[1] - 95, lim_y[0] + 22, "100 um", fontsize=8)
    st = pd.DataFrame(stats, columns=["face", "t0", "grid_speed_median", "cell_rms"])
    co = S.m4["b_coherence"]
    save(fig, "gland_surface_flow", f"""
No streams or swirls on the bud surface: the 2-h mean flow of surface cells is a weak, patchy drift, a fifth of the cells' own speed.
Arrows: mean tangential velocity (30-min displacements of S15 nuclei, gland drift removed, normal at the start) over 2-h windows in {h:.0f}-um bins of the x-y projection, bins with >= 8 samples; rows bottom (glass side), top (filter side), rim (x-y part of its velocity); columns movie 2-4 h, 10-12 h, 18-20 h; grey outline = x-y footprint, triangles = clefts (curvature < -1/60 per um).
Binned mean speed (median over bins) {st['grid_speed_median'].median():.1f} um/h against the cells' rms 30-min speed {st['cell_rms'].median():.1f} um/h; the face's mean flow carries {co['bottom']['mean_flow_share_median']:.1%} (bottom), {co['top']['mean_flow_share_median']:.1%} (top), {co['rim']['mean_flow_share_median']:.1%} (rim) of the cells' squared 30-min displacement.
No arrow pattern persists from one window to the next; the gland clefts (5 -> 9 clefts on the outline) without a visible stream into or out of the clefts.
""")


# ============================================================================ 4: coherence and global motions
def fig_coherence(S):
    m, m2 = S.m4, S.m2
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.9), gridspec_kw={"wspace": 0.42})
    co = m["b_coherence"]; r = np.array(co["r_mid_spacings"])
    ax = axs[0]
    for fc, col in FACE_C.items():
        ax.plot(r, co[fc]["C_r"]["face_mean"], color=col, lw=1.6, label=f"{fc}")
        ax.plot(r, co[fc]["C_r"]["local50"], color=col, lw=0.9, ls="--")
        ax.plot(r, co[fc]["C_r"]["shuffled"], color=col, lw=0.7, ls=":")
    ax.axhline(1 / np.e, color=MUTED, lw=0.8); ax.text(4.5, 0.39, "1/e", fontsize=7, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_ylim(-0.15, 0.5)
    ax.text(3.0, -0.13, "solid face mean removed, dashed\n50-um local mean removed, dotted shuffled", fontsize=6.5)
    ax.legend(loc="upper right")
    panel(ax, "a", "C(r) of 30-min tangential velocities", "r (spacings of 13.9 um)")
    ax = axs[1]
    nd = m["b_neighbour_displacement_corr"]
    shells = list(nd["5min"].keys())
    rm = [np.mean([float(x) for x in s.split("-")]) for s in shells]
    cols = plt.cm.Reds(np.linspace(0.35, 0.95, len(nd)))
    for (L, v), col in zip(nd.items(), cols):
        ax.plot(rm, [v[s]["c"] for s in shells], color=col, marker="o", ms=3, label=L)
    ax.legend(loc="upper right", title="lag", title_fontsize=7)
    ax.set_ylim(0, 0.5)
    panel(ax, "b", "displacement shared by a pair, c",
          f"pair distance at start (S15 spacings of {m['layer_spacing_um']['S15']:.1f} um)")
    ax = axs[2]
    for mv, mk, ls in ((m, "o", "-"), (m2, "s", "--")):
        rows = mv["b_flows_windows"]
        t = [hours(rw["t0"] + 12) for rw in rows]
        for fc, col in FACE_C.items():
            ax.plot(t, [rw[fc]["omega_deg_h"] if fc in rw else np.nan for rw in rows], color=col, marker=mk, ms=3,
                    ls=ls, lw=1, label=fc if mv is m else None)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.98, 0.02, "solid 25x-4, dashed 25x-2", transform=ax.transAxes, ha="right", fontsize=7)
    panel(ax, "c", "rotation of each face (deg/h)", "movie time (h), 2-h windows")
    ax = axs[3]
    for mv, col, nm in ((m, RED, "25x-4"), (m2, BLUE, "25x-2")):
        rows = [rw for rw in mv["b_flows_windows"] if "top_minus_bottom_um_h" in rw]
        t = [hours(rw["t0"] + 12) for rw in rows]
        tb = np.array([rw["top_minus_bottom_um_h"] for rw in rows])
        ax.plot(t, tb[:, 0], color=col, marker="o", ms=3, label=f"{nm}: x")
        ax.plot(t, tb[:, 1], color=col, marker="s", ms=3, ls="--", label=f"{nm}: y")
    ax.axhline(0, color=MUTED, lw=0.6); ax.legend(loc="upper left", fontsize=7, ncol=2)
    panel(ax, "d", "top minus bottom face velocity (um/h)", "movie time (h), 2-h windows")
    om = [abs(rw[fc]["omega_deg_h"]) for rw in m["b_flows_windows"] for fc in FACE_C if fc in rw]
    om2 = np.array([[rw[fc]["omega_deg_h"] for fc in FACE_C] for rw in m2["b_flows_windows"]])
    early2 = om2[:5].mean(); late2 = om2[5:].mean()
    tb = np.array([rw["top_minus_bottom_um_h"] for rw in m["b_flows_windows"] if "top_minus_bottom_um_h" in rw])
    save(fig, "gland_surface_coherence", f"""
Surface cells move independently of their neighbours: the velocity correlation length is below one cell spacing on every face, and the bud surface does not rotate.
a: C(r) = <v_i.v_j>/<|v|^2> of 30-min tangential displacements of S15 nuclei, per face and 30-min window (44 windows, ~170-245 cells each), pooled; C at the first bin (0.5-1 spacings of 13.9 um) {co['bottom']['C_r']['face_mean'][0]:.2f} / {co['top']['C_r']['face_mean'][0]:.2f} / {co['rim']['C_r']['face_mean'][0]:.2f} (bottom/top/rim); exp21.velocity_correlation_length returns its floor, 0.73 spacings (< 10 um), in every window and face, the same as velocities shuffled among the cells.
b: the displacement two cells share grows with lag (slow, smooth tissue flow): nearest pairs c {m['b_neighbour_displacement_corr']['5min']['0-1.25']['c']:.2f} at 5 min, {m['b_neighbour_displacement_corr']['30min']['0-1.25']['c']:.2f} at 30 min, {m['b_neighbour_displacement_corr']['60min']['0-1.25']['c']:.2f} at 1 h, {m['b_neighbour_displacement_corr']['120min']['0-1.25']['c']:.2f} at 2 h; pairs 5-8 spacings apart {m['b_neighbour_displacement_corr']['120min']['5-8']['c']:.2f} at 2 h.
c: face rotation omega = sum (r x v)_z / sum r^2 about each face's centroid, 2-h windows: movie 25x-4 |omega| <= {max(om):.2f} deg/h with changing sign (at the rim, 160 um out, <= {np.radians(max(om)) * 160:.1f} um/h against ~7 um/h cell speed) -- no global rotation; movie 25x-2 turns slowly counter-clockwise in its first 10 h (all faces, mean {early2:+.2f} deg/h, ~{np.radians(early2) * 160:.1f} um/h at the rim), {late2:+.2f} deg/h after.
d: one steady signal: the top (filter-side) face moves relative to the bottom (glass-side) face by ({tb[:, 0].mean():+.1f}, {tb[:, 1].mean():+.1f}) um/h in movie 25x-4, x always positive in both movies: a slow shear of the pancake (rolling, or a depth-dependent registration drift; tracks cannot tell).
""")


# ============================================================================ 5: MSD, persistence, speeds, turns
def fig_msd(S):
    m = S.m4
    fig, axs = plt.subplots(1, 5, figsize=(20, 3.9), gridspec_kw={"wspace": 0.42})
    ax = axs[0]
    for g, col in (("S15", RED), ("interior", BLUE)):
        c = m["c_msd"][g]; lag = np.array(c["lag_h"]) * 60
        ax.loglog(lag, c["msd_t_um2"], color=col, lw=1.8, label=f"{g} tangential")
        ax.loglog(lag, np.array(c["msd_t_um2"]) - c["noise_t_um2"], color=col, lw=0.9, ls="--",
                  label=f"{g} minus noise")
    c = m["c_msd"]["S15"]
    ax.loglog(np.array(c["lag_h"]) * 60, c["msd_dd_um2"], color=RED, lw=1.0, ls=":", label="S15 depth change")
    x = np.array([5, 180.0])
    ax.loglog(x, 1.0 * (x / 5) ** 1, color=MUTED, lw=0.7); ax.text(150, 30, "slope 1", fontsize=7, color=MUTED)
    ax.loglog(x[:1].tolist() + [40], [0.6, 0.6 * 64], color=MUTED, lw=0.7, ls="-.")
    ax.text(30, 30, "slope 2", fontsize=7, color=MUTED)
    ax.legend(loc="upper left", fontsize=6.5)
    panel(ax, "a", "MSD (um2)", "lag (min)")
    ax = axs[1]
    for g, col in (("S15", RED), ("interior", BLUE)):
        c = m["c_msd"][g]; lag = np.array(c["lag_h"]); y = np.array(c["msd_t_um2"])
        for yy, ls, lab in ((y, "-", "raw"), (y - c["noise_t_um2"], "--", "minus noise")):
            sl = np.gradient(np.log(yy), np.log(lag))
            ax.plot(lag * 60, sl, color=col, ls=ls, lw=1.4 if ls == "-" else 0.9, label=f"{g} {lab}")
    ax.axhline(1, color=MUTED, lw=0.6); ax.set_xscale("log"); ax.set_ylim(0.8, 2.0)
    ax.legend(loc="upper right", fontsize=6.5)
    panel(ax, "b", "local exponent alpha (tangential MSD)", "lag (min)")
    ax = axs[2]
    v = m["c_vacf_tangential_S15"]; lag = np.array(v["lag_h"])
    ax.plot(lag, v["C"], color=RED, marker="o", ms=3, label="<u0.uk> / <|u|2>")
    ax.plot(lag, v["cos"], color=RED, marker="s", ms=3, ls="--", lw=0.9, label="<cos angle>")
    fp = v["fit_A_exp_minus_lag_over_P_plus_b"]
    xx = np.linspace(0.05, 3, 100)
    ax.plot(xx, fp["A"] * np.exp(-xx / fp["P_h"]) + fp["b"], color=INK, lw=0.7,
            label=f"{fp['A']:.2f} exp(-t/{fp['P_h']:.2f} h) + {fp['b']:.3f}")
    ax.axhline(0, color=MUTED, lw=0.5); ax.legend(loc="upper right", fontsize=7)
    panel(ax, "c", "autocorrelation, 15-min steps (S15)", "lag (h)")
    ax = axs[3]
    sd = m["c_speed_distributions"]["S15"]
    for L, col in (("5min", RED), ("30min", "#e8a07f")):
        e = np.array(sd[L]["edges"]); hh = np.array(sd[L]["hist"], float); w = np.diff(e)
        dens = hh / (hh.sum() * w)
        ax.step(e[:-1], dens, where="post", color=col, lw=1.5, label=f"{L} median {sd[L]['median']:.1f}, CV {sd[L]['cv']:.2f}")
        mean = sd[L]["mean"]; sig = mean / np.sqrt(np.pi / 2)
        xs = np.linspace(0, e[-1], 200)
        ax.plot(xs, xs / sig ** 2 * np.exp(-xs ** 2 / (2 * sig ** 2)), color=col, lw=0.7, ls=":")
    ax.text(0.97, 0.55, "dotted: Rayleigh (2D Gaussian\nvelocity) of the same mean, CV 0.52", transform=ax.transAxes,
            ha="right", fontsize=6.5)
    ax.legend(loc="upper right", fontsize=7)
    panel(ax, "d", "density, S15 tangential speed", "speed over the lag (um/h)")
    ax = axs[4]
    tc = m["c_turn_correlation"]
    labs = ["same turn sign", "corr of turn (sin)", "corr of velocity change", "corr of velocity"]
    keys = ["same_turn_sign_frac", "turn_sin_corr", "dv_corr", "v_corr"]
    xx = np.arange(len(keys))
    near = [tc["near"][k] - (0.5 if k == "same_turn_sign_frac" else 0) for k in keys]
    far = [tc["far"][k] - (0.5 if k == "same_turn_sign_frac" else 0) for k in keys]
    ax.bar(xx - 0.18, near, 0.36, color=RED, label=f"neighbours < 1.25 spacings ({tc['near']['pairs']} pairs)")
    ax.bar(xx + 0.18, far, 0.36, color="#e8a07f", label=f"3-5 spacings ({tc['far']['pairs']} pairs)")
    ax.set_xticks(xx); ax.set_xticklabels(["same sign\n- 0.5", "turn\ncorr", "dv\ncorr", "v\ncorr"], fontsize=7)
    ax.axhline(0, color=MUTED, lw=0.6); ax.legend(loc="upper left", fontsize=6.5)
    panel(ax, "e", "do neighbours turn together? (15-min steps)")
    cs = m["c_msd"]["S15"]; tr = m["c_speed_distributions"]["track_mean_tangential_30min"]
    save(fig, "gland_surface_msd_persistence", f"""
A persistent random walk with a 10-15 min memory, independent between neighbours, on top of a slow shared drift that keeps the MSD slightly super-diffusive out to 3 h.
a-b: tangential MSD of S15 cells (drift removed, any two spots of a track L frames apart): log-log slope {cs['alpha_t_5_30min']:.2f} over 5-30 min ({cs['alpha_t_noise_removed_5_30min']:.2f} with the fitted noise {cs['noise_t_um2']:.1f} um2 removed), {cs['alpha_t_30_120min']:.2f} over 30-120 min, {cs['alpha_t_120_180min']:.2f} over 2-3 h; interior {m['c_msd']['interior']['alpha_t_30_120min']:.2f} over 30-120 min. exp11.prw_fit on the 3D MSD (lags 5 min - 2 h): v {cs['prw3']['v_um_h']:.1f} um/h, P {cs['prw3']['P_h']:.2f} h (interior {m['c_msd']['interior']['prw3']['v_um_h']:.1f} um/h, P {m['c_msd']['interior']['prw3']['P_h']:.2f} h); MSD(1 h) {cs['msd3_1h_um2']:.0f} um2 3D, {cs['msd_t_1h_um2']:.0f} tangential.
c: 15-min tangential steps decorrelate as {v['fit_A_exp_minus_lag_over_P_plus_b']['A']:.2f} exp(-lag / {v['fit_A_exp_minus_lag_over_P_plus_b']['P_h']:.2f} h) + {v['fit_A_exp_minus_lag_over_P_plus_b']['b']:.3f}: the plateau is the shared slow flow.
d: tangential speeds: 5-min median {sd['5min']['median']:.1f} um/h (p10-p90 {sd['5min']['p10_p90'][0]:.1f}-{sd['5min']['p10_p90'][1]:.1f}), 30-min net {sd['30min']['median']:.1f} um/h; CV {sd['5min']['cv']:.2f} vs Rayleigh 0.52: cells differ -- per-track means over non-overlapping 30-min windows hold {tr['between_track_variance_share']:.2f} of the variance vs {tr['between_track_variance_share_shuffled']:.2f} shuffled, an intrinsic cell-to-cell speed CV of ~{tr['intrinsic_cv_of_cell_speed']:.2f}.
e: neighbours do not turn together: turn signs agree {tc['near']['same_turn_sign_frac']:.3f} (neighbours) vs {tc['far']['same_turn_sign_frac']:.3f} (3-5 spacings); velocity-change correlation {tc['near']['dv_corr']:.2f} vs {tc['far']['dv_corr']:.2f}.
""")


# ============================================================================ 6: neighbour exchange
def fig_exchange(S):
    m, m2 = S.m4, S.m2
    fig, axs = plt.subplots(1, 4, figsize=(17, 5.0), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 1.25, 1]})
    fig.subplots_adjust(bottom=0.3)
    ax = axs[0]
    for g, col, ls, lab in (("S8", RED, "--", "S8 (d0 ~14 um)"), ("S15", RED, "-", "S15 (d0 ~10.5 um)"),
                            ("interior", BLUE, "-", "interior (d0 ~9.7 um)")):
        cv = m["d_retention"][g]["curve"]
        x = [v["lag_h"] for v in cv.values() if v["mean"] is not None and v["windows"] >= 5]
        y = [v["mean"] for v in cv.values() if v["mean"] is not None and v["windows"] >= 5]
        e = [v["sd_over_windows"] for v in cv.values() if v["mean"] is not None and v["windows"] >= 5]
        ax.errorbar(x, y, yerr=e, color=col, ls=ls, marker="o", ms=3, capsize=2, lw=1.3, label=lab)
    cv = m2["d_retention"]["S15"]["curve"]
    ax.plot([v["lag_h"] for v in cv.values()], [v["mean"] for v in cv.values()], color=RED, lw=0.8, ls=":",
            label="S15, movie 25x-2")
    ax.scatter([1.0], [0.883], marker="*", s=90, c=INK, zorder=6, label="exp 21 target 0.883")
    ax.set_ylim(0.3, 1.02); ax.legend(loc="lower left", fontsize=7)
    panel(ax, "a", "pairs < 1.25 d0 still < 1.5 d0", "lag (h)")
    ax = axs[1]
    k1 = m["d_retention"]["S15_kept_1h_vs_fixed_d0_um"]; k2 = m2["d_retention"]["S15_kept_1h_vs_fixed_d0_um"]
    x = np.array([float(a) for a in k1])
    ax.plot(x, list(k1.values()), color=RED, marker="o", ms=4, label="movie 25x-4")
    ax.plot(x, list(k2.values()), color=BLUE, marker="s", ms=4, label="movie 25x-2")
    ax.axvline(14, color=MUTED, ls=":", lw=0.8); ax.text(14.3, 0.32, "model\ncell 14 um", fontsize=7)
    ax.axvline(7.3, color=MUTED, ls="--", lw=0.8); ax.text(7.5, 0.32, "nuclear\nspacing ~7-8 um", fontsize=7)
    ax.set_ylim(0.15, 1.0); ax.legend(loc="lower right", fontsize=7)
    panel(ax, "b", "kept at 1 h, S15 layer", "d0 handed to the estimator (um)")
    ax = axs[2]
    cats = ["dive", "insertion", "T1_flank", "intercalation", "untracked", "empty"]
    names = ["partner dived", "cell from below\ninserted", "flanking cell\nmoved in (T1-like)",
             "other layer cell\nmoved in", "untracked cell\nbetween", "no nucleus\nbetween"]
    cc = ["#1f6fb8", "#7a4fa0", "#c0522a", "#e8a07f", "0.65", "0.85"]
    rows = [("25x-4, 30 min", m["d_mechanism_30min"]), ("25x-4, 1 h", m["d_mechanism_1h"]),
            ("25x-2, 30 min", m2["d_mechanism_30min"]), ("25x-2, 1 h", m2["d_mechanism_1h"])]
    for k, (lab, me) in enumerate(rows):
        left = 0.0
        for cat, col, nm in zip(cats, cc, names):
            w = me["lost_share"][cat]
            ax.barh(k, w, left=left, color=col, label=nm if k == 0 else None, height=0.7)
            left += w
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7)
    panel(ax, "c", "how lost S15 pairs came apart (share)")
    ax = axs[3]
    labs, vals, cols = [], [], []
    for fc in ("bottom", "top", "rim"):
        labs.append(fc); vals.append(m["d_retention"][f"S15_{fc}"]["curve"]["60min"]["mean"]); cols.append(FACE_C[fc])
    for cls in ("tip", "flank", "cleft"):
        labs.append(f"rim {cls}"); vals.append(m["d_rim_tip_vs_cleft"][cls]["60min"]["mean"]); cols.append(FACE_C["rim"])
    ax.bar(range(len(vals)), vals, color=cols, alpha=0.9)
    for k, v in enumerate(vals):
        ax.text(k, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, rotation=30, fontsize=7.5); ax.set_ylim(0, 1)
    panel(ax, "d", "kept at 1 h by place (S15)")
    s8, s15 = m["d_retention"]["S8"], m["d_retention"]["S15"]
    me = m["d_mechanism_1h"]
    save(fig, "gland_surface_neighbour_exchange", f"""
Surface neighbours part gradually, at a rate set by how far apart 'neighbours' are taken to be: kept_1h 0.883 at d0 14 um, 0.73 at 10 um, 0.58 at 8 um.
a: exp21.neighbour_retention, the windows and d0 of tools/exp21_wang_neighbours.py (reproduced: S8 {s8['curve']['30min']['mean']:.3f} / {s8['curve']['60min']['mean']:.3f} / {s8['curve']['120min']['mean']:.3f} at 0.5 / 1 / 2 h); S8 time course {s8['curve']['5min']['mean']:.3f} (5 min), {s8['curve']['15min']['mean']:.3f} (15 min), {s8['curve']['45min']['mean']:.3f} (45 min), {s8['curve']['90min']['mean']:.3f} (1.5 h); S15 {s15['curve']['30min']['mean']:.2f} / {s15['curve']['60min']['mean']:.2f} / {s15['curve']['120min']['mean']:.2f} / {s15['curve']['180min']['mean']:.2f} at 0.5 / 1 / 2 / 3 h; exponential exchange time 9.3 h (S8), 4.5 h (S15), 3.8 h (interior). Points with < 5 windows dropped (S8 has only 2-5 beyond 2 h).
b: kept_1h against the absolute d0 handed to the estimator (S15 nuclei): the gate's 0.883 is the value for 14-um cells; the gland's own nuclei are ~7-10 um apart.
c: lost pairs (25x-4, 1 h): partner dived {me['lost_share']['dive']:.0%}, cell from below inserted {me['lost_share']['insertion']:.0%}, flanking cell moved in (T1-like) {me['lost_share']['T1_flank']:.0%}, other layer cell {me['lost_share']['intercalation']:.0%}, untracked/no nucleus between {me['lost_share']['untracked'] + me['lost_share']['empty']:.0%} (2/3 of nuclei are untracked). They part from {me['lost_r_t0_over_d0_median']:.2f} to {me['lost_r_t1_over_d0_median']:.2f} d0 in 1 h; the largest 15-min rise is {me['lost_largest_15min_rise_share_median']:.2f} of the whole rise vs {me['lost_largest_15min_rise_share_brownian_bridge_null']:.2f} for a diffusive Brownian-bridge null: gradual sliding, no discrete T1 jump.
d: kept_1h by face (bottom {m['d_retention']['S15_bottom']['curve']['60min']['mean']:.2f}, top {m['d_retention']['S15_top']['curve']['60min']['mean']:.2f}, rim {m['d_retention']['S15_rim']['curve']['60min']['mean']:.2f}) and along the rim within 20 um of a tip / flank / cleft ({m['d_rim_tip_vs_cleft']['tip']['60min']['mean']:.2f} / {m['d_rim_tip_vs_cleft']['flank']['60min']['mean']:.2f} / {m['d_rim_tip_vs_cleft']['cleft']['60min']['mean']:.2f}; 10-14 pairs per window, 11-31 windows: tips exchange somewhat faster).
""")


# ============================================================================ 7: dives
def fig_dives(S):
    df, C, m = S.df, S.C, S.m4
    out, ev = M.dives(df, C)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"wspace": 0.35})
    ax = axs[0]
    tid_of = df["tid"].values
    rng = np.random.default_rng(1)
    for ret, col, n in ((True, RED, 12), (False, BLUE, 12)):
        e = ev[ev["returned"] == ret]
        for _, row in e.sample(min(n, len(e)), random_state=1).iterrows():
            tr = df[df["tid"] == tid_of[int(row["i_last"])]]
            tt = tr["t"].values
            k = (tt >= row["t_dive"] - 24) & (tt <= (row["t_last"] if not ret else row["t_return"]) + 24)
            ax.plot((tt[k] - row["t_dive"]) * M.DT_H, tr["d"].values[k], color=col, lw=0.7, alpha=0.7)
    ax.plot([], [], color=RED, label="returned to the layer"); ax.plot([], [], color=BLUE, label="track ended deep")
    ax.axhspan(-15, 5, color=RED, alpha=0.08, lw=0); ax.axhline(M.DEEP, color=MUTED, ls=":", lw=0.8)
    ax.legend(loc="lower left", fontsize=7); ax.set_ylim(-40, 5)
    panel(ax, "a", "d (um), 24 dives aligned at the dive", "time from dive (h)")
    ax = axs[1]
    ax.hist(ev["d_min"], bins=np.arange(-50, -19, 1.5), color=BLUE, alpha=0.8)
    panel(ax, "b", f"dives ({len(ev)} in 18,300 surface cell-hours)", "deepest d reached (um)")
    ax = axs[2]
    for mv, col, nm in ((m, RED, "25x-4"), (S.m2, BLUE, "25x-2")):
        e = mv["e_dives"]
        ax.plot([0, 1, 2, 4], [1, e["still_surface_after_1h"], e["still_surface_after_2h"], e["still_surface_after_4h"]],
                color=col, marker="o", ms=4, label=nm)
    ax.axhline(0.930, color=MUTED, ls=":", lw=0.8); ax.text(0.1, 0.934, "Wang's on-surface ratio 0.930", fontsize=7)
    ax.set_ylim(0.8, 1.01); ax.legend(loc="lower left")
    panel(ax, "c", "S15 cells still in the layer", "lag (h)")
    save(fig, "gland_surface_dives", f"""
Surface cells leave the layer about once a day and their track usually ends while deep, as a Type I division's would; 86 % are still in the layer 4 h later.
a-b: a dive = a surface-layer track (d > -15 um) reaching d < -20 um (hysteresis, gaps <= 2 frames bridged): {out['n_dives']} dives in {out['surface_cell_hours']:.0f} surface cell-hours = {out['dives_per_surface_cell_hour']:.3f} per cell-hour (one per ~{1 / out['dives_per_surface_cell_hour']:.0f} h; 25x-2 {S.m2['e_dives']['dives_per_surface_cell_hour']:.3f}); deepest d median {out['d_min_um_median']:.1f} um (p10 {out['d_min_um_p10']:.1f}); {out['returned_in_same_track_frac']:.0%} return within the same track after {out['deep_duration_h_returned_median']:.1f} h (p90 {out['deep_duration_h_returned_p90']:.1f} h), {out['track_ended_while_deep_frac']:.0%} of tracks end while deep -- consistent with Wang's dive-divide-return (Fig 2A-B), where daughters get new ids; a new track starts within 12 um / 15 min after {out['ended_deep_with_new_track_within_12um_3frames_frac']:.0%} of those ends vs {out['control_random_deep_spot_with_new_track_frac']:.0%} around random deep spots.
c: of the S15 cells at t, still S15 at t + 1 / 2 / 4 h: {out['still_surface_after_1h']:.3f} / {out['still_surface_after_2h']:.3f} / {out['still_surface_after_4h']:.3f} (Wang's on-surface ratio of manually tracked cells, ~5 h: 0.930).
""")


# ============================================================================ 8: 3D tracks (+ movie)
def _track_segments(df, C, t_from, t_to, sel_rows=None, tids=None):
    """Line segments (drift NOT removed: the picture is the gland as imaged) of tracks between t_from and t_to,
    for tracks whose first spot in the window is selected (or the given track ids); each segment carries its frame."""
    g = df[(df["t"] >= t_from) & (df["t"] <= t_to)]
    if tids is None:
        first = g.groupby("tid").head(1)
        tids = set(first.loc[first.index.isin(sel_rows), "tid"])
    keep = tids
    g = g[g["tid"].isin(keep)].sort_values(["tid", "t"])
    P = g[["x", "y", "z"]].values; t = g["t"].values; tid = g["tid"].values
    ok = (tid[1:] == tid[:-1]) & (np.diff(t) <= 2)
    seg = np.stack([P[:-1][ok], P[1:][ok]], 1)
    return seg, t[1:][ok]


def fig_tracks3d(S, movie=True):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    df, C = S.df, S.C
    t_a, t_b = 145, 169                                         # 2 h
    sel = df.index[(df["t"] == t_a) & (df["d"] > M.S15)]
    seg, tseg = _track_segments(df, C, t_a, t_b, set(sel))
    ctx = df[df["t"] == t_a]
    fig = plt.figure(figsize=(17, 7.5))
    ax = fig.add_axes([0.0, 0.0, 0.6, 0.95], projection="3d")
    ax.scatter(ctx["x"], ctx["y"], ctx["z"], s=2, c="0.75", alpha=0.25, lw=0)
    norm = plt.Normalize(hours(t_a), hours(t_b))
    lc = Line3DCollection(seg, colors=plt.cm.plasma(norm(hours(tseg))), linewidths=1.5)
    ax.add_collection3d(lc)
    rng_ = np.ptp(ctx[["x", "y", "z"]].values, 0)
    ax.set_box_aspect(rng_, zoom=1.45)
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()
    ax.text2D(0.04, 0.93, "a", transform=ax.transAxes, fontweight="bold", fontsize=13)
    ax.text2D(0.07, 0.93, f"surface-layer tracks, movie {hours(t_a):.0f}-{hours(t_b):.0f} h, over the gland's nuclei",
              transform=ax.transAxes, fontsize=9)
    ax2 = fig.add_axes([0.6, 0.08, 0.3, 0.8])
    bsel = df.index[(df["t"] == t_a) & (df["face"] == "bottom")]
    seg2, t2 = _track_segments(df, C, t_a, t_b, set(bsel))
    ax2.scatter(ctx["x"], ctx["y"], s=2, c="0.85", lw=0)
    ax2.add_collection(LineCollection(seg2[:, :, :2], colors=plt.cm.plasma(norm(hours(t2))), linewidths=1.1))
    ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    x0, y0 = ctx["x"].min(), ctx["y"].min()
    ax2.plot([x0, x0 + 50], [y0 - 8] * 2, color=INK, lw=1.5); ax2.text(x0, y0 - 5, "50 um", fontsize=8)
    panel(ax2, "b", "bottom (glass-side) face, x-y: tracks coloured by time", box=True)
    for s in ax2.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="plasma"), cax=cax)
    cb.set_label("movie time (h)"); cb.outline.set_visible(False)
    stem = "gland_surface_tracks_3d"
    if movie:
        _movie_tracks(S, stem)
    n_tr = len(set(df.loc[sel, "tid"]))
    save(fig, stem, f"""
The gland's surface cells over 2 h: short, wandering, uncoordinated paths along the surface.
a: tracks of the {n_tr} nuclei in the surface layer (d > -15 um) at movie {hours(t_a):.0f} h, followed for 2 h (frames {t_a}-{t_b}), drawn as imaged (gland drift kept), coloured by time, over every tracked nucleus at the start (grey); the gland is a lobed pancake between filter and glass (~430 x 350 x 100 um).
b: the bottom face in x-y projection: neighbouring paths point every way; net 2-h displacements are ~10 um (30-min net speed ~6 um/h; 5-min steps ~12 um/h along the surface, largely localisation noise and jitter).
Movie {stem}.mp4: the same layer over 6 h with 1-h tails, rotating.
""")


def _movie_tracks(S, stem, t_start=121, n_frames=72, tail=12):
    import shutil
    import tempfile
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    df = S.df
    tmp = tempfile.mkdtemp(prefix="exp21_gland_mov_")
    P_all = df[["x", "y", "z"]].values
    lo, hi = P_all.min(0), P_all.max(0)
    for k in range(n_frames):
        t = t_start + k
        sel = df.index[(df["t"] == t) & (df["d"] > M.S15)]
        seg, ts = _track_segments(df, S.C, t - tail, t, tids=set(df.loc[sel, "tid"]))
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        cur = df[df["t"] == t]
        ax.scatter(cur["x"], cur["y"], cur["z"], s=1.5, c="0.8", alpha=0.25, lw=0)
        age = (t - ts) / tail
        cols = plt.cm.plasma(0.05 + 0.7 * age)                 # recent = dark purple, an hour old = orange
        cols[:, 3] = 1 - 0.6 * age
        ax.add_collection3d(Line3DCollection(seg, colors=cols, linewidths=1.6))
        s = cur[cur["d"] > M.S15]
        ax.scatter(s["x"], s["y"], s["z"], s=7, c=RED, lw=0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.9); ax.view_init(elev=30, azim=-70 + 1.0 * k); ax.set_axis_off()
        fig.text(0.03, 0.94, f"surface-layer nuclei (red) with 1-h tails (dark = recent), movie {hours(t):.1f} h",
                 fontsize=12, color=INK)
        fig.savefig(os.path.join(tmp, f"{k:05d}.png"), facecolor="white")
        plt.close(fig)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", "12", "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=1280:720", "-pix_fmt", "yuv420p", "-c:v", "libx264", os.path.join(FIGS, stem + ".mp4")],
                   check=True)
    shutil.rmtree(tmp)


FIGURES = {"layer": fig_layer, "speed": fig_speed, "flow": fig_flow, "coherence": fig_coherence, "msd": fig_msd,
           "exchange": fig_exchange, "dives": fig_dives, "tracks3d": fig_tracks3d}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default=",".join(FIGURES))
    a = ap.parse_args()
    S = Data()
    for k in a.only.split(","):
        FIGURES[k](S)


if __name__ == "__main__":
    main()
