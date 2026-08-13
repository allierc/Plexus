#!/usr/bin/env python
"""test_00_spheroid -- the first operator test: the spheroid on its own, and nothing else.

    python test_00_spheroid.py            ->  log/okuda_ECM/00_spheroid/

THE EASY ONE, AND IT IS FIRST FOR THAT REASON. Before any operator between the three entities is
written, the entity every one of them attaches to has to be reproducible on its own and drawn in the
form every later test will be compared against. So this folder contains the epithelium of the nominal
run and no basement membrane, no matrix, no adhesion: the tissue `153_nominal_material_E100` grew
against, replayed from the pass-1 cache it was built from, measured and rendered.

WHAT IS TESTED HERE is the growth line -- `mesh_seed`, `cell_geometry`, `cell_grow`,
`cell_mechanics`, `cell_divide`, `edge_flip` -- through what it produced: cell number, apical
radius, the two semi-axes, and the T1 rate. Those four are what every later operator has to leave
alone, so they are written to `metrics.json` and plotted.

THE ARTEFACTS, in the shape the rest of the prototype uses: `movie.mp4` with the 3D view beside the
true cross-section, `section.mp4` zoomed on the epithelium's own wall, `3d.png` for the end state, and
`spec.yaml` naming the cache and the pass-1 parameters that made it.

WHERE THE MODEL COMES FROM, since a test that cites nothing reads as a model with no provenance. The
epithelium is the 3D active vertex model of Okuda, S., Inoue, Y., Eiraku, M., Sasai, Y., Adachi, T.
(2013) Biomech. Model. Mechanobiol. 12(4):627 -- the reversible network reconnection that
`edge_flip` implements -- in the form Okuda, S., Miura, T., Inoue, Y., Adachi, T., Eiraku, M.
(2018) Sci. Rep. 8:2386 uses, and its ancestor is Honda, H., Tanemura, M., Nagai, T. (2004)
J. Theor. Biol. 226(4):439. The mesh is Tyssue (github.com/DamCB/tyssue).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "discovery_okuda", "ops")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import ListedColormap

import ecm_render as RD
import ecm_spec as ES

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
# THE NOMINAL'S OWN TISSUE, by name. `153_nominal_material_E100` and every run in the archived line
# grew against this cache; taking a fresh one would make the first test of the growth operators a test
# of a different tissue.
TISSUE = os.path.join(LOG, "_tissue", "cellfix_B_new_f401_x4_2cedf4bcc6.npz")
NOMINAL = "153_nominal_material_E100"


def measure(z, Tis):
    """The four numbers the growth line is responsible for, per frame."""
    n = np.asarray(z["n_cells"], float)
    r = np.asarray(z["r_apical"], float)
    r_eq = np.asarray(z["r_eq"], float) if "r_eq" in z.files else r
    r_ax = np.asarray(z["r_ax"], float) if "r_ax" in z.files else r
    t1 = np.asarray(z["t1_trace"], float) if "t1_trace" in z.files else np.zeros((1, 4))
    return dict(frames=int(len(n)), cells_start=int(n[0]), cells_end=int(n[-1]),
                r_start=float(r[0]), r_end=float(r[-1]), fold_radius=float(r[-1] / max(r[0], 1e-9)),
                fold_area=float((r[-1] / max(r[0], 1e-9)) ** 2),
                aspect_end=float(r_eq[-1] / max(r_ax[-1], 1e-9)),
                t1_total=int(t1[-1, 2]) if t1.size and t1.shape[1] > 2 else 0,
                t1_per_cell_per_frame=(float(t1[:, 1].sum() / max(t1[:, 3].mean(), 1) / max(len(t1), 1))
                                       if t1.size and t1.shape[1] > 3 else 0.0),
                series=dict(n_cells=n.tolist(), r_apical=r.tolist(),
                            r_eq=r_eq.tolist(), r_ax=r_ax.tolist()))


def plot_growth(m, out):
    """The measurement plot: what grew, by how much, and whether it stayed round."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), facecolor="white")
    n = np.asarray(m["series"]["n_cells"]); r = np.asarray(m["series"]["r_apical"])
    eq = np.asarray(m["series"]["r_eq"]); ax_ = np.asarray(m["series"]["r_ax"])
    t = np.arange(len(n))
    axes[0].plot(t, n, color="#1f8a5c", lw=1.6)
    axes[0].set_ylabel("cells"); axes[0].set_title(f"{m['cells_start']} -> {m['cells_end']}",
                                                   fontsize=9)
    axes[1].plot(t, r, color="#4aa3ff", lw=1.6)
    axes[1].set_ylabel("apical radius (tissue units)")
    axes[1].set_title(f"x{m['fold_radius']:.2f} in radius, x{m['fold_area']:.1f} in area", fontsize=9)
    # ROUNDNESS, not just size: an operator that changes the aspect has changed the geometry every
    # later test is measured against, and the ratio is the cheapest place to see it.
    axes[2].plot(t, eq / np.maximum(ax_, 1e-9), color="#e08a2e", lw=1.6)
    axes[2].axhline(1.0, color="#999", lw=0.8, ls="--")
    axes[2].set_ylabel("$r_{eq}/r_{ax}$"); axes[2].set_title(f"aspect {m['aspect_end']:.3f} at the end",
                                                             fontsize=9)
    for a in axes:
        a.set_xlabel("frame"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)


def render(Tis, d, frames=150, fps=15):
    """`movie.mp4` (3D + section), `section.mp4` (zoomed), `3d.png` (end state).

    Drawn through `ecm_render`'s own routines, with an EMPTY matrix rather than a stand-in for one:
    `q` is a (0,3) array, so the panels that would show stroma show none. A proxy cloud here would be
    the same mistake `run_ecm` documents for the prescribed sphere -- drawing something in the slot
    where a real entity belongs.
    """
    meshes = Tis["meshes"]
    cmap = ListedColormap(ES.STRESS_COLORS)
    q = np.zeros((0, 3)); band = np.zeros(0, np.uint8)
    L3 = Tis["Lbox"] * 1.60
    L2 = L3 * 1.15
    keep = [meshes[int(round(f * (len(meshes) - 1)))]
            for f in np.linspace(0, 1, min(frames, len(meshes)))]
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    fig = plt.figure(figsize=(11.0, 5.5), facecolor="black")
    axs = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False, facecolor="black")
    axc = fig.add_subplot(1, 2, 2, facecolor="black")
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02)

    def frame(t, mt, ax3, ax2, label=True):
        vp = mt["pos"]
        RD.draw_3d(ax3, mt, vp, q, band, cmap, RD.CAM_SIDE, 0.72 * L3,
                   div=RD.divided_mask(mt), brk=RD.broken_mask(mt, vp, "00"))
        RD.draw_cross(ax2, mt, vp, q, band, cmap, L2, np.eye(3)[2], 0.055)
        if label:
            ax3.text2D(0.03, 0.95, f"00_spheroid   frame {t}\n{int(mt['nF'])} cells",
                       transform=ax3.transAxes, color="white", fontsize=11, va="top")

    wri = FFMpegWriter(fps=fps, metadata={"title": "00_spheroid"})
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for t, mt in keep:
            axs.clear(); axc.clear()
            frame(t, mt, axs, axc)
            wri.grab_frame()
    # the end state, from the same code path so the still cannot drift from the movie
    axs.clear(); axc.clear()
    frame(keep[-1][0], keep[-1][1], axs, axc)
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)

    # --- section.mp4: the wall, zoomed, in the frame every later test will use
    fig2 = plt.figure(figsize=(5.6, 5.6), facecolor="black")
    a2 = fig2.add_subplot(111, facecolor="black")
    fig2.subplots_adjust(0, 0, 1, 1)
    wri2 = FFMpegWriter(fps=fps, metadata={"title": "00_spheroid section"})
    with wri2.saving(fig2, os.path.join(d, "section.mp4"), dpi=100):
        for t, mt in keep:
            a2.clear()
            vp = mt["pos"]
            RD.draw_cross(a2, mt, vp, q, band, cmap, L2, np.eye(3)[2], 0.055, dot_scale=2.0,
                          zoom_half=2.6)
            a2.text(0.02, 0.98, f"00_spheroid   frame {t}   {int(mt['nF'])} cells\n"
                                f"the epithelial wall, no membrane and no matrix",
                    transform=a2.transAxes, color="white", fontsize=11, va="top",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=4.0))
            wri2.grab_frame()
    plt.close(fig2)


def strip(Tis, d, n_col=8):
    """`strip.png`: the same three views as the movies, sampled across the run in one sheet.

    THREE ROWS, NOT `run_ecm`'s FOUR. Its strip carries a cutaway row that removes the octant of MATRIX
    nearest the camera; with no matrix there is nothing to cut away and the row would be a duplicate of
    the one above it. The rows here are the two panels of `movie.mp4` plus the frame `section.mp4` uses,
    so a still and a movie of this folder cannot disagree.
    """
    meshes = Tis["meshes"]
    cmap = ListedColormap(ES.STRESS_COLORS)
    q = np.zeros((0, 3)); band = np.zeros(0, np.uint8)
    L3 = Tis["Lbox"] * 1.60
    L2 = L3 * 1.15
    idx = [meshes[int(round(f * (len(meshes) - 1)))] for f in np.linspace(0, 1, n_col)]
    fig = plt.figure(figsize=(3.4 * n_col, 10.6), facecolor="black")
    for i, (t, mt) in enumerate(idx):
        vp = mt["pos"]
        ax3 = fig.add_subplot(3, n_col, i + 1, projection="3d", computed_zorder=False,
                              facecolor="black")
        RD.draw_3d(ax3, mt, vp, q, band, cmap, RD.CAM_SIDE, 0.72 * L3,
                   div=RD.divided_mask(mt), brk=RD.broken_mask(mt, vp, "00"))
        ax3.text2D(0.04, 0.95, f"frame {t}\n{int(mt['nF'])} cells", transform=ax3.transAxes,
                   color="white", fontsize=12, va="top")
        axc = fig.add_subplot(3, n_col, n_col + i + 1, facecolor="black")
        RD.draw_cross(axc, mt, vp, q, band, cmap, L2, np.eye(3)[2], 0.055)
        axz = fig.add_subplot(3, n_col, 2 * n_col + i + 1, facecolor="black")
        RD.draw_cross(axz, mt, vp, q, band, cmap, L2, np.eye(3)[2], 0.055, dot_scale=2.0,
                      zoom_half=2.6)
    fig.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02, hspace=0.02)
    fig.savefig(os.path.join(d, "strip.png"), dpi=95, facecolor="black")
    plt.close(fig)


def main():
    d = os.path.join(LOG, "00_spheroid")
    os.makedirs(d, exist_ok=True)
    z = np.load(TISSUE)
    Tis = RD.load_tissue(TISSUE, 1.0)
    m = measure(z, Tis)
    yaml.safe_dump(dict(
        what="the epithelium of the nominal run, alone: no basement membrane, no matrix, no adhesion",
        tissue_cache=os.path.relpath(TISSUE, _ROOT), grown_for=NOMINAL,
        pass1=dict(frames=401, buffer_x=4, myosin=1.0, name="cellfix_B_new"),
        operators_exercised=["mesh_seed", "cell_geometry", "cell_grow", "cell_mechanics",
                             "cell_divide", "edge_flip", "topo_record"],
        drawn_frames=len(Tis["meshes"]),
        measures=["n_cells", "r_apical", "r_eq/r_ax", "t1 rate"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    plot_growth(m, os.path.join(d, "growth.png"))
    strip(Tis, d)
    if "--strip-only" not in sys.argv:
        render(Tis, d)
    print(f"[00_spheroid] {m['cells_start']} -> {m['cells_end']} cells, radius x{m['fold_radius']:.2f} "
          f"(area x{m['fold_area']:.1f}), aspect {m['aspect_end']:.3f}, "
          f"T1 {m['t1_per_cell_per_frame']:.4f} per cell per frame -> {d}", flush=True)


if __name__ == "__main__":
    main()
