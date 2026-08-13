#!/usr/bin/env python
"""test_01_junction -- the second operator test: per-junction myosin, and what it does to the tissue.

    python test_01_junction.py [--strip-only]   ->  log/okuda_ECM/01_junction/

WHY THIS ONE IS SECOND. It is the other operator already exercised by the nominal, so like folder 00 it
needs no new physics -- and unlike folder 00 it acts on an EDGE set. `junction_myosin` is a Lateral
operator on \\code{junction} $\\subseteq$ \\code{vertex}$^2$: each cell--cell contact carries its own
myosin, the contact pulls harder where myosin accumulates, and shortening a contact is how two cells
exchange neighbours. So this folder is the first test of the pattern every later operator uses -- state
living on a relation rather than on a body.

WHAT IT MEASURES. Myosin is not observable on its own; what it does is set the T1 rate, and the T1 rate
is what changes tissue shape without changing cell number. The three numbers here are the myosin
distribution over the run, the cumulative T1 count, and the rate per cell per frame -- the last being
the one the nominal is quoted by, and the one an operator claiming to change junctional mechanics has to
move.

THE 2x2 PANEL, as in `153_nominal_material_E100`: the whole tissue in 3D beside its true cross-section,
and underneath, the junction network on its own from two cameras with a magnified patch inset in each.
153's bottom-left is the basement membrane; there is none here, so that panel becomes the second view
of the network rather than an empty box.
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

# AT IMPORT, NOT INSIDE ONE RENDERER. It lived in `_panels`, so any path that skipped that function --
# `--section-only` -- reached the writer with matplotlib still looking for a bare `ffmpeg` on PATH.
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
TISSUE = os.path.join(LOG, "_tissue", "cellfix_B_new_f401_x4_2cedf4bcc6.npz")
NOMINAL = "153_nominal_material_E100"


def measure(z, Tis):
    """Myosin per junction over the run, the T1 count it drives, and the junction lengths."""
    t1 = np.asarray(z["t1_trace"], float) if "t1_trace" in z.files else np.zeros((1, 4))
    frames, myo_mean, myo_p98, n_junc, len_mean = [], [], [], [], []
    for t, mt in Tis["meshes"]:
        if "myo" not in mt:
            continue
        raw = np.asarray(mt["myo"], float).ravel()
        es, et = np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"])
        ef = np.asarray(mt["E_face"]); nF = int(mt["nF"])
        vp = np.asarray(mt["pos"])
        nv = min(int(mt["Nv"]), vp.shape[0])
        # REFUSED, NOT TRUNCATED. Myosin is indexed positionally against the half-edges, so a myosin
        # array of a different length is not a shorter version of the right answer -- it is a different
        # junction's value in every slot past the first divergence. This used to be silently cut to
        # `min(len)` and it is how 56 of 200 snapshots came to be measured against the wrong edges.
        if raw.size != es.size:
            raise RuntimeError(
                f"frame {int(t)}: myosin has {raw.size} entries against {es.size} half-edges. The "
                f"tissue was built without `junction_sync` in its schedule, so the recorded "
                f"myosin belongs to the topology of an earlier point in the frame. Rebuild the cache.")
        # LIVE JUNCTIONS ONLY, by the same `E_face < nF` mask the operator writes through. The whole
        # array includes the reservoir, whose slots hold the filler 1.0; averaging over it reports a
        # number whose value depends on how full the buffer is, which is an allocation and not a tissue.
        ok = (ef < nF) & (es < nv) & (et < nv) & np.isfinite(raw)
        m = raw[ok]
        L = np.linalg.norm(vp[et[ok]] - vp[es[ok]], axis=1)
        frames.append(int(t)); myo_mean.append(float(m.mean()) if m.size else 0.0)
        myo_p98.append(float(np.percentile(m, 98)) if m.size else 0.0)
        n_junc.append(int(ok.sum())); len_mean.append(float(L.mean()) if L.size else 0.0)
    # THE NUMBER THE NOMINAL IS QUOTED BY, AND ITS DENOMINATOR IS FRAMES, NOT ROWS. `t1_trace` rows are
    # (frame, flips since the last row, cumulative, cells) and `edge_flip` is scheduled with
    # `every: 4`, so there is one row per FOUR frames: dividing the flip total by `len(t1)` gives flips
    # per cell per RECORD and calls it per frame. On the nominal that reported 0.0089 where the rate is
    # 0.0022 -- a factor of exactly the T1 stride, in the one number the operator is judged by.
    span = (float(t1[-1, 0] - t1[0, 0]) if t1.size and len(t1) > 1 else 0.0)
    n_frames = max(span, 1.0)
    return dict(
        frames=frames, myo_mean=myo_mean, myo_p98=myo_p98, n_junctions=n_junc, junction_len=len_mean,
        t1_total=int(t1[-1, 2]) if t1.size and t1.shape[1] > 2 else 0,
        t1_frames=n_frames, t1_stride=(n_frames / max(len(t1) - 1, 1)) if t1.size else 0.0,
        t1_per_cell_per_frame=(float(t1[:, 1].sum() / max(t1[:, 3].mean(), 1) / n_frames)
                               if t1.size and t1.shape[1] > 3 else 0.0),
        t1_series=[[float(a) for a in r] for r in t1[::5]] if t1.size else [])


def plot_junction(m, out):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), facecolor="white")
    f = np.asarray(m["frames"], float)
    axes[0].plot(f, m["myo_mean"], color="#2b6cb0", lw=1.6, label="mean")
    axes[0].plot(f, m["myo_p98"], color="#4aa3ff", lw=1.2, ls="--", label="p98")
    axes[0].set_ylabel("myosin per junction"); axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("what the operator carries", fontsize=9)
    if m["t1_series"]:
        t1 = np.asarray(m["t1_series"], float)
        axes[1].plot(t1[:, 0], t1[:, 2], color="#e08a2e", lw=1.6)
    axes[1].set_ylabel("cumulative T1 exchanges")
    axes[1].set_title(f"{m['t1_total']} total, {m['t1_per_cell_per_frame']:.4f} per cell per frame",
                      fontsize=9)
    axes[2].plot(f, m["junction_len"], color="#1f8a5c", lw=1.6)
    axes[2].set_ylabel("mean junction length (tissue units)")
    axes[2].set_title("what a T1 shortens", fontsize=9)
    for a in axes:
        a.set_xlabel("frame"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=140, facecolor="white"); plt.close(fig)


def _myo_scale(Tis):
    """One fixed colour scale for the whole run: normalising per frame would hide a sheet-wide drift,
    which is exactly what an activity parameter produces."""
    # LIVE JUNCTIONS ONLY, as in `measure`: the reservoir's filler 1.0s would drag the 98th percentile
    # toward 1 as the buffer empties, so the colour scale would drift for a reason that is allocation.
    allm = []
    for _, mt in Tis["meshes"]:
        if "myo" not in mt:
            continue
        v_ = np.asarray(mt["myo"], float).ravel()
        ef = np.asarray(mt["E_face"]); nF = int(mt["nF"])
        if v_.size == ef.size:
            allm.append(v_[ef < nF])
    if not allm:
        return None
    v = np.concatenate(allm)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 98)) if v.size else None


def _panels(Tis, keep, d, myo_sc, movie=True, fps=15):
    """The 2x2 of `153_nominal_material_E100`, with its two magnified insets."""
    cmap = ListedColormap(ES.STRESS_COLORS)
    q = np.zeros((0, 3)); band = np.zeros(0, np.uint8)
    L3 = Tis["Lbox"] * 1.60
    L2 = L3 * 1.15
    Lt = 0.72 * L3
    fig = plt.figure(figsize=(11.0, 11.0), facecolor="black")
    axs = fig.add_subplot(2, 2, 1, projection="3d", computed_zorder=False, facecolor="black")
    axc = fig.add_subplot(2, 2, 2, facecolor="black")
    # THREE PANELS, NOT FOUR. The second camera on the same network showed the same thing twice: a
    # sphere of junctions is the same object from any angle, and the poles the top view was meant to
    # expose are not where anything happens here. The slot is left empty rather than filled.
    axz = fig.add_subplot(2, 2, 3, projection="3d", computed_zorder=False, facecolor="black")
    inz = fig.add_axes([0.335, 0.035, 0.155, 0.155], facecolor="black", zorder=20)
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02, hspace=0.02)

    def frame(t, mt):
        vp = mt["pos"]
        for a in (axs, axc, axz, inz):
            a.clear()
        RD.draw_3d(axs, mt, vp, q, band, cmap, RD.CAM_SIDE, Lt,
                   div=RD.divided_mask(mt), brk=RD.broken_mask(mt, vp, "01"))
        RD.draw_cross(axc, mt, vp, q, band, cmap, L2, np.eye(3)[2], 0.055, dot_scale=0.85)
        # TWO CAMERAS ON THE SAME NETWORK. 153 spends the bottom-left panel on the basement membrane;
        # there is none in this folder, and drawing the network twice from one angle would be a
        # duplicate, so the second panel is the top view -- the one that shows the poles the side
        # camera puts on the silhouette.
        RD.draw_junctions_3d(axz, mt, vp, RD.CAM_SIDE, Lt, myo_hi=myo_sc)
        RD.draw_zoom(inz, mt, vp, mem_q=None, mem_s=None, name="01", frac=0.16, lw=2.4,
                     r_ref=Lt, myo_hi=myo_sc)
        for a in (inz,):
            for sp in a.spines.values():
                sp.set_color("#666"); sp.set_visible(True)
            a.set_xticks([]); a.set_yticks([])
        axs.text2D(0.02, 0.96, f"01_junction   frame {t}   {int(mt['nF'])} cells",
                   transform=axs.transAxes, color="white", fontsize=11, va="top")
        axz.text2D(0.03, 0.95, "junction network, side\ncoloured by myosin", transform=axz.transAxes,
                   color="white", fontsize=10, va="top")
        # THE COUNT, ON THE PANEL IT COUNTS. Half-edges are counted in pairs and only those whose endpoints are
        # live vertices count: `E_srce`/`E_trgt` index the full buffer, which is four times the live
        # mesh, so an uncorrected length would report the reservoir rather than the tissue.
        _es, _et = np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"])
        _nv = min(int(mt["Nv"]), vp.shape[0])
        _nj = int(((_es < _nv) & (_et < _nv)).sum())
        axz.text2D(0.03, 0.04, f"{_nj} junctions", transform=axz.transAxes, color="white",
                   fontsize=12, ha="left", va="bottom")

    if movie:
        wri = FFMpegWriter(fps=fps, metadata={"title": "01_junction"})
        with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
            for t, mt in keep:
                frame(t, mt)
                wri.grab_frame()
    frame(*keep[-1])
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)


def _section(Tis, keep, d, myo_sc, fps=15):
    """`section.mp4`: the wall in section beside the junction patch that lives in it."""
    cmap = ListedColormap(ES.STRESS_COLORS)
    q = np.zeros((0, 3)); band = np.zeros(0, np.uint8)
    L2 = Tis["Lbox"] * 1.60 * 1.15
    Lt = 0.72 * Tis["Lbox"] * 1.60
    fig = plt.figure(figsize=(11.0, 5.6), facecolor="black")
    a1 = fig.add_subplot(1, 2, 1, facecolor="black")
    # A PLAIN 2D AXIS. `draw_zoom` adds LineCollections, and a 3D axis projects its children by calling
    # `do_3d_projection` on each -- which a 2D collection does not have. The panel rendered anyway and
    # then died on save, which is why `section.mp4` exists and the script exited non-zero.
    a2 = fig.add_subplot(1, 2, 2, facecolor="black")
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02)
    wri = FFMpegWriter(fps=fps, metadata={"title": "01_junction section"})
    with wri.saving(fig, os.path.join(d, "section.mp4"), dpi=100):
        for t, mt in keep:
            a1.clear(); a2.clear()
            vp = mt["pos"]
            RD.draw_cross(a1, mt, vp, q, band, cmap, L2, np.eye(3)[2], 0.055, dot_scale=2.0,
                          zoom_half=2.6)
            RD.draw_zoom(a2, mt, vp, mem_q=None, mem_s=None, name="01", frac=0.16, lw=2.4,
                         r_ref=Lt, myo_hi=myo_sc)
            a1.text(0.02, 0.98, f"01_junction   frame {t}   {int(mt['nF'])} cells\n"
                                f"the epithelial wall in section",
                    transform=a1.transAxes, color="white", fontsize=11, va="top",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=4.0))
            a2.text(0.03, 0.96, "the junctions in that patch,\ncoloured by myosin",
                    transform=a2.transAxes, color="white", fontsize=10, va="top",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3.0))
            wri.grab_frame()
    plt.close(fig)


def strip(Tis, d, myo_sc, n_col=8):
    """ALWAYS. Three rows: the tissue, its section, and the junction network coloured by myosin."""
    cmap = ListedColormap(ES.STRESS_COLORS)
    q = np.zeros((0, 3)); band = np.zeros(0, np.uint8)
    meshes = Tis["meshes"]
    L3 = Tis["Lbox"] * 1.60; L2 = L3 * 1.15; Lt = 0.72 * L3
    idx = [meshes[int(round(f * (len(meshes) - 1)))] for f in np.linspace(0, 1, n_col)]
    fig = plt.figure(figsize=(3.4 * n_col, 10.6), facecolor="black")
    for i, (t, mt) in enumerate(idx):
        vp = mt["pos"]
        a1 = fig.add_subplot(3, n_col, i + 1, projection="3d", computed_zorder=False,
                             facecolor="black")
        RD.draw_3d(a1, mt, vp, q, band, cmap, RD.CAM_SIDE, Lt,
                   div=RD.divided_mask(mt), brk=RD.broken_mask(mt, vp, "01"))
        a1.text2D(0.04, 0.95, f"frame {t}\n{int(mt['nF'])} cells", transform=a1.transAxes,
                  color="white", fontsize=12, va="top")
        a2 = fig.add_subplot(3, n_col, n_col + i + 1, facecolor="black")
        RD.draw_cross(a2, mt, vp, q, band, cmap, L2, np.eye(3)[2], 0.055)
        a3 = fig.add_subplot(3, n_col, 2 * n_col + i + 1, projection="3d", computed_zorder=False,
                             facecolor="black")
        RD.draw_junctions_3d(a3, mt, vp, RD.CAM_SIDE, Lt, myo_hi=myo_sc)
    fig.subplots_adjust(0.004, 0.004, 0.996, 0.996, wspace=0.02, hspace=0.02)
    fig.savefig(os.path.join(d, "strip.png"), dpi=95, facecolor="black")
    plt.close(fig)



# ---------------------------------------------------------------------------------------------------
# THE MODEL ITSELF, CHECKED AGAINST ITS OWN OUTPUT
# ---------------------------------------------------------------------------------------------------
def _edge_table(Tis):
    """Per snapshot: {(v_lo, v_hi): (myosin, length)}, plus the reference length the operator uses.

    THE KEY IS THE VERTEX PAIR, not the row index. Half-edge rows are rebuilt every snapshot as cells
    divide and T1s fire, so row 400 is a different junction at frame 0 and frame 200; the unordered pair
    of vertex indices is the identity that survives, and it is what makes "the same junction over time"
    a meaningful phrase here. Two half-edges share one pair, so their myosin is averaged.
    """
    out, ref, frames = [], [], []
    for t, mt in Tis["meshes"]:
        if "myo" not in mt:
            continue
        vp = np.asarray(mt["pos"]); nv = min(int(mt["Nv"]), vp.shape[0])
        es, et = np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"])
        ef = np.asarray(mt["E_face"]); nF = int(mt["nF"])
        myo = np.asarray(mt["myo"], float).ravel()
        # SAME REFUSAL AS `measure`. The `min(len(...))` truncation that used to be here is what made the
        # recorded myosin trace jump 0.14 per step across a frame where a division had lengthened the
        # half-edge arrays -- fifteen times what the tau=20 integrator can move, and the entire source of
        # the high-frequency buzz on `junction_model.png`.
        if myo.size != es.size:
            raise RuntimeError(f"frame {int(t)}: myosin {myo.size} vs {es.size} half-edges; the cache "
                               f"predates `junction_sync`. Rebuild it.")
        ok = (ef < nF) & (es < nv) & (et < nv) & np.isfinite(myo)
        L = np.linalg.norm(vp[et[ok]] - vp[es[ok]], axis=1)
        keys = [(int(min(a, b)), int(max(a, b))) for a, b in zip(es[ok], et[ok])]
        d = {}
        for k, mv, lv in zip(keys, myo[ok], L):
            if k in d:
                d[k] = (0.5 * (d[k][0] + mv), lv)
            else:
                d[k] = (float(mv), float(lv))
        out.append(d); ref.append(float(L.mean()) if L.size else 1.0); frames.append(int(t))
    return frames, ref, out


def _pick(tables, n_survive=3, n_vanish=2, n_born=2):
    """A few junctions of each kind: ones that last, ones a division or T1 consumes, ones born from a
    division. The third class is the interesting one -- a newborn junction starts at `myo_new`, so its
    trace is the operator's initial condition made visible."""
    early, late = tables[3], tables[-4]
    lifetime = {}
    for k in early:
        last = 0
        for i, d in enumerate(tables):
            if k in d:
                last = i
        lifetime[k] = last
    surv = [k for k, v in lifetime.items() if v >= len(tables) - 4]
    van = [k for k, v in lifetime.items() if 20 < v < len(tables) - 20]
    born = [k for k in late if k not in early and all(k not in d for d in tables[:20])]
    rng = np.random.default_rng(0)
    pick = lambda lst, n: [lst[i] for i in rng.choice(len(lst), min(n, len(lst)), replace=False)] \
        if lst else []
    return pick(surv, n_survive), pick(van, n_vanish), pick(born, n_born)


def plot_equation(Tis, d, activity=1.0, tau=20.0, myo_new=1.0, lam=None, k_perim=None, gam=None):
    """`junction_model.png`: the leaky integrator, on one junction at a time.

    THE FIRST VERSION PUT SEVEN JUNCTIONS ON TWO AXES AND WAS UNREADABLE. What the operator does is a
    statement about ONE junction -- its length sets a target, its myosin chases that target with a lag
    tau -- and that is invisible in a bundle of overlapping traces. So each column is one junction:
    length on top with the tissue's mean length beside it, and underneath the two myosin curves the
    equations relate, the target and the value chasing it. The recorded myosin and the ODE integrated
    from that junction's own length are both drawn; where they separate, the operator is not the model.
    """
    frames, ref, tables = _edge_table(Tis)
    surv, van, born = _pick(tables, n_survive=8, n_vanish=6, n_born=6)

    def series(k):
        i = [j for j, dd in enumerate(tables) if k in dd]
        if not i:
            return None
        t = np.asarray([frames[j] for j in i], float)
        m = np.asarray([tables[j][k][0] for j in i])
        L = np.asarray([tables[j][k][1] for j in i])
        rf = np.asarray([ref[j] for j in i])
        ss = activity * L / np.maximum(rf, 1e-9)
        pred = np.empty_like(m); pred[0] = m[0]
        for q in range(1, len(m)):
            dt_ = max(t[q] - t[q - 1], 1.0)
            pred[q] = pred[q - 1] + (ss[q - 1] - pred[q - 1]) * dt_ / tau
        return dict(t=t, m=m, L=L, ref=rf, ss=ss, pred=pred)

    # THE CLEANEST SURVIVOR, chosen by how little it jitters. Every junction obeys the same equation;
    # picking the quietest one is a choice about legibility, and it is stated here rather than hidden.
    cand = [(k, series(k)) for k in surv]
    cand = [(k, v) for k, v in cand if v is not None and len(v["t"]) > 60]
    cand.sort(key=lambda kv: float(np.mean(np.abs(np.diff(kv[1]["m"])))))
    k_long, s_long = cand[0]

    # A DIVISION, SEEN FROM BOTH SIDES: a junction the event consumes, and the newborn nearest in time.
    vs = [(k, series(k)) for k in van]
    vs = [(k, v) for k, v in vs if v is not None and len(v["t"]) > 25]
    k_gone, s_gone = (vs[0] if vs else (None, None))
    # THE NEWBORN HAS TO BE THE SAME EVENT, or the caption is a claim the figure does not support.
    # `_pick` samples a handful of candidates, and the nearest of those was 110 frames after the
    # junction it was supposed to have replaced -- a different division entirely. So the newborn is
    # searched over EVERY key here, and if the closest is still far the label says so instead.
    t_gone = s_gone["t"][-1] if s_gone is not None else None
    k_new, s_new, dt_new = None, None, None
    if t_gone is not None:
        early_keys = set().union(*[set(dd) for dd in tables[:6]])
        cands = []
        for j, dd in enumerate(tables):
            if abs(frames[j] - t_gone) > 40:
                continue
            for k in dd:
                if k in early_keys or all(k not in tables[q] for q in range(max(j - 1, 0))):
                    pass
        # a newborn is a key whose FIRST appearance is near t_gone
        first_seen = {}
        for j, dd in enumerate(tables):
            for k in dd:
                first_seen.setdefault(k, frames[j])
        cands = [(k, f0) for k, f0 in first_seen.items() if f0 > frames[3]]
        if cands:
            k_new, f0 = min(cands, key=lambda kv: abs(kv[1] - t_gone))
            s_new = series(k_new); dt_new = float(f0 - t_gone)
    t_div = s_gone["t"][-1] if s_gone is not None else None

    fig = plt.figure(figsize=(13.4, 5.6), facecolor="white")
    axE = fig.add_axes([0.005, 0.06, 0.235, 0.88]); axE.axis("off")
    axL1 = fig.add_axes([0.315, 0.585, 0.30, 0.345])
    axM1 = fig.add_axes([0.315, 0.115, 0.30, 0.395])
    axL2 = fig.add_axes([0.685, 0.585, 0.30, 0.345])
    axM2 = fig.add_axes([0.685, 0.115, 0.30, 0.395])

    axE.text(0.0, 1.00, "junction_myosin", fontsize=12, fontweight="bold", va="top",
             family="monospace")
    axE.text(0.0, 0.925, "a Lateral operator on the junction edge set", fontsize=8.5, va="top",
             color="#444")
    axE.text(0.0, 0.79, r"$m^{\mathrm{ss}}_e \;=\; a\,\dfrac{\ell_e}{\bar{\ell}}$", fontsize=15,
             va="top")
    axE.text(0.0, 0.60, r"$\dfrac{dm_e}{dt} \;=\; \dfrac{m^{\mathrm{ss}}_e - m_e}{\tau}$",
             fontsize=15, va="top")
    axE.text(0.0, 0.40,
             "a stretched junction raises its target;\n"
             "myosin follows with a lag $\\tau$ --- a leaky\n"
             "integrator, so $m_e$ smooths $m^{\\mathrm{ss}}_e$",
             fontsize=8.5, va="top", color="#333")
    axE.text(0.0, 0.235,
             f"$a$ = {activity:g}   $\\tau$ = {tau:g} frames   $m_{{new}}$ = {myo_new:g}\n"
             "$\\bar{\\ell}$ = the current mean live edge length",
             fontsize=8.5, va="top", color="#333")
    axE.text(0.0, 0.045, "Rauzi et al. 2008 Nat Cell Biol 10:1401\n"
                         "Fernandez-Gonzalez et al. 2009 Dev Cell 17:736",
             fontsize=7.5, va="bottom", color="#666")

    def draw(axL, axM, s_, title, extra=None, t_mark=None):
        axL.plot(s_["t"], s_["L"], color="#1f8a5c", lw=1.4, label=r"$\ell_e$, this junction")
        axL.plot(s_["t"], s_["ref"], color="#999", lw=1.1, ls="--", label=r"$\bar{\ell}$, tissue mean")
        axM.plot(s_["t"], s_["ss"], color="#e08a2e", lw=1.1, label=r"$m^{\mathrm{ss}}_e$ (target)")
        axM.plot(s_["t"], s_["m"], color="#2b6cb0", lw=1.6, label=r"$m_e$ recorded")
        axM.plot(s_["t"], s_["pred"], color="#4aa3ff", lw=1.1, ls=":", label="the ODE, integrated")
        if extra is not None:
            axL.plot(extra["t"], extra["L"], color="#7bbf6a", lw=1.4)
            axM.plot(extra["t"], extra["ss"], color="#f0b46a", lw=1.0)
            axM.plot(extra["t"], extra["m"], color="#7fc4ff", lw=1.4)
            axM.plot(extra["t"][0], extra["m"][0], "o", color="#7fc4ff", ms=6, mfc="white")
        if t_mark is not None:
            for a in (axL, axM):
                a.axvline(t_mark, color="#c33", lw=1.0, ls="-.")
        axL.set_title(title, fontsize=9)
        axL.set_ylabel("length (tissue units)"); axM.set_ylabel("myosin")
        axM.set_xlabel("frame")
        axL.tick_params(labelbottom=False)
        for a in (axL, axM):
            a.spines[["top", "right"]].set_visible(False)
        axL.legend(fontsize=7, frameon=False, loc="best")
        axM.legend(fontsize=7, frameon=False, loc="best")

    draw(axL1, axM1, s_long, "one junction that lasts the run")
    if s_gone is not None:
        same = dt_new is not None and abs(dt_new) <= 6
        lab = ("across one division: the junction it consumes (dark)\n"
               + ("and the junction born in its place (light)" if same else
                  f"and the nearest newborn, {abs(dt_new):.0f} frames later (light)"))
        draw(axL2, axM2, s_gone, lab, extra=s_new, t_mark=t_div)
    fig.savefig(os.path.join(d, "junction_model.png"), dpi=150, facecolor="white")
    plt.close(fig)
    return dict(long_junction=[int(x) for x in k_long],
                consumed=[int(x) for x in k_gone] if k_gone else None,
                born=[int(x) for x in k_new] if k_new else None,
                division_frame=(float(t_div) if t_div is not None else None),
                snapshots=len(tables))


def main():
    d = os.path.join(LOG, "01_junction")
    os.makedirs(d, exist_ok=True)
    z = np.load(TISSUE)
    Tis = RD.load_tissue(TISSUE, 1.0)
    myo_sc = _myo_scale(Tis)
    m = measure(z, Tis)
    yaml.safe_dump(dict(
        what="per-junction myosin on the nominal's epithelium: an operator on an EDGE set, alone",
        tissue_cache=os.path.relpath(TISSUE, _ROOT), grown_for=NOMINAL,
        pass1=dict(frames=401, buffer_x=4, myosin=1.0, myo_tau=20.0, myo_beta=1.0,
                   myo_keyed_on="length", name="cellfix_B_new"),
        operators_exercised=["junction_myosin", "cell_mechanics", "edge_flip"],
        plexus2=dict(kind="Lateral", acts_on="junction (edge set on vertex)",
                     state="myosin per half-edge"),
        myosin_colour_full_scale=myo_sc,
        measures=["myosin mean and p98", "cumulative T1", "T1 per cell per frame",
                  "mean junction length"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    plot_junction(m, os.path.join(d, "myosin.png"))
    tr = plot_equation(Tis, d)
    m["tracked_junctions"] = tr
    keep = [Tis["meshes"][int(round(f * (len(Tis["meshes"]) - 1)))]
            for f in np.linspace(0, 1, min(150, len(Tis["meshes"])))]
    only = {a for a in sys.argv if a.startswith("--")}
    if "--eq-only" in only:
        json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=1)
        print(f"[01_junction] junction_model.png: {tr}", flush=True)
        return
    if not only or "--strip-only" in only:
        strip(Tis, d, myo_sc)
    if not only or "--panels-only" in only:
        _panels(Tis, keep, d, myo_sc)
    if not only or "--section-only" in only:
        _section(Tis, keep, d, myo_sc)
    print(f"[01_junction] myosin p98 {myo_sc:.3g}, {m['t1_total']} T1 exchanges, "
          f"{m['t1_per_cell_per_frame']:.4f} per cell per frame, mean junction length "
          f"{m['junction_len'][0]:.3f} -> {m['junction_len'][-1]:.3f} -> {d}", flush=True)


if __name__ == "__main__":
    main()
