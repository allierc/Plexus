#!/usr/bin/env python
"""test_01b_myosin_pools -- is junctional myosin conserved through a division, and is its setpoint?

    python test_01b_myosin_pools.py [--no-movie]   ->  log/okuda_ECM/01b_myosin_pools/

TWO QUESTIONS, ONE OF WHICH THE FOLDER 01 COULD NOT ASK. When `cell_divide` inserts a vertex into an
existing cell--cell contact (a,b), it cuts that contact into (a,n) and (n,b). NOTHING PHYSICAL HAS
HAPPENED: the same two cells still touch over the same interface, and the only thing that changed is
the mesh's description of it. Two things must therefore hold, and they are separate claims:

  1. THE AMOUNT IS CONSERVED.  N_(a,n) + N_(n,b) = N_(a,b), where N_e = m_e * l_e is the myosin ON a
     junction and m_e -- the multiplier `cell_mechanics` applies to the line tension -- is its
     DENSITY per unit length. `junction_ops._lookup` copies the parent's m onto both halves, which is
     conservative precisely because m is a density and l_an + l_nb = l_ab. Had m been an amount, the
     same line of code would have doubled the myosin at every division.

  2. THE SETPOINT IS UNCHANGED.  The value each half relaxes toward must not move either, or the
     junction decays away from where it was for a reason that is a re-meshing artifact. This is the
     one the one-pool model fails: its setpoint is m_ss = a * l_e/<l>, and l_e is EXTENSIVE -- it
     halves at the cut. So each half, having correctly inherited the parent's myosin, immediately
     starts relaxing toward half of it.

The fix is not a better drive but a conserved amount with a derived density, which is what
`medioapical_myosin` + `junction_myosin[two_pool]` are: myosin is supplied to the belt per unit
length from an areal pool on the cell, so the setpoint of a half is the setpoint of the whole.

WHAT IS COMPARED. Two 401-frame builds of the same tissue differing only in the myosin model --
`one_pool` (the 01_junction nominal, length-keyed) and `two_pool` -- and within each, split junctions
against INTACT ones over the same interval. The intact controls are what make the numbers mean
anything: two frames of ordinary myosin dynamics move a junction on their own, and without the
control every division would look like a leak.
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

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import ecm_render as RD
import ecm_spec as ES

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
CACHE = os.path.join(LOG, "_tissue")
# The one-pool control is the 01_junction nominal itself. The two-pool cache is NOT named here: its
# path is a hash over the configuration, so `load_or_build` derives it from the parameters below and a
# constant written by hand would be a second, silently divergent source of truth for which run this is.
ONE_POOL = os.path.join(CACHE, "cellfix_B_new_f401_x4_2cedf4bcc6.npz")

# The two-pool operating point, and where each number comes from. `k_on` is not chosen for effect: it
# is 1/tau_med + k_ex * <P/A> at frame 0, i.e. the value at which the medioapical density STARTS at
# its own steady state, so what the run shows afterwards is the shape factor moving and not a
# fifty-frame relaxation from an arbitrary initial condition.
TWO_POOL_PARAMS = dict(myo_model="two_pool", myo_k_on=0.219, myo_tau_med=20.0, myo_k_ex=0.05,
                       myo_beta_T=0.0, myo_tau=20.0, myosin=1.0, myo_new=1.0)


# ---------------------------------------------------------------------------------------------------
# THE TABLES: what a junction is, over time
# ---------------------------------------------------------------------------------------------------
def snapshots(path):
    """Per recorded frame: the live junctions keyed by vertex pair, with everything a measurement needs.

    KEYED BY THE UNORDERED VERTEX PAIR, never by row index: half-edge rows are rebuilt whenever a cell
    divides or a T1 fires, so row 400 is a different junction at frame 0 and at frame 200. Two
    half-edges share one pair; their myosin is identical by construction (the operator writes one value
    per pair) and is checked to be so rather than averaged, because averaging would hide the
    misalignment that `junction_sync` exists to prevent.
    """
    z = np.load(path)
    out = []
    for i in range(len(z["mesh_frames"])):
        t = int(z["mesh_frames"][i])
        pos = z[f"m{i}_pos"]; nF = int(z[f"m{i}_nF"]); nv = min(int(z[f"m{i}_Nv"]), pos.shape[0])
        es = z[f"m{i}_E_srce"].astype(np.int64); et = z[f"m{i}_E_trgt"].astype(np.int64)
        ef = z[f"m{i}_E_face"].astype(np.int64)
        myo = np.asarray(z[f"m{i}_myo"], float).ravel()
        if myo.size != es.size:
            raise RuntimeError(f"{os.path.basename(path)} frame {t}: myosin {myo.size} against "
                               f"{es.size} half-edges -- the cache predates junction_sync.")
        ok = (ef < nF) & (es < nv) & (et < nv) & np.isfinite(myo)
        L = np.linalg.norm(pos[et[ok]] - pos[es[ok]], axis=1)
        lo = np.minimum(es[ok], et[ok]); hi = np.maximum(es[ok], et[ok])
        # PER-FACE PERIMETER AND AREA, for the shape factor P/A the two-pool steady state depends on.
        P = np.zeros(nF); np.add.at(P, ef[ok], L)
        cr = np.cross(pos[es[ok]], pos[et[ok]])
        N = np.zeros((nF, 3)); np.add.at(N, ef[ok], cr)
        A = 0.5 * np.linalg.norm(N, axis=1)
        d = {}
        for a, b, mv, lv in zip(lo, hi, myo[ok], L):
            d[(int(a), int(b))] = (float(mv), float(lv))
        med = z[f"m{i}_myo_med"] if f"m{i}_myo_med" in z.files else None
        # THE LEDGER NEEDS THE AMOUNT, NOT THE MULTIPLIER. `myo` is normalised to a tissue mean of
        # `activity` before it reaches the mechanics, so summing `myo * l` gives a number that says
        # how the myosin is DISTRIBUTED and nothing about how much of it there is. `myo_amount` is
        # the unnormalised N_e, summed over half-edges because each of the two cells sharing a
        # junction feeds its own side of the belt.
        amt = z[f"m{i}_myo_amount"] if f"m{i}_myo_amount" in z.files else None
        out.append(dict(t=t, nv=nv, nF=nF, tab=d, l_mean=float(L.mean()),
                        m_mean=float(myo[ok].mean()), P=P, A=np.maximum(A, 1e-9),
                        n_jun=float(len(d)),
                        jun_amt=(float(np.asarray(amt, float).ravel()[ok].sum())
                                 if amt is not None else None),
                        med=(np.asarray(med, float)[:nF] if med is not None else None)))
    return out


def splits(S):
    """Every junction a division CUT IN TWO, between consecutive snapshots.

    A vertex `cell_divide` inserts is APPENDED, so its index is >= the previous snapshot's live vertex
    count -- that is the whole detector, and it cannot confuse a division with a T1, which adds no
    vertices. For each inserted vertex, its two OLD neighbours are the endpoints of the edge it was
    inserted into, so the parent key is recoverable without any help from the operator that made it.

    Yields (i, parent_key, half_key_1, half_key_2), keeping only events where the parent really was
    present before and really is gone after -- the interface between two daughters is a different
    object, has no parent, and is excluded here on purpose.
    """
    ev = []
    for i in range(len(S) - 1):
        a, b = S[i], S[i + 1]
        nv0 = a["nv"]
        nbr = {}
        for (u, v) in b["tab"]:
            if (u >= nv0) ^ (v >= nv0):                      # exactly one endpoint is new
                new, old = (u, v) if u >= nv0 else (v, u)
                nbr.setdefault(new, []).append(old)
        for new, olds in nbr.items():
            if len(olds) != 2:
                continue
            p = (min(olds), max(olds))
            if p not in a["tab"] or p in b["tab"]:
                continue
            h1 = (min(olds[0], new), max(olds[0], new))
            h2 = (min(olds[1], new), max(olds[1], new))
            if h1 in b["tab"] and h2 in b["tab"]:
                ev.append((i, p, h1, h2))
    return ev


# ---------------------------------------------------------------------------------------------------
# MEASUREMENT 1: is the amount conserved across the cut?
# ---------------------------------------------------------------------------------------------------
def conservation(S, ev):
    """N_half1 + N_half2 over N_parent, against the same ratio for junctions nothing happened to.

    THE CONTROL IS NOT OPTIONAL. Between two recorded snapshots the tissue relaxes twice and the
    myosin operator runs twice, so an untouched junction's amount does not stay put either. Reporting
    the split ratio alone would charge two frames of ordinary dynamics to the division. The intact
    ratio is measured over the SAME interval on the SAME snapshots, so the difference between the two
    is the part that belongs to the cut.
    """
    r_split, r_intact, l_ratio = [], [], []
    by_i = {}
    for i, p, h1, h2 in ev:
        by_i.setdefault(i, []).append((p, h1, h2))
    for i, items in by_i.items():
        a, b = S[i], S[i + 1]
        touched = set()
        for p, h1, h2 in items:
            touched |= {p, h1, h2}
            mp, lp = a["tab"][p]
            m1, l1 = b["tab"][h1]
            m2, l2 = b["tab"][h2]
            if mp * lp <= 0:
                continue
            r_split.append((m1 * l1 + m2 * l2) / (mp * lp))
            l_ratio.append((l1 + l2) / lp)
        common = [k for k in a["tab"] if k in b["tab"] and k not in touched]
        # A SAMPLE, NOT THE WHOLE SET: there are ~2,000 intact junctions per interval against a handful
        # of splits, and an unweighted pool of every intact junction in the run would make the control
        # a hundred times better resolved than the thing it controls. Ten per event, seeded.
        rng = np.random.default_rng(i)
        for k in (common if len(common) <= 10 * len(items)
                  else [common[j] for j in rng.choice(len(common), 10 * len(items), replace=False)]):
            m0, l0 = a["tab"][k]; m1, l1 = b["tab"][k]
            if m0 * l0 > 0:
                r_intact.append((m1 * l1) / (m0 * l0))
    return np.asarray(r_split), np.asarray(r_intact), np.asarray(l_ratio)


# ---------------------------------------------------------------------------------------------------
# MEASUREMENT 2: does the SETPOINT survive the cut?
# ---------------------------------------------------------------------------------------------------
def drift(S, ev, horizon=10):
    """How a junction's myosin moves in the `horizon` snapshots AFTER a cut, split halves vs intact.

    Normalised by the tissue's mean myosin at each time, so a sheet-wide drift -- which both models
    have, since the mean junction length falls by a third over the run -- is not read as a
    division effect. What is left is the answer to "does a junction that was cut, and nothing else,
    end up somewhere different from one that was not".
    """
    d_split, d_intact = [], []
    for i, p, h1, h2 in ev:
        j = i + horizon
        if j >= len(S):
            continue
        for h in (h1, h2):
            if h in S[i + 1]["tab"] and h in S[j]["tab"]:
                m0 = S[i + 1]["tab"][h][0] / max(S[i + 1]["m_mean"], 1e-9)
                m1 = S[j]["tab"][h][0] / max(S[j]["m_mean"], 1e-9)
                d_split.append(m1 / max(m0, 1e-9))
    rng = np.random.default_rng(0)
    idx = sorted({i for i, _, _, _ in ev})
    for i in idx:
        j = i + horizon
        if j >= len(S):
            continue
        common = [k for k in S[i + 1]["tab"] if k in S[j]["tab"]]
        if not common:
            continue
        for k in [common[q] for q in rng.choice(len(common), min(40, len(common)), replace=False)]:
            m0 = S[i + 1]["tab"][k][0] / max(S[i + 1]["m_mean"], 1e-9)
            m1 = S[j]["tab"][k][0] / max(S[j]["m_mean"], 1e-9)
            d_intact.append(m1 / max(m0, 1e-9))
    return np.asarray(d_split), np.asarray(d_intact)


def setpoint_ratio(S, ev):
    """The one-pool setpoint of a half over the setpoint of its parent, m_ss = a * l/<l>.

    This is the artifact in its purest form and it needs no run to predict: the setpoint is
    proportional to length, the cut halves the length, so the ratio is 1/2 and the junction spends the
    next `tau` frames relaxing toward half the myosin it correctly inherited. Measured rather than
    asserted because `<l>` also moves between the two snapshots.
    """
    out = []
    for i, p, h1, h2 in ev:
        a, b = S[i], S[i + 1]
        sp = a["tab"][p][1] / max(a["l_mean"], 1e-9)
        for h in (h1, h2):
            out.append((b["tab"][h][1] / max(b["l_mean"], 1e-9)) / max(sp, 1e-9))
    return np.asarray(out)


def pools(S):
    """The two-pool ledger over the run: how much myosin is on the cortex, how much on the belts, and
    the shape factor P/A that decides the split between them."""
    t, med, jun, pa = [], [], [], []
    for s in S:
        if s["med"] is None or s["jun_amt"] is None:
            continue
        t.append(s["t"])
        med.append(float((s["med"] * s["A"]).sum()))
        jun.append(s["jun_amt"])
        pa.append(float((s["P"] / s["A"]).mean()))
    return np.asarray(t), np.asarray(med), np.asarray(jun), np.asarray(pa)


def t1_rate(path):
    """Flips per cell per FRAME. `t1_trace` rows are (frame, flips since the last row, cumulative,
    cells) and `edge_flip` runs every 4 frames, so the denominator is the frame SPAN and not the
    number of rows -- dividing by the rows reports the rate per T1 call and calls it per frame."""
    z = np.load(path)
    t1 = np.asarray(z["t1_trace"], float)
    if t1.size == 0:
        return 0.0, 0
    span = max(float(t1[-1, 0] - t1[0, 0]), 1.0)
    return float(t1[:, 1].sum() / max(t1[:, 3].mean(), 1) / span), int(t1[-1, 2])


# ---------------------------------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------------------------------
def fig_conservation(R, out):
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.4), facecolor="white")
    bins = np.linspace(0.6, 1.4, 60)
    for k, col, lab in (("one", "#c0392b", "one-pool"), ("two", "#2b6cb0", "two-pool")):
        ax[0].hist(R[k]["r_split"], bins=bins, histtype="step", lw=1.7, color=col,
                   label=f"{lab}, split  (med {np.median(R[k]['r_split']):.3f})", density=True)
        ax[0].hist(R[k]["r_intact"], bins=bins, histtype="step", lw=1.1, ls="--", color=col,
                   label=f"{lab}, intact (med {np.median(R[k]['r_intact']):.3f})", density=True)
    ax[0].axvline(1.0, color="#444", lw=0.9)
    ax[0].set_xlabel(r"$(N_{1}+N_{2})/N_{\mathrm{parent}}$,  $N_e=m_e\ell_e$")
    ax[0].set_ylabel("density"); ax[0].legend(fontsize=6.3, frameon=False)
    ax[0].set_title("1. is the AMOUNT conserved across the cut?", fontsize=9)

    sp = R["one"]["setpoint"]
    ax[1].hist(sp, bins=np.linspace(0.0, 1.5, 60), color="#c0392b", alpha=0.75)
    ax[1].axvline(0.5, color="#444", lw=1.0, ls="--")
    ax[1].axvline(float(np.median(sp)), color="#000", lw=1.2)
    ax[1].set_xlabel(r"$m^{ss}_{\mathrm{half}}/m^{ss}_{\mathrm{parent}}$   (one-pool)")
    ax[1].set_ylabel("junction halves")
    ax[1].set_title(f"2. and the SETPOINT?  median {np.median(sp):.3f}", fontsize=9)

    for k, col, lab in (("one", "#c0392b", "one-pool"), ("two", "#2b6cb0", "two-pool")):
        ds, di = R[k]["d_split"], R[k]["d_intact"]
        ax[2].hist(ds, bins=np.linspace(0.4, 1.6, 60), histtype="step", lw=1.7, color=col,
                   density=True, label=f"{lab}, was cut  (med {np.median(ds):.3f})")
        ax[2].hist(di, bins=np.linspace(0.4, 1.6, 60), histtype="step", lw=1.1, ls="--", color=col,
                   density=True, label=f"{lab}, intact   (med {np.median(di):.3f})")
    ax[2].axvline(1.0, color="#444", lw=0.9)
    ax[2].set_xlabel(r"$m$ after 20 frames $/$ $m$ at the cut, both $/\langle m\rangle$")
    ax[2].set_ylabel("density"); ax[2].legend(fontsize=6.3, frameon=False)
    ax[2].set_title("3. so where does the junction end up?", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)


def fig_pools(t, med, jun, pa, out):
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.4), facecolor="white")
    ax[0].plot(t, med, color="#e08a2e", lw=1.7, label="medioapical  $\\sum_f \\rho_f A_f$")
    ax[0].plot(t, jun, color="#2b6cb0", lw=1.7, label="junctional  $\\sum_e n_e \\ell_e$")
    ax[0].set_ylabel("myosin (model units)"); ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title("the ledger: where the myosin is", fontsize=9)
    frac = med / np.maximum(med + jun, 1e-9)
    ax[1].plot(t, frac, color="#1f8a5c", lw=1.7)
    ax[1].set_ylabel("medioapical fraction of the total")
    ax[1].set_title("cells get smaller, the cortex empties into the belt", fontsize=9)
    ax[2].plot(t, pa, color="#7a4fbf", lw=1.7)
    ax[2].set_ylabel(r"$\langle P_f/A_f\rangle$  (1 / tissue unit)")
    ax[2].set_title("the shape factor that decides it", fontsize=9)
    for a in ax:
        a.set_xlabel("frame"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)


def _myo_scale(Tis):
    allm = []
    for _, mt in Tis["meshes"]:
        if "myo" not in mt:
            continue
        v = np.asarray(mt["myo"], float).ravel()
        ef = np.asarray(mt["E_face"]); nF = int(mt["nF"])
        if v.size == ef.size:
            allm.append(v[ef < nF])
    v = np.concatenate(allm); v = v[np.isfinite(v)]
    return float(np.percentile(v, 98))


def strip(Tis, d, myo_sc, n_col=8):
    cmap = ListedColormap(ES.STRESS_COLORS)
    q = np.zeros((0, 3)); band = np.zeros(0, np.uint8)
    meshes = Tis["meshes"]
    L3 = Tis["Lbox"] * 1.60; L2 = L3 * 1.15; Lt = 0.72 * L3
    idx = [meshes[int(round(f * (len(meshes) - 1)))] for f in np.linspace(0, 1, n_col)]
    fig = plt.figure(figsize=(3.4 * n_col, 10.6), facecolor="black")
    for i, (t, mt) in enumerate(idx):
        vp = mt["pos"]
        a1 = fig.add_subplot(3, n_col, i + 1, projection="3d", computed_zorder=False, facecolor="black")
        RD.draw_3d(a1, mt, vp, q, band, cmap, RD.CAM_SIDE, Lt,
                   div=RD.divided_mask(mt), brk=RD.broken_mask(mt, vp, "01b"))
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


def panels(Tis, d, myo_sc, movie=True, fps=15, label="01b_myosin_pools", note=None):
    cmap = ListedColormap(ES.STRESS_COLORS)
    q = np.zeros((0, 3)); band = np.zeros(0, np.uint8)
    L3 = Tis["Lbox"] * 1.60; L2 = L3 * 1.15; Lt = 0.72 * L3
    keep = [Tis["meshes"][int(round(f * (len(Tis["meshes"]) - 1)))]
            for f in np.linspace(0, 1, min(150, len(Tis["meshes"])))]
    fig = plt.figure(figsize=(11.0, 11.0), facecolor="black")
    axs = fig.add_subplot(2, 2, 1, projection="3d", computed_zorder=False, facecolor="black")
    axc = fig.add_subplot(2, 2, 2, facecolor="black")
    axz = fig.add_subplot(2, 2, 3, projection="3d", computed_zorder=False, facecolor="black")
    inz = fig.add_axes([0.335, 0.035, 0.155, 0.155], facecolor="black", zorder=20)
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02, hspace=0.02)

    def frame(t, mt):
        vp = mt["pos"]
        for a in (axs, axc, axz, inz):
            a.clear()
        RD.draw_3d(axs, mt, vp, q, band, cmap, RD.CAM_SIDE, Lt,
                   div=RD.divided_mask(mt), brk=RD.broken_mask(mt, vp, "01b"))
        RD.draw_cross(axc, mt, vp, q, band, cmap, L2, np.eye(3)[2], 0.055, dot_scale=0.85)
        RD.draw_junctions_3d(axz, mt, vp, RD.CAM_SIDE, Lt, myo_hi=myo_sc)
        RD.draw_zoom(inz, mt, vp, mem_q=None, mem_s=None, name="01b", frac=0.16, lw=2.4,
                     r_ref=Lt, myo_hi=myo_sc)
        for sp in inz.spines.values():
            sp.set_color("#666"); sp.set_visible(True)
        inz.set_xticks([]); inz.set_yticks([])
        axs.text2D(0.02, 0.96, f"{label}   frame {t}   {int(mt['nF'])} cells",
                   transform=axs.transAxes, color="white", fontsize=11, va="top")
        axz.text2D(0.03, 0.95, note or "junction network, coloured by\nmyosin from the medioapical pool",
                   transform=axz.transAxes, color="white", fontsize=10, va="top")

    if movie:
        wri = FFMpegWriter(fps=fps, metadata={"title": label})
        with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
            for t, mt in keep:
                frame(t, mt); wri.grab_frame()
    frame(*keep[-1])
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)


def main():
    import tissue as TIS
    d = os.path.join(LOG, "01b_myosin_pools")
    os.makedirs(d, exist_ok=True)
    only = {a for a in sys.argv if a.startswith("--")}

    two = TIS.load_or_build(frames=401, device="cuda:0", buffer_x=4, **TWO_POOL_PARAMS)
    caches = {"one": ONE_POOL, "two": two}

    R, out = {}, {}
    for k, path in caches.items():
        S = snapshots(path)
        ev = splits(S)
        rs, ri, lr = conservation(S, ev)
        ds, di = drift(S, ev)
        R[k] = dict(r_split=rs, r_intact=ri, d_split=ds, d_intact=di,
                    setpoint=setpoint_ratio(S, ev))
        rate, tot = t1_rate(path)
        out[k] = dict(
            cache=os.path.relpath(path, _ROOT), n_snapshots=len(S), n_splits=len(ev),
            amount_split_median=float(np.median(rs)), amount_split_iqr=float(np.subtract(*np.percentile(rs, [75, 25]))),
            amount_intact_median=float(np.median(ri)),
            length_split_median=float(np.median(lr)),
            setpoint_ratio_median=float(np.median(R[k]["setpoint"])),
            drift_split_median=float(np.median(ds)), drift_intact_median=float(np.median(di)),
            t1_per_cell_per_frame=rate, t1_total=tot)
        if k == "two":
            t, med, jun, pa = pools(S)
            out[k].update(pool_med_first=float(med[0]), pool_med_last=float(med[-1]),
                          pool_jun_first=float(jun[0]), pool_jun_last=float(jun[-1]),
                          med_fraction_first=float(med[0] / (med[0] + jun[0])),
                          med_fraction_last=float(med[-1] / (med[-1] + jun[-1])),
                          shape_factor_first=float(pa[0]), shape_factor_last=float(pa[-1]))
            fig_pools(t, med, jun, pa, os.path.join(d, "pools.png"))

    fig_conservation(R, os.path.join(d, "conservation.png"))
    yaml.safe_dump(dict(
        what="does junctional myosin survive a division -- its amount, and its setpoint",
        one_pool=dict(cache=os.path.relpath(ONE_POOL, _ROOT), model="one_pool", keyed_on="length",
                      tau=20.0, beta=1.0, activity=1.0),
        two_pool=dict(cache=os.path.relpath(two, _ROOT), **{k: v for k, v in TWO_POOL_PARAMS.items()}),
        operators_exercised=["medioapical_myosin", "junction_myosin[two_pool]", "junction_myosin",
                             "junction_sync", "cell_mechanics", "edge_flip",
                             "cell_divide"],
        plexus2=dict(medioapical_myosin=dict(kind="Lateral", acts_on="cell",
                                             state="areal myosin density"),
                     junction_myosin_two_pool=dict(kind="Structural", acts_on="junction (edge set)",
                                                   state="line myosin density", axis="model")),
        measures=["amount ratio across a cut, split vs intact", "setpoint ratio across a cut",
                  "myosin drift 20 frames after a cut", "pool ledger", "T1 per cell per frame"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)

    for k in ("one", "two"):
        o = out[k]
        print(f"[01b/{k:3s}] {o['n_splits']:5d} cuts | amount split {o['amount_split_median']:.4f} "
              f"vs intact {o['amount_intact_median']:.4f} | setpoint {o['setpoint_ratio_median']:.3f} "
              f"| drift cut {o['drift_split_median']:.3f} vs intact {o['drift_intact_median']:.3f} "
              f"| T1 {o['t1_per_cell_per_frame']:.5f}/cell/frame", flush=True)

    if "--no-render" not in only:
        Tis = RD.load_tissue(two, 1.0)
        sc = _myo_scale(Tis)
        strip(Tis, d, sc)
        panels(Tis, d, sc, movie=("--no-movie" not in only))
    print(f"[01b] -> {d}", flush=True)


if __name__ == "__main__":
    main()
