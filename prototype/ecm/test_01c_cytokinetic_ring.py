#!/usr/bin/env python
"""test_01c_cytokinetic_ring -- the newborn junction: what it starts with, and where that comes from.

    python test_01c_cytokinetic_ring.py [--no-movie|--no-render]  ->  log/okuda_ECM/01c_cytokinetic_ring/

THE DEFECT THIS FOLDER EXISTS FOR WAS SEEN IN A MOVIE. In the two-pool run of `01b`, the myosin at a
division site sits on the two HALVES of a contact the division cut (m/<m> = 1.066) while the
daughter--daughter interface -- the one place a real dividing cell puts almost all of its myosin -- is
the DIMMEST thing in the frame at 0.628. Both halves of that are wrong, and for different reasons.

THE DARK NEWBORN IS A UNITS ACCIDENT. `myo_new = 1.0` was an absolute line density in a model whose
mean line density runs 1.07 -> 1.97 -> 1.48 over 401 frames, so a newborn junction was pinned at
between 51% and 93% of whatever its neighbours happened to hold, and the number that decided it was
the frame index. The repair is to state it against the only local scale the model has,
n*_f = tau_jun * k_ex * rho_f -- the density the supply into that cell's belt sustains -- so that
`myo_new` means "a newborn junction starts at this FRACTION of a mature one here".

THE MISSING RING IS A MECHANISM. The cytokinetic ring is the most myosin-II-rich structure a cell
assembles and it constricts exactly at the nascent interface; in epithelia the new adherens junction
is built out of it, with the neighbouring cell contributing (Herszterg et al., Dev. Cell 24:256,
2013; Founounou et al., Dev. Cell 24:242, 2013). `cytokinetic_ring` deposits `ring * n*_f` there and
DEBITS the medioapical pool that built it, so the ring moves myosin rather than creating it and the
conservation ledger of `01b` still closes.

THREE RUNS, one variable at a time: the `01b` two-pool tissue (absolute `myo_new`, no ring), the same
with `myo_new` relative, and that plus the ring. What is reported is the myosin of a junction by CLASS
-- survivor, split half, newborn interface -- and how a newborn's myosin decays afterwards, which is
the test that the deposit relaxes on the belt's own timescale rather than needing one of its own.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "prototype", "Tyssue")):
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
import test_01b_myosin_pools as B

LOG = os.path.join(_ROOT, "log", "okuda_ECM")

# One variable at a time. `absolute` IS the 01b two-pool run -- same parameters, same cache -- so the
# first column of every table below is the number that folder reported, not a re-run of it.
BASE = dict(myosin=1.0, myo_tau=20.0, myo_new=1.0, myo_model="two_pool",
            myo_k_on=0.219, myo_tau_med=20.0, myo_k_ex=0.05, myo_beta_T=0.0)
RUNS = [("absolute", dict(myo_new_rel=False, myo_ring=0.0)),
        ("relative", dict(myo_new_rel=True, myo_ring=0.0)),
        ("relative+ring", dict(myo_new_rel=True, myo_ring=3.0))]
COL = {"absolute": "#c0392b", "relative": "#e08a2e", "relative+ring": "#2b6cb0"}


def classify(S):
    """Every live junction at each snapshot, labelled by how it got there.

    survivor      -- present in the previous snapshot too
    split half    -- one endpoint is a vertex `divide_3d` appended: half of a contact it cut
    newborn face  -- BOTH endpoints appended: the interface between the two daughters

    The two-new-endpoints test is the same one `cytokinetic_ring` uses to decide what to seed, so the
    measurement and the mechanism agree on what a newborn is by construction rather than by comment.
    """
    out = {"survivor": [], "half": [], "born": []}
    born_at = {}
    for i in range(len(S) - 1):
        a, b = S[i], S[i + 1]
        nv0, mb = a["nv"], b["m_mean"]
        cut = set()
        nbr = {}
        for (u, v) in b["tab"]:
            if (u >= nv0) ^ (v >= nv0):
                new, old = (u, v) if u >= nv0 else (v, u)
                nbr.setdefault(new, []).append(old)
        for new, olds in nbr.items():
            if len(olds) != 2:
                continue
            p = (min(olds), max(olds))
            if p not in a["tab"] or p in b["tab"]:
                continue
            for o in olds:
                cut.add((min(o, new), max(o, new)))
        for k, (mv, lv) in b["tab"].items():
            if k[0] >= nv0 and k[1] >= nv0:
                out["born"].append(mv / mb); born_at.setdefault(k, i + 1)
            elif k in cut:
                out["half"].append(mv / mb)
            elif k in a["tab"]:
                out["survivor"].append(mv / mb)
    return {k: np.asarray(v) for k, v in out.items()}, born_at


def newborn_decay(S, born_at, horizon=15):
    """A newborn interface's myosin against snapshots since it was born, relative to the tissue mean.

    THE POINT OF THE CURVE, not just its endpoints. If the ring deposit needed a decay timescale of
    its own, this would relax on some rate nobody specified; it does not, because `dN/dt = J - N/tau`
    already pulls the interface from `ring * n*` back to `n*`. The curve is therefore a prediction of
    the belt's own turnover time and not a fitted one.
    """
    curve = np.full((horizon + 1, 2), np.nan)
    acc = [[] for _ in range(horizon + 1)]
    for k, j0 in born_at.items():
        for d in range(horizon + 1):
            j = j0 + d
            if j < len(S) and k in S[j]["tab"]:
                acc[d].append(S[j]["tab"][k][0] / max(S[j]["m_mean"], 1e-9))
    for d, v in enumerate(acc):
        if v:
            curve[d] = (np.median(v), len(v))
    return curve


def fig_newborn(R, out):
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.4), facecolor="white")
    names = [n for n, _ in RUNS]
    x = np.arange(3)
    w = 0.26
    for i, n in enumerate(names):
        c = R[n]["cls"]
        vals = [np.median(c["survivor"]), np.median(c["half"]), np.median(c["born"])]
        ax[0].bar(x + (i - 1) * w, vals, w, color=COL[n], label=n)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax[0].text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.4)
    ax[0].axhline(1.0, color="#444", lw=0.9)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["survivor", "split half", "newborn\ninterface"])
    ax[0].set_ylabel(r"$m/\langle m\rangle$ at the snapshot after")
    ax[0].legend(fontsize=6.6, frameon=False); ax[0].set_title("where the myosin is, by class", fontsize=9)

    for n in names:
        c = R[n]["decay"]
        d = np.arange(c.shape[0]) * 2.0                 # snapshots are every 2 frames
        ax[1].plot(d, c[:, 0], color=COL[n], lw=1.7, marker="o", ms=2.6, label=n)
    ax[1].axhline(1.0, color="#444", lw=0.9)
    ax[1].set_xlabel("frames since the interface was born")
    ax[1].set_ylabel(r"$m/\langle m\rangle$")
    ax[1].legend(fontsize=6.6, frameon=False)
    ax[1].set_title(r"and how it relaxes ($\tau_{\rm jun}=20$ frames)", fontsize=9)

    for n in names:
        c = R[n]["cls"]["born"]
        ax[2].hist(c, bins=np.linspace(0, 4, 70), histtype="step", lw=1.7, color=COL[n],
                   density=True, label=f"{n} (med {np.median(c):.2f})")
    ax[2].axvline(1.0, color="#444", lw=0.9)
    ax[2].set_xlabel(r"$m/\langle m\rangle$, newborn interfaces only")
    ax[2].set_ylabel("density"); ax[2].legend(fontsize=6.6, frameon=False)
    ax[2].set_title("the junction a cytokinetic ring builds", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)


def fig_ledger(R, out):
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.4), facecolor="white")
    for n in [x for x, _ in RUNS]:
        t, med, jun, pa = R[n]["pools"]
        ax[0].plot(t, med, color=COL[n], lw=1.5, ls="--")
        ax[0].plot(t, jun, color=COL[n], lw=1.7, label=n)
        ax[1].plot(t, med / np.maximum(med + jun, 1e-9), color=COL[n], lw=1.7, label=n)
    ax[0].set_ylabel("myosin (model units)")
    ax[0].set_title("the ledger: junctional (solid), medioapical (dashed)", fontsize=9)
    ax[0].legend(fontsize=6.6, frameon=False)
    ax[1].set_ylabel("medioapical fraction of the total")
    ax[1].set_title("the ring moves myosin, it does not make it", fontsize=9)
    ax[1].legend(fontsize=6.6, frameon=False)
    names = [n for n, _ in RUNS]
    ax[2].bar(np.arange(len(names)), [R[n]["t1"] for n in names],
              color=[COL[n] for n in names], width=0.55)
    for i, n in enumerate(names):
        ax[2].text(i, R[n]["t1"] + 2e-5, f"{R[n]['t1']:.5f}", ha="center", fontsize=7)
    ax[2].set_xticks(np.arange(len(names))); ax[2].set_xticklabels(names, fontsize=7.5)
    ax[2].set_ylabel("T1 per cell per frame")
    ax[2].set_title("and what it costs in neighbour exchange", fontsize=9)
    for a in ax[:2]:
        a.set_xlabel("frame")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)


def main():
    import tissue as TIS
    d = os.path.join(LOG, "01c_cytokinetic_ring")
    os.makedirs(d, exist_ok=True)
    only = {a for a in sys.argv if a.startswith("--")}

    R, out, paths = {}, {}, {}
    for name, kw in RUNS:
        p = TIS.load_or_build(frames=401, device="cuda:0", buffer_x=4, **BASE, **kw)
        paths[name] = p
        S = B.snapshots(p)
        cls, born_at = classify(S)
        rate, tot = B.t1_rate(p)
        t, med, jun, pa = B.pools(S)
        R[name] = dict(cls=cls, decay=newborn_decay(S, born_at), pools=(t, med, jun, pa), t1=rate)
        out[name] = dict(
            cache=os.path.relpath(p, _ROOT), **{k: float(v) for k, v in kw.items()},
            n_survivor=int(cls["survivor"].size), n_half=int(cls["half"].size),
            n_born=int(cls["born"].size),
            m_survivor=float(np.median(cls["survivor"])), m_half=float(np.median(cls["half"])),
            m_born=float(np.median(cls["born"])),
            born_decay=[float(v) for v in R[name]["decay"][:, 0]],
            med_fraction_first=float(med[0] / (med[0] + jun[0])),
            med_fraction_last=float(med[-1] / (med[-1] + jun[-1])),
            t1_per_cell_per_frame=rate, t1_total=tot)

    fig_newborn(R, os.path.join(d, "newborn.png"))
    fig_ledger(R, os.path.join(d, "ledger.png"))
    yaml.safe_dump(dict(
        what="the newborn daughter-daughter junction: an absolute myo_new, a relative one, and a ring",
        runs={n: dict(**{k: (float(v) if not isinstance(v, bool) else v) for k, v in kw.items()},
                      cache=os.path.relpath(paths[n], _ROOT)) for n, kw in RUNS},
        common=BASE,
        operators_exercised=["medioapical_myosin", "junction_myosin[two_pool]", "cytokinetic_ring",
                             "junction_myosin_sync", "shape_energy_3d", "reconnect_t1_3d",
                             "divide_3d"],
        plexus2=dict(cytokinetic_ring=dict(kind="Structural", acts_on="junction (edge set)",
                                           state="seeds the keyed store, debits the cell pool")),
        measures=["myosin by junction class", "newborn decay toward the local steady state",
                  "pool ledger", "T1 per cell per frame"]),
        open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)

    for n, _ in RUNS:
        o = out[n]
        print(f"[01c/{n:14s}] survivor {o['m_survivor']:.3f}  split half {o['m_half']:.3f}  "
              f"NEWBORN {o['m_born']:.3f}  (n={o['n_born']})  |  med frac "
              f"{o['med_fraction_first']:.3f}->{o['med_fraction_last']:.3f}  |  "
              f"T1 {o['t1_per_cell_per_frame']:.5f}", flush=True)

    if "--no-render" not in only:
        Tis = RD.load_tissue(paths["relative+ring"], 1.0)
        sc = B._myo_scale(Tis)
        B.strip(Tis, d, sc)
        B.panels(Tis, d, sc, movie=("--no-movie" not in only), label="01c_cytokinetic_ring",
                 note="junction network, coloured by myosin\nbright = a ring just built this one")
    print(f"[01c] -> {d}", flush=True)


if __name__ == "__main__":
    main()
