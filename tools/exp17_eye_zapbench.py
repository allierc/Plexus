"""exp17 appendix: the trained two-eye rig driven by ZAPBench's 2-h session, and the recorded neurons that follow it
(Cedric, 2026-10-08: "generate zapbench-like data with the rig simulation ... which block is the most relevant to compare
the generated data to the observed data? ... find correlation"). Local, no training.

THE SIMULATION is the trainer's own test of a test-only run: `Plexus_Main.py -o test zf_eye2_rig_zapbench` rolls the
trained zf_eye2_rig out on the corpus t9_zapbench_eye (plexus.tasks.zapbench: the session's rotation and lateral turning
drifts as conjugate image slip, 1 deg/s, every other condition 0) and records the 285 cells' voltage on the recording's
0.914-s clock (`task.record`). The cells' rate is tanh(v), the rig's activation.

THE BLOCK is ROTATION (frames 6623-7278, 10 min): 10 cycles of 30 s leftward then 30 s rightward rotating grating, the
one condition that is an optokinetic stimulus proper, the conjugate horizontal slip the rig was trained on, and long
enough (30-s holds against the integrator's 8-s time constant) for the circuit's dynamics -- ramps that saturate, cells
that integrate or merely follow -- to differ from the stimulus itself.

THE MATCH, per recorded neuron (all 100,759 destriped traces; the constant fill traces left out):
  model regressors   each model cell's rate through a nuclear-GCaMP kernel, an exponential of TAU_CA s (an assumption:
                     the release gives no kernel), on the 0.914-s clock
  stimulus control   the rotation direction (+1 left / -1 right) through the same kernel -- the regressor any neuron
                     that merely reports the stimulus will follow
  stimulus bank      the fair control: the direction through leaky integrators of N_BANK time constants, 0.1 to 30 s
                     (log-spaced, unit gain), both signs, then the kernel -- every follower and every leaky integrator
                     of the stimulus, given the same choice-of-best the 285 model cells get. A model cell says
                     something about the CIRCUIT only where it beats the bank
  cross-validation   the 10 cycles split even / odd: the best model cell (largest correlation) and the control's sign are
                     chosen on one half and scored on the other, both ways, averaged -- so picking the best of 285 cells
                     earns nothing by selection
  gain               held-out r(best model cell) - held-out r(stimulus control); gain_bank the same against the bank
Each neuron's region is the smallest Z-Brain mask holding it (data/atlas_destripe.npz).

THE POOLS (Cedric, 2026-10-08: "a proper localisation"; papers/Horizontal_integrator_abducens_literature_summary_v3.pdf):
each model type is matched only against recorded neurons where the literature puts it, on its own side --
  pretectum   AF5 cells        Z-Brain Pretectum + its Gad1b, dopaminergic and anterior vmat2 clusters (Kubo 2014: the
                               pretectum is necessary and sufficient for the OKR slow phase)
  r7/8        integrator       Rhombomere 7 (Z-Brain's r7 runs on through r8) + Medial Vestibular Nucleus, 50-200 um
                               caudal of the Mauthner soma (Miri 2011, Goncalves 2014: 50-150 um; Lee 2015)
  r5/6        abducens         Rhombomere 5 + 6, motor (AMN) and internuclear (AIN) cells both (Brysch 2019: AMN 30-70 um
                               ventral to the MLF, AIN medial / dorsal; Z-Brain has no MLF mask, so the match decides)
Side: the atlas midline is Z-Brain's width / 2 (247.8 um); which half is the fish's left is not known from the
registration, so both conventions are scored and the one the model fits better is kept (the rig's left and right cells
are mirror images under a conjugate drive, so the wrong convention pairs each neuron with its opposite).

    PYTHONPATH=src:tools python tools/exp17_eye_zapbench.py
-> presentation/figs/eye_zapbench_session.png, eye_zapbench_rotation.png; data/eye_zapbench.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FIGS = os.path.join(EXP, "presentation", "figs")
RUN, CORPUS, BLOCK = "zf_eye2_rig_zapbench", "t9_zapbench_eye", "rotation"
TAU_CA = 3.0                                    # s, the assumed nuclear-GCaMP decay
N_SHOW = 60                                     # recorded neurons in the rotation raster
N_BANK = 20                                     # leaky-integrator time constants of the stimulus bank
N_POOL_SHOW = 20                                # recorded neurons per pool in the rotation raster
MIDLINE_UM = 621 * 0.798 / 2                    # Z-Brain's width / 2
POOLS = (("pretectum", ("AF5",), ("Diencephalon - Pretectum", "Diencephalon - Pretectal Gad1b Cluster",
                                  "Diencephalon - Pretectal dopaminergic cluster",
                                  "Diencephalon - Anterior pretectum cluster of vmat2 Neurons"), None),
         ("r7/8", ("INTG",), ("Rhombencephalon - Rhombomere 7", "Rhombencephalon - Medial Vestibular Nucleus"), (50.0, 200.0)),
         ("r5/6", ("AMN", "AIN"), ("Rhombencephalon - Rhombomere 5", "Rhombencephalon - Rhombomere 6"), None))


def model_types(spec):
    """[(type, count)] of the rig's neuron set, in row order (the model yaml's `type_layout: ordered`)."""
    import yaml
    m = yaml.safe_load(open(os.path.join(ROOT, spec["model"]) if not os.path.isabs(spec["model"]) else spec["model"]))
    return [(k, int(v["count"])) for k, v in m["sets"]["neuron"]["types"].items()]


def calcium(x, dt, tau=TAU_CA):
    """An exponential kernel of `tau` s along axis 0, unit gain, on a clock of `dt` s."""
    a = np.exp(-dt / tau)
    y = np.empty_like(x, dtype=np.float64)
    acc = np.zeros(x.shape[1:])
    for k in range(len(x)):
        acc = a * acc + (1.0 - a) * x[k]
        y[k] = acc
    return y


def _z(x):
    x = x - x.mean(0)
    return x / np.maximum(x.std(0), 1e-12)


def load_sim():
    from plexus import trainer as T
    from plexus.tasks.trainer import load_split
    spec = T.load(RUN)
    res = os.path.join(T.out_dir(spec), "results")
    z = np.load(os.path.join(res, f"{RUN}_test_neuron_voltage.npz"))
    v = z["x"][0, :, :, 0].astype(np.float64)                              # [frames, 285]
    U, Y, _ = load_split(CORPUS, "test", "cpu")
    gaze = np.load(os.path.join(res, f"{RUN}_test_traces.npy"))[0]          # left eye, every 60-Hz frame
    test = json.load(open(os.path.join(res, f"{RUN}_test.json")))
    return spec, np.tanh(v), z["t_s"], U[0].numpy(), Y[0].numpy(), gaze, test


def load_rec():
    from plexus.paths import graphs_data_path
    d = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"), allow_pickle=True)
    a = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"), allow_pickle=True)
    return d, a


def match(r, S_rec, off, names, dff, regions, rnames):
    """The cross-validated match on the rotation block (see the docstring)."""
    b = list(names).index(BLOCK)
    B = np.arange(off[b], off[b + 1])
    stim = S_rec[:, 19].astype(np.float64)
    dt = 0.914
    Sm = calcium(r, dt)[B]                                                   # [nB, 285]
    sc = calcium(stim[:, None], dt)[B, 0]                                    # [nB]
    taus = np.logspace(-1, np.log10(30.0), N_BANK)
    bank = np.concatenate([calcium(calcium(stim[:, None], dt, tau), dt) for tau in taus], 1)[B]
    bank = np.concatenate([bank, -bank], 1)                                  # [nB, 2 N_BANK]
    O = dff[B].astype(np.float64)                                            # [nB, N]
    ok = O.std(0) > 1e-6
    seg = np.r_[0, np.flatnonzero(np.diff(stim[B]) != 0) + 1]
    cyc = np.zeros(len(B), int)
    for i, s0 in enumerate(seg):
        cyc[s0:] = i // 2
    folds = [cyc % 2 == 0, cyc % 2 == 1]

    def corr(fold):
        Oz, Sz, cz, Bz = _z(O[fold][:, ok]), _z(Sm[fold]), _z(sc[fold][:, None])[:, 0], _z(bank[fold])
        n = fold.sum()
        return Oz.T @ Sz / n, Oz.T @ cz / n, Oz.T @ Bz / n                   # [N_ok, 285], [N_ok], [N_ok, 2 N_BANK]

    C = [corr(f) for f in folds]
    r_model, r_stim, r_bank, best = np.zeros(ok.sum()), np.zeros(ok.sum()), np.zeros(ok.sum()), []
    for a_, b_ in ((0, 1), (1, 0)):
        i_ = C[a_][0].argmax(1)
        best.append(i_)
        r_model += C[b_][0][np.arange(len(i_)), i_] / 2
        r_stim += np.sign(C[a_][1]) * C[b_][1] / 2
        k_ = C[a_][2].argmax(1)
        r_bank += C[b_][2][np.arange(len(k_)), k_] / 2
    # the cell kept for display: the one the full block picks (both halves agree for the neurons that matter)
    Cf = corr(np.ones(len(B), bool))[0]
    best_full = Cf.argmax(1)
    idx = np.flatnonzero(ok)
    size = regions.sum(0)
    def region(j):
        m = np.flatnonzero(regions[j])
        return str(rnames[m[np.argmin(size[m])]]) if len(m) else "outside the atlas"
    return dict(B=B, Sm=Sm, sc=sc, O=O, idx=idx, r_model=r_model, r_stim=r_stim, r_bank=r_bank, C=C, Cf=Cf,
                gain=r_model - r_stim, gain_bank=r_model - r_bank, taus=taus,
                best=best_full, region=region, cyc=cyc, n_cycles=int(cyc.max() + 1),
                model_vs_stim=np.array([np.corrcoef(Sm[:, i], sc)[0, 1] if Sm[:, i].std() > 1e-9 else 0.0
                                        for i in range(Sm.shape[1])]))


def match_pools(m, tnames, P, R, rnames):
    """The match restricted to each model type's own anatomy and side (see THE POOLS) -> {convention: {pool: ...}}."""
    rnames = [str(n) for n in rnames]
    idx = m["idx"]
    Pk, Rk = P[idx], R[idx]
    yM = float(np.median(P[R[:, rnames.index("Rhombencephalon - Mauthner")]][:, 1]))
    side = np.array([t[-1] for t in tnames])
    out = {}
    for conv in ("L", "R"):                                   # the fish's side of the half x < midline
        nside = np.where(Pk[:, 0] < MIDLINE_UM, conv, "R" if conv == "L" else "L")
        res = {}
        for name, prefs, masks, band in POOLS:
            inp = np.zeros(len(idx), bool)
            for mk in masks:
                inp |= Rk[:, rnames.index(mk)]
            if band is not None:
                inp &= (Pk[:, 1] - yM >= band[0]) & (Pk[:, 1] - yM <= band[1])
            j = np.flatnonzero(inp)
            allow = (np.array([any(t.startswith(p_) for p_ in prefs) for t in tnames])[None, :]
                     & (side[None, :] == nside[j][:, None]))                       # [n_pool, 285]
            rm = np.zeros(len(j))
            for a_, b_ in ((0, 1), (1, 0)):
                ca = np.where(allow, m["C"][a_][0][j], -np.inf)
                i_ = ca.argmax(1)
                rm += m["C"][b_][0][j, i_] / 2
            best = np.where(allow, m["Cf"][j], -np.inf).argmax(1)
            res[name] = dict(j=j, r_model=rm, r_bank=m["r_bank"][j], best=best, y_mauthner_um=yM)
        out[conv] = res
    score = {c: float(np.mean(np.concatenate([v["r_model"] for v in out[c].values()]))) for c in out}
    keep = max(score, key=score.get)
    return out, keep, score


def session_fig(path, r, t_s, U, Y, gaze, d, types):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    off, names = d["offsets"], d["names"]
    dt60 = t_s[1] / max(int(round(t_s[1] / (1.0 / 60.0))), 1)
    fig = plt.figure(figsize=(13, 7.6), facecolor="black")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.0, 3.2], hspace=0.12, left=0.11, right=0.985, top=0.93, bottom=0.08)
    T60 = np.arange(len(U)) / 60.0
    ax = [fig.add_subplot(gs[i]) for i in range(3)]
    for a in ax:
        a.set_facecolor("black")
        a.tick_params(colors="white", labelsize=8)
        for s_ in a.spines.values():
            s_.set_color("0.5")
    k = slice(None, None, 27)                                              # 60 Hz -> ~0.45 s for drawing
    ax[0].plot(T60[k], U[k, 0], color="#e6a03c", lw=0.7)
    ax[0].set_ylabel("slip\n(deg/s)", color="white", fontsize=9)
    ax[1].plot(T60[k], Y[k, 0], color="#6fd08c", lw=0.9, label="target (the integrator law)")
    ax[1].plot(T60[:len(gaze)][k], gaze[k], color="white", lw=0.6, label="model, left eye")
    ax[1].set_ylabel("gaze\n(deg)", color="white", fontsize=9)
    ax[1].legend(loc="upper left", fontsize=7, frameon=False, labelcolor="white", ncol=2)
    vm = np.percentile(np.abs(r), 99)
    ax[2].imshow(r.T, aspect="auto", cmap="gray", vmin=-0.3 * vm, vmax=vm, interpolation="nearest",  # the deck's grey LUT
                 extent=(t_s[0], t_s[-1] + 0.914, r.shape[1], 0))
    y0 = 0
    for nm, c in types:
        ax[2].axhline(y0, color="#e6a03c", lw=0.4)
        ax[2].text(-0.006, 1 - (y0 + c / 2) / r.shape[1], f"{nm.replace('_', ' ')} ({c})", transform=ax[2].transAxes,
                   ha="right", va="center", color="white", fontsize=6)
        y0 += c
    ax[2].set_yticks([])
    ax[2].set_xlabel("time in the session (s)", color="white", fontsize=9)
    for a in ax:
        a.set_xlim(0, t_s[-1] + 0.914)
        for o in off[1:-1]:
            a.axvline(o * 0.914, color="0.45", lw=0.5, ls=":")
    for i, nm in enumerate(names):
        ax[0].text((off[i] + off[i + 1]) / 2 * 0.914, 1.04, str(nm), transform=ax[0].get_xaxis_transform(),
                   ha="center", va="bottom", color="white", fontsize=7.5)
    ax[0].set_xticklabels([]); ax[1].set_xticklabels([])
    ax[2].text(1.0, -0.075, f"rate tanh(v), grey from {-0.3 * vm:.2f} to {vm:.2f}", transform=ax[2].transAxes, ha="right",
               color="0.7", fontsize=7)
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)


def rotation_fig(path, m, types, d, r, gaze_rot, pools=None, tnames=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B = m["B"]
    t = (B - B[0]) * 0.914
    fig = plt.figure(figsize=(13, 7.6), facecolor="black")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 2.3, 2.3], width_ratios=[1, 0.012], hspace=0.16, wspace=0.01,
                          left=0.20, right=0.97, top=0.95, bottom=0.08)
    a0, a1, a2 = (fig.add_subplot(gs[i, 0]) for i in range(3))
    for a in (a0, a1, a2):
        a.set_facecolor("black"); a.tick_params(colors="white", labelsize=8)
        for s_ in a.spines.values():
            s_.set_color("0.5")
    a0.plot(t, m["sc"] / np.abs(m["sc"]).max(), color="#e6a03c", lw=1.0, label="stimulus control (direction, GCaMP)")
    a0.plot(t, gaze_rot / np.abs(gaze_rot).max(), color="white", lw=0.8, label="model gaze, left eye")
    a0.legend(loc="upper right", fontsize=7, frameon=False, labelcolor="white", ncol=2)
    a0.set_yticks([]); a0.set_xticklabels([])
    a0.set_ylabel("+ left\n- right", color="white", fontsize=8)
    Sm = _z(m["Sm"])
    a1.imshow(Sm.T, aspect="auto", cmap="gray", vmin=-0.75, vmax=2.5, interpolation="nearest",
              extent=(t[0], t[-1] + 0.914, Sm.shape[1], 0))
    y0 = 0
    for nm, c in types:
        a1.axhline(y0, color="#e6a03c", lw=0.4)
        a1.text(-0.006, 1 - (y0 + c / 2) / Sm.shape[1], f"{nm.replace('_', ' ')}", transform=a1.transAxes,
                ha="right", va="center", color="white", fontsize=5.5)
        y0 += c
    a1.set_yticks([]); a1.set_xticklabels([])
    a1.text(0.0, 1.01, "model: the 285 cells, rate through the GCaMP kernel, z-scored", transform=a1.transAxes,
            color="white", fontsize=8)
    rows, labels, seps = [], [], []
    for name, v in pools.items():
        o_ = np.argsort(-v["r_model"])[:N_POOL_SHOW]
        seps.append((len(rows), name, len(o_)))
        rows += [m["idx"][v["j"][k]] for k in o_]
        labels += [f"{str(tnames[v['best'][k]]).replace('_', ' ')}  {v['r_model'][k]:.2f}" for k in o_]
    O = _z(m["O"][:, rows])
    a2.imshow(O.T, aspect="auto", cmap="gray", vmin=-0.75, vmax=2.5, interpolation="nearest",
              extent=(t[0], t[-1] + 0.914, len(rows), 0))
    a2.set_yticks(np.arange(len(rows)) + 0.5)
    a2.set_yticklabels(labels, fontsize=3.6, color="white")
    for y0, name, n_ in seps:
        a2.axhline(y0, color="#e6a03c", lw=0.8)
        a2.text(-0.115, 1 - (y0 + n_ / 2) / len(rows), name, transform=a2.transAxes, ha="right", va="center",
                color="white", fontsize=8)
    a2.set_xlabel("time in the rotation block (s)", color="white", fontsize=9)
    a2.text(0.0, 1.01, f"recorded: per pool, the {N_POOL_SHOW} neurons most like a model cell of that pool and side "
                       f"(the cell and its held-out r on the left), z-scored dF/F", transform=a2.transAxes,
            color="white", fontsize=8)
    for a in (a0, a1, a2):
        a.set_xlim(t[0], t[-1] + 0.914)
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    spec, r, t_s, U, Y, gaze, test = load_sim()
    types = model_types(spec)
    d, a = load_rec()
    session_fig(os.path.join(FIGS, "eye_zapbench_session.png"), r, t_s, U, Y, gaze, d, types)
    m = match(r, d["stimulus"], d["offsets"], d["names"], d["dff"], a["regions"], a["names"])
    tnames = np.concatenate([[nm] * c for nm, c in types])
    pools, conv, conv_score = match_pools(m, tnames, a["atlas_um"], a["regions"], a["names"])
    k60 = np.round(m["B"] * 0.914 * 60).astype(int)
    rotation_fig(os.path.join(FIGS, "eye_zapbench_rotation.png"), m, types, d, r, gaze[np.minimum(k60, len(gaze) - 1)],
                 pools=pools[conv], tnames=tnames)
    g, gb, rm, rs, rb = m["gain"], m["gain_bank"], m["r_model"], m["r_stim"], m["r_bank"]
    top = np.argsort(-rm)[:200]
    by_type = {}
    for j in top:
        tt = str(tnames[m["best"][j]])
        by_type.setdefault(tt, []).append(m["region"](m["idx"][j]))
    from collections import Counter
    out = {"run": RUN, "corpus": CORPUS, "block": BLOCK, "tau_ca_s": TAU_CA, "frames": [int(m["B"][0]), int(m["B"][-1])],
           "n_cycles": m["n_cycles"], "n_recorded": int(d["dff"].shape[1]), "n_scored": int(len(m["idx"])),
           "test_rmse_deg": test["rmse"], "test_target_rms_deg": float(np.sqrt(test["target_variance"])),
           "model_cell_vs_stimulus_r": {"median": float(np.median(m["model_vs_stim"])),
                                        "min": float(m["model_vs_stim"].min()),
                                        "frac_above_0.9": float((np.abs(m["model_vs_stim"]) > 0.9).mean())},
           "bank_taus_s": [float(x) for x in m["taus"]],
           "held_out": {"n_r_model_gt_0.5": int((rm > 0.5).sum()), "n_r_stim_gt_0.5": int((rs > 0.5).sum()),
                        "n_r_bank_gt_0.5": int((rb > 0.5).sum()),
                        "median_gain_bank_where_r_model_gt_0.5": float(np.median(gb[rm > 0.5])),
                        "n_gain_bank_gt_0.05_and_r_gt_0.5": int(((gb > 0.05) & (rm > 0.5)).sum()),
                        "n_gain_gt_0.1_and_r_gt_0.5": int(((g > 0.1) & (rm > 0.5)).sum()),
                        "n_gain_gt_0.2_and_r_gt_0.5": int(((g > 0.2) & (rm > 0.5)).sum()),
                        "gain_p99": float(np.percentile(g, 99)), "gain_max": float(g.max())},
           "top": [{"neuron": int(m["idx"][j]), "region": m["region"](m["idx"][j]), "model_cell": int(m["best"][j]),
                    "model_type": str(tnames[m["best"][j]]), "r_model": float(rm[j]), "r_stim": float(rs[j]),
                    "r_bank": float(rb[j])}
                   for j in top[:25]],
           "top200_by_model_type": {k: {"n": len(v), "regions": Counter(v).most_common(3)} for k, v in
                                    sorted(by_type.items(), key=lambda kv: -len(kv[1]))}}
    pj = {}
    for name, v in pools[conv].items():
        rm_, rb_ = v["r_model"], v["r_bank"]
        o_ = np.argsort(-rm_)
        pj[name] = {"n": int(len(rm_)), "n_r_model_gt_0.5": int((rm_ > 0.5).sum()), "n_r_bank_gt_0.5": int((rb_ > 0.5).sum()),
                    "median_r_model": float(np.median(rm_)), "median_r_bank": float(np.median(rb_)),
                    "top20_mean_r_model": float(rm_[o_[:20]].mean()), "top20_mean_r_bank": float(rb_[o_[:20]].mean()),
                    "median_model_minus_bank_where_r_gt_0.5": (float(np.median((rm_ - rb_)[rm_ > 0.5]))
                                                               if (rm_ > 0.5).any() else None),
                    "top": [{"neuron": int(m["idx"][v["j"][k]]), "model_type": str(tnames[v["best"][k]]),
                             "region": m["region"](m["idx"][v["j"][k]]), "r_model": float(rm_[k]), "r_bank": float(rb_[k])}
                            for k in o_[:10]]}
    out["pools"] = {"side_convention": f"x < {MIDLINE_UM:.1f} um is the fish's {conv}",
                    "mean_held_out_r_by_convention": conv_score,
                    "y_mauthner_um": float(next(iter(pools[conv].values()))["y_mauthner_um"]), "by_pool": pj}
    json.dump(out, open(os.path.join(EXP, "data", "eye_zapbench.json"), "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "top"} for k, v in pj.items()}, indent=1))
    print("side", out["pools"]["side_convention"], conv_score)
    print(json.dumps({k: out[k] for k in ("model_cell_vs_stimulus_r", "held_out", "top200_by_model_type")}, indent=1))
    for row in out["top"][:12]:
        print(row)


if __name__ == "__main__":
    main()
