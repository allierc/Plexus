"""exp17: A FEW NEURONS, RECORDED AGAINST LEARNED (Cedric, 2026-10-05: "a few traces true vs inferred, with the figure
template of connectome_gnn.pdf Supp. Fig. 14; first choose neurons representative of different parts of the fish; a map
to point the probed neurons"). Local.

THE CHOICE, from the RECORDING ONLY (blind to the model, so not picked for a good fit): the neurons that move (SD > 0)
split by position into K regions (k-means on x, y, z, um); in each region the neuron whose recorded trace correlates
most with its region's mean trace -- the most typical neuron of that part of the brain. Numbered head to tail.
THE FIGURE (black): left, the brain from above and from the side with the probed neurons numbered; right, each neuron's
free-rollout trace over the 2 h at the rollout's 800 saved frames (one every ~9 s), recorded green, learned white, on the
neuron's own scale, its Pearson r over the 2 h at the right; the stimulus conditions as blocks above.

    PYTHONPATH=src:tools python tools/exp17_traces.py zap_e15_cur_siren_mesh3 [K]
Writes presentation/figs/traces_<run>.png and data/traces_<run>.json (+ png/).

MORE LOCATIONS (Cedric, 2026-10-05: "reproduce on other locations x5, I want to see more examples"):
    PYTHONPATH=src:tools python tools/exp17_traces.py zap_g17_mesh3 --sets 5 [K]
the moving neurons split into 5 x K regions by position (k-means), the regions ranked head to tail and dealt in turn to
5 sets, so each set holds K regions spread over the whole brain; in each region the most typical neuron, as above.
Writes traces_<run>_s1..s5 (png + json, the json with each set's region centres for tools/exp17_terms.py --sets).
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
G = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "log", "training", "zapbench")


def pick(X, pos, K, seed=0):
    """K representative neurons: k-means regions on position, in each the neuron most correlated with its region mean."""
    from sklearn.cluster import KMeans
    act = np.where(X.std(0) > 1e-6)[0]
    lab = KMeans(K, n_init=4, random_state=seed).fit_predict(pos[act])
    Z = (X[:, act] - X[:, act].mean(0)) / X[:, act].std(0)
    out = []
    for k in range(K):
        m = np.where(lab == k)[0]
        mean = Z[:, m].mean(1)
        mean = (mean - mean.mean()) / mean.std()
        r = (Z[:, m] * mean[:, None]).mean(0)
        out.append((int(act[m[np.argmax(r)]]), float(r.max()), int(len(m)), pos[act[m]].mean(0)))
    return out


def pick_sets(X, pos, K, n_sets, seed=0):
    """n_sets sets of K representative neurons at other locations: K * n_sets k-means regions on position, ranked head to
    tail (x) and dealt in turn to the sets; in each region the neuron most correlated with its region mean. Returns
    [(picks as pick(), region centres [K, 3])] per set."""
    from sklearn.cluster import KMeans
    act = np.where(X.std(0) > 1e-6)[0]
    km = KMeans(K * n_sets, n_init=4, random_state=seed).fit(pos[act])
    lab, C = km.labels_, km.cluster_centers_
    rank = np.argsort(C[:, 0])                                                    # head to tail
    out = []
    for s_ in range(n_sets):
        regs = rank[s_::n_sets]
        sel = []
        for k in regs:
            m = np.where(lab == k)[0]
            Z = (X[:, act[m]] - X[:, act[m]].mean(0)) / X[:, act[m]].std(0)
            mean = Z.mean(1)
            mean = (mean - mean.mean()) / mean.std()
            r = (Z * mean[:, None]).mean(0)
            sel.append((int(act[m[np.argmax(r)]]), float(r.max()), int(len(m)), pos[act[m]].mean(0)))
        out.append((sel, C[regs]))
    return out


def main(run, K=12, n_sets=0, resid=False):
    import matplotlib
    matplotlib.use("Agg")
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import _brain_view
    spec = T.load(run)
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    X = np.asarray(rec["dff"], np.float64)
    P = _brain_view(np.asarray(rec["pos_um"], np.float64))
    if resid:      # the same neurons as the plain figure, read back from its json
        j_ = json.load(open(os.path.join(EXP, "data", f"traces_{run}.json")))
        sel = [(n["index"], n["r_with_region_mean"], n["region_size"], np.asarray(n["pos_view_um"]))
               for n in j_["neurons"]]
        draw_recorded_resid(sel, X, P, rec)
        return
    if n_sets:
        for s_, (sel, C) in enumerate(pick_sets(X, P, K, n_sets), 1):
            draw(run, sorted(sel, key=lambda t: t[3][0]), X, P, rec, K, f"_s{s_}", {"set": s_, "n_sets": n_sets,
                                                                                  "region_centres_view_um": C.tolist()})
        return
    draw(run, sorted(pick(X, P, K), key=lambda t: t[3][0]), X, P, rec, K, "", {})      # head (left) to tail


def resid_on_brain_mean(A):
    """Each column regressed on the brain mean of A's own columns (exp17_slides.local_r's regression): A_i - a_i -
    beta_i b, b the mean over the columns, beta_i the least-squares slope."""
    A = A - A.mean(0)
    b = A.mean(1)
    b = b - b.mean()
    beta = (A * b[:, None]).sum(0) / max(float((b * b).sum()), 1e-30)
    return A - b[:, None] * beta[None]


def draw_recorded_resid(sel, X, P, rec):
    """THE RECORDING, THE BRAIN MEAN REMOVED (Cedric, 2026-10-06: "orange should be observed minus regression, not the
    learned"): no model. Left the probed neurons on the brain; middle their recorded dF/F (green) with the brain mean b
    below it; right the same traces minus their regression on b, e_i = x_i - a_i - beta_i b (orange). The whole
    recording, a neuron on its own scale (its recorded trace's 1st-99.5th percentile, the same in both panels); at the
    right the share of its variance b carries, r_i^2. -> figs/traces_recorded_resid.png, data/traces_recorded_resid.json"""
    import matplotlib.pyplot as plt
    T = X.shape[0]
    b = X.mean(1)
    ids = [n for n, *_r in sel]
    E = resid_on_brain_mean(X)[:, ids]
    Xs = X[:, ids]
    r_b = [float(np.corrcoef(X[:, n], b)[0, 1]) for n in ids]
    tm = np.arange(T) * 0.914 / 60
    cond = rec["condition"]
    names = [str(n) for n in rec["names"]]
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    K = len(sel)
    cols = plt.get_cmap("tab20")(np.arange(K) % 20)
    GR, OR = "#2ca02c", "#ff9f1c"
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10.6), facecolor="black")
    o = np.argsort(P[:, 2])
    for ys, h, view in ((0.55, 0.36, "top"), (0.30, 0.22, "side")):
        ax = fig.add_axes([0.0, ys, 0.215, h])
        ax.axis("off")
        Y = P[:, 1] if view == "top" else P[:, 2]
        ax.scatter(P[o, 0], Y[o], s=0.05, c="0.32", linewidths=0)
        for i, n in enumerate(ids):
            ax.scatter(P[n, 0], Y[n], s=40, c=[cols[i]], linewidths=0)         # dots (Cedric, 2026-10-07)
            ax.text(P[n, 0], Y[n] + (14 if view == "top" else 9), str(i + 1), color=cols[i], fontsize=9, ha="center",
                    va="bottom", weight="bold")
        ax.set_aspect("equal")
        fig.text(0.005, ys + h + 0.005, "from above (head left)" if view == "top" else "from the side", color="white",
                 fontsize=11)
    for k_, (A_, c_, lab_) in enumerate(((Xs, GR, "recorded dF/F"), (E, OR, "recorded minus its regression on the brain mean"))):
        x0 = 0.245 + k_ * 0.375
        tr = fig.add_axes([x0, 0.37, 0.33, 0.56])                    # the same rows in both panels
        for i in range(K):
            lo, hi = np.percentile(Xs[:, i] - Xs[:, i].mean(), [1, 99.5])
            sc = max(hi - lo, 1e-6)
            y_ = (A_[:, i] - A_[:, i].mean() - lo) / sc
            off = (K - 1 - i) * 1.25
            tr.plot(tm, y_ + off, color=c_, lw=0.5)
            if k_ == 0:
                tr.text(-1.5, off + 0.4, str(i + 1), color=cols[i], fontsize=10, ha="right", va="center", weight="bold")
            else:
                tr.text(tm[-1] + 1.0, off + 0.4, f"$r^2$ {100 * r_b[i] ** 2:.0f} %", color="0.85", fontsize=8.5, va="center")
        for i_, (a_, b2_) in enumerate(zip(st, en)):
            tr.axvspan(tm[a_], tm[b2_], color=("0.16" if i_ % 2 else "0.08"), lw=0, zorder=0)
            tr.text((tm[a_] + tm[b2_]) / 2, K * 1.25 + 0.05, names[int(cond[a_])], color="0.75", fontsize=7, ha="center")
        tr.set_xlim(tm[0], tm[-1])
        tr.set_ylim(-0.4, K * 1.25 + 0.5)
        tr.set_yticks([])
        tr.set_title(lab_, color=c_, fontsize=11, loc="left", pad=16)
        tr.set_facecolor("black")
        for s_ in ("top", "right", "left"):
            tr.spines[s_].set_visible(False)
        if k_ == 0:
            tr.set_xticklabels([])
            # Cedric, 2026-10-07: the brain mean alone on its own scale, above the one with the neurons' band
            b1 = fig.add_axes([x0, 0.225, 0.33, 0.11])
            bm = fig.add_axes([x0, 0.07, 0.33, 0.11])
            for ax_ in (b1, bm):
                for i_, (a_, b2_) in enumerate(zip(st, en)):
                    ax_.axvspan(tm[a_], tm[b2_], color=("0.16" if i_ % 2 else "0.08"), lw=0, zorder=0)
                ax_.set_xlim(tm[0], tm[-1])
                ax_.tick_params(labelsize=8)
                ax_.set_facecolor("black")
                for s_ in ("top", "right"):
                    ax_.spines[s_].set_visible(False)
            b1.plot(tm, b, color="white", lw=0.6)
            b1.set_xticklabels([])
            b1.set_ylabel("brain mean\n$b$, dF/F", fontsize=8.5)
            # the range of the neurons' activity around it (Cedric, 2026-10-06): per frame, the 2.5th-97.5th percentile
            # of every neuron's dF/F -- the band 95 % of the neurons fall in
            q_ = np.concatenate([np.percentile(X[a_:a_ + 500].astype(np.float32), [2.5, 97.5], axis=1)
                                 for a_ in range(0, T, 500)], axis=1)
            bm.fill_between(tm, q_[0], q_[1], color="white", alpha=0.25, lw=0, label="95 % of the neurons")
            bm.plot(tm, b, color="white", lw=0.6, label="the brain mean")
            bm.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
            bm.set_xlim(tm[0], tm[-1])
            bm.set_ylabel("dF/F", fontsize=8.5)
            bm.tick_params(labelsize=8)
            bm.set_xlabel("time, min (the whole recording)", fontsize=9)
            bm.set_facecolor("black")
            for s_ in ("top", "right"):
                bm.spines[s_].set_visible(False)
        else:
            tr.set_xlabel("time, min (the whole recording)", fontsize=9)
            tr.tick_params(labelsize=8)
            for i in range(K):
                tr.text(-1.5, (K - 1 - i) * 1.25 + 0.4, str(i + 1), color=cols[i], fontsize=10, ha="right", va="center",
                        weight="bold")
    path = os.path.join(EXP, "presentation", "figs", "traces_recorded_resid.png")
    fig.savefig(path, dpi=120, facecolor="black", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    r2 = np.array(r_b) ** 2
    doc = {"neurons": [{"index": n, "r_with_brain_mean": r} for n, r in zip(ids, r_b)], "frames": int(T),
           "band_95_median_width": float(np.median(q_[1] - q_[0])),
           "r2_median": float(np.median(r2)), "r2_min": float(r2.min()), "r2_max": float(r2.max())}
    json.dump(doc, open(os.path.join(EXP, "data", "traces_recorded_resid.json"), "w"), indent=1)
    print("[traces recorded resid]", path, f"r^2 median {doc['r2_median']:.2f} ({doc['r2_min']:.2f} .. {doc['r2_max']:.2f})")


def draw_resid_split(run, sel, X, P, rec):
    """THE BRAIN MEAN REMOVED, recorded and learned apart (Cedric, 2026-10-06: "split the green and white in two plots,
    green and orange; the brain mean trace below"): left the probed neurons on the brain; middle each neuron's recorded
    residual e_i (green), right its learned residual (orange) -- each trace regressed on its OWN brain mean, the recorded
    on the recorded, the learned on the learned, over the rollout's frames (resid_on_brain_mean); a neuron on the same
    scale in both panels (its recorded residual's 1st-99.5th percentile), r of the two at the right. Below each panel the
    brain mean it was regressed on. -> figs/traces_<run>_resid.png, data/traces_<run>_resid.json"""
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(G, run, "results", f"{run}_movie.npz"))
    fr = z["frames"]
    pf = z["pred"].astype(np.float64)
    fin = np.isfinite(pf).all(0)
    Xf, Pf = X[fr][:, fin], pf[:, fin]
    Rx, Rp = resid_on_brain_mean(Xf), resid_on_brain_mean(Pf)
    col = np.cumsum(fin) - 1                                   # a neuron's column among the finite ones
    bx, bp = Xf.mean(1), Pf.mean(1)
    tm = fr * 0.914 / 60
    cond = rec["condition"][fr]
    names = [str(n) for n in rec["names"]]
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    K = len(sel)
    cols = plt.get_cmap("tab20")(np.arange(K) % 20)
    GR, OR = "#2ca02c", "#ff9f1c"
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10.6), facecolor="black")
    o = np.argsort(P[:, 2])
    for ys, h, view in ((0.52, 0.40, "top"), (0.10, 0.30, "side")):
        ax = fig.add_axes([0.0, ys, 0.215, h])
        ax.axis("off")
        Y = P[:, 1] if view == "top" else P[:, 2]
        ax.scatter(P[o, 0], Y[o], s=0.05, c="0.32", linewidths=0)
        for i, (n, *_r) in enumerate(sel):
            ax.scatter(P[n, 0], Y[n], s=55, facecolors="none", edgecolors=[cols[i]], linewidths=1.4)
            ax.text(P[n, 0], Y[n] + (14 if view == "top" else 9), str(i + 1), color=cols[i], fontsize=9, ha="center",
                    va="bottom", weight="bold")
        ax.set_aspect("equal")
        fig.text(0.005, ys + h + 0.005, "from above (head left)" if view == "top" else "from the side", color="white",
                 fontsize=11)
    rs = []
    panels = []
    for k_, (R_, b_, c_, lab_) in enumerate(((Rx, bx, GR, "recorded, the brain mean removed"),
                                            (Rp, bp, OR, "learned, the brain mean removed"))):
        x0 = 0.245 + k_ * 0.375
        tr = fig.add_axes([x0, 0.25, 0.33, 0.68])
        bm = fig.add_axes([x0, 0.07, 0.33, 0.13])
        panels.append(tr)
        for i, (n, *_r) in enumerate(sel):
            e_, ex_ = R_[:, col[n]], Rx[:, col[n]]
            lo, hi = np.percentile(ex_, [1, 99.5])
            sc = max(hi - lo, 1e-6)
            off = (K - 1 - i) * 1.25
            tr.plot(tm, (e_ - lo) / sc + off, color=c_, lw=0.7)
            if k_ == 0:
                tr.text(-1.5, off + 0.4, str(i + 1), color=cols[i], fontsize=10, ha="right", va="center", weight="bold")
            else:
                r = float(np.corrcoef(Rx[:, col[n]], Rp[:, col[n]])[0, 1])
                rs.append(r)
                tr.text(tm[-1] + 1.0, off + 0.4, f"r {r:+.2f}", color="0.85", fontsize=8.5, va="center")
        for i_, (a_, b2_) in enumerate(zip(st, en)):
            for ax_ in (tr, bm):
                ax_.axvspan(tm[a_], tm[b2_], color=("0.16" if i_ % 2 else "0.08"), lw=0, zorder=0)
            tr.text((tm[a_] + tm[b2_]) / 2, K * 1.25 + 0.05, names[int(cond[a_])], color="0.75", fontsize=7, ha="center")
        tr.set_xlim(tm[0], tm[-1])
        tr.set_ylim(-0.4, K * 1.25 + 0.5)
        tr.set_yticks([])
        tr.set_xticklabels([])
        tr.set_title(lab_, color=c_, fontsize=11, loc="left", pad=16)
        bm.plot(tm, b_, color=c_, lw=0.9)
        bm.set_xlim(tm[0], tm[-1])
        lo_, hi_ = min(bx.min(), bp.min()), max(bx.max(), bp.max())     # one dF/F scale for both brain means
        bm.set_ylim(lo_ - 0.03 * (hi_ - lo_), hi_ + 0.03 * (hi_ - lo_))
        bm.set_ylabel("brain\nmean, dF/F", fontsize=8.5)
        bm.tick_params(labelsize=8)
        bm.set_xlabel("time, min (the free rollout of the 2 h, 800 frames)", fontsize=9)
        for ax_ in (tr, bm):
            ax_.set_facecolor("black")
            for s_ in ("top", "right"):
                ax_.spines[s_].set_visible(False)
        tr.spines["left"].set_visible(False)
    path = os.path.join(EXP, "presentation", "figs", f"traces_{run}_resid.png")
    fig.savefig(path, dpi=120, facecolor="black", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    doc = {"run": run, "K": K, "resid": True, "neurons": [{"index": n, "r_learned_vs_recorded_resid": r}
                                                           for (n, *_r), r in zip(sel, rs)],
           "r_median": float(np.median(rs)), "r_min": float(np.min(rs)), "r_max": float(np.max(rs)),
           "brain_mean_r": float(np.corrcoef(bx, bp)[0, 1])}
    json.dump(doc, open(os.path.join(EXP, "data", f"traces_{run}_resid.json"), "w"), indent=1)
    print("[traces resid]", path, f"r median {doc['r_median']:+.2f} ({doc['r_min']:+.2f} .. {doc['r_max']:+.2f})")


def draw(run, sel, X, P, rec, K, suf, extra, resid=False):
    """The figure and json of one set of probed neurons (suf "" for the K-region pick, "_s<k>" for a set). resid: each
    trace first regressed on its OWN brain mean -- the recorded on the recorded brain mean, the learned on the learned
    (Cedric, 2026-10-06: "these traces after mean subtraction with regression"), over the rollout's frames and the neurons
    whose learned trace is finite."""
    import matplotlib.pyplot as plt
    z = np.load(os.path.join(G, run, "results", f"{run}_movie.npz"))
    fr = z["frames"]
    pred = z["pred"]
    if resid:
        pf = pred.astype(np.float64)
        fin = np.isfinite(pf).all(0)
        Rx, Rp = np.zeros_like(pf), np.zeros_like(pf)
        Rx[:, fin] = resid_on_brain_mean(X[fr][:, fin])
        Rp[:, fin] = resid_on_brain_mean(pf[:, fin])
        X, pred = np.zeros_like(X), Rp                     # X indexed at fr below: put the residuals there
        X[fr] = Rx
    tm = fr * 0.914 / 60
    cond = rec["condition"][fr]
    names = [str(n) for n in rec["names"]]
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    cols = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[::2][:K] if K <= 10 else plt.get_cmap("tab20")(np.arange(K) % 20)
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9.0), facecolor="black")
    o = np.argsort(P[:, 2])
    for j, (ys, h, view) in enumerate(((0.50, 0.42, "top"), (0.08, 0.30, "side"))):
        ax = fig.add_axes([0.01, ys, 0.33, h])
        ax.set_facecolor("black")
        ax.axis("off")
        Y = P[:, 1] if view == "top" else P[:, 2]
        ax.scatter(P[o, 0], Y[o], s=0.08, c="0.32", linewidths=0)
        for i, (n, *_r) in enumerate(sel):
            ax.scatter(P[n, 0], Y[n], s=70, facecolors="none", edgecolors=[cols[i]], linewidths=1.6)
            ax.text(P[n, 0], Y[n] + (14 if view == "top" else 9), str(i + 1), color=cols[i], fontsize=9, ha="center",
                    va="bottom", weight="bold")
        ax.set_aspect("equal")
        fig.text(0.015, ys + h + 0.01, "from above (head left)" if view == "top" else "from the side", color="white",
                 fontsize=11)
    tr = fig.add_axes([0.40, 0.06, 0.55, 0.86])
    tr.set_facecolor("black")
    rs = []
    for i, (n, rrep, size, _c) in enumerate(sel):
        x_, p_ = X[fr, n], pred[:, n].astype(np.float64)
        lo, hi = np.percentile(x_, [1, 99.5])
        sc = max(hi - lo, 1e-6)
        off = (K - 1 - i) * 1.25
        tr.plot(tm, (x_ - lo) / sc + off, color="#2ca02c", lw=0.8)
        tr.plot(tm, (p_ - lo) / sc + off, color="white", lw=0.7)
        r = float(np.corrcoef(x_, p_)[0, 1])
        rs.append(r)
        tr.text(-1.5, off + 0.4, str(i + 1), color=cols[i], fontsize=10, ha="right", va="center", weight="bold")
        tr.text(tm[-1] + 1.0, off + 0.4, f"r {r:+.2f}", color="0.85", fontsize=8.5, va="center")
    for i_, (a_, b_) in enumerate(zip(st, en)):
        tr.axvspan(tm[a_], tm[b_], color=("0.16" if i_ % 2 else "0.08"), lw=0, zorder=0)
        tr.text((tm[a_] + tm[b_]) / 2, K * 1.25 + 0.05, names[int(cond[a_])], color="0.75", fontsize=7.5, ha="center")
    tr.set_xlim(tm[0], tm[-1])
    tr.set_ylim(-0.4, K * 1.25 + 0.5)
    tr.set_yticks([])
    for s_ in ("top", "right", "left"):
        tr.spines[s_].set_visible(False)
    tr.set_xlabel("time, min (the free rollout of the 2 h, 800 frames)", fontsize=9)
    fig.text(0.40, 0.955, ("each trace minus its regression on its own brain mean -- " if resid else "")
             + "recorded (green), learned (white); each neuron on its own scale (its 1st-99.5th percentile)",
             color="0.85", fontsize=10)
    path = os.path.join(EXP, "presentation", "figs", f"traces_{run}{suf}.png")
    fig.savefig(path, dpi=120, facecolor="black", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    doc = {"run": run, "K": K, "neurons": [{"index": n, "region_size": size, "r_with_region_mean": rrep,
                                           "pos_view_um": c.tolist(), "r_learned_vs_recorded": r}
                                          for (n, rrep, size, c), r in zip(sel, rs)],
           "r_median": float(np.median(rs)), "r_min": float(np.min(rs)), "r_max": float(np.max(rs))} | extra
    json.dump(doc, open(os.path.join(EXP, "data", f"traces_{run}{suf}.json"), "w"), indent=1)
    print("[traces]", path, f"r median {doc['r_median']:+.2f} ({doc['r_min']:+.2f} .. {doc['r_max']:+.2f})")


if __name__ == "__main__":
    a_ = sys.argv[1:]
    ns_ = int(a_[a_.index("--sets") + 1]) if "--sets" in a_ else 0
    a_ = [x for i, x in enumerate(a_) if x != "--sets" and (i == 0 or a_[i - 1] != "--sets")]
    rs_ = "--resid" in a_
    a_ = [x for x in a_ if x != "--resid"]
    main(a_[0], *(int(x) for x in a_[1:2]), n_sets=ns_, resid=rs_)
