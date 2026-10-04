"""exp17: THE CONSENSUS FLOW OVER GRAPHS (Cedric, 2026-10-03: "now that you have movies spanning different graphs, make
an average and a median of them: if there is a structure in the flow it should remain in the average and/or median").

Local analysis. The wind fields of tools/exp17_wind.py (the learned messages, sender to receiver, gridded in the top
view and smoothed over 25 um; excitatory and inhibitory apart) of every listed run, on ONE grid (same recording, same
positions) and at the same 800 movie frames. Each run's two fields are divided by that run's own strength (the 99th
percentile of its wind speed, over both signs), so no run outweighs the others. Then, frame by frame and cell by cell:
    mean     the vector mean over the runs
    median   the component-wise median over the runs (robust to one odd graph)
each rendered as a wind movie (two maps, excitatory and inhibitory; under them the runs' mean learned dF/F).

AGREEMENT: for each run, the cosine similarity of its time-averaged field with the mean of the OTHER runs (inside the
brain; 1 = the same flow everywhere, 0 = unrelated). A structure the data imposes comes back in every graph.

    PYTHONPATH=src:tools python tools/exp17_wind_consensus.py --tag g17 zap_e15_cur zap_g17_rot45 ...
Writes experiments/exp17_zapbench_graphcast/data/wind_consensus_<tag>/results/movie_wind_{mean,median}.mp4,
data/wind_consensus_<tag>.json, and copies of the movies and a still in png/.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")


def cosine(a, b, ins):
    """Cosine similarity of two vector fields [2, ny, nx] over the cells inside the brain."""
    a_, b_ = a[:, ins].reshape(-1), b[:, ins].reshape(-1)
    return float(a_ @ b_ / max(np.linalg.norm(a_) * np.linalg.norm(b_), 1e-12))


def similarity_figure(doc, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = [r.replace("zap_", "").replace("g17_", "") for r in doc["matrix_runs"]]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, k in zip(axs, ("ex", "in")):
        M = np.array(doc["similarity"][k]["movie"])
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(lab)))
        ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(lab)))
        ax.set_yticklabels(lab, fontsize=8)
        for i in range(len(lab)):
            for j in range(len(lab)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] < 0.6 else "black")
        ax.set_title(f"{'excitatory' if k == 'ex' else 'inhibitory'} flow: similarity of the movies", fontsize=10, loc="left")
    fig.colorbar(im, ax=axs, fraction=0.025)
    path = os.path.join(EXP, "presentation", "figs", f"wind_similarity_{tag}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))


def main():
    import exp17_wind as W
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tag", default="g17")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ref", nargs="*", default=[], help="runs in the similarity matrix only, not in the consensus "
                                                          "(e.g. the base graph's seed 1: two trainings, one graph)")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--median-on-recorded", action="store_true",
                    help="render only the MEDIAN flow, over the RECORDED dF/F (gridded, inferno) instead of the learned")
    a = ap.parse_args()
    ex, inh, acts, preds, D0 = [], [], [], [], None
    allr = list(a.runs) + [r for r in a.ref if r not in a.runs]
    for r in allr:
        D = W.fields(r, 160, a.device)
        s = max(np.percentile(D["speed"]["ex"][:, D["inside"]], 99), np.percentile(D["speed"]["in"][:, D["inside"]], 99))
        ex.append(D["wind"]["ex"] / s)
        inh.append(D["wind"]["in"] / s)
        acts.append(D["act"])
        preds.append(D["pred"].mean(1))
        if D0 is None:
            D0 = D
        elif not np.array_equal(D["fr"], D0["fr"]) or D["inside"].shape != D0["inside"].shape:
            raise SystemExit(f"{r}: not the same movie frames or grid as {a.runs[0]}")
        del D
    ex, inh = np.stack(ex), np.stack(inh)                              # [runs (+ refs), F, 2, ny, nx]
    ins = D0["inside"]
    # THE SIMILARITY OF THE FLOW MOVIES, pair by pair (Cedric, 2026-10-03): per frame, the cosine similarity of the two
    # runs' vector fields over the cells inside the brain, averaged over the frames ("movie"); and of their
    # time-averaged fields ("time mean")
    sim = {}
    for name_, arr in (("ex", ex), ("in", inh)):
        U = arr[:, :, :, ins].reshape(arr.shape[0], arr.shape[1], -1)  # [R, F, 2 * cells]
        U = U / np.maximum(np.linalg.norm(U, axis=2, keepdims=True), 1e-12)
        movie = np.einsum("afd,bfd->ab", U, U) / U.shape[1]
        T_ = arr.mean(1)[:, :, ins].reshape(arr.shape[0], -1)
        T_ = T_ / np.maximum(np.linalg.norm(T_, axis=1, keepdims=True), 1e-12)
        sim[name_] = {"movie": movie.tolist(), "time_mean": (T_ @ T_.T).tolist()}
    nr = len(a.runs)
    off = lambda M: float(np.mean([M[i][j] for i in range(nr) for j in range(nr) if i != j]))
    summary = {k_: {"movie_mean_offdiag": off(v_["movie"]), "time_mean_offdiag": off(v_["time_mean"])} for k_, v_ in sim.items()}
    if a.ref:                                                            # the same graph, another training
        for k_, v_ in sim.items():
            i0, j0 = 0, allr.index(a.ref[0])
            summary[k_]["seed_ref_movie"] = float(v_["movie"][i0][j0])
            summary[k_]["seed_ref_time_mean"] = float(v_["time_mean"][i0][j0])
    ex, inh = ex[:nr], inh[:nr]
    agree = {}
    for k, (name, arr) in enumerate((("ex", ex), ("in", inh))):
        tm = arr.mean(1)                                               # [runs, 2, ny, nx]
        for i, r in enumerate(a.runs):
            others = np.delete(tm, i, 0).mean(0)
            agree.setdefault(r, {})[name] = cosine(tm[i], others, ins)
    out = os.path.join(EXP, "data", f"wind_consensus_{a.tag}")
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    doc = {"runs": a.runs, "matrix_runs": allr, "similarity": sim, "similarity_summary": summary,
           "agreement_with_the_others": agree,
           "mean_agreement": {k: float(np.mean([agree[r][k] for r in a.runs])) for k in ("ex", "in")}}
    if not a.median_on_recorded:                       # that mode renders only; it keeps the similarity file as it is
        json.dump(doc, open(os.path.join(EXP, "data", f"wind_consensus_{a.tag}.json"), "w"), indent=1)
    print(json.dumps({"similarity_summary": summary, "agreement_with_the_others": agree}, indent=1))
    similarity_figure(doc, a.tag)
    if a.no_render:
        return
    base = dict(act=np.mean(acts, 0), inside=ins, grid=D0["grid"], fr=D0["fr"], rec=D0["rec"], out=out,
                pred=np.stack(preds, 0).mean(0)[:, None] * np.ones((1, 1), np.float32), P=D0["P"])
    if a.median_on_recorded:                       # Cedric, 2026-10-03: the median flow on top of the recorded activity
        from scipy.ndimage import gaussian_filter
        from scipy.sparse import csr_matrix
        x0, y0, h = D0["grid"]
        ny, nx = ins.shape
        Pv = D0["P"]
        cid = (np.clip(((Pv[:, 1] - y0) / h).astype(int), 0, ny - 1) * nx + np.clip(((Pv[:, 0] - x0) / h).astype(int), 0, nx - 1))
        A = csr_matrix((np.ones(len(cid)), (cid, np.arange(len(cid)))), shape=(ny * nx, len(cid)))
        cnt = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
        Xr = D0["rec"]["dff"][D0["fr"]].astype(np.float64)                 # [F, N] the recorded frames
        rec_act = (A @ Xr.T).T / cnt
        rec_act = np.stack([gaussian_filter(r_.reshape(ny, nx), W.SIGMA_UM / h) for r_ in rec_act]).astype(np.float32)
        w = {"ex": np.median(ex, 0).astype(np.float32), "in": np.median(inh, 0).astype(np.float32)}
        Dk = dict(base, act=rec_act, bg_style="activity", wind=w, speed={k: np.linalg.norm(v, axis=1) for k, v in w.items()})
        W.render(f"consensus_median_rec_{a.tag}", Dk)
        shutil.move(os.path.join(out, "results", "movie_wind.mp4"), os.path.join(out, "results", "movie_wind_median_rec.mp4"))
        print(f"[consensus] median on the recorded dF/F: {os.path.join(out, 'results', 'movie_wind_median_rec.mp4')}")
        return
    for kind, f in (("mean", lambda x: x.mean(0)), ("median", lambda x: np.median(x, 0))):
        w = {"ex": f(ex).astype(np.float32), "in": f(inh).astype(np.float32)}
        Dk = dict(base, wind=w, speed={k: np.linalg.norm(v, axis=1) for k, v in w.items()})
        nm = f"consensus_{kind}_{a.tag}"
        W.render(nm, Dk)
        src = os.path.join(out, "results", "movie_wind.mp4")
        shutil.move(src, os.path.join(out, "results", f"movie_wind_{kind}.mp4"))
        print(f"[consensus] {kind}: {os.path.join(out, 'results', f'movie_wind_{kind}.mp4')}")


if __name__ == "__main__":
    main()
