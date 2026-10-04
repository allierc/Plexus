"""exp17: is there PROPAGATING activity in ZAPBench at ~1 Hz? (Cedric, 2026-09-30: measure before adding waves to the law.)

Local analysis of the recording only (the cluster rule: no analysis job goes to the cluster). Writes
experiments/exp17_zapbench_graphcast/data/waves.json and data/figs/waves.png.

THE MEASURE. The 71,721 neurons are averaged into cubic voxels (`--voxel` um, a voxel kept when it holds >= 10 neurons),
each voxel's trace z-scored within each stimulus condition (no lag is read across a condition boundary). For every
pair of neighbouring voxels a, b:

    c(+1) = corr(a(t), b(t+1)),  c(0),  c(-1) = corr(a(t+1), b(t))
    asymmetry  A_ab = c(+1) - c(-1)                     > 0: a leads b
    lag        l_ab = (c(+1) - c(-1)) / (2 (2 c(0) - c(+1) - c(-1)))   frames, the parabola's peak (b after a)

A voxel's FLOW is the mean of A_ab times the unit vector a -> b over its in-plane (x-y) neighbours: a field that
points the way activity travels. A travelling wave makes that field the SAME in two independent halves of the
recording; noise does not. The test is the split-half vector correlation of the field (first vs second half of every
condition) against a null of the same fields with the voxels shuffled.

Twice: on the recording, and on its residual after the stimulus-evoked response (the leave-one-out stimulus lookup)
is removed -- a sequence of stimulus responses with different latencies is not a wave.

THE ACQUISITION CAVEAT. A light-sheet volume is scanned plane by plane in z over its 0.914 s, so neighbours in z are
sampled at different times and show a lag whatever the brain does. The flow is therefore read in x-y only, and the
lag between z-neighbours is reported on its own: a lag of one sign everywhere along z is the scan, not the brain.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FRAME_S = 0.914


def voxelise(P, L, min_n=10):
    """Voxel index per neuron (-1 if its voxel is too sparse), the kept voxels' integer coordinates and centres."""
    q = np.floor((P - P.min(0)) / L).astype(np.int64)
    keys, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    keep = cnt >= min_n
    newid = -np.ones(len(keys), np.int64)
    newid[keep] = np.arange(keep.sum())
    vid = newid[inv.reshape(-1)]
    cen = np.stack([P[vid == v].mean(0) for v in range(keep.sum())])
    return vid, keys[keep], cen


def voxel_traces(X, vid, nv):
    """[T, nv] mean trace per voxel."""
    m = vid >= 0
    idx = torch.as_tensor(vid[m], device=X.device)
    Xm = X[:, torch.as_tensor(np.where(m)[0], device=X.device)]
    s = torch.zeros(X.shape[0], nv, device=X.device, dtype=X.dtype).index_add_(1, idx, Xm)
    c = torch.zeros(nv, device=X.device, dtype=X.dtype).index_add_(0, idx, torch.ones_like(idx, dtype=X.dtype))
    return s / c


def segments(cond):
    """[(t0, t1)] of each condition's frames (contiguous runs)."""
    cut = np.where(np.diff(cond) != 0)[0] + 1
    b = np.concatenate([[0], cut, [len(cond)]])
    return list(zip(b[:-1], b[1:]))


def lagged(Y, segs, pairs):
    """c(-1), c(0), c(+1) per pair over the given segments, each segment z-scored on its own."""
    a = torch.as_tensor(pairs[:, 0], device=Y.device)
    b = torch.as_tensor(pairs[:, 1], device=Y.device)
    acc = torch.zeros(3, len(pairs), device=Y.device, dtype=torch.float64)
    n = 0
    for t0, t1 in segs:
        if t1 - t0 < 4:
            continue
        Z = Y[t0:t1]
        Z = (Z - Z.mean(0)) / Z.std(0).clamp(min=1e-9)
        A, B = Z[:, a].double(), Z[:, b].double()
        acc[0] += (A[1:] * B[:-1]).sum(0)          # c(-1): b leads a
        acc[1] += (A * B).sum(0) * (len(Z) - 1) / len(Z)
        acc[2] += (A[:-1] * B[1:]).sum(0)          # c(+1): a leads b
        n += len(Z) - 1
    return (acc / n).cpu().numpy()


def pair_stats(C):
    cm, c0, cp = C
    asym = cp - cm
    den = 2 * (2 * c0 - cp - cm)
    lag = np.where(np.abs(den) > 1e-9, (cp - cm) / np.where(np.abs(den) > 1e-9, den, 1), np.nan)
    return asym, np.clip(lag, -1.5, 1.5)


def flow(asym, pairs, cen, nv):
    """[nv, 2] the mean of asym * unit(a -> b) in x-y over each voxel's pairs (both ends: b gets -asym toward a)."""
    d = cen[pairs[:, 1], :2] - cen[pairs[:, 0], :2]
    u = d / np.linalg.norm(d, axis=1, keepdims=True).clip(1e-9)
    F = np.zeros((nv, 2))
    cnt = np.zeros(nv)
    np.add.at(F, pairs[:, 0], asym[:, None] * u)
    np.add.at(F, pairs[:, 1], asym[:, None] * u)        # b's flow: activity arriving from a also points a -> b
    np.add.at(cnt, pairs[:, 0], 1)
    np.add.at(cnt, pairs[:, 1], 1)
    return F / cnt.clip(1)[:, None]


def vcorr(F1, F2):
    return float((F1 * F2).sum() / np.sqrt((F1 ** 2).sum() * (F2 ** 2).sum()))


def analyse(Y, segs, pairs_xy, pairs_z, cen, nv, rng, n_null=200):
    h1 = [(t0, (t0 + t1) // 2) for t0, t1 in segs]
    h2 = [((t0 + t1) // 2, t1) for t0, t1 in segs]
    asym, lag = pair_stats(lagged(Y, segs, pairs_xy))
    F = flow(asym, pairs_xy, cen, nv)
    F1 = flow(pair_stats(lagged(Y, h1, pairs_xy))[0], pairs_xy, cen, nv)
    F2 = flow(pair_stats(lagged(Y, h2, pairs_xy))[0], pairs_xy, cen, nv)
    r = vcorr(F1, F2)
    null = np.array([vcorr(F1, F2[rng.permutation(nv)]) for _ in range(n_null)])
    az, lz = pair_stats(lagged(Y, segs, pairs_z))
    return {"flow": F, "lag_xy": lag, "asym_xy": asym, "lag_z": lz,
            "split_half_r": r, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_p95": float(np.percentile(null, 95)),
            "median_abs_lag_xy_frames": float(np.nanmedian(np.abs(lag))),
            "mean_lag_z_frames": float(np.nanmean(lz)), "frac_lag_z_positive": float(np.nanmean(lz > 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=32.0)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    from plexus.tasks import trace_recording as TR
    rec = TR.load("zapbench")
    X = torch.as_tensor(rec["dff"], device=a.device)
    P = rec["pos_um"].astype(np.float64)
    vid, ijk, cen = voxelise(P, a.voxel)
    nv = len(cen)
    lut = {tuple(k): i for i, k in enumerate(ijk)}
    pxy, pz = [], []
    for i, (x, y, z) in enumerate(ijk):
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):        # each in-plane pair once
            j = lut.get((x + dx, y + dy, z))
            if j is not None:
                pxy.append((i, j))
        j = lut.get((x, y, z + 1))                               # z-neighbour, b deeper than a
        if j is not None:
            pz.append((i, j))
    pxy, pz = np.array(pxy), np.array(pz)
    segs = segments(rec["condition"])
    rng = np.random.default_rng(0)
    Yraw = voxel_traces(X, vid, nv)
    kid = TR.stimulus_keys(rec["stimulus"], rec["condition"])
    Yres = voxel_traces(X - TR.stimulus_lookup(X, kid), vid, nv)
    out = {"voxel_um": a.voxel, "voxels": int(nv), "pairs_xy": int(len(pxy)), "pairs_z": int(len(pz)),
           "frame_s": FRAME_S, "scan_lag_expected_frames_per_z_voxel": a.voxel / float(np.ptp(P[:, 2]))}
    res = {}
    for name, Y in (("raw", Yraw), ("residual", Yres)):
        r = analyse(Y, segs, pxy, pz, cen, nv, rng)
        res[name] = r
        out[name] = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        sp = a.voxel * np.sqrt(1.5) / np.maximum(np.abs(r["lag_xy"]) * FRAME_S, 1e-9)   # mean in-plane neighbour distance
        out[name]["median_speed_um_per_s"] = float(np.nanmedian(sp))
        print(f"[waves] {name:8s} split-half r {r['split_half_r']:+.3f} (null {r['null_mean']:+.3f} +- {r['null_sd']:.3f}, "
              f"p95 {r['null_p95']:+.3f}); median |lag| x-y {r['median_abs_lag_xy_frames']:.3f} frames; "
              f"z-lag mean {r['mean_lag_z_frames']:+.3f} frames, {100 * r['frac_lag_z_positive']:.0f} % positive")
    os.makedirs(os.path.join(EXP, "data", "figs"), exist_ok=True)
    json.dump(out, open(os.path.join(EXP, "data", "waves.json"), "w"), indent=1)
    figure(res, cen, P, os.path.join(EXP, "data", "figs", "waves.png"), out)
    print(json.dumps(out, indent=1))


def figure(res, cen, P, path, out):
    """Black, labels above. a, b: the in-plane flow field (dorsal view, voxels collapsed over z), raw and residual,
    arrows where it is consistent between halves; c: the neighbour lags; d: the z-lag (the scan check)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 6.2), facecolor="black")

    def style(ax):
        ax.set_facecolor("black")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("0.6")
        ax.tick_params(colors="0.8", labelsize=8)

    for j, name in enumerate(("raw", "residual")):
        ax = fig.add_axes([0.01 + 0.25 * j, 0.06, 0.23, 0.78])
        ax.set_facecolor("black")
        ax.axis("off")
        ax.scatter(P[::7, 0], P[::7, 1], s=0.05, c="0.3", linewidths=0)
        F = res[name]["flow"]
        mag = np.linalg.norm(F, axis=1)
        ax.quiver(cen[:, 0], cen[:, 1], F[:, 0], F[:, 1], mag, cmap="plasma", scale=np.nanpercentile(mag, 95) * 25,
                  width=0.004)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        r = res[name]
        fig.text(0.01 + 0.25 * j, 0.97, f"{'ab'[j]}   flow, {'the recording' if j == 0 else 'stimulus response removed'}",
                 color="white", fontsize=11, va="top")
        fig.text(0.01 + 0.25 * j, 0.915, f"split-half r {r['split_half_r']:+.2f}  (shuffled {r['null_mean']:+.2f} "
                 f"+- {r['null_sd']:.2f})", color="0.75", fontsize=9, va="top")
    c = fig.add_axes([0.56, 0.12, 0.19, 0.68])
    style(c)
    bins = np.linspace(-1.5, 1.5, 61)
    for name, col in (("raw", "#9ecae1"), ("residual", "#fd8d3c")):
        c.hist(res[name]["lag_xy"] * FRAME_S, bins * FRAME_S, histtype="step", color=col, lw=1.4, label=name)
    c.set_xlabel("lag between in-plane neighbours, s", color="0.8", fontsize=9)
    c.legend(frameon=False, labelcolor="white", fontsize=8)
    fig.text(0.56, 0.97, "c   neighbour lags (x-y)", color="white", fontsize=11, va="top")
    d = fig.add_axes([0.80, 0.12, 0.19, 0.68])
    style(d)
    for name, col in (("raw", "#9ecae1"), ("residual", "#fd8d3c")):
        d.hist(res[name]["lag_z"] * FRAME_S, bins * FRAME_S, histtype="step", color=col, lw=1.4, label=name)
    d.axvline(out["scan_lag_expected_frames_per_z_voxel"] * FRAME_S, color="white", ls=":", lw=1)
    d.set_xlabel("lag to the next voxel in z, s", color="0.8", fontsize=9)
    fig.text(0.80, 0.97, "d   z-neighbour lags (the scan)", color="white", fontsize=11, va="top")
    fig.text(0.80, 0.915, "dotted: one sweep of the volume over its depth", color="0.7", fontsize=8, va="top")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)


if __name__ == "__main__":
    main()
