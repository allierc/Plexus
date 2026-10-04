"""A recording of TRACES on a set -- one value per element per frame at fixed positions (exp17: ZAPBench, 71,721
neurons x 7,879 frames of dF/F) -- as the trainer's `trace_recording` reference reads it, and the arithmetic every
number of that reference is computed with: the model's forecasts and the baselines go through the same functions.

STANDARD MSE EVERYWHERE (Cedric, 2026-09-30). The headline is the SKILL OVER THE MEAN BASELINE,
skill_h = 1 - MSE_model(h) / MSE_mean(h), MSE_mean(h) the best of "each element's mean over its last W frames",
W = 1..6, on the grand average (ZAPBench eq. 10; tools/zapbench_baselines.py measured it before any training).
MSE per origin is the mean over elements; per condition the mean over origins whose origin frame lies in it; the
grand average the mean over conditions.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

H_MAX, W_MAX = 32, 6
SHORT, LONG = (1, 3), (16, 32)                      # inclusive step ranges of the two headline skills


def load(name: str = "zapbench") -> dict:
    """graphs_data/zebrafish/<name>_recording.npz (tools/export_zapbench_recording.py)."""
    from plexus.paths import graphs_data_path
    z = np.load(graphs_data_path("zebrafish", f"{name}_recording.npz"))
    rec = {k: z[k] for k in z.files}
    rec["names"] = [str(n) for n in rec["names"]]
    # THE DRAWING ORIENTATION (exp20, Cedric 2026-10-03: "consistent with exp17's views", head LEFT): render_movie lays a
    # long-x brain with its low-x end on the left; a recording whose provenance says its head is at +x (`"head": "+x"`,
    # the gut-brain fish) is drawn mirrored in x. Drawing only: the positions the law reads (pos_um) never change.
    rec["pos_view"] = rec["pos_um"]
    prov = graphs_data_path("zebrafish", f"{name}_recording.json")
    if os.path.exists(prov) and json.load(open(prov)).get("head") == "+x":
        rec["pos_view"] = rec["pos_um"] * np.array([-1.0, 1.0, 1.0])
    pub = graphs_data_path("zebrafish", f"{name}_published.json")
    if os.path.exists(pub):                        # the reference's published results, drawn beside every run
        rec["published"] = json.load(open(pub))
    return rec


def mean_pred(X: torch.Tensor, o: torch.Tensor, W: int) -> torch.Tensor:
    """[len(o), N]: each element's mean over frames o-W+1 .. o (W = 1 is persistence)."""
    return torch.stack([X[o - w] for w in range(W)], 0).mean(0)


def mse_per_origin(pred_fn, X: torch.Tensor, o: torch.Tensor, hs=range(1, H_MAX + 1)) -> np.ndarray:
    """[len(hs), len(o)]: the MSE over elements of `pred_fn(o, h)` against X[o + h]."""
    # ACCUMULATED IN FLOAT64: a float32 mean over 71,721 elements carries ~1e-7 of relative rounding, which two
    # identical forecasts summed in a different order already disagree by (the identity gate's own resolution).
    return np.stack([((X[o + h] - pred_fn(o, h)) ** 2).double().mean(1).cpu().numpy() for h in hs], 0)


N_VISUAL = 22             # the release's visual stimulus features: the first columns of every recording's `stimulus`


def stimulus_keys(S: np.ndarray, cond: np.ndarray, cap: int = 64) -> np.ndarray:
    """Per frame, the id of (condition, stimulus vector, frames since the stimulus last changed, capped). The VISUAL
    columns only (the first N_VISUAL): a recording with ephys columns appended (zapbench_destripe_ephys: swim and
    turn power, continuous) would otherwise give nearly every frame its own key and the lookup baseline no repeats."""
    S = np.asarray(S)[:, :N_VISUAL]
    keys, since = [], 0
    for t in range(len(S)):
        if t > 0 and (cond[t] != cond[t - 1] or not np.array_equal(S[t], S[t - 1])):
            since = 0
        elif t > 0:
            since = min(since + 1, cap)
        keys.append((int(cond[t]),) + tuple(np.round(S[t], 4)) + (since,))
    uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return np.array([uniq[k] for k in keys])


def stimulus_lookup(X: torch.Tensor, kid: np.ndarray) -> torch.Tensor:
    """[T, N]: ZAPBench's stimulus-evoked baseline (eq. 11) read as the response phase-locked to the stimulus --
    each element's mean over all OTHER frames with the same stimulus key (leave-one-out); a key seen once gives
    the element's mean."""
    k = torch.as_tensor(kid, device=X.device)
    nk = int(k.max()) + 1
    sums = torch.zeros(nk, X.shape[1], device=X.device, dtype=X.dtype).index_add_(0, k, X)
    cnt = torch.zeros(nk, device=X.device, dtype=X.dtype).index_add_(0, k, torch.ones(len(k), device=X.device,
                                                                                       dtype=X.dtype))
    pred = (sums[k] - X) / (cnt[k] - 1).clamp(min=1)[:, None]
    single = cnt[k] < 2
    pred[single] = X.mean(0)
    return pred


# ============================================================================== ZAPBench's own split
# zapbench/constants.py and its data sources: each condition is trimmed by PADDING_FRAMES = 1 at both ends; of the rest
# the first frames train, then int(0.1 n) validate, then the last int(0.2 n) test; condition 3 (TAXIS) is held out
# whole and scored from its MAX_CONTEXT_LENGTH = 256-th frame on. A test origin is the last context frame of a
# window whose H forecast frames all lie in the test block (Cedric, 2026-10-01: compare on ZAPBench's unseen frames).
ZB_PAD, ZB_VAL, ZB_TEST, ZB_HOLDOUT, ZB_MAXCTX = 1, 0.1, 0.2, 3, 256
SPLIT_LABEL = {"train": 0, "val": 1, "test": 2, "holdout": 3}


def zapbench_split(offsets: np.ndarray, T: int) -> np.ndarray:
    """[T] per frame: 0 train, 1 val, 2 test, 3 the held-out condition, -1 a pad frame."""
    lab = -np.ones(T, np.int64)
    for c in range(len(offsets) - 1):
        lo, hi = int(offsets[c]) + ZB_PAD, int(offsets[c + 1]) - ZB_PAD
        if c == ZB_HOLDOUT:
            lab[lo:hi] = 3
            continue
        n = hi - lo
        n_test, n_val = int(n * ZB_TEST), int(n * ZB_VAL)
        n_train = n - n_test - n_val
        lab[lo:lo + n_train] = 0
        lab[lo + n_train:lo + n_train + n_val] = 1
        lab[lo + n_train + n_val:hi] = 2
    return lab


def split_origins(lab: np.ndarray, cond: np.ndarray, part: str, n_in: int, H: int) -> np.ndarray:
    """Origins o (the last context frame) of the windows of one part. `train`: every frame o-n_in+1 .. o+H is a
    training frame (nothing of the window is ever scored). `val` / `test` / `holdout`: the H forecast frames lie in
    that part; the context frames are any non-pad frames of the same condition (ZAPBench: a test window's context
    reaches back into validation). `holdout` starts at the condition's 256-th frame, as ZAPBench's."""
    k = SPLIT_LABEL[part]
    T = len(lab)
    out = []
    for o in range(n_in - 1, T - H):
        if part == "train":
            ok = bool((lab[o - n_in + 1:o + H + 1] == 0).all())
        else:
            ok = bool((lab[o + 1:o + H + 1] == k).all()) and bool((lab[o - n_in + 1:o + 1] >= 0).all()) \
                and len(set(cond[o - n_in + 1:o + H + 1].tolist())) == 1
            if ok and part == "holdout":
                ok = o >= int(np.argmax(lab == 3)) - ZB_PAD + ZB_MAXCTX - 1 + ZB_PAD
        if ok:
            out.append(o)
    return np.array(out, np.int64)


def stimulus_lookup_fit(X: torch.Tensor, kid: np.ndarray, fit: np.ndarray) -> torch.Tensor:
    """[T, N]: the stimulus-evoked lookup LEARNED ON THE `fit` FRAMES ONLY (ZAPBench's stimulus baseline is built on
    its training split): each frame gets the mean over the fit frames with its key (leave-one-out where the frame is
    itself a fit frame); a key no fit frame has gets each element's mean over the fit frames."""
    k = torch.as_tensor(kid, device=X.device)
    f = torch.as_tensor(fit, device=X.device, dtype=X.dtype)
    nk = int(k.max()) + 1
    sums = torch.zeros(nk, X.shape[1], device=X.device, dtype=X.dtype).index_add_(0, k, X * f[:, None])
    cnt = torch.zeros(nk, device=X.device, dtype=X.dtype).index_add_(0, k, f)
    s_k, c_k = sums[k] - X * f[:, None], cnt[k] - f
    mu = (X * f[:, None]).sum(0) / f.sum().clamp(min=1)
    pred = s_k / c_k.clamp(min=1)[:, None]
    pred[c_k < 1] = mu
    return pred


def denoise(X: torch.Tensor) -> torch.Tensor:
    """THE DENOISED RECORDING, fixed before any model: each element's 3-frame mean (t-1, t, t+1)."""
    Xd = X.clone()
    Xd[1:-1] = (X[:-2] + X[1:-1] + X[2:]) / 3
    return Xd


def r2_frames(pred: torch.Tensor, X: torch.Tensor) -> np.ndarray:
    """R^2 per frame over elements: 1 - sum_i (pred - x)^2 / sum_i (x - mean_i x)^2."""
    num = ((pred - X) ** 2).sum(-1)
    den = ((X - X.mean(-1, keepdim=True)) ** 2).sum(-1)
    return (1 - num / den).detach().cpu().numpy()


def by_condition(per_origin: np.ndarray, oc: np.ndarray, n_cond: int) -> np.ndarray:
    """[..., n_origin] -> [n_cond, ...]: the mean over the origins of each condition."""
    return np.stack([per_origin[..., oc == c].mean(-1) for c in range(n_cond)], 0)


def skills(model_c: np.ndarray, mean_c: np.ndarray) -> dict:
    """From per-condition MSE [C, H] of the model and of the best mean baseline: the headline numbers."""
    ga_m, ga_b = model_c.mean(0), mean_c.mean(0)
    sk = 1 - ga_m / ga_b
    sk_c = 1 - model_c / mean_c
    s0, s1, l0, l1 = SHORT[0] - 1, SHORT[1], LONG[0] - 1, LONG[1]
    return {"skill": sk.tolist(), "skill_short": float(sk[s0:s1].mean()), "skill_long": float(sk[l0:l1].mean()),
            "skill_long_by_condition": sk_c[:, l0:l1].mean(1).tolist(),
            "skill_short_by_condition": sk_c[:, s0:s1].mean(1).tolist(),
            "n_conditions_long_positive": int((sk_c[:, l0:l1].mean(1) > 0).sum())}


# ============================================================================== figures
def render_curves(res: dict, path: str, names, gates: dict | None = None, history: list | None = None,
                  brain_mean: tuple | None = None):
    """THE PREDICTION AGAINST THE BASELINES, black (the deck and the watcher).
    a  MSE per step ahead: the grand average of the learned law (thick), of the best mean baseline, persistence, the
       stimulus lookup and the noise floor sigma^2, and the learned law in each condition (thin, one colour each);
    b  the training loss against the updates, one curriculum stage per horizon: a line where a stage begins and its
       horizon above it (Cedric, 2026-10-01: the loss per horizon in place of the skill panel);
    c  the free rollout of the whole recording: R^2 per frame, raw and denoised, the conditions as bands; or, with
       `brain_mean` = (t_min, recorded, learned), the brain-mean dF/F, recorded (green) and learned (white) -- the
       curves the brain-mean R2 is computed on (exp17, Cedric 2026-10-04: its main metric)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.arange(1, H_MAX + 1)
    plt.style.use("dark_background")               # black, as every figure of the deck and the watcher
    fig = plt.figure(figsize=(13, 7.5), facecolor="black")
    a = fig.add_axes([0.06, 0.55, 0.40, 0.38])
    b = fig.add_axes([0.56, 0.55, 0.40, 0.38])
    c = fig.add_axes([0.06, 0.08, 0.90, 0.34])
    cmap = plt.get_cmap("tab10")
    mc = np.asarray(res["mse_model_by_condition"])
    for ci, name in enumerate(names):
        a.plot(h, mc[ci], lw=0.8, color=cmap(ci), alpha=0.85, label=name)
    a.plot(h, res["mse_model"], color="white", lw=2.4, label="learned law, grand average")
    a.plot(h, res["mse_mean"], color="0.75", lw=1.6, ls="--", label="mean baseline (best W per step)")
    a.plot(h, res["mse_persistence"], color="0.55", lw=1.0, ls=":", label="persistence")
    a.plot(h, res["mse_lookup"], color="#c44e52", lw=1.4, label="stimulus-evoked lookup")
    pub = (gates or {}).get("published")
    if pub:                                        # ZAPBench's best model: its published skill over its own mean
        bc = pub["best_ctx4"]                      # baseline, applied to THIS run's mean baseline (MSE on these frames)
        zbc = (np.asarray(bc["mse_test"]) if res.get("split") == "zapbench" and "mse_test" in bc     # same windows:
               else (1 - np.asarray(bc["skill"])) * np.asarray(res["mse_mean"]))                    # its own MSE
        a.plot(h, zbc, color="#7aa6ff", lw=1.8, ls="-.",
               label=f"ZAPBench best ({bc['method']}, ctx 4)")
    a.axhline(res["noise_sigma2"], color="0.45", ls=":", lw=1.0, label="noise floor $\\sigma^2$")
    a.set_xlabel("steps ahead h (0.914 s each)")
    a.set_ylabel("MSE, dF/F$^2$")
    a.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
    a.text(0.0, 1.04, "a", transform=a.transAxes, fontsize=12)
    if history:
        it = np.array([r["it"] for r in history])
        lo = np.array([r["loss"] for r in history], dtype=float)
        hz = np.array([r.get("horizon", 0) for r in history])
        b.plot(it, lo, color="0.75", lw=0.3)
        k = max(1, len(lo) // 400)                 # a running median, so the trend reads through the batch noise
        if len(lo) > 3 * k:
            sm = np.array([np.median(lo[max(0, q - k):q + k + 1]) for q in range(len(lo))])
            b.plot(it, sm, color="white", lw=1.4)
        starts = np.where(np.diff(hz, prepend=hz[0] - 1) != 0)[0]
        ymax = np.nanpercentile(lo, 99.5)
        for q, st in enumerate(starts):
            b.axvline(it[st], color="0.35", lw=0.6)
            if hz[st] in (1, 2, 3, 5, 10, 15, 20, 25, 30, 40):   # a few horizons named, never crowded
                b.text(it[st], 1.01, f"{hz[st]}", transform=b.get_xaxis_transform(), fontsize=6.5, color="0.8",
                       ha="center")
        b.set_yscale("log")
        b.set_ylim(np.nanpercentile(lo, 0.5) * 0.9, ymax * 1.1)
        b.set_xlabel("training updates (Adam steps; one curriculum stage per horizon)")
        b.set_ylabel("training loss")
        b.text(1.0, 1.08, "horizon of the stage", transform=b.transAxes, fontsize=7, color="0.8", ha="right")
    else:
        b.text(0.5, 0.5, "no training history", transform=b.transAxes, ha="center", color="0.6")
    b.text(0.0, 1.04, "b", transform=b.transAxes, fontsize=12)
    t = np.asarray(res["free_t_s"]) / 60
    off = res["offsets"]
    for ci in range(len(names)):
        c.axvspan(off[ci] * res["frame_s"] / 60, off[ci + 1] * res["frame_s"] / 60,
                  color=cmap(ci), alpha=0.15, lw=0)
        c.text((off[ci] + off[ci + 1]) / 2 * res["frame_s"] / 60, 1.02, names[ci], fontsize=7, ha="center",
               transform=c.get_xaxis_transform())
    if brain_mean is not None:
        tb, bo, bp = (np.asarray(v, dtype=float) for v in brain_mean)
        c.plot(tb, bo, color="#2ca02c", lw=1.1, label="recorded")
        c.plot(tb, bp, color="white", lw=0.8, label="learned")
        ok = np.isfinite(bo) & np.isfinite(bp)
        lo_, hi_ = np.percentile(np.concatenate([bo[ok], bp[ok]]), [0.2, 99.8])
        c.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.15 * (hi_ - lo_))
        bad = np.zeros(0, bool)
    else:
        c.plot(t, res["free_r2_raw"], color="0.6", lw=0.6, label="R$^2$ raw")
        c.plot(t, res["free_r2_denoised"], color="#4c72b0", lw=0.8, label="R$^2$ denoised")
        c.axhline(0, color="0.6", lw=0.7)
        r2d = np.asarray(res["free_r2_denoised"], dtype=float)
        c.set_ylim(0, 1)                           # every R2 axis 0..1 (Cedric, 2026-10-02): runs compare at a glance
        bad = ~np.isfinite(r2d) | (r2d < -1)
    if bad.any():                                  # where the rollout leaves the axis for good, said in words
        c.text(0.99, 0.04, f"diverges from t = {t[int(np.argmax(bad))]:.0f} min (R$^2$ < -1)", color="#d62728",
               fontsize=8, ha="right", transform=c.transAxes)
    sil = np.asarray(res.get("free_silenced") or [0])
    if sil[-1]:                                    # top right, above the traces: runaway neurons frozen and left out of R^2 (finding 21)
        c.text(0.99, 0.93, f"{int(sil[-1]):,} exploding neuron{'s' if sil[-1] != 1 else ''} silenced, the first at "
               f"t = {t[int(np.argmax(sil > 0))]:.0f} min", color="#ff7f0e", fontsize=8, ha="right", transform=c.transAxes)
    c.set_xlabel("time, min (free rollout from the law's own context frames)")
    c.set_ylabel("brain-mean dF/F" if brain_mean is not None else "R$^2$ per frame over neurons")
    c.legend(frameon=False, fontsize=8, loc="lower left")
    c.text(0.0, 1.08, "c", transform=c.transAxes, fontsize=12)
    for ax in (a, b, c):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    plt.style.use("default")


def _pca_rgb(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = a - a.mean(0)
    u, s, vt = np.linalg.svd(a[:: max(1, len(a) // 20000)], full_matrices=False)
    pc = a @ vt[:3].T
    lo, hi = np.percentile(pc, 2, 0), np.percentile(pc, 98, 0)
    rgb = np.clip((pc - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
    # an embedding narrower than 3 gives fewer than 3 channels (a 2-wide one crashed every zap_cn plot, 2026-09-30):
    # the missing channels are held at mid-grey
    return pc, np.concatenate([rgb, np.full((len(rgb), 3 - rgb.shape[1]), 0.5)], 1)


def _ffmpeg() -> str:
    """An ffmpeg that exists HERE: on the PATH, else imageio-ffmpeg's own, else the devcontainer's (a cluster job has no
    /workspace, and a hard-coded devcontainer path crashed every exp17 plot phase there, 2026-09-30)."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                                     # noqa: BLE001
        return "/workspace/.conda_envs/MPM-pytorch/bin/ffmpeg"


INPUT_ON = 0.01          # |B_i| above which a neuron counts as a stimulus input (normalised activity per unit feature)


_MOVIE: dict = {}          # the movie's data, set before the frame workers fork (they read it, never copy it in)


def render_movie(obs: np.ndarray, pred: np.ndarray, frames: np.ndarray, pos: np.ndarray, r2_raw: np.ndarray,
                 r2_den: np.ndarray, frame_s: float, cond_of: np.ndarray, names, path: str,
                 emb: np.ndarray | None = None, labels: np.ndarray | None = None, fps: int = 25,
                 ffmpeg: str | None = None, emb_name: str = "embedding", law: str = "GraphCast law",
                 inputs: np.ndarray | None = None, rec_name: str = "ZAPBench", r2_t: np.ndarray | None = None,
                 r2_raw_all: np.ndarray | None = None, r2_den_all: np.ndarray | None = None, workers: int = 16,
                 silenced_all: np.ndarray | None = None, mean_obs_all: np.ndarray | None = None,
                 mean_pred_all: np.ndarray | None = None, cond_all: np.ndarray | None = None,
                 split_all: np.ndarray | None = None, split_name: str = "all", mean_obs=None, mean_pred=None,
                 metric: str = "cells"):
    """TWO PANELS, recorded LEFT and learned RIGHT: every neuron a point at its position (dorsal view, deeper
    drawn first, the brain vertical and head up) coloured by dF/F on one scale; black background, labels above.
    `frames` sample the FREE ROLLOUT of the whole recording (Cedric, 2026-10-02: the full 2 h, ~800 movie frames, not
    a 200-frame restart). Under the learned panel the free rollout's R2 per frame over the whole recording (`r2_t` the
    frames, `r2_*_all` the values; denoised white, raw grey, axis 0..1) with a cursor; top right its mean +- SD up to
    the cursor, or the time it diverged. A strip of insets below when the law has neuron constants or an embedding:
    PC1-PC2 by cluster, PCA as RGB on the brain, the clusters on the brain, the stimulus input map.
    `metric` "brain_mean" (exp20, Cedric 2026-10-04: "the main metric is the brain-mean dF/F R2"; needs mean_obs_all and
    mean_pred_all): top right the brain-mean R2 and RMSE of the free rollout up to the cursor, and under the learned
    panel the brain-mean R2 in a 5-min window centred on each frame, in place of the per-cell R2 and its divergence."""
    import os
    import shutil
    import subprocess
    import tempfile
    import multiprocessing as mp
    ffmpeg = ffmpeg or _ffmpeg()
    if np.ptp(pos[:, 0]) > np.ptp(pos[:, 1]):
        # head up first (zap-inr's anatomy frame has its long axis along x: drawn as (y, -x), the destriped slides'
        # mapping; ZAPBench's long axis is already along y, head up) ...
        pos = np.stack([pos[:, 1], -pos[:, 0], pos[:, 2]], 1)
    # ... then THE BRAIN HORIZONTAL, HEAD LEFT (Cedric, 2026-10-02: a vertical brain left most of each panel blank)
    pos = np.stack([-pos[:, 1], pos[:, 0], pos[:, 2]], 1)
    order = np.argsort(pos[:, 2])
    if r2_t is None:                                # no whole-rollout trace: the sampled frames' own R2
        r2_t, r2_raw_all, r2_den_all = frames, r2_raw, r2_den
    tmp = tempfile.mkdtemp(prefix="trace_movie_")
    _MOVIE.clear()
    _MOVIE.update(dict(obs=obs, pred=pred, frames=np.asarray(frames), P=pos[order], order=order,
                       vmax=float(np.percentile(obs, 97)), frame_s=frame_s, cond_of=cond_of, names=list(names),
                       emb=emb, labels=labels, emb_name=emb_name, law=law, inputs=inputs, rec_name=rec_name,
                       r2_t=np.asarray(r2_t), r2r=np.asarray(r2_raw_all, float), r2d=np.asarray(r2_den_all, float),
                       sil=None if silenced_all is None else np.asarray(silenced_all),
                       mo=mean_obs_all, mp_=mean_pred_all, cond=cond_all, split=split_all, split_name=split_name,
                       metric=metric if mean_obs_all is not None else "cells", tmp=tmp))
    ks = np.arange(len(frames))
    workers = min(workers, len(os.sched_getaffinity(0)))          # a cluster job's slots, not the node's cores
    chunks = [c for c in np.array_split(ks, max(1, min(workers, len(ks)))) if len(c)]
    if len(chunks) > 1:
        with mp.get_context("fork").Pool(len(chunks)) as pool:   # matplotlib only in the workers, no CUDA
            pool.map(_movie_frames, chunks)
    else:
        _movie_frames(chunks[0])
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", path],
                   check=True)
    shutil.copy(os.path.join(tmp, f"{len(frames) // 2:05d}.png"), path.replace(".mp4", ".png"))
    shutil.rmtree(tmp)
    _MOVIE.clear()


def _movie_frames(ks):
    """One worker: the figure built ONCE (the insets and the R2 traces are static), then per frame only the two
    panels' colours, the cursor and the texts change before each save."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _MOVIE
    P, order, vmax, frames = d["P"], d["order"], d["vmax"], d["frames"]
    emb, labels, inputs = d["emb"], d["labels"], d["inputs"]
    ins = emb is not None
    # THE INPUT MAP (Cedric, 2026-09-30): which neurons the stimulus enters -- |B_i|, the norm of each neuron's row of
    # stimulus weights, bright where it is large; a row the lasso zeroed is dark. A law with no per-neuron stimulus
    # weights (GraphCast, connectome: u enters every element's update) draws every neuron as an input, uniformly.
    if inputs is not None:
        nrm = np.asarray(inputs, dtype=np.float64)
        on = nrm > INPUT_ON
        lv = np.clip(np.log10(np.maximum(nrm, 1e-4)) + 4, 0, None) / max(np.log10(nrm.max() + 1e-12) + 4, 1e-9)
        inp_col = plt.get_cmap("magma")(0.08 + 0.92 * lv)[:, :3]
        inp_title = (f"stimulus input |B_i| per neuron (log)\n{on.sum():,} of {len(nrm):,} above {INPUT_ON:g} "
                     f"({100 * on.mean():.0f} %)")
    else:
        inp_col = np.tile(np.array([[0.93, 0.55, 0.25]]), (len(P), 1))
        inp_title = "stimulus input: every neuron\n(no per-neuron input weights)"
    fig = plt.figure(figsize=(12, 8.6 if ins else 7.0), facecolor="black")
    top = 0.40 if ins else 0.16
    sc = []
    for j, lab in enumerate((f"recorded ({d['rec_name']})", f"learned ({d['law']})")):
        ax = fig.add_axes([0.5 * j, top, 0.5, 0.88 - top])           # the labels sit above, never on the brain
        ax.set_facecolor("black")
        ax.axis("off")
        sc.append(ax.scatter(P[:, 0], P[:, 1], c=np.zeros(len(P)), s=0.5, cmap="inferno", vmin=0, vmax=vmax,
                             linewidths=0))
        ax.set_aspect("equal")
        fig.text(0.5 * j + 0.03, 0.985, lab, color="white", fontsize=12, va="top")
    t_txt = fig.text(0.03, 0.945, "", color="0.7", fontsize=9, va="top")
    r_txt = fig.text(0.97, 0.955, "", color="white", fontsize=9, va="top", ha="right")
    tm = d["r2_t"] * d["frame_s"] / 60

    def strip(x0):
        m_ = fig.add_axes([x0, top - 0.115, 0.42, 0.085])
        m_.set_facecolor("black")
        for k_, sp in m_.spines.items():
            sp.set_visible(k_ in ("left", "bottom"))
            sp.set_color("0.5")
        m_.set_xlim(tm[0], tm[-1])
        m_.tick_params(colors="0.6", labelsize=7, length=2)
        return m_

    def runs(lab):
        """[(first, last, value)] the consecutive blocks of a per-frame label."""
        lab = np.asarray(lab)
        cut = np.flatnonzero(np.diff(lab)) + 1
        st = np.r_[0, cut]
        en = np.r_[cut, len(lab)] - 1
        return [(int(a_), int(b_), lab[a_]) for a_, b_ in zip(st, en)]
    # LEFT: THE BRAIN-MEAN dF/F over the whole free rollout, recorded (green) and learned (white), the stimulus
    # CONDITIONS as alternating blocks with their names (Cedric, 2026-10-02: which blocks the law holds flat)
    cur = []
    if d["mo"] is not None:
        m = strip(0.05)
        for i_, (a_, b_, c_) in enumerate(runs(d["cond"])):
            m.axvspan(tm[a_], tm[b_], color=("0.30" if i_ % 2 else "0.18"), alpha=0.6, lw=0, zorder=0)
            m.text((tm[a_] + tm[b_]) / 2, 1.02, d["names"][int(c_)], color="0.75", fontsize=6, ha="center", va="bottom",
                   transform=m.get_xaxis_transform())
        m.plot(tm, d["mo"], color="#2ca02c", lw=0.6, zorder=2)
        m.plot(tm, d["mp_"], color="white", lw=0.6, zorder=3)
        lo_, hi_ = np.nanpercentile(np.r_[d["mo"], d["mp_"]], [0.5, 99.5])
        m.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.1 * (hi_ - lo_))
        m.set_yticks([])
        cur.append(m.axvline(tm[0], color="#ff7f0e", lw=0.9, zorder=4))
        fig.text(0.05, top - 0.005, "brain-mean dF/F: recorded (green), learned (white); time, min", color="0.7",
                 fontsize=8, va="bottom")
    # RIGHT: THE FREE ROLLOUT'S R2 per frame, 0..1, the split's parts behind it in transparent colours
    m = strip(0.55)
    SPLIT = {0: ("train", "#1f77b4"), 1: ("val", "#ffbf00"), 2: ("test", "#d62728"), 3: ("held-out condition", "#9467bd")}
    if d["split_name"] == "recording":              # exp20: the recording's own split, its test part = held-out trials
        SPLIT[2] = ("held-out trials", SPLIT[2][1])
    seen = []
    if d["split"] is not None and d["split_name"] in ("zapbench", "recording"):
        for a_, b_, v_ in runs(d["split"]):
            if int(v_) in SPLIT:
                m.axvspan(tm[a_], tm[b_], color=SPLIT[int(v_)][1], alpha=0.22, lw=0, zorder=0)
                seen.append(int(v_))
    bm = d["metric"] == "brain_mean"
    if bm:                                          # the brain-mean R2 in a 5-min window centred on each frame
        o_, p_ = np.asarray(d["mo"], np.float64), np.asarray(d["mp_"], np.float64)
        h_ = max(2, int(round(150.0 / d["frame_s"])))
        r2w = np.full(len(o_), np.nan)
        for i_ in range(len(o_)):
            a_, b_ = max(0, i_ - h_), min(len(o_), i_ + h_ + 1)
            den = ((o_[a_:b_] - o_[a_:b_].mean()) ** 2).sum()
            if den > 0:
                r2w[i_] = 1 - ((p_[a_:b_] - o_[a_:b_]) ** 2).sum() / den
        m.plot(tm, np.clip(r2w, -1, 1), color="white", lw=0.8, zorder=3)
        m.axhline(0, color="0.5", lw=0.5, zorder=1)
        m.set_ylim(-1, 1)
        m.set_yticks([-1, 0, 1])
    else:
        m.plot(tm, d["r2r"], color="0.55", lw=0.6, zorder=2)
        m.plot(tm, d["r2d"], color="white", lw=0.8, zorder=3)
        m.set_ylim(0, 1)
        m.set_yticks([0, 1])
    cur.append(m.axvline(tm[0], color="#ff7f0e", lw=0.9, zorder=4))
    split_txt = ("; behind: " + ", ".join(f"{SPLIT[v][0]}" for v in sorted(set(seen)))) if seen else \
        ("; every frame trained (no split)" if d["split_name"] not in ("zapbench", "recording") else "")
    fig.text(0.55, top - 0.005, ("brain-mean R$^2$ in a 5-min window (white)" if bm else
                                 "free rollout R$^2$: denoised (white), raw (grey)") + split_txt, color="0.7",
             fontsize=8, va="bottom")
    if seen:                                        # the split's colours, named in their own colour, left to right
        for i_, v in enumerate(sorted(set(seen))):
            fig.text(0.55 + 0.11 * i_, top - 0.14, SPLIT[v][0], color=SPLIT[v][1], fontsize=7, ha="left", va="top")
    if ins:
        pc, rgb = _pca_rgb(emb)
        cm = plt.get_cmap("tab10")
        lab_rgb = cm(labels % 10)[:, :3] if labels is not None else rgb
        for j, (title, kind) in enumerate(((f"{d['emb_name']} PC1-PC2, by cluster", "scatter"),
                                           (f"{d['emb_name']} on the brain (PCA as RGB)", "rgb"),
                                           ("its clusters on the brain", "labels"),
                                           (inp_title, "inputs"))):
            ax = fig.add_axes([0.01 + 0.25 * j, 0.01, 0.23, 0.19])       # below its title, never under it
            ax.set_facecolor("black")
            ax.axis("off")
            if kind == "scatter":
                ax.scatter(pc[::7, 0], pc[::7, 1], s=0.4, c=lab_rgb[::7], linewidths=0)
            elif kind == "inputs":
                ax.scatter(P[:, 0], P[:, 1], s=0.2, c=inp_col[order], linewidths=0)
                ax.set_aspect("equal")
            else:
                col = (rgb if kind == "rgb" else lab_rgb)[order]
                ax.scatter(P[:, 0], P[:, 1], s=0.2, c=col, linewidths=0)
                ax.set_aspect("equal")
            fig.text(0.01 + 0.25 * j + 0.115, 0.215, title, color="0.8", fontsize=7, ha="center", va="bottom",
                     linespacing=1.1)
    r2t, r2r, r2d = d["r2_t"], d["r2r"], d["r2d"]
    bad = ~np.isfinite(r2d) | (r2d < -1)
    if bm:                                          # the brain-mean R2 and RMSE up to each frame, from running sums
        okb = np.isfinite(o_) & np.isfinite(p_)
        o0, p0 = np.where(okb, o_, 0.0), np.where(okb, p_, 0.0)
        n_c, s1, s2, e2 = np.cumsum(okb), np.cumsum(o0), np.cumsum(o0 ** 2), np.cumsum((p0 - o0) ** 2)
    for k in ks:
        f = frames[k]
        sc[0].set_array(np.asarray(d["obs"][k], np.float32)[order])
        sc[1].set_array(np.asarray(d["pred"][k], np.float32)[order])
        t_txt.set_text(f"{d['names'][d['cond_of'][k]]}   t = {f * d['frame_s'] / 60:5.1f} min")
        upto = r2t <= f
        nsil = int(d["sil"][upto][-1]) if d["sil"] is not None and upto.any() else 0
        sil_txt = f"\n{nsil:,} exploding neuron{'s' if nsil != 1 else ''} silenced" if nsil else ""
        if bm:
            j_ = int(np.flatnonzero(upto)[-1]) if upto.any() else 0
            den = s2[j_] - s1[j_] ** 2 / max(n_c[j_], 1)
            r_txt.set_text(f"brain-mean R2 {1 - e2[j_] / den:+.3f}, RMSE {np.sqrt(e2[j_] / max(n_c[j_], 1)):.3f} dF/F\n"
                           "(the free rollout up to t)" if den > 0 and n_c[j_] > 2 else "")
            r_txt.set_color("white")
        elif (bad & upto).any():
            r_txt.set_text(f"diverged at t = {tm[int(np.argmax(bad))]:.0f} min\n(R2 denoised < -1)" + sil_txt)
            r_txt.set_color("#ff6b6b")
        else:
            rr, rd = r2r[upto], r2d[upto]
            r_txt.set_text((f"R2 raw {np.mean(rr):+.3f} +- {np.std(rr):.3f}\n"
                            f"R2 denoised {np.mean(rd):+.3f} +- {np.std(rd):.3f}" if len(rd) else "") + sil_txt)
            r_txt.set_color("#ffb366" if nsil else "white")
        for c_ in cur:
            c_.set_xdata([f * d["frame_s"] / 60] * 2)
        fig.savefig(os.path.join(d["tmp"], f"{k:05d}.png"), dpi=90, facecolor="black")
    plt.close(fig)
