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


def stimulus_keys(S: np.ndarray, cond: np.ndarray, cap: int = 64) -> np.ndarray:
    """Per frame, the id of (condition, stimulus vector, frames since the stimulus last changed, capped)."""
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
def render_curves(res: dict, path: str, names, gates: dict | None = None):
    """THE PREDICTION AGAINST THE BASELINES (white background, an analysis plot).
    a  MSE per step ahead, grand average: the model, the best mean baseline, persistence, the stimulus lookup,
       the noise floor sigma^2;
    b  skill over the mean baseline per step ahead: grand average (thick) and each condition (thin), the short and
       long windows shaded, the published best (the gates' full lines) dashed;
    c  the free rollout of the whole recording: R^2 per frame, raw and denoised, the conditions as bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.arange(1, H_MAX + 1)
    plt.style.use("dark_background")               # black, as every figure of the deck and the watcher
    fig = plt.figure(figsize=(13, 7.5), facecolor="black")
    a = fig.add_axes([0.06, 0.55, 0.40, 0.38])
    b = fig.add_axes([0.56, 0.55, 0.40, 0.38])
    c = fig.add_axes([0.06, 0.08, 0.90, 0.34])
    a.plot(h, res["mse_model"], color="#4c72b0", lw=2.0, label="learned law")
    a.plot(h, res["mse_mean"], color="0.85", lw=1.6, label="mean baseline (best W per step)")
    a.plot(h, res["mse_persistence"], color="0.6", lw=1.0, ls="--", label="persistence")
    a.plot(h, res["mse_lookup"], color="#c44e52", lw=1.4, label="stimulus-evoked lookup")
    a.axhline(res["noise_sigma2"], color="0.5", ls=":", lw=1.0, label="noise floor $\\sigma^2$")
    a.set_xlabel("steps ahead h (0.914 s each)")
    a.set_ylabel("MSE, dF/F$^2$ (grand average)")
    a.legend(frameon=False, fontsize=8)
    a.text(0.0, 1.04, "a", transform=a.transAxes, fontsize=12)
    for s0, s1, col in ((SHORT[0] - 0.5, SHORT[1] + 0.5, "0.22"), (LONG[0] - 0.5, LONG[1] + 0.5, "0.14")):
        b.axvspan(s0, s1, color=col, lw=0)
    sk_c = 1 - np.asarray(res["mse_model_by_condition"]) / np.asarray(res["mse_mean_by_condition"])
    cmap = plt.get_cmap("tab10")
    for ci, name in enumerate(names):
        b.plot(h, sk_c[ci], lw=0.8, color=cmap(ci), alpha=0.8, label=name)
    b.plot(h, res["skill"], color="white", lw=2.2, label="grand average")
    b.plot(h, 1 - np.asarray(res["mse_lookup"]) / np.asarray(res["mse_mean"]), color="#c44e52", lw=1.2, ls="--",
           label="stimulus lookup")
    pub = (gates or {}).get("published")
    if pub:                                        # ZAPBench's best published model, on ZAPBench's own split
        bc = pub["best_ctx4"]
        b.plot(h, bc["skill"], color="#7aa6ff", lw=1.8, ls="-.",
               label=f"ZAPBench best, {bc['method']} ctx 4 (published)")
    for key, lab in (("short_full", "published best, short"), ("long_full", "published best, long")):
        if gates and key in gates:
            rng = SHORT if key.startswith("short") else LONG
            b.plot(rng, [gates[key]] * 2, color="0.7", ls=":", lw=1.5)
            b.text(rng[0], gates[key] + 0.015, lab, fontsize=7, va="bottom")
    b.axhline(0, color="0.6", lw=0.7)
    top = max(float(np.nanmax(sk_c)) + 0.1, (gates or {}).get("long_full", 0.0) + 0.1)
    b.set_ylim(max(-1.0, float(np.nanmin(sk_c)) - 0.05), min(1.0, top))
    b.set_xlabel("steps ahead h")
    b.set_ylabel("MSE skill over the mean baseline")
    b.legend(frameon=False, fontsize=6, ncol=2, loc="lower right")
    b.text(0.0, 1.04, "b", transform=b.transAxes, fontsize=12)
    t = np.asarray(res["free_t_s"]) / 60
    off = res["offsets"]
    for ci in range(len(names)):
        c.axvspan(off[ci] * res["frame_s"] / 60, off[ci + 1] * res["frame_s"] / 60,
                  color=cmap(ci), alpha=0.15, lw=0)
        c.text((off[ci] + off[ci + 1]) / 2 * res["frame_s"] / 60, 1.02, names[ci], fontsize=7, ha="center",
               transform=c.get_xaxis_transform())
    c.plot(t, res["free_r2_raw"], color="0.6", lw=0.6, label="R$^2$ raw")
    c.plot(t, res["free_r2_denoised"], color="#4c72b0", lw=0.8, label="R$^2$ denoised")
    c.axhline(0, color="0.6", lw=0.7)
    r2d = np.asarray(res["free_r2_denoised"], dtype=float)
    fin = r2d[np.isfinite(r2d)]                    # a diverged rollout's -inf / nan frames must not set the axis
    lo = np.percentile(fin, 2) if len(fin) else -3.0
    c.set_ylim(max(lo - 0.1, -3), 1)
    if len(fin) < len(r2d):
        c.text(0.99, 0.04, f"not finite from frame {int(np.argmax(~np.isfinite(r2d)))}", color="#d62728", fontsize=8,
               ha="right", transform=c.transAxes)
    c.set_xlabel("time, min (free rollout from the law's own context frames)")
    c.set_ylabel("R$^2$ per frame over neurons")
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


def render_movie(obs: np.ndarray, pred: np.ndarray, frames: np.ndarray, pos: np.ndarray, r2_raw: np.ndarray,
                 r2_den: np.ndarray, frame_s: float, cond_of: np.ndarray, names, path: str,
                 emb: np.ndarray | None = None, labels: np.ndarray | None = None, fps: int = 25,
                 mean_obs: np.ndarray | None = None, mean_pred: np.ndarray | None = None,
                 ffmpeg: str | None = None, emb_name: str = "embedding", law: str = "GraphCast law",
                 inputs: np.ndarray | None = None):
    """TWO PANELS, recorded LEFT and learned RIGHT: every neuron a point at its position (dorsal view, deeper
    drawn first) coloured by dF/F on one scale; black background, labels inside; `R2 raw` and `R2 denoised`, mean
    +- SD over the frames so far, top right; under the recorded panel the brain-mean dF/F, recorded and learned, with
    a cursor; a strip of insets below, titled above -- the embedding's PC1-PC2 coloured by its clusters, the
    embedding on the brain (PCA as RGB), its clusters on the brain -- when the law learns one. `frames` are
    consecutive: one movie frame per recorded frame, at `fps` (the pace of the deck's data movie)."""
    import os
    import shutil
    import subprocess
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ffmpeg = ffmpeg or _ffmpeg()
    order = np.argsort(pos[:, 2])
    P = pos[order]
    vmax = float(np.percentile(obs, 97))
    tmp = tempfile.mkdtemp(prefix="trace_movie_")
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
        inp_col = np.tile(np.array([[0.93, 0.55, 0.25]]), (len(pos), 1))
        inp_title = "stimulus input: every neuron\n(no per-neuron input weights)"
    if ins:
        pc, rgb = _pca_rgb(emb)
        cm = plt.get_cmap("tab10")
        lab_rgb = cm(labels % 10)[:, :3] if labels is not None else rgb
    for k in range(len(frames)):
        fig = plt.figure(figsize=(12, 8.6 if ins else 7.0), facecolor="black")
        top = 0.40 if ins else 0.16
        for j, (img, lab) in enumerate(((obs[k], "recorded (ZAPBench)"), (pred[k], f"learned ({law})"))):
            ax = fig.add_axes([0.5 * j, top, 0.5, 0.88 - top])       # the labels sit above, never on the brain
            ax.set_facecolor("black")
            ax.axis("off")
            ax.scatter(P[:, 0], P[:, 1], c=img[order], s=0.5, cmap="inferno", vmin=0, vmax=vmax, linewidths=0)
            ax.set_aspect("equal")
            fig.text(0.5 * j + 0.03, 0.985, lab, color="white", fontsize=12, va="top")
            if j == 0:
                t = frames[k] * frame_s
                fig.text(0.03, 0.945, f"{names[cond_of[k]]}   t = {t / 60:5.1f} min", color="0.7", fontsize=9, va="top")
            else:
                fig.text(0.97, 0.955, f"R2 raw {np.mean(r2_raw[:k + 1]):+.3f} +- {np.std(r2_raw[:k + 1]):.3f}\n"
                         f"R2 denoised {np.mean(r2_den[:k + 1]):+.3f} +- {np.std(r2_den[:k + 1]):.3f}",
                         color="white", fontsize=9, va="top", ha="right")
        if mean_obs is not None:
            m = fig.add_axes([0.05, top - 0.10, 0.40, 0.07])
            m.set_facecolor("black")
            for sp in m.spines.values():
                sp.set_visible(False)
            m.set_xticks([])
            m.set_yticks([])
            tt = np.arange(len(frames))
            m.plot(tt, mean_obs, color="#2ca02c", lw=1.0)
            if mean_pred is not None:
                m.plot(tt, mean_pred, color="white", lw=1.0)
            m.axvline(k, color="0.7", lw=0.8)
            m.set_xlim(0, len(frames) - 1)
            fig.text(0.05, top - 0.02, "brain-mean dF/F: recorded (green), learned (white)", color="0.7", fontsize=8,
                     va="bottom")
        if ins:
            for j, (title, kind) in enumerate(((f"{emb_name} PC1-PC2, by cluster", "scatter"),
                                               (f"{emb_name} on the brain (PCA as RGB)", "rgb"),
                                               ("its clusters on the brain", "labels"),
                                               (inp_title, "inputs"))):
                ax = fig.add_axes([0.01 + 0.25 * j, 0.01, 0.23, 0.19])   # below its title, never under it
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
        fig.savefig(os.path.join(tmp, f"{k:05d}.png"), dpi=90, facecolor="black")
        plt.close(fig)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", path],
                   check=True)
    shutil.copy(os.path.join(tmp, f"{len(frames) // 2:05d}.png"), path.replace(".mp4", ".png"))
    shutil.rmtree(tmp)
