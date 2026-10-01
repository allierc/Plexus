"""A recorded FIELD, as a training REFERENCE: what a tissue's scalar map actually did, voxel by voxel.

    rec = load("hlo_washout")                  # graphs_data/redox/hlo_washout_recording.npz
    c   = coarsen(rec, 4)                      # the model's grid: 14 x 128 x 128
    tr, te = split(c, n_test=15)               # the last 15 volumes are held out
    s   = score(pred, c, origins, h)           # RMSE on the organoid, per horizon

The recording is a sequence of volumes x(t) [Z, Y, X] with a mask M(t) of where the tissue is
(the redox organoid: 69 volumes, 600 s apart). It has no cells and no tracks, so a model is
compared with it voxel against voxel, and only where the tissue is at both ends of the forecast:

    RMSE_h = sqrt( mean over (t, v) with M(t-h)[v] and M(t)[v] of (xhat_h(t)[v] - x(t)[v])^2 )

xhat_h(t) is the forecast of volume t made from volumes up to t - h. The voxel set is the same for
every predictor at a given h, so two predictors' RMSEs are comparable; it shrinks with h as the
organoid moves, and `score` reports its size.

WHY BLOCK MEANS OVER THE MASK, AND WHY HALF. The model runs on a coarser grid than the microscope
(`coarsen`, a factor f per axis in x and y, z untouched). A coarse voxel is the mean of the fine
voxels of its f x f block that lie on the mask, and it is on the coarse mask when at least half of
the block is -- a boundary voxel mostly outside the tissue would be a mean of a few pixels, noisier
than its neighbours by up to sqrt(f^2 / 2), and would weigh on the score as if it were tissue.

The frozen file is written once by `tools/export_redox_recording.py`. The baselines every model
must beat (`baselines`) are here, beside the scorer, so a model and its baselines are scored by the
same arithmetic.
"""
from __future__ import annotations

import os

import numpy as np

from plexus.paths import graphs_data_path

GROUP = "redox"


def load(name, group=GROUP):
    """The frozen recording `name` as a dict of numpy arrays: ratio [T,Z,Y,X] float32, mask
    [T,Z,Y,X] bool, t_s [T] (s since the first volume), dx_um, dz_um, and the lab's trend table."""
    p = graphs_data_path(group, f"{name}_recording.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} does not exist -- freeze it first: "
                                f"PYTHONPATH=src python tools/export_{group}_recording.py")
    z = np.load(p)
    rec = {k: z[k] for k in z.files}
    rec["name"] = name
    rec["dx_um"], rec["dz_um"] = float(rec["dx_um"]), float(rec["dz_um"])
    return rec


def coarsen(rec, f=4, keep=0.5):
    """Block means of the ratio over the mask, f x f in (y, x); z untouched. Returns a recording
    dict of the same shape as `load`'s, at the coarse grid, with `dx_um` multiplied by f."""
    x, m = rec["ratio"], rec["mask"]
    T, Z, Y, X = x.shape
    if Y % f or X % f:
        raise ValueError(f"a {Y} x {X} plane does not divide into {f} x {f} blocks")
    xs = (x * m).reshape(T, Z, Y // f, f, X // f, f).sum((3, 5))
    ms = m.reshape(T, Z, Y // f, f, X // f, f).sum((3, 5))
    cm = ms >= keep * f * f
    cx = np.where(cm, xs / np.maximum(ms, 1), 0.0).astype(np.float32)
    out = dict(rec)
    out.update(ratio=cx, mask=cm, dx_um=rec["dx_um"] * f, coarsen=f)
    return out


def split(rec, n_test=15):
    """Indices of the TRAIN and TEST volumes: the last `n_test` volumes are the test targets. A
    forecast of a test volume may start from a train volume -- the origin is observed; the target
    is what is held out."""
    T = rec["ratio"].shape[0]
    return np.arange(T - n_test), np.arange(T - n_test, T)


def pairs(targets, h):
    """(origin, target) index pairs for horizon h, targets whose origin exists."""
    return [(int(t) - h, int(t)) for t in targets if int(t) - h >= 0]


def score(forecast, rec, targets, h):
    """RMSE of `forecast(origin, h) -> [Z,Y,X]` over the voxels on the mask at origin AND target,
    pooled over every target in `targets`. Returns dict(rmse, mse, n_vox, n_pairs)."""
    x, m = rec["ratio"], rec["mask"]
    se, n = 0.0, 0
    ps = pairs(targets, h)
    for o, t in ps:
        v = m[o] & m[t]
        d = forecast(o, h)[v].astype(np.float64) - x[t][v]
        se += float((d * d).sum())
        n += int(v.sum())
    mse = se / max(n, 1)
    return dict(rmse=float(np.sqrt(mse)), mse=float(mse), n_vox=n, n_pairs=len(ps))


# ------------------------------------------------------------------------------ the predictors
def persistence(rec):
    """xhat(t) = x(t - h): the last observed volume."""
    return lambda o, h: rec["ratio"][o]


def window_mean(rec, K):
    """xhat(t) = the mean of each voxel over the last K volumes ending at the origin, over the
    volumes where it is on the mask (persistence where it never is). Denoising in time."""
    x, m = rec["ratio"], rec["mask"]

    def f(o, h):
        lo = max(0, o - K + 1)
        w = m[lo:o + 1]
        s, c = (x[lo:o + 1] * w).sum(0), w.sum(0)
        return np.where(c > 0, s / np.maximum(c, 1), x[o])
    return f


def linear_trend(rec, K):
    """xhat(t) = a + b t_t, each voxel's least-squares line through its last K volumes ending at
    the origin (on the mask only), read at the target's REAL time -- so the one 885 s gap in the
    recording bends nothing. A voxel with fewer than 2 points on the mask keeps persistence."""
    x, m, ts = rec["ratio"], rec["mask"], rec["t_s"]

    def f(o, h):
        lo = max(0, o - K + 1)
        w = m[lo:o + 1].astype(np.float64)
        tt = (ts[lo:o + 1] - ts[o])[:, None, None, None]
        c = w.sum(0)
        tbar = (w * tt).sum(0) / np.maximum(c, 1)
        xbar = (w * x[lo:o + 1]).sum(0) / np.maximum(c, 1)
        stt = (w * (tt - tbar) ** 2).sum(0)
        stx = (w * (tt - tbar) * (x[lo:o + 1] - xbar)).sum(0)
        b = np.where(stt > 0, stx / np.maximum(stt, 1e-12), 0.0)
        pred = xbar + b * ((ts[o + h] - ts[o]) - tbar)
        return np.where(c >= 2, pred, x[o]).astype(np.float32)
    return f


def smoothed(rec, sigma):
    """xhat(t) = the origin volume blurred in (y, x) by a Gaussian of `sigma` voxels, normalised
    by the blurred mask so the tissue's edge is not darkened by the zeros beside it. Denoising in
    space; z is not blurred (its step is 5 um against 0.83 um in x and y at full resolution)."""
    from scipy.ndimage import gaussian_filter
    x, m = rec["ratio"], rec["mask"]
    cache = {}

    def f(o, h):
        if o not in cache:
            mm = m[o].astype(np.float32)
            num = gaussian_filter(x[o] * mm, sigma=(0, sigma, sigma), mode="constant")
            den = gaussian_filter(mm, sigma=(0, sigma, sigma), mode="constant")
            cache[o] = np.where(den > 1e-6, num / np.maximum(den, 1e-6), x[o]).astype(np.float32)
        return cache[o]
    return f


def space_time(rec, knob):
    """xhat(t) = `window_mean` over the last K volumes, then blurred in (y, x) by sigma voxels as
    `smoothed` blurs -- denoising in time AND space; `knob` is (K, sigma). The best non-learned
    baseline at the model grid (exp16, 2026-09-29): it beats either alone at every horizon."""
    from scipy.ndimage import gaussian_filter
    K, sigma = knob
    x, m = rec["ratio"], rec["mask"]
    wm = window_mean(rec, K)

    def f(o, h):
        v = wm(o, h)
        mm = m[max(0, o - K + 1):o + 1].any(0).astype(np.float32)
        num = gaussian_filter(v * mm, sigma=(0, sigma, sigma), mode="constant")
        den = gaussian_filter(mm, sigma=(0, sigma, sigma), mode="constant")
        return np.where(den > 1e-6, num / np.maximum(den, 1e-6), v).astype(np.float32)
    return f


# ------------------------------------------------------------------------------ what the data allow
def structure_function(rec, targets, hs=(1, 2, 3, 4, 6)):
    """D(h) = mean over pairs (t - h, t), both on the mask, of (x(t) - x(t - h))^2, for each h.

    Under x = s + n, with s a signal that changes with time and n noise independent between
    volumes, D(h) = 2 var(n) + S(h), S(0) = 0. So the intercept of D at h -> 0 (the NUGGET) is
    twice the noise variance, and sqrt(nugget / 2) is the NOISE FLOOR: the RMSE of a forecaster
    that knew the signal exactly and could not know the noise."""
    return {int(h): score(persistence(rec), rec, targets, h)["mse"] for h in hs}


def noise_floor(D, fit_hs=(1, 2, 3)):
    """sqrt(nugget / 2) from a straight line through D(h) at `fit_hs`, and the line itself."""
    h = np.array(fit_hs, float)
    d = np.array([D[int(k)] for k in fit_hs])
    b, a = np.polyfit(h, d, 1)
    return dict(sigma=float(np.sqrt(max(a, 0.0) / 2)), nugget=float(a), slope_per_step=float(b),
                fit_hs=list(fit_hs))


def mask_drift(rec, targets, h):
    """How far the tissue itself moves in h steps: 1 - IoU of the masks, the shift of the mask's
    centroid in um, and the mask volume ratio, averaged over the (origin, target) pairs."""
    m = rec["mask"]
    Z, Y, X = m.shape[1:]
    zz, yy, xx = np.meshgrid(np.arange(Z) * rec["dz_um"], np.arange(Y) * rec["dx_um"],
                             np.arange(X) * rec["dx_um"], indexing="ij")
    out = []
    for o, t in pairs(targets, h):
        a, b = m[o], m[t]
        iou = (a & b).sum() / max((a | b).sum(), 1)
        ca = np.array([zz[a].mean(), yy[a].mean(), xx[a].mean()])
        cb = np.array([zz[b].mean(), yy[b].mean(), xx[b].mean()])
        out.append((1 - iou, float(np.linalg.norm(cb - ca)), b.sum() / max(a.sum(), 1)))
    o = np.array(out)
    return dict(one_minus_iou=float(o[:, 0].mean()), centroid_shift_um=float(o[:, 1].mean()),
                volume_ratio=float(o[:, 2].mean()))


def organoid_mean(rec):
    """[T] the mean ratio over the mask, per volume: the washout response a model must follow."""
    x, m = rec["ratio"], rec["mask"]
    return np.array([float(x[t][m[t]].mean()) for t in range(x.shape[0])])


# ------------------------------------------------------------------------------ the 3-D movie
def _r2_upto(obs, pred, mask):
    """[T] x 2: the mean and SD, over frames 0..t, of each frame's R2 = 1 - sum (pred - obs)^2 /
    sum (obs - xbar)^2 over that frame's tissue, xbar the recording's mean over the WHOLE rollout -- so
    it is the total R2 (the organoid-wide level counts), frame by frame."""
    xbar = float(obs[mask].mean())
    r = []
    for t in range(obs.shape[0]):
        m = mask[t]
        x, p = obs[t][m].astype(np.float64), pred[t][m].astype(np.float64)
        r.append(1 - float(((p - x) ** 2).sum()) / max(float(((x - xbar) ** 2).sum()), 1e-12))
    r = np.array(r)
    return np.array([(r[:t + 1].mean(), r[:t + 1].std()) for t in range(len(r))])


def _inset(frame, png, box, label=None):
    """Paste `png` into `frame` [H, W, 3] at box (x0, y0, w, h), aspect kept, a thin grey frame."""
    from PIL import Image
    im = Image.open(png).convert("RGB")
    x0, y0, w, h = box
    im.thumbnail((w, h))
    a = np.asarray(im)
    hh, ww = a.shape[:2]
    frame[y0:y0 + hh, x0:x0 + ww] = a
    frame[y0:y0 + hh, [x0, x0 + ww - 1]] = 90
    frame[[y0, y0 + hh - 1], x0:x0 + ww] = 90
    return frame


def render_pair_3d(obs, pred, mask, dx_um, dz_um, path, labels=("recorded", "learned"), times_min=None,
                   clim=None, duration_s=10.0, window=(1200, 560), elev=35.0, azim=-60.0,
                   show_r2=True, insets=(), mask_pred=None, obs_denoised=None, mask_obs=None, iou=None):
    """A movie of two volume sequences side by side -- the recording LEFT, the model RIGHT -- each frame
    the redox ratio of the tissue as a translucent volume under an OBLIQUE camera (elevation `elev`,
    azimuth `azim`, degrees), black background, white labels inside the panels.

    obs, pred [T, Z, Y, X]; mask [T, Z, Y, X] bool, the SCORED voxels (the R2 is read there);
    `mask_obs` the domain the recording is drawn on (its own tissue at each frame; default `mask`),
    `mask_pred` the domain the model is drawn on (default `mask`). Voxels are dx_um x dx_um x dz_um, so
    the 5-um planes stand at their true depth. Opacity is 0 off the tissue and rises with the ratio
    over `clim` (default the recording's p2-p98 on the tissue), so a high-NADH voxel is both brighter
    and more opaque. Off-screen VTK (`DISPLAY` must be unset). Returns the path."""
    import os
    os.environ.pop("DISPLAY", None)
    import imageio.v2 as imageio
    import pyvista as pv
    # CROPPED TO THE TISSUE (the bounding box of both panels' domains over all frames, 4 voxels of margin),
    # so the organoid fills the panel instead of a 424-um box that is mostly empty.
    # THE LEARNED PANEL IS DRAWN ON ITS OWN DOMAIN, `mask_pred` (the tissue at the rollout's start): the law
    # draws no boundary -- it fills the box with ratio-like values -- so drawing it on the RECORDED mask of
    # frame t would lend it the recording's change of shape (exp16, 2026-09-29: the organoid shrank to 0.76x
    # and the law was credited with it). The R2 is still read on the scored voxels, `mask`.
    mask_pred = mask if mask_pred is None else np.broadcast_to(mask_pred, mask.shape)
    mask_obs = mask if mask_obs is None else mask_obs
    both = mask | mask_pred | mask_obs
    ys, xs = np.where(both.any((0, 1)))
    y0, y1 = max(ys.min() - 4, 0), min(ys.max() + 5, both.shape[2])
    x0, x1 = max(xs.min() - 4, 0), min(xs.max() + 5, both.shape[3])
    obs, pred, mask = obs[:, :, y0:y1, x0:x1], pred[:, :, y0:y1, x0:x1], mask[:, :, y0:y1, x0:x1]
    mask_pred, mask_obs = mask_pred[:, :, y0:y1, x0:x1], mask_obs[:, :, y0:y1, x0:x1]
    T, Z, Y, X = obs.shape
    r2 = _r2_upto(obs, pred, mask) if show_r2 else None
    r2d = None
    if show_r2 and obs_denoised is not None:
        r2d = _r2_upto(obs_denoised[:, :, y0:y1, x0:x1], pred, mask)
    if clim is None:
        clim = tuple(np.percentile(obs[mask], [2, 98]))
    lo, hi = clim
    opacity = [0.0, 0.0, 0.05, 0.15, 0.35]                 # over [below clim .. clim hi]

    def grid(v, m):
        g = pv.ImageData(dimensions=(X + 1, Y + 1, Z + 1), spacing=(dx_um, dx_um, dz_um))
        # VTK's x runs fastest: [Z, Y, X] in C order is exactly its cell order.
        vv = np.where(m, np.clip(v, lo, hi), lo - (hi - lo)).astype(np.float32)
        g.cell_data["ratio"] = vv.reshape(-1)
        return g.cell_data_to_point_data()

    fps = max(1, int(round(T / float(duration_s))))
    with imageio.get_writer(path, fps=fps, macro_block_size=8) as w:
        for t in range(T):
            p = pv.Plotter(off_screen=True, shape=(1, 2), window_size=window, border=False)
            for j, (v, lab, mk) in enumerate(((obs[t], labels[0], mask_obs[t]), (pred[t], labels[1], mask_pred[t]))):
                p.subplot(0, j)
                p.set_background("black")
                p.add_volume(grid(v, mk), scalars="ratio", cmap="viridis", clim=(lo - (hi - lo), hi),
                             opacity=opacity, show_scalar_bar=False, shade=False)
                txt = lab + (f"   t = {times_min[t]:.0f} min" if times_min is not None else "")
                p.add_text(txt, position="upper_left", font_size=11, color="white")
                if j == 1 and r2 is not None:
                    # VARIANCE EXPLAINED, frame by frame: mean +- SD over frames 0..t (`_r2_upto`).
                    txt2 = f"R2 raw       {r2[t, 0]:+.3f} +- {r2[t, 1]:.3f}"
                    if r2d is not None:
                        txt2 += f"\nR2 denoised  {r2d[t, 0]:+.3f} +- {r2d[t, 1]:.3f}"
                    if iou is not None:                # a death operator: the model's tissue against the recorded
                        txt2 += f"\nshape IoU    {iou[t]:.3f}"
                    p.add_text(txt2, position="upper_right", font_size=11, color="white")
                p.camera_position = "xy"
                p.camera.elevation = elev
                p.camera.azimuth = azim
                p.reset_camera()
                p.camera.zoom(1.4)
            img = np.ascontiguousarray(p.screenshot(return_img=True)[..., :3])
            p.close()
            # INSETS in a strip UNDER the learned panel, so they never cover the organoid: static pictures
            # (the embedding, its domains). The strip's height keeps the frame a multiple of 8 for the codec.
            if insets:
                iw, ih = window[0] // 4 - 9, 208
                img = np.concatenate([img, np.zeros((ih + 8, img.shape[1], 3), img.dtype)], 0)
                x0 = window[0] - len(insets) * (iw + 6)          # right-aligned, under the learned panel first
                for q, png in enumerate(insets):
                    img = _inset(img, png, (x0 + q * (iw + 6), window[1] + 4, iw, ih))
            w.append_data(img)
    return path


def render_embedding_3d(a, mask, dx_um, dz_um, path, title="embedding", window=(520, 420), elev=35.0, azim=-60.0):
    """The per-voxel embedding in 3-D: its first three principal components over the tissue as RGB on the
    tissue's voxels, under the movies' oblique camera, black background, title inside."""
    import os
    os.environ.pop("DISPLAY", None)
    import pyvista as pv
    v = a[:, mask].T
    v = v - v.mean(0)
    _, _, Vt = np.linalg.svd(v[:: max(1, len(v) // 50000)], full_matrices=False)
    pc = v @ Vt[: min(3, Vt.shape[0])].T
    lo, hi = np.percentile(pc, 2, axis=0), np.percentile(pc, 98, axis=0)
    rgb = np.clip((pc - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
    if rgb.shape[1] < 3:
        rgb = np.concatenate([rgb, np.full((len(rgb), 3 - rgb.shape[1]), 0.5)], 1)
    Z, Y, X = mask.shape
    col = np.zeros((Z, Y, X, 3), np.float32)
    col[mask] = rgb
    lab = np.where(mask, 0, -1)
    ys, xs = np.where(mask.any(0))
    sl = (slice(None), slice(max(ys.min() - 2, 0), ys.max() + 3), slice(max(xs.min() - 2, 0), xs.max() + 3))
    lab, col = lab[sl], col[sl]
    Z, Y, X = lab.shape
    g = pv.ImageData(dimensions=(X + 1, Y + 1, Z + 1), spacing=(dx_um, dx_um, dz_um))
    g.cell_data["on"] = lab.reshape(-1).astype(np.float32)
    g.cell_data["rgb"] = (col.reshape(-1, 3) * 255).astype(np.uint8)
    body = g.threshold(-0.5, scalars="on")
    p = pv.Plotter(off_screen=True, window_size=window)
    p.set_background("black")
    p.add_mesh(body, scalars="rgb", rgb=True)
    p.add_text(title, position="upper_left", font_size=10, color="white")
    p.camera_position = "xy"
    p.camera.elevation = elev
    p.camera.azimuth = azim
    p.reset_camera()
    p.camera.zoom(1.3)
    p.screenshot(path)
    p.close()
    return path


def domains(lab8, min_vox=2):
    """The 3-D connected domains of each cluster (6-connected), numbered 0.. over all clusters; -1 off
    the tissue and for domains under `min_vox` voxels. A domain is a candidate cell."""
    from scipy import ndimage
    out = np.full(lab8.shape, -1, dtype=np.int64)
    nxt = 0
    for c in range(int(lab8.max()) + 1):
        cc, n = ndimage.label(lab8 == c)
        if n == 0:
            continue
        sizes = np.bincount(cc.reshape(-1))
        keep = np.where(sizes >= min_vox)[0]
        keep = keep[keep > 0]
        remap = np.full(n + 1, -1, dtype=np.int64)
        remap[keep] = np.arange(nxt, nxt + len(keep))
        out = np.where(cc > 0, np.maximum(out, remap[cc]), out)
        nxt += len(keep)
    return out


def render_domains_3d(dom, dx_um, dz_um, path, title="domains", window=(520, 420), elev=35.0, azim=-60.0, seed=0):
    """Each 3-D domain (`domains`) one random colour -- one colour per candidate cell -- oblique, black."""
    import os
    os.environ.pop("DISPLAY", None)
    import pyvista as pv
    ys, xs = np.where((dom >= 0).any(0))
    dom = dom[:, max(ys.min() - 2, 0):ys.max() + 3, max(xs.min() - 2, 0):xs.max() + 3]
    Z, Y, X = dom.shape
    n = int(dom.max()) + 1
    pal = (np.random.default_rng(seed).uniform(0.25, 1.0, size=(max(n, 1), 3)) * 255).astype(np.uint8)
    col = np.zeros((Z, Y, X, 3), np.uint8)
    col[dom >= 0] = pal[dom[dom >= 0]]
    g = pv.ImageData(dimensions=(X + 1, Y + 1, Z + 1), spacing=(dx_um, dx_um, dz_um))
    g.cell_data["dom"] = dom.reshape(-1).astype(np.float32)
    g.cell_data["rgb"] = col.reshape(-1, 3)
    body = g.threshold(-0.5, scalars="dom")
    p = pv.Plotter(off_screen=True, window_size=window)
    p.set_background("black")
    p.add_mesh(body, scalars="rgb", rgb=True)
    p.add_text(title, position="upper_left", font_size=10, color="white")
    p.camera_position = "xy"
    p.camera.elevation = elev
    p.camera.azimuth = azim
    p.reset_camera()
    p.camera.zoom(1.3)
    p.screenshot(path)
    p.close()
    return path


# ------------------------------------------------------------------------------ the embedding's clusters
def cluster_embedding(a, mask, dx_um, dz_um, ks=range(2, 9), seed=0, n_fit=20000):
    """Do the voxels' learned embedding fall into CLUSTERS, and are those clusters DOMAINS in 3-D?

    a [k, Z, Y, X] the embedding; mask [Z, Y, X] the tissue. Two questions, answered apart:

      in embedding space   k-means on the tissue voxels' embedding (standardised), k chosen in `ks` by
                           the silhouette score on `n_fit` voxels; the silhouette itself says how
                           separated the clusters are (1 = apart, 0 = one continuous cloud cut in pieces)
      in the volume        `coherence`: of the pairs of 6-neighbouring tissue voxels, the excess fraction
                           with the same label over chance (the label frequencies' sum f_c^2):
                           (P_same - P_chance) / (1 - P_chance) -- 0 for labels scattered at random (the
                           shuffled null is reported), 1 for perfectly contiguous domains. And at a fixed
                           8 clusters, the median volume of the 3-D connected domains (6-connected, 2
                           voxels or more), um^3 -- the scale to hold against a cell's. A component-size
                           measure at the chosen k fails: at k = 2 each cluster percolates through the
                           organoid (above the ~31 % site-percolation threshold), the shuffled null too.

    Returns (labels [Z, Y, X] int, -1 off the tissue; dict of the numbers; the 8-cluster labels)."""
    from scipy import ndimage
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    v = a[:, mask].T.astype(np.float64)
    if float(v.std(0).max()) < 1e-6:
        # A DEGENERATE EMBEDDING (every voxel alike -- one that never learned, exp16 v34): one cluster, nothing to
        # measure. Reported as such rather than handed to k-means and the silhouette, which refuse one label.
        lab = np.where(mask, 0, -1).astype(np.int64)
        return lab, {"k": 1, "silhouette": 0.0, "silhouette_by_k": {}, "cluster_fraction": [1.0], "coherence": 0.0,
                     "coherence_null": 0.0, "coherence_k8": 0.0, "domain_um3_median_k8": 0.0, "n_domains_k8": 0,
                     "voxel_um3": dx_um * dx_um * dz_um, "degenerate": True}, lab
    v = (v - v.mean(0)) / np.maximum(v.std(0), 1e-12)
    rng = np.random.default_rng(seed)
    sub = v[rng.choice(len(v), min(n_fit, len(v)), replace=False)]
    best = None
    sil = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(sub)
        sil[int(k)] = float(silhouette_score(sub, km.labels_, sample_size=min(5000, len(sub)), random_state=seed))
        if best is None or sil[int(k)] > sil[best[0]]:
            best = (int(k), km)
    k, km = best
    lab = np.full(mask.shape, -1, dtype=np.int64)
    lab[mask] = km.predict(v)
    vox = dx_um * dx_um * dz_um

    def coherence(L):
        """Excess agreement of 6-neighbours on the tissue: (P_same - P_chance) / (1 - P_chance)."""
        same = tot = 0
        for ax in range(3):
            sl0, sl1 = [slice(None)] * 3, [slice(None)] * 3
            sl0[ax], sl1[ax] = slice(0, -1), slice(1, None)
            A, B = L[tuple(sl0)], L[tuple(sl1)]
            both = (A >= 0) & (B >= 0)
            same += int((A[both] == B[both]).sum())
            tot += int(both.sum())
        f = np.bincount(L[L >= 0]) / max((L >= 0).sum(), 1)
        pc = float((f ** 2).sum())
        return ((same / max(tot, 1)) - pc) / max(1 - pc, 1e-12)

    # AT A FIXED 8 CLUSTERS no cluster holds more than a fraction of the tissue, so none percolates
    # and its connected domains have a size (at 2 clusters each spans the organoid, null included).
    km8 = KMeans(n_clusters=8, n_init=4, random_state=seed).fit(sub)
    lab8 = np.full(mask.shape, -1, dtype=np.int64)
    lab8[mask] = km8.predict(v)
    sizes = []
    for c in range(8):
        cc, n = ndimage.label(lab8 == c)
        sizes += list(np.bincount(cc.reshape(-1))[1:])
    sizes = np.array(sizes, dtype=np.float64)
    shuf = np.full(mask.shape, -1, dtype=np.int64)
    shuf[mask] = rng.permutation(lab[mask])
    frac = np.bincount(lab[mask], minlength=k) / mask.sum()
    return lab, {"k": k, "silhouette": sil[k], "silhouette_by_k": sil, "cluster_fraction": frac.tolist(),
                 "coherence": coherence(lab), "coherence_null": coherence(shuf),
                 "coherence_k8": coherence(lab8),
                 "domain_um3_median_k8": float(np.median(sizes[sizes >= 2]) * vox) if (sizes >= 2).any() else 0.0,
                 "n_domains_k8": int((sizes >= 2).sum()), "voxel_um3": vox}, lab8


def render_labels_3d(labels, dx_um, dz_um, path, title="", window=(900, 700), elev=35.0, azim=-60.0):
    """The cluster labels as coloured voxel blocks under the same oblique camera as the movies; black
    background, the title inside the panel. One colour per cluster (tab10)."""
    import os
    os.environ.pop("DISPLAY", None)
    import pyvista as pv
    Z, Y, X = labels.shape
    ys, xs = np.where((labels >= 0).any(0))
    labels = labels[:, max(ys.min() - 2, 0):ys.max() + 3, max(xs.min() - 2, 0):xs.max() + 3]
    Z, Y, X = labels.shape
    g = pv.ImageData(dimensions=(X + 1, Y + 1, Z + 1), spacing=(dx_um, dx_um, dz_um))
    g.cell_data["cluster"] = labels.reshape(-1).astype(np.float32)
    body = g.threshold(-0.5, scalars="cluster")
    p = pv.Plotter(off_screen=True, window_size=window)
    p.set_background("black")
    k = int(labels.max()) + 1
    p.add_mesh(body, scalars="cluster", cmap="tab10", clim=(-0.5, 9.5), show_scalar_bar=False,
               categories=True, n_colors=10)
    p.add_text(title, position="upper_left", font_size=11, color="white")
    p.camera_position = "xy"
    p.camera.elevation = elev
    p.camera.azimuth = azim
    p.reset_camera()
    p.camera.zoom(1.3)
    p.screenshot(path)
    p.close()
    return path


def denoise(rec, box=(3, 3, 3), frames=3):
    """The recording averaged over a BOX of `box` voxels in (z, y, x) and `frames` consecutive frames centred
    on each frame (t-1, t, t+1 for 3), over the tissue only (a voxel's mean takes the tissue voxels of its
    box, never the zeros beside the organoid); the mask unchanged. A TARGET for scoring beside the raw one
    (Cedric, 2026-09-29: 3 voxels in x, y, z and 3 frames): voxel noise is 56 % of the raw variance at the
    model grid. Fixed in advance, never tuned to a model. At the first and last frame the time window is
    the frames that exist."""
    from scipy.ndimage import uniform_filter
    x, m = rec["ratio"].astype(np.float64), rec["mask"].astype(np.float64)
    size = (int(frames),) + tuple(int(b) for b in box)
    num = uniform_filter(x * m, size=size, mode="constant")
    den = uniform_filter(m, size=size, mode="constant")
    out = np.where(rec["mask"], num / np.maximum(den, 1e-9), 0.0)
    d = dict(rec)
    d["ratio"], d["denoise"] = out.astype(np.float32), {"box": list(box), "frames": int(frames)}
    return d


def residual_stats(r, mask):
    """Is a residual r [T, Z, Y, X] (on `mask`) pure Gaussian white noise? Its sd; its skewness and
    excess kurtosis (0 and 0 for a Gaussian); its correlation with itself one voxel away along z, y and
    x, and one frame later at the same voxel (0 for white noise); and the correlation of its magnitude
    with the level -- here the voxel's value in `r`'s reference is not known, so the caller passes the
    residual only and `level_corr` is left to it."""
    v = r[mask].astype(np.float64)
    mu, sd = v.mean(), v.std()
    z = (v - mu) / max(sd, 1e-12)
    out = {"mean": float(mu), "sd": float(sd), "skew": float((z ** 3).mean()), "excess_kurtosis": float((z ** 4).mean() - 3)}
    rc = np.where(mask, r - mu, 0.0)
    for name, ax in (("t", 0), ("z", 1), ("y", 2), ("x", 3)):
        sl0, sl1 = [slice(None)] * 4, [slice(None)] * 4
        sl0[ax], sl1[ax] = slice(0, -1), slice(1, None)
        both = mask[tuple(sl0)] & mask[tuple(sl1)]
        a, b = rc[tuple(sl0)][both], rc[tuple(sl1)][both]
        out[f"autocorr_{name}"] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))
    return out



def render_embedding_scatter(a, mask, labels, path, title="embedding", size=(5.2, 4.2)):
    """The tissue voxels in the embedding's first two principal components, one dot per voxel, coloured by
    their cluster (`labels`, tab10) -- connectome-gnn's embedding scatter, where cell types separate. Black
    background, white text, to sit in the movie."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = a[:, mask].T.astype(np.float64)
    v = v - v.mean(0)
    _, _, Vt = np.linalg.svd(v[:: max(1, len(v) // 50000)], full_matrices=False)
    pc = v @ Vt[:2].T
    lab = labels[mask]
    sub = np.random.default_rng(0).choice(len(pc), min(20000, len(pc)), replace=False)
    fig, ax = plt.subplots(figsize=size, facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(pc[sub, 0], pc[sub, 1], c=lab[sub], cmap="tab10", vmin=-0.5, vmax=9.5, s=1.5, alpha=0.6, linewidths=0)
    for sp in ax.spines.values():
        sp.set_color("0.6")
    ax.tick_params(colors="0.7", labelsize=7)
    ax.set_xlabel("PC1", color="0.8", fontsize=8)
    ax.set_ylabel("PC2", color="0.8", fontsize=8)
    ax.text(0.02, 0.97, title, transform=ax.transAxes, color="white", fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)
    return path
