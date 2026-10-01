"""exp16's rulers: a trained field law's held-out forecast of the redox organoid, read from its test.

    run_measure("exp16.forecast", "training/redox/hlo_gc_l2h16")

A training run has no trajectory; its test (`plexus.trainer._test_field`) writes
`results/<name>_test.json`: per horizon h in {1, 3, 6} the model's RMSE and persistence's on the test
volumes (both `plexus.tasks.field_recording.score`), and one FREE rollout from the last volume before
the test window through all of it. This ruler turns that file into the numbers the gates read; it
recomputes nothing the trainer scored, and it reads the washout target from the frozen recording's
copy of the lab's `Development_Time_Trend.xlsx`, so no target is typed.

    rmse_h{h}, persistence_h{h}   RMSE of the ratio on the tissue, model and persistence
    skill_h{h}                    1 - MSE_model / MSE_persistence
    identity_gap                  max over h of |rmse - persistence| -- 0 for the untrained law
    slope_model_per_h             least-squares slope of the free rollout's organoid-mean ratio, per hour
    slope_recorded_per_h          the same, of the recording on the same voxels
    xlsx_slope_per_h              the lab's trend over the test volumes (mean NADH / mean FAD), per hour
    xlsx_slope_se_per_h           its standard error, from the table's own volume-to-volume jitter
    slope_error_per_h             |slope_model - xlsx_slope|
    free_frac_beats_persistence   fraction of the free rollout's steps whose RMSE is <= persistence's
    free_rmse_ratio_last          the free rollout's RMSE over persistence's, at its last step
"""
from __future__ import annotations

import numpy as np

from .common import finite, register_run


def xlsx_slope(recording="hlo_washout", test_volumes=(55, 69)):
    """The lab's trend over the test volumes: slope per hour and its standard error.

    The table's jitter is read from its second differences over ALL volumes: for white noise of
    standard deviation s on a smooth trend, sd(second difference) = s sqrt(6). The slope's standard
    error is then s / sqrt(sum (t - tbar)^2) over the test volumes' real times."""
    from plexus.tasks import field_recording as FR
    rec = FR.load(recording)
    y, t = rec["trend_ratio"], rec["t_s"] / 3600.0
    a, b = test_volumes[0] - 1, test_volumes[1]
    slope = np.polyfit(t[a:b], y[a:b], 1)[0]
    s = np.std(np.diff(y, 2)) / np.sqrt(6)
    se = s / np.sqrt(((t[a:b] - t[a:b].mean()) ** 2).sum())
    return float(slope), float(se), float(s)


def forecast(T, recording=None, step_s=600.0):
    res = T.results.get(f"{T.name}_test")
    if res is None:
        raise ValueError(f"{T.dir}: no results/{T.name}_test.json -- the run has not been tested")
    out = {}
    gaps = []
    for h, r in sorted(res["horizons"].items(), key=lambda kv: int(kv[0])):
        out[f"rmse_h{h}"] = finite(r["rmse"])
        out[f"persistence_h{h}"] = finite(r["persistence_rmse"])
        out[f"skill_h{h}"] = finite(r["skill"])
        gaps.append(abs(r["rmse"] - r["persistence_rmse"]))
    out["identity_gap"] = finite(max(gaps))
    fr = res["free"]
    n = len(fr["rmse"])
    t = np.arange(1, n + 1) * step_s / 3600.0
    out["slope_model_per_h"] = finite(np.polyfit(t, fr["organoid_mean_model"], 1)[0])
    out["slope_recorded_per_h"] = finite(np.polyfit(t, fr["organoid_mean_recorded"], 1)[0])
    rec_name = recording or T.spec.get("task", {}).get("reference", {}).get("field_recording", "hlo_washout")
    slope, se, jitter = xlsx_slope(rec_name, tuple(res["test_volumes"]))
    out["xlsx_slope_per_h"], out["xlsx_slope_se_per_h"], out["xlsx_jitter"] = slope, se, jitter
    out["slope_error_per_h"] = finite(abs(out["slope_model_per_h"] - slope))
    m, p = np.array(fr["rmse"]), np.array(fr["persistence_rmse"])
    out["free_frac_beats_persistence"] = finite(np.mean(m <= p))
    out["free_rmse_ratio_last"] = finite(m[-1] / p[-1])
    out["free_finite"] = float(np.isfinite(m).all())
    return out


register_run("exp16.forecast", forecast, doc="exp16: a trained field law's held-out forecast of the redox organoid "
             "(RMSE and skill per horizon, the free rollout's washout slope against the lab's trend)")


def embedding(T, recording=None):
    """The learned per-voxel embedding (`results/<name>_embedding.npz`, written by analyse): whether it
    has structure at the scale of a cell.

        corr_len_um      the lag, in um in x-y, at which the tissue voxels' embedding (its first
                         principal component, mean removed) correlates with itself at 1/e -- a cell-sized
                         domain gives a length near a cell's radius; noise gives about one voxel
        pc_var           the variance fraction of the first three principal components
        spread           the standard deviation of PC1 on the tissue, in embedding units
    """
    import os
    from plexus.tasks import field_recording as FR
    f = os.path.join(T.dir, "results", f"{T.name}_embedding.npz")
    if not os.path.exists(f):
        raise ValueError(f"{f} missing -- the run has no embedding, or analyse has not run")
    a = np.load(f)["a"]                                                  # [k, Z, Y, X]
    ref = T.spec.get("task", {}).get("reference", {})
    rec = FR.coarsen(FR.load(recording or ref.get("field_recording", "hlo_washout")), int(ref.get("coarsen", 1)))
    tr, _ = FR.split(rec, int(ref.get("n_test", 15)))
    m = rec["mask"][int(tr[-1]) - int(ref.get("n_val", 6))]
    v = a[:, m].T
    v = v - v.mean(0)
    _, S, Vt = np.linalg.svd(v[:: max(1, len(v) // 50000)], full_matrices=False)
    pc1 = np.zeros(m.shape)
    pc1[m] = v @ Vt[0]
    lags, corr = [], []
    for L in range(0, 13):
        num = den = 0.0
        for ax in (1, 2):
            sl0 = [slice(None)] * 3
            sl1 = [slice(None)] * 3
            sl0[ax], sl1[ax] = slice(0, m.shape[ax] - L), slice(L, None)
            both = m[tuple(sl0)] & m[tuple(sl1)]
            num += float((pc1[tuple(sl0)] * pc1[tuple(sl1)])[both].sum())
            den += float(((pc1[tuple(sl0)] ** 2 + pc1[tuple(sl1)] ** 2) / 2)[both].sum())
        lags.append(L)
        corr.append(num / max(den, 1e-12))
    corr = np.array(corr)
    below = np.where(corr < np.exp(-1))[0]
    if len(below):
        j = int(below[0])
        x0, x1, y0, y1 = lags[j - 1], lags[j], corr[j - 1], corr[j]
        L_e = x0 + (np.exp(-1) - y0) * (x1 - x0) / (y1 - y0)
    else:
        L_e = float(lags[-1])
    ev = S ** 2 / max((S ** 2).sum(), 1e-12)
    return {"corr_len_um": finite(L_e * rec["dx_um"]), "pc_var_1": finite(ev[0]),
            "pc_var_3": finite(ev[:3].sum()), "spread": finite(pc1[m].std()), "k": float(a.shape[0])}


register_run("exp16.embedding", embedding, doc="exp16: the learned per-voxel embedding's correlation length "
             "on the tissue (um) and its principal-component variance -- do cell-sized domains emerge?")


def fullfit(T):
    """PHASE 2 (Cedric, 2026-09-29): what the law needs to REPRODUCE the recording. Reads the full
    rollout from t0 (`plexus.trainer._test_field_full`, `results/<name>_test.json`, mode "full") and,
    when the run learned an embedding, its clusters (`results/<name>_clusters.json`).

        r2, r2_spatial        variance explained over the whole rollout; spatial = each frame's tissue
                              mean removed first (a law that tracks only the level scores 0 there)
        ceiling_r2(_spatial)  1 - noise variance / variance: what a perfect law of the signal reaches
        mean_rmse             the organoid-mean ratio's RMSE over the rollout, model against recording
        mean_sd               the recorded organoid mean's own standard deviation over the rollout
        frac_beats_hold       fraction of steps whose RMSE is <= holding the first volume
        finite                1 when every step is finite
        coherence_k8, domain_um3_k8, silhouette   the embedding's clusters (when there is one)
    """
    res = T.results.get(f"{T.name}_test")
    if res is None or res.get("mode") != "full":
        raise ValueError(f"{T.dir}: no full-rollout test (results/{T.name}_test.json, mode full)")
    m, p = np.array(res["rmse"]), np.array(res["persistence_rmse"])
    mm, mr = np.array(res["organoid_mean_model"]), np.array(res["organoid_mean_recorded"])
    out = {k: finite(res[k]) for k in ("r2", "r2_spatial", "persistence_r2", "persistence_r2_spatial",
                                       "ceiling_r2", "ceiling_r2_spatial", "noise_sigma")}
    for k in ("r2_denoised", "r2_spatial_denoised", "persistence_r2_denoised", "ceiling_r2_denoised"):
        out[k] = finite(res[k]) if k in res else None      # runs tested before 2026-09-29 16:30 lack it
    out.update(mean_rmse=finite(np.sqrt(np.mean((mm - mr) ** 2))), mean_sd=finite(mr.std()),
               frac_beats_hold=finite(np.mean(m <= p)), finite=float(np.isfinite(m).all()))
    cl = T.results.get(f"{T.name}_clusters")
    out["coherence_k8"] = finite(cl["coherence_k8"]) if cl else 0.0
    out["domain_um3_k8"] = finite(cl["domain_um3_median_k8"]) if cl else 0.0
    out["silhouette"] = finite(cl["silhouette"]) if cl else 0.0
    return out


register_run("exp16.fullfit", fullfit, doc="exp16 phase 2: variance explained by a full rollout from t0 over "
             "the whole recording, total and spatial, and the embedding's clusters")
