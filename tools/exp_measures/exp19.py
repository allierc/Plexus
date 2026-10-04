"""exp19's rulers: exp16's field law on WHOLISTIC recording f338, read from its test.

    run_measure("exp19.forecast", "training/wholistic/wl_gc_base")
    run_measure("exp19.rollout",  "training/wholistic/wl_gc_base")

The trainer's test (`plexus.trainer._test_field`) scores the held-out volumes at h = 1, 3, 6 only
(hard-coded there) and writes one free rollout. exp19's target reads h = 1..10, so `horizons()` scores
the kept checkpoint at every h with the trainer's OWN forecaster and the same scorer
(`field_recording.score`), once, into `results/<name>_horizons.json`; nothing else is recomputed.

Every skill is read against the BEST WINDOW MEAN of exp19's `data/baselines.json` (each voxel's mean
over its last W = 1..6 volumes, W chosen on validation; `tools/wholistic_baselines.py`), measured on the
same recording, targets and voxels:

    skill_h{h}            1 - MSE_model(h) / MSE_baseline(h)
    skill_short           mean of skill_h over h 1-3          (G-short)
    skill_long            mean of skill_h over h 5-10         (G-long)
    identity_gap          max over the trainer's h of |RMSE_model - RMSE_persistence|   (G-identity)
    *_gcamp, skill_long_gap   on a GCaMP run: its skills, and skill_long minus the skill_long of its
                          mCherry TWIN (the same spec name + `_mc`, trained on the calcium-blind channel);
                          absent until the twin has landed                            (G-calcium)

    exp19.rollout: frac_beats_hold (the free rollout's steps whose RMSE is <= holding its origin
    volume, G-stable), finite (cap), and band_power_ratio (G-rhythm): over the fish, the power of the
    free rollout's per-voxel time course in the 1-5 min band and in the 6.7-20 s band (Ruetten 2026
    Fig. 3c; the fast end is the ~0.3 Hz Nyquist limit), each over the recording's power in the same
    band on the same voxels, the smaller of the two capped at 1. Whole fish until the midline and
    muscle labels exist (exp19 G-organs).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .common import finite, register_run

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASELINES = os.path.join(ROOT, "experiments", "exp19_wholistic_graphcast", "data", "baselines.json")
SHORT, LONG = (1, 2, 3), (5, 6, 7, 8, 9, 10)
BANDS_S = {"midline_1_5min": (60.0, 300.0), "muscle_6.7_20s": (6.7, 20.0)}


def _channel(T):
    ref = T.spec.get("task", {}).get("reference", {}).get("field_recording", "")
    for ch in ("gcamp", "mcherry"):
        if ref.endswith("_" + ch):
            return ch
    raise ValueError(f"{T.dir}: reference {ref!r} is neither a _gcamp nor a _mcherry recording")


def horizons(T, hs=tuple(range(1, 11)), device=None):
    """MSE of the kept checkpoint and of persistence at every h in `hs` on the test volumes, through the
    trainer's own setup, forecaster and scorer. Cached in results/<name>_horizons.json."""
    p = os.path.join(T.dir, "results", f"{T.name}_horizons.json")
    if os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime(os.path.join(T.dir, "models", "best.pt")):
        return json.load(open(p))
    import torch
    from plexus import engine, trainer
    from plexus.tasks import field_recording as FR
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = trainer.load(T.name)
    engine.quiet(True)
    ck = torch.load(os.path.join(T.dir, "models", "best.pt"), weights_only=False, map_location=device)
    learn = trainer.Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = trainer._field_setup(spec, device)
    rec, te = box["rec"], box["test"]
    f = trainer._field_forecaster(trainer._field_sims(spec, hs, train=False), learn, box, device)
    out = {"name": T.name, "test_volumes": [int(te[0]) + 1, int(te[-1]) + 1], "h": {}}
    for h in hs:
        m, q = FR.score(f, rec, te, h), FR.score(FR.persistence(rec), rec, te, h)
        out["h"][str(h)] = {"mse": m["mse"], "persistence_mse": q["mse"], "n_vox": m["n_vox"], "n_pairs": m["n_pairs"]}
    json.dump(out, open(p, "w"), indent=1)
    return out


def _skills(T, base):
    hz = horizons(T)["h"]
    b = base[_channel(T)]["h"]
    sk = {int(h): 1 - r["mse"] / b[h]["baseline_mse"] for h, r in hz.items() if h in b}
    out = {f"skill_h{h}": finite(v) for h, v in sorted(sk.items())}
    out["skill_short"] = finite(np.mean([sk[h] for h in SHORT if h in sk]))
    out["skill_long"] = finite(np.mean([sk[h] for h in LONG if h in sk]))
    return out


def forecast(T, baselines=BASELINES, twin_suffix="_mc"):
    res = T.results.get(f"{T.name}_test")
    if res is None:
        raise ValueError(f"{T.dir}: no results/{T.name}_test.json -- the run has not been tested")
    base = json.load(open(baselines))
    out = {"identity_gap": finite(max(abs(r["rmse"] - r["persistence_rmse"]) for r in res["horizons"].values()))}
    out.update(_skills(T, base))
    ch = _channel(T)
    out["channel"] = ch
    if ch == "gcamp":
        out["skill_short_gcamp"], out["skill_long_gcamp"] = out["skill_short"], out["skill_long"]
        twin = os.path.join(os.path.dirname(T.dir.rstrip("/")), T.name + twin_suffix)
        if os.path.exists(os.path.join(twin, "results", f"{T.name}{twin_suffix}_test.json")):
            from .common import TrainingRun
            tw = _skills(TrainingRun(twin), base)
            out["skill_long_mcherry"] = tw["skill_long"]
            out["skill_long_gap"] = finite(out["skill_long"] - tw["skill_long"])
    return out


register_run("exp19.forecast", forecast, doc="exp19: the field law's held-out forecast of f338, skill per horizon "
             "1-10 against the best window mean, short / long, and the GCaMP-minus-mCherry long-range gap")


def band_power(x, dt_s, band_s):
    """Power of x [T, N] (each column a voxel's time course, its mean removed) in periods band_s (s)."""
    x = x - x.mean(0, keepdims=True)
    f = np.fft.rfftfreq(x.shape[0], dt_s)
    with np.errstate(divide="ignore"):
        per = np.where(f > 0, 1.0 / f, np.inf)
    sel = (per >= band_s[0]) & (per <= band_s[1])
    return float((np.abs(np.fft.rfft(x, axis=0)[sel]) ** 2).sum()), int(sel.sum())


def rollout(T):
    res = T.results.get(f"{T.name}_test")
    if res is None:
        raise ValueError(f"{T.dir}: no results/{T.name}_test.json -- the run has not been tested")
    fr = res["free"]
    m, p = np.array(fr["rmse"], float), np.array(fr["persistence_rmse"], float)
    out = {"finite": float(np.isfinite(m).all()), "frac_beats_hold": finite(np.mean(m <= p)),
           "free_steps": int(len(m))}
    z = np.load(os.path.join(T.dir, "results", f"{T.name}_free.npz"))
    pred, o = z["pred"].astype(np.float32), int(z["origin"])
    from plexus.tasks import field_recording as FR
    ref = T.spec["task"]["reference"]
    rec = FR.coarsen(FR.load(ref["field_recording"]), int(ref.get("coarsen", 1)))
    n = pred.shape[0]
    obs = rec["ratio"][o + 1:o + 1 + n]
    vox = rec["mask"][o] & rec["mask"][o + 1:o + 1 + n].all(0)
    dt = float(np.median(np.diff(rec["t_s"])))
    ratios = {}
    for k, band in BANDS_S.items():
        pm, nb = band_power(pred[:, vox], dt, band)
        po, _ = band_power(obs[:, vox], dt, band)
        ratios[k] = pm / po if po > 0 and nb > 0 else None
        out[f"band_ratio_{k}"] = finite(ratios[k]) if ratios[k] is not None else None
        out[f"band_bins_{k}"] = nb
    ok = [r for r in ratios.values() if r is not None]
    out["band_power_ratio"] = finite(min(1.0, min(ok))) if len(ok) == len(BANDS_S) else None
    return out


register_run("exp19.rollout", rollout, doc="exp19: the free rollout over the held-out window -- finite, steps "
             "beating the held origin, band power at the paper's two rhythms against the recording's")
