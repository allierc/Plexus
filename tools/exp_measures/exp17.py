"""exp17's rulers: a GraphCast law on ZAPBench, read from its trace test (`plexus.trainer._test_trace`).

    run_measure("exp17.forecast", "training/zapbench/zap_gc_base")

A training run has no trajectory; its test writes `results/<name>_test.json` (mode "trace"): per step ahead
h = 1..32, the MSE of the law, of the best mean baseline (each neuron's mean over its last W frames, W = 1..6,
best per h), of persistence and of the stimulus lookup, per condition and in the grand average, and one FREE
rollout of the whole recording. These rulers turn that file into the numbers the gates read; they recompute
nothing the trainer scored (STANDARD MSE, Cedric 2026-09-30).

    forecast   skill_short, skill_long         1 - MSE_model / MSE_mean over h 1-3 and 16-32, grand average
               skill_long_minus_lookup         skill_long minus the stimulus lookup's
               n_conditions_long_positive      conditions (of 9) with skill_long > 0
               identity_gap                    max_h |MSE_model - MSE_persistence| / MSE_persistence
               skill_h{h}                      per step, grand average
    free       r2_raw, r2_denoised (+ _sd), finite   the free rollout's R^2 per frame over neurons, mean and SD
    embedding  ami_tuning                      clusters of the embedding against the neurons' stimulus tuning

THE NETWORK QUESTION (Cedric, 2026-10-10: does the trained graph carry network dynamics, or relay the stimulus from
the input neurons?). Three more rulers, for the gates of that question:

    steady     pn_r, pn_sd, n, n_flat, bm_r    the free rollout's per-neuron r and brain-mean r on the STEADY part
               pn_r_input, n_input,            (block 1 of 24, ~5 min, the transient from the recorded start, left
               pn_r_other, n_other,            out), as tools/exp17_meanfield_stats.py scores them: per neuron, the
               n_flat_other                    learned and recorded traces over the movie's frames, each regressed
                                               on its own brain mean, then correlated; ONE neuron set fixed by the
                                               recording alone (recorded residual moving), so two runs' values are
                                               on the same neurons and their difference is a paired one; a flat or
                                               non-finite learned residual scored 0. Split by the input mask (the
                                               run's clamp_in mask, else input_mask_destripe_bal20's `mask`).
                                               kw: rollout ("" the nominal, "_Wall0", "_W0", ...), input_mask, input_array
    clamp      relay_gap_pn (+ _p),            data/clamp_<run>.json (tools/exp17_clamp.py), the free neurons only:
               plain_gap_pn (+ _p),            clamp_in - clamp_in_W0 (what W carries to the free neurons when the
               full_pn, w0_pn, clamp_in_pn,    input neurons are given their recorded traces), full - W = 0, each
               clamp_in_w0_pn, identity_max_abs,  rollout's per-neuron r; identity_max_abs = max |W = 0 - clamp_in_W0|
               n_free, beyond_relay_share      on the free neurons (must be 0: with W = 0 the clamp reaches nobody);
                                               beyond_relay_share = (clamp_in_cut - clamp_in_W0) / relay gap, None
                                               until the file carries a `clamp_in_cut` rollout (input->free edges 0)
    spectrum   abscissa, leak_abscissa,        data/jacobian_<run>.json (tools/exp17_jacobian.py): the linearised law's
               n_slow, n_unstable, max_abs_im,  50 rightmost eigenvalues against the leak-only spectrum (W = 0: -1/tau_i);
               n_slow_brainwide,               n_slow = those slower than the slowest leak; n_slow_brainwide = of them,
               impulse_outside_share           the modes whose mass spreads over at least half the uniform spread
                                               (mode_spread_um >= 0.5 uniform_spread_um); a phase law: the max over blocks
Every key of clamp and spectrum is None until its file exists: a gate reading it scores "no value" (pending), never 0.
Their answers change when the file is rewritten without the run's own test json changing, so their CHANGED time is
the newest such file's (the scorer re-measures every cached value older than it).
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np

from .common import CHANGED, finite, register_run

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast", "data")
N_BLOCKS = 24                                   # the free rollout cut into 24 equal blocks (~5 min over the 2 h)
DEFAULT_MASK = ("zebrafish/input_mask_destripe_bal20.npz", "mask")
BRAINWIDE = 0.5                                 # a mode is brain-wide when its spread >= this x the uniform spread
_REC: dict = {}


def _test(T):
    res = T.results.get(f"{T.name}_test")
    if res is None or res.get("mode") != "trace":
        raise ValueError(f"{T.name}: no trace test (results/{T.name}_test.json, mode 'trace')")
    return res


def forecast(T):
    r = _test(T)
    out = {"skill_short": finite(r["skill_short"]), "skill_long": finite(r["skill_long"]),
           "skill_long_minus_lookup": finite(r["skill_long_minus_lookup"]),
           "lookup_skill_long": finite(r["lookup_skill_long"]),
           "n_conditions_long_positive": int(r["n_conditions_long_positive"]),
           "identity_gap": finite(r["identity_gap"])}
    for h in (1, 2, 3, 8, 16, 24, 32):
        out[f"skill_h{h}"] = finite(r["skill"][h - 1])
    for name, v in zip(r["names"], r["skill_long_by_condition"]):
        out[f"skill_long_{name.replace(' ', '_')}"] = finite(v)
    return out


def free(T):
    f = _test(T)["free"]
    return {"r2_raw": finite(f["r2_raw"]), "r2_raw_sd": finite(f["r2_raw_sd"]),
            "r2_denoised": finite(f["r2_denoised"]), "r2_denoised_sd": finite(f["r2_denoised_sd"]),
            "finite": float(f["finite"])}


def embedding(T):
    c = T.results.get(f"{T.name}_clusters") or {}
    # None, not 0, when the law learns no embedding (the known ODE writes `ami_tuning_params`, its constants' AMI,
    # which is kept apart as circular) or the plot phase has not run: 0 is the gate's zero line and would read as
    # "the embedding carries no tuning" (review, 2026-09-30)
    return {"ami_tuning": finite(c.get("ami_tuning"))}


# ---------------------------------------------------------------------------------- the network question (2026-10-10)
def _recording(name):
    """The recording's dF/F [frames, neurons], loaded once per process (tests plant their own here)."""
    if name not in _REC:
        from plexus.tasks import trace_recording as TR
        _REC[name] = np.asarray(TR.load(name)["dff"])
    return _REC[name]


def _input_mask(T, input_mask, input_array):
    """The input neurons, bool [N]: kw, else the run's clamp_in rollout's mask, else bal20's `mask` (any column)."""
    if input_mask is None:
        ro = {r.get("name"): r for r in ((T.spec.get("task") or {}).get("rollouts") or [])}
        c = (ro.get("clamp_in") or {}).get("clamp") or {}
        input_mask, input_array = c.get("mask", DEFAULT_MASK[0]), c.get("array", input_array or DEFAULT_MASK[1])
    if not os.path.isabs(input_mask):
        from plexus.paths import graphs_data_path
        input_mask = graphs_data_path(*input_mask.split("/", 1))
    m = np.asarray(np.load(input_mask)[input_array or DEFAULT_MASK[1]]).astype(bool)
    return m.any(1) if m.ndim > 1 else m


def _local_r(P, X, keep):
    """Per neuron of `keep`, r of the learned and recorded traces [frames, N] after each is regressed on its own brain
    mean (the mean over the finite learned neurons, and the recorded over the same ones), as
    exp17_meanfield_stats.local_from; a flat (residual variance <= 1e-12) or non-finite learned trace scores 0."""
    fin = np.isfinite(P).all(0)
    ap = (P[:, fin] - P[:, fin].mean(0)).mean(1)
    ax = (X[:, fin] - X[:, fin].mean(0)).mean(1)
    ap, ax = ap - ap.mean(), ax - ax.mean()
    r = np.zeros(int(keep.sum()))
    ok = fin[keep]
    Pk = P[:, keep][:, ok]
    Xk = X[:, keep][:, ok]
    Pk, Xk = Pk - Pk.mean(0), Xk - Xk.mean(0)
    Rp = Pk - np.outer(ap, (Pk * ap[:, None]).mean(0) / max(float((ap * ap).mean()), 1e-30))
    Rx = Xk - np.outer(ax, (Xk * ax[:, None]).mean(0) / max(float((ax * ax).mean()), 1e-30))
    vp, vx = (Rp * Rp).mean(0), (Rx * Rx).mean(0)
    live = (vp > 1e-12) & (vx > 1e-12)
    rk = np.zeros(Pk.shape[1])
    rk[live] = np.clip((Rp[:, live] * Rx[:, live]).mean(0) / np.sqrt(vp[live] * vx[live]), -1.0, 1.0)
    r[ok] = rk
    flat = np.ones(int(keep.sum()), bool)
    flat[np.flatnonzero(ok)[live]] = False
    return r, flat


def steady(T, rollout="", input_mask=None, input_array=None):
    z = np.load(os.path.join(T.dir, "results", f"{T.name}{rollout}_movie.npz"))
    fr = np.asarray(z["frames"]).astype(int)
    f0, f1 = int(fr[0]), int(fr[-1])
    blk = lambda f: np.minimum((np.asarray(f) - f0) * N_BLOCKS // (f1 - f0 + 1), N_BLOCKS - 1)   # noqa: E731
    st = blk(fr) > 0
    X = np.asarray(_recording(T.spec["task"]["reference"]["trace_recording"])[fr[st]], np.float64)
    P = np.asarray(z["pred"], np.float64)[st]
    Xc = X - X.mean(0)                           # THE NEURON SET, from the recording alone: recorded residual moving
    b = Xc.mean(1)
    keep = (Xc - np.outer(b, (Xc * b[:, None]).sum(0) / (b * b).sum())).std(0) > 1e-9
    r, flat = _local_r(P, X, keep)
    o, p = np.asarray(z["mean_obs_all"], np.float64), np.asarray(z["mean_pred_all"], np.float64)
    ok = (blk(np.arange(len(o)) + f0) > 0) & np.isfinite(o) & np.isfinite(p)
    out = {"pn_r": finite(r.mean()), "pn_sd": finite(r.std()), "n": int(keep.sum()), "n_flat": int(flat.sum()),
           "bm_r": finite(np.corrcoef(o[ok], p[ok])[0, 1]), "frames_steady": int(st.sum()), "free_frames_steady": int(ok.sum())}
    inp = _input_mask(T, input_mask, input_array)[keep]
    for lab, m in (("input", inp), ("other", ~inp)):
        out[f"pn_r_{lab}"] = finite(r[m].mean()) if m.any() else None
        out[f"n_{lab}"] = int(m.sum())
        out[f"n_flat_{lab}"] = int(flat[m].sum())
    return out


def _data_file(stem):
    f = os.path.join(DATA, stem)
    return json.load(open(f)) if os.path.exists(f) else None


def clamp(T):
    keys = ("relay_gap_pn", "relay_gap_p", "plain_gap_pn", "plain_gap_p", "full_pn", "w0_pn", "clamp_in_pn",
            "clamp_in_w0_pn", "identity_max_abs", "n_free", "beyond_relay_share")
    d = _data_file(f"clamp_{T.name}.json")
    if d is None:
        return dict.fromkeys(keys)
    laws = d.get("laws") or {}
    pn = lambda k: finite(((laws.get(k) or {}).get("local_r") or {}).get("estimate"))   # noqa: E731

    def test(a, b):
        t = next((t for t in d.get("tests") or [] if t.get("a") == a and t.get("b") == b), None)
        return (finite(t["local_r"]["difference"]), finite(t["local_r"]["p"])) if t else (None, None)
    relay, relay_p = test("clamp_in", "clamp_in, W = 0")
    plain, plain_p = test("full", "W = 0")
    cut, w0c = pn("clamp_in_cut"), pn("clamp_in, W = 0")
    return {"relay_gap_pn": relay, "relay_gap_p": relay_p, "plain_gap_pn": plain, "plain_gap_p": plain_p,
            "full_pn": pn("full"), "w0_pn": pn("W = 0"), "clamp_in_pn": pn("clamp_in"), "clamp_in_w0_pn": w0c,
            "identity_max_abs": finite(d.get("max_abs_w0_minus_clamp_in_w0")), "n_free": d.get("n_free_scored"),
            "beyond_relay_share": (finite((cut - w0c) / relay) if None not in (cut, w0c, relay) and relay else None)}


def _spectrum_one(J, slow_rate, uniform):
    re = np.asarray(J.get("eig_re") or [], float)
    spread = np.asarray(J.get("mode_spread_um") or [np.nan] * len(re), float)
    slow = re > -slow_rate
    u = J.get("uniform_spread_um", uniform)
    wide = slow & (spread >= BRAINWIDE * u) if u else np.zeros_like(slow)
    return {"abscissa": J.get("abscissa"), "n_slow": int(slow.sum()), "n_unstable": int((re > 0).sum()),
            "max_abs_im": J.get("max_abs_im", J.get("max_abs_im_LR")), "n_slow_brainwide": int(wide.sum()),
            "impulse_outside_share": (J.get("impulse") or {}).get("outside_share_all")}


def spectrum(T):
    keys = ("abscissa", "leak_abscissa", "n_slow", "n_unstable", "max_abs_im", "n_slow_brainwide", "impulse_outside_share")
    d = _data_file(f"jacobian_{T.name}.json")
    if d is None:
        return dict.fromkeys(keys)
    leak = d.get("leak") or {}
    slow_rate = float(leak["slowest_rate_per_s"])
    Js = d.get("per_block") or ([d["J"]] if "J" in d else [])
    one = [_spectrum_one(J, slow_rate, d.get("uniform_spread_um")) for J in Js]
    if not one:
        return dict.fromkeys(keys)
    out = {"leak_abscissa": finite(leak.get("abscissa", -slow_rate))}
    for k in ("abscissa", "n_slow", "n_unstable", "max_abs_im", "n_slow_brainwide", "impulse_outside_share"):
        v = [o[k] for o in one if o[k] is not None]
        out[k] = (int(max(v)) if k.startswith("n_") else finite(max(v))) if v else None
    return out


def _newest(pattern):
    """The minute after the newest data file matching `pattern` was written (the CHANGED format), or None."""
    fs = glob.glob(os.path.join(DATA, pattern))
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(max(map(os.path.getmtime, fs)) + 60)) if fs else None


for _m, _pat in (("exp17.clamp", "clamp_*.json"), ("exp17.spectrum", "jacobian_*.json")):
    if _newest(_pat):
        CHANGED[_m] = max(CHANGED.get(_m, ""), _newest(_pat))


register_run("exp17.forecast", forecast, doc="skill over the mean baseline, short and long, per condition")
register_run("exp17.free", free, doc="the free rollout of the whole recording, R2 raw and denoised")
register_run("exp17.embedding", embedding, doc="the embedding's clusters against the neurons' stimulus tuning")
register_run("exp17.steady", steady, doc="per-neuron r and brain-mean r on the steady part, input / other split")
register_run("exp17.clamp", clamp, doc="the input-neuron clamp: what W carries to the free neurons")
register_run("exp17.spectrum", spectrum, doc="the linearised law's slow modes against the leak-only spectrum")
