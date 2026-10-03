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
"""
from __future__ import annotations

from .common import finite, register_run


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


register_run("exp17.forecast", forecast, doc="skill over the mean baseline, short and long, per condition")
register_run("exp17.free", free, doc="the free rollout of the whole recording, R2 raw and denoised")
register_run("exp17.embedding", embedding, doc="the embedding's clusters against the neurons' stimulus tuning")
