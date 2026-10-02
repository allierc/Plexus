"""exp20's rulers: exp17's trace law on a fish of Chen, James, Ruetten et al. 2026, read from its test and from a
trial scorer that runs the law's OWN forecaster (`plexus.trainer._trace_rollout`) over each held-out trial.

    run_measure("exp20.forecast", "training/gutbrain/gb_ng_base")
    run_measure("exp20.trial",    "training/gutbrain/gb_ng_base")

The trainer's test (`_test_trace`, `split: recording`) scores 32-step forecasts from every origin whose forecast frames
lie in a held-out window, against the best mean baseline and persistence, and one free rollout of the whole session:

    forecast   skill_short, skill_long        1 - MSE_model / MSE_mean over h 1-3 and 16-32, held-out windows
               identity_gap                   max_h |MSE_model - MSE_persistence| / MSE_persistence
    free       r2_raw, r2_denoised, finite    the free rollout's R^2 per frame over cells, mean

A TRIAL is scored from o = onset (the pulse volume, replaced by the one before by the exporter) through the 55-s window (the exporter's
`window_volumes`), on the gut-responsive cells of data/baselines_<recording>_cells.npz, against the stimulus-triggered
mean of data/baselines_<recording>.json (`tools/gutbrain_baselines.py`, the same windows and cells). Cached in
results/<name>_trials.json (re-made when the checkpoint is newer):

    trial      skill_gut         1 - sum MSE_law / sum MSE_STA over the held-out gut trials' windows   (G-trial)
               skill_gut_all     the same over every cell
               r_law, r_rec      the held-out control pulse's evoked change (the law's forecast, the recording) over the
                                 RECORDED gut pulses' -- both over the same denominator: a ratio of the law's own control
                                 and gut responses is meaningless when the law forecasts no gut response (the smoke run:
                                 -0.003 / -0.026 read 0.13, the recording's ratio, by coincidence)
               ctrl_excess       max(0, r_law - r_rec)                                              (G-control)
               lglucose_fraction the L-glucose twin's (name + `_lglu`) forecast evoked change over this law's, each on
                                 its own fish's REPLICABLE cells (baselines `top1`: the top 1 % by positive t of the
                                 training gut pulses)                                                (G-Lglucose)
    integration gvgm_ratio_rel   (G-region) None until the region boxes exist
"""
from __future__ import annotations

import json
import os

import numpy as np

from .common import finite, register_run

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast", "data")
EVOKED_S, PRE_S = 20.0, 10.0
CTRL_SITE = 1


# ----------------------------------------------------------------------------- pure arithmetic (planted tests)
def window_skill(mse_law, mse_ref) -> float:
    """1 - sum(law) / sum(ref) over every (trial, step) entry: the two lists are per trial, per step."""
    a = float(np.sum([np.sum(m) for m in mse_law]))
    b = float(np.sum([np.sum(m) for m in mse_ref]))
    return 1.0 - a / b if b > 0 else float("nan")


def evoked(window: np.ndarray, before: np.ndarray, n_ev: int) -> float:
    """The evoked change of a forecast: mean over cells of (mean of the first n_ev forecast frames - mean of the
    recorded frames before the pulse). window [H, N], before [P, N]."""
    return float((window[:n_ev].mean(0) - before.mean(0)).mean())


def ctrl_excess(r_law: float, r_rec: float) -> float:
    return max(0.0, float(r_law) - float(r_rec))


def gvgm_rel(law: dict, rec: dict) -> float:
    """log(midbrain GV/GM / hindbrain GV/GM) of the law over the same of the recording; dicts {'mid': x, 'hind': y}."""
    a, b = np.log(law["mid"] / law["hind"]), np.log(rec["mid"] / rec["hind"])
    return float(a / b) if b != 0 else float("nan")


# ----------------------------------------------------------------------------- the trial scorer
def _recording(T):
    return T.spec["task"]["reference"]["trace_recording"]


def trials(T, device=None):
    p = os.path.join(T.dir, "results", f"{T.name}_trials.json")
    ck_p = os.path.join(T.dir, "models", "best.pt")
    if os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime(ck_p):
        return json.load(open(p))
    import torch
    from plexus import engine, trainer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = trainer.load(T.name)
    engine.quiet(True)
    ck = torch.load(ck_p, weights_only=False, map_location=device)
    learn = trainer.Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = trainer._trace_setup(spec, device)
    X, rec = box["X"], box["rec"]
    rname = _recording(T)
    base = json.load(open(os.path.join(DATA, f"baselines_{rname}.json")))
    cz = np.load(os.path.join(DATA, f"baselines_{rname}_cells.npz"))
    resp = torch.as_tensor(np.where(cz["responsive"])[0], device=X.device)
    top1 = torch.as_tensor(np.where(cz["top1"])[0], device=X.device)
    pre, post = base["window"]
    n_ev = int(round(EVOKED_S / base["frame_s"]))
    sim = trainer._trace_sim(spec, False, post)
    out = {"name": T.name, "recording": rname, "window": [pre, post], "trials": []}
    with torch.no_grad():
        for b in base["trials"]:
            o = b["onset"]                       # the pulse volume (tools/gutbrain_baselines.py: the same origin)
            pred = trainer._trace_rollout(sim, learn, spec, box, o, post, device, False)       # [post, N]
            tgt = X[o + 1:o + post + 1]
            mse = lambda sel: (((pred - tgt)[:, sel]) ** 2).double().mean(1).cpu().numpy().tolist()
            before = X[max(0, b["onset"] - pre):b["onset"]]
            out["trials"].append({"onset": b["onset"], "kind": b["kind"], "site": b["site"],
                                  "mse_resp": mse(resp), "mse_all": mse(slice(None)),
                                  "evoked_law": evoked(pred[:, resp].cpu().numpy(), before[:, resp].cpu().numpy(), n_ev),
                                  "evoked_rec": b["evoked_resp"],
                                  "evoked_law_top1": evoked(pred[:, top1].cpu().numpy(), before[:, top1].cpu().numpy(), n_ev),
                                  "evoked_rec_top1": b["evoked_top1"],
                                  "mse_sta_resp": b["mse_sta_resp"], "mse_sta_all": b["mse_sta_all"]})
    json.dump(out, open(p, "w"), indent=1)
    return out


def trial(T):
    r = trials(T)
    g = [t for t in r["trials"] if t["kind"] == "gut"]
    c = [t for t in r["trials"] if t["kind"] == "control"]
    res = {"skill_gut": finite(window_skill([t["mse_resp"] for t in g], [t["mse_sta_resp"] for t in g])),
           "skill_gut_all": finite(window_skill([t["mse_all"] for t in g], [t["mse_sta_all"] for t in g])),
           "n_gut_trials": len(g), "n_ctrl_trials": len(c)}
    if g and c:
        g_rec = np.mean([t["evoked_rec"] for t in g])
        r_law = np.mean([t["evoked_law"] for t in c]) / g_rec
        r_rec = np.mean([t["evoked_rec"] for t in c]) / g_rec
        res.update({"r_law": finite(r_law), "r_rec": finite(r_rec), "ctrl_excess": finite(ctrl_excess(r_law, r_rec))})
    twin = os.path.join(os.path.dirname(T.dir.rstrip("/")), T.name + "_lglu")
    if os.path.exists(os.path.join(twin, "models", "best.pt")):
        from .common import TrainingRun
        tw = trials(TrainingRun(twin))
        ev_tw = np.mean([t["evoked_law_top1"] for t in tw["trials"] if t["kind"] == "gut"])
        ev_me = np.mean([t["evoked_law_top1"] for t in g])
        res["lglucose_fraction"] = finite(ev_tw / ev_me if ev_me else float("nan"))
    return res


def forecast(T):
    r = T.results.get(f"{T.name}_test")
    if r is None or r.get("mode") != "trace":
        raise ValueError(f"{T.name}: no trace test (results/{T.name}_test.json, mode 'trace')")
    out = {"skill_short": finite(r["skill_short"]), "skill_long": finite(r["skill_long"]),
           "identity_gap": finite(r["identity_gap"]), "origins": int(r["origins"])}
    for h in (1, 2, 3, 8, 16, 24, 32):
        out[f"skill_h{h}"] = finite(r["skill"][h - 1])
    return out


def free(T):
    f = T.results[f"{T.name}_test"]["free"]
    return {"r2_raw": finite(f["r2_raw"]), "r2_denoised": finite(f["r2_denoised"]), "finite": float(f["finite"])}


def integration(T):
    return {"gvgm_ratio_rel": None}           # the region boxes of the fish are not drawn yet (gates.yaml G-region)


register_run("exp20.forecast", forecast, doc="exp20: skill over the mean baseline on the held-out windows, short / long")
register_run("exp20.trial", trial, doc="exp20: held-out gut trials against the stimulus-triggered mean; control excess")
register_run("exp20.free", free, doc="exp20: the free rollout of the whole session")
register_run("exp20.integration", integration, doc="exp20: midbrain / hindbrain GV/GM of the law's input ablations")
