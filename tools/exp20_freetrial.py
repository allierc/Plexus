"""exp20: the held-out trials read INSIDE THE FREE ROLLOUT of the whole session -- the stimuli-only reading of the trial
gate (Cedric, 2026-10-03: "the inference only driven by stimuli"; "network dynamics, not a function of the stimulus").

    PYTHONPATH=src:tools python tools/exp20_freetrial.py <run> [<run> ...] [--device cuda:0] [--no-w0]

One free rollout of the whole session from the law's first frame (the trainer's `_trace_rollout_stream`, as its own
`_trace_free`), driven by the forcings only; a runaway cell (outside SILENCE dF/F or not finite) is frozen at its mean
over the TRAINING frames (not the trainer's all-frame mean, which reads held-out frames). At every held-out pulse
(the exporter's trials, column 6), over the fish's gut-responsive cells (baselines `responsive`):

    evoked     mean over 0-20 s after the onset minus the 10 s before, of the rollout and of the recording
    pattern_r  the correlation over those cells of the rollout's evoked change with the recording's
    and over the whole window (10 s before to 55 s after): the MSE of the rollout against the recording

for the full law and, for a neuron-graph law, the same rollout with every edge weight W = 0 (exp17_ablation's
`weights_zeroed`): what the network adds to the response, not to a whole-brain R2. Writes results/<run>_freetrial.json.
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
DATA = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast", "data")
SILENCE = (-1.0, 3.0)
PRE_S, EVOKED_S, POST_S = 10.0, 20.0, 55.0
CTRL_SITE = 1


def rollout_windows(T, spec, learn, box, device, windows):
    """The free rollout of the whole session; returns {onset: pred [pre + post + 1, N]} and the silenced count."""
    w = T._warmup(spec)
    o0 = box["n_in"] - 1 + w
    n = box["T"] - 1 - o0
    X = box["X"]
    lab = box.get("lab")
    tr = torch.as_tensor(lab == 0, device=X.device) if lab is not None else torch.ones(X.shape[0], dtype=torch.bool,
                                                                                         device=X.device)
    fill = X[tr].mean(0)
    dead = torch.zeros(X.shape[1], dtype=torch.bool, device=X.device)
    lo, hi = SILENCE

    def silence(blk):
        v = blk.reshape(-1, blk.shape[-1])
        bad = ~torch.isfinite(v).all(1) | (v < lo).any(1) | (v > hi).any(1)
        dead.logical_or_(bad)
        if not bool(dead.any()):
            return None
        v = v.clone()
        v[dead] = fill[dead][:, None].to(v.dtype)
        return v.reshape(blk.shape)
    want = {}
    for f, (a, b) in windows.items():
        for t in range(a, b + 1):
            want.setdefault(t, []).append(f)
    got = {f: {} for f in windows}

    def on_pred(k, p):
        t = o0 + k + 1
        for f in want.get(t, ()):
            got[f][t] = p.detach().clone()
    sim = T._model(spec, train=False, n_frames=w + n - 1)
    with torch.no_grad():
        T._trace_rollout_stream(sim, learn, spec, box, o0, n, device, on_pred, silence=silence, w=w)
    return {f: torch.stack([got[f][t] for t in range(a, b + 1)]) for f, (a, b) in windows.items()}, int(dead.sum())


def run_one(name, device, w0=True):
    from plexus import engine
    from plexus import trainer as T
    import exp17_ablation as A
    engine.quiet(True)
    spec = T.load(name)
    out = T.out_dir(spec, None)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = T.Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = T._trace_setup(spec, device)
    X, rec = box["X"], box["rec"]
    rname = spec["task"]["reference"]["trace_recording"]
    resp = np.load(os.path.join(DATA, f"baselines_{rname}_cells.npz"))["responsive"]
    ri = torch.as_tensor(np.where(resp)[0], device=X.device)
    dt = float(np.median(np.diff(rec["t_s"])))
    pre, post, ev = int(round(PRE_S / dt)), int(np.ceil(POST_S / dt)), int(round(EVOKED_S / dt))
    tr = rec["trials"]
    held = [(int(f), int(s)) for f, s, full, h in zip(tr[:, 0], tr[:, 2], tr[:, 5], tr[:, 6]) if h and full]
    windows = {f: (f - pre, f + post) for f, _ in held}
    keys = {e["param"][2:]: T.Learnables.key(e) for e in spec["learnable"]
            if e.get("op") == "state_diffuse" and str(e.get("param", "")).startswith("W_")}
    arms = {"full": contextlib.nullcontext()}
    if w0 and keys:
        arms["W0"] = A.weights_zeroed(learn, keys, {s: None for s in keys})
    res = {"run": name, "recording": rname, "responsive_cells": int(resp.sum()), "window": [pre, post],
           "evoked_s": EVOKED_S, "arms": {}}
    for arm, cm in arms.items():
        t0 = time.time()
        with cm:
            preds, n_dead = rollout_windows(T, spec, learn, box, device, windows)
        rows = []
        for f, site in held:
            P = preds[f][:, ri]                       # [pre + post + 1, n_resp]: frames f - pre .. f + post
            R = X[f - pre:f + post + 1][:, ri]
            e_p = (P[pre:pre + ev].mean(0) - P[:pre].mean(0))
            e_r = (R[pre:pre + ev].mean(0) - R[:pre].mean(0))
            c = float(torch.corrcoef(torch.stack([e_p, e_r]))[0, 1])
            rows.append({"onset": f, "site": site, "kind": "control" if site == CTRL_SITE else "gut",
                         "evoked_free": float(e_p.mean()), "evoked_rec": float(e_r.mean()), "pattern_r": c,
                         "mse_window": float(((P - R) ** 2).double().mean())})
        g = [r for r in rows if r["kind"] == "gut"]
        c_ = [r for r in rows if r["kind"] == "control"]
        res["arms"][arm] = {"silenced": n_dead, "trials": rows,
                            "evoked_gut_free_over_rec": float(np.mean([r["evoked_free"] for r in g]) /
                                                              np.mean([r["evoked_rec"] for r in g])) if g else None,
                            "pattern_r_gut": float(np.mean([r["pattern_r"] for r in g])) if g else None,
                            "evoked_ctrl_free_over_rec_gut": float(np.mean([r["evoked_free"] for r in c_]) /
                                                                   np.mean([r["evoked_rec"] for r in g])) if g and c_ else None}
        a = res["arms"][arm]
        print(f"[freetrial] {name} {arm}: {time.time() - t0:.0f} s, {n_dead} silenced; gut evoked free/rec "
              f"{a['evoked_gut_free_over_rec']:+.2f}, pattern r {a['pattern_r_gut']:+.2f}; control free / rec gut "
              f"{a['evoked_ctrl_free_over_rec_gut']}; " + ", ".join(
                  f"{r['kind'][0]}{r['onset']} {r['evoked_free']:+.3f}/{r['evoked_rec']:+.3f}" for r in a["trials"]),
              flush=True)
    json.dump(res, open(os.path.join(out, "results", f"{name}_freetrial.json"), "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-w0", action="store_true")
    a = ap.parse_args()
    for r in a.runs:
        run_one(r, a.device, w0=not a.no_w0)


if __name__ == "__main__":
    main()
