#!/usr/bin/env python
"""exp19, Cedric's point 2 (2026-10-03): has the law learned NETWORK dynamics, or only a function of the forcing?

    PYTHONPATH=src python tools/exp19_ablation.py wl_gc_I [wl_gc_I_now ...] [--device cuda:0]

For each trained run (log/training/wholistic/<run>/models/best.pt), the free rollout of its full test
(`trainer._test_field_full`: from volume `inputs`, every later volume, nothing recorded enters -- tests/test_exp19_no_leak.py)
is re-run twice through the trainer's own setup and rollout:

    full      the law as trained
    W = 0     the same law with every message between voxels zeroed (`diffuse[graphcast]` `messages = False` on the
              trained instance), the forcing I(t) kept -- exp17's slide 26; skipped for a run trained without messages
              (that run IS exp17's slide 27 control, `wl_gc_I_now`)

and each is read as exp17 reads its controls (devcontainer-14, 2026-10-03; exp17 trainer.py:2773, exp17_slides.py:1970):

    r2, r2_spatial            variance explained against the RAW recording, total and with each frame's mean removed
    r2_denoised               against the recording averaged over 3 voxels x 3 frames (`field_recording.denoise`)
    fish_mean_sd              the fish-mean trace -- the unweighted mean over the voxels on the mask at the origin and
                              the frame -- its SD over the free frames (numpy, ddof 0); `recorded_fish_mean_sd` beside it
    fish_mean_corr            the correlation of that trace with the recorded one
    fish_mean_sd_frac         fish_mean_sd / recorded_fish_mean_sd
    brain_mean_r2, _rmse      Cedric's headline (exp17, 2026-10-03): 1 - sum_t (pbar_t - obar_t)^2 / sum_t (obar_t - mean obar)^2
                              on the fish-mean traces, and their RMSE in F / F0 -- exp17's own `trainer._brain_mean_metrics`

and a third variant, `no_stimulus` (exp17's `drive: off`): the trained law with its learned forcing (I, or the SIREN's
I_mlp) set to 0 for the whole rollout, then restored -- what the network carries without the stand-in stimulus.

Writes log/training/wholistic/<run>/results/<run>_ablation.json and the two fish-mean traces in <run>_ablation.npz.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


class _OwnState:
    """Stands in for the recording at the seed of a later segment: index origin - k returns the MODEL's own state
    k volumes back, channel k of its last field. `_field_rollout` reads only `X[origin - k]`, k < inputs."""

    def __init__(self, origin, grid):
        self.origin, self.grid = origin, grid

    def __getitem__(self, i):
        k = self.origin - i
        return self.grid[k:k + 1]


def rollout(spec, learn, box, device, messages, seg=200):
    """The free rollout from volume `inputs - 1` to the end, in segments of `seg` volumes kept on the CPU: the first
    seeded from the recorded origin volumes, each later one from the model's OWN last field -- nothing recorded
    enters after the origin (the same rollout as `_test_field_full`'s, without its [n, C, *grid] stack on the GPU)."""
    import torch
    from plexus import trainer
    o = box["n_in"] - 1
    n = box["rec"]["ratio"].shape[0] - 1 - o
    ready = learn.ready

    def ready_off(H):                                # the trained instance, its messages switched as asked
        ready(H)
        for op in H.operators:
            if hasattr(op, "messages"):
                op.messages = messages
    learn.ready = ready_off
    out, cur, done, b = [], o, 0, dict(box)
    try:
        with torch.no_grad():
            while done < n:
                h = min(seg, n - done)
                sims = trainer._field_sims(spec, [h], train=False)
                roll = trainer._field_rollout(sims, learn, b, cur, h, device, False)
                out.append(roll[:, 0].cpu().numpy())
                b = dict(box, X=_OwnState(cur + h, roll[-1]))
                cur, done = cur + h, done + h
                del roll
    finally:
        learn.ready = ready
    return np.concatenate(out, 0), o, n


def read(free, rec, recd, o, n):
    from plexus import trainer
    obs, obs_d = rec["ratio"][o + 1:o + 1 + n], recd["ratio"][o + 1:o + 1 + n]
    m = rec["mask"][o + 1:o + 1 + n] & rec["mask"][o]
    r2, r2s = trainer._r2_parts(free, obs, m)
    r2d, r2ds = trainer._r2_parts(free, obs_d, m)
    fm = np.array([free[k][m[k]].mean() for k in range(n)])
    om = np.array([obs[k][m[k]].mean() for k in range(n)])
    bm = trainer._brain_mean_metrics(om, fm)
    return ({"r2": r2, "r2_spatial": r2s, "r2_denoised": r2d, "r2_spatial_denoised": r2ds, **bm,
             "fish_mean_sd": float(fm.std()), "recorded_fish_mean_sd": float(om.std()),
             "fish_mean_sd_frac": float(fm.std() / om.std()), "fish_mean_corr": float(np.corrcoef(fm, om)[0, 1]),
             "finite": bool(np.isfinite(free).all())}, fm, om)


def run(name, device):
    import torch
    from plexus import engine, trainer
    from plexus.tasks import field_recording as FR
    engine.quiet(True)
    spec = trainer.load(name)
    out = trainer.out_dir(spec)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = trainer.Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = trainer._field_setup(spec, device)
    rec = box["rec"]
    recd = FR.denoise(rec)
    trained_with_messages = all(o.get("messages", True) for o in yaml_ops(spec))
    res, traces = {"run": name, "trained_with_messages": trained_with_messages}, {}
    forcing = [k for k in learn.p if k in ("diffuse.I", "diffuse.I_mlp")]
    for tag, msg in (("full", True), ("W0", False), ("no_stimulus", True)):
        if tag == "W0" and not trained_with_messages or tag == "no_stimulus" and not forcing:
            continue
        saved = {k: learn.p[k].detach().clone() for k in forcing} if tag == "no_stimulus" else {}
        with torch.no_grad():
            for k in saved:
                learn.p[k].zero_()
        try:
            free, o, n = rollout(spec, learn, box, device, msg)
        finally:
            with torch.no_grad():
                for k, v in saved.items():
                    learn.p[k].copy_(v)
        res[tag], traces[f"fish_mean_{tag}"], traces["fish_mean_recorded"] = read(free, rec, recd, o, n)
        r = res[tag]
        print(f"[{name} {tag}] brain-mean R2 {r['brain_mean_r2']:+.3f} (RMSE {r['brain_mean_rmse']:.4f}); "
              f"R2 {r['r2']:+.4f} (spatial {r['r2_spatial']:+.4f}), denoised {r['r2_denoised']:+.4f}; "
              f"fish-mean SD {r['fish_mean_sd']:.5f} = {r['fish_mean_sd_frac']:.0%} of recorded "
              f"{r['recorded_fish_mean_sd']:.5f}, corr {r['fish_mean_corr']:+.3f}", flush=True)
    res.update(origin_volume=o + 1, steps=n)
    json.dump(res, open(os.path.join(out, "results", f"{name}_ablation.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out, "results", f"{name}_ablation.npz"), **traces)
    return res


def yaml_ops(spec):
    import yaml
    return [o for o in yaml.safe_load(open(spec["model"]))["operators"] if o.get("model") == "graphcast"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    for r in a.runs:
        run(r, a.device)


if __name__ == "__main__":
    main()
