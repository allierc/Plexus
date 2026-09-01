"""The TESTER: roll the learned model forward on HELD-OUT data and score the trajectory.

WHY A SEPARATE ENGINE FROM THE TRAINER, and it is not tidiness. The trainer reports a loss, and a
loss is not a result: it is measured on the training split, at the horizon the objective happened to
use, in whatever units the increment happened to have. The tester answers the question a reader
actually has -- GIVEN THE FIRST FRAME, HOW LONG DOES THIS MODEL STAY RIGHT? -- on frames the fit
never saw, at horizons the fit never trained on, in the unit of the phenomenon.

The distinction has teeth here. The Kuramoto fit's own loss falls monotonically while `K` sits at
4% of its true value; a rollout on held-out frames is what makes that visible.

WHAT IT DOES, and it is the same three lines as the recurrent train step with the gradient off:

    load the first HELD-OUT frame into the model hierarchy
    step its schedule, one record at a time, feeding the model its OWN output
    score every step against what was recorded, through `metrics.rollout`

FED ITS OWN OUTPUT, NEVER RE-ANCHORED. A "rollout" that re-reads the observation at each step is
measuring one-step accuracy K times, which is a different and much easier question -- and the one a
model with a large per-cell parameter field will always look good at.

NOTHING IS COMPUTED HERE. Every number comes from `metrics.py`, which is the single entry point, so
the tester's console output and the gate table cannot disagree about what R^2 meant.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

import metrics
import vtk_toy
from model_hierarchy import ModelHierarchy


def run(fit, out_root: str, device: str = "cuda", horizon: int | None = None,
        n_starts: int = 4, split: str = "test", out_dir: str | None = None) -> dict:
    """Roll the spec's model out on `split` and score it. Returns the metrics dict."""
    from trainer import _load_field

    fb = fit.fit
    field = fb.field
    run_dir = fb.run if os.path.isdir(fb.run) else os.path.join(out_root, fb.run)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"fit.data.run is {fb.run!r}; no such run directory")
    init_names = list(fb.data.get("init") or [field])
    fields = {n: _load_field(run_dir, n, device).float()
              for n in dict.fromkeys(init_names + [field])}
    u = fields[field]
    n_rec = u.shape[0]
    stride = fb.record_stride or max(1, round(fit.n_frames / max(1, n_rec - 1)))

    lo, hi = fb.split.get(split, [0, fit.n_frames])
    a, b = int(lo * n_rec / fit.n_frames), int(hi * n_rec / fit.n_frames)
    held = {k: v[a:b] for k, v in fields.items()}
    obs = held[field]
    if obs.shape[0] < 2:
        raise ValueError(f"the {split!r} split holds {obs.shape[0]} records; a rollout needs 2+")
    K = int(horizon or (fb.rollout or {}).get("horizon", 8))
    K = min(K, obs.shape[0] - 1)

    model = ModelHierarchy(fit.path, device=device)
    masks = {n: (v.abs().amax(dim=0).amax(dim=0) > 1e-12).to(v.dtype) for n, v in fields.items()}
    mask = masks[field]
    model.bind_shapes({op: masks.get(model.at_of(op), mask) for op in model.names})
    # THE LEARNED PARAMETERS, read back from the fit's own artifacts. A tester that re-fitted would
    # be reporting a different model from the one the trainer wrote.
    loaded = _load_learned(model, run_dir)

    starts = np.linspace(0, obs.shape[0] - K - 1, num=min(n_starts, obs.shape[0] - K),
                         dtype=int) if obs.shape[0] > K else np.array([0])
    true_steps, pred_steps = [], []
    with torch.no_grad():
        for s0 in starts:
            # THE LATENT FIELDS ARE GIVEN ONCE, at the window's first frame, and never again --
            # after that the model carries its own state, which is what makes this a rollout.
            for name, arr in held.items():
                model.load(name, arr[int(s0)].clone())
            for k in range(K):
                model.step(stride)                       # ONE RECORD per rollout step
                pred_steps.append(model.read(field).detach().cpu().numpy())
                true_steps.append(obs[int(s0) + k + 1].cpu().numpy())
    # [n_starts*K, ...] -> [K, n_starts, ...] so a per-step statistic pools the starts
    P = np.stack(pred_steps).reshape(len(starts), K, *pred_steps[0].shape).transpose(1, 0, *range(2, 2 + pred_steps[0].ndim))
    T = np.stack(true_steps).reshape(len(starts), K, *true_steps[0].shape).transpose(1, 0, *range(2, 2 + true_steps[0].ndim))

    m = mask.cpu().numpy()
    cell_mask = np.broadcast_to(m, T.shape[1:]) if m.shape != T.shape[1:] else m
    res = metrics.rollout(T, P, mask=cell_mask)
    res.update(split=split, starts=[int(x) for x in starts], record_stride=stride,
               learned_from=loaded, run=run_dir)
    print(metrics.format_rollout(res, name=f"{fit.name} [{split}]"))

    # THE FIT'S OUTPUTS BELONG TO THE FIT, not to the dataset it read. `log/<fit>/` holds the
    # metrics and the movie; `log/<dataset>/` stays the archived recording plus whatever the
    # trainer wrote there. Mixing them means a second fit on the same data overwrites the first's
    # test results without either run's directory saying so.
    out_dir = out_dir or os.path.join(out_root, f"{fit.name}_{fit.model.tag()}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{split}.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if not isinstance(v, np.ndarray)}, f, indent=2)

    # GROUND TRUTH BESIDE INFERRED, one rollout, one shared colour scale. Channel 0 is the
    # observable in every spec here (`u` for transport, `sin phi` for Kuramoto).
    s0 = 0
    gt = T[:, s0] if T.ndim > 2 + (T.ndim - 3) else T
    gt, pr = T[:, s0], P[:, s0]
    if gt.ndim == T.ndim - 1 and gt.shape[0] == K and gt.ndim > 3:
        gt, pr = gt[:, 0], pr[:, 0]                       # drop the channel axis
    mp4 = os.path.join(out_dir, f"rollout_{split}.mp4")
    try:
        vtk_toy.pair_movie(gt, pr, mp4, labels=(f"ground truth  [{split}]",
                                                f"inferred  R2 {res['r2_mean']:.3f}  "
                                                f"r {res['pearson_mean']:.3f}"))
        res["movie"] = mp4
        print(f"  movie: {mp4}")
    except Exception as e:                                # a missing renderer must not lose metrics
        print(f"  movie skipped: {type(e).__name__}: {e}")
    return res


def _load_learned(model, run_dir: str) -> list[str]:
    """Restore what the trainer wrote: scalars from trainer.json, fields from their .npy."""
    loaded = []
    jpath = os.path.join(run_dir, "trainer.json")
    learned = {}
    if os.path.exists(jpath):
        with open(jpath) as f:
            learned = json.load(f).get("learned", {})
    named = dict(model.named_parameters())
    for k, p in named.items():
        npy = os.path.join(run_dir, f"{k}.npy")
        if os.path.exists(npy):
            p.data.copy_(torch.as_tensor(np.load(npy), device=p.device))
            loaded.append(k)
        elif k in learned and not isinstance(learned[k], dict):
            p.data.fill_(float(learned[k]))
            loaded.append(k)
    if not loaded:
        raise FileNotFoundError(
            f"no fitted parameters in {run_dir}: expected trainer.json and/or <param>.npy. "
            f"Run `-o train` first -- a tester that silently scored an UNFITTED model would "
            f"report the initialisation as a result.")
    return loaded
