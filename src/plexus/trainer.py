"""Train a Plexus model for a task. The engine simulates; this module trains.

    python Plexus_Main.py -o train_test_analyse t1_integrator_perfect_zf285
    python -m plexus.trainer -o train test analyse config/training/neural/t1_integrator_perfect_zf285.yaml

A TRAINING SPEC IS A DECLARATION IN THREE PARTS, beside a model it never edits (plexus2.tex,
"Training differentiable models"):

    model:      the forward spec -- exactly the file `Plexus_Main.py -o generate` runs
    learnable:  what is free, and in which representation
    task:       what is observed, against what reference, driven by what, scored by which loss
    training:   how the gradient is obtained and used -- the one part that is not a claim

    name: t1_integrator_perfect_zf285
    model: config/neural/zf_circuit_285.yaml
    learnable:
      - {block: w, of: synapse}
      - {block: tau, of: neuron, bounds: [0.005, 10.0]}
    task:
      reference: {corpus: t1_integrator_perfect, n_train: 512, n_val: 60, n_test: 80}
      drive: {set: sensor, block: signal}
      observe: {set: output, block: value, channel: 0}
      loss: mse
    training: {optimizer: adam, lr: 0.005, schedule: cosine, clip: 1.0, epochs: 100, batch: 32,
               seed: 0, horizon: [0.25, 0.5, 0.75, 1.0], horizon_min: 60}

THE TRAINER OWNS THE PARAMETERS; THE ENGINE NEVER SEES A LEARNABLE. `engine.run` builds a fresh
Hierarchy -- operators included -- on every call, which is what makes a run reproducible, so a
tensor made inside one rollout does not survive into the next. The parameters therefore live
here, in `Learnables`, and are written into the seeded state by the engine's one neutral hook,
`on_seeded`, before the batch axis exists: every rollout of a run writes the SAME leaves, and
`repeat` along the batch axis carries their gradient back from all B trials. The engine attaches
no meaning to the write.

EVERY KEY IS READ OR REFUSED. A key that nothing reads -- a representation named before it is
implemented, a loss misspelt -- would be dropped, and the run would train something other than
what its spec says. `load` refuses it instead, with the list of what is read.

What is implemented, and nothing else: learnable STATE BLOCKS in the `tensor` representation (one
free value per element, starting at the model's own value), with an optional step size and
bounds per block; a CORPUS reference (`plexus.tasks`: stimulus, teacher, condition grid, disjoint
splits) driven through a state block and observed through another; the `mse` loss; Adam with a
cosine schedule, gradient clipping, a horizon curriculum and a divergence guard.

Output, one folder per run under the model's own folder, mirroring `config/training/<model>/`:

    log/training/<model>/<name>/
        config.yaml                  the training spec, copied
        model.yaml                   the forward spec it trained
        models/best.pt               the learnables, by validation error
        results/report.json          what training did
        results/<name>_test.json     the held-out rollout
        results/<name>_test.png      the analysis figure
        results/<name>_graph.json    the graph phase (`-o graph`, a trace run): what the learned graph does
        results/graph/               its rollouts (cached), the card, the pulse movie
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import re
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

import plexus.operators                                          # noqa: F401  self-registers
from plexus import engine
from plexus.models.registry import get_contract
from plexus.paths import get_repo_root, log_path, training_spec
from plexus.schema import load as load_model
from plexus.tasks.generate import task_dir
from plexus.tasks.trainer import load_split, n_condition_cells, with_context


# ============================================================================== the declaration
_KEYS = {
    "top": {"name", "model", "learnable", "task", "training"},
    "learnable": {"block", "of", "with", "lr", "bounds", "over", "K", "extent", "param", "op",
                  "prior", "field", "levels", "features", "log2_table", "base", "scale", "title"},
    "task": {"reference", "drive", "observe", "loss", "settle_s", "u_weight", "mask", "warmup", "rollouts",
             "movie_metric", "record", "graph"},
    "record": {"set", "block", "every_s"},
    "rollout": {"name", "zero", "drive", "clamp", "messages", "prune", "zero_input", "pulse", "clock"},
    "clamp": {"rois", "mask", "array"},
    # THE GRAPH PHASE (`-o graph`, exp17 2026-10-10): what the trained law's graph does -- its settings, declared
    "graph": {"controls", "tests", "inputs", "regions", "steady_skip_min", "bootstrap", "prune", "prune_b", "spectrum",
              "pulse", "movies", "every_frame"},
    "graph_controls": {"no_w", "mean_field", "seed", "random"},
    "roi": {"box", "sphere", "units"},
    "reference": {"corpus", "n_train", "n_val", "n_test", "context", "shape", "size", "centre",
                  "recording", "beats", "field_recording", "coarsen", "points", "split", "normalise",
                  "trace_recording", "test_stride"},
    "drive": {"set", "block", "prescribe", "width", "window"},
    "observe": {"set", "block", "channel", "unit", "measure", "grid", "of", "field", "alive", "elements"},
    "training": {"optimizer", "lr", "lr_min", "lr_min_frac", "schedule", "clip", "epochs", "batch",
                 "seed", "horizon", "horizon_min", "snapshot_every", "guard", "stages", "render",
                 "iters", "save_every", "anneal", "select", "init_from", "circuit_movie", "resume"},
    "term": {"term", "weight", "reduction"},
    "stage": {"resolution", "iters", "horizon"},
}
REPRESENTATIONS = ("tensor", "lattice", "hash")
_HASH_KEYS = ("levels", "features", "log2_table", "base", "scale")
REFERENCES = ("corpus", "shape", "recording", "field_recording", "trace_recording")
_LOSS_FOR = {"corpus": ("mse",), "shape": ("log_mse",), "recording": ("affine_mse",),
             "field_recording": ("masked_mse",), "trace_recording": ("trace_mse",)}
PRIORS = ("shrink", "shrink_to_mean", "smooth", "l1", "l2", "group_l1", "row_l1", "sign",
          "monotone", "pin", "input_group_l1", "dale", "dale_keep")   # from `monotone` on: computed by the operator (`prior_term`)
# THE EVIDENCE TERMS each reference kind can score, by name. `task.loss` is one name or a list of
# {term, weight, reduction}; the loop computes the quantities, `_objective` weighs and records them.
TERMS = {"corpus": ("mse",), "shape": ("log_mse", "volume", "point_mse"),
         "recording": ("affine_mse", "affine_A", "affine_u"), "field_recording": ("masked_mse", "global_mse", "mask_iou", "mask_bce"),
         "trace_recording": ("trace_mse", "trace_increment")}
REDUCTIONS = ("mean", "norm2", "huber", "relative_l2")
OPTIMIZERS = ("adam",)
SCHEDULES = ("cosine", "none")
GUARDS = ("restore_and_halve", "none")


def _refuse_unread(where, d, allowed):
    extra = sorted(set(d) - allowed)
    if extra:
        raise ValueError(f"{where}: key(s) {extra} are read by nothing; the keys read here are "
                         f"{sorted(allowed)}. An unread key would be dropped and the run would "
                         f"train something other than what the spec says.")


def _check_rollouts(path, s, kind):
    """`task.rollouts` (Cedric, 2026-10-03): the free rollouts the test phase makes besides the nominal one, each a
    named VARIANT of the same learned model -- `zero:` learnables set to 0 (the network's W: "W = 0"), `drive: off`
    (no known stimulus), `clamp: {rois: [...]}` (the elements inside the ROIs given their RECORDED value every frame:
    measured leaders or hypothesised input neurons; they are left out of every score), or, on a trace law, `clamp:
    {mask: <npz>, array: mask}` (the elements nonzero in that array: exp17's input neurons). Trace references, and field
    references (exp19, 2026-10-03): there `messages: off` is the W = 0 of a message-passing law (`diffuse[graphcast]`'s
    `messages`), `zero: [I]` / `[I_mlp]` removes its learned forcing (the field's stand-in for a stimulus, it has no
    drive), and `clamp:` gives the voxels inside the ROIs their recorded values (x, y, z in um from the voxel size)."""
    ros = s["task"].get("rollouts")
    if ros is None:
        return
    if kind not in ("trace_recording", "field_recording"):
        raise ValueError(f"{path}: task.rollouts is implemented for trace_recording and field_recording references")
    names = {e.get("param") or e.get("block") for e in s["learnable"]}
    seen = set()
    for j, ro in enumerate(ros):
        w = f"{path}: task.rollouts[{j}]"
        _refuse_unread(w, ro, _KEYS["rollout"])
        nm = str(ro.get("name", ""))
        if not re.fullmatch(r"[A-Za-z0-9_]+", nm) or nm in seen:
            raise ValueError(f"{w}: `name:` must be a unique word (letters, digits, _), it is {nm!r}")
        seen.add(nm)
        bad = [z for z in ro.get("zero", []) if z not in names]
        if bad:
            raise ValueError(f"{w}: zero {bad} are not learnables of this run (its learnables: {sorted(names)})")
        if ro.get("drive", "on") not in ("on", "off"):
            raise ValueError(f"{w}: `drive:` on (default) or off")
        if "drive" in ro and kind == "field_recording":
            raise ValueError(f"{w}: a field recording has no drive; its stand-in stimulus is a learnable -- `zero: [I]`")
        if ro.get("messages", "on") not in ("on", "off"):
            raise ValueError(f"{w}: `messages:` on (default) or off")
        if "messages" in ro and kind != "field_recording":
            raise ValueError(f"{w}: `messages:` is a field law's switch (diffuse[graphcast]); a trace law zeroes its W")
        if ro.get("drive") == "off" and "drive" not in s["task"]:
            raise ValueError(f"{w}: `drive: off` but the task has no drive")
        # THE GRAPH PHASE'S VARIANTS (exp17, 2026-10-10): weights below a threshold cut (`prune:`), the input of some
        # elements cut (`zero_input:`), some elements held at a level for a while (`pulse:`, the virtual perturbation),
        # the frame clock the law's functions of time read shifted or frozen (`clock:`)
        if "prune" in ro:
            _check_prune(w, ro["prune"], names, kind)
        if "zero_input" in ro:
            _check_zero_input(w, ro["zero_input"], s, kind, names)
        if "pulse" in ro:
            _check_pulse(w, ro["pulse"], kind)
        if "clock" in ro:
            _check_clock(w, ro["clock"], kind)
        if "clamp" in ro:
            _refuse_unread(f"{w}.clamp", ro["clamp"], _KEYS["clamp"])
            if "mask" in ro["clamp"] or "array" in ro["clamp"]:
                _check_clamp_mask(w, ro["clamp"], kind)
                continue
            rois = ro["clamp"].get("rois") or []
            if not rois:
                raise ValueError(f"{w}.clamp needs `rois:` (a list of box / sphere)")
            for i, r in enumerate(rois):
                _refuse_unread(f"{w}.clamp.rois[{i}]", r, _KEYS["roi"])
                if ("box" in r) == ("sphere" in r):
                    raise ValueError(f"{w}.clamp.rois[{i}]: exactly one of `box:` [[lo, hi] per axis, 2 (x, y: every z) "
                                     f"or 3] or `sphere:` [x, y, z, r]")
                if r.get("units", "um") not in ("um", "fraction"):
                    raise ValueError(f"{w}.clamp.rois[{i}]: `units:` um (default) or fraction (of the bounding box)")
                if "box" in r and (len(r["box"]) not in (2, 3) or any(len(b) != 2 or b[0] > b[1] for b in r["box"])):
                    raise ValueError(f"{w}.clamp.rois[{i}]: `box:` is [[lo, hi], [lo, hi]] or with a third [lo, hi]")
                if "sphere" in r and len(r["sphere"]) != 4:
                    raise ValueError(f"{w}.clamp.rois[{i}]: `sphere:` is [x, y, z, r]")


def _check_prune(w, p, names, kind):
    """`prune: {thresholds: {<learnable>: t, ...}}` (exp17's pruning ladder, Cedric 2026-10-08 / 2026-10-10): every
    value of that learnable whose magnitude is below t is set to 0 for the rollout and restored after -- an edge set's
    weak edges removed, the law otherwise untouched."""
    if kind != "trace_recording":
        raise ValueError(f"{w}: `prune:` is a trace law's variant (its edge weights cut below a threshold)")
    if not isinstance(p, dict) or set(p) != {"thresholds"} or not isinstance(p["thresholds"], dict) or not p["thresholds"]:
        raise ValueError(f"{w}: `prune:` is {{thresholds: {{<learnable>: t, ...}}}} (t >= 0, one per learnable cut)")
    for k, t in p["thresholds"].items():
        if k not in names:
            raise ValueError(f"{w}.prune: {k!r} is not a learnable of this run (its learnables: {sorted(names)})")
        if not isinstance(t, (int, float)) or isinstance(t, bool) or t < 0:
            raise ValueError(f"{w}.prune: the threshold of {k!r} is a number >= 0, it is {t!r}")


def _check_zero_input(w, c, s, kind, names):
    """`zero_input: {quantile: q} | {threshold: t} | {mask: <npz>, array: <name>}` (Cedric, 2026-10-10: "set a
    threshold on B_i to identify the input neurons"): the rows of the `input` block (B_i, the elements' weights on the
    stimulus) set to 0 for the elements whose |B_i| is below the q-th quantile of the input elements' (those with a
    nonzero row), or below t, or who are nonzero in the mask's array; restored after. `columns: [j, ...]` restricts
    |B_i| and the cut to those columns of the block (exp17: the stimulus features, not the block markers that reach
    every neuron)."""
    if kind != "trace_recording":
        raise ValueError(f"{w}: `zero_input:` is a trace law's variant (its elements' stimulus weights)")
    if "drive" not in s["task"] or "input" not in names:
        raise ValueError(f"{w}: `zero_input:` needs a task drive and the `input` block among the learnables")
    if not isinstance(c, dict) or len([k for k in ("quantile", "threshold", "mask") if k in c]) != 1 \
            or set(c) - {"quantile", "threshold", "mask", "array", "columns"}:
        raise ValueError(f"{w}: `zero_input:` is one of {{quantile: q}}, {{threshold: t}}, {{mask: <npz>, array: <name>}}"
                         " (+ columns: [j, ...])")
    if "columns" in c and not (isinstance(c["columns"], (list, tuple)) and len(c["columns"]) > 0
                               and all(isinstance(j, int) and not isinstance(j, bool) and j >= 0 for j in c["columns"])):
        raise ValueError(f"{w}.zero_input: `columns:` a non-empty list of column indices of the `input` block")
    if "quantile" in c and not (isinstance(c["quantile"], (int, float)) and 0 <= float(c["quantile"]) < 1):
        raise ValueError(f"{w}.zero_input: `quantile:` in [0, 1)")
    if "threshold" in c and not (isinstance(c["threshold"], (int, float)) and float(c["threshold"]) >= 0):
        raise ValueError(f"{w}.zero_input: `threshold:` >= 0")
    if "mask" in c:
        _check_clamp_mask(f"{w}.zero_input", c, kind)


def _check_pulse(w, p, kind):
    """`pulse: {rois: [...] | mask: <npz>, array:, level_z: 2.0, start_s: 120, duration_s: 10}` (the virtual
    perturbation, Cedric 2026-10-10): the selected elements HELD at the level mu + level_z sd (the law's own
    normalisation) from start_s after the rollout's first free frame for duration_s, as `clamp:` holds recorded
    frames, then released; every element stays in the scores. With `drive: off` it is a pulse into the resting law."""
    if kind != "trace_recording":
        raise ValueError(f"{w}: `pulse:` is a trace law's variant")
    if not isinstance(p, dict) or set(p) - {"rois", "mask", "array", "level_z", "start_s", "duration_s"}:
        raise ValueError(f"{w}: `pulse:` is {{rois | mask (+ array), level_z, start_s, duration_s}}")
    if ("rois" in p) == ("mask" in p):
        raise ValueError(f"{w}.pulse: the elements by `rois:` or by `mask:`, one of them")
    if "mask" in p:
        _check_clamp_mask(f"{w}.pulse", p, kind)
    elif not p["rois"]:
        raise ValueError(f"{w}.pulse needs `rois:` (a list of box / sphere)")
    for k, lo in (("level_z", None), ("start_s", 0.0), ("duration_s", 1e-9)):
        v = p.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or (lo is not None and float(v) < lo):
            raise ValueError(f"{w}.pulse: `{k}:` is a number" + (f" >= {lo:g}" if lo else "") + f", it is {v!r}")


def _check_clock(w, c, kind):
    """`clock: {shift_frames: n} | {freeze_at: f}` (the time-memorisation test, 2026-10-10): the frame clock the law's
    functions of absolute time read (Omega(x, t), alpha(t): FRAME_CLOCK) shifted by n frames, or frozen at frame f,
    while the stimulus keeps its true time. A law with no function of time is unchanged by it."""
    if kind != "trace_recording":
        raise ValueError(f"{w}: `clock:` is a trace law's variant")
    if not isinstance(c, dict) or len(c) != 1 or set(c) - {"shift_frames", "freeze_at"}:
        raise ValueError(f"{w}: `clock:` is {{shift_frames: n}} or {{freeze_at: f}}")
    v = list(c.values())[0]
    if not isinstance(v, int) or isinstance(v, bool) or ("freeze_at" in c and v < 0):
        raise ValueError(f"{w}.clock: an integer frame count (freeze_at >= 0), it is {v!r}")


def _check_graph(path, s, kind):
    """`task.graph` (the graph phase, `-o graph`; exp17, Cedric 2026-10-10): the trained controls to compare with
    (`controls: {no_w, mean_field, seed, random}`, training spec names of runs trained without the graph, with the
    mean field, with another seed, on a random graph), the tests to run, the input neurons (`inputs: {mask, array}`,
    default the law's own input mask), the regions (an atlas npz: `regions` [N, R] bool, `names`, `inside`,
    `atlas_um`), the steady window, the bootstrap, the pruning ladders, the spectrum and the pulse settings, movies
    on / off, every-frame traces on / off. Absent: the phase runs every test that applies with its defaults."""
    g = s["task"].get("graph")
    if g is None:
        return
    if kind != "trace_recording":
        raise ValueError(f"{path}: task.graph is the graph phase of a trace_recording task")
    _refuse_unread(f"{path}: task.graph", g, _KEYS["graph"])
    c = g.get("controls") or {}
    _refuse_unread(f"{path}: task.graph.controls", c, _KEYS["graph_controls"])
    for k, v in c.items():
        if v is not None and not isinstance(v, str):
            raise ValueError(f"{path}: task.graph.controls.{k} is a training spec name (or null), it is {v!r}")
    if "tests" in g:
        from plexus.tasks.graph_analysis import TESTS
        bad = [t for t in (g["tests"] or []) if t not in TESTS]
        if not isinstance(g["tests"], list) or bad:
            raise ValueError(f"{path}: task.graph.tests names tests among {list(TESTS)}; unknown: {bad}")
    if "inputs" in g:
        _check_clamp_mask(f"{path}: task.graph.inputs", g["inputs"], kind)
    if "regions" in g and not (isinstance(g["regions"], str) and g["regions"].endswith(".npz")):
        raise ValueError(f"{path}: task.graph.regions is an atlas npz (regions, names, inside, atlas_um)")
    if "steady_skip_min" in g and not (isinstance(g["steady_skip_min"], (int, float)) and g["steady_skip_min"] >= 0):
        raise ValueError(f"{path}: task.graph.steady_skip_min is a number of minutes >= 0")
    for k, keys in (("bootstrap", {"blocks", "resamples", "seed"}), ("prune", {"ladder", "span", "quantiles"}),
                    ("prune_b", {"quantiles"}), ("spectrum", {"k", "k_imag", "state", "solver"}),
                    ("pulse", {"level_z", "duration_s", "settle_s", "window_s", "regions"})):
        if k in g:
            if not isinstance(g[k], dict) or set(g[k]) - keys:
                raise ValueError(f"{path}: task.graph.{k} is a dict with keys among {sorted(keys)}")
    if "spectrum" in g and g["spectrum"].get("state", "recording_mean") not in ("recording_mean", "rest"):
        raise ValueError(f"{path}: task.graph.spectrum.state is recording_mean (default) or rest")
    if "spectrum" in g and g["spectrum"].get("solver", "krylov_schur") not in ("krylov_schur", "arpack"):
        raise ValueError(f"{path}: task.graph.spectrum.solver is krylov_schur (default, on the device) or arpack (CPU)")
    for k in ("movies", "every_frame"):
        if k in g and not isinstance(g[k], bool):
            raise ValueError(f"{path}: task.graph.{k} is true or false")


def _brain_mean_metrics(obs, pred):
    """R2 and RMSE (dF/F) of the learned brain-mean trace against the recorded one, over the free frames."""
    obs, pred = np.asarray(obs, np.float64), np.asarray(pred, np.float64)
    ok = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[ok], pred[ok]
    den = ((o - o.mean()) ** 2).sum()
    return {"brain_mean_r2": float(1 - ((p - o) ** 2).sum() / den) if den > 0 else float("nan"),
            "brain_mean_rmse": float(np.sqrt(((p - o) ** 2).mean())) if len(o) else float("nan")}


def _roi_mask(rois, pos):
    """bool [N]: the elements inside ANY of the ROIs. `units: fraction` reads a box or sphere in fractions of the
    positions' bounding box along each axis (a sphere's radius in fractions of its largest side)."""
    pos = np.asarray(pos, np.float64)
    lo, ext = pos.min(0), np.ptp(pos, 0)
    m = np.zeros(len(pos), bool)
    for r in rois:
        frac = r.get("units", "um") == "fraction"
        if "box" in r:
            k = np.ones(len(pos), bool)
            for ax, (a, b) in enumerate(r["box"]):
                a, b = (lo[ax] + a * ext[ax], lo[ax] + b * ext[ax]) if frac else (a, b)
                k &= (pos[:, ax] >= a) & (pos[:, ax] <= b)
        else:
            c, rad = np.asarray(r["sphere"][:3], float), float(r["sphere"][3])
            if frac:
                c, rad = lo + c * ext, rad * ext.max()
            k = ((pos - c) ** 2).sum(1) <= rad ** 2
        m |= k
    return m


def _check_clamp_mask(w, c, kind):
    """`clamp: {mask: <npz under graphs_data>, array: <name, default mask>}` (exp17, 2026-10-10): the elements whose
    value in that array is nonzero are given their recorded traces -- the input neurons, the rest left free. A trace
    law's neuron mask; the file is read when the rollout is made (_clamp_mask), where its length is checked."""
    if "rois" in c:
        raise ValueError(f"{w}.clamp: `rois:` or `mask:`, not both")
    if not isinstance(c.get("mask"), str) or not c["mask"].endswith(".npz"):
        raise ValueError(f"{w}.clamp: `mask:` is an npz under graphs_data (e.g. zebrafish/input_mask_destripe_bal20.npz)")
    if not re.fullmatch(r"[A-Za-z0-9_]+", str(c.get("array", "mask"))):
        raise ValueError(f"{w}.clamp: `array:` is the name of one array of the npz (default mask)")
    if kind != "trace_recording":
        raise ValueError(f"{w}.clamp: `mask:` is a trace law's neuron mask; a field recording clamps `rois:`")


def _clamp_mask(c, pos):
    """bool [N]: a rollout's `clamp:` elements -- inside its `rois:` (_roi_mask), or nonzero in its `mask:` file's
    `array:` (one value per element)."""
    if "rois" in c:
        return _roi_mask(c["rois"], pos)
    from plexus.paths import graphs_data_path
    m = np.asarray(np.load(graphs_data_path(str(c["mask"])))[str(c.get("array", "mask"))])
    if m.shape != (len(pos),):
        raise ValueError(f"clamp mask {c['mask']} [{c.get('array', 'mask')}]: shape {m.shape}, one value per element "
                         f"({len(pos)}) wanted")
    return m != 0


@contextlib.contextmanager
def _zeroed(learn, spec, names):
    """The learnables named in `names` (a `param:` or a `block:`) set to 0, restored on exit."""
    keys = [Learnables.key(e) for e in spec["learnable"] if (e.get("param") or e.get("block")) in set(names)]
    saved = {}
    with torch.no_grad():
        for k in keys:
            saved[k] = learn.p[k].detach().clone()
            learn.p[k].zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for k, v in saved.items():
                learn.p[k].copy_(v)


def _learnable_key(spec, name):
    """The `learn.p` key of the learnable called `name` (a `param:` or a `block:`)."""
    for e in spec["learnable"]:
        if (e.get("param") or e.get("block")) == name:
            return Learnables.key(e)
    raise ValueError(f"{name!r} is not a learnable of this run")


@contextlib.contextmanager
def _pruned(learn, spec, thresholds):
    """`prune:` -- each named learnable's values of magnitude below its threshold set to 0, restored on exit; yields
    {name: {"cut": n, "of": n_total, "threshold": t}}."""
    saved, cut = {}, {}
    with torch.no_grad():
        for nm, t in thresholds.items():
            k = _learnable_key(spec, nm)
            saved[k] = learn.p[k].detach().clone()
            m = learn.p[k].abs() < float(t)
            learn.p[k][m] = 0.0
            cut[nm] = {"cut": int(m.sum()), "of": int(m.numel()), "threshold": float(t)}
    try:
        yield cut
    finally:
        with torch.no_grad():
            for k, v in saved.items():
                learn.p[k].copy_(v)


def _input_rows(learn, spec, box, c):
    """`zero_input:` -- the elements whose `input` rows the variant cuts (bool [N]) and the cut's record: the input
    elements are those with a nonzero row of B (|B_i| its L2 norm); `quantile:` cuts the input elements below that
    quantile of their |B_i|, `threshold:` those below t, `mask:` the elements nonzero in the file's array; with
    `columns:` the norm and the cut run over those columns of B only."""
    k = _learnable_key(spec, "input")
    B = learn.p[k].detach().reshape(learn.p[k].shape[0], -1)
    cols = _input_cols(c, B.shape[1])
    norm = (B[:, cols] if cols is not None else B).norm(2, dim=1)
    inp = norm > 0
    rec = {"n_input": int(inp.sum()), "columns": len(cols) if cols is not None else "all"}
    if "mask" in c:
        sel = torch.as_tensor(_clamp_mask(c, box["rec"]["pos_um"]), device=B.device) & inp
        rec["by"] = "mask"
    else:
        t = float(torch.quantile(norm[inp].float(), float(c["quantile"]))) if "quantile" in c else float(c["threshold"])
        sel = inp & (norm < t)
        rec.update(by="quantile" if "quantile" in c else "threshold", threshold=t)
    rec["n_cut"] = int(sel.sum())
    return sel, rec


def _input_cols(c, width):
    """`zero_input.columns` as python ints, checked against the block's width; None when the whole row is meant."""
    if not c.get("columns"):
        return None
    cols = [int(j) for j in c["columns"]]
    if max(cols) >= width:
        raise ValueError(f"zero_input: `columns:` index {max(cols)} beyond the input block's {width} columns")
    return cols


@contextlib.contextmanager
def _input_zeroed(learn, spec, box, c):
    """`zero_input:` applied: the selected rows of the `input` block (their `columns:` when given) at 0, restored on
    exit; yields the cut's record."""
    k = _learnable_key(spec, "input")
    sel, rec = _input_rows(learn, spec, box, c)
    with torch.no_grad():
        saved = learn.p[k].detach().clone()
        P = learn.p[k].view(learn.p[k].shape[0], -1)
        cols = _input_cols(c, P.shape[1])
        if cols is None:
            P[sel] = 0.0
        else:
            P[sel.nonzero(as_tuple=True)[0][:, None], torch.as_tensor(cols, device=P.device)[None, :]] = 0.0
    try:
        yield rec
    finally:
        with torch.no_grad():
            learn.p[k].copy_(saved)


def _pulse_fn(p, box, frame_s):
    """`pulse:` -- (perturb(k, block) -> block or None, record): the selected elements held at mu + level_z sd (the
    law's normalisation: `box["norm"]`) over the free frames [start, start + duration), every history column of
    the observed block, as `clamp:` holds recorded frames."""
    sel = torch.as_tensor(_clamp_mask(p, box["rec"]["pos_um"]), device=box["X"].device)
    if not bool(sel.any()) or bool(sel.all()):
        raise ValueError(f"pulse: {int(sel.sum())} of {len(sel)} elements selected; a pulse needs some and not all")
    mu, sd, _ = box["norm"]
    level = float(mu + float(p["level_z"]) * sd)
    k0 = int(round(float(p["start_s"]) / frame_s))
    nk = max(1, int(round(float(p["duration_s"]) / frame_s)))

    def perturb(k, blk):
        if k0 <= k < k0 + nk:
            v = blk.clone()
            v[sel] = level
            return v
        return None
    rec = {"n_pulsed": int(sel.sum()), "level_dff": level, "level_z": float(p["level_z"]), "start_frame": k0,
           "frames": nk}
    return perturb, rec


def _loss_terms(task, kind):
    """`task.loss` as a list of {term, weight, reduction}: one name is one term of weight 1."""
    L = task.get("loss", _LOSS_FOR[kind][0])
    if isinstance(L, str):
        return [{"term": L}]
    if not isinstance(L, list) or not all(isinstance(x, dict) and "term" in x for x in L):
        raise ValueError("task.loss is a term name or a list of {term, weight, reduction}")
    return [dict(x) for x in L]


def _anneal(spec, t):
    """1 - exp(-rate t), the factor every prior carries; 1 when nothing is annealed."""
    an = spec["training"].get("anneal")
    return 1.0 if an is None else float(1.0 - math.exp(-float(an["rate"]) * float(t)))


def _objective(spec, learn, evidence, t, parts, prior=True):
    """THE ONE PLACE A LOSS IS ASSEMBLED: the task's weighted evidence terms plus the learnables'
    priors (annealed by `t`), every term written into `parts` under its own name so the history
    shows what entered the gradient. `evidence` maps a term name to a function of its reduction."""
    tot = None
    for term in _loss_terms(spec["task"], spec["_kind"]):
        v = evidence[term["term"]](term.get("reduction", "mean"))
        w = float(term.get("weight", 1.0))
        v = v if w == 1.0 else w * v
        parts[f"loss.{term['term']}"] = float(v.detach())
        tot = v if tot is None else tot + v
    pr = learn.prior(scale=_anneal(spec, t), parts=parts) if prior else None
    return tot if pr is None else tot + pr


def _reduce(r, reduction, target=None):
    """A residual to a scalar -- connectome-gnn's fit_residual_loss reductions."""
    if reduction == "mean":
        return (r ** 2).mean()
    if reduction == "norm2":
        return r.norm(2)
    if reduction == "huber":
        return torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction="mean", delta=1.0)
    return r.norm(2) / (target.norm(2) + 1e-8)


def load(path_or_name) -> dict:
    """Read and validate a training spec. Returns the dict with `_path` and `_model_dir` added."""
    path = training_spec(path_or_name)
    with open(path) as f:
        s = yaml.safe_load(f)
    for k in ("name", "model", "learnable", "task", "training"):
        if k not in s:
            raise ValueError(f"{path}: a training spec needs `{k}:` -- the model, what is "
                             f"learnable, the task and the training scheme.")
    _refuse_unread(path, s, _KEYS["top"])
    if not os.path.isabs(s["model"]):                    # repo-relative, like every spec path
        s["model"] = os.path.join(get_repo_root(), s["model"])
    raw = yaml.safe_load(open(s["model"]))
    sets = raw.get("sets") or {}
    ops = {o.get("op") for o in (raw.get("operators") or [])}
    an = s["training"].get("anneal")
    if an is not None and (not isinstance(an, dict) or set(an) - {"rate", "per"} or "rate" not in an
                           or an.get("per", "epoch") not in ("epoch", "iter")):
        raise ValueError(f"{path}: training.anneal is {{rate: r, per: epoch|iter}} -- every prior is "
                         f"multiplied by 1 - exp(-r t) so the fit leads and the priors follow")
    for i, e in enumerate(s["learnable"]):
        _refuse_unread(f"{path}: learnable[{i}]", e, _KEYS["learnable"])
        for k in (e.get("prior") or {}):
            if k not in PRIORS:
                raise ValueError(f"{path}: learnable[{i}] prior {k!r}; implemented: {list(PRIORS)}")
        if "param" in e or "op" in e:
            # A PARAMETER OF AN ACTIVITY: the operator is stated, one of its constants is free.
            if "param" not in e or "op" not in e or "block" in e or "of" in e:
                raise ValueError(f"{path}: learnable[{i}] is EITHER {{block:, of:}} (a state of a "
                                 f"set) OR {{param:, op:}} (a parameter of an activity)")
            if e["op"] not in ops:
                raise ValueError(f"{path}: learnable[{i}] frees `{e['op']}.{e['param']}`, but no "
                                 f"operator line of the model is `{e['op']}` ({sorted(ops)})")
            continue
        if "field" in e:
            # A FIELD OF THE MODEL, learned whole: a per-voxel EMBEDDING the law reads (exp16), in the
            # `tensor` representation (one free vector per voxel) or `hash` (Instant-NGP over the
            # voxels' coordinates, `plexus.models.hashgrid`: levels x features numbers per voxel).
            if any(k in e for k in ("block", "of")):
                raise ValueError(f"{path}: learnable[{i}] is a `field:`; it has no block or set")
            if e["field"] not in (raw.get("fields") or {}):
                raise ValueError(f"{path}: learnable[{i}] frees field {e['field']!r}, not a field of "
                                 f"{s['model']} ({sorted(raw.get('fields') or {})})")
            rep = e.get("with", "tensor")
            if rep not in ("tensor", "hash"):
                raise ValueError(f"{path}: learnable[{i}] a field is learned as `tensor` or `hash`, not {rep!r}")
            if rep == "hash" and any(k not in e for k in ("levels", "features")):
                raise ValueError(f"{path}: learnable[{i}] a hash needs `levels:` and `features:` -- "
                                 f"their product is the field's component count")
            if rep != "hash" and any(k in e for k in _HASH_KEYS):
                raise ValueError(f"{path}: learnable[{i}] sets hash keys on a {rep!r} representation")
            continue
        for k in ("block", "of"):
            if k not in e:
                raise ValueError(f"{path}: learnable[{i}] needs `{k}:`")
        if e["of"] not in sets:
            raise ValueError(f"{path}: learnable[{i}] fits `{e['of']}.{e['block']}`, but "
                             f"`{e['of']}` is not a set of {s['model']} ({sorted(sets)})")
        rep = e.get("with", "tensor")
        if rep not in REPRESENTATIONS:
            raise ValueError(f"{path}: learnable[{i}] asks for representation {rep!r}; "
                             f"implemented: {list(REPRESENTATIONS)}.")
        if rep == "lattice":
            if e.get("over") != "material":
                raise ValueError(f"{path}: learnable[{i}] is a lattice, which is read at each "
                                 f"element's MATERIAL coordinate -- say `over: material`")
            if "K" not in e or len(e.get("extent") or []) != 4:
                raise ValueError(f"{path}: learnable[{i}] lattice needs `K:` (nodes per axis) and "
                                 f"`extent: [cx, cy, cz, half_width]` in world units")
        elif any(k in e for k in ("over", "K", "extent")):
            raise ValueError(f"{path}: learnable[{i}] sets lattice keys on a {rep!r} representation")
        if rep == "hash" or any(k in e for k in _HASH_KEYS):
            raise ValueError(f"{path}: learnable[{i}] `hash` is implemented for a `field:` learnable only")
    t = s["task"]
    _refuse_unread(f"{path}: task", t, _KEYS["task"])
    ref = t.get("reference") or {}
    kinds = [k for k in REFERENCES if k in ref]
    if len(kinds) != 1:
        raise ValueError(f"{path}: task.reference must name exactly one of {list(REFERENCES)} -- a "
                         f"corpus under graphs_data/task/ or a shape from the library; it names {kinds}")
    kind = kinds[0]
    parts = ("drive", "observe") if kind in ("corpus", "recording") else ("observe",)
    if kind == "trace_recording" and "drive" in t:
        parts = ("drive", "observe")                 # the known stimulus is optional (an ablation arm drops it)
    for k in ("reference",) + parts:
        if k not in t:
            raise ValueError(f"{path}: task needs `{k}:`")
        _refuse_unread(f"{path}: task.{k}", t[k], _KEYS[k])
    for part in (parts if kind != "field_recording" else ()):
        # THE EDGE BAND PRESCRIBES A SET'S POSITIONS AND VELOCITIES, so it names the set only.
        for k in (("set",) if (kind == "recording" and part == "drive") else ("set", "block")):
            if k not in t[part]:
                raise ValueError(f"{path}: task.{part} needs `{k}:`")
        if t[part]["set"] not in sets:
            raise ValueError(f"{path}: task.{part}.set {t[part]['set']!r} is not a set of the model")
    for j, term in enumerate(_loss_terms(t, kind)):
        _refuse_unread(f"{path}: task.loss[{j}]", term, _KEYS["term"])
        if term["term"] not in TERMS.get(kind, _LOSS_FOR[kind]):
            raise ValueError(f"{path}: task.loss term {term['term']!r} does not score a {kind} "
                             f"reference; implemented for it: {list(TERMS.get(kind, _LOSS_FOR[kind]))}")
        if term.get("reduction", "mean") not in REDUCTIONS or (term.get("reduction", "mean") != "mean"
                                                                and term["term"] not in ("mse", "trace_mse", "trace_increment")):
            raise ValueError(f"{path}: task.loss term {term['term']!r}: reduction "
                             f"{term.get('reduction')!r}; only `mse` takes one of {list(REDUCTIONS)}")
    _check_rollouts(path, s, kind)
    _check_graph(path, s, kind)
    tr = s["training"]
    _refuse_unread(f"{path}: training", tr, _KEYS["training"])
    if kind == "shape":
        if "drive" in t:
            raise ValueError(f"{path}: a shape task has no drive -- nothing is fed in while it grows")
        if t["observe"].get("measure") != "grid_mass" or t["observe"].get("grid") not in (raw.get("fields") or {}):
            raise ValueError(f"{path}: a shape task observes `measure: grid_mass` on a `grid:` field "
                             f"of the model ({sorted(raw.get('fields') or {})})")
        if any(x["term"] == "point_mse" for x in _loss_terms(t, kind)) and "/" not in str(ref.get("points", "")):
            raise ValueError(f"{path}: the `point_mse` term scores each particle against ITS OWN target, "
                             f"so task.reference needs `points: <shape>/<part>` (same order as the seed)")
        for k in ("size", "centre"):
            if k not in ref:
                raise ValueError(f"{path}: task.reference needs `{k}:` -- where the shape sits and "
                                 f"how long its longest axis is, in world units")
        if not tr.get("stages"):
            raise ValueError(f"{path}: a shape task trains in `stages:` -- a list of {{resolution, iters}}")
        if any("horizon" in x for x in tr["stages"]):
            raise ValueError(f"{path}: a shape task's stages set `resolution`, not `horizon`")
        for j, st in enumerate(tr["stages"]):
            _refuse_unread(f"{path}: training.stages[{j}]", st, _KEYS["stage"])
            for name in (st.get("resolution") or {}):
                if name not in sets and name not in (raw.get("fields") or {}):
                    raise ValueError(f"{path}: training.stages[{j}].resolution names {name!r}, "
                                     f"which is neither a set nor a field of the model")
    elif kind == "recording":
        if not ref.get("beats"):
            raise ValueError(f"{path}: task.reference names the `beats:` it is fitted on")
        if t.get("drive", {}).get("prescribe") != "edge_band" or "width" not in t["drive"]:
            raise ValueError(f"{path}: a recording task drives its `edge_band` of `width:` from the "
                             f"recording -- the tissue beyond the field of view enters there")
        if t["observe"].get("measure") != "cell_affine" or t["observe"].get("of") not in sets:
            raise ValueError(f"{path}: a recording task observes `measure: cell_affine` of the "
                             f"points' parent set `of:`")
        if "iters" not in tr:
            raise ValueError(f"{path}: a recording task trains for `iters:` steps")
    elif kind == "field_recording":
        # A RECORDED FIELD HAS NO DRIVE: the observed volume at the forecast's origin IS the
        # starting state, written into the field before the first tick, and nothing enters after.
        if "drive" in t:
            raise ValueError(f"{path}: a field_recording task has no drive -- the recorded volume at "
                             f"each origin is the starting state")
        if t["observe"].get("field") not in (raw.get("fields") or {}):
            raise ValueError(f"{path}: a field_recording task observes `field:`, a field of the model "
                             f"({sorted(raw.get('fields') or {})})")
        if set(t["observe"]) - {"field", "alive"}:
            raise ValueError(f"{path}: a field_recording task observes a whole field (`field:`) and, with a death "
                             f"operator, the field of its alive fraction (`alive:`); nothing else is read")
        if "alive" in t["observe"] and t["observe"]["alive"] not in (raw.get("fields") or {}):
            raise ValueError(f"{path}: observe.alive `{t['observe']['alive']}` is not a field of the model")
        hz = tr.get("horizon")
        if "iters" not in tr or not hz or not all(isinstance(h, int) and h >= 1 for h in hz):
            raise ValueError(f"{path}: a field_recording task trains for `iters:` steps over a "
                             f"`horizon:` curriculum of whole recorded intervals, e.g. [1, 3, 6]")
    elif kind == "trace_recording":
        # A RECORDING OF TRACES ON A SET (exp17): the observed block holds each element's value and its history,
        # seeded from the recording at each origin; the known stimulus, if any, is the drive -- a window of
        # frames written into a set's block before every step. The curriculum is a list of STAGES, one horizon
        # each (plexus2.tex: "a curriculum is a list of stages, each overriding the rollout's horizon").
        if (ref.get("split") or "all") not in ("all", "zapbench", "recording"):
            raise ValueError(f"{path}: a trace_recording task's `split:` is `all` (train and score on every frame, "
                             f"Cedric 2026-09-30), `zapbench` (ZAPBench's own split: train on its training frames, "
                             f"score its unseen test frames, Cedric 2026-10-01) or `recording` (the recording's own "
                             f"per-frame `split` array, 0 train / 2 held out: exp20's held-out trials)")
        if set(t["observe"]) - {"set", "block"}:
            raise ValueError(f"{path}: a trace_recording task observes `set:` and `block:` only")
        if "drive" in t:
            w = t["drive"].get("window")
            if not (isinstance(w, list) and len(w) == 2 and w[0] <= 0 <= w[1]):
                raise ValueError(f"{path}: task.drive.window is [first, last] frame offsets around t, "
                                 f"e.g. [-5, 1] (GraphCast's forcings at the inputs' times and the target's)")
        st = tr.get("stages")
        if not st or any(set(x) != {"horizon", "iters"} for x in st):
            raise ValueError(f"{path}: a trace_recording task trains in `stages:`, each {{horizon, iters}}")
        for j, x in enumerate(st):
            if not (isinstance(x["horizon"], int) and x["horizon"] >= 1 and int(x["iters"]) >= 0):
                raise ValueError(f"{path}: training.stages[{j}] needs an integer horizon >= 1 and iters >= 0")
        if tr.get("select", "last") not in ("last",):
            raise ValueError(f"{path}: training.select {tr.get('select')!r}; implemented: ['last'] -- no frame "
                             f"is held out, so no checkpoint can be picked on held-out data (a declared exception)")
    elif "select" in tr:
        raise ValueError(f"{path}: `select` is read by a trace_recording task only")
    elif any(k in tr for k in ("stages", "render", "lr_min")):
        raise ValueError(f"{path}: `stages`, `render` and `lr_min` belong to a shape task's scheme")
    if (s.get("task") or {}).get("observe", {}).get("elements", "all") != "all":
        raise ValueError(f"{path}: task.observe.elements is `all` (every element of the observed set, one column each) "
                         f"or absent (element 0)")
    if "resume" in tr and kind != "trace_recording":
        raise ValueError(f"{path}: `training.resume` (restart from this run's own models/<name>.pt) is read by a "
                         f"trace_recording task only")
    if "circuit_movie" in tr:                       # the circuit at work (plot_trainer.circuit_movie), a corpus run's
        cm = tr["circuit_movie"]
        if kind != "corpus":
            raise ValueError(f"{path}: `training.circuit_movie` is read by a corpus task only")
        if not isinstance(cm, dict) or set(cm) - {"trials", "fps", "stride"}:
            raise ValueError(f"{path}: training.circuit_movie is {{trials, fps, stride}} (any of them, or {{}})")
    if "record" in t:                               # the test's activity read-out (`test`), a corpus run's
        rc = t["record"]
        if kind != "corpus":
            raise ValueError(f"{path}: `task.record` is read by a corpus task's test only")
        _refuse_unread(f"{path}: task.record", rc, _KEYS["record"])
        if set(rc) != _KEYS["record"] or rc["set"] not in sets or float(rc["every_s"]) <= 0:
            raise ValueError(f"{path}: task.record is {{set, block, every_s}}, a set of the model and a period > 0 s")
    if kind == "corpus" and ("init_from" in tr or int(tr.get("epochs", 1)) == 0):
        # A TEST-ONLY RUN (Cedric, 2026-10-08: the trained two-eye rig on ZAPBench's 2-h session): `epochs: 0` and
        # `init_from:` the trained run, together or not at all -- the run trains nothing, its test and plots read
        # that run's checkpoint (`_restore`) and write into this run's own folder.
        if int(tr.get("epochs", 1)) != 0 or not tr.get("init_from"):
            raise ValueError(f"{path}: a corpus run reads `training.init_from` only as a test-only run: "
                             f"`epochs: 0` and `init_from: <the trained run>`, both")
    s["_kind"] = kind
    for k, allowed in (("optimizer", OPTIMIZERS), ("schedule", SCHEDULES), ("guard", GUARDS)):
        if k in tr and tr[k] not in allowed:
            raise ValueError(f"{path}: training.{k} {tr[k]!r}; implemented: {list(allowed)}")
    s["_path"] = os.path.abspath(path)
    s["_model_dir"] = os.path.basename(os.path.dirname(s["_path"]))
    return s


def out_dir(spec, root=None) -> str:
    return os.path.join(root or log_path(), "training", spec["_model_dir"], spec["name"])


# ============================================================================== the learnables
class Learnables:
    """The free quantities of one run, held here across rollouts and written into each one.

    Each entry is a state block of a set. Its parameter is made the FIRST time a rollout is
    seeded, from the value the model put there -- a connectome's measured weights, a declared
    time constant -- so what training does is a departure from the stated model and can be read
    as one. After that, every rollout writes the same leaf back, functionally (clone, assign,
    publish), so the tape connects the parameter to every later use of the state.
    """

    def __init__(self, entries, device="cpu"):
        self.entries = [dict(e) for e in entries]
        self.device = device
        self.p: dict[str, nn.Parameter] = {}
        self._pre: dict[str, torch.Tensor] = {}   # sender index of an edge-set block, for `sign`
        self._ops: dict = {}                       # the operator holding each {param, op} learnable
        self.scale = 1.0               # a DISPLAY rollout may spread one deformation over more frames
        self.mods: dict[str, nn.Module] = {}      # a `hash` field learnable's encoder, its table in `p`
        self._coords: dict[str, torch.Tensor] = {}

    @staticmethod
    def key(e) -> str:
        if "field" in e:
            return f"field.{e['field']}"
        return f"{e['op']}.{e['param']}" if "param" in e else f"{e['of']}.{e['block']}"

    def _inject_field(self, H, e):
        """A learned FIELD, written whole into the model's field before the first tick. `tensor`:
        the field's own values, starting at the model's (zeros for a declared grid field). `hash`:
        an Instant-NGP multiresolution grid read at every voxel's centre, coordinates in [0, 1] per
        axis, its finest level capped at the voxel -- `levels` x `features` numbers per voxel, a
        spatial prior for free (nearby voxels share table rows at the coarse levels)."""
        fld = H.fields[e["field"]]
        shape = tuple(fld.grid.shape)                                  # [k, *grid]
        k = self.key(e)
        if e.get("with", "tensor") == "tensor":
            if k not in self.p:
                self.p[k] = nn.Parameter(fld.grid.detach().clone())
            fld.grid = self.p[k].clone()
            return
        if k not in self.mods:
            from plexus.models.hashgrid import MultiResHashGrid
            D, L, F = len(shape) - 1, int(e["levels"]), int(e["features"])
            if L * F != shape[0]:
                raise ValueError(f"learnable `{k}`: a hash of {L} levels x {F} features gives {L * F} "
                                 f"numbers per voxel; the field has {shape[0]} components")
            mod = MultiResHashGrid(n_input_dims=D, n_levels=L, n_features_per_level=F,
                                   log2_hashmap_size=int(e.get("log2_table", 16)),
                                   base_resolution=e.get("base", 4), per_level_scale=e.get("scale", 2.0),
                                   max_resolution=list(shape[1:])).to(fld.grid.device)
            if k not in self.p:                                        # else restored: keep its table
                self.p[k] = mod.table
            axes = [(torch.arange(n, device=fld.grid.device, dtype=torch.float32) + 0.5) / n for n in shape[1:]]
            self._coords[k] = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, D)
            self.mods[k] = mod
        mod = self.mods[k]
        mod.table = self.p[k]                                          # always the held leaf
        fld.grid = mod(self._coords[k]).T.reshape(shape)

    def ready(self, H):
        """Parameters of ACTIVITIES, set on the operator instances once they exist (`on_ready`).
        Made the first time from the value the model gave the operator -- its `fit:` file or its
        inline numbers -- and handed to every later instance, so each rollout reads the same leaf."""
        for e in self.entries:
            if "param" not in e:
                continue
            insts = [o for n, o in zip(H.operator_names, H.operators) if n == e["op"]]
            if len(insts) != 1:
                raise ValueError(f"learnable `{self.key(e)}`: {len(insts)} instances of {e['op']!r}")
            par = self.p.get(self.key(e))
            if par is None:
                v = getattr(insts[0], e["param"], None)
                if not torch.is_tensor(v):
                    raise ValueError(f"learnable `{self.key(e)}`: the operator holds no tensor "
                                     f"{e['param']!r} to start from (it has {type(v).__name__})")
                par = self.p[self.key(e)] = nn.Parameter(v.detach().clone().to(self.device))
            setattr(insts[0], e["param"], par)
            self._ops[self.key(e)] = insts[0]                  # for the priors the operator computes itself

    def prior(self, scale=1.0, parts=None):
        """The priors declared WITH the learnables -- claims about the unknown, not the evidence.

            shrink: l          l * mean(x^2)                        toward zero
            shrink_to_mean: l  l * mean((x - mean x)^2)             toward uniform
            smooth: l          l * sum of squared steps along the last axis
            l1: l              l * ||x||_1                          sparse (connectome-gnn coeff_*_L1)
            l2: l              l * ||x||_2                          small (coeff_*_L2) -- a NORM, not a mean
            group_l1: l        l * sum_c ||x[..., c]||_2            whole columns to zero (group lasso)
            row_l1: l          l * sum_i ||x[i, :]||_2              whole rows (elements) to zero: a set block's
                                                                    element loses all of it (exp17: most neurons
                                                                    lose their stimulus input, Cedric)
            sign: l            l * ||std_i tanh(10 x_e)||_2         Dale: one sign per sender i, over
                                                                    the edges it sends (edge-set blocks)

        `scale` is the annealing factor; `parts`, a dict, receives every term by name.
        """
        tot = None
        for e in self.entries:
            x = self.p.get(self.key(e))
            for kind, lam in (e.get("prior") or {}).items():
                lam = float(lam) * float(scale)
                if kind == "shrink":
                    term = x.pow(2).mean()
                elif kind == "shrink_to_mean":
                    term = (x - x.mean()).pow(2).mean()
                elif kind == "smooth" and e.get("with") == "lattice":
                    # IN SPACE, not across components: a lattice is stored flat as [K^3, width], so
                    # the last axis is the rate's six entries. Steps between neighbouring NODES along
                    # x, y and z of the [K, K, K, width] view (found by the exp04 session).
                    K = int(e["K"])
                    g = x.reshape(K, K, K, -1)
                    term = sum((g.narrow(ax, 1, K - 1) - g.narrow(ax, 0, K - 1)).pow(2).sum() for ax in range(3))
                elif kind == "smooth":
                    term = (x[..., 1:] - x[..., :-1]).pow(2).sum()
                elif kind == "l1":
                    term = x.abs().sum()
                elif kind == "l2":
                    term = x.norm(2)
                elif kind == "group_l1":
                    term = x.reshape(-1, x.shape[-1]).norm(2, dim=0).sum()
                elif kind == "row_l1":
                    term = x.reshape(x.shape[0], -1).norm(2, dim=1).sum()
                elif kind in ("monotone", "pin", "input_group_l1", "dale", "dale_keep"):
                    # A PRIOR ONLY THE MODEL CAN EVALUATE (connectome-gnn's g_phi_diff, g_phi_norm, input-group
                    # lasso): the activity that holds the parameter computes it from its own current values.
                    op = self._ops.get(self.key(e))
                    if op is None or not hasattr(op, "prior_term"):
                        raise ValueError(f"prior `{kind}` on `{self.key(e)}`: the operator computes no such prior")
                    term = op.prior_term(e["param"], kind)
                else:
                    pre = self._pre.get(self.key(e))
                    if pre is None:
                        raise ValueError(f"prior `sign` on `{self.key(e)}`: it groups an edge set's "
                                         f"weights by sender, and `{e.get('of')}` is not an edge set")
                    v = torch.tanh(10.0 * x.reshape(-1))
                    n = int(pre.max()) + 1
                    cnt = torch.zeros(n, device=v.device).index_add(0, pre, torch.ones_like(v))
                    mu = torch.zeros(n, device=v.device).index_add(0, pre, v) / cnt.clamp(min=1)
                    m2 = torch.zeros(n, device=v.device).index_add(0, pre, v * v) / cnt.clamp(min=1)
                    term = ((m2 - mu * mu).clamp(min=0) * (cnt > 1)).sqrt().norm(2)
                tot = lam * term if tot is None else tot + lam * term
                if parts is not None:
                    parts[f"prior.{self.key(e)}.{kind}"] = float((lam * term).detach())
        return tot

    def inject(self, H):
        for e in self.entries:
            if "param" in e:
                continue
            if "field" in e:
                self._inject_field(H, e)
                continue
            lvl = H.level(e["of"])
            if e["block"] not in lvl.state_schema:
                raise ValueError(f"learnable `{self.key(e)}`: {e['of']!r} has no block "
                                 f"{e['block']!r} (it has {sorted(lvl.state_schema)})")
            a, b = lvl.state_schema[e["block"]]
            par = self.p.get(self.key(e))
            if "sign" in (e.get("prior") or {}) and self.key(e) not in self._pre \
                    and getattr(lvl, "is_edge_set", False):
                self._pre[self.key(e)] = lvl.incidence("pre").detach().long()
            if e.get("with", "tensor") == "lattice":
                if par is None:
                    # A LATTICE STARTS WHERE THE MODEL DOES ONLY WHEN THE MODEL SAYS "NOTHING": a
                    # uniform zero block is exactly the zero lattice, and anything else would need
                    # a fit to reproduce it, which is not implemented and so is refused.
                    if float(lvl.state[:, a:b].abs().max()) > 0:
                        raise ValueError(f"learnable `{self.key(e)}`: a lattice starts from a zero "
                                         f"block; the model seeds a non-zero one")
                    K = int(e["K"])
                    par = self.p[self.key(e)] = nn.Parameter(
                        torch.zeros(K ** 3, b - a, device=lvl.state.device, dtype=lvl.state.dtype))
                val = self._lattice(e, par, lvl.get("pos").detach())
            else:
                if par is None:
                    par = self.p[self.key(e)] = nn.Parameter(lvl.state[:, a:b].detach().clone())
                val = par
            st = lvl.state.clone()
            st[:, a:b] = val if self.scale == 1.0 else val * self.scale
            lvl.state = st

    @staticmethod
    def _lattice(e, par, X0):
        """The block's value at each element: trilinear in a K^3 cube of control nodes, read at the
        element's MATERIAL coordinate X0 -- its position at set-up, so the field is Lagrangian and
        means the same thing at any number of elements.

            a(X0) = sum_c w_c(X0) theta_c        w = trilinear weights, c = the 8 corners
        """
        K = int(e["K"])
        cx, cy, cz, half = (float(v) for v in e["extent"])
        c = torch.tensor([cx, cy, cz], device=X0.device, dtype=X0.dtype)
        u = (((X0 - c) / (2.0 * half) + 0.5) * (K - 1)).clamp(0.0, K - 1.001)
        b = u.floor().long()
        f = u - b.to(u.dtype)
        val = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (((1 - f[:, 0]) if dx == 0 else f[:, 0])
                         * ((1 - f[:, 1]) if dy == 0 else f[:, 1])
                         * ((1 - f[:, 2]) if dz == 0 else f[:, 2]))
                    i = (((b[:, 0] + dx).clamp(0, K - 1) * K + (b[:, 1] + dy).clamp(0, K - 1)) * K
                         + (b[:, 2] + dz).clamp(0, K - 1))
                    val = val + w[:, None] * par[i]
        return val

    def parameters(self) -> list:
        return [self.p[self.key(e)] for e in self.entries]

    def groups(self, lr) -> list:
        return [{"params": [self.p[self.key(e)]], "lr": float(e.get("lr", lr))}
                for e in self.entries]

    @torch.no_grad()
    def clamp_(self):
        for e in self.entries:
            if "bounds" in e:
                lo, hi = (None if v is None else float(v) for v in e["bounds"])
                self.p[self.key(e)].clamp_(min=lo, max=hi)

    def snapshot(self) -> dict:
        return {k: v.detach().clone() for k, v in self.p.items()}

    @torch.no_grad()
    def restore(self, values: dict):
        for k, v in values.items():
            if k in self.p:
                self.p[k].copy_(v.to(self.p[k].device))
            else:
                self.p[k] = nn.Parameter(v.to(self.device).clone())


# ============================================================================== the task
def write_drive(H, frame, u, drive_set, drive_block):
    """Write frame `frame` of the stimulus into the drive set's block.

    ONE CHANNEL PER SENSORY ELEMENT, not one value broadcast over all of them. A corpus with C
    channels -- a stimulus plus, for a teacher-varying grid, a one-hot of the condition cell --
    has to arrive on C DIFFERENT lines or the circuit cannot tell them apart. Element i takes
    channel i; a drive set wider than the corpus leaves its spare lines as they were seeded.
    """
    lvl = H.level(drive_set)
    a, b = lvl.state_schema[drive_block]
    t = min(frame, u.shape[-2] - 1)
    st = lvl.state.clone()
    C = u.shape[-1]
    if C == 1:
        st[..., a:b] = u[..., t, :].unsqueeze(-2).expand(*st.shape[:-1], b - a)
    else:
        if lvl.n < C:
            raise ValueError(
                f"the corpus has {C} input channels but `{drive_set}` has only {lvl.n} "
                f"element(s). Give the drive set at least one element per channel -- with a "
                f"context one-hot the channel count is 1 + the number of condition cells.")
        st[..., :C, a:a + 1] = u[..., t, :].unsqueeze(-1)
    lvl.state = st


def rollout(sim, learn, u, task, device="cpu", grad=True, watch=None):
    """Drive the model with `u` ([T, C] or [B, T, C]) and return (H, observed trace).

    The trace is the observed set's element 0 at the end of every tick, trial-major like the
    corpus: [T, w] or [B, T, w]. `watch(H)` is called at the same moment, for a caller that needs
    more of the state than the observable (the analysis reads the voltages).

    THE CLOCK IS A RECURRENT NETWORK'S: h_k = f(h_{k-1}, u_k), y_k = g(h_k) -- trace frame k has seen
    the drive up to and including u_k, and no further. Two engine facts make that take care:
      - `on_frame` fires at the END of a tick, after its step, so a drive written there is read by
        the NEXT tick. u_0 is therefore written at `on_ready` (after the batch axis exists, before
        tick 0) and u_{k+1} at the end of tick k. Writing u_k at the end of tick k -- as this did
        until 2026-10-08 -- made tick 0 step on the seeded drive (0) and every later tick on the
        previous frame's input.
      - a block the engine does not integrate (`integration: none`, e.g. `readout`'s `into:`) is
        written IN PLACE during the tick, from the state BEFORE that tick's step: at the end of tick
        k it describes the state of frame k-1. Its trace is read one hook later (frame 0 dropped),
        so that it describes the same step as the integrated blocks of frame k.
    The cost of the old clock was measured on the HD twin with the reference's trained weights
    transplanted (tools/hd_transplant_check.py): heading RMSE 2.09 deg instead of the reference's
    1.50 deg over 512 held-out trials -- the input two 10-ms frames late, at turns of ~150 deg/s.
    """
    drv, obs = task["drive"], task["observe"]
    ch_all = int(obs.get("channel", 0)) if obs.get("elements") == "all" else None
    batch = int(u.shape[0]) if u.dim() == 3 else 1
    if u.dim() == 3 and batch == 1:                        # the engine runs unbatched at B = 1
        u = u[0]
    trace = []

    def ready(H):
        learn.ready(H)
        write_drive(H, 0, u, drv["set"], drv["block"])     # u_0, read by tick 0

    def hook(H, frame):
        v_ = H.level(obs["set"]).get(obs["block"])
        # `observe.elements: all` (the two-eye task, Cedric 2026-10-08): channel `channel` of EVERY element, one
        # column per element, scored column for column against the corpus's targets; otherwise element 0, all channels
        trace.append(v_[..., :, ch_all].clone() if ch_all is not None else v_[..., 0, :].clone())
        if watch is not None:
            watch(H)
        write_drive(H, frame + 1, u, drv["set"], drv["block"])   # u_{k+1}, read by tick k+1

    # `on_ready` hands every {param:, op:} learnable to its operator instance (`Learnables.ready`), as
    # the recording trainer does; a spec with none is unaffected (exp18's broadcast angles needed it).
    H, _ = engine.run(sim, device=device, progress=False, grad=grad, on_frame=hook,
                      batch=batch, on_seeded=learn.inject, on_ready=ready)
    if not trace:
        return H, None
    y = torch.stack(trace)
    if _written_in_place(H, obs):
        y = y[1:]
    return H, y.transpose(0, 1) if batch > 1 else y


def _written_in_place(H, obs) -> bool:
    """True when the observed block is not engine-integrated, so its value at the end of tick k was computed from the
    state before tick k's step (`rollout`)."""
    blk = next((b for b in H.level(obs["set"]).state_schema.blocks if b.name == obs["block"]), None)
    return blk is not None and blk.integration == "none"


def _mse(y, target, ch):
    """Squared error, aligned at frame 0 and truncated to the shorter series; batch-transparent.

    The mean is over frames AND trials. Indexed from the right, so [T, w] against [T, 1] and
    [B, T, w] against [B, T, 1] are the same call.
    """
    n = min(y.shape[-2], target.shape[-2])
    if target.shape[-1] > 1:                       # several targets (`observe.elements: all`): column for column
        K = target.shape[-1]
        return ((y[..., :n, :K] - target[..., :n, :K]) ** 2).mean()
    return ((y[..., :n, ch] - target[..., :n, 0]) ** 2).mean()


def _log_mse(m, m_ref):
    """Squared error of log(1 + nodal count) on the grid -- the Eulerian shape loss.

    A target shape has no particle correspondence, so a position loss has nothing to match where
    the two shapes differ; comparing where material IS on a grid does. `log` because the two
    fields differ by orders of magnitude between "material" and "none".
    """
    return ((torch.log1p(m) - torch.log1p(m_ref)) ** 2).mean()


LOSSES = {"mse": _mse, "log_mse": _log_mse}


def _corpus(spec):
    return spec["task"]["reference"]["corpus"]


def _unit(spec):
    u = spec["task"]["observe"].get("unit")
    return (f" {u}", f" {u}^2") if u else ("", "")


def _data(spec, split, n_cond, device):
    U, Y, c = load_split(_corpus(spec), split, device)
    return with_context(U, c, n_cond), Y, c


def _model(spec, train=True, resolution=None, n_frames=None):
    """The model, as TRAINING or as DISPLAY runs it. Same file, same law, two numerical choices.

    TRAINING picks each operator's `differentiable` implementation where the model pins none --
    the default MPM bodies write in place, which a captured CUDA graph needs and autograd cannot
    have. DISPLAY leaves the choice to the engine, which picks its fastest bodies (warp on CUDA).

    SUBSTEP CAPTURE IS OFF IN BOTH, for two measured reasons. Under a tape, a replayed graph writes
    its static buffers in place, so a tensor autograd saved at tick 0 is a different tensor at
    backward time ("[5000] ... at version 2; expected version 1"). And with warp bodies, a captured
    substep does not see a deformation gradient written by an operator OUTSIDE the block: the same
    trained morph ended as the untouched ball (10.85 x 10.67 x 10.65 um) with capture and as the
    trained cow (14.13 x 7.00 x 4.32 um) without it, or with the torch bodies. That is an engine
    defect, not fixed here; `resolution` resizes sets (`n`) and fields (`n_grid`) for a stage.
    """
    sim = load_model(spec["model"])
    if sim.learnable:
        raise ValueError(f"{spec['model']} declares `learnable:` itself. What is learnable belongs "
                         f"to the training spec; the model file stays the forward description.")
    for name, v in (resolution or {}).items():
        if name in sim.sets:
            sim.sets[name]["n"] = int(v)
        else:
            sim.fields[name]["n_grid"] = int(v)
    if n_frames is not None:
        sim.n_frames = int(n_frames)
    if train:
        for o in sim.operators:
            if o.impl is None and "differentiable" in get_contract(o.op).implementations:
                o.impl = "differentiable"
            o.params["_train"] = True       # a law that trains differently from how it tests reads it (exp17 batch 21)
    for blk in sim.schedule:
        if isinstance(blk, dict) and "steps" in blk:
            blk["capture"] = False
    return sim


# ============================================================================== train
def train(spec, device="cpu", root=None):
    if spec.get("_kind") == "shape":
        return _train_shape(spec, device, root)
    if spec.get("_kind") == "recording":
        return _train_recording(spec, device, root)
    if spec.get("_kind") == "field_recording":
        return _train_field(spec, device, root)
    if spec.get("_kind") == "trace_recording":
        return _train_trace(spec, device, root)
    tr, task = spec["training"], spec["task"]
    if int(tr.get("epochs", 1)) == 0:
        raise ValueError(f"{spec['name']} is a test-only run (`epochs: 0`, `init_from: {tr.get('init_from')}`): "
                         f"run `-o test` / `-o plot`, not `train`")
    ref = task["reference"]
    torch.manual_seed(int(tr.get("seed", 0)))
    out = out_dir(spec, root)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    shutil.copyfile(spec["_path"], os.path.join(out, "config.yaml"))
    shutil.copyfile(spec["model"], os.path.join(out, "model.yaml"))
    print(f"[run] {spec['name']} -> {out}")

    corpus = _corpus(spec)
    n_cond = n_condition_cells(corpus) if ref.get("context", True) else 1
    U, Y, cU = _data(spec, "train", n_cond, device)
    has_val = os.path.isdir(os.path.join(task_dir(corpus), "val"))
    Uv, Yv, cV = _data(spec, "val", n_cond, device) if has_val else (U[:8], Y[:8], cU[:8])
    if n_cond > 1:
        print(f"[data] {n_cond} condition cells -> a one-hot context channel is appended "
              f"({U.shape[2] - n_cond} -> {U.shape[2]} inputs)")
    n_tr = min(int(ref.get("n_train", 32)), U.shape[0])
    n_va = min(int(ref.get("n_val", 8)), Uv.shape[0])
    print(f"[data] corpus {corpus}: using {n_tr} of {U.shape[0]} train trials, "
          f"{n_va} val  ({U.shape[1]} frames each)")

    engine.quiet(True)
    sim = _model(spec)
    learn = Learnables(spec["learnable"], device)
    # THE FIRST ROLLOUT MAKES THE PARAMETERS, from the seeded state, and proves the drive and the
    # observable name real blocks before an optimiser exists to hide a failure behind a loss.
    rollout(sim, learn, U[0], task, device, grad=True)
    params = learn.parameters()
    n_par = sum(p.numel() for p in params)
    print(f"[fit] {len(params)} tensor(s), {n_par} values: " + ", ".join(learn.p))

    epochs = int(tr.get("epochs", 30))
    batch = max(1, int(tr.get("batch", 8)))
    T_full = int(sim.n_frames)
    # TRUNCATED BPTT OVER A GROWING PREFIX: short horizons first, where the gradient still reaches
    # the start of the trajectory, then longer. Duplicates dropped, so a floor that collapses the
    # early stages costs no epochs.
    horizon = sorted({max(int(tr.get("horizon_min", 60)), int(T_full * f))
                      for f in tr.get("horizon", [0.125, 0.25, 0.5, 0.75, 1.0])})
    print(f"[fit] horizon curriculum {horizon} frames of {T_full};  "
          f"gradient averaged over {batch} trials per step, ONE rollout per step")
    lr = float(tr.get("lr", 1e-2))
    opt = torch.optim.Adam(learn.groups(lr), lr=lr)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
           if tr.get("schedule", "cosine") == "cosine" else None)
    clip = float(tr.get("clip", 1.0))
    guard = tr.get("guard", "restore_and_halve")
    loss_fn = _mse                                         # the validation METRIC; the loss is _objective
    ch = int(task["observe"].get("channel", 0))
    U1, U2 = _unit(spec)
    hist, best, best_state, t0 = [], np.inf, None, time.time()
    last_ok = learn.snapshot()

    for ep in range(epochs):
        h = horizon[min(len(horizon) - 1, int(len(horizon) * ep / max(epochs, 1)))]
        sim.n_frames = h
        perm = torch.randperm(n_tr)
        tot, n_step, n_bad, terms = 0.0, 0, 0, {}
        for k in range(0, n_tr - batch + 1, batch):
            idx = perm[k:k + batch]
            _, y = rollout(sim, learn, U[idx], task, device, grad=True)
            Yb = Y[idx]
            nf = min(y.shape[-2], Yb.shape[-2])
            if Yb.shape[-1] > 1:                       # several targets: one column per observed element
                r, ref_ = y[..., :nf, :Yb.shape[-1]] - Yb[..., :nf, :], Yb[..., :nf, :]
            else:
                r, ref_ = y[..., :nf, ch] - Yb[..., :nf, 0], Yb[..., :nf, 0]
            parts = {}
            loss = _objective(spec, learn, {"mse": lambda red: _reduce(r, red, ref_)}, ep, parts)
            for kk, vv in parts.items():
                terms[kk] = terms.get(kk, 0.0) + vv
            if not torch.isfinite(loss) and guard == "restore_and_halve":
                # A DIVERGED ROLLOUT ENDS THE STEP, NOT THE RUN: the last finite values come back
                # and every step size halves. Without it one NaN poisons Adam's moments for good.
                learn.restore(last_ok)
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                n_bad += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            learn.clamp_()
            last_ok = learn.snapshot()
            tot += float(loss.detach()); n_step += 1
        if sch is not None:
            sch.step()
        sim.n_frames = T_full                              # always scored on the full trial
        with torch.no_grad():
            _, yv = rollout(sim, learn, Uv[:n_va], task, device, grad=False)
            v = float(loss_fn(yv, Yv[:n_va], ch))
        if int(tr.get("snapshot_every", 0)) and ep % int(tr["snapshot_every"]) == 0:
            from plexus.tasks import plot_trainer as PT
            PT.snapshot(out, ep, ep * max(n_step, 1),
                        Uv[:4, :, :1].cpu().numpy(), Yv[:4].cpu().numpy(),
                        yv[:4].detach().cpu().numpy(), dt=float(sim.dt), unit=U1.strip(),
                        title=f"{spec['name']}  epoch {ep}")
        tr_mse = tot / max(n_step, 1)
        hist.append({"epoch": ep, "horizon": h, "train_mse": tr_mse, "val_mse": v,
                     "diverged_steps": n_bad,
                     "terms": {kk: vv / max(n_step, 1) for kk, vv in terms.items()}})
        print(f"  ep {ep:3d}  horizon {h:4d}  train {tr_mse:10.5f}  val {v:10.5f}{U2}"
              f"   (|err| {np.sqrt(v):.4f}{U1} rms)" + (f"  [{n_bad} diverged]" if n_bad else ""))
        if v < best:
            best, best_state = v, learn.snapshot()

    torch.save({"fitted": best_state, "model": spec["model"], "task": task,
                "learnable": spec["learnable"], "n_cond": int(n_cond)},
               os.path.join(out, "models", "best.pt"))
    rep = {"name": spec["name"], "model": spec["model"], "corpus": corpus,
           "n_params": n_par, "n_trials": n_tr, "epochs": epochs, "batch": batch,
           "best_val_mse": best, "history": hist,
           "target_variance": float((Y[:n_tr] ** 2).mean()),
           "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(out, "results", "report.json"), "w"), indent=2)
    print(f"[done] best val {best:.5f}{U2} ({np.sqrt(best):.4f}{U1} rms) "
          f"({best / rep['target_variance']:.4f} of target variance) in {rep['seconds']:.0f} s")
    return out


def _restore(spec, device="cpu", root=None):
    """The model, the trained learnables and the run folder -- everything a held-out rollout needs. A test-only run
    (`training: {epochs: 0, init_from: <run>}`, `load`) reads the named run's checkpoint and keeps its own folder."""
    out = out_dir(spec, root)
    tr = spec.get("training") or {}
    src = out_dir(load(str(tr["init_from"])), root) if int(tr.get("epochs", 1)) == 0 else out
    ck = torch.load(os.path.join(src, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    return _model(spec), learn, ck, out


# ============================================================================== test
def test(spec, device="cpu", root=None):
    """Roll the kept checkpoint out on held-out trials. One number per trial, never only a mean."""
    if spec.get("_kind") == "shape":
        return _test_shape(spec, device, root)
    if spec.get("_kind") == "recording":
        return _test_recording(spec, device, root)
    if spec.get("_kind") == "trace_recording":
        return _test_trace(spec, device, root)
    if spec.get("_kind") == "field_recording":
        if spec["task"]["reference"].get("split", "holdout") == "all":
            return _test_field_full(spec, device, root)
        return _test_field(spec, device, root)
    engine.quiet(True)
    sim, learn, ck, out = _restore(spec, device, root)
    task, corpus = spec["task"], _corpus(spec)
    split = "test" if os.path.isdir(os.path.join(task_dir(corpus), "test")) else "train"
    U, Y, cond = _data(spec, split, int(ck.get("n_cond", 1)), device)
    n = min(int(task["reference"].get("n_test", 24)), U.shape[0])
    ch = int(task["observe"].get("channel", 0))
    sim.n_frames = int(U.shape[1])                   # the corpus's trial, whatever length the model file declares
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    rec = _Record(task.get("record"), float(sim.dt), int(U.shape[1]))
    with torch.no_grad():
        _, Yp = rollout(sim, learn, U[:n], task, device, grad=False, watch=rec if rec.on else None)
    Yp = Yp if Yp.dim() == 3 else Yp[None]           # one trial: the engine ran unbatched, `rollout` returns [T, w]
    if rec.on:
        rec.save(os.path.join(out, "results", f"{spec['name']}_{split}"), n)
    per = [float(_mse(Yp[i], Y[i], ch)) for i in range(n)]
    traces = [Yp[i, :, ch].cpu().numpy() for i in range(min(n, 6))]
    var = float((Y[:n] ** 2).mean())
    nn_ = min(Yp.shape[-2], Y.shape[-2])
    err = (Yp[:n, :nn_, ch] - Y[:n, :nn_, 0]).cpu().numpy()
    # THE SETTLED SCORE BESIDE THE FULL ONE: every rollout starts from the model's seeded state,
    # so the opening fraction of a second is the circuit catching up from rest.
    settle_s = float(task.get("settle_s", 0.5))
    k = min(int(round(settle_s / float(sim.dt))), nn_ - 1)
    yl = Y[:n, k:nn_, 0].cpu().numpy()
    mse_settled = float((err[:, k:] ** 2).mean())
    var_settled = float((yl ** 2).mean())
    cells = sorted(set(cond[:n].tolist()))
    teacher = torch.load(os.path.join(task_dir(corpus), "teacher.pt"), weights_only=False)
    res = {"name": spec["name"], "model": spec["model"], "corpus": corpus, "split": split,
           "n_trials": n, "mse": float(np.mean(per)), "mse_per_trial": per,
           # PER CONDITION CELL, EACH AGAINST ITS OWN TARGET VARIANCE: a pooled normalised number
           # moves the wrong way when a louder law joins, so it cannot say whether one circuit
           # holds several laws or drops some.
           "mse_per_cell": {str(c): float(np.mean([per[i] for i in range(n) if cond[i] == c]))
                            for c in cells},
           "normalised_per_cell": {
               str(c): float(np.mean([per[i] for i in range(n) if cond[i] == c])
                             / max(float((Y[:n][cond[:n] == c, :, 0] ** 2).mean()), 1e-30))
               for c in cells},
           "cell_names": list(teacher.get("names") or ["all"]),
           "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt(np.mean(per))),
           "unit": task["observe"].get("unit"),
           "target_variance": var, "normalised_mse": float(np.mean(per)) / var,
           "settle_s": settle_s, "mse_settled": mse_settled,
           "normalised_mse_settled": mse_settled / var_settled if var_settled > 0 else None,
           "fitted": {k: {"n": int(v.numel()), "mean": float(v.mean()), "sd": float(v.std())}
                      for k, v in ck["fitted"].items()}}
    p = os.path.join(out, "results", f"{spec['name']}_{split}.json")
    json.dump(res, open(p, "w"), indent=2)
    np.save(os.path.join(out, "results", f"{spec['name']}_{split}_traces.npy"), np.array(traces))
    U1, U2 = _unit(spec)
    print(f"[test] {split}: mean |err| {res['mae']:.4f}{U1}   rmse {res['rmse']:.4f}{U1}")
    print(f"[test] {split}: {n} trials  mse {res['mse']:.5f}{U2}  "
          f"({res['normalised_mse']:.5f} of target variance)")
    if len(cells) > 1:
        print("[test] per cell, of its own variance: " + "  ".join(
            f"{res['cell_names'][int(c)] if int(c) < len(res['cell_names']) else c} "
            f"{res['normalised_per_cell'][str(c)]:.5f}" for c in cells))
    print(f"[test] wrote {p}")
    return res


class _Record:
    """`task.record: {set, block, every_s}` -- a block of a set, sampled every `every_s` seconds through the test's
    rollout (Cedric, 2026-10-08: the two-eye rig's 285 cells on ZAPBench's 0.914-s volume clock). Sample i is trace
    frame round(i every_s / dt) (`rollout`'s clock: it has seen the drive up to that frame); a block written in place
    is read one hook later, as `rollout` reads its observable. -> <stem>_<set>_<block>.npz: `x` [trials, samples,
    elements, width] and `t_s` [samples]."""

    def __init__(self, rc, dt, T):
        self.on = bool(rc)
        if not self.on:
            return
        self.set, self.block = rc["set"], rc["block"]
        k = np.round(np.arange(0.0, T * dt, float(rc["every_s"])) / dt).astype(int)
        self.frames = k[k < T]
        self.t_s = self.frames * dt
        self.tick, self.shift, self.x = 0, None, []

    def __call__(self, H):
        if self.shift is None:
            self.shift = 1 if _written_in_place(H, {"set": self.set, "block": self.block}) else 0
            self.want = set((self.frames + self.shift).tolist())
        if self.tick in self.want:
            self.x.append(H.level(self.set).get(self.block).detach().float().cpu())
        self.tick += 1

    def save(self, stem, n):
        x = torch.stack(self.x).numpy()                                 # [samples, (trials,) elements, width]
        x = x[:, None] if x.ndim == 3 else x
        x = np.moveaxis(x, 1, 0)[:n]                                    # [trials, samples, elements, width]
        p = f"{stem}_{self.set}_{self.block}.npz"
        np.savez_compressed(p, x=x, t_s=self.t_s[:x.shape[1]])
        print(f"[test] recorded {self.set}.{self.block} {tuple(x.shape)} every {np.diff(self.t_s).mean():.3f} s "
              f"on average (each sample the nearest model frame) -> {p}")


# ============================================================================== analyse
def circuit_poles(H, fitted, slope=None):
    """The trained circuit's poles, linearised where it operates: A = -diag(a) + diag(g) W diag(rho'(v)).

    a is the leak (1/tau, per neuron when a `tau` block exists), g the coupling gain, W the
    recurrent matrix and rho'(v) = 1 - tanh(v)^2 averaged over a rollout. EVERYTHING IS READ FROM
    `H`, WHICH MUST BE A TRAINED ROLLOUT. The analyser this replaces read W from the checkpoint but
    tau from a fresh, untrained run of the model, so whenever tau was learnable its poles mixed a
    trained connectome with the starting time constants.

    The recurrent edge set is found structurally -- the fitted edge set whose two endpoints are
    the same set -- not by its name. Under sign-lock the effective weight is |w| times the
    sender's sign. Returns (poles_at_operating_point, poles_at_origin), or None when no recurrent
    block was fitted.
    """
    rec = next((k for k in fitted if k.endswith(".w")
                and (lambda e: e.is_edge_set and e.pre_name == e.post_name)(H.level(k.split(".")[0]))),
               None)
    if rec is None:
        return None
    es = H.level(rec.split(".")[0])
    W = es.get("w").detach().cpu().numpy().ravel()
    lvl = H.level(es.post_name)
    n = lvl.n
    sig = dict(zip(H.operator_names, H.operators)).get("neuron_signal")
    if sig is not None and getattr(sig, "dale", False):
        W = np.abs(W) * sig._dale_sign(lvl, es).squeeze(-1).detach().cpu().numpy()
    M = np.zeros((n, n))
    M[es.post.cpu().numpy(), es.pre.cpu().numpy()] = W
    P = (lvl.type_params[lvl.node_type] if getattr(lvl, "type_params", None) is not None
         else torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]).expand(n, 6))
    a = P[:, 0].detach().cpu().numpy()                  # leak, 1/s
    g = P[:, 2].detach().cpu().numpy()                  # coupling gain, dimensionless
    upd = dict(zip(H.operator_names, H.operators)).get("neuron_update")
    tb = getattr(upd, "tau_block", None)
    if tb is not None and tb in lvl.state_schema:
        a = 1.0 / lvl.get(tb)[:, 0].detach().cpu().numpy().clip(min=float(getattr(upd, "tau_min", 1e-4)))
    rho = np.ones(n) if slope is None else np.asarray(slope, float).reshape(n)
    return (np.linalg.eigvals(-np.diag(a) + np.diag(g) @ M @ np.diag(rho)),
            np.linalg.eigvals(-np.diag(a) + np.diag(g) @ M))


def analyse(spec, device="cpu", root=None):
    """Did it recover the law, or only reduce the error? The figure, and the poles."""
    if spec.get("_kind") == "shape":
        return _analyse_shape(spec, device, root)
    if spec.get("_kind") == "recording":
        print("[analyse] recording tasks: the per-beat scores are the analysis (see test)")
        return None
    if spec.get("_kind") == "field_recording":
        return _analyse_field(spec, device, root)
    if spec.get("_kind") == "trace_recording":
        return _analyse_trace(spec, device, root)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks.render import BG, INK, MUTED, _ax

    engine.quiet(True)
    sim, learn, ck, out = _restore(spec, device, root)
    task, corpus = spec["task"], _corpus(spec)
    split = "test" if os.path.isdir(os.path.join(task_dir(corpus), "test")) else "train"
    U, Y, _c = _data(spec, split, int(ck.get("n_cond", 1)), device)
    rep = json.load(open(os.path.join(out, "results", "report.json")))
    res_p = os.path.join(out, "results", f"{spec['name']}_{split}.json")
    res = json.load(open(res_p)) if os.path.exists(res_p) else {}
    ch = int(task["observe"].get("channel", 0))
    with torch.no_grad():
        _, Yp = rollout(sim, learn, U[:4], task, device, grad=False)
        vs = []
        H1, _ = rollout(sim, learn, U[0], task, device, grad=False,
                        watch=lambda H: vs.append(H.level("neuron").get("voltage").squeeze(-1).clone())
                        if "neuron" in H.levels else None)
    preds = [Yp[i, :, ch].cpu().numpy() for i in range(4)]
    dt = float(sim.dt)
    t = np.arange(len(preds[0])) * dt
    U1, U2 = _unit(spec)
    poles, v_typ, slope = None, float("nan"), None
    if vs:
        V = torch.stack(vs)
        slope, v_typ = (1.0 - torch.tanh(V) ** 2).mean(0).cpu().numpy(), float(V.abs().mean())
        poles = circuit_poles(H1, ck["fitted"], slope=slope)

    fig = plt.figure(figsize=(14.5, 7.6), facecolor=BG)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)
    axa = _ax(fig.add_subplot(gs[0, 0]), ylabel="input", letter="a")
    for i in range(4):
        axa.plot(t[:U.shape[1]], U[i, :, 0].cpu(), color="#4a4a4a", lw=0.7, alpha=0.8)
    axb = _ax(fig.add_subplot(gs[0, 1]), ylabel=f"{task['observe']['set']}.{task['observe']['block']}{U1}",
              letter="b")
    for i, pr in enumerate(preds):
        n = min(len(pr), Y.shape[1])
        axb.plot(t[:n], Y[i, :n, 0].cpu(), color="#2e8b4f", lw=2.2, alpha=0.9)
        axb.plot(t[:n], pr[:n], color="#000000", lw=0.9)
    axb.text(0.985, 0.04, "green: ground truth   black: model", transform=axb.transAxes,
             ha="right", va="bottom", color=MUTED, fontsize=8)
    axd = _ax(fig.add_subplot(gs[1, 0]), xlabel="Re(lambda) (1/s)", ylabel="Im/2pi (Hz)", letter="d")
    truth = torch.load(os.path.join(task_dir(corpus), "teacher.pt"), weights_only=False)
    tp = truth["per_cell"][0]
    if poles is not None:
        at_op, at_origin = poles
        axd.scatter(at_origin.real, at_origin.imag / (2 * np.pi), s=10, color="0.72", alpha=0.7)
        axd.scatter(at_op.real, at_op.imag / (2 * np.pi), s=12, color="#000000", alpha=0.9)
    axd.scatter(tp["poles_real"], np.asarray(tp["poles_imag"]) / (2 * np.pi), s=80, marker="x",
                color="#2e8b4f", lw=2.0, zorder=5)
    axd.axvline(0, color="#c0272a", lw=0.8, ls="--")
    axd.text(0.985, 0.04, "green: ground truth   black: at the operating point   grey: at v=0",
             transform=axd.transAxes, ha="right", va="bottom", color=MUTED, fontsize=7.5)
    axe = _ax(fig.add_subplot(gs[1, 1]), xlabel="time (s)", ylabel=f"residual{U1}", letter="e")
    for i, pr in enumerate(preds):
        n = min(len(pr), Y.shape[1])
        axe.plot(t[:n], pr[:n] - Y[i, :n, 0].cpu().numpy(), color="#c0522a", lw=0.8)
    axe.set_ylim(axb.get_ylim())
    axf = _ax(fig.add_subplot(gs[1, 2]), xlabel="epoch", ylabel="mse", letter="f")
    ep = [h["epoch"] for h in rep["history"]]
    axf.plot(ep, [h["train_mse"] for h in rep["history"]], color="#1f6fb8", lw=1.1, label="train")
    axf.plot(ep, [h["val_mse"] for h in rep["history"]], color="#9a7d1a", lw=1.1, label="val")
    axf.set_yscale("log")
    for x in axf.legend(frameon=False, fontsize=8, loc="upper right").get_texts():
        x.set_color(INK)
    axc = _ax(fig.add_subplot(gs[0, 2]), letter="c")
    axc.axis("off")
    blocks = [f"{k}({v.numel()})" for k, v in ck["fitted"].items()]
    pr_max = list(tp.get("poles_real") or [])
    lines = ["", spec["name"], f"model  {os.path.basename(spec['model'])}", f"corpus {corpus}",
             f"split  {split}", "", f"fitted {rep['n_params']} values over {len(blocks)} block(s):"]
    lines += ["   " + "  ".join(blocks[i:i + 2]) for i in range(0, len(blocks), 2)]
    lines += ["", f"mean |err|  {res.get('mae', float('nan')):9.4f}{U1}",
              f"best val    {rep['best_val_mse']:9.5f}{U2}",
              f"{split} mse    {res.get('mse', float('nan')):9.5f}{U2}",
              f"normalised  {res.get('normalised_mse', float('nan')):9.5f} of target variance",
              f"trained on  {rep['n_trials']} trials, {rep['epochs']} epochs, {rep['seconds']:.0f} s"]
    if poles is not None:
        at_op, at_origin = poles
        lines += ["", f"|v| at operation {v_typ:.3f}  (rho' = {float(np.mean(slope)):.3f})",
                  f"max Re(lam)  v=0 {float(max(p.real for p in at_origin)):+.4f}   "
                  f"op {float(max(p.real for p in at_op)):+.4f}   truth "
                  + (f"{max(pr_max):+.4f}" if pr_max else "n/a") + "  1/s"]
    axc.text(0.02, 0.95, "\n".join(lines), transform=axc.transAxes, va="top", ha="left",
             color=INK, fontsize=8, family="monospace")
    p = os.path.join(out, "results", f"{spec['name']}_{split}.png")
    fig.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    # THE MOVIE THE WATCHER SHOWS (`tools/exp_record.py` links results/movie.mp4 of a training run).
    # One held-out trial per condition cell, at most six, so a multi-law circuit shows every law it
    # holds. A fit that lands without its movie is still a fit, so a failure here is printed, not raised.
    try:
        from plexus.tasks.plot_trainer import rollout_movie
        cells = _c.tolist()
        pick = [cells.index(c) for c in sorted(set(cells))][:6] or [0]
        with torch.no_grad():
            _, Ym = rollout(sim, learn, U[pick], task, device, grad=False)
        n_m = min(Ym.shape[-2], Y.shape[1])
        rollout_movie(U[pick][..., :1].cpu().numpy(), Y[pick][:, :n_m, :1].cpu().numpy(),
                      Ym[..., :n_m, ch:ch + 1].cpu().numpy(), out=os.path.join(out, "results", "movie.mp4"),
                      dt=dt, fps=48, unit=(task["observe"].get("unit") or ""), n_show=len(pick),
                      title=f"{spec['name']}  ({split}, one trial per condition cell)")
    except Exception as exc:                                       # noqa: BLE001
        print(f"[analyse] movie not written: {type(exc).__name__}: {exc}")
    if "circuit_movie" in spec["training"]:
        # THE CIRCUIT AT WORK (Cedric, 2026-10-08): the connectivity matrix, the kinograph, the task's trace and, for
        # an angle, the eye -- results/movie_circuit.mp4. Printed, not raised, like the movie above.
        try:
            from plexus.tasks.plot_trainer import circuit_movie
            circuit_movie(spec, device=device, root=root)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[analyse] circuit movie not written: {type(exc).__name__}: {exc}")
    if poles is not None:
        res["poles"] = {"max_re_at_op": float(max(p.real for p in poles[0])),
                        "max_re_at_origin": float(max(p.real for p in poles[1])),
                        "truth_max_re": (max(pr_max) if pr_max else None),
                        "v_mean_abs": v_typ}
        json.dump(res, open(res_p, "w"), indent=2)
        print(f"[analyse] max Re(lambda) at the operating point {res['poles']['max_re_at_op']:+.4f} "
              f"1/s, at v=0 {res['poles']['max_re_at_origin']:+.4f}, truth "
              + (f"{max(pr_max):+.4f}" if pr_max else "n/a"))
    print(f"[analyse] wrote {p}")
    return p



# ============================================================================== shape tasks
# ONE WORLD, NO TRIALS: the model starts from its set-up, grows for `n_frames`, and its final
# state is scored against a shape. So there is no batch, no validation split and no horizon --
# the scheme is a list of resolution STAGES, coarse to fine, each with its own Adam and its own
# cosine from `lr` to `lr_min`. It works only because a lattice over MATERIAL coordinates means
# the same thing at 5,000 points as at 50,000: a cheap stage buys the parameters a dear one needs.
def _shape_target(spec, n, rng):
    """`n` points uniform inside the reference mesh, centred and scaled as the task says."""
    from plexus import shapes
    from plexus.morph import sample_inside
    ref = spec["task"]["reference"]
    path, _part = shapes.resolve(ref["shape"])
    if os.path.isdir(path):
        objs = sorted(glob.glob(os.path.join(path, "*.obj")))
        if not objs:
            raise FileNotFoundError(f"shape {ref['shape']!r}: {path} holds no .obj")
        path = objs[0]
    pts = sample_inside(path, n, np.asarray(ref["centre"], float), float(ref["size"]), rng)
    return pts, path


def _point_targets(spec, set_name, n, device):
    """Each particle's own target, for `point_mse`: the part `reference.points` names, put through
    EXACTLY the placement the model's `cloud_seed` gives the seeded part -- x = origin + scale q,
    every `every`-th point -- so particle i and target i are the same atom in the same frame.
    A contained (one copy per parent) or rotated cloud is refused: its placement is not built here."""
    from plexus import shapes
    raw = yaml.safe_load(open(spec["model"]))
    seeds = [o for o in (raw.get("seed") or []) + (raw.get("operators") or [])
             if o.get("op") == "cloud_seed" and o.get("at") == set_name]
    if len(seeds) != 1:
        raise ValueError(f"point_mse: {set_name!r} is not placed by exactly one `cloud_seed` "
                         f"({len(seeds)}), so its particles have no order to match targets against")
    cs = seeds[0]
    if (raw.get("sets", {}).get(set_name) or {}).get("parent") or cs.get("rotate"):
        raise ValueError("point_mse: a contained or rotated cloud is not supported yet")
    shape, part = str(spec["task"]["reference"]["points"]).split("/", 1)
    path = next((os.path.join(r, shape, "points.npz") for r in shapes.roots()
                 if os.path.exists(os.path.join(r, shape, "points.npz"))), None)
    if path is None:
        raise FileNotFoundError(f"point_mse: no shapes/{shape}/points.npz")
    with np.load(path) as z:
        q = np.asarray(z[part], np.float64)[:: max(1, int(cs.get("every", 1)))]
    if len(q) != n:
        raise ValueError(f"point_mse: {shape}/{part} gives {len(q)} targets for {n} particles")
    x = np.asarray(cs.get("origin") or [0.5, 0.5, 0.5], float) + float(cs.get("scale", 1.0)) * q
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _grid_mass(sim, X, grid):
    """Unit-weight nodal counts on the model's own `grid` field, at its current resolution."""
    from plexus.morph import mass_grid
    return mass_grid(X, torch.ones(X.shape[0], device=X.device, dtype=X.dtype),
                     float(sim.world_size[0]), int(sim.fields[grid]["n_grid"]), X.device, torch)


def _shape_rollout(sim, learn, set_name, device, grad, keep=None):
    """Run the model once; return (H, final positions, realised volume sum(p_vol det F))."""
    hook = None
    if keep is not None:
        def hook(H, frame):
            keep.append(H.level(set_name).get("pos").detach().cpu().numpy().copy())
    H, _ = engine.run(sim, device=device, progress=False, grad=grad, on_seeded=learn.inject,
                      on_frame=hook)
    q = H.level(set_name)
    vol = (q.p_vol * torch.linalg.det(q.F)).sum() if hasattr(q, "F") else torch.tensor(float("nan"))
    return H, q.get("pos"), vol


def _extent(X, um):
    e = (X.max(0).values - X.min(0).values).detach().cpu().numpy() * (um or 1.0)
    return [round(float(v), 2) for v in e]


def _train_shape(spec, device="cpu", root=None):
    tr, task = spec["training"], spec["task"]
    obs = task["observe"]
    seed = int(tr.get("seed", 0))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.backends.cuda.matmul.allow_tf32 = True          # measured free on morph: 0.015% of the loss
    torch.backends.cudnn.allow_tf32 = True
    out = out_dir(spec, root)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    shutil.copyfile(spec["_path"], os.path.join(out, "config.yaml"))
    shutil.copyfile(spec["model"], os.path.join(out, "model.yaml"))
    print(f"[run] {spec['name']} -> {out}")
    engine.quiet(True)
    learn = Learnables(spec["learnable"], device)
    lr = float(tr.get("lr", 1.0))
    lr_min = float(tr.get("lr_min", lr / 20.0))
    clip = tr.get("clip")
    guard = tr.get("guard", "restore_and_halve")
    hist, stages, t_all = [], [], time.time()
    for si, st in enumerate(tr["stages"]):
        sim = _model(spec, train=True, resolution=st.get("resolution"))
        um = getattr(sim.units, "length_um", None) if getattr(sim.units, "declared", False) else None
        n, iters = int(sim.sets[obs["set"]]["n"]), int(st["iters"])
        tgt_np, mesh = _shape_target(spec, n, rng)
        tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=device)
        T_pts = (_point_targets(spec, obs["set"], n, device)
                 if spec["task"]["reference"].get("points") else None)
        with torch.no_grad():
            m_ref = _grid_mass(sim, tgt, obs["grid"])
            if not learn.p:                            # the first rollout makes the parameters
                _shape_rollout(sim, learn, obs["set"], device, grad=False)
        opt = torch.optim.Adam(learn.groups(lr), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1), eta_min=lr_min)
        last_ok, vol0, t_st = learn.snapshot(), None, time.time()
        print(f"  stage {si}: {n:,} points, grid {sim.fields[obs['grid']]['n_grid']}, {iters} iterations, "
              f"target {os.path.basename(mesh)}")
        for it in range(iters):
            opt.zero_grad()
            _, X, vol = _shape_rollout(sim, learn, obs["set"], device, grad=True)
            vol0 = vol0 if vol0 is not None else float(vol.detach())
            parts = {}
            loss = _objective(spec, learn, {
                "log_mse": lambda red: _log_mse(_grid_mass(sim, X, obs["grid"]), m_ref),
                # THE REALISED VOLUME, sum(p_vol det F), against the stage's first rollout: the mass
                # loss alone is gameable by imploding onto a dense target (morph.py, --vol-weight).
                "volume": lambda red: (vol / vol0 - 1.0) ** 2,
                # EACH PARTICLE AGAINST ITS OWN TARGET, when the targets are known (exp04: the atoms
                # shared by two deposited structures). An outline loss cannot tell which atom went where.
                "point_mse": lambda red: ((X - T_pts) ** 2).sum(-1).mean()}, len(hist), parts)
            if not torch.isfinite(loss) and guard == "restore_and_halve":
                learn.restore(last_ok)
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                print(f"    iter {it:3d}  loss not finite -- restored, step sizes halved", flush=True)
                continue
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(learn.parameters(), float(clip))
            opt.step()
            sch.step()
            learn.clamp_()
            last_ok = learn.snapshot()
            hist.append({"stage": si, "iter": it, "loss": float(loss.detach()),
                         "volume_ratio": float(vol.detach()) / vol0, "terms": parts})
            if it % 20 == 0 or it == iters - 1:
                # VOLUME BESIDE THE EXTENT: a body can approach the target's box while the material
                # it is made of quietly disappears, and the extent alone would not say so.
                print(f"    iter {it:3d}  loss {float(loss):.6f}  vol {hist[-1]['volume_ratio']:.3f}x  "
                      f"shape {_extent(X, um)}  target {_extent(tgt, um)}"
                      + (" um" if um else ""), flush=True)
        stages.append({"stage": si, "n": n, "iters": iters, "final_loss": hist[-1]["loss"],
                       "seconds": round(time.time() - t_st, 1)})
        print(f"  stage {si}: {iters} iterations in {(time.time() - t_st) / 60:.1f} min", flush=True)
    torch.save({"fitted": learn.snapshot(), "model": spec["model"], "task": task,
                "learnable": spec["learnable"]}, os.path.join(out, "models", "best.pt"))
    rep = {"name": spec["name"], "model": spec["model"], "shape": task["reference"]["shape"],
           "final_loss": hist[-1]["loss"], "stages": stages, "history": hist,
           "n_params": sum(p.numel() for p in learn.parameters()),
           "seconds": round(time.time() - t_all, 1)}
    json.dump(rep, open(os.path.join(out, "results", "report.json"), "w"), indent=2)
    print(f"[done] final loss {rep['final_loss']:.6f} in {rep['seconds'] / 60:.1f} min")
    return out


def _last_resolution(spec):
    return spec["training"]["stages"][-1].get("resolution")


def _test_shape(spec, device="cpu", root=None):
    """The trained learnables at the last stage's resolution, scored once more without a tape."""
    engine.quiet(True)
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    obs = spec["task"]["observe"]
    sim = _model(spec, train=True, resolution=_last_resolution(spec))
    um = getattr(sim.units, "length_um", None) if getattr(sim.units, "declared", False) else None
    rng = np.random.default_rng(int(spec["training"].get("seed", 0)) + 1)   # a FRESH target draw
    tgt_np, mesh = _shape_target(spec, int(sim.sets[obs["set"]]["n"]), rng)
    tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, X0, vol0 = _shape_rollout(_model(spec, train=True, resolution=_last_resolution(spec), n_frames=0),
                                     learn, obs["set"], device, grad=False)
        _, X, vol = _shape_rollout(sim, learn, obs["set"], device, grad=False)
        loss = float(_log_mse(_grid_mass(sim, X, obs["grid"]), _grid_mass(sim, tgt, obs["grid"])))
    if spec["task"]["reference"].get("points"):
        Tp = _point_targets(spec, obs["set"], int(sim.sets[obs["set"]]["n"]), device)
        rms = float(((X - Tp) ** 2).sum(-1).mean().sqrt())
        rms0 = float(((X0 - Tp) ** 2).sum(-1).mean().sqrt())
        print(f"[test] point RMS to own target: start {rms0 * (um or 1):.4f} -> end {rms * (um or 1):.4f}"
              + (" um" if um else " (world)"))
    res = {"name": spec["name"], "shape": spec["task"]["reference"]["shape"], "mesh": mesh,
           "loss": loss, "shape_extent": _extent(X, um), "target_extent": _extent(tgt, um),
           "unit": "um" if um else None, "volume_ratio": float(vol) / float(vol0),
           **({"point_rms_start": rms0, "point_rms_end": rms} if spec["task"]["reference"].get("points") else {})}
    p = os.path.join(out, "results", f"{spec['name']}_test.json")
    json.dump(res, open(p, "w"), indent=2)
    print(f"[test] loss {loss:.6f} against a fresh draw of the target; shape {res['shape_extent']} vs "
          f"target {res['target_extent']} {res['unit'] or ''}; volume {res['volume_ratio']:.3f}x of the start")
    return res


def _analyse_shape(spec, device="cpu", root=None):
    """The trained growth, rendered as glass by `plexus.morph`'s own renderer.

    A DISPLAY ROLLOUT MAY RUN LONGER THAN THE TRAINED ONE: `render.frames` spreads the SAME total
    rest-shape change over more frames (every rate multiplied by trained_frames / render_frames),
    so the material grows more slowly for longer and the movie is smoother. It is not the trained
    trajectory sampled finer -- the mechanics get more time -- which is why it is said here.
    """
    from plexus.morph import render_movie
    engine.quiet(True)
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    obs = spec["task"]["observe"]
    rd = spec["training"].get("render") or {}
    base = _model(spec, train=False, resolution=_last_resolution(spec))
    rf = int(rd.get("frames", base.n_frames))
    sim = _model(spec, train=False, resolution=_last_resolution(spec), n_frames=rf)
    learn.scale = float(base.n_frames) / float(rf)
    frames = []
    with torch.no_grad():
        _shape_rollout(sim, learn, obs["set"], device, grad=False, keep=frames)
    rng = np.random.default_rng(int(spec["training"].get("seed", 0)) + 2)
    tgt_np, _mesh = _shape_target(spec, int(sim.sets[obs["set"]]["n"]), rng)
    d = os.path.join(out, "results")
    stats = render_movie(d, np.stack(frames), tgt_np, float(sim.world_size[0]), spec["name"],
                         px=int(rd.get("px", 1100)))
    np.savez_compressed(os.path.join(d, f"{spec['name']}_frames.npz"), frames=np.stack(frames),
                        target=tgt_np, scale=learn.scale)
    print(f"[analyse] {len(frames)} frames rendered into {d} ({stats.get('s_per_frame_total', float('nan')):.2f} s a frame)")
    return d


# ============================================================================== recording tasks
# A TISSUE AGAINST ITS OWN RECORDING, beat by beat. Each beat is its own rollout from its own rest,
# its own length, with its own edge band prescribed from what that beat did -- the tissue beyond
# the field of view enters there -- and the loss is the per-cell affine mismatch of POSITIONS
# (plexus.tasks.recording.cell_affine: A_j the cell's mean deformation, u_j its centroid
# displacement), each channel scaled by the recording's own spread. Gradients of all beats are
# summed before one step and the priors are added once, so memory is one beat's and the cost is
# one rollout per beat. This is prototype/cardio_mpm/strain/fit.py's scheme, run by the trainer.
def _affine_mse(A, u, A_ref, u_ref, w_A, w_u, cells=None):
    if cells is not None:
        A, u, A_ref, u_ref = A[:, cells], u[:, cells], A_ref[:, cells], u_ref[:, cells]
    return w_A * ((A - A_ref) ** 2).mean() + w_u * ((u - u_ref) ** 2).mean()


LOSSES["affine_mse"] = _affine_mse


def _r2(pred, ref):
    return float(1.0 - ((pred - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum())


def _recording_setup(spec, device):
    """The recording, its beat windows with their affine maps, and the model's rest layout: the
    material points' positions X0 after seeding, their cell index, the edge band and the interior
    cells (none of whose points lie in the band) -- which the loss is restricted to, because a
    cell with points in the band has part of its motion dictated rather than modelled."""
    from plexus.tasks import recording as R
    task = spec["task"]
    ref, obs, drv = task["reference"], task["observe"], task["drive"]
    rec = R.load(ref["recording"], device=device)
    wins = [R.beat_window(rec, int(b)) for b in ref["beats"]]
    for w in wins:
        if (w["onset"] - w["span"][0]) - R.PRE != 0:
            raise ValueError(f"beat {w['k']} opens {w['onset'] - w['span'][0]} frames before its "
                             f"onset, not {R.PRE}: a truncated window shifts the clock, which this "
                             f"trainer does not re-align")
    refs = [R.window_affine(rec, w) for w in wins]
    box = {}
    rest = _model(spec, train=False, n_frames=0)
    with torch.no_grad():
        H, _ = engine.run(rest, device=device, progress=False)
    q = H.level(drv["set"])
    X0 = q.get("pos").detach().clone()
    cid = H.lift_index(drv["set"], obs["of"]).long() + 1
    C = rec["n_cells"]
    if int(cid.max()) > C:
        raise ValueError(f"the model has {int(cid.max())} cells, the recording {C}")
    wd = float(drv["width"])
    band = ((X0[:, 0] < R.DOM_LO + wd) | (X0[:, 0] > R.DOM_HI - wd)
            | (X0[:, 1] < R.DOM_LO + wd) | (X0[:, 1] > R.DOM_HI - wd))
    interior = R.interior_cells(cid, band, C) if task.get("mask", "interior") == "interior" else None
    pres = [R.band_prescription(A, u, X0, cid, band) for A, u in refs]
    box.update(rec=rec, wins=wins, refs=refs, X0=X0, cid=cid, band=band, interior=interior,
               pres=pres, C=C)
    return box


def _recording_rollout(sim, learn, spec, box, bi, device, grad, watch=None):
    """One beat: prescribe the band from the recording each frame, record every cell's affine map.
    `watch(H, tick)` is called at the same moment, for a caller that needs more of the state."""
    from plexus.tasks import recording as R
    set_name = spec["task"]["drive"]["set"]
    ub, band, C = box["pres"][bi], box["band"], box["C"]
    As, us, st = [], [], {}

    def hook(H, tick):
        q = H.level(set_name)
        if tick == 0:
            st["X0"] = q.get("pos").detach().clone()
        p0, p1 = q.state_schema["pos"]
        v0, v1 = q.state_schema["vel"]
        S0 = q.state
        S = S0.clone()
        m = band[:, None]
        S[:, p0:p1] = torch.where(m, st["X0"] + ub[min(tick, ub.shape[0] - 1)], S0[:, p0:p1])
        S[:, v0:v1] = torch.where(m, torch.zeros_like(S0[:, v0:v1]), S0[:, v0:v1])
        q.state = S
        A, u = R.cell_affine(q.get("pos"), st["X0"], box["cid"], C)
        As.append(A)
        us.append(u)
        if watch is not None:
            watch(H, tick)

    engine.run(sim, device=device, progress=False, grad=grad, on_frame=hook,
               on_seeded=learn.inject, on_ready=learn.ready)
    return torch.stack(As), torch.stack(us)


def _train_recording(spec, device="cpu", root=None):
    tr, task = spec["training"], spec["task"]
    torch.manual_seed(int(tr.get("seed", 0)))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = out_dir(spec, root)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    shutil.copyfile(spec["_path"], os.path.join(out, "config.yaml"))
    shutil.copyfile(spec["model"], os.path.join(out, "model.yaml"))
    print(f"[run] {spec['name']} -> {out}")
    engine.quiet(True)
    box = _recording_setup(spec, device)
    eye = torch.eye(2, device=device)
    A0, u0 = box["refs"][0]
    w_A = 1.0 / ((A0 - eye) ** 2).mean()                      # the first beat's spread, as fit.py
    w_u = float(task.get("u_weight", 1.0)) / (u0 ** 2).mean()
    sims = [_model(spec, train=True, n_frames=len(w["frames"]) - 1) for w in box["wins"]]
    n_int = int(box["interior"].sum()) if box["interior"] is not None else box["C"]
    print(f"[data] {task['reference']['recording']}: beats {task['reference']['beats']} "
          f"({', '.join(str(len(w['frames'])) for w in box['wins'])} frames), {box['X0'].shape[0]:,} "
          f"points, {int(box['band'].sum()):,} in the band, {n_int} of {box['C']} cells in the loss")
    learn = Learnables(spec["learnable"], device)
    with torch.no_grad():                                     # the first rollout makes the parameters
        _recording_rollout(_model(spec, train=True, n_frames=1), learn, spec, box, 0, device, False)
    params = learn.parameters()
    print(f"[fit] {len(params)} tensor(s), {sum(p.numel() for p in params)} values: " + ", ".join(learn.p))
    iters = int(tr["iters"])
    lr = float(tr.get("lr", 1e-2))
    frac = float(tr.get("lr_min_frac", 0.05))
    opt = torch.optim.Adam(learn.groups(lr), lr=lr)
    # COSINE FROM 1x TO `lr_min_frac` OF EACH GROUP'S OWN STEP SIZE, so twelve families whose
    # natural steps differ fifty-fold all anneal in proportion.
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: frac + (1 - frac) * 0.5 * (1 + math.cos(math.pi * min(i, iters) / max(iters, 1))))
    guard = tr.get("guard", "restore_and_halve")
    save_every = int(tr.get("save_every", 25))
    last_ok, log, t_all = learn.snapshot(), [], time.time()
    for it in range(iters):
        t0 = time.time()
        opt.zero_grad()
        loss_v, finite, terms = 0.0, True, {}
        for bi, sim in enumerate(sims):
            A_b, u_b = box["refs"][bi]
            A, u = _recording_rollout(sim, learn, spec, box, bi, device, True)
            c, nb = box["interior"], len(sims)
            Ac, uc, Abc, ubc = ((A, u, A_b, u_b) if c is None else (A[:, c], u[:, c], A_b[:, c], u_b[:, c]))
            parts = {}
            # EVERY BEAT'S EVIDENCE, THE PRIORS ONCE, on the last beat's graph -- fit.py's order.
            loss_b = _objective(spec, learn, {
                "affine_mse": lambda red: _affine_mse(A, u, A_b, u_b, w_A, w_u, c) / nb,
                "affine_A": lambda red: w_A * ((Ac - Abc) ** 2).mean() / nb,
                "affine_u": lambda red: ((uc - ubc) ** 2).mean() / (u0 ** 2).mean() / nb},
                it, parts, prior=(bi == nb - 1))
            for kk, vv in parts.items():
                terms[kk] = terms.get(kk, 0.0) + vv
            if torch.isfinite(loss_b):
                loss_b.backward()
            else:
                finite = False
            loss_v += float(loss_b.detach())
        if not finite and guard == "restore_and_halve":
            learn.restore(last_ok)
            for g in opt.param_groups:
                g["lr"] *= 0.5
            sch.step()
            print(f"  it {it:4d} loss not finite -- restored the last finite values, step sizes halved", flush=True)
            continue
        opt.step()
        sch.step()
        learn.clamp_()
        last_ok = learn.snapshot()
        log.append({"it": it, "loss": loss_v, "seconds": time.time() - t0, "terms": terms})
        if it % 5 == 0 or it == iters - 1:
            print(f"  it {it:4d} loss {loss_v:.5f}  {log[-1]['seconds']:.1f} s", flush=True)
        if (it + 1) % save_every == 0 or it == iters - 1:
            torch.save({"fitted": learn.snapshot(), "model": spec["model"], "task": task,
                        "learnable": spec["learnable"]}, os.path.join(out, "models", "best.pt"))
    rep = {"name": spec["name"], "model": spec["model"], "recording": task["reference"]["recording"],
           "beats": task["reference"]["beats"], "final_loss": log[-1]["loss"], "history": log,
           "n_params": sum(p.numel() for p in params), "cells_in_loss": n_int,
           "seconds": round(time.time() - t_all, 1),
           "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 2 ** 30
                           if str(device).startswith("cuda") else None)}
    json.dump(rep, open(os.path.join(out, "results", "report.json"), "w"), indent=2)
    print(f"[done] final loss {rep['final_loss']:.5f} in {rep['seconds'] / 60:.1f} min")
    return out


def _test_recording(spec, device="cpu", root=None):
    """Every fitted beat, rolled out again without a tape and scored as the prototype scored it:
    R^2 of A - I over (frames, cells, 4 components) and of u over (frames, cells, 2), interior cells."""
    engine.quiet(True)
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = _recording_setup(spec, device)
    eye = torch.eye(2, device=device)
    cells = box["interior"] if box["interior"] is not None else slice(None)
    res = {"name": spec["name"], "beats": {}}
    for bi, w in enumerate(box["wins"]):
        sim = _model(spec, train=False, n_frames=len(w["frames"]) - 1)
        with torch.no_grad():
            A, u = _recording_rollout(sim, learn, spec, box, bi, device, False)
        A_ref, u_ref = box["refs"][bi]
        r = {"window": list(w["span"]), "r2_A": _r2(A[:, cells] - eye, A_ref[:, cells] - eye),
             "r2_u": _r2(u[:, cells], u_ref[:, cells])}
        res["beats"][str(w["k"])] = r
        print(f"[test] beat {w['k']} window {tuple(w['span'])}: R2(A) {r['r2_A']:.4f}  R2(u) {r['r2_u']:.4f}")
    json.dump(res, open(os.path.join(out, "results", f"{spec['name']}_test.json"), "w"), indent=2)
    return res

# ============================================================================== field recordings
# A RECORDED FIELD (`task.reference.field_recording`): a sequence of volumes with a tissue mask and
# no cells or tracks, as `plexus.tasks.field_recording` freezes it. The model is a field whose own
# law is learned; a rollout starts from the recorded volume at an origin (`on_seeded`) and is
# compared with the recorded volumes after it, on the voxels that are tissue at both ends. The
# scorer is `field_recording.score`, the same arithmetic the baselines were measured with, so a
# model's RMSE and persistence's are one number apart and nothing else.
def _field_setup(spec, device):
    """The recording at the model's grid, its train / val / test targets, and the check that the
    model's field is that grid. The last `n_test` volumes are test targets, the `n_val` before them
    validation targets (they pick the checkpoint), and the rest train targets."""
    from plexus.tasks import field_recording as FR
    ref, fname = spec["task"]["reference"], spec["task"]["observe"]["field"]
    rec = FR.coarsen(FR.load(ref["field_recording"]), int(ref.get("coarsen", 1)))
    if ref.get("split", "holdout") == "all":
        # EVERY VOLUME IS TRAINED ON (Cedric, 2026-09-29): the question is what the law needs to REPRODUCE
        # the recording, scored by a full rollout from t0 (`_test_field_full`). The last `n_val` volumes
        # are still rolled out to watch the fit, in sample; nothing is held out.
        tr, te = np.arange(rec["ratio"].shape[0]), np.arange(0)
        va = tr[-int(ref.get("n_val", 6)):]
    else:
        tr, te = FR.split(rec, int(ref.get("n_test", 15)))
        n_val = int(ref.get("n_val", 6))
        va, tr = tr[-n_val:], tr[:-n_val]
    sim = _model(spec, train=False, n_frames=0)
    with torch.no_grad():
        H, _ = engine.run(sim, device=device, progress=False)
    shape = tuple(H.fields[fname].grid.shape[1:])
    if shape != tuple(rec["ratio"].shape[1:]):
        raise ValueError(f"the model's field `{fname}` is {shape}, the recording at coarsen "
                         f"{ref.get('coarsen', 1)} is {tuple(rec['ratio'].shape[1:])}")
    X = torch.as_tensor(rec["ratio"], device=device)[:, None]          # [T, 1, Z, Y, X]
    M = torch.as_tensor(rec["mask"], device=device)                     # [T, Z, Y, X]
    # A FIELD OF k CHANNELS HOLDS THE k LAST VOLUMES, newest first (`diffuse[graphcast] inputs: k`), so a
    # rollout from origin o is seeded with x(o), x(o - 1), ...; an origin needs k - 1 volumes behind it.
    n_in = int(H.fields[fname].grid.shape[0])
    U = M.any(0)                                  # every voxel that is tissue in ANY frame: the global mask
    # GRAPHCAST'S NORMALISATION, when asked (`reference.normalise: true`): the tissue's mean and SD over the training
    # frames, and the SD of the one-step difference on voxels that are tissue at both ends -- handed to the law at
    # `on_ready` (`_field_rollout`), as exp17's trace trainer hands `state_diffuse[graphcast]` its own.
    norm = (0.0, 1.0, 1.0)
    if ref.get("normalise"):
        xt, mt = rec["ratio"][tr], rec["mask"][tr]
        d = [(rec["ratio"][t + 1] - rec["ratio"][t])[rec["mask"][t] & rec["mask"][t + 1]]
             for t in tr if t + 1 < rec["ratio"].shape[0]]
        norm = (float(xt[mt].mean()), float(xt[mt].std()), float(np.sqrt(np.mean(np.concatenate(d) ** 2))))
    # THE ALIVE FRACTION (`observe.alive`, exp16 2026-10-01): a death operator's field, seeded with the recorded tissue
    # mask at each rollout's origin and scored against the mask of every target frame (`mask_iou`, `mask_bce`).
    alive = spec["task"]["observe"].get("alive")
    return dict(rec=rec, X=X, M=M, U=U, train=tr, val=va, test=te, field=fname, n_in=n_in, norm=norm, alive=alive)


def _field_rollout(sims, learn, box, origin, h, device, grad, messages=None, clamp=None):
    """h recorded intervals from volume `origin`: [h, C, *grid], frame k the forecast of origin+k+1.
    `sims[h]` is the model built for that horizon (n_frames = h - 1: one step per tick, and
    `on_frame` fires after each). The `task.rollouts` hooks (exp19, 2026-10-03), both off by default:
    `messages` True / False set on every operator that has the switch (`diffuse[graphcast]`); `clamp`, a bool
    tensor over the grid, writes the RECORDED volume origin+k+1 into those voxels' newest channel after tick k,
    before the law reads the next step -- the only way anything recorded enters after the origin, by request."""
    fname, frames = box["field"], []
    if origin - box["n_in"] + 1 < 0:
        raise ValueError(f"origin {origin} has fewer than {box['n_in'] - 1} volumes behind it")
    x0 = torch.cat([box["X"][origin - k] for k in range(box["n_in"])], 0)

    alive = box.get("alive")

    def seeded(H):
        H.fields[fname].grid = x0.clone()
        if alive:                                 # the tissue at the origin is alive, the rest is not
            H.fields[alive].grid = box["M"][origin].to(x0.dtype)[None].clone()
        learn.inject(H)                           # the trainer's field learnables (an embedding)

    def hook(H, tick):
        # WITH A DEATH OPERATOR THE ALIVE FRACTION RIDES AS THE LAST CHANNEL: pred[:, 0] stays the ratio everywhere
        # it is read, pred[:, -1] is A (`_mask_iou`, `_mask_bce`, `_test_field_full`).
        g = H.fields[fname].grid
        if clamp is not None and origin + len(frames) + 1 < len(box["X"]):
            g = g.clone()
            g[0][clamp] = box["X"][origin + len(frames) + 1][0][clamp]
            H.fields[fname].grid = g
        frames.append(torch.cat([g, H.fields[alive].grid], 0) if alive else g)

    def ready(H):
        learn.ready(H)
        for o in H.operators:                     # a law that reads the global forcing I(t) needs t
            if hasattr(o, "t_origin"):
                o.t_origin = origin
                o.norm = box.get("norm", (0.0, 1.0, 1.0))
            if messages is not None and hasattr(o, "messages"):
                o.messages = bool(messages)

    engine.run(sims[h], device=device, progress=False, grad=grad, on_frame=hook,
               on_seeded=seeded, on_ready=ready)
    return torch.stack(frames[:h])


class _OwnState:
    """Stands in for the recording at the seed of a later segment of a free rollout: index origin - k returns the
    MODEL's own state k volumes back (channel k of its last field) -- `_field_rollout` reads only X[origin - k]."""

    def __init__(self, origin, grid, X):
        self.origin, self.grid, self.X = origin, grid, X

    def __getitem__(self, i):
        k = self.origin - i
        return self.grid[k:k + 1] if 0 <= k < self.grid.shape[0] else self.X[i]

    def __len__(self):
        return len(self.X)


def _field_free(spec, learn, box, device, variant=None, seg=200):
    """The free rollout from volume `inputs - 1` to the last, in segments of `seg` volumes moved to the CPU as they
    are made (the [n, C, *grid] stack of a 1,594-volume whole-body rollout is 19 GB), each later segment seeded from
    the model's OWN last field; equal to one `_field_rollout` (tests/test_exp19_no_leak.py). `variant`, one entry of
    `task.rollouts`: its `zero:` learnables at 0, its `messages:`, its `clamp:` voxels given their recorded values.
    Returns (pred [n, *grid] numpy, origin, n, clamped [*grid] bool numpy)."""
    v = variant or {}
    o = box["n_in"] - 1
    n = box["rec"]["ratio"].shape[0] - 1 - o
    msg = None if "messages" not in v else v["messages"] == "on"
    cl = None
    if "clamp" in v:
        rec = box["rec"]
        Z, Y, X = rec["ratio"].shape[1:]
        zz, yy, xx = np.meshgrid((np.arange(Z) + 0.5) * rec["dz_um"], (np.arange(Y) + 0.5) * rec["dx_um"],
                                 (np.arange(X) + 0.5) * rec["dx_um"], indexing="ij")
        pos = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)          # x, y, z: a 2-axis box spans every z
        cl = torch.as_tensor(_roi_mask(v["clamp"]["rois"], pos).reshape(Z, Y, X), device=device)
    out, cur, done, b = [], o, 0, dict(box)
    with _zeroed(learn, spec, v.get("zero", [])), torch.no_grad():
        while done < n:
            h = min(seg, n - done)
            roll = _field_rollout(_field_sims(spec, [h], train=False), learn, b, cur, h, device, False,
                                  messages=msg, clamp=cl)
            out.append(roll[:, 0].cpu().numpy())
            b = dict(box, X=_OwnState(cur + h, roll[-1], box["X"]))
            cur, done = cur + h, done + h
            del roll
    clm = cl.cpu().numpy() if cl is not None else np.zeros(box["rec"]["ratio"].shape[1:], bool)
    return np.concatenate(out, 0), o, n, clm


def _field_scores(free, rec, recd, o, n, clamped):
    """A field free rollout read as `_test_field_full` reads it, on the voxels on the mask at the origin and the frame
    and NOT clamped, plus exp17's brain-mean metrics on the field's mean trace (raw recording, unweighted voxel mean,
    every free frame: `_brain_mean_metrics`, its SDs (ddof 0) and correlation)."""
    obs, obs_d = rec["ratio"][o + 1:o + 1 + n], recd["ratio"][o + 1:o + 1 + n]
    m = rec["mask"][o + 1:o + 1 + n] & rec["mask"][o] & ~clamped
    r2, r2s = _r2_parts(free, obs, m)
    r2d, r2ds = _r2_parts(free, obs_d, m)
    fm = np.array([free[k][m[k]].mean() for k in range(n)])
    om = np.array([obs[k][m[k]].mean() for k in range(n)])
    return {"r2": r2, "r2_spatial": r2s, "r2_denoised": r2d, "r2_spatial_denoised": r2ds,
            **_brain_mean_metrics(om, fm), "brain_mean_sd_obs": float(om.std()), "brain_mean_sd_pred": float(fm.std()),
            "brain_mean_r": float(np.corrcoef(fm, om)[0, 1]) if fm.std() > 0 and om.std() > 0 else float("nan"),
            "n_clamped": int(clamped.sum()), "finite": bool(np.isfinite(free).all())}, fm, om


def _field_variants(spec, learn, box, device, out, stem, names=None):
    """Every `task.rollouts` variant but the nominal one, for a field recording: {name: its scores}; each writes
    results/<stem>_<name>_brain_mean.npz (the learned and recorded mean traces, the per-voxel rollout is not kept:
    2.5 GB a variant at exp19's grid)."""
    from plexus.tasks import field_recording as FR
    res, recd = {}, None
    for v in spec["task"].get("rollouts") or []:
        if v["name"] == "nominal" or (names and v["name"] not in names):
            continue
        t1 = time.time()
        recd = recd if recd is not None else FR.denoise(box["rec"])
        free, o, n, clm = _field_free(spec, learn, box, device, variant=v)
        sc, fm, om = _field_scores(free, box["rec"], recd, o, n, clm)
        np.savez_compressed(os.path.join(out, "results", f"{stem}_{v['name']}_brain_mean.npz"), pred=fm, obs=om)
        res[v["name"]] = {**sc, "spec": v}
        print(f"[test] rollout `{v['name']}` ({time.time() - t1:.0f} s): brain-mean R2 {sc['brain_mean_r2']:+.3f} "
              f"(RMSE {sc['brain_mean_rmse']:.4f}, SD {sc['brain_mean_sd_pred']:.4f} against {sc['brain_mean_sd_obs']:.4f} "
              f"recorded, r {sc['brain_mean_r']:+.2f}); R2 denoised {sc['r2_denoised']:+.3f} ({sc['n_clamped']:,} voxels given)")
        del free
    return res


def field_rollouts(name, device="cuda:0", root=None, names=None):
    """A landed field run's `task.rollouts` variants made from its best model (as `trace_rollouts`), the test json's
    `rollouts` updated."""
    engine.quiet(True)
    spec = load(name)
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = _field_setup(spec, device)
    res = _field_variants(spec, learn, box, device, out, spec["name"], names)
    tj = os.path.join(out, "results", f"{spec['name']}_test.json")
    if os.path.exists(tj):
        j = json.load(open(tj))
        j["rollouts"] = {**j.get("rollouts", {}), **res}
        json.dump(j, open(tj, "w"), indent=2)
    return res


def _field_sims(spec, hs, train):
    return {h: _model(spec, train=train, n_frames=h - 1) for h in sorted(set(hs))}


def _field_forecaster(sims, learn, box, device):
    """`forecast(origin, h) -> [Z, Y, X]` numpy, for `field_recording.score`; one rollout each."""
    def f(o, h):
        with torch.no_grad():
            return _field_rollout(sims, learn, box, o, h, device, False)[h - 1, 0].cpu().numpy()
    return f


def _masked_mse(pred, box, origin, h, reduction="mean"):
    """Mean over the h forecast steps of the reduced residual on voxels that are tissue at the origin
    and at that step's target -- the scorer's voxel set, so with `mean` the loss is the scorer's MSE."""
    tot = 0.0
    for k in range(h):
        v = box["M"][origin] & box["M"][origin + k + 1]
        tgt = box["X"][origin + k + 1, 0][v]
        tot = tot + _reduce(pred[k, 0][v] - tgt, reduction, tgt)
    return tot / h


LOSSES["masked_mse"] = _masked_mse


def _global_mse(pred, box, origin, h, reduction="mean"):
    """The rollout against the recording on the GLOBAL mask -- every voxel that is tissue in any frame --
    with the recording's own 0 where there is no tissue at that frame (Cedric, 2026-09-29). With it the law
    must drive a voxel to 0 where the organoid retreats and fill it where it grows; `masked_mse` alone scores
    only voxels that are tissue at both ends, so the law never learned a boundary (finding 16)."""
    U = box["U"]
    tot = 0.0
    for k in range(h):
        tgt = box["X"][origin + k + 1, 0][U]
        tot = tot + _reduce(pred[k, 0][U] - tgt, reduction, tgt)
    return tot / h


LOSSES["global_mse"] = _global_mse


def _mask_iou(pred, box, origin, h, reduction="mean"):
    """1 - the SOFT IoU of the alive fraction A (pred[:, -1]) with the recorded tissue mask M of each target frame,
    sum A M / sum (A + M - A M), averaged over the h steps (exp16, Cedric 2026-10-01: "train with the IoU mask"). Too
    much tissue and too little cost alike, and its gradient does not vanish where A is 0 or 1. `reduction` is unused."""
    tot = 0.0
    for k in range(h):
        A, M = pred[k, -1], box["M"][origin + k + 1].to(pred.dtype)
        inter = (A * M).sum()
        tot = tot + 1.0 - inter / (A.sum() + M.sum() - inter).clamp(min=1.0)
    return tot / h


def _mask_bce(pred, box, origin, h, reduction="mean"):
    """The binary cross-entropy of A against the recorded mask, on the GLOBAL mask's voxels (tissue in any frame):
    the per-voxel rival of `mask_iou` (exp16 arm 6). `reduction` is unused."""
    U = box["U"]
    tot = 0.0
    for k in range(h):
        A, M = pred[k, -1][U].clamp(1e-6, 1 - 1e-6), box["M"][origin + k + 1][U].to(pred.dtype)
        tot = tot + torch.nn.functional.binary_cross_entropy(A, M)
    return tot / h


LOSSES["mask_iou"] = _mask_iou
LOSSES["mask_bce"] = _mask_bce


def _train_field(spec, device="cpu", root=None):
    from plexus.tasks import field_recording as FR
    tr = spec["training"]
    torch.manual_seed(int(tr.get("seed", 0)))
    rng = np.random.default_rng(int(tr.get("seed", 0)))
    out = out_dir(spec, root)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    shutil.copyfile(spec["_path"], os.path.join(out, "config.yaml"))
    shutil.copyfile(spec["model"], os.path.join(out, "model.yaml"))
    print(f"[run] {spec['name']} -> {out}")
    engine.quiet(True)
    box = _field_setup(spec, device)
    hs = [int(h) for h in tr["horizon"]]
    sims = _field_sims(spec, hs, train=True)
    evals = _field_sims(spec, hs, train=False)
    tr_t, va_t = box["train"], box["val"]
    print(f"[data] {spec['task']['reference']['field_recording']}: grid {tuple(box['X'].shape[2:])}, "
          f"{len(tr_t)} train / {len(va_t)} val / {len(box['test'])} test target volumes "
          f"(val {va_t[0] + 1}-{va_t[-1] + 1}, "
          + (f"test {box['test'][0] + 1}-{box['test'][-1] + 1}" if len(box["test"]) else "none held out: val is in sample")
          + ", 1-based)")
    learn = Learnables(spec["learnable"], device)
    with torch.no_grad():                                     # the first rollout makes the parameters
        _field_rollout(sims, learn, box, int(tr_t[box["n_in"] - 1]), hs[0], device, False)
    params = learn.parameters()
    n_par = sum(p.numel() for p in params)
    print(f"[fit] {len(params)} tensor(s), {n_par} values: " + ", ".join(learn.p))
    base = {h: FR.score(FR.persistence(box["rec"]), box["rec"], va_t, h)["rmse"] for h in sorted(set(hs))}
    print("[val] persistence RMSE " + "  ".join(f"h={h}: {v:.5f}" for h, v in base.items()))
    # THE BAR IS THE BEST NON-LEARNED BASELINE, not persistence: at this grid the voxel jitter is most
    # of persistence's error, so a law that only blurs beats it (exp16, finding 3). The blur's width
    # is chosen on the TRAIN targets, as `tools/redox_baselines.py` chooses it.
    blur = {}
    for h in sorted(set(hs)):
        sig = min((0.5, 1.0, 2.0, 4.0), key=lambda g: FR.score(FR.smoothed(box["rec"], g), box["rec"], tr_t[12:], h)["rmse"])
        blur[h] = FR.score(FR.smoothed(box["rec"], sig), box["rec"], va_t, h)["rmse"]
    print("[val] best blur RMSE   " + "  ".join(f"h={h}: {v:.5f}" for h, v in blur.items()))
    iters = int(tr["iters"])
    batch = max(1, int(tr.get("batch", 4)))
    lr = float(tr.get("lr", 1e-3))
    frac = float(tr.get("lr_min_frac", 0.05))
    clip = float(tr.get("clip", 1.0))
    opt = torch.optim.Adam(learn.groups(lr), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: frac + (1 - frac) * 0.5 * (1 + math.cos(math.pi * min(i, iters) / max(iters, 1))))
    guard = tr.get("guard", "restore_and_halve")
    save_every = int(tr.get("save_every", 50))
    hist_p = os.path.join(out, "results", "history.jsonl")
    open(hist_p, "w").close()

    def val_row(it):
        f = _field_forecaster(evals, learn, box, device)
        r = {"it": it}
        for h in sorted(set(hs)):
            r[f"val_rmse_h{h}"] = FR.score(f, box["rec"], va_t, h)["rmse"]
            r[f"val_persistence_h{h}"] = base[h]
            r[f"val_blur_h{h}"] = blur[h]
        return r

    best, best_state, last_ok, log, t_all = np.inf, learn.snapshot(), learn.snapshot(), [], time.time()
    v0 = val_row(0)
    best = sum(v0[f"val_rmse_h{h}"] ** 2 for h in set(hs))
    _field_snapshot(out, spec["name"], learn, evals, box, device, 0, [v0], base, hs, blur)
    with open(hist_p, "a") as fh:
        fh.write(json.dumps(v0) + "\n")
    for it in range(iters):
        # THE CURRICULUM: the horizons in the order given, an equal share of the iterations each.
        h = hs[min(len(hs) - 1, it * len(hs) // max(iters, 1))]
        ok_origins = [int(o) for o in tr_t if int(o) + h <= int(tr_t[-1]) and int(o) >= box["n_in"] - 1]
        t0 = time.time()
        opt.zero_grad()
        loss_v, finite, parts = 0.0, True, {}
        for o in rng.choice(ok_origins, size=min(batch, len(ok_origins)), replace=False):
            pred = _field_rollout(sims, learn, box, int(o), h, device, True)
            # THE ONE OBJECTIVE (`_objective`): the task's weighted terms and the learnables' priors,
            # annealed by the iteration. Per origin, divided by the batch, so the priors enter once.
            p_o = {}
            loss_b = _objective(spec, learn, {
                "masked_mse": lambda red, pred=pred, o=int(o): _masked_mse(pred, box, o, h, red),
                "global_mse": lambda red, pred=pred, o=int(o): _global_mse(pred, box, o, h, red),
                "mask_iou": lambda red, pred=pred, o=int(o): _mask_iou(pred, box, o, h, red),
                "mask_bce": lambda red, pred=pred, o=int(o): _mask_bce(pred, box, o, h, red)}, it, p_o) / batch
            for k, v in p_o.items():
                parts[k] = parts.get(k, 0.0) + v / batch
            if torch.isfinite(loss_b):
                loss_b.backward()
            else:
                finite = False
            loss_v += float(loss_b.detach())
        if not finite and guard == "restore_and_halve":
            learn.restore(last_ok)
            for g in opt.param_groups:
                g["lr"] *= 0.5
            sch.step()
            print(f"  it {it:5d} loss not finite -- restored the last finite values, step sizes halved", flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sch.step()
        learn.clamp_()
        last_ok = learn.snapshot()
        ev = parts.get("loss.masked_mse", loss_v)
        row = {"it": it + 1, "horizon": h, "loss": loss_v, "rmse": math.sqrt(max(ev, 0.0)),
               "seconds": time.time() - t0, **parts}
        log.append(row)
        if (it + 1) % save_every == 0 or it == iters - 1:
            row.update(val_row(it + 1))
            score = sum(row[f"val_rmse_h{k}"] ** 2 for k in set(hs))
            if score < best:
                best, best_state = score, learn.snapshot()
            _field_snapshot(out, spec["name"], learn, evals, box, device, it + 1, log, base, hs, blur)
            print(f"  it {it + 1:5d} h {h}  loss {loss_v:.6f} (rmse {row['rmse']:.5f})  val (model/persistence/blur) "
                  + "  ".join(f"h{k} {row[f'val_rmse_h{k}']:.5f}/{base[k]:.5f}/{blur[k]:.5f}" for k in sorted(set(hs)))
                  + f"  {row['seconds']:.2f} s/it", flush=True)
        with open(hist_p, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    torch.save({"fitted": best_state, "model": spec["model"], "task": spec["task"],
                "learnable": spec["learnable"]}, os.path.join(out, "models", "best.pt"))
    rep = {"name": spec["name"], "model": spec["model"],
           "field_recording": spec["task"]["reference"]["field_recording"], "horizons": hs,
           "n_params": n_par, "iters": iters, "batch": batch, "val_persistence_rmse": base,
           "best_val_sum_mse": best, "history": log, "seconds": round(time.time() - t_all, 1),
           "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 2 ** 30
                           if str(device).startswith("cuda") else None)}
    json.dump(rep, open(os.path.join(out, "results", "report.json"), "w"), indent=2)
    print(f"[done] {iters} iterations in {rep['seconds'] / 60:.1f} min, best val sum of MSE {best:.6f}")
    return out


def _field_snapshot(out, name, learn, evals, box, device, it, log, base, hs, blur=None):
    """`results/live.png`, rewritten at every validation, so a watcher sees the run while it trains:
    the loss and validation curves so far, and the mid plane of the last validation volume --
    recorded, forecast from the volume `max(hs)` intervals before it, and their difference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    va, h = box["val"], max(hs)
    o, t = int(va[-1]) - h, int(va[-1])
    with torch.no_grad():
        pred = _field_rollout(evals, learn, box, o, h, device, False)[h - 1, 0].cpu().numpy()
    obs, msk = box["rec"]["ratio"][t], box["rec"]["mask"][t]
    zm = obs.shape[0] // 2
    lo, hi = np.percentile(obs[msk], [2, 98])
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax[0].spines[["top", "right"]].set_visible(False)
    tr_rows = [r for r in log if "loss" in r]
    if tr_rows:
        ax[0].semilogy([r["it"] for r in tr_rows], [r["rmse"] for r in tr_rows], color="0.6", lw=0.8,
                       label="train RMSE (batch)")
    for k, c in zip(sorted(set(hs)), ("tab:red", "tab:blue", "tab:orange", "tab:purple")):
        vr = [r for r in log if f"val_rmse_h{k}" in r]
        ax[0].semilogy([r["it"] for r in vr], [r[f"val_rmse_h{k}"] for r in vr], "o-", ms=3, color=c,
                       label=f"val RMSE, {k} step(s)")
        ax[0].axhline(base[k], color=c, ls="--", lw=0.8)
        if blur:
            ax[0].axhline(blur[k], color=c, ls=":", lw=1.2)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("RMSE of the ratio (dashed: persistence, dotted: best blur)")
    ax[0].legend(frameon=False, fontsize=8)
    for a, img, lab, cm, v in ((ax[1], obs[zm], f"recorded, volume {t + 1}", "viridis", (lo, hi)),
                               (ax[2], pred[zm], f"forecast, {h} steps from {o + 1}", "viridis", (lo, hi)),
                               (ax[3], pred[zm] - obs[zm], "forecast - recorded", "RdBu_r",
                                (-(hi - lo) / 2, (hi - lo) / 2))):
        a.imshow(np.where(msk[zm], img, np.nan), cmap=cm, vmin=v[0], vmax=v[1])
        a.set_axis_off()
        a.set_title(f"{lab}, plane {zm}", fontsize=10, loc="left")
    fig.suptitle(f"{name}  iteration {it}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "results", "live.png"), dpi=90, facecolor="white")
    plt.close(fig)


def _r2_parts(pred, obs, m):
    """Variance explained on the voxels m, over (t, v): TOTAL, and SPATIAL -- each frame's tissue mean
    removed from both first, so a law that only follows the organoid-wide level scores 0 there."""
    x, p = obs[m], pred[m]
    r2 = 1.0 - float(((p - x) ** 2).sum()) / float(((x - x.mean()) ** 2).sum())
    num = den = 0.0
    for t in range(obs.shape[0]):
        mt = m[t]
        if mt.sum() < 2:
            continue
        xt, pt = obs[t][mt], pred[t][mt]
        xt, pt = xt - xt.mean(), pt - pt.mean()
        num += float(((pt - xt) ** 2).sum())
        den += float((xt ** 2).sum())
    return r2, 1.0 - num / max(den, 1e-12)


def _test_field_full(spec, device="cpu", root=None):
    """ONE FREE ROLLOUT FROM t0 through the whole recording, compared with it volume by volume
    (Cedric, 2026-09-29): variance explained, total and spatial (`_r2_parts`), on the voxels that are
    tissue at the origin and at the target -- beside the same numbers for persistence (x(t0) held) and
    for the ceiling the voxel noise leaves, 1 - sigma^2 / variance, sigma the train volumes' noise floor."""
    from plexus.tasks import field_recording as FR
    engine.quiet(True)
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = _field_setup(spec, device)
    rec = box["rec"]
    o = box["n_in"] - 1
    T = rec["ratio"].shape[0]
    n = T - 1 - o
    if box.get("alive"):                           # a death operator's law needs its alive channel: one stacked rollout
        sims = _field_sims(spec, [n], train=False)
        with torch.no_grad():
            roll = _field_rollout(sims, learn, box, o, n, device, False)
        free = roll[:, 0].cpu().numpy()
        alive = roll[:, -1].cpu().numpy()
        del roll                                   # the GPU stack, before the `task.rollouts` variants need the room
    else:
        # IN SEGMENTS (exp19, 2026-10-03): the [n, C, *grid] stack of a whole-body rollout with 12 input volumes was
        # 56 GB and ran a 96 GB card out of memory; `_field_free` is the same rollout (tests/test_exp19_no_leak.py)
        free, _, _, _ = _field_free(spec, learn, box, device)
        alive = None
    obs = rec["ratio"][o + 1:]
    m = rec["mask"][o + 1:] & rec["mask"][o]
    hold = np.broadcast_to(rec["ratio"][o], obs.shape)
    r2, r2s = _r2_parts(free, obs, m)
    p2, p2s = _r2_parts(hold, obs, m)
    D = FR.structure_function(rec, np.arange(T), hs=(1, 2, 3))
    sig = FR.noise_floor(D)["sigma"]
    var = float(obs[m].var())
    var_s = float(np.mean([obs[t][m[t]].var() for t in range(n) if m[t].sum() > 1]))
    # THE DENOISED TARGET (Cedric, 2026-09-29): the same rollout scored against the recording averaged over
    # 3 voxels in z, y, x and 3 frames (`field_recording.denoise`), with its own noise ceiling from its own
    # structure function. Beside the raw score, never instead of it: the box is fixed in advance.
    recd = FR.denoise(rec)
    obs_d = recd["ratio"][o + 1:]
    r2d, r2ds = _r2_parts(free, obs_d, m)
    p2d, p2ds = _r2_parts(hold, obs_d, m)
    # ITS NOISE FROM LAGS 3-6 ONLY: the 3-frame average makes frames 1 and 2 apart share data, which
    # shrinks D(1), D(2) and drove the nugget to 0 (a ceiling of exactly 1). Agrees with the raw noise
    # floor over the box's 81 samples (0.979 against 0.981, measured 2026-09-29).
    sig_d = FR.noise_floor(FR.structure_function(recd, np.arange(T), hs=(3, 4, 5, 6)), fit_hs=(3, 4, 5, 6))["sigma"]
    res = {"name": spec["name"], "mode": "full", "origin_volume": o + 1, "steps": n,
           "r2": r2, "r2_spatial": r2s, "persistence_r2": p2, "persistence_r2_spatial": p2s,
           "ceiling_r2": 1 - sig ** 2 / var, "ceiling_r2_spatial": 1 - sig ** 2 / var_s, "noise_sigma": sig,
           "r2_denoised": r2d, "r2_spatial_denoised": r2ds, "persistence_r2_denoised": p2d,
           "persistence_r2_spatial_denoised": p2ds, "noise_sigma_denoised": sig_d,
           "ceiling_r2_denoised": 1 - sig_d ** 2 / float(obs_d[m].var()),
           # IS WHAT IS LEFT GAUSSIAN WHITE NOISE? The model's residual against the denoised recording, and
           # -- the check on the target itself -- the recording's own residual against it.
           "residual_model": {**FR.residual_stats(free - obs_d, m),
                              "level_corr": float(np.corrcoef(np.abs(free - obs_d)[m], obs_d[m])[0, 1])},
           "residual_data": {**FR.residual_stats(obs - obs_d, m),
                             "level_corr": float(np.corrcoef(np.abs(obs - obs_d)[m], obs_d[m])[0, 1])},
           "rmse": [float(np.sqrt(np.mean((free[k][m[k]] - obs[k][m[k]]) ** 2))) for k in range(n)],
           "persistence_rmse": [float(np.sqrt(np.mean((hold[k][m[k]] - obs[k][m[k]]) ** 2))) for k in range(n)],
           "organoid_mean_model": [float(free[k][m[k]].mean()) for k in range(n)],
           "organoid_mean_recorded": [float(obs[k][m[k]].mean()) for k in range(n)],
           "t_min": ((rec["t_s"][o + 1:] - rec["t_s"][o]) / 60).tolist()}
    # THE SHAPE (2026-09-29): the model's tissue -- its ratio above 0.15, half the recording's p5 on the
    # tissue (0.33), model choice -- against the recorded tissue at each frame, as IoU on the global mask;
    # beside it, frame o's own tissue held (the shape of persistence).
    U = rec["mask"].any(0)
    Mt = rec["mask"][o + 1:]
    # WITH A DEATH OPERATOR THE MODEL'S TISSUE IS ITS OWN ALIVE FIELD, A > 1/2 (exp16, 2026-10-01)
    pt = (alive > 0.5) if alive is not None else (free > 0.15) & U
    iou = [float((pt[k] & Mt[k]).sum() / max((pt[k] | Mt[k]).sum(), 1)) for k in range(n)]
    iou0 = [float((rec["mask"][o] & Mt[k]).sum() / max((rec["mask"][o] | Mt[k]).sum(), 1)) for k in range(n)]
    res.update(shape_iou=iou, shape_iou_hold=iou0, shape_iou_mean=float(np.mean(iou)),
               shape_iou_hold_mean=float(np.mean(iou0)),
               tissue_voxels_model=[int(pt[k].sum()) for k in range(n)], tissue_voxels_recorded=[int(Mt[k].sum()) for k in range(n)])
    np.savez_compressed(os.path.join(out, "results", f"{spec['name']}_free.npz"), pred=free.astype(np.float16), origin=o,
                        **({"alive": alive.astype(np.float16)} if alive is not None else {}))
    # EXP17'S BRAIN-MEAN METRICS (Cedric's headline, 2026-10-03) on the field's mean trace, and the `task.rollouts` variants
    om_, fm_ = np.array(res["organoid_mean_recorded"]), np.array(res["organoid_mean_model"])
    res.update(_brain_mean_metrics(om_, fm_), brain_mean_sd_obs=float(om_.std()), brain_mean_sd_pred=float(fm_.std()),
               brain_mean_r=float(np.corrcoef(fm_, om_)[0, 1]) if fm_.std() > 0 and om_.std() > 0 else float("nan"))
    print(f"[test] brain-mean R2 {res['brain_mean_r2']:+.3f} (RMSE {res['brain_mean_rmse']:.4f}, SD "
          f"{res['brain_mean_sd_pred']:.4f} against {res['brain_mean_sd_obs']:.4f} recorded, r {res['brain_mean_r']:+.2f})")
    res["rollouts"] = _field_variants(spec, learn, box, device, out, spec["name"])
    json.dump(res, open(os.path.join(out, "results", f"{spec['name']}_test.json"), "w"), indent=2)
    print(f"[test] full rollout from volume {o + 1}, {n} steps: R2 {r2:+.4f} (spatial {r2s:+.4f}); persistence "
          f"{p2:+.4f} ({p2s:+.4f}); noise ceiling {res['ceiling_r2']:.4f} ({res['ceiling_r2_spatial']:.4f})")
    print(f"[test] shape: IoU of the model's tissue with the recorded, mean {res['shape_iou_mean']:.3f} "
          f"(frame {o + 1}'s shape held: {res['shape_iou_hold_mean']:.3f}); tissue voxels last frame model "
          f"{res['tissue_voxels_model'][-1]} recorded {res['tissue_voxels_recorded'][-1]}")
    print(f"[test] against the recording denoised (3 voxels x 3 frames): R2 {r2d:+.4f} (spatial {r2ds:+.4f}); "
          f"persistence {p2d:+.4f}; ceiling {res['ceiling_r2_denoised']:.4f}")
    for k in ("residual_model", "residual_data"):
        q = res[k]
        print(f"[test] {k}: sd {q['sd']:.4f} skew {q['skew']:+.2f} kurt {q['excess_kurtosis']:+.2f}  autocorr t {q['autocorr_t']:+.2f} "
              f"z {q['autocorr_z']:+.2f} y {q['autocorr_y']:+.2f} x {q['autocorr_x']:+.2f}  |r| vs level {q['level_corr']:+.2f}")
    return res


def _test_field(spec, device="cpu", root=None):
    """The kept checkpoint on the TEST volumes, per horizon, against persistence scored by the same
    call; and one FREE rollout from the last volume before the test window through all of it, whose
    organoid-mean ratio is the washout response the model must follow (exp16's G20)."""
    from plexus.tasks import field_recording as FR
    engine.quiet(True)
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = _field_setup(spec, device)
    rec, te = box["rec"], box["test"]
    H_TEST = (1, 3, 6)
    n_free = len(te)
    sims = _field_sims(spec, list(H_TEST) + [n_free], train=False)
    f = _field_forecaster(sims, learn, box, device)
    res = {"name": spec["name"], "test_volumes": [int(te[0]) + 1, int(te[-1]) + 1], "horizons": {}}
    for h in H_TEST:
        m, p = FR.score(f, rec, te, h), FR.score(FR.persistence(rec), rec, te, h)
        m.update(persistence_rmse=p["rmse"], skill=1 - m["mse"] / p["mse"])
        res["horizons"][str(h)] = m
        print(f"[test] h={h} ({h * 10} min): RMSE {m['rmse']:.5f}  persistence {p['rmse']:.5f}  "
              f"skill {m['skill']:+.4f}  ({m['n_vox']:,} voxels over {m['n_pairs']} volumes)")
    o = int(te[0]) - 1
    with torch.no_grad():
        free = _field_rollout(sims, learn, box, o, n_free, device, False)[:, 0].cpu().numpy()
    obs = rec["ratio"][o + 1:o + 1 + n_free]
    msk = rec["mask"][o + 1:o + 1 + n_free]
    m0 = rec["mask"][o]
    res["free"] = {
        "origin_volume": o + 1,
        "rmse": [float(np.sqrt(np.mean((free[k][m0 & msk[k]] - obs[k][m0 & msk[k]]) ** 2))) for k in range(n_free)],
        "persistence_rmse": [float(np.sqrt(np.mean((rec["ratio"][o][m0 & msk[k]] - obs[k][m0 & msk[k]]) ** 2)))
                             for k in range(n_free)],
        # BOTH ORGANOID MEANS ARE READ ON THE SCORER'S VOXELS, tissue at the origin AND at the target.
        # Read on the target's mask alone, the model's mean took in the voxels the tissue moved INTO,
        # where the field started at 0 -- persistence then read 0.04 below the recording (v0).
        "organoid_mean_model": [float(free[k][m0 & msk[k]].mean()) for k in range(n_free)],
        "organoid_mean_recorded": [float(obs[k][m0 & msk[k]].mean()) for k in range(n_free)],
    }
    np.savez_compressed(os.path.join(out, "results", f"{spec['name']}_free.npz"),
                        pred=free.astype(np.float16), origin=o)
    json.dump(res, open(os.path.join(out, "results", f"{spec['name']}_test.json"), "w"), indent=2)
    return res


def _analyse_field(spec, device="cpu", root=None):
    """One figure: RMSE per horizon against persistence, the free rollout's organoid mean against
    the recording's, and the mid plane of the last test volume -- recorded, forecast, difference --
    plus a movie of the free rollout beside the recording."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = out_dir(spec, root)
    res = json.load(open(os.path.join(out, "results", f"{spec['name']}_test.json")))
    if res.get("mode") == "full":
        return _analyse_field_full(spec, res, out)
    box = _field_setup(spec, "cpu")
    rec = box["rec"]
    z = np.load(os.path.join(out, "results", f"{spec['name']}_free.npz"))
    free, o = z["pred"].astype(np.float32), int(z["origin"])
    n = free.shape[0]
    obs, msk = rec["ratio"][o + 1:o + 1 + n], rec["mask"][o + 1:o + 1 + n] & rec["mask"][o]
    zm = free.shape[1] // 2
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.2))
    for a in ax[:2]:
        a.spines[["top", "right"]].set_visible(False)
    hs = sorted(int(h) for h in res["horizons"])
    ax[0].plot([h * 10 for h in hs], [res["horizons"][str(h)]["persistence_rmse"] for h in hs], "o-",
               color="0.4", label="persistence")
    ax[0].plot([h * 10 for h in hs], [res["horizons"][str(h)]["rmse"] for h in hs], "o-", color="tab:red",
               label="model")
    ax[0].set_xlabel("forecast horizon (min)")
    ax[0].set_ylabel("RMSE of the ratio on the organoid")
    ax[0].legend(frameon=False)
    ax[0].set_title("test volumes, one forecast per origin", fontsize=10, loc="left")
    tt = (rec["t_s"][o + 1:o + 1 + n] - rec["t_s"][o]) / 60
    ax[1].plot(tt, res["free"]["organoid_mean_recorded"], "o-", color="black", ms=3, label="recorded")
    ax[1].plot(tt, res["free"]["organoid_mean_model"], "o-", color="green", ms=3, label="model, free rollout")
    ax[1].set_xlabel(f"minutes after volume {o + 1}")
    ax[1].set_ylabel("organoid-mean ratio")
    ax[1].legend(frameon=False)
    ax[1].set_title("the washout response over the test window", fontsize=10, loc="left")
    k = n - 1
    lo, hi = np.percentile(obs[k][msk[k]], [2, 98])
    for a, img, lab in ((ax[2], np.where(msk[k, zm], obs[k, zm], np.nan), f"recorded, volume {o + 1 + n}"),
                        (ax[3], np.where(msk[k, zm], free[k, zm], np.nan), f"model, {n} steps from {o + 1}"),
                        (ax[4], np.where(msk[k, zm], free[k, zm] - obs[k, zm], np.nan), "model - recorded")):
        a.imshow(img, cmap="viridis" if a is not ax[4] else "RdBu_r",
                 vmin=lo if a is not ax[4] else -(hi - lo) / 2, vmax=hi if a is not ax[4] else (hi - lo) / 2)
        a.set_axis_off()
        a.set_title(f"{lab}, plane {zm}", fontsize=10, loc="left")
    fig.tight_layout()
    p = os.path.join(out, "results", f"{spec['name']}_test.png")
    fig.savefig(p, dpi=110, facecolor="white")
    plt.close(fig)
    insets = []
    names = _identity_fields(spec)
    if names:
        # A failure here must not cost the run its movie (exp16 v34-v37 lost theirs).
        try:
            insets = _field_embedding_figure(spec, names, box, out)
        except Exception as ex:                                                 # noqa: BLE001
            print(f"[analyse] no per-voxel identity analysis: {type(ex).__name__}: {ex}")
    _field_movie(out, obs, free, msk, rec, (rec["t_s"][o + 1:o + 1 + n] - rec["t_s"][o]) / 60, insets)
    print(f"[analyse] -> {p}")
    return p


def _field_movie(out, obs, pred, msk, rec, t_min, insets=(), own=None):
    """`results/movie.mp4`: the recording LEFT and the model RIGHT as 3-D oblique volumes
    (`field_recording.render_pair_3d`) -- the watcher's movie of the run."""
    from plexus.tasks import field_recording as FR
    try:
        # BOTH PANELS ON THE SCORED VOXELS, the per-frame mask (Cedric, 2026-09-29, from v30 on): tissue at t0
        # AND at frame t, so the outline in both panels is the recording's -- the law draws no boundary of its
        # own (findings 16, 18). The movies of v18-v29 were rendered once on the global mask instead, which
        # shows the model's own volume. R2 raw and against the denoised recording.
        o = rec["ratio"].shape[0] - 1 - obs.shape[0]
        obs_d = FR.denoise(rec)["ratio"][o + 1:o + 1 + obs.shape[0]]
        # `own` = (the model's tissue A > 1/2, the recorded tissue, IoU per frame) for a run with a death operator:
        # each panel on its own tissue, the shape IoU printed under the R2 (Cedric, 2026-10-01).
        kw = {} if own is None else dict(mask_pred=own[0], mask_obs=own[1], iou=own[2])
        FR.render_pair_3d(obs, pred, msk, rec["dx_um"], rec["dz_um"], os.path.join(out, "results", "movie.mp4"),
                          times_min=t_min, insets=insets, obs_denoised=obs_d, **kw)
        print(f"[analyse] movie -> {os.path.join(out, 'results', 'movie.mp4')}")
    except Exception as e:                        # a movie is a convenience; the figure is the result
        print(f"[analyse] no movie: {type(e).__name__}: {e}")


def _analyse_field_full(spec, res, out):
    """The full rollout from t0 against the recording: RMSE per volume (model, persistence), the
    organoid-mean ratio (recorded, model), the variance explained, and the mid plane at three times;
    then the 3-D movie of the whole rollout, and the embedding's analysis when there is one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    box = _field_setup(spec, "cpu")
    rec = box["rec"]
    z = np.load(os.path.join(out, "results", f"{spec['name']}_free.npz"))
    free, o = z["pred"].astype(np.float32), int(z["origin"])
    n = free.shape[0]
    obs, msk = rec["ratio"][o + 1:o + 1 + n], rec["mask"][o + 1:o + 1 + n] & rec["mask"][o]
    t = np.array(res["t_min"]) / 60
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.3))
    for a in ax[:2]:
        a.spines[["top", "right"]].set_visible(False)
    ax[0].plot(t, res["persistence_rmse"], color="0.4", label="persistence (volume held)")
    ax[0].plot(t, res["rmse"], color="tab:red", label="model, free rollout")
    ax[0].axhline(res["noise_sigma"], color="0.6", ls="--", label="noise floor")
    ax[0].set_xlabel(f"hours after volume {o + 1}")
    ax[0].set_ylabel("RMSE of the ratio on the tissue")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(f"R2 {res['r2']:+.3f} (spatial {res['r2_spatial']:+.3f}); persistence {res['persistence_r2']:+.3f} "
                    f"({res['persistence_r2_spatial']:+.3f}); ceiling {res['ceiling_r2']:.3f}", fontsize=9, loc="left")
    ax[1].plot(t, res["organoid_mean_recorded"], color="black", label="recorded")
    ax[1].plot(t, res["organoid_mean_model"], color="green", label="model")
    ax[1].set_xlabel(f"hours after volume {o + 1}")
    ax[1].set_ylabel("organoid-mean ratio")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("the washout response, the whole recording", fontsize=10, loc="left")
    zm = free.shape[1] // 2
    lo, hi = np.percentile(obs[msk], [2, 98])
    for a, k in zip(ax[2:], (n // 3, 2 * n // 3, n - 1)):
        both = np.concatenate([np.where(msk[k, zm], obs[k, zm], np.nan), np.full((obs.shape[2], 4), np.nan),
                               np.where(msk[k, zm], free[k, zm], np.nan)], 1)
        a.imshow(both, vmin=lo, vmax=hi)
        a.set_axis_off()
        a.set_title(f"plane {zm}, {t[k]:.1f} h: recorded | model", fontsize=10, loc="left")
    fig.tight_layout()
    p = os.path.join(out, "results", f"{spec['name']}_test.png")
    fig.savefig(p, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"[analyse] -> {p}")
    insets = []
    names = _identity_fields(spec)
    if names:
        # A failure here must not cost the run its movie (exp16 v34-v37 lost theirs).
        try:
            insets = _field_embedding_figure(spec, names, box, out)
        except Exception as ex:                                                 # noqa: BLE001
            print(f"[analyse] no per-voxel identity analysis: {type(ex).__name__}: {ex}")
    own = None
    if "alive" in z.files:                        # a death operator: the model drawn on ITS OWN tissue, A > 1/2
        own = (z["alive"].astype(np.float32) > 0.5, rec["mask"][o + 1:o + 1 + n], res.get("shape_iou"))
    _field_movie(out, obs, free, msk, rec, np.array(res["t_min"]), insets, own)
    return p


KNOWN_ODE_FIELDS = ("rest", "rate", "beta", "barrier")


def _identity_fields(spec):
    """The learned fields that say what each voxel IS: the embedding when the law has one; for the known ODE, its
    per-voxel constants (set point r*, rate 1/tau, sensitivity beta, barrier b) -- the interpretable law's own
    identity, clustered the same way so both laws answer "do voxels group into cells?" (Cedric, 2026-10-01)."""
    learned = [e["field"] for e in spec["learnable"] if "field" in e]
    ko = [n for n in KNOWN_ODE_FIELDS if n in learned]
    if ko:
        # A KNOWN ODE IS READ BY ITS CONSTANTS, even when it also carries an embedding: v34's never learned (its gate
        # had no gradient), and clustering it grouped the hash grid's smooth starting noise into false domains.
        return ko
    return ["embedding"] if "embedding" in learned else []


def _field_embedding_figure(spec, names, box, out, device="cpu"):
    """The learned per-voxel EMBEDDING, where cells might emerge (exp16): its first three principal
    components over the tissue voxels drawn as RGB on three planes, and the tissue voxels scattered
    in the first two. The tissue is the mask of the last training volume. Writes
    `results/<name>_embedding.png` and `.npz` (the embedding itself, for the rulers)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    sim = _model(spec, train=False, n_frames=0)
    with torch.no_grad():
        H, _ = engine.run(sim, device=device, progress=False, on_seeded=learn.inject)
    a = np.concatenate([H.fields[n].grid.detach().cpu().numpy() for n in names], 0)   # [k, Z, Y, X]
    if len(names) > 1:
        # SEVERAL CONSTANTS OF DIFFERENT SCALES (a rate ~0.2 per tick, a barrier ~0.01): each standardised over the
        # tissue first, so the principal components and the clusters weigh them alike.
        mm = box["rec"]["mask"][int(box["train"][-1])]
        a = (a - a[:, mm].mean(1)[:, None, None, None]) / np.maximum(a[:, mm].std(1), 1e-12)[:, None, None, None]
    np.savez_compressed(os.path.join(out, "results", f"{spec['name']}_embedding.npz"), a=a.astype(np.float32))
    m = box["rec"]["mask"][int(box["train"][-1])]
    v = a[:, m].T                                                           # [n tissue voxels, k]
    v = v - v.mean(0)
    U, S, Vt = np.linalg.svd(v[:: max(1, len(v) // 50000)], full_matrices=False)
    pc = (a.reshape(a.shape[0], -1).T - a[:, m].T.mean(0)) @ Vt[:3].T      # [N, <=3]
    pc = pc.reshape(a.shape[1:] + (pc.shape[1],))
    lo, hi = np.percentile(pc[m], 2, axis=0), np.percentile(pc[m], 98, axis=0)
    rgb = np.clip((pc - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
    if rgb.shape[-1] < 3:
        rgb = np.concatenate([rgb, np.zeros(rgb.shape[:-1] + (3 - rgb.shape[-1],))], -1)
    Z = a.shape[1]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for j, z in enumerate((Z // 4, Z // 2, 3 * Z // 4)):
        img = np.where(m[z][..., None], rgb[z], 1.0)
        ax[j].imshow(img)
        ax[j].set_axis_off()
        ax[j].set_title(f"embedding PC1-3 as RGB, plane {z}", fontsize=10, loc="left")
    pm = pc[m]
    sub = pm[:: max(1, len(pm) // 20000)]
    ax[3].scatter(sub[:, 0], sub[:, 1] if sub.shape[1] > 1 else np.zeros(len(sub)), s=1, c="0.3", alpha=0.3)
    ax[3].spines[["top", "right"]].set_visible(False)
    ax[3].set_xlabel("PC1")
    ax[3].set_ylabel("PC2")
    ev = S ** 2 / max((S ** 2).sum(), 1e-12)
    ax[3].set_title(f"tissue voxels; variance explained {', '.join(f'{x:.2f}' for x in ev[:3])}", fontsize=10, loc="left")
    fig.tight_layout()
    q = os.path.join(out, "results", f"{spec['name']}_embedding.png")
    fig.savefig(q, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"[analyse] embedding ({a.shape[0]} per voxel) -> {q}")
    # DO CELLS EMERGE? The embedding clustered, the clusters mapped back into the volume
    # (`field_recording.cluster_embedding`): separation in embedding space, contiguity in 3-D.
    from plexus.tasks import field_recording as FR
    lab, st, lab8 = FR.cluster_embedding(a, m, box["rec"]["dx_um"], box["rec"]["dz_um"])
    json.dump(st, open(os.path.join(out, "results", f"{spec['name']}_clusters.json"), "w"), indent=1)
    insets = []
    try:
        pe = os.path.join(out, "results", f"{spec['name']}_embedding3d.png")
        what = "embedding" if names == ["embedding"] else "learned constants " + ", ".join(names)
        FR.render_embedding_3d(a, m, box["rec"]["dx_um"], box["rec"]["dz_um"], pe, title=f"{what}, PC1-3 as RGB")
        dom = FR.domains(lab8)
        pd = os.path.join(out, "results", f"{spec['name']}_domains3d.png")
        FR.render_domains_3d(dom, box["rec"]["dx_um"], box["rec"]["dz_um"], pd,
                             title=f"{int(dom.max()) + 1} domains, median {st['domain_um3_median_k8']:.0f} um3")
        ps = os.path.join(out, "results", f"{spec['name']}_embedding_scatter.png")
        FR.render_embedding_scatter(a, m, lab8, ps, title=f"{what}, tissue voxels, 8 clusters")
        insets = [ps, pe, pd]
    except Exception as ex:
        print(f"[analyse] no 3-D embedding render: {type(ex).__name__}: {ex}")
    print(f"[analyse] clusters: k {st['k']} silhouette {st['silhouette']:.2f}, coherence {st['coherence']:.2f} "
          f"(shuffled {st['coherence_null']:+.2f}); at 8 clusters {st['coherence_k8']:.2f}, median domain "
          f"{st['domain_um3_median_k8']:.0f} um3")
    return insets


# ============================================================================== entry points
# ============================================================================== trace recordings
# A RECORDING OF TRACES ON A SET (`task.reference.trace_recording`, exp17): one value per element per frame at
# fixed positions, as `plexus.tasks.trace_recording` loads it. The observed block holds each element's value and
# its `inputs - 1` past values, newest first, seeded from the recording at an origin; the known stimulus, when the
# task drives one, is a window of frames written into a set's block before every step (the drive enters through
# the state of a named entity, plexus2.tex). Every frame is trained on (Cedric: no time blocks); the curriculum is
# connectome-gnn's -- one stage per horizon, every step scored, uniform, a constant learning rate -- and the last
# checkpoint is kept. The score is STANDARD MSE as skill over the mean baseline (`trace_recording.skills`).
def _trace_setup(spec, device):
    from plexus.tasks import trace_recording as TR
    t = spec["task"]
    rec = TR.load(t["reference"]["trace_recording"])
    X = torch.as_tensor(rec["dff"], device=device)
    sim = _model(spec, train=False, n_frames=0)
    with torch.no_grad():
        H, _ = engine.run(sim, device=device, progress=False)
    obs = t["observe"]
    lvl = H.level(obs["set"])
    a, b = lvl.state_schema[obs["block"]]
    if lvl.n != X.shape[1]:
        raise ValueError(f"the model's set `{obs['set']}` has {lvl.n} elements, the recording {X.shape[1]}")
    drv, S = t.get("drive"), None
    if drv:
        dl = H.level(drv["set"])
        da, db = dl.state_schema[drv["block"]]
        w0, w1 = drv["window"]
        if dl.n != rec["stimulus"].shape[1] or db - da != w1 - w0 + 1:
            raise ValueError(f"the drive set `{drv['set']}` is {dl.n} x {db - da}; the stimulus window needs "
                             f"{rec['stimulus'].shape[1]} elements x {w1 - w0 + 1} frames")
        S = torch.as_tensor(rec["stimulus"], device=device)
    d1 = X[1:] - X[:-1]
    # GRAPHCAST'S NORMALISATION (supplement 3.7), from the reference: inputs to zero mean and unit variance, the
    # output in units of the one-step difference's standard deviation. Handed to the law at `on_ready`.
    norm = (float(X.mean()), float(X.std()), float(d1.std()))
    split, lab = t["reference"].get("split", "all"), None
    if split in ("zapbench", "recording"):
        # ZAPBENCH'S SPLIT, or THE RECORDING'S OWN (`split` in its npz, 0 train / 2 held out: exp20's held-out trial
        # windows, written by its exporter): the labels per frame, and the normalisation from the TRAINING frames
        # only (no statistic of a test frame reaches the law)
        lab = (TR.zapbench_split(rec["offsets"], int(X.shape[0])) if split == "zapbench"
               else np.asarray(rec["split"]).astype(np.int64))
        tr_ = torch.as_tensor(lab == 0, device=X.device)
        Xt = X[tr_]
        norm = (float(Xt.mean()), float(Xt.std()), float(d1[tr_[1:] & tr_[:-1]].std()))
        del Xt
    del d1
    return dict(rec=rec, X=X, S=S, n_in=b - a, norm=norm, T=int(X.shape[0]), split=split, lab=lab)


def restore_trace(spec, device="cpu", root=None):
    """A LANDED TRACE RUN, RESTORED FOR ANALYSIS (the graph phase; the tools that read a trained law): its learnables
    from models/<checkpoint or best>.pt, the recording and the normalisation (`_trace_setup`), and the law's operator
    with the fitted parameters set on it by a one-frame engine run (`learn.ready`), so `op.W_*`, `op.norm`, the frame
    clock and the per-element blocks (tau, rest, input: `blocks`) read as every rollout reads them. Honours
    `spec["_checkpoint"]`. -> dict(out, stem, ckpt, it, learn, box, op, H, blocks)."""
    engine.quiet(True)
    out = out_dir(spec, root)
    ckn = spec.get("_checkpoint")
    stem = f"{spec['name']}_{ckn}" if ckn else spec["name"]
    ck = torch.load(os.path.join(out, "models", f"{ckn or 'best'}.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = _trace_setup(spec, device)
    got = {}

    def ready(H):
        learn.ready(H)
        for op in H.operators:
            if getattr(op, "NORM_FROM_REFERENCE", False):
                op.norm = box["norm"]
        _clock(H, 0, box["T"])
        got["H"] = H
    with torch.no_grad():
        engine.run(_model(spec, train=False, n_frames=1), device=device, progress=False, grad=False,
                   on_seeded=learn.inject, on_ready=ready)
    H = got["H"]
    ops = [o for nm, o in zip(H.operator_names, H.operators) if nm == "state_diffuse"]
    if len(ops) != 1:
        raise ValueError(f"restore_trace: {len(ops)} state_diffuse operators in the model, one expected")
    lvl = H.level(spec["task"]["observe"]["set"])
    blocks = {k: lvl.state[:, a:b].detach() for k, (a, b) in lvl.state_schema.items()}
    return dict(out=out, stem=stem, ckpt=ckn or "best", it=ck.get("it"), learn=learn, box=box, op=ops[0], H=H,
                blocks=blocks)


def _frame_s(rec) -> float:
    """Seconds per recorded frame: the recording's own clock (`t_s`), else ZAPBench's 0.914 s (exp17's default; every
    exp17 recording has t_s = frame x 0.914, so its numbers do not move). exp20's sessions run at 1.117 s."""
    t = rec.get("t_s") if isinstance(rec, dict) else None
    return float(np.median(np.diff(np.asarray(t)))) if t is not None and len(t) > 1 else 0.914


def _clock(H, tc, T):
    """THE FRAME CLOCK: a law whose terms depend on the absolute recording time (`FRAME_CLOCK`, the neuron graph's
    modulation Omega(x, y, z, t)) is told the frame of the step it is about to take, and the recording's length."""
    H._frame_clock = (int(tc), int(T))          # remembered: at `on_seeded` the operators do not exist yet
    for op in getattr(H, "operators", ()):
        if getattr(op, "FRAME_CLOCK", False):
            op.frame, op.n_frames_ref = int(tc), int(T)


def _warmup(spec) -> int:
    """`task.warmup` (exp17 batch 12, Cedric 2026-10-02): recorded frames written into the observed block before the
    free steps, so the law's hidden states (adaptation, a calcium latent) settle on the real trajectory; 0 = off."""
    return int(spec["task"].get("warmup", 0) or 0)


def _trace_sim(spec, train, K):
    """The model for a K-frame forecast, with room for the warm-up ticks before it."""
    return _model(spec, train=train, n_frames=_warmup(spec) + K - 1)


def _trace_rollout(sim, learn, spec, box, o, K, device, grad):
    """K frames from origin o: [K, N], row k the forecast of frame o+k+1 (`sim` from `_trace_sim`). With
    `task.warmup` w the rollout starts at o - w and the observed block is overwritten with the recorded frames
    o-w+1 .. o (teacher forcing) before the K free steps; only those K are returned."""
    t = spec["task"]
    obs, drv, X, S, T = t["observe"], t.get("drive"), box["X"], box["S"], box["T"]
    w = _warmup(spec)
    o, o_free = o - w, o
    if o - box["n_in"] + 1 < 0 or o_free + K >= T:
        raise ValueError(f"origin {o_free} with horizon {K}: needs {box['n_in'] - 1 + w} frames behind it and {K} ahead")
    x0 = torch.stack([X[o - j] for j in range(box["n_in"])], 1)                 # [N, inputs], newest first
    frames = []

    def window(H, tc):
        """The stimulus window around frame tc, for the step from tc to tc + 1 (read by the law at that step)."""
        _clock(H, tc, T)
        if drv is None:
            return
        lv = H.level(drv["set"])
        a, b = lv.state_schema[drv["block"]]
        idx = (torch.arange(drv["window"][0], drv["window"][1] + 1, device=X.device) + tc).clamp(0, T - 1)
        st = lv.state.clone()
        st[..., a:b] = S[idx].T
        lv.state = st

    def seeded(H):
        lv = H.level(obs["set"])
        a, b = lv.state_schema[obs["block"]]
        st = lv.state.clone()
        st[..., a:b] = x0
        lv.state = st
        window(H, o)
        learn.inject(H)

    def hook(H, tick):
        lv = H.level(obs["set"])
        a, b = lv.state_schema[obs["block"]]
        if tick < w:                       # the warm-up: the recorded frame replaces the law's
            st = lv.state.clone()
            st[..., a:b] = torch.stack([X[o + tick + 1 - j] for j in range(box["n_in"])], 1)
            lv.state = st
        else:
            frames.append(lv.state[..., a].reshape(-1))
        window(H, o + tick + 1)

    def ready(H):
        learn.ready(H)
        for op in H.operators:
            if getattr(op, "NORM_FROM_REFERENCE", False):
                op.norm = box["norm"]
        if hasattr(H, "_frame_clock"):
            _clock(H, *H._frame_clock)
        for op in H.operators:             # a latent state inferred from the recorded frames (calcium_indicator)
            if hasattr(op, "init_latent"):
                op.init_latent(H)

    engine.run(sim, device=device, progress=False, grad=grad, on_frame=hook, on_seeded=seeded, on_ready=ready)
    return torch.stack(frames[:K])


def _trace_residual(pred, box, o, term):
    """The residual of each step. `trace_mse`: the state, GraphCast's (pred - x). `trace_increment`:
    connectome-gnn's, the error of each step's increment along the model's own trajectory."""
    X, K = box["X"], pred.shape[0]
    tgt = X[o + 1:o + K + 1]
    if term == "trace_mse":
        return pred - tgt, tgt
    prev = torch.cat([X[o][None], pred[:-1]], 0)
    return (pred - prev) - (tgt - X[o:o + K]), tgt


def _trace_evidence(pred, box, o):
    def ev(term):
        def f(reduction):
            r, tgt = _trace_residual(pred, box, o, term)
            if reduction == "norm2":
                # CONNECTOME-GNN'S SCALE: its fit term is ||r / ynorm||_2, the residual in units of the one-step
                # change's spread, so its regulariser coefficients (coeff_*) keep their balance with the fit.
                r = r / box["norm"][2]
            return sum(_reduce(r[k], reduction, tgt[k]) for k in range(r.shape[0])) / r.shape[0]
        return f
    return {"trace_mse": ev("trace_mse"), "trace_increment": ev("trace_increment")}


def _trace_eval(spec, learn, box, device, origins, hmax=32):
    """MSE over elements per step ahead [hmax, len(origins)] for the law and for the best mean baseline's W's."""
    from plexus.tasks import trace_recording as TR
    sim = _trace_sim(spec, False, hmax)
    X = box["X"]
    m = np.zeros((hmax, len(origins)))
    with torch.no_grad():
        for i, o in enumerate(origins):
            p = _trace_rollout(sim, learn, spec, box, int(o), hmax, device, False)
            m[:, i] = ((p - X[o + 1:o + hmax + 1]) ** 2).double().mean(1).cpu().numpy()   # float64, as the baselines
    ot = torch.as_tensor(np.asarray(origins), device=X.device)
    base = np.stack([TR.mse_per_origin(lambda oo, h, W=W: TR.mean_pred(X, oo, W), X, ot, range(1, hmax + 1))
                     for W in range(1, TR.W_MAX + 1)], 0)                             # [W, h, origin]
    return m, base


def _train_trace(spec, device="cpu", root=None):
    from plexus.tasks import trace_recording as TR
    tr = spec["training"]
    seed = int(tr.get("seed", 0))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = out_dir(spec, root)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    shutil.copyfile(spec["_path"], os.path.join(out, "config.yaml"))
    shutil.copyfile(spec["model"], os.path.join(out, "model.yaml"))
    print(f"[run] {spec['name']} -> {out}")
    engine.quiet(True)
    box = _trace_setup(spec, device)
    T, n_in = box["T"], box["n_in"]
    stages = [(int(s["horizon"]), int(s["iters"])) for s in tr["stages"]]
    kmax = max(k for k, _ in stages)
    # THE ORIGINS ARE DRAWN FROM ONE RANGE IN EVERY STAGE, bounded by the LARGEST horizon (connectome-gnn's
    # get_training_frame_sampling): the curriculum is then the only thing that changes between stages.
    w = _warmup(spec)                       # the warm-up frames, behind each origin and inside the training frames
    origins = np.arange(n_in - 1 + w, T - 1 - kmax)
    if box["lab"] is not None:               # every frame of every training window is a training frame
        origins = TR.split_origins(box["lab"], box["rec"]["condition"], "train", n_in + w, kmax)
    sims = {k: _trace_sim(spec, True, k) for k in sorted({k for k, _ in stages})}
    learn = Learnables(spec["learnable"], device)
    _trace_rollout(sims[stages[0][0]], learn, spec, box, int(origins[0]), stages[0][0], device, False)  # creates the leaves
    if tr.get("init_from"):
        # A WARM START (exp17, Cedric 2026-10-05: "smoke tests on top of 15.14"): every learnable starts at another
        # run's trained values (its models/best.pt); a learnable it does not have, or of another shape, is refused
        src_ = load(str(tr["init_from"]))
        fit_ = torch.load(os.path.join(out_dir(src_, root), "models", "best.pt"), weights_only=False,
                          map_location=device)["fitted"]
        bad_ = [k for k, v in learn.p.items() if k not in fit_ or fit_[k].shape != v.shape]
        if bad_:
            raise ValueError(f"training.init_from {tr['init_from']}: no trained value of the same shape for {bad_}")
        learn.restore({k: fit_[k] for k in learn.p})
        print(f"[init] {len(learn.p)} learnables from {tr['init_from']}", flush=True)
        _trace_rollout(sims[stages[0][0]], learn, spec, box, int(origins[0]), stages[0][0], device, False)
    start_it = 0
    if tr.get("resume"):
        # A RESUME (Cedric, 2026-10-08: 23.3 and 23.9 stopped ~1,500 updates from their end): this run's own
        # models/<resume>.pt -- its learnables and its update count -- and the curriculum picks up at that update
        # (`skip` below). Adam's moments and the origin draws are not in a checkpoint: the moments restart from
        # zero and the origins are drawn afresh, so a resumed run is not bit-identical to an uninterrupted one.
        ck_ = torch.load(os.path.join(out, "models", f"{tr['resume']}.pt"), weights_only=False, map_location=device)
        learn.restore({k: ck_["fitted"][k] for k in learn.p})
        start_it = int(ck_["it"])
        print(f"[resume] {len(learn.p)} learnables and update {start_it} from models/{tr['resume']}.pt", flush=True)
        _trace_rollout(sims[stages[0][0]], learn, spec, box, int(origins[0]), stages[0][0], device, False)
    params = learn.parameters()
    n_par = sum(p.numel() for p in params)
    lr, batch, clip = float(tr.get("lr", 1e-3)), max(1, int(tr.get("batch", 4))), float(tr.get("clip", 1.0))
    guard = tr.get("guard", "restore_and_halve")
    opt = torch.optim.Adam(learn.groups(lr), lr=lr)
    # A CONSTANT LEARNING RATE (connectome-gnn's recurrent scheme) unless the spec asks for the cosine.
    iters = sum(n for _, n in stages)
    sch = None if tr.get("schedule", "none") == "none" else torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    print(f"[data] {spec['task']['reference']['trace_recording']}: {T} frames x {box['X'].shape[1]} elements, "
          f"inputs {n_in}, norm (mean, sd, one-step sd) {tuple(round(v, 5) for v in box['norm'])}, "
          f"{len(origins)} origins; stages {stages}")
    print(f"[fit] {len(params)} tensor(s), {n_par} values; lr {lr} ({'constant' if sch is None else 'cosine'}), "
          f"batch {batch}, {iters} updates")
    # 16 fixed origins, the live evaluation; from frame 5 at the earliest, or the mean baseline's W = 6 reads X[-1..]
    # (the recording's end) for a 1-frame law (review, 2026-09-30)
    ev_o = np.linspace(max(int(origins[0]), 5), T - 34, 16).astype(int)
    rec_ = box["rec"]
    kid_ = TR.stimulus_keys(rec_["stimulus"], rec_["condition"])
    if box["split"] == "zapbench":            # the live evaluation on ZAPBench's VALIDATION windows, never its test
        vo = TR.split_origins(box["lab"], rec_["condition"], "val", n_in + w, 32)
        ev_o = vo[np.linspace(0, len(vo) - 1, 16).astype(int)]
    elif box["split"] == "recording":         # no validation part: the live evaluation on TRAINING windows, never
        vo = TR.split_origins(box["lab"], rec_["condition"], "train", n_in + w, 32)    # a held-out one
        ev_o = vo[np.linspace(0, len(vo) - 1, 16).astype(int)]
    # THE STIMULUS LOOKUP ON THOSE ORIGINS, once: the live figure draws its per-step skill beside the model's.
    look = (TR.stimulus_lookup_fit(box["X"], kid_, box["lab"] == 0) if box["lab"] is not None
            else TR.stimulus_lookup(box["X"], kid_))
    lk_mse = TR.mse_per_origin(lambda oo, h: look[oo + h], box["X"], torch.as_tensor(ev_o, device=box["X"].device)).mean(1)
    del look
    save_every = int(tr.get("save_every", 500))
    hist_p = os.path.join(out, "results", "history.jsonl")
    if not start_it:
        open(hist_p, "w").close()                  # a resume appends to the interrupted run's history
    for _ in range(start_it if sch is not None else 0):
        sch.step()
    log, last_ok, t_all, it = [], learn.snapshot(), time.time(), start_it
    skip = start_it
    for si, (K, n_it) in enumerate(stages):
        done_ = min(skip, n_it)
        skip -= done_
        for _ in range(n_it - done_):
            t0 = time.time()
            opt.zero_grad()
            parts, loss_v, finite = {}, 0.0, True
            for o in rng.choice(origins, size=batch, replace=False):
                pred = _trace_rollout(sims[K], learn, spec, box, int(o), K, device, True)
                p_o = {}
                loss_b = _objective(spec, learn, _trace_evidence(pred, box, int(o)), it, p_o) / batch
                for k2, v in p_o.items():
                    parts[k2] = parts.get(k2, 0.0) + v / batch
                if torch.isfinite(loss_b):
                    loss_b.backward()
                else:
                    finite = False
                loss_v += float(loss_b.detach())
            if not finite and guard == "restore_and_halve":
                learn.restore(last_ok)
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                print(f"  it {it:6d} loss not finite -- restored the last finite values, step sizes halved", flush=True)
                it += 1
                continue
            gn_ = torch.nn.utils.clip_grad_norm_(params, clip)
            if not torch.isfinite(gn_) and guard == "restore_and_halve":
                # A FINITE LOSS WITH A NON-FINITE GRADIENT (Cedric, 2026-10-08: 20.4, 23.3 and 23.9 each stopped near
                # update 48,000): clipping a NaN norm scales every gradient to NaN, the step writes NaN into the
                # learnables, and the next rollout seeds a NaN state -- which the engine's tick-0 write check then
                # reported as "state_diffuse wrote the integrated state directly". The step is skipped as a
                # non-finite loss is.
                learn.restore(last_ok)
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                print(f"  it {it:6d} gradient not finite -- step skipped, last finite values kept, step sizes halved",
                      flush=True)
                it += 1
                continue
            opt.step()
            if sch is not None:
                sch.step()
            learn.clamp_()
            last_ok = learn.snapshot()
            it += 1
            row = {"it": it, "stage": si, "horizon": K, "loss": loss_v, "seconds": time.time() - t0, **parts}
            if it % save_every == 0 or it == iters:
                m, base = _trace_eval(spec, learn, box, device, ev_o)
                bm = base.mean(2).min(0)
                sk = 1 - m.mean(1) / bm
                lk = 1 - lk_mse / bm
                mh = m.mean(1)                     # MSE per step ahead over the 16 origins (Cedric: MSE, no skill)
                row.update({"eval_skill_short": float(sk[0:3].mean()), "eval_skill_long": float(sk[15:32].mean()),
                            "eval_skill": sk.tolist(), "eval_mse_short": float(mh[0:3].mean()),
                            "eval_mse_long": float(mh[15:32].mean()), "eval_mse": mh.tolist(),
                            "eval_mse_mean": bm.tolist()})
                torch.save({"fitted": learn.snapshot(), "model": spec["model"], "task": spec["task"],
                            "learnable": spec["learnable"], "it": it, "select": "interim"},
                           os.path.join(out, "models", "best.pt"))
                _trace_live(out, log + [row], mh, bm, lk_mse)
                print(f"  it {it:6d} stage {si} K {K}  loss {loss_v:.6f}  eval skill short {row['eval_skill_short']:+.3f} "
                      f"long {row['eval_skill_long']:+.3f}  {row['seconds']:.2f} s/it", flush=True)
            log.append(row)
            with open(hist_p, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        # one checkpoint per curriculum stage, named by its horizon K (Cedric, 2026-09-30): the full test can then
        # be run on any stage, to see whether training stopped at a given horizon forecasts better than the last
        torch.save({"fitted": learn.snapshot(), "model": spec["model"], "task": spec["task"],
                    "learnable": spec["learnable"], "it": it, "stage": si, "horizon": K},
                   os.path.join(out, "models", f"stage_{K:02d}.pt"))
    # THE LAST CHECKPOINT IS KEPT (`select: last`, Cedric): no frame is held out to pick one on.
    torch.save({"fitted": learn.snapshot(), "model": spec["model"], "task": spec["task"],
                "learnable": spec["learnable"], "it": it, "select": "last"}, os.path.join(out, "models", "best.pt"))
    rep = {"name": spec["name"], "model": spec["model"], "trace_recording": spec["task"]["reference"]["trace_recording"],
           "stages": stages, "n_params": n_par, "iters": iters, "batch": batch, "lr": lr, "select": "last",
           "norm": box["norm"], "seconds": round(time.time() - t_all, 1),
           "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 2 ** 30
                           if str(device).startswith("cuda") else None)}
    json.dump(rep, open(os.path.join(out, "results", "report.json"), "w"), indent=2)
    print(f"[done] {iters} updates in {rep['seconds'] / 60:.1f} min")
    return out


def _trace_live(out, log, mh, bm, lk=None):
    """`results/live.png`, rewritten at every evaluation so the watcher sees the run while it trains (black, as every
    exp17 figure), in MSE (Cedric, 2026-10-01: no skill): left, the training loss per update, a line where each
    curriculum stage begins; middle, the evaluation's MSE short (h 1-3) and long (h 16-32) over the updates (16 fixed
    origins), the mean baseline's and ZAPBench's best as lines; right, the LAST SAVED model's MSE per step ahead beside
    the mean baseline, the stimulus lookup and ZAPBench's best (its published skill applied to this mean baseline)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    pub_p = graphs_data_path("zebrafish", "zapbench_published.json")
    bs = np.asarray(json.load(open(pub_p))["best_ctx4"]["skill"]) if os.path.exists(pub_p) else None
    mh, bm = np.asarray(mh) * 1e3, np.asarray(bm) * 1e3
    with plt.style.context("dark_background"):
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.3), facecolor="black")
        a.plot([r["it"] for r in log], [r["loss"] for r in log], lw=0.6, color="0.75")
        a.set_yscale("log")
        a.set_xlabel("training updates (Adam steps, 4 origins each)")
        a.set_ylabel("training loss")
        for k in sorted({r["horizon"] for r in log}):      # where each curriculum stage starts
            a.axvline(next(r["it"] for r in log if r["horizon"] == k), color="0.25", lw=0.5)
        ev = [r for r in log if "eval_mse_short" in r]
        b.plot([r["it"] for r in ev], [1e3 * r["eval_mse_short"] for r in ev], "o-", ms=3, color="#e07b39",
               label="short (h 1-3)")
        b.plot([r["it"] for r in ev], [1e3 * r["eval_mse_long"] for r in ev], "o-", ms=3, color="#7aa6ff",
               label="long (h 16-32)")
        for v, col, lab in ((bm[0:3].mean(), "#e07b39", "mean baseline, short"), (bm[15:32].mean(), "#7aa6ff",
                                                                                   "mean baseline, long")):
            b.axhline(v, color=col, lw=0.8, ls="--", label=lab)
        if bs is not None:
            zb = (1 - bs[:len(bm)]) * bm
            for v, col, lab in ((zb[0:3].mean(), "#e07b39", "ZAPBench best, short"), (zb[15:32].mean(), "#7aa6ff",
                                                                                    "ZAPBench best, long")):
                b.axhline(v, color=col, lw=1.2, ls=":", label=lab)
        b.set_xlabel("training updates")
        b.set_ylabel("MSE, $10^{-3}$ dF/F$^2$ (16 origins)")
        b.legend(frameon=False, fontsize=7)
        hh = np.arange(1, len(mh) + 1)
        c.plot(hh, bm, color="0.75", lw=1.4, ls="--", label="mean baseline (best W)")
        if lk is not None:
            c.plot(hh, np.asarray(lk) * 1e3, color="#c44e52", lw=1.2, label="stimulus lookup")
        if bs is not None:
            c.plot(hh, (1 - bs[:len(hh)]) * bm, color="#7aa6ff", lw=1.6, ls="-.",
                   label="ZAPBench best (U-Net, ctx 4)")
        c.plot(hh, mh, color="white", lw=2.2, label=f"this law, update {ev[-1]['it'] if ev else 0}")
        c.set_xlabel("steps ahead h (0.914 s each)")
        c.set_ylabel("MSE, $10^{-3}$ dF/F$^2$")
        c.legend(frameon=False, fontsize=7, loc="upper left")
        for ax in (a, b, c):
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "results", "live.png"), dpi=110, facecolor="black")
        plt.close(fig)


def _test_trace(spec, device="cpu", root=None):
    """THE SCORE. (1) Forecasts of 32 steps from origins every `test_stride` frames over the whole recording: MSE
    per step against the best mean baseline, persistence and the stimulus lookup, per condition and in the grand
    average, as skill (`trace_recording.skills`). (2) ONE FREE ROLLOUT of the whole recording from its first
    `inputs` frames: R^2 per frame over elements, raw and denoised. Written to results/<name>_test.json and
    results/<name>_free.npz; `-o plot` draws them."""
    from plexus.tasks import trace_recording as TR
    engine.quiet(True)
    out = out_dir(spec, root)
    ckn = spec.get("_checkpoint")                 # a per-stage checkpoint (`--checkpoint stage_05`), or the run's own
    stem = f"{spec['name']}_{ckn}" if ckn else spec["name"]
    ck = torch.load(os.path.join(out, "models", f"{ckn or 'best'}.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = _trace_setup(spec, device)
    X, T, rec = box["X"], box["T"], box["rec"]
    H_ = TR.H_MAX
    stride = int(spec["task"]["reference"].get("test_stride", 8))
    # one grid for every arm (from frame 5, the 6-frame base's first origin); a longer history only drops the
    # origins without enough frames behind them, so arms of different history are scored on the same frames
    origins = np.arange(5, T - H_ - 1, stride)
    origins = origins[origins >= box["n_in"] - 1 + _warmup(spec)]
    zb = box["split"] == "zapbench"
    if zb:                                    # ZAPBench's test windows, all of them (stride 1), and the held-out TAXIS
        o_test = TR.split_origins(box["lab"], rec["condition"], "test", box["n_in"], H_)
        o_hold = TR.split_origins(box["lab"], rec["condition"], "holdout", box["n_in"], H_)
        origins, stride = np.concatenate([o_test, o_hold]), 1
        # ZAPBench's windows unchanged under a warm-up: its frames are read before each window's origin (inputs, never
        # scored), which every test window has room for (the test segments follow 80 % of each condition)
        if origins.min() < box["n_in"] - 1 + _warmup(spec):
            raise ValueError(f"a ZAPBench window at frame {origins.min()} has no room for the warm-up")
    elif box["split"] == "recording":         # every window whose 32 forecast frames lie in a held-out part
        origins, stride = TR.split_origins(box["lab"], rec["condition"], "test", box["n_in"] + _warmup(spec), H_), 1
        if not len(origins):
            raise ValueError(f"the recording's held-out parts hold no window of {H_} frames")
    t0 = time.time()
    m, base = _trace_eval(spec, learn, box, device, origins, H_)
    ot = torch.as_tensor(origins, device=X.device)
    kid = TR.stimulus_keys(rec["stimulus"], rec["condition"])
    look = TR.stimulus_lookup_fit(X, kid, box["lab"] == 0) if box["lab"] is not None else TR.stimulus_lookup(X, kid)
    lk = TR.mse_per_origin(lambda oo, h: look[oo + h], X, ot)
    oc, nc = rec["condition"][origins], len(rec["names"])
    # THE GRAND AVERAGE: every condition, or with ZAPBench's split its 8 training conditions' test windows (TAXIS,
    # held out, is reported on its own) -- ZAPBench's own average
    gsel = [c for c in range(nc) if c != TR.ZB_HOLDOUT] if zb else list(range(nc))
    base_c = TR.by_condition(base, oc, nc)                                      # [C, W, h]
    best_w = base_c[gsel].mean(0).argmin(0)                                     # per h, on the grand average
    mean_c = base_c[:, best_w, np.arange(H_)]                                   # [C, h]
    model_c, look_c = TR.by_condition(m, oc, nc), TR.by_condition(lk, oc, nc)
    pers_c = base_c[:, 0, :]

    def _sk(mc, bc):                          # per-condition fields over all conditions, the grand ones over gsel
        r, g = TR.skills(mc, bc), TR.skills(mc[gsel], bc[gsel])
        r.update({k: g[k] for k in ("skill", "skill_short", "skill_long", "n_conditions_long_positive")})
        return r
    res = _sk(model_c, mean_c)
    sk_look = _sk(look_c, mean_c)
    zbj = {}
    if zb:
        # ZAPBENCH'S OWN SKILL: against its mean baseline at context 4, the mean of the last W = 4 frames (our scorer
        # reproduces its published mean-baseline MSE on these very windows, data/baselines.json scorer_check), so
        # this number stands beside its published 0.206 / 0.438 like for like
        w4 = base_c[:, 3, :]
        z = TR.skills(model_c[gsel], w4[gsel])
        zh = TR.skills(model_c[[TR.ZB_HOLDOUT]], w4[[TR.ZB_HOLDOUT]])
        zbj = {"split": "zapbench", "n_test_windows": int(len(o_test)), "n_holdout_windows": int(len(o_hold)),
               "skill_short_zapbench": z["skill_short"], "skill_long_zapbench": z["skill_long"],
               "skill_zapbench": z["skill"], "taxis_skill_short_zapbench": zh["skill_short"],
               "taxis_skill_long_zapbench": zh["skill_long"],
               "lookup_skill_long_zapbench": TR.skills(look_c[gsel], w4[gsel])["skill_long"]}
        print(f"[test] ZAPBench split: {len(o_test)} test + {len(o_hold)} TAXIS windows; skill vs its mean (W 4): "
              f"short {z['skill_short']:+.4f} long {z['skill_long']:+.4f} (published best 0.206 / 0.438); "
              f"TAXIS short {zh['skill_short']:+.4f} long {zh['skill_long']:+.4f}")
    ident = float(np.max(np.abs(model_c.mean(0) - pers_c.mean(0)) / pers_c.mean(0)))
    D1 = float(((X[1:] - X[:-1]) ** 2).mean())
    D = [float(((X[h:] - X[:-h]) ** 2).mean()) for h in (1, 2, 3)]
    sigma2 = max(float(np.polyfit([1, 2, 3], D, 1)[1]), 0.0) / 2
    print(f"[test] {len(origins)} origins (stride {stride}), {time.time() - t0:.0f} s: skill short "
          f"{res['skill_short']:+.4f} long {res['skill_long']:+.4f}; lookup long {sk_look['skill_long']:+.4f}; "
          f"conditions long > 0: {res['n_conditions_long_positive']}/{len(gsel)}; identity gap {ident:.2e}")
    # THE FREE ROLLOUT of the whole recording, and the movie sampled from it
    t1 = time.time()
    free, r2r, r2d, o0, n, finite, n_dead = _trace_free(spec, learn, box, device, out, stem)
    print(f"[test] free rollout of {n} frames in {time.time() - t1:.0f} s: R2 raw {free['r2_raw']:+.3f} +- "
          f"{free['r2_raw_sd']:.3f}, denoised {free['r2_denoised']:+.3f} +- {free['r2_denoised_sd']:.3f}, "
          f"finite {bool(finite)}")
    out_j = {"name": spec["name"], "mode": "trace", "origins": int(len(origins)), "stride": stride,
             "names": rec["names"], "offsets": rec["offsets"].tolist(), "frame_s": _frame_s(rec),
             **res, "identity_gap": ident, "best_W": (best_w + 1).tolist(),
             "mse_model": model_c[gsel].mean(0).tolist(), "mse_mean": mean_c[gsel].mean(0).tolist(),
             "mse_persistence": pers_c[gsel].mean(0).tolist(), "mse_lookup": look_c[gsel].mean(0).tolist(),
             "mse_model_by_condition": model_c.tolist(), "mse_mean_by_condition": mean_c.tolist(),
             "lookup_skill_short": sk_look["skill_short"], "lookup_skill_long": sk_look["skill_long"],
             "skill_long_minus_lookup": res["skill_long"] - sk_look["skill_long"],
             "noise_sigma2": sigma2, "one_step_msd": D1, "free": free,
             "free_t_s": ((o0 + 1 + np.arange(n)) * _frame_s(rec)).tolist(), "free_r2_raw": r2r.tolist(),
             "free_r2_denoised": r2d.tolist(), "free_silenced": n_dead.tolist(), "it": ck.get("it"), "select": ck.get("select", "last"), **zbj}
    out_j["checkpoint"] = ckn or "best"
    out_j["trace_recording"] = spec["task"]["reference"]["trace_recording"]   # which recording these scores are on
    out_j["rollouts"] = _trace_variants(spec, learn, box, device, out, stem)
    json.dump(out_j, open(os.path.join(out, "results", f"{stem}_test.json"), "w"))
    return out_j


def _trace_variants(spec, learn, box, device, out, stem, names=None):
    """Every `task.rollouts` variant but the nominal one (made above): {name: its free-rollout summary}; each writes
    results/<stem>_<name>_free.npz and _movie.npz, which `-o analyse` renders as results/movie_<name>.mp4."""
    res = {}
    for v in spec["task"].get("rollouts") or []:
        if v["name"] == "nominal" or (names and v["name"] not in names):
            continue
        t1 = time.time()
        fv = _trace_free(spec, learn, box, device, out, f"{stem}_{v['name']}", variant=v)[0]
        res[v["name"]] = {**fv, "spec": v}
        print(f"[test] rollout `{v['name']}` ({time.time() - t1:.0f} s): R2 denoised {fv['r2_denoised']:+.3f} over "
              f"{box['X'].shape[1] - fv['n_clamped']:,} free elements ({fv['n_clamped']:,} given), brain-mean SD "
              f"{fv['brain_mean_sd_pred']:.4f} against {fv['brain_mean_sd_obs']:.4f} recorded, r {fv['brain_mean_r']:.2f}, "
              f"{fv['silenced']} silenced")
    return res


def trace_rollouts(name, device="cuda:0", root=None, names=None):
    """A landed run's `task.rollouts` variants made from its best model without re-scoring the test windows (as
    `trace_movie`), the test json's `rollouts` updated, and each variant's movie rendered."""
    engine.quiet(True)
    spec = load(name)
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = _trace_setup(spec, device)
    res = _trace_variants(spec, learn, box, device, out, spec["name"], names)
    tj = os.path.join(out, "results", f"{spec['name']}_test.json")
    if os.path.exists(tj):
        j = json.load(open(tj))
        j["rollouts"] = {**j.get("rollouts", {}), **res}
        json.dump(j, open(tj, "w"))
    del box
    for nm in res:
        _render_variant(spec, out, spec["name"], nm)
    return res


def _render_variant(spec, out, stem, nm, res_dir=None, variant=None, path=None):
    """results/movie_<nm>.mp4: a rollout variant's movie, recorded LEFT and the variant RIGHT. `res_dir` (default
    results/) holds the variant's movie npz; `variant` is its entry (default: the spec's `task.rollouts` entry called
    `nm`); `path` the mp4 to write (default <res_dir>/movie_<nm>.mp4)."""
    from plexus.tasks import trace_recording as TR
    rd = res_dir or os.path.join(out, "results")
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    mv = np.load(os.path.join(rd, f"{stem}_{nm}_movie.npz"))
    fr = mv["frames"]
    v = variant or next(r for r in spec["task"]["rollouts"] if r["name"] == nm)
    rec_ref = str(spec["task"]["reference"].get("trace_recording"))
    what = ", ".join(([f"{', '.join(v['zero'])} = 0"] if v.get("zero") else []) + (["no stimulus"] if v.get("drive") == "off"
                     else []) + ([f"{int(mv['clamped'].sum()):,} given"] if "clamp" in v else [])
                     + ([f"|w| < {', '.join(f'{t:g}' for t in v['prune']['thresholds'].values())} cut"] if "prune" in v else [])
                     + (["inputs cut"] if "zero_input" in v else []) + (["a pulse"] if "pulse" in v else [])
                     + ([f"clock {v['clock']}"] if v.get("clock") else []))
    TR.render_movie(rec["dff"][fr], mv["pred"].astype(np.float32), fr, rec.get("pos_view", rec["pos_um"]), mv["r2_raw"], mv["r2_denoised"],
                    _frame_s(rec), rec["condition"][fr], rec["names"], path or os.path.join(rd, f"movie_{nm}.mp4"),
                    r2_t=mv["r2_t"], r2_raw_all=mv["r2_raw_all"], r2_den_all=mv["r2_denoised_all"],
                    silenced_all=mv["silenced_all"], mean_obs_all=mv["mean_obs_all"], mean_pred_all=mv["mean_pred_all"],
                    cond_all=mv["cond_all"], split_all=mv["split_all"],
                    split_name=spec["task"]["reference"].get("split", "all"), law=f"{_law_name(spec)}: {nm} ({what})",
                    rec_name=("destriped" if "destripe" in rec_ref else rec_ref),
                    metric=spec["task"].get("movie_metric", "cells"))


SILENCE_RANGE = (-1.0, 3.0)   # dF/F; the free rollout freezes an element outside it (finding 21)


def _trace_free(spec, learn, box, device, out, stem, n_movie=800, variant=None, res_dir=None, traces_path=None,
                n_frames=None):
    """ONE FREE ROLLOUT of the whole recording from the law's first `inputs` frames: R^2 per frame over the elements,
    raw and against the denoised recording (results/<stem>_free.npz), and THE MOVIE sampled from this same rollout
    (Cedric, 2026-10-02: the whole 2 h, not a 200-frame restart): `n_movie` frames evenly spaced over it, the learned
    frame and the recorded one (results/<stem>_movie.npz). Returns (free summary, r2 raw, r2 denoised, first origin,
    frames, finite). `variant`, one entry of `task.rollouts` (None: the nominal rollout): its `zero:` learnables at 0,
    its `drive: off`, its `clamp:` elements given their recorded frames and left out of R2 and the brain mean; the
    graph phase's kinds too (`prune:`, `zero_input:`, `pulse:`, `clock:`), each recorded in the summary. `res_dir`:
    where the npz go (default results/); `traces_path`: an .npy written with EVERY predicted frame, float16 [T, N],
    NaN before the first free frame and for silenced elements (the every-frame traces of the graph phase); `n_frames`:
    only that many free frames from the start (a short rollout: the graph phase's pulses), default the whole recording."""
    from plexus.tasks import trace_recording as TR
    X, T = box["X"], box["T"]
    variant = variant or {}
    rd = res_dir or os.path.join(out, "results")
    os.makedirs(rd, exist_ok=True)
    clamp = None
    if "clamp" in variant:
        clamp = torch.as_tensor(_clamp_mask(variant["clamp"], box["rec"]["pos_um"]), device=X.device)
        if not bool(clamp.any()) or bool(clamp.all()):
            raise ValueError(f"rollout `{variant.get('name')}`: its clamp holds {int(clamp.sum())} of {len(clamp)} elements; "
                             "a clamp needs some elements given and some free")
    w = _warmup(spec)
    o0 = box["n_in"] - 1 + w                # the free rollout's first origin, after its warm-up
    n = T - 1 - o0 if n_frames is None else max(1, min(int(n_frames), T - 1 - o0))
    sim = _model(spec, train=False, n_frames=w + n - 1)
    Xd = TR.denoise(X)
    r2r, r2d, finite = np.zeros(n), np.zeros(n), True
    keep = set(np.unique(np.linspace(0, n - 1, min(n_movie, n)).round().astype(int)).tolist())
    rec_v, perturb, traces = {}, None, None
    if "pulse" in variant:
        perturb, rec_v["pulse"] = _pulse_fn(variant["pulse"], box, _frame_s(box["rec"]))
    if variant.get("clock"):
        rec_v["clock"] = dict(variant["clock"])
    if traces_path:
        traces = np.lib.format.open_memmap(traces_path, mode="w+", dtype=np.float16, shape=(T, X.shape[1]))
        traces[:o0 + 1] = np.nan
    mv = []
    # RUNAWAY ELEMENTS SILENCED (Cedric, 2026-10-02; finding 21): an element whose value leaves SILENCE_RANGE (dF/F; the
    # recording is clipped to [-0.25, 1.5]) or is not finite is frozen at its recorded mean for the rest of the
    # rollout -- its neighbours read a constant, not an explosion -- and left out of R^2; `silenced` counts them per
    # frame. One explicit-Euler element past its stability limit otherwise takes the R^2 over all 100,759 to -1e15.
    dead = torch.zeros(X.shape[1], dtype=torch.bool, device=X.device)
    fill = X.mean(0)
    n_dead = np.zeros(n, dtype=np.int64)
    mean_obs, mean_pred = np.zeros(n, np.float32), np.zeros(n, np.float32)      # brain-mean dF/F per frame
    lo, hi = SILENCE_RANGE

    def silence(blk):
        v = blk.reshape(-1, blk.shape[-1])
        bad = ~torch.isfinite(v).all(1) | (v < lo).any(1) | (v > hi).any(1)
        dead.logical_or_(bad)
        if not bool(dead.any()):
            return None
        v = v.clone()
        v[dead] = fill[dead][:, None].to(v.dtype)
        return v.reshape(blk.shape)

    def r2(p, x, alive):
        num = ((p[alive] - x[alive]) ** 2).sum()
        den = ((x[alive] - x[alive].mean()) ** 2).sum()
        return float(1 - num / den)

    def on_pred(k, p):
        nonlocal finite
        tt = o0 + k + 1
        alive = ~dead if clamp is None else ~dead & ~clamp
        n_dead[k] = int(dead.sum())
        finite = finite and bool(torch.isfinite(p[alive]).all())
        r2r[k] = r2(p, X[tt], alive)
        r2d[k] = r2(p, Xd[tt], alive)
        mean_obs[k], mean_pred[k] = float(X[tt][alive].mean()), float(p[alive].mean())
        if k in keep or traces is not None:
            q = p.clone()
            q[dead] = float("nan")                  # drawn blank in the movie
            qh = q.half().cpu().numpy()
            if k in keep:
                mv.append(qh)
            if traces is not None:
                traces[tt] = qh
    with torch.no_grad(), _zeroed(learn, spec, variant.get("zero", [])), contextlib.ExitStack() as stack:
        if "prune" in variant:
            rec_v["prune"] = stack.enter_context(_pruned(learn, spec, variant["prune"]["thresholds"]))
        if "zero_input" in variant:
            rec_v["zero_input"] = stack.enter_context(_input_zeroed(learn, spec, box, variant["zero_input"]))
        _trace_rollout_stream(sim, learn, spec, box, o0, n, device, on_pred, silence=silence, w=w,
                              drive_on=variant.get("drive", "on") == "on", clamp=clamp, perturb=perturb,
                              clock=variant.get("clock"))
    if traces is not None:
        traces.flush()
        del traces
    free = {"r2_raw": float(np.nanmean(r2r)), "r2_raw_sd": float(np.nanstd(r2r)), "r2_denoised": float(np.nanmean(r2d)),
            "r2_denoised_sd": float(np.nanstd(r2d)), "finite": 1.0 if finite else 0.0, "origin": o0, "frames": n,
            "silenced": int(n_dead[-1]), "silenced_first_t_s": (float((o0 + 1 + int(np.argmax(n_dead > 0))) * _frame_s(box["rec"]))
                                                                 if n_dead[-1] else None),
            "silence_range": list(SILENCE_RANGE),
            # the brain-mean dF/F over the free (not given, not silenced) elements: its SD over the rollout, recorded and
            # learned, and their correlation -- the network test of exp17 (finding 26)
            "brain_mean_sd_obs": float(mean_obs.std()), "brain_mean_sd_pred": float(mean_pred.std()),
            "brain_mean_r": float(np.corrcoef(mean_obs, mean_pred)[0, 1]) if mean_pred.std() > 0 else float("nan"),
            # THE BRAIN-MEAN METRICS (Cedric, 2026-10-03: "better than the R2 on the volume, it shows the learning of the
            # network dynamics"): R2 = 1 - sum_t (pred_t - obs_t)^2 / sum_t (obs_t - mean obs)^2 and the RMSE in dF/F,
            # between the recorded and learned brain-mean traces over every free frame
            **_brain_mean_metrics(mean_obs, mean_pred),
            "variant": variant.get("name", "nominal"), "n_clamped": int(clamp.sum()) if clamp is not None else 0,
            **rec_v}
    np.savez_compressed(os.path.join(rd, f"{stem}_free.npz"), r2_raw=r2r, r2_denoised=r2d, silenced=n_dead)
    ks = np.array(sorted(keep))
    fr_m = o0 + 1 + ks
    np.savez_compressed(os.path.join(rd, f"{stem}_movie.npz"), frames=fr_m, pred=np.stack(mv),
                        r2_raw=r2r[ks], r2_denoised=r2d[ks], r2_t=o0 + 1 + np.arange(n), r2_raw_all=r2r,
                        r2_denoised_all=r2d, silenced=n_dead[ks], silenced_all=n_dead, kind="free",
                        mean_obs_all=mean_obs, mean_pred_all=mean_pred, variant=variant.get("name", "nominal"),
                        clamped=(clamp.cpu().numpy() if clamp is not None else np.zeros(X.shape[1], bool)),
                        cond_all=np.asarray(box["rec"]["condition"])[o0 + 1 + np.arange(n)],
                        split_all=(np.asarray(box["lab"])[o0 + 1 + np.arange(n)] if box.get("lab") is not None
                                   else np.zeros(n, np.int64)))
    return free, r2r, r2d, o0, n, finite, n_dead


def trace_movie(name, device="cuda:0", root=None):
    """Rebuild one landed run's free rollout and movie frames from its best model, without re-scoring the test
    windows (exp17's movies over the whole recording, 2026-10-02); then `-o analyse` renders them."""
    engine.quiet(True)
    spec = load(name)
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = _trace_setup(spec, device)
    t1 = time.time()
    free, r2r, r2d, o0, n, finite, n_dead = _trace_free(spec, learn, box, device, out, spec["name"])
    tj = os.path.join(out, "results", f"{spec['name']}_test.json")
    if os.path.exists(tj):                  # the scores' free-rollout fields follow the rollout just made
        j = json.load(open(tj))
        j.update({"free": free, "free_r2_raw": r2r.tolist(), "free_r2_denoised": r2d.tolist(),
                  "free_silenced": n_dead.tolist(), "free_t_s": ((o0 + 1 + np.arange(n)) * _frame_s(box["rec"])).tolist()})
        json.dump(j, open(tj, "w"))
    print(f"[movie] {name}: free rollout of {free['frames']} frames in {time.time() - t1:.0f} s, R2 denoised "
          f"{free['r2_denoised']:+.3f}, {free['silenced']} element(s) silenced")
    return free


def _trace_rollout_stream(sim, learn, spec, box, o, n, device, on_pred, silence=None, w=0, drive_on=True, clamp=None,
                          perturb=None, clock=None):
    """A long rollout (the free one), each frame handed to `on_pred(k, frame)` and dropped, not stacked. `silence`,
    when given, edits the observed block before each frame is read (`silence(block) -> block or None`): the free
    rollout freezes its runaway elements with it. `drive_on` False writes the drive as 0 every frame (a rollout
    variant with no stimulus); `clamp` (bool [N]) overwrites those elements with their RECORDED frames every tick,
    before the frame is read and before the next step reads them (a rollout variant with given leaders).
    `perturb(k, block) -> block or None` (k the free step) edits the observed block after the clamp and before the
    silence (a rollout variant with a pulse); `clock` ({shift_frames: n} or {freeze_at: f}) moves the frame clock the
    law's functions of absolute time read, while the stimulus keeps its true time (the time-memorisation test)."""
    t = spec["task"]
    obs, drv, X, S, T = t["observe"], t.get("drive"), box["X"], box["S"], box["T"]
    o = o - w                                  # `w` warm-up ticks of recorded frames first, then n free ones
    x0 = torch.stack([X[o - j] for j in range(box["n_in"])], 1)
    clock = clock or {}

    def window(H, tc):
        _clock(H, int(clock["freeze_at"]) if "freeze_at" in clock else tc + int(clock.get("shift_frames", 0)), T)
        if drv is None:
            return
        lv = H.level(drv["set"])
        a, b = lv.state_schema[drv["block"]]
        idx = (torch.arange(drv["window"][0], drv["window"][1] + 1, device=X.device) + tc).clamp(0, T - 1)
        st = lv.state.clone()
        st[..., a:b] = S[idx].T if drive_on else 0.0
        lv.state = st

    def seeded(H):
        lv = H.level(obs["set"])
        a, b = lv.state_schema[obs["block"]]
        st = lv.state.clone()
        st[..., a:b] = x0
        lv.state = st
        window(H, o)
        learn.inject(H)

    def hook(H, tick):
        if tick - w >= n:
            return
        lv = H.level(obs["set"])
        a, b = lv.state_schema[obs["block"]]
        if tick < w:
            st = lv.state.clone()
            st[..., a:b] = torch.stack([X[o + tick + 1 - j] for j in range(box["n_in"])], 1)
            lv.state = st
            window(H, o + tick + 1)
            return
        if clamp is not None:                  # the given elements: their recorded frames, newest first
            st = lv.state.clone()
            st[clamp, a:b] = torch.stack([X[o + tick + 1 - j][clamp] for j in range(box["n_in"])], 1).to(st.dtype)
            lv.state = st
        if perturb is not None:                # the pulsed elements: held at their level while the pulse lasts
            blk = perturb(tick - w, lv.state[..., a:b])
            if blk is not None:
                st = lv.state.clone()
                st[..., a:b] = blk
                lv.state = st
        if silence is not None:
            blk = silence(lv.state[..., a:b])
            if blk is not None:
                st = lv.state.clone()
                st[..., a:b] = blk
                lv.state = st
        on_pred(tick - w, lv.state[..., a].reshape(-1))
        window(H, o + tick + 1)

    def ready(H):
        learn.ready(H)
        for op in H.operators:
            if getattr(op, "NORM_FROM_REFERENCE", False):
                op.norm = box["norm"]
        if hasattr(H, "_frame_clock"):
            _clock(H, *H._frame_clock)
        for op in H.operators:             # a latent state inferred from the recorded frames (calcium_indicator)
            if hasattr(op, "init_latent"):
                op.init_latent(H)
    engine.run(sim, device=device, progress=False, grad=False, on_frame=hook, on_seeded=seeded, on_ready=ready)


def _law_name(spec) -> str:
    """The movie's label for the learned law: the `model:` of the spec's state_diffuse line."""
    import re
    try:
        txt = open(spec["model"]).read()
    except OSError:
        return "learned law"
    m = re.search(r"op:\s*state_diffuse,\s*model:\s*(\w+)", txt)
    ind = " + calcium indicator" if re.search(r"op:\s*calcium_indicator", txt) else ""
    model = m.group(1) if m else ""
    if model == "neuron_graph" and not any(str(e.get("param", "")).startswith("W_") for e in spec.get("learnable", [])):
        return "known ODE, no network" + ind          # no edge weight learned: W = 0 from the start (exp17 / exp20 *_now)
    return {"graphcast": "GraphCast law", "connectome": "connectome law", "known_ode": "known ODE on the mesh",
            "neuron_graph": "known ODE on the neuron graph"}.get(model, "learned law") + ind


def _analyse_trace(spec, device="cpu", root=None):
    """The figures of a trace run: results/<name>_test.png, the prediction against the baselines
    (`trace_recording.render_curves`), and results/movie.mp4, recorded LEFT and learned RIGHT over the free rollout
    (`render_movie`), with the embedding's insets when the law learns one; results/<name>_clusters.json, the
    embedding's clusters against the neurons' stimulus tuning (adjusted mutual information)."""
    from plexus.tasks import trace_recording as TR
    out = out_dir(spec, root)
    ckn = spec.get("_checkpoint")                 # a per-stage checkpoint: its own <name>_<checkpoint>_* files
    name = f"{spec['name']}_{ckn}" if ckn else spec["name"]
    res = json.load(open(os.path.join(out, "results", f"{name}_test.json")))
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    gates = {"short_full": 0.206, "long_full": 0.438, "published": rec.get("published")}
    hp = os.path.join(out, "results", "history.jsonl")
    hist = [json.loads(l) for l in open(hp)] if os.path.exists(hp) else None
    TR.render_curves(res, os.path.join(out, "results", f"{name}_test.png"), res["names"], gates, history=hist)
    ck = torch.load(os.path.join(out, "models", f"{ckn or 'best'}.pt"), weights_only=False, map_location="cpu")
    emb, labels, clus, src = None, None, {}, "embedding"
    N = rec["dff"].shape[1]
    for k, v in ck["fitted"].items():
        if "embedding" in k and v.ndim == 2 and v.shape[0] == N:
            emb = v.detach().float().numpy()
    fit = ck["fitted"]
    have = [b for b in ("tau", "rest", "gain", "input") if f"neuron.{b}" in fit]
    if emb is None and len(have) >= 2:
        # a law with no embedding (the known ODE): cluster each neuron's own learned constants instead (Cedric,
        # 2026-09-30) -- its rate 1/tau (softplus, per frame), rest V, gain on the mesh g and stimulus weights B (22).
        # Each block is z-scored and weighted to unit total variance, so B's 22 columns count as much as tau's one.
        blocks = [torch.nn.functional.softplus(fit[f"neuron.{b}"].float()) if b == "tau" else fit[f"neuron.{b}"].float()
                  for b in have]
        emb = torch.cat([((b - b.mean(0)) / b.std(0).clamp(min=1e-9)) / b.shape[1] ** 0.5 for b in blocks], 1).numpy()
        src = "parameters (" + ", ".join({"tau": "1/tau", "rest": "V", "gain": "g", "input": "B"}[b] for b in have) + ")"
    if emb is not None and emb.shape[1] > 0:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_mutual_info_score
        labels = KMeans(8, n_init=4, random_state=0).fit_predict(emb)
        X = torch.as_tensor(rec["dff"], device=device)
        kid = torch.as_tensor(TR.stimulus_keys(rec["stimulus"], rec["condition"]), device=device)
        nk = int(kid.max()) + 1
        prof = torch.zeros(nk, X.shape[1], device=device).index_add_(0, kid, X)
        prof = prof / torch.zeros(nk, device=device).index_add_(0, kid, torch.ones_like(kid, dtype=X.dtype))[:, None]
        prof = ((prof - prof.mean(0)) / prof.std(0).clamp(min=1e-6)).T.cpu().numpy()     # [N, keys]
        u, s, vt = np.linalg.svd(prof[::10], full_matrices=False)
        tun = KMeans(8, n_init=4, random_state=0).fit_predict(prof @ vt[:16].T)
        ami = float(adjusted_mutual_info_score(tun, labels))
        # the parameters' AMI is kept apart: B IS a stimulus tuning, so it would score G-embedding circularly
        clus = {"k": 8, "source": src, ("ami_tuning" if src == "embedding" else "ami_tuning_params"): ami,
                "sizes": np.bincount(labels, minlength=8).tolist()}
        del X
    json.dump(clus, open(os.path.join(out, "results", f"{name}_clusters.json"), "w"), indent=1)
    mv = np.load(os.path.join(out, "results", f"{name}_movie.npz"))
    frames = mv["frames"]
    TR.render_movie(rec["dff"][frames], mv["pred"].astype(np.float32), frames, rec.get("pos_view", rec["pos_um"]),
                    mv["r2_raw"], mv["r2_denoised"], _frame_s(rec), rec["condition"][frames], rec["names"],
                    os.path.join(out, "results", f"movie_{ckn}.mp4" if ckn else "movie.mp4"), emb=emb, labels=labels,
                    **({"r2_t": mv["r2_t"], "r2_raw_all": mv["r2_raw_all"], "r2_den_all": mv["r2_denoised_all"]}
                       if "r2_t" in mv else {}),
                    **({"silenced_all": mv["silenced_all"]} if "silenced_all" in mv else {}),
                    **({"mean_obs_all": mv["mean_obs_all"], "mean_pred_all": mv["mean_pred_all"],
                        "cond_all": mv["cond_all"], "split_all": mv["split_all"],
                        "split_name": spec["task"]["reference"].get("split", "all")} if "mean_obs_all" in mv else {}),
                    emb_name="embedding" if src == "embedding" else "neuron constants " + src[len("parameters "):],
                    inputs=(fit["neuron.input"].float().norm(dim=1).numpy() if "neuron.input" in fit else None),
                    law=_law_name(spec), metric=spec["task"].get("movie_metric", "cells"),
                    rec_name=("destriped" if "destripe" in str(spec["task"]["reference"].get("trace_recording")) else "ZAPBench"
                              if str(spec["task"]["reference"].get("trace_recording")).startswith("zapbench")
                              else str(spec["task"]["reference"].get("trace_recording")).replace("gutbrain_", "gut-brain ")))
    for v in spec["task"].get("rollouts") or []:          # the rollout variants' movies (`task.rollouts`)
        if v["name"] != "nominal" and os.path.exists(os.path.join(out, "results", f"{name}_{v['name']}_movie.npz")):
            _render_variant(spec, out, name, v["name"])
    print(f"[plot] {name}: {name}_test.png, {f'movie_{ckn}.mp4' if ckn else 'movie.mp4'} ({len(frames)} frames)" +
          (f", {src} AMI with tuning {clus.get('ami_tuning', clus.get('ami_tuning_params')):.3f}" if clus else ""))
    return clus


def graph(spec, device="cpu", root=None):
    """IS THE LEARNED GRAPH DOING SOMETHING, WHAT, AND HOW (exp17, Cedric 2026-10-10): the graph phase of a trace
    run, after `test` and `analyse` -- the graph silenced, cut and compared with trained controls, its linearised
    dynamics, its response to a pulse, its constants; scores in results/<stem>_graph.json, figures and movies under
    results/graph/. `plexus.tasks.graph_analysis.run` does the work."""
    if spec.get("_kind") != "trace_recording":
        raise ValueError("graph: the phase of a trace_recording task (a neuron-graph law fitted to a recording)")
    from plexus.tasks.graph_analysis import run as _run
    return _run(spec, device=device, root=root)


PHASES = {"train": train, "test": test, "analyse": analyse, "graph": graph}


def run_phases(name_or_path, phases, device="cpu", root=None, checkpoint=None, controls=None, regions=None):
    """`Plexus_Main.py -o train_test_analyse <name>` lands here. `checkpoint` (a trace run's test/analyse/graph only):
    score models/<checkpoint>.pt, e.g. `stage_05`, and write its results under <name>_<checkpoint>_*. `controls`
    ({no_w, mean_field, seed, random}: run names) are the graph phase's trained controls, the CLI twin of
    `task.graph.controls` (`--graph-controls`)."""
    spec = load(name_or_path)
    if checkpoint:
        if "train" in phases:
            raise ValueError("--checkpoint scores a saved checkpoint: test / plot / graph only, not train")
        spec["_checkpoint"] = str(checkpoint)
    if controls:
        bad = sorted(set(controls) - _KEYS["graph_controls"])
        if bad:
            raise ValueError(f"--graph-controls: {bad} are not controls; the controls are {sorted(_KEYS['graph_controls'])}")
        spec["_graph_controls"] = dict(controls)
    if regions:
        spec["_graph_regions"] = str(regions)          # the graph phase's atlas (--graph-regions)
    for ph in phases:
        PHASES[ph](spec, device=device, root=root)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--option", nargs="+", required=True,
                    help="phases (train / test / analyse) and training spec names or paths, any order")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default=None, help="overrides the log/ root")
    a = ap.parse_args(argv)
    phases = [o for o in a.option if o in PHASES] or ["train"]
    names = [o for o in a.option if o not in PHASES]
    if not names:
        ap.error("no training spec given")
    for n in names:
        for p in (glob.glob(n) if any(c in n for c in "*?[") else [n]):
            run_phases(p, phases, device=a.device, root=a.root)


if __name__ == "__main__":
    main()
