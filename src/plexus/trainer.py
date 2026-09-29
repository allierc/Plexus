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
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

import plexus.operators                                          # noqa: F401  self-registers
from plexus import engine
from plexus.paths import get_repo_root, log_path, training_spec
from plexus.schema import load as load_model
from plexus.tasks.generate import task_dir
from plexus.tasks.trainer import load_split, n_condition_cells, with_context


# ============================================================================== the declaration
_KEYS = {
    "top": {"name", "model", "learnable", "task", "training"},
    "learnable": {"block", "of", "with", "lr", "bounds"},
    "task": {"reference", "drive", "observe", "loss", "settle_s"},
    "reference": {"corpus", "n_train", "n_val", "n_test", "context"},
    "drive": {"set", "block"},
    "observe": {"set", "block", "channel", "unit"},
    "training": {"optimizer", "lr", "schedule", "clip", "epochs", "batch", "seed", "horizon",
                 "horizon_min", "snapshot_every", "guard"},
}
REPRESENTATIONS = ("tensor",)
OPTIMIZERS = ("adam",)
SCHEDULES = ("cosine", "none")
GUARDS = ("restore_and_halve", "none")


def _refuse_unread(where, d, allowed):
    extra = sorted(set(d) - allowed)
    if extra:
        raise ValueError(f"{where}: key(s) {extra} are read by nothing; the keys read here are "
                         f"{sorted(allowed)}. An unread key would be dropped and the run would "
                         f"train something other than what the spec says.")


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
    for i, e in enumerate(s["learnable"]):
        _refuse_unread(f"{path}: learnable[{i}]", e, _KEYS["learnable"])
        for k in ("block", "of"):
            if k not in e:
                raise ValueError(f"{path}: learnable[{i}] needs `{k}:`")
        if e["of"] not in sets:
            raise ValueError(f"{path}: learnable[{i}] fits `{e['of']}.{e['block']}`, but "
                             f"`{e['of']}` is not a set of {s['model']} ({sorted(sets)})")
        if e.get("with", "tensor") not in REPRESENTATIONS:
            raise ValueError(f"{path}: learnable[{i}] asks for representation {e['with']!r}; "
                             f"implemented: {list(REPRESENTATIONS)}.")
    t = s["task"]
    _refuse_unread(f"{path}: task", t, _KEYS["task"])
    for k in ("reference", "drive", "observe"):
        if k not in t:
            raise ValueError(f"{path}: task needs `{k}:`")
        _refuse_unread(f"{path}: task.{k}", t[k], _KEYS[k])
    if "corpus" not in t["reference"]:
        raise ValueError(f"{path}: task.reference needs `corpus:` -- the only reference "
                         f"implemented is a corpus under graphs_data/task/")
    for part in ("drive", "observe"):
        for k in ("set", "block"):
            if k not in t[part]:
                raise ValueError(f"{path}: task.{part} needs `{k}:`")
        if t[part]["set"] not in sets:
            raise ValueError(f"{path}: task.{part}.set {t[part]['set']!r} is not a set of the model")
    if t.get("loss", "mse") not in LOSSES:
        raise ValueError(f"{path}: task.loss {t['loss']!r}; registered: {sorted(LOSSES)}")
    tr = s["training"]
    _refuse_unread(f"{path}: training", tr, _KEYS["training"])
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

    @staticmethod
    def key(e) -> str:
        return f"{e['of']}.{e['block']}"

    def inject(self, H):
        for e in self.entries:
            lvl = H.level(e["of"])
            if e["block"] not in lvl.state_schema:
                raise ValueError(f"learnable `{self.key(e)}`: {e['of']!r} has no block "
                                 f"{e['block']!r} (it has {sorted(lvl.state_schema)})")
            a, b = lvl.state_schema[e["block"]]
            par = self.p.get(self.key(e))
            if par is None:
                par = self.p[self.key(e)] = nn.Parameter(lvl.state[:, a:b].detach().clone())
            st = lvl.state.clone()
            st[:, a:b] = par
            lvl.state = st

    def parameters(self) -> list:
        return [self.p[self.key(e)] for e in self.entries]

    def groups(self, lr) -> list:
        return [{"params": [self.p[self.key(e)]], "lr": float(e.get("lr", lr))}
                for e in self.entries]

    @torch.no_grad()
    def clamp_(self):
        for e in self.entries:
            if "bounds" in e:
                lo, hi = (float(v) for v in e["bounds"])
                self.p[self.key(e)].clamp_(lo, hi)

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
    """
    drv, obs = task["drive"], task["observe"]
    batch = int(u.shape[0]) if u.dim() == 3 else 1
    if u.dim() == 3 and batch == 1:                        # the engine runs unbatched at B = 1
        u = u[0]
    trace = []

    def hook(H, frame):
        write_drive(H, frame, u, drv["set"], drv["block"])
        trace.append(H.level(obs["set"]).get(obs["block"])[..., 0, :].clone())
        if watch is not None:
            watch(H)

    H, _ = engine.run(sim, device=device, progress=False, grad=grad, on_frame=hook,
                      batch=batch, on_seeded=learn.inject)
    if not trace:
        return H, None
    y = torch.stack(trace)
    return H, y.transpose(0, 1) if batch > 1 else y


def _mse(y, target, ch):
    """Squared error, aligned at frame 0 and truncated to the shorter series; batch-transparent.

    The mean is over frames AND trials. Indexed from the right, so [T, w] against [T, 1] and
    [B, T, w] against [B, T, 1] are the same call.
    """
    n = min(y.shape[-2], target.shape[-2])
    return ((y[..., :n, ch] - target[..., :n, 0]) ** 2).mean()


LOSSES = {"mse": _mse}


def _corpus(spec):
    return spec["task"]["reference"]["corpus"]


def _unit(spec):
    u = spec["task"]["observe"].get("unit")
    return (f" {u}", f" {u}^2") if u else ("", "")


def _data(spec, split, n_cond, device):
    U, Y, c = load_split(_corpus(spec), split, device)
    return with_context(U, c, n_cond), Y, c


def _model(spec, n_frames=None):
    sim = load_model(spec["model"])
    if sim.learnable:
        raise ValueError(f"{spec['model']} declares `learnable:` itself. What is learnable belongs "
                         f"to the training spec; the model file stays the forward description.")
    if n_frames is not None:
        sim.n_frames = int(n_frames)
    return sim


# ============================================================================== train
def train(spec, device="cpu", root=None):
    tr, task = spec["training"], spec["task"]
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
    loss_fn = LOSSES[task.get("loss", "mse")]
    ch = int(task["observe"].get("channel", 0))
    U1, U2 = _unit(spec)
    hist, best, best_state, t0 = [], np.inf, None, time.time()
    last_ok = learn.snapshot()

    for ep in range(epochs):
        h = horizon[min(len(horizon) - 1, int(len(horizon) * ep / max(epochs, 1)))]
        sim.n_frames = h
        perm = torch.randperm(n_tr)
        tot, n_step, n_bad = 0.0, 0, 0
        for k in range(0, n_tr - batch + 1, batch):
            idx = perm[k:k + batch]
            _, y = rollout(sim, learn, U[idx], task, device, grad=True)
            loss = loss_fn(y, Y[idx], ch)
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
                     "diverged_steps": n_bad})
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
    """The model, the trained learnables and the run folder -- everything a held-out rollout needs."""
    out = out_dir(spec, root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    return _model(spec), learn, ck, out


# ============================================================================== test
def test(spec, device="cpu", root=None):
    """Roll the kept checkpoint out on held-out trials. One number per trial, never only a mean."""
    engine.quiet(True)
    sim, learn, ck, out = _restore(spec, device, root)
    task, corpus = spec["task"], _corpus(spec)
    split = "test" if os.path.isdir(os.path.join(task_dir(corpus), "test")) else "train"
    U, Y, cond = _data(spec, split, int(ck.get("n_cond", 1)), device)
    n = min(int(task["reference"].get("n_test", 24)), U.shape[0])
    ch = int(task["observe"].get("channel", 0))
    with torch.no_grad():
        _, Yp = rollout(sim, learn, U[:n], task, device, grad=False)
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


# ============================================================================== entry points
PHASES = {"train": train, "test": test, "analyse": analyse}


def run_phases(name_or_path, phases, device="cpu", root=None):
    """`Plexus_Main.py -o train_test_analyse <name>` lands here."""
    spec = load(name_or_path)
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
