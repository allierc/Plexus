"""Train a Plexus SPEC against a task, through `engine.run(grad=True)`.

    python -m plexus.tasks.spec_trainer -o train test plot config/run/eye_rig_fit.yaml

The difference from `tasks.trainer`, which is a standalone torch module: this one fits the model
the spec describes, by running the engine. Nothing is transcribed, so nothing can drift -- the
thing being trained IS the thing `Plexus_Main.py -o generate` would run, operator for operator.

    forward spec     config/neural/ctrnn_eyeG_rig.yaml -- unchanged by training
    learnable:       which blocks are fitted (the run spec injects it)
    task             a corpus under graphs_data/task/
    run spec         names the three and the hyperparameters

WHY THIS IS SLOW AND WHAT IT COSTS, stated rather than discovered: `engine.run` carries no batch
axis, so a step is one trial. That is R0 of notes/campaigns/TRAINING_IN_PLEXUS.md and it is not
done. The consequence is a per-step cost of a full 480-frame rollout, so this trains on tens of
trials where `tasks.trainer` trains on hundreds. It is the correct model rather than a fast one,
and the two are checked against each other rather than trusted.

THE STIMULUS ENTERS THROUGH A BLOCK, NOT THROUGH A SPEC EDIT. The rig's `retina.signal` is written
each frame from the corpus by a callback the engine already supports (`on_frame`), so the spec
that describes the model is the same file whatever it is driven by -- which is what makes one
model runnable against several tasks.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import shutil
import time

import numpy as np
import torch
import yaml

import plexus.operators                                          # noqa: F401  self-registers
from plexus import engine
from plexus.schema import load
from plexus.tasks.generate import task_dir
from plexus.tasks.trainer import load_split, log_dir


def load_run(path) -> dict:
    with open(path) as f:
        r = yaml.safe_load(f)
    for k in ("name", "spec", "task", "learnable", "training"):
        if k not in r:
            raise ValueError(
                f"{path}: a spec-run needs `{k}:`. It names the forward SPEC, the TASK, what is "
                f"LEARNABLE and the hyperparameters -- the three files plus how they are paired.")
    r["_path"] = os.path.abspath(path)
    return r


def build(run, device="cpu"):
    """The Spec with its `learnable:` injected. The forward file itself is never edited."""
    sim = load(run["spec"])
    sim.learnable = list(run["learnable"])
    sim.n_frames = int(run.get("n_frames", sim.n_frames))
    return sim


def rollout(sim, u, drive_set, drive_block, read_set, read_block, device="cpu", grad=True):
    """One trial: write the stimulus into a block each frame, return the read-out trace.

    `on_frame` is the engine's own per-frame hook, so driving a run from outside costs no change
    to the engine and no edit to the spec.
    """
    trace = []

    def hook(H, frame):
        lvl = H.level(drive_set)
        a, b = lvl.state_schema[drive_block]
        t = min(frame, u.shape[0] - 1)
        st = lvl.state.clone()
        st[:, a:b] = u[t].expand(lvl.n, b - a)
        lvl.state = st
        trace.append(H.level(read_set).get(read_block)[0].clone())

    H, _ = engine.run(sim, device=device, progress=False, grad=grad, on_frame=hook)
    return H, torch.stack(trace) if trace else None


def _mse(y, target, ch):
    """Aligned at frame 0 and truncated to the shorter.

    THE ENGINE RECORDS ONE FRAME MORE THAN IT STEPS: `on_frame` fires before any dynamics, so
    trace[0] is the initial state and trace[1..n] are the n steps. The teacher's y[0] is likewise
    its output at t = 0, which for a strictly proper law is zero, so the two series START
    TOGETHER and the extra frame is on the end. Truncating is therefore correct and padding would
    not be; the alternative -- dropping trace[0] -- would shift the whole comparison by one step
    and report a lag the model does not have.
    """
    n = min(y.shape[0], target.shape[0])
    return ((y[:n, ch] - target[:n, 0]) ** 2).mean()


def train(run, root=None, device="cpu"):
    tr = run["training"]
    torch.manual_seed(int(tr.get("seed", 0)))
    out = log_dir(run["name"], root)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    shutil.copyfile(run["_path"], os.path.join(out, "config.yaml"))
    shutil.copyfile(run["spec"], os.path.join(out, "forward_spec.yaml"))
    print(f"[run] {run['name']} -> {out}")

    io = run["io"]
    U, Y, _ = load_split(run["task"], "train", device)
    has_val = os.path.isdir(os.path.join(task_dir(run["task"]), "val"))
    Uv, Yv, _ = load_split(run["task"], "val", device) if has_val else (U[:8], Y[:8], None)
    n_tr = min(int(tr.get("n_trials", 32)), U.shape[0])
    n_va = min(int(tr.get("n_val", 8)), Uv.shape[0])
    print(f"[data] task {run['task']}: using {n_tr} of {U.shape[0]} train trials, "
          f"{n_va} val  ({U.shape[1]} frames each)")
    print(f"[data] ONE TRIAL PER STEP -- engine.run has no batch axis (R0), so this is the "
          f"correct model rather than a fast one")

    engine.quiet(True)          # one banner per RUN, not one per rollout -- see engine.quiet
    sim = build(run, device)
    probe, _ = rollout(sim, U[0], io["drive_set"], io["drive_block"],
                       io["read_set"], io["read_block"], device, grad=True)
    params = list(probe.parameters())
    n_par = sum(p.numel() for p in params)
    if not params:
        raise ValueError(
            f"{run['name']}: nothing is learnable. `learnable:` produced no tensor leaf, so "
            f"there is nothing for an optimiser to move.")
    print(f"[fit] {len(params)} tensor(s), {n_par} values: "
          + ", ".join(f"{e['of']}.{e['block']}" for e in run["learnable"] if "block" in e))

    epochs = int(tr.get("epochs", 30))
    # TRUNCATED BPTT, A GROWING PREFIX. A 480-step rollout from an unfitted W is a hard credit
    # assignment: the gradient of a late frame passes through hundreds of tanh's and arrives
    # tiny and uninformative, so early epochs move the weights almost at random. Training on a
    # short prefix first gives a clean gradient for the part of the trajectory that is still
    # reachable, and lengthening it hands the network a task it can already half do. The
    # reference does exactly this (60 -> 480 frames) and it is the difference between fitting
    # and not; leaving it out was worth 0.65 of target variance against 0.0008 on a task of
    # comparable difficulty. Duplicates dropped so a floor collapsing the early stages costs no
    # epochs -- the defect that pinned the reference's horizon at 60 frames for a whole run.
    accum = max(1, int(tr.get("accum", 8)))      # trials averaged before a step; see the loop
    T_full = int(sim.n_frames)
    sched = sorted({max(int(tr.get("horizon_min", 60)), int(T_full * f))
                    for f in tr.get("curriculum", [0.125, 0.25, 0.5, 0.75, 1.0])})
    print(f"[fit] horizon curriculum {sched} frames of {T_full};  "
          f"gradient averaged over {accum} trials per step")
    opt = torch.optim.Adam(params, lr=float(tr.get("lr", 1e-2)))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ch = int(io.get("read_channel", 0))
    hist, best, best_state, t0 = [], np.inf, None, time.time()

    for ep in range(epochs):
        h = sched[min(len(sched) - 1, int(len(sched) * ep / max(epochs, 1)))]
        sim.n_frames = h
        perm = torch.randperm(n_tr)
        tot = 0.0
        # GRADIENT ACCUMULATION, WHICH IS NOT THE BATCH AXIS AND DOES NOT PRETEND TO BE. R0 would
        # make a rollout carry B trials and cost one rollout; this still costs `accum` rollouts,
        # so it buys no wall-clock at all. What it buys is the ONLY other thing a batch gives:
        # a gradient averaged over several trials instead of one. Stepping per trial made the
        # validation error bounce 27 -> 76 deg^2 between consecutive epochs -- not a failure to
        # learn but an optimiser being handed a different task each step. Averaging first is a
        # three-line change against an engine refactor, and it fixes the half of the problem
        # that was actually hurting.
        opt.zero_grad()
        for k, i in enumerate(perm.tolist()):
            H, y = rollout(sim, U[i], io["drive_set"], io["drive_block"],
                           io["read_set"], io["read_block"], device, grad=True)
            loss = _mse(y, Y[i], ch) / accum
            loss.backward()
            tot += float(loss.detach()) * accum
            if (k + 1) % accum == 0 or k == len(perm) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
        sch.step()
        sim.n_frames = T_full            # ALWAYS SCORED ON THE FULL TRIAL, whatever it trained on
        with torch.no_grad():
            v = 0.0
            for i in range(n_va):
                _, yv = rollout(sim, Uv[i], io["drive_set"], io["drive_block"],
                                io["read_set"], io["read_block"], device, grad=False)
                v += float(_mse(yv, Yv[i], ch))
            v /= max(n_va, 1)
        hist.append({"epoch": ep, "horizon": h, "train_mse": tot / n_tr, "val_mse": v})
        print(f"  ep {ep:3d}  horizon {h:4d}  train {tot / n_tr:10.5f}  val {v:10.5f}  deg^2")
        if v < best:
            best = v
            best_state = {k: p.detach().clone() for k, p in sim.fitted.items()}

    torch.save({"fitted": best_state, "spec": run["spec"], "task": run["task"],
                "learnable": run["learnable"], "io": io},
               os.path.join(out, "models", "best.pt"))
    rep = {"name": run["name"], "spec": run["spec"], "task": run["task"],
           "n_params": n_par, "n_trials": n_tr, "epochs": epochs,
           "best_val_mse": best, "history": hist,
           "target_variance": float((Y[:n_tr] ** 2).mean()),
           "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(out, "results", "report.json"), "w"), indent=2)
    print(f"[done] best val {best:.5f} deg^2 "
          f"({best / rep['target_variance']:.4f} of target variance) in {rep['seconds']:.0f} s")
    return out


# `_sync` is not needed: `engine.apply_learnable_blocks` holds each parameter on the SPEC and
# reuses it, so every rollout of one run writes the same leaf into state and the optimiser is
# always moving the tensor the next rollout will read.


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--option", nargs="+", default=["train"],
                    help="phases (train) and the run spec(s), in any order")
    ap.add_argument("--root", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    fns = {"train": train}
    phases = [o for o in a.option if o in fns] or ["train"]
    specs = [o for o in a.option if o not in fns]
    paths = [p for s in specs for p in (glob.glob(s) if any(c in s for c in "*?[") else [s])]
    if not paths:
        ap.error("no run spec given")
    for p in paths:
        run = load_run(p)
        for ph in phases:
            fns[ph](run, root=a.root, device=a.device)


if __name__ == "__main__":
    main()
