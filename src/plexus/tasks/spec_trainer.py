"""Train a Plexus SPEC against a task, through `engine.run(grad=True)`.

    python -m plexus.tasks.spec_trainer -o train test analyse config/run/eye_rig.yaml

The difference from `tasks.trainer`, which is a standalone torch module: this one fits the model
the spec describes, by running the engine. Nothing is transcribed, so nothing can drift -- the
thing being trained IS the thing `Plexus_Main.py -o generate` would run, operator for operator.

    forward spec     config/neural/ctrnn_eyeG_rig.yaml -- unchanged by training
    learnable:       which blocks are fitted (the run spec injects it)
    task             a corpus under graphs_data/task/
    run spec         names the three and the hyperparameters

B TRIALS IN ONE ROLLOUT. `engine.run(batch=B)` gives every set a leading trial axis, so a step
costs one rollout however many trials it averages -- R0 of notes/campaigns/TRAINING_IN_PLEXUS.md,
measured at 18.4x on this rig at B = 32 (171 -> 9.3 ms per trial of 120 frames). The axis is a
numerical fact and appears nowhere in the spec: the file being trained is the same file
`Plexus_Main.py -o generate` would run, operator for operator, so nothing can drift between the
model and the thing fitted.

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
from plexus.tasks.trainer import load_split, log_dir, n_condition_cells, with_context


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


def write_drive(H, frame, u, drive_set, drive_block):
    """Write frame `frame` of the stimulus into the drive set's block. THE ONE RULE, in one place.

    ONE CHANNEL PER SENSORY ELEMENT, not one value broadcast over all of them. A corpus with C
    channels -- a stimulus plus, for a teacher-varying grid, a one-hot of the condition cell --
    has to arrive on C DIFFERENT lines or the circuit cannot tell them apart, which is the whole
    point of sending the context at all. Element i takes channel i; a drive set wider than the
    corpus leaves its spare lines at whatever they were seeded to, so one spec with a fixed
    sensory bank serves tasks of several widths.

    `operating_slope` used to carry its own copy of this and kept the broadcast version after
    `rollout` moved on, so every multi-channel run trained fine and then died in `analyse`.
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


def rollout(sim, u, drive_set, drive_block, read_set, read_block, device="cpu", grad=True):
    """Write the stimulus into a block each frame, return the read-out trace.

    `u` is `[T, C]` for one trial or `[B, T, C]` for B of them -- B frames of the same world run
    side by side on a leading tensor axis, NOT B worlds. The return matches: `[T, w]` or
    `[B, T, w]`, the read-out set's element 0 every frame.

    `on_frame` is the engine's own per-frame hook, so driving a run from outside costs no change
    to the engine and no edit to the spec.
    """
    batch = int(u.shape[0]) if u.dim() == 3 else 1
    trace = []

    def hook(H, frame):
        write_drive(H, frame, u, drive_set, drive_block)
        trace.append(H.level(read_set).get(read_block)[..., 0, :].clone())

    H, _ = engine.run(sim, device=device, progress=False, grad=grad, on_frame=hook, batch=batch)
    if not trace:
        return H, None
    y = torch.stack(trace)                                 # [T, w] or [T, B, w]
    return H, y.transpose(0, 1) if batch > 1 else y        # trial-major, like the corpus


def _mse(y, target, ch):
    """Aligned at frame 0 and truncated to the shorter; batch-transparent.

    THE TWO SERIES START TOGETHER AND THE EXTRA FRAME IS ON THE END. `engine.run` iterates
    `range(n_frames + 1)` and `on_frame` fires at the top of each tick, so trace[k] is the state
    after k integration steps -- trace[0] the initial condition, and the last tick's step never
    recorded. The teacher's y[0] is likewise its output at t = 0, which for a strictly proper law
    is zero. So trace[k] and y[k] name the same instant, truncating to the shorter is correct,
    and padding would not be; the alternative -- dropping trace[0] -- would shift the whole
    comparison by one step and report a lag the model does not have.

    Indexed from the right, so `[T, w]` against `[T, 1]` and `[B, T, w]` against `[B, T, 1]` are
    the same call. The mean is over frames AND trials, which is exactly the gradient that
    accumulating B single-trial losses used to produce.
    """
    n = min(y.shape[-2], target.shape[-2])
    return ((y[..., :n, ch] - target[..., :n, 0]) ** 2).mean()


def _pole_max(tp) -> str:
    """The ground truth's slowest pole, or "n/a" when its law has none.

    A STATIC GAIN AND A PURE DELAY HAVE NO POLES. `t0_gain_unity` is y = k u and `t0_delay_100ms`
    is y(t) = u(t - d): neither is a rational transfer function with a denominator, so
    `teacher.pt` records an empty pole list for them and `max()` raised. `tasks.trainer` had
    already learned this and prints "n/a"; this one had only ever been pointed at laws that have
    poles, so the first task without any killed the whole analyse phase.
    """
    pr = list(tp.get("poles_real") or [])
    return f"{max(pr):+.4f}" if pr else "     n/a"


def units(run):
    """(unit, unit^2) for the read-out block, from the run spec's `io.unit:`.

    THE UNIT IS DECLARED, NOT DERIVED. `eye.pose` carries `unit=None` in its schema -- the degrees
    come from the characterisation the eye fit was made against, not from anything the engine
    knows -- so the only honest place for it is the run that pairs a spec with a task. A run that
    does not declare one prints bare numbers, which is the same rule `Units.describe()` applies to
    a whole simulation: t0..t4 read a teacher output that is not an angle, and labelling their
    error "deg^2" was a unit asserted by the printf rather than by the model.
    """
    u = (run.get("io") or {}).get("unit")
    return (f" {u}", f" {u}^2") if u else ("", "")


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
    U, Y, cU = load_split(run["task"], "train", device)
    has_val = os.path.isdir(os.path.join(task_dir(run["task"]), "val"))
    Uv, Yv, cV = load_split(run["task"], "val", device) if has_val else (U[:8], Y[:8], cU[:8])
    # See `trainer.with_context`: a grid over the TEACHER is unanswerable unless the circuit is
    # told which cell it is in. On by default whenever the corpus has more than one.
    n_cond = n_condition_cells(run["task"]) if run.get("context", True) else 1
    if n_cond > 1:
        print(f"[data] {n_cond} condition cells -> a one-hot context channel is appended "
              f"({U.shape[2]} -> {U.shape[2] + n_cond} inputs)")
    U, Uv = with_context(U, cU, n_cond), with_context(Uv, cV, n_cond)
    n_tr = min(int(tr.get("n_trials", 32)), U.shape[0])
    n_va = min(int(tr.get("n_val", 8)), Uv.shape[0])
    print(f"[data] task {run['task']}: using {n_tr} of {U.shape[0]} train trials, "
          f"{n_va} val  ({U.shape[1]} frames each)")

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
    # TRIALS PER STEP. `batch:` is the name now that the engine carries the axis; `accum:` is the
    # old name for the same QUANTITY -- the number of trials the gradient is averaged over before
    # the optimiser moves -- and is honoured so the recorded A/B runs keep their meaning. What
    # changed is the price: `accum: 8` cost eight rollouts, `batch: 8` costs one.
    batch = max(1, int(tr.get("batch", tr.get("accum", 8))))
    T_full = int(sim.n_frames)
    sched = sorted({max(int(tr.get("horizon_min", 60)), int(T_full * f))
                    for f in tr.get("curriculum", [0.125, 0.25, 0.5, 0.75, 1.0])})
    print(f"[fit] horizon curriculum {sched} frames of {T_full};  "
          f"gradient averaged over {batch} trials per step, ONE rollout per step")
    opt = torch.optim.Adam(params, lr=float(tr.get("lr", 1e-2)))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ch = int(io.get("read_channel", 0))
    U1, U2 = units(run)
    hist, best, best_state, t0 = [], np.inf, None, time.time()

    for ep in range(epochs):
        h = sched[min(len(sched) - 1, int(len(sched) * ep / max(epochs, 1)))]
        sim.n_frames = h
        perm = torch.randperm(n_tr)
        tot, n_step = 0.0, 0
        # ONE ROLLOUT PER STEP, B TRIALS WIDE. The loss is the mean over frames AND trials, which
        # is the same gradient the old accumulation produced by summing B single-trial losses
        # divided by B -- the arithmetic is unchanged and only the dispatch is saved. Averaging
        # over trials is what matters for the optimiser: stepping per trial made the validation
        # error bounce 27 -> 76 deg^2 between consecutive epochs, an optimiser handed a different
        # task each step rather than a failure to learn.
        #
        # A SHORT LAST GROUP IS DROPPED, not padded: `n_tr` trials in groups of `batch` leaves a
        # remainder that would take a step on fewer trials and a noisier gradient, at the end of
        # every epoch. The trials are permuted each epoch, so nothing is systematically unseen.
        for k in range(0, n_tr - batch + 1, batch):
            idx = perm[k:k + batch]
            H, y = rollout(sim, U[idx], io["drive_set"], io["drive_block"],
                           io["read_set"], io["read_block"], device, grad=True)
            loss = _mse(y, Y[idx], ch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss.detach()); n_step += 1
        sch.step()
        sim.n_frames = T_full            # ALWAYS SCORED ON THE FULL TRIAL, whatever it trained on
        with torch.no_grad():
            _, yv = rollout(sim, Uv[:n_va], io["drive_set"], io["drive_block"],
                            io["read_set"], io["read_block"], device, grad=False)
            v = float(_mse(yv, Yv[:n_va], ch))
        if int(tr.get("snapshot_every", 0)) and ep % int(tr["snapshot_every"]) == 0:
            # See `tasks.trainer`: one module owns every figure this repository draws of a fit.
            from plexus.tasks import plot_trainer as PT
            PT.snapshot(out, ep, ep * max(n_step, 1),
                        Uv[:4, :, :1].cpu().numpy(), Yv[:4].cpu().numpy(),
                        yv[:4].detach().cpu().numpy(), dt=float(sim.dt), unit=U1.strip(),
                        title=f"{run['name']}  epoch {ep}")
        tr_mse = tot / max(n_step, 1)
        hist.append({"epoch": ep, "horizon": h, "train_mse": tr_mse, "val_mse": v})
        print(f"  ep {ep:3d}  horizon {h:4d}  train {tr_mse:10.5f}  val {v:10.5f}{U2}"
              f"   (|err| {np.sqrt(v):.4f}{U1} rms)")
        if v < best:
            best = v
            best_state = {k: p.detach().clone() for k, p in sim.fitted.items()}

    torch.save({"fitted": best_state, "spec": run["spec"], "task": run["task"],
                "learnable": run["learnable"], "io": io, "n_cond": int(n_cond)},
               os.path.join(out, "models", "best.pt"))
    rep = {"name": run["name"], "spec": run["spec"], "task": run["task"],
           "n_params": n_par, "n_trials": n_tr, "epochs": epochs, "batch": batch,
           "best_val_mse": best, "history": hist,
           "target_variance": float((Y[:n_tr] ** 2).mean()),
           "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(out, "results", "report.json"), "w"), indent=2)
    print(f"[done] best val {best:.5f}{U2} ({np.sqrt(best):.4f}{U1} rms) "
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
    fns = {"train": train, "test": test, "analyse": analyse}
    phases = [o for o in a.option if o in fns] or ["train"]
    specs = [o for o in a.option if o not in fns]
    paths = [p for s in specs for p in (glob.glob(s) if any(c in s for c in "*?[") else [s])]
    if not paths:
        ap.error("no run spec given")
    for p in paths:
        run = load_run(p)
        for ph in phases:
            fns[ph](run, root=a.root, device=a.device)



def _restore(run, device="cpu"):
    """The Spec with the checkpoint's fitted values installed, ready to roll out.

    `sim.fitted` is what `engine.apply_learnable_blocks` reuses, so writing the checkpoint's
    tensors there before the first run is the whole of "load a model" -- there is no separate
    state_dict, because what was fitted is a block of the spec's own state.
    """
    out = log_dir(run["name"])
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    sim = build(run, device)
    sim.fitted = {k: torch.nn.Parameter(v.to(device)) for k, v in ck["fitted"].items()}
    return sim, ck, out


def test(run, root=None, device="cpu"):
    """Roll the checkpoint out on held-out trials. One number per trial, never only a mean."""
    engine.quiet(True)
    sim, ck, out = _restore(run, device)
    io = run["io"]
    split = "test" if os.path.isdir(os.path.join(task_dir(run["task"]), "test")) else "train"
    U, Y, cond = load_split(run["task"], split, device)
    U = with_context(U, cond, int(ck.get("n_cond", 1)))   # the width the FIT was made at
    n = min(int(run["training"].get("n_test", 24)), U.shape[0])
    ch = int(io.get("read_channel", 0))

    # ALL n TRIALS IN ONE ROLLOUT, then scored one at a time. The per-trial number is the point --
    # a mean hides the trial the fit failed on -- but computing it does not require running the
    # trials separately now that the engine carries them side by side.
    with torch.no_grad():
        _, Ypred = rollout(sim, U[:n], io["drive_set"], io["drive_block"],
                           io["read_set"], io["read_block"], device, grad=False)
    per = [float(_mse(Ypred[i], Y[i], ch)) for i in range(n)]
    traces = [Ypred[i, :, ch].cpu().numpy() for i in range(min(n, 6))]
    var = float((Y[:n] ** 2).mean())
    # IN DEGREES AS WELL AS DEGREES SQUARED. The reference this rig reproduces
    # (`train_eyeG.py` on connectome-gnn's feat/oculomotor) reports `gaze_err_mean_deg`, the mean
    # ABSOLUTE gaze error, and quotes 0.088 deg for a 64-unit ctRNN. A mean squared error in deg^2
    # cannot be compared with that by eye -- 2.12 deg^2 reads smaller than 0.088 and is 16x worse
    # -- so both are written, and the mean absolute error is the one to line up against the note.
    nn_ = min(Ypred.shape[-2], Y.shape[-2])
    err = (Ypred[:n, :nn_, ch] - Y[:n, :nn_, 0]).cpu().numpy()
    # THE SETTLED SCORE, BESIDE THE FULL ONE. Every rollout starts from the spec's seeded state
    # while the ground truth starts wherever its own law puts it, so the opening fraction of a
    # second is the circuit catching up from rest. It costs nothing when the ground truth also
    # starts at zero -- which is every task here whose teacher is a filter -- and everything on an
    # ALL-PASS law, where `tasks.trainer` measures the transient at 226x the settled residual
    # power. See `tasks.trainer.test` for the measurement; both are kept because both are true and
    # they answer different questions.
    settle_s = float(run["training"].get("settle_s", 0.5))
    k = min(int(round(settle_s / float(sim.dt))), nn_ - 1)
    yl = Y[:n, k:nn_, 0].cpu().numpy()
    mse_settled = float((err[:, k:] ** 2).mean())
    var_settled = float((yl ** 2).mean())
    res = {"name": run["name"], "spec": run["spec"], "task": run["task"], "split": split,
           "n_trials": n, "mse": float(np.mean(per)), "mse_per_trial": per,
           "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt(np.mean(per))),
           "unit": (run.get("io") or {}).get("unit"),
           "target_variance": var, "normalised_mse": float(np.mean(per)) / var,
           "settle_s": settle_s, "mse_settled": mse_settled,
           "normalised_mse_settled": mse_settled / var_settled if var_settled > 0 else None,
           "fitted": {k: {"n": int(v.numel()), "mean": float(v.mean()), "sd": float(v.std())}
                      for k, v in ck["fitted"].items()}}
    p = os.path.join(out, "results", f"{run['name']}_{split}.json")
    json.dump(res, open(p, "w"), indent=2)
    np.save(os.path.join(out, "results", f"{run['name']}_{split}_traces.npy"), np.array(traces))
    U1, U2 = units(run)
    print(f"[test] {split}: mean |err| {res['mae']:.4f}{U1}   rmse {res['rmse']:.4f}{U1}")
    print(f"[test] {split}: {n} trials  mse {res['mse']:.4f}{U2}  "
          f"({res['normalised_mse']:.4f} of target variance)")
    print(f"[test] settled (after {settle_s:g} s, the start-from-rest transient dropped): "
          f"{res['normalised_mse_settled']:.5f} of target variance")
    print(f"[test] per-trial spread {min(per):.3f} .. {max(per):.3f}{U2}")
    print(f"[test] wrote {p}")
    return res


def analyse(run, root=None, device="cpu"):
    """Did it recover the law, or only reduce the error? Those are different questions.

    Four panels, and only the last two are the analyser's own work -- the first two are what the
    tester already measured, drawn.

        a   target vs the fitted chain's gaze, held out
        b   the residual, on the same scale
        c   WHAT TRAINING DID TO EACH FITTED BLOCK: learned against its starting value, which for
            a measured connectome is learned-against-measured. The `plot_recovery_panels`
            template of connectome-gnn, at the one panel that applies when there is no synthetic
            ground truth: the departure IS the result.
        d   the circuit's own linearised poles against the teacher's, in one plane. A small error
            with the poles in the wrong place is what an unidentifiable task produces, and it is
            only visible here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks.render import BG, INK, MUTED, _ax

    engine.quiet(True)
    sim, ck, out = _restore(run, device)
    io = run["io"]
    split = "test" if os.path.isdir(os.path.join(task_dir(run["task"]), "test")) else "train"
    U, Y, cA = load_split(run["task"], split, device)
    U = with_context(U, cA, int(ck.get("n_cond", 1)))
    rep = json.load(open(os.path.join(out, "results", "report.json")))
    res_p = os.path.join(out, "results", f"{run['name']}_{split}.json")
    res = json.load(open(res_p)) if os.path.exists(res_p) else {}

    with torch.no_grad():
        _, Yp = rollout(sim, U[:4], io["drive_set"], io["drive_block"], io["read_set"],
                        io["read_block"], device, grad=False)
    preds = [Yp[i, :, int(io.get("read_channel", 0))].cpu().numpy() for i in range(4)]
    dt = float(sim.dt)
    t = np.arange(len(preds[0])) * dt

    # 3 x 2, the same layout `tasks.trainer.plot` uses. TOP ROW IS WHAT THE RUN WAS -- what went
    # in, what came out, the numbers. BOTTOM ROW IS WHETHER IT WORKED -- dynamics, error, history.
    U1, U2 = units(run)
    fig = plt.figure(figsize=(14.5, 7.6), facecolor=BG)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)

    # a: the stimulus the circuit actually saw
    axa = _ax(fig.add_subplot(gs[0, 0]), ylabel="input", letter="a")
    for i in range(4):
        axa.plot(t[:U.shape[1]], U[i, :, 0].cpu(), color="#4a4a4a", lw=0.7, alpha=0.8)
    if U.shape[2] > 1:
        axa.text(0.985, 0.04, f"channel 1 of {U.shape[2]}", transform=axa.transAxes,
                 ha="right", va="bottom", color=MUTED, fontsize=8)

    # b: target against prediction. Green is ground truth and thicker, so a good fit reads as a
    # black line sitting inside a green one rather than as two lines that disagree.
    axb = _ax(fig.add_subplot(gs[0, 1]), ylabel=f"gaze theta{U1}", letter="b")
    for i, pr in enumerate(preds):
        n = min(len(pr), Y.shape[1])
        axb.plot(t[:n], Y[i, :n, 0].cpu(), color="#2e8b4f", lw=2.2, alpha=0.9)
        axb.plot(t[:n], pr[:n], color="#000000", lw=0.9)
    axb.text(0.985, 0.04, "green: ground truth   black: circuit", transform=axb.transAxes,
             ha="right", va="bottom", color=MUTED, fontsize=8)

    # d: the recovery panel. Each point is an eigenvalue lambda = sigma + i omega of the circuit's
    # linearisation, so a mode behaves as exp(sigma t)(cos omega t + i sin omega t):
    #   Re, the x-axis in 1/s, is the ENVELOPE -- negative decays with time constant 1/|sigma|
    #     seconds, positive diverges, zero neither, which is what a perfect memory is.
    #   Im/2pi, the y-axis in Hz, is the RINGING inside it -- zero is a pure exponential.
    # Mirror-symmetric about y = 0 because the matrix is real and complex eigenvalues come in
    # conjugate pairs; a pair is ONE oscillating mode. Dashed red at Re = 0 is the stability
    # boundary. A small error with the poles in the wrong place is what an unidentifiable task
    # produces, and it is visible nowhere else.
    axd = _ax(fig.add_subplot(gs[1, 0]), xlabel="Re(lambda) (1/s)", ylabel="Im/2pi (Hz)",
              letter="d")
    plain = engine.run(build(run, device), device=device, progress=False)[0]
    slope, v_typ = operating_slope(sim, run, U[0], device)
    poles = _circuit_poles(sim, plain, ck, slope=slope)
    if poles is not None:
        at_op, at_origin = poles
        axd.scatter(at_origin.real, at_origin.imag / (2 * np.pi), s=10, color="0.72", alpha=0.7)
        axd.scatter(at_op.real, at_op.imag / (2 * np.pi), s=12, color="#000000", alpha=0.9)
    truth = torch.load(os.path.join(task_dir(run["task"]), "teacher.pt"), weights_only=False)
    tp = truth["per_cell"][0]
    axd.scatter(tp["poles_real"], np.asarray(tp["poles_imag"]) / (2 * np.pi), s=80, marker="x",
                color="#2e8b4f", lw=2.0, zorder=5)
    axd.axvline(0, color="#c0272a", lw=0.8, ls="--")
    # CAPTIONED, NOT LEGENDED, like panel b: a legend box goes in a corner of the axes and this
    # cloud fills the plane, so it sat on the data whichever corner it chose.
    axd.text(0.985, 0.04, "green: ground truth   black: circuit at its operating point   "
                          "grey: at v=0", transform=axd.transAxes,
             ha="right", va="bottom", color=MUTED, fontsize=7.5)

    # e: the residual, on the target's own scale
    axe = _ax(fig.add_subplot(gs[1, 1]), xlabel="time (s)", ylabel=f"residual{U1}", letter="e")
    for i, pr in enumerate(preds):
        n = min(len(pr), Y.shape[1])
        axe.plot(t[:n], pr[:n] - Y[i, :n, 0].cpu().numpy(), color="#c0522a", lw=0.8)
    axe.set_ylim(axb.get_ylim())

    # f: the learning curve
    axf = _ax(fig.add_subplot(gs[1, 2]), xlabel="epoch", ylabel="mse", letter="f")
    ep = [h["epoch"] for h in rep["history"]]
    axf.plot(ep, [h["train_mse"] for h in rep["history"]], color="#1f6fb8", lw=1.1, label="train")
    axf.plot(ep, [h["val_mse"] for h in rep["history"]], color="#9a7d1a", lw=1.1, label="val")
    axf.set_yscale("log")
    lg = axf.legend(frameon=False, fontsize=8, loc="upper right")
    for x in lg.get_texts():
        x.set_color(INK)

    # c: the numbers. No axes at all -- a frame around text has no axis to be a frame for.
    axc = _ax(fig.add_subplot(gs[0, 2]), letter="c")
    axc.axis("off")
    # THE BLOCK LIST IS WRAPPED, NOT ONE LINE EACH. Seven fitted blocks plus the pole block ran
    # this panel off the bottom and into panel f's legend.
    blocks = [f"{k.split('.')[0]}.{k.split('.')[1]}({v.numel()})" for k, v in ck["fitted"].items()]
    lines = ["", f"{run['name']}", f"spec   {os.path.basename(run['spec'])}",
             f"task   {run['task']}", f"split  {split}", "",
             f"fitted {rep['n_params']} values over {len(ck['fitted'])} block(s):"]
    lines += ["   " + "  ".join(blocks[i:i + 2]) for i in range(0, len(blocks), 2)]
    lines += ["", f"mean |err|  {res.get('mae', float('nan')):9.4f}{U1}   <- the comparable number",
              f"rmse        {res.get('rmse', float('nan')):9.4f}{U1}",
              f"best val    {rep['best_val_mse']:9.4f}{U2}",
              f"{split} mse     {res.get('mse', float('nan')):9.4f}{U2}",
              f"normalised  {res.get('normalised_mse', float('nan')):9.4f} of target variance",
              f"  settled   {res.get('normalised_mse_settled', float('nan')):9.5f} "
              f"(after {res.get('settle_s', 0.5):g} s from rest)",
              f"trained on  {rep['n_trials']} trials, {rep['epochs']} epochs",
              f"            {rep['seconds']:.0f} s"]
    if poles is not None:
        at_op, at_origin = poles
        lines += ["", f"|v| at operation {v_typ:.3f}  (rho' = {float(slope.mean()):.3f})",
                  f"max Re(lam)  v=0 {float(max(p.real for p in at_origin)):+.4f}   "
                  f"op {float(max(p.real for p in at_op)):+.4f}   "
                  f"truth {_pole_max(tp)}  1/s"]
    axc.text(0.02, 0.95, "\n".join(lines), transform=axc.transAxes, va="top", ha="left",
             color=INK, fontsize=8, family="monospace")

    p = os.path.join(out, "results", f"{run['name']}_{split}.png")
    fig.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyse] wrote {p}")
    return p


def _circuit_poles(sim, H, ck, slope=None):
    """The fitted circuit's poles, linearised where it ACTUALLY OPERATES.

        A = -diag(a) + diag(g) W diag(rho'(v))

    a is the leak and g the coupling gain, separate columns of the type table `p`; conflating
    them is right only when they are equal, which they happen to be in `ctrnn_eyeG_rig` (both
    2.0 = 1/tau0) and need not be anywhere else.

    `slope` is rho'(v) = 1 - tanh(v)^2 averaged over a rollout. THIS ARGUMENT IS THE WHOLE POINT
    AND OMITTING IT GAVE A WRONG ANSWER. An earlier version linearised at v = 0 and said so --
    "taken at the origin because that is where the task holds the state" -- which was simply
    false for this fit: the trained circuit runs at |v| mean 4.4 and max 23.9, where tanh' is
    0.272, not 1. At the origin it reported max Re(lambda) = +14.46 1/s and 2 unstable modes; at
    the operating point the same matrix gives +0.0094 1/s and 1. The first number describes a
    point the dynamics never visit, and it is the more alarming of the two, so reporting it alone
    would have been a false alarm dressed as a measurement.

    Both are returned, because the DIFFERENCE is diagnostic: a circuit whose two linearisations
    agree is operating in its linear regime, and one whose they do not is relying on saturation.

    Returns (poles_at_operating_point, poles_at_origin), or None when no recurrent block was
    fitted -- a spec without one has no circuit poles, and inventing some would be worse.
    """
    # THE RECURRENT EDGE SET IS FOUND STRUCTURALLY, not by its name. It is the fitted edge set
    # whose two endpoints are the SAME set -- that is what "recurrent" means -- and matching the
    # string instead found nothing on the zebrafish rig, whose recurrent connectome is called
    # `synapse`. Panel d then came out empty with no complaint, which reads as "this circuit has
    # no poles" rather than "this analyser could not find them".
    rec = next((k for k in ck["fitted"]
                if k.endswith(".w")
                and (lambda e: e.is_edge_set and e.pre_name == e.post_name)(H.level(k.split(".")[0]))),
               None)
    if rec is None:
        return None
    W = ck["fitted"][rec].detach().cpu().numpy().ravel()
    es = H.level(rec.split(".")[0])
    lvl = H.level(es.post_name)
    n = lvl.n
    # UNDER SIGN-LOCK THE EFFECTIVE WEIGHT IS |w| TIMES THE SENDER'S SIGN, so the fitted `w` is
    # not the matrix the dynamics use. Reading `w` raw would put the poles of a circuit that does
    # not exist on the plot -- and would do it silently, since both matrices have the same sparsity
    # and the same magnitudes.
    sig = dict(zip(H.operator_names, H.operators)).get("neuron_signal")
    if sig is not None and getattr(sig, "dale", False):
        W = np.abs(W) * sig._dale_sign(lvl, es).squeeze(-1).detach().cpu().numpy()
    M = np.zeros((n, n))
    M[es.post.cpu().numpy(), es.pre.cpu().numpy()] = W
    # THE LEAK AND THE COUPLING GAIN ARE SEPARATE COLUMNS OF `p`, and conflating them is right
    # only by coincidence. `neuron_update` contributes -a v and `neuron_signal` contributes
    # g W tanh(v), so the linearisation at v = 0 is
    #
    #     A = -diag(a) + diag(g) W          NOT   diag(a) (-I + W)
    #
    # The two agree exactly when a == g, which they do in `ctrnn_eyeG_rig` (both 2.0 = 1/tau0),
    # so the wrong form would have produced the right number here and a silently wrong one on
    # any spec whose gain differs from its leak.
    P = (lvl.type_params[lvl.node_type] if getattr(lvl, "type_params", None) is not None
         else torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]).expand(n, 6))
    a = P[:, 0].detach().cpu().numpy()                  # leak, 1/s
    g = P[:, 2].detach().cpu().numpy()                  # coupling gain, dimensionless
    # A PER-NEURON `tau:` BLOCK OVERRIDES THE TYPE TABLE'S LEAK, exactly as `neuron_update` does
    # it -- and once tau is FITTED, the table's column 0 is the value the run started from rather
    # than the one it ended at. On the ctRNN rig the two agreed by coincidence (p[0] = 2.0 =
    # 1/0.5 s) and on the zebrafish pool they do not (p[0] = 10.0 against a learned tau).
    upd = dict(zip(H.operator_names, H.operators)).get("neuron_update")
    tb = getattr(upd, "tau_block", None)
    if tb is not None and tb in lvl.state_schema:
        tmin = float(getattr(upd, "tau_min", 1e-4))
        a = 1.0 / lvl.get(tb)[:, 0].detach().cpu().numpy().clip(min=tmin)
    rho = np.ones(n) if slope is None else np.asarray(slope, float).reshape(n)
    at_op = np.linalg.eigvals(-np.diag(a) + np.diag(g) @ M @ np.diag(rho))
    at_origin = np.linalg.eigvals(-np.diag(a) + np.diag(g) @ M)
    return at_op, at_origin


def operating_slope(sim, run, U, device="cpu"):
    """rho'(v) per unit, averaged over one rollout -- where the circuit actually sits.

    Measured rather than assumed, because assuming it is 1 is exactly the error above.
    """
    io = run["io"]
    vs = []

    def hook(H, frame):
        write_drive(H, frame, U, io["drive_set"], io["drive_block"])
        vs.append(H.level("neuron").get("voltage").squeeze(-1).clone())

    with torch.no_grad():
        engine.run(sim, device=device, progress=False, grad=False, on_frame=hook)
    V = torch.stack(vs)
    return (1.0 - torch.tanh(V) ** 2).mean(0).cpu().numpy(), float(V.abs().mean())

if __name__ == "__main__":
    main()
