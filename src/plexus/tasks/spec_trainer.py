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
    n = min(int(run["training"].get("n_test", 24)), U.shape[0])
    ch = int(io.get("read_channel", 0))

    per, traces = [], []
    with torch.no_grad():
        for i in range(n):
            _, y = rollout(sim, U[i], io["drive_set"], io["drive_block"],
                           io["read_set"], io["read_block"], device, grad=False)
            per.append(float(_mse(y, Y[i], ch)))
            if i < 6:
                traces.append(y[:, ch].cpu().numpy())
    var = float((Y[:n] ** 2).mean())
    res = {"name": run["name"], "spec": run["spec"], "task": run["task"], "split": split,
           "n_trials": n, "mse": float(np.mean(per)), "mse_per_trial": per,
           "target_variance": var, "normalised_mse": float(np.mean(per)) / var,
           "fitted": {k: {"n": int(v.numel()), "mean": float(v.mean()), "sd": float(v.std())}
                      for k, v in ck["fitted"].items()}}
    p = os.path.join(out, "results", f"{run['name']}_{split}.json")
    json.dump(res, open(p, "w"), indent=2)
    np.save(os.path.join(out, "results", f"{run['name']}_{split}_traces.npy"), np.array(traces))
    print(f"[test] {split}: {n} trials  mse {res['mse']:.4f} deg^2  "
          f"({res['normalised_mse']:.4f} of target variance)")
    print(f"[test] per-trial spread {min(per):.3f} .. {max(per):.3f} deg^2")
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
    U, Y, _ = load_split(run["task"], split, device)
    rep = json.load(open(os.path.join(out, "results", "report.json")))
    res_p = os.path.join(out, "results", f"{run['name']}_{split}.json")
    res = json.load(open(res_p)) if os.path.exists(res_p) else {}

    with torch.no_grad():
        preds = [rollout(sim, U[i], io["drive_set"], io["drive_block"], io["read_set"],
                         io["read_block"], device, grad=False)[1][:, int(io.get("read_channel", 0))]
                 .cpu().numpy() for i in range(4)]
    dt = float(sim.dt)
    t = np.arange(len(preds[0])) * dt

    fig = plt.figure(figsize=(14.5, 7.0), facecolor=BG)
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30)

    axa = _ax(fig.add_subplot(gs[0, 0]), ylabel="gaze theta (deg)", letter="a")
    for i, p in enumerate(preds):
        n = min(len(p), Y.shape[1])
        axa.plot(t[:n], Y[i, :n, 0].cpu(), color="#2e8b4f", lw=1.3, alpha=0.9)
        axa.plot(t[:n], p[:n], color=INK, lw=0.8, ls="--")
    axa.text(0.985, 0.04, "green: target   dashed: fitted chain", transform=axa.transAxes,
             ha="right", va="bottom", color=MUTED, fontsize=8)

    axb = _ax(fig.add_subplot(gs[1, 0]), xlabel="time (s)", ylabel="residual (deg)", letter="b")
    for i, p in enumerate(preds):
        n = min(len(p), Y.shape[1])
        axb.plot(t[:n], p[:n] - Y[i, :n, 0].cpu().numpy(), color="#c0522a", lw=0.8)
    axb.set_ylim(axa.get_ylim())

    # c: the departure from the starting value, per fitted block
    axc = _ax(fig.add_subplot(gs[0, 1]), xlabel="initial (the spec's own)", ylabel="learned",
              letter="c")
    plain = engine.run(build(run, device), device=device, progress=False)[0]
    for k, v in ck["fitted"].items():
        sname, blk = k.split(".")
        init = plain.level(sname).get(blk).detach().cpu().numpy().ravel()
        got = v.detach().cpu().numpy().ravel()
        axc.scatter(init, got, s=4, alpha=0.4, label=f"{k} (n={got.size})")
    lim = axc.get_xlim() + axc.get_ylim()
    lo, hi = min(lim), max(lim)
    axc.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
    lg = axc.legend(frameon=False, fontsize=7, loc="upper left")
    for x in lg.get_texts():
        x.set_color(INK)

    # d: the circuit's poles against the teacher's
    axd = _ax(fig.add_subplot(gs[1, 1]), xlabel="Re(lambda) (1/s)", ylabel="Im/2pi (Hz)",
              letter="d")
    slope, v_typ = operating_slope(sim, run, U[0], device)
    poles = _circuit_poles(sim, plain, ck, slope=slope)
    if poles is not None:
        at_op, at_origin = poles
        axd.scatter(at_origin.real, at_origin.imag / (2 * np.pi), s=10, color="0.78", alpha=0.7,
                    label="circuit, at v=0")
        axd.scatter(at_op.real, at_op.imag / (2 * np.pi), s=12, color=MUTED, alpha=0.9,
                    label="circuit, at its operating point")
    truth = torch.load(os.path.join(task_dir(run["task"]), "teacher.pt"), weights_only=False)
    tp = truth["per_cell"][0]
    axd.scatter(tp["poles_real"], np.asarray(tp["poles_imag"]) / (2 * np.pi), s=80, marker="x",
                color="#2e8b4f", lw=2.0, label="teacher", zorder=5)
    axd.axvline(0, color="#c0272a", lw=0.8, ls="--")
    lg = axd.legend(frameon=False, fontsize=8, loc="upper left")
    for x in lg.get_texts():
        x.set_color(INK)

    axe = _ax(fig.add_subplot(gs[:, 2]), letter="e")
    axe.set_xticks([]); axe.set_yticks([])
    lines = ["", f"{run['name']}", f"spec   {os.path.basename(run['spec'])}",
             f"task   {run['task']}", f"split  {split}", "",
             f"fitted {rep['n_params']} values over {len(ck['fitted'])} block(s):"]
    for k, v in ck["fitted"].items():
        lines.append(f"   {k:16s} n={v.numel():5d}")
    lines += ["", f"best val    {rep['best_val_mse']:9.4f} deg^2",
              f"{split} mse     {res.get('mse', float('nan')):9.4f} deg^2",
              f"normalised  {res.get('normalised_mse', float('nan')):9.4f} of target variance",
              f"trained on  {rep['n_trials']} trials, {rep['epochs']} epochs",
              f"            {rep['seconds']:.0f} s"]
    if poles is not None:
        at_op, at_origin = poles
        lines += ["", f"|v| at operation   {v_typ:8.3f}   (rho' = {float(slope.mean()):.3f})",
                  f"max Re(lam)  at v=0     {float(max(p.real for p in at_origin)):+8.4f} 1/s",
                  f"             operating  {float(max(p.real for p in at_op)):+8.4f} 1/s",
                  f"             teacher    {max(tp['poles_real']):+8.4f} 1/s"]
    axe.text(0.05, 0.95, "\n".join(lines), transform=axe.transAxes, va="top", ha="left",
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
    rec = next((k for k in ck["fitted"] if "recurrent" in k), None)
    if rec is None:
        return None
    W = ck["fitted"][rec].detach().cpu().numpy()
    es = H.level(rec.split(".")[0])
    n = H.level(es.post_name).n
    M = np.zeros((n, n))
    M[es.post.cpu().numpy(), es.pre.cpu().numpy()] = W.ravel()
    lvl = H.level(es.post_name)
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
        lvl = H.level(io["drive_set"])
        a, b = lvl.state_schema[io["drive_block"]]
        st = lvl.state.clone()
        st[:, a:b] = U[min(frame, U.shape[0] - 1)].expand(lvl.n, b - a)
        lvl.state = st
        vs.append(H.level("neuron").get("voltage").squeeze(-1).clone())

    with torch.no_grad():
        engine.run(sim, device=device, progress=False, grad=False, on_frame=hook)
    V = torch.stack(vs)
    return (1.0 - torch.tanh(V) ** 2).mean(0).cpu().numpy(), float(V.abs().mean())

if __name__ == "__main__":
    main()
