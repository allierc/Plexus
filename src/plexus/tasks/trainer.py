"""Fit a circuit to a task, roll it out, and say whether it recovered the law.

    python -m plexus.tasks.trainer -o train config/run/t1_integrator_ctrnn64.yaml
    python -m plexus.tasks.trainer -o test plot config/run/t1_integrator_ctrnn64.yaml
    python -m plexus.tasks.trainer -o train test plot config/run/*.yaml

THE THIRD SPEC. A model spec says what the circuit is; a task spec says what the data is; neither
can say which of them is paired with which, what the objective is, or how long to train. That is
a RUN, and it gets its own file naming the other two:

    task:      the corpus, by name, under graphs_data/task/
    circuit:   hidden, tau0, and how W starts
    training:  epochs, batch, lr, seed, curriculum

Output follows the layout every other run in this lineage uses, so a run of this is findable the
same way a GNN_Main.py run is:

    log/task/<name>/
        config.yaml                 the run spec, copied -- the run is self-describing
        models/best.pt              the checkpoint, by validation error
        results/report.json         what training did
        results/<name>_test.json    the held-out rollout
        results/<name>_test.png     traces, and the recovery figure below

WHY A STANDALONE MODULE AND NOT THE ENGINE. The circuit below is arithmetically the composition
`project -> neuron_update -> neuron_signal -> readout` from `plexus.operators`, and
`test_trainer_matches_the_operators` asserts that against the registered operators rather than
asserting it here in prose. It is written out because `engine.run(grad=True)` carries no batch
axis: fitting is 512 trials x 200 epochs, and stepping one trial at a time through the engine is
about five orders of magnitude of wasted wall-clock for a model that is four matmuls. The spec
remains the description; this is the differentiable evaluation of it.

WHAT IS SCORED, AND WHY IT IS NOT ONLY THE LOSS. A loss says the output is close. For an LTI
teacher the truth is analytic, so the run also asks whether the circuit acquired the RIGHT
DYNAMICS: drive the trained network with the chirp probe, take the ratio of output to input
spectra, and compare the poles that implies with the teacher's own, which `teacher.pt` recorded
when the corpus was generated. A circuit can reach a small error on a corpus and still have the
wrong poles -- that is exactly what an unidentifiable task produces -- and only the second
measurement can tell.
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

from plexus.paths import graphs_data_path
from plexus.tasks.generate import task_dir


def log_dir(name, root=None) -> str:
    return os.path.join(root or graphs_data_path().replace("graphs_data", "log"), "task", str(name))


# --------------------------------------------------------------------------- #
#  the circuit
# --------------------------------------------------------------------------- #
class CircuitRNN(nn.Module):
    """W_in -> a continuous-time recurrent population -> W_out. The rig, without the organ.

        I(t)     = W_in u(t)                                   `project`
        tau dv   = -v + W_hat r + I,   r = tanh(v)             `neuron_update` + `neuron_signal`
        y(t)     = W_out r(t)                                  `readout`

    v is the membrane state of each unit, dimensionless; tau its time constant in seconds, learnt
    per unit and stored as a log so it stays positive under gradient descent without a clamp.
    W_hat is the recurrent matrix, W_in the input map and W_out the output map.

    W STARTS AT ZERO, following the reference prototype. A recurrent matrix drawn at random is a
    random dynamical system, and the first epochs are then spent undoing it; starting at zero
    makes the network a bank of independent leaky integrators whose time constants are already
    right for the task, and every recurrent weight that becomes nonzero is one training put
    there. It also makes the learnt W readable: it is the deviation from no coupling at all.

    THE READOUT IS LINEAR AND HAS NO RECTIFIER, unlike the oculomotor rig's, because a muscle
    pulls or does nothing while a task target is a signed quantity. `send: tanh` is kept -- what
    leaves a unit is its rate, not its voltage.
    """

    def __init__(self, n_in, n_out, hidden=64, tau0=0.5, dt=1 / 60, w_init="zeros", seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.dt, self.hidden = float(dt), int(hidden)
        self.log_tau = nn.Parameter(torch.full((hidden,), float(np.log(tau0))))
        if w_init == "zeros":
            W = torch.zeros(hidden, hidden)
        else:                                       # the alternative, kept so it is measurable
            W = torch.randn(hidden, hidden, generator=g) / np.sqrt(hidden)
        self.W = nn.Parameter(W)
        self.W_in = nn.Parameter(torch.randn(hidden, n_in, generator=g) * 0.5 / np.sqrt(n_in))
        self.W_out = nn.Parameter(torch.randn(n_out, hidden, generator=g) / np.sqrt(hidden))
        self.b_out = nn.Parameter(torch.zeros(n_out))

    def forward(self, u, want_rates=False):
        """u (B, T, n_in) -> y (B, T, n_out), and the rates if asked."""
        alpha = (self.dt / self.log_tau.exp()).clamp(1e-4, 1.0)      # dt / tau, per unit
        B, T, _ = u.shape
        v = torch.zeros(B, self.hidden, device=u.device, dtype=u.dtype)
        I = u @ self.W_in.T
        rates = []
        for t in range(T):
            r = torch.tanh(v)
            rates.append(r)
            v = v + alpha * (-v + r @ self.W.T + I[:, t])
        R = torch.stack(rates, 1)
        y = R @ self.W_out.T + self.b_out
        return (y, R) if want_rates else y

    # -- what the circuit IS, read off rather than asserted ------------------------------- #
    def jacobian_poles(self):
        """The continuous-time poles of the linearisation at v = 0: eig(diag(1/tau)(-I + W)).

        At v = 0, tanh'(0) = 1, so the linearised system is dv/dt = diag(1/tau)(-I + W) v. Its
        eigenvalues are in the same units as a teacher's poles and so are directly comparable.

        IT IS THE ORIGIN'S JACOBIAN AND THAT IS ONLY MEANINGFUL WHILE THE CIRCUIT SITS NEAR THE
        ORIGIN. An earlier version of this docstring claimed the task holds the state there; on
        the oculomotor rig that was false by a wide margin -- the trained circuit runs at |v|
        mean 4.4, where tanh' is 0.272 -- and the origin's Jacobian reported max Re(lambda) =
        +14.46 1/s and two unstable modes where the operating point gives +0.0094 and one. The
        alarming number described a point the dynamics never visit.

        Check it before quoting it: if |v| is not small, linearise at the operating point
        instead, A = -diag(a) + diag(g) W diag(tanh'(v)), as `spec_trainer._circuit_poles` does.
        The DIFFERENCE between the two is itself diagnostic -- a circuit whose linearisations
        agree is working in its linear regime, one whose do not is relying on saturation.
        """
        with torch.no_grad():
            A = torch.diag(1.0 / self.log_tau.exp()) @ (-torch.eye(self.hidden) + self.W)
            return np.linalg.eigvals(A.cpu().numpy())


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
def load_split(task, split, device="cpu"):
    import zarr
    d = os.path.join(task_dir(task), split)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"{d} does not exist. Generate the corpus first:\n"
            f"    python -m plexus.tasks.generate config/task/{task}.yaml")
    u = np.asarray(zarr.load(os.path.join(d, "stimulus.zarr")), np.float32)
    y = np.asarray(zarr.load(os.path.join(d, "target.zarr")), np.float32)
    c = np.asarray(zarr.load(os.path.join(d, "cond.zarr")))
    return (torch.as_tensor(u).to(device), torch.as_tensor(y).to(device), c)


def with_context(U, cond, n_cond):
    """Append a one-hot of the trial's CONDITION CELL to every frame of the stimulus.

    WHAT THIS FIXES IS AN ILL-POSED TASK, NOT A BADLY TRAINED ONE. A corpus whose `conditions:`
    grid varies the TEACHER -- `t1_integrator_tau_sweep` over four time constants,
    `t2_resonator_damping` over three dampings, `t3_lowpass_order` over six family x order pairs
    -- shows the circuit trials from several different systems with nothing in the input saying
    which. The best any causal predictor can do is the average over teachers, so a large residual
    is the CORRECT answer and no amount of training removes it. Measured: the same circuit, the
    same hyperparameters, `t1_integrator_tau_sweep` restricted to its tau = 8 s cell alone goes
    from 0.2598 of target variance to 0.0005, a factor of 520, and its slowest pole comes back
    (-0.121 against the teacher's -0.125 1/s) where the grid left it 15x too fast.

    The one-hot is CONSTANT IN TIME, which is the point: it is context, not stimulus. It carries
    no information about what the target does moment to moment and every bit of information about
    which law is producing it -- so a circuit that still fails has failed at the dynamics rather
    than at telling the trials apart.

    The alternative is one run per condition cell, which answers a different question: it asks
    whether the circuit can fit each law, not whether one circuit can hold all of them at once.
    """
    if n_cond <= 1:
        return U
    oh = torch.zeros(U.shape[0], n_cond, device=U.device, dtype=U.dtype)
    oh[torch.arange(U.shape[0]), torch.as_tensor(cond, dtype=torch.long)] = 1.0
    return torch.cat([U, oh[:, None, :].expand(-1, U.shape[1], -1)], dim=-1)


def n_condition_cells(task) -> int:
    """How many condition cells the corpus has, from its own provenance."""
    prov = json.load(open(os.path.join(task_dir(task), "provenance.json")))
    return max(1, len(prov.get("cells") or []))


def load_run(path) -> dict:
    with open(path) as f:
        r = yaml.safe_load(f)
    for k in ("name", "task", "circuit", "training"):
        if k not in r:
            raise ValueError(f"{path}: a run spec needs `{k}:`")
    r["_path"] = os.path.abspath(path)
    return r


# --------------------------------------------------------------------------- #
#  train
# --------------------------------------------------------------------------- #
def train(run, root=None, device=None):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tr = run["training"]
    torch.manual_seed(int(tr.get("seed", 0)))
    out = log_dir(run["name"], root)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    shutil.copyfile(run["_path"], os.path.join(out, "config.yaml"))

    prov = json.load(open(os.path.join(task_dir(run["task"]), "provenance.json")))
    dt = float(prov["dt"])
    U, Y, cU = load_split(run["task"], "train", dev)
    has_val = os.path.isdir(os.path.join(task_dir(run["task"]), "val"))
    Uv, Yv, cV = load_split(run["task"], "val", dev) if has_val else (U[:64], Y[:64], cU[:64])
    # THE CONDITION CELL AS A CONSTANT INPUT CHANNEL, on by default for a corpus that has more
    # than one. Without it a teacher-varying grid is not a hard task but an unanswerable one;
    # see `with_context`. `context: false` in the run spec turns it off, which is how the
    # unanswerable version stays measurable.
    n_cond = n_condition_cells(run["task"]) if run.get("context", True) else 1
    if n_cond > 1:
        print(f"[data] {n_cond} condition cells -> a one-hot context channel is appended to the "
              f"stimulus ({U.shape[2]} -> {U.shape[2] + n_cond} inputs)")
    U, Uv = with_context(U, cU, n_cond), with_context(Uv, cV, n_cond)
    print(f"[run] {run['name']} -> {out}")
    print(f"[data] task {run['task']}: train {tuple(U.shape)}  val {tuple(Uv.shape)}"
          + ("" if has_val else "  (no val split -- scoring on a slice of train, reported as such)"))
    if not prov["excitation"]["identifiable"]:
        print("[data] WARNING: this corpus is NOT IDENTIFIABLE -- at least one of the teacher's "
              "poles sits where the stimulus has no power, so a low error here does NOT mean the "
              "circuit acquired the right dynamics.")

    c = run["circuit"]
    model = CircuitRNN(U.shape[2], Y.shape[2], hidden=int(c.get("hidden", 64)),
                       tau0=float(c.get("tau0", 0.5)), dt=dt,
                       w_init=c.get("w_init", "zeros"), seed=int(tr.get("seed", 0))).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[circuit] {model.hidden} units, {n_par} parameters, W init {c.get('w_init','zeros')}")

    epochs, batch = int(tr.get("epochs", 200)), int(tr.get("batch", 32))
    opt = torch.optim.Adam(model.parameters(), lr=float(tr.get("lr", 1e-2)))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    T = U.shape[1]
    # Truncated-BPTT curriculum: train on a growing prefix. Duplicates dropped so the early
    # stages collapsing at the floor cost no epochs -- the reference lost every run to an
    # off-by-one here, pinning the horizon at 60 frames while scoring on all 480.
    sched = sorted({max(30, int(T * f)) for f in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)}) \
        if tr.get("curriculum", True) else [T]

    hist, best, best_state = [], np.inf, None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        h = sched[min(len(sched) - 1, int(len(sched) * ep / max(epochs, 1)))]
        perm = torch.randperm(U.shape[0], device=dev)
        tot = 0.0
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            loss = ((model(U[j, :h]) - Y[j, :h]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss.detach()) * len(j)
        sch.step()
        # A FIGURE WHILE IT TRAINS, not only when it is over. `plot_trainer` owns the layout and
        # the naming. ITS OWN CADENCE, not the validation block's: nested inside `ep % 10` it
        # would fire only where the two coincide, so `snapshot_every: 2` would mean every tenth
        # epoch and say so nowhere. One forward on four trials is cheap enough to pay separately.
        if int(tr.get("snapshot_every", 0)) and ep % int(tr["snapshot_every"]) == 0:
            from plexus.tasks import plot_trainer as PT
            model.eval()
            with torch.no_grad():
                Pv = model(Uv[:4])
            PT.snapshot(out, ep, ep * max(1, len(perm) // batch),
                        Uv[:4, :, :1].cpu().numpy(), Yv[:4].cpu().numpy(), Pv.cpu().numpy(),
                        dt=dt, title=f"{run['name']}  epoch {ep}")
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                v = float(((model(Uv) - Yv) ** 2).mean())
            hist.append({"epoch": ep, "horizon": h, "train_mse": tot / len(perm), "val_mse": v})
            print(f"  ep {ep:4d}  horizon {h:4d}  train {tot / len(perm):.6f}  val {v:.6f}")
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"state": model.state_dict(), "circuit": c, "task": run["task"], "dt": dt,
                "n_in": int(U.shape[2]), "n_out": int(Y.shape[2]), "n_cond": int(n_cond)},
               os.path.join(out, "models", "best.pt"))
    rep = {"name": run["name"], "task": run["task"], "n_params": n_par,
           "best_val_mse": best, "epochs": epochs, "history": hist,
           "seconds": round(time.time() - t0, 1),
           "target_variance": float((Y ** 2).mean()),
           "identifiable": prov["excitation"]["identifiable"],
           "has_val_split": has_val}
    json.dump(rep, open(os.path.join(out, "results", "report.json"), "w"), indent=2)
    print(f"[done] best val mse {best:.6f}  ({best / rep['target_variance']:.4f} of target "
          f"variance)  in {rep['seconds']:.0f} s\n[done] wrote {out}")
    return out


# --------------------------------------------------------------------------- #
#  test -- the held-out rollout, and pole recovery
# --------------------------------------------------------------------------- #
def test(run, root=None, device=None):
    dev = torch.device(device or "cpu")
    out = log_dir(run["name"], root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=dev)
    model = CircuitRNN(ck["n_in"], ck["n_out"], hidden=int(ck["circuit"].get("hidden", 64)),
                       tau0=float(ck["circuit"].get("tau0", 0.5)), dt=ck["dt"]).to(dev)
    model.load_state_dict(ck["state"]); model.eval()

    split = "test" if os.path.isdir(os.path.join(task_dir(run["task"]), "test")) else "train"
    U, Y, cond = load_split(run["task"], split, dev)
    # REBUILT FROM THE CHECKPOINT, not from the run spec: the model's W_in has the width the fit
    # was made at, and a run spec edited between train and test would otherwise raise a shape
    # error here or, worse, line the one-hot up against a different number of cells.
    U = with_context(U, cond, int(ck.get("n_cond", 1)))
    with torch.no_grad():
        P = model(U)
    mse = float(((P - Y) ** 2).mean())
    var = float((Y ** 2).mean())
    # THE SETTLED SCORE, BESIDE THE FULL ONE AND NOT INSTEAD OF IT. Every rollout starts at v = 0
    # while the ground truth starts whereever its own law puts it, so the first fraction of a
    # second is the circuit catching up from rest. For every teacher that is itself a filter
    # starting from rest this costs nothing -- y(0) = 0 and there is nothing to catch up to -- but
    # `t0_gain_unity` is an ALL-PASS law, the one law whose output must be nonzero the instant the
    # input is, and its |y(0)| averages 0.92. There the transient is 226x the settled residual
    # power and the full-trial score reads 0.0033 where the settled one reads 0.00023, the best in
    # the battery. Scored in-band the identity map is reproduced to 2 parts in 10,000; the 0.0033
    # is the scoring window, not the fit.
    #
    # BOTH ARE REPORTED because both are true and they answer different questions. "Can this
    # circuit track the law from rest" is a real property of a leaky continuous-time system -- its
    # fastest learned tau here is 70 ms -- and dropping the transient would flatter it. "Did it
    # acquire the law" is what the battery exists to ask, and only the settled number answers it.
    settle_s = float(run.get("training", {}).get("settle_s", 0.5))
    k = min(int(round(settle_s / float(ck["dt"]))), Y.shape[1] - 1)
    mse_settled = float(((P[:, k:] - Y[:, k:]) ** 2).mean())
    var_settled = float((Y[:, k:] ** 2).mean())
    # per condition cell, because a grid exists to be read per cell
    per_cell = {}
    for c in sorted(set(cond.tolist())):
        m = cond == c
        per_cell[int(c)] = float(((P[m] - Y[m]) ** 2).mean())

    truth = torch.load(os.path.join(task_dir(run["task"]), "teacher.pt"), weights_only=False)
    # EVERY CELL'S TEACHER, NOT CELL 0's. With the context channel the circuit is asked to hold
    # ALL the laws in the grid at once, so the timescale it has to be able to reach is the
    # SLOWEST of them -- and comparing against cell 0 alone reported `t1_integrator_tau_sweep`'s
    # circuit as 1.97 1/s faster-growing than "the" teacher when the teacher it was being judged
    # against was the tau = 0.5 s cell and the grid also contains tau = 32 s.
    cells = truth["per_cell"]
    true_f = sorted({round(f, 6) for c in cells for f in c["pole_freq_hz"]})
    got = model.jacobian_poles()
    # the circuit's slowest modes are the ones a task's poles have to live among
    slow = sorted(got, key=lambda p: abs(p.real))[:8]
    got_f = [float(abs(p.imag) / (2 * np.pi)) if abs(p.imag) > 1e-12
             else float(abs(p.real) / (2 * np.pi)) for p in slow]

    # STABILITY IS JUDGED AGAINST THE TEACHER, NOT AGAINST ZERO. A perfect integrator's pole IS
    # at the origin, so a circuit that reproduces it must sit at the stability boundary; calling
    # that "unstable" would mark the correct answer as a failure. What is meaningful is the GAP
    # between the circuit's slowest mode and the teacher's: for 1/s that gap is how far from a
    # true line attractor the circuit landed, in inverse seconds.
    true_max_re = max([float(r) for c in cells
                       for r in np.atleast_1d(np.asarray(c["poles_real"], float))],
                      default=float("-inf"))
    circ_max_re = float(max(p.real for p in got))
    gap = circ_max_re - true_max_re if np.isfinite(true_max_re) else None

    res = {"name": run["name"], "task": run["task"], "split": split,
           "n_trials": int(U.shape[0]), "mse": mse, "target_variance": var,
           "normalised_mse": mse / var if var > 0 else None,
           "settle_s": settle_s, "mse_settled": mse_settled,
           "normalised_mse_settled": mse_settled / var_settled if var_settled > 0 else None,
           "mse_per_cell": per_cell,
           "teacher_pole_freq_hz": true_f, "n_condition_cells": len(cells),
           "circuit_slowest_pole_freq_hz": got_f,
           "circuit_max_real_eig": circ_max_re,
           "teacher_max_real_pole": None if not np.isfinite(true_max_re) else float(true_max_re),
           "max_real_gap": None if gap is None else float(gap),
           "identifiable": json.load(open(os.path.join(
               task_dir(run["task"]), "provenance.json")))["excitation"]["identifiable"]}
    p = os.path.join(out, "results", f"{run['name']}_{split}.json")
    json.dump(res, open(p, "w"), indent=2, default=str)
    print(f"[test] {split}: {U.shape[0]} trials  mse {mse:.6f}  "
          f"({res['normalised_mse']:.4f} of target variance)")
    print(f"[test] settled (after {settle_s:g} s, the start-from-rest transient dropped): "
          f"{res['normalised_mse_settled']:.5f} of target variance")
    print(f"[test] teacher poles {np.round(true_f, 4)} Hz   "
          f"circuit slowest {np.round(got_f[:4], 4)} Hz")
    if gap is None:
        print(f"[test] circuit max Re(lambda) = {circ_max_re:+.4f} 1/s (the law has no poles to "
              f"compare against)")
    else:
        verdict = ("matches the teacher's own boundary" if abs(gap) < 0.05 else
                   "SLOWER than the teacher" if gap < 0 else "FASTER-GROWING than the teacher")
        print(f"[test] max Re(lambda): circuit {circ_max_re:+.4f} vs slowest teacher "
              f"{true_max_re:+.4f} 1/s   gap {gap:+.4f} -- {verdict}")
    print(f"[test] wrote {p}")
    return res


# --------------------------------------------------------------------------- #
#  plot
# --------------------------------------------------------------------------- #
def _f(v, w=8):
    """`+0.0042`, or `n/a` -- a law with no poles (a delay, a static gain) has no real part to
    report, and formatting None crashed the figure after the run had already trained."""
    return f"{v:+.4f}" if isinstance(v, (int, float)) else "     n/a"


def _wrap_poles(label, freqs, per_line=6):
    """A pole list as several short lines. `t3_lowpass_order` has 14 of them across six condition
    cells, and one line ran three panel-widths off the right edge of the figure."""
    f = [f"{x:.3f}" for x in np.round(np.asarray(freqs, float).ravel(), 3)]
    if not f:
        return [f"{label} (none)"]
    pad = " " * len(label)
    return [f"{label if i == 0 else pad} {' '.join(f[i:i + per_line])}"
            + (" Hz" if i + per_line >= len(f) else "")
            for i in range(0, len(f), per_line)]


def plot(run, root=None, device=None, n_show=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks.render import BG, INK, MUTED, _ax

    dev = torch.device(device or "cpu")
    out = log_dir(run["name"], root)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=dev)
    model = CircuitRNN(ck["n_in"], ck["n_out"], hidden=int(ck["circuit"].get("hidden", 64)),
                       tau0=float(ck["circuit"].get("tau0", 0.5)), dt=ck["dt"]).to(dev)
    model.load_state_dict(ck["state"]); model.eval()
    split = "test" if os.path.isdir(os.path.join(task_dir(run["task"]), "test")) else "train"
    U, Y, cond = load_split(run["task"], split, dev)
    # REBUILT FROM THE CHECKPOINT, not from the run spec: the model's W_in has the width the fit
    # was made at, and a run spec edited between train and test would otherwise raise a shape
    # error here or, worse, line the one-hot up against a different number of cells.
    U = with_context(U, cond, int(ck.get("n_cond", 1)))
    with torch.no_grad():
        P, R = model(U, want_rates=True)
    rep = json.load(open(os.path.join(out, "results", "report.json")))
    dt, T = ck["dt"], U.shape[1]
    t = np.arange(T) * dt

    # 3 x 2. TOP ROW IS WHAT THE RUN WAS -- what went in, what came out, and the numbers.
    # BOTTOM ROW IS WHETHER IT WORKED -- the dynamics, the error, and the fit's own history.
    fig = plt.figure(figsize=(14.5, 7.6), facecolor=BG)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)

    # a: THE STIMULUS, which no earlier version of this figure showed. A reader could not tell a
    # band-limited noise task from a PRBS one, and `t4_unexcited_12hz` -- whose whole point is a
    # stimulus too slow to excite the teacher -- looked exactly like a task that works.
    axa = _ax(fig.add_subplot(gs[0, 0]), ylabel="input", letter="a")
    for i in range(min(n_show, U.shape[0])):
        axa.plot(t, U[i, :, 0].cpu(), color="#4a4a4a", lw=0.7, alpha=0.8)
    if U.shape[2] > 1:
        axa.text(0.985, 0.04, f"channel 1 of {U.shape[2]}", transform=axa.transAxes,
                 ha="right", va="bottom", color=MUTED, fontsize=8)

    # b: target against prediction. GREEN IS GROUND TRUTH AND BLACK IS THE PREDICTION, the
    # repository's convention; the teacher is drawn thicker and underneath so that a good fit
    # reads as a black line sitting inside a green one rather than as two lines that disagree.
    axb = _ax(fig.add_subplot(gs[0, 1]), ylabel="target / prediction", letter="b")
    for i in range(min(n_show, U.shape[0])):
        axb.plot(t, Y[i, :, 0].cpu(), color="#2e8b4f", lw=2.2, alpha=0.9)
        axb.plot(t, P[i, :, 0].cpu(), color="#000000", lw=0.9)
    axb.text(0.985, 0.04, "green: ground truth   black: circuit", transform=axb.transAxes,
             ha="right", va="bottom", color=MUTED, fontsize=8)

    # d: THE RECOVERY PANEL, and the only one that can tell a fit from a coincidence. Each point
    # is an eigenvalue lambda = sigma + i omega of the circuit's linearisation, so a mode of the
    # circuit behaves as exp(lambda t) = exp(sigma t)(cos omega t + i sin omega t):
    #
    #   Re(lambda), the x-axis in 1/s, is the ENVELOPE. Negative decays with time constant
    #     1/|sigma| seconds, positive diverges, zero neither -- which is what a perfect memory is.
    #     So -10 1/s is a 0.1 s transient and -0.1 1/s is a 10 s memory.
    #   Im(lambda)/2pi, the y-axis in Hz, is the RINGING inside that envelope. Zero is a pure
    #     exponential; nonzero oscillates at that many cycles per second.
    #
    # The panel is mirror-symmetric about y = 0 because the matrix is real and its complex
    # eigenvalues come in conjugate pairs -- a pair is ONE oscillating mode, not two. The dashed
    # red line at Re = 0 is the stability boundary.
    axd = _ax(fig.add_subplot(gs[1, 0]), xlabel="Re(lambda)  (1/s)", ylabel="Im/2pi  (Hz)",
              letter="d")
    got = model.jacobian_poles()
    axd.scatter(got.real, got.imag / (2 * np.pi), s=10, color="#000000", alpha=0.8)
    truth = torch.load(os.path.join(task_dir(run["task"]), "teacher.pt"), weights_only=False)
    for cellrec in truth["per_cell"]:
        axd.scatter(cellrec["poles_real"], np.asarray(cellrec["poles_imag"]) / (2 * np.pi),
                    s=80, marker="x", color="#2e8b4f", lw=2.0, zorder=5)
    axd.axvline(0, color="#c0272a", lw=0.8, ls="--")
    # CAPTIONED, NOT LEGENDED, like panel b. A legend box is placed in a corner of the axes and
    # this cloud fills the whole plane, so it sat on top of the data whichever corner it chose.
    axd.text(0.985, 0.04, "green: ground truth   black: circuit", transform=axd.transAxes,
             ha="right", va="bottom", color=MUTED, fontsize=8)

    # e: the residual, ON THE TARGET'S OWN SCALE, so "close" is a number and not an impression.
    axe = _ax(fig.add_subplot(gs[1, 1]), xlabel="time (s)", ylabel="residual", letter="e")
    for i in range(min(n_show, U.shape[0])):
        axe.plot(t, (P[i, :, 0] - Y[i, :, 0]).cpu(), color="#c0522a", lw=0.8)
    axe.set_ylim(axb.get_ylim())

    # f: the learning curve
    axf = _ax(fig.add_subplot(gs[1, 2]), xlabel="epoch", ylabel="mse", letter="f")
    ep = [h["epoch"] for h in rep["history"]]
    axf.plot(ep, [h["train_mse"] for h in rep["history"]], color="#1f6fb8", lw=1.1, label="train")
    axf.plot(ep, [h["val_mse"] for h in rep["history"]], color="#9a7d1a", lw=1.1, label="val")
    axf.set_yscale("log")
    lg = axf.legend(frameon=False, fontsize=8, loc="upper right")
    for txt in lg.get_texts():
        txt.set_color(INK)

    # c: the numbers. NO AXES AT ALL -- `_ax` drops the top and right spines, which is right for
    # a plot and wrong for words: a frame around text has no axis to be a frame for. The letter
    # is drawn in axes coordinates and survives turning them off.
    axc = _ax(fig.add_subplot(gs[0, 2]), letter="c")
    axc.axis("off")
    res_path = os.path.join(out, "results", f"{run['name']}_{split}.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    lines = ["", f"{run['name']}", f"task    {run['task']}", f"split   {split}",
             f"units   {model.hidden}    params {rep['n_params']}", "",
             f"best val mse   {rep['best_val_mse']:.6f}",
             f"{split} mse       {res.get('mse', float('nan')):.6f}",
             f"normalised     {res.get('normalised_mse', float('nan')):.4f}  of target variance",
             f"  settled      {res.get('normalised_mse_settled', float('nan')):.5f}  "
             f"(after {res.get('settle_s', 0.5):g} s from rest)",
             "",
             ] + _wrap_poles("ground-truth poles", res.get("teacher_pole_freq_hz", [])) + [
             f"circuit slowest{np.round(res.get('circuit_slowest_pole_freq_hz', [])[:4], 4).tolist()} Hz",
             f"max Re(lam)    circuit {_f(res.get('circuit_max_real_eig'))} 1/s",
             f"               truth   {_f(res.get('teacher_max_real_pole'))} 1/s",
             f"               gap     {_f(res.get('max_real_gap'))} 1/s",
             "",
             ("corpus IDENTIFIABLE" if res.get("identifiable", True) else
              "corpus NOT IDENTIFIABLE -- a low\nerror here does not mean the\ncircuit has the "
              "right dynamics")]
    axc.text(0.02, 0.95, "\n".join(lines), transform=axc.transAxes, va="top", ha="left",
             color=INK, fontsize=8, family="monospace")

    p = os.path.join(out, "results", f"{run['name']}_{split}.png")
    fig.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {p}")
    return p


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # PHASES AND SPECS SHARE `-o`, as GNN_Main.py does: `-o train test plot <spec>`. They are
    # told apart by membership, not by position, so the order a reader writes them in does not
    # matter and a typo'd phase becomes an unreadable path rather than being silently dropped.
    ap.add_argument("-o", "--option", nargs="+", default=["train"],
                    help="phases (train / test / plot) and the run spec(s), in any order")
    ap.add_argument("--root", default=None, help="overrides the log/ root")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    fns = {"train": train, "test": test, "plot": plot}
    phases = [o for o in a.option if o in fns] or ["train"]
    specs = [o for o in a.option if o not in fns]
    if not specs:
        ap.error("no run spec given. Usage: -o train test plot config/run/<name>.yaml")
    paths = [p for s in specs for p in (glob.glob(s) if any(c in s for c in "*?[") else [s])]
    for p in paths:
        if not os.path.exists(p):
            ap.error(f"{p!r} is neither a phase (train / test / plot) nor an existing run spec")
    for p in paths:
        run = load_run(p)
        for phase in phases:
            fns[phase](run, root=a.root, device=a.device)


if __name__ == "__main__":
    main()
