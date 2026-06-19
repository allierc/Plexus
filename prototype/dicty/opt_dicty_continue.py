"""opt_dicty_continue.py -- RESUMABLE dicty optimizer (continues a prior search).

Same UCB / kNN-surrogate search and combined density + PIV-velocity loss as
opt_dicty.py, but it CONTINUES instead of starting cold:

  * it persists the full evaluation history to `opt_state.npz` (X in [0,1]^D, Y=-loss)
    and the winners list + counters to `opt_state.json`, saving every few evals and on
    exit -- so each launch picks up exactly where the last one stopped;
  * on a FRESH start (no state) it warm-starts the surrogate from the existing
    `specs/dicty_opt_winner_*.yaml` (re-evaluated under the CURRENT targets) plus a few
    random probes, so prior winners seed the search rather than being thrown away.

Winner specs/gifs keep incrementing (`dicty_opt_winner_<k>`), and WINNERS_dicty.md is
rebuilt from the persisted winners list.

CLI:
    cd prototype/dicty
    # continue the search for 1 hour (repeat the command to keep going):
    PYTHONPATH=../../src python opt_dicty_continue.py 3600
    # run until you kill it (Ctrl-C; state is saved on exit):
    PYTHONPATH=../../src python opt_dicty_continue.py
    # start over from scratch (wipe history + winners):
    PYTHONPATH=../../src python opt_dicty_continue.py --reset 3600
    # pick GPU:
    DEVICE=cuda:1 PYTHONPATH=../../src python opt_dicty_continue.py 3600
"""
from __future__ import annotations

import os
import sys
import glob
import json
import time
import signal
import numpy as np
import yaml

# reuse everything from the base optimizer (paths, chdir, loss, bounds, render, index)
import opt_dicty as base
from opt_dicty import (BOUNDS, PARAMS, D, IMPROVE, evaluate, render_winner,
                       render_winner_gif, write_index, _target)

STATE_NPZ = "opt_state.npz"
STATE_JSON = "opt_state.json"
SAVE_EVERY = 5

# Best-guess starting configs (physical units, in PARAMS order) -- the search is SEEDED here
# (not resumed) so UCB starts in a sensible region and refines. Index 1 is now INFLOW.RATE
# (cells entering the volume / frame, ~1.5-2.5 to grow 767 -> ~1400 like the movie), NOT
# division. Guided by the diagnostics: strong chemotaxis + strong long-range adhesion (kadh,
# r_on) to coalesce the many small mounds into few large ones.
#          n   inflow gain sec  diff   decay krep  r0    kadh  r_on  muf   rw    dt
SEEDS = [
    [767, 2.0, 35, 8.0, 0.010, 0.25, 55, 0.025, 180, 0.25, 0.05, 0.002, 0.60],  # strong adh + chemo
    [600, 1.5, 25, 8.0, 0.020, 0.20, 50, 0.020, 180, 0.25, 0.05, 0.003, 0.60],  # long-range field
    [767, 2.5, 39, 7.0, 0.004, 0.29, 63, 0.037, 167, 0.25, 0.03, 0.000, 0.66],  # strong chemo, low diffusion
    [600, 1.6, 30, 6.0, 0.040, 0.20, 50, 0.020, 120, 0.20, 0.05, 0.003, 0.55],  # long-range cAMP -> fewer mounds
    [767, 1.8, 30, 9.0, 0.006, 0.25, 60, 0.022, 150, 0.22, 0.05, 0.002, 0.60],  # tight mounds
]


def _to_u(p):
    return np.clip((np.array(p, float) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0]), 0.0, 1.0)


def winner_to_u(path):
    """Invert a winner spec back to a normalized parameter vector u in [0,1]^D."""
    raw = yaml.safe_load(open(path))
    o = {op["op"]: op for op in raw["operators"]}
    sp = o["spring"]
    p = np.array([
        raw["sets"]["cell"]["n"],
        o.get("inflow", o.get("divide", {})).get("rate", 0.0), o["sense"]["gain"], o["secrete"]["rate"],
        raw["fields"]["camp"]["diffusion"], raw["fields"]["camp"]["decay"],
        sp["k_rep"], sp["r0"], sp.get("kadh", 0.0), sp.get("r_on", sp["r0"]), sp["mu_f"],
        o["random_walk"]["strength"], raw["dt"],
    ], float)
    u = (p - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    return np.clip(u, 0.0, 1.0)


def load_state():
    if not (os.path.exists(STATE_NPZ) and os.path.exists(STATE_JSON)):
        return None
    z = np.load(STATE_NPZ)
    meta = json.load(open(STATE_JSON))
    return list(z["X"]), list(z["Y"]), meta["winners"], meta["best"], meta["winners_list"]


def save_state(X, Y, winners, best, winners_list):
    np.savez(STATE_NPZ, X=np.array(X), Y=np.array(Y))
    json.dump({"winners": winners, "best": best, "winners_list": winners_list},
              open(STATE_JSON, "w"))


def main(budget=float("inf")):
    rng = np.random.default_rng(0)
    t_frac, target, vtarget = _target()
    t0 = time.time()

    st = load_state()
    if st is None:
        X, Y, winners, best, winners_list = [], [], 0, 1e9, []
    else:
        X, Y, winners, best, winners_list = st
        print(f"[resume] {len(X)} prior evals, {winners} winners, best loss={best:.5f}", flush=True)

    def maybe_winner(u, loss):
        nonlocal winners, best
        if loss > best - IMPROVE:
            return
        winners += 1; gif = f"dicty_opt_winner_{winners}.gif"
        render_winner_gif(u, winners, loss)
        p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
        winners_list.append({"k": winners, "loss": float(loss), "params": [float(v) for v in p], "gif": gif})
        write_index(winners_list); best = loss
        print(f"[WINNER {winners}] loss={loss:.5f} -> {gif}  "
              + ", ".join(f"{k}={v:.3f}" for k, v in zip(PARAMS, p)), flush=True)

    # save on Ctrl-C / SIGTERM too
    def _onsig(*_):
        save_state(X, Y, winners, best, winners_list)
        print(f"\n[saved] {len(X)} evals -> {STATE_NPZ} (relaunch to continue)", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGINT, _onsig); signal.signal(signal.SIGTERM, _onsig)

    if st is None:                                          # SEED the search with best guesses
        print(f"[seed] evaluating {len(SEEDS)} best-guess configs + random probes", flush=True)
        for sp in SEEDS:
            u = _to_u(sp); loss = evaluate(u, t_frac, target, vtarget)
            X.append(u); Y.append(-loss)
            print(f"  seed loss={loss:.5f}", flush=True)
            maybe_winner(u, loss)
        for _ in range(3):                                  # a few random probes for surrogate spread
            u = rng.random(D); loss = evaluate(u, t_frac, target, vtarget)
            X.append(u); Y.append(-loss); maybe_winner(u, loss)

    print(f"[run] UCB ({D} levers, density+PIV loss) on {base.DEVICE}, budget={budget}s", flush=True)

    n_since_save = 0
    while time.time() - t0 < budget:
        Xa, Ya = np.array(X), np.array(Y)
        best_u = Xa[Ya.argmax()]
        glob_c = rng.random((2500, D))
        loc = np.clip(best_u + rng.normal(0, 0.07, (2500, D)), 0, 1)
        cand = np.vstack([glob_c, loc])
        d = np.linalg.norm(cand[:, None, :] - Xa[None, :, :], axis=2)
        w = np.exp(-(d / 0.22) ** 2) + 1e-9
        mean = (w * Ya[None, :]).sum(1) / w.sum(1)
        unc = d.min(1)
        scale = (Ya.max() - Ya.min()) or 1.0
        u = cand[(mean + 0.6 * scale * unc).argmax()]       # UCB acquisition
        loss = evaluate(u, t_frac, target, vtarget); X.append(u); Y.append(-loss)
        print(f"  eval {len(Y):4d}: loss={loss:.5f}  best={min(best, loss):.5f}  ({time.time()-t0:.0f}s)", flush=True)
        maybe_winner(u, loss)
        n_since_save += 1
        if n_since_save >= SAVE_EVERY:
            save_state(X, Y, winners, best, winners_list); n_since_save = 0

    save_state(X, Y, winners, best, winners_list)
    print(f"[DONE] {len(Y)} total evals, {winners} winners, best loss={best:.5f}  "
          f"(state saved -> relaunch to continue)", flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--reset" in argv:
        for f in (STATE_NPZ, STATE_JSON):
            if os.path.exists(f):
                os.remove(f)
        print("[reset] cleared optimizer state (winners kept; delete specs/dicty_opt_winner_* to wipe those)", flush=True)
        argv = [a for a in argv if a != "--reset"]
    budget = float(argv[0]) if argv and argv[0].replace('.', '').isdigit() else float("inf")
    main(budget)
