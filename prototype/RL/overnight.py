"""overnight.py -- run the full RL-prep battery unattended.

Three phases, each resumable and each writing its own artifacts under runs/<stamp>/:
  1. characterize  -- reward signal-to-noise + identifiability (fast, once)
  2. baseline      -- CEM inversion over many targets (the bar RL must beat)
  3. dataset       -- (u, observation) pairs for the recognition net / RL warm-start

The battery is time-boxed by --hours; whatever has been written is valid and
resumable (re-run to continue). Everything uses the codebase operators + engine.

    PYTHONPATH=../../src nohup python overnight.py --hours 10 > runs/overnight.log 2>&1 &
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))


def run_phase(name, argv, deadline):
    if time.time() > deadline:
        print(f"[overnight] skip {name}: out of time", flush=True)
        return
    print(f"[overnight] === {name} ===", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "src")
    p = subprocess.Popen([sys.executable] + argv, cwd=_HERE, env=env)
    while p.poll() is None:
        if time.time() > deadline:
            print(f"[overnight] time up -> terminating {name}", flush=True)
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            break
        time.sleep(5)
    print(f"[overnight] {name} exited ({p.returncode})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--out", default=os.path.join(_HERE, "runs", "overnight"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-targets", type=int, default=60)
    ap.add_argument("--n-dataset", type=int, default=40000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    deadline = time.time() + args.hours * 3600
    print(f"[overnight] start; deadline in {args.hours}h; out={args.out}", flush=True)

    # 1. characterize the reward (cheap, gives SNR + degeneracy up front)
    run_phase("characterize",
              ["test_characterize_metric.py", "--device", args.device,
               "--out", os.path.join(args.out, "characterize")], deadline)

    # 2. CEM inversion baseline (the bar) -- give it ~40% of the budget
    run_phase("baseline",
              ["test_inverse_baseline.py", "--device", args.device, "--n", str(args.n_targets),
               "--iters", "20", "--batch", "24", "--out", os.path.join(args.out, "baseline")],
              min(deadline, time.time() + 0.4 * args.hours * 3600))

    # 3. dataset for the recognition net -- the rest of the night
    run_phase("dataset",
              ["build_dataset.py", "--device", args.device, "--n", str(args.n_dataset),
               "--out", os.path.join(args.out, "dataset")], deadline)

    print("[overnight] battery complete (or time-boxed). Artifacts under "
          f"{args.out}", flush=True)


if __name__ == "__main__":
    main()
