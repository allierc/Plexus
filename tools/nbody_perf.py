#!/usr/bin/env python
"""ALL-PAIRS GRAVITY BENCHMARK: one particle count, one implementation, one row.

    python tools/nbody_perf.py --sweep                      # write the config/inverse_square/nb_perf_* specs
    python tools/nbody_perf.py --run nb_perf_500k_torch --device cuda:0
    python tools/nbody_perf.py --table

WHAT IS BEING MEASURED. `squared_law[all_pairs: true]` is an O(N^2) reduction: every particle pulls
on every other, so a step is N^2 interactions no matter how it is written. What DOES change with the
implementation is how much memory that reduction touches, and that is the whole question here --
the same one the MPM warp work answered. `_inv_square_sum` builds the [N, N] separation matrices
explicitly:

    r2      [N, N] fp32          4 N^2 bytes
    dk      [N, N] per axis      4 N^2 bytes, built TWICE per axis (once into r2, once into pull)
    inv_r3  [N, N] fp32          4 N^2 bytes

so the working set is ~12 N^2 bytes -- 7.5 GB at 25,000 particles, 120 GB at 100,000, and 12 TB at
1,000,000. The arithmetic is ~20 flop per pair; the traffic is ~12 byte per pair. At an A100's 19.5
TFLOP/s fp32 against 1.55 TB/s that is 1.0 us of compute per 0.6 us of traffic per million pairs --
so the kernel is not compute-bound, it is bound by writing down numbers it uses once and discards.

A tiled kernel does the same N^2 arithmetic and touches O(N): each thread holds one receiver's
accumulator in registers and streams sources through shared memory. That is the comparison this tool
exists to make, and the particle counts are chosen so the torch path FAILS on two of them -- an
out-of-memory row is a result, not a gap, and it is reported as one.

WHAT IS TIMED. Frames `warmup+1 .. warmup+timed`, `cuda.synchronize()` at each end, no renderer
(`out_path=None`), no recording. Process start, hierarchy build and the first-call compile are all
outside the window. `--compile` rows pay a one-off torch.compile which `warmup` must cover.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import yaml

BASE = "galaxy_collision_3d"
FOLDER = "inverse_square"
SIZES = {"25k": 25_000, "100k": 100_000, "500k": 500_000, "1m": 1_000_000}
ROWS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "graphs_data",
                    "_perf", "nbody_rows.jsonl")


class _Enough(Exception):
    pass


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _spec_path(name):
    return os.path.join(_root(), "config", FOLDER, f"{name}.yaml")


def write_sweep(impls=("torch", "compile", "warp")):
    """One spec per (size, implementation). The disc MASS is held fixed as N grows."""
    src = yaml.safe_load(open(_spec_path(BASE)))
    made = []
    for tag, n in SIZES.items():
        for impl in impls:
            s = yaml.safe_load(open(_spec_path(BASE)))
            name = f"nb_perf_{tag}_{impl}"
            s["general"]["name"] = name
            s["general"]["n_frames"] = 60
            s["general"]["save_data"] = False
            s["general"].pop("record_cap", None)
            s["sets"]["star"]["n"] = int(n)
            # PER-STAR MASS SCALES AS 1/N so the two discs keep the mass they had at 25,000 -- the
            # orbit, and therefore the timestep that resolves it, must not change underneath a
            # benchmark whose whole point is that only N changed.
            m0 = float(src["sets"]["star"]["types"]["red"]["mass"]) * src["sets"]["star"]["n"] / n
            for t in s["sets"]["star"]["types"].values():
                t["mass"] = float(f"{m0:.6g}")
            op = [o for o in s["operators"] if o.get("op") == "squared_law"][0]
            op["compile"] = (impl == "compile")
            if impl == "warp":
                op["implementation"] = "warp"
            else:
                op.pop("implementation", None)
            s.pop("descriptions", None)
            s["plotting"] = {"renderer": "none"}
            with open(_spec_path(name), "w") as f:
                yaml.safe_dump(s, f, sort_keys=False)
            made.append((name, n, impl))
    for name, n, impl in made:
        print(f"  {name:<24} {n:>9,} stars   squared_law[{impl}]")
    return made


def run_one(name, device, timed=10, warmup=6):
    import torch

    import plexus.operators                                          # noqa: F401
    from plexus.engine import run
    from plexus.schema import load

    sim = load(_spec_path(name))
    times, state = [], {}
    n_part = int(sim.sets["star"]["n"])

    def hook(H, tick):
        if tick == warmup:
            torch.cuda.synchronize(device)
            state["t0"] = time.perf_counter()
            state["peak_pre"] = torch.cuda.max_memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        elif tick > warmup:
            if tick >= warmup + timed:
                torch.cuda.synchronize(device)
                state["t1"] = time.perf_counter()
                state["peak"] = max(state["peak_pre"], torch.cuda.max_memory_allocated(device))
                raise _Enough()

    buf = io.StringIO()
    err = None
    try:
        with contextlib.redirect_stdout(buf):
            run(sim, out_path=None, device=device, progress=False, on_frame=hook)
    except _Enough:
        pass
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # AN OOM IS A ROW. At 100,000 stars the [N, N] intermediates are 120 GB and no A100 holds
        # them; recording that is the measurement, and dropping the row would leave a table that
        # looks like the torch path merely was not tried.
        err = "OOM" if "out of memory" in str(e).lower() else f"{type(e).__name__}"
    if err is None and "t1" not in state:
        err = "ended early"

    row = {"name": name, "n": n_part, "device": torch.cuda.get_device_name(device),
           "host": platform.node(), "when": time.strftime("%Y-%m-%d %H:%M")}
    if err:
        row["error"] = err
        row["ms_per_step"] = None
        row["peak_GB"] = None
    else:
        wall = state["t1"] - state["t0"]
        row["ms_per_step"] = 1000.0 * wall / timed
        row["peak_GB"] = state["peak"] / 1024 ** 3
        row["pairs_per_s"] = n_part * n_part * timed / wall
    os.makedirs(os.path.dirname(os.path.abspath(ROWS)), exist_ok=True)
    with open(ROWS, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  {name:<24} {n_part:>9,}  " +
          (f"{err}" if err else
           f"{row['ms_per_step']:9.1f} ms/step  {row['peak_GB']:7.2f} GB peak  "
           f"{row['pairs_per_s'] / 1e9:8.1f} G pair/s"), flush=True)
    return row


def table():
    if not os.path.isfile(ROWS):
        print("no rows yet"); return
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    best = {}
    for r in rows:                                          # last row per (name, device) wins
        best[(r["name"], r["device"])] = r
    impls = ["torch", "compile", "warp"]
    for dev in sorted({k[1] for k in best}):
        print(f"\n  {dev}")
        print(f"  {'stars':>9} " + "".join(f"{i:>26}" for i in impls))
        for tag, n in SIZES.items():
            cells = []
            for i in impls:
                r = best.get((f"nb_perf_{tag}_{i}", dev))
                if r is None:
                    cells.append(f"{'-':>26}")
                elif r.get("error"):
                    cells.append(f"{r['error']:>26}")
                else:
                    cells.append(f"{r['ms_per_step']:>13.1f} ms {r['peak_GB']:>7.2f} GB ")
            print(f"  {n:>9,} " + "".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--timed", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.sweep:
        write_sweep()
    if a.run:
        run_one(a.run, a.device, a.timed, a.warmup)
    if a.table:
        table()


if __name__ == "__main__":
    main()
