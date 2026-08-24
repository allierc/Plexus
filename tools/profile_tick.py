#!/usr/bin/env python
"""Where does a generate actually spend its time? Per-operator, integrate, record, snapshot.

WHY A SEPARATE HARNESS AND NOT `-o generate --profile`. The question is not "is it slow" -- the
tqdm bar answers that -- but "slow AT WHAT", and the honest answer needs the GPU to be synchronised
at every boundary. A CUDA kernel launch RETURNS IMMEDIATELY; without a `synchronize` between two
operators the second one's timer collects the first one's work, and the profile blames whichever
call happens to be followed by something that blocks. Synchronising costs real time, so the run
being measured is deliberately not the run being shipped, and the total is reported both ways.

INIT IS NOT COUNTED. Building the hierarchy, seeding 1,500 particles, allocating the 96^3 grid and
importing torch are all one-time and none of them repeat per frame; folded into a 100-frame average
they invent a per-frame cost that does not exist. The clock starts after `--warmup` frames, which
also lets CUDA finish its lazy kernel compilation.

    python tools/profile_tick.py cell/cell_02_nucleus_bounce --frames 100 --device cuda:1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--frames", type=int, default=100, help="frames TIMED, after the warm-up")
    ap.add_argument("--warmup", type=int, default=10, help="frames run and discarded first")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--no-sync", action="store_true",
                    help="skip cuda synchronisation -- a faster but unattributable total")
    a = ap.parse_args()

    import plexus.operators  # noqa: F401  self-register the operator library
    from plexus.schema import load
    from plexus.paths import resolve_config
    from plexus import engine as E
    from plexus.models import registry as R

    yaml_file, pre_folder, name = resolve_config(a.config)
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition
    Courant_Friedrichs_Lewy_condition(yaml_file)
    sim = load(yaml_file)
    sim.n_frames = a.frames + a.warmup

    dev = a.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] {dev} unavailable -> cpu")
        dev = "cpu"
    cuda = dev.startswith("cuda")
    sync = (lambda: torch.cuda.synchronize(dev)) if (cuda and not a.no_sync) else (lambda: None)

    # --------------------------------------------------------------- the clocks
    t_op = defaultdict(float)                 # operator token -> seconds
    n_op = defaultdict(int)
    phase = defaultdict(float)                # integrate / record / snapshot / substep_bookkeeping
    state = {"on": False, "t_loop": 0.0}

    # WRAP THE CLASSES, NOT THE INSTANCES. The engine constructs its operators inside `run`, so
    # there is nothing to wrap until it is too late; the registry's classes exist now.
    seen = set()
    for contract in R._OP_CONTRACTS.values():
        for cls in {contract.get()} | set(getattr(contract, "_impls", {}).values() if
                                          isinstance(getattr(contract, "_impls", None), dict) else []):
            if cls in seen or not hasattr(cls, "forward"):
                continue
            seen.add(cls)
            orig, label = cls.forward, cls.__name__

            def timed(self, *args, _o=orig, _l=label, **kw):
                if not state["on"]:
                    return _o(self, *args, **kw)
                sync(); t0 = time.perf_counter()
                r = _o(self, *args, **kw)
                sync()
                t_op[_l] += time.perf_counter() - t0
                n_op[_l] += 1
                return r
            cls.forward = timed

    def wrap_phase(mod, fn, key):
        orig = getattr(mod, fn)

        def timed(*args, _o=orig, _k=key, **kw):
            if not state["on"]:
                return _o(*args, **kw)
            sync(); t0 = time.perf_counter()
            r = _o(*args, **kw)
            sync()
            phase[_k] += time.perf_counter() - t0
            return r
        setattr(mod, fn, timed)
        return orig

    wrap_phase(E, "_integrate", "integrate")

    from plexus import live as L
    wrap_phase(L, "snapshot", "live snapshot (3d.png)")

    # THE WARM-UP BOUNDARY. `on_frame` is called by the engine once per tick, which is the only
    # hook into the loop that does not mean re-implementing it here.
    marks = {}

    def on_frame(H, tick):
        if tick == a.warmup:
            sync()
            state["on"] = True
            marks["t0"] = time.perf_counter()
        elif tick == a.warmup + a.frames:
            sync()
            marks["t1"] = time.perf_counter()
            state["on"] = False

    print(f"[profile] {name}  device={dev}  sync={'off' if a.no_sync else 'on'}  "
          f"{a.warmup} warm-up + {a.frames} timed frames", flush=True)
    E.run(sim, out_path=None, device=dev, on_frame=on_frame, progress=False)
    if "t1" not in marks:
        marks["t1"] = time.perf_counter()
    total = marks["t1"] - marks["t0"]

    # --------------------------------------------------------------- the report
    F = a.frames
    rows = sorted(t_op.items(), key=lambda kv: -kv[1]) + sorted(phase.items(), key=lambda kv: -kv[1])
    acct = sum(v for _k, v in rows)
    rows.append(("OTHER (python loop, delta clones, dict ops)", total - acct))
    print(f"\n  {'component':<44} {'ms/frame':>10} {'calls/frame':>12} {'share':>8}")
    print("  " + "-" * 78)
    for k, v in rows:
        c = n_op.get(k, 0) / F
        print(f"  {k:<44} {v / F * 1000:10.2f} {c:12.1f} {v / total * 100:7.1f}%")
    print("  " + "-" * 78)
    print(f"  {'TOTAL (timed loop)':<44} {total / F * 1000:10.2f} {'':12} {100.0:7.1f}%")
    print(f"\n  {F} frames in {total:.1f} s  ->  {F / total:.2f} frame/s"
          + ("   (synchronised; the shipped run is faster)" if not a.no_sync and cuda else ""))

    g = (sim.fields or {}).get("mpm_grid") or {}
    ng = int(g.get("n_grid", 0))
    npart = sum(int(v.get("per_parent", 0)) * int((sim.sets.get(v.get("parent"), {}) or {}).get("n", 1))
                for v in sim.sets.values() if v.get("parent"))
    if ng:
        sub = max(1, round(sim.dt / float(next(
            (s["substep_dt"] for s in sim.schedule if isinstance(s, dict) and "substep_dt" in s),
            sim.dt))))
        print(f"\n  the two sizes that set the cost:")
        print(f"    particles           {npart:>12,}")
        print(f"    grid nodes          {ng ** 3:>12,}   ({ng}^3)")
        print(f"    ratio               {ng ** 3 / max(npart, 1):>12,.0f} grid nodes per particle")
        print(f"    substeps per frame  {sub:>12}")
        print(f"    grid-node updates   {ng ** 3 * sub:>12,} per frame")


if __name__ == "__main__":
    main()
