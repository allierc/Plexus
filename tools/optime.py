"""Where a frame's time goes, per operator: the instrument behind the apicobasal warp port.

    python tools/optime.py tissue/spheroid_organelles_nucleus [--device cuda:0] [--frames N]

Runs the spec through `engine.run` with NO rendering and NO recording to disk, with every
operator's `forward` wrapped in a synchronised wall-clock timer (a `torch.cuda.synchronize()`
on both sides, so GPU work an operator queued is charged to it and not to whoever syncs next).
Prints ms/frame per operator class, calls and ms/call, and the remainder -- what the engine
itself spends per tick on integration and the trajectory record.

THE SYNCS MAKE THE TOTAL A LITTLE WORSE THAN A REAL RUN (the run overlaps CPU and GPU where it
can; this forbids it), so read the table for the SHARES and `-o generate --no-viz` for the number.
On `spheroid_organelles_nucleus` (2026-09-11) it read 243 ms/frame of a 265 ms frame in
`cell_mechanics[apicobasal]` -- thirty autograd passes over ~1,400 vertices -- which is what the
warp gradient in `vertex_ops.apicobasal_energy_grad_warp` was written against.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch                                                   # noqa: E402

import plexus.operators                                        # noqa: E402,F401  registers everything
from plexus import engine                                      # noqa: E402
from plexus.models import registry as R                        # noqa: E402
from plexus.paths import resolve_config                        # noqa: E402
from plexus.schema import load                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=None, help="override general.n_frames")
    args = ap.parse_args()

    T = collections.defaultdict(float)
    N = collections.defaultdict(int)
    sync = torch.cuda.synchronize if args.device.startswith("cuda") else (lambda: None)

    def wrap(cls):
        f = cls.__dict__.get("forward")               # this class's OWN forward, not an inherited one
        if f is None or getattr(f, "_optime", False):
            return

        def timed(self, H, mask=None):
            sync(); t = time.perf_counter()
            out = f(self, H, mask)
            sync(); T[cls.__name__] += time.perf_counter() - t; N[cls.__name__] += 1
            return out
        timed._optime = True
        cls.forward = timed

    seen = set()
    for c in R._OP_CONTRACTS.values():
        for impl in getattr(c, "implementations", {}).values():
            if impl not in seen:
                seen.add(impl); wrap(impl)
    for cls in R._OPERATOR_REGISTRY.values():
        if cls not in seen:
            seen.add(cls); wrap(cls)

    yaml_file, _pre, _name = resolve_config(args.config)
    sim = load(yaml_file)
    if args.frames:
        sim.n_frames = args.frames; sim.record_cap = args.frames + 2
    sync(); t0 = time.perf_counter()
    _H, out = engine.run(sim, None, args.device, progress=False)
    sync(); total = time.perf_counter() - t0
    fm = out["frame_ms"]; nf = len(fm)
    print(f"\n{sim.name}: {nf} frames, loop {fm.sum() / 1e3:.1f} s = {fm.mean():.1f} ms/frame "
          f"(last 100: {fm[-100:].mean():.1f}); run() {total:.1f} s incl. setup")
    used = 0.0
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s} {v * 1e3 / nf:8.2f} ms/frame   {N[k]:5d} calls  {v * 1e3 / N[k]:8.2f} ms/call")
        used += v
    print(f"  {'<engine: integrate + record>':30s} {(fm.sum() / 1e3 - used) * 1e3 / nf:8.2f} ms/frame")


if __name__ == "__main__":
    main()
