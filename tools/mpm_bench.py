#!/usr/bin/env python
"""MPM throughput benchmark: ms/frame AND effective memory bandwidth, swept over particle count.

WHY BANDWIDTH AND NOT JUST WALL CLOCK. Explicit MPM is a memory-bound algorithm -- Wyser et al.
(GMD 14:7749, 2021) make the point and report 52-88% of peak on V100/A100 -- so "ms/frame" alone
cannot say whether a run is slow because the problem is big or because the implementation is bad.
Effective throughput can, and it is comparable across cards and across implementations. Measured on
this repo's PyTorch operators it comes out at 0.8% of an A6000's peak, which is the whole case for
a fused kernel.

WHAT COUNTS AS "ESSENTIAL" TRAFFIC. Per particle per substep, the MLS-MPM cycle must at minimum:
read pos/vel/C/F/mass/mu/la (27 floats), scatter mass and momentum to 27 neighbours (27*4), and
gather velocity back from 27 (27*3). That is 216 floats = 864 B. It is a LOWER BOUND on the memory
a correct implementation must move, so the ratio it produces is a floor on the achievable speedup:
an implementation at 0.8% of peak is leaving at least that much on the table. It deliberately does
NOT count the intermediates PyTorch materialises -- counting those would flatter the current code
by pretending its extra traffic was necessary.

    python tools/mpm_bench.py --spec config/material/material_3d_water_lots_x10.yaml \
                              --sizes 100000,350000,945000 --frames 30 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

FLOATS_PER_PARTICLE_SUBSTEP = 3 + 3 + 9 + 9 + 1 + 1 + 1 + 27 * 4 + 27 * 3   # = 216


def _particle_sets(spec):
    return [n for n, st in (spec.get("sets") or {}).items()
            if isinstance(st, dict) and "per_parent" in st]


def run_one(spec_path, n_particles, frames, warmup, device, capture, compile_on, impl=None):
    import yaml
    import torch
    import plexus.operators  # noqa: F401
    from plexus.schema import load
    from plexus import engine as E

    spec = yaml.safe_load(open(spec_path))
    psets = _particle_sets(spec)
    if not psets:
        raise SystemExit(f"  {spec_path} has no contained particle set")
    # scale every particle set proportionally, keeping the spec's composition
    base = sum(spec["sets"][p]["per_parent"] * int(spec["sets"][spec["sets"][p]["parent"]].get("n", 1))
               for p in psets)
    if n_particles:
        k = n_particles / base
        for p in psets:
            spec["sets"][p]["per_parent"] = max(1, int(round(spec["sets"][p]["per_parent"] * k)))
    total = sum(spec["sets"][p]["per_parent"] * int(spec["sets"][spec["sets"][p]["parent"]].get("n", 1))
                for p in psets)

    for o in spec["operators"]:
        if o.get("op") in ("mpm_scatter", "mpm_gather") and impl:
            # `--impl` OVERRIDES; absent leaves the spec alone, so a spec that DECLARES an
            # implementation is benchmarked as written. `--impl default` forces the torch path.
            o.pop("implementation", None)
            if impl != "default":
                o["implementation"] = impl
                if o["op"] == "mpm_scatter":
                    o.setdefault("polar", "higham")
    blk = next((s for s in spec["schedule"] if isinstance(s, dict) and "substep_dt" in s), None)
    if blk is not None:
        blk.pop("capture", None); blk.pop("compile", None)
        if capture:
            blk["capture"] = True
        else:
            blk["capture"] = False
        if compile_on:
            blk["compile"] = True
    substeps = max(1, round(float(spec["general"]["dt"]) / float(blk["substep_dt"]))) if blk else 1
    spec["general"]["n_frames"] = frames + warmup
    spec["general"]["record_cap"] = 2                    # the recorder is not the subject

    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(spec, f); f.close()
    sim = load(f.name); os.unlink(f.name)

    marks = {}

    def on_frame(H, tick):
        if tick == warmup:
            torch.cuda.synchronize(device); marks["a"] = time.perf_counter()
        elif tick == warmup + frames:
            torch.cuda.synchronize(device); marks["b"] = time.perf_counter()

    torch.cuda.reset_peak_memory_stats(device)
    E.run(sim, out_path=None, device=device, on_frame=on_frame, progress=False)
    ms = (marks["b"] - marks["a"]) / frames * 1000
    gb_frame = total * FLOATS_PER_PARTICLE_SUBSTEP * 4 * substeps / 1e9
    return {
        "particles": total, "substeps": substeps, "ms_per_frame": ms,
        "ms_per_substep": ms / substeps,
        "gb_per_frame": gb_frame, "gb_per_s": gb_frame / (ms / 1000),
        "peak_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="config/material/material_3d_water_lots_x10.yaml")
    ap.add_argument("--sizes", default="100000,350000,945000",
                    help="comma-separated TOTAL particle counts; 0 = the spec's own")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--capture", action="store_true", help="CUDA-graph capture the substep")
    ap.add_argument("--compile", dest="compile_on", action="store_true")
    ap.add_argument("--impl", default=None,
                    help="implementation for mpm_scatter AND mpm_gather, e.g. warp")
    ap.add_argument("--json", default=None, help="also write the rows here")
    a = ap.parse_args()

    import torch
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    spec_path = a.spec if os.path.isabs(a.spec) else os.path.join(root, a.spec)
    p = torch.cuda.get_device_properties(a.device)
    # published peak bandwidth, GB/s -- the denominator for the % column
    PEAK = {"A100": 1555, "A6000": 768, "H100": 3350, "L4": 300, "V100": 900, "T4": 320}
    peak = next((v for k, v in PEAK.items() if k in p.name), None)

    print(f"\n  {p.name}  {p.multi_processor_count} SMs  {p.total_memory/2**30:.0f} GiB"
          f"{f'  peak ~{peak} GB/s' if peak else ''}")
    print(f"  {os.path.basename(spec_path)}   capture={a.capture} compile={a.compile_on} impl={a.impl or 'default'}   "
          f"torch {torch.__version__}   {platform.node()}")
    hdr = f"\n  {'particles':>11}{'sub':>5}{'ms/frame':>11}{'ms/substep':>12}{'GB/s':>9}"
    print(hdr + (f"{'% peak':>8}" if peak else "") + f"{'peak GiB':>10}")
    print("  " + "-" * (len(hdr) + 18))
    rows = []
    for s in [int(x) for x in a.sizes.split(",")]:
        try:
            r = run_one(spec_path, s, a.frames, a.warmup, a.device, a.capture, a.compile_on,
                        impl=a.impl)
        except Exception as e:                       # OOM is a result, not a crash
            print(f"  {s:>11,}   {type(e).__name__}: {str(e).splitlines()[0][:52]}", flush=True)
            continue
        r["device"] = p.name
        rows.append(r)
        pct = f"{r['gb_per_s']/peak*100:>7.1f}%" if peak else ""
        print(f"  {r['particles']:>11,}{r['substeps']:>5}{r['ms_per_frame']:>11.1f}"
              f"{r['ms_per_substep']:>12.2f}{r['gb_per_s']:>9.1f}{pct}{r['peak_gib']:>10.2f}",
              flush=True)
    if a.json and rows:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\n  rows -> {a.json}")
    print()


if __name__ == "__main__":
    main()
