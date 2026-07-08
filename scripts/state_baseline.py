#!/usr/bin/env python
"""Byte-identical regression harness for the StateSchema refactor (PR 1).

The refactor makes engine build/integrate/record schema-driven. The invariant is:
*existing Plexus specs must not know this refactor happened.* This script proves it.

Rule (before any schema work): baseline generation must be deterministic and
reproducible twice in the same checkout, or we would chase nondeterminism as if it
were a refactor bug. So the sequence is:

  gate     run a cheap representative set (1 spatial / 1 boids / 1 slime-field /
           1 material-MPM) TWICE on CPU and confirm identical hashes. This proves
           determinism before any baseline is trusted.
  capture  run every config/**.yaml for a short prefix and hash pos + occ + field
           grids + (when present) MPM substep buffers. Write baseline_hashes.json.
  compare  re-run and diff against the baseline; exit non-zero on any divergence.

Run `gate` first, then `capture` BEFORE the refactor, then `compare` AFTER. Same
seed + same truncated frame count => identical trajectory prefix, so a hash match
is a byte-identical proof over the frames tested (integration runs every frame, so
a short prefix already exercises the whole build -> integrate -> record path).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Thread count matters for the ORACLE, not the physics: a run's result is identical at
# any thread count *run alone*, but two multi-threaded torch processes fighting for cores
# get order-nondeterministic CPU reductions (scatter/index_add, MPM P2G) and diverge. So:
#   PLEXUS_BASELINE_THREADS=1 (default) -- single-thread: byte-identical and contention-PROOF
#                                          (the robust CI default, "force determinism").
#   PLEXUS_BASELINE_THREADS=0           -- all cores: ~5x faster, but ONLY run one instance
#                                          at a time (no concurrent capture/compare).
# Set before importing torch so intra-op threads honour it.
_THREADS = int(os.environ.get("PLEXUS_BASELINE_THREADS", "1"))
if _THREADS > 0:
    os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(_THREADS))

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
if _THREADS > 0:
    torch.set_num_threads(_THREADS)

import plexus.operators  # noqa: F401  self-register the operator library
from plexus.schema import load
from plexus.engine import run

CONFIG = os.path.join(ROOT, "config")

# The cheap-but-strong determinism gate: one spec per major code path.
GATE_SPECS = [
    "attraction_repulsion/arbitrary_2.yaml",   # spatial, non-MPM, radius graph
    "boids/boids_16.yaml",                     # active matter / boids
    "slime/slime_default.yaml",                # agents + a diffusing field
    "material/material_3balls_bouncy.yaml",    # MLS-MPM substep + F/C/Jp/material
]

# Persistent MPM particle buffers: F/C deformation, Jp plastic ratio, mass, Lame
# params, and the material masks. Hashed wherever a set carries them, so a pos-only
# hash cannot mask a deformation / plasticity / material regression.
MPM_BUFFERS = ("F", "C", "Jp", "mass", "mu", "la",
               "is_liquid", "is_snow", "is_visco", "visco_tau", "p_vol")


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def run_spec(path: str, frames: int) -> dict:
    """Run one spec for a short prefix on CPU and return a dict of hashes covering
    every piece of state the refactor could perturb: recorded pos+occ (and any
    future `state` group), field grids, and MPM substep buffers off the final state."""
    sim = load(path)
    sim.n_frames = min(sim.n_frames, frames)
    H, out = run(sim, out_path=None, device="cpu", progress=False)

    rec: dict = {"frames": sim.n_frames, "sets": {}, "fields": {}, "mpm": {}, "live": {}}

    for name in sorted(out["sets"]):
        s = out["sets"][name]
        parts = []
        if s.get("pos") is not None:            # spatial trajectory (None for a non-spatial set)
            parts.append(s["pos"])
        parts.append(s["occ"])
        st = s.get("state")                     # recorded state blocks: a {block: array} dict, or an array
        if isinstance(st, dict):
            for k in sorted(st):
                parts.append(st[k])
        elif st is not None:
            parts.append(st)
        shape = list(np.asarray(s["pos"]).shape) if s.get("pos") is not None else None
        rec["sets"][name] = {"shape": shape, "hash": _digest(*parts)}

    for fn in sorted(out.get("fields", {})):     # field grids (slime pheromone, mpm fields, ...)
        rec["fields"][fn] = _digest(out["fields"][fn]["grid"])

    for lvlname in sorted(H.levels):
        lvl = H.level(lvlname)
        # final in-memory occupancy -- a cross-check on the recording path: if the new
        # code records occ wrongly, the recorded hash could hide it, but this catches it.
        rec["live"][lvlname] = {"occ_hash": _digest(lvl.occ.detach().cpu().numpy())}
        # persistent MPM buffers off the final state
        got, arrs = [], []
        for buf in MPM_BUFFERS:
            if hasattr(lvl, buf):
                got.append(buf)
                arrs.append(getattr(lvl, buf).detach().cpu().numpy())
        if got:
            rec["mpm"][lvlname] = {"buffers": got, "hash": _digest(*arrs)}
    return rec


def _run_many(keys, frames):
    out = {}
    t0 = time.time()
    for i, key in enumerate(keys):
        p = os.path.join(CONFIG, key)
        t = time.time()
        try:
            out[key] = run_spec(p, frames)
            print(f"[{i + 1:3d}/{len(keys)}] {key:58s} {time.time() - t:6.2f}s", flush=True)
        except Exception as e:
            out[key] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{i + 1:3d}/{len(keys)}] {key:58s} ERROR {type(e).__name__}: {e}", flush=True)
    print(f"total {time.time() - t0:.1f}s over {len(keys)} specs", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "capture", "compare"])
    ap.add_argument("--out", default="/tmp/plexus_state_baseline.json")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--gate-only", action="store_true",
                    help="capture/compare only the 4 GATE_SPECS (fast dev loop against the full baseline)")
    args = ap.parse_args()

    all_keys = sorted(os.path.relpath(p, CONFIG)
                      for p in glob.glob(os.path.join(CONFIG, "**", "*.yaml"), recursive=True))
    if args.gate_only:
        all_keys = list(GATE_SPECS)

    if args.mode == "gate":
        print("=== determinism gate: run each representative spec TWICE, require identical hashes ===")
        r1 = _run_many(GATE_SPECS, args.frames)
        print("--- second pass ---")
        r2 = _run_many(GATE_SPECS, args.frames)
        ok = True
        for k in GATE_SPECS:
            if "error" in r1[k] or "error" in r2[k]:
                print(f"FAIL {k}: errored"); ok = False; continue
            same = r1[k] == r2[k]
            print(f"{'PASS' if same else 'FAIL (NONDETERMINISTIC)'}  {k}")
            ok = ok and same
        print("\nGATE " + ("PASSED — generation is deterministic; safe to capture baseline."
                            if ok else "FAILED — nondeterminism present; fix before any schema work."))
        sys.exit(0 if ok else 1)

    results = _run_many(all_keys, args.frames)

    if args.mode == "capture":
        with open(args.out, "w") as f:
            json.dump({"frames": args.frames, "results": results}, f)
        n_err = sum(1 for v in results.values() if "error" in v)
        print(f"baseline written -> {args.out}  ({len(results)} specs, {n_err} errored)")
        return

    base = json.load(open(args.out))["results"]
    n_ok = n_diff = n_new = n_err = 0
    for k, v in results.items():
        if "error" in v:
            n_err += 1; print(f"ERROR (now) {k}: {v['error']}"); continue
        b = base.get(k)
        if b is None:
            n_new += 1; print(f"NEW (no baseline) {k}"); continue
        if "error" in b:
            continue
        if v == b:
            n_ok += 1
        else:
            n_diff += 1
            for grp in ("sets", "fields", "mpm", "live"):
                for kk in v.get(grp, {}):
                    if v[grp].get(kk) != b.get(grp, {}).get(kk):
                        print(f"DIFF {grp}:{kk}  {k}")
    print(f"\ncompare: {n_ok} identical, {n_diff} DIFF, {n_new} new, {n_err} newly-errored")
    sys.exit(1 if (n_diff or n_new or n_err) else 0)


if __name__ == "__main__":
    main()
