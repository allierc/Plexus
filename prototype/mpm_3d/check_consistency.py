#!/usr/bin/env python
"""check_consistency -- assert the dimension-generic mpm stack did NOT change 2D.

The 3D modifications (mpm_grid / mpm_strain / p2g / mpm_grid_update / g2p / the entity
provision / aggregate / gravity / the engine child-schema) are all written so the
D == 2 path reduces bit-identically to the original 2D operators. This script runs a
couple of 2D `config/material/*` specs for a few frames and checks every per-particle
buffer (pos / vel / F / C / Jp) is byte-for-byte unchanged against a saved baseline.

Usage (repo root, conda env + PYTHONPATH=src):
    # 1) on the ORIGINAL operators, write the baseline:
    python prototype/mpm_3d/check_consistency.py --save
    # 2) after the 3D edits, verify:
    python prototype/mpm_3d/check_consistency.py
"""
import sys

import torch
import plexus.operators            # noqa: F401  registers the operator library
import plexus.schema as S
import plexus.engine as E
from plexus.paths import resolve_config

SPECS = ["material_two_drops_nost", "material_snow_pile"]   # liquid + snow paths
BASELINE = "/tmp/mpm2d_baseline.pt"
KEYS = ("pos", "vel", "F", "C", "Jp")


def capture(name, n=5, device="cpu"):
    yf, _, _ = resolve_config(name)
    sim = S.load(yf); sim.n_frames = n
    H, _ = E.run(sim, out_path=None, device=device)
    p = H.levels["mpm_particle"]
    return {"pos": p.get("pos").clone(), "vel": p.get("vel").clone(),
            "F": p.F.clone(), "C": p.C.clone(), "Jp": p.Jp.clone()}


def main():
    cur = {nm: capture(nm) for nm in SPECS}
    if "--save" in sys.argv:
        torch.save(cur, BASELINE)
        print(f"saved baseline -> {BASELINE}")
        return
    base = torch.load(BASELINE)
    ok = True
    for nm in SPECS:
        identical = all(torch.equal(cur[nm][k], base[nm][k]) for k in KEYS)
        if not identical:
            diffs = {k: float((cur[nm][k] - base[nm][k]).abs().max()) for k in KEYS}
            print(f"{nm}: CHANGED  max|Δ| = {diffs}")
        else:
            print(f"{nm}: bit-identical ✓")
        ok = ok and identical
    print("2D MPM consistency:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
