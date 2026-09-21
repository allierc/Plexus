#!/usr/bin/env python
"""Is momentum conserved, and is the body still one body? The two invariants of a swimmer.

    python tools/platynereis_momentum.py plat_r13_torque

THESE ARE NOT DIAGNOSTICS, THEY ARE THE PASS CONDITION. A swimming model that creates momentum is
not a worse swimmer, it is not a swimmer at all -- the thrust it appears to produce is an
arithmetic artefact, and every number downstream of it means nothing. R12 failed both and looked
fine in every picture: total momentum ran 0.58 -> 45.9 -> 2.4 from a standing start, the body's
momentum sat 9 degrees from the water's instead of 180, and the animal inflated to 168% of its
own radius while its centre of mass travelled 105 um.

WHAT IS MEASURED.

  * TOTAL MOMENTUM of every material set, summed. It starts at zero and must stay there. What it
    cannot do is grow: an isolated scene has no external force except the walls, and the walls
    can only take momentum away.
  * THE ANGLE between the body's momentum and the water's. Newton's third law puts it at 180
    degrees -- the animal goes one way and the fluid it pushed goes the other. Zero means both
    are being shoved by a common source, which is what momentum creation looks like from the
    outside.
  * THE BODY'S OWN RADIUS about its centre of mass. A body that is being propelled keeps its
    shape; one that is being inflated does not. This separates "it swam" from "it exploded",
    which the centre-of-mass displacement alone cannot.
  * THE WALL MOMENTUM, as the difference: whatever the scene lost that the walls must have taken.
    A run in which the walls dominate is measuring its container.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))

UM = 195.9


def load(name: str):
    from plexus.paths import graphs_data_path
    G = graphs_data_path()
    for sub in ("studio", "platynereis"):
        t = os.path.join(G, sub, name, "trajectory.npz")
        if os.path.exists(t):
            return np.load(t, allow_pickle=True), os.path.join(G, sub, name, "spec.yaml")
    raise SystemExit(f"no trajectory for {name}")


def masses(spec_path):
    """Per-particle mass of every material set, from the spec rather than guessed."""
    import yaml
    d = yaml.safe_load(open(spec_path))
    out = {}
    for k, v in (d.get("sets") or {}).items():
        if "density" not in v:
            continue
        pm = v.get("particle_mass")
        if pm is not None:
            out[k] = float(pm)
            continue
        r = float(v.get("radius", 0.02))
        per = v.get("per_parent")
        if isinstance(per, dict):
            per = max(int(x) for x in per.values())
        per = int(per) if per else 1
        out[k] = float(v["density"]) * (4.0 / 3.0 * math.pi * r ** 3) / max(per, 1)
    return out


def measure(tr, m_of, dt):
    sets, tot = {}, None
    for k, m in m_of.items():
        key = f"{k}__pos"
        if key not in tr.files:
            continue
        P = np.asarray(tr[key])
        occ = tr.get(f"{k}__occ")
        if occ is not None:
            live = np.asarray(occ)[0].astype(bool)
            if live.ndim == 1 and live.shape[0] == P.shape[1]:
                P = P[:, live, :]
        p = m * (np.diff(P, axis=0) / dt).sum(1)               # [T-1, 3]
        sets[k] = {"p": p, "mass": m * P.shape[1], "n": P.shape[1]}
        tot = p if tot is None else tot + p

    body = sum((v["p"] for k, v in sets.items() if k in ("body_point", "mpm_particle")),
               np.zeros_like(tot))
    water = sets.get("water_particle", {}).get("p", np.zeros_like(tot))
    cos = (body * water).sum(1) / np.maximum(
        np.linalg.norm(body, axis=1) * np.linalg.norm(water, axis=1), 1e-30)

    B = np.asarray(tr["body_point__pos"])
    rad = np.linalg.norm(B - B.mean(1, keepdims=True), axis=2).mean(1) * UM
    return {
        "sets": sets, "total": tot, "angle": np.degrees(np.arccos(np.clip(cos, -1, 1))),
        "radius": rad, "com": np.linalg.norm(B - B[0].mean(0), axis=2).mean(1) * UM,
        "com_travel": float(np.linalg.norm(B[-1].mean(0) - B[0].mean(0)) * UM),
    }


def report(st, name):
    tot = st["total"]
    n = tot.shape[0]
    print(f"{name}\n")
    print(f"  {'set':16s} {'n':>9s} {'total mass':>12s}")
    for k, v in st["sets"].items():
        print(f"  {k:16s} {v['n']:9,d} {v['mass']:12.3e}")
    print(f"\n  TOTAL MOMENTUM (must stay at zero; the walls can only remove it):")
    for f in (0, n // 8, n // 3, 2 * n // 3, n - 1):
        print(f"    frame {f:4d}: |p| = {np.linalg.norm(tot[f]):.4e}")
    print(f"    peak       : |p| = {np.linalg.norm(tot, axis=1).max():.4e}")
    print(f"\n  BODY vs WATER momentum angle (180 deg is Newton's third law):")
    print(f"    median {np.median(st['angle'][n // 8:]):.1f} deg   "
          f"final {st['angle'][-1]:.1f} deg")
    print(f"\n  IS THE BODY STILL ONE BODY?")
    print(f"    mean radius about its own centre of mass "
          f"{st['radius'][0]:.1f} -> {st['radius'][-1]:.1f} um "
          f"({100 * st['radius'][-1] / st['radius'][0]:.0f}%)")
    print(f"    centre of mass travelled {st['com_travel']:.1f} um")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="plat_r13_torque")
    ap.add_argument("--dt", type=float, default=0.05)
    a = ap.parse_args()
    tr, sp = load(a.name)
    st = measure(tr, masses(sp), a.dt)
    report(st, a.name)
