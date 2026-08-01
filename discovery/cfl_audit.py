#!/usr/bin/env python
"""cfl_audit -- is every spec in this repository actually integrable?

WHY THIS IS ITS OWN FILE. The stability bound was a COMMENT in translate.py for as long as the
campaign has existed, asserting `dt*chi*max(d_a,d_h) = 0.02*65*0.16 = 0.21 <= 1`. True when
written. Then the defaults moved, Phase 2 widened the diffusivity box to reach Okuda's phi = 10,
and the arithmetic in the comment silently stopped describing the code. A comment cannot be
re-derived; it can only be re-read, and nobody had cause to re-read it.

So the bound is now computed, and this file asks the question of EVERY spec at once, on both
paths that produce one:

  HAND-WRITTEN configs (config/okuda/*.yaml) carry chi as the engine will see it.
      stable iff   dt * chi * max(d_a, d_h) <= limit

  COMPOSITIONS compiled by translate carry chi scaled by RD_PER_FRAME = 1/dt (the D5a clock
  fix, which exists because cell_react and cell_diffuse EMIT=velocity into `chem` and were
  otherwise integrated on the mechanics substep). The engine therefore steps
      dt * (chi/dt) * d  =  chi * d
  and the dt cancels. Counting it anyway is what made the old bound 50x too permissive.

THE TWO PATHS DISAGREE, AND THAT IS THE FINDING. A preset and the composition built from it are
not automatically the same experiment: round_44_base runs finite as a hand config at 0.056 and
compiles to 2.8 as a composition. Any preset whose two columns differ is a recipe whose archived
result does not belong to the composition that bears its name.

    python cfl_audit.py            # the table
    python cfl_audit.py --strict   # exit non-zero if anything is unstable
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "agents"))

CONFIGS = os.path.join(ROOT, "config", "okuda")


def hand_cfl(cfg):
    """A hand-written config: chi is already what the engine sees, so dt counts."""
    d = next((o for o in cfg.get("operators", []) if o.get("op") == "cell_diffuse"), None)
    if d is None:
        return None
    dt = float(cfg["general"]["dt"])
    return dt * float(d.get("chi", 1.0)) * max(float(d.get("d_a", 0.0)), float(d.get("d_h", 0.0)))


def emitted_cfl(spec):
    """A translate-produced spec. chi is PRE-SCALED by 1/dt and the engine still multiplies by
    dt, so the CFL number is dt * chi_emitted * d -- which equals chi_unscaled * d. Dropping the
    dt here (my first version) inflates every composition by 1/dt and reports 10.4 for a spec
    whose real number is 0.208. The same class of error as the bug being audited, made while
    auditing it."""
    d = next((o for o in spec.get("operators", []) if o.get("op") == "cell_diffuse"), None)
    if d is None:
        return None
    dt = float(spec["general"]["dt"])
    return dt * float(d.get("chi", 1.0)) * max(float(d.get("d_a", 0.0)), float(d.get("d_h", 0.0)))


def audit_configs():
    rows = []
    for f in sorted(os.listdir(CONFIGS)):
        if not f.endswith(".yaml"):
            continue
        try:
            cfg = yaml.safe_load(open(os.path.join(CONFIGS, f)))
        except Exception:
            continue
        c = hand_cfl(cfg)
        if c is not None:
            rows.append((f[:-5], c))
    return rows


def audit_presets():
    """Every trusted preset, as a COMPOSITION, through the real translator."""
    import translate as T
    from composition_space import reference_recipes
    rows = []
    try:
        presets = T.load_presets()
    except Exception:
        presets = {}
    items = [(n, T.from_preset(p)) for n, p in presets.items()]
    items += [(f"ref:{n}", g) for n, g in reference_recipes().items()]
    for name, g in items:
        try:
            spec = T.to_spec(g, name=name, frames=10)
            rows.append((name, emitted_cfl(spec), "compiles"))
        except ValueError as e:
            msg = str(e)
            rows.append((name, None, "REFUSED: " + ("CFL" if "integrable" in msg else msg[:40])))
        except Exception as e:
            rows.append((name, None, f"{type(e).__name__}"))
    return rows


def main(strict=False):
    from translate import CFL_LIMIT
    print("=" * 94)
    print("CFL AUDIT -- can every spec in this repository actually be integrated?")
    print("=" * 94)

    hand = audit_configs()
    bad_hand = [r for r in hand if r[1] > CFL_LIMIT]
    print(f"\nHAND-WRITTEN CONFIGS  ({len(hand)} with chemistry)   bound: dt*chi*max(d) <= {CFL_LIMIT}")
    print(f"  {'stable':>8}   {len(hand) - len(bad_hand)}")
    for n, c in sorted(bad_hand, key=lambda r: -r[1]):
        print(f"  {'UNSTABLE':>8}   {n:36} {c:7.2f}")
    if not bad_hand:
        print(f"  {'':8}   (worst is {max(c for _, c in hand):.2f})")

    comp = audit_presets()
    print(f"\nCOMPOSITIONS THROUGH translate   bound: chi*max(d) <= {CFL_LIMIT}  (dt cancels)")
    for n, c, note in comp:
        if c is None:
            flag = "no chem" if note == "compiles" else "REFUSED"
        else:
            flag = "ok" if c <= CFL_LIMIT else "UNSTABLE"
        print(f"  {flag:>8}   {n:36} {('%7.2f' % c) if c is not None else '     --'}  {note}")

    # the disagreement: a preset whose two paths give different numbers is two experiments
    hand_by = dict(hand)
    print("\nDO THE TWO PATHS AGREE?  a preset and its composition should be the same experiment")
    dis = []
    for n, c, _ in comp:
        h = hand_by.get(n)
        if h is not None and c is not None and abs(h - c) > 1e-6:
            dis.append((n, h, c))
    if not dis:
        print("  they agree everywhere they can be compared")
    for n, h, c in sorted(dis, key=lambda r: -abs(r[2] - r[1])):
        print(f"  {n:36} hand {h:7.3f}   composition {c:7.2f}   x{c / max(h, 1e-9):.0f}")

    n_bad = len(bad_hand) + sum(1 for _, c, _ in comp if c is not None and c > CFL_LIMIT)
    print(f"\n  {n_bad} unstable, {len(dis)} disagreements between the two paths")
    print("=" * 94)
    return 1 if (strict and (n_bad or dis)) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    raise SystemExit(main(ap.parse_args().strict))
