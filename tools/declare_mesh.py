#!/usr/bin/env python
"""Write the `mesh: half_edge` declaration into every spec that already builds one.

WHY A MIGRATION AND NOT A DEFAULT. `engine._build_mesh` allocates the half-edge table when a set
DECLARES it, and a declaration only means something if the specs carry it: a spec that schedules
`mesh_seed` and says nothing about a mesh is a spec whose central structure is invisible until an
operator happens to run. The pairing goes with it -- `cell_set:` on the set rather than repeated as
a parameter on `mesh_seed`, `cell_divide`, `cell_die` and defaulted to the literal string "cell"
inside `edge_flip`.

WHAT IT WRITES, and nothing else:

    sets:
      vertex:
        n: 120000
        mesh: half_edge        <- added
        cell_set: cell         <- added, taken from the spec's own `mesh_seed`

The set is the one `mesh_seed` targets (`at:`), the cell set is that operator's own `cell_set:`, so
the declaration states what the spec already does rather than deciding anything for it. A spec that
already carries both keys is left alone, which makes this idempotent and safe to re-run after the
campaign writes new specs.

THE FALLBACK STAYS. `mesh_seed` still creates a table when the set has not declared one, because
the generators (`make_basis.py`, `make_apop_geo.py`, `round.py` ...) write specs at run time and a
migration cannot reach a file that does not exist yet.

    python tools/declare_mesh.py --dry        list what would change
    python tools/declare_mesh.py              write it
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def migrate(path, write=True):
    """(changed, reason). `reason` is why not, when it did not change."""
    with open(path) as f:
        raw = f.read()
    # BOTH SPELLINGS. `seed_mesh` is the canonical post-seed-refactor name and `mesh_seed` the
    # alias; the corpus is split 324/137 between them, so a migration that knew only one would
    # silently skip most of it and report success.
    if "seed_mesh" not in raw and "mesh_seed" not in raw:
        return False, "no mesh seed operator"
    cfg = yaml.safe_load(raw)
    if not isinstance(cfg, dict) or "sets" not in cfg:
        return False, "not a spec"
    ops = list(cfg.get("operators") or []) + list(cfg.get("seed") or [])
    seeds = [o for o in ops if isinstance(o, dict) and o.get("op") in ("seed_mesh", "mesh_seed")]
    if not seeds:
        return False, "the name appears but no operator declares it"
    if len(seeds) > 1:
        return False, f"{len(seeds)} mesh-seed operators -- one table per set is the assumption"
    op = seeds[0]
    sname = op.get("at", "vertex")
    cs = op.get("cell_set", "cell")
    s = (cfg["sets"] or {}).get(sname)
    if s is None:
        return False, f"mesh_seed targets set {sname!r}, which the spec does not declare"
    if cs not in cfg["sets"]:
        return False, f"cell_set {cs!r} is not a set in this spec"
    if s.get("mesh") == "half_edge" and s.get("cell_set") == cs:
        return False, "already declared"
    s["mesh"] = "half_edge"
    s["cell_set"] = cs
    if write:
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    return True, f"{sname}: mesh half_edge, cell_set {cs}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--glob", default=os.path.join(ROOT, "config", "okuda", "*.yaml"))
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob))
    done, skipped = [], {}
    for p in files:
        ch, why = migrate(p, write=not a.dry)
        (done.append((os.path.basename(p), why)) if ch
         else skipped.setdefault(why, []).append(os.path.basename(p)))
    print(f"  {len(done)} of {len(files)} spec(s) {'would gain' if a.dry else 'gained'} the declaration")
    for why, names in sorted(skipped.items()):
        print(f"    skipped ({why}): {len(names)}" + (f"  e.g. {names[0]}" if names else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
