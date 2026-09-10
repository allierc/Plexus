"""The seed table: what every vertex-tissue spec seeds, before any dynamics.

    python tools/seed_conventions.py            rebuild tests/regression/seed_conventions.json
    python tools/seed_conventions.py --check    compare the current seeds with the table (exit 1 on drift)

For every spec under config/tissue, config/gates and config/cell that seeds a vertex tissue, the
Hierarchy is built and seeded on CPU (seconds, no dynamics) and the mesh table's V0f median, v_ref,
convention (`v0_from`), cell and vertex counts are recorded. c671fb31 changed the default
convention and re-targeted about eighty of these specs at load; this table fails on that in
seconds, before any trajectory is run. Layer C of tests/REGRESSION_PLAN.md.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
TABLE = os.path.join(ROOT, "tests", "regression", "seed_conventions.json")
FAMILIES = ("tissue", "gates", "cell")
SEED_OPS = ("seed_mesh", "mesh_seed")


def specs_with_vertex_seed():
    out = []
    for fam in FAMILIES:
        for f in sorted(glob.glob(os.path.join(ROOT, "config", fam, "*.yaml"))):
            if os.path.basename(f).startswith("_regression_"):
                continue
            try:
                s = yaml.safe_load(open(f))
            except Exception:                            # noqa: BLE001
                continue
            if not isinstance(s, dict):
                continue
            ops = (s.get("seed") or []) + (s.get("operators") or [])
            if any(isinstance(o, dict) and o.get("op") in SEED_OPS for o in ops):
                out.append((fam, f))
    return out


def seed_row(path: str) -> dict:
    """Build + seed on CPU; read the vertex set's mesh table."""
    from plexus import schema, engine
    sim = schema.load(path)
    H = engine.build(sim, "cpu")
    engine.seed(H, sim, "cpu")
    rows = {}
    for lname, lvl in H.levels.items():
        m = getattr(lvl, "_mesh", None)
        if m is None or not int(m.get("Nv", 0)):
            continue
        nF = int(m["nF"])
        v0 = m.get("V0f")
        v0 = v0.detach().cpu().numpy()[:nF] if torch.is_tensor(v0) else None
        rows[lname] = {
            "Nv": int(m["Nv"]), "nF": nF,
            "V0f_median": None if v0 is None or not v0.size else float(np.median(v0)),
            "V0f_sum": None if v0 is None or not v0.size else float(v0.sum()),
            "v_ref": float(m["v_ref"]) if "v_ref" in m else None,
            "v0_from": str(m.get("v0_from", "wedge")),
            "centre": [float(v) for v in m["centre"]] if "centre" in m else None,
        }
    return rows


def build_table() -> dict:
    table = {}
    for fam, f in specs_with_vertex_seed():
        key = f"{fam}/{os.path.basename(f)[:-5]}"
        try:
            table[key] = seed_row(f)
        except Exception as e:                           # noqa: BLE001
            table[key] = {"load_error": f"{type(e).__name__}: {str(e)[:160]}"}
    return table


def compare(ref: dict, now: dict, rtol: float = 1e-5) -> list[str]:
    bad = []
    for key, r in ref.items():
        n = now.get(key)
        if n is None:
            bad.append(f"{key}: spec gone"); continue
        if "load_error" in r or "load_error" in n:
            if r.get("load_error") != n.get("load_error"):
                bad.append(f"{key}: load status changed: {r.get('load_error')} -> {n.get('load_error')}")
            continue
        for lname, rr in r.items():
            nn = n.get(lname)
            if nn is None:
                bad.append(f"{key}: set {lname} no longer seeds a mesh"); continue
            for k, rv in rr.items():
                nv = nn.get(k)
                if isinstance(rv, (int, str)) or rv is None or isinstance(rv, list):
                    if nv != rv:
                        bad.append(f"{key}/{lname}: {k} {nv!r} != {rv!r}")
                elif abs(float(nv) - float(rv)) > rtol * max(abs(float(rv)), 1e-12):
                    bad.append(f"{key}/{lname}: {k} {nv:.6g} != {rv:.6g}")
    for key in now:
        if key not in ref:
            bad.append(f"{key}: new spec, not in the table (rebuild the table in its own commit)")
    return bad


def main():
    check = "--check" in sys.argv
    now = build_table()
    if check:
        ref = json.load(open(TABLE))
        bad = compare(ref, now)
        print(f"seed table: {len(now)} specs, {'clean' if not bad else f'{len(bad)} drift(s)'}")
        for b in bad:
            print("   " + b)
        sys.exit(1 if bad else 0)
    json.dump(now, open(TABLE, "w"), indent=1, sort_keys=True)
    errs = [k for k, v in now.items() if "load_error" in v]
    print(f"wrote {os.path.relpath(TABLE, ROOT)}: {len(now)} specs, {len(errs)} fail to load")
    for k in errs:
        print(f"   {k}: {now[k]['load_error']}")


if __name__ == "__main__":
    main()
