"""Capture what the three quantity consumers compute today, so a refactor can be checked byte for byte.

    python tools/quantities_baseline.py capture [--label <name>] [--rows 60] [folder ...]
    python tools/quantities_baseline.py compare <labelA> <labelB>

`capture` writes log/quantities_baseline/<label>/<archive>.json for every archive (default: the
registered working points' archives), holding, computed by TODAY'S code paths:

  curve        the movie panel's `_curve_row` for every quantity it knows, at five frames, on the
               replay level exactly as `-o plot` builds it (float32 tensors, numpy conversions);
  gates        every function in `tools/gate_measures.MEASURES` on the archive's trajectory reader
               (CoreTraj or ParticleTraj), default arguments, the first `--rows` rows;
  fingerprint  `tools/regression_lib.fingerprint_archive` at a 150-row cut.

Floats are written with Python's repr, which round-trips exactly, so `compare` is a real equality:
two labels agree only if every number is the same bits. A measure that raises is recorded as the
exception's class and message, which is also compared.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "log", "quantities_baseline")
CURVE_QUANTITIES = ("cells", "phase", "cycle_progress", "myosin", "radius", "area", "volume")


def _git():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                    # noqa: BLE001
        return "nogit"


def _py(v):
    """JSON-safe, repr-exact: numpy scalars to Python, arrays to nested lists, NaN kept as a string."""
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, np.ndarray):
        return _py(v.tolist())
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return "nan" if f != f else ("inf" if f in (float("inf"), float("-inf")) else f)
    if isinstance(v, (np.integer, int, bool, str)) or v is None:
        return v.item() if isinstance(v, np.integer) else v
    return repr(v)


class _Head:
    """A trajectory reader with its row count capped: every measure that loops `range(T.n_rows())`
    then reads only the first `k` rows, which bounds the capture to minutes without changing any
    number it does produce."""

    def __init__(self, T, k):
        self._T, self._k = T, int(k)

    def n_rows(self):
        return min(self._k, self._T.n_rows())

    def __getattr__(self, name):
        return getattr(self._T, name)


def capture_curve(folder, spec, z, frames):
    from plexus import live_movie as LM
    import torch
    mesh_sets = sorted({k[: -len("__mesh_nF")] for k in z.files if k.endswith("__mesh_nF")})
    if not mesh_sets:
        return {"skipped": "no mesh set"}
    cs = {}
    for ms in mesh_sets:
        c = (spec.get("sets", {}).get(ms, {}) or {}).get("cell_set")
        if c is None:
            for o in (spec.get("seed") or []) + (spec.get("operators") or []):
                if isinstance(o, dict) and o.get("op") in ("seed_mesh", "mesh_seed") and o.get("at") == ms:
                    c = o.get("cell_set", "cell")
        cs[ms] = c or "cell"
    H = LM._ReplayState(z, "cpu", cell_sets=cs)
    lvl = H.level(mesh_sets[0])
    obj = LM.LiveMovie.__new__(LM.LiveMovie)
    obj.style = spec.get("plotting") or {}
    out = {"mesh_set": mesh_sets[0], "cell_set": cs[mesh_sets[0]], "rows": {}}
    ntype = LM.LiveMovie._curve_types(obj, H, lvl)
    out["ntype_max"] = None if ntype is None else int(np.max(ntype))
    for t in frames:
        H.seek(int(t))
        row = {}
        for q in CURVE_QUANTITIES:
            nt = 4 if q == "phase" else (1 if ntype is None else int(np.max(ntype)) + 1)
            try:
                r = LM.LiveMovie._curve_row(obj, H, lvl, q, ntype, nt)
                row[q] = _py(np.asarray(r))
            except Exception as e:                       # noqa: BLE001
                row[q] = f"ERROR {type(e).__name__}: {str(e)[:120]}"
        out["rows"][str(int(t))] = row
    H.seek(0)
    return out


def capture_gates(folder, z, rows):
    import gate_measures as GM
    path = os.path.join(folder, "trajectory.npz")
    has_mesh = any(k.endswith("__mesh_nF") for k in z.files)
    try:
        T = GM.CoreTraj(path) if has_mesh else GM.ParticleTraj(path)
    except Exception as e:                               # noqa: BLE001
        return {"skipped": f"{type(e).__name__}: {str(e)[:120]}"}
    Th = _Head(T, rows)
    out = {"reader": type(T).__name__, "rows": int(Th.n_rows()), "measures": {}}
    for name, fn in GM.MEASURES.items():
        try:
            v = fn(Th)
            out["measures"][name] = _py(v)
        except Exception as e:                           # noqa: BLE001
            out["measures"][name] = f"ERROR {type(e).__name__}: {str(e)[:120]}"
    return out


def capture_fingerprint(folder):
    import regression_lib as R
    try:
        fp = R.fingerprint_archive(folder, 150)
    except Exception as e:                               # noqa: BLE001
        return {"skipped": f"{type(e).__name__}: {str(e)[:160]}"}
    for k in ("archived_on", "generated_at_commit", "because", "archive"):
        fp.pop(k, None)
    return _py(fp)


def default_folders():
    import regression_lib as R
    out = []
    for n in R.registered():
        fp = R.read_fingerprint(n)
        if fp.get("archive") and os.path.exists(os.path.join(fp["archive"], "trajectory.npz")):
            out.append(fp["archive"])
    return out


def capture(label, folders, rows):
    dst = os.path.join(OUT, label)
    os.makedirs(dst, exist_ok=True)
    for folder in folders:
        name = os.path.basename(folder.rstrip("/"))
        spec = yaml.safe_load(open(os.path.join(folder, "spec.yaml")))
        z = np.load(os.path.join(folder, "trajectory.npz"))
        posk = [k for k in z.files if k.endswith("__pos")]
        n = int(z[posk[0]].shape[0]) - 1
        frames = sorted({0, n // 4, n // 2, (3 * n) // 4, n})
        rec = {"archive": folder, "frames": frames}
        for part, fn in (("curve", lambda: capture_curve(folder, spec, z, frames)),
                         ("gates", lambda: capture_gates(folder, z, rows)),
                         ("fingerprint", lambda: capture_fingerprint(folder))):
            try:
                rec[part] = fn()
            except Exception as e:                       # noqa: BLE001
                rec[part] = {"skipped": f"{type(e).__name__}: {str(e)[:160]}", "trace": traceback.format_exc()[-600:]}
        json.dump(rec, open(os.path.join(dst, name + ".json"), "w"), indent=1, sort_keys=True)
        c = rec["curve"]; g = rec["gates"]; f = rec["fingerprint"]
        print(f"[baseline] {name:28s} curve {'ok' if 'rows' in c else c.get('skipped')}  "
              f"gates {len(g.get('measures', {}))} measures ({sum(1 for v in g.get('measures', {}).values() if isinstance(v, str) and v.startswith('ERROR'))} raise)  "
              f"fingerprint {'ok' if 'checkpoints' in f else f.get('skipped')}", flush=True)
    print(f"[baseline] wrote {len(folders)} archives -> {os.path.relpath(dst, ROOT)}")


def _flatten(v, prefix=""):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from _flatten(x, f"{prefix}/{k}")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from _flatten(x, f"{prefix}[{i}]")
    else:
        yield prefix, v


def compare(a, b):
    da, db = os.path.join(OUT, a), os.path.join(OUT, b)
    names = sorted(set(os.listdir(da)) & set(os.listdir(db)))
    total = 0
    for n in names:
        A = json.load(open(os.path.join(da, n))); B = json.load(open(os.path.join(db, n)))
        A.pop("archive", None); B.pop("archive", None)
        fa, fb = dict(_flatten(A)), dict(_flatten(B))
        keys = sorted(set(fa) | set(fb))
        diffs = [(k, fa.get(k, "<absent>"), fb.get(k, "<absent>")) for k in keys if fa.get(k, "<absent>") != fb.get(k, "<absent>")]
        total += len(diffs)
        print(f"{n[:-5]:28s} {len(fa):6d} values  {len(diffs):5d} differ")
        for k, x, y in diffs[:8]:
            print(f"     {k}: {x!r} -> {y!r}")
    print(f"[baseline] {a} vs {b}: {total} differing values over {len(names)} archives")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("folders", nargs="*"); c.add_argument("--label", default=None)
    c.add_argument("--rows", type=int, default=60)
    d = sub.add_parser("compare"); d.add_argument("a"); d.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "capture":
        capture(args.label or _git(), args.folders or default_folders(), args.rows)
    else:
        sys.exit(1 if compare(args.a, args.b) else 0)


if __name__ == "__main__":
    main()
