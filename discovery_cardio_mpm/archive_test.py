#!/usr/bin/env python
"""archive_test -- the 324 archived fits as a test set for the metrics.

WHY THIS IS THE RIGHT TEST SET
================================================================================================
The whole loop will rank on these measurements, so they have to be right, and the cheapest way to
find out is to point them at models whose verdict we already know. The previous campaign left 324
finished fits with their checkpoints. We know something about them collectively that no single run
tells us: **they are mostly wrong** -- 302 of 324 beat predicting nothing, and not one of the 324
beats copying the previous beat.

So a metric that cannot separate them from the recording, or that flatters them, is a metric the
loop must not use. And a defect that shows up in one run may be a quirk; the same defect in three
hundred is a property of the apparatus.

Two questions here, both answered without running a simulation:

  1. **How much of each learned field sits on its bounds?** A field pressed against the box we
     drew is not a material property. On the joint-best fit it was 54% of the sheet for stiffness.
     Is that typical, or was that run unusual?
  2. **How flattering is the dashboard?** A third of its panels are tied to the recording and
     score a perfect 1.000. On the best fit that inflated the picture by +0.17. Typical?

    python archive_test.py --fields        # scan every checkpoint's learned fields
    python archive_test.py --report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
ARCHIVE = os.path.abspath(os.path.join(HERE, "..", "prototype", "cardio_mpm", "archive"))
sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

NUM = r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"


def _trainer():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_tr", os.path.join(HERE, "train.py"))
    tr = importlib.util.module_from_spec(spec)
    sys.modules["_tr"] = tr
    spec.loader.exec_module(tr)
    return tr


def claimed(run_dir):
    """What the run recorded for itself."""
    import re
    p = os.path.join(run_dir, "progress.txt")
    if not os.path.exists(p):
        return None
    t = open(p, errors="replace").read()
    m = re.search(r"it=(\d+)/(\d+).*?LS=" + NUM + r"\s+LS_SD=" + NUM, t)
    if not m:
        return None
    r2 = re.search(r"R2=" + NUM, t)
    return {"it": int(m.group(1)), "n_iter": int(m.group(2)), "ls": float(m.group(3)),
            "ls_sd": float(m.group(4)), "r2": float(r2.group(1)) if r2 else None}


def run_args(run_dir):
    p = os.path.join(run_dir, "config.json")
    if not os.path.exists(p):
        return {}
    c = json.load(open(p))
    return c.get("args", {}) or {}


def field_saturation(run_dir, tr, res=None):
    """What fraction of each learned field sits within 1% of its bounds.

    Evaluated on the same grid the trainer uses. No simulation: this is the network alone.
    """
    import torch
    ck = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "model_*.pt")))
    if not ck:
        return None
    try:
        sd = torch.load(ck[-1], map_location="cpu", weights_only=False)
    except Exception:
        return None
    a = run_args(run_dir)

    def f(k, d):
        try:
            return float(a.get(k, d))
        except Exception:
            return d
    om, hid, lay = f("--siren_omega", 30.0), int(f("--siren_hidden", 256)), int(f("--siren_layers", 3))
    RES = res or tr.RES
    xy = torch.stack(torch.meshgrid(torch.linspace(0, 1, RES), torch.linspace(0, 1, RES),
                                    indexing="ij"), -1).reshape(-1, 2)
    out = {"checkpoint": os.path.basename(ck[-1]), "siren_omega": om}
    for key, lab, bounds in (("stiff_siren", "stiffness", (f("--stiff_lo", 100.0), f("--stiff_hi", 150.0))),
                             ("gain_siren", "gain", (f("--gain_lo", 0.1), f("--gain_hi", 2.5))),
                             ("fibre_siren", "fibre_dtheta", (-f("--fibre_dev", 1.5708), f("--fibre_dev", 1.5708)))):
        if key not in sd:
            continue
        net = tr.Siren(in_features=2, hidden_features=hid, hidden_layers=lay, out_features=1,
                       outermost_linear=True, first_omega_0=om, hidden_omega_0=om)
        try:
            net.load_state_dict(sd[key])
        except Exception:
            continue
        net.eval()
        with torch.no_grad():
            raw = net(xy)[:, 0]
            # stiffness and gain are sigmoid-bounded; the fibre deviation is tanh-bounded
            u = (torch.tanh(raw) * 0.5 + 0.5) if key == "fibre_siren" else torch.sigmoid(raw)
            u = u.numpy()
        lo, hi = bounds
        out[lab] = {"at_floor": float((u < 0.01).mean()), "at_ceiling": float((u > 0.99).mean()),
                    "saturated": float(((u < 0.01) | (u > 0.99)).mean()),
                    "median": float(lo + (hi - lo) * np.median(u)), "bounds": [lo, hi]}
    return out


def scan(limit=None, verbose=True):
    tr = _trainer()
    runs = sorted(d for d in glob.glob(os.path.join(ARCHIVE, "*")) if os.path.isdir(d))
    if limit:
        runs = runs[:limit]
    rows = []
    for i, d in enumerate(runs):
        c = claimed(d)
        if not c:
            continue
        s = field_saturation(d, tr)
        if not s:
            continue
        rows.append({"run": os.path.basename(d), **c, "fields": s})
        if verbose and (i % 40 == 0):
            print(f"  [{i}/{len(runs)}] {os.path.basename(d)[:44]}", flush=True)
    return rows


def report(rows):
    print(f"\n{'=' * 104}\n  THE ARCHIVE AS A TEST SET -- {len(rows)} fits with a checkpoint and a score"
          f"\n{'=' * 104}")
    ls = np.array([r["ls"] for r in rows])
    print(f"  claimed LoopScore: min {ls.min():+.3f}  median {np.median(ls):+.3f}  max {ls.max():+.3f}")
    print(f"    above the do-nothing null (+0.070):  {(ls > 0.070).sum():3d}/{len(ls)}")
    print(f"    above the held-out replay bar (+0.62): {(ls > 0.62).sum():3d}/{len(ls)}")

    print(f"\n  HOW MUCH OF EACH LEARNED FIELD SITS ON ITS BOUNDS")
    print(f"  {'field':<14s} {'runs':>5s} {'median':>8s} {'p25':>7s} {'p75':>7s} {'max':>7s} "
          f"{'>10% of the sheet':>18s}")
    for lab in ("stiffness", "gain", "fibre_dtheta"):
        v = np.array([r["fields"][lab]["saturated"] for r in rows if lab in r["fields"]])
        if not v.size:
            continue
        print(f"  {lab:<14s} {v.size:>5d} {np.median(v) * 100:>7.1f}% {np.percentile(v, 25) * 100:>6.1f}% "
              f"{np.percentile(v, 75) * 100:>6.1f}% {v.max() * 100:>6.1f}% "
              f"{(v > 0.10).sum():>10d}/{v.size}")

    # does saturation track the score? if the best fits are the most saturated, the campaign was
    # rewarding the optimiser for reaching the box edge
    st = [(r["ls"], r["fields"]["stiffness"]["saturated"]) for r in rows if "stiffness" in r["fields"]]
    if len(st) > 8:
        a = np.array(st)
        rr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        top = a[a[:, 0] >= np.percentile(a[:, 0], 90)]
        bot = a[a[:, 0] <= np.percentile(a[:, 0], 10)]
        print(f"\n  DOES SATURATION TRACK THE SCORE?  correlation {rr:+.3f}")
        print(f"    best 10% of fits: stiffness {top[:, 1].mean() * 100:.1f}% saturated")
        print(f"    worst 10%       : stiffness {bot[:, 1].mean() * 100:.1f}% saturated")
    print("=" * 104)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)
    out = os.path.join(HERE, "_metrology", "archive_fields.json")
    if a.fields or not os.path.exists(out):
        rows = scan(a.limit)
        json.dump(rows, open(out, "w"), indent=1)
    else:
        rows = json.load(open(out))
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
