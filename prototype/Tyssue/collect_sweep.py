#!/usr/bin/env python
"""Collect + rank the Fig-5 cluster sweep by the GOAL score. Reads archive/<preset>/diag.json for the
sweep presets and ranks by score = protr - 1.5*area_cv - 2*hollow  (want tubes that stick out [protr>1.5],
uniform cells [area_cv low], clean mesh [hollow low]). Prints a table so the next round targets the best
region of (vth, rho, cone).

    python collect_sweep.py            # rank sw_* presets
    python collect_sweep.py fig5_*     # rank a different glob
"""
import json, glob, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
pat = sys.argv[1] if len(sys.argv) > 1 else "sw_*"
rows = []
for d in sorted(glob.glob(os.path.join(HERE, "archive", pat, "diag.json"))):
    try:
        r = json.load(open(d))
    except Exception:
        continue
    if "area_cv" not in r or "error" in r:
        rows.append({"name": os.path.basename(os.path.dirname(d)), "err": r.get("error", "no metrics")}); continue
    r["score"] = round(r.get("protr", 1.0) - 1.5 * r.get("area_cv", 1.0) - 2.0 * r.get("hollow_frac", 1.0), 3)
    rows.append(r)
ok = [r for r in rows if "score" in r]
ok.sort(key=lambda r: r["score"], reverse=True)
print(f"{'name':18s} {'score':>6s} {'protr':>6s} {'areaCV':>7s} {'hollow':>7s} {'cells':>6s} {'spots':>5s}  params")
for r in ok:
    print(f"{r['name']:18s} {r['score']:6.3f} {r.get('protr',0):6.3f} {r.get('area_cv',0):7.3f} "
          f"{r.get('hollow_frac',0):7.3f} {r.get('cells_end',0):6d} {r.get('spots',0):5d}  "
          f"vth={r.get('vth')} rho={r.get('rho')} cone={r.get('n_spots')}")
for r in rows:
    if "err" in r:
        print(f"{r['name']:18s}  ERR/pending: {r['err']}")
print(f"\n{len(ok)}/{len(rows)} with metrics. Best: {ok[0]['name'] if ok else '(none yet)'}")
