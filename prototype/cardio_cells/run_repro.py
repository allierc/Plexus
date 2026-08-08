import json, numpy as np
import beat as B, repro_seg as R
uv = B.load()
print(f"  {'sigma':>6s} {'h':>6s} {'nA':>5s} {'nB':>5s} {'ARI':>8s} {'ARI null':>9s} "
      f"{'bnd dist':>9s} {'null':>7s}   verdict")
rows = []
for sigma in (0.8, 1.5, 2.5):
    for h in (0.08, 0.14, 0.22):
        r = R.compare(uv, sigma, h)
        rows.append(r)
        real = r["ari"] > 4 * max(r["ari_null"], 1e-3) and r["bd"] < 0.7 * r["bd_null"]
        print(f"  {sigma:>6.1f} {h:>6.2f} {r['n_A']:>5d} {r['n_B']:>5d} {r['ari']:>8.3f} "
              f"{r['ari_null']:>9.3f} {r['bd']:>9.2f} {r['bd_null']:>7.2f}   "
              f"{'REPRODUCIBLE' if real else 'not distinguishable from chance'}", flush=True)
json.dump(rows, open("repro_seg.json", "w"), indent=1)
best = max(rows, key=lambda r: r["ari"] - r["ari_null"])
print(f"\n  best: sigma={best['sigma']} h={best['h']}  ARI {best['ari']:.3f} vs null "
      f"{best['ari_null']:.3f}   boundary distance {best['bd']:.2f} vs {best['bd_null']:.2f} grid pts")
