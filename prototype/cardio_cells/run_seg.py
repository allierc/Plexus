import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import beat as B, segment as S

uv = B.load(); b, nb = B.mean_beat(uv)
zn, bnd, w, ang, amp, aniso = S.axis_field(b)
res = {}
for h in (0.06, 0.10, 0.16, 0.24):
    lab, n = S.watershed_cells(bnd, w, h=h)
    st = S.region_stats(lab, ang, w)
    sz = np.array([s["size"] for s in st]); al = np.array([s["align_rad"] for s in st])
    el = np.array([s["elong"] for s in st])
    # a random tiling would align at 45 degrees on average (uniform over 0..90)
    res[h] = {"n": n, "median_size": float(np.median(sz)), "median_elong": float(np.median(el)),
              "median_align_deg": float(np.degrees(np.median(al))),
              "frac_aligned_within_30deg": float((np.degrees(al) < 30).mean())}
    print(f"  h={h:.2f}  {n:4d} regions  median size {np.median(sz):5.0f} pts "
          f"({np.median(sz)*15*15/1e3:5.1f}k px^2)  elong {np.median(el):.2f}  "
          f"axis-vs-shape {np.degrees(np.median(al)):5.1f} deg  "
          f"within30 {(np.degrees(al)<30).mean():.0%}", flush=True)
json.dump(res, open("seg_sweep.json", "w"), indent=1)
