"""calibrate_cells -- let the nuclei choose the scale, not a threshold I picked.

One nucleus per cell. So sweep the watershed depth and keep the setting whose regions hold ONE
nucleus each -- too coarse and a region swallows three, too fine and most hold none. The nuclei
come from the image; the regions come from the motion; neither saw the other.

The distribution matters more than the mean: a segmentation that is right for most cells and
merges a few gives a sharp peak at 1 with a small tail, while one that is merely right ON AVERAGE
spreads across 0, 1, 2 and 3.
"""
import json
import numpy as np
from scipy import ndimage as ndi
import beat as B, repro_seg as R

nb_ = np.load("/tmp/nuclei_best.npy")            # [n,3] y, x, sigma
D = np.load("/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/"
            "Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy", mmap_mode="r")
X0, Y0 = np.asarray(D[0, :, :, 0]), np.asarray(D[0, :, :, 1])
sx, sy = X0[0, 1] - X0[0, 0], Y0[1, 0] - Y0[0, 0]
gj = np.clip(((nb_[:, 1] - X0[0, 0]) / sx).round().astype(int), 0, 136)
gi = np.clip(((nb_[:, 0] - Y0[0, 0]) / sy).round().astype(int), 0, 136)
print(f"  {len(nb_)} nuclei -> grid coords; field 2048px, nucleus density "
      f"{len(nb_)/(2048*2048/1e6):.0f}/Mpx")

uv = B.load(); b, _ = B.mean_beat(uv)
rows = []
for sigma in (0.6, 0.8, 1.2):
    for h in (0.03, 0.05, 0.07, 0.10, 0.14):
        lab, n, ang, w = R.seg_from(b, sigma, h, min_size=3)
        cnt = np.bincount(lab[gi, gj], minlength=n + 1)[1:]
        per = np.bincount(np.clip(cnt, 0, 5), minlength=6)
        rows.append({"sigma": sigma, "h": h, "n": n, "mean_nuc": float(cnt.mean()),
                     "frac1": float((cnt == 1).mean()), "hist": per.tolist()})
        print(f"  sigma={sigma:.1f} h={h:.2f}  {n:4d} regions  nuclei/region mean "
              f"{cnt.mean():.2f}  =1: {(cnt==1).mean():.0%}   hist[0..5+] {per.tolist()}", flush=True)
json.dump(rows, open("calib_cells.json", "w"), indent=1)
best = max(rows, key=lambda r: r["frac1"])
print(f"\n  BEST by one-nucleus-per-region: sigma={best['sigma']} h={best['h']} "
      f"-> {best['n']} cells, {best['frac1']:.0%} hold exactly one nucleus")
