"""null_test -- getting the COUNT right is not the same as getting the PLACES right.

465 motion regions against 472 nuclei is a 1.5% agreement in density, and it is tempting. But any
tiling with the right average cell size would match a count. The question is whether each region
contains ITS OWN nucleus, and the null that answers it keeps the segmentation exactly as it is and
moves the nuclei -- same number, same spatial statistics, wrong correspondence.

If the real nuclei score no better than displaced ones, the method has found the SCALE of the cells
and not the cells.
"""
import numpy as np
import beat as B, repro_seg as R

nb_ = np.load("/tmp/nuclei_best.npy")
D = np.load("/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/"
            "Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy", mmap_mode="r")
X0, Y0 = np.asarray(D[0, :, :, 0]), np.asarray(D[0, :, :, 1])
sx, sy = X0[0, 1] - X0[0, 0], Y0[1, 0] - Y0[0, 0]

def to_grid(yx):
    gj = np.clip(((yx[:, 1] - X0[0, 0]) / sx).round().astype(int), 0, 136)
    gi = np.clip(((yx[:, 0] - Y0[0, 0]) / sy).round().astype(int), 0, 136)
    return gi, gj

uv = B.load(); b, _ = B.mean_beat(uv)
lab, n, ang, w = R.seg_from(b, 0.8, 0.03, min_size=3)
gi, gj = to_grid(nb_)
cnt = np.bincount(lab[gi, gj], minlength=n + 1)[1:]
real1 = (cnt == 1).mean(); realv = cnt.var()

rng = np.random.default_rng(0)
f1, vv = [], []
for k in range(60):
    dy, dx = rng.integers(-60, 60, 2) * 1.0
    sh = nb_.copy(); sh[:, 0] = (sh[:, 0] + dy) % 2048; sh[:, 1] = (sh[:, 1] + dx) % 2048
    gi2, gj2 = to_grid(sh)
    c = np.bincount(lab[gi2, gj2], minlength=n + 1)[1:]
    f1.append((c == 1).mean()); vv.append(c.var())
f1 = np.array(f1); vv = np.array(vv)
print(f"  {n} regions, {len(nb_)} nuclei, mean {cnt.mean():.2f} per region")
print(f"  exactly one nucleus:   real {real1:.1%}   nuclei displaced {f1.mean():.1%} +/- {f1.std():.1%}"
      f"   z = {(real1-f1.mean())/max(f1.std(),1e-9):+.1f}")
print(f"  variance of the count: real {realv:.3f}   displaced {vv.mean():.3f} +/- {vv.std():.3f}"
      f"   z = {(realv-vv.mean())/max(vv.std(),1e-9):+.1f}   (lower = more one-per-region)")
print()
print("  A segmentation that finds cells puts MORE regions on exactly one nucleus and has LOWER")
print("  count variance than the same regions with the nuclei moved. Anything else means the")
print("  density is right and the placement is not.")
