"""check_labels -- the label image the engine reads and the labels the recording's nodes carry
must be the SAME map in the SAME frame. Samples the padded tif exactly as LabelImageField does
(flip rows, permute, nearest index) at every node's world rest position."""
import os, sys, numpy as np, tifffile, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import recording as R
rec = R.load()
img = tifffile.imread(os.path.join(HERE, "data", "cells_2560.tif"))
v = torch.tensor(img[::-1, :].astype("int64")).permute(1, 0).contiguous()
nx, ny = v.shape
pos = rec["pos"][0]
gx = (pos[:, 0].clamp(0, 1 - 1e-6) * nx).long().clamp(0, nx - 1)
gy = (pos[:, 1].clamp(0, 1 - 1e-6) * ny).long().clamp(0, ny - 1)
sampled = v[gx, gy]
agree = (sampled == rec["labels"]).float().mean()
print(f"nodes whose sampled label equals their own: {agree:.4f}  (labels 1..{rec['n_cells']}, "
      f"sampled range {sampled.min()}..{sampled.max()})")
assert agree > 0.98, "the label image and the node labels are not in the same frame"
