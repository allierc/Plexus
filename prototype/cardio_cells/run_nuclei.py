import numpy as np, tifffile
from scipy import ndimage as ndi
import nuclei as NU
RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
with tifffile.TiffFile(f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif") as tf:
    img = tf.pages[0].asarray()
b = NU.detect(img)
for pol, arr in b.items():
    r = arr[:, 2] * np.sqrt(2)
    print(f"  {pol:>6s}: {len(arr):5d} blobs   median radius {np.median(r):5.1f} px "
          f"-> diameter {2*np.median(r):5.1f} px   density {len(arr)/ (2048*2048/1e6):.0f} per Mpx")
np.savez("/tmp/nuclei.npz", **{k: v for k, v in b.items()})
