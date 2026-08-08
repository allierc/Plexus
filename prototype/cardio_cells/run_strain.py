import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage as ndi
import beat as B, strain as ST, segment as S

uv = B.load(); b, nb = B.mean_beat(uv)
E = ST.strain_series(b)
ang, e2, aniso, tpk, emax = ST.contraction_axis(E)
print(f"strain: |e2| median {np.median(e2):.4f}  p95 {np.percentile(e2,95):.4f}")
print(f"anisotropy median {np.median(aniso):.3f}")

w = np.clip(aniso, 0, 1) * np.clip(e2 / np.percentile(e2, 90), 0, 1)
z = w * np.exp(2j * ang)
for sg in (1.0,):
    zs = ndi.gaussian_filter(z.real, sg) + 1j * ndi.gaussian_filter(z.imag, sg)
    zn = zs / np.maximum(ndi.gaussian_filter(w, sg), 1e-9)
gy, gx = np.gradient(zn); bnd = np.abs(gx) + np.abs(gy)

print("\n  STRAIN-based axis, watershed sweep:")
for h in (0.06, 0.10, 0.16, 0.24):
    lab, n = S.watershed_cells(bnd, w, h=h)
    st = S.region_stats(lab, ang, w)
    sz = np.array([s["size"] for s in st]); al = np.degrees([s["align_rad"] for s in st])
    el = np.array([s["elong"] for s in st])
    print(f"  h={h:.2f}  {n:4d} regions  median size {np.median(sz):5.0f}  elong {np.median(el):.2f}"
          f"  axis-vs-shape {np.median(al):5.1f} deg  within30 {(al<30).mean():.0%}", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(19, 6.6), facecolor="black")
for a in ax: a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")
ax[0].imshow(hsv_to_rgb(np.stack([(ang % np.pi)/np.pi, np.clip(aniso,0,1),
                                  np.clip(e2/np.percentile(e2,95),0,1)], -1)))
ax[0].set_title("CONTRACTION axis from STRAIN (hue)", color="white", fontsize=11)
im=ax[1].imshow(bnd, cmap="inferno", vmin=0, vmax=np.percentile(bnd,97))
ax[1].set_title("boundary strength |grad exp(2i.theta)| -- strain axis", color="white", fontsize=11)
plt.colorbar(im, ax=ax[1], fraction=0.046).ax.tick_params(colors="white", labelsize=7)
im=ax[2].imshow(tpk, cmap="twilight"); ax[2].set_title("frame of peak strain (activation sweep)", color="white", fontsize=11)
plt.colorbar(im, ax=ax[2], fraction=0.046).ax.tick_params(colors="white", labelsize=7)
fig.tight_layout(); fig.savefig("fig03_strain.png", dpi=95, facecolor="black")
np.savez("/tmp/strain_axis.npz", ang=ang, e2=e2, aniso=aniso, w=w, bnd=bnd, tpk=tpk)
print("wrote fig03_strain.png")
