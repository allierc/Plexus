import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import beat as B

uv = B.load()
b, nb = B.mean_beat(uv)
ang, amp, aniso = B.axis_and_anisotropy(b)
ph, pamp, k = B.phase(b)
print(f"averaged {nb} beats; fundamental harmonic k={k}")
print(f"amplitude px: median {np.median(amp):.2f} p95 {np.percentile(amp,95):.2f}")
print(f"anisotropy l1/l2: median {np.median(aniso):.2f} p90 {np.percentile(aniso,90):.2f}")

def hsv(angle, sat, val, mod=np.pi):
    h = (angle % mod) / mod
    s = np.clip(sat, 0, 1); v = np.clip(val, 0, 1)
    return hsv_to_rgb(np.stack([h, s, v], -1))

fig, ax = plt.subplots(2, 3, figsize=(19, 12.4), facecolor="black")
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")

def show(a, m, t, cmap="viridis", p=(2, 98)):
    lo, hi = np.percentile(m, p)
    im = a.imshow(m, cmap=cmap, vmin=lo, vmax=hi)
    a.set_title(t, color="white", fontsize=10.5)
    plt.colorbar(im, ax=a, fraction=0.046).ax.tick_params(colors="white", labelsize=7)

show(ax[0, 0], amp, "beat amplitude (px)", "magma")
show(ax[0, 1], np.log10(aniso), "log10 anisotropy l1/l2  (high = moves along a line)", "cividis")
a = ax[0, 2]
a.imshow(hsv(ang, np.clip((aniso - 1) / 3, 0, 1), np.clip(amp / np.percentile(amp, 95), 0, 1)))
a.set_title("CONTRACTION AXIS (hue) x anisotropy (sat) x amplitude (val)", color="white", fontsize=10.5)
show(ax[1, 0], ph, "beat PHASE (rad) -- when in the cycle this point moves", "twilight", (0, 100))
a = ax[1, 1]
a.imshow(hsv(ph, np.ones_like(ph), np.clip(pamp / np.percentile(pamp, 95), 0, 1), mod=2 * np.pi))
a.set_title("PHASE (hue) x its strength (val)", color="white", fontsize=10.5)
# how sharply does the axis turn between neighbours? a boundary should light up
gy, gx = np.gradient(np.exp(2j * ang))
show(ax[1, 2], np.abs(gx) + np.abs(gy), "axis DISCONTINUITY  |grad exp(2i.theta)|", "inferno", (2, 97))
fig.suptitle(f"per-point beat descriptors, averaged over {nb} beats -- computed over TIME, "
             f"one grid point at a time, no spatial segmentation yet",
             color="white", fontsize=13)
fig.tight_layout(); fig.savefig("fig02_beat_descriptors.png", dpi=95, facecolor="black")
print("wrote fig02_beat_descriptors.png")
