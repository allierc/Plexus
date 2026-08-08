import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import beat as B, strain as ST, scale as SC

uv = B.load(); b, _ = B.mean_beat(uv)
ang_d, amp, aniso = B.axis_and_anisotropy(b)
w_d = np.clip((aniso - 1) / 6, 0, 1) * np.clip(amp / np.percentile(amp, 90), 0, 1)
E = ST.strain_series(b, sigma=0.6)
ang_s, e2, _, tpk, _ = ST.contraction_axis(E)
w_s = np.clip(e2 / np.percentile(e2, 90), 0, 1)

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
res = {}
for nm, a_, w_, c in (("displacement axis", ang_d, w_d, "tab:blue"),
                      ("strain axis", ang_s, w_s, "tab:red")):
    C = SC.corr_vs_r(a_, w_)
    r = np.arange(1, len(C) + 1)
    half = np.argmax(C < 0.5) + 1 if (C < 0.5).any() else len(C)
    e1 = np.argmax(C < np.exp(-1)) + 1 if (C < np.exp(-1)).any() else len(C)
    res[nm] = (C, half, e1)
    ax[0].plot(r * 15, C, "-o", ms=3, color=c, label=f"{nm}  (half at {half*15} px)")
    print(f"  {nm:<20s} C(1)={C[0]:.3f}  C(2)={C[1]:.3f}  half-decay {half} grid pts = {half*15} px"
          f"   1/e at {e1} pts = {e1*15} px")
ax[0].axhline(0.5, color="0.6", lw=0.8, ls="--")
ax[0].set_xlabel("separation (pixels)"); ax[0].set_ylabel("axis agreement  <cos 2 d(theta)>")
ax[0].set_title("how far does the contraction axis stay the same?"); ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

# a cardiomyocyte is ~100 um; what is a pixel? infer nothing -- just mark plausible cell sizes
for px, lab in ((100, ""), (200, ""), (400, "")):
    ax[0].axvline(px, color="0.85", lw=0.7, zorder=0)
ax[1].imshow(tpk, cmap="twilight"); ax[1].set_xticks([]); ax[1].set_yticks([])
ax[1].set_title("frame of peak strain -- the activation sweep")
fig.tight_layout(); fig.savefig("fig04_scale.png", dpi=110)
print("wrote fig04_scale.png")
