"""exp17: THE FIELD ITSELF, NOT ITS PARTICLES (Cedric, 2026-10-04: "slides to show the field, not the flows, for the
smooth and the mid smooth"). Local. The time-averaged wind field of one run (tools/exp17_wind.py, saved as
<run>_wind_fields[_sigma<S>][_<view>].npz: the messages m_ji = W_ji tanh(z_j) Omega_i as arrows sender -> receiver,
gridded, smoothed over S um, averaged over the 800 movie frames), excitatory (m > 0) and inhibitory (m < 0) apart, from
above and from the side: arrows every 3rd cell, length and opacity by sqrt of the local strength |w|, excitatory red, inhibitory
blue, over |w| in grey (where the field is strong).

    PYTHONPATH=src:tools python tools/exp17_wind_fieldmaps.py zap_g17_mesh4 25 10
Writes presentation/figs/field_<run>_sigma<S>.png (+ png/).
"""
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
G = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "log", "training", "zapbench")


def main(run, sigmas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for S in sigmas:
        suf = "" if S == 25 else f"_sigma{S:g}"
        fig = plt.figure(figsize=(16, 9.4), facecolor="black")
        for r, (view, y0, h) in enumerate((("top", 0.50, 0.44), ("side", 0.19, 0.27))):
            z = np.load(os.path.join(G, run, "results", f"{run}_wind_fields{suf}{'' if view == 'top' else '_side'}.npz"))
            ins = z["inside"]
            for j, (k, col, lab) in enumerate((("ex", (1.0, 0.25, 0.2), "EXCITATORY"), ("inh", (0.0, 0.85, 1.0), "INHIBITORY"))):
                w = z[k]                                                    # [2, ny, nx]
                mag = np.hypot(w[0], w[1]) * ins
                ref = np.percentile(mag[ins], 99)
                ax = fig.add_axes([0.005 + 0.5 * j, y0, 0.49, h])
                ax.set_facecolor("black")
                ax.imshow(np.clip(mag / ref, 0, 1), cmap="gray", vmin=0, vmax=1.6, origin="lower", interpolation="bilinear")
                ny, nx = mag.shape
                yy, xx = np.mgrid[1:ny:3, 1:nx:3]
                u, v = w[0][1::3, 1::3], w[1][1::3, 1::3]
                m_ = np.hypot(u, v)
                keep = ins[1::3, 1::3] & (m_ > 0.05 * ref)
                a_ = np.clip(np.sqrt(m_[keep] / ref), 0.3, 1.0)
                cols = np.concatenate([np.tile(col, (keep.sum(), 1)), a_[:, None]], 1)
                L_ = 2.8 * np.sqrt(np.clip(m_[keep] / ref, 0, 1))             # length ~ sqrt(strength), <= 2.8 cells
                ax.quiver(xx[keep], yy[keep], u[keep] / m_[keep] * L_, v[keep] / m_[keep] * L_, color=cols, angles="xy",
                          scale_units="xy", scale=1.0, width=0.0024, headwidth=3.2, headlength=3.5)
                ax.set_xlim(0, nx)
                ax.set_ylim(0, ny)
                ax.set_aspect("equal")
                ax.axis("off")
                fig.text(0.01 + 0.5 * j, y0 + h + 0.008, f"{'from above' if view == 'top' else 'from the side'}: {lab} "
                         f"field, time-averaged", color="white", fontsize=12)
        # the brain-mean dF/F over the 2 h, recorded (green) and learned (white) (Cedric, 2026-10-04)
        z = np.load(os.path.join(G, run, "results", f"{run}_movie.npz"))
        t = np.asarray(z["r2_t"]) * 0.914 / 60
        m_ = fig.add_axes([0.05, 0.035, 0.90, 0.10])
        m_.set_facecolor("black")
        for k_, sp in m_.spines.items():
            sp.set_visible(k_ in ("left", "bottom"))
            sp.set_color("0.5")
        m_.plot(t, z["mean_obs_all"], color="#2ca02c", lw=0.7)
        m_.plot(t, z["mean_pred_all"], color="white", lw=0.7)
        m_.set_xlim(t[0], t[-1])
        m_.set_yticks([])
        m_.tick_params(colors="0.6", labelsize=7)
        fig.text(0.05, 0.145, "brain-mean dF/F: recorded (green), learned (white); time, min", color="0.75", fontsize=9)
        fig.text(0.5, 0.995, f"{run}: the messages' field, smoothed over {S:g} um, averaged over the 2 h; arrows every 3rd "
                 "cell, length by the square root of the strength; grey = |field|", color="0.75", fontsize=10, ha="center", va="top")
        path = os.path.join(EXP, "presentation", "figs", f"field_{run}_sigma{S:g}.png")
        fig.savefig(path, dpi=110, facecolor="black")
        plt.close(fig)
        shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
        print("[field]", path)


if __name__ == "__main__":
    main(sys.argv[1], [float(a) for a in sys.argv[2:]])
