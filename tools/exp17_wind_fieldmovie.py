"""exp17: THE FIELD AS A MOVIE (Cedric, 2026-10-04: "make the field slides movies, same colours as the flow slides").
Local. The wind field of one run frame by frame (tools/exp17_wind.py: the messages m_ji = W_ji tanh(z_j) Omega_i as arrows
sender -> receiver, gridded, smoothed over S um; excitatory m > 0 and inhibitory m < 0 apart), from above and from the
side, as arrows every 3rd cell -- excitatory red, inhibitory cyan, length and opacity by the square root of the local
strength on ONE scale over the whole movie (the 99th percentile of |w|) -- over the RECORDED dF/F in grey (the flow
slides' colours); below, the brain-mean dF/F, recorded (green) and learned (white), with a cursor.

    PYTHONPATH=src:tools python tools/exp17_wind_fieldmovie.py zap_g17_mesh4 --sigma 25
Writes presentation/Movies/field_<run>_sigma<S>.mp4 (+ .png) and copies in png/.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
RED, CYAN = np.array([0.95, 0.10, 0.06]), np.array([0.0, 0.85, 1.0])


def main(run, sigma, device="cuda:0", step=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import exp17_wind as W
    from exp17_wind_views import recorded_act
    from plexus.tasks import trace_recording as TR
    V = {v: W.fields(run, 160, device, sigma, v) for v in ("top", "side")}
    D0 = V["top"]
    rec, fr = D0["rec"], D0["fr"]
    F = len(fr)
    cond = rec["condition"][fr]
    tm = fr * 0.914 / 60
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    names = [str(x) for x in rec["names"]]
    mo, mp = rec["dff"][fr].mean(1), D0["pred"].mean(1)
    Hf = 9.4
    fig = plt.figure(figsize=(16, Hf), facecolor="black")
    Q, bgims, bgs = {}, {}, {}
    for v, (y0, h) in (("top", (0.50, 0.44)), ("side", (0.19, 0.27))):
        D = V[v]
        ins = D["inside"]
        a = recorded_act(D)
        bgs[v] = np.clip(a / np.percentile(a[:, ins], 99), 0, 1) * ins * 0.42
        ny, nx = ins.shape
        yy, xx = np.mgrid[1:ny:step, 1:nx:step]
        keep = ins[1::step, 1::step]
        for j, (k, col, lab) in enumerate((("ex", RED, "EXCITATORY (red)"), ("in", CYAN, "INHIBITORY (cyan)"))):
            w = D["wind"][k]                                                   # [F, 2, ny, nx]
            ref = np.percentile(np.hypot(w[:, 0], w[:, 1])[:, ins], 99)
            ax = fig.add_axes([0.005 + 0.5 * j, y0, 0.49, h])
            ax.set_facecolor("black")
            ax.axis("off")
            bgims[(v, k)] = ax.imshow(np.zeros((ny, nx, 3)), origin="lower", interpolation="bilinear")
            q = ax.quiver(xx[keep], yy[keep], np.zeros(keep.sum()), np.zeros(keep.sum()), color=np.tile(np.r_[col, 1.0], (keep.sum(), 1)),
                          angles="xy", scale_units="xy", scale=1.0, width=0.0024, headwidth=3.2, headlength=3.5)
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.set_aspect("equal")
            Q[(v, k)] = (q, w[:, :, 1::step, 1::step][:, :, keep], ref, col)
            fig.text(0.01 + 0.5 * j, y0 + h + 0.008, f"{'from above' if v == 'top' else 'from the side'}: {lab} field",
                     color="white", fontsize=12)
    m_ = fig.add_axes([0.05, 0.035, 0.90, 0.10])
    m_.set_facecolor("black")
    for k_, sp in m_.spines.items():
        sp.set_visible(k_ in ("left", "bottom"))
        sp.set_color("0.5")
    for i_, (a_, b_) in enumerate(zip(st, en)):
        m_.axvspan(tm[a_], tm[b_], color=("0.30" if i_ % 2 else "0.18"), alpha=0.6, lw=0)
        m_.text((tm[a_] + tm[b_]) / 2, 1.02, names[int(cond[a_])], color="0.75", fontsize=7, ha="center", va="bottom",
                transform=m_.get_xaxis_transform())
    m_.plot(tm, mo, color="#2ca02c", lw=0.8)
    m_.plot(tm, mp, color="white", lw=0.8)
    m_.set_xlim(tm[0], tm[-1])
    m_.set_yticks([])
    m_.tick_params(colors="0.6", labelsize=7)
    cur = m_.axvline(tm[0], color="#ff7f0e", lw=0.9)
    fig.text(0.05, 0.155, "brain-mean dF/F: recorded (green), learned (white); time, min", color="0.75", fontsize=9)
    fig.text(0.5, 0.995, f"{run}: the messages' field, smoothed over {sigma:g} um, frame by frame; arrows every {step}rd cell, "
             "length by the square root of the strength (one scale over the movie); grey = the recorded dF/F",
             color="0.75", fontsize=10, ha="center", va="top")
    t_txt = fig.text(0.70, 0.975, "", color="0.75", fontsize=10, va="top")
    tmp = tempfile.mkdtemp(prefix="fieldmovie_")
    for f in range(F):
        for (v, k), (q, w, ref, col) in Q.items():
            bgims[(v, k)].set_data(np.repeat(bgs[v][f][..., None], 3, -1))
            u, vv = w[f, 0], w[f, 1]
            m = np.hypot(u, vv)
            r = np.sqrt(np.clip(m / ref, 0, 1))
            L = 2.8 * r / np.maximum(m, 1e-12)
            q.set_UVC(u * L, vv * L)
            q.set_facecolor(np.concatenate([np.tile(col, (len(r), 1)), np.clip(r, 0.0, 1.0)[:, None]], 1))
        t_txt.set_text(f"{names[int(cond[f])]}   t = {tm[f]:5.1f} min")
        cur.set_xdata([tm[f]] * 2)
        fig.savefig(os.path.join(tmp, f"{f:05d}.png"), dpi=100, facecolor="black")
    plt.close(fig)
    path = os.path.join(EXP, "presentation", "Movies", f"field_{run}_sigma{sigma:g}.mp4")
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error", "-framerate", "25", "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", path], check=True)
    shutil.copy(os.path.join(tmp, f"{F // 2:05d}.png"), path.replace(".mp4", ".png"))
    for f_ in (path, path.replace(".mp4", ".png")):
        shutil.copy(f_, os.path.join(EXP, "png", os.path.basename(f_)))
    shutil.rmtree(tmp)
    print(f"[fieldmovie] {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sigma", type=float, default=25.0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    main(a.run, a.sigma, a.device)
