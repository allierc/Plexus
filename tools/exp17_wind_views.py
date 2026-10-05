"""exp17: THE FLOW FROM ABOVE AND FROM THE SIDE (Cedric, 2026-10-04: "excitatory and inhibitory, top view, side view,
the brain-mean dF/F; for the flows the background activity in grey, excitatory in red, inhibitory in blue"). Local.

One run's learned messages as wind (tools/exp17_wind.py: m_ji = W_ji tanh(z_j) Omega_i deposited as arrows sender ->
receiver, smoothed over 25 um, excitatory m > 0 and inhibitory m < 0 apart) in two projections of the same edges:
  top    head left, seen from above (exp17_wind's maps)
  side   head left, depth down: the arrows projected on the sagittal plane
four maps -- top excitatory | top inhibitory over side excitatory | side inhibitory -- each on the RECORDED dF/F of
that view in grey (gridded, smoothed over 25 um), excitatory particles red, inhibitory blue, one speed scale per view;
below, the brain-mean dF/F, recorded (green) and learned (white), with a cursor.

    PYTHONPATH=src:tools python tools/exp17_wind_views.py zap_g17_mesh4
Writes presentation/Movies/flow_views_<run>.mp4 (+ .png) and copies in png/.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")


def recorded_act(D):
    """The recorded dF/F at the movie frames, gridded on D's view and smoothed as the wind."""
    from scipy.ndimage import gaussian_filter
    from scipy.sparse import csr_matrix
    import exp17_wind as W
    x0, y0, h = D["grid"]
    ny, nx = D["inside"].shape
    P = D["P"]
    cid = np.clip(((P[:, 1] - y0) / h).astype(int), 0, ny - 1) * nx + np.clip(((P[:, 0] - x0) / h).astype(int), 0, nx - 1)
    A = csr_matrix((np.ones(len(cid)), (cid, np.arange(len(cid)))), shape=(ny * nx, len(cid)))
    cnt = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    X = D["rec"]["dff"][D["fr"]].astype(np.float64)
    a = (A @ X.T).T / cnt
    return np.stack([gaussian_filter(r.reshape(ny, nx), W.SIGMA_UM / h) for r in a]).astype(np.float32)


def main(run, n_part=5000, device="cuda:0", sigma=None, modes=("split", "combined")):
    import exp17_wind as W
    sigma = W.SIGMA_UM if sigma is None else sigma
    views = ("top", "oblique", "side") if "combined" in modes else ("top", "side")
    V = {v: W.fields(run, 160, device, sigma, v) for v in views}
    bg = {}
    for v, D in V.items():
        a = recorded_act(D)
        bg[v] = np.clip(a / np.percentile(a[:, D["inside"]], 99), 0, 1)
    suf = "" if sigma == W.SIGMA_UM else f"_sigma{sigma:g}"
    for md in modes:
        render(run, V, bg, n_part, combined=md == "combined", suffix=suf)   # Cedric, 2026-10-04: "red and blue in one movie"


def render(run, V, bg, n_part, combined, suffix=""):
    """split: four maps (top / side x excitatory / inhibitory); combined: one map per view, red and blue together."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import exp17_wind as W
    from plexus.tasks import trace_recording as TR
    rng = np.random.default_rng(0)
    RED, BLUE = (1.0, 0.22, 0.18), (0.30, 0.55, 1.0)
    maps = {}
    for v, D in V.items():
        ins = D["inside"]
        vref = max(np.percentile(D["speed"]["ex"][:, ins], 99), np.percentile(D["speed"]["in"][:, ins], 99))
        maps[v] = {"ex": W._Map(D["wind"]["ex"], ins, vref, RED, n_part, rng),
                   "in": W._Map(D["wind"]["in"], ins, vref, BLUE, n_part, rng)}
    D0 = V["top"]
    rec, fr = D0["rec"], D0["fr"]
    F = len(fr)
    cond = rec["condition"][fr]
    tm = fr * 0.914 / 60
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    names = [str(x) for x in rec["names"]]
    mo, mp = rec["dff"][fr].mean(1), D0["pred"].mean(1)
    # combined (Cedric, 2026-10-04: "a 2 x 2 template, not 3 x 1"): above | oblique over side | the brain-mean dF/F
    rows = (("top", (4.35, 4.55)), ("side", (1.75, 2.25))) if not combined else \
        (("top", (4.75, 4.15)), ("oblique", (4.75, 4.15)), ("side", (0.55, 3.75)))
    Hf = 9.2
    fig = plt.figure(figsize=(16, Hf), facecolor="black")
    ims = {}
    for v, (y0_, h_) in rows:
        ny, nx = V[v]["inside"].shape
        S = maps[v]["ex"].S
        panels = ((("both", {"top": 0.005, "oblique": 0.505, "side": 0.005}[v], 0.49),) if combined
                  else (("ex", 0.005, 0.49), ("in", 0.505, 0.49)))
        for k, x_, w_ in panels:
            ax = fig.add_axes([x_, y0_ / Hf, w_, h_ / Hf])
            ax.axis("off")
            ims[(v, k)] = ax.imshow(np.zeros((ny * S, nx * S, 3)), origin="lower", interpolation="bilinear")
            lab = {"ex": "EXCITATORY (red)", "in": "INHIBITORY (blue)", "both": "excitatory (red), inhibitory (cyan)"}[k]
            vn = {"top": "from above", "side": "from the side", "oblique": "oblique, from 45 deg above"}[v]
            fig.text(x_ + 0.005, (y0_ + h_ + 0.05) / Hf, f"{vn}: {lab}",
                     color="white", fontsize=11, va="bottom")
    m_ = fig.add_axes([0.05, 0.28 / Hf, 0.90, 0.95 / Hf] if not combined else [0.54, 1.3 / Hf, 0.43, 2.4 / Hf])
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
    fig.text(*((0.05, 1.42 / Hf) if not combined else (0.54, 3.85 / Hf)), "brain-mean dF/F: recorded (green), learned "
             "(white); time, min", color="0.75", fontsize=9 if not combined else 11)
    cur = m_.axvline(tm[0], color="#ff7f0e", lw=0.9)
    t_txt = fig.text(0.70, (Hf - 0.05) / Hf, "", color="0.75", fontsize=10, va="top")
    tmp = tempfile.mkdtemp(prefix="wind_views_")
    for f in range(F):
        for v, _ in rows:
            ins = V[v]["inside"]
            S = maps[v]["ex"].S
            g = (bg[v][f] * ins * 0.42)[..., None].repeat(3, -1).repeat(S, 0).repeat(S, 1)     # the activity, grey
            for k in ("ex", "in"):
                maps[v][k].step(f)
            if combined:
                # Cedric, 2026-10-04: "adjust the red and blue LUT so that we do not reach the white": each map's
                # trail intensity tone-mapped (tanh) onto a pure red / pure blue, the grey under it faded by the
                # stronger of the two; red + blue overlap gives magenta, never white
                a_e = np.tanh(maps[v]["ex"].canvas[..., 0] / 0.6)[..., None]
                a_i = np.tanh(maps[v]["in"].canvas[..., 2] / 0.6)[..., None]
                # the two hues mixed by their weights, at the stronger one's intensity: red + cyan overlap goes grey-lilac,
                # never white (a plain sum of red and cyan is white)
                a_m = np.maximum(a_e, a_i)
                hue = (a_e * PURE_RED + a_i * CYAN) / np.maximum(a_e + a_i, 1e-6)
                img = g * (1 - a_m) + a_m * hue
                ims[(v, "both")].set_data(np.clip(img, 0, 1))
            else:
                for k in ("ex", "in"):
                    ims[(v, k)].set_data(np.clip(g + maps[v][k].canvas, 0, 1))
        t_txt.set_text(f"{names[int(cond[f])]}   t = {tm[f]:5.1f} min")
        cur.set_xdata([tm[f]] * 2)
        fig.savefig(os.path.join(tmp, f"{f:05d}.png"), dpi=100, facecolor="black")
    plt.close(fig)
    path = os.path.join(EXP, "presentation", "Movies", f"flow_views_{run}{'_combined' if combined else ''}{suffix}.mp4")
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error", "-framerate", "25", "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", path], check=True)
    shutil.copy(os.path.join(tmp, f"{F // 2:05d}.png"), path.replace(".mp4", ".png"))
    for f_ in (path, path.replace(".mp4", ".png")):
        shutil.copy(f_, os.path.join(EXP, "png", os.path.basename(f_)))
    shutil.rmtree(tmp)
    print(f"[views] {path}")


PURE_RED, PURE_BLUE = np.array([0.95, 0.10, 0.06]), np.array([0.12, 0.32, 0.95])
CYAN = np.array([0.0, 0.85, 1.0])                      # Cedric, 2026-10-04: the inhibitory in cyan, blue hard to see

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--sigma", type=float, default=None, help="the wind's smoothing, um (default exp17_wind's 25)")
    ap.add_argument("--modes", nargs="+", default=["split", "combined"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    main(a.run, device=a.device, sigma=a.sigma, modes=a.modes)
