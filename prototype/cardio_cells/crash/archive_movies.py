"""archive_movies -- the crash-test candidates as loop movies, in the ablation format.

Each panel is one tracer's loop over the 150-frame window: GREEN is the reference the estimator was
asked to reproduce, RED is what the rollout with that candidate's parameters actually did. The dot
is the current frame, which is the only way timing shows in a picture of a shape. The frame colour
is that loop's own score, green through red.

Same conventions as log/cardio_mpm/ablation_p3_b49_s2_fs2/: centred and scaled per panel, so a
panel says whether the SHAPE and the TIMING match and never whether the size does.
"""
import glob, json, os, subprocess, sys, tempfile
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, "/workspace/Plexus/discovery_cardio_mpm")

OUT = "/workspace/Plexus/log/cardio_mpm/synthetic_crash_test"


def per_loop_score(sim, real):
    """Per-tracer loopscore, via the campaign's own inherited implementation."""
    import torch, harmonic_inherited as H
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    return H._pernode_score(t(sim), t(real), None).numpy()


def render(real, sim, out, label, fps=12, loops=2):
    G, N, _ = real.shape
    n = int(np.ceil(np.sqrt(N)))
    sc = per_loop_score(sim, real)
    cmap = mcolors.LinearSegmentedColormap.from_list("gr", ["#B3261E", "#B26B00", "#1B7F3B"])
    norm = mcolors.Normalize(vmin=-0.3, vmax=1.0)
    fig, axes = plt.subplots(n, n, figsize=(n * 1.22, n * 1.22 + 0.95), facecolor="black")
    P, Q, art = [], [], []
    for k in range(n * n):
        ax = axes[k // n, k % n]
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
        if k >= N:
            for sp in ax.spines.values(): sp.set_visible(False)
            P.append(None); Q.append(None); art.append(None); continue
        p, q = real[:, k], sim[:, k]
        c = p.mean(0); p, q = p - c, q - c
        P.append(p); Q.append(q)
        lg, = ax.plot([], [], color="#22DD22", lw=1.1)
        lr, = ax.plot([], [], color="#FF3B30", lw=1.1)
        dg, = ax.plot([], [], "o", color="#22DD22", ms=3.0)
        dr, = ax.plot([], [], "o", color="#FF3B30", ms=3.0)
        art.append((lg, lr, dg, dr))
        r = max(np.abs(np.concatenate([p, q])).max(), 1e-12) * 1.15
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_color(cmap(norm(sc[k]))); sp.set_linewidth(2.4)
        ax.text(0.03, 0.85, f"{sc[k]:+.2f}", color="white", fontsize=6, fontweight="bold",
                transform=ax.transAxes)
    title = fig.suptitle("", color="white", fontsize=10, y=0.995)
    fig.subplots_adjust(hspace=.06, wspace=.06, top=.905, bottom=.008, left=.008, right=.992)
    tmp = tempfile.mkdtemp(prefix="loopmov_")
    for f in range(G * loops):
        t = f % G
        for k in range(N):
            lg, lr, dg, dr = art[k]; p, q = P[k], Q[k]
            lg.set_data(p[:t + 1, 0], p[:t + 1, 1]); lr.set_data(q[:t + 1, 0], q[:t + 1, 1])
            dg.set_data([p[t, 0]], [p[t, 1]]); dr.set_data([q[t, 0]], [q[t, 1]])
        title.set_text(f"{label}    frame {t+1}/{G}    green = reference, red = rollout with these "
                       f"parameters    mean per-loop score {sc.mean():+.3f}")
        fig.savefig(os.path.join(tmp, f"f_{f:05d}.png"), dpi=100, facecolor="black")
    plt.close(fig)
    from plexus.plot import _ffmpeg
    r = subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", os.path.join(tmp, "f_%05d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
                       capture_output=True, text=True)
    sz = os.path.getsize(out) if os.path.exists(out) else 0
    for p_ in glob.glob(os.path.join(tmp, "*.png")): os.remove(p_)
    os.rmdir(tmp)
    return sz, float(sc.mean())


WANT = [
    ("tracks_round5_score_s1.npz", "theta_true|gauged", "01_ceiling_theta_true",
     "CEILING: the true per-cell parameters"),
    ("tracks_round5_scorebw_s1.npz", "s90210/T8/eiv_box0.1_10|gauged", "02_winner_eiv_box",
     "THE WINNER: EIV-corrected + box-constrained, measured F injected, T=8"),
    ("tracks_round5_score_s1.npz", "s90210/T8/naive|gauged", "03_naive_no_correction",
     "NAIVE: same data, no errors-in-variables correction and no box"),
]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    made = []
    for fn, key, stem, label in WANT:
        if not os.path.exists(fn):
            print(f"  missing {fn}"); continue
        z = np.load(fn)
        if key not in z:
            print(f"  {fn} has no {key}; has {[k for k in z if '|' in k][:6]}"); continue
        real, sim = np.asarray(z["real20"]), np.asarray(z[key])
        out = os.path.join(OUT, f"{stem}.mp4")
        sz, m = render(real, sim, out, label)
        print(f"  {stem}.mp4  {sz/1e6:.1f} MB  mean per-loop {m:+.3f}", flush=True)
        made.append({"file": f"{stem}.mp4", "source": fn, "candidate": key,
                     "mean_per_loop_score": m, "bytes": sz})
    json.dump(made, open(os.path.join(OUT, "movies.json"), "w"), indent=1)
