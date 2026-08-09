"""mpm_gt_vs_learned -- the material itself, ground truth beside the learned parameters.

The loop movies show 100 tracers. This shows the SHEET: every material point, coloured by the cell
it belongs to, ground truth on the left and the rollout with the recovered per-cell (E, gain) on the
right, from the same initial condition and free-running for a whole beat. A third panel carries the
per-particle position error so the disagreement is visible where it happens rather than averaged.

Uses crash_test.rollout unmodified (keep_full=True), so what is drawn is the same integration the
scores were computed from, not a second implementation.
"""
import os, glob, json, subprocess, sys, tempfile
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from types import SimpleNamespace
sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/Plexus/prototype/cardio_cells/algebraic")
import crash_test as CT

OUT = "/workspace/Plexus/log/cardio_mpm/synthetic_crash_test"
T0, WIN, KEY = 165, 150, "round5_norm_s90210_sF0.0039|T8|eiv_box"


def main():
    args = SimpleNamespace(device="cuda:1", cells=100, per_parent=100, n_grid=128,
                           warmup=T0, window=WIN, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, print)
        th = sy.theta_true.double()
        Z = np.load("theta_round5.npz")
        key = KEY if KEY in Z else [k for k in Z if "eiv_box" in k and "s90210" in k][0]
        hat = torch.tensor(Z[key], device=th.device, dtype=th.dtype)
        print(f"  learned = {key}", flush=True)
        tr = {20: torch.arange(sy.Np, device=th.device)[:1]}
        gt = CT.rollout(sy, th, T0, WIN, tr, keep_full=True)
        lr = CT.rollout(sy, hat, T0, WIN, tr, keep_full=True)

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def grab(r):
        for k in ("full", "X", "pos", "positions"):
            if isinstance(r, dict) and k in r and r[k] is not None: return _np(r[k])
        if isinstance(r, (tuple, list)):
            for e in r:
                if getattr(e, "ndim", 0) == 3: return _np(e)
        raise SystemExit(f"cannot find full positions: {type(r)} "
                         f"{list(r) if isinstance(r, dict) else len(r)}")
    A, B = grab(gt), grab(lr)
    print("  shapes", A.shape, B.shape, flush=True)
    Np = A.shape[1]
    per = Np // 100
    cid = np.arange(Np) // per
    rng = np.random.default_rng(4)
    LUT = hsv_to_rgb(np.stack([rng.permutation(100) / 100,
                               np.full(100, .72), np.ones(100)], -1))
    col = LUT[np.clip(cid, 0, 99)]
    err = np.linalg.norm(A - B, axis=-1)
    dx = 1.0 / 128
    print(f"  rms error over the window: {err.mean()/dx:.4f} dx   peak {err.max()/dx:.3f} dx")

    tmp = tempfile.mkdtemp(prefix="mpmgt_")
    vmax = float(np.percentile(err, 99.5)) / dx
    for t in range(A.shape[0]):
        fig, ax = plt.subplots(1, 3, figsize=(19.5, 7.1), facecolor="black")
        for a in ax:
            a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal")
            a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")
        ax[0].scatter(A[t, :, 0], A[t, :, 1], s=2.2, c=col, marker=".", linewidths=0)
        ax[0].set_title("GROUND TRUTH -- the planted per-cell (E, gain)", color="white", fontsize=11)
        ax[1].scatter(B[t, :, 0], B[t, :, 1], s=2.2, c=col, marker=".", linewidths=0)
        ax[1].set_title("LEARNED -- solved from the injected data, then free-run",
                        color="white", fontsize=11)
        s = ax[2].scatter(B[t, :, 0], B[t, :, 1], s=2.2, c=err[t] / dx, cmap="inferno",
                          vmin=0, vmax=max(vmax, 1e-9), marker=".", linewidths=0)
        ax[2].set_title("|GT - learned| per particle, in grid cells", color="white", fontsize=11)
        if t == 0:
            cb = fig.colorbar(s, ax=ax[2], fraction=.046); cb.ax.tick_params(colors="white", labelsize=7)
        fig.suptitle(f"synthetic crash test -- same initial condition, free 150-frame rollout    "
                     f"frame {t+1}/{A.shape[0]}    rms {err[t].mean()/dx:.4f} dx",
                     color="white", fontsize=12, y=0.995)
        fig.tight_layout()
        fig.savefig(os.path.join(tmp, f"f_{t:05d}.png"), dpi=92, facecolor="black")
        plt.close(fig)
        if t % 40 == 0: print(f"   frame {t}", flush=True)
    from plexus.plot import _ffmpeg
    out = os.path.join(OUT, "04_mpm_gt_vs_learned.mp4")
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", "20",
                    "-i", os.path.join(tmp, "f_%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
                   capture_output=True, text=True)
    sz = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"  -> {out}  {sz/1e6:.1f} MB")
    json.dump({"candidate": key, "rms_dx": float(err.mean()/dx), "peak_dx": float(err.max()/dx)},
              open(os.path.join(OUT, "mpm_movie.json"), "w"), indent=1)
    for p in glob.glob(os.path.join(tmp, "*.png")): os.remove(p)
    os.rmdir(tmp)


if __name__ == "__main__":
    main()
