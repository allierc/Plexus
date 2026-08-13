#!/usr/bin/env python
"""gate_03e -- is the contact rate-limited? The test that turns S13--S15 from a failure into a bound.

    python gate_03e_press_rate.py [--device cuda:0]  ->  log/okuda_ECM/03e_press_rate/

THE FAILURE THIS EXPLAINS. `gates_03` swept the three numbers that are not physics and found the
answer moving with all of them: over a sixteenfold change in the penalty the penetration fell only
1.92 -> 1.27 cells (a log-log slope of -0.15 where a force balance predicts -1) while the block's
compression ROSE 13.88 -> 15.66%, and halving the time step moved it again, 14.95 -> 18.50%.

THE DIAGNOSIS, WHICH IS ONE SENTENCE AND IS TESTABLE. A penalty permits a depth d = f/k only if it
has time to reach that depth. Here the surface is PRESCRIBED: it advances v*dt per step whatever the
matrix does, so if the penalty cannot accelerate a particle out of the way within a step, the depth
is set by the advance and not by the force -- and every one of those three sweeps is then a sweep of
how much advance happens per step. That regime has a name (rate-limited) and a signature: the answer
must CONVERGE as the press slows at fixed total travel, and the k-dependence must vanish with it.

SO THE SWEEP IS THE PRESS SPEED, at MATCHED TOTAL TRAVEL. Four runs at v = 2, 1, 1/2, 1/4 of the
nominal, with the frame count scaled inversely so every run presses exactly as far and lifts exactly
as far. If compression converges as v falls, the diagnosis holds and the last converged speed is the
domain of validity for every number 03 and 04 report. If it does not, the penalty is the wrong
method here and BFEMP (Li et al. 2022) is the answer rather than a slower press.

THE SECOND HALF OF THE GATE. At the slowest speed the k-sweep is repeated: if the regime is
rate-limited, the compression's dependence on k must be much weaker there than at the nominal speed.
That is what makes this a gate on the MECHANISM rather than a convergence study.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.animation import FFMpegWriter                        # noqa: E402
from matplotlib.colors import ListedColormap                         # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import ecm_spec as ES                                                # noqa: E402
import test_03_mesh_contact as T3                                    # noqa: E402

LOG = os.path.join(_ROOT, "log", "okuda_ECM")
CMAP = ListedColormap(ES.STRESS_COLORS)
MESH_C = "#e8dcc0"
BASE_FRAMES = 1600           # 03b's own operating point
BASE_V = 1.0                 # `MeshOnMatrix.v_press`


def _panel(ax, letter):
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")


def run(dev, v, keep=None, **kw):
    """One rig at press speed `v`, with the frame count scaled so the TRAVEL is the same.

    Matching the travel and not the frame count is the whole design: a slower press over the same
    number of frames would simply press less far, and the compression would fall for a reason that
    has nothing to do with the contact.
    """
    frames = int(round(BASE_FRAMES * BASE_V / v))
    rig = T3.MeshOnMatrix(dev=dev, floor=0.18, **kw)
    rig.v_press = v
    drawn = []
    every = max(1, frames // 120)
    t0 = time.time()
    for t in range(frames):
        rig.press_v = -rig.v_press if t < 0.6 * frames else +0.7 * rig.v_press
        rig.step()
        if keep is not None and t % every == 0:
            drawn.append((t, rig.x.detach().cpu().numpy().copy(),
                          rig.V.detach().cpu().numpy().copy(),
                          np.abs(rig.J.detach().cpu().numpy() - 1.0)))
    r = rig.res
    con = [i for i, n in enumerate(r["n_pen"]) if n > 0]
    out = dict(v_press=float(v), frames=frames, dt=float(rig.dt), k_pen=float(rig.k_pen),
               wall_s=float(time.time() - t0),
               momentum_max=float(max(r["momentum"])),
               depth_max_box=float(max(r["depth"])), depth_max_cells=float(max(r["depth"]) / rig.dx),
               contacts_max=int(max(r["n_pen"])),
               f_norm_sum=float(np.sum(r["f_norm"])),
               slip_mean=float(np.mean([r["slip"][i] for i in con]) if con else 0.0),
               height_start=float(r["height"][0]), height_min=float(min(r["height"])),
               compression=float(1 - min(r["height"]) / max(r["height"][0], 1e-9)),
               # THE ADVANCE PER STEP, which is the quantity the diagnosis is about: how far the
               # prescribed surface moves while the penalty has one step to respond.
               advance_per_step=float(v * rig.dt),
               series={k: [float(x) for x in v_[::8]] for k, v_ in r.items()})
    if keep is not None:
        keep.extend(drawn)
    return out


def movie(drawn, d, name, dx):
    """The slowest press, which is the run the gate is about. Section only: the compression is a
    height, and a height is what a section shows."""
    allS = np.concatenate([k[3][::7] for k in drawn])
    hi = float(np.percentile(allS[np.isfinite(allS)], 99)) or 1.0
    fig = plt.figure(figsize=(5.6, 5.6), facecolor="black")
    wri = FFMpegWriter(fps=20, metadata={"title": name})
    with wri.saving(fig, os.path.join(d, "movie.mp4"), dpi=100):
        for (t, X, V, S) in drawn:
            fig.clf()
            a = fig.add_subplot(1, 1, 1, facecolor="black")
            sl = np.abs(X[:, 1] - 0.5) < 0.02
            a.scatter(X[sl][:, 0], X[sl][:, 2], s=7, c=np.clip(S[sl] / hi, 0, 1), cmap=CMAP,
                      vmin=0, vmax=1, marker=".", linewidths=0)
            nx = int(np.sqrt(V.shape[0]))
            a.plot(V[:, 0].reshape(nx, nx)[:, nx // 2], V[:, 2].reshape(nx, nx)[:, nx // 2], "-",
                   color=MESH_C, lw=1.6)
            a.plot([0.25, 0.75], [0.18, 0.18], "-", color="#9aa0a6", lw=2.5)
            a.set_xlim(0.25, 0.75); a.set_ylim(0.13, 0.66); a.set_aspect("equal"); a.axis("off")
            a.text(0.02, 0.98, f"{name}   step {t}", transform=a.transAxes, color="white",
                   fontsize=10, va="top")
            wri.grab_frame()
    fig.savefig(os.path.join(d, "3d.png"), dpi=110, facecolor="black")
    plt.close(fig)


def main():
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
    d = os.path.join(LOG, "03e_press_rate")
    os.makedirs(d, exist_ok=True)
    out, drawn = {}, []
    speeds = (2.0, 1.0, 0.5, 0.25)
    for v in speeds:
        keep = drawn if v == min(speeds) else None
        out[f"v{v:g}"] = run(dev, v, keep=keep)
        o = out[f"v{v:g}"]
        print(f"[03e] v={v:<5g} {o['frames']:5d} frames | advance/step {o['advance_per_step']:.2e} "
              f"| compression {100 * o['compression']:.2f}% | penetration "
              f"{o['depth_max_cells']:.3f} cells | residual {o['momentum_max']:.2e} "
              f"| {o['wall_s']:.0f} s", flush=True)
    # --- the second half: does the k-dependence weaken at the slow speed?
    slow = min(speeds)
    for kf in (0.075, 0.30):
        out[f"v{slow:g}_k{kf:g}"] = run(dev, slow, k_frac=kf)
        print(f"[03e] v={slow:g} k_frac={kf}: compression "
              f"{100 * out[f'v{slow:g}_k{kf:g}']['compression']:.2f}%", flush=True)

    c = np.array([out[f"v{v:g}"]["compression"] for v in speeds])
    vv = np.array(speeds)
    # convergence: the change between the two slowest, relative to the slowest
    conv = float(abs(c[-1] - c[-2]) / max(c[-1], 1e-12))
    k_slow = [out[f"v{slow:g}_k0.075"]["compression"], out[f"v{slow:g}"]["compression"],
              out[f"v{slow:g}_k0.3"]["compression"]]
    spread_slow = float((max(k_slow) - min(k_slow)) / max(k_slow[1], 1e-12))
    out["gates"] = dict(
        G_press_converges=dict(
            threshold="change between the two slowest presses < 0.05 of the slowest",
            measured=conv,
            why="if the compression converges as the press slows, the failure of S13-S15 is the "
                "rate-limited regime and the converged speed is the domain of validity"),
        G_k_dependence_weakens=dict(
            threshold="spread over k at the slow press < 0.10 (it is 0.119 at the nominal)",
            measured=spread_slow,
            why="a rate-limited contact must lose its k-dependence when the rate falls; if it does "
                "not, the penalty is the wrong method and BFEMP is the answer"),
        G_momentum=dict(threshold="< 1e-6 in every run",
                        measured=float(max(x["momentum_max"] for x in out.values()
                                           if isinstance(x, dict) and "momentum_max" in x)),
                        why="the reaction is an identity and must not care about any of this"))
    json.dump(out, open(os.path.join(d, "metrics.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.4), facecolor="white")
    ax[0].semilogx(vv, 100 * c, "o-", color="#e0452b")
    ax[0].set_xlabel("press speed, relative to nominal")
    ax[0].set_ylabel("block compression (%)")
    _panel(ax[0], "a")
    ax[1].semilogx(vv, [out[f"v{v:g}"]["depth_max_cells"] for v in speeds], "o-", color="#2b6cb0")
    ax[1].set_xlabel("press speed, relative to nominal")
    ax[1].set_ylabel("max penetration (grid cells)")
    _panel(ax[1], "b")
    kk = [0.075, 0.15, 0.30]
    ax[2].semilogx(kk, [100 * x for x in k_slow], "o-", color="#1f8a5c", label=f"v = {slow:g}")
    ax[2].semilogx(kk, [13.88, 14.95, 15.66], "s--", color="#999", label="v = 1 (gates_03)")
    ax[2].set_xlabel(r"$k_{\rm frac}$")
    ax[2].set_ylabel("block compression (%)")
    ax[2].legend(fontsize=7, frameon=False)
    _panel(ax[2], "c")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(d, "gate.png"), dpi=150, facecolor="white")
    plt.close(fig)

    if drawn:
        movie(drawn, d, "03e_press_rate", 1.0 / 64)
    for k, v in out["gates"].items():
        print(f"[gate] {k}: {v['measured']:.4g}  (threshold: {v['threshold']})", flush=True)
    print(f"[03e] -> {d}", flush=True)


if __name__ == "__main__":
    main()
