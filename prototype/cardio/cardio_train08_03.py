#!/usr/bin/env python
"""train_08_03 -- phase map + FULL-DATASET cycle-batched training + RMSE loss.

Per 8.3: train over the WHOLE real dataset, not just the first/strongest beat. Partition
the real sequence into its individual cycles (real_all_beats), and each iteration draw a
BATCH of cycles; the loss is plain RMSE (per-node position) averaged over the batch -- no
open/length/shape composite. The model has one beat shape (tissue properties are fixed),
so this robustly fits the cycle-average using all the data.

The true-vs-learned mp4 RESPECTS THE QUIESCENT: both sequences are built explicitly as
[beat -> rest(quiescent) -> beat -> rest ...], so the tissue sits still between beats and
the silent periods align frame-for-frame.

Outputs archive/train_08_03/: true_vs_learned.mp4 (+ .png), phase.png, u_traces.png,
unet.png, run.log.

Run:  PYTHONPATH=src .../python prototype/cardio/cardio_train08_03.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import cardio_stage2 as C            # noqa: E402
import cardio_real_fit as R          # noqa: E402
import cardio_train08_phase as PH    # noqa: E402
from cardio_unet import UNet, maps_from_unet, load_image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = "/workspace/.conda_envs/neural-graph-linux/bin/ffmpeg"
_G = int(os.environ.get("CARDIO_GRID", "64"))                 # train_06 uses 137 (native real resolution)
GRID = (_G, _G)
N_ITER = int(os.environ.get("CARDIO_N_ITER", 400))            # overnight: e.g. CARDIO_N_ITER=200000 + MAX_SECONDS
BATCH = 3
K_ANCHOR = 0.06
# spring-anchor (substrate) stiffness: fixed by default (drifts toward 0 when free);
# set CARDIO_K_ANCHOR_LEARNABLE=1 to fit it as a global scalar (kept >= a small floor)
K_ANCHOR_LEARNABLE = os.environ.get("CARDIO_K_ANCHOR_LEARNABLE", "0") == "1"
ARCH_NAME = os.environ.get("CARDIO_ARCH_NAME", "train_08_03")  # train_08_05 sets this + k_anchor learnable
COARSE, MAX_LAG = 6, 18.0
N_CYCLES_RENDER = 6
Q_FRAMES = 14            # quiescent (rest) frames inserted between beats in the render
AMP = 10.0
FPS = 24
CKPT_EVERY = int(os.environ.get("CARDIO_CKPT_EVERY", 50))     # render + model-checkpoint cadence
DEVICE = os.environ.get("CARDIO_DEVICE", "cpu")               # "cpu" | "cuda:0" | "cuda:1"
MAX_SECONDS = float(os.environ.get("CARDIO_MAX_SECONDS", "0"))  # >0 -> wall-clock stop (overrides N_ITER cap)
USE_COMPILE = os.environ.get("CARDIO_COMPILE", "0") == "1"    # try torch.compile on the training forward


def rmse(a, b):
    return (a - b).pow(2).mean().clamp(min=1e-20).sqrt()


# --------------------------------------------------------------------------- #
#  True-vs-learned render with an explicit quiescent (rest) period between beats
# --------------------------------------------------------------------------- #
def _tile_quiescent(beats_seq, Q):
    """beats_seq: list of [L,M,2] displacement beats -> [sum(L+Q), M, 2] with rest gaps."""
    out = []
    for w in beats_seq:
        out.append(w)
        out.append(np.zeros((Q,) + w.shape[1:], np.float32))   # quiescent: at rest
    return np.concatenate(out, 0)


def render_true_vs_learned(learned_beat, real_beats, shape, out, grid_n=10, tail=70):
    """real(blue) vs learned(green) 10x10 trajectories over [beat->rest->beat...]; quiescent respected."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import LineCollection
    if os.path.exists(FFMPEG):
        plt.rcParams["animation.ffmpeg_path"] = FFMPEG
    Hy, Wx = shape
    ii = np.linspace(0, Hy - 1, grid_n).round().astype(int)
    jj = np.linspace(0, Wx - 1, grid_n).round().astype(int)
    sel = (ii[:, None] * Wx + jj[None, :]).ravel()
    gx, gy = np.meshgrid(np.linspace(0, 1, Wx)[jj], np.linspace(0, 1, Hy)[ii])
    rest = np.stack([gx, gy], -1).reshape(-1, 2)               # [M,2]
    nb = real_beats.shape[0]
    L_seq = _tile_quiescent([learned_beat[:, sel]] * N_CYCLES_RENDER, Q_FRAMES)
    R_seq = _tile_quiescent([real_beats[c % nb][:, sel] for c in range(N_CYCLES_RENDER)], Q_FRAMES)
    Lp = rest[None] + AMP * L_seq; Rp = rest[None] + AMP * R_seq
    fig, ax = plt.subplots(figsize=(7, 7)); ax.set_position([0, 0, 1, 1]); ax.axis("off")
    fig.set_facecolor("black"); ax.set_facecolor("black")
    ax.set_xlim(-.2, 1.2); ax.set_ylim(-.2, 1.2); ax.set_aspect("equal")
    # convention: green = TRUE (real), blue = LEARNED (fit)
    lb = LineCollection([], colors="lime", linewidths=1.1); ax.add_collection(lb)            # real (green)
    lg = LineCollection([], colors=(0, .6, 1, .8), linewidths=1.0); ax.add_collection(lg)    # learned (blue)
    hb = ax.scatter(Rp[0, :, 0], Rp[0, :, 1], s=7, c="lime")                                 # real
    hg = ax.scatter(Lp[0, :, 0], Lp[0, :, 1], s=7, c="deepskyblue")                          # learned

    def seg(A, t):
        w = A[max(0, t - tail):t + 1]
        return list(np.stack([w[:-1], w[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)) if t >= 1 else []
    w = FFMpegWriter(fps=FPS, bitrate=4000)
    with w.saving(fig, out, dpi=120):
        for t in range(Lp.shape[0]):
            lb.set_segments(seg(Rp, t)); lg.set_segments(seg(Lp, t))
            hb.set_offsets(Rp[t]); hg.set_offsets(Lp[t]); w.grab_frame()
    plt.close(fig)
    # static png (full accumulated)
    fig, ax = plt.subplots(figsize=(7, 7)); ax.set_position([0, 0, 1, 1]); ax.axis("off")
    fig.set_facecolor("black"); ax.set_facecolor("black")
    ax.set_xlim(-.2, 1.2); ax.set_ylim(-.2, 1.2); ax.set_aspect("equal")
    for Pn, col in [(Rp, "lime"), (Lp, (0, .6, 1, .8))]:
        ax.add_collection(LineCollection(list(np.stack([Pn[:-1], Pn[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)),
                                         colors=col, linewidths=0.8))
    fig.savefig(out.replace(".mp4", ".png"), dpi=130, facecolor="black"); plt.close(fig)
    print(f"saved {out} (+ .png)  [quiescent {Q_FRAMES} frames between beats]")


# --------------------------------------------------------------------------- #
def main():
    print(f"=== train_08_03 [{ARCH_NAME}]: phase + cycle-batched RMSE  (device={DEVICE}) ===")
    dev = torch.device(DEVICE)
    img = load_image(GRID).to(dev)
    geo, base = R.setup_fit(GRID)
    geo = tuple(g.to(dev) for g in geo); base_1d = base[:, 0].to(dev).clone()
    beats_cpu = R.real_all_beats(GRID); nb = beats_cpu.shape[0]
    beats_up = torch.stack([R.upsample_beat(beats_cpu[b], R.T_FIT) for b in range(nb)]).to(dev)  # boundary anchor
    beats = beats_cpu.to(dev)
    net = UNet().to(dev); pm = PH.PhaseMap(GRID, COARSE, MAX_LAG).to(dev); sc = R.Scalars().to(dev)
    sc.k_anchor.requires_grad_(K_ANCHOR_LEARNABLE)            # fixed unless CARDIO_K_ANCHOR_LEARNABLE=1
    with torch.no_grad():
        sc.k_anchor.fill_(K_ANCHOR)
    forward_beat = R.forward_beat
    if USE_COMPILE:
        try:
            forward_beat = torch.compile(R.forward_beat, dynamic=False)
            print("  torch.compile: ENABLED for forward_beat (first iter compiles, then caches)")
        except Exception as e:                               # noqa: BLE001
            print(f"  torch.compile failed ({e}); falling back to eager")
    print(f"  k_anchor {'LEARNABLE' if K_ANCHOR_LEARNABLE else 'fixed'} @ init {K_ANCHOR}; "
          f"N_ITER={N_ITER} ckpt_every={CKPT_EVERY} max_seconds={MAX_SECONDS:.0f} compile={USE_COMPILE}")
    params = [p for p in list(net.parameters()) + list(pm.parameters()) + list(sc.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=4e-3)
    x = img[None, None]
    gtr = torch.Generator().manual_seed(0)
    d = os.path.join(HERE, "archive", ARCH_NAME); ckdir = os.path.join(d, "checkpoints")
    os.makedirs(ckdir, exist_ok=True)
    rspec = {"model": {"pulse": {"period": 90, "dur": 6, "amp": 2.0},
                       "nagumo": {"D": 2.0, "a": 0.7, "b": 0.8, "eps": 0.3},
                       "signal": {"theta": 0.0, "eta": 0.3},
                       "mechanics": {"k_anchor": K_ANCHOR, "substeps": 4},
                       "boundary_width": R.BWIDTH}, "render": {"cycles": N_CYCLES_RENDER, "rec": 2}}
    real_rec = R.real_full(GRID, N_CYCLES_RENDER * 90)[::2]

    def render_ckpt(tag, sk, gn, fb, phi):                    # render on CPU (engine is CPU-based)
        with torch.no_grad():
            fp, _, _, _ = PH.full_forward_phase(sk.detach().cpu(), gn.detach().cpu(), fb.detach().cpu(),
                                                phi.detach().cpu(), sc, GRID, rspec)
        C.render_traj_png(fp, os.path.join(ckdir, f"true_vs_learned_{tag}.png"), GRID, amp=AMP, real=real_rec)
        return fp

    t_start = time.time(); it = 0
    while it < N_ITER and (MAX_SECONDS <= 0 or time.time() - t_start < MAX_SECONDS):
        opt.zero_grad()
        sk, gn, fb = maps_from_unet(net(x))
        phased = PH.phase_shift(base_1d, pm())
        idx = torch.randint(0, nb, (min(BATCH, nb),), generator=gtr)
        # each sampled cycle's boundary band is pinned to THAT cycle's real motion (Dirichlet)
        loss = torch.stack([rmse(R.resample(forward_beat(sk, gn, fb, sc, geo, phased, real_disp=beats_up[b])),
                                  beats[b]) for b in idx]).mean()
        loss.backward(); opt.step()
        with torch.no_grad():
            sc.beta.clamp_(0.05, 2.0); sc.gamma.clamp_(0.05, 1.0); sc.aniso.clamp_(0.0, 1.2)
            if K_ANCHOR_LEARNABLE:
                sc.k_anchor.clamp_(0.01, 0.5)                 # keep a floor so the tissue can't drift freely
        if it % 50 == 0:
            with torch.no_grad():
                full_rmse = torch.stack([rmse(R.resample(forward_beat(sk, gn, fb, sc, geo, phased,
                                              real_disp=beats_up[b])), beats[b]) for b in range(nb)]).mean().item()
            el = time.time() - t_start
            print(f"  it {it:5d}  [{el/60:.1f}min {el/max(it,1)*1000:.0f}ms/it]  batch RMSE {loss.item():.5f}  "
                  f"all-cycle {full_rmse:.5f}  beta={sc.beta.item():.3f} k_anch={sc.k_anchor.item():.3f} "
                  f"φ±{pm().abs().max().item():.1f}t", flush=True)
        if it % CKPT_EVERY == 0:
            render_ckpt(f"{it:05d}", sk, gn, fb, pm())
            torch.save({"it": it, "net": net.state_dict(), "pm": pm.state_dict(), "sc": sc.state_dict(),
                        "batch_rmse": loss.item()}, os.path.join(ckdir, f"model_{it:05d}.pt"))
        it += 1
    print(f"  stopped at it {it} after {(time.time()-t_start)/60:.1f} min")

    with torch.no_grad():
        sk, gn, fb = maps_from_unet(net(x)); phi = pm()
    rec_sc = {k: round(getattr(sc, k).item(), 3) for k in ("beta", "k_anchor", "gamma", "aniso")}

    # final full render on CPU
    sk_c, gn_c, fb_c, phi_c = sk.detach().cpu(), gn.detach().cpu(), fb.detach().cpu(), phi.detach().cpu()
    fit_pos, lvl, _, u_raw = PH.full_forward_phase(sk_c, gn_c, fb_c, phi_c, sc, GRID, rspec)
    np.save(os.path.join(d, "fit_pos.npy"), fit_pos)               # cache for cheap re-render
    torch.save({"it": N_ITER - 1, "net": net.state_dict(), "pm": pm.state_dict(), "sc": sc.state_dict(),
                "scalars": rec_sc}, os.path.join(d, "model_final.pt"))
    C.render_trajectories(fit_pos, os.path.join(d, "true_vs_learned.mp4"), GRID, amp=AMP, fps=FPS, real=real_rec)
    C.render_traj_png(fit_pos, os.path.join(d, "true_vs_learned.png"), GRID, amp=AMP, real=real_rec)
    PH.render_phase(phi_c.numpy(), GRID, os.path.join(d, "phase.png"))
    PH.render_utraces(u_raw, phi_c, GRID, os.path.join(d, "u_traces.png"))
    R._unet_png(img.detach().cpu(), sk_c, gn_c, fb_c, GRID, os.path.join(d, "unet.png"))
    with open(os.path.join(d, "run.log"), "w") as f:
        f.write(f"experiment={ARCH_NAME}: phase + FULL-dataset cycle-batched RMSE; k_anchor "
                f"{'LEARNABLE' if K_ANCHOR_LEARNABLE else 'fixed'}\n")
        f.write(f"grid={GRID} n_beats={nb} batch={BATCH} n_iter={N_ITER} loss=RMSE\n")
        f.write(f"scalars={rec_sc} phi_max_ticks={phi.abs().max().item():.2f}\n")
    print(f"  archived -> {d}")


if __name__ == "__main__":
    main()
