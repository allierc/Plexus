"""make_velocity_target.py -- PIV (optical-flow) velocity target from the real movie.

Density says WHERE cells end up; this says HOW they stream there. We compute dense
optical flow (Farneback ~ PIV) between consecutive movie frames at each of K
timepoints, restricted to the cell regions, then summarize it two ways:

  * speed[K,g,g]      -- coarse per-bin flow SPEED field (for visualization);
  * inflow[K,nbin]    -- the RADIAL inward-speed profile around the cell centre-of-mass
                         (mean of -v . rhat vs RMS-normalized radius). This is
                         translation+rotation invariant -> robust to the movie's
                         rotating box / FOV drift, and is exactly the streaming-toward-
                         the-aggregate signal. Normalized to unit L2 (shape only;
                         absolute speed depends on an arbitrary frame<->dt mapping).

Writes velocity_target.npz {t_frac, inflow[K,nbin], speed[K,g,g]} + a quiver PNG.

    PYTHONPATH=... python make_velocity_target.py
"""
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = "/groups/saalfeld/home/allierc/GraphData/graphs_data/dicty/260228InterfaceTransferMirrorB1-DB-floor-fl2.mp4"
from make_target import cell_mask, CROP, WIN, K   # reuse the panel crop + window + K

GBIN = 20            # coarse grid for the speed field (viz)
NBIN = 10            # radial profile bins
STRIDE = 2           # frames between the optical-flow pair


def radial_inflow(flow, mask, nbin=NBIN):
    """Mean inward radial speed (-v . rhat) vs RMS-normalized radius, over cell pixels."""
    iy, ix = np.nonzero(mask)                                # image rows (y-down) for indexing flow
    if len(ix) < 20:
        return np.zeros(nbin, np.float32)
    v = flow[iy, ix]                                          # (M,2) vx,vy  (flow_y already y-up)
    xs = ix.astype(np.float32); ys = (mask.shape[0] - 1 - iy).astype(np.float32)   # y-up positions
    com = np.array([xs.mean(), ys.mean()])
    rel = np.stack([xs - com[0], ys - com[1]], 1).astype(np.float32)
    rad = np.linalg.norm(rel, axis=1) + 1e-6
    rhat = rel / rad[:, None]
    inward = -(v * rhat).sum(1)                               # >0 = moving toward COM
    rms = np.sqrt((rad ** 2).mean())
    rn = np.clip(rad / (2.5 * rms), 0, 1)                     # normalized radius in [0,1]
    prof = np.zeros(nbin, np.float32)
    idx = np.clip((rn * nbin).astype(int), 0, nbin - 1)
    for b in range(nbin):
        sel = idx == b
        if sel.any():
            prof[b] = inward[sel].mean()
    n = np.linalg.norm(prof) + 1e-6
    return (prof / n).astype(np.float32)                      # shape only


def coarse_speed(flow, mask, g=GBIN):
    h, w = mask.shape
    sp = np.linalg.norm(flow, axis=2) * mask
    sp = cv2.resize(sp, (g, g), interpolation=cv2.INTER_AREA)
    m = sp.max()
    return (sp / m).astype(np.float32) if m > 0 else sp.astype(np.float32)


def main():
    cap = cv2.VideoCapture(VIDEO)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, x1 = int(CROP["x0"] * W), int(CROP["x1"] * W)
    y0, y1 = int(CROP["y0"] * H), int(CROP["y1"] * H)
    t_frac = np.linspace(WIN[0], WIN[1], K)
    fr_idx = (t_frac * (n - 1)).astype(int)

    def gray(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
        p = fr[y0:y1, x0:x1]
        return cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), p

    inflow, speed, quivers, frame_speeds = [], [], [], []
    for i in fr_idx:
        g0, p0 = gray(i); g1, _ = gray(min(i + STRIDE, n - 1))
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 25, 3, 5, 1.2, 0)
        flow[..., 1] = -flow[..., 1]                           # image y-down -> world y-up
        m = cell_mask(p0)
        inflow.append(radial_inflow(flow, m)); speed.append(coarse_speed(flow, m))
        iy, ix = np.nonzero(m)
        frame_speeds.append(np.linalg.norm(flow[iy, ix], axis=1) if len(ix) else np.zeros(1))
        quivers.append((p0[::-1], flow))
    cap.release()
    inflow = np.stack(inflow); speed = np.stack(speed)
    # per-frame [mean, std] of the speed distribution, normalized by the sequence-mean speed
    # (scale-free) -- matched against the sim's sim_speed_feats in the optimizer loss.
    seqmean = np.mean(np.concatenate(frame_speeds)) + 1e-9
    vfeat = np.array([[s.mean() / seqmean, s.std() / seqmean] for s in frame_speeds], np.float32)
    np.savez(os.path.join(HERE, "velocity_target.npz"), t_frac=t_frac, inflow=inflow,
             speed=speed, vfeat=vfeat, nbin=NBIN, gbin=GBIN)
    print(f"PIV target: vfeat {vfeat.shape}, per-frame [mean,std] speed (seq-normalized):\n{np.round(vfeat,3)}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, K, figsize=(3 * K, 3)); fig.patch.set_facecolor("black")
    for ax, (img, flow), tf in zip(axs, quivers, t_frac):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), origin="lower")
        h, w = flow.shape[:2]; s = 40
        Y, X = np.mgrid[0:h:s, 0:w:s]
        fy = -flow[::s, ::s, 1]            # back to image coords for display over imshow
        fx = flow[::s, ::s, 0]
        ax.quiver(X, h - Y, fx, fy, color="cyan", scale=300, width=0.003)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"PIV t={tf:.2f}", color="white", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "velocity_target.png"), dpi=70, facecolor="black")
    print("wrote velocity_target.npz + velocity_target.png", flush=True)


if __name__ == "__main__":
    main()
