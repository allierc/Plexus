"""make_target.py -- extract a coarse cell-DENSITY target from the real dicty movie,
for the simulation-vs-video loss (the user chose coarse density-map matching).

The movie is a 3D-rendered lightsheet: a rotating yellow bounding box, a small inset
in the top-right corner, a scale bar / timestamp, and the cells themselves as bright
ORANGE blobs over a darker GREEN chemical haze. To make the target robust to the box
rotation + field-of-view drift we:

  1. crop away the inset + margins (keep the main panel only);
  2. detect cell pixels by colour (orange: R high and R > B);
  3. for K timepoints across the AGGREGATION WINDOW (skip the empty/drifted tail),
     build a coarse occupancy histogram AFTER centring on the cells' centre-of-mass
     and scaling by their RMS radius -> a translation+scale-invariant density map.

Writes target_density.npz {t_frac[K], dens[K,G,G], grid G} + a montage PNG.

    PYTHONPATH=... python make_target.py
"""
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = "/groups/saalfeld/home/allierc/GraphData/graphs_data/dicty/260228InterfaceTransferMirrorB1-DB-floor-fl2.mp4"

GRID = 48                       # coarse density resolution (matches sim loss)
K = 6                           # number of matched timepoints
WIN = (0.02, 0.80)             # aggregation window as a fraction of the movie (skip drifted tail)
# main-panel crop as fractions of (W,H): drop the top-right inset, scale bar, borders
CROP = dict(x0=0.06, x1=0.74, y0=0.03, y1=0.97)


def cell_mask(bgr):
    """Orange amoebae: red strong and clearly above blue. Returns a bool [h,w]."""
    b, g, r = bgr[..., 0].astype(np.int16), bgr[..., 1].astype(np.int16), bgr[..., 2].astype(np.int16)
    return (r > 110) & (r - b > 35)


def detect_centroids(panel, min_area=4, max_area=4000):
    """Count individual amoebae in a panel and return their centroids normalized to the
    panel's [0,1]^2 (origin bottom-left, matching the sim's y-up convention). Used to
    initialize the simulation's frame 0 with the REAL cell number + positions."""
    h, w = panel.shape[:2]
    m = cell_mask(panel).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    pts = []
    for i in range(1, n):                                 # skip background label 0
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area:
            cx, cy = cents[i]
            pts.append([cx / w, 1.0 - cy / h])            # flip y -> bottom-left origin
    return np.array(pts, np.float32) if pts else np.zeros((0, 2), np.float32)


def _hist(p2d, grid=GRID):
    """Occupancy histogram of points already centred + scaled into the +-1 box, sum 1."""
    h, _, _ = np.histogram2d(p2d[:, 0], p2d[:, 1], bins=grid, range=[[-1, 1], [-1, 1]])
    s = h.sum()
    return (h / s).astype(np.float32) if s > 0 else np.full((grid, grid), 1.0 / (grid * grid), np.float32)


def seq_densities(pts_list, grid=GRID):
    """K density maps for a sequence of point sets, using PER-FRAME COM (translation-robust to
    FOV drift) but a SHARED scale = frame-0 RMS spread. The shared scale is the fix that keeps
    the aggregation signal: as cells concentrate, the maps genuinely peak up (per-frame RMS
    scaling instead rescales every frame to the same spread and erases concentration)."""
    p0 = pts_list[0].astype(np.float32)
    scale0 = np.sqrt(((p0 - p0.mean(0)) ** 2).sum(1).mean()) + 1e-6
    out = []
    for pts in pts_list:
        pts = pts.astype(np.float32)
        out.append(_hist((pts - pts.mean(0)) / (2.5 * scale0), grid))
    return np.stack(out)


def normalized_density(ys, xs, grid=GRID):
    """Single-frame density (per-frame COM + RMS-scale) -- kept for display in compare_*.py."""
    if len(xs) < 10:
        return np.full((grid, grid), 1.0 / (grid * grid), np.float32)
    p = np.stack([xs, ys], 1).astype(np.float32)
    p = p - p.mean(0)
    rms = np.sqrt((p ** 2).sum(1).mean()) + 1e-6
    return _hist(p / (2.5 * rms), grid)


def main():
    cap = cv2.VideoCapture(VIDEO)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, x1 = int(CROP["x0"] * W), int(CROP["x1"] * W)
    y0, y1 = int(CROP["y0"] * H), int(CROP["y1"] * H)
    fr_idx = (np.linspace(WIN[0], WIN[1], K) * (n - 1)).astype(int)
    pts_list, counts, init_pos = [], [], None
    for j, i in enumerate(fr_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
        if not ok:
            continue
        panel = fr[y0:y1, x0:x1]
        m = cell_mask(panel)
        ys, xs = np.nonzero(m)
        ys = m.shape[0] - 1 - ys                          # image rows are y-down -> flip to y-up (match sim + init_pos)
        pts_list.append(np.stack([xs, ys], 1).astype(np.float32))
        counts.append(int(m.sum()))
        if j == 0:                                        # frame 0: real cell count + positions
            init_pos = detect_centroids(panel)
    cap.release()
    dens = seq_densities(pts_list); t_frac = np.linspace(WIN[0], WIN[1], K)
    np.savez(os.path.join(HERE, "target_density.npz"), t_frac=t_frac, dens=dens, grid=GRID,
             crop=np.array([x0, x1, y0, y1]), counts=np.array(counts), init_pos=init_pos)
    print(f"frames {n}, sampled {list(fr_idx)}, cell-pixel counts {counts}", flush=True)
    print(f"frame-0 detected cells N0={len(init_pos)} (seed positions for the sim)", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, K, figsize=(3 * K, 3)); fig.patch.set_facecolor("black")
    for ax, d, tf in zip(axs, dens, t_frac):
        ax.imshow(d.T, origin="lower", cmap="inferno"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={tf:.2f}", color="white", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "target_density.png"), dpi=70, facecolor="black")
    print("wrote target_density.npz + target_density.png", flush=True)


if __name__ == "__main__":
    main()
