"""exp17: THE DESTRIPED NEURONS IN THE Z-BRAIN ATLAS (Cedric, 2026-10-07: "I have made the BigWarp registration,
experiments/exp17_zapbench_graphcast/landmarks.csv; for convenience, head left, I rotated the reference tif to the left").
Z-Brain: Randlett et al. 2015, Nature Methods 12, 1039 ("Whole-brain activity mapping onto a zebrafish brain atlas").

THE CHAIN. The destriped neurons' positions (zapbench_destripe_recording pos_um) are in the anatomy volume's frame, um,
x y z -- BigWarp's MOVING frame (anatomy_t0_v2.ome.zarr, 0.406 x 0.406 x 1 um). The landmarks' FIXED side is the Z-Brain
reference (0.798 x 0.798 x 2 um, 621 x 1406 x 138 voxels, head up) after Fiji's "Rotate 90 Degrees Left": a rotated
pixel (x', y') is the original (W - 1 - y', x'), W = 621. The thin-plate spline of tools/exp20_bigwarp.py (BigWarp's own
kernel, r^2 log r) carries a neuron to the rotated reference; the rotation is undone; each neuron then gets the Z-Brain
regions its voxel lies in (MaskDatabase.mat, 294 overlapping masks).
CHECKS: the leave-one-out landmark error, and the share of the neurons that land inside the Z-Brain brain (the union of
the masks), for the left rotation and, as a control, the right one.

    PYTHONPATH=src:tools python tools/exp17_atlas.py                       # the mapping
    PYTHONPATH=src:tools python tools/exp17_atlas.py summary [--iso|--merged]   # the per-region figure
-> data/atlas_destripe.npz (atlas um, inside, regions [N, 294] bool, names), data/atlas_destripe.json,
   presentation/figs/atlas_regions{,_iso,_merged}.png, data/atlas_regions.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
ZB = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast", "data", "atlas", "zbrain", "Additional_mat_files")
ZVOX = (0.798, 0.798, 2.0)


def unrotate(A, W, rot):
    """Rotated-reference um -> original Z-Brain um. rot 'left': x = (W-1) vox - y', y = x'; 'right': x = y', y = (H-1) - x'."""
    out = A.copy()
    if rot == "left":
        out[:, 0] = (W - 1) * ZVOX[0] - A[:, 1]
        out[:, 1] = A[:, 0]
    else:
        out[:, 0] = A[:, 1]
        out[:, 1] = (1406 - 1) * ZVOX[1] - A[:, 0]
    return out


def main():
    import h5py
    from scipy.sparse import csc_matrix
    from exp20_bigwarp import read_landmarks, tps_apply, tps_fit
    from plexus.paths import graphs_data_path
    mv, fx = read_landmarks(os.path.join(EXP, "landmarks.csv"))
    ok = np.isfinite(mv).all(1) & np.isfinite(fx).all(1)
    mv, fx = mv[ok], fx[ok]
    model = tps_fit(mv, fx)
    loo = [float(np.linalg.norm(tps_apply(tps_fit(np.delete(mv, k, 0), np.delete(fx, k, 0)), mv[k:k + 1])[0] - fx[k]))
           for k in range(len(mv))]
    P = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))["pos_um"].astype(np.float64)
    Ar = tps_apply(model, P)
    with h5py.File(os.path.join(ZB, "MaskDatabase.mat")) as h:
        H, W, Z = int(h["height"][0, 0]), int(h["width"][0, 0]), int(h["Zs"][0, 0])
        names = ["".join(chr(c) for c in h[r][:].ravel()) for r in h["MaskDatabaseNames"][:, 0]]
        g = h["MaskDatabase"]
        S = csc_matrix((g["data"][:], g["ir"][:], g["jc"][:]), shape=(H * W * Z, len(names))).tocsr()
    res = {}
    for rot in ("left", "right"):
        A = unrotate(Ar, W, rot)
        ix = np.round(A[:, 0] / ZVOX[0]).astype(int)
        iy = np.round(A[:, 1] / ZVOX[1]).astype(int)
        iz = np.round(A[:, 2] / ZVOX[2]).astype(int)
        box = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (iz >= 0) & (iz < Z)
        lin = np.clip(iy, 0, H - 1) + H * np.clip(ix, 0, W - 1) + H * W * np.clip(iz, 0, Z - 1)   # column-major (y, x, z)
        reg = S[lin].astype(bool).toarray() & box[:, None]
        inside = reg.any(1)
        res[rot] = (A, inside, reg)
        print(f"[atlas] rotation {rot}: {100 * box.mean():.1f} % in the atlas box, {100 * inside.mean():.1f} % inside the brain")
    rot = max(res, key=lambda r: res[r][1].mean())
    A, inside, reg = res[rot]
    np.savez_compressed(os.path.join(EXP, "data", "atlas_destripe.npz"), atlas_um=A.astype(np.float32), inside=inside,
                        regions=reg, names=np.array(names))
    cnt = reg.sum(0)
    doc = {"landmarks": int(len(mv)), "loo_um_median": float(np.median(loo)), "loo_um_max": float(np.max(loo)),
           "rotation_used": rot, "inside_share": {r: float(res[r][1].mean()) for r in res}, "neurons": int(len(P)),
           "regions_nonempty": int((cnt > 0).sum()),
           "top_regions": [(names[i], int(cnt[i])) for i in np.argsort(cnt)[::-1][:25]]}
    json.dump(doc, open(os.path.join(EXP, "data", "atlas_destripe.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "top_regions"}, indent=1))
    for n, c in doc["top_regions"][:15]:
        print(f"   {c:7,d}  {n}")


# the regions of the table, anatomical (no transgene masks), head to tail (Cedric, 2026-10-07: "work with the atlas now")
REGIONS = [("Telencephalon - Olfactory Bulb", "olfactory bulb"), ("Telencephalon - Pallium", "pallium"),
           ("Telencephalon - Subpallium", "subpallium"), ("Diencephalon - Habenula", "habenula"),
           ("Diencephalon - Dorsal Thalamus", "dorsal thalamus"), ("Diencephalon - Ventral Thalamus", "ventral thalamus"),
           ("Diencephalon - Pretectum", "pretectum"), ("Diencephalon - Preoptic Area", "preoptic area"),
           ("Diencephalon - Intermediate Hypothalamus", "hypothalamus (intermediate)"),
           ("Diencephalon - Posterior Tuberculum", "posterior tuberculum"),
           ("Mesencephalon - Tectum Stratum Periventriculare", "tectum (periventricular)"),
           ("Mesencephalon - Torus Semicircularis", "torus semicircularis"), ("Mesencephalon - Tegmentum", "tegmentum"),
           ("Rhombencephalon - Cerebellum", "cerebellum"), ("Rhombencephalon - Rhombomere 1", "rhombomere 1"),
           ("Rhombencephalon - Rhombomere 2", "rhombomere 2"), ("Rhombencephalon - Rhombomere 3", "rhombomere 3"),
           ("Rhombencephalon - Rhombomere 4", "rhombomere 4"), ("Rhombencephalon - Rhombomere 5", "rhombomere 5"),
           ("Rhombencephalon - Rhombomere 6", "rhombomere 6"), ("Rhombencephalon - Rhombomere 7", "rhombomere 7"),
           ("Rhombencephalon - Noradrendergic neurons of the Interfascicular and Vagal areas", "noradrenergic (IFN, vagal)"),
           ("Rhombencephalon - X Vagus motorneuron cluster", "vagus motor neurons"), ("Spinal Cord", "spinal cord")]


def iso_images(rlist, rfull, rcol, names, sizes=((1640, 1000), (1800, 620))):
    """THE REGIONS AS SURFACES (Cedric, 2026-10-07: "a twin with VTK isosurfaces, as the Plexus watcher"): each table
    region's Z-Brain mask, halved in x and y, contoured at 0.5 and smoothed, in its table colour, semi-opaque; the brain
    (the union of the division masks) a faint grey shell. Shown head left as the scatter (x' = y, y' = (W-1) dx - x), from
    above and from the side, parallel projection on black, each render the shape of its panel. Returns the two images."""
    import h5py
    import pyvista as pv
    from scipy.sparse import csc_matrix
    pv.OFF_SCREEN = True
    with h5py.File(os.path.join(ZB, "MaskDatabase.mat")) as h:
        H, W, Z = int(h["height"][0, 0]), int(h["width"][0, 0]), int(h["Zs"][0, 0])
        g = h["MaskDatabase"]
        S = csc_matrix((g["data"][:], g["ir"][:], g["jc"][:]), shape=(H * W * Z, len(names)))

    def volume(col):
        v = np.zeros(H * W * Z, np.float32)
        a, b = S.indptr[col], S.indptr[col + 1]
        v[S.indices[a:b]] = 1.0
        return v.reshape((Z, W, H)).transpose(2, 1, 0)[::2, ::2, :]       # column-major (y, x, z) -> [y, x, z], halved

    def surface(vol):
        img = pv.ImageData(dimensions=vol.shape, spacing=(2 * ZVOX[1], 2 * ZVOX[0], ZVOX[2]))
        img.point_data["m"] = vol.ravel(order="F")
        srf = img.contour([0.5], scalars="m")
        if srf.n_points == 0:
            return None
        srf = srf.smooth(n_iter=30)
        P = np.asarray(srf.points)                                      # (y, x, z) um -> head left (x', y', z)
        srf.points = np.column_stack([P[:, 0], (W - 1) * ZVOX[0] - P[:, 1], P[:, 2]])
        return srf
    brain = None
    for key in ("Telencephalon -", "Diencephalon -", "Mesencephalon -", "Rhombencephalon -", "Spinal Cord"):
        if key in names:
            v_ = volume(names.index(key))
            brain = v_ if brain is None else np.maximum(brain, v_)
    out = []
    for view, size in zip(("top", "side"), sizes):
        pl = pv.Plotter(off_screen=True, window_size=size)
        pl.set_background("black")
        sb = surface(brain)
        if sb is not None:
            pl.add_mesh(sb, color="#8c8c8c", opacity=0.10, smooth_shading=True)
        for r_ in rlist:
            sr = surface(volume(names.index(rfull[r_])))
            if sr is not None:
                pl.add_mesh(sr, color=rcol[r_][:3], opacity=0.75, smooth_shading=True, specular=0.3)
        pl.enable_parallel_projection()
        b = sb.bounds if sb is not None else (0, 1, 0, 1, 0, 1)
        c = ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
        if view == "top":
            pl.camera.position = (c[0], c[1], c[2] + 3000.0)
            pl.camera.up = (0.0, 1.0, 0.0)
            pl.camera.parallel_scale = max((b[3] - b[2]) / 2, (b[1] - b[0]) / 2 * size[1] / size[0]) * 1.06
        else:
            pl.camera.position = (c[0], c[1] - 3000.0, c[2])
            pl.camera.up = (0.0, 0.0, 1.0)
            pl.camera.parallel_scale = max((b[5] - b[4]) / 2, (b[1] - b[0]) / 2 * size[1] / size[0]) * 1.06
        pl.camera.focal_point = c
        out.append(pl.screenshot(return_img=True))
        pl.close()
    return out


def summary(iso=False):
    """THE ANALYSES PER REGION: per Z-Brain region, its neurons, their lead over the brain mean, the share of input neurons
    (the balanced 20 % mask), of swim- or turn-tuned neurons, of neurons whose response repeats and of neurons that ramp
    up in open loop (tools/exp17_classic.py). iso False: the neurons as dots; True: the regions as surfaces; "merged":
    one column of four panels, all at ONE scale. -> figs/atlas_regions{,_iso,_merged}.png, data/atlas_regions.json"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    A, reg, names = za["atlas_um"], za["regions"], [str(x) for x in za["names"]]
    zl = np.load(os.path.join(EXP, "data", "brain_mean_lag_zapbench_destripe.npz"))
    lead, wl = -zl["lag_s"], zl["with_lag"]
    zc = np.load(os.path.join(EXP, "data", "classic_per_neuron.npz"))
    gn = [str(x) for x in zc["groups"]]
    motor = zc["tuned"] & np.isin(zc["best"], [gn.index(g) for g in ("swim", "turn left", "turn right")])
    relb = zc["reliable_block"] >= 0
    rampup = zc["ramp"] > 0.5
    inp = np.load(graphs_data_path("zebrafish", "input_mask_destripe_bal20.npz"))["mask"] > 0
    cols = [("neurons", None), ("median lead, s", None), ("lead > 1 s, %", None), ("input neurons, %", None),
            ("swim / turn tuned, %", None), ("response repeats, %", None), ("ramps up in open loop, %", None)]
    rows, out = [], {}
    for full, short in REGIONS:
        if full not in names:
            continue
        m = reg[:, names.index(full)]
        if m.sum() < 50:
            continue
        ml = m & wl
        v = [int(m.sum()), float(np.median(lead[ml])) if ml.sum() > 20 else np.nan,
             100 * float((lead[ml] > 1.0).mean()) if ml.sum() > 20 else np.nan, 100 * float(inp[m].mean()),
             100 * float(motor[m].mean()), 100 * float(relb[m].mean()), 100 * float(rampup[m].mean())]
        rows.append((short, v))
        out[short] = dict(zip([c for c, _ in cols], v))
    json.dump(out, open(os.path.join(EXP, "data", "atlas_regions.json"), "w"), indent=1)
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8.6), facecolor="black")
    # the neurons in the atlas, head left (the reference turned as Cedric turned it), each coloured by its TABLE region
    # (Cedric, 2026-10-07: "only 5 regions in the fish, many more in the table"): of the table's regions it lies in, the
    # most specific (fewest neurons); grey the neurons in none
    W = 621
    xd, yd = A[:, 1], (W - 1) * ZVOX[0] - A[:, 0]
    rlist = [short for short, _ in rows]
    rfull = {short: full for full, short in REGIONS}
    pal = list(plt.get_cmap("tab20")(np.arange(20))) + list(plt.get_cmap("Set1")(np.arange(9)))
    rcol = {r_: pal[k % len(pal)] for k, r_ in enumerate(rlist)}
    lab_ = np.full(len(A), -1)
    best_n = np.full(len(A), np.inf)
    for k, r_ in enumerate(rlist):
        m = reg[:, names.index(rfull[r_])]
        n_ = m.sum()
        upd = m & (n_ < best_n)
        lab_[upd], best_n[upd] = k, n_
    ins_ = lab_ >= 0
    if iso == "merged":
        # ONE SCALE for the four panels (Cedric, 2026-10-07: "top and side views are not to scale"): um per inch the
        # same everywhere, each panel's height its own extent at that scale, stacked from the top
        ext = lambda v: float(np.percentile(v[ins_], 99.9) - np.percentile(v[ins_], 0.1))     # noqa: E731
        ex, ey, ez = ext(xd), ext(yd), ext(A[:, 2]) + 40.0
        FW, FH, TT = 14.0, 8.6, 0.28                       # figure inches; a title line
        sc = min(0.58 * FW / ex, (0.96 * FH - 4 * TT) / (2 * ey + 2 * ez))    # inches per um
        wf = ex * sc / FW
        panels, ycur = [], 0.985
        for kind, view, ttl in (("dots", "top", "from above, head left: the neurons"),
                                ("iso", "top", "from above: the regions as surfaces"),
                                ("dots", "side", "from the side: the neurons"),
                                ("iso", "side", "from the side: the regions as surfaces")):
            hf = (ey if view == "top" else ez) * sc / FH
            ycur -= TT / FH + hf
            panels.append((kind, [0.01, ycur, wf, hf], view, ttl))
    elif iso:
        panels = [("iso", [0.0, 0.40, 0.58, 0.54], "top", "the regions as surfaces, from above, head left"),
                  ("iso", [0.0, 0.05, 0.58, 0.30], "side", "from the side")]
    else:
        panels = [("dots", [-0.03, 0.37, 0.62, 0.62], "top", "in Z-Brain, from above, head left"),
                  ("dots", [-0.03, 0.02, 0.62, 0.34], "side", "from the side")]

    def trim(im, pad=6):                                  # a render cut to the brain
        on = np.argwhere(im[:, :, :3].max(2) > 12)
        (y0, x0), (y1, x1) = on.min(0), on.max(0)
        return im[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
    imgs = ({k: trim(v) for k, v in zip(("top", "side"), iso_images(rlist, rfull, rcol, names))}
            if any(k == "iso" for k, *_ in panels) else {})
    for kind, rect, view, ttl in panels:
        ax = fig.add_axes(rect)
        ax.axis("off")
        ax.set_title(ttl + ("" if iso == "merged" else " -- the colours of the table's names"), fontsize=9.5, loc="left",
                     x=0.06, pad=2)
        if kind == "iso":
            ax.imshow(imgs[view], aspect="auto" if iso == "merged" else "equal")
            continue
        Y = yd if view == "top" else A[:, 2]
        ax.scatter(xd[~ins_], Y[~ins_], s=0.15, color="0.25", lw=0, rasterized=True)
        if view == "side":                                # a 100-um bar under the side view
            yb_ = np.percentile(Y, 0.5) - 25.0
            xb_ = np.percentile(xd, 99.5) - 100.0
            ax.plot([xb_, xb_ + 100.0], [yb_, yb_], color="white", lw=2.0)
            ax.text(xb_ + 50.0, yb_ - 8.0, "100 µm", color="white", fontsize=9, ha="center", va="top")
        for k, r_ in enumerate(rlist):
            m = lab_ == k
            ax.scatter(xd[m], Y[m], s=0.35 if iso != "merged" else 0.25, color=rcol[r_], lw=0, rasterized=True)
        if iso == "merged":                               # the brain's own extent, the panel's
            ax.set_xlim(np.percentile(xd[ins_], 0.1), np.percentile(xd[ins_], 99.9))
            ax.set_ylim(np.percentile(Y[ins_], 0.1) - (40.0 if view == "side" else 0.0), np.percentile(Y[ins_], 99.9))
            ax.set_aspect("auto")
        else:
            ax.set_aspect("equal")
    # right: the table, each column shaded on its own scale
    ax = fig.add_axes([0.60, 0.05, 0.395, 0.88])           # the table right, its names in its own left part
    ax.axis("off")
    M = np.array([v for _, v in rows], float)
    nr, nc = M.shape
    for j in range(nc):
        col = M[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        for i in range(nr):
            x = col[i]
            if np.isfinite(x) and j > 0:
                t = (x - lo) / max(hi - lo, 1e-9) if j != 1 else 0.5 + 0.5 * x / max(abs(lo), abs(hi), 1e-9)
                fc = plt.get_cmap("coolwarm" if j == 1 else "magma")(0.15 + 0.7 * t)
                ax.add_patch(plt.Rectangle((j, nr - 1 - i), 1, 1, color=fc, alpha=0.75))
            s_ = "--" if not np.isfinite(x) else (f"{int(x):,}" if j == 0 else (f"{x:+.2f}" if j == 1 else f"{x:.0f}"))
            ax.text(j + 0.5, nr - 0.5 - i, s_, ha="center", va="center", fontsize=8, color="white")
    for i, (short, _) in enumerate(rows):
        ax.text(-0.1, nr - 0.5 - i, short, ha="right", va="center", fontsize=8.5, color=rcol[short], weight="bold")
    for j, (c, _) in enumerate(cols):
        ax.text(j + 0.5, nr + 0.2, c, ha="left", va="bottom", fontsize=8.5, rotation=35)
    ax.set_xlim(-4.6, nc + 1.2)                            # room for the names (left) and the last column title
    ax.set_ylim(0, nr + 2.6)
    fig.savefig(os.path.join(EXP, "presentation", "figs", {"merged": "atlas_regions_merged.png", True: "atlas_regions_iso.png",
                                                           False: "atlas_regions.png"}[iso]), dpi=130, facecolor="black")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary(iso="merged" if "--merged" in sys.argv else ("--iso" in sys.argv))
    else:
        main()
