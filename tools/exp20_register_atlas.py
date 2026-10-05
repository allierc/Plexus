"""exp20: EVERY FISH IN THE Z-BRAIN ATLAS, automatically (Cedric, 2026-10-05: "register all 23 zebrafish; what matters is
what slide 132 quantifies; define a score to improve iteratively").

THE CHAIN: fish k -> fish 1 -> Z-Brain.
  fish k -> fish 1   SimpleITK on the two mean anatomies (volume0.hdf5 volume_mean, the volumes the cells were segmented
                     in; the same microscope and the same indicator, so intensity registration is well posed): both turned
                     head up (rows head to tail, a rotation, no mirror -- tools/exp20_bigwarp.py's frame), resampled to
                     RES_UM, an affine by Mattes mutual information, then (variant) a B-spline; fixed = fish k, so the
                     transform carries a fish-k point straight to fish 1
  fish 1 -> Z-Brain  Cedric's 30 BigWarp landmarks (data/atlas/bigwarp/fish_1_landmarks.csv), the thin-plate spline of
                     tools/exp20_bigwarp.py (leave-one-out 19 um)
Each cell then gets the Z-Brain regions its atlas voxel lies in (MaskDatabase.mat, 294 masks).

    PYTHONPATH=src:tools python tools/exp20_register_atlas.py orient                 # the 24 volumes head up: check first
    PYTHONPATH=src:tools python tools/exp20_register_atlas.py run [--fish glucose_f2 ...] [--variant aff_bs]
    PYTHONPATH=src:tools python tools/exp20_register_atlas.py report [--variant aff_bs]
    PYTHONPATH=src:tools python tools/exp20_register_atlas.py check [--variant aff_bs]   # one figure per fish, for the eye

THE SCORE (per fish; the gut response is never used, so the biology read off the atlas does not tune the registration):
  ncc     normalised cross-correlation of fish k's anatomy, carried into fish 1's frame, with fish 1's, over fish 1's
          brain (its anatomy above the 60th percentile), 1 = identical
  inside  the share of fish k's cells that land inside the Z-Brain brain (the union of the 294 masks)
  score = ncc x inside, in 0..1; the variant's score = the median over the 23 fish (fish 1: Cedric's landmarks).
Writes data/atlas/reg/<variant>/<fish>.npz (atlas position per cell, regions, the score), .../scores.json, and with
`report`: presentation/figs/atlas_<fish>.png (slide 132 for every fish), atlas_montage.png (the 24 fish from above),
atlas_regions_24.png (the gut-responsive share per region and fish) and data/atlas/reg/<variant>/regions_24.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
DATA = os.path.join(EXP, "data")
FIGS = os.path.join(EXP, "presentation", "figs")
ZB = os.path.join(DATA, "atlas", "zbrain", "Additional_mat_files")
VOX = (0.8125, 0.8125, 2.0)                 # x, y, z um (export_gutbrain_recording.VOXEL_UM, inferred)
ZVOX = (0.798, 0.798, 2.0)
RES_UM = 4.0
FISH = ([("glucose", k) for k in range(1, 7)] + [("glutamate", k) for k in range(1, 6)] + [("Lglucose", k) for k in range(1, 5)]
        + [("fish_water", k) for k in range(1, 5)] + [("blood_glucose", k) for k in range(1, 6)])
VARIANTS = {"aff": {"bspline": None}, "aff_bs": {"bspline": (6, 3, 3)}}
REGIONS = [("area postrema", ["Area Postrema"]), ("vagal ganglia (nodose)", ["Ganglia - Vagal Ganglia"]),
           ("X vagus motor (DMNX, DVC)", ["X Vagus motorneuron cluster"]),
           ("noradrenergic, interfascicular + vagal", ["Noradrendergic neurons of the Interfascicular and Vagal areas"]),
           ("rhombomere 1 (PBN within)", ["Rhombomere 1"]), ("rhombomere 7 (caudal medulla)", ["Rhombomere 7"]),
           ("tectum (SPV)", ["Tectum Stratum Periventriculare"]), ("hypothalamus (any)", ["Hypothalamus"]),
           ("cerebellum", ["Rhombencephalon - Cerebellum"])]


def tag(cond, k):
    return f"{cond}_f{k}"


def head_up(cond, k):
    """The fish's mean anatomy turned head up (z, rows head to tail, cols), as tools/exp20_bigwarp.py turns fish 1."""
    import h5py
    with h5py.File(os.path.join(DATA, f"{cond}_fish{k}_volume0.hdf5")) as h:
        V = h["volume_mean"][:].astype(np.float32)                    # (z, y, x), head at +x (every fish's provenance)
    return np.ascontiguousarray(V[:, :, ::-1].transpose(0, 2, 1)), V.shape[2]


def cells_in_frame(P, nx):
    """Raw pos_um (x, y, z) -> the head-up frame's physical (X, Y, Z) um, as tools/exp20_bigwarp.cells_in_moving."""
    return np.stack([P[:, 1], (nx - 1) * VOX[0] - P[:, 0], P[:, 2]], 1)


def _sitk(arr):
    import SimpleITK as sitk
    lo, hi = np.percentile(arr, [1, 99.8])
    img = sitk.GetImageFromArray(np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32))   # (z, y, x) -> x, y, z
    img.SetSpacing(VOX)
    sm = sitk.SmoothingRecursiveGaussian(img, RES_UM / 2)
    size = [int(round(sz * sp / RES_UM)) for sz, sp in zip(img.GetSize(), VOX)]
    return sitk.Resample(sm, size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), (RES_UM,) * 3, img.GetDirection(), 0.0)


def register(fixed, moving, variant):
    """A transform carrying FIXED-space points into MOVING space (SimpleITK's convention): fixed = fish k, moving = fish 1."""
    import SimpleITK as sitk
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2, seed=0)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 300, relaxationFactor=0.7)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SetInitialTransform(init, inPlace=False)
    aff = R.Execute(fixed, moving)
    T = sitk.CompositeTransform(3)
    T.AddTransform(aff)
    bs = VARIANTS[variant]["bspline"]
    if bs:
        B = sitk.BSplineTransformInitializer(fixed, list(bs))
        R2 = sitk.ImageRegistrationMethod()
        R2.SetMetricAsMattesMutualInformation(32)
        R2.SetMetricSamplingStrategy(R2.RANDOM)
        R2.SetMetricSamplingPercentage(0.2, seed=0)
        R2.SetInterpolator(sitk.sitkLinear)
        R2.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)
        R2.SetMovingInitialTransform(aff)
        R2.SetInitialTransform(B, inPlace=True)
        R2.SetShrinkFactorsPerLevel([2, 1])
        R2.SetSmoothingSigmasPerLevel([1, 0])
        R2.Execute(fixed, moving)
        T.AddTransform(B)
    return T, aff


def ncc_in_fish1(T, fixed, moving):
    """fish k resampled into fish 1's frame (the inverse is not needed: resample fish 1 into fish k's frame and compare
    there, over fish 1's brain carried along)."""
    import SimpleITK as sitk
    m = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, T, sitk.sitkLinear, 0.0))
    f = sitk.GetArrayFromImage(fixed)
    brain = m > np.percentile(m[m > 0], 60) if (m > 0).any() else np.zeros_like(m, bool)
    a, b = f[brain] - f[brain].mean(), m[brain] - m[brain].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def _atlas():
    import h5py
    from scipy.sparse import csc_matrix
    with h5py.File(os.path.join(ZB, "MaskDatabase.mat")) as h:
        H, W, Z = int(h["height"][0, 0]), int(h["width"][0, 0]), int(h["Zs"][0, 0])
        names = ["".join(chr(c) for c in h[r][:].ravel()) for r in h["MaskDatabaseNames"][:, 0]]
        g = h["MaskDatabase"]
        S = csc_matrix((g["data"][:], g["ir"][:], g["jc"][:]), shape=(H * W * Z, len(names))).tocsr()
    return S, names, (H, W, Z)


def label(A, S, shape):
    H, W, Z = shape
    iy = np.clip(np.round(A[:, 1] / ZVOX[1]).astype(int), 0, H - 1)
    ix = np.clip(np.round(A[:, 0] / ZVOX[0]).astype(int), 0, W - 1)
    iz = np.clip(np.round(A[:, 2] / ZVOX[2]).astype(int), 0, Z - 1)
    box = (A[:, 0] >= 0) & (A[:, 0] < W * ZVOX[0]) & (A[:, 1] >= 0) & (A[:, 1] < H * ZVOX[1]) & (A[:, 2] >= 0) & (A[:, 2] < Z * ZVOX[2])
    reg = S[iy + H * ix + H * W * iz].astype(bool).toarray()
    reg[~box] = False
    return reg


QC_LABELS = [("area postrema", "Area Postrema", "#ff4040"), ("vagal ganglia", "Ganglia - Vagal Ganglia", "#ffa726"),
             ("tectum", "Tectum Stratum Periventriculare", "#4dd0e1"), ("cerebellum", "Rhombencephalon - Cerebellum", "#ffffff")]


def _dense_labels(S, shape):
    """A few Z-Brain regions as one label volume (z, y, x) on the atlas grid: 1.. len(QC_LABELS), 0 elsewhere."""
    H, W, Z = shape
    names = _atlas.names
    L = np.zeros((Z, H, W), np.uint8)
    for i, (_, pat, _) in enumerate(QC_LABELS, 1):
        j = next(j for j, n in enumerate(names) if pat in n)
        lin = S[:, j].nonzero()[0] if hasattr(S, "nonzero") else None
        lin = S.tocsc()[:, j].nonzero()[0]
        L[lin // (H * W), lin % H, (lin // H) % W] = i
    return L


def check_volumes(T, fixed, tps, F, L):
    """THE CHECK (Cedric, 2026-10-05: "slides to evaluate the registrations myself"): on fish k's own 4-um grid, its
    anatomy, the Z-Brain reference carried onto it through the whole chain (fish k -> fish 1 -> Z-Brain, sampled
    there), and a few Z-Brain regions carried the same way. No inversion needed: every fish-k voxel is sent forward."""
    import SimpleITK as sitk
    import exp20_bigwarp as BW
    from scipy.ndimage import map_coordinates
    size, org, sp = fixed.GetSize(), np.array(fixed.GetOrigin()), np.array(fixed.GetSpacing())
    zz, yy, xx = np.meshgrid(*[np.arange(n) for n in size[::-1]], indexing="ij")
    p = np.stack([xx, yy, zz], -1).reshape(-1, 3) * sp + org                       # physical x, y, z per voxel
    if T is None:
        q = p
    else:
        D = sitk.TransformToDisplacementField(T, sitk.sitkVectorFloat64, size, fixed.GetOrigin(), fixed.GetSpacing(),
                                              fixed.GetDirection())
        q = p + sitk.GetArrayFromImage(D).reshape(-1, 3)
    a = BW.tps_apply(tps, q)                                                      # Z-Brain um (x, y, z)
    c = np.stack([a[:, 2] / ZVOX[2], a[:, 1] / ZVOX[1], a[:, 0] / ZVOX[0]])       # (z, y, x) atlas voxels
    shp = tuple(size[::-1])
    zref = map_coordinates(F, c, order=1, mode="constant", cval=0).reshape(shp).astype(np.float32)
    lab = map_coordinates(L, c, order=0, mode="constant", cval=0).reshape(shp).astype(np.uint8)
    return sitk.GetArrayFromImage(fixed).astype(np.float32), zref, lab


def run(fish, variant):
    import SimpleITK as sitk
    import exp20_bigwarp as BW
    from plexus.tasks import trace_recording as TR
    out = os.path.join(DATA, "atlas", "reg", variant)
    os.makedirs(out, exist_ok=True)
    mv, fx = BW.read_landmarks(os.path.join(BW.BW, "fish_1_landmarks.csv"))
    tps = BW.tps_fit(mv, fx)
    S, names, shape = _atlas()
    _atlas.names = names
    import tifffile
    F = tifffile.imread(os.path.join(DATA, "atlas", "bigwarp", "zbrain_reference.tif")).astype(np.float32)
    L = _dense_labels(S, shape)
    M1, nx1 = head_up("glucose", 1)
    moving = _sitk(M1)
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(os.environ.get("SITK_THREADS", "16")))   # several fish at once
    for cond, k in fish:
        t0 = time.time()
        P = np.asarray(TR.load(f"gutbrain_{cond}_f{k}")["pos_um"], np.float64)
        if (cond, k) == ("glucose", 1):
            Q, ncc = cells_in_frame(P, nx1), 1.0
            fx_img, T = moving, None
        else:
            Mk, nxk = head_up(cond, k)
            fixed = _sitk(Mk)
            T, aff = register(fixed, moving, variant)
            ncc = ncc_in_fish1(T, fixed, moving)
            Q = np.array([T.TransformPoint(tuple(p)) for p in cells_in_frame(P, nxk)])       # fish k -> fish 1 frame
            fx_img = fixed
            np.save(os.path.join(out, f"{tag(cond, k)}_affine.npy"),                       # the affine, kept for reading
                    np.r_[aff.GetNthTransform(0).GetParameters() if hasattr(aff, "GetNthTransform") else aff.GetParameters()])
        A = BW.tps_apply(tps, Q)                                                           # fish 1 frame -> Z-Brain um
        reg = label(A, S, shape)
        inside = float(reg.any(1).mean())
        sc = {"ncc": ncc, "inside": inside, "score": ncc * inside, "cells": int(len(P)), "seconds": round(time.time() - t0)}
        json.dump(sc, open(os.path.join(out, f"{tag(cond, k)}.json"), "w"), indent=1)    # one file per fish: runs in parallel
        fv, zr, lb = check_volumes(T, fx_img, tps, F, L)
        np.savez_compressed(os.path.join(out, f"{tag(cond, k)}_check.npz"), fish=fv, zref=zr, labels=lb)
        np.savez_compressed(os.path.join(out, f"{tag(cond, k)}.npz"), atlas_um=A.astype(np.float32), regions=reg,
                            names=np.array(names))
        print(f"[register] {variant} {tag(cond, k)}: ncc {ncc:.3f}, inside {inside:.3f}, score {ncc * inside:.3f} "
              f"({sc['seconds']} s)", flush=True)
    scores = collect(out)
    rest = [v["score"] for f_, v in scores.items() if f_ != "glucose_f1"]
    print(f"[register] {variant}: the score, median over the {len(rest)} fish done: {np.median(rest):.3f}")


def collect(out):
    """Every fish's score (data/atlas/reg/<variant>/<fish>.json), gathered into scores.json."""
    sc = {tag(c, k): json.load(open(os.path.join(out, f"{tag(c, k)}.json"))) for c, k in FISH
          if os.path.exists(os.path.join(out, f"{tag(c, k)}.json"))}
    json.dump(sc, open(os.path.join(out, "scores.json"), "w"), indent=1)
    return sc


def _responsive(cond, k):
    p = os.path.join(DATA, f"baselines_gutbrain_{cond}_f{k}_cells.npz")
    return np.load(p)["responsive"] if os.path.exists(p) else None


def report(variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile
    out = os.path.join(DATA, "atlas", "reg", variant)
    scores = collect(out)
    F = tifffile.imread(os.path.join(DATA, "atlas", "bigwarp", "zbrain_reference.tif"))
    top = F.max(0)
    ext = [0, F.shape[2] * ZVOX[0], F.shape[1] * ZVOX[1], 0]
    table = {}
    done = [(c, k) for c, k in FISH if os.path.exists(os.path.join(out, f"{tag(c, k)}.npz"))]
    fig_m, ax_m = plt.subplots(4, 6, figsize=(18, 17), facecolor="black")
    for a in ax_m.ravel():
        a.set_facecolor("black"); a.axis("off")
    for i, (cond, k) in enumerate(done):
        z = np.load(os.path.join(out, f"{tag(cond, k)}.npz"), allow_pickle=True)
        A, reg, names = z["atlas_um"], z["regions"], [str(n) for n in z["names"]]
        resp = _responsive(cond, k)
        entry = reg[:, [j for j, n in enumerate(names) if "Area Postrema" in n or "Ganglia - Vagal Ganglia" in n]].any(1)
        row = {}
        for lab, pats in REGIONS:
            m = reg[:, [j for j, n in enumerate(names) if any(p in n for p in pats)]].any(1)
            row[lab] = {"cells": int(m.sum()), "gut_responsive": int((m & resp).sum()) if resp is not None else None}
        table[tag(cond, k)] = {"score": scores.get(tag(cond, k)), "regions": row,
                               "gut_responsive": int(resp.sum()) if resp is not None else None}
        a = ax_m.ravel()[i]
        a.imshow(top, cmap="gray", extent=ext, vmax=2.2 * np.percentile(top, 99.5))
        a.scatter(A[::10, 0], A[::10, 1], s=0.2, c="#29b6f6", alpha=0.5, lw=0)
        if resp is not None:
            a.scatter(A[resp, 0], A[resp, 1], s=0.6, c="#ffeb3b", lw=0)
        a.scatter(A[entry, 0], A[entry, 1], s=1.5, c="#ff4040", lw=0)
        sc = scores.get(tag(cond, k), {})
        ap = row["area postrema"]
        a.set_title(f"{cond.replace('_', ' ')} {k}: score {sc.get('score', float('nan')):.2f}\n"
                    f"AP {ap['gut_responsive']} of {ap['cells']} gut-responsive", color="white", fontsize=9)
        a.set_aspect("equal")
        a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3])
    fig_m.text(0.5, 0.005, "Z-Brain from above; blue the fish's cells (1 in 10), yellow its gut-responsive cells, red its cells "
               "in the area postrema and the vagal ganglia; score = ncc x inside (registration only)", color="0.85",
               fontsize=11, ha="center")
    fig_m.tight_layout(rect=(0, 0.02, 1, 1))
    fig_m.savefig(os.path.join(FIGS, "atlas_montage.png"), dpi=110, facecolor="black")
    plt.close(fig_m)
    # the gut-responsive share per region and fish (slide 132's bars, every fish at once)
    keys = [l for l, _ in REGIONS]
    fishes = list(table)
    M = np.array([[100.0 * table[f]["regions"][r]["gut_responsive"] / max(table[f]["regions"][r]["cells"], 1)
                   if table[f]["regions"][r]["gut_responsive"] is not None else np.nan for f in fishes] for r in keys])
    fig, ax = plt.subplots(figsize=(18, 5.6), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=60, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            c = table[fishes[j]]["regions"][keys[i]]["cells"]
            ax.text(j, i, f"{M[i, j]:.0f}\n({c})" if np.isfinite(M[i, j]) else "", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] < 35 else "black")
    ax.set_yticks(range(len(keys)), keys, color="0.9", fontsize=9)
    ax.set_xticks(range(len(fishes)), [f.replace("blood_glucose", "blood").replace("fish_water", "water").replace("_f", " ")
                                      for f in fishes], color="0.9", fontsize=8, rotation=45, ha="right")
    cb = fig.colorbar(im, ax=ax, fraction=0.02)
    cb.ax.tick_params(colors="0.8")
    cb.set_label("% of the region's cells that are gut-responsive", color="0.9")
    ax.set_title("every fish in Z-Brain: the gut-responsive share per region (in brackets the region's cells)",
                 color="white", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "atlas_regions_24.png"), dpi=140, facecolor="black")
    plt.close(fig)
    json.dump(table, open(os.path.join(out, "regions_24.json"), "w"), indent=1)
    print(f"[report] {variant}: {len(done)} fish -> atlas_montage.png, atlas_regions_24.png, regions_24.json")


def check(variant):
    """One figure per fish for the eye: green its anatomy, magenta the Z-Brain reference carried onto it (white where they
    agree), four Z-Brain regions outlined -- from above in three depth slabs (dorsal, middle, ventral thirds) and from
    the side. Writes presentation/figs/atlas_check_<fish>.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(DATA, "atlas", "reg", variant)
    sc = collect(out)
    def norm(v, lo_pct=0.0):          # the fish's anatomy sits on a bright background: its 40th percentile made black
        pos = v[v > 0]
        if not pos.size:
            return np.zeros_like(v)
        lo, hi = np.percentile(pos, lo_pct), np.percentile(pos, 99.5)
        return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    for cond, k in FISH:
        f = os.path.join(out, f"{tag(cond, k)}_check.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        fv, zr, lb = z["fish"], z["zref"], z["labels"]
        nz = fv.shape[0]
        thirds = [(0, nz // 3, "dorsal third"), (nz // 3, 2 * nz // 3, "middle third"), (2 * nz // 3, nz, "ventral third")]
        fig, ax = plt.subplots(1, 4, figsize=(16, 7.2), facecolor="black", gridspec_kw={"width_ratios": [1, 1, 1, 0.75]})
        for a, (z0, z1, lab_) in zip(ax[:3], thirds):
            g, m = norm(fv[z0:z1].max(0), 40), norm(zr[z0:z1].max(0))
            a.imshow(np.stack([m, g, m], -1))
            for i, (nm, _, col) in enumerate(QC_LABELS, 1):
                msk = (lb[z0:z1] == i).any(0)
                if msk.any():
                    a.contour(msk, levels=[0.5], colors=[col], linewidths=0.9)
            a.set_title(f"from above, {lab_}", color="white", fontsize=10)
            a.axis("off")
        g, m = norm(fv.max(2).T, 40), norm(zr.max(2).T)                                  # side: rows head to tail, cols depth
        ax[3].imshow(np.stack([m, g, m], -1), aspect=1.0)
        for i, (nm, _, col) in enumerate(QC_LABELS, 1):
            msk = (lb == i).any(2).T
            if msk.any():
                ax[3].contour(msk, levels=[0.5], colors=[col], linewidths=0.9)
        ax[3].set_title("from the side", color="white", fontsize=10)
        ax[3].axis("off")
        s_ = sc.get(tag(cond, k), {})
        fig.suptitle(f"{cond.replace('_', ' ')} fish {k}: green the fish's anatomy, magenta Z-Brain carried onto it (white "
                     f"where they agree) -- ncc {s_.get('ncc', float('nan')):.2f}, inside {s_.get('inside', float('nan')):.2f}, "
                     f"score {s_.get('score', float('nan')):.2f}", color="white", fontsize=11)
        fig.text(0.5, 0.01, "outlines: " + ", ".join(f"{nm}" for nm, _, _ in QC_LABELS) + " (red, orange, cyan, white)",
                 color="0.8", fontsize=9, ha="center")
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(os.path.join(FIGS, f"atlas_check_{tag(cond, k)}.png"), dpi=110, facecolor="black")
        plt.close(fig)
        print(f"[check] {tag(cond, k)}", flush=True)


def orient():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 12, figsize=(24, 12), facecolor="black", gridspec_kw={"width_ratios": [2, 1] * 6})
    for a in ax.ravel():
        a.set_facecolor("black"); a.axis("off")
    for i, (cond, k) in enumerate(FISH):
        M, _ = head_up(cond, k)
        r, c = i // 6, 2 * (i % 6)
        ax[r, c].imshow(M.max(0), cmap="gray", vmax=np.percentile(M.max(0), 99.7), aspect=1.0)
        ax[r, c + 1].imshow(M.max(2).T, cmap="gray", vmax=np.percentile(M.max(2), 99.7), aspect=VOX[1] / VOX[2])
        ax[r, c].set_title(f"{cond} {k}", color="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "png", "atlas_orient_24.png"), dpi=60, facecolor="black")
    print("[orient] png/atlas_orient_24.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["orient", "run", "report", "check"])
    ap.add_argument("--fish", nargs="*", default=None)
    ap.add_argument("--variant", default="aff_bs", choices=list(VARIANTS))
    a = ap.parse_args()
    if a.what == "orient":
        orient()
    elif a.what == "run":
        fish = [f for f in FISH if a.fish is None or tag(*f) in a.fish]
        run(fish, a.variant)
    elif a.what == "check":
        check(a.variant)
    else:
        report(a.variant)
