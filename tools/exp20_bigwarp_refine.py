"""exp20: ASSESS AND REFINE FISH 1's BIGWARP LANDMARKS (Cedric, 2026-10-04: "assess the current transform and modify the
landmarks to improve the results ... pairing properly in x-y and in z").

The landmarks (data/atlas/bigwarp/fish_1_landmarks.csv, BigWarp's export: name, active, moving x y z, fixed x y z, um)
pair points of fish 1's mean anatomy (fish1_moving.tif, 0.8125 x 0.8125 x 2 um) with the Z-Brain reference
(zbrain_reference.tif, 0.798 x 0.798 x 2 um). Cedric's 30 lie on 4 fish planes, and every landmark of a plane was given
ONE atlas depth (moving z 93 / 137 / 187 / 229 um -> fixed z 114.5 / 142.7 / 194.5 / 225.1): the fit assumes the
fish's planes are parallel to the atlas's (no pitch, no roll).

assess   the thin-plate spline (kernel r^2 log r + affine, BigWarp's) from the landmarks; fish 1 warped into the atlas
         grid (x, y every 2 voxels, z every slice); scores, inside the atlas brain:
           nmi        normalised mutual information (H(a) + H(b)) / H(a, b) of the warped fish and the atlas (2 = same
                      image, 1 = independent): the two stains differ (fish 1: its own mean fluorescence; the atlas: the
                      Z-Brain reference), so their intensities are compared through their joint histogram, not a
                      correlation
           per-slice  the same per atlas depth
           loo        each landmark predicted by the spline of the other 29, its error in um (atlas frame)
         and overlays (atlas magenta, warped fish green; white where both are bright): planes from above at 7 depths,
         a sagittal section at the midline and a transverse one.
refine   each landmark's ATLAS point searched again in 3-D: fish 1's neighbourhood of the moving point (64 x 64 x 20 um),
         carried by the spline's local linear part, against the atlas around the current atlas point, shifts of
         +-20 um in x, y and +-16 um in z; the shift of highest NMI kept when it beats no shift by a margin, else the
         landmark stays. Writes fish_1_landmarks_refined.csv (BigWarp's format; the moving points unchanged) and
         re-scores the refined spline the same way.

    PYTHONPATH=src:tools python tools/exp20_bigwarp_refine.py assess|refine
Outputs in data/atlas/bigwarp/refine/ (Cedric's fish_1_landmarks.csv is only read).
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BW = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast", "data", "atlas", "bigwarp")
OUT = os.path.join(BW, "refine")
MV, FV = np.array([0.8125, 0.8125, 2.0]), np.array([0.798, 0.798, 2.0])      # um per voxel, x y z
DS = 2                                                                         # atlas x, y step of the warped grid


def load_landmarks(path):
    rows = [r for r in csv.reader(open(path))]
    names = [r[0] for r in rows]
    A = np.array([[float(v) for v in r[2:8]] for r in rows])
    return names, A[:, :3], A[:, 3:]


def save_landmarks(path, names, m, f):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        for n, a, b in zip(names, m, f):
            w.writerow([n, "true"] + [repr(float(v)) for v in a] + [repr(float(v)) for v in b])


def tps(src, dst):
    """The thin-plate spline src -> dst (kernel r^2 log r with an affine part, exact at the landmarks)."""
    from scipy.interpolate import RBFInterpolator
    return RBFInterpolator(src, dst, kernel="thin_plate_spline", degree=1)


def volumes():
    import tifffile
    M = tifffile.imread(os.path.join(BW, "fish1_moving.tif")).astype(np.float32)       # z, y, x
    F = tifffile.imread(os.path.join(BW, "zbrain_reference.tif")).astype(np.float32)
    return M, F


def warp(M, Fshape, f2m):
    """Fish 1 sampled on the atlas grid (z every slice, y and x every DS voxels): [nz, ny, nx] and its coverage."""
    from scipy.ndimage import map_coordinates
    nz, ny, nx = Fshape
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(0, ny, DS), np.arange(0, nx, DS), indexing="ij")
    P = np.stack([xx.ravel() * FV[0], yy.ravel() * FV[1], zz.ravel() * FV[2]], 1)
    out = np.empty(len(P), np.float32)
    cov = np.empty(len(P), bool)
    for a in range(0, len(P), 2_000_000):
        q = f2m(P[a:a + 2_000_000]) / MV                                               # moving voxel x, y, z
        idx = np.stack([q[:, 2], q[:, 1], q[:, 0]])
        out[a:a + 2_000_000] = map_coordinates(M, idx, order=1, mode="constant", cval=np.nan)
        cov[a:a + 2_000_000] = np.all((q >= 0) & (q <= np.array(M.shape[::-1]) - 1), axis=1)
    return out.reshape(zz.shape), cov.reshape(zz.shape)


def brain_mask(Fd):
    """The atlas brain: the smoothed reference above a fraction of its bright level."""
    from scipy.ndimage import gaussian_filter, binary_fill_holes
    s = gaussian_filter(Fd, (1.0, 3.0, 3.0))
    m = s > 0.06 * np.percentile(s, 99.5)
    return np.stack([binary_fill_holes(k) for k in m])


def nmi(a, b, bins=24):
    """(H(a) + H(b)) / H(a, b) of two samples, each binned by its own percentiles."""
    qa = np.clip(np.searchsorted(np.percentile(a, np.linspace(0, 100, bins + 1)[1:-1]), a), 0, bins - 1)
    qb = np.clip(np.searchsorted(np.percentile(b, np.linspace(0, 100, bins + 1)[1:-1]), b), 0, bins - 1)
    j = np.bincount(qa * bins + qb, minlength=bins * bins).astype(np.float64)
    j /= j.sum()
    pa, pb = j.reshape(bins, bins).sum(1), j.reshape(bins, bins).sum(0)
    h = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))                          # noqa: E731
    return (h(pa) + h(pb)) / h(j)


def scores(W, cov, Fd, mask):
    ok = mask & cov & np.isfinite(W)
    per = []
    for k in range(W.shape[0]):
        o = ok[k]
        per.append(float(nmi(W[k][o], Fd[k][o])) if o.sum() > 2000 else float("nan"))
    return {"nmi": float(nmi(W[ok], Fd[ok])), "voxels": int(ok.sum()), "per_slice": per}


def loo(m, f):
    e = []
    for i in range(len(m)):
        k = np.arange(len(m)) != i
        e.append(float(np.linalg.norm(tps(m[k], f[k])(m[i:i + 1])[0] - f[i])))
    return np.array(e)


def overlays(W, Fd, tag, mask):
    """Atlas magenta, warped fish green, each scaled to its own 1-99.5 % range inside the brain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def norm(v, ref):
        lo, hi = np.nanpercentile(ref, [1, 99.5])
        return np.clip((np.nan_to_num(v, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    wn, fn = norm(W, W[mask & np.isfinite(W)]), norm(Fd, Fd[mask])

    def rgb(a, b):
        return np.stack([b, a, b], -1)                                               # magenta atlas, green fish
    nz = W.shape[0]
    zs = [int(round(z / FV[2])) for z in (90, 110, 130, 150, 170, 190, 210, 230) if z / FV[2] < nz]
    fig, axs = plt.subplots(1, len(zs), figsize=(2.4 * len(zs), 5.6), facecolor="black")
    for ax, k in zip(axs, zs):
        ax.imshow(rgb(wn[k], fn[k]), aspect=FV[1] * DS / (FV[0] * DS))
        ax.set_title(f"atlas z {k * FV[2]:.0f} um", color="white", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{tag}: atlas (magenta) and fish 1 warped (green), from above", color="white")
    fig.savefig(os.path.join(OUT, f"{tag}_axial.png"), dpi=110, facecolor="black")
    plt.close(fig)
    xm = W.shape[2] // 2                                                            # the midline column
    ys = [int(W.shape[1] * f) for f in (0.30, 0.45, 0.60)]
    fig, axs = plt.subplots(1, 1 + len(ys), figsize=(16, 4.4), facecolor="black",
                            gridspec_kw={"width_ratios": [W.shape[1] * DS * FV[1]] + [W.shape[2] * DS * FV[0]] * len(ys)})
    axs[0].imshow(rgb(wn[:, :, xm], fn[:, :, xm]), aspect=FV[2] / (FV[1] * DS))
    axs[0].set_title("sagittal, midline (head left, dorsal up)", color="white", fontsize=9)
    for ax, y in zip(axs[1:], ys):
        ax.imshow(rgb(wn[:, y, :], fn[:, y, :]), aspect=FV[2] / (FV[0] * DS))
        ax.set_title(f"transverse, atlas y {y * DS * FV[1]:.0f} um", color="white", fontsize=9)
    for ax in axs:
        ax.axis("off")
    fig.savefig(os.path.join(OUT, f"{tag}_sections.png"), dpi=110, facecolor="black")
    plt.close(fig)


def assess(names, m, f, tag, M, F):
    Fd = F[:, ::DS, ::DS]
    mask = brain_mask(Fd)
    W, cov = warp(M, F.shape, tps(f, m))
    sc = scores(W, cov, Fd, mask)
    e = loo(m, f)
    sc.update({"loo_um_median": float(np.median(e)), "loo_um_max": float(e.max()),
               "loo_um": dict(zip(names, np.round(e, 1).tolist()))})
    overlays(W, Fd, tag, mask)
    json.dump(sc, open(os.path.join(OUT, f"{tag}_scores.json"), "w"), indent=1)
    print(f"[{tag}] NMI {sc['nmi']:.4f} over {sc['voxels']:,} voxels; leave-one-out median {sc['loo_um_median']:.1f} um, "
          f"max {sc['loo_um_max']:.1f} um ({max(sc['loo_um'], key=sc['loo_um'].get)})")
    return sc, W, cov, Fd, mask


def quantize(v, ref, bins):
    """Integer bins of v by the percentiles of ref (a sample inside the brain)."""
    edges = np.percentile(ref, np.linspace(0, 100, bins + 1)[1:-1])
    return np.clip(np.searchsorted(edges, v), 0, bins - 1).astype(np.int32)


def nmi_q(qa, qb, bins):
    j = np.bincount(qa * bins + qb, minlength=bins * bins).astype(np.float64)
    j /= j.sum()
    pa, pb = j.reshape(bins, bins).sum(1), j.reshape(bins, bins).sum(0)
    h = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))                          # noqa: E731
    return (h(pa) + h(pb)) / h(j)


def match(M, Fd, mask, m, f, m2f, half_xy=32.0, half_z=10.0, sxy=13, sz=8, bins=16):
    """THE BLOCK MATCH OF ONE LANDMARK: fish 1 around m carried into the atlas frame by the spline's local linear part J
    (numerical Jacobian at m), against the atlas around f + delta for every delta of the grid -- x, y in steps of the
    warped grid's 1.6 um (+-sxy steps), z in 2-um slices (+-sz). Returns the NMI landscape [dz, dy, dx] and the deltas."""
    from scipy.ndimage import map_coordinates
    h = 4.0
    J = np.stack([(m2f(m[None] + h * e) - m2f(m[None] - h * e))[0] / (2 * h) for e in np.eye(3)], 1)   # d fixed / d moving
    Ji = np.linalg.inv(J)
    gs = np.array([FV[0] * DS, FV[1] * DS, FV[2]])                              # the warped grid's step, um
    c = np.round(f / gs).astype(int)                                            # the atlas voxel nearest f (x, y, z)
    nx_, nz_ = int(round(half_xy / gs[0])), int(round(half_z / gs[2]))
    ox, oy, oz = (np.arange(-nx_, nx_ + 1), np.arange(-nx_, nx_ + 1), np.arange(-nz_, nz_ + 1))
    Z, Y, X = np.meshgrid(oz, oy, ox, indexing="ij")
    u = np.stack([X.ravel() * gs[0], Y.ravel() * gs[1], Z.ravel() * gs[2]], 1) + (c * gs - f)   # offsets from f, um
    q = (m[None] + u @ Ji.T) / MV                                               # fish 1 voxel x, y, z
    pm = map_coordinates(M, np.stack([q[:, 2], q[:, 1], q[:, 0]]), order=1, mode="constant", cval=np.nan)
    good = np.isfinite(pm)
    land = np.full((2 * sz + 1, 2 * sxy + 1, 2 * sxy + 1), np.nan)
    if good.mean() < 0.6:
        return land, None
    qm = quantize(pm[good], M[::3, ::6, ::6][M[::3, ::6, ::6] > np.percentile(M, 30)], bins)
    qa_all = quantize(Fd, Fd[mask], bins)
    for iz, dz in enumerate(range(-sz, sz + 1)):
        for iy, dy in enumerate(range(-sxy, sxy + 1)):
            for ix, dx in enumerate(range(-sxy, sxy + 1)):
                zz, yy, xx = c[2] + dz + Z.ravel(), c[1] + dy + Y.ravel(), c[0] + dx + X.ravel()
                ok = (zz >= 0) & (zz < Fd.shape[0]) & (yy >= 0) & (yy < Fd.shape[1]) & (xx >= 0) & (xx < Fd.shape[2])
                k = good & ok
                if k.mean() < 0.6:
                    continue
                qa = qa_all[zz[k], yy[k], xx[k]]
                land[iz, iy, ix] = nmi_q(qa, qm[k[good]], bins)
    return land, gs


def pick(land, gs, sxy, sz, near_um=8.0):
    """The landscape's peak and how DISTINCT it is: (best - best farther than near_um from it) / (best - median); a
    ridge (a midline that looks the same along its length) or two equal peaks give a small value."""
    k = np.unravel_index(np.nanargmax(land), land.shape)
    Z, Y, X = np.meshgrid(*(np.arange(n) for n in land.shape), indexing="ij")
    dist = np.sqrt(((X - k[2]) * gs[0]) ** 2 + ((Y - k[1]) * gs[1]) ** 2 + ((Z - k[0]) * gs[2]) ** 2)
    far = np.nanmax(np.where(dist > near_um, land, np.nan))
    best, med = land[k], np.nanmedian(land)
    d = np.array([(k[2] - sxy) * gs[0], (k[1] - sxy) * gs[1], (k[0] - sz) * gs[2]])
    edge = k[0] in (0, 2 * sz) or k[1] in (0, 2 * sxy) or k[2] in (0, 2 * sxy)
    return d, float(best), float(land[sz, sxy, sxy]), float((best - far) / max(best - med, 1e-9)), bool(edge)


def refine2(names, m, f, M, F, tag="refined2", margin=0.004, distinct=0.15):
    """Two passes: a wide search (+-20 um in x, y; +-30 um in z) kept only where the peak is inside the window,
    distinct and above no shift by `margin`; the spline refitted; then a narrow polish (+-8 um) of every landmark."""
    Fd = F[:, ::DS, ::DS]
    mask = brain_mask(Fd)
    f_new, rep = f.copy(), {n: {} for n in names}
    for pas, (sxy, sz) in enumerate(((13, 15), (5, 4))):
        m2f = tps(m, f_new)
        for i, n in enumerate(names):
            land, gs = match(M, Fd, mask, m[i], f_new[i], m2f, half_z=12.0, sxy=sxy, sz=sz)
            if gs is None or not np.isfinite(land).any():
                rep[n][f"pass{pas + 1}"] = "no fish-1 data around it"
                continue
            d, best, at0, dis, edge = pick(land, gs, sxy, sz)
            take = (best - at0 > margin) and not edge and dis > distinct
            if take:
                f_new[i] = f_new[i] + d
            rep[n][f"pass{pas + 1}"] = {"shift_um": np.round(d, 1).tolist(), "nmi_at_0": round(at0, 4),
                                        "nmi_best": round(best, 4), "distinct": round(dis, 2), "on_edge": edge,
                                        "taken": bool(take)}
            print(f"  pass {pas + 1} {n:6s} shift x {d[0]:+6.1f} y {d[1]:+6.1f} z {d[2]:+6.1f} um  NMI {at0:.4f} -> "
                  f"{best:.4f}  distinct {dis:.2f}{'  EDGE' if edge else ''}{'  taken' if take else ''}", flush=True)
    tot = f_new - f
    for i, n in enumerate(names):
        rep[n]["total_shift_um"] = np.round(tot[i], 1).tolist()
    json.dump(rep, open(os.path.join(OUT, f"{tag}_shifts.json"), "w"), indent=1)
    save_landmarks(os.path.join(BW, f"fish_1_landmarks_{tag}.csv"), names, m, f_new)
    return f_new, rep


def register(names, m, f, M, F, tag="registered", mesh=(8, 16, 5)):
    """WHOLE-BRAIN REGISTRATION ON TOP OF THE LANDMARK SPLINE (the per-landmark patch match proved unreliable: Pt-24 moved
    9.6 um off the midline for a higher patch score): fish 1 warped by the spline into the atlas grid, then registered to
    the atlas by Mattes mutual information inside the atlas brain -- an affine stage (a global tilt or scale the 4 planes
    missed), then a cubic B-spline (control points every ~`mesh` cells of the brain's box) -- 3 resolutions each.
    The atlas point y then maps to fish 1 at m = spline_f2m(T(y)). Each landmark's ATLAS point is re-paired by
    solving T(y) = spline_m2f(m) for y (fixed-point iteration); written as fish_1_landmarks_<tag>.csv."""
    import SimpleITK as sitk
    Fd = F[:, ::DS, ::DS]
    mask = brain_mask(Fd)
    f2m = tps(f, m)
    W, cov = warp(M, F.shape, f2m)
    sp = (float(FV[0] * DS), float(FV[1] * DS), float(FV[2]))
    fix = sitk.GetImageFromArray(Fd.astype(np.float32))
    fix.SetSpacing(sp)
    mov = sitk.GetImageFromArray(np.nan_to_num(W, nan=0.0).astype(np.float32))
    mov.SetSpacing(sp)
    fm = sitk.GetImageFromArray((mask & cov).astype(np.uint8))
    fm.SetSpacing(sp)

    def method(shrink=(4, 2, 1), sig=(4.0, 2.0, 1.0)):
        R_ = sitk.ImageRegistrationMethod()
        R_.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
        R_.SetMetricSamplingStrategy(R_.RANDOM)
        R_.SetMetricSamplingPercentage(0.05, seed=0)
        R_.SetMetricFixedMask(fm)
        R_.SetInterpolator(sitk.sitkLinear)
        R_.SetShrinkFactorsPerLevel(list(shrink))
        R_.SetSmoothingSigmasPerLevel(list(sig))
        R_.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        return R_
    c = fix.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in fix.GetSize()])
    aff0 = sitk.AffineTransform(3)
    aff0.SetCenter(c)
    Ra = method()
    Ra.SetInitialTransform(aff0, inPlace=False)
    Ra.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-3, numberOfIterations=150,
                                               relaxationFactor=0.6, gradientMagnitudeTolerance=1e-6)
    Ra.SetOptimizerScalesFromPhysicalShift()
    aff = Ra.Execute(fix, mov)
    print(f"[register] affine: MI {Ra.GetMetricValue():.4f} after {Ra.GetOptimizerIteration()} iterations "
          f"({Ra.GetOptimizerStopConditionDescription()[:60]})", flush=True)
    bs0 = sitk.BSplineTransformInitializer(fix, list(mesh), 3)
    Rb = method()
    Rb.SetMovingInitialTransform(aff)
    Rb.SetInitialTransform(bs0, inPlace=True)
    Rb.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=80, maximumNumberOfCorrections=5,
                            maximumNumberOfFunctionEvaluations=400, costFunctionConvergenceFactor=1e7)
    bs = Rb.Execute(fix, mov)
    print(f"[register] B-spline: MI {Rb.GetMetricValue():.4f} after {Rb.GetOptimizerIteration()} iterations", flush=True)
    T = sitk.CompositeTransform(3)
    T.AddTransform(aff)
    T.AddTransform(bs)                                   # applied first: the B-spline, then the affine
    try:
        sitk.WriteTransform(T, os.path.join(OUT, f"{tag}_atlas_to_splinewarp.h5"))
    except RuntimeError as e_:
        print(f"[register] transform not written: {e_}")

    def Tf(Y):                                           # atlas um -> the spline-warped grid's um (same frame)
        return np.array([T.TransformPoint(tuple(map(float, y))) for y in Y])
    # the scores of the composed warp, as `assess`: fish 1 at spline_f2m(T(y)) on the atlas grid
    from scipy.ndimage import map_coordinates
    nz, ny, nx = Fd.shape
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    disp = sitk.TransformToDisplacementField(T, sitk.sitkVectorFloat64, fix.GetSize(), fix.GetOrigin(), fix.GetSpacing(),
                                             fix.GetDirection())
    D = sitk.GetArrayFromImage(disp)                      # [z, y, x, 3] um
    P = np.stack([xx * sp[0], yy * sp[1], zz * sp[2]], -1).reshape(-1, 3) + D.reshape(-1, 3)
    W2 = np.empty(len(P), np.float32)
    cov2 = np.empty(len(P), bool)
    for a in range(0, len(P), 2_000_000):
        q = f2m(P[a:a + 2_000_000]) / MV
        W2[a:a + 2_000_000] = map_coordinates(M, np.stack([q[:, 2], q[:, 1], q[:, 0]]), order=1, mode="constant",
                                              cval=np.nan)
        cov2[a:a + 2_000_000] = np.all((q >= 0) & (q <= np.array(M.shape[::-1]) - 1), axis=1)
    W2, cov2 = W2.reshape(Fd.shape), cov2.reshape(Fd.shape)
    sc = scores(W2, cov2, Fd, mask)
    overlays(W2, Fd, tag, mask)
    # re-pair each landmark: the atlas point y with T(y) = spline_m2f(m) (= Cedric's f), by fixed-point iteration
    f_new = f.copy()
    for i in range(len(f)):
        y = f[i].copy()
        for _ in range(30):
            y = y - (Tf(y[None])[0] - f[i])
        f_new[i] = y
    e = loo(m, f_new)
    sc.update({"loo_um_median": float(np.median(e)), "loo_um_max": float(e.max()),
               "landmark_shift_um": dict(zip(names, np.round(f_new - f, 1).tolist()))})
    json.dump(sc, open(os.path.join(OUT, f"{tag}_scores.json"), "w"), indent=1)
    save_landmarks(os.path.join(BW, f"fish_1_landmarks_{tag}.csv"), names, m, f_new)
    print(f"[{tag}] NMI {sc['nmi']:.4f} over {sc['voxels']:,} voxels (the registration itself); its 30 landmarks re-paired: "
          f"leave-one-out median {sc['loo_um_median']:.1f} um, max {sc['loo_um_max']:.1f} um")
    return f_new, sc


def refine(names, m, f, M, F, tag="refined", margin=0.004):
    Fd = F[:, ::DS, ::DS]
    mask = brain_mask(Fd)
    m2f = tps(m, f)
    f_new, rep = f.copy(), {}
    for i, n in enumerate(names):
        land, gs = match(M, Fd, mask, m[i], f[i], m2f)
        if gs is None or not np.isfinite(land).any():
            rep[n] = {"kept": "no fish-1 data around it"}
            continue
        sz, sxy = (land.shape[0] - 1) // 2, (land.shape[1] - 1) // 2
        k = np.unravel_index(np.nanargmax(land), land.shape)
        d = np.array([(k[2] - sxy) * gs[0], (k[1] - sxy) * gs[1], (k[0] - sz) * gs[2]])
        gain = float(land[k] - land[sz, sxy, sxy])
        edge = k[0] in (0, 2 * sz) or k[1] in (0, 2 * sxy) or k[2] in (0, 2 * sxy)
        take = gain > margin and not edge
        if take:
            f_new[i] = f[i] + d
        rep[n] = {"shift_um": np.round(d, 1).tolist(), "nmi_at_0": round(float(land[sz, sxy, sxy]), 4),
                  "nmi_best": round(float(land[k]), 4), "on_edge": bool(edge), "taken": bool(take)}
        print(f"  {n:6s} shift x {d[0]:+6.1f} y {d[1]:+6.1f} z {d[2]:+6.1f} um  NMI {land[sz, sxy, sxy]:.4f} -> {land[k]:.4f}"
              f"{'  EDGE' if edge else ''}{'  taken' if take else ''}", flush=True)
    json.dump(rep, open(os.path.join(OUT, f"{tag}_shifts.json"), "w"), indent=1)
    save_landmarks(os.path.join(BW, f"fish_1_landmarks_{tag}.csv"), names, m, f_new)
    return f_new, rep


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names, m, f = load_landmarks(os.path.join(BW, "fish_1_landmarks.csv"))
    M, F = volumes()
    if sys.argv[1] == "assess":
        assess(names, m, f, "cedric", M, F)
    elif sys.argv[1] == "refine":
        f2, rep = refine(names, m, f, M, F)
        assess(names, m, f2, "refined", M, F)
    elif sys.argv[1] == "register":
        f2, sc = register(names, m, f, M, F)
        assess(names, m, f2, "registered_landmarks", M, F)
    elif sys.argv[1] == "refine2":
        f2, rep = refine2(names, m, f, M, F)
        assess(names, m, f2, "refined2", M, F)
