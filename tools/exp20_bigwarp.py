"""exp20: GLUCOSE FISH 1 ONTO THE Z-BRAIN ATLAS WITH BIGWARP (Cedric, 2026-10-04: "fit fish 1 to the atlas, use BigWarp").

BigWarp (Fiji: Plugins > BigDataViewer > Big Warp; Bogovic et al. 2016) is landmark-driven: Cedric clicks matching
points in the two volumes, BigWarp fits a thin-plate spline. This tool prepares both volumes and then reads his
landmarks back to place every cell in the atlas.

    PYTHONPATH=src:tools python tools/exp20_bigwarp.py export        # the two TIFF stacks for BigWarp
    PYTHONPATH=src:tools python tools/exp20_bigwarp.py apply         # after the landmarks are saved: cells -> regions

export
    moving  data/atlas/bigwarp/fish1_moving.tif: the fish's mean anatomy (volume0.hdf5 volume_mean, the volume the
            cells were segmented in), turned into the atlas's frame -- head UP (the rows run head to tail; the raw
            volume has the head at +x) -- a rotation, no mirror; z kept (slice 0 the first light-sheet plane). 16-bit,
            ImageJ calibration 0.8125 x 0.8125 x 2 um (exp20's inferred voxel)
    fixed   data/atlas/bigwarp/zbrain_reference.tif: the Z-Brain reference brain (Randlett 2015; Janelia Figshare
            7272617 ReferenceBrain.mat anat_stack_norm), head up, z 0 dorsal, 0.798 x 0.798 x 2 um
    and data/atlas/bigwarp/frame.json: how a cell's raw position (pos_um x, y, z) maps into the moving stack's
            physical frame: (X, Y, Z) = (y, 831.19 - x, z) um.

apply (needs data/atlas/bigwarp/fish1_landmarks.csv, BigWarp's File > Export landmarks)
    a thin-plate spline (kernel r^2 log r, BigWarp's) fitted from the moving landmarks to the fixed ones carries every
    cell into the atlas; each cell then gets the Z-Brain regions (MaskDatabase.mat, 294 masks, overlapping) its atlas
    voxel lies in. Writes data/atlas/fish1_regions.npz (atlas position per cell, region indices) and a summary of the
    gut-responsive cells by region.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
DATA = os.path.join(EXP, "data")
BW = os.path.join(DATA, "atlas", "bigwarp")
ZB = os.path.join(DATA, "atlas", "zbrain", "Additional_mat_files")
REC = "gutbrain_glucose_f1"
VOX = (0.8125, 0.8125, 2.0)            # fish x, y, z um per voxel (export_gutbrain_recording.VOXEL_UM, inferred)
ZVOX = (0.798, 0.798, 2.0)             # Z-Brain x, y, z (Randlett 2015)


def _u16(a):
    lo, hi = np.percentile(a, [0.5, 99.9])
    return (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 65535).astype(np.uint16)


def export():
    import h5py
    import scipy.io as sio
    import tifffile
    os.makedirs(BW, exist_ok=True)
    with h5py.File(os.path.join(DATA, "glucose_fish1_volume0.hdf5")) as h:
        V = h["volume_mean"][:].astype(np.float32)                    # (z, y, x), head at +x
    nx = V.shape[2]
    M = np.ascontiguousarray(V[:, :, ::-1].transpose(0, 2, 1))       # (z, rows = head to tail, cols = y)
    tifffile.imwrite(os.path.join(BW, "fish1_moving.tif"), _u16(M), imagej=True,
                     resolution=(1 / VOX[0], 1 / VOX[1]), metadata={"spacing": VOX[2], "unit": "um", "axes": "ZYX"})
    R = sio.loadmat(os.path.join(ZB, "ReferenceBrain.mat"), variable_names=["anat_stack_norm"])["anat_stack_norm"]
    F = np.ascontiguousarray(R.astype(np.float32).transpose(2, 0, 1))  # (z, y, x): head up, z 0 dorsal
    tifffile.imwrite(os.path.join(BW, "zbrain_reference.tif"), _u16(F), imagej=True,
                     resolution=(1 / ZVOX[0], 1 / ZVOX[1]), metadata={"spacing": ZVOX[2], "unit": "um", "axes": "ZYX"})
    frame = {"moving": "fish1_moving.tif", "fixed": "zbrain_reference.tif", "moving_shape_zyx": list(M.shape),
             "fixed_shape_zyx": list(F.shape), "moving_voxel_xyz_um": VOX, "fixed_voxel_xyz_um": ZVOX,
             "cell_to_moving": "(X, Y, Z) = (y, (nx - 1) * vx - x, z) um, (x, y, z) a cell's raw pos_um",
             "nx": nx, "vx": VOX[0]}
    json.dump(frame, open(os.path.join(BW, "frame.json"), "w"), indent=1)
    print(f"[bigwarp] moving {M.shape} (z, y, x) at {VOX} um, fixed {F.shape} at {ZVOX} um -> {BW}")


def cells_in_moving(P):
    """Raw pos_um (x, y, z) -> the moving stack's physical frame (X, Y, Z), um."""
    nx = 1024
    return np.stack([P[:, 1], (nx - 1) * VOX[0] - P[:, 0], P[:, 2]], 1)


def tps_fit(src, dst):
    """3-D thin-plate spline src -> dst with BigWarp's kernel U(r) = r^2 log r: f(p) = A p + b + sum_k w_k U(|p - s_k|)."""
    n = len(src)
    d = np.linalg.norm(src[:, None] - src[None], axis=2)
    K = np.where(d > 0, d ** 2 * np.log(np.where(d > 0, d, 1.0)), 0.0)
    Pm = np.c_[np.ones(n), src]
    L = np.zeros((n + 4, n + 4))
    L[:n, :n], L[:n, n:], L[n:, :n] = K, Pm, Pm.T
    Y = np.zeros((n + 4, 3))
    Y[:n] = dst
    coef = np.linalg.solve(L, Y)
    return src, coef


def tps_apply(model, p, chunk=20000):
    src, coef = model
    n = len(src)
    out = np.zeros((len(p), 3))
    for s in range(0, len(p), chunk):
        q = p[s:s + chunk]
        d = np.linalg.norm(q[:, None] - src[None], axis=2)
        U = np.where(d > 0, d ** 2 * np.log(np.where(d > 0, d, 1.0)), 0.0)
        out[s:s + chunk] = U @ coef[:n] + np.c_[np.ones(len(q)), q] @ coef[n:]
    return out


def read_landmarks(path):
    """BigWarp's landmark CSV: name, active, moving x, y, z, fixed x, y, z (physical units)."""
    import csv
    mv, fx = [], []
    for row in csv.reader(open(path)):
        if len(row) >= 8 and row[1].strip().strip('"').lower() == "true":
            v = [float(c) for c in row[2:8]]
            mv.append(v[:3]); fx.append(v[3:])
    return np.array(mv), np.array(fx)


def apply():
    import h5py
    from plexus.tasks import trace_recording as TR
    mv, fx = read_landmarks(os.path.join(BW, "fish1_landmarks.csv"))
    print(f"[bigwarp] {len(mv)} landmark pairs")
    model = tps_fit(mv, fx)
    loo = []                                            # leave-one-out error: how well the landmarks agree
    for k in range(len(mv)):
        keep = np.arange(len(mv)) != k
        loo.append(float(np.linalg.norm(tps_apply(tps_fit(mv[keep], fx[keep]), mv[k:k + 1])[0] - fx[k])))
    P = np.asarray(TR.load(REC)["pos_um"], np.float64)
    A = tps_apply(model, cells_in_moving(P))            # atlas um (x, y, z)
    with h5py.File(os.path.join(ZB, "MaskDatabase.mat")) as h:
        H, W, Z = int(h["height"][0, 0]), int(h["width"][0, 0]), int(h["Zs"][0, 0])
        names = ["".join(chr(c) for c in h[r][:].ravel()) for r in h["MaskDatabaseNames"][:, 0]]
        from scipy.sparse import csc_matrix
        g = h["MaskDatabase"]
        S = csc_matrix((g["data"][:], g["ir"][:], g["jc"][:]), shape=(H * W * Z, len(names)))
    iy = np.clip(np.round(A[:, 1] / ZVOX[1]).astype(int), 0, H - 1)
    ix = np.clip(np.round(A[:, 0] / ZVOX[0]).astype(int), 0, W - 1)
    iz = np.clip(np.round(A[:, 2] / ZVOX[2]).astype(int), 0, Z - 1)
    inside = (A[:, 0] >= 0) & (A[:, 0] < W * ZVOX[0]) & (A[:, 1] >= 0) & (A[:, 1] < H * ZVOX[1]) & (A[:, 2] >= 0) & (A[:, 2] < Z * ZVOX[2])
    lin = iy + H * ix + H * W * iz                      # MATLAB column-major (y, x, z)
    Sr = S.tocsr()
    reg = Sr[lin]                                       # [cells, 294]
    np.savez_compressed(os.path.join(DATA, "atlas", "fish1_regions.npz"), atlas_um=A, inside=inside,
                        regions=reg.astype(bool).toarray(), names=np.array(names), loo_um=np.array(loo))
    resp = np.load(os.path.join(DATA, f"baselines_{REC}_cells.npz"))["responsive"]
    cnt = np.asarray(reg[resp].sum(0)).ravel()
    top = np.argsort(cnt)[::-1][:15]
    print(f"[bigwarp] leave-one-out landmark error: median {np.median(loo):.1f} um, max {max(loo):.1f} um; "
          f"{int(inside.sum()):,} of {len(P):,} cells inside the atlas box")
    for i in top:
        print(f"   {cnt[i]:5d} gut-responsive cells in {names[i]}")


if __name__ == "__main__":
    {"export": export, "apply": apply}[sys.argv[1]]()
