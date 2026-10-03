#!/usr/bin/env python
"""Freeze WHOLISTIC recording f338 as two training REFERENCES, one per channel, in exp16's schema.

    PYTHONPATH=src python tools/export_wholistic_recording.py [--src DIR] [--f 4] [--name wholistic_f338]

The source is exp19's streamed file, `experiments/exp19_wholistic_graphcast/data/f338_b<B>.npy` (uint16
[T, Z, C, Y, X], each value the rounded mean of B x B camera pixels; `data/stream_bin_nd2.py`), with its
JSON beside it (voxel size, per-volume times). This script block-averages it once more, F x F in (y, x),
to the LAW'S grid, and writes

    graphs_data/redox/<name>_gcamp_recording.npz      GCaMP7f: calcium, plus tissue moving through the voxel
    graphs_data/redox/<name>_mcherry_recording.npz    mCherry-CAAX (beta-actin2 promoter): motion, no calcium

each with exp16's keys, so `plexus.tasks.field_recording` and the trainer read them unchanged:

    ratio   [T, Z, Y, X] float32   the channel's signal F / F0 on the fish, 0 off it (the key keeps
                                   exp16's name; it holds a normalised fluorescence here, not a ratio)
    mask    [T, Z, Y, X] bool      the fish: one mask for every volume (the larva is held in agarose)
    t_s     [T] float64            each volume's first-plane time, s since the first volume
    dx_um, dz_um                   the law-grid voxel edge in x and y, and the plane step, um

WHY graphs_data/redox/. The trainer loads a field recording by name from `field_recording.GROUP`
("redox", `trainer.py` `FR.load(ref["field_recording"])`); a `group:` key in the reference is the clean
fix, left to the trainer's owner. The `wholistic_` prefix keeps the two apart (exp19 `## Decisions`).

WHY F / F0. F is the channel minus its background (the median, over time, of the voxels off the fish, per
plane); F0 is each voxel's median over the TRAINING volumes only (the first 1 - test - val of the
recording), so nothing of the held-out window enters the normalisation. Both channels then sit near 1,
and a skill is read against each channel's own baseline.

THE GRID IS NOT SQUARE. f338's planes are 1,708 x 2,304 camera pixels (`data/f338_inspect.json`): the fish lies
along x. The field keeps that shape, unpadded, through `ScalarField`'s `per_axis: true` (axis k holds
round(world[k] R) voxels), with R the y count and world = [Z / R, 1, X / R].

THE SPECS ARE DERIVED (`--specs`). The forward spec `config/redox/wl_f338_in4.yaml` (the grid: R = Y, world
[Z / R, 1, X / R], `per_axis: true`, spacing [dz, dx, dx] um) and the training specs `wl_gc_persist`, `wl_gc_base` and its mCherry twin
`wl_gc_base_mc` (split sizes) are written from the exported recording, so no grid number is typed. An existing
spec is never overwritten.

WHY A FLOOR ON F0 (2026-10-02, after batch 1). Without it, voxels at the fish's edge whose background-subtracted
baseline F0 is near 0 reached F / F0 = 818: 8.6 % of the GCaMP voxels went above 10 at some volume and carried 82 %
of the squared deviation every R2 is computed on (mCherry: 0.24 % of voxels, 92 %). Batch 1 was scored on them. A
voxel whose F0 is below `f0_floor` x the fish's median F0 is now OFF the mask: it is near background, and what moves
there is tissue sliding in and out, not a cell's calcium. Flooring F0 instead would keep those voxels and their
bright transients.

WHY ONE MASK. The fish is the voxels whose time-mean mCherry (a membrane marker in every cell) is above
Otsu's threshold on its log; holes in a plane are filled. A per-volume mask would move with the
tissue and the scorer would count motion as a mask change, which is the motion question exp19 asks
of the mCherry channel instead.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from plexus.paths import graphs_data_path                              # noqa: E402

SRC = os.path.join(ROOT, "experiments", "exp19_wholistic_graphcast", "data")
CHANNELS = {0: "gcamp", 1: "mcherry"}    # nd2 channel order of f338; checked against the JSON's names below


def otsu(v, bins=256):
    """Otsu's threshold of the 1-D sample v."""
    h, e = np.histogram(v, bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def block_mean(x, f):
    """(..., Y, X) -> (..., Y // f, X // f), the remainder cropped."""
    *lead, Y, X = x.shape
    ny, nx = Y // f, X // f
    return x[..., :ny * f, :nx * f].reshape(*lead, ny, f, nx, f).mean(axis=(-3, -1))


def fish_mask(mc_mean):
    """[Z, Y, X] bool: Otsu on the log time-mean mCherry, holes filled per plane."""
    from scipy.ndimage import binary_fill_holes
    lv = np.log(np.maximum(mc_mean, 1e-3))
    m = lv > otsu(lv.ravel())
    return np.stack([binary_fill_holes(p) for p in m])


def split_sizes(T, test_frac=0.2, val_frac=0.1):
    """exp19's split: the last 20 % of volumes held out, the 10 % before them for validation."""
    n_test = int(round(test_frac * T))
    n_val = int(round(val_frac * T))
    return n_test, n_val


def export(src, f, name, chunk=32, f0_floor=0.25):
    npys = sorted(glob.glob(os.path.join(src, "f338_b*.npy")))
    npys = [p for p in npys if not p.endswith(".done.npy")]
    if len(npys) != 1:
        raise SystemExit(f"{src}: expected one f338_b<B>.npy, found {npys}")
    npy = npys[0]
    meta = json.load(open(npy[:-4] + ".json"))
    if not meta.get("complete"):
        raise SystemExit(f"{npy}: the stream is not complete ({meta.get('complete')}) -- rerun `stream_bin_nd2.py bin`")
    raw = np.load(npy, mmap_mode="r")
    T, Z, C, Y, X = raw.shape
    chans = meta.get("channels")
    print(f"{os.path.basename(npy)}: T {T}, Z {Z}, C {C}, {Y} x {X}; channels {chans}")
    if chans:
        for c, k in CHANNELS.items():
            # f338 names its channels by excitation line: "488" (GCaMP7f) and "594" (mCherry), Ruetten 2026 Methods
            if not any(w in str(chans[c]).lower() for w in (("gcamp", "488") if k == "gcamp" else ("cherry", "594"))):
                print(f"  WARNING: channel {c} is named {chans[c]!r}, expected {k}; check CHANNELS")
    law = np.empty((C, T, Z, Y // f, X // f), np.float32)
    for t0 in range(0, T, chunk):
        blk = np.asarray(raw[t0:t0 + chunk], dtype=np.float32)              # [t, Z, C, Y, X]
        law[:, t0:t0 + chunk] = np.moveaxis(block_mean(blk, f), 2, 0)
    mask3 = fish_mask(law[1].mean(0))
    n_test, n_val = split_sizes(T)
    n_train = T - n_test - n_val
    times = np.asarray(meta["volume_times_s"], np.float64)
    times = times - times[0]
    dx = float(meta["binned_voxel_um"]["x"]) * f
    dz = float(meta["voxel_um"]["z"])
    sha = hashlib.sha256(open(npy[:-4] + ".json", "rb").read()).hexdigest()
    out = {}
    for c, k in CHANNELS.items():
        x = law[c]
        bg = np.array([np.median(x[:, z][:, ~mask3[z]]) if (~mask3[z]).any() else 0.0 for z in range(Z)])
        F = x - bg[None, :, None, None]
        F0 = np.median(F[:n_train], axis=0)
        f0_med = float(np.median(F0[mask3]))
        ok = mask3 & (F0 >= f0_floor * f0_med)
        sig = np.where(ok[None], F / np.where(ok, F0, 1.0)[None], 0.0).astype(np.float32)
        mask = np.broadcast_to(ok, sig.shape).copy()
        p = graphs_data_path("redox", f"{name}_{k}_recording.npz")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez(p, ratio=sig, mask=mask, t_s=times, dx_um=dx, dz_um=dz)
        prov = {"source": os.path.basename(npy), "source_json_sha256": sha, "url": meta.get("url"),
                "channel_index": c, "channel": k, "channel_name": chans[c] if chans else None,
                "stored_bin_px": meta.get("bin_factor_xy"), "law_bin": f, "dx_um": dx, "dz_um": dz,
                "shape_TZYX": list(sig.shape), "n_train": n_train, "n_val": n_val, "n_test": n_test,
                "fish_voxels": int(ok.sum()), "background_per_plane": bg.tolist(),
                "f0_floor": f0_floor, "f0_median_fish": f0_med, "dropped_low_f0": int((mask3 & ~ok).sum()),
                "volume_interval_s_median": float(np.median(np.diff(times))) if T > 1 else None}
        v = sig[::8][:, ok]
        vm = v.max(0)
        share = float(((v[:, vm > 10] - 1) ** 2).sum() / max(((v - 1) ** 2).sum(), 1e-12))
        prov.update(p99_99=float(np.percentile(v, 99.99)), max=float(v.max()), voxels_ever_above_10=int((vm > 10).sum()),
                    share_sq_dev_from_above_10=share)
        json.dump(prov, open(p[:-4] + ".provenance.json", "w"), indent=1)
        print(f"  {k}: dropped {int((mask3 & ~ok).sum()):,} voxels with F0 < {f0_floor} x the fish median; "
              f"p99.99 {prov['p99_99']:.2f}, max {prov['max']:.1f}, voxels ever > 10: {prov['voxels_ever_above_10']:,}, "
              f"their share of the squared deviation {share:.3f}")
        print(f"  {k}: -> {p}  {sig.shape}, fish {ok.sum():,} voxels of {ok.size:,}, "
              f"F/F0 on fish p5 {np.percentile(sig[:, ok], 5):.3f} median {np.median(sig[:, ok]):.3f} "
              f"p95 {np.percentile(sig[:, ok], 95):.3f}")
        out[k] = prov
    print(f"split: train {n_train}, val {n_val}, test {n_test} volumes; voxel {dx:.2f} x {dx:.2f} x {dz:.2f} um")
    return out


FORWARD = """general:
  name: {fwd}
  seed: 0
  n_frames: 6
  dt: 1.0
  boundary: wall
  dim: 3
  world: [{wz}, 1.0, {wx}]
  save_data: false
sets: {{}}
fields:
  ratio: {{frame: grid, res: {R}, per_axis: true, components: 4, title: "{chan_title} of each voxel, F / F0, now and 3 volumes before"}}
operators:
- {{op: diffuse, model: graphcast, at: ratio, latent: 16, layers: 2, inputs: 4, spacing: [{dz}, {dx}, {dx}], seed: 0,
   title: the voxels' own dynamics, learned by message passing between lattice neighbours}}
schedule: [diffuse]
"""

TRAINING = """name: {name}
model: config/redox/{fwd}.yaml
learnable:
- {{param: theta, op: diffuse, title: GraphCast model of whole-body calcium (exp19)}}
task:
  reference: {{field_recording: {rec}, coarsen: 1, n_test: {n_test}, n_val: {n_val}}}
  observe: {{field: ratio}}
  loss: masked_mse
training: {training}
"""

BASE_TRAINING = ("{optimizer: adam, lr: 0.001, lr_min_frac: 0.05, clip: 1.0, iters: 1500, horizon: [1, 1, 3, 6], "
                 "batch: 4, seed: 0, save_every: 50}")
PERSIST_TRAINING = "{optimizer: adam, lr: 0.001, iters: 0, horizon: [1], batch: 1, seed: 0}"


def write_specs(name, prov, fwd="wl_f338_in4", prefix="wl_gc"):
    """The forward spec and exp19's first training specs, from the exported grid. Never overwrites."""
    g = prov["gcamp"]
    T, Z, R, Xn = g["shape_TZYX"]
    files = {os.path.join(ROOT, "config", "redox", f"{fwd}.yaml"):
             FORWARD.format(fwd=fwd, wz=repr(Z / R), wx=repr(Xn / R), R=R, dz=round(g["dz_um"], 4), dx=round(g["dx_um"], 4),
                            chan_title="whole-body GCaMP7f (or its mCherry twin)")}
    for run, ch, tr in ((f"{prefix}_persist", "gcamp", PERSIST_TRAINING), (f"{prefix}_base", "gcamp", BASE_TRAINING),
                        (f"{prefix}_base_mc", "mcherry", BASE_TRAINING)):
        files[os.path.join(ROOT, "config", "training", "wholistic", f"{run}.yaml")] = TRAINING.format(
            name=run, fwd=fwd, rec=f"{name}_{ch}", n_test=g["n_test"], n_val=g["n_val"], training=tr)
    for p, text in files.items():
        if os.path.exists(p):
            print(f"  keep  {os.path.relpath(p, ROOT)} (exists)")
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(text)
        print(f"  wrote {os.path.relpath(p, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC, help="folder holding f338_b<B>.npy and its JSON")
    ap.add_argument("--f", type=int, default=4, help="block mean, stored voxels per law voxel in x and y")
    ap.add_argument("--name", default="wholistic_f338")
    ap.add_argument("--f0-floor", type=float, default=0.25, help="drop voxels whose F0 is below this x the fish median")
    ap.add_argument("--specs", action="store_true", help="also write the forward and training specs from the grid")
    ap.add_argument("--fwd", default="wl_f338_in4", help="--specs: the forward spec's name")
    ap.add_argument("--prefix", default="wl_gc", help="--specs: the training specs' prefix")
    a = ap.parse_args()
    prov = export(a.src, a.f, a.name, f0_floor=a.f0_floor)
    if a.specs:
        write_specs(a.name, prov, a.fwd, a.prefix)


if __name__ == "__main__":
    main()
