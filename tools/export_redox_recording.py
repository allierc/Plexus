#!/usr/bin/env python
"""Freeze the redox organoid recording as a training REFERENCE: graphs_data/redox/<name>_recording.npz.

    PYTHONPATH=src python tools/export_redox_recording.py            # hlo_washout

The recording is one human liver organoid (HLO, wild type) imaged by two-photon at 770 nm for a
12 h washout, one 14-plane z-stack every 600 s. The Wang lab's MATLAB script
(`Cell_Redox_Ratio_Analysis_3D_Time_Trend.m`, beside the raw file) turned each z-stack into a
redox-ratio volume: per plane, FAD (channel 2) and NADH (channel 3) minus their own 1st
percentile, a cell mask where either exceeds 200 counts (objects under 500 pixels removed, holes
filled), and ratio = G(NADH) / (G(FAD) + eps) with G a Gaussian of sigma 1 pixel, 0 off the mask.
Those volumes are the data; this script only reads them, it never recomputes the ratio.

WHY 69 VOLUMES AND NOT 200. The folder is named `..._T200_...` but that is the script's name
template: the raw `.oir` itself records a TIMELAPSE axis of maxSize 69 (its acquisition settings
planned 73, 0-43,800 s at 600 s), and the MATLAB loop runs `t = 1:reader.getSizeT()`. The byte
count agrees: 1.539 GB of raw data is 2,935 planes of 512 x 512 x 2 bytes, 69 x 14 z x 3 channels
plus ~1% of header. The other 131 never existed.

WHY THE TIMES ARE READ, NOT ASSUMED. Each plane's frame header in the raw file carries its own
TIMELAPSE position, in ms. They show a 600.0 s cycle except ONE gap of 885 s between volumes 4
and 5 (1-based), and 0.730 s between the 14 planes of a stack (9.5 s per stack). A forecast whose
step is "10 min" is one step off that clock only there, but a trend fitted on the index instead of
the time would be bent by it, so the times are stored and every baseline reads them.

WHY THE MASK IS `ratio != 0`. The script writes 0 off its cell mask and a strictly positive ratio
on it (G(NADH) > 0 wherever the smoothed NADH is, which it is inside a mask built from counts
above 200). The mask read back is checked plane by plane against the script's own
`CellPixelCount` (Redox_Ratio_Detailed_Layers.xlsx): a disagreement stops the export.

    ratio      [T, Z, Y, X] float32   the redox ratio NADH/FAD, 0 off the organoid; T=69, Z=14, 512 x 512
    mask       [T, Z, Y, X] bool      the script's cell mask, ratio != 0
    t_s        [T]  float64           each stack's first-plane time, s since the first stack
    t_plane_s  [T, Z] float64         every plane's own time, s
    z_um       [Z]  float64           the z-drive position of each plane, um
    dx_um, dz_um                      pixel size in x and y, and the z step, um
    trend_t, trend_fad, trend_nadh, trend_ratio   Development_Time_Trend.xlsx: per stack, the mean
                                      over the 14 planes of each plane's masked mean FAD, NADH
                                      and their ratio (mean NADH / mean FAD, not a mean of ratios)
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from plexus.paths import graphs_data_path                              # noqa: E402

NAME = "hlo_washout"
STEM = "0012-011-HLO-WT070326-770nm-512-P8-30X-zstack-12h-Resonant-washout_A01_G001_0001"
RAW = [STEM + ".oir", STEM + "_00001"]           # Olympus splits one acquisition over two files
RATIO_DIR = "Organoid_Redox_Ratio_Analysis_T200_3D_Every_10_min"
SCRIPT = "Cell_Redox_Ratio_Analysis_3D_Time_Trend.m"

FRAME = re.compile(rb"<commonframe:axisType>ZSTACK</commonframe:axisType>\s*<commonframe:position>"
                   rb"([-\d.eE]+)</commonframe:position>\s*</commonframe:axisValue>\s*<commonframe:axisValue>"
                   rb"\s*<commonframe:axisType>TIMELAPSE</commonframe:axisType>\s*<commonframe:position>([-\d.eE]+)<")
AXIS = re.compile(rb"<commonimage:axis[^>]*>\s*<commonparam:axis>(\w+)</commonparam:axis>\s*<commonparam:startPosition>"
                  rb"([-\d.]+)</commonparam:startPosition>\s*<commonparam:endPosition>([-\d.]+)"
                  rb"</commonparam:endPosition>\s*<commonparam:step>([-\d.]+)</commonparam:step>\s*"
                  rb"<commonparam:maxSize>(\d+)</commonparam:maxSize>")
PIXEL = re.compile(rb"<commonphase:length>\s*<commonparam:x>([\d.]+)</commonparam:x>\s*<commonparam:y>([\d.]+)"
                   rb"</commonparam:y>\s*<commonparam:z>([\d.]+)</commonparam:z>")


def sha256_file(p, chunk=64 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def raw_metadata(folder):
    """Per-plane (z, t) stamps, in file order, and the image's own axes, from the .oir XML headers.

    Every frame header carries `axisValue` ZSTACK (um) then TIMELAPSE (ms). The first `imageInfo`
    block lists the RECORDED axes (maxSize = what was acquired); a later block, the settings."""
    stamps, axes, pixel = [], None, None
    for name in RAW:
        buf = b""
        with open(os.path.join(folder, name), "rb") as fh:
            for c in iter(lambda: fh.read(64 << 20), b""):
                buf = buf[-8192:] + c
                for m in FRAME.finditer(buf):
                    stamps.append((float(m.group(1)), float(m.group(2))))
                if axes is None and AXIS.search(buf):
                    axes = {a.decode(): dict(start=float(s), end=float(e), step=float(st), size=int(n))
                            for a, s, e, st, n in AXIS.findall(buf)[:2]}
                if pixel is None and PIXEL.search(buf):
                    pixel = tuple(float(v) for v in PIXEL.search(buf).groups())
    out = []                                     # the 8 kB overlap can report a stamp twice
    for s in stamps:
        if not out or out[-1] != s:
            out.append(s)
    return np.array(out), axes, pixel


def main(name=NAME):
    import pandas as pd
    import tifffile

    src = graphs_data_path("redox")
    rdir = os.path.join(src, RATIO_DIR)
    tifs = sorted(f for f in os.listdir(rdir) if re.search(r"_RedoxRatio_T\d{3}\.tif$", f))
    idx = [int(re.search(r"_T(\d{3})\.tif$", f).group(1)) for f in tifs]
    assert idx == list(range(1, len(tifs) + 1)), f"volumes are not T001..T{len(tifs):03d}: {idx}"

    stamps, axes, pixel = raw_metadata(src)
    T, Z = axes["TIMELAPSE"]["size"], axes["ZSTACK"]["size"]
    assert len(tifs) == T, f"{len(tifs)} ratio volumes, but the raw file records {T} timepoints"
    assert stamps.shape == (T * Z, 2), f"{stamps.shape[0]} frame stamps, expected {T} x {Z}"
    z_um = stamps[:Z, 0]
    assert (stamps[:, 0].reshape(T, Z) == z_um).all(), "the z positions change between stacks"
    t_plane_s = stamps[:, 1].reshape(T, Z) / 1000.0
    t_plane_s -= t_plane_s[0, 0]
    t_s = t_plane_s[:, 0].copy()

    ratio = np.stack([tifffile.imread(os.path.join(rdir, f)) for f in tifs]).astype(np.float32)
    assert ratio.shape[1:] == (Z, 512, 512), ratio.shape
    assert np.isfinite(ratio).all() and (ratio >= 0).all(), "a ratio that is negative or not finite"
    mask = ratio != 0

    layers = pd.read_excel(os.path.join(rdir, "Redox_Ratio_Detailed_Layers.xlsx"))
    n_script = np.zeros((T, Z), np.int64)
    n_script[layers.TimePoint.values - 1, layers.ZStack.values - 1] = layers.CellPixelCount.values
    n_ours = mask.sum((2, 3))
    bad = np.argwhere(n_script != n_ours)
    assert len(bad) == 0, (f"{len(bad)} planes whose mask count differs from the script's, first "
                           f"(t, z) = {bad[0] + 1}: ours {n_ours[tuple(bad[0])]}, script {n_script[tuple(bad[0])]}")

    trend = pd.read_excel(os.path.join(rdir, "Development_Time_Trend.xlsx"))
    assert list(trend.TimePoint) == list(range(1, T + 1))

    out = graphs_data_path("redox", f"{name}_recording.npz")
    np.savez_compressed(out, ratio=ratio, mask=mask, t_s=t_s, t_plane_s=t_plane_s, z_um=z_um,
                        dx_um=np.float64(pixel[0]), dz_um=np.float64(axes["ZSTACK"]["step"]),
                        trend_t=trend.TimePoint.values.astype(np.int64),
                        trend_fad=trend.mean_Mean_FAD.values, trend_nadh=trend.mean_Mean_NADH.values,
                        trend_ratio=trend.mean_Redox_Ratio.values)
    dt = np.diff(t_s)
    prov = dict(
        name=name, organoid="human liver organoid (HLO) wild type, 770 nm two-photon, 12 h washout",
        source=dict(raw=RAW, ratio_dir=RATIO_DIR, script=SCRIPT,
                    sha256_script=sha256_file(os.path.join(src, SCRIPT)),
                    sha256_tif={f: sha256_file(os.path.join(rdir, f)) for f in tifs},
                    sha256_trend_xlsx=sha256_file(os.path.join(rdir, "Development_Time_Trend.xlsx")),
                    sha256_layers_xlsx=sha256_file(os.path.join(rdir, "Redox_Ratio_Detailed_Layers.xlsx"))),
        raw_axes=axes, channels=dict(fad=2, nadh=3, filters={"2": "BA495-540", "3": "BA410-460"}),
        shape=list(ratio.shape), dx_um=pixel[0], dz_um=axes["ZSTACK"]["step"],
        interval_s=dict(median=float(np.median(dt)), min=float(dt.min()), max=float(dt.max()),
                        gaps=[dict(after_volume=int(i + 1), seconds=float(dt[i]))
                              for i in np.where(np.abs(dt - np.median(dt)) > 1.0)[0]]),
        span_h=float(t_s[-1] / 3600), stack_s=float(t_plane_s[0, -1] - t_plane_s[0, 0]),
        mask_fraction=float(mask.mean()), mask_check="per-plane count == Redox_Ratio_Detailed_Layers.CellPixelCount",
        sha256_ratio=hashlib.sha256(ratio.tobytes()).hexdigest(),
        sha256_mask=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest())
    json.dump(prov, open(out[:-4] + ".provenance.json", "w"), indent=1)
    on = ratio[mask]
    print(f"{name}: {T} stacks x {Z} planes x 512 x 512, {t_s[-1] / 3600:.2f} h, interval median "
          f"{np.median(dt):.1f} s (gaps {prov['interval_s']['gaps']}), mask {mask.mean():.1%} of voxels, "
          f"ratio on mask p5 {np.percentile(on, 5):.3f} / median {np.median(on):.3f} / p95 "
          f"{np.percentile(on, 95):.3f} -> {out}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or [NAME]))
