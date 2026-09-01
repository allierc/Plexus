"""WHAT EACH LEVEL OF A HASH ENCODING CONTRIBUTES, in space and in time.

Three panels, ported from `ngp-demo/scripts/gui_scalar_time.py`, which is where they were designed
and measured. All three rest on one thing -- `MultiResHashGrid.set_level_window(alpha)` -- because

    what level k adds  =  f(levels <= k)  -  f(levels <= k-1)

and without a way to switch levels off there is no per-level picture at all, only the total.

    level_map        per block, the FINEST level still contributing there. Asks "is capacity
                     spent where the data is". On a masked field it has a real answer: this toy's
                     fine rule occupies 15.4% of the frame, so the discs should read finer than
                     the background.
    level_montage    what each level ADDS, at one time slice. The question that always has an
                     answer, and the panel to read when `level_map` saturates.
    level_kymograph  what each level adds in TIME. Two sweeps, because one line answers half the
                     question: x across the middle row, then y down the middle column, t downward
                     in both. A structure running along one sweep is invisible in it and obvious
                     in the other.

A WARNING CARRIED ACROSS FROM THE ORIGINAL, because it is the kind that costs an afternoon: with
16 levels `level_map` saturates. The finest level clears any per-block threshold almost everywhere
-- measured, levels 13-15 in 253 of 256 blocks -- and the panel degenerates into a 2 px mesh. It is
informative at 8-12 levels and decorative at 16; the montage is the one that stays legible.

TWO SCALING RULES, both of which were wrong once in the original before they were fixed there:

  * EACH TILE IS DRAWN AT ITS OWN LEVEL'S LATTICE, capped at the tile size. A level with 12 cells
    across cannot represent anything finer than 12 cells, and rendering it at 1024 interpolates
    that fact into a blur that reads as detail the level does not have.
  * THE COLOUR SCALE COMES FROM THE DIFFERENCES, not from the field. A fraction of the field's
    range saturates every tile into one flat colour whenever the field is large and the per-level
    contributions are small -- on one dataset the field ran to 1235 counts and a level's share to
    hundreds. The 99th percentile of |difference| over the whole sheet puts it where the
    differences are, and it is written onto the first tile so the sheet carries its own units.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

CMAP = "viridis"


def _cm(x01):
    import matplotlib
    return (matplotlib.colormaps[CMAP](np.clip(x01, 0, 1))[..., :3] * 255).astype(np.uint8)


def _signed(d, vmax):
    """Signed difference: red positive, blue negative, black at zero."""
    x = np.clip(np.asarray(d, np.float64) / max(vmax, 1e-12), -1, 1)
    rgb = np.zeros(x.shape + (3,), np.uint8)
    rgb[..., 0] = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    rgb[..., 2] = (np.clip(-x, 0, 1) * 255).astype(np.uint8)
    return rgb


def _sheet(tiles, labels, path, cols=4, gap=2):
    th, tw, _ = tiles[0].shape
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * (th + gap) - gap, cols * (tw + gap) - gap, 3), np.uint8)
    for i, t in enumerate(tiles):
        y, x = (i // cols) * (th + gap), (i % cols) * (tw + gap)
        sheet[y:y + th, x:x + tw] = t
    im = Image.fromarray(sheet)
    dr = ImageDraw.Draw(im)
    for i, lab in enumerate(labels):
        dr.text(((i % cols) * (tw + gap) + 2, (i // cols) * (th + gap) + 1), lab,
                fill=(255, 255, 255))
    im.save(path)
    return path


def _fit(rgb, hs, ws):
    return (rgb if rgb.shape[:2] == (hs, ws)
            else np.asarray(Image.fromarray(rgb).resize((ws, hs), Image.NEAREST)))


@torch.no_grad()
def level_montage(op, path, t: float = 0.0, side: int = 110, cols: int = 4, channel: int = 0):
    """What each level ADDS in space, at one time slice. `op` is a bound `hash_encoding`."""
    enc, shape = op.grid, op._shape
    h, w = shape[0], shape[1]
    sub = max(1, round(max(h, w) / side))
    hs, ws = max(8, h // sub), max(8, w // sub)
    cache = {}

    def upto(k, ny, nx):
        key = (k, ny, nx)
        if key not in cache:
            enc.set_level_window(float(max(0, k + 1)) if k >= 0 else 0.0)
            cache[key] = op.sample((ny, nx), t)[channel].cpu().numpy()
        return cache[key]

    levels = list(range(enc.n_levels))[-(cols * cols - 1):]
    tiles, labels, raw = [], [], []
    try:
        for j, l in enumerate([levels[0] - 1] + levels):
            r = int(enc.resolutions[max(0, l)][0])
            ny, nx = min(hs, r + 1), min(ws, r + 1)
            if j == 0:
                a = upto(l, ny, nx)
                tiles.append(_fit(_cm(a / max(1e-9, np.abs(a).max()) * .5 + .5), hs, ws))
                labels.append(f"0..{l}")
            else:
                raw.append((len(tiles), upto(l, ny, nx) - upto(l - 1, ny, nx)))
                tiles.append(None)
                labels.append(f"L{l}/{r}c")
    finally:
        enc.set_level_window(float(enc.n_levels))

    vmax = float(np.percentile(np.abs(np.concatenate([d.ravel() for _, d in raw])), 99)) or 1e-6
    for i, d in raw:
        tiles[i] = _fit(_signed(d, vmax), hs, ws)
    labels[0] += f"  +-{vmax:.3g}"
    return _sheet(tiles, labels, path, cols)


@torch.no_grad()
def level_kymograph(op, path, n_frames: int, side: int = 110, cols: int = 4, channel: int = 0):
    """What each level adds in TIME. Requires an operator built with `use_time: true`."""
    if not op.use_time:
        raise ValueError("level_kymograph needs a time-encoded operator (use_time: true); a "
                         "static encoding has no time axis to sweep and the panel would be blank")
    enc, shape = op.grid, op._shape
    ny, nx = max(16, min(side, n_frames)), max(16, side)
    cache = {}

    def upto(k, ax):
        if (k, ax) not in cache:
            enc.set_level_window(float(max(0, k + 1)) if k >= 0 else 0.0)
            rows = []
            for i in range(ny):
                tt = i / max(1, ny - 1)
                f = op.sample((shape[0], shape[1]), tt)[channel]
                rows.append((f[:, shape[1] // 2] if ax == "y" else f[shape[0] // 2, :]))
            r = torch.stack(rows).cpu().numpy()
            cache[(k, ax)] = np.asarray(Image.fromarray(r).resize((nx, ny), Image.BILINEAR)) \
                if r.shape[1] != nx else r
        return cache[(k, ax)]

    levels = list(range(enc.n_levels))[-(cols * cols // 2 - 1):]
    tiles, labels, raw = [], [], []
    try:
        for ax in ("x", "y"):
            b0 = levels[0] - 1
            a = upto(b0, ax)
            tiles.append(_cm(a / max(1e-9, np.abs(a).max()) * .5 + .5))
            labels.append(f"0..{b0} {ax}")
            for l in levels:
                raw.append((len(tiles), upto(l, ax) - upto(l - 1, ax)))
                tiles.append(None)
                labels.append(f"L{l}/{enc.resolutions[l][2]}t {ax}")
    finally:
        enc.set_level_window(float(enc.n_levels))
    vmax = float(np.percentile(np.abs(np.concatenate([d.ravel() for _, d in raw])), 99)) or 1e-6
    for i, d in raw:
        tiles[i] = _signed(d, vmax)
    labels[0] += f"  +-{vmax:.3g}"
    return _sheet(tiles, labels, path, cols)


@torch.no_grad()
def level_map(op, path, t: float = 0.0, block_px: int = 64, sub: int = 4, thresh: float = 0.08,
              channel: int = 0):
    """Per block, the FINEST level still contributing there -- and how many pixels its cell spans.

    Saturates at 16 levels; see the module docstring. Reported as a picture AND as the per-block
    numbers, because the number is what a gate would read.
    """
    enc, shape = op.grid, op._shape
    h, w = shape[0], shape[1]
    hs, ws = max(8, h // sub), max(8, w // sub)
    bs = max(2, block_px // sub)
    prev, deltas = None, []
    try:
        for k in range(enc.n_levels + 1):
            enc.set_level_window(float(k))
            out = op.sample((hs, ws), t)[channel]
            if prev is not None:
                deltas.append((out - prev).abs())
            prev = out
    finally:
        enc.set_level_window(float(enc.n_levels))
    D = torch.stack(deltas)
    nb = torch.nn.functional.avg_pool2d(D[None], bs, stride=bs, ceil_mode=True)[0]
    peak = nb.amax(0, keepdim=True)
    alive = peak > 0.02 * float(nb.max())
    sig = (nb > thresh * peak) & alive
    lev = torch.arange(nb.shape[0], device=nb.device)[:, None, None].expand_as(nb)
    dom = torch.where(sig, lev, torch.zeros_like(lev)).amax(0)
    dom = torch.where(sig.any(0), dom, nb.argmax(0)).cpu().numpy()
    rgb = _cm(dom / max(1, enc.n_levels - 1))
    im = Image.fromarray(np.asarray(Image.fromarray(rgb).resize((ws * 4, hs * 4), Image.NEAREST)))
    ImageDraw.Draw(im).text((3, 2), f"finest level per {block_px} px block  (0..{enc.n_levels-1})",
                            fill=(255, 255, 255))
    im.save(path)
    cell_px = [round(w / enc.resolutions[int(l)][0], 2) for l in dom.ravel()]
    return {"path": path, "level_mean": float(dom.mean()), "level_max": int(dom.max()),
            "cell_px_median": float(np.median(cell_px)), "n_blocks": int(dom.size)}
