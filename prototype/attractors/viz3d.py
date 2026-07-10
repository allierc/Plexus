"""viz3d -- a from-scratch 3D **additive-glow** point-cloud renderer for the attractor clouds.

Not the Plexus `plot.py` splat (that alpha-composites tight sprites, src-over, so the cloud
reads as a *solid* object). A strange attractor is emissive gas -- a fractal filament of
light -- so this renderer accumulates every point ADDITIVELY into a density buffer, blurs it
to a soft glow, and tone-maps `1 - exp(-d/ref)` so dense filaments saturate to a white-hot
core while thin wisps stay in the attractor's hue. There is no occlusion (light adds up), which
is exactly right for luminous gas and is what gives the neon look of the reference plate.

3D is carried entirely by the CAMERA: points are rotated into a yaw(azim)+pitch(elev) camera
frame (world +z is up), the camera slowly ORBITS over the movie (the strongest depth cue), and
depth both fogs far points toward black and weights near points brighter. A short temporal
PERSISTENCE (`trail_decay`) leaves silky motion trails so the moving swarm reads as continuous
filaments rather than a fizz of dots.

Rasterisation runs in torch on the GPU (scatter-add + separable Gaussian conv), so a
120k-point cloud renders a ~700-frame movie in well under a minute. Only numpy / torch /
imageio-ffmpeg / PIL are used -- no matplotlib, no Plexus imports; the module stands alone.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  camera + rasterisation (torch, on-device)
# --------------------------------------------------------------------------- #
def _rot_camera(P, azim, elev):
    """Rotate centred world points [N,3] into the camera frame: yaw by `azim` about world
    up (+z), then pitch by `elev` (0 = top-down, ~pi/2 = side-on). Returns
    (screen_x, screen_y, depth); depth increases INTO the screen (away from camera)."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ca, sa = np.cos(azim), np.sin(azim)
    x1 = ca * x - sa * y
    y1 = sa * x + ca * y
    ce, se = np.cos(elev), np.sin(elev)
    y2 = ce * y1 - se * z
    depth = se * y1 + ce * z
    return x1, y2, depth


def _gauss_kernel(sigma, device):
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return (k / k.sum()).view(1, 1, -1)


def _blur(img, k):
    """Separable Gaussian blur of a [res,res] image with 1D kernel `k` ([1,1,L])."""
    r = k.shape[-1] // 2
    x = img[None, None]                                   # [1,1,H,W]
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    x = F.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))
    return x[0, 0]


def _accumulate(sx, sy, w, res):
    """Scatter-add per-point weights `w` into a raw [res,res] density buffer at integer pixels
    (sx, sy); points outside the frame are dropped (no edge smear)."""
    gx = sx.round().long(); gy = sy.round().long()
    m = (gx >= 0) & (gx < res) & (gy >= 0) & (gy < res)
    dens = torch.zeros(res * res, device=sx.device, dtype=torch.float32)
    flat = (gy[m] * res + gx[m])
    dens.index_put_((flat,), w[m], accumulate=True)
    return dens.view(res, res)


def _frame_raw(P, azim, elev, span, res, point_weight):
    """One frame's raw (un-blurred) depth-weighted density [res,res] from centred points `P`.
    Depth both weights near points brighter (fog) and is returned for the caller."""
    sx, sy, depth = _rot_camera(P, azim, elev)
    px = (sx / (2 * span) + 0.5) * res                    # world -> pixel (isotropic half-width span)
    py = (sy / (2 * span) + 0.5) * res
    dn = (0.5 + 0.5 * depth / span).clamp(0.0, 1.0)       # 0 = far, 1 = near
    w = point_weight * (0.22 + 0.78 * dn)                 # strong fog: near filaments much brighter
    return _accumulate(px, py, w, res)


def _tonemap(dens_raw, color, ref, kcore, khalo, gamma, core_p, core_amt, bloom):
    """Raw density -> RGB [res,res,3] on black with a BLOOM model that keeps the fractal
    filaments crisp: a sharp colored core (structure) + a soft glow halo (neon), and only the
    densest peaks whiten to a hot core. `ref` is a high density quantile, so the bulk of the
    cloud lands in the graded (un-saturated) band and internal striations survive."""
    core = _blur(dens_raw, kcore)                         # crisp: reveals the layered sheets
    halo = _blur(dens_raw, khalo)                         # soft: the surrounding glow
    Ic = (1.0 - torch.exp(-core / ref)).clamp(0.0, 1.0) ** gamma            # [res,res]
    Ih = (1.0 - torch.exp(-halo / (ref * 1.7))).clamp(0.0, 1.0) ** (gamma * 1.15)
    col = color.view(1, 1, 3)
    img = col * Ic[..., None] + col * (bloom * Ih[..., None])   # hue from crisp core + colored bloom
    hot = (Ic ** core_p)[..., None]                       # only genuine density peaks whiten
    img = img + (1.0 - col) * hot * core_amt
    return img.clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
#  driver: trajectory [T,N,3] -> movie.mp4 + strip.png
# --------------------------------------------------------------------------- #
def _robust_frame(traj, view_q=0.985):
    """A fixed view centre + half-span from the whole trajectory, so the attractor sits still in
    the frame while the camera orbits. The centre is the per-axis MEDIAN and the span is the
    `view_q` quantile of the radius -- both robust to escapees (e.g. Rabinovich-Fabrikant leaks a
    fraction of its cloud to infinity; a low `view_q` frames the bounded core and the render then
    simply drops the out-of-frame escapees)."""
    T = traj.shape[0]
    idx = np.linspace(0, T - 1, min(T, 40)).astype(int)
    sample = traj[idx].reshape(-1, 3)
    center = np.median(sample, axis=0)
    r = np.linalg.norm(sample - center[None], axis=1)
    span = float(np.quantile(r, view_q)) * 1.06 + 1e-6
    return center.astype(np.float32), span


def _calibrate_ref(P_all, frames, center, span, res, point_weight, kcore, q):
    """A fixed brightness reference (so the movie doesn't flicker): the median over a few
    sample frames of the `q`-quantile of the crisp density. A HIGH `q` keeps most of the cloud
    in the graded band (so internal striations survive) and lets only the top (1-q) whiten."""
    vals = []
    for i in frames:
        dens = _blur(_frame_raw(P_all[i] - center, 0.6, 1.15, span, res, point_weight), kcore)
        nz = dens[dens > 0]
        if nz.numel():
            vals.append(torch.quantile(nz, q).item())
    return max(float(np.median(vals)) if vals else 1.0, 1e-6)


def render(traj, out_dir, name, color, style=None, device="cuda:0", seconds=20.0,
           max_frames=640):
    """Render a trajectory `traj` [T,N,3] to `out_dir`/movie.mp4 (+ strip.png, fig_final.png).

    color : [3] base hue (0..1). style : dict of render knobs (res, splat_core, splat_halo,
    ref_quantile, gamma, core_power, core_amt, bloom, trail_decay, point_weight,
    camera_azim/elev/turns/elev_drift). Returns the movie path."""
    s = dict(res=900, splat_core=1.0, splat_halo=3.4, ref_quantile=0.93, gamma=0.72,
             core_power=3.5, core_amt=0.55, bloom=0.5, trail_decay=0.45, point_weight=1.0,
             view_quantile=0.985, camera_azim=0.6, camera_elev=1.12, camera_turns=0.35,
             camera_elev_drift=0.10)
    s.update(style or {})
    # tolerate legacy knob names from older specs
    s.setdefault("splat_core", s.get("splat_sigma", 1.0))
    os.makedirs(out_dir, exist_ok=True)
    res = int(s["res"]) & ~1                               # even (yuv420p)
    dev = device if (isinstance(device, str) and device.startswith("cuda")
                     and torch.cuda.is_available()) else "cpu"

    traj = np.asarray(traj, np.float32)
    T = traj.shape[0]
    center_np, span = _robust_frame(traj, float(s["view_quantile"]))
    center = torch.tensor(center_np, device=dev)
    color_t = torch.tensor(color, device=dev, dtype=torch.float32)
    kcore = _gauss_kernel(float(s["splat_core"]), dev)
    khalo = _gauss_kernel(float(s["splat_halo"]), dev)
    P_all = torch.tensor(traj, device=dev)                # [T,N,3]

    stride = max(1, -(-T // max_frames))
    out_idx = list(range(0, T, stride))
    n_out = len(out_idx)
    fps = max(1, round(n_out / seconds))
    ref = _calibrate_ref(P_all, np.linspace(int(0.35 * T), T - 1, 6).astype(int),
                         center, span, res, float(s["point_weight"]), kcore,
                         float(s["ref_quantile"]))

    def _rgb(i, acc):
        """Blended-with-trails RGB uint8 for output frame index position `i` (into out_idx)."""
        t = out_idx[i]
        frac = i / max(1, n_out - 1)
        azim = float(s["camera_azim"]) + float(s["camera_turns"]) * 2 * np.pi * frac
        elev = float(s["camera_elev"]) + float(s["camera_elev_drift"]) * np.sin(np.pi * frac)
        dens = _frame_raw(P_all[t] - center, azim, elev, span, res, float(s["point_weight"]))
        acc = dens if acc is None else (float(s["trail_decay"]) * acc + dens)
        img = _tonemap(acc, color_t, ref, kcore, khalo, float(s["gamma"]),
                       float(s["core_power"]), float(s["core_amt"]), float(s["bloom"]))
        u8 = (img * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
        return u8, acc

    # --- movie (raw frames -> ffmpeg, H.264 CRF) --------------------------- #
    import imageio_ffmpeg
    out_path = os.path.join(out_dir, "movie.mp4")
    writer = imageio_ffmpeg.write_frames(
        out_path, size=(res, res), fps=fps, macro_block_size=1, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-crf", "16", "-preset", "slow"])
    writer.send(None)
    acc = None
    strip_targets = {int(round(f * (n_out - 1))): f for f in (0.03, 0.12, 0.30, 0.60, 1.0)}
    strip = {}
    for i in range(n_out):
        u8, acc = _rgb(i, acc)
        writer.send(np.ascontiguousarray(u8))
        if i in strip_targets:
            strip[strip_targets[i]] = u8.copy()
    writer.close()

    # --- development strip + final still (PIL, no matplotlib) -------------- #
    _save_strip(strip, out_dir, name)
    if strip:
        from PIL import Image
        Image.fromarray(strip[max(strip)]).save(os.path.join(out_dir, "fig_final.png"))
    print(f"[viz3d] {name}: {n_out} frames @ {fps}fps -> {out_path}", flush=True)
    return out_path


def _save_strip(strip, out_dir, name):
    """Tile the 5 development snapshots horizontally with % labels -> strip.png."""
    if not strip:
        return
    from PIL import Image, ImageDraw
    fracs = sorted(strip)
    tile = 300
    imgs = [Image.fromarray(strip[f]).resize((tile, tile)) for f in fracs]
    sheet = Image.new("RGB", (tile * len(imgs), tile), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for j, (f, im) in enumerate(zip(fracs, imgs)):
        sheet.paste(im, (j * tile, 0))
        dr.text((j * tile + 6, 6), f"{int(f * 100)}%", fill=(235, 235, 235))
    dr.text((6, tile - 16), name, fill=(180, 180, 180))
    sheet.save(os.path.join(out_dir, "strip.png"))
