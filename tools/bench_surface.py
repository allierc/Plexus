#!/usr/bin/env python
"""Four ways to turn MPM points into a continuous, realistic picture -- timed side by side.

    PYTHONPATH=src python tools/bench_surface.py <run_dir> [--frame 120] [--px 900]

Writes one PNG per route into `<run_dir>` plus `surface_bench.json`. Nothing is simulated: the
frame comes from the run's own trajectory.

THE FOUR ROUTES, and where each forms the surface.

  points        the page's renderer today: one lit sphere imposter per particle, formed nowhere.
                A body reads as a cloud of balls, and that is the thing to improve on.

  screen_space  VAN DER LAAN's curvature-flow route (I3D 2009): splat depth, smooth the DEPTH
                BUFFER, take the normal from the smoothed depth, shade it. The surface exists only
                in the frame that is being drawn -- no grid, no mesh, no polygonisation artefacts,
                and the cost is pixels rather than particles. It cannot cast a shadow from the
                fluid onto itself, because there is no geometry to cast one.

  gauss_splat   the MPM -> GAUSSIAN route (PhysGaussian, CVPR 2024): every material point is
                already an anisotropic ellipsoid -- its covariance is F F^T scaled by the rest
                size, and the solver updates F for free -- so the splats need no training. Here
                the trajectory stores positions only, so the covariance is isotropic and the
                comparison is a LOWER bound on what the live path would show.

  levelset_mc   the film route: splat into a density grid, smooth it, march cubes, hand the mesh
                to a renderer. It is the only one that produces real geometry -- shadows,
                refraction, a mesh you can export -- and the only one whose cost is the grid.

All four read the same particles, the same camera and the same per-body colours.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))


# --------------------------------------------------------------------------- the camera
class Cam:
    """A look-at perspective camera, shared by every route so the pictures are comparable."""

    def __init__(self, box, px, azim=35.0, elev=14.0, dist=1.5, fovy=0.6):
        lo, hi = box
        self.centre = 0.5 * (lo + hi)
        self.span = float(np.max(hi - lo))
        a, e = np.radians(azim), np.radians(elev)
        d = np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])
        self.eye = self.centre + d * (dist * self.span)
        self.fwd = -d / np.linalg.norm(d)
        up = np.array([0.0, 1.0, 0.0])
        self.right = np.cross(self.fwd, up); self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.fwd)
        self.px = px
        self.fovy = fovy
        self.f = 0.5 * px[1] / np.tan(0.5 * fovy)              # focal length in pixels

    def project(self, pos):
        """(x_px, y_px, depth) of every point, in the image frame."""
        import torch
        p = pos - torch.as_tensor(self.eye, dtype=pos.dtype, device=pos.device)
        z = p @ torch.as_tensor(self.fwd, dtype=pos.dtype, device=pos.device)
        x = p @ torch.as_tensor(self.right, dtype=pos.dtype, device=pos.device)
        y = p @ torch.as_tensor(self.up, dtype=pos.dtype, device=pos.device)
        zc = z.clamp(min=1e-6)
        return (self.px[0] * 0.5 + self.f * x / zc), (self.px[1] * 0.5 - self.f * y / zc), z


def _splat_zbuffer(cam, xs, ys, zs, rad_px, W, H, keep_index=True):
    """Nearest-surface depth per pixel (and which particle it was), by scattering discs."""
    import torch
    dev = xs.device
    Z = torch.full((H * W,), float("inf"), device=dev)
    I = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    k = int(rad_px.max().item()) + 1
    xi, yi = xs.round().long(), ys.round().long()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            r2 = float(dx * dx + dy * dy)
            m = (r2 <= (rad_px * rad_px))
            if not bool(m.any()):
                continue
            px, py = xi[m] + dx, yi[m] + dy
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            idx = (py[ok] * W + px[ok])
            # the sphere's own surface, not its centre: z - sqrt(r^2 - d^2) in world units
            rr = rad_px[m][ok]
            dz = torch.sqrt((rr * rr - r2).clamp(min=0.0)) * zs[m][ok] / cam.f
            znew = zs[m][ok] - dz
            Z.scatter_reduce_(0, idx, znew, reduce="amin", include_self=True)
            if keep_index:
                hit = (Z[idx] >= znew - 1e-9)
                I[idx[hit]] = torch.nonzero(m, as_tuple=False).flatten()[ok][hit]
    return Z.view(H, W), I.view(H, W)


def _blur(img, k=5, it=1):
    import torch
    import torch.nn.functional as F
    x = torch.linspace(-2.0, 2.0, k, device=img.device)
    g = torch.exp(-0.5 * x * x); g = g / g.sum()
    out = img[None, None]
    pad = k // 2
    for _ in range(it):
        out = F.conv2d(F.pad(out, (pad, pad, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
        out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="replicate"), g.view(1, 1, k, 1))
    return out[0, 0]


def _shade(depth, col, thick, cam, bg=(0.0, 0.0, 0.0)):
    """Normals from the depth image, then diffuse + Blinn specular + a fresnel sky and a
    Beer-Lambert tint by thickness -- the shading half of the screen-space route."""
    import torch
    H, W = depth.shape
    finite = torch.isfinite(depth)
    z = torch.where(finite, depth, torch.zeros_like(depth))
    dzdx = torch.zeros_like(z); dzdy = torch.zeros_like(z)
    dzdx[:, 1:-1] = 0.5 * (z[:, 2:] - z[:, :-2])
    dzdy[1:-1, :] = 0.5 * (z[2:, :] - z[:-2, :])
    sx = z / cam.f                                             # world units per pixel at that depth
    n = torch.stack([-dzdx / sx.clamp(min=1e-9), dzdy / sx.clamp(min=1e-9), torch.ones_like(z)], -1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    L = torch.tensor([0.35, 0.75, 0.55], device=z.device); L = L / L.norm()
    V = torch.tensor([0.0, 0.0, 1.0], device=z.device)
    ndl = (n * L).sum(-1).clamp(min=0.0)
    hvec = (L + V); hvec = hvec / hvec.norm()
    spec = ((n * hvec).sum(-1).clamp(min=0.0)) ** 48.0
    fres = 0.04 + 0.96 * (1.0 - (n * V).sum(-1).clamp(min=0.0)) ** 5.0
    sky = torch.tensor([0.45, 0.62, 0.95], device=z.device)
    t = (thick / thick[finite].median().clamp(min=1e-6)) if finite.any() else thick
    absorb = torch.exp(-0.35 * t)[..., None]
    rgb = col * (0.25 + 0.85 * ndl)[..., None] * absorb + 0.9 * spec[..., None] \
        + fres[..., None] * sky * 0.35
    out = torch.where(finite[..., None], rgb.clamp(0, 1), torch.tensor(bg, device=z.device))
    return (out * 255).to(torch.uint8).cpu().numpy()


# --------------------------------------------------------------------------- the four routes
def route_screen_space(pos, col, cam, radius, smooth=6):
    import torch
    W, H = cam.px
    xs, ys, zs = cam.project(pos)
    rad = (radius * cam.f / zs.clamp(min=1e-6)).clamp(1.0, 24.0)
    Z, I = _splat_zbuffer(cam, xs, ys, zs, rad, W, H)
    Zs = _blur(torch.where(torch.isfinite(Z), Z, torch.zeros_like(Z)), k=9, it=smooth)
    Zs = torch.where(torch.isfinite(Z), Zs, torch.full_like(Zs, float("inf")))
    C = torch.zeros(H, W, 3, device=pos.device)
    hit = I >= 0
    C[hit] = col[I[hit]]
    C = torch.stack([_blur(C[..., k], k=7, it=2) for k in range(3)], -1)
    thick = torch.zeros(H * W, device=pos.device)
    xi, yi = xs.round().long().clamp(0, W - 1), ys.round().long().clamp(0, H - 1)
    thick.scatter_add_(0, yi * W + xi, torch.ones_like(zs))
    thick = _blur(thick.view(H, W), k=9, it=3)
    return _shade(Zs, C, thick, cam)


def route_gauss_splat(pos, col, cam, radius, sigma_mul=0.6):
    """Isotropic 3-D Gaussians rasterised with a depth-weighted accumulation.

    THE COVARIANCE IS THE PART THIS CANNOT SHOW. PhysGaussian's ellipsoid is F F^T of the material
    point; a trajectory stores positions only, so every splat here is a ball. What is measured is
    the RASTERISER -- projection, per-pixel weight, compositing -- which is what the live path
    would also pay."""
    import torch
    W, H = cam.px
    xs, ys, zs = cam.project(pos)
    sig = (radius * sigma_mul * cam.f / zs.clamp(min=1e-6)).clamp(0.8, 20.0)
    Z, _ = _splat_zbuffer(cam, xs, ys, zs, sig * 2.0, W, H, keep_index=False)
    acc = torch.zeros(H * W, 3, device=pos.device)
    wsum = torch.zeros(H * W, device=pos.device)
    k = int((sig * 2.0).max().item()) + 1
    xi, yi = xs.round().long(), ys.round().long()
    zf = Z.view(-1)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            r2 = float(dx * dx + dy * dy)
            w = torch.exp(-0.5 * r2 / (sig * sig))
            m = w > 0.02
            if not bool(m.any()):
                continue
            px, py = xi[m] + dx, yi[m] + dy
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            idx = py[ok] * W + px[ok]
            # front-to-back without a sort: weight each splat by how close it is to the nearest
            # surface at that pixel, so what is behind contributes almost nothing
            dz = (zs[m][ok] - zf[idx]).clamp(min=0.0)
            wz = w[m][ok] * torch.exp(-dz / (3.0 * float(radius)))
            acc.index_add_(0, idx, col[m][ok] * wz[:, None])
            wsum.index_add_(0, idx, wz)
    C = (acc / wsum.clamp(min=1e-9)[:, None]).view(H, W, 3)
    lit = _shade(_blur(torch.where(torch.isfinite(Z), Z, torch.zeros_like(Z)), k=5, it=1),
                 C, wsum.view(H, W) * 0.0 + 1.0, cam)
    return np.where((wsum.view(H, W, 1) > 1e-6).cpu().numpy(), lit, 0)


_WP_READY = {"ok": False}


def _zhu_warp(pos_np, lo, dims, dx, h):
    """Zhu-Bridson's two grid fields (sum of weights, weighted position sum) with a warp hash grid.

    ONE THREAD PER GRID CELL, asking the hash grid which particles are within `h` -- instead of the
    torch route's (2k+1)^3 passes over every particle, which is where its half second went. The
    grid is built once per frame and the query is what an SPH neighbour search costs anyway.
    """
    import warp as wp
    if not _WP_READY["ok"]:
        wp.init(); _WP_READY["ok"] = True

    @wp.kernel
    def _acc(grid: wp.uint64, pts: wp.array(dtype=wp.vec3), lo: wp.vec3, dx: float, h: float,
             nx: int, ny: int, nz: int,
             wsum: wp.array(dtype=float), xbar: wp.array(dtype=wp.vec3)):
        c = wp.tid()
        k = c % nz
        j = (c // nz) % ny
        i = c // (ny * nz)
        x = lo + wp.vec3(float(i) * dx, float(j) * dx, float(k) * dx)
        query = wp.hash_grid_query(grid, x, h)
        index = int(0)
        w_tot = float(0.0)
        xb = wp.vec3(0.0, 0.0, 0.0)
        while wp.hash_grid_query_next(query, index):
            d = wp.length(x - pts[index]) / h
            if d < 1.0:
                w = (1.0 - d * d) * (1.0 - d * d) * (1.0 - d * d)
                w_tot += w
                xb += pts[index] * w
        wsum[c] = w_tot
        xbar[c] = xb

    nx, ny, nz = (int(v) for v in dims)
    dev = "cuda:0" if wp.get_cuda_device_count() else "cpu"
    pts = wp.array(np.ascontiguousarray(pos_np, np.float32), dtype=wp.vec3, device=dev)
    grid = wp.HashGrid(dim_x=64, dim_y=64, dim_z=64, device=dev)
    grid.build(points=pts, radius=float(h))
    n = nx * ny * nz
    wsum = wp.zeros(n, dtype=float, device=dev)
    xbar = wp.zeros(n, dtype=wp.vec3, device=dev)
    wp.launch(_acc, dim=n, inputs=[grid.id, pts, wp.vec3(*[float(v) for v in lo]), float(dx), float(h),
                                   nx, ny, nz, wsum, xbar], device=dev)
    wp.synchronize()
    return wsum.numpy().reshape(nx, ny, nz), xbar.numpy().reshape(nx, ny, nz, 3)


def route_levelset(pos, col, cam, radius, ngrid=0, iso=0.32, smooth=2, method="zhu",
                   cell_mul=1.0, kernel_mul=3.0, taubin=0, backend="warp"):
    """Points -> an implicit surface -> marching cubes -> a shaded mesh.

    TWO IMPLICIT SURFACES, and the difference is the whole quality question.

      `blobby`  phi(x) = sum_i W(|x - x_i|), thresholded. The classic metaball sum: it bulges
                where particles pile up and dips between them, so a FLAT face comes out lumpy and
                the only cure is to blur -- which rounds the corners off with it. That is what the
                first version of this route did, and why it read as blurry.

      `zhu`     ZHU & BRIDSON (SIGGRAPH 2005), the film default: phi(x) = |x - xbar(x)| - rbar(x)
                where xbar is the WEIGHTED AVERAGE PARTICLE POSITION near x and rbar the weighted
                average radius. It is a distance field to a smoothed point set rather than a sum of
                bumps, so a flat wall of particles gives a flat surface and an edge stays an edge:
                no blur is needed to hide the sampling, so none is applied to the corners.

    The grid is sized from the PARTICLE SPACING (`cell_mul` x spacing), not from a fixed count: a
    grid finer than the spacing costs cubically and resolves nothing that is there. `kernel_mul` is
    the support radius in spacings -- 2 is the usual choice, under 1.5 the surface breaks up.

    Returns (image, seconds to build the mesh, seconds to render it, triangle count).
    """
    import torch
    import torch.nn.functional as F
    import pyvista as pv
    spacing = radius / 0.62                                     # the caller derives radius from it
    dx = float(cell_mul * spacing) if not ngrid else float((pos.max(0).values - pos.min(0).values).max()) / ngrid
    h = float(kernel_mul * spacing)                             # kernel support
    lo = pos.min(0).values - 2 * h
    hi = pos.max(0).values + 2 * h
    dims = ((hi - lo) / dx).long() + 3
    W_, H_, D_ = (int(x) for x in dims)
    t0 = time.time()
    n = W_ * H_ * D_
    if method == "zhu" and backend == "warp":
        ws, xb = _zhu_warp(pos.detach().cpu().numpy(), lo.cpu().numpy(), (W_, H_, D_), dx, h)
        wsum = torch.as_tensor(ws.reshape(-1), device=pos.device)
        xbar = torch.as_tensor(xb.reshape(-1, 3), device=pos.device)
        q = None
    else:
        wsum = torch.zeros(n, device=pos.device)
        xbar = torch.zeros(n, 3, device=pos.device)
        q = (pos - lo) / dx
    base = q.floor().long() if q is not None else None
    k = int(np.ceil(h / dx))
    off = torch.arange(-k, k + 1, device=pos.device) if q is not None else []
    for ox in (off.tolist() if q is not None else []):
        for oy in off.tolist():
            for oz in off.tolist():
                gi = base + torch.tensor([ox, oy, oz], device=pos.device)
                gi = gi.clamp(torch.zeros(3, dtype=torch.long, device=pos.device),
                              torch.tensor([W_ - 1, H_ - 1, D_ - 1], device=pos.device))
                gx = lo + gi.float() * dx                       # that cell's world position
                r = (gx - pos).norm(dim=1) / h
                w = (1.0 - r * r).clamp(min=0.0) ** 3           # smooth, compact, cheap
                m = w > 0
                if not bool(m.any()):
                    continue
                idx = gi[:, 0] * H_ * D_ + gi[:, 1] * D_ + gi[:, 2]
                wsum.index_add_(0, idx[m], w[m])
                xbar.index_add_(0, idx[m], pos[m] * w[m][:, None])
    if method == "zhu":
        gxi = torch.stack(torch.meshgrid(torch.arange(W_, device=pos.device),
                                         torch.arange(H_, device=pos.device),
                                         torch.arange(D_, device=pos.device), indexing="ij"), -1)
        gx = lo + gxi.float() * dx
        xb = (xbar / wsum.clamp(min=1e-12)[:, None]).view(W_, H_, D_, 3)
        phi = (gx - xb).norm(dim=-1) - float(radius * 1.35)
        phi = torch.where(wsum.view(W_, H_, D_) > 1e-9, phi, torch.full_like(phi, 3 * dx))
        vol, level = phi, 0.0
        # ONE light pass, on the DISTANCE field, not on a density: it moves the surface by a
        # fraction of a cell instead of eating the corners.
        if smooth:
            kx = torch.tensor([1.0, 2.0, 1.0], device=vol.device); kx = kx / kx.sum()
            for _ in range(max(1, smooth // 2)):
                for ax in range(3):
                    shape = [1, 1, 1, 1, 1]; shape[2 + ax] = 3
                    vol = F.conv3d(F.pad(vol[None, None], (1, 1, 1, 1, 1, 1), mode="replicate"),
                                   kx.view(*shape))[0, 0]
    else:
        vol = wsum.view(W_, H_, D_)
        if smooth:
            kx = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=vol.device); kx = kx / kx.sum()
            for _ in range(smooth):
                for ax in range(3):
                    shape = [1, 1, 1, 1, 1]; shape[2 + ax] = 5
                    vol = F.conv3d(F.pad(vol[None, None], (2, 2, 2, 2, 2, 2), mode="replicate"),
                                   kx.view(*shape))[0, 0]
        level = iso * float(vol.max())
    t_field = time.time() - t0
    v = vol.detach().cpu().numpy()
    grid = pv.ImageData(dimensions=v.shape, spacing=(dx, dx, dx), origin=tuple(lo.cpu().numpy()))
    grid.point_data["d"] = v.flatten(order="F")
    surf = grid.contour([level], scalars="d")
    if taubin and surf.n_points:
        surf = surf.smooth_taubin(n_iter=int(taubin), pass_band=0.1)
    surf = surf.compute_normals(consistent_normals=False, auto_orient_normals=False)
    t_build = time.time() - t0
    route_levelset.last = dict(field=t_field, mc=t_build - t_field, cells=(W_, H_, D_), dx=dx)
    t0 = time.time()
    p = pv.Plotter(off_screen=True, window_size=cam.px, border=False)
    p.add_mesh(surf, color=(0.55, 0.72, 0.95), smooth_shading=True, specular=0.6,
               specular_power=30, pbr=False, ambient=0.25, diffuse=0.8)
    p.set_background("black")
    p.camera.focal_point = tuple(float(x) for x in cam.centre)
    p.camera.position = tuple(float(x) for x in cam.eye)
    p.camera.up = (0.0, 1.0, 0.0)
    p.camera.view_angle = np.degrees(cam.fovy)
    p.render(); img = np.asarray(p.screenshot(return_img=True))[..., :3]
    t_render = time.time() - t0
    p.close()
    return img, t_build, t_render, int(surf.n_cells)


def route_points(pos, col, cam, radius, lit=True):
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=cam.px, border=False)
    cloud = pv.PolyData(pos.cpu().numpy().astype(np.float32))
    cloud["rgb"] = (np.clip(col.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    px = max(2.0, 2.0 * radius * cam.f / (1.5 * cam.span))
    p.add_mesh(cloud, scalars="rgb", rgb=True, render_points_as_spheres=True,
               lighting=lit, ambient=0.3 if lit else 1.0, diffuse=0.7 if lit else 0.0,
               specular=0.3 if lit else 0.0, specular_power=20, point_size=px)
    p.set_background("black")
    p.camera.focal_point = tuple(float(x) for x in cam.centre)
    p.camera.position = tuple(float(x) for x in cam.eye)
    p.camera.up = (0.0, 1.0, 0.0)
    p.camera.view_angle = np.degrees(cam.fovy)
    p.render()
    t0 = time.time()
    for _ in range(4):
        p.render(); img = np.asarray(p.screenshot(return_img=True))[..., :3]
    dt = (time.time() - t0) / 4
    p.close()
    return img, dt


def main():
    import torch
    import imageio.v3 as iio
    import render_anari as RA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--px", type=int, default=900)
    ap.add_argument("--azim", type=float, default=35.0)
    ap.add_argument("--elev", type=float, default=14.0)
    ap.add_argument("--dist", type=float, default=1.5)
    ap.add_argument("--ngrid", type=int, default=192, help="grid for the blobby reference only")
    ap.add_argument("--cell", type=float, default=1.0, help="Zhu-Bridson cell size, in particle spacings")
    ap.add_argument("--kernel", type=float, default=2.5, help="kernel support, in particle spacings")
    ap.add_argument("--smooth", type=int, default=2)
    ap.add_argument("--taubin", type=int, default=0, help="Taubin mesh smoothing iterations (shape-preserving)")
    ap.add_argument("--device", default="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") else "cpu")
    ap.add_argument("--repeat", type=int, default=4)
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"

    pos_np, box = RA.read_frame(a.run_dir, a.frame)
    col_np = RA.body_colours(a.run_dir, len(pos_np))
    if col_np is None or len(col_np) != len(pos_np):
        col_np = RA.colours(pos_np, *box)
    pos = torch.as_tensor(pos_np, device=dev)
    col = torch.as_tensor(np.asarray(col_np, np.float32), device=dev)
    cam = Cam(box, (a.px, a.px), a.azim, a.elev, a.dist)
    # the particle spacing sets every radius: a surface needs the primitives to touch
    from scipy.spatial import cKDTree
    q = pos_np[:: max(1, len(pos_np) // 20000)]
    sp = float(np.median(cKDTree(q).query(q, k=2)[0][:, 1]))
    radius = 0.62 * sp
    print(f"[data] {len(pos_np):,} particles, spacing {sp * 1000:.2f} mm, radius {radius * 1000:.2f} mm", flush=True)

    out, times = {}, {}
    img, dt = route_points(pos, col, cam, radius, lit=True)
    out["points"], times["points"] = img, dt
    for name, fn in (("screen_space", route_screen_space), ("gauss_splat", route_gauss_splat)):
        fn(pos, col, cam, radius)                               # warm
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.repeat):
            img = fn(pos, col, cam, radius)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        out[name], times[name] = img, (time.time() - t0) / a.repeat
    for nm, kw in (("levelset_blobby", dict(method="blobby", ngrid=a.ngrid, smooth=2)),
                   ("levelset_torch", dict(method="zhu", cell_mul=a.cell, kernel_mul=a.kernel,
                                           smooth=a.smooth, taubin=a.taubin, backend="torch")),
                   ("levelset_mc", dict(method="zhu", cell_mul=a.cell, kernel_mul=a.kernel,
                                        smooth=a.smooth, taubin=a.taubin, backend="warp"))):
        if kw.get("backend") == "warp":
            route_levelset(pos, col, cam, radius, **kw)         # warp compiles its kernel once
        img, tb, tr, ntri = route_levelset(pos, col, cam, radius, **kw)
        out[nm], times[nm] = img, tb + tr
        times[nm + "_build"], times[nm + "_render"], times[nm + "_triangles"] = tb, tr, ntri
        _last = getattr(route_levelset, "last", {})
        times[nm + "_field"], times[nm + "_mc"] = _last.get("field"), _last.get("mc")
        times[nm + "_grid"] = _last.get("cells")
    try:                                                        # the ray tracer, for reference
        sc = RA.AnariScene(pos_np, np.asarray(col_np, np.float32), box, (a.px, a.px), radius=radius, spp=1)
        sc.look(a.azim, a.elev, a.dist); sc.render()
        t0 = time.time(); img = sc.render(); times["anari_spheres"] = time.time() - t0
        out["anari_spheres"] = img
    except Exception as e:                                      # noqa: BLE001
        print(f"[bench] anari unavailable: {type(e).__name__}: {e}", flush=True)

    for k, v in out.items():
        iio.imwrite(os.path.join(a.run_dir, f"surf_{k}.png"), np.asarray(v, np.uint8))
    times["particles"] = int(len(pos_np))
    times["pixels"] = a.px * a.px
    json.dump(times, open(os.path.join(a.run_dir, "surface_bench.json"), "w"), indent=1)
    print(f"[bench] {len(pos_np):,} particles at {a.px}x{a.px} on {dev}")
    for k in ("points", "screen_space", "gauss_splat", "levelset_blobby", "levelset_torch",
              "levelset_mc", "anari_spheres"):
        if k in times:
            extra = ""
            if k.startswith("levelset"):
                extra = (f"  (build {times[k + '_build'] * 1000:.0f} ms, "
                         f"render {times[k + '_render'] * 1000:.0f} ms, "
                         f"{times[k + '_triangles']:,} triangles"
                         + (f"; field {times[k + '_field'] * 1000:.0f} ms + marching cubes "
                            f"{times[k + '_mc'] * 1000:.0f} ms on a {'x'.join(str(c) for c in times[k + '_grid'])} grid"
                            if times.get(k + "_field") else "") + ")")
            print(f"[bench] {k:14s} {times[k] * 1000:7.0f} ms/frame  ({1 / times[k]:6.1f} fps){extra}")


if __name__ == "__main__":
    main()
