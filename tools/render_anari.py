#!/usr/bin/env python
"""Ray-trace a recorded run's particles with ANARI, and time it against the VTK point path.

    PYTHONPATH=src python tools/render_anari.py <run_dir> --bench
    PYTHONPATH=src python tools/render_anari.py <run_dir> --turntable 240 --out movie_anari.mp4

WHY ANARI AND NOT OptiX DIRECTLY. ANARI is the Khronos rendering API: one scene description, and
the backend is whatever is installed -- `barney`/OptiX on an NVIDIA card, a CPU path elsewhere --
so a Plexus renderer written against it is not tied to one vendor's SDK. `pynari` is its Python
binding. A particle is a first-class `sphere` geometry there: positions and one radius, no
triangles, no glyph expansion, and the BVH is built by the backend.

WHAT IT READS. `<run_dir>/simulation.zarr` (the recorded trajectory), positions per frame per set,
occupancy to drop the parked pool. Nothing is simulated here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def read_frame(run_dir: str, frame: int = -1):
    """(positions [N, 3] of the live particles, the world box) of one recorded frame.

    The zarr when it has one, else `trajectory.npz` -- a run whose zarr writer failed (or whose
    zarr version disagreed with the reader) still leaves the npz, and the two hold the same rows."""
    zpath = os.path.join(run_dir, "simulation.zarr")
    pos = occ = None
    try:
        import zarr
        z = zarr.open(zpath, "r")
        sets = [k for k in z.group_keys() if "pos" in z[k].array_keys()]
        best = max(sets, key=lambda k: z[k]["pos"].shape[1])
        a = z[best]["pos"]
        t = a.shape[0] + frame if frame < 0 else frame
        pos = np.asarray(a[t], np.float32)
        occ = np.asarray(z[best]["occ"][t]) if "occ" in z[best].array_keys() else None
    except Exception:                                          # noqa: BLE001 -- fall back to the npz
        with np.load(os.path.join(run_dir, "trajectory.npz")) as z:
            keys = [k for k in z.files if k.endswith("__pos")]
            best = max(keys, key=lambda k: z[k].shape[1])
            a = z[best]
            t = a.shape[0] + frame if frame < 0 else frame
            pos = np.asarray(a[t], np.float32)
            ok = best[:-len("__pos")] + "__occ"
            occ = np.asarray(z[ok][t]) if ok in z.files else None
    if occ is not None:
        pos = pos[occ > 0]
    pos = pos[np.abs(pos).max(1) < 1e5]                       # parked slots sit at -1e6
    lo, hi = pos.min(0), pos.max(0)
    return pos, (lo, hi)


def body_colours(run_dir: str, n: int):
    """One colour per body from the run's own spec, mapped through `mpm_particle`'s parent -- the
    picture the page draws, not a height ramp."""
    import yaml
    sp = os.path.join(run_dir, "spec.yaml")
    if not os.path.exists(sp):
        return None
    spec = yaml.safe_load(open(sp))
    cols = ((spec.get("plotting") or {}).get("colors") or {})
    types = ((spec.get("sets") or {}).get("cell") or {}).get("types") or {}
    if not cols or not types:
        return None
    per = ((spec.get("sets") or {}).get("mpm_particle") or {}).get("per_parent")
    if not isinstance(per, int):
        return None
    tab = np.asarray([cols.get(t, [0.8, 0.8, 0.8]) for t in types], np.float32)
    body = (np.arange(n) // per) % len(tab)
    return tab[body]


def colours(pos, lo, hi, up=1):
    """A height ramp, the renderer's own fallback -- the trajectory stores no types here."""
    t = (pos[:, up] - lo[up]) / max(float(hi[up] - lo[up]), 1e-9)
    return np.stack([np.clip(1.4 - 1.6 * t, 0, 1), np.clip(0.35 + 0.5 * t, 0, 1),
                     np.clip(0.25 + 1.1 * t, 0, 1)], 1).astype(np.float32)


class AnariScene:
    """The particles as ANARI spheres, once; every frame only moves the camera."""

    def __init__(self, pos, col, box, px=(1000, 1000), radius=None, ao=0.5, spp=1,
                 metallic=0.0, roughness=0.35):
        import pynari as anari
        self.an = anari
        self.px = px
        lo, hi = box
        self.centre = 0.5 * (lo + hi)
        self.span = float(np.max(hi - lo))
        # A SURFACE NEEDS THE SPHERES TO TOUCH. The default is half the median nearest-neighbour
        # spacing of the points, so neighbouring spheres just meet and a body reads as a solid;
        # smaller and it is a cloud of dots, larger and thin sheets close up.
        self.radius = float(radius or self._spacing(pos) * 0.62)
        d = anari.newDevice("default")
        self.dev = d
        g = d.newGeometry("sphere")
        g.setParameterArray1D("vertex.position", anari.FLOAT32_VEC3, pos)
        g.setParameterArray1D("vertex.color", anari.FLOAT32_VEC3, col)
        g.setParameter("radius", anari.FLOAT, self.radius)
        g.commitParameters()
        m = d.newMaterial("physicallyBased")
        m.setParameter("baseColor", anari.STRING, "color")     # per-sphere colour
        m.setParameter("metallic", anari.FLOAT, float(metallic))
        m.setParameter("roughness", anari.FLOAT, float(roughness))
        m.commitParameters()
        s = d.newSurface()
        s.setParameter("geometry", anari.GEOMETRY, g)
        s.setParameter("material", anari.MATERIAL, m)
        s.commitParameters()
        w = d.newWorld()
        w.setParameterArray1D("surface", anari.SURFACE, [s])
        lights = []
        for direction, irr in (((-0.4, -1.0, -0.5), 3.0), ((0.7, -0.3, 0.6), 1.2), ((0.2, 0.6, -0.9), 0.6)):
            li = d.newLight("directional")
            li.setParameter("direction", anari.FLOAT32_VEC3, direction)
            li.setParameter("irradiance", anari.FLOAT, irr)
            li.commitParameters()
            lights.append(li)
        w.setParameterArray1D("light", anari.LIGHT, lights)
        w.commitParameters()
        self.world = w
        self.cam = d.newCamera("perspective")
        self.cam.setParameter("aspect", anari.FLOAT, px[0] / px[1])
        self.cam.setParameter("fovy", anari.FLOAT, 0.55)
        r = d.newRenderer("default")
        # ONE SAMPLE PER PIXEL. The backend path-traces, so its cost is pixels x samples and hardly
        # depends on the particle count at all (36k and 5M measured within 2x of each other).
        r.setParameter("pixelSamples", anari.INT32, int(spp))
        r.setParameter("ambientRadiance", anari.FLOAT, float(ao))
        r.setParameter("background", anari.FLOAT32_VEC4, (0.0, 0.0, 0.0, 1.0))
        r.commitParameters()
        self.frame = d.newFrame()
        self.frame.setParameter("size", anari.UINT32_VEC2, px)
        self.frame.setParameter("channel.color", anari.DATA_TYPE, anari.UFIXED8_RGBA_SRGB)
        self.frame.setParameter("world", anari.OBJECT, w)
        self.frame.setParameter("camera", anari.OBJECT, self.cam)
        self.frame.setParameter("renderer", anari.OBJECT, r)
        self.frame.commitParameters()

    @staticmethod
    def _spacing(pos, k=20000):
        from scipy.spatial import cKDTree
        q = np.asarray(pos, np.float64)[:: max(1, len(pos) // k)]
        nn = cKDTree(q).query(q, k=2)[0][:, 1]
        return float(np.median(nn[np.isfinite(nn)]))

    def look(self, azim_deg, elev_deg=18.0, dist=2.1):
        a, e = np.radians(azim_deg), np.radians(elev_deg)
        d = np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])
        p = self.centre + d * (dist * self.span)
        self.cam.setParameter("position", self.an.FLOAT32_VEC3, tuple(float(x) for x in p))
        self.cam.setParameter("direction", self.an.FLOAT32_VEC3, tuple(float(-x) for x in d))
        self.cam.setParameter("up", self.an.FLOAT32_VEC3, (0.0, 1.0, 0.0))
        self.cam.commitParameters()

    def render(self):
        self.frame.render()
        return np.asarray(self.frame.get("channel.color"))[..., :3]


def bench_vtk(pos, col, box, px=(1000, 1000), n=8):
    """The path the page and the movie use today: a VTK point cloud, one pixel-sized dot each."""
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=px, border=False)
    cloud = pv.PolyData(pos.astype(np.float32))
    cloud["rgb"] = (np.clip(col, 0, 1) * 255).astype(np.uint8)
    p.add_mesh(cloud, scalars="rgb", rgb=True, render_points_as_spheres=True, lighting=False,
               ambient=1.0, diffuse=0.0, specular=0.0, point_size=2.0)
    p.set_background("black")
    lo, hi = box
    c = 0.5 * (lo + hi); span = float(np.max(hi - lo))
    p.camera.focal_point = tuple(float(x) for x in c)
    p.camera.position = tuple(float(x) for x in (c + np.array([1.9, 0.8, 1.9]) * span))
    p.camera.up = (0.0, 1.0, 0.0)
    p.render(); p.screenshot(return_img=True)                  # warm the pipeline
    t0 = time.time()
    for _ in range(n):
        p.render(); img = np.asarray(p.screenshot(return_img=True))
    dt = (time.time() - t0) / n
    p.close()
    return dt, img[..., :3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--px", type=int, default=1000)
    ap.add_argument("--radius", type=float, default=None, help="sphere radius in world units")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="subsample to this many particles (0 = all)")
    ap.add_argument("--spp", type=int, default=1, help="samples per pixel")
    ap.add_argument("--cpu", action="store_true", help="hide the GPUs from the backend, to see which it was using")
    ap.add_argument("--turntable", type=int, default=0, help="frames of a camera orbit -> mp4")
    ap.add_argument("--stills", action="store_true", help="one still per material, for comparing renders")
    ap.add_argument("--azim", type=float, default=35.0)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--dist", type=float, default=2.1)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="movie_anari.mp4")
    a = ap.parse_args()

    if a.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.time()
    pos, box = read_frame(a.run_dir, a.frame)
    if a.n and a.n < len(pos):
        pos = pos[:: max(1, len(pos) // a.n)][:a.n]
    col = body_colours(a.run_dir, len(pos))
    if col is None or len(col) != len(pos):
        col = colours(pos, *box)
    print(f"[data] {len(pos):,} live particles from {a.run_dir} in {time.time() - t0:.1f}s; "
          f"box {np.round(box[0], 3).tolist()}..{np.round(box[1], 3).tolist()}", flush=True)
    px = (a.px, a.px)

    if a.bench:
        t0 = time.time()
        sc = AnariScene(pos, col, box, px, radius=a.radius, spp=a.spp)
        build = time.time() - t0
        sc.look(35.0)
        sc.render()                                            # first render builds the BVH
        first = time.time() - t0
        t0 = time.time()
        for k in range(8):
            sc.look(35.0 + 3.0 * k); img = sc.render()
        an_dt = (time.time() - t0) / 8
        import imageio.v3 as iio
        iio.imwrite(os.path.join(a.run_dir, "anari_bench.png"), img)
        vt_dt, vimg = bench_vtk(pos, col, box, px)
        iio.imwrite(os.path.join(a.run_dir, "vtk_bench.png"), vimg)
        print(f"[bench] {len(pos):,} particles at {a.px}x{a.px}")
        print(f"[bench] ANARI spheres : scene build {build:.1f}s, first frame (BVH) {first:.1f}s, "
              f"then {an_dt * 1000:.0f} ms/frame  ({1 / an_dt:.1f} fps)")
        print(f"[bench] VTK points    : {vt_dt * 1000:.0f} ms/frame  ({1 / vt_dt:.1f} fps)")

    if a.stills:
        import imageio.v3 as iio
        for nm, kw in (("matte", dict(metallic=0.0, roughness=1.0, spp=a.spp)),
                       ("glossy", dict(metallic=0.0, roughness=0.12, spp=a.spp)),
                       ("metal", dict(metallic=1.0, roughness=0.2, spp=a.spp)),
                       ("path16", dict(metallic=0.0, roughness=0.35, spp=16)),
                       ("fat", dict(metallic=0.0, roughness=0.5, spp=a.spp, radius_mul=1.6)),
                       ("dots", dict(metallic=0.0, roughness=0.6, spp=a.spp, radius_mul=0.45))):
            mul = kw.pop("radius_mul", 1.0)
            sc = AnariScene(pos, col, box, px, radius=(a.radius * mul if a.radius else None), **kw)
            if a.radius is None and mul != 1.0:                 # scale the measured default too
                sc = AnariScene(pos, col, box, px, radius=sc.radius * mul, **kw)
            sc.look(a.azim, a.elev, a.dist)
            t0 = time.time(); img = sc.render(); dt = time.time() - t0
            out = os.path.join(a.run_dir, f"anari_{nm}.png")
            iio.imwrite(out, img)
            print(f"[still] {nm:7s} {dt * 1000:6.0f} ms -> {out}", flush=True)

    if a.turntable:
        import imageio.v2 as iio
        sc = AnariScene(pos, col, box, px, radius=a.radius, spp=a.spp)
        out = a.out if os.path.isabs(a.out) else os.path.join(a.run_dir, a.out)
        w = iio.get_writer(out, fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
        t0 = time.time()
        for k in range(a.turntable):
            sc.look(360.0 * k / a.turntable, elev_deg=14.0 + 10.0 * np.sin(2 * np.pi * k / a.turntable))
            w.append_data(sc.render())
            if k % 30 == 0:
                print(f"  frame {k}/{a.turntable}  {(time.time() - t0) / max(k, 1) * 1000:.0f} ms/frame", flush=True)
        w.close()
        print(f"[turntable] {a.turntable} frames -> {out} in {time.time() - t0:.1f}s "
              f"({(time.time() - t0) / a.turntable * 1000:.0f} ms/frame)")


if __name__ == "__main__":
    main()
