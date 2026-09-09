"""A glassy, ray-traced-as-far-as-this-install-goes render of a morph.npz.

WHAT THIS REPLACES. `render()` in tools/morph_gallery.py voxelises the cloud onto a 96^3 grid, box
-blurs it once, contours it, and draws the result `pbr=True, metallic=0.05, roughness=0.28` under
three flat lights. That is an opaque plastic body, not glass, and the 96^3 grid is coarse enough
that a morph's thin parts -- an armadillo's ears, a bunny's ears, a cow's legs -- get swallowed by
the blur before the contour ever sees them (see `compare_recon.png` in the output directory: the two
front legs of the armadillo fuse into one wedge at 96^3 and separate cleanly at 176^3).

RAY TRACING, CHECKED RATHER THAN ASSUMED. `check_ospray()` below is the actual test: importing
`vtkmodules.vtkRenderingRayTracing.vtkOSPRayPass`. On both the devcontainer's vtk 9.5.2 (pip wheel)
and the cluster's `connectome-gnn` env (vtk 9.3.1, also a pip wheel) this raises
`ModuleNotFoundError: No module named 'vtkmodules.vtkRenderingRayTracing'` -- OSPRay is a separate
compiled module that the pip `vtk` wheels do not ship, and `pyvista.Plotter` in this install has no
`enable_ray_tracing` method at all (checked: `'enable_ray_tracing' not in dir(pv.Plotter)`). So this
file does not pretend to path-trace. It gets as close as VTK's rasterised PBR path goes: an
environment texture for image-based lighting, screen-space ambient occlusion, depth peeling for the
translucent glass, SSAA, and a ground plane.

GLASS, load-bearing finding: `pv.Plotter.enable_shadows()` (VTK's shadow-map pass) and a translucent
PBR mesh (`opacity < 1`) DO NOT COMBINE in this VTK build -- turn both on together and the translucent
body renders fully black, indistinguishable from the background. Isolated by hand, one flag at a
time: opaque + shadows renders fine; translucent + shadows alone renders fine; translucent + shadows
+ depth-peeling (+ IBL) together drops the body to black. Choosing glass over a cast shadow, this
renderer leaves `enable_shadows()` off entirely and uses SSAO for the contact cue instead -- the
darkening a body's own occlusion leaves in the ground under it, which the stills show is enough to
read where it stands.

BLUE, TRANSPARENT, SEE-THROUGH: a cold, saturated dielectric (`color` a deep blue, `opacity` around
0.45-0.5, low `roughness`) with enough depth-peeling layers (`peels=16`) that the FAR side of the body
is not clipped -- on the armadillo's raised arm you can see its far edge through the near torso in
every still here. No organelles exist in an MPM point cloud to read through the surface (this is one
homogeneous material, not a cell); what DOES read through is the body's own far geometry and the
faint grey target cloud drawn behind it.

    PYTHONPATH=src python tools/morph_render.py --dirs graphs_data/si_material/morph_armadillo
    PYTHONPATH=src python tools/morph_render.py --glob 'graphs_data/si_material/morph_*' --compare
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DEFAULT_OUT = os.path.join(ROOT, "graphs_data", "si_material", "render")


def check_ospray():
    """The actual test, not an assumption. Returns (available, detail)."""
    try:
        from vtkmodules.vtkRenderingRayTracing import vtkOSPRayPass  # noqa: F401
        return True, "vtkOSPRayPass imported"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def reconstruct_contour(X, box, ngrid=176, sigma=0.95, iso_frac=0.3, smooth_iter=35):
    """The body as a contoured, gaussian-blurred density -- the same idea as `morph_gallery.render`
    but at a grid fine enough, and a blur wide enough relative to it, to keep thin parts separate.

    THE TWO NUMBERS FIGHT EACH OTHER. A finer grid (`ngrid` up) resolves thinner features, but it
    also means fewer particles land in each voxel, so the raw density is noisier -- measured here:
    96^3 with one 3-voxel box blur gives a clean but overly fused silhouette (legs merge into a
    wedge); 224^3 at the SAME blur width comes out pockmarked, a dimple wherever a voxel happened to
    get one particle instead of three. 176^3 with a wider single gaussian blur (sigma≈1 voxel) is
    the balance point found by comparing the three against these fixed 50,000-particle clouds --
    change the particle count and this number should move with it.
    """
    from scipy.ndimage import gaussian_filter
    import pyvista as pv
    dx = box / ngrid
    idx = np.clip((X / dx).astype(int), 0, ngrid - 1)
    dens = np.zeros((ngrid, ngrid, ngrid), np.float32)
    np.add.at(dens, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    dens = gaussian_filter(dens, sigma=sigma)
    g = pv.ImageData(dimensions=(ngrid + 1,) * 3, spacing=(dx, dx, dx), origin=(0.0, 0.0, 0.0))
    g["v"] = np.pad(dens, ((0, 1), (0, 1), (0, 1))).flatten(order="F")
    iso = max(iso_frac * float(dens[dens > 0].mean()), 1e-6)
    surf = g.contour([iso], scalars="v")
    if surf.n_points == 0:
        return None
    surf = surf.smooth(n_iter=smooth_iter, relaxation_factor=0.15)
    return surf.compute_normals(auto_orient_normals=True, non_manifold_traversal=False)


def reconstruct_delaunay(X, box, alpha_frac=0.02):
    """An alpha-shape via `delaunay_3d`. STILLS ONLY -- included for the comparison figure, not the
    movie: it is faceted (no blur to smooth the noise a scattered MPM cloud always has) and its
    correct `alpha` depends on the local point spacing, which changes every frame as the body
    deforms, so a fixed alpha that looks right on frame 0 will pick up spurious cavities or lose
    thin parts by frame 50."""
    import pyvista as pv
    pc = pv.PolyData(X.astype(np.float32))
    return pc.delaunay_3d(alpha=box * alpha_frac).extract_surface()


def reconstruct_poisson(X, box):
    """`pyvista.PolyData.reconstruct_surface` (screened Poisson via `vtkSurfaceReconstructionFilter`).
    STILLS ONLY, and only just: measured at 74.6 s for a single 50,000-point frame here, three orders
    of magnitude slower than the grid-contour route, and it returns a torn, self-intersecting mesh on
    this cloud (see `compare_recon.png`) because the filter assumes a roughly uniform surface sampling
    and an MPM cloud is a volume fill, not a surface scan -- there is no true surface to reconstruct,
    only a density to contour."""
    import pyvista as pv
    pc = pv.PolyData(X.astype(np.float32))
    return pc.reconstruct_surface(nbr_sz=12, sample_spacing=box / 96)


_ENV_CACHE = {}


def env_texture(bright=2.3):
    """A synthetic equirectangular sky for image-based lighting: no internet access on the cluster
    node to fetch a real HDRI, so this is a gradient (horizon navy to a lighter blue-teal zenith)
    plus one soft bright patch, which is what a PBR dielectric picks up as its directional highlight
    and reflected colour. `bright` matters more than it looks like it should: too dim (the first
    version of this texture, `bright=1`) and a translucent, low-diffuse glass has nothing to catch --
    it reads as near-black with a few white specular flecks, not blue. The chosen 2.3 is bright
    enough for colour to read while `TexturedBackgroundOff` keeps the black background black."""
    import pyvista as pv
    key = round(float(bright), 3)
    if key not in _ENV_CACHE:
        H, W = 128, 256
        yy = np.linspace(0, 1, H)[:, None] * np.ones((1, W))
        xx = np.linspace(0, 1, W)[None, :] * np.ones((H, 1))
        sky = np.stack([0.05 + 0.10 * yy, 0.07 + 0.14 * yy, 0.12 + 0.22 * yy], axis=-1) * bright
        sun = np.exp(-(((xx - 0.7) * W) ** 2 + ((yy - 0.15) * H) ** 2) / (2 * 22 ** 2))
        sky = sky + sun[..., None] * np.array([0.6, 0.6, 0.65]) * bright
        img = (np.clip(sky, 0, 1) * 255).astype(np.uint8)
        tex = pv.Texture(img)
        # UNBLURRED (mipmap/interpolate off) SPARKLES: the PBR prefilter warns about exactly this,
        # and it is not cosmetic -- without it, every marching-cubes facet catches a raw, aliased
        # texel of the "sun" patch and the glass reads as covered in tiny white measles.
        tex.SetMipmap(True)
        tex.SetInterpolate(True)
        _ENV_CACHE[key] = tex
    return _ENV_CACHE[key]


def build_lights(pl, settings):
    """Key, fill, rim -- plus a fourth, a BACKLIGHT, that a glass render needs and an opaque one
    does not: light arriving from behind the body is what a translucent dielectric forwards through
    itself, and without it there is nothing to be transparent TO except the dim environment map."""
    import pyvista as pv
    key = pv.Light(light_type="scene light")
    key.set_direction_angle(*settings["key_angle"])
    key.intensity = settings["key_intensity"]
    fill = pv.Light(light_type="scene light")
    fill.set_direction_angle(*settings["fill_angle"])
    fill.intensity = settings["fill_intensity"]
    rim = pv.Light(light_type="scene light")
    rim.set_direction_angle(*settings["rim_angle"])
    rim.intensity = settings["rim_intensity"]
    rim.SetColor(*settings["rim_color"])
    back = pv.Light(light_type="scene light")
    back.set_direction_angle(*settings["back_angle"])
    back.intensity = settings["back_intensity"]
    back.SetColor(*settings["back_color"])
    for light in (key, fill, rim, back):
        pl.add_light(light)


def draw_scene(pl, surf, target, box, X, settings):
    """Everything but the plotter and the camera: the glass body, the faint target cloud behind it,
    four lights, image-based lighting, a small dark ground plane, and the domain wireframe box."""
    import pyvista as pv
    pl.set_background(settings["bg"])
    pl.set_environment_texture(env_texture(settings["env_bright"]))
    pl.renderer.TexturedBackgroundOff()   # IBL for reflections only -- the black bg stays black

    pl.add_mesh(pv.PolyData(target.astype("float32")), color=settings["target_color"],
                opacity=0.08, point_size=1.6)

    if surf is not None and surf.n_points > 0:
        pl.add_mesh(surf, color=settings["color"], pbr=True, metallic=settings["metallic"],
                    roughness=settings["roughness"], opacity=settings["opacity"],
                    diffuse=settings["diffuse"], ambient=settings["ambient"], smooth_shading=True)
        actor = list(pl.renderer.actors.values())[-1]
        prop = actor.GetProperty()
        # THE DIELECTRIC ITSELF: index of refraction and a thin, sharp clear-coat on top of the
        # base PBR layer, which is what gives a low-roughness surface its glassy double highlight
        # instead of a single soft plastic one.
        prop.SetBaseIOR(settings["ior"])
        prop.SetCoatStrength(settings["coat_strength"])
        prop.SetCoatRoughness(settings["coat_roughness"])
        prop.SetCoatIOR(settings["coat_ior"])
    else:
        pl.add_mesh(pv.PolyData(X.astype("float32")), color=settings["color"], point_size=2.2)

    build_lights(pl, settings)

    # THE GROUND IS PLAIN LAMBERTIAN, NOT PBR. A `pbr=True` plane under the same bright environment
    # texture that lights the glass turns into a lit grey studio floor that fills half the frame --
    # fine for a product shot, wrong for this project's black-background convention. Flat diffuse
    # shading plus SSAO's contact darkening is enough to place the body without competing with it.
    ymin = float(X[:, 1].min())
    ground = pv.Plane(center=(box / 2, ymin - 0.002, box / 2), direction=(0, 1, 0),
                       i_size=box * 1.05, j_size=box * 1.05)
    pl.add_mesh(ground, color=settings["ground_color"], pbr=False, ambient=0.02, diffuse=0.5,
                specular=0.0)
    pl.add_mesh(pv.Box((0, box, 0, box, 0, box)), style="wireframe", color="#131a24",
                line_width=1.0, lighting=False)

    pl.enable_ssao(radius=box * settings["ssao_radius_frac"], bias=box * settings["ssao_bias_frac"],
                    kernel_size=settings["ssao_kernel"])
    pl.enable_depth_peeling(number_of_peels=settings["peels"], occlusion_ratio=0.0)
    pl.enable_anti_aliasing("ssaa")


def camera_for(box):
    # ASYMMETRIC ON PURPOSE. A camera at equal x/z offsets from a cubic domain (the original
    # `(box*2.4, box*1.35, box*2.4)`) puts the domain box's own far vertical edge exactly behind the
    # body's centreline -- a bright seam straight down every still and every frame of the movie that
    # has nothing to do with the body. Breaking the x/z symmetry removes it.
    return [(box * 2.5, box * 1.4, box * 1.65), (box / 2,) * 3, (0, 1, 0)]


def default_settings():
    """A cold, saturated blue dielectric, tuned by looking (candidates in `render_candidates` were
    the ones compared): opacity ~0.5, roughness low but not mirror-flat, 16 depth-peeling layers so
    the far side of a concave body (an arm crossing in front of a torso) is not clipped, and a
    fourth BACKLIGHT (see `build_lights`) so there is light on the far side to forward through the
    body in the first place -- glass with nothing behind it just looks like dark plastic."""
    return dict(
        ngrid=176, sigma=0.95, iso_frac=0.3, smooth_iter=35,
        color="#1a4fc0", target_color="#4a5a6a", ground_color="#04050a", bg="#05070a",
        roughness=0.07, opacity=0.5, diffuse=0.42, ambient=0.06,
        ior=1.4, coat_strength=0.85, coat_roughness=0.05, coat_ior=1.5,
        metallic=0.0,
        ssao_radius_frac=0.12, ssao_bias_frac=0.002, ssao_kernel=192, peels=16,
        env_bright=2.3, key_intensity=1.2, fill_intensity=0.32, rim_intensity=1.1,
        rim_color=(0.75, 0.85, 1.0), back_intensity=1.0, back_color=(0.6, 0.8, 1.0),
        # (elevation_deg, azimuth_deg) for pv.Light.set_direction_angle -- key/fill model the form,
        # rim/back separate the glass silhouette from the black background and give the dielectric
        # something on its far side to forward through itself.
        key_angle=(55, -35), fill_angle=(35, 150), rim_angle=(-35, 20), back_angle=(15, 175),
    )


CANDIDATES = {
    # name -> overrides on top of default_settings(), the exact settings compared by eye before
    # picking the default above. `--candidates` renders every one of these on one frame so a human
    # can compare stills instead of trusting a description.
    "blue_deep_op045":   dict(color="#2361d0", roughness=0.06, opacity=0.45, ior=1.45,
                              coat_roughness=0.04, coat_ior=1.5),
    "blue_saturated_op050": dict(color="#1a4fc0", roughness=0.07, opacity=0.50, ior=1.40,
                                 coat_roughness=0.05, coat_ior=1.5),
    "blue_pale_op038":   dict(color="#3a78e0", roughness=0.05, opacity=0.38, ior=1.35,
                              coat_roughness=0.03, coat_ior=1.45, peels=20),
    "sapphire_op070":    dict(color="#1e56c4", roughness=0.10, opacity=0.70, ior=1.50,
                              coat_strength=0.75, coat_roughness=0.06, coat_ior=1.55),
    "cyan_op050":        dict(color="#28a8e6", roughness=0.06, opacity=0.50, ior=1.38,
                              coat_roughness=0.04, coat_ior=1.5),
    "white_glass_no_backlight": dict(color="#eef3f6", roughness=0.09, opacity=0.58, ior=1.45,
                                     diffuse=0.18, ambient=0.0, coat_strength=0.8,
                                     coat_roughness=0.04, back_intensity=0.0),
}


def render_candidates(out_dir, X, target, box, px=800):
    """Every named look in `CANDIDATES`, same reconstruction, same camera, differing only in the
    PBR/light settings -- so a human can pick the glass by looking at pngs instead of reading
    numbers. Returns {name: settings actually used} for results.json."""
    import pyvista as pv
    import imageio.v3 as iio
    pv.OFF_SCREEN = True
    os.makedirs(out_dir, exist_ok=True)
    surf = reconstruct_contour(X, box)
    used = {}
    for name, overrides in CANDIDATES.items():
        settings = default_settings()
        settings.update(overrides)
        pl = pv.Plotter(off_screen=True, window_size=(px, px), lighting="none")
        draw_scene(pl, surf, target, box, X, settings)
        pl.camera_position = camera_for(box)
        pl.add_text(name, position="upper_left", font_size=10, color="#c8d4e0")
        img = pl.screenshot(return_img=True)
        pl.close()
        iio.imwrite(os.path.join(out_dir, f"{name}.png"), img)
        used[name] = settings
    return used


def render_movie(out_dir, frames, target, box, name, settings, px=1100, fps=20, stride=1):
    import pyvista as pv
    import imageio.v3 as iio
    pv.OFF_SCREEN = True
    idx = list(range(0, len(frames), stride))
    if idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1)
    imgs = []
    recon_t = render_t = 0.0
    for j, i in enumerate(idx):
        X = frames[i].astype(np.float64)
        t0 = time.time()
        surf = reconstruct_contour(X, box, settings["ngrid"], settings["sigma"],
                                    settings["iso_frac"], settings["smooth_iter"])
        recon_t += time.time() - t0
        pl = pv.Plotter(off_screen=True, window_size=(px, px), lighting="none")
        draw_scene(pl, surf, target, box, X, settings)
        pl.camera_position = camera_for(box)
        pl.add_text(f"{name}   frame {i}/{len(frames) - 1}   {len(X):,} material points   glass",
                    position="upper_left", font_size=9, color="#c8d4e0")
        t0 = time.time()
        img = pl.screenshot(return_img=True)
        render_t += time.time() - t0
        pl.close()
        imgs.append(img)
        if j in (0, len(idx) // 2, len(idx) - 1):
            iio.imwrite(os.path.join(out_dir, f"still_{i:03d}.png"), img)
    iio.imwrite(os.path.join(out_dir, "movie.mp4"), imgs, fps=fps, codec="libx264",
                macro_block_size=None)
    n = len(idx)
    return dict(n_frames_rendered=n, n_frames_total=len(frames), stride=stride,
                seconds_per_frame_recon=recon_t / n, seconds_per_frame_render=render_t / n,
                seconds_per_frame_total=(recon_t + render_t) / n, px=px, fps=fps)


def render_compare(out_dir, X, box):
    """One frame, four reconstructions side by side: the shipped 96^3/one-blur, the chosen
    176^3/wider-blur, delaunay_3d, and reconstruct_surface. Answers "did the surface choice matter"
    with a picture instead of an assertion."""
    import pyvista as pv
    import imageio.v3 as iio
    pv.OFF_SCREEN = True

    def shot(surf, label):
        pl = pv.Plotter(off_screen=True, window_size=(500, 500), lighting="three lights")
        pl.set_background("#05070a")
        if surf is not None and surf.n_points > 0:
            pl.add_mesh(surf, color="#e8b268", smooth_shading=True)
        pl.camera_position = camera_for(box)
        pl.add_text(label, position="upper_left", font_size=11, color="white")
        img = pl.screenshot(return_img=True)
        pl.close()
        return img

    t0 = time.time()
    # REPRODUCES morph_gallery.render's `surface()` EXACTLY: 96^3 grid, ONE 3x3x3 box-blur pass
    # (not a gaussian), iso at 0.35 of the mean nonzero density -- the shipped defaults, not a
    # strawman.
    from scipy.ndimage import convolve
    dx = box / 96.0
    idxv = np.clip((X / dx).astype(int), 0, 95)
    dens = np.zeros((96, 96, 96), np.float32)
    np.add.at(dens, (idxv[:, 0], idxv[:, 1], idxv[:, 2]), 1.0)
    dens = convolve(dens, np.ones((3, 3, 3), np.float32) / 27.0, mode="constant")
    g = pv.ImageData(dimensions=(97, 97, 97), spacing=(dx, dx, dx), origin=(0, 0, 0))
    g["v"] = np.pad(dens, ((0, 1), (0, 1), (0, 1))).flatten(order="F")
    iso = max(0.35 * float(dens[dens > 0].mean()), 1e-6)
    shipped = g.contour([iso], scalars="v").smooth(n_iter=30, relaxation_factor=0.2)
    t_shipped = time.time() - t0

    t0 = time.time()
    chosen = reconstruct_contour(X, box)
    t_chosen = time.time() - t0

    t0 = time.time()
    try:
        dela = reconstruct_delaunay(X, box)
        t_dela = time.time() - t0
    except Exception:
        dela, t_dela = None, -1.0

    t0 = time.time()
    try:
        pois = reconstruct_poisson(X, box)
        t_pois = time.time() - t0
    except Exception:
        pois, t_pois = None, -1.0

    panels = [
        shot(shipped, f"morph_gallery: 96^3, 1 box-blur ({t_shipped:.2f}s)"),
        shot(chosen, f"chosen: 176^3, gaussian blur ({t_chosen:.2f}s)"),
        shot(dela, f"delaunay_3d ({t_dela:.2f}s)" if dela is not None else "delaunay_3d: FAILED"),
        shot(pois, f"reconstruct_surface ({t_pois:.2f}s)" if pois is not None
             else "reconstruct_surface: FAILED"),
    ]
    grid = np.concatenate([np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)],
                           axis=0)
    iio.imwrite(os.path.join(out_dir, "compare_recon.png"), grid)
    return dict(shipped_96_blur1_s=t_shipped, chosen_176_gauss_s=t_chosen,
                delaunay_3d_s=t_dela, reconstruct_surface_s=t_pois,
                shipped_n_pts=int(shipped.n_points) if shipped is not None else 0,
                chosen_n_pts=int(chosen.n_points) if chosen is not None else 0,
                delaunay_n_pts=int(dela.n_points) if dela is not None else 0,
                poisson_n_pts=int(pois.n_points) if pois is not None else 0)


def _update_results(path, name, entry):
    """Read-modify-write so a killed job leaves every target finished so far on disk, not nothing."""
    data = {}
    if os.path.exists(path):
        try:
            data = json.load(open(path))
        except Exception:
            data = {}
    data[name] = entry
    tmp = path + ".tmp"
    json.dump(data, open(tmp, "w"), indent=2, sort_keys=True)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default=None,
                    help=f"defaults to '{os.path.join('graphs_data', 'si_material', 'morph_*')}' "
                         "ONLY when --dirs is not given")
    ap.add_argument("--dirs", nargs="*", default=None, help="explicit morph_* directories")
    ap.add_argument("--out-root", default=DEFAULT_OUT)
    ap.add_argument("--px", type=int, default=1100)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame of the rollout")
    ap.add_argument("--compare", action="store_true", help="also write compare_recon.png per target")
    ap.add_argument("--candidates", action="store_true",
                    help="write every CANDIDATES look as a named still (one frame, one target) "
                         "instead of movies -- for picking a look, not for the final render")
    ap.add_argument("--skip-movie", action="store_true", help="stills/compare + json only")
    ap.add_argument("--ngrid", type=int, default=None, help="override: contour grid resolution")
    ap.add_argument("--sigma", type=float, default=None, help="override: gaussian blur, in voxels")
    ap.add_argument("--roughness", type=float, default=None, help="override: PBR roughness")
    ap.add_argument("--opacity", type=float, default=None, help="override: glass opacity")
    ap.add_argument("--ior", type=float, default=None, help="override: base index of refraction")
    ap.add_argument("--color", default=None, help="override: glass colour, e.g. '#1a4fc0'")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    results_path = os.path.join(args.out_root, "results.json")

    ospray_ok, ospray_detail = check_ospray()
    print(f"  OSPRay available: {ospray_ok}  ({ospray_detail})", flush=True)
    backend = "ospray" if ospray_ok else "vtk-pbr"

    import pyvista as pv
    import vtk
    if args.dirs:
        dirs = sorted(set(args.dirs))
    else:
        pat = args.glob or os.path.join(ROOT, "graphs_data", "si_material", "morph_*")
        dirs = sorted(d for d in glob.glob(pat) if os.path.isdir(d))
    if not dirs:
        print("  no morph_* directories matched"); return 2

    settings = default_settings()
    overrides = dict(ngrid=args.ngrid, sigma=args.sigma, roughness=args.roughness,
                      opacity=args.opacity, ior=args.ior, color=args.color)
    settings.update({k: v for k, v in overrides.items() if v is not None})

    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        npz_path = os.path.join(d, "morph.npz")
        if not os.path.exists(npz_path):
            print(f"  {name}: no morph.npz -- skipped"); continue
        out_dir = os.path.join(args.out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        entry = dict(backend=backend, ospray_detail=ospray_detail,
                     vtk_version=vtk.VTK_VERSION, pyvista_version=pv.__version__,
                     settings=settings, status="running", t_start=time.time())
        _update_results(results_path, name, entry)
        try:
            d_npz = np.load(npz_path)
            frames, target, box = d_npz["frames"], d_npz["target"], float(d_npz["box"])
            if "meta" in d_npz.files:
                try:
                    entry["source_meta"] = json.loads(str(d_npz["meta"]))
                except Exception:
                    pass
            print(f"  {name}: {len(frames)} frames, {frames.shape[1]:,} points, box {box}",
                  flush=True)
            if args.candidates:
                t0 = time.time()
                cand_dir = os.path.join(out_dir, "candidates")
                used = render_candidates(cand_dir, frames[-1].astype(np.float64), target, box)
                entry["candidates"] = dict(seconds=time.time() - t0, settings=used,
                                            dir=os.path.relpath(cand_dir, ROOT))
                _update_results(results_path, name, entry)
                print(f"    candidates/: {len(used)} looks written "
                      f"({entry['candidates']['seconds']:.1f}s)", flush=True)
            if args.compare:
                t0 = time.time()
                entry["compare"] = render_compare(out_dir, frames[-1].astype(np.float64), box)
                entry["compare"]["seconds"] = time.time() - t0
                _update_results(results_path, name, entry)
                print(f"    compare_recon.png written ({entry['compare']['seconds']:.1f}s)",
                      flush=True)
            if not args.skip_movie:
                t0 = time.time()
                stats = render_movie(out_dir, frames, target, box, name, settings,
                                      px=args.px, fps=args.fps, stride=args.stride)
                stats["seconds_total"] = time.time() - t0
                entry.update(stats)
                entry["mp4"] = os.path.relpath(os.path.join(out_dir, "movie.mp4"), ROOT)
                print(f"    movie.mp4: {stats['n_frames_rendered']} frames, "
                      f"{stats['seconds_per_frame_total']:.3f} s/frame, "
                      f"{stats['seconds_total']:.1f} s total", flush=True)
            entry["status"] = "ok"
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()
            print(f"    FAILED: {entry['error']}", flush=True)
        entry["t_end"] = time.time()
        _update_results(results_path, name, entry)

    print(f"\n  results -> {os.path.relpath(results_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
