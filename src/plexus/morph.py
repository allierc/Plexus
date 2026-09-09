"""Morph a ball into a named shape by optimising a FIELD of deformation-gradient rates, and render
the result as glass.

One file, on purpose. Four agents spent a round on this loop across seven scripts; three things
survived, and they are all here: the optimiser, the two speed wins that were verified, and the
renderer that made the shapes judgeable. Everything else was measurement scaffolding and is gone.

    A_p = sum_i w_i(X0_p) A_i           w = trilinear weights on a `--ctrl` cube of control nodes
    F_p <- expm(-A_p dt) F_p            once a frame

Lagrangian on purpose: the trilinear weights are computed once from each particle's position at
t = 0 and never again, so a point carries its own instruction with it as it moves. An Eulerian field
would instruct whatever happens to be passing through a place, which is a different and much harder
thing to optimise. It is also what lets a coarse-to-fine schedule work, since the parameters mean
the same thing at 5,000 particles as at 50,000.

    loss    log nodal mass on the grid, the Eulerian loss the paper argues for: a target shape has
            no particle correspondence, so a position loss has nothing to match against exactly
            where the two shapes differ. `log` because the two mass fields differ by orders of
            magnitude between "material" and "none".
    target  points sampled INSIDE a mesh by ray parity against its surface, scaled into the box.

WHAT WORKED, measured, and merged below:

    torch.compile on the substep block    1.85x wall clock and 1.7x lower peak memory
    warp for the final gradient-free      3.6x at 12,500 particles on an L4, 5.9x at 50,000 on an
    rollout                               A100, both against the ALREADY-COMPILED torch path, with
                                          particle positions agreeing to 1.5e-07, float32 epsilon
    TF32                                  free, 0.015% change in final loss; kept, but not a lever

WHAT DID NOT WORK. Each of these is still runnable, because a negative result is only credible
while the code that produced it is still here.

    a spatial smoothness penalty on the control field
        Monotonically harmful. Weight 0 gives 0.003442, then 0.003835, 0.005142, 0.008727, 0.019674,
        and 0.055271 at weight 0.1. Total-variation behaves the same. It was borrowed reasoning: the
        paper's stabiliser is a TEMPORAL low-pass on F between control layers, not a spatial term,
        and its control is per-particle so neighbouring-node disagreement cannot even arise there.

    --control-dof 9, letting the control rotate material rather than only stretch it
        A tie on every target at identical cost: teapot 0.002060 against 0.002049, armadillo
        0.007383 against 0.007391, spot 0.002587 against 0.002533. Six components store a symmetric
        rate, so expm(-A dt) is a pure stretch and a material point can never turn in place; nine
        store the full 3x3 whose antisymmetric part is a rotation rate. It was a plausible reason
        appendages fail. It is not the reason.

    --trace-free, constraining the control to preserve volume by construction
        Made volume WORSE: 0.316x of initial against the baseline's 0.817x, with a worse loss
        (0.008465 against 0.007014) even though it matched the target bounding box almost exactly.
        det(expm(-A dt)) = 1 constrains the GROWTH tensor; realised volume is sum(p_vol * det F) and
        the elastic part is free to compress whatever the growth part asks for. Volume is set by the
        MPM solve, not by the control's determinant.

    --control ngp, a multiresolution hash encoding in place of the grid
        Diverged. Not under-trained -- diverged.

THE ONE THING TO KNOW BEFORE COMPARING ANYTHING. Scatter atomics make training non-deterministic:
repeat runs of one configuration move the final loss by 18 to 44 per cent. A sweep that ranked six
configurations inside a 1.9 per cent spread was entirely noise, and one arm of the apparent 28 per
cent "winner" scored worse than the baseline it beat. Run repeat seeds or do not claim an ordering.

    PYTHONPATH=src python -m plexus.morph --targets cow,armadillo --iters 60
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "papers", "morph_models")


# --------------------------------------------------------------------------------------------
# the model, the geometry, and the loss's grid
# --------------------------------------------------------------------------------------------

def spec(n_pts, box, n_grid, frames, dt, sub, differentiable=True):
    """The smallest model that can be asked this question: one material, one grid, no forces.

    `differentiable` names the operator bodies. The default bodies write in place, which is what a
    captured CUDA graph needs and what autograd cannot have; the differentiable ones compute the
    same numbers and rebind instead. Training needs them. The final rollout does not, and warp --
    which the engine selects on its own once the pin is dropped -- is several times faster.
    """
    impl = dict(implementation="differentiable") if differentiable else {}
    return dict(
        general=dict(name="morph", seed=0, n_frames=frames, dt=dt, record_cap=3,
                     boundary="wall", dim=3, world=[box, box, box], units=dict(length_um=100.0)),
        sets={"mpm_particle": dict(n=n_pts, types={"cyto": dict(
            fraction=1.0, youngs=90.0, density=1.0,
            block=[0.4 * box, 0.4 * box, 0.4 * box, 0.6 * box, 0.6 * box, 0.6 * box])})},
        fields=dict(mpm_grid=dict(frame="mpm_grid", n_grid=n_grid)),
        operators=[dict(op="mpm_strain", at="mpm_particle", **impl),
                   dict(op="mpm_scatter", at="mpm_particle", to="mpm_grid", drag=0.5,
                        a_max=200.0, **impl),
                   dict(op="mpm_grid_update", at="mpm_grid", wall_damp=0.9, **impl),
                   dict(op="mpm_gather", at="mpm_particle", **{"from": "mpm_grid"},
                        wall_damp=0.9, vmax=1.0e9, **impl)],
        # `compile: true` on the substep block is worth 1.85x under grad for about 35 s of one-time
        # compilation, and cuts peak memory 1.7x. The MPM substep is a few hundred small kernels and
        # a fusing compiler is exactly what that wants. `max-autotune` and `reduce-overhead` both
        # FAIL here -- they conflict with the engine's own CUDA-graph capture -- so `default` only.
        schedule=[dict(substep_dt=sub, compile=True,
                       steps=["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"])],
        plotting={},
    )


def ball(n, c, r, rng):
    """n points uniform in a ball -- the body this starts as."""
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return c + u * (r * rng.random((n, 1)) ** (1.0 / 3.0))


def sample_inside(path, n, box_c, size, rng):
    """`n` points uniform inside a mesh, centred at `box_c` and scaled to `size` across its longest
    axis. Rejection against the surface itself, so the sample is the SOLID and not the shell."""
    import pyvista as pv
    m = pv.read(path)
    if m.n_points == 0:
        raise ValueError(f"{path}: empty mesh")
    m = m.clean().triangulate()
    # A HOLE MAKES `inside` MEANINGLESS -- the Stanford bunny is famously open at the base, and a
    # ray from an interior point escapes through it, so parity says "outside" for half the body.
    m = m.fill_holes(1e9).clean()
    m = m.compute_normals(auto_orient_normals=True, consistent_normals=True)
    lo, hi = np.array(m.bounds).reshape(3, 2).T
    ctr = 0.5 * (lo + hi)
    scale = size / float((hi - lo).max())
    out = []
    while sum(len(o) for o in out) < n:
        q = rng.random((4 * n, 3)) * (hi - lo) + lo
        sel = pv.PolyData(q).select_enclosed_points(m, tolerance=0.0, check_surface=False)
        keep = q[np.asarray(sel["SelectedPoints"]) > 0]
        if len(keep) == 0:
            raise ValueError(f"{path}: no interior points found -- is the surface closed?")
        out.append(keep)
    return (np.concatenate(out)[:n] - ctr) * scale + box_c


def mass_grid(X, m, box, n_grid, dev, torch):
    """The grid's own nodal mass, by the trilinear weights the solver would use.

    UNIT WEIGHTS, NOT THE PARTICLE MASS, at every call site. A material point here weighs about
    1e-8, so log1p of its nodal mass is 1e-8 too and the loss underflows -- it read 0.000000 with a
    gradient of 6e-17, which looks like convergence and is arithmetic. Counting particles per node
    makes the field O(1) and lets the log do what it is there for.
    """
    h = box / n_grid
    g = (X / h).clamp(0, n_grid - 1.001)
    b = g.floor().long()
    f = g - b.float()
    acc = torch.zeros(n_grid ** 3, device=dev, dtype=X.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (((1 - f[:, 0]) if dx == 0 else f[:, 0])
                     * ((1 - f[:, 1]) if dy == 0 else f[:, 1])
                     * ((1 - f[:, 2]) if dz == 0 else f[:, 2]))
                i = ((b[:, 0] + dx).clamp(0, n_grid - 1) * n_grid
                     + (b[:, 1] + dy).clamp(0, n_grid - 1)) * n_grid \
                    + (b[:, 2] + dz).clamp(0, n_grid - 1)
                acc = acc.index_add(0, i, w * m)
    return acc


def trilinear(X0n, c, R, K, dev, torch):
    """Each particle's eight corner nodes in the control cube, and its weight on each, from its
    MATERIAL coordinate. Returns (indices, weights), eight of each."""
    u = np.clip((X0n - (c - R)) / (2 * R) * (K - 1), 0, K - 1.001)
    b = np.floor(u).astype(int)
    f = u - b
    idx, wts = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (((1 - f[:, 0]) if dx == 0 else f[:, 0])
                     * ((1 - f[:, 1]) if dy == 0 else f[:, 1])
                     * ((1 - f[:, 2]) if dz == 0 else f[:, 2]))
                i = ((b[:, 0] + dx).clip(0, K - 1) * K + (b[:, 1] + dy).clip(0, K - 1)) * K \
                    + (b[:, 2] + dz).clip(0, K - 1)
                idx.append(torch.as_tensor(i, device=dev))
                wts.append(torch.as_tensor(w, dtype=torch.float32, device=dev))
    return idx, wts


# --------------------------------------------------------------------------------------------
# the render: a blue dielectric, not an opaque body
# --------------------------------------------------------------------------------------------
# These are VTK's physically-based-rendering path, NOT path tracing: OSPRay is unavailable in both
# the devcontainer and the cluster environment (`import vtkmodules.vtkRenderingRayTracing` raises
# ModuleNotFoundError -- checked, not assumed). A cloud of 50,000 opaque dots hides the silhouette a
# morph is judged on, which is why a surface is always reconstructed rather than points drawn.

_ENV_CACHE = {}


def default_settings():
    """A cold, saturated blue dielectric: opacity about 0.5, roughness low but not mirror-flat,
    sixteen depth-peeling layers so the far side of a concave body (an arm crossing in front of a
    torso) is not clipped, and a fourth BACKLIGHT so there is light on the far side to forward
    through the body at all -- glass with nothing behind it just looks like dark plastic."""
    return dict(
        ngrid=176, sigma=0.95, iso_frac=0.3, smooth_iter=35,
        color="#1a4fc0", target_color="#4a5a6a", ground_color="#04050a", bg="#05070a",
        roughness=0.07, opacity=0.5, diffuse=0.42, ambient=0.06,
        ior=1.4, coat_strength=0.85, coat_roughness=0.05, coat_ior=1.5, metallic=0.0,
        ssao_radius_frac=0.12, ssao_bias_frac=0.002, ssao_kernel=192, peels=16,
        env_bright=2.3, key_intensity=1.2, fill_intensity=0.32, rim_intensity=1.1,
        rim_color=(0.75, 0.85, 1.0), back_intensity=1.0, back_color=(0.6, 0.8, 1.0),
        # (elevation_deg, azimuth_deg) for pv.Light.set_direction_angle -- key and fill model the
        # form, rim and back separate the glass silhouette from the black background.
        key_angle=(55, -35), fill_angle=(35, 150), rim_angle=(-35, 20), back_angle=(15, 175),
    )


def reconstruct_contour(X, box, ngrid=176, sigma=0.95, iso_frac=0.3, smooth_iter=35):
    """The body as a contoured, gaussian-blurred density.

    THE TWO NUMBERS FIGHT EACH OTHER. A finer grid resolves thinner features, but fewer particles
    land in each voxel so the raw density is noisier -- measured: 96^3 with one 3-voxel box blur is
    clean but overly fused (legs merge into a wedge); 224^3 at the same blur width comes out
    pockmarked, a dimple wherever a voxel got one particle instead of three. 176^3 with a single
    gaussian of about one voxel is the balance point for these 50,000-particle clouds. Change the
    particle count and this number should move with it.
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


def env_texture(bright=2.3):
    """A synthetic equirectangular sky for image-based lighting -- there is no internet on the
    cluster node to fetch a real HDRI. A gradient from horizon navy to a lighter blue-teal zenith
    plus one soft bright patch, which is what a physically-based dielectric picks up as its
    directional highlight and its reflected colour. `bright` matters more than it looks like it
    should: at 1.0 a translucent, low-diffuse glass has nothing to catch and reads as near-black
    with a few white flecks rather than blue."""
    import pyvista as pv
    key = round(float(bright), 3)
    if key not in _ENV_CACHE:
        H, W = 128, 256
        yy = np.linspace(0, 1, H)[:, None] * np.ones((1, W))
        xx = np.linspace(0, 1, W)[None, :] * np.ones((H, 1))
        sky = np.stack([0.05 + 0.10 * yy, 0.07 + 0.14 * yy, 0.12 + 0.22 * yy], axis=-1) * bright
        sun = np.exp(-(((xx - 0.7) * W) ** 2 + ((yy - 0.15) * H) ** 2) / (2 * 22 ** 2))
        sky = sky + sun[..., None] * np.array([0.6, 0.6, 0.65]) * bright
        tex = pv.Texture((np.clip(sky, 0, 1) * 255).astype(np.uint8))
        # MIPMAPPING IS NOT COSMETIC HERE: without it every marching-cubes facet catches a raw,
        # aliased texel of the bright patch and the glass reads as covered in tiny white measles.
        tex.SetMipmap(True)
        tex.SetInterpolate(True)
        _ENV_CACHE[key] = tex
    return _ENV_CACHE[key]


def build_lights(pl, s):
    """Key, fill, rim -- plus a fourth, a BACKLIGHT, that a glass render needs and an opaque one
    does not: light arriving from behind the body is what a translucent dielectric forwards through
    itself, and without it there is nothing to be transparent to except the dim environment map."""
    import pyvista as pv
    for angle, intensity, color in ((s["key_angle"], s["key_intensity"], None),
                                    (s["fill_angle"], s["fill_intensity"], None),
                                    (s["rim_angle"], s["rim_intensity"], s["rim_color"]),
                                    (s["back_angle"], s["back_intensity"], s["back_color"])):
        light = pv.Light(light_type="scene light")
        light.set_direction_angle(*angle)
        light.intensity = intensity
        if color is not None:
            light.SetColor(*color)
        pl.add_light(light)


def draw_scene(pl, surf, target, box, X, s):
    """Everything but the plotter and the camera: the glass body, the faint target cloud behind it,
    four lights, image-based lighting, a small dark ground plane, and the domain wireframe."""
    import pyvista as pv
    pl.set_background(s["bg"])
    pl.set_environment_texture(env_texture(s["env_bright"]))
    pl.renderer.TexturedBackgroundOff()   # image-based lighting for reflections only; bg stays black

    pl.add_mesh(pv.PolyData(target.astype("float32")), color=s["target_color"],
                opacity=0.08, point_size=1.6)

    if surf is not None and surf.n_points > 0:
        pl.add_mesh(surf, color=s["color"], pbr=True, metallic=s["metallic"],
                    roughness=s["roughness"], opacity=s["opacity"], diffuse=s["diffuse"],
                    ambient=s["ambient"], smooth_shading=True)
        # THE DIELECTRIC ITSELF: an index of refraction and a thin, sharp clear-coat over the base
        # layer, which is what gives a low-roughness surface its glassy double highlight instead of
        # a single soft plastic one.
        prop = list(pl.renderer.actors.values())[-1].GetProperty()
        prop.SetBaseIOR(s["ior"])
        prop.SetCoatStrength(s["coat_strength"])
        prop.SetCoatRoughness(s["coat_roughness"])
        prop.SetCoatIOR(s["coat_ior"])
    else:
        pl.add_mesh(pv.PolyData(X.astype("float32")), color=s["color"], point_size=2.2)

    build_lights(pl, s)

    # THE GROUND IS PLAIN LAMBERTIAN, NOT PBR. A pbr=True plane under the same bright environment
    # texture that lights the glass turns into a lit grey studio floor filling half the frame --
    # fine for a product shot, wrong for this project's black-background convention.
    ymin = float(X[:, 1].min())
    ground = pv.Plane(center=(box / 2, ymin - 0.002, box / 2), direction=(0, 1, 0),
                      i_size=box * 1.05, j_size=box * 1.05)
    pl.add_mesh(ground, color=s["ground_color"], pbr=False, ambient=0.02, diffuse=0.5, specular=0.0)
    pl.add_mesh(pv.Box((0, box, 0, box, 0, box)), style="wireframe", color="#131a24",
                line_width=1.0, lighting=False)

    pl.enable_ssao(radius=box * s["ssao_radius_frac"], bias=box * s["ssao_bias_frac"],
                   kernel_size=s["ssao_kernel"])
    pl.enable_depth_peeling(number_of_peels=s["peels"], occlusion_ratio=0.0)
    pl.enable_anti_aliasing("ssaa")


def camera_for(box):
    # ASYMMETRIC ON PURPOSE. A camera at equal x and z offsets from a cubic domain puts the domain
    # box's own far vertical edge exactly behind the body's centreline -- a bright seam straight
    # down every frame that has nothing to do with the body.
    return [(box * 2.5, box * 1.4, box * 1.65), (box / 2,) * 3, (0, 1, 0)]


def render_movie(out_dir, frames, target, box, name, settings=None, px=1100, fps=20, n_stills=7):
    """The morph as movie.mp4 plus `n_stills` evenly spaced pngs."""
    import pyvista as pv
    import imageio.v3 as iio
    s = settings or default_settings()
    pv.OFF_SCREEN = True
    os.makedirs(out_dir, exist_ok=True)
    still_at = {int(round(v)) for v in np.linspace(0, len(frames) - 1, n_stills)}
    imgs, t_recon, t_draw = [], 0.0, 0.0
    for i, Xf in enumerate(frames):
        X = Xf.astype(np.float64)
        t0 = time.time()
        surf = reconstruct_contour(X, box, s["ngrid"], s["sigma"], s["iso_frac"], s["smooth_iter"])
        t_recon += time.time() - t0
        pl = pv.Plotter(off_screen=True, window_size=(px, px), lighting="none")
        draw_scene(pl, surf, target, box, X, s)
        pl.camera_position = camera_for(box)
        pl.add_text(f"{name}   frame {i}/{len(frames) - 1}   {len(X):,} material points",
                    position="upper_left", font_size=9, color="#c8d4e0")
        t0 = time.time()
        img = pl.screenshot(return_img=True)
        t_draw += time.time() - t0
        pl.close()
        imgs.append(img)
        if i in still_at:
            iio.imwrite(os.path.join(out_dir, f"still_{i:04d}.png"), img)
    iio.imwrite(os.path.join(out_dir, "movie.mp4"), imgs, fps=fps, codec="libx264",
                macro_block_size=None)
    n = max(len(frames), 1)
    return dict(n_frames=len(frames), px=px, fps=fps, s_per_frame_reconstruct=t_recon / n,
                s_per_frame_draw=t_draw / n, s_per_frame_total=(t_recon + t_draw) / n)


# --------------------------------------------------------------------------------------------
# the optimisation
# --------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="cow,armadillo",
                    help="comma list of .obj basenames in papers/morph_models")
    ap.add_argument("--n-pts", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=20, help="frames the control is OPTIMISED over")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--ctrl", type=int, default=12, help="control nodes per axis")
    ap.add_argument("--lr", type=float, default=3.0)
    ap.add_argument("--n-grid", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(ROOT, "graphs_data", "si_material"))
    ap.add_argument("--name", default=None, help="output folder name; defaults to morph_<target>")
    ap.add_argument("--px", type=int, default=1100)
    ap.add_argument("--render-frames", type=int, default=0,
                    help="frames in the FINAL rollout, the one that becomes the movie. The rate is "
                         "divided by the same factor, so the total deformation is identical and "
                         "only the sampling changes -- a morph optimised over 20 frames and "
                         "rendered at 400 is the same trajectory seen twenty times more finely, "
                         "not twenty times more of it.")
    ap.add_argument("--stages", default="",
                    help="coarse-to-fine, as `pts:grid:iters,pts:grid:iters,...`. The control is a "
                         "function of MATERIAL coordinates, so it transfers between stages exactly "
                         "-- a cheap stage buys the parameters a dear one would have.")
    ap.add_argument("--vol-weight", type=float, default=0.0,
                    help="penalise the REALISED volume of the rollout, sum(p_vol * det F), against "
                         "its value at the start. Not the control's trace: constraining that "
                         "kinematically makes volume worse, because the loss is lost through the "
                         "dynamics the kinematic proxy cannot see. Differentiated end to end, "
                         "which is the channel the mass loss is blind to.")
    ap.add_argument("--trace-free", action="store_true",
                    help="project the rate onto its deviatoric part. Measured to make realised "
                         "volume WORSE, not better -- see the module docstring. Kept to be rerun, "
                         "not to be used.")
    ap.add_argument("--control-dof", type=int, default=6, choices=[6, 9],
                    help="components per control node: 6 a symmetric rate (pure stretch), 9 the "
                         "full 3x3 (stretch plus rotation). Measured to be a tie on every target.")
    ap.add_argument("--control", default="grid", choices=["grid", "ngp"],
                    help="`grid` is a cube of rates read trilinearly. `ngp` is the multiresolution "
                         "hash encoding plus a small head, A(X0) = MLP(hash(X0)); it diverged.")
    args = ap.parse_args()

    import torch
    # TF32 is free and changes nothing: 1.002 -> 0.990 s an iteration at 12,500 particles on an L4,
    # inside that sweep's own noise, with the final loss moving 0.015% relative.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from . import engine, operators  # noqa: F401  -- importing operators registers them
    from .schema import load

    BOX, C, R = 0.35, 0.175, 0.055
    DOF = args.control_dof
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)
    c = np.array([C, C, C])
    os.makedirs(args.out, exist_ok=True)

    def build(raw):
        f = os.path.join(tempfile.mkdtemp(prefix="morph_"), "spec.yaml")
        yaml.safe_dump(raw, open(f, "w"), sort_keys=False)
        return load(f)

    def ngp_control():
        from .models.hashgrid import MultiResHashGrid
        enc = MultiResHashGrid(n_input_dims=3, n_levels=8, n_features_per_level=2,
                               log2_hashmap_size=15, base_resolution=4,
                               per_level_scale=1.6).to(dev)
        head = torch.nn.Sequential(torch.nn.Linear(8 * 2, 32), torch.nn.GELU(),
                                   torch.nn.Linear(32, DOF)).to(dev)
        with torch.no_grad():                        # start from no deformation at all
            head[-1].weight.mul_(0.0); head[-1].bias.mul_(0.0)
        return enc, head

    stages = ([tuple(int(v) for v in st.split(":")) for st in args.stages.split(",")]
              if args.stages else [(args.n_pts, args.n_grid, args.iters)])

    for name in [t.strip() for t in args.targets.split(",") if t.strip()]:
        path = os.path.join(MODELS, f"{name}.obj")
        theta = enc = head = None
        t_target = time.time()
        print(f"\n  {name}: {len(stages)} stage(s), control {args.control}, dof {DOF}", flush=True)

        for si, (n_pts, n_grid, iters) in enumerate(stages):
            sim = build(spec(n_pts, BOX, n_grid, args.frames, 0.002, 3.4e-4))
            X0n = ball(n_pts, c, R, rng)
            X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
            one = torch.ones(n_pts, device=dev)
            tgt_np = sample_inside(path, n_pts, c, 2.6 * R, rng)
            tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
            with torch.no_grad():
                rho_t = mass_grid(tgt, one, BOX, n_grid, dev, torch)

            if args.control == "ngp":
                Xm = torch.as_tensor(np.clip((X0n - (c - R)) / (2 * R), 0, 1),
                                     dtype=torch.float32, device=dev)
                if enc is None:
                    enc, head = ngp_control()
                params = list(enc.parameters()) + list(head.parameters())
            else:
                idx, wts = trilinear(X0n, c, R, args.ctrl, dev, torch)
                if theta is None:
                    theta = torch.zeros(args.ctrl ** 3, DOF, device=dev, requires_grad=True)
                params = [theta]

            opt = torch.optim.Adam(params, lr=args.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1),
                                                              eta_min=args.lr / 20.0)
            eye = torch.eye(3, device=dev)
            dt = float(sim.dt)
            t_stage = time.time()

            def rollout(keep=None):
                a = (head(enc(Xm)) if args.control == "ngp"
                     else sum(w[:, None] * theta[i] for i, w in zip(idx, wts)))
                if DOF == 9:
                    A = a.reshape(-1, 3, 3)
                else:
                    A = torch.zeros(a.shape[0], 3, 3, device=dev) + torch.diag_embed(a[:, :3])
                    A[:, 0, 1] = A[:, 1, 0] = a[:, 3]
                    A[:, 0, 2] = A[:, 2, 0] = a[:, 4]
                    A[:, 1, 2] = A[:, 2, 1] = a[:, 5]
                if args.trace_free:
                    A = A - torch.diagonal(A, dim1=1, dim2=2).mean(1)[:, None, None] * eye
                Adt = -A * dt
                G = eye + Adt + 0.5 * (Adt @ Adt)   # 2nd order: A dt is small, and 40x cheaper
                # THE INITIAL CONDITION GOES IN THROUGH `on_frame`, because `engine.run` builds and
                # seeds its own hierarchy: a ball written into a hierarchy built here is discarded
                # and the run starts from whatever the spec's `block` says -- measured once as a
                # 7 um cube where an 11 um ball was intended, with a loss that looked plausible.
                def cb(Hh, tick, G=G):
                    q = Hh.level("mpm_particle")
                    if tick == 0:
                        q0, q1 = q.state_schema["pos"]
                        with torch.no_grad():
                            q.state[:, q0:q1] = X0
                    else:
                        q.F = torch.bmm(G, q.F)     # functionally, so the tape survives
                    if keep is not None:
                        keep.append(q.get("pos").detach().cpu().numpy().copy())
                Hh, _ = engine.run(sim, device=args.device, on_frame=cb, progress=False,
                                   grad=(keep is None))
                q = Hh.level("mpm_particle")
                return q.get("pos"), (q.p_vol * torch.linalg.det(q.F)).sum()

            vol0 = None
            for it in range(iters):
                opt.zero_grad()
                Xs, vol = rollout()
                if vol0 is None:
                    vol0 = float(vol.detach())
                loss = ((torch.log1p(mass_grid(Xs, one, BOX, n_grid, dev, torch))
                         - torch.log1p(rho_t)) ** 2).mean()
                # THE MASS LOSS ALONE IS GAMEABLE. It supervises where material IS, not how much of
                # it there is, so the cheapest way to match a dense target is to implode -- measured
                # at max|J - 1| = 2.33, which is J crossing zero into inverted elements.
                if args.vol_weight > 0.0:
                    loss = loss + args.vol_weight * (vol / vol0 - 1.0) ** 2
                loss.backward()
                opt.step(); sched.step()
                if it % 20 == 0 or it == iters - 1:
                    e = ((Xs.max(0).values - Xs.min(0).values) * 100).tolist()
                    et = ((tgt.max(0).values - tgt.min(0).values) * 100).tolist()
                    # VOLUME IS PRINTED BESIDE THE EXTENT because a 55% collapse was invisible in
                    # every number this line used to carry: the shape can approach the target while
                    # the body it is made of quietly disappears.
                    print(f"    stage {si} ({n_pts:,} pts, grid {n_grid}) iter {it:3d}  "
                          f"loss {float(loss):.6f}  vol {float(vol) / vol0:.3f}x  "
                          f"shape {e[0]:5.1f} x {e[1]:5.1f} x {e[2]:5.1f}  "
                          f"target {et[0]:5.1f} x {et[1]:5.1f} x {et[2]:5.1f} um", flush=True)
            print(f"    stage {si}: {iters} iterations in {(time.time()-t_stage)/60:.1f} min "
                  f"({(time.time()-t_stage)/max(iters,1):.2f} s an iteration)", flush=True)

        # THE FINAL ROLLOUT NEEDS NO TAPE, so it should not pay for one. `differentiable=False`
        # lets the engine select warp: 3.6x at 12,500 particles on an L4 and 5.9x at 50,000 on an
        # A100 against the already-compiled torch path, with positions agreeing to 1.5e-07.
        with torch.no_grad():
            frames = []
            rf = args.render_frames or args.frames
            sim = build(spec(n_pts, BOX, n_grid, rf, 0.002, 3.4e-4, differentiable=False))
            dt = float(sim.dt) * args.frames / rf    # same total deformation, finer sampling
            rollout(keep=frames)
        frames = np.stack(frames)

        d = os.path.join(args.out, args.name or f"morph_{name}")
        os.makedirs(d, exist_ok=True)
        ctrl = (theta.detach().cpu().numpy() if theta is not None
                else np.concatenate([q.detach().cpu().numpy().ravel()
                                     for q in list(enc.parameters()) + list(head.parameters())]))
        # THE CONTROL WITHOUT ITS SHAPE IS NOT A CONTROL. The array alone does not say how many
        # nodes per axis it has nor which family it came from, so nothing downstream can rebuild the
        # rollout from it. Comparing two runs that lacked this block, as if they differed in only
        # one respect, is how one wrong conclusion got made.
        meta = dict(control_kind=args.control, ctrl_K=(args.ctrl if theta is not None else 0),
                    control_dof=DOF, vol_weight=args.vol_weight, trace_free=bool(args.trace_free),
                    iters=args.iters, lr=args.lr, stages=args.stages,
                    extent=[C, C, C, R], opt_frames=args.frames, render_frames=len(frames),
                    dt=0.002, n_pts=n_pts, n_grid=n_grid, target=name,
                    seconds_optimise=time.time() - t_target)
        np.savez_compressed(os.path.join(d, "morph.npz"), frames=frames, target=tgt_np,
                            box=BOX, control=ctrl, meta=np.array(json.dumps(meta)))
        t0 = time.time()
        stats = render_movie(d, frames, tgt_np, BOX, name, px=args.px)
        meta["render"] = stats
        meta["seconds_total"] = time.time() - t_target
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2, sort_keys=True)
        print(f"    -> {os.path.relpath(d, ROOT)}  ({len(frames)} frames, "
              f"{stats['s_per_frame_total']:.2f} s a frame to render, "
              f"{(time.time() - t_target) / 60:.1f} min total)", flush=True)


if __name__ == "__main__":
    main()
