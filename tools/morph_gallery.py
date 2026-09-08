"""Morph a ball into a named shape by optimising a FIELD of deformation-gradient rates.

This is `tools/shape_control.py` with the control enlarged, which is the one axis that decides what
a morph can reach. Six global numbers are a single linear rate: they can flatten a ball into a disc
or draw it into a column -- measured, both, in opposite directions -- and they cannot put a leg on
it, because every material point gets the same instruction. The published method
(papers/Xu_2024_mpm_shape_morphing.pdf) optimises a control PER PARTICLE at many timesteps. Here the
control is a coarse grid of symmetric rates, read at each particle's MATERIAL coordinate:

    A_p = sum_i w_i(X0_p) A_i           w = trilinear weights on a `--ctrl` cube of nodes
    F_p <- expm(-A_p dt) F_p            once a frame, as `deform_control` does globally

Lagrangian on purpose: the weights are computed once from the particle's position at t = 0 and never
again, so a point carries its own instruction with it as it moves. An Eulerian field would instruct
whatever happens to be passing through a place, which is a different (and much harder) thing to
optimise.

    loss    log nodal mass on the grid, the Eulerian loss the paper argues for: a target shape has
            no particle correspondence, so a position loss has nothing to match against exactly
            where the two shapes differ. `log` because the fields differ by orders of magnitude
            between "material" and "none".
    target  points sampled INSIDE a mesh, by ray parity against the surface (pyvista's
            `select_enclosed_points`), scaled into the box.

WHAT TO EXPECT, said before the run rather than after: a 4x4x4 control grid is 384 numbers against
the paper's per-particle-per-timestep control, and it reaches the SILHOUETTE of a shape -- its
bounding proportions and its large lobes -- not its ears. The number to read is the loss and the
extent, both printed; the pictures are what they are.

    PYTHONPATH=src python tools/morph_gallery.py --targets cow,armadillo --iters 60
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
MODELS = os.path.join(ROOT, "papers", "morph_models")


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
        pc = pv.PolyData(q)
        sel = pc.select_enclosed_points(m, tolerance=0.0, check_surface=False)
        keep = q[np.asarray(sel["SelectedPoints"]) > 0]
        if len(keep) == 0:
            raise ValueError(f"{path}: no interior points found -- is the surface closed?")
        out.append(keep)
    P = np.concatenate(out)[:n]
    return (P - ctr) * scale + box_c


def render(out_dir, frames, target, box, name, px=1100):
    """The morph as an mp4 and stills, drawn where it happened rather than as a table of numbers.

    THE MATERIAL'S OWN COLOUR, not a height ramp. A default renderer colours a point cloud by its
    initial height, which is a useful material TAG when you want to see where material went and a
    misleading one here: the question is what shape the elastic body took, and the body is one
    material. Orange is this repo's elastic; the target is drawn once, faint, behind it.
    """
    import pyvista as pv
    import imageio.v3 as iio
    pv.OFF_SCREEN = True
    ORANGE, GREY = "#ff8c1a", "#3a4a5a"
    imgs = []
    for i, X in enumerate(frames):
        pl = pv.Plotter(off_screen=True, window_size=(px, px))
        pl.set_background("black")
        pl.add_mesh(pv.PolyData(target.astype("float32")), color=GREY, opacity=0.10,
                    point_size=2.0, render_points_as_spheres=False)
        pl.add_mesh(pv.PolyData(X.astype("float32")), color=ORANGE, point_size=2.2,
                    render_points_as_spheres=False)
        pl.add_mesh(pv.Box((0, box, 0, box, 0, box)), style="wireframe", color="#555555",
                    line_width=1.0, lighting=False)
        pl.camera_position = [(box * 2.6, box * 1.5, box * 2.6), (box / 2,) * 3, (0, 1, 0)]
        pl.add_text(f"{name}   frame {i}/{len(frames) - 1}   {len(X):,} material points\n"
                    f"orange: the cell   grey: the target",
                    position="upper_left", font_size=9, color="white")
        img = pl.screenshot(return_img=True)
        pl.close()
        imgs.append(img)
        if i in (0, len(frames) // 2, len(frames) - 1):
            iio.imwrite(os.path.join(out_dir, f"still_{i:03d}.png"), img)
    iio.imwrite(os.path.join(out_dir, "movie.mp4"), imgs, fps=8, codec="libx264",
                macro_block_size=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="cow,armadillo")
    ap.add_argument("--n-pts", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--ctrl", type=int, default=4, help="control nodes per axis")
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--n-grid", type=int, default=40)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=os.path.join(ROOT, "graphs_data", "si_material"))
    args = ap.parse_args()

    import torch
    import plexus.operators  # noqa: F401
    from plexus.schema import load
    from plexus import engine
    from shape_control import spec, ball, mass_grid

    BOX, C, R = 0.35, 0.175, 0.055
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)
    raw = spec(args.n_pts, BOX, args.n_grid, args.frames, 0.002, 2.5e-4)
    tmp = os.path.join(tempfile.mkdtemp(prefix="morph_"), "spec.yaml")
    yaml.safe_dump(raw, open(tmp, "w"), sort_keys=False)
    sim = load(tmp)
    c = np.array([C, C, C])
    X0n = ball(args.n_pts, c, R, rng)
    X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
    one = torch.ones(args.n_pts, device=dev)

    # trilinear weights of every particle in the CONTROL cube, from its MATERIAL coordinate
    K = args.ctrl
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

    os.makedirs(args.out, exist_ok=True)
    for name in [t.strip() for t in args.targets.split(",") if t.strip()]:
        path = os.path.join(MODELS, f"{name}.obj")
        tgt_np = sample_inside(path, args.n_pts, c, 2.6 * R, rng)
        tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
        with torch.no_grad():
            rho_t = mass_grid(tgt, one, BOX, args.n_grid, dev, torch)
        theta = torch.zeros(K ** 3, 6, device=dev, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=args.lr)
        # A DECAYING STEP, because the two halves of this optimisation want different ones. The
        # first iterations move a ball most of the way to a silhouette and want a large step; the
        # last ones are placing a limb against a target that is already nearly met, and the same
        # step overshoots it every time -- the loss stops falling and starts rattling. Cosine from
        # `lr` to a twentieth of it over the run.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters,
                                                           eta_min=args.lr / 20.0)
        eye = torch.eye(3, device=dev)
        dt = float(sim.dt)
        print(f"\n  {name}: {args.n_pts:,} points, control {K}^3 x 6 = {K**3*6} numbers, "
              f"{args.iters} iterations", flush=True)

        for it in range(args.iters):
            opt.zero_grad()
            a = sum(w[:, None] * theta[i] for i, w in zip(idx, wts))      # [N,6] per particle
            A = torch.zeros(args.n_pts, 3, 3, device=dev)
            A = A + torch.diag_embed(a[:, :3])
            A[:, 0, 1] = A[:, 1, 0] = a[:, 3]
            A[:, 0, 2] = A[:, 2, 0] = a[:, 4]
            A[:, 1, 2] = A[:, 2, 1] = a[:, 5]
            # SECOND ORDER, NOT `matrix_exp`, on [N,3,3]: A dt is small by construction and the
            # series is exact to the order that matters, differentiable and ~40x cheaper.
            Adt = -A * dt
            G = eye + Adt + 0.5 * (Adt @ Adt)

            def on_frame(Hh, tick, G=G):
                q = Hh.level("mpm_particle")
                if tick == 0:
                    p0, p1 = q.state_schema["pos"]
                    with torch.no_grad():
                        q.state[:, p0:p1] = X0
                    return
                q.F = torch.bmm(G, q.F)

            Hh, _ = engine.run(sim, device=args.device, on_frame=on_frame, progress=False,
                               grad=True)
            Xs = Hh.level("mpm_particle").get("pos")
            loss = ((torch.log1p(mass_grid(Xs, one, BOX, args.n_grid, dev, torch))
                     - torch.log1p(rho_t)) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
            if it % 10 == 0 or it == args.iters - 1:
                e = ((Xs.max(0).values - Xs.min(0).values) * 100).tolist()
                et = ((tgt.max(0).values - tgt.min(0).values) * 100).tolist()
                print(f"    iter {it:3d}  loss {float(loss):.6f}  "
                      f"lr {sched.get_last_lr()[0]:.3f}  "
                      f"shape {e[0]:5.1f} x {e[1]:5.1f} x {e[2]:5.1f}  "
                      f"target {et[0]:5.1f} x {et[1]:5.1f} x {et[2]:5.1f} um", flush=True)

        # the final rollout, kept frame by frame, and the target beside it
        with torch.no_grad():
            frames = []

            def keep(Hh, tick, G=G):
                q = Hh.level("mpm_particle")
                if tick == 0:
                    p0, p1 = q.state_schema["pos"]
                    q.state[:, p0:p1] = X0
                else:
                    q.F = torch.bmm(G, q.F)
                frames.append(q.get("pos").detach().cpu().numpy().copy())

            engine.run(sim, device=args.device, on_frame=keep, progress=False)
        d = os.path.join(args.out, f"morph_{name}")
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "morph.npz"), frames=np.stack(frames),
                            target=tgt_np, control=theta.detach().cpu().numpy(), box=BOX)
        render(d, np.stack(frames), tgt_np, BOX, name)
        print(f"    -> {os.path.relpath(d, ROOT)}  ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
