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
import json
import os
import sys
import tempfile
import time

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


def render(out_dir, frames, target, box, name, px=1100, glass=True):
    """The morph as an mp4 and stills -- the blue dielectric the user approved.

    DELEGATES TO tools/morph_render.py, which the RENDER agent built and which is the reference
    look: a real dielectric (an index of refraction, sixteen depth peels so the far surface reads
    through the near one, image-based lighting from a synthetic environment, screen-space ambient
    occlusion, and a cool backlight that gives the body light to carry through itself) in place of
    the opaque amber this function used to draw. `glass` is accepted for call-site compatibility
    and no longer branches: a surface is always reconstructed, because a cloud of 50,000 opaque
    dots hides the silhouette a morph is judged on.

    OSPRAY IS NOT AVAILABLE HERE and these images are therefore VTK's PBR path, not path-traced --
    checked, not assumed: `import vtkmodules.vtkRenderingRayTracing` raises ModuleNotFoundError in
    both the devcontainer and the cluster environment. The settings live in
    graphs_data/si_material/render/results.json under `_approved_look`, and
    tools/morph_render.py's docstring records what was tried and rejected.
    """
    import morph_render as R
    R.render_movie(out_dir, frames, target, box, name, R.default_settings(), px=px, fps=20,
                   stride=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="cow,armadillo")
    ap.add_argument("--n-pts", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--ctrl", type=int, default=4, help="control nodes per axis")
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--n-grid", type=int, default=40)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=os.path.join(ROOT, "graphs_data", "si_material"))
    ap.add_argument("--vol-weight", type=float, default=0.0,
                    help="penalise the REALISED volume of the rollout, sum(p_vol * det F) at the "
                         "end, against its value at the start. Not the control's trace: the "
                         "MORE_PRECISE agent measured that the body loses 55% of its volume "
                         "(ratio 0.448) and that constraining the control kinematically makes it "
                         "WORSE (trace-free 0.233, a trace-based penalty 0.243), because the "
                         "volume is lost through the physical dynamics that the kinematic proxy "
                         "cannot see. This term is differentiated end to end through the rollout, "
                         "which is the channel the loss was blind to.")
    ap.add_argument("--render-frames", type=int, default=0,
                    help="frames in the FINAL rollout, the one that becomes the movie. The rate is "
                         "divided by the same factor, so the total deformation is identical and "
                         "only the sampling changes -- a morph optimised over 20 frames rendered at "
                         "100 is the same trajectory seen five times more finely, not five times "
                         "more of it.")
    ap.add_argument("--control", default="grid", choices=["grid", "ngp"],
                    help="`grid`: a cube of rates, trilinear. `ngp`: the multiresolution hash "
                         "encoding this repo already has (plexus.models.hashgrid) plus a small "
                         "head -- A(X0) = MLP(hash(X0)). Multi-resolution BY CONSTRUCTION, so the "
                         "coarse levels find the silhouette in the first iterations and the fine "
                         "ones add detail later, without anyone choosing a grid size.")
    ap.add_argument("--stages", default="",
                    help="coarse-to-fine, as `pts:grid:iters,pts:grid:iters,...`. The control is a "
                         "function of MATERIAL coordinates, so it transfers between stages exactly "
                         "-- a cheap stage buys the same parameters a dear one would have.")
    ap.add_argument("--batch", action="store_true",
                    help="optimise every target in ONE rollout, each in its own corner of a larger "
                         "box. Measured: a differentiable rollout costs 98.7 ms/frame at 12,500 "
                         "particles and 66.8 at 50,000 -- four times the work for 0.68 times the "
                         "time -- so below about 50,000 the solve is paying fixed per-substep "
                         "overhead and the extra trajectories ride along for free.")
    args = ap.parse_args()

    import torch
    # TF32 costs nothing and changes nothing: measured 1.002 -> 0.990 s per optimisation iteration
    # at 12,500 particles on an L4 (a 1.2% gain, inside that sweep's own run-to-run noise), with the
    # final loss moving 0.015% relative -- 0.0013540 to 0.0013542 over 60 iterations from the same
    # seed. Kept because it is free, not because it is a lever.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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

    # THE CONTROL, EITHER WAY, IS A FUNCTION OF THE MATERIAL COORDINATE -- the particle's position
    # at t = 0, never its current one. Lagrangian: a point carries its own instruction as it moves.
    # That is also what lets a coarse-to-fine schedule work at all, since the parameters mean the
    # same thing at 5,000 points as at 50,000.
    def ngp_control(dev):
        from plexus.models.hashgrid import MultiResHashGrid
        grid = MultiResHashGrid(n_input_dims=3, n_levels=8, n_features_per_level=2,
                                log2_hashmap_size=15, base_resolution=4,
                                per_level_scale=1.6).to(dev)
        head = torch.nn.Sequential(torch.nn.Linear(8 * 2, 32), torch.nn.GELU(),
                                   torch.nn.Linear(32, 6)).to(dev)
        with torch.no_grad():                        # start from no deformation at all
            head[-1].weight.mul_(0.0); head[-1].bias.mul_(0.0)
        return grid, head

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
    # COARSE TO FINE. `pts:grid:iters` per stage; the default is a single stage at the flags given.
    # A stage is cheaper than the next one in BOTH of the things that cost -- particles and grid
    # nodes -- and buys parameters the next stage starts from, because the control is a function of
    # the material coordinate and means the same thing at any resolution.
    stages = ([tuple(int(v) for v in st.split(":")) for st in args.stages.split(",")]
              if args.stages else [(args.n_pts, args.n_grid, args.iters)])

    for name in [t.strip() for t in args.targets.split(",") if t.strip()]:
        path = os.path.join(MODELS, f"{name}.obj")
        theta = grid_enc = head = None
        t_target = time.time()
        print(f"\n  {name}: {len(stages)} stage(s), control {args.control}", flush=True)
        for si, (n_pts, n_grid, iters) in enumerate(stages):
            raw = spec(n_pts, BOX, n_grid, args.frames, 0.002, 3.4e-4)
            f = os.path.join(tempfile.mkdtemp(prefix="morph_"), "spec.yaml")
            yaml.safe_dump(raw, open(f, "w"), sort_keys=False)
            sim = load(f)
            X0n = ball(n_pts, c, R, rng)
            X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
            one = torch.ones(n_pts, device=dev)
            tgt_np = sample_inside(path, n_pts, c, 2.6 * R, rng)
            tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
            with torch.no_grad():
                rho_t = mass_grid(tgt, one, BOX, n_grid, dev, torch)
            Xm = torch.as_tensor(np.clip((X0n - (c - R)) / (2 * R), 0, 1),
                                 dtype=torch.float32, device=dev)

            if args.control == "ngp":
                if grid_enc is None:
                    grid_enc, head = ngp_control(dev)
                params = list(grid_enc.parameters()) + list(head.parameters())
            else:
                K = args.ctrl
                u = np.clip((X0n - (c - R)) / (2 * R) * (K - 1), 0, K - 1.001)
                bb = np.floor(u).astype(int); ff = u - bb
                idx, wts = [], []
                for dx in (0, 1):
                    for dy in (0, 1):
                        for dz in (0, 1):
                            w = (((1 - ff[:, 0]) if dx == 0 else ff[:, 0])
                                 * ((1 - ff[:, 1]) if dy == 0 else ff[:, 1])
                                 * ((1 - ff[:, 2]) if dz == 0 else ff[:, 2]))
                            i = ((bb[:, 0] + dx).clip(0, K - 1) * K
                                 + (bb[:, 1] + dy).clip(0, K - 1)) * K + (bb[:, 2] + dz).clip(0, K - 1)
                            idx.append(torch.as_tensor(i, device=dev))
                            wts.append(torch.as_tensor(w, dtype=torch.float32, device=dev))
                if theta is None:
                    theta = torch.zeros(K ** 3, 6, device=dev, requires_grad=True)
                params = [theta]

            opt = torch.optim.Adam(params, lr=args.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1),
                                                               eta_min=args.lr / 20.0)
            eye = torch.eye(3, device=dev)
            dt = float(sim.dt)
            t_stage = time.time()

            def rates():
                if args.control == "ngp":
                    return head(grid_enc(Xm))
                return sum(w[:, None] * theta[i] for i, w in zip(idx, wts))

            def rollout(keep=None):
                a6 = rates()
                A = torch.zeros(a6.shape[0], 3, 3, device=dev) + torch.diag_embed(a6[:, :3])
                A[:, 0, 1] = A[:, 1, 0] = a6[:, 3]
                A[:, 0, 2] = A[:, 2, 0] = a6[:, 4]
                A[:, 1, 2] = A[:, 2, 1] = a6[:, 5]
                Adt = -A * dt
                G = eye + Adt + 0.5 * (Adt @ Adt)     # 2nd order: A dt is small, and 40x cheaper
                def cb(Hh, tick, G=G):
                    q = Hh.level("mpm_particle")
                    if tick == 0:
                        q0, q1 = q.state_schema["pos"]
                        with torch.no_grad():
                            q.state[:, q0:q1] = X0
                    else:
                        q.F = torch.bmm(G, q.F)
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
                # it there is, so the cheapest way to match a dense target is to implode -- measured,
                # max |J - 1| reaching 2.33, which is J crossing zero into inverted elements. This
                # term is on the volume the rollout ACTUALLY realised.
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

        with torch.no_grad():
            frames = []
            if args.render_frames and args.render_frames != args.frames:
                rf = args.render_frames
                raw = spec(n_pts, BOX, n_grid, rf, 0.002, 3.4e-4)
                fr = os.path.join(tempfile.mkdtemp(prefix="render_"), "spec.yaml")
                yaml.safe_dump(raw, open(fr, "w"), sort_keys=False)
                sim = load(fr)
                dt = float(sim.dt) * args.frames / rf     # same total deformation, finer sampling
            _Xf, _volf = rollout(keep=frames)
        d = os.path.join(args.out, f"morph_{name}")
        os.makedirs(d, exist_ok=True)
        ctrl = (theta.detach().cpu().numpy() if theta is not None
                else np.concatenate([q.detach().cpu().numpy().ravel()
                                     for q in list(grid_enc.parameters()) + list(head.parameters())]))
        # THE CONTROL WITHOUT ITS SHAPE IS NOT A CONTROL. The array alone does not say how many
        # nodes per axis it has, nor which family it came from, so nothing downstream can rebuild
        # the rollout from it -- which is exactly what the render agent could not do. Saved with
        # everything needed to reproduce this morph: the kind, K, the cube's extent in world
        # coordinates, the rollout the control was optimised for, and the material.
        meta = dict(control_kind=args.control, ctrl_K=(args.ctrl if theta is not None else 0),
                    extent=[C, C, C, R], opt_frames=args.frames, render_frames=len(frames),
                    dt=0.002, n_pts=n_pts, n_grid=n_grid, target=name,
                    ngp=dict(n_levels=8, n_features_per_level=2, log2_hashmap_size=15,
                             base_resolution=4, per_level_scale=1.6, head=[32, 6]))
        np.savez_compressed(os.path.join(d, "morph.npz"), frames=np.stack(frames),
                            target=tgt_np, box=BOX, control=ctrl,
                            seconds=time.time() - t_target,
                            meta=np.array(json.dumps(meta)))
        render(d, np.stack(frames), tgt_np, BOX, name)
        print(f"    -> {os.path.relpath(d, ROOT)}  ({len(frames)} frames, "
              f"{(time.time() - t_target) / 60:.1f} min total)", flush=True)


if __name__ == "__main__":
    main()
