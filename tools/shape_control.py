"""Make a cell take a shape by OPTIMISING its deformation gradient, not by pushing it with a wall.

THE CONTACT ROUTE DOES NOT DO THIS, and the reason is not a tuning failure. A contact is a boundary
condition: it acts only on material that has crossed the surface, so a cage leaves everything
already inside it untouched, and a cage that starts smaller than the cell cannot reach the material
that is far outside it. Measured both ways in `config/cell/squeeze_prism.yaml` and
`config/cell/grow_in_cage.yaml`.

The literature's answer is DEFORMATION-GRADIENT CONTROL: make the per-particle F the control
variable, let the solver carry the material, and take the loss on the GRID rather than on particle
positions -- a target shape has no particle correspondence, so a position loss has nothing to match
against and fails exactly where the shapes differ most. This is that, at its smallest:

    control   one 3x3 rate A, symmetric, six numbers. Every frame the rest shape changes by
              G = expm(-A dt), applied as F <- G F, which is the same multiplicative growth
              `polar_growth` writes -- generalised from "a stretch along the polarity" to "any
              linear rate", and now a PARAMETER rather than a constant of the specification.
    loss      the log nodal mass, mean over the grid of (log(1+rho_sim) - log(1+rho_target))^2.
              Log, because the two mass fields differ by orders of magnitude where one has material
              and the other has none, and a squared difference of raw masses is then dominated by
              the interior it already agrees about.
    gradient  through the whole rollout. `engine.run(grad=True)` keeps the tape, and falls back to
              the torch MPM bodies because the warp kernels register no backward.

WHAT THIS IS AND IS NOT. Six parameters and one rate can turn a ball into an ellipsoid; they cannot
put corners on it. The published method optimises per-particle controls at many timesteps, which is
the same loop with a bigger control vector -- this establishes that the loop CLOSES in Plexus (the
gradient exists, the loss falls, the shape moves) before spending anything on the size of it.

    PYTHONPATH=src python tools/shape_control.py --iters 40

Reference, both in papers/:
    Xu, M., Song, C. Y., Levin, D. I. W. & Hyde, D. (2025). A Differentiable Material Point Method
    Framework for Shape Morphing. IEEE TVCG 31(10):9140-9153. arXiv:2409.15746.
    -> papers/Xu_2024_mpm_shape_morphing.pdf
    Song, C. Y. & Hyde, D. (2025). PhysMorph-GS: render-guided volumetric morphing with
    differentiable physics.  -> papers/Song_2025_physmorph_gs.pdf
    Xu, M. & Levin, D. I. W. (2023). Deformation Gradient Control of Physically Simulated Amorphous
    Solids. SCA '23. doi 10.1145/3606037.3606840 (the two-page original; no open PDF).
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


def spec(n_pts, box, n_grid, frames, dt, sub):
    """The smallest model that can be asked this question: one material, one grid, no forces."""
    return dict(
        general=dict(name="shape_control", seed=0, n_frames=frames, dt=dt, record_cap=3,
                     boundary="wall", dim=3, world=[box, box, box], units=dict(length_um=100.0)),
        sets={"mpm_particle": dict(n=n_pts, types={"cyto": dict(
            fraction=1.0, youngs=90.0, density=1.0,
            block=[0.4 * box, 0.4 * box, 0.4 * box, 0.6 * box, 0.6 * box, 0.6 * box])})},
        fields=dict(mpm_grid=dict(frame="mpm_grid", n_grid=n_grid)),
        operators=[dict(op="mpm_strain", at="mpm_particle", implementation="default"),
                   dict(op="mpm_scatter", at="mpm_particle", to="mpm_grid", drag=0.5,
                        a_max=200.0, implementation="default"),
                   dict(op="mpm_grid_update", at="mpm_grid", wall_damp=0.9),
                   dict(op="mpm_gather", at="mpm_particle", **{"from": "mpm_grid"},
                        wall_damp=0.9, vmax=1.0e9, implementation="default")],
        schedule=[dict(substep_dt=sub, steps=["mpm_strain", "mpm_scatter", "mpm_grid_update",
                                              "mpm_gather"])],
        plotting={},
    )


def ball(n, c, r, rng):
    """n points uniform in a ball -- the cell this starts as."""
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return c + u * (r * rng.random((n, 1)) ** (1.0 / 3.0))


def target_points(kind, n, c, rng):
    """The shape being asked for. Three, because one is an anecdote: a morph that reaches a flat
    disc says nothing about whether the loop can also make a column, and the two need OPPOSITE
    signs on the same rate."""
    if kind == "disc":                       # squashed: wide and flat
        return prism(n, c, 0.075, 0.050, rng, sides=32)
    if kind == "column":                     # the other way: narrow and tall
        return prism(n, c, 0.035, 0.180, rng, sides=32)
    if kind == "prism":                      # the epithelial cell of this repo's own scenes
        return prism(n, c, 0.075, 0.050, rng, sides=6)
    if kind == "bar":                        # anisotropic in the plane: long in x, thin in z
        p = rng.random((n, 3)) * np.array([0.170, 0.070, 0.045]) - np.array([0.085, 0.035, 0.0225])
        return p + c
    raise ValueError(f"unknown target {kind!r}")


def prism(n, c, r, h, rng, sides=6):
    """n points uniform in a hexagonal prism -- the shape it is asked to become."""
    out = []
    while sum(len(o) for o in out) < n:
        p = rng.random((n, 3)) * np.array([2 * r, h, 2 * r]) - np.array([r, h / 2, r])
        a = np.arctan2(p[:, 2], p[:, 0])
        rad = np.linalg.norm(p[:, [0, 2]], axis=1)
        # inside a regular polygon: radius under the edge at this angle
        lim = r * np.cos(np.pi / sides) / np.cos((a % (2 * np.pi / sides)) - np.pi / sides)
        out.append(p[rad <= lim])
    return np.concatenate(out)[:n] + c


def mass_grid(X, m, box, n_grid, dev, torch):
    """The grid's own nodal mass, by the trilinear weights the solver would use."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pts", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.6)
    ap.add_argument("--n-grid", type=int, default=40)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--target", default="disc", choices=["disc", "column", "prism", "bar"])
    ap.add_argument("--write-spec", action="store_true",
                    help="write config/cell/morph_<target>.yaml with the optimised rate as a "
                         "`deform_control` operator, so the answer is a runnable model")
    args = ap.parse_args()

    import torch
    import plexus.operators  # noqa: F401
    from plexus.schema import load
    from plexus import engine

    BOX, C, R, HP = 0.35, 0.175, 0.055, 0.05
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)
    raw = spec(args.n_pts, BOX, args.n_grid, args.frames, 0.002, 2.5e-4)
    tmp = os.path.join(tempfile.mkdtemp(prefix="shapectl_"), "spec.yaml")
    yaml.safe_dump(raw, open(tmp, "w"), sort_keys=False)
    sim = load(tmp)

    c = np.array([C, C, C])
    X0 = torch.as_tensor(ball(args.n_pts, c, R, rng), dtype=torch.float32, device=dev)
    tgt = torch.as_tensor(target_points(args.target, args.n_pts, c, rng),
                          dtype=torch.float32, device=dev)

    # SIX NUMBERS: the symmetric part of a rate. A rotation would move the shape without changing
    # it, so the antisymmetric part is not a control and is not given one.
    theta = torch.zeros(6, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=args.lr)
    eye = torch.eye(3, device=dev)

    def rate(t):
        A = torch.zeros(3, 3, device=dev, dtype=t.dtype)
        A = A + torch.diag(t[:3])
        off = torch.zeros(3, 3, device=dev, dtype=t.dtype)
        off[0, 1] = off[1, 0] = t[3]; off[0, 2] = off[2, 0] = t[4]
        off[1, 2] = off[2, 1] = t[5]
        return A + off

    hist = []
    for it in range(args.iters):
        opt.zero_grad()
        G = torch.matrix_exp(-rate(theta) * float(sim.dt))    # the rest-shape change per frame

        # THE INITIAL CONDITION GOES IN THROUGH `on_frame`, because `engine.run` BUILDS AND SEEDS
        # ITS OWN HIERARCHY: a ball written into a hierarchy built here is simply discarded, and the
        # run starts from whatever the spec's `block` says -- measured, a 7 um cube where a 11 um
        # ball was intended, with a loss that looked plausible and meant nothing.
        def on_frame(Hh, tick, G=G):
            q = Hh.level("mpm_particle")
            if tick == 0:
                a, b = q.state_schema["pos"]
                with torch.no_grad():
                    q.state[:, a:b] = X0                      # start as the ball
                return
            q.F = torch.matmul(G, q.F)                        # F <- G F, functionally: keeps the tape

        Hh, _ = engine.run(sim, device=args.device, on_frame=on_frame, progress=False, grad=True)
        Xs = Hh.level("mpm_particle").get("pos")
        # UNIT WEIGHTS, NOT THE PARTICLE MASS. A material point here weighs about 1e-8, so log1p of
        # its nodal mass is 1e-8 too and the loss underflows to zero -- it read 0.000000 with a
        # gradient of 6e-17, which looks like convergence and is arithmetic. Counting particles per
        # node makes the field O(1) and the log do what it is there for.
        one = torch.ones(Xs.shape[0], device=dev, dtype=Xs.dtype)
        rho_s = mass_grid(Xs, one, BOX, args.n_grid, dev, torch)
        with torch.no_grad():
            rho_t = mass_grid(tgt, one, BOX, args.n_grid, dev, torch)
        loss = ((torch.log1p(rho_s) - torch.log1p(rho_t)) ** 2).mean()
        loss.backward()
        gn = float(theta.grad.norm()) if theta.grad is not None else float("nan")
        opt.step()
        ext = ((Xs.max(0).values - Xs.min(0).values) * 100).tolist()
        hist.append((it, float(loss), gn, ext))
        print(f"  iter {it:3d}  loss {float(loss):.6f}  |grad| {gn:.3e}  "
              f"shape {ext[0]:5.1f} x {ext[1]:5.1f} x {ext[2]:5.1f} um  "
              f"A diag {[round(float(v), 3) for v in theta[:3]]}", flush=True)
        if it == 0 and (not np.isfinite(gn) or gn == 0.0):
            print("\n  THE GRADIENT IS ZERO OR NOT FINITE: the tape does not survive the rollout, "
                  "and nothing below this line would mean anything. Stopping.", flush=True)
            break

    tw = ((tgt.max(0).values - tgt.min(0).values) * 100).tolist()
    print(f"\n  target {args.target}: {tw[0]:.1f} x {tw[1]:.1f} x {tw[2]:.1f} um; "
          f"loss {hist[0][1]:.6f} -> {hist[-1][1]:.6f}", flush=True)
    if args.write_spec:
        # THE OPTIMISATION'S ANSWER AS A MODEL, not as a number in a log: the rate goes into a
        # `deform_control` operator and the spec runs through Plexus_Main like anything else, so the
        # morph can be watched rather than believed.
        raw2 = spec(args.n_pts, BOX, args.n_grid, args.frames * 3, 0.002, 2.5e-4)
        raw2["general"]["name"] = f"morph_{args.target}"
        raw2["general"]["record_cap"] = 60
        raw2["sets"]["mpm_particle"]["types"]["cyto"]["block"] = [
            C - R, C - R, C - R, C + R, C + R, C + R]
        raw2["operators"] = [dict(op="deform_control", at="mpm_particle",
                                  rate=[round(float(v), 5) for v in theta.detach()],
                                  over=args.frames)] + raw2["operators"]
        raw2["schedule"] = ["deform_control"] + raw2["schedule"]
        raw2["plotting"] = dict(renderer="vtk_points", background="black", up_axis=1,
                                box_frame=True, render_3d="points", dot_size=1.6, fps=30,
                                slow_motion=2)
        f2 = os.path.join(ROOT, "config", "cell", f"morph_{args.target}.yaml")
        yaml.safe_dump(raw2, open(f2, "w"), sort_keys=False)
        print(f"  -> {os.path.relpath(f2, ROOT)}  "
              f"(rate {[round(float(v), 4) for v in theta.detach()]})", flush=True)


if __name__ == "__main__":
    main()
