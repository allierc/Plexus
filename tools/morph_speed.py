"""Speed only. Same physics, same loss, fewer seconds per optimiser iteration.

This is the differentiable MPM shape-morphing loop (`tools/shape_control.py`,
`tools/morph_gallery.py`) reduced to a stopwatch: build the identical rollout --
`implementation: differentiable` mpm_strain/mpm_scatter/mpm_grid_update/mpm_gather, a control
field read at each particle's material coordinate, F <- expm(-A_p dt) F once a frame, loss = log
nodal mass on the grid -- and time optimiser iterations under a grid of engineering knobs: TF32,
the polar-decomposition iteration count, torch.compile mode, grid resolution, and batching several
independent trajectories into one rollout.

NOTHING HERE CHANGES WHAT IS COMPUTED. Every knob is either a numerics setting already exposed by
the differentiable operators (`polar_iters`, `compile`/`compile_mode`, `dt_sub`, `n_grid`) or a
global torch precision flag (TF32, matmul precision) toggled from this process and reported
alongside the loss it produced, so a speed-up that silently moved the answer is visible as a
loss-column number rather than an assertion. Batching is the one case that changes the SHAPE of
the problem (several trajectories sharing one grid) rather than a precision knob; the world and
control field scale with the batch count so the physics and the per-particle volume are the ones
a single trajectory would get (see `_layout`).

    PYTHONPATH=src python tools/morph_speed.py --suite headline --device cuda:0

Results are appended, one experiment at a time, to `<out>/results.json` -- so a killed job leaves
whatever ran so far on disk rather than nothing.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

# batch count -> (cells along x, y, z). Kept close to a cube so the shared grid does not become a
# long thin slab, which would waste grid nodes on empty space between the balls.
_LAYOUT = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1), 8: (2, 2, 2), 16: (4, 2, 2)}

DEFAULTS = dict(
    n_pts=50000, n_grid=40, ctrl=12, frames=20, dt=0.002, substeps=8,
    compile="off", compile_mode="default", polar_iters=6, tf32=False,
    matmul_precision="highest", batch=1, lr=3.0, iters=10, warmup=3, seed=0,
)


def _spec(n_pts, world, n_grid, frames, dt, dt_sub, compile_mode, polar_iters):
    """The four differentiable MPM operators, the identical schedule shape_control.spec builds --
    a private copy so this file does not import (and cannot be blamed for editing) shape_control.py.
    """
    ops = [
        dict(op="mpm_strain", at="mpm_particle", implementation="differentiable"),
        dict(op="mpm_scatter", at="mpm_particle", to="mpm_grid", drag=0.5, a_max=200.0,
             implementation="differentiable", dt_sub=dt_sub, polar_iters=polar_iters),
        dict(op="mpm_grid_update", at="mpm_grid", wall_damp=0.9, implementation="differentiable"),
        dict(op="mpm_gather", at="mpm_particle", **{"from": "mpm_grid"}, wall_damp=0.9,
             vmax=1.0e9, implementation="differentiable"),
    ]
    sched = dict(substep_dt=dt_sub,
                 steps=["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"])
    if compile_mode and compile_mode != "off":
        sched["compile"] = True
        sched["compile_mode"] = compile_mode
        sched["compile_recompile_limit"] = 128
    wx, wy, wz = world
    # BLOCK SPANS THE WHOLE WORLD, at the same 0.4..0.6 fraction shape_control uses on a single
    # box. p_vol = block_volume / n, and this fraction scales block_volume with the world exactly
    # as batching scales n (see `_layout`), so the per-particle volume a single trajectory gets is
    # the one every trajectory gets here too -- batching changes wall-clock, not physics.
    block = [0.4 * wx, 0.4 * wy, 0.4 * wz, 0.6 * wx, 0.6 * wy, 0.6 * wz]
    return dict(
        general=dict(name="morph_speed", seed=0, n_frames=frames, dt=dt, record_cap=3,
                     boundary="wall", dim=3, world=[wx, wy, wz], units=dict(length_um=100.0)),
        sets={"mpm_particle": dict(n=n_pts, types={"cyto": dict(
            fraction=1.0, youngs=90.0, density=1.0, block=block)})},
        fields=dict(mpm_grid=dict(frame="mpm_grid", n_grid=n_grid)),
        operators=ops,
        schedule=[sched],
        plotting={},
    )


def _layout(batch):
    if batch not in _LAYOUT:
        raise ValueError(f"morph_speed: no layout for batch={batch}; add one to _LAYOUT")
    return _LAYOUT[batch]


def _mass_grid(X, m, world, n_grid, dev, torch):
    """The grid's own nodal mass, laid out exactly as `MPMGrid` lays out its nodes: dx =
    world[1]/n_grid, and nx/ny/nz = round(world[k]/dx) on each axis.

    `shape_control.mass_grid` assumes a CUBIC world (n_grid**3 nodes on every axis), which is
    exactly right for a single trajectory and silently wrong for a batched one: a 4-cell world is
    4x wider in x than it is in y, so clamping x into the same `n_grid` bins as y would clip every
    cell past the first one out of the loss entirely. Reproduced here (not imported) so the
    histogram always matches the node count the physics grid actually has; at batch=1 the world is
    cubic and this is arithmetically the same function.
    """
    dx = world[1] / n_grid
    nx = max(1, round(world[0] / dx))
    ny = max(1, round(world[1] / dx))
    nz = max(1, round(world[2] / dx))
    g = torch.stack([(X[:, 0] / dx).clamp(0, nx - 1.001),
                      (X[:, 1] / dx).clamp(0, ny - 1.001),
                      (X[:, 2] / dx).clamp(0, nz - 1.001)], dim=1)
    b = g.floor().long()
    f = g - b.float()
    acc = torch.zeros(nx * ny * nz, device=dev, dtype=X.dtype)
    for ddx in (0, 1):
        for ddy in (0, 1):
            for ddz in (0, 1):
                w = (((1 - f[:, 0]) if ddx == 0 else f[:, 0])
                     * ((1 - f[:, 1]) if ddy == 0 else f[:, 1])
                     * ((1 - f[:, 2]) if ddz == 0 else f[:, 2]))
                i = ((b[:, 0] + ddx).clamp(0, nx - 1) * ny
                     + (b[:, 1] + ddy).clamp(0, ny - 1)) * nz + (b[:, 2] + ddz).clamp(0, nz - 1)
                acc = acc.index_add(0, i, w * m)
    return acc


def _control_weights(X0n, lo, hi, K, dev, torch):
    """Trilinear weights of every particle in a KxKxK control cube -- the same construction
    `morph_gallery.py` uses, generalised to an arbitrary bounding box (`lo`, `hi`) so it also
    covers a multi-cell batched world."""
    u = np.clip((X0n - lo) / (hi - lo) * (K - 1), 0, K - 1.001)
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


def build_problem(cfg, dev, torch):
    """Everything an iteration needs: the compiled sim, the initial/target positions, the control
    parameters and a `rollout()` / `loss_fn()` pair. One call per experiment -- a fresh Spec and a
    fresh set of operator instances, so torch.compile's cache never straddles two configs."""
    from shape_control import ball, target_points

    cellbox = 0.35
    nx, ny, nz = _layout(cfg["batch"])
    world = (cellbox * nx, cellbox * ny, cellbox * nz)
    n_per = cfg["n_pts"]
    n_total = n_per * cfg["batch"]
    dt_sub = cfg["dt"] / cfg["substeps"]

    raw = _spec(n_total, world, cfg["n_grid"], cfg["frames"], cfg["dt"], dt_sub,
                cfg["compile"], cfg["polar_iters"])
    tmp = os.path.join(tempfile.mkdtemp(prefix="mspeed_"), "spec.yaml")
    yaml.safe_dump(raw, open(tmp, "w"), sort_keys=False)
    from plexus.schema import load
    sim = load(tmp)

    rng = np.random.default_rng(cfg["seed"])
    R = 0.055 / 0.35 * cellbox
    X0_parts, tgt_parts = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = np.array([(i + 0.5) * cellbox, (j + 0.5) * cellbox, (k + 0.5) * cellbox])
                X0_parts.append(ball(n_per, c, R, rng))
                tgt_parts.append(target_points("disc", n_per, c, rng))
    X0n = np.concatenate(X0_parts)
    tgt_np = np.concatenate(tgt_parts)
    X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
    tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
    one = torch.ones(n_total, device=dev)
    with torch.no_grad():
        rho_t = _mass_grid(tgt, one, world, cfg["n_grid"], dev, torch)

    K = cfg["ctrl"]
    lo = np.zeros(3); hi = np.array(world)
    idx, wts = _control_weights(X0n, lo, hi, K, dev, torch)
    theta = torch.zeros(K ** 3, 6, device=dev, requires_grad=True)
    eye = torch.eye(3, device=dev)
    dt = float(sim.dt)

    def rollout(grad, keep=None):
        a6 = sum(w[:, None] * theta[i] for i, w in zip(idx, wts))
        A = torch.zeros(a6.shape[0], 3, 3, device=dev) + torch.diag_embed(a6[:, :3])
        A[:, 0, 1] = A[:, 1, 0] = a6[:, 3]
        A[:, 0, 2] = A[:, 2, 0] = a6[:, 4]
        A[:, 1, 2] = A[:, 2, 1] = a6[:, 5]
        Adt = -A * dt
        G = eye + Adt + 0.5 * (Adt @ Adt)

        def cb(Hh, tick, G=G):
            q = Hh.level("mpm_particle")
            if tick == 0:
                q0, q1 = q.state_schema["pos"]
                with torch.no_grad():
                    q.state[:, q0:q1] = X0
            else:
                q.F = torch.bmm(G, q.F)
            if keep is not None:
                keep.append(q.get("pos").detach())

        from plexus import engine
        Hh, _ = engine.run(sim, device=str(dev), on_frame=cb, progress=False, grad=grad)
        return Hh.level("mpm_particle").get("pos")

    def loss_fn(Xs):
        rho_s = _mass_grid(Xs, one, world, cfg["n_grid"], dev, torch)
        return ((torch.log1p(rho_s) - torch.log1p(rho_t)) ** 2).mean()

    return sim, theta, rollout, loss_fn, n_total, tgt_np, world


def measure(cfg, dev, torch):
    """One experiment: build fresh, warm up, time `iters` optimiser steps, and separately time a
    forward-only pass (grad=False, same operators) so the forward/backward split is measured
    rather than guessed."""
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(cfg["tf32"])
    torch.set_float32_matmul_precision(cfg["matmul_precision"])

    sim, theta, rollout, loss_fn, n_total, _tgt_np, _world = build_problem(cfg, dev, torch)
    opt = torch.optim.Adam([theta], lr=cfg["lr"])

    def one_iter():
        opt.zero_grad(set_to_none=True)
        Xs = rollout(grad=True)
        loss = loss_fn(Xs)
        loss.backward()
        opt.step()
        return float(loss.detach())

    losses = {}
    t_compile0 = time.time()
    losses[0] = one_iter()               # tick 0: pays compilation if compile is on
    torch.cuda.synchronize(dev)
    compile_s = time.time() - t_compile0

    for it in range(1, cfg["warmup"]):
        losses[it] = one_iter()
    torch.cuda.synchronize(dev)

    times = []
    n_meas = cfg["iters"]
    t0 = it0 = cfg["warmup"]
    for it in range(it0, it0 + n_meas):
        torch.cuda.synchronize(dev)
        t0 = time.time()
        losses[it] = one_iter()
        torch.cuda.synchronize(dev)
        times.append(time.time() - t0)

    peak_mem = torch.cuda.max_memory_allocated(dev) / 2**30

    # FREE THE FIRST PROBLEM BEFORE BUILDING THE SECOND. `sim`/`theta`/`rollout`/`opt` are still
    # live references here, and a fresh `build_problem` call allocates a WHOLE SECOND set of grid
    # and particle buffers on top of them -- measured, this is what turned a 200,000-particle
    # forward-only pass into an OOM 186 MiB short of a fresh 80 GiB A100, when the grad-mode peak
    # for the same config had already fit in under 78 GiB by itself. The forward-only number is
    # supposed to isolate "no backward", not "no backward, plus whatever the grad run left behind".
    del sim, theta, rollout, loss_fn, opt
    torch.cuda.empty_cache()

    # forward-only split: same operators, grad=False, no backward/step. A fresh set of instances
    # (fresh Spec) so the compiled cache from the grad=True run above cannot be blamed for this
    # number -- `compile` under grad=False recompiles once more, so this too pays its own warm-up.
    sim2, theta2, rollout2, loss_fn2, _, _tgt_np2, _world2 = build_problem(cfg, dev, torch)
    with torch.no_grad():
        rollout2(grad=False)             # warm-up / compile
        torch.cuda.synchronize(dev)
        fwd_times = []
        for _ in range(min(5, n_meas)):
            torch.cuda.synchronize(dev)
            tf = time.time()
            rollout2(grad=False)
            torch.cuda.synchronize(dev)
            fwd_times.append(time.time() - tf)

    return dict(
        s_per_iter_mean=statistics.mean(times),
        s_per_iter_median=statistics.median(times),
        s_per_iter_all=times,
        compile_s=compile_s,
        forward_only_s=statistics.median(fwd_times),
        peak_mem_gib=peak_mem,
        loss_first=losses[0],
        loss_last=losses[it0 + n_meas - 1],
        n_total_particles=n_total,
        gpu=torch.cuda.get_device_name(dev),
    )


def render_check(cfg, dev, torch, out_dir, name, iters=60, full_movie=True):
    """The correctness picture behind a speed number: run the REAL optimisation (not the handful
    of iterations `measure()` times), then hand the resulting trajectory to
    `morph_gallery.render()` -- the same glass-surface renderer every other morph in this repo
    uses -- rather than a second, private renderer that could quietly draw a different-looking
    answer. `full_movie=False` keeps only the final frame, for a variant this is being compared
    against a baseline picture rather than watched end to end.
    """
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from morph_gallery import render as gallery_render

    sim, theta, rollout, loss_fn, n_total, tgt_np, world = build_problem(cfg, dev, torch)
    opt = torch.optim.Adam([theta], lr=cfg["lr"])
    loss_hist = []
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        Xs = rollout(grad=True)
        loss = loss_fn(Xs)
        loss.backward()
        opt.step()
        loss_hist.append(float(loss.detach().cpu()))
    with torch.no_grad():
        keep = []
        rollout(grad=False, keep=keep)
    frames_np = np.stack([k.cpu().numpy() for k in keep])
    if not full_movie:
        frames_np = frames_np[-1:]                    # final shape only -- a still, not a movie

    d = os.path.join(out_dir, f"render_{name}")
    os.makedirs(d, exist_ok=True)
    gallery_render(d, frames_np, tgt_np, box=max(world), name=name)
    png_path = os.path.join(d, f"still_{len(frames_np) - 1:03d}.png")
    mp4_path = os.path.join(d, "movie.mp4")
    print(f"    render: loss {loss_hist[0]:.6f} -> {loss_hist[-1]:.6f}  -> {d}/", flush=True)
    return png_path, mp4_path, loss_hist


# =================================================================================================
# THE SUITE. Every entry is a diff against DEFAULTS; `measure()` builds the config, runs it, and
# the name says what changed. Grouped by the question it answers.
# =================================================================================================
SUITE = {
    # -- reproduce the four numbers in the brief, on THIS gpu, before touching anything --
    "base_50k_8sub":   dict(n_pts=50000, n_grid=40, substeps=8),
    "base_50k_6sub":   dict(n_pts=50000, n_grid=40, substeps=6),
    "base_12500_6sub": dict(n_pts=12500, n_grid=32, substeps=6),

    # -- torch.compile: mode, and the on/off delta at the small scale the brief measured it at --
    "compile_default_12500_6sub": dict(n_pts=12500, n_grid=32, substeps=6, compile="default"),
    "compile_maxauto_12500_6sub": dict(n_pts=12500, n_grid=32, substeps=6, compile="max-autotune"),
    "compile_reduceoverhead_12500_6sub": dict(n_pts=12500, n_grid=32, substeps=6,
                                              compile="reduce-overhead"),
    "compile_default_50k_8sub":   dict(n_pts=50000, n_grid=40, substeps=8, compile="default"),
    "compile_maxauto_50k_8sub":   dict(n_pts=50000, n_grid=40, substeps=8, compile="max-autotune"),

    # -- TF32 / matmul precision, on top of the compiled 12,500 baseline --
    "tf32_on_12500_6sub":       dict(n_pts=12500, n_grid=32, substeps=6, compile="default",
                                     tf32=True),
    "matmul_high_12500_6sub":   dict(n_pts=12500, n_grid=32, substeps=6, compile="default",
                                     matmul_precision="high"),
    "matmul_medium_12500_6sub": dict(n_pts=12500, n_grid=32, substeps=6, compile="default",
                                     matmul_precision="medium"),

    # -- polar_iters: is 6 more than the deformation needs? --
    "polar3_12500_6sub": dict(n_pts=12500, n_grid=32, substeps=6, compile="default", polar_iters=3),
    "polar4_12500_6sub": dict(n_pts=12500, n_grid=32, substeps=6, compile="default", polar_iters=4),
    "polar5_12500_6sub": dict(n_pts=12500, n_grid=32, substeps=6, compile="default", polar_iters=5),

    # -- grid resolution at fixed particle count --
    "grid24_12500_6sub": dict(n_pts=12500, n_grid=24, substeps=6, compile="default"),
    "grid48_12500_6sub": dict(n_pts=12500, n_grid=48, substeps=6, compile="default"),

    # -- batching: N independent 12,500-point trajectories in ONE rollout vs paying N x the
    # single-trajectory cost. Verifies/extends the 1.35s -> 3.18s (1.7x, not 4x) note. --
    "batch1_12500_6sub_compiled": dict(n_pts=12500, n_grid=32, substeps=6, compile="default",
                                       batch=1),
    # a same-card denominator for the batching-efficiency ratio -- batch2/4/8/16 above ran on
    # whichever card had room (L4 for 2, A100 for 4/8/16), so N*t(batch=1)/t(batch=N) needs a
    # batch=1 number measured on EACH of those cards, not just L4's.
    "batch1_12500_6sub_compiled_a100": dict(n_pts=12500, n_grid=32, substeps=6, compile="default",
                                            batch=1),
    "batch2_12500_6sub_compiled": dict(n_pts=12500, n_grid=32, substeps=6, compile="default",
                                       batch=2),
    "batch4_12500_6sub_compiled": dict(n_pts=12500, n_grid=32, substeps=6, compile="default",
                                       batch=4),
    "batch8_12500_6sub_compiled": dict(n_pts=12500, n_grid=32, substeps=6, compile="default",
                                       batch=8),
    "batch16_12500_6sub_compiled": dict(n_pts=12500, n_grid=32, substeps=6, compile="default",
                                        batch=16),

    # -- the headline: best combination found above, applied at full 50,000/8-substep scale --
    "best_50k_8sub": dict(n_pts=50000, n_grid=40, substeps=8, compile="default", polar_iters=4),
}
SUITE["headline"] = None  # resolved in main(): base_50k_8sub, base_50k_6sub, base_12500_6sub,
                          # compile_default_12500_6sub, best_50k_8sub -- the fast subset for a
                          # first pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _locked(path, fn):
    """Run `fn()` while holding an exclusive advisory lock on `<path>.lock`.

    TWO OF MY OWN JOBS RACED ON THIS FILE ONCE ALREADY: `l4restore` and `a100d`, submitted
    together, each did read-doc -> add-one-record -> write-whole-doc with no lock, and the second
    write landed on a doc it had loaded BEFORE the first write, silently erasing the first job's
    record. Measured, not hypothetical -- `batch1_12500_6sub_compiled_a100` printed its own
    success line and a `-> results.json` line and is still missing from the file. The lock does
    not make the two writes agree on ordering, it just stops one from being invisible to the
    other: each holds the lock only across its own load+modify+write, so the second one always
    starts from a doc that already contains the first one's record.
    """
    import fcntl
    with open(path + ".lock", "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def update_results(path, section, name, rec):
    """Load, replace-or-append `rec` under `doc[section]` keyed by `name`, write -- all under one
    lock, so two concurrent morph_speed.py processes sharing this results.json cannot lose an
    entry to each other (see `_locked`)."""
    def _do():
        doc = load_results(path)
        doc.setdefault(section, [])
        doc[section] = [e for e in doc[section] if e["name"] != name] + [rec]
        write_results(path, doc)
        return doc
    return _locked(path, _do)


def load_results(path):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return dict(agent="faster", updated=_now(), baseline=dict(
        s_per_iter=3.20, points=50000, frames=20, substeps=8,
        note="from the brief: RTX A6000, cow target, 20-frame rollouts; this agent's own numbers "
             "are measured on gpu_l4 (see gpu field per experiment) and are not directly "
             "comparable in absolute seconds -- only the RATIOS this agent measures on its own "
             "hardware are load-bearing."),
        experiments=[], recommended_patches=[], open_questions=[])


def write_results(path, doc):
    doc["updated"] = _now()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="headline",
                    help="comma list of SUITE keys, or 'all' for every entry")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=None, help="override DEFAULTS['iters']")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(
        "/groups/saalfeld/home/allierc/GraphData/graphs_data/si_material/faster"))
    ap.add_argument("--render", default="",
                    help="comma list of SUITE keys to run as a real (--render-iters) optimisation "
                         "and save via morph_gallery.render() to <out>/render_<name>/ -- full "
                         "movie + stills, e.g. a baseline/best pair")
    ap.add_argument("--render-still", default="",
                    help="comma list of SUITE keys rendered as a FINAL-FRAME STILL only (still "
                         "goes through morph_gallery.render() too, just with a 1-frame "
                         "trajectory) -- for a variant being compared against a baseline picture "
                         "rather than watched end to end")
    ap.add_argument("--render-iters", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "results.json")

    import torch
    import plexus.operators  # noqa: F401
    dev = torch.device(args.device)

    names = list(SUITE.keys()) if args.suite == "all" else [s.strip() for s in args.suite.split(",")]
    if "headline" in names:
        names = (["base_50k_8sub", "base_50k_6sub", "base_12500_6sub",
                  "compile_default_12500_6sub", "best_50k_8sub"]
                 + [n for n in names if n != "headline"])

    def _stamp_gpu():
        doc = load_results(json_path)
        doc["gpu_used_this_run"] = torch.cuda.get_device_name(dev)
        write_results(json_path, doc)
    _locked(json_path, _stamp_gpu)

    for name in names:
        if name == "headline" or SUITE.get(name) is None:
            continue
        cfg = {**DEFAULTS, **SUITE[name]}
        if args.iters is not None:
            cfg["iters"] = args.iters
        if args.warmup is not None:
            cfg["warmup"] = args.warmup
        print(f"\n=== {name}  {cfg} ===", flush=True)
        t_exp = time.time()
        try:
            torch.cuda.reset_peak_memory_stats(dev)
            r = measure(cfg, dev, torch)
            rec = dict(name=name, config=cfg, result=r,
                       wall_s=time.time() - t_exp, ok=True,
                       command=(f"PYTHONPATH=src python tools/morph_speed.py --suite {name} "
                                f"--device {args.device}"))
            print(f"    s/iter median {r['s_per_iter_median']:.3f}  mean {r['s_per_iter_mean']:.3f}  "
                  f"forward-only {r['forward_only_s']:.3f}  peak {r['peak_mem_gib']:.2f} GiB  "
                  f"loss {r['loss_first']:.6f} -> {r['loss_last']:.6f}", flush=True)
        except Exception as e:
            import traceback
            rec = dict(name=name, config=cfg, ok=False, error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc(), wall_s=time.time() - t_exp,
                       command=(f"PYTHONPATH=src python tools/morph_speed.py --suite {name} "
                                f"--device {args.device}"))
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
        # locked read-modify-write: a re-run supersedes its own name, and a concurrent process
        # updating a DIFFERENT name cannot lose this one (or have this one lose its own).
        update_results(json_path, "experiments", name, rec)
        torch.cuda.empty_cache()

    render_plan = ([(n.strip(), True) for n in args.render.split(",") if n.strip()]
                  + [(n.strip(), False) for n in args.render_still.split(",") if n.strip()])
    for rname, full_movie in render_plan:
        cfg = {**DEFAULTS, **SUITE[rname]}
        kind = "movie" if full_movie else "still"
        print(f"\n=== render check ({kind}): {rname}  {cfg} ===", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(cfg["tf32"])
        torch.set_float32_matmul_precision(cfg["matmul_precision"])
        png_path, mp4_path, loss_hist = render_check(cfg, dev, torch, args.out, rname,
                                                      iters=args.render_iters, full_movie=full_movie)
        update_results(json_path, "renders", rname, dict(
            name=rname, kind=kind, config=cfg, png=png_path, mp4=mp4_path,
            loss_first=loss_hist[0], loss_last=loss_hist[-1], iters=args.render_iters))
        torch.cuda.empty_cache()

    print(f"\n-> {json_path}", flush=True)


if __name__ == "__main__":
    main()
