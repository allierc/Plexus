"""Lower the morph's loss AT FIXED WALL-CLOCK, not at fixed iterations -- and test whether a
smoothness penalty on the control field is what stops the cow from tearing.

This is `tools/morph_gallery.py`'s loop (read that file first) with the LOSS side opened up. The
gallery is owned by another agent; this script re-derives the same rollout so the two stay
comparable -- same spec, same ball, same trilinear control cube, same `mass_grid` -- and adds the
things worth trying on the objective itself:

    smoothness   a Laplacian (`--smooth-kind l2`) or total-variation (`tv`) penalty on the K^3 x 6
                 control cube: sum over each pair of grid-adjacent nodes of |A_i - A_j|^2 (or
                 |A_i - A_j|). THE HYPOTHESIS UNDER TEST: two neighbouring nodes with opposing
                 rates -- one stretching, one compressing -- pull the material between them apart
                 every frame; nothing in the loss currently penalises that, because the log-mass
                 loss only sees where mass ended up, not how violently neighbouring instructions
                 disagreed to get it there. Cost: a sum over 6*(K-1)*K^2 pairs, K <= 16, so under
                 20,000 scalar terms against a rollout that costs 2.7 s -- unmeasurable overhead.
    f-smooth     the paper's OWN answer to a related but different symptom (Xu et al., Sec. VI-C,
                 Table II gamma): temporal smoothing of F, F <- (1-g) F_new + g F_old, applied
                 every frame during the rollout (their post-process, made a first-class part of
                 the rollout instead). Smooths a particle's OWN history; does not touch neighbours.
    multiscale   the log-mass loss summed over several grid resolutions rather than one. Nearly
                 free: the rollout's forward pass dominates at 2.7 s/iteration, and three extra
                 grid-scatters over the *already computed* Xs cost single-digit milliseconds.
    chamfer      a symmetric nearest-neighbour term between (subsampled) Xs and the target cloud,
                 alongside the mass loss -- the position information the log-mass loss discards
                 because it only reads a voxel occupancy.
    passes       the paper's Table I trade-off (1x12, 3x4, 6x2, 12x1 -- same wall-clock, best loss
                 at 3-6 passes): here, `--passes P` splits the SAME iteration budget into P
                 cosine-annealing warm restarts instead of one, since our control has no per-
                 timestep layers to stage the way theirs does.
    anneal-ctrl  grow the control cube's resolution DURING a stage (e.g. `4,8,12,16`), upsampling
                 the K^3 parameter cube by trilinear interpolation at each step -- coarse rates
                 first, detail added without discarding what the coarse ones already found.
    ngp lr split a separate learning rate for the hash table and its head, when `--control ngp`.

WHAT COUNTS AS AN ANSWER: final loss AT THE SAME grid resolution the baseline used (40^3 for the
last stage) and AT THE SAME wall-clock, never at the same iteration count -- a regulariser that
costs 20% more per iteration and converges just as fast in iterations is not an improvement.

    PYTHONPATH=src python tools/morph_loss.py --target cow --ctrl 12 --smooth-kind l2 \
        --smooth-weight 3.0 --stages 5000:24:80,12500:32:80,50000:40:80 --run-name smooth_l2_3p0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
MODELS = os.path.join(ROOT, "papers", "morph_models")
RESULTS_DIR = "/groups/saalfeld/home/allierc/GraphData/graphs_data/si_material/better"
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def update_results_json(entry):
    """Append/replace one experiment by name and rewrite the shared file NOW -- so a run that is
    interrupted partway through a long sweep still leaves everything finished so far on disk."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    doc = {"agent": "better", "updated": _now(), "baseline": {
        "final_loss": 0.00247, "control": "12^3", "minutes": 4.4, "grid_eval": 40,
        "note": "loss is only comparable at the same evaluation grid; three coarse-to-fine "
                "stages (5000/24^3, 12500/32^3, 50000/40^3), 60 iters each, 20-frame rollouts, "
                "2.68 s/iteration"},
        "experiments": [], "recommended_patches": [], "open_questions": []}
    if os.path.exists(RESULTS_JSON):
        try:
            doc = json.load(open(RESULTS_JSON))
        except Exception:
            pass
    doc["updated"] = _now()
    doc.setdefault("experiments", [])
    doc["experiments"] = [e for e in doc["experiments"] if e.get("name") != entry.get("name")]
    doc["experiments"].append(entry)
    tmp = RESULTS_JSON + ".tmp"
    json.dump(doc, open(tmp, "w"), indent=2)
    os.replace(tmp, RESULTS_JSON)


def trilinear_weights(X0n, c, R, K, dev, torch):
    """Every particle's index/weight pairs into a K^3 control cube, from its MATERIAL coordinate.
    Identical construction to `morph_gallery.py` so a run here is the same control as there."""
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


def upsample_theta(theta, K_old, K_new, torch):
    """Grow the control cube's resolution mid-run: the K_old^3 x 6 cube of rates, trilinearly
    resampled to K_new^3 x 6. Coarse rates are a good INITIALISATION for fine ones -- the field is
    smooth by construction of a coarse grid, so growing it does not throw away what converged."""
    import torch.nn.functional as F
    if K_old == K_new:
        return theta.detach().clone().requires_grad_(True)
    vol = theta.detach().reshape(K_old, K_old, K_old, 6).permute(3, 0, 1, 2)[None]
    up = F.interpolate(vol, size=(K_new, K_new, K_new), mode="trilinear", align_corners=True)
    out = up[0].permute(1, 2, 3, 0).reshape(K_new ** 3, 6).contiguous()
    return out.requires_grad_(True)


def spatial_roughness(theta, K, kind):
    """Sum over grid-adjacent node pairs of the control cube, `l2` (Laplacian/Dirichlet energy) or
    `tv` (total variation) on the 6 rate components. THE TEST OF THE TEARING HYPOTHESIS: a large
    value means neighbouring nodes disagree, which is exactly the configuration -- one node
    stretching material toward it, its neighbour compressing away -- that pulls the body apart."""
    a = theta.reshape(K, K, K, 6)
    terms = []
    for axis in (0, 1, 2):
        d = a.narrow(axis, 1, K - 1) - a.narrow(axis, 0, K - 1)
        terms.append(d)
    if kind == "l2":
        return sum((d ** 2).sum() for d in terms) / theta.numel()
    if kind == "tv":
        return sum(d.abs().sum() for d in terms) / theta.numel()
    raise ValueError(kind)


def chamfer(Xs, tgt, n_sub, rng_t):
    """Symmetric nearest-neighbour distance, subsampled on both sides so the O(n^2) cdist stays
    cheap: 3,000 x 3,000 is 9e6 entries, sub-millisecond on a GPU, against a 2.7 s rollout."""
    import torch
    ia = torch.randperm(Xs.shape[0], device=Xs.device)[:n_sub]
    ib = rng_t[:n_sub]
    a, b = Xs[ia], tgt[ib]
    d2 = torch.cdist(a, b) ** 2
    return d2.min(1).values.mean() + d2.min(0).values.mean()


_BATCH_LAYOUT = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1), 8: (2, 2, 2), 16: (4, 2, 2)}


def run_batch_screen(args):
    """Screen `--batch-values` smooth-weights in ONE rollout instead of one each -- the FASTER
    agent measured batching a rollout at ~2x throughput, flat across 2/4/8 replicas (its own
    results.json), because the fixed per-substep overhead is paid once for N trajectories instead
    of N times. Each replica is its OWN ball, in its own cell of a multi-cell world (the same
    layout `tools/morph_speed.py` uses), with its OWN K^3 control cube and its OWN smooth-weight;
    the replicas never interact because MPM's shape functions have compact support and the cells
    are a full `BOX` apart. One Adam step optimises every replica's theta at once; the mass loss
    is computed PER REPLICA on its own locally-shifted coordinates (identical numbers to a
    standalone run at the same n_pts/n_grid), so each replica's final loss is directly comparable
    to a non-batched run -- only the wall-clock is shared.

    SCOPE: one stage (the first entry of --stages; this is a SCREENING tool for the coarse/cheap
    resolutions, not a substitute for a full coarse-to-fine run), grid control only, and the one
    axis this sweep needed most -- --smooth-kind's weight. No render: the multi-cell world does
    not fit `morph_gallery.render`'s single-BOX camera, and the point of this mode is many loss
    numbers fast, not a picture. Confirm the winner afterward with a normal (non-batched) run.
    """
    import torch
    import plexus.operators  # noqa: F401
    from plexus.schema import load
    from plexus import engine
    from shape_control import ball, mass_grid
    from morph_gallery import sample_inside

    if args.smooth_kind == "none":
        raise SystemExit("--batch-values needs --smooth-kind l2|tv (that is the axis it sweeps)")
    values = [float(v) for v in args.batch_values.split(",") if v != ""]
    B = len(values)
    if B not in _BATCH_LAYOUT:
        raise SystemExit(f"--batch-values: {B} values, but no layout for batch={B} "
                          f"(supported: {sorted(_BATCH_LAYOUT)})")
    nx, ny, nz = _BATCH_LAYOUT[B]
    n_pts, n_grid, iters = (int(v) for v in args.stages.split(",")[0].split(":"))
    BOX, R = 0.35, 0.055
    dev = torch.device(args.device)
    path = os.path.join(MODELS, f"{args.target}.obj")
    world = (BOX * nx, BOX * ny, BOX * nz)

    ops = [dict(op="mpm_strain", at="mpm_particle", implementation="differentiable"),
           dict(op="mpm_scatter", at="mpm_particle", to="mpm_grid", drag=0.5, a_max=200.0,
                implementation="differentiable"),
           dict(op="mpm_grid_update", at="mpm_grid", wall_damp=0.9, implementation="differentiable"),
           dict(op="mpm_gather", at="mpm_particle", **{"from": "mpm_grid"}, wall_damp=0.9,
                vmax=1.0e9, implementation="differentiable")]
    wx, wy, wz = world
    raw = dict(
        general=dict(name="morph_loss_batch", seed=0, n_frames=args.frames, dt=0.002,
                     record_cap=3, boundary="wall", dim=3, world=[wx, wy, wz],
                     units=dict(length_um=100.0)),
        sets={"mpm_particle": dict(n=n_pts * B, types={"cyto": dict(
            fraction=1.0, youngs=90.0, density=1.0,
            block=[0.4 * wx, 0.4 * wy, 0.4 * wz, 0.6 * wx, 0.6 * wy, 0.6 * wz])})},
        # n_grid UNSCALED, matching morph_speed.py's own batched spec: scaling it by max(nx,ny,nz)
        # (to keep node density per cell constant) looked more physically consistent but the grid
        # tensors scale with n_grid^3 and it OOM'd a batch of 4 at 50,000 total particles that the
        # FASTER agent's own equivalent config (21.06 GiB measured) just barely fit -- the coarser,
        # unscaled grid is the one known to fit in 22 GB.
        fields=dict(mpm_grid=dict(frame="mpm_grid", n_grid=n_grid)),
        operators=ops,
        schedule=[dict(substep_dt=3.4e-4, compile=True,
                       steps=["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"])],
        plotting={},
    )
    f = os.path.join(tempfile.mkdtemp(prefix="morphbatch_"), "spec.yaml")
    yaml.safe_dump(raw, open(f, "w"), sort_keys=False)
    sim = load(f)

    cells = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]
    centers = [np.array([(i + 0.5) * BOX, (j + 0.5) * BOX, (k + 0.5) * BOX]) for i, j, k in cells]
    X0_parts, tgt_parts, thetas, idx_all, wts_all = [], [], [], [], []
    K = args.ctrl
    one_per = torch.ones(n_pts, device=dev)
    for b, c in enumerate(centers):
        rng_b = np.random.default_rng(args.seed * 1000 + b)
        X0n_b = ball(n_pts, c, R, rng_b)
        X0_parts.append(X0n_b)
        tgt_parts.append(sample_inside(path, n_pts, c, 2.6 * R, rng_b))
        idx_b, wts_b = trilinear_weights(X0n_b, c, R, K, dev, torch)
        idx_all.append(idx_b); wts_all.append(wts_b)
        thetas.append(torch.zeros(K ** 3, 6, device=dev, requires_grad=True))
    X0n = np.concatenate(X0_parts)
    tgt_np = np.concatenate(tgt_parts)
    X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
    tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
    with torch.no_grad():
        rho_t = [mass_grid(tgt[b * n_pts:(b + 1) * n_pts] - torch.as_tensor(centers[b] - BOX / 2,
                            dtype=torch.float32, device=dev), one_per, BOX, n_grid, dev, torch)
                 for b in range(B)]

    opt = torch.optim.Adam(thetas, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1),
                                                       eta_min=args.lr / 20.0)
    eye = torch.eye(3, device=dev)
    dt = float(sim.dt)
    t_run = time.time()
    print(f"\n  {args.run_name}: BATCHED screen, {B} replicas ({nx}x{ny}x{nz}), "
          f"{args.smooth_kind} weights={values}, {n_pts:,} pts/replica, grid {n_grid}, "
          f"{iters} iters", flush=True)

    def rollout(keep=None):
        a6 = torch.cat([sum(w[:, None] * th[i] for i, w in zip(idx_b, wts_b))
                        for th, idx_b, wts_b in zip(thetas, idx_all, wts_all)], dim=0)
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
                keep.append(q.get("pos").detach().cpu().numpy().copy())
        Hh, _ = engine.run(sim, device=args.device, on_frame=cb, progress=False,
                           grad=(keep is None))
        return Hh.level("mpm_particle").get("pos")

    last_losses = [float("nan")] * B
    for it in range(iters):
        opt.zero_grad()
        Xs = rollout()
        total = 0.0
        per_loss, per_rough = [], []
        for b in range(B):
            off = torch.as_tensor(centers[b] - BOX / 2, dtype=torch.float32, device=dev)
            Xb = Xs[b * n_pts:(b + 1) * n_pts] - off
            lm = ((torch.log1p(mass_grid(Xb, one_per, BOX, n_grid, dev, torch))
                  - torch.log1p(rho_t[b])) ** 2).mean()
            rough = spatial_roughness(thetas[b], K, args.smooth_kind)
            total = total + lm + values[b] * rough
            per_loss.append(lm); per_rough.append(rough)
        total.backward()
        opt.step(); sched.step()
        last_losses = [float(x.detach()) for x in per_loss]
        last_roughs = [float(x.detach()) for x in per_rough]
        if it % 20 == 0 or it == iters - 1:
            row = "  ".join(f"w={values[b]:g}: mass {last_losses[b]:.5f} rough {last_roughs[b]:.3f}"
                            for b in range(B))
            print(f"    iter {it:3d}  {row}", flush=True)

    total_s = time.time() - t_run
    print(f"\n  {args.run_name}: {iters} iterations in {total_s/60:.1f} min "
          f"({total_s/max(iters,1):.2f} s/iteration, {B} replicas)", flush=True)
    cmd = "PYTHONPATH=src python tools/morph_loss.py " + " ".join(sys.argv[1:])
    for b in range(B):
        update_results_json(dict(
            name=f"{args.run_name}_w{values[b]:g}", target=args.target, control="grid",
            ctrl=args.ctrl, smooth_kind=args.smooth_kind, smooth_weight=values[b],
            final_loss=last_losses[b], final_rough_l2=(last_roughs[b] if args.smooth_kind == "l2"
                                                        else float("nan")),
            final_rough_tv=(last_roughs[b] if args.smooth_kind == "tv" else float("nan")),
            grid_eval=n_grid, minutes=total_s / 60.0, stages=args.stages,
            batched=True, batch_size=B, batch_group=args.run_name, command=cmd, updated=_now()))
    print(f"  -> wrote {B} results.json entries under batch_group={args.run_name!r}", flush=True)


def run_split(args):
    """Drive `--split-stages`: one child process per stage, each warm-started from the previous
    stage's own morph.npz, because that is the only thing that reliably frees an L4's 22 GB
    between a 5,000-point stage and a 50,000-point one (see the flag's own help string). The
    driver does the ONE results.json write, aggregating wall-clock across every child so the
    number a variant is compared on on is the true end-to-end cost of running it this way -- the
    per-stage python startup is a few seconds, paid identically by baseline and every variant."""
    stages = [st for st in args.stages.split(",") if st]
    tmp_root = tempfile.mkdtemp(prefix="morphsplit_")
    prev_npz = ""
    t_driver = time.time()
    for si, stage_str in enumerate(stages):
        is_last = si == len(stages) - 1
        out_dir, run_name = (args.out, args.run_name) if is_last else (tmp_root, f"stage{si}")
        argv = [sys.executable, os.path.abspath(__file__),
                "--target", args.target, "--frames", str(args.frames), "--ctrl", str(args.ctrl),
                "--anneal-ctrl", args.anneal_ctrl, "--lr", str(args.lr), "--device", args.device,
                "--out", out_dir, "--run-name", run_name, "--stages", stage_str,
                "--control", args.control, "--smooth-kind", args.smooth_kind,
                "--smooth-weight", str(args.smooth_weight),
                "--f-smooth-gamma", str(args.f_smooth_gamma), "--multiscale", args.multiscale,
                "--chamfer-weight", str(args.chamfer_weight), "--chamfer-n", str(args.chamfer_n),
                "--passes", str(args.passes), "--seed", str(args.seed),
                "--stage-index", str(si), "--no-results"]
        argv += (["--render-frames", str(args.render_frames)] if (is_last and not args.no_render)
                 else ["--no-render", "--render-frames", "0"])
        if args.ngp_lr_grid is not None:
            argv += ["--ngp-lr-grid", str(args.ngp_lr_grid)]
        if args.ngp_lr_head is not None:
            argv += ["--ngp-lr-head", str(args.ngp_lr_head)]
        if prev_npz:
            argv += ["--warm-start", prev_npz]
        print(f"[split] stage {si}/{len(stages)-1}: {stage_str} -> {run_name}", flush=True)
        env = dict(os.environ); env["_MORPHLOSS_CHILD"] = "1"
        t0 = time.time()
        r = subprocess.run(argv, env=env)
        if r.returncode != 0:
            print(f"[split] stage {si} FAILED (exit {r.returncode}) -- aborting", flush=True)
            sys.exit(r.returncode)
        prev_npz = os.path.join(out_dir, run_name, "morph.npz")
        print(f"[split] stage {si} done in {(time.time()-t0)/60:.2f} min -> {prev_npz}", flush=True)

    total_s = time.time() - t_driver
    z = np.load(prev_npz)
    hist = z["history"]
    final_loss, final_rough_l2, final_rough_tv = float(hist[-1, 3]), float(hist[-1, 4]), \
        float(hist[-1, 5])
    n_grid_final = int(stages[-1].split(":")[1])
    print(f"\n  {args.run_name} (split): final loss {final_loss:.6f} (grid {n_grid_final}), "
          f"rough(l2/tv) {final_rough_l2:.4f}/{final_rough_tv:.4f}, {total_s/60:.1f} min total "
          f"-> {os.path.relpath(prev_npz, ROOT) if prev_npz.startswith(ROOT) else prev_npz}",
          flush=True)
    if not args.no_results:
        cmd = "PYTHONPATH=src python tools/morph_loss.py " + " ".join(sys.argv[1:])
        update_results_json(dict(
            name=args.run_name, target=args.target, control=args.control, ctrl=args.ctrl,
            anneal_ctrl=args.anneal_ctrl, smooth_kind=args.smooth_kind,
            smooth_weight=args.smooth_weight, f_smooth_gamma=args.f_smooth_gamma,
            multiscale=args.multiscale, chamfer_weight=args.chamfer_weight, passes=args.passes,
            final_rough_l2=final_rough_l2, final_rough_tv=final_rough_tv,
            final_loss=final_loss, grid_eval=n_grid_final, minutes=total_s / 60.0,
            stages=args.stages, split_stages=True, command=cmd, updated=_now()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="cow")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--ctrl", type=int, default=12, help="control nodes per axis (ignored if "
                    "--anneal-ctrl is given)")
    ap.add_argument("--anneal-ctrl", default="", help="e.g. 4,8,12,16 -- grow the control cube "
                    "through these resolutions, evenly across the LAST stage's iterations")
    ap.add_argument("--lr", type=float, default=3.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--stages", default="5000:24:80,12500:32:80,50000:40:80")
    ap.add_argument("--render-frames", type=int, default=100)
    ap.add_argument("--control", default="grid", choices=["grid", "ngp"])
    ap.add_argument("--ngp-lr-grid", type=float, default=None, help="overrides --lr for the hash "
                    "table's own parameters, when --control ngp")
    ap.add_argument("--ngp-lr-head", type=float, default=None, help="overrides --lr for the MLP "
                    "head, when --control ngp")
    ap.add_argument("--smooth-kind", default="none", choices=["none", "l2", "tv"])
    ap.add_argument("--smooth-weight", type=float, default=0.0)
    ap.add_argument("--f-smooth-gamma", type=float, default=0.0, help="paper's temporal EMA on F: "
                    "F <- (1-g) F_new + g F_old, every frame")
    ap.add_argument("--multiscale", default="", help="e.g. 24,32,48 -- extra grid resolutions "
                    "the log-mass loss is summed over, alongside the stage's own --n-grid")
    ap.add_argument("--chamfer-weight", type=float, default=0.0)
    ap.add_argument("--chamfer-n", type=int, default=3000)
    ap.add_argument("--volume-weight", type=float, default=0.0, help="penalise "
                    "(mean(det(F)) - 1)^2 directly -- more_precise measured the body losing more "
                    "than half its volume (ratio 0.448) under the plain log-mass loss, and their "
                    "trace-free/volume-penalty-on-the-CONTROL fixes made it WORSE (0.233, 0.243). "
                    "This penalises the OUTCOME instead of constraining the control's form.")
    ap.add_argument("--passes", type=int, default=1, help="cosine warm restarts within a stage's "
                    "iteration budget, at fixed total iterations (paper Table I)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--split-stages", action="store_true", help="run each stage in its OWN "
                    "subprocess, warm-starting the control from the previous stage's morph.npz. "
                    "An L4 has 22 GB against the 49 GB an A6000 has: three coarse-to-fine stages "
                    "(5,000/12,500/50,000 points) OOM 16 MiB into the last stage's first matmul "
                    "when run in one process, at ~22.05/22.06 GB already resident, because "
                    "torch.compile keeps a memory pool alive per shape it has seen. A fresh "
                    "process per stage is the only thing that reliably frees it; the cost is one "
                    "python-and-import startup (a few seconds) per stage, paid identically by "
                    "every variant, so equal-wall-clock comparisons stay fair.")
    ap.add_argument("--warm-start", default="", help="[internal] a morph.npz to load the control "
                    "from before optimising this stage")
    ap.add_argument("--no-results", action="store_true", help="[internal] skip the results.json "
                    "write -- used for every non-final stage when --split-stages drives them")
    ap.add_argument("--stage-index", type=int, default=None, help="[internal] this stage's index "
                    "in the ORIGINAL --stages list, for RNG seeding independent of process split")
    ap.add_argument("--batch-values", default="", help="e.g. 0,1e-5,1e-4,1e-3,1e-2 -- screen "
                    "these --smooth-kind weights as independent replicas in ONE rollout (see "
                    "run_batch_screen's docstring). Mutually exclusive with --split-stages; only "
                    "the FIRST --stages entry is used (this is a screening tool, not a "
                    "coarse-to-fine run), and nothing is rendered.")
    args = ap.parse_args()

    if args.batch_values:
        return run_batch_screen(args)
    if args.split_stages and not os.environ.get("_MORPHLOSS_CHILD"):
        return run_split(args)

    import torch
    import plexus.operators  # noqa: F401
    from plexus.schema import load
    from plexus import engine
    from shape_control import spec, ball, mass_grid
    from morph_gallery import sample_inside, render

    BOX, C, R = 0.35, 0.175, 0.055
    dev = torch.device(args.device)
    c = np.array([C, C, C])
    path = os.path.join(MODELS, f"{args.target}.obj")

    stages = [tuple(int(v) for v in st.split(":")) for st in args.stages.split(",")]
    anneal = [int(v) for v in args.anneal_ctrl.split(",")] if args.anneal_ctrl else []
    multiscale = [int(v) for v in args.multiscale.split(",")] if args.multiscale else []

    theta = grid_enc = head = None
    K = anneal[0] if anneal else args.ctrl
    t_run = time.time()
    history = []
    print(f"\n  {args.run_name}: target={args.target} control={args.control} K0={K} "
          f"smooth={args.smooth_kind}/{args.smooth_weight} f_smooth={args.f_smooth_gamma} "
          f"multiscale={multiscale} chamfer={args.chamfer_weight} passes={args.passes} "
          f"anneal={anneal}", flush=True)

    n_pts = n_grid = None
    for si, (n_pts, n_grid, iters) in enumerate(stages):
        # INDEPENDENT PER-STAGE RNG, keyed by the stage's position in the ORIGINAL --stages list
        # (passed explicitly as --stage-index by a --split-stages driver, since a child process
        # only ever sees ONE stage and would otherwise always draw stage 0's sequence). This also
        # makes a stage's sample reproducible on its own, rather than depending on how many random
        # draws every earlier stage's rejection sampling happened to consume.
        stage_idx = args.stage_index if args.stage_index is not None else si
        rng = np.random.default_rng(args.seed * 1000 + stage_idx)
        if si > 0:
            # AN L4 HAS 22 GB, NOT THE 49 GB OF THE A6000 THE BASELINE NUMBERS WERE MEASURED ON.
            # `torch.compile` caches one CUDA-graph memory pool PER INPUT SHAPE, keyed by particle
            # count, and never frees an earlier stage's pool just because that stage ended --
            # measured, three coarse-to-fine stages (5,000 / 12,500 / 50,000 points) on gpu_l4 ran
            # stage 0 and 1 fine and then OOM'd 16 MiB into stage 2's first `bmm`, with 22.05/22.06
            # GB already resident before that allocation. Clearing the compile cache between
            # stages costs one recompilation (tens of seconds) and is the difference between
            # finishing and not.
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            try:
                torch._dynamo.reset()
            except Exception:
                pass
        raw = spec(n_pts, BOX, n_grid, args.frames, 0.002, 3.4e-4)
        f = os.path.join(tempfile.mkdtemp(prefix="morphloss_"), "spec.yaml")
        yaml.safe_dump(raw, open(f, "w"), sort_keys=False)
        sim = load(f)
        X0n = ball(n_pts, c, R, rng)
        X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
        one = torch.ones(n_pts, device=dev)
        tgt_np = sample_inside(path, n_pts, c, 2.6 * R, rng)
        tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
        with torch.no_grad():
            rho_t = {ng: mass_grid(tgt, one, BOX, ng, dev, torch) for ng in [n_grid] + multiscale}
        Xm = torch.as_tensor(np.clip((X0n - (c - R)) / (2 * R), 0, 1),
                             dtype=torch.float32, device=dev)

        last_stage = (si == len(stages) - 1)
        ctrl_schedule = (anneal if (last_stage and anneal and args.control == "grid")
                         else [K] * iters if args.control == "grid" else [])

        def make_ngp():
            from plexus.models.hashgrid import MultiResHashGrid
            ge = MultiResHashGrid(n_input_dims=3, n_levels=8, n_features_per_level=2,
                                  log2_hashmap_size=15, base_resolution=4,
                                  per_level_scale=1.6).to(dev)
            hd = torch.nn.Sequential(torch.nn.Linear(8 * 2, 32), torch.nn.GELU(),
                                     torch.nn.Linear(32, 6)).to(dev)
            with torch.no_grad():
                hd[-1].weight.mul_(0.0); hd[-1].bias.mul_(0.0)
            return ge, hd

        # WARM-START, from the PREVIOUS STAGE's own morph.npz -- how `--split-stages` carries the
        # control across a process boundary, since a fresh process has no python object to hand it
        # in memory. Grid: the K_prev^3 cube, trilinearly resized to this stage's K. NGP: the
        # flat parameter vector, poured back into a freshly built grid+head of the SAME shape.
        if args.warm_start and theta is None and grid_enc is None:
            z = np.load(args.warm_start)
            ctrl_prev = np.asarray(z["control"], dtype=np.float32)
            if args.control == "grid":
                K_prev = int(round(ctrl_prev.shape[0] ** (1.0 / 3.0)))
                theta = upsample_theta(torch.as_tensor(ctrl_prev, device=dev), K_prev, K, torch)
                print(f"    warm-start: {K_prev}^3 control from "
                      f"{os.path.basename(args.warm_start)} -> resized to {K}^3", flush=True)
            else:
                grid_enc, head = make_ngp()
                vec = torch.as_tensor(ctrl_prev, device=dev)
                torch.nn.utils.vector_to_parameters(
                    vec, list(grid_enc.parameters()) + list(head.parameters()))
                print(f"    warm-start: ngp params from {os.path.basename(args.warm_start)}",
                      flush=True)

        if args.control == "ngp":
            if grid_enc is None:
                grid_enc, head = make_ngp()
            lr_g = args.ngp_lr_grid if args.ngp_lr_grid is not None else args.lr
            lr_h = args.ngp_lr_head if args.ngp_lr_head is not None else args.lr
            opt = torch.optim.Adam([{"params": grid_enc.parameters(), "lr": lr_g},
                                    {"params": head.parameters(), "lr": lr_h}])
            idx = wts = None
        else:
            if theta is None:
                theta = torch.zeros(K ** 3, 6, device=dev, requires_grad=True)
            idx, wts = trilinear_weights(X0n, c, R, K, dev, torch)
            opt = torch.optim.Adam([theta], lr=args.lr)

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
            G = eye + Adt + 0.5 * (Adt @ Adt)
            gamma = args.f_smooth_gamma
            F_last = [None]

            def cb(Hh, tick, G=G):
                q = Hh.level("mpm_particle")
                if tick == 0:
                    q0, q1 = q.state_schema["pos"]
                    with torch.no_grad():
                        q.state[:, q0:q1] = X0
                else:
                    F_new = torch.bmm(G, q.F)
                    if gamma > 0.0:
                        F_new = (1.0 - gamma) * F_new + gamma * q.F
                    q.F = F_new
                    F_last[0] = q.F
                if keep is not None:
                    keep.append(q.get("pos").detach().cpu().numpy().copy())
            Hh, _ = engine.run(sim, device=args.device, on_frame=cb, progress=False,
                               grad=(keep is None))
            # VOLUME RATIO, end over start: every particle starts at F = identity (det = 1), so the
            # MEAN of det(F) at the final frame IS the ratio of total material volume to what it
            # started as -- the same physical quantity more_precise's volume_ratio_end_over_start
            # tracks, computed here from the F this rollout already carries, at no extra grid cost.
            vol_ratio = torch.det(F_last[0]).mean() if F_last[0] is not None \
                else torch.tensor(1.0, device=dev)
            return Hh.level("mpm_particle").get("pos"), vol_ratio

        # cosine warm restarts: `passes` cycles over the SAME total iteration budget. Base LRs are
        # captured once per param group (so an ngp run's grid/head split survives every restart)
        # and re-applied at the start of each cycle.
        base_lrs = [g["lr"] for g in opt.param_groups]
        cyc = max(1, iters // max(1, args.passes))
        rng_t_idx = torch.randperm(n_pts, device=dev)[:args.chamfer_n] if args.chamfer_weight > 0 \
            else None
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cyc, eta_min=args.lr / 20.0)

        for it in range(iters):
            if it > 0 and it % cyc == 0:
                for g, base in zip(opt.param_groups, base_lrs):
                    g["lr"] = base
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cyc,
                                                                    eta_min=args.lr / 20.0)
            if ctrl_schedule and it < len(ctrl_schedule) and ctrl_schedule[it] != K:
                K_new = ctrl_schedule[it]
                theta = upsample_theta(theta, K, K_new, torch)
                K = K_new
                idx, wts = trilinear_weights(X0n, c, R, K, dev, torch)
                opt = torch.optim.Adam([theta], lr=args.lr)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(cyc - it % cyc, 1), eta_min=args.lr / 20.0)

            opt.zero_grad()
            Xs, vol_ratio = rollout()
            loss_mass = ((torch.log1p(mass_grid(Xs, one, BOX, n_grid, dev, torch))
                         - torch.log1p(rho_t[n_grid])) ** 2).mean()
            loss = loss_mass
            for ng in multiscale:
                loss = loss + ((torch.log1p(mass_grid(Xs, one, BOX, ng, dev, torch))
                               - torch.log1p(rho_t[ng])) ** 2).mean()
            if args.smooth_weight > 0.0 and args.control == "grid":
                loss = loss + args.smooth_weight * spatial_roughness(theta, K, args.smooth_kind)
            if args.chamfer_weight > 0.0:
                loss = loss + args.chamfer_weight * chamfer(Xs, tgt, args.chamfer_n, rng_t_idx)
            if args.volume_weight > 0.0:
                # PENALISE NET COMPACTION DIRECTLY, in the loss the optimiser actually sees, rather
                # than constraining the CONTROL's form (more_precise's trace-free attempt did that
                # and made volume_ratio_end_over_start WORSE, 0.448 -> 0.233): mean(det(F)) is the
                # ratio of current to original material volume, computed from the F this rollout
                # already carries, so a mismatch is visible to the same gradient that drives shape.
                loss = loss + args.volume_weight * (vol_ratio - 1.0) ** 2
            loss.backward()
            if os.environ.get("_MORPHLOSS_DEBUG_NGP") and args.control == "ngp":
                gh = head[-1].weight.grad
                gg = next(iter(grid_enc.parameters())).grad
                print(f"    [dbg] it={it} |W_last| before={float(head[-1].weight.norm()):.4e} "
                      f"grad_head_last={None if gh is None else float(gh.norm()):.4e} "
                      f"grad_grid0={None if gg is None else float(gg.norm()):.4e} "
                      f"lr_groups={[g['lr'] for g in opt.param_groups]}", flush=True)
            opt.step(); sched.step()
            if os.environ.get("_MORPHLOSS_DEBUG_NGP") and args.control == "ngp":
                print(f"    [dbg] it={it} |W_last| after ={float(head[-1].weight.norm()):.4e}",
                      flush=True)
            loss_v, mass_v = float(loss.detach()), float(loss_mass.detach())
            # THE HYPOTHESIS, MEASURED, WHETHER OR NOT IT IS BEING PENALISED: how much do
            # grid-adjacent control nodes disagree, in both norms, regardless of --smooth-weight.
            # Read alongside the rendered tearing, this is the number that says whether opposing
            # neighbours are even present -- if roughness stays flat while the body still tears,
            # the tear is not coming from this field.
            rough_l2 = rough_tv = float("nan")
            if args.control == "grid" and theta is not None:
                with torch.no_grad():
                    rough_l2 = float(spatial_roughness(theta, K, "l2"))
                    rough_tv = float(spatial_roughness(theta, K, "tv"))
            vol_v = float(vol_ratio.detach())
            if it % 20 == 0 or it == iters - 1:
                e = ((Xs.max(0).values - Xs.min(0).values) * 100).tolist()
                print(f"    stage {si} ({n_pts:,} pts, grid {n_grid}, K={K}) iter {it:3d}  "
                      f"loss {loss_v:.6f}  mass {mass_v:.6f}  rough(l2/tv) "
                      f"{rough_l2:.4f}/{rough_tv:.4f}  vol {vol_v:.4f}  "
                      f"shape {e[0]:5.1f} x {e[1]:5.1f} x {e[2]:5.1f} um", flush=True)
            history.append(dict(stage=si, iter=it, loss=loss_v, loss_mass=mass_v,
                                rough_l2=rough_l2, rough_tv=rough_tv, vol_ratio=vol_v,
                                t=time.time() - t_run))
        st_s = time.time() - t_stage
        print(f"    stage {si}: {iters} iterations in {st_s/60:.1f} min "
              f"({st_s/max(iters,1):.2f} s/iteration)", flush=True)

    with torch.no_grad():
        frames = []
        if args.render_frames and args.render_frames != args.frames:
            rf = args.render_frames
            raw = spec(n_pts, BOX, n_grid, rf, 0.002, 3.4e-4)
            fr = os.path.join(tempfile.mkdtemp(prefix="renderloss_"), "spec.yaml")
            yaml.safe_dump(raw, open(fr, "w"), sort_keys=False)
            sim = load(fr)
            dt = float(sim.dt) * args.frames / rf     # same total deformation, finer sampling
        _, eval_vol_ratio = rollout(keep=frames)

    d = os.path.join(args.out, args.run_name)
    os.makedirs(d, exist_ok=True)
    total_s = time.time() - t_run
    final_loss = history[-1]["loss_mass"]
    final_rough_l2, final_rough_tv = history[-1]["rough_l2"], history[-1]["rough_tv"]
    final_vol_ratio = float(eval_vol_ratio)
    ctrl_arr = (theta.detach().cpu().numpy() if theta is not None
               else np.concatenate([q.detach().cpu().numpy().ravel()
                                    for q in list(grid_enc.parameters()) + list(head.parameters())]))
    np.savez_compressed(os.path.join(d, "morph.npz"), frames=np.stack(frames), target=tgt_np,
                        box=BOX, control=ctrl_arr, seconds=total_s, vol_ratio=final_vol_ratio,
                        history=np.array([(h["stage"], h["iter"], h["loss"], h["loss_mass"],
                                           h["rough_l2"], h["rough_tv"], h["vol_ratio"], h["t"])
                                          for h in history]))
    if not args.no_render:
        render(d, np.stack(frames), tgt_np, BOX, args.target)
    print(f"\n  {args.run_name}: final loss {final_loss:.6f} (grid {n_grid}), rough(l2/tv) "
          f"{final_rough_l2:.4f}/{final_rough_tv:.4f}, volume_ratio {final_vol_ratio:.4f}, "
          f"{total_s/60:.1f} min total -> "
          f"{os.path.relpath(d, ROOT) if d.startswith(ROOT) else d}", flush=True)

    if not args.no_results:
        cmd = "PYTHONPATH=src python tools/morph_loss.py " + " ".join(sys.argv[1:])
        update_results_json(dict(
            name=args.run_name, target=args.target, control=args.control, ctrl=args.ctrl,
            anneal_ctrl=args.anneal_ctrl, smooth_kind=args.smooth_kind,
            smooth_weight=args.smooth_weight, f_smooth_gamma=args.f_smooth_gamma,
            multiscale=args.multiscale, chamfer_weight=args.chamfer_weight, passes=args.passes,
            volume_weight=args.volume_weight,
            final_rough_l2=final_rough_l2, final_rough_tv=final_rough_tv,
            volume_ratio_end_over_start=final_vol_ratio,
            final_loss=final_loss, grid_eval=n_grid, minutes=total_s / 60.0, stages=args.stages,
            command=cmd, updated=_now()))


if __name__ == "__main__":
    main()
