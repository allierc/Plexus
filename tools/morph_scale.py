"""How far the deformation-gradient morph (tools/shape_control.py, tools/morph_gallery.py) scales
in PARTICLE COUNT before something breaks, and what the standard remedies (segment chaining /
gradient checkpointing, bf16) buy back.

WHY MEMORY IS THE WALL, NOT SPEED. `engine.run(grad=True)` keeps one autograd tape for the WHOLE
rollout: every substep's intermediate tensors (P2G weights, grid mass/momentum, G2P gathers) stay
alive until `loss.backward()` walks back through all of them. The tape's size is (particle count) x
(substep count), so it is the ROLLOUT LENGTH x the CLOUD SIZE that has to fit in 24 GB on an L4, not
either alone. Xu et al. 2024 (papers/Xu_2024_mpm_shape_morphing.pdf) say this outright: "the memory
required is directly proportional to the number of timesteps," which is why their own method is a
CHAINED multi-pass scheme -- optimise a segment of the rollout, save its final state, free the tape,
continue. That is `--exp checkpoint` below: `torch.utils.checkpoint` around one `engine.run` call
per segment, so only one segment's tape is ever resident, at the cost of recomputing each segment's
forward pass a second time during backward (the standard checkpointing trade).

EXPERIMENTS (`--exp`):
    sweep        peak memory + s/iteration vs PARTICLE COUNT, frames fixed, no checkpointing --
                 finds the wall.
    frames       peak memory + s/iteration vs ROLLOUT LENGTH, particle count fixed -- the other
                 axis of the same product.
    checkpoint   the same particle counts, chained over segments of `--seg` frames each, so memory
                 is bounded by ONE segment instead of the whole rollout. Reports the recompute
                 overhead against the plain rollout at sizes small enough to run both ways.
    verify       correctness, not scale: checkpointed vs plain give the SAME theta.grad (they must,
                 since checkpointing changes only where activations are recomputed, not what the
                 forward computes) -- a tiny rollout, both ways, compared.
    transfer     whether a coarse stage's control can be READ at a fine stage's particles with no
                 further optimisation -- train theta at n_coarse, evaluate (no_grad, no training) at
                 n_fine with weights recomputed from the fine particles' own material coordinates,
                 and compare that zero-shot loss against theta=0 (no control) and against the
                 trained-coarse loss.
    precision    float32 vs bf16-autocast on the same rollout: peak memory and s/iteration.
    render       one demonstrative morph at the largest configuration that fits, written as mp4+png
                 via `morph_gallery.render` (imported, not modified).

Every run appends to a shared `results.json` (schema fixed by the team, not this file) so the other
five agents can read what was measured without re-running it.

    PYTHONPATH=src python tools/morph_scale.py --exp sweep --device cuda:0
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
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

RESULTS_DIR = "/groups/saalfeld/home/allierc/GraphData/graphs_data/si_material/larger"
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")

BOX, CTR, R = 0.35, 0.175, 0.055           # same box/ball as shape_control.py and morph_gallery.py
DT, SUB = 0.002, 3.4e-4                     # frame dt, MPM substep dt -- ~6 substeps/frame

# n_grid per particle count. NOT a pure cube-root rule: 12500/50000 match the existing coarse-to-fine
# stages in jobs/morph.sh exactly (5000:24, 12500:32, 50000:40) so the 50,000-point number here is
# checked against the FASTER agent's own baseline at the SAME grid, not a different one.
GRID_TABLE = {12500: 32, 50000: 40, 200000: 64, 500000: 88, 1000000: 112, 2000000: 144}


def auto_grid(n_pts: int) -> int:
    if n_pts in GRID_TABLE:
        return GRID_TABLE[n_pts]
    lo = max((k for k in GRID_TABLE if k <= n_pts), default=12500)
    return max(16, round(GRID_TABLE[lo] * (n_pts / lo) ** (1.0 / 3.0)))


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


LOCK_DIR = RESULTS_JSON + ".lock"


@contextlib.contextmanager
def _results_lock(timeout=60):
    """An `mkdir`-based lock, not `fcntl.flock` -- this file is read-modify-written from TWO
    concurrent cluster jobs at once (an L4 sweep and an A100 sweep, deliberately, to get both
    cards' numbers in parallel), and flock's semantics over NFS are exactly as reliable as the NFS
    server's lockd, which is to say not guaranteed. `mkdir` is atomic on NFS the way `open(O_EXCL)`
    is not always trusted to be."""
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK_DIR)
            break
        except FileExistsError:
            if time.time() - t0 > timeout:
                try:
                    os.rmdir(LOCK_DIR)          # a lock older than the timeout is a crashed holder
                except OSError:
                    pass
                continue
            time.sleep(0.2 + 0.3 * np.random.random())
    try:
        yield
    finally:
        try:
            os.rmdir(LOCK_DIR)
        except OSError:
            pass


def load_results():
    if os.path.exists(RESULTS_JSON):
        try:
            return json.load(open(RESULTS_JSON))
        except Exception:
            pass
    return {"agent": "larger", "updated": _now(), "scaling": [], "largest_that_runs": {},
            "what_stopped_the_next": "", "vs_paper": {}, "recommended_patches": [],
            "open_questions": []}


def save_results(doc):
    doc["updated"] = _now()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = RESULTS_JSON + f".tmp.{os.getpid()}"
    json.dump(doc, open(tmp, "w"), indent=2)
    os.replace(tmp, RESULTS_JSON)


def _key(e):
    return (e.get("points"), e.get("frames"), e.get("checkpointing"), e.get("seg_frames"),
            e.get("dtype"), e.get("exp"), e.get("gpu"))


def gpu_name(dev):
    """The card, by name, from CUDA itself -- not asserted from which queue this happened to land
    on. `L4` and `A100` jobs both say `gpu_l4`/`gpu_a100` in the submit command, but a number
    without the card it ran on is not comparable, per the coordinator's note."""
    import torch
    try:
        return torch.cuda.get_device_name(dev)
    except Exception:
        return "unknown"


def record_scaling(entry):
    """Append/replace-by-key and rewrite NOW, so an interrupted sweep leaves everything finished
    so far on disk -- the coordinator's requirement, not a nicety. Locked: an L4 sweep and an A100
    sweep write this file from two DIFFERENT cluster jobs at once, on purpose."""
    with _results_lock():
        doc = load_results()
        doc.setdefault("scaling", [])
        doc["scaling"] = [e for e in doc["scaling"] if _key(e) != _key(entry)]
        doc["scaling"].append(entry)
        save_results(doc)
    print(f"    [results.json] {entry.get('exp')} pts={entry.get('points')} "
          f"frames={entry.get('frames')} ckpt={entry.get('checkpointing')} "
          f"seg={entry.get('seg_frames')} gpu={entry.get('gpu')} -> fits={entry.get('fits')} "
          f"peak={entry.get('peak_mem_GB')} GB  s/it={entry.get('s_per_iter')}", flush=True)


def update_doc(**kw):
    with _results_lock():
        doc = load_results()
        doc.update(kw)
        save_results(doc)


def append_open_question(q):
    with _results_lock():
        doc = load_results()
        doc.setdefault("open_questions", [])
        if q not in doc["open_questions"]:
            doc["open_questions"].append(q)
        save_results(doc)


# ------------------------------------------------------------------------------------------- setup
def build_sim(n_pts, n_grid, frames, dt=DT, sub=SUB, compile=True, capture=True):
    """`compile=False, capture=False` for the CHECKPOINTED path, and the reason is a real bug this
    tool found rather than a style choice.

    `engine.run`'s substep block tries to capture a CUDA graph of the substep on tick 1 REGARDLESS
    of `compile` (`step.get("capture", True)`, engine.py:2169) -- a SEPARATE mechanism from
    torch.compile. Its warm-up snapshots every live buffer, replays the substep 3x on a side stream,
    then `_t.copy_(_k)` UNDOES the warm-up back onto the ORIGINAL tensor objects (engine.py:2206) --
    an in-place write, twice (once after warm-up, once after the real capture pass). Inside ONE
    self-contained `engine.run` call that is harmless: nothing outside that call needs the
    pre-capture-attempt tensor. It stops being harmless the moment a caller hands `engine.run` an
    EXTERNALLY OWNED tensor as the tick-0 state -- which chaining does, by construction, so a coarse
    stage's or a previous SEGMENT's state can carry over. `torch.utils.checkpoint` keeps that exact
    tensor object to recompute from during backward, and the in-place `copy_` bumps its version
    counter from under it.

    MEASURED: `RuntimeError: ... modified by an inplace operation: [torch.cuda.FloatTensor
    [1000, 3, 3]] ... is at version 2; expected version 0` -- version 2 because the undo copies
    happen exactly twice -- the first time a checkpointed 2-frame segment's backward ran, at 1,000
    points. Turning `compile` off alone did NOT fix it (`torch.compile ON` stopped printing but
    `substep captured as a CUDA graph` kept happening and the crash was identical) -- `capture` is
    the flag that actually gates this code path. The plain (single, self-contained) rollout is
    unaffected either way and keeps `compile=True, capture=True` to match the existing baseline."""
    from shape_control import spec
    from plexus.schema import load
    raw = spec(n_pts, BOX, n_grid, frames, dt, sub)
    raw["schedule"][0]["compile"] = bool(compile)
    raw["schedule"][0]["capture"] = bool(capture)
    f = os.path.join(tempfile.mkdtemp(prefix="morphscale_"), "spec.yaml")
    yaml.safe_dump(raw, open(f, "w"), sort_keys=False)
    return load(f)


def trilinear_weights(X0n, K, dev, torch):
    c = np.array([CTR, CTR, CTR])
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


def rates(theta, idx, wts):
    return sum(w[:, None] * theta[i] for i, w in zip(idx, wts))


def G_from_a6(a6, dt, dev, torch):
    n = a6.shape[0]
    A = torch.zeros(n, 3, 3, device=dev) + torch.diag_embed(a6[:, :3])
    A[:, 0, 1] = A[:, 1, 0] = a6[:, 3]
    A[:, 0, 2] = A[:, 2, 0] = a6[:, 4]
    A[:, 1, 2] = A[:, 2, 1] = a6[:, 5]
    Adt = -A * dt
    eye = torch.eye(3, device=dev)
    return eye + Adt + 0.5 * (Adt @ Adt)          # 2nd-order expm: A dt is small


def _sync(dev):
    import torch
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


# ---------------------------------------------------------------------------------- plain rollout
def rollout_plain(sim, X0, G, device_str, keep=None):
    import torch
    from plexus import engine

    def cb(Hh, tick, G=G):
        q = Hh.level("mpm_particle")
        if tick == 0:
            pa, pb = q.state_schema["pos"]
            S = q.state.clone()
            S[:, pa:pb] = X0
            q.state = S
        else:
            q.F = torch.bmm(G, q.F)
        if keep is not None:
            keep.append(q.get("pos").detach().cpu().numpy().copy())
    Hh, _ = engine.run(sim, device=device_str, on_frame=cb, progress=False, grad=(keep is None))
    return Hh.level("mpm_particle").get("pos")


def oom_guard(fn, dev):
    """Run `fn()`. On CUDA OOM: report what memory WAS reached (not an assertion that it failed for
    that reason vaguely -- `torch.cuda.max_memory_allocated` still reads back after the failed
    alloc) and leave the allocator clean for the next config in the sweep.

    ALSO catches a torch op that plain refuses a dtype (`NotImplementedError`, e.g.
    `torch.linalg.det` has no BFloat16 cuBLAS path) -- a DIFFERENT failure mode from running out of
    room, and worth telling apart in the record: one says "buy a bigger card or checkpoint more",
    the other says "this precision is not supported here yet, on this torch version"."""
    import torch
    try:
        return fn(), True, None
    except NotImplementedError as e:
        return None, False, f"NotImplementedError: {str(e).splitlines()[0][:280]}"
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" not in msg.lower():
            raise
        torch.cuda.synchronize(dev)
        return None, False, msg.splitlines()[0][:300]
    finally:
        import gc
        gc.collect()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------- exp: sweep
def one_config(n_pts, frames, device_str, iters, ctrl=8, lr=2.0, n_grid=None, dtype="float32"):
    """One (points, frames) config, plain (no checkpointing) rollout: peak memory + s/iteration."""
    import torch
    import plexus.operators  # noqa: F401
    dev = torch.device(device_str)
    n_grid = n_grid or auto_grid(n_pts)
    sim = build_sim(n_pts, n_grid, frames)
    rng = np.random.default_rng(0)
    from shape_control import ball, mass_grid, target_points
    c = np.array([CTR, CTR, CTR])
    X0n = ball(n_pts, c, R, rng)
    X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
    tgt_np = target_points("disc", n_pts, c, rng)
    tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
    one = torch.ones(n_pts, device=dev)
    with torch.no_grad():
        rho_t = mass_grid(tgt, one, BOX, n_grid, dev, torch)
    idx, wts = trilinear_weights(X0n, ctrl, dev, torch)
    theta = torch.zeros(ctrl ** 3, 6, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    dtf = float(sim.dt)
    use_bf16 = dtype == "bf16"

    def step():
        opt.zero_grad()
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16
               else contextlib.nullcontext())
        with ctx:
            a6 = rates(theta, idx, wts)
            G = G_from_a6(a6, dtf, dev, torch)
            Xs = rollout_plain(sim, X0, G, device_str)
            loss = ((torch.log1p(mass_grid(Xs, one, BOX, n_grid, dev, torch))
                     - torch.log1p(rho_t)) ** 2).mean()
        loss.backward()
        opt.step()
        return float(loss)

    torch.cuda.reset_peak_memory_stats(dev)
    times, losses = [], []
    for it in range(iters):
        _sync(dev); t0 = time.time()
        l = step()
        _sync(dev); t1 = time.time()
        times.append(t1 - t0); losses.append(l)
        print(f"      it {it}  loss {l:.6f}  {t1 - t0:.3f} s", flush=True)
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    reserved = torch.cuda.max_memory_reserved(dev) / 1e9
    s_per_iter = float(np.mean(times[1:])) if len(times) > 1 else float(times[0])
    return dict(n_grid=n_grid, peak_mem_GB=peak, peak_reserved_GB=reserved,
                iter_times=[round(t, 4) for t in times], s_per_iter=s_per_iter,
                losses=[round(l, 6) for l in losses])


def exp_sweep(args):
    import torch
    dev = torch.device(args.device)
    pts_list = [int(v) for v in args.pts.split(",")]
    for n_pts in pts_list:
        print(f"\n  [sweep] {n_pts:,} points, {args.frames} frames", flush=True)
        (res, fits, err) = oom_guard(
            lambda: one_config(n_pts, args.frames, args.device, args.iters, ctrl=args.ctrl,
                                lr=args.lr, dtype=args.dtype), dev)
        entry = dict(exp="sweep", points=n_pts, frames=args.frames, gpu=gpu_name(dev),
                     n_grid=(res or {}).get("n_grid", auto_grid(n_pts)), ctrl=args.ctrl,
                     checkpointing=False, seg_frames=None, dtype=args.dtype,
                     peak_mem_GB=round((res or {}).get("peak_mem_GB", torch.cuda.max_memory_allocated(dev) / 1e9), 3),
                     peak_reserved_GB=round((res or {}).get("peak_reserved_GB", 0.0), 3),
                     s_per_iter=round(res["s_per_iter"], 4) if fits else None,
                     iter_times=(res or {}).get("iter_times", []),
                     fits=fits, error=err or "",
                     command=(f"PYTHONPATH=src python tools/morph_scale.py --exp sweep "
                              f"--pts {n_pts} --frames {args.frames} --device {args.device} "
                              f"--iters {args.iters} --ctrl {args.ctrl}"),
                     timestamp=_now())
        record_scaling(entry)
        if not fits:
            print(f"    -> OOM at {n_pts:,} points: {err}", flush=True)
            break


def exp_frames(args):
    """The OTHER axis: peak memory + s/iteration vs ROLLOUT LENGTH at a fixed particle count."""
    import torch
    dev = torch.device(args.device)
    n_pts = args.fixed_pts
    for frames in [int(v) for v in args.frames_list.split(",")]:
        print(f"\n  [frames] {n_pts:,} points, {frames} frames", flush=True)
        (res, fits, err) = oom_guard(
            lambda: one_config(n_pts, frames, args.device, args.iters, ctrl=args.ctrl,
                                lr=args.lr, dtype=args.dtype), dev)
        entry = dict(exp="frames", points=n_pts, frames=frames, gpu=gpu_name(dev),
                     n_grid=(res or {}).get("n_grid", auto_grid(n_pts)), ctrl=args.ctrl,
                     checkpointing=False, seg_frames=None, dtype=args.dtype,
                     peak_mem_GB=round((res or {}).get("peak_mem_GB", torch.cuda.max_memory_allocated(dev) / 1e9), 3),
                     peak_reserved_GB=round((res or {}).get("peak_reserved_GB", 0.0), 3),
                     s_per_iter=round(res["s_per_iter"], 4) if fits else None,
                     iter_times=(res or {}).get("iter_times", []),
                     fits=fits, error=err or "",
                     command=(f"PYTHONPATH=src python tools/morph_scale.py --exp frames "
                              f"--fixed-pts {n_pts} --frames-list {frames} --device {args.device}"),
                     timestamp=_now())
        record_scaling(entry)
        if not fits:
            print(f"    -> OOM at {frames} frames: {err}", flush=True)
            break


# ---------------------------------------------------------------------------------- exp: checkpoint
def build_state0(n_pts, dev, X0, torch):
    F = torch.eye(3, device=dev).expand(n_pts, 3, 3).contiguous()
    C = torch.zeros(n_pts, 3, 3, device=dev)
    Jp = torch.ones(n_pts, device=dev)
    vel = torch.zeros_like(X0)
    return X0, vel, F, C, Jp


def make_segment_fn(device_str):
    import torch
    from plexus import engine

    def seg_fn(pos, vel, F, C, Jp, G, sim_seg):
        def cb(Hh, tick, G=G):
            q = Hh.level("mpm_particle")
            if tick == 0:
                pa, pb = q.state_schema["pos"]
                va, vb = q.state_schema["vel"]
                S = q.state.clone()
                S[:, pa:pb] = pos
                S[:, va:vb] = vel
                q.state = S
                q.F = F
                q.C = C
                q.Jp = Jp
            else:
                q.F = torch.bmm(G, q.F)
        Hh, _ = engine.run(sim_seg, device=device_str, on_frame=cb, progress=False, grad=True)
        q = Hh.level("mpm_particle")
        pa, pb = q.state_schema["pos"]
        va, vb = q.state_schema["vel"]
        return (q.state[:, pa:pb].contiguous(), q.state[:, va:vb].contiguous(),
                q.F.contiguous(), q.C.contiguous(), q.Jp.contiguous())
    return seg_fn


def rollout_checkpoint(n_pts, n_grid, frames, seg_frames, X0, G, device_str, torch):
    """CHAINED multi-pass: one `engine.run` per segment, wrapped in `torch.utils.checkpoint` so only
    ONE segment's tape is resident at a time. The recomputation (checkpoint re-runs `seg_fn` during
    backward) is the price -- roughly a second forward pass per segment -- for memory that no longer
    scales with the WHOLE rollout, only with the longest segment."""
    from torch.utils.checkpoint import checkpoint
    dev = X0.device
    n_full = frames // seg_frames
    rem = frames % seg_frames
    segs = [seg_frames] * n_full + ([rem] if rem else [])
    sim_cache = {}

    def get_sim(nf):
        if nf not in sim_cache:
            sim_cache[nf] = build_sim(n_pts, n_grid, nf, compile=False, capture=False)
        return sim_cache[nf]

    seg_fn = make_segment_fn(device_str)
    pos, vel, F, C, Jp = build_state0(n_pts, dev, X0, torch)
    for nf in segs:
        sim_seg = get_sim(nf)
        pos, vel, F, C, Jp = checkpoint(seg_fn, pos, vel, F, C, Jp, G, sim_seg, use_reentrant=False)
    return pos


def rollout_checkpoint_eval(n_pts, n_grid, frames, seg_frames, X0, G, device_str, torch, keep=None):
    """The same chained segments as `rollout_checkpoint`, but for the FINAL no-grad capture pass:
    no tape, so no need for `torch.utils.checkpoint` at all -- just walk the segments in order,
    carrying the state, collecting frames. Skips the tick-0 re-injection frame on every segment
    AFTER the first, which would otherwise be a duplicate of that segment's own last frame."""
    from plexus import engine
    dev = X0.device
    n_full = frames // seg_frames
    rem = frames % seg_frames
    segs = [seg_frames] * n_full + ([rem] if rem else [])
    pos, vel, F, C, Jp = build_state0(n_pts, dev, X0, torch)
    for si, nf in enumerate(segs):
        sim_seg = build_sim(n_pts, n_grid, nf, compile=False, capture=False)

        def cb(Hh, tick, G=G, first=(si == 0)):
            q = Hh.level("mpm_particle")
            if tick == 0:
                pa, pb = q.state_schema["pos"]
                va, vb = q.state_schema["vel"]
                S = q.state.clone()
                S[:, pa:pb] = pos
                S[:, va:vb] = vel
                q.state = S
                q.F, q.C, q.Jp = F, C, Jp
            else:
                q.F = torch.bmm(G, q.F)
            if keep is not None and (tick > 0 or first):
                keep.append(q.get("pos").detach().cpu().numpy().copy())
        Hh, _ = engine.run(sim_seg, device=device_str, on_frame=cb, progress=False, grad=False)
        q = Hh.level("mpm_particle")
        pa, pb = q.state_schema["pos"]
        va, vb = q.state_schema["vel"]
        pos, vel, F, C, Jp = q.state[:, pa:pb], q.state[:, va:vb], q.F, q.C, q.Jp
    return pos


def checkpoint_config(n_pts, frames, seg_frames, device_str, iters, ctrl=8, lr=2.0, n_grid=None):
    import torch
    import plexus.operators  # noqa: F401
    from shape_control import ball, mass_grid, target_points
    dev = torch.device(device_str)
    n_grid = n_grid or auto_grid(n_pts)
    rng = np.random.default_rng(0)
    c = np.array([CTR, CTR, CTR])
    X0n = ball(n_pts, c, R, rng)
    X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
    tgt_np = target_points("disc", n_pts, c, rng)
    tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
    one = torch.ones(n_pts, device=dev)
    with torch.no_grad():
        rho_t = mass_grid(tgt, one, BOX, n_grid, dev, torch)
    idx, wts = trilinear_weights(X0n, ctrl, dev, torch)
    theta = torch.zeros(ctrl ** 3, 6, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    dtf = DT

    def step():
        opt.zero_grad()
        a6 = rates(theta, idx, wts)
        G = G_from_a6(a6, dtf, dev, torch)
        Xs = rollout_checkpoint(n_pts, n_grid, frames, seg_frames, X0, G, device_str, torch)
        loss = ((torch.log1p(mass_grid(Xs, one, BOX, n_grid, dev, torch))
                 - torch.log1p(rho_t)) ** 2).mean()
        loss.backward()
        opt.step()
        return float(loss)

    torch.cuda.reset_peak_memory_stats(dev)
    times, losses = [], []
    for it in range(iters):
        _sync(dev); t0 = time.time()
        l = step()
        _sync(dev); t1 = time.time()
        times.append(t1 - t0); losses.append(l)
        print(f"      it {it}  loss {l:.6f}  {t1 - t0:.3f} s", flush=True)
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    reserved = torch.cuda.max_memory_reserved(dev) / 1e9
    s_per_iter = float(np.mean(times[1:])) if len(times) > 1 else float(times[0])
    return dict(n_grid=n_grid, peak_mem_GB=peak, peak_reserved_GB=reserved,
                iter_times=[round(t, 4) for t in times], s_per_iter=s_per_iter,
                losses=[round(l, 6) for l in losses])


def exp_checkpoint(args):
    import torch
    dev = torch.device(args.device)
    pts_list = [int(v) for v in args.pts.split(",")]
    seg_list = [int(v) for v in args.seg.split(",")]
    for n_pts in pts_list:
        for seg_frames in seg_list:
            if seg_frames > args.frames:
                continue
            print(f"\n  [checkpoint] {n_pts:,} points, {args.frames} frames, "
                  f"seg={seg_frames}", flush=True)
            (res, fits, err) = oom_guard(
                lambda: checkpoint_config(n_pts, args.frames, seg_frames, args.device, args.iters,
                                           ctrl=args.ctrl, lr=args.lr), dev)
            entry = dict(exp="checkpoint", points=n_pts, frames=args.frames, gpu=gpu_name(dev),
                         n_grid=(res or {}).get("n_grid", auto_grid(n_pts)), ctrl=args.ctrl,
                         checkpointing=True, seg_frames=seg_frames, dtype="float32",
                         peak_mem_GB=round((res or {}).get("peak_mem_GB", torch.cuda.max_memory_allocated(dev) / 1e9), 3),
                         peak_reserved_GB=round((res or {}).get("peak_reserved_GB", 0.0), 3),
                         s_per_iter=round(res["s_per_iter"], 4) if fits else None,
                         iter_times=(res or {}).get("iter_times", []),
                         fits=fits, error=err or "",
                         command=(f"PYTHONPATH=src python tools/morph_scale.py --exp checkpoint "
                                  f"--pts {n_pts} --frames {args.frames} --seg {seg_frames} "
                                  f"--device {args.device}"),
                         timestamp=_now())
            record_scaling(entry)
            if not fits:
                print(f"    -> OOM at {n_pts:,} pts, seg {seg_frames}: {err}", flush=True)


# -------------------------------------------------------------------------------------- exp: verify
def exp_verify(args):
    """Checkpointed vs plain MUST give the same theta.grad -- checkpointing changes only where
    activations are recomputed, never what the forward computes. Small enough to run both ways."""
    import torch
    import plexus.operators  # noqa: F401
    from shape_control import ball, mass_grid, target_points
    dev = torch.device(args.device)
    n_pts, frames, seg_frames, ctrl = args.verify_pts, args.frames, args.seg_verify, args.ctrl
    n_grid = auto_grid(n_pts)
    rng = np.random.default_rng(0)
    c = np.array([CTR, CTR, CTR])
    X0n = ball(n_pts, c, R, rng)
    X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
    tgt_np = target_points("disc", n_pts, c, rng)
    tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
    one = torch.ones(n_pts, device=dev)
    with torch.no_grad():
        rho_t = mass_grid(tgt, one, BOX, n_grid, dev, torch)
    idx, wts = trilinear_weights(X0n, ctrl, dev, torch)

    def run_once(checkpointing):
        theta = torch.zeros(ctrl ** 3, 6, device=dev, requires_grad=True)
        with torch.no_grad():
            theta.add_(0.05 * torch.randn_like(theta))    # away from the theta=0 saddle
        theta.requires_grad_(True)
        a6 = rates(theta, idx, wts)
        G = G_from_a6(a6, DT, dev, torch)
        if checkpointing:
            sim = None
            Xs = rollout_checkpoint(n_pts, n_grid, frames, seg_frames, X0, G, args.device, torch)
        else:
            sim = build_sim(n_pts, n_grid, frames)
            Xs = rollout_plain(sim, X0, G, args.device)
        loss = ((torch.log1p(mass_grid(Xs, one, BOX, n_grid, dev, torch))
                 - torch.log1p(rho_t)) ** 2).mean()
        loss.backward()
        return float(loss), theta.grad.detach().clone()

    # SAME theta INIT BOTH TIMES: seed the RNG identically around each call.
    torch.manual_seed(0)
    loss_a, grad_a = run_once(False)
    torch.manual_seed(0)
    loss_b, grad_b = run_once(True)
    rel = float((grad_a - grad_b).norm() / grad_a.norm().clamp(min=1e-12))
    loss_rel = abs(loss_a - loss_b) / max(abs(loss_a), 1e-12)
    print(f"\n  [verify] plain loss {loss_a:.6f}  checkpoint loss {loss_b:.6f}  "
          f"|grad_plain-grad_ckpt|/|grad_plain| = {rel:.3e}", flush=True)
    entry = dict(exp="verify", points=n_pts, frames=frames, seg_frames=seg_frames, gpu=gpu_name(dev),
                 checkpointing="both", dtype="float32", loss_plain=round(loss_a, 6),
                 loss_checkpoint=round(loss_b, 6), loss_rel_diff=loss_rel,
                 grad_rel_diff=rel, agrees=bool(rel < 1e-2),
                 command=(f"PYTHONPATH=src python tools/morph_scale.py --exp verify "
                          f"--verify-pts {n_pts} --frames {frames} --seg-verify {seg_frames} "
                          f"--device {args.device}"),
                 timestamp=_now())
    record_scaling(entry)
    if rel >= 1e-2:
        append_open_question(
            f"verify: checkpointed vs plain theta.grad differ by {rel:.3e} (relative) at "
            f"{n_pts} pts / {frames} frames / seg {seg_frames} -- ABOVE the 1e-2 tolerance, "
            f"needs a look before trusting the checkpoint numbers at scale.")


# ------------------------------------------------------------------------------------- exp: transfer
def exp_transfer(args):
    """Can a coarse stage's control be READ at a fine stage's particles with NO further
    optimisation? Train theta at n_coarse against a REAL target mesh, then evaluate (no_grad) at
    n_fine with weights recomputed from the fine particles' own material coordinates. Compare that
    zero-shot loss to theta=0 (no control at all) and to the trained-coarse loss itself."""
    import torch
    import plexus.operators  # noqa: F401
    from shape_control import ball, mass_grid
    from morph_gallery import sample_inside
    dev = torch.device(args.device)
    n_coarse, n_fine, ctrl = args.coarse, args.fine, args.ctrl
    target_path = os.path.join(MODELS, f"{args.target}.obj")
    rng = np.random.default_rng(0)
    c = np.array([CTR, CTR, CTR])

    def make(n_pts):
        n_grid = auto_grid(n_pts)
        X0n = ball(n_pts, c, R, rng)
        X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
        tgt_np = sample_inside(target_path, n_pts, c, 2.6 * R, rng)
        tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
        one = torch.ones(n_pts, device=dev)
        with torch.no_grad():
            rho_t = mass_grid(tgt, one, BOX, n_grid, dev, torch)
        return n_grid, X0n, X0, one, rho_t

    n_grid_c, X0n_c, X0_c, one_c, rho_t_c = make(n_coarse)
    idx_c, wts_c = trilinear_weights(X0n_c, ctrl, dev, torch)
    theta = torch.zeros(ctrl ** 3, 6, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=args.lr)
    sim_c = build_sim(n_coarse, n_grid_c, args.frames)
    for it in range(args.iters):
        opt.zero_grad()
        a6 = rates(theta, idx_c, wts_c)
        G = G_from_a6(a6, float(sim_c.dt), dev, torch)
        Xs = rollout_plain(sim_c, X0_c, G, args.device)
        loss = ((torch.log1p(mass_grid(Xs, one_c, BOX, n_grid_c, dev, torch))
                 - torch.log1p(rho_t_c)) ** 2).mean()
        loss.backward(); opt.step()
        if it % 20 == 0 or it == args.iters - 1:
            print(f"      coarse it {it}  loss {float(loss):.6f}", flush=True)
    loss_coarse = float(loss)

    n_grid_f, X0n_f, X0_f, one_f, rho_t_f = make(n_fine)
    idx_f, wts_f = trilinear_weights(X0n_f, ctrl, dev, torch)   # SAME ctrl cube, FINE particles' own weights
    sim_f = build_sim(n_fine, n_grid_f, args.frames)

    # theta=0 baseline: no control at all (identity growth every frame)
    with torch.no_grad():
        G_zero = torch.eye(3, device=dev).expand(n_fine, 3, 3).contiguous()
        Xs_zero = rollout_plain(sim_f, X0_f, G_zero, args.device)
        loss_zero = float(((torch.log1p(mass_grid(Xs_zero, one_f, BOX, n_grid_f, dev, torch))
                            - torch.log1p(rho_t_f)) ** 2).mean())

    with torch.no_grad():
        a6_transfer = rates(theta, idx_f, wts_f)
        G_transfer = G_from_a6(a6_transfer, float(sim_f.dt), dev, torch)
        Xs_zeroshot = rollout_plain(sim_f, X0_f, G_transfer, args.device)
        loss_zeroshot = float(((torch.log1p(mass_grid(Xs_zeroshot, one_f, BOX, n_grid_f, dev, torch))
                               - torch.log1p(rho_t_f)) ** 2).mean())

    print(f"\n  [transfer] {args.target}: coarse({n_coarse:,})={loss_coarse:.6f}  "
          f"fine-zero-shot({n_fine:,})={loss_zeroshot:.6f}  fine-no-control={loss_zero:.6f}",
          flush=True)
    entry = dict(exp="transfer", target=args.target, coarse_pts=n_coarse, fine_pts=n_fine,
                 gpu=gpu_name(dev), ctrl=ctrl, frames=args.frames, iters_coarse=args.iters,
                 loss_coarse=round(loss_coarse, 6), loss_fine_zero_shot=round(loss_zeroshot, 6),
                 loss_fine_no_control=round(loss_zero, 6),
                 transfers=bool(loss_zeroshot < 0.5 * loss_zero),
                 command=(f"PYTHONPATH=src python tools/morph_scale.py --exp transfer "
                          f"--coarse {n_coarse} --fine {n_fine} --target {args.target} "
                          f"--ctrl {ctrl} --iters {args.iters} --device {args.device}"),
                 timestamp=_now())
    record_scaling(entry)


# ------------------------------------------------------------------------------------ exp: precision
def exp_precision(args):
    import torch
    dev = torch.device(args.device)
    n_pts = args.prec_pts
    for dtype in ["float32", "bf16"]:
        print(f"\n  [precision] {n_pts:,} points, dtype={dtype}", flush=True)
        (res, fits, err) = oom_guard(
            lambda: one_config(n_pts, args.frames, args.device, args.iters, ctrl=args.ctrl,
                                lr=args.lr, dtype=dtype), dev)
        entry = dict(exp="precision", points=n_pts, frames=args.frames, dtype=dtype, gpu=gpu_name(dev),
                     checkpointing=False, seg_frames=None,
                     n_grid=(res or {}).get("n_grid", auto_grid(n_pts)),
                     peak_mem_GB=round((res or {}).get("peak_mem_GB", torch.cuda.max_memory_allocated(dev) / 1e9), 3),
                     s_per_iter=round(res["s_per_iter"], 4) if fits else None,
                     fits=fits, error=err or "",
                     command=(f"PYTHONPATH=src python tools/morph_scale.py --exp precision "
                              f"--prec-pts {n_pts} --device {args.device}"),
                     timestamp=_now())
        record_scaling(entry)


# -------------------------------------------------------------------------------------- exp: rerender
def exp_rerender(args):
    """Redraw an already-saved `morph.npz` with the CURRENT `morph_gallery.render` -- no GPU
    training repeated. Exists because the renderer changed (the RENDER agent's approved look,
    176^3 reconstruction) AFTER some of our rollouts had already been computed: the expensive part
    (the optimisation) is still good, only the picture needs to be redrawn with the same look
    everyone else's stills use."""
    from morph_gallery import render
    d = args.rerender_dir
    z = np.load(os.path.join(d, "morph.npz"))
    render(d, z["frames"], z["target"], float(z["box"]), args.rerender_name)
    print(f"    -> re-rendered {os.path.relpath(d, ROOT)}", flush=True)


# --------------------------------------------------------------------------------------- exp: render
def exp_render(args):
    """One demonstrative morph at the largest configuration measured to fit, via
    `morph_gallery.render` (imported, not modified) -- so the scale claim has a picture.

    `--render-seg > 0` trains through the CHECKPOINTED path (needed the moment `render-pts` is
    past the plain L4 wall, ~50,000 points at 20 frames) and captures the final movie by walking
    the same chained segments once more under `no_grad` -- no checkpoint needed there, since there
    is no tape to save memory on."""
    import torch
    import plexus.operators  # noqa: F401
    from shape_control import ball, mass_grid
    from morph_gallery import sample_inside, render
    dev = torch.device(args.device)
    n_pts, n_grid, frames, ctrl = args.render_pts, auto_grid(args.render_pts), args.frames, args.ctrl
    seg = args.render_seg
    rng = np.random.default_rng(0)
    c = np.array([CTR, CTR, CTR])
    path = os.path.join(MODELS, f"{args.target}.obj")
    X0n = ball(n_pts, c, R, rng)
    X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
    tgt_np = sample_inside(path, n_pts, c, 2.6 * R, rng)
    tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
    one = torch.ones(n_pts, device=dev)
    with torch.no_grad():
        rho_t = mass_grid(tgt, one, BOX, n_grid, dev, torch)
    idx, wts = trilinear_weights(X0n, ctrl, dev, torch)
    theta = torch.zeros(ctrl ** 3, 6, device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=args.lr)
    dtf = DT if seg else float(build_sim(n_pts, n_grid, frames).dt)
    t0 = time.time()
    for it in range(args.iters):
        opt.zero_grad()
        a6 = rates(theta, idx, wts)
        G = G_from_a6(a6, dtf, dev, torch)
        Xs = (rollout_checkpoint(n_pts, n_grid, frames, seg, X0, G, args.device, torch) if seg
              else rollout_plain(build_sim(n_pts, n_grid, frames), X0, G, args.device))
        loss = ((torch.log1p(mass_grid(Xs, one, BOX, n_grid, dev, torch))
                 - torch.log1p(rho_t)) ** 2).mean()
        loss.backward(); opt.step()
        if it % 10 == 0 or it == args.iters - 1:
            print(f"      it {it}  loss {float(loss):.6f}", flush=True)
    with torch.no_grad():
        frames_out = []
        a6 = rates(theta, idx, wts)
        G = G_from_a6(a6, dtf, dev, torch)
        if seg:
            rollout_checkpoint_eval(n_pts, n_grid, frames, seg, X0, G, args.device, torch,
                                    keep=frames_out)
        else:
            rollout_plain(build_sim(n_pts, n_grid, frames), X0, G, args.device, keep=frames_out)
    d = os.path.join(RESULTS_DIR, f"morph_{args.target}_{n_pts}")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "morph.npz"), frames=np.stack(frames_out), target=tgt_np,
                        box=BOX, control=theta.detach().cpu().numpy(), seconds=time.time() - t0)
    render(d, np.stack(frames_out), tgt_np, BOX, f"{args.target} ({n_pts:,} pts)")
    print(f"    -> {os.path.relpath(d, ROOT)}", flush=True)
    entry = dict(exp="render", points=n_pts, frames=frames, target=args.target, gpu=gpu_name(dev),
                 checkpointing=bool(seg), seg_frames=(seg or None),
                 minutes=(time.time() - t0) / 60.0, out_dir=d, timestamp=_now())
    record_scaling(entry)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True,
                    choices=["sweep", "frames", "checkpoint", "verify", "transfer", "precision",
                             "render", "rerender"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--ctrl", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bf16"])
    # sweep / precision
    ap.add_argument("--pts",
                    default="12500,25000,50000,75000,100000,150000,200000,300000,500000,"
                            "1000000,2000000")
    # frames
    ap.add_argument("--fixed-pts", type=int, default=50000)
    ap.add_argument("--frames-list", default="10,20,40,80")
    # checkpoint
    ap.add_argument("--seg", default="2,4,10")
    # verify
    ap.add_argument("--verify-pts", type=int, default=2000)
    ap.add_argument("--seg-verify", type=int, default=2)
    # transfer
    ap.add_argument("--coarse", type=int, default=12500)
    ap.add_argument("--fine", type=int, default=100000)
    ap.add_argument("--target", default="cow")
    # precision
    ap.add_argument("--prec-pts", type=int, default=200000)
    # render
    ap.add_argument("--render-pts", type=int, default=200000)
    ap.add_argument("--render-seg", type=int, default=0,
                    help="checkpoint segment length in frames; 0 = plain rollout (only valid "
                         "under the plain L4/A100 wall)")
    # rerender
    ap.add_argument("--rerender-dir", default="")
    ap.add_argument("--rerender-name", default="")
    args = ap.parse_args()

    {"sweep": exp_sweep, "frames": exp_frames, "checkpoint": exp_checkpoint,
     "verify": exp_verify, "transfer": exp_transfer, "precision": exp_precision,
     "render": exp_render, "rerender": exp_rerender}[args.exp](args)


if __name__ == "__main__":
    main()
