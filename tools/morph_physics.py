"""Physical-fidelity diagnostics for the MPM shape morph -- owned by the MORE_PRECISE agent.

`tools/morph_gallery.py` morphs a ball into a target by optimising a coarse grid of
deformation-gradient RATES A_i (symmetric 3x3, trilinearly read at each particle's material
coordinate) and applying F <- expm(-A_p dt) F once a FRAME. This file does not change that loop
-- it asks two questions of it:

1. WHAT CAUSES "the break in the cow." Candidates, each with the measurement that would convict
   it and none other:
       a. neighbouring control nodes pulling oppositely  -> per-particle CONTROL DISAGREEMENT
          score: the max pairwise ||A_i - A_j||_F among the (up to 8) trilinear corner nodes a
          particle actually blends, correlated against that particle's own max|J-1| over the run.
       b. too few particles in a grid cell -> occupied-cell particle-count histogram each frame,
          correlated the same way.
       c. the a_max clamp on external acceleration -> irrelevant BY CONSTRUCTION on this spec:
          `mpm_particle` has no ancestor set, so `_ancestor_accel` returns all zeros regardless of
          a_max (verified at runtime, not assumed -- see `_check_a_max_inert`).
       d. drag -- an ablation on a frozen control (drag changes the ROLLOUT, not what the
          optimiser found, so this isolates the dynamical effect from the optimisation effect).
       e. the substep against the CFL bound -- computed analytically (dx/sqrt(E/rho)) AND swept
          empirically past the bound to find where it actually breaks.
   Also checked: whether the 2nd-order series for expm(-A dt) the rollout uses is close enough to
   `torch.matrix_exp` at the rates the optimiser actually finds.

2. WHETHER THE MORPH CONSERVES VOLUME. The control is symmetric, not deviatoric, so its trace is
   free: material volume V(t) = sum_p p_vol_p * J_p(t) can drift. Tracked every frame; a
   trace-free (deviatoric) control and a volume penalty in the loss are both tested as fixes and
   compared to what the paper does (see NOTE_PAPER_VOLUME below -- neither; it relies solely on
   the log-mass loss and a POST-hoc temporal smoothing of F across control layers, not treated
   here since it is a wholly different control law -- see NOTE_PAPER_CONTROL).

NOTE_PAPER_CONTROL: papers/Xu_2024_mpm_shape_morphing.pdf ADDS a control deformation gradient,
P(F + F~), directly to the elastic F every step; it never multiplies F by expm(-A dt). Its
stabiliser is a TEMPORAL low-pass on F between control layers (Fn+1 <- (1-g)Fn+1 + g Fn, g~0.95),
applied in both forward and backward passes -- not a spatial smoothness term on a control grid,
which does not exist in the paper (their control is per-PARTICLE, not a coarse grid interpolated
between neighbours). This repo's control has no temporal filter at all, and reads a coarse grid by
trilinear interpolation, which is the one thing the paper's design never has to contend with.

NOTE_PAPER_VOLUME: the paper reports no explicit volume/incompressibility term anywhere (checked
the full text). It relies entirely on the log-based nodal mass loss (Eq. 7) to keep the morph from
"mass ejection," and says nothing about total volume drift.

Every run here is a fresh cluster job (queue gpu_l4, jobs/morph_physics.sh); nothing runs on a
local GPU. Outputs land in
    /groups/saalfeld/home/allierc/GraphData/graphs_data/si_material/more_precise/
one `results_<tag>.json` per experiment (merged into the shared `results.json` by the orchestrator
so concurrent cluster jobs never race on one file), plus a movie.mp4 / still_*.png for the main
`tear` experiment.

    PYTHONPATH=src python tools/morph_physics.py --tag tear --exp tear --targets cow
"""
from __future__ import annotations

import argparse
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

BOX, C, R = 0.35, 0.175, 0.055                    # the box / ball centre / ball radius shape_control.py uses
OUT_DEFAULT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/si_material/more_precise"


# ============================================================================================
# control field: trilinear weights on a KxKxK cube, read at the MATERIAL coordinate -- the same
# construction morph_gallery.py uses, duplicated here (a few lines) rather than imported, since
# morph_gallery.py keeps it inlined in `main()` and is not this agent's file to refactor.
# ============================================================================================
def ctrl_weights(X0n, c, R, K):
    u = np.clip((X0n - (c - R)) / (2 * R) * (K - 1), 0, K - 1.001)
    b = np.floor(u).astype(int)
    f = u - b
    idx_np, wts_np = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (((1 - f[:, 0]) if dx == 0 else f[:, 0])
                     * ((1 - f[:, 1]) if dy == 0 else f[:, 1])
                     * ((1 - f[:, 2]) if dz == 0 else f[:, 2]))
                i = ((b[:, 0] + dx).clip(0, K - 1) * K + (b[:, 1] + dy).clip(0, K - 1)) * K \
                    + (b[:, 2] + dz).clip(0, K - 1)
                idx_np.append(i)
                wts_np.append(w)
    return idx_np, wts_np


def sym_from_a6(a6):
    """(...,6) -> (...,3,3) symmetric. a6 = [Axx, Ayy, Azz, Axy, Axz, Ayz]."""
    n = a6.shape[0]
    A = np.zeros((n, 3, 3), dtype=a6.dtype)
    A[:, 0, 0], A[:, 1, 1], A[:, 2, 2] = a6[:, 0], a6[:, 1], a6[:, 2]
    A[:, 0, 1] = A[:, 1, 0] = a6[:, 3]
    A[:, 0, 2] = A[:, 2, 0] = a6[:, 4]
    A[:, 1, 2] = A[:, 2, 1] = a6[:, 5]
    return A


def control_disagreement(theta_np, idx_np, wts_np, wmin=0.02):
    """Per particle: the largest ||A_i - A_j||_F among the (up to 8) trilinear corner control
    nodes it actually blends between (weight > wmin). Zero for a particle sitting on a single
    control node (no disagreement possible); large for one straddling a boundary between two
    control cells whose rates point opposite ways. THIS is the measurement for "neighbouring
    control-grid nodes pulling in opposite directions" -- not an assertion, a per-particle number
    that can be correlated against where the body actually tears."""
    A = sym_from_a6(theta_np)                                     # (K^3, 3, 3)
    corners = np.stack(idx_np, axis=1)                            # (n, 8)
    wts = np.stack(wts_np, axis=1)                                # (n, 8)
    Ac = A[corners]                                                # (n, 8, 3, 3)
    mask = wts > wmin
    diff = Ac[:, :, None] - Ac[:, None, :]                        # (n, 8, 8, 3, 3)
    fro = np.sqrt((diff ** 2).sum(axis=(-1, -2)))                 # (n, 8, 8)
    pair_ok = mask[:, :, None] & mask[:, None, :]
    fro = np.where(pair_ok, fro, -1.0)
    return fro.max(axis=(1, 2))                                    # (n,)


def control_grid_discontinuity(theta_np, K):
    """Neighbour-to-neighbour ||A_i - A_j||_F on the KxKxK control lattice itself (not per
    particle): max / mean / where. The lattice-level twin of `control_disagreement`."""
    A = sym_from_a6(theta_np).reshape(K, K, K, 3, 3)
    out = {}
    diffs = []
    for axis, name in enumerate(("x", "y", "z")):
        d = np.linalg.norm((A[1:] - A[:-1]) if axis == 0 else
                            (A[:, 1:] - A[:, :-1]) if axis == 1 else
                            (A[:, :, 1:] - A[:, :, :-1]), axis=(-1, -2))
        diffs.append(d.ravel())
        out[f"max_{name}"] = float(d.max())
        out[f"mean_{name}"] = float(d.mean())
    all_d = np.concatenate(diffs)
    out["max"] = float(all_d.max())
    out["mean"] = float(all_d.mean())
    return out


def particles_per_cell(pos_np, box, n_grid):
    """Hard-binned (nearest cell, not trilinear) particle count per grid cell -- the MPM rule of
    thumb wants ~2^3 = 8 per cell in 3D for the stress sample to mean anything. Returns
    (counts_of_occupied_cells, per_particle_cell_count)."""
    dx = box / n_grid
    idx = np.clip((pos_np / dx).astype(np.int64), 0, n_grid - 1)
    flat = (idx[:, 0] * n_grid + idx[:, 1]) * n_grid + idx[:, 2]
    counts = np.bincount(flat, minlength=n_grid ** 3)
    occ = counts[counts > 0]
    per_particle = counts[flat]
    return occ, per_particle


def mass_grid_loss(pos, tgt_rho, box, n_grid, dev, torch):
    """The training loss, recomputed post-hoc on a frozen rollout for an apples-to-apples number."""
    from shape_control import mass_grid
    one = torch.ones(pos.shape[0], device=dev)
    rho = mass_grid(pos, one, box, n_grid, dev, torch)
    return float(((torch.log1p(rho) - torch.log1p(tgt_rho)) ** 2).mean())


def cfl_dt(box, n_grid, youngs, rho=1.0):
    dx = box / n_grid
    c = (youngs / rho) ** 0.5
    return dx / c, dx, c


# ============================================================================================
# spec construction -- shape_control.spec(), with the overrides this agent needs to test
# (drag / a_max / youngs / substep) EXPOSED as call kwargs rather than hand-edited each time.
# nu is not a spec field anywhere in this codebase (`plexus.models.entities._NU = 0.2` is a
# module CONSTANT, not read from any type dict) -- see `nu_patch` below for how it is tested.
# ============================================================================================
def make_spec(n_pts, n_grid, frames, dt, sub, youngs=90.0, drag=0.5, a_max=200.0):
    from shape_control import spec as base_spec
    raw = base_spec(n_pts, BOX, n_grid, frames, dt, sub)
    raw["sets"]["mpm_particle"]["types"]["cyto"]["youngs"] = float(youngs)
    for op in raw["operators"]:
        if op["op"] == "mpm_scatter":
            op["drag"] = float(drag)
            op["a_max"] = float(a_max)
    return raw


def load_sim(raw):
    from plexus.schema import load
    f = os.path.join(tempfile.mkdtemp(prefix="morphphys_"), "spec.yaml")
    yaml.safe_dump(raw, open(f, "w"), sort_keys=False)
    return load(f)


class nu_patch:
    """Poisson ratio is not a spec field: `plexus.models.entities._lame` reads a module-level
    constant `_NU = 0.2`. Testing nu != 0.2 without editing that shared file means patching the
    constant around the `schema.load` call that provisions the MPM buffers, then restoring it --
    a context manager so a crash cannot leave the module in a changed state for anyone after."""

    def __init__(self, nu):
        self.nu = nu

    def __enter__(self):
        if self.nu is None:
            self._old = None
            return self
        import plexus.models.entities as entities
        self._old = entities._NU
        entities._NU = float(self.nu)
        return self

    def __exit__(self, *a):
        if self._old is not None:
            import plexus.models.entities as entities
            entities._NU = self._old


# ============================================================================================
# training -- the same coarse-to-fine grid-control loop as morph_gallery.py, instrumented and
# parameterised for the physics knobs this agent tests. Deliberately NOT imported from
# morph_gallery.py (which keeps this inlined in `main()`, not a function this file could call).
# ============================================================================================
def train_control(target, stages, K, lr, frames, dev, rng, youngs=90.0, drag=0.5, a_max=200.0,
                   dt_sub=3.4e-4, trace_free=False, vol_weight=0.0, expm_method="series2",
                   nu=None, log=print):
    import torch
    import plexus.operators  # noqa: F401
    from plexus import engine
    from shape_control import ball, mass_grid
    from morph_gallery import sample_inside

    theta = None
    path = os.path.join(MODELS, f"{target}.obj")
    history = []
    t0 = time.time()
    n_pts = n_grid = None
    idx = wts = idx_np = wts_np = X0n = tgt_np = X0 = None

    for si, (n_pts, n_grid, iters) in enumerate(stages):
        raw = make_spec(n_pts, n_grid, frames, 0.002, dt_sub, youngs=youngs, drag=drag, a_max=a_max)
        with nu_patch(nu):
            sim = load_sim(raw)
        X0n = ball(n_pts, np.array([C, C, C]), R, rng)
        X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
        one = torch.ones(n_pts, device=dev)
        tgt_np = sample_inside(path, n_pts, np.array([C, C, C]), 2.6 * R, rng)
        tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
        with torch.no_grad():
            rho_t = mass_grid(tgt, one, BOX, n_grid, dev, torch)

        idx_np, wts_np = ctrl_weights(X0n, np.array([C, C, C]), R, K)
        idx = [torch.as_tensor(i, device=dev) for i in idx_np]
        wts = [torch.as_tensor(w, dtype=torch.float32, device=dev) for w in wts_np]
        if theta is None:
            theta = torch.zeros(K ** 3, 6, device=dev, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1), eta_min=lr / 20.0)
        eye = torch.eye(3, device=dev)
        dt = float(sim.dt)
        t_stage = time.time()

        def rates():
            a6 = sum(w[:, None] * theta[i] for i, w in zip(idx, wts))
            if trace_free:
                tr = a6[:, 0] + a6[:, 1] + a6[:, 2]
                a6 = a6.clone()
                a6[:, 0] = a6[:, 0] - tr / 3.0
                a6[:, 1] = a6[:, 1] - tr / 3.0
                a6[:, 2] = a6[:, 2] - tr / 3.0
            return a6

        def rollout(keep=None, keepF=False):
            a6 = rates()
            A = torch.zeros(a6.shape[0], 3, 3, device=dev) + torch.diag_embed(a6[:, :3])
            A[:, 0, 1] = A[:, 1, 0] = a6[:, 3]
            A[:, 0, 2] = A[:, 2, 0] = a6[:, 4]
            A[:, 1, 2] = A[:, 2, 1] = a6[:, 5]
            if expm_method == "exact":
                G = torch.matrix_exp(-A * dt)
            else:
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
                if keepF is not False and tick > 0:
                    keepF.append(float(torch.linalg.det(q.F).abs().sub(1).abs().max()))
            Hh, _ = engine.run(sim, device=str(dev), on_frame=cb, progress=False, grad=(keep is None))
            return Hh, Hh.level("mpm_particle").get("pos")

        for it in range(iters):
            opt.zero_grad()
            _, Xs = rollout()
            loss_mass = ((torch.log1p(mass_grid(Xs, one, BOX, n_grid, dev, torch))
                          - torch.log1p(rho_t)) ** 2).mean()
            loss = loss_mass
            if vol_weight > 0:
                a6 = rates()
                if trace_free:
                    loss_vol = torch.zeros((), device=dev)
                else:
                    tr = a6[:, 0] + a6[:, 1] + a6[:, 2]
                    Gdet = torch.exp(-tr * dt * frames)          # det(expm(-A dt))^frames, constant-rate approx
                    loss_vol = (Gdet.mean() - 1.0) ** 2
                loss = loss + vol_weight * loss_vol
            loss.backward()
            opt.step(); sched.step()
            if it % 20 == 0 or it == iters - 1:
                e = ((Xs.max(0).values - Xs.min(0).values) * 100).tolist()
                log(f"    stage {si} ({n_pts:,} pts, grid {n_grid}) iter {it:3d}  "
                    f"loss {float(loss):.6f} (mass {float(loss_mass):.6f})  "
                    f"shape {e[0]:5.1f} x {e[1]:5.1f} x {e[2]:5.1f} um", flush=True)
            history.append(dict(stage=si, iter=it, loss=float(loss), loss_mass=float(loss_mass)))
        log(f"    stage {si}: {iters} iterations in {(time.time()-t_stage)/60:.2f} min "
            f"({(time.time()-t_stage)/max(iters,1):.2f} s/iter)", flush=True)
        # MEMORY HYGIENE BETWEEN STAGES. Each stage builds its own `sim` (fresh CUDA-graph pool +
        # torch.compile artifacts for a shape gpu_l4's own baseline never has to revisit -- a
        # single-stage run). Measured: the 3-stage default (5000:24:80, 12500:32:80, 50000:40:80)
        # OOMs entering the 50000/grid-40 stage on gpu_l4's 22 GiB (`CUDA out of memory ... 22.05
        # GiB memory in use`), where the README's baseline numbers were measured on a 48 GiB RTX
        # A6000. Without this, the prior stage's compiled kernels and captured graph memory pool
        # are still resident when the next stage tries to compile and capture its own -- freed
        # here so three stages fit in gpu_l4's budget instead of one.
        del sim, opt, sched, X0, tgt, rho_t, one, idx, wts
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.compiler.reset()
        except Exception:
            pass

    return dict(theta=theta.detach().cpu().numpy(), idx_np=idx_np, wts_np=wts_np, K=K,
                X0n=X0n, tgt_np=tgt_np, n_pts=n_pts, n_grid=n_grid, history=history,
                seconds=time.time() - t0, dt_sub=dt_sub, youngs=youngs, drag=drag, a_max=a_max,
                trace_free=trace_free, vol_weight=vol_weight, expm_method=expm_method, nu=nu,
                frames=frames)


# ============================================================================================
# diagnosis -- replay a FROZEN control (from train_control, or an ablation override) through the
# full rollout, at whatever render-frame count / physics knobs are asked, and record everything
# needed to answer "what tears" and "does it conserve volume."
# ============================================================================================
def diagnose_rollout(trained, dev, render_frames=None, drag=None, a_max=None, dt_sub=None,
                      youngs=None, expm_method=None, nu=None, log=print):
    import torch
    import plexus.operators  # noqa: F401
    from plexus import engine

    theta_np = trained["theta"]; idx_np = trained["idx_np"]; wts_np = trained["wts_np"]
    n_pts = trained["n_pts"]; n_grid = trained["n_grid"]; X0n = trained["X0n"]; tgt_np = trained["tgt_np"]
    frames = render_frames or trained["frames"]
    drag = trained["drag"] if drag is None else drag
    a_max = trained["a_max"] if a_max is None else a_max
    dt_sub = trained["dt_sub"] if dt_sub is None else dt_sub
    youngs = trained["youngs"] if youngs is None else youngs
    expm_method = trained["expm_method"] if expm_method is None else expm_method
    nu = trained.get("nu") if nu is None else nu

    dt_frame = 0.002 * trained["frames"] / frames         # same total deformation, finer/coarser sampling
    raw = make_spec(n_pts, n_grid, frames, dt_frame, dt_sub, youngs=youngs, drag=drag, a_max=a_max)
    with nu_patch(nu):
        sim = load_sim(raw)
    dt = float(sim.dt)

    X0 = torch.as_tensor(X0n, dtype=torch.float32, device=dev)
    theta = torch.as_tensor(theta_np, dtype=torch.float32, device=dev)
    idx = [torch.as_tensor(i, device=dev) for i in idx_np]
    wts = [torch.as_tensor(w, dtype=torch.float32, device=dev) for w in wts_np]
    eye = torch.eye(3, device=dev)

    diag = dict(volume=[], max_absJm1=[], min_ppc=[], p05_ppc=[], mean_ppc=[], frac_ppc_lt8=[],
                J_final=None, ppc_final=None, ancestors_mpm_particle=None, cfl=None)
    frames_pos = []
    state = {"p_vol0": None, "V0": None, "amax_hit": False}

    def record(q, tag):
        F = q.F.detach()
        J = torch.linalg.det(F)
        if state["p_vol0"] is None:
            state["p_vol0"] = q.p_vol.detach().clone()
            state["V0"] = float((state["p_vol0"] * 1.0).sum())
        V = float((state["p_vol0"] * J).sum())
        diag["volume"].append(V)
        diag["max_absJm1"].append(float((J - 1).abs().max()))
        pos = q.get("pos").detach().cpu().numpy()
        occ, ppc = particles_per_cell(pos, BOX, n_grid)
        diag["min_ppc"].append(int(occ.min()) if occ.size else 0)
        diag["p05_ppc"].append(float(np.percentile(occ, 5)) if occ.size else 0.0)
        diag["mean_ppc"].append(float(occ.mean()) if occ.size else 0.0)
        diag["frac_ppc_lt8"].append(float((occ < 8).mean()) if occ.size else 1.0)
        if tag == "final":
            diag["J_final"] = (J - 1).abs().detach().cpu().numpy()
            diag["ppc_final"] = ppc

    def cb(Hh, tick):
        q = Hh.level("mpm_particle")
        if diag["ancestors_mpm_particle"] is None:
            diag["ancestors_mpm_particle"] = list(Hh.ancestors(q.name))
        if tick == 0:
            q0, q1 = q.state_schema["pos"]
            with torch.no_grad():
                q.state[:, q0:q1] = X0
            record(q, "init")
            return
        a6 = sum(w[:, None] * theta[i] for i, w in zip(idx, wts))
        A = torch.zeros(a6.shape[0], 3, 3, device=dev) + torch.diag_embed(a6[:, :3])
        A[:, 0, 1] = A[:, 1, 0] = a6[:, 3]
        A[:, 0, 2] = A[:, 2, 0] = a6[:, 4]
        A[:, 1, 2] = A[:, 2, 1] = a6[:, 5]
        if expm_method == "exact":
            G = torch.matrix_exp(-A * dt)
        else:
            Adt = -A * dt
            G = eye + Adt + 0.5 * (Adt @ Adt)
        with torch.no_grad():
            q.F = torch.bmm(G, q.F)
        record(q, "final" if tick == frames else "mid")
        frames_pos.append(q.get("pos").detach().cpu().numpy().copy())

    with torch.no_grad():
        Hh, _ = engine.run(sim, device=str(dev), on_frame=cb, progress=False, grad=False)

    from shape_control import mass_grid
    Xs = Hh.level("mpm_particle").get("pos")
    tgt = torch.as_tensor(tgt_np, dtype=torch.float32, device=dev)
    one = torch.ones(n_pts, device=dev)
    with torch.no_grad():
        rho_t = mass_grid(tgt, one, BOX, n_grid, dev, torch)
    diag["final_loss"] = mass_grid_loss(Xs, rho_t, BOX, n_grid, dev, torch)
    diag["volume_ratio"] = diag["volume"][-1] / diag["volume"][0]
    diag["cfl_dt"], diag["dx"], diag["wave_speed"] = cfl_dt(BOX, n_grid, youngs)
    diag["dt_sub"] = dt_sub
    diag["dt_sub_over_cfl"] = dt_sub / diag["cfl_dt"]
    diag["substeps_per_frame"] = max(1, round(dt / dt_sub))
    diag["frames_pos"] = np.stack(frames_pos) if frames_pos else None
    diag["n_pts"], diag["n_grid"] = n_pts, n_grid
    return diag


# ============================================================================================
# results.json -- one fragment PER EXPERIMENT (results_<tag>.json), so parallel cluster jobs
# never race on a single shared file. The orchestrator merges fragments into results.json.
# ============================================================================================
def _sanitize(o):
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def write_fragment(out_dir, tag, payload):
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(payload)
    payload["tag"] = tag
    payload["updated"] = datetime.datetime.utcnow().isoformat() + "Z"
    f = os.path.join(out_dir, f"results_{tag}.json")
    tmp = f + ".tmp"
    json.dump(_sanitize(payload), open(tmp, "w"), indent=2)
    os.replace(tmp, f)
    print(f"  -> {f}", flush=True)


# ============================================================================================
# experiments
# ============================================================================================
def exp_tear(args, dev, rng, out_dir):
    """Full training + instrumented final rollout on the cow: the main measurement for what
    tears and whether volume is conserved."""
    stages = ([tuple(int(v) for v in st.split(":")) for st in args.stages.split(",")]
              if args.stages else [(args.n_pts, args.n_grid, args.iters)])
    log_lines = []
    log = lambda *a, **k: (print(*a, **k), log_lines.append(" ".join(str(x) for x in a)))
    trained = train_control(args.target, stages, args.ctrl, args.lr, args.frames, dev, rng,
                             youngs=args.youngs, drag=args.drag, a_max=args.a_max,
                             dt_sub=args.dt_sub, expm_method=args.expm, log=log)
    diag = diagnose_rollout(trained, dev, render_frames=args.render_frames, log=log)

    score = control_disagreement(trained["theta"], trained["idx_np"], trained["wts_np"])
    lattice = control_grid_discontinuity(trained["theta"], trained["K"])
    Jf = diag["J_final"]; ppc_f = diag["ppc_final"]
    r_disagree = float(np.corrcoef(score, Jf)[0, 1]) if Jf is not None else None
    r_density = float(np.corrcoef(1.0 / np.maximum(ppc_f, 1), Jf)[0, 1]) if Jf is not None else None

    # a_max: PROVEN irrelevant if mpm_particle has no ancestor (a_ext is then identically zero
    # for ANY a_max) -- checked at runtime, not assumed.
    a_max_inert = (diag["ancestors_mpm_particle"] == [])
    a_max_confirm = None
    if a_max_inert:
        diag_lo = diagnose_rollout(trained, dev, render_frames=args.render_frames, a_max=1.0, log=log)
        a_max_confirm = dict(a_max_1=diag_lo["max_absJm1"][-1], a_max_default=diag["max_absJm1"][-1],
                              identical=bool(np.isclose(diag_lo["max_absJm1"][-1], diag["max_absJm1"][-1])))

    # expm accuracy at the rates the optimiser actually found
    A = sym_from_a6(trained["theta"])
    import torch
    At = torch.as_tensor(A, dtype=torch.float32, device=dev)
    dtf = 0.002
    Adt = -At * dtf
    G2 = torch.eye(3, device=dev) + Adt + 0.5 * (Adt @ Adt)
    Gx = torch.matrix_exp(Adt)
    rel_err = float((torch.linalg.norm(G2 - Gx, dim=(-1, -2))
                      / torch.linalg.norm(Gx, dim=(-1, -2)).clamp_min(1e-9)).max())
    diag_exact = diagnose_rollout(trained, dev, render_frames=args.render_frames, expm_method="exact", log=log)

    d = os.path.join(out_dir, f"morph_{args.target}_{args.tag}")
    os.makedirs(d, exist_ok=True)
    if diag["frames_pos"] is not None:
        try:
            from morph_gallery import render
            render(d, diag["frames_pos"], trained["tgt_np"], BOX, f"{args.target} ({args.tag})")
        except Exception as e:
            log(f"  render failed (non-fatal): {e}")
        np.savez_compressed(os.path.join(d, "morph_diag.npz"),
                             frames=diag["frames_pos"], target=trained["tgt_np"], box=BOX,
                             control=trained["theta"], J_final=Jf, ppc_final=ppc_f,
                             control_disagreement=score)

    tearing_cause = ("control-field discontinuity at control-cell boundaries"
                      if (r_disagree is not None and r_disagree > 0.3
                          and (r_density is None or abs(r_disagree) > abs(r_density)))
                      else "grid under-resolution (low particles/cell)"
                      if (r_density is not None and r_density > 0.3) else "inconclusive (see r_*)")

    payload = dict(
        experiment="tear", target=args.target, command=" ".join(sys.argv),
        train=dict(n_pts=trained["n_pts"], n_grid=trained["n_grid"], K=trained["K"],
                   frames=trained["frames"], seconds=trained["seconds"],
                   final_loss_mass=trained["history"][-1]["loss_mass"]),
        diagnosis=dict(
            tearing_cause=tearing_cause,
            r_disagreement_vs_maxJm1=r_disagree,
            r_inverse_density_vs_maxJm1=r_density,
            control_disagreement_score=dict(max=float(score.max()), mean=float(score.mean()),
                                             p95=float(np.percentile(score, 95))),
            control_lattice_discontinuity=lattice,
            a_max_ancestors_of_mpm_particle=diag["ancestors_mpm_particle"],
            a_max_provably_inert=a_max_inert,
            a_max_confirmation_run=a_max_confirm,
            evidence="max|J-1| at the final frame correlated (Pearson r) against two independent "
                     "per-particle scores: control_disagreement (candidate: boundary between "
                     "opposing control nodes) and inverse local particle density (candidate: "
                     "under-resolved grid support). The larger |r| identifies the cause; a_max is "
                     "shown structurally inert first (empty ancestor list -> a_ext==0 for any "
                     "a_max), then confirmed by a a_max=1 vs a_max=200 rerun.",
        ),
        volume=dict(
            volume_ratio_end_over_start=diag["volume_ratio"],
            volume_series=diag["volume"],
            conserved=bool(abs(diag["volume_ratio"] - 1.0) < 0.02),
        ),
        cfl=dict(cfl_dt=diag["cfl_dt"], dx=diag["dx"], wave_speed=diag["wave_speed"],
                 dt_sub=diag["dt_sub"], dt_sub_over_cfl=diag["dt_sub_over_cfl"],
                 substeps_per_frame=diag["substeps_per_frame"]),
        particles_per_cell=dict(min_over_run=min(diag["min_ppc"]), p05_over_run=min(diag["p05_ppc"]),
                                 mean_final=diag["mean_ppc"][-1], frac_lt8_final=diag["frac_ppc_lt8"][-1]),
        expm=dict(max_rel_err_series2_vs_exact=rel_err,
                  volume_ratio_exact=diag_exact["volume_ratio"],
                  max_absJm1_exact=diag_exact["max_absJm1"][-1],
                  final_loss_exact=diag_exact["final_loss"],
                  volume_ratio_series2=diag["volume_ratio"],
                  max_absJm1_series2=diag["max_absJm1"][-1],
                  final_loss_series2=diag["final_loss"]),
        max_absJm1_over_run=diag["max_absJm1"],
        log_tail=log_lines[-40:],
    )
    write_fragment(out_dir, args.tag, payload)
    return payload


def exp_volume_fix(args, dev, rng, out_dir):
    stages = ([tuple(int(v) for v in st.split(":")) for st in args.stages.split(",")]
              if args.stages else [(args.n_pts, args.n_grid, args.iters)])
    results = {}
    for variant in args.variant.split(","):
        variant = variant.strip()
        trace_free = variant in ("trace_free", "both")
        vol_weight = args.vol_weight if variant in ("vol_penalty", "both") else 0.0
        log_lines = []
        log = lambda *a, **k: (print(*a, **k), log_lines.append(" ".join(str(x) for x in a)))
        # RESEEDED PER VARIANT, not the outer `rng` consumed across the loop: trace_free,
        # vol_penalty and baseline must see the IDENTICAL initial ball and target sample, or a
        # difference in volume/loss could be sampling noise rather than the control law.
        rng_v = np.random.default_rng(0)
        trained = train_control(args.target, stages, args.ctrl, args.lr, args.frames, dev, rng_v,
                                 youngs=args.youngs, drag=args.drag, a_max=args.a_max,
                                 dt_sub=args.dt_sub, trace_free=trace_free, vol_weight=vol_weight,
                                 log=log)
        diag = diagnose_rollout(trained, dev, render_frames=args.render_frames, log=log)
        results[variant] = dict(
            final_loss_mass=trained["history"][-1]["loss_mass"],
            volume_ratio=diag["volume_ratio"], volume_series=diag["volume"],
            max_absJm1=diag["max_absJm1"][-1], seconds=trained["seconds"],
        )
        d = os.path.join(out_dir, f"morph_{args.target}_{args.tag}_{variant}")
        os.makedirs(d, exist_ok=True)
        if diag["frames_pos"] is not None:
            try:
                from morph_gallery import render
                render(d, diag["frames_pos"], trained["tgt_np"], BOX, f"{args.target} ({variant})")
            except Exception as e:
                log(f"  render failed (non-fatal): {e}")
        payload = dict(experiment="volume_fix", variant=variant, command=" ".join(sys.argv),
                       result=results[variant], log_tail=log_lines[-40:])
        write_fragment(out_dir, f"{args.tag}_{variant}", payload)
    return results


def exp_ablate_drag(args, dev, rng, out_dir):
    stages = [(args.n_pts, args.n_grid, args.iters)]
    log_lines = []
    log = lambda *a, **k: (print(*a, **k), log_lines.append(" ".join(str(x) for x in a)))
    trained = train_control(args.target, stages, args.ctrl, args.lr, args.frames, dev, rng,
                             youngs=args.youngs, drag=args.drag, a_max=args.a_max,
                             dt_sub=args.dt_sub, log=log)
    sweep = {}
    for d in [float(x) for x in args.sweep.split(",")]:
        diag = diagnose_rollout(trained, dev, render_frames=args.render_frames, drag=d, log=log)
        sweep[str(d)] = dict(max_absJm1=diag["max_absJm1"][-1], volume_ratio=diag["volume_ratio"],
                              min_ppc=min(diag["min_ppc"]), final_loss=diag["final_loss"])
        log(f"  drag {d}: max|J-1| {diag['max_absJm1'][-1]:.4f}  vol_ratio {diag['volume_ratio']:.4f}  "
            f"loss {diag['final_loss']:.6f}")
    payload = dict(experiment="ablate_drag", command=" ".join(sys.argv), train_drag=args.drag,
                   sweep=sweep, log_tail=log_lines[-60:])
    write_fragment(out_dir, args.tag, payload)
    return payload


def exp_ablate_amax(args, dev, rng, out_dir):
    stages = [(args.n_pts, args.n_grid, args.iters)]
    log_lines = []
    log = lambda *a, **k: (print(*a, **k), log_lines.append(" ".join(str(x) for x in a)))
    trained = train_control(args.target, stages, args.ctrl, args.lr, args.frames, dev, rng,
                             youngs=args.youngs, drag=args.drag, a_max=args.a_max,
                             dt_sub=args.dt_sub, log=log)
    ancestors = None
    sweep = {}
    for a in [float(x) for x in args.sweep.split(",")]:
        diag = diagnose_rollout(trained, dev, render_frames=args.render_frames, a_max=a, log=log)
        ancestors = diag["ancestors_mpm_particle"]
        sweep[str(a)] = dict(max_absJm1=diag["max_absJm1"][-1], volume_ratio=diag["volume_ratio"],
                              final_loss=diag["final_loss"])
        log(f"  a_max {a}: max|J-1| {diag['max_absJm1'][-1]:.6f}  loss {diag['final_loss']:.6f}")
    identical = len({round(v["max_absJm1"], 6) for v in sweep.values()}) == 1
    payload = dict(experiment="ablate_amax", command=" ".join(sys.argv),
                   ancestors_of_mpm_particle=ancestors, all_identical_across_sweep=identical,
                   sweep=sweep, log_tail=log_lines[-60:])
    write_fragment(out_dir, args.tag, payload)
    return payload


def exp_ablate_substep(args, dev, rng, out_dir):
    cfl, dx, c = cfl_dt(BOX, args.n_grid, args.youngs)
    log_lines = []
    log = lambda *a, **k: (print(*a, **k), log_lines.append(" ".join(str(x) for x in a)))
    log(f"  CFL: dx={dx:.5f} c={c:.3f} dt_CFL={cfl:.6f}  (default dt_sub={args.dt_sub:.6f}, "
        f"{args.dt_sub/cfl:.2f} of CFL)")
    stages = [(args.n_pts, args.n_grid, args.iters)]
    trained = train_control(args.target, stages, args.ctrl, args.lr, args.frames, dev, rng,
                             youngs=args.youngs, drag=args.drag, a_max=args.a_max,
                             dt_sub=args.dt_sub, log=log)
    sweep = {}
    for frac in [float(x) for x in args.sweep.split(",")]:
        sub = frac * cfl
        diag = diagnose_rollout(trained, dev, render_frames=args.render_frames, dt_sub=sub, log=log)
        sweep[str(frac)] = dict(dt_sub=sub, substeps_per_frame=diag["substeps_per_frame"],
                                 max_absJm1=diag["max_absJm1"][-1], volume_ratio=diag["volume_ratio"],
                                 final_loss=diag["final_loss"],
                                 finite=bool(np.isfinite(diag["max_absJm1"][-1])))
        log(f"  dt_sub={sub:.6f} ({frac:.2f} CFL, {diag['substeps_per_frame']} substeps/frame): "
            f"max|J-1| {diag['max_absJm1'][-1]:.4f}  loss {diag['final_loss']:.6f}")
    # ONE retrain at a coarser-but-still-inside-CFL substep, to see whether the OPTIMISER's answer
    # (not just the frozen-control rollout) changes.
    retrain = None
    if args.retrain_frac:
        sub2 = float(args.retrain_frac) * cfl
        log(f"  retraining at dt_sub={sub2:.6f} ({args.retrain_frac} CFL) to see if the optimiser's "
            f"own answer changes, not just a frozen control replayed at a different substep")
        rng2 = np.random.default_rng(0)           # same ball/target sample as the baseline training
        trained2 = train_control(args.target, stages, args.ctrl, args.lr, args.frames, dev, rng2,
                                  youngs=args.youngs, drag=args.drag, a_max=args.a_max,
                                  dt_sub=sub2, log=log)
        diag2 = diagnose_rollout(trained2, dev, render_frames=args.render_frames, log=log)
        retrain = dict(dt_sub=sub2, final_loss_mass=trained2["history"][-1]["loss_mass"],
                       max_absJm1=diag2["max_absJm1"][-1], volume_ratio=diag2["volume_ratio"],
                       seconds=trained2["seconds"])
    payload = dict(experiment="ablate_substep", command=" ".join(sys.argv), cfl_dt=cfl, dx=dx,
                   wave_speed=c, train_dt_sub=args.dt_sub, sweep_frozen_control=sweep,
                   retrain_at_different_substep=retrain, log_tail=log_lines[-80:])
    write_fragment(out_dir, args.tag, payload)
    return payload


def exp_expm_check(args, dev, rng, out_dir):
    stages = [(args.n_pts, args.n_grid, args.iters)]
    log_lines = []
    log = lambda *a, **k: (print(*a, **k), log_lines.append(" ".join(str(x) for x in a)))
    trained = train_control(args.target, stages, args.ctrl, args.lr, args.frames, dev, rng,
                             youngs=args.youngs, drag=args.drag, a_max=args.a_max,
                             dt_sub=args.dt_sub, log=log)
    import torch
    A = sym_from_a6(trained["theta"])
    At = torch.as_tensor(A, dtype=torch.float32, device=dev)
    dtf = 0.002
    Adt = -At * dtf
    G2 = torch.eye(3, device=dev) + Adt + 0.5 * (Adt @ Adt)
    Gx = torch.matrix_exp(Adt)
    rel_err = (torch.linalg.norm(G2 - Gx, dim=(-1, -2))
               / torch.linalg.norm(Gx, dim=(-1, -2)).clamp_min(1e-9))
    a6 = trained["theta"]
    rate_norm = np.linalg.norm(a6, axis=1)

    diag2 = diagnose_rollout(trained, dev, render_frames=args.render_frames, expm_method="series2", log=log)
    diagx = diagnose_rollout(trained, dev, render_frames=args.render_frames, expm_method="exact", log=log)
    payload = dict(
        experiment="expm_check", command=" ".join(sys.argv),
        rel_err=dict(max=float(rel_err.max()), mean=float(rel_err.mean()), p95=float(rel_err.quantile(0.95))),
        rate_norm=dict(max=float(rate_norm.max()), mean=float(rate_norm.mean())),
        series2=dict(volume_ratio=diag2["volume_ratio"], max_absJm1=diag2["max_absJm1"][-1],
                     final_loss=diag2["final_loss"]),
        exact=dict(volume_ratio=diagx["volume_ratio"], max_absJm1=diagx["max_absJm1"][-1],
                   final_loss=diagx["final_loss"]),
        log_tail=log_lines[-60:],
    )
    write_fragment(out_dir, args.tag, payload)
    return payload


def exp_material_sweep(args, dev, rng, out_dir):
    combos = [tuple(float(v) for v in c.split(":")) for c in args.combos.split(",")]  # youngs:nu
    sweep = {}
    for youngs, nu in combos:
        log_lines = []
        log = lambda *a, **k: (print(*a, **k), log_lines.append(" ".join(str(x) for x in a)))
        stages = [(args.n_pts, args.n_grid, args.iters)]
        cfl, dx, c = cfl_dt(BOX, args.n_grid, youngs)
        log(f"  youngs={youngs} nu={nu}: CFL dt={cfl:.6f}, dt_sub={args.dt_sub:.6f} "
            f"({args.dt_sub/cfl:.2f} of CFL)")
        rng_v = np.random.default_rng(0)          # same ball/target sample at every combo
        trained = train_control(args.target, stages, args.ctrl, args.lr, args.frames, dev, rng_v,
                                 youngs=youngs, drag=args.drag, a_max=args.a_max,
                                 dt_sub=args.dt_sub, nu=nu, log=log)
        diag = diagnose_rollout(trained, dev, render_frames=args.render_frames, log=log)
        key = f"E{youngs:g}_nu{nu:g}"
        sweep[key] = dict(youngs=youngs, nu=nu, cfl_dt=cfl, dt_sub_over_cfl=args.dt_sub / cfl,
                           final_loss_mass=trained["history"][-1]["loss_mass"],
                           max_absJm1=diag["max_absJm1"][-1], volume_ratio=diag["volume_ratio"],
                           min_ppc=min(diag["min_ppc"]), seconds=trained["seconds"])
        payload = dict(experiment="material_sweep", combo=key, command=" ".join(sys.argv),
                       result=sweep[key], log_tail=log_lines[-40:])
        write_fragment(out_dir, f"{args.tag}_{key}", payload)
    return sweep


def exp_ppc_sweep(args, dev, rng, out_dir):
    sweep = {}
    for n_pts in [int(x) for x in args.sweep.split(",")]:
        log_lines = []
        log = lambda *a, **k: (print(*a, **k), log_lines.append(" ".join(str(x) for x in a)))
        stages = [(n_pts, args.n_grid, args.iters)]
        rng_v = np.random.default_rng(0)
        trained = train_control(args.target, stages, args.ctrl, args.lr, args.frames, dev, rng_v,
                                 youngs=args.youngs, drag=args.drag, a_max=args.a_max,
                                 dt_sub=args.dt_sub, log=log)
        diag = diagnose_rollout(trained, dev, render_frames=args.render_frames, log=log)
        avg_ppc_init = n_pts / ((4.0 / 3.0 * np.pi * R ** 3) / (BOX / args.n_grid) ** 3)
        sweep[str(n_pts)] = dict(n_pts=n_pts, avg_ppc_at_seed=float(avg_ppc_init),
                                  min_ppc_over_run=min(diag["min_ppc"]),
                                  p05_ppc_over_run=min(diag["p05_ppc"]),
                                  frac_ppc_lt8_final=diag["frac_ppc_lt8"][-1],
                                  max_absJm1=diag["max_absJm1"][-1], volume_ratio=diag["volume_ratio"],
                                  final_loss_mass=trained["history"][-1]["loss_mass"],
                                  seconds=trained["seconds"])
        payload = dict(experiment="ppc_sweep", n_pts=n_pts, command=" ".join(sys.argv),
                       result=sweep[str(n_pts)], log_tail=log_lines[-40:])
        write_fragment(out_dir, f"{args.tag}_{n_pts}", payload)
    return sweep


EXPS = dict(tear=exp_tear, volume_fix=exp_volume_fix, ablate_drag=exp_ablate_drag,
            ablate_amax=exp_ablate_amax, ablate_substep=exp_ablate_substep,
            expm_check=exp_expm_check, material_sweep=exp_material_sweep,
            ppc_sweep=exp_ppc_sweep)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=list(EXPS))
    ap.add_argument("--tag", required=True, help="names the output subfolder and results_<tag>.json")
    ap.add_argument("--target", default="cow")
    ap.add_argument("--n-pts", type=int, default=12500)
    ap.add_argument("--n-grid", type=int, default=32)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--render-frames", type=int, default=100)
    ap.add_argument("--ctrl", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3.0)
    ap.add_argument("--stages", default="", help="pts:grid:iters,... coarse-to-fine (exp=tear/volume_fix)")
    ap.add_argument("--youngs", type=float, default=90.0)
    ap.add_argument("--drag", type=float, default=0.5)
    ap.add_argument("--a-max", type=float, default=200.0)
    ap.add_argument("--dt-sub", type=float, default=3.4e-4)
    ap.add_argument("--expm", default="series2", choices=["series2", "exact"])
    ap.add_argument("--variant", default="trace_free,vol_penalty", help="exp=volume_fix")
    ap.add_argument("--vol-weight", type=float, default=1.0, help="exp=volume_fix")
    ap.add_argument("--sweep", default="", help="comma list, meaning depends on --exp")
    ap.add_argument("--retrain-frac", default="", help="exp=ablate_substep: also retrain at this "
                    "fraction of the CFL bound (blank = skip)")
    ap.add_argument("--combos", default="90:0.2,90:0.45,45:0.2,180:0.2", help="exp=material_sweep, youngs:nu")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    dev = args.device
    import torch
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)
    os.makedirs(args.out, exist_ok=True)

    defaults = dict(ablate_drag="0.0,0.25,0.5,1.0,2.0", ablate_amax="1.0,50.0,200.0,1e4,1e6",
                    ablate_substep="0.1,0.25,0.4,0.75,0.95,1.2", ppc_sweep="3000,12500,50000,100000")
    # 150000 pts at grid 32, compiled + grad, OOMs OUTRIGHT on gpu_l4's 22 GiB (single stage, no
    # cross-stage carryover involved) -- `torch.OutOfMemoryError` allocating a (1350000,3,3) buffer
    # with 22.03 GiB already resident. That is the LARGER agent's territory (500k-2M points); capped
    # here at 100000 so this sweep measures particles/cell, not the memory ceiling.
    if not args.sweep and args.exp in defaults:
        args.sweep = defaults[args.exp]

    print(f"[morph_physics] exp={args.exp} tag={args.tag} device={args.device}", flush=True)
    result = EXPS[args.exp](args, dev, rng, args.out)
    print(f"[morph_physics] done: {args.exp}/{args.tag}", flush=True)
    return result


if __name__ == "__main__":
    main()
