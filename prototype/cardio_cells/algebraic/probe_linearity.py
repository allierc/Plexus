"""probe_linearity.py -- IS THE ONE-STEP MLS-MPM ACCELERATION LINEAR IN THE PER-CELL PARAMETERS?

THE CLAIM UNDER TEST
====================================================================================================
The fixed-corotated stress in `mpm_scatter` is

    sigma = 2 mu (F - R) F^T + lambda J (J - 1) I ,      mu, lambda both LINEAR in E (see `_lame`)

and the active body force is linear in a per-cell gain. Positions are observed, so F, R, J and the
B-spline weights are all KNOWN at a frame. If one solver step is then AFFINE in
theta = (E_1..E_C, gain_1..gain_C), the momentum balance is an algebraic constraint

    A(x) theta = b(x)

and no time integration is needed to fit theta. AFFINE, not linear: the theta-independent part
(m*v momentum, m*C affine, drag) is a constant offset, so every test below subtracts the theta = 0
response first. Reporting `a(P+Q) == a(P) + a(Q)` WITHOUT that subtraction is a different (and
false) claim; both numbers are printed.

WHAT IS MEASURED
----------------------------------------------------------------------------------------------------
A fixed state (pos, vel, F, C, Jp) is snapshotted after a warm-up. Every probe restores that exact
state, overwrites mu/la from a per-cell E and the particle body-force delta from a per-cell gain,
runs N substeps of [mpm_strain, mpm_scatter, mpm_grid_update, mpm_gather], and reads

    a = (v_new - v_old) / dt_sub          [Np, 2]

Everything is a relative L2 deviation against a repeat-run NULL: the same theta run twice. That
null is the floor -- float32 + CUDA `index_add_` atomics are not bit-reproducible -- and a test
"passes" only if its deviation is at that floor, not merely "small".

usage:
  PYTHONPATH=/workspace/Plexus/src python probe_linearity.py [--device cuda:1] [--per-parent 500]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, "/workspace/Plexus/src")

import plexus.operators                                   # noqa: F401  self-register the library
from plexus.engine import build, _resolve_emit, _selector_mask
from plexus.models.entities import _lame
from plexus.models.registry import get_operator
from plexus.schema import load

CONFIG = "/workspace/Plexus/config/material/material_cardio_cells.yaml"
SUBSTEP_TOKENS = ["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"]


# --------------------------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------------------------- #
def rel(d, ref, m=None):
    """||d|| / ||ref||, optionally restricted to a particle mask."""
    if m is not None:
        d, ref = d[m], ref[m]
    if d.numel() == 0:
        return float("nan")                      # an empty mask (e.g. the no-wall control)
    n = ref.norm().item()
    return float(d.norm().item() / max(n, 1e-30))


def relmax(d, ref, m=None):
    """max_i |d_i| / rms_i |ref_i| -- the worst single particle, in units of a typical one."""
    if m is not None:
        d, ref = d[m], ref[m]
    if d.numel() == 0:
        return float("nan")
    scale = ref.norm(dim=1).pow(2).mean().sqrt().item()
    return float(d.norm(dim=1).max().item() / max(scale, 1e-30))


# --------------------------------------------------------------------------------------------- #
#  the probe: build once, snapshot a state, replay one substep for any theta
# --------------------------------------------------------------------------------------------- #
class Probe:
    def __init__(self, device="cuda:1", per_parent=500, n_grid=None, warmup=12, seed=0,
                 dtype="float32", shrink=1.0):
        self.dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
        torch.set_default_dtype(self.dtype)
        raw = yaml.safe_load(open(CONFIG))
        raw["general"]["n_frames"] = max(warmup + 2, 4)
        raw["general"]["name"] = "probe_cardio_cells"
        raw["sets"]["mpm_particle"]["per_parent"] = int(per_parent)
        if n_grid is not None:
            raw["fields"]["mpm_grid"]["n_grid"] = int(n_grid)
            raw["fields"]["activation"]["res"] = int(n_grid)
        # a scratch spec file: `plexus.schema.load` takes a path. Nothing is generated / written
        # into graphs_data; only the label tif + props json are READ from the shared data root.
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="probe_cardio_")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(raw, f)
        self.sim = load(path)
        os.unlink(path)
        self.device = device
        self.dt = float(self.sim.dt)
        self.dt_sub = float([s for s in self.sim.schedule
                             if isinstance(s, dict) and "substep_dt" in s][0]["substep_dt"])
        self.n_sub_per_frame = max(1, round(self.dt / self.dt_sub))

        self.H = build(self.sim, device)
        self.H.emit_order = _resolve_emit(self.sim, self.H)
        self.inst = [(o.op,
                      get_operator(o.op, variant=o.impl)(
                          {**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, device),
                      o.on)
                     for o in self.sim.operators]
        self.p = self.H.level("mpm_particle")
        self.g = self.H.field("mpm_grid")
        self.outer_tokens = [t for t in self.sim.schedule if not isinstance(t, dict)]

        with torch.no_grad():
            self._warmup(warmup)
            # NO-WALL CONTROL: contract the sheet about the domain centre so that no particle's
            # B-spline stencil can reach the 3-cell reflective slabs. Same operators, same state,
            # only the geometry is moved out of contact -- so if the residual vanishes, the wall
            # handling was the cause, and nothing else was changed to make it vanish.
            self.shrink = float(shrink)
            if self.shrink != 1.0:
                p = self.H.level("mpm_particle")
                a, b = p.state_schema["pos"]
                st = p.state.clone()
                st[:, a:b] = 0.5 + (st[:, a:b] - 0.5) * self.shrink
                p.state = st
            self._snapshot()

    # -- engine replay ---------------------------------------------------------------------- #
    def _tok(self, token):
        for nm, ob, sel in self.inst:
            if nm != token:
                continue
            deltas = ob(self.H, _selector_mask(self.H, sel))
            block = getattr(ob, "INTEGRAND", None)
            for lvlname, d in deltas.items():
                self.H.add_delta(lvlname, d, block)

    def _warmup(self, n):
        """Run `n` complete engine ticks so the probe state has real deformation and motion."""
        for tick in range(n):
            self.H.frame = tick
            self.H.zero_delta()
            for tok in self.outer_tokens:
                self._tok(tok)
            self.H.sub_dt = self.dt_sub
            for _ in range(self.n_sub_per_frame):
                for tok in SUBSTEP_TOKENS:
                    self._tok(tok)
            self.H.sub_dt = None
            # mpm_particle is not engine-integrated (all its ops emit mpm_acceleration);
            # the substep owns advection, exactly as `engine.run` does.

    def _snapshot(self):
        """Freeze the probe state: particle buffers, the outer-schedule deltas split into their
        ACTIVE and PASSIVE parts, and the per-cell index map."""
        p, H = self.p, self.H
        tick = self.sim.n_frames  # a fresh tick, activation recomputed for it
        H.frame = tick
        H.zero_delta()
        # run the outer ops ONE AT A TIME, diffing the accumulated delta, so the active-force
        # contribution can be scaled by a per-cell gain independently of drag.
        self.delta_parts = {}
        prev = {k: v.clone() for k, v in H._delta.items()}
        for tok in self.outer_tokens:
            self._tok(tok)
            self.delta_parts[tok] = {k: (v - prev[k]).clone() for k, v in H._delta.items()}
            prev = {k: v.clone() for k, v in H._delta.items()}
        self.delta_full = {k: v.clone() for k, v in H._delta.items()}
        self.act0 = self.delta_parts["active_force"]["mpm_particle"].clone()       # gain-scalable
        self.pass0 = (self.delta_full["mpm_particle"] - self.act0).clone()          # drag etc.
        self.cell_delta = self.delta_full["cell"].clone()                           # a_max clamps THIS

        self.state0 = p.state.clone()
        self.F0, self.C0, self.Jp0 = p.F.clone(), p.C.clone(), p.Jp.clone()
        self.v0 = p.get("vel").clone()
        self.x0 = p.get("pos").clone()
        self.cid = p.cell_id.long()                       # [Np] 1..C, from seed_from_segmentation
        self.n_cells = int(self.cid.max().item())
        self.E0_cell = torch.zeros(self.n_cells + 1, device=self.device)
        self.E0_cell.index_copy_(0, self.cid, p.youngs)   # per-cell E (particles of a cell share it)
        self.materials = {
            "is_liquid": int(p.is_liquid.sum().item()),
            "is_snow": int(p.is_snow.sum().item()),
            "is_visco": int(p.is_visco.sum().item()),
            "Jp_deviation_from_1": float((p.Jp - 1).abs().max().item()),
            "F_res_inv_present": getattr(p, "F_res_inv", None) is not None,
            "H.active_stress_present": getattr(self.H, "active_stress", None) is not None,
            "H.part_accel_present": getattr(self.H, "part_accel", None) is not None,
        }
        self.scatter = [ob for nm, ob, _ in self.inst if nm == "mpm_scatter"][0]
        self.gather = [ob for nm, ob, _ in self.inst if nm == "mpm_gather"][0]
        self.gridup = [ob for nm, ob, _ in self.inst if nm == "mpm_grid_update"][0]

    def restore(self):
        p = self.p
        p.state = self.state0.clone()
        p.F, p.C, p.Jp = self.F0.clone(), self.C0.clone(), self.Jp0.clone()

    # -- one probe -------------------------------------------------------------------------- #
    def step(self, E_cell=None, gain_cell=None, n_sub=1, diag=False):
        """Restore the frozen state, set theta, run `n_sub` substeps. Returns acceleration [Np,2].

        E_cell    [C+1] per-cell Young's modulus (index 0 unused). None = the seeded values.
        gain_cell [C+1] per-cell active gain.    None = 1 everywhere (the spec's amplitude).
        """
        H, p = self.H, self.p
        self.restore()
        if E_cell is None:
            E_cell = self.E0_cell
        Ep = E_cell[self.cid]
        mu, la = _lame(Ep)
        p.mu, p.la = mu, la
        H.zero_delta()
        for k, v in self.delta_full.items():
            H._delta[k] = v.clone()
        if gain_cell is None:
            H._delta["mpm_particle"] = (self.pass0 + self.act0).clone()
        else:
            H._delta["mpm_particle"] = self.pass0 + gain_cell[self.cid][:, None] * self.act0
        H.sub_dt = self.dt_sub
        d = {}
        for i in range(n_sub):
            self._tok("mpm_strain")
            self._tok("mpm_scatter")
            if diag and i == 0:
                d.update(self._grid_diag())
            self._tok("mpm_grid_update")
            if diag and i == 0:
                d.update(self._gather_diag())
            self._tok("mpm_gather")
        H.sub_dt = None
        a = (p.get("vel") - self.v0) / (self.dt_sub * n_sub)
        return (a, d) if diag else a

    # -- clamp diagnostics (read-only; the operators are never modified) ---------------------- #
    def _grid_diag(self):
        """After mpm_scatter, before mpm_grid_update: how many MASSIVE grid nodes would the
        reflective-wall clamp in mpm_grid_update actually CHANGE? (`clamp` is the one
        piecewise-linear op in the grid solve; wall_damp is a fixed multiply and is linear.)"""
        g = self.g
        gv = (g.mv / g.m.clamp(min=1e-10)[:, None]).view(g.nx, g.ny, 2)
        occ = (g.m > 1e-12).view(g.nx, g.ny)
        bnd = 3
        # (i) the NORMAL-component clamp: gv_n.clamp(min=0) / .clamp(max=0) on the 4 slabs
        hit = torch.zeros_like(occ)
        hit[:bnd, :] |= (gv[:bnd, :, 0] < 0)
        hit[g.nx - bnd + 1:, :] |= (gv[g.nx - bnd + 1:, :, 0] > 0)
        hit[:, :bnd] |= (gv[:, :bnd, 1] < 0)
        hit[:, g.ny - bnd + 1:] |= (gv[:, g.ny - bnd + 1:, 1] > 0)
        hit &= occ
        # (ii) the SIGN-CONDITIONAL tangential damp on the x-slabs only:
        #      torch.where(gv_y > 0, gv_y*wd, gv_y). A second, separate nonlinearity.
        #      (the y-slabs damp gv_x unconditionally -- a fixed multiply, and so LINEAR.)
        damp = torch.zeros_like(occ)
        damp[:bnd, :] |= (gv[:bnd, :, 1] > 0)
        damp[g.nx - bnd + 1:, :] |= (gv[g.nx - bnd + 1:, :, 1] > 0)
        damp &= occ
        return {"grid_nodes_massive": int(occ.sum().item()),
                "grid_nodes_wall_normal_clamped": int(hit.sum().item()),
                "grid_nodes_sign_conditional_damped": int(damp.sum().item()),
                "grid_nodes_nonlinear_total": int((hit | damp).sum().item())}

    def _gather_diag(self):
        """After mpm_grid_update, before mpm_gather: does the CFL speed cap or the position
        clamp bite? Both are hard nonlinearities in mpm_gather."""
        g, p = self.g, self.p
        from plexus.operators.mpm_grid import stencil_offsets, bspline
        X = p.get("pos")
        offsets = stencil_offsets(2, X.device)
        fx, weight, flat = bspline(X, g.inv_dx, offsets, g.shape, False)
        gvn = g.v[flat].view(p.n, offsets.shape[0], 2)
        new_V = (weight[..., None] * gvn).sum(1)
        cb = self.gather.wall_contact
        near = (X[:, 0] < cb) | (X[:, 0] > 1.0 - cb) | (X[:, 1] < cb) | (X[:, 1] > 1.0 - cb)
        new_V = torch.where(near[:, None], new_V * self.gather.wall_damp, new_V)
        vmax = min(self.gather.vmax, 0.4 * g.dx / self.dt_sub)
        Xn = X + self.dt_sub * new_V
        lo, hi = 2 * g.dx, 1.0 - 2 * g.dx
        return {"particles": int(p.n),
                "cfl_cap_hits": int((new_V.norm(dim=1) > vmax).sum().item()),
                "cfl_vmax": float(vmax),
                "max_speed": float(new_V.norm(dim=1).max().item()),
                "pos_clamp_hits": int(((Xn < lo) | (Xn > hi)).any(1).sum().item()),
                "wall_contact_particles": int(near.sum().item())}

    # -- masks ------------------------------------------------------------------------------- #
    def interior_mask(self, margin_cells=8.0):
        """Particles whose 3x3 B-spline stencil cannot touch the clamped wall slabs.
        (The clamp lives in grid rows/cols < 3 and > n-3; a particle reads base..base+2 with
        base = floor(X/dx - 0.5), so margin >= 6 dx is already causally isolated in ONE substep.)"""
        m = margin_cells * self.g.dx
        X = self.x0
        return (X[:, 0] > m) & (X[:, 0] < 1.0 - m) & (X[:, 1] > m) & (X[:, 1] < 1.0 - m)

    def split_cells(self, k=2, seed=0):
        """k disjoint cell groups (a fixed random partition of the cell ids)."""
        gcpu = torch.Generator().manual_seed(seed)
        assign = torch.randint(0, k, (self.n_cells + 1,), generator=gcpu).to(self.device)
        return [(assign == i) for i in range(k)]


def zero_E(pr):
    return torch.zeros(pr.n_cells + 1, device=pr.device)


def zero_gain(pr):
    return torch.zeros(pr.n_cells + 1, device=pr.device)


def one_gain(pr):
    return torch.ones(pr.n_cells + 1, device=pr.device)


# --------------------------------------------------------------------------------------------- #
#  the tests
# --------------------------------------------------------------------------------------------- #
def run_all(pr, tol_factor=10.0, log=print):
    R = {}
    dev = pr.device
    E0, C = pr.E0_cell, pr.n_cells

    # ---- 0. NULL CONTROLS ----------------------------------------------------------------- #
    # (a) REPEAT: the identical computation twice. Catches nondeterministic CUDA index_add_.
    a_ref, diag = pr.step(diag=True)
    a_ref2 = pr.step()
    floor_repeat = rel(a_ref2 - a_ref, a_ref)
    floor_repeat = max(floor_repeat, rel(pr.step() - a_ref, a_ref))

    # (b) SENSITIVITY: perturb theta by one unit in the last place and see how far the answer
    #     moves. THIS is the real floor for a theta-dependent comparison -- a superposition
    #     residual at or below it is indistinguishable from round-off in the parameters, however
    #     "large" it looks in absolute terms. (a) is bit-reproducibility, (b) is conditioning.
    a_E0 = pr.step(E_cell=zero_E(pr))                       # E = 0 everywhere (gain at 1)
    a_g0 = pr.step(gain_cell=zero_gain(pr))                 # gain = 0 (E at seeded values)
    eps = float(torch.finfo(pr.dtype).eps)
    gcpu = torch.Generator().manual_seed(3)
    u = (torch.randint(0, 2, (C + 1,), generator=gcpu).to(pr.device) * 2 - 1).to(pr.dtype)
    a_pert = pr.step(E_cell=E0 * (1 + eps * u))
    floor_theta = rel(a_pert - a_ref, a_ref - a_E0)
    tol = max(tol_factor * floor_theta, floor_repeat * tol_factor, 4 * eps)
    R["null_repeat_deviation"] = floor_repeat
    R["null_theta_ulp_sensitivity"] = floor_theta
    R["dtype_eps"] = eps
    R["tolerance"] = tol
    R["diagnostics"] = diag
    R["materials"] = pr.materials
    R["n_cells"] = C
    R["n_particles"] = int(pr.p.n)
    R["dt_sub"] = pr.dt_sub
    R["substeps_per_frame"] = pr.n_sub_per_frame
    log(f"[0] NULL (a) repeat-run deviation      = {floor_repeat:.3e}")
    log(f"[0] NULL (b) 1-ulp theta sensitivity   = {floor_theta:.3e}   (eps={eps:.2e})")
    log(f"[0]      -> PASS tolerance             = {tol:.3e}  ({tol_factor:g}x the larger floor)")
    log(f"    clamp diagnostics: {diag}")
    log(f"    material masks:    {pr.materials}")

    # ---- 0c. the b-terms: drag and the parent (gravity) channel ---------------------------- #
    #  the task's third worry: "the drag operator is linear in velocity, so it moves to b, but
    #  CONFIRM". Measured, not assumed: drag's returned delta must equal -k*v exactly at the
    #  frozen state, and it must carry no theta dependence (it reads `vel` only).
    kdrag = [o.params["k"] for o in pr.sim.operators if o.op == "drag"][0]
    dd = pr.delta_parts["drag"]["mpm_particle"]
    R["T0c_drag"] = {
        "k": float(kdrag),
        "rel_err_vs_minus_k_v": rel(dd + float(kdrag) * pr.v0, dd),
        "drag_frac_of_body_force": float(dd.norm().item()
                                         / max((pr.pass0 + pr.act0).norm().item(), 1e-30)),
        "mpm_scatter_own_drag_param": float(pr.scatter.drag),
        "note": "drag reads only the frozen `vel`; it is recomputed by the OUTER schedule, not "
                "inside the substep loop, so within one substep it is a constant -> belongs to b",
    }
    log(f"[0c] drag: ||delta + k*v||/||delta|| = {R['T0c_drag']['rel_err_vs_minus_k_v']:.3e} "
        f"(k={kdrag}); drag is {R['T0c_drag']['drag_frac_of_body_force']:.1%} of the body force; "
        f"mpm_scatter's own drag param = {pr.scatter.drag}")
    R["offset_frac_E"] = rel(a_E0, a_ref)
    R["offset_frac_gain"] = rel(a_g0, a_ref)
    log(f"    affine offset ||a(E=0)||/||a|| = {R['offset_frac_E']:.4f}   "
        f"||a(gain=0)||/||a|| = {R['offset_frac_gain']:.4f}")

    def homog(name, mk, base):
        """response(s) == s * response(1)?  response(s) = a(s*theta0) - a(0)."""
        out = {}
        r1 = pr.step(**mk(1.0)) - base
        for s in (0.5, 1.0, 2.0, 4.0):
            rs = pr.step(**mk(s)) - base
            out[str(s)] = {"rel": rel(rs - s * r1, s * r1), "relmax": relmax(rs - s * r1, s * r1)}
        worst = max(v["rel"] for v in out.values())
        out["worst"] = worst
        out["PASS"] = bool(worst <= tol)
        log(f"[{name}] homogeneity worst rel dev = {worst:.3e}  "
            f"{'PASS' if out['PASS'] else 'FAIL'}   " +
            "  ".join(f"s={k}:{v['rel']:.2e}" for k, v in out.items() if k not in ("worst", "PASS")))
        return out

    # ---- 1. HOMOGENEITY in E -------------------------------------------------------------- #
    R["T1_homogeneity_E"] = homog("1", lambda s: {"E_cell": s * E0}, a_E0)

    # ---- 3a. HOMOGENEITY in gain ---------------------------------------------------------- #
    R["T3a_homogeneity_gain"] = homog("3a", lambda s: {"gain_cell": s * one_gain(pr)}, a_g0)

    wallm = ~pr.interior_mask(8.0)                      # particles whose stencil can reach a slab

    def superpose(name, groups, mk, base, mask=None, extra=""):
        """a(P+Q) - a(0) == [a(P)-a(0)] + [a(Q)-a(0)]?  (and the naive, offset-free version)"""
        parts = [pr.step(**mk(gmask)) for gmask in groups]
        allm = groups[0].clone()
        for gm in groups[1:]:
            allm = allm | gm
        a_all = pr.step(**mk(allm))
        lin = base + sum(p - base for p in parts)
        resid = a_all - lin
        naive = a_all - sum(parts)
        rn = resid.norm().item()
        out = {"rel": rel(resid, a_all - base, mask),
               "rel_vs_total": rel(resid, a_all, mask),
               "relmax": relmax(resid, a_all - base, mask),
               "naive_no_offset_rel": rel(naive, a_all),
               # WHERE does the residual live? 1.0 = entirely on particles that can touch a wall slab
               "resid_frac_on_wall_particles": float(resid[wallm].norm().item() / max(rn, 1e-300)),
               "wall_particle_frac": float(wallm.float().mean().item())}
        out["PASS"] = bool(out["rel"] <= tol) if out["rel"] == out["rel"] else None
        log(f"[{name}] superposition{extra}: rel dev (offset-corrected) = {out['rel']:.3e}  "
            f"{'PASS' if out['PASS'] else 'FAIL'}   "
            f"[vs total a: {out['rel_vs_total']:.3e}; naive a(P+Q)-a(P)-a(Q): "
            f"{out['naive_no_offset_rel']:.3e}; resid on wall particles: "
            f"{out['resid_frac_on_wall_particles']:.4f} of ||resid||, "
            f"they are {out['wall_particle_frac']:.1%} of particles]")
        return out

    P, Q = pr.split_cells(2)
    mkE = lambda m: {"E_cell": torch.where(m, E0, torch.zeros_like(E0))}
    mkG = lambda m: {"gain_cell": m.float()}

    # ---- 2. SUPERPOSITION in E (THE DECISIVE TEST) ---------------------------------------- #
    R["T2_superposition_E"] = superpose("2", [P, Q], mkE, a_E0)
    R["T2b_superposition_E_4groups"] = superpose("2b", pr.split_cells(4, seed=1), mkE, a_E0,
                                                 extra=" (4 disjoint groups)")

    # ---- 3b. SUPERPOSITION in gain -------------------------------------------------------- #
    R["T3b_superposition_gain"] = superpose("3b", [P, Q], mkG, a_g0, extra=" [gain]")

    # ---- 4. does the a_max clamp bite? ---------------------------------------------------- #
    cd = pr.cell_delta
    amax = pr.scatter.a_max
    R["T4_a_max"] = {
        "a_max": amax,
        "cell_delta_absmax": float(cd.abs().max().item()),
        "frac_cell_delta_at_clamp": float((cd.abs() >= amax).float().mean().item()),
        "cell_delta_is_all_zero": bool(cd.abs().max().item() == 0.0),
        "note": "a_max clamps H.delta(parent='cell') ONLY -- theta-independent in this spec",
    }
    log(f"[4] a_max={amax}: |cell delta|max = {R['T4_a_max']['cell_delta_absmax']:.4g}, "
        f"fraction at clamp = {R['T4_a_max']['frac_cell_delta_at_clamp']:.3%}")
    old = pr.scatter.a_max
    pr.scatter.a_max = 1e12
    R["T4_superposition_E_amax_high"] = superpose("4", [P, Q], mkE, pr.step(E_cell=zero_E(pr)),
                                                  extra=" with a_max=1e12")
    pr.scatter.a_max = old

    # ---- 5. interior particles only (away from the wall) ---------------------------------- #
    #  If the wall clamp were the source of the residual, the residual would live AT the wall:
    #  the interior number would drop and the near-wall number would rise. Both are reported.
    R["T5_interior"] = {}
    for margin in (4.0, 8.0, 16.0):
        m = pr.interior_mask(margin)
        R["T5_interior"][f"margin_{margin:g}dx"] = {
            "n_particles": int(m.sum().item()),
            **superpose("5", [P, Q], mkE, a_E0, mask=m,
                        extra=f" INTERIOR only (margin {margin:g}dx, "
                              f"{int(m.sum().item())}/{pr.p.n} particles)")}
    mw = ~pr.interior_mask(8.0)
    R["T5_nearwall"] = {"n_particles": int(mw.sum().item()),
                        **superpose("5", [P, Q], mkE, a_E0, mask=mw,
                                    extra=f" NEAR-WALL only ({int(mw.sum().item())}/{pr.p.n})")}
    # WHICH of the two boundary nonlinearities is it? mpm_grid_update's `if wd != 1.0` block holds
    # the SIGN-CONDITIONAL tangential damp `where(gv_y > 0, gv_y*wd, gv_y)` on the x-slabs; the
    # normal-component `clamp` sits outside it and stays on. Setting the GRID op's wall_damp to 1
    # therefore removes exactly the sign-conditional term and nothing else. (mpm_gather's own
    # wall_damp is left alone -- it is a fixed per-particle multiply, i.e. linear.)
    wd_old = pr.gridup.wall_damp
    pr.gridup.wall_damp = 1.0
    R["T5b_no_sign_conditional_damp"] = superpose(
        "5b", [P, Q], mkE, pr.step(E_cell=zero_E(pr)),
        extra=" with grid wall_damp=1 (sign-conditional damp OFF, normal clamp still ON)")
    pr.gridup.wall_damp = wd_old

    # HOW MANY COLUMNS OF A ARE CONTAMINATED? A per-cell parameter is polluted if the cell owns
    # even one particle that can reach a slab -- that is the practical cost of the wall clamp.
    bad = torch.zeros(C + 1, dtype=torch.bool, device=dev)
    bad[pr.cid[mw]] = True
    R["T5_cells_touching_wall"] = {"n_cells_contaminated": int(bad[1:].sum().item()),
                                   "n_cells": C,
                                   "frac": float(bad[1:].float().mean().item())}
    log(f"[5] cells owning >=1 wall-reachable particle: "
        f"{R['T5_cells_touching_wall']['n_cells_contaminated']}/{C} "
        f"({R['T5_cells_touching_wall']['frac']:.1%}) -- these columns of A are the polluted ones")

    # ---- 6. EXTRA: one FRAME (10 substeps) instead of one substep -------------------------- #
    #  the measured data is at FRAME resolution; a frame is round(dt/dt_sub) substeps, and the
    #  state each substep sees then depends on theta. This is the test that decides whether
    #  "inject the measured positions at every frame" really removes the time integration.
    R["T6_multistep"] = {}
    for ns in sorted({1, 2, 5, pr.n_sub_per_frame}):
        base_ns = pr.step(E_cell=zero_E(pr), n_sub=ns)
        mkE_ns = lambda m, ns=ns: {"E_cell": torch.where(m, E0, torch.zeros_like(E0)), "n_sub": ns}
        tag = f" over {ns} substep(s)" + (" = ONE FRAME" if ns == pr.n_sub_per_frame else "")
        d = superpose("6", [P, Q], mkE_ns, base_ns, extra=tag)
        # homogeneity too: over ns substeps, is the passive response still proportional to s?
        h = {}
        r1 = pr.step(E_cell=E0, n_sub=ns) - base_ns
        for s in (0.5, 2.0):
            rs = pr.step(E_cell=s * E0, n_sub=ns) - base_ns
            h[str(s)] = rel(rs - s * r1, s * r1)
        d["homogeneity_worst"] = max(h.values())
        d["homogeneity"] = h
        log(f"      ...homogeneity over {ns} substep(s): worst = {d['homogeneity_worst']:.3e}")
        R["T6_multistep"][f"n_sub_{ns}"] = d
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--per-parent", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--n-grid", type=int, default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--shrink", type=float, default=1.0,
                    help="contract the sheet about the domain centre (no-wall control; try 0.8)")
    ap.add_argument("--tol-factor", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    pr = Probe(device=args.device, per_parent=args.per_parent, n_grid=args.n_grid,
               warmup=args.warmup, dtype=args.dtype, shrink=args.shrink)
    print(f"[probe] built + warmed up {args.warmup} frames in {time.time()-t0:.1f}s  "
          f"device={args.device}  dtype={args.dtype}  shrink={args.shrink}  particles={pr.p.n}  "
          f"cells={pr.n_cells}  dt_sub={pr.dt_sub}  substeps/frame={pr.n_sub_per_frame}",
          flush=True)
    with torch.no_grad():
        R = run_all(pr, tol_factor=args.tol_factor)
    R["config"] = {"device": args.device, "per_parent": args.per_parent, "warmup": args.warmup,
                   "n_grid": args.n_grid, "dtype": args.dtype, "shrink": args.shrink}
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"linearity_{args.device.replace(':','')}_"
                                   f"{args.dtype}_pp{args.per_parent}"
                                   f"{'' if args.shrink == 1.0 else f'_shrink{args.shrink:g}'}.json")
    json.dump(R, open(out, "w"), indent=2, default=float)
    print(f"[probe] wrote {out}   total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
