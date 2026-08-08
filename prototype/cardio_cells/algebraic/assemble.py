"""assemble.py -- TASK C: build A and b for the algebraic (no-integration) cardio fit, and test
whether  A . theta_true = b  actually holds at the theta the simulation used.

THE OBJECT
====================================================================================================
theta = (E_1..E_C, gain_1..gain_C)     per-cell Young's modulus and per-cell active-force gain.

For a FROZEN state x = (pos, vel, F, C, Jp) and a frozen outer-schedule body force, one MLS-MPM
substep [mpm_strain, mpm_scatter, mpm_grid_update, mpm_gather] produces a particle acceleration

    a(theta) = (v_new - v_old) / dt_sub          [Np, 2]  -> flattened to [2Np]

If that map is AFFINE (task A and task B both measure that it is, for one substep, away from the
wall), then

    a(theta) = A(x) theta + a0(x),      a0 = a(0)  (drag, m.C affine momentum, m.v transport)

and the algebraic constraint stated in the brief is

    A theta = b,        b := a_obs - a0 = the theta-dependent part of the observed acceleration.

WHAT IS VERIFIED
----------------------------------------------------------------------------------------------------
  residual = ||A theta_true - b|| / ||b||       at the theta the simulation actually used.

A is assembled COLUMN BY COLUMN from the solver itself: column j = (a(s.e_j) - a0)/s. For an affine
map that difference is the exact derivative for ANY s -- which is checked (scale sweep), and which
makes the residual above a genuine C-way superposition test, not a tautology: nothing in the
assembly knows that the 2C one-hot responses should add up to the response at theta_true.

CONTROLS (each result is reported with its null)
  * fidelity      : this replay vs plexus.engine.run on the same spec  -> must be ~0
  * scale         : columns at s and 10s                               -> must agree to fp floor
  * wrong-A null  : A from a different frame, same theta               -> must be O(1)
  * shuffle null  : A theta_permuted vs b                              -> must be O(1)
  * inset control : cells pulled off the domain wall                   -> isolates the wall kink
  * substep sweep : n_sub = 1, 2, 5, 10 (10 = one FRAME)               -> the design question

NOTE ON THE GAIN (task B blocker, confirmed here). `material_cardio_cells.yaml` uses `active_force`,
whose amplitude is a GLOBAL python float -- there is no per-cell gain in the spec. So a per-cell gain
is introduced HERE, in the harness, in the only place it can go without touching an operator another
agent owns: the frozen per-particle body force delta is split into its active-force part and its
passive part (drag), and the active part is scaled by gain_{c(p)}. That is exactly the route task B
describes (mpm_scatter.py:75 -> the mass.V term of the momentum scatter). The warm-up uses the same
gains, so theta_true is genuinely the theta the trajectory was produced with.

usage:
  PYTHONPATH=/workspace/Plexus/src python assemble.py --device cuda:1 [--cells 24] [--per-parent 40]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, "/workspace/Plexus/src")

import plexus.operators                                    # noqa: F401  self-register the library
from plexus.engine import build, _resolve_emit, _selector_mask
from plexus.models.entities import _lame
from plexus.models.registry import get_operator
from plexus.schema import load

CONFIG = "/workspace/Plexus/config/material/material_cardio_cells.yaml"
HERE = "/workspace/Plexus/prototype/cardio_cells/algebraic"
SUBSTEP_TOKENS = ["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"]


# --------------------------------------------------------------------------------------------- #
#  a small label map: a Voronoi tessellation, either filling the domain (like the real
#  segmentation, which covers 100% of its tif) or inset away from the wall (the control)
# --------------------------------------------------------------------------------------------- #
def make_label_tif(path, n_cells, size=256, mode="full", seed=3):
    import tifffile
    rng = np.random.default_rng(seed)
    lo, hi = (0.02, 0.98) if mode == "full" else (0.20, 0.80)
    sites = rng.uniform(lo, hi, size=(n_cells, 2))
    gx, gy = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size,
                         indexing="ij")
    d = ((gx[..., None] - sites[:, 0]) ** 2 + (gy[..., None] - sites[:, 1]) ** 2)
    lab = (np.argmin(d, axis=-1) + 1).astype(np.int32)          # [size,size], x-major
    if mode == "inset":                                          # background outside the inset box
        outside = (gx < lo) | (gx > hi) | (gy < lo) | (gy > hi)
        lab[outside] = 0
    # LabelImageField reads a tif as [row, col] = [y, x], flips rows, then permutes to [nx, ny].
    img = lab.T[::-1, :].copy()
    tifffile.imwrite(path, img)
    return path


# --------------------------------------------------------------------------------------------- #
#  the system: build the small spec, warm it up on theta_true, freeze a state, replay one substep
# --------------------------------------------------------------------------------------------- #
class System:
    def __init__(self, device="cuda:1", n_cells=24, per_parent=40, n_grid=64, warmup=12,
                 dtype="float64", mode="full", gain_lo=0.5, gain_hi=1.5, tif_size=256,
                 real=False, wall_damp=None):
        self.dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
        torch.set_default_dtype(self.dtype)
        self.device, self.n_cells, self.mode = device, int(n_cells), mode

        raw = yaml.safe_load(open(CONFIG))
        raw["general"]["n_frames"] = int(warmup)
        if real:
            # THE ACTUAL SYSTEM: the measured 472-cell segmentation + its measured props, untouched.
            self.mode = mode = "real"
            raw["general"]["name"] = "assemble_real"
        else:
            tif = os.path.join(HERE, f"small_labels_{mode}_{n_cells}.tif")
            make_label_tif(tif, n_cells, size=tif_size, mode=mode)
            raw["general"]["name"] = f"assemble_small_{mode}"
            raw["fields"]["cells"]["source"] = tif               # absolute path -> read directly
            raw["fields"]["mpm_grid"]["n_grid"] = int(n_grid)
            raw["fields"]["activation"]["res"] = int(n_grid)
            raw["sets"]["cell"]["per_parent"] = int(n_cells)
            raw["sets"]["mpm_particle"]["per_parent"] = int(per_parent)
            for o in raw["operators"]:                           # no props json -> deterministic E
                if o.get("op") == "seed_from_segmentation":
                    o.pop("props", None)
        if wall_damp is not None:
            # MITIGATION TEST (task A's suggestion): the sign-conditional tangential damp
            # `where(gv_y > 0, gv_y*wd, gv_y)` is a kink only because wd != 1.
            for o in raw["operators"]:
                if o.get("op") == "mpm_grid_update":
                    o["wall_damp"] = float(wall_damp)
        self.raw = raw
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="assemble_small_")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(raw, f)
        self.spec_path = path
        self.sim = load(path)

        self.dt = float(self.sim.dt)
        self.dt_sub = float([s for s in self.sim.schedule
                             if isinstance(s, dict) and "substep_dt" in s][0]["substep_dt"])
        self.n_sub_per_frame = max(1, round(self.dt / self.dt_sub))
        self.outer_tokens = [t for t in self.sim.schedule if not isinstance(t, dict)]

        self.H = build(self.sim, device)
        self.H.emit_order = _resolve_emit(self.sim, self.H)
        self.inst = [(o.op,
                      get_operator(o.op, variant=o.impl)(
                          {**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, device),
                      o.on)
                     for o in self.sim.operators]
        self.p = self.H.level("mpm_particle")
        self.g = self.H.field("mpm_grid")
        self.gather = [ob for nm, ob, _ in self.inst if nm == "mpm_gather"][0]

        with torch.no_grad():
            # the seeded per-cell Young's modulus (one pass of the outer schedule runs the seed)
            self._outer(0, gain_cell=None)
            self.cid = self.p.cell_id.long()
            assert int(self.cid.min()) >= 1
            self.C = int(self.cid.max().item())
            self.E_true = torch.zeros(self.C + 1, device=device, dtype=self.dtype)
            self.E_true.index_copy_(0, self.cid, self.p.youngs.to(self.dtype))
            gg = torch.Generator().manual_seed(101)
            self.gain_true = torch.zeros(self.C + 1, device=device, dtype=self.dtype)
            self.gain_true[1:] = (gain_lo + (gain_hi - gain_lo)
                                  * torch.rand(self.C, generator=gg)).to(device, self.dtype)
            self.theta_true = torch.cat([self.E_true[1:], self.gain_true[1:]])
            self.warmup_frames = int(warmup)
            self._warmup(self.warmup_frames)
            self._snapshot(self.warmup_frames)

    # -- engine replay ------------------------------------------------------------------------ #
    def _tok(self, token):
        for nm, ob, sel in self.inst:
            if nm != token:
                continue
            deltas = ob(self.H, _selector_mask(self.H, sel))
            block = getattr(ob, "INTEGRAND", None)
            for lvlname, d in deltas.items():
                self.H.add_delta(lvlname, d, block)

    def _outer(self, tick, gain_cell=None):
        """One pass of the OUTER schedule, splitting the active-force delta out so a per-cell gain
        can scale it. Returns (act, passive) for the particle level."""
        H = self.H
        H.frame = tick
        H.zero_delta()
        act = None
        prev = H.delta("mpm_particle").clone()
        for tok in self.outer_tokens:
            self._tok(tok)
            cur = H.delta("mpm_particle")
            if tok == "active_force":
                act = (cur - prev).clone()
            prev = cur.clone()
        full = H.delta("mpm_particle").clone()
        if act is None:
            act = torch.zeros_like(full)
        passive = full - act
        if gain_cell is not None:
            H._delta["mpm_particle"] = passive + gain_cell[self.cid][:, None] * act
        return act, passive

    def _warmup(self, n):
        for tick in range(n):
            self._outer(tick, gain_cell=self.gain_true)
            self.H.sub_dt = self.dt_sub
            for _ in range(self.n_sub_per_frame):
                for tok in SUBSTEP_TOKENS:
                    self._tok(tok)
            self.H.sub_dt = None

    def _snapshot(self, tick):
        p = self.p
        self.act0, self.pass0 = self._outer(tick, gain_cell=self.gain_true)
        self.cell_delta_norm = float(self.H.delta("cell").abs().max().item())
        self.state0 = p.state.clone()
        self.F0, self.C0, self.Jp0 = p.F.clone(), p.C.clone(), p.Jp.clone()
        self.v0 = p.get("vel").clone()
        self.x0 = p.get("pos").clone()
        self.Np = int(p.n)

    def restore(self):
        p = self.p
        p.state = self.state0.clone()
        p.F, p.C, p.Jp = self.F0.clone(), self.C0.clone(), self.Jp0.clone()

    # -- the one map under test ---------------------------------------------------------------- #
    def step(self, E_cell, gain_cell, n_sub=1):
        """Restore the frozen state, install theta, run n_sub substeps, return a [2Np] flat."""
        H, p = self.H, self.p
        self.restore()
        mu, la = _lame(E_cell[self.cid])
        p.mu, p.la = mu, la
        H.zero_delta()
        H._delta["mpm_particle"] = self.pass0 + gain_cell[self.cid][:, None] * self.act0
        H.sub_dt = self.dt_sub
        for _ in range(n_sub):
            for tok in SUBSTEP_TOKENS:
                self._tok(tok)
        H.sub_dt = None
        a = (p.get("vel") - self.v0) / (self.dt_sub * n_sub)
        return a.reshape(-1).clone()

    def a_of_theta(self, theta, n_sub=1):
        E = torch.zeros(self.C + 1, device=self.device, dtype=self.dtype)
        gn = torch.zeros_like(E)
        E[1:] = theta[:self.C]
        gn[1:] = theta[self.C:]
        return self.step(E, gn, n_sub=n_sub)

    # -- assembly ------------------------------------------------------------------------------ #
    def assemble(self, n_sub=1, sE=100.0, sg=1.0):
        """A [2Np, 2C], a0 [2Np]. Column j = (a(s e_j) - a0)/s -- exact for an affine map."""
        t0 = time.time()
        z = torch.zeros(2 * self.C, device=self.device, dtype=self.dtype)
        a0 = self.a_of_theta(z, n_sub=n_sub)
        A = torch.zeros(a0.numel(), 2 * self.C, device=self.device, dtype=self.dtype)
        for j in range(2 * self.C):
            s = sE if j < self.C else sg
            e = z.clone()
            e[j] = s
            A[:, j] = (self.a_of_theta(e, n_sub=n_sub) - a0) / s
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return A, a0, time.time() - t0

    # -- masks / diagnostics -------------------------------------------------------------------- #
    def interior_particle_mask(self, margin_cells=4.0):
        m = margin_cells * self.g.dx
        X = self.x0
        return (X[:, 0] > m) & (X[:, 0] < 1.0 - m) & (X[:, 1] > m) & (X[:, 1] < 1.0 - m)

    def flat_mask(self, pm):
        return pm[:, None].expand(-1, 2).reshape(-1)

    def wall_diag(self):
        """How much wall machinery is live at the frozen state (the known kink sources)."""
        from plexus.operators.mpm_grid import stencil_offsets, bspline
        self.restore()
        mu, la = _lame(self.E_true[self.cid])
        self.p.mu, self.p.la = mu, la
        self.H.zero_delta()
        self.H._delta["mpm_particle"] = self.pass0 + self.gain_true[self.cid][:, None] * self.act0
        self.H.sub_dt = self.dt_sub
        self._tok("mpm_strain")
        self._tok("mpm_scatter")
        g = self.g
        gv = (g.mv / g.m.clamp(min=1e-10)[:, None]).view(g.nx, g.ny, 2)
        occ = (g.m > 1e-12).view(g.nx, g.ny)
        bnd = 3
        hit = torch.zeros_like(occ)
        hit[:bnd, :] |= (gv[:bnd, :, 0] < 0)
        hit[g.nx - bnd + 1:, :] |= (gv[g.nx - bnd + 1:, :, 0] > 0)
        hit[:, :bnd] |= (gv[:, :bnd, 1] < 0)
        hit[:, g.ny - bnd + 1:] |= (gv[:, g.ny - bnd + 1:, 1] > 0)
        hit &= occ
        damp = torch.zeros_like(occ)
        damp[:bnd, :] |= (gv[:bnd, :, 1] > 0)
        damp[g.nx - bnd + 1:, :] |= (gv[g.nx - bnd + 1:, :, 1] > 0)
        damp &= occ
        self._tok("mpm_grid_update")
        X = self.p.get("pos")
        offsets = stencil_offsets(2, X.device)
        fx, weight, flat = bspline(X, g.inv_dx, offsets, g.shape, False)
        gvn = g.v[flat].view(self.p.n, offsets.shape[0], 2)
        new_V = (weight[..., None] * gvn).sum(1)
        cb = self.gather.wall_contact
        near = (X[:, 0] < cb) | (X[:, 0] > 1 - cb) | (X[:, 1] < cb) | (X[:, 1] > 1 - cb)
        vmax = min(self.gather.vmax, 0.4 * g.dx / self.dt_sub)
        self.H.sub_dt = None
        return {
            "grid_nodes_massive": int(occ.sum().item()),
            "grid_nodes_normal_clamped": int(hit.sum().item()),
            "grid_nodes_sign_conditional_damped": int(damp.sum().item()),
            "particles_in_wall_contact_band": int(near.sum().item()),
            "cfl_hits": int((new_V.norm(dim=1) > vmax).sum().item()),
            "cfl_vmax": float(vmax),
            "max_speed": float(new_V.norm(dim=1).max().item()),
            "max_abs_cell_delta_(a_max_clamp_target)": self.cell_delta_norm,
        }


# --------------------------------------------------------------------------------------------- #
def rel(x, ref):
    return float(x.norm().item() / max(ref.norm().item(), 1e-300))


def report_system(sy, args, log):
    light = bool(getattr(args, "light", False))
    R = {"mode": sy.mode, "C": sy.C, "Np": sy.Np, "n_grid": sy.g.nx, "dtype": str(sy.dtype),
         "dt": sy.dt, "dt_sub": sy.dt_sub, "substeps_per_frame": sy.n_sub_per_frame,
         "warmup_frames": sy.warmup_frames,
         "E_true_range": [float(sy.E_true[1:].min()), float(sy.E_true[1:].max())],
         "gain_true_range": [float(sy.gain_true[1:].min()), float(sy.gain_true[1:].max())],
         "norm_active_force_delta": float(sy.act0.norm()),
         "norm_passive_delta": float(sy.pass0.norm())}
    R["wall_diag"] = sy.wall_diag()
    log(f"\n[{sy.mode}] C={sy.C} cells, Np={sy.Np} particles, grid {sy.g.nx}^2, "
        f"dtype={sy.dtype}, warmup={sy.warmup_frames} frames")
    log(f"   ||active_force delta|| = {R['norm_active_force_delta']:.4g}   "
        f"||passive (drag) delta|| = {R['norm_passive_delta']:.4g}")
    log(f"   wall diag: {R['wall_diag']}")

    # ---- ASSEMBLE ------------------------------------------------------------------------- #
    A, a0, t_asm = sy.assemble(n_sub=1)
    a_true = sy.a_of_theta(sy.theta_true, n_sub=1)
    b = a_true - a0
    resid_vec = A @ sy.theta_true - b
    R["A_shape"] = list(A.shape)
    R["assembly_seconds"] = t_asm
    R["solver_evaluations"] = 2 * sy.C + 1
    R["norm_a_obs"] = float(a_true.norm())
    R["norm_a0_offset"] = float(a0.norm())
    R["offset_over_total"] = float(a0.norm() / a_true.norm())
    R["norm_b"] = float(b.norm())
    R["RESIDUAL_rel_b"] = rel(resid_vec, b)
    R["RESIDUAL_rel_a_obs"] = rel(resid_vec, a_true)
    log(f"   A shape {tuple(A.shape)}  assembled in {t_asm:.3f} s "
        f"({2*sy.C+1} solver evaluations)")
    log(f"   ||a_obs||={R['norm_a_obs']:.6g}  ||a0||={R['norm_a0_offset']:.6g} "
        f"(offset/total={R['offset_over_total']:.3f})  ||b||={R['norm_b']:.6g}")
    log(f"   *** RESIDUAL ||A.theta_true - b||/||b|| = {R['RESIDUAL_rel_b']:.3e}   "
        f"(/||a_obs|| = {R['RESIDUAL_rel_a_obs']:.3e})")

    # interior / wall split of the residual
    pm = sy.interior_particle_mask(4.0)
    fm = sy.flat_mask(pm)
    R["interior_particle_fraction"] = float(pm.float().mean())
    if pm.any():
        R["RESIDUAL_interior"] = rel(resid_vec[fm], b[fm])
    if (~pm).any():
        R["RESIDUAL_wallband"] = rel(resid_vec[~fm], b[~fm])
    R["resid_frac_on_wall_particles"] = float(
        (resid_vec[~fm].norm() ** 2 / resid_vec.norm().clamp(min=1e-300) ** 2).item())
    log(f"   interior (margin 4dx, {100*R['interior_particle_fraction']:.1f}% of particles): "
        f"{R.get('RESIDUAL_interior', float('nan')):.3e}   "
        f"wall band: {R.get('RESIDUAL_wallband', float('nan')):.3e}   "
        f"(frac of residual^2 on wall particles: {R['resid_frac_on_wall_particles']:.4f})")

    # ---- CONTROLS ---------------------------------------------------------------------------- #
    # (1) repeat null: the same theta twice
    a_rep = sy.a_of_theta(sy.theta_true, n_sub=1)
    R["null_repeat"] = rel(a_rep - a_true, b)
    # (2) scale independence of the columns (affine => column is s-independent)
    if light:
        R["null_scale_10x_columns"] = R["null_scale_10x_residual"] = float("nan")
    else:
        A10, a0b, _ = sy.assemble(n_sub=1, sE=1000.0, sg=10.0)
        R["null_scale_10x_columns"] = float((A10 - A).norm().item() / max(A.norm().item(), 1e-300))
        R["null_scale_10x_residual"] = rel(A10 @ sy.theta_true - b, b)
        del A10
    # (3) wrong-A null: columns from a DIFFERENT theta-independent state are not available without
    #     a second frame, so use the strongest cheap null: A with its columns randomly permuted.
    perm = torch.randperm(2 * sy.C, device=A.device)
    R["null_permuted_columns"] = rel(A[:, perm] @ sy.theta_true - b, b)
    # (4) shuffled theta
    R["null_shuffled_theta"] = rel(A @ sy.theta_true[perm] - b, b)
    # (5) zero-model null (A = 0): how big is b relative to itself -> 1 by construction
    R["null_zero_A"] = 1.0
    log(f"   nulls: repeat={R['null_repeat']:.3e}  columns@10x={R['null_scale_10x_columns']:.3e} "
        f"(resid {R['null_scale_10x_residual']:.3e})  permuted-A={R['null_permuted_columns']:.3e} "
        f" shuffled-theta={R['null_shuffled_theta']:.3e}")

    # ---- residual at OTHER thetas (the affine claim is not special to theta_true) ------------- #
    gth = torch.Generator().manual_seed(7)
    others = {}
    for tag, mk in (("0.5x", lambda t: 0.5 * t),
                    ("2x", lambda t: 2.0 * t),
                    ("random", lambda t: t * (0.2 + 2.0 * torch.rand(t.shape, generator=gth)
                                              .to(t.device, t.dtype)))):
        th = mk(sy.theta_true)
        bb = sy.a_of_theta(th, n_sub=1) - a0
        others[tag] = rel(A @ th - bb, bb)
    R["residual_other_theta"] = others
    log(f"   residual at other theta: " + "  ".join(f"{k}={v:.3e}" for k, v in others.items()))

    # ---- E-only and gain-only blocks ---------------------------------------------------------- #
    AE, Ag = A[:, :sy.C], A[:, sy.C:]
    R["block_norm_E"] = float(AE.norm())
    R["block_norm_gain"] = float(Ag.norm())
    R["contrib_norm_E"] = float((AE @ sy.theta_true[:sy.C]).norm())
    R["contrib_norm_gain"] = float((Ag @ sy.theta_true[sy.C:]).norm())
    log(f"   block contributions: ||A_E theta_E||={R['contrib_norm_E']:.4g}  "
        f"||A_g theta_g||={R['contrib_norm_gain']:.4g}")

    # ---- GRAM / identifiability (exact-data least squares) ------------------------------------ #
    Ad = A.double()
    if light:                                   # tall-skinny: singular values from the Gram matrix
        G = Ad.T @ Ad
        sv = torch.linalg.eigvalsh(G).clamp(min=0).flip(0).sqrt()
    else:
        sv = torch.linalg.svdvals(Ad)
    R["singular_values"] = [float(x) for x in sv[:5]] + ["..."] + [float(x) for x in sv[-3:]]
    R["rank_tol1e-10"] = int((sv > sv[0] * 1e-10).sum().item())
    R["cond_A"] = float(sv[0] / sv[-1])
    R["cond_G"] = float((sv[0] / sv[-1]) ** 2)
    # cond(A) in raw units mixes E (~100) with gain (~1); the meaningful conditioning is of the
    # DIMENSIONLESS map A.diag(theta_true) (response to a fractional change in each parameter).
    Adim = Ad * sy.theta_true.double()[None, :]
    if light:
        svd = torch.linalg.eigvalsh(Adim.T @ Adim).clamp(min=0).flip(0).sqrt()
    else:
        svd = torch.linalg.svdvals(Adim)
    R["cond_A_dimensionless"] = float(svd[0] / svd[-1])
    R["cond_G_dimensionless"] = float((svd[0] / svd[-1]) ** 2)
    colE = Adim[:, :sy.C].norm(dim=0)
    colg = Adim[:, sy.C:].norm(dim=0)
    R["per_cell_1pct_signal_over_b_E"] = [float(0.01 * colE.min() / b.norm()),
                                          float(0.01 * colE.median() / b.norm()),
                                          float(0.01 * colE.max() / b.norm())]
    R["per_cell_1pct_signal_over_b_gain"] = [float(0.01 * colg.min() / b.norm()),
                                             float(0.01 * colg.median() / b.norm()),
                                             float(0.01 * colg.max() / b.norm())]
    R["eff_rank_dimensionless"] = {f"sv>{t:g}*svmax": int((svd > svd[0] * t).sum().item())
                                   for t in (1e-2, 1e-4, 1e-6, 1e-8)}
    # noise propagation on ONE frame: perturb b by 1e-3 relative white noise, resolve
    gnoise = torch.Generator(device=Ad.device).manual_seed(11)
    nz = torch.randn(b.shape, generator=gnoise, device=Ad.device, dtype=torch.float64)
    bn = b.double() + 1e-3 * b.double().norm() / nz.norm() * nz
    Gm = Ad.T @ Ad
    sol_n = torch.linalg.solve(Gm, Ad.T @ bn)
    R["theta_rel_error_at_1e-3_noise_on_b"] = float(
        (sol_n - sy.theta_true.double()).norm() / sy.theta_true.double().norm())
    log(f"   effective rank (dimensionless): {R['eff_rank_dimensionless']} of {2*sy.C};  "
        f"theta error at 1e-3 noise on b = {R['theta_rel_error_at_1e-3_noise_on_b']:.3e}")
    log(f"   dimensionless cond(A)={R['cond_A_dimensionless']:.3e}  "
        f"cond(G)={R['cond_G_dimensionless']:.3e};  1% single-cell signal / ||b||: "
        f"E {R['per_cell_1pct_signal_over_b_E'][1]:.2e} (median), "
        f"gain {R['per_cell_1pct_signal_over_b_gain'][1]:.2e} (median)")
    if light:                                   # normal equations (A is 3.6 GB at the real size)
        G = Ad.T @ Ad
        sol = torch.linalg.solve(G, Ad.T @ b.double())
    else:
        sol = torch.linalg.lstsq(Ad, b.double().unsqueeze(1)).solution.squeeze(1)
    R["lstsq_theta_rel_error_noisefree"] = float(
        (sol - sy.theta_true.double()).norm() / sy.theta_true.double().norm())
    log(f"   cond(A)={R['cond_A']:.3e}  cond(G)={R['cond_G']:.3e}  rank={R['rank_tol1e-10']}/{2*sy.C}"
        f"   lstsq theta recovery (noise-free) rel err = "
        f"{R['lstsq_theta_rel_error_noisefree']:.3e}")

    # ---- THE DESIGN QUESTION: substep vs frame cadence ---------------------------------------- #
    sweep = {}
    for ns in ((1, sy.n_sub_per_frame) if light else (1, 2, 5, sy.n_sub_per_frame)):
        An, a0n, tn = sy.assemble(n_sub=ns)
        bn = sy.a_of_theta(sy.theta_true, n_sub=ns) - a0n
        sweep[ns] = {"residual_rel_b": rel(An @ sy.theta_true - bn, bn),
                     "residual_rel_a_obs": rel(An @ sy.theta_true - bn,
                                               sy.a_of_theta(sy.theta_true, n_sub=ns)),
                     "assembly_s": tn}
    R["substep_sweep"] = sweep
    log("   substep sweep (n_sub -> residual/||b||): " +
        "  ".join(f"{k}:{v['residual_rel_b']:.3e}" for k, v in sweep.items()) +
        f"   [{sy.n_sub_per_frame} substeps = ONE FRAME]")

    # ---- multi-frame: assemble at several frames --------------------------------------------- #
    return R, A, a0, b


def frame_scan(args, log, frames):
    """Re-warm the same system to several different frames and report the residual at each."""
    out = {}
    for w in frames:
        sy = System(device=args.device, n_cells=args.cells, per_parent=args.per_parent,
                    n_grid=args.n_grid, warmup=w, dtype=args.dtype, mode=args.mode)
        with torch.no_grad():
            A, a0, t = sy.assemble(n_sub=1)
            b = sy.a_of_theta(sy.theta_true, n_sub=1) - a0
            r1 = rel(A @ sy.theta_true - b, b)
            Af, a0f, _ = sy.assemble(n_sub=sy.n_sub_per_frame)
            bf = sy.a_of_theta(sy.theta_true, n_sub=sy.n_sub_per_frame) - a0f
            rf = rel(Af @ sy.theta_true - bf, bf)
        out[w] = {"residual_1substep": r1, "residual_1frame": rf, "assembly_s": t}
        log(f"   frame {w:3d}: 1-substep residual {r1:.3e}   1-frame residual {rf:.3e}")
        del sy
        torch.cuda.empty_cache()
    return out


def cost_projection(args, log):
    """Time one substep at several particle counts, project the assembly cost for 472 cells."""
    out = {}
    for (C, pp) in [(24, 40), (24, 500), (100, 500), (472, 500)]:   # last = the REAL system size
        sy = System(device=args.device, n_cells=C, per_parent=pp, n_grid=128,
                    warmup=4, dtype=args.dtype, mode="full")
        with torch.no_grad():
            z = torch.zeros(2 * sy.C, device=sy.device, dtype=sy.dtype)
            for _ in range(3):
                sy.a_of_theta(z)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                sy.a_of_theta(z)
            torch.cuda.synchronize()
            t = (time.time() - t0) / 20
        out[f"C{C}_pp{pp}_N{sy.Np}"] = {"N": sy.Np, "substep_eval_s": t}
        log(f"   N={sy.Np:7d}: one substep evaluation {1e3*t:.2f} ms")
        del sy
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cells", type=int, default=24)
    ap.add_argument("--per-parent", type=int, default=40)
    ap.add_argument("--n-grid", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--mode", default="full", choices=["full", "inset"])
    ap.add_argument("--fidelity", action="store_true", help="check the replay against engine.run")
    ap.add_argument("--frames", default="", help="comma list of warm-up frames to scan")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--both-modes", action="store_true")
    ap.add_argument("--real", action="store_true", help="the ACTUAL 472-cell spec, untouched")
    ap.add_argument("--light", action="store_true", help="skip the memory-heavy extras (big runs)")
    ap.add_argument("--wall-damp", type=float, default=None,
                    help="override the grid wall_damp (1.0 removes the sign-conditional kink)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    RES = {"argv": vars(args)}
    torch.manual_seed(0)

    with torch.no_grad():
        modes = ["full", "inset"] if args.both_modes else [args.mode]
        for mode in modes:
            a2 = argparse.Namespace(**vars(args))
            a2.mode = mode
            sy = System(device=args.device, n_cells=args.cells, per_parent=args.per_parent,
                        n_grid=args.n_grid, warmup=args.warmup, dtype=args.dtype, mode=mode,
                        real=args.real, wall_damp=args.wall_damp)
            if args.fidelity:
                RES[f"fidelity_{mode}"] = fidelity(sy, args, log)
            R, A, a0, b = report_system(sy, a2, log)
            RES[sy.mode] = R
            del sy, A
            torch.cuda.empty_cache()

        if args.frames:
            log("\n[frame scan]")
            RES["frame_scan"] = frame_scan(args, log, [int(x) for x in args.frames.split(",")])
        if args.cost:
            log("\n[cost]")
            RES["cost"] = cost_projection(args, log)

    out = args.out or os.path.join(HERE, f"assemble_{args.mode}_{args.dtype}_C{args.cells}.json")
    json.dump(RES, open(out, "w"), indent=1, default=str)
    log(f"\nwrote {out}")


def fidelity(sy, args, log):
    """This replay vs plexus.engine.run: same spec, same frames, compare final positions.

    The replay applies per-cell gains, which engine.run cannot; so the check is run with
    gain == 1 everywhere, which IS the spec's own dynamics."""
    from plexus.engine import run as engine_run
    sy2 = System(device=args.device, n_cells=args.cells, per_parent=args.per_parent,
                 n_grid=args.n_grid, warmup=0, dtype=args.dtype, mode=sy.mode,
                 gain_lo=1.0, gain_hi=1.0)
    n = args.warmup
    sy2._warmup(n)
    xr = sy2.p.get("pos").clone()
    sim = load(sy2.spec_path)
    sim.n_frames = n - 1                      # engine.run does ticks 0..n_frames inclusive
    He, _ = engine_run(sim, None, args.device)
    xe = He.level("mpm_particle").get("pos")
    d = float((xr - xe).norm() / max(xe.norm().item(), 1e-30))
    log(f"   [fidelity] replay vs engine.run after {n} ticks: rel pos diff = {d:.3e} "
        f"(max abs {float((xr-xe).abs().max()):.3e})")
    del sy2, He
    torch.cuda.empty_cache()
    return {"rel_pos_diff": d}


if __name__ == "__main__":
    main()
