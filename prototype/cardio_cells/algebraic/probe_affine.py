"""probe_affine.py -- TASK B measurements backing CODEMAP.md.

Three read-only experiments on the live material_cardio_cells state, all at frame ~15
(peak of the first pulse):

  A) BOUNDARY BREAKDOWN in mpm_grid_update: how many massful grid cells are touched by
     (a) the reflective clamp (lines 118-119, a KINK in theta),
     (b) the sign-conditional tangential damp (lines 121-122, a KINK),
     (c) the unconditional tangential damp (lines 123-124, LINEAR).

  B) AFFINITY of the map theta = (E_1..E_472) -> particle positions after
       (b1) ONE substep, and (b2) ONE frame (10 substeps),
     measured as the midpoint superposition defect
       d = || g(.5a+.5b) - .5 g(a) - .5 g(b) ||  /  || g(b) - g(a) ||
     Null control: the same quantity with the physics replaced by a known-affine map is 0;
     here the null is the FP-roundoff floor, estimated by re-running g(a) twice.

  C) COLUMN SPARSITY: perturb ONE cell's E and count how many particles' one-substep
     displacement changes -> the cost model for building a column of A.

Usage: PYTHONPATH=/workspace/Plexus/src python probe_affine.py --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from plexus import schema
from plexus import engine as E
from plexus.paths import config_path
from plexus.models.entities import _lame


def snapshot(H):
    p = H.level("mpm_particle")
    return dict(state=p.state.clone(), F=p.F.clone(), C=p.C.clone(),
                delta={k: v.clone() for k, v in H._delta.items()})


def restore(H, snap):
    p = H.level("mpm_particle")
    p.state = snap["state"].clone(); p.F = snap["F"].clone(); p.C = snap["C"].clone()
    H._delta = {k: v.clone() for k, v in snap["delta"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frame", type=int, default=15)
    args = ap.parse_args()
    dev = args.device

    sim = schema.load(config_path("material", "material_cardio_cells.yaml"))
    sim.n_frames = args.frame
    out = {}

    # ---- A) boundary breakdown -------------------------------------------------------- #
    orig_update = E.get_operator("mpm_grid_update").forward
    bnd_stats = {"clamp_x": 0, "clamp_y": 0, "damp_cond": 0, "damp_uncond": 0, "massful": 0}

    def update(self, H, mask=None):
        g = H.field(self.at)
        gv = (g.mv / g.m.clamp(min=1e-10)[:, None]).view(g.nx, g.ny, 2)
        nx, ny = g.nx, g.ny
        massful = (g.m > 1e-8).view(nx, ny)
        ix = torch.arange(nx, device=gv.device); iy = torch.arange(ny, device=gv.device)
        lox, hix = ix < 3, ix > nx - 3
        loy, hiy = iy < 3, iy > ny - 3
        # (a) reflective clamps actually bite where the normal component points into the wall
        cx = ((gv[lox, :, 0] < 0) & massful[lox, :]).sum() + ((gv[hix, :, 0] > 0) & massful[hix, :]).sum()
        cy = ((gv[:, loy, 1] < 0) & massful[:, loy]).sum() + ((gv[:, hiy, 1] > 0) & massful[:, hiy]).sum()
        # (b) sign-conditional tangential damp on the x-walls (lines 121-122)
        dc = ((gv[lox, :, 1] > 0) & massful[lox, :]).sum() + ((gv[hix, :, 1] > 0) & massful[hix, :]).sum()
        # (c) unconditional tangential damp on the y-walls (lines 123-124)
        du = (massful[:, loy].sum() + massful[:, hiy].sum())
        for k, v in (("clamp_x", cx), ("clamp_y", cy), ("damp_cond", dc), ("damp_uncond", du)):
            bnd_stats[k] = max(bnd_stats[k], int(v))
        bnd_stats["massful"] = max(bnd_stats["massful"], int(massful.sum()))
        return orig_update(self, H, mask)

    E.get_operator("mpm_grid_update").forward = update
    H, _ = E.run(sim, out_path=None, device=dev, progress=False)
    E.get_operator("mpm_grid_update").forward = orig_update
    out["A_boundary_max_over_substeps"] = bnd_stats

    # ---- rebuild the substep loop by hand from the same spec -------------------------- #
    from plexus.models.registry import get_operator
    ops = {o.op: get_operator(o.op)({**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, dev)
           for o in sim.operators}
    sub = [t for st in sim.schedule if isinstance(st, dict) for t in st["steps"]]
    dt_sub = [float(st["substep_dt"]) for st in sim.schedule if isinstance(st, dict)][0]
    n_sub = max(1, round(sim.dt / dt_sub))
    out["substep_tokens"] = sub
    out["n_substeps_per_frame"] = n_sub

    p = H.level("mpm_particle")
    cid = p.cell_id.long()                      # 1..472 per particle
    n_cells = int(cid.max())
    E_cell0 = torch.zeros(n_cells + 1, device=dev)
    E_cell0.scatter_(0, cid, p.youngs)          # per-cell E (particles in a cell share it exactly)
    snap = snapshot(H)

    def g_of(theta_cell, steps):
        """positions after `steps` substeps with per-cell Young's modulus `theta_cell` [n_cells+1]."""
        restore(H, snap)
        pp = H.level("mpm_particle")
        py = theta_cell[cid]
        mu, la = _lame(py)
        pp.mu, pp.la = mu, la
        H.sub_dt = dt_sub
        for _ in range(steps):
            for nm in sub:
                ops[nm](H, None)
        H.sub_dt = None
        return H.level("mpm_particle").get("pos").clone()

    torch.manual_seed(0)
    ta = E_cell0.clone()
    tb = E_cell0 * (1.0 + 0.30 * torch.rand(n_cells + 1, device=dev))    # +0..30% per cell
    tm = 0.5 * (ta + tb)

    for steps, tag in ((1, "one_substep"), (n_sub, "one_frame")):
        ga, gb, gm_ = g_of(ta, steps), g_of(tb, steps), g_of(tm, steps)
        ga2 = g_of(ta, steps)
        defect = (gm_ - 0.5 * ga - 0.5 * gb).norm()
        signal = (gb - ga).norm()
        floor = (ga2 - ga).norm()
        out[f"B_affinity_{tag}"] = {
            "superposition_defect_norm": float(defect),
            "parameter_signal_norm": float(signal),
            "relative_defect": float(defect / signal.clamp(min=1e-30)),
            "fp_repeat_floor_norm": float(floor),
            "max_abs_defect_per_particle": float((gm_ - 0.5 * ga - 0.5 * gb).norm(dim=1).max()),
            "max_abs_signal_per_particle": float((gb - ga).norm(dim=1).max()),
        }

    # ---- C) column sparsity + timing --------------------------------------------------- #
    base = g_of(ta, 1)
    for c in (7, 200, 400):
        tc = ta.clone(); tc[c] = tc[c] * 1.10
        gc = g_of(tc, 1)
        d = (gc - base).norm(dim=1)
        nz = int((d > 0).sum())
        out[f"C_column_support_cell{c}"] = {
            "particles_in_cell": int((cid == c).sum()),
            "particles_moved": nz,
            "fraction_of_all": nz / int(p.n),
            "max_disp_change": float(d.max()),
        }

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        g_of(ta, 1)
    torch.cuda.synchronize()
    out["timing_one_substep_incl_restore_s"] = (time.time() - t0) / 20

    restore(H, snap)
    torch.cuda.synchronize()
    t0 = time.time()
    H.sub_dt = dt_sub
    for _ in range(50):
        for nm in sub:
            ops[nm](H, None)
    torch.cuda.synchronize()
    out["timing_one_substep_pure_s"] = (time.time() - t0) / 50
    out["n_particles"] = int(p.n)
    out["n_cells"] = n_cells

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
