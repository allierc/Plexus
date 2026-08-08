"""probe_clamps.py -- TASK B instrumentation: which non-linearities on the
per-cell-parameter -> particle-acceleration path are ACTIVE at the cardio_cells settings?

Read-only. Builds the material_cardio_cells spec in-process, runs a few frames, and wraps the
four substep operators to record, per substep:

  * H.delta('cell') magnitude              -> is the a_max=200 clamp (mpm_scatter:66) live?
  * H.delta('mpm_particle') magnitude      -> the body force that a_max does NOT clamp
  * snow / liquid / visco / occ masks      -> are the material branches live?
  * grid_update boundary clamp             -> how many MASSFUL grid cells does it change?
  * gather wall_contact band               -> fraction of particles scaled by wall_damp
  * gather CFL cap                         -> max|v| vs vmax (cap fires iff max == vmax)
  * gather position clamp                  -> particles pinned at 2dx / box-2dx

Usage:  PYTHONPATH=/workspace/Plexus/src python probe_clamps.py --device cuda:1 --frames 40
"""
from __future__ import annotations

import argparse
import copy
import json

import torch

from plexus import schema
from plexus import engine as E
from plexus.paths import config_path
from plexus.operators.mpm_grid import stencil_offsets, bspline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--per_parent", type=int, default=0, help="override particles/cell (0 = spec)")
    args = ap.parse_args()

    sim = schema.load(config_path("material", "material_cardio_cells.yaml"))
    sim.n_frames = args.frames
    if args.per_parent:
        sim.sets["mpm_particle"]["per_parent"] = args.per_parent

    dev = args.device
    stats = {"substeps": 0}
    rec = {k: 0.0 for k in
           ("cell_delta_absmax", "part_delta_absmax", "gv_bnd_changed", "gv_bnd_massful",
            "grid_massful", "near_wall_frac", "vmax_hits", "maxspeed", "pos_pinned_frac",
            "J_min", "J_max", "act_stress_present")}

    orig_scatter = E.get_operator("mpm_scatter").forward
    orig_update = E.get_operator("mpm_grid_update").forward
    orig_gather = E.get_operator("mpm_gather").forward

    state = {}

    def scatter(self, H, mask=None):
        p = H.level(self.at)
        rec["cell_delta_absmax"] = max(rec["cell_delta_absmax"], float(H.delta("cell").abs().max()))
        rec["part_delta_absmax"] = max(rec["part_delta_absmax"], float(H.delta(p.name).abs().max()))
        rec["act_stress_present"] = float(getattr(H, "active_stress", None) is not None)
        F = p.F
        J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
        rec["J_min"] = min(rec["J_min"] or 1e9, float(J.min())) if stats["substeps"] else float(J.min())
        rec["J_max"] = max(rec["J_max"], float(J.max()))
        out = orig_scatter(self, H, mask)
        g = H.field(self.to)
        state["gv_raw"] = g.mv / g.m.clamp(min=1e-10)[:, None]
        state["gm"] = g.m.clone()
        return out

    def update(self, H, mask=None):
        out = orig_update(self, H, mask)
        g = H.field(self.at)
        gv0, gm = state["gv_raw"], state["gm"]
        nx, ny = g.nx, g.ny
        bnd = torch.zeros(nx, ny, dtype=torch.bool, device=gm.device)
        bnd[:3, :] = True; bnd[nx - 2:, :] = True     # ix<3 or ix>nx-3  (the code's `> nx-bnd`)
        bnd[:, :3] = True; bnd[:, ny - 2:] = True
        bnd = bnd.reshape(-1)
        massful = gm > 1e-8
        diff = (g.v - gv0).abs().max(dim=1).values > 0
        rec["gv_bnd_changed"] = max(rec["gv_bnd_changed"], float((diff & massful).sum()))
        rec["gv_bnd_massful"] = max(rec["gv_bnd_massful"], float((bnd & massful).sum()))
        rec["grid_massful"] = max(rec["grid_massful"], float(massful.sum()))
        return out

    def gather(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.frm)
        X = p.get("pos")
        box = [float(b) for b in H.world_size][:2]
        cb = self.wall_contact
        near = (X[:, 0] < cb) | (X[:, 0] > box[0] - cb) | (X[:, 1] < cb) | (X[:, 1] > box[1] - cb)
        rec["near_wall_frac"] = max(rec["near_wall_frac"], float(near.float().mean()))
        out = orig_gather(self, H, mask)
        V = p.get("vel"); Xn = p.get("pos")
        dt = float(getattr(H, "sub_dt"))
        vmax = min(self.vmax, 0.4 * g.dx / dt)
        sp = V.norm(dim=1)
        rec["maxspeed"] = max(rec["maxspeed"], float(sp.max()))
        rec["vmax_hits"] = max(rec["vmax_hits"], float((sp >= vmax * (1 - 1e-6)).sum()))
        lo, hi = 2 * g.dx, box[0] - 2 * g.dx
        pinned = (Xn[:, 0] <= lo) | (Xn[:, 0] >= hi) | (Xn[:, 1] <= lo) | (Xn[:, 1] >= box[1] - 2 * g.dx)
        rec["pos_pinned_frac"] = max(rec["pos_pinned_frac"], float(pinned.float().mean()))
        stats["substeps"] += 1
        return out

    E.get_operator("mpm_scatter").forward = scatter
    E.get_operator("mpm_grid_update").forward = update
    E.get_operator("mpm_gather").forward = gather

    once = {}

    def on_frame(H, tick):
        if tick == 1:
            p = H.level("mpm_particle")
            once.update(
                n_particles=int(p.n),
                parent_name=str(getattr(p, "parent_name", None)),
                is_snow_any=bool(p.is_snow.any()), is_liquid_any=bool(p.is_liquid.any()),
                is_visco_any=bool(p.is_visco.any()),
                occ_all_one=bool((p.occ == 1).all()),
                mu_over_E=float(p.mu[0] / p.youngs[0]), la_over_E=float(p.la[0] / p.youngs[0]),
                youngs_min=float(p.youngs.min()), youngs_max=float(p.youngs.max()),
                p_vol=float(p.p_vol[0]), mass=float(p.mass[0]),
                emit_order=dict(H.emit_order),
                F_res_inv=bool(getattr(p, "F_res_inv", None) is not None),
                gain_buffer=bool(getattr(p, "gain", None) is not None),
                obstacles=list(getattr(H, "obstacles", []) or []),
                world_size=[float(x) for x in H.world_size],
                periodic=bool(getattr(H, "periodic", False)),
            )

    E.run(sim, out_path=None, device=dev, on_frame=on_frame, progress=True)
    out = {"once": once, "max_over_substeps": rec, "substeps": stats["substeps"]}
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
