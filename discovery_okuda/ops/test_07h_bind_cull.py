#!/usr/bin/env python
"""07d -- 07c's cell-owned adhesion on a sheet that refines SMOOTHLY.

    python test_07d_local.py --frames 401

WHAT CHANGES FROM 07c, and it is one operator. `Sheet.refine()` is a global 1->4 split fired when the
MEAN edge passes a target, so the sheet can only jump by a factor of four: twice in 401 frames, at 126
and 342, and every distribution built on it sawtooths. Here `bm_refine_local` bisects the longest edge
of whichever faces are over the trigger, propagating into the neighbour when it disagrees, so the mesh
gains tens of elements a frame instead of tripling. Its split invariance is certified separately
(`bm_refine_local.gates`, on a LOADED sheet): lambda 2.0e-16, volume, area, mass and energy exactly
unchanged, and the mesh stays closed.

THE HOOK IS `_epi_anchor`, which the parent calls once a frame, so `frame()` is not copied. The global
trigger is disabled by putting `edge_target` out of reach rather than by editing the parent -- the
operator it guards is still there, and a run that wanted it back would only move the number.

AND THE ADHESION STILL DOES NOT FOLLOW THE MESH. 07c established that a refinement re-points plaques
and never re-seeds them; local refinement changes the node set far more often than twice, so that rule
is doing much more work here, which is the point of running it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_05b_plaque as B                                              # noqa: E402
from bm_refine_batch import BatchRefiner                                 # noqa: E402
from bm_refine_local import LocalRefiner                                 # noqa: E402
from test_07c_cell_plaque import Rig07c                                  # noqa: E402

NAME = "07h_bind_and_cull"
SRC = "06_spheroid_ecm"


class Rig07d(Rig07c):
    """07c, with `bm_refine_local` in place of the global 1->4 split.

    `batched=True` swaps the sequential Rivara recursion for the GPU pass of `bm_refine_batch`, whose
    split invariance is certified to the same tolerances (lambda 6.1e-13, area and energy exactly 0)
    and which reaches a converged mesh in seconds rather than minutes. 07d runs the CPU one and 07e
    the GPU one, on the same tissue, so the difference between the folders is the operator.
    """

    def __init__(self, split_budget=150, edge_trigger=1.45, every=10, batched=False,
                 cull_below=0.01, bind_max=3.0, **P):
        super().__init__(**P)
        self.e0 = float((self.sheet.x[self.sheet.Ed[:, 1]]
                         - self.sheet.x[self.sheet.Ed[:, 0]]).norm(dim=1).mean())
        self.local_target = edge_trigger * self.e0
        self.split_budget = int(split_budget)
        # `every` IS THE ENGINE'S OWN NAME FOR THIS. `plexus.engine._gate` reads `every` from an
        # operator's params and runs it when `tick % every == 0`, beside `after_frame` and
        # `before_frame` -- a multi-rate schedule is already something the container expresses, so
        # this rig uses that word rather than inventing one. Rivara bisection is a sequential python
        # recursion and cannot be batched, so running it every frame left the run at 3% GPU with the
        # refinement, not the physics, as the cost. The mesh does not need rewiring at the frame
        # rate: over 401 frames it grows by a factor of a few, and a tick of 10 still gives forty
        # refinement events against the global scheme's two.
        self.every = max(1, int(every))
        self.edge_target = 1e30              # the global trigger, put out of reach
        self.batched = bool(batched)
        self.cull_below = float(cull_below)
        # A CLUSTER FORMS WHERE THE MEMBRANE IS CLOSE ENOUGH TO BIND. 07g seeds a plaque at
        # whatever separation the sheet happens to have -- 3.47 um at frame 0, five rest
        # lengths -- which is a bond across a gap no integrin spans. Beyond `bind_max` rest
        # lengths no plaque is created at all, which makes G73 a property of the seeding
        # rather than a threshold to check afterwards.
        self.bind_max = float(bind_max)
        self._ref = BatchRefiner(self.sheet) if self.batched else LocalRefiner(self.sheet)
        self._splits = 0
        self._culled = 0
        print(f"[07d] seeded edge {self.e0:.5f}, split when a face's longest edge exceeds "
              f"{self.local_target:.5f} ({edge_trigger}x), up to {self.split_budget} every "
              f"{self.every} frames, {'batched GPU' if self.batched else 'sequential CPU'}",
              flush=True)

    def _cull(self):
        """NO BONDS, NO PLAQUE -- and its receptors go back to the cell.

        `k_off(f)` removes BONDS; nothing removed the entry, so a plaque that had let go stayed in the
        contact set, kept being drawn, and kept entering the length statistics as though it were an
        adhesion. Measured on 07g's last frame: 98.1% of plaques sit within 3 l0 holding ~0.75 bonds,
        and the 33-l0 maximum that failed G72 is ONE entry of 72,924 whose bonds are 0.017. The gate
        was reading bookkeeping, not adhesion.

        This is not `plaque_rupture` returning as a length threshold -- that operator was deleted on
        purpose when rupture became a rate. It is the rate's own consequence: the cluster is gone when
        its last bond is, and the receptors it was holding are the cell's again.
        """
        nb = self.clutch.Nb
        if not nb.numel():
            return 0
        # NO BONDS **AND** OUT OF REACH. Culling on the bond count alone wipes the set at frame 0:
        # a plaque is seeded with zero bonds and the clutch fills it over the following frames, so
        # "no bonds" is the state every cluster is BORN in. A cluster is gone when it has let go and
        # the membrane has moved beyond binding distance -- one that is still in contact can re-bind,
        # which is what k_on is for.
        att = (self.x_epi[self.ct_tri] * self.ct_w[:, :, None]).sum(1)
        sep = (self.sheet.x[self.ct_node] - att).norm(dim=1)
        gone = (nb < self.cull_below) & (sep > self.bind_max * self.clutch.l0)
        if not bool(gone.any()):
            return 0
        keep = ~gone
        back = torch.zeros(self._nF, device=self.dev, dtype=self.dtype)
        back.index_add_(0, self.ct_face[gone], nb[gone])
        self.clutch.Nf = self.clutch.Nf + back            # conserved: G75 still has to hold
        self.ct_node, self.ct_face = self.ct_node[keep], self.ct_face[keep]
        self.ct_w, self.ct_tri = self.ct_w[keep], self.ct_tri[keep]
        self.clutch.Nb, self.clutch.D = nb[keep], self.clutch.D[keep]
        return int(gone.sum())

    def _epi_anchor(self, t):
        # the parent calls this once a frame, before the substeps: the one place a topology change
        # can happen without a frame being half-integrated against two different meshes
        self._culled += self._cull()
        if int(t) % self.every:
            return super()._epi_anchor(t)
        n = (self._ref.refine_where(self.local_target) if self.batched else
             self._ref.refine_where(self.local_target, max_splits=self.split_budget))
        if n:
            self._splits += n
            self.build_contact()             # re-points the plaques; never re-seeds them
            self.lam_el, self._pv = self.sheet.spectral_rate(iters=40, return_vec=True)
            self.n_sub = self._nsub()
        return super()._epi_anchor(t)


def build(cls=None, extra=None, add_args=None, pass_args=(), default_name=None,
          return_rig=False, **over):
    """Parse this ladder's arguments, build `cls`, and run it.

    `add_args` registers a rig's OWN options on the same parser and `pass_args` names which of them
    reach its constructor, so a descendant adds a knob without owning a parser or a loop.
    """
    cls = cls or Rig07d
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=401)
    ap.add_argument("--N0", type=int, default=12)
    ap.add_argument("--Nf0", type=float, default=300.0)
    ap.add_argument("--cull-below", dest="cull_below", type=float, default=0.01)
    ap.add_argument("--bind-max", dest="bind_max", type=float, default=3.0)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--batched", action="store_true")
    # THE POOL HAS TO HOLD WHAT THE TRIGGER ASKS FOR. The batched refiner converges on
    # the edge target rather than stopping at a split budget, and holding a tissue that
    # grows 4x in radius at 1.45x the seeded edge needs about 16x the faces -- which is
    # 81,920, exactly what max_refine=2 allocates, so it ran out. This is sizing, not
    # physics: the same mesh, with room to exist.
    ap.add_argument("--max-refine", dest="max_refine", type=int, default=3)
    ap.add_argument("--name", default=default_name or NAME)
    if add_args:
        add_args(ap)
    a = ap.parse_args()
    over.update({k: getattr(a, k) for k in pass_args})
    d = os.path.join(B.LOG, a.name)
    os.makedirs(d, exist_ok=True)
    kw = dict(subdiv=4, E=400.0, thickness=2.0e-3, nu=0.3, kn=5.0, sigma_T=7.0, zeta=20.0,
             s_target=1.0, k_drive=50.0, dev=a.device, max_refine=a.max_refine,
             edge_trigger=1.45,
              reseed=True, tau_bm=40.0, rho_crit=0.0)
    kw.update(over)
    rig = cls(N0=a.N0, Nf0=a.Nf0, split_budget=a.budget, every=a.every,
              batched=a.batched, cull_below=a.cull_below, bind_max=a.bind_max, **kw)
    run(rig, a, d, extra=extra)
    return rig if return_rig else None


def run(rig, a, d, extra=None):
    """The loop, the store, the two mesh gates and the spec -- shared by every rig in this ladder.

    07d, 07e, 07g and 07h were made by substituting strings into one another and had begun to drift:
    the same loop existed four times, so a fix to the store or a new series had to be made four times
    and was not. The rig is the only thing that differs between them, so the rig is the argument.
    """
    S = {k: [] for k in ("t", "cells", "plaques", "ppc", "nb_med", "nf_mean", "receptor_total",
                         "lam", "n_face", "n_node", "edge", "splits", "n_sub")}
    store, i, t0 = {}, 0, time.time()
    for t in range(a.frames):
        rig.frame(t)
        if not rig.alive():
            print(f"[07d] DIVERGED at {t}", flush=True)
            break
        if t % 20 == 0:
            print(f"[07e] frame {t:4d}  faces {rig.sheet.m:7d}  nodes {rig.sheet.n:7d}  "
                  f"cells {rig._nF:5d}  plaques {rig.ct_node.numel():7d}  "
                  f"splits {rig._splits:7d}  culled {rig._culled:6d}  "
                  f"{time.time()-t0:6.0f}s", flush=True)
        if t % 2 == 0:
            nb = rig.clutch.Nb
            l1, _ = rig.sheet.stretch_geo()
            X = rig.sheet.x
            S["t"].append(t); S["cells"].append(int(rig._nF))
            S["plaques"].append(int(rig.ct_node.numel()))
            S["ppc"].append(float(rig.ct_node.numel() / max(rig._nF, 1)))
            S["nb_med"].append(float(nb.median())); S["nf_mean"].append(float(rig.clutch.Nf.mean()))
            S["receptor_total"].append(float(nb.sum() + rig.clutch.Nf.sum()))
            S["lam"].append(float(l1.mean()))
            S["n_face"].append(int(rig.sheet.m)); S["n_node"].append(int(rig.sheet.n))
            S["edge"].append(float((X[rig.sheet.Ed[:, 1]]
                                    - X[rig.sheet.Ed[:, 0]]).norm(dim=1).mean()) / rig.e0)
            S["splits"].append(int(rig._splits))
            # THE SUBSTEP COUNT IS NOT A CONSTANT and census.py had to take it from a seeded rig
            # because no run recorded it: it tracks lambda_max of the elastic Hessian, which rises
            # with stretch (05a: 21 -> 194 over 401 frames), so every per-frame cost quoted from the
            # seeded value is a lower bound. One int a frame settles it.
            S["n_sub"].append(int(rig.n_sub))
            # A RIG MAY HAVE SERIES OF ITS OWN. 07l carries myosin, which no earlier rig has and
            # which its gates are written on; asking the rig rather than widening this dict keeps the
            # loop shared and lets the next one add a state without touching it.
            for k, v in (rig.extra_series() if hasattr(rig, "extra_series") else {}).items():
                S.setdefault(k, []).append(float(v))
            att = (rig.x_epi[rig.ct_tri] * rig.ct_w[:, :, None]).sum(1)
            for k, v in (("t", np.int32(t)), ("x", X.float().cpu().numpy()),
                         ("f", rig.sheet.Fc.cpu().numpy().astype(np.int32)),
                         ("v", l1.float().cpu().numpy()),
                         ("r", (rig.sheet.areal_density() / rig.sheet.rho0).float().cpu().numpy()),
                         ("e", rig.x_epi.float().cpu().numpy()),
                         ("n", rig.ct_node.cpu().numpy().astype(np.int32)),
                         ("p", att.float().cpu().numpy()),
                         ("nb", nb.float().cpu().numpy()),
                         ("nf", rig.clutch.Nf.float().cpu().numpy()),
                         ("cf", rig.ct_face.cpu().numpy().astype(np.int32))):
                store[f"{k}{i}"] = v
            i += 1
    np.savez_compressed(os.path.join(d, "bm_frames.npz"), n_kept=np.int32(i),
                        FE=rig.F_epi.cpu().numpy().astype(np.int32),
                        centre=rig.c.float().cpu().numpy(), scale=np.float64(rig.scale), **store)
    step = np.abs(np.diff(np.asarray(S["n_face"], float))) / np.maximum(
        np.asarray(S["n_face"], float)[:-1], 1.0)
    res = dict(run=a.name, N0=a.N0, frames=len(S["t"]), splits=int(rig._splits),
               G78=dict(name="the sheet refines smoothly: no kept-frame interval grows the face "
                             "count by more than 20%",
                        max_step=float(step.max()) if step.size else 0.0, threshold=0.20,
                        steps_that_move=int((step > 1e-12).sum()), of=int(step.size),
                        passed=bool(step.size and step.max() <= 0.20)),
               G44=dict(name="mean edge stays in [0.8, 1.7] x seeded, every frame",
                        range=[float(min(S["edge"])), float(max(S["edge"]))],
                        passed=bool(min(S["edge"]) >= 0.8 and max(S["edge"]) <= 1.7)),
               series=S)
    json.dump(res, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    from spec_06 import write_spec
    write_spec(d, rig, name=a.name, frames=a.frames, matrix_src=SRC,
               extra={**dict(kind="cell-owned adhesion + local refinement", N0=a.N0,
                             split_budget=a.budget, splits=int(rig._splits)), **(extra or {})})
    g, h = res["G78"], res["G44"]
    print(f"[07d] {len(S['t'])} kept frames in {time.time()-t0:.0f}s -- faces {S['n_face'][0]} -> "
          f"{S['n_face'][-1]} in {rig._splits} bisections, cells {S['cells'][0]} -> "
          f"{S['cells'][-1]}, plaques {S['plaques'][0]} -> {S['plaques'][-1]}, ppc "
          f"{S['ppc'][0]:.2f} -> {S['ppc'][-1]:.2f}, lam_geo {S['lam'][-1]:.3f}", flush=True)
    print(f"[07d] G78 {'PASS' if g['passed'] else 'FAIL'} (max step {100*g['max_step']:.1f}%, moves "
          f"on {g['steps_that_move']} of {g['of']}), G44 {'PASS' if h['passed'] else 'FAIL'} "
          f"(edge {h['range'][0]:.3f}..{h['range'][1]:.3f}x) -> {d}", flush=True)


def main():
    build()


if __name__ == "__main__":
    main()
