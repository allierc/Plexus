"""The energy ONE ion feels along a channel's ion path, term by term -- the ruler for "why does nothing pass".

    PYTHONPATH=src python tools/channel_axis_energy.py channel/exp04_v14e [--ion K] [--n 121] [--json out.jsonl]

The run's spec is built exactly as the engine builds it (one frame, on the local GPU), then every other ion is moved
to the bottom of the box, beyond every pair law's cutoff, and a single probe ion is walked down the centre line of
the barrier's first ion path ([z, rp, R, dry, cx, cy] rows). At each point every operator that pushes that ion set
is asked for its velocity on the probe, with its per-frame guards OFF (`step_max`, `f_max`, `hard`: they cap steps,
they are not energies). The force is v / mobility, projected on the path's tangent, and integrated:

    U_k(s) = - integral_0^s F_k . t ds'        (sim energy; kT = 1 in these specs)

per operator k (the protein's excluded volume, the charged residues, the lumen carbonyls, the Born barrier, the
field), and their sum. The printed barrier of a term is max U_k - min U_k over the stretch inside the membrane core.

What it is NOT: a free energy. The carbonyls are held at their seeded places (their breathing is not averaged), the
other ions are gone (no knock-on), and the electrolyte field is the one the first frame solved. A barrier here that
is tens of kT is a wall no thermal ion crosses in a run; a flat profile that still passes nothing points at the
many-ion physics instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _spec_path(run):
    for p in (os.path.join(ROOT, "graphs_data", run, "spec.yaml"), os.path.join(ROOT, "config", run + ".yaml")):
        if os.path.isfile(p):
            return p
    raise SystemExit(f"no spec for {run}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="<group>/<name>, e.g. channel/exp04_v14e")
    ap.add_argument("--ion", default=None, help="the ion set walked down the path (default: the spec's cation)")
    ap.add_argument("--n", type=int, default=121, help="points along the path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", default=None, help="append one line with the profile to this file")
    a = ap.parse_args()

    import plexus.operators  # noqa: F401  (fills the registry)
    import plexus.operators.channel_ops as C
    from plexus import engine
    from plexus.schema import load

    sim = load(_spec_path(a.run))
    sim.n_frames = 1
    made = []
    for cls in (C.PairPotential, C.SlabBarrier, C.FieldForce):
        init0 = cls.__init__

        def init(self, *args, _init0=init0, **kw):
            _init0(self, *args, **kw)
            made.append(self)
        cls.__init__ = init
    H, _ = engine.run(sim, device=a.device)
    ion = a.ion or next((s for s in ("K", "Na") if s in H.levels), None)
    L = float(getattr(sim.units, "length_um", 1e-3) or 1e-3) * 1e3                      # nm per world unit

    def pushes(op):
        if isinstance(op, C.PairPotential):
            return ion in op.A
        return getattr(op, "at", None) == ion

    ops = [op for op in made if pushes(op)]
    bar = next((op for op in ops if isinstance(op, C.SlabBarrier) and op.pores is not None), None)
    if bar is None:
        raise SystemExit("no slab_barrier with ion paths on this ion")
    P = bar.pores[0].double().cpu().numpy()                             # [z, rp, R, dry, cx, cy]
    z0, h = float(bar.z0), float(bar.h)
    zs = np.linspace(P[:, 0].min(), P[:, 0].max(), a.n)
    cx = np.interp(zs, P[:, 0], P[:, 4]); cy = np.interp(zs, P[:, 0], P[:, 5]); rp = np.interp(zs, P[:, 0], P[:, 1])
    pts = np.stack([cx, cy, zs], 1)
    tang = np.gradient(pts, axis=0); ds = np.linalg.norm(tang, axis=1); tang = tang / ds[:, None].clip(1e-12)

    # the guards off: they cap a step, they are not an energy
    for op in ops:
        if isinstance(op, C.PairPotential):
            op.f_max = 0.0
            op.step_max = {k: None for k in op.step_max}
        if isinstance(op, C.SlabBarrier):
            op.step_max = 0.0; op.hard = False

    # every other ion to the bottom of the box, out of every pair law's reach
    g = torch.Generator().manual_seed(0)
    for nm, lv in H.levels.items():
        if nm in ("K", "Na", "Cl"):
            a_, b_ = lv.state_schema["pos"]
            st = lv.state.clone()
            n = st.shape[0]
            far = torch.stack([torch.rand(n, generator=g), torch.rand(n, generator=g), 0.01 + 0.04 * torch.rand(n, generator=g)], 1)
            st[:, a_:b_] = far.to(st)
            lv.state = st
    lv = H.levels[ion]
    a_, b_ = lv.state_schema["pos"]

    names, F = [], np.zeros((len(ops), a.n))
    for k, op in enumerate(ops):
        label = type(op).__name__
        if isinstance(op, C.PairPotential):
            label += " " + "+".join(op.laws) + " <-> " + (op.B[0] + (".." + op.B[-1] if len(op.B) > 1 else ""))
        names.append(label)
    for i in range(a.n):
        st = lv.state.clone(); st[0, a_:b_] = torch.tensor(pts[i], dtype=st.dtype); lv.state = st
        for k, op in enumerate(ops):
            if isinstance(op, C.SlabBarrier):
                op._prev = None
            out = op.forward(H) or {}
            v = out.get(ion)
            if v is None or (torch.is_tensor(v) and v.numel() == 0):
                continue
            v0 = v[0].detach().double().cpu().numpy()
            # v = mu F on the probe: the pair laws carry the mobility per set, the others one number
            if isinstance(op, C.PairPotential):
                ma = op._side(H, op.A, op.q_a, op.mu_a)[7]
                mu = float(ma[0]) if ma is not None else float(op.mu_a)
            else:
                mu = float(op.mu)
            F[k, i] = float(np.dot(v0 / float(mu), tang[i]))
    U = -np.concatenate([np.zeros((len(ops), 1)), np.cumsum(0.5 * (F[:, 1:] + F[:, :-1]) * ds[1:], axis=1)], 1)
    core = np.abs(zs - z0) <= h
    U = U - U[:, [0]]
    tot = U.sum(0)
    print(f"\n{a.run}: one {ion} walked down path 1 ({a.n} points, z {zs[0] * L:.2f} -> {zs[-1] * L:.2f} nm; "
          f"the core {(z0 - h) * L:.2f}-{(z0 + h) * L:.2f} nm); energies in kT")
    print(f"{'term':58s} {'barrier in core':>16s} {'U at core centre':>17s}")
    ic = int(np.argmin(np.abs(zs - z0)))
    for k, nm in enumerate(names):
        Uc = U[k][core]
        print(f"{nm[:58]:58s} {Uc.max() - Uc.min():16.1f} {U[k, ic]:17.1f}")
    Tc = tot[core]
    print(f"{'TOTAL':58s} {Tc.max() - Tc.min():16.1f} {tot[ic]:17.1f}")
    print("\n z (nm)   rp (nm)   total   " + "  ".join(f"T{k}" for k in range(len(names))))
    for i in range(0, a.n, max(1, a.n // 30)):
        print(f"{zs[i] * L:7.2f}  {rp[i] * L:7.3f}  {tot[i]:7.1f}   " + "  ".join(f"{U[k, i]:6.1f}" for k in range(len(names))))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"spec": a.run, "ion": ion, "z_nm": (zs * L).round(3).tolist(), "rp_nm": (rp * L).round(4).tolist(),
                                "terms": {nm: U[k].round(2).tolist() for k, nm in enumerate(names)},
                                "barrier_in_core_kT": {nm: round(float(U[k][core].max() - U[k][core].min()), 2)
                                                       for k, nm in enumerate(names)},
                                "total_barrier_in_core_kT": round(float(Tc.max() - Tc.min()), 2)}) + "\n")


if __name__ == "__main__":
    main()
