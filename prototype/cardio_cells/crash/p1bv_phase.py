"""p1bv_phase.py -- how much of the beat has ANY gain signal at all?

p1bv_base.py found ||A_gain|| = 0 EXACTLY at ticks 100 and 210, at every linearization base:
the gain columns of A vanish, the normal equations are singular, and no F, however perfect, buys
per-cell gain anything at those phases.  Probe B ran at tick 165 only.

The gain multiplies `act0` = the active_force delta split out by System._outer (assemble.py:184-204,
217).  If act0 == 0 at a tick, then H._delta["mpm_particle"] = pass0 + gain*0 and theta_gain is
structurally unidentifiable at that frame -- not attenuated, ABSENT.

This walks the beat and records ||act0|| and ||pass0|| per tick, plus, at the ticks where act0 is
nonzero, the actual gain-column norm of the base-0 one-frame assembly (a 3-column probe, not the
full 200, so it is cheap).

usage: PYTHONPATH=/workspace/Plexus/src python p1bv_phase.py --device cuda:1 --ticks 240
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("/workspace/Plexus/src", ALG, DISC, HERE):
    sys.path.insert(0, _p)

from assemble import System, SUBSTEP_TOKENS                          # noqa: E402
from recover import install_E                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="p1bv_phase")
    ap.add_argument("--ticks", type=int, default=240)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    a = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent,
                           n_grid=a.n_grid, warmup=0, window=150, dtype="float64",
                           mode="full", e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    t_start = time.time()
    torch.manual_seed(0)
    with torch.no_grad():
        sy = System(device=args.device, n_cells=args.cells, per_parent=args.per_parent,
                    n_grid=args.n_grid, warmup=0, dtype=args.dtype, mode=args.mode)
        C = sy.C
        g = torch.Generator().manual_seed(2026)
        E = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=g)).to(sy.device, sy.dtype)
        gn = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=g)).to(sy.device, sy.dtype)
        sy.E_true[1:], sy.gain_true[1:] = E, gn
        sy.theta_true = torch.cat([sy.E_true[1:], sy.gain_true[1:]])
        install_E(sy, sy.E_true)

        rec = []
        for tick in range(a.ticks):
            act, pas = sy._outer(tick, gain_cell=sy.gain_true)
            rec.append({"tick": tick, "norm_act0": float(act.norm()),
                        "norm_pass0": float(pas.norm()),
                        "max_abs_act0": float(act.abs().max())})
            sy.H.sub_dt = sy.dt_sub
            for _ in range(sy.n_sub_per_frame):
                for tok in SUBSTEP_TOKENS:
                    sy._tok(tok)
            sy.H.sub_dt = None

        on = [r for r in rec if r["norm_act0"] > 0]
        log(f"[phase] {a.ticks} ticks; ||act0|| > 0 at {len(on)}/{a.ticks} "
            f"({100.0*len(on)/a.ticks:.1f}%) of frames")
        log(f"        ticks with act0 != 0: "
            f"{[r['tick'] for r in on][:80]}{' ...' if len(on) > 80 else ''}")
        log(f"        ||act0|| max {max(r['norm_act0'] for r in rec):.5g} at tick "
            f"{max(rec, key=lambda r: r['norm_act0'])['tick']}; ||pass0|| median "
            f"{sorted(r['norm_pass0'] for r in rec)[len(rec)//2]:.5g}")
        log("\n    tick   ||act0||     ||pass0||")
        for r in rec:
            if r["tick"] % 5 == 0 or r["norm_act0"] > 0:
                log(f"    {r['tick']:>4d} {r['norm_act0']:>11.5g} {r['norm_pass0']:>13.5g}")

    R = {"args": vars(a), "per_tick": rec,
         "n_on": len(on), "n_ticks": a.ticks,
         "duty_cycle": len(on) / a.ticks,
         "wall_seconds": time.time() - t_start}
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json  [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()
