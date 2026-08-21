#!/usr/bin/env python
"""08b -- pass 2: the tissue re-solved with the membrane as a brake, and a bud out of the hole.

    python test_08b_bud.py --list
    python test_08b_bud.py gate --frames 401 --every 10 --batched --device cuda:1

WHAT PASS 2 IS. 08a phase 1 ran the membrane against the REPLAYED tissue and recorded `bm_gate.npz`:
mean bound integrin per direction, per frame, where an empty bin means no adhesion and therefore a
hole. This rebuilds the vertex model with `ecm_gate_growth` reading that map -- so a cell under intact
membrane has its cycle braked and a cell under the hole does not -- and then runs the SAME membrane rig
against the new tissue. The hole is in the same place (the cap is a fixed direction, not a consequence
of the tissue), so the two runs are comparable frame for frame and the only difference is what the
cells were allowed to do.

WHY IT IS A SWEEP AND NOT ONE RUN. Whether a patch of faster-cycling cells becomes a BUD or just a
slightly thicker region is a question about the mechanics that resist it, not about the growth gate.
The variants below vary one mechanism each, so a failure says which one was in the way.

THE CACHE KEY CARRIES THE OVERRIDES. `tissue.load_or_build` hashes every argument that changes the
result, including `op_overrides` and `append_ops`, so two variants cannot collide on one tissue --
which is a bug this file's own history records twice.

AND THE RIG'S CACHE IS REBOUND, NOT PARAMETERISED. `RealDriver` reads a module-level `CACHE`; pass 2
points it at the tissue it just built. That is a rebinding of a constant and it is done in ONE place,
loudly, because a run that silently read the wrong tissue would look exactly like a run that worked.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_05b_plaque as B                                              # noqa: E402
import test_05l_supply as L                                              # noqa: E402
import test_06_breach as BR                                              # noqa: E402
import test_07h_bind_cull as H                                           # noqa: E402
import tissue as TIS                                                     # noqa: E402
from test_08a_protrusion import KDEG, OFF_PHI, OFF_THETA, SPOT, Rig08a   # noqa: E402

GATE_SRC = "08a_hole_rot"          # the phase-1 run whose ligation map every variant reads

MAP = None                         # filled in build_tissue: the ligation map this run reads


# `sharp`, `a_sw` AND `hill` ARE SET BY MEASUREMENT, not by taste. The map has real bin-to-bin
# scatter -- at frame 0 a bin holds 2.5 sheet faces -- so a normal cell's deficit reaches 0.6 raw, and
# at the first setting tried (sharp 2, a_sw 0.3, hill 4) the Hill returned 0.70 for such a cell: the
# BULK was being read as bare and the tissue reached 751 cells by frame 40 where the reference has
# 203. Scanned over the map itself, sharp 2 / a_sw 0.5 / hill 6 puts the 95th percentile of the bulk's
# own scatter at Hill = 0.0000 and the hole's core at 0.92 -- a 10x growth contrast that fires on 0.7%
# of directions, which is the hole's own solid angle.
def _sense(sharp=2.0):
    """`bm_sense` before `cell_grow`, so the morphogen the growth law reads is this frame's."""
    return [dict(module="bm_sense_ops", after="cell_geometry",
                 op=dict(op="bm_sense", at="vertex", cell_set="cell", map=None,
                         p_ref=1.0, sharp=sharp, chan=0))]


# THE GROWTH CONTRAST IS ONE PARAMETER. `cell_grow` is rate * (rho + Hill(a)): the membrane-covered
# bulk has a = 0 and grows at rate * rho, a cell over the hole has a -> 1 and grows at
# rate * (rho + 1). Holding the bulk at cellfix_B_new's own 0.03 means rate = 0.03 / rho, and the
# contrast is then (1 + rho) / rho -- 11x at rho = 0.1, 21x at rho = 0.05. Compensating the rate is
# what makes this an experiment rather than a slowdown: without it the whole spheroid nearly stops
# and the "bud" is the rest of the tissue failing to grow.
RATE0 = 0.03


def _grow(rho, a_sw=0.50, hill=6.0):
    return {"rho": rho, "rate": round(RATE0 / rho, 5), "a_sw": a_sw, "hill": hill, "chan": 0}


# `max_cycle` IS A BACKSTOP AND cellfix_B_new SETS IT SHORT. mesh_ops says it outright -- "past
# max_cycle, which is a backstop and must be set long or it becomes the rate". At 12 division-calls
# with `every: 4` every cell divides at least every 48 frames whatever its volume, which puts a floor
# under the bulk's proliferation and caps any growth contrast at about 3x however hard the morphogen
# pushes. Raising it lets the gate be the rate, which is the thing being tested.
NO_BACKSTOP = {"max_cycle": 10 ** 9}
# OKUDA'S TUBE MECHANISM. `orient_iface` orients an ACTIVATED cell's septum along the axis from the
# vesicle centre to the activated tip, so daughters STACK into a protrusion instead of spreading it
# flat across the surface. It reads `cell.chem` -- which is exactly why the deficit is written there
# as a morphogen rather than used to gate growth directly -- and `orient_asw` is a fraction of the
# field's own maximum, so 0.5 selects the half of the activated patch nearest the hole.
ORIENT = {"orient_iface": True, "orient_asw": 0.5, "cell_set": "cell"}

VARIANTS = {
    "sense": dict(
        why="the membrane deficit as a morphogen, gating growth through cell_grow's own Hill: "
            "11x where the hole is, the bulk held at its reference rate",
        append_ops=_sense(), op_overrides={"cell_grow": _grow(0.10),
                                           "cell_divide": dict(NO_BACKSTOP)}),
    "sense_orient": dict(
        why="the same, plus Okuda's oriented division: daughters stack along the bud axis instead "
            "of spreading the extra cells flat",
        append_ops=_sense(), op_overrides={"cell_grow": _grow(0.10),
                                           "cell_divide": dict(NO_BACKSTOP, **ORIENT)}),
    "orient_free": dict(
        why="the same, with the radial spring off -- K_R holds the vesicle spherical and a bud has "
            "to fight it",
        append_ops=_sense(), op_overrides={"cell_grow": _grow(0.10),
                                           "cell_divide": dict(NO_BACKSTOP, **ORIENT),
                                           "cell_mechanics": {"K_R": 0.0}}),
    "bud_max": dict(
        why="everything: 21x contrast, oriented division, no radial spring, and r023_15's own "
            "mechanics -- volume-stiff and low line tension, which is what let a neck survive there",
        append_ops=_sense(sharp=3.0),
        op_overrides={"cell_grow": _grow(0.05),
                      "cell_divide": dict(NO_BACKSTOP, **ORIENT),
                      "cell_mechanics": {"K_R": 0.0, "K_V": 20.0, "Lambda": 0.5}}),
    "control": dict(
        why="THE NULL. Identical to bud_max except that the map has its spatial pattern removed, so "
            "every cell reads the same deficit. Any bud that survives this is not the membrane's.",
        flat_map=True, append_ops=_sense(sharp=3.0),
        op_overrides={"cell_grow": _grow(0.05),
                      "cell_divide": dict(NO_BACKSTOP, **ORIENT),
                      "cell_mechanics": {"K_R": 0.0, "K_V": 20.0, "Lambda": 0.5}}),
}


def build_tissue(v, frames, device):
    gate = os.path.join(B.LOG, GATE_SRC, "bm_gate.npz")
    if not os.path.exists(gate):
        raise SystemExit(f"[08b] no ligation map at {gate} -- run phase 1 first")
    if v.get("flat_map"):
        # THE NULL'S MAP IS THE REAL ONE WITH ITS PATTERN REMOVED, not a constant pulled out of the
        # air: same mean, same frames, same normalisation, no spatial structure. So the control run
        # differs from the experiment in exactly one thing.
        import numpy as np
        z = np.load(gate)
        P = np.asarray(z["pmap"], np.float32)
        flat = np.repeat(np.repeat(P.mean(axis=(1, 2))[:, None, None], P.shape[1], 1), P.shape[2], 2)
        gate = os.path.join(B.LOG, GATE_SRC, "bm_gate_flat.npz")
        np.savez_compressed(gate, pmap=flat.astype(np.float32),
                            note=np.str_("the ligation map with its spatial pattern removed"))
        print(f"[08b] NULL: flat map written to {os.path.basename(gate)}", flush=True)
    ap_ = [dict(e, op=dict(e["op"], map=gate)) for e in (v.get("append_ops") or [])]
    kw = dict(frames=frames, device=device, buffer_x=4)
    kw.update(v.get("load_or_build", {}))
    path = TIS.load_or_build(op_overrides=v.get("op_overrides") or None,
                             append_ops=ap_ or None, **kw)
    print(f"[08b] tissue -> {os.path.relpath(path, B.LOG)}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("variant", nargs="?", default=None)
    ap.add_argument("--list", action="store_true")
    a, rest = ap.parse_known_args()
    if a.list or not a.variant:
        for k, v in VARIANTS.items():
            print(f"  {k:16s} {v.get('why', '')}")
        return
    v = VARIANTS[a.variant]
    sys.argv = [sys.argv[0]] + rest

    ap2 = argparse.ArgumentParser(add_help=False)
    ap2.add_argument("--frames", type=int, default=401)
    ap2.add_argument("--device", default="cuda:0")
    b, _ = ap2.parse_known_args()

    cache = build_tissue(v, b.frames, b.device)
    # THE ONE REBINDING, SAID OUT LOUD.
    L.CACHE = cache
    print(f"[08b] RealDriver.CACHE rebound to the pass-2 tissue", flush=True)

    H.build(Rig08a, default_name=f"08b_{a.variant}",
            add_args=lambda ap_: (ap_.add_argument("--spot", type=float, default=SPOT),
                                  ap_.add_argument("--off-theta", dest="off_theta", type=float,
                                                   default=OFF_THETA),
                                  ap_.add_argument("--off-phi", dest="off_phi", type=float,
                                                   default=OFF_PHI)),
            pass_args=("spot", "off_theta", "off_phi"),
            extra=dict(kind="pass 2: growth gated by integrin ligation", phase=2,
                       variant=a.variant, why=v.get("why", ""),
                       gate_src=GATE_SRC, tissue=os.path.basename(cache),
                       changes=repr({k: v[k] for k in ("load_or_build", "op_overrides",
                                                       "append_ops") if v.get(k)}),
                       cap=f"theta {OFF_THETA} deg, phi {OFF_PHI} deg off the camera axis"),
            rho_crit=BR.RHO_CRIT, s_mode="homeostatic",
            kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3,
            K_timp=BR.BASE["K"], hetero=BR.BASE["hetero"],
            s_timp=1.0 * BR.BASE["K"] * (1.0 - BR.BASE["bound"]) / 8.0,
            s_timp3=1.0 * BR.BASE["K"] * BR.BASE["bound"] / 40.0,
            s_mmp=0.0, s_mt1=0.0, k_deg=KDEG, mt1_frac=BR.BASE["mt1_frac"], seed_mt1=3)


if __name__ == "__main__":
    main()
