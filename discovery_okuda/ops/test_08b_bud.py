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

# THE OPERATING POINT IS THE ONE THAT WAS MEASURED, not the one that was predicted. A 23-agent
# design pass read the operator sources and then RAN the arms; three of its tissues are on disk and
# read bud_excess +0.959, +1.117 and +1.191 against a reference tissue's +0.002 and a noise floor of
# +/-0.04 across 80 archived runs. Two numbers differ from the settings this file first carried and
# each was measured alone over 401 frames: `p_ref` 1.0 -> 0.8 with `sharp` 2.0 -> 1.0 (which also
# converts a tapering finger into a constant-calibre tube), and `mesh_seed.vseed_cv` 0.15 -> 0.02.
#
# `vseed_cv` IS THE BODY-QUIESCENCE LEVER AND IT IS NOT THE OBVIOUS ONE. The seed cells keep their
# volume draw for all 401 frames because they rarely divide, so the spread of that draw -- not
# `cell_divide.cycle_cv` -- decides how much of the BODY drifts over the division threshold on its
# own: cycle_cv 0.15 -> 0.03 changed 69 divisions into 67, while vseed_cv 0.15 -> 0.02 cut body
# divisions from 34 to 15. A quiet body is what makes a bud legible.
SENSE_P_REF, SENSE_SHARP = 0.8, 1.0
A_SW, HILL = 0.25, 4.0


def _sense(map_name="bm_gate.npz", p_ref=SENSE_P_REF, sharp=SENSE_SHARP):
    """`bm_sense` immediately before `cell_grow`, so the morphogen is this frame's.

    Inserted after `cell_geometry`, which puts it ahead of `cell_grow`'s Hill AND ahead of
    `cell_divide.orient_iface`'s septum axis -- both read `cell.chem`, which is the whole reason the
    deficit is written as a morphogen rather than used to gate growth directly.
    """
    return [dict(module="bm_sense_ops", after="cell_geometry",
                 op=dict(op="bm_sense", at="vertex", cell_set="cell", map=map_name,
                         p_ref=p_ref, sharp=sharp, chan=0))]


def _grow(rate, rho):
    """`cell_grow` is rate * (rho + Hill(a)): the bulk has a = 0 and grows at rate * rho, a cell over
    the hole has a -> 1 and grows at rate * (rho + 1). `vth_frac` 4.0 keeps the sizer reachable now
    that `max_cycle`'s backstop is gone."""
    return {"rate": rate, "rho": rho, "a_sw": A_SW, "hill": HILL, "chan": 0, "vth_frac": 4.0}


# `max_cycle` IS A BACKSTOP AND cellfix_B_new SETS IT SHORT. mesh_ops says it outright -- "past
# max_cycle, which is a backstop and must be set long or it becomes the rate". At 12 division-calls
# with `every: 4` every cell divides within 48 frames whatever its volume, putting a floor under the
# braked body and capping any contrast at about 3x. `g1_ramp` gives each daughter its ACTUAL birth
# volume as its target instead of half the mother's, which is what stops K_V driving the tiny faces of
# a fast-proliferating tip into inverted caps.
DIVIDE = {"max_cycle": 10 ** 6, "orient_iface": True, "orient_asw": 0.5, "cell_set": "cell",
          "g1_ramp": True}
QUIET = {"vseed_cv": 0.02}

VARIANTS = {
    "s1_finger": dict(
        why="THE MEASURED ARM: the membrane deficit as a morphogen, a quiet body, oriented division. "
            "Both of its edits are on disk as finished tissues reading +0.959 and +1.191",
        append_ops=_sense(),
        op_overrides={"mesh_seed": dict(QUIET), "cell_grow": _grow(0.0055, 0.05),
                      "cell_mechanics": {"K_R": 0.0, "K_V": 20.0, "Lambda": 0.5},
                      "cell_divide": dict(DIVIDE)}),
    "s2_big": dict(
        why="the same mechanism with the body allowed to grow -- body volume x20 over the run, so "
            "the bud is large in ABSOLUTE size and the mesh can resolve its neck",
        buffer_x=6, append_ops=_sense(),
        op_overrides={"mesh_seed": dict(QUIET), "cell_grow": _grow(0.025, 0.10),
                      "cell_mechanics": {"K_R": 0.1, "K_V": 20.0, "Lambda": 0.5},
                      "cell_divide": dict(DIVIDE)}),
    "s3_both": dict(
        why="both readings of the same map at once: the deficit releases the hole and "
            "ecm_gate_growth brakes the intact membrane. The most contrast and the most confounded",
        buffer_x=6, append_ops=_sense(),
        load_or_build=dict(gate_npz="bm_gate.npz", gate_p_half=0.32, gate_hill=12.0,
                           gate_floor=0.5, gate_smooth_frames=5, gate_smooth_phi=15.0),
        op_overrides={"mesh_seed": dict(QUIET), "cell_grow": _grow(0.025, 0.15),
                      "cell_mechanics": {"K_R": 0.1, "K_V": 20.0, "Lambda": 0.5},
                      "cell_divide": dict(DIVIDE)}),
    "s4_myo": dict(
        why="s1 plus junction myosin -- the only localisable line tension in the code -- to pull a "
            "waist and turn a tube into a bud with a real neck",
        append_ops=_sense(),
        load_or_build=dict(myosin=1.0, myo_beta=2.0, myo_tau=20.0, myo_new=1.0,
                           myo_keyed_on="tension", myo_destabilising=1),
        # `Lam`/`Gam` ARE NOT TYPOS. tissue.py hands junction_myosin its mechanics constants under
        # the keys "K_P", "Lam", "Gam" while the spec file carries "Lambda"/"Gamma", so as shipped
        # the myosin tension term reads defaults instead of this run's values. Setting both spellings
        # is a workaround for that mismatch and is verified by reading, not by running -- the log must
        # show junction_myosin instantiating and MYO_SKIPPED must stay empty, or the frame relaxed
        # without myosin and the arm is a silent null.
        op_overrides={"mesh_seed": dict(QUIET), "cell_grow": _grow(0.0055, 0.05),
                      "cell_mechanics": {"K_R": 0.0, "K_V": 20.0, "Lambda": 0.5,
                                         "Lam": 0.5, "Gam": 0.4},
                      "cell_divide": dict(DIVIDE)}),
    "s5_rot180": dict(
        why="THE CONTROL, and a stronger one than a flat map: s1 to the last digit with the hole "
            "moved 180 degrees in longitude. The bud must move with it or it is not the membrane's",
        append_ops=_sense(map_name="bm_gate_phi180.npz"),
        op_overrides={"mesh_seed": dict(QUIET), "cell_grow": _grow(0.0055, 0.05),
                      "cell_mechanics": {"K_R": 0.0, "K_V": 20.0, "Lambda": 0.5},
                      "cell_divide": dict(DIVIDE)}),
}


def build_tissue(v, frames, device):
    """The pass-2 tissue: the reference spec with this variant's overrides, built or reused.

    `load_or_build` hashes op_overrides and append_ops into the cache key, so two variants cannot
    collide on one tissue -- a bug this ladder's history records twice, once where a caps+plane run
    silently loaded the caps-only tissue and reported its semi-axes to three decimals.
    """
    def _m(name):
        f = os.path.join(B.LOG, GATE_SRC, name)
        if not os.path.exists(f):
            raise SystemExit(f"[08b] no map at {f} -- run 08a phase 1 first")
        return f

    ap_ = [dict(e, op=dict(e["op"], map=_m(e["op"]["map"]))) for e in (v.get("append_ops") or [])]
    # THE RESERVOIR IS SIZED PER VARIANT. At the reference buffer `cell_divide` refused 5,546
    # divisions for want of vertex rows -- its own message says "capped by its array, not by its
    # biology -- every later measurement describes the reservoir". A bud measured against a full
    # array is a measurement of the array. This is a memory allocation: it changes what the run is
    # ALLOWED to do, not what it is trying to do. The quiet-body arms need very little (their
    # tissues end at 288-360 cells); the growing-body arms need x6.
    kw = dict(frames=frames, device=device, buffer_x=v.get("buffer_x", 4))
    kw.update(v.get("load_or_build", {}))
    if kw.get("gate_npz"):
        kw["gate_npz"] = _m(kw["gate_npz"])
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
