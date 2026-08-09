#!/usr/bin/env python
"""make_basis -- write the 16 specs the campaign builds from, as a GRID rather than a collection.

Cedric, 8 August: *"put the current 12 folders of log/okuda into archive, you write a new set of 12
specs, run and test them on l4 clusters, to form a sound basis for the agentic loop. Route A can
pull from this basis, not limited to two specs."*

WHY THE OLD SET WAS NOT A BASIS. Sixteen folders from four phases: six different frame counts (201,
301, 401, 501, 801, 901), `grip_peak` absent from all sixteen because they predate the registry the
campaign scores on, one folder with no spec at all, and one (`repair_l_th_frac`, act_cv 4.97) whose
number came from the `mode: tip` artifact. Three were in the parent pool and exactly two could be a
Route A base. A basis has to let you compare its members to each other, and those could not be
compared to each other or to a 900-frame campaign run.

WHY THIS IS GENERATED AND NOT TWELVE HAND-WRITTEN FILES. Twelve hand-written recipes rebuild the
same problem one phase later: each would drift on some parameter nobody meant to vary, and "pull
from the basis" would mean "pick a folder" instead of "choose which axis to move". Here the twelve
ARE the cells of a 4 x 2 x 2, every non-axis parameter is written once, and the diff between any
two members is exactly the axes that separate them.

THE THREE AXES

  chemistry (4)   none | gray_scott | gierer_meinhardt | brusselator
                  `none` is not an empty slot -- it is the control that grows without patterning,
                  which is what `cellfix_B_new` was for. `cell_react` also offers `brusselator`;
                  two RD models plus a null says more than three RD models and no null.

  material (2)    static | uniform | gated -- two per chemistry column, see AXES
                  static: division with NO growth operator; cells subdivide and the body adds
                  nothing. The substrate every growth claim is read against, and the regime the old
                  `factor 1.5/1.8` sweep points fell into by accident.
                  uniform: growth ignores the activator (a_sw = 0, the gate is open everywhere) --
                  material arrives everywhere.
                  gated: growth is opened by the activator through the Hill gate. This is the axis
                  the campaign's whole question sits on, and until round 1 of this campaign nobody
                  had ever swept the knob (`rho`) that sets it.

  mechanics (2)   plain | shaping
                  plain: area/perimeter/volume elasticity only. shaping: plus dihedral bending
                  (K_bend), lumen incompressibility (K_lumen), and a line tension on the red/white
                  interface (rd_interface_tension.K_purse) -- the three terms that can neck a lobe
                  into a finger.

                  ALL TWELVE RUN `shape_energy_3d` WITH ITS DEFAULT MODEL, and that is a change.
                  `coral_gate_div` ran `model: monolayer`, which reads k_v/kappa_s/h0/gamma and has
                  NO K_bend and NO K_lumen in its parameter list at all -- so those terms were not
                  merely missing from its spec, they were inexpressible in its model, and no edit
                  the Proposer could write would have reached them. The default model carries them.
                  The cost is that the coral chemistry was tuned against `monolayer`, so whether the
                  pattern survives on this mechanics is one of the things this batch measures.

                  K_extrude STAYS AT ZERO in all twelve. It is the other half of
                  rd_interface_tension and it multiplies an energy that FALLS as red cells move
                  outward -- a composition carrying it is paid to have a protrusion, which is why
                  `round.parents` sorts it last. K_purse is the sound half: an ordinary line tension.

WHAT "SOUND" MEANS HERE, and it is checkable: every member passes every ACTIVE premise (P1, P2, P4,
P8, P9, P11, P12, P13 -- see crew/flow.yaml `premises.retired`). The one member of the old set that
was really broken, `coral_gate_div_defaultE`, failed P1 with `conserve_amount: true` and a growth
operator running: volume 509.7 -> 507.1, so anything that grew was built by taking material from
somewhere else. Growth here adds material; `conserve_amount` is off wherever growth runs.

    python make_basis.py            write config/okuda/b*.yaml and print the grid
    python make_basis.py --check    also run the static premises on each
"""
import argparse
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(os.path.dirname(HERE), "config", "okuda")

# --------------------------------------------------------------------------- the fixed substrate
# EVERY NON-AXIS VALUE, WRITTEN ONCE. If a number is here, no member of the basis varies it; if a
# member varies it, it is an axis and it is in AXES below. That is the whole invariant.
FRAMES = 900                 # the campaign's own run length, so a base is comparable to its children
SEED_CELLS = 2000
CELL_BUF = 60000             # room to grow: rho=2.0 on the old coral reached 14,196 cells, and the
VERT_BUF = 120000            # array must not be what stops a base. ~1.3 GB of trajectory at worst.
WORLD = 80.0

GROWTH_RATE = 0.000866       # the rate every campaign run has used
# rho MEANS TWO DIFFERENT THINGS DEPENDING ON THE GATE, so there are two values.
#
# Growth is rate * (rho + Hill(activator)). On a GATED base the Hill term is the activator's
# contribution and rho is the baseline every white cell gets anyway, so the tip-to-body growth
# ratio is (rho+1):rho -- rho is the LOCALIZATION knob and low means localized. On an UNGATED base
# (a_sw = 0, or no chemistry at all) the Hill term is 0 and rho is the ONLY growth term, so
# lowering it does not localize anything, it just grows slower.
#
# The first basis used 1.0 everywhere, which puts every gated member at 2:1 -- a red cell grows
# twice as fast as a white one. That is a bias, not localization, and all sixteen basis runs and
# all forty-six runs of rounds 1-4 came back spheres. Cedric, 8 August: "this is exactly the role
# of the morphogen activity, that cells grow locally on red spot??" It is, and it was turned down
# almost off. 0.1 is 11:1, and it centres every Route B run -- which inherits its parent's rho --
# in the regime the campaign is about, instead of leaving that to Route A to find.
RHO_GATED = 0.1              # tip:body = 11:1 on the four *_gated_* members
RHO_UNGATED = 1.0            # the only growth term where there is no gate; not a ratio
HILL = 4.0
VTH_FRAC = 2.5
DIVIDE_FACTOR = 2.0          # OF v_ref, THE SEED-TIME MEDIAN CELL VOLUME, and that reference
                             # CHANGED on 8 August. `divide_3d`'s default model is now `sizer`, an
                             # ABSOLUTE threshold, where it used to be `factor x THIS CELL'S OWN
                             # BIRTH VOLUME`. Ginzberg, Kafri & Kirschner (Science 2015): "both the
                             # cell's target size and its actual size must be evaluated on absolute
                             # rather than relative scales". The old rule was relative, so it
                             # corrected no size deviation at all and this campaign measured the
                             # result -- cell-volume CV 0.160 at seed to 0.53 by frame 900, in
                             # every basis run. It survives as `model: doubler`, the null.
                             #
                             # Must still sit BELOW vth_frac or a cell can never reach the size
                             # that divides it -- the relation that used to be premise P3.

A_SW_GATED = 0.35            # a FRACTION of the activator's own maximum (the absolute version
A_SW_OPEN = 0.0              # selected zero cells in every run of the last campaign)

MECH = dict(K_A=1.0, K_P=1.0, K_V=20.0, K_R=0.0, Lambda=0.5, Gamma=0.4, p0=3.5,
            mu=1.0, dt=1.0, relax_iters=30, eta=0.08, cap_frac=0.12)
# K_V = 20 AND Lambda = 0.5, AND THE FIRST DRAFT OF THIS LINE HAD 2.0 AND 3.0 -- copied from
# `cellfix_B_new`, which grew x22.7 with them. It produced twelve bases carrying a growth operator
# that raises its target and never moves the tissue, and premise P1 is the only thing that caught it.
#
# `grow_3d` writes a WISH (per-cell target volume); shape_energy_3d decides whether to grant it, by
# minimising a sum of competing terms. K_V pays to reach the target volume; Lambda charges for total
# edge length, and inflating a ball stretches every edge on it. cellfix_B_new had 200 cells; these
# have 2000, so each cell is ~12x smaller and the sphere carries ~6000 short edges instead of ~600.
# The same Lambda buys twelve times more resistance while K_V pulls on a twelve-times-smaller volume
# error, so the cheapest shape was to ignore the targets entirely. Measured, same spec, 300 frames:
#
#     K_V 2.0  Lambda 3.0   volume 509 -> 506   P1 BROKEN   (target asked for 753)
#     K_V 8.0  Lambda 1.0   volume 509 -> 507   P1 BROKEN
#     K_V 20.0 Lambda 0.5   volume 509 -> 759   all premises pass
#
# The archived recipes that DID grow at 2000 cells all ran `shape_energy_3d model: monolayer`, whose
# Lambda defaults to 0.0 -- no edge-length bill at all, and k_v 6.0. That model cannot express
# K_bend or K_lumen, which is why the basis is not on it; this is the same balance restated in the
# default model's parameters.
#
# K_R = 0.0 differs from every archived spec, which ran 0.4. K_R is a radial spring to a fixed R0.
# `grow_3d` does update R0 to the radius enclosing the target volume, so K_R would also transmit
# growth -- but measured side by side at 400 frames, K_R 0.0 and 0.4 gave 2018 vs 2019 cells: it is
# not what moves the shell here. A basis whose job is to let a shape leave the sphere should not
# carry a term that pays it to stay.

SHAPING = dict(K_bend=0.02, K_lumen=0.5)     # off (0.0) in the `plain` half
K_PURSE = 1.0                                # rd_interface_tension, `shaping` half only

RD = dict(d_a=0.08, d_h=0.16, chi=1.3)
GS = dict(F=0.046, kk=0.062, rate=1.0)                 # the coral spots the campaign knows
# GM READS A DIFFERENT SET OF KEYS, AND THE FIRST DRAFT OF THIS LINE DID NOT. It said
# `GM = dict(F=0.046, kk=0.062, rate=1.0)` -- "same feed, different model at the slot" -- and
# `CellReactGiererMeinhardt` reads none of F or kk. It reads gm_rho, mu_a, mu_h, a0, rate, sat.
# So four of the twelve shipped carrying two keys their operator ignores and ran at bare defaults
# nobody had validated: in the launch batch all four `b_gm_*` sat at 2000 cells while their
# Gray-Scott twins were at 5,000-6,500 on identical growth and mechanics.
#
# `a0 > 0` IS LOAD-BEARING, not decoration: da = rho*a^2/h - mu_a*a + a0, so da(0) = a0, and at
# a0 = 0 the zero state is a fixed point -- premise P4 says so in as many words and censors the
# run. `sat` caps the autocatalysis; the space's own note records that sat = 0.1 takes the
# activator ceiling from 1489 to a usable range without flattening the pattern (act_cv 4.14 ->
# 2.80). Both come from the declared box in composition_space, which is where this campaign's
# validated values live.
# rate = 0.4, AND THE FIRST LAUNCH USED 1.0, WHICH CRITIC RULE R1d REFUSES ON MEASURED GROUNDS:
# gierer_meinhardt WITH divide_3d must advance dt*rate <= 0.5 or "the activator diverges uniformly
# and the run measures an integrator, not a mechanism". At 1.0 the four gm members reached 53,000
# cells with a PRISTINE mesh -- broken_frac 0.0, ray_single_frac 1.0, reduced_volume 0.97 -- and
# premise P12 broken: the shell was perfect and the chemistry on it was not finite. The rule was
# right and I walked across it; `round.py --check` refused all four as parents before the loop
# could build on them, which is the only reason this is a corrected value and not a campaign.
GM = dict(a0=0.01, mu_h=1.0, sat=0.1, rate=0.4)
# THE THIRD RD MODEL, AND THE CAMPAIGN HAS NEVER RUN IT. Cedric, 8 August: "I think we miss RD
# basis, got to 16 specs, add 4 to study RD mechanisms in route A." `cell_react` registers three
# models and the basis carried two, so "which reaction-diffusion mechanism" was an axis with a
# hole in it -- and an axis the loop cannot vary is an axis it cannot learn anything about.
#
#     da/dt = gamma ( A - (B+1) a + a^2 h )
#     dh/dt = gamma ( B a - a^2 h )
#
# Turing-unstable for B > 1 + A^2, so A = 1, B = 3 sits inside the unstable region (3 > 2) rather
# than on its edge. These are also the values `translate._emit_react` hard-wires, so the basis and
# anything the loop compiles agree. gamma is the reaction rate; `rate`, `F` and `kk` are NOT read
# by this model and are therefore absent -- the unread gate would refuse the spec otherwise, which
# is how the gierer_meinhardt column was caught carrying Gray-Scott's parameters.
# gamma = 0.05 AND THE FIRST LAUNCH USED 0.3, WHICH IS THE SPACE'S OLD DEFAULT AND DIVERGES. All
# four brusselator members came back with act_max = NaN and P12 + P4 + P1 broken, mesh frozen at
# 2000 cells: NaN chemistry makes NaN growth targets and the mechanics stops. `reaction_stiffness`
# had the answer before the runs did -- brusselator's linear decay is gamma*(B+1), so 0.3 gives
# 1.2 against gray_scott's 0.108, and explicit Euler at dt = 1 does not carry it. 0.05 gives 0.2.
BRU = dict(gamma=0.05, A=1.0, B=3.0)
BETA = 0.5                                             # shape -> chemistry, the second arrow
F0 = 0.046

# THE GRID, WRITTEN OUT, because it is not a clean product: `gated` needs an activator to gate on,
# so the `none` column cannot supply it. The twelve are 3 chemistry columns x 2 material x 2
# mechanics where the material pair differs by column -- {static, uniform} with no chemistry,
# {uniform, gated} with it. Every column still contributes four, and every row still differs from
# its neighbour in exactly one axis.
AXES = {
    "chem": ["none", "gs", "gm", "bru"],
    "mat":  {"none": ["static", "uniform"], "gs": ["uniform", "gated"],
             "gm": ["uniform", "gated"], "bru": ["uniform", "gated"]},
    "mech": ["plain", "shaping"],
}


def _name(chem, mat, mech):
    return f"b_{chem}_{mat}_{mech}"


def build(chem, mat, mech):
    """One cell of the grid -> a full spec dict."""
    has_chem = chem != "none"
    gated = mat == "gated"
    grows = mat != "static"
    shaping = mech == "shaping"

    if gated and not has_chem:
        raise ValueError("a gate with nothing to gate on")

    ops = [
        {"op": "seed_mesh_3d", "at": "vertex", "cell_set": "cell", "before_frame": 1,
         "n_cells": SEED_CELLS, "seed": 0, "vseed_cv": 0.15, "radius": 5.0, "jitter": 0.15,
         "p0": 3.5},
        {"op": "cell_geometry_3d", "at": "cell"},
    ]
    if has_chem:
        ops += [
            {"op": "cell_adjacency", "at": "cell"},
            {"op": "seed_cell_rd", "at": "cell", "seed": 0, "before_frame": 3,
             "mode": "scatter", "seed_frac": 0.06},
            {"op": "cell_diffuse", "at": "cell", "implementation": "graph_laplacian", **RD},
            {"op": "cell_react", "at": "cell",
             "model": {"gs": "gray_scott", "gm": "gierer_meinhardt",
                       "bru": "brusselator"}[chem],
             **{"gs": GS, "gm": GM, "bru": BRU}[chem]},
            # THE SECOND ARROW. shape_to_chem makes a bulging cell feed faster, so the loop closes:
            # curvature -> feed -> activator -> growth gate -> curvature. Present in every
            # chemistry member, because `beta = 0` as the null is what `coral_gate` was for and the
            # `none` column is a stronger null than a zeroed coefficient.
            {"op": "shape_to_chem", "at": "cell", "model": "curvature", "vertex_set": "vertex",
             "beta": BETA, "F0": F0, "rate": 1.0},
        ]
    if grows:
        ops.append(
            {"op": "grow_3d", "at": "vertex", "cell_set": "cell",
             "rate": GROWTH_RATE, "a_sw": A_SW_GATED if gated else A_SW_OPEN, "hill": HILL,
             "rho": (RHO_GATED if gated else RHO_UNGATED), "vth_frac": VTH_FRAC, "after_frame": 100,
             # P1: growth ADDS material here. conserve_amount redistributes it, which is what made
             # coral_gate_div_defaultE fail P1 with a growth operator running.
             "conserve_amount": False})
    ops += [
        {"op": "shape_energy_3d", "at": "vertex", **MECH,
         **(SHAPING if shaping else {"K_bend": 0.0, "K_lumen": 0.0})},
    ]
    if shaping and has_chem:
        # a_sw IS NOT OPTIONAL HERE. Omitting it took the operator's default, which was 1.0 --
        # "cells above 100% of the maximum" -- so the purse-string never fired in a single run of
        # the basis or of nineteen campaign rounds, and Route A's K_purse ladder measured nothing
        # four times over. A fraction of the activator's own maximum: 0.6 is the top 40%.
        ops.append({"op": "rd_interface_tension", "at": "vertex", "cell_set": "cell",
                    "K_purse": K_PURSE, "K_extrude": 0.0, "a_sw": 0.6})
    ops += [
        {"op": "divide_3d", "at": "vertex", "cell_set": "cell", "factor": DIVIDE_FACTOR,
         # `reset_noise` IS NOT HERE, and the first draft copied it from coral_gate_div. It was
         # the legacy uniform jitter, read only when cycle_cv == 0, and it was REMOVED from
         # divide_3d on 6 August -- so every spec still carrying it names a key nothing reads.
         # Caught by the unread gate below, which is the entire reason that gate exists.
         "cycle_cv": 0.15, "min_cycle": 4, "max_cycle": 10 ** 9, "p0": 3.5,
         "every": 4, "engine_clock": True},
        {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.28, "every": 1, "max_flips": 300},
        {"op": "topo_snapshot_3d", "at": "vertex", "every": 1},
    ]

    name = _name(chem, mat, mech)
    return {
        "general": {"name": name, "seed": 0, "n_frames": FRAMES, "dt": 1.0,
                    "record_cap": FRAMES + 2, "record_every": 1, "boundary": "free", "dim": 3,
                    "world": [WORLD, WORLD, WORLD]},
        "_run": {"target_cells": CELL_BUF, "seed_cells": SEED_CELLS},
        "sets": {"vertex": {"n": VERT_BUF},
                 "cell": {"n": CELL_BUF,
                          "state": {"chem": {"width": 2, "integration": "first_order"},
                                    "cen": {"width": 3}, "area": {"width": 1}}}},
        "fields": {},
        "operators": ops,
        "schedule": [o["op"] for o in ops],
        "_basis": {"chem": chem, "mat": mat, "mech": mech,
                   "grid": "4 chemistry x 2 material x 2 mechanics",
                   "why": f"chemistry={chem}, material={mat}, mechanics={mech} -- one cell of the "
                          f"basis grid written by make_basis.py; every value not on an axis is "
                          f"shared with the other fifteen."},
    }


def grid():
    """The sixteen, in a fixed order: four per chemistry column.

    `static` is DIVISION WITHOUT GROWTH -- cells subdivide and the body adds nothing. It is the
    substrate every growth claim is read against, and it is the regime the old `factor 1.5/1.8`
    sweep points fell into by accident; here it is a declared member rather than an accident."""
    return [(chem, mat, mech)
            for chem in AXES["chem"]
            for mat in AXES["mat"][chem]
            for mech in AXES["mech"]]


def _unread(spec):
    """Keys each operator in `spec` never looks up -- reported as `op.key UNREAD`."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "prototype", "Tyssue"))
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
    try:
        import plexus.operators                                          # noqa: F401
        import tyssue_ops3d, tyssue_rd_ops, tyssue_monolayer             # noqa: F401
        import tyssue_shape_to_chem, tyssue_t1_ops3d                     # noqa: F401
        from op_probe import unread_params
    except Exception as e:
        return [f"(unread check unavailable: {type(e).__name__})"]
    out = []
    for o in spec["operators"]:
        params = {k: v for k, v in o.items()
                  # the engine consumes these; the operator class never sees them as its own
                  if k not in ("op", "model", "implementation", "every", "connect",
                               "after_frame", "before_frame", "reads", "writes")}
        params["_at"] = o.get("at")
        try:
            dead, err = unread_params(o["op"], params, o.get("model") or o.get("implementation"))
        except Exception:
            continue
        if err or dead is None:
            continue
        for k in dead:
            if k not in ("at", "cell_set", "vertex_set", "seed"):   # resolved by the engine
                out.append(f"{o['op']}.{k} UNREAD")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run the static premises on each")
    a = ap.parse_args()

    cells = grid()
    os.makedirs(CONFIG, exist_ok=True)
    print(f"{'name':<26}{'chem':<6}{'mat':<9}{'mech':<9}{'ops':>4}  {'premises'}")
    bad = 0
    for chem, mat, mech in cells:
        spec = build(chem, mat, mech)
        name = spec["general"]["name"]
        path = os.path.join(CONFIG, f"{name}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
        note = ""
        if a.check:
            sys.path.insert(0, HERE)
            import biologist as B
            res = B.check(spec)
            fails = [r.pid for r in res if r.status == "fail"]
            # AND EVERY KEY MUST BE ONE THE OPERATOR ACTUALLY READS. This gate exists because the
            # first launch of this basis shipped four members carrying `F` and `kk` on a
            # gierer_meinhardt slot, which that model does not read: it takes gm_rho, mu_a, mu_h,
            # a0, rate, sat. Nothing complained. The premises passed, the specs compiled, the jobs
            # ran -- and a third of the basis was measuring an unvalidated default instead of the
            # recipe its name claims. A wrong VALUE is a result; a key nobody reads is a lie about
            # what was tested, and it is statically detectable, so it is checked here.
            fails += _unread(spec)
            bad += bool(fails)
            note = "BROKEN " + ",".join(fails) if fails else "static ok"
        print(f"{name:<26}{chem:<6}{mat:<9}{mech:<9}{len(spec['operators']):>4}  {note}")
    print(f"\n{len(cells)} specs -> {CONFIG}")
    if a.check and bad:
        print(f"{bad} SPEC(S) BROKEN before the GPU")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
