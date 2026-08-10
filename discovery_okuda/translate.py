"""translate -- CompositionGraph  <->  runnable Plexus spec (the Tyssue 3D-AVM backend).

The composition space is substrate-agnostic; this file is the ONLY place that knows how an
abstract mechanism graph becomes a spec the engine can run. Two directions:

    to_spec(graph, ...)   -> cfg dict  (config/okuda/<name>.yaml)
    from_preset(name)     -> CompositionGraph   -- proves a known-good hand recipe is EXPRESSIBLE

`from_preset` is the validation gate. Before the campaign may run, the space must be able to
express the recipes we already trust; if it cannot, the search is exploring a different space
from the one our evidence is about.

--------------------------------------------------------------------------------------------
DEFECT FIXES APPLIED AT TRANSLATION -- every generated config is born correct
--------------------------------------------------------------------------------------------
D1  clock double-gating.  The engine gates operators AND several operators kept a private
    `every`, so the effective period was every^2 (a `divide_3d every=2` fired every 4). We emit
    every=1 and let the engine own the clock. NOTE: the operator-side private counters must ALSO
    be deleted in prototype/Tyssue -- this fix is necessary but not sufficient on its own.

D2  composition-dependent dt.  run_tyssue_round.make() sets
        dt = 1.0 if (cones and not rd) else 0.02
    i.e. adding/removing the RD operators -- the campaign's single most important edit -- ALSO
    rescaled chemical:mechanical time by 50x. Every signalling verdict would be confounded.
    We emit ONE global dt for the whole campaign.

D3  recording-stride mismatch.  Vertex positions and mesh topology were recorded on different
    strides; the analysis then paired one frame's coordinates with another frame's connectivity
    and reported phantom inverted cells. We pin topo_snapshot_3d every=1 and record_cap so both
    series have equal length, and the runner asserts it.
"""
from __future__ import annotations

import copy
import json
import os

import yaml

from composition_space import (DIVIDE_CALL_PERIOD_BEFORE_D1, OPERATORS,
                               CompositionGraph, seed)

HERE = os.path.dirname(os.path.abspath(__file__))
TYSSUE = os.path.abspath(os.path.join(HERE, "..", "prototype", "Tyssue"))
# REPO-RELATIVE. The devcontainer mounts the NFS export at /workspace and the cluster mounts the
# SAME export at /groups/saalfeld/home/allierc/Graph, so an absolute path baked into a TRACKED
# config is portable to exactly one of the two. run_one.py resolves this against its own location.
CKPT = os.path.join("prototype", "Tyssue", "archive", "smoke_hom", "ckpt.npz")

# THE RESERVOIRS ARE NO LONGER CONSTANTS. They are derived from the cell count the run is aiming
# at, because a closed sheet is trivalent and Euler fixes V = 2F - 4: a vertex reservoir of size V
# caps the cells at (V+4)/2 whatever the biology wants. 3552 vertices give exactly 1778 cells --
# which is what all 32 runs of the overnight study reported as a finding, and what all 27 runs of
# the weekend battery reported as evidence. Twice in one week, from the same arithmetic.
#
# `VBUF, CBUF = 30000, 16000` sat here as a pair of magic numbers. They were generous enough that
# nobody noticed they were a ceiling (15002 cells) rather than a size -- and they did not help the
# hand-written configs, which carried their own smaller pair. The destination now sets the size,
# and `grounder.buffer_for` is the single place that arithmetic lives.
from agents.grounder import buffer_for, max_cells_for                       # noqa: E402

VBUF_FALLBACK, CBUF_FALLBACK = 30000, 16000    # only when no target can be inferred


# Shared with round.MAX_CELLS -- one number, one env override, both sizing paths.
MAX_CELLS = int(os.environ.get("OKUDA_MAX_CELLS", 50_000))


def _reservoirs(n_cells_seed, frames, growth_headroom=40.0):
    """How big must the reservoirs be for a run that STARTS at n_cells_seed?

    A growing vesicle roughly doubles every cell cycle, so the destination is the seed times
    however many doublings the run has time for. `growth_headroom` is deliberately generous: the
    cost of an oversized reservoir is memory, and the cost of an undersized one is a batch of
    runs that measure the array. We have now paid the second cost three times.

    RAISED FROM 8 TO 40, and the arithmetic says it should have been done long ago. Measured on
    an A6000 (49 GB), for a 2000-cell seed over 900 frames:

        headroom  8   cap  20,804 cells   0.45 GB of recorded trajectory
        headroom 40   cap 104,004 cells   2.25 GB

    The instantaneous state is 0.4 MB either way -- a rounding error. The only cost that scales is
    the trajectory, it scales linearly, and 2.25 GB against 49 GB is not a constraint. Against
    that: `wk_pressure_pos_s0` grew 150 -> 1778 cells by frame 323 of 900 and then added ZERO for
    the remaining 575 frames, because 1778 is exactly the (V+4)/2 cap of a buffer sized for a
    150-cell start. Two thirds of that run measured a full array. Cedric saw it as division
    stopping two seconds into a six-second movie.

    THIS DOES NOT HELP A RECON REPLAY. Those re-run a spec VERBATIM, stale reservoir included --
    that is what verbatim means, and rewriting the buffer would make the replay a different
    experiment. It applies to every composition the loop builds from here.
    """
    # THE SAME CEILING AS THE RECON PATH. Two paths sized buffers -- this one for compositions
    # the loop builds, _resize_reservoir for verbatim replays -- and a cap applied to only one of
    # them is not a cap. A 50k body is not more informative than a 20k one for a map looking for
    # a BUD, and the cost is superlinear while the information is not.
    target = min(max(int(n_cells_seed) * growth_headroom, 2000), MAX_CELLS)
    b = buffer_for(target)
    return b["vertex"], b["cell"], int(target)
# ONE dt for the whole campaign, never composition-dependent (D2) -- and its value is 1.0,
# because that is the only value any run that ever produced a pattern has used.
#
# D2 set it to 0.02 and nothing that patterned was ever re-run to check. coral_fixed_ball and
# wk_null_s0 -- spatial spread 0.78, stable to the last frame -- carry general.dt 1.0, mech.dt 1.0,
# relax_iters 30, rate 1.0, chi 1.3. At 0.02 the frame clock and the mechanics clock both moved,
# and the two guards that bound the reaction were set against each other:
#
#   Biologist P5   the reaction must advance ~1 time unit per frame. At dt=0.02 it advances
#                  dt*rate = 0.02, so 900 frames buy 18 time units where a Gray-Scott pattern
#                  needs ~500. "No pattern formed" would be a statement about the clock.
#   Critic R1c     the reaction must not advance more than 2.0 per frame. The 1/dt scaling that
#                  satisfies P5 at dt=0.02 makes it 65, and the chemistry goes non-finite.
#
# Both are right. Neither can be satisfied at dt=0.02, and both are satisfied at 1.0: advance 1.3,
# CFL 0.208. The contradiction was never between the guards -- it was a dt nothing had validated.
DT_GLOBAL = 1.0

# D5  THE CHEMISTRY RAN 50x TOO SLOW, AND THE CELLS COULD NEVER DIVIDE.
# ------------------------------------------------------------------------------------------------
# dt=0.02 is the MECHANICS substep -- the vesicle relaxes toward force balance many times per
# biological event. But cell_react and cell_diffuse both EMIT=velocity into `chem`, so the engine
# integrated the chemistry with that same dt: 300 frames bought 300*0.02 = 6 units of Gray-Scott
# time, against the ~500 the validated minisite spec (dt=1.0, 500 frames) needed. The activator sat
# at its seed value for the whole run and every measured "no pattern" was an artefact of the clock.
# Scaling reaction AND diffusion together by 1/dt restores one minisite time unit per frame and
# leaves their RATIO -- the thing that actually selects the Turing wavelength -- untouched.
# CFL: chi*max(d_a,d_h) = 1.3*0.16 = 0.21, against a limit of 0.45.
# MEASURED 2026-08-01, and this factor is now 1.0. The reasoning that made it 1/dt was right
# about the disease and wrong about the cure.
#
# cell_react and cell_diffuse EMIT a velocity, so the engine integrates them on the MECHANICS
# SUBSTEP rather than once per frame -- that was D5a, and it was real. The fix multiplied chi and
# rate by 1/dt to restore one time unit per frame. But there are 1/dt SUBSTEPS in a frame, so the
# substep clock ALREADY supplies that factor: advancing chi per substep across 1/dt substeps of
# length dt advances chi per frame, with no scaling at all. Applied on top, the reaction ran 1/dt
# = 50x too fast.
#
# It was invisible while DT_GLOBAL was 1.0, where the two are equal. The D2 reform set one global
# dt of 0.02 for the mechanics, and the reaction went with it:
#
#   chi_spec 1.3 (dt 1.0)   coral_fixed_ball, wk_null_s0   pattern, spatial spread 0.78, stable
#   chi_spec 65  (dt 0.02)  every round-1 run              act 0.01 -> 12.1 -> 1.41e6 -> NaN by
#                                                          frame 115, SPATIALLY UNIFORM
#
# Same d_a, same d_h, same ratio. A uniform blow-up is an ODE exploding; a diffusion breach would
# have made a checkerboard. critic.R1c_REACTION_UNSTABLE now bounds this before a run costs
# anything, and composition_space.reaction_advance is the quantity it bounds.
RD_PER_FRAME = 1.0 / DT_GLOBAL

# The growth ceiling must sit ABOVE the division trigger. `grow_3d` caps each cell's
# target volume at vth_frac*v_ref, while `divide_3d` fires at factor*Vbirth -- and vth_frac was
# 1.5 against factor 2.0, so a cell's target could never reach the size that makes it divide.
# Volume-triggered division was arithmetically impossible; the only divisions ever seen came from
# the max_cycle timeout. Deriving the ceiling from the trigger keeps them from drifting apart.
DIV_FACTOR = 2.0
GROWTH_CEILING = DIV_FACTOR * 1.25    # 25% headroom so cells cross the trigger, not asymptote to it

# The engine executes the schedule in this order regardless of graph insertion order: readouts
# first, then patterning, then growth, then mechanics, then topology, then recording.
SCHEDULE_ORDER = [
    "load_mesh_3d", "seed_mesh_3d", "cell_geometry_3d",
    "seed_cell_rd", "cell_adjacency", "cell_diffuse", "cell_react", "shape_to_chem",
    # DEATH BEFORE THE MECHANICS. Growth sets the targets, death overrides them for the cells it
    # has sentenced, and the relaxation then sees both -- so a cell extruded this tick has its hole
    # closed this tick instead of leaving a raw gap for one frame. This is the order the dedicated
    # geometry tests certify (make_apop_geo.py: euler 2 at every frame, n_apop exactly the loss).
    "grow_3d", "apoptosis_3d", "shape_energy_3d", "interface_line_tension_3d",
    "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d",
]

# composition-space operator -> engine operator name
ENGINE_NAME = {
    "seed_mesh_3d": None,             # resolved by implementation (see _emit_seed)
    "shape_energy_3d": "shape_energy_3d",
    "reconnect_t1_3d": "reconnect_t1_3d",
    "grow_3d": "grow_3d",
    "divide_3d": "divide_3d",
    "apoptosis_3d": "apoptosis_3d",
    "cell_geometry_3d": "cell_geometry_3d",
    "cell_adjacency": "cell_adjacency",
    "cell_diffuse": "cell_diffuse",
    "cell_react": "cell_react",
    "seed_cell_rd": "seed_cell_rd",
    "shape_to_chem": "shape_to_chem",
    "interface_line_tension_3d": "interface_line_tension_3d",
}


def _p(graph, node_id, name, default=None):
    """theta lookup: '<node_id>.<param>' with the vocabulary default as fallback."""
    key = f"{node_id}.{name}"
    if key in graph.params:
        return graph.params[key]
    op = next(o["op"] for o in graph.ops if o["id"] == node_id)
    spec = OPERATORS[op]["params"].get(name)
    return spec[2] if spec else default


# --------------------------------------------------------------------------- per-operator emit
def _emit_seed(g, n, ga):
    impl = g.impl_of(n)
    if impl == "checkpoint":
        return {"op": "load_mesh_3d", "at": "vertex", "cell_set": "cell",
                "ckpt": CKPT, "before_frame": 1}
    # `before_frame: 1` is MANDATORY: without it seed_mesh_3d rebuilds the sphere every tick,
    # wiping `_mesh` -- and `hist` lives inside `_mesh`, so the topology history is destroyed and
    # the D3 alignment assertion fires. (Found by that assertion on the first real run.)
    return {"op": "seed_mesh_3d", "at": "vertex", "cell_set": "cell", "before_frame": 1,
            "n_cells": int(_p(g, n["id"], "n_cells")),
            "radius": float(_p(g, n["id"], "radius")),
            "jitter": float(_p(g, n["id"], "jitter")),
            "seed": SEED_SENTINEL,          # filled from general.seed -- see _seed_the_run
            "vseed_cv": float(_p(g, n["id"], "vseed_cv"))}


def _emit_shape_energy(g, n, ga):
    i = n["id"]
    if g.impl_of(n) == "monolayer":
        return {"op": "shape_energy_3d", "model": "monolayer", "at": "vertex",
                "k_v": float(_p(g, i, "K_V")), "kappa_s": float(_p(g, i, "kappa_s")),
                "h0": float(_p(g, i, "h0")), "gamma": float(_p(g, i, "gamma")),
                "mu": float(_p(g, i, "mu")), "dt": DT_GLOBAL,
                "relax_iters": int(_p(g, i, "relax_iters")),
                "K_bend": float(_p(g, i, "K_bend")), "K_lumen": float(_p(g, i, "K_lumen")),
                "eta": 0.08, "cap_frac": 0.12}          # eta/cap_frac: solver numerics, not knobs
    return {"op": "shape_energy_3d", "at": "vertex",
            "p0": float(_p(g, i, "p0")),
            "K_A": float(_p(g, i, "K_A")), "K_P": float(_p(g, i, "K_P")),
            "Gamma": float(_p(g, i, "Gamma")), "Lambda": float(_p(g, i, "Lambda")),
            "K_V": float(_p(g, i, "K_V")), "K_R": float(_p(g, i, "K_R")),
            "K_bend": float(_p(g, i, "K_bend")), "K_lumen": float(_p(g, i, "K_lumen")),
            "antiinv": 0.0,
            "mu": float(_p(g, i, "mu")), "dt": DT_GLOBAL,
            "relax_iters": int(_p(g, i, "relax_iters")),
            "eta": 0.08, "cap_frac": 0.12}


def _emit_rd_seed(g, n, ga):
    i, impl = n["id"], g.impl_of(n)
    # the engine's mode name for a fixed-angle cone is "cones" (plural). Emitting "cone" here
    # silently fell through to a different seeding mode -- caught by the V9 parameter check.
    # `tip` REMOVED 6 August -- see seed_cell_rd's docstring. `amp` removed with it: the operator
    # reads no such parameter in any mode, so every spec this campaign wrote carried a number that
    # nothing looked at. Found by the UNREAD probe in one second, having survived 14 rounds.
    ENGINE_MODE = {"cone": "cones", "spot": "cones", "scatter": "scatter"}
    d = {"op": "seed_cell_rd", "at": "cell", "mode": ENGINE_MODE[impl],
         "n_spots": int(_p(g, i, "n_spots"))}
    if impl == "scatter":
        # random seeds over the whole shell -- the validated minisite condition. n_spots/amp are
        # meaningless here, so do not emit them: an ignored parameter in a spec reads as if it
        # were doing something.
        # `before_frame: 1` IS THE WHOLE POINT OF A SCATTER SEED. It is an INITIAL CONDITION, and
        # without the guard the engine re-applies it on every tick -- and seed_cell_rd sits BEFORE
        # cell_diffuse and cell_react in the schedule, so each frame went: overwrite the chemistry
        # with the seed, let the reaction advance it by one step, overwrite it again. The activator
        # was pinned for the entire campaign. The signature in the record is unmistakable once you
        # look: the acted-ledger shows seed_cell_rd firing 501 times in a 500-frame run, and
        # act_max varies by 1.2e-3 across the whole run -- exactly one step of Gray-Scott.
        # `cone` maintains a source and is not an initial condition either. This one is.
        # (That note used to read "`tip` re-seeds every frame ON PURPOSE" -- the defect was
        # diagnosed here, fixed for `scatter`, and left standing for `tip`, which is the mode the
        # whole campaign then ran on.)
        d = {"op": "seed_cell_rd", "at": "cell", "mode": "scatter",
             "seed_frac": float(_p(g, i, "seed_frac")), "before_frame": 3}
    elif impl == "cone":
        d["cone_deg"] = float(_p(g, i, "cone_deg"))
    else:                                                     # a frozen spot -- the DOME control
        d["cone_deg"] = float(_p(g, i, "cone_deg"))
        d["before_frame"] = 3
    return d


def _emit_react(g, n, ga):
    i, impl = n["id"], g.impl_of(n)
    base = {"op": "cell_react", "at": "cell", "model": impl,
            "rate": float(_p(g, i, "rate")) * RD_PER_FRAME}   # D5: physical time, not substeps
    if impl == "gierer_meinhardt":
        # `sat` MUST BE EMITTED OR IT DOES NOT EXIST. It was added to OPERATORS' tunable table and
        # to okuda_route's parameters, and neither had any effect: this emitter is what writes the
        # spec, and it did not pass the key. Measured after that "fix": r002c_11 still reported
        # act_max_peak = 9.51e5, the identical unsaturated value. A parameter set in the search
        # space and dropped by the translator is a parameter that only exists in the record.
        #
        # Meinhardt's saturation kappa in a^2/(h(1 + kappa a^2)). At 0 the autocatalysis is
        # unbounded, which refused 10 of 15 runs in the previous campaign as
        # P3_CHEMISTRY_DIVERGED; 0.1 caps the activator at 1489 with alive_frac 0.375.
        base.update({"sat": float(_p(g, i, "sat")),
                     "gm_rho": 1.0, "mu_a": 1.0, "mu_h": float(_p(g, i, "mu_h")),
                     "a0": float(_p(g, i, "a0"))})
    elif impl == "gray_scott":
        base.update({"F": float(_p(g, i, "F")), "kk": float(_p(g, i, "kk"))})
    else:
        base.update({"gamma": float(_p(g, i, "gamma")), "A": 1.0, "B": 3.0})
    return base


def _emit_growth(g, n, ga):
    i = n["id"]
    return {"op": "grow_3d", "at": "vertex", "cell_set": "cell",
            "rate": float(_p(g, i, "rate")), "a_sw": float(_p(g, i, "a_sw")),
            "hill": float(_p(g, i, "hill")), "rho": float(_p(g, i, "rho")),
            # `dt` REMOVED 6 August: Grow3D has no `self.dt` and never looks the key up.
            # It is a leftover from before D1 gave the engine the clock (`_engine_owns_clock`), and
            # a parameter in a spec reads as if it does something -- which is the whole failure this
            # phase is about. Found by op_probe's UNREAD probe in under a second.
            "vth_frac": GROWTH_CEILING, "after_frame": ga,
            "conserve_amount": g.impl_of(n) == "hill_conserve_amount"}


def _emit_divide(g, n, ga):
    i = n["id"]
    return {"op": "divide_3d", "at": "vertex", "factor": DIV_FACTOR, "reset_noise": 0.12,
            "cycle_cv": float(_p(g, i, "cycle_cv")), "p0": 3.90,
            "every": 1,                                        # D1: the ENGINE owns the clock
            # max_div is ALSO per-call, and cap_div = max(max_div, max_div_frac*nF) makes it a
            # FLOOR that DOMINATES at realistic cell counts (at nF=1431: max(120, 42) = 120), so
            # rescaling max_div_frac alone was ENTIRELY MASKED. 120/4 = 30 restores the archived
            # per-frame budget exactly at every scale.
            "cell_set": "cell",
            "min_cycle": int(_p(g, i, "min_cycle")), "max_cycle": int(_p(g, i, "max_cycle")),
            "after_frame": ga, "orient_iface": g.impl_of(n) == "orient_iface",
            "orient_asw": float(_p(g, i, "orient_asw")), "g1_ramp": False}


def _emit_interface_tension(g, n, ga):
    i = n["id"]
    # NO `K_extrude`. The forcing is a separate operator (`extrusion_forcing_3d`) that this
    # vocabulary does not contain, so there is no key here to set and nothing to default to zero.
    return {"op": "interface_line_tension_3d", "at": "vertex", "cell_set": "cell",
            "K_purse": float(_p(g, i, "K_purse")),
            "a_sw": float(_p(g, i, "a_sw")), "eta": 0.05, "iters": 4, "after_frame": ga}


def _emit_apoptosis(g, n, ga):
    i = n["id"]
    # `mode` IS NOT A COMPOSITION-SPACE PARAMETER, so it is not read through _p: the twelve modes
    # are `model` variants, and Route A sweeps them from the plan the way it sweeps divide_3d's
    # sizer/adder/timer. The operator's own default (`competition`) is what an `add_op` writes, and
    # it is a firing rule on purpose -- the previous default, `list` with no `cells`, could never
    # fire, and an operator that is inert by construction is the failure this campaign exists to
    # eliminate.
    #
    # `p0` MATCHES THE MECHANICS, not the operator's own default. A dying cell's target perimeter
    # is rebuilt from its shrinking volume through the shape index, so a p0 that disagrees with
    # shape_energy_3d's would have the two operators pulling the same cell toward two different
    # shapes.
    p0 = next((float(_p(g, m["id"], "p0")) for m in g.ops if m["op"] == "shape_energy_3d"), 3.72)
    return {"op": "apoptosis_3d", "at": "vertex", "cell_set": "cell", "p0": p0,
            "max_mark_frac": float(_p(g, i, "max_mark_frac")),
            "min_age": int(_p(g, i, "min_age")),
            "shrink_rate": float(_p(g, i, "shrink_rate")),
            "critical_frac": float(_p(g, i, "critical_frac")),
            "stall_frac": float(_p(g, i, "stall_frac")),
            "every": 1, "after_frame": ga}                       # D1: the ENGINE owns the clock


EMIT = {
    "seed_mesh_3d": _emit_seed,
    "apoptosis_3d": _emit_apoptosis,
    "shape_energy_3d": _emit_shape_energy,
    "seed_cell_rd": _emit_rd_seed,
    "cell_react": _emit_react,
    "grow_3d": _emit_growth,
    "divide_3d": _emit_divide,
    "interface_line_tension_3d": _emit_interface_tension,
    "reconnect_t1_3d": lambda g, n, ga: {
        "op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": float(_p(g, n["id"], "l_th_frac")) * 7.0,
        "every": 1, "max_flips": 300},                          # D1
    "cell_geometry_3d": lambda g, n, ga: {"op": "cell_geometry_3d", "at": "cell"},
    "cell_adjacency": lambda g, n, ga: {"op": "cell_adjacency", "at": "cell"},
    # `implementation` MUST be emitted. Without it the spec silently runs the default
    # (graph_laplacian) while the composition hash records `interface_weighted` -- so the search
    # would log a DISTINCT hypothesis that is byte-identical to its control. That is exactly the
    # "silent no-op recorded as evidence" failure this campaign exists to eliminate, and it would
    # have made the shape-to-chemistry ABLATION -- the whole reason the operator was written --
    # report "no effect" while never once running the new code.
    "cell_diffuse": lambda g, n, ga: {
        "op": "cell_diffuse", "at": "cell", "implementation": g.impl_of(n),
        "d_a": float(_p(g, n["id"], "d_a")),
        "d_h": float(_p(g, n["id"], "d_h")),
        "chi": float(_p(g, n["id"], "chi")) * RD_PER_FRAME},        # D5: scaled WITH the reaction
    "shape_to_chem": lambda g, n, ga: {
        "op": "shape_to_chem", "at": "cell", "model": g.impl_of(n),
        "vertex_set": "vertex",
        "beta": float(_p(g, n["id"], "beta")), "F0": float(_p(g, n["id"], "F0")),
        # the feedback is chemistry and must run on the SAME clock as the reaction it modulates,
        # or beta would mean something different at every dt (defect D5a, one more time)
        "rate": RD_PER_FRAME},
}


# --------------------------------------------------------------------------- graph -> spec

# Params the emitter renames on the way out, so "set but not emitted" is not a lie about them.
# EMPTY, AND IT STAYS EMPTY. This mapped four graph-side names onto four engine-side names --
# rd_rate/rate, alpha/hill, mono_gamma/gamma, l_th/l_th_frac -- and every consumer that did not
# consult it silently read None. That is not a translation problem, it is TWO NAMES FOR ONE THING,
# and the alias table was the workaround rather than the fix. R1d asked for `rate`, `cell_react`
# declared `rd_rate`, and so it refused every gierer_meinhardt-with-division composition
# unconditionally for weeks while printing "rate None".
#
# The declarations now use the ENGINE's names, because the engine is what actually runs. An entry
# here would reintroduce the disease; if a rename is ever needed, rename.
_ALIASED = {}


def _assert_params_consumed(graph, node, spec_op):
    """A parameter that was SET and is not passed to the engine must not be silently dropped.

    THE DEFECT THIS CLOSES, found by the pre-launch on 1 August. `cell_react` accepts `F` and
    `kk` -- they are Gray-Scott's parameters -- and the emitter forwards them only when the
    implementation IS Gray-Scott. Setting them on a Gierer-Meinhardt node was accepted by the
    vocabulary, accepted by the Critic, recorded in the composition, and then discarded at
    translation with nothing said. The run that followed produced NaN chemistry, and the
    composition on record claims a calibration that never reached the engine.

    That is the same family as the inert-operator defect (D4): the difference between "this
    setting did not work" and "this setting was never applied" is the difference between a result
    and a fiction, and only the second one is silent. An operator whose implementation cannot
    consume a parameter must REFUSE it, not ignore it.
    """
    from composition_space import OPERATORS
    nid = node["id"]
    defaults = OPERATORS.get(node["op"], {}).get("params", {})
    # Only a DELIBERATE setting can be a lie. `graph.params` also carries vocabulary defaults,
    # and a default sitting unused on an implementation that ignores it is harmless noise --
    # `cell_react` offers Gray-Scott's F and Gierer-Meinhardt's mu_h from one contract, and
    # whichever implementation is chosen leaves the other's defaults untouched. What must never
    # pass is a value someone CHANGED and the engine never saw.
    set_here = set()
    for k, v in graph.params.items():
        if not k.startswith(nid + ".") or k.startswith("_"):
            continue
        name = k.split(".", 1)[1]
        d = defaults.get(name)
        if d is None or v != d[2]:
            set_here.add(name)
    if not set_here:
        return
    emitted = set(spec_op) | {_ALIASED.get(k, k) for k in spec_op}
    reverse = {v: k for k, v in _ALIASED.items()}
    emitted |= {reverse[k] for k in spec_op if k in reverse}
    dropped = sorted(p for p in set_here if p not in emitted and _ALIASED.get(p, p) not in emitted)
    if dropped:
        raise ValueError(
            f"{node['op']}:{graph.impl_of(node)} was given {', '.join(dropped)}, and its emitter "
            f"does not pass {'them' if len(dropped) > 1 else 'it'} to the engine. A parameter "
            f"that is set and then discarded makes the composition claim a setting the run never "
            f"had -- refuse it rather than ignore it. (Emitted: {sorted(spec_op)})")



# The explicit diffusion step is stable only while dt * D <= 1 per unit cell spacing. On the
# degree-normalised graph Laplacian this operator uses, D is `chi * d`, so the bound is
# dt * chi * max(d_a, d_h) <= 1. Kept a little under 1: the mesh is not uniform, and a cell with
# more neighbours than average sees a larger effective coefficient than this average implies.
CFL_LIMIT = 0.8


def _assert_rd_stable(ops, dt, name=""):
    """Refuse a spec whose diffusion step cannot be integrated stably.

    THE DEFECT THIS CLOSES. This bound lived in a COMMENT in this file, asserting
    `dt*chi*max(d_a,d_h) = 0.02*65*0.16 = 0.21 <= 1`. It was true when written, against the
    defaults of the time. The vocabulary has since moved to chi = 4.0 (x50 by the clock fix =
    200) and d_h = 0.7, and Phase 2 widened d_h to 12.5 so Okuda's phi = 10 would be reachable
    at all. Nobody re-derived the bound, because a comment cannot be re-derived -- it can only be
    re-read, and nobody had a reason to.

    The result: 0.02 * 200 * 0.7 = 2.8, four times over. The chemistry goes non-finite within a
    hundred frames, the Biologist correctly refuses the run, and every composition in that region
    is unusable -- including `okuda_route`, the recipe named for the target. At the widened
    ceiling d_h = 12.5 the number is 50.

    A stability limit that is widened by one phase and relied upon by another has to be a CHECK.
    """
    d = next((o for o in ops if o.get("op") == "cell_diffuse"), None)
    if d is None:
        return
    chi = float(d.get("chi", 1.0))
    dmax = max(float(d.get("d_a", 0.0)), float(d.get("d_h", 0.0)))
    cfl = dt * chi * dmax
    if cfl > CFL_LIMIT:
        raise ValueError(
            f"{name or 'this composition'} is not integrable: dt*chi*max(d_a,d_h) = "
            f"{dt} * {chi} * {dmax} = {cfl:.2f}, over the {CFL_LIMIT} limit for an explicit "
            f"step. The chemistry will go non-finite and every number after that is about the "
            f"solver, not the tissue. Lower chi or the diffusivities, or raise dt -- but note "
            f"that chi and the reaction rate are BOTH scaled by 1/dt, so raising dt does not "
            f"help. (working reference: coral_fixed_ball sits at 0.21)")



# ============================================================================ RUN-TO-RUN SEEDING
# THE DEFECT: `general.seed` was written into every spec and read by NOTHING. Both stochastic
# operators take a `seed` parameter -- seed_mesh_3d for the vertex jitter, seed_cell_rd for which
# cells are nucleated -- and the translator passed neither. So a batch of "three seeds" was three
# copies of one run: measured on 2026-08-01, seeds 0/1/2 at seed_frac 0.06 gave act_max 0.501,
# 0.501, 0.501 and red_frac 0.374, 0.374, 0.374 -- bit-identical to three decimal places.
#
# Replication is not a nicety here. The campaign's whole objective is to tell "this MECHANISM
# makes tubes" from "this RUN made a tube", and the only instrument for that is running the same
# composition again differently. Every spread this campaign has quoted across seeds describes the
# floating-point reproducibility of one trajectory.
SEED_SENTINEL = "__RUN_SEED__"
SEEDED_OPS = ("seed_mesh_3d", "seed_cell_rd")


def _seed_the_run(ops, seed_):
    """Give every stochastic operator this run's seed, and refuse if none would take it.

    ONE seed, shared, because that is what the working specs do. `archive_rounded.py`,
    `archive_vh_rd_coral.py` and `run_tyssue_fig5.py` all pass the same SEED to seed_mesh_3d and
    seed_cell_rd. I first offset them per operator, reasoning that the mesh jitter and the
    nucleation are independent draws -- true, and beside the point: a composition that reproduces
    an archived run has to draw the same numbers it did, and the archive's convention is shared.
    Departing from it would have made every reproduction check compare two different experiments,
    which is the error this campaign keeps finding in other places.
    """
    n_seeded = 0
    for o in ops:
        if o.get("op") in SEEDED_OPS or o.get("seed") == SEED_SENTINEL:
            o["seed"] = int(seed_)
            n_seeded += 1
    if n_seeded == 0:
        raise ValueError(
            "this spec declares general.seed and no operator consumes it, so changing the seed "
            "cannot change the run. A batch of replicates would be one run repeated -- which is "
            "what happened for the whole campaign until 2026-08-01. Either add a stochastic "
            "operator or stop calling the batch replicated.")
    return n_seeded


def to_spec(graph: CompositionGraph, *, name="okuda", frames=350, seed_=0, grow_after=None,
            record_every=1):
    """Compile a CompositionGraph into a runnable Plexus spec dict.

    Raises if the graph is not runnable -- a composition with an unmet precondition or a
    dangling slot must never reach the cluster (D4: it would silently no-op and its metrics
    would be recorded as evidence that the mechanism cannot work).
    """
    if grow_after is None:                      # from_preset stashes the preset's grow_after
        grow_after = int(graph.params.get("_run.grow_after", 100))
    ok, why = graph.is_runnable()
    if not ok:
        raise ValueError(f"refusing to compile a non-runnable composition: {why}")

    # Size the reservoirs from where this run is GOING, not from a constant. A seeding node is
    # what tells us where it starts; a checkpoint start carries its own count and falls back.
    seed_cells = None
    for node in graph.ops:
        if node["op"] == "seed_mesh_3d":
            seed_cells = int(_p(graph, node["id"], "n_cells"))
    if seed_cells is not None:
        vbuf, cbuf, target_cells = _reservoirs(seed_cells, frames)
    else:
        vbuf, cbuf, target_cells = VBUF_FALLBACK, CBUF_FALLBACK, None

    ops = []
    for node in graph.ops:
        emit = EMIT.get(node["op"])
        if emit is None:
            raise ValueError(f"no backend emitter for operator {node['op']!r}")
        spec_op = emit(graph, node, grow_after)
        _assert_params_consumed(graph, node, spec_op)
        ops.append(spec_op)

    _seed_the_run(ops, seed_)
    _assert_rd_stable(ops, DT_GLOBAL, name)

    # D3: topology must be recorded on the SAME stride as positions, and this is asserted
    # downstream. This is the fix for the phantom "97% hollow" result.
    ops.append({"op": "topo_snapshot_3d", "at": "vertex", "every": record_every})

    unordered = sorted({o["op"] for o in ops} - set(SCHEDULE_ORDER))
    if unordered:
        raise ValueError(
            f"operators missing from SCHEDULE_ORDER: {unordered}. They would sort to the END of "
            f"the schedule -- e.g. a growth operator running AFTER the recorder. Add them.")
    order = {n: i for i, n in enumerate(SCHEDULE_ORDER)}
    ops.sort(key=lambda o: order.get(o["op"], 999))
    sched = [o["op"] for o in ops]

    cfg = {
        "general": {"name": name, "seed": int(seed_), "n_frames": int(frames),
                    "dt": DT_GLOBAL, "record_cap": int(frames) + 2,
                    "record_every": int(record_every),
                    "boundary": "free", "dim": 3, "world": [16 * 5.0] * 3},
        # WHERE THIS RUN IS GOING, recorded so it can be CHECKED. A reservoir is only
        # meaningful against a destination: the weekend battery seeded 150 into a buffer holding
        # 1778 -- twelve times the seed, generous by any fixed multiple -- and every run still
        # stopped on it, because the tissue wanted more. No rule of thumb catches that. A run
        # that states its target can be refused before it burns a GPU; one that does not, cannot.
        "_run": {"target_cells": target_cells, "seed_cells": seed_cells},
        "sets": {"vertex": {"n": vbuf},
                 "cell": {"n": cbuf, "state": {"chem": {"width": 2, "integration": "first_order"},
                                               "cen": {"width": 3}, "area": {"width": 1}}}},
        "fields": {},
        "operators": ops,
        "schedule": sched,
        # provenance: what this spec IS, so a config file alone identifies its hypothesis
        "_discovery": {
            "comp_hash": None,                                 # filled by write_config
            "structure": graph.structure(),
            "region": graph.name_region(),
            "defect_fixes": ["D1 engine owns the clock (every=1)",
                             f"D2 one global dt={DT_GLOBAL}",
                             f"D3 topo_snapshot every={record_every} == position stride"],
        },
    }
    return cfg


def write_config(graph, name, out_dir=None, **kw):
    """Write config/okuda/<name>.yaml -- the Plexus contract path."""
    from run_record import comp_hash
    out_dir = out_dir or os.path.abspath(os.path.join(HERE, "..", "config", "okuda"))
    os.makedirs(out_dir, exist_ok=True)
    cfg = to_spec(graph, name=name, **kw)
    cfg["_discovery"]["comp_hash"] = comp_hash(graph)
    # THE COMPOSITION, BESIDE THE SPEC. A spec records operators and their implementations but
    # NOT the connections between them, so a graph cannot be rebuilt from one -- which is why the
    # Archivist, having correctly named the three best starting points on disk, had to report
    # "no composition.json -- its spec cannot be rebuilt as a graph" for all three and fall back
    # to the reference recipes. It knew where to start and could not get there.
    #
    # Writing it costs a few hundred bytes and makes every run from here on a candidate parent.
    # The 63 already on disk stay unreachable; that is the price of not having done this earlier.
    try:
        json.dump({"ops": graph.ops, "conns": graph.conns, "params": graph.params},
                  open(os.path.join(out_dir, f"{name}.composition.json"), "w"), indent=1)
    except Exception as e:
        print(f"[translate] could not record the composition for {name}: {type(e).__name__}")
    path = os.path.join(out_dir, f"{name}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path, cfg


# --------------------------------------------------------------------------- preset -> graph
def from_preset(p: dict) -> CompositionGraph:
    """Express a hand-written run_tyssue_round preset as a CompositionGraph.

    THE VALIDATION GATE. If a recipe we already trust cannot be built from legal one-edit moves,
    the campaign is searching a space that does not contain our own evidence.
    """
    g = CompositionGraph(ops=[
        {"id": "seed_mesh_3d0", "op": "seed_mesh_3d", "impl": "checkpoint"},
        {"id": "cell_geometry_3d0", "op": "cell_geometry_3d", "impl": "scatter_add"},
    ])
    add = lambda gg, op, impl: gg.apply(("add_op", op, impl))[0]

    cones = "spots" in p
    rd = bool(p.get("rd", False))

    if cones:
        g = add(g, "seed_cell_rd", p.get("seed_mode", "cones").replace("cones", "cone"))
    if rd or not cones:
        g = add(g, "cell_adjacency", "shared_edge")
        g = add(g, "cell_diffuse", "graph_laplacian")
        g = add(g, "cell_react", p.get("rd_impl", "brusselator"))
        if not cones:
            g = add(g, "seed_cell_rd", "spot")

    g = add(g, "grow_3d",
            "hill_conserve_amount" if p.get("conserve_amount", True) else "hill_no_conserve")
    g = add(g, "shape_energy_3d", "monolayer" if p.get("monolayer") else "default")
    # ONLY THE PURSE-STRING SURVIVES THE SPLIT. A preset asking for K_extrude is asking for
    # `extrusion_forcing_3d`, which is not in this vocabulary -- an archived forcing control cannot
    # be replayed through the search space, and should not be.
    if p.get("K_purse", 0.0) > 0:
        g = add(g, "0", "default")
    g = add(g, "reconnect_t1_3d", "length_threshold")
    g = add(g, "divide_3d", "orient_iface" if p.get("orient_iface") else "hertwig")

    # route the morphogen: whichever node produces it feeds growth (+ axis / site if present)
    src = next((o["id"] for o in g.ops if "morphogen" in OPERATORS[o["op"]]["outputs"]), None)
    if src is None:
        return g
    for dst_op, slot in (("grow_3d", "gate"), ("divide_3d", "axis"),
                         ("interface_line_tension_3d", "site")):
        dst = next((o["id"] for o in g.ops if o["op"] == dst_op), None)
        if dst is None:
            continue
        if dst_op == "divide_3d" and not p.get("orient_iface"):
            continue                                            # hertwig uses the cell's own axis
        g = g.apply(("connect", src, dst, slot))[0]

    # carry theta across so the compiled spec is numerically the preset, not the defaults
    pm = g.default_params()
    pm["_run.grow_after"] = int(p.get("grow_after", 0))
    pm["_run.frames"] = int(p.get("frames", 350))
    node = lambda op: next((o["id"] for o in g.ops if o["op"] == op), None)
    mapping = [
        ("shape_energy_3d", "K_V", p.get("K_V")), ("shape_energy_3d", "relax_iters", p.get("relax")),
        ("shape_energy_3d", "kappa_s", p.get("kappa_s")), ("shape_energy_3d", "h0", p.get("h0")),
        ("grow_3d", "rate", p.get("rate")), ("grow_3d", "a_sw", p.get("a_sw")),
        ("grow_3d", "hill", p.get("hill")), ("grow_3d", "rho", p.get("rho")),
        ("divide_3d", "cycle_cv", p.get("cycle_cv")),
        # CLOCK RE-ANCHORING: these are per-CALL in the operator and the archived configs ran
        # divide_3d once every 4 frames. Rescale so the replay preserves the archived
        # wall-clock behaviour under the corrected clock (see composition_space header).
        ("divide_3d", "min_cycle",
         (p["min_cycle"] * DIVIDE_CALL_PERIOD_BEFORE_D1) if p.get("min_cycle") else None),
        ("divide_3d", "max_cycle",
         (p["max_cycle"] * DIVIDE_CALL_PERIOD_BEFORE_D1)
         if (p.get("max_cycle") and p["max_cycle"] < 10**8) else None),
        ("shape_energy_3d", "Gamma", p.get("Gamma")), ("shape_energy_3d", "Lambda", p.get("Lambda")),
        ("shape_energy_3d", "p0", p.get("p0")),
        ("interface_line_tension_3d", "a_sw", p.get("iface_asw", p.get("a_sw"))),
        ("divide_3d", "orient_asw", p.get("orient_asw", p.get("a_sw"))),
        ("seed_cell_rd", "n_spots", p.get("spots")),
        ("cell_react", "F", p.get("F")), ("cell_react", "kk", p.get("kk")),
        ("cell_react", "mu_h", p.get("mu_h")),
        ("shape_energy_3d", "gamma", p.get("gamma")),
        ("cell_diffuse", "chi", p.get("chi")), ("cell_diffuse", "d_a", p.get("d_a")),
        ("cell_diffuse", "d_h", p.get("d_h")),
        ("cell_react", "rate", p.get("rate")), ("cell_react", "a0", p.get("a0")),
        ("cell_react", "gamma", p.get("gamma")),
        ("seed_cell_rd", "cone_deg", p.get("cone_deg")),
        ("reconnect_t1_3d", "l_th_frac", p.get("l_th_frac")),
    ]
    for op, pname, val in mapping:
        nid = node(op)
        if nid is not None and val is not None:
            pm[f"{nid}.{pname}"] = val
    return g.with_params(pm)


def load_presets():
    """Import run_tyssue_round.PRESETS without running anything."""
    import importlib.util
    import sys
    sys.path.insert(0, TYSSUE)
    spec = importlib.util.spec_from_file_location(
        "run_tyssue_round", os.path.join(TYSSUE, "run_tyssue_round.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # heavy imports may fail; the presets are what we want
        print(f"  (preset module partially loaded: {type(e).__name__}: {str(e)[:70]})")
    return getattr(mod, "PRESETS", {})
