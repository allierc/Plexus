#!/usr/bin/env python
"""Two reaction-diffusion species in one tissue: one patterns growth, the other patterns death.

Cedric, 11 August: *"could you implement two species ... the second species could trigger cell death
after a third of total frames, not right away? could you modify eg log/okuda/r001_03, make a few
variants with this 2-species RD, one for growth one for cell death?"*

WHY TWO SPECIES IS A DIFFERENT CLAIM FROM ONE. Every run this campaign has made carries a single
Gray-Scott pair, and every mechanism reads the same activator: growth is gated on it, death is gated
on it, the purse-string selects on it. So the tissue has exactly one map, and every operator is
looking at the same picture -- which means "where it grows" and "where it dies" cannot be different
places by construction. That is why six rounds produced buds and lobes and never a finger: a bulge
becomes a FINGER when the tissue grows at the tip and stops growing, or is removed, at the flanks,
and one field cannot say both.

Two species is two maps. `chem` is width 4: columns (0,1) are species A, columns (2,3) species B,
each with its own seeder, its own diffusion and its own reaction, and the operators say which they
read through `chan`. They cannot leak into each other -- certified directly: react and diffuse at
chan 0 write only columns 0,1 and at chan 2 only columns 2,3.

DEATH STARTS LATE, AND THAT IS THE POINT. `after_frame = n_frames // 3` on the death species. With
death running from frame 0 the tissue never gets a shape to sculpt -- it is 2,000 cells and the
death rule removes cells from a sphere. Letting species A build the lobes first and only then
switching species B on makes the second species a SCULPTOR of an existing form rather than a brake
on its formation. This is the operator's own argument for a seed window, one level up: when a
mechanism starts is part of the mechanism.

    python make_two_species.py            write the specs
    python make_two_species.py --check    also run the static premises and the unread-key gate
"""
import argparse
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(os.path.dirname(HERE), "config", "okuda")
BASE = "r001_03"          # a gs gated shaping member: chemistry, growth, purse-string, mechanics

# SPECIES B'S OWN KINETICS, deliberately NOT a copy of species A's. Two identical Gray-Scott systems
# on the same mesh would produce the same pattern from the same seed and the second map would be
# redundant. The Turing wavelength is set by the diffusion ratio, so B is given a coarser one: it
# should paint a FEW large domains where A paints many small spots, which is what makes "grow at the
# spots, die between the domains" a spatial statement rather than a noise difference.
#
# F and kk are Cedric's 2D two-species values (F 0.039, kk 0.058) rather than the campaign's
# (0.046, 0.062): a lower feed and kill pair sits in Gray-Scott's coral/negatons region and gives
# broader, slower domains than the spot regime species A runs in.
SPECIES_B = dict(F=0.039, kk=0.058, rate=1.0)
DIFFUSE_B = dict(d_a=0.16, d_h=0.32, chi=1.3)      # 2x A's diffusivities -> a coarser wavelength
SEED_B = dict(mode="scatter", seed_frac=0.12, seed=7)   # its own seed, or it copies A's pattern


def _ops(spec):
    return {o["op"]: o for o in spec["operators"]}


def build(tag, *, death_chan=None, growth_chan=0, death_mode="chem_low", a_sw=0.25,
          max_mark_frac=0.005, shrink_rate=0.05, after_frac=3.0, critical_frac=0.15):
    """`r001_03` + a second RD species. `death_chan=None` means species B only patterns growth."""
    spec = copy.deepcopy(yaml.safe_load(open(os.path.join(CONFIG, f"{BASE}.yaml"))))
    spec["general"]["name"] = tag
    n_frames = int(spec["general"]["n_frames"])
    spec.pop("_discovery", None)

    # FOUR CHANNELS. The engine sizes the buffer from this, and every operator's `chan` indexes into
    # it, so widening the state is the whole of the structural change.
    spec["sets"]["cell"]["state"]["chem"]["width"] = 4

    ops, out = _ops(spec), []
    for o in spec["operators"]:
        out.append(o)
        # species B is seeded, diffused and reacted immediately after A's own operator, so the two
        # systems advance on the same clock and in the same order
        if o["op"] == "seed_cell_rd":
            out.append({"op": "seed_cell_rd", "at": "cell", "before_frame": o.get("before_frame", 3),
                        "chan": 2, **SEED_B})
        elif o["op"] == "cell_diffuse":
            out.append({"op": "cell_diffuse", "at": "cell",
                        "implementation": o.get("implementation", "graph_laplacian"),
                        "chan": 2, **DIFFUSE_B})
        elif o["op"] == "cell_react":
            out.append({"op": "cell_react", "at": "cell", "model": "gray_scott",
                        "chan": 2, **SPECIES_B})
    spec["operators"] = out

    _ops(spec)["grow_3d"]["chan"] = int(growth_chan)
    # `reset_noise` is not read by divide_3d and rides in from the base spec; the unread gate
    # refuses the spec while it is there, and it is not a key this experiment wants.
    for o in spec["operators"]:
        o.pop("reset_noise", None)

    if death_chan is not None:
        # DEATH ON ITS OWN MAP, AND LATE. `chem_low` is the mode that reads a chemical field: cells
        # die where species B is BELOW a fraction of its own maximum -- i.e. between B's domains --
        # so B's pattern is a map of where the tissue is spared.
        spec["operators"].append(
            {"op": "apoptosis_3d", "at": "vertex", "cell_set": "cell",
             "p0": _ops(spec)["shape_energy_3d"].get("p0", 3.5),
             "mode": death_mode, "chan": int(death_chan), "a_sw": a_sw,
             "max_mark_frac": max_mark_frac, "min_age": 4, "shrink_rate": shrink_rate,
             "critical_frac": critical_frac,
             # A THIRD OF THE RUN by default, so species A has built a form before B removes any.
             "after_frame": int(n_frames // after_frac)})

    # SCHEDULE ORDER IS translate.SCHEDULE_ORDER's, read from it rather than restated, with the
    # second species' operators sitting beside their own kind.
    sys.path.insert(0, HERE)
    from translate import SCHEDULE_ORDER
    rank = {n: i for i, n in enumerate(SCHEDULE_ORDER)}
    spec["operators"].sort(key=lambda o: (rank.get(o["op"], 999), o.get("chan", 0)))
    spec["schedule"] = [o["op"] for o in spec["operators"]]
    spec["_two_species"] = {
        "base": BASE, "growth_reads_chan": growth_chan,
        # `is not None`, NOT truthiness: chan 0 is a legal channel and `if death_chan` reported
        # "no death" for the ts_death_same_a variant while the operator had after_frame 600.
        "death_reads_chan": death_chan,
        "death_after_frame": int(n_frames // after_frac) if death_chan is not None else None,
        "max_mark_frac": max_mark_frac, "shrink_rate": shrink_rate,
        "why": ("species A (chem 0,1) and species B (chem 2,3) are independent RD systems in one "
                "buffer; B has 2x the diffusivities so its wavelength is coarser. A single-species "
                "run cannot put growth and death in different places, because every operator reads "
                "the same field."),
    }
    return spec


VARIANTS = [
    # the control for the pair: two species present, but only A does anything. Isolates the COST of
    # carrying a second field from its effect.
    ("ts_ctrl_growth_a", dict(death_chan=None, growth_chan=0)),
    # growth reads the SECOND species -- coarser wavelength, so fewer and broader growth domains.
    # This is the "one for growth" variant.
    ("ts_growth_b", dict(death_chan=None, growth_chan=2)),
    # the pair that is the point: A grows the lobes, B kills between its own domains, starting a
    # third of the way in. This is the "one for cell death" variant.
    ("ts_death_b_late", dict(death_chan=2, growth_chan=0)),
    # the same, with death biting harder -- a_sw 0.5 marks everything below half of B's maximum
    # rather than a quarter, so the spared regions are narrower.
    ("ts_death_b_sharp", dict(death_chan=2, growth_chan=0, a_sw=0.5)),
    # the adversarial control: death on the SAME species that drives growth. If this matches
    # ts_death_b_late then the second map bought nothing and the result is about death, not about
    # two species.
    ("ts_death_same_a", dict(death_chan=0, growth_chan=0)),
]

# ---------------------------------------------------------------------------------------------
# A LOT OF DEATH. Cedric, 11 August: "the death series has landed, I do not see much death, make a
# new series with a lot of cell death."
#
# THE FIRST SERIES WAS CAP-LIMITED, NOT RULE-LIMITED, and it proved it by accident:
# `ts_death_b_late` (a_sw 0.25) and `ts_death_b_sharp` (a_sw 0.5) came back BIT-IDENTICAL -- 301
# deaths, 11,955 cells, every metric equal. Two different selection thresholds, one outcome. The
# reason is `max_mark_frac = 0.005`: both thresholds propose far more cells than 0.5% of the
# tissue, so the cap takes the worst 0.5% either way and the threshold never binds. Turning a_sw
# up cannot produce more death; it only reorders a queue whose length is fixed elsewhere.
#
# So the ladder is on the CAP, and on the one other quantity that sets throughput. Deaths per unit
# time are (how many are under sentence at once) / (how long each takes to clear), and the second
# term is `shrink_rate`: a marked cell shrinks by that fraction per tick until it passes
# `critical_frac x v_ref` and is extruded, so ~ln(0.15)/ln(1-rate) ticks at 37 for rate 0.05 and 12
# for 0.15. Raising both multiplies.
#
# 301 deaths of ~12,000 cells is 2.5% of the tissue over 1,200 frames. The top of this ladder is
# fifty times the cap and three times the clearing rate, which on the same tissue is a different
# regime rather than more of the same -- and the r020 measurements say where it breaks: uncapped,
# death took a parent from protr 1.513 to 1.131 with 1,660 of 7,424 cells dead. That is the wall
# this ladder is meant to find, so the series is built to cross it rather than to stop short.
DEATH_SERIES = [
    ("tsd_cap02",      dict(death_chan=2, max_mark_frac=0.02)),
    ("tsd_cap05",      dict(death_chan=2, max_mark_frac=0.05)),
    ("tsd_cap10",      dict(death_chan=2, max_mark_frac=0.10)),
    ("tsd_cap25",      dict(death_chan=2, max_mark_frac=0.25)),
    # the same sentence, cleared three times faster: throughput without widening the queue, which
    # separates "how many are dying" from "how fast each one goes"
    ("tsd_cap10_fast", dict(death_chan=2, max_mark_frac=0.10, shrink_rate=0.15)),
    # both levers at once, and death from a tenth of the way in rather than a third -- the most
    # death this vocabulary can express short of removing the cap
    ("tsd_max",        dict(death_chan=2, max_mark_frac=0.25, shrink_rate=0.15, after_frac=10.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all", action="store_true", help="the first series too, not just the death ladder")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    print(f"{'spec':<20}{'death':>6}{'cap':>7}{'shrink':>7}{'after':>7}{'ops':>5}  gate")
    bad = 0
    todo = (VARIANTS + DEATH_SERIES) if a.all else DEATH_SERIES
    for tag, kw in todo:
        spec = build(tag, **kw)
        with open(os.path.join(CONFIG, f"{tag}.yaml"), "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
        note = ""
        if a.check:
            import biologist as B
            from make_basis import _unread
            fails = [r.pid for r in B.check(spec) if r.status == "fail"] + _unread(spec)
            bad += bool(fails)
            note = "BROKEN " + ",".join(fails) if fails else "ok"
        ts = spec["_two_species"]
        print(f"{tag:<20}{str(ts['death_reads_chan']):>6}{ts['max_mark_frac']:>7.3f}"
              f"{ts['shrink_rate']:>7.2f}{str(ts['death_after_frame']):>7}"
              f"{len(spec['operators']):>5}  {note}")
    print(f"\n{len(todo)} specs -> {CONFIG}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
