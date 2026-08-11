#!/usr/bin/env python
"""Cell death as a SCULPTOR: can a spatial pattern of dying cells carve a 3D form?

Cedric, 11 August: *"can you try to make new variants where cell death sculpts the 3D shape somehow,
you are free to make crazy tests."*

WHAT THE LADDER TAUGHT, and it corrected me. I built the previous series on the assumption that
`max_mark_frac` -- how much tissue may be under sentence at once -- was the lever. Measured:

    cap        0.005   0.02   0.05   0.10   0.25   0.10+fast   0.25+fast+early
    deaths       301    732    415    481    518      2261          3460
    protr_f    1.082  1.082  1.063  1.062  1.062     1.168         1.140
    invag_f    0.114  0.124  0.117  0.084  0.081     0.196         0.111

Fifty times the cap moved deaths by a factor of 1.7. Tripling `shrink_rate` at a FIXED cap moved
them by 4.7. The queue never fills: what binds is how fast a sentenced cell shrinks to the
extrusion threshold, not how many may be sentenced. And the run with the most SHAPE change is the
fast one, not the wide one. So every variant here runs `shrink_rate = 0.15` -- roughly 12 ticks
from sentence to extrusion instead of 37 -- and none of them leans on the cap.

WHY A SPATIAL PATTERN AND NOT A RATE. Every death this campaign has run selects on a per-cell
state -- too small, too slow, too dim -- which produces death scattered over the whole surface, and
scattered death is erosion, not sculpture. It thins a ball into a smaller ball. To CARVE, the dying
cells have to be somewhere in particular, and the geometric modes are the only ones that say where:
`band`, `cone` and `list` choose a POPULATION once, and are deliberately exempt from the rate cap
because a set chosen once has no flux. On a growing tissue that is a chisel: growth inflates, and a
band of dead cells is a groove that the inflation then has to accommodate.

THE SEVEN, ordered by how much they ask of the topology. Each is `r001_03` -- a Gray-Scott gated
member with growth, purse-string and the full mechanics -- plus one death operator, so anything
that happens is attributable to it.

    waist       one equatorial band. Does a groove become a NECK, and a neck a bud?
    segments    nine bands. Does a stack of grooves segment the body, as somites do?
    crown       a polar cap. Does removing the pole make a CUP -- the invagination twenty rounds
                never produced, and the one Okuda morphology with no outward mechanism?
    antiphase   die where the GROWTH activator is low, at speed. The finger hypothesis stated
                directly: grow at the spots, remove the flanks, and a bulge should sharpen.
    coarse_cut  die in the valleys of the COARSE second species while growth follows the fine
                first one -- two length scales, one tissue.
    old_core    kill the oldest cells fastest. The tips are the newest tissue, so this removes the
                body and keeps the growing front.
    waist_purse the waist again with the purse-string tripled: a line tension on the activator
                interface pulling the groove closed while death widens it.

    python make_sculpt.py            write the specs
    python make_sculpt.py --check    also run the static premises and the unread-key gate
"""
import argparse
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(os.path.dirname(HERE), "config", "okuda")
BASE = "r001_03"

# THE THROUGHPUT SETTING, measured rather than chosen: shrink_rate 0.15 clears a sentenced cell in
# ~ln(0.15)/ln(0.85) = 12 ticks against 37 at 0.05, and it is the difference between 481 deaths and
# 2,261 on otherwise identical runs. `critical_frac` stays at 0.15 so the extrusion threshold is
# the one every previous death run used.
FAST = dict(shrink_rate=0.15, critical_frac=0.15, min_age=4)
# a cap that is present but not binding, so a state-selected mode still has a rate; the geometric
# modes ignore it by design
CAP = 0.05


def _ops(spec):
    return {o["op"]: o for o in spec["operators"]}


def build(tag, death, *, after_frac=3.0, two_species=False, k_purse=None, note="",
          inhib=None, growth_chan=0, fast=None):
    spec = copy.deepcopy(yaml.safe_load(open(os.path.join(CONFIG, f"{BASE}.yaml"))))
    spec["general"]["name"] = tag
    n_frames = int(spec["general"]["n_frames"])
    spec.pop("_discovery", None)
    for o in spec["operators"]:
        o.pop("reset_noise", None)          # unread by divide_3d; rides in from the base

    if two_species:
        # the coarse second map, exactly as make_two_species builds it
        spec["sets"]["cell"]["state"]["chem"]["width"] = 4
        out = []
        for o in spec["operators"]:
            out.append(o)
            if o["op"] == "seed_cell_rd":
                out.append({"op": "seed_cell_rd", "at": "cell", "chan": 2, "seed": 7,
                            "before_frame": o.get("before_frame", 3), "mode": "scatter",
                            "seed_frac": 0.12})
            elif o["op"] == "cell_diffuse":
                out.append({"op": "cell_diffuse", "at": "cell", "chan": 2,
                            "implementation": o.get("implementation", "graph_laplacian"),
                            # chi 0.4, not 1.3: at 1.3 with doubled diffusivities gate G16
                            # measured this field EXTINCT (act_max 0.0). At Cedric's own value it
                            # patterns at 15 spots, spacing 3.18 -- FINER than A's 6.99, so the
                            # coarse map is A and the fine one is B.
                            "d_a": 0.08, "d_h": 0.16, "chi": 0.4})
            elif o["op"] == "cell_react":
                out.append({"op": "cell_react", "at": "cell", "chan": 2, "model": "gray_scott",
                            "F": 0.039, "kk": 0.058, "rate": 1.0})
        spec["operators"] = out

    if inhib is not None:
        # THE SECOND MORPHOGEN STOPS GROWTH. Every growth law here has been purely activating --
        # `rate * (rho + hill(a))` -- so the slowest a cell could grow was the rho baseline and
        # nothing could say STOP. A bulge sharpens into a finger when the tip grows and the flanks
        # do not, which needs a zero, and an activating field has none.
        _ops(spec)["grow_3d"].update(inhib)
    _ops(spec)["grow_3d"]["chan"] = int(growth_chan)
    if k_purse is not None:
        _ops(spec)["interface_line_tension_3d"]["K_purse"] = float(k_purse)

    # NO DEATH OPERATOR AT ALL when the variant asks for none. An empty `death` dict used to fall
    # through to `apoptosis_3d`'s own default mode -- `competition` -- so the four variants meant to
    # isolate INHIBITION would each have carried a death rule nobody asked for, and any difference
    # between them and the control would have been unattributable. A mechanism that arrives by
    # default is the same defect as a parameter nothing reads.
    if death:
        spec["operators"].append(
            {"op": "apoptosis_3d", "at": "vertex", "cell_set": "cell",
             "p0": _ops(spec)["shape_energy_3d"].get("p0", 3.5),
             "max_mark_frac": CAP, **(fast or FAST), **death,
             # LATE, so growth has built something to carve; from frame 0 it erodes a sphere.
             "after_frame": int(n_frames // after_frac)})

    sys.path.insert(0, HERE)
    from translate import SCHEDULE_ORDER
    rank = {n: i for i, n in enumerate(SCHEDULE_ORDER)}
    spec["operators"].sort(key=lambda o: (rank.get(o["op"], 999), o.get("chan", 0)))
    spec["schedule"] = [o["op"] for o in spec["operators"]]
    spec["_sculpt"] = {"base": BASE, "death": death, "after_frame": int(n_frames // after_frac),
                       "two_species": two_species, "k_purse": k_purse, "inhib": inhib,
                       "growth_chan": growth_chan, "asks": note}
    return spec


VARIANTS = [
    # ---- the chisels: a population chosen once, exempt from the rate cap, so the cut is a SHAPE
    ("sc_waist",     dict(mode="band", band_deg=6.0, n_bands=1), {},
     "does an equatorial groove become a neck, and a neck a bud?"),
    ("sc_segments",  dict(mode="band", band_deg=3.0, n_bands=9), {},
     "do nine grooves segment the body into a stack, as somites do?"),
    ("sc_crown",     dict(mode="cone", cone_deg=25.0), {},
     "does removing the pole make a CUP -- the invagination no outward mechanism can produce?"),
    # ---- the chemical carvers: death follows a field, at speed
    ("sc_antiphase", dict(mode="chem_low", a_sw=0.45), {},
     "the finger hypothesis: grow at the spots, remove the flanks, and a bulge should sharpen"),
    ("sc_coarse_cut", dict(mode="chem_low", chan=2, a_sw=0.45), dict(two_species=True),
     "two length scales in one tissue: fine growth, coarse carving"),
    ("sc_old_core",  dict(mode="older", stall_frac=0.6), {},
     "kill the OLDEST fastest -- the tips are the newest tissue, so this removes the body"),
    # ---- the one that opposes two mechanisms
    ("sc_waist_purse", dict(mode="band", band_deg=6.0, n_bands=1), dict(k_purse=3.0),
     "a line tension pulling the groove closed while death widens it -- which wins?"),
]

# ---------------------------------------------------------------------------------------------
# THE INHIBITOR SERIES. Cedric, 11 August: "make variants where the blue morphogen stops cell
# growth, so that we see the blue, and only red spots growing -- that could be useful next for the
# agentic loop."
#
# WHY THIS IS THE MISSING HALF. Every growth law in this project is ACTIVATING ONLY:
# `rate * (rho + hill(a))`. The fastest a cell grows is set by the activator and the SLOWEST is the
# rho baseline, which is never zero -- so the tissue between the spots keeps inflating and a bulge
# can never become a finger. Six rounds of lobes are exactly what an all-activating law predicts.
# An inhibitor supplies the zero: where the second morphogen saturates, growth stops outright.
#
# `inhib_chan: 2` reads the COARSE species, so the picture is fine red spots growing inside broad
# teal regions that do not -- which is the configuration the user asked to see, and the one worth
# handing the loop: it is a one-edit `set_param` away from every existing gs member.
#
# The ladder is on `inhib_sw`, the fraction of the inhibitor's own maximum at which growth is half
# off. Low = almost everywhere inhibited (only the very brightest spots grow); high = only the
# inhibitor's peaks stop growth. The two ends bracket "does inhibition sharpen a bulge or freeze
# the tissue", which is the question, and `sc_inh_none` is the control that carries the coarse
# species and does NOT read it -- so the effect is attributable to the inhibition rather than to
# the presence of a second field.
# ---------------------------------------------------------------------------------------------
# SLOW DEATH, SO THE PATTERN IS VISIBLE. Cedric, 11 August: "make variants where cell death is
# slower so that we see the blue pattern for a while -- if the cell death is super fast, as soon as
# the blue pattern appears they disappear under cell deaths."
#
# This is a legitimate complaint about the INSTRUMENT and not about the mechanism. A cell is drawn
# teal while its growth is inhibited and drawn at all only while it is alive, so the time a pattern
# is legible on screen is the time between a cell being sentenced and being extruded. The previous
# series set `shrink_rate = 0.15` deliberately -- it is the throughput lever, worth 4.7x the deaths
# -- and that same choice clears a cell in about twelve ticks. The pattern is real and lasts a fifth
# of a second of movie.
#
# The clearing time is exactly ln(critical_frac)/ln(1 - shrink_rate) ticks:
#
#     shrink_rate    0.15    0.05    0.02    0.01     0.02 with critical_frac 0.05
#     ticks to go      12      37      94     189                      148
#
# so the ladder is on the same arithmetic that produced the deaths, run backwards. `sc_slow_deep`
# lowers the extrusion threshold instead of the rate -- a cell has to shrink FURTHER before it goes,
# which keeps it on screen while visibly dwindling rather than simply lingering at full size. And
# `sc_slow_late` moves the start to two thirds of the run rather than one, so the blue pattern is
# untouched for twice as long before anything is removed at all.
#
# All five keep the inhibitor at `inhib_sw 0.35` -- the mid setting -- because the point is to SEE
# the blue, and a variant that could not show it would answer a different question.
SLOW = dict(inhib_chan=2, inhib_sw=0.35, inhib_hill=4.0)
SLOW_SERIES = [
    ("sc_slow_37",   dict(shrink_rate=0.05, critical_frac=0.15, min_age=4), 3.0,
     "37 ticks from sentence to extrusion -- 3x the previous series"),
    ("sc_slow_94",   dict(shrink_rate=0.02, critical_frac=0.15, min_age=4), 3.0,
     "94 ticks -- a marked cell is on screen for 5% of the run"),
    ("sc_slow_189",  dict(shrink_rate=0.01, critical_frac=0.15, min_age=4), 3.0,
     "189 ticks -- the pattern persists over a tenth of the whole movie"),
    ("sc_slow_deep", dict(shrink_rate=0.02, critical_frac=0.05, min_age=4), 3.0,
     "shrink FURTHER before going: visibly dwindling rather than lingering at full size"),
    ("sc_slow_late", dict(shrink_rate=0.02, critical_frac=0.15, min_age=4), 1.5,
     "death starts two thirds in, so the blue is untouched for twice as long"),
]

INHIB_SERIES = [
    ("sc_inh_none",  dict(two_species=True, inhib=None),
     "control: the coarse species is present and growth ignores it"),
    ("sc_inh_soft",  dict(two_species=True, inhib=dict(inhib_chan=2, inhib_sw=0.65, inhib_hill=4.0)),
     "only the inhibitor's peaks stop growth"),
    ("sc_inh_mid",   dict(two_species=True, inhib=dict(inhib_chan=2, inhib_sw=0.35, inhib_hill=4.0)),
     "half the field stops growth -- red spots inside teal"),
    ("sc_inh_hard",  dict(two_species=True, inhib=dict(inhib_chan=2, inhib_sw=0.15, inhib_hill=8.0)),
     "almost everything is off; only the brightest spots grow at all"),
    # inhibition AND death, the two ways of saying no: growth stopped between the domains and the
    # cells there removed as well
    ("sc_inh_death", dict(two_species=True, inhib=dict(inhib_chan=2, inhib_sw=0.35, inhib_hill=4.0)),
     "inhibition and death together: stop growing there AND remove it"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--inhib", action="store_true", help="the inhibitor series instead")
    ap.add_argument("--slow", action="store_true", help="the slow-death series instead")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    print(f"{'spec':<18}{'death mode':<34}{'after':>7}{'ops':>5}  gate")
    bad, names = 0, []
    todo = [(t, d, k, n) for t, d, k, n in VARIANTS]
    if a.slow:
        todo = [(t, dict(mode="chem_low", chan=2, a_sw=0.45),
                 dict(two_species=True, inhib=SLOW, fast=f, after_frac=af), n)
                for t, f, af, n in SLOW_SERIES]
    elif a.inhib:
        todo = [(t, ({} if t != "sc_inh_death"
                     else dict(mode="chem_low", chan=2, a_sw=0.45)), k, n)
                for t, k, n in INHIB_SERIES]
    for tag, death, kw, note in todo:
        spec = build(tag, death, note=note, **kw)
        with open(os.path.join(CONFIG, f"{tag}.yaml"), "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
        names.append(tag)
        msg = ""
        if a.check:
            import biologist as B
            from make_basis import _unread
            fails = [r.pid for r in B.check(spec) if r.status == "fail"] + _unread(spec)
            bad += bool(fails)
            msg = "BROKEN " + ",".join(fails) if fails else "ok"
        print(f"{tag:<18}{str(death)[:33]:<34}{spec['_sculpt']['after_frame']:>7}"
              f"{len(spec['operators']):>5}  {msg}")
    print(f"\n{len(names)} specs -> {CONFIG}")
    print("  python cluster.py run " + " ".join(names) + " --frames 1800")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
