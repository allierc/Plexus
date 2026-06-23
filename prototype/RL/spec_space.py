"""spec_space.py -- the action space + random spec generator for inverse design.

A `Mechanism` = a fixed family of codebase operators (a base config) + a box of
continuous LEVERS over that family's parameters. `sample_u` draws a random spec;
`apply_u` maps a unit-cube vector onto a concrete `plexus.schema.Spec`, mutating
ONLY registered-operator params and the per-type properties the operators read --
never the engine. Everything runs through the codebase operators + engine.

Three 2-D testbeds:
  slime  -- single-type Physarum (deposit/diffuse/decay/sense/bounce/advance), set 'cell'
  boids  -- 4-type flocking (radius_graph + cohesion/alignment/separation), set 'particle'
  ar     -- 2-type attraction-repulsion (radius_graph + attraction_repulsion), set 'particle'

A lever is (kind, *addr, lo, hi):
  ('op',      op_name, key)         -> spec.operators[op].params[key]
  ('type',    type_name, key)       -> sets[set]['types'][type][key]   (scalar; '*' = all types)
  ('typevec', type_name, key, idx)  -> sets[set]['types'][type][key][idx]  (list element)
"""
from __future__ import annotations

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "src"))

import plexus.operators  # noqa: F401  self-register the codebase operator library
from plexus.schema import load


class Mechanism:
    def __init__(self, name, config, levers, n, frames, set_name="cell", res=None):
        self.name = name
        self.config = config
        self.levers = levers
        self.n, self.frames, self.res = n, frames, res
        self.set_name = set_name
        self.lo = np.array([l[-2] for l in levers], dtype=float)
        self.hi = np.array([l[-1] for l in levers], dtype=float)
        self.D = len(levers)

    def base_spec(self, seed=0):
        sim = load(os.path.join(_REPO, self.config))
        sim.sets[self.set_name]["n"] = int(self.n)
        sim.n_frames = int(self.frames)
        if self.res is not None and "chemical" in sim.fields:
            sim.fields["chemical"]["res"] = int(self.res)
        sim.seed = int(seed)
        return sim

    def apply_u(self, u, seed=0):
        """u in [0,1]^D -> (Spec, params). Mutates only operator params + type props."""
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        p = self.lo + u * (self.hi - self.lo)
        s = self.base_spec(seed)
        types = s.sets[self.set_name].get("types", {})
        for i, lev in enumerate(self.levers):
            kind = lev[0]
            if kind == "op":
                _, opname, key = lev[:3]
                for o in s.operators:
                    if o.op == opname:
                        o.params[key] = float(p[i])
            elif kind == "type":
                _, tname, key = lev[:3]
                for tn, td in types.items():
                    if tname == "*" or tn == tname:
                        td[key] = float(p[i])
            elif kind == "typevec":
                _, tname, key, idx = lev[:4]
                for tn, td in types.items():
                    if tname == "*" or tn == tname:
                        td[key] = list(td[key]); td[key][idx] = float(p[i])
        return s, p

    def sample_u(self, rng):
        return rng.uniform(0.0, 1.0, self.D)


SLIME = Mechanism(
    name="slime", config="config/slime/slime_random_full.yaml", set_name="cell",
    levers=[
        ("op",   "deposit", "amount", 0.010, 0.080),
        ("op",   "diffuse", "rate",   0.005, 0.040),
        ("op",   "decay",   "rate",   0.0002, 0.0040),
        ("op",   "sense",   "cross", -1.5,    0.0),
        ("type", "*", "move_speed",   0.0020, 0.0120),
        ("type", "*", "turn_speed",   0.20,   0.70),
        ("type", "*", "sensor_angle", 10.0,   45.0),
        ("type", "*", "sensor_dist",  0.015,  0.050),
    ],
    n=6000, frames=200, res=140,
)

# boids: 4 types x {cohesion, alignment, separation} = 12 levers
BOIDS = Mechanism(
    name="boids", config="config/boids/boids_rand_4t.yaml", set_name="particle",
    levers=[("type", t, k, 0.0, 100.0)
            for t in ("t00", "t01", "t02", "t03")
            for k in ("cohesion", "alignment", "separation")],
    n=1024, frames=150,
)

# attraction-repulsion: 2 types x 4-vector interaction law p = 8 levers
AR = Mechanism(
    name="ar", config="config/attraction_repulsion/arbitrary_rand_2t.yaml", set_name="particle",
    levers=[("typevec", t, "p", j, 1.0, 2.0)
            for t in ("t0", "t1") for j in range(4)],
    n=1500, frames=120,
)

# multi-type slime: same 8 levers, but `cross` (inter-type sensing weight) is now
# IDENTIFIABLE (it drives how the C field channels couple). Motion params shared across
# types (as in the configs); cross spans attract(+) and repel(-).
_MT_LEVERS = [
    ("op",   "deposit", "amount", 0.010, 0.080),
    ("op",   "diffuse", "rate",   0.005, 0.040),
    ("op",   "decay",   "rate",   0.0002, 0.0040),
    ("op",   "sense",   "cross", -1.5,    1.5),
    ("type", "*", "move_speed",   0.0020, 0.0120),
    ("type", "*", "turn_speed",   0.20,   0.70),
    ("type", "*", "sensor_angle", 10.0,   45.0),
    ("type", "*", "sensor_dist",  0.015,  0.050),
]
SLIME2 = Mechanism(name="slime2", config="config/slime/slime_two_attract.yaml",
                   set_name="cell", levers=_MT_LEVERS, n=6000, frames=200, res=140)
SLIME4 = Mechanism(name="slime4", config="config/slime/slime_four.yaml",
                   set_name="cell", levers=_MT_LEVERS, n=6000, frames=200, res=140)

MECHANISMS = {m.name: m for m in [SLIME, BOIDS, AR, SLIME2, SLIME4]}
