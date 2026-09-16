"""The task spec language -- a SEPARATE language from `plexus.schema`, deliberately.

A model spec declares sets, fields, operators and a schedule. A task spec declares none of those:
it has a stimulus, a teacher, a grid of conditions and a set of splits, and it names no entity at
all. Threading a `task:` branch through `plexus.schema.load` would put a second grammar inside
every validation path of the first, for two languages that share not one key. So: two loaders.

    general      name, duration_s, dt, channels, and the seed discipline
    stimulus     process: <registered name>, plus its parameters
    teacher      law: <registered name>, plus its parameters
    conditions   a dict of lists, CROSSED -- the grid a task is a family over
    splits       per split: n_per_cond and seed0

WHAT A CONDITION IS, because it is the key that makes this a task and not a dataset. Each entry
of `conditions:` names a parameter of the stimulus or of the teacher and lists the values it
takes; the grid is their full cross product, and every split draws `n_per_cond` trials of EACH
cell. So a task is a family of related problems with a stated structure, and a result can be read
per cell -- "it fails on fast targets with sharp turns" -- rather than only in aggregate. The
oculomotor corpus is 2 shapes x 2 motions x 3 speeds x 2 angles = 24 cells at 150 trials each.

SEEDS ARE PER TRIAL AND DERIVED, NOT DRAWN. Trial i of a split is seeded `seed0 + i`, so a split
is a pure function of (spec, seed0) and two splits with different `seed0` are disjoint by
construction rather than by a check afterwards. This is the same discipline the reference corpus
uses, and it is why regenerating a corpus on another machine gives the same data.
"""
from __future__ import annotations

import itertools
import os

import yaml

from plexus.tasks import get_stimulus, get_teacher

REQUIRED = ("general", "stimulus", "teacher", "splits")


class TaskSpec:
    """A validated task declaration. Nothing here runs; `generate.py` does that."""

    def __init__(self, raw, path=None):
        self.raw, self.path = raw, path
        g = raw["general"]
        self.name = str(g["name"])
        self.duration_s = float(g["duration_s"])
        self.dt = float(g["dt"])
        self.channels = int(g.get("channels", 1))
        self.targets = int(g.get("targets", self.channels))
        self.T = int(round(self.duration_s / self.dt))
        self.stimulus = dict(raw["stimulus"])
        self.teacher = dict(raw["teacher"])
        self.process_name = self.stimulus.pop("process")
        self.law_name = self.teacher.pop("law")
        self.conditions = {k: list(v) for k, v in (raw.get("conditions") or {}).items()}
        self.splits = {k: dict(v) for k, v in raw["splits"].items()}

    # -- the grid ---------------------------------------------------------- #
    @property
    def cells(self) -> list:
        """The full cross product of `conditions:`, as a list of dicts. `[{}]` when there is no
        grid, so a task with no conditions is the one-cell case and needs no special path."""
        if not self.conditions:
            return [{}]
        keys = sorted(self.conditions)
        return [dict(zip(keys, vals)) for vals in
                itertools.product(*(self.conditions[k] for k in keys))]

    def n_trials(self, split) -> int:
        return len(self.cells) * int(self.splits[split]["n_per_cond"])

    def describe(self) -> str:
        cells = len(self.cells)
        parts = [f"{s}: {self.n_trials(s)} trials" for s in sorted(self.splits)]
        return (f"{self.name}: {cells} condition cell{'s' if cells != 1 else ''} x "
                f"{self.T} frames ({self.duration_s} s at dt={self.dt}); " + ", ".join(parts))


def load_task(path) -> TaskSpec:
    """Read and validate a task yaml. Every refusal below is a defect that would otherwise show
    up as a corpus that runs and means something other than what it says."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a task spec is a mapping, got {type(raw).__name__}")
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing required block(s) {missing}. A task spec needs "
                         f"{list(REQUIRED)}; `conditions:` is optional.")
    for k in ("name", "duration_s", "dt"):
        if k not in raw["general"]:
            raise ValueError(f"{path}: general needs `{k}:`")
    if "process" not in raw["stimulus"]:
        raise ValueError(f"{path}: stimulus needs `process:` naming a registered input ensemble")
    if "law" not in raw["teacher"]:
        raise ValueError(f"{path}: teacher needs `law:` naming a registered teacher")

    spec = TaskSpec(raw, os.path.abspath(path))
    get_stimulus(spec.process_name)                    # raises with the registered names
    get_teacher(spec.law_name)
    if spec.T < 2:
        raise ValueError(f"{path}: duration_s / dt is {spec.T} frames; a trial needs at least 2")
    for s, cfg in spec.splits.items():
        for k in ("n_per_cond", "seed0"):
            if k not in cfg:
                raise ValueError(f"{path}: split {s!r} needs `{k}:`")

    # DISJOINTNESS IS CHECKED, NOT ASSUMED. Trial i of a split is seeded seed0 + i, so two
    # splits overlap exactly when their seed ranges do -- and a corpus whose validation set
    # shares trials with its training set reports a number that means nothing, silently.
    ranges = {s: (int(c["seed0"]), int(c["seed0"]) + spec.n_trials(s)) for s, c in
              spec.splits.items()}
    for a, b in itertools.combinations(sorted(ranges), 2):
        (lo_a, hi_a), (lo_b, hi_b) = ranges[a], ranges[b]
        if lo_a < hi_b and lo_b < hi_a:
            raise ValueError(
                f"{path}: splits {a!r} [{lo_a}, {hi_a}) and {b!r} [{lo_b}, {hi_b}) share seeds, "
                f"so they share TRIALS. Move a `seed0:` apart by at least the trial count.")

    # A condition must name a parameter something actually reads, or it silently does nothing
    # and the grid is a lie about what was varied.
    known = set(spec.stimulus) | set(spec.teacher) | {
        "process", "law", "amplitude", "cutoff_hz", "order", "family", "band", "tau_s",
        "seconds", "speed", "f_max_hz", "f_min_hz", "hold_s", "turn_rate_hz", "gain", "k"}
    unknown = [k for k in spec.conditions if k not in known]
    if unknown:
        raise ValueError(
            f"{path}: conditions {unknown} name no parameter of stimulus "
            f"{spec.process_name!r} or teacher {spec.law_name!r}. A condition that reaches "
            f"nothing makes the grid a claim about variation that did not happen.")
    return spec
