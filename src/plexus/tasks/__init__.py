"""Tasks: the (stimulus, target) pairs a circuit is fitted to.

A task is declared, generated once, and frozen -- the same discipline the operator library uses
for mechanisms, applied to data. What a task spec names is four things:

    stimulus     a parameterised random process: what goes IN
    teacher      a law: the target is this law applied to the stimulus
    conditions   a grid, crossed, so a task is a family and not one trial
    splits       seeded and disjoint

and nothing else. In particular a task spec names NO SET, NO OPERATOR AND NO CIRCUIT. That is
the property worth protecting rather than a simplification: it means a corpus can be generated,
inspected and argued about before the circuit that will be fitted to it exists, and it means one
corpus scores several circuits without being regenerated. The connectome-gnn generator this is
modelled on has the same property, and records the circuit only as a hash in a provenance file
written beside the data.

WHY THE TARGET IS NOT GENERATED. It is COMPUTED: `y = teacher(u)`. So the only free choice in
making a task is which input ensemble to draw, which collapses a vague question ("how do we
generate all kinds of task?") into a short closed list -- the one system identification settled
decades ago, and the one `processes.py` implements.

THE TEACHER IS THE SCIENTIFIC CLAIM, AND THAT IS WHY IT IS A REGISTRY. "The target is the leaky
integral of velocity with tau = 8 s" is the hypothesis under test. If it were an anonymous
callable, a failed fit would be uninterpretable -- one could not tell whether the circuit cannot
integrate or whether the generator computed a different integral. A registered, named law with
its parameters in the spec keeps the claim readable, comparable between tasks, and checkable
against the fit afterwards.

Two registries, because the two halves are varied independently and by different people: one may
ask what a circuit does under white noise and under a chirp with the teacher fixed, or hold the
stimulus and vary the law.

    register_stimulus(name)   a process: (rng, T, dt, channels, **p) -> [T, C]
    register_teacher(name)    a law:     (u [N, T, C], dt, **p)      -> [N, T, K]

Modules:

    processes.py   impulse, step, chirp, band_limited_noise, prbs, trajectory
    lti.py         laplace (num/den), and the named filter families over scipy
    schema.py      the task spec loader -- a SEPARATE language from plexus.schema
    generate.py    spec -> stimulus.zarr + target.zarr + provenance.json
"""
from __future__ import annotations

from typing import Callable

_STIMULI: dict[str, Callable] = {}
_TEACHERS: dict[str, Callable] = {}


def _register(table, label):
    def deco(name, **meta):
        def wrap(fn):
            if name in table:
                raise ValueError(
                    f"{label} {name!r} is already registered by "
                    f"{getattr(table[name], '__module__', '?')}. Two laws under one name is the "
                    f"defect a registry exists to prevent -- a spec naming it would get whichever "
                    f"module imported last.")
            fn.task_name = name
            fn.task_meta = dict(meta)
            table[name] = fn
            return fn
        return wrap
    return deco


register_stimulus = _register(_STIMULI, "stimulus process")
register_teacher = _register(_TEACHERS, "teacher law")


def _get(table, name, label):
    if name not in table:
        raise KeyError(
            f"unknown {label} {name!r}. Registered: {sorted(table)}. A task may only name a law "
            f"that exists -- the alternative is a spec that describes a corpus nobody can "
            f"regenerate.")
    return table[name]


def get_stimulus(name):
    return _get(_STIMULI, name, "stimulus process")


def get_teacher(name):
    return _get(_TEACHERS, name, "teacher law")


def stimuli() -> list:
    return sorted(_STIMULI)


def teachers() -> list:
    return sorted(_TEACHERS)


from plexus.tasks import processes    # noqa: E402,F401  self-registers the input ensembles
from plexus.tasks import lti          # noqa: E402,F401  self-registers the LTI teacher laws

__all__ = ["register_stimulus", "register_teacher", "get_stimulus", "get_teacher",
           "stimuli", "teachers", "processes", "lti"]
