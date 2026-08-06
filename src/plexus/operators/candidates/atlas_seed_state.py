"""atlas_seed_state -- write constant initial values into named state blocks. HARNESS, NOT BIOLOGY.

Read this before you consider promoting it: **this operator was not extracted from any paper.**
It exists because a Plexus set's non-spatial state blocks start at zero and the engine has no
per-block `init:` in the spec, so a cell whose `radius` is 0 cannot grow and a cell whose
`division_rate` is 0 cannot divide. The Okuda track solves the same problem the same way
(`seed_cell_rd` seeds the chemistry before frame 1); this is its general form.

It is deliberately kept in the anti-chamber and deliberately given no `family`: it carries no
biological semantics, it must never appear in the atlas ledger as vocabulary extracted from
jax-morph, and promoting it would put a harness convenience into the biological language. If the
engine ever grows a real `init:` in the spec schema, delete this file.

Usage (with the engine's own frame gate, so it runs once and then stops):

    - { op: seed_state, at: cell, before_frame: 1, values: { radius: 0.5, growth_rate: 0.4 } }
"""
from __future__ import annotations

import torch

from plexus.models.base import Operator
from plexus.models.registry import register_operator


@register_operator("seed_state", set="cell", kind="seed")
class SeedState(Operator):
    EMIT = None                                  # writes state directly; returns no delta
    MAY_MUTATE_INTEGRATED_STATE = True           # that write is the whole point, and it is declared
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS: list = []
    WRITES: list = []                            # set per instance from `values:` (see below)
    MAPS: list = []
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["values"]
    MECHANISM_TAGS: list = []                    # none: this is not a mechanism
    PARAM_ROLES = {"values": "initial_condition"}
    REFERENCE = "Atlas harness. Not from jax-morph; see the module docstring."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.values = dict(params.get("values") or {})
        self.at = params.get("_at", "cell")
        self.WRITES = sorted(self.values)         # per-instance, so the ledger sees what it writes

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        live = lvl.occ > 0 if mask is None else (lvl.occ > 0) & mask
        st = lvl.state.clone()
        for name, v in self.values.items():
            if name not in lvl.state_schema:
                raise KeyError(
                    f"seed_state: set {self.at!r} has no state block {name!r}; declared blocks "
                    f"are {sorted(lvl.state_schema._slices)}. A silent skip here would look "
                    f"exactly like a parameter that had no effect.")
            a, b = lvl.state_schema[name]
            val = torch.as_tensor(v, dtype=st.dtype, device=st.device).reshape(-1)
            if val.numel() not in (1, b - a):
                raise ValueError(f"seed_state: {name!r} wants width {b - a}, got {val.numel()}")
            st[live, a:b] = val
        lvl.state = st
        return {}
