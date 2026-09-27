"""`cell_divide` refuses, counted, when the CELL buffer is full -- it used to raise instead.

exp12 round 5b (2026-09-26): a flat apico-basal disc carries ~2 vertices a cell, so the cell set's rows
ran out before the vertex buffer and `set_cell_block` raised "expanded size (20000) must match (20227)".

    PYTHONPATH=src python -m pytest tests/test_divide_cell_buffer.py -q
"""
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _ready_disc(tmp_path, cell_rows):
    import torch
    from plexus import engine, schema
    from plexus.models.registry import get_operator
    from plexus.operators.vertex_ops import set_cell_block
    d = yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", "exp12_disc_v5.yaml")))
    d["sets"]["cell"]["n"] = cell_rows
    p = tmp_path / "disc.yaml"
    yaml.safe_dump(d, open(p, "w"))
    sim = schema.load(str(p))
    H = engine.build(sim, device="cpu")
    engine.seed(H, sim, device="cpu")
    ops = {o.op: get_operator(o.op, variant=o.impl)({**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, "cpu")
           for o in sim.operators}
    m = H.level("vertex")._mesh
    nF = int(m["nF"])
    m["age"] = torch.full((nF,), 10.0)
    set_cell_block(H, "cell", "Vbirth", np.zeros(nF), nF)        # every cell ready to divide
    H.frame = 100
    return H, ops, m, nF


def test_a_full_cell_buffer_refuses_instead_of_raising(tmp_path):
    H, ops, m, nF = _ready_disc(tmp_path, cell_rows=460)        # 446 seeded cells: room for 14
    ops["cell_divide"](H)
    assert int(m["nF"]) == nF and m["buf_full"] and m["div_blocked"] > 0


def test_room_in_both_buffers_divides_as_before(tmp_path):
    H, ops, m, nF = _ready_disc(tmp_path, cell_rows=12000)
    ops["cell_divide"](H)
    assert int(m["nF"]) > nF and not m["buf_full"]
