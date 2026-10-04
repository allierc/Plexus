"""`cell_divide[model: postmitotic]` (src/plexus/operators/vertex_ops.py, exp 10).

WHY THEY EXIST. Serra et al. 2019 (Nature 569:66) grow the organoid from one cell; its first symmetry
break is a Delta-high, Notch-low cell that becomes a POST-MITOTIC Paneth cell. The variant is the
default size rule with those cells struck from the division candidates:

    vetoed_j = chem_j[veto_col] > veto_th  (and chem_j[veto_below_col] < veto_below_th, if set)

These tests pin: it is registered; with nothing vetoed it IS the default model, down to the recorded
trajectory of a 40-frame exp10_c2c run; and a planted post-mitotic cell never divides. CPU only.

    PYTHONPATH=src python -m pytest tests/test_cell_divide_postmitotic.py -q
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import plexus.operators  # noqa: F401,E402  registers every operator
from plexus.models.registry import get_operator  # noqa: E402
from plexus.operators.vertex_ops import (Divide3D, Divide3DPostmitotic, resolve_cell_set,  # noqa: E402
                                         set_cell_block)

SPEC = os.path.join(ROOT, "config", "tissue", "exp10_c2c.yaml")


def _divide_params(sim):
    o = next(o for o in sim.operators if o.op == "cell_divide")
    return {**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}


def _ready_c2c(tmp_path):
    """exp10_c2c built and seeded on CPU, with every one of its 12 cells ripe to divide."""
    from plexus import engine, schema
    p = tmp_path / "c2c.yaml"
    yaml.safe_dump(yaml.safe_load(open(SPEC)), open(p, "w"))
    sim = schema.load(str(p))
    H = engine.build(sim, device="cpu")
    engine.seed(H, sim, device="cpu")
    m = H.level("vertex")._mesh
    nF = int(m["nF"])
    m["age"] = torch.full((nF,), 10.0)                          # past min_cycle 4 (calls since birth)
    set_cell_block(H, "cell", "divjit", np.zeros(nF), nF)       # threshold 0 x v_ref: every cell ripe
    H.frame = 100                                               # past any settling window
    return H, sim, m, nF


def _plant_chem(H, nF, rows, col, value):
    chem = H.level("cell").get("chem")
    chem[:nF, col] = 0.0
    chem[rows, col] = value


def test_registered_and_resolvable():
    cls = get_operator("cell_divide", model="postmitotic")
    assert cls is Divide3DPostmitotic and issubclass(cls, Divide3D)
    assert get_operator("cell_divide") is Divide3D


def test_max_cycle_backstop_is_refused():
    with pytest.raises(ValueError):
        Divide3DPostmitotic({"max_cycle": 20}, "cpu")
    with pytest.raises(ValueError):
        Divide3DPostmitotic({"veto_below_col": 0}, "cpu")


def test_nothing_vetoed_takes_the_default_path(tmp_path):
    H, sim, m, nF = _ready_c2c(tmp_path)
    op = Divide3DPostmitotic(_divide_params(sim), "cpu")
    op.cell_set = resolve_cell_set(H, op.at, None)
    _plant_chem(H, nF, [], 1, 0.0)
    assert op._ready_mask(H, m, nF) is None                     # Delta 0 everywhere
    _plant_chem(H, nF, [0, 3], 1, 1.0)
    op_hi = Divide3DPostmitotic({**_divide_params(sim), "veto_th": 1e9}, "cpu")
    op_hi.cell_set = op.cell_set
    assert op_hi._ready_mask(H, m, nF) is None                  # threshold above every cell
    op_nb = Divide3DPostmitotic({**_divide_params(sim), "veto_block": "nope"}, "cpu")
    op_nb.cell_set = op.cell_set
    assert op_nb._ready_mask(H, m, nF) is None                  # undeclared block: announced, no veto


def test_planted_postmitotic_cells_are_not_candidates(tmp_path):
    H, sim, m, nF = _ready_c2c(tmp_path)
    _plant_chem(H, nF, [0, 3], 1, 1.0)                          # Delta high on cells 0 and 3
    _plant_chem(H, nF, [3], 0, 1.0)                             # ... but cell 3 is also Notch high
    op = Divide3DPostmitotic(_divide_params(sim), "cpu")
    op.cell_set = resolve_cell_set(H, op.at, None)
    mask = op._ready_mask(H, m, nF)
    assert mask is not None and not mask[0] and not mask[3] and mask.sum() == nF - 2
    op_w = Divide3DPostmitotic({**_divide_params(sim), "veto_below_col": 0, "veto_below_th": 0.5}, "cpu")
    op_w.cell_set = op.cell_set
    mask_w = op_w._ready_mask(H, m, nF)                         # only the Notch-low winner is vetoed
    assert not mask_w[0] and mask_w[3] and mask_w.sum() == nF - 1

    op(H)                                                       # the real division
    assert int(m["nF"]) == 2 * nF - 2                           # 12 cells, 10 divide
    ndiv = m["ndiv"].cpu().numpy()[:nF]
    assert ndiv[0] == 0 and ndiv[3] == 0 and (np.delete(ndiv, [0, 3]) == 1).all()
    assert m["div_vetoed"] == 2


def _run(tmp_path, name, divide_extra):
    d = yaml.safe_load(open(SPEC))
    d["general"]["n_frames"] = 40
    d["general"]["name"] = name
    for o in d["operators"]:
        if o["op"] == "cell_divide":
            o["factor"] = 1.0                                   # divisions within 40 frames (12 -> 27 cells)
            o.update(divide_extra)
    cfg = os.path.join(ROOT, "config", "tissue", f"{name}.yaml")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    try:
        yaml.safe_dump(d, open(cfg, "w"), sort_keys=False)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.path.join(ROOT, "src")}
        subprocess.run([sys.executable, os.path.join(ROOT, "Plexus_Main.py"), "-o", "generate",
                        f"tissue/{name}", "--device", "cpu", "--force", "--no-viz", "--no-describe",
                        "--output_root", str(out)], cwd=ROOT, env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        if os.path.exists(cfg):
            os.remove(cfg)
    return np.load(out / "graphs_data" / "tissue" / name / "trajectory.npz")


def test_run_with_nothing_vetoed_is_the_default_model(tmp_path):
    a = _run(tmp_path, "zz_test_pm_default", {})
    b = _run(tmp_path, "zz_test_pm_veto_off", {"model": "postmitotic", "veto_th": 1e9})
    occ_a = a["cell__occ"].reshape(a["cell__occ"].shape[0], -1).sum(1)
    occ_b = b["cell__occ"].reshape(b["cell__occ"].shape[0], -1).sum(1)
    assert occ_a[-1] > occ_a[0]                                 # the run did divide
    assert np.array_equal(occ_a, occ_b)
    assert np.array_equal(a["cell__chem"], b["cell__chem"])     # and bit-identically so
