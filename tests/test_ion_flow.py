"""The ion-flow element, tested alone: a membrane with a hole, K+ and Cl- on both sides, a clamped voltage.

Every exp04 channel is judged on whether ions flow through it; this is the machinery that counts them
(`compartment_count`), keeps them out of the membrane except through the pore (`slab_barrier`, the
lipid beads), drives them (`field_force` on `electrolyte_conduction`'s clamped potential) and moves them
(`brownian`, `pair_potential`). If it is wrong here, no channel's flow can be trusted.

Three runs of `tools/channel_spec.build_hole` through the engine, side by side on the GPU:

    open      a 1.2 nm hole at -250 mV: cations flow IN, anions OUT, and the counted current matches
              the continuum conductance the same geometry has (the solver's own) within a factor 3
    reversed  the same hole at +250 mV: the charge moves the other way
    shut      no hole: not one ion changes side

    pytest tests/test_ion_flow.py -m engine        (deselected by default: minutes on a GPU)
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
E_C = 1.602176634e-19

pytestmark = pytest.mark.engine

RUNS = {"pyO": {"hole_nm": 1.2, "psi_mV": -250.0},
        "pyR": {"hole_nm": 1.2, "psi_mV": 250.0},
        "pyS": {"hole_nm": 0.0, "psi_mV": -250.0}}
FRAMES = 40000


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:                                            # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not _cuda():
        pytest.skip("needs a GPU")
    import channel_spec as S
    out = tmp_path_factory.mktemp("ionflow")
    gd = out / "graphs_data"
    gd.mkdir()
    from plexus.paths import graphs_data_path
    os.symlink(os.path.join(graphs_data_path(), "shapes"), gd / "shapes")      # the seeds' point clouds
    preds, procs = {}, {}
    for lab, ch in RUNS.items():
        S.version(lab, "hole", None, {**ch, "n_frames_override": FRAMES}, f"pytest ion flow {lab}")
        P = dict(S.HOLE); P.update(S.VERSIONS[lab]["change"])
        _, preds[lab] = S.build_hole(lab, P)
        cmd = [PY, "Plexus_Main.py", "-o", "generate", f"channel/exp04_v{lab}", "--output_root", str(out),
               "--no-viz", "--no-describe", "--device", "cuda:0", "--force"]
        procs[lab] = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                      env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    res = {}
    for lab, pr in procs.items():
        _, err = pr.communicate(timeout=3600)
        assert pr.returncode == 0, f"{lab}: the engine failed:\n{err.decode()[-2000:]}"
        z = np.load(out / "graphs_data" / "channel" / f"exp04_v{lab}" / "trajectory.npz")
        res[lab] = {k: np.asarray(z[f"cell__{k}"])[:, 0, 0] for k in ("nK_in", "nCl_in", "q_in", "g_channel")}
        res[lab]["pred"] = preds[lab]
    return res


def _counted_nS(r):
    T = r["pred"]["sim_time_ns"] * 1e-9
    return abs(r["q_in"][-1]) * E_C / T / (abs(r["pred"]["psi_mV"]) * 1e-3) * 1e9


def test_open_hole_carries_the_continuum_current(runs):
    r = runs["pyO"]
    assert r["q_in"][-1] > 0, "at -250 mV positive charge must move INTO the cell"
    assert r["nK_in"][-1] >= r["nK_in"][0] and r["nCl_in"][-1] <= r["nCl_in"][0], "K+ in, Cl- out"
    G_cont = r["g_channel"][0] / 1000.0                              # the solver's own, pS -> nS
    G = _counted_nS(r)
    assert 0.33 * G_cont <= G <= 3.0 * G_cont, f"counted {G:.2f} nS against the continuum {G_cont:.2f} nS"


def test_reversed_voltage_reverses_the_flow(runs):
    assert runs["pyR"]["q_in"][-1] < 0, "at +250 mV positive charge must move OUT of the cell"


def test_shut_membrane_passes_nothing(runs):
    r = runs["pyS"]
    assert r["nK_in"].min() == r["nK_in"].max() == r["nK_in"][0], "a K+ crossed a membrane with no hole"
    assert r["nCl_in"].min() == r["nCl_in"].max() == r["nCl_in"][0], "a Cl- crossed a membrane with no hole"
