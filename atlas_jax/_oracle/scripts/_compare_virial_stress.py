"""Compute the mechanosense <-> VirialStress differential metric (Plexus/torch side).

PRIMARY (through the engine): read the `stress` block run_spec.py recorded (frame 0) from the
engine's zarr, compare cell-for-cell to the oracle's VirialStress(Morse) reference stress on the
same byte-identical IC. Metric: max_abs and peak-normalized rel over LIVE cells; dead-slot masking
checked separately.

SUPPLEMENTARY (direct operator, the paths the uniform-radius engine run cannot reach): instantiate
mechanosense directly on (a) the SAME uniform IC for the other four pair laws, and (b) a second
UNEQUAL-RADII IC for all five laws -- exercising sigma = r_i + r_j and the per-cell d-ball V_i.
Compare to the oracle's stress_<law> / vr_stress_<law> arrays.

Run with the Plexus python (torch), NOT through oracle.py:
    $PY _oracle/scripts/_compare_virial_stress.py
"""
import json
import os
import sys

import numpy as np
import torch
import zarr

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))

from plexus.models.base import Hierarchy, Level                       # noqa: E402
from plexus.models.registry import get_operator                      # noqa: E402
import plexus.operators.candidates.jax_morph_virial_stress  # noqa: E402,F401  (registers mechanosense)

REF = os.path.join(HERE, "..", "runs", "diff_virial_stress", "reference.npz")
ZARR = "/groups/saalfeld/home/allierc/GraphData/graphs_data/atlas/virial_stress/simulation.zarr"
LAW_EPS = {"morse": 3.0, "soft_sphere": 1.0, "hertzian": 1.0, "harmonic": 1.0, "lennard_jones": 1.0}

ref = np.load(REF)
N, CAP = int(ref["N"]), int(ref["CAP"])
live = ref["alive"]                                                   # [CAP] bool


def metric(plexus, reference, mask):
    """max_abs over masked cells + peak-normalized rel (peak = max|reference| over masked cells)."""
    d = np.abs(plexus[mask] - reference[mask])
    peak = float(np.abs(reference[mask]).max())
    max_abs = float(d.max())
    return max_abs, peak, (max_abs / peak if peak > 0 else 0.0)


def run_op(pos, radius, alive, law, eps):
    """Run mechanosense directly on a one-set world (radius as a per-cell buffer, stress as a block)."""
    n = pos.shape[0]
    state = torch.zeros(n, 5)                                         # pos(2) vel(2) stress(1)
    schema = {"pos": (0, 2), "vel": (2, 4), "stress": (4, 5)}
    state[:, :2] = torch.as_tensor(pos, dtype=torch.float32)
    occ = torch.as_tensor(alive.astype(np.float32))
    lvl = Level("cell", state=state, state_schema=schema, occ=occ)
    lvl.register_buffer("radius", torch.as_tensor(radius, dtype=torch.float32))
    H = Hierarchy(); H.add_level(lvl); H.dim = 2
    op = get_operator("mechanosense")({"potential": law, "epsilon": eps}, "cpu")
    out = op(H, lvl.active)
    assert out == {}, "mechanosense must return {} (pure sensing)"
    return lvl.get("stress")[:, 0].detach().numpy()


report = {}

# ---- PRIMARY: engine-recorded Morse stress vs oracle -------------------------------------------- #
zstress = zarr.open_group(ZARR, mode="r")["cell"]["state"]["stress"][0, :, 0]   # frame 0, [CAP]
ref_morse = ref["stress_morse"]
# all recorded frames identical (sensor moves nothing)?
allf = zarr.open_group(ZARR, mode="r")["cell"]["state"]["stress"][:, :, 0]
frames_identical = bool(np.allclose(allf, allf[0], atol=1e-7))
max_abs, peak, rel = metric(zstress, ref_morse, live)
dead_absmax_plexus = float(np.abs(zstress[~live]).max())
dead_absmax_ref = float(np.abs(ref_morse[~live]).max())
report["primary_engine_morse"] = {
    "max_abs": max_abs, "peak_ref": peak, "rel": rel,
    "dead_absmax_plexus": dead_absmax_plexus, "dead_absmax_ref": dead_absmax_ref,
    "frames_identical": frames_identical,
    "pass_rel_1e-4": bool(rel <= 1e-4),
    "pass_dead_1e-6": bool(max(dead_absmax_plexus, dead_absmax_ref) <= 1e-6),
}

# ---- SUPPLEMENT A: all five laws on the SAME uniform IC (direct operator) ----------------------- #
report["supp_uniform_laws"] = {}
for law, eps in LAW_EPS.items():
    p = run_op(ref["p0"], ref["radius"], live, law, eps)
    ma, pk, rl = metric(p, ref[f"stress_{law}"], live)
    report["supp_uniform_laws"][law] = {"max_abs": ma, "peak_ref": pk, "rel": rl,
                                        "pass_rel_1e-4": bool(rl <= 1e-4)}

# ---- SUPPLEMENT B: unequal radii, all five laws (sigma=r_i+r_j, per-cell V_i) ------------------- #
vr_live = ref["vr_alive"]
report["supp_unequal_radii_laws"] = {}
for law, eps in LAW_EPS.items():
    p = run_op(ref["vr_p0"], ref["vr_radius"], vr_live, law, eps)
    ma, pk, rl = metric(p, ref[f"vr_stress_{law}"], vr_live)
    report["supp_unequal_radii_laws"][law] = {"max_abs": ma, "peak_ref": pk, "rel": rl,
                                              "pass_rel_1e-4": bool(rl <= 1e-4)}

print(json.dumps(report, indent=2))
with open(os.path.join(HERE, "..", "runs", "diff_virial_stress", "compare.json"), "w") as f:
    json.dump(report, f, indent=2)

# overall verdict
prim = report["primary_engine_morse"]
supp_ok = all(v["pass_rel_1e-4"] for v in report["supp_uniform_laws"].values()) and \
          all(v["pass_rel_1e-4"] for v in report["supp_unequal_radii_laws"].values())
print("\nPRIMARY pass (rel<=1e-4 & dead<=1e-6 & frames identical):",
      prim["pass_rel_1e-4"] and prim["pass_dead_1e-6"] and prim["frames_identical"])
print("SUPPLEMENT pass (all 10 direct-operator law/radii checks rel<=1e-4):", supp_ok)
