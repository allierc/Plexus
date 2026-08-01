"""Differential score for `morphogen:free_space_greens_function` -- torch vs the jax reference.

Reads _oracle/runs/diff_free_screened_diffusion/reference.npz (the fixed matched states + the
reference field), runs the Plexus candidate operator on the IDENTICAL states, and compares the
written `chemical` field. This is an ISOLATED field diff: the operator is a quasistatic pure
state->field map (dt/key ignored, no integrator, moves no cell), so there is nothing to conflate --
the number is pure algorithm agreement.

    python diff_free_screened_diffusion.py        # writes diff.json into the oracle run dir

METRIC (fixed here, before the run): value = max over all configs, all cells (live AND dead), all
species of the relative error |c_plx - c_ref| / max(|c_ref|, FLOOR), FLOOR=1e-6. Gated on finite +
every dead slot exactly 0.0 on the Plexus side. Threshold below.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))

import plexus.operators.candidates.jax_morph_free_screened_diffusion as _m  # noqa: F401 registers morphogen
from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator
from plexus.models.state import (
    Block, StateSchema, BOUNDARY_FREE, BOUNDARY_WORLD, NONE, SECOND_ORDER_COORDINATE,
)

RUN = os.path.join(HERE, "_oracle", "runs", "diff_free_screened_diffusion")
FLOOR = 1e-6
THRESHOLD = 1.0e-3            # relative; justified in the atlas record's evidence.threshold


def build_hierarchy(pos, radius, alive, secretion):
    """The identical fixed matched state, as a torch cell set (mirrors the property test's _cell)."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    N, Dd = pos.shape
    rad = torch.as_tensor(radius, dtype=torch.float32).reshape(N, 1)
    S = torch.as_tensor(secretion, dtype=torch.float32)
    if S.dim() == 1:
        S = S[:, None]
    ns = S.shape[1]
    blocks = [
        Block("pos", Dd, role="coordinate", integration=SECOND_ORDER_COORDINATE, boundary=BOUNDARY_WORLD),
        Block("radius", 1, integration=NONE, boundary=BOUNDARY_FREE),
        Block("secretion_rate", ns, integration=NONE, boundary=BOUNDARY_FREE),
        Block("chemical", ns, integration=NONE, boundary=BOUNDARY_FREE),
    ]
    state = torch.cat([pos, rad, S, torch.zeros(N, ns)], dim=1)
    lvl = Level("cell", state=state, state_schema=StateSchema(blocks))
    alive = torch.as_tensor(np.asarray(alive, bool))
    lvl.occ[~alive] = 0.0                        # kill the dead slots (the reference `alive` mask)
    H = Hierarchy()
    H.add_level(lvl)
    return H


def run_plexus(cfg):
    """Plexus steady field c_plx [cap, ns] for one config (float32)."""
    H = build_hierarchy(cfg["pos"], cfg["radius"], cfg["alive"], cfg["secretion"])
    params = dict(_at="cell", n_space_dim=int(cfg["n_space_dim"]),
                  diffusion=cfg["diffusion"], degradation=cfg["degradation"])
    op = get_operator("morphogen", "free_space_greens_function")(params, "cpu")
    op(H, None)
    return H.level("cell").get("chemical").detach().cpu().numpy().astype(np.float32)


def main():
    z = np.load(os.path.join(RUN, "reference.npz"), allow_pickle=False)
    names = json.load(open(os.path.join(RUN, "summary.json")))["configs"]

    per = {}
    worst_rel = 0.0
    worst_abs = 0.0
    all_finite = True
    dead_leak = []
    for name in names:
        cfg = dict(
            pos=z[f"{name}_pos"], radius=z[f"{name}_radius"], alive=z[f"{name}_alive"],
            secretion=z[f"{name}_secretion"],
            diffusion=z[f"{name}_diffusion"].tolist(),      # 0-d -> float, 1-d -> per-species list
            degradation=z[f"{name}_degradation"].tolist(),
            n_space_dim=int(z[f"{name}_n_space_dim"]),
        )
        c_ref = z[f"{name}_c_ref"].astype(np.float32)
        c_plx = run_plexus(cfg)
        finite = bool(np.isfinite(c_plx).all())
        all_finite = all_finite and finite

        abs_err = np.abs(c_plx - c_ref)
        rel_err = abs_err / np.maximum(np.abs(c_ref), FLOOR)
        max_abs = float(abs_err.max())
        max_rel = float(rel_err.max())
        idx = np.unravel_index(int(np.argmax(rel_err)), rel_err.shape)

        live = np.asarray(cfg["alive"], bool)
        dead = ~live
        dead_max_plx = float(np.abs(c_plx[dead]).max()) if dead.any() else 0.0
        if dead_max_plx != 0.0:
            dead_leak.append(name)

        worst_rel = max(worst_rel, max_rel)
        worst_abs = max(worst_abs, max_abs)
        per[name] = {
            "n_space_dim": int(cfg["n_space_dim"]), "n_live": int(live.sum()),
            "n_species": int(c_ref.shape[1]),
            "max_abs_err": max_abs, "max_rel_err": max_rel,
            "argmax_rel_cell": int(idx[0]), "argmax_rel_species": int(idx[1]),
            "c_ref_at_argmax": float(c_ref[idx]), "c_plx_at_argmax": float(c_plx[idx]),
            "c_ref_live_max": float(np.abs(c_ref[live]).max()),
            "dead_slots_field_max_abs_plexus": dead_max_plx,
            "finite": finite,
        }

    passed = bool(worst_rel < THRESHOLD and all_finite and not dead_leak)
    diff = {
        "operator": "morphogen:free_space_greens_function",
        "reference_run": "diff_free_screened_diffusion",
        "metric": "max relative error |c_plx - c_ref| / max(|c_ref|, 1e-6) over all cells/species/configs",
        "floor": FLOOR, "threshold": THRESHOLD,
        "value": worst_rel, "max_abs_err_overall": worst_abs,
        "all_finite": all_finite, "dead_slot_leaks": dead_leak,
        "passed": passed, "per_config": per,
    }
    with open(os.path.join(RUN, "diff.json"), "w") as f:
        json.dump(diff, f, indent=2)
    print(json.dumps(diff, indent=2))
    print(f"\n{'PASS' if passed else 'FAIL'}: value={worst_rel:.3e}  "
          f"(max_abs={worst_abs:.3e})  threshold={THRESHOLD:.1e}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
