"""Differential test for `cell_divide:volume_conserving` -- the PLEXUS side.

Runs the anchor's division composition through the Plexus engine over M seeds and measures the
SAME pooled per-macro-step hazard the oracle side measures, by instrumenting cell_divide's occ
before/after each call (elig = live cells at the decision; committed = the wake count that call).

Two economies, both proven here rather than assumed:
  1. Only cell_divide consumes H.rng in this spec (start positions are explicit; seed_state /
     relax / grow_radius are deterministic and draw nothing). So dropping relax + grow_radius
     leaves the division RNG stream untouched and the count trajectory BIT-IDENTICAL. Verified at
     seed 0: full composition and division-only must both give 124.
  2. p_hat is a per-step hazard, so it is independent of the engine's 41-vs-40 frame convention
     and of the initial condition -- the right invariant to diff a stochastic operator on.

Run with the plexus env, e.g.
  /workspace/.conda_envs/neural-graph-linux/bin/python atlas_jax_morph/_oracle/scripts/division_plexus.py
Writes summary.json into log/atlas/division/ alongside the run_spec evidence.
"""
import copy, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))
sys.path.insert(0, os.path.join(PLEXUS, "atlas_jax_morph"))

from run_spec import load_atlas_candidates             # registers cell_divide:volume_conserving
import plexus.operators                                # noqa: F401
load_atlas_candidates()
from plexus import engine
from plexus.schema import load

M = int(os.environ.get("M_SEEDS", "48"))
SPEC = os.path.join(PLEXUS, "config", "atlas", "division.yaml")
OUT = os.path.join(PLEXUS, "log", "atlas", "division")

_real_get_operator = engine.get_operator
_tap = {"elig": 0, "committed": 0, "calls": 0, "overflow": 0.0}

def _install_tap():
    """Wrap cell_divide so every call records (live-before, live-after) on the cell set."""
    def watched(name, impl=None):
        cls = _real_get_operator(name, impl)
        if name != "cell_divide":
            return cls
        class Tapped(cls):                                   # noqa: N801
            def forward(self, H, mask=None):
                lvl = H.level(self.at)
                before = int((lvl.occ > 0).sum().item())
                out = super().forward(H, mask)
                after = int((lvl.occ > 0).sum().item())
                _tap["elig"] += before                       # cells eligible at the decision
                _tap["committed"] += (after - before)        # daughters woken this call (no death here)
                _tap["calls"] += 1
                ov = getattr(lvl, "division_overflow", None)
                if ov is not None:
                    _tap["overflow"] = float(ov.item())
                return out
        Tapped.__name__ = f"Tapped{cls.__name__}"
        return Tapped
    engine.get_operator = watched

def _variant(sim, division_only):
    s = copy.deepcopy(sim)
    if division_only:
        keep = {"seed_state", "cell_divide"}
        s.operators = [o for o in s.operators if o.op in keep]
        s.schedule = [t for t in s.schedule if t in keep]
    return s

def final_count(sim, seed, division_only):
    s = _variant(sim, division_only)
    s.seed = seed
    _, out = engine.run(s, out_path=None, device="cpu", progress=False)
    occ = out["sets"]["cell"]["occ"]          # [n_rec, N] bool, post-schedule per recorded tick
    return int(occ[-1].sum())

def main():
    base = load(SPEC)
    _install_tap()

    # (1) prove the division-only reduction is count-neutral at seed 0 (both must be 124).
    full0 = final_count(base, 0, division_only=False)
    tap_after_full = dict(_tap)
    div0 = final_count(base, 0, division_only=True)
    print(f"seed 0: full-composition final={full0}   division-only final={div0}   "
          f"(anchor/flagged = 124)", flush=True)
    reduction_ok = (full0 == div0 == 124)

    # (2) reset the tap and pool over M division-only seeds.
    for k in _tap:
        _tap[k] = 0 if k != "overflow" else 0.0
    finals = np.zeros(M, dtype=np.int64)
    t0 = time.time()
    for seed in range(M):
        finals[seed] = final_count(base, seed, division_only=True)
        if seed < 3 or seed % 12 == 0:
            print(f"seed {seed}: final={finals[seed]}  [{time.time()-t0:.1f}s]", flush=True)

    p_hat = _tap["committed"] / _tap["elig"]
    p_theory = float(-np.expm1(-0.08 * 1.0))
    summary = {
        "role": "plexus", "operator": "cell_divide:volume_conserving",
        "M_seeds": M, "reduction_count_neutral_at_seed0": reduction_ok,
        "full_seed0_final": full0, "divonly_seed0_final": div0,
        "p_theory": p_theory,
        "p_hat": p_hat, "divisions_total": _tap["committed"],
        "eligible_cellsteps_total": _tap["elig"], "div_calls_per_run": _tap["calls"] // M,
        "p_hat_se": float(np.sqrt(p_hat * (1 - p_hat) / _tap["elig"])),
        "division_overflow_last": _tap["overflow"],
        "final_count_mean": float(finals.mean()), "final_count_std": float(finals.std(ddof=1)),
        "final_count_min": int(finals.min()), "final_count_max": int(finals.max()),
        "final_counts": finals.tolist(),
        "plexus_124_in_sd": float((124 - finals.mean()) / finals.std(ddof=1)),
        "smoke_82_in_sd": float((82 - finals.mean()) / finals.std(ddof=1)),
        "wall_s": round(time.time() - t0, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "diff_plexus_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
