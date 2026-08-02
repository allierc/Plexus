"""Differential test for `reorient` -- the PLEXUS side.

Runs the same free active-Brownian gas (config/atlas/active_brownian_dynamics2_d.yaml: reorient +
glide, NoForce, no translational noise) through the Plexus engine and measures the SAME ensemble
orientational autocorrelation the oracle side measures, C(t) = <e(t).e(0)> over all N cells, then
diffs the two curves.

Two things this script does that run_spec cannot:
  1. HEADING TAP -- run_spec records positions, not headings, and its acted ledger cannot see a
     heading-buffer write (so it flags `reorient` INERT, a known blind-spot for heading-steerers).
     Here an on_frame hook snapshots `cell.heading` every tick, and a wrapper around `reorient`
     records whether it changed the heading on every call -- the HONEST acted ledger for the leg
     under test (Var(dtheta) > 0, every call moves the heading).
  2. THE METRIC -- C(t) from the tapped heading trajectory, diffed frame-for-frame against the
     oracle (reference.npz / summary.json). `reorient` is gated after_frame=1, so tick 0 is the
     pristine initial heading and tick t is exactly t rotational-diffusion steps -- frame-aligned
     to the oracle's s_0..s_40.

Run with the plexus env:
  /workspace/.conda_envs/neural-graph-linux/bin/python \
      atlas_jax_morph/_oracle/scripts/active_brownian_dynamics2_d_plexus.py
Writes diff_plexus_summary.json into log/atlas_jax/active_brownian_dynamics2_d/ (beside run_spec's
evidence), and prints the PASS/FAIL verdict against the pre-registered threshold 0.05.
"""
import json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(PLEXUS, "src"))
sys.path.insert(0, os.path.join(PLEXUS, "atlas_jax_morph"))

from run_spec import load_atlas_candidates             # registers `reorient`
import plexus.operators                                # noqa: F401
load_atlas_candidates()
from plexus import engine
from plexus.schema import load

SPEC = os.path.join(PLEXUS, "config", "atlas", "active_brownian_dynamics2_d.yaml")
ORACLE_DIR = os.path.join(HERE, "..", "runs", "diff_active_brownian_dynamics2_d")
OUT = os.path.join(PLEXUS, "log", "atlas_jax", "active_brownian_dynamics2_d")
DT, D_R, THRESHOLD = 1.0, 0.1, 0.05


# --- honest acted ledger for `reorient`: did each call actually move the heading? ------------- #
_real_get_operator = engine.get_operator
_tap = {"calls": 0, "acted": 0}

def _install_reorient_tap():
    def watched(name, impl=None):
        cls = _real_get_operator(name, impl)
        if name != "reorient":
            return cls
        class Tapped(cls):                                   # noqa: N801
            def forward(self, H, mask=None):
                before = H.level(self.at).heading.clone()
                out = super().forward(H, mask)
                after = H.level(self.at).heading
                _tap["calls"] += 1
                if float((after - before).norm(dim=-1).max()) > 0.0:
                    _tap["acted"] += 1
                return out
        Tapped.__name__ = f"Tapped{cls.__name__}"
        return Tapped
    engine.get_operator = watched


def _signed_angle(a, b):
    """Signed rotation angle from unit vector a to unit vector b, per cell, in (-pi, pi]."""
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    return np.arctan2(cross, dot)


def main():
    sim = load(SPEC)
    _install_reorient_tap()

    heading_frames = []
    def capture(H, tick):
        heading_frames.append(H.level("cell").heading.detach().cpu().numpy().copy())

    t0 = time.time()
    _, _out = engine.run(sim, out_path=None, device="cpu", on_frame=capture, progress=False)
    wall = time.time() - t0

    head = np.stack(heading_frames, axis=0)                 # [T+1, N, 2]; head[t] = t reorient steps
    T = head.shape[0] - 1
    e0 = head[0]
    C = np.einsum("tnd,nd->tn", head, e0).mean(axis=1)      # [T+1] orientational autocorrelation

    dtheta = _signed_angle(head[:-1], head[1:])            # [T, N] per-step angle increment
    var_dtheta = float(dtheta.var())                       # theory 2 D_r dt = 0.2
    mean_dtheta = float(dtheta.mean())                     # theory 0

    tt = np.arange(T + 1) * DT
    sig = C > 0.05
    sig[0] = False
    x, y = tt[sig], -np.log(C[sig])
    D_r_eff = float((x * y).sum() / (x * x).sum())

    # --- diff against the oracle ------------------------------------------------------------- #
    with open(os.path.join(ORACLE_DIR, "summary.json")) as f:
        osum = json.load(f)
    C_oracle = np.asarray(osum["C"])
    assert len(C_oracle) == len(C), f"frame mismatch: oracle {len(C_oracle)} vs plexus {len(C)}"
    dC = np.abs(C - C_oracle)
    D_C = float(dC[1:].max())                              # max over t=1..40 (t=0 is 1.0 on both)
    argmax_t = int(np.argmax(dC[1:]) + 1)

    passed = bool(D_C <= THRESHOLD)
    reorient_acted = (_tap["acted"] == _tap["calls"] and _tap["calls"] > 0)

    summary = {
        "role": "plexus", "operator": "reorient", "spec": os.path.relpath(SPEC, PLEXUS),
        "N": int(head.shape[1]), "n_steps": T, "dt": DT, "rot_diffusion": D_R,
        # --- the honest acted ledger for the heading-steering leg ---
        "reorient_calls": _tap["calls"], "reorient_acted": _tap["acted"],
        "reorient_acted_every_call": reorient_acted,
        # --- corroborating invariants (Plexus side) ---
        "D_r_eff": D_r_eff, "D_r_input": D_R,
        "var_dtheta": var_dtheta, "var_dtheta_theory": 2 * D_R * DT, "mean_dtheta": mean_dtheta,
        "C_plexus": C.tolist(),
        # --- the metric ---
        "diff_metric": "D_C = max_{t=1..40} |C_plexus(t) - C_oracle(t)|",
        "value": D_C, "argmax_frame": argmax_t,
        "threshold": THRESHOLD, "passed": passed,
        # oracle corroboration, for the note
        "oracle_D_r_eff": osum["D_r_eff"], "oracle_var_dtheta": osum["var_dtheta"],
        "C_plexus_samples": {str(t): float(C[t]) for t in (0, 5, 10, 20, 40)},
        "C_oracle_samples": {str(t): float(C_oracle[t]) for t in (0, 5, 10, 20, 40)},
        "wall_s": round(wall, 1),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "diff_plexus_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("C_plexus",)}, indent=2))
    print(f"\nreorient acted {_tap['acted']}/{_tap['calls']} calls "
          f"(honest ledger; run_spec's INERT flag is the heading-buffer blind-spot)")
    print(f"C samples  t: 0,5,10,20,40")
    print(f"  plexus:  {[round(float(C[t]),4) for t in (0,5,10,20,40)]}")
    print(f"  oracle:  {[round(float(C_oracle[t]),4) for t in (0,5,10,20,40)]}")
    print(f"\nD_C = {D_C:.4f}  (argmax frame {argmax_t})   threshold {THRESHOLD}   "
          f"-> {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
