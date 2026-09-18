"""spec_parity -- the prototype's `on_frame` rollout against the same model written as a SPEC.

The prototype injects the active strain from a callback and binds the per-cell parameters as Level
attributes. `src/plexus/operators/contractile_ops.py` does the same thing as two operators reading
the parameters from state blocks of the `cell` set, which `seed_state_from_file` loads from the
fit. If the transfer is right the two rollouts are the same rollout.

BIT-IDENTICAL IS NOT THE GATE AND CANNOT BE. `mpm_scatter` accumulates with `index_add`, whose
CUDA form is atomic and reorders run to run: `fd_check` measures the prototype disagreeing with
ITSELF by 5.96e-07 on a loss of 0.435 in float32, and by 1.33e-15 in float64. So the gate is the
one the batch axis used -- agreement to the arithmetic, checked in float64 where the floor is
machine epsilon, and reported in float32 beside it so the practical number is visible too.

    python spec_parity.py --device cpu --dtype float64 --per-parent 12 --n-grid 48 --frames 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M  # noqa: E402


def spec_dict(fit_npz, n_grid, per_parent, n_frames, n_cells, drag_k, sub, nu, anchor_k=0.0):
    """`model.build_spec` plus the three things that make the parameters part of the SPEC:
    per-cell state blocks, a seed that loads the fit into them, and the two operators."""
    raw = M.build_spec(n_grid=n_grid, per_parent=per_parent, n_frames=n_frames, n_cells=n_cells,
                       drag_k=drag_k, sub=sub, anchor_k=anchor_k, compile=False,
                       differentiable=True)
    raw["sets"]["cell"]["state"] = {
        "pos": {"width": 2, "role": "coordinate", "integration": "second_order_coordinate",
                "boundary": "world"},
        "vel": {"width": 2, "role": "rate", "integration": "second_order_rate",
                "boundary": "free"},
        "phi": {"width": 1, "integration": "none", "boundary": "free"},
        "g": {"width": 1, "integration": "none", "boundary": "free"},
        "g2": {"width": 1, "integration": "none", "boundary": "free"},
        "logE": {"width": 1, "integration": "none", "boundary": "free"},
        "delay": {"width": 1, "integration": "none", "boundary": "free"},
        "logtau": {"width": 1, "integration": "none", "boundary": "free"},
        "logtr": {"width": 1, "integration": "none", "boundary": "free"},
        "logdur": {"width": 1, "integration": "none", "boundary": "free"},
        "amode": {"width": 2, "integration": "none", "boundary": "free"},
        "gam_prev": {"width": 1, "integration": "none", "boundary": "free"},
    }
    raw["seed"] = [dict(op="seed_state_from_file", at="cell", file=os.path.abspath(fit_npz))]
    ops = list(raw["operators"])
    i = next(k for k, o in enumerate(ops) if o["op"] == "drag")
    ops[i:i] = [dict(op="material_from_cell", at="mpm_particle", parent="cell", nu=nu),
                dict(op="active_strain", at="mpm_particle", parent="cell",
                     fit=os.path.abspath(fit_npz))]
    raw["operators"] = ops
    # AT THE END OF THE TICK, NOT THE START. `engine.run` calls `on_frame` at the BOTTOM of the
    # tick loop, after the schedule and after `_integrate`, so the prototype's callback applies
    # the active strain AFTER that frame's substeps -- frame t's contraction reaches the stress
    # at frame t+1. Putting the operator at the head of the schedule instead applies it one step
    # early; the two are both defensible discretisations and the fit was made against this one,
    # so a reproduction has to match it. Measured cost of the other order: 23% of the per-cell
    # strain signal.
    raw["schedule"] = list(raw["schedule"]) + ["material_from_cell", "active_strain"]
    return raw


def run_spec(raw, device, n_cells):
    """The spec's own rollout. The per-cell affine map stays in the driver: it is the OBSERVABLE,
    not a law, and the prototype computes it the same way."""
    from plexus import engine
    from recording import cell_affine
    sim = M.load_sim(raw)
    box, A_list, u_list = {}, [], []

    def cb(H, tick):
        q = H.level("mpm_particle")
        if tick == 0:
            box["cid"] = q.cell_id.long()
            box["X0"] = q.get("pos").detach().clone()
        x = q.get("pos")
        A, u = cell_affine(x, box["X0"], box["cid"], n_cells)
        A_list.append(A); u_list.append(u)

    engine.run(sim, device=device, on_frame=cb, progress=False, grad=False)
    return torch.stack(A_list), torch.stack(u_list)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", default=os.path.join(HERE, "out/fits/healthy_allbeats/params.npz"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--per-parent", type=int, default=12)
    ap.add_argument("--n-grid", type=int, default=48)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--n-cells", type=int, default=472)
    ap.add_argument("--drag", type=float, default=150.0)
    ap.add_argument("--sub", type=float, default=2e-4)
    ap.add_argument("--nu", type=float, default=0.3)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64 if a.dtype == "float64" else torch.float32)

    z = np.load(a.fit, allow_pickle=True)
    n_modes = int(z["n_modes"])
    print(f"[fit] {a.fit}\n[fit] {n_modes} temporal mode(s), clock {np.round(z['clock'], 4)}")

    # --- the prototype, exactly as `model.rollout` runs it -----------------------------------
    raw = M.build_spec(n_grid=a.n_grid, per_parent=a.per_parent, n_frames=a.frames,
                       n_cells=a.n_cells, drag_k=a.drag, sub=a.sub, compile=False)
    sim = M.load_sim(raw)
    P = M.Params(a.n_cells, a.device, nu=a.nu, n_frames=z["psi"].shape[1], n_modes=n_modes,
                 clock_mode=str(z["clock_mode"]))
    P.load({k: z[k] for k in z.files if k != "interior"})
    proto = M.rollout(sim, P, a.device, a.n_cells, grad=False)

    # --- the same model as a spec --------------------------------------------------------- #
    spec = run_spec(spec_dict(a.fit, a.n_grid, a.per_parent, a.frames, a.n_cells,
                              a.drag, a.sub, a.nu), a.device, a.n_cells)

    A0, u0 = proto["A"].double(), proto["u"].double()
    A1, u1 = spec[0].double(), spec[1].double()
    n = min(A0.shape[0], A1.shape[0])
    dA = (A1[:n] - A0[:n]).abs().max().item()
    du = (u1[:n] - u0[:n]).abs().max().item()
    sA = (A0[:n] - torch.eye(2, dtype=A0.dtype)).abs().max().item()
    su = u0[:n].abs().max().item()
    print(f"[parity] {n} frames, {a.n_cells} cells, {a.dtype} on {a.device}")
    print(f"[parity] per-cell A: max |spec - prototype| = {dA:.3e}   against a signal of {sA:.3e}"
          f"   -> {dA / max(sA, 1e-30):.2e} relative")
    print(f"[parity] per-cell u: max |spec - prototype| = {du:.3e}   against a signal of {su:.3e}"
          f"   -> {du / max(su, 1e-30):.2e} relative")
    # THE GATE, AND WHY IT IS NOT MACHINE EPSILON. Every fitted parameter is float32 on disk and
    # both sides were checked to hold bit-identical values, so the residual is arithmetic ORDER
    # inside the clock -- measured at 4.9e-09 relative, 25x below float32 epsilon (1.2e-07), which
    # is the precision the parameters themselves carry. 1e-07 is therefore a real bound and not a
    # relaxed one: the ordering bug this test was built to catch showed up at 2.3e-01, seven
    # orders above it.
    tol = 1e-7 if a.dtype == "float64" else 1e-3
    ok = max(dA / max(sA, 1e-30), du / max(su, 1e-30)) < tol
    print(f"[parity] {'MATCH' if ok else 'DIFFER'} at {tol:g} relative")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
