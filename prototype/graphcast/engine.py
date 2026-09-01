"""One entry point for the prototype, mirroring connectome-gnn's `GNN_Main.py -o <task> <config>`.

    python engine.py -o generate config/toy_small.yaml    run the forward spec, write the GT zarr
    python engine.py -o train    config/toy_small.yaml
    python engine.py -o test     config/toy_small.yaml [checkpoint]
    python engine.py -o plot     config/toy_small.yaml
    python engine.py -o gates    config/toy_small.yaml    run the gate table, write gates.csv + .tex

Every task reads ONE yaml. There is no second source of truth: the four architectural options, the
data paths, the split, the optimiser and the units all live in that file (see `spec_schema.py`), and
the only numbers hardcoded anywhere in this prototype are the gate thresholds in `gates.py`, which
is deliberate -- a threshold you can edit in a yaml between runs is not a threshold.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLEXUS_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
for p in (_HERE, _PLEXUS_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import gates as gates_mod
# EVERY OPERATOR MODULE IS IMPORTED HERE, at the entry point, because `plexus.schema.load`
# validates a spec's operators against the registry -- and an operator that has not been imported
# is not in it. The failure is confusing rather than obvious: the spec is rejected as naming an
# unknown operator, and the list of "available" ones is the library's, so it reads as though the
# prototype's operators were never written.
import ops_embedding  # noqa: F401  ngp_embedding
import ops_gnn        # noqa: F401  gnn_field
import ops_known_ode  # noqa: F401  transport_known_ode, kuramoto_known_ode
import ops_toy        # noqa: F401  advect_field, kuramoto_field, wave_field, gradient_gain
import spec_schema
import toy as toy_mod
import viz


# --------------------------------------------------------------------------------------- #
# stage-0 gate checks. Each returns (measured, note); the runner records against the
# threshold that was fixed in gates.py before any of this existed.
# --------------------------------------------------------------------------------------- #

def _all_option_combinations():
    """The 24 = 2 x 2 x 2 x 3 combinations the plan gates on.

    `simple` is only defined at one pass -- it carries no edge state, so repeating it is a
    different model than it claims to be, and `ModelSpec.parse` rejects that combination. So the
    grid is not message x passes but four legal (message, passes) PAIRS:

        2 encoder_decoder  x  4 (message, n_passes)  x  3 embedding  =  24

    which is the same 24 the plan pre-registered, reached without counting a combination the
    schema refuses. Enumerated explicitly so the count is visible rather than inferred.
    """
    msg_pass = [("simple", 1), ("graphcast", 1), ("graphcast", 4), ("graphcast", 16)]
    return [
        {"encoder_decoder": ed, "message": m, "n_passes": p, "embedding": emb}
        for ed, (m, p), emb in itertools.product(
            ("off", "on"), msg_pass, ("none", "free", "multires"))
    ]


def gate_G1_parse(spec_path: str):
    """Every option combination must parse. Built by overriding the model block of a real spec."""
    import copy
    import tempfile
    import yaml

    with open(spec_path) as f:
        raw = yaml.safe_load(f)
    combos = _all_option_combinations()
    ok, failures, flags = 0, [], []
    for combo in combos:
        r = copy.deepcopy(raw)
        r.setdefault("model", {}).update(combo)
        if combo["encoder_decoder"] == "on" and not r["model"].get("mesh_resolution"):
            r["model"]["mesh_resolution"] = [16, 16, 16]
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(r, tf)
            tmp = tf.name
        try:
            spec_schema.load(tmp)
            ok += 1
            flags.append(True)
        except Exception as e:                      # noqa: BLE001 -- the gate wants the message
            failures.append(f"{combo}: {type(e).__name__}: {e}")
            flags.append(False)
        finally:
            os.unlink(tmp)
    note = f"{len(combos)} enumerated" + ("" if not failures else f"; first failure: {failures[0]}")
    return float(ok), note, [viz.option_matrix(combos, flags, os.path.join(_ARTDIR[0], "G1_options.png"))]


def gate_G2_no_hardcoding(_spec_path: str):
    """Scan the prototype's own .py for dataset identity, on the AST rather than on the text.

    THE DISTINCTION THE FIRST VERSION MISSED. A line-by-line regex flagged `toy.py`'s module
    docstring for the word "ZAPBench", which is prose explaining why a toy exists -- documentation,
    not a hardcoded value. Stripping docstrings from the scan would have hidden the opposite case
    too, because a hardcoded path IS a string literal. So the scan walks the AST and checks every
    string and numeric constant EXCEPT docstrings: a dataset path or dimension used as a value
    still fails, and naming a dataset in prose does not. Patterns and exemptions live in gates.py.
    """
    import ast

    offenders, counts = [], {}
    for root, dirs, files in os.walk(_HERE):
        dirs[:] = [d for d in dirs if d not in gates_mod.G2_EXEMPT_DIRS]
        for fn in files:
            if not fn.endswith(".py") or fn in gates_mod.G2_EXEMPT_FILES:
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, _HERE)
            src = open(path).read()
            counts[rel] = len(src.splitlines())
            tree = ast.parse(src, filename=path)
            docstrings = {id(ast.get_docstring(n, clean=False)) for n in ast.walk(tree)
                          if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                            ast.AsyncFunctionDef))}
            doc_nodes = set()
            for n in ast.walk(tree):
                if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = getattr(n, "body", [])
                    if body and isinstance(body[0], ast.Expr) and \
                            isinstance(body[0].value, ast.Constant) and \
                            isinstance(body[0].value.value, str):
                        doc_nodes.add(id(body[0].value))
            for n in ast.walk(tree):
                if not isinstance(n, ast.Constant) or id(n) in doc_nodes:
                    continue
                text = n.value if isinstance(n.value, str) else (
                    repr(n.value) if isinstance(n.value, (int, float)) else None)
                if text is None:
                    continue
                for pat, why in gates_mod.FORBIDDEN_PATTERNS:
                    if re.search(pat, str(text), re.I):
                        offenders.append(f"{rel}:{n.lineno} {why}")
    note = "; ".join(offenders[:3]) if offenders else (
        f"{len(counts)} files, {sum(counts.values())} lines, constants only (docstrings exempt)")
    return float(len(offenders)), note, [viz.scan_coverage(counts, offenders,
                                          os.path.join(_ARTDIR[0], "G2_scan.png"))]


def gate_G4_partition_of_unity(spec_path: str):
    """The transfer weights must sum to one, or every application changes the total.

    Tested directly on `plexus.operators.mpm_ops.bspline`, the transfer the encoder/decoder option
    wraps -- so this needs no model and no wiring, which is why its stage is 0 rather than 5.
    Swept over 2-D and 3-D and three resolutions, because a stencil bug can be dimension-specific.
    """
    import torch
    from plexus.operators.mpm_ops import bspline, stencil_offsets

    torch.manual_seed(0)
    errs, worst = [], 0.0
    for D in (2, 3):
        for res in (32, 64, 128):
            off = stencil_offsets(D)
            X = torch.rand(20000, D)
            _, w, _ = bspline(X, float(res), off, (res,) * D, False)
            e = float((w.sum(1) - 1.0).abs().max())
            errs.append((D, res, e))
            worst = max(worst, e)

    # negative control: drop the middle B-spline lobe and confirm the check notices
    D, res = 2, 64
    off = stencil_offsets(D)
    X = torch.rand(5000, D)
    fx = X * res - (X * res - 0.5).floor()
    bad = torch.stack([0.5 * (1.5 - fx) ** 2, torch.zeros_like(fx),
                       0.5 * (fx - 0.5) ** 2], dim=1)
    wbad = torch.ones(X.shape[0], off.shape[0])
    for k in range(D):
        wbad = wbad * bad[:, off.long()[:, k], k]
    control = float((wbad.sum(1) - 1.0).abs().max())

    png = viz.partition_of_unity(errs, control, os.path.join(_ARTDIR[0], "G4_partition.png"))
    return worst, (f"worst over D=2,3 x res=32,64,128; float32 eps is 1.2e-7; "
                   f"negative control (middle lobe dropped) reads {control:.3g}"), [png]


def gate_G7_units(spec_path: str):
    """Units declared, and every measurement-tier threshold carries a phenomenon unit."""
    fit = spec_schema.load(spec_path)
    table = gates_mod.build_table()
    mesh_units = {"grid cells", "cells", "voxels", "pixels", "steps"}
    bad = [g.gid for g in table.values()
           if g.tier == "measurement" and (not g.unit or g.unit.lower() in mesh_units)]
    ok = fit.units.declared and not bad
    note = (f"length_um={fit.units.length_um}, time_s={fit.units.time_s}"
            + ("" if not bad else f"; measurement gates in mesh units: {bad}"))
    return float(bool(ok)), note, [viz.unit_ladder(fit.units,
                                    os.path.join(_ARTDIR[0], "G7_units.png"))]


def gate_G16_types_are_spatially_mixed(spec_path: str):
    """G16: the toy's types must be unlearnable from position, or G11/G12 prove nothing."""
    fit = spec_schema.load(spec_path)
    gt_path = os.path.join(_ARTDIR[0], "ground_truth.npz")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"{gt_path} not found -- run `-o generate {spec_path}` first")
    import numpy as np
    gt = np.load(gt_path)
    by_res = {r: toy_mod.spatial_type_purity(gt["positions"], gt["node_type"], r)
              for r in (2, 4, 8, 16, 32)}
    worst = max(by_res.values())
    png = viz.toy_summary(gt, os.path.join(_ARTDIR[0], "G16_toy.png"), purity_by_res=by_res)
    mp4 = viz.state_movie(gt["voltage"], gt["positions"],
                          os.path.join(_ARTDIR[0], "G16_state.mp4"))
    note = "worst over " + ", ".join(f"{r}:{v:.2f}" for r, v in sorted(by_res.items()))
    return float(worst), note, [p for p in (png, mp4) if p]


def _toy_stats(spec_path: str):
    """The stage-1b numbers, computed once and shared by G21-G25. DATA ONLY -- no model exists."""
    import numpy as np
    fit = spec_schema.load(spec_path)
    gt = np.load(os.path.join(_ARTDIR[0], "ground_truth.npz"))
    v, grad, ei = gt["voltage"], gt["grad"], gt["edge_index"]
    pre, post = ei
    N = v.shape[1]
    lo, hi = 200, min(v.shape[0] - 1, 2200)
    sl = slice(lo, hi)
    dv, V, G = np.diff(v, axis=0)[sl], v[sl], grad[sl]

    r2_rule, gain_fit = [], []
    for i in range(N):
        y = dv[:, i]
        A = np.stack([V[:, i], G[:, i], np.ones(len(y))]).T
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        r2_rule.append(1 - ((y - A @ c) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30))
        gain_fit.append(c[1])
    dt = float(fit.plexus.dt)
    gain_fit = np.array(gain_fit) / max(dt, 1e-12)

    r2_nb = []
    for i in range(0, N, 8):
        nb = pre[post == i]
        A = np.concatenate([V[:, nb] - V[:, i:i + 1], np.ones((len(V), 1))], axis=1)
        y = G[:, i]
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        r2_nb.append(1 - ((y - A @ c) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30))

    nb_corr = [abs(np.corrcoef(V[:, post[e]], V[:, pre[e]])[0, 1])
               for e in range(0, ei.shape[1], 37)]
    nb_corr = np.array([c for c in nb_corr if np.isfinite(c)])
    return gt, {"r2_rule": np.array(r2_rule), "r2_grad_nb": np.array(r2_nb),
                "gain_true": gt["gain"], "gain_fit": gain_fit, "nb_corr": nb_corr}


_STATS = {}


def _stats_for(spec_path):
    if spec_path not in _STATS:
        _STATS[spec_path] = _toy_stats(spec_path)
    return _STATS[spec_path]


def gate_G21_travelling_wave(spec_path: str):
    """The coarse rule really is a wave moving left to right, at the speed the spec asked for."""
    import numpy as np
    import zarr
    fit = spec_schema.load(spec_path)
    z = zarr.open(os.path.join(_ARTDIR[0], "toy.zarr"), "r")
    fname = next(iter(fit.plexus.fields))
    grid = np.asarray(z[fname]["grid"])          # [frames, channels, nx, ny], strided
    g = grid[:, 0].mean(axis=2)                  # average out y -> [frames, nx]
    nx = g.shape[1]
    # PHASE FROM THE FFT of the dominant spatial mode, not from argmax. argmax on a 128-cell grid
    # quantises to whole cells, and the wave moves about half a cell per frame, so the estimator's
    # own resolution was coarser than the quantity it measured -- the same class of mistake as
    # measuring type purity on a grid finer than the sampling.
    # PROJECT ONTO THE KNOWN WAVELENGTH, not onto the nearest integer FFT bin. With nx = 128 and
    # lambda = 0.15 the true mode is k = 1/lambda = 6.67 bins, and rounding it to 7 biases the
    # speed by exactly 6.67/7 = 0.952 -- which is the 5% this gate first reported. The estimator
    # has to be as sharp as the threshold it is judged against.
    wf0 = next(o for o in fit.plexus.operators if o.op == "wave_field")
    lam = float(wf0.params.get("wavelength", 0.5))
    xs = (np.arange(nx) + 0.5) / nx
    basis = np.exp(-2j * np.pi * xs / lam)
    proj = (g - g.mean(axis=1, keepdims=True)) @ basis
    phase = np.unwrap(np.angle(proj))
    d = -np.diff(phase) * lam * nx / (2 * np.pi)    # cells per recorded frame
    stride = max(1, int(round(fit.plexus.n_frames / max(grid.shape[0] - 1, 1))))
    wf = next(o for o in fit.plexus.operators if o.op == "wave_field")
    expect = nx * float(wf.params.get("wavelength", 0.5)) / float(wf.params.get("period", 100.0))
    measured = float(np.median(np.abs(d))) / stride
    rel = abs(measured - expect) / max(expect, 1e-9)
    png = viz.field_movie(grid, os.path.join(_ARTDIR[0], "G21_field.mp4"), stride=1)
    return rel, (f"measured {measured:.3f} cells/frame vs {expect:.3f} expected, "
                 f"{grid.shape[0]} recorded frames"), [p for p in (png,) if p]


def gate_G22_rule_recoverable(spec_path: str):
    gt, st = _stats_for(spec_path)
    png = viz.identifiability_panels(st, os.path.join(_ARTDIR[0], "G22_identifiability.png"))
    return float(st["r2_rule"].min()), (
        f"mean {st['r2_rule'].mean():.4f}, median {np.median(st['r2_rule']):.4f}"), [png]


def gate_G23_gradient_from_neighbours(spec_path: str):
    gt, st = _stats_for(spec_path)
    png = viz.identifiability_panels(st, os.path.join(_ARTDIR[0], "G22_identifiability.png"))
    return float(np.mean(st["r2_grad_nb"])), (
        f"worst node {np.min(st['r2_grad_nb']):.4f}"), [png]


def gate_G24_heterogeneity_readable(spec_path: str):
    gt, st = _stats_for(spec_path)
    c = float(np.corrcoef(st["gain_fit"], st["gain_true"])[0, 1])
    png = viz.heterogeneity_map(gt["positions"], gt["gain"], gt["node_type"],
                                os.path.join(_ARTDIR[0], "G24_heterogeneity.png"))
    ratio = float(np.median(st["gain_fit"] / np.where(np.abs(st["gain_true"]) > 1e-6,
                                                      st["gain_true"], np.nan)))
    return c, f"fitted/true gain ratio {ratio:.3f} (1.0 == exact)", [png]


def gate_G26_graph_is_necessary(spec_path: str):
    """Can a node-local predictor reproduce dv from that node's OWN history alone?

    If it can, the graph is decoration and no result about a graph model means anything. The
    baseline is deliberately generous -- the node's state, its drive, and both of their recent
    histories -- because the gate is only informative if the thing it rules out was given every
    chance. It uses no neighbour, and that is the whole of the comparison.
    """
    gt, st = _stats_for(spec_path)
    fit = spec_schema.load(spec_path)
    v, stim = gt["voltage"], gt["stim"]
    lo, hi = 200, min(v.shape[0] - 1, 2200)
    withheld = str(fit.data.stimulus).lower() in ("none", "null", "")
    lags = 4
    r2 = []
    for i in range(0, v.shape[1], 8):
        cols = [v[lo - k:hi - k, i] for k in range(lags)]
        if not withheld:
            cols += [stim[lo - k:hi - k, i] for k in range(lags)]
        A = np.stack(cols + [np.ones(hi - lo)]).T
        y = np.diff(v, axis=0)[lo:hi, i]
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        r2.append(1 - ((y - A @ c) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30))
    r2 = np.array(r2)
    png = viz.necessity_panel(r2, st["r2_rule"], os.path.join(_ARTDIR[0], "G26_necessity.png"),
                              withheld=withheld, coarse=str(gt["coarse_model"]))
    return float(np.median(r2)), (
        f"coarse rule '{gt['coarse_model']}', drive "
        f"{'withheld' if withheld else 'observed'}; node-local mean {r2.mean():.3f}, "
        f"neighbour-informed mean {st['r2_rule'].mean():.3f}"), [png]


def gate_G25_not_collinear(spec_path: str):
    gt, st = _stats_for(spec_path)
    png = viz.identifiability_panels(st, os.path.join(_ARTDIR[0], "G22_identifiability.png"))
    return float(st["nb_corr"].mean()), (
        f"max {st['nb_corr'].max():.3f} over {len(st['nb_corr'])} sampled edges"), [png]


_ARTDIR = [os.path.join(_HERE, "log")]           # set by run_gates to the run's own directory


STAGE_CHECKS = {
    "G1": gate_G1_parse,
    "G2a": gate_G2_no_hardcoding,
    "G4": gate_G4_partition_of_unity,
    "G7": gate_G7_units,
    "G16": gate_G16_types_are_spatially_mixed,
    "G21": gate_G21_travelling_wave,
    "G22": gate_G22_rule_recoverable,
    "G23": gate_G23_gradient_from_neighbours,
    "G24": gate_G24_heterogeneity_readable,
    "G25": gate_G25_not_collinear,
    "G26": gate_G26_graph_is_necessary,
}


def run_gates(spec_path: str, out_dir: str, only: list[str] | None = None) -> int:
    table = gates_mod.build_table()
    _ARTDIR[0] = out_dir
    todo = [g for g in (only or list(STAGE_CHECKS)) if g in STAGE_CHECKS]
    for gid in todo:
        arts = []
        try:
            measured, note, arts = STAGE_CHECKS[gid](spec_path)
        except Exception as e:                      # noqa: BLE001
            measured, note = None, f"check raised {type(e).__name__}: {e}"
        table[gid].record(measured, note, arts)

    csv_path = gates_mod.write_csv(table, os.path.join(out_dir, "gates.csv"))
    # Two copies of the table, and the difference is the figure paths. The run-dir copy is the
    # archive, with paths relative to itself; the prototype-root copy is what the note \input's,
    # with paths relative to the note so \includegraphics resolves where latexmk runs.
    md_path = gates_mod.write_md(table, os.path.join(out_dir, "GATES.md"))
    gates_mod.write_md(table, os.path.join(_HERE, "GATES.md"), rel_to=_HERE)
    gates_mod.write_csv(table, os.path.join(_HERE, "gates.csv"))

    width = max(len(g.what) for g in table.values())
    print(f"\ngate table for {os.path.basename(spec_path)}")
    for gid in sorted(table, key=gates_mod._order):
        g = table[gid]
        if g.outcome == gates_mod.SKIP and gid not in todo:
            continue
        meas = "---" if g.measured is None else f"{g.measured:.6g}"
        print(f"  {g.gid:<4} {g.what:<{width}}  {meas:>12}  {g.outcome}"
              + (f"   [{g.note}]" if g.note else ""))
    print("\n" + gates_mod.summary(table))
    print(f"wrote {csv_path}\nwrote {md_path}")
    return sum(1 for g in table.values() if g.outcome == gates_mod.FAIL)


def _not_yet(task: str, stage: int):
    raise NotImplementedError(
        f"`-o {task}` lands at stage {stage} of the approved plan; stage 0 delivers the spec "
        f"schema, this entry point and gates G1/G2/G7. Run `-o gates <config>` for now.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GraphCast-in-Plexus prototype")
    ap.add_argument("-o", "--option", nargs="+", required=True,
                    metavar=("TASK", "CONFIG"),
                    help="task (generate|train|test|plot|gates) followed by the config path")
    ap.add_argument("--out_root", default=None,
                    help="root for log/ output (default: the prototype's own log/)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="rollout length for `-o test` (default: the spec's fit.rollout.horizon)")
    ap.add_argument("--gate", nargs="+", default=None,
                    help="run only these gate ids, e.g. --gate G1 G2")
    args = ap.parse_args(argv)

    task = args.option[0]
    if len(args.option) < 2:
        ap.error("give a config path after the task, e.g. -o gates config/toy_small.yaml")
    config = args.option[1]
    if not os.path.exists(config):
        config_here = os.path.join(_HERE, config)
        if os.path.exists(config_here):
            config = config_here
        else:
            ap.error(f"config not found: {args.option[1]}")

    fit = spec_schema.load(config)
    out_dir = os.path.join(args.out_root or os.path.join(_HERE, "log"),
                           f"{fit.name}_{fit.model.tag()}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"spec      {config}\nname      {fit.name}\noptions   {fit.model.tag()}\n"
          f"units     length_um={fit.units.length_um} time_s={fit.units.time_s}\n"
          f"out       {out_dir}")

    if task == "gates":
        return run_gates(config, out_dir, args.gate)
    if task == "train" and fit.fit is not None:
        # A SPEC CARRYING `trainer:` IS RUN BY THE TRAINER ENGINE, not by the scalar train loop.
        # Which one runs is a property of the FILE -- it declared a schedule or it declared a bag
        # of scalars -- rather than of a flag someone has to remember to pass.
        # A SPEC CARRYING `fit:` IS RUN BY THE FIT ENGINE. Which one runs is a property of the
        # FILE -- it declared a fit block or it did not -- rather than of a flag to remember. The
        # run to fit is named IN the spec (`fit.data.run`), so there is no --data_dir either.
        import trainer as trainer_mod
        out = trainer_mod.run(fit, args.out_root or os.path.join(_HERE, "log"),
                              device=fit.training.device)
        print("learned:", out["learned"])
        return 0
    if task == "test" and fit.fit is not None:
        # THE TESTER IS A SEPARATE ENGINE because a loss is not a result: it is measured on the
        # training split at the objective's own horizon. `-o test` rolls out on HELD-OUT frames.
        import tester as tester_mod
        tester_mod.run(fit, args.out_root or os.path.join(_HERE, "log"),
                       device=fit.training.device, horizon=args.horizon)
        return 0
    if task == "generate":
        summary = toy_mod.generate(fit, out_dir, device=fit.data.device)
        for k, v in summary.items():
            print(f"  {k:14s} {v}")
        return 0
    if task == "train":
        import train as train_mod
        out = train_mod.train(fit, out_dir, device=fit.training.device)
        print("  final:", {k: round(v, 4) for k, v in out["history"][-1].items()})
        return 0
    if task in ("test", "plot"):
        _not_yet(task, 2)
    ap.error(f"unknown task {task!r} (expected generate|train|test|plot|gates)")


if __name__ == "__main__":
    raise SystemExit(main())
