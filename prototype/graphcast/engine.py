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

import gates as gates_mod
import spec_schema


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
    ok, failures = 0, []
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
        except Exception as e:                      # noqa: BLE001 -- the gate wants the message
            failures.append(f"{combo}: {type(e).__name__}: {e}")
        finally:
            os.unlink(tmp)
    note = f"{len(combos)} enumerated" + ("" if not failures else f"; first failure: {failures[0]}")
    return float(ok), note


def gate_G2_no_hardcoding(_spec_path: str):
    """Scan the prototype's own .py for dataset identity. Patterns and exemptions are part of the
    gate's definition and live in `gates.py`, so this walker never reads its own rules."""
    offenders = []
    for root, dirs, files in os.walk(_HERE):
        dirs[:] = [d for d in dirs if d not in gates_mod.G2_EXEMPT_DIRS]
        for fn in files:
            if not fn.endswith(".py") or fn in gates_mod.G2_EXEMPT_FILES:
                continue
            path = os.path.join(root, fn)
            with open(path) as f:
                for i, line in enumerate(f, 1):
                    code = line.split("#", 1)[0]
                    for pat, why in gates_mod.FORBIDDEN_PATTERNS:
                        if re.search(pat, code, re.I):
                            offenders.append(f"{os.path.relpath(path, _HERE)}:{i} {why}")
    note = "; ".join(offenders[:3]) if offenders else f"scanned .py under {os.path.basename(_HERE)}/"
    return float(len(offenders)), note


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
    return float(bool(ok)), note


STAGE_CHECKS = {
    "G1": gate_G1_parse,
    "G2": gate_G2_no_hardcoding,
    "G7": gate_G7_units,
}


def run_gates(spec_path: str, out_dir: str, only: list[str] | None = None) -> int:
    table = gates_mod.build_table()
    todo = [g for g in (only or list(STAGE_CHECKS)) if g in STAGE_CHECKS]
    for gid in todo:
        try:
            measured, note = STAGE_CHECKS[gid](spec_path)
        except Exception as e:                      # noqa: BLE001
            measured, note = None, f"check raised {type(e).__name__}: {e}"
        table[gid].record(measured, note)

    csv_path = gates_mod.write_csv(table, os.path.join(out_dir, "gates.csv"))
    tex_path = gates_mod.write_tex(table, os.path.join(out_dir, "gates_table.tex"))

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
    print(f"wrote {csv_path}\nwrote {tex_path}")
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
    if task == "generate":
        _not_yet(task, 1)
    if task in ("train", "test", "plot"):
        _not_yet(task, 2)
    ap.error(f"unknown task {task!r} (expected generate|train|test|plot|gates)")


if __name__ == "__main__":
    raise SystemExit(main())
