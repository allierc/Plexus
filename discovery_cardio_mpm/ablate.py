#!/usr/bin/env python
"""ablate -- which of the learned fields is actually doing the work?

THE QUESTION, AND WHY IT IS ASKED THIS WAY
================================================================================================
A fit that learns four spatial fields has four chances to look good for the wrong reason. The
honest test is to take a TRAINED model, neutralise one field at a time, and re-measure -- if the
score does not move, that field was decoration.

**A field's null is a UNIFORM field, not zero.** Setting stiffness to zero changes how hard the
tissue is; the score would then move because the material changed, not because its pattern
mattered. Replacing each field by its own mean holds the magnitude fixed and removes only the
spatial structure, which is the question actually being asked: *does the model need this to vary in
space?* Prestress is the exception -- its neutral value really is the identity tensor.

WHAT THIS CAN AND CANNOT ANSWER
------------------------------------------------------------------------------------------------
It measures what an ALREADY-TRAINED model loses without a field. It does NOT measure what a model
would gain from a field it was never trained with: the other three fields were fitted around this
one's presence, and removing it leaves them mis-tuned rather than showing the field's true worth.
For that -- adding prestress to a model that never had it, say -- there is no shortcut past
retraining, and this file will say so rather than pretend.

Read on the corrected 10x10 grid with the five instruments that survived the fitted floor, plus
loopscore reported (never cited) for comparability with the archive.

    python ablate.py --run p3_b49_s2_fs2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
ARCHIVE = os.path.abspath(os.path.join(HERE, "..", "prototype", "cardio_mpm", "archive"))
PY = sys.executable
sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import metrics as M                                                 # noqa: E402

FIT_SPEC = os.path.join(HERE, "config", "material", "material_aniso_cardio_fit.yaml")
# the five that cleared the threshold once the fitted floor was folded in, plus the objective
REPORT = ["loopscore", "orientation_error", "chirality_match", "peak_excursion",
          "coordination", "path_length"]
FIELDS = ["stiff", "gain", "fibre", "prestress"]


def run_args(run_dir):
    c = json.load(open(os.path.join(run_dir, "config.json")))
    a = c.get("args", {}) or {}
    out = []
    for k, v in a.items():
        if k in ("--outdir", "--resume", "--device", "--n_iter", "--eval_dump", "--redash",
                 "--ablate"):
            continue
        out += [k, str(v)]
    return out


def which_fields(ckpt):
    """Which learned fields this checkpoint actually carries. Asked, not assumed."""
    import torch
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    return {"stiff": "stiff_siren" in sd, "gain": "gain_siren" in sd,
            "fibre": "fibre_siren" in sd, "prestress": "residual_siren" in sd}


def evaluate(ckpt, args_, ablate, device, work):
    """One forward pass from the checkpoint with `ablate` neutralised. Returns the dump path."""
    d = os.path.join(work, f"abl_{ablate or 'none'}")
    os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
    dump = os.path.join(d, "dump.npz")
    env = dict(os.environ, PYTHONPATH=SRC + ":" + os.environ.get("PYTHONPATH", ""))
    cmd = [PY, os.path.join(HERE, "train.py"), FIT_SPEC, *args_, "--resume", ckpt,
           "--eval_dump", dump, "--outdir", d, "--device", device]
    if ablate:
        cmd += ["--ablate", ablate]
    log = os.path.join(d, "eval.log")
    with open(log, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env, timeout=7200)
    if not os.path.exists(dump):
        tail = (open(log, errors="replace").read().strip().splitlines() or ["no output"])[-1]
        return None, tail[:150]
    return dump, None


def score(dump):
    from scipy.spatial import cKDTree
    import data as D
    z = np.load(dump)
    sim, real, rest = (z["sim_d"].astype(np.float64), z["real_d"].astype(np.float64),
                       z["rest"].astype(np.float64))
    P = D.open_npz(expect_sha256=D.HEALTHY_POS_SHA256)["pos"].astype(np.float64)
    Pm = D.DOM_LO + D.DOM * P
    idx = cKDTree(rest).query(Pm[0][M.select_grid_nodes(margin=M.MARGIN_SAFE)])[1]
    mask = np.zeros(rest.shape[0], bool); mask[idx] = True
    out = {}
    for n in REPORT:
        try:
            out[n] = M.REGISTRY[n](sim, real, mask)
        except Exception as e:
            out[n] = float("nan")
            print(f"    ({n}: {type(e).__name__})", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p3_b49_s2_fs2")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--work", default=os.path.join(HERE, "_ablate"))
    ap.add_argument("--plots", action="store_true", help="also draw the 10x10 grid per ablation")
    a = ap.parse_args(argv)

    run_dir = os.path.join(ARCHIVE, a.run)
    ck = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "model_*.pt")))
    if not ck:
        raise SystemExit(f"no checkpoint in {run_dir}")
    # copy out of the archive: it is read-only evidence, and train.py writes beside its checkpoint
    os.makedirs(a.work, exist_ok=True)
    local = os.path.join(a.work, os.path.basename(ck[-1]))
    if not os.path.exists(local):
        shutil.copy2(ck[-1], local)
    args_ = run_args(run_dir)
    present = which_fields(local)
    print(f"\n  run {a.run}  checkpoint {os.path.basename(ck[-1])}")
    print(f"  fields in this checkpoint: "
          + "  ".join(f"{k}={'yes' if v else 'NO'}" for k, v in present.items()))
    todo = [""] + [f for f in FIELDS if present.get(f)]
    absent = [f for f in FIELDS if not present.get(f)]
    if absent:
        print(f"  not ablated because this model never had them: {', '.join(absent)} "
              f"-- what they would ADD cannot be measured here, only by retraining")

    rows = {}
    for abl in todo:
        lab = abl or "none (the model as fitted)"
        print(f"\n  [{lab}] evaluating ...", flush=True)
        dump, err = evaluate(local, args_, abl, a.device, a.work)
        if not dump:
            print(f"    FAILED: {err}")
            continue
        rows[abl] = score(dump)
        print("    " + "  ".join(f"{k} {v:+.4f}" for k, v in rows[abl].items()))
        if a.plots:
            import grid_plot
            out = os.path.join(HERE, "figures", f"grid_{a.run}_{abl or 'none'}.png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            grid_plot.draw(dump, out, title=f"{a.run}  --  {lab}")
            print(f"    plot -> {out}")

    if "" not in rows:
        raise SystemExit("the un-ablated baseline failed; nothing to compare against")
    base = rows[""]

    print(f"\n{'=' * 112}\n  WHAT EACH LEARNED FIELD IS WORTH -- change when it is replaced by its "
          f"own mean\n{'=' * 112}")
    print(f"  {'field removed':<22s}" + "".join(f"{n[:15]:>17s}" for n in REPORT))
    print(f"  {'(none: as fitted)':<22s}" + "".join(f"{base[n]:>+17.4f}" for n in REPORT))
    print("  " + "-" * 108)
    for abl in todo[1:]:
        if abl not in rows:
            continue
        print(f"  {abl:<22s}" + "".join(f"{rows[abl][n] - base[n]:>+17.4f}" for n in REPORT))
    print("\n  Rows after the first are DIFFERENCES from the fitted model. A field whose row is all "
          "zeros\n  was decoration. Note the sign convention: orientation_error is an error, so "
          "POSITIVE is worse;\n  the rest are scores, so NEGATIVE is worse.")

    # the honest bar: is any of this bigger than the noise?
    fit = json.load(open(os.path.join(HERE, "_metrology", "noise_fits.json"))) \
        if os.path.exists(os.path.join(HERE, "_metrology", "noise_fits.json")) else {}
    beat = json.load(open(os.path.join(HERE, "_metrology", "noise_beats.json"))) \
        if os.path.exists(os.path.join(HERE, "_metrology", "noise_beats.json")) else {}
    print(f"\n  {'':<22s}" + "".join(f"{n[:15]:>17s}" for n in REPORT))
    units = {}
    for n in REPORT:
        f = (fit.get("metrics") or {}).get(n, {})
        b = (beat.get("metrics") or {}).get(n, {})
        cand = [v for v in (b.get("sd"), f.get("same_seed_difference"), f.get("seed_sd"))
                if v is not None and np.isfinite(v)]
        units[n] = max(cand) if cand else float("nan")
    print(f"  {'the working unit':<22s}" + "".join(f"{units[n]:>17.4f}" for n in REPORT))
    print(f"  {'3x it (a difference)':<22s}" + "".join(f"{3 * units[n]:>17.4f}" for n in REPORT))
    print("\n  A change smaller than three working units is INDISTINGUISHABLE and may not be "
          "called a finding.")
    print("=" * 112)

    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)
    json.dump({"run": a.run, "fields_present": present, "rows": rows, "units": units},
              open(os.path.join(HERE, "_metrology", f"ablate_{a.run}.json"), "w"),
              indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
