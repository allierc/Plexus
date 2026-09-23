"""spec -> stimulus.zarr + target.zarr + provenance.json, and the excitation check.

    python -m plexus.tasks.generate config/task/leaky_integrator_8s.yaml
    python -m plexus.tasks.generate config/task/*.yaml --force

THE LAYOUT IS THE ONE EVERY OTHER CORPUS IN `graphs_data` ALREADY USES, deliberately -- a split
per directory, one zarr per named array, a `meta.json` beside them, a figure per split, and the
generating parameters in a `.pt`. Someone who knows `zebrafish_hd_si_task_917_*` knows this.

    graphs_data/task/<name>/
        train/  stimulus.zarr (trials, T, channels)
                target.zarr   (trials, T, targets)
                cond.zarr     (trials,)  the index into `spec.cells` -- read a result PER CELL
                meta.json
        val/ test/  the same
        teacher.pt              the GROUND TRUTH: poles, and the law's own coefficients
        provenance.json         the spec verbatim, its hash, and the excitation report
        task_traces_<split>.png one figure per split

`teacher.pt` IS THIS PACKAGE'S `ode_params.pt`. That file holds the parameters a synthetic ODE
dataset was generated from -- tau_i, W, edge_index -- so a fit can be scored by PARAMETER
RECOVERY and not only by trajectory error. The analogue here is stronger, not weaker: for an LTI
teacher the ground truth is complete and analytic, so a fitted circuit can be probed with a chirp
and its measured poles compared against the true ones, pole for pole.

No set, no operator, no circuit appears anywhere -- a corpus generated here can be inspected and
argued about before the circuit that will be fitted to it exists, and one corpus scores several
circuits without regeneration.

THE EXCITATION CHECK IS THE POINT OF DOING THIS AHEAD OF TIME. A mode the stimulus never excites
cannot be identified, however long a fit runs; no amount of data repairs it, and the failure is
silent -- the loss goes down, the mode is simply unconstrained. It is decidable in advance,
because it is a property of the STIMULUS and the TEACHER'S POLES, not of the circuit: compute the
input's power spectrum, find the frequencies the poles sit at, and ask whether there is power
there.

This lineage has already paid for not doing it. A trained 285-cell circuit carried a 12 Hz
unstable oscillation that the objective could not see, because the plant it was scored through is
a ~1 Hz low-pass -- discovered by fitting five seeds and comparing them, rather than predicted
from the spectra before the first run. `spectral_coverage` below is that prediction.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np

from plexus.paths import graphs_data_path
from plexus.tasks import get_stimulus, get_teacher
from plexus.tasks.schema import load_task


def task_dir(name, root=None) -> str:
    return os.path.join(root or graphs_data_path(), "task", str(name))


# --------------------------------------------------------------------------- #
#  the excitation report
# --------------------------------------------------------------------------- #
def spectral_coverage(u, dt, poles, floor_db=-20.0) -> dict:
    """Does the stimulus put power where the teacher's poles are?

    A pole at continuous-time `p` responds at frequency `|Im(p)|/2pi` if it is complex (a
    resonance) and acts over a band up to `|Re(p)|/2pi` if it is real (a relaxation). Either way
    it has a frequency, and a pole whose frequency carries no input power is a mode the data
    cannot constrain.

    `floor_db` is how far below the input's own PEAK spectral density still counts as excited;
    -20 dB is one hundredth of the peak power, which is a generous floor deliberately -- the
    check exists to catch modes with NO excitation, not to adjudicate marginal ones.

    Returns per pole: its frequency, the input's power there relative to the peak in dB, and
    whether it clears the floor. `identifiable` is False if ANY pole fails, which is the useful
    summary: one unexcited mode is enough to make a fit's verdict conditional.
    """
    if poles is None or len(poles) == 0:
        return {"poles": [], "identifiable": True,
                "note": "the law has no poles (static or a pure delay); nothing to excite"}
    T = u.shape[1]
    freqs = np.fft.rfftfreq(T, d=dt)
    # average the periodogram over trials and channels -- the ENSEMBLE's spectrum, since that is
    # what the fit sees, not any one trial's
    psd = (np.abs(np.fft.rfft(u, axis=1)) ** 2).mean(axis=(0, 2))
    peak = psd.max()
    rows, ok = [], True
    for p in np.atleast_1d(poles):
        p = complex(p)
        f = abs(p.imag) / (2 * np.pi) if abs(p.imag) > 1e-12 else abs(p.real) / (2 * np.pi)
        k = int(np.argmin(np.abs(freqs - f)))
        db = 10.0 * np.log10(max(psd[k], 1e-300) / max(peak, 1e-300))
        clear = bool(db >= floor_db)
        ok &= clear
        rows.append({"pole_real": p.real, "pole_imag": p.imag, "freq_hz": float(f),
                     "input_power_db_rel_peak": float(db), "excited": clear})
    return {"poles": rows, "identifiable": bool(ok), "floor_db": float(floor_db),
            "nyquist_hz": float(0.5 / dt),
            "note": ("every pole sits where the stimulus has power" if ok else
                     "AT LEAST ONE POLE IS UNEXCITED: that mode cannot be identified from this "
                     "stimulus, however long the fit runs")}


# --------------------------------------------------------------------------- #
#  generation
# --------------------------------------------------------------------------- #
def build_split(spec, split, verbose=True):
    """(stimulus [N, T, C], target [N, T, K], cond [N]) for one split.

    Trial i is seeded `seed0 + i` and its condition cell is `i % n_cells`, so the grid is
    balanced exactly rather than in expectation, and the trial -> seed map is a pure function of
    the spec. Regenerating on another machine gives the same corpus.
    """
    proc = get_stimulus(spec.process_name)
    cells = spec.cells
    n_per = int(spec.splits[split]["n_per_cond"])
    seed0 = int(spec.splits[split]["seed0"])
    N = len(cells) * n_per

    U = np.empty((N, spec.T, spec.channels), np.float64)
    cond = np.empty(N, np.int32)
    for i in range(N):
        c = i % len(cells)
        cond[i] = c
        rng = np.random.default_rng(seed0 + i)
        p = {**spec.stimulus, **{k: v for k, v in cells[c].items() if k in _proc_keys(proc, spec)}}
        U[i] = proc(rng, spec.T, spec.dt, spec.channels, **p)

    # The teacher is applied PER CELL, in one call, because a cell may override a teacher
    # parameter (a different tau per condition) and because vectorising over trials is what
    # makes an lsim-based law affordable at these counts.
    Y = np.empty((N, spec.T, spec.channels), np.float64)
    for c, cell in enumerate(cells):
        idx = np.where(cond == c)[0]
        tp = {**spec.teacher, **{k: v for k, v in cell.items() if k not in _proc_keys(proc, spec)}}
        # THE LAW ITSELF MAY BE THE THING THAT VARIES, which is what `teachers:` is for. `name`
        # is the cell's label and no law's parameter, so it is dropped before the call.
        law = get_teacher(tp.pop("law", spec.law_name))
        tp.pop("name", None)
        Y[idx] = law(U[idx], spec.dt, **tp)
    if verbose:
        print(f"  [{split}] {N} trials x {spec.T} frames  "
              f"|u| rms {U.std():.4f}  |y| rms {Y.std():.4f}")
    return U, Y.astype(np.float64), cond


def _proc_keys(proc, spec):
    """Which condition keys belong to the STIMULUS rather than the teacher.

    A condition names one or the other, and guessing wrongly would silently pass a teacher
    parameter to a process (where `**_` swallows it) and vary nothing. Decided by the process's
    own signature, so it cannot drift from the code.
    """
    import inspect
    sig = inspect.signature(proc)
    return {k for k in sig.parameters if k not in ("rng", "T", "dt", "channels")} | set(spec.stimulus)


def generate(path, root=None, force=False, verbose=True) -> str:
    spec = load_task(path)
    out = task_dir(spec.name, root)
    prov_path = os.path.join(out, "provenance.json")
    if os.path.exists(prov_path) and not force:
        if verbose:
            print(f"[task] {spec.name}: already at {out} (pass --force to regenerate)")
        return out
    os.makedirs(out, exist_ok=True)
    # A REGENERATION REPLACES, IT DOES NOT ACCUMULATE. Leaving a previous layout's files beside
    # the new ones gives a directory holding two corpora, and a reader cannot tell which arrays
    # the provenance describes.
    import shutil as _sh
    for f in os.listdir(out):
        q = os.path.join(out, f)
        _sh.rmtree(q) if os.path.isdir(q) else os.remove(q)
    if verbose:
        print(f"[task] {spec.describe()}")
        print(f"[task] stimulus {spec.process_name}({spec.stimulus})  "
              f"teacher {spec.law_name}({spec.teacher})")

    import torch
    import zarr
    law = get_teacher(spec.law_name)
    prov = {"spec": spec.raw, "spec_sha256": _sha(spec.raw), "spec_path": spec.path,
            "process": spec.process_name, "law": spec.law_name,
            "T": spec.T, "dt": spec.dt, "channels": spec.channels,
            "cells": spec.cells, "splits": {}}

    built = {}
    for split in sorted(spec.splits):
        U, Y, cond = build_split(spec, split, verbose)
        built[split] = (U, Y, cond)
        d = os.path.join(out, split)
        os.makedirs(d, exist_ok=True)
        zarr.save_array(os.path.join(d, "stimulus.zarr"), U)
        zarr.save_array(os.path.join(d, "target.zarr"), Y)
        zarr.save_array(os.path.join(d, "cond.zarr"), cond)
        seed0 = int(spec.splits[split]["seed0"])
        meta = {"n_trials": int(U.shape[0]), "T": spec.T, "channels": int(U.shape[2]),
                "targets": int(Y.shape[2]), "dt": spec.dt,
                "seed_range": [seed0, seed0 + int(U.shape[0])],
                "n_per_cond": int(spec.splits[split]["n_per_cond"]), "cells": spec.cells}
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        prov["splits"][split] = meta

    # The excitation report, computed on the TRAIN split since that is the one a fit sees, and
    # PER CONDITION CELL because a grid may vary the teacher: a sweep over filter ORDER has
    # different poles in every cell, and reporting the first cell's would say the task is
    # identifiable on the strength of the easiest member of a family.
    train = "train" if "train" in spec.splits else sorted(spec.splits)[0]
    U, _Y, cond = build_split(spec, train, verbose=False)
    proc = get_stimulus(spec.process_name)
    pkeys = _proc_keys(proc, spec)
    per_cell = []
    names = spec.cell_names()
    for c, cell in enumerate(spec.cells):
        tp = {**spec.teacher, **{k: v for k, v in cell.items() if k not in pkeys}}
        law = get_teacher(tp.pop("law", spec.law_name)); tp.pop("name", None)
        poles = law.poles(spec.dt, **tp) if hasattr(law, "poles") else None
        rep = spectral_coverage(U[cond == c], spec.dt, poles)
        rep["cell"] = cell
        rep["name"] = names[c]
        per_cell.append(rep)
    prov["excitation"] = {
        "identifiable": all(r["identifiable"] for r in per_cell),
        "n_cells": len(per_cell),
        "unidentifiable_cells": [r["cell"] for r in per_cell if not r["identifiable"]],
        "per_cell": per_cell,
    }
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2, default=str)

    # `teacher.pt` -- the ground truth, in the shape a scoring pass wants it: the poles to
    # compare a fitted circuit's measured ones against, and the law's own coefficients so the
    # target can be recomputed without re-reading the yaml. Per cell, because a grid over the
    # teacher has a different truth in every cell and one file holding only the first would be
    # the same defect the excitation report had.
    truths = []
    for c, cell in enumerate(spec.cells):
        tp = {**spec.teacher, **{k: v for k, v in cell.items() if k not in pkeys}}
        law = get_teacher(tp.pop("law", spec.law_name)); tp.pop("name", None)
        pl = law.poles(spec.dt, **tp) if hasattr(law, "poles") else np.array([])
        truths.append({"cell": cell, "name": names[c], "params": tp,
                       "poles_real": np.real(pl).tolist(), "poles_imag": np.imag(pl).tolist(),
                       "pole_freq_hz": [float(abs(p.imag) / (2 * np.pi)) if abs(p.imag) > 1e-12
                                        else float(abs(p.real) / (2 * np.pi))
                                        for p in np.atleast_1d(pl)]})
    torch.save({"law": spec.law_name, "dt": spec.dt, "channels": spec.channels,
                "names": names, "targets": int(spec.targets),
                "per_cell": truths}, os.path.join(out, "teacher.pt"))

    # A figure per split, written HERE rather than by a later pass, because a corpus nobody
    # looks at is a corpus whose defects are found by a training run instead of by a glance.
    from plexus.tasks.render import render_split
    for split, (U_, Y_, c_) in built.items():
        render_split(spec, split, U_, Y_, c_, prov["excitation"],
                     os.path.join(out, f"task_traces_{split}.png"))

    if verbose:
        e = prov["excitation"]
        verdict = ("IDENTIFIABLE: every pole of every cell sits where the stimulus has power"
                   if e["identifiable"] else
                   f"NOT IDENTIFIABLE in {len(e['unidentifiable_cells'])} of {e['n_cells']} "
                   f"cells -- those modes cannot be recovered from this stimulus, however long "
                   f"the fit runs")
        print(f"[task] excitation: {verdict}")
        for r in e["per_cell"]:
            if e["identifiable"] and len(e["per_cell"]) > 1 and r is not e["per_cell"][0]:
                continue                       # one example is enough when all cells pass
            tag = (" ".join(f"{k}={v}" for k, v in r["cell"].items())) or "(single cell)"
            for q in r["poles"]:
                if e["identifiable"] or not q["excited"]:
                    mark = "ok" if q["excited"] else "NOT EXCITED"
                    print(f"         {tag:28s} pole {q['freq_hz']:8.3f} Hz  "
                          f"{q['input_power_db_rel_peak']:+7.1f} dB rel peak   {mark}")
        print(f"[task] wrote {out}")
    return out


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", help="task yaml(s), or a glob")
    ap.add_argument("--force", action="store_true", help="regenerate even if already present")
    ap.add_argument("--root", default=None, help="overrides graphs_data_path()")
    a = ap.parse_args()
    paths = [p for s in a.specs for p in (glob.glob(s) if any(c in s for c in "*?[") else [s])]
    for p in paths:
        generate(p, root=a.root, force=a.force)


if __name__ == "__main__":
    main()
