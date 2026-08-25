#!/usr/bin/env python
"""Generate `paper/mpm_warp_tables.tex` from measurement, for the note that compares the two MPM
implementations line by line.

NOTHING IN THE TABLES IS TRANSCRIBED. The tensor shapes and byte counts come from a
`TorchDispatchMode` trace of a real substep (`/tmp/trace_ops_<impl>.json`, written by the tracer in
this file's `--trace` mode); the timings come from `tools/mpm_bench.py`'s json rows. A table typed by
hand from a terminal is a table that drifts from the code the first time the code changes, and the
whole point of this note is that the numbers are checkable.

    python tools/mpm_warp_note.py --trace          # re-measure the intermediates on this box
    python tools/mpm_warp_note.py                  # rebuild the tex from whatever json exists
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

# The essential per-particle-per-substep traffic an MLS-MPM cycle CANNOT avoid, in floats. Same
# constant `tools/mpm_bench.py` uses, and for the same reason: it is the denominator that turns
# "megabytes" into "how many times more than necessary".
ESSENTIAL_B = 216 * 4

VIEW_OPS = {"view", "_unsafe_view", "expand", "unsqueeze", "squeeze", "select", "slice",
            "transpose", "detach", "alias", "permute", "reshape", "as_strided", "t"}
ITEMSIZE = {"float32": 4, "int64": 8, "bool": 1, "int32": 4, "float64": 8, "int16": 2}


def esc(s):
    return str(s).replace("_", r"\_")


def load_trace(impl):
    p = f"/tmp/trace_ops_{impl}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def rows_for(trace, tag, floor_mb=40.0):
    """The allocating ops of one operator, largest total first."""
    out = []
    for (key, count) in trace["rows"]:
        t, op, shape, dt = key
        if t != tag or op in VIEW_OPS:
            continue
        nb = ITEMSIZE.get(dt, 4)
        for s in shape:
            nb *= s
        if nb * count >= floor_mb * 1e6:
            out.append((op, shape, dt, nb, count, nb * count))
    return sorted(out, key=lambda r: -r[5])


def per_op_totals(trace):
    tot = collections.Counter()
    for (key, count) in trace["rows"]:
        t, op, shape, dt = key
        if op in VIEW_OPS:
            continue
        nb = ITEMSIZE.get(dt, 4)
        for s in shape:
            nb *= s
        tot[t] += nb * count
    return tot


def tbl_intermediates(trace, tag, caption_lbl):
    N = trace["N"]
    rs = rows_for(trace, tag)
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"aten op & shape & dtype & \multicolumn{1}{c}{each} & \multicolumn{1}{c}{$\times$} "
         r"& \multicolumn{1}{c}{total} \\", r"\midrule"]
    for op, shape, dt, nb, count, tot in rs:
        L.append(f"\\code{{{esc(op)}}} & \\code{{{esc(str(list(shape)))}}} & {esc(dt)} & "
                 f"{nb/1e6:.1f}\\,MB & {count} & {tot/1e6:.1f}\\,MB \\\\")
    grand = per_op_totals(trace)[tag]
    L += [r"\midrule",
          f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{all allocating ops}}}} & "
          f"\\textbf{{{grand/1e6:.0f}\\,MB}} \\\\",
          f"\\multicolumn{{5}}{{@{{}}l}}{{per particle}} & "
          f"\\textbf{{{grand/N:.0f}\\,B}} \\\\",
          f"\\multicolumn{{5}}{{@{{}}l}}{{$\\div$ essential ({ESSENTIAL_B}\\,B)}} & "
          f"\\textbf{{{grand/N/ESSENTIAL_B:.1f}$\\times$}} \\\\",
          r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


CAPTEX = r" \;{\footnotesize +capture}"


def tbl_ablation(path="/tmp/abl.json"):
    if not os.path.exists(path):
        return "% no ablation json"
    rows = json.load(open(path))
    base = rows[0][1]
    L = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
         r"\code{mpm\_scatter} & \code{mpm\_gather} & ms/frame & peak & speedup \\", r"\midrule"]
    for lab, ms, gib in rows:
        sc, _, ga = lab.partition("/")
        sc, ga = sc.strip(), ga.strip()
        cap = "+capture" in ga
        ga = ga.replace("+capture", "").strip()
        b = lambda s: (r"\textbf{" + esc(s) + "}") if s == "warp" else esc(s)
        L.append(f"{b(sc)} & {b(ga)}{CAPTEX if cap else ''} & "
                 f"{ms:.1f} & {gib:.2f}\\,GiB & {base/ms:.1f}$\\times$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def tbl_sweep(pattern, peak, label):
    files = []
    for pat in pattern.split("|"):
        files += sorted(glob.glob(pat))
    rows = []
    for f in files:
        try:
            rows += json.load(open(f))
        except Exception:
            pass
    rows = sorted({r["particles"]: r for r in rows}.values(), key=lambda r: r["particles"])
    if not rows:
        return "% no rows for " + label
    L = [r"\begin{tabular}{@{}rrrrrr@{}}", r"\toprule",
         r"particles & sub & ms/frame & ms/substep & GB/s & \% peak \\", r"\midrule"]
    for r in rows:
        L.append(f"{r['particles']:,} & {r['substeps']} & {r['ms_per_frame']:.1f} & "
                 f"{r['ms_per_substep']:.2f} & {r['gb_per_s']:.1f} & "
                 f"{r['gb_per_s']/peak*100:.1f}\\% \\\\".replace(",", r"\,"))
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# ==========================================================================================================
# THE TRACER. Every tensor the torch operators actually allocate during one real substep.
#
# `TorchDispatchMode` sees every aten call the operator makes, including the ones the source does not
# name -- the broadcast temporaries, the `index` gathers behind `w[:, oidx[:, k], k]`, the int64
# index arithmetic. Reading the source and multiplying shapes by hand misses exactly those, and they
# are most of the cost. VIEWS ARE EXCLUDED: `expand`/`unsqueeze`/`view` return a new tensor object
# over the same storage and move nothing, so counting them would inflate the default's bill with
# bytes it never touches.
# ==========================================================================================================
def trace(impl, spec_rel="config/material/material_3d_water_bench.yaml", device="cuda:1"):
    import tempfile

    import torch
    import yaml
    from torch.utils._python_dispatch import TorchDispatchMode

    import plexus.operators                                              # noqa: F401
    from plexus import engine as E
    from plexus.operators import mpm_ops as M
    from plexus.schema import load
    if impl != "default":
        __import__(f"plexus.operators.mpm_{impl}")
    torch.cuda.init(); torch.zeros(1, device=device)

    rec, order = {}, []

    class Trace(TorchDispatchMode):
        on, tag = False, ""

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if self.on:
                nm = str(func).split(".")[-2] if "." in str(func) else str(func)
                if nm not in VIEW_OPS:
                    for t in (out if isinstance(out, (list, tuple)) else [out]):
                        if isinstance(t, torch.Tensor) and t.numel() > 1000:
                            k = (self.tag, nm, tuple(t.shape),
                                 str(t.dtype).replace("torch.", ""))
                            if k not in rec:
                                order.append(k)
                            rec[k] = rec.get(k, 0) + 1
            return out

    tr = Trace()
    spec = yaml.safe_load(open(os.path.join(ROOT, spec_rel)))
    for o in spec["operators"]:
        if o.get("op") in ("mpm_scatter", "mpm_gather"):
            o.pop("implementation", None)
            if impl != "default":
                o["implementation"] = impl
    for st in spec["schedule"]:
        if isinstance(st, dict) and "substep_dt" in st:
            st["capture"] = False; st.pop("compile", None)
    spec["general"]["n_frames"] = 2; spec["general"]["record_cap"] = 2
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(spec, f); f.close()
    sim = load(f.name); os.unlink(f.name)

    # ONE SUBSTEP ONLY. The first call of each operator is traced and the flag then latches, so the
    # numbers are per substep and the dispatch overhead does not distort the run.
    for cls, tag in ((M.MPMGather, "gather"), (M.MPMScatter, "scatter"),
                     (M.MPMStrain, "strain"), (M.MPMGridUpdate, "grid_update")):
        orig = cls.forward

        def wrap(orig=orig, tag=tag):
            def f2(self, H, mask=None):
                if getattr(H, "_traced_" + tag, False):
                    return orig(self, H, mask)
                setattr(H, "_traced_" + tag, True)
                tr.on, tr.tag = True, tag
                try:
                    return orig(self, H, mask)
                finally:
                    tr.on = False
            return f2
        cls.forward = wrap()

    with tr:
        H, _ = E.run(sim, out_path=None, device=device, progress=False)
    N = H.level("mpm_particle").n
    out = {"N": N, "impl": impl, "rows": [[list(k), rec[k]] for k in order]}
    dst = f"/tmp/trace_ops_{impl}.json"
    json.dump(out, open(dst, "w"))
    tot = per_op_totals(out)
    print(f"\n  {impl:<8} N={N:,}   " + "   ".join(
        f"{k} {v/1e6:.0f} MB ({v/N:.0f} B/p)" for k, v in tot.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "mpm_warp_tables.tex"))
    ap.add_argument("--trace", action="store_true", help="re-measure the intermediates")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()

    if a.trace:
        for impl in ("default", "warp"):
            trace(impl, device=a.device)

    parts = []
    td = load_trace("default")
    tw = load_trace("warp")
    if td:
        parts.append(r"\newcommand{\tblScatterDefault}{%" + "\n" +
                     tbl_intermediates(td, "scatter", "scatter") + "\n}")
        parts.append(r"\newcommand{\tblGatherDefault}{%" + "\n" +
                     tbl_intermediates(td, "gather", "gather") + "\n}")
        tot = per_op_totals(td); N = td["N"]
        parts.append(r"\newcommand{\Ntrace}{" + f"{N:,}".replace(",", r"\,") + "}")
        for k, v in tot.items():
            parts.append(r"\newcommand{\mb" + k.replace("_", "") + "}{" + f"{v/1e6:.0f}" + "}")
            parts.append(r"\newcommand{\bp" + k.replace("_", "") + "}{" + f"{v/N:.0f}" + "}")
        parts.append(r"\newcommand{\mbtotal}{" + f"{sum(tot.values())/1e6:.0f}" + "}")
        parts.append(r"\newcommand{\ampltotal}{" +
                     f"{sum(tot.values())/N/ESSENTIAL_B:.1f}" + "}")
    if tw:
        tot = per_op_totals(tw)
        parts.append(r"\newcommand{\mbtotalwarp}{" + f"{sum(tot.values())/1e6:.0f}" + "}")
        parts.append(r"\newcommand{\ampltotalwarp}{" +
                     f"{sum(tot.values())/tw['N']/ESSENTIAL_B:.1f}" + "}")
        parts.append(r"\newcommand{\warpfusedmb}{" +
                     f"{tot.get('scatter', 0)/1e6 + tot.get('gather', 0)/1e6:.0f}" + "}")
    parts.append(r"\newcommand{\tblAblation}{%" + "\n" + tbl_ablation() + "\n}")
    parts.append(r"\newcommand{\tblSweepAsix}{%" + "\n" +
                 tbl_sweep("/tmp/warp_sweep.json|/tmp/a6000_*.json", 768, "A6000") + "\n}")
    parts.append(r"\newcommand{\tblSweepAhun}{%" + "\n" +
                 tbl_sweep(os.path.join(ROOT, "log/bench/a100_*_warp_cap.json"), 1555, "A100")
                 + "\n}")
    parts.append(r"\newcommand{\tblSweepAhunPlain}{%" + "\n" +
                 tbl_sweep(os.path.join(ROOT, "log/bench/a100_*_warp.json"), 1555, "A100 nocap")
                 + "\n}")
    with open(a.out, "w") as f:
        f.write("% GENERATED by tools/mpm_warp_note.py -- do not edit\n" + "\n\n".join(parts) + "\n")
    print(f"  -> {a.out}   ({len(parts)} macros)")


if __name__ == "__main__":
    main()
