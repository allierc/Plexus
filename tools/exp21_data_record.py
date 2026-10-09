"""The watcher record of exp 21's DATA analyses -- the salivary gland's own cell motion (Wang et al. 2021 tracks) and
any figure written into experiments/exp21_mpm_epithelium/figs/ -- so the human follows the data as it is analysed,
beside the runs.

    PYTHONPATH=src:tools python tools/exp21_data_record.py

The runs' record (experiments/exp21_mpm_epithelium/) is DERIVED from the experiment's results by tools/exp_record.py,
which removes any step no run backs; a figure therefore gets a record of its own, experiments/exp21-gland-data/, in
the same layout (png/NNNN.png, why/NNNN.txt, mp4/NNNN.mp4, journal.txt), which the watcher lists beside exp 21.

Step NNNN is the Nth figure of figs/ in the order it was written (its mtime). The step's `why` is the figure's sidecar
`<figure>.txt` when there is one (what it shows and its numbers), else the figure's name; a same-named `.mp4` beside
the figure becomes the step's movie. Links are relative, so nothing is copied. Rerun after every new figure: steps
no figure backs any more are removed.
"""
from __future__ import annotations

import glob
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "experiments", "exp21_mpm_epithelium", "figs")
DST = os.path.join(ROOT, "experiments", "exp21-gland-data")


def build() -> int:
    figs = sorted(glob.glob(os.path.join(SRC, "*.png")), key=lambda p: (os.path.getmtime(p), p))
    for sub in ("png", "why", "mp4", "caption"):
        os.makedirs(os.path.join(DST, sub), exist_ok=True)
    keep = set()
    journal = []
    for i, f in enumerate(figs, start=1):
        n = f"{i:04d}"
        stem = os.path.splitext(f)[0]
        name = os.path.basename(stem)
        for sub, src, ext in (("png", f, ".png"), ("mp4", stem + ".mp4", ".mp4")):
            dst = os.path.join(DST, sub, n + ext)
            if os.path.lexists(dst):
                os.remove(dst)
            if os.path.exists(src):
                os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
                keep.add(dst)
        side = stem + ".txt"
        text = open(side).read().strip() if os.path.exists(side) else name.replace("_", " ")
        w = os.path.join(DST, "why", n + ".txt")
        with open(w, "w") as fh:
            fh.write(f"step   {n}  =  {name}  (experiments/exp21_mpm_epithelium/figs/{os.path.basename(f)})\n\n{text}\n")
        keep.add(w)
        first = text.splitlines()[0] if text else name
        journal.append(f"{n}  {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(f)))}  {name}  -- {first}")
    for sub in ("png", "why", "mp4"):
        for p in glob.glob(os.path.join(DST, sub, "*")):
            if p not in keep:
                os.remove(p)
    with open(os.path.join(DST, "journal.txt"), "w") as fh:
        fh.write("\n".join(journal) + ("\n" if journal else ""))
    return len(figs)


if __name__ == "__main__":
    print(f"[exp21 data record] {DST}: {build()} step(s)")
