"""Exhaustive slime test suite -- runs every spec, renders gif + montage, checks
intent, and writes results.md + run.log + gallery.png (mirrors prototype/water).

    PYTHONPATH=../../src python run_slime.py            # full suite
    PYTHONPATH=../../src python run_slime.py slime_two_repel slime_three   # subset

Intent (not just "it ran"):
  coverage  -- fraction of the domain inked at the end (network exists, not blank).
  contrast  -- std/mean over inked pixels (filaments give high contrast vs a blob).
  reinforce -- peak-trail late/early (the field self-reinforces, doesn't fade out).
  overlap   -- multi-species only: mean pairwise channel overlap (LOW = segregated).
A run PASSES if a persistent, structured network formed; multi-species repel runs
additionally require low overlap, attract runs require high overlap.
"""

import os
import sys
import traceback

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_slime

# spec name -> (one-line description, intent rule)
SUITE = [
    # single-species morphology
    ("slime_default",     "baseline Physarum network: a disc reticulates into a web"),
    ("slime_fine",        "short reach + low diffuse -> fine capillary mesh"),
    ("slime_coarse",      "long reach -> few large transport cells"),
    ("slime_filaments",   "narrow cone + low turn -> long straight filaments"),
    ("slime_curly",       "wide cone + high turn -> curly foam texture"),
    ("slime_sparse",      "high decay -> faint, transient sparse trails"),
    ("slime_dense",       "low decay + strong deposit -> thick space-filling web"),
    ("slime_smear",       "very high diffuse -> trails smear into blobs"),
    # spawn modes
    ("slime_ring_in",     "ring spawn pointing inward (Lague signature)"),
    ("slime_point",       "single point source -> radiating burst"),
    ("slime_random_full", "uniform seed -> global space-filling network"),
    ("slime_torus",       "periodic boundary -> seamless endless network"),
    # multi-species (categorical separation)
    ("slime_two_repel",   "2 species, cross=-1 -> territorial segregation"),
    ("slime_two_attract", "2 species, cross=+1 -> shared co-aligned network"),
    ("slime_two_asym",    "2 species, different settings, repelling"),
    ("slime_three",       "3 mutually-repelling species -> triple junctions"),
    ("slime_four",        "4 mutually-repelling species (full RGBA showcase)"),
    ("slime_six",         "6 mutually-repelling species -> six colour territories"),
    ("slime_eight",       "8 mutually-repelling species -> eight colour territories"),
]


def verdict(name, m):
    structured = (m["coverage"] > 0.02 and m["contrast"] > 0.25 and m["reinforce"] > 0.5)
    if m["overlap"] is None:
        return "PASS" if structured else "WEAK"
    if "attract" in name:
        return "PASS" if (structured and m["overlap"] > 0.30) else "WEAK"
    return "PASS" if (structured and m["overlap"] < 0.35) else "WEAK"


def main(names):
    descmap = dict(SUITE)
    log = open("run.log", "w")
    rows = []
    finals = []
    for name in names:
        desc = descmap.get(name, name.replace("sweep_", "sweep: ").replace("_", " "))
        try:
            m = render_slime.render(name, device=os.environ.get("SLIME_DEVICE", "cuda"))
            v = verdict(name, m)
            ov = "-" if m["overlap"] is None else "%.3f" % m["overlap"]
            line = ("%-20s | %s | cover=%.3f contrast=%.2f reinforce=%.2f overlap=%s | %s"
                    % (name, v, m["coverage"], m["contrast"], m["reinforce"], ov, desc))
            rows.append((name, v, m, ov, desc))
            finals.append("specs/%s_montage.png" % name)
        except Exception as e:                            # a failure is data, not a crash
            line = "%-20s | ERROR | %s" % (name, e)
            traceback.print_exc()
        print(line, flush=True)
        log.write(line + "\n"); log.flush()
    log.close()

    # results.md table
    with open("results.md", "w") as f:
        f.write("# Slime test suite (Plexus port of SebLague/Slime-Simulation)\n\n")
        f.write("name | verdict | coverage | contrast | reinforce | overlap | notes\n")
        f.write("---|---|---|---|---|---|---\n")
        for name, v, m, ov, desc in rows:
            f.write("%s | %s | %.3f | %.2f | %.2f | %s | %s\n"
                    % (name, v, m["coverage"], m["contrast"], m["reinforce"], ov, desc))

    # gallery: final-frame thumbnails stacked
    thumbs = []
    for name, *_ in rows:
        p = "specs/%s_montage.png" % name
        if os.path.exists(p):
            im = Image.open(p).convert("RGB")
            w, h = im.size
            thumbs.append((name, im.crop((w - w // 6, 0, w, h))))      # last frame panel
    if thumbs:
        tw, th = thumbs[0][1].size
        cols = 4
        rows_n = (len(thumbs) + cols - 1) // cols
        gal = Image.new("RGB", (tw * cols, th * rows_n), "black")
        for k, (_, im) in enumerate(thumbs):
            gal.paste(im, ((k % cols) * tw, (k // cols) * th))
        gal.save("gallery.png")
        print("[gallery] %d panels -> gallery.png" % len(thumbs))


if __name__ == "__main__":
    import glob
    if sys.argv[1:]:
        names = sys.argv[1:]
    else:                                                 # all specs: curated SUITE first, then sweeps
        all_specs = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob("specs/*.yaml"))
        curated = [n for n, _ in SUITE if n in all_specs]
        names = curated + [n for n in all_specs if n not in set(curated)]
    main(names)
