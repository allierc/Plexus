#!/usr/bin/env python
"""caption_wave -- caption a whole wave of runs with ONE model load, on the devcontainer side.

Two problems, one fix.

  1. THE CLUSTER CANNOT CAPTION. The partition's `connectome-gnn` environment has no
     `transformers`, so every cluster job recorded

         VLM caption UNAVAILABLE: ModuleNotFoundError: No module named 'transformers'

     and wrote `UNAVAILABLE` into description.txt. That is the honest behaviour -- it failed
     loudly rather than pretending -- but the consequence is severe: **the Watcher was blind on
     every cluster run**, which is exactly the population a week-long campaign produces. The
     eye/number divergence defence, the thing that has caught more errors in this project than
     any other, was inert where it matters most.

  2. ONE MODEL LOAD PER JOB is wasteful: a 23 GB model loaded once per run, eight times a round.

Captioning therefore runs HERE, after the wave lands: the devcontainer has the model and the
libraries, and one load captions the whole batch. This is still always-caption -- it is not
deferred to an end-of-campaign pass, it is part of closing the round, and a round does not
finish until its captions exist.

    python caption_wave.py r001t_00_5e3159 r001t_01_5e3159 ...
    python caption_wave.py --round 1
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(ROOT, "log", "okuda")


# WHAT THE MOVIE ACTUALLY CONTAINS, told to the model because it cannot know. run_one lays out
# two 3D viewpoints of the SAME vesicle side by side, with a cross-section inset bottom-right.
# Unwarned, the model reads three panels as three objects: the inset becomes a second body beside
# the tissue, and the two viewpoints become two balls. The eye-check is the only role that looks
# at SHAPE, so a layout it misreads corrupts the one channel that is not a number.
OKUDA_LAYOUT = """PANELS, left to right:
  LEFT   a 3D view of the cell vesicle, from the side.
  RIGHT  the SAME vesicle from a second viewpoint (roughly from above). It is not a second object.
  BOTTOM-RIGHT INSET, small  a CROSS-SECTION cut through the middle of that same vesicle, showing
         the cells in the cut plane. Its rectangular border is the inset frame and not a
         structure. What looks like a flat sheet or a ring there is the interior of the ball.
COLOUR: cells are shaded white (low) to red (high) by the activator chemical they carry. A green
  tint means the cell divided recently; magenta means the cell is broken. The colour is a
  measurement painted on the tissue, not a material."""


def caption_wave(names, n_frames=8, force=False):
    """Caption every named run. Returns {name: 'ok' | 'skipped' | 'no_movie' | reason}."""
    todo = []
    for n in names:
        d = os.path.join(LOG, n)
        mp4 = os.path.join(d, "movie.mp4")
        dst = os.path.join(d, "description.txt")
        if not os.path.exists(mp4):
            continue
        if os.path.exists(dst) and not force:
            txt = open(dst, errors="ignore").read()
            if txt and not txt.startswith("UNAVAILABLE"):
                continue                      # a real caption already exists
        todo.append((n, mp4, dst))
    if not todo:
        print("[caption] nothing to do")
        return {}

    sys.path.insert(0, os.path.join(ROOT, "VLLM"))
    t0 = time.time()
    try:
        import torch
        import describe_video as DV
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[caption] loading the VLM ONCE for {len(todo)} run(s) on {dev} ...", flush=True)
        proc = AutoProcessor.from_pretrained(DV.GEMMA)
        model = AutoModelForMultimodalLM.from_pretrained(DV.GEMMA, dtype="bfloat16",
                                                         device_map=dev)
    except Exception as e:
        # NOT silent, and NOT written as a caption: a missing model must never look like a
        # verdict. The runs keep their UNAVAILABLE marker so the ledger can tell "no caption"
        # from "caption agreed".
        print(f"[caption] VLM UNAVAILABLE: {type(e).__name__}: {str(e)[:140]}", flush=True)
        return {n: f"unavailable: {type(e).__name__}" for n, _, _ in todo}

    out = {}
    for i, (n, mp4, dst) in enumerate(todo, 1):
        try:
            txt = DV.describe_one(proc, model, mp4, n_frames, layout=OKUDA_LAYOUT)
            with open(dst, "w") as f:
                f.write(txt if txt else "UNAVAILABLE -- the model returned nothing.\n")
            out[n] = "ok" if txt else "empty"
            first = (txt or "").replace("\n", " ")[:88]
            print(f"  [{i}/{len(todo)}] {n}: {first}...", flush=True)
        except Exception as e:
            out[n] = f"failed: {type(e).__name__}"
            print(f"  [{i}/{len(todo)}] {n}: FAILED {type(e).__name__}", flush=True)
    # FREE THE MODEL. This is the difference between here and VLLM/describe_video.py, and it is
    # not cosmetic: that one is a standalone script, so it exits and the OS reclaims. This runs
    # INSIDE a round, and without the release below the 23 GB stays pinned on cuda:0 for the
    # remaining ten minutes while Act 2's readers and the whole of Act 3 execute. Measured after
    # the recon round: 24,565 MiB still held on GPU 0 with no compute process listed.
    #
    # A round holding a fifth of a GPU it has finished with is a round that can be killed by the
    # next thing that needs one, and a death from outside leaves no traceback -- which is exactly
    # the failure that has not been explained. This does not prove that was the cause. It removes
    # a real way for it to happen, which is the most that can honestly be claimed.
    try:
        del model, proc
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        free = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
        print(f"[caption] model released; {free:.1f} GB still reserved by this process",
              flush=True)
    except Exception as e:
        print(f"[caption] could not release the model: {type(e).__name__}", flush=True)
    print(f"[caption] {len(todo)} run(s) in {time.time() - t0:.0f}s "
          f"(one model load, not {len(todo)})", flush=True)
    return out


def runs_in_round(rid):
    pat = os.path.join(LOG, f"r{rid:03d}*")
    return sorted(os.path.basename(p) for p in glob.glob(pat) if os.path.isdir(p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = a.names or (runs_in_round(a.round) if a.round is not None else [])
    if not names:
        ap.error("give run names or --round N")
    caption_wave(names, force=a.force)
