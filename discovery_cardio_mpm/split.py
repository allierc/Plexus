#!/usr/bin/env python
"""split -- decide what a run may see, once, in writing, before anything is fitted.

WHY THIS IS ITS OWN STEP, AND WHY IT IS LAST IN PHASE 1
================================================================================================
Sixty batches fitted one beat of one specimen and scored the same beat of the same specimen. The
diseased sheet sat on disk the whole time and was never opened. There was no split, so there was
never a moment at which someone could have chosen one badly -- and that is the point: **a split
invented after a result is a split negotiated with the result already in hand.**

So it is fixed here, in a file, with checksums, before a single fit is run. Everything downstream
reads it and nothing may recompute it.

It is also **irreversible**, which is why it is the last item of Phase 1: once the diseased sheet
is sealed, the value of the one-shot test depends on nobody having looked. That includes us.

SEALING BY CONTENT, NOT BY PATH
------------------------------------------------------------------------------------------------
The dataset holds two specimens under five filenames. `Cardio_0/derivatives.npy` sounds like a
third specimen and is a second registration of the diseased one; `diseased.npy` is a third.
**Sealing a path would seal nothing** -- the same measurement is reachable under names that do not
say so. So the seal is a fingerprint of the DISPLACEMENT FIELD, and any file whose content matches
is sealed however it is named.

WHAT IS BEING GIVEN UP, STATED PLAINLY
------------------------------------------------------------------------------------------------
A held-out BEAT is not a real test -- measured, not assumed: predict any one beat from the mean of
the other three and you already score 0.98. The beats are near-copies. Held-out beats are kept in
the split anyway, because they are the only way to measure whether a fit has absorbed beat-to-beat
variation, but **they may not be cited as evidence of generalisation.** The only genuine held-out
specimen is the diseased sheet, and there is exactly one of it.

    python split.py --freeze     # write it, once
    python split.py --check      # verify nothing has moved, and that the seal holds
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import data as D                                                    # noqa: E402

SPLIT = os.path.join(HERE, "_data", "split.json")
SOURCE_ROOT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"

# Every file on disk that could carry the diseased specimen, whatever it is called. The seal is by
# content; this list only says where to look when computing the fingerprints.
DISEASED_CANDIDATES = [
    os.path.join(SOURCE_ROOT, "Cardio_1", "1_HCM_15kPa_MR44_W3_1_MMStack_Pos0.ome.tif.derivatives.npy"),
    os.path.join(SOURCE_ROOT, "Cardio_0", "derivatives.npy"),
    os.path.join(SOURCE_ROOT, "diseased.npy"),
]


class SealBroken(RuntimeError):
    """Raised when a run reaches for the held-out specimen."""


def _fingerprint_npy(path, max_frames=60):
    """A content id for a derivative stack or a plain array, comparable across file formats.

    Uses the displacement field referenced to frame 0, rounded, exactly as `data.specimen_id`
    does, so a file re-saved in another layout still fingerprints the same.
    """
    A = np.load(path, mmap_mode="r")
    A = np.asarray(A[:max_frames]).astype(np.float64)
    if A.ndim == 4 and A.shape[-1] >= 2:                        # [T,H,W,C] derivative stack
        A = A[..., 0:2].reshape(A.shape[0], -1, 2)
    d = np.round((A - A[0]) * 1e6).astype(np.int64)
    return hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest()


def build():
    z = D.open_npz(expect_sha256=D.HEALTHY_POS_SHA256)
    P = z["pos"].astype(np.float64)
    b = D.beats(P)
    onsets = b["onsets"]

    # Complete beats only. The stretch after the last onset is NOT a beat and is excluded by name
    # rather than by accident -- the inherited code treated it as one.
    spans = [[int(onsets[i]), int(onsets[i + 1])] for i in range(len(onsets) - 1)]
    G = min(e - s for s, e in spans)

    # The fit beat is the one the previous campaign used (`--fit_beat -2` resolves to onset 152),
    # kept deliberately so a comparison against the old record is honest rather than convenient.
    fit_i = 3
    fit = spans[fit_i]
    held = [s for i, s in enumerate(spans) if i != fit_i]

    fit_frames = list(range(fit[0], fit[0] + G))
    held_frames = sorted({f for s in held for f in range(s[0], s[0] + G)})
    overlap = sorted(set(fit_frames) & set(held_frames))

    # The evaluation mask is frozen HERE, once, from the recording alone -- never derived from a
    # model setting. The inherited mask was defined through the Dirichlet band width, so sweeping
    # the band changed the model AND the set of nodes the score was computed on at the same time.
    amp = np.linalg.norm(P - P[0], axis=-1).max(axis=0)
    thresh = float(0.2 * np.percentile(amp, 99))
    mov = amp > thresh

    diseased = []
    for p in DISEASED_CANDIDATES:
        if os.path.exists(p):
            diseased.append({"path": p, "content_id": _fingerprint_npy(p),
                             "bytes": os.path.getsize(p)})

    split = {
        "written": "frozen by split.py --freeze",
        "recording": {"path": D.DEFAULT_NPZ, "pos_sha256": D.HEALTHY_POS_SHA256,
                      "specimen_id": D.specimen_id(P)},
        "beats": {"onsets": onsets, "gaps": b["gaps"], "mean_gap": b["mean_gap"],
                  "complete_spans": spans, "common_length": int(G),
                  "excluded_tail": [int(onsets[-1]), int(P.shape[0])],
                  "note": "the stretch after the last onset is not a beat and is excluded"},
        "fit": {"beat_index": fit_i, "span": fit, "frames": [fit_frames[0], fit_frames[-1]],
                "n_frames": len(fit_frames)},
        "heldout_beats": {"spans": held, "n_frames": len(held_frames),
                          "caveat": "leave-one-beat-out R2 is 0.978-0.986: the beats are "
                                    "near-copies. These may NOT be cited as evidence of "
                                    "generalisation. They measure absorbed beat-to-beat variation."},
        "disjoint": {"overlap_frames": overlap, "ok": len(overlap) == 0},
        "eval_mask": {"rule": "amplitude > 0.2 * p99(amplitude), over the whole recording",
                      "threshold": thresh, "n_nodes": int(mov.sum()),
                      "n_total": int(mov.size),
                      "sha256": hashlib.sha256(np.packbits(mov).tobytes()).hexdigest(),
                      "note": "frozen from the RECORDING alone. Never derive it from a model "
                              "setting: the inherited mask was defined through the band width, so "
                              "sweeping the band moved the model and the ruler together."},
        "sealed": {"specimen": "diseased sheet (HCM)", "files": diseased,
                   "opened": False,
                   "rule": "sealed by CONTENT. Two specimens sit on disk under five filenames; "
                           "sealing a path would seal nothing."},
        "ceiling": {"note": "measured by data_report.py; no model may be scored above it",
                    "loopscore_beat_to_beat_median": 0.7054},
    }
    return split, mov


def freeze(force=False):
    os.makedirs(os.path.dirname(SPLIT), exist_ok=True)
    if os.path.exists(SPLIT) and not force:
        print(f"[split] already frozen: {SPLIT}\n"
              f"        Refusing to overwrite. A split that can be rewritten is not a split.")
        return 1
    split, mov = build()
    if not split["disjoint"]["ok"]:
        print(f"[split] REFUSED: fit and held-out frames overlap: {split['disjoint']['overlap_frames'][:8]}")
        return 1
    np.save(os.path.join(HERE, "_data", "eval_mask.npy"), mov)
    body = json.dumps(split, indent=1, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    with open(SPLIT, "w") as f:
        f.write(body)
    with open(SPLIT + ".sha256", "w") as f:
        f.write(digest + "\n")
    print(f"[split] frozen -> {SPLIT}")
    print(f"        fit beat {split['fit']['span']} ({split['fit']['n_frames']} frames), "
          f"{len(split['heldout_beats']['spans'])} held-out beats, "
          f"{split['eval_mask']['n_nodes']} scored nodes")
    print(f"        sealed: {len(split['sealed']['files'])} files carrying the diseased specimen")
    print(f"        sha256 {digest[:32]}")
    return 0


def load():
    if not os.path.exists(SPLIT):
        raise FileNotFoundError(f"no split has been frozen: {SPLIT} (run split.py --freeze)")
    return json.load(open(SPLIT))


def assert_not_sealed(path, unseal_token=None):
    """Raise if `path` carries the sealed specimen. Called by anything that opens an array.

    The check is on CONTENT, so renaming the file does not evade it. An `unseal_token` is required
    to proceed, and creating one is a deliberate act recorded in the split.
    """
    if not os.path.exists(SPLIT):
        return                                                   # nothing sealed yet
    s = load()
    if unseal_token and unseal_token == s.get("unseal_token"):
        return
    try:
        fid = _fingerprint_npy(path)
    except Exception:
        return                                                   # not an array we can fingerprint
    for f in s["sealed"]["files"]:
        if f["content_id"] == fid:
            raise SealBroken(
                f"SEALED: {path}\n"
                f"  content matches the held-out {s['sealed']['specimen']} "
                f"(id {fid[:16]}).\n"
                f"  It is sealed by CONTENT, so renaming does not help. Opening it before the "
                f"prediction is registered spends the project's only one-shot test.")


def check():
    if not os.path.exists(SPLIT):
        print(f"[split] FAIL -- nothing frozen yet: {SPLIT}")
        return 1
    s = load()
    body = json.dumps(s, indent=1, sort_keys=True)
    want = open(SPLIT + ".sha256").read().strip()
    got = hashlib.sha256(body.encode()).hexdigest()
    ok = True

    def row(name, good, detail=""):
        nonlocal ok
        ok = ok and good
        print(f"  [{'  ok  ' if good else ' FAIL '}] {name:<44s} {detail}")

    row("the split file is unaltered", got == want, got[:24])
    row("fit and held-out frames are disjoint", s["disjoint"]["ok"],
        f"{s['fit']['n_frames']} fit / {s['heldout_beats']['n_frames']} held-out frames")
    row("the recording is the one declared",
        D.specimen_id(D.open_npz(expect_sha256=D.HEALTHY_POS_SHA256)["pos"])
        == s["recording"]["specimen_id"], s["recording"]["specimen_id"][:16])
    row("the evaluation mask is on disk",
        os.path.exists(os.path.join(HERE, "_data", "eval_mask.npy")),
        f"{s['eval_mask']['n_nodes']} nodes")
    row("the seal names files by content", all("content_id" in f for f in s["sealed"]["files"]),
        f"{len(s['sealed']['files'])} files")
    row("the seal is unopened", not s["sealed"].get("opened"), "")
    # the seal must actually refuse
    fired = False
    for f in s["sealed"]["files"]:
        if os.path.exists(f["path"]):
            try:
                assert_not_sealed(f["path"])
            except SealBroken:
                fired = True
            break
    row("the seal REFUSES the held-out specimen", fired, "watched firing")
    print(f"\n  SPLIT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split (do not)")
    a = ap.parse_args(argv)
    if a.freeze:
        return freeze(a.force)
    return check()


if __name__ == "__main__":
    sys.exit(main())
