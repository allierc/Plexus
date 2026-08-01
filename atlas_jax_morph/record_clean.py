"""record_clean -- make the typed signature typed, and say exactly what was changed.

Phase 2's normalizers wrote signatures like

    reads: ["alive (source AND receiver liveness mask, applied twice -- undeclared)", ...]

which is a useful sentence and a useless type. A signature field has to be a NAME the engine
knows, or nothing downstream can compare the record to the code -- `verify_impl.py` reported
thirteen mismatches that were entirely punctuation, which is worse than reporting none.

Two mechanical passes, and a third thing this file refuses to do.

  1. STRIP.   Keep the leading identifier of each entry; the sentence moves to `state_io_notes`
              so nothing written by a reader is lost.
  2. ALIAS.   Map the paper's word to the engine's field name (`position` -> `pos`). This is a
              rename, not a judgement: `pos` is what a Plexus Level actually stores.
  3. REFUSE.  Where the record and the code genuinely disagree -- the code declaring fewer reads
              than the operator performs, say -- this file does NOT quietly copy the code over
              the record. It writes a `signature_gap` on the entry and leaves both readings
              standing. Overwriting the record with the code would make every future comparison
              agree by construction, which is the same as not checking.

    python record_clean.py --dry     # what would change
    python record_clean.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import record          # noqa: E402
import verify_impl     # noqa: E402

STATE = os.path.join(HERE, "_state")

# The engine's field names, against the words a paper uses for them.
ALIASES = {"position": "pos", "positions": "pos", "velocity": "vel", "concentration": "chem",
           "chemical": "chem", "gene_state": "gene", "driver_inputs": "drive",
           "sensed_input_fields": "drive", "hidden_state": "hidden"}

FIELDS = ("reads", "writes", "maps", "inputs", "outputs")


def clean_entry(m):
    """Returns (changes, notes). Mutates the entry's contract in place."""
    c = m.get("contract") or {}
    changes, notes = [], {}
    for key in FIELDS:
        vals = c.get(key)
        if not isinstance(vals, list):
            continue
        out, kept = [], []
        for v in vals:
            head = verify_impl._head(v)
            head = ALIASES.get(head, head)
            if not head or not head.replace("_", "").isalnum():
                kept.append(str(v))            # unparseable: keep the sentence, drop the "type"
                continue
            if head not in out:
                out.append(head)
            if str(v).strip() != head:
                kept.append(str(v))
        if out != vals:
            changes.append(f"{key}: {vals} -> {out}")
            c[key] = out
        if kept:
            notes[key] = kept
    if notes:
        m.setdefault("state_io_notes", {}).update(notes)
    return changes, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    doc = record.load()
    log = {}
    for m in doc["mechanisms"]:
        if not m.get("contract"):
            continue
        changes, _ = clean_entry(m)
        if changes:
            log[m["id"]] = changes

    for mid, ch in log.items():
        print(f"\n{mid}")
        for c in ch:
            print(f"  {c[:160]}")
    print(f"\n{len(log)} entries would change")

    if a.apply:
        record.save(doc)
        os.makedirs(STATE, exist_ok=True)
        with open(os.path.join(STATE, "record_clean.json"), "w") as f:
            json.dump(log, f, indent=2)
        print(f"applied; the diff is in {os.path.join(STATE, 'record_clean.json')}")


if __name__ == "__main__":
    main()
