# The comparison process

The loop the promotion runs on, written down because it spans many comparisons over many sessions
and a process that lives in a conversation is lost at the next context boundary.

    Cedric, 23 August:
      1/ run a series of comparisons
      2/ two issues:
         2a the operator is missing or needs modification to be compared to okuda
            -> proceed with the codebase modification and redo the comparison until it passes
         2b the comparison fails
            -> redo the comparison by SUCCESSIVE ABLATIONS to isolate the issue. Start with the
               simple comparison, ablate most of the operators; if it is ok, do another iteration
               adding one operator at a time. Run a batch of ablations to isolate the operator.

---

## 1. Run a series of comparisons

```
python tools/promotion_identical.py --phase ECM  --batch 8      # the 13 runnable archived rigs
python tools/promotion_identical.py --phase BASE --batch 8      # a 20-row ladder through log/okuda
python tools/promotion_identical.py --phase G    --batch 8      # every lifted gate spec
```

Both sides run **fresh**, **together**, on `gpu_l4`. A stored okuda result is never the reference:
`log/okuda` is a live tree that `round.py` and `staged.py` still write into, so a promotion checked
against a file there proves the file has not changed, which is not the question. `--batch N`
submits `N` pairs at a time and waits for a slot, so eighty jobs do not queue behind each other with
the first failure invisible until the last one lands.

Each pair lands in `log/promotion/<phase>_<spec>/` with `A/`, `B/`, `compare.png` and `compare.mp4`
— the two sides side by side, stepped together, verdict in the footer.

## 2a. The operator is missing, or needs changing

The symptom is a **side that does not run**, not a side that disagrees, and the error names it:

| what you see | what it is |
|---|---|
| `operator 'X' not in registry` | a rename that left the corpus behind. Three so far: `seed_mesh`, `seed_cell_chem`, `seed_ecm`. Fix by registering the archived spelling as an **alias**, canonical name first. |
| `KeyError: 'vertex'` on the okuda side | `run_one.py` is a MESH harness — it reads `H.level("vertex")` in three places. A mesh-free spec has no okuda twin; compare `core@<ref>` against `core` instead and label it a regression check. |
| `FileNotFoundError` on a cluster path | the spec copy wrote a devcontainer path. Everything crossing to a node goes through `cluster.cpath`. |
| `simulation missing required key: 'fields'` | the archived spec predates the requirement. Check first whether it is a spec at all — 80 of the 96 `log/okuda_ECM` directories hold prose, not specs. |

**Then redo the comparison.** A fix is not a fix until its row is green, and the row must be re-run,
not remembered.

## 2b. The comparison fails — bisect by ablation

A `DIFFER` on a 15-operator spec names a spec, not a cause. `tools/ablate.py` turns it into a cause:

```
python tools/ablate.py --spec base/r015_06 --batch 8
```

It builds a **ladder**: rung 0 is the spec stripped to its seeds and the minimum that will run, and
each later rung adds back **one operator in schedule order**. Every rung is a full A/B pair, and the
ladder is submitted in batches. The first rung that DIFFERs names the operator — the one added at
that rung — and the rung below it is the largest configuration that still agrees.

Two rules that make the answer trustworthy:

* **The rungs are generated from the failing spec itself**, never retyped. A ladder whose rung 3
  differs from the original in a second place has isolated the wrong thing.
* **A rung that fails to RUN is not a rung that DIFFERS.** They are reported apart, because
  "removing `cell_geometry` makes the spec invalid" and "adding `cell_geometry` breaks agreement"
  point at opposite ends of the codebase.

### Before believing the answer

Check the comparison before blaming the model. Three of the DIFFERs so far were the harness:

* **the crop** — okuda slices `chemf[t][:nF]`; slicing by `cell__occ[t].sum()` differs by one cell
  around a death, and reported `act_20: shape (2529,) vs (2530,)` on two runs that agreed at every
  recorded tick.
* **the frame set** — okuda keeps ~60 rows, the core keeps every tick. Compare at okuda's `ticks`.
* **a side that did not finish** — a killed plot stage leaves a complete trajectory and a `3d.png`
  that is a mid-run live snapshot.

And check the run before blaming the promotion: `r023_07` is half NaN from frame 889 **on both
sides**, and its own `diag.json` records that and stamps `valid_evidence: True` anyway.

## The floor

Byte-identity is the criterion because the platform delivers it: two runs of one spec, same code,
1,800 frames, 2,000 → 12,272 cells with division and T1 in the loop, give `max |delta| = 0.0`
exactly. Re-measure with `--phase R` if the queue, the driver or the GPU model changes. A tolerance
would have to be justified by a floor above zero; there is not one.
