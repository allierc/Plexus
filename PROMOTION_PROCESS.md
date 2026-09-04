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

Each pair lands as **two sibling directories** — `log/promotion/<phase>_<spec>_A` and `..._B` —
beside `<phase>_<spec>_compare.png` and `_compare.mp4`, the two sides stepped together with the
verdict in the footer.

**Flat, since 4 September, and the old shape is worth knowing because it was actively misleading.**
There used to be a pair directory above the two sides, and inside it each run was kept TWICE: once
where the runner natively wrote it — `<pair>/graphs_data/promotion/<name>/` for the core,
`<worktree>/log/okuda/<name>/` for okuda — and once in the `A/`/`B/` mirror the comparison read.
Measured on one pair, that was the same 7,507,620,532-byte `trajectory.npz` at two inodes. The disk
was the smaller problem: kill a run and the mirror holds one attempt while the native tree holds
another, with nothing in either name to say which. Now the poll copies only the small artefacts so a
run can be watched, and `promote_side` **moves** the rest into the side directory when it lands, so
the trajectory exists once and the directory a name points at is the only copy of it.

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

## What a green row does not prove — the renumber_set defect, 23 August

Nineteen rows were green while the promotion was silently scrambling the chemistry of every run
that killed a cell. The defect is written up here rather than only in a commit message because
**the code fix landed in `7347c4d2`, whose subject is about a figure** — a concurrent session ran a
broad `git add -A` and swept the staged files into its own commit. `git log --oneline` will not
lead anyone here.

`Hierarchy.renumber_set` opened with

```python
lvl = self.levels.get(level_name) if hasattr(self.levels, "get") else None
```

and `Hierarchy.levels` is an **`nn.ModuleDict`, which has no `.get`**. The guard was False on every
call, so the method returned False having touched nothing, and both call sites discarded the bool.
Promotion step 3 did not move the per-death renumber into the engine — it deleted it, and with it
all three permutes: `state`, `occ`, and the pending deltas.

**Why one spec and not the others.** It needs a death operator *and* chemistry. `cell_geometry`
rewrites `cen`, `area` and `occ` from `nF` every tick, so those self-heal a tick later; `chem` has
no rewriter and stays mis-indexed for good. `r023_07` went non-finite at frame 889 and froze at
2,995 cells against the archive's 12,608. `r020_00_ctrl` is the same spec minus `cell_die`, and was
clean throughout.

**Three things that should have caught it, and why none did.**

* **The unit test** built its fixture as `H.levels = {"cell": lvl}` — a plain dict, **which does
  have `.get`**. It drove the live branch while production drove the dead one. *A fixture that
  cannot reproduce production is not a test:* `tools/test_mesh_carry.py` now constructs a real
  `Hierarchy()`, and fails without the fix.
* **The twin rows.** Every row but phase 0 had *both sides on the current tree*, so a change that
  moved both sides identically was invisible. **A suite of same-tree rows cannot detect a
  regression, only a divergence.** At least one row per phase must pin side A to a commit — the
  `BISECT` row (`okuda@0da57dd0` vs `core`) is that row, and it is the one that proved the fix.
* **The return value.** `renumber_set` reported failure by returning False and nobody looked. Both
  call sites now check it, print what a silent failure means, and increment `renumber_failed`,
  which is in `MeshTable.SCALAR_RECORD` — **so a gate can assert it is 0 instead of a human having
  to notice a printed line.**

**What this costs the record:** every gate row and every twin row graded before 23 August was
produced with the broken renumber, and none of them counts until it is re-run.

## NO NEW RECORDED ARRAYS UNTIL THE PROMOTION IS DONE

The pristine side of every row is a worktree at `0da57dd0`. It runs the code of that commit, so it
can only ever write the arrays that commit knew about. **Add a recorded quantity to the core and the
two sides stop being comparable** — not "differ", but *uncomparable*: one side has a key the other
cannot produce, and the harness has nothing to hold it against. The byte-identity gate is the only
instrument that can see a regression here, and a change like this switches it off.

So any change that adds to `MeshTable.FACE_RECORD`, `EDGE_RECORD`, `SCALAR_RECORD`, `snapshot()` or
`topo_record`'s `hist` waits until the promotion's rows are green and the gate is retired or re-based
onto a new pristine commit. **This is a sequencing rule, not a judgement about the change.**

The live case, deferred on 23 August: a persistent per-cell `uid` + `parent_uid`, so a cell can be
followed across the renumbering that `cell_die` and `edge_flip` perform. It is wanted — without it,
"plot the mother's area and her daughters' areas over time" needs the permutation reconstructed from
the recorded `apop` flag, and that reconstruction is unsound because `edge_flip` drops faces without
setting `apop`. It is still deferred, because it would cost the gate that is currently the only thing
standing between this promotion and another silent `renumber_set`.

## The floor

Byte-identity is the criterion because the platform delivers it: two runs of one spec, same code,
1,800 frames, 2,000 → 12,272 cells with division and T1 in the loop, give `max |delta| = 0.0`
exactly. Re-measure with `--phase R` if the queue, the driver or the GPU model changes. A tolerance
would have to be justified by a floor above zero; there is not one.

## A retired instrument: `tools/mpm_identity_gate.py`, 4 September

`config/cell/` — the composed-cell ladder, a nucleus and a cytosol of two protein species and a
membrane, each its own contained set — was **deleted as a dead end**, with `log/cell` and 11 GB of
`graphs_data/cell`. Three of its specs were not science, though; they were **instruments**, and the
gate they served is retired with them rather than left broken.

`mpm_identity_gate.py` answered a question this file's harness deliberately does not: *"I rewrote
the inside of an MPM operator for speed — did any bit move?"* — a reference captured before the edit
and re-checked after, in one command, with no cluster. Its five fixtures were chosen so that the MPM
branches could not hide behind one another, and its own docstring says why three of them had to be
the cell ladder:

    cell_02_nucleus_bounce    3D, ONE particle set     the plain scatter/gather path
    cell_03_nucleus_cytosol   3D, TWO sets             the shared-grid ACCUMULATE path, and CSF
    cell_05_membrane          3D, THREE sets           what makes "who zeroes the grid" a question
                                                       rather than a tautology

    "A change that is identical on cell_02 and wrong on cell_05 is the exact shape of the
     shared-grid bug: one set scattering is a special case in which overwriting and accumulating
     agree."

**So what is now unguarded is precisely that.** The two surviving fixtures — `material_3d_multimaterial`
and `material_two_drops_st` — cover multi-TYPE and 2D, not multi-SET, so the accumulate path has no
byte-identity gate at all. `mpm_warp.py` and `mpm_triton.py` are gated by a TOLERANCE against the
default (atomics are order-dependent and cannot be bit-identical), which is the right gate for them
and the wrong one for a refactor of the default itself.

Rebuilding it means authoring one-, two- and three-set MPM specs under `config/material/` and
showing the gate returns the same verdicts on them. That is real work and it is not on the
apico-basal ladder, so it is recorded here as a gap rather than quietly carried.
