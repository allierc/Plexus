#!/usr/bin/env python
"""findings -- the register of CLAIMS: what we asserted, what happened to it, and why.

WHY THIS EXISTS
================================================================================================
Cedric, on watching a claim of mine be born, challenged, tested and retracted inside one afternoon:
*"this is a genuine story -- how not to lose it, since we have not started the agentic loop, not
started the knowledge."*

He is right that we had nowhere to put it. The campaign already records:

    run_record.py   every RUN         -- inputs, outputs, a content hash
    scoreboard.py   every MORPHOLOGY  -- what we can reproduce, what the map covers
    PREMISES.md     every ASSUMPTION  -- what we take as known about tissue

and nothing at all records every CLAIM. So the most valuable thing produced on 31 July -- not a
result, but a full cycle of assert / challenge / measure / retract / find the cause -- existed only
in a git message, a session log and a README. The loop will read none of those.

WHAT MAKES THIS DIFFERENT FROM A RESULTS TABLE
------------------------------------------------------------------------------------------------
A retracted claim is not deleted. It stays, with its cause of death attached. That is deliberate,
and it is the whole point:

  * The claim "the buckling transition is physical" was WRONG, and the reason it survived for
    hours -- the genus check passed, and genus cannot see self-intersection -- is worth more than
    the claim would have been if it were right. It produced premise 11.
  * A register that only keeps its winners teaches nothing about how it goes wrong. This campaign
    has retracted four conclusions; each retraction changed an instrument.
  * When the loop finally runs, its Reflection and Meta-review roles need to read what kinds of
    mistake this campaign actually makes. That is not derivable from a list of successes.

THE LIFECYCLE a claim can have
------------------------------------------------------------------------------------------------
    open        asserted, not yet tested
    standing    tested and survived a genuine attempt to kill it
    retracted   killed. `killed_by` says what killed it and `cause` says why it was believed
    superseded  replaced by a sharper statement (`superseded_by`)

A claim with no `evidence` cannot be `standing`. A claim that has never faced an attempt to refute
it is `open`, not `standing` -- surviving because nobody looked is not survival.

    python findings.py                 # the register, newest first
    python findings.py --standing      # only what currently holds
    python findings.py --lessons       # the retractions and what each one changed
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "_findings.jsonl")

STATUS = ("open", "standing", "retracted", "superseded")


def add(fid, claim, status, date, evidence=None, challenged_by=None, killed_by=None,
        cause=None, survives=None, changed=None, superseded_by=None, tags=None):
    """Append one claim. The ledger is append-only: an entry is amended by adding a NEW entry that
    supersedes it, never by editing history. A claim whose fate we later learn gets a second row."""
    if status not in STATUS:
        raise ValueError(f"status must be one of {STATUS}")
    if status == "standing" and not evidence:
        raise ValueError(f"{fid}: a claim with no evidence cannot be 'standing'")
    if status == "retracted" and not killed_by:
        raise ValueError(f"{fid}: a retracted claim must say what killed it")
    rec = dict(id=fid, claim=claim, status=status, date=date, evidence=evidence,
               challenged_by=challenged_by, killed_by=killed_by, cause=cause,
               survives=survives, changed=changed, superseded_by=superseded_by,
               tags=tags or [])
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load():
    """Current state of every claim: the LAST row for each id wins, history preserved on disk."""
    if not os.path.exists(LEDGER):
        return []
    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    latest = {}
    for r in rows:
        latest[r["id"]] = r
    return list(latest.values())


def _wrap(t, w, ind):
    out, line = [], ""
    for word in str(t).split():
        if len(line) + len(word) + 1 > w:
            out.append(ind + line); line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(ind + line)
    return out


def show(rows, lessons_only=False):
    mark = {"standing": "STANDS", "retracted": "RETRACTED", "open": "open", "superseded": "superseded"}
    for r in sorted(rows, key=lambda x: x["id"]):
        if lessons_only and r["status"] != "retracted":
            continue
        print(f"\n{r['id']}  [{mark.get(r['status'], r['status'])}]  {r['date']}")
        for l in _wrap(r["claim"], 92, "    "):
            print(l)
        for key, label in (("evidence", "evidence "), ("challenged_by", "challenged"),
                           ("killed_by", "killed by"), ("cause", "why believed"),
                           ("survives", "survives "), ("changed", "changed  ")):
            if r.get(key):
                ls = _wrap(r[key], 78, "")
                print(f"      {label}: {ls[0].strip()}")
                for l in ls[1:]:
                    print(f"                  {l.strip()}")


# ------------------------------------------------------------------ seed: what we learned by hand
def seed():
    """The claims made on 31 July, before the loop existed to make them. Recorded in the same form
    the loop will use, so its first round inherits a register rather than starting empty."""
    if os.path.exists(LEDGER):
        print(f"ledger already exists at {LEDGER} -- not reseeding")
        return
    D = "2026-07-31"

    add("F001", "The 32-run overnight study shows that making the chemistry shape the tissue and "
                "keeping the tissue intact are mutually exclusive.",
        "retracted", D,
        evidence="32 runs sweeping five settings.",
        killed_by="Arithmetic. Every run ended at exactly 1778 cells = (3552+4)/2, the vertex "
                  "buffer. Every run hit the wall and stopped.",
        cause="The identical endpoint was noticed and written down as 'remarkable' instead of "
              "being questioned. A constant across a sweep is a rail, not a result.",
        changed="Buffers sized for the destination; a run that saturates now says so.",
        tags=["retracted", "instrument"])

    add("F002", "13 of 24 runs peaked only after the mesh had already broken, so their protrusion "
                "measurements are invalid.",
        "retracted", D,
        evidence="A blended damage score crossing threshold before the protrusion peak.",
        challenged_by="Cedric: the movie looks fine.",
        killed_by="corr(hollow_n, n_tip) = +0.971. The blend was counting CELL DIVISION as damage.",
        cause="A composite indicator was trusted without checking what dominated it.",
        changed="The evidence horizon keys on broken_n alone and refuses the blend.",
        tags=["retracted", "instrument"])

    add("F003", "A tube can be drawn from existing tissue by reshaping it; the body need not grow.",
        "retracted", D,
        evidence="Runs at rho = 0 producing protrusions.",
        challenged_by="Cedric: 'that is a strong mistake, cells grow by external nutriment, not "
                      "reshaping only'.",
        killed_by="Physiology, not measurement. A tissue that adds no material can only move it.",
        cause="rho = 0 was treated as a parameter setting rather than as a claim about biology.",
        changed="Premise 1 (cells grow by taking material in) and its check; the baseline growth "
                "floor is above zero unless an ablation is declared.",
        tags=["retracted", "biology"])

    add("F004", "The buckling transition in mini_grow_divide_bigger is physical: a growing shell "
                "accumulates stress, then buckles and releases it.",
        "retracted", D,
        evidence="Sharp transition at ~3500 cells; broken_n 0 throughout; genus 0 and euler 2 at "
                 "every frame; survives quadrupling relax_iters (30 -> 120), so not a solver "
                 "artefact; residual force barely falls with 4x the work, so not unconverged lag.",
        challenged_by="Cedric asked whether 'the tissue is at force balance at every instant' was "
                      "too strong a prior -- can a tissue not accumulate stress for a long time? "
                      "Then a 33-agent adversarial review flagged the geometry.",
        killed_by="Ray-casting from the tissue centroid. 100% of rays cross the surface exactly "
                  "once at frame 384; 0% at frame 423, median 13 crossings, rising to 17. The "
                  "sheet folds through itself. Then, with the cause fixed, mean cell volume is "
                  "FLAT (7.42 -> 7.55) where it had fallen 7.00 -> 3.16 -- so the compression was "
                  "not real either.",
        cause="The topology check passed and was mistaken for a geometry check. Euler "
              "characteristic is COMBINATORIAL: it reads connectivity, never coordinates, so it "
              "reports 'sphere (as built)' for a shell crumpled through its own centre. I had "
              "written that caveat down myself and then under-weighted it.",
        survives="The relax_iters test was sound for what it tested -- the transition was not a "
                 "SOLVER artefact. It was a MODEL artefact, a wrong term in the energy, and I did "
                 "not distinguish the two. Also survives: at 5240 cells the corrected run still "
                 "fails premise 7 (top 5% of cells at shape index 4.60), which is mild, genuine, "
                 "and the first thing a premise check found on its own.",
        changed="Premise 11 (a tissue cannot pass through itself) and the ray test that enforces "
                "it, kept deliberately separate from premise 9 because the topological check does "
                "not imply the geometric one. Plus the R0 fix below.",
        tags=["retracted", "geometry", "instrument"])

    add("F005", "Our own run is evidence that a tissue accumulates stress at force balance over "
                "hundreds of frames.",
        "retracted", D,
        evidence="Cells compressed 3.63 -> 2.45 in volume against a rising target; force_mean "
                 "8.4 -> 22.9; residual force barely dropping with 4x the relaxation.",
        challenged_by="Follows from F004's cause being found.",
        killed_by="With R0 corrected, mean cell volume is flat for the whole run. The compression "
                  "was the radial spring, not the growth.",
        cause="A real phenomenon (tissues do store stress) was illustrated with a run that was "
              "not an example of it. The general claim was right and the evidence was wrong, "
              "which is the harder error to notice.",
        survives="The general claim, on laboratory evidence: laser-ablation recoil, tumour "
                 "spheroids under confinement, residual stress in arteries and plant stems. "
                 "Premise 5 now rests on those and not on our simulation.",
        changed="Premise 5 rewritten to forbid only inertia and unrelaxed transients, with the "
                "force-balance/zero-stress distinction spelled out; its example withdrawn.",
        tags=["retracted", "biology"])

    add("F006", "cell_grow never updates R0, so the radial spring in cell_mechanics "
                "holds the shell at its seed radius while cell target volumes grow sixteenfold.",
        "standing", D,
        evidence="Read from source: the radial term is mesh_ops.py:85, R0 is set at seeding "
                 ":217 and rescaled only by cell_grow :409. Fixing it (R0 from the enclosing "
                 "sphere of the current TARGET volume) removes the self-intersection completely: "
                 "rays cross exactly once at every frame, reduced volume 0.985 -> 0.977 instead "
                 "of -> 0.229, mean cell volume flat instead of collapsing.",
        survives="This is the root cause of F004 and F005.",
        changed="R0 now tracks the target volume every frame. Deliberately NOT the measured mean "
                "radius, which would make the spring chase the shell's own excursions and "
                "penalise a growing bud -- the one shape the campaign exists to produce.",
        tags=["standing", "mechanism", "defect"])

    add("F007", "cell_grow.conserve_amount extinguishes a Gray-Scott pattern.",
        "standing", D,
        evidence="Dilution of 1% per step kills the activator within 250 steps in pure chemistry, "
                 "while the undiluted pattern reaches 53% coverage by step 250 and holds "
                 "indefinitely. In the full model, mini_coral dies (act 0.428 -> 0.000) and "
                 "mini_coral_nodilute lives (0.470 -> 0.502 across 500 frames). An adversarial "
                 "reviewer showed the chem tensor is bit-identical to pure RD once it is off, so "
                 "it is the only channel growth has into the chemistry.",
        cause="Dilution is CORRECT physics -- growing a cell does dilute its contents. The defect "
              "is that Gray-Scott's activator is sustained by a quadratic term, so any steady "
              "multiplicative loss beats it. Correct physics plus a fragile mechanism is still a "
              "broken model.",
        changed="Premise 4's check; the coral runs with it off. The proper fix (dilute the "
                "activator column only) is not yet applied.",
        tags=["standing", "mechanism", "defect"])

    add("F008", "Every defect found in this campaign so far was found by a human looking at a "
                "picture, and none by the loop.",
        "standing", D,
        evidence="Ten defects on 31 July: a ball that shrank, a tissue that turned green all at "
                 "once, a coral that began uniformly red, a cross-section that changed character. "
                 "Each found by Cedric scanning an ordinary output. The loop's checks -- did the "
                 "run finish, is the mesh valid, are the numbers finite -- passed on all ten.",
        cause="The loop validates the SIMULATION and has never validated the SPECIMEN. Those are "
              "different questions and only the second catches a ball that quietly implodes.",
        changed="biologist.py: eleven premises about tissue, executable, gated into every run, "
                "each certified against a case whose answer was already known.",
        tags=["standing", "method"])

    add("F009", "Our `chi` cannot be compared with Okuda's, because it is a solver rate labelled "
                "as a spatial scale.",
        "standing", D,
        evidence="cell_chem_diffuse applies a DEGREE-NORMALISED graph Laplacian -- 'mean of my "
                 "neighbours minus me' -- which contains no dx at all, so d*chi is a dimensionless "
                 "per-frame mixing fraction. The operator nonetheless declares "
                 "PARAM_ROLES chi = 'spatial_scale'. Measured on a 2000-cell ball, chi does three "
                 "unrelated jobs: 1.3 gives one domain of 1067 cells, 4.0 kills scattered seeds, "
                 "13 saturates the integrator, 40 gives 109 single-cell specks.",
        changed="Decision taken with Cedric: calibrate chi against what Okuda REPORTS SEEING "
                "(~5 spots on a 2000-cell ball) to unblock now, AND measure the Turing wavelength "
                "in cell diameters on every run so results become comparable to the paper. The "
                "wavelength metric is not yet written.",
        tags=["standing", "instrument", "open-work"])

    print(f"seeded {len(load())} findings -> {LEDGER}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--standing", action="store_true")
    ap.add_argument("--lessons", action="store_true")
    a = ap.parse_args()
    if a.seed:
        seed(); raise SystemExit
    rows = load()
    if not rows:
        print("no findings recorded -- run `python findings.py --seed`"); raise SystemExit
    if a.standing:
        rows = [r for r in rows if r["status"] == "standing"]
    show(rows, lessons_only=a.lessons)
    n = {s: sum(1 for r in load() if r["status"] == s) for s in STATUS}
    print(f"\n  {n['standing']} standing, {n['retracted']} retracted, {n['open']} open, "
          f"{n['superseded']} superseded")
