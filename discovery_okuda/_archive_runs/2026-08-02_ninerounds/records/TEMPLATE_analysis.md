<!-- RECORD ONLY. NEVER READ BY AN AGENT.

     This file is the human record and nothing in the loop consults it. That is deliberate and it
     is what lets it be complete: an append-only log that no prompt has to fit inside can hold
     every slot of every round forever, and it costs no context to do so. The knowledge lives in
     memory.md, which IS read, and which is therefore kept short.

     If you find yourself wanting an agent to read this file, the thing you want is in memory.md
     or it is not yet knowledge.

     THE SHAPE OF ONE ENTRY. Written by collector.py from the files on disk, never by an agent:
     the Proposer used to write this and that put the agent under evaluation in charge of its own
     record, which is how "parent 2 is fully PROPOSED" became coverage.

     APPEND ONLY. A record that can be revised is not a record.

     The form follows the connectome-gnn-cx exploration log, which ran this discipline over ~26
     batches. Four things it has that a prose summary does not, and each one is load-bearing:

       Node/parent   the ANCESTRY is in the log. Without it a reader cannot tell a line of
                     descent from a scatter of attempts, and neither can the Archivist.
       Mutation      the DIFF from the parent, written out. "we changed the model" is not a
                     mutation; `cell_react: gierer_meinhardt -> gray_scott` is.
       Hypothesis    quoted VERBATIM as it was posed, before the evidence. A hypothesis
                     paraphrased afterwards is a rationalisation.
       Verdict       supported | falsified | partial | inconclusive, WITH THE NUMBER that
                     decided it, and naming which prediction. -->

## Round N — slot S: <supported | falsified | partial | inconclusive>

Node: id=<comp hash>, parent=<parent comp hash, or `none` for a control>
Track: <A = understand the operator landscape | B = reproduce an Okuda morphology>
Hypothesis tested: "<the prediction exactly as posed, before the run>"
Config: <run name, n_cells, dt, chi, d_a, d_h, and any parameter this slot moved>
Measured: protr_peak=<>, protr_final=<>, ta_n_tubes_final=<>, pattern=<spatial spread>, cells=<>->
Specimen: <valid | ambiguous | invalid | valid (declared)> — <the premises broken, or "all hold">
Reader: phenotype=<>, forced_or_grown=<>, specimen=<sound|compromised>
Eye-check: <what the movie showed; "agrees" or the disagreement. Never a score.>
Mutation: <op/param: before -> after. `none (control)` when nothing moved.>
Verdict: <supported | falsified | partial | inconclusive> — <=30 words, CITING THE NUMBER and
         saying which prediction it settled. `inconclusive` when the prediction could not be
         checked: it leaves the surprise denominator and buys nothing.
Next: parent=<comp hash the next round should breed from, and why in <=15 words>

<!-- ROUND SUMMARY, one per round after its slots -->

## Round N — summary

Posed: <n>   Evidence: <n>   Refused: <n> (<reason codes>)
Surprise: <n wrong>/<n checkable> — a round that only confirms has bought coverage and no knowledge
Tracks: <n> Track A, <n> Track B
Specimens: <valid n, ambiguous n, invalid n>
Frontier after: <comp hashes carried forward>
Diagnosis: <the Diagnostician's cause + guard, if it ran. Otherwise "not called".>
