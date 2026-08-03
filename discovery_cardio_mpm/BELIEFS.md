# BELIEFS — what this campaign has actually earned

**Zero entries. That is correct and it is the point.**

The previous campaign reached sixty rounds and a confident ledger. Emptying it took three
measurements, none of them about biology: the random seed was never set, so identical runs
differed by most of the signal; neither score was ever read against its own null, so nobody could
say whether anything had been achieved; and nothing was ever held out, so *"it fits"* and *"it is
right"* were never separated.

So this register starts empty, and a claim enters it only by clearing every one of these:

| requirement | why |
|---|---|
| measured with a **certified** instrument | an uncertified number is not evidence, however sophisticated the code |
| effect larger than the **measured noise** | the noise floor is measured in Phase 2, not assumed |
| scored on a split the model was **not fitted to** | with gradients, fitting better is nearly free |
| beats its **capacity-matched control** | adding free parameters improves any fit and means nothing |
| the parameter is **identifiable** from this recording | an unidentifiable mechanism looks exactly like an absent one |
| the fit **converged** | an unfinished optimisation is a fact about the optimiser |
| a **run directory that still re-derives it** | a number nobody can reproduce is a rumour |

Entries are one line each, and the file has a **hard length cap of 120 lines, enforced by
`certify_apparatus.check_registers`** — the previous campaign's ledger reached 1,668 lines and its
log 7,289. Overflow forces a merge or a retraction, never an append. Every row must carry all seven
evidence columns; a row that does not is refused, and the gate has been watched refusing one.

A claim that is later contradicted moves to `retractions.jsonl` (present, empty) with its cause of
death attached.
It is never edited and never deleted — a register that keeps only its winners teaches nothing
about how it goes wrong.

---

| id | claim | metric | split | effect (σ) | identifiability | seeds | commit | run |
|---|---|---|---|---|---|---|---|---|

*(empty)*
