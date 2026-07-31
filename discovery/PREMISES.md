# What we take as known about cell tissue

Ten basics. Not a literature review — the minimum a person needs to look at one of our runs and
say "that cannot be right". Each one is written as something you could **check against a run**,
because a premise nobody can check is an opinion.

The raw material is `_premises_raw.md`: 70 candidates with full sourcing, mined from the papers and
attacked on three lenses (is it true / can it be checked / does it contradict another one). Most
were cut for being too specific — true of mouse salivary gland, not true of tissue. What survives
here is the part that constrains *any* run we do.

Grades: **certain** (textbook, no serious dissent) · **usual** (true unless the tissue is doing
something special, and then you must say so).

---

## 1. Cells grow by taking material in — **certain**

A cell is mostly water. Its volume changes because it takes up or loses material, not because it
rearranges what it already has. So a tissue that gets bigger has *added* something.

*Constrains:* `morphogen_growth_3d.rho` — the baseline growth floor.
*Check:* total tissue volume at the end > total tissue volume when growth started.
*If violated:* a protrusion with `rho = 0` is made entirely of material taken from somewhere else
in the same body. That is a deliberate ablation, not a setting — say so, and say what the rest of
the body gave up.

## 2. A morphogen sets *where* and *how fast*, not *whether* — **usual**

Every cell in a proliferating epithelium grows, signal or no signal. A limb bud, a branch, a tube:
the tip adds material faster, while the body keeps adding it too.

*Constrains:* `rho` together with the `cell_react → morphogen_growth_3d.gate` connection.
*Check:* if the gate is connected at all, then `rho > 0`.
*If violated:* the tip-to-body growth ratio is infinite — a growing tip on a frozen body. No tissue
does that.

## 3. A cell divides because it got big — **certain**

Roughly a doubling of birth volume, then mitosis. Cycle times vary, but tightly — order 10%, not a
factor of two. Division itself adds nothing: two daughters sum to the mother.

*Constrains:* `divide_3d.factor`, `min_cycle`, `cycle_cv`, and `morphogen_growth_3d.vth_frac`.
*Check:* **the growth ceiling must sit above the division trigger.** Mean cell volume must be
roughly steady over the run, not drifting down.
*Caught:* this is defect D5b. `vth_frac` capped a cell's target at 1.5× while `factor` demanded
2.0× — the ceiling was *below* the trigger, so volume-triggered division was arithmetically
impossible and the only divisions we ever saw came from a timeout.

## 4. Growing a cell dilutes what is inside it — **certain**

Concentration is amount over volume. Grow the volume and the concentration falls, with no chemistry
involved. A reaction whose fixed point sits at zero is driven to extinction by growth alone.

*Constrains:* `morphogen_growth_3d.conserve_amount`.
*Check:* with the chemistry switched off and growth on, every cell's concentration must fall.
*If violated:* the tissue is manufacturing morphogen in proportion to its own growth, which feeds
the tip from nothing.

## 5. Mechanics is fast, biology is slow — **certain**

An epithelium relaxes elastic stress in seconds to minutes. It grows and divides over hours. So at
every instant the tissue is essentially at force balance, and there is no inertia.

*Constrains:* the relationship between the mechanics substep and the biology step — `dt`,
`relax_iters`, and the rates of every biological operator.
*Check:* one growth or division step must correspond to *many* mechanical relaxation steps, and
each biological operator must advance in **biological** time, not in solver substeps.
*Caught:* this is defect D5a. `dt = 0.02` is the mechanics substep, and the chemistry was being
integrated with it too — 300 frames bought 6 units of reaction time instead of ~500. Every "no
pattern formed" reading in the campaign was an artefact of the clock.

## 6. A resting vesicle rests — **certain**

A closed epithelial vesicle floating in medium is attached to nothing and has no preferred distance
from any point in space. Left alone it holds its shape. Surface tension is real and does pull
inward, but in a real tissue it is balanced — by cell volume, by lumen pressure, by cortical
pressure. Nothing in a tissue implodes when you stop poking it.

*Constrains:* `shape_energy_3d` — every inward term (`kappa_s`, `gamma`) needs a counterpart.
*Check:* run the mechanics **alone**, with no growth, no chemistry, no division. The radius must
hold.
*Caught:* this is defect D1, and it is the cheapest test in this document. The ball fell from 5.00
to 1.80 in twenty frames.

## 7. A confluent sheet does not absorb added area by stretching — **usual**

Add area to a confluent epithelium faster than it can make cells, and it does not simply get
thinner. It buckles, or it divides, or it extrudes a cell. Individual cells resist being stretched.

*Constrains:* the balance between the growth rate and the division rate.
*Check:* mean cell volume steady, cell shape index not drifting upward, and the distribution of
cell shapes not developing a long stretched tail.
*Explains:* the very thin cells in the tube wall that Cedric spotted on the strip — we were adding
area faster than the tissue could make cells, so it thinned instead of buckling or dividing.
*Predicts:* the buckling now seen in `p1_ko_divide_3d`, where division is knocked out and the
growing shell folds instead (folded cells 0 → 145, then healing back to 17).

## 8. The shape index decides whether tissue flows — **certain (floor) / usual (threshold)**

Perimeter over root-area. It cannot go below **3.545** — that is a circle, and it is geometry, not
biology. Around **3.81** a confluent sheet stops being able to rearrange and behaves like a solid;
above it, like a fluid. The exact threshold shifts on a curved closed surface.

*Constrains:* `p0`, and the interpretation of every `shape_idx` we record.
*Check:* any measured shape index below 3.545 is a **broken measurement**, not a finding.
*Use:* whether a run should be able to form a tube at all is partly this number.

## 9. A closed epithelium is a sphere with no holes — **certain**

Every interface shared by exactly two cells, every cell with strictly positive volume, no gaps, no
overlaps, no free edges. The vesicle is topologically a sphere, and stays one unless something
genuinely changes its topology.

*Constrains:* every mesh-validity check we run.
*Check:* Euler characteristic / genus constant. A cell of zero or negative volume is a solver
failure, not a small cell.
*Use:* this is what separates a discovery from corruption — a change in genus is either a real
topological event or a broken mesh, and we must say which.

## 10. Neighbour exchange is rare, and happens only when a junction has actually collapsed —
**usual**

Cells swap neighbours when the interface between them shrinks to nothing and the swap lowers the
energy. It is the rarest thing an epithelium does — not a per-frame housekeeping operation.

*Constrains:* `reconnect_t1_3d`.
*Check:* T1 events per cell per unit time should be small, and each should follow a junction that
really did collapse.
*If violated:* T1s fired as maintenance are a remesher, not a mechanism, and any "the tissue
flowed" conclusion is about the remesher.

---

## Why this document exists

Three defects were found today — a vesicle that collapsed under its own tension, chemistry running
50× too slow, and a growth ceiling below the division trigger. All three were found by Cedric
looking at a picture and saying it looked wrong.

Premises **6**, **5** and **3** above each state one of them in advance, as a check that costs
seconds to run. That is the point of the corpus: not to make the agents better read, but to give
them something they can *fail*.

The rule that follows: **every premise here is a claim about the specimen, not the simulation.**
The loop already checks that a run completed, that the mesh is valid, that the metrics are finite.
It has never once checked that the thing it simulated was a tissue.
