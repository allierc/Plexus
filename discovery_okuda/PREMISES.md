# What we take as known about cell tissue

Eleven basics. Not a literature review — the minimum a person needs to look at one of our runs and
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

*Constrains:* `cell_grow.rho` — the baseline growth floor.
*Check:* total tissue volume at the end > total tissue volume when growth started.
*If violated:* a protrusion with `rho = 0` is made entirely of material taken from somewhere else
in the same body. That is a deliberate ablation, not a setting — say so, and say what the rest of
the body gave up.

## 2. A morphogen sets *where* and *how fast*, not *whether* — **usual**

Every cell in a proliferating epithelium grows, signal or no signal. A limb bud, a branch, a tube:
the tip adds material faster, while the body keeps adding it too.

*Constrains:* `rho` together with the `cell_chem_react → cell_grow.gate` connection.
*Check:* if the gate is connected at all, then `rho > 0`.
*If violated:* the tip-to-body growth ratio is infinite — a growing tip on a frozen body. No tissue
does that.

## 3. A cell divides because it got big — **certain**

Roughly a doubling of birth volume, then mitosis. Cycle times vary, but tightly — order 10%, not a
factor of two. Division itself adds nothing: two daughters sum to the mother.

*Constrains:* `cell_divide.factor`, `min_cycle`, `cycle_cv`, and `cell_grow.vth_frac`.
*Check:* **the growth ceiling must sit above the division trigger.** Mean cell volume must be
roughly steady over the run, not drifting down.
*Caught:* this is defect D5b. `vth_frac` capped a cell's target at 1.5× while `factor` demanded
2.0× — the ceiling was *below* the trigger, so volume-triggered division was arithmetically
impossible and the only divisions we ever saw came from a timeout.

## 4. Growing a cell dilutes what is inside it — **certain**

Concentration is amount over volume. Grow the volume and the concentration falls, with no chemistry
involved. A reaction whose fixed point sits at zero is driven to extinction by growth alone.

*Constrains:* `cell_grow.conserve_amount`.
*Check:* with the chemistry switched off and growth on, every cell's concentration must fall.
*If violated:* the tissue is manufacturing morphogen in proportion to its own growth, which feeds
the tip from nothing.

## 5. Mechanics is fast, biology is slow — **certain**, but read it narrowly

An epithelium relaxes elastic stress in seconds to minutes. It grows and divides over hours. So on
the timescale of morphogenesis the tissue has no inertia, and the configuration you see at any
instant is a mechanical equilibrium of the current cell targets.

**This does NOT mean the tissue is unstressed.** Cedric pushed on the first draft of this premise,
which said "at every instant the tissue is essentially at force balance" and read as if stress
could not build up. Force balance and zero stress are different things — a stretched drumhead is
at perfect force balance and stores large tension indefinitely — and tissues really do accumulate
stress over long times:

  * cut an epithelial junction with a laser and it recoils; that tension had been sitting there,
    at equilibrium, for hours
  * tumour spheroids under confinement build compressive stress over days
  * arteries and plant stems carry residual stress that only appears when you cut them open
  * differential growth in a constrained geometry leaves residual stress for the whole of
    development

*(An earlier version of this section cited one of our own runs as an example of stored stress —
cells compressed from 3.63 to 2.45 in volume before the shell released. **That was an artefact and
the example is withdrawn.** The compression came from a radial spring whose target radius was
frozen at the seed value while the cells grew sixteenfold; with the spring corrected, mean cell
volume is flat for the whole run and there is no compression at all. The premise stands on the
laboratory evidence above, not on our simulation.)*

So the premise forbids exactly two things: **inertia**, and **unrelaxed transients** — the
configuration lagging behind the forces because the solver ran out of iterations.

*Constrains:* `dt`, `relax_iters`, and the rate of every biological operator.
*Check, part A (the clock):* each biological operator must advance in **biological** time, not in
solver substeps. One growth or division step must correspond to many mechanical relaxation steps.
*Check, part B (convergence):* the relaxation must actually reach equilibrium each frame. The
residual force after relaxation must not grow as the tissue does — `relax_iters` is a constant
while the system it has to relax keeps getting bigger, so this is a real risk and not a
hypothetical one.
*Caught by part A:* defect D5a. `dt = 0.02` is the mechanics substep and the chemistry was being
integrated with it too — 300 frames bought 6 units of reaction time instead of ~500. Every "no
pattern formed" reading in the campaign was an artefact of the clock.

## 6. A resting vesicle rests — **certain**

A closed epithelial vesicle floating in medium is attached to nothing and has no preferred distance
from any point in space. Left alone it holds its shape. Surface tension is real and does pull
inward, but in a real tissue it is balanced — by cell volume, by lumen pressure, by cortical
pressure. Nothing in a tissue implodes when you stop poking it.

*Constrains:* `cell_mechanics` — every inward term (`kappa_s`, `gamma`) needs a counterpart.
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

*Constrains:* `edge_flip`.
*Check:* T1 events per cell per unit time should be small, and each should follow a junction that
really did collapse.
*If violated:* T1s fired as maintenance are a remesher, not a mechanism, and any "the tissue
flowed" conclusion is about the remesher.

## 11. A tissue cannot pass through itself — **certain**

Two parts of the same epithelium cannot occupy the same space. It is a physical body. When a bud
grows back into the ball, something has to give — the tissue deforms, or it stops.

*Constrains:* every geometric reading we take, because a self-overlapping surface makes all of them
meaningless.
*Check:* cast rays from the tissue centroid. A simple closed shell gives **exactly one** crossing
per ray.
*Why it is separate from #9:* **the Euler characteristic cannot see this.** Genus is
*combinatorial* — it reads the connectivity, never the coordinates — so a shell crumpled seventeen
layers through itself still reports genus 0, "sphere (as built)". Measured on
`mini_grow_divide_bigger`: genus 0 at *every* frame, while single-ray crossings go from 100% at
frame 384 to **0%, median 13**, at frame 423.
*Caught:* the buckling transition was reported as physical on the strength of the genus check
alone. The transition is not a solver artefact — it survives quadrupling the relaxation — but the
state it produces is not a tissue. Likely cause: the radial spring's target radius is frozen at the
seed value while cell target volumes grow sixteenfold, so the shell is held at radius 5 while its
cells demand far more area than that sphere has. It has nowhere to go but through itself.

## 12. A concentration is non-negative and finite — **certain**

Amount over volume. Neither term can be negative and neither can be infinite, so a concentration
that goes negative or non-finite is arithmetic failing, not chemistry happening.

*Constrains:* `cell_chem_diffuse`, `cell_chem_react` — the integrator, not the model.
*Check:* every recorded concentration is finite and >= 0.
*If violated:* every downstream number is about the integrator. This is the check that caught the
reaction running 50x too fast: the activator went 0.01 -> 1.41e6 -> NaN while spatially uniform.

## 13. A tissue that stops growing because the ARRAY filled is not evidence about growth — **certain**

The mesh lives in a fixed vertex buffer, and when it fills, division stops. The tissue did not
decide anything; it was refused. The distinction between *"the tissue stopped dividing"* and
*"the tissue was not allowed to divide"* is invisible in every number we record, and it is the
difference between a result and an artefact.

*Constrains:* the vertex reservoir against the growth a composition asks for. For a trivalent
closed mesh `V = 2F - 4`, so a buffer of `V` vertices caps the tissue at `(V+4)/2` cells.
*Check:* the cell count must not reach a constant and hold it for a large tail of the run. The
engine also reports it directly — `div_blocked` counts divisions refused for want of buffer, and
`buf_full` says the array is at its ceiling.
*Caught:* `wk_pressure_pos_s0` grew 150 -> 1778 cells by frame 323 of 900 and then added **zero**
for the remaining 575 frames. 1778 is exactly the cap of a buffer sized for a 150-cell start. Two
thirds of the run measured a full array, every other premise passed it, and the only thing that
noticed was a human watching cell division stop two seconds into a six-second movie.
*Grade of the verdict:* **ambiguous**, not invalid — the phase before the cap is real evidence.
What is inadmissible is everything after it, and any endpoint metric that reads the plateau as a
biological steady state.

---

## Why this document exists

Ten defects were found in a single day — a vesicle that collapsed under its own tension, chemistry
running 50× too slow, a growth ceiling below the division trigger, a pattern re-stamped every frame,
gauges that could not see what they measured. All ten were found by Cedric looking at a picture and
saying it looked wrong.

Premises **6**, **5** and **3** above each state one of them in advance, as a check that costs
seconds to run. Premise **11** was added afterwards, from a mistake of exactly the kind this
document exists to prevent: a result was called physical because the topology check passed, when
the check simply could not see the failure. That is the point of the corpus: not to make the agents better read, but to give
them something they can *fail*.

The rule that follows: **every premise here is a claim about the specimen, not the simulation.**
The loop already checks that a run completed, that the mesh is valid, that the metrics are finite.
It has never once checked that the thing it simulated was a tissue.
