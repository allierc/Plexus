# Cell size and cell cycle: the alignment plan

Branch `size-cycle-align`, opened 2026-09-11 from the two audits of the same day. Reference for
what "size control" has to mean: Ginzberg, Kafri & Kirschner, *On being the right (cell) size*,
Science 348:1245075 (2015), with Cadart et al. Nat. Commun. 9:3275 (2018) for the mammalian
near-adder and the two-channel finding, Schmoller et al. Nature 526:268 (2015) and Zatulovskiy et
al. Science 369:466 (2020) for inhibitor dilution, Smith & Martin PNAS 70:1263 (1973) for the
constant hazard. The test those papers give is one regression per run: volume added over a cycle
against volume at birth, both in units of the seed-time median cell volume `v_ref`. Slope -1 is a
sizer, 0 an adder, +1 a timer under exponential growth -- and the population median must be
stationary, or the rule is not controlling anything.

## 0. What the audits found

Measured on the archives that this branch withdrew (`tools/size_report.py` reproduces the numbers
on any replacement run):

1. **`cv_*` measured target tracking, not size.** No growth, no division: `CV(V) = CV(V0f) + lag`,
   the lag set by `kappa_s / (k_v h0)`. Two of ten arms were rails -- `p0` is not read by the
   apico-basal energy (`cv_shape_low` equal to the baseline to four decimals), and `k_v >= 8` at
   `eta 0.08` sat past the explicit-Euler bound (CV 0.10 in the frame-0 record, thickness spread
   0.05 -> 0.16 over the run: the solver, not the model).
2. **The adder arms behaved as sizers.** Slope -1.05, CV of division volume 0.04, on
   `cvd2_adder_tension`. `Vbirth` at a division is the arithmetic half `vf * p` with `p = 0.5`
   ([vertex_ops.py:1979](../../src/plexus/operators/vertex_ops.py#L1979)), so in a converged
   population every stored birth volume is `v_ref` and `Vbirth + delta v_ref` is `2 v_ref`, the
   sizer's threshold. The realised daughter volumes did vary (CV 0.27 at the septum, 0.16 twelve
   frames later); the rule never read them. The doubler collapsed the same way.
3. **The four cycle arms were four timers.** Growth `rate 0.003457` on the linear scale is a volume
   rate of `3 * rate = 0.0104` per frame: x3.9 over the 130-frame S+G2+M clock, x12 over a
   240-frame cycle. Division removes x2, so 85-96 % of daughters were born above the G1 checkpoint
   after frame 200, the G1 fraction was 0-5 %, and the median volume ran 1.3 -> 6.0 `v_ref` in
   400 frames. Condition for a checkpoint to act at all: `exp(3 rate (t_s + t_g2 + t_m)) < 2`.
4. **The cycle sizer is an adder for small cells.**
   [vertex_ops.py:3304](../../src/plexus/operators/vertex_ops.py#L3304) divides added volume by
   `(g1_size - 1) v_ref`, not by `v* - V_b`: a cell born below `v_ref` leaves G1 after adding
   `0.6 v_ref`, below the checkpoint. Chosen because `Vbirth` and `v_now` were in two volume
   conventions; `cell_size()` has since unified them.
5. **Inhibitor dilution carries no size information.** Concentration reset to 1.0 at every birth
   ([vertex_ops.py:3421](../../src/plexus/operators/vertex_ops.py#L3421)), exit at `c <= 0.6`, so
   G1 ends at `V = V_b / 0.6` for every cell -- a relative rule. The mechanism works because a
   size-independent *amount* gives `c_birth = A / V_b`, higher in small cells.
6. **Three size readers, two conventions, one switch that reaches two of them.** `cell_divide`
   reads the polyhedron whenever `sep` exists ([:1786](../../src/plexus/operators/vertex_ops.py#L1786));
   `cell_cycle` and `cell_die` only under `v0_from: polyhedron`
   ([:182](../../src/plexus/operators/vertex_ops.py#L182)); `cell_grow[sizer]` reads the target
   `V0f`. And the growth-rate channel (`cell_grow` sizer / balance) was used by no tissue spec.

7. **The apico-basal seed is not at rest, and the target is not the volume.** Found while doing
   R1. `cell_mechanics[apicobasal]` holds a cell at `V_eq = mono_k * V0f + mono_delta`: `mono_k`
   converts a wedge-unit target into the polyhedron volume (0.53 on the reference spheroid) and
   `mono_delta` is a solved additive offset that puts the *mid-surface radius* at rest. Under
   `v0_from: polyhedron` the conversion was applied to a target already in polyhedron units, and
   the offset then inflated every cell x1.7 over the opening frames while the size readers cached
   their reference from the un-inflated seed. Even with `mono_k = 1` the seeded shell thickens
   (h 0.88 -> 1.38, or 1.38 -> 1.92: `tools/equilibrium_h.py`), so the polyhedron volume ramps
   x1.4 in ~40 frames whatever `h0` says. Consequences for size control: an actual volume written
   into a target inflates the cell by the offset (tissue volume x2.25 in 60 frames when tried),
   and a first generation whose `Vbirth` is the seed's target is born 40 % "small".
   Handled in R1 by a declared settle window (`seed_mesh ref_frame`) and by keeping targets
   affine (split the mother's target in proportion to the measured pieces; `Vbirth` measured);
   the rest offset itself is R5's business (a seed at rest in every degree of freedom).

8. **`v0_from: polyhedron` crumples the shell, and every R0-R1c archive was crumpled.** Seen in
   the movies, not the tables: a jagged mass of inverted prisms where the mechanics-only control
   is a clean sphere. `tools/spheroid_gauge.py` (asphericity, inverted wedges, inward thickness
   vectors, thickness CV and min/median) puts numbers on it -- `size_sizer` R0: asphericity 0.18,
   24 % inverted wedges, thickness CV 0.37, thinnest cell 0.8 % of the median; `divide_growing_ball`:
   0.02 / 0 / 0.09 / 0.71 over 801 frames. Ablated on 200-frame cuts: the one key that matters is
   `v0_from: polyhedron` (with it, asphericity 0.12 and thickness 0.05 by frame 20 at either growth
   rate; without it a clean ball at either rate). The polyhedron target convention is not a
   working mechanics, whatever the readers do with it. Every spec is rebuilt from
   `divide_growing_ball`'s conventions and the gauge is a gate on every rung from here on.
9. **Thickness spread grows with division waves, and with the growth rate.** On current code
   `divide_growing_ball` at its own rate (0.000578) holds thickness CV 0.10 over 400 frames; at
   0.00096 it passes 0.15 at frame 320 and reaches 0.39 by 400 as divisions come in waves. The
   rung runs at the working point's rate -- one doubling per 400 frames -- with the cycle clock
   scaled to it (183/133/67/17) and 1601 frames per run.

10. **R1's proportional target split is what crumples the shell after division.** Bisected with
    `tools/spheroid_gauge.py` on `divide_growing_ball`, 801 frames, deterministic
    (`PLEXUS_STRICT_DETERMINISM=1`): the archive's own commit (0dc401f1), Sep 8, Sep 9, Sep 10,
    the warp-gradient commit (51b8d9f1) and the branch base (7659a887) are all spheroids with
    the same numbers (thickness CV 0.093, thinnest/median 0.67); HEAD after R1 crumples from
    frame 500 (CV 0.67). Giving each daughter a target in proportion to its septum piece hands
    a small piece a small target with the same footprint, its thickness collapses, and the
    spread compounds with every division. Fixed: the targets are the mother's halves again;
    `Vbirth` alone is measured (the piece at the cut, re-read at the next call). R2's
    "restart" findings 8-9 stand, but the rate was not the driver of the creep: this was.
11. **The non-deterministic warp path crumples the same commit in one run and not another.**
    7983f784 and 7659a887 crumpled from frame 400-600 without `PLEXUS_STRICT_DETERMINISM`
    (through `regression_lib.run_cut`) and were clean with it; 51b8d9f1 without it was clean
    once. The apico-basal mechanics sits close enough to an instability that atomics order
    decides. The rung runs deterministic (the job scripts set the flag); the fixed-order
    `wp.atomic_add` path is a finding for R5, not this ladder.
12. **A G1 exit parked exactly on the boundary froze the cycle.** Float32 rounding put `p` back
    under `cut[G1]` next frame; all four R2 cycle arms sat at 304 cells with every cell "in S".
    Fixed (f364c6ea): the exit lands `1e-5` past the boundary.

13. **The rules must read the polyhedron, and it was R2's wedge trigger that crumpled the shell,
    not R1's split.** Bisected commit by commit (deterministic, 801 frames): R1 and R1c are
    spheroids (thickness CV 0.12); R2 -- `cell_divide` routed through `cell_size`, which read
    the WEDGE for a spec without `v0_from` -- crumples from frame 600; restoring the even target
    split alone did not help. With every reader on the polyhedron whenever a separation exists,
    `divide_growing_ball` is a spheroid again (CV 0.090, the archive's number) and so is
    `size_sizer` over 1001 frames (0.102). Finding 10 is corrected: the proportional split was
    withdrawn anyway (targets are the mother's halves; `Vbirth` alone is measured), but the
    driver was the trigger's volume. Physically: a cell whose thickness runs away doubles its
    polyhedron and divides; the wedge cannot see thickness, so thick cells persist and the
    spread compounds.

## 1. What this branch has already done

- Withdrawn: 36 specs (`cv_*`, `cvd_*`, `cvd2_*`, `cyc4_*`, `cycle_phases`), their 35 archives,
  the fingerprints of the five that were working points, `tools/cv_report.py`,
  `tools/cycle_report.py`. The regression registry, `tests/README.md`, `REGRESSION_PLAN.md` and
  the ECM roadmap point at the replacements.
- Written: `tools/size_report.py`, the one report (slope, `r(cycle length, V_b)`, CV of division
  volume, median drift over the last two cycles, CV of the population).
- Written: eleven specs, one growth rate for all of them -- `rate 0.00096`, one volume doubling
  per 240 frames, so a rule that is a sizer has exactly x2 to remove -- with `vseed_cv 0.15` and
  `split_cv 0.12` so there is birth-size variance for the rule to act on, and
  `v0_from: polyhedron` so every reader is in the cell's own volume:

  | spec | rule under test | expected after R1 |
  |---|---|---|
  | `size_sizer` | `cell_divide` sizer, factor 2 | slope -1, CV(V_d) = cycle_cv 0.15 |
  | `size_adder` | `cell_divide` adder, delta 1 | slope 0 |
  | `size_doubler` | `cell_divide` doubler (the null) | slope ~ +1, CV grows |
  | `size_timer` | `cell_divide` timer, 60 calls x every 4 | slope ~ +1, CV grows |
  | `size_grow_sizer` | `cell_grow` sizer under a timer division | slope < 0 with no checkpoint: the growth channel alone |
  | `size_two_channel` | `cell_grow` sizer + `cell_cycle` sizer | tightest CV: Kafri 2013 / Cadart 2018 |
  | `cycle_sizer` | `cell_cycle` G1 size checkpoint | slope -1, G1 fraction ~ 45 % |
  | `cycle_timer` | `cell_cycle` all-clock | slope ~ +1 |
  | `cycle_hazard` | Smith-Martin constant hazard | slope ~ +1, exponential tail in L |
  | `cycle_dilution` | Whi5 / Rb dilution | slope -1 once c_birth = v_ref / V_b |
  | `mech_target_percell` | twin of `mech_uniform_target`, per-cell targets | CV(V) = CV(V0f) + 0.006 |

## 2. The plan, v2 (2026-09-12): layers, each gated before the next is read

What R0-R3 taught: the size and cycle rules are READERS on top of a mechanics-and-topology layer
that was never gauged, and every table produced before `tools/spheroid_gauge.py` existed was a
table about a bent mesh. So the plan is re-ordered into layers. A layer has its own archives
under `log/size_cycle/<rung>`, its own gate, and nothing above it is read until it is green.

**Layer 0 -- mechanics and topology.** The apico-basal shell under growth, T1 and division must
stay a spheroid of prisms for four doublings: `spheroid_gauge` SPHEROID on the shell bands
(asphericity, inversions, inward vectors, thickness spread) AND on the prism bands (trapezoid,
shear, tilt, in-cell thickness). Instruments: the gauge; `tools/equilibrium_h.py` (settling time,
38 frames at 30 steps, 20 at 90); per-flip and per-division prism damage measured directly.
Levers, in the order the evidence ranks them: the radial pin (finding 17, off), the T1 flip on
prisms (finding 16; what a flip does to the four cells' caps), the septum (what the cut does to
two cells' caps and how `local_relax` should heal them without the mid-surface energy -- the
shipped `local_relax` uses the wrong energy on an apico-basal mesh, `sweep_r578_i30_heal20`), the
seed at rest (finding 7), and the energy itself: it has no term that prefers a prism over a
frustum, so every kick random-walks the caps apart (finding 15). A prism term -- apical and basal
tensions as separate parameters, which is what the 3D vertex models of Okuda and Misra carry --
is the candidate, gated by the prism bands on `mech_target_percell` under growth alone.

**Layer 1 -- readers.** One volume convention (the polyhedron whenever a separation exists,
finding 13), one reference (`ref_frame`), `Vbirth` measured. Done in R2b; kept under the gate.

**Layer 2 -- rate laws.** `cell_grow`, `cell_cycle`, `cell_divide`, `cell_die` as rate laws,
each with its own test: Kirschner slopes and stationarity for size (`size_report`), phase
fractions and cycle-length distributions for the cycle, death onset and the shrink-shed-extrude
sequence for apoptosis. The "one when" consolidation (size rules live in `cell_cycle`;
`cell_divide` keeps the septum) and the dilution rule's defect (scores like a timer, R2b) are
here.

**Layer 3 -- rigs.** Interaction experiments on the accepted working point, each a spec family
plus one report. The first is the apoptosis rig: `cell_die` models (`smaller`, `crowded`,
`stalled`, `competition`, `prescribed`) x size rules (sizer / adder / timer) x cycle models x
topology (T1 on / off; shell / sheet), with `tools/death_report.py` joining every death to the
cell's volume, birth volume, phase, age and neighbour count at death, and the gauge run
alongside so a death is never read off a bent mesh. Later rigs: morphogen x size, ECM x size.

**Representation and engine (cross-cutting, R5 as before).** Per-cell state on the cell set,
`cell_id`, time units, default recording of `Vbirth`/`phase`/`cycle_progress`; seed-time writes
in seed operators; the fixed-order atomics path (finding 11); `p0` off the apico-basal contract.

## 3. Log

**R0 (2026-09-11, code at 7659a887, archives `log/size_cycle/R0`).** Every divide rule scored as
a sizer, which is the audit's finding 2 reproduced on fresh specs: the old target split put
`V0f_m / 2` on both daughters, the mechanics pulled the realised pieces to it within a few
frames, and no rule ever saw birth-size variance.

| spec | cells | cycles | slope | r(L,V_b) | CV(V_d) | L | med V/v_ref | drift | CV(V) |
|---|---|---|---|---|---|---|---|---|---|
| size_sizer | 200->2480 | 2080 | -0.70 | -0.38 | 0.15 | 164 | 0.98 | 0.02 | 0.18 |
| size_adder | 200->2272 | 1872 | -0.72 | -0.33 | 0.13 | 197 | 1.00 | 0.01 | 0.18 |
| size_doubler | 200->12756 | 12357 | -0.29 | 0.04 | 0.15 | 34 | 0.78 | 0.00 | 0.27 |
| size_timer | 200->1540 | 1140 | -0.67 | -0.03 | 0.19 | 237 | 1.13 | -0.01 | 0.19 |
| size_grow_sizer | 200->1527 | 1127 | -0.77 | 0.03 | 0.16 | 238 | 1.08 | -0.03 | 0.16 |

| size_two_channel | 200->12510 | 12110 | -0.67 | -0.18 | 0.16 | 95 | 1.07 | 0.21 | 0.20 |
| cycle_sizer | 200->12758 | 12358 | -0.52 | -0.13 | 0.15 | 100 | 0.81 | 0.03 | 0.16 |
| cycle_timer | 200->2434 | 2034 | -0.65 | -0.01 | 0.18 | 224 | 1.00 | -0.04 | 0.19 |
| cycle_hazard | 200->2941 | 2541 | -0.41 | 0.05 | 0.22 | 194 | 0.94 | -0.05 | 0.23 |
| cycle_dilution | 200->2795 | 2395 | -0.20 | 0.11 | 0.26 | 188 | 0.93 | -0.01 | 0.28 |
| mech_target_percell | 200->200 | - | - | - | - | - | 0.99 | - | 0.018 |

`size_doubler`'s 34-frame cycle is the relative rule plus `split_cv` feeding it ever-smaller
nominal birth volumes; `cycle_sizer` and `size_two_channel` ran to the 12,800-cell buffer because
the un-capped G1 rate crossed S and G2 in one step (fixed in R1).

**R1 (2026-09-11, feeb37cf, archives `log/size_cycle/R1`).** Fixes 2, 4, 5 as planned, plus
three found on the way: a G1 exit is capped at the G1/S boundary (an unbounded G1 rate crossed S
and G2 in one step), `mono_k = 1` under the polyhedron convention, and the settle window
(finding 7). The dilution rule is also normalised by the cell's own `c_b - thresh`, the same
defect as the sizer's denominator. Local 300-400-frame cuts before launch: `size_adder` newborns
at V/V_b = 1.00, `cycle_dilution` cycle 159 frames with a 30 % G1, `cycle_sizer` r(L,V_b) -0.5.
Regression: `sheet_divide`, `mech_uniform_target`, `apop2_ks0p1` unchanged; `divide_growing_ball`
235 vs 226 cells at frame 150 (the measured-birth rule), refreshed in ed3154c5.

| spec | cells | cycles | slope | r(L,V_b) | CV(V_d) | L | med V/v_ref | drift | CV(V) |
|---|---|---|---|---|---|---|---|---|---|
| size_sizer | 200->1481 | 1081 | -0.77 | -0.74 | 0.18 | 197 | 0.98 | -0.02 | 0.45 |
| size_adder | 200->1289 | 890 | -0.04 | -0.52 | 0.25 | 229 | 1.06 | -0.13 | 0.52 |
| size_doubler | 200->10133 | 9733 | 0.83 | 0.79 | 4.78 | 29 | 0.00 | -0.88 | 3.99 |
| size_timer | 200->1278 | 878 | 0.66 | -0.02 | 0.48 | 236 | 0.99 | -0.21 | 0.74 |
| size_grow_sizer | 200->1289 | 889 | -0.11 | 0.06 | 0.27 | 236 | 0.88 | -0.17 | 0.42 |
| size_two_channel | 200->1444 | 1044 | -0.68 | -0.62 | 0.21 | 250 | 1.04 | 0.06 | 0.38 |
| cycle_sizer | 200->1573 | 1173 | -0.25 | -0.52 | 0.31 | 224 | 0.90 | -0.06 | 0.56 |
| cycle_timer | 200->1992 | 1592 | 0.63 | -0.01 | 0.55 | 224 | 0.65 | -0.33 | 0.75 |
| cycle_hazard | 200->2315 | 1915 | 0.60 | 0.10 | 0.70 | 196 | 0.49 | -0.40 | 0.97 |
| cycle_dilution | 200->1960 | 1560 | 0.08 | -0.39 | 0.39 | 212 | 0.76 | -0.09 | 0.58 |
| mech_target_percell | 200->200 | - | - | - | - | - | 1.00 | - | 0.091 |

The rules separate the way the review says: sizer -0.77, adder -0.04, timer +0.66, hazard +0.60,
doubler +0.83 and a collapse to zero volume (asymmetric division amplified by a relative rule,
the review's own argument). Not accepted as R1, for one reason: `mech_target_percell` went from
lag CV 0.007 to 0.095, and every CV(V) roughly doubled. Isolated on 200-frame mechanics-only
cuts: `h0 1.38` alone does it (0.088 -> 0.007 at either `mono_k`); at that thickness the tension
term sets the volume and `k_v` no longer holds per-cell targets. The "equilibrium" thickness of
`tools/equilibrium_h.py` is where the rest offset inflates the shell to, not a regime to seed in.

**R1b (687cd2ce, archives `log/size_cycle/R1b`).** R1 with `h0 0.88`; the settle window covers
the 0.88 -> 1.16 ramp.

| spec | cells | cycles | slope | r(L,V_b) | CV(V_d) | L | med V/v_ref | drift | CV(V) |
|---|---|---|---|---|---|---|---|---|---|
| size_sizer | 200->2165 | 1765 | -0.61 | -0.58 | 0.17 | 186 | 0.83 | -0.01 | 0.28 |
| size_adder | 200->1294 | 894 | -0.04 | -0.45 | 0.26 | 221 | 1.04 | -0.07 | 0.41 |
| size_doubler | 200->797 | 398 | 0.69 | -0.09 | 0.42 | 280 | 1.49 | 0.07 | 0.57 |
| size_timer | 200->1269 | 869 | 0.60 | -0.00 | 0.51 | 236 | 0.95 | -0.20 | 0.63 |
| size_grow_sizer | 200->1277 | 877 | -0.16 | -0.02 | 0.27 | 236 | 0.95 | -0.11 | 0.35 |
| size_two_channel | 200->4068 | 3668 | -0.49 | -0.55 | 0.19 | 167 | 0.83 | -0.01 | 0.25 |
| cycle_sizer | 200->2578 | 2178 | -0.15 | -0.48 | 0.26 | 176 | 0.74 | 0.01 | 0.27 |
| cycle_timer | 200->1980 | 1580 | 0.53 | 0.09 | 0.48 | 224 | 0.73 | -0.28 | 0.54 |
| cycle_hazard | 200->2353 | 1953 | 0.49 | 0.08 | 0.53 | 197 | 0.66 | -0.26 | 0.58 |
| cycle_dilution | 200->3553 | 3153 | 0.07 | -0.33 | 0.34 | 163 | 0.64 | -0.05 | 0.29 |
| mech_target_percell | 200->200 | - | - | - | - | - | 0.99 | - | 0.018 |

The mechanics control is back (0.018), the checkpoint arms are stationary (drift <= 7 %), the
two-channel arm has the tightest dividing-population CV (0.25). Not accepted: the sizers divide
at 1.05 x the settled median and are born at 0.75, i.e. at 2 x the SEED's median. The seed had
pre-filled `v_ref_poly` with the pre-ramp median (1.35 against 2.59 settled), so the settle window
never cached a reference and no seeded `Vbirth` was ever reset.

**R1c (46efe89f, archives `log/size_cycle/R1c`).** The seed writes no `v_ref_poly` under a settle
window. Not scored: the movies showed every dividing arm of R0-R1c crumpled (finding 8), so
R0-R1c are archives of a broken mesh and their tables are withdrawn from evidence. What survives
of R1: the arithmetic fixes (measured `Vbirth`, the sizer's denominator, the inhibitor at birth,
the G1 cap, the settle window), all convention-independent.

**R2 (archives `log/size_cycle/R2`).** Restart on `divide_growing_ball`'s conventions: no
`v0_from`, `h0 0.88`, rate 0.000578, `ref_frame 60`, 1601 frames; one reader (`cell_size` in
`cell_divide` and `cell_grow[sizer]`, the private polyhedron branch gone); `tools/spheroid_gauge.py`
must say SPHEROID on every arm before its `size_report` row counts. Landed and gauged: every
dividing arm CRUMPLED (thickness CV past 0.15 between frames 350 and 1200; `size_sizer` 12 %
inverted cells by 1601), the four cycle arms frozen at 304 cells (finding 12). Findings 10-12
came out of it; not scored.

**R2b (b3ef2ca6, archives `log/size_cycle/R2b`, run locally on two A6000s, deterministic).** R2
with every size reader on the polyhedron when a separation exists (finding 13), the even target
split (finding 10) and the G1 boundary fix (finding 12).

Gauge (`tools/spheroid_gauge.py`, every 50 frames): every arm is a spheroid until the
thinnest-cell band trips -- asphericity <= 0.05 and inverted wedges <= 0.8 % at the point of
refusal on all of them, so the shell is a shell; what drifts is a slowly widening thickness
spread (finding 14). Scored inside each arm's clean window (`size_report --gauge`):

| spec | clean to | cells | cycles | slope | r(L,V_b) | CV(V_d) | L | med V/v_ref | drift | CV(V) |
|---|---|---|---|---|---|---|---|---|---|---|
| size_sizer | 1150 | 200->1299 | 899 | -0.93 | -0.56 | 0.21 | 357 | 1.05 | 0.02 | 0.32 |
| size_adder | 1400 | 200->1640 | 1240 | -0.22 | -0.11 | 0.22 | 375 | 1.41 | 0.04 | 0.31 |
| size_doubler | 700 | 200->420 | 50 | -0.43 | -0.32 | 0.21 | 241 | 1.52 | 0.18 | 0.28 |
| size_timer | 1350 | 200->1457 | 1057 | -0.34 | -0.05 | 0.22 | 392 | 1.37 | 0.09 | 0.32 |
| size_grow_sizer | 1300 | 200->1322 | 922 | -0.63 | 0.06 | 0.18 | 389 | 0.82 | -0.09 | 0.30 |
| size_two_channel | 1250 | 200->1214 | 814 | -1.05 | -0.32 | 0.19 | 474 | 0.95 | 0.20 | 0.29 |
| cycle_sizer | 1150 | 200->1416 | 1016 | -0.88 | -0.37 | 0.22 | 392 | 1.03 | 0.15 | 0.30 |
| cycle_timer | 1100 | 200->1421 | 1021 | -0.27 | 0.03 | 0.21 | 370 | 0.93 | -0.01 | 0.32 |
| cycle_hazard | 950 | 200->1226 | 826 | -0.08 | 0.04 | 0.28 | 315 | 0.78 | -0.11 | 0.40 |
| cycle_dilution | 950 | 200->1155 | 755 | -0.32 | -0.07 | 0.24 | 336 | 0.84 | -0.03 | 0.36 |
| mech_target_percell | 1601 | 200->200 | - | - | - | - | - | - | - | 0.018 (polyhedron) |

Accepted as the baseline of the ladder. The rules separate the way the review predicts and the
checkpoint arms are stationary; sizers at -0.9, the adder at -0.2, timers at -0.3, the hazard at
-0.1. Every slope sits about 0.3 below its textbook value because the even target split pulls
both daughters toward the same half-target over the cycle, so a large piece shrinks and a small
one grows whatever the rule (the R0 effect, now bounded rather than total). Two open items go to
R3: `cycle_dilution` scores like a timer (slope -0.32, r(L,V_b) -0.07: the checkpoint is not
binding), and `size_grow_sizer` reads -0.63 with no cycle-length coupling, which is the growth
channel doing what Ginzberg et al. say it does.

14. **Thickness spread widens slowly with cell count on every dividing arm.** With everything
    else fixed, the thinnest cell falls under half the median between frames 700 and 1400
    (1,200-1,600 cells), and thickness CV passes 0.15 around the same time, on a shell whose
    asphericity is still 0.03. The 2026-09-06 archive shows the same slope (0.03 -> 0.09 over 801
    frames). A single-cell minimum is a harsh statistic and T1 flips leave thin cells behind; the
    band and the cause are R5's (engine) business. Until then a rung is scored inside its clean
    window, which is >= 2.5 doublings on every arm.

**R3 -- time and topology (opened 2026-09-12).** The user's diagnosis after the R2b movies:
the shell is a sphere but a rough one, and the division rate has to be set against the
mechanics' own relaxation time -- daughters must not arrive faster than energy minimisation can
answer a septum. Two instruments and one sweep:

- `tools/spheroid_gauge.py` now judges the PRISM as well as the shell (finding 15): apical/basal
  cap area over the ratio the curvature imposes (`trapezoid`), cap-centroid offset per thickness
  (`shear`), `sep` against the local cap normal (`tilt`), thickness range around the ring
  (`h_in_cell`). The mechanics-only control: 0 / 0 / 0 / 0.07. Every dividing run, the accepted
  archive included: 45-54 % trapezoids from frame 400, 16 % in-cell thickness range. This is the
  roughness the eye sees, and it is there long before the shell-level bands trip.
- `sweep_r{578,289,145}_i{30,90}` and `sweep_r578_i30_heal20` (archives `log/size_cycle/R3sweep`):
  `size_sizer` at growth rates 0.000578 / 0.000289 / 0.000145 (doubling in 400 / 800 / 1600
  frames), `relax_iters` 30 / 90, and the daughter-healing relax `local_relax 20`. Gauged for
  the working point where the prisms stay prisms over the run; the ratio doubling-time /
  relaxation-time is the number to report, with the relaxation time measured on
  `mech_target_percell` as the frames to settle after a target step.
- The dilution rule and the "one when" consolidation move to R4.

15. **Growth and division bend the prisms.** See R3 above: trapezoid fraction 0 on the control,
    ~0.5 on every dividing arm, `h_in_cell` 0.06 -> 0.16. The sweep (`log/size_cycle/R3sweep`,
    seven arms) bends them the same way at every growth rate and relaxation depth: 0.10 by
    frame 10 with the first T1 flips, before any division. Time was not the lever.
16. **A T1 flip moved two vertices and left their thickness vectors behind.** Re-aimed now
    (`edge_flip`): each moved vertex keeps its thickness and takes its ring neighbours' direction.
    Correct, and not the driver -- the trapezoid fraction did not move.
17. **The radial pin is the roughness.** `K_R 0.4` pulls every vertex to `R0 = (3 V0 / 4 pi)^(1/3)`,
    a solid ball's radius from the summed wedge targets, which grows as N^(1/3) under division at
    constant cell volume; a shell of cells that keep their footprint grows as N^(1/2). Pinned to
    it, the growing shell is held too small, footprints shrink, T1 flips come in storms and the
    prisms bend. `size_sizer`, 800 frames, deterministic: K_R 0.4 -> trapezoids 0.54, 2,952
    flips; K_R 0 -> 0.21, 276 flips, thickness CV 0.09, asphericity 0.03-0.04, and the shell
    expands R 4.9 -> 7.9 for 200 -> 593 cells (N^0.44). Giving the pin the shell's area-based
    radius instead does not help (0.53): the pin itself is the defect. K_R 0 on every dividing
    spec from R3b on; what still bends after that tracks the flip count and is layer 0's next
    question.

**R3b (62bcd98f, archives `log/size_cycle/R3b`).** `K_R 0` on the ten dividing specs. Gauge on
`size_sizer`: the shell bands hold to ~1200 frames (asphericity 0.03-0.04, thickness CV <= 0.11,
thinnest cell >= 0.65 of the median) -- longer and thicker than R2b -- but the prism band trips
at frame 200 and the trapezoid fraction climbs to 0.54 by 1601 with the flip count (2,564). Per
division: a daughter is 7 % trapezoidal as a mother, 16 % at the cut, 20 % eight frames later,
never straightened. Scored windows cut at the shell bands hold no cycles (asphericity sits on
the 0.04 band from frame ~250 without the pin), so R3b is a layer-0 rung with no layer-2 table.

18. **A stiffness on the thickness field makes the prisms prisms.** `kappa_h`, the Dirichlet energy
    of `sep` along the ring edges (R3d): on `size_sizer`, 800 frames, trapezoids 0.000 at every
    sampled frame (0.21 with `K_R 0` alone, 0.54 as shipped), in-cell thickness range 0.09,
    thickness CV 0.06, T1 flips 216, asphericity <= 0.03; with `K_R 0.1` as well, asphericity
    0.008 but the cells thicken 1.03 -> 1.67 as the pin still squeezes footprints, so the rung
    runs `K_R 0`. Finding 15's mechanism -- the energy's indifference between a prism and a
    frustum -- is closed by it.

**R3d (8fb4f80d, archives `log/size_cycle/R3d`) -- ACCEPTED, the ladder's working point.**
`kappa_h 0.2`, `K_R 0` on the eleven specs, 1601 frames, deterministic, on gpu_l4.

Gauge, both band sets: `size_adder`, `size_grow_sizer`, `size_two_channel`, `cycle_dilution`,
`mech_target_percell` SPHEROID for the whole run; `size_sizer`, `size_timer`, `cycle_sizer`,
`cycle_timer` to frames 1450-1600 (the prism band at 0.02-0.04); `cycle_hazard` to 1050;
`size_doubler` destroys its mesh from frame 600 (the null: a relative rule under asymmetric
division drives cells to zero volume, asphericity 2.5, half the cells inverted -- the review's
own argument, with a mesh to show for it). Thickness CV 0.05-0.09, thinnest cell >= 0.78 of the
median, in-cell thickness range 0.09-0.12 on every accepted arm.

Scored on polyhedron volumes inside the shell window (`size_report --gauge`, cb936177):

| spec | cells | cycles | slope | r(L,V_b) | CV(V_d) | L | med V/v_ref | drift | CV(V) |
|---|---|---|---|---|---|---|---|---|---|
| size_sizer | 200->2160 | 1760 | -1.00 | -0.70 | 0.17 | 365 | 0.96 | -0.07 | 0.33 |
| size_adder | 200->2318 | 1918 | -0.13 | -0.43 | 0.14 | 376 | 1.03 | -0.05 | 0.33 |
| size_doubler (<600) | 200->405 | 37 | 0.44 | 0.43 | 0.50 | 196 | 1.41 | -0.01 | 0.33 |
| size_timer | 200->2144 | 1744 | 0.68 | 0.01 | 0.22 | 391 | 1.06 | -0.02 | 0.35 |
| size_grow_sizer | 200->2144 | 1744 | -0.09 | 0.08 | 0.12 | 391 | 0.99 | 0.11 | 0.26 |
| size_two_channel | 200->1439 | 1039 | -1.23 | -0.71 | 0.10 | 556 | 1.25 | 0.06 | 0.19 |
| cycle_sizer | 200->2314 | 1914 | -1.14 | -0.78 | 0.14 | 409 | 1.26 | -0.04 | 0.25 |
| cycle_timer (<1550) | 200->3259 | 2859 | 0.76 | 0.04 | 0.24 | 372 | 0.87 | -0.10 | 0.35 |
| cycle_hazard (<1050) | 200->1480 | 1080 | 0.68 | 0.08 | 0.36 | 318 | 0.72 | -0.23 | 0.60 |
| cycle_dilution | 200->2375 | 1975 | -0.76 | -0.69 | 0.19 | 419 | 1.21 | 0.02 | 0.27 |
| mech_target_percell | 200->200 | - | - | - | - | 0.99 | - | 0.019 |

The layer-2 gate (R4's line in the ladder) is met on the working point without touching the
rules again: sizers -1.00 / -1.14, adder -0.13, timers +0.68 / +0.76, hazard +0.68, dilution
-0.76 (a sizer with a molecule under it, as it should be), the two-channel arm the tightest
population (CV 0.19, CV(V_d) 0.10), the growth channel alone adder-like (-0.09, CV(V_d) 0.12),
and every checkpoint arm stationary (|drift| <= 0.11). The doubler is the null and behaves as
one. What R4 still owes is consolidation, not correctness.

**The milestones, rendered (`config/tissue/ms*`, archives in `graphs_data/tissue`).** The sizer
arm under each rung's mechanics, 1601 frames, so the story can be watched side by side:

| spec | mechanics | gauge |
|---|---|---|
| `ms1_pinned_shell` | K_R 0.4, as shipped (R2b) | prisms bend from frame 100; by the end trapezoids 0.80, thickness CV 0.24, thinnest cell 9 % of the median |
| `ms2_unpinned_shell` | K_R 0 (R3b) | prisms bend from frame 200 (0.54 at the end); the shell breathes to asphericity 0.13 |
| `ms3_prism_shell` | K_R 0, kappa_h 0.2 (R3d) | a spheroid of prisms to frame 1500; trapezoids 0.04-0.08 at the end, thickness CV 0.09, thinnest 0.79 |
| `ms4_round_prism_shell` | K_R 0.1, kappa_h 0.2 | round early (asphericity 0.008) but the light pin squeezes the grown shell: trapezoids 0.76 and thickness CV 0.23 by the end |

| `ms5_two_channel` | ms3 + growth-rate control and a G1 checkpoint together | SPHEROID for the whole run; trapezoids 0.003, thickness CV 0.074 -- the tightest arm of the ladder (Kafri 2013, Cadart 2018) |
| `ms6_apoptosis` | ms3 + `cell_die[small]` | SPHEROID to 1600; 29 deaths, each shrinking to three neighbours and extruding |
| `ms6b_apoptosis_noT1` | the same with T1 flips off | trapezoids 0.29 and asphericity 0.10 by the end, and NOT ONE cell removed: finding 19 in one movie |
| `ms7_cycle_adder` | ms3 + the adder stated on the cycle (R4c) | SPHEROID; the "one when" form of the same rule |

`ms3` is the working point: no pin, the thickness field stiff. `ms1` -> `ms3` is the mechanics
story, `ms3` -> `ms7` the rules story, `ms6`/`ms6b` the topology one.

**R4a (3a7db6f5).** `cell_id` / `parent_id` on the cell set; `size_report` reads lineage off them
(identical numbers to the age-based reading on a 700-frame cut: 83 cycles, slope -0.93).

**R4b (archives `log/size_cycle/R4rig`).** The apoptosis rig: `rig_apop_{small,smaller,crowded,
competition}_sizer`, `rig_apop_{small,crowded}_timer`, `rig_apop_{small,crowded}_sizer_noT1` on
the working-point mechanics, 1601 frames; `tools/death_report.py` joins every death to volume at
death, birth volume, age, phase, neighbours and mark-to-removal latency. `cell_die`'s size and
growth readings are the measured volume now (they read the target, which is the mother's half
for every daughter, so `small` could not see a small daughter).

Landed (`log/size_cycle/R4rig`, 1601 frames, gpu_l4). Gauge: every arm's shell holds its
thickness and prism bands (thickness CV <= 0.13, trapezoids <= 0.06); asphericity drifts to
0.045-0.063 on the sizer arms (the 0.04 band is marginal without the pin; to be widened for the
rig) and a single inverting cell appears from frame 950 -- an extrusion in progress, which the
`inv_wedge == 0` band should tolerate on apoptosis runs. `tools/death_report.py`:

| spec | cells | deaths | per 100 cell-frames | V_birth | age | neighbours at removal | latency | first death |
|---|---|---|---|---|---|---|---|---|
| rig_apop_small_sizer | 200->2382 | 29 | 0.002 | 0.77 | 160 | 3 | 64 | 188 |
| rig_apop_smaller_sizer | 200->2416 | 33 | 0.003 | 0.87 | 220 | 3 | 56 | 208 |
| rig_apop_crowded_sizer | 200->2452 | 19 | 0.001 | 0.92 | 336 | 3 | 148 | 252 |
| rig_apop_competition_sizer | 200->2287 | 22 | 0.002 | 0.90 | 274 | 3 | 136 | 256 |
| rig_apop_small_timer | 200->2088 | 26 | 0.002 | 0.83 | 156 | 3 | 86 | 416 |
| rig_apop_crowded_timer | 200->1971 | 25 | 0.002 | 0.93 | 168 | 3 | 56 | 524 |
| rig_apop_small_sizer_noT1 | 200->2443 | 0 | 0 | - | - | - | - | - |
| rig_apop_crowded_sizer_noT1 | 200->2403 | 0 | 0 | - | - | - | - | - |

(volumes over `v_ref`, ages and latencies in frames; every death leaves at 3 neighbours and
~zero volume, the shrink-shed-extrude path.)

19. **Death needs topology.** Without T1 flips not one marked cell is removed in 1601 frames on
    either rule: a marked cell shrinks but cannot shed neighbours, never reaches the triangle
    the extrusion needs, and the shell deforms around the cells that cannot leave (asphericity
    0.10-0.15, 3.7 % inverted wedges). The shrink-shed-extrude sequence is a topological
    operation as much as a mechanical one.
20. **The death models select what they say they select, and the size rule sets when.** `small`
    takes cells born at 0.77 of the reference and young (160 frames), `smaller` 0.87, `crowded`
    and `competition` normal-born cells (0.90-0.93) and old (274-336). Under a timer the first
    death comes 200-300 frames later than under a sizer at the same threshold: a sizer makes
    small cells early (asymmetric septa on a cell that divides at a fixed size), a timer does not.
    Rates are low (0.001-0.003 per 100 cell-frames) at `max_mark_frac 0.005` and these thresholds;
    the rig's next arms turn that dial.

**R4c (archives `log/size_cycle/R4c`).** One "when": the size rules on `cell_cycle` (adder and
doubler added as G1 rules), the five divide-family specs rewritten as degenerate cycles
(`t_s = t_g2 = t_m = 0`, `cell_divide[model: cycle]`), the divide-family models deprecated and
kept for the okuda archive. A daughter's cycle now starts when its birth volume has been read
(finding 21). ACCEPTED: the rules keep their identity and every arm's division volumes tighten.

| spec | slope R3d -> R4c | CV(V_d) R3d -> R4c | cells at 1601 |
|---|---|---|---|
| size_sizer | -1.00 -> -1.08 | 0.17 -> 0.05 | 2667 |
| size_adder | -0.13 -> -0.20 | 0.14 -> 0.08 | 2392 |
| size_doubler | +0.44 -> +0.85 | 0.50 -> 0.29 | 1579 (<1400) |
| size_timer | +0.68 -> +0.63 | 0.22 -> 0.16 | 2038 |
| size_grow_sizer | -0.09 -> -0.24 | 0.12 -> 0.09 | 2038 |
| size_two_channel | -1.23 -> -1.07 | 0.10 -> 0.07 | 1146 |
| cycle_sizer | -1.14 -> -1.03 | 0.14 -> 0.13 | 2151 |
| cycle_timer | +0.76 -> +0.64 | 0.24 -> 0.23 | 2546 |
| cycle_hazard | +0.68 -> +0.49 | 0.36 -> 0.36 | 958 (<1000) |
| cycle_dilution | -0.76 -> -0.97 | 0.19 -> 0.15 | 2128 |
| mech_target_percell | CV(V) 0.019, unchanged | | 200 |

Eight of the ten move by <= 0.16; the two that move further are the stochastic hazard and
`cycle_dilution`, which becomes the sizer-with-a-molecule it is supposed to be (-0.97 against
the sizer's -1.03). The tightening of CV(V_d) is the consolidation's own effect: the cycle form
waits for the re-read `Vbirth` before it starts integrating, so no rule spends its first frames
chasing the septum's geometric piece.

21. **A daughter's cycle must start after the mechanics has answered the septum.** The cycle
    integrated added volume from the cut; the piece the septum makes is restored to its target
    within a frame, and that jump counted as growth, so the cycle adder left G1 at a fixed
    `piece + delta` and scored as a sizer (-0.96). Held at p = 0 until `cell_divide` has re-read
    `Vbirth` (age 0 -> 1): -0.12, the divide-family adder's number.

22. **The seed is not at rest in thickness, and the rest offset is why (R3e, measured, parked).**
    `tools/equilibrium_h.py` on the working-point energy: seeded at 0.88 the shell settles at
    1.163; seeded at 1.163 it settles at 1.243. There is no fixed point to seed at because the
    rest offset `mono_delta` is solved with `sep` frozen at the seeded thickness -- the
    calibration answers "which target puts THIS shell's mid-surface at rest" and the thickness
    then moves under it. A seed at rest in every degree of freedom means calibrating with the
    thickness free, which is a mechanics change with its own gate; the `ref_frame` window
    (60 frames; the ramp settles in 56-78) covers it for the ladder, so R3e stays parked behind
    R5 rather than opened now.

**R6 (2026-09-12).** Four working points registered from their archives in `graphs_data/tissue`:
`ms3_prism_shell` (the working point), `ms5_two_channel`, `ms6_apoptosis`, `ms7_cycle_adder`;
`ms3` joins the quick set. One defect found registering them (finding 23).

23. **The regression reruns were not deterministic and the archives are.** Every archive on disk
    comes from `jobs/*.sh`, which export `PLEXUS_STRICT_DETERMINISM=1`; `regression_lib.run_cut`
    did not, so a rerun took the TF32 and compiled paths and landed about 1 % away in median
    radius over 150 frames -- enough to fail the 1 % band on physics that had not changed. The
    runner sets it now. The two references refreshed earlier today under the old runner
    (`divide_growing_ball`, `mesh_mpm_spheroid_nominal`) were refreshed again under the new one.

## 4. The ladder, v2

| rung | layer | change | gate |
|---|---|---|---|
| R3b | 0 | `K_R 0` on the dividing specs; flip re-aims `sep` | gauge SPHEROID (shell + prism) for >= 3 doublings on `size_sizer`; the prism bands are the ones to watch |
| R3c | 0 | the septum and `local_relax` on prisms; per-event prism damage | trapezoid fraction flat across a division wave |
| R3d | 0 | `kappa_h`, a stiffness on the thickness field (finding 18) | prism bands green on every arm for >= 3 doublings; shell bands too |
| R3e | 0 | the seed at rest (findings 7, 22): calibrate the rest offset with the thickness free; `ref_frame` retired | frame-0 volumes within 5 % of frame-60 -- parked behind R5 |
| R4a-c | 2 | `cell_id`; the apoptosis rig; one "when" | DONE: R3d reproduced (8/10 within 0.16), CV(V_d) tighter everywhere |
| R4b | 3 | the apoptosis rig | `death_report` rows for every arm; deaths never off a bent mesh |
| R5 | eng | representation and engine, as before | tick-0 invariant; flags 29 -> <= 20 |
| R6 | -- | DONE: ms3/ms5/ms6/ms7 registered, `QUICK` re-pointed, the runner made deterministic (finding 23) | the registry green |

Out of scope: MPM, ECM -- touched only through the shared reader, and gated there.
