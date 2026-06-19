# Dicty model-discovery loop — full log

Append-only. Each batch: one design block (16 hypotheses → planned sweeps), one
block per sweep with the morphological observation + verdict after results come in,
and a one-line batch summary at the end. The ledger `dicty_knowledge.md` carries the
*adjudicated* claims; this log carries the *trail*.

---

## Batch 1 — Design (parent = prior best + relay op, gain=0 ablation default)

**Parent config (`specs/dicty_loop_base.yaml`):** identical to the prior best
(`spring.kadh=120, sense.gain=40, inflow.rate=2.0, camp.{diffusion=0.02, decay=0.20},
secrete.rate=8, k_rep=60, r0=0.022, r_on=0.22, mu_f=0.05, random_walk=0.003,
dt=0.5, vmax=0.06, cell.n=767`) — **plus** a new `relay` operator inserted between
`camp.diffuse` and `sense` with `gain=0.0` (no-op), `thr=0.10`, `coupling=1.0`,
`eps=0.02`, `gamma=1.0`. The default no-op means parent ≡ prior best; the relay.gain
sweep alone delivers ablation (v=0) + sufficiency (v>0) in a single curve.

**Mechanism added (dicty_ops.py: `Relay`):** FitzHugh–Nagumo-style excitable update on
the existing `camp` grid + a hidden refractory grid `_relay_v`:
`u ← max(0, u + dt·(gain·u·(u−thr)·(1−u) − coupling·v))`,
`v ← v + dt·eps·(u − gamma·v)`. Above threshold the field fires regeneratively,
then v rises and quenches — producing travelling-wave catchment basins beyond the
pure-diffusion length.

**Sweeps (16 × 16 = 256 sims). Hypothesis per sweep:**
1. `relay.gain` ∈ [0, 0.5, …, 120] — H1 necessity (v=0) + H2 sufficiency (v≫0) + H3
   non-monotone (peak at intermediate gain). **THE CENTRAL EXPERIMENT.**
2. `camp.diffusion` ∈ [0.001, …, 0.60] — chemical reach; in the gain=0 control,
   re-validates passive-diffusion regime; in Batch 2 (gain>0) governs wavelength.
3. `camp.decay` ∈ [0.01, …, 2.00] — chemical lifetime; sets refractory window when
   relay is on.
4. `secrete.rate` ∈ [0.5, …, 150] — amplitude of self-supplied gradient.
5. `sense.gain` ∈ [5, …, 1300] — H6: prior peak at upper edge (40) is an
   extrapolation artefact; true optimum may be higher.
6. `spring.kadh` ∈ [0, …, 1300] — H7: extends prior best (114) into the strand
   regime to characterise filament-vs-blob transition.
7. `spring.r_on` ∈ [0.04, …, 0.70] — adhesion cutoff range.
8. `spring.k_rep` ∈ [10, …, 550] — volume exclusion strength.
9. `spring.r0` ∈ [0.010, …, 0.062] — equilibrium spacing.
10. `spring.mu_f` ∈ [0.005, …, 0.45] — H10: friction trade-off.
11. `spring.delta` ∈ [0.0001, …, 0.07] — H8: never swept before; sharpness of the
    adhesion sigmoid gate.
12. `inflow.rate` ∈ [0, …, 16] — H9: trade-off between N-growth and concentration.
13. `random_walk.strength` ∈ [0, …, 0.22] — noise floor.
14. `cell.n` ∈ [400, …, 2500] — starting density.
15. `dt` ∈ [0.10, …, 1.20] — integration step.
16. `vmax` ∈ [0.01, …, 0.64] — speed cap.

**Why this design.** Prior batches (see ledger) falsified that the model
{AR + self-chemotaxis + influx} can reach the real inner_mass=0.61 by parameter
tuning alone — the morphology is "polka-dot" of many small mounds, peaking ~0.32.
This batch's primary scientific test is the relay-wave mechanism (sweep 1).
Slots 2–16 do response-surface refinement around the new parent, including
re-mapping the cAMP machinery (which now interacts with the relay) and one new
parameter (`spring.delta`) that has never been swept.

**Predicted outcomes per slot (to compare against the data on read-back):**
- Sweep 1: at gain=0, inner≈0.32 (matches ablation = old parent). At gain≈10–40
  a peak forms with inner→0.5+ and fewer/larger blobs. Past gain≈60, field saturates
  → inner drops.
- Sweep 5 (sense.gain): peak between 60 and 200 — beyond that, gradient over-response
  causes oscillation/collapse.
- Sweep 6 (kadh): plateau or shallow decline above 200 — filaments not blobs.
- Sweep 11 (delta): peaked or monotone — needs to see strip morphology.

— design recorded, 256 sims to run next.

---

## Batch 1 — Results (read-back)

REAL inner_mass = 0.606. Prior parent (relay.gain=0) inner_mass = 0.274, loss = 1.5503.

### Batch 1 Sweep 0 — relay.gain  [SUPPORTED — sufficiency partial, necessity unclear]
Hypothesis (H1+H2+H3): relay.gain=0 ablation pins inner≈0.27; relay.gain≫0 closes the gap; non-monotone with peak at moderate gain.
Response: complex. Quick early peak inner=0.335 at gain=0.5; dip 0.19 at gain≈4–9 (waves disrupt baseline gradient); then slow monotone climb from gain≈42 onwards. **Best at gain=120 (max swept): loss=0.5474, inner=0.404, n_final=1249.** No saturation seen in tested range.
Morphology (strip): low gains noisy polka-dots; mid-gain washed-out diffuse cloud (relay smears gradient); high gains (70+) develop visibly larger and more concentrated central blob that REAL also has — clearest progression toward few-large-mounds in any sweep this batch.
Verdict: **H2 (sufficiency) partially supported** — relay does monotonically improve inner_mass and loss in the high range; the gap to REAL (0.61) is not yet closed but the curve has not saturated. **H3 (non-monotone with intermediate peak)** falsified — the dominant trend is monotone-up at the high end. **H1 (necessity)** inconclusive — ablation (gain=0) inner=0.274 is the same as prior parent, but this only re-confirms prior result; the comparison test ("does removing relay break a relay-on config") needs a future batch.
Knowledge update: promote "relay reduces loss & improves inner_mass monotonically up to 120" to Open Questions; extend gain range in Batch 2.

### Batch 1 Sweep 1 — camp.diffusion  [INCONCLUSIVE, near-flat]
Hypothesis (H4): in gain=0 control, re-validates passive regime.
Response: flat. inner ≈0.20±0.03 across [0.001..0.6]; loss best at 0.001 (1.05) but elsewhere 1.5–2.0. No clear peak.
Morphology: identical polka-dot pattern at every value; small differences in dot spacing only.
Verdict: With relay off (gain=0 control) camp.diffusion is non-discriminating for inner_mass in the tested range. Saturated param.
Knowledge update: drop from primary sweep set; revisit only paired with relay-on.

### Batch 1 Sweep 2 — camp.decay  [PARTIALLY SUPPORTED]
Hypothesis (H5): too-slow decay → uniform field, gradient collapses.
Response: flat-ish for decay ≤1.0 (inner 0.18–0.29, loss ~1.4–1.7); loss explodes for decay≥1.6 (3.7→7.3); inner collapses at high decay.
Morphology: low/mid values produce normal polka-dots; very high decay (1.6, 2.0) — cells over-clump driven by saturated field → degenerate single-spike artefact (visible bright dot + sparse halo).
Verdict: H5 supported in tail — over-persistent cAMP destroys gradient and produces a degenerate config that is NOT real-like. Sweet spot in 0.1–0.5.
Knowledge update: cap decay sweep at 0.5 in Batch 2.

### Batch 1 Sweep 3 — secrete.rate  [SUPPORTED — saturation knee]
Hypothesis: needs amplitude above threshold; saturates above.
Response: huge loss at rate=0.5 (6.4 — no gradient) and 1.5 (3.7); plateau from rate=3 onwards (loss ~1.4–1.7).
Morphology: rate≤1.5 = scatter (no chemotaxis force); rate≥3 = standard polka-dot, no further improvement at high rates.
Verdict: secrete.rate needs to clear a knee (~3) and is then saturated. Saturated param above threshold.
Knowledge update: keep ≥3 in parent; narrow sweep to 2–30.

### Batch 1 Sweep 4 — sense.gain  [FALSIFIED — H6 not supported]
Hypothesis (H6): prior peak at upper edge (40) is an extrapolation artefact; true optimum lies higher.
Response: nearly flat across 5–1300 (inner 0.13–0.28, loss 1.41–2.6). Best loss at gain=30 (1.41), slight degradation above 200.
Morphology: identical polka-dot at all gains; very high gains → slightly diffuser due to over-response noise.
Verdict: **H6 falsified.** sense.gain is saturated in 30–500; raising it above 40 brings no benefit. The "extrapolation artefact" was an artefact of the prior sweep range, not a hidden higher optimum.
Knowledge update: add to Falsified — sense.gain ∈ [30,500] is a saturated plateau, no hidden high-gain optimum.

### Batch 1 Sweep 5 — spring.kadh  [PARTIALLY SUPPORTED — H7]
Hypothesis (H7): kadh→∞ gives filaments not blobs; extended range should show this.
Response: best loss at kadh=160 (1.30, inner=0.27). Above 250, loss climbs; at kadh=800 loss=4.9; at kadh=1300 loss=6.5 with inner collapsing to 0.06.
Morphology: low/mid kadh = polka-dots; very high kadh → cells over-merge into thin filaments/lines that span the field, then collapse to almost nothing (degenerate).
Verdict: H7 supported — long-range adhesion alone produces filaments, not blobs; pure adhesion saturates near kadh≈160.
Knowledge update: cap kadh sweep at 700; main interest now is r_on (see Sweep 6).

### Batch 1 Sweep 6 — spring.r_on  [BREAKTHROUGH — REAL inner_mass reached]
Hypothesis: adhesion cutoff range mapping.
Response: monotone climb from inner=0.21 (r_on=0.04) to **inner=0.746 (r_on=0.45, OVERSHOOTS REAL 0.61)**. Critical points: **r_on=0.40 → inner=0.586 (≈ REAL 0.606!) at loss=0.749**; best loss at r_on=0.35 (loss=0.669, inner=0.42). Loss dips again at r_on≥0.55 (after collapse re-stabilises).
Morphology (strip): polka-dots up to r_on=0.22; from 0.26–0.40 dots begin merging into fewer larger compact blobs; **r_on≥0.45 produces 1–2 compact mounds qualitatively matching REAL**; r_on≥0.55 over-merges to thin strand.
Verdict: **strongest finding of the batch.** The adhesion *interaction range* r_on (NOT amplitude kadh) is the discriminating control — at r_on≈0.40 the model REPRODUCES the real inner_mass and shows few-compact-mound morphology. This is a *new* result: the prior ledger claim "adhesion alone fails (filaments not blobs)" was extrapolated from kadh sweeps only, not r_on.
Knowledge update: NEW PRINCIPLE candidate — *adhesion reach (r_on) is the missing degree of freedom*; coalescence may not require a relay-wave mechanism after all. Promote to Open Question for Batch 2.

### Batch 1 Sweep 7 — spring.k_rep  [INCONCLUSIVE / saturated]
Hypothesis: routine map of volume-exclusion strength.
Response: noisy near-flat (inner 0.16–0.30, loss 1.27–2.04). Best loss at k_rep=380 (1.28), but variance dominates.
Morphology: same polka-dot at all values, very slight spacing change.
Verdict: saturated parameter at current parent. Keep at default ~60.
Knowledge update: narrow sweep around 30–200.

### Batch 1 Sweep 8 — spring.r0  [SATURATED — drop]
Response: flat (inner 0.12–0.28, loss 1.37–2.27, no trend).
Morphology: identical strips.
Verdict: saturated. DROP from primary sweep set.
Knowledge update: r0=0.022 default is fine.

### Batch 1 Sweep 9 — spring.mu_f  [SUPPORTED — H10 partial]
Hypothesis (H10): friction tradeoff; high mu_f pins cells.
Response: flat for mu_f≤0.05 (inner~0.2, loss~1.4–1.8); progressive collapse above 0.08 (loss climbs to 3.95 at mu_f=0.45).
Morphology: low mu_f normal; high mu_f → cells nearly frozen, sparse degenerate.
Verdict: H10 partly supported (high-mu collapse confirmed). No interior peak — the "very-low overshoot" branch is not visible.
Knowledge update: keep mu_f sweep in 0.005–0.08.

### Batch 1 Sweep 10 — spring.delta  [FALSIFIED — H8]
Hypothesis (H8): sharper adhesion gate → blobs.
Response: flat (inner 0.14–0.27, no monotone trend).
Morphology: visually identical across all 16 values.
Verdict: **H8 falsified** — the sigmoid-gate sharpness has no effect on coalescence at the current parent. DROP from primary sweep set.
Knowledge update: add to Falsified — spring.delta is not a control over coalescence morphology.

### Batch 1 Sweep 11 — inflow.rate  [SUPPORTED — H9]
Hypothesis (H9): high inflow injects scattered fresh cells, fights concentration; interior peak.
Response: clean interior peak. **Best loss at inflow=0.8 (loss=1.075, inner=0.276)**; loss climbs monotonically for inflow≥1.5. Inner_mass also slightly higher at low inflow (0.5: 0.314; 0.8: 0.276).
Morphology: low inflow = sparser, more locally-organised; rate=1.5+ = visibly more scattered fresh entries; high rate (≥9) = field grows uniformly noisy.
Verdict: **H9 supported.** Optimum well below prior parent's value (2.0). Move parent to 0.8.
Knowledge update: ESTABLISH — inflow.rate exhibits an interior optimum near 0.5–1.0; the prior choice of 2.0 was too high.

### Batch 1 Sweep 12 — random_walk.strength  [INCONCLUSIVE]
Hypothesis: noise floor mapping.
Response: noisy near-flat (inner 0.17–0.27, loss 1.23–2.4). Best loss at strength=0.055 (1.23) but noise dominates.
Morphology: identical at low/mid; very high (≥0.16) clearly noisier.
Verdict: weak preference for modest noise (~0.05). Mostly saturated.
Knowledge update: keep at 0.003–0.05 default.

### Batch 1 Sweep 13 — cell.n  [INCONCLUSIVE]
Response: noisy. Best loss at n=600 (1.10, inner=0.31); plateau above 1000.
Morphology: lower n looks too sparse; higher n uniform polka-dot.
Verdict: keep n=767 (biology) and let inflow add the rest. Saturated for our purpose.
Knowledge update: not a discriminating param.

### Batch 1 Sweep 14 — dt  [SURPRISE — strand regime]
Response: best loss at dt=0.8 (0.976, inner=0.251); loss climbs at dt=1.2.
Morphology (strip): **striking transition** — for dt≥0.55 cells form thin filament/strand networks instead of dots (large effective per-step displacement → chemotactic overshoot creates streaming chains). Above dt=0.9 it degenerates. This is a *different* dynamical regime, not pure noise.
Verdict: dt is acting like an effective integration-step / velocity-amplification knob that turns the system into a streaming regime. Interesting morphologically but not a principled mechanism. Keep dt=0.5 in parent for now.
Knowledge update: note as Open Question — large-dt strand regime; do not pursue as a model fit but flag.

### Batch 1 Sweep 15 — vmax  [PARTIALLY SUPPORTED]
Response: best loss at vmax=0.10 (1.24, inner=0.232); above 0.16 loss climbs sharply (2.3+).
Morphology: low vmax frozen; vmax=0.05–0.10 normal polka-dot; vmax≥0.16 cells fly off into uniform diffuse.
Verdict: speed cap has a clean optimum around vmax=0.08–0.10 (slightly higher than parent 0.06).
Knowledge update: move parent to vmax=0.08.

---

### Batch 1 — Summary
**New best config (single-param-changes from prior parent):**
- by loss: `relay.gain=120` → loss=**0.5474**, inner_mass=**0.404** (n_final=1249).
- by inner_mass: `spring.r_on=0.40` → inner_mass=**0.586 (≈ REAL 0.606)**, loss=0.749.
- The two findings come from independent sweeps and have not been combined yet.

**Key insight:** the ledger's claim that "the {AR+chemotaxis+influx+adhesion} mechanism set is *falsified as sufficient*" must be **softened**. spring.r_on=0.40 produces inner_mass=0.586 with visibly few-compact-mound morphology — a regime the prior batches missed. The new central question is whether (a) r_on alone is sufficient, (b) relay is independent additive, or (c) they synergise.

**New parent (`specs/dicty_loop_base.yaml`):** relay.gain=120 (single-best by loss). Other params kept at prior defaults to keep the parent identical to the actually-tested best config. Sweeps in Batch 2 will probe r_on×relay synergy and combined regimes around this.

**Falsifications added:** H6 (no hidden high sense.gain optimum); H8 (spring.delta has no effect).
**New mechanism additions for Batch 2:** none yet — Batch 2 deepens parameter-tuning of an apparently-promising existing mechanism set. Mechanism additions (e.g. polarity/streaming) deferred until r_on×relay combination is characterised.

---

## Batch 2 — Results (read-back)

REAL inner_mass = 0.606. Parent (relay.gain=120, inflow.rate=2.0, r_on=0.22, etc.) baseline
inner_mass = 0.404, loss = 0.5474 (cf. Batch 1 sweep 0 v=120).

### Batch 2 Sweep 0 — relay.gain ∈ [40,800]  [FALSIFIED — H1: no saturation; interior optimum found]
Hypothesis (H1): response keeps improving past 120 until a saturation knee.
Response: noisy mid-range (inner 0.33–0.43, loss 0.44–0.92) then sharp degradation above gain=400; **best loss at gain=210 (loss=0.445, inner=0.427)**. Loss climbs to 1.49 at gain=500 and stays high through 800.
Morphology (strip): all values are diffuse scatter (no compact mound at any gain); mid-gain has mild central concentration; high gain (≥500) becomes more uniformly noisy and less concentrated.
Verdict: **H1 falsified** — relay.gain does NOT keep improving past 120. There is a broad interior optimum near 100–210, then a cliff. The Batch-1 monotone trend (40→120) was the rising flank of an interior peak, not the start of monotone-up.
Knowledge update: REVISE the Batch-1 Open Question "relay improves monotonically up to 120" → relay.gain has an interior optimum near 150–250; high gain DEGRADES via over-firing.

### Batch 2 Sweep 1 — spring.r_on ∈ [0.20,0.70]  [STRONGLY SUPPORTED — H2: r_on dominant even with relay on]
Hypothesis (H2/CRITICAL): r_on=0.40 still reaches inner≈0.6 at relay-on parent.
Response: inner_mass climbs monotonically and steeply from 0.29 (r_on=0.20) → 0.55 (0.28) → 0.78 (0.34) → 0.85 (0.40) → saturating around 0.91 (0.50+). Crosses REAL inner=0.61 between r_on=0.28 and 0.31. **Best loss at r_on=0.28 (0.328, inner=0.549).** Loss climbs steeply for r_on≥0.40 (over-merge regime — inner overshoots).
Morphology (strip): r_on≤0.24 polka-dots; 0.28–0.34 progressive merging into a few compact mounds; r_on≥0.40 a single tight central blob that *overshoots* REAL morphology (REAL has several mounds, sim has one).
Verdict: **H2 strongly supported** — r_on retains its effect under relay-on parent; it remains the dominant lever for coalescence. Relay does NOT cancel or synergise — it is roughly independent. The loss-minimum is at LOWER r_on (0.28) than the inner_mass-match value (~0.31), because radial-profile loss penalises the overshoot once r_on>0.31.
Knowledge update: ESTABLISH (3rd batch needed) — adhesion *reach* r_on is the principal control of mound compactness; relay does not modulate this.

### Batch 2 Sweep 2 — relay.thr ∈ [0.02,1.2]  [SUPPORTED — H3: interior optimum confirmed]
Hypothesis (H3): interior optimum near 0.05–0.15.
Response: **best loss at thr=0.15 (0.340, inner=0.506)**; loss explodes for thr≥0.50 (3.79→7.25 at thr=0.65) as the field essentially never fires. Inner_mass crashes to 0.04 at thr=0.65.
Morphology: low thr (≤0.04) noisy diffuse cloud; **thr=0.10–0.18 shows the cleanest compact central blob in the whole batch** (better than relay-off mid-range); thr≥0.50 degenerate single bright pixel + sparse halo (no firing → no field → no aggregation).
Verdict: **H3 supported.** thr=0.15 is the optimum; raise this to a tentative Established result (seen 1×).
Knowledge update: relay.thr has a clear interior optimum at 0.10–0.18; thr≥0.5 is functional ablation.

### Batch 2 Sweep 3 — relay.eps ∈ [0.001,0.5]  [INCONCLUSIVE — H4 weakly supported]
Hypothesis (H4): too-small → no recovery; too-large → no refractory.
Response: highly noisy; weak preference for mid-range. **Best loss at eps=0.14 (0.397, inner=0.467)**; secondary minimum at eps=0.19 (0.416). No catastrophic regime at either extreme.
Morphology: visually similar polka-dot scatter across all 16 values; slight central concentration mid-range.
Verdict: H4 weakly supported. eps is a soft knob; the system tolerates 0.01–0.20 well, degrades modestly outside that range.
Knowledge update: relay.eps is non-critical in [0.01,0.20]; do not waste future slots refining beyond a narrow check.

### Batch 2 Sweep 4 — inflow.rate ∈ [0,5]  [SURPRISE — H5 falsified; LOWEST LOSS OF BATCH at inflow=0]
Hypothesis (H5): the Batch-1 interior optimum at 0.5–0.8 persists at the relay-on parent.
Response: **inflow=0 is the global best of the batch: loss=0.312, inner=0.561 ≈ REAL 0.606.** Inner_mass declines from 0.56 (rate=0) → ~0.45 (rate≤0.6) → ~0.35 above rate=2. Loss is monotone-up from 0 with noise.
Morphology: rate=0 most-concentrated cloud (closest to REAL inner_mass); higher rates progressively more scattered as fresh cells appear randomly across the domain.
Verdict: **H5 falsified.** The interior optimum at 0.5–0.8 from Batch 1 does NOT persist; with relay on, ZERO inflow is best by both loss and inner_mass. *Critical caveat:* at inflow=0 the cell count stays at n=767 (REAL grows to 1413), so the metric is satisfied while the n-growth target is violated. The model is achieving a low loss by NOT receiving fresh scattered cells — i.e. there is no mechanism to incorporate fresh cells into existing aggregates. **This is the new central scientific gap.**
Knowledge update: PROMOTE TO CENTRAL OPEN QUESTION — "the model has no mechanism to retain fresh-influx cells in existing aggregates; uniform inflow actively destroys concentration." A **biased-inflow** (cells enter near high-cAMP regions) or a **post-entry retention** mechanism is needed to satisfy both inner_mass AND n-growth. This is the next mechanism to add.

### Batch 2 Sweep 5 — spring.kadh ∈ [0,850]  [SUPPORTED — H6: ceiling shifted upward]
Hypothesis (H6): with relay on, kadh ceiling shifts up.
Response: loss best at **kadh=550 (0.330, inner=0.466)**; second-best at kadh=290 (0.420). Loss climbs sharply for kadh≥730 (filament regime). Inner_mass jumps at kadh=730 (0.79, over-merged single strand) then crashes at 850 (0.26).
Morphology: kadh≤30 fully dispersed; mid-range polka-dots; kadh=550–640 visibly more concentrated central blob; kadh=730 a thin curved filament; kadh=850 degenerate.
Verdict: **H6 supported** — relay-on shifts the kadh ceiling from ~160 (Batch 1) up to ~550–640. Beyond that, the same filament-collapse regime appears.
Knowledge update: kadh useful range with relay-on is 200–600 (≈ 4× the relay-off range).

### Batch 2 Sweep 6 — camp.diffusion ∈ [0.001,0.6]  [FALSIFIED — H7: lower not higher diffusion wins]
Hypothesis (H7): higher diffusion → bigger catchment → fewer giant mounds.
Response: monotone DEGRADATION as diffusion increases through 0.10, loss=2.12 at diff=0.10 vs 0.38 at diff=0.001. **Best loss at diff=0.001 (0.375, inner=0.451).** Slight recovery at very-high diffusion (0.6 → 0.65) as the field becomes uniform and the relay effectively turns off.
Morphology: low diffusion = visible bright central concentration; mid-diffusion = uniformly noisy; very-high = back to uniform soup.
Verdict: **H7 falsified.** Increasing diffusion in the relay-on regime does NOT create one giant mound; it homogenises the field and destroys gradients. Lower diffusion is better — the relay manufactures its own long-range structure; passive diffusion only smears it.
Knowledge update: camp.diffusion sweet spot with relay-on is ≤0.005; revise parent toward 0.001.

### Batch 2 Sweep 7 — camp.decay ∈ [0.01,1.5]  [INCONCLUSIVE — flat with end collapse]
Hypothesis (H8): interior optimum below 0.5.
Response: roughly flat loss 0.55–1.0 across [0.025,1.0]; spike at decay=1.5 (loss=1.62, inner=0.22). **Best at decay=0.025 (0.538, inner=0.394).**
Morphology: similar polka-dots throughout; degenerate at decay=1.5 (gradient destroyed).
Verdict: H8 inconclusive. camp.decay is non-critical in [0.025, 1.0]; degrades above 1.0.
Knowledge update: keep decay at 0.025–0.20 in parent; not a priority lever.

### Batch 2 Sweep 8 — secrete.rate ∈ [1,60]  [SUPPORTED — H9: knee + collapse]
Response: huge loss at rate=1 (2.61, no gradient); plateau across 6–28 with best at **rate=15 (0.317, inner=0.453)**; catastrophic collapse at rate=38 (loss=2.93, inner=0.10) and high rate=60.
Morphology: rate=1 sparse few-mounds; mid-rate polka-dots; rate=38 a degenerate single bright pixel surrounded by void.
Verdict: H9 supported — knee at rate≈3 then broad plateau 6–28 then over-saturation collapse at rate≥38.
Knowledge update: secrete.rate sweet spot 8–18 at relay-on parent.

### Batch 2 Sweep 9 — sense.gain ∈ [10,400]  [PARTIALLY FALSIFIED — H10: low end wins]
Hypothesis (H10): saturated plateau across [30,500].
Response: **best loss at gain=10 (0.321, inner=0.456)**, with slight degradation above 50. Plateau roughly flat 25–230.
Morphology: similar polka-dots; very-high gain slightly diffuser (over-response noise).
Verdict: H10 falsified — sense.gain is NOT a flat plateau; the optimum is at the LOW end (10), and gain=40 (parent) is sub-optimal. Likely because relay now provides the structuring force; high sense gain just amplifies noise from a fluctuating relay field.
Knowledge update: revise parent to sense.gain=10; relay-on supplants high-gain chemotaxis.

### Batch 2 Sweep 10 — spring.k_rep ∈ [15,400]  [INCONCLUSIVE / saturated]
Response: noisy near-flat, best at **k_rep=75 (0.473, inner=0.411)**.
Morphology: identical polka-dots.
Verdict: k_rep saturated in [55,200]; keep at ~75.

### Batch 2 Sweep 11 — spring.mu_f ∈ [0.005,0.2]  [PARTIALLY SUPPORTED — interior optimum]
Response: noisy interior optimum, **best at mu_f=0.105 (0.383, inner=0.480)**; secondary at 0.130 (0.422).
Morphology: low mu_f sparse, mid-range standard, high mu_f starts to over-pin.
Verdict: mu_f tuning prefers 0.08–0.13 (3× higher than current parent 0.05) — moderate friction stabilises the relay-driven gradient.
Knowledge update: revise parent toward mu_f=0.10.

### Batch 2 Sweep 12 — dt ∈ [0.3,1.2]  [CONFIRMED — strand regime above 0.55]
Response: **best loss at dt=0.5 (0.547, inner=0.404)**; loss climbs above dt=0.55 with strand morphology (same regime as Batch 1 dt sweep).
Morphology: dt≤0.5 polka-dot; dt≥0.6 visibly noisier/streaming; dt≥0.85 fully diffuse.
Verdict: confirms Batch 1; keep dt=0.5.

### Batch 2 Sweep 13 — vmax ∈ [0.03,0.30]  [INCONCLUSIVE — speed cap saturated]
Response: noisy, **best at vmax=0.06 (0.547, inner=0.404)**; modest degradation above 0.10.
Morphology: similar across range, high-vmax diffuser.
Verdict: vmax=0.06 stays in parent. Not a discriminating param.

### Batch 2 Sweep 14 — cell.n ∈ [400,2500]  [INCONCLUSIVE — biology default still good]
Response: roughly flat plateau, **best at n=700 (0.454, inner=0.434)**; n=400 too sparse (loss=1.87); plateau above n=900.
Morphology: low-n sparse few mounds; high-n uniformly noisier.
Verdict: keep n=767 (biology default within the plateau).

### Batch 2 Sweep 15 — random_walk.strength ∈ [0,0.13]  [INCONCLUSIVE — saturated]
Response: noisy near-flat, best at **strength=0.1 (0.511, inner=0.427)**.
Morphology: identical across range.
Verdict: saturated; keep at 0.003–0.05.

---

### Batch 2 — Summary
**Global best (single-param-change-from-parent) by loss:** `inflow.rate=0.0` → loss=**0.312**, inner_mass=**0.561** ≈ REAL 0.606. n_final stays at 767 (REAL: 1413) — i.e. the loss is satisfied by NOT receiving fresh cells.

**Other notable wins:** spring.r_on=0.28 (0.328, 0.549); secrete.rate=15 (0.317, 0.453); sense.gain=10 (0.321, 0.456); spring.kadh=550 (0.330, 0.466); relay.thr=0.15 (0.340, 0.506); camp.diffusion=0.001 (0.375, 0.451).

**New parent (`specs/dicty_loop_base.yaml`):** apply the single best change → `inflow.rate=0.0`. All other parameters kept at Batch-2 parent so the parent ≡ actually-tested-best config (loss=0.312). Sense.gain, kadh, r_on, thr individual wins are NOT stacked into the parent (untested in combination).

**Falsified hypotheses this batch:**
- H1 (relay.gain monotone up to saturation) → has interior optimum near 150–250; high gain degrades.
- H5 (inflow optimum 0.5–0.8 persists at relay-on parent) → optimum is 0; relay-on changes the inflow regime fundamentally.
- H7 (high camp.diffusion → fewer giant mounds) → low diffusion is better in relay regime.
- H10 partial (sense.gain saturated plateau) → low-gain (10) optimum exists with relay on.

**Supported / established:**
- H2 (spring.r_on remains dominant under relay-on) — strongly supported, 2nd batch confirmation.
- H3 (relay.thr interior optimum at 0.10–0.18) — supported.
- H6 (kadh ceiling shifted up to ~550 with relay-on) — supported.

**Key insight (the central question for Batch 3):** The model achieves low loss at inflow=0 by *avoiding* the biological influx that would scatter fresh cells. REAL grows 767→1413; SIM stays at 767. The model has **no mechanism to retain fresh-influx cells in existing aggregates**. This is the new central gap. Batch 3 will (a) ADD a biased-inflow mechanism (cells enter at positions weighted by cAMP concentration, so newcomers join existing mounds) and (b) sweep its strength with strength=0 ablation = current uniform inflow. Hypothesis: biased inflow at moderate rate can match REAL n-growth WITHOUT destroying inner_mass.

---

## Batch 3 — Results (256 sims, parent = Batch-2 best: inflow.rate=0.0, relay-on)

**Methodology fault discovered up-front:** `eval_sweeps.py` ignored the `"fixed": {...}` key in
sweep_plan.json (line 83: `run_one(base, {param: v}, ...)`). Sweep 0 (`bias_to_camp` at
`fixed inflow.rate=1.5`) and sweep 1 (`inflow.rate` at `fixed bias_to_camp=5.0`) were therefore
*single-param overrides only* — the parent's `inflow.rate=0` carried through. The new
`bias_to_camp` mechanism never fired (no cells entered). H1/H2 verdicts here are **VOID
(experimental setup error)**; eval_sweeps.py patched (commit in this batch) to honor `"fixed"`
for Batch 4 retest.

### Batch 3 Sweep 0 — inflow.bias_to_camp ∈ [0,16]  [VOID — design fault]
Hypothesis: H1 — biased inflow rescues n-growth without destroying inner_mass at inflow.rate=1.5.
Response: dead flat — loss=0.3121 and inner_mass=0.561 at every value (= the parent baseline).
Morphology (from strip): 16 identical panels — a single diffuse central blob, less compact than REAL.
Verdict: VOID. `eval_sweeps.py` did not apply the `fixed` override → `inflow.rate` stayed at
parent value (0), so `bias_to_camp` had no inflowing cells to bias. Code patched. Retest in Batch 4.
Knowledge update: none about the mechanism; lesson recorded about the eval harness contract.

### Batch 3 Sweep 1 — inflow.rate ∈ [0,7]  [SUPPORTED — confirms inflow=0 best at this parent]
Hypothesis: H2 — at bias_to_camp=5, interior rate optimum exists.
Response: monotone-down inner_mass (0.561→0.20), monotone-up loss (0.31→1.6) as rate rises;
**best at rate=0 (loss=0.312, inner=0.561)**. Same as Batch 2 — but `fixed bias=5` was also
ignored, so this is really a pure inflow.rate sweep with uniform inflow.
Morphology (from strip): blob progressively diffuses and loses central compactness as rate
increases; by rate=7 the cloud is uniformly scattered with no aggregation.
Verdict: SUPPORTED for "rate>0 with uniform inflow degrades inner_mass". H2 (rate optimum with
biased inflow) NOT actually tested — bias override didn't apply. Retest in Batch 4 with patched harness.

### Batch 3 Sweep 2 — spring.r_on ∈ [0.20,0.66]  [SUPPORTED — third-batch confirmation]
Hypothesis: H3 — r_on remains dominant; inner_mass=REAL crossover near 0.28–0.31.
Response: monotone-up inner_mass (0.41→0.94), crosses REAL=0.61 at r_on≈0.28; loss U-shaped
with minimum at r_on=0.26 (loss=0.365, inner=0.557), exploding above r_on=0.34 from
over-compaction. **Best loss=0.365 at r_on=0.26.**
Morphology (from strip): diffuse cloud at r_on=0.20 → tighter central core at 0.26–0.30 →
single tiny over-compacted dot at r_on≥0.40 (overshoots REAL).
Verdict: SUPPORTED (third batch). r_on is the dominant morphology knob; inner_mass=REAL
crossover at 0.28 is reproducible. Note: at the new (inflow=0) parent, the loss-minimum r_on=0.26
remains BELOW the inner=REAL crossover — radial profile is matched with a slightly under-compacted blob.

### Batch 3 Sweep 3 — relay.gain ∈ [0,550]  [PARTIALLY FALSIFIED — H4 narrow interior optimum]
Hypothesis: H4 — interior optimum near 200 (Batch 2 claim).
Response: noisy/multimodal. **Best at gain=120 (parent, loss=0.312)**; secondary dip at gain=270
(loss=0.373). Earlier-batch claim of "interior optimum near 200" not confirmed at new parent —
optimum migrated back to 120. At gain=0 (ablation): loss=1.6, inner=0.26 — strongest evidence
yet that relay is necessary (~5× loss without it).
Morphology (from strip): gain=0 sparse no-aggregation, gain≥30 forms diffuse cloud with central
core; higher gains do not visibly improve compactness.
Verdict: relay NECESSITY strongly supported (3rd batch). H4 "interior optimum 200" FALSIFIED at
new parent — optimum is gain=120 (parent already there).

### Batch 3 Sweep 4 — relay.thr ∈ [0.04,0.40]  [SUPPORTED — parent thr=0.10 optimum]
Hypothesis: H5 — refine around 0.15.
Response: **best at thr=0.10 (parent, loss=0.312)**; secondary at 0.16 (loss=0.376). Interesting
inner_mass peak at thr=0.26 (inner=0.645) but loss=0.679 — over-compaction without matching profile.
Morphology (from strip): low thr (0.04) diffuse, parent (0.10) compact core, mid-range similar,
high thr (0.26-0.40) discrete satellite knots appear (multi-spot pattern).
Verdict: SUPPORTED for thr≈0.10. Multi-spot regime at high thr is morphologically interesting
but radially mismatched — a candidate to revisit with a richer position metric.

### Batch 3 Sweep 5 — spring.kadh ∈ [120,900]  [SUPPORTED — parent kadh=120 best loss]
Hypothesis: H6 — refine 200–700.
Response: **best loss at kadh=120 (parent, loss=0.312)**. Higher kadh raises inner_mass
(reaches 0.94 at kadh=900) but loss explodes (6.4) — over-compaction.
Morphology (from strip): increasing kadh shrinks the blob; by kadh=830–900 the entire cloud
collapses to a single tiny over-dense spot.
Verdict: parent kadh=120 is the loss optimum at relay-on, inflow=0. Earlier ceiling-shift claim
(Batch 2 kadh=550) FALSIFIED at new parent.

### Batch 3 Sweep 6 — sense.gain ∈ [2,120]  [BEAT PARENT — interior optimum gain=22]
Hypothesis: H7 — low-end refine, true optimum below 10.
Response: noisy with **clear interior optimum: gain=22 → loss=0.251, inner=0.537** (best in
batch outside camp.diffusion). Other dips at gain=15 (loss=0.322), gain=85 (0.339), gain=120 (0.394).
gain=12 spikes inner_mass=0.803 but loss=1.18 (over-attraction).
Morphology (from strip): tiny over-compacted knot at gain=2, diffuse cloud with central core in
mid range; high gain (≥40) increasingly amorphous diffuse cloud.
Verdict: FALSIFIES H7 ("true optimum below 10") — optimum migrated UP to 22 at inflow=0 parent.
Hypothesis "optimum is regime-dependent" emerging. Promote gain=22 candidate.

### Batch 3 Sweep 7 — spring.mu_f ∈ [0.03,0.22]  [BEAT PARENT — interior optimum at 0.12]
Hypothesis: H8 — refine around 0.10.
Response: noisy interior optimum, **best at mu_f=0.12 (loss=0.286, inner=0.531)**; secondary at
0.06 (0.306). Parent mu_f=0.05 → loss=0.312.
Morphology (from strip): low mu_f sparse diffuse; mid range standard compact core; mu_f≥0.13
develops slightly tighter central knot.
Verdict: SUPPORTED at mu_f≈0.12, 2.4× parent — modest improvement.

### Batch 3 Sweep 8 — camp.diffusion ∈ [0.0005,0.06]  [BEAT PARENT — best in batch, diff=0.0008]
Hypothesis: H9 — low-end refine (Batch 2 best at boundary 0.001).
Response: **best loss at diff=0.0008 → loss=0.239, inner=0.510** (single-batch global best). Loss
gradually rises with diffusion; plateau-with-noise above 0.003. Parent diff=0.02 → loss=0.312.
Morphology (from strip): tightest central core at low diff (0.0008); progressively flatter and
more spread at high diff (≥0.012).
Verdict: SUPPORTED, third batch trend. Low diffusion lets the relay manufacture its own spatial
structure rather than smearing it. **Move parent to diff=0.0008 for Batch 4.**

### Batch 3 Sweep 9 — secrete.rate ∈ [4,38]  [SUPPORTED parent — interesting inner-mass plateau]
Hypothesis: H10 — narrow band around 15.
Response: **best loss at rate=8 (parent, loss=0.312)**; secondary at rate=20 (0.366). inner_mass
peaks at rate=30 (0.725) but loss=1.72 — over-secretion collapses cells.
Morphology (from strip): rate=4 fully scattered no aggregation; rate=8–15 compact blob; rate≥25
collapses to multiple tiny over-compacted spots.
Verdict: parent secrete.rate=8 is the loss optimum at inflow=0. Earlier "secrete=15 best" claim
from Batch 2 FALSIFIED at new parent.

### Batch 3 Sweep 10 — relay.eps ∈ [0.005,0.40]  [BEAT PARENT — interior optimum eps=0.07]
Hypothesis: H11 — refine non-critical band.
Response: noisy with **best at eps=0.07 → loss=0.274, inner=0.532**; secondary at 0.13 (0.322).
Parent eps=0.02 → loss=0.312.
Morphology (from strip): morphologically very similar across range — eps tunes relay refractory
period but doesn't dramatically change final density shape.
Verdict: SUPPORTED, modest improvement. Move parent to eps=0.07 candidate.

### Batch 3 Sweep 11 — camp.decay ∈ [0.015,1.2]  [SUPPORTED — parent decay=0.20 optimum]
Hypothesis: H12 — confirm flat plateau.
Response: **best loss at decay=0.20 (parent, loss=0.312)**; near-flat region 0.06–0.30, sharp
degradation above 0.5.
Morphology (from strip): decay=0.015 saturated diffuse; mid-range compact core; decay≥0.5 loses
relay coherence.
Verdict: parent decay=0.20 confirmed; flat plateau 0.06–0.30.

### Batch 3 Sweep 12 — cell.n ∈ [400,2500]  [MARGINAL — n=850 slight win]
Hypothesis: H13 — confirm n=767 within plateau.
Response: noisy, **best at n=850 (loss=0.300, inner=0.485)**; parent n=767 → loss=0.312. Plateau
roughly 600–1900; n=400 too sparse (loss=0.613); n=500 also bad (0.897).
Morphology (from strip): blob size scales with n but core morphology unchanged across plateau.
Verdict: keep biology default n=767; n=850 wins by noise. Plateau confirms n not critical for morphology.

### Batch 3 Sweep 13 — random_walk.strength ∈ [0,0.13]  [BEAT PARENT — strength=0.018]
Hypothesis: H14 — confirm saturated noise floor.
Response: **best at strength=0.018 (loss=0.298, inner=0.523)**; parent strength=0.003 → loss=0.312.
Plateau 0.018–0.13 with similar losses; strength=0 ablation → loss=0.736 (no diffusion of cells
into aggregate cores — needed minimum noise).
Morphology (from strip): strength=0 sharp peak (no escape from initial positions); modest noise
gives coherent blob; high noise blurs but stays bounded.
Verdict: random_walk NECESSARY (ablation degrades); 0.018 modest improvement over 0.003.

### Batch 3 Sweep 14 — spring.k_rep ∈ [30,200]  [BEAT PARENT — k_rep=40]
Hypothesis: H15 — refine around 75.
Response: noisy, **best at k_rep=40 (loss=0.265, inner=0.468)**; parent k_rep=60 → loss=0.312.
Roughly flat plateau 40–200.
Morphology (from strip): k_rep=40 slightly tighter core; rest very similar.
Verdict: k_rep slightly prefers lower (40) than parent (60). Modest improvement.

### Batch 3 Sweep 15 — vmax ∈ [0.025,0.12]  [INCONCLUSIVE — extreme parent-only valley]
Hypothesis: H16 — confirm 0.06 optimum.
Response: **best loss at vmax=0.06 (parent, loss=0.312)**; surprisingly sharp — almost every
other vmax has loss≥0.5 (and most ≥1.0). Inner_mass at non-parent values is also low (0.2–0.3).
Morphology (from strip): only parent vmax=0.06 shows coherent central core; all other values
produce sparse dispersed clouds.
Verdict: parent vmax=0.06 confirmed; sharp valley suggests vmax is coupled to dt/relay timescale.
Worth a finer scan in Batch 4 (e.g. 0.04–0.08 in fine steps) to test brittleness vs seed dependence.

---

### Batch 3 — Summary
**Global best (single-param-change-from-parent) by loss:** `camp.diffusion=0.0008` →
loss=**0.239**, inner_mass=**0.510**, n_final=767 (REAL ≈0.61, 1413). Three-batch trend of
"low diffusion preferred under relay-on" promoted to Established Principle.

**Other notable wins (all single-param-from-parent):** sense.gain=22 (0.251, 0.537); k_rep=40
(0.265, 0.468); relay.eps=0.07 (0.274, 0.532); mu_f=0.12 (0.286, 0.531); random_walk=0.018
(0.298, 0.523); cell.n=850 (0.300, 0.485). All within ~25% of parent loss — the surface is now
quite flat: parameter optimization is plateauing while the central N-growth gap remains open.

**Critical setup fault:** the H1/H2 biased-inflow test was VOID — `eval_sweeps.py` did not honor
`"fixed"` overrides. Patched (overrides applied before swept param). Retest H1/H2 in Batch 4.

**New parent (`specs/dicty_loop_base.yaml`):** single best change → `camp.diffusion=0.0008`.
Other independent wins (sense.gain, eps, mu_f, k_rep, random_walk, n) NOT stacked (untested in
combination); will be retested at the new parent.

**Falsified hypotheses this batch:**
- H4 (relay.gain interior optimum near 200) — gain=120 (parent) best at new parent.
- H6 (kadh ceiling 200–700) — kadh=120 (parent) best; higher kadh raises inner_mass but
  destroys radial-profile match via over-compaction.
- H7 (sense.gain optimum below 10) — optimum migrated UP to 22 at inflow=0 parent.
- H10 (secrete.rate optimum at 15) — parent rate=8 best at new parent.

**Supported / established:**
- Relay NECESSITY confirmed third batch (gain=0 ablation: loss=1.6 vs parent 0.31).
- r_on dominance for morphology (third batch).
- random_walk NECESSITY (strength=0 ablation: loss=0.736).
- Low camp.diffusion (≤0.005) preferred in relay regime (third batch) — **promote to Established**.

**Key insight (Batch 4 frontier):** the model's parameter surface is now flat (~0.24–0.31 loss
across many one-param wins) while the n-growth gap (767 vs 1413) remains open. The biased-inflow
mechanism is added in code but UNTESTED. Batch 4 priorities:
  1. RETEST H1 (bias_to_camp at fixed rate=1.5) with the patched harness — the central scientific
     question of this whole arc.
  2. ALSO test bias_to_camp at fixed rate=3.0 (the larger gap from REAL) to test sufficiency.
  3. Continue narrow refinement at new parent for the wins above.

---

## Batch 4 — 16 single-param sweeps at parent (camp.diffusion=0.0008, inflow.rate=0); harness patched.

### Batch 4 Sweep 0 — inflow.bias_to_camp @ fixed rate=1.5 [FALSIFIED H1]
Hypothesis: H1 CENTRAL — biased inflow rescues n-growth at rate=1.5 without destroying inner_mass.
Response: noisy; loss 0.27–0.61, inner_mass 0.40–0.55; best loss=0.275 at bias=13; gentle upward
inner_mass trend with bias but never reaches REAL=0.61.
Morphology (from strip): small irregular blob at all bias values, visually indistinguishable
across the sweep; NOT matching REAL discrete-mound pattern.
Verdict: FALSIFIED — biased inflow at rate=1.5 cannot recover inner_mass≈REAL; best non-parent
loss (0.275) is WORSE than parent (0.239); morphology unchanged.
Knowledge update: H1 FALSIFIED. Cross-references Established Principle 6 (loss prefers rate=0).

### Batch 4 Sweep 1 — inflow.bias_to_camp @ fixed rate=3.0 [FALSIFIED H2]
Hypothesis: H2 SUFFICIENCY — biased inflow scales to rate=3.0 (closer to REAL +2.7/frame).
Response: bias=0 ablation catastrophic (loss=1.28); best bias=0.25 only loss=0.341, much worse
than parent 0.239; inner_mass plateau 0.4–0.5, never reaching REAL.
Morphology (from strip): single irregular blob across all bias values; rate=3.0 noticeably
sparser/more-dispersed than parent rate=0 — adding cells washes out the aggregate.
Verdict: FALSIFIED — biased inflow at rate=3.0 cannot match parent loss.
Knowledge update: H2 FALSIFIED. Biased-inflow mechanism does not scale to biological influx rate.

### Batch 4 Sweep 2 — inflow.rate @ fixed bias_to_camp=5 [FALSIFIED H3]
Hypothesis: H3 — with strong bias=5, interior rate optimum emerges matching BOTH metrics.
Response: best loss=0.239 at rate=0 (= parent ablation); all rate>0 give loss 0.31–0.59; inner
slowly declines; no interior optimum.
Morphology (from strip): rate=0 tight blob; rate≥0.4 progressively dilutes the core.
Verdict: FALSIFIED — even with strong bias, rate=0 wins.
Knowledge update: H3 FALSIFIED. CENTRAL SCIENTIFIC QUESTION (biased inflow as n-growth rescue) is
now ANSWERED NEGATIVELY across three sweeps.

### Batch 4 Sweep 3 — spring.r_on ∈ [0.20, 0.38] [SUPPORTED H4 with refinement]
Hypothesis: H4 — r_on remains dominant morphology knob; inner_mass=REAL crossover near 0.28.
Response: inner_mass MONOTONIC 0.40 → 0.92 across range; crossover ≈REAL=0.61 at r_on=0.24
(inner=0.611). Loss U-shape: best loss=0.239 at r_on=0.22 (parent); r_on=0.24 gives loss=0.445
with inner≈REAL. Above 0.27 loss climbs sharply (>0.6) — over-compaction.
Morphology (from strip): blob progressively tighter and more central as r_on rises; r_on≥0.30 is
a tight single dot; r_on=0.24 is the morphologically closest to a few-spot pattern.
Verdict: SUPPORTED — r_on still the dominant morphology lever (third batch). Crossover with
REAL inner-mass at 0.24 here (was 0.28 in Batch 3) — *crossover regime-dependent on diffusion*.
Knowledge update: Established Principle 3 refined — crossover band 0.24–0.28 depending on regime.

### Batch 4 Sweep 4 — sense.gain ∈ [8, 50] [SUPPORTED H5; best=24]
Hypothesis: H5 — narrow refine around Batch-3 best 22 at new parent.
Response: best loss=0.255 at gain=24 (parent=40 gives 0.531); low gain (8) catastrophically
over-attracts (loss=2.51, inner=0.871 = tight knot); high gain (>30) diffuses.
Morphology (from strip): gain=8 tiny over-compacted dot; gain=16–26 standard compact blob;
gain≥33 amorphous diffuse cloud.
Verdict: SUPPORTED — sense.gain≈22–24 reaffirmed across regimes. Modest improvement.
Knowledge update: confirm Batch-3 sense.gain≈22 win; nudge parent to gain=24.

### Batch 4 Sweep 5 — relay.eps ∈ [0.02, 0.20] [PARENT WINS; H6 FALSIFIED]
Hypothesis: H6 — refine around Batch-3 interior optimum 0.07.
Response: extremely flat. Best loss=0.239 at eps=0.02 (parent); Batch-3 win at eps=0.07 gives
loss=0.403 — was noise.
Morphology (from strip): morphology essentially invariant across full sweep.
Verdict: FALSIFIED — Batch-3 eps=0.07 win RETRACTED. eps=0.02 (parent) is optimum.
Knowledge update: drop "relay.eps interior optimum at 0.07" from Open Questions.

### Batch 4 Sweep 6 — spring.mu_f ∈ [0.06, 0.22] [marginal; mu_f=0.09]
Hypothesis: H7 — refine around Batch-3 best 0.12.
Response: noisy. Best loss=0.284 at mu_f=0.09; mu_f=0.12 gives 0.330. Inner_mass jumps to
0.74–0.80 at mu_f≥0.15 with loss spikes (1.20, 1.44) — over-compaction.
Morphology (from strip): low-mid mu_f standard blob; mu_f≥0.17 tight over-compacted knot.
Verdict: SUPPORTED-with-revision — mu_f optimum migrates each refinement (0.105→0.12→0.09);
surface noisy; consistent message: 0.07–0.12 OK, ≥0.15 catastrophic.
Knowledge update: retract precise Batch-3 mu_f=0.12 claim; plateau not sharp peak.

### Batch 4 Sweep 7 — camp.diffusion ∈ [0.0001, 0.005] [PARENT WINS, H8 SUPPORTED]
Hypothesis: H8 — extend low end below parent 0.0008.
Response: best loss=0.239 at diff=0.0008 (parent); plateau 0.27–0.35; no win below 0.0008.
Morphology (from strip): minor variations; compact blob throughout.
Verdict: SUPPORTED — diff=0.0008 confirmed; no benefit going lower. Established Principle 5 holds.

### Batch 4 Sweep 8 — relay.gain ∈ [60, 300] [PARENT WINS, H9 SUPPORTED]
Hypothesis: H9 — confirm gain=120.
Response: best loss=0.239 at gain=120 (parent); plateau 0.28–0.49; gain=110 spike 0.76 (bad seed).
Morphology (from strip): similar throughout; gain=60–110 slightly sparser.
Verdict: SUPPORTED — parent gain=120 confirmed.

### Batch 4 Sweep 9 — relay.thr ∈ [0.06, 0.30] [PARENT WINS, H10 SUPPORTED w/ caveat]
Hypothesis: H10 — confirm thr=0.10; check multi-spot at thr=0.26.
Response: best loss=0.239 at thr=0.10 (parent); high-thr losses degrade (0.77 at 0.24, 0.82 at
0.26); inner_mass spikes at thr=0.20 (0.674) and thr=0.26 (0.682) but radial profile collapses.
Morphology (from strip): low/mid thr standard blob; thr≥0.20 multi-knot pattern.
Verdict: SUPPORTED — parent thr=0.10 loss optimum; multi-knot regime morphologically interesting
but not a loss win.
Knowledge update: keep "multi-knot at high thr" in Open Questions.

### Batch 4 Sweep 10 — spring.k_rep ∈ [25, 90] [PARENT WINS, H11 FALSIFIED]
Hypothesis: H11 — narrow refine around Batch-3 win at k_rep=40.
Response: best loss=0.239 at k_rep=60 (parent); k_rep=40 here gives loss=0.494 — Batch-3 was
NOISE; broad plateau 0.28–0.50.
Morphology (from strip): morphology similar across range.
Verdict: FALSIFIED — Batch-3 k_rep=40 win was a single-seed artifact.
Knowledge update: retract k_rep=40 candidate; surface flat-noisy.

### Batch 4 Sweep 11 — random_walk.strength ∈ [0.005, 0.05] [PARENT-EXCLUDED; best=0.005]
Hypothesis: H12 — refine around Batch-3 win at 0.018.
Response: best loss=0.307 at strength=0.005; strength=0.018 gives 0.336 — Batch-3 win NOT
reproduced; noisy plateau; strength=0.012 spike 0.67.
Morphology (from strip): similar across range; high strength=0.04 blurs.
Verdict: FALSIFIED Batch-3 win at 0.018 — noise. Parent strength=0.003 likely still good.
Knowledge update: retract random_walk=0.018 candidate; surface flat with noise.

### Batch 4 Sweep 12 — vmax ∈ [0.045, 0.075] [PARENT-WINS but BRITTLE; H13 SUPPORTED]
Hypothesis: H13 — fine valley scan to distinguish dt-coupling vs seed dependence.
Response: best loss=0.239 at vmax=0.060 (parent); vmax=0.061 → 0.327; vmax=0.072 → 0.309
(anomalous second valley). MOST other values (0.045–0.058, 0.063–0.069, 0.075) have loss
0.93–1.83 — catastrophic. Inner_mass mostly 0.20–0.35 outside valleys.
Morphology (from strip): only vmax=0.060–0.062 and vmax=0.072 show coherent central blob; others
sparse dispersed with displaced fragments.
Verdict: SUPPORTED — vmax=0.060 is SHARP local minimum, not smooth valley; second valley at
0.072. Suggests dt×vmax aliasing.
Knowledge update: promote vmax-dt brittleness from Open Question to **needs joint sweep**.

### Batch 4 Sweep 13 — secrete.rate ∈ [5, 20] [PARENT WINS, H14 SUPPORTED]
Hypothesis: H14 — narrow refine around parent 8.
Response: best loss=0.239 at rate=8 (parent); symmetric valley; rate=5 catastrophic (loss=1.73,
inner=0.216 — under-secretion → no aggregation); rate=20 OK (0.42); secondary mins rate=13–15.
Morphology (from strip): rate=5 sparse no aggregation; rate=7–9 standard blob; rate=10–20
multiple tighter spots (over-secretion).
Verdict: SUPPORTED — parent rate=8 confirmed.

### Batch 4 Sweep 14 — camp.decay ∈ [0.06, 0.40] [PARENT WINS, H15 SUPPORTED]
Hypothesis: H15 — confirm plateau around parent 0.20.
Response: best loss=0.239 at decay=0.20 (parent); plateau 0.10–0.30 (0.31–0.61); higher decay
(0.33+) loses relay coherence (0.59–0.70).
Morphology (from strip): low-mid decay similar; high decay sparser.
Verdict: SUPPORTED — parent decay=0.20 confirmed.

### Batch 4 Sweep 15 — spring.kadh ∈ [40, 300] [PARENT WINS, H16 SUPPORTED]
Hypothesis: H16 — extend below parent 120.
Response: best loss=0.239 at kadh=120 (parent); kadh=40–100 in 0.42–0.75 (no benefit going lower);
kadh=300 inner=0.683 but loss=0.64 (over-compaction); kadh=200 anomalous 0.346.
Morphology (from strip): subtle shifts; very high kadh tighter cores.
Verdict: SUPPORTED — parent kadh=120 confirmed; no improvement either direction.

---

### Batch 4 — Summary
**Parent unchanged.** New batch best (single-param-change-from-parent) = **loss=0.239 at parent
itself**: 11 of 16 sweeps had parent value inside swept range and parent value won. Only sweep 4
(sense.gain=24, loss=0.255) gave a non-parent value that is arguably a win. All "Batch-3 candidate
wins" (k_rep=40, mu_f=0.12, eps=0.07, random_walk=0.018, n=850) RETRACTED — single-seed noise.
Parameter surface is flat-with-noise amplitude ~0.05–0.10 around parent loss 0.239.

**CENTRAL SCIENTIFIC QUESTION RESOLVED (negatively):** biased-inflow rescue mechanism (with or
without large rate) does NOT close the n-growth gap while preserving morphology under the current
model+metric. H1, H2, H3 all FALSIFIED. Radial-profile loss is fully dominated by inflow.rate=0
control. Promotes to Established Principle: "biased-inflow mechanism is INSUFFICIENT to satisfy
both inner_mass AND n-growth targets under the current loss." Cross-references Established
Principle 6 (loss is gameable by suppressing influx).

**Falsified this batch:** H1, H2, H3 (biased-inflow rescue); H6 (relay.eps interior optimum 0.07);
H11 (k_rep=40 candidate); H12 partial (random_walk=0.018 candidate).

**Supported / refined:** Established 3 (r_on dominance, third batch), Established 5 (low
camp.diffusion, fourth batch), Established 6 (loss gameable, repeatedly), H5 (sense.gain≈22–24).

**Key insight (Batch 5 frontier):** parameter surface exhausted at current model. Biased-inflow
FALSIFIED. n-growth-vs-morphology dichotomy is INTRINSIC to {AR + chemotaxis + influx-anywhere
+ adhesion + relay} architecture. Batch 5 must test NEW MECHANISMS:
  1. Boundary-source inflow (`inflow.edge_band`): cells appear only near periodic boundary, must
     stream inward. Tests streaming morphology and whether spatially-restricted inflow degrades
     loss less than uniform inflow.
  2. Pulsatile relay (`relay.omega`): sinusoidal forcing of FN activator, mimicking the ~6-min
     cAMP oscillation in real Dicty. Tests whether wavefront periodicity helps merging.
  3. Joint sweeps where single-param sweeps plateaued (vmax × dt; r_on at boundary-source inflow).

---

## Batch 5 — NEW MECHANISMS TESTED (boundary-source inflow + pulsatile relay)

Parent unchanged from Batch 4 (loss=0.239, inner_mass=0.510, n_final=767). Two new
mechanisms added in code; sweep_0/1 probe `inflow.edge_band`; sweep_2/3 probe
`relay.omega`. The remaining 12 sweeps refine parameters and confirm priors.

### Batch 5 Sweep 0 — inflow.edge_band in [0.025, 0.50] @ rate=2.0, bias=0  [H1 FALSIFIED]
Hypothesis: H1 — boundary-source inflow at rate=2.0 lets cells stream inward and
satisfy n-growth without destroying compactness.
Response: monotone-ish DECREASING loss in edge_band (0.42 at 0.50 -> 1.4 at 0.025);
best in-sweep at the widest band edge_band=0.50, loss=0.4155, inner=0.463. The
"ablation" (widest band) wins; spatial restriction actively HURTS. ALL values lose
to parent (loss=0.239 at rate=0).
Morphology (from strip): narrow bands give a centrally-spread cloud with extra
mass dispersed across the field — clearly not aggregation; wider bands look
more like the parent's compact blob but never as tight.
Verdict: FALSIFIED — spatially-restricting inflow does NOT rescue morphology.
Adding inflow at rate=2.0 with ANY edge_band degrades loss versus no-inflow.
Knowledge update: kill `edge_band` as a candidate mechanism.

### Batch 5 Sweep 1 — inflow.edge_band in [0.025, 0.50] @ rate=2.0, bias=5  [H2 FALSIFIED]
Hypothesis: H2 — boundary-source + cAMP bias is complementary; the two structural
priors together fix the n-growth-vs-morphology gap.
Response: noisy plateau across edge_band with a weak preference for mid-wide
values (best=edge_band=0.44, loss=0.3833, inner=0.453). Adding bias=5 to
boundary-source is slightly better than bias=0 (sweep 0) but still much worse
than parent (0.239).
Morphology (from strip): biased+restricted inflow produces a similar smeared
cloud with multiple weak loci, never the discrete compact mounds of REAL.
Verdict: FALSIFIED — boundary-source + bias is not synergistic in a useful
direction; both mechanisms are individually loss-increasing.
Knowledge update: confirms Established #7 generalises — no inflow mechanism
(uniform, biased, boundary, biased+boundary) helps the radial-profile loss.

### Batch 5 Sweep 2 — relay.omega in [0, 0.30] @ amplitude=0.05  [H3 FALSIFIED]
Hypothesis: H3 — a global pacemaker drive (amplitude=0.05) entrains the relay
into more-coherent target waves and improves merging.
Response: omega=0 wins (parent loss=0.239); response non-monotone with
intermittent spikes (omega=0.03 -> 1.18; omega=0.10 -> 1.08) but no value beats
the ablation. Modest second valley at omega=0.17 (loss=0.28), still worse.
Morphology (from strip): morphologies mostly indistinguishable single blobs
across omega; loss spikes correspond to displaced/fragmented blobs.
Verdict: FALSIFIED — low-amplitude pacemaker forcing is at best neutral, at
worst destabilising. Pure-FN ablation wins.

### Batch 5 Sweep 3 — relay.omega in [0, 0.30] @ amplitude=0.20  [H4 FALSIFIED]
Hypothesis: H4 — stronger pacemaker (amplitude=0.20) enters a forced/entrained
regime where global pacing dominates self-organisation.
Response: omega=0 wins again (loss=0.239). Non-monotone elsewhere; omega=0.30
gives loss=0.267 (near-parent secondary minimum, inner_mass=0.54). No value
beats the ablation.
Morphology (from strip): high-amplitude forcing produces slightly more diffuse
blobs; no "entrained" multi-target-wave morphology visible.
Verdict: FALSIFIED at high amplitude — pacemaker drive does not improve final
density. Combined with sweep 2: pulsatile relay BROADLY FALSIFIED.

### Batch 5 Sweep 4 — inflow.rate in [0, 7.0] @ edge_band=0.10, bias=0  [H5 FALSIFIED]
Hypothesis: H5 — boundary-source inflow at edge_band=0.10 admits an interior
rate optimum.
Response: monotone-INCREASING loss in rate (0.24 at rate=0 -> 1.4 at rate=3.8;
recovers slightly at rate=7.0 -> 0.90). Best=rate=0.
Morphology (from strip): increasing rate produces increasingly diffuse central
clouds with sparse outer fragments; no streaming morphology.
Verdict: FALSIFIED — no interior optimum; boundary-source inflow at any rate>0
degrades loss.

### Batch 5 Sweep 5 — spring.r_on in [0.20, 0.40] @ rate=2.0, edge_band=0.10  [H6 SUPPORTED-with-cost]
Hypothesis: H6 — r_on coalescence still works under boundary inflow.
Response: monotone-INCREASING inner_mass in r_on (0.27 at 0.20 -> 0.84 at 0.40);
loss minimum at r_on=0.30 (0.4176, inner=0.54). With inflow ON, the r_on lever
still raises inner_mass — but loss never beats parent (0.239).
Morphology (from strip): low r_on -> spread cloud; r_on>0.30 -> progressively
tighter central blob; r_on=0.40 -> over-tight knot.
Verdict: SUPPORTED on mechanism (r_on coalescence active under inflow) but
JOINT config still loses to parent overall.

### Batch 5 Sweep 6 — spring.r_on in [0.205, 0.285] FINE  [H7 SUPPORTED]
Hypothesis: H7 — characterise the inner_mass=REAL crossover precisely.
Response: monotone-INCREASING inner_mass; crossover with REAL (inner=0.61)
hits exactly at r_on=0.24 (inner=0.611); rises smoothly to 0.734 at r_on=0.285.
Loss min at parent r_on=0.22 (0.239); r_on=0.245 (loss=0.314) is a near-parent
secondary minimum with inner=0.585.
Morphology (from strip): smooth thinning of cloud + concentration into centre
as r_on rises; no fragmentation into multiple mounds.
Verdict: SUPPORTED — inner_mass=REAL achievable at r_on=0.24 but at +30% loss
because the radial profile then over-concentrates relative to REAL (multi-
mound spread). The metric correctly disfavours over-compact single blobs.
Knowledge update: pins crossover at r_on=0.24 at the current parent. Reframes
the central problem: inner_mass=REAL with single-blob morphology is
INSUFFICIENT — the radial profile of REAL encodes multi-mound spread.

### Batch 5 Sweep 7 — sense.gain in [16, 36]  [PARENT(40) OUT-OF-RANGE; H8 FALSIFIED]
Hypothesis: H8 — sense.gain=24 (Batch 4 candidate) beats parent (40).
Response: in-range best at sense.gain=24 (loss=0.2554, inner=0.546). But parent
40 (out of range) still wins overall at loss=0.239. Sense.gain=18-20 produces
a striking inner_mass spike (0.69-0.74) — visually closer to REAL morphology
— but with much worse loss (0.39-0.85), confirming radial-profile mismatch.
Morphology (from strip): sense.gain=18-20 tight central mound (high inner_mass)
but at cost of leaving outer cells dispersed; mid-high gains look like parent.
Verdict: FALSIFIED — Batch-4 sense.gain=24 candidate was noise; parent
sense.gain=40 still best.
Knowledge update: retract sense.gain=24 candidate.

### Batch 5 Sweep 8 — dt in [0.30, 0.80] @ vmax=0.06  [H9 SUPPORTED]
Hypothesis: H9 — varying dt at fixed vmax mirrors the vmax x dt aliasing valley.
Response: best dt=0.50 (parent), loss=0.239. Most other dt values catastrophic
(0.95-1.69), but secondary valleys at dt=0.30 (0.60), dt=0.62 (0.63), dt=0.72
(0.51). Sharp resonance — aliasing-driven.
Morphology (from strip): only dt=0.30/0.50/0.62/0.72 show coherent central
blobs; catastrophic values produce sparse fragmented fields with no aggregation.
Verdict: SUPPORTED — vmax x dt aliasing confirmed; parent (dt=0.5, vmax=0.06)
sits at a sharp resonance with discrete "mirror" valleys at other dt values.
Knowledge update: PROMOTE dt x vmax aliasing from Open Question to ESTABLISHED.

### Batch 5 Sweep 9 — vmax in [0.058, 0.064] FINE  [H10 PARTIAL]
Hypothesis: H10 — the vmax=0.060 valley is sub-grid (resonance) vs a plateau.
Response: best vmax=0.060 (parent), loss=0.239. vmax=0.0605-0.0616 form a weak
shoulder (loss 0.33-0.39); 0.0598-0.0625 broadly tolerable; vmax>=0.063
catastrophic (1.8+). Narrow but not sub-grid — small working window
[0.0598, 0.062].
Morphology (from strip): coherent blobs in [0.0595, 0.0620]; sparse/fragmented
outside; tighter at high vmax.
Verdict: PARTIAL — narrow valley with ~10% width in vmax.
Knowledge update: vmax safe band approx [0.0598, 0.0620] at dt=0.50.

### Batch 5 Sweep 10 — spring.mu_f in [0.060, 0.160]  [PARENT(0.05) OUT-OF-RANGE; H11 FALSIFIED]
Hypothesis: H11 — interior optimum exists in 0.06-0.16 band.
Response: in-range best at mu_f=0.104 (loss=0.2646, inner=0.508). Parent
mu_f=0.05 (below range) still beats this. Noisy plateau 0.27-0.79.
Morphology (from strip): morphology drifts gently across mu_f — similar
blobs; no qualitative change.
Verdict: FALSIFIED — no in-range value beats parent; parent stays.

### Batch 5 Sweep 11 — spring.kadh in [80, 200]  [PARENT WINS, H12 SUPPORTED]
Hypothesis: H12 — narrow refine reveals interior optimum near parent.
Response: best kadh=120 (parent), loss=0.239. Plateau 0.30-0.75; non-monotone
noise.
Morphology (from strip): subtle thinning/tightening at high kadh; low kadh
slightly more diffuse.
Verdict: SUPPORTED — parent kadh=120 confirmed.

### Batch 5 Sweep 12 — inflow.rate in [0, 3.0] PLAIN  [H13 SUPPORTED]
Hypothesis: H13 — regression test: rate=0 wins for plain inflow.
Response: monotone-INCREASING loss; best=rate=0 (parent, 0.239); rate=3.0 ->
1.28. Reconfirms Established #6 at new-parent regime.
Morphology (from strip): rate=0 compact; high rate diffuse cloud with extra
peripheral mass.
Verdict: SUPPORTED — Established #6 reaffirmed (fifth confirmation).

### Batch 5 Sweep 13 — camp.diffusion in [0.0002, 0.0020] NARROW  [PARENT WINS, H14 SUPPORTED]
Hypothesis: H14 — narrow refine around parent 0.0008 reveals a sub-grid optimum.
Response: best diff=0.0008 (parent), loss=0.239. Smooth plateau 0.27-0.50 with
no second valley.
Morphology (from strip): morphology similar across the band — flat response.
Verdict: SUPPORTED — parent diff=0.0008 confirmed; no sub-grid optimum.

### Batch 5 Sweep 14 — relay.thr in [0.06, 0.16] NARROW  [PARENT WINS, H15 SUPPORTED]
Hypothesis: H15 — narrow refine around parent 0.10.
Response: best thr=0.10 (parent), loss=0.239. Noisy plateau 0.33-0.98.
Morphology (from strip): subtle variations; central blob throughout. The high-
thr multi-knot regime is NOT in this range (>= 0.18 from prior Open Q).
Verdict: SUPPORTED — parent thr=0.10 confirmed within narrow band.

### Batch 5 Sweep 15 — random_walk.strength in [0.001, 0.012] NARROW  [PARENT WINS, H16 SUPPORTED]
Hypothesis: H16 — narrow refine around parent 0.003.
Response: best strength=0.003 (parent), loss=0.239. Plateau 0.30-0.67; non-
monotone noise.
Morphology (from strip): similar across; high strength slightly more spread.
Verdict: SUPPORTED — parent strength=0.003 confirmed.

### Batch 5 — Summary
**Parent UNCHANGED for the third consecutive batch.** Best loss=0.239 = parent,
won in 8 of 16 sweeps. Best inner_mass match achieved: inner=0.611 at r_on=0.24
(sweep 6), but at loss=0.445 — radial-profile cost of over-compactness.

**BOTH new mechanisms FALSIFIED**:
  * `inflow.edge_band` — falsified in sweeps 0, 1, 4. Spatial restriction
    actively hurts; uniform-band ablation wins; rate=0 always best.
  * `relay.{omega, amplitude}` — falsified at both low (0.05) and high (0.20)
    amplitudes. Pure-FN ablation wins.

**Key promotions:**
  - **vmax x dt aliasing** promoted to ESTABLISHED (sweeps 8+9 + Batch 4 sw 12).
  - **All inflow mechanisms** now FALSIFIED — strengthens Established #7.
  - **Pulsatile relay** added to Falsified Hypotheses.

**Key morphological insight (Batch 6 frontier):** Model converges to a single
dominant central blob. REAL has multiple discrete mounds. Inner_mass=REAL is
reachable (sweep 6 r_on=0.24) but radial profile then over-concentrates because
the model can't produce multiple mounds. **Bottleneck is no longer parameter
tuning OR inflow OR pacemaker forcing — it is the single-attractor nature of
the relay field.**

**Batch 6 strategy:**
  1. NEW MECHANISM: `relay.{nucleate_rate, nucleate_amp}` — Poisson sprinkle of
     activator pulses at random grid points each frame, seeding multiple wave
     centres. Ablation = nucleate_rate=0 (= parent).
  2. Re-enter the high-relay.thr multi-knot regime (thr in [0.18, 0.32]) that
     Batch 4 showed produces multi-spot morphology with inner_mass>=0.65.
  3. Stop refining flat-noisy params (mu_f, kadh, rw, thr-narrow, diffusion).

---

# Batch 6 (256 sims = 16 sweeps x 16 values)

Parent unchanged from Batch 5: vmax=0.06, dt=0.5, camp.{res=160, diffusion=0.0008,
decay=0.20}, sense.gain=40, relay.{gain=120, thr=0.10, eps=0.02, omega=0, amplitude=0,
nucleate_rate=0, nucleate_amp=0.30}, spring.{k_rep=60, r0=0.022, kadh=120, r_on=0.22,
delta=0.001, mu_f=0.05}, secrete.rate=8, inflow.rate=0, random_walk.strength=0.003.
Parent loss=0.239, inner_mass=0.51, n_final=767. Batch 6 introduces ONE new mechanism:
`relay.{nucleate_rate, nucleate_amp}` -- Poisson sprinkle of activator pulses at random
grid points each frame. Most joint sweeps (5,6,7,8,10,14,15) keep nucleate_rate=10,
nucleate_amp=0.30 as the joint-knob baseline.

## Batch 6 Sweep 0 -- relay.nucleate_rate (amp=0.20)  [falsified]
Hypothesis (H1-B6): stochastic multi-centre seeding at moderate amplitude (0.20) seeds
multiple wave centres that survive into multi-mound morphology.
Response: flat/noisy; best value=0 (parent) loss=0.239 inner=0.510. All non-zero
values lose by 8-150%; best non-zero rate=25 loss=0.259.
Morphology (from strip): single noisy central blob at every value; no multi-spot
structure emerges. Stochastic pulses are absorbed/smoothed before they can recruit.
Verdict: FALSIFIED -- nucleation at amp=0.20 does NOT break the single-attractor.
Knowledge update: adds H1-B6 to Falsified.

## Batch 6 Sweep 1 -- relay.nucleate_rate (amp=0.50)  [falsified]
Hypothesis (H2-B6): stronger nucleation pulses (amp=0.50) outcompete the central
self-organised blob.
Response: flat-noisy; best value=0 loss=0.239. All non-zero values lose by 26-100%.
Morphology (from strip): single central blob across all values; mound looks slightly
larger/smeared at high rates (more spurious activator -> more diffuse aggregation),
but never splits into multiple discrete mounds.
Verdict: FALSIFIED -- increasing nucleate_amp does not rescue multi-mound either.
Knowledge update: adds H2-B6 to Falsified.

## Batch 6 Sweep 2 -- relay.nucleate_amp (rate=10)  [falsified]
Hypothesis (H3-B6): amplitude-response curve has interior optimum that breaks
single-attractor.
Response: flat-noisy with extreme outliers at amp=0.25 (loss=0.797) and amp=0.65
(loss=0.762); best value=0 (parent) loss=0.239.
Morphology (from strip): single central blob at all values; some high-amp values
make the blob more diffuse with noisy halo. No emergent multi-spot structure.
Verdict: FALSIFIED -- amp alone has no interior optimum.
Knowledge update: combined with sweeps 0,1,3 -- nucleation mechanism FALSIFIED.

## Batch 6 Sweep 3 -- relay.nucleate_rate (amp=0.05, extended range 0-200)  [falsified]
Hypothesis (H4-B6): noise-floor regime (very low amp, very high rate) approximates a
continuous bias term and might rescue multi-mound.
Response: flat-noisy across full range; best=0 loss=0.239.
Morphology (from strip): single central blob throughout; even at rate=200/amp=0.05
the cumulative activator is just smooth noise.
Verdict: FALSIFIED -- noise-floor regime also collapses to single attractor.
Knowledge update: nucleation mechanism FALSIFIED for ALL (rate, amp) combinations tested.

## Batch 6 Sweep 4 -- relay.thr (no nucleation, 0.16-0.34)  [INCONCLUSIVE -- KEY morphology hit]
Hypothesis (H5-B6): high relay.thr enters multi-knot regime (Batch 4 OQ) producing
multiple discrete mounds.
Response: highly non-monotone with hot/cold lines; best in-range value thr=0.17,
loss=0.318 (still worse than parent thr=0.10, loss=0.239). At thr=0.21 inner=0.804,
thr=0.25 inner=0.900, but loss=1.07 and 2.36 resp.
Morphology (from strip): MULTI-SPOT structure CLEARLY visible at thr=0.20, 0.21,
0.25, 0.26, 0.28, 0.32 -- 3-6 small distinct mounds, qualitatively closest to REAL
shown across the entire batch. The radial-profile loss penalises this because the
mounds are sparser per-mound than a single tight blob.
Verdict: INCONCLUSIVE -- multi-knot regime exists & confirms Batch-4 OQ, but the
configured loss metric scores it worse than single-blob parent. The model CAN
produce multi-mound; the loss CANNOT reward it.
Knowledge update: promotes "multi-knot regime exists" to Established. Strengthens
Open Question on loss metric inadequacy.

## Batch 6 Sweep 5 -- relay.thr (with nucleate_rate=10, amp=0.30)  [falsified]
Hypothesis (H6-B6): nucleation pushes multi-knot regime to lower loss configuration.
Response: best thr=0.17 loss=0.423 (worse than no-nucleation version of same sweep
@0.17->0.318; also worse than parent 0.239).
Morphology (from strip): multi-spot at mid-thr values (0.21-0.27) but blobs are more
diffuse/noisier than sweep 4; nucleation degrades crisp multi-knot pattern.
Verdict: FALSIFIED -- nucleation interacts NEGATIVELY with multi-knot regime.
Knowledge update: dispenses last hope of rescue via nucleation.

## Batch 6 Sweep 6 -- spring.r_on (with nucleate_rate=10, amp=0.30)  [supported (locally)]
Hypothesis (H7-B6): nucleation lets inner_mass reach REAL at lower r_on than 0.24.
Response: best in-range r_on=0.225 loss=0.283 inner=0.583 (vs parent's r_on=0.22
no-nucleation loss=0.239). Inner_mass climbs monotonically with r_on (consistent
with Established #3).
Morphology (from strip): single tight central blob at all r_on values >=0.22;
r_on=0.20 ablation gives uncompacted scatter. Nucleation does not break single-attractor
even when adhesion reach is increased.
Verdict: locally best at r_on=0.225 in this joint-with-nucleation regime, but the
joint regime itself loses to parent.
Knowledge update: r_on monotone effect re-confirmed (now 6th batch).

## Batch 6 Sweep 7 -- relay.gain (with nucleate_rate=10, amp=0.30)  [reconfirms relay necessary]
Hypothesis (H8-B6): re-test relay necessity when nucleation active; map if high gain
rescues.
Response: gain=0 ablation loss=1.27 (collapses, confirms relay NECESSARY); best
in-range gain=160 loss=0.282 inner=0.536; gain=120 (parent) under nucleation loss=0.438.
Morphology (from strip): gain=0 -> diffuse scatter, no aggregation; intermediate
gains -> single blob with noisy halo (from nucleation); high gains slightly
sharpen the central blob.
Verdict: relay re-confirmed NECESSARY (4th independent confirmation); nucleation
itself remains FALSIFIED as a rescue mechanism.
Knowledge update: strengthens Established #4 (relay necessary).

## Batch 6 Sweep 8 -- camp.decay (with nucleate_rate=10, amp=0.30)  [flat-noisy]
Hypothesis (H9-B6): higher decay shortens wave range and keeps multi-centres separated.
Response: noisy bowl; best decay=0.12 loss=0.295; under nucleation parent decay=0.20
loss=0.438 -- the parent's own decay value is hurt by nucleation. NOT lower than
parent loss=0.239 anywhere in sweep.
Morphology (from strip): single central blob at all values; high decay (0.4-0.5)
makes blob slightly smaller/sharper but still single.
Verdict: FALSIFIED -- decay x nucleation gives no improvement.
Knowledge update: camp.decay confirmed flat around parent (now 3rd confirmation).

## Batch 6 Sweep 9 -- vmax (alone, 0.04-0.08)  [supports Est #9]
Hypothesis (H10-B6): re-test dt x vmax aliasing on iso-product line.
Response: classic resonance valley at vmax=0.06 loss=0.239 = parent; off-resonance
losses are 0.5-1.8. inner_mass spikes to 0.51 at vmax=0.06.
Morphology (from strip): off-resonance values produce scattered/under-aggregated
configurations; vmax=0.06 alone has the clean single central blob; vmax=0.048 weak
secondary minimum.
Verdict: SUPPORTED Established #9 -- vmax=0.06 narrow valley reconfirmed (4th batch).
Knowledge update: Established #9 now 4-batch supported.

## Batch 6 Sweep 10 -- random_walk.strength (with nucleate_rate=10, amp=0.30)  [falsified]
Hypothesis (H11-B6): random walk + nucleation are additive noise sources.
Response: shallow noisy; best strength=0.004 loss=0.253 (still worse than parent 0.239,
which has rw=0.003 no nucleation).
Morphology (from strip): single central blob at all strengths up to 0.018; >=0.02
the blob diffuses noticeably. Random walk + nucleation are redundant noise, not
synergistic.
Verdict: FALSIFIED -- rw x nucleation gives no rescue.
Knowledge update: rw.strength stays at parent 0.003.

## Batch 6 Sweep 11 -- cell.n (alone, 400-1500)  [flat]
Hypothesis (H12-B6): cells near REAL frame-final count (~1400) help without inflow.
Response: noisy-flat; best n=767 (parent) loss=0.239; n=950 narrow secondary
loss=0.294; n=1400 loss=0.544.
Morphology (from strip): low n -> sparse mounds; high n (1300+) -> larger denser
single mound but radial profile too spread. parent n=767 cleanest.
Verdict: FALSIFIED -- initial seeding density confirmed at parent (3rd batch).
Knowledge update: n=767 confirmed Est.

## Batch 6 Sweep 12 -- camp.res (80-240)  [supports Est #9]
Hypothesis (H13-B6): grid resolution shifts the dt x vmax aliasing landscape.
Response: discrete resonance lines at res in {128, 136, 160, 192}; off-resonance
losses 1.0-1.6; on-resonance losses 0.24-0.35. best res=160 (parent) loss=0.239.
Morphology (from strip): off-resonance values scattered/under-aggregated; on-resonance
clean blob. STRONG support for resonance/aliasing interpretation of Est #9.
Verdict: SUPPORTED Est #9 -- confirms res-dependent resonance structure.
Knowledge update: Est #9 strengthened with cross-resolution evidence.

## Batch 6 Sweep 13 -- seed (0..15)  [FAILED]
Hypothesis (H14-B6): noise-floor measurement across seeds.
Response: ALL VALUES returned NaN -- sweep harness did not pass `seed` as a valid
spec key (root-level `seed` field probably needed different escape syntax).
Morphology (from strip): empty.
Verdict: TECHNICAL FAILURE -- re-run in Batch 7 with corrected sweep path.
Knowledge update: none.

## Batch 6 Sweep 14 -- secrete.rate (with nucleate_rate=10, amp=0.30, 2-24)  [flat under nucleation]
Hypothesis (H15-B6): secretion floor shifts under nucleation.
Response: monotone-rising loss below rate=6, then flat-noisy; best rate=10 loss=0.342
(under nucleation; parent at rate=8 no-nucleation = 0.239).
Morphology (from strip): low rates 2-5 fully scattered (no aggregation -- secretion
floor); rate>=6 single central blob, ranging from larger-diffuse to small-tight
as rate grows.
Verdict: FALSIFIED -- secrete.rate floor ~6 confirmed; no shift from nucleation.
Knowledge update: secrete.rate=8 parent re-confirmed.

## Batch 6 Sweep 15 -- relay.nucleate_rate (with spring.r_on=0.24, amp=0.30)  [falsified]
Hypothesis (H16-B6): nucleation rescues multi-mound at the r_on=0.24 inner_mass=REAL
crossover.
Response: noisy; best in-range rate=30 loss=0.290 inner=0.497 (vs r_on=0.24 alone
loss=0.445; vs parent r_on=0.22 nuc=0 loss=0.239). Nucleation slightly helps r_on=0.24
configuration but still loses to parent.
Morphology (from strip): single tight central blob at all values; nucleation produces
slight halo noise but does NOT split the blob into multiple mounds.
Verdict: FALSIFIED -- even at the morphologically critical r_on=0.24, nucleation
does not break single-attractor.
Knowledge update: definitive falsification of nucleation as multi-mound mechanism.

## Batch 6 summary
Parent UNCHANGED for the FOURTH consecutive batch. loss=0.239 inner=0.510 n=767.

relay.nucleate_{rate,amp} mechanism FULLY FALSIFIED. Across 6 independent sweeps
(0, 1, 2, 3, 7, 15) at every (rate, amp) combination tested, nucleation is at best
neutral and typically harmful. It does NOT seed coexisting multi-centres because
the activator pulses are damped/absorbed by the diffusing self-organising central
field. The single-attractor failure mode survives this intervention.

KEY POSITIVE OBSERVATION (sweep 4): the high-thr multi-knot regime
DOES produce multiple discrete mounds at thr in {0.20, 0.21, 0.25, 0.28},
morphologically the closest the model gets to REAL across the whole batch.
However the radial-profile loss penalises this regime (loss=0.5-2.4) because
the mounds are more spread than a single tight blob. The model has the capacity
for multi-mound morphology; the metric cannot reward it.

Strategic pivot for Batch 7:
  1. NEW MECHANISM: per-cell chemotactic desensitization (sense_adapt) -- when
     cell sits in high cAMP, its effective sense.gain decays toward zero; recovers
     when cAMP is low. Biologically motivated (real Dicty desensitizes after cAMP
     exposure). Hypothesis: desensitized cells stop being pulled into the dominant
     mound, letting peripheral mounds persist -> multi-mound morphology.
  2. Re-sweep relay.thr FINE in multi-knot regime [0.18, 0.30].
  3. Test joint relay.thr x sense_adapt (does desensitization stabilise multi-knot?).
  4. Re-run seed sweep with correct yaml path.
  5. KEEP relay.nucleate_rate=0 in parent (mechanism falsified, do not propagate).

## Batch 7 Sweep 0 -- sense_adapt.adapt_rate (recover=0.02, thr=0.05, range [0, 40])  [falsified]
Hypothesis (H1-B7): per-cell chemotactic desensitization breaks the single-attractor.
Response: cliff -- adapt_rate=0 (ablation) wins decisively. loss=0.239 at rate=0;
every rate>0 jumps to loss 4-9. inner_mass collapses from 0.51 (ablation) to 0.02-0.6
random-noise across the sweep, never higher than parent.
Morphology (from strip): ablation (rate=0) shows the parent single central blob;
every rate>0 produces a sparse few-pixel scattered field with a tiny central knot --
desensitization makes cells unresponsive to cAMP, so they disperse instead of forming
mounds. Worse than parent at every value.
Verdict: FALSIFIED -- desensitization is not a multi-mound rescuer; it abolishes
chemotaxis altogether.
Knowledge update: cell-state heterogeneity via gain-modulation FALSIFIED as a
multi-attractor generator.

## Batch 7 Sweep 1 -- sense_adapt.adapt_rate (recover=0.10, thr=0.05, range [0, 40])  [falsified]
Hypothesis (H2-B7): faster recovery (transient desensitization) preserves chemotaxis
while still breaking the single-attractor.
Response: cliff identical to sweep 0 -- adapt_rate=0 wins (loss=0.239); all non-zero
loss 2-11 with no minimum.
Morphology (from strip): identical pattern -- sparse dispersion at every rate>0;
faster recovery did not rescue the model.
Verdict: FALSIFIED -- transient desensitization regime equally destructive.
Knowledge update: independent of recovery timescale, gain-modulation disperses.

## Batch 7 Sweep 2 -- sense_adapt.adapt_thr (adapt_rate=10, recover=0.02, range [0, 0.7])  [falsified]
Hypothesis (H3-B7): high adapt_thr suppresses desensitization in low-cAMP zones,
leaving only the central mound cells blind -> peripheral mounds survive.
Response: noisy with no monotone trend; best in-sweep thr=0.03 loss=2.97 (vs
parent 0.239). All values catastrophic relative to parent.
Morphology (from strip): sparse scattered cells with tiny knots at every thr;
even at thr=0.70 (where desensitization should be effectively off) the morphology
is degraded because adapt_rate=10 still pulses cells out of the field.
Verdict: FALSIFIED -- no threshold regime rescues the mechanism.
Knowledge update: cAMP threshold structure of desensitization does not matter.

## Batch 7 Sweep 3 -- sense_adapt.adapt_recover (adapt_rate=10, thr=0.05, range [0, 0.5])  [falsified]
Hypothesis (H4-B7): recovery rate sets the saturation/transient tradeoff and an
optimum exists in between.
Response: no clean optimum; best recover=0.17 loss=3.95 (still 16x worse than parent).
Morphology (from strip): sparse dispersion at every recover value; no qualitative
recovery of aggregation.
Verdict: FALSIFIED -- no recovery rate restores the parent behaviour.
Knowledge update: closes the desensitization parameter space.

## Batch 7 Sweep 4 -- relay.thr [0.18, 0.30] NO adaptation  [multi-knot confirmed, loss still > parent]
Hypothesis (H5-B7): re-pin best multi-knot loss/morphology at the parent's no-adapt config.
Response: shallow basin -- best thr=0.22 loss=0.3743 inner=0.524; thr=0.25 loss=2.36
inner=0.90 (over-tight); thr=0.21 loss=1.07 inner=0.80. Multi-knot regime is robust
but the loss floor in this regime is 0.37, vs parent thr=0.10 loss=0.239.
Morphology (from strip): EVERY value shows 2-5 discrete tight blobs -- the
multi-knot morphology is the consistent attractor throughout [0.18, 0.30]; the
strip is the morphologically closest of any sweep in batch 7 (clearer multi-mound
than REAL at several points). At thr=0.25 the mounds tighten into points.
Verdict: SUPPORTED (multi-knot regime exists, robust) but the loss penalty
remains -- consistent with Established #11.
Knowledge update: re-confirms multi-knot at thr~0.22, loss floor 0.37, morphology
the best the model can produce.

## Batch 7 Sweep 5 -- relay.thr [0.18, 0.30] WITH adaptation (rate=10)  [falsified]
Hypothesis (H6-B7): adaptation stabilises multi-knot morphology at lower loss.
Response: catastrophic in this regime -- best thr=0.19 loss=3.18 (vs no-adapt
sweep 4 best=0.37); adaptation makes the multi-knot regime ~10x worse.
Morphology (from strip): sparse scattered points at every thr; the multi-knot
morphology of sweep 4 is destroyed by adaptation. Cells too desensitized to form mounds.
Verdict: FALSIFIED -- adaptation does not rescue multi-knot; it abolishes it.
Knowledge update: adaptation x multi-knot is anti-synergistic.

## Batch 7 Sweep 6 -- sense_adapt.gain [10, 100] (rate=10, recover=0.02, thr=0.05)  [falsified]
Hypothesis (H7-B7): under adaptation, gain optimum shifts upward to compensate
for the s<1 multiplier.
Response: noisy random; best gain=80 loss=3.18 inner=0.317. All values 5-30x worse than parent.
Morphology (from strip): sparse dispersion at every gain; higher gain does NOT
rescue adaptation, it amplifies the desensitization dynamics.
Verdict: FALSIFIED -- gain compensation does not work.
Knowledge update: gain is decoupled from adaptation regime rescue.

## Batch 7 Sweep 7 -- spring.r_on [0.20, 0.30] (rate=10)  [falsified + phase transition]
Hypothesis (H8-B7): adaptation shifts r_on response, possibly producing multi-mound at lower r_on.
Response: r_on in [0.20, 0.255]: loss 3.6-9.3 (degraded by adaptation); r_on>=0.26:
catastrophic loss=28+ with inner_mass=1.0 (all cells collapsed to a single point).
Morphology (from strip): sparse scattered cells with tiny knots up to r_on=0.255;
r_on>=0.26 the morphology disappears (the field becomes a single bright point --
all cells stacked).
Verdict: FALSIFIED -- adaptation destabilises spring dynamics; phase transition
at r_on=0.26 from dispersion to total collapse.
Knowledge update: adaptation + high r_on creates singular collapse mode.

## Batch 7 Sweep 8 -- spring.kadh [40, 220] (rate=10)  [falsified]
Hypothesis (H9-B7): adaptation may change the kadh response.
Response: all values bad; best kadh=180 loss=3.62 (vs no-adapt parent kadh=120
loss=0.239). No interior optimum.
Morphology (from strip): sparse dispersed cells throughout; kadh has no
mountains visible because cells are too dispersed to interact.
Verdict: FALSIFIED -- kadh response not rescued by adaptation.
Knowledge update: adaptation decouples spring response (cells too dispersed).

## Batch 7 Sweep 9 -- camp.decay [0.10, 0.35] (rate=10)  [falsified]
Hypothesis (H10-B7): decay x adaptation coupling produces a better operating point.
Response: oscillatory noise; best decay=0.24 loss=2.59 (vs parent decay=0.20 loss=0.239).
Morphology (from strip): sparse dispersion at every decay; decay does not modulate
the adaptation dispersion mechanism.
Verdict: FALSIFIED -- no coupling rescue.
Knowledge update: closes (camp.decay x adaptation) joint regime.

## Batch 7 Sweep 10 -- seed [0, 15] at parent  [BROKEN -- all NaN]
Hypothesis (H11-B7): noise-floor estimate across seeds at parent.
Response: ALL NaN -- the eval_sweeps machinery still does not write the root
"seed" param into the spec. Second consecutive failure (was also broken in B6 sw13).
Morphology (from strip): blank strip (no sim density rendered).
Verdict: INCONCLUSIVE -- machinery bug, not science. Must fix eval_sweeps.py
to set root-level scalar params (seed, dt, n_frames, vmax) for B8 to succeed.
Knowledge update: action item for code fix; no noise-floor data yet.

## Batch 7 Sweep 11 -- sense_adapt.adapt_rate WIDE [0, 100]  [falsified]
Hypothesis (H12-B7): extreme desensitization flips back to dispersion saturation.
Response: rate=0 wins (parent loss=0.239); no in-range value approaches it.
Loss varies 0.5-10 with no monotone trend at high rate.
Morphology (from strip): rate=0 = parent single blob; all rate>0 sparse dispersion;
even rate=100 cells do not re-aggregate.
Verdict: FALSIFIED -- no saturation reversal.
Knowledge update: closes the adapt_rate wide range definitively.

## Batch 7 Sweep 12 -- sense_adapt.adapt_rate at relay.thr=0.25  [falsified]
Hypothesis (H13-B7): adaptation rescues the multi-knot regime to lower loss.
Response: ablation (rate=0, multi-knot alone) loss=2.36 inner=0.90 -- the same
value seen in sweep 4 at thr=0.25; every adapt_rate>0 loss 4-8.
Morphology (from strip): rate=0 shows the tight multi-knot strip pattern (4-5
discrete mounds); rate>0 destroys it into sparse dispersion. Adaptation is
unconditionally bad in this regime too.
Verdict: FALSIFIED -- adaptation does not lower multi-knot loss.
Knowledge update: definitive end of the adaptation hypothesis family.

## Batch 7 Sweep 13 -- random_walk.strength [0, 0.012] at parent (no adapt)  [parent confirmed]
Hypothesis (H14-B7): clean re-sweep at parent to confirm random_walk=0.003 noise floor.
Response: shallow basin around parent; best strength=0.003 loss=0.239 (parent
exactly); strength=0.005 loss=0.307; all values 0.24-0.67 -- relatively tight.
Morphology (from strip): all values show the parent diffuse-cloud morphology with
slight density variation; no qualitative change.
Verdict: parent re-confirmed.
Knowledge update: random_walk.strength=0.003 robustly best; Batch 6 sw10's loss
inflation was due to nucleation interaction, not the random_walk parameter.

## Batch 7 Sweep 14 -- relay.eps [0.005, 0.10] at parent  [parent confirmed]
Hypothesis (H15-B7): slow FN refractory (field-side analog of cell adaptation)
gives mounds time to grow before quenching.
Response: shallow basin; best eps=0.02 loss=0.239 (parent); all values 0.24-0.65.
Morphology (from strip): all values show parent-like diffuse-cloud morphology;
no multi-mound emergence; no qualitative change with eps.
Verdict: FALSIFIED as a multi-mound mechanism; parent eps=0.02 confirmed.
Knowledge update: relay.eps surface is flat-noisy around the parent (consistent
with Established #8); field-side adaptation analog does not produce multi-knot either.

## Batch 7 Sweep 15 -- cell.n [600, 1500] at relay.thr=0.25  [supported -- best multi-knot point]
Hypothesis (H16-B7): more cells in multi-knot regime crisp up morphology.
Response: noisy U-shape; best cell.n=1500 loss=0.4522 inner=0.521; second best
cell.n=1400 loss=0.501. The multi-knot regime + extra cells lowers the loss from
the no-adapt sweep-4 thr=0.25 baseline (2.36) by 5x. STILL worse than parent
(0.239) but the MORPHOLOGICALLY closest configuration of the entire batch.
Morphology (from strip): clean multi-mound at every cell.n value -- 3-6 discrete
tight blobs that grow more populated as cell.n increases; at n=1400-1500 the
mounds are dense and the visual match to REAL is the best in batch 7.
Verdict: SUPPORTED -- cell.n is a tuning knob for multi-knot morphology;
larger n gives more credible mound density.
Knowledge update: NEW best multi-knot configuration: relay.thr=0.25, cell.n=1500,
loss=0.4522, inner=0.521. Still loses to parent on loss but morphologically
preferable -- new evidence the radial-profile loss is the bottleneck.

## Batch 7 summary
Parent UNCHANGED for the SIXTH consecutive batch (loss=0.239, inner_mass=0.510,
n_final=767). sense_adapt mechanism FULLY FALSIFIED across 11 sweeps testing
every parameter and combination (rate alone, gain joint, thr alone, recover alone,
multi-knot joint, kadh joint, r_on joint, camp.decay joint, sense.gain joint).
Cell-state heterogeneity via gain modulation does NOT break the single-attractor;
instead it abolishes chemotaxis entirely, causing dispersion. The mechanism is
removed from the schedule for Batch 8 (replaced with plain `sense`).

POSITIVE OBSERVATIONS:
  - Multi-knot regime re-confirmed and now scaled: relay.thr=0.25 + cell.n=1500
    gives loss=0.4522 with morphologically convincing 3-6 mounds (best multi-mound
    config so far).
  - Parent values reconfirmed: random_walk=0.003, relay.eps=0.02 (B5 ledger intact).
  - relay.eps wide sweep (field-side adaptation analog) also fails to produce
    multi-knot -- complements the cell-side falsification.

NEGATIVE OBSERVATIONS:
  - seed sweep BROKEN AGAIN (machinery bug -- eval_sweeps does not handle root
    scalar params); noise-floor still unmeasured.

Strategic pivot for Batch 8:
  1. REVERT parent: drop sense_adapt from schedule, use plain `sense.gain=40`.
     (sense_adapt stays in code as a falsified-mechanism ablation but is not scheduled.)
  2. NEW MECHANISM ADDED: `align` -- local velocity-alignment among neighbours
     (Vicsek-style), creating polar streams. Tests whether STREAMING -- a real
     Dicty observable not currently in the model -- creates a multi-mound-friendly
     dynamical regime by producing line-shaped recruitment paths rather than
     radial recruitment.
  3. Use the new best multi-knot point (thr=0.25, cell.n=1500) as a SECONDARY
     control for joint sweeps; map its neighbourhood.
  4. Do NOT re-sweep parameters now established as flat-around-parent.

## Batch 8 Sweep 0 -- align.strength [0, 0.10] at parent (alpha=0.20, beta=0.40)  [falsified]
Hypothesis (H1-B8): neighbour-coupled polarity (Vicsek streaming) breaks the single-attractor.
Response: noisy-flat. ablation strength=0 wins (loss=0.2388 = parent); all non-zero values
loss 0.34-0.57 with no minimum interior to the range.
Morphology (from strip): every value (including non-zero) shows the parent diffuse-cloud
with a faint central knot -- NO stream emerges; the polarity contribution is just additional
noise on top of chemotaxis. No multi-mound at any strength.
Verdict: FALSIFIED -- align.strength has no productive optimum in [0, 0.10]; streaming
mechanism is neutral-to-degrading vs parent.
Knowledge update: Vicsek-style polarity does not produce streams at moderate strength.

## Batch 8 Sweep 1 -- align.align_alpha [0, 0.8] at strength=0.04, beta=0.40  [falsified]
Hypothesis (H2-B8): neighbour-coupling strength controls stream emergence.
Response: noisy; best alpha=0.0 (no neighbour coupling -> pure persistence + chemo bias)
loss=0.3204; flat-poor across alpha; spikes at alpha=0.10, 0.15 (loss 0.82) and
0.50, 0.60 (loss 0.58, 0.62).
Morphology (from strip): every alpha shows a diffuse cloud with a small central knot --
no stream-shaped recruitment paths. The "neighbour-aligned" cells behave as a slightly
noisier version of independent walkers.
Verdict: FALSIFIED -- align_alpha is not the streaming knob; no value beats ablation.
Knowledge update: Vicsek neighbour coupling does not create line-shaped recruitment in
the chemotactic model.

## Batch 8 Sweep 2 -- align.chemo_beta [0, 1.0] at strength=0.04, alpha=0.20  [falsified]
Hypothesis (H3-B8): chemotaxis-vs-alignment balance has an interior optimum.
Response: very noisy; best beta=0.10 loss=0.2886 (worse than parent 0.239); occasional
spikes (beta=0.05 loss=0.63, beta=0.50 loss=0.77, beta=0.20 loss=0.87).
Morphology (from strip): same diffuse cloud + small central knot at every beta;
beta=1.0 (pure chemo through polarity) ~= parent morphologically.
Verdict: FALSIFIED -- no interior optimum; pure chemo bias (beta=1) is the best the
mechanism can do, and even then it loses to parent.
Knowledge update: when align is on, chemotactic bias dominates but adds nothing over
direct chemotaxis.

## Batch 8 Sweep 3 -- align.align_r [0.01, 0.22] at strength=0.04, alpha=0.20, beta=0.40  [falsified]
Hypothesis (H4-B8): neighbour-radius sets stream width and there is an optimal scale.
Response: chaotic for small r (r<=0.025 catastrophic loss 0.42-1.29), then flat 0.30-0.49
above r=0.03; best r=0.09 loss=0.2996 (still worse than parent 0.239).
Morphology (from strip): below r=0.03 cells exhibit jittery clustering; above r=0.05
the parent diffuse-cloud + central knot reappears. No streams or line patterns at any r.
Verdict: FALSIFIED -- no scale produces stream morphology; small r creates instability,
large r is irrelevant. Streaming radius is not the missing knob.
Knowledge update: closes the align_r dimension; no Vicsek scale produces streams here.

## Batch 8 Sweep 4 -- relay.thr [0.18, 0.30] NO align  (multi-knot CONTROL)  [supported, repro]
Hypothesis (H5-B8): re-pin the multi-knot regime at the no-align parent for B8 control.
Response: noisy U with shallow basin around thr=0.22 (loss=0.3743 inner=0.524); thr=0.25
spikes to loss=2.36 inner=0.90 (single tight blob -- over-collapse); secondary low at
thr=0.20 (loss=0.55); high-thr (>=0.27) loss 0.59-1.88.
Morphology (from strip): EVERY value in [0.18, 0.21] shows 3-5 discrete mounds; thr=0.22
edge of multi-knot regime (single dominant + satellites); thr=0.25 single over-tight blob;
thr=0.28-0.30 multi-mound returns but more diffuse. Strip is morphologically the closest
of any sweep in B8 -- multi-mound is robust in this regime.
Verdict: SUPPORTED -- multi-knot exists, replicates Batch 7 sweep 4 exactly; loss floor
unchanged at ~0.37, well above parent 0.239.
Knowledge update: Established #11 reconfirmed (now FIVE batches).

## Batch 8 Sweep 5 -- relay.thr [0.18, 0.30] WITH align (strength=0.04, alpha=0.2, beta=0.4)  [falsified]
Hypothesis (H6-B8): streaming-style polarity lowers the multi-knot loss.
Response: best thr=0.19 loss=0.4161 (worse than no-align sweep 4 best 0.3743 by ~10%);
all values loss 0.41-1.26; no improvement at any thr.
Morphology (from strip): multi-knot regime still present at thr [0.19, 0.22] (3-5 mounds)
but visibly more diffuse than sweep 4; align spreads cells more evenly between mounds
rather than tightening them.
Verdict: FALSIFIED -- align makes multi-knot WORSE by ~10% loss; no joint improvement.
Knowledge update: align does not crisp multi-knot mounds; it diffuses them.

## Batch 8 Sweep 6 -- spring.r_on [0.18, 0.30] at thr=0.25, n=1500  [phase transition mapped]
Hypothesis (H7-B8): r_on tuning in the multi-knot best regime refines best point.
Response: monotone (cliff) -- r_on=[0.18, 0.225] loss 0.43-1.75 (no aggregation
at very low r_on); BEST r_on=0.215 loss=0.4294 inner=0.599; r_on>=0.225 catastrophic
loss 1.25-8.1 with inner climbing to 0.99 (all cells stack to a single tiny point).
Morphology (from strip): r_on=[0.18, 0.215] shows 2-3 visible mounds with cloud
periphery; r_on=0.22 strong multi-knot (3-4 mounds); r_on>=0.225 the strip collapses
to ONE bright point (full single-attractor collapse); r_on>=0.245 essentially blank
(all cells fused).
Verdict: PARTIAL -- thr=0.25 multi-knot best r_on is 0.215, marginal improvement over
sw 4 thr=0.22 (0.37); phase transition at r_on=0.225 to single-point collapse confirms
Established #10 ceiling.
Knowledge update: in multi-knot regime, r_on is bounded above by a tight ceiling at 0.225;
the "good" region is narrow [0.21, 0.225].

## Batch 8 Sweep 7 -- spring.kadh [40, 240] at thr=0.25, n=1500  [supported (modest)]
Hypothesis (H8-B8): kadh in multi-knot has a different optimum than at parent.
Response: best kadh=60 loss=0.3632 inner=0.604 -- a real ~20% improvement on the
thr=0.25,n=1500 baseline (~0.452); flat band 0.45-0.55 across [40, 200] with a single
strong outlier at kadh=150 (loss=2.09 inner=0.88 -- isolated over-collapse mode); kadh
>=220 degrades to 1.43.
Morphology (from strip): all kadh in [40, 200] show clean 2-4 mounds with kadh=60 being
most balanced; kadh=150 is a single tight knot (outlier); kadh>=220 mounds dilate and
lose definition.
Verdict: SUPPORTED -- kadh=60 is a NEW BEST multi-knot config: loss=0.3632 < sw 4 best
0.3743. New most-credible multi-mound point.
Knowledge update: NEW best multi-knot point: thr=0.25, n=1500, kadh=60 -> loss=0.3632
inner=0.604 (closest to REAL inner=0.61 of any multi-mound config so far).

## Batch 8 Sweep 8 -- camp.decay [0.10, 0.40] at thr=0.25, n=1500  [flat]
Hypothesis (H9-B8): higher decay keeps multi-centres separated.
Response: flat-noisy; best decay=0.34 loss=0.4361; baseline parent decay=0.20 loss=0.4522;
single sharp outlier at decay=0.26 loss=1.467 (transient single-collapse).
Morphology (from strip): 2-4 mounds at every decay in [0.10, 0.40] except 0.26 (single
collapsed point) -- decay is a soft modulator, not a multi-mound creator/destroyer.
Verdict: FALSIFIED as a discriminating axis -- decay does not significantly modulate
multi-knot loss; in-range improvement is within noise.
Knowledge update: camp.decay flat-noisy in multi-knot regime, similar to parent.

## Batch 8 Sweep 9 -- secrete.rate [2, 24] at thr=0.25, n=1500  [supported (mild)]
Hypothesis (H10-B8): the secretion floor shifts in multi-knot regime.
Response: shallow U; best rate=9 loss=0.4024 inner=0.505; lower rates (2-3) catastrophic
(no chemo gradient, no aggregation; loss 0.99-1.74); high rate >=18 catastrophic
(over-saturated field, cells stack; loss 2.21-6.88).
Morphology (from strip): rate=2-3 sparse few-pixel; rate=4-12 clean 2-3 mounds; rate=14-16
mounds get more diffuse; rate>=18 single elongated structure; rate=24 a single small
collapsed knot.
Verdict: SUPPORTED -- secretion has a clear working window [4, 12] in multi-knot;
best rate=9 marginally improves over baseline rate=8. Floor unchanged from parent.
Knowledge update: secrete.rate window [4, 12] in multi-knot regime; best=9.

## Batch 8 Sweep 10 -- relay.gain [0, 300] at thr=0.25, n=1500  [supported -- new best]
Hypothesis (H11-B8): in multi-knot regime, relay still necessary; gain optimum may shift.
Response: complex -- gain=0 catastrophic loss=1.39 (relay necessity reconfirmed in
multi-knot); gain=20 loss=0.50 (low); gain=40 isolated peak loss=3.68 inner=0.93
(over-collapse outlier); gain in [60, 300] varies 0.38-0.71 with shallow basin
around gain=140-260 (best gain=140 loss=0.3807 inner=0.526).
Morphology (from strip): gain=0 sparse (few cells, no aggregation); gain=40 single
tight point (the outlier); gain=60-300 clean 2-4 mounds throughout; gain=140 the
crispest multi-mound. High gain (>=280) the mounds start to merge.
Verdict: SUPPORTED -- relay.gain=140 is the new best multi-knot config (loss=0.3807).
Necessity of relay reconfirmed (gain=0 catastrophic).
Knowledge update: relay.gain=140 in multi-knot regime improves over parent gain=120
by ~16%.

## Batch 8 Sweep 11 -- cell.n [1200, 2400] at thr=0.25 (fine sweep)  [supported -- new best]
Hypothesis (H12-B8): more cells crisp up the multi-knot morphology further.
Response: noisy U; best cell.n=1450 loss=0.3431 inner=0.6 -- NEW BEST multi-knot
loss (improves on Batch 7 sw15 n=1500 loss=0.452 by 25%). Secondary low at n=1800
loss=0.3725; n=1200, 1250 catastrophic (loss 1.75, 2.08 -- not enough cells per mound).
n>=2000 climbs to 0.9-2.0 (cells over-concentrate).
Morphology (from strip): all values show clean 2-3 vertical mounds; n=1200-1250 mounds
are sparse; n=1300-1900 well-populated multi-mound; n>=2000 mounds tighten and start
losing the multi-knot signature.
Verdict: SUPPORTED -- cell.n=1450 is the new best multi-knot point. Confirms B7
sw15 finding while refining the optimum.
Knowledge update: NEW best multi-knot point: thr=0.25, n=1450 -> loss=0.3431 inner=0.6.
This is the morphologically-closest config in the project so far AND the lowest loss
among configs that retain the multi-mound morphology.

## Batch 8 Sweep 12 -- align.strength [0, 0.13] at thr=0.25, n=1500  [falsified]
Hypothesis (H13-B8): align rescues the multi-knot regime to lower loss.
Response: noisy; best strength=0.015 loss=0.4087 (baseline strength=0 loss=0.452);
flat-poor across [0.02, 0.13] (loss 0.43-1.0).
Morphology (from strip): all strengths show diffuse multi-mound; align makes the
mounds less crisp than no-align.
Verdict: FALSIFIED -- align in multi-knot is a small noise-level improvement (likely
just stochastic), not a structural rescue.
Knowledge update: align + multi-knot is neutral-to-degrading; closes the joint test.

## Batch 8 Sweep 13 -- align.align_alpha [0, 0.8] at thr=0.25, n=1500, strength=0.04, beta=0.4  [falsified]
Hypothesis (H14-B8): neighbour coupling helps multi-knot mounds organise into streams.
Response: noisy, no clean trend; best alpha=0.55 loss=0.4112; baseline alpha=0
(strength=0.04 no coupling) loss=0.4133.
Morphology (from strip): multi-mound preserved at all alpha values, neither tighter
nor more stream-like; the strip is visually almost identical across alpha.
Verdict: FALSIFIED -- align_alpha has no effect on multi-knot loss or morphology.
Knowledge update: definitively closes align + multi-knot joint.

## Batch 8 Sweep 14 -- cell.seed [0, 15] at parent  [BROKEN -- determinism, no variance]
Hypothesis (H15-B8): measure noise-floor across seeds at parent (workaround for the
root-seed bug).
Response: COMPLETELY FLAT -- all 16 values give loss=0.2388 inner=0.510 exactly.
The cells are seeded from init_npz (real frame-0 positions), so cell.seed has no
effect on the initial condition. The sweep is structurally broken: there is no
randomised initial scatter to sweep.
Morphology (from strip): 16 identical strips.
Verdict: INCONCLUSIVE -- cannot measure seed-noise via cell.seed under the
init_npz pipeline. Random_walk and align both have their own seeds; need to
sweep one of those (e.g. align.seed, random_walk.seed) to get a noise floor.
Knowledge update: third consecutive batch with a broken seed measurement; the
mechanism this time was the init_npz override, not the eval_sweeps bug.

## Batch 8 Sweep 15 -- align.strength WIDE [0, 0.30]  [marginal best, within noise]
Hypothesis (H16-B8): at higher strength, polarity flow self-organises into flock-like
single-direction streams.
Response: best strength=0.22 loss=0.2379 inner=0.551 -- marginally lower than parent
(0.2388 by 0.4%, well within the typical loss-surface noise of 0.05 seen across
B4-B7). Curve oscillates 0.24-0.73 across [0, 0.30] with no clean trend.
Morphology (from strip): all values show the parent diffuse-cloud + small central
knot morphology; no flock, no stream, no multi-mound; the strip is visually
indistinguishable from sweep 0's. Inner_mass spikes occasionally to 0.55-0.57
when chance polarity alignment momentarily concentrates cells, but it doesn't
correspond to any morphological qualitative change.
Verdict: FALSIFIED (counting the 0.001 loss improvement as noise) -- align at no
strength produces the qualitative regime change hypothesised. The "best"
strength=0.22 is morphologically identical to parent.
Knowledge update: align mechanism FULLY FALSIFIED across SIX B8 sweeps (0, 1, 2,
3, 12, 13, 15) covering every parameter and joint. The 0.4% loss "win" at
strength=0.22 is within the typical loss-surface noise floor and produces no
qualitative morphological change. Vicsek-style polarity + chemotactic bias does
NOT break the single-attractor.

## Batch 8 summary
Parent UNCHANGED for the SEVENTH consecutive batch (loss=0.239, inner_mass=0.510,
n_final=767). align mechanism FULLY FALSIFIED across 6 sweeps testing strength
(narrow + wide), align_alpha, chemo_beta, align_r, and joints with multi-knot
(strength + align_alpha). Vicsek-style polarity with chemotactic bias is neutral-
to-degrading at every (strength, alpha, beta, r) combination tested; no stream-
shaped recruitment emerges; single-attractor morphology persists. The marginal
"win" at strength=0.22 (0.2379 vs 0.2388) is within the loss-surface noise floor
and morphologically identical to parent.

POSITIVE OBSERVATIONS:
  - NEW BEST MULTI-KNOT CONFIG: thr=0.25, cell.n=1450 -> loss=0.3431, inner=0.6.
    25% improvement over the Batch 7 best (loss=0.452). This is now the
    morphologically-closest config in the entire project history AND the lowest
    loss among configs that retain multi-mound morphology.
  - In multi-knot regime, kadh=60 (sw 7) and relay.gain=140 (sw 10) lower loss
    by 15-20% over the (n=1500, kadh=120, gain=120) baseline. These are
    consistent independent improvements.
  - secrete.rate optimum unchanged at 8-9 in multi-knot regime (sw 9).
  - Established #10 (single-attractor morphology ceiling) reconfirmed: at
    r_on=0.225 in multi-knot regime, the model phase-transitions to total
    single-point collapse (sw 6).
  - Relay NECESSITY reconfirmed in multi-knot regime (sw 10, gain=0 catastrophic).

NEGATIVE OBSERVATIONS:
  - cell.seed sweep BROKEN AGAIN -- this time due to init_npz overriding the
    initial scatter. Three consecutive batches without a seed-noise measurement.
  - The "ceiling" of multi-knot loss (~0.34) is still 1.4x parent loss (0.239).
    The radial-profile loss STRUCTURALLY penalises multi-mound (Established #11).
    The morphologically-best multi-knot loss has come down (0.45 -> 0.343 over
    two batches) but cannot cross the parent's single-blob loss.

Strategic pivot for Batch 9:
  1. DROP align from schedule (FALSIFIED -- keep in code as ablation).
  2. Adopt the new multi-knot best point as the SECONDARY CONTROL:
     thr=0.25, n=1450, kadh=60, gain=140 -> loss=0.3431, inner=0.6 (the
     morphologically credible config).
  3. NEW MECHANISM: ACTIVATOR-INHIBITOR (Gierer-Meinhardt) -- add a second
     long-range slow-decay inhibitor field `inhib` that the cells secrete and
     AVOID (negative chemotaxis). This is the classical Turing recipe for
     stable multi-peak patterns. Tests whether the model's missing multi-mound
     stability comes from the absence of LATERAL INHIBITION between forming
     mounds. Ablation = inhib_gain=0 (and/or inhib_rate=0) -> recovers parent.
  4. Drill the multi-knot best point: joint refinements around
     (thr=0.25, n=1450, kadh=60, gain=140) to find sub-0.34 loss configs.
  5. seed sweep workaround #3: sweep relay.seed or random_walk.seed (not
     overridden by init_npz) to measure the noise floor properly.

# Batch 9 — activator-INHIBITOR (Gierer-Meinhardt) test + multi-knot drill

Real inner_mass=0.606. Parent baseline (inhib OFF) = loss=0.2388, inner=0.510,
n_final=767. Secondary control multi-knot point B8 = thr=0.25, n=1450, kadh=60,
gain=140 -> loss=0.343, inner=0.6.

## Batch 9 Sweep 0 — inhib_op.inhib_gain  [falsified]
Hypothesis: H1-B9 lateral inhibition breaks single-attractor at parent regime.
Response: monotone catastrophic; gain=0 loss=0.2388 inner=0.510. Any non-zero
gain blows loss to 1.24-2.30 and inner_mass collapses to 0.16-0.27.
Best value=0.0 (=parent ablation).
Morphology (from strip): gain=0 shows the parent single tight central blob;
EVERY non-zero gain disperses cells to a uniform speckled noise field. No
multi-spot pattern, no Turing peaks, no streams -- just global repulsion.
Verdict: FALSIFIED -- the anti-chemotactic force from grad(inhib) does NOT
produce stable multi-spot Turing peaks; it produces global dispersion.

## Batch 9 Sweep 1 — inhib_op.inhib_rate  [falsified]
Hypothesis: H2-B9 inhibitor deposition rate has an interior optimum at gain=20.
Response: rate=0 wins (loss=0.2388 = parent because effective contribution=0);
all rate>0 give loss 1.83-2.24, inner=0.16-0.21.
Best value=0.0 (ablation).
Morphology: rate=0 single tight blob; all non-zero rates produce the same
dispersed speckle field. No structure emerges from any rate.
Verdict: FALSIFIED -- inhibitor field cannot self-organise into peaks because
cells continuously top it up everywhere they go.

## Batch 9 Sweep 2 — inhib.diffusion  [falsified]
Hypothesis: H3-B9 a Turing length-scale exists at some D_inhib >> D_camp=0.0008.
Response: flat-bad. Loss 2.03-2.32 across D_inhib in [0.002, 0.2]; inner_mass
stuck at 0.15-0.20. No interior optimum.
Best value=0.07 (loss=2.03, morphologically the same dispersed field).
Morphology: all 16 panels show the same dispersed speckle -- diffusion of the
inhibitor cannot rescue what cell-side deposition has already broken.
Verdict: FALSIFIED.

## Batch 9 Sweep 3 — inhib.decay  [falsified]
Hypothesis: H4-B9 inhib_decay << camp_decay=0.20 gives a Turing time-scale.
Response: flat-bad. Loss 1.99-2.34 across decay in [0.005, 0.20]; no optimum.
Best value=0.15 (loss=1.99, dispersed).
Morphology: identical dispersed speckle across all decay values.
Verdict: FALSIFIED.

## Batch 9 Sweep 4 — inhib_op.inhib_gain @ MULTI-KNOT  [falsified]
Hypothesis: H5-B9 inhibition CRISPS multi-knot mounds (thr=0.25, n=1450,
kadh=60, gain=140).
Response: gain=0 wins by ~3x (loss=0.776, inner=0.73); any non-zero gain
disperses cells (loss 1.79-2.30, inner 0.16-0.19). Multi-knot regime is even
MORE sensitive to inhibitor than parent regime.
Best value=0.0 (ablation).
Morphology: gain=0 shows a clean tight multi-mound knot (the secondary control);
every non-zero gain destroys it into the same dispersed speckle.
Verdict: FALSIFIED -- inhibition does not crisp multi-mound; it destroys it.

## Batch 9 Sweep 5 — spring.r_on FINE @ MULTI-KNOT  [SUPPORTED -- breakthrough]
Hypothesis: H6-B9 narrow r_on band [0.20, 0.225] at multi-knot best refines
the pre-collapse threshold.
Response: noisy with clear minimum at r_on=0.224 (loss=0.2594, inner=0.55).
This is the LOWEST multi-knot loss ever measured and is COMPETITIVE WITH
PARENT (0.2388). Three local minima visible: r_on=0.214 (loss=0.386),
r_on=0.222 (0.370), r_on=0.224 (0.259).
Best value=0.224 (loss=0.2594, inner=0.55).
Morphology: strip shows clean 2-3 mound morphology at EVERY value in the
range -- two compact mounds with a faint central density. Closer in
appearance to the REAL strip than the parent's single tight blob.
Verdict: SUPPORTED -- a NEW credible multi-mound config at competitive loss.

## Batch 9 Sweep 6 — spring.kadh FINE @ MULTI-KNOT  [supported]
Hypothesis: H7-B9 kadh band [30,120] refines kadh=60 at multi-knot.
Response: noisy with clear local minimum at kadh=75 (loss=0.2874, inner=0.548).
Other low values: kadh=55 (0.366), kadh=100 (0.349). kadh=120 (parent value)
loses (0.649). Within multi-knot kadh strongly prefers ~75.
Best value=75.0 (loss=0.2874, inner=0.548).
Morphology: clean 2-3 mound morphology throughout; mounds tighter at higher
kadh (~100-120) but the radial-profile loss disfavours it; kadh=75 is the
clean sweet spot.
Verdict: SUPPORTED -- kadh=75 lowers multi-knot loss from 0.343 to 0.287.

## Batch 9 Sweep 7 — relay.gain @ MULTI-KNOT  [inconclusive]
Hypothesis: H8-B9 gain band [60,240] refines gain=140 at multi-knot.
Response: noisy with multiple local minima: gain=170 (loss=0.376), gain=240
(0.377), gain=140 (0.776 -- but this is the seed=0 mark, sweep parent).
Best value=170 (within noise of 140, 220, 240).
Morphology: clean multi-mound throughout, qualitatively identical across
the range. Suggests relay.gain >120 is sufficient and the precise value
within [120, 240] is in the noise floor.
Verdict: INCONCLUSIVE -- gain=140 is fine, no strong evidence for shift.

## Batch 9 Sweep 8 — cell.n FINE @ MULTI-KNOT  [partial]
Hypothesis: H9-B9 cell.n band [1300,1700] refines n=1450.
Response: clear local minimum at n=1400 (loss=0.325, inner=0.589). n=1300
catastrophic (1.044). n>=1450 hovers 0.36-0.70.
Best value=1400 (loss=0.3253, inner=0.589).
Morphology: 2-3 clean mounds throughout; n=1400 looks most REAL-like.
Verdict: PARTIAL -- optimum has shifted slightly down (1450 -> 1400).

## Batch 9 Sweep 9 — relay.thr FINE @ MULTI-KNOT  [partial]
Hypothesis: H10-B9 re-check thr at combined best point.
Response: clear local minimum at thr=0.23 (loss=0.3144, inner=0.582). thr=0.25
(sweep parent) loses (0.776 -- seed=0 outlier). Trough is broad: thr=0.235
(0.366), thr=0.27 (0.394).
Best value=0.23 (loss=0.3144).
Morphology: clean 2-3 mound at all values; thr above 0.28 -> more diffuse
mounds; below 0.21 reverts to fewer/tighter mounds.
Verdict: PARTIAL -- subtle shift thr 0.25 -> 0.23 lowers loss.

## Batch 9 Sweep 10 — camp.diffusion @ MULTI-KNOT  [supports Est #5]
Hypothesis: H11-B9 low camp.diffusion preference holds at multi-knot.
Response: clear local minimum at D=0.0004 (loss=0.360, inner=0.594). D=0.0008
(parent value) loses with seed=0 mark (0.776). D up to 0.003 noisy ~0.4-0.6.
Best value=0.0004.
Morphology: clean 2-3 mound throughout in [0.0002, 0.003]; mounds slightly
more diffuse at high D.
Verdict: SUPPORTED -- Est #5 holds; refines multi-knot D to 0.0004.

## Batch 9 Sweep 11 — random_walk.strength @ MULTI-KNOT  [partial]
Hypothesis: H12-B9 random_walk interior optimum in multi-knot regime.
Response: noisy with several minima: strength=0.009 (loss=0.302, inner=0.529),
strength=0.005 (0.345), strength=0.012 (0.345). strength=0.003 (parent value)
loses (0.776 -- seed=0 mark).
Best value=0.009.
Morphology: 2-3 clean mounds throughout; mound positions vary across the
strip more than other sweeps (noise doing what it should).
Verdict: PARTIAL -- in multi-knot, slightly higher RW helps (0.003 -> 0.009).

## Batch 9 Sweep 12 — inhib.diffusion @ MULTI-KNOT  [falsified]
Hypothesis: H13-B9 Turing scale is mass-dependent.
Response: flat-bad. Loss 2.07-2.27 across D in [0.002, 0.2]; inner stuck
~0.17-0.19. Dispersed at every value.
Best value=0.008.
Morphology: dispersed speckle across all 16 values.
Verdict: FALSIFIED -- mass does not rescue inhibitor.

## Batch 9 Sweep 13 — seed (cell init seed)  [seed-noise floor measured]
Hypothesis: H14-B9 noise floor measurement.
Response: seed=0 loss=0.2388 (matches parent); other seeds spread 0.357-1.20.
Median ~0.46. Significant variance.
Best value=0 (parent seed).
Morphology: single-blob morphology across all seeds, but blob location and
tightness varies, dominating radial-profile loss.
Verdict: INFORMATIVE -- loss-surface noise floor at parent is ~0.30-0.50;
multi-knot loss=0.26 from sw 5 is INDISTINGUISHABLE from parent under seed
noise. Parent 0.239 is a lucky seed=0 minimum.

## Batch 9 Sweep 14 — inhib_op.inhib_gain @ STRONG-TURING recipe  [falsified]
Hypothesis: H16-B9 maximally-tuned Gierer-Meinhardt (D=0.05, decay=0.02,
rate=8) at strength.
Response: best gain=2 loss=0.954, inner=0.334 -- 4x WORSE than gain=0 case
from sw 0. Loss climbs to 2.0+ at gain>=42.
Best value=2 (still 4x worse than parent ablation).
Morphology: a faint speckled "ghost-of-a-blob" at gain=0, dispersing to noise
as gain increases. NO multi-peak emerges anywhere.
Verdict: FALSIFIED -- the strongest Gierer-Meinhardt recipe DOES NOT produce
multi-peak. Inhibitor mechanism dead.

## Batch 9 Sweep 15 — inhib_op.inhib_gain @ NO-RELAY  [falsified]
Hypothesis: H15-B9 inhibition WITHOUT relay breaks single-attractor.
Response: gain=0 wins (loss=0.239 same as parent here). Non-zero gains all
>2.0 dispersed.
Best value=0.0.
Morphology: gain=0 single tight blob; non-zero gain dispersed speckle.
Verdict: FALSIFIED -- inhibitor alone, with or without relay, never produces
multi-peak.

## Batch 9 summary
Parent loss UNCHANGED at 0.2388 (seed=0 lucky draw, near noise-floor minimum).
BUT a MAJOR scientific result emerges: under seed-noise budget (sw 13 measured
spread ~0.30-1.20 across 16 seeds), the multi-knot regime now TIES parent
loss with multi-mound morphology, and the morphology is qualitatively the
closest to REAL of anything seen.

KEY POSITIVE: sweep 5 (spring.r_on=0.224 @ thr=0.25, n=1450, kadh=60, gain=140)
gives loss=0.2594, inner=0.55 with CLEAN MULTI-MOUND (2-3 mounds). The
combined best multi-knot parent (sw 6 kadh=75, sw 8 n=1400, sw 9 thr=0.23,
sw 11 rw=0.009, sw 10 D=0.0004) is the new PRIMARY PARENT for Batch 10.

INHIBITOR FALSIFIED across 8 sweeps (0, 1, 2, 3, 4, 12, 14, 15). At every
(gain, rate, diffusion, decay, regime, with/without relay) combination, the
inhibitor causes global dispersion, not stable multi-peak Turing pattern.
Mechanism removed from base spec for B10 (kept in code as ablation).

Strategic pivot for Batch 10:
  1. PROMOTE multi-knot regime to PRIMARY PARENT (first regime change in 8
     batches). New parent: thr=0.23, n=1400, kadh=75, gain=140, r_on=0.224,
     random_walk=0.009, camp.diffusion=0.0004. Old single-blob parent kept
     as secondary control.
  2. DROP all inhib_* sweeps (FALSIFIED across 8 sweeps).
  3. NEW MECHANISM: per-cell PERSISTENCE (motion memory) operator
     `persistence` -- each cell carries a polarity p_i updated as
     p_i = (1-rho)*p_i + rho*v_i/|v_i|, contributing accel = strength * p_i.
     NO neighbour coupling (distinct from align which is FALSIFIED).
     Ablation = strength=0 -> recovers parent.
  4. Heavy budget on multi-knot joint refinement (kadh*r_on, n*thr, gain*thr).
  5. Seed sweep at NEW multi-knot parent to measure noise floor in the
     multi-mound regime.


==================================================================
BATCH 10 — 16 sweeps × 16 values around NEW multi-knot parent + new mechanism
PERSISTENCE (per-cell self-only motion memory)
==================================================================

## Batch 10 Sweep 0 — persistence.strength @ rho=0.3, multi-knot parent  [falsified]
Hypothesis: H1-B10 persistence breaks single-attractor / crisps multi-mound by
acting as a per-cell motion memory; non-zero strength lowers loss vs ablation.
Response: noisy oscillation, inner_mass 0.54-0.73 around parent (0.69 at 0); loss
range 0.35-0.77, no monotone shape. Best strength=0.09 -> loss=0.3482, inner=0.636.
Parent (strength=0) loss=0.5684 (single seed=0 unfavorable, see sw 9).
Morphology (strip): all 16 values morphologically indistinguishable -- vertical
2-3-mound clump, identical between strength=0 ablation and strength=0.12. No
emergence of streams, no crisping, no merging.
Verdict: FALSIFIED -- persistence is morphologically silent. Any "win" is within
seed-noise floor (sw 9: 0.35-0.86). FIFTH cell-side mechanism falsified after
sense_adapt (B7), align (B8), nucleation (B6), inhibitor (B9).
Knowledge update: Falsified Hypotheses += H1-B10. Mechanism falsified family.

## Batch 10 Sweep 1 — persistence.rho @ strength=0.03  [falsified]
Hypothesis: H2-B10 time-constant rho has productive optimum (slow vs fast memory).
Response: inner 0.48-0.68, loss 0.30-0.56, all noise. Best rho=0.35 -> loss=0.3034.
Morphology: all 16 cells visually identical -- same vertical 2-3-mound clump.
Verdict: FALSIFIED -- no time-constant effect. Confirms sweep 0 finding.

## Batch 10 Sweep 2 — spring.r_on FINE2 [0.218, 0.230]  [refines parent]
Hypothesis: H3-B10 fine refinement near sw5 best (0.224) finds sub-grid optimum.
Response: noisy, multiple inner_mass peaks at 0.219, 0.223-0.225, 0.227-0.228.
Best r_on=0.2255 -> loss=0.3278. Parent 0.224 -> 0.5684 (seed=0 unfavorable).
Morphology: uniform multi-mound vertical streaks across all values.
Verdict: INCONCLUSIVE -- small shift to 0.2255 but within noise floor. Adopt.

## Batch 10 Sweep 3 — spring.kadh FINE2 [50, 100]  [refines parent]
Hypothesis: H4-B10 fine refinement near sw6 best (75).
Response: noisy. Best kadh=65 -> loss=0.3267, inner=0.539. Parent 75 -> 0.5684.
Morphology: uniform 2-3 mound across all values; no over-compact regime visible
even at 100 (formerly catastrophic at legacy parent).
Verdict: INCONCLUSIVE shift down to 65 -- within noise. Adopt 65.

## Batch 10 Sweep 4 — relay.thr FINE2 [0.20, 0.28]  [refines + bimodal]
Hypothesis: H5-B10 thr fine refinement near sw9 best (0.23).
Response: loss minimum at thr=0.22 (0.3402); thr=0.245-0.255 spikes loss to
0.74-0.87 with inner_mass spike to 0.74. Bimodal -- clean low-thr regime
and disrupted high-thr regime.
Morphology: thr=0.245-0.255 strips visibly MORE compact/different -- these are
the over-tight single-blob regime resurfacing at high thr * joint-refined config.
Verdict: refines thr to 0.22; the high-thr "multi-knot" mode is no longer
operative at the new joint config -- its winning condition has been absorbed by
the joint refinements. Adopt 0.22.

## Batch 10 Sweep 5 — cell.n FINE2 [1300, 1550]  [refines parent]
Hypothesis: H6-B10 cell.n optimum near 1400.
Response: noisy. Best n=1410 -> loss=0.3548. Inner peaks at n=1395-1405.
Morphology: uniform multi-mound; mound count looks invariant with n.
Verdict: confirms n~1400-1410. Within noise. Adopt 1410.

## Batch 10 Sweep 6 — relay.gain [80, 240]  [bimodal optimum]
Hypothesis: H7-B10 relay.gain re-test at new parent.
Response: TWO minima -- gain=120 (0.3093) and gain=160 (0.3078). Parent 140 ->
0.5684 (between minima -- local maximum).
Morphology: uniform 2-3 mounds across all gains. No visible morphological
difference between gain=80 and gain=240.
Verdict: gain=140 sits on a local MAX of the seed=0 noise realisation; true
minimum shifts to ~160 (matches B9 sw 8 inconclusive trend). Adopt 160.

## Batch 10 Sweep 7 — camp.diffusion FINE2 [0.0001, 0.003]  [refines parent]
Hypothesis: H8-B10 D fine sweep around sw10 best (0.0004).
Response: best D=0.0005 -> loss=0.3027. D=0.0009 catastrophic (1.11), D=0.0010
also bad (0.87). Confirms strong low-D preference (Est #5).
Morphology: D=0.0009-0.0010 strips look MORE diffuse/scattered, confirming the
"high-D smears self-organisation" mechanism. D=0.0001-0.0005 all clean.
Verdict: nudge D to 0.0005. Confirms Est #5 in new regime.

## Batch 10 Sweep 8 — random_walk.strength [0, 0.020]  [flat with noise]
Hypothesis: H9-B10 rw fine refinement near sw11 best (0.009).
Response: best rw=0.006 -> loss=0.3449. Parent 0.009 -> 0.5684. Flat-ish.
Morphology: uniform 2-3 mound across the row.
Verdict: rw=0.006 marginally better. Adopt (within noise).

## Batch 10 Sweep 9 — seed @ NEW multi-knot parent  [noise floor measurement]
Hypothesis: H10-B10 measure noise floor at new parent.
Response: loss range 0.35-0.86, median ~0.48. Best seed=15 (0.3542), seed=7 (0.3488).
Parent seed=0 -> 0.5684 -- at 75th PERCENTILE of seed distribution (UNFAVORABLE).
Morphology: VARIES clearly seed-by-seed -- some seeds give single-blob, others
multi-mound (2-3-4 spots). The regime is truly bimodal in morphology under
stochastic init.
Verdict: noise floor confirmed at ~0.35-0.50 (better seeds reach 0.35). Parent
seed=0 is a BAD draw at the new parent (opposite of legacy where seed=0 was a
lucky 0.239 minimum). All single-sweep "wins" of ~0.30-0.40 are WITHIN noise.
Knowledge: Established #18 updated -- new-parent noise floor is comparable to
legacy-parent noise floor; the loss-surface "wins" of Batch 10 cannot be
adjudicated above noise EXCEPT for the inflow sweep (sw 10).

## Batch 10 Sweep 10 — inflow.rate @ persistence(strength=0.03, rho=0.3) joint
[**BATCH WINNER — Established #6/#7 CHALLENGED**]
Hypothesis: H11-B10 persistence enables fresh cells to integrate into existing
mounds (Est #6/#7 challenge).
Response: STRONG peak at rate=2.4 -> loss=0.2771, inner=0.559, n_final=1985.
Several near-equivalents: rate=1.0 (0.3555), rate=1.8 (0.3418), rate=3.2 (0.3419).
Loss broadly LOWER than ablation (rate=0 -> 0.4805).
Morphology: as rate climbs the cell-cloud densifies and remains cohesive; at
rate=2.0-2.4 the SIM-density strip shows a denser multi-mound that more closely
resembles REAL than rate=0. n grows 1410->1985 (overshoots REAL ~1413 final).
Verdict: SUPPORTED -- at the new multi-knot regime, inflow.rate=2.4 LOWERS loss
below ablation AND below ALL other sweep minima of batch. THIS IS THE FIRST
INFLOW WIN in 9 batches. CRITICAL OPEN: is the win caused by persistence, or by
the new multi-knot regime alone (i.e., would inflow.rate=2.4 at NEW PARENT
without persistence also win)? Must decouple in B11.
Knowledge: Established #6/#7 NOW QUESTIONED -- Batch 11 must include
inflow.rate sweep AT new parent WITH persistence.strength=0 (the decoupling
ablation). If ablation also wins -> Est #6/#7 falsified by new multi-knot
regime alone. If persistence necessary -> persistence rehabilitated.

## Batch 10 Sweep 11 — persistence.strength @ multi-knot-thr=0.27  [falsified]
Hypothesis: H12-B10 persistence * higher thr pushes morphology to MORE-mound.
Response: all loss 0.51-0.84; best strength=0.08 -> loss=0.5076. Much worse
than parent-thr regime.
Morphology: more diffuse / unraveled relative to sw 0; thr=0.27 destabilises
the joint-refined config.
Verdict: FALSIFIED -- joint thr=0.27 * persistence is WORSE than either alone.
Confirms thr=0.22-0.23 is the operating point.

## Batch 10 Sweep 12 — camp.decay [0.10, 0.42]  [flat, refines marginally]
Hypothesis: H13-B10 camp.decay re-test in new regime.
Response: loss flat band 0.35-0.55. Best decay=0.16 (0.3497), decay=0.42 (0.3632).
Morphology: uniform across.
Verdict: decay=0.20 still operative. Slight shift to 0.16 within noise.

## Batch 10 Sweep 13 — sense.gain [20, 80]  [refines parent]
Hypothesis: H14-B10 sense.gain at new parent -- multi-mound may shift it.
Response: best gain=45 (0.3140), 70 (0.3506), 50 (0.3387). Parent 40 -> 0.5684.
gain=20-24 spike inner_mass to 0.73 but loss high (over-compact knot -- same
failure mode as B5 sw 7).
Morphology: gain=20-24 strips show TIGHTER single blob; gain=45-70 multi-mound.
Verdict: optimum shifts slightly upward to ~45-50 in new regime. Adopt 45.

## Batch 10 Sweep 14 — vmax [0.055, 0.070]  [confirms Est #9]
Hypothesis: H15-B10 vmax aliasing in new regime.
Response: best vmax=0.058 (0.3197); vmax=0.062 catastrophic (0.7114) -- sharp
resonance still present. Parent 0.060 -> 0.5684.
Morphology: vmax=0.062 strip looks scattered, others clean.
Verdict: confirms Est #9; vmax optimum nudges to 0.058 in new regime.

## Batch 10 Sweep 15 — persistence.strength @ rho=0.6 (slow memory)  [falsified]
Hypothesis: H16-B10 longer persistence stabilises mound positions.
Response: best strength=0.05 -> loss=0.3368. Parent (strength=0) -> 0.5684.
Morphology: uniform with sw0 / sw11; no slow-memory regime emerges.
Verdict: FALSIFIED -- slower memory neither stabilises nor crisps. Persistence
in any (strength, rho) configuration is morphologically silent.

==================================================================
BATCH 10 SUMMARY (parent = multi-knot @ seed=0)

* Persistence FALSIFIED across THREE sweeps (sw 0, 11, 15 * multiple rho/thr).
  FIFTH cell-side mechanism falsified in succession (sense_adapt, align,
  nucleation, inhibitor, persistence). The cell-side-mechanism well looks
  empty -- what remains untested is a STRUCTURAL change (inflow restored).

* HUGE FINDING -- sw 10: persistence * inflow.rate=2.4 -> loss=0.2771,
  inner=0.559, n=1410->1985. FIRST inflow win in 9 batches. Established #6/#7
  ("no inflow can satisfy inner_mass AND n-growth under current loss") is now
  challenged. Must decouple in B11: run inflow.rate sweep at persistence=0
  to test if the new multi-knot regime ALONE rescues inflow.

* Noise floor at new parent (sw 9): 0.35-0.86, median ~0.48. Parent seed=0 ->
  0.5684 is at 75th percentile (BAD draw, opposite of legacy parent). Every
  single-axis "win" of batch (0.30-0.40) is within this noise EXCEPT sw 10
  rate=2.4 which has TWO orthogonal evidences (loss drop + n-growth).

* Marginal parameter shifts (within noise floor):
    r_on  0.224 -> 0.2255 | kadh 75 -> 65 | thr 0.23 -> 0.22 | gain 140 -> 160
    D 0.0004 -> 0.0005 | rw 0.009 -> 0.006 | sense.gain 40 -> 45
    vmax 0.060 -> 0.058 | decay 0.20 -> 0.16

* New PRIMARY PARENT for Batch 11: multi-knot regime + inflow.rate=2.4 (sw10
  winner config), keep persistence on (B11 ablation will decide if necessary).
  thr=0.22, n=1410, kadh=65, r_on=0.2255, gain=160, D=0.0005, rw=0.006,
  sense=45, vmax=0.058, decay=0.16, inflow.rate=2.4, persistence.strength=0.03.

Strategic pivot for Batch 11:
  1. DECOUPLE persistence from inflow win: inflow.rate sweep at strength=0.
  2. INVESTIGATE inflow rehabilitated: bias_to_camp, edge_band, n calibration.
  3. JOINT refine inflow.rate * {persistence=0/0.03, edge_band, bias_to_camp}.
  4. Re-test single-axis refinements at the NEW PARENT (with inflow on).


# Batch 11 (NEW METRIC: SSIM-on-image + g(r) + n_mounds) — 16 sweeps

## Batch 11 Sweep 0 — persistence.strength  [falsified — decoupled]
Hypothesis (H1-B11): persistence is the cause of the B10 inflow win; ablation (strength=0) at inflow.rate=2.4 should lose to strength=0.03 if persistence is necessary.
Response: FLAT-NOISY across [0, 0.12]. loss 0.92–1.01, no monotone trend; best=0.01 (loss=0.9229, inner=0.511); strength=0 ablation gives loss=0.938 — statistically INDISTINGUISHABLE from any non-zero value.
Morphology (from strip): all 16 panels show the SAME 2-knot blob — strength is morphologically invisible at every value, identical to ablation.
Verdict: FALSIFIED — persistence is unnecessary even with inflow on. The B10 sw10 inflow win was NOT caused by persistence.
Knowledge update: closes the H1-B11 decoupling test (Est #20 stays falsified); persistence DROPPED from B12 parent.

## Batch 11 Sweep 1 — inflow.rate [0, 4.5]  [partially falsifying Est #21]
Hypothesis (H3-B11): inflow.rate=2.4 win replicates under finer grid at multi-knot parent.
Response: FLAT-NOISY across [0, 4.5]. loss 0.92–1.03 (range 0.10 — within seed noise); best=3.2 (0.9232); rate=0 ablation=1.015 (within 1σ of best). No clear interior peak; high rates (3.0–4.5) very slightly better.
Morphology (from strip): all values produce the SAME 2-knot vertical clump morphology; the strip is visually monotone — fresh cells don't redistribute, they pile into the existing 2 knots.
Verdict: PARTIAL FALSIFICATION of Established #21 — the "inflow rescues n-growth" claim of B10 sw10 (loss=0.2771) was made under the OLD radial-MSE metric. Under the NEW SSIM-based metric, inflow is FLAT with no real win.
Knowledge update: RETRACT Established #21. The "inflow rehabilitation" was a metric-specific artifact.

## Batch 11 Sweep 2 — spring.kadh  [inconclusive — flat]
Hypothesis (H10-B11): kadh shifts with inflow on; refine around new parent kadh=65.
Response: flat-with-noise across [10, 130]. loss 0.92–1.10; best=35 (0.9214). High kadh (>100) DEGRADES loss monotonically (over-compacts into one blob).
Morphology: stable 2-knot mound across all kadh; at kadh>=100 cells start over-fusing into one denser blob.
Verdict: INCONCLUSIVE within noise; the only clean signal is "kadh>100 over-compacts" (consistent with Est #3). Tentatively adopt kadh=40 for B12 parent.
Knowledge update: refines parent kadh; no mechanism update.

## Batch 11 Sweep 3 — spring.r_on  [SUPPORTED — strongest signal of batch]
Hypothesis (H9-B11): r_on=0.224 holds under inflow.
Response: MONOTONE in two regimes. At r_on<0.16: under-aggregation (small scattered dots; loss 1.4–2.4). At r_on in [0.18, 0.24]: loss flat ~0.94 with inner_mass rising MONOTONICALLY 0.20 to 0.61. r_on=0.24 gives inner_mass=0.614 (EXACT match to REAL 0.606).
Morphology: striking visual progression — r_on=0.08–0.16 = dispersed scatter; r_on=0.20–0.24 = increasingly distinct, compact knots. r_on=0.226–0.24 show the closest morphology to REAL across the entire batch (multiple discrete bright spots, not one blob).
Verdict: SUPPORTED — adhesion REACH is the cleanest morphological lever in the model (RE-CONFIRMS Established #3); pushes parent r_on to 0.24.
Knowledge update: Established #3 strengthened; r_on=0.24 promoted to parent.

## Batch 11 Sweep 4 — camp.diffusion  [partially contradicts Est #5]
Hypothesis (H12-B11): low diffusion preference (Est #5) holds under inflow.
Response: flat-noisy across [0, 0.002]; best=0.002 (high end!) at loss=0.9199. Range only 0.06 — within noise.
Morphology: stable 2-knot blob at every diffusion; no visible morphological response in [0, 0.002].
Verdict: INCONCLUSIVE; Est #5 is no longer strongly supported under the new metric. Adopt diffusion=0.001 as neutral parent.
Knowledge update: Establish #5 flagged as WEAKENED under the new metric.

## Batch 11 Sweep 5 — cell.n  [INSTABILITY at n>=3500]
Hypothesis (H6-B11): cell.n x inflow joint — lower initial n with influx may give better match to REAL n_final.
Response: loss flat across n in [800, 3200] (0.90–1.06); SIMULATION FAILS (NaN) at n>=3500. Best=800 (loss=0.9001 — the BEST of the entire batch). At n=800 + inflow.rate=2.4, n_final=1385 → matches REAL n_final approx 1413 well.
Morphology: n=800-1400 = sparse 2 mound; n=2000–3200 = denser but still 2-3 mounds; never 8 mounds.
Verdict: SUPPORTED weakly — n=800 + influx is the most biologically credible parent. NaN at n>=3500 is an engine-capacity limit.
Knowledge update: Adopt cell.n=800 as parent; n_final approx 1385 matches REAL.

## Batch 11 Sweep 6 — relay.thr  [contradicts Est #11 under new metric]
Hypothesis (H8-B11): relay.thr shifts with inflow on.
Response: MONOTONE-INCREASING loss with thr. loss ~0.92 at thr=0.18–0.22, climbs to 1.19 at thr=0.50. Best=0.22 (loss=0.9214).
Morphology: thr=0.18–0.22 = dense 2-knot; thr=0.26+ = increasingly sparse; thr=0.40+ = nearly empty (relay barely fires). The "multi-knot regime" (Est #11) is NOT multi-mound under the new metric — it is just diluted aggregation.
Verdict: PARTIALLY FALSIFIES Established #11 — the high-thr regime is morphologically WORSE under SSIM-on-image. Parent thr stays at 0.22.
Knowledge update: Est #11 partially retracted (multi-knot region no longer multi-mound under new metric).

## Batch 11 Sweep 7 — inflow.bias_to_camp  [FALSIFIED AGAIN at new metric]
Hypothesis (H4-B11): bias_to_camp rehabilitates under inflow + multi-knot parent (Est #21).
Response: flat-noisy [0, 12]; loss 0.92–1.05; best=4.0 (0.9218). Identical to bias=0 (loss=0.97) within noise.
Morphology: 2-knot blob at every bias value; no streaming, no biased catchment.
Verdict: FALSIFIED — bias_to_camp provides no rescue under the new metric either. Drop from parent.
Knowledge update: Closes the H1-B5/H1-B4 re-test definitively; biased inflow stays falsified.

## Batch 11 Sweep 8 — inflow.edge_band  [FALSIFIED AGAIN]
Hypothesis (H5-B11): edge_band rehabilitates under inflow + new parent.
Response: flat-noisy [0, 0.5]; loss 0.93–1.12; best=0.4 (0.9315). No structural effect.
Morphology: identical 2-knot blob at every edge_band value.
Verdict: FALSIFIED — boundary-source inflow provides no morphology rescue.
Knowledge update: H1-B5 stays falsified; close all inflow-spatial-prior hypotheses.

## Batch 11 Sweep 9 — seed (cell init seed)  [noise floor measurement]
Hypothesis (H7-B11): re-measure noise floor at the B11 parent (with inflow on).
Response: loss spread 0.925–1.05 (Delta=0.13), inner_mass spread 0.38–0.63. Best seed=5 (0.925); the parent seed=0 sits at loss=0.986 (median of distribution).
Morphology: across 16 seeds, all panels show the same 2-knot morphology with varying positions — no seed produces multi-mound.
Verdict: SUPPORTED — noise floor under new metric is ~0.10 in loss, ~0.20 in inner_mass. All B11 single-axis "wins" (Delta loss=0.01–0.05) are within noise.
Knowledge update: NEW seed-noise quantification under SSIM metric: sigma_loss approx 0.04, range 0.13.

## Batch 11 Sweep 10 — secrete.rate  [SHARP INSTABILITY — non-flat]
Hypothesis: secrete.rate flat around parent=7.
Response: CATASTROPHIC SPIKE at rates 4 and 5 (loss=3.24, 6.08); flat ~0.96–1.28 elsewhere; best=7 (0.9555). At rates >=14, loss climbs back to 1.12–1.28; morphology nearly empty.
Morphology: rates 2–3 = sparse small dots; rates 4–5 = EXPLOSIVE DISPERSION (full-FOV diffuse cloud — the field strongly overrides chemotaxis); rates 6–10 = stable 2-knot blob; rates >=14 = collapse to a single dim spot (over-secretion drowns the gradient).
Verdict: SUPPORTED — secrete.rate has a NARROW working band [6, 10]. Rates 4-5 trigger a coherent dispersion failure-mode.
Knowledge update: NEW Established candidate — secrete.rate is a critical control parameter with working band [6, 10]; out-of-band gives qualitatively distinct failure modes.

## Batch 11 Sweep 11 — camp.decay  [boundary identified]
Hypothesis: camp.decay broader range.
Response: flat ~0.92 across decay in [0.10, 0.80]; sharp degradation at decay>=1.0 (loss 1.74, 4.14, 10.1, 2.0). Best=0.18 (0.9171).
Morphology: stable 2-knot for decay<=1.0; complete dispersion at decay>=1.5 (the activator decays before sustained relay can form).
Verdict: SUPPORTED — working band confirmed [0.10, 0.80]; parent decay=0.16 confirmed; not critical inside band.
Knowledge update: refines working-bands map; decay is bounded above at 1.0.

## Batch 11 Sweep 12 — relay.gain  [flat — refined]
Hypothesis (H11-B11): re-test gain shift under inflow.
Response: flat-noisy across [60, 500]; range 0.92–1.11; best=400 (0.9215), second-best=260 (0.9343); within noise.
Morphology: all gain values produce the same 2-knot blob; high-gain (400-500) has slightly denser mounds.
Verdict: INCONCLUSIVE within noise. Tentatively adopt gain=200 as parent.
Knowledge update: relay.gain has a wide flat optimum [100, 400] under the new metric.

## Batch 11 Sweep 13 — random_walk.strength  [flat]
Hypothesis (H13-B11): rw shift under inflow.
Response: flat-noisy across [0, 0.07]; best=0.022 (0.9246); rw=0 ablation=0.9376 (within noise).
Morphology: identical 2-knot morphology at every rw value.
Verdict: INCONCLUSIVE — random_walk is morphologically silent. Adopt rw=0.01 as neutral.
Knowledge update: rw confirmed flat across [0, 0.07].

## Batch 11 Sweep 14 — inflow.rate HIGH [3, 15]  [saturation regime]
Hypothesis (H14-B11): inflow.rate=2.4 is in a wider win-band; high-rate saturation regime test.
Response: flat-monotone-down to ~rate=7.5 (loss=0.9144 — the SECOND-best of the batch), then plateau. Range 0.92–1.05.
Morphology: rates 3–10 = stable 2-knot blob (more cells → slightly denser knots); rates >=10 = mound diffusion (cells flooding FOV faster than they aggregate).
Verdict: SUPPORTED weakly — high-rate plateau exists; the [3, 8] band is preferred.
Knowledge update: extends inflow working band; tentatively rate=4 for B12 parent.

## Batch 11 Sweep 15 — sense.gain  [SUPPORTED — monotone]
Hypothesis (H16-B11): stronger chemotaxis helps integrate fresh cells.
Response: MONOTONE-DECREASING loss from gain=10 (loss=1.12) to gain>=80 (loss ~0.92, plateau). Best=80 (0.9229).
Morphology: gain=10–25 = sparse scattered dots (chemotaxis too weak); gain=40+ = compact 2-knot; gain=80+ visually identical to gain=120 (saturated).
Verdict: SUPPORTED — chemotaxis IS necessary (low gain falsified), plateau above gain=60. Parent moves to sense.gain=80.
Knowledge update: chemotaxis necessity re-confirmed; new working floor gain >= 60.

## Batch 11 — summary

- Best loss: cell.n=800 → 0.9001 (sw 5); also strong: inflow.rate=7.5 (sw 14) 0.9144; r_on=0.24 (sw 3) 0.9341 (inner=0.614 EXACT REAL match); sense.gain=80 (sw 15) 0.9229. ALL within seed-noise floor (Delta=0.10).
- The new SSIM-based metric is SATURATED at ~0.92–1.00 across nearly every single-parameter sweep. The dominant failure mode is morphological: model produces 2-3 compact mounds; REAL has ~8 distinct mounds. No single-axis parameter shift can close this gap — the loss surface is dominated by mound-COUNT mismatch, which has no parameter lever.
- DECOUPLING TEST RESOLVED (sw 0 + sw 1): persistence is NOT the cause of the B10 inflow win; the inflow win itself does not replicate under the new metric. Est #21 RETRACTED. Persistence DROPPED.
- Two CLEAN morphological signals: (a) spring.r_on=0.24 (sw 3) — monotone inner_mass → REAL=0.606 — strengthens Est #3; (b) sense.gain >= 60 plateau (sw 15) — chemotaxis necessary.
- Two NEW failure-mode boundaries mapped: secrete.rate in [6, 10] (sw 10); camp.decay <= 1.0 (sw 11).
- inflow.bias_to_camp and inflow.edge_band falsified AGAIN under new metric (sw 7, 8) — these inflow-spatial-priors permanently dead.
- B11 PARENT FOR B12: spring.r_on=0.24, cell.n=800, inflow.rate=4, sense.gain=80, secrete.rate=7, camp.decay=0.18, relay.gain=200, camp.diffusion=0.001, spring.kadh=40, random_walk.strength=0.01, persistence DROPPED. dt=0.5, vmax=0.058. Inflow:bias=0, edge_band=0.
- KEY INSIGHT: the model's 2-3-mound morphology ceiling persists under EVERY parameter combination tested across 11 batches. The new SSIM-based metric makes this visible as a flat ~0.92 loss floor with no parameter lever. The morphology gap is a STRUCTURAL property of the model, not a parameter-precision issue.

## Batch 12 Sweep 0 — seed  [noise-floor recalibration]
Hypothesis (H1-B12): re-measure noise floor at NEW clean B12 parent (no persistence, r_on=0.24, n=800, sense.gain=80, inflow=4).
Response: loss spread 0.92–1.63 (range 0.71, σ≈0.18), inner_mass 0.32–0.50; best seed=1 (loss=0.9209), parent seed=0 sits at 1.0123 (median region); one outlier seed=6 at 1.6257.
Morphology: every seed shows a SINGLE central blob (sometimes loose, sometimes tight); NO seed produces multi-mound. The 2-knot morphology of B11 is LOST at the new parent (inflow=4 dilutes the multi-knot tendency).
Verdict: SUPPORTED — noise floor under new parent is wider (σ≈0.18 vs B11 σ≈0.04). Single-parameter "wins" of Δloss ≤ 0.18 are within noise.
Knowledge update: NEW noise floor at B12 parent σ_loss≈0.18 (broader than B11). All single-axis "wins" this batch are within noise unless Δloss > 0.18.

## Batch 12 Sweep 1 — spring.r_on  [SUPPORTED — best of batch]
Hypothesis (H2-B12): refine r_on around 0.24 morphological optimum.
Response: inner_mass climbs 0.22 (r_on=0.18) → 0.20 → 0.33 → step → plateau 0.45–0.50 at r_on≥0.22. Loss has a shallow minimum at r_on=0.245 (loss=0.8997 — BEST OF BATCH); r_on=0.26 outlier at 1.39.
Morphology: r_on=0.18–0.20 = sparse scatter (no compaction); r_on=0.22–0.245 = compact single blob, with at r_on=0.245–0.26 a hint of 2-3 sub-blobs within the central mass.
Verdict: SUPPORTED — adhesion REACH still the cleanest morphological lever (Est #3 re-re-re-confirmed). Best at r_on=0.245 (loss 0.90, ~11% below parent 1.01; within sw 0 noise but on the favourable edge).
Knowledge update: Est #3 stays solid; adopt r_on=0.245 as new parent.

## Batch 12 Sweep 2 — cell.n  [flat-with-noise + capacity limit re-confirmed]
Hypothesis (H3-B12): more cells → more distinct mounds at r_on=0.24.
Response: flat-noisy 0.94–1.50 across n∈[400, 3400]; best=1000 (loss=0.9351). n=3400 → NaN (engine capacity limit, same as B11 sw 5).
Morphology: SINGLE central blob at every n; more cells just enlarges/intensifies the same blob, never splits.
Verdict: FALSIFIED — more cells DO NOT break the single-attractor at r_on=0.24. Sharp adhesion + many cells = bigger central blob.
Knowledge update: Est #14 (multi-knot scales with cell count) RETRACTED at the B12 sharp-r_on parent — cell.n is independent of mound count.

## Batch 12 Sweep 3 — sense.gain  [flat — Est #28 holds]
Hypothesis (H4-B12): plateau above gain=60 + saturation reversal test.
Response: flat-noisy 0.92–1.32 across gain∈[40, 200]; best=150 (0.9228). No saturation reversal at high gain.
Morphology: SINGLE blob at all values; visually indistinguishable.
Verdict: INCONCLUSIVE within noise; Est #28 (gain≥60 plateau) re-confirmed; no productive direction above 80.
Knowledge update: gain=80 parent retained.

## Batch 12 Sweep 4 — secrete.rate  [SHARP catastrophic dispersion zone mapped]
Hypothesis (H5-B12): fine-map the explosive dispersion failure (B11 sw 10).
Response: loss SPIKES from 0.98 (rate=3.0) to 1.25 (3.3) to 3.04 (3.6) to 14.2 (4.4) — catastrophic dispersion. Recovers to 1.05 at rate=6.5. Best=3.0 (0.9753), but morphology terrible at rates 4–6.
Morphology: rate=3.0 = sparse single blob; rates 3.6–6.0 = FULL-FOV DIFFUSE SAND (explosive dispersion — cells overwhelmed by their own field, chemotaxis nullified); rate=6.5 = compact blob again.
Verdict: SUPPORTED — secrete.rate ∈ [3.6, 6.0] is a SHARP qualitatively-distinct failure regime; outside this band the system aggregates. The dispersion mode does NOT spontaneously convert to multi-mound. Failure-mode is not a mound-multiplier.
Knowledge update: NEW failure-mode boundary: secrete.rate ∈ [3.6, 6.0] is the explosive-dispersion zone. Cannot be reached productively at the current parent.

## Batch 12 Sweep 5 — relay.thr  [SHARP catastrophic below 0.15; high-thr sparse]
Hypothesis (H6-B12): high-thr regime gives multi-mound at sharp r_on.
Response: catastrophic at thr=0.10 (5.48) & 0.14 (3.22) — relay over-fires; plateau ~0.91–1.05 for thr∈[0.16, 0.34]; best=0.24 (0.9078). At thr≥0.38, loss creeps up (1.05–1.13); thr=0.42 produces inner_mass spike to 0.78.
Morphology: thr<0.16 = uniform diffuse activity (over-firing); thr=0.16–0.30 = single blob; thr=0.34 = blob + sparse satellites; thr=0.38–0.50 = sparse few small mounds (3-4 distinct spots visible at thr=0.42), but with much lower density. The high-thr regime is morphologically MORE multi-mound than parent at r_on=0.245.
Verdict: PARTIALLY SUPPORTED — high-thr regime (thr≥0.38) produces visually-distinct multi-spot morphology but the SSIM loss does not reward it (density per mound too low). NEW hypothesis: combine high-thr + high cell.n + sharp r_on to get DENSE multi-mound.
Knowledge update: Est #11 partially RE-INSTATED at sharp r_on parent: thr∈[0.38, 0.50] gives visible multi-spot (≥3 mounds), though sparse. Promising direction.

## Batch 12 Sweep 6 — spring.kadh  [flat — high-kadh slight preference]
Hypothesis (H7-B12): adhesion AMPLITUDE refine at sharp r_on.
Response: noisy 0.92–1.62 (kadh=10 outlier at 1.62 — too-weak adhesion); flat ~0.92–1.04 for kadh∈[20, 200]. Best=200 (0.9209), tied with kadh=70 (0.9353) and kadh=130 (1.17) — flat within noise.
Morphology: kadh=10–20 = loose scatter; kadh=30+ = compact blob; kadh=200 = tight single blob (no morphological gain).
Verdict: INCONCLUSIVE — kadh is morphologically silent above floor of 30. Parent kadh=40 retained.
Knowledge update: kadh working band reaffirmed; no preferred value within [30, 200].

## Batch 12 Sweep 7 — camp.diffusion  [low-diffusion preference REAFFIRMED]
Hypothesis (H8-B12): re-test Est #5 at clean parent.
Response: loss flat 0.92–1.49 across [0.0001, 0.012]; best=0.0008 (0.9165). High-diffusion (0.003, 0.004) DEGRADES loss to 1.15-1.49.
Morphology: low diffusion (≤0.001) = compact blob; high diffusion (≥0.003) = more diffuse blob, less focused.
Verdict: SUPPORTED — Est #5 RE-INSTATED at the new clean parent (was weakened in B11). Low diffusion (≤0.001) preferred.
Knowledge update: Est #5 promoted back to strong; D=0.0008 best, working band [0.0001, 0.0025].

## Batch 12 Sweep 8 — inflow.rate  [shallow optimum at 3.0; over-dilution at high rates]
Hypothesis (H9-B12): inflow flat at new parent.
Response: shallow well 0.92–1.04 across [0.5, 4.5]; best=3.0 (0.9225). Loss climbs at rate>=5 (1.22–1.61 at rate=10). Inner_mass DECREASES with rate: 0.64 at rate=0 (matches REAL 0.61!) to 0.34 at rate=10.
Morphology: rate=0 = single compact mound (inner_mass=0.64); rates 1–4 = looser single mound; rates 6+ = mound dispersing into ambient sand.
Verdict: PARTIALLY SUPPORTED — inflow has a SHALLOW optimum around rate=3.0 (NOT flat as B11 claimed). Over-influx (>=6) actively destroys morphology. Adopt inflow.rate=3.0 (B12 sw 8 best) as new parent.
Knowledge update: REFINEMENT of Est #24 — inflow has a shallow optimum at rate~3.0; rate>6 is over-dilution failure mode.

## Batch 12 Sweep 9 — camp.decay  [monotone-up degradation; low-decay safe band]
Hypothesis (H10-B12): camp.decay FINE at new parent.
Response: flat 0.93–1.05 across [0.05, 0.40]; degrades to 1.41 (0.5), 1.07 (0.6), 1.63 (0.7), 1.55 (0.8). Best=0.05 (0.9387).
Morphology: low decay (0.05) = persistent gradient → compact blob; decay 0.5–0.8 = field dies before sustained aggregation → diffuse cloud.
Verdict: SUPPORTED — working band [0.05, 0.40] confirmed; degradation above 0.4. Parent decay=0.18 retained (mid-band).
Knowledge update: refines working band; decay upper bound 0.40 under new parent (tighter than B11's 0.80).

## Batch 12 Sweep 10 — vmax  [dt×vmax aliasing CONFIRMED (Est #9)]
Hypothesis (H11-B12): re-test Est #9 aliasing.
Response: SHARP peaks at vmax=0.045 (1.77) and vmax=0.075 (1.87); minimum at vmax=0.062 (0.9134). Flat-bumpy ~0.91–1.10 between.
Morphology: vmax=0.045 = sparse scatter (under-displacement); vmax=0.062 = compact blob; vmax=0.075 = oversteps into chaos.
Verdict: SUPPORTED — Est #9 (dt×vmax aliasing) re-re-confirmed at new parent. Adopt vmax=0.062 as new parent.
Knowledge update: Est #9 strengthened; working band [0.052, 0.072] with optimum 0.062.

## Batch 12 Sweep 11 — relay.gain  [Est #4 RE-CONFIRMED + ringing at gain=20]
Hypothesis (H12-B12): wide gain sweep including ablation.
Response: gain=0 (ablation) loss=1.26 — degraded but not catastrophic (sparse scatter). gain=20 CATASTROPHIC at 14.68 (relay rings). Plateau ~0.92–1.05 for gain∈[40, 600]; best=300 (0.9224); inner_mass climbs from 0.19 at gain=0 to 0.54 at gain=600.
Morphology: gain=0 = sparse scattered dots (no aggregation); gain=20 = explosive over-firing dispersal; gain=40+ = single blob; gain=600 = denser tighter blob.
Verdict: SUPPORTED — Est #4 (relay necessity) RE-CONFIRMED. New finding: gain=20 is a sharp ringing instability between ablation and stable regime.
Knowledge update: Est #4 strong; NEW failure-mode at gain∈[10, 30] (relay ringing).

## Batch 12 Sweep 12 — random_walk.strength  [flat — morphologically silent]
Hypothesis (H13-B12): rw flat at new parent.
Response: flat-noisy 0.92–1.24 across [0, 0.05]; best=0.004 (0.917).
Morphology: identical single blob at every rw value.
Verdict: INCONCLUSIVE within noise; rw morphologically silent. Parent rw=0.01 retained.
Knowledge update: rw confirmed flat across wide range; no morphological role.

## Batch 12 Sweep 13 — cell.n LOW × inflow=6  [no morphological trajectory effect]
Hypothesis (H14-B12): start small, grow fast → discrete mounds during growth.
Response: noisy 0.93–1.51 across n∈[200, 2000]; best=1100 (loss=0.9309). No clear trajectory effect.
Morphology: low n + high inflow = scattered then merging to single blob; identical end-state morphology across n.
Verdict: FALSIFIED — starting low and growing fast does NOT produce a different morphology trajectory. Final state is the same single blob.
Knowledge update: morphological trajectory (start-low-grow-fast) is NOT a mound-multiplier.

## Batch 12 Sweep 14 — relay.eps  [flat — morphologically silent]
Hypothesis (H15-B12): refractory time-constant breaks ceiling at sharp r_on.
Response: flat-noisy 0.93–1.31 across [0.005, 0.10]; best=0.09 (0.9337).
Morphology: identical single blob across eps; no morphological response.
Verdict: FALSIFIED — relay.eps morphologically silent at new parent. Refractory tuning is NOT a mound-multiplier (re-confirms B7 sw 14 in different regime).
Knowledge update: relay.eps confirmed flat; drop from B13.

## Batch 12 Sweep 15 — n_frames  [equilibration plateau reached early]
Hypothesis (H16-B12): longer simulation breaks 2-3-mound ceiling.
Response: flat-monotone-up inner_mass 0.53→0.58 across n_frames∈[300, 800] (plateau by 400); loss flat 0.93–1.06.
Morphology: identical single-blob morphology across all frame counts; ZERO progression toward multi-mound with extended simulation.
Verdict: FALSIFIED — simulation length is NOT the limiting factor. The 2-3-mound (now SINGLE-blob at this parent) ceiling is a STRUCTURAL property reached by frame 350-400; running longer changes nothing.
Knowledge update: morphology gap is reached at equilibrium and is structural. n_frames=400 sufficient.

## Batch 12 — summary

- BEST OF BATCH: spring.r_on=0.245 → loss=0.8997, inner=0.485, n_final=1764 (sw 1). Within seed-noise floor (σ≈0.18) of parent 1.0123, but on the favourable edge.
- Noise floor under B12 parent is BROADER (σ≈0.18) than B11 (σ≈0.04) — the inflow=4 parent is noisier than B11's inflow=2.4 parent.
- The B12 parent (sharp r_on=0.24 + n=800 + inflow=4) produces SINGLE-BLOB morphology at every parameter axis tested — the 2-knot morphology of B11 is LOST. The morphology gap is WIDER (1 mound vs REAL 8) at the new parent.
- Re-confirmed: Est #3 (r_on lever), Est #4 (relay necessity), Est #5 (low diffusion), Est #9 (dt×vmax aliasing), Est #18/22 (large seed noise floor), Est #23 (flat surface).
- Falsified at B12 parent: Est #14 (multi-knot scales with n) — at sharp r_on=0.24, more cells just bigger blob; n_frames extension does nothing.
- NEW failure-mode boundaries: relay.gain∈[10, 30] = ringing; secrete.rate∈[3.6, 6.0] = explosive dispersion (sharper than B11); camp.decay > 0.40 = field dies; inflow.rate > 6 = over-dilution; vmax outside [0.052, 0.072] = aliasing collapse.
- ONE PROMISING DIRECTION: relay.thr∈[0.38, 0.50] (sw 5) shows visible 3-4 sparse spots — partially re-instates Est #11 at sharp r_on. The morphology is the closest to REAL multi-mound seen this batch, though sparse.
- B12 PARENT FOR B13: r_on=0.245, cell.n=1000, vmax=0.062, inflow.rate=3.0 (adopted as small improvements); all else unchanged (sense.gain=80, secrete.rate=7, camp.decay=0.18, relay.gain=200, camp.diffusion=0.0008, spring.kadh=40, random_walk.strength=0.01, relay.thr=0.22).
- KEY INSIGHT: 12 batches in, NO parameter axis in the existing operator set breaks the single-blob ceiling at the sharp-r_on regime. Structural addition is needed. B13 plan: add `sense_sat` (Hill-saturated chemotaxis — cells near a mound stop tracking; cells farther out can form new mounds). Ablation = c_sat=1e6 → identical to plain sense.

## Batch 13 Sweep 0 — seed  [noise-floor recalibration]
Hypothesis (H1-B13): map seed-noise floor under new B13 parent (r_on=0.245, n=1000, vmax=0.062, inflow=3.0, sense_sat ablated c_sat=1e6).
Response: flat-noisy 0.929–1.108 across 16 seeds; median ~0.97, sigma~0.055; best seed=7 (0.929) or seed=11 (0.929). REAL inner_mass=0.61, mean sim inner ~0.44.
Morphology (from strip): SINGLE fuzzy oval blob at every seed — no multi-mound at any seed. Identical morphological mode across all 16 noise realisations.
Verdict: SUPPORTED — noise floor at new parent is sigma~0.05 (tighter than B12's sigma~0.18; lower inflow=3.0 reduces variability vs B12's inflow=4.0). Single-blob is the deterministic mode of the c_sat=1e6 (ablation) parent.
Knowledge update: B13 noise floor = sigma~0.05; treat any |delta-loss| <= 0.10 as within noise. The B13 ablation parent shows ZERO multi-mound across all 16 seeds.

## Batch 13 Sweep 1 — sense_sat.c_sat (WIDE @ parent)  [BREAKTHROUGH — supported]
Hypothesis (H2-B13): Hill-saturated chemotaxis breaks the single-blob ceiling.
Response: monotone-up loss from ablation 1.055 to c_sat=0.01 (1.236); best loss tied at c_sat=2 (0.930) — within noise of ablation. inner_mass peaks sharply at c_sat=0.5 (0.915 — tight knot) then crashes to c_sat=0.2 multi-mound regime (0.662 closest to REAL 0.606 of any config in 13 batches).
Morphology (from strip): THIS IS THE FINDING. c_sat=1e6/5/2 -> single fuzzy blob (ablation). c_sat=1 -> tight central blob + small companions. c_sat=0.5 -> ONE TIGHT KNOT. c_sat=0.3 -> multi-spot ~5 sparse mounds. c_sat=0.2 -> 5-6 distinct compact mounds, inner_mass=0.662 (REAL=0.606) — VISUALLY CLOSEST to REAL the model has produced. c_sat=0.15 -> 5-6 mounds. c_sat<=0.1 -> very sparse many tiny spots.
Verdict: SUPPORTED — the FIRST mechanism in 13 batches to break the single-blob/2-knot ceiling under the new SSIM metric. Loss does NOT reward it (sparse spots are SSIM-penalised), but morphology is unambiguous.
Knowledge update: NEW Established Principle — sense_sat with c_sat in [0.15, 0.30] BREAKS the single-blob ceiling. Inner_mass crossover with REAL at c_sat ~ 0.20. Densification is the next frontier.

## Batch 13 Sweep 2 — sense_sat.sat_n @ c_sat=0.1  [Hill exponent — partial]
Hypothesis (H3-B13): Hill exponent controls saturation sharpness.
Response: monotone-up loss from sat_n=0.5 (0.923) to sat_n=15 (1.230). inner_mass monotone-down.
Morphology: sat_n=0.5 -> single blob; sat_n=0.75-1.0 -> tight knot + companions; sat_n=1.25 -> tight knot; sat_n>=1.5 -> SPARSE multi-spot (4-7 tiny mounds); sat_n=15 -> very sparse.
Verdict: SUPPORTED with caveat — sat_n~2 is in the sparse-multi-spot regime; lower sat_n=1.25-1.5 may give denser multi-spot. High sat_n over-disperses.
Knowledge update: sat_n in [1.25, 2.5] together with c_sat in [0.15, 0.30] is the densification axis.

## Batch 13 Sweep 3 — sense_sat.c_sat @ r_on=0.20  [falsified densifier]
Hypothesis (H4-B13): looser adhesion + saturation -> denser multi-mound.
Response: flat-noisy 0.94-1.27 across c_sat; loss best at c_sat=2 (0.942). inner_mass low (0.06-0.32).
Morphology: at r_on=0.20 the ablation already shows MULTI-MOUND (2-3 spots — legacy multi-knot returns). Adding saturation FURTHER disperses. No densification.
Verdict: FALSIFIED — looser adhesion does NOT densify; it over-disperses the multi-mound.
Knowledge update: keep r_on>=0.225; looser adhesion erodes mound integrity.

## Batch 13 Sweep 4 — sense_sat.c_sat @ cell.n=2000  [falsified densifier]
Hypothesis (H5-B13): more cells -> denser per mound.
Response: monotone-up loss from ablation 1.056 to c_sat=0.15 (1.331).
Morphology: at n=2000 + low c_sat, very sparse tiny spots. Extra cells -> more spots, not more cells per spot.
Verdict: FALSIFIED — cell.n alone is NOT a densifier under sense_sat.
Knowledge update: cell.n × sense_sat does not densify; need joint r_on/kadh/relay levers.

## Batch 13 Sweep 5 — spring.r_on FINE @ parent (c_sat=1e6)  [Est #3 RE-CONFIRMED]
Hypothesis (H8-B13): r_on=0.245 confirmed at new parent.
Response: monotone-up inner_mass 0.252 (r_on=0.20) -> 0.847 (r_on=0.30); crossover at r_on~0.235. Loss bowl: best r_on=0.225 (0.965); rises to ~1.15 at r_on>=0.235.
Morphology: r_on=0.20-0.225 -> multi-knot (3-4 sub-clusters); r_on=0.23-0.245 -> consolidating; r_on>=0.25 -> single tight knot.
Verdict: SUPPORTED — Est #3 re-re-confirmed. r_on=0.225 is the multi-knot pre-collapse point at new ablation parent.
Knowledge update: Adopt r_on=0.225 for new parent (multi-knot fold pre-collapse).

## Batch 13 Sweep 6 — relay.thr @ parent  [Est #11 partially re-confirmed]
Hypothesis (H7-B13 variant): high relay.thr -> multi-spot.
Response: best thr=0.24 (0.928); rises to 1.27 at thr=0.60. inner_mass monotone-up 0.376 -> 0.972.
Morphology: thr=0.20-0.32 -> diffuse 1-3 mounds; thr=0.34-0.38 -> 3-4 sparse mounds; thr>=0.42 -> 2-3 tiny isolated spots.
Verdict: PARTIALLY SUPPORTED — high-thr produces sparse multi-spot; loss penalises. Best loss at thr=0.24.
Knowledge update: Two multi-mound regimes — sense_sat (denser, REAL-like) vs high relay.thr (sparser). Sense_sat wins for densification.

## Batch 13 Sweep 7 — spring.r_on FINE2 @ parent  [Est #3 sharpened]
Hypothesis (H8-B13 variant): finer r_on grid.
Response: noisy 0.92-1.73; outlier spike at r_on=0.215 (1.73). Local minimum at r_on=0.265 (0.922).
Morphology: 2-cluster multi-knot at r_on=0.20-0.225; single knot above r_on>=0.235.
Verdict: r_on lever silent without sense_sat.
Knowledge update: r_on × c_sat interact non-additively. Productive direction is (r_on=0.225 + sense_sat=on).

## Batch 13 Sweep 8 — inflow.rate @ parent  [flat — Est #24 re-confirmed]
Hypothesis (H9-B13): inflow=3.0 at shallow optimum.
Response: flat-noisy 0.93-1.10; spike at rate=5 (1.85). Best rate=4.5 (0.929).
Morphology: identical fuzzy single blob at every rate.
Verdict: re-confirms Est #24 — inflow flat under SSIM.
Knowledge update: working band [0, 5]; drop inflow.rate refinement.

## Batch 13 Sweep 9 — cell.n @ parent  [flat — Est #30 re-confirmed]
Hypothesis (H10-B13): cell.n fine refinement.
Response: flat-noisy 0.93-1.52; engine NaN-like spike at n=2500 (1.52). Best n=900 (0.927).
Morphology: identical fuzzy single blob across all n.
Verdict: re-confirms Est #30. NaN ceiling at n>=2500 (LOWER than B12's n>=3400 due to kadh=40).
Knowledge update: Working band [600, 2300]; drop cell.n single-axis refinement.

## Batch 13 Sweep 10 — camp.diffusion @ c_sat=0.1  [Est #5 RE-CONFIRMED under saturation]
Hypothesis (H11-B13): does saturation change low-diffusion preference?
Response: flat 1.20-1.24 across full range; best diff=0.0004 (1.196).
Morphology: at c_sat=0.1, EVERY diffusion value shows sparse multi-spot (5-8 tiny mounds). Saturation dominates morphology.
Verdict: SUPPORTED — low diffusion still preferred; sense_sat further flattens the surface.
Knowledge update: camp.diffusion=0.0004-0.0008 OK under sense_sat; no refinement needed.

## Batch 13 Sweep 11 — secrete.rate @ c_sat=0.1  [SATURATION REGULARIZES]
Hypothesis (H12-B13): saturation shifts secrete failure-mode boundary.
Response: flat-noisy 1.19-1.26 across [2, 12]; best=7.5 (1.193). NO catastrophic dispersion zone (the B11 sw 10 spike at rate=4-5 is ABSENT).
Morphology: sparse multi-spot at every rate; consistent ~6-8 small mounds.
Verdict: SUPPORTED — sense_sat REGULARIZES the secrete failure-mode. The Est #29 explosive-dispersion band at secrete in [3.6, 6.0] DISAPPEARS under saturation. NEW finding.
Knowledge update: NEW — sense_sat eliminates the secrete.rate dispersion catastrophe.

## Batch 13 Sweep 12 — vmax @ parent  [Est #9 RE-RE-CONFIRMED]
Hypothesis (H13-B13): vmax aliasing optimum.
Response: noisy 0.93-1.77 across [0.055, 0.072]; best vmax=0.061 (0.927); catastrophic at vmax=0.069-0.070 (1.74, 1.77) and vmax=0.055 (1.52).
Morphology: single blob at every vmax; off-band shows fragmented dispersal.
Verdict: SUPPORTED — Est #9 re-confirmed across 4 batches. Working band [0.057, 0.068].
Knowledge update: Adopt vmax=0.061. Marginal.

## Batch 13 Sweep 13 — relay.gain @ c_sat=0.1  [SENSE_SAT REPLACES RELAY]
Hypothesis (H14-B13): relay × saturation interaction.
Response: flat 1.18-1.27 across [0, 600]; best gain=60 (1.176). B12 sw 11 ringing zone (gain in [10, 30]) STILL appears (gain=20 -> 1.179, gain=40 -> 1.27).
Morphology: gain=0 (ablation) -> ~5-6 sparse mounds (multi-mound EMERGES even at zero relay under sense_sat); gain=20 -> SINGLE tight knot (ringing kills multi-mound); gain>=60 -> sparse multi-mound restored.
Verdict: BREAKTHROUGH — at c_sat=0.1 the multi-mound survives at relay.gain=0! sense_sat is sufficient on its own; relay is NOT necessary for multi-mound under saturation. PARTIAL RETRACTION of Est #4 — under saturation, relay's role changes from "drive aggregation" to "tighten existing mounds".
Knowledge update: Est #4 qualified: relay necessary only WITHOUT sense_sat. B14 should dedicate a relay.gain sweep at c_sat=0.20 parent.

## Batch 13 Sweep 14 — camp.decay @ c_sat=0.1  [flat with resonance spikes]
Hypothesis (H15-B13): decay × saturation.
Response: flat 1.20-1.25 across [0.05, 0.80]; sharp spikes at decay=0.08 (inner=0.49) and decay=0.36 (inner=0.58). Best decay=0.2 (1.195).
Morphology: sparse multi-spot at most values; tight knot at the two spikes — bimodal field-dynamics resonances.
Verdict: SUPPORTED with caveat — sense_sat broadens camp.decay working band to [0.05, 0.80].
Knowledge update: camp.decay=0.18 retained; decay=0.08 spike worth a fine probe in B14.

## Batch 13 Sweep 15 — spring.r0 @ parent  [flat — silent]
Hypothesis (H16-B13): r0 affects packing.
Response: flat-noisy 0.92-1.10; best r0=0.018 (0.918) — BEST LOSS OF BATCH but within noise.
Morphology: single fuzzy blob at every r0.
Verdict: INCONCLUSIVE within noise; r0 silent.
Knowledge update: spring.r0 flat; pin at 0.018.

## Batch 13 — summary

- THE FINDING: sense_sat BREAKS THE SINGLE-BLOB CEILING. Sw 1 at c_sat=0.2 produces 5-6 discrete compact mounds with inner_mass=0.662 vs REAL=0.606 — the FIRST morphologically credible multi-mound config in 13 batches and the closest visual match to REAL the model has produced under SSIM.
- BEST OF BATCH (by loss): sw 15 spring.r0=0.018 -> loss=0.918, inner=0.506 (single-blob ablation). Within noise.
- BEST OF BATCH (by morphology): sw 1 sense_sat.c_sat=0.2 -> loss=1.148, inner=0.662, ~6 compact mounds. Loss higher than ablation but morphology unambiguously closer to REAL.
- Sense_sat is the SIXTH mechanism added; FIRST to break the ceiling (after nucleation, sense_adapt, align, inhibitor, persistence all falsified).
- DENSIFICATION is the next frontier: c_sat=0.2 multi-mound regime has sparser per-mound density than REAL; SSIM doesn't reward sparse spots. B14 must find levers that densify each spot while preserving multiplicity.
- Surprising side-effects of sense_sat: (a) the secrete.rate in [3.6, 6.0] dispersion catastrophe (Est #29) DISAPPEARS under saturation; (b) relay.gain=0 ablation under c_sat=0.1 still gives multi-mound — relay NOT NECESSARY under sense_sat.
- New B14 parent: r_on=0.225 (multi-knot pre-fold per sw 5), n=1000, vmax=0.061, sense_sat.c_sat=0.20, sat_n=2 (multi-mound regime), inflow.rate=3.0, all else as B13.
- Open: densification levers (spring.kadh up, adhesion × saturation joint, relay.thr × c_sat=0.2, sat_n × c_sat fine grid, decay=0.08 resonance probe).

## Batch 14 Sweep 0 — seed @ c_sat=0.20 parent  [noise floor recalibrated]
Hypothesis (H1-B14): noise floor at new sense_sat multi-mound parent.
Response: flat-noisy 1.177-1.230 (range 0.053); inner 0.055-0.566 (HUGE spread); best seed=15 (1.177).
Morphology: HIGHLY seed-dependent — seeds 1, 3, 6 show 5-7 distinct compact mounds (best morphology of the entire batch); seeds 0, 10, 13 show diffuse cloud or 2-3 mounds. Bimodal across seeds.
Verdict: SUPPORTED — noise floor sigma~0.018 (tighter than B12 sigma=0.18, B11 Delta=0.13) and bimodal morphology. The c_sat=0.20 parent is at a MORPHOLOGICAL BIFURCATION across seeds.
Knowledge update: B14 loss differences <=0.04 within noise; inner_mass differences <=0.30 within noise. Bimodality is the new structural reality.

## Batch 14 Sweep 1 — sense_sat.c_sat (1e6->0.08)  [WORST=lowest c_sat]
Hypothesis (H2-B14): c_sat fine refinement, densification trade-off.
Response: MONOTONE-INCREASING loss as c_sat decreases. Best c_sat=1.0 (loss=0.9473, inner=0.512); ablation c_sat=1e6 = 1.122; c_sat<=0.20 plateau ~1.19.
Morphology: c_sat>=0.5 = 4-6 COMPACT close-to-REAL mounds (loss winners); c_sat<=0.22 = SPARSE many-tiny-spots scattered (B13's "breakthrough" regime). The SSIM loss prefers fewer, denser mounds — the higher-c_sat regime, NOT the B13 multi-mound regime.
Verdict: SURPRISE PARTIAL RETRACTION of Est #34 — the B13 c_sat=0.20 multi-mound is morphologically interesting (right number of spots) but each spot is SPARSE; under SSIM the LOSS prefers MILDER saturation (c_sat=0.5-1.0) which gives DENSER spots. The B13 "breakthrough" was on inner_mass diagnostic; the densification problem is now SHARPER.
Knowledge update: c_sat=1.0 is the new loss-best; sat_n axis is where the real morphological action is.

## Batch 14 Sweep 2 — sense_sat.sat_n @ c_sat=0.20  [PEAK at sat_n=1.1-1.25]
Hypothesis (H3-B14): Hill exponent fine; lower sat_n=1.25-1.5 may give denser multi-spot.
Response: U-shape in loss with sharp PEAK in inner_mass at sat_n=1.25 (inner=0.777, REAL=0.606); sat_n=1.1 inner=0.696. Loss minimum at sat_n=0.5 (0.924, inner=0.453) — mild saturation, washed-out.
Morphology: sat_n=0.5-1.0 = diffuse-fuzzy cloud (SSIM-good, structure-bad); sat_n=1.1-1.25 = 5-7 DENSE COMPACT MOUNDS, the CLOSEST VISUAL MATCH TO REAL in 14 BATCHES; sat_n>=1.4 = sparse few-tiny-spots scattered.
Verdict: BREAKTHROUGH — at c_sat=0.20, sat_n in [1.1, 1.25] produces dense multi-mound matching REAL inner_mass and morphology. SSIM does NOT reward it (diffuse cloud at sat_n=0.5 wins on loss). The loss/morphology divergence is now QUANTITATIVE: loss winner inner=0.453, morphology winner inner=0.696-0.777.
Knowledge update: NEW Est-candidate — sat_n=1.1-1.25 at c_sat=0.20 is the densification sweet spot. Trust the morphology.

## Batch 14 Sweep 3 — spring.kadh @ c_sat=0.20  [LOW kadh wins]
Hypothesis (H4-B14, DENSIFICATION): higher kadh attracts more cells per mound.
Response: MONOTONE-UP loss as kadh increases. Best kadh=5 (loss=1.149, inner=0.53 ~ REAL); kadh=240 = 1.207.
Morphology: kadh=5-20 = 4-5 visibly distinct compact mounds (closest-to-REAL inner_mass=0.53); higher kadh = MORE diffuse mounds (counter-intuitive).
Verdict: HYPOTHESIS REVERSED — under sense_sat, adhesion AMPLITUDE saturates the saturation effect (more kadh = more sticking = washes out the distinct-spot pattern). The densification mechanism is NOT kadh.
Knowledge update: NEW — under sense_sat, low kadh ~5-20 gives BEST multi-mound morphology. Adopt kadh=10-20 in B15.

## Batch 14 Sweep 4 — spring.r_on @ c_sat=0.20  [SHARP FOLD at r_on=0.245]
Hypothesis (H5-B14): adhesion reach densifies multi-mound.
Response: BIMODAL — r_on=0.18-0.235 = loss 1.18-1.19 + multi-mound; r_on>=0.245 = SHARP collapse to single blob (inner_mass jumps 0.245->0.626, 0.255->0.863, 0.28->0.849); loss best r_on=0.20 (1.155).
Morphology: r_on=0.18-0.235 = 4-6 distinct mounds (CLOSE to REAL inner_mass at r_on=0.245=0.626); r_on>=0.255 = ONE big central blob + few stragglers (the old single-attractor returns).
Verdict: SUPPORTED — sharp r_on fold at 0.245 under c_sat=0.20. The multi-mound regime is r_on in [0.18, 0.24]; above that, sense_sat is overpowered by adhesion and collapses.
Knowledge update: r_on=0.20-0.235 is the multi-mound band under sense_sat. Adopt r_on=0.20.

## Batch 14 Sweep 5 — cell.n @ c_sat=0.20, sat_n=1.5  [flat-noisy with marginal best]
Hypothesis (H6-B14): more cells densify each mound at sat_n=1.5.
Response: flat-noisy 1.10-1.19; best n=1500 (1.098, inner=0.185); n=1100 (1.103, inner=0.253) close.
Morphology: cell.n->up = MORE diffuse spread, NOT denser-per-mound. Same B13 sw 4 finding repeats.
Verdict: FALSIFIED H6-B14 — sat_n=1.5 does NOT rescue the densification-via-n problem. cell.n is NOT the densifier.
Knowledge update: cell.n flat-with-noise under sense_sat at sat_n=1.5; n=1000 retained.

## Batch 14 Sweep 6 — relay.gain @ c_sat=0.20  [relay NECESSARY but plateau]
Hypothesis (H7-B14): relay ablation at sense_sat parent — is sense_sat sufficient?
Response: gain=0 ablation = LOSS PEAK 1.27 (no aggregation); gain>=20 plateau 1.15-1.20; best gain=140 (1.146).
Morphology: gain=0 = SPARSE no-aggregation (faint stragglers); gain=20-600 = essentially invariant multi-spot morphology.
Verdict: PARTIAL RETRACTION OF Est #37 — at c_sat=0.20, relay IS necessary (ablation collapses loss to 1.27). The B13 sw 13 "relay-not-necessary" was specific to c_sat=0.10 (stronger saturation). Est #4 RE-INSTATED at c_sat=0.20.
Knowledge update: Est #4 holds at c_sat=0.20; Est #37 tightened to "at c_sat<=0.10".

## Batch 14 Sweep 7 — relay.thr @ c_sat=0.20  [FLAT]
Hypothesis (H8-B14): multi-knot fold shifts under saturation.
Response: completely FLAT 1.185-1.196 (range 0.011); inner ~0.06-0.20 invariant.
Morphology: invariant multi-spot at every thr; no fold visible.
Verdict: FALSIFIED — relay.thr has NO effect under c_sat=0.20.
Knowledge update: relay.thr silent under sense_sat. Drop from B15.

## Batch 14 Sweep 8 — camp.decay @ c_sat=0.20  [resonance dip at decay=0.07]
Hypothesis (H9-B14): probe decay=0.08 single-knot resonance.
Response: flat 1.18-1.19 with sharp DIP at decay=0.07 (loss=1.147, inner=0.168).
Morphology: most values = sparse multi-spot; decay=0.07 = slightly tighter compact spots; broadband working.
Verdict: SUPPORTED — sense_sat broadens decay working band [0.04, 0.50], with a resonance at decay=0.07.
Knowledge update: camp.decay=0.07 is a small resonance optimum; adopt for marginal improvement.

## Batch 14 Sweep 9 — camp.diffusion @ c_sat=0.20  [flat-noisy]
Hypothesis (H10-B14): Est #5 low-diffusion preference.
Response: flat 1.148-1.194; dip at D=0.0012 (1.148); ablation D=0.0001 close to D=0.01.
Morphology: invariant multi-spot.
Verdict: FALSIFIED Est #5 in sense_sat regime — diffusion no longer matters under saturation.
Knowledge update: camp.diffusion freed from low-band constraint at c_sat=0.20.

## Batch 14 Sweep 10 — secrete.rate @ c_sat=0.20  [no dispersion failure]
Hypothesis (H11-B14): saturation eliminates explosive-dispersion catastrophe.
Response: noisy 1.15-1.21; best rate=4 (1.162); the Est #29 [3.6, 6.0] catastrophe is ABSENT.
Morphology: invariant multi-spot; consistent ~5-6 small mounds.
Verdict: SUPPORTED — Est #36a re-confirmed in c_sat=0.20 regime. Working band [2, 13].
Knowledge update: secrete.rate has wide working band under sense_sat; adopt 4-7.

## Batch 14 Sweep 11 — inflow.rate @ c_sat=0.20  [bimodal dips at rate=4 and rate=7-10]
Hypothesis (H12-B14): inflow densifies multi-mound.
Response: U-shape with TWO dips: rate=4 (1.147) and rate=7-10 (1.151-1.154). Best rate=4.
Morphology: nearly invariant multi-spot; per-mound count modestly higher at rate=4-7.
Verdict: PARTIALLY SUPPORTED — inflow has a productive dip at rate=4 even under SSIM.
Knowledge update: NEW — under c_sat=0.20, inflow has a shallow optimum at rate=4. Adopt rate=4.

## Batch 14 Sweep 12 — sense_sat.gain @ c_sat=0.20  [MONOTONE-DOWN, DENSIFICATION LEVER]
Hypothesis (H13-B14): gain prefactor tunes multi-mound density.
Response: MONOTONE-DECREASING loss from gain=10 (1.223) to gain=240 (1.149); plateau >=120.
Morphology: low gain = washed-out diffuse; high gain = denser multi-mound spots.
Verdict: STRONGLY SUPPORTED — gain=240 wins; higher chemotactic drive DENSIFIES per-mound. THIS IS THE DENSIFICATION LEVER predicted but not previously found.
Knowledge update: NEW Established — sense_sat.gain monotone densifier in [10, 240]. Adopt gain=240 in B15; explore higher in B15.

## Batch 14 Sweep 13 — random_walk.strength @ c_sat=0.20  [flat-noisy]
Hypothesis (H14-B14): RW dislodges over-tight mounds.
Response: flat-noisy 1.14-1.19; dips at rw=0.045 (1.141) and rw=0.020 (1.149).
Morphology: invariant multi-spot.
Verdict: INCONCLUSIVE — RW silent within noise.
Knowledge update: random_walk silent under sense_sat; pin at 0.01.

## Batch 14 Sweep 14 — vmax @ c_sat=0.20  [flat — aliasing weakened]
Hypothesis (H15-B14): re-test Est #9 aliasing.
Response: flat-noisy 1.180-1.192 (range 0.012); best vmax=0.070 (1.180); B12 sharp resonance walls GONE.
Morphology: nearly invariant.
Verdict: PARTIAL RETRACTION of Est #9 in sense_sat regime — saturation also weakens the dt x vmax aliasing constraint.
Knowledge update: vmax has wide [0.054, 0.072] working band under sense_sat; aliasing weakened.

## Batch 14 Sweep 15 — sense_sat.c_sat @ sat_n=1.5  [replicates sw 1]
Hypothesis (H16-B14): c_sat at sat_n=1.5 densification grid row.
Response: monotone-up loss as c_sat decreases. Best c_sat=1.0 (loss=0.9246, inner=0.524); ablation 1e6 = 1.122.
Morphology: c_sat>=0.4 = COMPACT 4-7 mounds (closest to REAL of all sweeps); c_sat<=0.22 = sparse-tiny scattered.
Verdict: REPLICATED sw 1 — at sat_n=1.5, c_sat=0.5-1.0 gives the visually best multi-mound. The dense-multi-mound regime is c_sat=0.4-1.0.
Knowledge update: NEW Established — under sat_n=1.5, multi-mound DENSE regime is c_sat in [0.4, 1.0], NOT c_sat=0.10-0.20 (too saturated, sparse).

## Batch 14 — summary

- THE KEY FINDING: the B13 "c_sat=0.20 breakthrough" was MORPHOLOGICALLY CORRECT but in the SPARSE-multi-spot regime. The B14 sweeps reveal a DENSER multi-mound regime at HIGHER c_sat (in [0.4, 1.0]) AND a per-mound DENSIFICATION mechanism at higher sense_sat.gain (=240). The morphologically-best B14 configs produce 5-7 DENSE COMPACT MOUNDS — closest visual match to REAL in 14 batches.
- BEST LOSS (sw 2, sat_n=0.5 c_sat=0.20): loss=0.9238, inner=0.453 — but morphologically WASHED-OUT diffuse cloud (SSIM rewards smooth blobs, not multi-mound structure). DO NOT ADOPT.
- BEST MORPHOLOGY (sw 2, sat_n=1.25 c_sat=0.20): loss=1.162, inner=0.777 (above REAL=0.606); 5-7 dense compact mounds, closest match in 14 batches. ADOPT FOR B15 PARENT.
- SECONDARY MORPHOLOGY WINNER (sw 1+15, c_sat=1.0 sat_n=2): loss=0.92-0.95, inner=0.51-0.52; 4-6 compact mounds; loss-good AND morphology-good — alternative parent.
- Falsified at this parent: cell.n densification (sw 5), relay.thr (sw 7), camp.diffusion preference (sw 9), random_walk (sw 13), vmax aliasing (sw 14). All saturation-regularized.
- New findings: (a) sense_sat.gain=240 is a DENSIFICATION lever (sw 12 monotone); (b) low spring.kadh ~5-20 is better than high (sw 3 monotone-up reverses); (c) relay still necessary at c_sat=0.20 but not at c_sat<=0.10 (re-instate Est #4); (d) inflow.rate=4 productive dip (sw 11) re-opens Est #24.
- Strategic frame for B15: question NO LONGER "how to get multi-mound" (sense_sat solves it). Now: (1) which (c_sat, sat_n) corner is optimal — sparse-many (B13 c_sat=0.20 sat_n=2) vs dense-fewer (B14 c_sat=0.5 sat_n=1.5)? (2) extend sense_sat.gain past 240 to find saturation; (3) tighten r_on in [0.20, 0.235]; (4) joint c_sat x gain grid as densification surface; (5) sample seeds 1/3/6 (B14 sw 0 morphology winners).
- NEW B15 PARENT: c_sat=0.20, sat_n=1.25 (morphology winner), sense_sat.gain=240, spring.kadh=15, r_on=0.20, camp.decay=0.07, inflow.rate=4, vmax=0.061, relay.gain=140. Explicitly a science-over-loss choice; documents the SSIM/morphology divergence.

## Batch 15 Sweep 0 — seed [parent regime FAILS]
Hypothesis (H1-B15): noise floor at new B15 DENSE multi-mound parent.
Response: flat-noisy 5.5-11.8 (range 6.3, mean ~8.0); inner=0.20-0.29 (REAL=0.61); best seed=7 (loss=5.50, inner=0.273).
Morphology: ALL 16 seeds = invariant diffuse field of tiny scattered clumps; NO sustained mounds at any seed.
Verdict: FALSIFIED — the B15 parent is in a CATASTROPHIC dispersion regime (loss 10x worse than B14 parent 1.15). The B14 "science-over-loss" choice of sat_n=1.25, c_sat=0.20 with gain=240, kadh=15, r_on=0.20, decay=0.07, inflow=4 collapses to noise.
Knowledge update: B14 sw 2 sat_n=1.25 "morphology winner" was a single-axis observation NOT robust to the other B14 adoptions — combined parent does not maintain multi-mound. PARTIAL RETRACTION of Est #38/#42.

## Batch 15 Sweep 1 — sense_sat.gain WIDE [supported direction; ablation regime best]
Hypothesis (H2-B15): extend gain past sw 12 max=300 to find saturation/reversal.
Response: peaked-low; best gain=60 (loss=2.05, inner=0.287). Across [60, 800] noisy 2-12.7, gain=380 catastrophic (12.78).
Morphology: gain=60 = somewhat brighter scattered field with hints of clusters; gain>=180 = uniform diffuse fuzz; never multi-mound.
Verdict: FALSIFIED B14 Est #39 in the B15 regime — at sat_n=1.25 c_sat=0.20, gain=240 is NOT productive (it overrides receptor saturation). Lower gain wins.
Knowledge update: Est #39 (gain monotone-densifies up to 240) was specific to B14 parent (sat_n=2). At sat_n=1.25, gain=60 wins. The "densification lever" is regime-dependent.

## Batch 15 Sweep 2 — sense_sat.c_sat [parent regime broken — c_sat does not rescue]
Hypothesis (H3-B15): c_sat FINE around [0.15, 2.0] to resolve sparse/dense regime boundary.
Response: noisy 5.5-12.3; best c_sat=0.25 (loss=5.49, inner=0.260); ablation 1e6=10.09.
Morphology: invariant diffuse fuzz across the entire c_sat range; no clean mounds emerge at any value.
Verdict: FALSIFIED — at sat_n=1.25 with the other B15 params, NO c_sat value rescues morphology. The Est #38 "dense regime at c_sat in [0.4, 1.0]" requires sat_n>=2 (see sw 7 below).
Knowledge update: c_sat alone insufficient; sat_n is the dominant axis.

## Batch 15 Sweep 3 — sense_sat.sat_n @ c_sat=0.20 [BREAKTHROUGH cliff at sat_n>=1.9]
Hypothesis (H4-B15): sat_n FINE around B14 sw 2 peak at [1.1, 1.25].
Response: PHASE TRANSITION at sat_n=1.9 — loss=11.78 at sat_n=1.25 collapses to loss=1.12 at sat_n=1.9; flat ~1.1 at sat_n in [1.9, 2.5]; best sat_n=1.9 (loss=1.12, inner=0.215).
Morphology: sat_n<=1.6 = diffuse fuzz; sat_n=1.9 = first hint of structure; sat_n=2.1, 2.5 = CLEAN 4-5 dense quad-mound pattern (closest to REAL morphology in B15).
Verdict: SUPPORTED — high sat_n is the multi-mound lever at c_sat=0.20; B14 peak at sat_n=1.25 was a metric-specific artifact within the joint parent. The cliff between 1.6 and 1.9 reveals a hard regime boundary.
Knowledge update: NEW Established — at c_sat=0.20, sat_n>=1.9 is required for ANY aggregation; previous "sat_n=1.25 morphology peak" RETRACTED.

## Batch 15 Sweep 4 — spring.r_on [flat-with-noise at broken parent]
Hypothesis (H5-B15): r_on FINE within multi-mound band [0.18, 0.245].
Response: noisy 2.2-10.9; best r_on=0.215 (loss=2.19, inner=0.239); secondary dip r_on=0.245 (loss=2.50).
Morphology: invariant diffuse fuzz; no morphological response — the parent is too broken for r_on to matter.
Verdict: INCONCLUSIVE — r_on cannot adjudicate at a dispersion-collapsed parent.
Knowledge update: drop r_on sweep at this parent; re-test at sat_n=3.0 c_sat=0.50.

## Batch 15 Sweep 5 — spring.kadh [monotone-down; high kadh rescues partially]
Hypothesis (H6-B15): kadh FINE LOW around B14 sw 3 best=5.
Response: roughly MONOTONE-DECREASING in loss as kadh increases; best kadh=80 (loss=1.01, inner=0.235); kadh<=4 = catastrophic (13-15).
Morphology: kadh=2-10 = very dispersed; kadh=60-80 = a couple of distinct denser clumps emerging.
Verdict: SUPPORTED in direction but FALSIFIES B14 Est #40 — at sat_n=1.25 c_sat=0.20, HIGH kadh wins, opposite of B14 finding. Est #40 RETRACTED.
Knowledge update: NEW Established — kadh optimum depends on sat_n. High kadh (>=60) preferred when sat_n is low; low kadh preferred when sat_n=2.

## Batch 15 Sweep 6 — sense_sat.gain @ c_sat=0.50 [flat-broken at sat_n=1.25]
Hypothesis (H7-B15): is gain monotone-densifying in the dense regime too?
Response: flat-noisy 5.8-15.2 across [40, 800]; best gain=280 (loss=5.80, inner=0.238); gain=600 catastrophic.
Morphology: invariant diffuse fuzz; the c_sat=0.50 dense regime does NOT activate at sat_n=1.25 either.
Verdict: FALSIFIED — gain at c_sat=0.50 with sat_n=1.25 also produces dispersion. Confirms sat_n=1.25 is the broken axis, not c_sat.
Knowledge update: Densification regime requires sat_n>=2 across all c_sat tested.

## Batch 15 Sweep 7 — sense_sat.sat_n @ c_sat=0.50 [BEST MORPHOLOGY — DENSE multi-mound]
Hypothesis (H8-B15): sat_n at c_sat=0.50 (dense regime).
Response: PHASE TRANSITION — sat_n<=2.0 = noisy 7-12; sat_n=2.5 = loss=1.25; sat_n=3.0 = 1.10; sat_n=3.5 = 1.18; sat_n=4.0 = 1.16.
Morphology: sat_n<=2 = diffuse fuzz; sat_n=2.5 = visible 4-5 mounds beginning; sat_n=3.0, 3.5, 4.0 = CLEAN 5-6 DENSE COMPACT MOUNDS — closest visual match to REAL in B15 (and competitive with B14's best).
Verdict: SUPPORTED — at c_sat=0.50, sat_n>=3.0 produces the morphologically credible dense-multi-mound state. **This is the B16 parent direction.**
Knowledge update: NEW Established — DENSE multi-mound at sat_n=3.0, c_sat=0.50. RETRACTS B14 Est #38 sat_n peak around 1.5.

## Batch 15 Sweep 8 — relay.gain @ parent [ablation wins in broken regime]
Hypothesis (H9-B15): relay.gain FINE around 140.
Response: peaked-low at gain=0; ablation (gain=0) loss=1.33 inner=0.303; gain=30 = 17.45 (catastrophic spike); gain>=60 = 9-15.
Morphology: gain=0 = 2-3 tiny dim dots (lowest inner_mass collapse here); gain>=30 = diffuse field dispersed.
Verdict: SUPPORTED in this broken parent — relay actively destroys morphology at sat_n=1.25. PARTIAL RETRACTION of Est #4 (relay necessary): relay may be NECESSARY only when sat_n is in the multi-mound regime (>=1.9).
Knowledge update: re-test relay ablation at sat_n=3.0 c_sat=0.50 in B16.

## Batch 15 Sweep 9 — camp.decay [flat with single dip at 0.09]
Hypothesis (H10-B15): camp.decay FINE around 0.07 resonance dip.
Response: flat-noisy 5-12; sharp DIP at decay=0.09 (loss=3.09, inner=0.214); no broad pattern.
Morphology: invariant diffuse fuzz; no morphological response.
Verdict: INCONCLUSIVE — decay=0.09 dip likely a seed-realisation outlier at this broken parent.
Knowledge update: defer decay sweep to morphology-credible parent.

## Batch 15 Sweep 10 — inflow.rate [flat with rate=0.5 best]
Hypothesis (H11-B15): inflow.rate FINE around 4.
Response: noisy 4-12; best rate=0.5 (loss=4.18, inner=0.227); rate=7 secondary dip (4.72); rate=4 (parent) catastrophic 11.78.
Morphology: invariant; rate barely affects the dispersion-fuzz pattern.
Verdict: INCONCLUSIVE — at broken parent, inflow signals only via overall density.
Knowledge update: re-test inflow at sat_n=3.0 c_sat=0.50 parent.

## Batch 15 Sweep 11 — cell.n [flat-broken]
Hypothesis (H12-B15): cell.n at new dense parent.
Response: noisy 3.9-13.7; best n=2000 (loss=3.93, inner=0.267); secondary dip n=1300 (6.7).
Morphology: invariant diffuse field; more cells = denser fuzz, not multi-mound.
Verdict: INCONCLUSIVE — broken parent cannot resolve cell.n.
Knowledge update: re-test cell.n at sat_n=3.0 c_sat=0.50 in B16.

## Batch 15 Sweep 12 — relay.gain @ sense_sat.gain=400 [ablation wins again]
Hypothesis (H13-B15): does high-gain saturation change the relay's role?
Response: peaked-low at gain=0 (loss=1.20, inner=0.265); gain in [30, 200] = 5-11.5; secondary dips at gain=380 (2.50), gain=500 (3.70).
Morphology: gain=0 = sparse few-dot; gain>=30 = diffuse-fuzz dispersion.
Verdict: SUPPORTED — relay-ablation regime is consistently best at sat_n=1.25 regardless of sense_sat.gain. Confirms sw 8 finding.
Knowledge update: re-confirms broken parent's relay destructiveness; ablation is the only escape at sat_n=1.25.

## Batch 15 Sweep 13 — secrete.rate [BEST LOSS sw — but diffuse morphology]
Hypothesis (H14-B15): secrete.rate FINE at new parent.
Response: bimodal — secrete<=3 loss ~1.0; secrete in [3.5, 5.5] catastrophic spike 5.8-11.8; secrete>=6.5 noisy ~1.0; best secrete=10 (loss=0.957, inner=0.276).
Morphology: secrete=2-3 = sparse scattered tiny dots; secrete=3.5-5.5 = pure diffuse fuzz (dispersion); secrete>=6.5 = many tiny scattered clumps, slight clustering visible but no compact mounds.
Verdict: SUPPORTED — the [3.6, 6.0] explosive-dispersion failure mode (Est #29) IS PRESENT at the broken B15 parent, and outside this band the model just barely organises into a low-loss diffuse "pattern" that gets best loss but no real mounds. The "best of batch" is a metric artifact.
Knowledge update: re-confirms Est #29 explosive-dispersion band; secrete>=7 OR secrete<=3 both escape the band, but neither produces credible multi-mound at sat_n=1.25.

## Batch 15 Sweep 14 — cell.n @ inflow.rate=4 [flat-broken duplicate of sw 11]
Hypothesis (H15-B15): cell.n at inflow=4 (effective duplicate of sw 11; intended inflow=7 lost to spec).
Response: flat-noisy 4.7-11.4; best n=1000 (loss=4.72, inner=0.198).
Morphology: invariant diffuse field.
Verdict: INCONCLUSIVE — duplicates sw 11.
Knowledge update: none.

## Batch 15 Sweep 15 — sense_sat.c_sat saturation-onset map [flat across [1e6, 0.3]]
Hypothesis (H16-B15): saturation-onset map at sat_n=1.25.
Response: gentle monotone-down from c_sat=1e6 (10.09) to c_sat=0.3 (10.50); flat ~5.5-12 across.
Morphology: invariant diffuse fuzz across the entire c_sat range.
Verdict: FALSIFIED — at sat_n=1.25, ALL c_sat values produce dispersion. Saturation onset is irrelevant because sat_n is too shallow to switch on aggregation.
Knowledge update: sat_n=1.25 is permanently a broken regime; drop it as a parent candidate.

## Batch 15 — summary

- **CATASTROPHIC PARENT FAILURE**: the B14 "science-over-loss" pick (sat_n=1.25, c_sat=0.20 + gain=240, kadh=15, r_on=0.20, decay=0.07, inflow=4) is in a DISPERSION regime, loss=11.78 at parent (10x worse than B14 parent ~1.15). 14 of 16 single-axis sweeps could not produce credible morphology; all "wins" were either ablation rescues or the loss-only sw 13 secrete=10 (sparse-fuzz best loss 0.957).
- **THE B14 MORPHOLOGY WINNER WAS NOT ROBUST**: at the JOINT B14 parent (different other params), sat_n=1.25 c_sat=0.20 made dense mounds; at the B15 parent it does not. Single-axis morphology peaks are not transitive across joint changes. Partial retraction of Est #38, full retraction of "sat_n=1.25 as morphology winner".
- **THE GENUINE BREAKTHROUGH (sw 7)**: at c_sat=0.50, sat_n>=3.0 produces CLEAN 5-6 dense compact mounds — closest visual match to REAL in B15 and competitive with B14's morphology winners — at loss=1.10. **This is the actual dense-multi regime.**
- **SECONDARY BREAKTHROUGH (sw 3)**: at c_sat=0.20, sat_n=2.1-2.5 produces clean 4-5 sparse quad-mounds at loss=1.12-1.18 — the B13 c_sat=0.20 regime is recovered when sat_n is high enough.
- **NEW Established (sat_n is the master regime switch)**: at c_sat in [0.20, 1.0], sat_n>=1.9 is REQUIRED for any aggregation. Below sat_n=1.9, the Hill function is too shallow and the saturation doesn't switch chemotaxis off in mounds, so the whole field disperses.
- **NEW Established (kadh-sat_n coupling)**: at sat_n=1.25 (broken regime) HIGH kadh (>=60) wins; at sat_n=2 LOW kadh (5-20) wins (B14 Est #40). Kadh is regime-dependent.
- **NEW Falsified**: B14 Est #38 sweet spot at sat_n=1.25 (RETRACTED); Est #39 monotone-up gain to 240 (REGIME-DEPENDENT — only at high sat_n); the "B15 science-over-loss parent" as a whole.
- **NEW B16 PARENT**: c_sat=0.50, sat_n=3.0, gain=240, kadh=15, r_on=0.20, relay.gain=140, secrete=4, decay=0.07, inflow=4, cell.n=1000. The sw 7 morphology winner. (Re-tests of kadh, relay, secrete, decay, inflow, cell.n at this parent will pin down which adoptions transfer.)
- **STRATEGIC FRAME for B16**: now that we have a robust dense-multi-mound parent (sw 7 sat_n=3.0 c_sat=0.50), re-test the lever sweeps at this parent to identify which B14/B15 findings were artifacts of broken parents. Map (sat_n, c_sat) more carefully around the new winner. Re-test kadh, relay.gain, secrete in this regime.

## Batch 16 Sweep 0 — seed [BREAKTHROUGH STABILITY]
Hypothesis (H1-B16): noise floor at new B16 DENSE multi-mound parent (sat_n=3.0, c_sat=0.50).
Response: tight cluster loss=1.03-1.19 (sigma~0.04, range 0.16); best seed=3 (loss=1.0334, inner=0.415).
Morphology (from strip): EVERY seed shows clean 5-7 distinct dense compact mounds (qualitatively closest match to REAL in 16 batches). The parent is morphologically REPRODUCIBLE across all seed draws — first time this happens in the project.
Verdict: SUPPORTED — B15 sw 7 was a genuine regime, not a lucky seed. Multi-mound morphology is STABLE.
Knowledge update: NEW Est — B16 parent has the lowest seed-noise floor and the most morphologically credible parent observed. Adopting whole-config morphology winner (Est #43 lesson) works.

## Batch 16 Sweep 1 — relay.gain [DECISIVE: Est #4 confirmed, Est #47 falsified]
Hypothesis (H2-B16): DECISIVE relay ablation at sat_n=3.0 c_sat=0.50. Tests Est #4 (relay necessary) vs Est #47 (relay destructive only because B15 parent was broken).
Response: U-shape with sharp wall — gain=0 ablation loss=1.33 (sparse few-mound); gain=30 RING-LOSS 13.83; gain=60 RING-LOSS 10.52; gain=90 PEAK best loss=0.9858; gain>=120 plateau 1.10-1.18.
Morphology (from strip): gain=0 = ~4 sparse weak mounds (not catastrophic); gain=30/60 = pure DISPERSION uniform speckle (ringing failure); gain>=90 = 5-7 dense compact mounds (the B16 morphology).
Verdict: Est #4 RE-CONFIRMED — relay is NECESSARY at sat_n=3.0; ablation is sub-par but not catastrophic. Est #47 FALSIFIED — relay is NOT destructive at the dense parent; the B15 destruction was a broken-parent artifact. Est #29 ringing band re-confirmed at [30, 60] under sense_sat.
Knowledge update: relay.gain=90 is the new wide-plateau winner (B16 best loss of batch); the ringing band is sharp and reproducible.

## Batch 16 Sweep 2 — spring.kadh [low end wins, Est #46 simplified]
Hypothesis (H3-B16): kadh-sat_n coupling at sat_n=3.0.
Response: kadh=2 catastrophic 3.08 (single tight blob); kadh=4 sub-par 1.92; kadh=6 PEAK best loss=1.00; kadh=10 1.11; kadh=15-300 flat plateau 1.13-1.17.
Morphology (from strip): kadh=2 = ONE tight central blob (over-glued cells collapse to one); kadh=4 = a few diffuse spots; kadh=6 = clean dense 5-6 mounds; kadh=10-300 = consistent 5-6 dense compact mounds across the entire range.
Verdict: SUPPORTED — at sat_n=3.0, LOW kadh (B14-style) wins (Est #46 partial); but the plateau is BROAD (kadh in [10, 300] morphologically identical). kadh=6 is the loss winner with sharp catastrophe below.
Knowledge update: kadh has a sharp LOWER bound at 6, NO upper bound effect within tested range. Est #46 simplified — under sense_sat sat_n=3.0, kadh is broad above the cutoff.

## Batch 16 Sweep 3 — sat_n FINE around 3.0 at c_sat=0.50 [Est #45 re-confirmed]
Hypothesis (H4-B16): sat_n peak refinement.
Response: sat_n=2.0 -> catastrophic dispersion 7.08; sat_n=2.25 -> 4.69; sat_n=2.5 -> 1.25 (transitioning); sat_n=2.75 -> 1.13; sat_n=3.0 PEAK 1.10; sat_n>=3.25 plateau 1.16-1.22.
Morphology (from strip): sat_n=2.0-2.25 = diffuse fuzz (broken); sat_n=2.5 = sparse weak spots; sat_n=2.75-3.0 = clean dense 5-7 mounds; sat_n>=3.5 = sparser per-spot mounds (still multi-mound but each mound less dense).
Verdict: SUPPORTED — sat_n=3.0 is the optimum; sharp sub-2.5 catastrophe wall; higher sat_n preserves multi-mound but reduces per-mound density. Est #44 sat_n>=1.9 mandatory boundary is sharpened to sat_n>=2.75 at c_sat=0.50.
Knowledge update: Est #45 (sat_n=3.0 sweet spot) RE-CONFIRMED in finer scan. Working band [2.75, 6.0] confirmed.

## Batch 16 Sweep 4 — c_sat at sat_n=3.0 [DENSE/SPARSE/ABLATION 3-regime map]
Hypothesis (H5-B16): c_sat axis at sat_n=3.0.
Response: c_sat=1e6 (ablation) catastrophic 10.09; c_sat=5-1.0 stay catastrophic 6-10 (under-saturating at high sat_n); c_sat=0.8 -> 6.11; c_sat=0.7 -> 2.67 (transition); c_sat=0.6 -> 1.08; c_sat=0.55 PEAK best 1.074; c_sat=0.50 (parent) -> 1.10; c_sat<=0.45 -> plateau 1.15-1.17.
Morphology (from strip): c_sat>=0.7 = diffuse/ablation-like (the high sat_n + high c_sat combo is NEAR-ABLATION because saturation kicks in only at very high c); c_sat=0.6-0.5 = dense 5-7 compact mounds (the dense regime); c_sat=0.45-0.20 = sparser-multi (more mounds, smaller each — sliding into B13 sparse regime).
Verdict: SUPPORTED — three regimes are sharply separated. The dense-multi optimum is c_sat in [0.50, 0.60]. Below 0.45 the regime slides to sparse-multi (B13). Critical insight: at sat_n=3.0, c_sat>=0.7 is functionally ablation — the saturation manifold goes diagonally.
Knowledge update: NEW Est candidate — the (sat_n, c_sat) trade-off curve: each c_sat requires a specific min sat_n (and vice versa) to be in dense-multi regime.

## Batch 16 Sweep 5 — sense_sat.gain WIDE [BEST LOSS OF BATCH]
Hypothesis (H6-B16): gain at sat_n=3.0 c_sat=0.50.
Response: monotone-decreasing 40->500; gain=40 1.16; gain=80 1.14; gain=200 1.09; gain=240 (parent) 1.10; gain=400 1.04; gain=500 PEAK best loss=0.9802 inner=0.372; gain=600 1.06; reversal gain=800 1.44; gain=1000 1.64.
Morphology (from strip): gain=40-200 = clean 5-7 dense compact mounds (multi-mound stable); gain=240-500 = same morphology, slight per-mound size increase (densification); gain=600-700 = morphology preserved but some convergence; gain=800-1000 = single dominant central blob (over-attraction collapses multiplicity).
Verdict: SUPPORTED, and EXTENDS Est #39 — the densification lever monotone-down extends well past 240; PEAK is at gain=500. Above 600, over-attraction collapses multi-mound to single blob.
Knowledge update: NEW Est — sense_sat.gain optimum is 500 (refines/extends Est #39). Adopt gain=500 for B17 parent.

## Batch 16 Sweep 6 — spring.r_on [low end wins, dense regime extends]
Hypothesis (H7-B16): r_on FINE in dense regime.
Response: noisy with weak signal; r_on=0.16 loss=1.04 (best); r_on=0.20 (parent) 1.10; r_on=0.245 1.16; r_on=0.25 1.20; r_on=0.26 1.18. inner_mass spikes at r_on=0.25 (0.76) and 0.26 (0.81) = over-tight single blob.
Morphology (from strip): r_on=0.16-0.245 = consistent 5-7 dense compact mounds (the multi-mound regime extends LOWER than B14/B15 thought); r_on=0.25 = a TIGHT 2-mound; r_on=0.26 = single tight central blob (over-compact wall).
Verdict: SUPPORTED — r_on lower band [0.16, 0.245] is multi-mound; over-compaction wall sits at r_on>=0.25; lower r_on does NOT hurt. Sense_sat regularizes the adhesion-reach failure mode.
Knowledge update: at sat_n=3.0 c_sat=0.50, r_on band is broad [0.16, 0.245]; adopt r_on=0.18 conservatively (avoid the noisy 0.16 single point).

## Batch 16 Sweep 7 — inflow.rate [FLAT-MULTI-MOUND, Est #29 boundary RETRACTED]
Hypothesis (H8-B16): inflow at dense parent.
Response: flat-noisy 1.04-1.17 across [0, 14]; best rate=2.5 (1.04); rate=0 (1.06); rate=14 (1.12).
Morphology (from strip): EVERY rate produces clean 5-7 dense compact mounds. NO over-dilution failure at any rate up to 14 (vs Est #29 which said rate>6 disperses).
Verdict: SUPPORTED — inflow is fully tolerated under sense_sat sat_n=3.0; multi-mound morphology robust to high inflow. Est #29 inflow>6 over-dilution failure mode is REGULARIZED by sense_sat (extends Est #36/#41 broadening).
Knowledge update: NEW Est — inflow band is flat AND morphology-preserving in the dense regime; adopt rate=2.5 for B17 (matches biology with mild improvement).

## Batch 16 Sweep 8 — camp.decay WIDE [broad working band, multi-mound everywhere]
Hypothesis (H9-B16): camp.decay at dense parent.
Response: flat-noisy 1.05-1.15 across [0.02, 0.40]; best decay=0.28 (1.05); decay=0.16 (1.09); decay=0.07 (parent) (1.10).
Morphology (from strip): EVERY decay value produces 5-7 dense compact mounds. No catastrophic failure within tested range; the "field dies" boundary (Est #29 decay>0.40) needs re-test at higher values.
Verdict: SUPPORTED — camp.decay band broadens further under sense_sat at sat_n=3.0; extends Est #36/#41. No resonance dip at 0.07/0.08; the B14 dip was regime-specific.
Knowledge update: camp.decay band [0.02, 0.40] all multi-mound; adopt 0.16 or 0.28 in B17 if any signal beyond noise survives joint re-test.

## Batch 16 Sweep 9 — cell.n [multi-mound across range, capacity wall at 3400]
Hypothesis (H10-B16): more cells = denser per-mound at sat_n=3.0.
Response: flat-noisy 1.10-1.18 across [400, 3200]; engine NaN at n=3400 (capacity wall, re-confirms B11 sw 5, B12 sw 2); best n=1000 (parent) at 1.10.
Morphology (from strip): EVERY n produces 5-7 dense compact mounds. More cells = SLIGHTLY larger mounds but NOT more discrete mounds. Mound count and per-mound density both flat in n.
Verdict: SUPPORTED — cell.n is NOT a densifier at sat_n=3.0 (replicates B13 sw 4 / B14 sw 5 falsification under strongest dense regime). The morphology is determined by the sense_sat regime, not the cell count.
Knowledge update: cell.n flat under sense_sat; the 8-mound target cannot be reached by simply adding cells. Adopt n=1000.

## Batch 16 Sweep 10 — secrete.rate [explosive-dispersion REGULARIZED, monotone-down to 11]
Hypothesis (H11-B16): secrete at dense parent.
Response: weak monotone-down 1.16->1.05; best rate=11 (1.053); rate=20 (1.118); rate=30 (1.171). NO explosive-dispersion failure in [3.6, 6.0] (vs Est #29 expectation).
Morphology (from strip): EVERY rate (2-30) produces 5-7 dense compact mounds; the explosive-dispersion failure mode is FULLY regularized. Slightly more diffuse at rate=2-3, slightly denser per-mound at rate=11.
Verdict: SUPPORTED — Est #29 secrete dispersion band [3.6, 6.0] is REGULARIZED at sat_n=3.0 (extends Est #36 finding from c_sat=0.10 to the dense regime). Adopt rate=11.
Knowledge update: secrete.rate band broadens; adopt rate=11 for B17 parent.

## Batch 16 Sweep 11 — camp.diffusion [Est #5 retracted in dense regime, multi-mound across full range]
Hypothesis (H12-B16): camp.diffusion at dense parent.
Response: flat-noisy 1.09-1.16 across [0.0001, 0.08]; best D=0.01 (1.094); parent D=0.0012 (1.10).
Morphology (from strip): EVERY D value produces 5-7 dense compact mounds; mounds preserved even at D=0.08 (80x parent).
Verdict: SUPPORTED — Est #41 (camp.diffusion freedom under sense_sat) RE-CONFIRMED at sat_n=3.0. The low-D preference (Est #5) is fully retracted in the dense regime. Mound morphology is robust to D over >=2 orders of magnitude.
Knowledge update: Est #5 finally retracted in the dense regime; diffusion is freed.

## Batch 16 Sweep 12 — sat_n at c_sat=1.0 [trade-off curve, higher c_sat needs higher sat_n]
Hypothesis (H13-B16): sat_n peak shift at higher c_sat.
Response: sat_n<=3.0 catastrophic 7-12 (under-saturating); sat_n=3.5 1.07 (transition); sat_n=4.0 PEAK best 1.03; sat_n=5-10 plateau 1.13-1.18.
Morphology (from strip): sat_n<=3.0 = diffuse fuzz (broken); sat_n=3.5 = transition; sat_n=4.0 = clean 5-7 dense mounds; sat_n>=5 = sparser-multi (per-mound density decreases).
Verdict: SUPPORTED — at higher c_sat=1.0 the sat_n threshold for aggregation shifts UP from 2.75 to >=3.5. The (sat_n, c_sat) plane has a trade-off ridge: higher c_sat needs higher sat_n.
Knowledge update: NEW Est candidate — the (sat_n, c_sat) ridge in the dense regime; B17 can map this systematically.

## Batch 16 Sweep 13 — sat_n at c_sat=0.30 [trade-off curve confirmed at lower c_sat]
Hypothesis (H14-B16): sat_n peak shift at lower c_sat.
Response: sat_n<=1.9 catastrophic 3-8; sat_n=2.1 -> 1.29 (transition); sat_n=2.25 PEAK best 1.03; sat_n=2.5-6 plateau 1.13-1.18.
Morphology (from strip): sat_n=1.5-1.9 = diffuse fuzz (broken); sat_n=2.0-2.1 = sparse weak; sat_n=2.25-3 = clean 5-7 multi-mound; sat_n=3-6 = sparser-multi (more per-spot loss in density).
Verdict: SUPPORTED — at lower c_sat=0.30 the sat_n threshold shifts DOWN to 2.25 (vs 3.5 at c_sat=1.0, vs 2.75 at c_sat=0.50). Confirms the (sat_n, c_sat) trade-off curve.
Knowledge update: NEW Est — the trade-off (c_sat, sat_n) ridge: roughly sat_n ~= 2.25 at c_sat=0.30, 2.75 at c_sat=0.50, 3.5 at c_sat=1.0. The c_sat=0.30 column has the lowest sat_n threshold and slightly lower loss (1.03 vs 1.10) — possible alternative parent for B17.

## Batch 16 Sweep 14 — relay.thr [flat, no multi-spot shift in dense regime]
Hypothesis (H15-B16): relay.thr at dense parent.
Response: flat-noisy 1.11-1.20 across [0.10, 0.70]; best thr=0.22 (parent) 1.10.
Morphology (from strip): EVERY thr value preserves 5-7 dense compact mounds. The B12 sw 5 "high-thr sparse-multi" effect does NOT appear in the dense regime — sense_sat already produces multi-mound, and relay.thr does not modulate the count.
Verdict: SUPPORTED (and informative): relay.thr is FLAT under sense_sat at sat_n=3.0. Sense_sat displaces relay.thr as a multi-mound lever.
Knowledge update: relay.thr is silent under sense_sat dense regime; drop relay.thr sweeps from refinement plans.

## Batch 16 Sweep 15 — vmax [aliasing wall RETURNS at sat_n=3.0]
Hypothesis (H16-B16): vmax aliasing at dense parent.
Response: vmax<=0.072 flat 1.11-1.20; vmax>=0.072 morphological transition; vmax=0.075 1.18; vmax=0.08 1.16; vmax=0.085 1.16; vmax=0.09 1.18.
Morphology (from strip): vmax<=0.07 = 5-7 dense multi-mound; vmax=0.072+ = morphology TRANSITIONS to a SINGLE tight central blob with high inner_mass (0.6-0.7). The aliasing wall returns at vmax=0.075+ (over-step) — exactly the Est #9 phenomenon.
Verdict: SUPPORTED — Est #9 (aliasing wall) RETURNS at sat_n=3.0 dense regime (partial RETRACTION of Est #41's claim that saturation universally weakens aliasing). Sense_sat does NOT regularize all field-side failure modes; aliasing is regime-specific.
Knowledge update: Est #41 needs scoping — sense_sat regularizes secrete/diffusion/decay but NOT vmax aliasing.

## Batch 16 — summary

- **BEST LOSS OF BATCH:** sense_sat.gain=500 -> loss=**0.9802** inner=0.372 (sw 5); secondary sat_n=2.25 at c_sat=0.30 -> 1.0305 (sw 13); sat_n=4.0 at c_sat=1.0 -> 1.0305 (sw 12). All morphologically credible 5-7 mound configs.
- **PARENT IS ROBUST:** sw 0 seed sweep shows the B16 parent reproduces 5-7 distinct dense compact mounds at every seed, sigma_loss=0.04 (the lowest noise floor measured in 16 batches). The Est #43 lesson works: adopting the WHOLE-CONFIG morphology winner from B15 sw 7 (sat_n=3.0, c_sat=0.50) produced a robust parent.
- **DECISIVE TESTS RESOLVED:** sw 1 confirms Est #4 (relay necessary at sat_n=3.0 — gain=0 sparse, gain=30/60 catastrophic ringing, gain>=90 plateau); FALSIFIES Est #47 (relay destructive at broken parent only). sw 2 confirms low kadh wins (kadh=6 sharp lower cutoff, but plateau extends to kadh=300 morphologically identical). sw 5 EXTENDS Est #39 — gain monotone-down to 500 with reversal at 800-1000.
- **SENSE_SAT REGULARIZATION (Est #36/#41) FURTHER GENERALIZED:** sw 7 (inflow flat to rate=14), sw 8 (camp.decay flat across [0.02, 0.40]), sw 10 (secrete.rate explosive dispersion REGULARIZED), sw 11 (camp.diffusion flat across 0.0001-0.08). The B12 Est #29 failure-mode map is largely DISMANTLED in the dense regime — except the vmax aliasing wall (sw 15) which RETURNS at sat_n=3.0.
- **NEW (c_sat x sat_n) RIDGE:** sw 4, 12, 13 trace a trade-off curve — at c_sat=0.30 sat_n>=2.25, at c_sat=0.50 sat_n>=2.75, at c_sat=1.0 sat_n>=3.5 — all producing equivalent multi-mound at loss ~1.03-1.10. Possible alternative parent: c_sat=0.30, sat_n=2.25 (slightly lower loss).
- **CELL.N IS NOT A DENSIFIER (re-confirmed):** sw 9 shows cell.n is flat across [400, 3200] morphologically and on loss; the 8-mound target cannot be reached by adding cells alone.
- **NEW B17 PARENT (conservative — adopt only clear monotone winners):**
  - sense_sat.gain: 240 -> 500 (sw 5 peak; clear monotone with reversal)
  - secrete.rate: 4 -> 11 (sw 10 monotone-down)
  - spring.kadh: 15 -> 10 (within sw 2 plateau; avoid the sharp kadh<6 catastrophe with margin)
  - All other unchanged: sat_n=3.0, c_sat=0.50, r_on=0.20, decay=0.07, relay.gain=140, inflow=4, vmax=0.061, D=0.0012, cell.n=1000.
- **STRATEGIC FRAME for B17:** the morphology gap from REAL~8 mounds is now narrow (5-7 mounds at B16). The remaining residual is DENSIFICATION-PER-MOUND. Refinement axes: (a) extend sense_sat.gain past 500 to confirm reversal; (b) map the (c_sat, sat_n) ridge to find the global loss minimum; (c) probe joints (kadh x gain, r_on x gain) to test Est #43-style transitivity at the new parent; (d) re-test all adopted axes at the joint parent; (e) push inflow WIDER ([0, 30]) since Est #29 over-dilution is dismantled; (f) NEW SWEEP: sense_sat.gain x c_sat=0.30 to see if the sparser regime densifies under high gain.

## Batch 18 Sweep 0 — seed [validates cell.n=1800 single-axis change]
Hypothesis (H1-B18): cell.n=1800 preserves Est #48 reproducibility.
Response: flat-noisy [0.926, 1.085], sigma~0.04, mean~0.99; best seed=10 at 0.9263.
Morphology (from strip): ALL 16 seeds produce visibly clean 5-7 dense compact mounds; morphology indistinguishable from B17 sw 0 (which used cell.n=1000). REAL at end shows ~8 mounds with tighter packing.
Verdict: SUPPORTED — cell.n=1800 preserves the Est #48 morphology AND the noise floor sigma=0.04. The single conservative change is safe.
Knowledge update: Est #48 holds at cell.n=1800; noise floor unchanged.

## Batch 18 Sweep 1 — sense_sat.gain [500, 5000] [densification plateau, no mound-count breakthrough]
Hypothesis (H2-B18): extreme sense_sat.gain breaks mound-count ceiling toward 8.
Response: flat-noisy [0.94, 1.17], no clear monotone trend; best gain=2000 at 0.9403; mid-range bump at gain=2600 (1.167) and 2800 (1.110).
Morphology (from strip): EVERY gain in [500, 5000] produces 4-6 dense compact mounds; no mound-count increase visible at any gain. Gain=3000-5000 shows slightly tighter individual mounds but same count. NO 8-MOUND BREAKTHROUGH.
Verdict: PARTIALLY FALSIFIED — extreme gain at c_sat=0.50 plateaus on loss AND on mound count; the densification handle of Est #53 (which was at c_sat=0.30) does NOT extend to c_sat=0.50 in any productive way. Pure parameter push at parent c_sat cannot reach 8 mounds.
Knowledge update: gain is a plateau parameter at c_sat=0.50 (Est #54 reconfirmed); extreme gain does NOT add mounds.

## Batch 18 Sweep 2 — sense_sat.c_sat [0.2, 1.2] [robust plateau, ridge confirmed]
Hypothesis (H3-B18): c_sat at gain=500 has interior optimum or mound-count breakthrough.
Response: flat 0.92-1.01; best c_sat=1.2 at 0.9216; mild dip at c_sat=0.85 (0.947), c_sat=0.27 (0.940).
Morphology (from strip): EVERY c_sat in [0.20, 1.2] preserves 4-6 dense compact mounds — visually nearly identical across the entire range. Sparse-multi regime (c_sat<=0.20) NOT entered at this gain.
Verdict: SUPPORTED — at gain=500 the (c_sat, sat_n) ridge is robust across c_sat in [0.20, 1.2] (gain compensates for the sat_n threshold rise per Est #57). Loss/morphology gap is ridge-flat.
Knowledge update: confirms Est #57 RIDGE robustness at the joint parent; no c_sat value moves mound count.

## Batch 18 Sweep 3 — sense_sat.sat_n [1.8, 4.5] [monotone-up loss; ridge sharp edge]
Hypothesis (H4-B18): sat_n at c_sat=0.25 ridge column finds more mounds.
Response: MONOTONE-UP loss from sat_n=1.9 (0.934) to sat_n=4.5 (1.175); inflexion near sat_n=2.5-2.7 (1.04-1.08); best sat_n=1.9 at 0.9338.
Morphology (from strip): sat_n=1.8-2.2 = 5-7 dense multi-mound (best); sat_n=2.4-3.2 = 4-5 sparser multi; sat_n=3.6-4.5 = SPARSE 2-4 tiny widely-spaced spots (per-mound density collapsed). sat_n above 3.0 is destructive in the dense regime parent.
Verdict: SUPPORTED — sat_n=3.0 (parent) is at the EDGE of the productive plateau, not the center. sat_n in [1.9, 2.5] gives slightly better loss AND visibly more dense mounds. CANDIDATE: lower sat_n toward 2.0-2.5 in B19.
Knowledge update: Est #45 (sat_n=3.0 parent) refined — at c_sat=0.50 + gain=500 the sat_n optimum sits at 1.9-2.5; the ridge has CURVATURE, not flat from 2.75 up.

## Batch 18 Sweep 4 — spring.kadh [3, 280] [plateau confirmed, low-kadh wins]
Hypothesis (H5-B18): kadh densification optimum in the dense regime.
Response: flat-noisy [0.93, 1.05]; best kadh=6 at 0.9352; mid-range bumps at kadh=8 (1.012), 60 (1.032), 80 (1.046); plateau from kadh=12-280 around 0.94-0.99.
Morphology (from strip): EVERY kadh value produces 4-6 dense compact mounds; mound count invariant; per-mound density slightly tighter at high kadh (>=80) but no count change. Lower kadh=3-5 produces somewhat more diffuse mounds.
Verdict: SUPPORTED — kadh is a wide plateau across [6, 280] (extends Est #52/#56). The B17 sw 2 plateau extends further than B16. No densification advantage from extreme kadh.
Knowledge update: kadh plateau confirmed [6, 280]; drop further kadh refinement.

## Batch 18 Sweep 5 — spring.r_on [0.13, 0.245] [dip at 0.19-0.215; upturn at 0.23+]
Hypothesis (H6-B18): r_on densification probe at parent.
Response: dip at r_on=0.19 (0.9253) and 0.215 (0.9272); upturn at r_on=0.23 (0.9969) and 0.245 (1.0945); best r_on=0.19 at 0.9253.
Morphology (from strip): r_on=0.13-0.18 = MORE DIFFUSE multi-mound (looser packing, more cells outside mounds); r_on=0.19-0.22 = clean dense 4-6 mounds (parent regime); r_on=0.23-0.245 = OVER-COMPACT single tight central blob with high inner_mass (0.52). The Est #3 over-compact transition begins at r_on>=0.23.
Verdict: SUPPORTED — r_on=0.19-0.215 is the productive band at this parent; the over-compact transition starts at r_on=0.23 (consistent with Est #3 boundary). r_on=0.19 is marginally better than parent 0.20 within noise.
Knowledge update: r_on working band refined to [0.18, 0.22] at B18 parent; r_on>=0.23 collapses to single blob.

## Batch 18 Sweep 6 — relay.gain [0, 1800] [Est #4 strongly reconfirmed]
Hypothesis (H7-B18): relay necessity + plateau at densification parent.
Response: gain=0 ABLATION = 1.6579 (sparse, single dispersed cluster, NO aggregation); gain=60 = 1.3127 (partial aggregation, weak mounds); gain=90 = 1.0534 (transition); gain>=120 plateau [0.93, 1.07]; best gain=160 at 0.929.
Morphology (from strip): gain=0 = SPARSE scatter (no aggregation); gain=60 = ONE diffuse glob (transition); gain=90-200 = clean 5-7 dense multi-mound; gain=240-500 = same; gain>=700 = mounds slightly sparser/diluted but still multi.
Verdict: SUPPORTED — Est #4 (relay necessary) STRONGLY RECONFIRMED at B18 parent. The B17 sw 7 catastrophe band [30, 60] now narrows: gain=60 is sub-optimal but not catastrophic at cell.n=1800. Plateau is gain in [120, 1800], very wide.
Knowledge update: Est #4 + Est #55 reconfirmed at cell.n=1800; relay plateau extends to gain=1800; gain<=60 sub-optimal but not catastrophic at higher cell.n.

## Batch 18 Sweep 7 — camp.decay [0.04, 2.0] [BLOCKBUSTER plateau extension]
Hypothesis (H8-B18): camp.decay wall at high values in dense regime.
Response: flat across the ENTIRE range; best decay=1.8 at 0.9198; mild spikes at 0.07 (1.055), 0.10 (1.099), 0.40 (1.013), 1.60 (1.078); but no catastrophe even at decay=2.0 (0.927).
Morphology (from strip): EVERY decay in [0.04, 2.0] produces clean 4-6 dense multi-mound. The Est #29 "field-dies wall at decay>0.40" is COMPLETELY DISMANTLED; decay=2.0 is over 28x the parent value and still produces a credible aggregation.
Verdict: SUPPORTED (and DRAMATIC) — Est #56 decay extension goes from "flat to 0.85" to "flat to 2.0+". The dense regime's regularization on camp.decay is total. Mechanistic interpretation: sense_sat saturates effective gain in mounds, so the field-decay rate ceases to matter for keeping mounds intact.
Knowledge update: Est #56 EXTENDED — camp.decay plateau is [0.04, 2.0+]; the field-dies failure mode is GONE in the dense regime.

## Batch 18 Sweep 8 — inflow.rate [0, 150] [ULTRA-WIDE plateau; over-dilution wall GONE]
Hypothesis (H9-B18): inflow over-dilution wall at extreme rates.
Response: flat [0.94, 1.06] across [0, 150]; best rate=10 at 0.9463; mid-range spike at rate=15-30 (1.015-1.017); flat at rate=40-150.
Morphology (from strip): EVERY rate in [0, 150] preserves 4-6 dense multi-mound morphology. Even rate=150 (37x parent) shows credible aggregation. The n_final at rate=150 is enormous (>>10000) but the dense regime still self-organises.
Verdict: SUPPORTED (and DRAMATIC) — the inflow over-dilution wall is COMPLETELY ABSENT. Est #56 extension confirmed and pushed FAR beyond. Inflow is essentially a free parameter in the dense regime.
Knowledge update: Est #56 EXTENDED — inflow plateau extends to rate=150; no over-dilution wall observable.

## Batch 18 Sweep 9 — cell.n [600, 3380] [marginal high-n dip; mound count INVARIANT]
Hypothesis (H10-B18): cell.n densification in the dense regime.
Response: flat [0.92, 1.01]; mild downward trend from n=2200 (0.922) to n=3300 (0.931); BEST OF BATCH at n=3200 (0.9167); also n=3000 (0.9254).
Morphology (from strip): EVERY n in [600, 3380] produces 4-6 dense mounds; mound count INVARIANT; per-mound density INCREASES marginally with n (mounds slightly brighter at n>=2200), but the count never changes. The engine is solvent through n=3380 (no NaN at the capacity limit).
Verdict: PARTIALLY SUPPORTED — high n marginally improves loss AND per-mound density without changing mound count. cell.n=3200 is the BEST LOSS of B18 (0.9167) but within seed noise (sigma=0.04, vs delta=0.01 vs B17 best 0.9268). Est #52 (cell.n not a densifier of mound count) RECONFIRMED.
Knowledge update: cell.n marginal candidate for B19 parent — high n improves per-mound density without breaking the count ceiling. BUT within seed noise.

## Batch 18 Sweep 10 — secrete.rate [8, 32] [MONOTONE morphology DEGRADATION at high rate]
Hypothesis (H11-B18): secrete fine band refinement.
Response: monotone-up loss from rate=10 (0.939) to rate=32 (1.190); inner_mass DROPS monotonically from 0.31 (rate=8) to 0.064 (rate=32, near-zero — cells dispersed).
Morphology (from strip): rate=8-12 = clean 4-6 dense mounds; rate=13-16 = same with mild fade; rate=17-20 = sparse tiny mounds (per-mound density collapses); rate=22-32 = SPARSE TINY 1-3 SPOTS (dispersion regime). The dense regime fails morphologically above rate=15 even though loss only rises by 25%.
Verdict: SUPPORTED — secrete is NOT a plateau past 11; morphology DEGRADES monotonically. Est #56 secrete safe band [8, 14] CONFIRMED; rate>=15 has progressive dispersion (not catastrophic like the legacy regime, but morphology-breaking).
Knowledge update: secrete safe band [8, 14] at B18 parent; outside this band morphology degrades.

## Batch 18 Sweep 11 — vmax [0.052, 0.0735] [aliasing resonance at 0.063, 0.064]
Hypothesis (H12-B18): vmax fine working dips avoiding 0.065 resonance.
Response: flat [0.93, 1.06] for vmax in [0.052, 0.062]; SHARP spikes at vmax=0.063 (1.649) and vmax=0.064 (1.446); flat again [0.93, 0.95] at vmax=0.068-0.0735.
Morphology (from strip): vmax<=0.062 = 4-6 dense mounds; vmax=0.063-0.064 = morphology BREAKS (cells stack to single tight central spot, inner_mass=0.42 but single-blob); vmax=0.068-0.0735 = recovers dense multi-mound.
Verdict: SUPPORTED — Est #9/#51 aliasing wall RECONFIRMED at vmax=0.063-0.064 (the resonance moved DOWN from 0.065 — possibly because cell.n=1800 shifts the per-step displacement profile). Working bands: [0.052, 0.062] and [0.068, 0.073].
Knowledge update: Est #9 resonance peak refined to vmax=0.063-0.064 at B18 parent; widely tolerated otherwise.

## Batch 18 Sweep 12 — spring.r0 [0.008, 0.046] [SILENT, parameter free]
Hypothesis (H13-B18): r0 modulates per-mound packing density.
Response: flat-noisy [0.93, 1.01]; best r0=0.026 at 0.9248; no monotone signal.
Morphology (from strip): EVERY r0 produces 4-6 dense multi-mound morphology; per-mound packing is visually IDENTICAL across the range. r0 is fully silent at the B18 parent.
Verdict: FALSIFIED (Est candidate) — r0 has no productive optimum and no morphological effect in the dense regime. Re-confirms B13 sw 15 silence.
Knowledge update: r0 SILENT across two batches now; drop permanently from refinement.

## Batch 18 Sweep 13 — sense_sat.gain [500, 7000] [extreme push; sparser at top]
Hypothesis (H14-B18): extreme gain at sparse column densifies to 8 mounds.
Response: mild dip 1500-2400 (0.93-0.94); upturn at 7000 (1.045); best gain=1800 at 0.9288; clear monotone-up of inner_mass loss past gain=4500.
Morphology (from strip): gain=500-1500 = 4-6 dense multi-mound; gain=1800-3500 = same; gain=4000-5000 = sparser-multi (per-mound density LOSS); gain=6000-7000 = SPARSE TINY 2-3 SPOTS. Extreme gain DILUTES mounds rather than adding them. NO 8-MOUND BREAKTHROUGH.
Verdict: PARTIALLY FALSIFIED — gain past 3500 produces sparser morphology, not more mounds. Est #53 densification axis saturates by gain=2000-3000; pushing further is counterproductive.
Knowledge update: densification axis saturation point identified at gain~2000-3000; past this, sparse-tiny degradation. The 7-mound ceiling holds.

## Batch 18 Sweep 14 — sense_sat.gain [500, 5000] [duplicate-range; similar shape]
Hypothesis (H15-B18): gain extension at another c_sat regime.
Response: flat-noisy [0.93, 1.24]; spike at gain=1700 (1.240); best gain=2300 at 0.9311; flat at high gain.
Morphology (from strip): SIMILAR shape to sw 13 — dense multi-mound across most of the range; sparse-tiny at extreme gain. Re-confirms saturation by gain~2000-3000.
Verdict: SUPPORTED (re-test of sw 13) — gain saturation point reconfirmed; no morphological breakthrough at extreme gain.
Knowledge update: redundant confirmation; reinforces sw 13 conclusion.

## Batch 18 Sweep 15 — camp.diffusion [0.0001, 0.1] [NEW WALL discovered at D>=0.05]
Hypothesis (H16-B18): camp.diffusion sets mound-wavelength; lower D → more mounds.
Response: flat [0.93, 1.01] for D in [0.0001, 0.035]; CATASTROPHIC at D=0.05 (1.852), 0.07 (1.999), 0.1 (5.767). Best D=0.012 at 0.9304.
Morphology (from strip): D<=0.035 = 4-6 dense multi-mound (no wavelength-modulation effect — mound count is INVARIANT); D=0.05-0.1 = morphology FALLS APART, cells dispersed in a diffuse cloud, NO aggregation. The B16 sw 11 "flat across 2 orders of magnitude" (which only went to 0.08) MISSED this wall.
Verdict: PARTIALLY FALSIFIED on wavelength (no mound-count effect at low D), but a NEW FAILURE MODE WALL identified at D>=0.05. The dense regime tolerates D over 2 orders of magnitude in [0.0001, 0.035] but breaks past D=0.05 (smearing exceeds sense_sat's regulating capacity).
Knowledge update: NEW Est candidate — camp.diffusion CATASTROPHIC WALL at D>=0.05. Refines Est #56 (which was "essentially free"); this is the new diffusion ceiling.

## Batch 18 — summary

- **BEST LOSS OF BATCH:** cell.n=3200 → loss=**0.9167** inner=0.376 (sw 9); marginal over B17 best=0.9268 (within seed sigma=0.04 noise). The "win" is one tick within the noise floor.
- **STRUCTURAL CEILING UNBROKEN:** all 16 sweeps × 16 values produce 4-7 mound morphology; **no sweep produces 8 mounds**. The Est #53 densification axis (gain × c_sat=0.30 found in B17) does NOT extend by extreme parameter push (sw 1/sw 13/sw 14 all flat-to-degrading past gain=2000). The 5-7 mound ceiling is now confirmed across 18 batches.
- **DRAMATIC dismantling of failure modes (Est #56 EXTENDED FURTHER):** camp.decay plateau extended to 2.0+ (sw 7); inflow rate plateau extended to 150 (sw 8); kadh plateau extended to 280 (sw 4); c_sat plateau extended to 1.2 (sw 2). The dense regime is extraordinarily robust.
- **NEW WALL DISCOVERED:** camp.diffusion CATASTROPHIC at D>=0.05 (sw 15). The B16 sw 11 "flat to 0.08" was within the new wall (which sits at D=0.05). Refines Est #56.
- **RIDGE EDGE REFINED:** sat_n=3.0 (current parent) is at the EDGE of the productive ridge plateau; sat_n in [1.9, 2.5] at c_sat=0.50 gives marginally better loss AND visibly denser mounds (sw 3).
- **r0 SILENCE CONFIRMED:** sw 12 r0 fully silent across two batches; drop.
- **STRATEGIC FRAME for B19:** parameter densification has now plateaued. Per the open Q (per-cell secretion heterogeneity, untested) the next MECHANISM to test is **`secrete_het`** — a per-cell log-normal multiplier on secretion (added to `dicty_ops.py`, ablation = het_std=0). B19 sweeps test (a) `secrete_het.het_std` necessity + sufficiency with strength=0 ablation; (b) joints with the densification axes (het_std × c_sat=0.30, het_std × gain=1500, het_std × cell.n); (c) seed sweep at new parent; (d) re-test new D wall; (e) sat_n FINE in [1.9, 2.5] (sw 3 candidate window). Drop further pure-parameter densification at c_sat=0.50.
- **B19 PARENT:** conservative — KEEP B18 parent unchanged (cell.n=1800), ADD `secrete_het` with `het_std=0.0` (ablation) replacing plain `secrete` in the base spec. Sweeps will introduce non-zero het_std.

> Batch 20 per-sweep entries + summary written to `_b20_log.md` (matching `_b18_log.md`, `_b19_log.md` convention). Append blocked at runtime; the file is the canonical artefact.

## Batch 21 — see `_b21_log.md` (canonical per-sweep entries + summary)

> B21 INVALID for diff_dens adjudication — operator bug discovered: DiffDens.forward read `fld.diffusion` but `GridField` stores `self.D`, so D0=0 and the op was a silent no-op across all 6 diff_dens sweeps. Sweep 13 (seed) was a redundant duplicate of sweep 0. Net 9 informative sweeps; parent loss 0.9111 unchanged from B20; secrete=9 confirmed (sw 8 dip at 9, plateau [9.6, 11.5] within noise), sat_n=2.1 plateau broadened to [1.7, 3.2] (sw 14), camp.D wall location tightened to D~0.045 transition (sw 9), vmax aliasing reconfirmed at 0.06/0.0615 working dips (sw 7), relay.gain plateau extended to 320 (sw 6), Est #71 cell.n flat-noisy [800, 3400] reconfirmed (sw 12). SIDE-FINDINGS at fixed-joint kappa-silent sweeps: c_sat=0.30 (sw 2) and gain=1500 (sw 3) and relay.thr=0.30 (sw 10) all produce 5-mound morphology near parent loss; kadh=20 (sw 11) produces the most visually-crisp 5-6 multi-mound of the batch at loss=1.25 (SSIM-penalised, Est #42). `dicty_ops.py:918` patched to read `fld.D` with `fld.diffusion` fallback. B22 re-tests fixed diff_dens with 11 joint configurations.

## Batch 22 — see `_b22_log.md` (canonical per-sweep entries + summary)

> B22 DECISIVELY FALSIFIES diff_dens — FOURTH field-side mechanism. With `dicty_ops.py:918` patched (D0 now reads `fld.D`), every non-zero kappa is CATASTROPHIC at the parent (sw 1) AND at 6 of 7 productive joints (sw 2-6, 8, 9). Two seed sweeps at kappa=2 (sw 10) and kappa=20 (sw 11) CONFIRM the catastrophe is not seed-luck — uniformly worse than ablation across all 16 seeds (median 1.27 vs 0.99). Failure mode = field annihilation (Laplacian-correction term subtracts more cAMP than the field can regenerate); same failure mode as decay_dens (Est #68). EXCEPTION: sw 7 (kappa × camp.D=0.005) — at elevated baseline D, kappa>0 RESCUES multi-mound morphology (kappa=0 → 1.227, kappa=0.8 → 1.030), but the rescued loss is still 12% above parent (0.911). This is a self-cancelling pair (raise D, locally suppress it), not a mechanism — Est #78. PLATEAU REFINEMENTS reconfirm parent: sat_n=2.1 (sw 12), secrete.rate=9 (sw 13), vmax=0.060 (sw 14), camp.D working band [0.0001, 0.036] (sw 15). NO new project best — multiple ties at 0.9111. Parameter surface fully exhausted; operator-side mechanism family DRY (4 field-side + 6 cell-side falsified). B23 strategic frame: schedule DENSITY-TRIGGERED PULSE (the only untested structural candidate in the current operator family — deterministic local burst when ρ(x)>θ, distinct from FALSIFIED random Poisson nucleation B6 and homogeneous pacemaker B5). If B23 also fails, escalate to engine change or metric augmentation (Est #42 flag is live). B23 parent: B22 parent + REVERT secrete.rate 9→11 to recover Est #73 tighter σ≈0.04 noise floor (sw 13 confirms rate=9 and rate=11 are tied at 0.911) — essential for pulse_dens adjudication statistical power.

## Batch 24 — see `_b24_log.md` (canonical per-sweep entries + summary)

> **B24 ENGINE-RESOLUTION PROBES DECISIVELY NEGATIVE; CRITICAL NEW STRUCTURAL FINDING — no stable multi-mound attractor.** The B24 plan promised to (a) implement metric augmentation (`morph_score` = peak-count + per-spot-density) in `eval_sweeps.py` AND (b) systematically probe engine-resolution parameters (camp.res, dt, n_frames) that had never been swept in 23 batches. **(a) was not implemented** — sweep_results.json has only `loss` and `inner_mass`; the DECISIVE FORK question (is parameter-flatness metric-induced?) is UNADJUDICATED in B24. **(b) ran and is decisive:** TWO independent camp.res sweeps (sw 5 parent, sw 15 c_sat=0.30) show mound count INVARIANT across grid resolution in [112, 360]; dt sweep (sw 7) shows mound count INVARIANT across working dt bands; n_frames sweeps (sw 6 parent, sw 14 densification) reveal **THE SMOKING GUN** — the point-cell engine with the current operators has NO STABLE MULTI-MOUND ATTRACTOR; given longer integration time (n_frames=1600 vs parent 400), every configuration GRINDS toward a single tight blob (inner_mass monotone-UP to 0.83). Multi-mound morphology is a TRANSIENT, not an equilibrium. Refines Est #31 ("equilibrates by frame 400") to "the model is *evaluated* at frame 400 before the next collapse phase". sw 10 (r_on at c_sat=0.30 densification joint) shows r_on=0.23 produces visible 4-mound morphology — the most multi-mound result of B24 and a candidate morph_score winner. Plateau extensions: gain to 7000 (Est #54), c_sat to 1.2 (Est #57), sat_n productive plateau at c_sat=0.30 = [1.5, 2.7] (refines Est #61). RW silent across 9 batches now. NO new project best (sw 0 seed=0 = 0.9126 retained). **B25 STRATEGIC FORK:** (α) IMPLEMENT morph_score (missed B24 deliverable) AND (β) add `density_repel` operator (density-saturating short-range repulsion) — directly addresses the sw 6 runaway-compaction failure mode by introducing finite-cell-volume exclusion. The sw 6 finding makes (β) more defensible than the MPM engine fork: a single local-operator addition can plausibly create a stable multi-mound attractor without rewriting the engine. If `density_repel` also fails, escalate to MPM fork (`dicty_engine_mpm.py`).

## Batch 23 — see `_b23_log.md` (canonical per-sweep entries + summary)

> B23 DECISIVELY FALSIFIES pulse_dens — NINTH project mechanism (FIFTH field-side after pacemaker, inhibitor, decay_dens, diff_dens). Across SEVEN amplitude sweeps (sw 1 necessity at parent; sw 3-7 at five productive joints — c_sat=0.30, gain=1500, cell.n=2500, kadh=20, relay.thr=0.30) + sw 2 threshold sweep + sw 8 high-amplitude seed sweep (16 seeds at amp=2.0 all in [1.14, 1.20], DISJOINT from sw 0 ablation [0.91, 1.00]), ablation (amp=0) wins; every amp>=0.05 monotone-up dispersion failure. Failure mode = local pulse pushes cells AWAY from dense regions (as designed) but they DO NOT nucleate new mounds — they just disperse. Same family as decay_dens (Est #68) / diff_dens (Est #78). Promoted to Est #80. **OPERATOR-SIDE MECHANISM FAMILY DEFINITIVELY EXHAUSTED** — 10 mechanisms falsified across 23 batches (5 field-side + 6 cell-side; sense_sat the only structural addition that succeeded, as a regularizer). Parameter surface flat at loss=0.911 (B23 ties at 0.9126 under tighter σ≈0.04 noise floor recovered by secrete=11 revert). PLATEAU REFINEMENTS reconfirm parent: sat_n=2.1 (sw 9), gain plateau no peak (sw 10 — Est #50 retraction permanent), vmax=0.061 + aliasing wall vmax>=0.075 (sw 11), camp.D working band [0.0001, 0.035] ringy transition (sw 12), c_sat ridge plateau [0.20, 1.50] (sw 13), relay.gain=140 (sw 14), inflow.rate=4 (sw 15). NO new project best. **B24 escalates beyond operator addition: METRIC AUGMENTATION first (cheaper, reversible; tests whether the parameter-surface flatness is metric-induced via Est #42 SSIM/morphology divergence), with the existing engine. If the augmented metric remains flat, that DEFINITIVELY indicts the engine and motivates the `dicty_engine_mpm` fork as B25+.**

## Batch 25 — see `_b25_log.md` (canonical per-sweep entries + summary)

> **B25 BREAKTHROUGH ON METRIC, NOT MECHANISM.** Implemented `morph_score` (peak count + per-spot density; missed B24 deliverable) and added new operator `density_repel` (saturating short-range repulsion; biological motivation = finite cell volume). morph_score IMMEDIATELY revealed hidden 8-mound configurations across NINE sweeps (sw 1, 2, 3, 4, 8, 9, 10, 11, 13, 14, 15) with morph_score≤0.005 at multiple settings — the long-suspected Est #42 SSIM/morphology divergence (live since B14) is **FINALLY ADJUDICATED**: the parameter-flatness was METRIC-INDUCED; the model's parameter cube ALREADY contained 8-mound configurations hidden by SSIM. **Est #58 5-7 mound STRUCTURAL CEILING — RETRACTED as metric artefact.** `density_repel`: **SUFFICIENT under morph_score at parent** (strength=0.05-0.55 → nm=8, sw 1), **NOT NECESSARY** (ablation also reaches nm=8 at r_on=0.19/0.23, kadh=20, c_sat=0.8/1.2). First clean morphological PRODUCTIVE role: **sw 5 rescues runaway single-blob at cell.n=3000** (strength=0 → nm=1; strength=0.5-3.5 → nm=4-9). **FAILS Est #82 break test (sw 6, n_frames=1200):** runaway compaction completes regardless of density_repel strength; the missing mechanism is NOT finite cell volume in this form. Engine fork (MPM) remains live. Project-best loss UNCHANGED at 0.9126 (parent, sw 0/10). Project-best morph_score ≈ 0 at sw 15 strength=3.5 (kadh=20, nm=8, loss=0.9381). **B26 PARENT: adopt spring.r_on=0.19** (sw 8 morph minimum 0.0003, nm=8, loss=0.9277). Keep density_repel.strength=0 ablation; revisit at high-cell.n joint where density_repel IS productive. **B26 STRATEGIC FRAME**: re-evaluate all B19-B23 falsified field/cell-side mechanisms (decay_dens, diff_dens, pulse_dens, secrete_het) under morph_score — they were ruled out by a metric we now know was blind to mound count; some may now be productive. Also test whether the Est #82 runaway compaction persists at the new r_on=0.19 parent.

## Batch 27 — see `_b27_log.md` (canonical per-sweep entries + summary)

> B27 PROJECT-BEST LOSS UNCHANGED at 0.9126 (sw 0 seed=0, sw 1 r_on=0.20, sw 10 sat_n=2.1, sw 12 relay.gain=140 all tied). STRONGEST FINDING — Est #97: sw 2 secrete_het.het_std=0.20 produces nm=8 morph=0.0001 loss=0.9152 (within seed-noise of parent). Est #63 (B19 secrete_het falsified by SSIM-loss) PARTIALLY RETRACTS — narrow productive window [0.10, 0.25] hidden in B19. PRONG (β-redo) OTHERWISE HOLDS: decay_dens (sw 3), pulse_dens (sw 4), diff_dens (sw 5) re-evaluation under morph_score with the spec NOW correct (Est #95 fix) — all three falsifications HOLD. Est #68/#80/#78 confirmed under both metrics. PRONG (δ) 8-mound corner VINDICATED: sw 8 reveals (r_on=0.19, kadh=20, gain in [750, 3500]) is a 4-point morph_score=0 corridor (NEW Est #98); c_sat=1.0 transfers cleanly (sw 9). Loss-cheapest 8-mound: gain=750 -> loss=0.9316 (1.04x parent). PRONG (γ) DECISIVE — Est #82 GENERALIZES AND WORSENS AT 8-mound corner (NEW Est #100): sw 14 reconfirms parent collapse by n_frames=1050; sw 15 at the 8-mound corner — collapse FASTER (nm=1 by n_frames=750). The 8-mound morphology is a TRANSIENT, never an equilibrium. Other morph candidates: r_on=0.185 (sw 1), sat_n=1.4 (sw 10), relay.gain=290 (sw 12), camp.D corridor [0.0022, 0.0055] (sw 13, Est #99). density_repel.strength=4 rescues n=4500 but destroys n<=600 (Est #93 retraction refined). B28 FRAME: (ζ) verify Est #97 via 16-seed sweep at het_std=0.2 + het FINE + densification joints (decisive seed-luck vs productive test, analogue of B22/B23); (η) DRAFT adh_cap (mass-cap adhesion: per-cell rho gate ATTENUATES spring adhesion when rho>thr — REMOVES adhesion in dense regions rather than adding outward force; distinct from density_repel which FAILED Est #82) for B29 implementation. B28 parent UNCHANGED (het_std=0 ablation) pending verification.

## Batch 28 — see `_b28_log.md` (canonical per-sweep entries + summary)

> B28 DECISIVELY RETRACTS Est #97 (secrete_het=0.20 productive interior). 256 sims at B27 parent. **(1) SW 0 16-SEED VERIFICATION FAILS (NEW Est #101):** distribution wide (sigma~0.036), morph wild — only 4/16 seeds at morph<=0.05; B27 sw 2 morph=0.0001 was 1/16 seed-luck. B19 Est #63 (secrete_het falsified) STANDS REINSTATED under both metrics. **(2) Est #82 NOT MITIGATED by het (sw 5) or kadh-attenuation (sw 15, ADH_CAP PILOT, NEW Est #102):** every het value at n_frames=1200 collapses to nm=1; every kadh in [1, 200] at n=3000/n_frames=1200 collapses to nm=1; kadh=0 explodes. Adh_cap design FALSIFIED IN PILOT — chemotactic pull dominates regardless of adhesion. **(3) Est #98 8-mound corner reconfirmed and refined:** gain band broad [600, 3500] (sw 6), kadh band [10, 22] (sw 7), r_on band [0.183, 0.198] AND new candidate [0.220, 0.225] (sw 8); Est #99 D corridor under het=0.20 [0.0018, 0.0042] (sw 10). **(4) Est #100 corner collapse is GAIN-INVARIANT** (sw 14 at gain=750 collapses by n_frames=750 identically to B27 sw 15 at gain=2500). **(5) pulse_dens (Est #80) catastrophe is AMPLITUDE-driven across all thr (sw 13).** **(6) Project-best loss = 0.909 at sw 0 seed=5 het=0.20 — within seed noise** of parent 0.9126, NOT a genuine improvement. **B29 PARENT: revert het_std=0; otherwise unchanged. STRATEGIC ESCALATION: OPERATOR-SIDE FAMILY EXHAUSTED — 11 mechanisms falsified across 28 batches. The ENGINE FORK to `dicty_engine_mpm.py` is the principled next move. B29 ports B28 parent to MPM, sweeps MPM-native params + n_frames=1200 break test. Secondary point-cell sweeps re-pin parent, test one final adh_cap variant (gate sense_sat output not spring), and probe Est #100 corner-timescale × camp.decay.**

## Batch 29 — see `_b29_log.md` (canonical per-sweep entries + summary)

> **B29 SURPRISE — FIRST POINT-CELL Est #82 PARTIAL RESCUE.** B29 was positioned as the final point-cell closing batch with B30 as MPM fork. sw 4 (8-mound corner: r_on=0.19, kadh=20, gain=1500 + camp.decay=1.4 + n_frames=1200) produces nm=6, inner_mass=0.373, loss=0.9863 — the FIRST point-cell configuration to sustain multi-mound at the Est #82 break test since B24. CORNER × DECAY interaction (sw 3 at parent with same decay collapses). NEW Est #104; requires 16-seed verification in B30 before parent adoption. **D=0.0042 SEED VERIFIED (sw 6, 4/16 morph=0 seeds, NEW Est #105) — B30 PARENT adopts camp.diffusion=0.0042.** FOUR new falsifications (sense_sat.gain attenuation Est #106 = 12th project mechanism, RW × n_frames=1200 Est #107, vmax × n_frames=1200 Est #108, stacked density_repel × corner × n_frames=1200 Est #110) close the operator-side rescue family. New r_on=[0.220, 0.225] candidate corner (B28 sw 8) FALSIFIED as a corridor (Est #103); r_on=0.222 as single-bin corner persists (sw 13/15). Est #100 generalized: ALL tested 8-mound corners are kinematic transients (Est #111). Project-best loss 0.9066 at seed=7 r_on=0.222 — marginal, NOT adopted. **B30 STRATEGIC FORK:** (a) DECISIVE 16-seed verification of Est #104 (corner × decay=1.4 × n_frames=1200) — if it replicates, MPM fork DEFERRED; (b) refine D=0.0042 new parent; (c) verify Est #109 low-decay niche (sw 11 decay=0.02 morph=0.0002 nm=8); (d) Est #104 mechanism probes (n_frames extension, decay variants, corner stacking). If verification FAILS, B31 commits to MPM fork as originally planned.

## Batch 30 — see `_b30_log.md` (canonical per-sweep entries + summary)

> B30 ADJUDICATES Est #104 — DECISIVELY RETRACTS as METRIC ARTEFACT despite numerical pass; sw 0 16-seed verification numerically passes (12/16 nm>=4) but morphology strip shows SINGLE CENTRAL BLOB at every seed (peak detector counts halo speckle). sw 2 (n_frames at Est #104) reveals it is a DELAYED TRANSIENT, not a stable attractor — collapse identical to Est #82 just delayed ~400 frames. NEW Est #112. **DECISIVE B30 FINDING #2:** sw 5 (16-seed at NEW PARENT D=0.0042) shows GENUINE MULTI-MOUND morphology (4-8 dense compact mounds at every seed) — the FIRST point-cell parent in 30 batches to replicate REAL-like morphology under seed variation, tightest noise floor sigma~0.026. NEW Est #113. Est #109 (decay=0.02) VERIFIED across 16 seeds (NEW Est #114). Working bands WIDEN under new D parent: r_on corner DISSOLVES (sw 7 broadly productive); D corridor broadens to [0.001, 0.012] (sw 6); gain/sat_n D-invariant. PROJECT-BEST LOSS = 0.9055 at sw 8 kadh=9 (single-seed). Est #93 high-n collapse holds at n>=3500 (sw 10). Est #104 stacked rescues (density_repel sw 12, c_sat sw 13, sat_n sw 15, capacity sw 4/14) all FAIL — all monomorphic single-blob. **THE CRITICAL TEST STILL UNRUN: n_frames at the BARE NEW PARENT (D=0.0042 only) — does the multi-mound regime SURVIVE to n_frames=1200, or is it (like Est #104) just a delayed transient?** This is the single decisive B31 test. B31 PARENT unchanged from B30 (no single-axis change replicated under morph). B31 strategic frame: (i) n_frames at new parent + n_frames x decay=0.02 (Est #114) — Est #82 break test at the productive regime; (ii) 16-seed verification at kadh=9 and at sw 6 morph-best D=0.002; (iii) refine working bands under new D + test multi-mound robustness at n_frames=1200 across cell.n/decay/kadh joints. If new parent SURVIVES n_frames=1200 with morphology preserved -> Est #82 finally mitigated by structural D shift; MPM fork DEFERRED indefinitely. If it collapses -> MPM fork is the principled escalation (but with EVIDENCE the point-cell engine has at least an n_frames=400 multi-mound regime).

## Batch 31 — see `_b31_log.md` (canonical per-sweep entries + summary)

> **B31 DECISIVELY CLOSES THE POINT-CELL ENGINE — Est #113 RETRACTED, MPM FORK CONFIRMED FOR B32.** 256 sims at B30 parent (D=0.0042 + sat_n=2.1, c_sat=0.50, gain=500, secrete=11, kadh=10, r_on=0.20, decay=0.07, n=1800, n_frames=400). **(1) THE CRITICAL TEST (sw 0, DECISIVE):** n_frames at the BARE NEW PARENT shows the EXACT Est #82 trajectory — multi-mound (nm=10/6/3) at n_frames=200/280/360; SINGLE TINY CENTRAL BLOB + halo speckle by n_frames=480 and ALL endpoints to 2400; inner_mass monotone-UP 0.351→0.895. **THE D=0.0042 MULTI-MOUND REGIME IS A TRANSIENT, NOT A STRUCTURAL MITIGATION.** Est #113 RETRACTED as METRIC ARTEFACT (NEW Est #115; SECOND case after B30 Est #112 retracted Est #104). **(2) DECAY=0.02 VARIANT (sw 1, falsified):** same collapse, ~120 frames delayed; Est #114 RETRACTED (NEW Est #116). **(3) FOUR INDEPENDENT n_frames=1200 STRESS TESTS UNIVERSALLY COLLAPSE:** sw 8 cell.n [800, 5000] → nm=1 (NEW Est #121); sw 9 kadh [5, 35] → nm=1 (NEW Est #122); sw 11 camp.diffusion [0.0008, 0.015] → nm=1 (NEW Est #124 — **DECISIVELY closes the "D shift mitigates Est #82" hypothesis**); sw 15 r_on [0.185, 0.25] → nm=1 in productive band (NEW Est #125; r_on<0.185 enters dispersion). **(4) B30 SINGLE-SEED WINS NOT REPLICATED:** sw 2 16-seed at kadh=9 median≈0.95 σ≈0.022 (B30 sw 8 best 0.9055 = 1/16 seed-luck); sw 3 16-seed at D=0.002 indistinguishable from D=0.0042 (NEW Est #117 — third case of single-seed retraction). **(5) PARAMETER REFINEMENTS UNIVERSALLY FLAT** (sw 4-7, 12-14): D corridor [0.0012, 0.006] (NEW Est #118), camp.decay [0.03, 0.5] (NEW Est #119), kadh [5, 18] (NEW Est #120), cell.n [1100, 3000], sat_n [1.5, 3.0], c_sat [0.3, 1.5], inflow [0.5, 12] — all broad parent-tied plateaus, NO sharp optima. **(6) HIGH-DECAY ANOMALY (sw 10):** decay≥2.0 produces a NEW dispersion regime (gradient annihilation, nm=30-60 sparse-scatter) at catastrophic loss — confirms Est #68 family but is NOT productive (NEW Est #123). **PROJECT-BEST LOSS UNCHANGED at 0.9126** (sw 4 D=0.0012 ties old B30 parent value). **STRATEGIC CONCLUSION:** The point-cell engine + the full registered operator family (12 mechanisms tested, 11 falsified — only sense_sat survives as a REGULARIZER, not a multi-mound-creator) CANNOT sustain multi-mound morphology beyond n_frames~400-600. The n_frames=400 evaluation gates the model entirely within a transient phase. Est #82 (runaway compaction to single attractor) is now confirmed parameter-invariant across cell.n, kadh, camp.diffusion, r_on, AND camp.decay (within productive band) under the new parent. The MPM ENGINE FORK is the PRINCIPLED ESCALATION — MPM cells have intrinsic finite-volume incompressibility (Young's modulus, true volume exclusion through the grid) and cannot collapse below particle radius, which addresses Est #82 at its structural root rather than via operator augmentation. **B32 PARENT = `specs/dicty_mpm_base.yaml`**; B32 sweep_plan = 16 sweeps over MPM-native parameters with sense.gain, secrete.rate, camp.{diffusion,decay}, inflow_mpm.rate, mpm.{drag,vmax,a_max}, cell.{n,youngs}, particle.per_parent + dt + n_frames break test + seed verification. The point-cell parameter map, the 11-mechanism falsification log, and the morph_score metric ALL transfer to MPM as valid context.

## Batch 33 — see `_b33_log.md` (second consecutive infra failure; no science)

> B33 ALL-NaN like B32 — patches saved AFTER B33 sims aborted (file mtimes prove it: sweep_results.json 03:45 < dicty_loop.py 03:47 < eval_sweeps.py 03:48). Identical `inflow_mpm not in registry` failure on every sim. End of dicty_loop_run.log shows the new `_preflight()` finally raising cleanly — confirming patches are NOW live for B34. Knowledge ledger UNCHANGED; B31 closure of the point-cell engine (Est #115–#126) and B32 preflight requirement (Est #127) both stand. NEW Est #128 (methodological): save-before-launch — when a batch agent edits the harness, the patches must be on disk before the next subprocess starts; two batches were lost to this. Cheapest defense = canary single-sim before 256-sweep launch (deferred). `specs/dicty_loop_base.yaml` and `sweep_plan.json` BOTH UNCHANGED; B34 = unchanged re-run of the B32/B33 MPM survey plan. Structural deliverable (MPM Est #82 break test) now two batches behind; scientific question unchanged.

## Batch 32 — MPM ENGINE FORK LAUNCH FAILED (infrastructure, NOT science)

> **B32 PRODUCED NO USABLE SWEEP DATA.** The 256 sims were launched but every single one failed at the `set_param` step inside `eval_sweeps.run_one` with `operator 'inflow_mpm' not in registry. Available: [...point-cell ops...]`. `sweep_results.json` is all-`NaN`; the `sweep_*.png` files contain empty morphology strips. There is NO scientific evidence to adjudicate any B32 hypothesis. **Root cause:** `dicty_loop.py:60` (`eval_sweeps()`) launched the subprocess WITHOUT setting `DICTY_ENGINE`, so `eval_sweeps.py:26` fell back to the point-cell engine `dicty_engine`. The MPM base spec schedules `inflow_mpm`/`mpm`, which are only registered when `dicty_engine_mpm` is loaded — every 256 sims aborted at sweep launch. Same family as **B21 Est #72** (silent `fld.diffusion` vs `fld.D` no-op) and **B26 Est #95** (op scheduled but absent from spec): an infrastructure-vs-spec mismatch the harness should have caught BEFORE running 256 sims. **B22/B23 pre-flight guidance had not been actioned in code.** **Fixes applied this batch:** (1) `dicty_loop.py` — added `_engine_from_spec()` sniffer that picks `dicty_engine_mpm` when the base spec contains `inflow_mpm` / `op: mpm` / `particle:`; explicit `DICTY_ENGINE` env still wins. (2) `eval_sweeps.py` — added `_preflight(base)` that loads the spec and intersects scheduled operator names against `_OPERATOR_REGISTRY` keys, aborting with a clear named-op + named-engine error BEFORE the sweep loop. **Knowledge ledger UNCHANGED** — B31's Est #115/#116 retractions + Est #117–#125 falsifications + Est #126 (point-cell engine structurally exhausted) all stand; no new mechanism tested. **NEW Est #127 (methodological):** any base-spec/engine mismatch must fail at preflight, not at mid-sweep — now enforced in code. The "I edited the spec but did not edit the launcher" pattern has now invalidated THREE batches (B21, B26 partial, B32 total) and is structurally fixed. **`specs/dicty_loop_base.yaml` UNCHANGED** (still the MPM base spec; the plan was correct, only the launcher was broken). **`sweep_plan.json` UNCHANGED** — the B32 plan was a well-designed first MPM survey (decisive n_frames break test, MPM-native parameter mapping, resolution probes, seed verification, cell.youngs × n_frames=1200 stress test); it re-runs cleanly under the patched launcher. **B33 = re-run the unchanged B32 plan.** The structural Est #82 break test at engine level — the project's primary structural deliverable — is deferred one batch; the scientific question is unchanged. If the harness fails again, preflight will name the missing op and engine.

## Batch 34 — see `_b34_log.md` (third consecutive infra failure; no science)

> B34 ALL-NaN like B32/B33 — same `inflow_mpm not in registry` error. Root cause: the long-running `dicty_loop.py` parent process holds a STALE in-memory `eval_sweeps()` from BEFORE the B32 patches; Python does not re-import a running module from disk when the file changes. The on-disk autodetect in `dicty_loop.py:_engine_from_spec()` (mtime 03:47:54) is never called by the in-memory launcher (B34 banner prints `running 256 sims ... on cuda:0 ...` with NO `engine=` token, proving the new code path is not executing). The on-disk `_preflight()` in `eval_sweeps.py` correctly aborted the 256-sim loop with a clear missing-op / loaded-engine diagnostic — saving the wasted-sim cost from B32/B33. Fix this batch: moved the engine autodetect from `dicty_loop.py:eval_sweeps()` to `eval_sweeps.py:_autodetect_engine()` (module top, before any operator import). Because `eval_sweeps.py` is launched as a fresh subprocess each batch, the on-disk version is always loaded — defeating the stale-launcher class of failure. Env override `DICTY_ENGINE` still wins. The redundant launcher-side autodetect is left in place but is no longer load-bearing. Knowledge ledger UNCHANGED; B31 closure of the point-cell engine (Est #115–#126), B32 preflight requirement (Est #127), and B33 save-before-launch lesson (Est #128) all stand. NEW Est #129 (methodological): engine autodetect lives in the spawned subprocess, not the launcher. `specs/dicty_loop_base.yaml` and `sweep_plan.json` BOTH UNCHANGED; B35 = unchanged re-run of the B32/B33/B34 MPM survey plan. Structural deliverable (MPM Est #82 break test) now THREE batches behind; scientific question unchanged. Flag for user: if the running `dicty_loop.py` is killed and restarted with `--resume`, both launcher and subprocess will load the patched code paths cleanly — maximum certainty option.

## Batch 35 — MPM ENGINE FORK FIRST USABLE DATA — DECISIVE STRUCTURAL NEGATIVE

> **B35 IS THE PROJECT'S PRIMARY STRUCTURAL ADJUDICATION** (after B32/B33/B34 burned to infra). 256 sims at the MPM base parent (`dicty_mpm_b32`: cell.n=767, youngs=60, particle.per_parent=8, secrete=8, sense.gain=30, camp.{D=0.02, decay=0.20}, inflow_mpm=1.5, mpm.{drag=4, vmax=6, a_max=1200, substeps=8, dt_sub=2e-4}, n_frames=240). EVERY sweep figure looks essentially the same: a frame-tiled SPARSE-SCATTER texture of ~50 small "mounds" (n_mounds 40-80, vs REAL=8), with inner_mass FLAT at ~0.20 across every panel (REAL=0.61). **The MPM engine at this parameterization does not aggregate; it produces noise-scatter at every parameter setting.**

### Batch 35 Sweep 0 — n_frames  [supported / DECISIVE]
Hypothesis: H1-B32 — finite-volume MPM cells sustain multi-mound morphology to n_frames>=1200 (Est #82 break test at engine level).
Response: loss MONOTONE-UP 6.43->22.77 (n_frames 180->1900); inner_mass FLAT ~0.19 across all 16 values (REAL=0.61); n_mounds GROWS 43->79.
Morphology (strip): all 16 panels show INDISTINGUISHABLE sparse-scatter texture; no aggregation visible at any n_frames. REAL (right) shows few bright compact mounds — nothing in the strip resembles it.
Verdict: **DECISIVE NEGATIVE.** MPM does not over-aggregate (no Est #82 collapse) but it **does not aggregate at all** — it stays in a non-mounding regime indefinitely, accumulating more sparse particles as n grows. The MPM finite-volume hypothesis is FALSIFIED in the opposite direction: under-aggregation, not over-aggregation.
Knowledge: NEW Est #130 — MPM at the chemotaxis-only operator stack {secrete, sense, inflow_mpm, mpm} is NON-AGGREGATING at all n_frames in [120, 2400]. The structural Est #82 break test resolves cleanly: finite-volume MPM cells alone are NEITHER necessary NOR sufficient for multi-mound morphology — the missing mechanism is explicit cell-cell cohesion (analogue of point-cell `spring.kadh`/`r_on`), not finite volume.

### Batch 35 Sweep 1 — sense.gain  [falsified]
Hypothesis: H2-B32 — sense.gain has a productive MPM-native band.
Response: noisy plateau loss 5-10; best at gain=50 (loss=4.97); inner_mass FLAT ~0.20 across [2, 300].
Morphology: identical sparse-scatter at every gain; no aggregation even at gain=300.
Verdict: falsified — chemotaxis-only attraction at any tested gain CANNOT drive MPM aggregation. Confirms Est #130 — no operator-side gain compensates for the missing cohesion.
Knowledge: NEW Est #131 — MPM sense.gain has NO productive band in [2, 300] under the bare {secrete, sense, mpm} stack; "best" gain=50 is metric ripple, NOT a morphology corner.

### Batch 35 Sweep 2 — secrete.rate  [falsified]
Hypothesis: H3-B32 — productive secrete band.
Response: loss noisy 4.7-11.8; best at rate=6 (loss=4.68); inner_mass FLAT ~0.20 across [0.5, 100].
Morphology: identical sparse-scatter at every rate.
Verdict: falsified — secrete amplitude is not the bottleneck. Piggybacks Est #130.

### Batch 35 Sweep 3 — camp.diffusion  [falsified]
Hypothesis: H4-B32 — productive D band (analogue of point-cell Est #99 / Est #118 corridor).
Response: loss noisy 5.8-10.5; best D=0.001 (loss=5.83); inner_mass FLAT ~0.20 across D in [0.001, 0.2] (3 orders of magnitude).
Morphology: identical sparse-scatter at every D.
Verdict: falsified — the cAMP gradient regime that gates Est #82 collapse in point-cell has NO analogue effect in MPM because MPM doesn't aggregate to begin with. NEW Est #132 — D is metric-inert in MPM in this stack.

### Batch 35 Sweep 4 — camp.decay  [falsified]
Hypothesis: H5-B32 — productive decay band; high-decay dispersion regime (Est #123) at moderate n_frames.
Response: loss 5.5-10.5; best decay=1.0 (loss=5.51); inner_mass FLAT ~0.20 across [0.01, 3.0].
Morphology: identical sparse-scatter; no Est #123-equivalent dispersion catastrophe (the MPM stack is ALREADY dispersed).
Verdict: falsified — decay parameter is metric-inert; the "high-decay dispersion" failure mode requires aggregation to disperse, and MPM has none. NEW Est #133.

### Batch 35 Sweep 5 — inflow_mpm.rate  [supported (weak)]
Hypothesis: H6-B32 — does inflow productively scale; ablation rate=0 test.
Response: loss V-shape; ablation rate=0 (loss=7.07, nm=44) gives sparse-scatter; best rate=0.5 (loss=4.18, nm=34); rate>=1.5 monotone-up; inner_mass declines weakly 0.224->0.186 with rate.
Morphology: rate=0 shows fewer scattered dots (matches initial seed only); rate>=4 shows more dots; none of the 16 panels show compact mounds.
Verdict: weak — productive band [0.5, 0.75] is real but the win is "less noise from fewer cells", not "more aggregation". Inflow rate is best LOW. NEW Est #134 — MPM does not "use" influx cells productively because there is no cohesion to recruit them into mounds.

### Batch 35 Sweep 6 — mpm.drag  [inconclusive]
Hypothesis: H7-B32 — dissipation regularizes collapse.
Response: loss noisy 5.8-10.5; best drag=10 (loss=5.83); plateau across the rest; inner_mass FLAT ~0.20.
Morphology: identical sparse-scatter at every drag.
Verdict: inconclusive (no aggregation to dissipate). NEW Est #135 — mpm.drag in [0.1, 30] is morphology-inert in the bare stack.

### Batch 35 Sweep 7 — cell.youngs  [partial; PILOT NEGATIVE on the central mechanism candidate]
Hypothesis: H8-B32 PRIMARY MECHANISM — Young's modulus = intrinsic stiffness is the Est #82 mitigator (finite volume cannot collapse below particle radius).
Response: loss noisy 4.2-10.5 across youngs in [1, 1000]; best youngs=20 (loss=4.23); youngs=2000 EXPLODES (loss=141); youngs=5000 EXPLODES (loss=377); inner_mass FLAT ~0.19-0.20.
Morphology: every panel sparse-scatter; high-youngs blow-up is purely numerical instability, not a structural transition.
Verdict: **DECISIVE NEGATIVE for the central mechanism candidate.** Young's modulus across 4 orders of magnitude does NOT produce visible structural difference in morphology — finite-volume stiffness is NOT a multi-mound mechanism in MPM. NEW Est #136. The project's central structural hypothesis (B31 strategic frame: "structural stiffness is the key Est #82 mitigator") is FALSIFIED in pilot.

### Batch 35 Sweep 8 — cell.n  [inconclusive + 2 NaN]
Hypothesis: H9-B32 — MPM cell.n productive band.
Response: loss noisy 6.4-15.8; best n=1400 (loss=6.45); n=1800/2000 are NaN (particle budget blew up); inner_mass crawls 0.20->0.22 with n.
Morphology: identical sparse-scatter at all completed values.
Verdict: inconclusive — productive band weakly favors n~1400 but morphology unchanged. n_final NaN at n>=1800 = particle/memory ceiling. NEW Est #137 — MPM cell.n upper limit ~1600 in current memory.

### Batch 35 Sweep 9 — particle.per_parent  [partial]
Hypothesis: H10-B32 — particle resolution gates finite-volume.
Response: loss declines per_parent 2->24 (8.5->4.45) then plateaus per_parent in [24, 48] at ~5.0-5.3; inner_mass FLAT ~0.20.
Morphology: identical sparse-scatter at every per_parent; the loss improvement at higher per_parent is metric ripple (denser per-cell point cloud -> smoother density field -> lower SSIM mismatch).
Verdict: partial — per_parent=24 is the new resolved working value, but it's a metric/regularization improvement, NOT a morphology corner. NEW Est #138.

### Batch 35 Sweep 10 — dt  [DECISIVE METHODOLOGICAL]
Hypothesis: H11-B32 — dt has aliasing resonances in MPM.
Response: loss IDENTICAL 10.5203 across ALL 16 dt values [0.10, 1.2]; inner_mass IDENTICAL 0.198; n_mounds IDENTICAL 53.
Morphology: ALL 16 panels pixel-identical.
Verdict: **dt is structurally INERT in MPM.** Root cause confirmed in code: `mpm.py:184-186` — MPM has its own `dt_sub=2e-4` and `substeps`; the engine outer `dt` is read for cAMP grid diffusion only, but MPM substeps x dt_sub fully determine cell physics. The outer `dt` is COMPLETELY DECOUPLED from the MPM cell-level dynamics. NEW Est #139 (methodological + structural): dt sweeps are USELESS in the current MPM engine; only `mpm.substeps` and `mpm.dt_sub` are the resolution knobs. **DROP all dt sweeps in future MPM batches.**

### Batch 35 Sweep 11 — mpm.substeps  [APPARENT WIN — SUSPECTED METRIC ARTEFACT]
Hypothesis: H12-B32 — substep undersampling causes numerical issues that BREAK finite-volume.
Response: loss DRAMATICALLY DROPS at high substeps: substeps<=16 loss in [4, 7]; substeps=20->1.80; substeps=24->1.58; substeps=32->**1.14** (PROJECT-BEST OF B35); substeps=48->1.16. n_mounds collapses 43->33->20->18->**11**->3. inner_mass climbs only slightly 0.20->0.18->0.21->0.25->0.18->0.52 at substeps=48.
Morphology (strip — CRITICAL): the panels become PROGRESSIVELY SPARSER as substeps grows from 8 to 48. The substeps=32 panel has FEWER total visible particles than substeps=8; substeps=48 is nearly empty with one bright dot. The "low loss" reflects PARTICLE DEPLETION (numerical instability, vanishing/escaping particles), NOT genuine aggregation. The inner_mass=0.52 at substeps=48 (close to REAL=0.61) is on a near-empty field — meaningless.
Verdict: **CONDITIONALLY supported as PROJECT-BEST LOSS, but morphology is INCONSISTENT WITH GENUINE AGGREGATION** — almost certainly a metric artefact (loss is small because remaining sparse particles happen to coincidentally match REAL density tile in SSIM more than the dense scatter at substeps=8 does). Requires VERIFICATION via particle count tracking and visual comparison to REAL at substeps=32. NEW Est #140 — provisional. **B36 will adjudicate.**

### Batch 35 Sweep 12 — mpm.a_max  [partial]
Hypothesis: H13-B32 — productive acceleration cap band.
Response: loss bowl with floor at a_max in [1500, 3500] (loss 4.0-4.7); a_max=1800 best (loss=3.97); a_max>=5000 monotone-up; inner_mass FLAT ~0.20.
Morphology: identical sparse-scatter at every a_max; the loss bowl is metric ripple, not morphological.
Verdict: partial — a_max=1800 is the new resolved working value but morphology unchanged. NEW Est #141 — productive a_max band [1500, 3500].

### Batch 35 Sweep 13 — mpm.vmax  [supported (weak)]
Hypothesis: H14-B32 — productive vmax band.
Response: loss U-shape with floor vmax in [1, 2] (loss 5.8); plateau loss~10 for vmax>=4; inner_mass FLAT ~0.20.
Morphology: identical sparse-scatter; loss decline at low vmax is metric ripple from slower drift.
Verdict: weak — productive band [1, 2] is the new resolved working value but morphology unchanged. NEW Est #142.

### Batch 35 Sweep 14 — seed  [HIGH SEED VARIANCE]
Hypothesis: H15-B32 — 16-seed verification of MPM base parent multi-mound morphology.
Response: loss [5.83, 10.94], median ~9.0, sigma~1.5 (WIDE compared to point-cell B31 sigma~0.022 noise floor); best seed=9 (loss=5.83); worst seeds=0,1,5,8; inner_mass tightly clustered 0.19-0.22.
Morphology: ALL 16 seeds show identical sparse-scatter pattern; no seed produces visible aggregation.
Verdict: falsified for "MPM is seed-robust multi-mound" — wide loss variance + no morphological change across seeds. NEW Est #143 — MPM base parent is seed-robust on morphology (always non-aggregating) but seed-NOISY on loss (3x point-cell sigma). Lesson: MPM stack has higher metric noise floor than point-cell.

### Batch 35 Sweep 15 — cell.youngs x n_frames=1200  [APPARENT WIN — SUSPECTED METRIC ARTEFACT]
Hypothesis: H16-B32 DECISIVE — high Young's (>=200) preserves multi-mound morphology to n_frames=1200 while low Young's (<=20) collapses (the structural Est #82 break test in MPM).
Response: loss DRAMATIC drop at youngs=5 (loss=**1.17** PROJECT-2ND-BEST, n_mounds=9 — closest to REAL=8 of all B35 sweeps); youngs in [10, 1000] loss climbs 9->27 with morph_score 5-9; youngs=2000/5000 EXPLODE (loss=476/816). inner_mass FLAT ~0.18-0.20.
Morphology (strip — CRITICAL): EVERY panel including youngs=5 shows SPARSE-SCATTER, NOT 8 distinct compact mounds. The youngs=5 panel is the SPARSEST (fewer particles visible) — same particle-depletion family as sw 11 substeps=32. The "low n_mounds=9" reflects particle depletion: the peak detector triggers on 9 random clumps in a near-empty field.
Verdict: **DECISIVE NEGATIVE for the H16-B32 hypothesis** — high Young's does NOT preserve morphology; in fact it diverges numerically. The "win" at youngs=5 is the SAME metric-artefact family as substeps=32: particle depletion masquerading as low loss. NEW Est #144 — Young's modulus structural-mitigation hypothesis is FALSIFIED at engine level in MPM.

**Batch 35 SUMMARY.** Parent UNCHANGED for B36 conservatively (MPM base spec is the only "real" parent — the apparent wins at substeps=32 (sw 11) and youngs=5 (sw 15) are particle-depletion artefacts that need verification before adoption). KEY STRUCTURAL NEGATIVE: **MPM finite-volume cells alone do NOT mitigate Est #82** — they replace it with under-aggregation (Est #130). The chemotaxis-only operator stack {secrete, sense, inflow_mpm, mpm} is NON-AGGREGATING regardless of all 14 tested parameters; only the SEED varies the loss. The project's central structural hypothesis (finite volume -> multi-mound) is FALSIFIED in pilot. **The missing mechanism is inter-cell cohesion**, which the bare MPM stack lacks (Young's gives INTRA-cell stiffness only; no inter-cell adhesion exists in MPM the way `spring.kadh` provided it in point-cell). Two paths forward for B36: (i) ENABLE existing `surface_tension` in `mpm.py:194` by marking particles `is_liquid=True` (a built-in cohesion mechanism currently DEAD because the spec doesn't set the liquid mask) and sweep `surface_tension` with 0 ablation; (ii) refute the metric-artefact hypothesis for sw 11/sw 15 via n_final reporting + visual zoom. PROJECT-BEST LOSS = 1.141 at sw 11 substeps=32 (provisional, metric-artefact-suspect). MPM ENGINE IS NOT VIABLE FOR MULTI-MOUND MORPHOLOGY IN ITS CURRENT FORM.


