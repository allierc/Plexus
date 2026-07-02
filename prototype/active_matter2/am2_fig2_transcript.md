

================================================================================
## FIG2 DESIGN 1 -- 2026-07-02 02:55:45
================================================================================
All four deliverables are written. Summary of Batch 1 (design step — I'm the Read/Edit/Write in-loop agent; the L4 cluster runs these slots next, verdicts land next batch):

## What I did

**Read the figure & mapped the states.** Paper Fig.2 panels: **a** active droplets (many small coherent blobs), **b** vortices (filled rainbow pinwheels + spiral c), **c** rings (open annuli), **d** streams (directed rivers), **e** silent polar bands (dark c), **f** polar bands with signalling. Panel **g**: at high ω, sweeping v0 left→right gives droplets→vortices→rings→polar-bands; mid ω = streams; low ω = no-pattern / silent bands.

**Distilled the causal map** (`knowledge_fig2.md`), including the *why* from Eqs 6-9: ω multiplies `ρ∇c` in the p-eqn → chemotactic self-attraction → condensation; v0 sets `-v0 div(p)` (density transport) and `-v0 (p·∇)s` → advective stirring that hollows/breaks condensates and, once it beats chemotaxis, drives Vicsek bands. Hypothesised map in our units (aggregation onset ω~1): at ω~1.8, v0 0.4→droplets, 0.6→vortices(established), 0.9→rings, 1.0→bands; drop ω→streams / silent bands.

**Designed 8 slots** (`am2_slots_fig2.md`), one physics var changed per slot from the `fig` anchor (v0 0.6, ω 1.8 = vortex):

| slot | v0 | ω | one-var from | expects |
|---|---|---|---|---|
| vortex | 0.6 | 1.8 | anchor | b |
| droplets | **0.4** | 1.8 | vortex (v0↓) | a |
| rings | **0.9** | 1.8 | vortex (v0↑) | c |
| bands_sig | **1.0** | 1.8 | rings (v0↑) | f |
| streams | 0.6 | **0.5** | vortex (ω↓) | d |
| silent_bands | 1.0 | **0.4** | bands_sig (ω↓) | e |
| probe_vr | **0.78** | 1.8 | vortex — vortex↔ring boundary | — |
| probe_w | 0.6 | **1.1** | vortex — aggregation-onset boundary | — |

All six states + two boundary probes; no `am2_hydro.py` edit needed (states reachable from the preset). `user_input.md` was empty.

**Biggest surprise to watch / the lever:** whether vortex→ring→band is a *pure v0 axis* at fixed high ω (probe_vr localises it). **Flagged risk:** the v0=1.0 band slots may blow up at dt=0.02 under strong chemotaxis — if a panel shows only initial noise (= FAIL), next batch drops v0 toward 0.9 or adds `--sigma 0.9` (alignment reaches bands at lower motility, cf. `PRESETS['bands']`).


================================================================================
## FIG2 DESIGN 2 -- 2026-07-02 03:09:58
================================================================================
All three files are updated and the slot design is written. Here's the summary of Batch 2.

## What the batch-1 montage told me

**The (v0,ω) plane is BINARY at the fig coefficients — not the 6-state sequence we hypothesized.**

| slot | v0 | ω | P | Nc | ctr | read | verdict vs paper |
|---|---|---|---|---|---|---|---|
| s0 vortex | 0.6 | 1.8 | 0.0 | 32 | 2.29 | vortex/aster lattice, spiral c | **correct state** (panel b), but too many/small |
| s1 droplets | 0.4 | 1.8 | 0.06 | 13 | 0.73 | diffuse, under-condensed | MISPLACED |
| s2 rings | 0.9 | 1.8 | 0.0 | 42 | 1.84 | *denser* vortex lattice | MISSING |
| s3 bands_sig | 1.0 | 1.8 | 0.0 | 45 | 1.66 | densest vortex lattice | MISSING |
| s4 streams | 0.6 | 0.5 | 0.94 | 5 | 0.27 | homogeneous polar sheet | MISPLACED |
| s5 silent_bands | 1.0 | 0.4 | 0.90 | 4 | 0.36 | polar sheet, c moderate | PARTIAL (right phase, no stripes) |
| s6 probe_vr | 0.78 | 1.8 | 0.0 | 38 | 2.24 | vortex lattice | boundary confirm |
| s7 probe_w | 0.6 | 1.1 | 0.95 | 4 | 0.31 | homogeneous polar | boundary confirm |

**Two phases only:** condensed **vortex/aster lattice** (ω ≳ 1.5, P~0) vs **homogeneous polar** (ω ≲ 1.1, P~0.9), split by a sharp ω threshold ~1.5 that's nearly v0-independent.

**Biggest surprises + the lever:**
1. Raising v0 at high ω only makes *more, smaller* vortices (Nc 32→38→42→45), never rings or bands — the vortex→ring→band "v0 axis" is **falsified**. v0 sets blob count, not topology.
2. Lowering v0 (0.4) *reduced* condensation — v0 **aids** aggregation here (via `−v0 div p`), the opposite sign from the paper's droplets.
3. ω=1.1 is already fully polar → aggregation onset is ~1.5, higher than the ledger's "~1".

**Why:** chemotaxis `ρω∇c` onto the excitable/spiral c-field mints +1 defects as soon as it beats polar growth `σ(ρ−1)p`. It flips the whole field at once → no graded window for the intermediates.

## Batch-2 design (8 slots)
The intermediates aren't reachable on (v0,ω) alone, so I reshape the **c-texture** and **polar-onset margin** (one var per named parent):
- **w_16 / w_14** — fine ω sweep across the sharp boundary → hunt streams/ring intermediates.
- **drop_chi (χ 0.1) / ring_alpha (α 0.85)** — kill the (p·∇)p curl / sharpen c wells → coherent single-hue compact **droplets**.
- **ring_Dc (Dc 2.2)** — broad annular c waves → density on moving fronts → **rings**.
- **band_sigma (σ 0.2) / band_chi (χ 0.9)** off silent_bands — sit near ρ_c / channelise → **discrete polar bands** vs the homogeneous sheet.
- **vortex_Dp (Dp 1.2)** — merge into fewer/larger vortices for a tighter paper-b match.

No edit to `am2_hydro.py` was needed — all levers are already CLI-plumbed. Files updated: `analysis_fig2.md` (Batch 2 section), `knowledge_fig2.md` (binary-map + causal levers), `am2_slots_fig2.md` (8 slots).


================================================================================
## FIG2 DESIGN 3 -- 2026-07-02 03:18:25
================================================================================
Done. Here's the Batch 3 summary.

## Reading the batch-2 montage (`fig2_b02_montage.png`)

**Per-slot classification (state → region → verdict):**

| slot | v0 | ω | P | Nc | ctr | sig | state | verdict |
|---|---|---|---|---|---|---|---|---|
| s0 w_16 | 0.6 | 1.6 | 0.0 | 20 | 2.5 | .60 | vortex lattice | correct; onset floor |
| s1 w_14 | 0.6 | 1.4 | .18 | 3 | .76 | .61 | polar + 1 relic | **pins onset ~1.5** |
| s2 drop_chi | 0.6 | 1.8 | .02 | 31 | 2.14 | .66 | **comma-streaks** | closest to droplets |
| s3 ring_alpha | 0.6 | 1.8 | 0.0 | 27 | 2.26 | .48 | filled vortices | no hollowing |
| s4 ring_Dc | 0.6 | 1.8 | 0.0 | 17 | 2.69 | .57 | few large vortices | best paper-b, still filled |
| s5 vortex_Dp | 0.6 | 1.8 | .15 | 8 | .74 | .59 | de-condensed | Dp smears, wrong way |
| s6 band_sigma | 1.0 | 0.4 | .99 | 2 | .05 | .63 | homogeneous polar | **bands FALSIFIED** |
| s7 band_chi | 1.0 | 0.4 | .95 | 3 | .30 | .61 | homogeneous polar | no bands |

**Biggest surprise + the lever:** dropping σ toward the paper's 0.02 made the polar sheet *more* uniform (contrast 0.05), not banded — the opposite of the batch-2 hypothesis. The reason clicked: a uniform flock has `div(p)≈0`, so the `-v0·div(p)` density-transport term never structures density. And **rho0 was hardcoded at 1.2 — well above the flocking onset rho_c=1**, so the whole field is a stable homogeneous flock with no dilute gas to coexist with a dense band. **Travelling bands are a density effect (Toner-Tu window near rho_c), not a (v0,ω,σ) effect.** Separately, the c-texture knobs (Dc/α/Dp/χ) only *resize* the vortex lattice — never change its topology — with χ→0.1 the one exception that nudges pinwheels into coherent comma-streaks (proto-droplets).

## Actions taken
- **Code:** added a `--rho0` knob to `am2_hydro.py` (default 1.2 preserved) + whitelisted it in `am2_job.py` — the minimal edit to reach the banding window.
- **analysis_fig2.md:** dated Batch 3 section with per-slot reads.
- **knowledge_fig2.md:** sharpened onset (abrupt at ~1.5, Nc 3→20 over Δω=0.2), the Dc/α/Dp/χ retuning rules, the σ-falsification, and the rho0→bands hypothesis.
- **am2_slots_fig2.md** — 8 slots, one var/parent:
  - **bands (4):** `band_rho` (rho0 1.05), `band_rho_lo` (1.0 — window vs no-pattern floor), `sig_band` (ω→1.0, signalling bands), `band_v0` (v0→0.6, motility floor)
  - **droplets (2):** `drop_chi0` (χ→0), `drop_chi_w` (ω→1.6, fewer coherent blobs)
  - **rings (1):** `ring_Dc_v0` (v0 0.95 on broad-Dc waves)
  - **boundary (1):** `onset_15` (pin the 1.5 threshold)

The band slots are the high-value bet: if rho0~1.05 produces travelling stripes, it places two missing states (silent + signalling bands) and reveals the phase diagram's hidden third axis.


================================================================================
## FIG2 DESIGN 4 -- 2026-07-02 03:26:13
================================================================================
Done. All four deliverables are updated. Here's the summary of Batch 4.

## Reading the batch-3 montage

**The biggest surprise — active droplets were hiding in plain sight.** They are *not* a separate regime: they are the condensed (vortex) phase at **low nucleation density**. The blob count is a continuum set by how close you sit to the ω≈1.5 onset and by ρ₀:

| slot | v0 | ω | ρ₀ | Nc | contrast | state |
|---|---|---|---|---|---|---|
| sig_band | 1.0 | 1.0 | 1.05 | 3 | **4.30** | **droplets** (3 huge isolated blobs on empty bg) |
| onset_15 | 0.6 | 1.5 | 1.2 | 16 | 2.56 | droplets/onset (sparse) |
| (base) | 0.6 | 1.8 | 1.2 | 32 | 2.29 | vortex lattice (space-filling) |

Lowering ω toward onset *or* lowering ρ₀ → fewer nucleation sites → isolated compact droplets. Raise either → the same blobs multiply and tile space into the vortex lattice. So **droplets↔vortices is one phase graded by count, not a phase boundary.**

**Per-slot verdicts:**
- **band_rho / band_v0** (ρ₀ 1.05, low ω): homogeneous polar, no stripes — MISSING.
- **band_rho_lo** (ρ₀ 1.0 = ρ_c): flock homogenises *and* c runs away to a uniform ceiling (contrast→0). Bands can't be judged until c is capped.
- **drop_chi0** (χ=0): first **open annular c loops** appeared — the ring precursor — but density blobs stayed filled. χ→0 is necessary-not-sufficient for rings.
- **ring_Dc_v0** (v0 0.95, Dc 2.2): cleanest regular vortex lattice yet (28 filled pinwheels) — best paper-b topology, still filled.

## Design (8 slots), one var from each named parent
- **Rings (3, best shot at a new topology):** `ring_chi_v0` (drop_chi0 + v0→1.0), `ring_hiDc` (drop_chi0 + Dc→3.0), `ring_eps` (ring_Dc_v0 + eps→0.02) — off-curl **plus** wave-expansion to pile density on an annular front.
- **Droplets refined (2):** `drop_1hue` (onset_15 + χ→0.1, single-hue near-onset = paper-a), `drop_lowrho` (sig_band + ρ₀→1.02, sparser still).
- **Bands retried with c capped (2):** `band_beta` (β→0.2) and `band_lowsig` (σ→0.3) at ρ₀=1.0 — tame the runaway so the Toner-Tu window can open.
- **Boundary probe (1):** `onset_probe` (onset_15 + ω→1.7) — pin the droplet(sparse)↔vortex(space-filling) count crossover.

Still fully missing after 3 batches: **rings, streams, and bands** — this batch attacks rings hardest (best odds of a genuinely new topology) while re-opening bands with the c-runaway diagnosed and fixed.


================================================================================
## FIG2 DESIGN 5 -- 2026-07-02 03:34:12
================================================================================
All three memory files are updated and the slots will parse correctly (`#` comments skipped, `--flag val` tokens captured; `Drho`/`rho0`/`omega` all flow through `run_hydro`). Here's the batch-5 readout.

## Batch 4 read — what each slot became

| slot | (v0, ω) + knob | P / Nc / ctr / sig | state | verdict |
|---|---|---|---|---|
| drop_1hue | 0.6, 1.5, chi 0.1 | .49/9/.63/.60 | near-onset mixed | chi↓ favours the **polar** side, not droplets |
| onset_probe | 0.6, 1.7 | .00/26/2.35/.64 | **vortices** | pins count continuum ω1.5→16, 1.7→26, 1.8→32 |
| drop_lowrho | 1.0, 1.0, rho0 1.02 | .20/3/2.64/.54 | **droplets (few-large)** | 3 isolated blobs, each ringed by expanding c arcs |
| ring_chi_v0 | 1.0, 1.8, chi 0 | .01/**65**/1.23/**1.05** | **droplets (many-small)** | = paper-a! chi=0+high v0 → many single-hue blobs |
| ring_hiDc | 0.6, 1.8, chi 0, Dc 3 | .26/12/1.25/.60 | **ring precursor** | c field paints arcs/crescents; fronts collide before closing |
| ring_eps | 0.95, 1.8, Dc 2.2, eps .02 | .00/37/2.39/.94 | vortices | eps FALSIFIED as ring lever |
| band_beta | 1.0, 0.4, rho0 1.0, beta .2 | .96/1/**0**/.32 | homogeneous polar | band FALSIFIED |
| band_lowsig | 1.0, 0.4, rho0 1.0, sig .3 | .91/3/**0**/.59 | homogeneous polar | band FALSIFIED |

## The three findings that moved the map
1. **DROPLETS are done** — both paper morphologies reproduced from the *same* condensed phase: few-large via low rho0/near-onset (`drop_lowrho`), many-small via chi=0 + high v0 (`ring_chi_v0`, 65 blobs). Count = onset-proximity + rho0; chi→0 = single-hue; v0↑ = fragments.
2. **Rings are a chemical-wave signature, not a density-topology flip.** chi=0 (off-curl) + high Dc makes the *c field* form open arcs — but at ω=1.8 the dense fronts collide before closing. The recipe needs **sparse** fronts. eps is not the lever.
3. **Biggest surprise / the lever:** chi=0 doesn't hollow one aster — it *multiplies* droplets. And bands remain the only fully-missing state after 3 tries: beta↓ and sigma↓ both leave a uniform ordered sheet (div p≈0 → no density transport). The **untried lever is Drho** — density diffusion (0.5) is smoothing away the Toner-Tu banding instability.

## Batch 5 design (8 slots)
- **Bands ×3 (the real bet):** ω=0 (chemotaxis off → pure Toner-Tu) × **Drho↓ 0.15** near rho_c, then v0↑ and a rho0-window probe. Bands would show as bright stripes in the top (rho) panel even with c saturated.
- **Rings ×2:** chi=0 + Dc 3 at **lower ω (1.4) / lower rho0 (1.08)** so isolated fronts close into full annuli.
- **Streams ×1:** high chi (1.2) to channelise the turbulent onset layer.
- **Boundary probe ×1:** many-small droplet count vs onset at chi=0.


================================================================================
## FIG2 DESIGN 6 -- 2026-07-02 03:46:00
================================================================================
All three files updated. Batch 6 complete. Here's the summary.

## Batch 6 read — Fig.2 hydrodynamic phase diagram

**Per-slot classification (batch-5 montage):**

| slot | (v0, ω) + key coeff | P / Nc / ctr | state | verdict |
|---|---|---|---|---|
| s0 band_silent | 1.0, 0.0, rho0 1.0 | .99/1/0.0 | homogeneous ordered | bands MISSING |
| s1 band_loDrho | 1.0, 0.0, Drho .15 | .98/1/0.0 | homogeneous ordered | **Drho bet FAILED** |
| s2 band_hiv0 | 1.6, 0.0, Drho .15 | 1.0/4/0.0 | homogeneous ordered | MISSING (v0 up no help) |
| s3 band_win | 1.0, 0.0, rho0 1.05 | .94/3/**0.4** | proto-band (diag streaks) | **first density signal** |
| s4 ring_sparse | 0.6, 1.4, χ0 Dc3 | .32/7/0.5 | c-crescents/arcs | ring precursor |
| s5 ring_lowrho | 0.6, 1.8, χ0 Dc3 rho0 1.08 | .14/10/1.5 | sparse filled blobs | droplets, not rings |
| s6 stream_chi | 0.6, 1.6, χ1.2 | .001/10/2.4 | discrete asters | **high χ ≠ streams** |
| s7 drop_probe | 1.0, 1.5, χ0 | .06/35/1.6 | many-small droplets | droplets DONE |

**Biggest surprise + the lever.** The whole Toner-Tu band bet collapsed: ω=0 (chemotaxis off) + rho0=rho_c + Drho dropped 0.5→0.15 *still* gives a flat homogeneous ordered sheet (P>0.98, contrast=0). The ordered state is **linearly stable** here — low Drho alone can't break it. The only density modulation appeared at **rho0=1.05, slightly *above* onset** (s3, faint diagonal streaks): at rho0=1.00 the order amplitude p₀→0 (nothing to transport); at 1.05, p₀≈0.24 turns on the near-onset banding instability (growth ~ v0·σ·p₀, damping ~ (Dp+Drho)q²). We lowered Drho but **never Dp** — the untried splay-damping lever.

**Distilled to knowledge_fig2.md:** bands = rho0≈1.05 (above onset) + **low Dp** + ω=0; Drho/beta/sigma all falsified. Streams ≠ high χ (makes asters) → need onset layer + advective stretch. Droplet count continuum nailed (ω 1.5→35, 1.8→65 blobs at χ=0). Rings = chi0+Dc3 crescents, need thinner nucleation to close.

**Batch-6 slots (8, one var each):** 4 band slots off band_win — the new **Dp↓0.2** lever (band_loDp → +v0), plus rho0 1.10 and chi 0.9; 2 ring slots thinning nucleation (rho0 1.10 / ω 1.2, the latter a boundary probe); 2 stream slots retrying via advective stretch (v0 1.3 @ ω1.45) vs channelization at the onset. No code edit yet — if the Dp lever also fails, batch 7 edits the integrator.


================================================================================
## FIG2 DESIGN 7 -- 2026-07-02 03:55:29
================================================================================
All four deliverables are complete. Here's the batch-7 summary.

## Batch 7 — Fig.2 hydrodynamic phase diagram

**The headline: BANDS are finally on the map.** The batch-6 Dp-down bet paid off, and I found *two* independent routes — both sitting slightly above the flocking onset (rho0 > 1.0) with weak damping on the modulation mode.

### Per-slot reads (batch 6)
| slot | (v0, ω) + key coeff | P / Nc / ctr | state | verdict |
|---|---|---|---|---|
| s0 band_loDp | 1.0, 0 · rho0 1.05, Dp 0.2 | .878/2/.47 | **bands (network)** | 1st real band structure — criss-cross density ridges |
| s1 band_loDp_v0 | 1.6, 0 · Dp 0.2 | .83/2/.76 | **bands (network)** | best contrast; stream-adjacent rivers |
| s2 band_rho11 | 1.0, 0 · rho0 1.10 | .997/2/.40 | **silent polar bands** | clean parallel stripes = best paper panel-e match |
| s3 band_chi | 1.0, 0 · chi 0.9 | .972/3/.32 | bands (wavy) | chi waves stripes, not a contrast lever |
| s4 ring_thin | 0.6, 1.4 · chi0, Dc3, rho0 1.10 | .683/4/.43 | **ring precursor** | best open arcs, nearly closing |
| s5 ring_om12 | 0.6, 1.2 · chi0, Dc3 | .604/4/.43 | under-condensed | w=1.2 too far below onset |
| s6 stream_v0 | 1.3, 1.45 | .019/27/2.66 | condensed droplets | **not streams** |
| s7 stream_chi2 | 0.6, 1.45 · chi1.0 | .003/6/2.54 | condensed droplets | high chi FALSIFIED again |

### Biggest surprises + levers
1. **Bands buckle when growth `v0·σ·p0` beats damping `(Dp+Drho)·q²`.** Three knobs cross it: rho0 up (raises p0), Dp/Drho down (releases the splay mode), v0 up. rho0=1.00 fails (p0→0); rho0≥1.05 works. ω=0 ⇒ *silent* bands.
2. **Streams can't be "stretched from the onset layer" — that corridor doesn't exist.** w=1.45 is already fully condensed (P≈0) even at v0=0.6; the polar↔condensed jump is sharp. Streams must live in the low-ω band regime with chemotaxis *weakly on*.

### Files updated
- **analysis_fig2.md** — dated Batch 7 section (per-slot metrics + verdicts).
- **knowledge_fig2.md** — new "Batch-7: BANDS SOLVED" block; rewrote the BANDS (now solved, 2 routes) and STREAMS (onset route dead, new bet) causal-map lines.
- **am2_slots_fig2.md** — 8 new slots: **signalling bands** (sigband_w05/w08/v0 — turn ω on in route B), **streams** (stream_w06/v0hi — weak ω on the low-Dp network → rivers), **rings** (ring_rho115/Dc4 — close the annuli), **band sharpen** (band_bothlev — both levers together). One variable per slot from a named parent; sigband_w08 doubles as the band→condensation boundary probe.


================================================================================
## FIG2 DESIGN 8 -- 2026-07-02 04:03:10
================================================================================
Done. All three files updated; no integrator edit needed (every state is now reachable).

## Batch 8 summary

**Batch-7 read — the three bets all paid off; 5 of 6 states are now placed:**

| slot | (v0, ω) | P / ctr | state | verdict |
|---|---|---|---|---|
| sigband_w05 | 1.0, 0.5 | .894/.50 | **signalling bands** | first ever; c co-travels |
| sigband_w08 | 1.0, 0.8 | .939/.49 | sig-bands condensing | boundary: onset >0.8 |
| **sigband_v0** | 1.5, 0.5 | .983/.67 | **signalling bands (clean = paper f)** | high v0 sharpens |
| stream_w06 | 1.0, 0.6 | .878/.54 | stream network | criss-cross |
| **stream_v0hi** | 1.4, 0.6 | .849/.70 | **streams (best)** | directed rivers, ctr highest |
| ring_rho115 | 0.6, 1.4 | .706/.43 | ring precursor | under-condensed |
| **ring_Dc4** | 0.6, 1.4 | .093/.54 | **rings (clearest c annulus)** | Dc closed it |
| band_bothlev | 1.0, 0.0 | .866/.43 | band network | low Dp dominates |

**Biggest surprise + lever:** ring_Dc4. Bumping **Dc 3→4** flipped P from 0.71 to 0.093 — partial-order arcs became a fully condensed aster with a *closed* c annulus. High Dc doesn't just texture c; broad/fast fronts nucleate condensation **and** wrap the density into a ring. Dc, not rho0, is the ring lever (rho0 1.15 at Dc=3 just under-condensed).

**Distilled to knowledge:** signalling bands = silent Route-B recipe (rho0≥1.10) + ω weakly ON (~0.5), v0 sharpens; streams = low-Dp network + weak ω + high v0; rings = chi=0 + Dc≥4 at ω~1.4; the two band routes are distinct (low Dp → network/streams, default Dp → parallel stripes), not additive.

**Batch-8 slots (8):** 3 ring-sharpen (Dc↑, v0↑ to hollow core, thin nucleation), 3 stream (ω↑ to organise one-way rivers, v0↑ to stretch, ω↓ boundary), 2 signalling-band (v0→2.0 for sharpest, ω→1.0 boundary probe for the band→condensed edge at high v0).


================================================================================
## FIG2 DESIGN 9 -- 2026-07-02 04:09:30
================================================================================
Done. All four tasks complete. Here's the Batch 9 read.

## Per-slot classification (batch 8 → paper state → (v0,ω) region)

| slot | (v0, ω) | P / Nc / ctr / sig | state | verdict |
|---|---|---|---|---|
| s0 ring_Dc5 | 0.6, 1.4 (Dc5) | .98/1/0.26/1.50 | **homogeneous polar** | RING OVERSHOT — Dc 4→5 over-diffuses c, gradient collapses |
| s1 ring_hollow | 0.9, 1.4 | .20/6/2.39/0.52 | **droplets** | v0 up condensed/fragmented, didn't hollow |
| s2 ring_thin12 | 0.6, 1.4 (ρ0 1.20) | .41/2/0.89/0.58 | **one big filled aster** | rho0 up → fewer FILLED asters, not rings |
| s3 stream_om09 | 1.4, 0.9 | .13/11/1.66/0.48 | **condensing foam net** | STREAM OVERSHOT — ω 0.9 condenses the rivers |
| s4 stream_v018 | 1.8, 0.6 | .98/3/0.92/0.46 | **streams (clean lanes)** | ✅ best directed-flow of the loop |
| s5 stream_om03 | 1.4, 0.3 | .94/5/0.64/0.51 | **criss-cross river net** | ω 0.3 too weak to channel one-way |
| s6 sigband_v20 | 2.0, 0.5 | .99/3/0.93/0.46 | **signalling bands** | ✅ cleanest paper-f match |
| s7 sigband_w10 | 1.5, 1.0 | .12/11/2.18/0.45 | **condensed lattice** | sig-bands condense between ω 0.5 and 1.0 |

## Biggest surprise + lever
**The RING is the one fragile, non-robust state.** ring_Dc4 is a knife-edge optimum — *every* single-variable push off it degraded it: Dc↑(5.0) over-diffuses c into a flat field → homogeneous polar (P 0.98); v0↑(0.9) → filled droplets; rho0↑(1.20) → one big filled aster. Dc is **non-monotonic** (3=arcs, 4=annulus, 5=nothing). Rings live in a tiny box and still only appear as one aster's annulus, not a field.

Counterpoint (positive): **high v0 (1.8–2.0) is the master flow-sharpener** — stream_v018 and sigband_v20 gave the cleanest directed-lane/stripe morphologies of the entire loop, no blow-up at v0=2.0. And two boundaries pinned: streams condense above ω~0.9; sig-bands above ω~0.8.

## Batch 9 slots (8, one var each from named parent)
- **Rings ×4** — attack the window from *inside* for a FIELD: `ring_Dc42` (top-bracket the window), `ring_v07` (gentle hollow), `ring_w15` + `ring_rho105` (more sites without leaving the box).
- **Streams ×2** — `stream_v016` (bracket lane-straightening 1.4→1.6→1.8), `stream_w075` (condensation boundary).
- **Sigbands ×2** — `sigband_v25` (push v0 / test blow-up), `sigband_w08` (pin condensation onset at v0 2.0).

Files updated: `analysis_fig2.md` (dated Batch 9 section), `knowledge_fig2.md` (ring-window optimum + v0-sharpener + two boundaries distilled into the causal map), `am2_slots_fig2.md` (8 new slots).


================================================================================
## FIG2 DESIGN 10 -- 2026-07-02 04:20:55
================================================================================
All four files are updated. Here is the Batch 10 summary.

## Batch 9 read — per-slot classification (v0, ω) → state

| slot | (v0, ω) + key coeff | P / Nc / ctr / sig | paper state | verdict |
|---|---|---|---|---|
| s0 ring_Dc42 | 0.6, 1.4, Dc 4.2 | .41/4/.47/.59 | condensed→polar | Dc **past** optimum — ring releasing |
| s1 ring_v07 | 0.7, 1.4, Dc 4.0 | .45/5/1.16/.57 | vortex (few) | 1–2 hollow asters, not a field |
| s2 ring_w15 | 0.6, **1.5**, Dc 4.0 | .93/5/.46/.59 | **homogeneous polar** | 🔴 SURPRISE — collapsed |
| s3 ring_rho105 | 0.6, 1.4, ρ₀ 1.05 | .25/9/.51/.58 | **rings (arcs)** | best ring-field candidate |
| s4 stream_v016 | 1.6, 0.6 | .96/5/.76/.48 | **streams** | clean directed lanes |
| s5 stream_w075 | 1.8, **0.75** | .17/9/1.52/.44 | condensed | stream boundary |
| s6 sigband_v25 | **2.5**, 0.5 | .97/2/.93/.50 | **signalling bands** | stable, sharpening saturated |
| s7 sigband_w08 | 2.0, **0.8** | .56/14/1.26/.51 | condensed foam | sigband boundary |

**Biggest surprise:** `ring_w15` — pushing ω 1.4→1.5 *toward* more condensation instead **collapsed the ring regime to a homogeneous polar sheet** (P 0.93). At Dc=4 the c field already sits at the over-diffusion edge; extra ω saturates c and kills grad(c), so chemotaxis loses its grip. **The lever:** inside the ring box ω is *not* a site-multiplier — **rho0-down is** (ρ₀ 1.05 gave Nc 2→9 arced sites, the best ring-field yet).

**Two other results:** v0-sharpening **saturates** at ~2.0 (sigband ctr 0.93 flat from v0 2.0→2.5, stable to 2.5); and two condensation boundaries tightened — stream→condense (v0 1.8) between ω 0.6/0.75, sigband→condense (v0 2.0) between ω 0.5/0.8.

## State of the phase diagram
5 of 6 paper states solidly placed (droplets, vortices, silent bands, streams, signalling bands). **Rings are the one still-approximate state** — reachable only as a field of open arcs, not closed hollow annuli.

## Files updated
- **analysis_fig2.md** — dated Batch 10 section (per-slot v0/ω/P/Nc/ctr/sig + verdicts + surprise/lever).
- **knowledge_fig2.md** — new "Batch-10 findings [FINAL]" block + tightened RINGS line (box {chi=0, Dc~4.0 edge <0.2, ω~1.4 exactly, ρ₀ 1.05–1.10}; ρ₀-down = field multiplier).
- **am2_slots_fig2.md** — 8 final slots: 4 attack the ring field (Dc 3.6/4.4, faster c-decay α 0.7, gentle v0 0.65 off `ring_rho105`) to close the 9 arcs into annuli; 4 fine boundary probes (ring Dc 4.1, ring w 1.45, stream w 0.68, sigband w 0.65). Each changes one variable from a named parent.
