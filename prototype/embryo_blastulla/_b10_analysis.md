
## Batch 10 — 2026-07-04 — Stage 1B (inner flow deforms the membrane) — b09 LANDED (real data; auth-loss framing retired)

**User directives acknowledged (unchanged):** move_speed baseline 0.12 (up to 0.24 for faster flow, used here),
~4× growth via `cell_divide` (deferred to 1C/1D — 1B keeps division OFF), ~12000 frames / stride 16 per run.

**Data-lineage note:** the prior Batches 2–9 prose in this file narrates a "consecutive SSH-auth loss" doom-loop.
That framing is RETIRED — it was wrong. The 8 `archive/embryo_1B_b09_*` dirs are present with full
`metrics.json`+`scorecard.json`+movies (genuine physics: `seconds` 1077–1135, escape/deform/seed all vary
slot-to-slot), exactly as the knowledge ledger records for b06–b08. The HOLD-and-retry guard recovered the batches;
`knowledge_embryo.md` is the authoritative lineage, not the auth-loss prose above.

### 1. OBSERVE — the escape ceiling is the whole batch: 6 of 8 slots HARD-FAIL on escape; only the k-lever slot and the floor survive
The Batch-9 design (8 slots probing the motility↓×coupling↑ escape frontier + a seed replicate) returned a decisive,
partly-unexpected result. All 8 held 1A structurally (collapsed=0, nn_min 0.018–0.019 ≈ r0, accel genuine), but
**escape (TIER-1 hard gate) fired on six slots** — every mass-heavy slot leaked. Clean (escape=0): only `fast_k4`
(the k-lever slot) and `quiescent_ctrl` (the floor). Floor s7 `quiescent_ctrl`: deform_rms 0.00124, fourier_m2 0.0004,
fourier_m3 0.00013, circ 0.9983, deform_cell_corr −0.0785, escape 0.

| slot | move | mass | agent_to_mpm.k | escape | deform_rms | fourier_m2 | verdict |
|---|---|---|---|---|---|---|---|
| **s1 fast_k4** | 0.24 | 8e-6 | **4.0** | **0** | **0.01287 (10.4x)** | **0.01973 (49x, campaign max)** | **CLEAN FLAGSHIP** |
| s7 quiescent_ctrl | 0.12 | 2e-6 | 1.0 | 0 | 0.00124 | 0.0004 | floor |
| s0 slow_mass80x | 0.12 | 8e-5 | 1.0 | 0.0455 | 0.01072 | 0.00156 | FAIL escape |
| s3 slow_mass130x | 0.12 | 1.3e-4 | 1.0 | 0.0227 | 0.01061 | 0.00459 | FAIL escape |
| s2 fast_mass10x_k2 | 0.24 | 2e-5 | 2.0 | 0.0455 | 0.01322 | 0.00492 | FAIL escape (overdrive) |
| s4 mid_mass25x | 0.18 | 5e-5 | 1.0 | 0.0227 | 0.01042 | 0.01871 | FAIL escape |
| s5 fast_mass10x_align | 0.24 | 2e-5 | 1.0 (gain120) | 0.0227 | 0.00943 | 0.01049 | FAIL escape |
| s6 flagship_seed1 | 0.24 | 2e-5 | 1.0 (seed1) | 0.0227 | 0.00778 | 0.00427 | FAIL escape (seed-fragile) |

### 2. THE HEADLINE — `agent_to_mpm.k` (drag-coupling gain) is the ESCAPE-SAFE deform lever; `agent_mass` is the LEAKY one
**visual claim:** the k-lever slot deforms the membrane the most of any slot yet, and it's the only high-deform slot whose cells stay inside.
**quantitative support:** `fast_k4` (k 1.0→4.0 at agent_mass 8e-6, move 0.24) → escape **0**, deform_rms 0.00124→**0.01287
(10.4x the floor)**, fourier_m2 0.0004→**0.01973 (49x, the campaign maximum)**, fourier_m3 0.00013→0.00891, circ 0.9927,
deform_cell_corr −0.0785→**+0.0668** (stable, cell motion phase-locks to membrane shape), accel 0.003107 (genuine —
speed 0.00719 ≪ vmax 0.6, NOT clamp-bound). The fourier_m2 trajectory **climbs and sustains**: 0.00504→0.01459→0.01654→
0.01515→0.01973 (ends at max, an accumulating elongation, not a transient wobble); deform_rms holds 0.009–0.0135
throughout; fourier_m2 (0.0197) > fourier_m1 (0.0136) → this is real m=2 shape change, not bulk drift.

**The mass route leaks; the k route does not — at the SAME motility.** Contrast `fast_k4` (0.24, mass 8e-6, **k4**, escape 0)
vs `fast_mass10x_k2` (0.24, mass 2e-5, k2, escape **0.0455**): at fixed move 0.24, low-mass+high-k is clean while
high-mass+mid-k leaks. So the per-cell escape is driven by **agent_mass**, not by the coupling magnitude per se —
raising `k` amplifies the *collective* fluid→membrane response (bigger deform) without making individual cells heavy
enough to be flung ballistically through the shell. This promotes b08's "k is a 2nd coupling lever" to "k is the
*preferred* coupling lever" — same-or-larger deform, escape-safe.

### 3. The pre-registered FALSIFIER FIRED — the escape ceiling is a fixed cell→fluid PUSH (coupling threshold), not a motility×coupling product
b09 pre-registered: "if `slow + mass 8e-5` also leaks, escape is coupling-driven regardless of motility → 1B needs a
containment fix, not more coupling." **It leaked.** `slow_mass80x` (move 0.12, mass 8e-5) → escape **0.0455**;
`slow_mass130x` (0.12, mass 1.3e-4) → escape **0.0227**. Since b08 s3 `mass25x_slow` (0.12, mass 5e-5) was escape-SAFE,
the mass-escape onset at slow motility sits in (5e-5, 8e-5]. This **refines** b08's "slowing motility decouples deform
from leak": slowing *raises* the mass-escape threshold (5e-5 safe at 0.12 but leaks at 0.24) but does **not remove** it —
past a coupling-gain threshold the cell→fluid→cell reaction flings a cell through the membrane regardless of motility.
**Implication: to keep exploiting agent_mass we now need a CONTAINMENT fix (raise the escape ceiling), whereas the k
route sidesteps the ceiling entirely.**

### 4. The b08 flagship (fast + mass 2e-5) is SEED-FRAGILE — cannot be a stable operating point
`flagship_seed1` (s6, embryo_1B_fast_seed1, mass 2e-5) → escape **0.0227**, but the identical recipe on seed 0 (b08 s0)
had escape **0**. mass 2e-5 at move 0.24 sits exactly on the escape boundary (b08 s0 r_cell_max 0.94) — one cell punches
out or not depending on seed. So the agent_mass-driven flagship **fails the ≥3-seed replication for [established]** and
must be retired as an operating point. The k route (`fast_k4`) is the replacement candidate: bigger clean deform with
a comfortable escape margin (r_cell_max 0.811 vs the leakers' 0.91–0.94) — its seed replication is Batch 10's priority.

### 5. Secondary reads (interpret, don't gate)
- **`flow_align` is NULL a THIRD time.** `fast_mass10x_align` (s5, gain 120 with strong coupling present) → polar_order
  0.0366 ≈ the 0.009 floor scale, net_circulation 0, deform_rms 0.00943 (LOWER than the plain fast+mass2e-5), and it
  still escaped. The earlier "maybe align was null only because there was no flow to align to" excuse is dead — even
  with a real coupling-driven flow field, flock alignment builds no coherence and no extra deform. [rejected] holds.
- **mid_mass25x (0.18, mass 5e-5)** gave the 2nd-highest fourier_m2 (0.01871) but escaped (0.0227) — consistent with the
  mass-escape onset being crossed by 0.18 at mass 5e-5 (safe only at 0.12 per b08 s3). Motility does modulate the
  threshold, but only weakly.
- **net_circulation ≈ 0 everywhere** (max 0.00188, s6) — the inner "flow" is still non-rotational bulk agitation, not a
  vortex; `mpm_spin` (baked at omega 0.3) still creates no measurable circulation, consistent with b07's [rejected].

### 6. HYPOTHESIS (Batch 10)
**`agent_to_mpm.k` is THE escape-safe 1B membrane-deform lever: raising it amplifies the collective inner-flow→membrane
deformation without the per-cell ballistic escape that `agent_mass` causes.** Predict k 4→6→8 (at move 0.24, mass 8e-6)
keeps escape=0 while fourier_m2 rises past 0.02 and deform_rms past 0.013, until some higher k eventually leaks (locating
the k ceiling). Secondary: a **containment fix raises the escape ceiling** enough to rescue the leaky mass route —
`mpm_to_agent.confine` 0.03→0.06 (stronger inward hold) and/or a **stiffer** membrane (youngs 200→500) should pull the
known-borderline `fast + mass 2e-5` point back to escape=0 (b08 falsified SOFTening as a *deform* lever; this tests
STIFFENING as a *containment* lever — a distinct role). Falsifier for the k claim: if `fast_k6`/`fast_k8` also escape,
then k too is bounded by the same fixed cell→fluid push and 1B's deform ceiling is set by containment, not coupling —
then the containment slots (confine↑, stiffen↑) become the only route and the operating point stays at `fast_k4`.

### 7. DESIGN — 8 slots (4 exploit / 3 explore / 1 control), all nodiv n44, 12000f, stride 16, confine 0.03 + repel 150
Judge every slot as Δdeform_rms / Δfourier_m2 AND escape vs the s7 `quiescent_ctrl` floor. New specs authored:
embryo_1B_fast_seed2.yaml (seed 2), embryo_1B_fast_stiff.yaml (membrane youngs 500). `agent_to_mpm.*`, `mpm_to_agent.*`
and per-op overrides are honoured by tune._apply; seed needs a spec file.
- `fast_k6` (exploit): _fast + agent_mass 8e-6 + `agent_to_mpm.k 6.0` — push the winning k lever past 4. Predict escape 0, fourier_m2 > 0.02, deform_rms > 0.013.
- `fast_k8` (exploit): _fast + agent_mass 8e-6 + `agent_to_mpm.k 8.0` — locate the k escape ceiling (if any). Predict deform up; escape onset test (the k-route falsifier).
- `fast_k4_seed1` (exploit): embryo_1B_fast_seed1 + agent_mass 8e-6 + `agent_to_mpm.k 4.0` — 2nd seed of the b09 clean flagship toward [established]. Predict escape 0, deform_rms ≈ 0.013, fourier_m2 ≈ 0.02.
- `fast_k4_seed2` (exploit): embryo_1B_fast_seed2 + agent_mass 8e-6 + `agent_to_mpm.k 4.0` — 3rd seed of the flagship → promote "k is the deform lever" to [established] if all 3 hold escape=0 + high deform.
- `k4_mass13` (explore): _fast + agent_mass 1.3e-5 + `agent_to_mpm.k 4.0` — add modest mass (below the leaky 2e-5) to the clean k4 winner for MORE deform; tests whether k and a sub-threshold mass stack cleanly. Predict deform_rms > 0.013, escape borderline.
- `mass20_confine6` (explore): _fast + agent_mass 2e-5 + `mpm_to_agent.confine 0.06` — CONTAINMENT test: does doubling the inward hold pull the borderline mass 2e-5 route (escaped on seed1) back to escape=0? If so, confine raises the escape ceiling. (Watch 1A: confine 0.05 was clean in b05.)
- `mass20_stiff` (explore): embryo_1B_fast_stiff (membrane youngs 500) + agent_mass 2e-5 — CONTAINMENT test #2: does a STIFFER shell contain the leaky mass route (inverse of b08's SOFTening-leaks)? Escape=0 with deform preserved ⇒ stiffness is a containment lever.
- `quiescent_ctrl` (control): _base, no driver — deform floor + 1A-hold seed replicate. Judge every slot vs THIS.
