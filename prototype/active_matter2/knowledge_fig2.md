# knowledge ledger -- Fig. 2 (hydrodynamic phase diagram)

CUMULATIVE. Seeded from initial manual runs. OUR units (aggregation threshold omega~1).

## Established
- Base PRESETS['fig'] (sigma0.7 delta0.6 chi0.5 Drho0.5 Dp0.6 Dc1.1 Q0.5 alpha0.42
  beta0.6 eps0.045 c_th=-1): with v0=0.6, omega=1.8 -> a VORTEX LATTICE (polarization +1
  defects = full HSV colour wheels) with chemical peaks. Strong match to a vortex state.
- omega is the aggregation/condensation lever: omega<~0.6 -> weak aggregation (streams/
  homogeneous+polar); omega>~1.5 -> strong condensation (droplets/vortices).
- v0 is the motility lever: low v0 + high omega -> droplets/vortices; raising v0 tends to
  elongate/split aggregates (toward rings/streams); high v0 -> polar bands (Vicsek-like).
- c_th < 0 (always-on refractory emission) is REQUIRED for a lively c field; c_th>0 with a
  seed decays to a near-uniform c (sub-excitable) -- FALSIFIED as the figure regime.
- WHY the levers act as they do (mechanism, from Eqs 6-9): omega multiplies rho*grad(c) in
  the p-equation -> it is CHEMOTACTIC self-attraction; large omega makes p climb its own
  signal gradient and condense density (aggregation). v0 sets -v0 div(p) in the rho-equation
  (density transport) and -v0 (p.grad)s in the s-equation (advective refractoriness): more v0
  = more advective stirring, which stretches/hollows/breaks condensates and, once it dominates
  chemotaxis, drives Vicsek-like travelling polar bands.

## VERIFIED (v0,omega)->state map [batch-2, from the batch-1 montage] -- the plane is BINARY
At the fig coefficients the (v0,omega) plane has only TWO phases, split by a sharp omega threshold:
  omega >~ 1.5 : CONDENSED VORTEX/ASTER LATTICE. P~0 (locally organised, net-zero), Nc 13-45,
    contrast 0.7-2.3, spiral/target c. v0 tunes ONLY blob count/size within this phase:
    Nc 32->38->42->45 as v0 0.60->0.78->0.90->1.00 (more, smaller vortices as v0 rises); v0=0.40
    UNDER-condenses (diffuse, ctr 0.73) because v0 feeds density transport (-v0 div p).
  omega <~ 1.1 : HOMOGENEOUS POLAR sheet. P~0.9, Nc~4, contrast~0.3, c ~0.6. Spans v0 0.6-1.0.
  boundary : aggregation onset is between omega 1.1 (polar) and 1.8 (vortex) -- ~1.5, and nearly
    v0-INDEPENDENT over v0 0.6-1.0. (Ledger's earlier "onset ~omega 1" is REVISED up to ~1.5.)
This FALSIFIES the batch-1 hypothesis that vortex->ring->band is a pure-v0 sequence: v0 never
changes the topology inside the condensed phase, and high v0 at high omega stays a vortex lattice
(does not break into bands). The four INTERMEDIATE paper states are NOT reachable on (v0,omega)
alone.

## Causal boundaries + the coefficients that shift them [the real levers, batch-2 hypotheses]
- VORTEX vs POLAR = chemotaxis (rho*omega*grad c) vs polar-growth (sigma*(rho-1)*p). omega is the
  master switch; the threshold ~1.5 is set by sigma and the c amplitude (beta/alpha). Vortices are
  +1 defects minted where chemotaxis pulls p onto the excitable/spiral c field.
- To make the jump GRADED (so droplets/rings/streams appear) you must reshape the c texture or the
  polar-onset margin, NOT just move (v0,omega):
    * Dc (c diffusion): higher -> broad annular waves -> density may pile on moving fronts = RINGS;
      lower -> tight localized wells -> compact condensates.
    * alpha (c decay): higher -> c decays fast -> sharp compact wells -> compact motile DROPLETS.
    * chi ((p.grad)p self-advection): drives the CURL that spirals p into pinwheel vortices. Lower
      chi -> less curl -> coherent single-hue blobs (droplet-like); higher chi -> channelises the
      polar sheet into finite STREAMS/BANDS.
    * sigma (density-coupled alignment; paper uses 0.02, we use 0.7): high sigma -> deeply ordered
      homogeneous polar (no coexistence). LOW sigma -> sit near rho_c=1 -> ordered density stripes
      coexist with disordered background = DISCRETE POLAR BANDS. This is the key untested lever.
- v0 lever sign in OUR model is OPPOSITE the paper for droplets: raising v0 aids aggregation
  (density transport) and shrinks/multiplies vortices; it does not open rings or make bands.

## Batch-3 findings: the vortex lattice is a ROBUST attractor; bands are a DENSITY effect
- Vortex<->polar onset SHARPENED: condensation survives at omega 1.6 (Nc=20 vortices) but is
  essentially gone by omega 1.4 (Nc=3, one relic defect on a polar sheet). Onset = ~1.5, ABRUPT
  (Nc jumps 3->20 over domega=0.2), nearly v0-independent. [w_14, w_16]
- The c-texture / polar-margin coefficients only RETUNE the vortex lattice, they never change its
  topology (confirmed, not hypothetical any more):
    * Dc UP (2.2) -> FEWER, LARGER filled vortices (Nc 32->17, ctr up) = best paper-b (few large
      disks) match, but still filled asters, NOT hollow rings.
    * alpha UP (0.85) -> fewer/larger vortices + weaker c (faster decay); no hollowing.
    * Dp UP (1.2) -> DE-CONDENSES (smears p): only 8 diffuse blobs, contrast collapses. Dp is a
      de-condensation knob, the OPPOSITE of "fewer clean vortices".
    * chi DOWN (0.1) -> the pinwheel winding relaxes into ELONGATED single-hue COMMA-streaks over
      elliptical c wells = the closest approach to coherent motile DROPLETS so far (but still ~31,
      not compact/round). chi is the pinwheel<->coherent-blob knob. Push chi->0 for true droplets.
- sigma is NOT the band lever. Dropping sigma 0.7->0.2 at (v0 1.0, omega 0.4) made the polar sheet
  MORE homogeneous (contrast 0.05), not banded. [band_sigma FALSIFIED] chi UP (0.9) there also stayed
  homogeneous. WHY: a uniform polar sheet has div(p)~0, so -v0 div(p) never structures density; you
  cannot band a flock that is uniformly deep in the ordered phase.
- THE band lever is rho0 (mean density). rho0=1.2 (hardcoded until batch-3) sits WELL ABOVE the
  flocking onset rho_c=1 -> a stable homogeneous flock with no dilute disordered gas to coexist with a
  dense band. Travelling polar bands (Toner-Tu/Vicsek microphase separation) only exist in a NARROW
  density window near rho_c. Added --rho0 to am2_hydro.py (default 1.2 unchanged); batch-3 tests
  rho0~1.0-1.05 at (v0 1.0, omega 0.4) for silent bands and omega~1.0 for signalling bands.

## Batch-4 findings: DROPLETS = sparse near-onset condensation; rings need off-curl+wave-expansion
- **DROPLETS ARE NOT A SEPARATE PHASE.** They are the condensed (vortex) phase at LOW NUCLEATION
  DENSITY. The blob count is a continuum set by proximity to the omega~1.5 onset AND by rho0:
    w 1.0 & rho0 1.05 -> 3 huge isolated blobs (sig_band, ctr 4.30, P 0.004) = clearest droplets;
    w 1.5 & rho0 1.2  -> 16 sparse isolated blobs (onset_15, ctr 2.56) = droplet/onset;
    w 1.8 & rho0 1.2  -> 32-blob SPACE-FILLING lattice (base) = vortices.
  Lowering omega toward onset or lowering rho0 = fewer nucleation sites = isolated compact droplets on
  an empty background; raising either fills space into the vortex lattice. So droplets<->vortices is
  ONE phase graded by count, NOT a phase boundary. (The paper puts droplets at low v0/high omega via a
  DIFFERENT axis convention; in OUR units the count knob is onset-proximity + rho0, ~v0-independent.)
- RING lever isolated: chi=0 (drop_chi0) minted the FIRST open annular c loops (hollow c rings between
  filled blobs). chi->0 removes the (p.grad)p curl that fills the aster core, but alone it does not
  hollow the DENSITY blob. Rings need chi~0 PLUS a wave-EXPANSION knob (high Dc for broad fast fronts,
  or low eps for a slow-recovery persistent refractory annulus) so density piles on an expanding front.
- ring_Dc_v0 (v0 0.95, Dc 2.2) = the CLEANEST regular vortex lattice yet (28 compact filled pinwheels
  on a triangular c-well array) = best paper-b topology match, still FILLED (high v0 does not hollow).
- c RUNAWAY at rho0=rho_c: rho0=1.0 (band_rho_lo) drives c to a uniform saturated ceiling (density
  contrast -> 0), homogenising the flock and masking any band. Retry bands with beta/alpha capping c.
- rho0 near onset (1.0-1.05) at low omega gave HOMOGENEOUS POLAR (P 0.89-0.98), NOT discrete bands.
  rho0 ALONE does not band at these coefficients -- bands remain the one fully-missing state (2 batches).

## Batch-5 findings: DROPLETS fully placed; RINGS are a c-WAVE signature; BANDS need LOW Drho + omega=0
- **DROPLETS are DONE -- both paper morphologies reproduced, and they are the SAME condensed phase tuned by
  count:**
    few-large : drop_lowrho (v0 1.0, w 1.0, rho0 1.02) -> 3 isolated compact blobs, each ringed by faint
      expanding c arcs. rho0 is the nucleation-count knob (1.02 -> ~3).
    many-small: ring_chi_v0 (v0 1.0, w 1.8, chi 0.0) -> 65 tiny single-hue blobs = paper-a "many small
      droplets". chi=0 keeps each blob a single coherent hue; HIGH v0 FRAGMENTS the condensate into many.
  So the droplet knobs are: rho0/omega (proximity to onset) sets COUNT; chi->0 makes each a coherent
  single-hue blob (not a pinwheel); v0 UP fragments into more/smaller. No new phase -- a tuned condensate.
- **RINGS are a CHEMICAL-WAVE phenomenon, not a density topology change.** chi=0 (off-curl) lets density sit
  on expanding c fronts; Dc UP (3.0) makes broad fast fronts -> the c FIELD forms open ARCS/CRESCENTS
  (ring_hiDc, Nc 12). But at w=1.8 the fronts are DENSE and COLLIDE before closing -> arcs, not full annuli.
  RING RECIPE (batch-5 test): chi=0 + high Dc + LOW nucleation density (lower omega toward onset OR lower
  rho0) so each isolated front can expand into a CLOSED annulus. eps is NOT a ring lever (ring_eps FALSIFIED
  -- slow recovery kept filled asters). The density blob only hollows where an isolated c ring can form.
- **BANDS: beta and sigma both FALSIFIED (3rd attempt).** band_beta (beta 0.2) and band_lowsig (sigma 0.3)
  at (v0 1.0, w 0.4, rho0 1.0) both stay a HOMOGENEOUS ORDERED sheet (P>0.9, contrast=0) with c saturated.
  Root cause (now firm): a uniformly ordered flock has div(p)~0, so -v0 div(p) never transports density; the
  Toner-Tu band needs a DENSITY MODULATIONAL INSTABILITY near rho_c, which Drho=0.5 is SMOOTHING AWAY.
  UNTRIED LEVER = Drho (density diffusion). Bands should appear at rho0~1.0-1.05 + LOW Drho (~0.15) + omega=0
  (chemotaxis OFF -> pure Toner-Tu; c decouples, its saturation no longer matters). This is the batch-5 bet.
  Note: with omega=0 the model reduces to Toner-Tu (drho=-v0 divp+Drho lap rho; dp=sigma(rho-1)p-delta|p|^2p
  +Dp lap p -chi(p.grad)p -Q grad rho) -- the canonical band-forming system. Density bands show in the TOP
  panel (rho-brightness) even if the c panel is saturated.

## Batch-6 findings: the Drho band bet FAILED; the band lever is rho0>onset + LOW Dp; streams != high chi
- **BANDS, 4th failure: omega=0 (pure Toner-Tu) + rho0=1.0 (=rho_c) + Drho 0.5->0.15 STILL gives a perfectly
  HOMOGENEOUS ORDERED sheet** (P 0.98-0.99, contrast=0). Low Drho ALONE does not destabilize the ordered
  state. v0 up to 1.6 there also stays uniform. So Drho is NOT the missing band lever (bet FALSIFIED).
- **The ONE band signal came from rho0 = 1.05 (slightly ABOVE onset), NOT 1.00 (=onset).** band_win (v0 1.0,
  w 0, rho0 1.05, Drho 0.15) showed the FIRST density modulation of any attempt: faint diagonal filaments,
  contrast 0.40, but still P 0.94 (too ordered). WHY: at rho0=1.00 the ordered amplitude p0->0 (marginal --
  nothing to transport, so it just aligns uniformly); at rho0=1.05 p0~=sqrt(sigma*0.05/delta)~0.24 and the
  near-onset banding instability turns on. Its linear growth ~ v0*sigma*p0; its damping ~ (Dp+Drho)*q^2. We
  lowered Drho but NEVER Dp -- Dp=0.6 in every band run smooths the polarization SPLAY mode that seeds bands.
  UNTRIED LEVER = Dp DOWN (~0.2) at rho0=1.05 to release the streaks into real bands. [batch-6 bet]
- **STREAMS: high chi FALSIFIED (2nd stream miss).** chi=1.2 at (v0 0.6, w 1.6) gave 10 discrete filled
  pinwheel ASTERS (P~0), not directed rivers. High chi channelizes a UNIFORM sheet but at high omega it just
  makes compact vortices. Streams likely need the ONSET layer (partial order, P~0.3) + ADVECTIVE STRETCHING
  (high v0), not chi. [batch-6 retry]
- **RINGS: crescents reconfirmed, not yet closed.** chi=0 + Dc3 at w=1.4 (ring_sparse, near onset) = 7 blobs
  with clear open c CRESCENTS/ARCS (P 0.32, ctr 0.5); lowering rho0 to 1.08 at w=1.8 (ring_lowrho) just gave
  sparse FILLED blobs. Fronts still merge before closing. Thin nucleation MORE (rho0 1.10 or omega 1.2 at
  chi=0/Dc3) so isolated fronts can expand into full annuli. [batch-6 retry]
- **DROPLETS: count continuum nailed down.** blob count vs omega (chi=0): w1.5->35 (drop_probe), w1.8->65
  (ring_chi_v0); with rho0 1.2: w1.5->16, w1.8->32. Count = onset-proximity(omega) x rho0; chi=0 keeps each
  a coherent single hue. DONE.

## Batch-7 findings: BANDS SOLVED (Dp-down OR rho0>onset); the polar<->condensed jump is sharp (no stream corridor)
- **BANDS ARE REACHED -- the batch-6 Dp bet was correct, and there are TWO routes, both above onset:**
    ROUTE A (low-Dp network): band_loDp (v0 1.0, w 0, rho0 1.05, Drho 0.15, Dp 0.2) -> CRISS-CROSS density
      ridge network (P 0.878, ctr 0.47); v0 up to 1.6 (band_loDp_v0) sharpens it to ctr 0.76 (best). The
      intersecting-river look is stream-adjacent.
    ROUTE B (rho0 stripes): band_rho11 (v0 1.0, w 0, rho0 1.10, Drho 0.15, default Dp 0.6) -> CLEAN PARALLEL
      VERTICAL bands (P 0.997, ctr 0.40) = cleanest match to paper panel-e SILENT bands.
  MECHANISM (now firm): the ordered flock buckles into bands when near-onset banding GROWTH (~ v0*sigma*p0,
  with p0=sqrt(sigma(rho0-1)/delta)) exceeds modulation DAMPING (~ (Dp+Drho)*q^2). Three independent knobs
  push it over: rho0 UP (raises p0), Dp or Drho DOWN (cuts damping -> releases the splay mode), v0 UP (raises
  drive). rho0=1.00 (=rho_c) fails because p0->0 (nothing to transport); rho0>=1.05 works. omega=0 => SILENT
  bands (c decoupled/passive). chi (band_chi) only WAVES the stripes, it is not a contrast lever. BANDS: DONE.
- **STREAMS still missing (3rd/4th failure) and the reason is now structural:** at w=1.45 the flock is ALREADY
  fully condensed even at v0=0.6 (stream_v0 P=0.019 Nc=27 droplets; stream_chi2 P=0.003 6 droplets). The
  polar<->condensed transition is SHARP (onset ~w 1.4) with NO partial-order corridor to "stretch". So the
  batch-6 "stretch the onset layer" idea is dead. Streams must live in the LOW-omega band regime with
  chemotaxis WEAKLY ON: take a band recipe (rho0>=1.05, low Dp) and add small omega (~0.5) so chemotaxis
  channels the density network into directed rivers, then advect with v0. [batch-7 stream bet]
- **RINGS closest yet but not closed:** ring_thin (v0 0.6, w 1.4, chi 0, Dc 3, rho0 1.10) -> open curved ARCS
  curving toward loops (P 0.683). Lowering omega to 1.2 (ring_om12) UNDER-condenses (diffuse, no fronts) --
  rings need w~1.4 (near onset), NOT lower. Close the annuli by thinning nucleation MORE at w~1.4: rho0 1.15
  or Dc 4 (broader faster isolated fronts). [batch-7 ring bet]

## Batch-8 findings: SIGNALLING BANDS + STREAMS + RINGS all placed -- 5/6 states landed
- **SIGNALLING BANDS = silent-band recipe + omega weakly ON.** From band_rho11 (rho0 1.10, Drho 0.15, omega 0),
  turning omega->0.5 (sigband_w05) makes c ACTIVE and CO-LOCATED on the density stripes = paper panel f (P 0.894,
  ctr 0.50). Raising v0 0.5->1.5 (sigband_v0) sharpens the wavy bands into CLEAN PARALLEL signalling stripes
  (P 0.983, ctr 0.67) -- the same v0-sharpening that works for silent bands, with signal still on. So omega is
  the SILENT(0)<->SIGNALLING(~0.5) switch; v0 is the band-sharpening knob for both. omega up to 0.8 (sigband_w08)
  starts nucleating defects (Nc 2->6): the band->condensed onset at v0=1.0 sits ABOVE w=0.8.
- **STREAMS = low-Dp band NETWORK + weak omega + HIGH v0.** From band_loDp (rho0 1.05, Dp 0.2, omega 0), adding
  weak omega 0.6 (stream_w06) channels the criss-cross network via chemotaxis, and raising v0->1.4 (stream_v0hi)
  advectively STRETCHES it into directed diagonal density RIVERS (ctr 0.70, best). Streams live at LOW omega +
  LOW Dp + HIGH v0. Caveat: the rivers CRISS-CROSS (intersecting lanes), not yet a single-direction field --
  the clearest directed-flow morphology so far but not a perfect paper-d match.
- **RINGS: the lever is Dc, NOT rho0.** ring_Dc4 (chi 0, Dc 4, w 1.4, rho0 1.10) shows a bright c ANNULUS round a
  dark core = clearest ring of the loop. SURPRISE: Dc 3->4 collapsed P 0.71->0.093 (partial arcs -> fully
  condensed aster) AND wrapped the density into a closed annular front. Broad/fast c fronts (high Dc) both
  NUCLEATE condensation and ring the core. rho0 1.15 at Dc=3 (ring_rho115) UNDER-condensed instead (P 0.71, diffuse)
  -- thinning nucleation without fast fronts does NOT close rings. Ring recipe: chi=0 + HIGH Dc (>=4) at w~1.4.
- **The two band routes are DISTINCT, not additive.** band_bothlev (rho0 1.10 + low Dp 0.2 together) gives the
  NETWORK morphology (criss-cross), NOT the sharpest parallel stripes -- LOW Dp dominates. Clean parallel stripes =
  rho0-up route at DEFAULT Dp; criss-cross network/streams = low-Dp route. Pick the route by target morphology.

## Batch-9 findings: the RING window is a NARROW non-monotonic optimum; high v0 is the master flow-sharpener
- **RINGS are the one FRAGILE state -- ring_Dc4 (v0 0.6, w 1.4, chi 0, Dc 4, rho0 1.10) is a knife-edge optimum.**
  EVERY single-variable push off it degraded the ring:
    * Dc UP (5.0): c over-diffuses -> grad(c) collapses -> NO condensation -> HOMOGENEOUS POLAR (P 0.98, ctr 0.26,
      sig 1.50 but FLAT). Dc is NON-MONOTONIC for rings: Dc 3=arcs, Dc 4=closed annulus, Dc 5=no pattern. Optimum ~4.
    * v0 UP (0.9): CONDENSES/fragments into ~6 compact FILLED droplets (ctr 2.39), does NOT hollow the core.
    * rho0 UP (1.20): ONE big FILLED aster (Nc 2, bright centre), NOT a hollow ring. Thinning gives fewer filled
      asters, not more hollow ones.
  So rings occupy a tiny box {chi=0, Dc~4, w~1.4, rho0~1.10} and remain the hardest paper state -- we get one aster's
  annulus, not a field of hollow rings. To get a FIELD, need MORE sites WITHOUT leaving the box (w up toward 1.5 or
  rho0 down toward 1.05 to nucleate more asters, each still ringed by an isolated fast front). [batch-9 bet]
- **HIGH v0 (1.8-2.0) is the master flow-sharpener for BOTH streams and signalling bands, and is numerically stable.**
    * stream_v018 (v0 1.8, w 0.6, rho0 1.05, Dp 0.2): criss-cross network -> BOLD single-hue near-parallel directed
      LANES (P 0.98, ctr 0.92) = cleanest stream/directed-flow of the whole loop.
    * sigband_v20 (v0 2.0, w 0.5, rho0 1.10): CLEAN vertical signalling stripes (P 0.99, ctr 0.93) = cleanest paper-f.
  v0 raises the banding drive (~v0*sigma*p0) AND advectively straightens criss-cross rivers into a coherent one-hue
  flow field. No blow-up at v0=2.0 (dt=0.02 stable). v0 is the sharpening knob; omega is the state selector.
- **Two boundaries pinned (phase-diagram edges):**
    * STREAM->CONDENSED onset: at v0 1.4 / low-Dp / rho0 1.05, omega 0.9 already CONDENSES the river network into an
      aster/foam lattice with defects (P 0.85->0.13). Streams live at omega ~0.6; omega>~0.8 condenses them. omega 0.3
      keeps an ordered CRISS-CROSS network (P 0.94, lanes intersect) -- too weak to channel into one-way flow.
    * SIGBAND->CONDENSED onset: at v0 1.5, omega 1.0 condenses the signalling bands into the vortex lattice (P 0.12,
      Nc 11); omega 0.5 = clean bands. The silent/signalling bands survive only for omega <~0.8.

## Batch-10 findings [FINAL]: the ring box is TIGHT; rho0-down is the ring field-multiplier; v0-sharpening SATURATES
- **RING window is tighter than the batch-9 box, and omega is NOT the field-multiplier inside it.** From ring_Dc4
  (v0 0.6, w 1.4, chi 0, Dc 4.0, rho0 1.10):
    * Dc UP 4.0->4.2 (ring_Dc42): ctr 0.09->0.47, P 0.09->0.41 -- already PAST the optimum (fronts over-broaden,
      grad c weakens, condensation releases toward polar). The Dc ring optimum is ~4.0, edge width <0.2.
    * omega UP 1.4->1.5 (ring_w15): COLLAPSES to HOMOGENEOUS POLAR (P 0.93, ctr 0.46). SURPRISE -- more omega was
      meant to add sites; instead, at Dc=4 the c field is already at the over-diffusion edge and extra omega
      saturates c -> grad(c) collapses -> chemotaxis loses grip (same failure mode as ring_Dc5). w must be ~1.4 EXACTLY.
    * v0 UP 0.6->0.7 (ring_v07): condenses 1-2 hollow-cored asters (ctr 1.16) but not a field; milder than the 0.9
      that fully fragments into filled droplets.
    * rho0 DOWN 1.10->1.05 (ring_rho105): Nc 2->9 sites at partial order (P 0.25) with visible c ARCS = BEST
      ring-FIELD candidate of the loop. **rho0-down (NOT omega-up) is the correct site-multiplier inside the ring box**
      -- thinning density nucleates more, smaller condensation sites, each still on an isolated fast c front. Arcs are
      not yet CLOSED annuli though. Ring box: {chi=0, Dc~4.0, w~1.4, rho0 1.05-1.10}.
- **v0-sharpening SATURATES above ~2.0 (with a stability ceiling well above).** sigband_v25 (v0 2.5, w 0.5, rho0 1.10):
  ctr 0.93 = IDENTICAL to v0 2.0 -- the stripe-sharpening plateaus, but the integrator stays stable to v0 2.5 (dt 0.02,
  no blow-up). v0 is the sharpener but with a diminishing-returns ceiling; 2.0 is already optimal.
- **Two condensation boundaries TIGHTENED (final phase-diagram edges):**
    * STREAM->CONDENSED onset at v0 1.8 (low-Dp, rho0 1.05): w 0.6 = clean bold lanes (stream_v016 at v0 1.6 ctr 0.76;
      stream_v018 ctr 0.92); w 0.75 (stream_w075) already CONDENSES the river net (P 0.96->0.17, Nc 9). Onset between
      w 0.6 and 0.75. Streams need omega <~0.7. v0 lane-straightening is monotonic 1.4->1.6->1.8 (ctr 0.70->0.76->0.92).
    * SIGBAND->CONDENSED onset at v0 2.0 (rho0 1.10): w 0.5 = clean vertical stripes (ctr 0.93); w 0.8 (sigband_w08)
      partially condenses into a foam/cellular defect network (P 0.99->0.56, Nc 2->14). Onset between w 0.5 and 0.8.

## Updated causal (v0,omega,coeff)->state map
- omega < ~1.4  : HOMOGENEOUS POLAR sheet (P~0.9), independent of sigma/chi/rho0(1.0-1.2). No bands yet.
- omega ~1.5    : SHARP condensation onset (polar<->condensed), nearly v0-independent.
- omega > ~1.5  : CONDENSED phase (one phase, topology = filled pinwheel/aster). BLOB COUNT continuum:
    near onset (w~1.5 or rho0~1.05) -> few, isolated = DROPLETS; deep (w~1.8, rho0 1.2) -> dense
    space-filling = VORTEX LATTICE. Within it: v0 up -> more/smaller; Dc/alpha up -> fewer/larger;
    Dp up -> de-condense; chi->0 -> coherent + first annular c loops (ring precursor).
- DROPLETS: DONE. few-large = low rho0/near-onset (drop_lowrho, 3 blobs); many-small = chi=0 + high v0
  (ring_chi_v0, 65 blobs). Count set by onset-proximity+rho0; chi->0 -> single-hue; v0 up -> fragments.
- RINGS: PLACED but FRAGILE [batch-8/9]. chi=0 + Dc~4 at w~1.4, rho0 1.10 -> ring_Dc4: condensed aster whose c
  wraps a bright ANNULUS round a dark core (P 0.093, clearest ring). Dc is the ring lever but NON-MONOTONIC with a
  sharp OPTIMUM at ~4: Dc 3=arcs, Dc 4=closed annulus, Dc 5=c over-diffuses -> gradient collapses -> HOMOGENEOUS
  POLAR (batch-9 ring_Dc5, P 0.98). EVERY push off ring_Dc4 degrades it: v0 up (0.9)=filled droplets, rho0 up
  (1.20)=one big filled aster, Dc up (5)=no pattern (all batch-9). Rings live in a tiny box; still one aster's
  annulus, NOT a field. [batch-10] Box TIGHTENED to {chi=0, Dc~4.0 (edge width <0.2: Dc 4.2 already releases toward
  polar), w~1.4 EXACTLY (w 1.5 COLLAPSES to homogeneous polar -- c saturates, grad c dies), rho0 1.05-1.10}. The
  FIELD-multiplier is rho0 DOWN, NOT omega up: ring_rho105 (rho0 1.05) = 9 arced sites (P 0.25, best ring-field
  candidate); arcs still OPEN, not closed annuli. Rings remain the one state we approach only as a field of arcs.
- BANDS (SILENT): SOLVED [batch-7]. omega=0 + (rho0>=1.05 above onset) + weak modulation damping. Two DISTINCT
  routes (NOT additive -- low Dp dominates if combined, batch-8 band_bothlev):
    A) rho0 1.05 + LOW Dp 0.2 -> criss-cross density-ridge NETWORK (ctr up to 0.76 with v0 1.6). = STREAM route.
    B) rho0 1.10 + default Dp 0.6 -> CLEAN PARALLEL vertical stripes (paper panel-e). P stays ~0.9-1.0
       (bands are a gentle density modulation ON an ordered flock). Levers: rho0 up / Dp,Drho down / v0 up
       all increase (growth v0*sigma*p0 - damping (Dp+Drho)q^2). chi only waves stripes (not contrast).
- BANDS (SIGNALLING): PLACED + SHARPENED [batch-8/9]. Route-B silent recipe (rho0>=1.10, Drho 0.15) + omega ON
  ~0.5 -> c co-travels ON the density stripes = paper f. v0 sharpens: v0 1.5 (sigband_v0, ctr 0.67) -> v0 2.0
  (batch-9 sigband_v20, P 0.99, ctr 0.93 = CLEANEST paper-f, no blow-up). omega=0<->0.5 is the silent<->signalling
  switch; v0 is the master sharpener for both. CONDENSATION onset: omega ~1.0 at v0 1.5 condenses the bands into
  the vortex lattice (batch-9 sigband_w10, P 0.12); sig-bands survive only omega <~0.8.
- STREAMS: PLACED + SHARPENED [batch-8/9]. LOW-omega band NETWORK route (rho0 1.05, LOW Dp 0.2) + WEAK omega
  (~0.6) + HIGH v0 -> directed density RIVERS. v0 is the lane-straightener: v0 1.4=criss-cross (stream_v0hi, ctr
  0.70); v0 1.8=BOLD single-hue near-parallel LANES (batch-9 stream_v018, P 0.98, ctr 0.92 = cleanest flow of the
  loop). omega WINDOW is tight: 0.3=criss-cross ordered net (too weak to channel), 0.6=directed rivers, >=0.9
  CONDENSES into an aster/foam lattice (batch-9 stream_om09, P 0.13). (Onset-layer route stays DEAD: w>=1.4 is
  fully condensed even at v0 0.6.)

## Open questions [batch-2]
- Does raising Dc (ring_Dc, Dc 2.2) hollow the condensate into open annuli = RINGS?
- Does killing chi (drop_chi, chi 0.1) or raising alpha (ring_alpha, alpha 0.85) turn pinwheel
  vortices into coherent single-hue compact DROPLETS?
- Does dropping sigma toward the paper's value (band_sigma, sigma 0.2 at v0 1.0 omega 0.4) split
  the homogeneous polar sheet into DISCRETE travelling bands (density stripes on empty bg)?
- Does the sharp vortex<->polar boundary layer (w_14, w_16 at v0 0.6) host STREAMS (partial
  condensation + directed flow) or a ring/spiral intermediate?
- Can we get FEWER/LARGER vortices to better match paper-b by raising Dp (vortex_Dp, Dp 1.2)?
- ANSWERED batch-2: aggregation onset ~omega 1.5 (not 1); v0 does not open rings/bands; high-v0
  slots did NOT blow up (dt=0.02 stable through v0=1.0).
- ANSWERED batch-3: Dc up = fewer/larger vortices but still FILLED (no rings); chi->0.1 = coherent
  comma-streaks (proto-droplets, not yet compact); alpha up = fewer/larger; Dp up = de-condense;
  sigma down does NOT band (FALSIFIED). Bands need rho0 near rho_c, not (v0,omega,sigma).
- OPEN [batch-3]: does rho0~1.05 at (v0 1.0, w 0.4) give SILENT travelling bands? at w~1.0 give
  SIGNALLING bands? Does chi->0 (near onset) turn comma-streaks into compact motile DROPLETS? Does
  high v0 on broad-Dc waves (ring_Dc + v0 0.95) finally hollow an aster into a RING?
- ANSWERED batch-4: rho0~1.05/1.0 at low omega = HOMOGENEOUS POLAR, NOT bands (rho0 alone insufficient;
  rho0=1.0 also runs c away). chi->0 = coherent blobs + FIRST annular c loops but filled density (ring
  precursor, not droplet). DROPLETS instead = SPARSE condensation near onset / low rho0 (sig_band 3
  blobs, onset_15 16 blobs) = same phase as vortices, count graded by onset-proximity. High v0 + Dc 2.2
  = clean filled vortex lattice, does NOT hollow.
- ANSWERED batch-5: DROPLETS DONE (few-large=drop_lowrho 3 blobs; many-small=ring_chi_v0 65 blobs, the
  paper-a morphology, via chi=0+high v0). chi=0+Dc3 (ring_hiDc) paints c-field ARCS but at w=1.8 fronts
  collide -> not closed annuli. eps FALSIFIED as ring lever. beta down & sigma down FALSIFIED as band
  levers (uniform ordered sheet, c saturated). droplet<->vortex count crossover ~omega 1.6-1.7 (w1.5->16,
  w1.7->26, w1.8->32 blobs).
- OPEN [batch-5]: does chi=0 + Dc3 at LOW omega/rho0 (sparse isolated fronts) CLOSE the c arcs into full
  RINGS + hollow the density? does omega=0 + LOW Drho (0.15) near rho_c finally give travelling POLAR
  BANDS (pure Toner-Tu, chemotaxis off)? does high chi at the turbulent onset layer channelise into
  STREAMS? where is the banding density window in rho0 (1.0<->1.05)?

## Rejected
- Weak omega (0.01-0.2, paper's literal axis) in OUR units: no aggregation, all homogeneous/
  polar -- confirms the axis rescaling. [FALSIFIED as literal reuse]
