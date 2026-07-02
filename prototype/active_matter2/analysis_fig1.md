# analysis log -- Fig. 1 reproduction (agent-based collective states)

Append one dated section per batch: the question, per-slot (state, key params,
P/Nc/contrast, match verdict), the winner per state, the reasoning.

## Batch 0 (manual seed, pre-loop)
Established the six states manually (see knowledge_fig1.md). Montage assembled at
archive/fig1_reproduction.png. Weakest matches: active droplets (fixed via v2 =
medium-range attraction) and aggregation (inherently low-order). Streams / vortices /
rings / bands match well. Loop starts from here.

## Batch 1 (2026-07-02) -- lay down all six states + probe two open levers
QUESTION: no montage exists yet. Place all six collective states with the base operator set
(one slot each, from the batch-0 specs) to produce the FIRST reproduced montage vs
paper_fig1.png, and spend 2 slots on one-variable ablations of the top open questions.

TARGET MORPHOLOGY read from paper_fig1.png (top row = orientation, bottom = chemical c):
- streams (Fig 1g,l): long directed rivers of aligned agents flowing to wave sources;
  orientation ~one dominant colour per lane; c = elongated traveling wave-front lanes.
- ring-streams (Fig 1e,j): agents circulate on CLOSED loops; orientation = full colour wheel
  arranged azimuthally round the loop; c = smooth annular/ring wave front (open centre).
- active-droplets (Fig 1k): compact motile blobs, coherently polarised (single colour) and
  migrating (paper draws a red velocity arrow); c = compact well/spot, sometimes a tail.
- vortices (Fig 1h,m): dense rotating disk; orientation = rainbow pinwheel round a core;
  c = spiral wave emanating from the core. FILLED centre (vs ring's open centre).
- polar-bands (Fig 1i,n): straight travelling stripe of aligned agents (single colour) on an
  almost-empty background; c = a matching straight band.
- aggregation (Fig 1, last col): disordered coarsening clusters, NO polar order (mottled
  colour); c = foam/labyrinth of wells; Nc decreases over time.

SLOTS (all --kind agent; params = deviations from AGENT_DEFAULTS):
- s0 streams        : defaults (omega0.38 gamma0.15 beta0.16 Dc0.16 decay0.02 n8000). parent.
- s1 rings          : gamma0.30 omega0.45 Dc0.22 decay0.014 eps0.035 beta0.18 n9000 res220.
- s2 vortex         : omega0.55 gamma0.18 beta0.22 eps0.04 Dc0.18 decay0.018 n12000 v0=0.005.
- s3 droplets       : omega0.80 beta0.24 Dc0.13 decay0.025 repel0.04 r0 0.016 radius0.04 n4500.
- s4 bands          : gamma0.42 omega0.06 beta0.08 v0=0.007 n9000.
- s5 aggregation    : gamma0.0 omega0.40 n8000, marker dot.
- s6 vortex_fast    : ABLATION A -- vortex with v0 0.005->0.011 (ONE var). Does faster self-
  propulsion open the filled vortex into an open ring? tests rings-vs-vortex lever.
- s7 streams_Dc     : ABLATION B -- streams with Dc 0.16->0.30 (ONE var). Does faster diffusion
  smooth the labyrinthine c into paper-like traveling wave fronts?

ANTICIPATED (to be judged against panels/montage next batch):
- SURPRISE to watch: rings vs vortices may be one knob apart (v0/omega). If s6 opens s2's disk
  into a ring, the lever is self-propulsion, not gamma/omega -- would collapse two "established"
  regimes into one axis.
- Expected weakest matches (batch-0 carryover): droplets reading as noisy rather than coherently
  coloured; aggregation inherently low-order; chemical fields labyrinthine vs paper's smooth
  wave-fronts (s7 tests the fix). Levers: droplets->gamma up; c-smoothness->Dc up / beta down.
- A slot with no panel.png FAILED -> redesign around it next batch.
VERDICTS: see Batch 2 below (montage fig1_b01_montage.png read).

## Batch 2 (2026-07-02) -- read the first montage; isolate the condensation + de-foam levers
QUESTION: the first montage is in. Only ONE state (bands) matches. Everything else is
too GAS-LIKE (sparse specks, no coherent aggregate) AND the chemical is LABYRINTHINE
foam rather than paper-like coherent waves (annuli/spirals/lanes/blobs). What single knob
condenses the gas, and what single knob de-foams the chemical?

PER-SLOT READ of Batch 1 (panel.png + progress.txt vs paper_fig1.png):
- s0 streams     P=0.309 Nc=13 ctr=2.43 sig=0.401 : MISMATCH. Sparse gas-like colour streaks;
  chemical a dense reticular FOAM, not the paper's elongated wave-front lanes. Lever: gamma / de-foam.
- s1 rings       P=0.157 Nc=6  ctr=4.59 sig=0.126 : MISMATCH. A few squiggly OPEN worm-streams,
  NO closed loop; chemical branching labyrinth, not an annulus. Lever: gamma up to close loops.
- s2 vortices    P=0.148 Nc=10 ctr=2.9  sig=0.416 : MISMATCH. Gas of tiny clusters, no filled
  rotating disk; chemical foam, no spiral. Lever: gamma up + density.
- s3 droplets    P=0.052 Nc=37 ctr=1.9  sig=0.384 : MISMATCH. FRAGMENTED into a gas of micro-wells
  (Nc=37!), not ONE compact motile blob -- the v1 failure mode returned. Lever: gamma / condensation.
- s4 bands       P=0.852 Nc=11 ctr=4.83 sig=0.093 : MATCH (WINNER). Coherent single-colour stripe
  (magenta->blue) on an empty background + a matching chemical band. Only strongly-ordered state.
  Curved (S-shaped) rather than straight, but morphology is right.
- s5 aggregation P=0.012 Nc=63 ctr=1.06 sig=0.596 : PARTIAL. Disordered specks (correct: no polar
  order) + fine foam chemical -- qualitatively right but the foam is too FINE/uniform, not the
  paper's coarsened compact clusters. Lever: sigma up to grow the correlation length.
- s6 vortex_fast P=0.05  Nc=28 ctr=1.5  sig=0.72  : ABLATION A (v0 0.005->0.011). Result: MORE
  gas-like (P 0.148->0.05), NOT a ring. => self-propulsion is a DISPERSER, not the ring lever. REJECT.
- s7 streams_Dc  P=0.076 Nc=13 ctr=2.59 sig=0.391 : ABLATION B (diffuse 0.16->0.30). Result: chemical
  blobbier but still FOAMY, particles MORE gas-like (P 0.309->0.076). => diffusion alone does NOT
  smooth foam into wave lanes. REJECT diffuse as the smoothness lever.

BIGGEST SURPRISE: both ablations moved AWAY from their targets in the SAME direction (more gas),
and bands -- the sole high-gamma state -- is the ONLY coherent match. This points at ONE master
lever: gamma (polar alignment) is the condensation/coherence knob; every failing state has
gamma<=0.30 and leans on chemotaxis, but on a foamy field chemotaxis just scatters agents into a
gas. Second lever (chemical de-foam): NOT diffuse (rejected s7); try sigma up / beta down.

WINNER per state: bands (s4) only. All others mismatch -> Batch 2 tests the two levers.

SLOTS (Batch 2): a 4-way single-variable lever sweep on the streams parent (s0=defaults) to find
what condenses/de-foams, then apply the leading direction (gamma up; sigma up for agg) to the other
mismatched states. Each slot = ONE variable changed from a named Batch-1 parent.
- s0 str_gamma  : parent b01_s0, gamma 0.15->0.35            (condensation lever)
- s1 str_sigma  : parent b01_s0, sigma 1.2->2.4              (chemical coherence: wider source)
- s2 str_beta   : parent b01_s0, beta 0.16->0.07             (de-foam: weaker, non-saturating emission)
- s3 str_slow   : parent b01_s0, move_speed 0.006->0.003     (dispersal lever)
- s4 ring_gamma : parent b01_s1, gamma 0.30->0.50            (close squiggles into loops)
- s5 vort_gamma : parent b01_s2, gamma 0.18->0.50            (fill the rotating disk)
- s6 drop_gamma : parent b01_s3, gamma 0.20->0.55            (coherent motile blob vs fragmented gas)
- s7 agg_sigma  : parent b01_s5, sigma 1.4->2.6              (coarsen the fine foam)
ANTICIPATED: if gamma-up condenses streams (s0) and simultaneously closes rings (s4)/fills vortex
(s5)/compacts droplets (s6), gamma is confirmed as the master coherence lever. If str_sigma/str_beta
smooth the chemical foam into lanes where diffuse (s7) failed, that isolates the de-foam knob. VERDICTS
pending next batch.

## Batch 3 (2026-07-02) -- read the gamma montage; the fill lever (loop -> disk)
QUESTION: gamma-up (batch 2) worked, but revealed a NEW structure. gamma is a single morphology
CONTINUUM: low=gas, medium=open directed streams, high=closed 1D loops (rings). Streams + rings are
now essentially SOLVED by gamma. But every "filled" target -- vortex ROTATING DISK, droplet compact
BLOB -- collapses to a THIN 1D loop, never a 2D filled region. What single knob FILLS a loop into a disk?

PER-SLOT READ of Batch 2 (panel.png + progress.txt vs paper_fig1.png):
- s0 str_gamma  P=0.132 Nc=12 ctr=5.27 sig=0.092 : MATCH (streams!). gamma 0.15->0.35 turned the gas
  into coherent wiggly directed RIVERS (rainbow-along-lane orientation) AND de-foamed the chemical into
  clean sinuous bright WAVE-FRONT LANES -- exactly the paper's stream chemical. P is LOW but misleading:
  global vector order is low because rivers point many ways; LOCAL order + morphology are right. WINNER streams.
- s1 str_sigma  P=0.301 Nc=16 ctr=2.79 sig=0.49  : REJECT sigma as de-foam lever. sigma 1.2->2.4 only
  widened the labyrinthine foam cells (still foam, signal way up), particles stayed gas. Not lanes.
- s2 str_beta   P=0.068 Nc=16 ctr=2.06 sig=0.341 : REJECT beta-down. Weaker emission -> less signal ->
  MORE gas-like, not de-foamed.
- s3 str_slow   P=0.087 Nc=32 ctr=2.37 sig=0.385 : REJECT slow. move_speed 0.006->0.003 did not condense.
- s4 ring_gamma P=0.355 Nc=4  ctr=6.11 sig=0.06  : MATCH-ish (rings). gamma 0.30->0.50 (P 0.157->0.355)
  produced a rainbow CLOSED LOOP (azimuthal colour wheel) + thin chemical wave-front lanes. Loops forming;
  irregular not circular yet. WINNER rings. Lever = gamma closes filaments into loops.
- s5 vort_gamma P=0.387 Nc=11 ctr=3.84 sig=0.09  : SURPRISE. gamma 0.18->0.50 (P 0.148->0.387) gave thick
  CLOSED LOOPS ("balloon-animal" contours) with a matching closed annular chemical front -- NOT a filled
  rotating disk. gamma condenses to 1D loops, never 2D disks. Vortex FILL unreached. Lever needed = FILL.
- s6 drop_gamma P=0.018 Nc=2  ctr=4.75 sig=0.058 : SURPRISE. gamma 0.20->0.55 collapsed the Nc=37 micro-well
  gas into just 2 compact HOLLOW rings (o-shapes), matching hollow chemical wells. Condensed but hollow, not
  a filled coherent-colour blob. P low because the 2 loops circulate. Droplet FILL unreached.
- s7 agg_sigma  P=0.01  Nc=63 ctr=0.94 sig=0.824 : PARTIAL. sigma 2.6 gave a COARSER foam that still FILLS
  the frame (contrast 0.94). Paper aggregation = compact isolated clusters with DARK gaps. Lever = decay up.

BIGGEST SURPRISE: gamma is not just a condenser -- it is a full morphology axis (gas->streams->loops), and
the DE-FOAM of the chemical is a FREE side-effect of it (aligned agents lay thin filament trails; sigma /
diffuse / beta all REJECTED as de-foam knobs). The deeper surprise: excitable relay makes TRAVELLING fronts
(thin ridges) that agents pack onto -> everything wants to be a 1D loop. The two "filled" states are a
DIFFERENT physics: a STATIC chemical well that agents fill in 2D. HYPOTHESIS: rings/streams = excitable
travelling-wave regime; vortices/droplets = quenched STATIC-well regime. The fill knob de-excites the wave:
eps down (slow refractory -> centre keeps emitting -> static dome) or decay up (wave can't propagate ->
localized well), or r0 up (agents physically spread into the interior).

WINNERS this batch: streams (s0 str_gamma), rings (s4 ring_gamma), bands (b01_s4, carried). Still mismatched:
vortices + droplets (both hollow loops, need FILL), aggregation (foam fills frame, need dark gaps).

SLOTS (Batch 3): 4 slots attack the FILL lever on the vortex/droplet parents (one variable each: eps down,
decay up, r0 up), 1 fixes aggregation (decay up for dark gaps), 3 anchor the solved states (rings/streams/
bands re-run as montage controls). Each fill slot changes exactly ONE var from its batch-2 parent.
- s0 vort_eps   : parent b02_s5 vort_gamma, eps 0.04->0.012   (FILL: slow refractory -> static well -> disk)
- s1 vort_decay : parent b02_s5 vort_gamma, decay 0.018->0.05 (FILL: localize chemical -> static well)
- s2 vort_repel : parent b02_s5 vort_gamma, r0 0.011->0.020   (FILL: physical spread into 2D interior)
- s3 drop_eps   : parent b02_s6 drop_gamma, eps 0.05->0.012   (FILL applied to droplets: hollow ring -> blob)
- s4 ring_gamma : re-run b02_s4 (gamma 0.50)                  RING anchor (montage control)
- s5 str_gamma  : re-run b02_s0 (gamma 0.35)                  STREAM anchor (montage control)
- s6 agg_decay  : parent b02_s7 agg_sigma, decay 0.03->0.08   (aggregation: dark gaps -> compact clusters)
- s7 bands      : re-run b01_s4 (gamma 0.42 omega 0.06)       BAND anchor (montage control)
ANTICIPATED: if eps-down (s0) OR decay-up (s1) fills the vortex loop into a rotating disk, the excitable->
static-well hypothesis is confirmed and the SAME knob should fill the droplet (s3). If r0-up (s2) fills
instead, the fill is mechanical (2D packing), not chemical. If NONE fill, the loop-attractor is topological
(needs a spawn/geometry change, not a rate). VERDICTS pending next batch.

## Batch 4 (2026-07-02) -- read the fill montage; eps is the FILL knob, omega is the hollow<->fill SELECTOR
QUESTION: batch 3 tested three fill knobs (eps down / decay up / r0 up) on the hollow vortex loop.
Which one turns a 1D loop into a 2D filled disk, and does the same knob fill the droplet?

PER-SLOT READ of Batch 3 (panel.png + progress.txt vs paper_fig1.png):
- s0 vort_eps   P=0.011 Nc=4  ctr=6.4  sig=0.031 : FILL BREAKTHROUGH (partial). eps 0.04->0.012 turned the
  hollow annular chemical into FILLED STATIC ROUND DOMES (bottom row = solid round/pill bright wells, dark
  centres GONE) and the surviving particle cluster is a THICK FILLED PILL (rainbow, top-right), not a hollow
  loop. Confirms excitable->static-well: slow adaptation -> centre keeps emitting -> static dome agents fill.
  BUT very sparse (Nc=4, most agents condensed away) and pill/capsule-shaped, not a round rotating disk.
- s1 vort_decay P=0.008 Nc=6  ctr=4.85 sig=0.047 : REJECT decay as fill. decay 0.018->0.05 left thick square
  HOLLOW loops ("balloon-animal" contours) with hollow ring wells -- no fill. Faster decay localizes but the
  front is still a thin travelling ridge agents rim-lock onto.
- s2 vort_repel P=0.518 Nc=5  ctr=2.46 sig=0.198 : REJECT r0 as fill -- and a NEW finding: r0 0.011->0.020
  BROKE the loops back into OPEN wiggly STREAMS (P jumped 0.387->0.518, sinuous chemical lanes). r0/repel is a
  loop->stream DISPERSER (opposite of fill): too much hard-core spread stops filaments from closing.
- s3 drop_eps   P=0.013 Nc=1  ctr=5.07 sig=0.048 : eps down did NOT fill the droplet. Same eps 0.012 as s0 but
  omega=0.8 (vs vortex 0.55) -> ONE big HOLLOW ring + hollow ring well. The ONLY difference from the filled s0
  is omega. => omega is the hollow<->fill SELECTOR: high omega (0.8) glues agents to the gradient RIM (hollow);
  moderate omega (0.55) lets them fill the static dome interior.
- s4 ring_gamma P=0.355 Nc=4  ctr=6.11 sig=0.06  : rings anchor. One clean rainbow closed ring (top-left) +
  a chemical ring (bottom-left) amid stream lanes. Sparse but a genuine ring. Carries as montage anchor.
- s5 str_gamma  P=0.132 Nc=12 ctr=5.27 sig=0.092 : streams anchor. MATCH -- wiggly directed rainbow rivers +
  clean sinuous bright chemical wave-front lanes. Best montage anchor.
- s6 agg_decay  P=0.008 Nc=62 ctr=0.97 sig=0.628 : REJECT decay for aggregation gaps. decay 0.03->0.08 left a
  space-FILLING foam (ctr 0.97, essentially unchanged from 0.94). Paper wants compact clusters + DARK gaps.
  Decay shrinks well amplitude uniformly; it does not carve dark background between clusters. Lever = diffuse
  down (pin chemical to clusters) and/or omega up (stronger chemotactic collapse into isolated peaks).
- s7 bands      P=0.852 Nc=11 ctr=4.83 sig=0.093 : bands anchor. MATCH -- coherent single-colour (magenta->
  blue) stripe on empty background + matching chemical band. Curved (S) not straight but morphology right.

BIGGEST SURPRISE: eps DOWN really is the FILL knob (s0: hollow annulus -> filled static dome), CONFIRMING the
excitable-travelling-wave vs quenched-static-well dichotomy. The deeper surprise is the CLEAN two-state
contrast s0(omega0.55, FILLED) vs s3(omega0.8, HOLLOW) at identical eps: omega is the hollow<->fill selector.
So FILL = eps DOWN (make the well static) + omega MODERATE (don't rim-lock). Bonus: r0 UP is the loop->stream
disperser (s2), the reverse direction on the gamma continuum.

WINNERS this batch: streams (s5), bands (s7), rings (s4 anchor). FILL knob IDENTIFIED (eps down) but not yet a
clean filled DISK (s0 sparse/pill). Still mismatched: vortex (filled rotating disk), droplet (filled coherent
blob), aggregation (dark gaps). Batch 4 tests omega-as-fill-selector on BOTH vortex + droplet.

SLOTS (Batch 4): the key experiment is omega DOWN on the two eps-down (static-well) parents -- if it fills
BOTH the vortex and the droplet, omega is confirmed as the hollow<->fill selector (two-state causal proof).
Plus density/gamma helpers for a rounder/coherent blob, two aggregation gap knobs, and streams/bands anchors.
- s0 vort_lowom : parent b03_s0 vort_eps,  omega 0.55->0.25  (FILL: unlock from rim -> filled rotating disk)
- s1 vort_dense : parent b03_s0 vort_eps,  n 12000->20000    (FILL helper: more mass to fill the disk)
- s2 drop_lowom : parent b03_s3 drop_eps,  omega 0.8->0.30   (FILL test on droplet: hollow ring -> filled blob)
- s3 drop_gamma : parent b03_s3 drop_eps,  gamma 0.55->0.78  (coherent SINGLE-COLOUR migrating blob)
- s4 agg_diffuse: parent b03_s6 agg_decay, diffuse 0.12->0.05 (aggregation: pin chemical -> dark gaps)
- s5 agg_omega  : parent b03_s6 agg_decay, omega 0.4->0.85   (aggregation: stronger collapse into peaks)
- s6 str_gamma  : re-run b03_s5 (streams anchor, montage control)
- s7 bands      : re-run b03_s7 (bands anchor, montage control)
ANTICIPATED: if omega-down fills BOTH s0 and s2, omega is the confirmed hollow/fill selector and the
excitable/static-well map is complete. If drop_gamma (s3) gives a single-colour compact blob, droplet is
solved (colour = gamma, fill = eps+omega). If agg_diffuse or agg_omega opens dark gaps (ctr drops well below
0.9), aggregation is solved. VERDICTS pending next batch.

## Batch 5 (2026-07-02) -- read the omega-fill montage; omega-as-fill-selector is REJECTED, the hollow is a MILL
QUESTION: batch 4 predicted omega DOWN would FILL both the eps-down vortex and droplet (the batch-3
vort_eps "filled pill" vs drop_eps "hollow ring" two-state read). Does omega-down actually fill?

PER-SLOT READ of Batch 4 (panel.png + progress.txt vs paper_fig1.png):
- s0 vort_lowom P=0.357 Nc=5  ctr=4.27 sig=0.069 : REJECT omega-down as fill. omega 0.55->0.25 (eps0.012) did
  NOT fill -- it DISPERSED the loop into OPEN wiggly worm-STREAMS (thin bright chemical ridge lanes, hollow).
  Weaker chemotaxis -> agents don't hold the well -> they string out along the travelling front = streams.
- s1 vort_dense P=0.016 Nc=2  ctr=5.13 sig=0.046 : REJECT density as fill AND fails to reproduce batch-3's
  "filled pill". n 12000->20000 at the SAME omega0.55/eps0.012 as b03_s0 gave a thick HOLLOW rounded-rectangle
  loop, not a filled dome. => the batch-3 "filled static dome/pill" reading was NOT reproducible; it was a
  transient/misread. More mass just thickens the hollow rim.
- s2 drop_lowom P=0.043 Nc=8  ctr=3.74 sig=0.092 : REJECT omega-down as fill (droplet). omega 0.8->0.30 stayed
  HOLLOW -- big open o-loops + hollow ring wells. omega down did not fill the droplet either.
- s3 drop_gamma P=0.362 Nc=2  ctr=5.18 sig=0.054 : gamma 0.55->0.78 (omega0.8) = one big HOLLOW rounded-square
  loop (banded colour) + hollow ring well. gamma sharpens the colour banding but does NOT fill; still a mill.
- s4 agg_diffuse P=0.012 Nc=58 ctr=0.94 sig=0.647 : diffuse 0.12->0.05 made the foam cells FINER but still a
  space-FILLING labyrinth (ctr 0.94). Pinning the chemical shrinks cells; it does not carve dark gaps.
- s5 agg_omega  P=0.009 Nc=53 ctr=1.45 sig=0.429 : BEST aggregation lever. omega 0.4->0.85 raised contrast
  0.94->1.45 -- clusters pulled into MORE SEPARATED compact bright blobs on a darker background (Keller-Segel
  collapse). Stronger chemotactic collapse = the gap-opener decay/sigma/diffuse all failed to be. Lever = omega UP.
- s6 str_gamma  P=0.132 Nc=12 ctr=5.27 sig=0.092 : streams anchor -- MATCH (wiggly rivers + sinuous lanes).
- s7 bands      P=0.852 Nc=11 ctr=4.83 sig=0.093 : bands anchor -- MATCH (single-colour stripe + band).

BIGGEST SURPRISE: omega DOWN does NOT fill -- it does the OPPOSITE on the vortex (disperses the loop into open
streams) and nothing on the droplet (stays hollow). Combined with vort_dense failing to reproduce the batch-3
"filled pill" at identical params, the whole "omega is the hollow<->fill selector" claim (batch 4) COLLAPSES.
The deeper realization: the hollow is a MILL. Self-propelled agents (v0>0) cannot come to rest in an attractive
well -- they ORBIT it, so any chemotactic trap becomes a hollow rotating ring, never a filled disk. omega only
sets how tightly they lock to the orbit (rim); it can't stop the orbiting. The UNTESTED fill lever is
move_speed (v0) DOWN: slow enough agents get pulled all the way into the core instead of orbiting. Also eps->0
(pure static Keller-Segel well, no travelling front at all) is untested at the limit.

WINNERS this batch: streams (s6), bands (s7); aggregation IMPROVED (s5 omega-up, ctr 1.45). Rings carried from
b03_s4. FILL still UNREACHED for vortex + droplet -- but the mechanism is now diagnosed as the mill, not omega.

SLOTS (Batch 5): the key experiment is move_speed DOWN on the vortex AND droplet static-well parents -- if slow
self-propulsion fills BOTH, "the hollow is a mill, v0 is the fill lever" is confirmed (two-state proof, replacing
the rejected omega selector). Plus eps->0 (pure static well) and eps-up (restore excitability -> spiral, the
paper vortex chemical IS a spiral) as bracketing controls, aggregation pushed further (omega up), + 3 anchors.
- s0 drop_slow   : parent b04_s2 drop_lowom,  move_speed 0.006->0.002  (FILL: kill the mill -> filled blob)
- s1 vort_slow   : parent b04_s1 vort_dense,  move_speed 0.005->0.0015 (FILL: kill the mill -> filled disk; two-state)
- s2 drop_eps0   : parent b04_s2 drop_lowom,  eps 0.012->0.0           (pure static Keller-Segel well, no front)
- s3 vort_excite : parent b04_s1 vort_dense,  eps 0.012->0.07          (RESTORE excitability -> spiral-wave vortex)
- s4 agg_omega2  : parent b04_s5 agg_omega,   omega 0.85->1.3          (push collapse -> isolated compact clusters)
- s5 str_gamma   : anchor (streams, montage control)
- s6 ring_gamma  : anchor (rings, montage control)
- s7 bands       : anchor (bands, montage control)
ANTICIPATED: if move_speed-down fills BOTH s0 and s1 (dark centre -> bright filled disk/blob), the mill diagnosis
is confirmed and v0 is the fill lever (omega selector retired). If eps0 (s2) gives the cleanest filled droplet,
droplet = pure static well + slow v0. If vort_excite (s3) shows a rotating spiral chemical core, that is the
paper's vortex regime (excitable, NOT static). If agg_omega2 (s4) pushes ctr well above 1.45, omega-up is the
confirmed aggregation gap-opener. VERDICTS pending next batch.

## Batch 6 (2026-07-02) -- read the mill/fill montage; v0-DOWN FILLS (droplet solved), vortex problem REFRAMED to CONSOLIDATION
QUESTION: batch 5 predicted move_speed (v0) DOWN would fill BOTH the vortex disk and the droplet blob (the
"hollow is a mill" two-state proof). Does slow self-propulsion actually fill, and does it fill the SAME way
for both states?

PER-SLOT READ of Batch 5 (panel.png + progress.txt vs paper_fig1.png):
- s0 drop_slow   P=0.336 Nc=5  ctr=3.17 sig=0.139 : DROPLET WINNER -- the mill IS killed. move_speed 0.006->0.002
  gave a COMPACT blob with a coherent RED core (single-colour polar order = the migration direction, matching the
  paper's red velocity arrow in Fig 1k) + a spiral migration tail; chemical = a mostly-FILLED compact well + tail.
  P 0.043->0.336. First real droplet: slow agents are pulled INTO the core instead of orbiting it. Confirms v0 is
  the fill lever. (A faint hole remains at the very centre + the tail is long -- refine with gamma / lower v0.)
- s1 vort_slow   P=0.14  Nc=25 ctr=3.21 sig=0.135 : FILL WORKS, but FRAGMENTS. move_speed 0.005->0.0015 (n20000)
  did NOT make ONE big filled disk -- it made ~25 SMALL rainbow PINWHEELS, each a mini FILLED rotating vortex.
  So v0-down kills the mill (every small cluster is a filled rotating disk, exactly the vortex morphology) but the
  population nucleates on MANY local wells. => fill is SOLVED; the remaining vortex mismatch is SCALE/CONSOLIDATION
  (get one dominant well), a DIFFERENT lever than fill. Retires "vortex fill is unreachable".
- s2 drop_eps0   P=0.453 Nc=3  ctr=1.52 sig=0.399 : REJECT eps->0. Removing adaptation lets the source ACCUMULATE
  into a SATURATED space-filling network of FAT channels (ctr 1.52 = washed out); agents pack into fat rivers, no
  compact droplet. Adaptation (source refractory/decay) is NECESSARY to localize the well. eps=0 is WORSE than 0.012.
- s3 vort_excite P=0.01  Nc=3  ctr=4.12 sig=0.07  : REJECT eps-up as fill; but it makes CRISP hollow RINGS. eps
  0.012->0.07 (more excitable) sharpens the travelling front -> agents rim-lock into clean CLOSED loops with
  azimuthal rainbow + a thin bright annular front. Beautiful ring-streams morphology (square = periodic-lattice
  artifact). => excitability sharpens RINGS, it does NOT fill vortices. Vortex is NOT the excitable regime.
- s4 agg_omega2  P=0.019 Nc=59 ctr=1.8  sig=0.354 : aggregation GOOD + omega-up confirmed. omega 0.85->1.3 raised
  ctr 1.45->1.8 -- rounded, coarsened compact wells packed on a darker ground (Keller-Segel foam), a good match to
  the paper's aggregation. Note Nc rose to 59 (many small wells): pushing omega further sharpens contrast but
  FRAGMENTS rather than coarsens. omega-up = the gap-opener, but there is a point of diminishing returns.
- s5 str_gamma   P=0.132 Nc=12 ctr=5.27 sig=0.092 : streams anchor -- MATCH (wiggly rivers + sinuous wave lanes).
- s6 ring_gamma  P=0.355 Nc=4  ctr=6.11 sig=0.06  : rings anchor -- MATCH (closed azimuthal loops + annular front).
- s7 bands       P=0.852 Nc=11 ctr=4.83 sig=0.093 : bands anchor -- MATCH (single-colour stripe + band).

BIGGEST SURPRISE: v0-down is a TRUE fill lever, but the two-state proof SPLIT. The DROPLET filled (compact coherent
blob) -- the mill diagnosis is confirmed. The VORTEX did NOT make one disk; it fragmented into ~25 small FILLED
pinwheels. So "fill" (kill the mill) is solved for both -- but the vortex now has a NEW, separate problem: too many
nucleation wells. The vortex montage panel looks gas-like only because the correct filled disks are TOO SMALL and
TOO NUMEROUS. This reframes the vortex from "cannot fill" to "cannot consolidate to a single dominant well".

WINNERS this batch: DROPLETS (s0 drop_slow, v0-down -- solved), aggregation (s4 omega1.3), streams/rings/bands
anchors. Vortex fill mechanism found (v0-down); consolidation is the last open lever.

SLOTS (Batch 6): attack vortex CONSOLIDATION -- grow the field correlation length so ONE well dominates and the
20000 agents pack into a single big filled rotating disk. Three orthogonal one-var levers off vort_slow:
diffuse-UP (fewer maxima), sigma-UP (wider well), and v0 0.0015->0.003 (bracket the frozen<->mill axis: bigger
disks that still rotate). Plus drop_slow re-run as the droplet anchor, and streams/rings/bands/aggregation anchors
for the full montage.
- s0 vort_diffuse : parent b05_s1 vort_slow,  diffuse 0.18->0.45  (smooth field -> single dominant maximum)
- s1 vort_sigma   : parent b05_s1 vort_slow,  sigma 1.3->3.0      (wider source -> one large well/disk)
- s2 vort_v003    : parent b05_s1 vort_slow,  move_speed 0.0015->0.003 (bracket v0: bigger rotating disks)
- s3 drop_slow    : re-run b05 winner (droplet anchor, v0-down compact coherent blob)
- s4 str_gamma    : anchor (streams)
- s5 ring_gamma   : anchor (rings)
- s6 bands        : anchor (bands)
- s7 agg_omega2   : anchor (aggregation, omega1.3 coarsened foam)
ANTICIPATED: if diffuse-up OR sigma-up collapses the 25 pinwheels into ONE big filled rotating disk (Nc->1-3, one
rainbow pinwheel), vortex is solved and the lever is field correlation length (consolidation), not fill. If v0=0.003
gives bigger disks that still rotate, it brackets the frozen(0.0015)<->mill(0.005) axis. If all three still fragment,
the barrier is the domain/interaction ratio (one well can't dominate at this box size) -> next batch shrinks n or box.
VERDICTS pending next batch.

## Batch 7 (2026-07-02) -- vortex CONSOLIDATION verdicts: consolidation & fill are in TENSION; pivot to "rotating droplet"
QUESTION: batch 6 tested three consolidators off vort_slow (the ~25-pinwheel fragment state) to merge the small
filled pinwheels into ONE big filled rotating disk: diffuse-UP, sigma-UP, v0-UP. Which grows the field correlation
length enough that a single well dominates -- and does the consolidated well stay FILLED?

PER-SLOT READ of Batch 6 (panel.png + progress.txt vs paper_fig1.png):
- s0 vort_diffuse P=0.069 Nc=21 ctr=3.36 sig=0.131 : REJECT diffuse as consolidator. diffuse 0.18->0.45 smoothed
  each well (soft round blobs) but did NOT reduce their NUMBER -- still ~21 small filled pinwheels, same fragment
  state as vort_slow. Smoothing lowers each maximum's sharpness but every nucleation site survives. diffuse is a
  blur knob, not a merge knob.
- s1 vort_sigma   P=0.259 Nc=4  ctr=3.22 sig=0.137 : CONSOLIDATION WORKS but reintroduces the HOLLOW. sigma 1.3->3.0
  (wider source) merged the population onto Nc=4 BIG wells -- consolidation achieved. BUT each big well is a THICK
  HOLLOW LOOP (a fat square channel with an empty interior): the mill RETURNS at large well size. The wider well is
  big enough that even slow (v0=0.0015) agents rim-lock onto its travelling FRONT and orbit -> hollow. Fewer wells,
  but not filled disks.
- s2 vort_v003    P=0.07  Nc=2  ctr=4.06 sig=0.075 : same story, stronger. v0 0.0015->0.003 consolidated to Nc=2 but
  a single big HOLLOW rectangular loop -- the faster speed is squarely back in the mill regime. Confirms v0-UP
  re-creates the hollow.

BIGGEST SURPRISE: FILL and CONSOLIDATION are in direct TENSION. b06 proves consolidation is achievable (sigma-up /
v0-up drop Nc 25->4->2) -- but every consolidated LARGE well is HOLLOW, because high-c is a thin travelling FRONT
(excitable relay ridge) and agents rim-lock onto a 1D curve. Small wells -> filled (pinwheel); large wells -> hollow
(loop). The root cause is the same one flagged in the ledger: "every filled target collapses to a thin loop" because
the chemical high-region is a 1D moving ridge, not a 2D basin. So chasing "one big well" via correlation length was
the WRONG axis -- it buys consolidation at the cost of re-hollowing.

THE PIVOT (re-reading the droplet anchor s3, P=0.336): the SOLVED droplet is ALREADY half a vortex. Its head is a
compact blob with AZIMUTHAL rainbow colour (red top / cyan-blue bottom = rotation) around a near-filled core, plus a
long single-colour MIGRATION TAIL of agents escaping the well. Read this way, droplet vs vortex is NOT a fill problem
and NOT a consolidation problem -- both are the SAME compact filled object; the only difference is whether its polar
order is TRANSLATIONAL (migrates -> comma/tail = droplet) or ROTATIONAL (recirculates -> pinwheel disk = vortex).
The tail exists because self-propulsion (v0) ejects agents faster than chemotaxis (omega) can turn them back into
the well. => VORTEX = a droplet whose escape-tail is suppressed so agents RECIRCULATE. The lever is omega UP (bind
tighter -> no escape -> rotating disk), NOT correlation length. This retires the whole "consolidate to one big well"
program for the vortex.

WINNERS this batch: vortex CONSOLIDATION diagnosed (sigma-up consolidates but re-hollows; diffuse doesn't merge;
tension named). Droplet re-read as a proto-vortex reframes the vortex from unreachable to "suppress the droplet tail".
All other states hold (streams/rings/bands/droplet/aggregation anchors matched in b06 montage).

SLOTS (Batch 7): test "vortex = rotating droplet". Off the SOLVED droplet (drop_slow, a filled comma with a rotating
head + escape tail), raise omega to suppress the migration tail and force recirculation into a filled rotating disk;
dose-response on omega, plus a v0-up bracket, plus ONE consolidated-well fill (decay-down) as a hedge for a BIGGER
disk. Four anchors for the montage.
- s0 vort_drop_om55 : parent drop_slow, omega 0.30->0.60  (bind tighter -> suppress tail -> rotating disk)
- s1 vort_drop_om10 : parent drop_slow, omega 0.30->1.00  (push further -- omega dose-response on the tail)
- s2 vort_drop_v04  : parent drop_slow, move_speed 0.002->0.004 (bracket: does more propulsion make a bigger rotating
                      disk, a mill, or disperse?)
- s3 vort_sig_slodec: parent b06 vort_sigma, decay 0.018->0.006 (fill the CONSOLIDATED big well: slow decay keeps the
                      interior high-c behind the front -> agents fill 2D instead of rim-locking -- a bigger-disk hedge)
- s4 drop_slow      : anchor (droplet)
- s5 str_gamma      : anchor (streams)
- s6 ring_gamma     : anchor (rings)
- s7 bands          : anchor (bands)
ANTICIPATED: if omega-up shortens the droplet tail and the head grows into a filled azimuthal PINWHEEL (rainbow disk,
no single-colour tail), vortex is SOLVED and the lever is omega (translational->rotational), not correlation length.
If om10 over-binds into a static point (no rotation) while om55 rotates, there is an optimal binding. If v0-up makes a
bigger rotating disk it is a size knob; if it re-hollows, the mill threshold in v0 is confirmed. If decay-down fills
the big consolidated well (Nc few, filled), that is an alternate BIGGER-disk route. Aggregation held from b06 (ctr 1.8).
VERDICTS pending next batch.

## Batch 8 (2026-07-02) -- "vortex = rotating droplet via omega-up" REJECTED; pivot to the EXCITABLE SPIRAL WAVE
QUESTION going in (batch 7): does raising omega off the SOLVED droplet (drop_slow) suppress the migration tail and
grow the rotating head into a filled azimuthal pinwheel DISK (= vortex)? Bracket with v0-up; hedge with a slow-decay
fill of the consolidated sigma-well.

PER-SLOT READ of Batch 7 (panel.png + progress.txt vs paper_fig1.png):
- s0 vort_drop_om55  omega0.30->0.60  P=0.178 Nc=23 ctr=3.0  : REJECT. omega-up did the OPPOSITE of suppressing the
  tail -- it RE-FRAGMENTED the single droplet into ~23 small mini-pinwheels scattered over the box (chemical = a foam
  of ~23 compact bright blobs). No single growing disk, no tail suppression. The droplet's one-well compactness at
  omega0.30 was NOT a bound state to tighten; it was simply that WEAK chemotaxis let the whole population share ONE
  well. Strong chemotaxis makes every local density bump its own attractor.
- s1 vort_drop_om10  omega0.30->1.00  P=0.06  Nc=32 ctr=2.92 : REJECT, harder. Even MORE fragmented (Nc 32), P
  collapsed to gas (0.06). Confirms the monotonic trend: omega is a NUCLEATION-COUNT knob at fixed density -- more
  omega -> more, smaller clusters -> more fragmented foam. It is NOT a tail-suppressor / disk-binder.
- s2 vort_drop_v04   v0 0.002->0.004  P=0.445 Nc=2  ctr=3.31 : v0-up consolidated to Nc=2 but into two long thin
  sinuous FILAMENTS (worm-streams), NOT a disk. Confirms (again) v0-up = the loop/disk -> thin-filament disperser.
- s3 vort_sig_slodec sigma3.0 decay0.018->0.006  P=0.256 Nc=5 ctr=3.28 : PARTIAL POSITIVE + a lesson. Slow decay DID
  fill 2D -- the chemical is now a FAT SPACE-FILLING CHANNEL/LABYRINTH network (thick filled channels, interior stays
  high-c behind the front, exactly as predicted) and the agents sit on thick azimuthal loops. So "slow decay keeps the
  interior high" is CONFIRMED as a 2D-fill mechanism -- but it fills EVERYWHERE the front has swept (a labyrinth), not
  a compact rotating DISK. Fill without localization = labyrinth, not vortex.
- s4 drop_slow  P=0.336 Nc=5 : droplet anchor holds (compact comma head w/ azimuthal rainbow + migration tail; faint
  central hole remains). GOOD droplet.
- s5 str_gamma  : streams anchor holds.
- s6 ring_gamma P=0.355 Nc=4 : REGRESSED on seed 3 -- sparse diagonal filaments + 1-2 short arcs, mostly empty box,
  no clean closed loops. Ring morphology is SEED-SENSITIVE at frames=1200; loops did not mature. Re-run w/ more frames.
- s7 bands : bands anchor holds.

BIGGEST SURPRISE: the entire "vortex = rotating droplet, suppress the tail" pivot from batch 6/7 is WRONG. omega does
not connect the droplet to the vortex -- pushed up it FRAGMENTS (Nc 5->23->32), pushed down it DISPERSES to streams.
The droplet and the vortex are NOT on an omega axis. Combined with the batch-6 tension result (v0/sigma consolidate
but RE-HOLLOW) and s3 here (slow decay fills but into a LABYRINTH), the conclusion is decisive: NO combination of the
mechanical/transport knobs (v0, omega, sigma, diffuse, decay, eps) yields a single compact FILLED ROTATING DISK,
because the chemical high-region under the current relay is always either a THIN 1D travelling front (-> agents
rim-lock onto a 1D curve -> hollow/loop/filament) or, with slow decay, a FAT 2D labyrinth (fill without localization).
The paper's vortex chemical (Fig 1h,m) is neither: it is a SPIRAL WAVE -- a 2D rotating structure with a phase
singularity at its core. A spiral sweeps ALL radii once per rotation period, so agents chemotaxing toward it fill a
rotating DISK (rainbow pinwheel). We have never produced a spiral because the relay is NOT actually an excitable
medium: the default c_th=-0.001 makes the emission gate Theta(c-c_th) ALWAYS ON (c>=0 > -0.001), so `relay` is just
constant emission modulated by refractory (1-s) -> target / ring / labyrinth fronts, never spirals. THE UNTESTED
LEVER (finally identified as the vortex route): a REAL threshold c_th>0 turns the medium excitable
(quiescent c<c_th / excited / refractory s->1), and in a noisy excitable medium broken fronts curl into rotating
SPIRALS. The moving, density-fluctuating agents supply the heterogeneity that breaks fronts.

MECHANISM ADDED (am2_ops.py Relay + am2_job.py): a `c_base` baseline emission. With c_th>0 and a zero-initialized
field the gate would be a DEAD medium (chicken-and-egg: agents relay only where c>c_th, but nothing seeds c). c_base
adds weak sub-threshold sourcing beta*c_base so the field can build to c_th and IGNITE; above c_th the relay is full
(beta). gate = clamp(Theta(c-c_th) + c_base, max=1). c_base=0 (default) reproduces the old behaviour EXACTLY, so all
other states are unchanged. Back-of-envelope: at c_base~0.05, beta~0.28, decay~0.02, n8000/res220, baseline
equilibrium c ~ (source_density)/decay lands near c_th~0.1 -> dense patches ignite, sparse ones stay quiescent =
proper excitable dynamics.

WINNERS this batch: a decisive NEGATIVE (omega does not link droplet->vortex; the mechanical-knob program for the
vortex is EXHAUSTED) + a reframing that finally names the vortex mechanism as an EXCITABLE SPIRAL WAVE + a new
`c_base` knob that makes c_th>0 testable. s3 confirms slow-decay = a real 2D-fill mechanism (labyrinth).

SLOTS (Batch 8): PROBE the excitable spiral-wave regime (the vortex route) with the new c_th/c_base knobs on a base
tuned for wave propagation (fast diffuse, slow-ish decay, slow eps=long refractory, moderate omega so agents ride but
don't fragment), + re-anchor the other five states for the montage (rings with MORE FRAMES to mature loops).
- s0 vort_spiral   : parent -- c_th0.10 c_base0.05 diffuse0.30 eps0.02 omega0.45 gamma0.25 v0 0.004 (excitable base)
- s1 vort_spir_th05: parent, c_th 0.10->0.05   (lower ignition threshold -> broader excited area / easier waves)
- s2 vort_spir_cb08: parent, c_base 0.05->0.08 (stronger seed -> more of the medium ignites; seed-sensitivity test)
- s3 vort_spir_dif : parent, diffuse 0.30->0.55 (faster propagation -> longer spiral wavelength / whole-box spirals)
- s4 str_gamma     : anchor (streams)
- s5 ring_more     : anchor (rings), frames 1200->1600 (mature the loops; b07 seed-3 was sparse/immature)
- s6 drop_slow     : anchor (droplet)
- s7 bands         : anchor (bands)
ANTICIPATED: if any c_th>0 slot yields a ROTATING SPIRAL in c (a curling front with a core) and agents form a filled
rotating rainbow disk over it, the VORTEX is SOLVED and its lever is EXCITABILITY (c_th>0 + c_base seed), not any
transport knob. th05 vs th10 brackets ignition; cb08 tests whether too much seed washes back to always-on foam;
diff-up sets the spiral wavelength. If ALL c_th>0 slots give a dead/quiescent field or just relabelled foam/target
waves, the spiral is unreachable with a single-field single-threshold relay (would need a two-variable
FitzHugh-Nagumo-style inhibitor) -- that itself bounds the mechanism. VERDICTS pending next batch.

## Batch 9 (2026-07-02) -- the excitable medium makes WAVES but only PLANE ones; NUCLEATE the spiral with a broken-front seed
QUESTION going in (batch 8): does a real threshold c_th>0 (+ the new c_base baseline seed) turn the relay into an
excitable medium whose broken fronts curl into a rotating SPIRAL, filling a rainbow disk (= the vortex)?

PER-SLOT READ of Batch 8 (panel.png + progress.txt vs paper_fig1.png):
- s0 vort_spiral    c_th0.10 c_base0.05 diffuse0.30  P=0.186 Nc=3 ctr=5.46 : PARTIAL/PATCHY. The medium is UNDER-seeded --
  only locally dense patches cross threshold, so it ignites in scattered spots -> one small rectangular HOLLOW loop + a few
  stream fragments + a fragmented patchy chemical (bright blobs, one hollow rectangle). No percolating wave, no spiral.
  Lever: more seed / lower threshold to get a connected excitable medium.
- s1 vort_spir_th05 c_th0.10->0.05                    P=0.009 Nc=2 ctr=6.01 : still LOOPS. Lower threshold gave two clean
  HOLLOW rainbow C-loops pinned at the box corners (periodic-lattice artifact) + matching hollow-loop chemical. Excitable
  enough to make crisp closed fronts, but agents rim-lock onto the 1D front -> loop, NOT a 2D spiral. Same failure as always.
- s2 vort_spir_cb08 c_base0.05->0.08                  P=0.804 Nc=1 ctr=5.23 : OVER-seeded -> ONE PLANE WAVE. The whole medium
  now ignites nearly simultaneously, so the chemical is a SINGLE clean percolating traveling front spanning the box (a wavy
  horizontal band) and all agents align along it (P=0.80, band-like). This is a genuine clean traveling WAVE FRONT (a real
  win vs foam) -- but it is a PLANE wave, not a curled spiral. Too much seed = coherent plane wave.
- s3 vort_spir_dif  diffuse0.30->0.55                 P=0.798 Nc=2 ctr=4.51 : PLANE waves, longer wavelength. Faster diffusion
  gave two long DIAGONAL parallel traveling fronts (agents strung along them, P=0.80). diffuse sets the wave SPEED/wavelength
  as predicted -- but the fronts stay straight PLANE waves, they do not break/curl. Confirms diffuse = wavelength knob.
- s4 str_gamma  P=0.132 Nc=12 : streams anchor holds (wiggly rivers + sinuous lanes).
- s5 ring_more  P=0.188 Nc=3 ctr=6.79 : rings anchor, frames1600. A couple of arcs/loops; better than b07's sparse seed-3 but
  loops still not fully closed/circular. Ring morphology remains seed-sensitive; acceptable montage anchor.
- s6 drop_slow  P=0.336 Nc=5 : droplet anchor holds (compact comma head w/ azimuthal rainbow + migration tail).
- s7 bands      P=0.852 Nc=11 : bands anchor holds (single-colour stripe + band).

BIGGEST SURPRISE: c_th>0 + c_base genuinely WORKS as an excitable medium -- for the first time the chemical is a set of clean
TRAVELLING WAVE FRONTS (thin, percolating, coherent) rather than foam or a fat labyrinth. But the fronts are PLANE/TARGET
waves: under-seeded (c_th high / c_base low, s0) the medium ignites patchily into disconnected hollow loops; over-seeded
(c_base or diffuse up, s2/s3) the whole medium lights up as ONE coherent plane wave (P~0.8, band-like). Neither breaks into a
rotating spiral. This is the textbook fact about excitable media: a SPIRAL does not self-nucleate from smooth/noisy initial
conditions -- it requires a BROKEN wave front (a free tip = phase singularity) that then winds up. We have the medium; we
lack the singularity. c_th brackets excitability (loops vs plane), c_base brackets ignition fraction (patchy vs plane),
diffuse sets wavelength -- all confirmed -- but none of them BREAKS a front.

MECHANISM ADDED (am2_ops.py SpiralSeed + am2_job.py): a `spiral_seed` operator that stamps the textbook one-shot cross-field
IC on the FIRST tick only: a half-plane wave FRONT (a vertical stripe of high c over the LOWER half of y, broken at mid-height
so its tip is free) + a REFRACTORY tail just behind it (agents there get s->0.95) so the wave advances into the fresh region and
the free tip curls. In a periodic box a single break yields a counter-rotating spiral PAIR. spiral_seed=0 (default) reproduces
the old behaviour EXACTLY, so all other states are unchanged. c_base/c_th keep the medium primed so the seeded wave propagates
rather than dying. (Op reviewed against relay for API parity; could not smoke-run this session -- python exec was gated -- so s3
keeps the SAME base with spiral_seed=0 as a control that runs even if the seed op faults.)

WINNERS this batch: a decisive mechanism finding -- the excitable relay makes real traveling WAVE FRONTS (foam/labyrinth is
gone) but only PLANE/TARGET waves; the spiral is a NUCLEATION (broken-front) problem, not a medium problem. New `spiral_seed`
knob makes the broken-front IC testable. All five other states hold as anchors.

SLOTS (Batch 9): NUCLEATE the spiral. Seed the broken front on a slow (v0=0.003, fill-regime) excitable base; vary ONE knob
per slot -- wavelength (diffuse), excitability (c_th) -- plus a no-seed control; 4 anchors complete the montage.
- s0 vort_seed      : parent -- excitable base (c_th0.08 c_base0.04 diffuse0.30 eps0.02 omega0.6 gamma0.25 v0 0.003 n12000) + spiral_seed=1
- s1 vort_seed_dif  : parent, diffuse 0.30->0.18   (shorter wavelength -> tighter spiral core, more turns fit the box)
- s2 vort_seed_th05 : parent, c_th 0.08->0.05      (more excitable -> free tip CURLS rather than retracts)
- s3 vort_noseed    : parent, spiral_seed 1->0     (CONTROL: no broken front -> confirms the spiral needs seeding, not dynamics)
- s4 str_gamma      : anchor (streams)
- s5 ring_more      : anchor (rings)
- s6 drop_slow      : anchor (droplet)
- s7 bands          : anchor (bands)
ANTICIPATED: if s0 yields a rotating SPIRAL in c (a curling front/pair with a phase-singularity core) and agents fill a
rotating rainbow DISK over it while s3 (no seed) stays a plane wave / patchy loops, the VORTEX is SOLVED and its lever is
NUCLEATION of a broken front in the (already-built) excitable medium -- not any transport knob. diffuse-down (s1) should
tighten the core; c_th-down (s2) should make the tip curl more readily (sub-excitable tips retract instead). If the seeded
spiral forms but then DRIFTS out / annihilates with its periodic partner, spiral persistence is the next open question
(pin the core / single-tip seed). If NO slot curls even when handed a broken front, the single-field single-threshold relay
cannot sustain a spiral and a two-variable (FitzHugh-Nagumo inhibitor) medium is required. VERDICTS pending next batch.

## Batch 10 (2026-07-02) -- the one-shot broken-front seed WASHES OUT; the vortex needs a FIELD (continuum) refractory -> add FitzHugh-Nagumo `refract`
QUESTION going in (batch 9): does the new `spiral_seed` one-shot broken-front IC nucleate a PERSISTENT rotating SPIRAL
(free tip curls, agents fill a rainbow disk), while the no-seed control stays a plane wave / patchy loops?

PER-SLOT READ of Batch 9 (panel.png + progress.txt vs paper_fig1.png):
- s0 vort_seed     spiral_seed1  P=0.29  Nc=6 ctr=4.38 : NOT a spiral. Final frame = a handful of SMALL FILLED rainbow
  pinwheels (each a correct mini rotating disk) + one sinuous worm-stream; chemical = compact blobs + a worm channel.
  The seeded broken front is GONE -- no rotating arm, no core, no percolating spiral.
- s1 vort_seed_dif diffuse0.18   P=0.105 Nc=6 ctr=4.59 : same attractor, shorter-range -> a few hollow C-loops + small
  filled blobs + fragments. Not a spiral.
- s2 vort_seed_th05 c_th0.05     P=0.09  Nc=10 ctr=4.81 : same -- several small filled rainbow pinwheels + a "b"-loop +
  a comma head. More excitable = MORE small blobs, not one big spiral.
- s3 vort_noseed   spiral_seed0  P=0.271 Nc=6 ctr=4.47 : the CONTROL is MORPHOLOGICALLY INDISTINGUISHABLE from s0
  (seed): small filled rainbow pinwheels + a C-shaped worm; chemical = compact blobs + a C worm. DECISIVE.
- s4 str_gamma P=0.132 : streams anchor holds.  s5 ring_more P=0.188 Nc=3 : rings anchor (a couple of arcs; not fully
  closed, still seed-sensitive).  s6 drop_slow P=0.336 Nc=5 : droplet anchor holds.  s7 bands P=0.852 : bands anchor holds.

BIGGEST SURPRISE: seed (s0) and NO-seed (s3) converge to the SAME final attractor (small filled rainbow pinwheels +
worm channels). The one-shot broken-front IC has NO lasting effect at frame 1800 -- exactly the batch-9 risk (the
seeded spiral drifts/annihilates). ROOT CAUSE (now nailed): the refractory recovery variable `s` lives on the AGENTS
(per-agent, Eq 5), NOT on the FIELD. The agents that carry `s` are themselves chemotactically pulled INTO the very
front they create, so they scramble/advect the refractory tail within a few ticks and the broken-front geometry is
destroyed before it can wind. Our "excitable medium" is therefore only excitable WHERE AGENTS ARE, and its recovery
is MOBILE -- so it cannot pin a phase singularity. A spiral needs a recovery variable fixed IN SPACE (a continuum
inhibitor field), which the single shared-field + per-agent-s model simply does not have. This is why b08's fronts
were only plane/target waves and b09's seed washed out: not a nucleation failure, a MEDIUM (missing-field-inhibitor)
failure. The pinwheel attractor is the model's honest best vortex proxy: each blob IS a correct mini rotating disk
(azimuthal rainbow, filled) -- fill works [b06], SCALE (one big disk) fails, because the medium can't hold a spiral.

MECHANISM ADDED (am2_ops.py `Refract` + Relay rf_th; am2_job.py rf_tau/rf_gain/rf_th): a per-voxel refractory FIELD
buffer `fld._rf` making the chemical a genuine two-variable (activator c / inhibitor rf) EXCITABLE MEDIUM in the
CONTINUUM. `refract` (scheduled after decay when rf_tau>0) evolves d_t rf = gain*Theta(c-c_th) - rf/tau, rf in [0,1];
`relay` with rf_th<1 is BLOCKED wherever rf>rf_th, so a just-passed front leaves a refractory WAKE fixed in space that
the next front cannot re-invade -> a broken front cannot heal, its free tip pins a phase singularity and winds into a
SUSTAINED rotating spiral. tau = refractory period (spiral core size / wavelength); gain = wake build-up rate. Fully
DEFAULT-OFF: rf_th=2.0 (rf clamped to 1 -> never blocks) and `refract` unscheduled unless rf_tau>0, so every other
state (and the no-rf vortex control) is byte-for-byte unchanged. (Python exec was gated this session, as in b09, so
the op ships reviewed-not-run; blast radius is contained to the rf_tau>0 slots, and a safe no-rf pinwheel anchor keeps
the montage's vortex tile populated either way.)

WINNERS this batch: a DECISIVE mechanism finding -- the one-shot broken-front seed does not persist (seed==no-seed at
the final frame) because refractoriness is carried by MOBILE agents, not the field; the vortex is a MEDIUM problem
(missing continuum inhibitor), not a nucleation problem. New continuum-refractory `refract` op makes the FitzHugh-Nagumo
route testable. All five other states hold as montage anchors; aggregation re-added for a complete 6-state final montage.

SLOTS (Batch 10 -- FINAL): TEST the continuum excitable medium as the vortex route + complete the 6-state montage.
- s0 vort_fhn     : excitable slow-fill base + spiral_seed1 + refract (rf_tau40 rf_gain0.10 rf_th0.5) -- does a
  space-fixed refractory wake let the seeded broken front WIND into a sustained rotating spiral disk?
- s1 vort_fhn_tau : s0, rf_tau 40->80 (ONE var: longer refractory period -> larger spiral core / fewer, bigger arms).
- s2 vort_pin     : CONTROL/anchor -- same base, rf_tau0 spiral_seed0 (the b09 small-pinwheel attractor: what the
  medium does WITHOUT continuum refractory). Guarantees a vortex tile in the montage even if the FHN op faults.
- s3 str_gamma / s4 ring_more / s5 drop_slow / s6 bands / s7 aggreg : the five solved-state anchors.
ANTICIPATED: if s0/s1 show a curling front with a persistent core in c and agents on a rotating arm (spiral) while s2
stays small pinwheels, the VORTEX is SOLVED and its lever is a CONTINUUM (field) refractory -- the paper's actual
excitable-medium, not any transport knob; rf_tau then sets the core size. If s0/s1 still give small pinwheels / a plane
wave / a dead field, the continuum inhibitor as implemented is insufficient (e.g. rf couples too weakly, or a spiral
needs the FULL FitzHugh-Nagumo activator nonlinearity), and the single-scalar-relay model's honest vortex proxy is the
mini-pinwheel field -- which is what the montage will report. Either way the parameter->state map for the other five
states is complete and the vortex mechanism is bounded.
