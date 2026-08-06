# knowledge ledger -- Fig. 1 (agent-based collective states)

CUMULATIVE causal memory. Seeded from the initial manual reproduction (batch 0). Update
each batch; reclassify if overturned; never erase.

## Target morphology per state (read from paper_fig1.png; top=orientation, bottom=c)
- streams: directed rivers, ~1 colour/lane; c = elongated wave-front lanes.
- ring-streams: closed loop, azimuthal colour wheel; c = annular front, OPEN centre.
- active-droplets: compact motile blob, single (coherent) colour + migration; c = compact well.
- vortices: rotating disk, rainbow pinwheel; c = spiral wave; FILLED centre.
- polar-bands: straight travelling stripe, single colour, empty background; c = matching band.
- aggregation: disordered coarsening clusters, mottled colour; c = foam/labyrinth; Nc falls.
Discriminators: ring vs vortex = OPEN vs FILLED centre; band vs stream = straight-stripe vs
directed-river + empty-vs-flowing background; droplet vs aggregation = coherent-colour+compact
vs mottled+coarsening.

## THE MASTER AXIS: gamma is a morphology CONTINUUM (Established @ batch2 montage)
gamma (polar alignment) is not merely a condenser -- it is the single axis that sweeps the whole
gas->streams->loops sequence, AND it de-foams the chemical for free. Confirmed by the batch-2 gamma sweep:
- gamma ~0.15 : GAS. Agents rely on chemotaxis over a fine foam -> chase random wells -> disperse. c = foam.
- gamma ~0.35 : OPEN STREAMS. Agents align into coherent wiggly directed RIVERS; the chemical DE-FOAMS into
  clean sinuous WAVE-FRONT LANES (aligned agents lay a thin filament trail). [str_gamma s0 -- streams WINNER]
- gamma ~0.50 : CLOSED 1D LOOPS = rings. Filaments curl and close into loops with azimuthal (rainbow)
  circulation + annular chemical fronts. [ring_gamma s4 -- rings WINNER]
MECHANISM: excitable relay makes TRAVELLING fronts (thin chemical ridges); aligned agents pack ONTO the
ridge -> 1D filament; more gamma -> the filament closes into a loop. This is why de-foam is a gamma
side-effect (the trail replaces the foam), and why every "filled" target collapses to a thin loop.

## THE FILL PROBLEM: the hollow is a MILL, v0 (move_speed) DOWN is the fill lever (Established @ batch6)
Vortices (rotating DISK) and droplets (compact BLOB) are FILLED 2D regions that USED to collapse to a hollow ring.
Batch-6 CONFIRMED the mill diagnosis and found the fill lever; the vortex problem is now REFRAMED:
- THE HOLLOW IS A MILL, and v0-DOWN FILLS IT. [Established @ batch6] Self-propelled agents (v0>0) orbit an
  attractive well -> hollow ring. Slowing them (v0 0.006->0.002) lets them be pulled INTO the core:
  * DROPLET [batch6 s0 drop_slow]: v0-down gave a COMPACT blob with a coherent RED core (single-colour polar order
    = migration direction, paper's red arrow) + migration tail; chemical = mostly-FILLED compact well. DROPLET SOLVED.
  * VORTEX [batch6 s1 vort_slow]: v0-down (0.005->0.0015, n20000) killed the mill too -- but the population nucleated
    on ~25 wells = ~25 SMALL FILLED rainbow PINWHEELS (each a correct mini rotating disk). Fill works; SCALE fails.
- eps->0 (remove adaptation) = REJECTED as the droplet well [batch6 s2 drop_eps0]. The source ACCUMULATES into a
  SATURATED space-filling FAT-channel network (ctr 1.52 washed out), NOT a compact droplet. Adaptation is NECESSARY
  to localize the well. eps=0.012 (small but nonzero) is right; eps=0 is worse.
- omega is NOT a hollow<->fill selector [REJECTED @ batch5]. omega DOWN dispersed the vortex to streams / left the
  droplet hollow. It only sets rim-lock tightness; the mill fix is v0, not omega.
- decay UP = NOT a fill knob [Rejected @ batch3 s1]. r0/repel UP = loop->STREAM disperser [batch3 s2].
- Paper's VORTEX chemical is a spiral, but excitability (eps UP) does NOT fill -- it SHARPENS RINGS
  [batch6 s3 vort_excite: eps 0.012->0.07 gave crisp hollow rainbow loops + annular front, not a filled disk].

## THE VORTEX: the seed WASHES OUT because refractoriness is MOBILE -> a spiral needs a FIELD (continuum) inhibitor (Established @ batch10)
Batch-9 handed the medium a broken front (seed_spiral one-shot IC); the result is DECISIVE and reframes the whole vortex:
- SEED (b09 s0 vort_seed) and NO-SEED (b09 s3 vort_noseed) are MORPHOLOGICALLY IDENTICAL at the final frame: both relax to a
  handful of SMALL FILLED rainbow PINWHEELS (each a correct mini rotating disk) + sinuous worm-streams; chemical = compact blobs
  + worm channels. The seeded broken front leaves NO trace. Lowering c_th (s2) or diffuse (s1) only changes the NUMBER/size of
  pinwheels, never makes one big spiral.
- ROOT CAUSE (Established @ batch10): the recovery variable `s` (Eq 5) lives on the AGENTS, not the FIELD. The agents carrying
  `s` are chemotactically pulled INTO the very front they emit, so they advect/scramble the refractory tail within a few ticks
  and the broken-front geometry dies before it can wind. The medium is excitable only WHERE AGENTS ARE and its recovery is
  MOBILE, so it cannot pin a phase singularity. => the spiral is NOT a nucleation problem (b09) -- it is a MEDIUM problem: the
  single shared-field + per-agent-s model lacks a space-fixed inhibitor. This also explains b08 (only plane/target waves) and b09
  (seed washes out). The mini-PINWHEEL field is the model's HONEST vortex proxy (fill works [b06]; one-big-disk SCALE fails).
- THE LEVER (under test @ batch10): NEW `refract` op maintains a per-voxel refractory FIELD `fld._rf` (d_t rf = gain*Theta(c-c_th)
  - rf/tau); `relay` with rf_th<1 is blocked where rf>rf_th, so a passed front leaves a space-FIXED wake the next front cannot
  re-invade -> a broken front pins a singularity and winds into a SUSTAINED spiral. rf_tau=refractory period (=core size). Fully
  default-off (rf_th=2.0, refract unscheduled unless rf_tau>0). If the seeded front now winds while the no-rf control stays
  pinwheels, the vortex lever is a CONTINUUM (field) inhibitor. If not, a full FitzHugh-Nagumo activator nonlinearity is needed.

## [SUPERSEDED @ batch10 -- reclassified from "nucleation" to "mobile-refractory MEDIUM" problem] THE VORTEX: excitable medium is BUILT & makes real WAVES, but only PLANE ones -> the spiral is a NUCLEATION problem (Established @ batch9)
The batch-8 c_th>0 + c_base experiment is IN and the mechanism story is now clean:
- c_th>0 + c_base genuinely turns the relay into an EXCITABLE MEDIUM: for the first time the chemical is a set of clean thin
  TRAVELLING WAVE FRONTS (percolating, coherent) -- foam and the fat labyrinth are GONE. This is a real, confirmed advance.
- BUT from random/noisy initial conditions the fronts are always PLANE or TARGET waves, never a curled spiral:
  * UNDER-seeded (c_th high 0.10 / c_base low 0.05) [b08 s0 vort_spiral, s1 th05]: the medium ignites PATCHILY -> disconnected
    HOLLOW loops + fragments (agents still rim-lock onto each 1D front). No percolation.
  * OVER-seeded (c_base 0.08 or diffuse 0.55) [b08 s2 cb08 P0.80 Nc1, s3 dif P0.80 Nc2]: the whole medium ignites nearly at
    once -> ONE coherent percolating PLANE wave (band-like, P~0.8), or long parallel diagonal plane waves. Clean, but straight.
- ROOT CAUSE (textbook excitable-media fact): a SPIRAL does NOT self-nucleate from smooth/noisy ICs -- it requires a BROKEN
  FRONT (a free wave tip = phase singularity) that then winds up. We built the medium but never created the singularity.
  c_th brackets excitability (loops<->plane), c_base brackets ignition FRACTION (patchy<->plane), diffuse sets WAVELENGTH --
  all three confirmed as knobs -- but NONE of them breaks a front. => the vortex is a NUCLEATION problem, not a medium problem.
- THE LEVER (under test @ batch9): NEW `seed_spiral` op stamps a one-shot broken-front IC (half-plane front + refractory tail)
  so the free tip curls into a spiral; agents chemotax onto the rotating arm -> filled rainbow pinwheel disk. seed_spiral=0 =
  old behaviour. If the seeded front curls & agents fill a rotating disk, the vortex is solved (lever = broken-front nucleation
  in the excitable medium). Open risk: seeded spiral may DRIFT/annihilate (periodic pair) -> need core-pinning / single tip.

## [batch8 hypothesis, now CONFIRMED-as-medium @ batch9] THE VORTEX: mechanical-knob program EXHAUSTED -> the vortex is an EXCITABLE SPIRAL WAVE (Reframed @ batch8)
The batch-7 "vortex = rotating droplet, suppress the tail via omega-up" hypothesis is REJECTED, and with it the whole
transport-knob program for the vortex:
- omega-UP off drop_slow (0.30->0.60->1.00): RE-FRAGMENTS the single droplet into MANY small pinwheels (Nc 5->23->32,
  P 0.336->0.178->0.06) [batch7 s0/s1]. omega is a NUCLEATION-COUNT knob at fixed density (more omega -> more, smaller
  clusters), NOT a tail-suppressor. The droplet's one-well compactness at low omega was just WEAK chemotaxis letting
  the population share ONE well. Droplet and vortex are NOT on an omega axis.
- v0-UP (0.002->0.004): consolidates to Nc2 but into thin sinuous FILAMENTS (worm-streams), not a disk [batch7 s2].
- slow decay on the consolidated sigma-well (0.018->0.006): DOES fill 2D, but into a FAT space-filling CHANNEL/
  LABYRINTH (interior stays high-c everywhere the front swept), not a compact disk [batch7 s3]. Fill without
  localization = labyrinth. (Confirms "slow decay keeps interior high" as a real 2D-fill mechanism -- just not compact.)
ROOT CAUSE (Established @ batch8): under the current relay the chemical high-region is ALWAYS either a THIN 1D
travelling front (agents rim-lock onto a 1D curve -> hollow loop/filament) or a FAT 2D labyrinth (slow decay). Neither
is the paper's vortex chemical (Fig 1h,m), which is a SPIRAL WAVE: a 2D rotating structure with a phase-singularity
core that sweeps ALL radii once per period -> agents fill a rotating rainbow DISK. NO transport knob (v0/omega/sigma/
diffuse/decay/eps) makes a spiral, because they don't change the medium from "constant emitter + refractory" to a
true EXCITABLE medium.
THE REAL LEVER (Open @ batch8): the relay was never excitable -- default c_th=-0.001 makes Theta(c-c_th) ALWAYS ON
(c>=0), so relay = constant emission * (1-s). A REAL threshold c_th>0 makes it excitable (quiescent/excited/
refractory); in a noisy excitable medium broken fronts curl into rotating SPIRALS (moving agents supply the
front-breaking heterogeneity). NEW knob `c_base` (am2_ops Relay, am2_job): weak sub-threshold baseline emission
beta*c_base that SEEDS the field so c_th>0 isn't a dead medium; gate=clamp(Theta(c-c_th)+c_base, max=1); c_base=0
(default) = old behaviour exactly. Under test batch8 s0-s3 (c_th 0.05/0.10, c_base 0.05/0.08, diffuse 0.30/0.55).

## [SUPERSEDED @ batch8] THE VORTEX: consolidation & fill are in TENSION -> pivot to "vortex = ROTATING DROPLET" (Reframed @ batch7)
The batch-6 "grow the correlation length so ONE big well dominates" program is REJECTED as the vortex route: it
buys consolidation at the price of RE-HOLLOWING. Batch-7 read of the three consolidators off vort_slow:
- diffuse UP (0.18->0.45): REJECTED as consolidator [batch6 s0]. Nc stayed ~21 -- smoothing softens each maximum but
  does NOT reduce their NUMBER. diffuse is a blur knob, not a merge knob.
- sigma UP (1.3->3.0): consolidates (Nc 25->4) but each big well is a THICK HOLLOW LOOP -- the MILL RETURNS at large
  well size [batch6 s1]. A wide well is big enough that even slow agents rim-lock onto its travelling front.
- v0 UP (0.0015->0.003): consolidates further (Nc->2) but a single big HOLLOW rectangle -- squarely back in the mill
  [batch6 s2].
ROOT CAUSE (Established): FILL and CONSOLIDATION fight. High-c is a thin travelling FRONT (excitable relay ridge), so
agents pack onto a 1D curve. SMALL well -> filled pinwheel; LARGE well -> hollow loop. Correlation length cannot give
a big FILLED disk. => stop chasing "one big well".
THE PIVOT (Open @ batch7): the SOLVED droplet [b06 s3] is already a proto-vortex -- a compact filled head with
AZIMUTHAL rainbow (rotation) + a single-colour MIGRATION TAIL of agents escaping the well. Droplet vs vortex is the
SAME compact filled object; the difference is TRANSLATIONAL polar order (migrates, comma+tail = droplet) vs
ROTATIONAL (recirculates, pinwheel disk = vortex). The tail exists because v0 ejects agents faster than omega turns
them back. HYPOTHESIS: VORTEX = a droplet whose escape-tail is SUPPRESSED so agents recirculate. Lever = omega UP
(bind tighter), NOT correlation length. Under test batch7 s0-s2 (omega dose-response + v0 bracket off drop_slow);
decay-down on the consolidated sigma-well [s3] kept as a bigger-disk hedge (slow decay -> interior stays high-c ->
fill 2D instead of rim-lock).

## Established (parameter regime -> state), all on a periodic world, dt=1
- POLAR BANDS: strong gamma~0.42, WEAK omega~0.06, move_speed~0.007, beta~0.08, decay~0.03.
  Coherent Vicsek travelling band, single colour, nearly empty background. [CONFIRMED @ batch1 s4, P=0.85]
- STREAMS: gamma~0.35 (the KEY change from the batch-1 gas), omega~0.38, beta~0.16, Dc~0.16, decay~0.02,
  n~8000. Directed rivers + sinuous chemical wave-front lanes. [CONFIRMED @ batch2 s0 str_gamma; P=0.13 is
  a misleading GLOBAL-order metric -- local order + morphology are right. Judge streams by morphology, not P.]
- RINGS (whispering-gallery): gamma~0.50 (closes filaments into loops) + omega~0.45 + Dc~0.22 +
  slow decay~0.014 + slow eps~0.035, n~9000. Closed azimuthal loop + annular front. [CONFIRMED @ batch2 s4
  ring_gamma, P=0.36; loops irregular not circular yet -- refinement open.]
- AGGREGATION: gamma=0 (no polar order) + STRONG chemotaxis omega~0.85 -> Keller-Segel collapse.
  [IMPROVED @ batch4 s5: omega 0.4->0.85 raised contrast 0.94->1.45 = clusters pulled into MORE SEPARATED
  compact bright blobs on a darker background. omega UP is the gap-opener; diffuse-DOWN, decay-UP, sigma-UP all
  only reshape the space-filling foam (rejected). Push omega further (batch5 s4) to sharpen isolated clusters.]
- VORTICES: [OPEN -- reframed @ batch10 as a MOBILE-REFRACTORY MEDIUM problem; continuum-inhibitor `refract` under test].
  Transport knobs exhausted (consolidation re-hollows [b6]; omega-up FRAGMENTS [b7]; v0-up->filaments [b7]; slow-decay->LABYRINTH
  [b7]). c_th>0 + c_base makes an excitable medium with clean travelling WAVE FRONTS [b8] but only PLANE/TARGET waves. The
  `seed_spiral` broken-front IC [b9] does NOT persist: SEED (b09 s0) and NO-SEED (b09 s3) relax to the SAME attractor -- small
  FILLED rainbow PINWHEELS + worm-streams. ROOT CAUSE [b10]: refractoriness `s` is carried by MOBILE AGENTS (Eq 5), which are
  pulled into their own front and scramble the tail -> no space-fixed inhibitor -> a spiral core cannot pin. The mini-pinwheel
  field is the model's honest vortex proxy (fill works [b6], SCALE fails). LEVER (test b10): NEW `refract` op = per-voxel
  refractory FIELD (rf_tau=core size) turning the chemical into a continuum FitzHugh-Nagumo medium so a broken front winds into
  a SUSTAINED spiral. eps-up sharpens RINGS not vortices [b6 s3]. omega-down disperses; omega-up fragments; c_th/c_base/diffuse =
  excitability/ignition/wavelength (none breaks a front).
- ACTIVE DROPLETS: [SOLVED @ batch6 s0 drop_slow] gamma0.55, omega0.30, eps0.012, n4500, move_speed 0.002 (v0-down
  kills the mill). Compact blob with a coherent RED core (single-colour = migration direction) + migration tail;
  chemical = mostly-filled compact well. eps->0 REJECTED (saturates to fat channels). Refine: gamma-up / v0 lower
  to close the faint central hole + shorten the tail.

## Cross-cutting levers
- gamma = the master morphology axis (gas->streams->loops) AND the chemical de-foam knob. [Established @ batch2]
- FILL (hollow loop -> filled disk/blob) = move_speed (v0) DOWN [Established @ batch6]: the hollow is a MILL; slow
  agents are pulled INTO the well instead of orbiting. Fills the droplet blob AND the (small) vortex disks. omega
  REJECTED as selector; eps=0 REJECTED (saturates). CONSOLIDATION via correlation length (diffuse/sigma up) is a DEAD
  END for the vortex [batch7]: it fights fill -- a large well RE-HOLLOWS (mill returns) because high-c is a thin
  front. Fill needs a SMALL well; the vortex disk must come from a rotating droplet, not a consolidated big well.
- omega = chemotactic collapse strength AND a nucleation-COUNT knob: UP = MORE, SMALLER clusters (Nc 5->23->32 off
  drop_slow [batch7]) not one tighter blob; DOWN = disperse to streams. Aggregation omega-up: ctr 0.94->1.45->1.8
  across batches 4-5, but Nc rises (fragments) past ~1.0 -- diminishing returns. NOT a droplet->vortex or fill lever.
- EXCITABILITY (c_th, c_base, seed_spiral, refract) = the medium-type levers [c_th/c_base @ b8; seed_spiral @ b9; refract NEW @ b10]:
  c_th=-0.001 (default) = gate always ON = constant emitter -> target/labyrinth fronts. c_th>0 = excitable medium making clean
  travelling WAVE FRONTS. Confirmed roles: c_th = EXCITABILITY (low->plane/crisp loops; high->patchy) ; c_base = IGNITION FRACTION
  (low->patchy loops; high->one plane wave P~0.8) ; diffuse = WAVELENGTH. NONE breaks a front. seed_spiral (broken-front IC) =
  REJECTED as the vortex fix [b10]: the seed WASHES OUT (seed==no-seed at final frame) because recovery is agent-borne and mobile.
  refract (NEW @ b10) = per-voxel refractory FIELD `fld._rf` (d_t rf = gain*Theta(c-c_th) - rf/tau; relay blocked where rf>rf_th):
  gives the medium a SPACE-FIXED inhibitor (continuum FitzHugh-Nagumo) so a broken front's wake pins a singularity -> SUSTAINED
  spiral. rf_tau = refractory period = spiral core size. Default-off (rf_th=2.0, unscheduled unless rf_tau>0). The vortex is a
  MEDIUM problem (missing field inhibitor), NOT a nucleation problem -- the phase singularity had nothing space-fixed to pin to.
- move_speed (v0) = the fill/mill axis: HIGH (0.005+) = mill/hollow ring; LOW (0.002) = filled blob; TOO LOW at high
  density = frozen fragments. Also a disperser at the gas end (batch1 s6). Different role at each regime.
- r0/repel UP = the loop->stream disperser (reverse of condensation).

## Open questions
- VORTEX = CONTINUUM-INHIBITOR EXCITABLE SPIRAL [batch10 primary]: b9 proved the `seed_spiral` broken-front IC WASHES OUT
  (seed==no-seed attractor: small pinwheels + worms) because recovery `s` is agent-borne/mobile. Does the NEW `refract` op
  (per-voxel refractory FIELD, continuum FitzHugh-Nagumo) let a broken front pin a phase singularity and wind into a SUSTAINED
  rotating spiral disk, while the no-rf control stays pinwheels? Does rf_tau set the core size (40 vs 80)? If YES, the vortex
  lever is a SPACE-FIXED inhibitor field (the paper's actual excitable medium). If s0/s1 still give pinwheels / a plane wave /
  a dead field, the additive gate-block is too weak and the FULL FitzHugh-Nagumo activator nonlinearity is required -- and the
  single-scalar-relay model's honest vortex proxy is the mini-pinwheel field.
- Rings: what regularizes the irregular loop into a clean CIRCULAR annulus (slower decay = standing ring)?
- Droplet refine: does gamma-up / lower v0 close the faint central hole and shorten the migration tail?
- Aggregation: omega-up past ~1.0 fragments (Nc up) -- is there a sigma/decay combo that coarsens WITHOUT fragmenting?

## Rejected / dead ends
- SELF-PROPULSION (v0) as the ring-vs-vortex lever. [Rejected @ batch1 s6] Raising v0 0.005->0.011
  on the vortex made it MORE gas-like (P 0.148->0.05), not an open ring. v0 is a disperser, not a
  shape selector. Rings and vortices do NOT share a v0 axis.
- DIFFUSE (Dc) as the chemical-smoothness lever. [Rejected @ batch1 s7] Dc 0.16->0.30 on streams
  left the chemical foamy (just blobbier) and made particles MORE gas-like (P 0.309->0.076). Higher
  Dc without other changes does not produce paper-like wave lanes.
- SIGMA (source width) as the DE-FOAM lever. [Rejected @ batch2 s1] sigma 1.2->2.4 on streams only
  widened the labyrinthine foam CELLS (still foam), particles stayed a gas. De-foam is gamma, not sigma.
- BETA-DOWN as the de-foam lever. [Rejected @ batch2 s2] beta 0.16->0.07 -> less signal -> MORE gas.
- MOVE_SPEED-DOWN as a condensation lever. [Rejected @ batch2 s3] 0.006->0.003 did not condense the gas.
- DECAY-UP as the FILL knob. [Rejected @ batch3 s1] decay 0.018->0.05 on the hollow vortex kept thick HOLLOW
  square loops -- shrinks amplitude uniformly, front stays a thin travelling ridge agents rim-lock onto.
- R0/REPEL-UP as the FILL knob. [Rejected @ batch3 s2] r0 0.011->0.020 did the OPPOSITE -- broke closed loops
  into OPEN streams (P 0.387->0.518). r0 is a disperser (loop->stream), not a filler.
- DECAY-UP as the AGGREGATION gap-opener. [Rejected @ batch3 s6] decay 0.03->0.08 left a space-filling foam
  (ctr 0.94->0.97, unchanged). Neither sigma nor decay carves dark background between clusters.
- OMEGA-DOWN as the FILL selector. [Rejected @ batch5 s0/s2] omega down did the OPPOSITE of fill: vortex
  0.55->0.25 DISPERSED the loop into open worm-streams; droplet 0.8->0.30 stayed hollow. Retires the batch-4
  "omega is the hollow<->fill selector" claim -- the hollow is a MILL, not a rim-lock omega can unlock.
- DENSITY-UP as the FILL knob. [Rejected @ batch5 s1] n 12000->20000 (omega0.55, eps0.012) gave a thicker
  HOLLOW loop, not a filled disk; also failed to reproduce the batch-3 "filled pill" at identical params.
- DIFFUSE-DOWN as the AGGREGATION gap-opener. [Rejected @ batch4 s4] diffuse 0.12->0.05 only made the foam
  cells FINER (ctr 0.94, unchanged); pinning the chemical shrinks cells, does not carve dark gaps. (omega-UP does.)
- EPS->0 as the DROPLET static well. [Rejected @ batch6 s2] Removing adaptation lets the source ACCUMULATE into a
  SATURATED fat-channel network (ctr 1.52), not a compact blob. Adaptation is necessary to localize the well.
- EPS-UP as the VORTEX fill/spiral knob. [Rejected @ batch6 s3] eps 0.012->0.07 sharpens the front into crisp
  HOLLOW rainbow RINGS, not a filled disk. Excitability is a RING sharpener, not a vortex filler.
- DIFFUSE-UP as the VORTEX CONSOLIDATOR. [Rejected @ batch6 s0] diffuse 0.18->0.45 kept Nc~21 -- softens each well
  but merges none. A blur knob, not a merge knob.
- CORRELATION-LENGTH CONSOLIDATION (sigma-up / v0-up) as the VORTEX route. [Rejected @ batch6 s1/s2] sigma 1.3->3.0
  and v0 0.0015->0.003 DO consolidate (Nc 25->4->2) but every consolidated big well RE-HOLLOWS into a thick loop --
  the mill returns at large well size. Fill and consolidation are in tension; "one big well" cannot be a filled disk.
- OMEGA-UP as the VORTEX "suppress the droplet tail" lever. [Rejected @ batch7 s0/s1] omega 0.30->0.60->1.00 off
  drop_slow RE-FRAGMENTED the single droplet into ~23->32 small pinwheels (P 0.336->0.06). omega is a nucleation-COUNT
  knob (more omega -> more, smaller clusters), not a tail-suppressor. Kills the batch-7 "vortex = rotating droplet"
  pivot: droplet and vortex are NOT on an omega axis. The vortex needs an excitable SPIRAL wave, not the droplet.
- SPONTANEOUS SPIRAL from noise in the excitable medium. [Rejected @ batch8] With c_th>0 + c_base and random ICs, the
  medium makes only PLANE/TARGET waves: under-seed (c_th0.10/c_base0.05) = patchy hollow loops [s0/s1]; over-seed
  (c_base0.08 / diffuse0.55) = one coherent plane wave, P~0.8 band-like [s2/s3]. No c_th/c_base/diffuse setting BREAKS a
  front, and a spiral requires a broken front (free tip). => a spiral cannot self-nucleate here; it must be SEEDED
  (new seed_spiral op, batch9). c_th/c_base/diffuse ARE confirmed as excitability/ignition/wavelength knobs, just not
  front-breakers.
- SPIRAL_SEED (one-shot broken-front IC) as the VORTEX fix. [Rejected @ batch10] SEED (b09 s0) and NO-SEED (b09 s3) relax to
  the IDENTICAL final attractor -- small filled rainbow pinwheels + worm-streams; the stamped broken front leaves no trace.
  Cause: recovery `s` is agent-borne and MOBILE (agents pulled into their own front scramble the refractory tail in a few ticks),
  so the geometry dies before winding. A one-shot IC cannot help when the medium can't HOLD a spiral. => the vortex needs a
  space-fixed (continuum) inhibitor FIELD, not a seed. seed_spiral retained only as a tip-initiator FOR the `refract` medium (b10).
- SLOW-DECAY as the VORTEX 2D-fill. [Partial/Rejected @ batch7 s3] decay 0.018->0.006 on the consolidated sigma-well
  DID fill 2D but into a fat space-filling LABYRINTH (fill everywhere the front swept), not a compact rotating disk.
  Confirms slow-decay as a real 2D-fill mechanism; rejects it as a route to the compact vortex DISK.
