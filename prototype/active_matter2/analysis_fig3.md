# analysis log -- Fig. 3 (coarsening + information)

Append one dated section per batch.

## Batch 1 -- 2026-07-02 (first coarsening sweep; design + calibration)

Setup this batch. I am the design step of the loop; the six-core `python` runs are executed by
`run_batch` after this (heavy python was not runnable interactively here), so per-slot Nc_final
numbers land with the montage. What I did read: `paper_fig3.png` and the full solver
(`am2_hydro.py` / `am2_job.py`), and I upgraded the coarsen panel so batch-1 output is judgeable
against Fig.3a (see below).

Paper phenomenology (targets):
- (a) Normalized Nc vs normalized time, log-log, family over v0=0.2..0.7. Shape = short plateau ->
  ~t^-1 (dashed guide) -> a steeper late drop. Higher v0 sits lower / decays sooner.
- (b) v0=0.5 snapshots at t=1200 / 5000 / 20000: many small DROPLETS (rainbow orientation) ->
  elongated STREAMS -> a few large VORTICES; chemical c below each.
- (c) Same count split by morphology: droplet / stream / vortex populations peak in succession.
- (d) Processing rate R(t): noisy, decays as order sets in.
- (f) Per-field information I(t): all fields lose information as mass condenses into few peaks.

Diagnostic upgrade (am2_job.py `_panel_coarsen`): Nc is now plotted vs REAL integration step
(frame_index*rec) on log-log with a `t^-1` guide anchored at the Nc peak, and info/R share the
same step axis. This lets the montage read off the early Ostwald slope and the late steepening
directly against Fig.3a instead of a bare frame-index curve. Also fixed a latent crash: the
info dict seeded a phantom `s_` field that was never filled, so `plot(steps, [])` would have
raised a length mismatch on the first coarsen render (this path had never run -- analysis was
empty); dropped `s_`.

Slots designed (8): six-v0 family v0_02..v0_07 at a COMMON nsteps=48000/N=180/seed=0 (one
variable = v0, so the curves are comparable), plus two endgame extensions v0_05_long and
v0_07_long (parent = v0_05 / v0_07, change ONLY nsteps -> 96000) to push into the few-vortex
lattice and expose the late faster-than-1/t drop + the information plateau.

Predictions to test against the montage:
- Nc(step) should show the plateau -> ~t^-1 -> steeper-drop shape; low v0 stays higher longer.
- v0_05 auto-snapshots (steps ~8k/28k/48k) should read droplet -> stream -> vortex.
- Fields: rho and c should shed the most information (few bright peaks compress hard); px/py may
  lag if orientation stays textured until vortices lock in.
- RISK: at v0=0.2 aggregation may be too slow to leave the plateau by 48k (Nc ~ flat) -- if so,
  bump low-v0 nsteps next batch. RISK: the 'fig' preset is tuned at v0=0.6, so v0=0.7 may
  over-coarsen to a single vortex (Nc->1) well before 96k.

## Batch 2 -- 2026-07-02 (read batch-1 montage; the v0 ordering is INVERTED + R was broken)

Read all 8 batch-1 panels + progress + fig3_b01_montage vs paper_fig3. Per-slot (nsteps/Nc_max ->
Nc_final):
- v0_02 (48k, 770->4)   monotone steep decay to a few big smooth DOMAINS (phase-separation look).
- v0_03 (48k, 764->3)   same; coarsens furthest of the family.
- v0_04 (48k, 763->13)  decay then ARRESTS ~13; structured multi-domain.
- v0_05 (48k, 764->13)  arrests ~13 (the paper's snapshot v0); Nc dips ~700 then flat.
- v0_06 (48k, 763->32)  Nc dips early then RECOVERS and plateaus ~32 -- a stable vortex crystal.
- v0_07 (48k, 767->35)  crisp REGULAR vortex lattice, FROZEN & identical across 8k/28k/48k.
- v0_05_long (96k, 764->14)  doubling steps moved Nc 13->14: NOT slow, ARRESTED.
- v0_07_long (96k, 767->31)  35->31: barely creeps. The lattice is a dynamic steady state.

Nc curve shape: NONE show the paper's plateau->t^-1->faster. Every slot decays FASTER than the
t^-1 guide from frame ~0 (no nucleation plateau -- the random IC fragments into ~765 specks
instantly, Nc_max at frame 0-1, then only merges). High v0 shows dip-then-RECOVER to a plateau
(lattice count re-nucleates); low v0 is monotone to a few.

Cascade verdict: NOT resolved by the batch-1 panels -- the three snapshots sat at steps
8k/28k/48k, all POST-cascade (the droplet stage is over by ~step 2k). Low-v0 late frames look like
coarsening domains, not droplet->stream->vortex; high-v0 frames are a frozen crystal. FIXED for
batch 2: `_panel_coarsen` now log-spaces snapshots (geomspace ~step 800 / 6k / 48k) to span it.

Information I(t): does NOT monotonically decay. It RISES to a peak ~28-30 kB around step ~1-2k
(max droplet count = max spatial entropy) then decays to a PLATEAU ~8-12 kB (arrest => no further
simplification). Field order: c stays HIGHEST (~10-14), rho/px/py settle ~7-9 -- so c retains the
most information, opposite the batch-1 guess that rho/c shed the most.

R(t): BROKEN in batch 1 -- run_hydro passed s=0 and c_th=-1 makes the gate always-on, so
R = beta*<rho> ~ 0.72 (conserved mass) for every slot; the tiny 1e-6 wiggles were numerical, not
physics. FIXED: am2_hydro.run now records s (frame idx 4) and run_hydro feeds the real s into
emission_rate, so R = <rho*beta*(1-s)> will actually track adaptation/decay in batch 2.

BIGGEST SURPRISE: the v0 ordering is INVERTED vs the batch-1 hypothesis (and vs a naive reading of
Fig.3a). Higher v0 does NOT coarsen sooner/lower -- it ARRESTS a vortex lattice at MORE clusters
(Nc 32-35), while LOW v0 coarsens all the way to a few domains (Nc 3-4). Activity STABILIZES a
finite vortex spacing (active crystal) instead of driving mass into one blob. The paper's endpoint
(few large vortices, continued info decay) therefore lives at our LOW-to-MID v0 (~0.3-0.4), NOT at
the preset v0=0.6. Batch 2 refines the coarsen<->arrest transition and probes its mechanism.

## Batch 3 -- 2026-07-02 (read batch-2 montage; transition MAPPED, pinning CONFIRMED, plateau artifact FIXED)

Read all 8 batch-2 panels + progress + fig3_b02_montage vs paper_fig3. Per-slot (v0/nsteps/seed ->
Nc_final, Nc_max, R_final):
- v0_015 (0.15,48k,s0)   Nc 2   (max 773, R 0.265)  coarsens to 2 big domains.
- v0_025 (0.25,48k,s0)   Nc 1   (max 770, R 0.263)  SINGLE VORTEX -- the paper endgame reached.
- v0_030 (0.30,48k,s0)   Nc 3   (max 764, R 0.261)  few large vortices.
- v0_035 (0.35,48k,s0)   Nc 6   (max 762, R 0.256)  partial coarsen, mid regime.
- v0_040 (0.40,48k,s0)   Nc 13  (max 763, R 0.256)  arrested vortex-lattice onset.
- v0_035_long (0.35,96k,s0) Nc 6 (max 762, R 0.257) = the 48k value -> mid v0 also ARRESTED, not slow.
- omega_release (v00.6,omega1.0,48k) Nc 3 (max 763, R 0.258)  vs batch-1 v0=0.6 (omega1.8)=Nc32.
- seed1_035 (0.35,48k,s1) Nc 11 (max 744, R 0.260)  vs seed0=6 -> arrested count is SEED-SENSITIVE.

Transition MAPPED. Nc_final(v0): 0.15->2, 0.25->1, 0.30->3, 0.35->6, 0.40->13 (seed0). A smooth
monotone ramp from the single/few-vortex ENDGAME (v0<=0.30) up into the arrested vortex-crystal
(v0>=0.40, Nc growing 13->32->35 through batch-1's 0.6-0.7). No sharp jump; the coarsen<->arrest
crossover is a gradual ~v0 0.30-0.45 band. v0=0.25 is the cleanest paper reproduction (Nc=1).

Nc curve shape: STILL no plateau->t^-1->faster in the panels. Every slot decays faster than the
t^-1 guide from frame ~0 (low v0 monotone to 1-3; mid/high dip-then-RECOVER to the lattice count).
ROOT CAUSE FOUND + FIXED this batch: Nc_max~765 at frame 0 is a MEASUREMENT ARTIFACT, not physics.
count_clusters (rel=0.55) thresholds the near-uniform IC (rho=1.2+0.05*noise) at ~mean+3%, catching
~29% of noise pixels (below 2D site percolation ~0.59) -> hundreds of speck blobs BEFORE any droplet
nucleates. So Nc starts noise-inflated and only decays -- the paper's nucleation plateau is erased.
FIX: added abs_frac contrast floor to count_clusters; coarsen now calls abs_frac=0.15, which leaves
the IC below threshold (Nc~0) so Nc should RISE from ~0 as real droplets condense -> plateau ->
~t^-1 merge -> faster. Batch-3 panels will test whether Fig.3a's shape finally appears.

Cascade verdict: NOW RESOLVED (log-spaced snapshots @ step ~800/6k/48k). v0_025 and omega_release
both read cleanly droplet(800) -> stream/swirl(6k) -> few-large-vortices(48k). v0_040 & seed1_035
read droplet -> REGULAR vortex lattice(6k) -> partial coarsen -> the arrested-crystal branch.

Information I(t): rises to ~28-30 kB peak at start (max droplet entropy) then decays to a plateau.
Field order confirmed: c (chemical) stays HIGHEST longest in the arrested slots (v0_035/040/seed1);
for the fully-coarsened v0_025 (Nc=1) all fields converge LOW (~7-8) -- continued info decay tracks
continued coarsening, exactly as the paper (info keeps falling only where mass keeps merging).

R(t): FIX from batch 2 works -- R now decays 0.72 -> ~0.26 plateau (was stuck ~0.72). R_final
varies weakly: low v0 slightly higher (0.265 @0.15) than mid (0.256 @0.35-0.40). Less noisy than
the paper's R, but the decay-as-order-sets-in trend is right.

BIGGEST SURPRISE: omega_release. Dropping chemotaxis omega 1.8->1.0 at fixed v0=0.6 collapses
Nc_final 32 -> 3 (~10x) -- decisive confirmation that the high-v0 arrest is CHEMOTACTIC VORTEX-
PINNING: chemotaxis holds vortex cores at a preferred separation; weaken it and the crystal melts
and coarsens to a few vortices. Secondary surprise: v0=0.25 reaches a TRUE single vortex (Nc=1) --
the low-v0 branch genuinely reaches the paper endpoint, while the mid-v0 arrested count is seed-
sensitive (6 vs 11 at v0=0.35), i.e. a metastable defect count set by the IC, not a sharp v0 law.

## Batch 4 -- 2026-07-02 (read batch-3 montage; the abs_frac fix REPRODUCES Fig.3a's Nc(t) shape)

Read all 8 batch-3 panels + progress + fig3_b03_montage vs paper_fig3. Per-slot (v0/omega/nsteps/seed
-> Nc_max, Nc_final, R_final; Nc curve shape; cascade):
- v0_005 (0.05,48k,s0)   max12 fin0  R0.265  RISE->peak~12->hugs t^-1->0. droplet(800)->smooth
                          streams(6k)->near-UNIFORM magenta(48k). Full single-blob endgame (below abs floor).
- v0_010 (0.10,48k,s0)   max15 fin0  R0.265  same shape; fully coarsened to one smooth domain (Nc->0).
- v0_020 (0.20,48k,s0)   max21 fin2  R0.264  clean rise->peak~21->t^-1->faster. droplet->swirl->few
                          large vortices. Textbook Fig.3a.
- v0_030b(0.30,48k,s0)   max29 fin3  R0.261  HIGHEST peak, cleanest t^-1 decade then faster late drop.
                          droplet->vortices->few large vortices. BEST Fig.3a match of the family.
- v0_020_long(0.20,96k,s0) max21 fin4  R0.265  doubling steps did NOT reach Nc=1; holds few vortices.
- omega_06(v0.6,w0.6,48k)  max11 fin3  R0.259  weak chemotaxis -> coarsens to few big vortices.
- omega_14(v0.6,w1.4,48k)  max23 fin7  R0.250  arrests at Nc=7 lattice; info shows EPISODIC re-bumps.
- v0_020_seed1(0.20,48k,s1) max19 fin4  R0.264  vs seed0 fin2 -> low-v0 endgame SEED-ROBUST (both few).

Nc curve shape -- THE WIN: with abs_frac=0.15 EVERY slot now shows Fig.3a's shape: Nc RISES from ~0
(IC below contrast floor) to a PEAK/plateau at Nc_max around step ~1-2k (max droplet count = nucleation
plateau), then DECAYS tracking the t^-1 guide, then a FASTER-than-t^-1 late drop. The three previous
batches' "no plateau, only decay" was PURELY the counting artifact -- now removed, the paper shape is
immediate and universal across the family. Nc_max is now modest+physical (11-29) and RISES with v0 in
the coarsening branch (0.05->12, 0.10->15, 0.20->21, 0.30->29): more activity nucleates MORE droplets.

Cascade verdict: RESOLVED across the family. droplet(800) -> stream/swirl(6k) -> few-large-vortex or
uniform(48k) for v0_020/030b/005; omega_14 reads droplet -> regular lattice -> arrested lattice.

Information I(t): rise to ~30 kB peak (step ~1k) -> decay to ~5-8 kB plateau. Fully-coarsened v0<=0.10
(Nc->0) all fields converge LOW ~5; c stays highest in arrested slots. omega_14 shows discrete info
BUMPS (~15k/20k/40k) = the metastable lattice re-organizing in events. R: 0.72 -> ~0.26 as before.

BIGGEST SURPRISE: three batches of "there is no nucleation plateau" were a metric artifact end-to-end
-- the abs_frac floor exposes rise->peak->t^-1->faster instantly across ALL v0. Secondary: (i) Nc_max
RISES with v0 (activity = more nucleation sites), inverse to the arrested-count logic; (ii) v0<=0.10
coarsens PAST a single vortex into ONE smooth domain (Nc=0, below the contrast floor) -- an endgame
beyond v0=0.25's single vortex; (iii) omega pinning is a THRESHOLD not a ramp: at v0=0.6, Nc_final is
~flat 3 for omega<=1.0, 7 at 1.4, then jumps to 32 at 1.8 -- steep onset near omega~1.6.

## Batch 5 -- 2026-07-02 (read batch-4 montage; the t^-1 decade is set by PHYSICAL BOX L, not grid N)

Read all 8 batch-4 panels + progress + fig3_b04_montage vs paper_fig3. Per-slot (v0/omega/N/nsteps/seed
-> Nc_max, Nc_final, R_final; Nc shape; cascade):
- N_256 (v0.30,N256,48k,s0)    max24 fin6  R0.261  rise->peak24->t^-1->faster; droplet(800)->vortices
                                (6k)->2 big smooth domains(48k). ~1 decade of decay -- SAME as N=180.
- N_320 (v0.30,N320,48k,s0)    max29 fin4  R0.262  rise->peak29->hugs t^-1->drop to 4; droplet->clear
                                vortex cores->one big blob. Info peaks ~90 kB (bigger array=more bytes)
                                then ~20. Decade STILL ~1 -- grid refinement did NOT lengthen it.
- N_256_v020 (v0.20,N256,48k,s0) max21 fin3 R0.264  same shape; Nc_max=21 = the N=180 v0=0.20 value.
- v0_030_long (v0.30,N180,96k,s0) max29 fin3 R0.261  96k == 48k value (fin3) -> v0=0.30 ARRESTS at ~3;
                                doubling steps does NOT reach a single vortex. Endgame = 2-3 vortices.
- omega_16 (v0.6,w1.6,48k,s0)  max31 fin20 R0.293  Nc rises ~31 -> partial drop to 20 lattice. BIG jump
                                from w1.4 (fin7) -> the pinning onset is between omega 1.4 and 1.6.
- omega_20 (v0.6,w2.0,48k,s0)  max41 fin35 R0.335  FROZEN regular lattice at 6k & 48k; Nc plateau ~41->35,
                                NO faster drop. c-info stays highest, info plateaus (arrested). R_final
                                HIGHEST of the whole loop (0.335) -- the packed crystal keeps most emission.
- v0_012 (v0.12,N180,48k,s0)   max15 fin0  R0.265  full smooth-blob endgame (contrast below abs floor).
- v0_030_seed2 (v0.30,s2)      max26 fin2  R0.261  vs seed0 fin3 -> low-v0 endgame SEED-ROBUST (2-3).

Nc curve shape: unchanged from batch 4 -- every coarsening slot shows rise->peak->t^-1->faster; the
arrested omega_16/omega_20 rise to a plateau and stay. THE t^-1 DECADE DID NOT LENGTHEN with grid N.

Cascade verdict: RESOLVED across the family (droplet@800 -> vortices@6k -> few-large/blob@48k). The bigger-
grid slots resolve vortex cores more crisply but show the SAME number of them.

Information I(t): rise-then-decay as before; N_320's peak is ~90 kB purely because the field array is bigger
(more raw bytes to compress), not more physical structure -- info scales with array size, a caveat for
cross-N comparison. omega_20 info plateaus high (arrested); coarsening slots decay to ~10-20.

BIGGEST SURPRISE: the N-scaling hypothesis is REJECTED, and the reason is a solver detail I had missed:
`HY.run` uses dx=L/N with L FIXED at 110. So raising N only shrinks dx (finer grid of the SAME physical
domain) -- droplet count = L/wavelength is set by the physics in PHYSICAL units and is invariant to dx.
Nc_max stayed flat 24-29 from N=180->320 and the t^-1 decade stayed ~1 decade. To reach the paper's ~1.5
decades I must enlarge the PHYSICAL BOX L (more wavelengths = more droplets), scaling N with L to hold dx.
FIX applied: run_hydro now reads --L and passes it to HY.run. Batch 5 = an L-scaling family (dx held ~0.61)
to finally lengthen the decade. Secondary: omega pinning onset bracketed to 1.4<omega<1.6 (Nc 7->20).

## Batch 6 -- 2026-07-02 (read batch-5 montage; PHYSICAL BOX L LENGTHENS THE t^-1 DECADE -- hypothesis CONFIRMED)

Read all 8 batch-5 panels + progress + fig3_b05_montage vs paper_fig3. Per-slot (v0/L/N/nsteps/seed ->
Nc_max, Nc_final, R_final; Nc shape; cascade):
- L_165 (v0.30,L165,N270,48k,s0)  max57  fin3   R0.261  rise->peak57->~1.3-decade t^-1->faster->3 big
                                   smooth domains. droplet(800)->vortices(6k)->2-3 streams/blobs(48k). Info 65->15.
- L_220 (v0.30,L220,N360,48k,s0)  max82  fin7   R0.260  peak82, ~1.4-decade t^-1 then faster to 7. droplet
                                   ->vortices->few big streams. Info peaks ~120 kB -> ~25 plateau.
- L_330 (v0.30,L330,N540,48k,s0)  max213 fin16  R0.256  peak213, LONGEST+CLEANEST t^-1 decade (~1.8 decades)
                                   then faster drop. droplet(800)->clear vortex field(6k)->few large streams/
                                   vortices(48k). Info peaks ~250 kB (huge array) -> ~50. BEST Fig.3a match of the loop.
- L_220_v020 (v0.20,L220,N360,48k,s0) max66 fin4 R0.263  same shape, lower peak than v0.30 (activity=nucleation).
- L_220_long (v0.30,L220,N360,96k,s0) max82 fin6 R0.261  96k ~ 48k (fin 6 vs 7): big box ARRESTS at a few
                                   vortices too; the extra decade of TIME barely lowers Nc. droplet->vortices->few big streams.
- L_220_seed1 (v0.30,L220,N360,48k,s1) max93 fin12 R0.258  vs seed0 fin7 -> big-box final count is SEED-SENSITIVE
                                   again (7 vs 12), unlike the small-box v0=0.30 endgame (seed-robust 2-3).
- omega_15 (v0.6,w1.5,N180,48k,s0) max29 fin16 R0.286  rise->31->partial drop to 16 lattice, then FROZEN crystal
                                   (6k==48k, ~16 regular vortices). c-info stays highest (~10), rho/px/py ~5. Fills pinning bracket.
- v0_013 (v0.13,N180,48k,s0)       max16 fin0  R0.265  full smooth-blob endgame (contrast below abs floor). Confirms edge.

Nc curve shape -- THE HEADLINE WIN: enlarging PHYSICAL box L lengthens the t^-1 decade exactly as predicted.
Nc_max climbs with L: L110->29, L165->57, L220->82, L330->213 (v0=0.30, dx held ~0.61). Fit ~ L^1.5-1.8
(near area L^2, slightly sub-area). The visible t^-1 decade of decay grows from ~1 decade (L=110) to ~1.8
(L=330), so L_330 now spans MORE decades than Fig.3a's ~1.5. The rise->peak->t^-1->faster shape holds at
every L; only the decade length (set by #droplets = #wavelengths across the box) grows. This is the lever
batches 1-4 were missing (grid N did nothing because dx=L/N shrinks with N at fixed L).

Cascade verdict: RESOLVED and SHARPER at big L. droplet(800) -> dense vortex field(6k) -> few large streams/
vortices(48k) across the L-family; the bigger box shows MORE droplets at peak and MORE final vortices (fin
grows with L: 3,7,16 for L 165,220,330) -- the endgame count also scales with box, so big-box coarsening
does NOT reach a single vortex, it arrests at a few-per-area.

Information I(t): rise-then-decay as before; the PEAK magnitude scales with N (array bytes) -- L_330 peaks
~250 kB, L_165 ~65 kB -- so info is comparable only WITHIN a fixed N (the batch-6 family fixes N=360, making
per-field info directly comparable across v0). All four fields decay together; c retains the most in arrested
slots (omega_15: c~10 vs rho/px/py~5). R: 0.72 -> ~0.26 as always; omega_15 higher (0.286, arrested crystal).

BIGGEST SURPRISE: the L hypothesis landed cleanly -- Nc_max ~ L^~1.7 and the t^-1 decade nearly DOUBLED
(L110->L330), so the paper's multi-decade Fig.3a is a FINITE-SIZE / box-size effect, not a physics knob we
had wrong. Secondary surprise: the big-box endgame is NOT a single vortex -- it arrests at a FEW vortices whose
count scales with box area (fin 3/7/16 for L 165/220/330), and that final count becomes SEED-SENSITIVE again
(7 vs 12 at L=220) because more droplets = more metastable merging pathways -- the small-box v0=0.30 seed-
robustness (2-3) was itself a small-number effect. Batch 6 = run the full v0 FAMILY at the large L=220 box
(fixed N=360, one variable=v0) so Fig.3a's family of multi-decade curves renders in a single montage.

## Batch 7 -- 2026-07-02 (read batch-6 montage; the v0 FAMILY rendered -- but at omega=1.8, the PINNING regime)

Read all 8 batch-6 panels + progress + fig3_b06_montage vs paper_fig3. The batch-6 family (L=220, N=360,
48k, s0) at the DEFAULT Fig.3 preset omega -- which is 1.8 (am2_hydro.py:74), i.e. ABOVE the pinning
threshold (~1.5, batch 4-6). So the whole family sat in the vortex-CRYSTAL regime. Per-slot (v0 -> Nc_max,
Nc_final, R_final; shape; cascade):
- fam_v020 (v0.20)  max66  fin4   R0.263  rise->peak66->~1.3-decade t^-1->faster. droplet(800)->dense
                                  vortex field(6k)->FEW LARGE swirls/blob(48k). Info: all fields together 120->25.
- fam_v030 (v0.30)  max82  fin7   R0.260  peak82, long t^-1 -> 7 large vortices. cascade droplet->vortices->few big.
- fam_v035 (v0.35)  max101 fin20  R0.256  CROSSOVER: t^-1 partial then arrests ABOVE the guide at ~20.
                                  cascade shows large streams WITH residual small vortices (mixed endgame).
- fam_v040 (v0.40)  max109 fin33  R0.256  arrests higher (33); lattice+large-domain mix.
- fam_v050 (v0.50)  max136 fin52  R0.261  mostly arrested crystal, ~52 vortices; only shallow late drop.
- fam_v060 (v0.60)  max143 fin87  R0.313  ARRESTED regular vortex crystal (~87); Nc sits FLAT far above t^-1.
- fam_v070 (v0.70)  max150 fin112 R0.372  DENSEST crystal (~112 vortices in a regular lattice, snapshots
                                  6k==48k). c-INFO RISES to ~50 & stays high; rho/px/py ~28. R dips 0.29 then CLIMBS to 0.37.
- fam_v020_long (v0.20,96k) max66 fin6 R0.264  96k ~ 48k (6 vs 4): big-box low-v0 ARRESTS at a few vortices,
                                  extra decade of time does NOT reach one vortex. Confirms batch 6.

Nc curve shape: EVERY slot shows rise->peak->t^-1->faster, and the FAMILY renders in one montage as intended.
But the family ORDERING IS THE INVERSION, now shown cleanly and monotonically: both Nc_max (66,82,101,109,
136,143,150) and Nc_final (4,7,20,33,52,87,112) RISE with v0. The paper (Fig.3a legend, our batch-1 note)
expects HIGHER v0 to sit LOWER / decay sooner -- we get the exact opposite because at omega=1.8 higher
activity feeds STRONGER chemotactic pinning -> a denser, more-arrested vortex crystal.

Cascade verdict: RESOLVED at every v0 (droplet@800 -> vortices@6k -> endgame@48k). The ENDGAME morphology,
though, is v0-graded: low v0 -> few large swirls/blob (true coarsening); high v0 -> a frozen regular lattice
of many small vortices. So the "droplet->stream->vortex" cascade is reproduced, but only LOW v0 continues past
the vortex stage into the few-large-domain coarsening the paper's Fig.3a tail implies.

Information I(t): coarsening slots (v0<=0.50) -- all four fields decay together, ~120 kB peak -> ~25 plateau.
Arrested slots (v0>=0.60) -- c (chemical) RISES and stays HIGH (~50) while rho/px/py settle ~28: the regular
crystal is texturally complex in the chemical field. So which field loses info first is REGIME-dependent: in
coarsening they fall together; in the crystal c does NOT lose info (it re-orders into a rich pattern).
R(step): coarsening flat ~0.26; arrested crystal RISES (v070: 0.29 dip -> 0.37) -- packed vortices keep the
most active emission, matching the batch-5/6 omega-ramp finding.

BIGGEST SURPRISE: the whole batch-6 family lived at omega=1.8 (default), the PINNING regime -- so the clean
inverted ordering is a chemotaxis-pinning artifact, NOT a v0-activity law. The paper's ordering (higher v0 ->
faster coarsening) should reappear if we DROP omega below the pinning threshold, where high v0 can no longer
lock a crystal. Batch 7 = the same v0 FAMILY at LOW omega=1.0 (below the ~1.5 onset). Prediction: the family
DE-inverts -- all v0 coarsen to few vortices, and if activity drives merging, higher v0 now sits LOWER
(paper-like). This is the direct test of the loop's central surprise.

## Batch 8 -- 2026-07-02 (read batch-7 montage; DROPPING omega KILLS THE INVERSION -> the family COLLAPSES)

Read all 8 batch-7 panels + progress + fig3_b07_montage vs paper_fig3. Batch 7 reran the batch-6 v0 family
(L=220, N=360, 48k, s0) but at LOW omega=1.0 -- BELOW the ~1.5 chemotactic-pinning onset. Direct test of the
central surprise. Per-slot (v0 -> Nc_max, Nc_final, R_final; shape; cascade):
- loW_v020 (v0.20,w1.0)  max33 fin2  R0.264  rise->peak33->~1.2-decade t^-1->faster. droplet(800)->streams(6k)
                                  ->FEW LARGE domains (~2 magenta/blue blobs, 48k). Info: 4 fields together 120->22.
- loW_v030 (v0.30,w1.0)  max37 fin6  R0.263  same shape; endgame few large swirls. Parent slot.
- loW_v040 (v0.40,w1.0)  max37 fin7  R0.262  peak flat vs v030; few large domains.
- loW_v050 (v0.50,w1.0)  max41 fin11 R0.261  Nc_final peaks HERE (11) -- top of a weak hump.
- loW_v060 (v0.60,w1.0)  max39 fin10 R0.257  hump descending; still few large swirls, NOT a crystal.
- loW_v070 (v0.70,w1.0)  max39 fin5  R0.247  coarsens to ~5 large streams/vortices -- ARREST GONE. Info decays
                                  together to ~22 (c ~25, marginally highest). R lowest of family (0.247).
- midW_v070 (v0.70,w1.4) max88 fin21 R0.253  PINNING RE-EMERGES: peak88, arrests at ~21 small vortices in a
                                  partial lattice; c-info stays elevated ~30 while rho/px/py ~18 (crystal signature).
- loW_v070_long (v0.70,w1.0,96k) max39 fin6 R0.250  96k ~ 48k (6 vs 5): unpinned high-v0 is arrested at a
                                  FEW-per-area too (big-box area law, batch 6), extra decade of time does nothing.

Nc curve shape: EVERY slot still shows rise->peak->t^-1->faster. The HEADLINE: dropping omega 1.8->1.0
COLLAPSES the family. Nc_max is now nearly FLAT across v0 (33,37,37,41,39,39) vs 66-150 at omega=1.8; Nc_final
is a narrow band (2,6,7,11,10,5) vs 4-112. The clean monotone INVERSION of batch 6/7 is KILLED -- higher v0 no
longer arrests at more clusters. This near-flat Nc_max family is the CLOSEST any batch has come to Fig.3a's
collapsed master curve (the paper normalizes Nc/Nc_max vs t/t_peak; a flat Nc_max means the raw curves already
nearly overlie). CAVEAT: it does NOT cleanly de-invert to the paper's *monotone* ordering (higher v0 -> lower)
either -- Nc_final is a weak non-monotone HUMP peaking at v0=0.50 (11) then falling to 5 at v0=0.70. So below
onset the v0 dependence largely WASHES OUT rather than flipping sign.

Cascade verdict: RESOLVED and now UNIFORM across the family. droplet(800) -> streams/large swirls(6k) -> a FEW
large domains(48k) at EVERY v0, including v0=0.70 -- which at omega=1.8 froze into a 112-vortex crystal but at
omega=1.0 coarsens to ~5 large domains. So "droplet->stream->few large domains" (the full Fig.3a cascade past
the vortex stage) is now universal in the sub-onset regime; the frozen-lattice endgame was purely a pinning
(high-omega) phenomenon.

Information I(t): in the coarsening regime (all 6 low-omega slots) all four fields decay TOGETHER, ~120 kB peak
-> ~22 plateau, c marginally highest late. Only the omega=1.4 bracket shows the arrested-crystal split (c stays
elevated ~30, rho/px/py ~18). R(step): flat ~0.25-0.26 for the whole low-omega family and now WEAKLY DECREASING
with v0 (0.264 @0.20 -> 0.247 @0.70) -- the OPPOSITE of the omega=1.8 crystal (which rose to 0.37). Unpinned
higher-v0 coarsening keeps marginally LESS active emission; arrest (crystal) was what preserved processing.

BIGGEST SURPRISE: the fix worked on the arrest but NOT on the sign. Dropping omega below onset removed the
inversion (Nc_max flat, no crystal at high v0) exactly as predicted -- but instead of flipping to the paper's
higher-v0-coarsens-faster ordering, the family simply COLLAPSED into a v0-independent band. This says the
paper's Fig.3a collapse is the *generic* sub-onset behavior (droplet count set by box wavelength, final count
set by the big-box area law ~5-10, both nearly v0-blind), and the strong v0 ordering in EITHER direction is an
above-onset pinning effect. Also confirmed: pinning re-emerges between omega 1.0 and 1.4 at the big box
(fin 5 -> 21), and low-omega high-v0 is arrested at a few-per-area (96k~48k), not a crystal. Batch 8 = push
DEEPER below onset (omega=0.6) across the v0 family to test whether the residual hump flattens into a truly
clean collapse, extend the family down to v0=0.10 (paper's low end), and bracket the onset with omega=1.2.

## Batch 9 -- 2026-07-02 (read batch-8 montage; omega=0.6 CLEANS the Nc_max collapse but the Nc_final HUMP SURVIVES)

Read all 8 batch-8 panels + progress + fig3_b08_montage vs paper_fig3. Batch 8 pushed DEEPER below the
chemotactic-pinning onset: the whole v0 family at omega=0.6 (vs omega=1.0 in batch 7), fixed L=220, N=360, 48k,
seed0, plus one onset bracket (v0.70, omega=1.2). Per-slot (v0 -> Nc_max, Nc_final, R_final; shape; cascade):
- vlo_v010 (v0.10,w0.6)  max20 fin2  R0.265  rise->peak20->~1-decade t^-1->faster. droplet(800)->streams(6k)
                                  ->one near-SMOOTH BLOB (48k, Nc merges to 2). Info: 4 fields together 120->~22.
- vlo_v020 (v0.20,w0.6)  max23 fin3  R0.2646 same shape; endgame ~3 large domains.
- vlo_v030 (v0.30,w0.6)  max24 fin3  R0.2645 few large swirls. Parent slot for the family.
- vlo_v040 (v0.40,w0.6)  max24 fin7  R0.2638 few large swirls/domains; Nc_max identical to v030 (nucleation is
                                  v0-BLIND at this omega) but Nc_final already climbing.
- vlo_v050 (v0.50,w0.6)  max23 fin10 R0.2597 more residual vortices survive to 48k.
- vlo_v060 (v0.60,w0.6)  max23 fin13 R0.2601 Nc_final PEAKS here (13) -- top of the residual hump.
- vlo_v070 (v0.70,w0.6)  max21 fin10 R0.2583 hump just past peak; ~10 streams/vortices, NOT a crystal. Info
                                  settles slightly higher (~30-35), c marginally top. R lowest of family.
- v070_om12 (v0.70,w1.2) max60 fin22 R0.2497 PINNING RE-EMERGING: Nc_max nearly TRIPLES 21->60 vs w0.6, arrests
                                  at ~22 small vortices (partial lattice, visible speckle at 48k); c-info elevated
                                  (~30) above rho/px/py (~22) -- the crystal signature. R LOWER (0.2497), not yet
                                  the crystal R-rise (that needs omega>=~1.5).

Nc curve shape: EVERY slot still shows rise->peak->t^-1->faster (goal SHAPE reproduced yet again). Two headline
numbers vs the omega-ramp of families:
  Nc_max(v0):  omega1.8 -> 66,82,109,136,143,150 (steep rise); omega1.0 -> 33,37,37,41,39,39 (flat);
               omega0.6 -> 20,23,24,24,23,23,21 (FLATTER and LOWER). Deeper below onset => a CLEANER, lower,
               essentially v0-INDEPENDENT Nc_max (~22 +/-2). This is the direct analogue of Fig.3a's normalized
               master curve: a flat Nc_max means the raw Nc(t) curves already nearly overlie without rescaling.
  Nc_final(v0):omega1.8 -> 4..112 (monotone INVERSION); omega1.0 -> 2,6,7,11,10,5 (hump peak v0=0.50);
               omega0.6 -> 2,3,3,7,10,13,10 (hump peak v0=0.60). The residual v0-ordering did NOT flatten at
               omega=0.6 -- if anything the hump grew (11->13) and shifted UP in v0. So the OPEN QUESTION is
               answered NO: dropping omega deeper cleans Nc_max but leaves Nc_final rising with v0.

THE KEY REALIZATION: Nc_max is v0-flat while Nc_final still rises with v0. Nucleation (droplet count) is v0-blind
at omega=0.6, so the entire residual Nc_final(v0) ordering is an ENDGAME/ARREST effect, not a difference in how
many droplets form. Higher v0 arrests coarsening at MORE final clusters even 3x below the small-box chemotactic
onset. That points to a SECOND arrest lever -- the advective self-transport (-v0*div p, -v0*(p.grad)s) itself --
distinct from chemotactic pinning (which needs omega>~1.5 and blows Nc_max up). Batch 9 must test whether this
residual survives to omega->0 (pure advective) or dies (residual weak chemotaxis).

Cascade verdict: RESOLVED and UNIFORM across the family (droplet->stream->few large domains at EVERY v0). v010
reaches the cleanest single near-blob endgame of the whole loop.

Information I(t): coarsening slots (all v0<=0.60) -- 4 fields decay TOGETHER 120->~22. v070 settles slightly
higher (~30-35) with a few residual vortices; the omega=1.2 bracket shows the crystal split (c ~30 > rho/px/py
~22). R(step): flat 0.25-0.265, WEAKLY DECREASING with v0 (0.265@0.10 -> 0.258@0.70) -- same sub-onset trend as
omega=1.0; the omega=1.2 bracket is LOWER still (0.2497), so R does not rise until the crystal actually locks
(omega>=~1.5).

Onset at the big box: v070 Nc_max 21(w0.6) -> 60(w1.2) -> 88(w1.4, batch8 midW). Pinning is ALREADY active by
omega=1.2 (Nc_max tripled), lower than the small-box ~1.5 onset -- the big box lowers/broadens the onset.

BIGGEST SURPRISE: the residual endgame inversion is DECOUPLED from nucleation. I expected omega=0.6 to erase the
Nc_final hump (as the pinning explanation would predict), but instead Nc_max collapsed to a clean v0-flat ~22
WHILE Nc_final kept rising with v0. This splits the arrest into two independent levers: (i) chemotactic pinning
(omega>~1.2-1.5, inflates Nc_max into a dense crystal) and (ii) an omega-independent advective arrest (higher v0
freezes coarsening at more per-area even at omega=0.6). Batch 9 = the v0 family at omega=0.3 (deepest chemotaxis-
alive), plus omega=0.0 probes (does the residual Nc_final ordering AND the cascade survive with ZERO chemotaxis?),
a 96k long run to test whether low-v0 big-box ever reaches a true single vortex, and one big-box onset bracket
(omega=0.9).

## Batch 10 -- 2026-07-02 (read batch-9 montage; PURE-ADVECTIVE ARREST + CASCADE-WITHOUT-CHEMOTAXIS confirmed)

Read all 8 batch-9 panels + progress + fig3_b09_montage vs paper_fig3. Batch 9 SPLIT the arrest test: run the v0
family at omega=0.3 (deepest chemotaxis-ALIVE), then push to omega=0.0 (chemotaxis OFF), a 96k low-v0 endgame run,
and one big-box onset bracket. Fixed L=220, N=360, 48k, seed0 unless noted. Per slot (v0,omega -> Nc_max, Nc_final,
R_final; shape; cascade):
- w03_v010 (v0.10,w0.3)  max22 fin2  R0.265   rise->peak22->t^-1->faster. droplet(800)->streams(6k)->few/blob.
- w03_v030 (v0.30,w0.3)  max21 fin0  R0.2648  coarsens to a smooth near-uniform blob (Nc drops below floor -> 0).
- w03_v050 (v0.50,w0.3)  max22 fin6  R0.2627  few large swirls + residual vortices. Nc_max identical to v030.
- w03_v070 (v0.70,w0.3)  max24 fin5  R0.2598  ~5 streams/domains; hump near/just past peak.
- w00_v030 (v0.30,w0.0)  max20 fin2  R0.265   ZERO CHEMOTAXIS. Full cascade STILL PRESENT: droplet(800)->
                                  stream(6k)->few large swirls/blob(48k). Nc rise->peak20->t^-1->faster intact.
                                  Info: 4 fields decay TOGETHER 120->~22 (no c re-rise -- no crystal).
- w00_v070 (v0.70,w0.0)  max24 fin6  R0.2611  ZERO CHEMOTAXIS. Cascade present; endgame ~6 large streams (NOT a
                                  crystal). Nc_final 6 > w00_v030's 2 -- the v0-rise SURVIVES omega=0.
- vlo_v010_long (v0.10,w0.6,96k) max20 fin0 R0.2652  DOUBLING TIME reaches Nc=0 smooth blob (vs Nc=2 at 48k). At
                                  the LOWEST v0 the big box DOES merge past few-per-area to a single blob given 2x time.
- w09_v070 (v0.70,w0.9)  max37 fin8  R0.251   onset bracket: Nc_max 21(w0.6)->37(w0.9)->60(w1.2). Pinning already
                                  rising by omega=0.9. R dips (0.251) -- partial lattice not yet R-raising.

Two headline numbers vs the omega ladder of families:
  Nc_max(v0): omega1.8 -> 66..150 (steep); omega1.0 -> 33..41 (flat); omega0.6 -> ~22 (flatter); omega0.3 ->
              22,21,22,24 (still v0-FLAT ~22); omega0.0 -> 20,-,-,24 (STILL ~22). Nucleation is v0-blind AND
              omega-blind all the way to zero chemotaxis. Droplet count = box area / wavelength^2, set purely by
              flocking+pressure physics, independent of both v0 and omega below the pinning onset.
  Nc_final(v0): omega0.6 -> 2,3,3,7,10,13,10 (hump v0=0.60); omega0.3 -> 2,0,6,5 (v0 0.10,0.30,0.50,0.70, rises);
              omega0.0 -> 2(v0.30),6(v0.70) (STILL RISES). The residual v0-ordering SURVIVES to omega=0.

THE ANSWER (batch-9 headline open question): the residual Nc_final v0-rise is PURE ADVECTIVE, NOT chemotactic. It
persists unchanged at omega=0.3 AND omega=0.0. With chemotaxis fully OFF, higher v0 STILL freezes coarsening at
more final clusters (v0.30->2 vs v0.70->6). So the two arrest levers are now DIRECTLY confirmed independent:
  (i) CHEMOTACTIC PINNING -- inflates Nc_max into a dense crystal, needs omega > ~1.0-1.2 (big box); OFF here.
  (ii) ADVECTIVE ARREST -- higher v0 (-v0*div p, -v0*(p.grad)s) freezes the endgame at more clusters, survives to
       omega=0. This is what sets the residual Nc_final(v0) hump.

SECOND ANSWER: the droplet->stream->vortex/domain cascade does NOT require chemotaxis. Both omega=0 panels show the
full cascade. With omega=0 the model reduces to Toner-Tu flocking + pressure (-Q grad rho) + advection, and that
suffices for the entire aggregation cascade. Chemotaxis only controls PINNING (whether the endgame is a frozen
crystal), not the cascade or the nucleation count.

THIRD ANSWER: low-v0 big-box DOES reach a true single blob with time. vlo_v010_long (v0=0.10, 96k) -> Nc=0 vs Nc=2
at 48k. So the big-box "arrests at few-per-area, time does nothing" (batch 6) holds for v0>=~0.30 but NOT at the
lowest v0: there the advective arrest is weak enough that 2x time completes the merge to a smooth blob. The
few-per-area arrest strength is v0-graded, consistent with lever (ii).

Cascade verdict: UNIFORM and chemotaxis-INDEPENDENT (droplet->stream->few large domains at every v0, and at
omega=0). Info I(t): with chemotaxis off, all 4 fields decay together 120->~22-25 kB, NO c re-rise -- confirming the
c-info re-rise is a crystal-ONLY (pinned) signature. R(step): flat ~0.25-0.265, weakly decreasing with v0; the
omega=0.9 bracket DIPS (0.251) -- R only rises once a full crystal locks (omega>=~1.5).

BIGGEST SURPRISE: chemotaxis -- the model's named aggregation driver (rho*omega*grad c) -- is INESSENTIAL to the
Fig.3 cascade. I expected omega=0 to kill or badly distort the droplet->stream->vortex sequence; instead it is
untouched, and even the residual v0-ordering of the endgame survives. The entire Fig.3a phenomenon (rise->peak->
t^-1->faster, plus the v0-graded endgame) is carried by flocking+pressure+advection; chemotaxis is a SEPARATE knob
that only decides whether the arrested state is a dense crystal (Nc_max inflation) above its onset. This cleanly
closes the loop: the paper's coarsening is a generic active-fluid Ostwald cascade, and the strong v0/omega
orderings we chased across batches 1-7 were pinning artifacts layered on top of it.
