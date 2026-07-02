# knowledge ledger -- Fig. 3 (coarsening + information)

CUMULATIVE. Seeded. Base = PRESETS['fig']; --mode coarsen records Nc(t), I(t), R(t).

## Established
- Aggregation is slow: the vortex lattice forms over ~40-48k steps (N~180). Nc peaks early
  (many droplets) then decays as they merge -> few vortices. [established]
- Information proxy = PNG-compressed size per field (rho,px,py,c). As mass condenses into a
  few peaks, the density field simplifies -> its information decreases. [expected, verify]
- Processing rate R ~ integral of the excitable emission rho*beta*(1-s)*Theta -- highest
  during active reorganization, settling as the medium orders. [expected, verify]

## Causal statements (distilled)
- v0 SETS THE COARSEN<->ARREST BALANCE, and the sense is OPPOSITE the naive guess. Measured
  Nc_final over v0=0.2..0.7 @48k: 4,3,13,13,32,35. HIGH v0 does NOT drive mass into one blob --
  it STABILIZES a regular vortex LATTICE (active crystal) at a finite spacing, arresting Nc high
  (~32-35). LOW v0 lets passive-like coarsening run to a FEW big domains (Nc 3-4). Mechanism:
  the same advective self-transport (-v0*div p, -v0*(p.grad)s) that would merge peaks also feeds
  the chemotactic rotation that PINS vortex cores at a preferred separation; above a threshold
  (here v0~0.35-0.4) pinning wins and coarsening freezes. [ESTABLISHED batch 1; refine transition]
- COARSENING IS ARRESTED, NOT MERELY SLOW, at v0>=0.5. Doubling nsteps 48k->96k barely moved Nc
  (v0=0.5: 13->14; v0=0.7: 35->31). The lattice is a dynamic steady state, so "longer nsteps for
  the vortex endgame" does NOT reach a single vortex -- reach it (if at all) by LOWERING v0.
  [established batch 1]
- NO t^-1 PLATEAU/CROSSOVER in our runs. The random IC fragments into ~765 specks within the first
  frames (Nc_max at frame 0-1), so Nc only ever MERGES, decaying FASTER than the t^-1 guide from
  the start -- there is no nucleation plateau to cross out of. High v0 even shows Nc dip-then-
  RECOVER as the lattice re-nucleates to its preferred count. To expose a paper-like plateau one
  would need a smoother/near-uniform IC that nucleates droplets gradually. [established batch 1]
- INFORMATION (PNG-size proxy) is NON-MONOTONE: it RISES to a peak (~28-30 kB) at ~step 1-2k when
  the droplet count (spatial entropy) is maximal, then DECAYS to a plateau as droplets merge, then
  FLATTENS once coarsening arrests. Continued info decay therefore also requires the coarsening
  (low-v0) regime. Field order: c retains the MOST information (~10-14 kB), rho/px/py settle
  ~7-9 -- the excitable chemical stays texturally complex longest. [established; corrects the
  batch-1 guess that rho/c shed the most]
- PROCESSING RATE R = <rho*beta*(1-s)*Theta(c-c_th)>. The batch-1 R was an ARTIFACT: s was passed
  as 0 and c_th=-1 makes the gate always-on, collapsing R to beta*<rho>~0.72 (conserved mass) for
  every slot. FIXED (s now recorded+fed): R decays 0.72 -> ~0.26 plateau as the medium orders.
  R_final varies weakly with v0 (0.265 @0.15 -> 0.256 @0.35-0.40): more coarsened/lower-v0 states
  keep marginally more emission. Trend matches the paper (R falls as order sets in), though our R
  is smoother/less noisy than Fig.3d. [ESTABLISHED batch 3]
- THE COARSEN<->ARREST TRANSITION IS A SMOOTH v0 RAMP, mapped Nc_final(v0)@48k,seed0:
  0.15->2, 0.25->1, 0.30->3, 0.35->6, 0.40->13, then (batch1) 0.6->32, 0.7->35. No sharp jump; a
  gradual crossover band ~v0 0.30-0.45 from the single/few-vortex ENDGAME (v0<=0.30) into the
  arrested vortex-CRYSTAL (v0>=0.40). v0=0.25 reaches a TRUE single vortex (Nc=1) -- the paper's
  endpoint lives in the low-v0 branch. [ESTABLISHED batch 3; supersedes batch-1 "3->13 jump" as
  really a smooth ramp once bracketed finely.]
- MID v0 IS ARRESTED TOO, not merely slow: v0=0.35 gives Nc=6 at both 48k and 96k. So "longer
  nsteps for the endgame" fails for v0>=~0.35 as well -- only LOWERING v0 (<=0.30) reaches few/one
  vortex. [ESTABLISHED batch 3; extends the batch-1 v0>=0.5 arrest finding down to ~0.35]
- CHEMOTACTIC VORTEX-PINNING IS THE ARREST MECHANISM -- now DIRECTLY CONFIRMED. At fixed v0=0.6,
  dropping omega 1.8->1.0 collapses Nc_final 32 -> 3 (~10x). Chemotaxis (rho*omega*grad c) holds
  vortex cores at a preferred separation (an active crystal); weaken it and the lattice melts and
  coarsens to a few vortices. So the arrest is set by the omega/v0 balance, not v0 alone: high v0
  pins ONLY while chemotaxis is strong enough. [ESTABLISHED batch 3]
- THE ARRESTED LATTICE COUNT IS SEED-SENSITIVE at mid v0 (v0=0.35: Nc=6 seed0 vs 11 seed1). The
  crystal is metastable -- its defect/vortex count is set by the initial nucleation pattern, not a
  sharp v0 law. Explains scatter in Nc_final(v0) near the crossover; the ENDGAME (Nc~1 at low v0)
  is expected to be more seed-robust (test batch 3). [ESTABLISHED batch 3]
- THE MISSING NUCLEATION PLATEAU WAS A MEASUREMENT ARTIFACT, now fixed. count_clusters(rel=0.55)
  thresholds the near-uniform IC (rho=1.2+0.05*noise) at ~mean+3%, so ~29% of noise pixels cross
  (below 2D site-percolation 0.59) -> hundreds of speck 'clusters' at frame 0 BEFORE any droplet
  nucleates. Nc thus started noise-inflated (~765) and only decayed -- erasing the paper's plateau.
  FIX: abs_frac contrast floor; coarsen calls count_clusters(abs_frac=0.15), leaving the IC below
  threshold (Nc~0) so Nc RISES as real droplets condense. Expected new shape: rise -> plateau (max
  droplet count) -> ~t^-1 merge -> faster, i.e. Fig.3a. [FIX applied batch 3; verify in batch-3 panels]
- FIG.3a's Nc(t) SHAPE IS NOW REPRODUCED, and the "missing plateau" was 100% a metric artifact. With
  abs_frac=0.15, EVERY slot in the coarsening family shows: Nc rises from ~0 (IC sub-threshold) -> PEAK
  at Nc_max around step ~1-2k (the nucleation plateau = max droplet count) -> decays along the t^-1
  guide -> a FASTER-than-t^-1 late drop. Batches 1-3 saw only decay purely because the near-uniform IC
  was counted as ~765 specks; nothing about the physics changed. [ESTABLISHED batch 4 -- the loop's goal]
- Nc_max (nucleation-plateau height) RISES WITH v0 in the coarsening branch: 0.05->12, 0.10->15,
  0.20->21, 0.30->29 (N=180). More activity nucleates MORE initial droplets. This is INDEPENDENT of the
  arrested-count logic (which also rises with v0 at high v0 via pinning) -- here at LOW v0 the endpoint
  still coarsens to few/zero, only the STARTING droplet count grows. [ESTABLISHED batch 4]
- THERE IS AN ENDGAME BEYOND THE SINGLE VORTEX: v0<=0.10 coarsens into ONE smooth density domain whose
  contrast falls BELOW the abs_frac floor, giving Nc_final=0 (near-uniform magenta field). So the low-v0
  branch order is: v0~0.20-0.25 -> single vortex (Nc~1-2); v0<=0.10 -> fully merged smooth blob (Nc=0).
  The Nc=0 threshold sits between v0=0.10 (->0) and v0=0.15 (->2). [ESTABLISHED batch 4]
- CHEMOTACTIC PINNING IS A THRESHOLD IN omega, NOT A LINEAR RAMP. At fixed v0=0.6, Nc_final(omega):
  0.6->3, 1.0->3, 1.4->7, 1.8->32. Flat (~3, coarsened) for omega<=1.0, then a STEEP onset -- the vortex
  crystal locks in abruptly near omega~1.6. So arrest is a near-transition in the omega/v0 ratio, not a
  gentle knob. [ESTABLISHED batch 4; sharpens the batch-3 pinning finding]
- THE LOW-v0 ENDGAME IS SEED-ROBUST, unlike the mid-v0 arrested count. v0=0.20: seed0->Nc2, seed1->Nc4
  (both few, Nc_max 21 vs 19). Contrast mid-v0 v0=0.35: seed0->6 vs seed1->11. Coarsening to a few large
  vortices erases IC memory; the metastable crystal preserves its nucleation defect count. [ESTABLISHED
  batch 4; answers the batch-3 open question]
- INFORMATION CAN RE-RISE IN ARRESTED LATTICES. The metastable vortex crystal (omega_14) reorganizes in
  DISCRETE events -> the info series shows episodic BUMPS (~15k/20k/40k) on top of the decay, whereas the
  fully-coarsening branch decays smoothly to a low plateau. Info decay tracks NET simplification; a
  reorganizing lattice transiently regains texture. [ESTABLISHED batch 4]
- THE t^-1 DECADE LENGTH IS SET BY PHYSICAL BOX SIZE L, NOT GRID RESOLUTION N. `HY.run` uses dx=L/N with
  L FIXED at 110, so raising N (180->256->320) only shrinks dx -- a finer grid of the SAME physical domain.
  The droplet count = L/(pattern wavelength) is set by the physics in PHYSICAL units, invariant to dx:
  measured Nc_max stayed flat 24-29 across N=180->320 at v0=0.30 (and 21 at v0=0.20 for both N=180,256).
  So the t^-1 decade did NOT lengthen with N. To add droplets / more decades one must enlarge L (more
  wavelengths), scaling N with L to hold dx~0.61 & CFL. [ESTABLISHED batch 5; run_hydro now takes --L].
  CAVEAT: the PNG-info proxy scales with raw array size -- N_320's I(t) peaks ~90 kB vs ~30 kB at N=180
  for the SAME physics, so info magnitudes are NOT comparable across N (only within a fixed N).
- OMEGA PINNING ONSET is between omega 1.4 and 1.6 at v0=0.6. Full ramp Nc_final(omega): 0.6->3, 1.0->3,
  1.4->7, 1.6->20, 1.8->32, 2.0->35 -- a steep jump 7->20 across 1.4->1.6 (threshold ~1.5), then saturating
  ~35 above 1.8. R_final RISES with omega through the transition (0.293@1.6 -> 0.335@2.0): the packed vortex
  crystal keeps the MOST emission of the whole loop (vs ~0.26 for coarsening slots) -- arrest preserves
  active processing. [ESTABLISHED batch 5; sharpens the batch-4 "threshold near 1.6"]
- THE Nc=0 FULL-BLOB ENDGAME extends to v0~0.12: 0.05->0, 0.10->0, 0.12->0, 0.13->0, then 0.15->2. Below
  ~v0 0.13 the medium coarsens past a single vortex into one smooth density domain whose contrast falls below
  the abs_frac floor. v0=0.30 (SMALL box L=110) ARRESTS at 2-3 vortices (96k==48k, seed-robust: seed0->3,
  seed2->2). So the small-box low-v0 branch: v0<=0.13 -> Nc0 blob; v0~0.15-0.30 -> few (1-3) vortices, seed-
  robust. [ESTABLISHED batch 5, edge confirmed batch 6 (v0_013->0)]
- PHYSICAL BOX SIZE L IS THE LEVER FOR THE t^-1 DECADE LENGTH -- CONFIRMED. Holding dx~0.61 (scale N with L),
  Nc_max rises with L: L110->29, L165->57, L220->82, L330->213 at v0=0.30. Fit ~ L^1.5-1.8 (near-area, slightly
  sub-quadratic). The VISIBLE t^-1 decade of the Nc decay grows from ~1 decade (L=110) to ~1.8 (L=330) --
  L_330 now spans MORE decades than Fig.3a's ~1.5. MECHANISM: #droplets = box area / (pattern wavelength)^2,
  set in physical units; a bigger box fits more wavelengths -> more nucleated droplets -> a longer Ostwald
  merge cascade before arrest. The rise->peak->t^-1->faster SHAPE is L-invariant; only the decade length grows.
  So Fig.3a's multi-decade family is a FINITE-SIZE effect, reachable by enlarging L (NOT grid N -- rejected
  batch 5 -- and NOT longer nsteps). [ESTABLISHED batch 6 -- the loop's headline lever]
- THE BIG-BOX ENDGAME IS A FEW VORTICES PER AREA, NOT A SINGLE VORTEX. Nc_final scales with box: L165->3,
  L220->7, L330->16 at v0=0.30. Doubling time (L220 96k) barely moves it (7->6): the big box ARRESTS at a
  vortex count proportional to area, it does not merge to one. So "longer nsteps reaches a single vortex" fails
  in the big box too -- the single/blob endgame is a SMALL-box phenomenon. [ESTABLISHED batch 6]
- THE ARRESTED VORTEX COUNT IS SEED-SENSITIVE ONCE THERE ARE MANY DROPLETS. L=220 v0=0.30: seed0->7 vs
  seed1->12 (Nc_max 82 vs 93). Contrast the SMALL-box v0=0.30 (seed-robust 2-3) -- that robustness was a small-
  number effect. More droplets = more metastable merging pathways = more scatter in the final count, same as the
  mid-v0 crystal. So seed-robustness of the endgame decreases as box (droplet count) grows. [ESTABLISHED batch 6]
- OMEGA PINNING ONSET is a STEEP ramp centred ~1.45. Full Nc_final(omega)@v0.6,N180,48k: 0.6->3, 1.0->3,
  1.4->7, 1.5->16, 1.6->20, 1.8->32, 2.0->35. The steepest segment is 1.4->1.5 (7->16); the crystal is
  essentially locked by omega~1.5-1.6. [ESTABLISHED batch 6; supersedes the batch-5 "1.4<onset<1.6" with the
  midpoint measured]
- INFO PEAK MAGNITUDE SCALES WITH GRID N (array bytes), not physics: L330/N540 peaks ~250 kB vs L165/N270 ~65 kB
  for the SAME v0. Per-field info is comparable ONLY within a fixed N. (The batch-6 v0 family fixes N=360 so its
  I(t) curves ARE directly comparable across v0.) [reaffirmed batch 6]
- THE FULL v0 FAMILY IS NOW MAPPED AT FIXED BIG BOX (L=220,N=360,48k,s0) -- and the ordering is a CLEAN MONOTONE
  INVERSION of the paper. Nc_final(v0): 0.20->4, 0.30->7, 0.35->20, 0.40->33, 0.50->52, 0.60->87, 0.70->112;
  Nc_max(v0): 66,82,101,109,136,143,150. BOTH rise with v0. Paper expects higher v0 to sit LOWER/decay sooner --
  we get the opposite. Every slot still shows the rise->peak->t^-1->faster SHAPE; only the endgame count inverts.
  [ESTABLISHED batch 7]
- **THE INVERSION IS CAUSED BY THE DEFAULT omega=1.8, WHICH IS DEEP IN THE PINNING REGIME (onset ~1.5).** The
  Fig.3 base preset (am2_hydro.py:74) sets omega=1.8, so the batch-1..6 v0 families were ALL run above the
  chemotactic-pinning threshold. There, higher v0 feeds STRONGER pinning (rho*omega*grad c is amplified by the
  extra advective self-transport) -> a denser, more-arrested vortex crystal -> MORE clusters. The inversion is
  therefore an omega-regime artifact, NOT a v0-activity law. To recover the paper's ordering, drop omega below
  ~1.5 so high v0 can no longer lock a crystal. [ESTABLISHED batch 7 -- the mechanism behind the loop's headline surprise]
- THE ENDGAME MORPHOLOGY IS v0-GRADED (the cascade completes only at low v0). Cascade droplet(800)->vortices(6k)
  is reproduced at EVERY v0, but the 48k endgame differs: low v0 (<=0.30) -> a FEW large swirls/blob (true
  coarsening past the vortex stage, matching Fig.3a's tail); high v0 (>=0.60) -> a FROZEN regular lattice of many
  small vortices (arrested at the vortex stage). Mid v0 (0.35-0.50) is a mix of large streams + residual vortices.
  So "droplet->stream->vortex" is universal; "-> few large domains" is a LOW-v0 (unpinned-endgame) phenomenon.
  [ESTABLISHED batch 7]
- WHICH FIELD LOSES INFORMATION FIRST IS REGIME-DEPENDENT. In the COARSENING branch (v0<=0.50) all four fields
  (rho,px,py,c) decay TOGETHER (~120 kB peak -> ~25 plateau). In the ARRESTED CRYSTAL (v0>=0.60) the chemical
  field c does NOT lose info -- it RISES to ~50 kB and stays high while rho/px/py settle ~28, because the regular
  vortex lattice is texturally rich in c. So info decay tracks NET simplification only when coarsening actually
  proceeds; a crystal re-orders c into a complex, high-info pattern. [ESTABLISHED batch 7; refines the batch-1..4
  "c retains the most" -- true in BOTH regimes, but in the crystal c actively GAINS info]
- R(step) IN THE FAMILY: coarsening slots flat ~0.26; arrested crystal RISES with v0 (v0.60->0.313, v0.70->0.372,
  each dipping ~0.29 first then climbing). Packed vortices keep the MOST active emission -- arrest preserves
  processing, consistent with the batch-5/6 omega-ramp R trend. [ESTABLISHED batch 7]
- **DROPPING omega BELOW ONSET KILLS THE INVERSION BUT DOES NOT FLIP THE SIGN -- IT COLLAPSES THE FAMILY.** Rerun
  the batch-6 v0 family (L=220,N=360,48k,s0) at omega=1.0 (below the ~1.5 pinning onset): Nc_max goes NEARLY FLAT
  across v0 (33,37,37,41,39,39 for v0 0.20..0.70) vs 66-150 at omega=1.8, and Nc_final drops to a NARROW band
  (2,6,7,11,10,5) vs 4-112. The clean monotone inversion is gone -- high v0 no longer locks a denser crystal. But
  the family does NOT re-order into the paper's monotone higher-v0->lower-Nc; Nc_final is a WEAK NON-MONOTONE HUMP
  peaking at v0=0.50 then falling. So below onset the v0 dependence largely WASHES OUT. This near-flat-Nc_max family
  is the closest match to Fig.3a's COLLAPSED master curve of the whole loop -- the paper's collapse is the generic
  SUB-ONSET behavior, and strong v0 ordering (either sign) is an ABOVE-ONSET pinning artifact. [ESTABLISHED batch 8]
- THE CASCADE ENDGAME IS UNIFORM (droplet->stream->few large domains at EVERY v0) ONCE BELOW ONSET. At omega=1.0
  even v0=0.70 -- which froze into a 112-vortex crystal at omega=1.8 -- coarsens to ~5 large domains. So the frozen-
  lattice endgame was purely a high-omega (pinning) phenomenon; the full Fig.3a cascade PAST the vortex stage into
  few large domains is universal in the sub-onset regime. [ESTABLISHED batch 8; generalizes the batch-7 "endgame is
  v0-graded" -- it was omega-graded all along]
- UNPINNED HIGH-v0 IS ARRESTED AT A FEW-PER-AREA (big-box area law), NOT A CRYSTAL, AND NOT COARSENING FURTHER.
  loW_v070 96k~48k (fin 6 vs 5): even with weak chemotaxis, the big box freezes at ~5-6 large domains and extra time
  does nothing. So the two arrest mechanisms are DISTINCT: (i) big-box area law -> few-per-area (~5-16, set by L,
  batch 6), operates at ALL omega; (ii) chemotactic pinning -> dense crystal (up to ~112), operates only ABOVE
  onset. Below onset only (i) remains. [ESTABLISHED batch 8]
- PINNING ONSET RE-EMERGES BETWEEN omega 1.0 AND 1.4 AT THE BIG BOX. midW_v070 (v0.70,omega1.4,L220): Nc_final
  jumps 5 -> 21, Nc_max 39 -> 88, and c-info stays elevated (~30 vs rho/px/py ~18: partial-crystal signature). So
  the ~1.5 onset measured at the small box (N180, batch 4-6) holds at L=220 too. [ESTABLISHED batch 8]
- R DECREASES WEAKLY WITH v0 BELOW ONSET (0.264@0.20 -> 0.247@0.70), the OPPOSITE of the above-onset crystal (which
  rose to 0.37). Arrest (crystal) is what PRESERVES active emission; unpinned higher-v0 coarsening keeps marginally
  less. So R's v0-trend flips sign across the pinning onset, mirroring the Nc inversion. [ESTABLISHED batch 8]
- **ARREST HAS TWO INDEPENDENT LEVERS -- CHEMOTACTIC PINNING (sets Nc_max) AND ADVECTIVE ARREST (sets Nc_final),
  DECOUPLED BY THE omega=0.6 FAMILY.** Deeper below onset (omega 1.0->0.6, v0 family L=220 N=360 48k s0): Nc_max
  collapses FLATTER and LOWER (20,23,24,24,23,23,21 for v0 0.10..0.70) -- essentially v0-INDEPENDENT (~22+/-2, the
  cleanest analogue of Fig.3a's normalized master curve: flat Nc_max => raw Nc(t) curves already overlie). BUT
  Nc_final still RISES with v0 (2,3,3,7,10,13,10), a weak hump peaking v0=0.60 -- it did NOT flatten (vs omega=1.0
  peak 11@v0.50; the hump actually grew 11->13 and shifted up in v0). Since Nc_max (nucleation/droplet count) is
  v0-BLIND here while Nc_final rises, the residual v0-ordering is PURELY an endgame/arrest effect, NOT more
  droplets. So higher v0 freezes coarsening at MORE final clusters even 3x below the small-box chemotactic onset --
  via the advective self-transport (-v0*div p, -v0*(p.grad)s) itself, a lever distinct from chemotactic pinning
  (which needs omega>~1.2-1.5 and blows Nc_max UP into a dense crystal). [ESTABLISHED batch 9; splits the batch-7
  "inversion is a pinning artifact" -- pinning explains the Nc_MAX inflation, but a separate advective arrest
  explains the residual Nc_FINAL rise that survives deep sub-onset. Test omega->0 to confirm it is non-chemotactic.]
- BIG-BOX CHEMOTACTIC-PINNING ONSET IS LOWER/BROADER THAN THE SMALL-BOX ~1.5. At L=220, v0=0.70, Nc_max(omega):
  0.6->21, 1.2->60, 1.4->88 -- pinning is ALREADY active by omega=1.2 (Nc_max nearly triples 21->60), well below the
  small-box (N=180) ~1.5 onset. R is NOT yet raised at omega=1.2 (0.2497, still falling) -- the crystal R-rise needs
  omega>=~1.5. So the big box shifts the pinning onset down toward omega~1.0-1.2. [ESTABLISHED batch 9; refines the
  batch-8 "onset re-emerges between 1.0 and 1.4" to a measured 21->60->88 ramp over 0.6->1.2->1.4]
- **THE RESIDUAL Nc_final v0-RISE IS PURE ADVECTIVE, NOT CHEMOTACTIC -- it SURVIVES to omega=0.** Ran the v0 family
  at omega=0.3 (Nc_final 2,0,6,5 for v0 0.10/0.30/0.50/0.70) and omega=0.0 (Nc_final 2@v0.30 vs 6@v0.70). At BOTH,
  Nc_max stays v0-flat ~20-24 while Nc_final still RISES with v0 -- unchanged from omega=0.6. So with chemotaxis
  FULLY OFF, higher v0 still freezes coarsening at more final clusters. This DIRECTLY CONFIRMS the batch-9 two-lever
  split: (i) chemotactic pinning inflates Nc_max into a dense crystal, needs omega>~1.0-1.2; (ii) an omega-INDEPENDENT
  advective arrest (-v0*div p, -v0*(p.grad)s) sets the residual Nc_final(v0) hump and survives to omega=0. The levers
  are orthogonal: (i) acts on nucleation count (Nc_max), (ii) on the endgame count (Nc_final). [ESTABLISHED batch 10 --
  answers the batch-9 headline open question; the residual arrest is non-chemotactic]
- **THE DROPLET->STREAM->VORTEX CASCADE DOES NOT REQUIRE CHEMOTAXIS.** Both omega=0.0 panels (v0=0.30, 0.70) show the
  FULL cascade: droplet(step~800) -> stream(~6k) -> few large swirls/domains(48k), with Nc rise->peak(~20)->t^-1->
  faster intact. At omega=0 the model reduces to Toner-Tu flocking + pressure (-Q grad rho) + advection, and that
  ALONE reproduces the entire Fig.3 aggregation cascade AND the v0-blind nucleation count. Chemotaxis (rho*omega*grad c
  -- the model's NAMED aggregation driver) is INESSENTIAL to the cascade; it only decides, above its onset, whether the
  arrested state is a dense crystal. [ESTABLISHED batch 10 -- the loop's closing surprise: the paper's coarsening is a
  generic active-fluid Ostwald cascade, chemotaxis is a separate pinning knob layered on top]
- NUCLEATION (Nc_max) IS v0-BLIND AND omega-BLIND BELOW THE PINNING ONSET. Nc_max ~22 (+/-3) across the ENTIRE v0
  range at omega=1.0, 0.6, 0.3 AND 0.0. The droplet count = box area / (pattern wavelength)^2 is set purely by the
  flocking+pressure length scale, independent of both activity v0 and chemotaxis omega until pinning (omega>~1.2, big
  box) inflates it into a crystal. This flat Nc_max is why the sub-onset raw Nc(t) curves already nearly overlie
  (Fig.3a's normalized master curve needs no rescaling). [ESTABLISHED batch 10; unifies the batch-8/9 flat-Nc_max obs
  down to omega=0]
- INFO c-RE-RISE IS A CRYSTAL-ONLY (PINNED) SIGNATURE. With chemotaxis OFF (omega=0) all four fields (rho,px,py,c)
  decay TOGETHER 120->~22-25 kB -- NO c re-rise. The elevated/rising c-info seen at omega>=1.2 is exclusively the
  regular vortex lattice re-ordering c into a texturally rich pattern. So "c retains/gains the most info" is true only
  in the pinned regime; in the generic coarsening (any omega below onset, incl. 0) c simplifies with the rest.
  [ESTABLISHED batch 10; sharpens the batch-7 regime-dependence to: c-gain requires the crystal]
- LOW-v0 BIG-BOX DOES REACH A TRUE SINGLE BLOB WITH TIME -- but only at the lowest v0. vlo_v010_long (v0=0.10, L=220,
  96k) -> Nc=0 (smooth blob) vs Nc=2 at 48k. So the batch-6 "big box arrests at few-per-area, extra time does nothing"
  holds for v0>=~0.30 but FAILS at v0=0.10: there advective arrest (lever ii) is weak enough that 2x time completes the
  merge. The few-per-area arrest strength is v0-GRADED, consistent with the advective-arrest lever. [ESTABLISHED batch
  10; refines the batch-6 big-box endgame -- it's a few-per-area only when advective arrest is strong (higher v0)]
- BIG-BOX PINNING ONSET IS A GRADUAL RAMP FROM omega~0.9. Nc_max(omega)@v0.70,L220: 0.6->21, 0.9->37, 1.2->60,
  1.4->88. Already rising by omega=0.9 (21->37), broader/lower than the small-box (N180) ~1.5 threshold. R DIPS through
  the partial-lattice regime (0.258@0.6 -> 0.251@0.9 -> 0.2497@1.2) and only RISES once a full crystal locks
  (omega>=~1.5). [ESTABLISHED batch 10; completes the big-box onset ramp with the omega=0.9 midpoint]

## Open questions
- ANSWERED batch 10: the residual Nc_final v0-rise is PURE ADVECTIVE -- it survives unchanged at omega=0.3 AND
  omega=0.0 (Nc_final still 2@v0.30 vs 6@v0.70 with chemotaxis OFF), while Nc_max stays v0-flat ~22. Confirms the
  two-lever split: chemotactic pinning sets Nc_max (needs omega>~1.0-1.2), advective arrest sets Nc_final (survives
  to omega=0).
- ANSWERED batch 10: the droplet->stream->vortex cascade SURVIVES omega=0 -- both omega=0 panels show the full
  cascade and v0-blind nucleation. Flocking+pressure+advection alone reproduce Fig.3; chemotaxis is only a pinning
  knob. The paper's coarsening is a generic active-fluid Ostwald cascade.
- ANSWERED batch 10: low-v0 big-box DOES reach a true single blob with time, but only at the lowest v0 -- vlo_v010
  (96k) -> Nc=0 vs Nc=2 at 48k. At v0>=0.30 the few-per-area arrest holds (96k~48k, batch 6). Arrest strength is
  v0-graded (advective lever).
- Do the paper's THREE morphology sub-populations (droplet/stream/vortex peaking in succession, Fig.3c) show up if
  we classify aggregates by shape? -> would need a shape diagnostic (never built across the 10-batch loop; future).
- Would a cross-slot NORMALIZED overlay (Nc/Nc_max vs t/t_peak, all v0 on one axis) render Fig.3a's collapsed
  family directly? -> the sub-onset flat-Nc_max families (batch 8-10) are the right input; the collapse is now
  established numerically (flat Nc_max => raw curves overlie) but never drawn as a single overlay montage (future).
- ANSWERED batch 9: omega=0.6 does NOT flatten the Nc_final hump -- it cleans the Nc_MAX collapse (v0-flat ~22) but
  Nc_final keeps rising with v0 (hump peak 13@v0.60). Arrest splits into TWO levers: chemotactic pinning (sets
  Nc_max, needs omega>~1.2) and advective arrest (sets Nc_final, survives sub-onset). Big-box pinning onset is lower
  (Nc_max 21->60 over omega 0.6->1.2). v0=0.10 at the big box coarsens to a near-smooth blob (Nc merges to 2).
- ANSWERED batch 8: dropping omega to 1.0 KILLS the inversion (Nc_max flat 33-41, no crystal at high v0) but does
  NOT flip to the paper's monotone ordering -- the family COLLAPSES into a v0-independent band (Nc_final hump 2-11).
  The near-flat Nc_max is the closest match to Fig.3a's collapse of the loop. Pinning re-emerges between omega
  1.0 and 1.4 at the big box (fin 5->21). Low-omega high-v0 is arrested at a few-per-area (96k~48k), not a crystal.
- ANSWERED batch 6: PHYSICAL L lengthens the t^-1 decade (Nc_max~L^~1.7, decade ~1->~1.8 for L 110->330); the
  big-L endgame ARRESTS at a few vortices per area (not one); the big-L count is seed-SENSITIVE (7 vs 12).
- ANSWERED batch 7: the v0 family DOES render at L=220 in one montage; the ordering is a clean monotone INVERSION
  of the paper (Nc_final 4->112 for v0 0.20->0.70), CAUSED by the default omega=1.8 being in the pinning regime;
  the big-box low-v0 (0.20) endgame arrests (96k~48k: 6 vs 4), confirming batch 6.
- Do the paper's THREE morphology sub-populations (droplet/stream/vortex peaking in succession,
  Fig.3c) show up if we classify aggregates by shape? -> would need a shape diagnostic (future).
- Would a cross-slot NORMALIZED overlay (Nc/Nc_max vs t/t_peak, all v0 on one axis) render Fig.3a's
  collapsed family directly? -> the batch-6 fixed-L family is the right input for such an overlay (future montage post-proc).

## Rejected
- "Higher v0 => Nc decays sooner and sits lower." REJECTED, then REFINED: above the pinning onset (omega>=1.5)
  the ordering is INVERTED (higher v0 -> more clusters, denser crystal) [batch 1-7]; below onset (omega<=1.0) the
  v0 ordering WASHES OUT -- Nc_max goes flat and Nc_final collapses to a narrow non-monotone band, it neither
  inverts nor flips to the paper's monotone descent. So the paper's higher-v0-coarsens-faster is NOT reproduced in
  either regime; what IS reproduced sub-onset is Fig.3a's family COLLAPSE. [batch 8]
- "Longer nsteps reaches the few-vortex / single-vortex endgame." REJECTED at v0>=~0.35: coarsening
  is arrested; 96k ~ 48k (v0=0.35: 6->6; v0=0.5: 13->14; v0=0.7: 35->31). Lower v0 (<=0.30) instead.
  [batch 1+3 data]
- "The coarsen<->arrest transition is a sharp jump (Nc 3->13 at v0 0.3->0.4)." REFINED->REJECTED:
  fine bracketing shows a SMOOTH monotone ramp 0.15->2, 0.25->1, 0.30->3, 0.35->6, 0.40->13.
  [batch 3 data]
- "Nc_max~765 is real (the random IC fragments into ~765 droplets)." REJECTED: it is an IC-noise
  counting artifact (near-uniform field thresholded below percolation), not droplets. Fixed with
  abs_frac. [batch 3]
- "rho and c lose information first/most." REJECTED: c retains the most; info is non-monotone
  (rises then decays), not a clean monotone fall. [batch 1 data]
- "The model cannot reproduce Fig.3a's plateau->t^-1->faster shape / it only ever merges." REJECTED:
  it was the count_clusters IC-noise artifact all along. abs_frac=0.15 exposes rise->peak->t^-1->faster
  across the whole v0 family. [batch 4 data]
- "omega scales arrest as a smooth monotone ramp." REFINED->REJECTED: it is a THRESHOLD -- flat ~3 for
  omega<=1.0, then a steep jump (7@1.4, 20@1.6, 32@1.8, sat ~35@2.0). Onset bracketed to 1.4-1.6. [batch 4+5]
- "Grid resolution N extends the t^-1 decade / Nc_max scales as (N/180)^2." REJECTED: dx=L/N with L fixed,
  so bigger N is only a finer grid of the SAME domain -- Nc_max flat 24-29 across N=180-320, decade unchanged.
  The relevant knob is PHYSICAL L, not grid N. [batch 5 data] -- CONFIRMED by batch 6: enlarging L (dx held)
  DID scale Nc_max (29->213 for L 110->330) and lengthen the decade, exactly what N could not do.
- "The big-box coarsening endgame reaches a single vortex given enough time." REJECTED: it arrests at a few
  vortices proportional to area (fin 3/7/16 for L 165/220/330); 96k~48k at L=220. Single/blob endgame is
  small-box only. [batch 6 data]
- "The v0=0.30 endgame is seed-robust." REFINED: robust only in the SMALL box (2-3, small-number effect);
  at L=220 it scatters 7 vs 12 with seed. [batch 6 data]
