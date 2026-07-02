# analysis log -- Fig. 2 (hydrodynamic phase diagram)

Append one dated section per batch.

## Batch 1 (2026-07-02) -- lay down all six states + two boundary probes
QUESTION: no montage exists yet. Place ALL SIX hydrodynamic states across (v0,omega) with
the fig anchor (v0 0.6, omega 1.8 = vortex, established) as parent, to produce the FIRST
snapshot montage vs paper_fig2.png, and spend 2 slots on boundary probes. Deliverable is the
(v0,omega)->state map, not a loss.

TARGET MORPHOLOGY read from paper_fig2.png (top of each panel = polarization orientation, HSV
hue = p-angle, brightness = rho; bottom = chemical c). Panel (v0,omega) are the PAPER's axis
(aggregation near omega~0.05); OUR omega axis is rescaled ~20-40x (aggregation near omega~1):
- a active droplets  : MANY small compact motile blobs on near-empty background; each blob a
  single/low-spread hue (coherently polarised), NOT a pinwheel; c = scattered compact wells.
  Paper: low v0, high omega (top-left of g).
- b vortices         : fewer, LARGER rotating disks; orientation = full rainbow pinwheel round
  a core (all hues present azimuthally); c = spiral wave from the core. FILLED centre.
  Paper: low-mid v0, high omega.
- c rings            : OPEN annuli (hollow centre); orientation = colour wheel round the loop,
  dark middle; c = annular wave front. Paper: mid-high v0, high-mid omega.
- d streams          : long directed rivers of aligned p flowing to wave sources; ~one dominant
  hue per lane; c = elongated travelling-wave lanes. Paper: mid v0, mid-low omega.
- e silent polar bands: straight travelling stripe(s) of aligned p, single hue, on empty bg;
  c NEARLY DARK (little signalling). Paper: high v0, LOW omega (bottom-right of g).
- f polar bands w/ signalling: same banded polar order but c ACTIVE (bright travelling bands
  co-moving with the density stripes). Paper: high v0, high omega (top-right of g).
Panel g (phase diagram): x=Motility v0, y=Signal susceptibility omega. Sweeping v0 left->right
at HIGH omega: droplets -> vortices -> rings -> polar bands. Middle band: streams. LOW omega
(bottom): "no pattern" (left) and silent polar bands (right).

SLOTS (all --kind hydro --mode snapshot; deviations from the fig anchor v0=0.6 omega=1.8):
- s0 vortex       : anchor (v0 0.6, omega 1.8). Re-confirms the established vortex lattice at
  N=180 / 40k steps. Expected paper panel b.
- s1 droplets     : parent vortex, v0 0.6->0.4 (ONE var). Lower motility -> weaker advective
  break-up/rotation -> compact, numerous, non-rotating condensates. Expected paper a.
- s2 rings        : parent vortex, v0 0.6->0.9 (ONE var). Higher motility hollows/opens the
  filled vortex disk into an annulus. Expected paper c.
- s3 bands_sig    : parent rings, v0 0.9->1.0 (ONE var). Push past rings into alignment-
  dominated travelling stripes, omega still high -> c stays active. Expected paper f.
- s4 streams      : parent vortex, omega 1.8->0.5 (ONE var). Below aggregation onset -> no
  condensation, polar order channels into directed rivers. Expected paper d.
- s5 silent_bands : parent bands_sig, omega 1.0... -> omega 0.4 (ONE var, at v0 1.0). High v0,
  low omega -> banded polar order with a NEARLY DARK c field. Expected paper e. (Pairs with s3
  as the silent-vs-signalling bands contrast at fixed v0=1.0.)
- s6 probe_vr     : parent vortex, v0 0.6->0.78 (ONE var). BOUNDARY probe: where does the
  filled vortex open into a ring? brackets s0(0.6) and s2(0.9).
- s7 probe_w      : parent vortex, omega 1.8->1.1 (ONE var). BOUNDARY probe: aggregation onset
  -- does condensation survive near our omega~1 threshold or dissolve toward streams?

ANTICIPATED (to be judged against panels/montage next batch):
- SURPRISE to watch: rings vs vortices may be a single v0 knob apart. If s2/s6 open s0's disk
  into an annulus, the rings-vs-vortex lever is pure motility, collapsing two regimes onto the
  v0 axis at fixed high omega. probe_vr localises that boundary.
- RISK / possible FAILures: high-v0 slots (s3, s5 at v0=1.0) with dt=0.02 and strong chemotaxis
  (omega 1.8 in s3) may blow up -> snapshot could show only the initial noise = FAIL. If so,
  next batch lowers v0 toward 0.9 or adds --sigma 0.9 (stronger alignment reaches bands at
  lower v0, cf. PRESETS['bands']). Levers if a state is misplaced: droplets-too-few -> v0 down
  further / omega up; rings-still-filled -> v0 up; streams-still-aggregating -> omega down;
  bands-not-forming -> v0 up or sigma up.
- A slot with no panel.png FAILED -> redesign around it next batch.
VERDICTS: pending (first montage renders after this batch runs).

## Batch 2 (2026-07-02) -- READING batch-1 montage: the plane is BINARY, not a 6-state sequence
QUESTION: did the batch-1 (v0,omega) grid lay down all six paper states? Verdict per slot below,
then redesign to unlock the MISSING intermediates (droplets/rings/streams/discrete bands).

PER-SLOT READ (all N=180; snapshot = final frame; P = |<p>|/<|p|>, so P~0 = locally organised
net-zero = VORTEX LATTICE, P~0.9 = globally aligned = HOMOGENEOUS POLAR):
- s0 vortex   v0 0.60 w 1.80 : P=0.0  Nc=32 ctr=2.29 sig=0.663. VORTEX/ASTER LATTICE -- ~32 small
  pinwheels (full HSV wheel round each core) over round c wells. Paper panel b. VERDICT: correct
  STATE (topology + spiral c match), but ours is a dense lattice of MANY SMALL vortices; paper b
  shows FEWER, LARGER disks. Best-matched state of the batch.
- s1 droplets v0 0.40 w 1.80 : P=0.062 Nc=13 ctr=0.73 sig=0.611. NOT droplets -- diffuse, UNDER-
  condensed coarsening patches (a couple of pinwheels in corners). VERDICT: MISPLACED. SURPRISE:
  lowering v0 REDUCED condensation (ctr 2.29->0.73). In our model v0 AIDS aggregation (via -v0 div p
  density transport), so droplets are NOT at low v0 as in the paper -- opposite lever sign.
- s2 rings    v0 0.90 w 1.80 : P=0.001 Nc=42 ctr=1.84 sig=1.226. NOT rings -- an even DENSER vortex
  lattice (smaller blobs, more of them). No hollow annuli. VERDICT: MISSING (state absent).
- s3 bands_sig v0 1.00 w 1.80: P=0.002 Nc=45 ctr=1.66 sig=1.442. NOT bands -- still a vortex lattice
  (densest yet). Chemotaxis wins even at v0=1.0. VERDICT: MISSING. SURPRISE: high v0 at high omega
  did NOT break into travelling bands; advection never overcame chemotaxis at w=1.8.
- s4 streams  v0 0.60 w 0.50 : P=0.935 Nc=5 ctr=0.27 sig=0.619. NOT directed rivers -- a HOMOGENEOUS
  POLAR sheet (few huge aligned domains), no condensation. VERDICT: MISPLACED (this is the polar
  phase, not channelised streams).
- s5 silent_bands v0 1.00 w 0.40: P=0.899 Nc=4 ctr=0.36 sig=0.6. Homogeneous polar, c moderate
  (0.6, not dark), NOT discrete stripes. VERDICT: PARTIAL -- right phase (polar, low signal) but the
  order is a continuous sheet, not finite-width bands.
- s6 probe_vr v0 0.78 w 1.80 : P=0.001 Nc=38 ctr=2.24 sig=0.846. Vortex lattice, between s0 and s2.
  VERDICT: confirms v0 0.6->0.9 at w=1.8 is ALL vortex lattice; v0 just sets blob count/size
  (Nc 32->38->42->45 as v0 0.60->0.78->0.90->1.00). The hypothesised vortex->ring->band v0-axis is
  FALSIFIED: v0 does not change the topology inside the condensed phase.
- s7 probe_w  v0 0.60 w 1.10 : P=0.948 Nc=4 ctr=0.31 sig=0.615. HOMOGENEOUS POLAR, NOT condensed.
  VERDICT: aggregation onset is HIGHER than the ledger's "omega~1" -- it sits between w=1.1 and 1.8.

BIGGEST SURPRISE + LEVER: the (v0,omega) plane at these coefficients is essentially BINARY --
CONDENSED VORTEX LATTICE for omega >~ 1.5 (P~0, Nc 13-45, ctr 0.7-2.3) vs HOMOGENEOUS POLAR for
omega <~ 1.1 (P~0.9, Nc~4, ctr~0.3) -- with a SHARP omega threshold (~1.5) that is nearly v0-
independent over v0 0.6-1.0. The four intermediate paper states (droplets/rings/streams/discrete
bands) DO NOT appear because the transition is a jump, not a graded sequence. WHY: chemotaxis
rho*omega*grad(c) acting on the excitable/spiral c field mints +1 topological defects (vortices/
asters) as soon as it beats the polar-growth term sigma*(rho-1)*p; below threshold the polar term
wins outright. The LEVER for intermediates is therefore NOT (v0,omega) alone -- it is the c-field
texture (Dc, alpha, chi) and the polar-onset margin (sigma). Batch-2 slots test those:
  - rings/droplets need the condensate to hollow (broad annular c waves: raise Dc) or compact into
    coherent single-hue blobs (kill the (p.grad)p curl: drop chi; sharpen c wells: raise alpha);
  - discrete bands need to sit NEAR the polar onset (rho~rho_c) -> drop sigma toward the paper's 0.02
    so ordered density stripes coexist with a disordered background, and raise chi to channelise;
  - streams should live IN the sharp boundary layer -> fine omega sweep (1.4, 1.6) at v0=0.6.

## Batch 3 (2026-07-02) -- READING batch-2 montage: c-texture levers only RESIZE vortices;
## bands need NEAR-ONSET density, not low sigma
QUESTION: did reshaping the c-texture (chi/alpha/Dc) and the polar margin (sigma/chi) unlock the
four missing intermediates (droplets/rings/discrete bands)? Verdict per slot, then a density-based
fix for bands + a chi->0 push for droplets.

PER-SLOT READ (all N=180, snapshot=final frame; parents in [brackets]):
- s0 w_16    v0 0.60 w 1.60 [anchor,w down]      : P=0.0  Nc=20 ctr=2.5  sig=0.599. VORTEX LATTICE,
  FEWER blobs than w1.8 (Nc 20 vs 32). Still full pinwheels + round c wells. State = vortex; omega
  1.6 is inside the condensed phase but near its floor. VERDICT: correct state, resolves the onset
  downward (20 vortices still condense at w1.6).
- s1 w_14    v0 0.60 w 1.40 [w_16, w down]       : P=0.178 Nc=3 ctr=0.76 sig=0.607. NEARLY DISSOLVED
  -- large smooth polar domains with ONE surviving vortex defect (single pinwheel + single c well).
  State = polar/onset. VERDICT: pins the vortex<->polar boundary SHARPLY between w1.4 (polar+1 relic)
  and w1.6 (20 vortices). Onset ~1.5, confirmed abrupt (Nc 3->20 over dw=0.2).
- s2 drop_chi v0 0.60 w 1.80 chi 0.10 [anchor,chi down] : P=0.024 Nc=31 ctr=2.14 sig=0.657. Blobs
  became ELONGATED single-hue COMMAS (a tail of one dominant hue, not a full pinwheel) over
  elliptical c wells. State = proto-droplet/streak. VERDICT: killing the (p.grad)p curl DID reduce
  the winding -> coherent-ish motile streaks, the CLOSEST thing to droplets yet, but still 31 blobs
  (not compact round droplets). SURPRISE: chi is the pinwheel<->coherent-blob knob, as hypothesised.
- s3 ring_alpha v0 0.60 w 1.80 alpha 0.85 [anchor,alpha up] : P=0.0 Nc=27 ctr=2.26 sig=0.484. Still
  FILLED pinwheel vortices, FEWER/LARGER (27), lower c (alpha decays c faster -> sig 0.484). NOT
  rings. VERDICT: alpha only shrinks c amplitude + count; no hollowing.
- s4 ring_Dc v0 0.60 w 1.80 Dc 2.20 [anchor,Dc up]  : P=0.001 Nc=17 ctr=2.69 sig=0.572. FEWER, LARGER
  filled pinwheels (Nc 17, ctr 2.69 = most condensed of batch) over BROAD round c wells. NOT rings
  (no hollow centre). VERDICT: high Dc = fewer/larger vortices = the BEST paper-b (few large disks)
  match so far, but topology still aster, not ring.
- s5 vortex_Dp v0 0.60 w 1.80 Dp 1.20 [anchor,Dp up] : P=0.146 Nc=8 ctr=0.74 sig=0.594. UNDER-
  condensed -- only 8 diffuse blobs, low contrast, P rising. VERDICT: raising Dp smears p, weakens
  condensation (opposite of the "fewer/larger clean vortices" goal). Dp is a de-condensation lever.
- s6 band_sigma v0 1.00 w 0.40 sigma 0.20 [silent_bands,sigma down] : P=0.99 Nc=2 ctr=0.05 sig=0.631.
  A COMPLETELY HOMOGENEOUS polar sheet (contrast 0.05 = flat density, one hue). NOT bands. VERDICT:
  the key batch-2 hypothesis is FALSIFIED -- dropping sigma toward the paper's 0.02 made the flock
  MORE uniform, not banded. WHY: a uniform polar sheet has div(p)~0, so -v0 div(p) never structures
  density; low sigma just weakens order without seeding a density instability.
- s7 band_chi v0 1.00 w 0.40 chi 0.90 [silent_bands,chi up] : P=0.947 Nc=3 ctr=0.3 sig=0.608.
  Homogeneous polar sheet again, chi did not channelise into finite bands. VERDICT: MISSING.

BIGGEST SURPRISE + LEVER: NONE of the c-texture / polar-margin knobs changed the TOPOLOGY -- above
onset everything stays a vortex lattice (chi/alpha/Dc only retune blob count/size/coherence: Dc up &
alpha up -> fewer/larger; Dp up -> de-condense; chi down -> coherent comma-streaks), and below onset
everything stays a homogeneous polar sheet even at sigma 0.2. The real reason DISCRETE BANDS never
appear: the initial/mean density rho0=1.2 sits WELL ABOVE the flocking onset rho_c=1, so the whole
field is a stable homogeneous flock -- there is no dilute disordered gas to coexist with a dense
ordered band. Travelling polar bands (Toner-Tu/Vicsek microphase separation) only exist in a NARROW
density window near rho_c. LEVER: rho0 (mean density), NOT sigma. Added a --rho0 knob to am2_hydro.py
(default 1.2 preserved) to test bands at rho0~1.0-1.05. Droplets: push chi->0 (extend the comma-streak
result) at/near onset for compact coherent blobs. Batch-3 slots below.

## Batch 4 (2026-07-02) -- READING batch-3 montage: DROPLETS = near-onset SPARSE condensation
## (not a new regime); bands still resist rho0; rings hint under chi=0
QUESTION: did rho0 near rho_c unlock discrete travelling bands, did chi->0 make compact droplets,
and did high-v0 broad-Dc waves hollow an aster into a ring? Verdict per slot, then push rings +
retry bands with the c-runaway tamed.

PER-SLOT READ (all N=180, snapshot=final frame; parents in [brackets]):
- s0 band_rho    v0 1.00 w 0.40 rho0 1.05 [silent_bands,rho0 down] : P=0.958 Nc=3 ctr=0.40 sig=0.566.
  HOMOGENEOUS POLAR sheet -- smooth flowing single-hue domains, faint density streaks, NO discrete
  stripes. VERDICT: MISSING. Dropping rho0 to 1.05 did NOT band; still one continuous flock.
- s1 band_rho_lo v0 1.00 w 0.40 rho0 1.00 [band_rho,rho0 down] : P=0.893 Nc=2 ctr=0.00 sig=0.588.
  Polar with 1-2 relic +1 defects BUT the c field SATURATED to a uniform bright ceiling (density
  contrast collapsed to 0.0). VERDICT: MISSING + c RUNAWAY. At rho0=1.0 (exactly rho_c) the flock
  homogenises and c production runs away to a flat maximum -> masks any banding. Need beta/alpha to
  cap c before bands can be judged here.
- s2 sig_band    v0 1.00 w 1.00 rho0 1.05 [sig_band] : P=0.004 Nc=3 ctr=4.30 sig=0.47. THREE large
  ISOLATED compact pinwheel condensates on a NEAR-EMPTY dark background; round bright c wells under
  each. VERDICT: **DROPLETS** (best droplet morphology yet -- compact motile blobs on empty bg,
  ctr=4.30 the highest of any slot precisely because the field is mostly empty). Not the paper's
  MANY-small, but the correct topology at last. Places droplets at low rho0 + moderate omega.
- s3 band_v0     v0 0.60 w 0.40 rho0 1.05 [band_rho,v0 down] : P=0.98 Nc=3 ctr=0.23 sig=0.589.
  HOMOGENEOUS POLAR, even smoother (lower v0). VERDICT: MISSING. Lowering v0 only calms the sheet.
- s4 drop_chi0   v0 0.60 w 1.80 chi 0.00 [drop_chi,chi->0] : P=0.063 Nc=17 ctr=1.29 sig=0.593. FEWER,
  LARGER coherent blobs; the c field shows several OPEN ANNULAR loops (hollow c rings) between blobs.
  VERDICT: proto-RING hint -- killing chi entirely gives the first annular c texture, but the density
  blobs are still filled. chi=0 is necessary-not-sufficient for rings.
- s5 drop_chi_w  v0 0.60 w 1.60 chi 0.10 [drop_chi,w down] : P=0.186 Nc=17 ctr=1.13 sig=0.602.
  TURBULENT near-onset -- elongated multi-hue domains + defects + partial c rings. VERDICT: onset
  boundary layer is disordered/transient, not a clean stream. Streams still MISSING.
- s6 ring_Dc_v0  v0 0.95 w 1.80 Dc 2.20 [ring_Dc,v0 up] : P=0.0 Nc=28 ctr=2.68 sig=0.766. A CLEAN
  REGULAR triangular LATTICE of ~28 compact filled pinwheel vortices over a regular array of round c
  wells. VERDICT: **VORTICES** -- the cleanest, most ordered vortex-lattice panel to date (best
  paper-b match for topology), but FILLED asters, not hollow rings. High v0 did NOT hollow them.
- s7 onset_15    v0 0.60 w 1.50 [onset_15] : P=0.001 Nc=16 ctr=2.56 sig=0.591. ~16 compact ISOLATED
  pinwheel condensates on a DARK background (sparser than the w1.8 space-filling lattice). VERDICT:
  **DROPLETS/onset** -- at the onset the lattice is SPARSE and isolated (droplet-like); as omega rises
  1.5->1.8 the same blobs multiply and fill space into the vortex lattice. Pins droplet<->vortex as a
  DENSITY continuum, not a phase change.

BIGGEST SURPRISE + LEVER: **active DROPLETS are the SAME condensation as vortices, just at LOWER
NUCLEATION DENSITY.** sig_band (rho0 1.05, w 1.0) gives 3 huge isolated blobs; onset_15 (w 1.5) gives
16 sparse ones; the base (w 1.8) gives a 32-blob space-filling lattice. Lowering omega toward the ~1.5
onset OR lowering rho0 reduces the number of nucleation sites, so the condensate is isolated compact
droplets on empty background instead of a wall-to-wall vortex lattice. The lever is PROXIMITY TO ONSET
(omega~1.5) and rho0, NOT a distinct droplet regime. droplets<->vortices is one phase graded by count;
the topological siblings still missing are RINGS (hollow) and STREAMS/BANDS (anisotropic). Secondary:
chi->0 (drop_chi0) minted the first OPEN ANNULAR c loops = the ring lever is off-curl PLUS a wave-
expansion knob (Dc/eps); and rho0->rho_c makes c RUN AWAY, so bands must be retried with beta/alpha
capping c. Batch-4 slots: rings (chi=0 x {v0,Dc,eps}), droplets refined (low-chi near onset; lower
rho0), bands retried with c capped (beta/sigma down at rho0~1.0), + a droplet<->vortex omega probe.

## Batch 5 (2026-07-02) -- READING batch-4 montage: DROPLETS placed (few-large AND many-small);
## RINGS live in the c FIELD (chi=0 + high Dc -> arcs); BANDS still resist -- Drho is the untried lever
QUESTION: did chi=0 x {v0, Dc, eps} finally hollow an aster into a RING; did c-capping (beta/sigma
down) open the Toner-Tu BAND window; where is the droplet<->vortex count crossover?
Base fig coeffs unless noted. Top panel = orientation(HSV) over rho-brightness; bottom = c (magma).
- s0 drop_1hue   v0 0.60 w 1.50 chi 0.10 [onset_15,chi down] : P=0.486 Nc=9 ctr=0.63 sig=0.595.
  LARGE smooth coherent polar DOMAINS (green/purple) with only 2 small +1 defects; diffuse c with faint
  arcs. VERDICT: near-onset MIXED (partly polar, partly condensing) -- chi=0.1 at w=1.5 did NOT give
  compact droplets; it kept most of the domain in the coherent polar sheet with a few nucleating defects.
  chi-down near onset FAVOURS the polar side, not droplets.
- s1 onset_probe v0 0.60 w 1.70 [onset_15,w up] : P=0.001 Nc=26 ctr=2.35 sig=0.639. CLEAN vortex lattice,
  26 filled pinwheels over a c-well array. VERDICT: **VORTICES/onset**. Pins the count continuum:
  w1.5->16, w1.7->26, w1.8->32 blobs. The droplet(sparse)<->vortex(space-filling) crossover sits ~w1.6-1.7.
- s2 drop_lowrho  v0 1.00 w 1.00 rho0 1.02 [sig_band,rho0 down] : P=0.203 Nc=3 ctr=2.64 sig=0.535. THREE
  isolated compact bright DROPLETS on a near-empty background, each ringed by FAINT EXPANDING c ARCS.
  VERDICT: **DROPLETS (few-large)** -- clearest isolated-droplet morphology; lowering rho0 to 1.02 keeps
  it at ~3 blobs (rho0 is the nucleation-count knob, as predicted). The expanding c rings around each blob
  are the ring mechanism in miniature.
- s3 ring_chi_v0  v0 1.00 w 1.80 chi 0.00 [drop_chi0,v0 up] : P=0.009 Nc=65 ctr=1.23 sig=1.053. SIXTY-FIVE
  small single-hue coherent blobs, c field densely dotted, signal the HIGHEST of any slot (1.05).
  VERDICT: **DROPLETS (many-small)** = the paper-a morphology at last (many small round motile blobs)!
  SURPRISE: chi=0 + high v0 + high omega multiplies condensation into MANY small droplets rather than
  hollowing anything. High v0 fragments (more, smaller); chi=0 keeps each a coherent single hue. Rings NO.
- s4 ring_hiDc    v0 0.60 w 1.80 chi 0.00 Dc 3.00 [drop_chi0,Dc up] : P=0.261 Nc=12 ctr=1.25 sig=0.596.
  FEWER (12), broad diffuse coherent domains; c field shows clear ARCS and C-shaped CRESCENTS (partial
  annuli). VERDICT: best **RING precursor** -- chi=0 + broad fast waves (Dc=3) make the c field form
  open arcs/rings; the density blobs stay coherent (not yet hollow) because at w=1.8 the fronts are dense
  and collide before closing. LEVER FOUND: rings are a c-FIELD wave phenomenon (off-curl + wave-expansion),
  needing SPARSE isolated fronts (lower omega/rho0) to close into full annuli.
- s5 ring_eps     v0 0.95 w 1.80 Dc 2.20 eps 0.02 [ring_Dc_v0,eps down] : P=0.001 Nc=37 ctr=2.39 sig=0.937.
  CLEAN FILLED vortex lattice (37 pinwheels). VERDICT: **VORTICES** -- slow recovery (eps down) did NOT
  hollow; eps is NOT a ring lever (FALSIFIED). The ring lever is chi=0 + Dc, not eps.
- s6 band_beta    v0 1.00 w 0.40 rho0 1.00 beta 0.20 [band_rho_lo,beta down] : P=0.962 Nc=1 ctr=0.00
  sig=0.323. HOMOGENEOUS POLAR (smooth blue-green orientation gradient); c still SATURATED uniform
  (sig lower at 0.32 but contrast still 0). VERDICT: MISSING -- beta down cut c amplitude a little but did
  NOT band; the flock stays a uniform ordered sheet (div p ~ 0 -> no density transport). beta FALSIFIED.
- s7 band_lowsig  v0 1.00 w 0.40 rho0 1.00 sigma 0.30 [band_rho_lo,sigma down] : P=0.906 Nc=3 ctr=0.00
  sig=0.588. HOMOGENEOUS POLAR with a few large-scale orientation DOMAIN WALLS/defect lines; c saturated.
  VERDICT: MISSING -- sigma down softens the orientation field (defect lines appear) but density stays
  uniform (ctr=0). Confirms: you cannot band a UNIFORMLY ORDERED sheet; need a density modulational
  instability, which Drho=0.5 is smoothing away.

BIGGEST SURPRISE + LEVER: **chi=0 does not make one ring -- it multiplies droplets (65 tiny blobs at high
v0) OR, with broad Dc waves, paints ARCS into the c field.** The rings in the paper are a CHEMICAL-WAVE
signature: at chi=0 the off-curl polarization lets density sit on expanding c fronts, and with Dc high the
c field forms open annuli -- but only where fronts are SPARSE enough to close (dense w=1.8 fronts collide
into arcs). So the ring recipe = chi=0 + high Dc + LOW nucleation density (lower omega toward onset or
lower rho0). Second: DROPLETS are now fully placed -- few-large (drop_lowrho, rho0 down, 3 blobs) and
many-small (ring_chi_v0, chi=0+high v0, 65 blobs) both reproduced. BANDS remain the ONLY fully-missing
state after 3 attempts: beta and sigma both leave a uniform ordered sheet with c saturated. The UNTRIED
lever is Drho -- density diffusion (0.5) smooths any incipient band; the pure Toner-Tu banding instability
near rho_c needs LOW Drho + omega=0 (truly silent, chemotaxis off). Batch-5: bands via omega=0 x low-Drho
near rho_c (3, the real new hypothesis); rings via chi=0+Dc3 at LOW omega/rho0 to close the annuli (2);
a stream probe (high chi at the turbulent onset layer, 1); a droplet-count boundary probe (1).

## Batch 6 -- 2026-07-02 (read of fig2_b05_montage.png + archive/f2_b05_s*)
Batch-5 tested THE band bet (omega=0 pure Toner-Tu x low Drho near rho_c), the ring-annulus-closing
recipe (chi=0+Dc3 at low omega/rho0), a high-chi stream probe, and a droplet-count probe.

- s0 band_silent  v0 1.00 w 0.00 rho0 1.00 [chemotaxis OFF] : P=0.99 Nc=1 ctr=0.00 sig=0.588.
  HOMOGENEOUS ORDERED sheet -- smooth blue->cyan orientation gradient, density perfectly uniform. omega=0
  (pure Toner-Tu) at rho0=rho_c did NOT band. VERDICT: bands MISSING.
- s1 band_loDrho  v0 1.00 w 0.00 rho0 1.00 Drho 0.15 [band_silent,Drho down] : P=0.981 Nc=1 ctr=0.00
  sig=0.588. STILL a homogeneous ordered sheet (blue gradient, flat density). **THE Drho bet FAILED** --
  dropping density diffusion 0.5->0.15 did not release a banding instability. VERDICT: MISSING.
- s2 band_hiv0    v0 1.60 w 0.00 rho0 1.00 Drho 0.15 [band_loDrho,v0 up] : P=0.998 Nc=4 ctr=0.00 sig=0.588.
  Homogeneous ordered flock with one large aster-like orientation swirl; density uniform. High advection
  did NOT band either. VERDICT: MISSING. (v0 up to 1.6 stays uniform at rho0=rho_c.)
- s3 band_win     v0 1.00 w 0.00 rho0 1.05 Drho 0.15 [band_loDrho,rho0 up] : P=0.939 Nc=3 ctr=0.40
  sig=0.564. **FIRST density modulation of any band attempt** -- faint diagonal density FILAMENTS/streaks
  (ctr 0.40) under large-scale orientation swirls. VERDICT: proto-band. LEVER FOUND: rho0 slightly ABOVE
  onset (1.05, not 1.00) is what seeds the modulation; rho0=1.00 (p0->0) stays uniform. Still too ordered
  globally (P 0.94) and streaks too weak to call bands -- needs sharpening.
- s4 ring_sparse  v0 0.60 w 1.40 chi 0.00 Dc 3.00 [ring_hiDc,omega down] : P=0.32 Nc=7 ctr=0.50 sig=0.604.
  Near-onset partial condensation: 7 coherent blobs, c field forms clear open CRESCENTS/ARCS. VERDICT:
  RING precursor (crescents, not closed). Lower omega (1.4) thinned nucleation vs w=1.8 but fronts still
  merge before closing. P=0.32 = partial order (onset/coexistence layer).
- s5 ring_lowrho  v0 0.60 w 1.80 chi 0.00 Dc 3.00 rho0 1.08 [ring_hiDc,rho0 down] : P=0.137 Nc=10 ctr=1.53
  sig=0.571. Sparse elongated blobs on a dark background (few nucleation sites), some elongated/comma;
  NOT hollow rings. VERDICT: sparse droplets/vortices. rho0 down thinned sites but at w=1.8 they still
  condense filled.
- s6 stream_chi   v0 0.60 w 1.60 chi 1.20 [drop_chi_w,chi up] : P=0.001 Nc=10 ctr=2.38 sig=0.605.
  10 DISCRETE round pinwheel asters (filled), high contrast, P~0. VERDICT: VORTICES/droplets -- **high chi
  did NOT channelize into streams** (FALSIFIED as stream lever); it kept compact asters. Streams MISSING.
- s7 drop_probe   v0 1.00 w 1.50 chi 0.00 [ring_chi_v0,omega down] : P=0.056 Nc=35 ctr=1.58 sig=0.765.
  35 small blobs, many-small droplet field (fewer than the 65 at w=1.8, more than the ~16 at w=1.5 rho0
  1.2). VERDICT: DROPLETS (many-small). Confirms the droplet<->vortex COUNT continuum: count rises with
  omega (onset-proximity) 3->16->35->65, chi=0 keeps each a coherent single hue. Droplets DONE.

BIGGEST SURPRISE + LEVER: **the entire Toner-Tu band bet failed -- omega=0, rho0=rho_c, AND Drho 0.5->0.15
STILL give a perfectly homogeneous ordered sheet (P>0.98, contrast=0).** The homogeneous ordered state at
sigma=0.7 is LINEARLY STABLE; low Drho alone does not destabilize it. The ONE thing that produced any
density structure was rho0=1.05 (s3, ctr 0.40, faint diagonal streaks) -- i.e. sitting slightly ABOVE
onset, not AT it. Why: at rho0=1.00 the ordered amplitude p0->0 (marginal), so there is nothing to
transport; at rho0=1.05 p0~0.24 and the near-onset banding instability (growth ~ v0*sigma*p0, damping ~
Dp*q^2 + Drho*q^2) has a small positive rate. The UNTRIED damping lever is **Dp** (polarization/splay
diffusion, still 0.6 in every band run) -- lowering Dp reduces the q^2 damping of the splay mode that
seeds bands. Batch-6 bet: from band_win (rho0 1.05, Drho 0.15, v0 1.0) lower Dp (0.2) to sharpen the
streaks into bands, then push v0/rho0/chi one at a time. Rings: crescents confirmed (chi=0+Dc3), close the
annuli by thinning nucleation further (rho0 1.10 / omega 1.2). Streams: high chi FALSIFIED -- retry via
advective stretching of the onset layer (high v0 at omega~1.45).

---

## Batch 7 -- 2026-07-02 : BANDS FINALLY APPEAR (Dp-down + rho0>onset). Streams still miss.
Read of `fig2_b06_montage.png` + `archive/f2_b06_s*/`. The batch-6 Dp-down bet PAID OFF: bands are
real now. Parent lineage: BANDS from band_win(v0 1.0,w 0,rho0 1.05,Drho 0.15); RINGS from
ring_sparse(v0 0.6,w 1.4,chi 0,Dc 3); STREAMS from onset(v0 0.6,w 1.45).

- s0 band_loDp    v0 1.00 w 0.00 rho0 1.05 Drho 0.15 Dp 0.20 [band_win, Dp down] : P=0.878 Nc=2 ctr=0.47
  sig=0.55. Density panel shows a CRISS-CROSS NETWORK of bright diagonal density ridges on a purple
  background -- the FIRST genuine band structure of the whole loop. Top: large diagonal polar domains.
  VERDICT: BANDS (network/stream-like). Dp DOWN released the splay mode exactly as predicted; P
  dropped 0.94->0.88 (order broken by the modulation). Bet CONFIRMED.
- s1 band_loDp_v0 v0 1.60 w 0.00 rho0 1.05 Drho 0.15 Dp 0.20 [band_loDp, v0 up] : P=0.83 Nc=2 ctr=0.76
  sig=0.48. STRONGER, sharper criss-cross network of density ridges; highest band contrast (0.76). Higher
  v0 increased the drive (growth ~ v0*sigma*p0). VERDICT: BANDS (network) -- best contrast. The
  intersecting-river look is arguably closer to STREAMS than to clean parallel bands.
- s2 band_rho11   v0 1.00 w 0.00 rho0 1.10 Drho 0.15 (Dp 0.6) [band_win, rho0 up] : P=0.997 Nc=2 ctr=0.40
  sig=0.569. Density panel shows CLEAN PARALLEL VERTICAL BANDS (two-three bright stripes). VERDICT: SILENT
  POLAR BANDS -- cleanest morphology match to paper panel e. rho0=1.10 (deeper into coexistence) gives
  parallel stripes even at default Dp=0.6; but P=0.997 = still globally very ordered (bands are a gentle
  density modulation on an ordered flock, exactly the near-onset picture).
- s3 band_chi     v0 1.00 w 0.00 rho0 1.05 Drho 0.15 chi 0.90 [band_win, chi up] : P=0.972 Nc=3 ctr=0.32
  sig=0.57. WAVY vertical bands, lower contrast than s2. VERDICT: BANDS (wavy). chi (front-steepening)
  bent the stripes but did not sharpen contrast; rho0 is the stronger band lever than chi.
- s4 ring_thin    v0 0.60 w 1.40 chi 0.00 Dc 3.00 rho0 1.10 [ring_sparse, rho0 up] : P=0.683 Nc=4 ctr=0.43
  sig=0.589. Density panel shows open CURVED ARCS, some curving toward closed LOOPS. VERDICT: RING
  precursor -- best arcs yet, nearly closing. Thinning nucleation (rho0 1.10) let isolated c fronts
  expand into loops; still not fully closed annuli.
- s5 ring_om12    v0 0.60 w 1.20 chi 0.00 Dc 3.00 [ring_sparse, omega down] : P=0.604 Nc=4 ctr=0.43
  sig=0.609. DIFFUSE turbulent domains, faint filaments, no clean fronts. VERDICT: under-condensed --
  w=1.20 is too far BELOW onset (~1.4) so no isolated fronts nucleate. Rings need to stay near w~1.4, NOT
  lower. Boundary result: condensation onset is at/just above w=1.4, sharp.
- s6 stream_v0    v0 1.30 w 1.45 (base chi 0.5) [onset, v0 up] : P=0.019 Nc=27 ctr=2.66 sig=0.778. Bottom:
  ~27 round DROPLETS; top: small pinwheel vortices. VERDICT: CONDENSED vortex/droplet lattice, NOT
  streams. SURPRISE: w=1.45 is already fully condensed (P~0.02), and high v0 pushed it deeper -- there
  is NO "partial-order onset layer" at w=1.45 to stretch into streams. Stream attempt FAILED (3rd miss).
- s7 stream_chi2  v0 0.60 w 1.45 chi 1.00 [onset, chi up] : P=0.003 Nc=6 ctr=2.54 sig=0.585. 6 large
  round pinwheel droplets/vortices. VERDICT: CONDENSED droplets, NOT streams. high chi FALSIFIED AGAIN.
  w=1.45+v0 0.6 = 6 large vortex droplets.

BIGGEST SURPRISE + LEVER: BANDS exist at last, and there are TWO independent routes to them, both sitting
slightly ABOVE the flocking onset (rho0>1.0) with a weak damping on the modulation mode: (1) LOW Dp
(0.2) at rho0=1.05 -> criss-cross density network (s0/s1); (2) rho0=1.10 at default Dp -> clean parallel
stripes (s2). The lever is the SIGN of (near-onset banding growth v0*sigma*p0) minus (splay/density damping
(Dp+Drho)*q^2): raise p0 (rho0 above onset), lower the damping (Dp or Drho down), or raise the drive (v0),
and the ordered sheet buckles into bands. omega=0 => these are SILENT bands (c passive). SECOND surprise:
w=1.45 is ALREADY condensed even at v0=0.6 (P~0), so streams cannot be made by "stretching the onset layer"
-- the polar<->condensed transition is sharp with no partial-order corridor there. Streams must instead
live in the LOW-omega polar/band regime with chemotaxis WEAKLY on (small omega channeling the band network
into directed rivers) -- the batch-7 stream bet. Rings: closest arcs yet (ring_thin), close them by
thinning nucleation MORE (rho0 1.15 or Dc 4).

---

## Batch 8 -- 2026-07-02 : SIGNALLING BANDS + STREAMS + RINGS all placed. Five of six states now landed.
Read of `fig2_b07_montage.png` + `archive/f2_b07_s*/`. The three batch-7 bets ALL paid off: turning omega
weakly ON in a band recipe gave signalling bands; the low-Dp network + weak omega + high v0 gave streams;
and Dc=4 finally closed a ring. Parents: SIGNALLING BANDS from band_rho11(v0 1.0,w 0,rho0 1.10,Drho 0.15);
STREAMS from band_loDp(v0 1.0,w 0,rho0 1.05,Drho 0.15,Dp 0.2); RINGS from ring_thin(v0 0.6,w 1.4,chi 0,Dc 3,rho0 1.10).

- s0 sigband_w05  v0 1.00 w 0.50 rho0 1.10 Drho 0.15 [band_rho11, omega ON] : P=0.894 Nc=2 ctr=0.50 sig=0.566.
  Diagonal bright density ridges; orientation = large coherent green/cyan domains crossed by a band; c field
  ACTIVE with bright ridges CO-LOCATED on the density stripes. VERDICT: **SIGNALLING POLAR BANDS** -- the first
  ever; weak omega (0.5) makes c co-travel with the density modulation (paper panel f). Bands slightly curved
  (chemotaxis bends them) vs the straight silent bands.
- s1 sigband_w08  v0 1.00 w 0.80 [sigband_w05, omega up] : P=0.939 Nc=6 ctr=0.49 sig=0.557. More MOTTLED
  orientation, small defects nucleating (Nc 2->6), bright branching c network. VERDICT: signalling bands
  STARTING TO CONDENSE. BOUNDARY: at v0=1.0 the band->condensed onset sits ABOVE w=0.8 (defects appear but the
  banded backbone survives). omega pushes bands toward the aster lattice, as expected.
- s2 sigband_v0   v0 1.50 w 0.50 [sigband_w05, v0 up] : P=0.983 Nc=2 ctr=0.67 sig=0.52. CLEAN PARALLEL VERTICAL
  stripes; c panel = clean bright vertical stripes co-located on the density bands. VERDICT: **SIGNALLING BANDS,
  cleanest match to paper f.** High v0 (drive ~ v0*sigma*p0) sharpens the wavy bands into straight parallel
  stripes AND keeps c co-travelling -- same v0-sharpening as the silent bands, signal stays on. ctr 0.67 (best).
- s3 stream_w06   v0 1.00 w 0.60 rho0 1.05 Drho 0.15 Dp 0.20 [band_loDp, omega ON] : P=0.878 Nc=2 ctr=0.54
  sig=0.53. CRISS-CROSS network of bright density rivers; c bright network co-located. VERDICT: stream-adjacent
  NETWORK -- weak omega on the low-Dp network channels it into rivers, but they intersect (not one-directional).
- s4 stream_v0hi  v0 1.40 w 0.60 [stream_w06, v0 up] : P=0.849 Nc=4 ctr=0.70 sig=0.50. STRONG diagonal criss-
  cross of bright density RIVERS; orientation = bold green diagonal streams crossing. Highest contrast (0.70).
  VERDICT: **STREAMS (best candidate yet)** -- high v0 advective stretching elongates the network into directed
  diagonal lanes flowing between c sources. Not yet a single-direction river field (lanes criss-cross), but the
  clearest directed-flow morphology of the whole loop. Places streams at LOW omega + low Dp + high v0.
- s5 ring_rho115  v0 0.60 w 1.40 chi 0.00 Dc 3.00 rho0 1.15 [ring_thin, rho0 up] : P=0.706 Nc=4 ctr=0.43
  sig=0.595. DIFFUSE broad domains, one big c well + faint edge arcs; partial order (P 0.71). VERDICT: RING
  precursor, WEAKER than Dc4 -- thinning to rho0 1.15 at Dc=3 under-condensed (too few sites, fronts too slow
  to close). rho0 is NOT the ring-closing lever at Dc=3.
- s6 ring_Dc4     v0 0.60 w 1.40 chi 0.00 Dc 4.00 rho0 1.10 [ring_thin, Dc up] : P=0.093 Nc=6 ctr=0.54 sig=0.582.
  CONDENSED aster (orientation = full colour wheel round a dark core); c panel shows a BRIGHT ANNULUS around a
  dark centre = the clearest RING signature of the loop. VERDICT: **RINGS.** SURPRISE: Dc 3->4 collapsed P
  0.71->0.093 (partial-order arcs -> fully condensed aster) AND ringed the core. Broad/fast c fronts (high Dc)
  don't just texture c -- they NUCLEATE condensation and wrap the density into an annular front. Dc, not rho0,
  is the ring lever.
- s7 band_bothlev v0 1.00 w 0.00 rho0 1.10 Drho 0.15 Dp 0.20 [band_rho11, Dp down] : P=0.866 Nc=2 ctr=0.43
  sig=0.557. CRISS-CROSS diamond network of density ridges (NOT parallel stripes); diagonal orientation domains.
  VERDICT: BANDS (network). Combining both levers (rho0 1.10 + low Dp 0.2) does NOT give the sharpest parallel
  stripes -- LOW Dp dominates and pushes it into the network morphology. Clean parallel stripes need DEFAULT Dp
  (rho0-only route); low Dp is the stream/network route. The two band routes are morphologically distinct, not
  additive.

BIGGEST SURPRISE + LEVER: **five of six states are now placed in one batch.** (1) SIGNALLING BANDS = a silent-band
recipe (rho0>=1.10, Drho low) with omega weakly ON (0.5): c co-travels with the density stripes, and high v0
(1.5) sharpens them into clean parallel signalling bands = paper f. omega is the silent<->signalling switch; v0
is the sharpening knob for BOTH. (2) STREAMS = the low-Dp band NETWORK (rho0 1.05, Dp 0.2) + weak omega (0.6) +
HIGH v0 (1.4): advection stretches the criss-cross network into directed density rivers (ctr 0.70). (3) RINGS =
chi=0 + Dc=4 at w=1.4: the SURPRISE lever is Dc -- raising it 3->4 flips partial-order arcs into a condensed
aster whose c wraps into a closed ANNULUS (P 0.71->0.09). Dc controls both front speed (ring closure) and
condensation onset. Only MISSING/rough now: streams are a criss-cross network not a clean one-way river field,
and the ring is one aster's annulus not a field of hollow rings. Batch-8 sharpens rings (Dc up / v0 up to hollow
the core / thin nucleation) and streams (omega up to organise rivers / v0 up to stretch), + signalling-band v0
push and two boundary probes.

---

## Batch 9 -- 2026-07-02 : the RING window is NARROW (Dc~4 optimum, non-monotonic); high v0 is the master flow-sharpener
Read of `fig2_b08_montage.png` + `archive/f2_b08_s*/`. Batch-8 tried to SHARPEN rings (Dc up / v0 up / thin),
streams (omega up / v0 up), and signalling bands (v0 up), + boundary probes. Parents: RINGS from
ring_Dc4(v0 0.6,w 1.4,chi 0,Dc 4,rho0 1.10); STREAMS from stream_v0hi(v0 1.4,w 0.6,rho0 1.05,Drho 0.15,Dp 0.2);
SIGBANDS from sigband_v0(v0 1.5,w 0.5,rho0 1.10,Drho 0.15).

- s0 ring_Dc5    v0 0.60 w 1.40 chi 0 Dc 5.0 rho0 1.10 [ring_Dc4, Dc up] : P=0.981 Nc=1 ctr=0.26 sig=1.497.
  Top: smooth green/purple HOMOGENEOUS POLAR sheet, no condensation. c: near-UNIFORM pink (sig 1.50 the highest of
  any slot but FLAT, ctr 0.26). VERDICT: **RING OVERSHOT.** Dc 4->5 diffuses c so broadly the chemotactic gradient
  grad(c) collapses -> no condensation -> homogeneous polar. SURPRISE: the "ring lever" is NON-MONOTONIC -- past
  Dc~4 it kills the ring entirely (P 0.09 -> 0.98). Dc~4 is an OPTIMUM, not a "more is better" knob.
- s1 ring_hollow v0 0.90 w 1.40 chi 0 Dc 4.0 rho0 1.10 [ring_Dc4, v0 up] : P=0.204 Nc=6 ctr=2.39 sig=0.521.
  Top: ~6 sparse COMPACT blobs, each a small single-hue core (green/blue/red) on dark bg; some elongated. c: bright
  compact/elongated wells. VERDICT: **DROPLETS, not hollowed rings.** v0 0.6->0.9 CONDENSED/fragmented into compact
  filled droplets (ctr 2.39) rather than hollowing the aster core. Advection does not hollow at chi=0/Dc4.
- s2 ring_thin12 v0 0.60 w 1.40 chi 0 Dc 4.0 rho0 1.20 [ring_Dc4, rho0 up] : P=0.41 Nc=2 ctr=0.89 sig=0.582.
  Top: ONE big pinwheel ASTER (full colour wheel round a central defect). c: broad bright FILLED blob (bright
  centre, not annular). VERDICT: single large FILLED aster. Thinning to rho0 1.20 gave fewer sites (Nc 2) but each
  is a filled aster, NOT a hollow ring. rho0 up does not close/hollow annuli at Dc4.
- s3 stream_om09 v0 1.40 w 0.90 rho0 1.05 Drho 0.15 Dp 0.20 [stream_v0hi, omega up] : P=0.131 Nc=11 ctr=1.66 sig=0.476.
  Top: turbulent domains with several defect cores. c: CELLULAR/foam NETWORK of bright ridges enclosing dark cells +
  a bright defect. VERDICT: **STREAM OVERSHOT -> condensing.** omega 0.6->0.9 pushed the stream network toward the
  condensed aster lattice (P 0.85->0.13, Nc 11 defects). BOUNDARY: at v0 1.4/low-Dp the stream->condensed onset sits
  BELOW w=0.9. omega channels rivers only weakly (<~0.7); above that it condenses them.
- s4 stream_v018 v0 1.80 w 0.60 rho0 1.05 Drho 0.15 Dp 0.20 [stream_v0hi, v0 up] : P=0.982 Nc=3 ctr=0.92 sig=0.459.
  Top: BOLD single-hue (green) wavy horizontal LANES; c: bright wavy horizontal ridges co-located. VERDICT:
  **STREAMS / clean directed lanes -- best flow morphology yet.** v0 1.4->1.8 organised the criss-cross network into
  near-parallel single-hue directed lanes (P 0.98, ctr 0.92 best). High v0 straightens the rivers into a coherent
  one-hue flow field. (Borderline with silent bands -- but c is active here, sig 0.46, = directed streams with signal.)
- s5 stream_om03 v0 1.40 w 0.30 rho0 1.05 Drho 0.15 Dp 0.20 [stream_v0hi, omega down] : P=0.943 Nc=5 ctr=0.64 sig=0.506.
  Top: diagonal blue/green CRISS-CROSS stripes (two hues). c: diagonal bright rivers crossing. VERDICT: ordered
  CRISS-CROSS river network (P 0.94). BOUNDARY: weak omega (0.3) keeps the silent-ish ordered criss-cross network
  (lanes intersect); it does not organise into one-way flow. omega must be ~0.6 (not lower) for directed channeling.
- s6 sigband_v20 v0 2.00 w 0.50 rho0 1.10 Drho 0.15 [sigband_v0, v0 up] : P=0.989 Nc=3 ctr=0.93 sig=0.463.
  Top: CLEAN vertical wavy stripes (cyan/green); c: bright vertical wavy stripes co-located. VERDICT: **SIGNALLING
  BANDS, cleanest of the whole loop (ctr 0.93, P 0.99).** v0=2.0 sharpens the signalling stripes further -- v0 is the
  master band-sharpening knob and does NOT blow up at 2.0. Best paper-f match.
- s7 sigband_w10 v0 1.50 w 1.00 rho0 1.10 Drho 0.15 [sigband_v0, omega up] : P=0.116 Nc=11 ctr=2.18 sig=0.451.
  Top: turbulent/condensed with defect cores; c: cellular network + bright wells. VERDICT: **sig-bands CONDENSED.**
  BOUNDARY: at v0=1.5 the sig-band->condensed onset sits between w=0.5 (clean bands) and w=1.0 (condensed aster
  network, P 0.12). omega ~0.8-1.0 is where signalling bands break into the vortex lattice. Confirms omega switch.

BIGGEST SURPRISE + LEVER: **the RING is the one NARROW, non-robust state -- ring_Dc4 (batch-7) is a knife-edge
optimum, and EVERY single-variable push off it this batch DEGRADED it:** Dc up (5.0) over-diffuses c -> gradient
collapses -> homogeneous polar (P 0.98); v0 up (0.9) condenses/fragments into filled droplets; rho0 up (1.20)
gives one big FILLED aster. So rings live in a tiny box (chi=0, Dc~4, w~1.4, rho0~1.10) and are the hardest paper
state to reproduce as a FIELD -- we still only get one aster's annulus. SECOND (positive) surprise: **high v0
(1.8-2.0) is the master flow-sharpener** -- both stream_v018 (ctr 0.92, P 0.98) and sigband_v20 (ctr 0.93, P 0.99)
gave the cleanest directed-lane/stripe morphologies of the entire loop, and v0=2.0 did NOT blow up. v0 raises the
banding drive (~v0*sigma*p0) and straightens criss-cross rivers into coherent one-hue lanes. THIRD: two boundaries
pinned -- stream->condensed onset is BELOW w=0.9 (at v0 1.4/low-Dp); sig-band->condensed onset is between w 0.5 and
1.0 (at v0 1.5). Batch-9: attack the ring window from inside (Dc 4.2 top-bracket, v0 0.7 gentle-hollow, w 1.5 &
rho0 1.05 for MORE ring sites = a field), bracket the stream lane-straightening (v0 1.6) and stream boundary
(w 0.75), + push sigband v0 2.5 and pin the sigband condensation onset (w 0.8).

## 2026-07-02 -- Batch 10 (batch-9 read: ring window attacked from inside; stream/sigband boundaries pinned)
Parent montage fig2_b09_montage.png. All slots produced a panel (none FAILED). Numbers from progress.txt.
Ring lineage from ring_Dc4 (v0 0.6, w 1.4, chi 0, Dc 4.0, rho0 1.10 = batch-7 knife-edge annulus).

- s0 ring_Dc42 v0 0.60 w 1.40 chi 0 Dc 4.2 rho0 1.10 [ring_Dc4, Dc up] : P=0.409 Nc=4 ctr=0.47 sig=0.591.
  Top: large SMOOTH polar patches (few, broad hues), only faint defects. c: diffuse warm blobs, no annuli.
  VERDICT: **Dc top-bracketed -- 4.2 is already PAST the optimum.** ctr fell 0.09->0.47 and P rose 0.09->0.41
  vs ring_Dc4: at Dc 4.2 the c fronts over-broaden, gradient weakens, condensation is releasing back toward
  polar. Confirms the ring optimum sits at Dc~4.0 (between 4.0 sharp-annulus and 4.2 diffuse), VERY sharp.
- s1 ring_v07 v0 0.70 w 1.40 chi 0 Dc 4.0 rho0 1.10 [ring_Dc4, v0 up] : P=0.451 Nc=5 ctr=1.16 sig=0.567.
  Top: smooth polar blobs + a clear bottom-right PINWHEEL defect (full HSV wheel round a DARK core). c: a couple
  of compact bright wells + faint arcs. VERDICT: gentle advection (v0 0.6->0.7) CONDENSES a few sites (ctr 0.47
  ->1.16) and the bottom-right defect shows a hollow-cored vortex, but it is 1-2 asters, NOT a field of rings.
  v0 0.7 is milder than the 0.9 that fully fragmented (batch-9) -- a middling few-aster state.
- s2 ring_w15 v0 0.60 w 1.50 chi 0 Dc 4.0 rho0 1.10 [ring_Dc4, omega up] : P=0.928 Nc=5 ctr=0.46 sig=0.588.
  Top: HOMOGENEOUS polar sheet (P 0.93!), smooth broad hues. c: diffuse warm blobs. VERDICT: **BIGGEST SURPRISE
  -- raising omega toward MORE condensation instead COLLAPSED to homogeneous polar.** Expected more nucleation
  sites; got the opposite. WHY: at Dc=4 the c field is already near the over-diffusion edge; adding omega tips c
  into smooth near-saturation -> grad(c) collapses -> chemotaxis has nothing to grip -> ordered polar sheet (same
  failure mode as ring_Dc5). The ring box is tighter than thought: w must be ~1.4 EXACTLY, not 1.5.
- s3 ring_rho105 v0 0.60 w 1.40 chi 0 Dc 4.0 rho0 1.05 [ring_Dc4, rho0 down] : P=0.251 Nc=9 ctr=0.51 sig=0.58.
  Top: smooth polar patches + a clear top-right blue/red PINWHEEL defect (dark core); Nc=9 defects total. c:
  warm ARCS/filaments + a bright well. VERDICT: **BEST ring-FIELD candidate of the batch.** Lowering rho0
  1.10->1.05 nucleated MORE sites (Nc 2->9) at partial order (P 0.25) with visible c arcs -- the "more sites
  without leaving the box" bet WORKED for count, but the arcs are not yet closed hollow annuli. Thinning density
  (rho0 down), not omega up, is the correct field-multiplier lever inside the ring box.
- s4 stream_v016 v0 1.60 w 0.60 rho0 1.05 Drho 0.15 Dp 0.20 [stream_v018, v0 down] : P=0.963 Nc=5 ctr=0.76 sig=0.483.
  Top: BOLD green diagonal single-hue LANES; c: warm diagonal ridges co-located. VERDICT: **clean directed STREAMS.**
  Brackets lane-straightening: v0 1.4 (ctr 0.70 criss-cross) -> 1.6 (ctr 0.76 lanes) -> 1.8 (ctr 0.92 boldest).
  v0 straightening is monotonic; 1.8 remains the best stream. Good paper-d match.
- s5 stream_w075 v0 1.80 w 0.75 rho0 1.05 Drho 0.15 Dp 0.20 [stream_v018, omega up] : P=0.169 Nc=9 ctr=1.52 sig=0.445.
  Top: turbulent with several defect cores/spirals; c: filamentary network + bright wells. VERDICT: **STREAM
  BOUNDARY pinned.** At v0 1.8, omega 0.75 already CONDENSES the river net (P 0.96->0.17, Nc 9, ctr 1.52). Combined
  with s4 (w 0.6 clean lanes): the stream->condensed onset at v0 1.8 sits between w 0.6 and 0.75. Tighter than the
  batch-9 bracket. Streams need omega below ~0.7.
- s6 sigband_v25 v0 2.50 w 0.50 rho0 1.10 Drho 0.15 [sigband_v20, v0 up] : P=0.97 Nc=2 ctr=0.93 sig=0.503.
  Top: CLEAN vertical cyan/green stripes; c: bright vertical bands co-located. VERDICT: **signalling bands, stable
  at v0=2.5, NO blow-up.** ctr 0.93 = identical to v0 2.0 -> the v0 sharpening SATURATES above ~2.0 (plateau), but
  the integrator stays stable to v0 2.5 (dt 0.02). Confirms v0 as sharpener with a ceiling; best paper-f tie.
- s7 sigband_w08 v0 2.00 w 0.80 rho0 1.10 Drho 0.15 [sigband_v20, omega up] : P=0.564 Nc=14 ctr=1.26 sig=0.507.
  Top: green polar patches breaking into a CELLULAR/foam network of ~14 defect bubbles; c: bright wells at nodes.
  VERDICT: **SIGBAND BOUNDARY pinned.** At v0 2.0, omega 0.8 partially condenses the stripes into a foam/cellular
  lattice (P 0.99->0.56, Nc 2->14). With s6 (w 0.5 clean): sig-band->condensed onset at v0 2.0 is between w 0.5
  and 0.8. Consistent with the batch-9 "survive only omega below ~0.8" -- now bracketed at v0 2.0.

BIGGEST SURPRISE + LEVER: **ring_w15 -- pushing omega 1.4->1.5 (toward MORE condensation) COLLAPSED the ring
regime to a homogeneous polar sheet (P 0.93), the opposite of "more sites".** At Dc=4 the c field is already at
the over-diffusion edge; extra omega saturates c and kills grad(c), so chemotaxis loses its grip (same mode as
Dc 5). Lever: inside the ring box omega is NOT a site-multiplier -- rho0 DOWN (s3, Nc 2->9) is. The ring window is
{chi=0, Dc~4.0, w~1.4, rho0 1.05-1.10}, and s3 (rho0 1.05, 9 arced sites) is the closest approach to a FIELD of
rings, though the arcs remain open. SECOND: v0 sharpening SATURATES -- sigband ctr plateaus at 0.93 from v0 2.0 to
2.5 (stable, no blow-up). THIRD: two boundaries tightened -- stream->condensed onset (v0 1.8) between w 0.6/0.75;
sig-band->condensed onset (v0 2.0) between w 0.5/0.8. Batch-10 (final): close the ring FIELD by hollowing the 9
rho0-1.05 sites (Dc just below 4, gentle v0, faster c-decay alpha), plus fine boundary probes (ring Dc 4.1, w 1.45;
stream w 0.68; sigband w 0.65).
