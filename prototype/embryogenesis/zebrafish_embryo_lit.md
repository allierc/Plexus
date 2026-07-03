# Zebrafish embryogenesis — quantitative reference

Scope: teleost (mostly zebrafish, *Danio rerio*) early development — blastula, epiboly, gastrulation,
germ-layer formation, body-axis elongation — as studied by (a) single-cell tracking, (b) division/lineage
tracking, and (c) quantitative morphodynamics (flow fields, strain, tissue mechanics, cell shape/packing).
Compiled as scoring targets for an in-silico (active-matter × MPM) blastula. Citations verified via web
search 2026-07; open-access PDFs (arXiv only, egress-restricted env) in `/workspace/Plexus/papers/zebrafish/`.

## Key papers

**Imaging + single-cell / digital-embryo tracking**
- Keller, Schmidt, Wittbrodt & Stelzer 2008, *Science* — DSLM light-sheet "digital embryo"; first in-toto
  reconstruction of zebrafish first 24 h; ~55M nuclear entries, cell positions/divisions/tracks; found a
  maternally-defined morphodynamic symmetry break defining the body axis. Observable: 3D nucleus positions + division/migration tracks.
- Tomer, Khairy, Amat & Keller 2012, *Nat. Methods* — SiMView simultaneous multiview light-sheet (4 arms,
  no rotation), 175M voxels/s; quantitative whole-embryo imaging enabling automated cell tracking.
- Royer, Lemon, Chhetri, Wan, Coleman, Myers & Keller 2016, *Nat. Biotechnol.* — AutoPilot adaptive
  light-sheet; 2–5× resolution/signal gain during large morphogenetic change; long-term whole-embryo imaging.

**Automated lineage / division tracking (validated on zebrafish)**
- Amat, Lemon, Mossing, McDole, Wan, Branson, Myers & Keller 2014, *Nat. Methods* — TGMM: nuclei as 3D
  Gaussians, sequential-Bayesian GMM segmentation+tracking; ~26k cells/min, fly/zebrafish/mouse. Observable: full lineage trees, division events.
- Stegmaier, Amat, Lemon, McDole, Wan, Teodoro, Mikut & Keller 2016, *Dev. Cell* — RACE real-time 3D
  cell-shape segmentation; 55–330× faster, 2–5× more accurate; yields cell-shape + tissue-anisotropy maps.
- Faure et al. 2016, *Nat. Commun.* (ncomms9674) — open workflow (BioEmergences) reconstructing cell-lineage
  trees from 3D+t in zebrafish/ascidian/sea-urchin; ~98% correct links between consecutive frames.
- Sugawara/Bhide et al. (ELEPHANT) 2022, *eLife* 69380 — incremental deep-learning nucleus detection+linking
  on sparse annotations, built on Mastodon/Fiji; interactive human-in-the-loop 3D lineage tracking.
- Mastodon-sc / TrackMate (Tinevez et al. 2017, *Methods*) + MaMuT — the Fiji large-scale tracking stack that
  ELEPHANT extends; standard editable-lineage tooling.

**Morphogenetic flow, strain, tissue mechanics**
- Behrndt, Salbreux, Campinho, Hauschild, Oswald, Roensch, Grill & Heisenberg 2012, *Science* — EVL epiboly
  driven by a YSL actomyosin ring via cable-constriction **and** flow-friction (retrograde actomyosin flow ×
  friction). Observable: myosin flow velocity, ring tension, spreading rate.
- Campinho et al. 2013, *Nat. Cell Biol.* — tension-oriented cell divisions limit anisotropic tissue tension
  during EVL epiboly. Observable: division-orientation vs tissue-tension axis.
- Pastor-Escuredo et al. 2016 (bioRxiv 054353) — kinematic analysis of reconstructed lineages; compression/
  expansion + distortion (shear) rate maps; zebrafish gastrula behaves as a compressible fluid.
- "Strain maps of convergence & extension" 2021, *Sci. Rep.* (s41598-021-98233-z; bioRxiv 407940) — multicell
  spherical domains → velocity fields → 3D strain-rate tensor (AP/ML/radial) + curl; maps compaction/expansion
  and L-R symmetric strain through epiboly→segmentation.
- Mongera, Rowghanian, Campàs et al. 2018, *Nature* — ferrofluid-droplet in-vivo rheology; tailbud fluid→solid
  jamming gradient underlies axis elongation. Observable: yield stress, viscoelastic relaxation, local rearrangement/velocity gradients.

**Cell shape / packing / segregation**
- Schötz et al. 2008, *HFSP J.* — germ-layer tissue surface tensions (ecto vs mesendo) set sorting order;
  E-cadherin knockdown reverses phase. Observable: tissue surface tension, envelopment/segregation order.
- Krieg et al. 2008, *Nat. Cell Biol.* — AFM shows actomyosin cortical tension (Nodal-regulated) governs
  germ-layer organization. Observable: single-cell cortex tension, adhesion force.
- Krens, Heisenberg et al. 2017, *Development* — CellFIT-3D force inference in the intact gastrula; interstitial
  osmolarity tunes differential tension driving in-vivo segregation. Observable: in-vivo TST, mixing/segregation index.

**Active-matter / vertex / self-propelled-Voronoi models (+ simulation stacks)**
- Bi, Lopez, Schwarz & Manning 2015, *Nat. Phys.* — density-independent rigidity transition in vertex model at
  shape index p₀ ≈ 3.81. Observable: shape index p = P/√A, shear modulus.
- Bi, Yang, Marchetti & Manning 2016, *Phys. Rev. X* — Self-Propelled Voronoi (SPV): glass/jamming set by
  motility v₀, persistence, target p₀; transition at ⟨p⟩ ≈ 3.81. Observable: MSD, Deff, p̄.
- Barton, Henkes, Marchetti & Sknepnek 2017, *PLoS Comput. Biol.* — Active Vertex Model in **SAMoS**
  (Delaunay-Voronoi, dynamic T1s); velocity correlations, growth/division/boundaries.
- Sussman 2017, *Comp. Phys. Commun.* — **cellGPU**: GPU-accelerated vertex/SPV (up to ~10³× speedups).
- Theis, Suzanne & Gay 2021, *JOSS* — **tyssue**: Python 2D/3D vertex-model library.

## Canonical quantitative observables (what to score a model against)

- **Cell velocity field** v(x,t) and **spatial velocity-correlation length** ξ (decay of ⟨v·v⟩); correlation time.
- **Strain-rate tensor** ε̇ from the velocity gradient: isotropic dilation (compaction/expansion) + deviatoric
  shear (distortion) + antisymmetric **vorticity/curl**; resolved along AP/ML/radial axes.
- **T1 (neighbor-exchange) rate** and net topological reconnection — the microscopic unit of tissue fluidity.
- **Division rate** and **division-axis orientation** distribution (vs tissue stress/tension principal axis).
- **Lineage trees**: link accuracy, cell-cycle length, clonal dispersion / fate-map coherence.
- **Neighbor-number (polygon-class) distribution**, cell **area** and **anisotropy/elongation**; **shape index** p = P/√A (fluid ⇄ solid near ≈3.81).
- **Segregation / mixing index** for two populations; tissue surface tension / cortex tension.
- **MSD & persistence**: MSD(τ) exponent (caged/subdiffusive → diffusive), velocity persistence time; effective Deff.
- **Tissue rheology**: yield stress, viscoelastic relaxation time (elastic <~few s, fluid >~1 min in tailbud).

## Template for hypothesis generation & tests

1. **Division axis follows stress.** H: cell-division orientation aligns with the local principal tissue-stress
   (tension) axis. Test: angle Δθ between measured division axis and principal-stress eigenvector; predict
   ⟨Δθ⟩ small and sharpening with tension anisotropy (cf. Campinho 2013). Metric: circular mean/variance of Δθ.
2. **Shape-index fluidization gradient.** H: an AP gradient in shape index p̄ crossing ≈3.81 co-locates with the
   fluid→solid jamming front. Test: map p̄(x) and T1-rate(x); predict rearrangement rate → 0 where p̄ < 3.81
   (cf. Mongera 2018, Bi 2016). Metric: p̄ vs T1-rate correlation, jamming-front position.
3. **Flow-friction epiboly.** H: EVL spreading rate is set by retrograde actomyosin-flow × friction, not just
   ring contraction. Test: perturb effective friction in silico, compare marginal flow-velocity and closure
   rate to Behrndt 2012 scaling. Metric: spreading rate vs friction/flow product.
4. **Correlation length ↔ motility/adhesion.** H: velocity-correlation length ξ grows as the tissue approaches
   jamming (↑persistence, ↑p₀→3.81). Test: sweep v₀, p₀; compare ξ(τ) and MSD exponent to SPV predictions and
   to nuclei-tracked ξ in gastrula. Metric: ξ, MSD slope, Deff.
5. **Strain-rate symmetry.** H: the model reproduces L-R-symmetric AP-expansion / ML-compaction bands plus
   rotational (curl) strain during convergence-extension. Test: compute ε̇ tensor fields; compare band geometry,
   sign, and curl to the strain-map study. Metric: strain-trace maps, dorsal/ventral asymmetry index, curl magnitude.
6. **Tension-driven segregation.** H: imposing differential surface/cortex tension reproduces germ-layer
   ecto-outside / mesendo-inside sorting and its reversal under reduced adhesion. Test: two populations with
   tunable interfacial tension; measure envelopment order and mixing index over time (cf. Schötz 2008, Krens 2017).
   Metric: segregation/mixing index vs ΔTST, envelopment correctness.
