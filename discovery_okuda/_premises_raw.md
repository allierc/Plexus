# Candidate tissue premises — RAW, unfiltered, un-attacked

**70 candidates from 5 miners. This is the raw material, not the corpus.**

Cedric asked for something *concise, the basics not the specific* — so most of these will be
cut. They are here to be read and struck through. Three adversarial lenses (truth /
checkability / coherence) are still ruling on them.

Grades: `textbook` certain · `typical` usually-but-conditional · `contested`.

---

## from: Papers read (text extracted with pypdf, then read in context windows around every growth/division mention):
- 

**1. In a proliferating epithelium every cell's reference (target) volume increases at a strictly positive rate, whether or not that cell is receiving a patterning signal; a morphogen biases where cells divide and how fast they grow, but it is not the on/off switch for growth existing at all.**  `[textbook]`

- constrains: `morphogen_growth_3d.rho (the baseline floor in growth = rate*(rho + hill(a))), and the presence/absence of a body-wide growth operator such as vesicle_growth in the schedule`
- check: `rho > 0, OR a body-wide growth operator (vesicle_growth/uniform_ramp) is present in `schedule`. Equivalently, measured: mean over cells of d(v_eq)/dt > 0 at every recorded frame, and total tissue volume V_tot(t_end) > V_tot(t_growth_start).`
- if violated: "rho = 0 with no uniform growth operator means the body adds no material at all: growth = rate*hill(a) is exactly zero for every cell whose activator is below a_sw, and the tissue can only redistribute the volume it was seeded with. Is that a deliberate ablation? If so, declare it and predict what the protrusion is made of; if not, set rho > 0 or add vesicle_growth to the schedule."
- source: SimuCell3D (Runser, Vetter & Iber, Nat. Comput. Sci. 4:299-309, 2024), Methods §"Pressure": "To model cell growth, V0 can evolve over time according to prescribed growth laws, such as the linear form dV0/dt = g, where g is a constant volumetric growth rate that can vary from cell to cell"; Table 2 lists g as a positive parameter with a measured biological range 0.1-1.8e-20 m^3/s (refs 83,84). Fig. 3 case study: "The cells were grown at a uniform volumetric rate without division until they had doubled in size." jax-morph (Deshpande et al., arXiv 2407.06295), SI §"Cell Growth": "All cell radii grow constantly in time, up to a specified maximum radius Rmax" — unconditional and identical for every cell; the gene network sets division propensity and secretion, never the growth rate.
- scope: 3D vertex / deformable-cell model of a proliferating epithelial monolayer or closed vesicle, morphogenesis timescales (hours-days). Not applicable to a declared-quiescent or terminally-differentiated tissue.

**2. A localized outgrowth (limb bud, branch, tube) in a proliferating epithelium is built by locally biased proliferation riding on top of body-wide growth; the surrounding tissue keeps adding material while the tip adds it faster. It is not built by a body that adds nothing and a tip that is the sole source of new material.**  `[typical]`

- constrains: `the cell_react -> morphogen_growth_3d `gate` connection in _discovery.structure.connections, jointly with morphogen_growth_3d.rho and morphogen_growth_3d.a_sw`
- check: `If the `gate` slot is connected (growth is signal-gated at all) then rho > 0, so ungated cells still grow. Measurable: the ratio (tip growth rate)/(body growth rate) must be finite; with rho = 0 and a Hill switch it is infinite because the denominator is exactly zero.`
- if violated: "With rho = 0 the gated-cell/ungated-cell growth ratio is infinite — a tip growing into a body that is frozen. jax-morph's elongation experiment, the closest published analogue, holds growth uniform and biases only division. Are you claiming a different mechanism (tip-only material addition)? Declare it and say what tissue does that; otherwise set rho > 0 and put the spatial bias in divide_3d."
- source: jax-morph (Deshpande et al., arXiv 2407.06295), Fig. 2 "Elongation": "Source cells (in red) secrete the growth factor and cannot divide. Proliferating cells (in gray) sense the growth factor and divide in response to it"; main text: "cell division events are concentrated at the distal tip where the chemical signal is weakest, ensuring directional growth and elongation" and "Like limb bud outgrowth, it achieves directional elongation through localized growth at the tip [36]" — all while every cell radius grows unconditionally (SI §Cell Growth).
- scope: Proliferating epithelial cluster / monolayer producing a single protrusion, morphogenesis timescales. Does not cover protrusions made by apical constriction or by lumen pressure (no such operator here).

**3. Cells divide because they got big: a cell reaches mitosis after roughly doubling its birth volume, and the division threshold is a size the cell actually attains, not an asymptote it approaches. A cycle clock that fires on cells that have not grown is not a cell cycle.**  `[textbook]`

- constrains: `divide_3d.min_cycle and divide_3d.vcap jointly with morphogen_growth_3d.rate, .rho and .vth_frac, and general.dt`
- check: `Two relations, both must hold. (i) Reachability: the growth ceiling must exceed the division trigger, vth_frac > vcap (currently both 1.5, so V asymptotes to the trigger and never crosses it). (ii) Timing: growth must carry a cell from birth volume to the trigger within one clock period, rate*(rho + hill_max)*min_cycle*dt >= (vcap - 1). In ref_okuda_route: 0.01*(0+1)*16*0.02 = 0.0032 vs required 0.5 — short by ~156x, so 100% of divisions are clock-forced.`
- if violated: "vth_frac = vcap = 1.5 means the target volume ceiling and the division trigger are the same number, and rate*min_cycle*dt = 0.0032 against a required 0.5, so no cell ever reaches its mitotic size and every division fires on the clock alone. Is division meant to be size-independent here? If so declare it and predict the cell-volume distribution; if not, raise vth_frac above vcap and set rate so that (vcap-1)/(rate*(rho+1)) equals min_cycle*dt."
- source: SimuCell3D Methods §"Cell division": "Cells are divided on the basis of a volume threshold, that is, if V > Vmax"; Table 2 gives Vmax = 1.4e-15 m^3 with daughters at ~Vmax/2 = 7e-16 m^3, i.e. division at a doubling, and dV0/dt = g is unbounded so Vmax is always crossed. jax-morph, Forward Model: "Cell division events are stochastic and produce two daughter cells, each with half the volume of a fully grown mother cell", and SI §Cell Growth caps radius at Rmax > birth radius so a cell always regrows to full size between divisions.
- scope: 3D vertex / deformable-cell epithelium with a size-triggered or size-gated division operator, morphogenesis timescales.

**4. Division itself is volume-neutral — one mother becomes two daughters summing to the mother's volume — so a tissue whose cell number rises without its total volume rising is not growing, it is fragmenting, and its mean cell volume falls as 2^-n.**  `[textbook]`

- constrains: `the recorded time series of total tissue volume V_tot and mean cell volume, against divide_3d.max_div_frac / .max_div / .min_cycle and morphogen_growth_3d.rho`
- check: `Over any window in which cell count multiplies by k, total volume must also rise by a comparable factor: V_tot(t1)/V_tot(t0) should track N(t1)/N(t0) within a stated tolerance, and mean cell volume must be stationary (not monotonically decreasing) after the initial transient. With rho = 0 and clock-forced division, N rises ~6x (500 -> ~2900 cells in the archived runs) while V_tot rises by at most rate*window = 5% in the activated spot only, so mean cell volume collapses ~6x.`
- if violated: "Cell count multiplies while total tissue volume is flat — the cells are being cut into ever-smaller pieces at constant mass, which is what a cleaving zygote does, not a growing epithelium. Is subdivision-at-constant-volume the intended regime? If so, declare it and say at what cell size the run stops being biological; if not, add baseline growth so V_tot tracks N."
- source: SimuCell3D Fig. 3: "cells were grown at a uniform volumetric rate without division until they had doubled in size" — the figure axis is "2x tissue volume growth", i.e. tissue volume, not just cell count, is the growth observable; cell death is a separate explicit operator (V < Vmin, Table 2: Vmin = 3.7e-16 m^3). jax-morph SI §"Cell Division": daughters are placed at +/-Rbirth about the mother centroid and then regrow, so mass is added by growth between divisions, never by the division event.
- scope: Proliferating epithelial monolayer or closed vesicle over many cell cycles. Excludes cleavage-stage embryos, where subdivision at constant volume IS the correct convention and must be declared.

**5. When the tissue carrying a chemical grows, the chemical is diluted: a concentration obeys dc/dt + (V_dot/V)c = reaction + diffusion, and the dilution rate acts as an extra decay on every species. A patterning chemistry survives growth only if the growth rate is small compared with the reaction rates that regenerate the pattern.**  `[textbook]`

- constrains: `morphogen_growth_3d.conserve_amount (true = divide cell.chem by the volume growth ratio each tick) jointly with cell_react.rate, .mu_a, .mu_h and morphogen_growth_3d.rate`
- check: `conserve_amount must be true whenever cell volume changes (otherwise concentration is silently created out of nothing), AND the dilution rate must be small against the reaction: rate*(rho + 1) << min(mu_a, mu_h)*cell_react.rate. In ref_okuda_route: dilution <= 0.01 vs mu_a*rate = 1.0, a ratio of 100 — comfortable — but the check must be stated, because at rho = 1 and rate = 0.03 with a slower reaction it is not.`
- if violated: "Which is it: conserve_amount = false (concentration is not diluted by growth, so total morphogen is manufactured by the growth operator) or a dilution rate comparable to the reaction rate (the pattern is being washed out by growth)? Declare which, and report the total morphogen amount and the peak activator level as functions of time so the reader can see it."
- source: Ledesma-Duran (arXiv 2308.12196), §II Eqs. (3)-(5): mass conservation on a domain of size l(t) gives dc/dt + (l_dot/l)c = (D/l^2)d2c/dxi2 + f(c), "This dilution term quantifies the local change in concentration due to growth"; §III Eq. (12) and the "General traits" bullet: the deviation of the homogeneous state from the reaction fixed point is proportional to r (the growth rate) and the solution is only valid "for slow variation of the size of the domain".
- scope: Any reaction-diffusion field carried on cells whose volume changes (3D vertex tissue with a growth operator), morphogenesis timescales. The << is an order-of-magnitude criterion, not a sharp bound.

**6. A chemistry whose fixed point sits at zero concentration is driven to extinction by growth-dilution; only a chemistry with a genuine basal source term holds a nonzero level in a growing tissue.**  `[typical]`

- constrains: `cell_react.a0 (Gierer-Meinhardt basal activator production, currently 0.01) jointly with morphogen_growth_3d.conserve_amount and morphogen_growth_3d.a_sw`
- check: `a0 > 0 whenever conserve_amount = true and the tissue grows; and, measurably, the steady basal activator level must stay above the Hill threshold in the spots: max_over_cells(a) > a_sw at every frame after the pattern forms, otherwise the gate hillv is ~0 and the growth operator is inert regardless of rho.`
- if violated: "With growth-dilution active and a0 at 0.01, does the activator field still exceed a_sw = 1.5 late in the run, or has dilution pulled the whole field below the switch? Report max(a) versus a_sw per frame. If the field is dead, say whether the intended claim is that growth extinguishes its own signal."
- source: Ledesma-Duran (arXiv 2308.12196), §II: the Brusselator "is an example of open reactors where a reactant that produces the activator is held constant, leading to a non-equilibrium steady state where the catalyst concentrations are non-zero. The BVAM model ... can be understood to reflect a closed reactor where concentrations will tend to zero. As we will see, this difference would affect the ability of an RDD system to change its fixed-point concentration in the long term and, therefore, the stability of perturbations."
- scope: Reaction-diffusion patterning on a growing epithelial vesicle/monolayer. Statement is about which reaction families survive dilution, not about a specific numeric a0.

**7. Diffusion does not scale with a growing body: as a tissue enlarges at fixed diffusion coefficients the effective diffusion length shrinks relative to the tissue, so the pattern must refine — spots split or new spots insert — rather than the existing spots simply getting bigger.**  `[textbook]`

- constrains: `cell_diffuse.d_a, cell_diffuse.d_h, cell_diffuse.chi, and the measured spot/peak count versus tissue linear size`
- check: `Over a window in which tissue linear size L grows by a factor s (s = (V_tot ratio)^(1/3)), the number of activator peaks should rise roughly as s^2 on a shell / s^d in d dimensions. A constant peak count across substantial growth means the chemistry is implicitly scaling with the body and that must be declared.`
- if violated: "The activator peak count did not change while the tissue grew by a factor s in linear size — at fixed D the Turing wavelength is fixed, so peaks should have multiplied. Is the pattern being re-seeded, frozen, or is growth actually negligible? Report peak count and V_tot together and say which."
- source: Ledesma-Duran (arXiv 2308.12196), §II Eq. (4): in the fixed (co-moving) coordinate the diffusion term becomes (D/l^2(t)) d2c/dxi2 — the diffusion coefficient enters divided by the square of the current domain size.
- scope: Isotropically growing domain carrying a two-component reaction-diffusion system; the paper derives it in 1D and states the generalisation to multi-component and curved/anisotropic domains is available elsewhere (its refs 14-16). The exponent on a curved growing shell is an extrapolation, hence the loose "roughly".

**8. A dividing epithelial cell is bisected by a plane through its own centroid, oriented either at random or perpendicular to its longest axis (Hertwig's rule); the daughters are contiguous, share the mother's material, and no volume is created at the division event.**  `[textbook]`

- constrains: `divide_3d implementation (hertwig vs orient_iface), divide_3d.factor (2.0) and divide_3d.reset_noise`
- check: `At every division tick, sum of daughter volumes == mother volume to within the cytokinesis tolerance, and the cleavage plane contains the mother's centroid. Recorded: V_tot must be continuous across division ticks (no step), and daughter volume ratio ~1:1 up to reset_noise.`
- if violated: "Does V_tot step at division ticks? If daughters do not sum to the mother, the division operator is a mass source or sink, and every volume-based conclusion in the run inherits it. State the per-division volume budget."
- source: SimuCell3D Methods §"Cell division": "Cells are divided on the basis of a volume threshold ... They are bisected by a plane running through their centroid, whose orientation can depend on the cell type. The orientation is either random or perpendicular to the cell's longest axis, as given by the eigenvector belonging to the smallest eigenvalue of its covariance matrix." jax-morph SI: daughters at +/-Rbirth about the mother position, "Each daughter cell inherits all other properties of its mother cell."
- scope: 3D deformable-cell / vertex epithelium. Asymmetric or oriented divisions driven by an explicit polarity cue are a declared exception, not the default.

**9. Proliferation in a confluent epithelium is suppressed by crowding: compressive stress activates the Hippo pathway and shuts down division, so cells do not keep dividing at a fixed per-cell rate as the sheet jams.**  `[typical]`

- constrains: `divide_3d.min_cycle, .max_div_frac and .max_div — currently a pure clock with a global throttle and no mechanical input`
- check: `The realised division rate must fall as the tissue compresses: divisions-per-cell-per-frame should anticorrelate with mean (v_eq - V)/v_eq or with mean cell volume. A per-frame division fraction that is constant while mean cell volume falls several-fold is an unfeedbacked clock.`
- if violated: "max_div_frac is a fixed fraction per frame with no stress input, so cells keep dividing while the sheet is being crushed. Is the absence of contact inhibition a deliberate ablation? If so, declare it and predict the endpoint (cell volume, packing); if not, gate division on cell volume or on compressive stress."
- source: jax-morph (arXiv 2407.06295), §"Mechano-chemical Regulation of Homogeneous Growth": "In the learned network, mechanical stress directly inhibits cell division, mirroring the role of the Hippo pathway in the wing disc. Experimentally, mechanical stress has been shown to activate the Hippo pathway [48], leading to Warts-mediated phosphorylation and inactivation of the growth-promoting transcription co-activator Yorkie (Yki), ultimately suppressing cell proliferation."
- scope: Confluent proliferating epithelium (Drosophila wing imaginal disc is the cited system), morphogenesis timescales. The strength of the feedback is system-specific; the sign is not.

**10. A confluent epithelium above the shape-index rigidity threshold is a fluid: it can change its outline at constant volume purely by cells rearranging past one another. Below it the sheet is solid and shape change requires adding material. Which side you are on decides whether a protrusion is evidence of growth.**  `[textbook]`

- constrains: `shape_energy_3d.p0 (default 3.90; vocabulary range 3.4-4.2) jointly with morphogen_growth_3d.rho and with any protrusion/aspect metric used as a success criterion`
- check: `If p0 exceeds the rigidity threshold (VertAX measures the solid-to-fluid transition at p0 = 3.85 in 2D) then a protrusion metric alone is not evidence of growth, and the run must additionally report delta_V_tot over the protrusion window. A protrusion with delta_V_tot ~ 0 is a fluid-flow rearrangement, not an outgrowth.`
- if violated: "p0 = 3.90 puts the sheet on the fluid side of the rigidity transition, where the tissue can be pulled into a tube at constant volume, and rho = 0 means no volume is being added. Is the claimed bud a growth bud or a fluidisation bud? Report protrusion length together with delta_V_tot so the two can be told apart."
- source: VertAX (Pasqui et al., arXiv 2604.06896), Fig. 3g "Tissue solid-to-fluid transition": fraction of zero-energy states versus target shape factor p0, sigmoid fit x0 = 3.85, described as "the classical rigidity transition from a solid-like to a fluid-like phase as the target shape factor p0 increases [ref 11]".
- scope: 2D confluent vertex model — the transition value 3.85 is measured in 2D by VertAX. The threshold for a 3D monolayer shape energy is NOT established by this source; treat 3.85 as an order marker, not the 3D number.

**11. Neighbour exchange in an epithelium happens when the exchange lowers the junctional energy; a T1 executed against the energy gradient is active work and needs a modelled force to pay for it.**  `[typical]`

- constrains: `reconnect_t1_3d.l_th_frac, .max_flips, .every — the operator flips on a pure length threshold with no energy acceptance test`
- check: `Each executed flip should reduce the total shape energy, or the flip count per frame must be bounded, recorded, and shown not to be saturating max_flips (300 in the flagship recipe).`
- if violated: "reconnect_t1_3d flips every edge below l_th_frac with no energy test and a cap of 300 flips per frame. Is the flip count saturating that cap? If so the topology is being driven by the cap rather than by mechanics — declare it, or add the energy-decrease acceptance test."
- source: VertAX (Pasqui et al., arXiv 2604.06896), Methods §"T1 transitions": "If an edge length falls below a user-defined threshold l_min, two candidate topological updates are evaluated: 1. The edge is stretched along its current orientation. 2. A T1 transition is executed ... The configuration that yields the lower total energy is accepted."
- scope: Vertex-model epithelium with topological remodelling. Adjacent to, not part of, the growth/division convention — included because unconditional T1s are the mechanism by which a no-growth tissue can still produce a protrusion.

_miner notes: WHAT I COULD NOT ESTABLISH

1. VertAX does not support growth or division premises. The word "growth" appears zero times in the full extracted text and there is no division/proliferation operator anywhere in the paper — it is a fixed-cell-number confluent 2D vertex model whose only topology change is T1. The two VertAX-sourced entries above are about the mechanical substrate (rigidity transition, _

## from: PAPERS (read via pypdf text extraction; pdftoppm unavailable so the Read tool could not render pages):
- /work

**12. No cell cross-section of any shape can have a perimeter-to-root-area ratio below 2*sqrt(pi) = 3.545 (that is a circle), and a straight-sided cell with n neighbours cannot go below 2*sqrt(n*tan(pi/n)) -- 3.722 for a hexagon, 3.812 for a pentagon, 4.0 for a square.**  `[textbook]`

- constrains: `shape_energy_3d.p0 and divide_3d.p0 (config/okuda/*.yaml; searchable range in discovery/composition_space.py is (3.4, 4.2, 3.90)). tyssue_ops3d sets P0 = p0 * sqrt(A0), so p0 IS the dimensionless 2D shape index.`
- check: `p0 >= 3.5449 always; p0 >= 3.7224 whenever the packing is hexagon-dominated and an unfrustrated (zero-energy) ground state is intended. Separately: measured shape_idx_mean >= 3.5449 -- a smaller measured value is an area/perimeter bug, not a compact tissue.`
- if violated: "p0 = 3.4 sits below the geometric floor 3.545, so no cell of any shape can reach both its target area and its target perimeter: every cell carries residual cortical tension for the whole run and the tissue is unconditionally frustrated. Is a permanently frustrated solid the specimen you intend? If so, declare it and state the consequence you predict (nonzero shear modulus, no spontaneous intercalation, frozen shape pattern); if not, what value of p0 above 3.72 do you intend?"
- source: Isoperimetric inequality (textbook geometry); the hexagon value sqrt(8*sqrt(3)) = 3.7224 is stated in Kim, Zhang & Schwarz 2024, Sec. II A 1 and Discussion.
- scope: epithelial monolayer / closed vesicle; surface (polygonal) vertex model; morphogenesis timescales; applies to any dimensionless 2D shape index p = P/sqrt(A).

**13. A confluent 3D tissue is a packing of space-filling polyhedra, and the index that controls it is s = A/V^(2/3): its absolute floor is the sphere at 4.836 and its floor among single space-filling polyhedra is the Kelvin truncated octahedron at 5.315 (rhombic dodecahedron 5.345, elongated dodecahedron 5.493, cube 6, tetrahedron 7.21). A cell drawn as a polygon on a surface is governed instead by p = P/sqrt(A); the two indices are not interchangeable and their critical values differ by ~1.5.**  `[textbook]`

- constrains: `voronoi_tension_3d.s0 and voronoi_tension_shell.s0 (default 5.4, prototype/Turing_vertex) versus shape_energy_3d.p0 (default 3.9, prototype/Tyssue) -- and any claim that the okuda/Tyssue recipe is a "3D vertex model".`
- check: `If cells are polyhedra carrying a volume term: 4.836 < s0, and s0 >= 5.315 for a reachable zero-energy confluent ground state. If cells are polygons on a shell: use p0 in 2D units and never compare it against 5.x. Assert that the operator's shape-index parameter and its geometry (polygon vs polyhedron) agree.`
- if violated: "Which object is the cell here -- a polyhedron with a volume, or a polygon on a shell whose 'volume' is the wedge to the shell centre? Declare it, because it decides whether the governing index is s0 (critical ~5.3-5.4) or p0 (critical ~3.7-3.8), and a number quoted under the wrong one is off by ~1.5."
- source: Kim, Zhang & Schwarz 2024, Sec. II B and Table III (truncated octahedron 5.31474, rhombic dodecahedron 5.34539, elongated dodecahedron 5.49324, cube 6, tetrahedron 7.21, icosahedron 5.148). The sphere value 4.836 follows from the isoperimetric inequality and is UNCITED in these four papers.
- scope: 3D vertex model / confluent bulk aggregate, and closed-vesicle surface models; no lumen-pressure organ, no ECM.

**14. Crossing the critical shape index is a statement about the tissue, not the solver: below it a cell cannot simultaneously attain its target area/surface and its target perimeter/volume, so a finite energy barrier separates neighbour configurations, the tissue has a nonzero shear modulus and can hold a shape against sustained stress; above it a zero-energy configuration exists, barriers vanish, and the tissue flows and cannot hold a shape. Reported critical values: 3.81 (2D disordered, neighbour exchange allowed), 3.72 (2D ordered hexagonal, no exchanges), 5.39 +/- 0.01 (3D disordered vertex model with reconnections), 5.315 (3D mean-field ordered).**  `[typical]`

- constrains: `shape_energy_3d.p0 / voronoi_tension_*.s0 versus the phenotype the recipe claims: a bud/tube that persists requires the rigid side; convergent extension or flow requires the fluid side.`
- check: `sign(p0 - 3.81) (or sign(s0 - 5.39)) must match the declared regime, AND the measured shape_idx_mean of the shape-bearing cells must fall on the same side. A run reporting a persistent protrusion while shape_idx_mean of the tube cells exceeds 3.81 is claiming a fluid holds a shape.`
- if violated: "Your target index puts the tissue on the fluid side of the rigidity transition, but the phenotype you are scoring is a shape that persists. Which is it: a fluid that must be continuously driven to keep the shape (then what drives it, and what happens when the driver stops?), or a solid (then lower p0/s0 below the transition and say so)?"
- source: Zhang & Schwarz 2022, Results "A rigidity transition in bulk" (s0* = 5.39 +/- 0.01 from the decay time of the neighbour-overlap function Qn); Kim, Zhang & Schwarz 2024, Discussion (3.813 with T1 events, attributed to Bi et al. 2015; 3.72 for the ordered hexagon; 5.315 mean-field truncated octahedron).
- scope: epithelial monolayer (2D values) and 3D confluent aggregate (3D values); morphogenesis timescales, i.e. slower than cortical remodelling; values are for the quadratic area+perimeter / surface+volume energy, not for foam-like linear-tension energies.

**15. In the fluid phase cells actually reach their preferred shape, so the mean realized shape index tracks the target; in the rigid phase they cannot, the realized mean sits away from the target and the distribution is markedly broader. What an experiment measures is the realized index, never the target.**  `[typical]`

- constrains: `tube_analysis.frame_metrics outputs shape_idx_mean / shape_idx_med / shape_idx_p95 versus the configured shape_energy_3d.p0.`
- check: `|shape_idx_mean - p0| small with a narrow spread iff the run is claimed fluid; if the run is claimed rigid, expect shape_idx_mean > p0 with a broad upper tail. Both numbers must be reported together; p0 alone is never "the cell shape".`
- if violated: "The realized shape index and the target disagree by more than the spread of the distribution. Is the tissue in the rigid phase (in which case say so -- the cells are frustrated and cannot reach p0), or is the mechanics simply not relaxed? Declare which, and give the measurement that separates them."
- source: Zhang & Schwarz 2022, Results p.7 and Supplemental Fig. S1 ("for s0 = 5.40 and above the average of the distribution tracks the target s0 ... while in the solid-like state, the average of the distribution cannot track the target s0 and the distribution is more broaden").
- scope: 3D confluent aggregate (as measured) and, by the same compatible/incompatible argument, epithelial monolayers; morphogenesis timescales.

**16. A cell is mostly water: on morphogenesis timescales its volume changes only by uptake or loss of material, never by mechanical compression, and it is never zero or negative. A polyhedron that has turned inside out is not a cell.**  `[textbook]`

- constrains: `shape_energy_3d.K_V and shape_energy_3d.antiinv (set to 0.0 -- inversion guard OFF -- in config/okuda/round_44_base.yaml while growth and division are on); voronoi_tension_3d.vmax (documented in the source as the "degenerate-polyhedron guard; needed when V0 grows"); measurables vol_cv, broken_frac, folded_frac.`
- check: `min_i V_i > 0 on every recorded frame and broken_frac == 0 up to the evidence horizon; with growth disabled, vol_cv should stay at or below ~0.1, i.e. the volume term must be stiff relative to the tension terms (the reference implementation uses K_V/K_A = 10).`
- if violated: "Cells with non-positive or inverted volume appear in the trajectory with the anti-inversion guard off. Are inverted cells part of the specimen? If they are an accepted numerical artefact, state the frame at which they first appear and treat everything after it as non-evidence; if not, turn the guard on and state the volume-fluctuation level you expect."
- source: Zhang & Schwarz 2022 Eq. (1) and Table I (K_V = 10 vs K_A = 1); Sarkar & Krajnc 2023 Methods Eq. (10), where kappa_V is named the "cell-incompressibility constant"; Kim, Zhang & Schwarz 2024 Sec. I ("the volume spring ... captures the volume compressibility of a cell given that it contains water, proteins, fats").
- scope: 3D vertex model / closed vesicle; morphogenesis timescales (hours-days); no cell death unless an apoptosis operator is present.

**17. Growth in tissue is an increase in cell volume, i.e. material taken up; a cell that only spreads its apical area at constant volume is thinning, and a closed sheet that only increases area at fixed enclosed volume must buckle. Area does not determine volume.**  `[typical]`

- constrains: `morphogen_growth_3d (rate, rho, conserve_amount), which drives the A0/P0 targets; shape_energy_3d's per-face target V0f, which in tyssue_ops3d is the wedge volume from the shell centre to the face (a partition of the lumen), not the cell's own volume; h0 in the monolayer implementation is the only thickness variable.`
- check: `Declare which quantity grows and verify the matching measurable moves together: cell volume (A*h with h tracked), apical area at fixed volume (then h must fall), or enclosed lumen volume (then the shell must buckle or inflate). Total area, total wedge volume and thickness are not free to disagree.`
- if violated: "The growth operator increases target area but nothing in the state carries cell thickness or true cell volume. Are these cells thinning as they spread -- and if so by how much, and does that thickness stay physical? If instead you mean the cells are growing, which state variable holds their volume?"
- source: Zhang & Schwarz 2022 Eq. (1) and Kim, Zhang & Schwarz 2024 Eq. (4) both give a cell an independent target volume V0 alongside its target surface area A0, so the two cannot be collapsed into one. The biological statement (growth = uptake of material) is UNCITED here and is flagged for a citation.
- scope: epithelial monolayer / closed vesicle with an explicit growth operator; morphogenesis timescales; no lumen-pressure organ.

**18. Cells swap neighbours only when the junction between them has actually collapsed: the shared interface shrinks to (near) zero and four cells transiently meet at a point. A junction still a sizeable fraction of a cell diameter long is a load-bearing contact, and severing it is not a rearrangement.**  `[typical]`

- constrains: `reconnect_t1_3d.l_th / reconnect_t1_3d.l_th_frac (threshold = l_th_frac * mean edge length in tyssue_t1_ops3d). config/okuda/round_44_base.yaml uses l_th_frac = 0.28; prototype/Tyssue scripts use 0.35.`
- check: `l_th / <edge length> << 1 -- of order 0.05 in the reference implementation (Zhang & Schwarz use l_th = 0.02 in units where the cell size V0^(1/3) = 1). Also: at t = 0 essentially no edge should be below threshold. If a substantial fraction is, the threshold is selecting ordinary junctions, not collapsed ones.`
- if violated: "The reconnection threshold is 28-35% of the mean junction length, roughly five times the reference value, so cells are being swapped while they still share a substantial interface. Is accelerated intercalation a deliberate choice? If so, state the intercalation rate you are targeting and how it was calibrated; if not, what threshold below 0.1 of the mean edge do you intend?"
- source: Sarkar & Krajnc 2023, "Topological transitions" (rearrangement proceeds by "merging vertex pairs of vanishingly short edges"); Sorichetti et al. 2026 Sec. 2.5.1 (rearrangements are "triggered when a cell-cell junction shrinks below a threshold"); the 0.02 reference value is Zhang & Schwarz 2022, Table I.
- scope: epithelial monolayer / closed vesicle / 3D confluent aggregate with an explicit neighbour-exchange operator; morphogenesis timescales.

**19. A neighbour exchange in tissue is a continuous mechanical event: the collapsing junction's energy goes to zero as its length does, so the tissue's energy immediately before and after the exchange can differ only by an amount of order that vanishing junction. An exchange that changes the mechanical energy by much more than that is a discontinuity injected by the algorithm, not something the tissue did.**  `[typical]`

- constrains: `reconnect_t1_3d -- specifically the accept/refuse test in tyssue_t1_ops3d.t1_flip_3d, which currently tests only geometric validity (non-degenerate, outward-facing, simple faces) and never the energy change.`
- check: `|Delta E| for a committed flip <= O(tension scale * l_th). Instrument accepted flips and report the distribution of Delta E against that bound; refuse flips that exceed it.`
- if violated: "Reconnections are being committed on geometric validity alone, with no bound on the energy jump they inject. What is the distribution of Delta E over accepted flips, and how does its tail compare with the energy of the junction that vanished? If large jumps are accepted, say so and predict their effect on the measured rheology."
- source: Zhang & Schwarz 2022, Model: "The first condition is that the change in energy before and after the reconnection event should be in the order of lth", following Okuda et al.
- scope: 3D vertex model and closed-vesicle surface vertex model with explicit reconnection; morphogenesis timescales.

**20. Intercalation in a developing tissue accumulates: a newly formed junction is remodelled after it appears, so neighbour exchanges yield net topological change rather than a junction oscillating between two states. Flipping an edge and flipping it back has spent topology events without rearranging the tissue.**  `[contested]`

- constrains: `reconnect_t1_3d: `new_len` is set equal to the flip threshold `thr`, so a freshly created junction sits exactly at the threshold and a single relaxation step can push it back below; also reconnect_t1_3d.every and max_flips (300/frame in round_44_base).`
- check: `Compare net neighbour change per cell over the run against the cumulative flip count (mesh key n_t1); the ratio must not approach zero. Structurally, the length assigned to a new junction must exceed l_th by a margin so it is not immediately re-flippable.`
- if violated: "Flips are being committed at a high rate while the neighbour topology barely changes -- the mesh may be buzzing between two states rather than rearranging. What is your net-change-to-flip ratio, and what sets the length of a newly created junction relative to the flip threshold?"
- source: Zhang & Schwarz 2022, Results: "Note that we have not yet incorporated a mechanism to prevent back-and-forth reconnection events. Recent work implements such a mechanism in two dimensions" (citing Das, Sastry & Bi, controlled neighbour exchanges).
- scope: epithelial monolayer / closed vesicle with explicit T1s; morphogenesis timescales. Whether real T1s are ratcheted is tissue-dependent; the measurement (net versus gross rearrangement) is not.

**21. In a confluent aggregate every interface is shared by exactly two cells, every edge is a tricellular junction shared by exactly three, and a generic vertex has exactly four edges (four cells). Two distinct cells share at most one interface, two interfaces share at most one edge, and two edges share at most one pair of vertices. Every cell is a closed polyhedron, hence has at least four faces. Higher-order vertices are transient states of a rearrangement, never resting configurations.**  `[textbook]`

- constrains: `topo_snapshot_3d output (E_srce/E_trgt/E_face, nF, Nv) and the validity tests in tyssue_t1_ops3d (_local_manifold_ok, _polygon_simple_2d); the broken_n / broken_frac measurable in tube_analysis.frame_metrics.`
- check: `Per frame: no cell with fewer than 4 faces (3D) or fewer than 3 sides (surface); no cell pair sharing two or more faces; no face pair sharing two or more edges; the count of vertices whose valence differs from 4 (3D) or 3 (surface) must be zero outside the frame in which a flip is executed.`
- if violated: "The topology contains configurations a confluent tissue cannot have (a cell pair sharing two interfaces, or a persistent higher-order vertex). Is this an accepted intermediate state that resolves within one frame? Show it resolving; if it persists, the packing is no longer confluent and the shape statistics computed on it are not tissue statistics."
- source: Zhang & Schwarz 2022, Model (the three Okuda topological-irreversibility sub-conditions: two edges must not share two vertices, two faces must not share two or more edges, two cells must not share two or more faces); Sarkar & Krajnc 2023, Fig. 1B-D and Eq. (8) (an edge IS a tricellular junction shared by cells l, m, n; the merged vertex is 6-fold and must be immediately resolved); Kim, Zhang & Schwarz 2024, Sec. II B criterion 3 (each vertex in the tiling has four edges).
- scope: 3D confluent aggregate and closed-vesicle surface packing; any regime; no free boundary except where an explicit boundary/empty-cell construct exists.

**22. A closed epithelial vesicle is a topological sphere with trivalent junctions, so it satisfies V - E + F = 2 and its cells have on average exactly 6 - 12/F sides -- essentially six neighbours each. A neighbour exchange rotates a junction and changes none of V, E, F. A tissue also cannot pass through itself.**  `[textbook]`

- constrains: `topo_snapshot_3d (nF, Nv, half-edge tables) and reconnect_t1_3d (asserted in its docstring to keep V, E, F and Euler = 2 fixed); the self-intersection/inversion guard shape_energy_3d.antiinv (0.0 in round_44_base).`
- check: `Euler(V, E, F) == 2 on every recorded frame; mean sides per cell == 6 - 12/nF to within a few 1e-3; V, E and F unchanged across a frame in which only T1s fired; and no two non-adjacent faces intersect.`
- if violated: "The mean neighbour count has drifted away from 6 - 12/F, or Euler is no longer 2. Has the sheet changed genus (a fusion or a perforation), or are vertices no longer trivalent, or are faces being dropped? Name which, because each is a different specimen -- and if a fusion is intended, which operator performs it?"
- source: Euler's formula plus trivalency (textbook: 3V = 2E gives E = 3(F-2) and mean sides 2E/F = 6 - 12/F); the invariance of V, E, F under a T1 is asserted in /workspace/Plexus/prototype/Tyssue/tyssue_t1_ops3d.py. The biological statement that epithelial cells average six neighbours is UNCITED in these four papers.
- scope: closed vesicle / epithelial monolayer represented as a closed trivalent surface mesh; genus 0; no lumen perforation or fusion operator.

**23. Real epithelia routinely form junctions where more than three cells meet (rosettes), both transiently during intercalation and stably in some tissues; a representation that admits only three-cell junctions cannot express them and will systematically undercount rearrangement precisely in the crowded regions where they occur.**  `[typical]`

- constrains: `the choice between the re-tessellated Voronoi route (voronoi_tension_3d / voronoi_graph_3d, where vertices are always shared by exactly three cells) and the explicit half-edge mesh (shape_energy_3d + reconnect_t1_3d); reconnect_t1_3d refuses any flip whose endpoints are not trivalent.`
- check: `Count and report flips refused because an endpoint was non-trivalent, and the number of would-be higher-order junctions. If the recipe's target biology involves rosettes (convergent extension, neural-tube-like folding), that count must be surfaced, not silently discarded.`
- if violated: "This representation cannot form a rosette, and refused rearrangements are being dropped without a count. Are rosettes out of scope for this specimen? Declare it and state which observed behaviour that excludes; otherwise report the refusal count so the intercalation rate can be corrected."
- source: Sorichetti et al. 2026, Sec. 2.5.1: Voronoi models are "not allowing the representation of rosettes -- configurations where more than three cells meet at a single point", and "cell-neighbor exchanges cannot be explicitly controlled, despite being tightly regulated in biological tissues".
- scope: epithelial monolayer / closed vesicle; morphogenesis timescales; relevant only where intercalation is part of the claimed mechanism.

**24. Morphogenesis is quasi-static at the cell scale: junctional and cortical mechanics relax within seconds to minutes while growth and division take hours, so at every instant the tissue sits close to mechanical force balance. Shapes recorded out of force balance are the integrator's transient, not tissue shapes.**  `[typical]`

- constrains: `shape_energy_3d.relax_iters (searchable range (10, 90), default 30), eta, cap_frac and the global dt; against morphogen_growth_3d.rate, divide_3d.max_div_frac / max_div, and reconnect_t1_3d.every (all every=1 in round_44_base).`
- check: `Per frame: growth increment / cell volume << 1 (rate*dt of order 1e-3 or less), dividing fraction << 1, and residual max|force| after the relaxation sweep small relative to the typical junctional tension. Convergence test: doubling relax_iters must not change the trajectory.`
- if violated: "Growth, division and reconnection all fire every frame while the mechanics gets a fixed, small number of relaxation sweeps. Is the mesh at force balance when the geometry is measured? Give the residual force after relaxation, and show the trajectory is unchanged when relax_iters is doubled; if it is not, the recorded shapes are integrator transients and cannot be compared with tissue."
- source: Sarkar & Krajnc 2023, Eq. (5) and Methods (inertia neglected; first-order overdamped vertex dynamics); Zhang & Schwarz 2022 Eq. (2) (overdamped Brownian dynamics, equilibration for dt = 10000 before data collection); Sorichetti et al. 2026 Sec. 2.5.1 (vertex positions evolve by minimization of the cell elastic energy). The seconds-versus-hours timescale separation itself is UNCITED in these four papers.
- scope: epithelial monolayer / closed vesicle / 3D vertex model; morphogenesis timescales (hours-days) driven by growth and division; does not apply to deliberately driven, fast-deformation protocols, which must declare their strain rate.

_miner notes: WHAT I COULD NOT ESTABLISH

1. The critical shape index for a VORONOI model in 3D. `voronoi_tension_3d` cites Merkel & Manning 2018, but that paper is not in papers/ and Zhang & Schwarz refer to it only as "a rigidity transition in a three-dimensional Voronoi model as a function of the three-dimensional shape index" with no number quoted. So my 5.39 (Zhang, 3D vertex, disordered, with reconnection_

## from: /workspace/Plexus/papers/okuda.pdf (15 pp, full text extracted to /tmp/okuda.txt via pypdf) — Okuda, Miura, In

**25. A cell divides only after it has grown: the volume at which it splits is strictly greater than the volume it was born at, so a lineage does not shrink from generation to generation.**  `[textbook]`

- constrains: `divide_3d.factor (2.0) x divide_3d.cycle_cv (0.4) — the trigger is vf >= factor*djit*Vbirth with djit ~ N(1, cycle_cv) clipped to [0.4, 1.8] (tyssue_ops3d.py _fresh_djit)`
- check: `factor * min(djit) > 1 strictly. With the clip floor 0.4 this requires factor > 2.5; at factor = 2.0 any cell drawing djit <= 0.5 (about 10% of cells at cycle_cv = 0.4) divides at or below its birth volume. Measurable: the distribution of (division volume / birth volume) must have support entirely above 1, and median cell volume must not decay across generations.`
- if violated: "About 10% of cells are being handed a division threshold below their own birth volume, so they divide the moment min_cycle expires without having grown at all — a lineage that halves every cycle. Is sub-birth-volume division a deliberate feature of this specimen? If so, say what stops the cells vanishing; if not, either raise factor above 1/min(djit) = 2.5 or lower cycle_cv so the clip floor is never reached."
- source: Okuda 2018 p.4 ("when v_eq increases up to v_th, the jth cell divides into two daughter cells with (1/2)v_th. Constant v_th is set to be (4/3)v_ref") — division volume is exactly 2x birth volume, deterministic; the variability lives in the growth RATE (lambda_ref Gaussian), not in the size threshold.
- scope: epithelial monolayer, closed vesicle, 3D vertex model, morphogenesis timescales (hours-days); proliferating (not quiescent) tissue

**26. When an epithelium grows, the added material becomes new cells, not bigger cells: mean cell volume stays roughly constant while cell number rises.**  `[textbook]`

- constrains: `the pairing of morphogen_growth_3d (rate, rho, cap, vth_frac, after_frame) with divide_3d (vcap, min_cycle, max_div_frac, after_frame). With rho = 0 the operator takes the `else` branch and clamps only the LINEAR scale s at cap (default 2.5), i.e. a volume scale of 15.6x; the per-cell v_eq cap at vth_frac*v_ref is gated behind `if self.rho > 0` and is therefore inactive in every ref_okuda_route / round_44 config.`
- check: `d(log N_cells)/d(log V_total) ~ 1 over the run and median cell volume drift < ~30%. Okuda's population lives in [2/3, 4/3] v_ref — a factor-2 band — and grows ~2,000 -> ~4,000 cells while the tubes form. Equivalently: max_j v_j / median_j v_j <= ~2 at all times.`
- if violated: "rho = 0 disables the v_eq cap, leaving only cap = 2.5 on the linear scale — a licence for one cell to reach 15x its neighbours' volume. Only divide_3d.vcap = 1.5 is holding cell size down, and vcap is a solver-side force-divide, not a growth law. Is hypertrophy the intended specimen? If not, report d(log N)/d(log V) and the cell-volume distribution, and state which mechanism is supposed to bound cell size."
- source: Okuda 2018 p.4 (v_th = (4/3) v_ref, daughters born at (1/2) v_th = (2/3) v_ref) together with the morphology-diagram figure caption ("The individual tissues were composed of about 4,000 cells", grown from the ~2,000-cell initial vesicle stated on p.7 and p.9).
- scope: epithelial monolayer, closed vesicle, 3D vertex model, morphogenesis timescales; proliferating tissue, no cell-death operator

**27. Molecules pass from one cell to the next through the junction the two cells share and are diluted into the receiving cell's cytoplasm, so transport depends on the contact area and on the cell volume — both of which the mechanics is continuously changing.**  `[textbook]`

- constrains: `cell_diffuse.implementation. Omitting the key (as config/okuda/ref_okuda_route.yaml and round_44_base.yaml both do) resolves to `graph_laplacian`, an unweighted neighbour average that reads neither shared area nor volume; the Okuda finite-volume form is `interface_weighted`.`
- check: `flux_ij must be proportional to s_ij / v_i with s and v recomputed every step. Direct test: freeze the chemistry and inflate one cell 2x — its concentration and its per-unit-volume influx must both change. Under graph_laplacian neither changes, i.e. the deformation -> patterning arm of the coupling is identically zero.`
- if violated: "With graph_laplacian the chemistry cannot see the geometry, so the coupling this recipe exists to study is one-way. Is the unidirectional model the intended specimen? If so, state that the run cannot test deformation -> patterning and withdraw any hysteresis or dilution claim; if not, set implementation: interface_weighted and re-run."
- source: Okuda 2018 Eq. (2) and Appendix A Eq. (A6): flux = mu * sum_k (m_k/v_k - m_j/v_j) * s_jk^cc / v_j, with modelling assumption (i) "the molecules transport within cells through junctional structures but not the outside of cells" and (iii) homogeneous gap width.
- scope: epithelial monolayer, closed vesicle, 3D vertex model; junctional (not extracellular) transport, no ECM

**28. Growing a cell dilutes what is already inside it — the cell does not manufacture morphogen in proportion to the volume it adds.**  `[textbook]`

- constrains: `morphogen_growth_3d.conserve_amount (must stay true; the operator rescales c <- c * (s_prev/s)^3 when v_eq grows as s^3)`
- check: `with cell_react and cell_diffuse disabled, sum_j c_j * v_j must be exactly constant under pure growth; per step, a volume scaling of s^3 must be matched by a concentration scaling of 1/s^3.`
- if violated: "With conserve_amount off, growth creates activator out of nothing precisely where the activator is already highest, because growth is gated on the activator — the tip becomes a self-feeding source and every bud or tube is a mass leak, not a Turing structure. Is volume-proportional production being declared as a kinetic term? If so put it in cell_react where it can be seen; if not, report the drift in sum_j c_j v_j over the run."
- source: Okuda 2018 Appendix A Eq. (A4): "the total number of molecules within individual cell compartments should be conserved unless any flux, production, and degradation, even while the cell compartments dynamically deform"; dilution effect discussed p.6 (gamma = 0.01 blurring).
- scope: epithelial monolayer, closed vesicle, 3D vertex model; growth present, no secretion/uptake operator

**29. A tissue on morphogenesis timescales has no inertia: at every instant the elastic forces on every vertex balance the drag, so the sheet is at mechanical equilibrium, not partway toward it.**  `[textbook]`

- constrains: `shape_energy_3d.relax_iters (fixed 30), .eta (0.08), .cap_frac (0.12) — a fixed-iteration unrolled relaxation with no residual test anywhere`
- check: `mean residual |grad U| at the end of the relaxation must fall below a declared threshold (Okuda: E_th = 1e-5); operationally, doubling relax_iters 30 -> 60 must not change the trajectory beyond tolerance, and the fraction of vertices whose step is clipped by cap_frac must be 0.`
- if violated: "relax_iters is a fixed count with no convergence test, so the force balance may never be reached. What is the measured mean residual at the end of the loop, and does the morphology change at relax_iters = 60? If it does, the tube shape is a property of the 30-iteration solver rather than of the tissue — declare the residual you are willing to accept and show it is met."
- source: Okuda 2018 Appendix C: "Eq. (1) is a self-consistent equation; it was solved by convergent calculations using the iterative method, in which the mean residual error should be under the threshold value E_th"; Table 2, E_th = 1e-5.
- scope: epithelial monolayer, closed vesicle, 3D vertex model, morphogenesis timescales (hours-days); overdamped, no inertia

**30. Neighbour exchange is the rarest thing an epithelium does: a junction shrinks essentially to a point before two cells swap partners, and between two such events the sheet re-equilibrates.**  `[textbook]`

- constrains: `reconnect_t1_3d.l_th_frac (0.28), .every (1, i.e. every mechanical frame), .max_flips (300)`
- check: `(i) threshold edge length / mean cell diameter <= ~0.05 (Okuda: delta_l_th = 0.05 with the cell diameter v_ref^(1/3) = 1); (ii) reconnection interval >> mechanical relaxation time (Okuda: delta_t_r / delta_t_v = 1.0 / 0.005 = 200); (iii) measured T1 events per cell per cell cycle = O(1), not O(100), and max_flips must never bind.`
- if violated: "l_th_frac = 0.28 flips a junction while it is still about a quarter of a cell across, and every = 1 attempts reconnection 200x more often relative to the mechanical step than Okuda does. How many T1s per cell per cell cycle does this run actually make, and does max_flips = 300 ever bind? If the rate is not O(1)/cell/cycle the specimen is a churning fluid and any shape it holds is set by the flip rate — declare that."
- source: Okuda 2018 Table 2 (delta_t_v = 0.005, delta_t_m = 0.0005, delta_t_r = 1.0, delta_l_th = 0.05) and Appendix C; reversible network reconnection model, Okuda et al. 2013 (ref. 29).
- scope: epithelial monolayer, closed vesicle, 3D vertex model; confluent tissue with cell rearrangement

**31. Growth is far slower than shape relaxation: a cell takes much longer to double than the sheet takes to settle into a new equilibrium shape, so the tissue is quasi-static with respect to growth.**  `[textbook]`

- constrains: `divide_3d.min_cycle (16 frames) and morphogen_growth_3d.rate (0.01 per frame) versus the mechanical relaxation delivered by shape_energy_3d.relax_iters * eta per frame`
- check: `tau_cycle / tau_mech >= ~50 (Okuda: tau_cycle = 50 in units of eta_c/kappa_s). Measure tau_mech by freezing growth and fitting the e-folding time of the total energy, then express the observed cell-cycle time in the same units.`
- if violated: "If the cell cycle is not much longer than the mechanical relaxation time, the tissue never reaches equilibrium between growth increments and the morphology is a transient of the growth rate rather than a shape the tissue holds. Report measured tau_mech and tau_cycle; if the ratio is below ~50, either slow the growth or declare this an explicitly non-quasi-static regime and predict what the departure changes."
- source: Okuda 2018 p.5: "By assuming a quasi-static process, the cell cycle tau_cycle was set to be much larger than the characteristic time of cell deformation, represented by eta_c/kappa_s"; Table 1 (tau_cycle = 50 eta_c/kappa_s).
- scope: epithelial monolayer, closed vesicle, 3D vertex model, morphogenesis timescales (hours-days)

**32. Cytoplasm is essentially water: a cell changes volume by moving material across its membrane, not by being elastically squeezed, so on these timescales cell volume tracks its target almost exactly.**  `[textbook]`

- constrains: `shape_energy_3d.K_V (6.0) relative to shape_energy_3d.kappa_s (0.2) — the volume term is a constraint penalty, not a soft spring`
- check: `K_V / (kappa_s * v_ref^(2/3)) >> 1 (Okuda: k_v = 10 against kappa_s = 0.2, a ratio of 50) AND, measurably, |v_j / v_eq,j - 1| < 0.05 for essentially all cells at all times.`
- if violated: "If cells are measurably compressible, growth can be absorbed as compression instead of shape change and the reported tube is soft in a way tissue is not. Report the distribution of v_j/v_eq,j; if the 95th-percentile deviation exceeds 5%, either raise K_V or declare the specimen compressible and say what physical process supplies the compressibility."
- source: Okuda 2018 p.5: "By assuming an incompressibility, the cell volume elasticity kv was set to be much larger than the characteristic surface energy of a cell, represented by kappa_s (v_ref)^(2/3)"; Table 1 (k_v = 10, kappa_s = 0.2).
- scope: epithelial monolayer, closed vesicle, 3D vertex model, morphogenesis timescales; no osmotic or lumen-pressure operator

**33. A mitogen threshold only produces shape if the pattern actually crosses it: it must sit between the low and high morphogen levels the tissue really reaches, or every cell grows (uniform inflation, no morphogenesis) or none does (a frozen body).**  `[textbook]`

- constrains: `morphogen_growth_3d.a_sw (1.5) relative to the fixed points of cell_react: gierer_meinhardt with gm_rho = 1, mu_a = 1, mu_h = 1, a0 = 0.01`
- check: `a_low < a_sw < a_high, ideally at the midpoint. For this GM form (da/dt = gm_rho*a^2/h - mu_a*a + a0, dh/dt = gm_rho*a^2 - mu_h*h) the nontrivial homogeneous state is a* = (mu_h + a0)/mu_a = 1.01 and the low state is a0/mu_a = 0.01 — so a_sw = 1.5 lies ABOVE the upper fixed point. Instrument: report max_j a_j and the fraction of cells with hill(a_j) > 0.5 at steady state; that fraction must be strictly between 0 and 1.`
- if violated: "a_sw = 1.5 sits above the homogeneous activator fixed point a* ~ 1.01, so all growth is riding on the overshoot of the Turing peaks above that fixed point. What is the measured max_j a_j, and what fraction of cells ever exceed a_sw? If that fraction is 0 the tissue cannot grow at all; if it is 1 the vesicle merely inflates. Report both, or move a_sw to the midpoint of the measured bimodal activator distribution."
- source: Okuda 2018 p.5: the steady states are (0,0) and (rho_u v_ref, rho_u v_ref), and "the switching concentration for cell growth was simply set to be the medial value of the steady state solutions rho_sw = (m1 + m2)/2 v_ref"; Table 1 (rho_sw = 0.5, rho_u = 1).
- scope: epithelial monolayer, closed vesicle, 3D vertex model; activator-inhibitor (Gierer-Meinhardt) mitogen gating

**34. A cell is well mixed inside, so the smallest chemical structure a tissue can carry is one cell wide; a Turing domain one or two cells across is the mesh, not a pattern, and the tissue must be many wavelengths across for the pattern to be intrinsic rather than set by the geometry.**  `[textbook]`

- constrains: `cell_diffuse.d_h and .chi (which set the domain size, ~ chi^(1/4) phi^(1/2)) together with seed_mesh_3d.n_cells (500 in ref_okuda_route)`
- check: `measured cells-per-activator-domain >= ~3 (Okuda chooses the inhibitor diffusivity phi = 10 precisely as the square of a 3-cell length scale, (3 v_ref^(1/3))^2 / (4 eta_c / 5 kappa_s)), AND the vesicle must hold several domains (Okuda: ~200 cells for arrested patterning tests, ~2,000 cells for every coupled run).`
- if violated: "How many cells wide is one activator spot, and how many spots fit on the vesicle? If a spot is 1-2 cells, the 'pattern' is the discretisation and the tube diameter it sets is just the cell diameter. With n_cells = 500 on a closed vesicle, state how many wavelengths fit and why that is enough — or raise n_cells toward Okuda's 2,000."
- source: Okuda 2018 p.5 (phi set to the square of the length scale of 3 cells; steady activator pattern length scale proportional to chi^(1/4) phi^(1/2)), p.7 and p.9 (all coupled runs use ~2,000-cell vesicles), Discussion p.11 ("the applicable area of the method is limited on a single cell level; intracellular patterning can be expressed over a length scale of cell-cell boundaries").
- scope: epithelial monolayer, closed vesicle, 3D vertex model; discrete (cell-compartment) reaction-diffusion

**35. Cells fill space without gaps or overlaps and every cell has strictly positive volume; a cell of zero or negative volume means the tissue surface has passed through itself, which no tissue does.**  `[textbook]`

- constrains: `cell_diffuse.vol_floor (0.05) and .w_cap (4.0) — guards that silently convert a collapsed or inverted wedge volume into a merely small one; also shape_energy_3d.antiinv`
- check: `min_j v_j > 0 at every frame AND the count of cells clamped by vol_floor must be 0, not merely bounded. Okuda has no such floor — Appendix C / Table 2 list only delta_l_th and E_th — so any floored cell is outside the reference model entirely.`
- if violated: "vol_floor turns an inverted cell into a small cell and the run continues, so a self-intersected mesh still produces a clean-looking movie. How many cells were floored, and at which frame did the first appear? If any, every morphology metric after that frame is undefined — declare whether the reported result postdates it."
- source: Okuda 2018 p.2 ("an individual cell shape is represented by a polyhedron, a boundary face between neighbouring cells expressed by a polygon, and the entire structure of a 3D cell aggregate expressed by a single network"); Appendix C / Table 2 (the complete list of numerical guards, which contains no volume floor).
- scope: epithelial monolayer, closed vesicle, 3D vertex model; confluent, non-self-intersecting

**36. Proliferation rate is set by the cell cycle, not by an administrative limit on how many cells may divide per timestep or how far a vertex may move — a tissue has no such quotas.**  `[textbook]`

- constrains: `divide_3d.max_div (30), .max_div_frac (0.0075), .vcap (1.5, a force-divide override); reconnect_t1_3d.max_flips (300); shape_energy_3d.cap_frac (0.12)`
- check: `the per-frame fraction of steps on which each cap binds must be 0. Okuda has none of these: a cell divides exactly when v_eq reaches v_th, reconnection is attempted for every edge and trigonal face, and vertex displacement is limited only by convergence to E_th.`
- if violated: "If max_div, max_div_frac, vcap, max_flips or cap_frac ever binds, the growth rate, the rearrangement rate or the vertex speed is being set by the throttle rather than by rate / a_sw / l_th_frac. Report the per-frame binding fraction for each; where it is nonzero, either raise the cap or adopt the throttle as part of the declared model and re-derive the effective growth rate from it."
- source: Okuda 2018 p.4-5 (division on reaching v_th) and Appendix C / Table 2 (no division cap, no flip cap, no displacement cap).
- scope: epithelial monolayer, closed vesicle, 3D vertex model, morphogenesis timescales; proliferating tissue

**37. When a cell divides its contents are partitioned between the daughters: each daughter starts at the mother's concentration, so division redistributes morphogen but does not create it.**  `[textbook]`

- constrains: `divide_3d (all implementations) via divide_3d.cell_set — the chem state written to the two fresh daughters`
- check: `at every division c_d1 = c_d2 = c_mother and m_d1 + m_d2 = m_mother, given v_d1 + v_d2 = v_mother. Instrument: total morphogen amount sum_j c_j v_j must show zero step change across division events.`
- if violated: "If each daughter inherits the mother's AMOUNT rather than her concentration, every division doubles the morphogen locally — and divisions happen preferentially where the activator is high, so proliferation would amplify the very pattern it is supposed to be driven by. Report the step in sum_j c_j v_j across division events; if it is nonzero, say which of the two conventions is intended."
- source: Okuda 2018 p.5: "In the division process, morphogen concentrations in daughter cells are set to be the same values with those of the mother cell."
- scope: epithelial monolayer, closed vesicle, 3D vertex model; discrete cell-compartment chemistry with division

**38. A monolayer vesicle floating in medium is attached to nothing: drag acts between a cell and the cells around it, so translating or rotating the whole tissue costs no energy and nothing pins it in the laboratory frame.**  `[textbook]`

- constrains: `general.boundary (free), general.world ([80, 80, 80]), and the friction model in shape_energy_3d (.eta, .mu) — whether drag is taken relative to the local tissue velocity field or to the lab frame`
- check: `net force and net torque on the tissue ~ 0; centre-of-mass drift over the run << one cell diameter; and the tissue bounding box must stay strictly inside `world` at all times.`
- if violated: "If drag is referenced to the lab frame, the medium is silently acting as an ECM that resists tissue translation and rotation — a substrate this preparation does not have; and a tube that reaches the world box is being shaped by the box. Report COM drift and the maximum bounding-box extent against world = 80. If either is significant, declare the substrate you are modelling."
- source: Okuda 2018 Eq. (1) and Eq. (4): eta_i (dr_i/dt - V_fi) = -grad_i U, with V_fi "the average velocity of the surrounding cells" and eta_i = sum over surrounding cells of eta_c — drag is relative to the neighbours, never to a fixed frame.
- scope: closed monolayer vesicle in suspension, 3D vertex model; no ECM, no substrate, no external body force

**39. Cell-cycle durations in a proliferating epithelium are variable but tightly distributed (order 10%), and in this model the disturbance that division injects into the morphogen field is itself the branching mechanism — so the noise level is a mechanism, not a free knob.**  `[typical]`

- constrains: `divide_3d.cycle_cv (0.4) and divide_3d.reset_noise (0.12)`
- check: `cycle_cv ~ 0.1 (Okuda: tau_sd = 0.1 tau_cycle, giving an inverse-Gaussian cycle-time distribution). If a run uses more, the branching morphology must be shown to survive at cycle_cv = 0.1. The variability must also enter as growth-RATE spread (which is what makes the timing distribution inverse-Gaussian), not as a jittered volume threshold.`
- if violated: "cycle_cv = 0.4 is four times Okuda's tau_sd/tau_cycle = 0.1, and the paper attributes branching to precisely this kind of division-induced disturbance of the pattern. Is the branching in this run a Turing/hysteresis effect or a division-noise effect? Re-run at cycle_cv = 0.1 and report whether the branches survive; and state why the CV is applied to the size threshold rather than to the growth rate."
- source: Okuda 2018 p.4 (lambda_ref drawn from a Gaussian; "constant tau_sd is empirically determined as 0.1 tau_cycle"; cell-cycle periods inverse-Gaussian) and p.9 ("there tend to remain a disturbance in molecular distribution by cell growth and division at a single cell level, which repatterns activator domains to be separated to form branch structures").
- scope: epithelial monolayer, closed vesicle, 3D vertex model; proliferating tissue with stochastic cell cycle

**40. Whether an epithelium holds a shape or flows is decided by where its target shape index sits relative to the rigidity transition — a fluid sheet rearranges away any shape imposed on it, a solid one buckles instead.**  `[typical]`

- constrains: `shape_energy_3d.p0 (3.9), .K_P, .Gamma, .Lambda; divide_3d.p0 (default 3.72)`
- check: `declare which side of the transition the specimen is on and show the tissue behaves accordingly. Citable landmarks: p0*(regular hexagon, unit area) = sqrt(8/sqrt(3)) ~ 3.7224 for the 2D compatible/incompatible transition (Kim, Zhang & Schwarz 2024, Eq. 1 and Appendix A), and s0* = 5.39 +/- 0.01 for the 3D bulk model (Zhang & Schwarz 2022, abstract). Measurable discriminator: the spontaneous T1 rate under zero growth — for a solid sheet it should be ~0.`
- if violated: "p0 = 3.9 is above the hexagonal compatible/incompatible value 3.72, so cells can meet both targets at zero energy and the sheet has no shear restoring force; the only things holding the tube's shape are the volume term and the T1 rate. Okuda's energy has no perimeter term at all, so this is an addition to the reference specimen. Is a fluid epithelium intended? Declare it and report the zero-growth T1 rate."
- source: Okuda 2018 Eq. (3) (the energy is volume elasticity plus surface energy ONLY — no perimeter or shape-index term, hence no rigidity transition in the reference model); thresholds from /workspace/Plexus/papers/Kim_Zhang_Schwarz_2024_3d_vertex_moduli.pdf and /workspace/Plexus/papers/Zhang_Schwarz_2022_tvm_3d_vertex.pdf.
- scope: epithelial monolayer, closed vesicle, 3D vertex model WITH an added perimeter/shape-index term; confluent tissue permitting T1s

_miner notes: WHAT I COULD NOT ESTABLISH, AND WHY.

1. There is no second reference paper. okuda.pdf and Turing_Vertex.pdf are byte-identical (md5 623d592156e80b59c2167b367894a900). Every premise above rests on one 15-page Scientific Reports paper; nothing here is cross-checked against an independent description of the same model.

2. The brief's framing of "Table 2 (the timescale separation: chemistry << mecha_

## from: CODE (read in full): /workspace/Plexus/prototype/Tyssue/tyssue_ops3d.py; /workspace/Plexus/prototype/Tyssue/ty

**41. A free epithelial vesicle floating in medium has no preferred distance from any point in space: its shape is set by its own surface mechanics and the fluid it encloses, so any restoring force toward a fixed radius corresponds to a named physical confinement (vitelline membrane, ECM capsule, gel) and to nothing else.**  `[textbook]`

- constrains: `shape_energy_3d.K_R (and the m["R0"] it is measured against). Hard-wired to 0.02 in translate._emit_shape_energy; it is absent from OPERATORS["shape_energy_3d"]["params"], so it is not theta, not in comp_hash, not perturbable by sample_params, and not removable by any legal one-edit.`
- check: `K_R == 0 unless an external confinement is declared, with its stiffness AND its centre. If declared, R0 must track the tissue: report the share of total U (and of the force on the 5% most protruded vertices) carried by the K_R term each frame.`
- if violated: "Every vertex is on a spring to |x| = R0 about the WORLD ORIGIN, and morphogen_growth_3d updates A0/P0/V0f but never R0 -- so in the okuda_route composition (local growth, no vesicle_growth) R0 stays frozen at the seed radius for the whole run. At r = 3*R0 that spring pulls each vertex inward with force 2*K_R*(r-R0) = 0.40, versus 0.20 for the Lambda line tension and ~0.26 for a 10% cell-volume error at K_V = 6. Is there a vitelline membrane in this experiment? If not, declare K_R = 0 and re-run every claim about whether growth alone can tubulate. If yes, expose K_R as theta so the confinement can be ablated like `extrude` is -- and note that `=shape_energy_3d:monolayer` currently removes K_R and Lambda silently along with the volume definition, so that 'one edit' is three."
- source: Okuda 2018 Eq. (3): U = sum_j [ (1/2) k_v (v_j - v_eq_j)^2 + kappa_s s_j ]. The effective energy contains only cell volume and cell surface terms; there is no term in the vertex position r_i.
- scope: closed epithelial vesicle / 3D vertex model / suspension culture (no ECM, no substrate) / morphogenesis timescales

**42. A cell's volume is unchanged when the cell merely moves: rigid translation adds no material, so it cannot advance the cell cycle or trigger division.**  `[textbook]`

- constrains: `divide_3d's trigger `vf >= factor*djit*Vbirth`, its `Vbirth`/`vcap`/`v_ref`, and morphogen_growth_3d's `vth_frac` -- all computed from face_geometry_3d's wedge volume v_f = (1/3)(cen_f . N_f), which is proportional to the distance from the world origin (v ~ r*A/3).`
- check: `Whatever volume measure drives division must satisfy V(x + c) = V(x) for any translation c. Screening test with no simulation: for a fixed mesh, translate the whole shell by one radius and confirm the per-cell volumes and the division candidate list are unchanged. Runtime test: with growth rate = 0 and the extrusion/driver on, the division count must be zero.`
- if violated: "A cell riding a bud outward from r = 5 to r = 10 doubles its wedge volume with no change of shape, so it crosses the 2 x V_birth threshold having grown nothing -- a tube would proliferate at its tip at growth rate zero, and K_V would then 'defend' the inflated volume. Is division meant to be driven by radial position? If not, use a translation-invariant volume (A_mid * h, as shape_energy_3d:monolayer already computes) for both Vbirth and the trigger. If the wedge is kept, run the rate = 0 control and report how many divisions it produces before any tube claim is recorded."
- source: UNCITED for the kinematic statement (Galilean invariance of a material volume). Okuda 2018 divides on v_eq reaching v_th where v_j is the true polyhedral cell volume (Eq. 3, Mechanochemical coupling section), which is translation-invariant.
- scope: closed epithelial vesicle / 3D vertex (apical-surface) model / any run with divide_3d present

**43. In an epithelial vesicle the cells are a thin rind of tissue and the cavity they enclose is lumen fluid; the two are different compartments with different drivers (biosynthesis and nutrient uptake versus transepithelial ion-and-water pumping), and cell material is a small fraction of the enclosed space.**  `[textbook]`

- constrains: `seed_mesh_3d's V0f / V0 / v_ref and shape_energy_3d.K_V in the default implementation -- the per-cell wedge volumes v_f are pyramids from the shell centre, so sum_f v_f is EXACTLY the enclosed cavity, i.e. the 'cell volumes' tile the lumen.`
- check: `Report the implied cell thickness h_f = 3 v_f / A_f. It must equal a declared epithelial thickness, be roughly uniform (max_f h_f / min_f h_f < 2), and NOT scale with the vesicle radius; and sum_f v_cell must be strictly less than the enclosed cavity volume. At the shipped seed (R = 5, 500 cells) h = R/3 = 1.67 against a cell width of 0.79 (aspect 2.1), and h grows in proportion to R as the shell inflates.`
- if violated: "Here sum_f v_f IS the enclosed cavity, so 'per-cell volume elasticity' is partitioning lumen fluid among the cells and 'growth' is pumping the lumen -- a reader cannot distinguish tissue mass added from fluid pumped in. Which compartment is your growth adding to? Declare it, state the implied cell thickness 3v/A and whether the epithelium is really supposed to thicken in proportion to vesicle radius, or switch to shape_energy_3d:monolayer where v = A_mid * h."
- source: Okuda 2018 Eq. (3) and Results: v_j is the polyhedral volume of the jth cell and the specimen is 'a spherical vesicle of a monolayer cell sheet', so the enclosed lumen is not cell material. Numeric band for h/cell-width is UNCITED.
- scope: closed epithelial vesicle (monolayer) / 3D apical-surface vertex model / no lumen-pressure organ

**44. An epithelial cell can change its apical area while holding its volume -- apical constriction and columnarization -- and this is the principal way an epithelial sheet bends; apical area is not a fixed function of cell volume.**  `[textbook]`

- constrains: `morphogen_growth_3d's locked scaling A0 = A0_init*s^2, P0 = P0_init*s, V0f = V0f_init*s^3; divide_3d's `iso` rule (a0d = A0/V0f^(2/3) * half^(2/3)) under g1_ramp; and tyssue_monolayer's h_cell = full(h0), a single global constant ('v1: uniform fixed thickness').`
- check: `The per-cell ratio A_apical / v^(2/3) must be a free variable: measure its spread across cells and its change from seed to bud. If it is constant by construction -- and it is in BOTH implementations, by A0 ~ v_eq^(2/3) in the default and by fixed h in the monolayer -- then apical constriction is excluded and must be declared as an exclusion.`
- if violated: "Can any cell in this composition constrict apically at constant volume? With A0 slaved to v_eq^(2/3), or thickness a global constant, no. A finding of the form 'growth alone cannot bud/tubulate' would then be a result about a tissue from which the standard sheet-bending mechanism has been removed. State the exclusion and predict its consequence, or free A0 from v_eq / make h a per-cell state variable."
- source: Okuda 2018 Eq. (3): the energy has NO target area and NO target perimeter -- only volume elasticity plus a linear surface tension kappa_s*s_j -- so in the model being reproduced the apical/lateral area split is emergent, not imposed. Apical constriction as the sheet-bending mechanism: UNCITED here (Odell et al. 1981; Martin & Goldstein 2014, not in papers/).
- scope: epithelial monolayer / closed vesicle / 3D vertex model / morphogenesis timescales

**45. Which morphology a growing, Turing-patterned vesicle adopts -- undulation, tubulation or branching -- is set by the RATIO of the patterning rate to the growth (cell-cycle) rate, not by either rate on its own.**  `[textbook]`

- constrains: `cell_react.rd_rate / gamma versus morphogen_growth_3d.rate and vesicle_growth.rate, jointly with ENGINE_DT. The engine integrates chemistry as chem += dt*rate*f (engine.py:624) while both growth operators apply their `rate` once per CALL and never read the `dt` that translate passes them, so the realised ratio is proportional to dt.`
- check: `Compute and report the dimensionless ratio gamma * tau_cycle for every run and place it on Okuda's morphology diagram. It must be invariant to the solver step: halving dt and doubling n_frames must leave the phenotype unchanged. As shipped it is not -- chemistry advances by dt*rate per frame, growth by rate per frame.`
- if violated: "Growth is per-frame and chemistry is per-unit-time, so the campaign's position on the gamma-chi morphology diagram is a function of the time step; a sweep over rd_rate at fixed growth rate is a sweep over that ratio, and an impossibility claim made at one point of it is not a claim about the mechanism. Is dt part of the mechanism? If not, express growth as a rate per unit time and re-anchor; either way report gamma*tau_cycle beside every phenotype."
- source: Okuda 2018 Abstract and Fig. 8: 'the morphological variety depends on the difference in time scales between patterning and deformation'; Table 1 varies gamma over 0.01-100 at fixed tau_cycle = 50 (eta_c/kappa_s); Figs. 5-7 show tubes at gamma ~ 1, branching at gamma = 0.01, undulation at gamma = 100 for the SAME chi.
- scope: growing epithelial vesicle with reaction-diffusion patterning / morphogenesis timescales (hours-days) / any composition containing both cell_react and a growth operator

**46. Proliferating epithelial cells divide at roughly twice their birth size, so a healthy epithelium's cell volumes span about a factor of two and a cell an order of magnitude larger than its neighbours is a different (hypertrophic or polyploid) cell type, not a big epithelial cell.**  `[textbook]`

- constrains: `morphogen_growth_3d.cap (default 2.5 on the LINEAR scale = 15.6x in volume; not emitted by translate and not present in the search vocabulary at all, so it is invisible), morphogen_growth_3d.vth_frac (translate emits 1.5; Okuda's is 4/3), and divide_3d.vcap / max_div / max_div_frac.`
- check: `Per frame: max_f v_f / median_f v_f <= ~2.5 and CV(v_f) <= ~0.4; and v_eq_f / v_ref must stay within [2/3, 4/3] times the declared cell-cycle jitter. Report both as standing metrics, not as diagnostics run after something looks wrong.`
- if violated: "With rho = 0 the growth branch is the legacy one, which caps v_eq at cap^3 = 15.6x its initial value, so an activated cell's target volume is ~15x a non-activated cell's (whose target never moves at all). Is this specimen a mixture of a hypertrophic cell type and a quiescent one? If not, cap the target at Okuda's v_th = (4/3) v_ref and report the realised max/median -- and note that the throttles (max_div, max_div_frac) can only make the realised spread worse by queueing ready cells while they keep ramping."
- source: Okuda 2018, Mechanochemical coupling: 'when v_eqj increases up to v_th, the jth cell divides into two daughter cells with (1/2)v_th. Constant v_th is set to be (4/3)v_ref as if the median of all v_eqj is around v_ref' -- i.e. volumes cycle in [2/3, 4/3] v_ref, a factor-of-two span.
- scope: proliferating epithelial monolayer / closed vesicle / no hypertrophy or endoreplication operator exists

**47. A chemical prepattern a tissue can act on spans several cell diameters; a pattern whose wavelength approaches one cell is a property of the cell lattice rather than a tissue-scale morphogen domain, and cannot define the wall of a bud or tube.**  `[textbook]`

- constrains: `cell_diffuse.d_a / d_h / chi (both implementations measure length in CELL HOPS, so the diffusion length comes out directly in cells) together with cell_react.mu_h / mu_a.`
- check: `Primary, measured: cells per connected activator domain >= ~5, and number of domains >= 2, at the time growth is switched on. Secondary, screening from theta: sqrt(d_h*chi/mu_h) >= ~3 and sqrt(d_a*chi/mu_a) >= ~1 in cell units (up to an O(1) stencil factor), with d_h/d_a >> 1 for the Turing instability to exist at all. At the shipped defaults (d_a = 0.02, d_h = 0.7, chi = 4, mu = 1) these evaluate to 1.67 and 0.28 cells.`
- if violated: "An activator range below one cell means the activator never reaches the neighbouring cell: the 'spot' is a single cell and the protrusion it drives is one cell wide, which is a lattice artefact and not a tube. Is a single-cell activator domain the intended pattern? If not, raise chi*d_a until the activator range is at least a cell and the inhibitor range about three (Okuda's own basis for phi), keeping dt*chi*max(d_a,d_h) <= 1, and report cells-per-domain alongside the morphology."
- source: Okuda 2018 Table 1: the inhibitor diffusivity phi is set from a THREE-CELL length scale, basis '(3(v_ref)^(1/3))^2 / (4 eta_c / 5 kappa_s)'; Appendix A(i): 'since the size of the pattern is limited over the single cell, the intracellular structure of the pattern cannot appear'.
- scope: cell-discrete Turing patterning on an epithelial sheet / closed vesicle / activator-inhibitor kinetics

**48. Tissue is a physical body: two parts of the same epithelium cannot occupy the same space -- when a bud folds back on the shell or two branches meet, they contact and adhere, they do not pass through one another.**  `[textbook]`

- constrains: `the entire vertex mechanics (shape_energy_3d, rd_interface_tension, divide_3d's local relax): there is no contact, collision or steric term anywhere in prototype/Tyssue, and it also constrains every shape metric computed on the result (run_one.frame_metrics `protr` = r95/rmed, r95, and the tube_len/tube_diam family).`
- check: `No pair of non-neighbouring faces intersects. Cheap proxy recorded every frame: min over non-adjacent cell-centroid pairs of |c_i - c_j| > ~0.5 * median cell width; report the first frame it is breached and truncate the run there rather than scoring it.`
- if violated: "If the shell self-intersects, the morphology metrics describe a surface no tissue could adopt, and a 'branching' result may be one branch passing through another -- seed_cell_rd allows up to 8 activation cones on one vesicle, which makes approach likely. Is contact declared unnecessary because the branches never come near each other? Then show the minimum non-neighbour distance over the run; if it goes to zero, the branch morphology is not evidence."
- source: UNCITED for the steric statement (impenetrability of matter). Okuda 2018 Fig. 6 shows branching in which the branches remain separated, so the published target never tests interpenetration either.
- scope: closed epithelial vesicle / 3D vertex model / any run producing protrusions, buds or branches

**49. An epithelial sheet is confluent and continuous: it has no holes and no free edges, and every cell-cell junction is shared by exactly two cells -- a hole in an epithelium is a wound, and a wound is a declared experiment.**  `[textbook]`

- constrains: `divide_3d's rebuilt half-edge table (E_srce/E_trgt/E_face), and every consumer of ShapeEnergy3D._twin_faces -- whose documented behaviour is 'fallback to self (no penalty) if no twin' -- including the K_bend dihedral term, cell_diffuse:interface_weighted (`shared = (twin != ef)`) and rd_interface_tension's interface ring.`
- check: `After every topology-changing operator: V - E + F == 2, every undirected edge has exactly two incident faces, every ring has >= 3 distinct vertices. tyssue_topology_ops3d._check_closed already computes exactly this; reconnect_t1_3d calls it, divide_3d does not, and no campaign metric in discovery/ records it (grep for 'euler' over discovery/*.py returns nothing).`
- if violated: "An unpaired half-edge silently becomes a perfectly insulating cell wall: no morphogen crosses it, no bending penalty applies, and no metric changes -- so a torn sheet is indistinguishable from an intact one in everything the campaign records. Is a free edge intended (a cut or wound experiment)? If not, assert Euler == 2 after each division and fail the run; a fallback here converts a broken specimen into a quiet result."
- source: Okuda 2013 (Reversible Network Reconnection, ref. 29 of Okuda 2018): topological operations are defined to preserve a closed, orientable cell network. Confluence of epithelia: textbook cell biology, UNCITED here.
- scope: closed epithelial vesicle / 3D vertex model / any run with division or T1

**50. An epithelium relaxes elastic stress in seconds to minutes while it grows and divides over hours, so during morphogenesis the tissue is at (or very near) mechanical force balance at every instant -- the shape observed is an equilibrium shape, not a viscous transient.**  `[textbook]`

- constrains: `shape_energy_3d.relax_iters (box 10-90) with eta and cap_frac (hard-wired 0.08 / 0.12 by translate), against the growth increment applied per frame by morphogen_growth_3d.rate and the division budget max_div / max_div_frac.`
- check: `At the end of each frame's relaxation: max_i |grad U|_i normalised by (kappa_s * median edge length) < ~1e-3 (Okuda solves Eq. 1 to residual E_th = 1e-5); the fraction of vertices still clipped by the cap_frac displacement limit -> 0; and the energy drop over the last relaxation iteration < 1% of the drop over the first. Separately, tau_cycle >> the mechanical relaxation time (Okuda: tau_cycle = 50 eta_c/kappa_s).`
- if violated: "If the relaxation has not converged when the next growth increment lands, the trajectory is set by the MOBILITY model rather than by the energy -- and the mobility here is uniform per vertex, whereas tissue drag scales with the number of adjacent cells (Okuda Eq. 4: eta_i = sum over surrounding cells of eta_c). Is the transient part of the claim? If not, report the per-frame residual and raise relax_iters until it plateaus. The existing Q test only relaxes the END state for 60 frames; it says nothing about force balance during the run."
- source: Okuda 2018, Physical parameter setting: 'By assuming a quasi-static process, the cell cycle tau_cycle was set to be much larger than the characteristic time of cell deformation, represented by eta_c/kappa_s' (Table 1: tau_cycle = 50 eta_c/kappa_s); Table 2: E_th = 1e-5; Eq. (4) for the friction.
- scope: epithelial monolayer / closed vesicle / morphogenesis timescales (hours-days), quasi-static regime

**51. Cell-cycle durations in a proliferating epithelium are variable but tightly distributed -- of order 10-30% of the mean, not a factor of two.**  `[typical]`

- constrains: `divide_3d.cycle_cv (vocabulary box 0.05-0.5, default 0.40; clamped in _fresh_djit to [0.4, 1.8]) and seed_mesh_3d.vseed_cv (default 0.15).`
- check: `cycle_cv <= ~0.2 unless a mixed population is explicitly declared; and the realised distribution of `age` at division must have a CV in the same band. Report the realised CV, not just the parameter.`
- if violated: "cycle_cv = 0.40 with the clamp at [0.4, 1.8] means one cell in twenty is triggered at 0.4x and another at 1.8x the reference threshold -- a 4.5-fold spread in cell-cycle length. Is this a stem/differentiating mixture? If the CV is set high to break up a synchronous division wave the relaxation cannot absorb, declare that: it is then a solver parameter wearing a biological name, and the wave, not the CV, is the thing to fix."
- source: Okuda 2018, Mechanochemical coupling: cell-cycle periods follow an inverse Gaussian of mean tau_cycle and standard deviation tau_sd, with 'constant tau_sd empirically determined as 0.1 tau_cycle'. The 10-30% in-vivo range is UNCITED.
- scope: proliferating epithelial monolayer / closed vesicle / a single cell type

**52. On the apical surface of an epithelium cells are polygons with about six neighbours -- essentially all cells have four to eight -- and junctions meet in threes (tricellular junctions), with four-fold vertices only as transient rearrangement intermediates.**  `[textbook]`

- constrains: `the mesh produced by seed_mesh_3d, divide_3d (divide_face_3d adds a vertex to each of the two neighbours whose shared edge is split) and reconnect_t1_3d; and, downstream, cell_adjacency, whose neighbour graph is the RD's entire notion of space.`
- check: `Every frame: mean face degree = 6 - 12/F (forced by Euler for a 3-valent closed shell, so the content is the distribution, not the mean); fraction of cells with degree < 4 or > 9 below ~2%; fraction of vertices of valence != 3 near zero and non-growing.`
- if violated: "A tissue whose cells have twelve neighbours is not an epithelium, and it would still produce a perfectly respectable protr = r95/rmed -- nothing in the campaign measures polygon degree. Is the degree distribution supposed to broaden here? If the tip cells of the tube are accumulating neighbours because divisions add vertices to their ring faster than T1 removes them, say so and report the distribution, because the RD is diffusing on that same graph."
- source: UNCITED in papers/ (Lewis 1928; Gibson et al., Nature 442:1038, 2006 -- the ~45% hexagon, 4-8 sided distribution; neither is in the local corpus). Okuda 2018 uses the RNR framework (ref. 29) in which stable vertices are 3-valent and 4-fold configurations are reconnection intermediates.
- scope: epithelial monolayer apical surface / closed vesicle / 3D vertex model

**53. A molecular concentration in a cell is non-negative, and a mitogen at zero concentration produces zero growth -- never negative growth.**  `[textbook]`

- constrains: `the integrated cell.chem block (engine.py:624, chem += dt*d) fed by cell_diffuse and cell_react, and read by morphogen_growth_3d's Hill term a**hill / (a_sw**hill + a**hill + 1e-12) and by divide_3d's orient_asw / rd_interface_tension's a_sw thresholds.`
- check: `min over cells and frames of chem >= 0, asserted rather than assumed. The reaction operators clamp only their own READ of the state (CellReactGiererMeinhardt clamps a>=0, h>=1e-3); nothing clamps the state itself, and interface_weighted's row weights are capped at w_cap = 4, i.e. up to 4x the stencil gain the CFL is evaluated at (T5 says that gain is underived).`
- if violated: "A negative activator raised to a non-integer Hill exponent is NaN in torch (verified: torch.tensor(-0.5)**3.7 -> nan; the vocabulary's alpha default is 4.0 but the box is continuous over [1, 49.5]). The NaN propagates into mg_scale and V0f, and the force pass then nan_to_num's the gradient to zero -- so the run continues, the mesh freezes, and the composition is recorded as 'this mechanism does nothing'. Is a negative concentration meaningful in your model? If not, assert non-negativity on the state and treat a breach as an integrator failure, in the same idiom as T1_DIFFUSION_UNSTABLE: evidence about a solver, not a mechanism."
- source: UNCITED (definition of a concentration). Okuda 2018 Eq. (2) tracks molecule NUMBERS m_j, which are non-negative by construction.
- scope: cell-discrete reaction-diffusion on an epithelial cell set / any composition with cell_react + morphogen_growth_3d

_miner notes: WHAT I DELIBERATELY DID NOT ASSERT.\n\n1. Shape index / rigidity transition. I drafted a premise that a confluent epithelium sits near p* = P/sqrt(A) ~ 3.81 and that shape_energy_3d.p0 (box 3.4-4.2, default 3.90) should be checked against the REALISED shape index rather than merely set. I dropped it. p* = 3.81 is a result for a FLAT 2D vertex model with no volume term; on a closed curved shell wit_

## from: PAPERS (read via text extraction of the PDFs at /workspace/Plexus/papers/):
- Tissue_active_matter.pdf — Brück

**54. An epithelial sheet has a finite bending rigidity, and that rigidity is what selects how many wrinkles or buds a growing sheet makes — a sheet relaxes many small wrinkles into fewer large ones, and ultimately one buckle, because that has the lowest bending energy.**  `[textbook]`

- constrains: `shape_energy_3d K_bend — hard-coded to 0.0 in discovery/translate.py:114 for the `default` implementation (the `monolayer` implementation instead generates bending emergently from h0)`
- check: `K_bend > 0, OR shape_energy_3d.implementation == 'monolayer' with h0 > 0. And: the measured buckle/bud wavelength must be many cell diameters, not 1–2 (a mesh-scale wavelength is the signature of zero bending rigidity).`
- if violated: "K_bend = 0 gives the sheet no resistance to folding, so nothing physical sets the wavelength of the wrinkles or the number of buds — the mesh spacing does. Is a zero-bending-rigidity sheet the intended specimen? If so, declare it and state that bud COUNT is therefore not a measurable of this model; if not, either set K_bend > 0 or switch to the monolayer implementation and report the emergent kappa."
- source: Noguchi & Elgeti 2024, §III: "the number of wrinkles decreases through more frequent fusion, since the tissues can relax into a shape of lower bending energy. Eventually, a single buckle is formed ... since it has the lowest bending energy"; Appendix A measures kappa ≈ 50 eps0 for their tissue sheet and maps it to kappa ~ 1e-15 – 1e-11 J for real epithelia (their ref. [59] = Hannezo, Prost & Joanny 2014 PNAS 111:27).
- scope: epithelial monolayer / closed vesicle, 3D vertex model, morphogenesis timescales (hours–days); no ECM

**55. A confluent epithelium does not absorb added area by stretching its cells. Once area is added faster than the cells can rearrange or the boundary can expand, the excess leaves the plane — as wrinkles, buckles or buds — because bending is far cheaper than stretching.**  `[textbook]`

- constrains: `shape_energy_3d K_A (area stiffness, emitted as 1.0) and K_P, against vesicle_growth.rate and morphogen_growth_3d.rate; and the run's mean-cell-area trajectory`
- check: `over any window where mean cell area rises more than ~20% without matching division, out-of-plane deflection must also rise. Formally: d(area)/dt > 0 and d(deflection)/dt ≈ 0 and d(N_cells)/dt ≈ 0 cannot hold simultaneously.`
- if violated: "Cells are growing in-plane with no out-of-plane response and no division. What is meant to be resisting the areal strain here, and how large is it relative to the growth forcing? If the sheet is deliberately pinned flat (substrate, external tension), declare that boundary condition — it is currently not in the model."
- source: Noguchi & Elgeti 2024, §III and §VII: "Freely suspended sheets form wrinkles for rapid tissue growth"; "the growing tissue forms buds when the growth rate is too high to maintain the 2D density." Also Brückner & Hannezo 2025 p.9: quasi-static global expansion leads to "morphological instabilities such as buckling."
- scope: epithelial monolayer / closed vesicle, free or weakly adhered, no imposed external tension; morphogenesis timescales

**56. Cells in a sheet hold a preferred 2D density and restore it by sliding past one another; when rearrangement is blocked or slowed, growth is stored as density and stress instead of being relieved, and the sheet eventually detaches or buds.**  `[textbook]`

- constrains: `reconnect_t1_3d l_th (0.01–0.12, default 0.04) and the run's T1-event count per frame`
- check: `with growth active, T1 events per frame > 0 and local cell density stays within ~±15% of its initial value. If the T1 count is zero, density must be shown to be rising monotonically — and that rise is stored stress, not biology.`
- if violated: "No T1 events are firing while the tissue grows, so this sheet cannot restore its preferred density. Is a jammed / solid specimen intended? Declare it, and predict where the stored stress goes — buckling, detachment, or numerical divergence."
- source: Noguchi & Elgeti 2024, §IV and Fig. 5: at zero substrate friction "the cells can move sufficiently to maintain their preferred density", whereas friction "slows relaxation, resulting in high density and high stress in the center region", and "Budding occurs, when this high stress overcomes substrate adhesion."
- scope: epithelial monolayer, fluid regime; morphogenesis timescales

**57. An epithelial cell divides when it reaches a critical size — roughly twice its birth volume — so a healthy proliferating epithelium does not accumulate large populations of oversized, undivided cells.**  `[textbook]`

- constrains: `divide_3d max_div (4–480, default 30 per frame) and max_div_frac (default 0.0075 per frame) — the per-frame division budget; and divide_3d vcap (0.0–3.0, default 1.5), where 0 disables the hard size cap`
- check: `max over live cells of V/V_birth stays below ~2–3, and the fraction of frames on which the division budget binds (n_wanted > cap) is ~0. vcap > 0.`
- if violated: "The per-frame division cap is binding, so cells that have already passed their division size keep inflating instead of dividing. Is a tissue of several-fold oversized undivided cells the intended specimen? If not, raise max_div / max_div_frac until the cap stops binding, or set vcap > 0 so oversized cells always divide. Report max(V/V_birth) either way."
- source: Noguchi & Elgeti 2024, §II: "cells can grow (i.e., expand in volume) under a finite force and fluctuation, and the cells divide when a critical size is reached"; "When the distance rcl,k exceeds the threshold value rdiv, the k-th cell exhibits division into two daughter cells."
- scope: proliferating epithelium (monolayer or vesicle), morphogenesis timescales

**58. Cell number in a tissue changes only by division and death, and the death rate is not cosmetic — it selects which shape a growing sheet takes: many wrinkles at zero death, a single buckle at moderate death, pore-opening and net shrinkage above a threshold.**  `[textbook]`

- constrains: `the operator vocabulary itself — divide_3d exists but no apoptosis / cell-removal operator does (`extrude` is a radial push force, K_extrude, not a removal)`
- check: `if a steady-state or homeostatic morphology is claimed, a cell-removal term must exist in the composition. Otherwise every reported morphology must be labelled a transient on the zero-death branch.`
- if violated: "There is no way for a cell to leave this tissue, so cell number can only rise and no steady state exists. Is an unconstrained growth transient the intended specimen? Declare it — and note that the wrinkle/bud count you are measuring is a function of a death rate you have silently fixed at zero."
- source: Brückner & Hannezo 2025, Box 2: "a key conservation law is the local conservation of cell number, which can only change due to local flows or cell division and apoptosis." Noguchi & Elgeti 2024, §III and Fig. 2(b–e): at ka = 0 many wrinkles; at ka = 0.015 a single buckle; at ka = 0.017 the tissue opens a pore and disappears.
- scope: growing epithelial sheet / closed vesicle; scoped to the current vocabulary, in which no cell-death operator exists

**59. Tissues stop growing when they are compressed — division and apoptosis balance at a preferred "homeostatic pressure" — so a growth law that reads only a chemical signal, and never local density or stress, has no mechanism by which it can ever stop.**  `[textbook]`

- constrains: `morphogen_growth_3d rate / rho / a_sw (the growth gate reads the activator only) and divide_3d (no pressure or density input at all)`
- check: `growth rate must be a decreasing function of local compression: d(growth)/d(pressure) < 0. If it is not, the run must be explicitly declared pressure-blind.`
- if violated: "Growth here is blind to how crowded the cell is — contact inhibition is absent. Is that a deliberate ablation? If so, state it and predict the consequence (unbounded density, or unbounded cell size); if not, gate growth on local volume or pressure and say which."
- source: Brückner & Hannezo 2025, p.9: "density-dependent growth and/or apoptosis results in tissues having a preferred 'homeostatic pressure' at which division and apoptosis are balanced and behaving as viscous materials over long timescales (Ranft et al. 2010)."
- scope: proliferating epithelium, morphogenesis timescales; no ECM, no lumen pressure

**60. A cell's perimeter cannot be smaller than that of a circle of equal area (dimensionless index 3.545), and a cell packed into a confluent tiling cannot beat a regular hexagon (3.722); a target below those asks for a shape no cell can have, so the perimeter term becomes a permanent, never-satisfiable contraction. Separately, the index ≈3.81 divides solid from fluid.**  `[textbook]`

- constrains: `shape_energy_3d p0 — searchable box currently (3.4, 4.2), default 3.90; the engine sets P0 = p0 * sqrt(A0) (tyssue_ops3d.py:202, 581), i.e. the 2D convention`
- check: `p0 >= 3.722 for a confluent sheet (3.545 is the absolute floor). And the mechanical state must be declared, not left implicit: p0 < 3.81 → solid/jammed, p0 > 3.81 → fluid.`
- if violated: "p0 below 3.722 asks every cell for a perimeter no tiling cell can achieve, so every cell sits permanently in perimeter tension. Is a uniformly hyper-contractile jammed epithelium the specimen you intend? If yes, say so and predict the consequence; if not, raise the lower bound of the p0 box to 3.722."
- source: The geometric floors are isoperimetric (circle 2*sqrt(pi) = 3.5449; regular hexagon 6/sqrt(3*sqrt(3)/2) = 3.7224 — computed here, not taken from a paper). The rigidity threshold p* ≈ 3.81 is Bi, Lopez, Schwarz & Manning 2015, Nat Phys 11:1074, cited in Brückner & Hannezo 2025 p.9 as "a noise-free jamming transition controlled by a critical value of the preferred single-cell shape (shape index) in confluent tissues" — the review does NOT restate the number 3.81; that value comes from Bi et al. directly, which I did not open here.
- scope: confluent epithelial monolayer, 2D (apical-polygon) shape-index convention

**61. In three dimensions the same dimensionless shape quantity is surface / volume^(2/3): it is floored by the sphere at 4.836 and its rigidity transition sits near 5.41. These numbers are disjoint from the 2D range, so a value that is legal in one convention is nonsense in the other.**  `[textbook]`

- constrains: `voronoi_tension_3d s0 (default 5.4, prototype/Turing_vertex/vertex3d_ops.py:367) versus shape_energy_3d p0 (default 3.90) — two different fields, two different conventions`
- check: `s0 >= 4.836 always (below that, no solid body exists with that surface-to-volume ratio); p0 in [3.545, ~4.5]. No numeric value should ever be copied between the two fields.`
- if violated: "A shape index has been set outside its convention's isoperimetric floor, or copied between the 2D and 3D operators. Which convention does this operator use — P/sqrt(A) or S/V^(2/3) — and is the value above that convention's floor?"
- source: The 3D floor (36*pi)^(1/3) = 4.83598 is isoperimetric (computed here). s0* ≈ 5.41 is Merkel & Manning 2018, New J Phys 20:022002 — the REFERENCE string declared on the operator itself at prototype/Turing_vertex/vertex3d_ops.py:360.
- scope: 3D vertex / self-propelled-Voronoi cells (space-filling polyhedra)

**62. A closed epithelial vesicle cannot be tiled by hexagons. With three-way vertices the neighbour counts obey an exact topological identity — the sum over cells of (6 − n_neighbours) equals 12, the twelve pentagons of a football.**  `[textbook]`

- constrains: `seed_mesh_3d output topology (fibonacci_sphere, n_cells 150–2000) and every mesh-integrity metric read off that mesh`
- check: `sum over live cells of (6 − n_neighbours) == 12, exactly, for a closed simply-connected 3-valent mesh. Any other value means a pore, a handle, or non-3-valent vertices.`
- if violated: "The neighbour-count sum is not 12, so this is not a closed sphere of three-way vertices. Which is it — a pore, a handle, or 4-valent vertices left over from an incomplete T1? Declare it before any shape metric is read off this mesh."
- source: Brückner & Hannezo 2025, "Tissue curvature and topology", p.15: "a topological invariant called Euler's characteristic constrains the possible number of vertices, faces, and edges of any triangulation of a sphere. In the simplest case, where cells are either hexagons or pentagons, there must be 12 pentagonal cells in a spherical monolayer, just like the 12 black pentagons on a football."
- scope: closed vesicle / spherical monolayer, 3D vertex model with three-way vertices

**63. On a curved closed monolayer the solid/fluid threshold is not the flat-sheet one: it depends on the ratio of cell size to vesicle radius, so the same shape index can be solid at one cell count and fluid at another.**  `[typical]`

- constrains: `shape_energy_3d p0 jointly with seed_mesh_3d n_cells (150–2000) — and specifically the planned 10x-reservoir re-run at ~18,000 cells`
- check: `any solid/fluid claim must be re-measured at each n_cells (e.g. via T1 rate or a shear-modulus probe). p0 on its own is not a sufficient label for the mechanical state of a curved sheet.`
- if violated: "A p0 calibrated at N cells is being reused at 10N. Has the rigidity been re-measured at the new cell count, or is the fluid/solid label being carried across untested? If carried across, that is a claim, not a setting — say so."
- source: Brückner & Hannezo 2025, p.15: "the presence of curvature introduces an additional parameter, namely, how the size of a cell compares to the radius of the sphere. Based on spherical vertex models, this ratio has been predicted to control the jamming transition of spherical monolayers, suggesting a mechanism of how cells could rigidify by tuning the curvature of the domain (Sussman 2020)."
- scope: closed vesicle / spherical monolayer; does not apply to flat sheets

**64. The foam / energy-minimisation picture of a tissue only holds while cells relax their shape faster than they divide; if division outpaces shape relaxation the packing is never at force balance and the energy parameters stop meaning what they mean.**  `[textbook]`

- constrains: `shape_energy_3d relax_iters (10–90, default 30) against divide_3d min_cycle (2–64 frames, default 16) and max_div_frac`
- check: `the residual force at the end of the relax loop must fall below a stated tolerance before the next division call fires — i.e. t_relax << t_div (mean inter-division interval).`
- if violated: "Division is outrunning force balance, so these configurations are not quasi-static equilibria and p0 / K_V no longer carry their foam-model meaning. Report the residual force at the end of the relax loop, or declare the run explicitly non-quasi-static and stop interpreting p0 as a rigidity knob."
- source: Brückner & Hannezo 2025, p.9: "cellular packings in the early embryo can be described by these energy-minimization concepts only if the division is sufficiently slow compared to cell shape relaxation (Giammona and Campàs 2021)."
- scope: 3D vertex model of a proliferating epithelium, morphogenesis timescales (hours–days)

**65. A free epithelial edge carries a line tension, so a hole in a sheet is never at rest: small pores close, large pores grow, and a population of pores coarsens by Ostwald ripening and coalescence. A hole held at a fixed size is not a thing an epithelium does.**  `[textbook]`

- constrains: `shape_energy_3d Lambda (edge line tension, 0.0–0.3, default 0.20) and any reported holed / labyrinth / porous morphology`
- check: `Lambda > 0; and pore area must be monotone (closing under net growth, growing under net shrinkage) with the pore count falling over time. A constant pore size or constant pore count over many relaxation times requires a declared stabiliser.`
- if violated: "These pores are neither closing nor coarsening. What holds them open at fixed size — is Lambda zero, or is the run too short for edge tension to act? Declare which, and say whether the tissue is in net growth (pores should close) or net loss (pores should grow)."
- source: Noguchi & Elgeti 2024, §VI: "larger pores grow; however, small pores shrink through Ostwald ripening ... Coalescence of the pores also occurs"; Appendix A measures the edge line tension Gamma ≈ 13 eps0/rdiv and notes these values "are sufficiently large to maintain a flat tissue sheet with a minimum edge length (circular edge for disk-shaped tissue)."
- scope: epithelial monolayer with free edges or pores; the sign of the pore's motion is set by whether the tissue is in net growth or net shrinkage

**66. Two out-of-plane protrusions on one continuous sheet lower the sheet's total bending energy by coming together, so neighbouring buds and tubes drift towards each other and fuse. A persistent field of many separate, evenly-spaced buds is not the default outcome.**  `[contested]`

- constrains: `the bud-count and bud-spacing readouts, together with shape_energy_3d K_bend (currently 0.0, see premise 1)`
- check: `over a long run, bud count should fall and nearest-neighbour bud distance should shrink. A flat bud count needs a named mechanism keeping the buds apart.`
- if violated: "Bud count is flat across the run. Is fusion suppressed deliberately, and by what — the morphogen pattern pinning the bud sites, or a bending term set to zero so buds cannot feel each other? Name the mechanism; if it is the zero bending term, the bud count is an artefact."
- source: Noguchi & Elgeti 2024, §I and §V: "Previously, Okuda et al. [26] simulated the formation of multiple cylindrical buds from growing spots in a spherical tissue; however, they did not observe bud fusion. Here, we demonstrate that buds fuse through attractive interaction generated by tissue bending"; "Bud fusion is caused by the membrane-mediated attraction, which reduces the bending energy of the adhered tissue."
- scope: growing epithelial sheet or vesicle carrying multiple protrusions, 3D. Contested against the reproduction target: Okuda et al. 2018 report non-fusing buds, and Noguchi & Elgeti attribute that to a harmonic constraint on vertical cell position in the vertex-model simulations.

**67. A tissue tube held only by surface tension is unstable once it is longer than its own circumference: it necks and pinches into droplets. A long-lived slender tube must therefore be held by something else — solid-side (elastic) tissue, lumen pressure, or a sustained source at the tip.**  `[typical]`

- constrains: `the tube aspect-ratio metric (the campaign's "aspect 7.5 -> 3.2") jointly with shape_energy_3d p0 (fluid vs solid) and extrude K_extrude (0.0–14.0, default 4.0)`
- check: `if aspect ratio L/R > 2*pi (= 6.28) and p0 > 3.81 (fluid side) and no lumen-pressure term exists, the tube should be pinching. If it is not, the stabiliser must be named.`
- if violated: "A fluid-side tissue is holding a tube past L/R = 2*pi without necking. What resists the Plateau-Rayleigh instability here — a solid-side p0, a lumen pressure operator, or the external K_extrude push? If it is K_extrude, then the tube is held open by the forcing term rather than by the tissue, and that is the answer to the campaign's central question, not a side note."
- source: Noguchi & Elgeti 2024, §V: "vesicle division often occurs in long tubular buds via tube pinch-off (like droplet formation by Plateau-Rayleigh instability [66,67])"; "Tubular bud elongation is arrested at a certain height."
- scope: epithelial tube / bud on a growing sheet or vesicle, fluid regime, no lumen-pressure operator present

**68. A diffusible signal in tissue has a finite reach, set by its own destruction: range = sqrt(D / decay). With no decay there is no gradient, only accumulation, and the apparent pattern scale is set by the domain size and elapsed time rather than by the chemistry.**  `[textbook]`

- constrains: `cell_react mu_h (inhibitor decay, 0.2–2.0) and the hard-coded mu_a = 1.0 in the Gierer-Meinhardt emit at discovery/translate.py:141, against cell_diffuse d_a and d_h`
- check: `decay rate > 0 for every species; and the implied ranges sqrt(d_a/mu_a) and sqrt(d_h/mu_h) must both be finite, both smaller than the vesicle circumference, and must bracket the intended bud spacing. Report both in units of cell diameters.`
- if violated: "One of the species has no decay, so its reach is the whole body and the pattern wavelength is being set by the mesh, not by the kinetics. Is that species meant to be conserved rather than degraded? Declare it; otherwise give it a decay rate above zero and report the resulting range in cell diameters."
- source: Ziepke, Maryshev, Aranson & Frey 2022 (Multiscale_active_matter.pdf), p.4: "Since the signaling field decays exponentially (with diffusion length Lc ~ sqrt(Dc/alpha))"; the -alpha*c destruction term is explicit in their Eqs. (3) and (8).
- scope: any diffusible morphogen in a multicellular medium. NOT applicable to strictly juxtacrine (contact-only) signalling, which has no diffusion length.

**69. A two-layer, layer-wise fate pattern cannot be produced by neighbour coupling that treats every neighbour alike: it requires each cell to couple at least as strongly across the layer boundary as within its own layer.**  `[typical]`

- constrains: `cell_diffuse implementation — `graph_laplacian` couples every neighbour equally and `interface_weighted` by shared area; both are symmetric and carry no apical-basal axis`
- check: `the laminar mode's eigenvalue a + b − 1 must be strictly negative, i.e. n_cross * w_cross >= n_within * w_within per cell. With equal weights and typical packing (~4–6 in-layer neighbours vs ~1–2 cross-layer) this condition is NOT met.`
- if violated: "The coupling is isotropic, so there is no apical-basal axis for a layer-wise pattern to align with; the accessible pattern is salt-and-pepper or in-plane, not laminar. Is an unpolarised specimen intended? If a bilayer is the target, supply the polarity as an explicit cross-layer weight and state its value."
- source: Moore, Dale & Woolley 2023, §3.2: reduced adjacency Eq. (3.5) with a_i, b_i built from n1*w1 (intralayer) and n2*w2 (interlayer) in Eq. (3.6); Lemma 3.3 gives the polarity-driven eigenvalue Lambda2 = diag(a_i + b_i − 1) with eigenvector sign structure [−,+] (the laminar mode); Theorem 3.1's convergence proof assumes n1_Li*w1 <= n2_Li*w2 (p.16, "2a_k − 1 <= 0 and 2b_k − 1 <= 0").
- scope: bilayer / laminar-patterned epithelium (glandular duct geometry); fate patterning only, no mechanics. The cited condition is SUFFICIENT for convergence, not proven necessary — so failing it means "not guaranteed", not "impossible".

**70. The ducts of glandular tissue — mammary, salivary, sweat — are bilayers of two distinct cell types, luminal cells inside and myoepithelial cells outside. They are not one cell type folded into a tube.**  `[textbook]`

- constrains: `seed_mesh_3d (a single cell population, no fate field anywhere in the vocabulary) and any target morphology labelled a gland bud, duct, or branch`
- check: `if the target is a duct or gland bud, the model must carry at least 2 cell fates; otherwise the write-up must state that only one of the two layers is being modelled.`
- if violated: "A duct is being claimed from a single-fate epithelium. Is this the outer (myoepithelial) layer only? Declare that scope explicitly — otherwise the specimen is missing the layer that makes a duct a duct, and the morphology is a tube, not a gland."
- source: Moore, Dale & Woolley 2023, §1 and Fig. 1A: ductal tissues are "[p]rimarily comprised of just two cell types" and form "bilayer ducts of layer-wise contrasting epithelial cells with the outer and inner layers of myoepithelial and luminal cells, respectively"; markers p63 (myoepithelial) and Notch (luminal).
- scope: glandular duct / gland bud (mammary, salivary, sweat) specifically — NOT a generic epithelial vesicle, where a single fate is fine

_miner notes: CEDRIC'S QUESTION — "in a normal epithelium a mechanism (surface tension? internal pressure?) should counteract the apparent stretching" — answered from these papers. There are FOUR mechanisms, and they are separable:

1. Surface tension, yes, and it is the dominant one. Brückner & Hannezo 2025 (pp.6-7) state that forces in epithelia are localised at the cell surface — cortex and cell-cell/cell-EC_
