# Grounding — round r022 vs Okuda 2018

**Same result as r021: bar cleared, wrong mechanism.** Six runs clear the tube
bar — r022_03 `protr_peak` **2.296**, _07 2.225, _06 2.031, _02 1.937, ctrl/_05
1.83 (all >1.3) — and the eye confirms genuine tapered red-tipped fingers ("star",
"starfish", "branching"), not bulges. Morphologically this is Okuda's **branched**
panel. BUT `mech_p_ratio` is **2.51–3.75** (≈3 pushed, ≈1 grown): r022_03 3.746,
_07 3.42. The star is extruded, not grown — the threshold is met, the actual
question (did the tissue MAKE the finger) is not. This is a REPLICATE of r021, not
new ground.

**The gap.** A protrusion at `mech_p_ratio` near **1** — nothing this round has it;
the sharpest arms are the most forced. Scale off: `n_spots` 3–6 vs Fig. 5a's ~10;
`spot_spacing_cells` ~40 = Fig. 5b budding, not 10-cell thin tubes. Grown-signature
runs stay spheres (r022_13 `protr_peak` 1.235/ratio 1.709; r022_15 1.108/0.0).

**Untested by the paper.** Remove `extrude` and reproduce `protr_peak`>1.3 from
growth+mechanics alone — the one experiment answering the open problem. r021's
grounding already named it and it was NOT run; r022 is the third round the finger
has been pushed. **Decision needed (Cedric/Proposer): stop re-forcing the star.**

**Losses:** r022_14 NaN blow-up (eye, post-1710f unreliable); r022_08–12 empty.

## Corpus candidates (operators carried this round)
- **A fluid-side protrusion is not growth** — a forced star at `mech_p_ratio` 3.75
  is a fluidisation/extrude bud; report ΔV_tot over the protrusion window, r022_03/_07
  violate. *(VertAX, Pasqui 2604.06896 Fig. 3g; premise 10/14.)*
- **Diffusion does not scale with the body** — cells ~doubled yet `n_spots` held 3–6;
  at fixed D spots should multiply ~s². *(Ledesma-Duran 2308.12196 §II Eq. 4; premise 7.)*
