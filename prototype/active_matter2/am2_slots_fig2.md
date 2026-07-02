# Fig.2 batch 10 [FINAL] -- batch-9 read: ring box is TIGHT (Dc~4.0 edge <0.2; w~1.4 EXACTLY -- w1.5 collapses to
# homogeneous polar); the ring FIELD-multiplier is rho0 DOWN, not omega up -- ring_rho105 (rho0 1.05) = 9 arced
# sites = best ring-field candidate, arcs still OPEN. v0-sharpening SATURATES at ~2.0 (ctr 0.93 flat 2.0->2.5,
# stable). Boundaries: stream->condense (v0 1.8) between w 0.6/0.75; sigband->condense (v0 2.0) between w 0.5/0.8.
# THIS BATCH (final): CLOSE the ring field -- hollow the 9 rho0-1.05 sites into annuli (Dc both ways, faster c-decay
# alpha, gentle v0), + FINE boundary probes to pin all four edges to <0.1 in the variable.
# One variable changed per slot from its NAMED parent. No panel.png = FAILED.
#
# lineage / target state:
#   ring_rho105(v0 0.6,w 1.4,chi 0,Dc 4.0,rho0 1.05 = 9 arced sites) --Dc-----> ring_Dc36  (sharper fronts CLOSE arcs?)
#   ring_rho105                                                      --Dc-----> ring_Dc44  (broader fronts CLOSE arcs?)
#   ring_rho105                                                      --alpha--> ring_a07   (faster c-decay -> tight wells -> closed rings?)
#   ring_rho105                                                      --v0-----> ring_v065  (gentle hollow of the 9 cores, no fragment)
#   ring_Dc4(v0 0.6,w 1.4,chi 0,Dc 4.0,rho0 1.10)                    --Dc-----> ring_Dc41  (pin single-aster ring optimum 4.0<->4.2)
#   ring_Dc4                                                         --w------> ring_w145  (pin ring->homog-polar collapse 1.4<->1.5)
#   stream_v018(v0 1.8,w 0.6,rho0 1.05,Drho .15,Dp .2)               --w------> stream_w068(pin stream->condense onset 0.6<->0.75)
#   sigband_v20(v0 2.0,w 0.5,rho0 1.10,Drho .15)                     --w------> sigband_w065(pin sigband->condense onset 0.5<->0.8)

ring_Dc36    : --kind hydro --v0 0.6 --omega 1.4 --chi 0.0 --Dc 3.6 --rho0 1.05 --nsteps 40000 --mode snapshot
ring_Dc44    : --kind hydro --v0 0.6 --omega 1.4 --chi 0.0 --Dc 4.4 --rho0 1.05 --nsteps 40000 --mode snapshot
ring_a07     : --kind hydro --v0 0.6 --omega 1.4 --chi 0.0 --Dc 4.0 --rho0 1.05 --alpha 0.7 --nsteps 40000 --mode snapshot
ring_v065    : --kind hydro --v0 0.65 --omega 1.4 --chi 0.0 --Dc 4.0 --rho0 1.05 --nsteps 40000 --mode snapshot
ring_Dc41    : --kind hydro --v0 0.6 --omega 1.4 --chi 0.0 --Dc 4.1 --rho0 1.10 --nsteps 40000 --mode snapshot
ring_w145    : --kind hydro --v0 0.6 --omega 1.45 --chi 0.0 --Dc 4.0 --rho0 1.10 --nsteps 40000 --mode snapshot
stream_w068  : --kind hydro --v0 1.8 --omega 0.68 --rho0 1.05 --Drho 0.15 --Dp 0.2 --nsteps 45000 --mode snapshot
sigband_w065 : --kind hydro --v0 2.0 --omega 0.65 --rho0 1.10 --Drho 0.15 --nsteps 45000 --mode snapshot
