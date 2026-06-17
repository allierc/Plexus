# Overnight water/material test suite

name | frames | particles | flatness | wall_stuck% | notes
---|---|---|---|---|---
ob_dam_break | 181 | 9000 | 0.029 | 0.0 | Dam break: a water column is released and collapses across the floor.
ob_dam_viscous | 181 | 9000 | 0.012 | 0.0 | Viscous dam break (honey): same column, high drag -> slow ooze vs water.
ob_funnel | 201 | 7000 | 0.038 | 0.0 | Funnel: water poured into a converging funnel collects and necks down.
ob_leak_tank | 221 | 8000 | 0.007 | 0.0 | Leaky tank: water in a tank drains out a hole in the side wall.
ob_pillars | 181 | 7000 | 0.109 | 0.0 | Pillars: water dropped over obstacle pillars flows around them.
ob_wedge | 221 | 7000 | 0.074 | 0.0 | Wedge: a sloped floor; water flows to the deep end and levels flat.
ob_zigzag | 221 | 7000 | 0.114 | 0.0 | Zigzag: water cascades down through alternating baffles.
ph_coalesce | 101 | 7000 | 0.028 | 0.0 | Coalescence: two adjacent water blobs in zero-g merge into one circle (CSF).
ph_slosh | 201 | 9000 | 0.119 | 0.0 | Slosh: a pool under tilted gravity sloshes and the surface tilts.
ph_hydrostatic | 141 | 10000 | 0.004 | 0.0 | Hydrostatic: a pool at rest should stay flat and still (stability check).
ph_snow_pile | 161 | 6000 | 0.402 | 0.0 | Snow pile: snow dropped from a point forms an angle-of-repose heap.
ph_snow_funnel | 181 | 6000 | 0.129 | 0.0 | Snow funnel: snow piles in a funnel (vs water which would drain).
ph_balls_bowl | 161 | 7200 | 0.011 | 0.0 | Balls in a bowl: stiff elastic balls dropped into a basin settle/pack.
ph_crown_splash | 181 | 14000 | 0.032 | 0.0 | Crown splash: an elastic ball plunges into a deep water pool.
