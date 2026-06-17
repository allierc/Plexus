# Ant colony test suite (Lague Ant-Simulation over Plexus)

All variants share one registry (`motility`+`colony`+`trail`x2+`secrete`x2+`mpm`, two pheromone fields); only spec parameters differ.

> **Reading the numbers.** These are *single-seed* runs and the bottleneck is the stochastic first discovery of food; once recruitment ignites it is vigorous (`ant_colony_s1`=213, `ant_many_food`=191, `ant_food1`=126), but a given cell's `delivered` is dominated by discovery luck, not the swept parameter (compare the three baseline seeds: 17 / 213 / 97). So treat `delivered>0` as "recruitment works in this regime", and the *gifs* (the trail morphology) as the real per-parameter signal — not the scalar count. Hard layouts (single far food behind a wall/maze) often fail to ignite in one seed.

name | frames | delivered | trail_coverage | intent | notes
---|---|---|---|---|---
ant_colony | 251 | 17 | 0.010 | WORKS | Baseline: central nest, three food discs; dual-pheromone recruitment.
ant_colony_s1 | 226 | 213 | 0.065 | WORKS | Baseline, seed 1: a different realization of the same dynamics.
ant_colony_s2 | 226 | 97 | 0.000 | WORKS | Baseline, seed 2: a different realization of the same dynamics.
ant_decay_008 | 226 | 6 | 0.005 | WORKS | Decay 0.008: trails persist -> thick, long-lived network.
ant_decay_015 | 226 | 17 | 0.000 | WORKS | Decay 0.015: moderately persistent trails.
ant_decay_035 | 226 | 19 | 0.000 | WORKS | Decay 0.035: faster turnover -> thinner trails.
ant_decay_060 | 226 | 0 | 0.000 | no delivery (regime too hard) | Decay 0.060: volatile trails; constant re-recruitment.
ant_diff_005 | 226 | 19 | 0.030 | WORKS | Diffusion 0.005: crisp, narrow trails.
ant_diff_010 | 226 | 38 | 0.004 | WORKS | Diffusion 0.010: narrow trails.
ant_diff_040 | 226 | 40 | 0.000 | WORKS | Diffusion 0.040: wide, blurry trails.
ant_diff_080 | 226 | 17 | 0.000 | WORKS | Diffusion 0.080: very diffuse pheromone clouds.
ant_turn_020 | 226 | 28 | 0.013 | WORKS | Turn 0.20: weak steering -> loose, wandering trails.
ant_turn_035 | 226 | 28 | 0.022 | WORKS | Turn 0.35: gentle trail-following.
ant_turn_070 | 226 | 6 | 0.000 | WORKS | Turn 0.70: strong steering -> tight, committed trails.
ant_turn_090 | 226 | 18 | 0.000 | WORKS | Turn 0.90: very sharp turns -> nervous, kinked trails.
ant_sensor_d03 | 226 | 7 | 0.000 | WORKS | Sensor dist 0.03: short reach -> ants hug the trail.
ant_sensor_d09 | 226 | 76 | 0.020 | WORKS | Sensor dist 0.09: long reach -> ants lock on from afar.
ant_sensor_a30 | 226 | 13 | 0.002 | WORKS | Sensor cone 0.3 rad: narrow -> follows the trail centre.
ant_sensor_a12 | 226 | 45 | 0.034 | WORKS | Sensor cone 1.2 rad: wide -> exploratory, fuzzy trails.
ant_n080 | 226 | 35 | 0.016 | WORKS | Small colony (80 ants): slower discovery, sparser trails.
ant_n200 | 226 | 8 | 0.000 | WORKS | Large colony (200 ants): faster discovery, denser network.
ant_n260 | 226 | 1 | 0.000 | WORKS | Huge colony (260 ants): rapid recruitment, thick highways.
ant_runout_150 | 226 | 0 | 0.000 | no delivery (regime too hard) | Runout 150: short trails (fade before bridging nest<->food).
ant_runout_300 | 226 | 0 | 0.000 | no delivery (regime too hard) | Runout 300: trails bridge most of the gap.
ant_runout_800 | 226 | 0 | 0.001 | no delivery (regime too hard) | Runout 800: long, slowly-fading recruitment gradients.
ant_rate_2 | 226 | 25 | 0.003 | WORKS | Deposit rate 2: faint trails, weaker recruitment.
ant_rate_8 | 226 | 19 | 0.020 | WORKS | Deposit rate 8: strong trails, fast lock-in.
ant_food1 | 276 | 126 | 0.056 | WORKS | One food disc: a single dominant trail.
ant_food2 | 226 | 28 | 0.000 | WORKS | Two food discs: two competing trails.
ant_many_food | 226 | 191 | 0.071 | WORKS | Five scattered sources: several trails coexist.
ant_perc_04 | 226 | 25 | 0.004 | WORKS | Perception 0.04: tiny homing radius -> more wandering near goals.
ant_perc_14 | 226 | 18 | 0.000 | WORKS | Perception 0.14: large homing radius -> snappy pickups/dropoffs.
ant_highway | 301 | 15 | 0.000 | WORKS | Highway: side nest -> one food across the arena; one strong trail.
ant_wall | 326 | 0 | 0.000 | no delivery (regime too hard) | Wall: a barrier with one gap between nest and food; trail routes through it.
ant_maze | 351 | 0 | 0.000 | no delivery (regime too hard) | Maze: a barrier with two gaps; the colony commits to a route.
ant_depleting | 301 | 29 | 0.000 | WORKS | Depleting food: each source holds 80 units; trails shift as food runs out.
ant_no_trail | 226 | 21 | 0.004 | ok (control: low delivery expected) | Ablation: trail-following off (turn=0) -> no recruitment; chance deliveries only.
