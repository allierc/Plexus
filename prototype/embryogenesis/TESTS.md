# Embryogenesis (active-matter x MPM) -- test log

Chronological log of every tuning run (successes AND failures). Media in `archive/<name>/` (blob_evolution.png, blob.mp4, fig_*_evolution.png).

| name | escaped | disc_growth | aniso | polar | s | key overrides |
|---|---|---|---|---|---|---|
| agent_mpm_disc_water_v3 | 0.0 | 0.0574 | 0.1394 | 0.1403 | 24.5 | n_grid=64 p2g.drag=0.4 mpm_grid_update.surface_tension=160 mpm_spin.omega=0.8 agent_to_mpm.agent_mass=2e-5 mpm_to_agent.confine=140 mpm_to_agent.k=0.8 agent.move_speed=0.5 polar_align.gamma=80 polar_align.noise=2.5 |
| agent_mpm_disc_elastic_v3 | 0.0025 | 0.0494 | 0.0841 | 0.4583 | 77.4 | n_grid=64 p2g.drag=0.3 mpm_grid_update.surface_tension=120 mpm_spin.omega=0.8 agent_to_mpm.agent_mass=4e-5 mpm_to_agent.confine=140 mpm_to_agent.k=0.8 agent.move_speed=0.5 polar_align.gamma=80 polar_align.noise=2.5 |
| agent_mpm_disc_water_st300 | 0.0 | 0.0514 | 0.1235 | 0.1379 | 22.4 | mpm_grid_update.surface_tension=300 |
| agent_mpm_disc_water_st460 | 0.0 | 0.044 | 0.1201 | 0.1191 | 22.4 | mpm_grid_update.surface_tension=460 |
| agent_mpm_disc_4types_smoke | 0.0 | 0.0073 | 0.023 | 0.0452 | 11.7 |  |
| agent_mpm_disc_4types_show | 0.0395 | 0.3266 | 0.1748 | 0.0147 | 167.4 |  |
| agent_mpm_disc_water_long_nospin | 0.0725 | 0.326 | 0.4843 | 0.0815 | 101.8 | p2g.drag=0.6 mpm_grid_update.surface_tension=300 mpm_spin.omega=0.0 agent_to_mpm.agent_mass=1e-5 |
| agent_mpm_disc_water_long_drag | 0.0325 | 0.3262 | 0.2311 | 0.2636 | 105.7 | p2g.drag=0.9 mpm_grid_update.surface_tension=420 mpm_spin.omega=0.4 |
| agent_mpm_disc_elastic_long_soft | 0.0075 | 0.3263 | 0.3858 | 0.2224 | 100.2 | cell.youngs=25 p2g.drag=0.5 mpm_grid_update.surface_tension=200 |
| agent_mpm_disc_elastic_long | 0.48 | 0.289 | 0.1984 | 0.2208 | 113.2 |  |
| agent_mpm_disc_stable_t1 | 0.0 | 0.326 | 0.2138 | 0.1279 | 156.1 |  |
| agent_mpm_disc_stable_anchor80 | 0.122 | 0.2774 | 0.1801 | 0.0227 | 58.0 | mpm_anchor.k=80 mpm_spin.omega=0.4 |
| agent_mpm_disc_stable_bdry | 0.037 | 0.294 | 0.1506 | 0.0816 | 57.7 | mpm_anchor.mode=boundary mpm_anchor.ring=0.06 mpm_anchor.k=60 mpm_spin.omega=0.4 |
| agent_mpm_embryo_t1 | 0.9935 | -0.0 | 0.0091 | 0.0022 | 54.4 |  |
| agent_mpm_embryo_conf1.5 | 0.0 | 0.0798 | 0.128 | 0.0451 | 27.8 | mpm_to_agent.confine=1.5 |
| agent_mpm_embryo_conf4 | 0.003 | 0.1181 | 0.1853 | 0.0036 | 28.4 | mpm_to_agent.confine=4 |
| agent_mpm_embryo_conf10 | 0.202 | 0.1707 | 0.4806 | 0.0564 | 27.9 | mpm_to_agent.confine=10 |
| agent_mpm_embryo_long | 0.0385 | 0.2519 | 0.5544 | 0.0108 | 92.8 |  |
| agent_mpm_embryo_epi1 | 0.0 | 0.0428 | 0.122 | 0.1311 | 100.4 |  |
| agent_mpm_embryo_div_smoke | 0.0 | 0.0022 | 0.018 | 0.042 | 15.2 |  |
| agent_mpm_embryo_div_grow | 0.0 | 0.0734 | 0.1948 | 0.0871 | 120.2 |  |
| agent_mpm_embryo_div_repel40 | 0.0 | 0.1195 | 0.3279 | 0.0447 | 77.0 | repel.strength=40 repel.r0=0.02 |
| agent_mpm_embryo_div_repel90 | 0.0 | 0.1263 | 0.3237 | 0.053 | 78.3 | repel.strength=90 repel.r0=0.02 |
| agent_mpm_embryo_div_tile | 0.0 | 0.0909 | 0.2448 | 0.1175 | 104.8 | repel.strength=30 repel.r0=0.035 polar_align.gamma=8 |
| agent_mpm_embryo_div_nocouple | 0.0 | 0.0052 | 0.0268 | 0.0323 | 62.8 | repel.strength=30 repel.r0=0.035 agent_to_mpm.agent_mass=1e-6 mpm_to_agent.k=0.0 polar_align.gamma=8 |
| agent_mpm_embryo_div_antimips | 0.0 | 0.0001 | 0.0026 | 0.004 | 76.7 | polar_align.gamma=0 polar_align.noise=22 repel.strength=45 repel.r0=0.03 flow_align.gain=60 |
| agent_mpm_blastula_t1 | 0.0 | 0.0277 | 0.1032 | 0.0237 | 82.7 |  |
| agent_mpm_blastula_reponly | 0.0 | 0.0243 | 0.1005 | 0.0084 | 98.3 | attraction_repulsion.sigma=0.02 |
| agent_mpm_blastula_pure | 0.0 | 0.0803 | 0.235 | 0.0179 | 113.1 | agent.p=0,1,12,1.6 agent.div_rate=0.7 |
| agent_mpm_blastula_diagB | 0.0 | 0.0008 | 0.0101 | 0.0075 | 88.9 | agent.n=3000 agent.p=0,1,15,1.6 cell_divide.rate=0 flow_align.gain=0 mpm_to_agent.k=0 agent_to_mpm.agent_mass=1e-6 |
| agent_mpm_blastula_diagA | 0.0 | -0.0 | 0.0045 | 0.0164 | 97.0 | agent.p=0,1,15,1.6 agent.div_rate=0.8 flow_align.gain=0 mpm_to_agent.k=0 agent_to_mpm.agent_mass=1e-6 |
| agent_mpm_blastula_iso_repel_only | 0.0 | 0.0 | 0.0009 | 0.0163 | 63.4 | agent.n=2500 agent.p=0,1,15,1.6 agent.div_rate=0 agent.move_speed=0 flow_align.gain=0 mpm_to_agent.k=0 agent_to_mpm.agent_mass=1e-9 mpm_spin.omega=0 |
| agent_mpm_blastula_iso_repel_glide | 0.0 | 0.0 | 0.0009 | 0.011 | 64.2 | agent.n=2500 agent.p=0,1,15,1.6 agent.div_rate=0 agent.move_speed=0.3 flow_align.gain=0 mpm_to_agent.k=0 agent_to_mpm.agent_mass=1e-9 mpm_spin.omega=0 |
| agent_mpm_blastula_iso_gentle_static | 0.0 | 0.0 | 0.0009 | 0.0084 | 63.5 | agent.n=2500 agent.p=1.0,1.0,1.3,1.8 attraction_repulsion.sigma=0.03 agent.div_rate=0 agent.move_speed=0 flow_align.gain=0 mpm_to_agent.k=0 agent_to_mpm.agent_mass=1e-9 mpm_spin.omega=0 radius_graph.radius=0.08 |
| agent_mpm_blastula_iso_gentle_glide | 0.0 | 0.0 | 0.0009 | 0.0109 | 64.7 | agent.n=2500 agent.p=1.0,1.0,1.3,1.8 attraction_repulsion.sigma=0.03 agent.div_rate=0 agent.move_speed=0.15 flow_align.gain=0 mpm_to_agent.k=0 agent_to_mpm.agent_mass=1e-9 mpm_spin.omega=0 radius_graph.radius=0.08 |
| agent_mpm_blastula_iso_matched_glide | 0.0 | 0.0 | 0.0009 | 0.0067 | 63.7 | agent.n=2500 agent.p=1.0,1.0,1.3,1.9 attraction_repulsion.sigma=0.013 agent.div_rate=0 agent.move_speed=0.1 flow_align.gain=0 mpm_to_agent.k=0 agent_to_mpm.agent_mass=1e-9 mpm_spin.omega=0 radius_graph.radius=0.045 |
| agent_mpm_blastula_iso_matched_static | 0.0 | 0.0 | 0.0009 | 0.0063 | 64.0 | agent.n=2500 agent.p=1.0,1.0,1.3,1.9 attraction_repulsion.sigma=0.013 agent.div_rate=0 agent.move_speed=0 flow_align.gain=0 mpm_to_agent.k=0 agent_to_mpm.agent_mass=1e-9 mpm_spin.omega=0 radius_graph.radius=0.045 |
| agent_mpm_blastula_4types_v1 | 0.0 | 0.0985 | 0.1846 | 0.0809 | 112.4 |  |
| agent_mpm_blastula_tiled_r016 | 0.0 | 0.0004 | 0.0165 | 0.0148 | 111.1 |  |
| agent_mpm_blastula_tiled_r022 | 0.0 | 0.0014 | 0.0183 | 0.0082 | 111.5 | repel.r0=0.022 repel.strength=8 |
| agent_mpm_blastula_tiled_matched | 0.0 | 0.0004 | 0.0123 | 0.0071 | 111.4 |  |
| agent_mpm_blastula_tiled_sun_div | 0.0 | 0.0002 | 0.0095 | 0.0364 | 95.0 |  |
| agent_mpm_blastula_tiled_sun_nodiv | 0.0 | 0.0002 | 0.0095 | 0.0364 | 94.9 | agent.div_rate=0 |
| agent_mpm_blastula_tiled_sun_colour | 0.0 | -0.0003 | 0.0083 | 0.0442 | 93.8 | agent.div_rate=0 |
| agent_mpm_blastula_tiled_clean_repel | 0.0 | 0.0 | 0.0007 | 0.0051 | 64.1 | agent.div_rate=0 agent.move_speed=0 flow_align.gain=0 mpm_to_agent.k=0 mpm_to_agent.confine=0 agent_to_mpm.agent_mass=1e-9 mpm_spin.omega=0 |
| agent_mpm_blastula_tiled_clean_couple | 0.0 | 0.0 | 0.0007 | 0.008 | 65.4 | agent.div_rate=0 agent.move_speed=0 flow_align.gain=0 mpm_spin.omega=0 |
| agent_mpm_blastula_tiled_n350_weakcouple | 0.0 | -0.0002 | 0.0009 | 0.0431 | 107.4 | agent.n=350 agent.div_rate=0 repel.r0=0.024 repel.strength=4 agent.move_speed=0.05 flow_align.gain=40 agent_to_mpm.agent_mass=8e-7 mpm_to_agent.k=0.5 |
| agent_mpm_blastula_tiled_n350_nocouple | 0.0 | -0.0002 | 0.0008 | 0.0146 | 107.7 | agent.n=350 agent.div_rate=0 repel.r0=0.024 repel.strength=4 agent.move_speed=0.05 flow_align.gain=0 agent_to_mpm.agent_mass=1e-9 mpm_to_agent.k=0 |
| agent_mpm_blastula_tiled_n350_static | 0.0 | 0.0 | 0.0007 | 0.0016 | 72.4 | agent.n=350 agent.div_rate=0 repel.r0=0.024 repel.strength=4 agent.move_speed=0 flow_align.gain=0 agent_to_mpm.agent_mass=1e-9 mpm_to_agent.k=0 mpm_to_agent.confine=0 mpm_spin.omega=0 |
| agent_mpm_blastula_tiled_strong20 | 0.0 | -0.0001 | 0.0016 | 0.0329 | 107.8 | agent.n=350 agent.div_rate=0 repel.r0=0.03 repel.strength=20 agent.move_speed=0.05 flow_align.gain=40 agent_to_mpm.agent_mass=5e-6 mpm_to_agent.k=0.6 |
| agent_mpm_blastula_tiled_strong50 | 0.0 | -0.0002 | 0.0019 | 0.0086 | 108.9 | agent.n=350 agent.div_rate=0 repel.r0=0.032 repel.strength=50 agent.move_speed=0.05 flow_align.gain=40 agent_to_mpm.agent_mass=5e-6 mpm_to_agent.k=0.6 |
| agent_mpm_blastula_tiled_tinymass | 0.0 | -0.0002 | 0.0008 | 0.0035 | 108.5 | agent.n=350 agent.div_rate=0 repel.r0=0.03 repel.strength=20 agent.move_speed=0.05 flow_align.gain=40 agent_to_mpm.agent_mass=1e-7 mpm_to_agent.k=0.6 |
| agent_mpm_blastula_tiled_nopush | 0.0 | -0.0002 | 0.0008 | 0.0225 | 109.2 | agent.n=350 agent.div_rate=0 repel.r0=0.03 repel.strength=20 agent.move_speed=0.05 flow_align.gain=40 agent_to_mpm.agent_mass=0 mpm_to_agent.k=0.6 |
