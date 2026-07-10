# 10 strange attractors — Plexus `attractor_flow` (dx/dt = f(x))

| attractor | spread_growth | extent (x,y,z) | occupancy_32 |
|--|--|--|--|
| halvorsen | 21.3× | [21.63, 21.64, 21.64] | 0.1425 |
| lorenz | 33.2× | [39.96, 52.79, 47.2] | 0.0588 |
| aizawa | 4.6× | [3.09, 3.1, 2.53] | 0.1288 |
| sprott_b | 5.6× | [14.57, 7.99, 15.17] | 0.0592 |
| thomas | 7.4× | [4.95, 4.95, 4.95] | 0.0441 |
| rossler | 18.2× | [19.12, 19.13, 26.76] | 0.0121 |
| dadras | 12.9× | [39.63, 22.81, 28.88] | 0.0424 |
| chen | 32.7× | [51.97, 59.84, 52.85] | 0.0853 |
| chua | 14.3× | [4.58, 0.83, 7.51] | 0.105 |
| rabinovich_fabrikant | 1787.2× | [81.49, 269.16, 1.99] | 0.007 |

_rabinovich_fabrikant leaks ~20% of its cloud to infinity (its chaotic attractor coexists with escaping orbits), so its extent/spread stats reflect the escapees; the movie frames the bounded urchin core (view_quantile)._
