# Causal lever-map

_23 runs · coverage **65%** (15/23 cells)_

The campaign's product. Specific questions are queries against this table.

## Coverage

| block | covered | total | |
|---|---|---|---|
| solo | 5 | 10 | 50% |
| pair | 7 | 10 | 70% |
| routing | 3 | 3 | 100% |

## Solo effects — what each operator does ALONE

| operator | n(with) | n(without) | Δscore | verdict | phenotypes seen |
|---|---|---|---|---|---|
| `reconnect_t1_3d` | 17 | 6 | 0.043 | neutral | unreadable×17 |
| `cell_geometry_3d` | 2 | 21 | 0.034 | insufficient | unreadable×2 |
| `morphogen_growth_3d` | 9 | 14 | -0.005 | neutral | unreadable×9 |
| `cell_adjacency` | 5 | 18 | -0.039 | insufficient | unreadable×5 |
| `cell_rd_seed` | 10 | 13 | -0.057 | neutral | unreadable×10 |
| `extrude` | 10 | 13 | -0.057 | neutral | unreadable×10 |
| `divide_3d` | 11 | 12 | -0.078 | neutral | unreadable×11 |
| `vesicle_growth` | 5 | 18 | -0.139 | insufficient | unreadable×5 |
| `seed_mesh_3d` | 23 | 0 | — | insufficient | — |
| `shape_energy_3d` | 23 | 0 | — | insufficient | — |

## Interactions — where the joint effect is NOT the sum

_The expensive half of the map: what cannot be read off the code._

| pair | n | observed | additive prediction | interaction | verdict |
|---|---|---|---|---|---|
| _(none established yet)_ | | | | | |

## Phenotypes observed

unreadable×23

