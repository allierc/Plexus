# Causal lever-map

_44 runs · coverage **68%** (23/34 cells)_

The campaign's product. Specific questions are queries against this table.

## Coverage

| block | covered | total | |
|---|---|---|---|
| solo | 7 | 10 | 70% |
| pair | 13 | 21 | 62% |
| routing | 3 | 3 | 100% |

## Solo effects — what each operator does ALONE

| operator | n(with) | n(without) | Δscore | verdict | phenotypes seen |
|---|---|---|---|---|---|
| `reconnect_t1_3d` | 36 | 8 | 0.091 | neutral | unreadable×36 |
| `cell_rd_seed` | 28 | 16 | 0.008 | neutral | unreadable×28 |
| `divide_3d` | 21 | 23 | -0.002 | neutral | unreadable×21 |
| `morphogen_growth_3d` | 9 | 35 | -0.021 | neutral | unreadable×9 |
| `cell_adjacency` | 8 | 36 | -0.027 | neutral | unreadable×8 |
| `extrude` | 10 | 34 | -0.059 | neutral | unreadable×10 |
| `cell_geometry_3d` | 5 | 39 | -0.094 | insufficient | unreadable×5 |
| `vesicle_growth` | 7 | 37 | -0.1 | neutral | unreadable×7 |
| `seed_mesh_3d` | 44 | 0 | — | insufficient | — |
| `shape_energy_3d` | 44 | 0 | — | insufficient | — |

## Interactions — where the joint effect is NOT the sum

_The expensive half of the map: what cannot be read off the code._

| pair | n | observed | additive prediction | interaction | verdict |
|---|---|---|---|---|---|
| _(none established yet)_ | | | | | |

## Phenotypes observed

unreadable×44

