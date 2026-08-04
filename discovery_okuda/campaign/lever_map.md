# Causal lever-map

_41 runs · coverage **68%** (23/34 cells)_

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
| `reconnect_t1_3d` | 33 | 8 | 0.072 | neutral | unreadable×33 |
| `cell_adjacency` | 8 | 33 | -0.007 | neutral | unreadable×8 |
| `cell_rd_seed` | 25 | 16 | -0.019 | neutral | unreadable×25 |
| `divide_3d` | 18 | 23 | -0.039 | neutral | unreadable×18 |
| `extrude` | 10 | 31 | -0.039 | neutral | unreadable×10 |
| `cell_geometry_3d` | 5 | 36 | -0.076 | insufficient | unreadable×5 |
| `vesicle_growth` | 7 | 34 | -0.082 | neutral | unreadable×7 |
| `morphogen_growth_3d` | 9 | 32 | -0.0 | neutral | unreadable×9 |
| `seed_mesh_3d` | 41 | 0 | — | insufficient | — |
| `shape_energy_3d` | 41 | 0 | — | insufficient | — |

## Interactions — where the joint effect is NOT the sum

_The expensive half of the map: what cannot be read off the code._

| pair | n | observed | additive prediction | interaction | verdict |
|---|---|---|---|---|---|
| _(none established yet)_ | | | | | |

## Phenotypes observed

unreadable×41

