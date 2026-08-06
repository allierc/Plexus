# Causal lever-map

_53 runs · coverage **68%** (42/62 cells)_

The campaign's product. Specific questions are queries against this table.

## Coverage

| block | covered | total | |
|---|---|---|---|
| solo | 10 | 13 | 77% |
| pair | 29 | 45 | 64% |
| routing | 3 | 4 | 75% |

## Solo effects — what each operator does ALONE

| operator | n(with) | n(without) | Δscore | verdict | phenotypes seen |
|---|---|---|---|---|---|
| `shape_to_chem` | 9 | 44 | 0.523 | raises | sphere×4, undulation×3, degenerate×2 |
| `vesicle_growth` | 6 | 47 | 0.485 | raises | bud×3, exploded×1, sphere×1, undulation×1 |
| `cell_react` | 12 | 41 | 0.444 | raises | sphere×6, degenerate×3, undulation×3 |
| `morphogen_growth_3d` | 40 | 13 | 0.384 | raises | bud×21, sphere×11, undulation×5, degenerate×3 |
| `cell_adjacency` | 27 | 26 | 0.2 | neutral | bud×13, sphere×8, degenerate×3, undulation×3 |
| `extrude` | 12 | 41 | -0.034 | neutral | bud×12 |
| `reconnect_t1_3d` | 36 | 17 | -0.101 | neutral | bud×17, sphere×13, undulation×4, exploded×1, degenerate×1 |
| `cell_geometry_3d` | 46 | 7 | -0.208 | neutral | bud×25, sphere×15, degenerate×3, undulation×3 |
| `divide_3d` | 17 | 36 | -0.267 | lowers | bud×9, sphere×8 |
| `seed_cell_rd` | 38 | 15 | -0.437 | lowers | bud×25, sphere×10, undulation×3 |
| `cell_diffuse` | 1 | 52 | — | insufficient | — |
| `seed_mesh_3d` | 53 | 0 | — | insufficient | — |
| `shape_energy_3d` | 53 | 0 | — | insufficient | — |

## Interactions — where the joint effect is NOT the sum

_The expensive half of the map: what cannot be read off the code._

| pair | n | observed | additive prediction | interaction | verdict |
|---|---|---|---|---|---|
| `seed_cell_rd+divide_3d` | 16 | 1.935 | 1.387 | **+0.548** | SYNERGY |
| `cell_react+shape_to_chem` | 9 | 2.525 | 3.058 | **-0.532** | ANTAGONISM |
| `cell_geometry_3d+seed_cell_rd` | 36 | 1.963 | 1.446 | **+0.518** | SYNERGY |

## Phenotypes observed

bud×25, sphere×18, undulation×6, degenerate×3, exploded×1

