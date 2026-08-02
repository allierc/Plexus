# Causal lever-map

_25 runs · coverage **53%** (20/38 cells)_

The campaign's product. Specific questions are queries against this table.

## Coverage

| block | covered | total | |
|---|---|---|---|
| solo | 7 | 13 | 54% |
| pair | 10 | 21 | 48% |
| routing | 3 | 4 | 75% |

## Solo effects — what each operator does ALONE

| operator | n(with) | n(without) | Δscore | verdict | phenotypes seen |
|---|---|---|---|---|---|
| `vesicle_growth` | 2 | 23 | 1.745 | insufficient | exploded×1, sphere×1 |
| `shape_to_chem` | 9 | 16 | 0.424 | raises | sphere×4, undulation×3, degenerate×2 |
| `cell_react` | 12 | 13 | 0.346 | raises | sphere×6, degenerate×3, undulation×3 |
| `morphogen_growth_3d` | 19 | 6 | 0.274 | raises | bud×7, sphere×6, degenerate×3, undulation×3 |
| `cell_geometry_3d` | 20 | 5 | -0.14 | insufficient | bud×10, sphere×6, degenerate×3, undulation×1 |
| `cell_adjacency` | 22 | 3 | -0.235 | insufficient | bud×9, sphere×7, degenerate×3, undulation×3 |
| `reconnect_t1_3d` | 16 | 9 | -0.246 | neutral | sphere×7, bud×6, exploded×1, degenerate×1, undulation×1 |
| `extrude` | 9 | 16 | -0.377 | lowers | bud×9 |
| `divide_3d` | 6 | 19 | -0.432 | lowers | bud×5, sphere×1 |
| `cell_rd_seed` | 11 | 14 | -0.487 | lowers | bud×10, sphere×1 |
| `cell_diffuse` | 1 | 24 | — | insufficient | — |
| `seed_mesh_3d` | 25 | 0 | — | insufficient | — |
| `shape_energy_3d` | 25 | 0 | — | insufficient | — |

## Interactions — where the joint effect is NOT the sum

_The expensive half of the map: what cannot be read off the code._

| pair | n | observed | additive prediction | interaction | verdict |
|---|---|---|---|---|---|
| `cell_rd_seed+extrude` | 9 | 2.013 | 1.39 | **+0.622** | SYNERGY |
| `cell_rd_seed+divide_3d` | 6 | 1.926 | 1.335 | **+0.591** | SYNERGY |

## Phenotypes observed

bud×10, sphere×8, degenerate×3, undulation×3, exploded×1

